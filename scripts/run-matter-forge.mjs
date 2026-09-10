import { createServer } from 'node:http';
import { readFile,realpath,mkdir,writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createHash } from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),serve=process.argv.includes('--serve');
if(process.argv.slice(2).some(x=>x!=='--serve'))throw Error('Only option: --serve');
const prefix='/experiments/matter-forge/',allowed=[prefix,'/node_modules/three-r186/'],routes=[],servedHashes={},sha=x=>createHash('sha256').update(x).digest('hex');
const server=createServer(async(req,res)=>{
 try{
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405).end();return;}
  let name=decodeURIComponent(new URL(req.url,'http://127.0.0.1').pathname);if(name==='/')name=prefix+'app.html';if(name==='/favicon.ico'){res.writeHead(204).end();return;}
  const match=allowed.find(a=>name.startsWith(a));if(!match||name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid route');
  const base=await realpath(path.join(root,match)),file=await realpath(path.join(root,name)),relative=path.relative(base,file);
  if(!relative||relative.startsWith('..')||path.isAbsolute(relative))throw Error('Outside source allowlist');
  const raw=await readFile(file),mime=name.endsWith('.html')?'text/html':name.endsWith('.png')?'image/png':name.endsWith('.json')?'application/json':'text/javascript';
  const digest=sha(raw);if(servedHashes[name]&&servedHashes[name]!==digest)throw Error('Served source changed');servedHashes[name]=digest;
  res.writeHead(200,{'Content-Type':mime,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(req.method==='HEAD'?undefined:raw);
 }catch(e){routes.push({url:req.url,error:e.message});res.writeHead(404).end();}
});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(serve?5201:0,'127.0.0.1',resolve);});
const url=`http://127.0.0.1:${server.address().port}/`;
if(serve)console.log(JSON.stringify({url,state:'Idle until Enter the foundry; runs bounded to12s wall/4s simulation'}));
else{
 const out=path.join(root,'results/development/matter-forge-ui',new Date().toISOString().replaceAll(':','-'));await mkdir(out,{recursive:true});
 const report={status:'failed',errors:[],routes,sourceHashes:{},sourceSnapshots:{},servedHashes};let browser,page;
 for(const name of ['app.html','app.mjs','gpu.mjs','cpu.mjs','casting-fixture.mjs','casting-fixture-v2.mjs','environment.mjs','voxelize.mjs','import-glb.mjs','liquid-surface.mjs']){const source=await readFile(path.join(root,prefix,name),'utf8');report.sourceHashes[prefix+name]=sha(source);report.sourceSnapshots[prefix+name]=source;}
 report.sourceSnapshots['/scripts/run-matter-forge.mjs']=await readFile(fileURLToPath(import.meta.url),'utf8');report.sourceHashes['/scripts/run-matter-forge.mjs']=sha(report.sourceSnapshots['/scripts/run-matter-forge.mjs']);
 const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),60000);
 try{
  browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(url);await page.click('#prepare');await page.waitForFunction(()=>window.matterForge?.evidence.prepared||window.matterForge?.evidence.errors.length,null,{timeout:30000});
  report.prepared=await page.evaluate(()=>window.matterForge.evidence);if(!report.prepared.prepared)throw Error(JSON.stringify(report.prepared.errors));
  const sourceFile=path.join(root,prefix,'assets/casting-blank.glb'),sourceBytes=await readFile(sourceFile);report.uploadArtifact={name:'casting-blank.glb',sha256:sha(sourceBytes),bytes:sourceBytes.length};
  await page.setInputFiles('#file',sourceFile);await page.waitForFunction(()=>window.matterForge.evidence.resetCount===2,null,{timeout:10000});
  report.imported=await page.evaluate(()=>window.matterForge.evidence.source);
  if(report.imported.kind!=='local closed GLB'||report.imported.metadata.sha256!==report.uploadArtifact.sha256||report.imported.sampling.pointCount!==1152)throw Error('Actual GLB import identity/count gate failed');
  await page.screenshot({path:path.join(out,'initial.png')});
  await page.evaluate(async()=>{for(let i=0;i<4;i++)await window.matterForge.advance(240);});
  report.cast=await page.evaluate(()=>window.matterForge.getState());await page.screenshot({path:path.join(out,'cast.png')});
  await page.evaluate(async()=>{await window.matterForge.release(1);for(let i=0;i<4;i++)await window.matterForge.advance(240);});
  report.solid=await page.evaluate(()=>window.matterForge.getState());await page.screenshot({path:path.join(out,'solid.png')});
  await page.click('#cut');await page.evaluate(async()=>{await window.matterForge.advance(240);await window.matterForge.advance(240);});
  report.cut=await page.evaluate(()=>window.matterForge.getState());await page.screenshot({path:path.join(out,'cut.png')});
  report.evidence=await page.evaluate(()=>window.matterForge.evidence);
  if(report.evidence.errors.length||report.errors.length||routes.length)throw Error('Browser errors');
  if(!report.evidence.edits.length)throw Error('No material edit');
  const priorReset=report.evidence.resetCount;
  await page.evaluate(()=>Promise.all([window.matterForge.reset(),window.matterForge.reset()]));report.reset=await page.evaluate(n=>window.matterForge.getState().steps===0&&window.matterForge.evidence.resetCount===n+1,priorReset);
  await page.evaluate(()=>window.matterForge.dispose());report.disposed=await page.evaluate(()=>window.matterForge.evidence.disposed);
  report.deviceLossReason=await page.evaluate(()=>window.matterForge.evidence.deviceLossReason);
  if(!report.reset||!report.disposed||report.deviceLossReason!=='destroyed')throw Error('Lifecycle failure');
  for(const [name,digest]of Object.entries(servedHashes))if(report.sourceHashes[name]&&report.sourceHashes[name]!==digest)throw Error('Served source differs from snapshot: '+name);
  for(const group of [report.sourceHashes,servedHashes])for(const [name,digest]of Object.entries(group))if(sha(await readFile(path.join(root,name)))!==digest)throw Error('Executed source changed: '+name);
  report.status='passed';
 }catch(e){report.failure=String(e.stack??e);process.exitCode=1;}
 finally{clearTimeout(watchdog);await browser?.close();await new Promise(r=>server.close(r));const raw=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),raw);await writeFile(path.join(out,'report.sha256'),sha(raw)+'\n');console.log(JSON.stringify({out,status:report.status,failure:report.failure,errors:report.errors,routes,edits:report.evidence?.edits},null,2));}
}
