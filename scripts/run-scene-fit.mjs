import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {chromium} from 'playwright-core';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const scene=process.argv.find(x=>x.startsWith('--scene='))?.slice(8)??'rover';
const seed=Number(process.argv.find(x=>x.startsWith('--seed='))?.slice(7)??11);
if(!['rover','robot','pavilion'].includes(scene)||![11,29,47].includes(seed))throw Error('Unknown declared scene/seed cell');
const serve=process.argv.includes('--serve'),hash=data=>createHash('sha256').update(data).digest('hex');
const inside=(base,target)=>{const relative=path.relative(base,target);return relative!==''&&!relative.startsWith('..')&&!path.isAbsolute(relative);};
const server=createServer(async(req,res)=>{
  try{
    let name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);if(name==='/')name='/experiments/scene-fit/index.html';
    if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid path');
    const prefix=['/experiments/scene-fit/','/node_modules/three/build/'].find(p=>name.startsWith(p));if(!prefix)throw Error('Invalid route');
    const [base,file]=await Promise.all([realpath(path.join(root,prefix)),realpath(path.join(root,name))]);if(!inside(base,file))throw Error('Route escape');
    const bytes=await readFile(file);res.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html':name.endsWith('.json')?'application/json':'text/javascript','Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cache-Control':'no-store'});res.end(bytes);
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(serve?5188:0,'127.0.0.1',resolve));
const url=`http://127.0.0.1:${server.address().port}/`;
if(serve){console.log(JSON.stringify({url,mode:'idle-on-load'}));}
else{
  const out=path.join(root,'results/development/scene-fit',new Date().toISOString().replaceAll(':','-')+`-${scene}-${seed}`);await mkdir(out,{recursive:true});
  const report={kind:'editable-scene-fit-v2',status:'failed',scene,seed,budget:96,errors:[],sources:{}};
  for(const file of ['scripts/run-scene-fit.mjs',...['PROTOCOL.md','PROTOCOL-V2.md','scenes.js','optimizers.mjs','least-squares.mjs','gpu-loss.js','gpu-equations.js','engine.js','app.js','index.html'].map(x=>'experiments/scene-fit/'+x),'node_modules/three/build/three.webgpu.js','node_modules/three/build/three.core.js'])report.sources[file]=hash(await readFile(path.join(root,file)));
  function telemetry(){try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',windowsHide:true,timeout:3000}).trim();}catch{return null;}}
  let browser,page;report.preGpu=telemetry();const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),45000);
  try{
    const chrome=process.env.CHROME_PATH?.trim();browser=await chromium.launch({...(chrome?{executablePath:chrome}:{channel:'chrome'}),headless:true});report.browser=browser.version();
    page=await browser.newPage({viewport:{width:1280,height:900}});page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
    await page.goto(url,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>!!window.sceneFit,null,{timeout:10000});
    report.fixture=await page.evaluate(({scene,seed})=>window.sceneFit.initialize(scene,seed),{scene,seed});
    const methods=['random','coordinate','candidate','least-squares'],offset=[11,29,47].indexOf(seed);report.order=[...methods.slice(offset),...methods.slice(0,offset)];report.results=[];
    for(const method of report.order){report.results.push(await page.evaluate(method=>window.sceneFit.fit(method,96,false),method));await new Promise(resolve=>setTimeout(resolve,600));}
    await page.evaluate(()=>window.sceneFit.show('least-squares'));await page.screenshot({path:path.join(out,'candidate.png'),fullPage:true});
    await page.evaluate(()=>window.sceneFit.dispose());report.finalState=await page.evaluate(()=>({phase:window.sceneFit.state.phase,disposed:window.sceneFit.state.disposed,errors:window.sceneFit.state.errors}));
    if(report.errors.length||report.finalState.errors.length)throw Error('Browser/GPU errors');report.status='passed';
  }catch(error){report.failure=String(error.stack??error);if(page){report.failedState=await page.evaluate(()=>({phase:window.sceneFit?.state.phase,errors:window.sceneFit?.state.errors})).catch(()=>null);await page.evaluate(()=>window.sceneFit?.dispose()).catch(()=>{});}}
  finally{
    clearTimeout(watchdog);await browser?.close();await new Promise(resolve=>server.close(resolve));report.postGpu=telemetry();
    const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.json.sha256'),hash(bytes)+'\n',{flag:'wx'});
    console.log(JSON.stringify({out,status:report.status,failure:report.failure,initial:report.fixture&&{train:report.fixture.initialTrain,heldout:report.fixture.initialHeldout},results:report.results?.map(r=>({method:r.method,loss:r.bestLoss,heldout:r.finalHeldout,evaluations:r.evaluations,wallMs:r.wallMs})),preGpu:report.preGpu,postGpu:report.postGpu},null,2));if(report.status!=='passed')process.exitCode=1;
  }
}
