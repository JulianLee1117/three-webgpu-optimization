import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),serve=args.includes('--serve');
if(args.some(arg=>arg!=='--serve'))throw Error('Supported options: --serve (idle UI server), or no options (bounded GPU probe)');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex'),served={},requests=[];
const prefixes=['/experiments/footprint-filtering/','/node_modules/three/src/'];
const exactPaths=new Set(['/experiments/material-mips/autograd.js','/experiments/constraint-editing/rebind.js']);
const inside=(base,file)=>{const relative=path.relative(base,file);return relative!==''&&!relative.startsWith('..')&&!path.isAbsolute(relative);};
const server=createServer(async(req,res)=>{
  try{
    if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,{'Allow':'GET, HEAD'}).end();return;}
    let name=decodeURIComponent(new URL(req.url,'http://127.0.0.1').pathname);
    if(name==='/')name='/experiments/footprint-filtering/'+(serve?'app.html':'probe.html');
    if(name==='/favicon.ico'){res.writeHead(204).end();return;}
    if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid path');
    const prefix=prefixes.find(p=>name.startsWith(p));if(!prefix&&!exactPaths.has(name))throw Error('Route outside experiment allowlist');
    const file=await realpath(path.join(root,name));
    const base=await realpath(path.join(root,prefix??path.posix.dirname(name)));
    if(!inside(base,file))throw Error('Route containment failed');
    const bytes=await readFile(file),digest=sha(bytes);
    if(!serve&&served[name]&&served[name]!==digest)throw Error('Served source changed');
    served[name]=digest;
    res.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html; charset=utf-8':name.endsWith('.json')?'application/json':name.endsWith('.md')?'text/plain; charset=utf-8':'text/javascript; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
    res.end(req.method==='HEAD'?undefined:bytes);
  }catch(error){requests.push({url:req.url,status:404,message:String(error.message)});res.writeHead(404,{'Content-Type':'text/plain'}).end('Not found');}
});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(serve?5197:0,'127.0.0.1',resolve);});
const url=`http://127.0.0.1:${server.address().port}/`;
if(serve)console.log(JSON.stringify({url,mode:'idle-until-click'}));
else{
  const out=path.join(root,'results/development/footprint-filtering',new Date().toISOString().replaceAll(':','-')+'-gpu');
  await mkdir(out,{recursive:true});
  const report={kind:'footprint-filtering-gpu-report-v1',status:'failed',errors:[],requestFailures:requests,sourceHashes:{},sourceSnapshots:{},servedHashes:served};
  const sourceFiles=['scripts/run-footprint-filtering.mjs','experiments/material-mips/autograd.js','experiments/constraint-editing/rebind.js',
    ...(await readdir(path.join(root,'experiments/footprint-filtering'))).filter(n=>/\.(js|mjs|html|md)$/.test(n)).map(n=>'experiments/footprint-filtering/'+n)];
  for(const name of sourceFiles){const bytes=await readFile(path.join(root,name));report.sourceHashes[name]=sha(bytes);if(/\.(js|mjs|html)$/.test(name))report.sourceSnapshots[name]=bytes.toString('utf8');}
  const telemetry=()=>{try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',windowsHide:true,timeout:3000}).trim();}catch{return null;}};
  let browser,page,watchdogExpired=false;
  report.preGpu=telemetry();
  const watchdog=setTimeout(()=>{watchdogExpired=true;browser?.close().catch(()=>{});},45000);
  try{
    browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();
    if(watchdogExpired)throw Error('Browser startup exceeded watchdog');
    page=await browser.newPage({viewport:{width:640,height:480}});
    page.on('pageerror',error=>report.errors.push(error.message));
    page.on('console',message=>{if(message.type()==='error')report.errors.push(message.text());});
    await page.goto(url,{timeout:15000});await page.waitForFunction(()=>window.footprintProbe,null,{timeout:15000});
    report.probe=await page.evaluate(()=>window.footprintProbe.runProbe());
    if(report.probe.status!=='passed')throw Error(report.probe.failure??'Footprint numerical gate failed');
    if(report.errors.length||requests.length)throw Error('Browser or route errors');
    if(!report.probe.lifecycle?.disposed)throw Error('Owned device was not disposed');
    for(const [name,digest]of Object.entries(served))if(sha(await readFile(path.join(root,name)))!==digest)throw Error('Served source changed during run');
    for(const [name,digest]of Object.entries(report.sourceHashes))if(sha(await readFile(path.join(root,name)))!==digest)throw Error('Captured source changed during run');
    report.status='passed';
  }catch(error){
    report.failure=String(error.stack??error);process.exitCode=1;
    if(!report.probe&&page&&!page.isClosed())try{report.partialProbe=await page.evaluate(()=>window.footprintProbeResult??null);}catch{}
  }finally{
    clearTimeout(watchdog);await browser?.close();await new Promise(resolve=>server.close(resolve));
    report.watchdogExpired=watchdogExpired;report.postGpu=telemetry();
    const bytes=JSON.stringify(report,(_,value)=>ArrayBuffer.isView(value)?Array.from(value):value,2)+'\n';
    await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.sha256'),sha(bytes)+'\n',{flag:'wx'});
    console.log(JSON.stringify({out,status:report.status,failure:report.failure,watchdogExpired,
      cells:report.probe?.cells.map(c=>({kind:c.kind,id:c.id,passed:c.passed,quality:c.quality})),lifecycle:report.probe?.lifecycle,
      errors:report.errors,requestFailures:requests,preGpu:report.preGpu,postGpu:report.postGpu},null,2));
  }
}
