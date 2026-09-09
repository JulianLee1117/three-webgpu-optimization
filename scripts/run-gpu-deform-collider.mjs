import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {chromium} from 'playwright-core';
import {correctTriangleModule} from '../experiments/gpu-deform-collider/closest-point-correction.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const arg=(key,fallback)=>process.argv.find(x=>x.startsWith(`--${key}=`))?.split('=')[1]??fallback;
const family=arg('family','wave'),segments=Number(arg('segments',16)),queryCount=Number(arg('queries',128));
const serve=process.argv.includes('--serve'),intervals=process.argv.includes('--intervals'),throughput=process.argv.includes('--throughput');
const corrected=process.argv.includes('--corrected');
const triangleProbe=process.argv.includes('--triangle-probe');
const derivatives=process.argv.includes('--derivatives');
const bugDemo=process.argv.includes('--bug-demo');
if(bugDemo&&corrected)throw Error('The side-by-side demo needs the original upstream module; omit --corrected');
if(bugDemo&&!serve)throw Error('--bug-demo requires --serve; use --triangle-probe for automated checks');
if(throughput&&!intervals)throw Error('Throughput screen requires --intervals');
const hash=b=>createHash('sha256').update(b).digest('hex');
const servedSources={};
const inside=(base,file)=>{const rel=path.relative(base,file);return rel!==''&&!rel.startsWith('..')&&!path.isAbsolute(rel);};
const server=createServer(async(req,res)=>{
  try{
    let name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);if(name==='/'){res.writeHead(302,{Location:`/experiments/gpu-deform-collider/${bugDemo?'triangle-demo':'index'}.html`}).end();return;}
    if(corrected&&/^\/experiments\/gpu-deform-collider\/triangle-demo\.(?:html|js)$/.test(name)){res.writeHead(409,{'Content-Type':'text/plain'}).end('Use npm run demo:triangle-query for the original-versus-corrected comparison.');return;}
    if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid path');
    const prefix=['/experiments/gpu-deform-collider/','/node_modules/three/build/','/node_modules/three-mesh-bvh/src/'].find(p=>name.startsWith(p));if(!prefix)throw Error('Invalid route');
    const [base,file]=await Promise.all([realpath(path.join(root,prefix)),realpath(path.join(root,name))]);if(!inside(base,file))throw Error('Route escape');
    let bytes=await readFile(file);
    if(corrected&&name==='/node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js')bytes=Buffer.from(correctTriangleModule(bytes.toString()));
    const digest=hash(bytes);if(!serve&&servedSources[name]&&servedSources[name]!==digest)throw Error('Source changed during experiment');servedSources[name]=digest;
    res.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html':name.endsWith('.json')?'application/json':name.endsWith('.md')?'text/plain':'text/javascript','Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cache-Control':'no-store'});res.end(bytes);
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(serve?(bugDemo?5190:5189):0,'127.0.0.1',resolve));const url=`http://127.0.0.1:${server.address().port}/`;
if(serve){console.log(JSON.stringify({url,mode:'idle-on-load'}));}
else {
  const out=path.join(root,'results/development/gpu-deform-collider',new Date().toISOString().replaceAll(':','-')+`-${family}-${segments}-${queryCount}`);await mkdir(out,{recursive:true});
  const report={kind:triangleProbe?'upstream-triangle-probe':'native-tsl-bounds-v2',phase:throughput?'throughput-screen':'canary',protocolComplete:false,status:'failed',family,segments,queryCount,intervals,triangleFunction:corrected?'local-correction':'upstream',errors:[],sources:{}};
  const files=['scripts/run-gpu-deform-collider.mjs','package-lock.json',...(await readdir(path.join(root,'experiments/gpu-deform-collider'))).map(x=>'experiments/gpu-deform-collider/'+x),'node_modules/three/build/three.webgpu.js','node_modules/three/build/three.core.js'];
  for(const file of files)report.sources[file]=hash(await readFile(path.join(root,file)));
  function telemetry(){try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',windowsHide:true,timeout:3000}).trim();}catch{return null;}}
  let browser,page;report.preGpu=telemetry();const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),45000);
  try{
    browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();page=await browser.newPage({viewport:{width:1200,height:1000}});
    page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
    page.on('response',r=>{if(r.status()>=400)report.errors.push(`${r.status()} ${new URL(r.url()).pathname}`);});
    await page.goto(url,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>!!window.colliderLab,null,{timeout:10000});
    if(triangleProbe){
      report.triangleProbe=await page.evaluate(async()=>{const {runTriangleProbe}=await import('./triangle-probe.js');return runTriangleProbe();});
      if(report.errors.length||report.triangleProbe.errors.length)throw Error('Triangle probe GPU error');
      if(corrected&&report.triangleProbe.mismatchCount!==0)throw Error('Corrected triangle probe failed');
      if(!corrected&&report.triangleProbe.knownFailureMismatchCount<1)throw Error('Original failure was not reproduced');
      report.status=corrected?'passed':'reproduced-upstream-failure';
    }else{
    report.fixture=await page.evaluate(async config=>{window.engine=await window.colliderLab.createEngine({...config,container:document.querySelector('#view')});return window.engine.info;},{family,segments,queryCount});
    if(derivatives){report.derivatives=await page.evaluate(()=>window.engine.validateDerivatives());if(!report.derivatives.ok)throw Error('Derivative numerical check failed');}
    if(intervals)report.compiler=await page.evaluate(()=>window.engine.enableIntervals());
    report.correctness=[];
    const methods=intervals?['refit','interval']:['refit'];
    for(const time of [0,.7,2.4])for(const method of methods)report.correctness.push(await page.evaluate(({time,method})=>window.engine.validate(time,method),{time,method}));
    if(report.correctness.some(r=>!r.ok))throw Error('Correctness gate failed');
    if(intervals){report.swept=[];for(const [start,end]of [[0,.75],[.75,1.5],[2.4,3.2]])report.swept.push(await page.evaluate(([a,b])=>window.engine.sweptCheck(a,b),[start,end]));if(report.swept.some(r=>!r.ok))throw Error('Swept sample gate failed');}
    if(throughput){report.negativeControls=await page.evaluate(()=>window.engine.negativeControls());if(!report.negativeControls.detected)throw Error('Negative controls failed to detect invalid bounds');report.throughput=await page.evaluate(()=>window.engine.batchedBenchmark());}
    report.timings=[];
    if(!throughput)for(const method of [...methods,...methods.slice().reverse()])report.timings.push(await page.evaluate(method=>window.engine.benchmark(method,20),method));
    await page.evaluate(()=>{window.engine.render(.7,'refit');document.querySelector('#status').textContent=JSON.stringify(window.engine.info,null,2);});
    await page.screenshot({path:path.join(out,'preview.png')});
    const proof=JSON.stringify(await page.evaluate(()=>window.engine.exportProof()));await writeFile(path.join(out,'proof.json'),proof,{flag:'wx'});report.proof={path:'proof.json',sha256:hash(proof),bytes:Buffer.byteLength(proof)};
    await page.evaluate(()=>window.engine.dispose());
    if(report.errors.length)throw Error('Browser/GPU errors');report.status='passed';
    }
  }catch(error){report.failure=String(error.stack??error);if(page){
    if(!report.proof)try{const proof=JSON.stringify(await page.evaluate(()=>window.engine?.exportProof()));if(proof){await writeFile(path.join(out,'proof.json'),proof,{flag:'wx'});report.proof={path:'proof.json',sha256:hash(proof),bytes:Buffer.byteLength(proof)};}}catch{}
    await page.evaluate(()=>window.engine?.dispose()).catch(()=>{});
  }}
  finally{
    clearTimeout(watchdog);await browser?.close();await new Promise(resolve=>server.close(resolve));report.postGpu=telemetry();
    report.servedSources=servedSources;report.changedSources=[];
    for(const [name,digest] of Object.entries(servedSources)){let bytes=await readFile(path.join(root,name));if(corrected&&name==='/node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js')bytes=Buffer.from(correctTriangleModule(bytes.toString()));if(hash(bytes)!==digest)report.changedSources.push(name);}
    if(report.changedSources.length){report.status='failed';report.failure='Sources changed during experiment';}
    const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.json.sha256'),hash(bytes)+'\n',{flag:'wx'});
    console.log(JSON.stringify({out,status:report.status,failure:report.failure,errors:report.errors,fixture:report.fixture,triangleProbe:report.triangleProbe&&{status:report.triangleProbe.status,mismatchCount:report.triangleProbe.mismatchCount,knownFailureMismatchCount:report.triangleProbe.knownFailureMismatchCount},correctness:report.correctness?.map(r=>({method:r.method,time:r.time,ok:r.ok,wrong:r.wrong,boundsErrors:r.bounds.errorCount})),timings:report.timings?.map(r=>({method:r.method,medianMs:r.medianMs})),throughput:report.throughput?.medians,preGpu:report.preGpu,postGpu:report.postGpu},null,2));if(!['passed','reproduced-upstream-failure'].includes(report.status))process.exitCode=1;
  }
}
