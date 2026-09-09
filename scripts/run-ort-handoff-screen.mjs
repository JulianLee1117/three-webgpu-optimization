import {createServer}from'node:http';
import{chromium}from'playwright-core';
import{mkdir,readFile,writeFile}from'node:fs/promises';
import{createHash}from'node:crypto';
import{execFileSync}from'node:child_process';
import path from'node:path';
import{fileURLToPath}from'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const stock=process.argv.includes('--stock');
const side=Number(process.argv.find(x=>x.startsWith('--size='))?.slice(7)??128);
if(![128,256].includes(side))throw Error('Invalid bounded size.');
const lib=stock?'/node_modules/three/build/':'/.generated/ort-surface/';
const html=`<!doctype html><html><head><link rel="icon" href="data:,"><script type="importmap">{"imports":{"three/webgpu":"${lib}three.webgpu.js","three/tsl":"${lib}three.tsl.js"}}</script></head><body><script type="module" src="/experiments/ort-handoff/screen.js"></script></body></html>`;
const types={'.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.onnx':'application/octet-stream','.wasm':'application/wasm'};
const server=createServer(async(req,res)=>{
  const url=new URL(req.url,'http://localhost'),name=decodeURIComponent(url.pathname);
  const headers={'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cache-Control':'no-store'};
  if(name==='/'){res.writeHead(200,{...headers,'Content-Type':'text/html'});res.end(html);return;}
  if(!['/experiments/ort-handoff/','/experiments/ort-surface/','/.generated/ort-surface/','/node_modules/three/build/','/node_modules/onnxruntime-web-dev/dist/'].some(p=>name.startsWith(p))){res.writeHead(404).end();return;}
  const file=path.resolve(root,'.'+name);if(!file.startsWith(root+path.sep)){res.writeHead(403).end();return;}
  try{const bytes=await readFile(file);res.writeHead(200,{...headers,'Content-Type':types[path.extname(file)]??'application/octet-stream'});res.end(bytes);}catch{res.writeHead(404).end();}
});
const hash=b=>createHash('sha256').update(b).digest('hex');
const out=path.join(root,'results/development/ort-handoff',new Date().toISOString().replaceAll(':','-')+`-${stock?'stock':'four-lane'}-${side}`);
await mkdir(out,{recursive:true});
const report={kind:'ort-handoff-screen-v1',status:'failed',stock,side,errors:[],sources:{}};
for(const file of ['experiments/ort-handoff/PROTOCOL.md','experiments/ort-handoff/screen.js','scripts/run-ort-handoff-screen.mjs','experiments/ort-surface/wave.onnx','experiments/ort-surface/wave.json',...['three.webgpu.js','three.core.js','three.tsl.js'].map(name=>lib.slice(1)+name),...(stock?[]:['.generated/ort-surface/manifest.json']),'node_modules/onnxruntime-web-dev/dist/ort.webgpu.bundle.min.mjs'])report.sources[file]=hash(await readFile(path.join(root,file)));
function telemetry(){try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{windowsHide:true,encoding:'utf8',timeout:3000}).trim();}catch{return null;}}
report.preGpu=telemetry();
await new Promise(r=>server.listen(0,'127.0.0.1',r));
let browser,page;const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),30000);
try{
  browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:[]});
  report.browser=browser.version();page=await browser.newPage({viewport:{width:500,height:400}});
  page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'domcontentloaded',timeout:10000});
  await page.waitForFunction(()=>!!window.handoffScreen,null,{timeout:10000});
  report.environment=await page.evaluate(({side,stock})=>window.handoffScreen.initialize(side,stock),{side,stock});
  report.proof=await page.evaluate(()=>window.handoffScreen.validate());
  if(!stock)report.timing=await page.evaluate(()=>window.handoffScreen.measure());
  await page.evaluate(()=>window.handoffScreen.dispose());
  report.finalState=await page.evaluate(()=>window.handoffScreen.state);
  if(report.errors.length||report.finalState.errors.length)throw Error('Browser/GPU errors.');
  report.status='passed';
}catch(e){report.failure=String(e.stack??e);if(page){report.failedState=await page.evaluate(()=>window.handoffScreen?.state).catch(()=>null);await page.evaluate(()=>window.handoffScreen?.dispose()).catch(()=>{});}}
finally{
  clearTimeout(watchdog);await browser?.close();await new Promise(r=>server.close(r));report.postGpu=telemetry();
  const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.json.sha256'),hash(bytes)+'\n',{flag:'wx'});
  console.log(JSON.stringify({out,status:report.status,failure:report.failure,summary:report.timing?.summary,preGpu:report.preGpu,postGpu:report.postGpu},null,2));if(report.status!=='passed')process.exitCode=1;
}
