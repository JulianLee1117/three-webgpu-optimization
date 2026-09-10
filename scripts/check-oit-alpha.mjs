import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';

if(process.argv.length>2) throw Error('No options: one bounded 32x32 GPU correctness run.');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sha=raw=>createHash('sha256').update(raw).digest('hex');
const oitPath='node_modules/three-r186/examples/jsm/tsl/display/OITPassNode.js';
const before='return vec4( mix( accumColor, beautyNode.rgb, revealageNode ), beautyNode.a );';
const after='return vec4( mix( accumColor, beautyNode.rgb, revealageNode ), float( 1 ).sub( revealageNode ).add( beautyNode.a.mul( revealageNode ) ) );';
const sourceFiles=['scripts/check-oit-alpha.mjs','experiments/oit-alpha-correctness/probe.html','experiments/oit-alpha-correctness/probe.js',oitPath,
  'node_modules/three-r186/src/renderers/common/RenderPipeline.js','node_modules/three-r186/src/renderers/webgpu/utils/WebGPUTextureUtils.js','node_modules/three-r186/package.json'];
const report={kind:'oit-alpha-correctness-report-v1',createdAt:new Date().toISOString(),status:'failed',sourceHashes:{},sourceSnapshots:{},servedHashes:{},errors:[],requestFailures:[],
  limits:{wallClockMs:45000,rendererCount:1,fixtureLaneRenders:15,rendererRenderCalls:35,pixelsPerTarget:1024}};
for(const name of sourceFiles){const raw=await readFile(path.join(root,name));report.sourceHashes[name]=sha(raw);report.sourceSnapshots[name]=raw.toString('utf8');}
const original=report.sourceSnapshots[oitPath];
if(original.split(before).length!==2) throw Error('Expected exact OIT return occurs other than once');
const corrected=original.replace(before,after);
report.correction={source:oitPath,originalSHA256:sha(original),servedPath:'/overlay/OITPassNode.js',correctedSHA256:sha(corrected),before,after,modifiedNodeModules:false};
const prefixes=['/experiments/oit-alpha-correctness/','/node_modules/three-r186/'];
const server=createServer(async(request,response)=>{
  try{
    if(!['GET','HEAD'].includes(request.method)){response.writeHead(405).end();return;}
    let name=decodeURIComponent(new URL(request.url,'http://127.0.0.1').pathname);
    if(name==='/')name='/experiments/oit-alpha-correctness/probe.html';
    if(name==='/favicon.ico'){response.writeHead(204).end();return;}
    if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..'))throw Error('Invalid route');
    let raw;
    if(name==='/overlay/OITPassNode.js') raw=Buffer.from(corrected);
    else {
      const prefix=prefixes.find(p=>name.startsWith(p));if(!prefix)throw Error('Outside allowlist');
      const base=await realpath(path.join(root,prefix)),file=await realpath(path.join(root,name)),relative=path.relative(base,file);
      if(!relative||relative.startsWith('..')||path.isAbsolute(relative))throw Error('Route containment');
      raw=await readFile(file);
    }
    const digest=sha(raw);if(report.servedHashes[name]&&report.servedHashes[name]!==digest)throw Error('Served source changed');report.servedHashes[name]=digest;
    response.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html; charset=utf-8':'text/javascript; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
    response.end(request.method==='HEAD'?undefined:raw);
  }catch(error){report.requestFailures.push({url:request.url,message:String(error.message)});response.writeHead(404).end('Not found');}
});
const output=path.join(root,'experiments/oit-alpha-correctness/evidence',new Date().toISOString().replaceAll(':','-'));
await mkdir(output,{recursive:true});
function telemetry(){try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',timeout:3000,windowsHide:true}).trim();}catch{return null;}}
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
let browser,page,expired=false;
report.preGpu=telemetry();
const watchdog=setTimeout(()=>{expired=true;browser?.close().catch(()=>{});},report.limits.wallClockMs);
try{
  browser=await chromium.launch({channel:'chrome',headless:true});if(expired)throw Error('Watchdog during launch');report.browser=browser.version();
  page=await browser.newPage({viewport:{width:96,height:96}});
  page.on('pageerror',error=>report.errors.push(error.message));page.on('console',message=>{if(message.type()==='error')report.errors.push(message.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}/`,{timeout:15000});await page.waitForFunction(()=>window.oitAlphaProbe,null,{timeout:15000});
  report.probe=await page.evaluate(()=>window.oitAlphaProbe.runProbe());
  if(report.probe.status!=='passed')throw Error(report.probe.failure??'Probe gates failed');
  if(report.errors.length||report.requestFailures.length)throw Error('Browser/route errors');
  const lifecycle=report.probe.lifecycle;
  if(lifecycle.rendererCount!==1||lifecycle.fixtureLaneRenders!==15||lifecycle.rendererRenderCalls!==35||!lifecycle.disposed||lifecycle.deviceLossReason!=='destroyed')throw Error('Lifecycle gate failed');
  for(const[name,digest]of Object.entries(report.sourceHashes))if(sha(await readFile(path.join(root,name)))!==digest)throw Error(`Captured source changed: ${name}`);
  for(const[name,digest]of Object.entries(report.servedHashes))if(sha(name==='/overlay/OITPassNode.js'?corrected:await readFile(path.join(root,name)))!==digest)throw Error(`Executed source changed: ${name}`);
  if(report.servedHashes['/'+oitPath]!==report.correction.originalSHA256||report.servedHashes[report.correction.servedPath]!==report.correction.correctedSHA256)throw Error('Actual stock/overlay module identity mismatch');
  report.status='passed';
}catch(error){report.failure=String(error.stack??error);process.exitCode=1;if(!report.probe&&page&&!page.isClosed())try{report.partialProbe=await page.evaluate(()=>window.oitAlphaProbeResult??null);}catch{}}
finally{
  clearTimeout(watchdog);await browser?.close();await new Promise(resolve=>server.close(resolve));report.watchdogExpired=expired;report.postGpu=telemetry();
  const raw=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(output,'report.json'),raw,{flag:'wx'});await writeFile(path.join(output,'report.sha256'),sha(raw)+'\n',{flag:'wx'});
  console.log(JSON.stringify({output,status:report.status,sha256:sha(raw),failure:report.failure,cells:report.probe?.cells.map(({fixture,lane,expectedAlpha,centerRGBA,matchesPorterDuff})=>({fixture,lane,expectedAlpha,centerRGBA,matchesPorterDuff})),lifecycle:report.probe?.lifecycle,errors:report.errors,requestFailures:report.requestFailures,preGpu:report.preGpu,postGpu:report.postGpu},null,2));
}
