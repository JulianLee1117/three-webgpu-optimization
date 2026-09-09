import { createServer } from 'node:http';
import { chromium } from 'playwright-core';
import { readFile, readdir, mkdir, writeFile, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const script=fileURLToPath(import.meta.url);
const root=path.resolve(path.dirname(script),'..');
const base=path.join(root,'results/development/zipdepth-export');
const hash=x=>createHash('sha256').update(x).digest('hex');
const fixtureFiles=new Set(['gradient-input.f32','gradient-reference.f32','gradient-original.f32','gradient-corrected.f32','original.onnx','corrected.onnx']);
const python=process.platform==='win32'?'.local-research/zipdepth-export-env/Scripts/python.exe':'.local-research/zipdepth-export-env/bin/python';
const prepareHelp=`Create the pinned CPU environment described in experiments/zipdepth-export/README.md, then run from the repository root:\n${python} experiments/zipdepth-export/prepare.py\n${python} experiments/zipdepth-export/probe.py`;
function inside(directory,file){
  const relative=path.relative(directory,file);
  return relative!=='' && relative!=='..' && !relative.startsWith('..'+path.sep) && !path.isAbsolute(relative);
}
export async function selectFixture(directory=base){
  let entries;
  try{entries=await readdir(directory,{withFileTypes:true});}
  catch(error){if(error.code==='ENOENT')throw Error('No CPU fixture directory.\n'+prepareHelp);throw error;}
  const canonicalBase=await realpath(directory);
  for(const entry of entries.filter(item=>item.isDirectory()).sort((a,b)=>b.name.localeCompare(a.name))){
    const candidate=path.join(directory,entry.name);
    if(!inside(canonicalBase,await realpath(candidate)))throw Error('Fixture directory leaves the selected results directory.');
    let bytes;
    try{bytes=await readFile(path.join(candidate,'report.json'));}
    catch(error){if(error.code==='ENOENT')continue;throw error;}
    const report=JSON.parse(bytes);
    if(report.status!=='passed' || !Array.isArray(report.shape) || report.shape.join(',')!=='1,3,384,384')continue;
    if(report.kind!=='zipdepth-export-fidelity-v1')throw Error('Unexpected fixture report kind.');
    if(hash(bytes)!==(await readFile(path.join(candidate,'report.json.sha256'),'utf8')).trim())throw Error('Fixture report checksum mismatch');
    if(!report.artifacts || [...fixtureFiles].some(file=>!Object.hasOwn(report.artifacts,file)))throw Error('Fixture report is missing required artifacts.\n'+prepareHelp);
    const canonicalCandidate=await realpath(candidate);
    for(const [file,item]of Object.entries(report.artifacts)){
      if(!/^[a-zA-Z0-9._-]+$/.test(file) || file==='.' || file==='..')throw Error('Invalid fixture artifact filename.');
      let actualFile;
      try{actualFile=await realpath(path.join(candidate,file));}
      catch(error){if(error.code==='ENOENT')throw Error(`Fixture artifact is missing: ${file}.\n${prepareHelp}`);throw error;}
      if(!inside(canonicalCandidate,actualFile))throw Error('Fixture artifact leaves the selected fixture directory.');
      if(hash(await readFile(actualFile))!==item.sha256)throw Error('Artifact checksum mismatch: '+file);
    }
    return {fixture:candidate,fixtureReport:report};
  }
  throw Error('No completed 384x384 CPU fixture report.\n'+prepareHelp);
}

// Decode before checking each route's own directory boundary. A repository-wide
// boundary would permit encoded traversal into unrelated local research files.
export function resolveRequestTarget(pathname,fixture){
  let name;
  try{name=decodeURIComponent(pathname);}catch{return null;}
  if(name.includes('\\') || name.includes('\0'))return null;
  const source=path.join(root,'experiments/zipdepth-export');
  if(name==='/')return {file:path.join(source,'browser.html'),directory:source};
  if(name.startsWith('/fixture/')){
    const file=name.slice('/fixture/'.length);
    return fixtureFiles.has(file)?{file:path.join(fixture,file),directory:fixture}:null;
  }
  for(const prefix of ['/experiments/zipdepth-export/','/node_modules/three/build/','/node_modules/onnxruntime-web-dev/dist/']){
    if(!name.startsWith(prefix))continue;
    const relative=name.slice(prefix.length);
    if(relative.split('/').includes('..'))return null;
    const directory=path.resolve(root,'.'+prefix),file=path.resolve(directory,relative);
    return inside(directory,file)?{file,directory}:null;
  }
  return null;
}
export async function readAllowedFile(target){
  const [directory,file]=await Promise.all([realpath(target.directory),realpath(target.file)]);
  if(!inside(directory,file))throw Error('Requested file leaves its allowed directory.');
  return readFile(file);
}

async function main(){
const {fixture,fixtureReport}=await selectFixture();
const types={'.js':'text/javascript','.mjs':'text/javascript','.html':'text/html','.json':'application/json','.onnx':'application/octet-stream','.wasm':'application/wasm','.f32':'application/octet-stream'};
const server=createServer(async(req,res)=>{
  const target=resolveRequestTarget(new URL(req.url,'http://localhost').pathname,fixture);
  if(!target){res.writeHead(404).end();return;}
  try{const bytes=await readAllowedFile(target);res.writeHead(200,{'Content-Type':types[path.extname(target.file)]??'application/octet-stream','Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp','Cache-Control':'no-store'});res.end(bytes);}catch{res.writeHead(404).end();}
});
await new Promise(r=>server.listen(process.argv.includes('--serve')?5187:0,'127.0.0.1',r));
const url=`http://127.0.0.1:${server.address().port}/`;
if(process.argv.includes('--serve')){
  console.log(JSON.stringify({url,fixture,mode:'idle-on-load'}));
}else{
  const out=path.join(root,'results/development/zipdepth-browser',new Date().toISOString().replaceAll(':','-'));
  await mkdir(out,{recursive:true});
  const report={kind:'zipdepth-browser-fidelity-v1',status:'failed',fixture:path.relative(root,fixture),fixtureSha256:hash(await readFile(path.join(fixture,'report.json'))),cpuPatchSha256:fixtureReport.patchSha256,errors:[],sources:{}};
  for(const file of ['scripts/run-zipdepth-browser.mjs','experiments/zipdepth-export/browser.html','experiments/zipdepth-export/browser.js','node_modules/three/build/three.webgpu.js','node_modules/three/build/three.core.js','node_modules/three/build/three.tsl.js','node_modules/onnxruntime-web-dev/dist/ort.webgpu.bundle.min.mjs'])report.sources[file]=hash(await readFile(path.join(root,file)));
  function telemetry(){try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{windowsHide:true,encoding:'utf8',timeout:3000}).trim();}catch{return null;}}
  report.preGpu=telemetry();let browser,page;
  const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),45000);
  try{
    const chromePath=process.env.CHROME_PATH?.trim();
    browser=await chromium.launch({headless:true,...(chromePath?{executablePath:chromePath}:{channel:'chrome'})});report.browser=browser.version();
    page=await browser.newPage({viewport:{width:1280,height:850}});
    page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
    await page.goto(url,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>!!window.zipdepthProbe,null,{timeout:10000});
    report.proof=await page.evaluate(()=>window.zipdepthProbe.run());
    await page.screenshot({path:path.join(out,'proof.png'),fullPage:true});
    await page.evaluate(()=>window.zipdepthProbe.dispose());report.finalState=await page.evaluate(()=>window.zipdepthProbe.state);
    if(report.errors.length || report.finalState.errors.length)throw Error('Browser/GPU errors.');report.status='passed';
  }catch(e){report.failure=String(e.stack??e);if(page){report.failedState=await page.evaluate(()=>window.zipdepthProbe?.state).catch(()=>null);await page.evaluate(()=>window.zipdepthProbe?.dispose()).catch(()=>{});}}
  finally{
    clearTimeout(watchdog);await browser?.close();await new Promise(r=>server.close(r));report.postGpu=telemetry();
    const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.json.sha256'),hash(bytes)+'\n',{flag:'wx'});
    console.log(JSON.stringify({out,status:report.status,proof:report.proof,failure:report.failure,preGpu:report.preGpu,postGpu:report.postGpu},null,2));if(report.status!=='passed')process.exitCode=1;
  }
}
}

if(process.argv[1] && path.resolve(process.argv[1])===script){
  main().catch(error=>{console.error(error.message??error);process.exitCode=1;});
}
