import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright-core';
if(process.argv.length>2)throw Error('No options: one bounded casting matrix, no tuning or repeated runs.');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),sha=raw=>createHash('sha256').update(raw).digest('hex');
const prefix='/experiments/matter-forge/',sourceFiles=['scripts/check-matter-forge-casting-gpu.mjs',...['casting-probe.html','casting-probe.mjs','cpu.mjs','gpu.mjs','casting-fixture.mjs','casting-fixture-v2.mjs'].map(f=>prefix.slice(1)+f)];
const allowed=new Set(sourceFiles.slice(1).map(f=>'/'+f)),referencePath='experiments/matter-forge/evidence/casting-cpu-reference.json';
const referenceSHA='23c1c8eeb892e7511fa453b261f86b28383102b50caf3394a6e2aff8de59a411';
const originalCPUReportSHA='b2652a80f8b7b531696f4eaf9847f031a359a6dc2974978f859b63333e73147e';
const report={kind:'matter-forge-casting-gpu-report-v2',createdAt:new Date().toISOString(),status:'failed',sourceHashes:{},sourceSnapshots:{},servedHashes:{},errors:[],requestFailures:[],
  limits:{wallClockMs:45000,devices:1,maximumActiveSimulators:1,simulators:3,gridN:24,particles:1216,totalSteps:2880,advanceChunkSteps:240,advanceCalls:12,expectedComputeDispatches:14400}};
const output=path.join(root,'results/development/matter-forge-casting-gpu',report.createdAt.replaceAll(':','-'));await mkdir(output,{recursive:true});
let browser,page,server,watchdog,expired=false;
const telemetry=()=>{try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',timeout:3000,windowsHide:true}).trim();}catch{return null;}};
try{
  const referenceRaw=await readFile(path.join(root,referencePath));if(sha(referenceRaw)!==referenceSHA)throw Error('Pinned CPU reference checksum mismatch');
  const cpu=JSON.parse(referenceRaw);if(cpu.kind!=='matter-forge-casting-cpu-reference-v1'||cpu.status!=='passed'||cpu.sourceReport.sha256!==originalCPUReportSHA)throw Error('Unexpected CPU reference provenance');
  report.cpuReference={path:referencePath,publicReferenceSHA256:referenceSHA,sha256:originalCPUReportSHA,kind:cpu.sourceReport.kind,lanes:cpu.lanes};
  for(const name of sourceFiles){const raw=await readFile(path.join(root,name));report.sourceHashes[name]=sha(raw);report.sourceSnapshots[name]=raw.toString('utf8');}
  for(const[name,digest]of Object.entries(cpu.sourceHashes))if(report.sourceHashes[name]!==digest)throw Error(`CPU reference source mismatch: ${name}`);
  const base=await realpath(path.join(root,prefix));
  server=createServer(async(request,response)=>{
    try{
      if(!['GET','HEAD'].includes(request.method)){response.writeHead(405).end();return;}
      let name=decodeURIComponent(new URL(request.url,'http://127.0.0.1').pathname);if(name==='/')name=prefix+'casting-probe.html';if(name==='/favicon.ico'){response.writeHead(204).end();return;}
      if(name.includes('\\')||name.includes('\0')||name.split('/').includes('..')||!allowed.has(name))throw Error('Outside exact source allowlist');
      const file=await realpath(path.join(root,name)),relative=path.relative(base,file);if(!relative||relative.startsWith('..')||path.isAbsolute(relative))throw Error('Route containment');
      const raw=await readFile(file),digest=sha(raw);if(digest!==report.sourceHashes[name.slice(1)])throw Error('Source changed after snapshot');report.servedHashes[name]=digest;
      response.writeHead(200,{'Content-Type':name.endsWith('.html')?'text/html; charset=utf-8':'text/javascript; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});response.end(request.method==='HEAD'?undefined:raw);
    }catch(error){report.requestFailures.push({url:request.url,message:error.message});response.writeHead(404).end('Not found');}
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});report.preGpu=telemetry();
  watchdog=setTimeout(()=>{expired=true;browser?.close().catch(()=>{});},45000);
  browser=await chromium.launch({channel:'chrome',headless:true});if(expired)throw Error('Watchdog during launch');report.browser=browser.version();
  page=await browser.newPage({viewport:{width:96,height:96}});page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}/`,{timeout:15000});await page.waitForFunction(()=>window.matterCastingProbe,null,{timeout:15000});
  const json=await page.evaluate(async reference=>JSON.stringify(await window.matterCastingProbe.runProbe(reference),(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v),report.cpuReference);report.probe=JSON.parse(json);
  if(report.probe.status!=='passed')throw Error(report.probe.failure??'GPU casting gate failed');if(report.errors.length||report.requestFailures.length)throw Error('Browser or route errors');if(!report.probe.lifecycle.pass)throw Error('Lifecycle gate failed');
  for(const[name,digest]of Object.entries(report.sourceHashes))if(sha(await readFile(path.join(root,name)))!==digest)throw Error('Source changed: '+name);
  for(const name of allowed)if(report.servedHashes[name]!==report.sourceHashes[name.slice(1)])throw Error('Source not executed: '+name);
  report.status='passed';
}catch(error){report.failure=String(error.stack??error);process.exitCode=1;if(!report.probe&&page&&!page.isClosed())try{report.partialProbe=JSON.parse(await page.evaluate(()=>JSON.stringify(window.matterCastingProbeResult??null,(_,v)=>ArrayBuffer.isView(v)?Array.from(v):v)));}catch{}}
finally{
  clearTimeout(watchdog);try{await browser?.close();}catch(error){report.cleanupFailure=error.message;report.status='failed';process.exitCode=1;}if(server?.listening)await new Promise(resolve=>server.close(resolve));
  report.watchdogExpired=expired;report.postGpu=telemetry();if(expired){report.status='failed';process.exitCode=1;}
  const raw=JSON.stringify(report)+'\n';await writeFile(path.join(output,'report.json'),raw,{flag:'wx'});await writeFile(path.join(output,'report.sha256'),sha(raw)+'\n',{flag:'wx'});
  console.log(JSON.stringify({output,status:report.status,sha256:sha(raw),failure:report.failure,gates:report.probe?.gates,stages:report.probe?.stages.map(s=>({id:s.id,status:s.status,completedSteps:s.completedSteps,measurement:s.measurement})),cpuComparison:report.probe?.cpuComparison,lifecycle:report.probe?.lifecycle,errors:report.errors,requestFailures:report.requestFailures,preGpu:report.preGpu,postGpu:report.postGpu},null,2));
}
