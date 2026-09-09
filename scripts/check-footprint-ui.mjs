import {chromium} from 'playwright-core';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),url=args.find(a=>a.startsWith('--url='))?.slice(6)??'http://127.0.0.1:5197/';
if(args.some(a=>a!==`--url=${url}`)||!['http://127.0.0.1:5197/','http://127.0.0.1:5198/'].includes(url))throw Error('Only the source or portable loopback demo URL is supported.');
const out=path.join(root,'results/development/footprint-filtering',new Date().toISOString().replaceAll(':','-')+'-ui');await mkdir(out,{recursive:true});
const report={kind:'footprint-ui-v1',url,status:'failed',errors:[],checks:{},sourceHashes:{}};
for(const name of ['experiments/footprint-filtering/app.js','experiments/footprint-filtering/app.html','experiments/footprint-filtering/compiler.js','experiments/footprint-filtering/fixtures.mjs','scripts/check-footprint-ui.mjs'])report.sourceHashes[name]=createHash('sha256').update(await readFile(path.join(root,name))).digest('hex');
const telemetry=()=>{try{return execFileSync('nvidia-smi',['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],{encoding:'utf8',windowsHide:true,timeout:3000}).trim();}catch{return null;}};
let browser;const guard=setTimeout(()=>browser?.close().catch(()=>{}),45000);report.preGpu=telemetry();
try{
  browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();
  const page=await browser.newPage({viewport:{width:1360,height:1080}});
  page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(url);await page.waitForFunction(()=>window.footprintDemo);
  report.checks.initial=await page.evaluate(()=>window.footprintDemo.status);if(report.checks.initial.allocated)throw Error('GPU allocated before click');
  // Deterministically reject an old initialization after a newer session opens.
  // This exercises the real asynchronous UI path, not a mock lifecycle model.
  await page.evaluate(()=>{
    const gpu=navigator.gpu,original=gpu.requestAdapter.bind(gpu);let first=true;
    window.restoreAdapter=()=>{gpu.requestAdapter=original;};
    gpu.requestAdapter=(...args)=>{if(first){first=false;return new Promise((resolve,reject)=>window.rejectOldOpen=reject);}return original(...args);};
    window.footprintDemo.open();window.footprintDemo.close();window.footprintDemo.open();
  });
  await page.waitForFunction(()=>window.footprintDemo.status.allocated&&!window.footprintDemo.status.opening,null,{timeout:18000});
  report.checks.staleOpenIsolation=await page.evaluate(async()=>{window.rejectOldOpen(Error('Injected obsolete initialization failure'));await Promise.resolve();await Promise.resolve();window.restoreAdapter();return window.footprintDemo.status;});
  if(!report.checks.staleOpenIsolation.allocated)throw Error('An obsolete initialization closed the newer session');
  await page.evaluate(()=>window.footprintDemo.close());
  await page.click('#start');await page.waitForFunction(()=>window.footprintDemo.status.allocated&&!window.footprintDemo.status.opening,null,{timeout:18000});
  report.checks.open=await page.evaluate(()=>window.footprintDemo.status);
  report.checks.patterns=[];
  for(const kind of ['energy','interference','weave']){
    await page.selectOption('#pattern',kind);await page.evaluate(()=>window.footprintDemo.setTime(.31));
    report.checks.patterns.push(await page.evaluate(()=>window.footprintDemo.status));
    if(!report.checks.patterns.at(-1).allocated)throw Error('Pattern switch disposed after error');
  }
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.locator('#density').fill('48');await page.locator('#exposure').fill('40');
  report.checks.controls=await page.evaluate(()=>({density:document.querySelector('#density-value').textContent,exposure:document.querySelector('#exposure-value').textContent}));
  if(report.checks.controls.density!=='48.00×'||report.checks.controls.exposure!=='40 ms')throw Error('Control labels did not update');
  await page.click('#play');await page.waitForFunction(()=>window.footprintDemo.status.playing);await page.waitForFunction(()=>!window.footprintDemo.status.playing,null,{timeout:15000});
  report.checks.boundedPreview=await page.evaluate(()=>({state:window.footprintDemo.status,message:document.querySelector('#status').textContent}));
  if(!report.checks.boundedPreview.message.includes('Preview complete'))throw Error('Preview did not complete its declared cap');
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.footprintDemo.render());
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  report.checks.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(report.checks.mobileOverflow)throw Error('Mobile overflow');
  await page.click('#stop');report.checks.closed=await page.evaluate(()=>window.footprintDemo.status);if(report.checks.closed.allocated)throw Error('Close retained GPU');
  await page.click('#start');await page.waitForFunction(()=>window.footprintDemo.status.allocated&&!window.footprintDemo.status.opening,null,{timeout:12000});
  await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));delete document.hidden;});
  report.checks.simulatedVisibilityHandler=await page.evaluate(()=>window.footprintDemo.status);if(report.checks.simulatedVisibilityHandler.allocated)throw Error('Visibility handler retained GPU');
  if(report.errors.length)throw Error('Browser errors');report.status='passed';
}catch(error){report.failure=String(error.stack??error);process.exitCode=1;}
finally{clearTimeout(guard);await browser?.close();report.postGpu=telemetry();const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.sha256'),createHash('sha256').update(bytes).digest('hex')+'\n',{flag:'wx'});console.log(JSON.stringify({out,...report},null,2));}
