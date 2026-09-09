import {chromium} from 'playwright-core';
import {mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
const url=process.argv.find(x=>x.startsWith('--url='))?.slice(6)??'http://127.0.0.1:5190/';
const out=path.resolve('deliverables/triangle-query',new Date().toISOString().replaceAll(':','-'));await mkdir(out,{recursive:true});
const report={url,status:'failed',errors:[],checks:[]};let browser;
const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),30000);
try{
  browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();
  const page=await browser.newPage({viewport:{width:1280,height:1120}});
  page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(url);await page.waitForFunction(()=>window.triangleDemo);
  report.initial=await page.evaluate(()=>window.triangleDemo.status);if(report.initial.gpuAllocated||report.initial.busy)throw Error('Page did not start idle');
  for(const height of ['0.001','0.01','1'])for(const order of ['ABC','ACB','BAC','BCA','CAB','CBA']){
    await page.selectOption('#height',height);await page.selectOption('#order',order);
    const value=await page.evaluate(async()=>{await window.triangleDemo.compare();return {status:window.triangleDemo.status,result:window.triangleDemo.lastResult};});
    if(value.status.state!=='ready'||value.status.gpuAllocated||value.status.busy||!value.result)throw Error('UI comparison did not complete and release');
    if(Math.abs(value.result.fixed.distance-2*Number(height))>1e-5)throw Error('Incorrect fixed UI distance');report.checks.push({height,order,...value});
    if(height==='0.001'&&order==='ABC')await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  }
  for(const height of ['0.001','0.01','1'])if(report.checks.filter(r=>r.height===height&&r.result.distanceRatio>1.01).length!==3)throw Error('Vertex-order failure control changed');
  await page.selectOption('#height','0.001');await page.selectOption('#order','ABC');
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.triangleDemo.compare());await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(report.mobileOverflow)throw Error('Mobile horizontal overflow');
  report.stopped=await page.evaluate(async()=>{const pending=window.triangleDemo.compare();window.triangleDemo.stop();await pending;return window.triangleDemo.status;});
  if(report.stopped.gpuAllocated||report.stopped.busy)throw Error('Stop left a device allocated');
  if(report.errors.length)throw Error('Browser errors');report.status='passed';
}catch(error){report.failure=String(error.stack??error);process.exitCode=1;}
finally{
  clearTimeout(watchdog);await browser?.close();const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes);await writeFile(path.join(out,'report.sha256'),createHash('sha256').update(bytes).digest('hex')+'\n');
  console.log(JSON.stringify({out,status:report.status,checks:report.checks.length,errors:report.errors,failure:report.failure},null,2));
}
