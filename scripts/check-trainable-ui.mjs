import {chromium} from 'playwright-core';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
const url=process.argv.find(a=>a.startsWith('--url='))?.slice(6)??'http://127.0.0.1:5192/';
const out=path.resolve('results/development/trainable-tsl-ui',new Date().toISOString().replaceAll(':','-'));await mkdir(out,{recursive:true});
const report={url,status:'failed',errors:[]};let browser;const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),30000);
try{
  browser=await chromium.launch({channel:'chrome',headless:true});report.browser=browser.version();const page=await browser.newPage({viewport:{width:1360,height:1000}});
  page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(url);await page.waitForFunction(()=>window.trainableLab);report.initial=await page.evaluate(()=>window.trainableLab.status);
  if(report.initial.running||report.initial.allocated)throw Error('Not idle on load');
  await page.selectOption('#field','islands');await page.evaluate(()=>window.trainableLab.train());
  report.reference=await page.evaluate(()=>window.trainableLab.result);if(!report.reference?.utilityPassed)throw Error('Reference fit did not pass');
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  const downloadPromise=page.waitForEvent('download');await page.click('#export');const download=await downloadPromise;await download.saveAs(path.join(out,'exported-weights.json'));
  const saved=JSON.parse(await readFile(path.join(out,'exported-weights.json'),'utf8'));
  report.exportExact=saved.field==='islands'&&saved.weights.length===28&&saved.weights.every((v,i)=>v===report.reference.weights[i]);if(!report.exportExact)throw Error('Export changed learned weights');
  await page.click('#restore');await page.locator('#examples').click({position:{x:110,y:90}});await page.evaluate(()=>window.trainableLab.train());
  report.painted=await page.evaluate(()=>window.trainableLab.result);
  if(!report.painted?.customSamples||report.painted.final.heldout!==null||!(report.painted.final.trainingRMSE<report.painted.initial.trainingRMSE))throw Error('Painted fit did not consume custom examples');
  report.samplesChanged=report.painted.samples.some((v,i)=>v!==report.reference.samples[i]);if(!report.samplesChanged)throw Error('Brush did not change actual training data');
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(report.mobileOverflow)throw Error('Mobile overflow');
  report.hidden=await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'));const state=window.trainableLab.status;delete document.hidden;return state;});if(report.hidden.allocated)throw Error('Hidden page did not release resources');
  report.activeStop=await page.evaluate(async()=>{
    const training=window.trainableLab.train();
    while(window.trainableLab.status.running&&!window.trainableLab.status.trainer?.busy)await new Promise(resolve=>setTimeout(resolve,1));
    const before=window.trainableLab.status;window.trainableLab.stop();await training;return {before,after:window.trainableLab.status};
  });
  if(!report.activeStop.before.trainer?.busy||report.activeStop.after.running||report.activeStop.after.allocated)throw Error('Active stop failed');
  if(report.errors.length)throw Error('Browser errors');report.status='passed';
}catch(error){report.failure=String(error.stack??error);process.exitCode=1;}
finally{clearTimeout(watchdog);await browser?.close();const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes);await writeFile(path.join(out,'report.sha256'),createHash('sha256').update(bytes).digest('hex')+'\n');console.log(JSON.stringify({out,status:report.status,errors:report.errors,failure:report.failure,exportExact:report.exportExact,samplesChanged:report.samplesChanged,paintedTrainingRMSE:report.painted&&[report.painted.initial.trainingRMSE,report.painted.final.trainingRMSE]},null,2));}
