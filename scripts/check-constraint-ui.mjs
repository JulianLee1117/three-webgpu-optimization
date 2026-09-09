import {chromium} from 'playwright-core';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {fixtureVariant} from '../experiments/constraint-editing/solver-constrained.mjs';
const args=process.argv.slice(2),get=(name,fallback)=>args.find(a=>a.startsWith('--'+name+'='))?.slice(name.length+3)??fallback;
const url=get('url','http://127.0.0.1:5195/'),kind=get('kind','ribbon'),variant=get('variant','original'),tag=get('tag','source');
assert.ok(['ribbon','tentacle'].includes(kind)&&['original','late-target'].includes(variant));assert.match(tag,/^[a-z0-9-]+$/);
const address=new URL(url);assert.equal(address.protocol,'http:');assert.equal(address.hostname,'127.0.0.1');assert.ok(['5195','5196'].includes(address.port));
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),out=path.join(root,'results/development/constraint-editing-ui',new Date().toISOString().replaceAll(':','-')+'-'+tag);
await mkdir(out,{recursive:true});const report={kind:'constraint-editing-ui-v1',field:kind,variant,url,status:'failed',errors:[]};
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');let browser;
const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),45000);
try{
  browser=await chromium.launch({channel:'chrome',headless:true});const page=await browser.newPage({viewport:{width:1360,height:1050}});report.browser=browser.version();
  page.on('pageerror',e=>report.errors.push(e.message));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(url);await page.waitForFunction(()=>window.constraintLab);
  report.initial=await page.evaluate(()=>window.constraintLab.status);assert.equal(report.initial.allocated,false);assert.equal(report.initial.busy,false);
  await page.selectOption('#program',kind);await page.selectOption('#phase',variant);await page.click('#start');
  await page.waitForFunction(()=>window.constraintLab.status.allocated&&!window.constraintLab.status.busy,null,{timeout:15000});
  const target=await page.evaluate(()=>window.constraintLab.targetScreen),bounds=await page.locator('#viewport canvas').boundingBox();
  await page.mouse.move(bounds.x+target.x,bounds.y+target.y);await page.mouse.down();await page.mouse.move(bounds.x+target.x,bounds.y+target.y-12,{steps:4});await page.mouse.up();
  report.drag=await page.evaluate(()=>({status:window.constraintLab.status,target:window.constraintLab.targetScreen}));assert.equal(report.drag.status.custom,true);assert.ok(Math.abs(report.drag.target.y-target.y)>3);
  await page.click('#fit');await page.waitForFunction(()=>!window.constraintLab.status.busy,null,{timeout:20000});await page.evaluate(()=>window.constraintLab.pause());
  report.result=await page.evaluate(()=>window.constraintLab.result);assert.ok(report.result&&report.result.custom);assert.notEqual(report.result.status,'stopped');
  const download=page.waitForEvent('download');await page.click('#export');const downloaded=await download;await downloaded.saveAs(path.join(out,'parameters.json'));
  report.exported=JSON.parse(await readFile(path.join(out,'parameters.json'),'utf8'));
  assert.deepEqual(report.exported.parameters,report.result.exportedParameters);assert.equal(report.exported.custom,true);assert.equal(report.exported.kind,kind);assert.equal(report.exported.variant,variant);
  assert.notDeepEqual(report.exported.target.position,fixtureVariant(kind,variant).target.position);
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert.equal(report.mobileOverflow,false);await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  assert.deepEqual(report.errors,[],'Viewport resize produced a browser/GPU error');
  await page.click('#reset');report.reset=await page.evaluate(()=>({status:window.constraintLab.status,result:window.constraintLab.result,exportDisabled:document.querySelector('#export').disabled}));assert.equal(report.reset.status.custom,false);assert.equal(report.reset.result,null);assert.equal(report.reset.exportDisabled,true);
  report.earlyStop=await page.evaluate(async()=>{const fitting=window.constraintLab.fit();const stopping=window.constraintLab.close();await fitting;await stopping;return window.constraintLab.status;});assert.equal(report.earlyStop.allocated,false);assert.equal(report.earlyStop.busy,false);
  await page.click('#start');await page.waitForFunction(()=>window.constraintLab.status.allocated&&!window.constraintLab.status.busy,null,{timeout:15000});
  report.reopen=await page.evaluate(()=>window.constraintLab.status);assert.equal(report.reopen.custom,false);
  // Explicit handler test: this does not claim that browser tab-switch delivery
  // itself was simulated by the headless browser.
  await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'));});
  await page.waitForFunction(()=>!window.constraintLab.status.allocated&&!window.constraintLab.status.busy);
  report.visibilityHandler={simulatedHiddenProperty:true,status:await page.evaluate(()=>window.constraintLab.status)};
  assert.equal(report.errors.length,0);report.status='passed';
}catch(error){report.failure=String(error.stack??error);process.exitCode=1;}
finally{clearTimeout(watchdog);await browser?.close();const bytes=JSON.stringify(report,null,2)+'\n';await writeFile(path.join(out,'report.json'),bytes,{flag:'wx'});await writeFile(path.join(out,'report.sha256'),hash(bytes)+'\n',{flag:'wx'});console.log(JSON.stringify({out,status:report.status,failure:report.failure,customQuality:report.result?.validation,errors:report.errors},null,2));}
