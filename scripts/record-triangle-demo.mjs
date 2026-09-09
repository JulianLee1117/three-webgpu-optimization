import {chromium} from 'playwright-core';
import {mkdir,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
const out=path.resolve('deliverables/triangle-query-video',new Date().toISOString().replaceAll(':','-'));await mkdir(out,{recursive:true});
let browser,context;const frames=[],errors=[];const watchdog=setTimeout(()=>browser?.close().catch(()=>{}),20000);
try{
  browser=await chromium.launch({channel:'chrome',headless:true});
  context=await browser.newContext({viewport:{width:1280,height:1060}});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:5190/');await page.waitForFunction(()=>window.triangleDemo);
  for(const order of ['ABC','ACB','ABC']){
    await page.selectOption('#order',order);await page.click('#compare');await page.waitForFunction(()=>window.triangleDemo.status.state==='ready');
    const result=await page.evaluate(()=>window.triangleDemo.lastResult);frames.push({order,result});
    await page.screenshot({path:path.join(out,`state-${frames.length-1}.png`)});
  }
  await context.close();context=null;
  if(errors.length)throw Error(errors.join('; '));
  const target=path.join(out,'triangle-distance-error.mp4');
  const manifest=path.join(out,'frames.txt');await writeFile(manifest,frames.map((_,i)=>`file 'state-${i}.png'\nduration 2\n`).join('')+`file 'state-2.png'\n`);
  execFileSync('ffmpeg',['-y','-f','concat','-safe','1','-i',manifest,'-vf','fps=15','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',target],{windowsHide:true,stdio:'pipe',timeout:10000});
  await writeFile(path.join(out,'recording.json'),JSON.stringify({frames,errors,metric:'Three screenshots of actual GPU comparisons, each held for two seconds. Unchanged geometry and height; only vertex order changes. This is a slide clip, not a frame-rate recording.'},null,2));
  console.log(JSON.stringify({out,video:target,frames:frames.map(f=>({order:f.order,distanceRatio:f.result.distanceRatio}))},null,2));
}finally{clearTimeout(watchdog);await context?.close();await browser?.close();}
