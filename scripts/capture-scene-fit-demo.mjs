// One bounded UI smoke and a presentation recording. Not a benchmark result.
import { chromium } from 'playwright-core';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'deliverables/scene-fit',new Date().toISOString().replaceAll(':','-'));
await mkdir(out,{recursive:true});
const chrome=process.env.CHROME_PATH?.trim();
const browser=await chromium.launch({...(chrome?{executablePath:chrome}:{channel:'chrome'}),headless:true});
const deadline=setTimeout(()=>browser.close().catch(()=>{}),30000);
const errors=[];let page;
try{
  page=await browser.newPage({viewport:{width:1280,height:1080}});
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto('http://127.0.0.1:5188/',{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>!!window.sceneFit);
  const initial=await page.evaluate(()=>window.sceneFit.initialize('robot',29));
  await page.evaluate(()=>{
    const canvas=document.createElement('canvas');canvas.width=1280;canvas.height=720;
    const ctx=canvas.getContext('2d'),source=document.querySelector('#view canvas');
    const draw=()=>{
      ctx.fillStyle='#101723';ctx.fillRect(0,0,1280,720);
      ctx.fillStyle='#66dfc0';ctx.font='14px system-ui';ctx.fillText('THREE.JS / WEBGPU · EDITABLE SCENE FITTING',38,43);
      ctx.fillStyle='#eaf2ff';ctx.font='bold 34px system-ui';ctx.fillText('Keep the scene. Solve its proportions.',38,96);
      ctx.font='18px system-ui';['Reference','Initial parameters','Fitted · still editable'].forEach((label,i)=>ctx.fillText(label,38+400*i,160));
      ctx.drawImage(source,38,180,1200,340);
      ctx.fillStyle='#c7deef';ctx.font='19px system-ui';
      ctx.fillText(window.sceneFit.state.view==='heldout'?'Second camera · never used to fit':'Fitting camera · eight structural dimensions',38,564);
      ctx.fillStyle='#94a9c4';ctx.font='16px system-ui';ctx.fillText('96 candidate renders · standard direction search · synthetic fixture',38,610);
      ctx.fillText('Regular Three.js geometry. Export the numeric parameters and keep editing.',38,649);
    };
    draw();const stream=canvas.captureStream(12),chunks=[];
    const recorder=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp9',videoBitsPerSecond:2500000});
    recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};
    const timer=setInterval(draw,1000/12);recorder.start();
    window.stopSceneRecording=()=>new Promise(resolve=>{
      recorder.onstop=()=>{clearInterval(timer);stream.getTracks().forEach(track=>track.stop());const blob=new Blob(chunks,{type:'video/webm'});const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.readAsDataURL(blob);};recorder.stop();
    });
  });
  await new Promise(resolve=>setTimeout(resolve,700));
  const fit=await page.evaluate(()=>window.sceneFit.fit('candidate',96,true));
  await page.evaluate(()=>window.sceneFit.show('candidate'));
  await page.screenshot({path:path.join(out,'fitted.png'),fullPage:true});
  await new Promise(resolve=>setTimeout(resolve,1100));
  await page.selectOption('#view-mode','heldout');
  await page.screenshot({path:path.join(out,'heldout.png'),fullPage:true});
  await new Promise(resolve=>setTimeout(resolve,1500));
  const video=await page.evaluate(()=>window.stopSceneRecording());
  await writeFile(path.join(out,'scene-fit.webm'),Buffer.from(video,'base64'),{flag:'wx'});
  await page.locator('#tuning input').first().evaluate(input=>{input.value=Number(input.min)+(Number(input.max)-Number(input.min))*.65;input.dispatchEvent(new Event('input',{bubbles:true}));});
  const expected=await page.evaluate(()=>window.sceneFit.state.parameterPatch);
  const pending=page.waitForEvent('download');await page.click('#save');const download=await pending;
  await download.saveAs(path.join(out,'edited-parameters.json'));
  if(JSON.stringify(JSON.parse(await readFile(path.join(out,'edited-parameters.json'),'utf8')))!==JSON.stringify(expected))throw Error('Export does not match edited scene');
  await page.evaluate(()=>window.sceneFit.dispose());
  const final=await page.evaluate(()=>({disposed:window.sceneFit.state.disposed,errors:window.sceneFit.state.errors}));
  if(errors.length||final.errors.length||!final.disposed)throw Error('UI/GPU lifecycle check failed');
  const startupCleanup=await page.evaluate(async()=>{
    const {createEngine}=await import('/experiments/scene-fit/engine.js');
    const original=GPUAdapter.prototype.requestDevice;let destroys=0,caught=false;
    GPUAdapter.prototype.requestDevice=async function(...args){const device=await original.apply(this,args),destroy=device.destroy.bind(device);device.destroy=()=>{destroys++;destroy();};return device;};
    try{await createEngine({build(){throw Error('Intentional startup fixture failure');}},null);}
    catch(error){caught=error.message==='Intentional startup fixture failure';}
    finally{GPUAdapter.prototype.requestDevice=original;}
    if(!caught||destroys!==1)throw Error('Failed startup did not release its device');
    return {caught,destroyedDevices:destroys};
  });
  const files={};for(const name of ['fitted.png','heldout.png','scene-fit.webm','edited-parameters.json'])files[name]=createHash('sha256').update(await readFile(path.join(out,name))).digest('hex');
  await writeFile(path.join(out,'ui-check.json'),JSON.stringify({kind:'scene-fit-ui-demo-check',status:'passed',initial,fit,exportMatchesManualEdit:true,final,startupCleanup,files},null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({out,status:'passed',fittingRenders:fit.evaluations,exportMatchesManualEdit:true,disposed:final.disposed}));
}finally{clearTimeout(deadline);await browser.close();}
