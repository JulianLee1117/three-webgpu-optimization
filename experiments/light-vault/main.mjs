import * as THREE from 'three/webgpu';
import {textureLoad,uv,vec2,vec3,ivec2,floor,float} from 'three/tsl';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {createWaveOptics} from './gpu.mjs';
import {createVaultScene} from './scene.mjs';
import {fft2,sourceFromPhase,propagate as cpuPropagate,transfer,intensity} from './wave.mjs';
import {OPTICS,presetAmplitude,aperture,maskAmplitude,matchImage} from './targets.mjs';
import {serializeSeal,parseSeal} from './seal.mjs';

const $=id=>document.getElementById(id),clamp=(x,a,b)=>Math.min(b,Math.max(a,x));
const state={ready:false,distance:.021,found:[],opening:0,won:false,capturing:false,fitting:false,changing:false,disposed:false,id:0,epoch:0};
const evidence={frames:0,propagations:0,errors:[],captures:[],fits:[],sourceUpdates:0};
let renderer,scene,camera,art,waves,environment,hudScene,hudCamera,receiverMaterial,phaseTexture;
let phase,source,spectrum,targets=[{amplitude:presetAmplitude('heart')},{amplitude:presetAmplitude('letterA')}];
let frameRequest=0,opticsDirty=true,resizeObserver,worker,watchdog,lastOpened=0;
let volumeWorker,volumeWatchdog,volumePoints,volumeEnds,volumeRevision=0;
let readbackTail=Promise.resolve(),readbackBusy=false;
let interactionMeasurement=null;
let drag=null;

function fail(error){const message=error.message??String(error);evidence.errors.push(message);$('loading').hidden=false;$('loading-text').textContent=message;console.error(error);}
function requestRender(){if(frameRequest||!renderer||state.disposed||document.hidden)return;frameRequest=requestAnimationFrame(render);}
function receiverX(){return -1.7+(state.distance-.004)/.034*4.9;}
function render(now){
  frameRequest=0;if(!state.ready||state.disposed)return;
  try{
    if(opticsDirty&&!readbackBusy){waves.propagate(state.distance);opticsDirty=false;evidence.propagations++;}
    if(state.won&&state.opening<1){state.opening=clamp((now-lastOpened)/1400,0,1);requestRender();}
    art.update({receiverX:receiverX(),opening:state.opening});
    if(volumePoints){
      // The card absorbs the light: include only sampled planes at or before it.
      const slice=clamp(Math.floor((state.distance-.004)/.034*31+1e-8),0,31);
      volumePoints.geometry.setDrawRange(0,volumeEnds[slice]);
    }
    const box=$('viewport').getBoundingClientRect(),glass=$('glass-window').getBoundingClientRect();
    renderer.setScissorTest(false);renderer.setViewport(0,0,box.width,box.height);renderer.autoClear=true;renderer.render(scene,camera);
    renderer.setViewport(glass.left-box.left,glass.top-box.top,glass.width,glass.height);
    renderer.setScissor(glass.left-box.left,glass.top-box.top,glass.width,glass.height);renderer.setScissorTest(true);renderer.render(hudScene,hudCamera);
    renderer.setScissorTest(false);renderer.setViewport(0,0,box.width,box.height);evidence.frames++;
    if(interactionMeasurement?.input&&interactionMeasurement.input.id!==interactionMeasurement.lastSubmitted){
      const measurement=interactionMeasurement,input={...measurement.input};measurement.lastSubmitted=input.id;
      renderer.backend.device.queue.onSubmittedWorkDone().then(()=>measurement.samples.push({id:input.id,distance:input.distance,inputToGpuCompletionMs:performance.now()-input.time})).catch(()=>{});
    }
  }catch(error){fail(error);}
}
function setDistance(distance){
  if(!state.ready||state.disposed)return;
  const next=clamp(Number(distance),...OPTICS.range);if(!Number.isFinite(next))return;
  state.distance=next;opticsDirty=true;$('card').setAttribute('aria-valuenow',(1000*next).toFixed(1));
  if(interactionMeasurement)interactionMeasurement.input={id:++interactionMeasurement.inputs,distance:next,time:performance.now()};
  $('card').setAttribute('aria-valuetext',`Glass ${Math.round(next*1000)} millimetres into the light`);
  $('depth').value=next*1000;
  if(!state.won)$('feedback').textContent=state.found.length?'One found. Search for the other picture.':'When a picture becomes clear, capture it.';
  requestRender();
}
function makePhaseTexture(){
  const data=new Uint8Array(OPTICS.n**2*4),amp=aperture();
  for(let i=0;i<phase.length;i++){
    const v=(phase[i]/(2*Math.PI)+1)%1,j=i*4;
    data[j]=amp[i]?(30+150*v):8;data[j+1]=amp[i]?(55+120*(1-v)):12;data[j+2]=amp[i]?(75+100*v):17;data[j+3]=255;
  }
  phaseTexture?.dispose();phaseTexture=new THREE.DataTexture(data,OPTICS.n,OPTICS.n,THREE.RGBAFormat);phaseTexture.colorSpace=THREE.SRGBColorSpace;phaseTexture.offset.set(.29,.29);phaseTexture.repeat.set(.42,.42);phaseTexture.needsUpdate=true;
  art.phaseMesh.material.map=phaseTexture;art.phaseMesh.material.needsUpdate=true;
}
function withWaveLock(operation){
  const result=readbackTail.then(async()=>{
    if(state.disposed)throw Error('The experiment is closed.');readbackBusy=true;
    try{return await operation();}finally{readbackBusy=false;requestRender();}
  });
  readbackTail=result.catch(()=>{});return result;
}
async function installPhase(value,newTargets=targets){
  state.epoch++;state.changing=true;
  try{await withWaveLock(()=>{
    phase=Float64Array.from(value);targets=newTargets;
    source=sourceFromPhase(phase,aperture());spectrum=fft2(source.slice(),OPTICS.n);
    waves.updateSource(spectrum);evidence.sourceUpdates++;makePhaseTexture();makeVolume();opticsDirty=true;
  });}finally{state.changing=false;}
}
function clearVolume(){
  volumeWorker?.terminate();volumeWorker=null;clearTimeout(volumeWatchdog);
  if(volumePoints){scene.remove(volumePoints);volumePoints.geometry.dispose();volumePoints.material.dispose();volumePoints=null;}
}
function makeVolume(){
  clearVolume();const id=++volumeRevision,copy=phase.slice();
  volumeWorker=new Worker(new URL('./volume-worker.mjs',import.meta.url),{type:'module'});
  const stop=()=>{volumeWorker?.terminate();volumeWorker=null;clearTimeout(volumeWatchdog);};
  // This preview is optional. Failure never changes the actual GPU receiver field.
  volumeWatchdog=setTimeout(()=>{stop();evidence.volume={status:'timed out'};},5000);
  volumeWorker.onmessage=({data})=>{
    if(state.disposed||data.id!==volumeRevision)return;stop();
    if(data.type==='error'){evidence.volume={status:'unavailable',message:data.message};return;}
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(data.positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(data.colors,3));
    const material=new THREE.PointsNodeMaterial({vertexColors:true,size:.024,transparent:true,opacity:.16,depthWrite:false,blending:THREE.AdditiveBlending,toneMapped:false});
    volumePoints=new THREE.Points(geometry,material);volumeEnds=data.ends;volumePoints.frustumCulled=false;scene.add(volumePoints);
    evidence.volume={status:'ready',points:data.positions.length/3,milliseconds:data.milliseconds,slices:data.ends.length,sourceRevision:id};requestRender();
  };
  volumeWorker.onerror=()=>{stop();evidence.volume={status:'unavailable'};};
  volumeWorker.postMessage({id,phase:copy},[copy.buffer]);
}
function reset(){
  if(state.disposed)return;state.epoch++;
  state.found=[];state.won=false;state.opening=0;$('won').hidden=true;
  $('chapter').textContent='01 / FIND WHAT THE LIGHT HOLDS';$('headline').innerHTML='Two pictures.<br>One secret.';
  $('instructions').innerHTML='Slide the glass through the light.<br>Capture a picture when it becomes clear.';
  $('feedback').textContent='Move slowly. A picture is hiding nearby.';
  for(let i=0;i<2;i++){$(`memory-${i}`).classList.remove('found');$(`memory-${i}`).querySelector('b').textContent='UNDISCOVERED';}
  setDistance(.021);
}
function drawIntensity(canvas,pixels){
  const ctx=canvas.getContext('2d'),image=ctx.createImageData(canvas.width,canvas.height),n=OPTICS.n;
  for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){
    const sx=Math.floor(n*.25+(x+.5)/canvas.width*n*.5),sy=Math.floor(n*.25+(y+.5)/canvas.height*n*.5),v=1-Math.exp(-pixels[sy*n+sx]*.34),j=(y*canvas.width+x)*4;
    image.data[j]=255*v;image.data[j+1]=226*v;image.data[j+2]=155*v;image.data[j+3]=255;
  }
  ctx.putImageData(image,0,0);
}
async function capture(){
  if(!state.ready||state.disposed||state.capturing||state.fitting||state.changing)return;
  state.capturing=true;$('capture').disabled=true;
  const epoch=state.epoch,at=state.distance,snapshotTargets=targets;
  try{
    const pixels=await withWaveLock(async()=>{
      if(epoch!==state.epoch)return null;
      waves.propagate(at);evidence.propagations++;opticsDirty=true;return await waves.readIntensity();
    });
    if(state.disposed||epoch!==state.epoch||!pixels)return;
    const match=matchImage(pixels,snapshotTargets);evidence.captures.push({distance:at,...match});
    if(!match.clear){$('feedback').textContent='Still interference. Slide gently until a picture resolves.';return;}
    if(state.found.includes(match.best)){$('feedback').textContent='You have this one. Find the other picture in the light.';return;}
    const slot=state.found.length;state.found.push(match.best);drawIntensity($(`memory-${slot}`).querySelector('canvas'),pixels);
    $(`memory-${slot}`).classList.add('found');$(`memory-${slot}`).querySelector('b').textContent='CAPTURED IN LIGHT';
    if(state.found.length===2){
      state.won=true;lastOpened=performance.now();$('won').hidden=false;$('chapter').textContent='02 / THE LIGHT REMEMBERS';$('headline').innerHTML='Both were<br>there all along.';
      $('instructions').innerHTML='One phase pattern held two different pictures.<br>You found them by moving through its light.';
      $('feedback').textContent='Both impressions found. The vault is open.';requestRender();
    }else $('feedback').textContent='One found. There is another hidden at a different depth.';
  }catch(error){if(!state.disposed)fail(error);}finally{state.capturing=false;if(!state.disposed)$('capture').disabled=false;}
}
function stopWorker(){state.id++;worker?.terminate();worker=null;clearTimeout(watchdog);state.fitting=false;$('fit').disabled=false;}
async function fit(){
  if(state.disposed||state.changing)return;stopWorker();const masks=[0,1].map(i=>{const rgba=$(`draw-${i}`).getContext('2d').getImageData(0,0,128,128).data;return Uint8Array.from({length:128**2},(_,j)=>rgba[j*4]);});
  try{masks.forEach(m=>maskAmplitude(m));}catch(error){$('fit-status').textContent=error.message;return;}
  state.fitting=true;$('fit').disabled=true;$('fit-status').textContent='Encoding both memories…';const id=++state.id;
  worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
  watchdog=setTimeout(()=>{stopWorker();$('fit-status').textContent='This fit took too long. Try two simpler marks.';},16000);
  worker.onmessage=async({data})=>{
    if(state.disposed||data.id!==state.id)return;
    if(data.type==='progress'){$('fit-status').textContent=`Encoding · ${Math.round(data.iteration/80*100)}%`;return;}
    stopWorker();
    if(data.type==='error'){$('fit-status').textContent=data.message;return;}
    const field=sourceFromPhase(data.solution.phase,data.solution.aperture);
    const quality=OPTICS.distances.map(z=>matchImage(intensity(cpuPropagate(field,OPTICS.n,transfer(OPTICS.n,OPTICS.pitch,OPTICS.wavelength,z))),data.targets));
    evidence.fits.push({milliseconds:data.solution.milliseconds,quality,history:data.solution.history});
    if(!quality.every((q,i)=>q.clear&&q.best===i)){$('fit-status').textContent='These marks overlap too much in light. Try larger, more different shapes.';return;}
    try{await installPhase(data.solution.phase,data.targets);}catch(error){if(!state.disposed)fail(error);return;}
    if(state.disposed)return;reset();$('maker').close();$('feedback').textContent='Your two memories are encoded. Find them in the light.';$('fit-status').textContent=`Encoded in ${(data.solution.milliseconds/1000).toFixed(1)}s.`;
  };
  worker.onerror=error=>{stopWorker();$('fit-status').textContent=error.message||'The encoder could not load. Reload the experiment and try again.';};
  worker.postMessage({id,masks},masks.map(m=>m.buffer));
}
function exportPhase(){
  if(!phase||state.changing)return;
  const blob=new Blob([serializeSeal(phase,targets)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='light-vault-seal.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function importSeal(file){
  if(!file||state.disposed||state.changing)return;stopWorker();state.changing=true;
  try{
    if(file.size>3500000)throw Error('This seal is too large. Choose a Light Vault seal smaller than 3.5 MB.');
    $('fit-status').textContent='Checking the light stored in this seal…';
    const decoded=parseSeal(await file.text());if(state.disposed)return;
    await installPhase(decoded.phase,decoded.targets);if(state.disposed)return;
    evidence.imports=(evidence.imports??0)+1;reset();$('maker').close();$('feedback').textContent='Seal loaded. Discover its two pictures in the light.';
  }catch(error){if(!state.disposed)$('fit-status').textContent=error.message;}
  finally{state.changing=false;$('import-file').value='';}
}
function dispose(){
  if(state.disposed)return readbackTail;state.disposed=true;state.epoch++;stopWorker();cancelAnimationFrame(frameRequest);resizeObserver?.disconnect();clearVolume();
  return readbackTail.then(()=>{art?.dispose();phaseTexture?.dispose();receiverMaterial?.dispose();hudScene?.children.forEach(o=>o.geometry?.dispose());waves?.dispose();environment?.dispose();renderer?.dispose();});
}

async function init(){
  if(!navigator.gpu)throw Error('This experiment needs WebGPU. Try a current Chrome or Edge browser.');
  renderer=new THREE.WebGPURenderer({antialias:true,powerPreference:'low-power'});await renderer.init();
  if(!renderer.backend.isWebGPUBackend)throw Error('A WebGPU backend is required for the wave simulation.');
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFShadowMap;$('viewport').append(renderer.domElement);
  scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0d10);scene.fog=new THREE.FogExp2(0x0b0d10,.023);
  camera=new THREE.PerspectiveCamera(43,1,.03,100);camera.position.set(6.5,4.6,8);camera.lookAt(-.8,1.3,0);
  const env=new RoomEnvironment(),pmrem=new THREE.PMREMGenerator(renderer);environment=pmrem.fromScene(env,.055);scene.environment=environment.texture;scene.environmentIntensity=.38;env.dispose();pmrem.dispose();
  art=createVaultScene(scene);await art.readyPromise;
  waves=createWaveOptics(renderer,OPTICS);
  const pixel=floor(vec2(uv().x,uv().y.oneMinus()).mul(.5).add(.25).mul(OPTICS.n));
  // Fixed 3×3 reconstruction smooths the sampled display only. Matching reads the unfiltered GPU field.
  let radiance=float(0);
  for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++)radiance=radiance.add(textureLoad(waves.texture,ivec2(pixel.add(vec2(x,y)))).r.mul((x===0?2:1)*(y===0?2:1)/16));
  const light=radiance.mul(.34).negate().exp().oneMinus();
  receiverMaterial=new THREE.MeshBasicNodeMaterial({colorNode:vec3(1,.78,.36).mul(light).add(vec3(.006,.010,.017)),side:THREE.DoubleSide,toneMapped:false,fog:false});
  art.receiverMesh.material=receiverMaterial;
  hudScene=new THREE.Scene();hudScene.background=new THREE.Color(0x090d13);hudCamera=new THREE.OrthographicCamera(-1,1,1,-1,.1,2);hudCamera.position.z=1;
  hudScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),receiverMaterial));
  const raw=await(await fetch('./assets/initial-phase.bin')).arrayBuffer();if(raw.byteLength!==OPTICS.n**2*4)throw Error('Incomplete phase plate');
  await installPhase(new Float32Array(raw));if(state.disposed)return;state.ready=true;$('loading').hidden=true;$('capture').disabled=false;
  const resize=()=>{if(state.disposed)return;const w=$('viewport').clientWidth,h=$('viewport').clientHeight;renderer.setSize(w,h);camera.aspect=w/h;
    if(w<700){camera.position.set(6.5,5.7,9);camera.lookAt(-2.2,1.6,0);camera.fov=55;}else{camera.position.set(6.5,4.6,8);camera.lookAt(-.8,1.3,0);camera.fov=43;}camera.updateProjectionMatrix();requestRender();};
  resizeObserver=new ResizeObserver(resize);resizeObserver.observe($('viewport'));resize();requestRender();
}

$('card').onpointerdown=e=>{if(!state.ready)return;e.preventDefault();drag={x:e.clientX,start:state.distance};$('card').setPointerCapture(e.pointerId);$('card').classList.add('dragging');};
$('card').onpointermove=e=>{if(drag)setDistance(drag.start+(e.clientX-drag.x)*.00007);};
function endDrag(){drag=null;$('card').classList.remove('dragging');}$('card').onpointerup=endDrag;$('card').onpointercancel=endDrag;
$('card').addEventListener('wheel',e=>{e.preventDefault();setDistance(state.distance+clamp(e.deltaY,-100,100)*.000008);},{passive:false});
$('card').onkeydown=e=>{const directions={ArrowLeft:-1,ArrowDown:-1,ArrowRight:1,ArrowUp:1};if(directions[e.key]){e.preventDefault();setDistance(state.distance+directions[e.key]*(e.shiftKey?.001:.00025));}};
$('capture').onclick=capture;$('reset').onclick=reset;
$('depth').oninput=e=>setDistance(Number(e.target.value)/1000);
$('maker').addEventListener('cancel',stopWorker);
for(const id of ['create','create-won'])$(id).onclick=()=>$('maker').showModal();$('maker-close').onclick=()=>{stopWorker();$('maker').close();};$('keep-exploring').onclick=()=>$('won').hidden=true;
$('fit').onclick=fit;$('export').onclick=exportPhase;$('about-open').onclick=()=>$('about').showModal();$('about-close').onclick=()=>$('about').close();
$('import-button').onclick=()=>$('import-file').click();$('import-file').onchange=e=>importSeal(e.target.files[0]);
for(let i=0;i<2;i++){
  const canvas=$(`draw-${i}`),ctx=canvas.getContext('2d',{willReadFrequently:true});let down=false,last=null;
  function clear(){ctx.fillStyle='#000';ctx.fillRect(0,0,128,128);}
  function letters(){clear();ctx.fillStyle='#fff';ctx.font=`bold ${$(`text-${i}`).value.length===1?94:65}px Arial`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText($(`text-${i}`).value.toUpperCase(),64,69,110);}
  letters();document.querySelector(`[data-text="${i}"]`).onclick=letters;document.querySelector(`[data-clear="${i}"]`).onclick=clear;
  function stroke(e){if(!down)return;const r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)/r.width*128,(e.clientY-r.top)/r.height*128];ctx.strokeStyle=ctx.fillStyle='#fff';ctx.lineCap=ctx.lineJoin='round';ctx.lineWidth=10;if(last){ctx.beginPath();ctx.moveTo(...last);ctx.lineTo(...p);ctx.stroke();}else{ctx.beginPath();ctx.arc(...p,5,0,Math.PI*2);ctx.fill();}last=p;}
  canvas.onpointerdown=e=>{down=true;last=null;canvas.setPointerCapture(e.pointerId);stroke(e);};canvas.onpointermove=stroke;canvas.onpointerup=canvas.onpointercancel=()=>{down=false;last=null;};
}
document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frameRequest);frameRequest=0;}else requestRender();});window.addEventListener('pagehide',dispose);
window.lightVault={state,evidence,setDistance,capture,reset,dispose,
  phase(){return Array.from(phase);},
  beginInteraction(){interactionMeasurement={started:performance.now(),inputs:0,lastSubmitted:0,samples:[]};},
  async endInteraction(){
    await renderer.backend.device.queue.onSubmittedWorkDone();const measurement=interactionMeasurement;interactionMeasurement=null;
    const times=measurement.samples.map(s=>s.inputToGpuCompletionMs).sort((a,b)=>a-b);
    return {...measurement,elapsedMs:performance.now()-measurement.started,p50:times[Math.floor(times.length*.5)],p95:times[Math.min(times.length-1,Math.floor(times.length*.95))],meaning:'Input event to matching render GPU submission completion; not display presentation time.'};
  },
  verify(){return withWaveLock(async()=>{
    const at=state.distance;waves.propagate(at);opticsDirty=true;const actual=await waves.readComplex(),gpu=actual.field??actual,cpu=cpuPropagate(source,OPTICS.n,transfer(OPTICS.n,OPTICS.pitch,OPTICS.wavelength,at));let error=0,power=0,maxError=0;
    for(let i=0;i<cpu.length;i++){error+=(gpu[i]-cpu[i])**2;power+=cpu[i]**2;maxError=Math.max(maxError,Math.abs(gpu[i]-cpu[i]));}
    const read=await waves.readIntensity(),pixels=read.pixels??read,reference=intensity(cpu);let textureError=0,texturePower=0;for(let i=0;i<pixels.length;i++){textureError+=(pixels[i]-reference[i])**2;texturePower+=reference[i]**2;}
    return{relativeComplexError:Math.sqrt(error/power),maxError,relativeTextureError:Math.sqrt(textureError/texturePower),gpuEnergy:pixels.reduce((a,b)=>a+b,0),cpuEnergy:power,match:matchImage(pixels,targets),distance:at};
  });},
  summary(){return {state:{...state},evidence:{...evidence}};}
};
init().catch(fail);
