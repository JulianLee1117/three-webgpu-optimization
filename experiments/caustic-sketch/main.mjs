import * as THREE from 'three/webgpu';
import { texture, vec2, vec3, uv } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { STLExporter } from 'three/addons/exporters/STLExporter.js';
import { createStudio } from './studio.mjs';
import { createPhotonOptics } from './gpu.mjs';
import { silhouette } from './solver.mjs';
import { sampleHeight, independentSnellTrace } from './optics.mjs';

const $ = id => document.getElementById(id);
const evidence = { ready:false, errors:[], solves:[], frames:0, rendering:'Three.js r186 WebGPU / TSL photon splats', rayCount:128**2 };
const state = { lens:null, flat:false, distance:10, view:'light', rays:false, solving:false, id:0, disposed:false };
let renderer, scene, camera, controls, studio, photons, glass, paths, environment, resizeObserver, receiverMaterial;
let worker, watchdog, debounce, renderRequest=0, lastResult=null;
const canvas=$('drawing'), ctx=canvas.getContext('2d',{willReadFrequently:true});
const undo=[]; let drawing=false, lastPoint=null;

function fail(error) {
  const message=error.message??String(error); evidence.errors.push(message);
  $('status').textContent=message; $('loading').textContent=message;
  $('loading').hidden=false; console.error(error);
}
function invalidate() {
  if (renderRequest || !renderer || state.disposed || document.hidden) return;
  renderRequest=requestAnimationFrame(()=>{
    renderRequest=0;
    try { renderer.render(scene,camera); evidence.frames++; } catch(error){ fail(error); }
  });
}
function status(text) { $('status').textContent=text; }
function maskData() {
  const rgba=ctx.getImageData(0,0,256,256).data, data=new Uint8Array(256**2);
  for(let i=0;i<data.length;i++)data[i]=rgba[i*4];
  return data;
}
function remember() {undo.push(ctx.getImageData(0,0,256,256));if(undo.length>10)undo.shift();}
function blank() {ctx.fillStyle='#000';ctx.fillRect(0,0,256,256);}
function preset(kind, fit=true) {
  remember(); const im=ctx.createImageData(256,256);
  for(let y=0;y<256;y++)for(let x=0;x<256;x++) {
    const i=(y*256+x)*4,v=silhouette(kind,(x+.5)/128-1,1-(y+.5)/128)?255:0;
    im.data[i]=im.data[i+1]=im.data[i+2]=v;im.data[i+3]=255;
  }
  ctx.putImageData(im,0,0); document.querySelectorAll('[data-preset]').forEach(b=>b.classList.toggle('active',b.dataset.preset===kind));
  if(fit)fitDrawing();
}
function lensGeometry(lens, subdivisions=96) {
  const positions=[], n=subdivisions;
  const point=(x,y,exit)=>[x,y,-(exit?sampleHeight(lens,x,y):lens.baseZ)];
  const tri=(a,b,c)=>positions.push(...a,...b,...c);
  for(let y=0;y<n;y++)for(let x=0;x<n;x++) {
    const x0=-1+2*x/n,x1=-1+2*(x+1)/n,y0=-1+2*y/n,y1=-1+2*(y+1)/n;
    const a=point(x0,y0,true),b=point(x1,y0,true),c=point(x0,y1,true),d=point(x1,y1,true);
    tri(a,c,b);tri(b,c,d);
  }
  // Flat entrance and four closed side walls. There are no hidden target meshes.
  const center=point(0,0,false);
  const edge=(x0,y0,x1,y1)=>{
    const p=point(x0,y0,false),q=point(x1,y1,false),r=point(x0,y0,true),s=point(x1,y1,true);
    tri(p,q,r);tri(q,s,r);tri(center,q,p);
  };
  for(let i=0;i<n;i++){const lo=-1+2*i/n,hi=-1+2*(i+1)/n;
    edge(hi,-1,lo,-1);edge(lo,1,hi,1);edge(-1,lo,-1,hi);edge(1,hi,1,lo);
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.computeVertexNormals();return geometry;
}
function activeLens() {
  if(!state.lens)return null;
  return state.flat?{...state.lens,height:new Float64Array(state.lens.grid**2)}:state.lens;
}
function updateOptics({geometry=true}={}) {
  if(state.disposed)return;
  const lens=activeLens();if(!lens || !photons)return;
  photons.update(lens,state.distance);photons.render();
  if(geometry){const old=glass.geometry;glass.geometry=lensGeometry(lens);old.dispose();}
  if(studio.receiverAssembly)studio.receiverAssembly.position.z=10-state.distance;
  else studio.receiverMesh.position.z=-state.distance;
  updatePaths(lens); invalidate();
}
function updatePaths(lens) {
  if(paths){paths.removeFromParent();paths.geometry.dispose();paths.material.dispose();paths=null;}
  if(!state.rays)return;
  const r=independentSnellTrace(lens,{resolution:8,offsetX:.5,offsetY:.5,receiverDistance:state.distance});
  const p=[],colors=[];
  function line(a,b,alpha){p.push(...a,...b);for(let j=0;j<2;j++)colors.push(.85*alpha,.67*alpha,.36*alpha);}
  for(let y=0;y<8;y++)for(let x=0;x<8;x++) {
    const ax=-1+2*(x+.5)/8,ay=-1+2*(y+.5)/8,i=y*8+x,h=sampleHeight(lens,ax,ay);
    line([ax,ay+1.6,4],[ax,ay+1.6,-h],.34);
    line([ax,ay+1.6,-h],[r.hits[2*i],r.hits[2*i+1]+1.6,-state.distance],.65);
  }
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(p,3));geo.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  paths=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.28,depthWrite:false,blending:THREE.AdditiveBlending}));scene.add(paths);
}
function stopWorker() {worker?.terminate();worker=null;clearTimeout(watchdog);clearTimeout(debounce);}
function fitDrawing() {
  if(state.disposed)return;
  const mask=maskData(); if(!mask.some(v=>v>20)){status('Make a mark to shape the glass.');return;}
  stopWorker();state.solving=true;state.flat=false;state.distance=10;state.id++;
  $('flatten').setAttribute('aria-pressed','false');$('flatten').textContent='Flatten the glass';$('distance').value=10;$('distance-label').textContent='In focus';
  if(state.view==='light'&&camera)setView('light');
  status('Shaping glass…');$('shape').textContent='Shaping glass…';
  const id=state.id;worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
  watchdog=setTimeout(()=>{stopWorker();state.solving=false;$('shape').innerHTML='Try again <span>↗</span>';status('That shape took too long. Try a bolder, simpler mark.');},16000);
  worker.onmessage=({data})=>{
    if(data.id!==state.id||state.disposed)return;
    if(data.type==='progress') {
      state.lens={height:data.height,grid:data.grid,ior:1.49,baseZ:-.5,extent:1,distance:10};
      updateOptics();status(`Shaping glass · ${Math.round(data.iteration/4)}%`);
    }else if(data.type==='result'){
      stopWorker();state.lens=data.lens;state.solving=false;lastResult=data;
      evidence.solves.push({id,milliseconds:data.pipelineMilliseconds,initialLoss:data.initialLoss,finalLoss:data.finalLoss,maxSlope:data.maxSlope});
      updateOptics();status(`Your mark, held in glass · ${(data.pipelineMilliseconds/1000).toFixed(1)}s`);
      $('shape').innerHTML='Shape the glass <span>↗</span>';$('loading').hidden=true;evidence.ready=true;
    }else if(data.type==='error') {stopWorker();state.solving=false;status(data.message);$('shape').innerHTML='Shape the glass <span>↗</span>';}
  };
  worker.onerror=error=>{stopWorker();state.solving=false;fail(error);};
  worker.postMessage({id,mask,width:256,height:256,resolution:48,grid:25,iterations:400},[mask.buffer]);
}
function setDistance(value) {
  if(state.disposed)return;
  state.distance=Number(value);$('distance').value=state.distance;
  $('distance-label').textContent=Math.abs(state.distance-10)<.05?'In focus':state.distance<10?'Before the focus':'Beyond the focus';
  updateOptics({geometry:false});if(state.view==='light')setView('light');
}
function setView(view) {
  state.view=view;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  const a=$('stage').clientWidth/$('stage').clientHeight;
  if(view==='light') {
    camera.position.set(.15,2.4,-state.distance+5.8);controls.target.set(0,2.4,-state.distance);
    camera.fov=a<1?62:55;
  }else if(view==='glass'){
    camera.position.set(3.1,3.3,-4.7);controls.target.set(0,1.7,0);camera.fov=44;
  }else{
    camera.position.set(9.6,6.8,12.4);controls.target.set(0,1,-3.8);camera.fov=42;
  }
  camera.updateProjectionMatrix();controls.update();invalidate();
}
function exportGlass() {
  if(!state.lens)return;
  const mesh=new THREE.Mesh(lensGeometry(activeLens(),192));
  const stl=new STLExporter().parse(mesh,{binary:true});mesh.geometry.dispose();
  const blob=new Blob([stl],{type:'application/octet-stream'}),url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='sunprint-experimental-glass.stl';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  status('Glass saved · approximate surface, arbitrary length units');
}
function dispose() {
  if(state.disposed)return;state.disposed=true;stopWorker();cancelAnimationFrame(renderRequest);resizeObserver?.disconnect();controls?.dispose();
  paths?.geometry.dispose();paths?.material.dispose();glass?.geometry.dispose();glass?.material.dispose();
  studio?.dispose();receiverMaterial?.dispose();photons?.dispose();environment?.dispose();renderer?.dispose();
}

async function init() {
  if(!navigator.gpu)throw Error('Sunprint needs WebGPU. Open it in a current Chrome or Edge browser.');
  renderer=new THREE.WebGPURenderer({antialias:true,powerPreference:'low-power'});await renderer.init();
  if(!renderer.backend.isWebGPUBackend)throw Error('WebGPU is required for the light simulation.');
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFShadowMap;
  $('stage').append(renderer.domElement);
  scene=new THREE.Scene();scene.background=new THREE.Color(0x102023);scene.fog=new THREE.FogExp2(0x102023,.018);
  camera=new THREE.PerspectiveCamera(43,1,.05,100);
  controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.minDistance=2;controls.maxDistance=35;controls.maxPolarAngle=Math.PI*.85;controls.addEventListener('change',invalidate);
  const envScene=new RoomEnvironment(),pmrem=new THREE.PMREMGenerator(renderer);
  environment=await pmrem.fromSceneAsync(envScene,.06);scene.environment=environment.texture;scene.environmentIntensity=.45;envScene.dispose();pmrem.dispose();
  studio=createStudio(scene);photons=createPhotonOptics(renderer,{grid:25,rayResolution:128,mapResolution:512});
  // WebGPU render-target rows start at the top; the physical plane's UV y grows upward.
  const receiverUV=vec2(uv().x,uv().y.oneMinus());
  receiverMaterial=new THREE.MeshBasicNodeMaterial({colorNode:texture(photons.texture,receiverUV).rgb.mul(vec3(1,.66,.24)).mul(.5).add(vec3(.008,.014,.017)),fog:false});
  studio.receiverMesh.material=receiverMaterial;
  const glassMaterial=new THREE.MeshPhysicalNodeMaterial({color:0xcce9e1,metalness:0,roughness:.075,transmission:1,ior:1.49,thickness:.5,side:THREE.DoubleSide,clearcoat:1,clearcoatRoughness:.07});
  glass=new THREE.Mesh(new THREE.BufferGeometry(),glassMaterial);glass.position.y=1.6;glass.name='Actual optimized glass heightfield';scene.add(glass);
  const resize=()=>{const w=$('stage').clientWidth,h=$('stage').clientHeight;renderer.setSize(w,h);camera.aspect=w/h;if(state.view==='light')camera.fov=w/h<1?62:55;camera.updateProjectionMatrix();invalidate();};
  resizeObserver=new ResizeObserver(resize);resizeObserver.observe($('stage'));resize();setView('light');
  preset('heart');
}

canvas.addEventListener('pointerdown',e=>{remember();drawing=true;lastPoint=null;canvas.setPointerCapture(e.pointerId);stroke(e);document.querySelectorAll('[data-preset]').forEach(b=>b.classList.remove('active'));});
function stroke(e) {
  if(!drawing)return;const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*256,y=(e.clientY-r.top)/r.height*256;
  ctx.strokeStyle='#fff';ctx.fillStyle='#fff';ctx.lineWidth=Number($('brush').value);ctx.lineCap=ctx.lineJoin='round';
  if(lastPoint){ctx.beginPath();ctx.moveTo(...lastPoint);ctx.lineTo(x,y);ctx.stroke();}else{ctx.beginPath();ctx.arc(x,y,ctx.lineWidth/2,0,Math.PI*2);ctx.fill();}lastPoint=[x,y];
}
canvas.addEventListener('pointermove',stroke);
canvas.addEventListener('pointerup',()=>{if(!drawing)return;drawing=false;lastPoint=null;clearTimeout(debounce);debounce=setTimeout(fitDrawing,450);});
canvas.addEventListener('pointercancel',()=>{drawing=false;lastPoint=null;});
$('clear').onclick=()=>{remember();blank();status('Draw a new mark, then let go.');};
$('undo').onclick=()=>{const im=undo.pop();if(im){ctx.putImageData(im,0,0);fitDrawing();}};
$('shape').onclick=fitDrawing;
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>preset(b.dataset.preset));
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
function typeWord(){const text=$('word').value.trim().toUpperCase();if(!text)return;remember();blank();ctx.fillStyle='#fff';ctx.font=`bold ${text.length===1?190:text.length===2?125:89}px Arial`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,128,137,228);fitDrawing();}
$('type').onclick=typeWord;$('word').onkeydown=e=>{if(e.key==='Enter')typeWord();};
$('distance').oninput=e=>setDistance(e.target.value);$('refocus').onclick=()=>setDistance(10);
$('flatten').onclick=()=>{state.flat=!state.flat;$('flatten').setAttribute('aria-pressed',String(state.flat));$('flatten').textContent=state.flat?'Restore your glass':'Flatten the glass';updateOptics();status(state.flat?'Flat glass spreads the light evenly. Your drawing has not changed.':'Your shape returns with the curved surface.');};
$('rays').onclick=()=>{state.rays=!state.rays;$('rays').setAttribute('aria-pressed',String(state.rays));$('rays').textContent=state.rays?'Hide light paths':'Show light paths';if(state.rays)setView('studio');if(state.lens)updatePaths(activeLens());invalidate();};
$('export').onclick=exportGlass;$('about-toggle').onclick=()=>$('about').showModal();$('about-close').onclick=()=>$('about').close();
document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(renderRequest);renderRequest=0;}else invalidate();});window.addEventListener('pagehide',dispose);
window.sunprint={evidence,state,preset,fitDrawing,setDistance,setView,dispose,
  async verify(){const lens=activeLens(),probe=await photons.probe(lens,state.distance);const reference=independentSnellTrace({...lens,height:Float64Array.from(probe.height)},{resolution:probe.resolution,offsetX:probe.offsetX,offsetY:probe.offsetY,receiverDistance:state.distance});let maxError=0;for(let i=0;i<reference.hits.length;i++)maxError=Math.max(maxError,Math.abs(reference.hits[i]-probe.hits[i]));return {maxError,gpuHits:Array.from(probe.hits),referenceHits:Array.from(reference.hits),energy:reference.energy,lost:reference.lost,grid:lens.grid,distance:state.distance};},
  async rasterEvidence(){
    const raster=await photons.readIrradiance(),bins=32,gpuBins=new Float64Array(bins*bins),cpuBins=new Float64Array(bins*bins);
    const lens=activeLens(),reference=independentSnellTrace({...lens,height:Float32Array.from(lens.height)},{resolution:128,offsetX:.5,offsetY:.5,receiverDistance:state.distance});
    const area=(2*raster.halfExtent)**2/(raster.width*raster.height);let flux=0,finite=true;
    for(let y=0;y<raster.height;y++)for(let x=0;x<raster.width;x++){
      const irradiance=raster.pixels[(y*raster.width+x)*4];finite&&=Number.isFinite(irradiance)&&irradiance>=0;
      const mass=irradiance*area;flux+=mass;gpuBins[Math.floor(y/raster.height*bins)*bins+Math.floor(x/raster.width*bins)]+=mass;
    }
    let cpuFlux=0;
    for(let i=0;i<reference.weights.length;i++){
      const x=Math.floor((reference.hits[2*i]/(2*raster.halfExtent)+.5)*bins),y=Math.floor((.5-reference.hits[2*i+1]/(2*raster.halfExtent))*bins);
      if(x>=0&&y>=0&&x<bins&&y<bins){const mass=reference.weights[i]*4/reference.weights.length;cpuBins[y*bins+x]+=mass;cpuFlux+=mass;}
    }
    let dot=0,a=0,b=0,flippedDot=0;
    for(let y=0;y<bins;y++)for(let x=0;x<bins;x++){const i=y*bins+x;dot+=gpuBins[i]*cpuBins[i];a+=gpuBins[i]**2;b+=cpuBins[i]**2;flippedDot+=gpuBins[i]*cpuBins[(bins-1-y)*bins+x];}
    return {finite,flux,cpuFlux,relativeFluxError:Math.abs(flux-cpuFlux)/cpuFlux,cosine:dot/Math.sqrt(a*b),flippedCosine:flippedDot/Math.sqrt(a*b),bins,width:raster.width,height:raster.height,rowOrigin:'top (+analytical y)',gpuBins:Array.from(gpuBins),cpuBins:Array.from(cpuBins)};
  },
  surface(){return Array.from(activeLens().height);},
  summary(){return {state:{distance:state.distance,flat:state.flat,view:state.view,solving:state.solving},evidence,result:lastResult?{initialLoss:lastResult.initialLoss,finalLoss:lastResult.finalLoss,milliseconds:lastResult.pipelineMilliseconds}:null};}
};
init().catch(fail);
