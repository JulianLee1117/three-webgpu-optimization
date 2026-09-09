import * as THREE from 'three/webgpu';
import {uv,uniform} from 'three/tsl';
import {compileFootprint} from './compiler.js';
import {KINDS,makeFixture} from './fixtures.mjs';

const $=s=>document.querySelector(s),elements={start:$('#start'),play:$('#play'),stop:$('#stop'),pattern:$('#pattern'),density:$('#density'),exposure:$('#exposure')};
const titles={'weave':'Correlated weave','energy':'Energy squares','interference':'Three-wave interference'};
for(const kind of KINDS){const option=document.createElement('option');option.value=kind;option.textContent=titles[kind]??kind.replaceAll('-',' ');elements.pattern.append(option);}
let current=null,opening=false,playing=false,raf=0,epoch=0,last=0,generation=0;
const setStatus=message=>$('#status').textContent=message;
const labels=()=>{$('#density-value').textContent=`${Number(elements.density.value).toFixed(2)}×`;$('#exposure-value').textContent=`${elements.exposure.value} ms`;};labels();
function buttonState(){elements.start.disabled=opening||!!current;elements.play.disabled=!current;elements.stop.disabled=!current&&!opening;elements.pattern.disabled=opening;}
function pause(){playing=false;cancelAnimationFrame(raf);raf=0;elements.play.textContent='Preview 12 seconds';}
function close(message='GPU released. Open the demo to continue.'){
  generation++;pause();const old=current;current=null;
  if(old){old.disposed=true;old.observer.disconnect();for(const resource of old.resources)resource.dispose?.();old.renderer.dispose();old.device.destroy();old.renderer.domElement.remove();}
  $('#empty').hidden=false;$('#empty').style.display='grid';opening=false;buttonState();setStatus(message);
}
function fail(error){close(`Stopped: ${error.message??error}`);}
function surface(){
  const geometry=new THREE.PlaneGeometry(4.6,4.2,80,64),p=geometry.attributes.position;
  for(let i=0;i<p.count;i++){const x=p.getX(i),y=p.getY(i);p.setXYZ(i,x,y,.43*Math.sin(x*2.4+y*.65)+.13*Math.cos(y*3));}
  geometry.computeVertexNormals();return geometry;
}
function configure(){
  if(!current)return;const c=current;
  const old=[c.raw,c.filtered].filter(Boolean);for(const m of old){m.dispose();c.resources.delete(m);}
  const root=makeFixture(elements.pattern.value,{uv:c.uv,time:c.time,frequency:c.frequency});
  const compiled=compileFootprint(root,{time:c.time,shutter:c.shutter,inputs:[c.uv,c.frequency]});
  const options={roughness:.58,metalness:.12,side:THREE.DoubleSide};
  c.raw=new THREE.MeshStandardNodeMaterial(options);c.raw.colorNode=root;
  c.filtered=new THREE.MeshStandardNodeMaterial(options);c.filtered.colorNode=compiled.node;
  c.resources.add(c.raw);c.resources.add(c.filtered);c.stats=compiled.stats;
  $('#terms').textContent=`${compiled.stats.realHarmonics} real harmonics.`;
  render();
}
function render(){
  if(!current)return;const c=current;
  const width=Math.min(1400,Math.floor($('#view').clientWidth)),height=Math.min(600,Math.floor($('#view').clientHeight));if(width<4||height<2)return;
  const left=Math.floor(width/2),right=width-left;
  if(c.width!==width||c.height!==height){c.renderer.setSize(width,height,false);c.width=width;c.height=height;}
  c.frequency.value=Number(elements.density.value);c.shutter.value=Number(elements.exposure.value)/1000;
  c.mesh.rotation.set(-.10,Math.sin(c.time.value*.35)*.18-.22,.06);
  c.renderer.setScissorTest(true);
  for(const [x,w,material]of [[0,left,c.raw],[left,right,c.filtered]]){
    c.camera.aspect=w/height;c.camera.position.z=7.6*Math.max(1,.85/c.camera.aspect);c.camera.updateProjectionMatrix();c.mesh.material=material;
    c.renderer.setViewport(x,0,w,height);c.renderer.setScissor(x,0,w,height);c.renderer.render(c.scene,c.camera);
  }
  c.renderer.setScissorTest(false);
}
async function open(){
  if(opening||current)return;opening=true;buttonState();setStatus('Compiling the two native material graphs…');const ownGeneration=++generation;
  let renderer,device,ownedSession=null,resources=new Set();
  try{
    const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('WebGPU is unavailable. Try a recent Chrome or Edge.');
    device=await adapter.requestDevice();renderer=new THREE.WebGPURenderer({device,antialias:true});await renderer.init();
    if(ownGeneration!==generation){renderer.dispose();device.destroy();return;}
    renderer.setPixelRatio(1);renderer.setClearColor(0x121a1b);renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.3;
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(39,1,.1,40);camera.position.set(.0,.2,7.6);camera.lookAt(0,0,0);
    scene.add(new THREE.HemisphereLight(0xe2fff0,0x2b294d,2.1));
    const key=new THREE.DirectionalLight(0xf0fff4,3.1);key.position.set(-3,5,6);scene.add(key);
    const fill=new THREE.DirectionalLight(0xadc0ff,1.2);fill.position.set(5,-2,1);scene.add(fill);
    const geometry=surface();resources.add(geometry);const mesh=new THREE.Mesh(geometry);mesh.material.dispose();mesh.material=null;mesh.frustumCulled=false;scene.add(mesh);
    const c={renderer,device,resources,scene,camera,mesh,uv:uv(),time:uniform(.23),frequency:uniform(64),shutter:uniform(.016),disposed:false,width:0,height:0};
    c.observer=new ResizeObserver(()=>{try{render();}catch(error){fail(error);}});ownedSession=c;current=c;
    device.addEventListener('uncapturederror',event=>{if(!c.disposed)fail(event.error);});
    device.lost.then(info=>{if(!c.disposed)fail(Error('Device lost: '+info.message));});
    $('#view').append(renderer.domElement);$('#empty').style.display='none';c.observer.observe($('#view'));configure();
    await device.queue.onSubmittedWorkDone();if(current!==c)return;
    opening=false;buttonState();setStatus('Ready. Adjust density or preview motion. This folded, lit surface uses a local footprint approximation; affine reference tests are separate.');
  }catch(error){
    // An old initialization/queue promise can reject after Close and a new Open.
    // Clean up only that attempt; it must not close or reset the newer session.
    if(ownGeneration!==generation){if(!ownedSession?.disposed){for(const r of resources)r.dispose?.();renderer?.dispose();device?.destroy();}return;}
    if(current===ownedSession&&current)fail(error);
    else{for(const r of resources)r.dispose?.();renderer?.dispose();device?.destroy();opening=false;buttonState();setStatus(`Unable to open: ${error.message}`);}
  }
}
function play(){
  if(!current)return;if(playing){pause();setStatus('Paused. No animation work is running.');return;}
  playing=true;epoch=performance.now();last=0;const start=current.time.value;elements.play.textContent='Pause preview';setStatus('Previewing material animation and a gentle orbit for twelve seconds.');
  const frame=now=>{if(!playing||!current)return;if(now-epoch>=12000){pause();setStatus('Preview complete. GPU is idle; release it with the button.');return;}
    try{if(now-last>=1000/30){current.time.value=start+(now-epoch)/1000;render();last=now;}}catch(error){fail(error);return;}
    raf=requestAnimationFrame(frame);
  };raf=requestAnimationFrame(frame);
}
elements.start.addEventListener('click',open);elements.play.addEventListener('click',play);elements.stop.addEventListener('click',()=>close());
elements.pattern.addEventListener('change',()=>{pause();try{configure();}catch(error){fail(error);}});
for(const input of [elements.density,elements.exposure])input.addEventListener('input',()=>{labels();try{render();}catch(error){fail(error);}});
document.addEventListener('visibilitychange',()=>{if(document.hidden)close('Page hidden; GPU released.');});window.addEventListener('pagehide',()=>close());
window.footprintDemo={open,close,pause,render,setTime(t){if(!Number.isFinite(t)||Math.abs(t)>100)throw Error('Invalid demo time');if(current){pause();current.time.value=t;render();}},get status(){return{allocated:!!current,opening,playing,kind:elements.pattern.value,stats:current?.stats??null};}};
buttonState();
