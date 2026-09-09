import * as THREE from 'three/webgpu';
import {storage,wgslFn} from 'three/tsl';
import {correctedTriangleWGSL} from './closest-point-correction.mjs';
import {closestPointToTriangle} from '../../node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js';

const $=id=>document.getElementById(id);
const ui=Object.fromEntries(['height','order','compare','stop','diagram','scale','plot-caption','ratio','ratio-caption','original-distance','fixed-distance','original-contact','fixed-contact','radius','status','raw'].map(id=>[id,$(id)]));
let busy=false,cancelled=false,active=null;
const cleanNumber=value=>Number.isFinite(value)?Number(value.toPrecision(7)).toString():'invalid';
const setStatus=(text,state='idle')=>{ui.status.textContent=text;ui.status.dataset.state=state;};

function fixture() {
  const height=Math.fround(Number(ui.height.value)),order=ui.order.value;
  if(![1,.01,.001].includes(Number(ui.height.value))||!['ABC','ACB','BAC','BCA','CAB','CBA'].includes(order))throw Error('Unsupported fixture.');
  const vertices={A:[0,0,0],B:[2,0,0],C:[-1,height,0]};
  return {height,order,vertices,p:[1,Math.fround(-2*height),0],radius:4*height};
}

function draw(f,result=null) {
  const project=p=>[310+p[0]*135,170-p[1]*135/f.height];
  const xy=p=>project(p).join(',');
  const text=(x,y,content,color='#b9c6cc',size=15)=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-family="system-ui,sans-serif">${content}</text>`;
  const triangle=Object.values(f.vertices).map(xy).join(' '),p=project(f.p);
  let content=`<defs><marker id="arrow-original" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#ff947d" stroke-width="1.3"/></marker><marker id="arrow-fixed" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#91e4be" stroke-width="1.3"/></marker></defs>`;
  for(const y of [35,170,305,440])content+=`<path d="M95,${y}H720" stroke="#303b40" stroke-width="1" stroke-dasharray="3 7"/>`;
  content+=`<path d="M310,23V465" stroke="#3b484f" stroke-width="1"/><polygon points="${triangle}" fill="#61767e" fill-opacity=".22" stroke="#a0b1b8" stroke-width="1.6"/>`;
  for(const [label,point] of Object.entries(f.vertices)){const v=project(point);content+=`<circle cx="${v[0]}" cy="${v[1]}" r="4" fill="#dce3e4"/>`+text(v[0]+(label==='C'?-25:8),v[1]-12,label);}
  if(result){
    for(const [method,color] of [['original','#ff947d'],['fixed','#91e4be']]){
      const q=project(result[method].closestPoint);
      content+=`<path d="M${p.join(',')}L${q.join(',')}" stroke="${color}" stroke-width="3" fill="none" marker-end="url(#arrow-${method})"/><circle cx="${q[0]}" cy="${q[1]}" r="7" fill="#192124" stroke="${color}" stroke-width="2.5"/>`;
    }
    if(Math.hypot(...result.original.closestPoint.map((v,i)=>v-result.fixed.closestPoint[i]))>1e-5){
      const o=project(result.original.closestPoint),q=project(result.fixed.closestPoint);
      content+=text((p[0]+o[0])/2-82,(p[1]+o[1])/2+6,'original','#ff947d',13);
      content+=text((p[0]+q[0])/2+13,(p[1]+q[1])/2+6,'corrected','#91e4be',13);
    }else content+=text(p[0]+15,310,'answers coincide','#91e4be',13);
  }else content+=text(403,265,'Press Compare to measure','#91a2aa',14);
  content+=`<circle cx="${p[0]}" cy="${p[1]}" r="7" fill="#eff0e9" stroke="#101315" stroke-width="2"/>`+text(p[0]+15,p[1]+6,'P · query point','#eff0e9',15);
  content+=text(100,485,`A (0, 0)    B (2, 0)    C (−1, ${cleanNumber(f.height)})`, '#84969e',12);
  ui.diagram.innerHTML=content;
  const magnification=1/Number(ui.height.value);
  ui.scale.textContent=magnification===1?'Actual proportions':`Y magnified ${magnification.toLocaleString('en-US')}×`;
  ui['plot-caption'].textContent=magnification===1?'Actual XY proportions. Both functions use exactly the same triangle and query.':'Only the vertical axis is magnified. Distances and contacts use actual coordinates.';
  ui.diagram.setAttribute('aria-label',result?`GPU comparison on triangle ${f.order}. Original distance ${result.original.distance}, corrected distance ${result.fixed.distance}. Vertical magnification ${magnification}.`:'Triangle ABC and query P. GPU answers appear after pressing Compare.');
}

function reset() {
  const f=fixture();draw(f);ui.radius.textContent=cleanNumber(f.radius);
  for(const id of ['ratio','original-distance','fixed-distance'])ui[id].textContent='—';
  for(const id of ['original-contact','fixed-contact']){ui[id].textContent='UNTESTED';ui[id].className='badge';}
  ui['ratio-caption'].textContent='Original distance ÷ corrected distance. Waiting for a GPU comparison.';
  ui.raw.textContent=JSON.stringify({coordinates:f,note:'No GPU output for these settings.'},null,2);
  window.triangleDemo.lastResult=null;
}

function release() {
  if(!active)return;
  const owned=active;active=null;owned.released=true;
  for(const attribute of owned.attributes){try{attribute.dispose();}catch{}}
  try{owned.renderer?.dispose();}catch{}
  try{owned.device?.destroy();}catch{}
}

function stop(reason='Stopped. GPU resources released.') {
  if(!busy){release();return;}
  cancelled=true;release();ui.stop.disabled=true;setStatus(reason);
}

function present(f,result) {
  draw(f,result);
  const ratio=result.original.distance/result.fixed.distance;
  ui.ratio.textContent=Number.isFinite(ratio)?`${ratio>=100?ratio.toFixed(1):ratio.toFixed(3)}×`:'—';
  ui['ratio-caption'].textContent=Math.abs(ratio-1)<1e-4?'These GPU answers agree for this vertex order.':'Original distance ÷ corrected distance. This measures geometric error.';
  for(const method of ['original','fixed']){
    ui[`${method}-distance`].textContent=cleanNumber(result[method].distance);
    const hit=result[method].distance<=f.radius,badge=ui[`${method}-contact`];
    badge.textContent=hit?'CONTACT HIT':'CONTACT MISS';badge.className=`badge ${hit?'hit':'miss'}`;
  }
  const record={kind:'interactive-triangle-gpu-comparison-v1',fixture:f,original:result.original,fixed:result.fixed,
    distanceRatio:ratio,contactRadius:f.radius,contact:{original:result.original.distance<=f.radius,fixed:result.fixed.distance<=f.radius},
    metric:'Distance ratio, not speed. Both functions executed in one GPU dispatch.',raw:result.raw};
  window.triangleDemo.lastResult=record;ui.raw.textContent=JSON.stringify(record,null,2);
}

async function compare() {
  if(busy)return;
  busy=true;cancelled=false;ui.compare.disabled=true;ui.stop.disabled=false;ui.height.disabled=true;ui.order.disabled=true;
  const f=fixture();reset();setStatus('Preparing one GPU comparison…','busy');
  let scopeOpen=false;
  try {
    if(!navigator.gpu)throw Error('WebGPU is unavailable. Open this local page in a WebGPU-capable browser.');
    // The static import selects the current TSL helper directly, avoiding the
    // ambiguous deprecated barrel export. Serve/bundle this source unmodified.
    const adapter=await navigator.gpu.requestAdapter();if(cancelled)return;if(!adapter)throw Error('No WebGPU adapter is available.');
    const device=await adapter.requestDevice();
    if(cancelled){device.destroy();return;}
    const owned={device,renderer:null,attributes:[],released:false,errors:[]};active=owned;
    device.addEventListener('uncapturederror',event=>owned.errors.push(event.error.message));
    device.lost.then(info=>{if(!owned.released)owned.errors.push(`Device lost: ${info.message}`);});
    const renderer=new THREE.WebGPURenderer({device,antialias:false});owned.renderer=renderer;
    await renderer.init();if(cancelled)return;
    const input=new Float32Array(16);[f.p,...[...f.order].map(letter=>f.vertices[letter])].forEach((p,i)=>input.set([...p,0],i*4));
    const inputAttribute=new THREE.StorageBufferAttribute(input,4),outputAttribute=new THREE.StorageBufferAttribute(new Float32Array(16),4);
    owned.attributes.push(inputAttribute,outputAttribute);
    const inputNode=storage(inputAttribute,'vec4',4).toReadOnly(),outputNode=storage(outputAttribute,'vec4',4);
    const correctedSource=correctedTriangleWGSL.replace(/\bfn\s+closestPointToTriangle\s*\(/,'fn correctedPointToTriangle(');
    if(correctedSource===correctedTriangleWGSL)throw Error('Unexpected correction source.');
    const corrected=wgslFn(correctedSource);
    const comparison=wgslFn(`fn compareTriangle(inputs:ptr<storage,array<vec4f>,read>, outputs:ptr<storage,array<vec4f>,read_write>) -> void {
      let p=inputs[0].xyz;let a=inputs[1].xyz;let b=inputs[2].xyz;let c=inputs[3].xyz;
      let oldBary=closestPointToTriangle(p,a,b,c);let newBary=correctedPointToTriangle(p,a,b,c);
      let oldPoint=a*oldBary.x+b*oldBary.y+c*oldBary.z;let newPoint=a*newBary.x+b*newBary.y+c*newBary.z;
      let oldDelta=p-oldPoint;let newDelta=p-newPoint;
      outputs[0]=vec4f(oldBary,0.0);outputs[1]=vec4f(oldPoint,dot(oldDelta,oldDelta));
      outputs[2]=vec4f(newBary,0.0);outputs[3]=vec4f(newPoint,dot(newDelta,newDelta));
    }`,[closestPointToTriangle,corrected]);
    const kernel=comparison({inputs:inputNode,outputs:outputNode}).computeKernel([1,1,1]);
    device.pushErrorScope('validation');scopeOpen=true;
    renderer.compute(kernel,[1,1,1]);await device.queue.onSubmittedWorkDone();if(cancelled)return;
    const error=await device.popErrorScope();scopeOpen=false;if(error)owned.errors.push(error.message);
    if(owned.errors.length)throw Error(owned.errors.join('\n'));
    const raw=Array.from(new Float32Array(await renderer.getArrayBufferAsync(outputAttribute)));if(cancelled)return;
    if(raw.length!==16||!raw.every(Number.isFinite)||raw[7]<0||raw[15]<=0)throw Error('The GPU returned invalid output.');
    const answer=offset=>({barycoord:raw.slice(offset,offset+3),closestPoint:raw.slice(offset+4,offset+7),distanceSq:raw[offset+7],distance:Math.sqrt(raw[offset+7])});
    const result={original:answer(0),fixed:answer(8),raw};
    // The exact answer for this family is known; never label an invalid local
    // correction as a successful fix merely because it differs from upstream.
    if(Math.hypot(result.fixed.closestPoint[0]-1,result.fixed.closestPoint[1],result.fixed.closestPoint[2])>1e-4 || Math.abs(result.fixed.distance-2*f.height)>1e-5)throw Error('The local correction failed this fixture’s analytic check.');
    present(f,result);setStatus('Compared on your GPU in one dispatch. GPU resources released.','ready');
  } catch(error) {
    if(!cancelled){setStatus(error?.message??String(error),'error');ui.raw.textContent=String(error?.stack??error);}
  } finally {
    if(scopeOpen&&active){try{await active.device.popErrorScope();}catch{}}
    release();busy=false;ui.compare.disabled=false;ui.stop.disabled=true;ui.height.disabled=false;ui.order.disabled=false;
  }
}

window.triangleDemo={lastResult:null,compare,stop,get busy(){return busy;},
  get status(){return {state:ui.status.dataset.state,message:ui.status.textContent,gpuAllocated:active!==null,busy};}};
ui.compare.addEventListener('click',compare);ui.stop.addEventListener('click',()=>stop());
for(const control of [ui.height,ui.order])control.addEventListener('change',()=>{reset();setStatus('Settings changed. Press Compare to measure these GPU answers.');});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop('Page hidden. GPU work stopped and resources released.');});
window.addEventListener('pagehide',()=>stop());
reset();
