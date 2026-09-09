import * as THREE from 'three/webgpu';
import {float, uv, uniform, vec2, vec3, vec4} from 'three/tsl';
import {compileFootprint} from './compiler.js';
import {KINDS, makeFixture, numericFixture, integrateBox, manualFilteredFixture} from './fixtures.mjs';
import {rebind} from '../constraint-editing/rebind.js';

export const GPU_PROTOCOL = Object.freeze({
  width:96, height:64, maximumRenders:64, timeoutMs:40000,
  pixelAbsolute:5e-4, coordinateAbsolute:2e-6, alphaAbsolute:2e-6,
  supersampling:'64 midpoint samples: 8x8 spatial when shutter=0; 4x4x4 space-time when shutter>0',
  scope:'Linear RGB of an affine UV full-screen quad. Exact independent uniform pixel/shutter boxes. No geometry, lighting, visibility, perspective, nonlinear warp, timing or general antialiasing claim.'
});
export const GPU_CASES = Object.freeze([
  {id:'magnified',scale:[.05,.05],origin:[-.31,.17],frequency:1,time:.137,shutter:0},
  {id:'minified',scale:[2,2],origin:[-.31,.17],frequency:8,time:.137,shutter:0},
  {id:'anisotropic',scale:[3,.4],origin:[-.31,.17],frequency:6,time:.137,shutter:0},
  {id:'combined-shutter',scale:[1.5,1],origin:[-.31,.17],frequency:6,time:.137,shutter:.18}
]);
const LANES = ['point','compiled','manual','supersample64'];
const f32Case = value => ({...value,scale:value.scale.map(Math.fround),origin:value.origin.map(Math.fround),
  frequency:Math.fround(value.frequency),time:Math.fround(value.time),shutter:Math.fround(value.shutter)});

/** Owns one ordinary WebGPU device. No allocations happen merely by importing. */
export async function createFootprintRenderer({width=GPU_PROTOCOL.width,height=GPU_PROTOCOL.height,maximumRenders=GPU_PROTOCOL.maximumRenders}={}) {
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>512||height>512)throw Error('Render dimensions exceed the bounded harness');
  if(!Number.isInteger(maximumRenders)||maximumRenders<1||maximumRenders>100)throw Error('Render budget must be 1..100');
  let device,renderer,disposed=false,busy=false,renders=0,timer;
  const errors=[],shaders=[],owned=[],programs=new Map();
  const check=()=>{if(disposed)throw Error('Footprint renderer disposed');if(errors.length)throw Error(errors.join('\n'));};
  const dispose=()=>{
    if(disposed)return;disposed=true;clearTimeout(timer);
    for(const value of owned)try{value.dispose?.();}catch(error){errors.push('Disposal: '+String(error.message??error));}
    try{renderer?.dispose();}finally{device?.destroy();renderer?.domElement.remove();}
  };
  try{
    const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('WebGPU unavailable');
    device=await adapter.requestDevice();
    device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
    device.lost.then(info=>{if(!disposed)errors.push('Device lost: '+info.message);});
    const createShaderModule=device.createShaderModule.bind(device);
    device.createShaderModule=descriptor=>{shaders.push(descriptor.code);return createShaderModule(descriptor);};
    timer=setTimeout(()=>{errors.push('40-second owned-device watchdog');dispose();},GPU_PROTOCOL.timeoutMs);
    renderer=new THREE.WebGPURenderer({device,antialias:false});await renderer.init();check();
    renderer.setPixelRatio(1);renderer.setSize(width,height,false);
    renderer.outputColorSpace=THREE.LinearSRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;
    const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(-1,1,1,-1,.1,10);camera.position.z=1;
    const geometry=new THREE.PlaneGeometry(2,2),mesh=new THREE.Mesh(geometry);mesh.frustumCulled=false;scene.add(mesh);owned.push(geometry);
    const target=new THREE.RenderTarget(width,height,{type:THREE.FloatType,format:THREE.RGBAFormat,depthBuffer:false,samples:0});
    target.texture.colorSpace=THREE.LinearSRGBColorSpace;owned.push(target);
    function material(node){const value=new THREE.MeshBasicNodeMaterial();value.fragmentNode=vec4(node,1);value.toneMapped=false;value.depthTest=false;value.depthWrite=false;owned.push(value);return value;}
    function program(kind){
      if(programs.has(kind))return programs.get(kind);
      if(!KINDS.includes(kind))throw Error('Unknown fixture');
      const coordinate=uv(),scale=uniform(new THREE.Vector2(1,1)),origin=uniform(new THREE.Vector2()),time=uniform(0),frequency=uniform(1),shutter=uniform(0);
      const point=makeFixture(kind,{uv:coordinate.mul(scale).add(origin),time,frequency});
      const compiled=compileFootprint(point,{time,shutter,inputs:[coordinate,scale,origin,frequency]});
      const manual=manualFilteredFixture(kind,{uv:coordinate.mul(scale).add(origin),time,frequency,
        du:vec2(scale.x.div(width),0),dv:vec2(0,scale.y.div(height)),dt:shutter});
      // Sample the same original native DAG. Midpoint offsets are in base UV;
      // the original scale/origin transform is retained by exact input rebinding.
      function sampledGraph(spatial,temporal){
        let sampled=vec3(0);
        for(let t=0;t<temporal;t++)for(let y=0;y<spatial;y++)for(let x=0;x<spatial;x++){
          const shiftedUV=coordinate.add(vec2(((x+.5)/spatial-.5)/width,((y+.5)/spatial-.5)/height));
          const shiftedTime=time.add(shutter.mul((t+.5)/temporal-.5));
          sampled=sampled.add(rebind(point,new Map([[coordinate,shiftedUV],[time,shiftedTime],[scale,scale],[origin,origin],[frequency,frequency]])));
        }
        return sampled.div(spatial*spatial*temporal);
      }
      const result={scale,origin,time,frequency,shutter,stats:compiled.stats,
        materials:{point:material(point),compiled:material(compiled.node),manual:material(manual),
          supersample64:material(sampledGraph(8,1)),supersampleTime64:material(sampledGraph(4,4))}};
      programs.set(kind,result);return result;
    }
    const compiledMaterials=new Set();
    async function draw(selected){
      if(busy)throw Error('Concurrent footprint render');check();if(renders>=maximumRenders)throw Error('Render budget exhausted');busy=true;
      try{
        mesh.material=selected;renderer.setRenderTarget(target);
        if(!compiledMaterials.has(selected)){await renderer.compileAsync(scene,camera);compiledMaterials.add(selected);check();}
        renderer.render(scene,camera);renders++;
        await device.queue.onSubmittedWorkDone();check();
        const raw=await renderer.readRenderTargetPixelsAsync(target,0,0,width,height);check();
        if(!(raw instanceof Float32Array))throw Error('Expected exact Float32 RGBA render-target readback');
        return Array.from(raw);
      }finally{renderer.setRenderTarget(null);busy=false;}
    }
    async function render(kind,lane,settings){
      if(!LANES.includes(lane))throw Error('Unknown render lane');
      const s=f32Case(settings);
      if([...s.scale,...s.origin,s.time,s.frequency,s.shutter].some(v=>!Number.isFinite(v))||s.scale.some(v=>v<0||v>4)||s.frequency<0||s.frequency>16||s.shutter<0||s.shutter>1)throw Error('Settings outside bounded protocol');
      const p=program(kind);p.scale.value.set(...s.scale);p.origin.value.set(...s.origin);p.time.value=s.time;p.frequency.value=s.frequency;p.shutter.value=s.shutter;
      const selected=lane==='supersample64'&&s.shutter>0?'supersampleTime64':lane;
      return {rgba:await draw(p.materials[selected]),settings:s,compiler:p.stats};
    }
    const calibration=material(vec3(uv(),0));
    return {render,calibrate:()=>draw(calibration),dispose,check,errors,shaders,
      adapter:adapter.info?{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description}:null,
      get status(){return {disposed,busy,renders,maximumRenders,width,height};}};
  }catch(error){dispose();throw error;}
}

function compare(rgba,expected,{tolerance=null}={}){
  let squared=0,maxAbsolute=0,maximumIndex=-1,mismatches=0,nonfinite=0,maxAlphaError=0;
  const failureExamples=[];
  for(let i=0;i<expected.length;i++){
    const actual=rgba[Math.floor(i/3)*4+i%3],difference=Math.abs(actual-expected[i]);
    if(!Number.isFinite(actual)||!Number.isFinite(expected[i])){nonfinite++;mismatches++;continue;}
    squared+=difference*difference;
    if(difference>maxAbsolute){maxAbsolute=difference;maximumIndex=i;}
    if(tolerance!==null&&difference>tolerance){mismatches++;if(failureExamples.length<12)failureExamples.push({pixel:Math.floor(i/3),channel:i%3,actual,expected:expected[i],difference});}
  }
  for(let i=3;i<rgba.length;i+=4)maxAlphaError=Math.max(maxAlphaError,Math.abs(rgba[i]-1));
  return {rmse:Math.sqrt(squared/expected.length),maxAbsolute,maximumIndex,mismatches,nonfinite,maxAlphaError,failureExamples};
}
function cpuImages(kind,s,width,height){
  const exact=[],point=[];
  const du=[s.scale[0]/width,0],dv=[0,s.scale[1]/height];
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    // PlaneGeometry UV is bottom-up; WebGPU copyTextureToBuffer rows are top-down.
    const u=(x+.5)/width*s.scale[0]+s.origin[0],v=(1-(y+.5)/height)*s.scale[1]+s.origin[1];
    point.push(...numericFixture(kind,u,v,s.time,{frequency:s.frequency}));
    exact.push(...integrateBox(kind,{u,v,t:s.time,du,dv,dt:s.shutter,frequency:s.frequency}));
  }
  return {exact,point,du,dv};
}

export async function runProbe(){
  const started=performance.now();let context;
  const result={kind:'native-tsl-footprint-gpu-v1',status:'failed',protocol:GPU_PROTOCOL,
    cases:GPU_CASES.map(f32Case),cells:[],errors:[],shaders:[],scope:GPU_PROTOCOL.scope};
  globalThis.footprintProbeResult=result;
  try{
    context=await createFootprintRenderer();result.adapter=context.adapter;result.errors=context.errors;result.shaders=context.shaders;
    const {width,height}=GPU_PROTOCOL,calibration=await context.calibrate(),expected=[];
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)expected.push((x+.5)/width,1-(y+.5)/height,0);
    result.calibration={rgba:calibration,comparison:compare(calibration,expected,{tolerance:GPU_PROTOCOL.coordinateAbsolute})};
    if(result.calibration.comparison.mismatches||result.calibration.comparison.maxAlphaError>GPU_PROTOCOL.alphaAbsolute)throw Error('Full-quad orientation/linear-output calibration failed');
    for(const kind of KINDS)for(const source of GPU_CASES){
      const settings=f32Case(source),cpu=cpuImages(kind,settings,width,height);
      const cell={kind,id:settings.id,settings,pixelEdges:{du:cpu.du,dv:cpu.dv},referenceRGB:cpu.exact,pointReferenceRGB:cpu.point,lanes:{}};
      result.cells.push(cell);
      for(const lane of LANES){
        const actual=await context.render(kind,lane,settings);
        cell.compiler=actual.compiler;
        const comparison=compare(actual.rgba,cpu.exact,{tolerance:['compiled','manual'].includes(lane)?GPU_PROTOCOL.pixelAbsolute:null});
        const entry={rgba:actual.rgba,comparison};cell.lanes[lane]=entry;
        if(lane==='point')entry.pointParity=compare(actual.rgba,cpu.point,{tolerance:GPU_PROTOCOL.pixelAbsolute});
        context.check();
      }
      const candidate=cell.lanes.compiled.comparison,baseline=cell.lanes.point.comparison,supersampled=cell.lanes.supersample64.comparison;
      cell.quality={pointRMSE:baseline.rmse,compiledRMSE:candidate.rmse,manualRMSE:cell.lanes.manual.comparison.rmse,supersample64RMSE:supersampled.rmse,
        reductionVersusPoint:baseline.rmse>0?1-candidate.rmse/baseline.rmse:null,
        reductionVersusSupersample64:supersampled.rmse>0?1-candidate.rmse/supersampled.rmse:null};
      cell.passed=LANES.every(lane=>{const a=cell.lanes[lane];return a.comparison.nonfinite===0&&a.comparison.maxAlphaError<=GPU_PROTOCOL.alphaAbsolute&&a.comparison.mismatches===0&&(!a.pointParity||a.pointParity.mismatches===0);});
    }
    result.passedCells=result.cells.filter(c=>c.passed).length;
    result.status=result.passedCells===KINDS.length*GPU_CASES.length&&result.errors.length===0?'passed':'mismatch';
  }catch(error){result.failure=String(error.stack??error);}
  finally{
    context?.dispose();result.lifecycle=context?.status??{disposed:true,renders:0};
    result.completedWallMs=performance.now()-started;
    result.timingScope='Descriptive total includes initialization, shader compilation, CPU reference, readback and all controls; no speed comparison.';
  }
  return result;
}
