import * as THREE from 'three/webgpu';
import { storage, vertexIndex, uint, float, vec3, mix } from 'three/tsl';
import * as ort from '/node_modules/onnxruntime-web-dev/dist/ort.webgpu.bundle.min.mjs';

const state={phase:'idle',errors:[],disposed:false};
let app;
const assert=(ok,message)=>{if(!ok)throw Error(message);};
const status=document.querySelector('#status');
async function array(name){return new Float32Array(await(await fetch('/fixture/'+name)).arrayBuffer());}
function compare(actual,expected){
  assert(actual.length===expected.length,'Output length mismatch');
  let maxAbs=0,absolute=0,denominator=0,failed=0;
  for(let i=0;i<actual.length;i++){
    assert(Number.isFinite(actual[i])&&Number.isFinite(expected[i]),'Nonfinite output');
    const delta=Math.abs(actual[i]-expected[i]);maxAbs=Math.max(maxAbs,delta);absolute+=delta;denominator+=Math.abs(expected[i]);
    if(delta>1e-5+1e-4*Math.abs(expected[i]))failed++;
  }
  return {maxAbs,meanAbs:absolute/actual.length,relativeL1:absolute/denominator,failedElements:failed,allclose:failed===0};
}
function draw(){
  app.renderer.setScissorTest(true);
  for(let i=0;i<3;i++){
    app.mesh.material=app.materials[i];
    app.renderer.setViewport(i*400,0,400,360);app.renderer.setScissor(i*400,0,400,360);
    app.renderer.render(app.scene,app.camera);
  }
  app.renderer.setScissorTest(false);
}
async function read(buffer){
  const staging=app.device.createBuffer({size:384*384*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
    const encoder=app.device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,staging,0,384*384*4);app.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);const result=new Float32Array(staging.getMappedRange()).slice();staging.unmap();return result;
  }finally{staging.destroy();}
}
async function run(){
  assert(state.phase==='idle','One run per page load');
  document.querySelector('#run').disabled=true;state.phase='loading';status.textContent='Loading pinned models…';
  const adapter=await navigator.gpu.requestAdapter();assert(adapter,'WebGPU adapter unavailable');
  const device=await adapter.requestDevice();app={device,sessions:[],attributes:[],materials:[]};
  device.addEventListener('uncapturederror',e=>state.errors.push(e.error.message));
  device.lost.then(info=>{if(!state.disposed)state.errors.push('Unexpected device loss: '+info.message);});
  for(const type of ['out-of-memory','internal','validation'])device.pushErrorScope(type);
  try{
    const [input,reference]=await Promise.all([array('gradient-input.f32'),array('gradient-reference.f32')]);
    assert(input.length===3*384*384 && reference.length===384*384,'Wrong fixture size');
    app.renderer=new THREE.WebGPURenderer({device,antialias:false});await app.renderer.init();
    app.renderer.setPixelRatio(1);app.renderer.setSize(1200,360);app.renderer.setClearColor(0x0a111c,1);
    document.querySelector('#view').appendChild(app.renderer.domElement);
    app.scene=new THREE.Scene();app.camera=new THREE.PerspectiveCamera(40,400/360,.1,100);
    app.camera.position.set(2.1,2.6,3.1);app.camera.lookAt(0,.25,0);
    app.geometry=new THREE.PlaneGeometry(2,2,191,191);
    for(let i=0;i<3;i++){
      const attr=new THREE.StorageBufferAttribute(i===0?reference:384*384,1);app.attributes.push(attr);
      const values=storage(attr,'float',384*384).toReadOnly();
      const col=vertexIndex.mod(uint(192)),row=vertexIndex.div(uint(192));
      const source=row.mul(uint(2*384)).add(col.mul(uint(2)));
      const height=values.element(source);
      const material=new THREE.MeshBasicNodeMaterial({side:THREE.DoubleSide});
      material.positionNode=vec3(float(col).div(191).mul(2).sub(1),height.mul(12),float(row).div(191).mul(2).sub(1));
      material.colorNode=mix(vec3(.02,.12,.32),vec3(.4,1,.7),height.div(.06).clamp());app.materials.push(material);
    }
    app.mesh=new THREE.Mesh(app.geometry,app.materials[0]);app.mesh.frustumCulled=false;app.scene.add(app.mesh);draw();
    await device.queue.onSubmittedWorkDone();
    ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
    ort.env.wasm.wasmPaths={wasm:'/node_modules/onnxruntime-web-dev/dist/ort-wasm-simd-threaded.asyncify.wasm'};
    const metrics={threeRevision:THREE.REVISION,ortVersion:ort.env.versions.web,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},shape:[1,3,384,384],cases:{}};
    for(const [index,lane] of [[1,'original'],[2,'corrected']]){
      assert(!state.errors.length,state.errors.join('; '));status.textContent=`Running ${lane} export…`;state.phase=lane;
      const buffer=app.renderer.backend.get(app.attributes[index]).buffer;
      const tensor=ort.Tensor.fromGpuBuffer(buffer,{dataType:'float32',dims:[1,1,384,384]});
      const model=new Uint8Array(await(await fetch('/fixture/'+lane+'.onnx')).arrayBuffer());
      const session=await ort.InferenceSession.create(model,{executionProviders:[{name:'webgpu',device,validationMode:'wgpuOnly'}]});app.sessions.push(session);
      const output=await session.run({image:new ort.Tensor('float32',input,[1,3,384,384])},{depth:tensor});
      assert(output.depth===tensor && output.depth.gpuBuffer===buffer,'Output buffer identity changed');
      draw();await device.queue.onSubmittedWorkDone();
      const actual=await read(buffer),expected=await array('gradient-'+lane+'.f32');
      metrics.cases[lane]={outputBufferSame:true,webgpuVsCpuOnnx:compare(actual,expected),webgpuVsReference:compare(actual,reference)};
      assert(metrics.cases[lane].webgpuVsCpuOnnx.allclose,'WebGPU does not match CPU ONNX: '+lane);
      if(lane==='corrected')assert(metrics.cases[lane].webgpuVsReference.allclose,'Corrected output fails reference parity');
      if(lane==='original')assert(!metrics.cases[lane].webgpuVsReference.allclose,'Expected discrepancy did not reproduce');
    }
    draw();await device.queue.onSubmittedWorkDone();
    state.metrics=metrics;state.phase='passed';status.textContent='Passed · GPU work finished';
    document.querySelector('#metrics').textContent=`Original export → reference: max absolute difference ${metrics.cases.original.webgpuVsReference.maxAbs.toExponential(3)}\nCorrected export → reference: max absolute difference ${metrics.cases.corrected.webgpuVsReference.maxAbs.toExponential(3)}\nBoth ONNX outputs match their CPU execution. Three reads the exact preallocated output buffers.`;
    return metrics;
  }finally{
    for(let i=0;i<3;i++){const e=await device.popErrorScope();if(e)state.errors.push(e.message);}
    assert(!state.errors.length,state.errors.join('; '));
  }
}
async function dispose(){
  if(!app||state.disposed)return;
  await app.device.queue.onSubmittedWorkDone();
  for(const material of app.materials)material.dispose();
  app.geometry?.dispose();
  for(const session of app.sessions)await session.release();
  for(const attribute of app.attributes)if(app.renderer?._attributes.has(attribute))app.renderer._attributes.delete(attribute);
  await app.renderer?.dispose();state.disposed=true;app.device.destroy();state.phase='disposed';
}
document.querySelector('#run').addEventListener('click',()=>run().catch(async e=>{state.phase='failed';status.textContent='Failed: '+e.message;await dispose();}));
window.zipdepthProbe={run,dispose,state};
