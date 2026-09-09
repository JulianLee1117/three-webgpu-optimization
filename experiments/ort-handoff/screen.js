import * as THREE from 'three/webgpu';
import { Fn, storage, uniform, uint, float, instanceIndex, vertexIndex, vec3, mix } from 'three/tsl';
import * as ort from '/node_modules/onnxruntime-web-dev/dist/ort.webgpu.bundle.min.mjs';
const weights = await (await fetch('/experiments/ort-surface/wave.json')).json();
const state = { phase: 'idle', errors: [], disposed: false };
let app;
function assert(c,m) { if (!c) throw new Error(m); }
const wait = ms => new Promise(r => setTimeout(r,ms));
const metric = (xs,q) => {
  const sorted=[...xs].sort((a,b)=>a-b),n=sorted.length;
  return q===.5 && n%2===0 ? (sorted[n/2-1]+sorted[n/2])/2 : sorted[Math.max(0,Math.ceil(q*n)-1)];
};

async function initialize(side, stock) {
  assert([128,256].includes(side), 'Grid size exceeds experiment bounds.');
  const adapter = await navigator.gpu.requestAdapter(); assert(adapter,'No adapter.');
  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', e => state.errors.push(e.error.message));
  device.lost.then(info => { if (!state.disposed) state.errors.push('Unexpected device loss: '+info.message); });
  const n = side*side;
  app = { side,n,device,adapter, lanes: stock ? ['internal'] : ['borrowed','internal','gpu-copy','cpu-copy'] };
  app.renderer = new THREE.WebGPURenderer({device,antialias:false});
  await app.renderer.init();
  app.renderer.setSize(384,256); app.renderer.setPixelRatio(1); app.renderer.setClearColor(0x091018,1);
  document.body.appendChild(app.renderer.domElement);
  app.target = new THREE.RenderTarget(384,256,{samples:0});
  app.scene = new THREE.Scene();
  app.camera = new THREE.PerspectiveCamera(42,384/256,.1,100); app.camera.position.set(3,3,4); app.camera.lookAt(0,0,0);
  app.geometry = new THREE.PlaneGeometry(2,2,side-1,side-1);
  app.inputAttribute = new THREE.StorageBufferAttribute(n*3,1);
  const input = storage(app.inputAttribute,'float',n*3);
  app.time = uniform(0);
  app.seed = Fn(()=>{
    const id=instanceIndex, base=id.mul(uint(3));
    input.element(base).assign(float(id.mod(uint(side))).div(side-1).mul(2).sub(1));
    input.element(base.add(uint(1))).assign(float(id.div(uint(side))).div(side-1).mul(2).sub(1));
    input.element(base.add(uint(2))).assign(app.time);
  })().compute(n);
  app.renderer.compute(app.seed); await device.queue.onSubmittedWorkDone();
  app.inputBuffer=app.renderer.backend.get(app.inputAttribute).buffer;
  app.inputTensor=ort.Tensor.fromGpuBuffer(app.inputBuffer,{dataType:'float32',dims:[n,3]});
  app.scratch=device.createBuffer({size:n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  app.scratchTensor=ort.Tensor.fromGpuBuffer(app.scratch,{dataType:'float32',dims:[n,1]});
  app.staging=device.createBuffer({size:n*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  app.attributes={}; app.materials={}; app.buffers={};
  for (const lane of app.lanes) {
    const attr=lane==='borrowed' ? new THREE.ExternalStorageBufferAttribute(app.scratch,{device,type:'float',count:n}) : new THREE.StorageBufferAttribute(n,1);
    app.attributes[lane]=attr;
    const values=storage(attr,'float',n).toReadOnly(), h=values.element(vertexIndex);
    const material=new THREE.MeshBasicNodeMaterial({side:THREE.DoubleSide});
    material.positionNode=Fn(()=>{
      const x=float(vertexIndex.mod(uint(side))).div(side-1).mul(2).sub(1);
      const z=float(vertexIndex.div(uint(side))).div(side-1).mul(2).sub(1);
      return vec3(x,h,z);
    })();
    material.colorNode=mix(vec3(.02,.1,.3),vec3(.4,1,.65),h.add(.7).div(1.4));
    app.materials[lane]=material;
    if (!app.mesh) { app.mesh=new THREE.Mesh(app.geometry,material); app.mesh.frustumCulled=false; app.scene.add(app.mesh); }
    app.mesh.material=material; app.renderer.setRenderTarget(app.target); app.renderer.render(app.scene,app.camera);
    app.buffers[lane]=app.renderer.backend.get(attr).buffer;
  }
  await device.queue.onSubmittedWorkDone();
  app.internalTensor=ort.Tensor.fromGpuBuffer(app.buffers.internal,{dataType:'float32',dims:[n,1]});
  ort.env.wasm.numThreads=1; ort.env.wasm.proxy=false;
  ort.env.wasm.wasmPaths={wasm:'/node_modules/onnxruntime-web-dev/dist/ort-wasm-simd-threaded.asyncify.wasm'};
  app.session=await ort.InferenceSession.create(new Uint8Array(await(await fetch('/experiments/ort-surface/wave.onnx')).arrayBuffer()),{
    executionProviders:[{name:'webgpu',device,validationMode:'wgpuOnly'}]
  });
  return {threeRevision:THREE.REVISION,ortVersion:ort.env.versions.web,stock,side,n,
    internalBufferOwnedByThree:app.renderer.backend.get(app.attributes.internal).ownsBuffer!==false,
    inputBufferSame:app.inputTensor.gpuBuffer===app.inputBuffer,
    internalOutputBufferSame:app.internalTensor.gpuBuffer===app.buffers.internal,
    adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description}};
}
async function download(buffer) {
  const enc=app.device.createCommandEncoder(); enc.copyBufferToBuffer(buffer,0,app.staging,0,app.n*4); app.device.queue.submit([enc.finish()]);
  await app.staging.mapAsync(GPUMapMode.READ);
  return new Float32Array(app.staging.getMappedRange());
}
async function update(lane,t) {
  assert(!state.errors.length,state.errors.join('; '));
  const start=performance.now();
  app.time.value=t; app.renderer.compute(app.seed);
  const output=lane==='internal'?app.internalTensor:app.scratchTensor;
  const result=await app.session.run({X:app.inputTensor},{Y:output});
  assert(result.Y===output && result.Y.gpuBuffer===output.gpuBuffer,'ORT did not preserve preallocated output.');
  if (lane==='gpu-copy') {
    const enc=app.device.createCommandEncoder(); enc.copyBufferToBuffer(app.scratch,0,app.buffers[lane],0,app.n*4); app.device.queue.submit([enc.finish()]);
  } else if (lane==='cpu-copy') {
    const data=await download(app.scratch); app.attributes[lane].array.set(data); app.staging.unmap(); app.attributes[lane].needsUpdate=true;
  }
  app.mesh.material=app.materials[lane]; app.renderer.render(app.scene,app.camera);
  await app.device.queue.onSubmittedWorkDone();
  return performance.now()-start;
}
function cpu(id,t) {
  const x=(id%app.side)/(app.side-1)*2-1,z=Math.floor(id/app.side)/(app.side-1)*2-1;
  let y=weights.C;
  for(let k=0;k<weights.hidden;k++)y+=Math.sin(x*weights.W[0][k]+z*weights.W[1][k]+t*weights.W[2][k]+weights.B[k])*weights.V[k];
  return y;
}
async function validate() {
  state.phase='correctness';
  const proof=[];
  for (const type of ['out-of-memory','internal','validation']) app.device.pushErrorScope(type);
  try {
    for(const t of [.375,2.25]) {
      let reference;
      for(const lane of app.lanes) {
        await update(lane,t);
        const actual=await download(app.buffers[lane]);
        let maxError=0; for(let i=0;i<app.n;i++)maxError=Math.max(maxError,Math.abs(actual[i]-cpu(i,t)));
        app.staging.unmap(); assert(Number.isFinite(maxError)&&maxError<2e-5,'Output mismatch: '+lane+' '+maxError);
        const image=await app.renderer.readRenderTargetPixelsAsync(app.target,0,0,384,256);
        if(!reference)reference=image;
        let unequalBytes=0;for(let i=0;i<image.length;i++)if(image[i]!==reference[i])unequalBytes++;
        assert(unequalBytes===0,'Image mismatch: '+lane);
        let content=0; for(let i=0;i<image.length;i+=4) if(image[i]!==image[0]||image[i+1]!==image[1]||image[i+2]!==image[2])content++;
        assert(content>1000,'Empty image');
        proof.push({lane,t,maxError,unequalBytes,nonBackgroundPixels:content});
      }
    }
  } finally { for(let i=0;i<3;i++){const e=await app.device.popErrorScope();if(e)state.errors.push(e.message);} }
  assert(!state.errors.length,state.errors.join('; '));
  return proof;
}
function permutations(xs){return xs.length?xs.flatMap((x,i)=>permutations(xs.filter((_,j)=>j!==i)).map(t=>[x,...t])):[[]];}
async function measure() {
  assert(!app.measured && app.lanes.length===4,'Only one four-lane measurement per page load.');
  app.measured=true;
  state.phase='warmup';
  for(const type of ['out-of-memory','internal','validation'])app.device.pushErrorScope(type);
  const samples=[],warmup=[];
  try{
    for(let round=0;round<8;round++)for(const lane of app.lanes){const wall=performance.now();warmup.push({lane,ms:await update(lane,.7+round*.13)});await wait(Math.max(0,1000/15-(performance.now()-wall)));}
    const orders=permutations(app.lanes); let seed=731;
    for(let i=orders.length-1;i>0;i--){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const j=seed%(i+1);[orders[i],orders[j]]=[orders[j],orders[i]];}
    state.phase='measurement';
    for(let block=0;block<orders.length;block++)for(let position=0;position<orders[block].length;position++){
      const lane=orders[block][position],wall=performance.now();
      const ms=await update(lane,.2+block*.21);samples.push({block,position,lane,ms});
      await wait(Math.max(0,1000/15-(performance.now()-wall)));
    }
  }finally{for(let i=0;i<3;i++){const e=await app.device.popErrorScope();if(e)state.errors.push(e.message);}}
  assert(!state.errors.length,state.errors.join('; '));
  return {warmup,samples,summary:Object.fromEntries(app.lanes.map(lane=>{const xs=samples.filter(s=>s.lane===lane).map(s=>s.ms);return[lane,{samples:xs.length,medianMs:metric(xs,.5),p95Ms:metric(xs,.95)}]}))};
}
async function dispose() {
  if(!app||state.disposed)return;
  await app.device.queue.onSubmittedWorkDone();
  for(const m of Object.values(app.materials??{}))m.dispose();app.seed?.dispose();app.geometry?.dispose();app.target?.dispose();
  app.attributes?.borrowed?.dispose();
  await app.session?.release();
  // Ordinary storage buffers do not expose public dispose in r185. Release their
  // renderer bookkeeping explicitly after their consumers and ORT session end.
  for(const attr of [app.inputAttribute,...Object.values(app.attributes??{})])if(attr && !attr.isExternalStorageBufferAttribute && app.renderer._attributes.has(attr))app.renderer._attributes.delete(attr);
  await app.renderer?.dispose();app.staging?.destroy();app.scratch?.destroy();
  state.disposed=true;app.device.destroy();state.phase='disposed';
}
window.handoffScreen={initialize,validate,measure,dispose,state};
