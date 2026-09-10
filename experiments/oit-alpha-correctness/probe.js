import * as THREE from 'three/webgpu';
import { oitPass as stockOIT } from 'three/addons/tsl/display/OITPassNode.js';
import { oitPass as correctedOIT } from '/overlay/OITPassNode.js';

// This is a compositing correctness probe, not a timing or quality benchmark.
// All colors are linear; readback is the actual RGBA8 render-target storage.
const SIZE = 32;
const CASES = Object.freeze([
  { id: 'one-half-transparent', backgroundAlpha: 0, opacities: [.5] },
  { id: 'empty-transparent', backgroundAlpha: 0, opacities: [] },
  { id: 'one-half-opaque-background', backgroundAlpha: 1, opacities: [.5] },
  { id: 'two-half-transparent', backgroundAlpha: 0, opacities: [.5, .5] },
  { id: 'one-half-quarter-background', backgroundAlpha: .25, opacities: [.5] }
]);
const TOLERANCE_BYTES = 2; // Includes revealage R8 quantization and RGBA8 output rounding.
const base64 = bytes => btoa(String.fromCharCode(...bytes));
const expectedAlpha = ({backgroundAlpha,opacities}) => 1-(1-backgroundAlpha)*opacities.reduce((a,x)=>a*(1-x),1);

export async function runProbe() {
  const result = window.oitAlphaProbeResult = {
    kind: 'oit-alpha-gpu-probe-v1', status: 'failed', revision: THREE.REVISION,
    colorSpace: 'LinearSRGBColorSpace', toneMapping: 'NoToneMapping', outputColorTransform: false,
    target: { width: SIZE, height: SIZE, format: 'RGBA8', samples: 0 }, toleranceBytes: TOLERANCE_BYTES,
    fixtures: CASES, cells: [], validationErrors: [],
    lifecycle: { rendererCount: 0, fixtureLaneRenders: 0, rendererRenderCalls: 0, disposed: false }
  };
  let renderer, device, target, geometry;
  const materials = [], passes = [], pipelines = [];
  let scopePushed = false;
  try {
    if (THREE.REVISION !== '186') throw Error('Wrong Three revision');
    renderer = new THREE.WebGPURenderer({ alpha: true, antialias: false });
    result.lifecycle.rendererCount++;
    renderer.setPixelRatio(1); renderer.setSize(SIZE,SIZE,false);
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping;
    await renderer.init();
    if (!renderer.backend.isWebGPUBackend || !renderer.backend.device) throw Error('Actual WebGPU required');
    device = renderer.backend.device;
    device.addEventListener('uncapturederror',event=>result.validationErrors.push(String(event.error.message)));
    device.pushErrorScope('validation'); scopePushed=true;
    result.adapter = device.adapterInfo ? Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,device.adapterInfo[k]])) : null;
    const originalRender=renderer.render.bind(renderer);
    renderer.render=(...args)=>{ result.lifecycle.rendererRenderCalls++; return originalRender(...args); };
    target = new THREE.RenderTarget(SIZE,SIZE,{ type: THREE.UnsignedByteType, format: THREE.RGBAFormat, samples:0, depthBuffer:true });
    target.texture.colorSpace=THREE.LinearSRGBColorSpace;
    geometry = new THREE.PlaneGeometry(2,2);
    const camera = new THREE.OrthographicCamera(-1,1,1,-1,.1,10); camera.position.z=2;
    for (const fixture of CASES) {
      const scene = new THREE.Scene();
      fixture.opacities.forEach((opacity,i)=>{
        const material = new THREE.MeshBasicNodeMaterial({ color:0xff0000, opacity, transparent:true, depthWrite:false });
        materials.push(material); const mesh=new THREE.Mesh(geometry,material); mesh.position.z=i*.1; scene.add(mesh);
      });
      for (const lane of ['direct','stock','corrected']) {
        renderer.setClearColor(0x000000,fixture.backgroundAlpha); renderer.setRenderTarget(target);
        if (lane==='direct') renderer.render(scene,camera);
        else {
          const pass=(lane==='stock'?stockOIT:correctedOIT)(scene,camera,{samples:0}); passes.push(pass);
          const pipeline=new THREE.RenderPipeline(renderer,pass); pipeline.outputColorTransform=false; pipelines.push(pipeline);
          pipeline.render();
        }
        result.lifecycle.fixtureLaneRenders++;
        if (result.lifecycle.fixtureLaneRenders>15 || result.lifecycle.rendererRenderCalls>35) throw Error('Render budget exceeded');
        const rawPixels=await renderer.readRenderTargetPixelsAsync(target,0,0,SIZE,SIZE);
        // r186 returns WebGPU's padded rows, with the last row left unpadded.
        const bytesPerRow=Math.ceil(SIZE*4/256)*256;
        if (!(rawPixels instanceof Uint8Array) || rawPixels.length!==(SIZE-1)*bytesPerRow+SIZE*4) throw Error('Unexpected RGBA8 readback layout');
        const pixels=new Uint8Array(SIZE*SIZE*4);
        for(let y=0;y<SIZE;y++) pixels.set(rawPixels.subarray(y*bytesPerRow,y*bytesPerRow+SIZE*4),y*SIZE*4);
        let minAlpha=255,maxAlpha=0,maxError=0; const expected=expectedAlpha(fixture);
        for(let y=4;y<SIZE-4;y++) for(let x=4;x<SIZE-4;x++) {
          const alpha=pixels[(y*SIZE+x)*4+3]; minAlpha=Math.min(minAlpha,alpha); maxAlpha=Math.max(maxAlpha,alpha); maxError=Math.max(maxError,Math.abs(alpha-expected*255));
        }
        const centerOffset=((SIZE/2)*SIZE+SIZE/2)*4;
        result.cells.push({fixture:fixture.id,lane,expectedAlpha:expected,centerRGBA:Array.from(pixels.subarray(centerOffset,centerOffset+4)),interiorAlphaBytes:{min:minAlpha,max:maxAlpha},maxAlphaErrorBytes:maxError,
          matchesPorterDuff:maxError<=TOLERANCE_BYTES, pixelsBase64:base64(pixels), rawReadback:{bytesPerRow,byteLength:rawPixels.length,base64:base64(rawPixels)}});
      }
    }
    await device.queue.onSubmittedWorkDone();
    const scoped=await device.popErrorScope(); scopePushed=false; if(scoped) result.validationErrors.push(scoped.message);
    if(result.validationErrors.length) throw Error('WebGPU validation failure');
    // Success means the controlled experiment completed and its reference/correction gates passed.
    // Stock failure count is an observation, never an assumed test outcome.
    if(result.cells.some(c=>c.lane!=='stock'&&!c.matchesPorterDuff)) throw Error('Direct blend or proposed correction does not match alpha oracle');
    result.stockFailures=result.cells.filter(c=>c.lane==='stock'&&!c.matchesPorterDuff).map(c=>c.fixture);
    result.status='passed';
  } catch(error) { result.failure=String(error.stack??error); }
  finally {
    try {
      if(scopePushed) { const error=await device.popErrorScope(); if(error) result.validationErrors.push(error.message); }
      if(renderer) renderer.setRenderTarget(null);
      pipelines.forEach(p=>p.dispose()); passes.forEach(p=>p.dispose()); materials.forEach(m=>m.dispose()); geometry?.dispose(); target?.dispose();
      if(device) { await device.queue.onSubmittedWorkDone(); const loss=device.lost; renderer.dispose(); result.lifecycle.deviceLossReason=(await loss).reason; }
      else renderer?.dispose();
      result.lifecycle.disposed=true;
    } catch(error) { result.cleanupFailure=String(error.stack??error); result.status='failed'; }
    if(result.validationErrors.length) result.status='failed';
  }
  return result;
}
