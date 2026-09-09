import * as THREE from 'three/webgpu';
import { GpuImageLoss } from './gpu-loss.js';
import { GpuNormalEquations } from './gpu-equations.js';

export const physicalValues=(spec,normalized)=>normalized.map((v,i)=>spec.parameters[i].min+v*(spec.parameters[i].max-spec.parameters[i].min));

export async function createEngine(spec,container){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('WebGPU adapter unavailable');
  const device=await adapter.requestDevice(),errors=[];
  let disposed=false;
  device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
  device.lost.then(info=>{if(!disposed)errors.push('Unexpected device loss: '+info.message);});
  let renderer;
  try {
  renderer=new THREE.WebGPURenderer({device,antialias:false});await renderer.init();
  // A fixed order avoids changing coincident-surface visibility when dimensions
  // are restored. This fixture is about dimensions, not sorting performance.
  renderer.sortObjects=false;
  renderer.setPixelRatio(1);renderer.setSize(1200,340);renderer.setClearColor(0x101723,1);
  renderer.toneMapping=THREE.NoToneMapping;
  container?.appendChild(renderer.domElement);
  const factory=spec.build(THREE);
  const camera=new THREE.PerspectiveCamera(38,1,.1,40);camera.position.set(4,3,5);camera.lookAt(0,.7,0);
  const heldout=camera.clone();heldout.position.set(-4,2.6,4);heldout.lookAt(0,.7,0);
  const displayCamera=camera.clone();displayCamera.aspect=400/340;displayCamera.updateProjectionMatrix();
  const displayHeldout=heldout.clone();displayHeldout.aspect=400/340;displayHeldout.updateProjectionMatrix();
  const targetOptions={type:THREE.UnsignedByteType,format:THREE.RGBAFormat,depthBuffer:true,samples:0};
  const reference=new THREE.RenderTarget(128,128,targetOptions),validationReference=new THREE.RenderTarget(128,128,targetOptions),candidate=new THREE.RenderTarget(128,128,targetOptions);
  function render(values,view,target){
    if(errors.length)throw Error(errors.join('; '));
    factory.update(values);renderer.setScissorTest(false);renderer.setRenderTarget(target);renderer.render(factory.scene,view);
  }
  render(spec.target,camera,reference);render(spec.target,heldout,validationReference);render(spec.target,camera,candidate);
  await device.queue.onSubmittedWorkDone();
  const native=target=>renderer.backend.get(target.texture).texture;
  const trainingLoss=new GpuImageLoss(device,native(reference),native(candidate));
  const validationLoss=new GpuImageLoss(device,native(validationReference),native(candidate));
  const centerImage=device.createTexture({size:[128,128],format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.TEXTURE_BINDING});
  const equations=new GpuNormalEquations(device,native(reference));
  function saveCenter(){
    const encoder=device.createCommandEncoder();encoder.copyTextureToTexture({texture:native(candidate)},{texture:centerImage},[128,128]);device.queue.submit([encoder.finish()]);
  }
  async function captureEquations(center,steps,{validate=false,onRender=()=>{}}={}){
    const denominators=[],images=[];
    // centerImage is the already evaluated best candidate. No hidden center render.
    for(let i=0;i<8;i++){
      const minus=Math.max(0,center[i]-steps[i]),plus=Math.min(1,center[i]+steps[i]);denominators.push(plus-minus);
      for(const [side,value]of [minus,plus].entries()){
        const vector=[...center];vector[i]=value;const start=performance.now();render(physicalValues(spec,vector),camera,candidate);
        const encoder=device.createCommandEncoder();equations.capture(encoder,native(candidate),2*i+side);device.queue.submit([encoder.finish()]);
        if(validate)images.push(await renderer.readRenderTargetPixelsAsync(candidate,0,0,128,128));
        await onRender(vector,performance.now()-start);
      }
    }
    const result=await equations.evaluate(centerImage,denominators);
    if(validate){
      // Restore only in correctness validation, outside any fitting budget.
      render(physicalValues(spec,center),camera,candidate);
      const [referencePixels,centerPixels]=await Promise.all([renderer.readRenderTargetPixelsAsync(reference,0,0,128,128),renderer.readRenderTargetPixelsAsync(candidate,0,0,128,128)]);
      const hessian=Array(64).fill(0),gradient=Array(8).fill(0),j=Array(8);
      for(let p=0;p<128*128;p++)for(let c=0;c<3;c++){
        const offset=4*p+c,residual=(centerPixels[offset]-referencePixels[offset])/255;
        for(let i=0;i<8;i++)j[i]=(images[2*i+1][offset]-images[2*i][offset])/(255*denominators[i]);
        for(let i=0;i<8;i++){gradient[i]+=j[i]*residual;for(let k=0;k<8;k++)hessian[8*i+k]+=j[i]*j[k];}
      }
      for(let i=0;i<64;i++)hessian[i]/=128*128*3;for(let i=0;i<8;i++)gradient[i]/=128*128*3;
      const deltas=[...hessian.map((v,i)=>Math.abs(v-result.hessian[i])),...gradient.map((v,i)=>Math.abs(v-result.gradient[i]))];
      const passed=[...hessian,...gradient].every((v,i)=>deltas[i]<=1e-6+1e-4*Math.abs(v));
      result.parity={passed,maxAbsoluteDifference:Math.max(...deltas),cpu:{hessian,gradient},readbackOutsideFitting:true,perturbationRenders:16,restorationRenders:1};
      if(!passed)throw Error('GPU/CPU normal equation disagreement: '+Math.max(...deltas));
    }
    return result;
  }
  async function evaluate(values,validation=false){
    const start=performance.now();render(values,validation?heldout:camera,candidate);
    const loss=await(validation?validationLoss:trainingLoss).evaluate();
    return {loss,completionMs:performance.now()-start};
  }
  async function validate(values){
    render(values,camera,candidate);const gpu=await trainingLoss.evaluate();
    const [a,b]=await Promise.all([
      renderer.readRenderTargetPixelsAsync(reference,0,0,128,128),
      renderer.readRenderTargetPixelsAsync(candidate,0,0,128,128)
    ]);
    let total=0;for(let i=0;i<a.length;i+=4)for(let c=0;c<3;c++){const d=(a[i+c]-b[i+c])/255;total+=d*d;}
    const cpu=total/(128*128*3),difference=Math.abs(cpu-gpu);
    if(difference>1e-7)throw Error('GPU/CPU loss disagreement: '+difference);
    render(spec.target,camera,candidate);const self=await trainingLoss.evaluate();if(self!==0){
      const targetPixels=await renderer.readRenderTargetPixelsAsync(candidate,0,0,128,128);
      let maxByte=0,changed=0;for(let i=0;i<a.length;i++){const d=Math.abs(a[i]-targetPixels[i]);maxByte=Math.max(maxByte,d);if(d)changed++;}
      throw Error(`Target self-loss is not zero: ${self}; max byte difference ${maxByte}; changed channels ${changed}`);
    }
    return {cpu,gpu,absoluteDifference:difference,targetSelfLoss:self,readbackOutsideFitting:true};
  }
  function display(initial,best,view='train'){
    renderer.setRenderTarget(null);renderer.setScissorTest(true);
    for(const [i,values]of [spec.target,initial,best].entries()){
      factory.update(values);renderer.setViewport(i*400,0,400,340);renderer.setScissor(i*400,0,400,340);renderer.render(factory.scene,view==='heldout'?displayHeldout:displayCamera);
    }
    renderer.setScissorTest(false);
  }
  async function dispose(){
    if(disposed)return;await device.queue.onSubmittedWorkDone();
    trainingLoss.dispose();validationLoss.dispose();equations.dispose();centerImage.destroy();reference.dispose();validationReference.dispose();candidate.dispose();factory.dispose();
    await renderer.dispose();disposed=true;device.destroy();
  }
  return {renderer,device,errors,evaluate,validate,display,dispose,spec,saveCenter,captureEquations,
    environment:{threeRevision:THREE.REVISION,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},resolution:[128,128]},
    get disposed(){return disposed;}};
  } catch(error) {
    disposed=true;
    try { await renderer?.dispose(); } catch { /* Device destruction is the final release. */ }
    device.destroy();
    throw error;
  }
}
