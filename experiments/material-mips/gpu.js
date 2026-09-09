import * as THREE from 'three/webgpu';
import {Fn, storage, instanceIndex, vec3, vec4, uniform} from 'three/tsl';
import {createMetallicShading, normalFromSlopes, numericBRDF} from './shading.js';
import {gradients} from './autograd.js';

export async function createContext() {
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('WebGPU unavailable');
  const device=await adapter.requestDevice(),errors=[],shaders=[],owned=[];let disposed=false;
  device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
  device.lost.then(info=>{if(!disposed)errors.push('Device lost: '+info.message);});
  const create=device.createShaderModule.bind(device);device.createShaderModule=desc=>{shaders.push(desc.code);return create(desc);};
  const renderer=new THREE.WebGPURenderer({device,antialias:false});await renderer.init();
  renderer.outputColorSpace=THREE.LinearSRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;
  const buffer=(array,size=4,type='vec4',readOnly=false)=>{
    const attribute=new THREE.StorageBufferAttribute(array,size);owned.push(attribute);
    const node=storage(attribute,type,array.length/size);if(readOnly)node.toReadOnly();return {attribute,node};
  };
  const read=async value=>Array.from(new Float32Array(await renderer.getArrayBufferAsync(value.attribute)));
  const check=()=>{if(errors.length)throw Error(errors.join('\n'));if(disposed)throw Error('Context disposed');};
  const dispose=()=>{if(disposed)return;disposed=true;for(const resource of owned)resource.dispose?.();renderer.dispose();device.destroy();};
  return {renderer,device,buffer,read,check,dispose,errors,shaders,owned,adapter:adapter.info?{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description}:null};
}

const normalize=a=>{const d=Math.hypot(...a);return a.map(x=>x/d);};
const F0=[.65,.3,.12];
const GPU_FD_STEPS=Object.freeze([.002,.001,.0005]);
// Local differentiability classification only. Do not count a perturbation that
// crosses a DFG bilinear-cell boundary or clamp kink as a smooth derivative test.
function shadingCell(point,L){
  const N=normalize([point[0],point[1],1]),rawNL=N.reduce((s,v,i)=>s+v*L[i],0),rawNV=N[2];
  const rough=Math.min(1,Math.max(.0525,point[2])),nl=Math.max(0,Math.min(1,rawNL)),nv=Math.max(0,Math.min(1,rawNV));
  const uv=[rough,nv,rough,nl],cell=uv.map(v=>Math.floor(16*v-.5));
  const phase=v=>v<=0?'low':v>=1?'high':'inside';
  const signature=[point[2]<=.0525?'rough-low':point[2]>=1?'rough-high':'rough-inside',phase(rawNL),phase(rawNV),...cell].join('/');
  // Guard tiny CPU-vs-GPU differences in branch classification, measured in UV.
  const distanceToGrid=v=>Math.abs(16*v-.5-Math.round(16*v-.5))/16;
  const margin=Math.min(...uv.map(distanceToGrid),Math.abs(point[2]-.0525),Math.abs(point[2]-1),Math.abs(rawNL),Math.abs(rawNV));
  return {signature,margin};
}
export async function runParity() {
  const context=await createContext(),{renderer,device,buffer,read,check,owned}=context;
  const result={kind:'stock-metallic-shading-parity-v1',status:'failed',adapter:context.adapter,cases:[],lights:[],errors:context.errors,shaders:context.shaders,
    tolerances:{pixelAbsolute:2e-4,pixelRelative:2e-3,cpuAbsolute:2e-4,cpuRelative:2e-3,gradientAbsolute:.002,gradientRelative:.025},
    gpuFiniteDifferenceProtocol:{steps:GPU_FD_STEPS,acceptedStep:GPU_FD_STEPS.at(-1),absolute:.002,relative:.025,
      boundaryMargin:5e-6,denominator:'Actual Float32 plus-minus parameter separation',
      scope:'Central differences of the unchanged hardware-sampled native forward graph. Coarser steps are diagnostics; only finest-step finite, same-cell points receive the derivative gate. Crossed/near-boundary points remain separately reported, not counted as passes.'}};
  try{
    const count=128,raw=new Float32Array(count*4);
    let seed=811;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
    for(let i=0;i<count;i++)raw.set([(random()-.5)*2.6,(random()-.5)*2.6,[.055,.13,.33,.72][i%4],0],4*i);
    const params=buffer(raw,4,'vec4',true),output=buffer(new Float32Array(count*4));
    const derivativeOutput=buffer(new Float32Array(count*4));
    const p=params.node.element(instanceIndex).xyz,light=uniform(new THREE.Vector3(0,0,1)),normal=normalFromSlopes(p.x,p.y);
    const graph=createMetallicShading({normal,view:vec3(0,0,1),light,roughness:p.z,f0:vec3(...F0),inputs:[p,light]});
    const objective=graph.forward.dot(vec3(.3,.5,.2));
    const diff=gradients(objective,[p],{constants:[light,...graph.constants],replacements:graph.replacements});
    result.compiler=diff.stats;result.shading=graph.stats??null;
    const kernel=Fn(()=>{output.node.element(instanceIndex).assign(vec4(graph.forward,1));derivativeOutput.node.element(instanceIndex).assign(vec4(diff.gradients[0],objective));})().compute(count);
    const fdOffset=uniform(new THREE.Vector3()),fdPoint=params.node.element(instanceIndex).xyz.add(fdOffset);
    const fdGraph=createMetallicShading({normal:normalFromSlopes(fdPoint.x,fdPoint.y),view:vec3(0,0,1),light,roughness:fdPoint.z,
      f0:vec3(...F0),inputs:[fdPoint,light]});
    const fdOutput=buffer(new Float32Array(count*4));
    // No derivative replacements here: these are actual hardware textureSampleLevel
    // values of the same source-derived forward expression, independently perturbed.
    const fdKernel=Fn(()=>{fdOutput.node.element(instanceIndex).assign(vec4(fdGraph.forward,fdGraph.forward.dot(vec3(.3,.5,.2))));})().compute(count);
    const scene=new THREE.Scene(),camera=new THREE.OrthographicCamera(0,16,8,0,.1,10);camera.position.z=5;
    const geometry=new THREE.PlaneGeometry(.96,.96),material=new THREE.MeshStandardNodeMaterial({metalness:1,roughness:1});owned.push(geometry,material);
    material.colorNode=vec3(...F0);material.normalNode=normalFromSlopes(params.node.element(instanceIndex).x,params.node.element(instanceIndex).y);material.roughnessNode=params.node.element(instanceIndex).z;
    const mesh=new THREE.InstancedMesh(geometry,material,count),matrix=new THREE.Matrix4();
    for(let i=0;i<count;i++){matrix.makeTranslation(i%16+.5,Math.floor(i/16)+.5,0);mesh.setMatrixAt(i,matrix);}mesh.frustumCulled=false;scene.add(mesh);
    const sunlight=new THREE.DirectionalLight(0xffffff,1);scene.add(sunlight);scene.add(sunlight.target);
    const target=new THREE.RenderTarget(128,64,{type:THREE.FloatType,format:THREE.RGBAFormat,depthBuffer:false});target.texture.colorSpace=THREE.LinearSRGBColorSpace;owned.push(target);
    for(const sourceLight of [[.2,-.3,1],[-.5,.6,1],[.9,.1,.25],[0,0,1]]){
      const L=normalize(sourceLight).map(Math.fround);light.value.set(...L);sunlight.position.set(...L);
      renderer.compute(kernel);renderer.setRenderTarget(target);renderer.render(scene,camera);renderer.setRenderTarget(null);await device.queue.onSubmittedWorkDone();check();
      const compute=await read(output),derivatives=await read(derivativeOutput),pixels=Array.from(await renderer.readRenderTargetPixelsAsync(target,0,0,128,64));
      const data={light:L,compute,derivatives,pixelCenters:[],cpu:[],cpuGradients:[],mismatches:[],maxPixelError:0,maxCPUError:0,maxGradientError:0};
      for(let i=0;i<count;i++){
        const point=Array.from(raw.slice(4*i,4*i+3)),N=normalize([point[0],point[1],1]);
        const reference=numericBRDF(N,point[2],L,[0,0,1],F0);
        // WebGPU texture buffer coordinates start at the top of the image.
        const pixelIndex=((7-Math.floor(i/16))*8+4)*128+(i%16)*8+4;
        const pixel=pixels.slice(4*pixelIndex,4*pixelIndex+3);data.pixelCenters.push(...pixel);data.cpu.push(...reference);
        for(let channel=0;channel<3;channel++){
          const gpu=compute[4*i+channel],a=Math.abs(gpu-pixel[channel]),b=Math.abs(gpu-reference[channel]);
          data.maxPixelError=Math.max(data.maxPixelError,a);data.maxCPUError=Math.max(data.maxCPUError,b);
          if(!Number.isFinite(a)||a>result.tolerances.pixelAbsolute+result.tolerances.pixelRelative*Math.abs(pixel[channel]))data.mismatches.push({case:i,channel,kind:'stock-pixel',gpu,pixel:pixel[channel]});
          if(!Number.isFinite(b)||b>result.tolerances.cpuAbsolute+result.tolerances.cpuRelative*Math.abs(reference[channel]))data.mismatches.push({case:i,channel,kind:'cpu-forward',gpu,cpu:reference[channel]});
        }
        for(let parameter=0;parameter<3;parameter++){
          const plus=[...point],minus=[...point],h=1e-5;plus[parameter]+=h;minus[parameter]-=h;
          const scalar=q=>numericBRDF(normalize([q[0],q[1],1]),q[2],L,[0,0,1],F0).reduce((s,v,k)=>s+v*[.3,.5,.2][k],0);
          const expected=(scalar(plus)-scalar(minus))/(2*h),error=Math.abs(derivatives[4*i+parameter]-expected);data.cpuGradients.push(expected);data.maxGradientError=Math.max(data.maxGradientError,error);
          if(!Number.isFinite(error)||error>result.tolerances.gradientAbsolute+result.tolerances.gradientRelative*Math.abs(expected))data.mismatches.push({case:i,parameter,kind:'gradient',gpu:derivatives[4*i+parameter],expected});
        }
      }
      data.gpuFiniteDifferences=[];
      data.gpuFiniteDifferenceSummary={finestSmoothChecked:0,finestCrossedOrNearBoundary:0,nonfinite:0,failures:0,maxSmoothError:0};
      for(const h of GPU_FD_STEPS)for(let parameter=0;parameter<3;parameter++){
        const readPerturbation=async sign=>{
          fdOffset.value.set(0,0,0);fdOffset.value.setComponent(parameter,Math.fround(sign*h));
          renderer.compute(fdKernel);await device.queue.onSubmittedWorkDone();check();return read(fdOutput);
        };
        const plus=await readPerturbation(1),minus=await readPerturbation(-1);
        const entry={h,parameter,plus,minus,plusParameters:[],minusParameters:[],comparisons:[]};
        const finest=h===GPU_FD_STEPS.at(-1);
        for(let i=0;i<count;i++){
          const center=Array.from(raw.slice(4*i,4*i+3)),a=[...center],b=[...center];
          a[parameter]=Math.fround(a[parameter]+Math.fround(h));b[parameter]=Math.fround(b[parameter]-Math.fround(h));
          entry.plusParameters.push(...a);entry.minusParameters.push(...b);
          const states=[shadingCell(center,L),shadingCell(a,L),shadingCell(b,L)];
          const crossed=states.some(s=>s.signature!==states[0].signature),nearBoundary=states.some(s=>s.margin<5e-6);
          const denominator=a[parameter]-b[parameter],value=(plus[4*i+3]-minus[4*i+3])/denominator;
          const ad=derivatives[4*i+parameter],error=Math.abs(value-ad),tolerance=.002+.025*Math.abs(ad);
          const finite=denominator>0&&[...plus.slice(4*i,4*i+4),...minus.slice(4*i,4*i+4),value,ad,error].every(Number.isFinite);
          const checked=finest&&!crossed&&!nearBoundary,passed=finite&&(!checked||error<=tolerance);
          entry.comparisons.push({case:i,denominator,finite,ad,finiteDifference:value,error,tolerance,crossedCellOrClamp:crossed,
            nearBoundary,cellSignatures:states.map(s=>s.signature),minimumBoundaryMargin:Math.min(...states.map(s=>s.margin)),
            checked,passed:checked?passed:null});
          if(!finite){data.gpuFiniteDifferenceSummary.nonfinite++;data.mismatches.push({case:i,parameter,h,kind:'gpu-fd-nonfinite'});}
          if(finest){
            if(crossed||nearBoundary)data.gpuFiniteDifferenceSummary.finestCrossedOrNearBoundary++;
            else{
              data.gpuFiniteDifferenceSummary.finestSmoothChecked++;
              data.gpuFiniteDifferenceSummary.maxSmoothError=Math.max(data.gpuFiniteDifferenceSummary.maxSmoothError,error);
              if(!passed){data.gpuFiniteDifferenceSummary.failures++;data.mismatches.push({case:i,parameter,h,kind:'gpu-forward-fd',ad,finiteDifference:value,error,tolerance});}
            }
          }
        }
        data.gpuFiniteDifferences.push(entry);
      }
      if(data.gpuFiniteDifferenceSummary.finestSmoothChecked===0)data.mismatches.push({kind:'gpu-fd-no-smooth-coverage'});
      result.lights.push(data);
    }
    result.parameters=Array.from(raw);result.mismatchCount=result.lights.reduce((s,l)=>s+l.mismatches.length,0);result.status=result.mismatchCount?'mismatch':'passed';
  }catch(error){result.failure=String(error.stack??error);}
  finally{context.dispose();result.disposed=true;}
  return result;
}
