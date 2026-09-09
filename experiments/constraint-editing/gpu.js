import * as THREE from 'three/webgpu';
import {Fn, storage, instanceIndex, vec4, select} from 'three/tsl';
import {gradients} from '../material-mips/autograd.js';
import {rebind} from './rebind.js';

export const GPU_LIMITS = Object.freeze({rows:4096, parameters:16, epsilon:.001});

/** Bind one already-authored pure native graph to a batched GPU evaluator.
 * No formula callback is invoked here. The caller supplies exact opaque inputs.
 * Flat returned Jacobians have layout [sample][xyz][parameter]. */
export async function createProgramEvaluator({root, position, time, parameters}) {
  if(!parameters.length || parameters.length>GPU_LIMITS.parameters)throw Error('Parameter budget exceeded');
  const n=parameters.length,errors=[],shaders=[],owned=[];
  let renderer,device,disposed=false,busy=false;
  try{
    const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('WebGPU unavailable');
    device=await adapter.requestDevice();
    device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
    device.lost.then(info=>{if(!disposed)errors.push('Device lost: '+info.message);});
    const create=device.createShaderModule.bind(device);device.createShaderModule=desc=>{shaders.push(desc.code);return create(desc);};
    renderer=new THREE.WebGPURenderer({device,antialias:true});await renderer.init();
    renderer.setPixelRatio(1);
    const buffer=(array,size=4,type='vec4',readOnly=false)=>{
      const attribute=new THREE.StorageBufferAttribute(array,size);owned.push(attribute);
      const node=storage(attribute,type,array.length/size);if(readOnly)node.toReadOnly();return {attribute,node};
    };
    const weights=buffer(new Float32Array(n*4),4,'vec4',true);
    const samples=buffer(new Float32Array(GPU_LIMITS.rows*4),4,'vec4',true);
    const positions=buffer(new Float32Array(GPU_LIMITS.rows*4));
    const derivatives=buffer(new Float32Array(GPU_LIMITS.rows*n*4));
    const w=parameters.map((_,j)=>weights.node.element(j).x);
    function bind(p,t,values=w){
      if(values.length!==n)throw Error('Wrong binding width');
      return rebind(root,new Map([[position,p],[time,t],...parameters.map((old,j)=>[old,values[j]])]));
    }
    const sample=samples.node.element(instanceIndex),p=sample.xyz,t=sample.w;
    const forward=bind(p,t);
    const axes=['x','y','z'],diff=axes.map(axis=>gradients(forward[axis],w,{constants:[p,t]}));
    const primal=Fn(()=>{positions.node.element(instanceIndex).assign(vec4(forward,1));})().compute(1);
    const ad=Fn(()=>{
      positions.node.element(instanceIndex).assign(vec4(forward,1));
      for(let j=0;j<n;j++)derivatives.node.element(instanceIndex.mul(n).add(j)).assign(vec4(...diff.map(d=>d.gradients[j]),0));
    })().compute(1);
    const row=instanceIndex.div(n).toUint(),parameter=instanceIndex.mod(n),fdSample=samples.node.element(row);
    const fdP=fdSample.xyz,fdT=fdSample.w;
    const plus=bind(fdP,fdT,w.map((v,j)=>v.add(select(parameter.equal(j),GPU_LIMITS.epsilon,0))));
    const minus=bind(fdP,fdT,w.map((v,j)=>v.sub(select(parameter.equal(j),GPU_LIMITS.epsilon,0))));
    const fd=Fn(()=>{derivatives.node.element(instanceIndex).assign(vec4(plus.sub(minus).div(2*GPU_LIMITS.epsilon),0));})().compute(n);
    owned.push(primal,ad,fd);
    const check=()=>{if(disposed)throw Error('Evaluator disposed');if(errors.length)throw Error(errors.join('\n'));};
    function load(values){
      check();if(values.length!==n||Array.from(values).some(v=>!Number.isFinite(v)||Math.abs(v)>100))throw Error('Invalid parameters');
      values.forEach((v,j)=>weights.attribute.array[4*j]=v);weights.attribute.needsUpdate=true;
    }
    async function evaluate(values,records,{jacobian:wantDerivatives=true,method='ad'}={}){
      if(busy)throw Error('Concurrent GPU evaluation');check();
      const rows=records.length/4;
      if(!['ad','fd'].includes(method)||!Number.isInteger(rows)||rows<1||rows>GPU_LIMITS.rows)throw Error('Invalid batch');
      if(Array.from(records).some(v=>!Number.isFinite(v)))throw Error('Samples must be finite [s,y,z,time] records');
      busy=true;const started=performance.now();
      try{
        load(values);samples.attribute.array.set(records);samples.attribute.needsUpdate=true;
        primal.count=ad.count=rows;fd.count=rows*n;
        if(wantDerivatives&&method==='ad')renderer.compute(ad);
        else {renderer.compute(primal);if(wantDerivatives)renderer.compute(fd);}
        await device.queue.onSubmittedWorkDone();check();
        const raw=new Float32Array(await renderer.getArrayBufferAsync(positions.attribute,null,0,rows*16));
        const result={positions:new Float64Array(rows*3),jacobians:null,completedWallMs:0};
        for(let i=0;i<rows;i++)result.positions.set(raw.subarray(4*i,4*i+3),3*i);
        if(wantDerivatives){
          const d=new Float32Array(await renderer.getArrayBufferAsync(derivatives.attribute,null,0,rows*n*16));
          result.jacobians=new Float64Array(rows*3*n);
          for(let i=0;i<rows;i++)for(let a=0;a<3;a++)for(let j=0;j<n;j++)result.jacobians[(i*3+a)*n+j]=d[(i*n+j)*4+a];
        }
        if(result.positions.some(v=>!Number.isFinite(v))||result.jacobians?.some(v=>!Number.isFinite(v)))throw Error('Nonfinite GPU output');
        result.completedWallMs=performance.now()-started;return result;
      }finally{busy=false;}
    }
    function dispose(){if(disposed)return;disposed=true;for(const resource of owned)resource.dispose?.();renderer.dispose();device.destroy();renderer.domElement.remove();}
    async function readParameters(){check();const raw=new Float32Array(await renderer.getArrayBufferAsync(weights.attribute));return Array.from({length:n},(_,j)=>raw[4*j]);}
    return {renderer,device,owned,evaluate,load,bind,dispose,check,readParameters,errors,shaders,stats:diff.map(d=>d.stats),
      adapter:adapter.info?{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description}:null,
      get status(){return {busy,disposed};}};
  }catch(error){disposed=true;for(const resource of owned)resource.dispose?.();renderer?.dispose();device?.destroy();throw error;}
}
