import * as THREE from 'three/webgpu';
import {Fn, storage, instanceIndex, positionLocal, vec3, uniform, float, select, wgslFn, globalId, mix, clamp} from 'three/tsl';
import {gradients} from './autograd.js';
import {FIELDS, field, initialWeights, numericField, makeSamples, loss, heldout} from './fields.js';

export const LIMITS=Object.freeze({samples:128,iterations:160,wallMs:20000,epsilon:.001});

export async function createTrainer({kind='neural',seed=11,container=null,customSamples=null}={}) {
  if(!FIELDS[kind] || !Number.isInteger(seed))throw Error('Unknown fixture');
  const errors=[],shaders=[],attributes=[],materials=[],geometries=[];
  let device,renderer,disposed=false,stopped=false,busy=false,renderPending=false;
  const count=FIELDS[kind].count, sampleCount=LIMITS.samples;
  if(customSamples && (customSamples.length!==sampleCount*4 || Array.from(customSamples).some(x=>!Number.isFinite(x)||Math.abs(x)>2)))throw Error('Expected 128 finite sample vec4s within [-2,2]');
  const initial=initialWeights(kind,seed), samples=customSamples?new Float32Array(customSamples):makeSamples(kind,seed,sampleCount);
  try {
    if(!navigator.gpu)throw Error('WebGPU is unavailable in this browser');
    const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No WebGPU adapter');
    device=await adapter.requestDevice();
    device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
    device.lost.then(info=>{if(!disposed)errors.push('Device lost: '+info.message);});
    const original=device.createShaderModule.bind(device);
    device.createShaderModule=desc=>{shaders.push(desc.code);return original(desc);};
    renderer=new THREE.WebGPURenderer({device,antialias:true});await renderer.init();
    renderer.setPixelRatio(1);renderer.setSize(1100,600);renderer.setClearColor(0x08131c);
    container?.appendChild(renderer.domElement);
    const make=(array,size,type,readOnly=false)=>{
      const a=new THREE.StorageBufferAttribute(array,size);attributes.push(a);
      const s=storage(a,type,array.length/size);return {a,s:readOnly?s.toReadOnly():s};
    };
    const stateArray=new Float32Array(count*4);initial.forEach((v,i)=>stateArray[4*i]=v);
    // The immutable reset image must not alias the mutable upload array. In
    // particular, importing learned weights must not change another lane's start.
    const weights=make(stateArray.slice(),4,'vec4');
    const observations=make(samples,4,'vec4',true);
    const partials=make(new Float32Array(count*sampleCount),1,'float');
    const predictions=make(new Float32Array(sampleCount),1,'float');
    const source=observations.s.element(instanceIndex),x=source.x,z=source.y,target=source.z,p=vec3(x,0,z);
    const weightRead=storage(weights.a,'vec4',count).toReadOnly();
    const w=Array.from({length:count},(_,i)=>weightRead.element(i).x);
    const forward=field(kind,p,w),residual=forward.sub(target),objective=residual.mul(residual);
    const compiled=gradients(objective,w,{constants:[x,z,target]});
    const ad=Fn(()=>{
      predictions.s.element(instanceIndex).assign(forward);
      for(let j=0;j<count;j++)partials.s.element(instanceIndex.mul(count).add(j)).assign(compiled.gradients[j]);
    })().compute(sampleCount);

    // Strong control: central finite differences of the SAME native forward field,
    // on the GPU, with the SAME observations, reduction and Adam update.
    const sampleId=instanceIndex.div(count).toUint(),parameterId=instanceIndex.mod(count);
    const fdSource=observations.s.element(sampleId),fdP=vec3(fdSource.x,0,fdSource.y);
    const perturb=sign=>w.map((value,j)=>value.add(select(parameterId.equal(j),sign*LIMITS.epsilon,0)));
    const plus=field(kind,fdP,perturb(1)).sub(fdSource.z),minus=field(kind,fdP,perturb(-1)).sub(fdSource.z);
    const fd=Fn(()=>{
      partials.s.element(instanceIndex).assign(plus.mul(plus).sub(minus.mul(minus)).div(2*LIMITS.epsilon));
    })().compute(sampleCount*count);
    const iteration=uniform(1,'uint');
    const updateFn=wgslFn(`fn trainableUpdate(id:vec3u, count:u32, samples:u32, step:u32, rate:f32,
      partials:ptr<storage,array<f32>,read>, weights:ptr<storage,array<vec4f>,read_write>) -> void {
      if(id.x>=count){return;} var g=0.0;
      for(var i=0u;i<samples;i++){g+=partials[i*count+id.x];} g/=f32(samples);
      let old=weights[id.x]; let m=0.9*old.y+0.1*g;let v=0.999*old.z+0.001*g*g;
      let correctedM=m/(1.0-pow(0.9,f32(step)));let correctedV=v/(1.0-pow(0.999,f32(step)));
      let next=clamp(old.x-rate*correctedM/(sqrt(correctedV)+1e-8),-8.0,8.0);
      weights[id.x]=vec4f(next,m,v,0.0);
    }`);
    const update=updateFn({id:globalId,count,samples:sampleCount,step:iteration,rate:FIELDS[kind].rate,
      partials:storage(partials.a,'float',count*sampleCount).toReadOnly(),weights:weights.s}).computeKernel([64,1,1]);
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1100/600,.1,40);
    camera.position.set(0,4.5,7.8);camera.lookAt(0,.1,0);
    for(const [index,offset] of [-1.45,1.45].entries()) {
      const geometry=new THREE.PlaneGeometry(2,2,64,64).rotateX(-Math.PI/2);geometries.push(geometry);
      const material=new THREE.MeshBasicNodeMaterial({side:THREE.DoubleSide});materials.push(material);
      const previewWeights=index?w:Array.from(initial,v=>float(v));
      const y=field(kind,positionLocal,previewWeights);
      material.positionNode=vec3(positionLocal.x,y,positionLocal.z);
      // A height contour palette makes changes visible without relying on stale
      // geometry normals for GPU displacement. This is a display shader only.
      const t=clamp(y.add(.4).div(1.45),0,1);
      material.colorNode=mix(vec3(.025,.20,.29),vec3(.54,.99,.75),t);
      const mesh=new THREE.Mesh(geometry,material);mesh.position.x=offset;mesh.frustumCulled=false;scene.add(mesh);
      const wireMat=new THREE.MeshBasicNodeMaterial({color:0x78ebda,wireframe:true,transparent:true,opacity:.12});materials.push(wireMat);
      wireMat.positionNode=material.positionNode;
      const wire=new THREE.Mesh(geometry,wireMat);wire.position.x=offset;wire.position.y=.002;wire.frustumCulled=false;scene.add(wire);
      const grid=new THREE.GridHelper(2.4,12,0x254858,0x152b39);grid.position.set(offset,-.42,0);scene.add(grid);geometries.push(grid.geometry);materials.push(grid.material);
    }
    const dotsGeometry=new THREE.SphereGeometry(.025,6,4);geometries.push(dotsGeometry);
    const dotsMaterial=new THREE.MeshBasicNodeMaterial({color:0xffc47c});materials.push(dotsMaterial);
    const dots=new THREE.InstancedMesh(dotsGeometry,dotsMaterial,sampleCount),matrix=new THREE.Matrix4();
    for(let i=0;i<sampleCount;i++){matrix.makeTranslation(samples[4*i]+1.45,samples[4*i+2]+.018,samples[4*i+1]);dots.setMatrixAt(i,matrix);}scene.add(dots);
    function check(){if(errors.length)throw Error(errors.join('\n'));if(disposed)throw Error('Trainer disposed');}
    function render(){if(disposed||renderPending)return;renderPending=true;try{renderer.render(scene,camera);}finally{renderPending=false;}}
    function resize(width=1100,height=600){if(disposed)return;renderer.setSize(Math.min(1400,width),Math.min(800,height));camera.aspect=Math.min(1400,width)/Math.min(800,height);camera.updateProjectionMatrix();render();}
    async function readWeights(){return Array.from(new Float32Array(await renderer.getArrayBufferAsync(weights.a))).filter((_,i)=>i%4===0);}
    async function reset(){weights.a.array.set(stateArray);weights.a.needsUpdate=true;renderer.compute(ad);await device.queue.onSubmittedWorkDone();check();}
    async function inspect(method='ad'){
      renderer.compute(method==='ad'?ad:fd);await device.queue.onSubmittedWorkDone();check();
      return Array.from(new Float32Array(await renderer.getArrayBufferAsync(partials.a)));
    }
    async function validate({resetFirst=true,checkUpdate=true}={}){
      if(busy)throw Error('Trainer busy');busy=true;
      try {
        if(resetFirst)await reset();const point=await readWeights(),actual=await inspect('ad'),finite=await inspect('fd');
        const comparison={count:actual.length,maxADAbsolute:0,maxFDAbsolute:0,adFailures:0,fdFailures:0,primalMaxError:0,weights:point,adGradients:actual,fdGradients:finite,cpuGradients:[]};
        const rawPred=new Float32Array(await renderer.getArrayBufferAsync(predictions.a));
        const h=1e-5;
        for(let i=0;i<sampleCount;i++) {
          const x=samples[4*i],z=samples[4*i+1],y=samples[4*i+2];
          comparison.primalMaxError=Math.max(comparison.primalMaxError,Math.abs(rawPred[i]-numericField(kind,x,z,point)));
          for(let j=0;j<count;j++){
            const a=Array.from(point),b=Array.from(point);a[j]+=h;b[j]-=h;
            const oracle=((numericField(kind,x,z,a)-y)**2-(numericField(kind,x,z,b)-y)**2)/(2*h);
            comparison.cpuGradients.push(oracle);
            const da=Math.abs(actual[i*count+j]-oracle),df=Math.abs(finite[i*count+j]-oracle);
            comparison.maxADAbsolute=Math.max(comparison.maxADAbsolute,da);comparison.maxFDAbsolute=Math.max(comparison.maxFDAbsolute,df);
            if(!Number.isFinite(da)||da>2e-4+2e-3*Math.abs(oracle))comparison.adFailures++;
            if(!Number.isFinite(df)||df>3e-3+5e-3*Math.abs(oracle))comparison.fdFailures++;
          }
        }
        comparison.passed=comparison.adFailures===0&&comparison.fdFailures===0&&comparison.primalMaxError<2e-5;
        if(checkUpdate){
          if(!resetFirst)throw Error('Adam first-step check requires reset');
          renderer.compute(ad);iteration.value=1;renderer.compute(update,[Math.ceil(count/64),1,1]);await device.queue.onSubmittedWorkDone();
          const gpuState=Array.from(new Float32Array(await renderer.getArrayBufferAsync(weights.a)));
          const expected=[],meanGradients=[],failures=[];
          for(let j=0;j<count;j++){
            let mean=0;for(let i=0;i<sampleCount;i++)mean+=actual[i*count+j];mean/=sampleCount;meanGradients.push(mean);
            const next=Math.max(-8,Math.min(8,point[j]-FIELDS[kind].rate*mean/(Math.abs(mean)+1e-8)));
            const state=[next,.1*mean,.001*mean*mean,0];expected.push(...state);
            for(let k=0;k<4;k++)if(!Number.isFinite(gpuState[4*j+k])||Math.abs(gpuState[4*j+k]-state[k])>(k===0?5e-4:2e-6))failures.push({parameter:j,component:k,actual:gpuState[4*j+k],expected:state[k]});
          }
          comparison.adam={passed:failures.length===0,failures,meanGradients,gpuState,expected};comparison.passed&&=comparison.adam.passed;
          await reset();
        }
        check();return comparison;
      }finally{busy=false;}
    }
    async function train({method='ad',steps=LIMITS.iterations,onProgress=null}={}){
      if(busy)throw Error('Trainer busy');if(!['ad','fd'].includes(method)||!Number.isInteger(steps)||steps<1||steps>LIMITS.iterations)throw Error('Training budget exceeded');
      busy=true;stopped=false;const started=performance.now();let completed=0;
      try {
        await reset();
        const startingWeights=await readWeights();
        if(startingWeights.some((value,i)=>value!==initial[i]))throw Error('GPU reset did not restore the declared initial weights');
        const initialMetrics={trainingRMSE:Math.sqrt(loss(kind,initial,samples)),heldout:customSamples?null:heldout(kind,initial)};
        const fitStart=performance.now();
        for(let i=0;i<steps&&!stopped;i++){
          if(performance.now()-started>LIMITS.wallMs){stopped=true;break;}
          iteration.value=i+1;renderer.compute(method==='ad'?ad:fd);renderer.compute(update,[Math.ceil(count/64),1,1]);
          // Finish this update before changing its uniform for the next submission.
          await device.queue.onSubmittedWorkDone();completed++;
          if(i%8===7){if(container)render();onProgress?.({iteration:completed,steps});await new Promise(resolve=>setTimeout(resolve,0));check();}
        }
        await device.queue.onSubmittedWorkDone();const completedWallMs=performance.now()-fitStart;
        const final=await readWeights();check();render();
        if(final.some(x=>!Number.isFinite(x)))throw Error('Nonfinite learned weights');
        return {kind,seed,method,completed,requested:steps,stopped,completedWallMs,
          initial:initialMetrics,final:{trainingRMSE:Math.sqrt(loss(kind,final,samples)),heldout:customSamples?null:heldout(kind,final)},
          weights:final,initialWeights:Array.from(initial),startingWeights,samples:Array.from(samples),
          customSamples:!!customSamples,utilityPassed:customSamples?null:completed===steps&&heldout(kind,final).rmse<=.2*initialMetrics.heldout.rmse,
          gpuGradientInvocationsPerStep:method==='ad'?sampleCount:sampleCount*count,
          timingScope:'CPU-observed completed training, including queue synchronization, progress callbacks and any requested preview renders; not active GPU time or a general speed claim'};
      }finally{busy=false;}
    }
    async function loadWeights(values){
      if(busy)throw Error('Trainer busy');if(!Array.isArray(values)||values.length!==count||values.some(v=>!Number.isFinite(v)||Math.abs(v)>8))throw Error('Invalid weight vector');
      const next=new Float32Array(count*4);values.forEach((v,i)=>next[4*i]=v);weights.a.array.set(next);weights.a.needsUpdate=true;
      const actual=await inspect('ad');render();return {weights:await readWeights(),gradients:actual};
    }
    function stop(){stopped=true;}
    function dispose(){if(disposed)return;stopped=true;disposed=true;for(const a of attributes)a.dispose();for(const m of materials)m.dispose();for(const g of geometries)g.dispose();renderer?.dispose();device?.destroy();renderer?.domElement.remove();}
    return {kind,seed,count,stats:compiled.stats,errors,shaders,validate,train,stop,dispose,render,resize,readWeights,loadWeights,
      adapter:adapter.info?{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description}:null,
      get status(){return {busy,stopped,disposed};}};
  } catch(error){disposed=true;for(const a of attributes)a.dispose();for(const m of materials)m.dispose();for(const g of geometries)g.dispose();renderer?.dispose();device?.destroy();throw error;}
}
