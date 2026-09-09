import * as THREE from 'three/webgpu';
import {Fn, storage, instanceIndex, vertexIndex, positionLocal, select, vec3, vec4, uniform, sin, cos, wgslFn, globalId} from 'three/tsl';
import {BVHComputeData} from 'three-mesh-bvh/webgpu';
import {GPUBVHRefitter} from './refitter.js';
import {bruteForceQueries, closestPointTriangle, validateBounds, validateMetadataUnchanged} from './reference.mjs';

// These are native TSL graphs. The same graph supplies vertex deformation and,
// when enabled, the interval compiler. There is no CPU deformation copy.
export function deformation(p,t,family) {
  if(family==='wave') return vec3(p.x,p.y.add(sin(p.x.mul(2.1).add(t)).mul(.65)),p.z);
  if(family==='twist') {
    const angle=p.y.mul(1.2).add(t.mul(.6)),c=cos(angle),s=sin(angle);
    return vec3(p.x.mul(c).sub(p.z.mul(s)),p.y,p.x.mul(s).add(p.z.mul(c)));
  }
  if(family==='ripple') return vec3(p.x,p.y.add(sin(p.x.mul(5).add(t)).mul(cos(p.z.mul(4).sub(t))).mul(.45)),p.z);
  throw Error('Unknown deformation family');
}

export async function createEngine({family='wave',segments=16,queryCount=128,container=null}={}) {
  if(!['wave','twist','ripple'].includes(family)||![8,16,32,40].includes(segments)||!Number.isInteger(queryCount)||queryCount<1||queryCount>4096)throw Error('Unsupported bounded fixture');
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('WebGPU unavailable');
  const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('timestamp-query')?['timestamp-query']:[]}),errors=[];let disposed=false,renderer,refitter,bvh;
  const shaderSources=[],createShaderModule=device.createShaderModule.bind(device);
  device.createShaderModule=descriptor=>{shaderSources.push({label:descriptor.label??'',code:descriptor.code});return createShaderModule(descriptor);};
  device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
  device.lost.then(info=>{if(!disposed)errors.push('Unexpected device loss: '+info.message);});
  const owned=[];
  try {
    renderer=new THREE.WebGPURenderer({device,antialias:false});await renderer.init();
    renderer.setPixelRatio(1);renderer.setSize(1100,600);renderer.setClearColor(0x090f1c);
    container?.appendChild(renderer.domElement);
    const geometry=family==='twist'?new THREE.TorusGeometry(1,.35,Math.max(8,segments/2),segments):new THREE.PlaneGeometry(4,4,segments,segments).rotateX(-Math.PI/2);
    const mesh=new THREE.Mesh(geometry);mesh.updateMatrixWorld(true);
    bvh=new BVHComputeData(mesh);bvh.update();
    if(!bvh.getRootObject().matrixWorld.equals(new THREE.Matrix4()))throw Error('Only identity TLAS roots are supported');
    const attr=Object.fromEntries(['nodes','index','attributes','transforms'].map(k=>[k,bvh.storage[k].proxyNode.value]));
    const nodeArray=attr.nodes.array.slice(),indexArray=attr.index.array.slice();
    const baseVertices=new Float32Array(attr.attributes.array.buffer.slice(0)),vertexCount=baseVertices.length/4;
    const makeStorage=(array,size,type,readOnly=false)=>{const a=new THREE.StorageBufferAttribute(array,size);owned.push(a);const s=storage(a,type,array.length/size);return {a,s:readOnly?s.toReadOnly():s};};
    const base=makeStorage(baseVertices,4,'vec4',true),vertices=storage(attr.attributes,'vec4',vertexCount);
    const p=base.s.element(instanceIndex).xyz,t=uniform(0),graph=deformation(p,t,family);
    const deform=Fn(()=>{vertices.element(instanceIndex).assign(vec4(graph,1));})().compute(vertexCount);
    const queries=new Float32Array(queryCount*4);
    let seed=4242;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
    for(let i=0;i<queryCount;i++)queries.set([(random()-.5)*4.6,(random()-.5)*2.8,(random()-.5)*4.6,0],i*4);
    const designed=[[0,0,0],[0,.001,0],[0,-.001,0],[2,0,2],[-2,0,-2],[2,0,-2],[-2,0,2],[0,2,0],[0,-2,0],[4,0,0],[-4,0,0],[0,0,4],[0,0,-4],[1.35,0,0],[.65,0,0],[1,0,.35]];
    for(let i=0;i<Math.min(designed.length,queryCount);i++)queries.set([...designed[i],0],i*4);
    const query=makeStorage(queries,4,'vec4',true),output=makeStorage(new Float32Array(queryCount*4),4,'vec4'),faceOutput=makeStorage(new Uint32Array(queryCount*4),4,'uvec4');
    const queryFn=wgslFn(`fn runQueries(id:vec3u, count:u32, queries:ptr<storage,array<vec4f>,read>, output:ptr<storage,array<vec4f>,read_write>, faces:ptr<storage,array<vec4u>,read_write>) -> void {
      if(id.x>=count){return;} var result:PointQueryResult;
      bvh_ClosestPointToPoint(queries[id.x].xyz,&result);
      output[id.x]=vec4f(result.closestPoint,result.distanceSq);faces[id.x]=result.faceIndices;
    }`,[bvh.fns.closestPointToPoint]);
    const queryKernel=queryFn({id:globalId,count:queryCount,queries:query.s,output:output.s,faces:faceOutput.s}).computeKernel([64,1,1]);
    function queryNow(){renderer.compute(queryKernel,[Math.ceil(queryCount/64),1,1]);}
    renderer.compute(deform);queryNow();await device.queue.onSubmittedWorkDone();
    const native=k=>renderer.backend.get(attr[k]).buffer;
    refitter=new GPUBVHRefitter({device,nodeBuffer:native('nodes'),indexBuffer:native('index'),vertexBuffer:native('attributes'),nodeArray,indexArray,vertexCount,transformArray:attr.transforms.array});
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(43,1100/600,.1,50);camera.position.set(5.3,4.3,6.8);camera.lookAt(0,0,0);
    mesh.material=new THREE.MeshNormalNodeMaterial({side:THREE.DoubleSide,flatShading:true});mesh.material.positionNode=storage(attr.attributes,'vec4',vertexCount).toReadOnly().element(vertexIndex).xyz;mesh.frustumCulled=false;scene.add(mesh);
    const grid=new THREE.GridHelper(8,24,0x354966,0x17263c);grid.position.y=-1.5;scene.add(grid);
    const qGeometry=new THREE.SphereGeometry(.025,5,4),qMaterial=new THREE.MeshBasicNodeMaterial({color:0x69f2cd});
    const points=new THREE.InstancedMesh(qGeometry,qMaterial,queryCount),matrix=new THREE.Matrix4();
    for(let i=0;i<queryCount;i++){matrix.makeTranslation(...queries.slice(i*4,i*4+3));points.setMatrixAt(i,matrix);}scene.add(points);
    const lineGeometry=new THREE.BufferGeometry();lineGeometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(queryCount*6),3));
    const lineMaterial=new THREE.LineBasicNodeMaterial({color:0x69e6ed,transparent:true,opacity:.35});
    const queryId=vertexIndex.div(2).toUint(),resultRead=storage(output.a,'vec4',queryCount).toReadOnly();
    lineMaterial.positionNode=select(vertexIndex.mod(2).equal(0),query.s.element(queryId).xyz,resultRead.element(queryId).xyz);
    const lines=new THREE.LineSegments(lineGeometry,lineMaterial);lines.frustumCulled=false;scene.add(lines);
    const hitMaterial=new THREE.MeshBasicNodeMaterial({color:0xffca79});hitMaterial.positionNode=positionLocal.add(resultRead.element(instanceIndex).xyz);
    const hits=new THREE.InstancedMesh(qGeometry,hitMaterial,queryCount);matrix.identity();for(let i=0;i<queryCount;i++)hits.setMatrixAt(i,matrix);hits.frustumCulled=false;scene.add(hits);
    let intervalKernel=null,intervalStats=null;
    const timeLo=uniform(0),timeHi=uniform(0);
    async function enableIntervals(){
      if(intervalKernel)return intervalStats;
      const {compileInterval}=await import('./tsl-intervals.js');
      const rest=new Float32Array(nodeArray.length),floats=new Float32Array(nodeArray.buffer);
      for(let n=0;n<nodeArray.length/8;n++){rest.set(floats.slice(n*8,n*8+3),n*8);rest.set(floats.slice(n*8+3,n*8+6),n*8+4);}
      if(rest.some(x=>!Number.isFinite(x)||Math.abs(x)>32))throw Error('Rest boxes exceed compiler input domain');
      const restBoxes=makeStorage(rest,4,'vec4',true),bounds=storage(attr.nodes,'float',nodeArray.length);
      intervalKernel=Fn(()=>{
        const lo=restBoxes.s.element(instanceIndex.mul(2)).xyz,hi=restBoxes.s.element(instanceIndex.mul(2).add(1)).xyz;
        const range=compileInterval(graph,new Map([[p,{lo,hi}],[t,{lo:timeLo,hi:timeHi}]]));intervalStats=range.stats;
        for(let j=0;j<3;j++){bounds.element(instanceIndex.mul(8).add(j)).assign(range.lo[['x','y','z'][j]]);bounds.element(instanceIndex.mul(8).add(j+3)).assign(range.hi[['x','y','z'][j]]);}
      })().compute(nodeArray.length/8);
      // Compilation is included in initialization, never in steady timings.
      renderer.compute(intervalKernel);await device.queue.onSubmittedWorkDone();check();return intervalStats;
    }
    function check(){if(errors.length)throw Error(errors.join('\n'));}
    function update(time,method='refit',end=time){
      if(!Number.isFinite(time)||!Number.isFinite(end)||Math.abs(time)>6||Math.abs(end)>6||end<time)throw Error('Time outside bounded domain');
      t.value=time;renderer.compute(deform);
      if(method==='refit'){const encoder=device.createCommandEncoder();refitter.encode(encoder);device.queue.submit([encoder.finish()]);}
      else if(method==='interval'){if(!intervalKernel)throw Error('Intervals not initialized');timeLo.value=time;timeHi.value=end;renderer.compute(intervalKernel);}
      else if(method!=='stale'&&method!=='deform-only')throw Error('Unknown update method');
    }
    const read=async(a,Type)=>new Type(await renderer.getArrayBufferAsync(a));
    const proof=[];
    async function validateCurrent(time,method,end=time){
      queryNow();await device.queue.onSubmittedWorkDone();check();
      const [v,n,out,faces]=await Promise.all([read(attr.attributes,Float32Array),read(attr.nodes,Uint32Array),read(output.a,Float32Array),read(faceOutput.a,Uint32Array)]);
      const oracleCount=Math.min(queryCount,64),oracle=bruteForceQueries(v,indexArray,queries.slice(0,oracleCount*4));let maxDistanceSqError=0,maxPointError=0,wrong=0,wrongTrianglePoint=0;
      for(let i=0;i<oracleCount;i++){
        const error=Math.abs(out[i*4+3]-oracle.distanceSq[i]);maxDistanceSqError=Math.max(maxDistanceSqError,error);
        const pointError=Math.hypot(...[0,1,2].map(j=>out[i*4+j]-oracle.closestPoints[i*4+j]));maxPointError=Math.max(maxPointError,pointError);
        if(!Number.isFinite(error)||error>2e-4*(1+oracle.distanceSq[i])||faces[i*4+3]>=indexArray.length/3)wrong++;
        const tri=faces[i*4+3],ids=Array.from(indexArray.slice(tri*3,tri*3+3));
        if(ids.length!==3||ids.some((id,j)=>id!==faces[i*4+j])){wrongTrianglePoint++;continue;}
        const winner=closestPointTriangle(Array.from(queries.slice(i*4,i*4+3)),...ids.map(id=>Array.from(v.slice(id*4,id*4+3))));
        if([0,1,2].some(j=>!Number.isFinite(out[i*4+j])||Math.abs(winner.closestPoint[j]-out[i*4+j])>2e-4)||Math.abs(winner.distanceSq-out[i*4+3])>2e-4*(1+winner.distanceSq))wrongTrianglePoint++;
      }
      const bounds=validateBounds(n,v,indexArray,{tolerance:0,requireChildContainment:method==='refit'}),metadata=validateMetadataUnchanged(nodeArray,n);
      proof.push({time,end,method,vertices:Array.from(v),nodes:Array.from(n),queryResults:Array.from(out.slice(0,oracleCount*4)),faces:Array.from(faces.slice(0,oracleCount*4))});
      return {time,end,method,oracleCount,ok:!wrong&&!wrongTrianglePoint&&bounds.ok&&metadata.ok,wrong,wrongTrianglePoint,maxDistanceSqError,maxPointError,bounds,metadata};
    }
    async function validate(time,method='refit',end=time){update(time,method,end);return validateCurrent(time,method,end);}
    async function benchmark(method,iterations=20){
      if(iterations<1||iterations>40)throw Error('Iteration cap');
      for(let i=0;i<3;i++){update(.4,method);queryNow();}await device.queue.onSubmittedWorkDone();
      const samples=[];
      for(let i=0;i<iterations;i++){const start=performance.now();update(.4+i*.017,method);queryNow();await device.queue.onSubmittedWorkDone();samples.push(performance.now()-start);}
      check();const sorted=samples.slice().sort((a,b)=>a-b);return {method,samples,medianMs:sorted[Math.floor(sorted.length/2)],metric:'serialized deformation + bound update + upstream queries + queue completion wall time; no render or readback'};
    }
    async function sweptCheck(start,end){
      update(start,'interval',end);await device.queue.onSubmittedWorkDone();const samples=[];
      for(let i=0;i<=8;i++){const time=start+(end-start)*i/8;update(time,'deform-only');samples.push(await validateCurrent(time,'swept',end));}
      return {start,end,samples,ok:samples.every(x=>x.ok),note:'Finite temporal samples test the implementation, not a formal proof or continuous collision narrow phase.'};
    }
    async function negativeControls(){
      device.queue.writeBuffer(native('nodes'),0,nodeArray);const stale=await validate(.7,'stale');
      // Union of exact endpoint boxes can miss an interior sinusoidal maximum.
      update(0,'refit');const left=await read(attr.nodes,Uint32Array);update(3.2,'refit');const right=await read(attr.nodes,Uint32Array);
      const merged=left.slice(),mf=new Float32Array(merged.buffer),lf=new Float32Array(left.buffer),rf=new Float32Array(right.buffer);
      for(let n=0;n<nodeArray.length/8;n++)for(let j=0;j<3;j++){mf[n*8+j]=Math.min(lf[n*8+j],rf[n*8+j]);mf[n*8+3+j]=Math.max(lf[n*8+3+j],rf[n*8+3+j]);}
      device.queue.writeBuffer(native('nodes'),0,merged);update(1.6,'deform-only');const endpoints=await validateCurrent(1.6,'endpoint-union',3.2);
      return {stale,endpoints,expectedFailure:true,detected:!stale.ok&&!endpoints.ok};
    }
    async function batchedBenchmark(){
      const {runPairedBatches}=await import('./timing.js');
      const result=await runPairedBatches({device,prepare(method,start,end){if(method==='swept')update(start,'interval',end);},step(method,time){update(time,method==='swept'?'deform-only':method);queryNow();},steps:16,rounds:6});check();return result;
    }
    async function validateDerivatives(){
      const {differentiate}=await import('./tsl-derivatives.js'),result=differentiate(graph,t,{constants:[p]});
      const velocity=makeStorage(new Float32Array(vertexCount*4),4,'vec4');
      const kernel=Fn(()=>{velocity.s.element(instanceIndex).assign(vec4(result.derivative,0));})().compute(vertexCount);
      const checks=[];
      for(const time of [0,.7,2.4]){
        t.value=time;renderer.compute(kernel);const actual=await read(velocity.a,Float32Array),epsilon=.001;
        update(time-epsilon,'deform-only');const left=await read(attr.attributes,Float32Array);
        update(time+epsilon,'deform-only');const right=await read(attr.attributes,Float32Array);
        let maxAbsoluteError=0,nonfinite=0;
        for(let i=0;i<actual.length;i++)if(i%4!==3){const error=Math.abs(actual[i]-(right[i]-left[i])/(2*epsilon));if(!Number.isFinite(error))nonfinite++;else maxAbsoluteError=Math.max(maxAbsoluteError,error);}
        checks.push({time,epsilon,maxAbsoluteError,nonfinite,ok:!nonfinite&&maxAbsoluteError<.003});
      }
      check();return {stats:result.stats,checks,ok:checks.every(x=>x.ok),note:'Generated analytic GPU derivatives compared with central finite differences of actual GPU deformation. Absolute tolerance .003; this is a numerical check, not a formal proof.'};
    }
    function exportProof(){return {family,segments,oracleCount:Math.min(queryCount,64),queries:Array.from(queries.slice(0,Math.min(queryCount,64)*4)),indices:Array.from(indexArray),originalNodes:Array.from(nodeArray),baseVertices:Array.from(baseVertices),samples:proof,shaderSources};}
    function render(time,method='refit'){update(time,method);queryNow();renderer.render(scene,camera);check();}
    function dispose(){if(disposed)return;disposed=true;refitter?.dispose();bvh?.dispose();for(const a of owned)a.dispose();geometry.dispose();mesh.material.dispose();qGeometry.dispose();qMaterial.dispose();lineGeometry.dispose();lineMaterial.dispose();hitMaterial.dispose();grid.geometry.dispose();grid.material.dispose();renderer.dispose();device.destroy();}
    return {validate,benchmark,batchedBenchmark,negativeControls,validateDerivatives,exportProof,enableIntervals,sweptCheck,render,dispose,errors,info:{family,segments,vertexCount,triangleCount:indexArray.length/3,nodeCount:nodeArray.length/8,queryCount,refitDispatches:refitter.dispatchCount,timestampQuery:device.features.has('timestamp-query'),adapter:adapter.info&&{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description}}};
  }catch(error){disposed=true;refitter?.dispose();bvh?.dispose();for(const a of owned)a.dispose();renderer?.dispose();device.destroy();throw error;}
}
