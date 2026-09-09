import * as THREE from 'three/webgpu';
import {storage, wgslFn, globalId} from 'three/tsl';
import {closestPointTriangle} from './reference.mjs';

const PERMUTATIONS = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]];
const SEED = 0x6d2b79f5;
const f32 = point => point.map(Math.fround);
const distanceSq = (a,b) => (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2;

function fixtures() {
  const result=[], corners=[[0,0,0],[2,0,0],[-1,1,0]];
  const probes=[
    ['known-obtuse-edge-failure',[1,-2,0]],
    ['face-interior',[1/3,1/3,0]],
    ['above-face',[1/3,1/3,1]],
    ['below-face',[1/3,1/3,-1]],
    ['edge-ab',[1,0,0]], ['edge-bc',[.5,.5,0]], ['edge-ca',[-.5,.5,0]],
    ['vertex-a',[-1,-2,0]], ['vertex-b',[3,-1,0]], ['vertex-c',[-2,2,0]],
  ];
  for (const [name,p] of probes) for (const permutation of PERMUTATIONS) {
    result.push({id:`${name}/${permutation.join('')}`,group:name,permutation:[...permutation],
      p:f32(p),a:f32(corners[permutation[0]]),b:f32(corners[permutation[1]]),c:f32(corners[permutation[2]])});
  }
  for(const height of [.01,.001])for(const permutation of PERMUTATIONS){
    const triangle=[[0,0,0],[2,0,0],[-1,height,0]],p=[1,-2*height,0];
    result.push({id:`thin-obtuse-${height}/${permutation.join('')}`,group:'thin-obtuse',permutation:[...permutation],p:f32(p),a:f32(triangle[permutation[0]]),b:f32(triangle[permutation[1]]),c:f32(triangle[permutation[2]])});
  }
  let seed=SEED, accepted=0;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  const point=scale=>f32([0,1,2].map(()=>scale*(random()-.5)));
  for (let attempt=0; accepted<128 && attempt<4096; attempt++) {
    const a=point(4),b=point(4),c=point(4),p=point(6);
    const ab=b.map((x,i)=>x-a[i]),ac=c.map((x,i)=>x-a[i]);
    const n=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];
    const longestSq=Math.max(distanceSq(a,b),distanceSq(a,c),distanceSq(b,c));
    // Exclude tiny/near-collinear geometry; this probe is not a degeneracy claim.
    if (longestSq<.04 || distanceSq(n,[0,0,0])<1e-3*longestSq**2) continue;
    result.push({id:`random/${accepted}`,group:'random',attempt,p,a,b,c}); accepted++;
  }
  if (accepted!==128 || result.length>512) throw Error('Bounded fixture generation failed.');
  return result;
}

/**
 * One isolated compute dispatch calling the CURRENT upstream TSL/WGSL triangle
 * helper, without any BVH, refitting, deformation, simulation or render pass.
 * An import-map/server overlay can replace the upstream helper at the same URL;
 * the probe and fixtures remain identical. No run happens when importing this
 * module. The external runner owns the wall-time watchdog and source hashes.
 */
export async function runTriangleProbe() {
  const cases=fixtures();
  const tolerances={distanceSqAbsolute:2e-5,distanceSqRelative:2e-5,pointAbsolute:2e-4,
    barycentricSumAbsolute:2e-5,barycentricLowerBound:-2e-5,cpuPointAbsolute:1e-10,cpuDistanceSqAbsolute:1e-10};
  const errors=[];
  const report={kind:'isolated-upstream-triangle-probe-v2',status:'not-run',caseCount:cases.length,
    randomSeed:SEED,randomCount:128,analyticCaseCount:72,permutations:PERMUTATIONS.map(p=>[...p]),
    tolerances,errors,cpuOracleMismatchCount:0,mismatchCount:null,knownFailureMismatchCount:null,
    cases:[],rawGpuOutput:null,adapter:null,
    scope:'Single-dispatch nondegenerate-triangle correctness probe; no BVH, deformation, collision dynamics, timing claim or general proof.'};
  const triangle=new THREE.Triangle(),cpuPoint=new THREE.Vector3(),queryPoint=new THREE.Vector3();
  const inputs=new Float32Array(cases.length*16);
  for (const [i,fixture] of cases.entries()) {
    const expected=closestPointTriangle(fixture.p,fixture.a,fixture.b,fixture.c);
    triangle.set(new THREE.Vector3(...fixture.a),new THREE.Vector3(...fixture.b),new THREE.Vector3(...fixture.c));
    triangle.closestPointToPoint(queryPoint.set(...fixture.p),cpuPoint);
    const threePoint=cpuPoint.toArray(),threeDistanceSq=distanceSq(fixture.p,threePoint);
    const cpuAgreement=Math.sqrt(distanceSq(expected.closestPoint,threePoint))<=tolerances.cpuPointAbsolute
      && Math.abs(expected.distanceSq-threeDistanceSq)<=tolerances.cpuDistanceSqAbsolute;
    if (!cpuAgreement) report.cpuOracleMismatchCount++;
    if (fixture.group==='known-obtuse-edge-failure' &&
      (distanceSq(expected.closestPoint,[1,0,0])>1e-24 || Math.abs(expected.distanceSq-4)>1e-12)) {
      throw Error('Independent oracle failed the analytic known-answer fixture.');
    }
    report.cases.push({...fixture,expected,threeCpu:{closestPoint:threePoint,distanceSq:threeDistanceSq},cpuAgreement,gpu:null,mismatch:null});
    [fixture.p,fixture.a,fixture.b,fixture.c].forEach((point,k)=>inputs.set([...point,0],i*16+k*4));
  }
  if (report.cpuOracleMismatchCount) { report.status='cpu-oracle-disagreement'; return report; }

  let device=null,renderer=null,disposed=false,scopeOpen=false;
  const attributes=[];
  try {
    // The barrel currently star-exports a deprecated helper with the same name.
    // Resolve its package URL, then select the exact current helper used by the
    // upstream point traversal, rather than importing an ambiguous barrel name.
    const helperUrl=new URL('./tsl/fns.js',import.meta.resolve('three-mesh-bvh/webgpu')).href;
    const {closestPointToTriangle}=await import(helperUrl);
    report.helperUrl=helperUrl;
    if (!globalThis.navigator?.gpu) throw Error('WebGPU unavailable.');
    const adapter=await navigator.gpu.requestAdapter();
    if (!adapter) throw Error('No WebGPU adapter.');
    report.adapter=adapter.info?{vendor:adapter.info.vendor,architecture:adapter.info.architecture,description:adapter.info.description}:null;
    device=await adapter.requestDevice();
    device.addEventListener('uncapturederror',event=>errors.push(event.error.message));
    device.lost.then(info=>{if(!disposed)errors.push(`Unexpected device loss: ${info.message}`);});
    renderer=new THREE.WebGPURenderer({device,antialias:false});
    await renderer.init();
    const inputAttribute=new THREE.StorageBufferAttribute(inputs,4);
    attributes.push(inputAttribute);
    const outputAttribute=new THREE.StorageBufferAttribute(new Float32Array(cases.length*8),4);
    attributes.push(outputAttribute);
    const inputStorage=storage(inputAttribute,'vec4',cases.length*4).toReadOnly();
    const outputStorage=storage(outputAttribute,'vec4',cases.length*2);
    const probe=wgslFn(`fn triangleProbe(id:vec3u, count:u32,
      inputs:ptr<storage,array<vec4f>,read>, outputs:ptr<storage,array<vec4f>,read_write>) -> void {
      if(id.x>=count){return;}
      let offset=4u*id.x;
      let p=inputs[offset].xyz; let a=inputs[offset+1u].xyz;
      let b=inputs[offset+2u].xyz; let c=inputs[offset+3u].xyz;
      let bary=closestPointToTriangle(p,a,b,c);
      let closest=a*bary.x+b*bary.y+c*bary.z;
      let delta=p-closest;
      outputs[2u*id.x]=vec4f(bary,0.0);
      outputs[2u*id.x+1u]=vec4f(closest,dot(delta,delta));
    }`,[closestPointToTriangle]);
    const kernel=probe({id:globalId,count:cases.length,inputs:inputStorage,outputs:outputStorage}).computeKernel([64,1,1]);
    device.pushErrorScope('validation'); scopeOpen=true;
    renderer.compute(kernel,[Math.ceil(cases.length/64),1,1]);
    await device.queue.onSubmittedWorkDone();
    const validationError=await device.popErrorScope(); scopeOpen=false;
    if(validationError)errors.push(validationError.message);
    if(errors.length)throw Error('WebGPU validation/device error; see errors.');
    const raw=new Float32Array(await renderer.getArrayBufferAsync(outputAttribute));
    report.rawGpuOutput=Array.from(raw);
    report.rawOutputLayout='8 f32 per case: bary.x,y,z,0, closest.x,y,z,distanceSq';
    report.mismatchCount=0; report.knownFailureMismatchCount=0;
    let maxPointError=0,maxDistanceSqError=0;
    for(const [i,record] of report.cases.entries()) {
      const barycoord=Array.from(raw.subarray(i*8,i*8+3));
      const closestPoint=Array.from(raw.subarray(i*8+4,i*8+7));
      const gpuDistanceSq=raw[i*8+7];
      const pointError=Math.sqrt(distanceSq(closestPoint,record.expected.closestPoint));
      const distanceSqError=Math.abs(gpuDistanceSq-record.expected.distanceSq);
      const finite=[...barycoord,...closestPoint,gpuDistanceSq].every(Number.isFinite);
      const barycentricValid=Math.abs(barycoord.reduce((x,y)=>x+y,0)-1)<=tolerances.barycentricSumAbsolute
        && barycoord.every(value=>value>=tolerances.barycentricLowerBound);
      record.gpu={barycoord,closestPoint,distanceSq:gpuDistanceSq};
      record.errors={pointError,distanceSqError,finite,barycentricValid};
      record.mismatch=!finite || !barycentricValid || pointError>tolerances.pointAbsolute
        || distanceSqError>tolerances.distanceSqAbsolute+tolerances.distanceSqRelative*record.expected.distanceSq;
      if(record.mismatch){report.mismatchCount++;if(record.group==='known-obtuse-edge-failure')report.knownFailureMismatchCount++;}
      if(Number.isFinite(pointError))maxPointError=Math.max(maxPointError,pointError);
      if(Number.isFinite(distanceSqError))maxDistanceSqError=Math.max(maxDistanceSqError,distanceSqError);
    }
    report.maxPointError=maxPointError; report.maxDistanceSqError=maxDistanceSqError;
    report.status=report.mismatchCount?'mismatch':'passed';
  } catch(error) {
    errors.push(error?.stack??String(error)); report.status='error';
  } finally {
    if(scopeOpen && device){try{const error=await device.popErrorScope();if(error)errors.push(error.message);}catch(error){errors.push(String(error));}}
    disposed=true;
    for(const attribute of attributes){try{attribute.dispose();}catch(error){errors.push(String(error));}}
    try{renderer?.dispose();}catch(error){errors.push(String(error));}
    try{device?.destroy();}catch(error){errors.push(String(error));}
    if(errors.length && report.status==='passed')report.status='error';
  }
  return report;
}
