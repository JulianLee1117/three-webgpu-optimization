// CPU-only, dependency-free audit of immutable retained GPU casting evidence.
// This checks source provenance and recorded numerical results, not new GPU work.
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

export const sha256=bytes=>createHash('sha256').update(bytes).digest('hex');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const GPU_SHA='4a36b8c64374c0d390df5e1b0ffaefa624be49527bd1072c6e4e444087c433bf';
const CPU_SHA='b2652a80f8b7b531696f4eaf9847f031a359a6dc2974978f859b63333e73147e';
const LAYOUT={strideFloats:36,byteStride:144,position:0,mode:3,velocity:4,mass:7,C:8,F:20,volume:32,id:34,matrixLayout:'column-major padded vec4 columns'};
const IDS=['casting','cast-solid-release','cast-liquid-release'];
const lengths={ids:1,mode:1,mass:1,volume:1,x:3,v:3,C:9,F:9};
const close=(a,b,tolerance,label)=>assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tolerance,`${label}: ${a} vs ${b}`);
const determinant=(m,k)=>m[k]*m[k+4]*m[k+8]+m[k+1]*m[k+5]*m[k+6]+m[k+2]*m[k+3]*m[k+7]-m[k+2]*m[k+4]*m[k+6]-m[k+1]*m[k+3]*m[k+8]-m[k]*m[k+5]*m[k+7];

export function assertNoHostPaths(value,location='root'){
  if(typeof value==='string')assert.ok(!/(?:\b[A-Za-z]:[\\/]|file:\/\/\/|\/Users\/|\/home\/)/.test(value),`Host absolute path at ${location}`);
  else if(value&&typeof value==='object')for(const[key,child]of Object.entries(value))assertNoHostPaths(child,location+'.'+key);
}

export function extractKernels(source){
  // Read the five literal WGSL strings without evaluating archived JavaScript.
  // ECMAScript template literals normalize CRLF to LF, reproduced explicitly here.
  const text=source.replaceAll('\r\n','\n'),common=text.match(/const common = \/\* wgsl \*\/`([\s\S]*?)`;/)?.[1];
  assert.ok(common,'Missing literal WGSL common block');const kernels={};
  for(const match of text.matchAll(/\b(clear|mass|stress|grid|gather):\s*\/\* wgsl \*\/`([\s\S]*?)`/g)){
    const full=common+match[2];assert.ok(!full.includes('${')&&!full.includes('\\'),'Unsupported template interpolation/escape');
    assert.ok(!kernels[match[1]],'Duplicate kernel');kernels[match[1]]=full;
  }
  assert.deepEqual(Object.keys(kernels),['clear','mass','stress','grid','gather']);return kernels;
}

function sampleFixture(protocol){
  const initial={ids:[],mode:[],mass:[],volume:[],x:[],v:[],C:[],F:[],time:0,steps:0,materialEdits:[]};
  for(const spec of [protocol.body,protocol.load])for(let z=spec.min[2]+spec.spacing/2;z<spec.max[2]-1e-10;z+=spec.spacing)
    for(let y=spec.min[1]+spec.spacing/2;y<spec.max[1]-1e-10;y+=spec.spacing)
      for(let x=spec.min[0]+spec.spacing/2;x<spec.max[0]-1e-10;x+=spec.spacing){
        initial.ids.push(initial.ids.length);initial.mode.push(spec.mode);initial.mass.push(spec.mass);initial.volume.push(spec.volume);
        initial.x.push(x,y,z);initial.v.push(0,0,0);initial.C.push(0,0,0,0,0,0,0,0,0);initial.F.push(1,0,0,0,1,0,0,0,1);
      }
  return initial;
}

function measureIndependent(state,fixture){
  const g=fixture.protocol.gate,bins=g.centralBins;
  const supports=fixture.releasedConfig.anchors.map(()=>0),centralBins=bins.xEdges.slice(1).map(()=>0),center=[0,0,0];
  let below=0,central=0,totalMass=0,loadMass=0,loadY=0,minJ=Infinity;
  for(let i=0;i<state.ids.length;i++){
    const x=state.x[3*i],y=state.x[3*i+1],z=state.x[3*i+2],m=state.mass[i];
    totalMass+=m;center[0]+=m*x;center[1]+=m*y;center[2]+=m*z;minJ=Math.min(minJ,determinant(state.F,9*i));
    if(i>=fixture.bodyCount){loadMass+=m;loadY+=m*y;continue;}
    if(y<g.belowGrilleY)below++;
    if(x>=g.centralRegion.min[0]&&x<=g.centralRegion.max[0]&&y>=g.centralRegion.min[1]&&y<=g.centralRegion.max[1]&&z>=g.centralRegion.min[2]&&z<=g.centralRegion.max[2])central++;
    fixture.releasedConfig.anchors.forEach((box,j)=>{if(x>=box.min[0]&&x<=box.max[0]&&y>=box.min[1]&&y<=box.max[1]&&z>=box.min[2]&&z<=box.max[2])supports[j]++;});
    if(y>=bins.minY&&y<=bins.maxY&&z>=bins.minZ&&z<=bins.maxZ)for(let j=0;j<centralBins.length;j++)if(x>=bins.xEdges[j]&&x<bins.xEdges[j+1])centralBins[j]++;
  }
  return {bodyFractionBelowGrille:below/fixture.bodyCount,centralBodyParticles:central,supportCounts:supports,centralBinCounts:centralBins,loadCenterY:loadY/loadMass,totalMass,center:center.map(v=>v/totalMass),minimumDeterminant:minJ};
}

function checkMeasurement(actual,recorded){
  assert.deepEqual(actual.supportCounts,recorded.supportCounts);assert.deepEqual(actual.centralBinCounts,recorded.centralBinCounts);
  for(const key of ['bodyFractionBelowGrille','centralBodyParticles','loadCenterY','totalMass'])close(actual[key],recorded[key],1e-12,key);
  actual.center.forEach((v,i)=>close(v,recorded.center[i],1e-12,'center'));
  close(actual.minimumDeterminant,recorded.minimumDeterminant,1e-12,'determinant');
}

export function auditReport(report,manifest,cpuReference){
  assert.equal(report.kind,'matter-forge-casting-gpu-report-v2');assert.equal(report.status,'passed');assert.equal(report.watchdogExpired,false);
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.requestFailures,[]);assertNoHostPaths(report);
  assert.deepEqual(report.sourceHashes,manifest.sourceHashes);
  for(const[name,digest]of Object.entries(report.sourceHashes))assert.equal(sha256(report.sourceSnapshots[name]),digest,`Captured source ${name}`);
  const expectedServed=Object.keys(report.sourceHashes).filter(p=>p.startsWith('experiments/')).map(p=>'/'+p);
  assert.deepEqual(Object.keys(report.servedHashes).sort(),expectedServed.sort());
  for(const[url,digest]of Object.entries(report.servedHashes))assert.equal(digest,report.sourceHashes[url.slice(1)]);
  const kernelText=extractKernels(report.sourceSnapshots['experiments/matter-forge/gpu.mjs']);
  assert.deepEqual(Object.fromEntries(Object.entries(kernelText).map(([k,v])=>[k,sha256(v)])),manifest.kernelSHA256);
  assert.deepEqual(manifest.layout,LAYOUT);
  assert.ok(kernelText.clear.includes('struct Particle { x: vec4f, v: vec4f, C: mat3x3f, F: mat3x3f, aux: vec4f }'));
  assert.ok(kernelText.gather.includes('p.F=(identity()+u.dt*C)*p.F'));
  assert.ok(kernelText.grid.includes('if(atomicLoad(&grid[k].solid)>0)'));
  assert.equal(cpuReference.kind,'matter-forge-casting-cpu-reference-v1');assert.equal(cpuReference.sourceReport.sha256,CPU_SHA);assert.equal(cpuReference.status,'passed');
  assert.equal(report.cpuReference.sha256,CPU_SHA);assert.deepEqual(report.cpuReference.lanes,cpuReference.lanes);
  for(const[name,digest]of Object.entries(cpuReference.sourceHashes))assert.equal(report.sourceHashes[name],digest);
  const p=report.probe,f=p.fixture;assert.equal(p.status,'passed');assert.deepEqual(p.errors,[]);
  assert.deepEqual(f.protocol,cpuReference.protocol);assert.deepEqual(f.castingConfig,cpuReference.castingConfig);assert.deepEqual(f.releasedConfig,cpuReference.releasedConfig);
  assert.equal(f.protocol.version,2);assert.equal(f.bodyCount,1152);assert.equal(f.loadCount,64);assert.equal(f.protocol.N,24);assert.equal(f.protocol.dt,1/240);
  assert.equal(f.protocol.castingSteps,960);assert.equal(f.protocol.releasedSteps,960);assert.equal(f.protocol.totalExecutedSteps,2880);
  assert.equal(f.releasedConfig.colliders.length,2);assert.deepEqual(f.releasedConfig.colliders.map(b=>b.name),['left-bank','right-bank']);
  assert.deepEqual(f.state,sampleFixture(f.protocol),'Independently sampled initial lattice differs');
  assert.deepEqual(p.stages.map(s=>s.id),IDS);assert.deepEqual(p.comparisonTolerance,{loadCenterYAbsolute:.1,totalCenterAbsolute:.1,bodyFractionBelowGrilleAbsolute:.1});
  assert.deepEqual(p.lifecycle,{devices:1,simulatorsCreated:3,simulatorsDisposed:3,activeSimulators:0,maximumActiveSimulators:1,completedSteps:2880,advanceCalls:12,deviceDestroyed:true,deviceLossReason:'destroyed',pass:true});
  let finiteScalars=0,rawParticleRecords=0;const summaries=[];
  function stateValid(state,expectedMode){
    for(const[key,width]of Object.entries(lengths)){
      assert.ok(Array.isArray(state[key]));assert.equal(state[key].length,1216*width,key);
      for(const value of state[key]){assert.ok(Number.isFinite(value),`Nonfinite ${key}`);finiteScalars++;}
    }
    for(const key of ['ids','mass','volume'])assert.deepEqual(state[key],f.state[key],`Conserved ${key}`);
    assert.deepEqual(state.mode,expectedMode);
    const m=measureIndependent(state,f);assert.ok(m.minimumDeterminant>.05&&m.minimumDeterminant<20,'Unsafe F determinant');return m;
  }
  function rawValid(raw,state,expectedMode){
    assert.equal(raw.strideFloats,36);assert.equal(raw.byteLength,1216*144);assert.equal(raw.float32.length,1216*36);
    for(const value of raw.float32)assert.ok(Number.isFinite(value)&&value===Math.fround(value),'Raw value not finite Float32');
    for(let i=0;i<1216;i++){
      const k=i*36,v=raw.float32;assert.equal(v[k+34],f.state.ids[i]);assert.equal(v[k+3],expectedMode[i]);assert.equal(v[k+7],f.state.mass[i]);assert.equal(v[k+32],f.state.volume[i]);
      for(let a=0;a<3;a++){assert.equal(v[k+a],state.x[i*3+a]);assert.equal(v[k+4+a],state.v[i*3+a]);}
      for(let a=0;a<9;a++){const gpuOffset=4*(a%3)+Math.floor(a/3);assert.equal(v[k+8+gpuOffset],state.C[i*9+a]);assert.equal(v[k+20+gpuOffset],state.F[i*9+a]);}
      rawParticleRecords++;
    }
    assert.equal(raw.metadataExact,true);assert.equal(raw.snapshotExact,true);
  }
  for(let i=0;i<p.stages.length;i++){
    const stage=p.stages[i];assert.equal(stage.status,'completed');assert.equal(stage.completedSteps,960);
    assert.deepEqual(stage.config,i===0?f.castingConfig:f.releasedConfig);const mode=i===1?Array(1216).fill(1):f.state.mode;
    stateValid(stage.initial,mode);stateValid(stage.uploaded,mode);
    for(const key of ['x','v','C','F','mass','volume'])assert.deepEqual(stage.uploaded[key],stage.initial[key].map(Math.fround),`Actual upload ${key}`);
    assert.equal(stage.uploaded.steps,stage.initial.steps);close(stage.uploaded.time,stage.initial.time,1e-12,'Upload time');rawValid(stage.rawUpload,stage.uploaded,mode);
    assert.deepEqual(stage.frames.map(v=>v.stageSteps),[240,480,720,960]);
    for(const frame of stage.frames){
      assert.equal(frame.state.steps,stage.initial.steps+frame.stageSteps);close(frame.state.time,stage.initial.time+frame.stageSteps/240,1e-12,'Frame time');
      checkMeasurement(stateValid(frame.state,mode),frame.measurement);
    }
    assert.deepEqual(stage.final,stage.frames[3].state);const m=stateValid(stage.final,mode);checkMeasurement(m,stage.measurement);rawValid(stage.rawFinal,stage.final,mode);summaries.push({id:stage.id,...m});
  }
  assert.deepEqual(p.stages[0].initial,f.state);const cast=p.stages[0].final;
  for(const stage of p.stages.slice(1))for(const key of ['ids','x','v','C','mass','volume','steps','time'])assert.deepEqual(stage.initial[key],cast[key],'Branches do not share physical state');
  assert.deepEqual(p.stages[2].initial,cast,'Liquid control modified');
  for(let i=0;i<1216;i++)assert.deepEqual(p.stages[1].initial.F.slice(i*9,i*9+9),i<1152?[1,0,0,0,1,0,0,0,1]:cast.F.slice(i*9,i*9+9),'Unexpected material-reference edit');
  assert.deepEqual(p.edit.selection,Array.from({length:1152},(_,i)=>i));assert.ok(Object.values(p.edit.preserved).every(Boolean));assert.equal(p.edit.newReferenceIsIdentity,true);assert.equal(p.edit.commonPhysicalState,true);assert.equal(p.edit.loadUnchanged,true);
  const [casting,solid,liquid]=summaries,g=f.protocol.gate;
  const computedGates={allStagesComplete:true,materialPassedGrille:casting.bodyFractionBelowGrille>g.fractionBelowGrilleGreaterThan,
    centralCastingPresent:casting.centralBodyParticles>=g.minimumCentralBodyParticles,bothSupportsPopulated:casting.supportCounts.every(c=>c>=g.minimumParticlesInEachSupport),
    allCentralBinsPopulated:casting.centralBinCounts.every(c=>c>=g.centralBins.minimumParticlesPerBin),editPreservesState:true,exactRawMetadataAndUploads:true,
    finalDeterminantsSafe:summaries.every(m=>m.minimumDeterminant>.05),solidSupportsLoad:solid.loadCenterY>=g.solidFinalLoadYAtLeast,
    solidBeatsLiquidControl:solid.loadCenterY-liquid.loadCenterY>=g.solidAboveLiquidByAtLeast,massRetained:summaries.every(s=>s.totalMass===152)};
  const cpuErrors=summaries.map((s,i)=>{
    const ref=cpuReference.lanes[i].measurement;assert.equal(cpuReference.lanes[i].id,s.id);
    const errors={loadCenterY:Math.abs(s.loadCenterY-ref.loadCenterY),totalCenter:Math.max(...s.center.map((v,j)=>Math.abs(v-ref.total.center[j]))),bodyFractionBelowGrille:Math.abs(s.bodyFractionBelowGrille-ref.bodyFractionBelowGrille)};
    for(const key of Object.keys(errors))close(errors[key],p.cpuComparison[i].errors[key],1e-12,'Recorded CPU difference');assert.deepEqual(p.cpuComparison[i].reference,ref);
    const pass=Object.values(errors).every(v=>v<=.1);assert.equal(p.cpuComparison[i].pass,pass);return {id:s.id,...errors,pass};
  });
  computedGates.coarseCPUReferenceAgreement=cpuErrors.every(e=>e.pass);assert.deepEqual(p.gates,computedGates);assert.ok(Object.values(computedGates).every(Boolean));
  return {status:'passed',finiteScalars,rawParticleRecords,sourceSnapshots:Object.keys(report.sourceHashes).length,executedSources:expectedServed.length,kernels:Object.keys(kernelText).length,steps:2880,particles:1216,summaries,cpuErrors,solidAboveLiquid:solid.loadCenterY-liquid.loadCenterY,scope:'Audit of retained data and captured runtime; no GPU rerun or validation of subsequent source edits.'};
}

export async function loadPublicEvidence(){
  const directory=path.join(root,'experiments/matter-forge/evidence'),manifestRaw=await readFile(path.join(directory,'manifest.json'));
  assert.equal(sha256(manifestRaw),(await readFile(path.join(directory,'manifest.sha256'),'utf8')).trim());const manifest=JSON.parse(manifestRaw);
  assert.equal(manifest.kind,'matter-forge-public-evidence-v1');assertNoHostPaths(manifest);assert.equal(manifest.gpuReport.rawSHA256,GPU_SHA);assert.deepEqual(manifest.gpuReport.redactions,[]);
  for(const record of [manifest.gpuReport,manifest.cpuReference])assert.equal(path.basename(record.file),record.file,'Evidence file must be a basename');
  const gzip=await readFile(path.join(directory,manifest.gpuReport.file));assert.equal(gzip.length,manifest.gpuReport.gzipBytes);assert.equal(sha256(gzip),manifest.gpuReport.gzipSHA256);
  const raw=gunzipSync(gzip,{maxOutputLength:20*1024*1024});assert.equal(raw.length,manifest.gpuReport.rawBytes);assert.equal(sha256(raw),GPU_SHA);
  const referenceRaw=await readFile(path.join(directory,manifest.cpuReference.file));assert.equal(referenceRaw.length,manifest.cpuReference.bytes);assert.equal(sha256(referenceRaw),manifest.cpuReference.sha256);
  const cpuReference=JSON.parse(referenceRaw);assertNoHostPaths(cpuReference);assert.equal(cpuReference.sourceReport.sha256,manifest.cpuReference.sourceReportSHA256);
  return {manifest,report:JSON.parse(raw),cpuReference};
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  assert.ok(process.argv.slice(2).every(a=>a==='--self-test'),'Only --self-test is supported; no GPU is used');
  const data=await loadPublicEvidence(),analysis=auditReport(data.report,data.manifest,data.cpuReference);
  if(process.argv.includes('--self-test')){
    const corruptions=[r=>{r.probe.stages[0].rawFinal.float32[34]=9999;},r=>{r.probe.stages[1].final.mass[0]*=2;},r=>{r.probe.stages[2].frames[0].state.x[0]+=.1;},r=>{r.probe.stages[1].config.colliders.push(structuredClone(r.probe.fixture.castingConfig.colliders.find(c=>c.name==='mold-floor')));}];
    for(const corrupt of corruptions){const clone=structuredClone(data.report);corrupt(clone);assert.throws(()=>auditReport(clone,data.manifest,data.cpuReference));}
    analysis.corruptionTests=corruptions.length;
  }
  console.log(JSON.stringify({reportSHA256:GPU_SHA,...analysis},null,2));
}
