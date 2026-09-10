import {castingFixtureV2} from './casting-fixture-v2.mjs';
import {makeState,transmute} from './cpu.mjs';
import {createSimulator} from './gpu.mjs';

const assert=(value,message)=>{if(!value)throw Error(message);};
const same=(a,b)=>a.length===b.length&&a.every((v,i)=>Object.is(v,b[i]));
// Independent determinant and state/metric readout; no CPU simulation is executed.
const det=(a,k)=>a[k]*(a[k+4]*a[k+8]-a[k+5]*a[k+7])+a[k+1]*(a[k+5]*a[k+6]-a[k+3]*a[k+8])+a[k+2]*(a[k+3]*a[k+7]-a[k+4]*a[k+6]);
export function measure(state,fixture){
  const {bodyCount,protocol,releasedConfig}=fixture,g=protocol.gate,bins=g.centralBins;
  let below=0,central=0,loadMass=0,loadY=0,totalMass=0,minJ=Infinity;
  const center=[0,0,0],supportCounts=releasedConfig.anchors.map(()=>0),centralBinCounts=bins.xEdges.slice(1).map(()=>0);
  for(const key of ['x','v','F','C','mass','volume'])assert(Array.from(state[key]).every(Number.isFinite),`Nonfinite ${key}`);
  for(let p=0;p<state.ids.length;p++){
    const xyz=Array.from(state.x.slice(p*3,p*3+3)),m=state.mass[p];
    minJ=Math.min(minJ,det(state.F,p*9));totalMass+=m;for(let a=0;a<3;a++)center[a]+=m*xyz[a];
    if(p<bodyCount){
      if(xyz[1]<g.belowGrilleY)below++;
      if(xyz.every((v,a)=>v>=g.centralRegion.min[a]&&v<=g.centralRegion.max[a]))central++;
      releasedConfig.anchors.forEach((box,i)=>{if(xyz.every((v,a)=>v>=box.min[a]&&v<=box.max[a]))supportCounts[i]++;});
      if(xyz[1]>=bins.minY&&xyz[1]<=bins.maxY&&xyz[2]>=bins.minZ&&xyz[2]<=bins.maxZ)
        centralBinCounts.forEach((_,i)=>{if(xyz[0]>=bins.xEdges[i]&&xyz[0]<bins.xEdges[i+1])centralBinCounts[i]++;});
    }else{loadMass+=m;loadY+=m*xyz[1];}
  }
  return {bodyFractionBelowGrille:below/bodyCount,centralBodyParticles:central,supportCounts,centralBinCounts,loadCenterY:loadY/loadMass,totalMass,center:center.map(v=>v/totalMass),minimumDeterminant:minJ};
}

async function rawRead(device,sim,expected,state){
  const size=sim.count*144,buffer=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ,label:'Independent casting particle readback'});
  try{
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(sim.particleBuffer,0,buffer,0,size);device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);const values=new Float32Array(buffer.getMappedRange().slice(0));
    assert(values.every(Number.isFinite),'Raw GPU state contains nonfinite value');
    for(let i=0;i<sim.count;i++){
      const k=i*36;
      assert(values[k+34]===expected.ids[i]&&values[k+3]===expected.mode[i]&&values[k+7]===Math.fround(expected.mass[i])&&values[k+32]===Math.fround(expected.volume[i]),'Raw GPU identity/mode/mass/volume changed');
      for(let a=0;a<3;a++)assert(values[k+a]===state.x[3*i+a]&&values[k+4+a]===state.v[3*i+a],'Raw GPU position/velocity differs from snapshot');
      for(let col=0;col<3;col++)for(let row=0;row<3;row++)assert(values[k+8+4*col+row]===state.C[i*9+3*row+col]&&values[k+20+4*col+row]===state.F[i*9+3*row+col],'Raw GPU matrix/snapshot mismatch');
    }
    return {float32:values,byteLength:size,strideFloats:36,matrixLayout:'GPU columns, CPU retained arrays row-major',metadataExact:true,snapshotExact:true};
  }finally{if(buffer.mapState==='mapped')buffer.unmap();buffer.destroy();}
}

export async function runProbe(cpuReference){
  const result=window.matterCastingProbeResult={kind:'matter-forge-casting-gpu-probe-v2',status:'failed',stages:[],errors:[],
    comparisonTolerance:{loadCenterYAbsolute:.1,totalCenterAbsolute:.1,bodyFractionBelowGrilleAbsolute:.1},
    lifecycle:{devices:0,simulatorsCreated:0,simulatorsDisposed:0,activeSimulators:0,maximumActiveSimulators:0,completedSteps:0,advanceCalls:0,deviceDestroyed:false}};
  let device,simulator;const abort=new AbortController();let scoped=false;
  try{
    const fixture=castingFixtureV2();result.fixture=fixture;
    assert(fixture.protocol.version===2&&fixture.protocol.castingSteps===960&&fixture.protocol.releasedSteps===960&&fixture.state.ids.length===1216,'Unexpected frozen fixture');
    const adapter=await navigator.gpu?.requestAdapter();assert(adapter,'WebGPU adapter unavailable');
    result.adapter=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,adapter.info?.[k]??null]));
    device=await adapter.requestDevice();result.lifecycle.devices++;
    device.addEventListener('uncapturederror',event=>{result.errors.push(event.error.message);abort.abort();});
    device.pushErrorScope('validation');scoped=true;
    async function stage(id,initial,config,steps){
      assert(!simulator&&result.lifecycle.activeSimulators===0,'Overlapping simulators');
      const record={id,config,initial,frames:[],status:'running',completedSteps:0};result.stages.push(record);
      simulator=await createSimulator(device,config,initial);result.lifecycle.simulatorsCreated++;result.lifecycle.activeSimulators++;
      result.lifecycle.maximumActiveSimulators=Math.max(result.lifecycle.maximumActiveSimulators,result.lifecycle.activeSimulators);
      try{
        const uploaded=await simulator.read();
        for(const key of ['x','v','F','C','mass','volume'])assert(uploaded[key].every((v,i)=>v===Math.fround(initial[key][i])),`Upload mismatch ${key}`);
        assert(same(uploaded.ids,initial.ids)&&same(uploaded.mode,initial.mode),'Upload metadata mismatch');
        record.uploaded=uploaded;record.rawUpload=await rawRead(device,simulator,initial,uploaded);
        let state=uploaded;
        for(let i=0;i<steps;i+=240){
          state=await simulator.advance(Math.min(240,steps-i),{signal:abort.signal});record.completedSteps+=240;result.lifecycle.completedSteps+=240;result.lifecycle.advanceCalls++;
          const measurement=measure(state,fixture);assert(measurement.minimumDeterminant>.05,'Deformation determinant below prospective GPU safety bound');
          for(const key of ['ids','mass','volume','mode'])assert(same(state[key],uploaded[key]),`Stage metadata lost: ${key}`);
          record.frames.push({stageSteps:record.completedSteps,state,measurement});
        }
        record.final=state;record.measurement=measure(state,fixture);record.rawFinal=await rawRead(device,simulator,initial,state);record.status='completed';return state;
      }finally{await device.queue.onSubmittedWorkDone();simulator.dispose();simulator=null;result.lifecycle.activeSimulators--;result.lifecycle.simulatorsDisposed++;}
    }
    const cast=await stage('casting',fixture.state,fixture.castingConfig,960),selection=Array.from({length:fixture.bodyCount},(_,i)=>i);
    const solid=transmute(cast,selection,1),liquid=makeState(cast);
    result.edit={selection,preserved:Object.fromEntries(['ids','x','v','C','mass','volume'].map(key=>[key,same(cast[key],solid[key])])),
      newReferenceIsIdentity:selection.every(p=>Array.from({length:9},(_,i)=>solid.F[9*p+i]===(i%4===0?1:0)).every(Boolean)),
      commonPhysicalState:['ids','x','v','C','mass','volume'].every(key=>same(solid[key],liquid[key])),
      loadUnchanged:Array.from({length:fixture.loadCount},(_,j)=>fixture.bodyCount+j).every(p=>solid.mode[p]===cast.mode[p]&&Array.from({length:9},(_,i)=>solid.F[9*p+i]===cast.F[9*p+i]).every(Boolean))};
    await stage('cast-solid-release',solid,fixture.releasedConfig,960);
    await stage('cast-liquid-release',liquid,fixture.releasedConfig,960);
    const [casting,solidLane,liquidLane]=result.stages,g=fixture.protocol.gate;
    result.gates={allStagesComplete:result.stages.every(s=>s.status==='completed'&&s.completedSteps===960),
      materialPassedGrille:casting.measurement.bodyFractionBelowGrille>g.fractionBelowGrilleGreaterThan,
      centralCastingPresent:casting.measurement.centralBodyParticles>=g.minimumCentralBodyParticles,
      bothSupportsPopulated:casting.measurement.supportCounts.every(c=>c>=g.minimumParticlesInEachSupport),
      allCentralBinsPopulated:casting.measurement.centralBinCounts.every(c=>c>=g.centralBins.minimumParticlesPerBin),
      editPreservesState:Object.values(result.edit.preserved).every(Boolean)&&result.edit.newReferenceIsIdentity&&result.edit.commonPhysicalState&&result.edit.loadUnchanged,
      exactRawMetadataAndUploads:result.stages.every(s=>s.rawUpload.metadataExact&&s.rawFinal.metadataExact&&s.rawFinal.snapshotExact),
      finalDeterminantsSafe:result.stages.every(s=>s.measurement.minimumDeterminant>.05),
      solidSupportsLoad:solidLane.measurement.loadCenterY>=g.solidFinalLoadYAtLeast,
      solidBeatsLiquidControl:solidLane.measurement.loadCenterY-liquidLane.measurement.loadCenterY>=g.solidAboveLiquidByAtLeast,
      massRetained:result.stages.every(s=>s.measurement.totalMass===measure(fixture.state,fixture).totalMass)};
    result.cpuComparison=result.stages.map(s=>{
      const ref=cpuReference.lanes.find(c=>c.id===s.id).measurement,m=s.measurement;
      const errors={loadCenterY:Math.abs(m.loadCenterY-ref.loadCenterY),totalCenter:Math.max(...m.center.map((v,i)=>Math.abs(v-ref.total.center[i]))),bodyFractionBelowGrille:Math.abs(m.bodyFractionBelowGrille-ref.bodyFractionBelowGrille)};
      return{id:s.id,reference:ref,errors,pass:errors.loadCenterY<=.1&&errors.totalCenter<=.1&&errors.bodyFractionBelowGrille<=.1};
    });
    result.gates.coarseCPUReferenceAgreement=result.cpuComparison.every(c=>c.pass);
    assert(Object.values(result.gates).every(Boolean),'One or more prospective casting gates failed');
    result.status='passed';
  }catch(error){result.failure=String(error.stack??error);}
  finally{
    try{
      if(device){await device.queue.onSubmittedWorkDone();if(simulator){simulator.dispose();simulator=null;result.lifecycle.activeSimulators--;result.lifecycle.simulatorsDisposed++;}
        if(scoped){const error=await device.popErrorScope();if(error)result.errors.push(error.message);}
        const lost=device.lost;device.destroy();result.lifecycle.deviceDestroyed=true;result.lifecycle.deviceLossReason=(await lost).reason;
      }
    }catch(error){result.cleanupFailure=String(error.stack??error);result.status='failed';}
    if(result.errors.length)result.status='failed';
    result.lifecycle.pass=result.lifecycle.devices===1&&result.lifecycle.simulatorsCreated===3&&result.lifecycle.simulatorsDisposed===3&&result.lifecycle.activeSimulators===0&&result.lifecycle.maximumActiveSimulators===1&&result.lifecycle.completedSteps===2880&&result.lifecycle.advanceCalls===12&&result.lifecycle.deviceDestroyed&&result.lifecycle.deviceLossReason==='destroyed';
    if(!result.lifecycle.pass)result.status='failed';
  }
  return result;
}
