// Prospective, bounded CPU interaction check. No Three renderer/GPU imports.
// This is sampled numerical/interaction evidence, not FEM or physical fidelity.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { prepareBasis, evaluateBasis } from '../experiments/local-elasticity/basis.mjs';
import { createMechanics } from '../experiments/local-elasticity/mechanics.mjs';
import { decodeSplat, normalizePositions, sampleQuadrature } from '../experiments/local-elasticity/assets.mjs';

const root=fileURLToPath(new URL('../',import.meta.url)),sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const settings=Object.freeze({points:256,centers:48,modesIncludingConstant:6,basisYoung:1000,
  young:40,density:1,poisson:.3,dt:1/60,dragSteps:30,releaseSteps:30,
  dragPercentile:.85,dragOffset:[.25,.08,0],dragStiffness:25,gravity:[0,-1.5,0],floor:-.62,
  maxIterations:10,maxEvaluations:160,maxMilliseconds:100,quadratureSeed:12345,
  heldoutCount:5000,heldoutSeed:98765,globalMilliseconds:15000,
  counterfactualStepsPerAsset:2,solverCallsMaximum:124});
const report={kind:'local-elasticity-interaction-cpu-v1',created:new Date().toISOString(),settings,
  scope:'Two actual source point clouds, UI solver controls, sampled deformation/interaction checks. No GPU, no recovered solid volume, no FEM/material calibration, no guarantee between checked points.',
  gates:{pinAndFloorTolerance:1e-8,objectiveRelativeAllowance:1e-9,minimumCheckedDeterminant:0,
    independentFComponentTolerance:1e-10,changedInputPointResponse:1e-6,
    convergence:'Every solve status is retained. Finite-budget nonconvergence is reported separately and is not silently converted to convergence.'},
  sources:{},assets:[],failures:[]};
const directory=path.join(root,'results/development/local-elasticity-interaction',report.created.replaceAll(':','-'));
await mkdir(directory,{recursive:true});
for(const name of ['scripts/check-local-elasticity-interaction.mjs','experiments/local-elasticity/assets.mjs',
  'experiments/local-elasticity/basis.mjs','experiments/local-elasticity/mechanics.mjs','experiments/local-elasticity/worker.mjs']){
  const bytes=await readFile(path.join(root,name));report.sources[name]={sha256:sha(bytes),text:bytes.toString('utf8')};
}
const started=performance.now();
const checkDeadline=()=>{if(performance.now()-started>=settings.globalMilliseconds)throw Error('Global cooperative 15-second CPU budget reached; incomplete evidence retained.');};
const vectorAt=(values,i)=>Array.from(values.subarray(i*3,i*3+3));
const maxDifference=(a,b)=>{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]-b[i]));return m;};
const determinant=F=>F[0]*(F[4]*F[8]-F[5]*F[7])-F[1]*(F[3]*F[8]-F[5]*F[6])+F[2]*(F[3]*F[7]-F[4]*F[6]);
function sourceSubset(positions,excluded,count,seed){
  const candidates=[];for(let i=0;i<positions.length/3;i++)if(!excluded.has(i))candidates.push(i);
  if(candidates.length<count)throw Error('Not enough independent held-out source points');
  let state=seed>>>0;
  for(let i=0;i<count;i++){
    state^=state<<13;state^=state>>>17;state^=state<<5;
    const j=i+Math.floor((state>>>0)/4294967296*(candidates.length-i));
    [candidates[i],candidates[j]]=[candidates[j],candidates[i]];
  }
  const indices=candidates.slice(0,count),selected=new Float64Array(count*3);
  indices.forEach((index,i)=>selected.set(positions.subarray(index*3,index*3+3),i*3));
  return {indices,positions:selected};
}
// Direct Eq.1 spatial differentiation, independent of mechanics' assembled G.
// All reported values here use Float64, not a simulated GPU arithmetic claim.
function deformation(q,positions,fields){
  const count=positions.length/3,H=fields.modeCount,out=new Float64Array(positions.length),
    Fs=new Float64Array(count*9),dets=new Float64Array(count),displacements=new Float64Array(count);
  let minDeterminant=Infinity,maxDisplacement=0,maxIndex=-1,inverted=0,nearCollapsed=0,meanSquaredDisplacement=0;
  for(let i=0;i<count;i++){
    const F=new Float64Array([1,0,0,0,1,0,0,0,1]),u=[0,0,0];
    for(let h=0;h<H;h++)for(let r=0;r<3;r++){
      const offset=h*12+r*4,w=fields.weights[i*H+h];let affine=q[offset+3];
      for(let a=0;a<3;a++)affine+=q[offset+a]*positions[i*3+a];
      u[r]+=w*affine;
      for(let a=0;a<3;a++)F[r*3+a]+=w*q[offset+a]+fields.gradients[(i*H+h)*3+a]*affine;
    }
    out.set(u.map((v,r)=>positions[i*3+r]+v),i*3);Fs.set(F,i*9);
    const det=determinant(F),length=Math.hypot(...u);dets[i]=det;displacements[i]=length;
    if(!Number.isFinite(det)||!Number.isFinite(length))throw Error('Nonfinite independently evaluated deformation');
    minDeterminant=Math.min(minDeterminant,det);if(length>maxDisplacement){maxDisplacement=length;maxIndex=i;}
    if(det<=0)inverted++;if(det>0&&det<.1)nearCollapsed++;meanSquaredDisplacement+=length*length/count;
  }
  return {positions:out,deformationGradients:Fs,determinants:dets,displacements,
    summary:{count,minDeterminant,inverted,positiveBelowPointOne:nearCollapsed,maxDisplacement,
      maximumDisplacementIndex:maxIndex,rmsDisplacement:Math.sqrt(meanSquaredDisplacement)}};
}
function summarizeSolves(steps){
  const statuses={};let minDeterminant=Infinity,maxPinError=0,maxFloorPenetration=0,maxCompletedWallMs=0,objectiveFailures=0;
  for(const step of steps){const d=step.diagnostics;statuses[d.status]=(statuses[d.status]??0)+1;
    minDeterminant=Math.min(minDeterminant,d.minDeterminant);maxPinError=Math.max(maxPinError,d.pinError);
    maxFloorPenetration=Math.max(maxFloorPenetration,d.floorPenetration);maxCompletedWallMs=Math.max(maxCompletedWallMs,d.completedWallMs);
    if(d.finalEnergy>d.initialEnergy+1e-9*Math.max(1,Math.abs(d.initialEnergy)))objectiveFailures++;
  }
  return {count:steps.length,statuses,minDeterminant,maxPinError,maxFloorPenetration,maxCompletedWallMs,objectiveFailures};
}
try{
  for(const asset of ['plant','spot']){
    checkDeadline();const assetStarted=performance.now(),bytes=await readFile(path.join(root,'experiments/local-elasticity/assets',`${asset}.splat`));
    const decoded=decodeSplat(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)),normalized=normalizePositions(decoded.positions),
      samples=sampleQuadrature(normalized.positions,settings.points,settings.quadratureSeed),excluded=new Set(samples.indices);
    const ordered=Array.from({length:decoded.count},(_,i)=>i).sort((a,b)=>normalized.positions[3*a+1]-normalized.positions[3*b+1]||a-b);
    // First source point at/above p85 that is not one of the integration points.
    const dragIndex=ordered.slice(Math.floor(settings.dragPercentile*(ordered.length-1))).find(i=>!excluded.has(i));
    const dragPosition=vectorAt(normalized.positions,dragIndex),dragFieldsPosition=Float64Array.from(dragPosition);
    const basis=prepareBasis({positions:samples.positions,volumes:samples.volumes,centerCount:settings.centers,
      modeCount:settings.modesIncludingConstant,young:settings.basisYoung,poisson:settings.poisson}),fields=evaluateBasis(basis,samples.positions);
    const input={restPositions:samples.positions,volumes:samples.volumes,...fields,density:settings.density,young:settings.young,poisson:settings.poisson},
      model=createMechanics(input),dragFields=evaluateBasis(basis,dragFieldsPosition),heldout=sourceSubset(normalized.positions,excluded,settings.heldoutCount,settings.heldoutSeed),
      heldoutFields=evaluateBasis(basis,heldout.positions);
    const lowest=Array.from({length:settings.points},(_,i)=>i).sort((a,b)=>samples.positions[a*3+1]-samples.positions[b*3+1]||a-b),
      pins=lowest.slice(0,3).map(index=>({index,target:vectorAt(samples.positions,index)})),
      drag={restPosition:dragPosition,weights:dragFields.weights,target:dragPosition.map((v,a)=>v+settings.dragOffset[a]),stiffness:settings.dragStiffness};
    const record={asset,assetSha256:sha(bytes),assetBytes:bytes.length,sourcePointCount:decoded.count,
      normalization:{center:normalized.center,scale:normalized.scale,flipY:normalized.flipY},
      quadratureIndices:samples.indices,quadratureAssumption:samples.assumption,input,basis,
      massDiagnostics:model.diagnostics,pins,dragSourceIndex:dragIndex,drag,heldout:{...heldout,fields:heldoutFields},
      steps:[],counterfactuals:[],endpoints:[]};report.assets.push(record);
    let state=model.createState();
    const options={dt:settings.dt,gravity:settings.gravity,pins,floor:settings.floor,maxIterations:settings.maxIterations,
      maxEvaluations:settings.maxEvaluations,maxMilliseconds:settings.maxMilliseconds};
    for(let index=0;index<settings.dragSteps+settings.releaseSteps;index++){
      checkDeadline();const currentDrag=index<settings.dragSteps?drag:null;
      // These controls change only the spring input, from the exact same state.
      let control=null;
      if(index===0||index===settings.dragSteps){
        control=model.step(state,{...options,drag:index===0?null:drag});
        record.counterfactuals.push({index,input:index===0?'no-drag':'continued-drag',q:control.state.q,
          velocity:control.state.velocity,diagnostics:control.diagnostics});checkDeadline();
      }
      const next=model.step(state,{...options,drag:currentDrag});
      if(!next.state.q.every(Number.isFinite)||Math.max(...next.state.q.map(Math.abs))>100)throw Error('State left the UI numerical envelope');
      const step={index,phase:currentDrag?'drag':'release',q:next.state.q,velocity:next.state.velocity,diagnostics:next.diagnostics};
      if(control){const a=model.evaluatePositions(next.state.q),b=model.evaluatePositions(control.state.q);
        step.changedInputMaximumPointResponse=maxDifference(a,b);step.changedInputMaximumQResponse=maxDifference(next.state.q,control.state.q);}
      record.steps.push(step);state=next.state;
      if(index===settings.dragSteps-1||index===settings.dragSteps+settings.releaseSteps-1){
        const quad=deformation(state.q,samples.positions,fields),held=deformation(state.q,heldout.positions,heldoutFields),native=model.energyGradient(state.q),
          point=deformation(state.q,dragFieldsPosition,dragFields),fError=maxDifference(quad.deformationGradients,native.deformationGradients);
        record.endpoints.push({phase:currentDrag?'pulled':'released',q:state.q,velocity:state.velocity,
          quadrature:{summary:quad.summary,determinants:quad.determinants,displacements:quad.displacements},
          heldout:{summary:held.summary,determinants:held.determinants,displacements:held.displacements},
          dragPosition:Array.from(point.positions),targetDistance:Math.hypot(...Array.from(point.positions,(v,a)=>v-drag.target[a])),
          independentVsMechanicsFMaxAbsolute:fError});
      }
    }
    record.trajectorySummary=summarizeSolves(record.steps);record.allSolveSummary=summarizeSolves([...record.steps,...record.counterfactuals]);
    const summary=record.allSolveSummary;
    record.gates={completed:record.steps.length===60&&record.counterfactuals.length===2&&record.endpoints.length===2,
      finiteFeasibleMonotone:summary.maxPinError<=1e-8&&summary.maxFloorPenetration<=1e-8&&summary.objectiveFailures===0,
      positiveQuadratureDeterminants:summary.minDeterminant>0&&record.endpoints.every(p=>p.quadrature.summary.minDeterminant>0),
      positiveHeldoutDeterminants:record.endpoints.every(p=>p.heldout.summary.minDeterminant>0),
      independentDeformationParity:record.endpoints.every(p=>p.independentVsMechanicsFMaxAbsolute<=1e-10),
      changedInputsChangeNextState:record.steps.filter(p=>p.changedInputMaximumPointResponse!==undefined).every(p=>p.changedInputMaximumPointResponse>1e-6),
      allSolverCallsConverged:summary.statuses.converged===62};
    record.pass=Object.values(record.gates).every(Boolean);record.completedWallMs=performance.now()-assetStarted;
    console.log(JSON.stringify({asset,pass:record.pass,gates:record.gates,solves:summary,
      endpoints:record.endpoints.map(p=>({phase:p.phase,quadrature:p.quadrature.summary,heldout:p.heldout.summary,targetDistance:p.targetDistance})),
      changedInputs:record.steps.filter(p=>p.changedInputMaximumPointResponse!==undefined).map(p=>({index:p.index,response:p.changedInputMaximumPointResponse}))}));
  }
  report.status=report.assets.length===2&&report.assets.every(a=>a.pass)?'passed':'failed-gates';
}catch(error){report.status='failed-or-incomplete';report.failures.push(String(error.stack??error).replaceAll(root,'<workspace>/'));}
report.completedWallMs=performance.now()-started;
const encoded=JSON.stringify(report,(_,value)=>ArrayBuffer.isView(value)?Array.from(value):value,2)+'\n';
await writeFile(path.join(directory,'report.json'),encoded,{flag:'wx'});
await writeFile(path.join(directory,'report.sha256'),`${sha(encoded)}  report.json\n`,{flag:'wx'});
console.log(JSON.stringify({status:report.status,report:path.relative(root,path.join(directory,'report.json')),sha256:sha(encoded),completedWallMs:report.completedWallMs}));
if(report.status!=='passed')process.exitCode=1;
