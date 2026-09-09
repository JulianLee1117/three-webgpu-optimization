/** CPU-only audit of retained v2 evidence; never calls a solver or uses a GPU.
 * Default reads the tracked compressed CPU v2 report. --report=... accepts an
 * original report plus report.sha256. --self-test rejects intentional corruption.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import assert from 'node:assert/strict';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const EVIDENCE='experiments/constraint-editing/evidence';
const FIRST_REPORT='results/development/constrained-motion-cpu/2026-09-09T21-04-49.206Z/report.json';
const FIRST_SHA='69fe78ff3fa33b5af96a993812752884704fd2071b0606881830adc830f1cd21';
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const plain=x=>JSON.parse(JSON.stringify(x));
const near=(a,b,label,absolute=1e-9,relative=1e-9)=>assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=absolute+relative*Math.abs(b),`${label}: ${a} != ${b}`);
function compare(a,b,label) {
  if(typeof b==='number')return near(a,b,label);
  if(b===null||typeof b!=='object')return assert.equal(a,b,label);
  assert.equal(Array.isArray(a),Array.isArray(b),`${label}: array type`);
  assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),`${label}: keys`);
  for(const k of Object.keys(b))compare(a[k],b[k],`${label}/${k}`);
}
const finiteArray=(a,n,label)=>assert.ok(Array.isArray(a)&&a.length===n&&a.every(Number.isFinite),`${label}: invalid finite array`);
const safeRelative=name=>typeof name==='string'&&!path.isAbsolute(name)&&!name.split(/[\\/]/).includes('..');
async function exists(p){try{await fs.access(p);return true;}catch{return false;}}
async function insideRead(base,name) {
  assert.ok(safeRelative(name),'Unsafe evidence path');
  const p=await fs.realpath(path.join(base,name)),relative=path.relative(base,p);
  assert.ok(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative),'Evidence path escapes root');
  return fs.readFile(p);
}

function numericPositions(fixture,params,modules) {
  const out=[];
  for(let i=0;i<fixture.samples.length/4;i++)out.push(...modules.programs.numericProgram(fixture.kind,
    Array.from(fixture.samples.slice(i*4,i*4+3)),fixture.samples[i*4+3],params));
  return out;
}

/** Reusable by the GPU auditor. Numeric reference and retained-position systems
 * are separate: Float32 GPU observations need not equal the double CPU oracle.
 */
export function recomputeConstrainedFinal(fit,fixture,modules) {
  const referencePositions=numericPositions(fixture,fit.params,modules);
  const numeric=modules.solverConstrained.constrainedSystem(fixture,fit.params,{positions:referencePositions});
  const retained=modules.solverConstrained.constrainedSystem(fixture,fit.params,{positions:fit.finalPositions});
  const objectiveCost=system=>system.objective.reduce((sum,r)=>sum+.5*r.value*r.value,0);
  return {referencePositions,numericDiagnostics:numeric.diagnostics,numericObjectiveCost:objectiveCost(numeric),
    retainedDiagnostics:retained.diagnostics,retainedObjectiveCost:objectiveCost(retained),
    validation:modules.fixtures.validate(fixture,fit.params),float32Validation:modules.fixtures.validate(fixture,Array.from(new Float32Array(fit.params))),
    uniformPatch:{kind:fixture.kind,variant:fixture.variant,parameters:[...fit.params],period:1}};
}

// Standard arithmetic reconstruction of a recorded PHR model, independent of
// solveConstrained's loop. No new state is selected or optimization performed.
function model(system,le,li,rho,P) {
  let cost=0;const g=Array(P).fill(0),H=Array.from({length:P},()=>Array(P).fill(0));
  const add=(v,d)=>{cost+=v*v/2;for(let k=0;k<P;k++){g[k]+=v*d[k];for(let j=0;j<=k;j++)H[k][j]+=d[k]*d[j];}};
  system.objective.forEach(r=>add(r.value,r.derivative));
  const scale=Math.sqrt(rho);
  system.equalities.forEach((r,i)=>add(scale*(r.value+le[i]/rho),r.derivative.map(x=>x*scale)));
  system.inequalities.forEach((r,i)=>{const shift=r.value+li[i]/rho;if(shift>0)add(scale*shift,r.derivative.map(x=>x*scale));});
  for(let k=0;k<P;k++)for(let j=0;j<k;j++)H[j][k]=H[k][j];
  return {cost,g,H};
}
function costOnly(system,le,li,rho) {
  let cost=system.objective.reduce((sum,r)=>sum+r.value*r.value/2,0);
  system.equalities.forEach((r,i)=>{const v=Math.sqrt(rho)*(r.value+le[i]/rho);cost+=v*v/2;});
  system.inequalities.forEach((r,i)=>{const v=Math.sqrt(rho)*Math.max(0,r.value+li[i]/rho);cost+=v*v/2;});
  return cost;
}
function linearStep(A,b) {
  const n=b.length,M=A.map((row,i)=>[...row,b[i]]);
  for(let k=0;k<n;k++) {
    let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[pivot][k]))pivot=i;
    if(Math.abs(M[pivot][k])<1e-20)return null;
    [M[k],M[pivot]]=[M[pivot],M[k]];
    for(let i=k+1;i<n;i++){const factor=M[i][k]/M[k][k];for(let j=k;j<=n;j++)M[i][j]-=factor*M[k][j];}
  }
  const x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let value=M[i][n];for(let j=i+1;j<n;j++)value-=M[i][j]*x[j];x[i]=value/M[i][i];}
  return x.every(Number.isFinite)?x:null;
}

export async function auditConstrainedSolve(fit,fixture,modules,counters={states:0,trialCandidates:0,trainingPoints:0}) {
  const L=modules.solverConstrained.CONSTRAINED_LIMITS,P=fixture.initial.length,label=`${fixture.kind}/${fixture.variant}`;
  finiteArray(fit.params,P,`${label}/params`);
  const bounded=p=>p.forEach((v,k)=>assert.ok(v>=fixture.lower[k]&&v<=fixture.upper[k],`${label}: parameter outside bounds`));
  bounded(fit.params);finiteArray(fit.finalPositions,fixture.samples.length/4*3,`${label}/finalPositions`);
  const final=recomputeConstrainedFinal(fit,fixture,modules);
  compare(fit.finalPositions,final.referencePositions,`${label}/finalPositions`);
  compare(fit.diagnostics,final.numericDiagnostics,`${label}/diagnostics`);
  near(fit.objectiveCost,final.numericObjectiveCost,`${label}/objectiveCost`);
  compare(fit.validation,final.validation,`${label}/validation`);compare(fit.float32Validation,final.float32Validation,`${label}/float32Validation`);
  compare(fit.uniformPatch,final.uniformPatch,`${label}/uniformPatch`);
  assert.ok(Array.isArray(fit.trace)&&fit.trace.length>0&&fit.trace.length<=L.outer,`${label}: outer budget`);
  const evaluate=modules.solver.makeNumericEvaluator(fixture.kind),systemAt=async(p,jacobian)=>{
    const evaluation=await evaluate(p,fixture.samples,{jacobian});counters.trainingPoints+=evaluation.positions.length/3;
    return modules.solverConstrained.constrainedSystem(fixture,p,evaluation);
  };
  let current=[...fixture.initial],penalty=L.initialPenalty,damping=L.initialDamping,previousViolation=Infinity;
  let le,li,accepted=0,evaluations=1,converged=false;
  for(let outer=0;outer<fit.trace.length;outer++) {
    const round=fit.trace[outer];assert.equal(round.outer,outer);near(round.penalty,penalty,`${label}/penalty`);
    assert.ok(Array.isArray(round.iterations)&&round.iterations.length<=L.inner,`${label}: inner budget`);
    for(let inner=0;inner<round.iterations.length;inner++) {
      const entry=round.iterations[inner];assert.equal(entry.inner,inner);
      finiteArray(entry.params,P,`${label}/trace/params`);bounded(entry.params);
      compare(entry.params,current,`${label}/state advancement`);near(entry.damping,damping,`${label}/damping`);
      const system=await systemAt(entry.params,true);evaluations++;counters.states++;
      le??=Array(system.equalities.length).fill(0);li??=Array(system.inequalities.length).fill(0);
      const m=model(system,le,li,penalty,P);near(entry.cost,m.cost,`${label}/cost`);compare(entry.diagnostics,system.diagnostics,`${label}/trace diagnostics`);
      const A=m.H.map((row,k)=>row.map((v,j)=>v+(j===k?damping*Math.max(m.H[k][k],1e-6):0)));
      const step=linearStep(A,m.g.map(v=>-v));
      assert.ok(Array.isArray(entry.trials)&&entry.trials.length<=L.backtracks,`${label}: backtrack budget`);
      let improved=false;
      if(step) {
        assert.notEqual(entry.singular,true,`${label}: false singular flag`);
        const cap=Math.min(1,L.maxStepNorm/Math.max(Math.hypot(...step),1e-20));
        for(let trial=0;trial<entry.trials.length;trial++) {
          const item=entry.trials[trial];assert.equal(item.trial,trial);
          const scale=cap*2**(-trial),candidate=current.map((v,k)=>Math.max(fixture.lower[k],Math.min(fixture.upper[k],v+step[k]*scale)));
          const directional=m.g.reduce((sum,g,k)=>sum+g*(candidate[k]-current[k]),0),candidateSystem=await systemAt(candidate,false);
          const cost=costOnly(candidateSystem,le,li,penalty),stepNorm=Math.hypot(...candidate.map((v,k)=>v-current[k]));
          evaluations++;counters.trialCandidates++;
          near(item.cost,cost,`${label}/candidate cost`,2e-8,2e-8);near(item.directional,directional,`${label}/directional`,2e-8,2e-8);
          near(item.stepNorm,stepNorm,`${label}/step norm`);assert.ok(stepNorm<=L.maxStepNorm+1e-10);
          const decision=directional<0&&cost<=m.cost+L.armijo*directional;
          assert.equal(item.accepted,decision,`${label}: Armijo decision differs`);
          if(item.accepted){assert.equal(trial,entry.trials.length-1);current=candidate;improved=true;accepted++;damping=Math.max(1e-9,damping/2);break;}
        }
      } else {assert.equal(entry.singular,true);assert.equal(entry.trials.length,0);}
      if(!improved)damping=Math.min(1e9,damping*10);
    }
    assert.ok(round.final,`${label}: incomplete outer round`);compare(round.final.params,current,`${label}/outer final params`);
    const system=await systemAt(current,true);evaluations++;
    le=le.map((v,i)=>v+penalty*system.equalities[i].value);li=li.map((v,i)=>Math.max(0,v+penalty*system.inequalities[i].value));
    compare(round.final.lambdaE,le,`${label}/equality multipliers`);compare(round.final.lambdaI,li,`${label}/inequality multipliers`);
    compare(round.final.diagnostics,system.diagnostics,`${label}/outer diagnostics`);
    const lagrangian=Array(P).fill(0);
    system.objective.forEach(r=>r.derivative.forEach((v,k)=>lagrangian[k]+=r.value*v));
    system.equalities.forEach((r,i)=>r.derivative.forEach((v,k)=>lagrangian[k]+=le[i]*v));
    system.inequalities.forEach((r,i)=>r.derivative.forEach((v,k)=>lagrangian[k]+=li[i]*v));
    const stationarity=Math.max(...lagrangian.map((g,k)=>(current[k]<=fixture.lower[k]+1e-10&&g>0)||(current[k]>=fixture.upper[k]-1e-10&&g<0)?0:Math.abs(g)));
    near(round.final.stationarity,stationarity,`${label}/stationarity`);
    const violation=system.diagnostics.maximumViolation;
    converged=violation<=L.feasibilityTolerance&&stationarity<=L.stationarityTolerance;
    if(converged){assert.equal(outer,fit.trace.length-1);break;}
    if(violation>previousViolation*.5)penalty=Math.min(L.maxPenalty,penalty*L.penaltyGrowth);
    previousViolation=violation;
  }
  compare(fit.params,current,`${label}/final state`);compare(fit.lambdaE,le,`${label}/final equality multipliers`);compare(fit.lambdaI,li,`${label}/final inequality multipliers`);
  near(fit.penalty,penalty,`${label}/final penalty`);assert.equal(fit.accepted,accepted);assert.equal(fit.evaluations,evaluations);
  assert.equal(fit.status,converged?'converged':'iteration-limit');
  assert.ok(Number.isFinite(fit.elapsedMs)&&fit.elapsedMs>=0);
  return {params:fit.params,status:fit.status,accepted,evaluations,recordedElapsedMs:fit.elapsedMs,diagnostics:final.numericDiagnostics,
    objectiveCost:final.numericObjectiveCost,validation:final.validation,float32Validation:final.float32Validation};
}

export async function auditConstrainedCPUReport(report,modules) {
  const {auditSolve}=await import('./analyze-constraint-editing.mjs');
  const {CASES,CONSTRAINED_LIMITS,fixtureVariant}=modules.solverConstrained,{LIMITS,validate}=modules.fixtures;
  assert.equal(report.kind,'constrained-motion-cpu-v2');assert.equal(report.sourceUnchanged,true);
  compare(report.limits,LIMITS,'limits');compare(report.constrainedLimits,CONSTRAINED_LIMITS,'constrained limits');
  compare(report.cells.map(c=>({kind:c.kind,variant:c.variant})),CASES,'case order');
  const cells=[],counters={states:0,trialCandidates:0,trainingPoints:0,recomputedTrainingPoints:0,recomputedHeldoutValidations:0};
  for(const cell of report.cells) {
    const fixture=fixtureVariant(cell.kind,cell.variant),label=`${cell.kind}/${cell.variant}`;
    compare(cell.fixture,{...fixture,samples:Array.from(fixture.samples)},`${label}/fixture`);
    const witness=validate(fixture,fixture.witness),initial=validate(fixture,fixture.initial);
    compare(cell.witness,witness,`${label}/witness`);compare(cell.initial,initial,`${label}/initial`);assert.equal(witness.pass,true);
    const tip=modules.programs.numericProgram(fixture.kind,fixture.target.p,fixture.target.time,fixture.initial),translation=fixture.target.position.map((v,k)=>v-tip[k]);
    compare(cell.translation,{translation,validation:validate(fixture,fixture.initial,{translation})},`${label}/translation`);
    const handleOnly=auditSolve(cell.handleOnly,fixture,`${label}/handleOnly`,false,modules,counters);
    const constrained=await auditConstrainedSolve(cell.constrained,fixture,modules,counters);
    const gates={trainingFeasibility:constrained.diagnostics.maximumViolation<=CONSTRAINED_LIMITS.feasibilityTolerance,
      quality:constrained.validation.pass,float32Export:constrained.float32Validation.pass,
      controlNecessity:handleOnly.validation.minClearance<=-LIMITS.controlPenetration,
      bounded:!['stopped','wall-limit'].includes(handleOnly.status)&&constrained.status!=='stopped'};
    compare(cell.gates,gates,`${label}/gates`);const status=Object.values(gates).every(Boolean)?'passed':'failed';assert.equal(cell.status,status);
    cells.push({kind:cell.kind,variant:cell.variant,status,gates,witness,initial,handleOnly,constrained});
  }
  assert.ok(Number.isFinite(report.elapsedMs)&&report.elapsedMs>=0);
  const capabilityPassed=report.elapsedMs<=20000&&cells.every(c=>c.status==='passed');assert.equal(report.status,capabilityPassed?'passed':'failed');
  return {capabilityPassed,experimentStatus:capabilityPassed?'passed':'failed',recordedElapsedMs:report.elapsedMs,cells,counters,
    limitations:['CPU-only synthetic, known-family feasibility screen; no GPU speed or general procedural-editor claim.',
      'Dense held-out temporal samples are not continuous collision detection.',
      'Constrained trace proposals are reconstructed from retained states; no new optimization or parameter selection occurs.',
      'Trace audit uses the pinned analytic residual chain with independently recomputed numeric forward positions and FD parameter Jacobians.',
      'The solver method and budget both changed from v1, so success cannot isolate one inequality as the cause.',
      'Recorded wall times are observations, not independently reproducible timing measurements.']};
}

async function loadInputs(args) {
  const explicit=args.filter(x=>x.startsWith('--report=')).map(x=>x.slice(9)),inputs=[];
  if(explicit.length)for(const filename of explicit) {
    const absolute=path.resolve(ROOT,filename),bytes=await fs.readFile(absolute),digest=hash(bytes);
    assert.ok(bytes.length<64*1024*1024);assert.equal((await fs.readFile(path.join(path.dirname(absolute),'report.sha256'),'utf8')).trim().split(/\s+/)[0],digest);
    if(path.relative(ROOT,absolute).replaceAll('\\','/')===FIRST_REPORT)assert.equal(digest,FIRST_SHA);
    inputs.push({id:path.basename(path.dirname(absolute)),report:JSON.parse(bytes),sha256:digest});
  } else {
    const base=path.join(ROOT,EVIDENCE),manifestBytes=await insideRead(base,'manifest.json');
    assert.equal((await insideRead(base,'manifest.sha256')).toString('utf8').trim().split(/\s+/)[0],hash(manifestBytes),'Manifest checksum mismatch');
    const manifest=JSON.parse(manifestBytes);
    assert.equal(manifest.kind,'constraint-editing-public-evidence-v1');
    assert.ok(Array.isArray(manifest.reports)&&manifest.reports.length>0&&manifest.reports.length<=32,'Manifest report budget');
    for(const item of manifest.reports.filter(r=>r.reportKind==='constrained-motion-cpu-v2')) {
      const compressed=await insideRead(base,item.file);assert.equal(hash(compressed),item.gzipSHA256);assert.equal(compressed.length,item.gzipBytes);
      const bytes=gunzipSync(compressed,{maxOutputLength:64*1024*1024});assert.equal(hash(bytes),item.publicJSONSHA256);assert.equal(bytes.length,item.publicBytes);
      assert.equal(item.originalRawSHA256,item.publicJSONSHA256);assert.deepEqual(item.redactions,[]);
      const snapshot=manifest.sourceSnapshots[item.sourceSnapshot];assert.ok(snapshot&&safeRelative(snapshot.directory));
      inputs.push({id:item.id,report:JSON.parse(bytes),sha256:item.publicJSONSHA256,snapshot,snapshotBase:path.join(base,snapshot.directory)});
    }
  }
  assert.ok(inputs.length>0&&inputs.length<=8,'Missing/bounded CPU-v2 cohort');return inputs;
}

async function modulesFor(input) {
  let base=ROOT,live=true;
  for(const [filename,expected] of Object.entries(input.report.sourceHashes)) {
    assert.match(expected,/^[a-f0-9]{64}$/);assert.ok(safeRelative(filename));
    if(input.snapshot){assert.equal(input.snapshot.files[filename],expected);assert.equal(hash(await insideRead(input.snapshotBase,filename)),expected,`Pinned source ${filename}`);}
    if(!await exists(path.join(ROOT,filename))||hash(await insideRead(ROOT,filename))!==expected)live=false;
  }
  if(!live){assert.ok(input.snapshotBase,'Live source changed and no pinned snapshot available');base=input.snapshotBase;}
  const load=name=>import(pathToFileURL(path.join(base,`experiments/constraint-editing/${name}`)).href);
  const [programs,fixtures,solver,solverConstrained]=await Promise.all([load('programs.js'),load('fixtures.mjs'),load('solver.mjs'),load('solver-constrained.mjs')]);
  return {programs,fixtures,solver,solverConstrained,sourceBase:live?'live files matching report hashes':input.snapshot.directory};
}

async function main() {
  const args=process.argv.slice(2);for(const arg of args)assert.ok(arg.startsWith('--report=')||['--self-test','--require-capability','--help'].includes(arg),`Unknown argument ${arg}`);
  if(args.includes('--help')){console.log('node scripts/analyze-constrained-motion.mjs [--report=report.json] [--self-test] [--require-capability]');return;}
  const inputs=await loadInputs(args),summaries=[];
  for(const input of inputs) {
    const modules=await modulesFor(input),summary=await auditConstrainedCPUReport(input.report,modules);
    const corruptionTests=[];
    if(args.includes('--self-test'))for(const [label,mutate] of [
      ['changed variant witness',r=>r.cells[0].fixture.witness[0]+=.01],
      ['changed retained position',r=>r.cells[0].constrained.finalPositions[0]+=.01],
      ['false training feasibility',r=>r.cells[0].constrained.diagnostics.maximumViolation=-1],
      ['corrupted trial cost',r=>r.cells[0].constrained.trace[0].iterations[0].trials[0].cost+=1]]) {
      const bad=plain(input.report);mutate(bad);await assert.rejects(()=>auditConstrainedCPUReport(bad,modules));corruptionTests.push(label);
    }
    summaries.push({id:input.id,reportSHA256:input.sha256,sourceBase:modules.sourceBase,...summary,corruptionTests});
  }
  const output={kind:'constrained-motion-cpu-analysis-v2',created:new Date().toISOString(),verificationPassed:true,reports:summaries};
  const directory=path.join(ROOT,'results/development/constrained-motion-analysis');await fs.mkdir(directory,{recursive:true});
  const filename=path.join(directory,output.created.replaceAll(':','-')+'.json'),bytes=Buffer.from(JSON.stringify(output,null,2)+'\n');
  await fs.writeFile(filename,bytes,{flag:'wx'});await fs.writeFile(filename.replace(/\.json$/,'.sha256'),`${hash(bytes)}  ${path.basename(filename)}\n`,{flag:'wx'});
  console.log(JSON.stringify({verificationPassed:true,capabilityPassed:summaries.every(r=>r.capabilityPassed),reportCount:summaries.length,
    output:path.relative(ROOT,filename).replaceAll('\\','/'),sha256:hash(bytes),counters:summaries.map(r=>r.counters)}));
  if(args.includes('--require-capability')&&summaries.some(r=>!r.capabilityPassed))process.exitCode=1;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
