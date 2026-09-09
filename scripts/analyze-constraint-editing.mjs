/**
 * CPU-only audit of retained constraint-editing evidence. Does not optimize,
 * launch a browser, request a GPU, or infer success from report status flags.
 *
 * node scripts/analyze-constraint-editing.mjs
 * node scripts/analyze-constraint-editing.mjs --report=path/to/report.json
 * node scripts/analyze-constraint-editing.mjs --self-test
 *
 * Default: supported records from committed evidence/manifest.json when available;
 * otherwise the first fixed CPU screen and the two pinned initial GPU probes.
 * This verifier runs only CPU code. A verified failed experiment exits 0.
 * --require-capability additionally exits 1 when its research gate failed.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';
import assert from 'node:assert/strict';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIRST_CPU_REPORT = 'results/development/constraint-editing-cpu/2026-09-09T20-54-27.837Z/report.json';
const FIRST_CPU_SHA256 = 'f57da1bf064596d130c872554b6f8ec0875cedf51861a4b38fa7076253400369';
const GPU_PARITY_REPORTS = {
  'results/development/constraint-editing-gpu/2026-09-09T20-59-37.966Z-tentacle-parity/report.json':'172fd9dff96b9b0c01f58f7060abf4451bbbafa7eca89c5ceaebf781cb8bd768',
  'results/development/constraint-editing-gpu/2026-09-09T20-59-59.242Z-ribbon-parity/report.json':'08c85d106d620e2b117cc3bba0cccd1d3c58bf4011b82c63bd7134abb49641d3'
};
const PUBLIC_MANIFEST = 'experiments/constraint-editing/evidence/manifest.json';
const CPU_SOURCES = ['experiments/constraint-editing/programs.js', 'experiments/constraint-editing/fixtures.mjs',
  'experiments/constraint-editing/solver.mjs', 'experiments/constraint-editing/PROTOCOL.md', 'scripts/check-constraint-editing.mjs'];
const GPU_CORE_SOURCES = ['experiments/constraint-editing/programs.js','experiments/constraint-editing/fixtures.mjs',
  'experiments/constraint-editing/gpu.js','experiments/constraint-editing/rebind.js','experiments/constraint-editing/validate.js',
  'experiments/material-mips/autograd.js'];
const GPU_TOLERANCES = Object.freeze({primalAbsolute:2e-5,primalRelative:5e-5,adAbsolute:1e-4,adRelative:.002,fdAbsolute:.001,fdRelative:.005});
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const plain = value => JSON.parse(JSON.stringify(value));
const finiteArray = (values, length, label) => {
  assert.ok(Array.isArray(values) && values.length === length, `${label}: wrong array length`);
  assert.ok(values.every(Number.isFinite), `${label}: nonfinite scalar`);
};
function near(actual, expected, label, absolute = 1e-10, relative = 1e-10) {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${label}: nonfinite value`);
  assert.ok(Math.abs(actual - expected) <= absolute + relative * Math.abs(expected), `${label}: ${actual} != ${expected}`);
}
function compare(actual, expected, label) {
  if (typeof expected === 'number') return near(actual, expected, label);
  if (expected === null || typeof expected !== 'object') return assert.equal(actual, expected, label);
  assert.equal(Array.isArray(actual), Array.isArray(expected), `${label}: shape differs`);
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${label}: fields differ`);
  for (const key of Object.keys(expected)) compare(actual[key], expected[key], `${label}/${key}`);
}
async function exists(filename) {try {await fs.access(filename); return true;} catch {return false;}}
function inside(base, filename) {
  const relative = path.relative(base, filename);
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}
async function verifySources(recorded, report, sourceSnapshot) {
  assert.ok(recorded && typeof recorded === 'object', 'Missing source hashes');
  const gpu=report.kind.startsWith('constraint-editing-gpu-'),required=gpu?[...GPU_CORE_SOURCES,
    ...(report.kind==='constraint-editing-gpu-v2'?['experiments/constraint-editing/solver-constrained.mjs','experiments/constraint-editing/render-parity.js']:[])]:CPU_SOURCES;
  for (const name of required) assert.match(recorded[name] ?? '', /^[a-f0-9]{64}$/, `Missing required source: ${name}`);
  const currentVerified=[],snapshotVerified=[],historicalRecordedOnly=[],servedVerified=[];
  const hashes=new Map();
  async function currentHash(name) {
    if(!hashes.has(name)) {
      const filename=await fs.realpath(path.join(ROOT,name));assert.ok(inside(ROOT,filename),'Source path leaves repository');
      hashes.set(name,sha256(await fs.readFile(filename)));
    }
    return hashes.get(name);
  }
  for (const [name, expected] of Object.entries(recorded)) {
    assert.match(expected, /^[a-f0-9]{64}$/, `Invalid source hash: ${name}`);
    assert.ok(!path.isAbsolute(name) && !name.split(/[\\/]/).includes('..'), 'Unsafe source path');
    if(await currentHash(name)===expected)currentVerified.push(name);
    else {
      assert.ok(!sourceSnapshot||/^[a-zA-Z0-9_-]+$/.test(sourceSnapshot),'Unsafe source snapshot id');
      const snapshot=path.join(ROOT,path.dirname(PUBLIC_MANIFEST),'source',...(sourceSnapshot?[sourceSnapshot]:[]),name);
      if(report.sourceSnapshots?.[name]!==undefined) {
        assert.equal(sha256(Buffer.from(report.sourceSnapshots[name])),expected,`Embedded source snapshot differs: ${name}`);snapshotVerified.push(name);
      }
      else if(await exists(snapshot)&&sha256(await fs.readFile(snapshot))===expected)snapshotVerified.push(name);
      else {
        assert.ok(gpu&&!required.includes(name)&&(report.kind==='constraint-editing-gpu-v1'||name.endsWith('.md')),`Source changed without matching snapshot: ${name}`);
        historicalRecordedOnly.push(name);
      }
      // Independently computed values must use exactly the executed oracle.
      assert.ok(!required.includes(name),`Imported oracle/kernel source changed: ${name}`);
    }
  }
  for(const [name,source]of Object.entries(report.sourceSnapshots??{})) {
    assert.ok(typeof source==='string'&&source.length<2e6,'Invalid embedded source snapshot');
    assert.equal(sha256(Buffer.from(source)),recorded[name],`Embedded snapshot/hash differs: ${name}`);
  }
  if(gpu) {
    assert.ok(report.servedHashes&&Object.keys(report.servedHashes).length<=2048,'Missing or excessive served source records');
    for(const name of required)assert.equal(report.servedHashes['/'+name],recorded[name],`Required source not served: ${name}`);
    for(const [route,digest]of Object.entries(report.servedHashes)) {
      assert.match(digest,/^[a-f0-9]{64}$/);assert.ok(route.startsWith('/')&&!route.includes('\\')&&!route.split('/').includes('..'),'Unsafe served path');
      const name=route.slice(1);
      if(recorded[name])assert.equal(digest,recorded[name],`Served/source mismatch: ${name}`);
      else {
        assert.ok(name.startsWith('node_modules/three/src/'),'Unexpected unpinned served source');
        assert.equal(await currentHash(name),digest,`Three runtime source differs: ${name}`);
      }
      servedVerified.push(route);
    }
  }
  return {currentVerified,snapshotVerified,historicalRecordedOnly,servedVerifiedCount:servedVerified.length,
    interpretation:'Current/snapshot bytes and served/source hash overlap are verified. Historical recorded-only files have no recovered byte snapshot; their runtime hash is a retained observation.'};
}

async function loadRaw(filename) {
  const absolute = path.resolve(ROOT, filename), bytes = await fs.readFile(absolute);
  assert.ok(bytes.length <= 64 * 1024 * 1024, 'Report exceeds bounded audit size');
  const digest = sha256(bytes), sidecar = path.join(path.dirname(absolute), 'report.sha256');
  assert.equal((await fs.readFile(sidecar, 'utf8')).trim().split(/\s+/)[0], digest, 'Original report checksum mismatch');
  const relative=path.relative(ROOT,absolute).replaceAll('\\','/');
  if (relative === FIRST_CPU_REPORT) assert.equal(digest, FIRST_CPU_SHA256, 'First screen changed');
  if(GPU_PARITY_REPORTS[relative])assert.equal(digest,GPU_PARITY_REPORTS[relative],'Pinned GPU parity report changed');
  return {id: path.basename(path.dirname(absolute)), report: JSON.parse(bytes), originalRawSHA256: digest,
    publicJSONSHA256: digest, input: path.relative(ROOT, absolute).replaceAll('\\', '/'), redactions: []};
}

async function loadPublic() {
  const filename = path.join(ROOT, PUBLIC_MANIFEST), manifestBytes=await fs.readFile(filename);
  assert.equal((await fs.readFile(path.join(path.dirname(filename),'manifest.sha256'),'utf8')).trim().split(/\s+/)[0],sha256(manifestBytes),'Manifest checksum mismatch');
  const manifest = JSON.parse(manifestBytes);
  assert.equal(manifest.kind, 'constraint-editing-public-evidence-v1', 'Unsupported public manifest');
  assert.ok(Array.isArray(manifest.reports) && manifest.reports.length >= 1 && manifest.reports.length <= 32, 'Invalid report count');
  const reports = [];
  for (const item of manifest.reports) {
    if(item.reportKind&&!['constraint-editing-cpu-v1','constraint-editing-gpu-v1','constraint-editing-gpu-v2'].includes(item.reportKind))continue;
    assert.equal(path.basename(item.file), item.file, 'Unsafe compressed report filename');
    const gzip = await fs.readFile(path.join(path.dirname(filename), item.file));
    assert.equal(sha256(gzip), item.gzipSHA256, 'Compressed checksum mismatch');
    const bytes = gunzipSync(gzip, {maxOutputLength: 64 * 1024 * 1024});
    assert.equal(sha256(bytes), item.publicJSONSHA256, 'Public JSON checksum mismatch');
    assert.equal(gzip.length, item.gzipBytes); assert.equal(bytes.length, item.publicBytes);
    if (!item.redactions?.length) assert.equal(item.originalRawSHA256, item.publicJSONSHA256, 'Unredacted original/public hashes differ');
    reports.push({id: item.id, report: JSON.parse(bytes), originalRawSHA256: item.originalRawSHA256,
      publicJSONSHA256: item.publicJSONSHA256, input: `${path.dirname(PUBLIC_MANIFEST)}/${item.file}`, redactions: item.redactions ?? [],
      manifestIncluded: item.included, exclusionReason: item.exclusionReason ?? null});
    reports.at(-1).sourceSnapshot=item.sourceSnapshot;
  }
  return reports;
}

function numericPositions(numericProgram, fixture, params) {
  const out = [];
  for (let i = 0; i < fixture.samples.length / 4; i++) out.push(...numericProgram(fixture.kind,
    Array.from(fixture.samples.slice(4 * i, 4 * i + 3)), fixture.samples[4 * i + 3], params));
  return out;
}

async function residualGradientDiagnostic(fixture, params, modules) {
  const {makeNumericEvaluator, residuals} = modules.solver;
  const evaluate = makeNumericEvaluator(fixture.kind), system = residuals(fixture, params, await evaluate(params, fixture.samples));
  const gradient = params.map((_, j) => system.values.reduce((sum, value, i) => sum + value * system.jacobians[i * params.length + j], 0));
  const series = [];
  for (const h of [1e-3, 1e-4, 1e-5, 1e-6]) {
    const finiteDifference = [];
    for (let j = 0; j < params.length; j++) {
      const plus = [...params], minus = [...params]; plus[j] += h; minus[j] -= h;
      const a = residuals(fixture, plus, await evaluate(plus, fixture.samples, {jacobian: false}));
      const b = residuals(fixture, minus, await evaluate(minus, fixture.samples, {jacobian: false}));
      finiteDifference.push((a.cost - b.cost) / (2 * h));
    }
    series.push({h, finiteDifference, maxAbsoluteError: Math.max(...finiteDifference.map((value, j) => Math.abs(value - gradient[j])))});
  }
  return {gradient, gradientNorm: Math.hypot(...gradient), series,
    scope: 'Independent scalar residual-cost differences around retained final parameters. Coarser steps may cross contact hinges; this does not test a new optimizer.'};
}

export function auditSolve(solve, fixture, label, avoidObstacle, modules, counters) {
  const {LIMITS, validate} = modules.fixtures, {numericProgram} = modules.programs, {residuals} = modules.solver;
  finiteArray(solve.params, LIMITS.parameters, `${label}/params`);
  solve.params.forEach((value, j) => assert.ok(value >= fixture.lower[j] && value <= fixture.upper[j], `${label}: parameter outside bounds`));
  assert.equal(solve.avoidObstacle, avoidObstacle, `${label}: incorrect lane`);
  const finalPositions = numericPositions(numericProgram, fixture, solve.params);
  finiteArray(solve.finalPositions, finalPositions.length, `${label}/finalPositions`);
  compare(solve.finalPositions, finalPositions, `${label}/finalPositions`); counters.recomputedTrainingPoints += finalPositions.length / 3;
  near(solve.cost, residuals(fixture, solve.params, {positions: finalPositions}, {avoidObstacle}).cost, `${label}/cost`);
  assert.deepEqual(solve.uniformPatch, {kind: fixture.kind, parameters: solve.params, period: 1}, `${label}: export patch differs`);
  const validation = validate(fixture, solve.params);
  compare(solve.validation, validation, `${label}/validation`); counters.recomputedHeldoutValidations++;
  assert.ok(Array.isArray(solve.trace) && solve.trace.length <= LIMITS.iterations, `${label}: iteration budget`);
  assert.equal(solve.iterations, solve.trace.length, `${label}: iteration count differs`);
  assert.deepEqual(solve.trace.length ? solve.trace[0].params : solve.params, fixture.initial, `${label}: starting parameters differ`);
  const referencePositions = numericPositions(numericProgram, fixture, fixture.initial);
  let referenceVelocitySq = 0;
  for (const [a, b] of fixture.velocityRows) for (let k = 0; k < 3; k++) {
    referenceVelocitySq += ((referencePositions[3 * b + k] - referencePositions[3 * a + k]) / (2 * LIMITS.velocityStep)) ** 2;
  }
  near(solve.referenceVelocityRMS, Math.sqrt(referenceVelocitySq / fixture.velocityRows.length), `${label}/referenceVelocityRMS`);
  let evaluations = 1, accepted = 0, lastAcceptedCost = Infinity;
  for (let i = 0; i < solve.trace.length; i++) {
    const entry = solve.trace[i];
    assert.equal(entry.iteration, i, `${label}: nonsequential trace`);
    finiteArray(entry.params, LIMITS.parameters, `${label}/trace/${i}/params`);
    entry.params.forEach((value, j) => assert.ok(value >= fixture.lower[j] && value <= fixture.upper[j], `${label}: trace parameter outside bounds`));
    const positions = numericPositions(numericProgram, fixture, entry.params);
    const cost = residuals(fixture, entry.params, {positions}, {avoidObstacle}).cost;
    near(entry.cost, cost, `${label}/trace/${i}/cost`); counters.recomputedTrainingPoints += positions.length / 3;
    assert.ok(cost <= lastAcceptedCost + 1e-10, `${label}: accepted cost increased`);
    assert.ok(Array.isArray(entry.trials) && entry.trials.length <= LIMITS.trialSteps, `${label}: trial budget`);
    evaluations++;
    let acceptance = null;
    for (let j = 0; j < entry.trials.length; j++) {
      const trial = entry.trials[j]; assert.equal(trial.trial, j, `${label}: nonsequential trial`);
      if (trial.singular === true) continue;
      evaluations++;
      assert.ok(Number.isFinite(trial.cost) && Number.isFinite(trial.stepNorm), `${label}: nonfinite trial`);
      assert.ok(trial.stepNorm <= LIMITS.maxStepNorm + 1e-10, `${label}: step bound`);
      assert.equal(trial.accepted, trial.cost < entry.cost, `${label}: incorrect acceptance decision`);
      if (trial.accepted) {assert.equal(j, entry.trials.length - 1, `${label}: continued after acceptance`); acceptance = trial; accepted++;}
    }
    const nextParams = i + 1 < solve.trace.length ? solve.trace[i + 1].params : solve.params;
    if (acceptance) {
      const positions = numericPositions(numericProgram, fixture, nextParams);
      near(acceptance.cost, residuals(fixture, nextParams, {positions}, {avoidObstacle}).cost, `${label}: accepted candidate cost`);
      near(acceptance.stepNorm, Math.hypot(...nextParams.map((x, k) => x - entry.params[k])), `${label}: accepted step norm`);
      lastAcceptedCost = acceptance.cost; counters.recomputedTrainingPoints += positions.length / 3;
    } else assert.deepEqual(nextParams, entry.params, `${label}: changed without acceptance`);
  }
  assert.equal(solve.accepted, accepted, `${label}: accepted-count mismatch`);
  assert.equal(solve.evaluations, evaluations, `${label}: evaluator-call count mismatch`);
  assert.ok(Number.isFinite(solve.elapsedMs) && solve.elapsedMs >= 0, `${label}: invalid recorded elapsed time`);
  return {params: solve.params, status: solve.status, cost: solve.cost, accepted, evaluations,
    recordedElapsedMs: solve.elapsedMs, validation};
}

export async function auditCPUReport(report, modules, {diagnostics = true} = {}) {
  const {KINDS, LIMITS, makeFixture, validate} = modules.fixtures, {numericProgram} = modules.programs;
  assert.equal(report.kind, 'constraint-editing-cpu-v1', 'Unsupported report schema; requires a versioned verifier');
  assert.equal(report.sourceUnchanged, true, 'Report observed source changes');
  assert.deepEqual(report.limits, LIMITS, 'Limits changed');
  assert.equal(report.cells.length, KINDS.length, 'Incomplete fixture cohort');
  assert.deepEqual(report.cells.map(c => c.kind), [...KINDS], 'Fixture order/membership changed');
  const counters = {recomputedTrainingPoints: 0, recomputedHeldoutValidations: 0}, cells = [];
  for (const cell of report.cells) {
    const fixture = makeFixture(cell.kind), label = cell.kind;
    compare(cell.fixture, {...fixture, samples: Array.from(fixture.samples)}, `${label}/fixture`);
    const witness = validate(fixture, fixture.witness), initial = validate(fixture, fixture.initial);
    compare(cell.witness, witness, `${label}/witness`); compare(cell.initial, initial, `${label}/initial`);
    counters.recomputedHeldoutValidations += 2;
    if (!witness.pass) {
      assert.equal(cell.status, 'invalid-witness'); cells.push({kind: cell.kind, status: cell.status, witness, initial}); continue;
    }
    const tip = numericProgram(cell.kind, fixture.target.p, fixture.target.time, fixture.initial);
    const translation = fixture.target.position.map((value, k) => value - tip[k]);
    compare(cell.translation.translation, translation, `${label}/translation`);
    const translationValidation = validate(fixture, fixture.initial, {translation});
    compare(cell.translation.validation, translationValidation, `${label}/translation/validation`); counters.recomputedHeldoutValidations++;
    const handleOnly = auditSolve(cell.handleOnly, fixture, `${label}/handleOnly`, false, modules, counters);
    const constrained = auditSolve(cell.constrained, fixture, `${label}/constrained`, true, modules, counters);
    const gates = {quality: constrained.validation.pass, controlNecessity: handleOnly.validation.minClearance <= -LIMITS.controlPenetration,
      bounded: !['wall-limit', 'stopped'].includes(constrained.status) && !['wall-limit', 'stopped'].includes(handleOnly.status)};
    assert.deepEqual(cell.gates, gates, `${label}: stored gates differ`);
    const status = Object.values(gates).every(Boolean) ? 'passed' : 'failed'; assert.equal(cell.status, status);
    cells.push({kind: cell.kind, status, gates, witness, initial, translation: {translation, validation: translationValidation},
      handleOnly, constrained, residualGradientDiagnostic: diagnostics ? await residualGradientDiagnostic(fixture, constrained.params, modules) : null});
  }
  const researchGatePassed = cells.every(cell => cell.status === 'passed');
  assert.equal(report.status, researchGatePassed ? 'passed' : 'failed', 'Stored overall status differs from recomputed gates');
  return {experimentStatus: researchGatePassed ? 'passed' : 'failed', researchGatePassed, cells, counters,
    limitations: ['CPU-only evidence; no GPU AD/FD parity, rendered-geometry validation or GPU speedup is implied.',
      'Held-out collision checks cover sampled times, not continuous time.',
      'Rejected trial parameter vectors were not retained; their recorded costs cannot be independently recomputed.',
      'Elapsed timings and run-time sourceUnchanged flags are recorded observations; source hashes are also checked against the current files.']};
}

function expectedProbeSamples() {
  const records=[];for(let i=0;i<67;i++)records.push((i+.31)/67,.02*Math.cos(i),.02*Math.sin(i),(i*.61803398875)%1);
  return Array.from(new Float32Array(records));
}

/** Rebuild numeric references from the pinned independent forward program;
 * never use the report's cpu arrays as the source of truth. */
export function auditGPUValidation(validation,kind,expectedParameters,modules) {
  const {numericProgram}=modules.programs,n=8;
  finiteArray(validation.parameters,n,'GPU parameters');
  assert.deepEqual(validation.parameters,expectedParameters.map(Math.fround),'Unexpected GPU validation parameters');
  assert.deepEqual(validation.startingParameters,validation.parameters,'GPU upload was not exact Float32');
  assert.deepEqual(validation.samples,expectedProbeSamples(),'Prospective probe sample set changed');
  assert.deepEqual(validation.tolerances,GPU_TOLERANCES,'Prospective GPU tolerance changed');
  const count=validation.samples.length/4,length=count*3;
  for(const [name,values]of Object.entries(validation.positions))finiteArray(values,length,'GPU positions/'+name);
  assert.deepEqual(Object.keys(validation.positions).sort(),['cpu','fdGPU','gpu']);
  for(const [name,values]of Object.entries(validation.jacobians))finiteArray(values,length*n,'GPU Jacobians/'+name);
  assert.deepEqual(Object.keys(validation.jacobians).sort(),['ad','cpu','fd']);
  const expectedPositions=[],expectedJacobian=[],failures=[];let maxPrimal=0,maxAD=0,maxFD=0;
  for(let sample=0;sample<count;sample++) {
    const p=validation.samples.slice(4*sample,4*sample+3),time=validation.samples[4*sample+3],weights=validation.parameters;
    const primal=numericProgram(kind,p,time,weights),partials=Array.from({length:n},(_,j)=>{
      const plus=[...weights],minus=[...weights];plus[j]+=1e-5;minus[j]-=1e-5;
      const a=numericProgram(kind,p,time,plus),b=numericProgram(kind,p,time,minus);
      return a.map((value,k)=>(value-b[k])/2e-5);
    });
    expectedPositions.push(...primal);
    for(let axis=0;axis<3;axis++) {
      const index=3*sample+axis;
      for(const [method,positions]of [['ad',validation.positions.gpu],['fd',validation.positions.fdGPU]]) {
        const actual=positions[index],error=Math.abs(primal[axis]-actual);maxPrimal=Math.max(maxPrimal,error);
        if(error>GPU_TOLERANCES.primalAbsolute+GPU_TOLERANCES.primalRelative*Math.abs(primal[axis]))failures.push({kind:method+'-primal',sample,axis,expected:primal[axis],actual});
      }
      for(let j=0;j<n;j++) {
        const expected=partials[j][axis];expectedJacobian.push(expected);
        for(const method of ['ad','fd']) {
          const actual=validation.jacobians[method][index*n+j],delta=Math.abs(actual-expected);
          if(method==='ad')maxAD=Math.max(maxAD,delta);else maxFD=Math.max(maxFD,delta);
          if(delta>GPU_TOLERANCES[method+'Absolute']+GPU_TOLERANCES[method+'Relative']*Math.abs(expected))failures.push({kind:method,sample,axis,parameter:j,expected,actual});
        }
      }
    }
  }
  compare(validation.positions.cpu,expectedPositions,'Retained numeric primal reference');
  compare(validation.jacobians.cpu,expectedJacobian,'Retained numeric Jacobian reference');
  compare(validation.failures,failures,'GPU failure list');
  near(validation.maxPrimal,maxPrimal,'GPU maximum primal error');near(validation.maxAD,maxAD,'GPU maximum AD error');near(validation.maxFD,maxFD,'GPU maximum FD error');
  assert.equal(validation.passed,failures.length===0,'Stored GPU parity flag');
  return {passed:failures.length===0,parameters:validation.parameters,points:count,scalarPrimals:2*length,scalarJacobianComparisons:2*length*n,
    numericDerivativeStep:1e-5,tolerances:GPU_TOLERANCES,maxPrimal,maxAD,maxFD,uploadExact:true,failures};
}

export function auditRenderParity(render) {
  assert.equal(render.kind,'constraint-editing-render-parity-v1');
  const expected={width:256,height:192,sections:192,sides:10,times:[0,.25,.59375],minimumPixelAllowance:2,relativePixelAllowance:.01,boundaryRadius:1};
  assert.deepEqual(render.limits,expected,'Prospective render tolerances changed');
  assert.equal(render.maskEncoding,'base64-packed-lsb-first-row-major');
  assert.equal(render.restVertices,193*11);assert.equal(render.triangles,192*10*2);
  compare(render.camera,{position:[5.4,3.7,7],lookAt:[1.4,.7,0],orthographic:[-2.6,2.6,1.95,-1.95,.1,40]},'Render camera');
  assert.deepEqual(render.cases.map(c=>c.time),expected.times);
  finiteArray(render.parameters,8,'Render parameters');assert.ok(render.parameters.every(x=>x===Math.fround(x)),'Render parameters were not Float32');
  assert.deepEqual(render.uploadedParameters,render.parameters,'Renderer parameter upload differs');assert.equal(render.uploadExact,true);
  const width=256,height=192,count=width*height,decode=text=>{
    assert.match(text,/^[A-Za-z0-9+/]*={0,2}$/);const bytes=Buffer.from(text,'base64');
    assert.equal(bytes.length,count/8);assert.equal(bytes.toString('base64'),text,'Noncanonical mask encoding');
    return Uint8Array.from({length:count},(_,i)=>(bytes[i>>3]>>(i&7))&1);
  };
  const cases=[];
  for(const item of render.cases) {
    const native=decode(item.masks.native),cpu=decode(item.masks.cpu);
    const at=(mask,x,y)=>x<0||x>=width||y<0||y>=height?0:mask[y*width+x];
    const boundaryNear=(mask,x,y)=>{
      for(let ey=Math.max(0,y-1);ey<=Math.min(height-1,y+1);ey++)for(let ex=Math.max(0,x-1);ex<=Math.min(width-1,x+1);ex++)
        for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)if(at(mask,ex,ey)!==at(mask,ex+dx,ey+dy))return true;
      return false;
    };
    let nativeForeground=0,cpuForeground=0,differingPixels=0,interiorMismatches=0;
    for(let i=0;i<count;i++) {
      nativeForeground+=native[i];cpuForeground+=cpu[i];
      if(native[i]!==cpu[i]) {
        differingPixels++;const x=i%width,y=Math.floor(i/width);
        if(!boundaryNear(native,x,y)||!boundaryNear(cpu,x,y))interiorMismatches++;
      }
    }
    const pixelAllowance=Math.max(2,Math.ceil(.01*Math.max(nativeForeground,cpuForeground)));
    const comparison={nativeForeground,cpuForeground,differingPixels,interiorMismatches,pixelAllowance,
      passed:nativeForeground>0&&cpuForeground>0&&differingPixels<=pixelAllowance&&interiorMismatches===0};
    assert.deepEqual(item.comparison,comparison,'Retained render-mask metrics');
    for(const [key,foreground]of [['native',nativeForeground],['cpu',cpuForeground]]) {
      const bytes=item.outputBytes[key];assert.equal(bytes.length,count*4);assert.equal(bytes.white,foreground);
      assert.ok(Number.isInteger(bytes.invalid)&&bytes.invalid>=0&&bytes.invalid<=count);
      assert.equal(bytes.black+bytes.white+bytes.invalid,count);assert.equal(bytes.valid,bytes.invalid===0);
    }
    const passed=comparison.passed&&item.outputBytes.native.valid&&item.outputBytes.cpu.valid;
    assert.equal(item.passed,passed);cases.push({time:item.time,...comparison,passed});
  }
  const passed=cases.every(c=>c.passed);assert.equal(render.passed,passed);
  return {passed,parameters:render.parameters,cases,limitations:['Masks validate one off-axis silhouette, not all hidden vertices or lighting/normal correctness.',
    'RGBA validity histograms are retained observations; full RGBA bytes were not retained. Binary mask comparisons are independently recomputed.']};
}

function auditGPUFit(fit,fixture,modules) {
  const {constrainedSystem,CONSTRAINED_LIMITS}=modules.solverConstrained;
  finiteArray(fit.params,8,'Fitted parameters');fit.params.forEach((v,j)=>assert.ok(v>=fixture.lower[j]&&v<=fixture.upper[j],'Fit outside bounds'));
  assert.equal(fit.stage,2,'Missing constrained stage identifier');
  assert.deepEqual(fit.firstUpload,{requested:fixture.initial,actual:fixture.initial.map(Math.fround)},'Solver lane reset differs');
  const weights=fit.params.map(Math.fround),positions=numericPositions(modules.programs.numericProgram,fixture,weights);
  finiteArray(fit.finalPositions,positions.length,'GPU fit final positions');
  let maximumPositionError=0;
  for(let i=0;i<positions.length;i++) {
    maximumPositionError=Math.max(maximumPositionError,Math.abs(fit.finalPositions[i]-positions[i]));
    near(fit.finalPositions[i],positions[i],'GPU final position',GPU_TOLERANCES.primalAbsolute,GPU_TOLERANCES.primalRelative);
  }
  const system=constrainedSystem(fixture,fit.params,{positions:fit.finalPositions});
  compare(fit.diagnostics,system.diagnostics,'GPU final constraint diagnostics');
  const objectiveCost=system.objective.reduce((sum,row)=>sum+.5*row.value*row.value,0);
  near(fit.objectiveCost,objectiveCost,'GPU final objective cost');
  const validation=modules.fixtures.validate(fixture,fit.params),float32Validation=modules.fixtures.validate(fixture,weights);
  compare(fit.validation,validation,'GPU fit heldout validation');compare(fit.float32Validation,float32Validation,'GPU uploaded fit heldout validation');
  const expectedPatch={kind:fixture.kind,variant:fixture.variant,parameters:fit.params,period:1};
  assert.deepEqual(fit.uniformPatch,expectedPatch,'Exported patch differs');assert.deepEqual(fit.reload.exported,expectedPatch,'Reloaded patch differs');
  const length=67*3;
  for(const name of ['before','after']) {
    finiteArray(fit.reload[name].positions,length,'Reload positions');finiteArray(fit.reload[name].jacobians,length*8,'Reload Jacobians');
  }
  assert.deepEqual(fit.reload.before.positions,fit.reload.after.positions,'Reload primal differs');
  assert.deepEqual(fit.reload.before.jacobians,fit.reload.after.jacobians,'Reload Jacobian differs');assert.equal(fit.reload.exact,true);
  assert.ok(Array.isArray(fit.trace)&&fit.trace.length<=CONSTRAINED_LIMITS.outer,'Outer iteration budget exceeded');
  let evaluations=1,accepted=0;
  for(const [outer,round]of fit.trace.entries()) {
    assert.equal(round.outer,outer);assert.ok(round.iterations.length<=CONSTRAINED_LIMITS.inner,'Inner iteration budget exceeded');
    for(const [inner,entry]of round.iterations.entries()) {
      assert.equal(entry.inner,inner);evaluations++;finiteArray(entry.params,8,'Solver trace parameters');
      if(outer===0&&inner===0)assert.deepEqual(entry.params,fixture.initial,'Initial iterate differs');
      assert.ok(entry.trials.length<=CONSTRAINED_LIMITS.backtracks,'Backtrack budget exceeded');
      for(const [trialIndex,trial]of entry.trials.entries()) {
        assert.equal(trial.trial,trialIndex);evaluations++;
        assert.ok(Number.isFinite(trial.cost)&&Number.isFinite(trial.directional)&&Number.isFinite(trial.stepNorm));
        assert.ok(trial.stepNorm<=CONSTRAINED_LIMITS.maxStepNorm+1e-10,'Step norm exceeded');
        const accept=trial.directional<0&&trial.cost<=entry.cost+CONSTRAINED_LIMITS.armijo*trial.directional;
        assert.equal(trial.accepted,accept,'Armijo decision differs');
        if(accept){accepted++;assert.equal(trialIndex,entry.trials.length-1,'Continued after acceptance');}
      }
    }
    if(round.final)evaluations++;
  }
  assert.equal(fit.evaluations,evaluations,'GPU evaluator-call count differs');assert.equal(fit.accepted,accepted,'GPU accepted-count differs');
  const utilityPassed=fit.status!=='stopped'&&validation.pass&&float32Validation.pass&&system.diagnostics.maximumViolation<=.001;
  assert.equal(fit.utilityPassed,utilityPassed,'Stored GPU utility gate differs');
  return {method:fit.method,utilityPassed,status:fit.status,params:fit.params,maximumPositionError,objectiveCost,
    diagnostics:system.diagnostics,validation,float32Validation,evaluations,accepted,recordedElapsedMs:fit.elapsedMs,
    limitations:['Final positions, final residual diagnostics, heldout metrics, budgets, and Armijo decisions are checked. GPU intermediate Jacobians and rejected candidate vectors were not retained, so the complete optimizer trajectory cannot be reconstructed independently.']};
}

export async function auditGPUReport(report,modules) {
  assert.ok(['constraint-editing-gpu-v1','constraint-editing-gpu-v2'].includes(report.kind),'Unsupported GPU report');
  assert.equal(report.ui,false,'UI screenshots require their own verifier');
  assert.deepEqual(report.errors,[],'Browser error observed');assert.ok(['ribbon','tentacle'].includes(report.field));
  const probe=report.probe;assert.equal(probe.kind,report.field);assert.deepEqual(probe.errors,[],'GPU error observed');
  assert.deepEqual(probe.finalState,{busy:false,disposed:true},'Evaluator cleanup not observed');
  assert.equal(probe.forwardFactoryCalls,1,'Forward formula reconstructed');
  assert.ok(Array.isArray(probe.compiler)&&probe.compiler.length===3,'Missing three output derivative compilers');
  assert.ok(Array.isArray(probe.shaders)&&probe.shaders.length>=3&&probe.shaders.length<=32,'Missing or excessive shaders');
  assert.ok(probe.shaders.every(s=>typeof s==='string'&&s.length>20&&s.length<2e6),'Invalid retained shader');
  const fixture=report.kind==='constraint-editing-gpu-v2'?modules.solverConstrained.fixtureVariant(report.field,probe.variant):modules.fixtures.makeFixture(report.field);
  if(report.kind==='constraint-editing-gpu-v2'){assert.equal(probe.stage,2);assert.equal(report.variant,probe.variant);}
  const fits=[];
  if(report.fit) {
    assert.equal(report.kind,'constraint-editing-gpu-v2','GPU v1 optimizer audit is not part of this cohort');
    const methods=report.field==='ribbon'?['ad','fd']:['fd','ad'];if(probe.variant==='late-target')methods.reverse();
    assert.equal(probe.fits.length,2);assert.deepEqual(probe.fits.map(f=>f.method),methods);
    for(const fit of probe.fits) {
      fits.push(auditGPUFit(fit,fixture,modules));
    }
  } else assert.deepEqual(probe.fits,[],'Unexpected optimizer result');
  assert.equal(probe.validations.length,1+probe.fits.length,'Missing initial/final parity snapshots');
  const validations=probe.validations.map((v,i)=>auditGPUValidation(v,report.field,i===0?fixture.initial:probe.fits[i-1].params,modules));
  for(const [i,fit]of probe.fits.entries()) {
    assert.deepEqual(fit.reload.before.positions,probe.validations[i+1].positions.gpu,'Reload primal differs from independently validated final snapshot');
    assert.deepEqual(fit.reload.before.jacobians,probe.validations[i+1].jacobians.ad,'Reload derivative differs from independently validated final snapshot');
  }
  const renderRecords=report.kind==='constraint-editing-gpu-v2'?[probe.initialRender,...probe.fits.map(f=>f.renderParity)]:[];
  const renders=renderRecords.map((record,i)=>{
    assert.equal(record.fixture,report.field);assert.deepEqual(record.parameters,(i===0?fixture.initial:probe.fits[i-1].params).map(Math.fround));
    return auditRenderParity(record);
  });
  const parityPassed=validations.every(v=>v.passed)&&renders.every(r=>r.passed);
  const researchGatePassed=report.fit?fits.every(f=>f.utilityPassed):null;
  assert.equal(probe.status,parityPassed&&researchGatePassed!==false?'passed':'failed','Stored probe status differs');
  assert.equal(report.status,probe.status,'Stored overall GPU status differs');
  return {experimentStatus:report.status,parityPassed,researchGatePassed,validations,renders,fits,
    counters:{recomputedGPUProbePoints:validations.reduce((sum,v)=>sum+v.points,0),
      scalarPrimalComparisons:validations.reduce((sum,v)=>sum+v.scalarPrimals,0),scalarJacobianComparisons:validations.reduce((sum,v)=>sum+v.scalarJacobianComparisons,0),renderMaskPairs:renders.reduce((sum,r)=>sum+r.cases.length,0)},
    limitations:['Compute parity is distinct from an editing quality gate and is not a speedup claim.',
      'Independent numeric derivatives are central differences of the pinned analytic forward program at h=1e-5.',
      'A cleanup state and absent error arrays are retained runtime observations, not post-hoc device inspection.',
      ...(renders.length?[]:['This report predates actual renderer silhouette validation.'])]};
}

async function main() {
  const args = process.argv.slice(2), reports = args.filter(a => a.startsWith('--report=')).map(a => a.slice(9));
  const selfTest = args.includes('--self-test'), requireCapability = args.includes('--require-capability');
  const outputArg = args.find(a => a.startsWith('--output=')),kindArg=args.find(a=>a.startsWith('--kind='))?.slice(7);
  for (const arg of args) assert.ok(arg.startsWith('--report=') || arg.startsWith('--output=') || arg.startsWith('--kind=') || ['--self-test', '--require-capability', '--help'].includes(arg), `Unknown argument ${arg}`);
  if(kindArg)assert.ok(['constraint-editing-cpu-v1','constraint-editing-gpu-v1','constraint-editing-gpu-v2'].includes(kindArg),'Unsupported kind filter');
  if (args.includes('--help')) {console.log('CPU-only evidence audit: node scripts/analyze-constraint-editing.mjs [--report=report.json] [--kind=constraint-editing-gpu-v2] [--self-test] [--require-capability]'); return;}
  const loaded = reports.length ? await Promise.all(reports.map(loadRaw)) : await exists(path.join(ROOT, PUBLIC_MANIFEST)) ? await loadPublic() : await Promise.all([FIRST_CPU_REPORT,...Object.keys(GPU_PARITY_REPORTS)].map(loadRaw));
  const inputs=kindArg?loaded.filter(input=>input.report.kind===kindArg):loaded;
  assert.ok(inputs.length>0,'No supported evidence records');
  for (const input of inputs) {
    input.sourceVerification=await verifySources(input.report.sourceHashes,input.report,input.sourceSnapshot);
    input.verifiedSourceCount=input.sourceVerification.currentVerified.length+input.sourceVerification.snapshotVerified.length;
  }
  const modules = {fixtures: await import('../experiments/constraint-editing/fixtures.mjs'),
    programs: await import('../experiments/constraint-editing/programs.js'), solver: await import('../experiments/constraint-editing/solver.mjs'),
    solverConstrained:await import('../experiments/constraint-editing/solver-constrained.mjs')};
  const results = [];
  for (const input of inputs) {
    const audit = input.report.kind==='constraint-editing-cpu-v1'?await auditCPUReport(input.report, modules):await auditGPUReport(input.report,modules);
    const {report, ...provenance} = input; results.push({...provenance,reportKind:report.kind, ...audit});
  }
  let selfTests = null;
  if (selfTest) {
    selfTests={};const original = inputs.find(input=>input.report.kind==='constraint-editing-cpu-v1')?.report;
    if(original){
    const mutations = [r => r.cells[0].constrained.validation.velocityNRMSE = 0,
      r => r.cells[0].constrained.finalPositions[5] += .1, r => r.cells[0].constrained.evaluations++,
      r => r.cells[0].status = 'passed'];
    for (const mutate of mutations) {const tampered = plain(original); mutate(tampered); await assert.rejects(() => auditCPUReport(tampered, modules, {diagnostics: false}));}
    Object.assign(selfTests,{tamperedMetricRejected: true, tamperedPositionRejected: true, tamperedBudgetRejected: true, tamperedStatusRejected: true});
    }
    const gpu=inputs.find(input=>input.report.kind.startsWith('constraint-editing-gpu-'))?.report;
    if(gpu) {
      const mutations=[r=>r.probe.validations[0].positions.cpu[2]+=.01,
        r=>r.probe.validations[0].jacobians.ad[7]+=.1,r=>r.probe.validations[0].startingParameters[0]+=.1,
        r=>r.probe.validations[0].tolerances.adAbsolute=1];
      for(const mutate of mutations){const tampered=plain(gpu);mutate(tampered);await assert.rejects(()=>auditGPUReport(tampered,modules));}
      Object.assign(selfTests,{tamperedGPUOracleRejected:true,tamperedGPUJacobianRejected:true,tamperedUploadRejected:true,tamperedToleranceRejected:true});
      if(gpu.kind==='constraint-editing-gpu-v2') {
        const tampered=plain(gpu);tampered.probe.initialRender.cases[0].comparison.differingPixels++;
        await assert.rejects(()=>auditGPUReport(tampered,modules));selfTests.tamperedRenderMetricRejected=true;
        if(gpu.fit) {
          const mutations=[r=>r.probe.fits[0].validation.velocityNRMSE=0,r=>r.probe.fits[0].finalPositions[12]+=.1,
            r=>r.probe.fits[0].diagnostics.maximumViolation=0,r=>r.probe.fits[0].evaluations++];
          for(const mutate of mutations){const edited=plain(gpu);mutate(edited);await assert.rejects(()=>auditGPUReport(edited,modules));}
          Object.assign(selfTests,{tamperedGPUFitMetricRejected:true,tamperedGPUFitPositionRejected:true,tamperedGPUConstraintMetricRejected:true,tamperedGPUFitBudgetRejected:true});
        }
      }
    }
  }
  const utilityResults=results.filter(result=>result.researchGatePassed!==null);
  const summary = {kind: 'constraint-editing-analysis-v1', analyzedAt: new Date().toISOString(), auditStatus: 'passed',
    researchGatePassed:utilityResults.length?utilityResults.every(result=>result.researchGatePassed):null,
    parityPassed:results.some(result=>result.parityPassed!==undefined)?results.filter(result=>result.parityPassed!==undefined).every(result=>result.parityPassed):null,
    reports: results, selfTests};
  const filename = outputArg ? path.resolve(ROOT, outputArg.slice(9)) : path.join(ROOT, 'results/development/constraint-editing-analysis', summary.analyzedAt.replaceAll(':', '-') + '.json');
  await fs.mkdir(path.dirname(filename), {recursive: true}); await fs.writeFile(filename, JSON.stringify(summary, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify({auditStatus: summary.auditStatus, researchGatePassed: summary.researchGatePassed,
    parityPassed:summary.parityPassed,reports: results.map(r => ({id: r.id, experimentStatus: r.experimentStatus, sourceHashesVerified: r.verifiedSourceCount,
      historicalSourceHashesOnly:r.sourceVerification.historicalRecordedOnly,counters: r.counters,
      fits:r.fits?.map(f=>({method:f.method,utilityPassed:f.utilityPassed,maximumPositionError:f.maximumPositionError,targetError:f.validation.targetError,
        minClearance:f.validation.minClearance,velocityNRMSE:f.validation.velocityNRMSE})),
      cells: r.cells?.map(c => ({kind: c.kind, status: c.status, targetError: c.constrained?.validation.targetError,
        minClearance: c.constrained?.validation.minClearance, velocityNRMSE: c.constrained?.validation.velocityNRMSE}))})),
    selfTests, output: path.relative(ROOT, filename).replaceAll('\\', '/')}));
  if (requireCapability && !summary.researchGatePassed) process.exitCode = 1;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => {
  console.error(JSON.stringify({auditStatus: 'failed', error: String(error?.message ?? error)})); process.exitCode = 1;
});
