/** CPU-only audit of the complete, compressed four-cell native TSL cohort. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const evidence = path.join(root, 'experiments/trainable-tsl/evidence');
const selected = ['2026-09-09T20-10-03.419Z-neural-11', '2026-09-09T20-10-11.217Z-islands-11',
  '2026-09-09T20-10-18.800Z-neural-29', '2026-09-09T20-10-27.198Z-islands-29'];
const excluded = '2026-09-09T20-08-22.166Z-neural-11';
const limits = Object.freeze({samples: 128, steps: 160, cpuFDStep: 1e-5, gpuFDStep: .001,
  adAbsolute: 2e-4, adRelative: 2e-3, fdAbsolute: 3e-3, fdRelative: 5e-3,
  primalAbsolute: 2e-5, firstAdamWeightAbsolute: 5e-4, firstAdamMomentAbsolute: 2e-6,
  maximumHeldoutRMSEFraction: .2});
const hash = value => createHash('sha256').update(value).digest('hex');
const near = (actual, expected, label, tolerance = 1e-10) => assert.ok(Number.isFinite(actual) && Number.isFinite(expected)
  && Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
const finiteArray = (array, length, label) => assert.ok(Array.isArray(array) && array.length === length && array.every(Number.isFinite), label);
const sameMap = (actual, expected, label) => assert.deepEqual(Object.entries(actual).sort(), Object.entries(expected).sort(), label);
const median = values => {const s = [...values].sort((a, b) => a - b); return (s[Math.floor((s.length - 1) / 2)] + s[Math.floor(s.length / 2)]) / 2;};
const fieldCount = kind => kind === 'neural' ? 49 : 28;

// Separate numeric forward model; this analyzer never imports or traverses AD.
// These are the frozen fields.js formulas, evaluated with JavaScript doubles.
function numericField(kind, x, z, w) {
  if (kind === 'islands') {
    let y = w[27];
    for (let i = 0; i < 9; i++) y += w[3*i] * Math.exp(-5*((x-w[3*i+1])**2+(z-w[3*i+2])**2));
    return y;
  }
  let y = w[48];
  for (let i = 0; i < 12; i++) y += Math.sin(x*w[4*i]+z*w[4*i+1]+w[4*i+2])*w[4*i+3];
  return y;
}
function target(kind, x, z) {
  if (kind === 'islands') return .9*Math.exp(-6*((x+.42)**2+(z+.22)**2))
    + .72*Math.exp(-8*((x-.48)**2+(z-.28)**2)) - .4*Math.exp(-7*((x-.2)**2+(z+.6)**2));
  return .48*Math.sin(2.6*x+.5)*Math.cos(2.1*z-.35)+.2*Math.cos(3.5*z+x);
}
function rng(seed) {let state = seed >>> 0; return () => {state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296;};}
function initialWeights(kind, seed) {
  const random = rng(seed), w = new Float32Array(fieldCount(kind));
  if (kind === 'islands') for (let i = 0; i < 9; i++) {
    w[3*i] = .035*(random()-.5); w[3*i+1] = (i%3-1)*.65; w[3*i+2] = (Math.floor(i/3)-1)*.65;
  } else for (let i = 0; i < 12; i++) {
    w[4*i] = (random()-.5)*5; w[4*i+1] = (random()-.5)*5;
    w[4*i+2] = (random()-.5)*2; w[4*i+3] = (random()-.5)*.08;
  }
  return Array.from(w);
}
function samplesFor(kind, seed) {
  const random = rng(seed ^ 0x971), observations = new Float32Array(limits.samples*4);
  for (let i = 0; i < limits.samples; i++) {
    const x = Math.fround(2*random()-1), z = Math.fround(2*random()-1);
    observations.set([x, z, target(kind, x, z), 1], i*4);
  }
  return Array.from(observations);
}
function metrics(kind, weights, samples) {
  let train = 0, held = 0, maximum = 0;
  for (let i = 0; i < samples.length; i += 4) train += (numericField(kind, samples[i], samples[i+1], weights)-samples[i+2])**2;
  for (let i = 0; i < 31; i++) for (let j = 0; j < 31; j++) {
    const x = -1+2*(i+.37)/31, z = -1+2*(j+.61)/31;
    const error = Math.abs(numericField(kind, x, z, weights)-target(kind, x, z));
    held += error*error; maximum = Math.max(maximum, error);
  }
  return {trainingRMSE: Math.sqrt(train/limits.samples), heldout: {count: 961, rmse: Math.sqrt(held/961), maxError: maximum}};
}
function verifyMetrics(actual, expected) {
  near(actual.trainingRMSE, expected.trainingRMSE, 'training RMSE');
  assert.equal(actual.heldout.count, 961);
  for (const key of ['rmse', 'maxError']) near(actual.heldout[key], expected.heldout[key], 'held-out '+key);
}
function gradientsFor(kind, weights, samples) {
  const result = [], w = [...weights], h = limits.cpuFDStep;
  for (let i = 0; i < limits.samples; i++) for (let j = 0; j < w.length; j++) {
    const old = w[j], x = samples[4*i], z = samples[4*i+1], y = samples[4*i+2];
    w[j] = old+h; const plus = numericField(kind, x, z, w)-y;
    w[j] = old-h; const minus = numericField(kind, x, z, w)-y;
    w[j] = old; result.push((plus*plus-minus*minus)/(2*h));
  }
  return result;
}
function auditSnapshot(kind, snapshot, expectedWeights, samples, label) {
  const count = fieldCount(kind), length = limits.samples*count;
  assert.deepEqual(snapshot.weights, expectedWeights, label+' point');
  assert.equal(snapshot.count, length);
  for (const key of ['adGradients', 'fdGradients', 'cpuGradients']) finiteArray(snapshot[key], length, label+' '+key);
  const cpu = gradientsFor(kind, snapshot.weights, samples);
  let maxADAbsolute = 0, maxFDAbsolute = 0, cpuOracleMaximumDelta = 0, adFailures = 0, fdFailures = 0;
  for (let i = 0; i < length; i++) {
    const reference = cpu[i], da = Math.abs(snapshot.adGradients[i]-reference), df = Math.abs(snapshot.fdGradients[i]-reference);
    near(snapshot.cpuGradients[i], reference, label+' retained CPU oracle');
    cpuOracleMaximumDelta = Math.max(cpuOracleMaximumDelta, Math.abs(snapshot.cpuGradients[i]-reference));
    maxADAbsolute = Math.max(maxADAbsolute, da); maxFDAbsolute = Math.max(maxFDAbsolute, df);
    if (da > limits.adAbsolute+limits.adRelative*Math.abs(reference)) adFailures++;
    if (df > limits.fdAbsolute+limits.fdRelative*Math.abs(reference)) fdFailures++;
  }
  near(snapshot.maxADAbsolute, maxADAbsolute, label+' reported AD error');
  near(snapshot.maxFDAbsolute, maxFDAbsolute, label+' reported FD error');
  assert.equal(snapshot.adFailures, adFailures); assert.equal(snapshot.fdFailures, fdFailures);
  assert.equal(adFailures, 0); assert.equal(fdFailures, 0); assert.equal(snapshot.passed, true);
  assert.ok(Number.isFinite(snapshot.primalMaxError) && snapshot.primalMaxError >= 0 && snapshot.primalMaxError < limits.primalAbsolute, label+' reported primal gate');
  let biasGradientInferredPrimalMaxError = 0;
  for (let i = 0; i < limits.samples; i++) {
    // Both forward programs have an additive final bias: d(loss)/d(bias)=2r.
    // This reconstructs predictions from retained AD gradients, not from the
    // missing direct prediction-buffer readback, and is labeled accordingly.
    const reconstructed = samples[4*i+2]+snapshot.adGradients[i*count+count-1]/2;
    biasGradientInferredPrimalMaxError = Math.max(biasGradientInferredPrimalMaxError,
      Math.abs(reconstructed-numericField(kind, samples[4*i], samples[4*i+1], snapshot.weights)));
  }
  assert.ok(biasGradientInferredPrimalMaxError < limits.primalAbsolute, label+' inferred primal gate');
  return {label, count: length, maxADAbsolute, maxFDAbsolute, cpuOracleMaximumDelta, adFailures, fdFailures,
    reportedDirectPrimalMaxError: snapshot.primalMaxError, biasGradientInferredPrimalMaxError,
    directPrimalReadbackRetained: false, cpu};
}
function auditAdam(kind, initial, snapshot, independentCPU) {
  const count = fieldCount(kind), adam = snapshot.adam;
  assert.equal(adam.passed, true); assert.deepEqual(adam.failures, []);
  finiteArray(adam.gpuState, count*4, 'Adam GPU vec4 state');
  finiteArray(adam.expected, count*4, 'recorded Adam expected state');
  finiteArray(adam.meanGradients, count, 'recorded mean gradients');
  const maxima = {fromRetainedGPUGradient: [0, 0, 0, 0], fromIndependentCPUGradient: [0, 0, 0, 0]};
  for (let j = 0; j < count; j++) {
    const means = [snapshot.adGradients, independentCPU].map(values => {
      let sum = 0; for (let i = 0; i < limits.samples; i++) sum += values[i*count+j]; return sum/limits.samples;
    });
    near(adam.meanGradients[j], means[0], 'mean reduction');
    for (let source = 0; source < 2; source++) {
      const g = means[source], expected = [Math.max(-8, Math.min(8, initial[j]-.025*g/(Math.abs(g)+1e-8))), .1*g, .001*g*g, 0];
      for (let k = 0; k < 4; k++) {
        if (source === 0) near(adam.expected[4*j+k], expected[k], 'recorded Adam expectation');
        const delta = Math.abs(adam.gpuState[4*j+k]-expected[k]);
        assert.ok(delta <= (k === 0 ? limits.firstAdamWeightAbsolute : limits.firstAdamMomentAbsolute), 'first Adam state');
        const key = source === 0 ? 'fromRetainedGPUGradient' : 'fromIndependentCPUGradient';
        maxima[key][k] = Math.max(maxima[key][k], delta);
      }
    }
  }
  return {parameters: count, checkedStateComponents: count*4, maximumAbsoluteByComponent: maxima, passed: true};
}

async function main() {
  assert.equal(process.argv.length, 2, 'Run without arguments; the selected cohort is frozen in this analyzer.');
  const manifestBytes = await readFile(path.join(evidence, 'manifest.json')), manifest = JSON.parse(manifestBytes);
  assert.equal(manifest.kind, 'trainable-tsl-public-evidence-v1');
  assert.deepEqual(manifest.reports.map(row => row.id).sort(), [...selected, excluded].sort());
  const reports = new Map();
  for (const entry of manifest.reports) {
    assert.equal(entry.file, entry.id+'.json.gz'); assert.equal(entry.included, selected.includes(entry.id));
    for (const name of ['originalRawSHA256', 'publicJSONSHA256', 'gzipSHA256']) assert.match(entry[name], /^[0-9a-f]{64}$/);
    const compressed = await readFile(path.join(evidence, entry.file));
    assert.equal(hash(compressed), entry.gzipSHA256, 'gzip checksum'); assert.equal(compressed.length, entry.gzipBytes);
    const raw = gunzipSync(compressed, {maxOutputLength: 16*1024*1024});
    assert.equal(hash(raw), entry.publicJSONSHA256, 'decompressed JSON checksum'); assert.equal(raw.length, entry.publicBytes);
    assert.ok(!/[A-Za-z]:[\\/]Users[\\/]|\/home\/|\/Users\//i.test(raw.toString()), 'absolute user path in public report');
    if (entry.included) {
      assert.equal(entry.originalRawSHA256, entry.publicJSONSHA256); assert.deepEqual(entry.redactions, []);
      assert.equal(entry.rawBytes, entry.publicBytes); assert.equal(entry.exclusionReason, null);
    } else {
      assert.ok(entry.exclusionReason && entry.redactions.length === 1);
      assert.equal(entry.redactions[0].path, '/failure');
    }
    // If private/local originals exist, verify their original sidecars too.
    // A fresh clone needs only the checked-in gzip files and manifest.
    let localOriginalVerified = false;
    try {
      const local = path.join(root, 'results/development/trainable-tsl', entry.id);
      const [original, sidecar] = await Promise.all([readFile(path.join(local, 'report.json')), readFile(path.join(local, 'report.sha256'), 'utf8')]);
      assert.equal(hash(original), sidecar.trim()); assert.equal(hash(original), entry.originalRawSHA256);
      localOriginalVerified = true;
    } catch (error) {if (error.code !== 'ENOENT') throw error;}
    reports.set(entry.id, {report: JSON.parse(raw), entry, localOriginalVerified});
  }
  const rejected = reports.get(excluded).report;
  assert.equal(rejected.status, 'failed'); assert.ok(rejected.failure.includes('<workspace>'));
  assert.ok(rejected.errors.some(error => error.includes('404')));
  assert.ok(rejected.probe.results.every(result => !Object.hasOwn(result, 'startingWeights')));
  const cohort = selected.map(id => ({id, ...reports.get(id)})), baseline = cohort[0].report;
  assert.deepEqual(Object.keys(baseline.servedHashes).sort(), [
    '/experiments/trainable-tsl/index.html', '/experiments/trainable-tsl/app.js',
    '/experiments/trainable-tsl/engine.js', '/experiments/trainable-tsl/fields.js', '/experiments/trainable-tsl/autograd.js',
    '/node_modules/three/build/three.tsl.js', '/node_modules/three/build/three.webgpu.js', '/node_modules/three/build/three.core.js',
  ].sort(), 'complete served runtime record');
  for (const {report} of cohort) {
    sameMap(report.sourceHashes, baseline.sourceHashes, 'same recorded source cohort');
    sameMap(report.servedHashes, baseline.servedHashes, 'same bytes served to all four cells');
    for (const [route, digest] of Object.entries(report.servedHashes)) if (route.startsWith('/experiments/')) assert.equal(digest, report.sourceHashes[route.slice(1)], 'served/source hash agreement');
  }
  const frozenRuntimeSourceChecks = {};
  for (const name of ['autograd.js', 'engine.js', 'fields.js']) {
    const relative = 'experiments/trainable-tsl/'+name, actual = hash(await readFile(path.join(root, relative)));
    assert.equal(actual, baseline.sourceHashes[relative], 'frozen current runtime '+name);
    frozenRuntimeSourceChecks[relative] = actual;
  }
  const cells = [];
  for (const {id, report, entry, localOriginalVerified} of cohort) {
    assert.equal(report.kind, 'native-trainable-tsl-v1'); assert.equal(report.status, 'passed');
    assert.deepEqual(report.errors, []); assert.deepEqual(report.probe.errors, []); assert.ok(!report.failure && !report.probe.failure);
    assert.equal(report.initialState.allocated, false); assert.equal(report.initialState.running, false);
    assert.deepEqual(report.probe.finalState, {busy: false, stopped: true, disposed: true});
    const kind = report.field, seed = report.seed, count = fieldCount(kind);
    assert.ok(['neural', 'islands'].includes(kind)); assert.ok([11, 29].includes(seed)); assert.ok(id.endsWith(`-${kind}-${seed}`));
    const initial = initialWeights(kind, seed), samples = samplesFor(kind, seed), stats = report.probe.stats;
    assert.equal(stats.backend, 'tsl'); assert.equal(stats.method, 'standard reverse-mode VJP'); assert.equal(stats.scalarParameters, count);
    assert.equal(stats.designatedInputs, count+3); assert.equal(stats.unusedInputs, 0);
    assert.ok(stats.nativeNodes > 0 && stats.nativeNodes <= 4096 && stats.scalarTapeOps > 0 && stats.scalarTapeOps <= 4096 && stats.maxDepth <= 128);
    assert.equal(report.probe.results.length, 2);
    assert.deepEqual(report.probe.results.map(result => result.method), seed === 11 ? ['ad', 'fd'] : ['fd', 'ad']);
    const snapshots = [auditSnapshot(kind, report.probe.validation, initial, samples, 'initial')];
    const adam = auditAdam(kind, initial, report.probe.validation, snapshots[0].cpu);
    const fits = [];
    for (const result of report.probe.results) {
      assert.equal(result.kind, kind); assert.equal(result.seed, seed); assert.equal(result.customSamples, false);
      assert.equal(result.completed, limits.steps); assert.equal(result.requested, limits.steps); assert.equal(result.stopped, false);
      assert.deepEqual(result.samples, samples); assert.deepEqual(result.initialWeights, initial); assert.deepEqual(result.startingWeights, initial);
      finiteArray(result.weights, count, 'learned weights'); assert.ok(result.weights.every(value => Math.abs(value) <= 8 && Math.fround(value) === value));
      const before = metrics(kind, initial, samples), after = metrics(kind, result.weights, samples);
      verifyMetrics(result.initial, before); verifyMetrics(result.final, after);
      const fraction = after.heldout.rmse/before.heldout.rmse;
      assert.ok(fraction <= limits.maximumHeldoutRMSEFraction); assert.equal(result.utilityPassed, true);
      assert.equal(result.gpuGradientInvocationsPerStep, result.method === 'ad' ? limits.samples : limits.samples*count);
      assert.equal(result.reloadExact, true);
      assert.ok(Number.isFinite(result.completedWallMs) && result.completedWallMs > 0);
      snapshots.push(auditSnapshot(kind, result.finalValidation, result.weights, samples, 'final-'+result.method));
      fits.push({method: result.method, initial: before, final: after, heldoutRMSEFraction: fraction,
        heldoutRMSEReductionPercent: 100*(1-fraction), startingWeightsExactlyRestored: true, completed: result.completed,
        reportedReloadExact: result.reloadExact, reloadReadbackRetained: false,
        gpuGradientInvocationsPerStep: result.gpuGradientInvocationsPerStep, completedWallMs: result.completedWallMs,
        timingScope: result.timingScope});
    }
    for (const snapshot of snapshots) delete snapshot.cpu;
    assert.ok(Array.isArray(report.probe.shaderSources) && report.probe.shaderSources.every(source => typeof source === 'string' && source.length > 0));
    const computeShaderCount = report.probe.shaderSources.filter(source => source.includes('@compute')).length;
    assert.equal(computeShaderCount, 3);
    cells.push({id, field: kind, seed, reportSHA256: entry.publicJSONSHA256, localOriginalVerified,
      browser: report.browser, adapter: report.probe.adapter, integrationStats: stats, snapshots, adam, fits,
      shaderModules: report.probe.shaderSources.length, computeShaderCount, shaderSHA256: report.probe.shaderSources.map(hash)});
  }
  assert.equal(new Set(cells.map(cell => cell.field+'/'+cell.seed)).size, 4);
  const allSnapshots = cells.flatMap(cell => cell.snapshots), allFits = cells.flatMap(cell => cell.fits);
  const differences = cells.map(cell => {
    const ad = cell.fits.find(fit => fit.method === 'ad'), fd = cell.fits.find(fit => fit.method === 'fd');
    return {field: cell.field, seed: cell.seed, adMinusFDHeldoutRMSE: ad.final.heldout.rmse-fd.final.heldout.rmse,
      adCompletedWallMs: ad.completedWallMs, fdCompletedWallMs: fd.completedWallMs};
  });
  return {kind: 'native-trainable-tsl-independent-audit-v1', status: 'passed', createdAt: new Date().toISOString(), limits,
    evidenceManifestSHA256: hash(manifestBytes), analyzerSHA256: hash(await readFile(fileURLToPath(import.meta.url))),
    sameRecordedSourceHashes: baseline.sourceHashes, sameServedHashes: baseline.servedHashes, frozenRuntimeSourceChecks,
    excluded: {id: excluded, ...reports.get(excluded).entry, originalSidecarVerifiedLocally: reports.get(excluded).localOriginalVerified},
    aggregate: {cells: cells.length, completeFits: allFits.length, snapshots: allSnapshots.length,
      comparedGradientComponentsPerMethod: allSnapshots.reduce((sum, snapshot) => sum+snapshot.count, 0),
      maxADAbsolute: Math.max(...allSnapshots.map(snapshot => snapshot.maxADAbsolute)),
      maxFDAbsolute: Math.max(...allSnapshots.map(snapshot => snapshot.maxFDAbsolute)),
      maxReportedDirectPrimalAbsolute: Math.max(...allSnapshots.map(snapshot => snapshot.reportedDirectPrimalMaxError)),
      maxBiasGradientInferredPrimalAbsolute: Math.max(...allSnapshots.map(snapshot => snapshot.biasGradientInferredPrimalMaxError)),
      maximumCPUOracleDelta: Math.max(...allSnapshots.map(snapshot => snapshot.cpuOracleMaximumDelta)),
      minimumHeldoutRMSEReductionPercent: Math.min(...allFits.map(fit => fit.heldoutRMSEReductionPercent)),
      maximumHeldoutRMSEReductionPercent: Math.max(...allFits.map(fit => fit.heldoutRMSEReductionPercent)),
      recordedReloadGatesPassed: allFits.length, exactResetChecksPassed: allFits.length,
      firstAdamStateComponentsChecked: cells.reduce((sum, cell) => sum+cell.adam.checkedStateComponents, 0),
      descriptiveMedianWallMs: Object.fromEntries(['ad', 'fd'].map(method => [method, median(allFits.filter(fit => fit.method === method).map(fit => fit.completedWallMs))]))},
    cells, differences,
    conclusion: 'The selected native forward-only TSL graphs are trainable within 160 GPU updates. No AD speed or quality advantage is established.',
    limitations: ['Single machine/session and two deliberately chosen noiseless, same-family synthetic targets; this is not general game or reconstruction evidence.',
      'All AD/FD/CPU gradients, weights, samples, first Adam states and shader text are retained. Direct GPU prediction and reload readbacks were not retained; their original summary gates remain explicitly distinguished from recomputed checks.',
      'Prediction reconstruction from the additive-bias gradient uses the known squared-residual identity; it is not a retained direct prediction-buffer readback.',
      'CPU-observed completed training includes synchronization and scheduling and is not active GPU time. Four paired cells are insufficient for a general performance or equivalence claim.']};
}

let summary;
try {summary = await main();} catch (error) {
  summary = {kind: 'native-trainable-tsl-independent-audit-v1', status: 'failed', createdAt: new Date().toISOString(),
    error: {name: error.name, message: error.message}};
  process.exitCode = 1;
}
const output = path.join(root, 'results/development/trainable-tsl-analysis', summary.createdAt.replaceAll(':', '-'));
await mkdir(output, {recursive: true});
const bytes = JSON.stringify(summary, null, 2)+'\n';
await writeFile(path.join(output, 'summary.json'), bytes, {flag: 'wx'});
await writeFile(path.join(output, 'summary.sha256'), hash(bytes)+'\n', {flag: 'wx'});
console.log(JSON.stringify({status: summary.status, summary: path.relative(root, path.join(output, 'summary.json')).replaceAll('\\', '/'),
  sha256: hash(bytes), aggregate: summary.aggregate, differences: summary.differences, error: summary.error}, null, 2));
