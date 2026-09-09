/** CPU-only audit of six explicitly selected, final-correction throughput runs. */
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {bruteForceQueries, closestPointTriangle, validateBounds, validateMetadataUnchanged} from '../experiments/gpu-deform-collider/reference.mjs';
import {correctTriangleModule, correctedTriangleWGSL} from '../experiments/gpu-deform-collider/closest-point-correction.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const inputBase = 'results/development/gpu-deform-collider';
const docPath = 'experiments/gpu-deform-collider/RESULTS.md';
export const SELECTED_RUNS = Object.freeze([
  '2026-09-09T19-17-44.481Z-wave-32-64',
  '2026-09-09T19-20-39.064Z-wave-32-1024',
  '2026-09-09T19-20-42.674Z-twist-32-64',
  '2026-09-09T19-20-44.950Z-twist-32-1024',
  '2026-09-09T19-20-48.052Z-ripple-32-64',
  '2026-09-09T19-20-50.763Z-ripple-32-1024',
]);
const methods = ['refit', 'interval', 'swept'];
const metrics = ['cpuSubmitMs', 'cpuSubmitPerStepMs', 'wallBatchMs', 'wallPerStepMs', 'gpuTimelineMs', 'gpuTimelinePerStepMs'];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const canonicalObject = object => JSON.stringify(Object.entries(object).sort(([a], [b]) => a.localeCompare(b)));
const median = values => {const s = [...values].sort((a, b) => a - b); return (s[Math.floor((s.length - 1) / 2)] + s[Math.floor(s.length / 2)]) / 2;};
const p95 = values => [...values].sort((a, b) => a - b)[Math.ceil(values.length * .95) - 1];
const near = (actual, expected, description) => assert.ok(Number.isFinite(actual) && Number.isFinite(expected) && Math.abs(actual - expected) <= 1e-11 * Math.max(1, Math.abs(expected)), `${description}: ${actual} != ${expected}`);
const normalizeWGSL = code => code.replace(/\s+/g, '');
const read = name => readFile(path.join(root, name));
const checkKey = sample => `${sample.method}/${sample.time}/${sample.end}`;
const floatBytes = values => Buffer.from(Float32Array.from(values).buffer);

function auditTiming(timing) {
  assert.equal(timing.status, 'completed');
  assert.equal(timing.steps, 16);
  assert.equal(timing.rounds, 6);
  assert.deepEqual(timing.methods, methods);
  assert.equal(timing.warmupBatches, 3);
  assert.equal(timing.completeSixPermutations, true);
  assert.equal(timing.timestampsEnabled, true);
  assert.equal(timing.start, .7);
  assert.equal(timing.end, 1.5);
  assert.equal(timing.times.length, 16);
  timing.times.forEach((time, i) => near(time, .7 + .8 * i / 15, 'same animation times'));
  assert.equal(timing.orders.length, 6);
  assert.equal(new Set(timing.orders.map(order => order.join(','))).size, 6);
  timing.orders.forEach(order => assert.deepEqual([...order].sort(), [...methods].sort()));
  assert.equal(timing.batches.length, 18);
  const medians = {}, distributions = {}, comparisons = {};
  for (let block = 0; block < 6; block++) {
    const batches = timing.batches.filter(batch => batch.block === block).sort((a, b) => a.position - b.position);
    assert.deepEqual(batches.map(batch => batch.position), [0, 1, 2]);
    assert.deepEqual(batches.map(batch => batch.method), timing.orders[block]);
    for (const batch of batches) for (const metric of metrics) assert.ok(Number.isFinite(batch[metric]) && batch[metric] >= 0, `nonfinite timing ${metric}`);
    for (const batch of batches) {
      for (const [batchMetric, stepMetric] of [['cpuSubmitMs', 'cpuSubmitPerStepMs'], ['wallBatchMs', 'wallPerStepMs'], ['gpuTimelineMs', 'gpuTimelinePerStepMs']]) {
        near(batch[stepMetric], batch[batchMetric] / 16, 'batch/step consistency');
      }
    }
  }
  for (const method of methods) {
    const batches = timing.batches.filter(batch => batch.method === method);
    medians[method] = {};
    distributions[method] = {};
    for (const metric of metrics) {
      const values = batches.map(batch => batch[metric]);
      medians[method][metric] = median(values);
      near(medians[method][metric], timing.medians[method][metric], 'reported median');
      distributions[method][metric] = {samples: values, median: median(values), nearestRankP95: p95(values), minimum: Math.min(...values), maximum: Math.max(...values)};
    }
  }
  assert.equal(timing.pairedDifferences.length, 6);
  for (const method of ['interval', 'swept']) {
    comparisons[method] = {};
    for (const metric of metrics) {
      const differences = Array.from({length: 6}, (_, block) => {
        const candidate = timing.batches.find(batch => batch.block === block && batch.method === method);
        const baseline = timing.batches.find(batch => batch.block === block && batch.method === 'refit');
        const difference = candidate[metric] - baseline[metric];
        const recorded = timing.pairedDifferences.find(item => item.block === block);
        near(recorded[method][metric], difference, 'reported paired difference');
        return difference;
      });
      comparisons[method][metric] = {
        percentChangeOfMedians: (medians[method][metric] / medians.refit[metric] - 1) * 100,
        medianPairedDifference: median(differences), pairedDifferences: differences,
        pairedWins: differences.filter(v => v < 0).length,
        pairedTies: differences.filter(v => v === 0).length,
        pairedRegressions: differences.filter(v => v > 0).length,
      };
    }
  }
  return {steps: 16, measuredBatches: 18, medians, distributions, comparisons, metricDefinitions: timing.metrics};
}

function positiveChecks(report) {
  assert.equal(report.correctness.length, 6);
  assert.deepEqual(report.correctness.map(sample => [sample.method, sample.time, sample.end]), [0, .7, 2.4].flatMap(time => ['refit', 'interval'].map(method => [method, time, time])));
  assert.equal(report.swept.length, 3);
  report.swept.forEach((window, i) => {
    const [start, end] = [[0, .75], [.75, 1.5], [2.4, 3.2]][i];
    assert.equal(window.start, start); assert.equal(window.end, end); assert.equal(window.ok, true);
    assert.equal(window.samples.length, 9);
    window.samples.forEach((sample, j) => {
      assert.equal(sample.method, 'swept'); assert.equal(sample.end, end);
      near(sample.time, start + (end - start) * j / 8, 'swept sample time');
    });
  });
  const checks = [...report.correctness, ...report.swept.flatMap(window => window.samples)];
  for (const sample of checks) {
    assert.equal(sample.oracleCount, 64);
    assert.equal(sample.ok, true); assert.equal(sample.wrong, 0); assert.equal(sample.wrongTrianglePoint, 0);
    assert.equal(sample.bounds.ok, true); assert.equal(sample.bounds.errorCount, 0);
    assert.equal(sample.bounds.requireChildContainment, sample.method === 'refit');
    assert.equal(sample.bounds.visitedNodeCount, report.fixture.nodeCount);
    assert.equal(sample.bounds.referencedTriangleCount, report.fixture.triangleCount);
    assert.equal(sample.bounds.unreferencedTriangleCount, 0);
    assert.equal(sample.metadata.ok, true); assert.equal(sample.metadata.mismatches, 0);
  }
  const controls = report.negativeControls;
  assert.equal(controls.detected, true); assert.equal(controls.expectedFailure, true);
  for (const key of ['stale', 'endpoints']) {
    assert.equal(controls[key].ok, false); assert.equal(controls[key].bounds.ok, false);
    assert.ok(controls[key].bounds.errorCount > 0); assert.ok(controls[key].wrong > 0);
  }
  return {checks, controls: [controls.stale, controls.endpoints]};
}

function revalidateSample(proof, sample, recorded, oracleCache) {
  const vertices = Float32Array.from(sample.vertices), indices = Uint32Array.from(proof.indices), queries = Float32Array.from(proof.queries);
  const nodes = Uint32Array.from(sample.nodes), originalNodes = Uint32Array.from(proof.originalNodes);
  assert.equal(sample.queryResults.length, 256); assert.equal(sample.faces.length, 256);
  const key = hash(Buffer.concat([Buffer.from(vertices.buffer), Buffer.from(indices.buffer), Buffer.from(queries.buffer)]));
  if (!oracleCache.has(key)) oracleCache.set(key, bruteForceQueries(vertices, indices, queries));
  const oracle = oracleCache.get(key);
  let wrong = 0, wrongTrianglePoint = 0, maxDistanceSqError = 0, maxPointError = 0, maxPointOnReportedTriangleError = 0, maxReturnedDistanceConsistencyError = 0, largestPointDifference;
  for (let i = 0; i < 64; i++) {
    const output = sample.queryResults, faces = sample.faces;
    const error = Math.abs(output[i * 4 + 3] - oracle.distanceSq[i]);
    maxDistanceSqError = Math.max(maxDistanceSqError, error);
    const pointDifference = Math.hypot(...[0, 1, 2].map(j => output[i * 4 + j] - oracle.closestPoints[i * 4 + j]));
    if (pointDifference > maxPointError) largestPointDifference = {probe: i, pointDifference, squaredDistanceError: error, oracleDistanceSq: oracle.distanceSq[i], returnedDistanceSq: output[i * 4 + 3]};
    maxPointError = Math.max(maxPointError, pointDifference);
    if (!Number.isFinite(error) || error > 2e-4 * (1 + oracle.distanceSq[i]) || faces[i * 4 + 3] >= indices.length / 3) wrong++;
    const tri = faces[i * 4 + 3], ids = Array.from(indices.slice(tri * 3, tri * 3 + 3));
    if (ids.length !== 3 || ids.some((id, j) => id !== faces[i * 4 + j])) { wrongTrianglePoint++; continue; }
    const winner = closestPointTriangle(Array.from(queries.slice(i * 4, i * 4 + 3)), ...ids.map(id => Array.from(vertices.slice(id * 4, id * 4 + 3))));
    maxPointOnReportedTriangleError = Math.max(maxPointOnReportedTriangleError, ...[0, 1, 2].map(j => Math.abs(winner.closestPoint[j] - output[i * 4 + j])));
    maxReturnedDistanceConsistencyError = Math.max(maxReturnedDistanceConsistencyError, Math.abs(winner.distanceSq - output[i * 4 + 3]));
    if ([0, 1, 2].some(j => !Number.isFinite(output[i * 4 + j]) || Math.abs(winner.closestPoint[j] - output[i * 4 + j]) > 2e-4)
      || Math.abs(winner.distanceSq - output[i * 4 + 3]) > 2e-4 * (1 + winner.distanceSq)) wrongTrianglePoint++;
  }
  const bounds = validateBounds(nodes, vertices, indices, {tolerance: 0, requireChildContainment: sample.method === 'refit'});
  const metadata = validateMetadataUnchanged(originalNodes, nodes);
  assert.equal(wrong, recorded.wrong); assert.equal(wrongTrianglePoint, recorded.wrongTrianglePoint);
  assert.equal(bounds.ok, recorded.bounds.ok); assert.equal(bounds.errorCount, recorded.bounds.errorCount);
  assert.equal(bounds.referencedTriangleCount, indices.length / 3); assert.equal(bounds.unreferencedTriangleCount, 0);
  assert.equal(metadata.ok, true); assert.equal(metadata.mismatches, 0);
  near(maxDistanceSqError, recorded.maxDistanceSqError, 'CPU-rechecked maximum squared-distance error');
  near(maxPointError, recorded.maxPointError, 'CPU-rechecked maximum point difference');
  return {method: sample.method, time: sample.time, end: sample.end, wrong, wrongTrianglePoint, boundsErrors: bounds.errorCount,
    maxDistanceSqError, maxPointError, maxPointOnReportedTriangleError, maxReturnedDistanceConsistencyError, largestPointDifference};
}

export async function analyze({revalidate = true} = {}) {
  const started = performance.now(), oracleCache = new Map(), runs = [];
  let baselineSources;
  const required = ['/experiments/gpu-deform-collider/engine.js', '/experiments/gpu-deform-collider/tsl-intervals.js', '/experiments/gpu-deform-collider/refitter.js', '/experiments/gpu-deform-collider/reference.mjs', '/experiments/gpu-deform-collider/timing.js', '/node_modules/three/build/three.tsl.js', '/node_modules/three/build/three.webgpu.js', '/node_modules/three/build/three.core.js', '/node_modules/three-mesh-bvh/src/webgpu/BVHComputeData.js', '/node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js', '/node_modules/three-mesh-bvh/src/webgpu/shapecastFns/getClosestPointToPointFn.js'];
  const correctionHash = hash(await read('experiments/gpu-deform-collider/closest-point-correction.mjs'));
  const referenceHash = hash(await read('experiments/gpu-deform-collider/reference.mjs'));
  for (const id of SELECTED_RUNS) {
    const reportPath = `${inputBase}/${id}/report.json`, reportBytes = await read(reportPath);
    const reportSha256 = hash(reportBytes);
    assert.equal(reportSha256, (await read(`${reportPath}.sha256`)).toString().trim(), `${id}: report SHA mismatch`);
    const report = JSON.parse(reportBytes);
    assert.equal(report.status, 'passed'); assert.equal(report.phase, 'throughput-screen');
    assert.equal(report.protocolComplete, false); assert.equal(report.triangleFunction, 'local-correction');
    assert.deepEqual(report.errors, []); assert.deepEqual(report.changedSources, []); assert.ok(!report.failure);
    assert.equal(report.segments, 32); assert.ok([64, 1024].includes(report.queryCount));
    assert.equal(report.fixture.queryCount, report.queryCount);
    assert.equal(report.sources['experiments/gpu-deform-collider/closest-point-correction.mjs'], correctionHash);
    assert.equal(report.sources['experiments/gpu-deform-collider/reference.mjs'], referenceHash, 'CPU oracle changed since run');
    for (const name of required) assert.match(report.servedSources[name] ?? '', /^[a-f0-9]{64}$/, `missing executed module ${name}`);
    for (const [name, digest] of Object.entries(report.servedSources)) {
      if (report.sources[name.slice(1)] !== undefined) assert.equal(digest, report.sources[name.slice(1)], `${name}: pre-run/served mismatch`);
    }
    if (!baselineSources) baselineSources = report.servedSources;
    else assert.equal(canonicalObject(report.servedSources), canonicalObject(baselineSources), 'Executed source hashes differ across selected runs');
    assert.equal(report.compiler.formalFloatProof, false); assert.equal(report.compiler.trigAbsoluteMargin, .001);
    const {checks, controls} = positiveChecks(report);
    assert.equal(report.proof.path, 'proof.json');
    const proofPath = `${inputBase}/${id}/proof.json`, proofBytes = await read(proofPath), proof = JSON.parse(proofBytes);
    assert.equal(hash(proofBytes), report.proof.sha256, `${id}: proof SHA mismatch`);
    assert.equal(proofBytes.byteLength, report.proof.bytes);
    assert.equal(proof.family, report.family); assert.equal(proof.segments, 32); assert.equal(proof.oracleCount, 64);
    assert.equal(proof.queries.length, 256); assert.equal(proof.indices.length, report.fixture.triangleCount * 3);
    assert.equal(proof.originalNodes.length, report.fixture.nodeCount * 8);
    assert.equal(proof.baseVertices.length, report.fixture.vertexCount * 4);
    const expected = [...checks, ...controls];
    assert.equal(proof.samples.length, 35);
    assert.deepEqual(proof.samples.map(checkKey), expected.map(checkKey));
    assert.ok(proof.shaderSources.some(shader => normalizeWGSL(shader.code).includes(normalizeWGSL(correctedTriangleWGSL))), 'saved GPU query shader lacks the selected correction');
    assert.ok(proof.shaderSources.some(shader => shader.label === 'refit-leaf-bounds'));
    assert.ok(proof.shaderSources.some(shader => shader.label === 'refit-parent-bounds'));
    assert.ok(proof.shaderSources.some(shader => shader.code.includes('ceil(') && shader.code.includes('floor(')), 'saved interval kernel missing range-extrema operations');
    const cpuRevalidation = revalidate ? proof.samples.map((sample, i) => revalidateSample(proof, sample, expected[i], oracleCache)) : null;
    const timing = auditTiming(report.throughput);
    runs.push({id, family: report.family, queryCount: report.queryCount, triangleCount: report.fixture.triangleCount,
      nodeCount: report.fixture.nodeCount, refitDispatches: report.fixture.refitDispatches, browser: report.browser, adapter: report.fixture.adapter,
      reportPath, reportSha256, proofPath, proofSha256: report.proof.sha256, proofBytes: report.proof.bytes,
      sourceManifest: report.sources, servedSources: report.servedSources,
      proofShaderDigests: proof.shaderSources.map(shader => ({label: shader.label, sha256: hash(shader.code), bytes: Buffer.byteLength(shader.code)})),
      queriesSha256: hash(floatBytes(proof.queries)), compiler: report.compiler,
      correctness: {instantChecks: 6, sweptChecks: 27, probesPerCheck: 64, positiveQueryComparisons: 33 * 64,
        wrong: 0, wrongTrianglePoint: 0, descendantBoundsEscapes: 0, metadataMismatches: 0,
        maxDistanceSqError: Math.max(...checks.map(sample => sample.maxDistanceSqError)),
        maxPointDifferenceFromCpuTieChoice: Math.max(...checks.map(sample => sample.maxPointError)),
        negativeControls: controls.map(sample => ({method: sample.method, wrong: sample.wrong, boundsErrors: sample.bounds.errorCount}))},
      cpuRevalidation, timing});
  }
  const correctedSource = correctTriangleModule((await read('node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js')).toString());
  assert.equal(hash(correctedSource), baselineSources['/node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js'], 'Recorded served correction cannot be reproduced');
  const archivedEnginePath = `results/development/gpu-deform-collider-sources/${baselineSources['/experiments/gpu-deform-collider/engine.js']}-engine.js`;
  assert.equal(hash(await read(archivedEnginePath)), baselineSources['/experiments/gpu-deform-collider/engine.js'], 'Archived executed engine SHA mismatch');
  const currentWorkspaceDifferences = [];
  for (const [name, recorded] of Object.entries(baselineSources)) {
    const bytes = name.endsWith('/webgpu/tsl/fns.js') ? Buffer.from(correctedSource) : await read(name.slice(1));
    if (hash(bytes) !== recorded) currentWorkspaceDifferences.push({path: name.slice(1), recordedSha256: recorded, currentSha256: hash(bytes)});
  }
  for (const family of ['wave', 'twist', 'ripple']) {
    const pair = runs.filter(run => run.family === family);
    assert.equal(pair.length, 2); assert.equal(pair[0].queriesSha256, pair[1].queriesSha256);
  }
  const comparisonCounts = {};
  for (const method of ['interval', 'swept']) {
    const changes = runs.map(run => run.timing.comparisons[method].wallPerStepMs.percentChangeOfMedians);
    comparisonCounts[method] = {medianWallWins: changes.filter(v => v < 0).length, medianWallRegressions: changes.filter(v => v > 0).length,
      minimumPercentChange: Math.min(...changes), maximumPercentChange: Math.max(...changes)};
  }
  const practicalGate = Object.fromEntries(['interval', 'swept'].map(method => {
    const changes = runs.filter(run => run.queryCount === 1024).map(run => ({family: run.family, percent: run.timing.comparisons[method].wallPerStepMs.percentChangeOfMedians}));
    const substantialFamilies = changes.filter(change => change.percent <= -20).length;
    return [method, {passed: substantialFamilies >= 2 && changes.every(change => change.percent <= 10), substantialFamilies, changes}];
  }));
  const recheckedPositive = runs.flatMap(run => (run.cpuRevalidation ?? []).filter(sample => ['refit', 'interval', 'swept'].includes(sample.method)));
  const largestPointDifference = recheckedPositive.reduce((best, sample) => !best || sample.maxPointError > best.pointDifference
    ? {...sample.largestPointDifference, method: sample.method, time: sample.time, end: sample.end} : best, null);
  return {kind: 'native-tsl-bounds-final-correction-audit-v1', createdAt: new Date().toISOString(), status: 'passed', gpuExecuted: false,
    selectedRuns: SELECTED_RUNS, selection: 'Explicit six final-correction cells; earlier failed/obsolete runs remain outside this analysis.',
    integrity: {reportHashesVerified: 6, proofHashesVerified: 6, executedSourcesIdenticalAcrossRuns: true,
      servedModuleCountPerRun: Object.keys(baselineSources).length, preRunAndServedOverlapsMatch: true, savedCorrectionShaderFoundEveryRun: true,
      correctionServedHashReproduced: true, archivedEngineHashVerified: true, archivedEnginePath,
      recordedSourceChanges: 0, recordedBrowserGpuErrors: 0, currentWorkspaceDifferences},
    correctness: {instantChecks: 36, sweptChecks: 162, positiveChecks: 198, positiveQueryComparisons: 12672,
      negativeControlChecks: 12, negativeControlsDetected: 12, zeroToleranceDescendantContainmentPassed: true,
      cpuProofRevalidation: revalidate, cpuSamplesRevalidated: revalidate ? 210 : 0,
      uniqueExhaustiveCpuOracles: oracleCache.size, queryComparisonsIncludingNegativeControls: revalidate ? 13440 : 0,
      maxDistanceSqError: Math.max(...runs.map(run => run.correctness.maxDistanceSqError)),
      maxPointDifferenceFromCpuTieChoice: Math.max(...runs.map(run => run.correctness.maxPointDifferenceFromCpuTieChoice)),
      maxPointOnReportedTriangleComponentError: revalidate ? Math.max(...recheckedPositive.map(sample => sample.maxPointOnReportedTriangleError)) : null,
      maxReturnedDistanceConsistencyError: revalidate ? Math.max(...recheckedPositive.map(sample => sample.maxReturnedDistanceConsistencyError)) : null,
      largestPointDifference},
    performance: {comparisonCounts, practicalGate, generalSpeedupSupported: false, fullProtocolComplete: false,
      comparisonToPr853: false, statisticalConfirmation: false}, runs,
    limitations: ['One browser version/device in one exploratory screen and six batches per lane/cell; no independent replication.',
      'GPU timeline includes submission gaps; it is not active kernel execution time.',
      'Only 64 saved probes are oracle-validated even when timing 1024 queries.',
      'Finite temporal samples and heuristic padding are not a formal floating-point enclosure proof or CCD.',
      'The local bottom-up refitter is the comparator; PR #853 was not benchmarked.',
      'The served triangle-distance primitive includes a documented local correction; dependency files were not modified.',
      'Recorded sources are consistent across these runs; later workspace edits do not retroactively alter them.'],
    analysisElapsedMs: performance.now() - started};
}

function makeMarkdown(summary, summaryPath) {
  const fmt = x => x.toFixed(6), percent = x => `${x >= 0 ? '+' : ''}${x.toFixed(2)}%`;
  const list = summary.runs.map(run => {
    const m = run.timing.medians, c = run.timing.comparisons;
    return `- **${run.family}, ${run.queryCount} queries:** completed wall milliseconds per step, exact / instant / swept: **${fmt(m.refit.wallPerStepMs)} / ${fmt(m.interval.wallPerStepMs)} / ${fmt(m.swept.wallPerStepMs)}**. Instant ${percent(c.interval.wallPerStepMs.percentChangeOfMedians)}; swept ${percent(c.swept.wallPerStepMs.percentChangeOfMedians)} versus exact. Paired wall wins: instant ${c.interval.wallPerStepMs.pairedWins}/6, swept ${c.swept.wallPerStepMs.pairedWins}/6. GPU timeline milliseconds per step in the same order: ${fmt(m.refit.gpuTimelinePerStepMs)} / ${fmt(m.interval.gpuTimelinePerStepMs)} / ${fmt(m.swept.gpuTimelinePerStepMs)}. CPU submission: ${fmt(m.refit.cpuSubmitPerStepMs)} / ${fmt(m.interval.cpuSubmitPerStepMs)} / ${fmt(m.swept.cpuSubmitPerStepMs)}.`;
  }).join('\n');
  const runs = summary.runs.map(run => `- [${run.id}](../../${run.reportPath}) — report SHA-256 \`${run.reportSha256}\`; [proof](../../${run.proofPath}) SHA-256 \`${run.proofSha256}\`.`).join('\n');
  const source = summary.runs[0].servedSources;
  return `# Native TSL bounds: final corrected throughput screen

The restricted native TSL compiler generated instantaneous and time-window BVH bounds that passed the saved geometric checks. **This screen does not show a general performance improvement.** Instantaneous bounds won three of six completed-wall medians and regressed in three. Swept reuse won the two ripple cells but regressed in all four wave/twist cells. Neither candidate met the proposed practical performance gate at 1,024 queries.

This is an exploratory implementation result on one NVIDIA Blackwell adapter, Chrome ${summary.runs[0].browser}, Three r185, and the pinned three-mesh-bvh integration. The reports identify vendor/architecture but do not retain the exact GPU model or driver. These six final-correction runs are selected explicitly, not by their timing outcome. Earlier failed and superseded experiments remain in the results directory. The narrower [throughput-screen protocol](THROUGHPUT-SCREEN.md) applies; every report explicitly sets \`protocolComplete:false\`. This is not completion of all gates in [PROTOCOL.md](PROTOCOL.md).

## What passed

- Report and proof SHA-256 checks passed for all six runs. All **${summary.integrity.servedModuleCountPerRun} recorded served-module hashes are identical across the six runs**; pre-run hashes agree wherever both manifests contain the file. Every run records zero browser/GPU errors and zero source changes during execution. The archived generated GPU query shaders contain the selected local triangle-function correction. These are reproducible application logs and hashes, not a cryptographic execution attestation.
- All **36 instantaneous checks plus 162 swept sample checks** passed: **12,672 nearest-query comparisons**, unchanged metadata, and zero descendant-vertex escapes with zero containment tolerance. All ancestors, including the packed TLAS, were checked. Exact refit additionally checks child-box nesting; interval bounds require descendant geometry containment.
- Stale-rest bounds and endpoint-only bounds were detected in every run: **12/12 negative controls**. These controls produced both geometry escapes and wrong nearest results. Endpoint unions alone therefore fail the selected interior-time cases.
- ${summary.correctness.cpuProofRevalidation ? `The analysis independently reran the existing CPU oracle against all **210 saved positive/control snapshots** (${summary.correctness.queryComparisonsIncludingNegativeControls.toLocaleString('en-US')} query comparisons including controls), reproducing the reported results. Identical vertex/query/index inputs share ${summary.correctness.uniqueExhaustiveCpuOracles} exhaustive oracle evaluations; every saved output and bound snapshot is still checked.` : 'CPU proof revalidation was explicitly skipped in this invocation; integrity and report checks still ran.'}
- Maximum positive squared-distance discrepancy: **${summary.correctness.maxDistanceSqError.toExponential(8)}**. Maximum point difference from the CPU tie choice: **${summary.correctness.maxPointDifferenceFromCpuTieChoice.toExponential(8)}**. Valid equidistant triangle ties are allowed; the returned point is also checked against its reported triangle.${summary.correctness.cpuProofRevalidation ? ` At that largest point difference, the squared-distance discrepancy is only ${summary.correctness.largestPointDifference.squaredDistanceError.toExponential(8)}. Across positive snapshots, maximum component error against the nearest point on the reported triangle is ${summary.correctness.maxPointOnReportedTriangleComponentError.toExponential(8)}. The large alternative-point separation is retained as a diagnostic and is not interpreted as a geometric residual.` : ''}

Only 64 probes per snapshot are checked, including in the 1,024-query timing cells. The time-window checks sample nine times in each of three windows. Finite sampling plus empirical float padding is **sample-validated enclosure**, not a formal portable enclosure proof, continuous collision detection, or guaranteed time-of-impact calculation.

## Performance, including losses

Each lane advances the same 16 times over [0.7,1.5], deforming current vertices and running the same query code. Exact refit rebuilds bounds bottom-up each step; instant bounds map every rest box each step; swept bounds prepare once and reuse the boxes. Swept preparation includes its extra deformation. Rendering and validation readbacks are outside timing. Six lane-order permutations provide six batches per lane, after one warmup per lane.

Numbers below are **conventional medians**. Percentages compare medians; negative is better. Paired wins compare candidate versus exact within each permutation block. These are different summaries, so near-tie medians can coexist with mixed paired outcomes. Raw batches, paired differences, median paired differences, and nearest-rank p95 are retained in the analysis JSON. With six batches, nearest-rank p95 equals the maximum and is not a stable tail estimate.

${list}

The primary completed-wall metric includes encoding/submission, measurement markers, queue completion and timestamp resolution/copy overhead; mapping the timestamp result is excluded. GPU timeline includes submission gaps and must not be labeled active kernel time. Submission time alone does not establish completed-work improvement. Per-step values divide a 16-step batch, including amortized preparation, and are not separately measured frame latencies.

At 1,024 queries, instant bounds change completed-wall medians by ${summary.runs.filter(run => run.queryCount === 1024).map(run => `${percent(run.timing.comparisons.interval.wallPerStepMs.percentChangeOfMedians)} (${run.family})`).join(', ')}. Swept bounds change them by ${summary.runs.filter(run => run.queryCount === 1024).map(run => `${percent(run.timing.comparisons.swept.wallPerStepMs.percentChangeOfMedians)} (${run.family})`).join(', ')}. This fails the proposed requirement of at least 20% reduction in two families with at most 10% regression in the third. No repeated-session, cross-device, bounds-volume, traversal-count, or active-kernel comparison is supplied here.

## Interpretation and provenance

The useful integration is interpreting an existing restricted pure native TSL deformation DAG to generate bounds over immutable rest BVH boxes, including a time interval. The interval lane writes independent node bounds in one dispatch, avoiding the exact baseline's dependent bottom-up passes. It retains GPU-authored positions and the existing packed hierarchy. Broader boxes can increase traversal work; this screen cannot attribute its timing differences to that mechanism without separate instrumentation.

Interval arithmetic, refitting and automatic differentiation are established methods. [three-mesh-bvh PR #853](https://github.com/gkjohnson/three-mesh-bvh/pull/853) already covers GPU BVH construction/refit and related GPU workflows. **That implementation was not performance-tested here.** This screen compares our local conventional GPU refitter, not the strongest possible refitter, and does not establish a first GPU collider/refit or algorithmic novelty.

All three timed lanes use the same [documented triangle-query correction](TRIANGLE-QUERY-FINDING.md). The server replaces only the emitted \`closestPointToTriangle\` function; installed dependency files remain unchanged. The upstream primitive had a reproducible obtuse-triangle nearest-point bug; using that broken primitive would invalidate the previous twist/ripple comparisons. This analysis selects only the six final corrected runs below.

Recorded executed engine SHA-256: \`${source['/experiments/gpu-deform-collider/engine.js']}\`.
The [archived executed engine](../../${summary.integrity.archivedEnginePath}) is available and its hash is independently verified by this analysis.
Recorded interval compiler SHA-256: \`${source['/experiments/gpu-deform-collider/tsl-intervals.js']}\`.
Recorded corrected served triangle module SHA-256: \`${source['/node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js']}\`.
${summary.integrity.currentWorkspaceDifferences.length ? `The current workspace has subsequently changed ${summary.integrity.currentWorkspaceDifferences.length} served file(s): ${summary.integrity.currentWorkspaceDifferences.map(item => `\`${item.path}\``).join(', ')}. The analysis records both hashes; this does not change archived consistency.` : 'Current served files also reproduce the recorded hashes, including the server-local correction.'}

## Reproduce the CPU audit

Run \`node scripts/analyze-tsl-bounds.mjs\` from the repository root. It launches no browser or GPU work, verifies the explicit six-run manifest, rechecks saved geometry/query data, recomputes timing summaries, writes a timestamped JSON under \`results/development/gpu-deform-collider-analysis\`, and regenerates this document. \`--no-revalidate\` skips only the exhaustive CPU proof pass and records that omission. No earlier run is deleted or overwritten.

Latest analysis: [timestamped JSON](../../${summaryPath}). Analysis status: ${summary.status}; generated ${summary.createdAt}.

${runs}
`;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const summary = await analyze({revalidate: !process.argv.includes('--no-revalidate')});
  summary.analysisSourceSha256 = hash(await read('scripts/analyze-tsl-bounds.mjs'));
  const outDir = 'results/development/gpu-deform-collider-analysis';
  await mkdir(path.join(root, outDir), {recursive: true});
  const summaryPath = `${outDir}/${summary.createdAt.replaceAll(':', '-')}.json`;
  summary.artifacts = {summaryPath, documentationPath: docPath};
  await writeFile(path.join(root, summaryPath), JSON.stringify(summary, null, 2) + '\n', {flag: 'wx'});
  await writeFile(path.join(root, docPath), makeMarkdown(summary, summaryPath));
  console.log(JSON.stringify({status: summary.status, summaryPath, documentationPath: docPath, integrity: summary.integrity,
    correctness: summary.correctness, performance: summary.performance, analysisElapsedMs: summary.analysisElapsedMs}, null, 2));
}
