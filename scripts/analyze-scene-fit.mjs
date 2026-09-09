import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const script = fileURLToPath(import.meta.url);
const root = path.resolve(path.dirname(script), '..');
const inputRoot = path.join(root, 'results/development/scene-fit');
const scenes = ['rover', 'robot', 'pavilion'];
const seeds = [11, 29, 47];
const v1Methods = ['random', 'coordinate', 'candidate'];
const v1SourcePaths = [
  'scripts/run-scene-fit.mjs',
  ...['PROTOCOL.md', 'scenes.js', 'optimizers.mjs', 'gpu-loss.js', 'engine.js', 'app.js', 'index.html']
    .map(name => `experiments/scene-fit/${name}`),
  'node_modules/three/build/three.webgpu.js', 'node_modules/three/build/three.core.js',
];
const versions = {
  'editable-scene-fit-v1': { methods: v1Methods, candidate: 'candidate', sourcePaths: v1SourcePaths },
  'editable-scene-fit-v2': { methods: [...v1Methods, 'least-squares'], candidate: 'least-squares',
    sourcePaths: [...v1SourcePaths, ...['PROTOCOL-V2.md', 'least-squares.mjs', 'gpu-equations.js'].map(name => `experiments/scene-fit/${name}`)] },
};
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const assert = (condition, message) => { if (!condition) throw Error(message); };
const finite = value => typeof value === 'number' && Number.isFinite(value);
const lossValue = value => finite(value) && value >= 0 && value <= 1;
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const close = (a, b, tolerance = 1e-12) => finite(a) && finite(b) && Math.abs(a - b) <= tolerance;
const normalized = value => Array.isArray(value) && value.length === 8 && value.every(x => finite(x) && x >= 0 && x <= 1);
const canonicalSources = sources => JSON.stringify(Object.fromEntries(Object.entries(sources).sort(([a], [b]) => a.localeCompare(b))));
const relative = filename => path.relative(root, filename).split(path.sep).join('/');

// Independent reproduction of the declared Mulberry32 initial-vector schedule.
function expectedInitial(seed) {
  let state = seed >>> 0;
  return Array.from({ length: 8 }, () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  });
}

function verifySourceManifest(report) {
  const version = versions[report.kind];
  assert(version, 'Unsupported scene-fit report version.');
  assert(report.sources && equal(Object.keys(report.sources).sort(), [...version.sourcePaths].sort()), 'Unexpected or incomplete source manifest.');
  for (const digest of Object.values(report.sources)) assert(/^[a-f0-9]{64}$/.test(digest), 'Invalid source SHA256.');
}

export function validateReport(report) {
  const version = versions[report.kind];
  assert(version, 'Unsupported scene-fit report version.');
  const methods = version.methods;
  assert(report.status === 'passed' && !report.failure, 'A failed browser cell is not a successful fit.');
  assert(scenes.includes(report.scene) && seeds.includes(report.seed), 'Unexpected scene/seed.');
  assert(report.budget === 96, 'Expected 96 training renders per solver.');
  assert(Array.isArray(report.errors) && report.errors.length === 0, 'Reported browser errors.');
  assert(report.finalState?.disposed === true && report.finalState.phase === 'disposed'
    && Array.isArray(report.finalState.errors) && report.finalState.errors.length === 0, 'Cleanup/GPU error gate failed.');
  assert(typeof report.browser === 'string' && report.browser.length > 0, 'Missing browser version.');
  verifySourceManifest(report);
  const fixture = report.fixture;
  assert(fixture?.scene === report.scene && fixture.seed === report.seed, 'Fixture scene/seed mismatch.');
  assert(fixture.environment?.threeRevision === '185' && equal(fixture.environment.resolution, [128, 128]), 'Unexpected renderer/resolution.');
  assert(Array.isArray(fixture.parameters) && fixture.parameters.length === 8, 'Expected eight structural parameters.');
  for (const parameter of fixture.parameters) assert(typeof parameter.name === 'string' && finite(parameter.min)
    && finite(parameter.max) && parameter.min < parameter.max, 'Invalid physical bounds.');
  assert(Array.isArray(fixture.target) && fixture.target.length === 8 && fixture.target.every((value, i) => finite(value)
    && value >= fixture.parameters[i].min && value <= fixture.parameters[i].max), 'Invalid declared target.');
  assert(normalized(fixture.initial) && equal(fixture.initial, expectedInitial(report.seed)), 'Initial vector does not match the prespecified seed.');
  const physical = vector => vector.map((value, i) => fixture.parameters[i].min + value * (fixture.parameters[i].max - fixture.parameters[i].min));
  assert(Array.isArray(fixture.initialPhysical) && fixture.initialPhysical.length === 8
    && physical(fixture.initial).every((value, i) => close(value, fixture.initialPhysical[i])), 'Initial physical values differ from normalized bounds.');
  assert(lossValue(fixture.initialTrain) && lossValue(fixture.initialHeldout), 'Invalid initial loss.');
  const parity = fixture.parity;
  assert(parity && lossValue(parity.cpu) && lossValue(parity.gpu) && finite(parity.absoluteDifference)
    && close(Math.abs(parity.cpu - parity.gpu), parity.absoluteDifference)
    && parity.absoluteDifference <= 1e-7 && close(parity.gpu, fixture.initialTrain, 1e-7)
    && parity.targetSelfLoss === 0 && parity.readbackOutsideFitting === true, 'GPU/CPU loss validation failed.');
  if (report.kind === 'editable-scene-fit-v2') validateEquationParity(fixture.equationParity);
  const offset = seeds.indexOf(report.seed);
  const order = [...methods.slice(offset), ...methods.slice(0, offset)];
  assert(equal(report.order, order), 'Method order differs from the prespecified seed rotation.');
  assert(Array.isArray(report.results) && report.results.length === methods.length
    && equal(report.results.map(result => result.method), order), 'Missing, repeated or reordered method.');

  for (const result of report.results) {
    assert(result.evaluations === 96, 'Unequal training-render budget.');
    if (report.kind === 'editable-scene-fit-v2') validateRenderedHistory(result, fixture);
    else assert(result.lossReadbackBytes === 96 * 16, 'Invalid V1 scalar-readback accounting.');
    assert(normalized(result.best) && lossValue(result.bestLoss) && lossValue(result.finalHeldout), 'Invalid final values.');
    assert(finite(result.wallMs) && result.wallMs >= 0 && finite(result.serializedEvaluationMs)
      && result.serializedEvaluationMs >= 0, 'Invalid descriptive wall time.');
    assert(Array.isArray(result.physical) && result.physical.length === 8
      && physical(result.best).every((value, i) => close(value, result.physical[i])), 'Final physical values differ from the fitted vector.');
    if (result.method !== 'least-squares') validateScalarHistory(result, fixture);
  }
  return report;
}

function validateScalarHistory(result, fixture) {
  assert(Array.isArray(result.history) && result.history.length === 96, 'Missing scalar evaluation trajectory.');
  let bestLoss = Infinity;
  let best;
  result.history.forEach((entry, i) => {
    assert(entry.evaluation === i + 1 && normalized(entry.parameters) && lossValue(entry.loss), 'Invalid evaluation ordinal/vector/loss.');
    if (i === 0) assert(equal(entry.parameters, fixture.initial) && close(entry.loss, fixture.initialTrain, 1e-7), 'Methods did not use the shared initial point/loss.');
    if (entry.loss < bestLoss) { bestLoss = entry.loss; best = entry.parameters; }
    assert(entry.bestLoss === bestLoss && equal(entry.best, best), 'Reported incumbent does not follow the recorded loss trajectory.');
  });
  assert(result.bestLoss === bestLoss && equal(result.best, best), 'Final best is not the trajectory incumbent.');
}

function validateEquationParity(parity) {
  const vector = (value, length) => Array.isArray(value) && value.length === length && value.every(finite);
  assert(parity?.passed === true && parity.readbackOutsideFitting === true && parity.perturbationRenders === 16
    && parity.restorationRenders === 1 && vector(parity.cpu?.hessian, 64) && vector(parity.cpu?.gradient, 8)
    && finite(parity.maxAbsoluteDifference) && parity.maxAbsoluteDifference >= 0, 'Missing or failed GPU normal-equation parity.');
  const cpu = [...parity.cpu.hessian, ...parity.cpu.gradient];
  assert(parity.maxAbsoluteDifference <= Math.max(...cpu.map(value => 1e-6 + 1e-4 * Math.abs(value))), 'Equation error exceeds even the largest allowed component tolerance.');
  for (let i = 0; i < 8; i++) {
    assert(parity.cpu.hessian[8 * i + i] >= 0, 'Negative CPU Gram diagonal.');
    for (let j = 0; j < 8; j++) assert(close(parity.cpu.hessian[8 * i + j], parity.cpu.hessian[8 * j + i]), 'CPU Gram matrix is not symmetric.');
  }
  if (parity.gpu) {
    assert(vector(parity.gpu.hessian, 64) && vector(parity.gpu.gradient, 8), 'Invalid retained GPU equation vectors.');
    const gpu = [...parity.gpu.hessian, ...parity.gpu.gradient];
    const deltas = cpu.map((value, i) => Math.abs(value - gpu[i]));
    assert(deltas.every((value, i) => value <= 1e-6 + 1e-4 * Math.abs(cpu[i]))
      && close(Math.max(...deltas), parity.maxAbsoluteDifference), 'Retained GPU/CPU equation values fail parity.');
  }
}

function validateRenderedHistory(result, fixture) {
  const rows = result.renderHistory;
  assert(Array.isArray(rows) && rows.length === 96, 'Expected all 96 rendered training images, including derivative probes.');
  rows.forEach((row, i) => assert(row.evaluation === i + 1 && normalized(row.parameters)
    && ['scalar', 'finite-difference'].includes(row.kind)
    && (row.kind === 'scalar' ? lossValue(row.loss) : !Object.hasOwn(row, 'loss')), 'Invalid rendered-image history.'));
  const scalarCount = rows.filter(row => row.kind === 'scalar').length;
  const probeCount = rows.length - scalarCount;
  assert(result.scalarQueries === scalarCount && Number.isInteger(result.equationBatches) && result.equationBatches >= 0
    && result.equationBatches * 16 === probeCount && result.scalarQueries + 16 * result.equationBatches === 96,
  'Finite-difference renders are not included in the equal budget.');
  assert(result.lossReadbackBytes === result.scalarQueries * 16 + result.equationBatches * 176, 'Invalid scalar/normal-equation readback accounting.');
  assert(rows[0].kind === 'scalar' && equal(rows[0].parameters, fixture.initial)
    && close(rows[0].loss, fixture.initialTrain, 1e-7), 'V2 methods do not share the seeded initial render.');
  if (result.method !== 'least-squares') {
    assert(result.equationBatches === 0 && Array.isArray(result.history) && result.history.length === 96, 'Unexpected derivative batch in a scalar baseline.');
    assert(rows.every((row, i) => equal(row.parameters, result.history[i].parameters) && row.loss === result.history[i].loss), 'Render history and optimizer history disagree.');
    return;
  }
  assert(Array.isArray(result.history) && result.history.length > 0, 'Missing least-squares decision history.');
  let cursor = 0, bestLoss = Infinity, best;
  let batches = 0;
  for (const entry of result.history) {
    assert(normalized(entry.parameters) && entry.firstEvaluation === cursor + 1, 'Missing/reordered decision span.');
    if (entry.kind === 'equations') {
      assert(entry.renderCost === 16 && entry.evaluation === cursor + 16 && equal(entry.parameters, best), 'Equation batch does not use the existing incumbent or costs more/less than 16 renders.');
      assert(Array.isArray(entry.steps) && entry.steps.length === 8 && entry.steps.every(value => finite(value) && value > 0), 'Invalid finite-difference steps.');
      for (let axis = 0; axis < 8; axis++) for (let side = 0; side < 2; side++) {
        const expected = [...entry.parameters];
        expected[axis] = side ? Math.min(1, expected[axis] + entry.steps[axis]) : Math.max(0, expected[axis] - entry.steps[axis]);
        const row = rows[cursor + axis * 2 + side];
        assert(row?.kind === 'finite-difference' && equal(row.parameters, expected), 'Incorrect central finite-difference probe or hidden center render.');
      }
      batches++;
    } else {
      assert(['initial', 'trial', 'fallback'].includes(entry.kind) && entry.renderCost === 1
        && entry.evaluation === cursor + 1, 'Invalid scalar decision span.');
      const row = rows[cursor];
      assert(row?.kind === 'scalar' && equal(row.parameters, entry.parameters) && row.loss === entry.loss, 'Scalar decision/render mismatch.');
      if (cursor === 0) assert(entry.kind === 'initial', 'Least squares must begin with the shared initial evaluation.');
      const accepted = entry.loss < bestLoss;
      assert(entry.accepted === accepted, 'Incorrect acceptance flag.');
      if (accepted) { bestLoss = entry.loss; best = entry.parameters; }
    }
    cursor += entry.renderCost;
    assert(entry.bestLoss === bestLoss && equal(entry.best, best), 'Least-squares incumbent disagrees with rendered scalar losses.');
  }
  assert(cursor === 96 && batches === result.equationBatches && result.bestLoss === bestLoss
    && equal(result.best, best), 'Incomplete least-squares accounting/final result.');
}

function winners(rows, key) {
  const minimum = Math.min(...rows.map(row => row[key]));
  return { methods: rows.filter(row => row[key] === minimum).map(row => row.method), loss: minimum };
}

function contrast(candidate, baseline, initial) {
  return {
    candidateLoss: candidate, strongestBaselineLoss: baseline.loss, strongestBaselineMethods: baseline.methods,
    absoluteImprovement: baseline.loss - candidate,
    relativeImprovementPercent: baseline.loss === 0 ? null : 100 * (baseline.loss - candidate) / baseline.loss,
    candidateBeatsStrongestBaseline: candidate < baseline.loss,
    candidateImprovesInitial: candidate < initial,
  };
}

export function summarizeReport(report) {
  validateReport(report);
  const version = versions[report.kind];
  const methods = version.methods;
  const candidate = report.results.find(row => row.method === version.candidate);
  const baselines = report.results.filter(row => row.method !== version.candidate);
  const training = contrast(candidate.bestLoss, winners(baselines, 'bestLoss'), report.fixture.initialTrain);
  const heldout = contrast(candidate.finalHeldout, winners(baselines, 'finalHeldout'), report.fixture.initialHeldout);
  return {
    scene: report.scene, seed: report.seed, order: report.order, candidateMethod: version.candidate,
    initial: { normalized: report.fixture.initial, physical: report.fixture.initialPhysical,
      trainingLoss: report.fixture.initialTrain, heldoutLoss: report.fixture.initialHeldout },
    trainingWinner: winners(report.results, 'bestLoss'), heldoutWinner: winners(report.results, 'finalHeldout'),
    candidateVersusStrongestBaseline: { training, heldout },
    candidateBeatsAllBaselinesOnBothViews: training.candidateBeatsStrongestBaseline && heldout.candidateBeatsStrongestBaseline,
    candidateVersusEachBaseline: baselines.map(row => ({ method: row.method,
      training: contrast(candidate.bestLoss, { loss: row.bestLoss, methods: [row.method] }, report.fixture.initialTrain),
      heldout: contrast(candidate.finalHeldout, { loss: row.finalHeldout, methods: [row.method] }, report.fixture.initialHeldout) })),
    methods: methods.map(method => {
      const row = report.results.find(item => item.method === method);
      return { method, trainingLoss: row.bestLoss, heldoutLoss: row.finalHeldout,
        evaluations: row.evaluations, normalized: row.best, physical: row.physical,
        wallMs: row.wallMs, legacySerializedEvaluationMsDoNotCompare: row.serializedEvaluationMs,
        scalarQueries: row.scalarQueries ?? row.evaluations, equationBatches: row.equationBatches ?? 0,
        fittingReadbackBytes: row.lossReadbackBytes };
    }),
    verification: { sharedSeededInitial: true, exactTrainingRendersPerMethod: 96,
      completeIncumbentTrajectories: true, lossParity: report.fixture.parity, cleanDisposal: true,
      equationParity: report.fixture.equationParity ?? null,
      independentlyRecheckedEquationComponents: Boolean(report.fixture.equationParity?.gpu) },
  };
}

async function sourceVerification(sources, sourceRefs) {
  const rows = [];
  for (const [filename, expected] of Object.entries(sources)) {
    let currentSha256 = null;
    try { currentSha256 = hash(await readFile(path.join(root, filename))); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const historicalAttempts = [];
    let matchedHistoricalSourceRef = null;
    if (currentSha256 !== expected && !filename.startsWith('node_modules/')) {
      for (const sourceRef of sourceRefs) {
        let sha256 = null;
        try { sha256 = hash(execFileSync('git', ['show', `${sourceRef}:${filename}`],
          { cwd: root, windowsHide: true, maxBuffer: 8 * 1024 * 1024, stdio: ['ignore', 'pipe', 'ignore'] })); } catch { /* Preserve the unverified status. */ }
        historicalAttempts.push({ sourceRef, sha256 });
        if (sha256 === expected) { matchedHistoricalSourceRef = sourceRef; break; }
      }
    }
    rows.push({ path: filename, expectedSha256: expected, currentSha256, historicalAttempts, matchedHistoricalSourceRef,
      matchedBy: currentSha256 === expected ? 'current-file' : matchedHistoricalSourceRef ? 'historical-git-blob' : null });
  }
  return { allMatched: rows.every(row => row.matchedBy !== null), historicalSourceRefs: sourceRefs, files: rows };
}

async function readReport(filename) {
  const [resolved, resolvedRoot] = await Promise.all([realpath(filename), realpath(inputRoot)]);
  const rel = path.relative(resolvedRoot, resolved);
  assert(rel !== '' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel)
    && path.basename(resolved) === 'report.json', 'Report must remain inside results/development/scene-fit.');
  const bytes = await readFile(resolved);
  const expected = (await readFile(resolved + '.sha256', 'utf8')).trim();
  assert(/^[a-f0-9]{64}$/.test(expected) && hash(bytes) === expected, `Report checksum mismatch: ${relative(resolved)}`);
  return { sourceReport: relative(resolved), sourceReportSha256: expected, report: JSON.parse(bytes.toString('utf8')) };
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes('--help')) {
    console.log('Usage: node scripts/analyze-scene-fit.mjs [--source-ref=<historical-commit> ...] [report.json ...]\nDefault: inventory scene-fit reports; V1/V2 and different source identities stay separate. Failed cells remain failures.');
    return;
  }
  const requestedRefs = args.filter(arg => arg.startsWith('--source-ref=')).map(arg => arg.slice('--source-ref='.length));
  assert(requestedRefs.length <= 8 && requestedRefs.every(Boolean), 'Supply at most eight nonempty historical source refs.');
  const sourceRefs = [...new Set(requestedRefs.map(ref => execFileSync('git', ['rev-parse', '--verify', `${ref}^{commit}`],
    { cwd: root, encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] }).trim()))];
  const explicit = args.filter(arg => !arg.startsWith('--source-ref='));
  assert(explicit.every(arg => !arg.startsWith('--')), 'Unknown option.');
  const filenames = explicit.length ? explicit.map(filename => path.resolve(filename))
    : (await readdir(inputRoot, { withFileTypes: true })).filter(entry => entry.isDirectory())
      .map(entry => path.join(inputRoot, entry.name, 'report.json')).sort();
  assert(filenames.length > 0, 'No scene-fit reports.');
  const inputs = await Promise.all(filenames.map(readReport));
  assert(new Set(inputs.map(input => input.sourceReport)).size === inputs.length, 'Duplicate report path.');
  const groups = new Map();
  const excluded = [];
  for (const input of inputs) {
    const report = input.report;
    if (!versions[report.kind]) {
      excluded.push({ sourceReport: input.sourceReport, sourceReportSha256: input.sourceReportSha256,
        kind: report.kind, reason: 'Unsupported experiment version; not pooled with supported reports.' });
      continue;
    }
    verifySourceManifest(report);
    assert(scenes.includes(report.scene) && seeds.includes(report.seed) && report.budget === 96, 'Unexpected scene/seed/budget.');
    const key = hash(report.kind + '\n' + canonicalSources(report.sources));
    if (!groups.has(key)) groups.set(key, { kind: report.kind, sourceIdentitySha256: key, sources: report.sources, inputs: [] });
    groups.get(key).inputs.push(input);
  }
  assert(groups.size > 0, 'No supported scene-fit reports found.');
  const cohorts = [];
  for (const group of groups.values()) {
    const seen = new Set();
    const sceneFixtures = new Map();
    const cells = [];
    const failures = [];
    for (const input of group.inputs) {
      const report = input.report;
      const key = `${report.scene}:${report.seed}`;
      assert(!seen.has(key), `Duplicate attempt for ${key} under one source identity; explicit selection/accounting is required.`);
      seen.add(key);
      const provenance = { sourceReport: input.sourceReport, sourceReportSha256: input.sourceReportSha256 };
      if (report.status !== 'passed') {
        failures.push({ ...provenance, scene: report.scene, seed: report.seed, status: report.status,
          failure: report.failure ?? null, failedState: report.failedState ?? null,
          recordedFitCount: report.results?.length ?? 0, includedInSuccessfulFitComparisons: false });
        continue;
      }
      const cell = summarizeReport(report);
      const fixture = JSON.stringify({ parameters: report.fixture.parameters, target: report.fixture.target, environment: report.fixture.environment, browser: report.browser });
      assert(!sceneFixtures.has(report.scene) || sceneFixtures.get(report.scene) === fixture, 'Scene fixture/environment changed across seeds.');
      sceneFixtures.set(report.scene, fixture);
      cells.push({ ...provenance, ...cell });
    }
    const sourceCheck = await sourceVerification(group.sources, sourceRefs);
    const missing = scenes.flatMap(scene => seeds.filter(seed => !seen.has(`${scene}:${seed}`)).map(seed => ({ scene, seed })));
    const complete = cells.length === 9 && failures.length === 0 && missing.length === 0;
    const counterexamples = cells.filter(cell => !cell.candidateBeatsAllBaselinesOnBothViews).map(cell => ({ scene: cell.scene, seed: cell.seed }));
    const candidate = versions[group.kind].candidate;
    cohorts.push({ kind: group.kind, candidateMethod: candidate, sourceIdentitySha256: group.sourceIdentitySha256, sourceVerification: sourceCheck,
      coverage: { expectedCells: 9, passedCells: cells.length, failedCells: failures.length, missing },
      decision: { completeNineCellMatrix: complete, allSourcesVerified: sourceCheck.allMatched,
        observedCandidateCounterexamples: counterexamples,
        candidatePracticalGateMet: complete && sourceCheck.allMatched && counterexamples.length === 0,
        interpretation: !complete ? 'Incomplete matrix; failed/missing cells are not passes.'
          : counterexamples.length ? 'Candidate does not beat every existing solver on both views throughout the declared matrix.'
            : !sourceCheck.allMatched ? 'Source verification incomplete.' : 'Descriptive practical gate met; no statistical or novelty claim.' },
      perScene: scenes.map(scene => ({ scene, cells: cells.filter(cell => cell.scene === scene).sort((a, b) => a.seed - b.seed),
        failures: failures.filter(cell => cell.scene === scene), missingSeeds: missing.filter(cell => cell.scene === scene).map(cell => cell.seed) })),
    });
  }
  const created = new Date().toISOString();
  const summary = { kind: 'editable-scene-fit-analysis-v2', created, analyzerSha256: hash(await readFile(script)),
    inputSelection: explicit.length ? 'explicit report paths; coverage remains explicit' : 'all reports present in scene-fit at invocation',
    endpoint: 'RGB image MSE at 128x128; lower is better. Held-out camera is evaluated only after fitting.',
    comparison: 'Strongest baseline is the lowest-loss alternative separately for each view: random/coordinate for V1 Powell, and random/coordinate/Powell for V2 least squares. Each baseline contrast is also retained. Endpoint-wise retrospective selection is not a deployable selector.',
    improvementSign: 'Positive baseline-minus-candidate improvement favors the candidate; negative values favor the strongest baseline.',
    methodLabels: { random: 'Uniform random search', coordinate: 'Coordinate pattern search', candidate: 'Budget-truncated Powell-style direction search with bounded Brent line search', 'least-squares': 'Central finite differences with GPU normal equations and bounded Levenberg-Marquardt' },
    limitations: [
      'Small synthetic known-family fixtures with fixed cameras, lighting and colors; not arbitrary screenshot-to-scene reconstruction or evidence about AI-generated scenes.',
      'The algorithms, finite differences, normal equations and GPU reductions are established methods. This evaluates an editable-scene integration, not optimizer novelty.',
      'Each solver uses 96 training renders including the initial point. V2 equation batches cost 16 renders each; scalarQueries + 16 * equationBatches must equal 96. Initial/held-out/parity validation renders are outside fitting.',
      'Per-scene, per-seed counterexamples and technical failures remain visible. There is no pooled superiority estimate or statistical confidence claim.',
      'Wall times are descriptive CPU-observed durations including waiting/UI work, not GPU timestamps or a speedup comparison.',
      'Do not compare the original V2 serializedEvaluationMs field: finite-difference batches include deliberate 4ms pacing pauses while scalar queries exclude them. The summary renames it legacySerializedEvaluationMsDoNotCompare; later timing-boundary repairs do not change these reports.',
      'Checksums establish internal artifact consistency, not independent authenticity. Original runners record source hashes before execution, not complete before/after served-source attestations.',
      'Matching recorded sources to current files or historical blobs does not prove those exact bytes were served. Screenshots are not covered by a recorded artifact checksum.',
      'V2 records a browser assertion that all 72 GPU/CPU normal-equation components pass, with CPU vectors and maximum difference. When GPU vectors are absent, this analyzer cannot independently reconstruct every component comparison.',
    ], excluded, cohorts };
  const parent = path.join(root, 'results/development/scene-fit-analysis');
  await mkdir(parent, { recursive: true });
  const out = path.join(parent, created.replaceAll(':', '-'));
  await mkdir(out);
  const bytes = Buffer.from(JSON.stringify(summary, null, 2) + '\n');
  await writeFile(path.join(out, 'summary.json'), bytes, { flag: 'wx' });
  await writeFile(path.join(out, 'summary.json.sha256'), hash(bytes) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ outputFile: relative(path.join(out, 'summary.json')),
    cohorts: cohorts.map(cohort => ({ kind: cohort.kind, candidateMethod: cohort.candidateMethod, sourceIdentitySha256: cohort.sourceIdentitySha256, coverage: cohort.coverage,
      decision: cohort.decision, cells: cohort.perScene.flatMap(scene => scene.cells.map(cell => ({ scene: cell.scene, seed: cell.seed,
        trainingWinner: cell.trainingWinner.methods, heldoutWinner: cell.heldoutWinner.methods,
        candidateImprovementPercent: { training: cell.candidateVersusStrongestBaseline.training.relativeImprovementPercent,
          heldout: cell.candidateVersusStrongestBaseline.heldout.relativeImprovementPercent } }))) })) }, null, 2));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.stack ?? error); process.exitCode = 1; });
}
