// Bounded CPU-only external-asset parity canary; no renderer or GPU is imported.
// Fixed before the first run: both bundled files, 256 equal-weight samples,
// 48 RKPM centers, six modes INCLUDING constant, E40/rho1/nu.3 mechanics.
// Four 1/60 steps: two with one top-point spring, then two after releasing it.
// These short trajectories are correctness checks, not a convergence/quality claim.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { prepareBasis, evaluateBasis } from '../experiments/local-elasticity/basis.mjs';
import { createMechanics } from '../experiments/local-elasticity/mechanics.mjs';
import { decodeSplat, normalizePositions, sampleQuadrature } from '../experiments/local-elasticity/assets.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const sha = data => createHash('sha256').update(data).digest('hex');
const plain = value => JSON.parse(JSON.stringify(value, (_, x) => ArrayBuffer.isView(x) ? Array.from(x) : x));
const sourcePaths = ['scripts/check-local-elasticity-cpu.mjs',
  'experiments/local-elasticity/reference.py', 'experiments/local-elasticity/basis.mjs',
  'experiments/local-elasticity/mechanics.mjs', 'experiments/local-elasticity/assets.mjs'];
const report = { kind: 'local-elasticity-cpu-canary-v1', created: new Date().toISOString(),
  scope: 'Independent parity inputs for two external assets; four steps each; no GPU and no physical calibration.',
  settings: { points: 256, centers: 48, modesIncludingConstant: 6, basisYoung: 1000,
    young: 40, density: 1, poisson: .3, dt: 1 / 60, steps: 4, dragSteps: 2,
    dragOffset: [.08, .02, 0], dragStiffness: 40, gravity: [0, -1.5, 0], floor: -.62,
    maxIterations: 24, maxEvaluations: 160, maxMilliseconds: 1000, seed: 12345 },
  sources: {}, assets: [], basisCases: [], mechanicsCases: [] };
const destination = path.join(root, 'results/development/local-elasticity-cpu',
  report.created.replaceAll(':', '-'));
await mkdir(destination, { recursive: true });
for (const source of sourcePaths) {
  const bytes = await readFile(path.join(root, source));
  report.sources[source] = { sha256: sha(bytes), text: bytes.toString('utf8') };
}
let failed = false;
try {
  for (const asset of ['plant', 'spot']) {
    const bytes = await readFile(path.join(root, 'experiments/local-elasticity/assets', `${asset}.splat`));
    const decoded = decodeSplat(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const normalized = normalizePositions(decoded.positions);
    const samples = sampleQuadrature(normalized.positions, report.settings.points, report.settings.seed);
    const basisInput = { positions: samples.positions, volumes: samples.volumes,
      centerCount: report.settings.centers, modeCount: report.settings.modesIncludingConstant,
      young: report.settings.basisYoung, poisson: report.settings.poisson };
    const basisStart = performance.now();
    const basis = prepareBasis(basisInput), fields = evaluateBasis(basis, samples.positions);
    report.assets.push({ id: asset, file: `experiments/local-elasticity/assets/${asset}.splat`,
      sha256: sha(bytes), bytes: bytes.length, splats: decoded.count,
      normalization: { center: normalized.center, scale: normalized.scale, flipY: normalized.flipY },
      sampleIndices: samples.indices, assumption: samples.assumption });
    report.basisCases.push({ asset, input: basisInput, actual: { ...basis, ...fields },
      elapsedMs: performance.now() - basisStart });
    const input = { restPositions: samples.positions, volumes: samples.volumes, ...fields,
      density: report.settings.density, young: report.settings.young, poisson: report.settings.poisson };
    const model = createMechanics(input);
    const record = { asset, input, massMatrix: model.massMatrix, diagnostics: model.diagnostics,
      states: [], steps: [] };
    report.mechanicsCases.push(record);
    const capture = (id, q, hessian = false) => record.states.push({ id,
      q: Float64Array.from(q), ...model.energyGradient(q, { hessian }) });
    capture('rest', model.createState().q);
    const perturbation = Float64Array.from({ length: model.dofs }, (_, i) => .0001 * Math.sin((i + 1) * 1.7));
    capture('fixed-independent-perturbation', perturbation, true);
    const ordered = Array.from({ length: samples.positions.length / 3 }, (_, i) => i)
      .sort((a, b) => samples.positions[3 * a + 1] - samples.positions[3 * b + 1]);
    const point = index => Array.from(samples.positions.subarray(3 * index, 3 * index + 3));
    const pins = ordered.slice(0, 3).map(index => ({ index, target: point(index) }));
    const top = ordered.at(-1), target = point(top).map((v, axis) => v + report.settings.dragOffset[axis]);
    let state = model.createState();
    for (let index = 0; index < report.settings.steps; index++) {
      const options = { dt: report.settings.dt, gravity: report.settings.gravity, pins,
        floor: report.settings.floor, maxIterations: report.settings.maxIterations,
        maxEvaluations: report.settings.maxEvaluations, maxMilliseconds: report.settings.maxMilliseconds,
        drag: index < report.settings.dragSteps ? { index: top, target, stiffness: report.settings.dragStiffness } : null };
      const before = plain(state), next = model.step(state, options);
      record.steps.push({ index, phase: options.drag ? 'displaced-target' : 'release',
        qBefore: before.q, velocityBefore: before.velocity, qAfter: Float64Array.from(next.state.q),
        velocityAfter: Float64Array.from(next.state.velocity), dt: options.dt, gravity: options.gravity,
        pins, drag: options.drag, floorHeight: options.floor,
        beforeObjective: next.diagnostics.initialEnergy, afterObjective: next.diagnostics.finalEnergy,
        diagnostics: next.diagnostics });
      state = next.state;
      capture(`step-${index + 1}`, state.q);
    }
    console.log(JSON.stringify({ asset, states: record.states.length, steps: record.steps.length,
      statuses: record.steps.map(x => x.diagnostics.status),
      maxPinError: Math.max(...record.steps.map(x => x.diagnostics.pinError)),
      maxFloorPenetration: Math.max(...record.steps.map(x => x.diagnostics.floorPenetration)),
      minDeterminant: Math.min(...record.steps.map(x => x.diagnostics.minDeterminant)) }));
  }
  report.status = 'completed';
} catch (error) {
  failed = true;
  report.status = 'failed';
  report.failure = String(error.stack ?? error).replaceAll(root, '<workspace>/');
}
const encoded = JSON.stringify(report, (_, x) => ArrayBuffer.isView(x) ? Array.from(x) : x, 2) + '\n';
await writeFile(path.join(destination, 'report.json'), encoded, { flag: 'wx' });
await writeFile(path.join(destination, 'report.sha256'), `${sha(encoded)}  report.json\n`, { flag: 'wx' });
console.log(JSON.stringify({ status: report.status, report: path.relative(root, path.join(destination, 'report.json')), sha256: sha(encoded) }));
if (failed) process.exitCode = 1;
