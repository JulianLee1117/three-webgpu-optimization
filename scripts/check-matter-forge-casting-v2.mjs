import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { castingFixtureV2 } from '../experiments/matter-forge/casting-fixture-v2.mjs';
import { stepCPU, transmute, makeState, particleTotals } from '../experiments/matter-forge/cpu.mjs';

// Frozen before first execution. One cast and two exact-state branches,2880steps.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), begin = performance.now();
const hash = data => createHash('sha256').update(data).digest('hex');
const fixture = castingFixtureV2(), { castingConfig, releasedConfig, protocol, bodyCount, loadCount } = fixture;
const report = { kind: 'matter-forge-casting-v2', createdAt: new Date().toISOString(), protocol, castingConfig, releasedConfig, bodyCount, loadCount,
  initial: fixture.state, sourceSnapshots: [], lanes: [] };
for (const relativePath of ['experiments/matter-forge/cpu.mjs', 'experiments/matter-forge/casting-fixture.mjs', 'experiments/matter-forge/casting-fixture-v2.mjs', 'scripts/check-matter-forge-casting-v2.mjs']) {
  const text = await fs.readFile(path.join(root, relativePath), 'utf8'); report.sourceSnapshots.push({ path: relativePath, sha256: hash(text), text });
}
const sameArray = (a, b) => a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
function measure(state) {
  let below = 0, central = 0, bodySpeed = 0, loadY = 0, loadSpeed = 0, loadMass = 0;
  const region = protocol.gate.centralRegion, cb = protocol.gate.centralBins;
  const supports = releasedConfig.anchors.map(box => ({ name: box.name, particles: 0, referenceVolume: 0 }));
  const centralBins = cb.xEdges.slice(0, -1).map((minX, i) => ({ minX, maxX: cb.xEdges[i + 1], particles: 0, referenceVolume: 0,
    binVolume: (cb.xEdges[i + 1] - minX) * (cb.maxY - cb.minY) * (cb.maxZ - cb.minZ) }));
  for (let p = 0; p < state.ids.length; p++) {
    const x = state.x[3 * p], y = state.x[3 * p + 1], z = state.x[3 * p + 2];
    const speed = Math.hypot(state.v[3 * p], state.v[3 * p + 1], state.v[3 * p + 2]);
    if (p < bodyCount) {
      if (y < protocol.gate.belowGrilleY) below++;
      if ([0, 1, 2].every(d => state.x[3 * p + d] >= region.min[d] && state.x[3 * p + d] <= region.max[d])) central++;
      bodySpeed += speed;
      releasedConfig.anchors.forEach((box, i) => { if ([0, 1, 2].every(d => state.x[3 * p + d] >= box.min[d] && state.x[3 * p + d] <= box.max[d])) { supports[i].particles++; supports[i].referenceVolume += state.volume[p]; } });
      if (y >= cb.minY && y <= cb.maxY && z >= cb.minZ && z <= cb.maxZ) for (const bin of centralBins) if (x >= bin.minX && x < bin.maxX) { bin.particles++; bin.referenceVolume += state.volume[p]; }
    } else { loadMass += state.mass[p]; loadY += state.mass[p] * y; loadSpeed += state.mass[p] * speed; }
  }
  return { bodyFractionBelowGrille: below / bodyCount, centralBodyParticles: central, bodyMeanSpeed: bodySpeed / bodyCount,
    supports, centralBins: centralBins.map(bin => ({ ...bin, referenceVolumeOccupancy: bin.referenceVolume / bin.binVolume })),
    loadCenterY: loadY / loadMass, loadMeanSpeed: loadSpeed / loadMass, total: particleTotals(state) };
}
const computeStart = performance.now();
function run(id, initial, config, steps) {
  const started = performance.now(), lane = { id, initial, frames: [], diagnostics: [], completedSteps: 0, status: 'running' }; let state = initial;
  try {
    for (let step = 1; step <= steps; step++) {
      if (performance.now() - computeStart > protocol.computeBudgetMs) throw new Error('declared 12-second whole-matrix CPU budget reached');
      const result = stepCPU(config, state); state = result.state; lane.completedSteps = step; lane.diagnostics.push(result.diagnostics);
      if (step % protocol.recordEvery === 0) lane.frames.push({ stageStep: step, measurement: measure(state), state });
    }
    lane.status = 'completed';
  } catch (error) { lane.status = 'failed-physics-or-budget-gate'; lane.failure = { message: error.message, details: error.details }; }
  lane.final = state; lane.measurement = measure(state); lane.elapsedMs = performance.now() - started;
  lane.maxRelativeMassError = Math.max(0, ...lane.diagnostics.map(d => Math.abs(d.massError) / Math.max(1, d.particleMass)));
  report.lanes.push(lane); return lane;
}
const casting = run('casting', fixture.state, castingConfig, protocol.castingSteps);
let solid, liquid;
if (casting.status === 'completed') {
  const selection = Array.from({ length: bodyCount }, (_, i) => i), solidState = transmute(casting.final, selection, 1), liquidState = makeState(casting.final);
  report.edit = { selection, preserved: Object.fromEntries(['ids', 'x', 'v', 'C', 'mass', 'volume'].map(key => [key, sameArray(casting.final[key], solidState[key])])),
    newReferenceIsIdentity: selection.every(p => [0, 1, 2, 3, 4, 5, 6, 7, 8].every(i => solidState.F[9 * p + i] === (i % 4 === 0 ? 1 : 0))),
    exactPhysicalStateAcrossBranches: ['ids', 'x', 'v', 'C', 'mass', 'volume'].every(key => sameArray(solidState[key], liquidState[key])) };
  solid = run('cast-solid-release', solidState, releasedConfig, protocol.releasedSteps);
  liquid = run('cast-liquid-release', liquidState, releasedConfig, protocol.releasedSteps);
}
report.gates = { allStagesComplete: report.lanes.length === 3 && report.lanes.every(l => l.status === 'completed'),
  materialPassedGrille: casting.measurement.bodyFractionBelowGrille > protocol.gate.fractionBelowGrilleGreaterThan,
  centralCastingPresent: casting.measurement.centralBodyParticles >= protocol.gate.minimumCentralBodyParticles,
  bothSupportsPopulated: casting.measurement.supports.every(s => s.particles >= protocol.gate.minimumParticlesInEachSupport),
  allCentralBinsPopulated: casting.measurement.centralBins.every(b => b.particles >= protocol.gate.centralBins.minimumParticlesPerBin),
  editPreservesState: !!report.edit && Object.values(report.edit.preserved).every(Boolean) && report.edit.newReferenceIsIdentity && report.edit.exactPhysicalStateAcrossBranches,
  exactParticleIdentityAndMassThroughout: report.lanes.every(l => ['ids', 'mass', 'volume'].every(key => sameArray(l.final[key], fixture.state[key]))),
  solidSupportsLoad: !!solid && solid.measurement.loadCenterY >= protocol.gate.solidFinalLoadYAtLeast,
  solidBeatsLiquidControl: !!solid && !!liquid && solid.measurement.loadCenterY - liquid.measurement.loadCenterY >= protocol.gate.solidAboveLiquidByAtLeast,
  massTransfer: report.lanes.every(l => l.maxRelativeMassError < protocol.gate.maxRelativeMassError) };
report.passed = Object.values(report.gates).every(Boolean); report.status = report.passed ? 'passed' : 'failed-prospective-gate'; report.elapsedMs = performance.now() - begin;
const directory = path.join(root, 'results/development/matter-forge-casting-v2', report.createdAt.replaceAll(':', '-'));
await fs.mkdir(path.dirname(directory), { recursive: true }); await fs.mkdir(directory);
const raw = JSON.stringify(report, (_, value) => ArrayBuffer.isView(value) ? Array.from(value) : value) + '\n';
await fs.writeFile(path.join(directory, 'report.json'), raw); await fs.writeFile(path.join(directory, 'report.sha256'), hash(raw) + '  report.json\n');
console.log(JSON.stringify({ report: path.relative(root, path.join(directory, 'report.json')), sha256: hash(raw), status: report.status, gates: report.gates,
  elapsedMs: report.elapsedMs, lanes: report.lanes.map(l => ({ id: l.id, status: l.status, completedSteps: l.completedSteps, measurement: l.measurement, failure: l.failure })) }, null, 2));
