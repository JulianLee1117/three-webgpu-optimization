import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { normalizeConfig, regularParticles, mergeStates, transmute, stepCPU, particleTotals } from '../experiments/matter-forge/cpu.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), started = performance.now();
const hash = x => createHash('sha256').update(x).digest('hex');
// Frozen before first run. Same particles/material parameters/loads in both lanes.
const protocol = { version: 1, steps: 480, recordEvery: 30, dt: 1 / 240, N: 24, seed: 'regular-lattice-no-randomness',
  beam: { min: [6, 8, 10], max: [18, 11, 13], spacing: .5, mass: .125, volume: .125, mode: 1 },
  load: { min: [11, 13, 10.5], max: [13, 15, 12.5], spacing: .5, mass: .125, volume: .125, mode: 1 },
  qualityGate: 'Both lanes finish finite; solid final load COM y>=8.0 and >=2.0 above liquid control, solid load speed<=2.5, max relative mass error<1e-10.',
  scope: 'A constrained elastic beam with real particle load; no mesh import, fracture, calibrated ice, or arbitrary crafted bridge claim. Shared-grid numerical contact, no hidden bridge collider.' };
const config = normalizeConfig({ N: protocol.N, dt: protocol.dt, gravity: [0, -9.81, 0], bulk: 120, gamma: 5, mu: 80, lambda: 120,
  colliders: [{ name: 'left-bank', min: [3, 2, 8], max: [8, 8, 15] }, { name: 'right-bank', min: [16, 2, 8], max: [21, 8, 15] }],
  anchors: [{ name: 'left-solid-clamp', min: [6, 8.1, 10], max: [8.1, 11.1, 13] }, { name: 'right-solid-clamp', min: [15.9, 8.1, 10], max: [18, 11.1, 13] }] });
const beam = regularParticles(protocol.beam), load = regularParticles(protocol.load), original = mergeStates([beam, load]);
const report = { kind: 'matter-forge-cpu-feasibility-v1', createdAt: new Date().toISOString(), protocol, config,
  beamCount: beam.ids.length, loadCount: load.ids.length, initial: original, sourceSnapshots: [], lanes: [] };
for (const relativePath of ['experiments/matter-forge/cpu.mjs', 'experiments/matter-forge/cpu.test.mjs', 'scripts/check-matter-forge-cpu.mjs']) {
  const text = await fs.readFile(path.join(root, relativePath), 'utf8'); report.sourceSnapshots.push({ path: relativePath, sha256: hash(text), text });
}
function measure(state) {
  let y = 0, speed = 0, mass = 0;
  for (let p = beam.ids.length; p < state.ids.length; p++) { mass += state.mass[p]; y += state.mass[p] * state.x[3 * p + 1]; speed += state.mass[p] * Math.hypot(...state.v.slice(3 * p, 3 * p + 3)); }
  return { loadCenterY: y / mass, loadMeanSpeed: speed / mass, total: particleTotals(state) };
}
for (const mode of ['solid', 'liquid']) {
  const begin = performance.now(), initial = mode === 'solid' ? original : transmute(original, Array.from({ length: beam.ids.length }, (_, i) => i), 0);
  const lane = { mode, initial, frames: [], diagnostics: [], completedSteps: 0, status: 'running' }; let state = initial;
  try {
    for (let step = 1; step <= protocol.steps; step++) {
      const result = stepCPU(config, state); state = result.state; lane.completedSteps = step; lane.diagnostics.push(result.diagnostics);
      if (step % protocol.recordEvery === 0) lane.frames.push({ step, measurement: measure(state), state });
    }
    lane.status = 'completed';
  } catch (error) { lane.status = 'failed-physics-gate'; lane.failure = { message: error.message, details: error.details }; }
  lane.final = state; lane.measurement = measure(state); lane.elapsedMs = performance.now() - begin;
  lane.maxRelativeMassError = Math.max(0, ...lane.diagnostics.map(d => Math.abs(d.massError) / Math.max(1, d.particleMass)));
  report.lanes.push(lane);
}
const [solid, liquid] = report.lanes;
report.gates = { completed: report.lanes.every(l => l.status === 'completed'), solidSupportsLoad: solid.measurement.loadCenterY >= 8,
  distinctFromLiquid: solid.measurement.loadCenterY - liquid.measurement.loadCenterY >= 2,
  solidSettles: solid.measurement.loadMeanSpeed <= 2.5, massTransfer: report.lanes.every(l => l.maxRelativeMassError < 1e-10) };
report.passed = Object.values(report.gates).every(Boolean); report.status = report.passed ? 'passed' : 'failed-prospective-gate'; report.elapsedMs = performance.now() - started;
const directory = path.join(root, 'results/development/matter-forge-cpu', report.createdAt.replaceAll(':', '-'));
await fs.mkdir(path.dirname(directory), { recursive: true }); await fs.mkdir(directory);
const raw = JSON.stringify(report, (_, v) => ArrayBuffer.isView(v) ? Array.from(v) : v) + '\n';
await fs.writeFile(path.join(directory, 'report.json'), raw); await fs.writeFile(path.join(directory, 'report.sha256'), hash(raw) + '  report.json\n');
console.log(JSON.stringify({ report: path.relative(root, path.join(directory, 'report.json')), sha256: hash(raw), status: report.status, gates: report.gates,
  elapsedMs: report.elapsedMs, particles: original.ids.length,
  lanes: report.lanes.map(l => ({ mode: l.mode, status: l.status, steps: l.completedSteps, measurement: l.measurement, failure: l.failure, maxRelativeMassError: l.maxRelativeMassError })) }, null, 2));
