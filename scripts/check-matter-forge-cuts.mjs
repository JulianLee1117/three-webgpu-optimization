import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { beamFixture } from '../experiments/matter-forge/fixtures.mjs';
import { stepCPU, transmute, particleTotals } from '../experiments/matter-forge/cpu.mjs';

// Prospective follow-up, written before any execution. No settings search.
const protocol = { kind: 'matter-forge-cuts-v1', steps: 960, recordEvery: 60, dt: 1 / 240,
  lanes: ['intact', 'liquid', 'cut'], edit: { afterCompletedStep: 360, time: 1.5, beamOnly: true, currentXInclusive: [10.5, 13.5], mode: 0 },
  gates: { allCompleteFinite: true, exactIdentityAndMassAtEdit: true, intactFinalLoadYAbove: 8, cutFinalLoadBelowIntactByAtLeast: 2 },
  interpretation: 'Follow-up extends time and makes an explicit local material intervention. The first two-second settling failure remains unchanged. No settling gate, no calibrated fracture or melting claim.' };
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), started = performance.now();
const hash = data => createHash('sha256').update(data).digest('hex');
const report = { kind: 'matter-forge-cuts-v1', createdAt: new Date().toISOString(), protocol, sourceSnapshots: [], lanes: [] };
for (const relativePath of ['experiments/matter-forge/cpu.mjs', 'experiments/matter-forge/fixtures.mjs', 'scripts/check-matter-forge-cuts.mjs']) {
  const text = await fs.readFile(path.join(root, relativePath), 'utf8'); report.sourceSnapshots.push({ path: relativePath, sha256: hash(text), text });
}
function measure(state, beamCount) {
  let y = 0, speed = 0, mass = 0;
  for (let p = beamCount; p < state.ids.length; p++) { mass += state.mass[p]; y += state.mass[p] * state.x[3 * p + 1]; speed += state.mass[p] * Math.hypot(...state.v.slice(3 * p, 3 * p + 3)); }
  return { loadCenterY: y / mass, loadMeanSpeed: speed / mass, total: particleTotals(state) };
}
for (const id of protocol.lanes) {
  const begin = performance.now(), fixture = beamFixture({ mode: id === 'liquid' ? 'liquid' : 'solid' });
  const { config, beamCount, loadCount } = fixture; let state = fixture.state;
  const lane = { id, config, beamCount, loadCount, initial: state, frames: [], diagnostics: [], completedSteps: 0, status: 'running' };
  try {
    for (let step = 1; step <= protocol.steps; step++) {
      const result = stepCPU(config, state); state = result.state; lane.diagnostics.push(result.diagnostics); lane.completedSteps = step;
      if (id === 'cut' && step === protocol.edit.afterCompletedStep) {
        const selection = Array.from({ length: beamCount }, (_, i) => i).filter(i => state.x[3 * i] >= protocol.edit.currentXInclusive[0] && state.x[3 * i] <= protocol.edit.currentXInclusive[1]);
        const before = state, after = transmute(before, selection, 0);
        const preserved = Object.fromEntries(['ids', 'x', 'v', 'C', 'F', 'mass', 'volume'].map(key => [key, before[key].length === after[key].length && before[key].every((value, i) => Object.is(value, after[key][i]))]));
        lane.edit = { before, after, selection, selectedIds: selection.map(i => before.ids[i]), preserved }; state = after;
      }
      if (step % protocol.recordEvery === 0) lane.frames.push({ step, measurement: measure(state, beamCount), state });
    }
    lane.status = 'completed';
  } catch (error) { lane.status = 'failed-physics-gate'; lane.failure = { message: error.message, details: error.details }; }
  lane.final = state; lane.measurement = measure(state, beamCount); lane.elapsedMs = performance.now() - begin;
  lane.maxRelativeMassError = Math.max(0, ...lane.diagnostics.map(d => Math.abs(d.massError) / Math.max(1, d.particleMass)));
  report.lanes.push(lane);
}
const [intact, liquid, cut] = report.lanes;
report.gates = { allCompleteFinite: report.lanes.every(l => l.status === 'completed'),
  exactIdentityAndMassAtEdit: !!cut.edit && cut.edit.selection.length > 0 && Object.values(cut.edit.preserved).every(Boolean),
  intactSupportsLoad: intact.measurement.loadCenterY > protocol.gates.intactFinalLoadYAbove,
  cuttingChangesSupport: intact.measurement.loadCenterY - cut.measurement.loadCenterY >= protocol.gates.cutFinalLoadBelowIntactByAtLeast };
report.passed = Object.values(report.gates).every(Boolean); report.status = report.passed ? 'passed' : 'failed-prospective-gate'; report.elapsedMs = performance.now() - started;
const directory = path.join(root, 'results/development/matter-forge-cuts', report.createdAt.replaceAll(':', '-'));
await fs.mkdir(path.dirname(directory), { recursive: true }); await fs.mkdir(directory);
const raw = JSON.stringify(report, (_, v) => ArrayBuffer.isView(v) ? Array.from(v) : v) + '\n';
await fs.writeFile(path.join(directory, 'report.json'), raw); await fs.writeFile(path.join(directory, 'report.sha256'), hash(raw) + '  report.json\n');
console.log(JSON.stringify({ report: path.relative(root, path.join(directory, 'report.json')), sha256: hash(raw), status: report.status, gates: report.gates, elapsedMs: report.elapsedMs,
  lanes: report.lanes.map(l => ({ id: l.id, status: l.status, steps: l.completedSteps, measurement: l.measurement, selected: l.edit?.selection.length, preserved: l.edit?.preserved, failure: l.failure, maxRelativeMassError: l.maxRelativeMassError })) }, null, 2));
