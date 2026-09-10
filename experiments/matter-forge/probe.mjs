import { normalizeConfig, makeState, regularParticles, stepCPU, particleTotals, determinant } from './cpu.mjs';
import { createSimulator, shaderSources } from './gpu.mjs';

// Declared before the first GPU execution. These bounds compare Float64 CPU
// transfers with Float32 shaders and 1e-6 fixed-point atomic grid accumulation.
// They are absolute state errors, not claims of continuum-physics accuracy.
export const TOLERANCES = Object.freeze({ x: 2e-4, v: 3e-3, C: 6e-3, F: 5e-4, momentum: 5e-3 });
export const LIMITS = Object.freeze({ wallClockMs: 38000, fixtures: 4, particlesPerFixture: 64, totalSteps: 25, gridN: 12 });
const FIELDS = ['x', 'v', 'C', 'F'];
const ARRAY_FIELDS = ['ids', 'mode', 'mass', 'volume', ...FIELDS];
const serialState = state => ({ ...Object.fromEntries(ARRAY_FIELDS.map(key => [key, Array.from(state[key])])), time: state.time, steps: state.steps });
const magnitude = values => Math.max(0, ...values.map(Math.abs));

function float32State(state) {
  return makeState({ ...state, ...Object.fromEntries(['mass', 'volume', ...FIELDS].map(key => [key, Float32Array.from(state[key])])) });
}

export function makeFixtures() {
  const common = { N: 12, dt: Math.fround(1 / 240), bulk: 120, gamma: 5, mu: 80, lambda: 120, boundary: 2 };
  const cloud = (mode = 1, volume = .125) => regularParticles({ min: [4, 4, 4], max: [6, 6, 6], spacing: .5, mode, mass: .125, volume });
  const affine = cloud();
  const A = [.12, -.08, .03, .04, -.06, .09, -.02, .07, .05], shift = [.2, -.1, .15];
  for (let p = 0; p < 64; p++) {
    affine.C.set(A, 9 * p);
    for (let r = 0; r < 3; r++) affine.v[3 * p + r] = shift[r] + A[3 * r] * (affine.x[3 * p] - 5) + A[3 * r + 1] * (affine.x[3 * p + 1] - 5) + A[3 * r + 2] * (affine.x[3 * p + 2] - 5);
  }
  const deformed = cloud(), F = [1.08, .13, -.04, .02, .94, .07, -.03, .01, 1.03];
  const B = [.06, -.09, .04, .02, -.04, .07, -.03, .08, .01];
  for (let p = 0; p < 64; p++) {
    deformed.F.set(F, 9 * p); deformed.C.set(B, 9 * p);
    for (let r = 0; r < 3; r++) deformed.v[3 * p + r] = B[3 * r] * (deformed.x[3 * p] - 5) + B[3 * r + 1] * (deformed.x[3 * p + 1] - 5) + B[3 * r + 2] * (deformed.x[3 * p + 2] - 5);
  }
  return [
    { id: 'one-step-affine-solid', steps: 1, config: { ...common, gravity: [0, 0, 0] }, initial: affine,
      description: 'Stress-free initial F; nonsymmetric affine velocity and C, no gravity. One APIC transfer should reproduce an affine field.' },
    { id: 'eight-step-gravity-solid', steps: 8, config: { ...common, gravity: [0, Math.fround(-9.81), 0] }, initial: cloud(),
      description: 'Interior stress-free block under gravity; no collision or boundary impulses.' },
    { id: 'eight-step-deformed-solid', steps: 8, config: { ...common, gravity: [0, Math.fround(-2), 0] }, initial: deformed,
      description: 'Nonsymmetric F and C test stress and matrix multiplication orientation over eight steps.' },
    { id: 'eight-step-pressure-fluid', steps: 8, config: { ...common, gravity: [0, Math.fround(-2), 0] }, initial: cloud(0, .25),
      description: 'Reference density .5, mass .125, volume .25; nonzero initial pressure exercises fluid stress, not just passive particles.' },
  ].map(fixture => ({ ...fixture, config: normalizeConfig(fixture.config), initial: float32State(fixture.initial) }));
}

function errorSummary(actual, expected, tolerance) {
  let maxAbs = 0, index = -1, squared = 0, mismatches = 0;
  if (actual.length !== expected.length) throw Error('Reference array length mismatch');
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) return { pass: false, nonfiniteIndex: i, tolerance };
    if (error > maxAbs) { maxAbs = error; index = i; }
    squared += error * error; if (error > tolerance) mismatches++;
  }
  return { pass: mismatches === 0, tolerance, maxAbs, rms: Math.sqrt(squared / actual.length), index,
    actualAtMaximum: index < 0 ? null : actual[index], expectedAtMaximum: index < 0 ? null : expected[index], mismatches, count: actual.length };
}

export function cpuReference(fixture) {
  let state = makeState(fixture.initial); const diagnostics = [];
  for (let i = 0; i < fixture.steps; i++) { const next = stepCPU(fixture.config, state); state = next.state; diagnostics.push(next.diagnostics); }
  const initialTotals = particleTotals(fixture.initial), finalTotals = particleTotals(state);
  const expectedMomentum = initialTotals.momentum.map((v, d) => v + initialTotals.mass * fixture.steps * fixture.config.dt * fixture.config.gravity[d]);
  const ledger = finalTotals.momentum.map((v, d) => v - expectedMomentum[d]);
  const gates = {
    particleCount: state.ids.length === 64,
    finite: FIELDS.every(key => Array.from(state[key]).every(Number.isFinite)),
    mass: finalTotals.mass === initialTotals.mass,
    noConstraintImpulse: diagnostics.every(d => magnitude(d.constraintImpulse) < 1e-12 && d.anchoredNodes === 0 && d.colliderNodes === 0),
    momentum: magnitude(ledger) < 1e-10,
    positiveSolidDeterminants: state.mode.every((mode, i) => !mode || determinant(state.F, i * 9) > .05),
    fluidPressureExercised: fixture.id !== 'eight-step-pressure-fluid' || diagnostics[0].maxPressure > 1,
  };
  if (fixture.id === 'one-step-affine-solid') {
    gates.affineVelocityReproduction = errorSummary(state.v, fixture.initial.v, 1e-7).pass;
    gates.affineCReproduction = errorSummary(state.C, fixture.initial.C, 1e-7).pass;
  }
  return { state, diagnostics, initialTotals, finalTotals, expectedMomentum, ledger, gates, pass: Object.values(gates).every(Boolean) };
}

async function readPackedParticles(device, simulator) {
  const bytes = simulator.count * 144;
  const buffer = device.createBuffer({ label: 'Independent particle metadata audit', size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(simulator.particleBuffer, 0, buffer, 0, bytes); device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    return new Float32Array(buffer.getMappedRange().slice(0));
  } finally { if (buffer.mapState === 'mapped') buffer.unmap(); buffer.destroy(); }
}

function auditPackedMetadata(packed, initial) {
  let actualMass = 0; const mismatches = [];
  for (let i = 0; i < initial.ids.length; i++) {
    const k = 36 * i; actualMass += packed[k + 7];
    for (const [name, offset, expected] of [['id', 34, initial.ids[i]], ['mode', 3, initial.mode[i]], ['mass', 7, initial.mass[i]], ['volume', 32, initial.volume[i]]]) {
      if (packed[k + offset] !== expected) mismatches.push({ particle: i, field: name, expected, actual: packed[k + offset] });
    }
  }
  return { pass: mismatches.length === 0, actualMass, expectedMass: particleTotals(initial).mass, mismatches,
    source: 'Direct GPU particleBuffer readback, independent of unpackState retained metadata', floatStride: 36 };
}

let started = false;
export async function runProbe() {
  if (started) throw Error('This page permits one bounded probe; reload for a separately recorded run.');
  started = true;
  const start = performance.now(), controller = new AbortController();
  const report = { kind: 'matter-forge-gpu-parity-v1', status: 'failed', limits: LIMITS, tolerances: TOLERANCES,
    parameterBoundary: 'CPU starts from exact Float32 upload values and Float32 dt/gravity. Float64 CPU arithmetic; Float32 GPU arithmetic and 1e-6 fixed-point grid scatter.',
    claimBoundary: 'Discrete solver agreement on four small cases. No continuum fidelity, long-run stability, speed, mixed-material contact, or gameplay claim.',
    cells: [], errors: [], shaderSources,
    lifecycle: { requestedDevices: 0, createdDevices: 0, simulatorAttempts: 0, createdSimulators: 0, disposedSimulators: 0, destroyedDevice: false, deviceLossReason: null, completedSteps: 0 } };
  if (typeof window !== 'undefined') window.matterForgeProbeResult = report;
  let device, lossPromise, scopeCount = 0, watchdogExpired = false;
  const stop = reason => { controller.abort(reason); device?.destroy(); };
  const watchdog = setTimeout(() => { watchdogExpired = true; stop('Probe wall-clock limit'); }, LIMITS.wallClockMs);
  const onVisibility = () => { if (document.hidden) stop('Page hidden'); };
  if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisibility);
  try {
    if (!navigator.gpu) throw Error('WebGPU unavailable');
    const fixtures = makeFixtures();
    // Precompute and validate references before requesting the device.
    const references = fixtures.map(cpuReference);
    if (references.some(r => !r.pass)) throw Error('A frozen CPU fixture failed its prerequisite gates');
    const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw Error('No WebGPU adapter');
    report.adapter = adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description } : null;
    report.lifecycle.requestedDevices++; device = await adapter.requestDevice({ label: 'Matter Forge one bounded parity probe' }); report.lifecycle.createdDevices++;
    lossPromise = device.lost.then(info => { report.lifecycle.deviceLossReason = info.reason; report.lifecycle.deviceLossMessage = info.message; });
    device.addEventListener('uncapturederror', event => report.errors.push({ type: event.error.constructor.name, message: event.error.message }));
    for (const filter of ['validation', 'out-of-memory', 'internal']) { device.pushErrorScope(filter); scopeCount++; }
    if (controller.signal.aborted) throw Error('Probe was cancelled before device initialization');
    for (let i = 0; i < fixtures.length; i++) {
      if (controller.signal.aborted) throw Error(String(controller.signal.reason));
      const fixture = fixtures[i], reference = references[i];
      const cell = { id: fixture.id, description: fixture.description, steps: fixture.steps, config: fixture.config,
        initial: serialState(fixture.initial), cpu: { final: serialState(reference.state), diagnostics: reference.diagnostics,
          initialTotals: reference.initialTotals, finalTotals: reference.finalTotals, expectedMomentum: reference.expectedMomentum, ledger: reference.ledger, gates: reference.gates }, status: 'failed' };
      report.cells.push(cell);
      let simulator;
      try {
        report.lifecycle.simulatorAttempts++;
        simulator = await createSimulator(device, fixture.config, fixture.initial); report.lifecycle.createdSimulators++;
        cell.compiledKernels = Object.keys(simulator.shaderSources);
        const actual = await simulator.advance(fixture.steps, { signal: controller.signal }); report.lifecycle.completedSteps += fixture.steps;
        const packed = await readPackedParticles(device, simulator);
        cell.gpu = { final: serialState(actual), packedFinal: Array.from(packed), totals: particleTotals(actual) };
        cell.metadata = auditPackedMetadata(packed, fixture.initial);
        cell.errors = Object.fromEntries(FIELDS.map(key => [key, errorSummary(actual[key], reference.state[key], TOLERANCES[key])]));
        cell.momentumError = cell.gpu.totals.momentum.map((v, d) => v - reference.expectedMomentum[d]);
        cell.minimumSolidDeterminant = actual.mode.some(mode => mode === 1) ? Math.min(...Array.from(actual.mode, (mode, p) => mode ? determinant(actual.F, 9 * p) : Infinity)) : null;
        cell.gates = {
          ...Object.fromEntries(FIELDS.map(key => [key, cell.errors[key].pass])), metadata: cell.metadata.pass,
          particleMass: cell.metadata.actualMass === reference.initialTotals.mass,
          momentum: magnitude(cell.momentumError) <= TOLERANCES.momentum,
          finite: Array.from(packed).every(Number.isFinite),
          solidDeterminants: cell.minimumSolidDeterminant === null || cell.minimumSolidDeterminant > .05,
          fluidFPreserved: fixture.id !== 'eight-step-pressure-fluid' || errorSummary(actual.F, fixture.initial.F, 0).pass,
          timeAndSteps: actual.steps === fixture.steps && actual.time === fixture.steps * fixture.config.dt,
        };
        cell.status = Object.values(cell.gates).every(Boolean) ? 'passed' : 'failed';
      } catch (error) { cell.failure = String(error.stack ?? error); throw error; }
      finally { if (simulator) { simulator.dispose(); report.lifecycle.disposedSimulators++; } }
      // Numerical failures remain recorded, but the other frozen fixtures still run.
    }
    if (report.cells.length !== 4 || report.cells.some(cell => cell.status !== 'passed')) throw Error('One or more fixed parity gates failed');
    report.status = 'passed';
  } catch (error) { report.failure = String(error.stack ?? error); }
  finally {
    clearTimeout(watchdog);
    if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', onVisibility);
    if (device) {
      while (scopeCount-- > 0) { try { const error = await device.popErrorScope(); if (error) report.errors.push({ type: error.constructor.name, message: error.message }); } catch (error) { report.errors.push({ type: 'ErrorScope', message: String(error.message) }); } }
      device.destroy(); report.lifecycle.destroyedDevice = true;
      if (lossPromise) await Promise.race([lossPromise, new Promise(resolve => setTimeout(resolve, 1000))]);
    }
    report.watchdogExpired = watchdogExpired;
    report.elapsedMs = performance.now() - start;
    report.lifecycle.pass = report.lifecycle.createdDevices === 1 && report.lifecycle.destroyedDevice && report.lifecycle.deviceLossReason === 'destroyed'
      && report.lifecycle.createdSimulators === 4 && report.lifecycle.disposedSimulators === 4 && report.lifecycle.completedSteps === 25;
    if (report.errors.length || watchdogExpired || !report.lifecycle.pass || controller.signal.aborted) report.status = 'failed';
  }
  return report;
}

if (typeof window !== 'undefined') window.matterForgeProbe = { runProbe };
