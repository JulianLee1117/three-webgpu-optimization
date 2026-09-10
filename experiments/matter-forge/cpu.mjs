/** Small independent CPU APIC / MLS-MPM canary, grid spacing and cell volume 1.
 * Quadratic transfers follow Jiang et al. APIC (2015), Hu et al. MLS-MPM (2018):
 * https://doi.org/10.1145/2766996
 * https://yzhu.io/publication/mpmmls2018siggraph/paper.pdf . Equations implemented independently.
 * Shared grid velocity gives numerical material coupling, not separate-body
 * frictional contact. Node-clamped boxes are explicitly sticky grid obstacles.
 * No material calibration, thermal phase transition, or stability guarantee. */
const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const configs = new WeakSet(), workspaces = new WeakMap();
const finite = (x, name, minimum = -Infinity) => {
  if (!Number.isFinite(x) || x < minimum) throw new RangeError(`${name} must be finite and >=${minimum}`);
  return x;
};
const vector = (v, name) => { if (!v || v.length !== 3) throw new RangeError(`${name} needs3values`); return Array.from(v, (x, i) => finite(x, `${name}[${i}]`)); };
function boxes(items, name) { return (items ?? []).map((item, i) => {
  if (typeof item.name !== 'string' || !item.name.trim()) throw new RangeError(`${name}[${i}] needs explicit name`);
  const min = vector(item.min, 'min'), max = vector(item.max, 'max');
  if (min.some((x, d) => max[d] <= x)) throw new RangeError('box extent must be positive');
  return { name: item.name, min, max };
}); }
export function normalizeConfig({ N = 24, dt = 1 / 240, gravity = [0, -9.81, 0], bulk = 120, gamma = 5,
  mu = 80, lambda = 120, boundary = 2, colliders = [], anchors = [], minimumJacobian = 1e-8 } = {}) {
  if (!Number.isInteger(N) || N < 8 || N > 64) throw new RangeError('N must be8..64');
  finite(dt, 'dt', Number.MIN_VALUE); if (dt > 1 / 60) throw new RangeError('dt must be<=1/60');
  if (!Number.isInteger(boundary) || boundary < 2 || boundary >= N / 2) throw new RangeError('invalid boundary width');
  const c = { N, dt, gravity: vector(gravity, 'gravity'), bulk: finite(bulk, 'bulk', 0), gamma: finite(gamma, 'gamma', 1),
    mu: finite(mu, 'mu', 0), lambda: finite(lambda, 'lambda', 0), boundary,
    colliders: boxes(colliders, 'colliders'), anchors: boxes(anchors, 'anchors'), minimumJacobian: finite(minimumJacobian, 'minimumJacobian', Number.MIN_VALUE) };
  configs.add(c); return Object.freeze(c);
}
function values(input, n, fill, name, Type = Float64Array) {
  if (input != null && input.length !== n) throw new RangeError(`${name} length must be${n}`);
  const out = input == null ? new Type(n).fill(fill) : new Type(input);
  for (const x of out) finite(x, name); return out;
}
export function determinant(a, offset = 0) {
  const o = offset;
  return a[o] * (a[o + 4] * a[o + 8] - a[o + 5] * a[o + 7])
    - a[o + 1] * (a[o + 3] * a[o + 8] - a[o + 5] * a[o + 6])
    + a[o + 2] * (a[o + 3] * a[o + 7] - a[o + 4] * a[o + 6]);
}
export function makeState({ x, v, C, F, mode, mass, volume, ids, time = 0, steps = 0, materialEdits = [] }) {
  if (!x || x.length % 3) throw new RangeError('x must contain triples');
  const count = x.length / 3;
  const state = { ids: ids == null ? Uint32Array.from({ length: count }, (_, i) => i) : values(ids, count, 0, 'ids', Uint32Array),
    x: values(x, 3 * count, 0, 'x'), v: values(v, 3 * count, 0, 'v'), C: values(C, 9 * count, 0, 'C'),
    F: values(F, 9 * count, 0, 'F'), mode: values(mode, count, 0, 'mode', Uint8Array),
    mass: values(mass, count, .125, 'mass'), volume: values(volume, count, .125, 'volume'),
    time: finite(time, 'time', 0), steps: finite(steps, 'steps', 0), materialEdits: structuredClone(materialEdits) };
  if (mode && Array.from(mode).some(m => m !== 0 && m !== 1)) throw new RangeError('mode must be0fluid/1solid');
  if (new Set(state.ids).size !== count) throw new RangeError('particle ids must be unique');
  for (let p = 0; p < count; p++) {
    if (!(state.mass[p] > 0) || !(state.volume[p] > 0)) throw new RangeError('mass and reference volume must be positive');
    if (F == null) state.F.set(I, 9 * p);
    if (state.mode[p] && determinant(state.F, 9 * p) <= 0) throw new RangeError('solid F must have positive determinant');
  }
  return state;
}
export function regularParticles({ min, max, spacing = .5, mode = 1, mass = .125, volume = .125 }) {
  min = vector(min, 'min'); max = vector(max, 'max'); finite(spacing, 'spacing', Number.MIN_VALUE);
  if (min.some((x, i) => max[i] <= x)) throw new RangeError('invalid sampling box');
  const x = [];
  for (let z = min[2] + spacing / 2; z < max[2] - 1e-10; z += spacing)
    for (let y = min[1] + spacing / 2; y < max[1] - 1e-10; y += spacing)
      for (let a = min[0] + spacing / 2; a < max[0] - 1e-10; a += spacing) x.push(a, y, z);
  return makeState({ x, mode: new Uint8Array(x.length / 3).fill(mode), mass: new Float64Array(x.length / 3).fill(mass), volume: new Float64Array(x.length / 3).fill(volume) });
}
export function mergeStates(states) {
  const output = {};
  for (const key of ['x', 'v', 'C', 'F', 'mode', 'mass', 'volume']) output[key] = states.flatMap(s => Array.from(s[key]));
  return makeState(output);
}
export function transmute(input, selection, mode) {
  if (mode !== 0 && mode !== 1) throw new RangeError('mode must be0fluid/1solid');
  const state = makeState(input), indices = selection == null ? Array.from({ length: state.ids.length }, (_, i) => i) : [...selection];
  if (new Set(indices).size !== indices.length || indices.some(i => !Number.isInteger(i) || i < 0 || i >= state.ids.length)) throw new RangeError('invalid particle selection');
  for (const p of indices) { state.mode[p] = mode; if (mode) state.F.set(I, p * 9); }
  state.materialEdits.push({ step: state.steps, time: state.time, ids: indices.map(i => state.ids[i]), mode,
    newStressFreeReference: mode === 1, interpretation: 'External material edit; position, mass and momentum retained. Reference reset changes stored elastic energy.' });
  return state;
}
export function particleTotals(state) {
  let mass = 0, kinetic = 0; const momentum = [0, 0, 0], center = [0, 0, 0];
  for (let p = 0; p < state.ids.length; p++) {
    mass += state.mass[p];
    for (let d = 0; d < 3; d++) { momentum[d] += state.mass[p] * state.v[3 * p + d]; center[d] += state.mass[p] * state.x[3 * p + d]; kinetic += .5 * state.mass[p] * state.v[3 * p + d] ** 2; }
  }
  return { mass, momentum, kinetic, center: center.map(x => mass ? x / mass : 0) };
}
function workspace(config, count) {
  let w = workspaces.get(config);
  if (!w || w.count !== count) {
    const nodes = config.N ** 3, taps = count * 27;
    w = { count, mass: new Float64Array(nodes), solidMass: new Float64Array(nodes), momentum: new Float64Array(3 * nodes), velocity: new Float64Array(3 * nodes),
      indices: new Uint32Array(taps), weights: new Float64Array(taps), offsets: new Float64Array(3 * taps), density: new Float64Array(count) };
    workspaces.set(config, w);
  }
  w.mass.fill(0); w.solidMass.fill(0); w.momentum.fill(0); w.velocity.fill(0); return w;
}
function inside(x, y, z, b) { return x >= b.min[0] && x <= b.max[0] && y >= b.min[1] && y <= b.max[1] && z >= b.min[2] && z <= b.max[2]; }
function failure(message, particleId, stage, value) { const error = new Error(message); error.details = { particleId, stage, value }; throw error; }
export function stepCPU(inputConfig, state, { retainGrid = false } = {}) {
  const c = configs.has(inputConfig) ? inputConfig : normalizeConfig(inputConfig), { N, dt } = c, count = state.ids.length;
  const w = workspace(c, count), before = particleTotals(state), wx = [], wy = [], wz = [];
  for (let p = 0; p < count; p++) {
    const p3 = 3 * p, p9 = 9 * p, x = state.x[p3], y = state.x[p3 + 1], z = state.x[p3 + 2];
    const bx = Math.floor(x - .5), by = Math.floor(y - .5), bz = Math.floor(z - .5);
    if (![x, y, z].every(Number.isFinite) || bx < 0 || by < 0 || bz < 0 || bx + 2 >= N || by + 2 >= N || bz + 2 >= N) failure('particle stencil outside declared grid', state.ids[p], 'p2g', [x, y, z]);
    const f = [x - bx, y - by, z - bz];
    for (let d = 0; d < 3; d++) { const a = [wx, wy, wz][d]; a[0] = .5 * (1.5 - f[d]) ** 2; a[1] = .75 - (f[d] - 1) ** 2; a[2] = .5 * (f[d] - .5) ** 2; }
    let tap = p * 27;
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++, tap++) {
      const node = bx + i + N * (by + j + N * (bz + k)), weight = wx[i] * wy[j] * wz[k], dx = bx + i - x, dy = by + j - y, dz = bz + k - z;
      w.indices[tap] = node; w.weights[tap] = weight; w.offsets[3 * tap] = dx; w.offsets[3 * tap + 1] = dy; w.offsets[3 * tap + 2] = dz;
      const m = weight * state.mass[p]; w.mass[node] += m; if (state.mode[p]) w.solidMass[node] += m;
      for (let d = 0; d < 3; d++) w.momentum[3 * node + d] += m * (state.v[p3 + d] + state.C[p9 + 3 * d] * dx + state.C[p9 + 3 * d + 1] * dy + state.C[p9 + 3 * d + 2] * dz);
    }
  }
  let gridMass = 0; const transferMomentum = [0, 0, 0];
  for (let n = 0; n < w.mass.length; n++) { gridMass += w.mass[n]; for (let d = 0; d < 3; d++) transferMomentum[d] += w.momentum[3 * n + d]; }
  let minJ = Infinity, maxPressure = 0, minDensity = Infinity, maxDensity = 0;
  const stress = new Float64Array(9);
  for (let p = 0; p < count; p++) {
    const p9 = 9 * p;
    let density = 0; for (let tap = p * 27; tap < (p + 1) * 27; tap++) density += w.weights[tap] * w.mass[w.indices[tap]];
    w.density[p] = density; minDensity = Math.min(minDensity, density); maxDensity = Math.max(maxDensity, density);
    stress.fill(0); let stressVolume;
    if (state.mode[p]) {
      const J = determinant(state.F, p9);
      if (!Number.isFinite(J) || J <= c.minimumJacobian) failure('invalid solid deformation determinant', state.ids[p], 'stress', J);
      minJ = Math.min(minJ, J); stressVolume = state.volume[p];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        let fft = 0; for (let k = 0; k < 3; k++) fft += state.F[p9 + 3 * a + k] * state.F[p9 + 3 * b + k];
        stress[3 * a + b] = c.mu * (fft - (a === b ? 1 : 0)) + (a === b ? c.lambda * Math.log(J) : 0);
      }
    } else {
      if (!(density > 0) || !Number.isFinite(density)) failure('invalid interpolated grid density', state.ids[p], 'density', density);
      const restDensity = state.mass[p] / state.volume[p], pressure = c.bulk * Math.max((density / restDensity) ** c.gamma - 1, 0);
      if (!Number.isFinite(pressure)) failure('nonfinite fluid pressure', state.ids[p], 'pressure', pressure);
      stress[0] = stress[4] = stress[8] = -pressure; stressVolume = state.mass[p] / density; maxPressure = Math.max(maxPressure, pressure);
    }
    for (let tap = p * 27; tap < (p + 1) * 27; tap++) {
      const n3 = 3 * w.indices[tap], t3 = 3 * tap, factor = -4 * dt * stressVolume * w.weights[tap];
      for (let d = 0; d < 3; d++) w.momentum[n3 + d] += factor * (stress[3 * d] * w.offsets[t3] + stress[3 * d + 1] * w.offsets[t3 + 1] + stress[3 * d + 2] * w.offsets[t3 + 2]);
    }
  }
  const stressMomentum = [0, 0, 0], constraintImpulse = [0, 0, 0], gravityImpulse = [0, 0, 0];
  let anchoredNodes = 0, colliderNodes = 0;
  for (let n = 0; n < w.mass.length; n++) {
    if (!w.mass[n]) continue;
    const xyz = [n % N, Math.floor(n / N) % N, Math.floor(n / (N * N))], m = w.mass[n];
    const collider = c.colliders.some(b => inside(...xyz, b));
    const anchored = w.solidMass[n] > 0 && c.anchors.some(b => inside(...xyz, b));
    if (collider) colliderNodes++; if (anchored) anchoredNodes++;
    for (let d = 0; d < 3; d++) {
      const index = 3 * n + d, accelerated = w.momentum[index] / m + dt * c.gravity[d];
      let velocity = accelerated;
      if (collider || anchored || (xyz[d] < c.boundary && velocity < 0) || (xyz[d] >= N - c.boundary && velocity > 0)) velocity = 0;
      w.velocity[index] = velocity; stressMomentum[d] += w.momentum[index]; gravityImpulse[d] += m * dt * c.gravity[d]; constraintImpulse[d] += m * (velocity - accelerated);
    }
  }
  const next = { ...state, x: new Float64Array(3 * count), v: new Float64Array(3 * count), C: new Float64Array(9 * count),
    F: new Float64Array(state.F), time: state.time + dt, steps: state.steps + 1 };
  let maxSpeed = 0;
  for (let p = 0; p < count; p++) {
    const p3 = 3 * p, p9 = 9 * p;
    for (let tap = p * 27; tap < (p + 1) * 27; tap++) {
      const n3 = 3 * w.indices[tap], t3 = 3 * tap, weight = w.weights[tap];
      for (let d = 0; d < 3; d++) {
        const velocity = w.velocity[n3 + d]; next.v[p3 + d] += weight * velocity;
        for (let a = 0; a < 3; a++) next.C[p9 + 3 * d + a] += 4 * weight * velocity * w.offsets[t3 + a];
      }
    }
    for (let d = 0; d < 3; d++) {
      next.x[p3 + d] = state.x[p3 + d] + dt * next.v[p3 + d];
      if (!Number.isFinite(next.x[p3 + d]) || !Number.isFinite(next.v[p3 + d])) failure('nonfinite particle state', state.ids[p], 'g2p', next.x[p3 + d]);
    }
    maxSpeed = Math.max(maxSpeed, Math.hypot(next.v[p3], next.v[p3 + 1], next.v[p3 + 2]));
    if (state.mode[p]) {
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        let result = state.F[p9 + 3 * a + b];
        for (let k = 0; k < 3; k++) result += dt * next.C[p9 + 3 * a + k] * state.F[p9 + 3 * k + b];
        next.F[p9 + 3 * a + b] = result;
      }
      const J = determinant(next.F, p9);
      if (!Number.isFinite(J) || J <= c.minimumJacobian) failure('invalid updated solid determinant', state.ids[p], 'F-update', J);
      minJ = Math.min(minJ, J);
    }
  }
  const after = particleTotals(next);
  const diagnostics = { particles: count, particleMass: before.mass, gridMass, massError: gridMass - before.mass, transferMomentum,
    transferMomentumError: transferMomentum.map((x, d) => x - before.momentum[d]),
    internalStressImpulse: stressMomentum.map((x, d) => x - transferMomentum[d]), gravityImpulse, constraintImpulse,
    momentumLedgerError: after.momentum.map((x, d) => x - stressMomentum[d] - gravityImpulse[d] - constraintImpulse[d]),
    minJ: minJ === Infinity ? null : minJ, minDensity: minDensity === Infinity ? null : minDensity, maxDensity,
    maxPressure, maxSpeed, anchoredNodes, colliderNodes, kinetic: after.kinetic };
  return { state: next, diagnostics, ...(retainGrid ? { grid: { mass: new Float64Array(w.mass), solidMass: new Float64Array(w.solidMass),
    momentum: new Float64Array(w.momentum), velocity: new Float64Array(w.velocity), density: new Float64Array(w.density) } } : {}) };
}
