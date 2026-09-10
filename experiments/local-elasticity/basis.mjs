/**
 * Float64 CPU preparation of the RKPM skinning eigenmodes from FreeForm
 * (Xiang et al., CVPR 2026), https://research.nvidia.com/labs/sil/projects/freeform/.
 * Independent JavaScript implementation of the published method. The reference
 * equations and constant-handle convention were checked against NVIDIA Kaolin,
 * kaolin/physics/simplicits/rkpm.py, revision
 * d52da9f86d460e8abcd99037e21ff0bec57997ed (Apache-2.0, NVIDIA 2026).
 * No Kaolin Python code, pretrained model, or learned weights are executed.
 *
 * Domain: a finite, three-dimensional set of integration points with positive
 * quadrature volumes. A point cloud alone does not establish its occupied volume.
 * We use volume-weighted quadrature, an isotropic coordinate change for stable
 * moment solves, and physical-coordinate gradients. These choices are explicit;
 * this is not byte-for-byte parity with Kaolin's default bbox normalization.
 *
 * modeCount counts ALL handles, including the final exact constant handle.
 * weights are [point, handle], gradients [point, handle, xyz], coefficients
 * [kernel, handle], and all matrices are row-major Float64Array. Bounds fail
 * closed. Singular moment/mass matrices are rejected, never silently regularized.
 */

export const BASIS_LIMITS = Object.freeze({ preparationPoints: 2048, centers: 64, modes: 12, queryPoints: 150000 });

function invariant(test, message) { if (!test) throw new Error(`RKPM: ${message}`); }
function finiteArray(value, name, multiple = 1, maximum = Infinity) {
  invariant(value != null && Number.isInteger(value.length) && value.length > 0 && value.length % multiple === 0,
    `${name} must be a nonempty flat array with width ${multiple}`);
  invariant(value.length / multiple <= maximum, `${name} exceeds bounded point count ${maximum}`);
  for (let i = 0; i < value.length; i++) invariant(Number.isFinite(value[i]), `${name}[${i}] must be finite`);
  return Float64Array.from(value);
}
function finiteInteger(value, lo, hi, name) {
  invariant(Number.isInteger(value) && value >= lo && value <= hi, `${name} must be an integer in [${lo}, ${hi}]`);
}
function sqDistance(a, ai, b, bi) {
  const x = a[ai] - b[bi], y = a[ai + 1] - b[bi + 1], z = a[ai + 2] - b[bi + 2];
  return x * x + y * y + z * z;
}

function cholesky(a, n, name) {
  const l = new Float64Array(n * n);
  let scale = 0, minimumPivot = Infinity;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(a[i * n + i]));
  invariant(Number.isFinite(scale) && scale > 0, `${name} has no positive diagonal`);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = a[i * n + j];
    for (let k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
    if (i === j) {
      invariant(Number.isFinite(s) && s > scale * 1e-14, `${name} is singular or ill-conditioned at pivot ${i}`);
      minimumPivot = Math.min(minimumPivot, s / scale);
      l[i * n + j] = Math.sqrt(s);
    } else l[i * n + j] = s / l[j * n + j];
  }
  return { l, minimumPivot };
}
function solveLower(l, b, n) {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) s -= l[i * n + j] * x[j];
    x[i] = s / l[i * n + i];
  }
  return x;
}
function solveUpperTranspose(l, b, n) {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= l[j * n + i] * x[j];
    x[i] = s / l[i * n + i];
  }
  return x;
}
function solveSPD(l, b, n) { return solveUpperTranspose(l, solveLower(l, b, n), n); }
function frobenius(a) { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s); }

function symmetricEigen(aInput, n) {
  const a = aInput.slice(), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  const scale = frobenius(a);
  invariant(Number.isFinite(scale) && scale > 0, 'invalid whitened stiffness');
  let sweep = 0, off = Infinity;
  for (; sweep < 100; sweep++) {
    off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += 2 * a[p * n + q] ** 2;
    off = Math.sqrt(off);
    if (off <= scale * 2e-13) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) <= scale * 1e-16) continue;
      const tau = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.hypot(1, tau));
      const c = 1 / Math.hypot(1, t), s = t * c;
      a[p * n + p] -= t * apq;
      a[q * n + q] += t * apq;
      a[p * n + q] = a[q * n + p] = 0;
      for (let k = 0; k < n; k++) {
        if (k !== p && k !== q) {
          const kp = a[k * n + p], kq = a[k * n + q];
          a[k * n + p] = a[p * n + k] = c * kp - s * kq;
          a[k * n + q] = a[q * n + k] = s * kp + c * kq;
        }
        const vp = v[k * n + p], vq = v[k * n + q];
        v[k * n + p] = c * vp - s * vq;
        v[k * n + q] = s * vp + c * vq;
      }
    }
  }
  invariant(sweep < 100, 'symmetric eigenproblem did not converge in 100 sweeps');
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[i * n + i] - a[j * n + j]);
  return { values: Float64Array.from(order, i => a[i * n + i]), vectors: v, order, sweeps: sweep, offDiagonalRelative: off / scale };
}

/** Deterministic farthest-point selection, initialized by input point zero. */
export function farthestCenters(positions, count) {
  const p = finiteArray(positions, 'positions', 3, BASIS_LIMITS.preparationPoints), n = p.length / 3;
  finiteInteger(count, 4, Math.min(n, BASIS_LIMITS.centers), 'centerCount');
  const nearest = new Float64Array(n).fill(Infinity), result = new Float64Array(count * 3);
  let selected = 0;
  for (let k = 0; k < count; k++) {
    result.set(p.subarray(3 * selected, 3 * selected + 3), 3 * k);
    let furthest = -1, next = 0;
    for (let i = 0; i < n; i++) {
      nearest[i] = Math.min(nearest[i], sqDistance(p, 3 * i, p, 3 * selected));
      if (nearest[i] > furthest) { furthest = nearest[i]; next = i; }
    }
    invariant(k === count - 1 || furthest > 0, 'not enough distinct input points for centers');
    selected = next;
  }
  return result;
}

/** Build corrected Gaussian kernels; radii use the upstream second-neighbor / 3× mean-neighbor rule. */
export function buildKernels({ positions, centers, centerCount = 48, radii } = {}) {
  const p = finiteArray(positions, 'positions', 3, BASIS_LIMITS.preparationPoints);
  const c = centers == null ? farthestCenters(p, centerCount) : finiteArray(centers, 'centers', 3, BASIS_LIMITS.centers);
  const n = p.length / 3, k = c.length / 3;
  finiteInteger(k, 4, Math.min(n, BASIS_LIMITS.centers), 'centerCount');
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i++) { min[i % 3] = Math.min(min[i % 3], p[i]); max[i % 3] = Math.max(max[i % 3], p[i]); }
  const origin = Float64Array.from(min, (v, i) => (v + max[i]) / 2);
  const scale = Math.max(...max.map((v, i) => v - min[i]));
  invariant(Number.isFinite(scale) && scale > 1e-12, 'positions have zero or invalid extent');
  for (let d = 0; d < 3; d++) invariant(max[d] - min[d] > 1e-10 * scale, 'integration domain must be three-dimensional');
  const normalizedCenters = Float64Array.from(c, (v, i) => (v - origin[i % 3]) / scale);
  let r;
  if (radii != null) {
    r = finiteArray(radii, 'radii');
    invariant(r.length === k, 'radii count differs from centers');
    for (const value of r) invariant(value > 0, 'radii must be positive');
  } else {
    let meanNearest = 0;
    for (let i = 0; i < n; i++) {
      let nearest = Infinity;
      for (let j = 0; j < n; j++) if (i !== j) nearest = Math.min(nearest, sqDistance(p, i * 3, p, j * 3));
      meanNearest += Math.sqrt(nearest) / n;
    }
    const floor = 3 * meanNearest;
    r = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      let first = Infinity, second = Infinity;
      for (let j = 0; j < k; j++) if (i !== j) {
        const d = sqDistance(c, i * 3, c, j * 3);
        invariant(d > scale * scale * 1e-24, 'duplicate or numerically coincident centers');
        if (d < first) { second = first; first = d; } else if (d < second) second = d;
      }
      r[i] = Math.max(Math.sqrt(second), floor);
    }
  }
  const normalizedRadii = Float64Array.from(r, value => value / scale);
  return { kind: 'freeform-rkpm-kernels-v1', centers: c, radii: r, normalizedCenters, normalizedRadii, origin, scale, centerCount: k };
}

function kernelAt(kernels, worldPoint) {
  const k = kernels.centerCount, { normalizedCenters: centers, normalizedRadii: radii, scale, origin } = kernels;
  const x = new Float64Array([1, (worldPoint[0] - origin[0]) / scale, (worldPoint[1] - origin[1]) / scale, (worldPoint[2] - origin[2]) / scale]);
  invariant(Math.max(Math.abs(x[1]), Math.abs(x[2]), Math.abs(x[3])) <= 8, 'query outside bounded RKPM domain');
  const m = new Float64Array(16), dm = new Float64Array(48), g = new Float64Array(k), dg = new Float64Array(k * 3);
  for (let i = 0; i < k; i++) {
    const p = [1, centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]], rr = radii[i] ** 2;
    const d = [x[1] - p[1], x[2] - p[2], x[3] - p[3]];
    g[i] = Math.exp(-(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) / rr);
    for (let axis = 0; axis < 3; axis++) dg[i * 3 + axis] = -2 * d[axis] * g[i] / rr;
    for (let row = 0; row < 4; row++) for (let col = 0; col < 4; col++) {
      const pp = p[row] * p[col], offset = row * 4 + col;
      m[offset] += g[i] * pp;
      for (let axis = 0; axis < 3; axis++) dm[axis * 16 + offset] += dg[i * 3 + axis] * pp;
    }
  }
  const factor = cholesky(m, 4, 'RKPM moment matrix'), c = solveSPD(factor.l, x, 4), dc = new Float64Array(12);
  for (let axis = 0; axis < 3; axis++) {
    const rhs = new Float64Array(4); rhs[axis + 1] = 1;
    for (let row = 0; row < 4; row++) for (let col = 0; col < 4; col++) rhs[row] -= dm[axis * 16 + row * 4 + col] * c[col];
    dc.set(solveSPD(factor.l, rhs, 4), axis * 4);
  }
  const weights = new Float64Array(k), gradients = new Float64Array(k * 3);
  for (let i = 0; i < k; i++) {
    const p = [1, centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]];
    let pc = 0; for (let j = 0; j < 4; j++) pc += p[j] * c[j];
    weights[i] = g[i] * pc;
    for (let axis = 0; axis < 3; axis++) {
      let pdc = 0; for (let j = 0; j < 4; j++) pdc += p[j] * dc[axis * 4 + j];
      gradients[i * 3 + axis] = (dg[i * 3 + axis] * pc + g[i] * pdc) / scale;
    }
  }
  return { weights, gradients, minimumPivot: factor.minimumPivot };
}

/** Exposes the partition-of-unity RKPM kernels for independent numerical checks. */
export function evaluateKernels(kernels, positions) {
  const p = finiteArray(positions, 'positions', 3, BASIS_LIMITS.preparationPoints), n = p.length / 3, k = kernels.centerCount;
  const weights = new Float64Array(n * k), gradients = new Float64Array(n * k * 3);
  let minimumMomentPivot = Infinity;
  for (let i = 0; i < n; i++) {
    const sample = kernelAt(kernels, p.subarray(i * 3, i * 3 + 3));
    weights.set(sample.weights, i * k); gradients.set(sample.gradients, i * k * 3);
    minimumMomentPivot = Math.min(minimumMomentPivot, sample.minimumPivot);
  }
  return { weights, gradients, pointCount: n, centerCount: k, minimumMomentPivot };
}

function materialField(value, n, name, valid) {
  const result = typeof value === 'number' ? new Float64Array(n).fill(value) : finiteArray(value, name);
  invariant(result.length === n, `${name} must have one entry per integration point`);
  for (const x of result) invariant(Number.isFinite(x) && valid(x), `invalid ${name}`);
  return result;
}

/** Prepare a small basis on the CPU. No GPU, workers, I/O, or global state are created. */
export function prepareBasis({ positions, volumes, centers, centerCount = 48, modeCount = 6, young = 1000, poisson = 0.3, radii } = {}) {
  const p = finiteArray(positions, 'positions', 3, BASIS_LIMITS.preparationPoints), n = p.length / 3;
  const v = finiteArray(volumes, 'volumes');
  invariant(v.length === n, 'one positive volume is required per integration point');
  let totalVolume = 0;
  for (const x of v) { invariant(x > 0, 'volumes must be positive'); totalVolume += x; }
  invariant(Number.isFinite(totalVolume) && totalVolume > 0, 'invalid total volume');
  const kernels = buildKernels({ positions: p, centers, centerCount, radii }), k = kernels.centerCount;
  finiteInteger(modeCount, 2, Math.min(BASIS_LIMITS.modes, k - 1), 'modeCount');
  const e = materialField(young, n, 'young', x => x > 0), nu = materialField(poisson, n, 'poisson', x => x > -1 && x < 0.4999);
  const { weights: phi, gradients: dphi, minimumMomentPivot } = evaluateKernels(kernels, p);
  const mass = new Float64Array(k * k), stiffness = new Float64Array(k * k);
  for (let point = 0; point < n; point++) {
    const mu = e[point] / (2 * (1 + nu[point]));
    const lambda = e[point] * nu[point] / ((1 + nu[point]) * (1 - 2 * nu[point]));
    const coefficient = v[point] * (lambda + 4 * mu);
    for (let i = 0; i < k; i++) for (let j = 0; j <= i; j++) {
      const offset = i * k + j, a = (point * k + i) * 3, b = (point * k + j) * 3;
      mass[offset] += v[point] * phi[point * k + i] * phi[point * k + j];
      stiffness[offset] += coefficient * (dphi[a] * dphi[b] + dphi[a + 1] * dphi[b + 1] + dphi[a + 2] * dphi[b + 2]);
    }
  }
  for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) {
    mass[j * k + i] = mass[i * k + j]; stiffness[j * k + i] = stiffness[i * k + j];
  }
  const factor = cholesky(mass, k, 'RKPM mass matrix'), x = new Float64Array(k * k), whitened = new Float64Array(k * k);
  for (let col = 0; col < k; col++) {
    const rhs = Float64Array.from({ length: k }, (_, row) => stiffness[row * k + col]);
    const solution = solveLower(factor.l, rhs, k);
    for (let row = 0; row < k; row++) x[row * k + col] = solution[row];
  }
  for (let row = 0; row < k; row++) whitened.set(solveLower(factor.l, x.subarray(row * k, (row + 1) * k), k), row * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) {
    const a = (whitened[i * k + j] + whitened[j * k + i]) / 2;
    whitened[i * k + j] = whitened[j * k + i] = a;
  }
  const eigen = symmetricEigen(whitened, k), h = modeCount;
  invariant(Math.abs(eigen.values[0]) <= Math.max(1, Math.abs(eigen.values[1])) * 1e-6, 'constant zero eigenmode was not recovered');
  invariant(eigen.values[1] > 0, 'nonconstant stiffness modes must be positive');
  const coefficients = new Float64Array(k * h), eigenvalues = eigen.values.slice(1, h), residuals = new Float64Array(h - 1);
  const stiffnessNorm = frobenius(stiffness), massNorm = frobenius(mass);
  for (let mode = 0; mode < h - 1; mode++) {
    const col = eigen.order[mode + 1], q = Float64Array.from({ length: k }, (_, i) => eigen.vectors[i * k + col]);
    const w = solveUpperTranspose(factor.l, q, k);
    let norm = 0, signIndex = 0;
    for (let i = 0; i < k; i++) {
      let row = 0; for (let j = 0; j < k; j++) row += mass[i * k + j] * w[j];
      norm += w[i] * row;
      if (Math.abs(w[i]) > Math.abs(w[signIndex])) signIndex = i;
    }
    invariant(Number.isFinite(norm) && norm > 0, 'invalid eigenvector mass norm');
    const multiplier = (w[signIndex] < 0 ? -1 : 1) / Math.sqrt(norm);
    let residualSq = 0, normSq = 0;
    for (let i = 0; i < k; i++) { w[i] *= multiplier; coefficients[i * h + mode] = w[i]; normSq += w[i] ** 2; }
    for (let i = 0; i < k; i++) {
      let r = 0;
      for (let j = 0; j < k; j++) r += (stiffness[i * k + j] - eigenvalues[mode] * mass[i * k + j]) * w[j];
      residualSq += r * r;
    }
    residuals[mode] = Math.sqrt(residualSq) / ((stiffnessNorm + Math.abs(eigenvalues[mode]) * massNorm) * Math.sqrt(normSq));
    invariant(Number.isFinite(residuals[mode]) && residuals[mode] < 1e-7, `eigen residual exceeds gate for mode ${mode}`);
  }
  for (let i = 0; i < k; i++) coefficients[i * h + h - 1] = 1;
  return { ...kernels, kind: 'freeform-rkpm-basis-v1', modeCount: h, nonconstantModeCount: h - 1, constantModeIndex: h - 1,
    coefficients, eigenvalues, residuals, massMatrix: mass, stiffnessMatrix: stiffness, allEigenvalues: eigen.values,
    pointCount: n, totalVolume, diagnostics: { minimumMomentPivot, minimumMassPivot: factor.minimumPivot,
      eigenSweeps: eigen.sweeps, eigenOffDiagonalRelative: eigen.offDiagonalRelative, zeroEigenvalue: eigen.values[0],
      coordinateConvention: 'isotropic-moment-normalization; world-gradient volume quadrature', noRegularization: true } };
}

/** Evaluate the fixed prepared basis at arbitrary nearby points; callers may batch in a worker. */
export function evaluateBasis(basis, positions, { includeGradients = true } = {}) {
  invariant(basis?.kind === 'freeform-rkpm-basis-v1', 'expected a prepared basis');
  const p = finiteArray(positions, 'positions', 3, BASIS_LIMITS.queryPoints), n = p.length / 3, k = basis.centerCount, h = basis.modeCount;
  const weights = new Float64Array(n * h), gradients = includeGradients ? new Float64Array(n * h * 3) : null;
  for (let point = 0; point < n; point++) {
    const sample = kernelAt(basis, p.subarray(point * 3, point * 3 + 3));
    for (let mode = 0; mode < h - 1; mode++) for (let j = 0; j < k; j++) {
      const c = basis.coefficients[j * h + mode];
      weights[point * h + mode] += sample.weights[j] * c;
      if (gradients) for (let axis = 0; axis < 3; axis++) gradients[(point * h + mode) * 3 + axis] += sample.gradients[j * 3 + axis] * c;
    }
    weights[point * h + h - 1] = 1;
  }
  for (const value of weights) invariant(Number.isFinite(value), 'nonfinite evaluated basis');
  if (gradients) for (const value of gradients) invariant(Number.isFinite(value), 'nonfinite evaluated basis gradient');
  return { weights, gradients, pointCount: n, modeCount: h };
}
