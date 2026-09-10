import test from 'node:test';
import assert from 'node:assert/strict';
import { buildKernels, evaluateKernels, prepareBasis, evaluateBasis, farthestCenters } from './basis.mjs';

function cloud(side = 6) {
  const p = [];
  for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) for (let k = 0; k < side; k++) {
    p.push(i / (side - 1) + 0.014 * Math.sin(7 * i + 3 * j + k),
      0.8 * j / (side - 1) + 0.01 * Math.cos(i + 7 * j + 2 * k),
      1.2 * k / (side - 1) + 0.011 * Math.sin(3 * i - j + 8 * k));
  }
  return new Float64Array(p);
}
const positions = cloud(), volumes = new Float64Array(positions.length / 3).fill(1 / (positions.length / 3));
const kernels = buildKernels({ positions, centerCount: 24 });
const query = new Float64Array([0.27, 0.31, 0.59, 0.9, 0.02, 1.1, -0.04, 0.71, 0.77]);
const near = (a, b, tolerance, message = '') => assert.ok(Math.abs(a - b) <= tolerance, `${message}: ${a} vs ${b} (tol ${tolerance})`);

test('corrected kernels reproduce constants, affine coordinates, and their exact gradients', () => {
  const { weights, gradients } = evaluateKernels(kernels, query), K = kernels.centerCount;
  for (let n = 0; n < query.length / 3; n++) {
    let sum = 0;
    const value = [0, 0, 0], gradSum = [0, 0, 0], affineGrad = new Float64Array(9);
    for (let j = 0; j < K; j++) {
      sum += weights[n * K + j];
      for (let a = 0; a < 3; a++) {
        value[a] += weights[n * K + j] * kernels.centers[j * 3 + a];
        gradSum[a] += gradients[(n * K + j) * 3 + a];
        for (let b = 0; b < 3; b++) affineGrad[a * 3 + b] += kernels.centers[j * 3 + a] * gradients[(n * K + j) * 3 + b];
      }
    }
    near(sum, 1, 2e-12, 'partition');
    for (let a = 0; a < 3; a++) {
      near(value[a], query[n * 3 + a], 2e-12, 'linear reproduction');
      near(gradSum[a], 0, 2e-12, 'constant gradient');
      for (let b = 0; b < 3; b++) near(affineGrad[a * 3 + b], a === b ? 1 : 0, 5e-12, 'linear gradient');
    }
  }
});

test('analytic RKPM derivatives agree with independent central differences off the sampling grid', () => {
  const exact = evaluateKernels(kernels, query), h = 1e-5, K = kernels.centerCount;
  for (let axis = 0; axis < 3; axis++) {
    const plus = query.slice(), minus = query.slice();
    for (let n = 0; n < query.length / 3; n++) { plus[n * 3 + axis] += h; minus[n * 3 + axis] -= h; }
    const a = evaluateKernels(kernels, plus), b = evaluateKernels(kernels, minus);
    for (let i = 0; i < a.weights.length; i++) near(exact.gradients[i * 3 + axis], (a.weights[i] - b.weights[i]) / (2 * h), 2e-9, `kernel ${i % K}`);
  }
});

test('prepared modes solve the generalized problem and are mass-orthonormal, with an explicit constant handle', () => {
  const basis = prepareBasis({ positions, volumes, centerCount: 24, modeCount: 6 });
  const { coefficients: c, massMatrix: M, stiffnessMatrix: S, modeCount: H, centerCount: K } = basis;
  assert.equal(H, 6); assert.equal(basis.constantModeIndex, 5); assert.equal(basis.eigenvalues.length, 5);
  for (let a = 0; a < H - 1; a++) {
    assert.ok(basis.residuals[a] < 1e-9);
    for (let b = 0; b < H - 1; b++) {
      let dot = 0;
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) dot += c[i * H + a] * M[i * K + j] * c[j * H + b];
      near(dot, a === b ? 1 : 0, 3e-8, 'mass orthogonality');
    }
    // Independently assemble the Rayleigh quotient, not the solver's residual.
    let energy = 0, mass = 0;
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      energy += c[i * H + a] * S[i * K + j] * c[j * H + a];
      mass += c[i * H + a] * M[i * K + j] * c[j * H + a];
    }
    near(energy / mass, basis.eigenvalues[a], 1e-5 * basis.eigenvalues[a], 'Rayleigh quotient');
  }
  const evaluated = evaluateBasis(basis, query);
  for (let n = 0; n < query.length / 3; n++) {
    assert.equal(evaluated.weights[n * H + H - 1], 1);
    assert.deepEqual(Array.from(evaluated.gradients.subarray((n * H + H - 1) * 3, (n * H + H) * 3)), [0, 0, 0]);
  }
});

test('basis analytic derivatives agree with finite differences after eigenmode projection', () => {
  const basis = prepareBasis({ positions, volumes, centerCount: 24, modeCount: 5 });
  const exact = evaluateBasis(basis, query), h = 1e-5;
  for (let axis = 0; axis < 3; axis++) {
    const plus = query.slice(), minus = query.slice();
    for (let n = 0; n < query.length / 3; n++) { plus[n * 3 + axis] += h; minus[n * 3 + axis] -= h; }
    const a = evaluateBasis(basis, plus, { includeGradients: false }), b = evaluateBasis(basis, minus, { includeGradients: false });
    assert.equal(a.gradients, null);
    for (let i = 0; i < a.weights.length; i++) near(exact.gradients[i * 3 + axis], (a.weights[i] - b.weights[i]) / (2 * h), 5e-8, 'projected derivative');
  }
});

test('isotropic physical scaling preserves weights and scales gradients and eigenvalues correctly', () => {
  const base = prepareBasis({ positions, volumes, centerCount: 20, modeCount: 5 });
  const factor = 3.2, offset = [13, -7, 101];
  const changed = prepareBasis({ positions: Float64Array.from(positions, (v, i) => v * factor + offset[i % 3]), volumes,
    centers: Float64Array.from(base.centers, (v, i) => v * factor + offset[i % 3]), modeCount: 5 });
  const a = evaluateBasis(base, query), b = evaluateBasis(changed, Float64Array.from(query, (v, i) => v * factor + offset[i % 3]));
  // Signs are arbitrary: determine each from the first nontrivial query value.
  for (let m = 0; m < base.modeCount - 1; m++) {
    near(changed.eigenvalues[m] * factor ** 2, base.eigenvalues[m], 2e-5, 'physical spectrum');
    const sign = Math.sign(a.weights[m] * b.weights[m]);
    for (let n = 0; n < query.length / 3; n++) {
      near(a.weights[n * base.modeCount + m], sign * b.weights[n * base.modeCount + m], 1e-7, 'scaled weight');
      for (let axis = 0; axis < 3; axis++) near(a.gradients[(n * base.modeCount + m) * 3 + axis], sign * factor * b.gradients[(n * base.modeCount + m) * 3 + axis], 2e-7, 'scaled gradient');
    }
  }
});

test('volume and stiffness scaling obey generalized-eigenvalue laws', () => {
  const options = { positions, volumes, centerCount: 20, modeCount: 4 };
  const a = prepareBasis(options), b = prepareBasis({ ...options, volumes: Float64Array.from(volumes, v => 7 * v) }),
    c = prepareBasis({ ...options, young: 3000 });
  for (let i = 0; i < a.eigenvalues.length; i++) {
    near(b.eigenvalues[i], a.eigenvalues[i], 2e-6, 'uniform volume cancels');
    near(c.eigenvalues[i], 3 * a.eigenvalues[i], 5e-6, 'stiffness linearity');
  }
});

test('48-center / 512-point canary is bounded and deterministic', () => {
  const p = cloud(8), v = new Float64Array(512).fill(1 / 512);
  const a = prepareBasis({ positions: p, volumes: v, centerCount: 48, modeCount: 6 });
  const b = prepareBasis({ positions: p, volumes: v, centers: a.centers, modeCount: 6 });
  assert.deepEqual(a.coefficients, b.coefficients);
  assert.ok(Math.max(...a.residuals) < 1e-8);
  assert.equal(a.diagnostics.noRegularization, true);
});

test('preparation and evaluation leave caller data and prepared arrays unchanged', () => {
  const p = positions.slice(), v = volumes.slice(), beforeP = p.slice(), beforeV = v.slice();
  const basis = prepareBasis({ positions: p, volumes: v, centerCount: 20, modeCount: 4 });
  const before = structuredClone(basis);
  evaluateBasis(basis, query); evaluateBasis(basis, query, { includeGradients: false });
  assert.deepEqual(p, beforeP); assert.deepEqual(v, beforeV); assert.deepEqual(basis, before);
});

test('invalid, degenerate, singular, and over-budget inputs fail closed', () => {
  const options = { positions, volumes, centerCount: 20, modeCount: 4 };
  assert.throws(() => prepareBasis({ ...options, modeCount: 13 }), /modeCount/);
  assert.throws(() => prepareBasis({ ...options, centerCount: 65 }), /centerCount/);
  assert.throws(() => prepareBasis({ ...options, young: Infinity }), /young/);
  assert.throws(() => prepareBasis({ ...options, poisson: 0.5 }), /poisson/);
  assert.throws(() => prepareBasis({ ...options, volumes: volumes.map((v, i) => i === 0 ? 0 : v) }), /positive/);
  assert.throws(() => prepareBasis({ ...options, positions: positions.map((v, i) => i === 0 ? NaN : v) }), /finite/);
  assert.throws(() => prepareBasis({ ...options, positions: positions.map((v, i) => i % 3 === 2 ? 0 : v) }), /three-dimensional/);
  const flatCenters = farthestCenters(positions, 20).map((v, i) => i % 3 === 2 ? 0 : v);
  assert.throws(() => prepareBasis({ ...options, centers: flatCenters }), /moment matrix/);
  assert.throws(() => prepareBasis({ positions: new Float64Array(2049 * 3), volumes: new Float64Array(2049) }), /bounded/);
  assert.throws(() => evaluateKernels(kernels, [100, 100, 100]), /bounded RKPM domain/);
});
