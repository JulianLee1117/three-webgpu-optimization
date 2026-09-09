/**
 * Bounded Levenberg-Marquardt / Gauss-Newton driver. Established numerical
 * methods; this module makes no claim of a new optimizer.
 *
 * Contract: exactly eight normalized [0, 1] parameters. evaluate(x) renders one
 * image and resolves to finite scalar MSE. captureEquations(center, steps)
 * renders exactly sixteen images (eight clamped +/- probes) and returns:
 *   { hessian: row-major J^T J / N (64 numbers), gradient: J^T r / N (8) }.
 * Multiplying BOTH equations by two is also valid. Here r = render - reference.
 * J[:,i] uses the ACTUAL denominator min(1,c[i]+h[i])-max(0,c[i]-h[i]).
 * The callback reuses the already-rendered center; a seventeenth center render
 * would violate this contract. No probe loss is exposed to this driver.
 *
 * Successful calls consume exactly budget renders, 1 <= budget <= 96. Equation
 * captures cost 16, not 1. history/onProgress are DECISION records: renderCost
 * and firstEvaluation..evaluation give their actual rendered-evaluation spans.
 * No GPU/device work occurs in this module; all eight-by-eight solves use JS.
 */

const DIMENSION = 8;
const CAPTURE_COST = 16;
const clamp = (value) => Math.max(0, Math.min(1, value));
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const norm = (a) => Math.hypot(...a);

function finiteVector(value, length, name) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length) {
    throw new TypeError(`${name} must contain ${length} numbers.`);
  }
  return Array.from(value, (item) => {
    if (typeof item !== 'number' || !Number.isFinite(item)) {
      throw new TypeError(`${name} must contain finite numbers.`);
    }
    return item;
  });
}

const multiply = (matrix, vector) => vector.map((_, row) =>
  vector.reduce((sum, value, col) => sum + matrix[row * DIMENSION + col] * value, 0));

// Cholesky factorization of a small symmetric positive-definite system.
// Null means damping must increase; there is no unchecked division by a pivot.
function choleskySolve(matrix, rhs) {
  const n = rhs.length;
  const factor = Array(n * n).fill(0);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col <= row; col++) {
      let value = matrix[row * n + col];
      for (let k = 0; k < col; k++) value -= factor[row * n + k] * factor[col * n + k];
      if (row === col) {
        if (!Number.isFinite(value) || value <= 1e-14) return null;
        factor[row * n + col] = Math.sqrt(value);
      } else factor[row * n + col] = value / factor[col * n + col];
    }
  }
  const y = Array(n).fill(0);
  for (let row = 0; row < n; row++) {
    let value = rhs[row];
    for (let col = 0; col < row; col++) value -= factor[row * n + col] * y[col];
    y[row] = value / factor[row * n + row];
  }
  const result = Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let value = y[row];
    for (let col = row + 1; col < n; col++) value -= factor[col * n + row] * result[col];
    result[row] = value / factor[row * n + row];
  }
  return result.every(Number.isFinite) ? result : null;
}

function dampedDirection(hessian, gradient, center, damping) {
  const scale = Math.max(1e-12, ...hessian.map(Math.abs));
  const active = center.map((x, i) =>
    (x <= 1e-10 && gradient[i] > 0) || (x >= 1 - 1e-10 && gradient[i] < 0));
  const matrix = hessian.map((value) => value / scale);
  const rhs = gradient.map((value) => -value / scale);
  for (let row = 0; row < DIMENSION; row++) {
    if (active[row]) {
      for (let col = 0; col < DIMENSION; col++) {
        matrix[row * DIMENSION + col] = 0;
        matrix[col * DIMENSION + row] = 0;
      }
      matrix[row * DIMENSION + row] = 1;
      rhs[row] = 0;
    } else {
      const index = row * DIMENSION + row;
      matrix[index] += damping * Math.max(Math.abs(matrix[index]), 1e-4) + 1e-8;
    }
  }
  const direction = choleskySolve(matrix, rhs);
  if (!direction) return null;
  const magnitude = norm(direction);
  const factor = magnitude > 0.35 ? 0.35 / magnitude : 1;
  return direction.map((value) => value * factor);
}

export async function optimizeLeastSquares({ initial, budget = 96, evaluate, captureEquations, onProgress } = {}) {
  const start = finiteVector(initial, DIMENSION, 'initial').map(clamp);
  if (!Number.isInteger(budget) || budget < 1 || budget > 96) {
    throw new RangeError('budget must be an integer from 1 through 96.');
  }
  if (typeof evaluate !== 'function' || typeof captureEquations !== 'function') {
    throw new TypeError('evaluate and captureEquations must be functions.');
  }
  if (onProgress !== undefined && typeof onProgress !== 'function') {
    throw new TypeError('onProgress must be a function when provided.');
  }
  let evaluations = 0;
  let best = start.slice();
  let bestLoss = Infinity;
  let equations = null;
  let damping = 0.03;
  let finiteDifferenceStep = 0.06;
  let failedTrials = 0;
  let fallbackIndex = 0;
  const history = [];
  const remaining = () => budget - evaluations;

  async function record(kind, renderCost, extra) {
    const entry = { kind, firstEvaluation: evaluations - renderCost + 1,
      evaluation: evaluations, renderCost, ...extra, best: best.slice(), bestLoss };
    history.push(entry);
    if (onProgress) await onProgress(structuredClone(entry));
  }

  async function trial(point, kind, extra = {}) {
    if (remaining() < 1) throw new Error('Internal error: budget exhausted.');
    const parameters = finiteVector(point, DIMENSION, 'trial').map(clamp);
    evaluations++;
    const loss = await evaluate(parameters.slice());
    if (typeof loss !== 'number' || !Number.isFinite(loss)) {
      throw new TypeError('evaluate must resolve to a finite numeric loss.');
    }
    const accepted = loss < bestLoss;
    if (accepted) { best = parameters.slice(); bestLoss = loss; }
    await record(kind, 1, { ...extra, parameters, loss, accepted });
    return { accepted, loss };
  }

  await trial(start, 'initial');
  while (remaining()) {
    // Reserve one further render to validate the normal-equation proposal.
    if (!equations && remaining() >= CAPTURE_COST + 1) {
      const center = best.slice();
      const steps = Array(DIMENSION).fill(finiteDifferenceStep);
      evaluations += CAPTURE_COST;
      const result = await captureEquations(center.slice(), steps.slice());
      const rawHessian = finiteVector(result?.hessian, DIMENSION ** 2, 'hessian');
      const gradient = finiteVector(result?.gradient, DIMENSION, 'gradient');
      // Round-off in independent GPU reductions can make H slightly asymmetric.
      const hessian = rawHessian.map((value, index) => {
        const row = Math.floor(index / DIMENSION), col = index % DIMENSION;
        return value / 2 + rawHessian[col * DIMENSION + row] / 2;
      });
      equations = { center, hessian, gradient };
      failedTrials = 0;
      await record('equations', CAPTURE_COST, { parameters: center, steps, damping });
    }

    let direction = null;
    let localGradient = null;
    if (equations) {
      // If insufficient budget remains to recapture at an accepted point, keep
      // the same local quadratic and update its gradient by H*(x-x_capture).
      const offset = best.map((value, i) => value - equations.center[i]);
      const shift = multiply(equations.hessian, offset);
      localGradient = equations.gradient.map((value, i) => value + shift[i]);
      for (let attempt = 0; attempt < 8 && !direction; attempt++) {
        direction = dampedDirection(equations.hessian, localGradient, best, damping);
        if (!direction) damping = Math.min(1e10, damping * 10);
      }
    }

    const proposed = direction?.map((value, i) => clamp(best[i] + value));
    const actualStep = proposed?.map((value, i) => value - best[i]);
    if (actualStep && norm(actualStep) > 1e-8) {
      const previousLoss = bestLoss;
      const predicted = -2 * dot(localGradient, actualStep)
        - dot(actualStep, multiply(equations.hessian, actualStep));
      const result = await trial(proposed, 'trial', { damping, stepNorm: norm(actualStep), predictedReduction: predicted });
      if (result.accepted) {
        const ratio = predicted > 1e-15 ? (previousLoss - result.loss) / predicted : 1;
        damping = Math.max(1e-6, damping * (ratio > 0.5 ? 0.35 : 0.7));
        finiteDifferenceStep = Math.max(0.02, finiteDifferenceStep * 0.7);
        failedTrials = 0;
        if (remaining() >= CAPTURE_COST + 1) equations = null;
      } else {
        damping = Math.min(1e10, damping * 4);
        failedTrials++;
      }
    } else {
      // With too little budget for equations, a singular/flat local model, or
      // all descent directions pinned at bounds, use bounded coordinate polls.
      // This also fills an equal-budget run without treating probes as free.
      const axis = Math.floor(fallbackIndex / 2) % DIMENSION;
      const sign = fallbackIndex % 2 === 0 ? -1 : 1;
      const level = Math.floor(fallbackIndex / (2 * DIMENSION));
      const step = Math.max(0.002, finiteDifferenceStep / 2 ** level);
      const point = best.slice();
      point[axis] = clamp(point[axis] + sign * step);
      if (point[axis] === best[axis]) point[axis] = clamp(point[axis] - sign * step);
      fallbackIndex++;
      const result = await trial(point, 'fallback', { axis, step, damping });
      if (result.accepted) {
        failedTrials = 0;
        if (remaining() >= CAPTURE_COST + 1) equations = null;
      } else failedTrials++;
    }
    if (equations && failedTrials >= 5 && remaining() >= CAPTURE_COST + 1) {
      // Refresh only after a successful move or several rejected trials.
      equations = null;
      finiteDifferenceStep = Math.max(0.02, finiteDifferenceStep * 0.7);
      damping = Math.max(0.03, Math.min(damping, 1));
    }
  }
  return { best: best.slice(), bestLoss, evaluations, history };
}

async function selfTest() {
  const { default: assert } = await import('node:assert/strict');
  function oracle(residuals) {
    let renders = 0;
    let incumbent = null;
    let incumbentLoss = Infinity;
    const residualImage = (x) => {
      assert.ok(x.every((value) => Number.isFinite(value) && value >= 0 && value <= 1));
      renders++;
      return residuals(x);
    };
    const loss = (r) => dot(r, r) / r.length;
    return { get renders() { return renders; }, evaluate: async (x) => {
      const value = loss(residualImage(x));
      if (value < incumbentLoss) { incumbentLoss = value; incumbent = x.slice(); }
      return value;
    },
      captureEquations: async (center, steps) => {
        assert.deepEqual(center, incumbent, 'Normal equations must use the already-rendered global best.');
        const r = residuals(center); // Reused center, no new rendered evaluation.
        const columns = [];
        for (let axis = 0; axis < DIMENSION; axis++) {
          const lo = center.slice(), hi = center.slice();
          lo[axis] = clamp(lo[axis] - steps[axis]);
          hi[axis] = clamp(hi[axis] + steps[axis]);
          const a = residualImage(lo), b = residualImage(hi);
          columns.push(a.map((_, i) => (b[i] - a[i]) / (hi[axis] - lo[axis])));
        }
        return { hessian: columns.flatMap((a) => columns.map((b) => dot(a, b) / r.length)),
          gradient: columns.map((column) => dot(column, r) / r.length) };
      } };
  }
  const target = [0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.45, 0.55];
  const initial = [0.8, 0.15, 0.6, 0.2, 0.8, 0.25, 0.8, 0.2];
  const linear = (x) => x.map((v, i) => v - target[i]);
  const coupled = (x) => {
    const r = linear(x);
    return [...r, ...r.slice(1).map((value, i) => 2 * value + r[i])];
  };
  const nonlinear = (x) => x.map((v, i) => Math.sin(v) - Math.sin(target[i]));
  const boundaryTarget = [-0.2, 1.2, ...target.slice(2)];
  const boundary = (x) => x.map((v, i) => v - boundaryTarget[i]);
  const summaries = [];
  for (const [name, residuals] of [['linear', linear], ['coupled', coupled], ['nonlinear', nonlinear], ['boundary', boundary]]) {
    const source = oracle(residuals);
    const observations = [];
    const result = await optimizeLeastSquares({ initial, evaluate: source.evaluate,
      captureEquations: source.captureEquations, onProgress: (entry) => { observations.push(entry); entry.best.fill(99); } });
    assert.equal(result.evaluations, 96);
    assert.equal(source.renders, 96);
    assert.equal(result.history.reduce((sum, entry) => sum + entry.renderCost, 0), 96);
    assert.equal(result.history.length, observations.length);
    assert.ok(result.history.every((entry, i, rows) => i === 0 || entry.bestLoss <= rows[i - 1].bestLoss));
    assert.ok(result.best.every((v) => v >= 0 && v <= 1));
    assert.ok(result.bestLoss < (name === 'boundary' ? 0.01001 : 1e-5), `${name}: ${result.bestLoss}`);
    summaries.push({ objective: name, loss: result.bestLoss, renders: source.renders });
  }
  for (const budget of [1, 16, 17, 18, 33, 96]) {
    const source = oracle(linear);
    const result = await optimizeLeastSquares({ initial, budget, evaluate: source.evaluate, captureEquations: source.captureEquations });
    assert.equal(source.renders, budget);
    assert.equal(result.evaluations, budget);
  }
  const zero = oracle(() => [0]);
  const flat = await optimizeLeastSquares({ initial, evaluate: zero.evaluate, captureEquations: zero.captureEquations });
  assert.equal(flat.bestLoss, 0);
  assert.equal(zero.renders, 96);
  const invalid = { initial, evaluate: async () => 1, captureEquations: async () => ({ hessian: Array(64).fill(0), gradient: Array(8).fill(0) }) };
  await assert.rejects(optimizeLeastSquares({ ...invalid, budget: 97 }), /budget/);
  await assert.rejects(optimizeLeastSquares({ ...invalid, initial: [1] }), /8 numbers/);
  await assert.rejects(optimizeLeastSquares({ ...invalid, evaluate: async () => NaN }), /finite numeric loss/);
  await assert.rejects(optimizeLeastSquares({ ...invalid, captureEquations: async () => ({ hessian: Array(64).fill(Infinity), gradient: Array(8).fill(0) }) }), /finite numbers/);
  console.log(JSON.stringify({ passed: true, summaries }, null, 2));
}

if (typeof process !== 'undefined' && process.argv?.includes('--self-test')
    && process.argv[1]?.replaceAll('\\', '/').endsWith('/least-squares.mjs')) await selfTest();
