/**
 * Small, deterministic black-box optimizers for the scene-fit experiment.
 * Every parameter has the same normalized [0, 1] bounds. No optimizer receives
 * scene names, reference parameters, pixels, or gradients, only a scalar loss.
 *
 * The candidate is a budget-truncated Powell-style direction-set search with
 * bounded Brent line searches (parabolic interpolation + golden-section steps).
 * These are established algorithms, not a new optimization method. Background:
 * https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
 */

export const METHOD_LABELS = Object.freeze({
  random: 'Uniform random search',
  coordinate: 'Coordinate pattern search',
  candidate: 'Powell-style direction search / bounded Brent',
});

const clamp = (x) => Math.max(0, Math.min(1, x));
const same = (a, b) => a.every((x, i) => x === b[i]);

function checkDimension(n) {
  if (!Number.isInteger(n) || n < 1 || n > 16) {
    throw new RangeError('Dimension must be an integer from 1 through 16.');
  }
}

/** Mulberry32 PRNG. Seeds are explicit unsigned 32-bit integers, including zero. */
export function seededRandom(seed = 0) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new RangeError('Seed must be an unsigned 32-bit integer.');
  }
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Call once per seed, then pass this same vector to every comparison method. */
export function initialVector(n, seed = 0) {
  checkDimension(n);
  const random = seededRandom(seed);
  return Array.from({ length: n }, () => random());
}

/**
 * evaluate receives a disposable array copy and must resolve to a finite number.
 * Initial evaluation counts toward budget; successful runs use exactly budget
 * evaluations, even on a constant objective. All calls and callbacks are serial.
 * history entries: { evaluation, parameters, loss, best, bestLoss }.
 */
export async function optimize({ method, initial, budget = 96, seed = 0, evaluate, onProgress } = {}) {
  if (!Object.hasOwn(METHOD_LABELS, method)) throw new RangeError('Unknown optimization method.');
  if (!Array.isArray(initial) && !ArrayBuffer.isView(initial)) {
    throw new TypeError('Initial parameters must be an array or typed array.');
  }
  checkDimension(initial.length);
  if (!Number.isInteger(budget) || budget < 1 || budget > 256) {
    throw new RangeError('Budget must be an integer from 1 through 256.');
  }
  if (typeof evaluate !== 'function') throw new TypeError('evaluate must be a function.');
  if (onProgress !== undefined && typeof onProgress !== 'function') {
    throw new TypeError('onProgress must be a function when supplied.');
  }
  const start = Array.from(initial, (value) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new TypeError('All initial parameters must be finite numbers.');
    }
    return clamp(value);
  });
  const random = seededRandom(seed);
  const dimension = start.length;
  // The first n draws belong to initialVector(n, seed). Advance past that
  // shared starting point so random search does not spend evaluation 2 on it.
  for (let i = 0; i < dimension; i++) random();
  let best = start.slice();
  let bestLoss = Infinity;
  let evaluations = 0;
  const history = [];
  const remaining = () => budget - evaluations;

  async function score(vector) {
    if (!remaining()) throw new Error('Internal error: evaluation budget exhausted.');
    const parameters = vector.map((value) => {
      if (!Number.isFinite(value)) throw new Error('Optimizer produced a non-finite parameter.');
      return clamp(value);
    });
    evaluations++;
    const loss = await evaluate(parameters.slice());
    if (typeof loss !== 'number' || !Number.isFinite(loss)) {
      throw new TypeError('evaluate must return a finite numeric loss.');
    }
    if (loss < bestLoss) {
      bestLoss = loss;
      best = parameters.slice();
    }
    const entry = { evaluation: evaluations, parameters, loss, best: best.slice(), bestLoss };
    history.push(entry);
    if (onProgress) {
      await onProgress({ ...entry, parameters: parameters.slice(), best: best.slice() });
    }
    return loss;
  }

  let current = start.slice();
  let currentLoss = await score(current);

  if (method === 'random') {
    while (remaining()) await score(Array.from({ length: dimension }, () => random()));
  } else if (method === 'coordinate') {
    // Commit the best of +/- for each coordinate, then move to the next axis.
    // Keep a step size until an entire sweep fails; then .20 -> .10 -> .05 ...
    let step = 0.20;
    while (remaining()) {
      let improved = false;
      for (let axis = 0; axis < dimension && remaining(); axis++) {
        const anchor = current.slice();
        const before = currentLoss;
        for (const sign of [-1, 1]) {
          if (!remaining()) break;
          const trial = anchor.slice();
          trial[axis] = clamp(trial[axis] + sign * step);
          // Retaining clipped repeats gives a fully specified equal-budget run.
          const loss = await score(trial);
          if (loss < currentLoss) {
            current = trial;
            currentLoss = loss;
          }
        }
        improved ||= currentLoss < before;
      }
      if (!improved) step = Math.max(step * 0.5, 1e-7);
    }
  } else {
    let directions = Array.from({ length: dimension }, (_, axis) =>
      Array.from({ length: dimension }, (_, i) => Number(i === axis)));

    async function lineSearch(origin, originLoss, direction) {
      let lower = -Infinity;
      let upper = Infinity;
      for (let i = 0; i < dimension; i++) {
        if (Math.abs(direction[i]) < 1e-14) continue;
        const a = -origin[i] / direction[i];
        const b = (1 - origin[i]) / direction[i];
        lower = Math.max(lower, Math.min(a, b));
        upper = Math.min(upper, Math.max(a, b));
      }
      if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper - lower < 1e-10) {
        return { point: origin, loss: originLoss };
      }
      const at = (alpha) => origin.map((value, i) => clamp(value + alpha * direction[i]));
      // Endpoint checks include true box minima, which an open-interval scalar
      // minimizer would only approach. Existing origin evaluation is reused.
      const samples = [{ x: 0, f: originLoss }];
      const lineBudget = Math.min(6, remaining());
      let used = 0;
      for (const alpha of [lower, upper]) {
        if (used >= lineBudget) break;
        if (Math.abs(alpha) <= 1e-12) continue;
        samples.push({ x: alpha, f: await score(at(alpha)) });
        used++;
      }
      samples.sort((a, b) => a.f - b.f);
      let { x, f: fx } = samples[0];
      let { x: w, f: fw } = samples[1] ?? samples[0];
      let { x: v, f: fv } = samples[2] ?? samples[0];
      let movement = 0;
      let previousMovement = upper - lower;
      const golden = (3 - Math.sqrt(5)) / 2;
      while (used < lineBudget && remaining()) {
        const middle = (lower + upper) / 2;
        const tolerance = 1e-6 * Math.abs(x) + 1e-6;
        if (upper - lower < 4 * tolerance) break;
        let useGolden = true;
        if (Math.abs(previousMovement) > tolerance) {
          const r = (x - w) * (fx - fv);
          let q = (x - v) * (fx - fw);
          let p = (x - v) * q - (x - w) * r;
          q = 2 * (q - r);
          if (q > 0) p = -p;
          q = Math.abs(q);
          const earlierMovement = previousMovement;
          previousMovement = movement;
          if (q > 0 && Math.abs(p) < Math.abs(q * earlierMovement / 2)
              && p > q * (lower - x) && p < q * (upper - x)) {
            movement = p / q;
            const proposed = x + movement;
            if (proposed - lower < 2 * tolerance || upper - proposed < 2 * tolerance) {
              movement = middle >= x ? tolerance : -tolerance;
            }
            useGolden = false;
          }
        }
        if (useGolden) {
          previousMovement = x < middle ? upper - x : lower - x;
          movement = golden * previousMovement;
        }
        const sign = movement >= 0 ? 1 : -1;
        const u = Math.max(lower, Math.min(upper, x + sign * Math.max(Math.abs(movement), tolerance)));
        const fu = await score(at(u));
        used++;
        if (fu <= fx) {
          if (u >= x) lower = x;
          else upper = x;
          v = w; fv = fw;
          w = x; fw = fx;
          x = u; fx = fu;
        } else {
          if (u < x) lower = u;
          else upper = u;
          if (fu <= fw || w === x) {
            v = w; fv = fw;
            w = u; fw = fu;
          } else if (fu <= fv || v === x || v === w) {
            v = u; fv = fu;
          }
        }
      }
      return { point: at(x), loss: fx };
    }

    while (remaining()) {
      const cycleStart = current.slice();
      const cycleLoss = currentLoss;
      const evaluationsBeforeCycle = evaluations;
      let largestDecrease = 0;
      let largestAxis = 0;
      for (let axis = 0; axis < dimension && remaining(); axis++) {
        const before = currentLoss;
        const result = await lineSearch(current, currentLoss, directions[axis]);
        current = result.point;
        currentLoss = result.loss;
        if (before - currentLoss > largestDecrease) {
          largestDecrease = before - currentLoss;
          largestAxis = axis;
        }
      }
      const displacement = current.map((value, i) => value - cycleStart[i]);
      const norm = Math.hypot(...displacement);
      if (remaining() && norm > 1e-8 && largestDecrease > 0) {
        const extrapolated = current.map((value, i) => clamp(value + displacement[i]));
        const extrapolatedLoss = same(extrapolated, current) ? currentLoss : await score(extrapolated);
        // Powell's usual extrapolation test, adapted to the constrained box.
        const change = cycleLoss - currentLoss;
        const t = 2 * (cycleLoss - 2 * currentLoss + extrapolatedLoss)
            * (change - largestDecrease) ** 2
          - largestDecrease * (cycleLoss - extrapolatedLoss) ** 2;
        if (remaining() && extrapolatedLoss < cycleLoss && t < 0) {
          const direction = displacement.map((value) => value / norm);
          const result = await lineSearch(current, currentLoss, direction);
          current = result.point;
          currentLoss = result.loss;
          directions[largestAxis] = directions[dimension - 1];
          directions[dimension - 1] = direction;
        }
      }
      // Degenerate directions can have zero feasible span at a box corner.
      // Reset to coordinate directions, and count one incumbent check to ensure
      // termination even on exact ties. No unbounded inner convergence loops.
      if (evaluations === evaluationsBeforeCycle) {
        directions = Array.from({ length: dimension }, (_, axis) =>
          Array.from({ length: dimension }, (_, i) => Number(i === axis)));
        await score(current);
      }
    }
  }
  return { best: best.slice(), bestLoss, evaluations, history };
}

async function selfTest() {
  const { default: assert } = await import('node:assert/strict');
  const methods = Object.keys(METHOD_LABELS);
  const sphere = (x) => x.reduce((sum, value, i) => sum + (value - [0.23, 0.68, 0.41, 0.8][i]) ** 2, 0);
  const coupled = (x) => {
    const a = x[0] - 0.23;
    const b = x[1] - 0.68;
    const c = x[2] - 0.41;
    // Rotated positive-definite quadratic, with a known non-axis-aligned basin.
    return (a + b) ** 2 + 4 * (a - b + c) ** 2 + 12 * (b + c) ** 2;
  };
  const boundary = (x) => (x[0] + 0.4) ** 2 + (x[1] - 1.3) ** 2 + (x[2] - 0.65) ** 2;
  const summaries = [];
  for (const [name, objective, dimension, optimum] of [
    ['sphere', sphere, 4, 0], ['coupled', coupled, 3, 0], ['boundary', boundary, 3, 0.25],
  ]) {
    for (const method of methods) {
      const initial = initialVector(dimension, 29);
      const calls = [];
      const progress = [];
      const options = { method, initial, budget: 96, seed: 29, evaluate: async (x) => {
        assert.ok(x.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
        calls.push(x.slice());
        await Promise.resolve();
        return objective(x);
      }, onProgress: async (entry) => {
        progress.push(entry.evaluation);
        entry.best.fill(99); // Progress observers must not mutate optimizer state.
      } };
      const result = await optimize(options);
      assert.equal(result.evaluations, 96);
      assert.equal(calls.length, 96);
      assert.equal(progress.length, 96);
      assert.deepEqual(calls[0], initial);
      assert.deepEqual(progress, Array.from({ length: 96 }, (_, i) => i + 1));
      assert.equal(result.bestLoss, objective(result.best));
      assert.ok(result.history.every((entry, i, rows) => i === 0 || entry.bestLoss <= rows[i - 1].bestLoss));
      const repeat = await optimize({ ...options, onProgress: undefined, evaluate: objective });
      assert.deepEqual(repeat, result);
      assert.ok(result.bestLoss < objective(initial), `${method} must make progress on ${name}`);
      if (method !== 'random') {
        assert.ok(result.bestLoss - optimum < 0.02, `${method}/${name}: ${result.bestLoss}`);
      }
      summaries.push({ objective: name, method, loss: result.bestLoss });
    }
  }
  for (const method of methods) {
    for (const budget of [1, 2, 3, 17, 256]) {
      const result = await optimize({ method, initial: [0, 1], budget, evaluate: () => 7 });
      assert.equal(result.evaluations, budget);
      assert.equal(result.bestLoss, 7);
    }
    const clamped = await optimize({ method, initial: [-1, 2], budget: 1, evaluate: sphere });
    assert.deepEqual(clamped.best, [0, 1]);
    const n16 = await optimize({ method, initial: initialVector(16, 11), budget: 96, evaluate: (x) => x.reduce((s, v) => s + v * v, 0) });
    assert.equal(n16.evaluations, 96);
    await assert.rejects(optimize({ method, initial: [0.5], evaluate: () => Infinity }), /finite numeric loss/);
    await assert.rejects(optimize({ method, initial: [0.5], evaluate: () => NaN }), /finite numeric loss/);
  }
  for (const override of [{ budget: 257 }, { budget: 0 }, { budget: 1.5 }, { initial: [] },
    { initial: Array(17).fill(0) }, { initial: [NaN] }, { seed: -1 }, { method: 'unknown' }]) {
    await assert.rejects(optimize({ method: 'candidate', initial: [0.5], evaluate: sphere, ...override }));
  }
  const rng = seededRandom(0);
  const values = Array.from({ length: 1000 }, () => rng());
  assert.ok(values.every((x) => x >= 0 && x < 1));
  assert.deepEqual(initialVector(4, 47), initialVector(4, 47));
  assert.notDeepEqual(initialVector(4, 11), initialVector(4, 47));
  console.log(JSON.stringify({ passed: true, evaluationsPerFit: 96, summaries }, null, 2));
}

if (typeof process !== 'undefined' && process.argv?.includes('--self-test')
    && process.argv[1]?.replaceAll('\\', '/').endsWith('/optimizers.mjs')) {
  await selfTest();
}
