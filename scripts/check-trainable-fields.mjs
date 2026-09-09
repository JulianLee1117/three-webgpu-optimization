/**
 * CPU-only, fixed-budget feasibility screen. Run: node scripts/check-trainable-fields.mjs
 * No AD compiler or GPU is used. The numeric forward program, targets, observations,
 * finite differences, reduction and Adam below are independent implementations.
 * Float64 optimization is an algorithmic sanity check, not GPU/f32 parity evidence.
 * Every prescribed run is retained, including failures; there are no tuning flags.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {FIELDS, initialWeights, makeSamples, numericField} from '../experiments/trainable-tsl/fields.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const configuration = Object.freeze({kinds: ['islands', 'neural'], seeds: [11, 29], samples: 128,
  steps: 160, epsilon: .001, rate: .025, beta1: .9, beta2: .999, adamEpsilon: 1e-8,
  lowerWeight: -8, upperWeight: 8, maximumHeldoutRMSEFraction: .2});
const sha256 = value => createHash('sha256').update(value).digest('hex');
const sourcePaths = ['experiments/trainable-tsl/fields.js', 'scripts/check-trainable-fields.mjs'];
const sourceSHA256 = Object.fromEntries(await Promise.all(sourcePaths.map(async name => [name, sha256(await readFile(path.join(root, name)))])));

function forward(kind, x, z, weights) {
  if (kind === 'islands') {
    let total = weights[27];
    for (let j = 0; j < 9; j++) {
      const offset = 3 * j, dx = x - weights[offset + 1], dz = z - weights[offset + 2];
      total += weights[offset] * Math.exp(-5 * (dx * dx + dz * dz));
    }
    return total;
  }
  let total = weights[48];
  for (let j = 0; j < 12; j++) {
    const offset = 4 * j;
    total += weights[offset + 3] * Math.sin(weights[offset] * x + weights[offset + 1] * z + weights[offset + 2]);
  }
  return total;
}

function target(kind, x, z) {
  const gaussian = (amplitude, precision, cx, cz) => amplitude * Math.exp(-precision * ((x - cx) ** 2 + (z - cz) ** 2));
  return kind === 'islands'
    ? gaussian(.9, 6, -.42, -.22) + gaussian(.72, 8, .48, .28) + gaussian(-.4, 7, .2, -.6)
    : .48 * Math.sin(2.6 * x + .5) * Math.cos(2.1 * z - .35) + .2 * Math.cos(3.5 * z + x);
}

function independentSamples(kind, seed) {
  let state = (seed ^ 0x971) >>> 0;
  const random = () => {state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296;};
  const observations = new Float32Array(configuration.samples * 4);
  for (let i = 0; i < configuration.samples; i++) {
    const x = Math.fround(2 * random() - 1), z = Math.fround(2 * random() - 1);
    observations.set([x, z, target(kind, x, z), 1], i * 4);
  }
  return observations;
}

function mse(kind, weights, samples) {
  let total = 0;
  for (let i = 0; i < samples.length; i += 4) {
    const residual = forward(kind, samples[i], samples[i + 1], weights) - samples[i + 2];
    total += residual * residual;
  }
  return total / (samples.length / 4);
}

function heldout(kind, weights) {
  let sum = 0, maxError = 0;
  for (let i = 0; i < 31; i++) for (let j = 0; j < 31; j++) {
    const x = -1 + 2 * (i + .37) / 31, z = -1 + 2 * (j + .61) / 31;
    const error = Math.abs(forward(kind, x, z, weights) - target(kind, x, z));
    sum += error * error;
    maxError = Math.max(maxError, error);
  }
  return {count: 31 * 31, rmse: Math.sqrt(sum / (31 * 31)), maxError};
}

function oracleParity(kind, weights, samples) {
  let maximumAbsolute = 0;
  for (let i = 0; i < samples.length; i += 4) {
    const delta = Math.abs(forward(kind, samples[i], samples[i + 1], weights) - numericField(kind, samples[i], samples[i + 1], weights));
    assert.ok(Number.isFinite(delta) && delta < 1e-12, 'Independent forward differs from numericField');
    maximumAbsolute = Math.max(maximumAbsolute, delta);
  }
  return maximumAbsolute;
}

function run(kind, seed) {
  const started = performance.now(), result = {kind, seed, status: 'running', completed: 0, history: []};
  try {
    assert.equal(FIELDS[kind].rate, configuration.rate, 'Fixture rate changed; review the frozen check');
    assert.equal(FIELDS[kind].count, kind === 'islands' ? 28 : 49, 'Fixture parameter count changed');
    const samples = independentSamples(kind, seed);
    assert.deepEqual(samples, makeSamples(kind, seed, configuration.samples), 'Independent observations differ');
    const weights = Float64Array.from(initialWeights(kind, seed)), count = weights.length;
    const first = new Float64Array(count), second = new Float64Array(count), gradient = new Float64Array(count);
    result.samples = Array.from(samples);
    result.samplesSHA256 = sha256(Buffer.from(samples.buffer));
    result.initialWeights = Array.from(weights);
    result.initialForwardOracleMaxError = oracleParity(kind, weights, samples);
    result.initial = {trainingRMSE: Math.sqrt(mse(kind, weights, samples)), heldout: heldout(kind, weights)};
    for (let step = 1; step <= configuration.steps; step++) {
      // Each per-observation squared-loss difference is reduced exactly as in the
      // GPU lane. Every parameter reads the same pre-update weight vector.
      for (let parameter = 0; parameter < count; parameter++) {
        const original = weights[parameter];
        let sum = 0;
        for (let i = 0; i < samples.length; i += 4) {
          weights[parameter] = original + configuration.epsilon;
          const plus = forward(kind, samples[i], samples[i + 1], weights) - samples[i + 2];
          weights[parameter] = original - configuration.epsilon;
          const minus = forward(kind, samples[i], samples[i + 1], weights) - samples[i + 2];
          sum += (plus * plus - minus * minus) / (2 * configuration.epsilon);
        }
        weights[parameter] = original;
        gradient[parameter] = sum / configuration.samples;
      }
      for (let parameter = 0; parameter < count; parameter++) {
        const g = gradient[parameter];
        assert.ok(Number.isFinite(g), 'Nonfinite numeric finite difference');
        first[parameter] = configuration.beta1 * first[parameter] + (1 - configuration.beta1) * g;
        second[parameter] = configuration.beta2 * second[parameter] + (1 - configuration.beta2) * g * g;
        const correctedFirst = first[parameter] / (1 - configuration.beta1 ** step);
        const correctedSecond = second[parameter] / (1 - configuration.beta2 ** step);
        const proposed = weights[parameter] - configuration.rate * correctedFirst / (Math.sqrt(correctedSecond) + configuration.adamEpsilon);
        assert.ok(Number.isFinite(proposed), 'Nonfinite Adam update');
        weights[parameter] = Math.max(configuration.lowerWeight, Math.min(configuration.upperWeight, proposed));
      }
      result.completed = step;
      result.weights = Array.from(weights);
      result.history.push({step, trainingRMSE: Math.sqrt(mse(kind, weights, samples)), heldout: heldout(kind, weights)});
    }
    result.final = result.history.at(-1);
    result.finalForwardOracleMaxError = oracleParity(kind, weights, samples);
    result.heldoutRMSEFraction = result.final.heldout.rmse / result.initial.heldout.rmse;
    result.heldoutRMSEReductionPercent = 100 * (1 - result.heldoutRMSEFraction);
    result.utilityPassed = result.completed === configuration.steps && result.heldoutRMSEFraction <= configuration.maximumHeldoutRMSEFraction;
    result.status = result.utilityPassed ? 'passed' : 'failed-utility';
  } catch (error) {
    result.status = 'failed-error';
    result.utilityPassed = false;
    result.error = {name: error.name, message: error.message, stack: error.stack};
  }
  result.cpuWallMs = performance.now() - started;
  return result;
}

const createdAt = new Date().toISOString();
const report = {kind: 'trainable-fields-cpu-feasibility-v1', createdAt, configuration, sourceSHA256,
  method: 'Independent numeric central finite differences of per-observation squared error, mean reduction, full-batch Adam',
  precision: 'Float32 initial weights and observations; Float64 forward, finite differences, optimizer state and held-out evaluation',
  scope: 'Four predetermined synthetic parameter-fitting checks. CPU feasibility only; not validation of native AD, GPU reduction, GPU arithmetic, speed, novelty, or general game utility.',
  limitations: ['The neural target is representable by a few sinusoidal units of this chosen model family.',
    'The islands target is a smooth sum of Gaussian bumps, deliberately similar to the learned family.',
    'Held-out observations are an independent offset grid of the same noiseless analytic target on the same domain.',
    'The 80% held-out RMSE reduction gate is fixed before these runs. No failed run is dropped or retuned.'], runs: []};
for (const kind of configuration.kinds) for (const seed of configuration.seeds) {
  const result = run(kind, seed);
  report.runs.push(result);
  console.log(JSON.stringify({kind, seed, status: result.status, completed: result.completed,
    initialHeldoutRMSE: result.initial?.heldout.rmse, finalHeldoutRMSE: result.final?.heldout.rmse,
    heldoutRMSEReductionPercent: result.heldoutRMSEReductionPercent, error: result.error?.message}));
}
report.sourceUnchanged = (await Promise.all(sourcePaths.map(async name => sha256(await readFile(path.join(root, name))) === sourceSHA256[name]))).every(Boolean);
report.passed = report.sourceUnchanged && report.runs.every(result => result.status === 'passed');
report.status = report.passed ? 'passed' : 'failed';
const output = path.join(root, 'results/development/trainable-fields-cpu', createdAt.replaceAll(':', '-'));
await mkdir(output, {recursive: true});
const bytes = JSON.stringify(report, null, 2) + '\n';
await writeFile(path.join(output, 'report.json'), bytes, {flag: 'wx'});
await writeFile(path.join(output, 'report.sha256'), `${sha256(bytes)}  report.json\n`, {flag: 'wx'});
console.log(JSON.stringify({status: report.status, report: path.relative(root, path.join(output, 'report.json')).replaceAll('\\', '/'), sha256: sha256(bytes)}));
if (!report.passed) process.exitCode = 1;
