import * as THREE from 'three/webgpu';
import { GaussianSplat } from 'three/addons/objects/GaussianSplat.js';
import { SPLATLoader } from 'three/addons/loaders/SPLATLoader.js';
import { createGaussianSplatGeometry } from 'three/addons/utils/GaussianSplatUtils.js';
import { createSplatDeformer } from './deform.js';
import { prepareBasis, evaluateBasis } from './basis.mjs';
import { decodeSplat, normalizePositions, sampleQuadrature } from './assets.mjs';

// Declared before the first GPU run. These compare against a double arithmetic
// oracle using the exact Float32 inputs sent to the GPU, not its output arrays.
export const TOLERANCES = Object.freeze({ centerAbsolute: 2e-6, centerRelative: 2e-6, covarianceAbsolute: 1e-9, covarianceRelative: 4e-5 });
const assert = (value, message) => { if (!value) throw Error(message); };
const f32 = values => Float32Array.from(values);
const sha = async buffer => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), x => x.toString(16).padStart(2, '0')).join('');

// Independent array/matrix CPU oracle. No helper from deform.js is used.
function oracle(input, state, omitGradient = false, keepRestCovariance = false) {
  const { positions: p, covariance: c, weights: w, gradients: dw, modeCount: H } = input;
  const count = p.length / 3, centers = new Float64Array(count * 3), covariance = new Float64Array(count * 6), jacobians = new Float64Array(count * 9);
  for (let point = 0; point < count; point++) {
    const X = [p[point * 3], p[point * 3 + 1], p[point * 3 + 2], 1];
    const J = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    centers.set(X.slice(0, 3), point * 3);
    for (let h = 0; h < H; h++) for (let a = 0; a < 3; a++) {
      let displacement = 0;
      for (let d = 0; d < 4; d++) displacement += state[h * 12 + a * 4 + d] * X[d];
      centers[point * 3 + a] += w[point * H + h] * displacement;
      for (let d = 0; d < 3; d++) J[a * 3 + d] += w[point * H + h] * state[h * 12 + a * 4 + d] + (omitGradient ? 0 : dw[(point * H + h) * 3 + d] * displacement);
    }
    jacobians.set(J, point * 9);
    const [a, b, d, e, f, g] = c.subarray(point * 6, point * 6 + 6), C = [a, b, d, b, e, f, d, f, g], out = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++) out[i * 3 + j] += J[i * 3 + k] * C[k * 3 + l] * J[j * 3 + l];
    covariance.set(keepRestCovariance ? c.subarray(point * 6, point * 6 + 6) : [out[0], out[1], out[2], out[4], out[5], out[8]], point * 6);
  }
  return { centers, covariance, jacobians };
}

function comparison(actual, expected, absolute, relative) {
  let maxAbsolute = 0, maxToleranceRatio = 0, worstIndex = -1;
  for (let i = 0; i < expected.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) return { passed: false, failure: 'nonfinite', index: i };
    const error = Math.abs(actual[i] - expected[i]), ratio = error / (absolute + relative * Math.abs(expected[i]));
    maxAbsolute = Math.max(maxAbsolute, error);
    if (ratio > maxToleranceRatio) { maxToleranceRatio = ratio; worstIndex = i; }
  }
  return { passed: maxToleranceRatio <= 1, maxAbsolute, maxToleranceRatio, worstIndex };
}

function synthetic(count = 96) {
  const positions = new Float32Array(count * 3), covariance = new Float32Array(count * 6), weights = new Float32Array(count * 3), gradients = new Float32Array(count * 9);
  for (let i = 0; i < count; i++) {
    const x = 0.42 * Math.sin(i * 1.618), y = 0.31 * Math.cos(i * 0.717), z = 0.36 * Math.sin(i * 0.419 + 0.4);
    positions.set([x, y, z], i * 3);
    const a = 0.008 + 0.00009 * i, b = 0.016, c = 0.020;
    covariance.set([a * a, a * 0.002, -a * 0.001, b * b + 0.002 ** 2, 0.002 * -0.001 + b * 0.003, c * c + 0.001 ** 2 + 0.003 ** 2], i * 6);
    weights.set([0.25 + x * y, 0.5 + x * z - z, 1], i * 3);
    gradients.set([y, x, 0, z, 0, x - 1, 0, 0, 0], i * 9);
  }
  return { id: `analytic-${count}`, positions, covariance, weights, gradients, modeCount: 3 };
}

async function spot() {
  const response = await fetch(new URL('./assets/spot.splat', import.meta.url));
  assert(response.ok, 'Spot asset request failed');
  const bytes = await response.arrayBuffer(), assetSHA256 = await sha(bytes), decoded = decodeSplat(bytes);
  const normalized = normalizePositions(decoded.positions), quadrature = sampleQuadrature(normalized.positions, 256);
  const basis = prepareBasis({ positions: quadrature.positions, volumes: quadrature.volumes, centerCount: 48, modeCount: 6 });
  const geometry = new SPLATLoader().parse(bytes), sourceCov = geometry.getAttribute('covariance').array;
  const indices = Uint32Array.from({ length: 128 }, (_, i) => Math.floor(i * (decoded.count - 1) / 127));
  const positions = new Float64Array(128 * 3), covariance = new Float32Array(128 * 6);
  for (let i = 0; i < 128; i++) {
    positions.set(normalized.positions.subarray(indices[i] * 3, indices[i] * 3 + 3), i * 3);
    for (let d = 0; d < 6; d++) covariance[i * 6 + d] = sourceCov[indices[i] * 6 + d] * normalized.scale ** 2 * (d === 1 || d === 4 ? -1 : 1);
  }
  geometry.dispose();
  const fields = evaluateBasis(basis, positions);
  return { id: 'spot-128', positions: f32(positions), covariance, weights: f32(fields.weights), gradients: f32(fields.gradients), modeCount: 6,
    preparation: { assetSHA256, indices, normalization: { scale: normalized.scale, center: normalized.center, flipY: normalized.flipY },
      quadrature, basis, float64QueryPositions: positions, float64Weights: fields.weights, float64Gradients: fields.gradients } };
}

function states(input) {
  const h = input.modeCount, zero = new Float32Array(h * 12), affine = zero.slice(), nonaffine = zero.slice();
  affine.set([0.08, 0.11, -0.025, 0.06, -0.035, -0.04, 0.055, -0.02, 0.07, -0.03, 0.045, 0.025], (h - 1) * 12);
  for (let i = 0; i < nonaffine.length; i++) nonaffine[i] = 0.018 * Math.sin(0.77 * i + 0.34);
  return input.id === 'analytic-97-tail' ? [{ id: 'nonaffine-tail', q: nonaffine }] :
    [{ id: 'rest', q: zero }, ...(input.id.startsWith('analytic') ? [{ id: 'affine', q: affine }] : []), { id: 'nonaffine', q: nonaffine }, { id: 'reset', q: zero.slice() }];
}

function guardChecks(splat, input) {
  const result = [], check = (name, callback) => { let message = null; try { callback(); } catch (e) { message = String(e.message); } assert(message, `Negative guard ${name} was accepted`); result.push({ name, rejected: true, message }); };
  check('mismatched-count', () => createSplatDeformer(splat, input.positions.subarray(0, input.positions.length - 3), input.covariance, input));
  check('short-gradients', () => createSplatDeformer(splat, input.positions, input.covariance, { ...input, gradients: input.gradients.subarray(3) }));
  const invalid = input.weights.slice(); invalid[invalid.length - 1] = NaN;
  check('nonfinite-last-weight', () => createSplatDeformer(splat, input.positions, input.covariance, { ...input, weights: invalid }));
  const indefinite = input.covariance.slice(); indefinite[0] = -1;
  check('indefinite-covariance', () => createSplatDeformer(splat, input.positions, indefinite, input));
  return result;
}

export async function runProbe() {
  if (window.localElasticityProbeResult) throw Error('This page permits one bounded probe run');
  const result = { kind: 'local-elasticity-gpu-probe-v1', status: 'running', revision: THREE.REVISION, tolerances: TOLERANCES,
    groups: [], cells: [], errors: [], lifecycle: { rendererCount: 0, renderCount: 0, deformationDispatches: 0, disposed: false } };
  window.localElasticityProbeResult = result;
  let renderer, device, scopePushed = false;
  try {
    assert(THREE.REVISION === '186', 'Wrong Three revision');
    // CPU preparation and file decoding finish before the GPU device is requested.
    const tail = synthetic(97); tail.id = 'analytic-97-tail';
    const inputs = [synthetic(), await spot(), tail];
    renderer = new THREE.WebGPURenderer({ antialias: false }); renderer.setPixelRatio(1); renderer.setSize(256, 192); renderer.setClearColor(0x102020);
    result.lifecycle.rendererCount++; await renderer.init();
    assert(renderer.backend.isWebGPUBackend, 'WebGPU fallback is not an eligible run');
    device = renderer.backend.device;
    device.addEventListener('uncapturederror', event => result.errors.push(String(event.error.message)));
    device.pushErrorScope('validation'); scopePushed = true;
    const info = device.adapterInfo; result.adapter = info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : null;
    const camera = new THREE.PerspectiveCamera(43, 256 / 192, 0.01, 30); camera.position.set(1.2, 0.7, 1.8); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
    for (const input of inputs) {
      const count = input.positions.length / 3, colors = new Uint8Array(count * 4).fill(255);
      const geometry = createGaussianSplatGeometry(input.positions.slice(), input.covariance.slice(), colors), splat = new GaussianSplat(geometry), scene = new THREE.Scene(); scene.add(splat);
      const group = { id: input.id, input, guards: guardChecks(splat, input) }; result.groups.push(group);
      const deformer = createSplatDeformer(splat, input.positions, input.covariance, input);
      const badQ = new Float64Array(input.modeCount * 12); badQ[badQ.length - 1] = Infinity;
      let rejected = false; try { deformer.update(renderer, badQ); } catch { rejected = true; } assert(rejected, 'Nonfinite update was accepted');
      group.guards.push({ name: 'nonfinite-update', rejected });
      let originalRest = null;
      for (const state of states(input)) {
        const expected = oracle(input, state.q);
        deformer.update(renderer, state.q); result.lifecycle.deformationDispatches++;
        assert(splat._sortInitialized === false, 'Deformation did not invalidate depth sorting');
        renderer.render(scene, camera); result.lifecycle.renderCount++;
        assert(splat._sortInitialized === true, 'Renderer did not sort updated splats');
        const raw = [];
        for (const attribute of deformer.outputAttributes) raw.push(new Float32Array(await renderer.getArrayBufferAsync(attribute)));
        const order = new Uint32Array(await renderer.getArrayBufferAsync(splat._sort.orderAttribute));
        const centers = new Float32Array(count * 3), covariance = new Float32Array(count * 6);
        for (let i = 0; i < count; i++) {
          centers.set(raw[0].subarray(i * 4, i * 4 + 3), i * 3);
          covariance.set([raw[1][i * 4], raw[1][i * 4 + 1], raw[1][i * 4 + 2], raw[1][i * 4 + 3], raw[2][i * 4], raw[2][i * 4 + 1]], i * 6);
          assert(raw[0][i * 4 + 3] === 0 && raw[2][i * 4 + 2] === 0 && raw[2][i * 4 + 3] === 0, 'Output packing sentinel mismatch');
          assert(Math.hypot(...centers.subarray(i * 3, i * 3 + 3)) <= splat.boundingSphere.radius, 'Center escaped sorting envelope');
        }
        const centerCheck = comparison(centers, expected.centers, TOLERANCES.centerAbsolute, TOLERANCES.centerRelative);
        const covarianceCheck = comparison(covariance, expected.covariance, TOLERANCES.covarianceAbsolute, TOLERANCES.covarianceRelative);
        assert(centerCheck.passed && covarianceCheck.passed, `${input.id}/${state.id} numerical parity failed`);
        assert(order.length === count && new Set(order).size === count && order.every(i => i < count), 'Depth order is not an exact index permutation');
        const m = splat._sortMatrix.value.elements, [near, far] = splat._sortDepthRange.value.toArray(), range = Math.max(far - near, 0.0001);
        const bins = Array.from(order, i => { const depth = -(m[2] * centers[i * 3] + m[6] * centers[i * 3 + 1] + m[10] * centers[i * 3 + 2] + m[14]); return Math.floor(Math.max(0, Math.min(1, (depth - near) / range)) * 4095); });
        // One-bin slack accounts for CPU-double vs GPU-Float32 boundary rounding.
        assert(bins.every((v, i) => i === 0 || v <= bins[i - 1] + 1), 'Deformed depth bins are not ordered');
        if (state.id === 'rest') originalRest = { centers: centers.slice(), covariance: covariance.slice() };
        if (state.id === 'reset') { assert(centers.every((v, i) => v === originalRest.centers[i]), 'Reset centers differ'); assert(covariance.every((v, i) => v === originalRest.covariance[i]), 'Reset covariance differs'); }
        const controls = {};
        if (state.id.startsWith('nonaffine')) {
          const omitted = oracle(input, state.q, true), stale = oracle(input, state.q, false, true);
          controls.omittedGradient = comparison(omitted.covariance, expected.covariance, TOLERANCES.covarianceAbsolute, TOLERANCES.covarianceRelative);
          controls.staleCovariance = comparison(stale.covariance, expected.covariance, TOLERANCES.covarianceAbsolute, TOLERANCES.covarianceRelative);
          assert(!controls.omittedGradient.passed && !controls.staleCovariance.passed, 'The covariance negatives were not distinguishable');
        }
        result.cells.push({ group: input.id, state: state.id, q: state.q, expected, actual: { centers, covariance, raw, order }, centerCheck, covarianceCheck,
          controls, sort: { radius: splat.boundingSphere.radius, matrix: m.slice(), range: [near, far], bins, initialized: splat._sortInitialized }, passed: true });
      }
      scene.remove(splat); geometry.dispose(); splat.geometry.dispose(); splat.material.dispose();
    }
    const error = await device.popErrorScope(); scopePushed = false;
    if (error) result.errors.push(error.message);
    assert(result.errors.length === 0, 'GPU validation errors');
    result.status = 'passed';
  } catch (error) { result.status = 'failed'; result.failure = String(error.stack ?? error); }
  finally {
    if (scopePushed) { const error = await device.popErrorScope(); if (error) result.errors.push(error.message); }
    if (renderer) { await renderer.dispose(); result.lifecycle.disposed = true; }
    if (device) { const loss = await device.lost; result.lifecycle.deviceLossReason = loss.reason; assert(loss.reason === 'destroyed', 'Unexpected device loss'); }
    if (result.errors.length) result.status = 'failed';
  }
  return result;
}
