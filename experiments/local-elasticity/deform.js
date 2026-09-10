import { StorageBufferAttribute, Sphere, Vector3, REVISION } from 'three/webgpu';
import { Fn, storage, instanceIndex, vec3, vec4, dot } from 'three/tsl';

// Narrow adapter to the private storage layout of Three r186 GaussianSplat.
// The displacement and covariance use the same affine field as the mechanics.
export function createSplatDeformer(splat, positions, covariance, fields) {
  if (REVISION !== '186') throw Error('This experimental GaussianSplat adapter requires Three r186.');
  const count = positions?.length / 3, H = fields?.modeCount;
  if (!Number.isInteger(count) || count < 1 || count > 150000 || !Number.isInteger(H) || H < 1 || H > 12) throw Error('Invalid bounded splat / mode count');
  const validate = (values, length, name, limit) => {
    if (!values?.subarray || values.length !== length) throw Error(`Invalid ${name} shape`);
    for (const value of values) if (!Number.isFinite(value) || Math.abs(value) > limit || !Number.isFinite(Math.fround(value))) throw Error(`Invalid ${name} value`);
  };
  validate(positions, count * 3, 'positions', 8);
  validate(covariance, count * 6, 'covariance', 1e6);
  validate(fields.weights, count * H, 'weights', 1e6);
  validate(fields.gradients, count * H * 3, 'gradients', 1e6);
  const buffers = splat?._buffers;
  if (!splat?.isGaussianSplat || buffers?.count !== count || buffers.sphericalHarmonicsDegree !== 0) throw Error('Expected matching r186 degree-zero GaussianSplat storage');
  for (const name of ['centerRead', 'covarianceARead', 'covarianceBRead']) {
    const value = buffers[name]?.value;
    if (!value?.isStorageBufferAttribute || value.itemSize !== 4 || value.count !== count || !(value.array instanceof Float32Array)) throw Error('Unexpected GaussianSplat storage layout');
  }
  // PSD is required for a covariance. Singular, very flat Gaussians are allowed.
  for (let i = 0; i < count; i++) {
    const [a, b, c, d, e, f] = covariance.subarray(i * 6, i * 6 + 6), scale = Math.max(a, d, f);
    const tolerance = 2e-6 * scale * scale;
    const determinant = a * d * f + 2 * b * c * e - a * e * e - d * c * c - f * b * b;
    if (a < 0 || d < 0 || f < 0 || b * b > a * d + tolerance || c * c > a * f + tolerance || e * e > d * f + tolerance || determinant < -2e-6 * scale ** 3) throw Error('Covariance must be positive semidefinite');
  }
  const packed = new Float32Array(count * H * 4);
  for (let i = 0; i < count * H; i++) {
    packed[i * 4] = fields.weights[i];
    packed.set(fields.gradients.subarray(i * 3, i * 3 + 3), i * 4 + 1);
  }
  const rest = new Float32Array(count * 4), ca = new Float32Array(count * 4), cb = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    rest.set(positions.subarray(i * 3, i * 3 + 3), i * 4); rest[i * 4 + 3] = 1;
    ca.set(covariance.subarray(i * 6, i * 6 + 4), i * 4);
    cb.set(covariance.subarray(i * 6 + 4, i * 6 + 6), i * 4);
  }
  const attr = values => new StorageBufferAttribute(values, 4);
  const qAttribute = attr(new Float32Array(H * 12));
  const sourceAttributes = [attr(rest), attr(ca), attr(cb), attr(packed), qAttribute];
  const [p0, c0, c1, basis, q] = sourceAttributes.map(a => storage(a, 'vec4', a.count).toReadOnly());
  const centerOut = storage(buffers.centerRead.value, 'vec4', count);
  const covAOut = storage(buffers.covarianceARead.value, 'vec4', count);
  const covBOut = storage(buffers.covarianceBRead.value, 'vec4', count);
  const node = Fn(() => {
    const i = instanceIndex, X = p0.element(i).toVar();
    const displacement = vec3(0).toVar();
    const f0 = vec3(1, 0, 0).toVar(), f1 = vec3(0, 1, 0).toVar(), f2 = vec3(0, 0, 1).toVar();
    for (let h = 0; h < H; h++) {
      const b = basis.element(i.mul(H).add(h)).toVar();
      const t0 = q.element(h * 3).toVar(), t1 = q.element(h * 3 + 1).toVar(), t2 = q.element(h * 3 + 2).toVar();
      const u = vec3(dot(t0, X), dot(t1, X), dot(t2, X)).toVar();
      displacement.addAssign(u.mul(b.x));
      f0.addAssign(t0.xyz.mul(b.x).add(b.yzw.mul(u.x)));
      f1.addAssign(t1.xyz.mul(b.x).add(b.yzw.mul(u.y)));
      f2.addAssign(t2.xyz.mul(b.x).add(b.yzw.mul(u.z)));
    }
    centerOut.element(i).assign(vec4(X.xyz.add(displacement), 0));
    const a = c0.element(i), b = c1.element(i);
    const multiply = v => vec3(dot(vec3(a.x, a.y, a.z), v), dot(vec3(a.y, a.w, b.x), v), dot(vec3(a.z, b.x, b.y), v));
    const cf0 = multiply(f0).toVar(), cf1 = multiply(f1).toVar(), cf2 = multiply(f2).toVar();
    covAOut.element(i).assign(vec4(dot(f0, cf0), dot(f0, cf1), dot(f0, cf2), dot(f1, cf1)));
    covBOut.element(i).assign(vec4(dot(f1, cf2), dot(f2, cf2), 0, 0));
  })().compute(count, [64]);
  // This bounds CENTER locations for r186's depth-sort bins, not splat extents.
  // ||x+ΣwT[x,1]|| <= max||x|| + Σ max(|w|·||[x,1]||)·||T||F.
  // Use the actual Float32 inputs plus an explicit small roundoff allowance.
  let restRadius = 0;
  const modeEnvelope = new Float64Array(H);
  for (let i = 0; i < count; i++) {
    const radius = Math.hypot(rest[i * 4], rest[i * 4 + 1], rest[i * 4 + 2]);
    restRadius = Math.max(restRadius, radius);
    for (let h = 0; h < H; h++) modeEnvelope[h] = Math.max(modeEnvelope[h], Math.abs(packed[(i * H + h) * 4]) * Math.hypot(1, radius));
  }
  splat.boundingSphere = new Sphere(new Vector3(), restRadius + 1e-5);
  splat.frustumCulled = false;
  return {
    node,
    update(renderer, values) {
      if (!renderer?.backend?.isWebGPUBackend || !renderer.backend.device) throw Error('The deformation adapter requires an initialized WebGPU backend');
      if (values?.length !== H * 12 || !values.every(value => Number.isFinite(value) && Math.abs(value) <= 100)) throw Error('Invalid deformation state');
      qAttribute.array.set(values); qAttribute.needsUpdate = true;
      let radius = restRadius;
      for (let h = 0; h < H; h++) {
        let normSquared = 0;
        for (let j = 0; j < 12; j++) normSquared += qAttribute.array[h * 12 + j] ** 2;
        radius += modeEnvelope[h] * Math.sqrt(normSquared);
      }
      splat.boundingSphere.radius = radius * (1 + 1e-5) + 1e-5;
      renderer.compute(node);
      splat._sortInitialized = false;
    },
    // Buffers are owned by the renderer/device, disposed with the demo renderer.
    get outputAttributes() { return [buffers.centerRead.value, buffers.covarianceARead.value, buffers.covarianceBRead.value]; },
  };
}

export function deformPoint(rest, weights, q) {
  const out = Array.from(rest);
  for (let h = 0; h < weights.length; h++) for (let a = 0; a < 3; a++) {
    const k = h * 12 + a * 4;
    out[a] += weights[h] * (q[k] * rest[0] + q[k + 1] * rest[1] + q[k + 2] * rest[2] + q[k + 3]);
  }
  return out;
}
