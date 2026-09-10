import * as THREE from 'three/webgpu';
import {
  Fn, float, uint, vec2, vec3, vec4, ivec2, uvec2,
  uniform, storage, instanceIndex, sin, cos, sqrt, max, dot,
  textureLoad, textureStore,
} from 'three/tsl';

/**
 * Forward angular-spectrum propagation of one fixed complex source spectrum.
 * No target image or target-plane data enter this module. All distances are SI.
 *
 * Input/output index = row*n+column, exactly as wave.mjs; row zero is the top
 * of the CPU image. No FFT shift, image flip, exposure or renormalization is
 * performed. The irrelevant common exp(i*k*z) carrier is omitted in both paths.
 *
 * Requires an initialized, native WebGPU Three.js r186 renderer. Each propagate
 * submits one initialization plus 2*log2(n) radix-2 DIT inverse-FFT dispatches.
 */
export function createWaveOptics(renderer, { n = 256, pitch = 10e-6, wavelength = 532e-9 } = {}) {
  const requireBackend = () => {
    if (THREE.REVISION !== '186' || renderer?.isWebGPURenderer !== true || renderer.backend?.isWebGPUBackend !== true || !renderer.backend.device) {
      throw new Error('Wave optics requires initialized Three.js r186 WebGPU. Call await renderer.init(); WebGL fallback is unsupported.');
    }
  };
  requireBackend();
  if (![128, 256].includes(n) || !Number.isFinite(pitch) || pitch < 0.5e-6 || pitch > 1e-3 || !Number.isFinite(wavelength) || wavelength < 200e-9 || wavelength > 2e-6) {
    throw new RangeError('Wave optics bounds: n=128 or 256, pitch [0.5 µm,1 mm], wavelength [200 nm,2 µm].');
  }
  const count = n * n, bits = Math.log2(n), distance = uniform(0);
  const fixedAttribute = new THREE.StorageBufferAttribute(count, 2);
  fixedAttribute.name = 'Fixed source angular spectrum';
  const pingAttributes = [0, 1].map((i) => {
    const attribute = new THREE.StorageBufferAttribute(count, 2);
    attribute.name = `Wave inverse FFT ${i}`;
    return attribute;
  });
  const fixed = storage(fixedAttribute, 'vec2', count).toReadOnly();
  const ping = pingAttributes.map((attribute) => storage(attribute, 'vec2', count));
  const output = new THREE.StorageTexture(n, n);
  output.name = 'Unscaled propagated wave intensity';
  output.type = THREE.FloatType;
  output.format = THREE.RGBAFormat;
  output.minFilter = output.magFilter = THREE.NearestFilter;
  output.generateMipmaps = false;
  output.mipmapsAutoUpdate = false;
  output.flipY = false;
  output.colorSpace = THREE.NoColorSpace;

  const reverseBits = (input) => {
    let result = uint(0);
    for (let bit = 0; bit < bits; bit++) result = result.bitOr(input.shiftRight(uint(bit)).bitAnd(uint(1)).shiftLeft(uint(bits - 1 - bit)));
    return result;
  };
  const complexMultiply = (a, b) => vec2(a.x.mul(b.x).sub(a.y.mul(b.y)), a.x.mul(b.y).add(a.y.mul(b.x)));
  const passes = [];
  const initialize = Fn(() => {
    const x = instanceIndex.mod(uint(n)), y = instanceIndex.div(uint(n));
    // Reverse BOTH axes before the separable DIT passes. Transfer frequencies
    // belong to the original spectrum coordinates, not the shuffled destination.
    const sx = reverseBits(x).toVar(), sy = reverseBits(y).toVar();
    const sourceIndex = sy.mul(uint(n)).add(sx);
    const fx = sx.lessThan(uint(n / 2)).select(float(sx), float(sx).sub(n)).mul(1 / (n * pitch)).toVar();
    const fy = sy.lessThan(uint(n / 2)).select(float(sy), float(sy).sub(n)).mul(1 / (n * pitch)).toVar();
    const s = fx.mul(fx).add(fy.mul(fy)).mul(wavelength * wavelength).toVar();
    // Rationalization avoids subtracting almost equal numbers and removes the
    // large common carrier, matching the independent CPU transfer() convention.
    const phase = s.mul(distance).mul(-2 * Math.PI / wavelength).div(sqrt(max(float(1).sub(s), 0)).add(1)).toVar();
    const transfer = vec2(cos(phase), sin(phase));
    const value = complexMultiply(fixed.element(sourceIndex), transfer).toVar();
    ping[0].element(instanceIndex).assign(s.lessThan(1).select(value, vec2(0)));
  })().compute(count, [64]);
  initialize.name = 'Wave transfer and two-axis bit reversal';
  passes.push(initialize);

  let sourceBuffer = 0;
  for (let axis = 0; axis < 2; axis++) for (let length = 2; length <= n; length *= 2) {
    // Capture these bindings per stage: Fn bodies are evaluated lazily by TSL.
    const inputBuffer = sourceBuffer, outputBuffer = 1 - sourceBuffer;
    const half = length / 2, last = axis === 1 && length === n;
    const stage = Fn(() => {
      const x = instanceIndex.mod(uint(n)), y = instanceIndex.div(uint(n));
      const coordinate = axis === 0 ? x : y;
      const subIndex = coordinate.mod(uint(length)).toVar();
      const j = subIndex.mod(uint(half)).toVar();
      const firstCoordinate = coordinate.sub(subIndex).add(j).toVar();
      const firstIndex = axis === 0 ? y.mul(uint(n)).add(firstCoordinate) : firstCoordinate.mul(uint(n)).add(x);
      const secondIndex = firstIndex.add(uint(axis === 0 ? half : half * n));
      const a = ping[inputBuffer].element(firstIndex).toVar();
      const b = ping[inputBuffer].element(secondIndex).toVar();
      const angle = float(j).mul(2 * Math.PI / length).toVar();
      const rotated = complexMultiply(b, vec2(cos(angle), sin(angle))).toVar();
      const sign = subIndex.lessThan(uint(half)).select(float(1), float(-1));
      const value = a.add(rotated.mul(sign)).mul(last ? 1 / count : 1).toVar();
      ping[outputBuffer].element(instanceIndex).assign(value);
      if (last) {
        const intensity = dot(value, value).toVar();
        textureStore(output, uvec2(x, y), vec4(vec3(intensity), 1)).toWriteOnly();
      }
    })().compute(count, [64]);
    stage.name = `Wave inverse FFT ${axis === 0 ? 'x' : 'y'} length ${length}`;
    passes.push(stage);
    sourceBuffer = outputBuffer;
  }
  const finalBuffer = sourceBuffer, spareBuffer = 1 - finalBuffer;
  // Debug readback copies the ACTUAL intensity texture into the idle scratch
  // buffer. It is not a CPU reconstruction from the complex readback.
  const copyIntensity = Fn(() => {
    const x = instanceIndex.mod(uint(n)), y = instanceIndex.div(uint(n));
    const value = textureLoad(output, ivec2(x, y), 0).r;
    ping[spareBuffer].element(instanceIndex).assign(vec2(value, 0));
  })().compute(count, [64]);
  copyIntensity.name = 'Read actual wave intensity texture';

  // In the installed r186 renderer, bare BufferAttribute.dispose() dispatches
  // an event but standalone storage allocations have no disposal subscriber.
  // A once-rendered ownership geometry provides the supported geometry cleanup
  // path, which explicitly deletes every attached attribute from the renderer.
  const ownerGeometry = new THREE.BufferGeometry();
  ownerGeometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  ownerGeometry.setAttribute('waveFixedStorage', fixedAttribute);
  ownerGeometry.setAttribute('wavePingStorage', pingAttributes[0]);
  ownerGeometry.setAttribute('wavePongStorage', pingAttributes[1]);
  const ownerMaterial = new THREE.MeshBasicNodeMaterial({ colorWrite: false, depthTest: false, depthWrite: false, toneMapped: false });
  ownerMaterial.fragmentNode = vec4(0);
  const ownerMesh = new THREE.Mesh(ownerGeometry, ownerMaterial);
  ownerMesh.frustumCulled = false;
  const ownerScene = new THREE.Scene(); ownerScene.add(ownerMesh);
  const ownerCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 2);
  ownerCamera.position.z = 1;
  const ownerTarget = new THREE.RenderTarget(1, 1, { depthBuffer: false, stencilBuffer: false, samples: 0 });
  let ownershipRegistered = false, disposed = false, cleaned = false, reading = false, hasSource = false, hasField = false;

  const alive = () => { if (disposed) throw new Error('Wave optics has been disposed.'); requireBackend(); };
  const idle = () => { alive(); if (reading) throw new Error('Wait for the active wave readback before modifying or reading the field.'); };
  function registerOwnership() {
    if (ownershipRegistered) return;
    const oldTarget = renderer.getRenderTarget(), oldFace = renderer.getActiveCubeFace(), oldLevel = renderer.getActiveMipmapLevel();
    const oldAutoClear = renderer.autoClear, oldXR = renderer.xr?.enabled;
    try {
      if (renderer.xr) renderer.xr.enabled = false;
      renderer.autoClear = false;
      renderer.setRenderTarget(ownerTarget);
      renderer.render(ownerScene, ownerCamera);
      ownershipRegistered = true;
    } finally {
      renderer.setRenderTarget(oldTarget, oldFace, oldLevel);
      renderer.autoClear = oldAutoClear;
      if (renderer.xr) renderer.xr.enabled = oldXR;
    }
  }

  function updateSource(spectrum) {
    idle();
    if (!(spectrum instanceof Float64Array) || spectrum.length !== count * 2 || !spectrum.every((v) => Number.isFinite(v) && Math.abs(v) <= count)) {
      throw new RangeError('Expected a finite interleaved Float64 spectrum for a source with at most unit sample amplitude.');
    }
    fixedAttribute.array.set(spectrum);
    fixedAttribute.needsUpdate = true;
    hasSource = true; hasField = false;
  }

  function propagate(distanceMeters) {
    idle();
    if (!hasSource) throw new Error('Call updateSource() before propagation.');
    if (!Number.isFinite(distanceMeters) || Math.abs(distanceMeters) > 0.5) throw new RangeError('Propagation distance must be finite and within ±0.5 metres.');
    registerOwnership();
    distance.value = distanceMeters;
    renderer.compute(passes);
    hasField = true;
    return output;
  }

  async function read(kind) {
    idle();
    if (!hasField) throw new Error('Propagate the current source before reading it.');
    reading = true;
    try {
      if (kind === 'intensity') renderer.compute(copyIntensity);
      const index = kind === 'intensity' ? spareBuffer : finalBuffer;
      // No retained ReadbackBuffer: Three copies, unmaps and destroys its
      // temporary staging allocation before this promise resolves.
      const values = new Float32Array(await renderer.getArrayBufferAsync(pingAttributes[index]));
      if (values.length !== count * 2 || !values.every(Number.isFinite)) throw new Error('GPU wave readback contains invalid values.');
      if (kind === 'complex') return values;
      const pixels = new Float32Array(count);
      for (let i = 0; i < count; i++) {
        if (values[2 * i] < 0) throw new Error('GPU wave intensity is negative.');
        pixels[i] = values[2 * i];
      }
      return pixels;
    } finally {
      reading = false;
      if (disposed) cleanup();
    }
  }

  function cleanup() {
    if (cleaned) return;
    cleaned = true;
    for (const pass of passes) pass.dispose();
    copyIntensity.dispose();
    ownerGeometry.dispose();
    ownerMaterial.dispose();
    ownerTarget.dispose();
    output.dispose();
    ownerScene.clear();
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    // A pending staging copy must finish before its source allocation is freed.
    if (!reading) cleanup();
  }
  return {
    texture: output, updateSource, propagate,
    readComplex: () => read('complex'), readIntensity: () => read('intensity'), dispose,
    metadata: Object.freeze({ n, pitch, wavelength, computePasses: passes.length, complexType: 'float32', textureType: 'rgba32float', rowOrder: 'CPU row zero first; no flip' }),
  };
}
