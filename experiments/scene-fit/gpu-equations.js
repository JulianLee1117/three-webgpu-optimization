// Central finite-difference Gauss-Newton equations for the bounded 128², 8D screen.
// Residual r = center - reference. H = mean(JᵀJ), g = mean(Jᵀr), over RGB.
// For the corresponding MSE, gradient = 2g and GN Hessian = 2H; solve Hδ = -g.
// GPU readback is 44 floats (176 bytes), never a residual/Jacobian image.

const SIDE = 128;
const DIMENSION = 8;
const LAYERS = DIMENSION * 2;
const UPPER_TRIANGLE = DIMENSION * (DIMENSION + 1) / 2;
const COEFFICIENTS = UPPER_TRIANGLE + DIMENSION;
const GROUPS = (SIDE / 8) ** 2;
const OUTPUT_BYTES = COEFFICIENTS * 4;

const firstShader = /* wgsl */`
struct Parameters {
  lower: vec4f,
  upper: vec4f,
}
@group(0) @binding(0) var reference: texture_2d<f32>;
@group(0) @binding(1) var center: texture_2d<f32>;
@group(0) @binding(2) var perturbations: texture_2d_array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;
@group(0) @binding(4) var<storage,read_write> partials: array<f32>;

// 44 coefficient reductions × 64 lanes × 4 bytes = 11,264 workgroup bytes.
var<workgroup> sums: array<f32,2816>;

fn denominator(axis: u32) -> f32 {
  if (axis < 4u) { return parameters.lower[axis]; }
  return parameters.upper[axis - 4u];
}

@compute @workgroup_size(8,8)
fn main(@builtin(workgroup_id) group: vec3u,
        @builtin(local_invocation_id) local: vec3u) {
  let pixel = vec2i(group.xy * 8u + local.xy);
  let lane = local.y * 8u + local.x;
  let residual = textureLoad(center,pixel,0).rgb - textureLoad(reference,pixel,0).rgb;
  var columns: array<vec3f,8>;
  for (var axis = 0u; axis < 8u; axis++) {
    let minus = textureLoad(perturbations,pixel,i32(axis * 2u),0).rgb;
    let plus = textureLoad(perturbations,pixel,i32(axis * 2u + 1u),0).rgb;
    columns[axis] = (plus - minus) / denominator(axis);
  }

  // Coefficients 0..35: row-major upper triangle, including the diagonal.
  // Coefficients 36..43: Jᵀr in parameter order.
  var coefficient = 0u;
  for (var row = 0u; row < 8u; row++) {
    for (var column = row; column < 8u; column++) {
      sums[coefficient * 64u + lane] = dot(columns[row],columns[column]);
      coefficient++;
    }
    sums[(36u + row) * 64u + lane] = dot(columns[row],residual);
  }
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      for (var term = 0u; term < 44u; term++) {
        let index = term * 64u + lane;
        sums[index] = sums[index] + sums[index + stride];
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let groupIndex = group.y * 16u + group.x;
    for (var term = 0u; term < 44u; term++) {
      partials[term * 256u + groupIndex] = sums[term * 64u];
    }
  }
}`;

const secondShader = /* wgsl */`
@group(0) @binding(0) var<storage,read> partials: array<f32>;
@group(0) @binding(1) var<storage,read_write> result: array<f32>;
var<workgroup> sums: array<f32,256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3u,
        @builtin(local_invocation_index) lane: u32) {
  let coefficient = group.x;
  sums[lane] = partials[coefficient * 256u + lane];
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { sums[lane] = sums[lane] + sums[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u) { result[coefficient] = sums[0] / (128.0 * 128.0 * 3.0); }
}`;

function checkTexture(texture, name, usage) {
  if (!texture || texture.width !== SIDE || texture.height !== SIDE ||
      texture.depthOrArrayLayers !== 1 || texture.dimension !== '2d' ||
      texture.sampleCount !== 1 || texture.format !== 'rgba8unorm') {
    throw Error(`${name} must be a single-layer, single-sample 128x128 rgba8unorm GPUTexture.`);
  }
  if ((texture.usage & usage) !== usage) throw Error(`${name} is missing its required GPUTexture usage.`);
}

function packDenominators(denominators) {
  if (!denominators || denominators.length !== DIMENSION) throw Error('Provide eight plus-minus parameter denominators.');
  const packed = new Float32Array(DIMENSION);
  for (let axis = 0; axis < DIMENSION; axis++) {
    const value = denominators[axis];
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || value > 1) {
      throw Error(`Denominator ${axis} must be finite and in (0, 1].`);
    }
    packed[axis] = value;
    if (packed[axis] <= 0 || !Number.isFinite(Math.fround(1 / packed[axis]))) {
      throw Error(`Denominator ${axis} is too small for finite float32 arithmetic.`);
    }
  }
  return packed;
}

export class GpuNormalEquations {
  /** referenceNativeTexture is borrowed and must remain alive until dispose(). */
  constructor(device, referenceNativeTexture, side = SIDE, dimension = DIMENSION) {
    if (side !== SIDE || dimension !== DIMENSION) throw Error('Normal equations support only side=128 and dimension=8.');
    checkTexture(referenceNativeTexture, 'Reference texture', GPUTextureUsage.TEXTURE_BINDING);
    if (device.limits.maxComputeWorkgroupStorageSize < 11264 || device.limits.maxComputeInvocationsPerWorkgroup < 256) {
      throw Error('Device limits do not support the bounded normal-equation reductions.');
    }
    this.device = device;
    this.readbacks = 0;
    this.readbackBytes = 0;
    this._owned = [];
    this._disposed = false;
    this._busy = false;
    this._captureMask = 0;
    const own = resource => { this._owned.push(resource); return resource; };
    try {
      this.perturbations = own(device.createTexture({
        label: 'scene-fit-finite-difference-layers',
        size: { width: SIDE, height: SIDE, depthOrArrayLayers: LAYERS },
        dimension: '2d', format: 'rgba8unorm', mipLevelCount: 1, sampleCount: 1,
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING
      }));
      this.parameters = own(device.createBuffer({ label: 'scene-fit-fd-denominators', size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
      this.partials = own(device.createBuffer({ label: 'scene-fit-equation-partials', size: COEFFICIENTS * GROUPS * 4,
        usage: GPUBufferUsage.STORAGE }));
      this.output = own(device.createBuffer({ label: 'scene-fit-equation-output', size: OUTPUT_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
      this.staging = own(device.createBuffer({ label: 'scene-fit-equation-readback', size: OUTPUT_BYTES,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
      this.first = device.createComputePipeline({ label: 'scene-fit-equations-pixels', layout: 'auto',
        compute: { module: device.createShaderModule({ label: 'scene-fit-equations-pixels', code: firstShader }), entryPoint: 'main' } });
      this.second = device.createComputePipeline({ label: 'scene-fit-equations-coefficients', layout: 'auto',
        compute: { module: device.createShaderModule({ label: 'scene-fit-equations-coefficients', code: secondShader }), entryPoint: 'main' } });
      this.referenceView = referenceNativeTexture.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1 });
      this.perturbationView = this.perturbations.createView({ dimension: '2d-array', baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: LAYERS });
      this.secondBindings = device.createBindGroup({ layout: this.second.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.partials } },
        { binding: 1, resource: { buffer: this.output } }
      ] });
    } catch (error) {
      this.dispose();
      throw error;
    }
  }

  /**
   * Encode a copy: layer 2*i is minus, layer 2*i+1 is plus.
   * The caller submits this encoder before evaluate(). All 16 layers must be
   * captured afresh per evaluation. The queue orders these copies before reads.
   */
  capture(encoder, candidateNativeTexture, layer) {
    if (this._disposed) throw Error('Normal equations have been disposed.');
    if (this._busy) throw Error('Await the current equation evaluation before capturing another batch.');
    if (!Number.isInteger(layer) || layer < 0 || layer >= LAYERS) throw Error('Capture layer must be an integer from 0 through 15.');
    checkTexture(candidateNativeTexture, 'Perturbed candidate texture', GPUTextureUsage.COPY_SRC);
    encoder.copyTextureToTexture(
      { texture: candidateNativeTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
      { texture: this.perturbations, mipLevel: 0, origin: { x: 0, y: 0, z: layer } },
      { width: SIDE, height: SIDE, depthOrArrayLayers: 1 }
    );
    this._captureMask |= 1 << layer;
  }

  /**
   * Return { hessian: Array(64), gradient: Array(8) }; H is symmetric row-major.
   * denominators[i] = min(1,x[i]+step) - max(0,x[i]-step), not assumed 2*step.
   * The center image must be rendered at the unperturbed x. Concurrent evaluate
   * calls and captures during evaluation are rejected because storage is reused.
   */
  async evaluate(centerNativeTexture, denominators) {
    if (this._disposed) throw Error('Normal equations have been disposed.');
    if (this._busy) throw Error('Normal-equation evaluations must be serialized.');
    if (this._captureMask !== 0xffff) throw Error('Capture and submit all 16 perturbation layers before evaluating.');
    checkTexture(centerNativeTexture, 'Center texture', GPUTextureUsage.TEXTURE_BINDING);
    const packed = packDenominators(denominators);
    this._busy = true;
    this._captureMask = 0;
    try {
      this.device.queue.writeBuffer(this.parameters, 0, packed);
      const firstBindings = this.device.createBindGroup({ layout: this.first.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.referenceView },
        { binding: 1, resource: centerNativeTexture.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1 }) },
        { binding: 2, resource: this.perturbationView },
        { binding: 3, resource: { buffer: this.parameters } },
        { binding: 4, resource: { buffer: this.partials } }
      ] });
      const encoder = this.device.createCommandEncoder({ label: 'scene-fit-normal-equations' });
      let pass = encoder.beginComputePass();
      pass.setPipeline(this.first); pass.setBindGroup(0, firstBindings); pass.dispatchWorkgroups(16, 16); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(this.second); pass.setBindGroup(0, this.secondBindings); pass.dispatchWorkgroups(COEFFICIENTS); pass.end();
      encoder.copyBufferToBuffer(this.output, 0, this.staging, 0, OUTPUT_BYTES);
      this.device.queue.submit([encoder.finish()]);
      await this.staging.mapAsync(GPUMapMode.READ, 0, OUTPUT_BYTES);
      if (this._disposed) throw Error('Normal equations were disposed during evaluation.');
      const coefficients = new Float32Array(this.staging.getMappedRange(0, OUTPUT_BYTES)).slice();
      this.readbacks++;
      this.readbackBytes += OUTPUT_BYTES;
      if (!coefficients.every(Number.isFinite)) throw Error('Nonfinite GPU normal-equation coefficient.');
      const hessian = new Array(DIMENSION * DIMENSION).fill(0);
      let index = 0;
      for (let row = 0; row < DIMENSION; row++) for (let column = row; column < DIMENSION; column++) {
        const value = coefficients[index++];
        if (row === column && value < 0) throw Error('Negative GPU normal-equation diagonal.');
        hessian[row * DIMENSION + column] = value;
        hessian[column * DIMENSION + row] = value;
      }
      return { hessian, gradient: Array.from(coefficients.subarray(UPPER_TRIANGLE)) };
    } finally {
      if (this.staging.mapState === 'mapped') this.staging.unmap();
      this._busy = false;
    }
  }

  /** Destroys only owned texture/buffers. Borrowed reference/candidate textures survive. */
  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    for (const resource of this._owned) resource.destroy();
    this._owned.length = 0;
    this.referenceView = this.perturbationView = this.secondBindings = null;
    this.first = this.second = null;
  }
}
