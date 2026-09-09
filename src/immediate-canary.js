export const IMMEDIATE_WGSL_FEATURE = 'immediate_address_space';
export const REQUIRED_IMMEDIATE_BYTES = 4;
export const DIRECT_IMMEDIATE_SEQUENCE = Object.freeze([2, 0, 3]);
export const BUNDLE_IMMEDIATE_SEQUENCE = Object.freeze([1, 3, 0]);
export const BUNDLE_MUTATED_IMMEDIATE_SEQUENCE = Object.freeze([3, 0, 2]);
export const INSTANCE_INDEX_SEQUENCE = Object.freeze([0, 1, 2]);
export const IMMEDIATE_CANARY_VALUES = Object.freeze([
  0x1020_3040,
  0x2468_ace0,
  0x1357_9bdf,
  0xfedc_ba98,
  0x89ab_cdef,
  0x55aa_9966,
]);

const CANARY_KIND = 'webgpu-immediates-correctness-canary';
const CANARY_SCHEMA_VERSION = 2;
const TARGET_FORMAT = 'r32uint';
const INDIRECT_FIRST_INSTANCE_FEATURE = 'indirect-first-instance';
const RAW_CANARY_SCOPE = 'raw WebGPU immediate-data capability and correctness only';

function requireCondition(condition, message, detail = undefined) {
  if (condition) return;
  const error = new Error(message);
  if (detail !== undefined) error.detail = structuredClone(detail);
  throw error;
}

function requireCapability(condition, message, detail = undefined) {
  if (condition) return;
  const error = new Error(message);
  error.canaryStatus = 'unsupported';
  if (detail !== undefined) error.detail = structuredClone(detail);
  throw error;
}

function serializeError(error) {
  return {
    name: String(error?.name ?? 'Error').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
    stack: error?.stack === undefined || error?.stack === null
      ? null
      : String(error.stack).slice(0, 8_192),
    detail: error?.detail === undefined ? null : structuredClone(error.detail),
    canaryStatus: error?.canaryStatus === 'unsupported' ? 'unsupported' : null,
  };
}

function serializeGpuError(error) {
  return {
    name: String(error?.constructor?.name ?? error?.name ?? 'GPUError').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
  };
}

function sortedFeatureNames(features) {
  if (features === null || features === undefined
    || typeof features[Symbol.iterator] !== 'function') return [];
  return [...features].map(String).sort();
}

function sameArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

export function isDistinctNonMonotonicSequence(values) {
  if (!Array.isArray(values) || values.length < 3 || new Set(values).size !== values.length) {
    return false;
  }
  let rose = false;
  let fell = false;
  for (let index = 1; index < values.length; index += 1) {
    rose ||= values[index] > values[index - 1];
    fell ||= values[index] < values[index - 1];
  }
  return rose && fell;
}

function valuesAtAddresses(addresses) {
  return addresses.map((address) => IMMEDIATE_CANARY_VALUES[address]);
}

function addressesForImmediates(immediates) {
  return immediates.map((base, index) => base + INSTANCE_INDEX_SEQUENCE[index]);
}

async function serializeAdapterInfo(adapter) {
  let info = adapter?.info ?? null;
  if (info === null && typeof adapter?.requestAdapterInfo === 'function') {
    info = await adapter.requestAdapterInfo();
  }
  if (info === null || typeof info !== 'object') return null;
  const result = {};
  for (const field of [
    'vendor',
    'architecture',
    'device',
    'description',
    'backend',
    'type',
    'driver',
    'isFallbackAdapter',
  ]) {
    const value = info[field];
    if (typeof value === 'string' || typeof value === 'boolean') result[field] = value;
  }
  return result;
}

export function buildImmediateCanaryShader() {
  return `requires immediate_address_space;

var<immediate> immediateBase : u32;

@group(0) @binding(0)
var<storage, read> values : array<u32>;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) @interpolate(flat) value : u32,
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex : u32,
  @builtin(instance_index) instanceIndex : u32,
) -> VertexOutput {
  let positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
  );
  let pixelCenter = -2.0 / 3.0 + (2.0 / 3.0) * f32(instanceIndex);
  var output : VertexOutput;
  output.position = vec4f(
    positions[vertexIndex].x / 3.0 + pixelCenter,
    positions[vertexIndex].y,
    0.0,
    1.0,
  );
  output.value = values[immediateBase + instanceIndex];
  return output;
}

@fragment
fn fragmentMain(input : VertexOutput) -> @location(0) u32 {
  return input.value;
}
`;
}

export function assessImmediateCapabilities(capabilities) {
  const reasons = [];
  if (capabilities?.navigatorGpu !== true) reasons.push('navigator.gpu is unavailable');
  if (capabilities?.wgslImmediateAddressSpace !== true) {
    reasons.push(`WGSL feature ${IMMEDIATE_WGSL_FEATURE} is unavailable`);
  }
  if (capabilities?.adapterAvailable !== true) reasons.push('WebGPU adapter is unavailable');
  if (capabilities?.nonFallbackAdapter !== true) {
    reasons.push('adapter is fallback or fallback status is unavailable');
  }
  if (!Number.isFinite(capabilities?.adapterMaxImmediateSize)
    || capabilities.adapterMaxImmediateSize < REQUIRED_IMMEDIATE_BYTES) {
    reasons.push('adapter maxImmediateSize is below four bytes or unavailable');
  }
  if (!Number.isFinite(capabilities?.deviceMaxImmediateSize)
    || capabilities.deviceMaxImmediateSize < REQUIRED_IMMEDIATE_BYTES) {
    reasons.push('device maxImmediateSize is below four bytes or unavailable');
  }
  if (capabilities?.renderPassSetImmediates !== true) {
    reasons.push('GPURenderPassEncoder.setImmediates is unavailable');
  }
  if (capabilities?.renderBundleSetImmediates !== true) {
    reasons.push('GPURenderBundleEncoder.setImmediates is unavailable');
  }
  return {
    pass: reasons.length === 0,
    requiredImmediateBytes: REQUIRED_IMMEDIATE_BYTES,
    reasons,
  };
}

export function assessThreeLaneEligibility(capabilities) {
  const reasons = [];
  const immediateCapability = assessImmediateCapabilities(capabilities);
  if (!immediateCapability.pass) reasons.push('raw immediate capability did not pass');
  if (capabilities?.indirectFirstInstanceAdapter !== true) {
    reasons.push(`${INDIRECT_FIRST_INSTANCE_FEATURE} is unavailable on the adapter`);
  }
  if (capabilities?.indirectFirstInstanceDevice !== true) {
    reasons.push(`${INDIRECT_FIRST_INSTANCE_FEATURE} was not enabled on the device`);
  }
  return {
    pass: reasons.length === 0,
    reasons,
    immediateCapabilityPass: immediateCapability.pass,
    indirectFirstInstanceAdapter: capabilities?.indirectFirstInstanceAdapter === true,
    indirectFirstInstanceDevice: capabilities?.indirectFirstInstanceDevice === true,
  };
}

function validExpectedRejection(control) {
  return control?.pass === true
    && control?.rejected === true
    && control?.synchronousException === null
    && Array.isArray(control?.scopedErrors)
    && control.scopedErrors.length === 1
    && control.scopedErrors[0]?.filter === 'validation'
    && control.scopedErrors[0]?.name === 'GPUValidationError'
    && control.scopedErrors[0]?.popFailed !== true
    && Array.isArray(control?.unexpectedScopedErrors)
    && control.unexpectedScopedErrors.length === 0
    && control.scopedErrors.some((error) => error.filter === 'validation');
}

function isSortedStringArray(values) {
  return Array.isArray(values)
    && values.every((value) => typeof value === 'string')
    && values.every((value, index) => index === 0 || values[index - 1] <= value);
}

function sameRecord(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function validRangeRecord(range, byteOffset) {
  return range?.byteOffset === byteOffset
    && range?.byteLength === REQUIRED_IMMEDIATE_BYTES;
}

export function validateImmediateCanaryResult(result) {
  const reasons = [];
  const capabilities = assessImmediateCapabilities(result?.capabilities);
  const threeLaneEligibility = assessThreeLaneEligibility(result?.capabilities);
  if (!capabilities.pass) reasons.push(...capabilities.reasons);
  const schemaVersion = result?.schemaVersion;
  if ((schemaVersion !== 1 && schemaVersion !== CANARY_SCHEMA_VERSION)
    || result?.kind !== CANARY_KIND
    || result?.scope !== RAW_CANARY_SCOPE) {
    reasons.push('canary result identity is invalid');
  }
  if (result?.status !== 'pass' || result?.failure !== null) {
    reasons.push('page canary did not report a clean pass');
  }
  if (result?.executionMode !== 'technical-canary'
    || result?.phase !== 'phase-0-raw-capability-canary'
    || result?.fullPhase0Pass !== false
    || result?.integratedCanaryInvoked !== false
    || result?.analysisEligible !== false
    || result?.efficacyAnalysisAllowed !== false
    || result?.numericalDecision !== null
    || result?.timingCaptured !== false
    || result?.efficacyEvaluated !== false) {
    reasons.push('correctness-only evidence boundary is invalid');
  }

  const rawCapabilities = result?.capabilities;
  const wgslLanguageFeatures = rawCapabilities?.wgslLanguageFeatures;
  const adapterFeatures = result?.adapterFeatures;
  const deviceFeatures = result?.deviceFeatures;
  if (!isSortedStringArray(wgslLanguageFeatures)
    || !wgslLanguageFeatures.includes(IMMEDIATE_WGSL_FEATURE)
    || rawCapabilities?.wgslImmediateAddressSpace
      !== wgslLanguageFeatures.includes(IMMEDIATE_WGSL_FEATURE)
    || !isSortedStringArray(adapterFeatures)
    || rawCapabilities?.indirectFirstInstanceAdapter
      !== adapterFeatures.includes(INDIRECT_FIRST_INSTANCE_FEATURE)
    || !isSortedStringArray(deviceFeatures)
    || rawCapabilities?.indirectFirstInstanceDevice
      !== deviceFeatures.includes(INDIRECT_FIRST_INSTANCE_FEATURE)
    || result?.adapterInfo === null
    || typeof result?.adapterInfo !== 'object'
    || result.adapterInfo.isFallbackAdapter !== false
    || rawCapabilities?.nonFallbackAdapter !== true
    || typeof result?.userAgent !== 'string'
    || result.userAgent.length === 0
    || result?.secureContext !== true
    || result?.crossOriginIsolated !== true) {
    reasons.push('reported capability fields are inconsistent or incomplete');
  }
  if (schemaVersion === CANARY_SCHEMA_VERSION) {
    const expectedRequestedFeatures = rawCapabilities?.indirectFirstInstanceAdapter === true
      ? [INDIRECT_FIRST_INSTANCE_FEATURE]
      : [];
    if (!sameArray(result?.requestedDeviceFeatures, expectedRequestedFeatures)
      || !sameRecord(result?.capabilityAssessment, capabilities)
      || !sameRecord(result?.threeLaneEligibility, threeLaneEligibility)) {
      reasons.push('capability and three-lane assessment records are inconsistent');
    }
  }
  if (!Array.isArray(result?.shaderCompilationMessages)
    || result.shaderCompilationMessages.some((message) => message?.type === 'error')
    || result?.pipelineLayoutDescriptor?.bindGroupLayoutCount !== 1
    || result?.pipelineLayoutDescriptor?.immediateSize !== REQUIRED_IMMEDIATE_BYTES) {
    reasons.push('shader compilation or explicit immediate pipeline layout is invalid');
  }
  const direct = result?.canaries?.directRenderPass;
  if (direct?.pass !== true
    || direct?.renderPassSetImmediates !== true
    || rawCapabilities?.renderPassSetImmediates !== direct.renderPassSetImmediates
    || direct?.addressExpression !== 'immediateBase + instanceIndex'
    || !isDistinctNonMonotonicSequence(direct?.immediateU32Sequence)
    || !sameArray(direct?.immediateU32Sequence, DIRECT_IMMEDIATE_SEQUENCE)
    || !sameArray(direct?.instanceIndexSequence, INSTANCE_INDEX_SEQUENCE)
    || !sameArray(
      direct?.addressSequence,
      addressesForImmediates(DIRECT_IMMEDIATE_SEQUENCE),
    )
    || !sameArray(
      direct?.expectedU32Sequence,
      valuesAtAddresses(addressesForImmediates(DIRECT_IMMEDIATE_SEQUENCE)),
    )
    || !sameArray(direct?.observedU32Sequence, direct?.expectedU32Sequence)) {
    reasons.push('direct render-pass non-monotonic address/output canary failed');
  }
  const bundle = result?.canaries?.renderBundleSnapshot;
  if (bundle?.pass !== true
    || bundle?.renderBundleSetImmediates !== true
    || rawCapabilities?.renderBundleSetImmediates !== bundle.renderBundleSetImmediates
    || bundle?.addressExpression !== 'immediateBase + instanceIndex'
    || !isDistinctNonMonotonicSequence(bundle?.immediateU32SequenceAtEncoding)
    || !sameArray(bundle?.immediateU32SequenceAtEncoding, BUNDLE_IMMEDIATE_SEQUENCE)
    || !sameArray(
      bundle?.sourceU32SequenceAfterFinish,
      BUNDLE_MUTATED_IMMEDIATE_SEQUENCE,
    )
    || bundle?.sourceDetachedAfterMutation !== true
    || bundle?.sourceByteLengthAfterDetach !== 0
    || !sameArray(bundle?.instanceIndexSequence, INSTANCE_INDEX_SEQUENCE)
    || !sameArray(
      bundle?.addressSequenceAtEncoding,
      addressesForImmediates(BUNDLE_IMMEDIATE_SEQUENCE),
    )
    || !sameArray(
      bundle?.expectedU32Sequence,
      valuesAtAddresses(addressesForImmediates(BUNDLE_IMMEDIATE_SEQUENCE)),
    )
    || !sameArray(
      bundle?.mutatedSourceWouldProduceU32Sequence,
      valuesAtAddresses(addressesForImmediates(BUNDLE_MUTATED_IMMEDIATE_SEQUENCE)),
    )
    || !sameArray(bundle?.observedU32Sequence, bundle?.expectedU32Sequence)
    || bundle?.observedU32Sequence.some(
      (value, index) => value === bundle.mutatedSourceWouldProduceU32Sequence[index],
    )
    || bundle?.snapshotPreserved !== true) {
    reasons.push('render-bundle non-monotonic snapshot canary failed');
  }
  const negativeControls = result?.negativeControls;
  const unsetControl = negativeControls?.unsetRequiredSlotAfterBundle;
  if (!validExpectedRejection(unsetControl)
    || unsetControl?.name !== 'unset-required-slot-after-bundle'
    || !validRangeRecord(unsetControl?.omittedRange, 0)
    || unsetControl?.priorBundleSetSameRange !== true
    || unsetControl?.directStateRestoredAfterExecuteBundles?.pipeline !== true
    || unsetControl?.directStateRestoredAfterExecuteBundles?.bindGroup0 !== true) {
    reasons.push('unsetRequiredSlotAfterBundle negative control was not cleanly rejected');
  }
  const misalignedControl = negativeControls?.misalignedRange;
  if (!validExpectedRejection(misalignedControl)
    || misalignedControl?.name !== 'misaligned-range'
    || !validRangeRecord(misalignedControl?.requestedRange, 2)
    || misalignedControl?.deviceMaxImmediateSize
      !== rawCapabilities?.deviceMaxImmediateSize) {
    reasons.push('misalignedRange negative control was not cleanly rejected');
  }
  const beyondControl = negativeControls?.rangeBeyondMaxImmediateSize;
  if (!validExpectedRejection(beyondControl)
    || beyondControl?.name !== 'range-beyond-max-immediate-size'
    || !validRangeRecord(
      beyondControl?.requestedRange,
      rawCapabilities?.deviceMaxImmediateSize,
    )
    || beyondControl?.deviceMaxImmediateSize
      !== rawCapabilities?.deviceMaxImmediateSize) {
    reasons.push('rangeBeyondMaxImmediateSize negative control was not cleanly rejected');
  }
  if (!Array.isArray(result?.deviceErrors?.uncapturedErrors)
    || result.deviceErrors.uncapturedErrors.length !== 0) {
    reasons.push('uncaptured WebGPU errors were observed');
  }
  if (!Array.isArray(result?.deviceErrors?.unexpectedDeviceLosses)
    || result.deviceErrors.unexpectedDeviceLosses.length !== 0) {
    reasons.push('unexpected WebGPU device loss was observed');
  }
  if (schemaVersion === CANARY_SCHEMA_VERSION
    && (!Array.isArray(result?.deviceErrors?.lifecycleScopedErrors)
      || result.deviceErrors.lifecycleScopedErrors.length !== 0
      || result?.deviceErrors?.finalDeviceLoss?.reason !== 'destroyed')) {
    reasons.push('device error-scope or final-loss completion is invalid');
  }
  return {
    pass: reasons.length === 0,
    reasons: [...new Set(reasons)],
    capabilities,
    threeLaneEligibility,
  };
}

async function withGpuErrorScopes(device, operation) {
  const filters = ['internal', 'out-of-memory', 'validation'];
  for (const filter of filters) device.pushErrorScope(filter);
  let value;
  let operationError = null;
  try {
    value = await operation();
  } catch (error) {
    operationError = error;
  }
  const scopedErrors = [];
  for (const filter of [...filters].reverse()) {
    try {
      const error = await device.popErrorScope();
      if (error !== null) scopedErrors.push({ filter, ...serializeGpuError(error) });
    } catch (error) {
      scopedErrors.push({ filter, popFailed: true, ...serializeError(error) });
    }
  }
  if (operationError !== null) throw operationError;
  requireCondition(scopedErrors.length === 0, 'WebGPU error scope captured an error.', {
    scopedErrors,
  });
  return value;
}

async function captureExpectedRejection(device, name, operation) {
  const filters = ['internal', 'out-of-memory', 'validation'];
  for (const filter of filters) device.pushErrorScope(filter);
  let synchronousException = null;
  try {
    await operation();
  } catch (error) {
    synchronousException = serializeError(error);
  }
  const scopedErrors = [];
  for (const filter of [...filters].reverse()) {
    try {
      const error = await device.popErrorScope();
      if (error !== null) scopedErrors.push({ filter, ...serializeGpuError(error) });
    } catch (error) {
      scopedErrors.push({ filter, popFailed: true, ...serializeError(error) });
    }
  }
  const validationErrors = scopedErrors.filter(
    (error) => error.filter === 'validation'
      && error.name === 'GPUValidationError'
      && error.popFailed !== true,
  );
  const unexpectedScopedErrors = scopedErrors.filter(
    (error) => error.filter !== 'validation'
      || error.name !== 'GPUValidationError'
      || error.popFailed === true,
  );
  const rejected = synchronousException === null && validationErrors.length === 1;
  return {
    name,
    pass: rejected
      && scopedErrors.length === 1
      && unexpectedScopedErrors.length === 0,
    rejected,
    synchronousException,
    scopedErrors,
    unexpectedScopedErrors,
  };
}

function createReadbackTarget(device, label, width) {
  const texture = device.createTexture({
    label: `${label}-target`,
    size: { width, height: 1, depthOrArrayLayers: 1 },
    format: TARGET_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: `${label}-readback`,
    size: 256,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return { texture, readback, width };
}

async function readU32Sequence(readback, count) {
  await readback.mapAsync(GPUMapMode.READ, 0, count * Uint32Array.BYTES_PER_ELEMENT);
  const copy = new Uint32Array(count);
  copy.set(new Uint32Array(readback.getMappedRange(0, count * 4)));
  readback.unmap();
  return [...copy];
}

function encodeCopy(commandEncoder, texture, readback, width) {
  commandEncoder.copyTextureToBuffer(
    { texture },
    { buffer: readback, bytesPerRow: 256, rowsPerImage: 1 },
    { width, height: 1, depthOrArrayLayers: 1 },
  );
}

function colorAttachment(texture) {
  return {
    view: texture.createView(),
    clearValue: { r: 0, g: 0, b: 0, a: 0 },
    loadOp: 'clear',
    storeOp: 'store',
  };
}

async function runDirectRenderPassCanary(device, pipeline, bindGroup) {
  const immediateSource = new Uint32Array(DIRECT_IMMEDIATE_SEQUENCE);
  const resources = createReadbackTarget(
    device,
    'direct-immediate-canary',
    immediateSource.length,
  );
  try {
    let renderPassSetImmediates = false;
    const observedU32Sequence = await withGpuErrorScopes(device, async () => {
      const commandEncoder = device.createCommandEncoder({
        label: 'direct-immediate-canary-commands',
      });
      const pass = commandEncoder.beginRenderPass({
        label: 'direct-immediate-canary-pass',
        colorAttachments: [colorAttachment(resources.texture)],
      });
      renderPassSetImmediates = typeof pass.setImmediates === 'function';
      requireCapability(renderPassSetImmediates,
        'GPURenderPassEncoder.setImmediates is unavailable.');
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let index = 0; index < immediateSource.length; index += 1) {
        pass.setImmediates(0, immediateSource, index, 1);
        pass.draw(6, 1, 0, index);
      }
      pass.end();
      encodeCopy(commandEncoder, resources.texture, resources.readback, resources.width);
      device.queue.submit([commandEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return readU32Sequence(resources.readback, immediateSource.length);
    });
    const addressSequence = addressesForImmediates(DIRECT_IMMEDIATE_SEQUENCE);
    const expectedU32Sequence = valuesAtAddresses(addressSequence);
    return {
      pass: sameArray(observedU32Sequence, expectedU32Sequence),
      renderPassSetImmediates,
      immediateU32Sequence: [...DIRECT_IMMEDIATE_SEQUENCE],
      instanceIndexSequence: [...INSTANCE_INDEX_SEQUENCE],
      addressExpression: 'immediateBase + instanceIndex',
      addressSequence,
      expectedU32Sequence,
      observedU32Sequence,
    };
  } finally {
    resources.texture.destroy();
    resources.readback.destroy();
  }
}

async function runRenderBundleSnapshotCanary(device, pipeline, bindGroup) {
  const immediateSource = new Uint32Array(BUNDLE_IMMEDIATE_SEQUENCE);
  const resources = createReadbackTarget(
    device,
    'bundle-immediate-canary',
    immediateSource.length,
  );
  try {
    let renderBundleSetImmediates = false;
    let sourceU32SequenceAfterFinish = null;
    let sourceDetachedAfterMutation = false;
    let sourceByteLengthAfterDetach = null;
    const observedU32Sequence = await withGpuErrorScopes(device, async () => {
      const bundleEncoder = device.createRenderBundleEncoder({
        label: 'bundle-immediate-canary-encoder',
        colorFormats: [TARGET_FORMAT],
      });
      renderBundleSetImmediates = typeof bundleEncoder.setImmediates === 'function';
      requireCapability(renderBundleSetImmediates,
        'GPURenderBundleEncoder.setImmediates is unavailable.');
      bundleEncoder.setPipeline(pipeline);
      bundleEncoder.setBindGroup(0, bindGroup);
      for (let index = 0; index < immediateSource.length; index += 1) {
        bundleEncoder.setImmediates(0, immediateSource, index, 1);
        bundleEncoder.draw(6, 1, 0, index);
      }
      const bundle = bundleEncoder.finish({ label: 'bundle-immediate-canary' });

      immediateSource.set(BUNDLE_MUTATED_IMMEDIATE_SEQUENCE);
      sourceU32SequenceAfterFinish = [...immediateSource];
      structuredClone(immediateSource.buffer, { transfer: [immediateSource.buffer] });
      sourceDetachedAfterMutation = immediateSource.byteLength === 0;
      sourceByteLengthAfterDetach = immediateSource.byteLength;

      const commandEncoder = device.createCommandEncoder({
        label: 'bundle-immediate-canary-commands',
      });
      const pass = commandEncoder.beginRenderPass({
        label: 'bundle-immediate-canary-pass',
        colorAttachments: [colorAttachment(resources.texture)],
      });
      pass.executeBundles([bundle]);
      pass.end();
      encodeCopy(commandEncoder, resources.texture, resources.readback, resources.width);
      device.queue.submit([commandEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return readU32Sequence(resources.readback, BUNDLE_IMMEDIATE_SEQUENCE.length);
    });
    const addressSequenceAtEncoding = addressesForImmediates(BUNDLE_IMMEDIATE_SEQUENCE);
    const expectedU32Sequence = valuesAtAddresses(addressSequenceAtEncoding);
    const mutatedSourceWouldProduceU32Sequence = valuesAtAddresses(
      addressesForImmediates(BUNDLE_MUTATED_IMMEDIATE_SEQUENCE),
    );
    const snapshotPreserved = sameArray(observedU32Sequence, expectedU32Sequence)
      && observedU32Sequence.every(
        (value, index) => value !== mutatedSourceWouldProduceU32Sequence[index],
      );
    return {
      pass: snapshotPreserved && sourceDetachedAfterMutation,
      renderBundleSetImmediates,
      immediateU32SequenceAtEncoding: [...BUNDLE_IMMEDIATE_SEQUENCE],
      sourceU32SequenceAfterFinish,
      sourceDetachedAfterMutation,
      sourceByteLengthAfterDetach,
      instanceIndexSequence: [...INSTANCE_INDEX_SEQUENCE],
      addressExpression: 'immediateBase + instanceIndex',
      addressSequenceAtEncoding,
      expectedU32Sequence,
      mutatedSourceWouldProduceU32Sequence,
      observedU32Sequence,
      snapshotPreserved,
    };
  } finally {
    resources.texture.destroy();
    resources.readback.destroy();
  }
}

async function runUnsetAfterBundleControl(device, pipeline, bindGroup) {
  const resources = createReadbackTarget(device, 'unset-after-bundle-control', 1);
  try {
    const result = await captureExpectedRejection(
      device,
      'unset-required-slot-after-bundle',
      async () => {
        const bundleEncoder = device.createRenderBundleEncoder({
          label: 'unset-after-bundle-control-bundle-encoder',
          colorFormats: [TARGET_FORMAT],
        });
        bundleEncoder.setPipeline(pipeline);
        bundleEncoder.setBindGroup(0, bindGroup);
        bundleEncoder.setImmediates(0, new Uint32Array([1]));
        bundleEncoder.draw(6, 1, 0, 0);
        const bundle = bundleEncoder.finish();

        const commandEncoder = device.createCommandEncoder({
          label: 'unset-after-bundle-control-commands',
        });
        const pass = commandEncoder.beginRenderPass({
          colorAttachments: [colorAttachment(resources.texture)],
        });
        pass.executeBundles([bundle]);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6, 1, 0, 0);
        pass.end();
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();
      },
    );
    return {
      ...result,
      omittedRange: { byteOffset: 0, byteLength: REQUIRED_IMMEDIATE_BYTES },
      priorBundleSetSameRange: true,
      directStateRestoredAfterExecuteBundles: {
        pipeline: true,
        bindGroup0: true,
      },
    };
  } finally {
    resources.texture.destroy();
    resources.readback.destroy();
  }
}

async function runInvalidRangeControl({
  device,
  pipeline,
  bindGroup,
  name,
  rangeOffset,
}) {
  const resources = createReadbackTarget(device, `${name}-control`, 1);
  try {
    const result = await captureExpectedRejection(device, name, async () => {
      const commandEncoder = device.createCommandEncoder({
        label: `${name}-control-commands`,
      });
      const pass = commandEncoder.beginRenderPass({
        colorAttachments: [colorAttachment(resources.texture)],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      try {
        pass.setImmediates(rangeOffset, new Uint32Array([1]));
        pass.draw(6, 1, 0, 0);
      } finally {
        pass.end();
      }
      device.queue.submit([commandEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    });
    return {
      ...result,
      requestedRange: {
        byteOffset: rangeOffset,
        byteLength: REQUIRED_IMMEDIATE_BYTES,
      },
      deviceMaxImmediateSize: Number(device.limits.maxImmediateSize),
    };
  } finally {
    resources.texture.destroy();
    resources.readback.destroy();
  }
}

async function executeImmediateCanary(state) {
  state.capabilities.navigatorGpu = typeof navigator.gpu === 'object';
  requireCapability(state.capabilities.navigatorGpu, 'navigator.gpu is unavailable.');
  state.capabilities.wgslLanguageFeatures = sortedFeatureNames(
    navigator.gpu.wgslLanguageFeatures,
  );
  state.capabilities.wgslImmediateAddressSpace =
    state.capabilities.wgslLanguageFeatures.includes(IMMEDIATE_WGSL_FEATURE);
  requireCapability(state.capabilities.wgslImmediateAddressSpace,
    `WGSL feature ${IMMEDIATE_WGSL_FEATURE} is unavailable.`);

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
    forceFallbackAdapter: false,
  });
  state.capabilities.adapterAvailable = adapter !== null;
  requireCapability(adapter !== null, 'WebGPU adapter request returned null.');
  state.adapterInfo = await serializeAdapterInfo(adapter);
  state.capabilities.nonFallbackAdapter = state.adapterInfo?.isFallbackAdapter === false;
  requireCapability(state.capabilities.nonFallbackAdapter,
    'A known non-fallback WebGPU adapter is required.', state.adapterInfo);
  state.adapterFeatures = sortedFeatureNames(adapter.features);
  state.capabilities.indirectFirstInstanceAdapter =
    state.adapterFeatures.includes(INDIRECT_FIRST_INSTANCE_FEATURE);
  state.capabilities.adapterMaxImmediateSize = Number(adapter.limits?.maxImmediateSize);
  requireCapability(Number.isFinite(state.capabilities.adapterMaxImmediateSize)
    && state.capabilities.adapterMaxImmediateSize >= REQUIRED_IMMEDIATE_BYTES,
  'Adapter maxImmediateSize is below four bytes or unavailable.', state.capabilities);

  let device;
  state.requestedDeviceFeatures = state.capabilities.indirectFirstInstanceAdapter
    ? [INDIRECT_FIRST_INSTANCE_FEATURE]
    : [];
  try {
    device = await adapter.requestDevice({
      requiredFeatures: state.requestedDeviceFeatures,
      requiredLimits: { maxImmediateSize: REQUIRED_IMMEDIATE_BYTES },
    });
  } catch (error) {
    error.canaryStatus = 'unsupported';
    throw error;
  }
  state.device = device;
  state.deviceFeatures = sortedFeatureNames(device.features);
  state.capabilities.indirectFirstInstanceDevice =
    state.deviceFeatures.includes(INDIRECT_FIRST_INSTANCE_FEATURE);
  state.capabilities.deviceMaxImmediateSize = Number(device.limits?.maxImmediateSize);
  device.addEventListener('uncapturederror', (event) => {
    event.preventDefault();
    state.deviceErrors.uncapturedErrors.push(serializeGpuError(event.error));
  });
  state.deviceLostPromise = device.lost.then((info) => {
    const record = {
      reason: String(info?.reason ?? 'unknown').slice(0, 256),
      message: String(info?.message ?? '').slice(0, 4_096),
    };
    state.deviceErrors.finalDeviceLoss = record;
    if (record.reason !== 'destroyed') {
      state.deviceErrors.unexpectedDeviceLosses.push({
        ...record,
      });
    }
    return record;
  });
  requireCapability(Number.isFinite(state.capabilities.deviceMaxImmediateSize)
    && state.capabilities.deviceMaxImmediateSize >= REQUIRED_IMMEDIATE_BYTES,
  'Device maxImmediateSize is below four bytes or unavailable.', state.capabilities);

  const lifecycleScopeFilters = ['internal', 'out-of-memory', 'validation'];
  for (const filter of lifecycleScopeFilters) device.pushErrorScope(filter);

  const shaderModule = device.createShaderModule({
    label: 'webgpu-immediates-canary-shader',
    code: buildImmediateCanaryShader(),
  });
  const compilationInfo = await shaderModule.getCompilationInfo();
  state.shaderCompilationMessages = compilationInfo.messages.map((message) => ({
    type: message.type,
    message: String(message.message).slice(0, 4_096),
    lineNum: message.lineNum,
    linePos: message.linePos,
    offset: message.offset,
    length: message.length,
  }));
  requireCondition(!state.shaderCompilationMessages.some((message) => message.type === 'error'),
    'Immediate canary WGSL did not compile.', state.shaderCompilationMessages);

  const valuesBuffer = device.createBuffer({
    label: 'webgpu-immediates-canary-values',
    size: IMMEDIATE_CANARY_VALUES.length * Uint32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  state.valuesBuffer = valuesBuffer;
  new Uint32Array(valuesBuffer.getMappedRange()).set(IMMEDIATE_CANARY_VALUES);
  valuesBuffer.unmap();
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'webgpu-immediates-canary-bind-group-layout',
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'read-only-storage' },
    }],
  });
  state.pipelineLayoutDescriptor = {
    bindGroupLayoutCount: 1,
    immediateSize: REQUIRED_IMMEDIATE_BYTES,
  };
  const pipelineLayout = device.createPipelineLayout({
    label: 'webgpu-immediates-canary-pipeline-layout',
    bindGroupLayouts: [bindGroupLayout],
    immediateSize: REQUIRED_IMMEDIATE_BYTES,
  });
  const pipeline = await withGpuErrorScopes(device, () => device.createRenderPipelineAsync({
    label: 'webgpu-immediates-canary-pipeline',
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: 'vertexMain' },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragmentMain',
      targets: [{ format: TARGET_FORMAT }],
    },
    primitive: { topology: 'triangle-list' },
  }));
  const bindGroup = device.createBindGroup({
    label: 'webgpu-immediates-canary-bind-group',
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: valuesBuffer } }],
  });

  state.canaries.directRenderPass = await runDirectRenderPassCanary(
    device,
    pipeline,
    bindGroup,
  );
  state.capabilities.renderPassSetImmediates =
    state.canaries.directRenderPass.renderPassSetImmediates;
  requireCapability(state.capabilities.renderPassSetImmediates,
    'GPURenderPassEncoder.setImmediates is unavailable.');
  requireCondition(state.canaries.directRenderPass.pass,
    'Direct render-pass non-monotonic address/output canary failed.',
    state.canaries.directRenderPass);

  state.canaries.renderBundleSnapshot = await runRenderBundleSnapshotCanary(
    device,
    pipeline,
    bindGroup,
  );
  state.capabilities.renderBundleSetImmediates =
    state.canaries.renderBundleSnapshot.renderBundleSetImmediates;
  requireCapability(state.capabilities.renderBundleSetImmediates,
    'GPURenderBundleEncoder.setImmediates is unavailable.');
  requireCondition(state.canaries.renderBundleSnapshot.pass,
    'Render-bundle non-monotonic snapshot canary failed.',
    state.canaries.renderBundleSnapshot);

  state.negativeControls.unsetRequiredSlotAfterBundle =
    await runUnsetAfterBundleControl(device, pipeline, bindGroup);
  state.negativeControls.misalignedRange = await runInvalidRangeControl({
    device,
    pipeline,
    bindGroup,
    name: 'misaligned-range',
    rangeOffset: 2,
  });
  state.negativeControls.rangeBeyondMaxImmediateSize = await runInvalidRangeControl({
    device,
    pipeline,
    bindGroup,
    name: 'range-beyond-max-immediate-size',
    rangeOffset: state.capabilities.deviceMaxImmediateSize,
  });
  for (const control of Object.values(state.negativeControls)) {
    requireCondition(validExpectedRejection(control),
      `Negative control ${control?.name ?? 'unknown'} was not cleanly rejected.`, control);
  }

  await device.queue.onSubmittedWorkDone();
  for (const filter of [...lifecycleScopeFilters].reverse()) {
    try {
      const error = await device.popErrorScope();
      if (error !== null) {
        state.deviceErrors.lifecycleScopedErrors.push({
          filter,
          ...serializeGpuError(error),
        });
      }
    } catch (error) {
      state.deviceErrors.lifecycleScopedErrors.push({
        filter,
        popFailed: true,
        ...serializeError(error),
      });
    }
  }
  await new Promise((resolve) => setTimeout(resolve, 0));
  requireCondition(state.deviceErrors.lifecycleScopedErrors.length === 0,
    'Lifecycle WebGPU error scope captured an error.', state.deviceErrors);
  requireCondition(state.deviceErrors.uncapturedErrors.length === 0,
    'Uncaptured WebGPU errors were observed.', state.deviceErrors);
  requireCondition(state.deviceErrors.unexpectedDeviceLosses.length === 0,
    'Unexpected WebGPU device loss was observed.', state.deviceErrors);
  const capabilityAssessment = assessImmediateCapabilities(state.capabilities);
  requireCondition(capabilityAssessment.pass, 'Immediate API capability gate failed.',
    capabilityAssessment);
}

export async function runWebGpuImmediatesCanary() {
  const state = {
    capabilities: {
      navigatorGpu: false,
      wgslLanguageFeatures: [],
      wgslImmediateAddressSpace: false,
      adapterAvailable: false,
      nonFallbackAdapter: false,
      indirectFirstInstanceAdapter: false,
      indirectFirstInstanceDevice: false,
      adapterMaxImmediateSize: null,
      deviceMaxImmediateSize: null,
      renderPassSetImmediates: false,
      renderBundleSetImmediates: false,
    },
    adapterInfo: null,
    adapterFeatures: [],
    requestedDeviceFeatures: [],
    deviceFeatures: [],
    shaderCompilationMessages: [],
    pipelineLayoutDescriptor: null,
    canaries: {
      directRenderPass: null,
      renderBundleSnapshot: null,
    },
    negativeControls: {
      unsetRequiredSlotAfterBundle: null,
      misalignedRange: null,
      rangeBeyondMaxImmediateSize: null,
    },
    deviceErrors: {
      uncapturedErrors: [],
      unexpectedDeviceLosses: [],
      lifecycleScopedErrors: [],
      finalDeviceLoss: null,
    },
    device: null,
    deviceLostPromise: null,
    valuesBuffer: null,
  };
  let failure = null;
  let status = 'pass';
  try {
    await executeImmediateCanary(state);
  } catch (error) {
    failure = serializeError(error);
    status = error?.canaryStatus === 'unsupported' ? 'unsupported' : 'fail';
  } finally {
    state.valuesBuffer?.destroy();
    if (state.device !== null) {
      state.device.destroy();
      await state.deviceLostPromise;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  const capabilityAssessment = assessImmediateCapabilities(state.capabilities);
  const threeLaneEligibility = assessThreeLaneEligibility(state.capabilities);
  const result = {
    schemaVersion: CANARY_SCHEMA_VERSION,
    kind: CANARY_KIND,
    status,
    scope: RAW_CANARY_SCOPE,
    executionMode: 'technical-canary',
    phase: 'phase-0-raw-capability-canary',
    fullPhase0Pass: false,
    integratedCanaryInvoked: false,
    analysisEligible: false,
    efficacyAnalysisAllowed: false,
    numericalDecision: null,
    timingCaptured: false,
    efficacyEvaluated: false,
    userAgent: navigator.userAgent,
    secureContext: globalThis.isSecureContext,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    adapterInfo: state.adapterInfo,
    adapterFeatures: state.adapterFeatures,
    requestedDeviceFeatures: state.requestedDeviceFeatures,
    deviceFeatures: state.deviceFeatures,
    capabilities: state.capabilities,
    capabilityAssessment,
    threeLaneEligibility,
    shaderCompilationMessages: state.shaderCompilationMessages,
    pipelineLayoutDescriptor: state.pipelineLayoutDescriptor,
    canaries: state.canaries,
    negativeControls: state.negativeControls,
    deviceErrors: state.deviceErrors,
    failure,
  };
  if (failure === null) {
    const validation = validateImmediateCanaryResult(result);
    if (!validation.pass) {
      result.status = 'fail';
      result.failure = {
        name: 'CanaryValidationError',
        message: 'Completed canary result failed its independent page validator.',
        stack: null,
        detail: validation,
        canaryStatus: null,
      };
    }
  }
  return result;
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  const published = {
    ready: false,
    result: null,
  };
  window.__WEBGPU_IMMEDIATES_CANARY__ = published;
  const statusElement = document.getElementById('status');
  void runWebGpuImmediatesCanary().then((result) => {
    published.result = result;
    published.ready = true;
    if (statusElement !== null) {
      statusElement.textContent = result.status === 'pass'
        ? 'All immediate-data correctness checks passed.'
        : `Immediate-data canary ${result.status}: ${result.failure?.message ?? 'unknown failure'}`;
    }
  }).catch((error) => {
    published.result = {
      schemaVersion: CANARY_SCHEMA_VERSION,
      kind: CANARY_KIND,
      status: 'fail',
      scope: RAW_CANARY_SCOPE,
      executionMode: 'technical-canary',
      phase: 'phase-0-raw-capability-canary',
      fullPhase0Pass: false,
      integratedCanaryInvoked: false,
      analysisEligible: false,
      efficacyAnalysisAllowed: false,
      numericalDecision: null,
      timingCaptured: false,
      efficacyEvaluated: false,
      failure: serializeError(error),
    };
    published.ready = true;
    if (statusElement !== null) statusElement.textContent = 'Immediate-data canary failed closed.';
  });
}
