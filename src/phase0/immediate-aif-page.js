import {
  BundleGroup,
  Color,
  DepthTexture,
  DirectionalLight,
  DoubleSide,
  Float32BufferAttribute,
  FloatType,
  HemisphereLight,
  InstancedBufferGeometry,
  Mesh,
  MeshBasicNodeMaterial,
  NoBlending,
  NoColorSpace,
  OrthographicCamera,
  PerspectiveCamera,
  REVISION,
  RGBAFormat,
  RenderTarget,
  Scene,
  SRGBColorSpace,
  UnsignedByteType,
  Vector3,
  WebGPUBackend,
  WebGPURenderer,
} from 'three/webgpu';
import {
  Fn,
  float,
  instanceIndex,
  positionGeometry,
  storage,
  uint,
  varyingProperty,
  vec2,
  vec4,
} from 'three/tsl';
import { CAMERA, VIEWPORT } from '../config.js';
import {
  STORAGE_TRANSFORM_ADDRESS_MODES,
  createVisibleIdAddressNode,
} from '../materials/storage-transform.js';
import {
  IMMEDIATE_AIF_PHASE0_LANES,
  IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
  IMMEDIATE_AIF_PHASE0_SCHEDULES,
  IMMEDIATE_AIF_PHASE0_WORKLOAD,
  IMMEDIATE_AIF_UNUSED_ADDRESS,
} from './immediate-aif-plan.js';
import {
  IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET,
  IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS,
  IMMEDIATE_AIF_PHASE0_PAGE_GLOBAL,
  IMMEDIATE_AIF_PHASE0_PAGE_KIND,
  IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN,
  bytesOf,
  compareImmediateAifPackedAddressBytes,
  createImmediateAifObservationChain,
  createImmediateAifObservationChallenge,
  createImmediateAifPackedAddressBytes,
  immediateAifBufferBindingLayoutSnapshot,
  drainImmediateAifErrorScopes,
  immediateAifExpectedDiagnosticVertexInputs,
  immediateAifExpectedBindingVisibility,
  immediateAifQueueWriteBufferSourceSelection,
  immediateAifPhase0PlanEvidence,
  normalizeImmediateAifCommonRenderStateProjection,
  realizeAndFreezeImmediateAifTransform,
  realizeImmediateAifReversedDepthCamera,
  serializeImmediateAifPhase0Error,
  sha256Bytes,
  sha256Text,
  validateImmediateAifPhase0PageResultShape,
  validateImmediateAifPhase0TargetConfiguration,
} from './immediate-aif-page-contract.js';
import { createIndexedGeometryFixtures } from '../scenes/geometry-fixtures.js';
import {
  fingerprintFixedSubsetScenario,
  fingerprintImmediateAifPhase0GeometryFixtures,
} from '../scenes/geometry-fingerprints.js';
import { buildImmediateAifPhase0Strategy } from '../strategies/immediate-aif-phase0.js';
import {
  runtimeStorageBindingEvidence,
  runtimeVertexInputEvidence,
} from '../strategies/first-instance-crossover.js';
import { collectLiveComputeLaneEvidence } from '../strategies/live-first-instance-crossover.js';
import { disposeStrategyResources } from '../strategies/resources.js';
import {
  createImmediateAifRenderShaderEvidence,
} from '../validation/first-instance-shader-evidence.js';

const RUNNER_INJECTED_TARGET_CONFIGURATION =
  /* THREE_IMMEDIATE_PHASE0_TARGET_CONFIGURATION_V1 */ null;

const FEATURE_INDIRECT_FIRST_INSTANCE = 'indirect-first-instance';
const FEATURE_IMMEDIATE_ADDRESS_SPACE = 'immediate_address_space';
const REQUIRED_IMMEDIATE_BYTES = 4;
const INDIRECT_WORDS = 5;
const THREE_ATTRIBUTE_TYPE_VERTEX = 1;
const THREE_ATTRIBUTE_TYPE_INDEX = 2;
const THREE_ATTRIBUTE_TYPE_STORAGE = 3;
const THREE_ATTRIBUTE_TYPE_INDIRECT = 4;
const EXPECTED_READBACK_STAGING_COUNT = 825;
const EXPECTED_READBACK_PHASE_COUNT = 666;
const ADDRESS_TRIANGLE = Object.freeze([
  -0.375, -0.375,
  0.375, -0.375,
  0, 0.375,
]);
const ADDRESS_MODE_BY_LANE = Object.freeze({
  A: STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  I: STORAGE_TRANSFORM_ADDRESS_MODES.IMMEDIATE_BASE,
  F: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
});
const TRACKED_RENDER_METHODS = Object.freeze([
  'setPipeline',
  'setBindGroup',
  'setIndexBuffer',
  'setVertexBuffer',
  'setViewport',
  'setScissorRect',
  'setBlendConstant',
  'setStencilReference',
  'setImmediates',
  'draw',
  'drawIndexed',
  'drawIndirect',
  'drawIndexedIndirect',
  'executeBundles',
  'beginOcclusionQuery',
  'endOcclusionQuery',
  'end',
]);
const TRACKED_COMPUTE_METHODS = Object.freeze([
  'setPipeline',
  'setBindGroup',
  'setImmediates',
  'dispatchWorkgroups',
  'dispatchWorkgroupsIndirect',
  'end',
]);

function fail(message, detail = null) {
  const error = new Error(message);
  error.detail = detail;
  throw error;
}

function requireCondition(condition, message, detail = null) {
  if (!condition) fail(message, detail);
}

function unsupported(message, detail = null) {
  const error = new Error(message);
  error.detail = detail;
  error.phase0Status = 'unsupported';
  throw error;
}

function serializeGpuError(error) {
  return {
    name: String(error?.constructor?.name ?? error?.name ?? 'GPUError').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
  };
}

async function drainPageErrorScopes(state, trigger) {
  if (state.errorScopeDrainage !== null) return state.errorScopeDrainage;
  const drainage = await drainImmediateAifErrorScopes({
    device: state.device,
    pushedFilters: state.pendingErrorScopeFilters,
    trigger,
  });
  state.pendingErrorScopeFilters = [];
  state.errorScopeDrainage = drainage;
  state.gpuErrors.scopeDrainage = drainage;
  state.gpuErrors.scoped = drainage.scopedErrors;
  return drainage;
}

function sortedFeatures(features) {
  return features == null || typeof features[Symbol.iterator] !== 'function'
    ? []
    : [...features].map(String).sort();
}

async function adapterInformation(adapter) {
  let information = adapter?.info ?? null;
  if (information === null && typeof adapter?.requestAdapterInfo === 'function') {
    information = await adapter.requestAdapterInfo();
  }
  if (information === null || typeof information !== 'object') return null;
  const result = {};
  for (const key of [
    'vendor', 'architecture', 'device', 'description', 'backend', 'type',
    'driver', 'isFallbackAdapter',
  ]) {
    if (typeof information[key] === 'string' || typeof information[key] === 'boolean') {
      result[key] = information[key];
    }
  }
  return result;
}

function replaceCallable(target, name, factory) {
  const original = target?.[name];
  requireCondition(typeof original === 'function', `${name} is not callable.`);
  const replacement = factory(original.bind(target));
  Object.defineProperty(target, name, {
    configurable: true,
    writable: true,
    value: replacement,
  });
  requireCondition(target[name] === replacement, `Could not instrument ${name}.`);
}

function descriptorSnapshot(descriptor) {
  const size = descriptor?.size;
  const sizeSnapshot = Array.isArray(size)
    ? Array.from(size, Number)
    : size !== null && typeof size === 'object'
      ? {
        width: Number(size.width),
        height: Number(size.height ?? 1),
        depthOrArrayLayers: Number(size.depthOrArrayLayers ?? 1),
      }
      : size ?? null;
  return {
    label: descriptor?.label ?? null,
    hasOwnImmediateSize: Object.hasOwn(descriptor ?? {}, 'immediateSize'),
    immediateSize: descriptor?.immediateSize ?? null,
    bindGroupLayoutCount: descriptor?.bindGroupLayouts?.length ?? null,
    colorFormats: descriptor?.colorFormats?.map(String) ?? null,
    depthStencilFormat: descriptor?.depthStencilFormat ?? null,
    sampleCount: descriptor?.sampleCount ?? null,
    size: sizeSnapshot,
    usage: descriptor?.usage ?? null,
    format: descriptor?.format ?? null,
    mappedAtCreation: descriptor?.mappedAtCreation ?? null,
    dimension: descriptor?.dimension ?? null,
    mipLevelCount: descriptor?.mipLevelCount ?? null,
    viewFormats: descriptor?.viewFormats == null
      ? null
      : Array.from(descriptor.viewFormats, String),
    textureBindingViewDimension: descriptor?.textureBindingViewDimension ?? null,
  };
}

function pipelineConstantsSnapshot(constants) {
  if (constants === undefined) return null;
  return Object.fromEntries(Object.entries(constants).map(([name, value]) => [
    name,
    typeof value === 'boolean' ? value : Number(value),
  ]));
}

function blendComponentSnapshot(component) {
  return component == null ? null : {
    operation: component.operation ?? 'add',
    srcFactor: component.srcFactor ?? 'one',
    dstFactor: component.dstFactor ?? 'zero',
  };
}

function stencilFaceSnapshot(face) {
  return face == null ? null : {
    compare: face.compare ?? 'always',
    failOp: face.failOp ?? 'keep',
    depthFailOp: face.depthFailOp ?? 'keep',
    passOp: face.passOp ?? 'keep',
  };
}

function renderPipelineDescriptorSnapshot(descriptor, identify) {
  return {
    label: descriptor?.label ?? null,
    hasOwnImmediateSize: Object.hasOwn(descriptor ?? {}, 'immediateSize'),
    immediateSize: descriptor?.immediateSize ?? null,
    layoutId: identify(descriptor?.layout, 'pipeline-layout'),
    vertex: descriptor?.vertex == null ? null : {
      moduleId: identify(descriptor.vertex.module, 'shader-module'),
      entryPoint: descriptor.vertex.entryPoint ?? null,
      constants: pipelineConstantsSnapshot(descriptor.vertex.constants),
      buffers: Array.from(descriptor.vertex.buffers ?? [], (buffer) => (
        buffer == null ? null : {
          arrayStride: Number(buffer.arrayStride),
          stepMode: buffer.stepMode ?? 'vertex',
          attributes: Array.from(buffer.attributes ?? [], (attribute) => ({
            format: attribute.format,
            offset: Number(attribute.offset),
            shaderLocation: Number(attribute.shaderLocation),
          })),
        }
      )),
    },
    fragment: descriptor?.fragment == null ? null : {
      moduleId: identify(descriptor.fragment.module, 'shader-module'),
      entryPoint: descriptor.fragment.entryPoint ?? null,
      constants: pipelineConstantsSnapshot(descriptor.fragment.constants),
      targets: Array.from(descriptor.fragment.targets ?? [], (target) => (
        target == null ? null : {
          format: target.format,
          writeMask: Number(target.writeMask ?? 0xf),
          blend: target.blend == null ? null : {
            color: blendComponentSnapshot(target.blend.color),
            alpha: blendComponentSnapshot(target.blend.alpha),
          },
        }
      )),
    },
    primitive: descriptor?.primitive == null ? null : {
      topology: descriptor.primitive.topology ?? 'triangle-list',
      stripIndexFormat: descriptor.primitive.stripIndexFormat ?? null,
      frontFace: descriptor.primitive.frontFace ?? 'ccw',
      cullMode: descriptor.primitive.cullMode ?? 'none',
      unclippedDepth: descriptor.primitive.unclippedDepth ?? false,
    },
    depthStencil: descriptor?.depthStencil == null ? null : {
      format: descriptor.depthStencil.format,
      depthWriteEnabled: descriptor.depthStencil.depthWriteEnabled ?? false,
      depthCompare: descriptor.depthStencil.depthCompare ?? 'always',
      stencilFront: stencilFaceSnapshot(descriptor.depthStencil.stencilFront),
      stencilBack: stencilFaceSnapshot(descriptor.depthStencil.stencilBack),
      stencilReadMask: Number(descriptor.depthStencil.stencilReadMask ?? 0xffff_ffff),
      stencilWriteMask: Number(descriptor.depthStencil.stencilWriteMask ?? 0xffff_ffff),
      depthBias: Number(descriptor.depthStencil.depthBias ?? 0),
      depthBiasSlopeScale: Number(descriptor.depthStencil.depthBiasSlopeScale ?? 0),
      depthBiasClamp: Number(descriptor.depthStencil.depthBiasClamp ?? 0),
    },
    multisample: {
      count: Number(descriptor?.multisample?.count ?? 1),
      mask: Number(descriptor?.multisample?.mask ?? 0xffff_ffff),
      alphaToCoverageEnabled: descriptor?.multisample?.alphaToCoverageEnabled ?? false,
    },
  };
}

function computePipelineDescriptorSnapshot(descriptor, identify) {
  return {
    label: descriptor?.label ?? null,
    hasOwnImmediateSize: Object.hasOwn(descriptor ?? {}, 'immediateSize'),
    immediateSize: descriptor?.immediateSize ?? null,
    layoutId: identify(descriptor?.layout, 'pipeline-layout'),
    compute: descriptor?.compute == null ? null : {
      moduleId: identify(descriptor.compute.module, 'shader-module'),
      entryPoint: descriptor.compute.entryPoint ?? null,
      constants: pipelineConstantsSnapshot(descriptor.compute.constants),
    },
  };
}

function renderPassDescriptorSnapshot(descriptor, identify, viewTextureIds) {
  const attachment = (value) => value == null ? null : {
    viewId: identify(value.view, 'texture-view'),
    textureId: viewTextureIds.get(value.view) ?? null,
    resolveTargetViewId: identify(value.resolveTarget, 'texture-view'),
    resolveTargetTextureId: viewTextureIds.get(value.resolveTarget) ?? null,
    depthSlice: value.depthSlice ?? null,
    loadOp: value.loadOp ?? null,
    storeOp: value.storeOp ?? null,
    clearValue: value.clearValue == null ? null : structuredClone(value.clearValue),
  };
  const depth = descriptor?.depthStencilAttachment;
  return {
    label: descriptor?.label ?? null,
    colorAttachments: Array.from(descriptor?.colorAttachments ?? [], attachment),
    depthStencilAttachment: depth == null ? null : {
      viewId: identify(depth.view, 'texture-view'),
      textureId: viewTextureIds.get(depth.view) ?? null,
      depthLoadOp: depth.depthLoadOp ?? null,
      depthStoreOp: depth.depthStoreOp ?? null,
      depthClearValue: depth.depthClearValue ?? null,
      depthReadOnly: depth.depthReadOnly ?? false,
      stencilLoadOp: depth.stencilLoadOp ?? null,
      stencilStoreOp: depth.stencilStoreOp ?? null,
      stencilClearValue: depth.stencilClearValue ?? null,
      stencilReadOnly: depth.stencilReadOnly ?? false,
    },
    maxDrawCount: descriptor?.maxDrawCount == null
      ? null
      : Number(descriptor.maxDrawCount),
    occlusionQuerySetId: identify(descriptor?.occlusionQuerySet, 'query-set'),
    timestampWrites: descriptor?.timestampWrites == null ? null : {
      querySetId: identify(descriptor.timestampWrites.querySet, 'query-set'),
      beginningOfPassWriteIndex:
        descriptor.timestampWrites.beginningOfPassWriteIndex ?? null,
      endOfPassWriteIndex: descriptor.timestampWrites.endOfPassWriteIndex ?? null,
    },
  };
}

function copyExtentSnapshot(size) {
  if (Array.isArray(size) || ArrayBuffer.isView(size)) {
    return {
      width: Number(size[0]),
      height: Number(size[1] ?? 1),
      depthOrArrayLayers: Number(size[2] ?? 1),
    };
  }
  return {
    width: Number(size?.width),
    height: Number(size?.height ?? 1),
    depthOrArrayLayers: Number(size?.depthOrArrayLayers ?? 1),
  };
}

function copyOriginSnapshot(origin) {
  if (Array.isArray(origin) || ArrayBuffer.isView(origin)) {
    return {
      x: Number(origin[0] ?? 0),
      y: Number(origin[1] ?? 0),
      z: Number(origin[2] ?? 0),
    };
  }
  return {
    x: Number(origin?.x ?? 0),
    y: Number(origin?.y ?? 0),
    z: Number(origin?.z ?? 0),
  };
}

function textureCopySnapshot(value, identify) {
  return {
    textureId: identify(value?.texture, 'texture'),
    mipLevel: Number(value?.mipLevel ?? 0),
    origin: copyOriginSnapshot(value?.origin),
    aspect: value?.aspect ?? 'all',
  };
}

function bufferCopySnapshot(value, identify) {
  return {
    bufferId: identify(value?.buffer, 'buffer'),
    offset: Number(value?.offset ?? 0),
    bytesPerRow: value?.bytesPerRow == null ? null : Number(value.bytesPerRow),
    rowsPerImage: value?.rowsPerImage == null ? null : Number(value.rowsPerImage),
  };
}

function immediateCallSnapshot(args, identify) {
  const [rangeOffset, data, suppliedDataOffset, suppliedDataSize] = args;
  const dataOffset = suppliedDataOffset ?? 0;
  const elementCount = ArrayBuffer.isView(data) ? data.length : data?.byteLength;
  const dataSize = suppliedDataSize ?? (elementCount - dataOffset);
  return {
    rangeOffset,
    sourceId: identify(data, 'typed-array'),
    backingBufferId: identify(data?.buffer, 'array-buffer'),
    sourceType: data?.constructor?.name ?? null,
    sourceElementCount: elementCount,
    sourceByteOffset: ArrayBuffer.isView(data) ? data.byteOffset : null,
    sourceByteLength: ArrayBuffer.isView(data) ? data.byteLength : null,
    dataOffset,
    dataSize,
    selectedValues: data instanceof Uint32Array
      ? Array.from({ length: dataSize }, (_, index) => data[dataOffset + index])
      : null,
  };
}

function immediateSourceSnapshot(events) {
  const calls = events.filter((event) => event.method === 'setImmediates');
  if (calls.length === 0) return null;
  const sourceIds = [...new Set(calls.map((event) => event.sourceId))];
  const backingBufferIds = [...new Set(calls.map((event) => event.backingBufferId))];
  const sourceTypes = [...new Set(calls.map((event) => event.sourceType))];
  const sourceElementCounts = [...new Set(calls.map((event) => event.sourceElementCount))];
  const sourceByteOffsets = [...new Set(calls.map((event) => event.sourceByteOffset))];
  const sourceByteLengths = [...new Set(calls.map((event) => event.sourceByteLength))];
  return {
    pass: sourceIds.length === 1
      && backingBufferIds.length === 1
      && sourceTypes.length === 1
      && sourceTypes[0] === 'Uint32Array'
      && sourceElementCounts.length === 1
      && sourceByteOffsets.length === 1
      && sourceByteLengths.length === 1
      && calls.every((event) => event.dataSize === 1
        && event.selectedValues?.length === 1),
    sourceId: sourceIds.length === 1 ? sourceIds[0] : null,
    backingBufferId: backingBufferIds.length === 1 ? backingBufferIds[0] : null,
    sourceType: sourceTypes.length === 1 ? sourceTypes[0] : null,
    sourceElementCount: sourceElementCounts.length === 1 ? sourceElementCounts[0] : null,
    sourceByteOffset: sourceByteOffsets.length === 1 ? sourceByteOffsets[0] : null,
    sourceByteLength: sourceByteLengths.length === 1 ? sourceByteLengths[0] : null,
    callCount: calls.length,
    dataOffsets: calls.map((event) => event.dataOffset),
    dataSizes: calls.map((event) => event.dataSize),
    recordedValues: calls.flatMap((event) => event.selectedValues ?? []),
  };
}

/** Capture actual WebGPU descriptors and encoder calls underneath Three.js. */
function installPhase0DeviceInstrumentation(device, observationNonce) {
  const ids = new WeakMap();
  const viewTextureIds = new WeakMap();
  const shaderObjects = new Map();
  const bufferObjects = new Map();
  const textureObjects = new Map();
  const queueWriteBufferSourceSnapshots = new Map();
  const immediateSourceObjects = new WeakMap();
  let nextId = 1;
  let sequence = 1;
  let shaderCreationOrdinal = 1;
  let capturePhase = null;
  let armedImmediateBundleFinishHook = null;
  const evidence = {
    patchedDeviceMethods: [],
    shaderModules: [],
    textureViews: [],
    bindGroupLayouts: [],
    pipelineLayouts: [],
    renderPipelines: [],
    computePipelines: [],
    renderBundleEncoders: [],
    renderPassEncoders: [],
    computePassEncoders: [],
    resources: [],
    resourceDestructions: [],
    queueWriteBuffers: [],
    queueWriteTextures: [],
    queueCopyExternalImagesToTexture: [],
    queueSubmissions: [],
    bufferMapEvents: [],
    commandEncoders: [],
    commandEncoderTransfers: [],
    markers: [],
  };

  const identify = (value, prefix) => {
    if (value === null || (typeof value !== 'object' && typeof value !== 'function')) {
      return null;
    }
    if (!ids.has(value)) ids.set(value, `${prefix}-${nextId ++}`);
    return ids.get(value);
  };

  replaceCallable(device, 'createShaderModule', (original) => (descriptor) => {
    const module = original(descriptor);
    const record = {
      sequence: sequence ++,
      capturePhase,
      observationNonce,
      creationOrdinal: shaderCreationOrdinal ++,
      moduleId: identify(module, 'shader-module'),
      label: descriptor?.label ?? null,
      code: descriptor?.code ?? null,
      compilationMessages: null,
    };
    evidence.shaderModules.push(record);
    shaderObjects.set(record, module);
    return module;
  });
  evidence.patchedDeviceMethods.push('createShaderModule');

  replaceCallable(device, 'createPipelineLayout', (original) => (descriptor) => {
    const layout = original(descriptor);
    evidence.pipelineLayouts.push({
      sequence: sequence ++,
      capturePhase,
      layoutId: identify(layout, 'pipeline-layout'),
      bindGroupLayoutIds: Array.from(descriptor?.bindGroupLayouts ?? [],
        (entry) => identify(entry, 'bind-group-layout')),
      ...descriptorSnapshot(descriptor),
    });
    return layout;
  });
  evidence.patchedDeviceMethods.push('createPipelineLayout');

  replaceCallable(device, 'createBindGroupLayout', (original) => (descriptor) => {
    const layout = original(descriptor);
    evidence.bindGroupLayouts.push({
      sequence: sequence ++,
      capturePhase,
      layoutId: identify(layout, 'bind-group-layout'),
      label: descriptor?.label ?? null,
      entries: Array.from(descriptor?.entries ?? [], (entry) => ({
        binding: entry.binding,
        visibility: entry.visibility,
        buffer: entry.buffer == null ? null : {
          type: entry.buffer.type ?? null,
          hasDynamicOffset: entry.buffer.hasDynamicOffset ?? false,
          minBindingSize: Number(entry.buffer.minBindingSize ?? 0),
        },
        sampler: entry.sampler == null ? null : { ...entry.sampler },
        texture: entry.texture == null ? null : { ...entry.texture },
        storageTexture: entry.storageTexture == null ? null : { ...entry.storageTexture },
        externalTexture: entry.externalTexture == null ? null : {
          ...entry.externalTexture,
        },
      })),
    });
    return layout;
  });
  evidence.patchedDeviceMethods.push('createBindGroupLayout');

  for (const type of ['Render', 'Compute']) {
    for (const suffix of ['', 'Async']) {
      const method = `create${type}Pipeline${suffix}`;
      replaceCallable(device, method, (original) => (descriptor) => {
        const fullDescriptor = type === 'Render'
          ? renderPipelineDescriptorSnapshot(descriptor, identify)
          : computePipelineDescriptorSnapshot(descriptor, identify);
        const record = {
          sequence: sequence ++,
          capturePhase,
          method,
          label: descriptor?.label ?? null,
          layoutId: identify(descriptor?.layout, 'pipeline-layout'),
          vertexModuleId: identify(descriptor?.vertex?.module, 'shader-module'),
          fragmentModuleId: identify(descriptor?.fragment?.module, 'shader-module'),
          computeModuleId: identify(descriptor?.compute?.module, 'shader-module'),
          vertexEntryPoint: descriptor?.vertex?.entryPoint ?? null,
          fragmentEntryPoint: descriptor?.fragment?.entryPoint ?? null,
          computeEntryPoint: descriptor?.compute?.entryPoint ?? null,
          hasOwnImmediateSize: Object.hasOwn(descriptor ?? {}, 'immediateSize'),
          immediateSize: descriptor?.immediateSize ?? null,
          pipelineId: null,
          colorFormats: descriptor?.fragment?.targets?.map((target) => target?.format ?? null) ?? [],
          depthStencilFormat: descriptor?.depthStencil?.format ?? null,
          primitive: descriptor?.primitive == null ? null : {
            topology: descriptor.primitive.topology ?? null,
            stripIndexFormat: descriptor.primitive.stripIndexFormat ?? null,
            frontFace: descriptor.primitive.frontFace ?? null,
            cullMode: descriptor.primitive.cullMode ?? null,
          },
          descriptor: fullDescriptor,
        };
        (type === 'Render' ? evidence.renderPipelines : evidence.computePipelines)
          .push(record);
        const value = original(descriptor);
        if (suffix === '') record.pipelineId = identify(value, `${type.toLowerCase()}-pipeline`);
        else void value.then((pipeline) => {
          record.pipelineId = identify(pipeline, `${type.toLowerCase()}-pipeline`);
        }, () => undefined);
        return value;
      });
      evidence.patchedDeviceMethods.push(method);
    }
  }

  for (const method of [
    'createBuffer',
    'createTexture',
    'createBindGroup',
    'createSampler',
    'createQuerySet',
  ]) {
    replaceCallable(device, method, (original) => (descriptor) => {
      const value = original(descriptor);
      const record = {
        sequence: sequence ++,
        capturePhase,
        method,
        resourceId: identify(value, method.slice('create'.length).toLowerCase()),
        ...descriptorSnapshot(descriptor),
      };
      if (method === 'createBuffer') {
        const mapRead = globalThis.GPUBufferUsage?.MAP_READ ?? 0x0001;
        const copyDestination = globalThis.GPUBufferUsage?.COPY_DST ?? 0x0008;
        record.resourceClass = (Number(descriptor?.usage ?? 0) & mapRead) !== 0
          && (Number(descriptor?.usage ?? 0) & copyDestination) !== 0
          ? 'readback-staging'
          : 'persistent-or-upload-buffer';
        record.mappedAtCreation = descriptor?.mappedAtCreation === true;
        record.destroyed = false;
        record.destroyCallCount = 0;
        record.destroySequence = null;
        record.destroyCapturePhase = null;
        bufferObjects.set(record, value);
        replaceCallable(value, 'destroy', (destroy) => (...args) => {
          const result = destroy(...args);
          record.destroyed = true;
          record.destroyCallCount += 1;
          record.destroySequence = sequence ++;
          record.destroyCapturePhase = capturePhase;
          evidence.resourceDestructions.push({
            sequence: record.destroySequence,
            capturePhase,
            resourceId: record.resourceId,
            resourceClass: record.resourceClass,
            method: 'destroy',
            callCount: record.destroyCallCount,
          });
          return result;
        });
        for (const mapMethod of ['mapAsync', 'getMappedRange', 'unmap']) {
          if (typeof value?.[mapMethod] !== 'function') continue;
          replaceCallable(value, mapMethod, (originalMapMethod) => (...args) => {
            evidence.bufferMapEvents.push({
              sequence: sequence ++,
              capturePhase,
              resourceId: record.resourceId,
              resourceClass: record.resourceClass,
              method: mapMethod,
              mode: mapMethod === 'mapAsync' ? Number(args[0]) : null,
              offset: mapMethod === 'unmap'
                ? null
                : Number((mapMethod === 'mapAsync' ? args[1] : args[0]) ?? 0),
              size: mapMethod === 'unmap'
                ? null
                : (mapMethod === 'mapAsync' ? args[2] : args[1]) === undefined
                  ? null
                  : Number(mapMethod === 'mapAsync' ? args[2] : args[1]),
            });
            return originalMapMethod(...args);
          });
        }
      }
      if (method === 'createTexture') {
        record.resourceClass = 'persistent-texture';
        record.destroyed = false;
        record.destroyCallCount = 0;
        record.destroySequence = null;
        record.destroyCapturePhase = null;
        textureObjects.set(record, value);
        replaceCallable(value, 'destroy', (destroy) => (...args) => {
          const result = destroy(...args);
          record.destroyed = true;
          record.destroyCallCount += 1;
          record.destroySequence = sequence ++;
          record.destroyCapturePhase = capturePhase;
          evidence.resourceDestructions.push({
            sequence: record.destroySequence,
            capturePhase,
            resourceId: record.resourceId,
            resourceClass: record.resourceClass,
            method: 'destroy',
            callCount: record.destroyCallCount,
          });
          return result;
        });
        replaceCallable(value, 'createView', (createView) => (viewDescriptor) => {
          const view = createView(viewDescriptor);
          const textureId = record.resourceId;
          const viewId = identify(view, 'texture-view');
          viewTextureIds.set(view, textureId);
          evidence.textureViews.push({
            sequence: sequence ++,
            capturePhase,
            viewId,
            textureId,
            descriptor: viewDescriptor == null ? null : {
              label: viewDescriptor.label ?? null,
              format: viewDescriptor.format ?? null,
              dimension: viewDescriptor.dimension ?? null,
              aspect: viewDescriptor.aspect ?? 'all',
              baseMipLevel: Number(viewDescriptor.baseMipLevel ?? 0),
              mipLevelCount: viewDescriptor.mipLevelCount == null
                ? null
                : Number(viewDescriptor.mipLevelCount),
              baseArrayLayer: Number(viewDescriptor.baseArrayLayer ?? 0),
              arrayLayerCount: viewDescriptor.arrayLayerCount == null
                ? null
                : Number(viewDescriptor.arrayLayerCount),
            },
          });
          return view;
        });
      }
      if (method === 'createBindGroup') {
        record.layoutId = identify(descriptor?.layout, 'bind-group-layout');
        record.entries = Array.from(descriptor?.entries ?? [], (entry) => {
          const resource = entry?.resource;
          const buffer = resource?.buffer ?? null;
          return {
            binding: entry?.binding ?? null,
            bufferId: identify(buffer, 'buffer'),
            bufferOffset: buffer === null ? null : Number(resource.offset ?? 0),
            bufferSize: buffer === null || resource.size === undefined
              ? null
              : Number(resource.size),
            resourceId: buffer === null ? identify(resource, 'binding-resource') : null,
          };
        });
      }
      if (method === 'createSampler') {
        record.sampler = {
          addressModeU: descriptor?.addressModeU ?? 'clamp-to-edge',
          addressModeV: descriptor?.addressModeV ?? 'clamp-to-edge',
          addressModeW: descriptor?.addressModeW ?? 'clamp-to-edge',
          magFilter: descriptor?.magFilter ?? 'nearest',
          minFilter: descriptor?.minFilter ?? 'nearest',
          mipmapFilter: descriptor?.mipmapFilter ?? 'nearest',
          lodMinClamp: Number(descriptor?.lodMinClamp ?? 0),
          lodMaxClamp: Number(descriptor?.lodMaxClamp ?? 32),
          compare: descriptor?.compare ?? null,
          maxAnisotropy: Number(descriptor?.maxAnisotropy ?? 1),
        };
      }
      if (method === 'createQuerySet') {
        record.querySet = {
          type: descriptor?.type ?? null,
          count: descriptor?.count == null ? null : Number(descriptor.count),
        };
      }
      evidence.resources.push(record);
      return value;
    });
    evidence.patchedDeviceMethods.push(method);
  }

  if (typeof device.importExternalTexture === 'function') {
    replaceCallable(device, 'importExternalTexture', (original) => (descriptor) => {
      const value = original(descriptor);
      evidence.resources.push({
        sequence: sequence ++,
        capturePhase,
        method: 'importExternalTexture',
        resourceId: identify(value, 'external-texture'),
        resourceClass: 'external-texture',
        sourceId: identify(descriptor?.source, 'external-image-source'),
        colorSpace: descriptor?.colorSpace ?? null,
      });
      return value;
    });
    evidence.patchedDeviceMethods.push('importExternalTexture');
  }

  replaceCallable(device.queue, 'writeBuffer', (original) => (...args) => {
    const [buffer, bufferOffset, data, dataOffset, size] = args;
    const selection = immediateAifQueueWriteBufferSourceSelection(
      data,
      dataOffset ?? 0,
      size,
    );
    const record = {
      sequence: sequence ++,
      capturePhase,
      bufferId: identify(buffer, 'buffer'),
      bufferOffset: Number(bufferOffset ?? 0),
      sourceId: identify(data, 'queue-write-source'),
      sourceBackingBufferId: identify(
        ArrayBuffer.isView(data) ? data.buffer : data,
        'array-buffer',
      ),
      sourceType: data?.constructor?.name ?? typeof data,
      sourceByteLength: Number(data?.byteLength ?? 0),
      sourceByteOffset: ArrayBuffer.isView(data) ? data.byteOffset : 0,
      sourceElementByteSize: selection.elementByteSize,
      sourceElementCount: selection.elementCount,
      dataOffset: Number(dataOffset ?? 0),
      size: size === undefined ? null : Number(size),
      selectedSourceElementOffset: selection.elementOffset,
      selectedSourceElementLength: selection.elementLength,
      selectedSourceByteOffset: selection.byteOffset,
      selectedSourceByteLength: selection.byteLength,
      selectedSourceSha256: null,
    };
    evidence.queueWriteBuffers.push(record);
    queueWriteBufferSourceSnapshots.set(record, selection.bytes);
    return original(...args);
  });
  evidence.patchedDeviceMethods.push('queue.writeBuffer');

  replaceCallable(device.queue, 'writeTexture', (original) => (...args) => {
    const [destination, data, dataLayout, size] = args;
    evidence.queueWriteTextures.push({
      sequence: sequence ++,
      capturePhase,
      textureId: identify(destination?.texture, 'texture'),
      mipLevel: Number(destination?.mipLevel ?? 0),
      origin: destination?.origin == null
        ? null
        : Array.isArray(destination.origin)
          ? Array.from(destination.origin, Number)
          : {
              x: Number(destination.origin.x ?? 0),
              y: Number(destination.origin.y ?? 0),
              z: Number(destination.origin.z ?? 0),
            },
      aspect: destination?.aspect ?? 'all',
      sourceType: data?.constructor?.name ?? typeof data,
      sourceByteLength: Number(data?.byteLength ?? 0),
      dataLayout: dataLayout == null ? null : {
        offset: Number(dataLayout.offset ?? 0),
        bytesPerRow: dataLayout.bytesPerRow == null
          ? null
          : Number(dataLayout.bytesPerRow),
        rowsPerImage: dataLayout.rowsPerImage == null
          ? null
          : Number(dataLayout.rowsPerImage),
      },
      size: Array.isArray(size)
        ? Array.from(size, Number)
        : {
            width: Number(size?.width),
            height: Number(size?.height ?? 1),
            depthOrArrayLayers: Number(size?.depthOrArrayLayers ?? 1),
          },
    });
    return original(...args);
  });
  evidence.patchedDeviceMethods.push('queue.writeTexture');

  replaceCallable(device.queue, 'copyExternalImageToTexture', (original) => (...args) => {
    const [source, destination, size] = args;
    evidence.queueCopyExternalImagesToTexture.push({
      sequence: sequence ++,
      capturePhase,
      source: {
        sourceId: identify(source?.source, 'external-image-source'),
        origin: copyOriginSnapshot(source?.origin),
        flipY: source?.flipY === true,
      },
      destination: {
        ...textureCopySnapshot(destination, identify),
        colorSpace: destination?.colorSpace ?? null,
        premultipliedAlpha: destination?.premultipliedAlpha === true,
      },
      copySize: copyExtentSnapshot(size),
    });
    return original(...args);
  });
  evidence.patchedDeviceMethods.push('queue.copyExternalImageToTexture');

  replaceCallable(device.queue, 'submit', (original) => (commandBuffers) => {
    evidence.queueSubmissions.push({
      sequence: sequence ++,
      capturePhase,
      commandBufferIds: Array.from(
        commandBuffers ?? [],
        (commandBuffer) => identify(commandBuffer, 'command-buffer'),
      ),
    });
    return original(commandBuffers);
  });
  evidence.patchedDeviceMethods.push('queue.submit');

  const wrapEncoder = (encoder, trace) => {
    for (const method of TRACKED_RENDER_METHODS) {
      if (typeof encoder[method] !== 'function') continue;
      replaceCallable(encoder, method, (original) => (...args) => {
        const value = original(...args);
        const event = { sequence: sequence ++, method };
        if (method === 'setImmediates') {
          Object.assign(event, immediateCallSnapshot(args, identify));
          const source = args[1];
          const prior = immediateSourceObjects.get(trace);
          if (prior === undefined) immediateSourceObjects.set(trace, source);
          else requireCondition(prior === source,
            'One render bundle used multiple immediate source objects.');
        }
        if (method === 'setPipeline') event.pipelineId = identify(args[0], 'render-pipeline');
        if (method === 'setBindGroup') {
          event.index = args[0];
          event.bindGroupId = identify(args[1], 'bind-group');
          event.dynamicOffsets = args[2] == null ? null : Array.from(args[2]);
          event.dynamicOffsetStart = args[3] ?? null;
          event.dynamicOffsetLength = args[4] ?? null;
          event.selectedDynamicOffsets = event.dynamicOffsets === null
            ? null
            : event.dynamicOffsets.slice(
              event.dynamicOffsetStart ?? 0,
              (event.dynamicOffsetStart ?? 0)
                + (event.dynamicOffsetLength ?? event.dynamicOffsets.length),
            );
        }
        if (method === 'setVertexBuffer') {
          event.slot = args[0];
          event.bufferId = identify(args[1], 'buffer');
          event.offset = args[2] ?? 0;
          event.size = args[3] ?? null;
        }
        if (method === 'setIndexBuffer') {
          event.bufferId = identify(args[0], 'buffer');
          event.indexFormat = args[1];
          event.offset = args[2] ?? 0;
          event.size = args[3] ?? null;
        }
        if (method === 'setViewport') {
          event.viewport = {
            x: args[0],
            y: args[1],
            width: args[2],
            height: args[3],
            minDepth: args[4],
            maxDepth: args[5],
          };
        }
        if (method === 'setScissorRect') {
          event.scissorRect = {
            x: args[0],
            y: args[1],
            width: args[2],
            height: args[3],
          };
        }
        if (method === 'setBlendConstant') {
          const color = args[0];
          event.color = Array.isArray(color) || ArrayBuffer.isView(color)
            ? Array.from(color)
            : {
                r: color?.r,
                g: color?.g,
                b: color?.b,
                a: color?.a,
              };
        }
        if (method === 'setStencilReference') event.reference = args[0];
        if (method === 'draw') {
          event.vertexCount = args[0];
          event.instanceCount = args[1] ?? 1;
          event.firstVertex = args[2] ?? 0;
          event.firstInstance = args[3] ?? 0;
        }
        if (method === 'drawIndexed') {
          event.indexCount = args[0];
          event.instanceCount = args[1] ?? 1;
          event.firstIndex = args[2] ?? 0;
          event.baseVertex = args[3] ?? 0;
          event.firstInstance = args[4] ?? 0;
        }
        if (method === 'drawIndexedIndirect' || method === 'drawIndirect') {
          event.bufferId = identify(args[0], 'buffer');
          event.indirectOffset = args[1];
        }
        if (method === 'executeBundles') {
          event.bundleIds = Array.from(args[0] ?? []).map(
            (bundle) => identify(bundle, 'render-bundle'),
          );
        }
        if (method === 'beginOcclusionQuery') event.queryIndex = args[0];
        trace.events.push(event);
        return value;
      });
    }
  };

  const wrapComputeEncoder = (encoder, trace) => {
    for (const method of TRACKED_COMPUTE_METHODS) {
      if (typeof encoder[method] !== 'function') continue;
      replaceCallable(encoder, method, (original) => (...args) => {
        const value = original(...args);
        const event = { sequence: sequence ++, method };
        if (method === 'setPipeline') {
          event.pipelineId = identify(args[0], 'compute-pipeline');
        }
        if (method === 'setBindGroup') {
          event.index = args[0];
          event.bindGroupId = identify(args[1], 'bind-group');
          event.dynamicOffsets = args[2] == null ? null : Array.from(args[2]);
          event.dynamicOffsetStart = args[3] ?? null;
          event.dynamicOffsetLength = args[4] ?? null;
          event.selectedDynamicOffsets = event.dynamicOffsets === null
            ? null
            : event.dynamicOffsets.slice(
              event.dynamicOffsetStart ?? 0,
              (event.dynamicOffsetStart ?? 0)
                + (event.dynamicOffsetLength ?? event.dynamicOffsets.length),
            );
        }
        if (method === 'setImmediates') {
          Object.assign(event, immediateCallSnapshot(args, identify));
        }
        if (method === 'dispatchWorkgroups') {
          event.dispatch = [args[0], args[1] ?? 1, args[2] ?? 1];
        }
        if (method === 'dispatchWorkgroupsIndirect') {
          event.bufferId = identify(args[0], 'buffer');
          event.offset = args[1];
        }
        trace.events.push(event);
        return value;
      });
    }
  };

  replaceCallable(device, 'createRenderBundleEncoder', (original) => (descriptor) => {
    const encoder = original(descriptor);
    const trace = {
      sequence: sequence ++,
      encoderId: identify(encoder, 'render-bundle-encoder'),
      capturePhase,
      descriptor: descriptorSnapshot(descriptor),
      callableSetImmediates: typeof encoder.setImmediates === 'function',
      events: [],
      bundleId: null,
      nativeFinishReturned: false,
      sourceSnapshot: null,
    };
    evidence.renderBundleEncoders.push(trace);
    wrapEncoder(encoder, trace);
    replaceCallable(encoder, 'finish', (nativeFinish) => (...args) => {
      const bundle = nativeFinish(...args);
      trace.nativeFinishReturned = true;
      trace.bundleId = identify(bundle, 'render-bundle');
      trace.sourceSnapshot = immediateSourceSnapshot(trace.events);
      trace.finishSequence = sequence ++;
      trace.events.push({ sequence: trace.finishSequence, method: 'finish' });
      if (trace.sourceSnapshot !== null && armedImmediateBundleFinishHook !== null) {
        const hook = armedImmediateBundleFinishHook;
        armedImmediateBundleFinishHook = null;
        const hookResult = hook({
          trace,
          bundle,
          bundleGpuId: trace.bundleId,
          finishSequence: trace.finishSequence,
          source: immediateSourceObjects.get(trace) ?? null,
          sourceSnapshot: structuredClone(trace.sourceSnapshot),
        });
        requireCondition(!(hookResult instanceof Promise),
          'The immediate bundle finish hook must be synchronous.');
        trace.finishHook = hookResult === undefined ? null : structuredClone(hookResult);
      }
      immediateSourceObjects.delete(trace);
      return bundle;
    });
    return encoder;
  });
  evidence.patchedDeviceMethods.push('createRenderBundleEncoder');

  replaceCallable(device, 'createCommandEncoder', (original) => (descriptor) => {
    const commandEncoder = original(descriptor);
    const commandEncoderId = identify(commandEncoder, 'command-encoder');
    const commandEncoderRecord = {
      sequence: sequence ++,
      capturePhase,
      commandEncoderId,
      descriptor: descriptorSnapshot(descriptor),
      renderPassEncoderIds: [],
      computePassEncoderIds: [],
      transferSequences: [],
      finishCallCount: 0,
      finishSequence: null,
      commandBufferId: null,
    };
    evidence.commandEncoders.push(commandEncoderRecord);
    for (const method of [
      'copyBufferToBuffer',
      'clearBuffer',
      'copyBufferToTexture',
      'copyTextureToBuffer',
      'copyTextureToTexture',
      'resolveQuerySet',
      'writeTimestamp',
    ]) {
      if (typeof commandEncoder[method] !== 'function') continue;
      replaceCallable(commandEncoder, method, (command) => (...args) => {
        const value = command(...args);
        const record = {
          sequence: sequence ++,
          capturePhase,
          commandEncoderId,
          method,
        };
        if (method === 'copyBufferToBuffer') {
          record.sourceBufferId = identify(args[0], 'buffer');
          record.sourceOffset = Number(args[1]);
          record.destinationBufferId = identify(args[2], 'buffer');
          record.destinationOffset = Number(args[3]);
          record.size = Number(args[4]);
        } else if (method === 'clearBuffer') {
          record.destinationBufferId = identify(args[0], 'buffer');
          record.destinationOffset = Number(args[1] ?? 0);
          record.size = args[2] == null ? null : Number(args[2]);
        } else if (method === 'copyBufferToTexture') {
          record.source = bufferCopySnapshot(args[0], identify);
          record.destination = textureCopySnapshot(args[1], identify);
          record.copySize = copyExtentSnapshot(args[2]);
        } else if (method === 'copyTextureToBuffer') {
          record.source = textureCopySnapshot(args[0], identify);
          record.destination = bufferCopySnapshot(args[1], identify);
          record.copySize = copyExtentSnapshot(args[2]);
        } else if (method === 'copyTextureToTexture') {
          record.source = textureCopySnapshot(args[0], identify);
          record.destination = textureCopySnapshot(args[1], identify);
          record.copySize = copyExtentSnapshot(args[2]);
        } else if (method === 'resolveQuerySet') {
          record.querySetId = identify(args[0], 'query-set');
          record.firstQuery = Number(args[1]);
          record.queryCount = Number(args[2]);
          record.destinationBufferId = identify(args[3], 'buffer');
          record.destinationOffset = Number(args[4]);
        } else if (method === 'writeTimestamp') {
          record.querySetId = identify(args[0], 'query-set');
          record.queryIndex = Number(args[1]);
        }
        evidence.commandEncoderTransfers.push(record);
        commandEncoderRecord.transferSequences.push(record.sequence);
        return value;
      });
    }
    replaceCallable(commandEncoder, 'beginRenderPass', (beginRenderPass) => (passDescriptor) => {
      const encoder = beginRenderPass(passDescriptor);
      const trace = {
        sequence: sequence ++,
        commandEncoderId,
        encoderId: identify(encoder, 'render-pass-encoder'),
        capturePhase,
        callableSetImmediates: typeof encoder.setImmediates === 'function',
        ...renderPassDescriptorSnapshot(passDescriptor, identify, viewTextureIds),
        events: [],
      };
      evidence.renderPassEncoders.push(trace);
      commandEncoderRecord.renderPassEncoderIds.push(trace.encoderId);
      wrapEncoder(encoder, trace);
      return encoder;
    });
    replaceCallable(commandEncoder, 'beginComputePass', (beginComputePass) => (passDescriptor) => {
      const encoder = beginComputePass(passDescriptor);
      const trace = {
        sequence: sequence ++,
        commandEncoderId,
        encoderId: identify(encoder, 'compute-pass-encoder'),
        capturePhase,
        label: passDescriptor?.label ?? null,
        timestampWrites: passDescriptor?.timestampWrites == null ? null : {
          querySetId: identify(passDescriptor.timestampWrites.querySet, 'query-set'),
          beginningOfPassWriteIndex:
            passDescriptor.timestampWrites.beginningOfPassWriteIndex ?? null,
          endOfPassWriteIndex:
            passDescriptor.timestampWrites.endOfPassWriteIndex ?? null,
        },
        events: [],
      };
      evidence.computePassEncoders.push(trace);
      commandEncoderRecord.computePassEncoderIds.push(trace.encoderId);
      wrapComputeEncoder(encoder, trace);
      return encoder;
    });
    replaceCallable(commandEncoder, 'finish', (finish) => (...args) => {
      const commandBuffer = finish(...args);
      commandEncoderRecord.finishCallCount += 1;
      commandEncoderRecord.finishSequence = sequence ++;
      commandEncoderRecord.commandBufferId = identify(commandBuffer, 'command-buffer');
      commandEncoderRecord.finishDescriptor = args[0] == null
        ? null
        : descriptorSnapshot(args[0]);
      return commandBuffer;
    });
    return commandEncoder;
  });
  evidence.patchedDeviceMethods.push('createCommandEncoder');

  return Object.freeze({
    evidence,
    identify,
    setCapturePhase(value) { capturePhase = value; },
    armImmediateBundleFinishHook(callback) {
      requireCondition(typeof callback === 'function',
        'Immediate bundle finish hook must be callable.');
      requireCondition(armedImmediateBundleFinishHook === null,
        'An immediate bundle finish hook is already armed.');
      armedImmediateBundleFinishHook = callback;
    },
    assertImmediateBundleFinishHookConsumed() {
      requireCondition(armedImmediateBundleFinishHook === null,
        'The armed immediate bundle finish hook was not consumed.');
    },
    async finalizeQueueWriteBufferSources() {
      for (const record of evidence.queueWriteBuffers) {
        const snapshot = queueWriteBufferSourceSnapshots.get(record);
        requireCondition(snapshot instanceof Uint8Array,
          'A queue.writeBuffer call has no retained source snapshot.', record);
        record.selectedSourceSha256 = await sha256Bytes(snapshot);
      }
      return evidence.queueWriteBuffers;
    },
    destroyUndestroyedPersistentBuffers() {
      const records = [];
      for (const [record, buffer] of bufferObjects) {
        if (record.resourceClass === 'readback-staging') continue;
        let attempted = false;
        let error = null;
        if (record.destroyCallCount === 0) {
          attempted = true;
          try {
            buffer.destroy();
          } catch (caught) {
            error = String(caught?.message ?? caught).slice(0, 4_096);
          }
        }
        records.push({
          resourceId: record.resourceId,
          attempted,
          error,
          destroyCallCount: record.destroyCallCount,
          destroyed: record.destroyed,
          destroySequence: record.destroySequence,
          destroyCapturePhase: record.destroyCapturePhase,
        });
      }
      return {
        pass: records.length > 0
          && records.every((record) => record.error === null
            && record.destroyed === true
            && record.destroyCallCount === 1),
        records,
      };
    },
    destroyUndestroyedPersistentTextures() {
      const records = [];
      for (const [record, texture] of textureObjects) {
        let attempted = false;
        let error = null;
        if (record.destroyCallCount === 0) {
          attempted = true;
          try {
            texture.destroy();
          } catch (caught) {
            error = String(caught?.message ?? caught).slice(0, 4_096);
          }
        }
        records.push({
          resourceId: record.resourceId,
          attempted,
          error,
          destroyCallCount: record.destroyCallCount,
          destroyed: record.destroyed,
          destroySequence: record.destroySequence,
          destroyCapturePhase: record.destroyCapturePhase,
        });
      }
      return {
        pass: records.length > 0
          && records.every((record) => record.error === null
            && record.destroyed === true
            && record.destroyCallCount === 1),
        records,
      };
    },
    mark(kind, detail = null) {
      requireCondition(typeof kind === 'string' && kind.length > 0,
        'Instrumentation marker kind must be a nonempty string.');
      const marker = {
        sequence: sequence ++,
        capturePhase,
        kind,
        detail: detail === null ? null : structuredClone(detail),
      };
      evidence.markers.push(marker);
      return structuredClone(marker);
    },
    async finalizeShaderEvidence() {
      for (const [record, module] of shaderObjects) {
        const information = await module.getCompilationInfo();
        record.compilationMessages = Array.from(information.messages, (message) => ({
          type: message.type,
          message: message.message,
          lineNum: message.lineNum,
          linePos: message.linePos,
          offset: message.offset,
          length: message.length,
        }));
        record.sha256 = await sha256Text(record.code ?? '');
      }
      return evidence.shaderModules;
    },
  });
}

function runtimeSignatureEvidence() {
  const setterSource = InstancedBufferGeometry.prototype.setIndirect.toString();
  const backendSource = WebGPUBackend.prototype._draw.toString();
  return {
    revision: REVISION,
    setIndirectHasImmediateBasesParameter:
      /setIndirect\( indirect, indirectOffset = 0, indirectImmediateBases = null \)/
        .test(setterSource),
    setIndirectStoresImmediateBases:
      /this\.indirectImmediateBases = indirectImmediateBases/.test(setterSource),
    backendHasSetImmediates:
      /setImmediates\( 0, indirectImmediateBases, i, 1 \)/.test(backendSource),
    backendHasIndexedIndirectAdjacencyMarker:
      /set one indexed-indirect base immediately before its draw/.test(backendSource),
  };
}

function freezeTransform(object) {
  object.matrixAutoUpdate = false;
  object.matrixWorldAutoUpdate = false;
}

function createCamera(renderer) {
  const camera = new PerspectiveCamera(CAMERA.fov, CAMERA.aspect, CAMERA.near, CAMERA.far);
  camera.position.fromArray(CAMERA.position);
  camera.lookAt(new Vector3().fromArray(CAMERA.target));
  realizeImmediateAifReversedDepthCamera(camera, renderer);
  return camera;
}

function createProductionScene(strategy) {
  const scene = new Scene();
  scene.name = 'phase0-production-scene';
  scene.background = new Color(0x030711);
  const hemisphere = new HemisphereLight(0x9bc5ff, 0x182038, 2.2);
  hemisphere.name = 'phase0-hemisphere-light';
  hemisphere.position.set(0, 1, 0);
  hemisphere.visible = true;
  hemisphere.layers.set(0);
  hemisphere.castShadow = false;
  hemisphere.receiveShadow = false;
  hemisphere.frustumCulled = true;
  hemisphere.renderOrder = 0;
  realizeAndFreezeImmediateAifTransform(hemisphere);
  const directional = new DirectionalLight(0xffffff, 3.2);
  directional.name = 'phase0-directional-light';
  directional.position.set(20, 35, 60);
  directional.visible = true;
  directional.layers.set(0);
  directional.castShadow = false;
  directional.receiveShadow = false;
  directional.frustumCulled = true;
  directional.renderOrder = 0;
  realizeAndFreezeImmediateAifTransform(directional);
  directional.target.name = 'phase0-directional-light-target';
  directional.target.position.set(0, 0, 0);
  directional.target.visible = true;
  directional.target.layers.set(0);
  directional.target.castShadow = false;
  directional.target.receiveShadow = false;
  directional.target.frustumCulled = true;
  directional.target.renderOrder = 0;
  realizeAndFreezeImmediateAifTransform(directional.target);
  scene.add(hemisphere);
  scene.add(directional);
  scene.add(directional.target);
  scene.add(strategy.root);
  scene.updateMatrixWorld(true);
  return scene;
}

function createProductionTarget() {
  return new RenderTarget(VIEWPORT.width, VIEWPORT.height, {
    depthBuffer: true,
    depthTexture: new DepthTexture(VIEWPORT.width, VIEWPORT.height, FloatType),
    stencilBuffer: false,
    samples: 0,
    format: RGBAFormat,
    type: UnsignedByteType,
    colorSpace: NoColorSpace,
  });
}

function createObjectIdMaterial(strategy, lane, scenario) {
  const shared = strategy.sharedResources;
  const matrixRead = storage(shared.attributes.matrix, 'mat4', scenario.objectCount).toReadOnly();
  const visibleRead = storage(
    shared.attributes.visibleIds,
    'uint',
    scenario.objectCount,
  ).toReadOnly();
  const addressMode = ADDRESS_MODE_BY_LANE[lane];
  const material = new MeshBasicNodeMaterial();
  material.name = `phase0-${lane}-object-id`;
  material.toneMapped = false;
  material.blending = NoBlending;
  material.side = DoubleSide;
  material.positionNode = Fn(() => {
    const address = createVisibleIdAddressNode({ addressMode })
      .toVar('phase0ObjectAddress');
    const objectId = visibleRead.element(address).toVar('phase0ObjectId');
    varyingProperty('uint', 'vPhase0ObjectId').assign(objectId);
    return matrixRead.element(objectId).mul(vec4(positionGeometry, 1)).xyz;
  })();
  material.colorNode = Fn(() => {
    const objectId = varyingProperty('uint', 'vPhase0ObjectId');
    const encoded = objectId.add(uint(1)).toVar('phase0EncodedObjectId');
    return vec4(
      float(encoded.bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(16)).bitAnd(uint(0xff))).div(255),
      1,
    );
  })();
  return material;
}

function createObjectIdDiagnostics(strategy, scenario) {
  const target = new RenderTarget(VIEWPORT.width, VIEWPORT.height, {
    depthBuffer: true,
    depthTexture: new DepthTexture(VIEWPORT.width, VIEWPORT.height, FloatType),
    stencilBuffer: false,
    samples: 0,
    format: RGBAFormat,
    type: UnsignedByteType,
    colorSpace: NoColorSpace,
  });
  const scenes = {};
  const materials = {};
  const roots = {};
  const meshes = {};
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const material = createObjectIdMaterial(strategy, lane, scenario);
    const mesh = new Mesh(strategy.laneStates[lane].geometry, material);
    mesh.name = `phase0-${lane}-object-id-mesh`;
    mesh.frustumCulled = false;
    freezeTransform(mesh);
    const root = new BundleGroup();
    root.name = `phase0-${lane}-object-id-bundle`;
    freezeTransform(root);
    root.add(mesh);
    const scene = new Scene();
    scene.add(root);
    materials[lane] = material;
    roots[lane] = root;
    meshes[lane] = mesh;
    scenes[lane] = scene;
  }
  return {
    target,
    scenes,
    materials,
    roots,
    meshes,
    markImmediateForRerecord() {
      roots.I.needsUpdate = true;
    },
    dispose() {
      target.dispose();
      for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
        scenes[lane].clear();
        materials[lane].dispose();
      }
    },
  };
}

function vertexSpans(sourceGeometries) {
  const spans = [];
  let cursor = 0;
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    const count = sourceGeometries[bucket].getAttribute('position')?.count;
    requireCondition(Number.isInteger(count) && count >= 3,
      `Geometry ${bucket} has no usable position stream.`);
    spans.push({ bucket, start: cursor, end: cursor + count });
    cursor += count;
  }
  return { spans, vertexCount: cursor };
}

function createChallengePositions({
  sourceGeometries,
  productionIndex,
  firstIndexes,
  sourceBaseByDraw,
  firstInstanceLane,
}) {
  const { spans, vertexCount } = vertexSpans(sourceGeometries);
  const positions = new Float32Array(vertexCount * 3);
  for (const span of spans) {
    const targetBase = IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketBases[span.bucket];
    const z = firstInstanceLane ? targetBase - sourceBaseByDraw[span.bucket] : targetBase;
    for (let vertex = span.start; vertex < span.end; vertex += 1) {
      positions[vertex * 3 + 2] = z;
    }
    const firstIndex = firstIndexes[span.bucket];
    const triangle = [
      productionIndex.array[firstIndex],
      productionIndex.array[firstIndex + 1],
      productionIndex.array[firstIndex + 2],
    ];
    requireCondition(new Set(triangle).size === 3
      && triangle.every((vertex) => vertex >= span.start && vertex < span.end),
    `Geometry ${span.bucket} first triangle is not independently addressable.`, triangle);
    for (let corner = 0; corner < triangle.length; corner += 1) {
      const vertex = triangle[corner];
      positions[vertex * 3] = ADDRESS_TRIANGLE[corner * 2];
      positions[vertex * 3 + 1] = ADDRESS_TRIANGLE[corner * 2 + 1];
    }
  }
  return { positions, spans };
}

function createAddressMaterial(strategy, lane) {
  const visibleRead = storage(
    strategy.sharedResources.attributes.visibleIds,
    'uint',
    IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount,
  ).toReadOnly();
  const addressMode = ADDRESS_MODE_BY_LANE[lane];
  const material = new MeshBasicNodeMaterial();
  material.name = `phase0-${lane}-all-address`;
  material.depthTest = false;
  material.depthWrite = false;
  material.toneMapped = false;
  material.blending = NoBlending;
  material.side = DoubleSide;
  material.vertexNode = Fn(() => {
    const sourceAddress = createVisibleIdAddressNode({ addressMode })
      .toVar('phase0SourceAddress');
    const objectId = visibleRead.element(sourceAddress).toVar('phase0AddressObjectId');
    varyingProperty('uint', 'vPhase0AddressObjectId').assign(objectId);
    const targetAddress = float(instanceIndex).add(positionGeometry.z)
      .toVar('phase0TargetAddress');
    const pixelX = targetAddress.mod(float(IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.width))
      .add(0.5);
    const pixelY = targetAddress.div(float(IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.width))
      .floor()
      .add(0.5);
    const center = vec2(
      pixelX.mul(2 / IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.width).sub(1),
      float(1).sub(pixelY.mul(2 / IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.height)),
    );
    const scale = vec2(
      2 / IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.width,
      2 / IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.height,
    );
    return vec4(center.add(positionGeometry.xy.mul(scale)), 0.5, 1);
  })();
  material.fragmentNode = Fn(() => {
    const objectId = varyingProperty('uint', 'vPhase0AddressObjectId');
    const encoded = objectId.add(uint(1)).toVar('phase0AddressEncodedId');
    return vec4(
      float(encoded.bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(16)).bitAnd(uint(0xff))).div(255),
      1,
    );
  })();
  return material;
}

function createAddressDiagnostics(strategy, sourceGeometries, camera) {
  const aGeometry = strategy.laneStates.A.geometry;
  const productionIndex = aGeometry.index;
  const firstIndexes = strategy.sharedResources.firstIndexes;
  const canonicalBases = IMMEDIATE_AIF_PHASE0_SCHEDULES.canonical.sourceBaseByDraw;
  const common = createChallengePositions({
    sourceGeometries,
    productionIndex,
    firstIndexes,
    sourceBaseByDraw: canonicalBases,
    firstInstanceLane: false,
  });
  const feature = createChallengePositions({
    sourceGeometries,
    productionIndex,
    firstIndexes,
    sourceBaseByDraw: canonicalBases,
    firstInstanceLane: true,
  });
  const commonPosition = new Float32BufferAttribute(common.positions, 3);
  const featurePosition = new Float32BufferAttribute(feature.positions, 3);
  const geometries = {};
  const scenes = {};
  const materials = {};
  const roots = {};
  const meshes = {};
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const geometry = new InstancedBufferGeometry();
    geometry.name = `phase0-${lane}-all-address-geometry`;
    geometry.setAttribute('position', lane === 'F' ? featurePosition : commonPosition);
    if (lane === 'A') {
      geometry.setAttribute('bucketBase', aGeometry.getAttribute('bucketBase'));
    }
    geometry.setIndex(productionIndex);
    geometry.instanceCount = IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCapacity;
    const laneState = strategy.laneStates[lane];
    geometry.setIndirect(
      laneState.indirectAttribute,
      Array.from(laneState.commandLayout.offsets),
      lane === 'I' ? laneState.indirectImmediateBases : null,
    );
    const material = createAddressMaterial(strategy, lane);
    const mesh = new Mesh(geometry, material);
    mesh.name = `phase0-${lane}-all-address-mesh`;
    mesh.frustumCulled = false;
    freezeTransform(mesh);
    const root = new BundleGroup();
    root.name = `phase0-${lane}-all-address-bundle`;
    freezeTransform(root);
    root.add(mesh);
    const scene = new Scene();
    scene.add(root);
    geometries[lane] = geometry;
    materials[lane] = material;
    roots[lane] = root;
    meshes[lane] = mesh;
    scenes[lane] = scene;
  }
  const target = new RenderTarget(
    IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.width,
    IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.height,
    {
      depthBuffer: false,
      stencilBuffer: false,
      samples: 0,
      format: RGBAFormat,
      type: UnsignedByteType,
      colorSpace: NoColorSpace,
    },
  );
  const targetCamera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  targetCamera.coordinateSystem = camera.coordinateSystem;
  targetCamera.position.z = 1;
  targetCamera.updateProjectionMatrix();

  function setImmediateBases(bases) {
    geometries.I.setIndirect(
      strategy.laneStates.I.indirectAttribute,
      Array.from(strategy.laneStates.I.commandLayout.offsets),
      bases,
    );
    roots.I.needsUpdate = true;
  }

  function setSchedule(scheduleId) {
    const sourceBases = IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId].sourceBaseByDraw;
    const array = featurePosition.array;
    for (const span of feature.spans) {
      const delta = IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketBases[span.bucket]
        - sourceBases[span.bucket];
      for (let vertex = span.start; vertex < span.end; vertex += 1) {
        array[vertex * 3 + 2] = delta;
      }
    }
    featurePosition.needsUpdate = true;
    setImmediateBases(strategy.laneStates.I.indirectImmediateBases);
  }

  return {
    target,
    camera: targetCamera,
    scenes,
    geometries,
    materials,
    roots,
    meshes,
    commonPosition,
    featurePosition,
    setImmediateBases,
    setSchedule,
    topologyEvidence() {
      return {
        pass: geometries.A.getAttribute('bucketBase')
            === strategy.laneStates.A.geometry.getAttribute('bucketBase')
          && geometries.I.getAttribute('bucketBase') === undefined
          && geometries.F.getAttribute('bucketBase') === undefined
          && geometries.A.index === productionIndex
          && geometries.I.index === productionIndex
          && geometries.F.index === productionIndex,
        target: { ...IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET },
        productionBucketBaseStreamReused: true,
        addedBucketBaseStreams: 0,
        bucketBaseAttributes: {
          A: geometries.A.getAttribute('bucketBase')?.id ?? null,
          I: null,
          F: null,
        },
        indexSharedAcrossLanes: new Set(
          IMMEDIATE_AIF_PHASE0_LANES.map((lane) => geometries[lane].index),
        ).size === 1,
        diagnosticPositionStreams: 2,
      };
    },
    dispose() {
      target.dispose();
      for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
        scenes[lane].clear();
        geometries[lane].setIndirect(null);
        geometries[lane].dispose();
        materials[lane].dispose();
      }
    },
  };
}

async function withRenderTarget(renderer, target, callback, {
  clearColor = 0x000000,
  clearAlpha = 0,
  clearDepth = true,
} = {}) {
  const previousTarget = renderer.getRenderTarget();
  const previousCubeFace = renderer.getActiveCubeFace();
  const previousMipmapLevel = renderer.getActiveMipmapLevel();
  const previousColor = renderer.getClearColor(new Color());
  const previousAlpha = renderer.getClearAlpha();
  const previousAutoClear = renderer.autoClear;
  try {
    renderer.setRenderTarget(target);
    renderer.setClearColor(clearColor, clearAlpha);
    renderer.autoClear = false;
    renderer.clear(true, clearDepth, false);
    return await callback();
  } finally {
    renderer.autoClear = previousAutoClear;
    renderer.setClearColor(previousColor, previousAlpha);
    renderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
  }
}

async function captureProductionOutput({ renderer, scene, camera, target }) {
  await withRenderTarget(renderer, target, async () => {
    renderer.render(scene, camera);
    await renderer.backend.device.queue.onSubmittedWorkDone();
  }, { clearColor: 0x030711, clearAlpha: 1, clearDepth: true });
  const color = await renderer.readRenderTargetPixelsAsync(
    target, 0, 0, VIEWPORT.width, VIEWPORT.height,
  );
  const depth = await renderer.backend.copyTextureToBuffer(
    target.depthTexture, 0, 0, VIEWPORT.width, VIEWPORT.height, 0,
  );
  requireCondition(color instanceof Uint8Array
    && color.byteLength === VIEWPORT.width * VIEWPORT.height * 4,
  'Production color readback has the wrong representation.');
  requireCondition(bytesOf(depth).byteLength === VIEWPORT.width * VIEWPORT.height * 4,
    'Production depth readback has the wrong representation.');
  const baselineRgba = Array.from(color.subarray(0, 4));
  let nonClearPixelCount = 0;
  for (let offset = 0; offset < color.length; offset += 4) {
    if (color[offset] !== baselineRgba[0]
      || color[offset + 1] !== baselineRgba[1]
      || color[offset + 2] !== baselineRgba[2]
      || color[offset + 3] !== baselineRgba[3]) nonClearPixelCount += 1;
  }
  const depthValues = new Float32Array(
    bytesOf(depth).buffer,
    bytesOf(depth).byteOffset,
    bytesOf(depth).byteLength / Float32Array.BYTES_PER_ELEMENT,
  );
  let nonzeroFiniteDepthCount = 0;
  let zeroDepthCount = 0;
  for (const value of depthValues) {
    if (value === 0) zeroDepthCount += 1;
    else if (Number.isFinite(value)) nonzeroFiniteDepthCount += 1;
  }
  return {
    color: {
      format: 'rgba8unorm',
      arrayType: color.constructor.name,
      byteLength: color.byteLength,
      sha256: await sha256Bytes(color),
      baselineRgba,
      nonClearPixelCount,
      contentPass: nonClearPixelCount > 0,
      rawBytes: color,
    },
    depth: {
      format: 'depth32float',
      arrayType: depth.constructor?.name ?? 'ArrayBuffer',
      byteLength: bytesOf(depth).byteLength,
      sha256: await sha256Bytes(depth),
      zeroDepthCount,
      nonzeroFiniteDepthCount,
      contentPass: zeroDepthCount > 0 && nonzeroFiniteDepthCount > 0,
      rawBytes: bytesOf(depth).slice(),
    },
  };
}

function bytesToBase64(value) {
  const bytes = bytesOf(value);
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function registerOutputWitness(runtime, {
  channel,
  output,
  context,
  capturePhase,
  observationKind,
  decoded = null,
}) {
  const rawBytes = output.rawBytes;
  requireCondition(rawBytes !== undefined,
    `Raw ${channel} bytes were discarded before witness registration.`);
  const bytes = bytesOf(rawBytes);
  const independentlyHashed = await sha256Bytes(bytes);
  requireCondition(independentlyHashed === output.sha256
    && bytes.byteLength === output.byteLength,
  `Raw ${channel} witness does not match its output record.`, {
      independentlyHashed,
      outputSha256: output.sha256,
      byteLength: bytes.byteLength,
      outputByteLength: output.byteLength,
    });
  const witnessId = `${channel}/${independentlyHashed}`;
  const prior = runtime.outputWitnesses.get(witnessId) ?? null;
  if (prior === null) {
    const retained = bytes.slice();
    runtime.outputWitnessRaw.set(witnessId, retained);
    runtime.outputWitnesses.set(witnessId, {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-output-byte-witness',
      witnessId,
      channel,
      format: output.format,
      arrayType: output.arrayType,
      byteLength: retained.byteLength,
      sha256: independentlyHashed,
      encoding: 'base64-exact-bytes',
      bytesBase64: bytesToBase64(retained),
      decoded: decoded === null ? null : structuredClone(decoded),
    });
  } else {
    const priorRaw = runtime.outputWitnessRaw.get(witnessId);
    requireCondition(prior.channel === channel
      && prior.format === output.format
      && prior.arrayType === output.arrayType
      && prior.byteLength === bytes.byteLength
      && priorRaw?.byteLength === bytes.byteLength
      && bytes.every((value, index) => value === priorRaw[index]),
    `Content-addressed ${channel} witness collision or metadata drift.`, {
        witnessId,
        prior,
      });
  }
  const observationId = `${capturePhase}/${channel}`;
  requireCondition(!runtime.outputWitnessObservationIds.has(observationId),
    `Duplicate output witness observation ${observationId}.`);
  runtime.outputWitnessObservationIds.add(observationId);
  runtime.outputWitnessObservations.push({
    ordinal: runtime.outputWitnessObservations.length + 1,
    observationId,
    observationKind,
    capturePhase,
    scenarioId: context.scenarioId,
    scheduleId: context.scheduleId,
    lane: context.lane,
    channel,
    witnessId,
    sha256: independentlyHashed,
    byteLength: bytes.byteLength,
  });
  output.witnessId = witnessId;
  delete output.rawBytes;
  return witnessId;
}

async function registerProductionOutputWitnesses(
  runtime,
  output,
  context,
  capturePhase,
  observationKind,
) {
  await registerOutputWitness(runtime, {
    channel: 'color',
    output: output.color,
    context,
    capturePhase,
    observationKind,
  });
  await registerOutputWitness(runtime, {
    channel: 'depth',
    output: output.depth,
    context,
    capturePhase,
    observationKind,
  });
}

function shaderDeclaredBindingCoordinates(stageModules) {
  const declarations = [];
  const declarationPattern = /((?:@\w+\s*\([^)]*\)\s*)+)var(?:\s*<\s*([^>]+)\s*>)?\s+([A-Za-z_]\w*)\s*:\s*([^;]+);/g;
  for (const { stage, module } of stageModules) {
    for (const match of module.code.matchAll(declarationPattern)) {
      const attributes = [...match[1].matchAll(
        /@(group|binding)\s*\(\s*(\d+)\s*\)/g,
      )];
      const group = attributes.find((entry) => entry[1] === 'group');
      const binding = attributes.find((entry) => entry[1] === 'binding');
      if (group === undefined || binding === undefined) continue;
      declarations.push({
        stage,
        moduleId: module.moduleId,
        variable: match[3],
        addressSpace: match[2]?.replace(/\s+/g, '') ?? null,
        declaredType: match[4].replace(/\s+/g, ''),
        group: Number(group[2]),
        binding: Number(binding[2]),
      });
    }
  }
  const coordinateMap = new Map();
  for (const declaration of declarations) {
    const key = `${declaration.group}/${declaration.binding}`;
    const coordinate = coordinateMap.get(key) ?? {
      group: declaration.group,
      binding: declaration.binding,
      declarations: [],
    };
    coordinate.declarations.push(declaration);
    coordinateMap.set(key, coordinate);
  }
  const coordinates = [...coordinateMap.values()].sort(
    (left, right) => left.group - right.group || left.binding - right.binding,
  );
  return { declarations, coordinates };
}

function shaderStageVisibility(stage) {
  if (stage === 'vertex') return 1;
  if (stage === 'fragment') return 2;
  if (stage === 'compute') return 4;
  return 0;
}

function nullBindingDescriptor() {
  return {
    buffer: null,
    sampler: null,
    texture: null,
    storageTexture: null,
    externalTexture: null,
  };
}

function wgslTextureViewDimension(token) {
  return ({
    '1d': '1d',
    '2d': '2d',
    '2d_array': '2d-array',
    cube: 'cube',
    cube_array: 'cube-array',
    '3d': '3d',
  })[token] ?? null;
}

function wgslBindingDescriptor(declaration) {
  const descriptor = nullBindingDescriptor();
  const addressParts = declaration.addressSpace?.split(',') ?? [];
  const addressSpace = addressParts[0] ?? null;
  const access = addressParts[1] ?? null;
  const declaredType = declaration.declaredType;
  if (addressSpace === 'uniform') {
    descriptor.buffer = {
      type: 'uniform',
      hasDynamicOffset: false,
      minBindingSize: 0,
    };
    return { pass: access === null, resourceClass: 'buffer', descriptor };
  }
  if (addressSpace === 'storage') {
    descriptor.buffer = {
      type: access === 'read_write' ? 'storage' : 'read-only-storage',
      hasDynamicOffset: false,
      minBindingSize: 0,
    };
    return {
      pass: access === null || access === 'read' || access === 'read_write',
      resourceClass: 'buffer',
      descriptor,
    };
  }
  if (addressSpace !== null) {
    return { pass: false, resourceClass: 'unsupported-address-space', descriptor };
  }
  if (declaredType === 'sampler' || declaredType === 'sampler_comparison') {
    descriptor.sampler = {
      type: declaredType === 'sampler_comparison' ? 'comparison' : 'filtering',
    };
    return { pass: true, resourceClass: 'sampler', descriptor };
  }
  if (declaredType === 'texture_external') {
    descriptor.externalTexture = {};
    return { pass: true, resourceClass: 'externalTexture', descriptor };
  }
  const storageTexture = declaredType.match(
    /^texture_storage_(1d|2d|2d_array|3d)<([^,>]+),(read|write|read_write)>$/,
  );
  if (storageTexture !== null) {
    descriptor.storageTexture = {
      access: ({
        read: 'read-only',
        write: 'write-only',
        read_write: 'read-write',
      })[storageTexture[3]],
      format: storageTexture[2],
      viewDimension: wgslTextureViewDimension(storageTexture[1]),
    };
    return {
      pass: descriptor.storageTexture.viewDimension !== null,
      resourceClass: 'storageTexture',
      descriptor,
    };
  }
  const multisampledTexture = declaredType.match(
    /^texture_multisampled_2d<(f32|i32|u32)>$/,
  );
  const sampledTexture = declaredType.match(
    /^texture_(1d|2d|2d_array|cube|cube_array|3d)<(f32|i32|u32)>$/,
  );
  const depthTexture = declaredType.match(
    /^texture_depth_(2d|2d_array|cube|cube_array)$/,
  );
  const depthMultisampledTexture = declaredType === 'texture_depth_multisampled_2d';
  if (multisampledTexture !== null || sampledTexture !== null
    || depthTexture !== null || depthMultisampledTexture) {
    const scalarType = multisampledTexture?.[1] ?? sampledTexture?.[2] ?? null;
    const viewToken = multisampledTexture !== null || depthMultisampledTexture
      ? '2d'
      : (sampledTexture?.[1] ?? depthTexture?.[1]);
    descriptor.texture = {
      sampleType: scalarType === 'i32'
        ? 'sint'
        : scalarType === 'u32'
          ? 'uint'
          : scalarType === null
            ? 'depth'
            : 'float',
      viewDimension: wgslTextureViewDimension(viewToken),
      multisampled: multisampledTexture !== null || depthMultisampledTexture,
    };
    return {
      pass: descriptor.texture.viewDimension !== null,
      resourceClass: 'texture',
      descriptor,
    };
  }
  return { pass: false, resourceClass: 'unsupported-type', descriptor };
}

function effectiveBindGroupLayoutEntry(entry) {
  const descriptor = nullBindingDescriptor();
  if (entry.buffer !== null && entry.buffer !== undefined) {
    descriptor.buffer = {
      type: entry.buffer.type ?? 'uniform',
      hasDynamicOffset: entry.buffer.hasDynamicOffset ?? false,
      minBindingSize: Number(entry.buffer.minBindingSize ?? 0),
    };
  }
  if (entry.sampler !== null && entry.sampler !== undefined) {
    descriptor.sampler = { type: entry.sampler.type ?? 'filtering' };
  }
  if (entry.texture !== null && entry.texture !== undefined) {
    descriptor.texture = {
      sampleType: entry.texture.sampleType ?? 'float',
      viewDimension: entry.texture.viewDimension ?? '2d',
      multisampled: entry.texture.multisampled ?? false,
    };
  }
  if (entry.storageTexture !== null && entry.storageTexture !== undefined) {
    descriptor.storageTexture = {
      access: entry.storageTexture.access ?? 'write-only',
      format: entry.storageTexture.format ?? null,
      viewDimension: entry.storageTexture.viewDimension ?? '2d',
    };
  }
  if (entry.externalTexture !== null && entry.externalTexture !== undefined) {
    descriptor.externalTexture = {};
  }
  const populatedClasses = Object.entries(descriptor)
    .filter(([, value]) => value !== null)
    .map(([key]) => key);
  return {
    pass: populatedClasses.length === 1,
    visibility: entry.visibility,
    resourceClass: populatedClasses.length === 1 ? populatedClasses[0] : null,
    descriptor,
  };
}

function bindingLayoutSemanticsEvidence(declared, layoutCoordinates) {
  return declared.coordinates.map((coordinate) => {
    const declarationSemantics = coordinate.declarations.map((declaration) => ({
      ...declaration,
      ...wgslBindingDescriptor(declaration),
    }));
    const descriptor = declarationSemantics[0]?.descriptor ?? null;
    const resourceClass = declarationSemantics[0]?.resourceClass ?? null;
    const declaredStageVisibility = declarationSemantics.reduce(
      (mask, declaration) => mask | shaderStageVisibility(declaration.stage),
      0,
    );
    const expectedVisibility = descriptor === null ? null
      : immediateAifExpectedBindingVisibility({
          resourceClass,
          bufferType: descriptor.buffer?.type ?? null,
          declaredStageVisibility,
        });
    const expected = descriptor === null ? null : {
      visibility: expectedVisibility,
      resourceClass,
      descriptor,
    };
    const layoutMatches = layoutCoordinates.filter(
      (record) => record.group === coordinate.group
        && record.binding === coordinate.binding,
    );
    const actual = layoutMatches.length === 1
      ? effectiveBindGroupLayoutEntry(layoutMatches[0].entry)
      : null;
    const declarationDescriptorsExact = declarationSemantics.length > 0
      && declarationSemantics.every((declaration) => declaration.pass
        && declaration.resourceClass === resourceClass
        && exactStructuredValue(declaration.descriptor, descriptor)
        && shaderStageVisibility(declaration.stage) !== 0);
    return {
      group: coordinate.group,
      binding: coordinate.binding,
      declarationSemantics,
      declaredStageVisibility,
      expectedVisibility,
      visibilityRule: resourceClass === 'buffer' && descriptor?.buffer?.type === 'uniform'
        ? 'pinned-three-node-uniforms-all-stages'
        : 'shader-declaration-stage-union',
      layoutMatchCount: layoutMatches.length,
      expected,
      actual,
      pass: declarationDescriptorsExact
        && actual?.pass === true
        && exactStructuredValue(actual, { pass: true, ...expected }),
    };
  });
}

function pipelineBindingTopologyEvidence({
  instrumentation,
  pipeline,
  layout,
  bindGroupLayouts,
  stageModules,
  events,
  finishSequence,
}) {
  const declared = shaderDeclaredBindingCoordinates(stageModules);
  const layoutCoordinates = bindGroupLayouts.flatMap((bindGroupLayout, group) => (
    bindGroupLayout.entries.map((entry) => ({ group, binding: entry.binding, entry }))
  ));
  const coordinateProjection = (records) => records.map(
    (record) => `${record.group}/${record.binding}`,
  ).sort();
  const bindGroupEvents = events.filter((event) => event.method === 'setBindGroup');
  const expectedGroupIndices = [...new Set(declared.coordinates.map(
    (record) => record.group,
  ))].sort((left, right) => left - right);
  const bindingSemantics = bindingLayoutSemanticsEvidence(
    declared,
    layoutCoordinates,
  );
  const boundGroups = expectedGroupIndices.map((group) => {
    const eventMatches = bindGroupEvents.filter((event) => event.index === group);
    const event = eventMatches.length === 1 ? eventMatches[0] : null;
    const bindGroups = instrumentation.evidence.resources.filter(
      (resource) => resource.method === 'createBindGroup'
        && resource.resourceId === event?.bindGroupId,
    );
    const bindGroup = bindGroups.length === 1 ? bindGroups[0] : null;
    const expectedLayoutId = layout.bindGroupLayoutIds[group] ?? null;
    const bindGroupLayout = bindGroupLayouts[group] ?? null;
    const expectedBindings = declared.coordinates.filter(
      (coordinate) => coordinate.group === group,
    ).map((coordinate) => coordinate.binding).sort((a, b) => a - b);
    const layoutBindings = bindGroupLayout?.entries.map(
      (entry) => entry.binding,
    ).sort((a, b) => a - b) ?? [];
    const entryBindings = bindGroup?.entries.map(
      (entry) => entry.binding,
    ).sort((a, b) => a - b) ?? [];
    return {
      group,
      eventMatchCount: eventMatches.length,
      setBindGroupEvent: event,
      bindGroupMatchCount: bindGroups.length,
      bindGroup,
      expectedLayoutId,
      bindGroupLayout,
      expectedBindings,
      layoutBindings,
      entryBindings,
      pass: eventMatches.length === 1
        && bindGroups.length === 1
        && bindGroupLayout !== null
        && bindGroup.layoutId === expectedLayoutId
        && bindGroupLayout.layoutId === expectedLayoutId
        && exactNumberArray(layoutBindings, expectedBindings)
        && exactNumberArray(entryBindings, expectedBindings)
        && new Set(layoutBindings).size === layoutBindings.length
        && new Set(entryBindings).size === entryBindings.length
        && event.dynamicOffsets === null
        && event.dynamicOffsetStart === null
        && event.dynamicOffsetLength === null
        && event.selectedDynamicOffsets === null
        && bindGroupLayout.sequence < pipeline.sequence
        && bindGroup.sequence < event.sequence
        && event.sequence < finishSequence,
    };
  });
  const pipelineEvents = events.filter((event) => event.method === 'setPipeline');
  const observedGroupIndices = bindGroupEvents.map(
    (event) => event.index,
  ).sort((left, right) => left - right);
  const expectedContiguousGroups = Array.from(
    { length: expectedGroupIndices.length },
    (_, index) => index,
  );
  const pass = declared.coordinates.length > 0
    && exactNumberArray(
      coordinateProjection(layoutCoordinates),
      coordinateProjection(declared.coordinates),
    )
    && layout.bindGroupLayoutIds.length === expectedGroupIndices.length
    && bindGroupLayouts.length === expectedGroupIndices.length
    && exactNumberArray(expectedGroupIndices, expectedContiguousGroups)
    && exactNumberArray(observedGroupIndices, expectedGroupIndices)
    && new Set(observedGroupIndices).size === observedGroupIndices.length
    && bindingSemantics.length === declared.coordinates.length
    && bindingSemantics.every((record) => record.pass)
    && boundGroups.every((record) => record.pass)
    && stageModules.every(({ module }) => module.sequence < pipeline.sequence)
    && layout.sequence < pipeline.sequence
    && pipelineEvents.length === 1
    && pipelineEvents[0].pipelineId === pipeline.pipelineId
    && pipeline.sequence < pipelineEvents[0].sequence
    && pipelineEvents[0].sequence < finishSequence;
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-pipeline-binding-topology',
    pass,
    pipelineId: pipeline.pipelineId,
    pipelineLayoutId: layout.layoutId,
    shaderDeclarations: declared.declarations,
    shaderCoordinates: declared.coordinates,
    layoutCoordinates,
    bindingSemantics,
    expectedGroupIndices,
    observedGroupIndices,
    boundGroups,
    pipelineEvents,
  };
}

async function captureDiagnosticProducer({
  runtime,
  diagnostics,
  diagnosticKind,
  lane,
  context,
  camera,
  capturePhase,
}) {
  const {
    renderer,
    instrumentation,
    strategy,
  } = runtime;
  const found = findProductionBundle(renderer, instrumentation, diagnostics.target, camera, {
    lane,
    root: diagnostics.roots[lane],
    mesh: diagnostics.meshes[lane],
  });
  const events = found.trace.events.map((event) => structuredClone(event));
  const draws = events.filter((event) => event.method === 'drawIndexedIndirect');
  const immediates = events.filter((event) => event.method === 'setImmediates');
  const expectedBases = Array.from(context.commandOracle.sourceBaseByDraw);
  const expectedOffsets = Array.from(context.commandOracle.offsets);
  const commandGpuBuffer = renderer.backend.get(context.indirectAttribute)?.buffer ?? null;
  const commandGpuBufferId = instrumentation.identify(commandGpuBuffer, 'buffer');
  const drawExact = draws.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
    && draws.every((event, index) => event.indirectOffset === expectedOffsets[index]
      && event.bufferId === commandGpuBufferId);
  const adjacent = draws.every((draw) => {
    const index = events.indexOf(draw);
    return lane !== 'I' || events[index - 1]?.method === 'setImmediates';
  });
  const sourceSnapshot = found.trace.sourceSnapshot === null
    ? null
    : structuredClone(found.trace.sourceSnapshot);
  const immediateExact = lane === 'I'
    ? immediates.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
      && sourceSnapshot?.pass === true
      && sourceSnapshot.recordedValues.every(
        (value, index) => value === expectedBases[index],
      )
      && immediates.every((event, index) => event.sourceId === sourceSnapshot.sourceId
        && event.backingBufferId === sourceSnapshot.backingBufferId
        && event.dataOffset === index
        && event.dataSize === 1
        && event.selectedValues?.[0] === expectedBases[index])
    : immediates.length === 0 && sourceSnapshot === null;
  const backendPipeline = renderer.backend.get(found.renderObject.pipeline);
  const pipelineId = instrumentation.identify(
    backendPipeline.pipeline,
    'render-pipeline',
  );
  const pipelineRecords = instrumentation.evidence.renderPipelines.filter(
    (record) => record.pipelineId === pipelineId,
  );
  requireCondition(pipelineRecords.length === 1,
    `${diagnosticKind}/${lane} pipeline descriptor is not unique.`, pipelineRecords);
  const pipeline = pipelineRecords[0];
  const vertexModules = instrumentation.evidence.shaderModules.filter(
    (record) => record.moduleId === pipeline.vertexModuleId,
  );
  const fragmentModules = instrumentation.evidence.shaderModules.filter(
    (record) => record.moduleId === pipeline.fragmentModuleId,
  );
  const layouts = instrumentation.evidence.pipelineLayouts.filter(
    (record) => record.layoutId === pipeline.layoutId,
  );
  const bindGroupLayouts = layouts.length === 1
    ? layouts[0].bindGroupLayoutIds.map((layoutId) => {
      const matches = instrumentation.evidence.bindGroupLayouts.filter(
        (record) => record.layoutId === layoutId,
      );
      requireCondition(matches.length === 1,
        `${diagnosticKind}/${lane} bind-group layout ${layoutId} is not unique.`, matches);
      return matches[0];
    })
    : [];
  requireCondition(vertexModules.length === 1
    && fragmentModules.length === 1
    && layouts.length === 1,
  `${diagnosticKind}/${lane} bound module/layout chain is not unique.`, pipeline);
  const vertexModule = vertexModules[0];
  const fragmentModule = fragmentModules[0];
  const builder = found.renderObject.getNodeBuilderState();
  requireCondition(builder.vertexShader === vertexModule.code
    && builder.fragmentShader === fragmentModule.code,
  `${diagnosticKind}/${lane} builder source differs from bound shader modules.`);
  const pipelineBindingTopology = pipelineBindingTopologyEvidence({
    instrumentation,
    pipeline,
    layout: layouts[0],
    bindGroupLayouts,
    stageModules: [
      { stage: 'vertex', module: vertexModule },
      { stage: 'fragment', module: fragmentModule },
    ],
    events,
    finishSequence: found.trace.finishSequence,
  });
  const vertexInputs = runtimeVertexInputEvidence(renderer, found.renderObject);
  const diagnosticGeometry = diagnostics.meshes[lane].geometry;
  const expectedVertexInputs = immediateAifExpectedDiagnosticVertexInputs({
    diagnosticKind,
    lane,
    resources: Object.fromEntries([
      ['position', diagnosticGeometry.getAttribute('position')],
      ...(lane === 'A'
        ? [['bucketBase', diagnosticGeometry.getAttribute('bucketBase')]]
        : []),
    ].map(([name, attribute]) => [name, {
      count: attribute?.count,
      resourceId: attribute?.id,
    }])),
  });
  const vertexInputsExact = exactStructuredValue(vertexInputs, expectedVertexInputs);
  const pipelineVertexInputs = (pipeline.descriptor.vertex?.buffers ?? [])
    .flatMap((buffer) => (buffer?.attributes ?? []).map((attribute) => ({
      shaderLocation: attribute.shaderLocation,
      format: attribute.format,
      stepMode: buffer.stepMode,
    })))
    .sort((left, right) => left.shaderLocation - right.shaderLocation);
  const expectedPipelineVertexInputs = expectedVertexInputs.map((input) => ({
    shaderLocation: input.shaderLocation,
    format: input.format,
    stepMode: input.stepMode,
  }));
  const pipelineVertexInputsExact = exactStructuredValue(
    pipelineVertexInputs,
    expectedPipelineVertexInputs,
  );
  const renderObjectAttributes = found.renderObject.getAttributes();
  const vertexBufferEvents = events.filter((event) => event.method === 'setVertexBuffer');
  const boundVertexInputs = vertexInputs.map((input, index) => {
    const attribute = renderObjectAttributes[index];
    const expectedGpuBufferId = instrumentation.identify(
      renderer.backend.get(attribute)?.buffer ?? null,
      'buffer',
    );
    const matches = vertexBufferEvents.filter(
      (event) => event.slot === index && event.bufferId === expectedGpuBufferId,
    );
    return {
      name: input.name,
      shaderLocation: input.shaderLocation,
      attributeId: attribute?.id ?? null,
      expectedGpuBufferId,
      observedGpuBufferIds: matches.map((event) => event.bufferId),
      pass: input.resourceId === attribute?.id && matches.length === 1,
    };
  });
  const boundVertexInputsExact = vertexBufferEvents.length === vertexInputs.length
    && boundVertexInputs.every((input) => input.pass);
  const indexBufferEvents = events.filter((event) => event.method === 'setIndexBuffer');
  const diagnosticIndex = diagnostics.meshes[lane].geometry.index;
  const expectedIndexGpuBufferId = instrumentation.identify(
    renderer.backend.get(diagnosticIndex)?.buffer ?? null,
    'buffer',
  );
  const mergedIndex = strategy.laneStates.A.geometry.index;
  const mergedIndexGpuBufferId = instrumentation.identify(
    renderer.backend.get(mergedIndex)?.buffer ?? null,
    'buffer',
  );
  const indexBinding = {
    pass: diagnosticIndex === mergedIndex
      && expectedIndexGpuBufferId === mergedIndexGpuBufferId
      && indexBufferEvents.length === 1
      && indexBufferEvents[0].bufferId === mergedIndexGpuBufferId
      && indexBufferEvents[0].indexFormat === 'uint32'
      && indexBufferEvents[0].offset === 0,
    attributeId: diagnosticIndex?.id ?? null,
    mergedAttributeId: mergedIndex?.id ?? null,
    expectedGpuBufferId: expectedIndexGpuBufferId,
    mergedGpuBufferId: mergedIndexGpuBufferId,
    events: indexBufferEvents,
  };
  const storageBindings = runtimeStorageBindingEvidence(
    vertexModule.code,
    found.renderObject,
    {
      matrixAttribute: diagnosticKind === 'object-id'
        ? strategy.sharedResources.attributes.matrix
        : null,
      visibleIdsAttribute: strategy.sharedResources.attributes.visibleIds,
    },
  );
  const bindGroupEvents = events.filter((event) => event.method === 'setBindGroup');
  const boundStorage = storageBindings.map((binding) => {
    const bindGroupEvent = bindGroupEvents.find((event) => event.index === binding.group);
    const bindGroups = instrumentation.evidence.resources.filter(
      (resource) => resource.method === 'createBindGroup'
        && resource.resourceId === bindGroupEvent?.bindGroupId,
    );
    const bindGroup = bindGroups.length === 1 ? bindGroups[0] : null;
    const entry = bindGroup?.entries?.find(
      (candidate) => candidate.binding === binding.binding,
    ) ?? null;
    const attribute = binding.semantic === 'matrix'
      ? strategy.sharedResources.attributes.matrix
      : strategy.sharedResources.attributes.visibleIds;
    const expectedGpuBuffer = renderer.backend.get(attribute)?.buffer ?? null;
    const expectedGpuBufferId = instrumentation.identify(expectedGpuBuffer, 'buffer');
    return {
      ...binding,
      bindGroupId: bindGroupEvent?.bindGroupId ?? null,
      bindGroupLayoutId: bindGroup?.layoutId ?? null,
      expectedGpuBufferId,
      observedGpuBufferId: entry?.bufferId ?? null,
      pass: bindGroups.length === 1
        && entry !== null
        && (bindGroupEvent.dynamicOffsets === null
          || bindGroupEvent.dynamicOffsets.length === 0)
        && bindGroupEvent.dynamicOffsetStart === null
        && bindGroupEvent.dynamicOffsetLength === null
        && entry.bufferId === expectedGpuBufferId,
    };
  });
  const executions = instrumentation.evidence.renderPassEncoders.filter((trace) => (
    trace.capturePhase === capturePhase
      && trace.events.some((event) => event.method === 'executeBundles'
        && event.bundleIds?.includes(found.bundleGpuId))
  ));
  const executionTrace = executions.length === 1 ? executions[0] : null;
  const executeEvents = executionTrace?.events.filter(
    (event) => event.method === 'executeBundles',
  ) ?? [];
  const executeEvent = executeEvents.length === 1 ? executeEvents[0] : null;
  const targetTexture = renderer.backend.get(diagnostics.target.texture)?.texture ?? null;
  const targetGpuTextureId = instrumentation.identify(targetTexture, 'texture');
  const colorAttachments = executionTrace?.colorAttachments ?? [];
  const colorTargetExact = colorAttachments.length === 1
    && colorAttachments[0]?.textureId === targetGpuTextureId;
  const targetIdentity = {
    renderTargetUuid: diagnostics.target.uuid,
    textureUuid: diagnostics.target.texture.uuid,
    gpuTextureId: targetGpuTextureId,
    colorAttachmentViewId: colorTargetExact ? colorAttachments[0].viewId : null,
    width: diagnostics.target.width,
    height: diagnostics.target.height,
    format: 'rgba8unorm',
    depthBuffer: diagnostics.target.depthBuffer === true,
    samples: diagnostics.target.samples,
  };
  const expectedImmediateSize = lane === 'I' ? REQUIRED_IMMEDIATE_BYTES : 0;
  const record = {
    schemaVersion: 1,
    kind: `immediate-aif-phase0-${diagnosticKind}-producer`,
    pass: found.data.version === diagnostics.roots[lane].version
      && found.trace.nativeFinishReturned
      && drawExact
      && adjacent
      && immediateExact
      && backendPipeline.immediateSize === expectedImmediateSize
      && events.filter((event) => event.method === 'setPipeline').length === 1
      && events.find((event) => event.method === 'setPipeline')?.pipelineId === pipelineId
      && vertexInputsExact
      && pipelineVertexInputsExact
      && boundVertexInputsExact
      && indexBinding.pass
      && pipelineBindingTopology.pass
      && boundStorage.every((binding) => binding.pass)
      && executions.length === 1
      && executeEvents.length === 1
      && executeEvent?.bundleIds?.length === 1
      && executeEvent.bundleIds[0] === found.bundleGpuId
      && targetIdentity.gpuTextureId !== null
      && colorTargetExact,
    diagnosticKind,
    lane,
    capturePhase,
    rootUuid: diagnostics.roots[lane].uuid,
    bundleGpuId: found.bundleGpuId,
    bundleVersion: diagnostics.roots[lane].version,
    finishSequence: found.trace.finishSequence,
    pipelineId,
    pipelineCreationSequence: pipeline.sequence,
    pipelineCreationPhase: pipeline.capturePhase,
    pipelineImmediateSize: backendPipeline.immediateSize ?? null,
    pipelineLayoutId: pipeline.layoutId,
    pipelineLayoutBindGroupLayoutIds: layouts[0].bindGroupLayoutIds,
    pipelineBindGroupLayouts: bindGroupLayouts,
    pipelineDescriptor: pipeline.descriptor,
    pipelineLayout: layouts[0],
    pipelineBindingTopology,
    vertexInputs,
    expectedVertexInputs,
    vertexInputsExact,
    pipelineVertexInputs,
    pipelineVertexInputsExact,
    boundVertexInputs,
    boundVertexInputsExact,
    indexBinding,
    vertexModuleId: vertexModule.moduleId,
    fragmentModuleId: fragmentModule.moduleId,
    boundModules: {
      vertex: {
        moduleId: vertexModule.moduleId,
        observationNonce: vertexModule.observationNonce,
        creationOrdinal: vertexModule.creationOrdinal,
        creationSequence: vertexModule.sequence,
        creationPhase: vertexModule.capturePhase,
        code: vertexModule.code,
      },
      fragment: {
        moduleId: fragmentModule.moduleId,
        observationNonce: fragmentModule.observationNonce,
        creationOrdinal: fragmentModule.creationOrdinal,
        creationSequence: fragmentModule.sequence,
        creationPhase: fragmentModule.capturePhase,
        code: fragmentModule.code,
      },
    },
    vertexShader: vertexModule.code,
    fragmentShader: fragmentModule.code,
    vertexSha256: await sha256Text(vertexModule.code),
    fragmentSha256: await sha256Text(fragmentModule.code),
    commandAttributeId: context.indirectAttribute.id,
    commandGpuBufferId,
    visibleIdsAttributeId: context.visibleIdsAttribute.id,
    visibleIdsGpuBufferId: instrumentation.identify(
      renderer.backend.get(context.visibleIdsAttribute)?.buffer ?? null,
      'buffer',
    ),
    storageBindings,
    boundStorage,
    drawCount: draws.length,
    setImmediatesCount: immediates.length,
    drawExact,
    immediateExact,
    adjacent,
    sourceSnapshot,
    rawEvents: events,
    traceSha256: await sha256Text(JSON.stringify(events)),
    execution: {
      pass: executions.length === 1
        && executeEvents.length === 1
        && executeEvent?.bundleIds?.length === 1
        && executeEvent.bundleIds[0] === found.bundleGpuId
        && colorTargetExact,
      renderPassEncoderId: executionTrace?.encoderId ?? null,
      executeSequence: executeEvent?.sequence ?? null,
      bundleIds: executeEvent?.bundleIds ?? [],
      cachedBundleId: found.bundleGpuId,
      colorAttachments,
      depthStencilAttachment: executionTrace?.depthStencilAttachment ?? null,
      colorTargetExact,
    },
    target: targetIdentity,
  };
  requireCondition(record.pass,
    `${diagnosticKind}/${lane} producer chain failed.`, record);
  return record;
}

function validateObjectIdPixels(pixels, scenario) {
  const visible = new Set(scenario.expectedVisibleIds);
  const observedCounts = new Map();
  let coveredPixelCount = 0;
  let backgroundPixelCount = 0;
  let outOfRangePixelCount = 0;
  let hiddenPixelCount = 0;
  let backgroundAlphaMismatchCount = 0;
  let activeAlphaMismatchCount = 0;
  for (let offset = 0; offset < pixels.length; offset += 4) {
    const encoded = pixels[offset]
      | (pixels[offset + 1] << 8)
      | (pixels[offset + 2] << 16);
    if (encoded === 0) {
      if (pixels[offset + 3] === 0) backgroundPixelCount += 1;
      else backgroundAlphaMismatchCount += 1;
      continue;
    }
    coveredPixelCount += 1;
    if (pixels[offset + 3] !== 0xff) activeAlphaMismatchCount += 1;
    const objectId = encoded - 1;
    if (objectId >= scenario.objectCount) outOfRangePixelCount += 1;
    else if (!visible.has(objectId)) hiddenPixelCount += 1;
    else observedCounts.set(objectId, (observedCounts.get(objectId) ?? 0) + 1);
  }
  const observedIdCounts = [...observedCounts]
    .sort(([left], [right]) => left - right)
    .map(([objectId, pixelCount]) => ({ objectId, pixelCount }));
  return {
    pass: coveredPixelCount > 0
      && outOfRangePixelCount === 0
      && hiddenPixelCount === 0
      && backgroundAlphaMismatchCount === 0
      && activeAlphaMismatchCount === 0,
    coveredPixelCount,
    backgroundPixelCount,
    outOfRangePixelCount,
    hiddenPixelCount,
    backgroundAlphaMismatchCount,
    activeAlphaMismatchCount,
    uniqueObservedIdCount: observedIdCounts.length,
    observedObjectIds: observedIdCounts.map(({ objectId }) => objectId),
    observedIdCounts,
  };
}

async function captureObjectIdOutput({
  runtime,
  diagnostics,
  camera,
  lane,
  context,
  scenario,
  capturePhase,
}) {
  const { renderer } = runtime;
  await withRenderTarget(renderer, diagnostics.target, async () => {
    renderer.render(diagnostics.scenes[lane], camera);
    await renderer.backend.device.queue.onSubmittedWorkDone();
  }, { clearColor: 0x000000, clearAlpha: 0, clearDepth: true });
  const producer = await captureDiagnosticProducer({
    runtime,
    diagnostics,
    diagnosticKind: 'object-id',
    lane,
    context,
    camera,
    capturePhase,
  });
  const pixels = await renderer.readRenderTargetPixelsAsync(
    diagnostics.target, 0, 0, VIEWPORT.width, VIEWPORT.height,
  );
  requireCondition(pixels instanceof Uint8Array
    && pixels.byteLength === VIEWPORT.width * VIEWPORT.height * 4,
  'Object-ID readback has the wrong representation.');
  const sha256 = await sha256Bytes(pixels);
  producer.outputSha256 = sha256;
  producer.outputByteLength = pixels.byteLength;
  return {
    format: 'rgba8unorm-object-id-plus-one',
    arrayType: pixels.constructor.name,
    byteLength: pixels.byteLength,
    sha256,
    validation: validateObjectIdPixels(pixels, scenario),
    producer,
    rawBytes: pixels,
  };
}

async function captureAddressOutput({
  runtime,
  diagnostics,
  lane,
  context,
  scenario,
  capturePhase,
}) {
  const { renderer } = runtime;
  await withRenderTarget(renderer, diagnostics.target, async () => {
    renderer.render(diagnostics.scenes[lane], diagnostics.camera);
    await renderer.backend.device.queue.onSubmittedWorkDone();
  }, { clearColor: 0x000000, clearAlpha: 0, clearDepth: false });
  const producer = await captureDiagnosticProducer({
    runtime,
    diagnostics,
    diagnosticKind: 'address',
    lane,
    context,
    camera: diagnostics.camera,
    capturePhase,
  });
  const actual = await renderer.readRenderTargetPixelsAsync(
    diagnostics.target,
    0,
    0,
    IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.width,
    IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.height,
  );
  requireCondition(actual instanceof Uint8Array
    && actual.byteLength === IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.byteLength,
  'All-address output has the wrong representation.');
  const expected = createImmediateAifPackedAddressBytes(context.addressOracle);
  const comparison = compareImmediateAifPackedAddressBytes({
    actual,
    expected,
    addressOracle: context.addressOracle,
    scenario,
  });
  const [sha256, expectedSha256] = await Promise.all([
    sha256Bytes(actual),
    sha256Bytes(expected),
  ]);
  producer.outputSha256 = sha256;
  producer.outputByteLength = actual.byteLength;
  return {
    pass: comparison.pass && sha256 === expectedSha256,
    format: IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.format,
    encoding: IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.encoding,
    width: IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.width,
    height: IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.height,
    byteLength: actual.byteLength,
    sha256,
    expectedSha256,
    activeCount: comparison.activeCount,
    paddingCount: comparison.paddingCount,
    exactExpectedBytes: comparison.exactExpectedBytes,
    mismatchCounts: comparison.mismatchCounts,
    producer,
    rawBytes: actual,
  };
}

async function primeDiagnosticLane({
  runtime,
  diagnostics,
  diagnosticKind,
  lane,
  context,
  camera,
  capturePhase,
}) {
  const clearDepth = diagnosticKind === 'object-id';
  await withRenderTarget(runtime.renderer, diagnostics.target, async () => {
    runtime.renderer.render(diagnostics.scenes[lane], camera);
    await runtime.renderer.backend.device.queue.onSubmittedWorkDone();
  }, { clearColor: 0x000000, clearAlpha: 0, clearDepth });
  return captureDiagnosticProducer({
    runtime,
    diagnostics,
    diagnosticKind,
    lane,
    context,
    camera,
    capturePhase,
  });
}

async function captureCommandEvidence(renderer, instrumentation, context, runtime) {
  const gpu = new Uint32Array(await renderer.getArrayBufferAsync(context.indirectAttribute));
  const expected = context.commandOracle.postCullCommands;
  const buffer = renderer.backend.get(context.indirectAttribute)?.buffer ?? null;
  const installation = runtime.activeScheduleInstallation;
  const preflight = installation?.commands?.[context.lane] ?? null;
  const gpuSha256 = await sha256Bytes(gpu);
  const expectedSha256 = await sha256Bytes(expected);
  const preflightBinding = {
    pass: installation?.scenarioId === context.scenarioId
      && installation.scheduleId === context.scheduleId
      && preflight?.lane === context.lane
      && preflight.attributeId === context.indirectAttribute.id
      && preflight.gpuBufferId === instrumentation.identify(buffer, 'buffer')
      && preflight.byteLength === gpu.byteLength
      && preflight.gpuSha256 === gpuSha256
      && preflight.frozenWords?.length === gpu.length
      && gpu.every((value, index) => value === preflight.frozenWords[index]),
    installationId: installation?.installationId ?? null,
    installationOrdinal: installation?.ordinal ?? null,
    scenarioId: installation?.scenarioId ?? null,
    scheduleId: installation?.scheduleId ?? null,
    preflightSha256: preflight?.gpuSha256 ?? null,
    preflightByteLength: preflight?.byteLength ?? null,
    frozenWordCount: preflight?.frozenWords?.length ?? null,
  };
  const wordsFour = Array.from(
    { length: IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount },
    (_, draw) => gpu[draw * INDIRECT_WORDS + 4],
  );
  return {
    pass: gpu.length === expected.length
      && gpu.every((value, index) => value === expected[index])
      && preflightBinding.pass,
    byteLength: gpu.byteLength,
    sha256: gpuSha256,
    expectedSha256,
    gpuBufferId: instrumentation.identify(buffer, 'buffer'),
    attributeId: context.indirectAttribute.id,
    version: context.indirectAttribute.version,
    offsets: [...context.commandOracle.offsets],
    gpuWords: Array.from(gpu),
    expectedWords: Array.from(expected),
    wordsFour,
    preflightBinding,
  };
}

function findProductionBundle(renderer, instrumentation, target, camera, context) {
  const renderContext = renderer?._renderContexts?.get(target, renderer._mrt);
  const renderBundle = renderer?._bundles?.get(context.root, camera, renderContext);
  const data = renderer?.backend?.get(renderBundle);
  const renderObjects = data?.renderObjects;
  requireCondition(Array.isArray(renderObjects)
    && renderObjects.length === 1
    && renderObjects[0]?.object === context.mesh
    && renderObjects[0]?.bundle === context.root,
  `Lane ${context.lane} production bundle topology is malformed.`);
  const bundleGpuId = instrumentation.identify(data.bundleGPU, 'render-bundle');
  const traces = instrumentation.evidence.renderBundleEncoders.filter(
    (trace) => trace.bundleId === bundleGpuId,
  );
  requireCondition(traces.length === 1,
    `Lane ${context.lane} production bundle trace is not unique.`, {
      bundleGpuId,
      traceCount: traces.length,
    });
  return { renderBundle, data, renderObject: renderObjects[0], bundleGpuId, trace: traces[0] };
}

async function captureBundleEvidence(
  renderer,
  instrumentation,
  target,
  camera,
  context,
  strategy,
  capturePhase,
) {
  const found = findProductionBundle(renderer, instrumentation, target, camera, context);
  const events = found.trace.events.map((event) => structuredClone(event));
  const draws = events.filter((event) => event.method === 'drawIndexedIndirect');
  const immediates = events.filter((event) => event.method === 'setImmediates');
  const expectedBases = Array.from(context.commandOracle.sourceBaseByDraw);
  const adjacent = draws.every((draw) => {
    const index = events.indexOf(draw);
    return context.lane !== 'I' || events[index - 1]?.method === 'setImmediates';
  });
  const immediateExact = context.lane === 'I'
    ? immediates.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
      && immediates.every((event, index) => (
        event.rangeOffset === 0
          && event.sourceType === 'Uint32Array'
          && event.sourceElementCount === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
          && event.dataOffset === index
          && event.dataSize === 1
          && event.selectedValues?.length === 1
          && event.selectedValues[0] === expectedBases[index]
      ))
    : immediates.length === 0;
  const drawExact = draws.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
    && draws.every((event, index) => event.indirectOffset === context.commandOracle.offsets[index]);
  const pipeline = renderer.backend.get(found.renderObject.pipeline);
  const pipelineId = instrumentation.identify(pipeline.pipeline, 'render-pipeline');
  const builder = found.renderObject.getNodeBuilderState();
  const pipelineRecords = instrumentation.evidence.renderPipelines.filter(
    (record) => record.pipelineId === pipelineId,
  );
  requireCondition(pipelineRecords.length === 1,
    `${context.lane} production pipeline descriptor is not unique.`, pipelineRecords);
  const pipelineRecord = pipelineRecords[0];
  const pipelineLayouts = instrumentation.evidence.pipelineLayouts.filter(
    (layout) => layout.layoutId === pipelineRecord.layoutId,
  );
  requireCondition(pipelineLayouts.length === 1,
    `${context.lane} production pipeline layout is not unique.`, pipelineRecord);
  const pipelineLayout = pipelineLayouts[0];
  const pipelineBindGroupLayouts = pipelineLayout.bindGroupLayoutIds.map((layoutId) => {
    const matches = instrumentation.evidence.bindGroupLayouts.filter(
      (layout) => layout.layoutId === layoutId,
    );
    requireCondition(matches.length === 1,
      `${context.lane} production bind-group layout is not unique.`, matches);
    return matches[0];
  });
  const vertexModules = instrumentation.evidence.shaderModules.filter(
    (module) => module.moduleId === pipelineRecord.vertexModuleId,
  );
  const fragmentModules = instrumentation.evidence.shaderModules.filter(
    (module) => module.moduleId === pipelineRecord.fragmentModuleId,
  );
  requireCondition(vertexModules.length === 1 && fragmentModules.length === 1
    && vertexModules[0].code === builder.vertexShader
    && fragmentModules[0].code === builder.fragmentShader,
  `${context.lane} production shader module chain is not unique.`);
  const pipelineBindingTopology = pipelineBindingTopologyEvidence({
    instrumentation,
    pipeline: pipelineRecord,
    layout: pipelineLayout,
    bindGroupLayouts: pipelineBindGroupLayouts,
    stageModules: [
      { stage: 'vertex', module: vertexModules[0] },
      { stage: 'fragment', module: fragmentModules[0] },
    ],
    events,
    finishSequence: found.trace.finishSequence,
  });
  const storageBindings = runtimeStorageBindingEvidence(
    builder.vertexShader,
    found.renderObject,
    {
      matrixAttribute: strategy.sharedResources.attributes.matrix,
      visibleIdsAttribute: strategy.sharedResources.attributes.visibleIds,
    },
  );
  const bindGroupEvents = events.filter((event) => event.method === 'setBindGroup');
  const boundStorage = storageBindings.map((binding) => {
    const bindGroupEvent = bindGroupEvents.find((event) => event.index === binding.group);
    const bindGroups = instrumentation.evidence.resources.filter(
      (resource) => resource.method === 'createBindGroup'
        && resource.resourceId === bindGroupEvent?.bindGroupId,
    );
    const bindGroup = bindGroups.length === 1 ? bindGroups[0] : null;
    const entry = bindGroup?.entries?.find(
      (candidate) => candidate.binding === binding.binding,
    ) ?? null;
    const attribute = binding.semantic === 'matrix'
      ? strategy.sharedResources.attributes.matrix
      : strategy.sharedResources.attributes.visibleIds;
    const expectedGpuBufferId = instrumentation.identify(
      renderer.backend.get(attribute)?.buffer ?? null,
      'buffer',
    );
    return {
      ...binding,
      bindGroupId: bindGroupEvent?.bindGroupId ?? null,
      bindGroupLayoutId: bindGroup?.layoutId ?? null,
      expectedGpuBufferId,
      observedGpuBufferId: entry?.bufferId ?? null,
      pass: bindGroups.length === 1
        && entry !== null
        && (bindGroupEvent.dynamicOffsets === null
          || bindGroupEvent.dynamicOffsets.length === 0)
        && bindGroupEvent.dynamicOffsetStart === null
        && bindGroupEvent.dynamicOffsetLength === null
        && entry.bufferId === expectedGpuBufferId,
    };
  });
  const boundStorageExact = exactNumberArray(
    boundStorage.map((binding) => binding.semantic).sort(),
    ['matrix', 'visibleIds'],
  ) && boundStorage.every((binding) => binding.pass);
  const expectedImmediateSize = context.lane === 'I' ? REQUIRED_IMMEDIATE_BYTES : 0;
  const sourceSnapshot = found.trace.sourceSnapshot === null
    ? null
    : structuredClone(found.trace.sourceSnapshot);
  const sourceSnapshotExact = context.lane === 'I'
    ? sourceSnapshot?.pass === true
      && sourceSnapshot.callCount === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
      && sourceSnapshot.sourceElementCount === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
      && sourceSnapshot.sourceByteOffset === 0
      && sourceSnapshot.sourceByteLength
        === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount * Uint32Array.BYTES_PER_ELEMENT
      && sourceSnapshot.dataOffsets.every((offset, index) => offset === index)
      && sourceSnapshot.dataSizes.every((size) => size === 1)
      && sourceSnapshot.recordedValues.every(
        (value, index) => value === expectedBases[index],
      )
    : sourceSnapshot === null;
  const executions = instrumentation.evidence.renderPassEncoders.filter((trace) => (
    trace.capturePhase === capturePhase
      && trace.events.some((event) => event.method === 'executeBundles'
        && event.bundleIds?.includes(found.bundleGpuId))
  ));
  const executionTrace = executions.at(-1) ?? null;
  const executeEvents = executionTrace?.events.filter(
    (event) => event.method === 'executeBundles',
  ) ?? [];
  const executeEvent = executeEvents.at(-1) ?? null;
  const targetTexture = renderer.backend.get(target.texture)?.texture ?? null;
  const targetGpuTextureId = instrumentation.identify(targetTexture, 'texture');
  const targetDepthTexture = target.depthTexture == null
    ? null
    : renderer.backend.get(target.depthTexture)?.texture ?? null;
  const targetDepthTextureId = instrumentation.identify(targetDepthTexture, 'texture');
  const colorAttachments = executionTrace?.colorAttachments ?? [];
  const depthStencilAttachment = executionTrace?.depthStencilAttachment ?? null;
  const colorTargetExact = colorAttachments.length === 1
    && colorAttachments[0]?.textureId === targetGpuTextureId;
  const depthTargetExact = target.depthBuffer !== true
    || (targetDepthTextureId !== null
      && depthStencilAttachment?.textureId === targetDepthTextureId);
  const execution = {
    pass: executions.length === 1
      && executeEvents.length === 1
      && executeEvent.bundleIds?.length === 1
      && executeEvent.bundleIds[0] === found.bundleGpuId
      && colorTargetExact
      && depthTargetExact,
    renderPassEncoderId: executionTrace?.encoderId ?? null,
    executeSequence: executeEvent?.sequence ?? null,
    bundleIds: executeEvent?.bundleIds ?? [],
    cachedBundleId: found.bundleGpuId,
    colorAttachments,
    depthStencilAttachment,
    colorTargetExact,
    depthTargetExact,
  };
  return {
    pass: found.data.version === context.root.version
      && found.trace.nativeFinishReturned
      && drawExact
      && immediateExact
      && sourceSnapshotExact
      && adjacent
      && execution.pass
      && pipelineBindingTopology.pass
      && boundStorageExact
      && pipeline.immediateSize === expectedImmediateSize,
    rootUuid: context.root.uuid,
    bundleGpuId: found.bundleGpuId,
    version: context.root.version,
    backendVersion: found.data.version,
    recordCount: strategy.laneStates[context.lane].bundleRecordCallbackCount,
    renderObjectCount: found.data.renderObjects.length,
    drawCount: draws.length,
    setImmediatesCount: immediates.length,
    drawExact,
    immediateExact,
    sourceSnapshotExact,
    adjacent,
    pipelineImmediateSize: pipeline.immediateSize ?? null,
    pipelineId,
    pipelineBindingTopology,
    storageBindings,
    boundStorage,
    boundStorageExact,
    execution,
    target: {
      renderTargetUuid: target.uuid,
      textureUuid: target.texture.uuid,
      gpuTextureId: targetGpuTextureId,
      depthTextureUuid: target.depthTexture?.uuid ?? null,
      depthGpuTextureId: targetDepthTextureId,
      width: target.width,
      height: target.height,
      samples: target.samples,
    },
    sourceSnapshot,
    finishSequence: found.trace.finishSequence ?? null,
    finishHook: found.trace.finishHook ?? null,
    rawEvents: events,
    traceSha256: await sha256Text(JSON.stringify(events)),
  };
}

async function captureProductionLaneCallback(runtime, context) {
  const {
    renderer,
    camera,
    productionScene,
    productionTarget,
    strategy,
    instrumentation,
  } = runtime;
  const phase = [
    'phase0', strategy.activeScenarioId, context.scheduleId, context.lane,
    `capture-${runtime.captureSerial ++}`,
  ].join('/');
  const phases = {
    production: `${phase}/production`,
    command: `${phase}/command-readback`,
    address: `${phase}/address`,
    objectId: `${phase}/object-id`,
    visibleIds: `${phase}/visible-id-readback`,
  };
  const installation = runtime.activeScheduleInstallation;
  requireCondition(installation?.scenarioId === context.scenarioId
    && installation.scheduleId === context.scheduleId
    && installation.pass === true,
  'Production callback is not bound to one exact schedule preflight.', {
      context: {
        scenarioId: context.scenarioId,
        scheduleId: context.scheduleId,
        lane: context.lane,
      },
      installationId: installation?.installationId ?? null,
    });
  const commonStateBefore = await captureCommonRenderStateCommitment(
    runtime,
    `${phase}/before-production`,
  );
  try {
    instrumentation.setCapturePhase(phases.production);
    const production = await captureProductionOutput({
      renderer,
      scene: productionScene,
      camera,
      target: productionTarget,
    });
    await registerProductionOutputWitnesses(
      runtime,
      production,
      context,
      phases.production,
      'challenged-render-callback',
    );
    const bundle = await captureBundleEvidence(
      renderer,
      instrumentation,
      productionTarget,
      camera,
      context,
      strategy,
      phases.production,
    );
    const commonStateAfter = await captureCommonRenderStateCommitment(
      runtime,
      `${phase}/after-production`,
    );
    const commonStateStable = commonStateBefore.sha256
        === installation.commonState.preflight.sha256
      && commonStateAfter.sha256 === installation.commonState.preflight.sha256
      && commonStateBefore.laneSelection.exact
      && commonStateAfter.laneSelection.exact
      && commonStateBefore.laneSelection.activeLane === context.lane
      && commonStateAfter.laneSelection.activeLane === context.lane;
    return {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-render-callback',
      pass: bundle.pass
        && production.color.contentPass
        && production.depth.contentPass
        && commonStateStable,
      enrichmentState: 'production-only',
      phase,
      phases,
      lane: context.lane,
      scenarioId: context.scenarioId,
      scheduleId: context.scheduleId,
      scheduleInstallationId: installation.installationId,
      schedulePreflightSha256: installation.commonState.preflight.sha256,
      commonState: {
        pass: commonStateStable,
        preflightSha256: installation.commonState.preflight.sha256,
        beforeSha256: commonStateBefore.sha256,
        afterSha256: commonStateAfter.sha256,
        beforeRawSha256: commonStateBefore.rawSha256,
        afterRawSha256: commonStateAfter.rawSha256,
        beforeLaneSelection: commonStateBefore.laneSelection,
        afterLaneSelection: commonStateAfter.laneSelection,
      },
      command: null,
      address: null,
      output: {
        width: VIEWPORT.width,
        height: VIEWPORT.height,
        color: production.color,
        depth: production.depth,
        objectId: null,
      },
      bundle,
      producerChainsDistinct: null,
    };
  } finally {
    instrumentation.setCapturePhase(null);
  }
}

async function enrichLaneCallback(runtime, context, callback) {
  const {
    renderer,
    camera,
    addressDiagnostics,
    objectIdDiagnostics,
    strategy,
    instrumentation,
  } = runtime;
  const { phases, bundle } = callback;
  requireCondition(callback?.enrichmentState === 'production-only'
    && callback.lane === context.lane
    && callback.scenarioId === context.scenarioId
    && callback.scheduleId === context.scheduleId,
  'Deferred callback enrichment received the wrong production callback.', {
      callback,
      context: {
        lane: context.lane,
        scenarioId: context.scenarioId,
        scheduleId: context.scheduleId,
      },
    });
  try {
    instrumentation.setCapturePhase(phases.command);
    const command = await captureCommandEvidence(renderer, instrumentation, context, runtime);
    instrumentation.setCapturePhase(phases.address);
    const address = await captureAddressOutput({
      runtime,
      diagnostics: addressDiagnostics,
      lane: context.lane,
      context,
      scenario: runtime.scenarios[strategy.activeScenarioId],
      capturePhase: phases.address,
    });
    const rawAddressBytes = address.rawBytes;
    delete address.rawBytes;
    const witnessId = `${context.scenarioId}/${context.scheduleId}/${context.lane}`;
    const priorWitness = runtime.addressWitnesses.get(witnessId) ?? null;
    if (priorWitness === null) {
      runtime.addressWitnessRaw.set(witnessId, rawAddressBytes.slice());
      runtime.addressWitnesses.set(witnessId, {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-address-byte-witness',
        witnessId,
        scenarioId: context.scenarioId,
        scheduleId: context.scheduleId,
        lane: context.lane,
        encoding: 'base64-rgba8unorm',
        byteLength: rawAddressBytes.byteLength,
        sha256: address.sha256,
        expectedSha256: address.expectedSha256,
        bytesBase64: bytesToBase64(rawAddressBytes),
      });
    } else {
      const priorRaw = runtime.addressWitnessRaw.get(witnessId);
      requireCondition(priorWitness.sha256 === address.sha256
        && priorWitness.byteLength === rawAddressBytes.byteLength
        && priorRaw?.length === rawAddressBytes.length
        && rawAddressBytes.every((value, index) => value === priorRaw[index]),
      `Address witness ${witnessId} changed across lane orders.`, {
          priorWitness,
          sha256: address.sha256,
        });
    }
    address.witnessId = witnessId;
    instrumentation.setCapturePhase(phases.objectId);
    const objectId = await captureObjectIdOutput({
      runtime,
      diagnostics: objectIdDiagnostics,
      camera,
      lane: context.lane,
      context,
      scenario: runtime.scenarios[strategy.activeScenarioId],
      capturePhase: phases.objectId,
    });
    const decodedObjectIds = objectId.validation;
    await registerOutputWitness(runtime, {
      channel: 'objectId',
      output: objectId,
      context,
      capturePhase: phases.objectId,
      observationKind: 'challenged-render-callback',
      decoded: decodedObjectIds,
    });
    objectId.validation = {
      pass: decodedObjectIds.pass,
      coveredPixelCount: decodedObjectIds.coveredPixelCount,
      backgroundPixelCount: decodedObjectIds.backgroundPixelCount,
      outOfRangePixelCount: decodedObjectIds.outOfRangePixelCount,
      hiddenPixelCount: decodedObjectIds.hiddenPixelCount,
      backgroundAlphaMismatchCount: decodedObjectIds.backgroundAlphaMismatchCount,
      activeAlphaMismatchCount: decodedObjectIds.activeAlphaMismatchCount,
      uniqueObservedIdCount: decodedObjectIds.uniqueObservedIdCount,
      decodedWitnessId: objectId.witnessId,
    };
    instrumentation.setCapturePhase(phases.visibleIds);
    const visibleIds = new Uint32Array(
      await renderer.getArrayBufferAsync(context.visibleIdsAttribute),
    );
    const visibleIdsGpuBuffer = renderer.backend.get(context.visibleIdsAttribute)?.buffer ?? null;
    address.visibleIdsAttributeId = context.visibleIdsAttribute.id;
    address.visibleIdsGpuBufferId = instrumentation.identify(visibleIdsGpuBuffer, 'buffer');
    address.visibleIdsSha256 = await sha256Bytes(visibleIds);
    address.visibleIdsCpuSha256 = await sha256Bytes(context.visibleIdsAttribute.array);
    address.visibleIdsExact = visibleIds.length === context.visibleIdsAttribute.array.length
      && visibleIds.every((value, index) => value === context.visibleIdsAttribute.array[index]);
    callback.output.objectId = objectId;
    const producerChainsDistinct = new Set([
      bundle.bundleGpuId,
      address.producer.bundleGpuId,
      objectId.producer.bundleGpuId,
    ]).size === 3
      && new Set([
        bundle.pipelineId,
        address.producer.pipelineId,
        objectId.producer.pipelineId,
      ]).size === 3;
    callback.command = command;
    callback.address = address;
    callback.producerChainsDistinct = producerChainsDistinct;
    callback.enrichmentState = 'complete';
    callback.pass = command.pass
      && bundle.pass
      && callback.commonState.pass
      && callback.output.color.contentPass
      && callback.output.depth.contentPass
      && address.pass
      && address.visibleIdsExact
      && address.producer.pass
      && objectId.validation.pass
      && objectId.producer.pass
      && producerChainsDistinct;
    return callback;
  } finally {
    instrumentation.setCapturePhase(null);
  }
}

async function captureLaneCallback(runtime, context) {
  const callback = await captureProductionLaneCallback(runtime, context);
  return enrichLaneCallback(runtime, context, callback);
}

function snapshotRelationalEvidence(laneOrders) {
  const callbacks = laneOrders.results.flatMap(
    (order) => order.records.map((record) => record.callback),
  );
  const fields = [
    ['color', (entry) => entry.output.color.sha256],
    ['depth', (entry) => entry.output.depth.sha256],
    ['objectId', (entry) => entry.output.objectId.sha256],
    ['address', (entry) => entry.address.sha256],
  ];
  const hashes = Object.fromEntries(fields.map(([name, select]) => [
    name,
    [...new Set(callbacks.map(select))],
  ]));
  return {
    pass: callbacks.length === 6 * 3
      && callbacks.every((entry) => entry?.pass === true)
      && Object.values(hashes).every((values) => values.length === 1),
    callbackCount: callbacks.length,
    allCallbacksPass: callbacks.every((entry) => entry?.pass === true),
    uniqueHashes: hashes,
  };
}

function challengedScenarioCallbacks(scenarios) {
  const callbacks = [];
  for (const scenario of scenarios) {
    for (const snapshot of scenario.snapshots) {
      if (snapshot.rerecord?.callback !== undefined) callbacks.push(snapshot.rerecord.callback);
      if (snapshot.laneOrders !== undefined) {
        for (const order of snapshot.laneOrders.results) {
          for (const record of order.records) callbacks.push(record.callback);
        }
      } else {
        callbacks.push(...snapshot.laneCaptures);
      }
    }
  }
  return callbacks;
}

function createOutputWitnessEvidence(runtime, scenarios) {
  const callbacks = challengedScenarioCallbacks(scenarios);
  const expectedCallbackCount = 126;
  const channels = ['color', 'depth', 'objectId'];
  const witnesses = Object.fromEntries(
    [...runtime.outputWitnesses].sort(([left], [right]) => left.localeCompare(right)),
  );
  const callbackReferences = callbacks.flatMap((callback, callbackOrdinal) => (
    channels.map((channel) => {
      const output = channel === 'objectId'
        ? callback.output.objectId
        : callback.output[channel];
      return {
        callbackOrdinal: callbackOrdinal + 1,
        phase: callback.phase,
        scenarioId: callback.scenarioId,
        scheduleId: callback.scheduleId,
        lane: callback.lane,
        channel,
        witnessId: output?.witnessId ?? null,
        sha256: output?.sha256 ?? null,
        byteLength: output?.byteLength ?? null,
      };
    })
  ));
  const objectIdBaselines = {};
  for (const scenarioId of ['v99', 'v20']) {
    for (const scheduleId of Object.keys(IMMEDIATE_AIF_PHASE0_SCHEDULES)) {
      const groupId = `${scenarioId}/${scheduleId}`;
      const laneWitnessIds = {};
      for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
        laneWitnessIds[lane] = [...new Set(callbacks
          .filter((callback) => callback.scenarioId === scenarioId
            && callback.scheduleId === scheduleId
            && callback.lane === lane)
          .map((callback) => callback.output.objectId.witnessId))];
      }
      const baselineWitnessId = laneWitnessIds.A.length === 1
        ? laneWitnessIds.A[0]
        : null;
      objectIdBaselines[groupId] = {
        scenarioId,
        scheduleId,
        baselineLane: 'A',
        baselineWitnessId,
        laneWitnessIds,
        exact: baselineWitnessId !== null
          && IMMEDIATE_AIF_PHASE0_LANES.every((lane) => (
            laneWitnessIds[lane].length === 1
              && laneWitnessIds[lane][0] === baselineWitnessId
          )),
      };
    }
  }
  const primeObservationCount = IMMEDIATE_AIF_PHASE0_LANES.length * 2;
  const expectedObservationCount = expectedCallbackCount * channels.length
    + primeObservationCount;
  const pass = callbacks.length === expectedCallbackCount
    && callbackReferences.every((reference) => (
      typeof reference.witnessId === 'string'
        && witnesses[reference.witnessId]?.sha256 === reference.sha256
        && witnesses[reference.witnessId]?.byteLength === reference.byteLength
        && witnesses[reference.witnessId]?.channel === reference.channel
    ))
    && runtime.outputWitnessObservations.length === expectedObservationCount
    && runtime.outputWitnessObservations.every((observation) => (
      witnesses[observation.witnessId]?.sha256 === observation.sha256
        && witnesses[observation.witnessId]?.byteLength === observation.byteLength
    ))
    && Object.values(objectIdBaselines).every((record) => record.exact);
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-output-byte-witnesses',
    pass,
    encoding: 'base64-exact-bytes',
    channels,
    expectedChallengedCallbackCount: expectedCallbackCount,
    challengedCallbackCount: callbacks.length,
    expectedObservationCount,
    observationCount: runtime.outputWitnessObservations.length,
    witnessCount: Object.keys(witnesses).length,
    witnesses,
    observations: runtime.outputWitnessObservations.map((record) => ({ ...record })),
    callbackReferences,
    objectIdBaselines,
  };
}

function compactLiveValidation(validation) {
  return {
    schemaVersion: validation.schemaVersion,
    kind: validation.kind,
    pass: validation.pass,
    scenarioId: validation.scenarioId,
    lane: validation.lane,
    scheduleId: validation.scheduleId,
    visibleReset: validation.visibleReset,
    commandExact: validation.commandExact,
    observedCanonicalOrder: validation.observedCanonicalOrder,
    paddingSentinelsExact: validation.paddingSentinelsExact,
    overflow: validation.overflow,
    correctness: validation.correctness,
    snapshot: {
      commands: Array.from(validation.snapshot.commands),
      visibleIds: Array.from(validation.snapshot.visibleIds),
      overflow: validation.snapshot.overflow,
    },
  };
}

async function registerGpuByteWitness(runtime, {
  bytes,
  semantic,
  gpuBufferId,
  capturePhase,
}) {
  const raw = bytesOf(bytes).slice();
  const sha256 = await sha256Bytes(raw);
  const witnessId = `sha256-${sha256}-${raw.byteLength}`;
  const prior = runtime.gpuByteWitnesses.get(witnessId) ?? null;
  if (prior === null) {
    runtime.gpuByteWitnessRaw.set(witnessId, raw);
    runtime.gpuByteWitnesses.set(witnessId, {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-gpu-byte-witness',
      witnessId,
      encoding: 'base64-exact-bytes',
      byteLength: raw.byteLength,
      sha256,
      bytesBase64: bytesToBase64(raw),
    });
  } else {
    const priorRaw = runtime.gpuByteWitnessRaw.get(witnessId);
    requireCondition(prior.sha256 === sha256
      && prior.byteLength === raw.byteLength
      && priorRaw?.length === raw.length
      && raw.every((value, index) => value === priorRaw[index]),
    `Content-addressed GPU byte witness ${witnessId} collided.`, {
        prior,
        semantic,
        gpuBufferId,
        capturePhase,
      });
  }
  runtime.gpuByteWitnessReferences.push({
    witnessId,
    semantic,
    gpuBufferId,
    capturePhase,
    byteLength: raw.byteLength,
    sha256,
  });
  return witnessId;
}

async function registerAddressPositionCpuWitness(runtime, bytes) {
  const raw = bytesOf(bytes).slice();
  const sha256 = await sha256Bytes(raw);
  const witnessId = `sha256-${sha256}-${raw.byteLength}`;
  const prior = runtime.addressPositionWitnesses.get(witnessId) ?? null;
  if (prior === null) {
    runtime.addressPositionWitnessRaw.set(witnessId, raw);
    runtime.addressPositionWitnesses.set(witnessId, {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-address-position-cpu-byte-witness',
      witnessId,
      encoding: 'base64-exact-bytes',
      byteLength: raw.byteLength,
      sha256,
      bytesBase64: bytesToBase64(raw),
    });
  } else {
    const priorRaw = runtime.addressPositionWitnessRaw.get(witnessId);
    requireCondition(prior.byteLength === raw.byteLength
      && prior.sha256 === sha256
      && priorRaw?.length === raw.length
      && raw.every((value, index) => value === priorRaw[index]),
    `Address-position CPU witness ${witnessId} collided.`, prior);
  }
  return witnessId;
}

function createAddressDiagnosticPositionEvidence(runtime) {
  const witnesses = Object.fromEntries(
    [...runtime.addressPositionWitnesses].sort(([left], [right]) => left.localeCompare(right)),
  );
  const records = runtime.scheduleInstallations.map(
    (installation) => installation.diagnosticPositionRealization,
  );
  const scheduleOrder = records.map((record) => `${record.scenarioId}/${record.scheduleId}`);
  const expectedScheduleOrder = ['v99', 'v20'].flatMap((scenarioId) => (
    ['canonical', 'S1', 'S2', 'canonical'].map(
      (scheduleId) => `${scenarioId}/${scheduleId}`,
    )
  ));
  const commonRecords = records.map((record) => record.records[0]);
  const featureRecords = records.map((record) => record.records[1]);
  const featureHashesBySchedule = Object.fromEntries(
    ['canonical', 'S1', 'S2'].map((scheduleId) => [
      scheduleId,
      [...new Set(records.filter((record) => record.scheduleId === scheduleId)
        .map((record) => record.records[1].cpuSha256))],
    ]),
  );
  const stableResourceIdentity = new Set(
    commonRecords.map((record) => `${record.attributeId}/${record.gpuBufferId}`),
  ).size === 1
    && new Set(
      featureRecords.map((record) => `${record.attributeId}/${record.gpuBufferId}`),
    ).size === 1;
  const exactScheduleContent = new Set(
    commonRecords.map((record) => record.cpuSha256),
  ).size === 1
    && Object.values(featureHashesBySchedule).every((hashes) => hashes.length === 1)
    && new Set(Object.values(featureHashesBySchedule).map((hashes) => hashes[0])).size === 3;
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-address-diagnostic-position-evidence',
    pass: records.length === 8
      && records.every((record) => record?.pass === true)
      && exactNumberArray(scheduleOrder, expectedScheduleOrder)
      && stableResourceIdentity
      && exactScheduleContent
      && Object.keys(witnesses).length === 4
      && Object.values(witnesses).every((witness) => (
        witness.witnessId === `sha256-${witness.sha256}-${witness.byteLength}`
          && witness.encoding === 'base64-exact-bytes'
      )),
    expectedRecordCount: 8,
    recordCount: records.length,
    witnessCount: Object.keys(witnesses).length,
    scheduleOrder,
    expectedScheduleOrder,
    stableResourceIdentity,
    exactScheduleContent,
    featureHashesBySchedule,
    witnesses,
    records,
  };
}

const QUEUE_WRITE_BUFFER_CATEGORY_ORDER = Object.freeze([
  'renderer-initialization',
  'scenario-matrix-realization',
  'live-compute',
  'shared-resource-commitment',
  'merged-geometry-realization',
  'schedule-attribute-realization',
  'schedule-diagnostic-position-realization',
  'production-render',
  'diagnostic-render',
]);

function queueWriteBufferCategory(capturePhase) {
  if (capturePhase === 'phase0/renderer-init') return 'renderer-initialization';
  if (/^phase0\/scenario-load\/(?:v99|v20)\/matrix-realization$/.test(capturePhase)) {
    return 'scenario-matrix-realization';
  }
  if (/^phase0\/live\/(?:v99|v20)\/[AIF]$/.test(capturePhase)) {
    return 'live-compute';
  }
  if (/^phase0\/resources\/(?:v99|v20)\/shared-commitment$/.test(capturePhase)) {
    return 'shared-resource-commitment';
  }
  if (capturePhase === 'phase0/resources/merged-geometry-realization') {
    return 'merged-geometry-realization';
  }
  if (/^phase0\/schedule-install\/(?:v99|v20)\/\d+-(?:canonical|S1|S2)\/attribute-realization$/.test(capturePhase)) {
    return 'schedule-attribute-realization';
  }
  if (/^phase0\/schedule-install\/(?:v99|v20)\/\d+-(?:canonical|S1|S2)\/diagnostic-position-realization$/.test(capturePhase)) {
    return 'schedule-diagnostic-position-realization';
  }
  if (/^phase0\/prime\/[AIF]$/.test(capturePhase)
    || /^phase0\/(?:v99|v20)\/(?:canonical|S1|S2)\/[AIF]\/capture-\d+\/production$/.test(capturePhase)) {
    return 'production-render';
  }
  if (/^phase0\/diagnostic-(?:prime|rerecord)\//.test(capturePhase)
    || /^phase0\/(?:v99|v20)\/(?:canonical|S1|S2)\/[AIF]\/capture-\d+\/(?:address|object-id)$/.test(capturePhase)) {
    return 'diagnostic-render';
  }
  return null;
}

function queueWriteBufferRawProjection(record) {
  return {
    sequence: record.sequence,
    capturePhase: record.capturePhase,
    bufferId: record.bufferId,
    bufferOffset: record.bufferOffset,
    sourceId: record.sourceId,
    sourceBackingBufferId: record.sourceBackingBufferId,
    sourceType: record.sourceType,
    sourceByteLength: record.sourceByteLength,
    sourceByteOffset: record.sourceByteOffset,
    sourceElementByteSize: record.sourceElementByteSize,
    sourceElementCount: record.sourceElementCount,
    dataOffset: record.dataOffset,
    size: record.size,
    selectedSourceElementOffset: record.selectedSourceElementOffset,
    selectedSourceElementLength: record.selectedSourceElementLength,
    selectedSourceByteOffset: record.selectedSourceByteOffset,
    selectedSourceByteLength: record.selectedSourceByteLength,
    selectedSourceSha256: record.selectedSourceSha256,
  };
}

function createQueueWriteBufferLedger(runtime, scenarios) {
  const { instrumentation, renderer, strategy, addressDiagnostics } = runtime;
  const rawRecords = instrumentation.evidence.queueWriteBuffers;
  const semanticBindings = [];
  const addSemanticBinding = (semantic, attribute) => {
    semanticBindings.push({
      semantic,
      attributeId: attribute.id,
      bufferId: instrumentation.identify(
        renderer.backend.get(attribute)?.buffer ?? null,
        'buffer',
      ),
    });
  };
  for (const [semantic, attribute] of Object.entries(strategy.sharedResources.attributes)) {
    addSemanticBinding(`shared.${semantic}`, attribute);
  }
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    addSemanticBinding(`command.${lane}`, strategy.laneStates[lane].indirectAttribute);
    const geometry = strategy.laneStates[lane].geometry;
    for (const [semantic, attribute] of Object.entries(geometry.attributes)) {
      addSemanticBinding(`geometry.${lane}.${semantic}`, attribute);
    }
    addSemanticBinding(`geometry.${lane}.index`, geometry.index);
  }
  addSemanticBinding('diagnostic.address.commonPosition', addressDiagnostics.commonPosition);
  addSemanticBinding('diagnostic.address.featurePosition', addressDiagnostics.featurePosition);
  const bufferCreations = instrumentation.evidence.resources.filter(
    (resource) => resource.method === 'createBuffer',
  );
  const bindGroups = instrumentation.evidence.resources.filter(
    (resource) => resource.method === 'createBindGroup',
  );
  const bindGroupLayouts = instrumentation.evidence.bindGroupLayouts;
  const uniformBindingProofs = (bufferId) => {
    const occurrences = bindGroups.flatMap((bindGroup) => (
      bindGroup.entries.filter((entry) => entry.bufferId === bufferId).map((entry) => ({
        bindGroup,
        entry,
      }))
    ));
    if (occurrences.length !== 1) return [];
    return occurrences.flatMap(({ bindGroup, entry }) => {
      const layoutMatches = bindGroupLayouts.filter(
        (layout) => layout.layoutId === bindGroup.layoutId,
      );
      const coordinateMatches = layoutMatches.flatMap((layout) => layout.entries.filter(
        (candidate) => candidate.binding === entry.binding,
      ));
      const effectiveBuffer = immediateAifBufferBindingLayoutSnapshot(
        coordinateMatches[0]?.buffer,
      );
      const effectiveBufferType = effectiveBuffer?.type ?? null;
      if (layoutMatches.length !== 1
        || coordinateMatches.length !== 1
        || coordinateMatches[0].buffer === null
        || effectiveBufferType !== 'uniform') return [];
      return [{
        exact: true,
        bindGroupId: bindGroup.resourceId,
        bindGroupCreateSequence: bindGroup.sequence,
        bindGroupCapturePhase: bindGroup.capturePhase,
        bindGroupLayoutId: bindGroup.layoutId,
        bindGroupLayoutCreateSequence: layoutMatches[0].sequence,
        binding: entry.binding,
        bufferId: entry.bufferId,
        bufferOffset: entry.bufferOffset,
        bufferSize: entry.bufferSize,
        bufferRawType: effectiveBuffer.rawType,
        bufferType: effectiveBufferType,
        hasDynamicOffset: coordinateMatches[0].buffer.hasDynamicOffset,
        minBindingSize: coordinateMatches[0].buffer.minBindingSize,
        layoutMatchCount: layoutMatches.length,
        coordinateMatchCount: coordinateMatches.length,
      }];
    });
  };
  const commandCompletionProof = (trace, useSequence, useKind, extra = {}) => {
    const commandEncoders = instrumentation.evidence.commandEncoders.filter(
      (encoder) => encoder.commandEncoderId === trace.commandEncoderId,
    );
    const commandEncoder = commandEncoders.length === 1 ? commandEncoders[0] : null;
    const submissions = commandEncoder === null ? []
      : instrumentation.evidence.queueSubmissions.filter((submission) => (
        exactNumberArray(submission.commandBufferIds, [commandEncoder.commandBufferId])
      ));
    const submission = submissions.length === 1 ? submissions[0] : null;
    return {
      useKind,
      useSequence,
      passEncoderId: trace.encoderId,
      commandEncoderId: trace.commandEncoderId,
      commandEncoderMatchCount: commandEncoders.length,
      commandBufferId: commandEncoder?.commandBufferId ?? null,
      commandEncoderFinishSequence: commandEncoder?.finishSequence ?? null,
      submissionMatchCount: submissions.length,
      submitSequence: submission?.sequence ?? null,
      ...extra,
      pass: commandEncoders.length === 1
        && submissions.length === 1
        && trace.capturePhase === submission.capturePhase
        && useSequence < commandEncoder.finishSequence
        && commandEncoder.finishSequence < submission.sequence,
    };
  };
  const activeBindGroupUseWitness = (
    events,
    bindGroupId,
    writeSequence,
    usePredicate,
  ) => {
    const activeSlots = new Map();
    for (const event of events) {
      if (event.method === 'setBindGroup') {
        activeSlots.set(event.index, {
          bindGroupId: event.bindGroupId,
          sequence: event.sequence,
        });
        continue;
      }
      if (!usePredicate(event) || event.sequence <= writeSequence) continue;
      const activeMatches = [...activeSlots].filter(
        ([, active]) => active.bindGroupId === bindGroupId
          && active.sequence < event.sequence,
      );
      if (activeMatches.length !== 1) continue;
      const [[bindGroupIndex, active]] = activeMatches;
      return {
        useEvent: event,
        bindGroupIndex,
        activeSetBindGroupSequence: active.sequence,
        activeTargetBindGroupMatchCount: activeMatches.length,
      };
    }
    return null;
  };
  const uniformExecutionProofs = (bindGroupId, capturePhase, writeSequence) => {
    const proofs = [];
    for (const trace of instrumentation.evidence.computePassEncoders.filter(
      (candidate) => candidate.capturePhase === capturePhase,
    )) {
      const witness = activeBindGroupUseWitness(
        trace.events,
        bindGroupId,
        writeSequence,
        (event) => event.method === 'dispatchWorkgroups'
          || event.method === 'dispatchWorkgroupsIndirect',
      );
      if (witness !== null) {
        proofs.push(commandCompletionProof(trace, witness.useEvent.sequence, 'compute-dispatch', {
          setBindGroupSequence: witness.activeSetBindGroupSequence,
          bindGroupIndex: witness.bindGroupIndex,
          activeSetBindGroupSequence: witness.activeSetBindGroupSequence,
          activeTargetBindGroupMatchCount: witness.activeTargetBindGroupMatchCount,
          bindGroupId,
        }));
      }
    }
    for (const trace of instrumentation.evidence.renderPassEncoders.filter(
      (candidate) => candidate.capturePhase === capturePhase,
    )) {
      const directWitness = activeBindGroupUseWitness(
        trace.events,
        bindGroupId,
        writeSequence,
        (event) => /^draw/.test(event.method),
      );
      let directProof = null;
      if (directWitness !== null) {
        directProof = commandCompletionProof(
          trace,
          directWitness.useEvent.sequence,
          'render-pass-draw',
          {
            setBindGroupSequence: directWitness.activeSetBindGroupSequence,
            bindGroupIndex: directWitness.bindGroupIndex,
            activeSetBindGroupSequence: directWitness.activeSetBindGroupSequence,
            activeTargetBindGroupMatchCount:
              directWitness.activeTargetBindGroupMatchCount,
            bindGroupId,
          },
        );
      }
      let bundleProof = null;
      for (const execute of trace.events.filter(
        (event) => event.method === 'executeBundles' && writeSequence < event.sequence,
      )) {
        for (const bundleId of execute.bundleIds ?? []) {
          const bundleMatches = instrumentation.evidence.renderBundleEncoders.filter(
            (bundle) => bundle.bundleId === bundleId,
          );
          const bundle = bundleMatches.length === 1 ? bundleMatches[0] : null;
          const bundleWitness = bundle === null ? null : activeBindGroupUseWitness(
            bundle.events,
            bindGroupId,
            Number.NEGATIVE_INFINITY,
            (event) => /^draw/.test(event.method),
          );
          if (bundle === null || bundleWitness === null) continue;
          const proof = commandCompletionProof(
            trace,
            execute.sequence,
            'executed-render-bundle',
            {
              bindGroupId,
              bindGroupIndex: bundleWitness.bindGroupIndex,
              activeSetBindGroupSequence: bundleWitness.activeSetBindGroupSequence,
              activeTargetBindGroupMatchCount:
                bundleWitness.activeTargetBindGroupMatchCount,
              bundleId,
              bundleMatchCount: bundleMatches.length,
              bundleSetBindGroupSequence: bundleWitness.activeSetBindGroupSequence,
              bundleDrawSequence: bundleWitness.useEvent.sequence,
              bundleFinishSequence: bundle.finishSequence,
            },
          );
          proof.pass = proof.pass
            && bundleMatches.length === 1
            && bundleWitness.activeSetBindGroupSequence < bundleWitness.useEvent.sequence
            && bundle.finishSequence < execute.sequence;
          bundleProof = proof;
          break;
        }
        if (bundleProof !== null) break;
      }
      const selectedTraceProof = [directProof, bundleProof]
        .filter((proof) => proof !== null)
        .sort((left, right) => left.useSequence - right.useSequence)[0] ?? null;
      if (selectedTraceProof !== null) proofs.push(selectedTraceProof);
    }
    return proofs;
  };
  const records = rawRecords.map((record) => ({
    category: queueWriteBufferCategory(record.capturePhase),
    ...queueWriteBufferRawProjection(record),
    destinationOwner: (() => {
      const creations = bufferCreations.filter(
        (resource) => resource.resourceId === record.bufferId,
      );
      const creation = creations.length === 1 ? creations[0] : null;
      const matchedSemanticBindings = semanticBindings.filter(
        (binding) => binding.bufferId === record.bufferId,
      ).map(({ semantic, attributeId }) => ({ semantic, attributeId }));
      const matchedUniformBindingProofs = uniformBindingProofs(record.bufferId);
      const destinationBindingClass = matchedSemanticBindings.length > 0
        ? 'known-attribute'
        : matchedUniformBindingProofs.length > 0
          ? 'uniform-bind-group-buffer'
          : 'unclassified-persistent-buffer';
      return {
        exact: creation !== null
          && (destinationBindingClass === 'known-attribute'
            || destinationBindingClass === 'uniform-bind-group-buffer'),
        resourceId: record.bufferId,
        resourceClass: creation?.resourceClass ?? null,
        createSequence: creation?.sequence ?? null,
        createCapturePhase: creation?.capturePhase ?? null,
        size: creation?.size ?? null,
        usage: creation?.usage ?? null,
        mappedAtCreation: creation?.mappedAtCreation ?? null,
        destinationBindingClass,
        semanticBindings: matchedSemanticBindings,
        uniformBindingProofs: matchedUniformBindingProofs,
      };
    })(),
  }));
  const partitions = QUEUE_WRITE_BUFFER_CATEGORY_ORDER.map((category) => {
    const categoryRecords = records.filter((record) => record.category === category);
    return { category, count: categoryRecords.length, records: categoryRecords };
  });
  const unclassified = records.filter((record) => record.category === null);
  const specializedSourceDefinitions = [
    ...scenarios.map((scenario) => ({
      ownerKind: 'scenario-matrix-realization',
      ownerId: scenario.scenarioId,
      capturePhase: scenario.matrixGpuRealization.capturePhase,
      records: scenario.matrixGpuRealization.queueWriteBuffers,
    })),
    ...scenarios.map((scenario) => ({
      ownerKind: 'shared-resource-commitment',
      ownerId: scenario.scenarioId,
      capturePhase: scenario.gpuResourceCommitments.capturePhase,
      records: scenario.gpuResourceCommitments.queueWriteBuffers,
    })),
    ...runtime.scheduleInstallations.map((installation) => ({
      ownerKind: 'schedule-attribute-realization',
      ownerId: installation.installationId,
      capturePhase: `${installation.basePhase}/attribute-realization`,
      records: installation.queueWriteBuffers,
    })),
    ...runtime.scheduleInstallations.map((installation) => ({
      ownerKind: 'schedule-diagnostic-position-realization',
      ownerId: installation.installationId,
      capturePhase: installation.diagnosticPositionRealization.capturePhase,
      records: installation.diagnosticPositionRealization.queueWriteBuffers,
    })),
  ];
  for (const source of specializedSourceDefinitions) {
    for (const record of source.records) {
      const raw = rawRecords.find((candidate) => candidate.sequence === record.sequence);
      if (raw !== undefined) {
        Object.assign(record, {
          sourceId: raw.sourceId,
          sourceBackingBufferId: raw.sourceBackingBufferId,
          sourceByteOffset: raw.sourceByteOffset,
          sourceElementByteSize: raw.sourceElementByteSize,
          sourceElementCount: raw.sourceElementCount,
          selectedSourceElementOffset: raw.selectedSourceElementOffset,
          selectedSourceElementLength: raw.selectedSourceElementLength,
          selectedSourceByteOffset: raw.selectedSourceByteOffset,
          selectedSourceByteLength: raw.selectedSourceByteLength,
          selectedSourceSha256: raw.selectedSourceSha256,
        });
      }
    }
  }
  const specializedSources = specializedSourceDefinitions.map((source) => {
    const sequences = source.records.map((record) => record.sequence);
    const rawMatches = rawRecords.filter((record) => sequences.includes(record.sequence));
    return {
      ownerKind: source.ownerKind,
      ownerId: source.ownerId,
      capturePhase: source.capturePhase,
      recordCount: source.records.length,
      sequences,
      records: source.records,
      rawExact: rawMatches.length === source.records.length
        && source.records.every((record) => rawMatches.some((raw) => (
          exactStructuredValue(
            queueWriteBufferRawProjection(raw),
            queueWriteBufferRawProjection(record),
          )
        ))),
    };
  });
  const specializedCategoryNames = new Set([
    'scenario-matrix-realization',
    'shared-resource-commitment',
    'schedule-attribute-realization',
    'schedule-diagnostic-position-realization',
  ]);
  const specializedLedgerSequences = records.filter(
    (record) => specializedCategoryNames.has(record.category),
  ).map((record) => record.sequence).sort((left, right) => left - right);
  const specializedSourceSequences = specializedSources.flatMap(
    (source) => source.sequences,
  ).sort((left, right) => left - right);
  const specializedCounts = Object.fromEntries([...specializedCategoryNames].map(
    (category) => [category, records.filter((record) => record.category === category).length],
  ));
  const expectedSpecializedCounts = {
    'scenario-matrix-realization': 1,
    'shared-resource-commitment': 2,
    'schedule-attribute-realization': 13,
    'schedule-diagnostic-position-realization': 7,
  };
  const uniformWriteGrammar = records.filter(
    (record) => record.destinationOwner.destinationBindingClass
      === 'uniform-bind-group-buffer',
  ).map((record) => {
    const creations = bufferCreations.filter(
      (candidate) => candidate.resourceId === record.bufferId,
    );
    const creation = creations.length === 1 ? creations[0] : null;
    const binding = record.destinationOwner.uniformBindingProofs[0] ?? null;
    const executionProofs = binding === null ? [] : uniformExecutionProofs(
      binding.bindGroupId,
      record.capturePhase,
      record.sequence,
    );
    const samePhaseDestinationWrites = records.filter(
      (candidate) => candidate.capturePhase === record.capturePhase
        && candidate.bufferId === record.bufferId,
    );
    const boundStart = binding?.bufferOffset ?? null;
    const boundByteLength = binding === null || creation === null
      ? null
      : binding.bufferSize ?? creation.size - binding.bufferOffset;
    const boundEnd = boundStart === null || boundByteLength === null
      ? null
      : boundStart + boundByteLength;
    const sourceRange = {
      start: record.selectedSourceByteOffset,
      byteLength: record.selectedSourceByteLength,
      end: record.selectedSourceByteOffset + record.selectedSourceByteLength,
    };
    const destinationRange = {
      start: record.bufferOffset,
      byteLength: record.selectedSourceByteLength,
      end: record.bufferOffset + record.selectedSourceByteLength,
    };
    const rangePositive = Number.isInteger(sourceRange.byteLength)
      && sourceRange.byteLength > 0
      && destinationRange.byteLength === sourceRange.byteLength;
    const rangeAligned = [
      sourceRange.start,
      sourceRange.byteLength,
      destinationRange.start,
      destinationRange.byteLength,
    ].every((value) => Number.isInteger(value) && value % 4 === 0);
    const sourceRangeWithinData = Number.isInteger(record.sourceByteLength)
      && sourceRange.start >= 0
      && sourceRange.end <= record.sourceByteLength;
    const destinationRangeWithinCreation = Number.isInteger(creation?.size)
      && destinationRange.start >= 0
      && destinationRange.end <= creation.size;
    const destinationRangeWithinBinding = Number.isInteger(boundStart)
      && Number.isInteger(boundByteLength)
      && boundByteLength > 0
      && destinationRange.start >= boundStart
      && destinationRange.end <= boundEnd;
    const sourceDestinationOffsetExact = Number.isInteger(boundStart)
      && sourceRange.start === destinationRange.start - boundStart;
    return {
      pass: creations.length === 1
        && creation?.usage === 72
        && binding !== null
        && binding.hasDynamicOffset === false
        && record.category !== 'renderer-initialization'
        && rangePositive
        && rangeAligned
        && sourceRangeWithinData
        && destinationRangeWithinCreation
        && destinationRangeWithinBinding
        && sourceDestinationOffsetExact
        && executionProofs.length === 1
        && executionProofs[0].pass,
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: record.category,
      bufferId: record.bufferId,
      creationMatchCount: creations.length,
      creation,
      binding,
      alignmentBytes: 4,
      sourceRange,
      destinationRange,
      boundRange: {
        start: boundStart,
        byteLength: boundByteLength,
        end: boundEnd,
      },
      rangePositive,
      rangeAligned,
      sourceRangeWithinData,
      destinationRangeWithinCreation,
      destinationRangeWithinBinding,
      sourceDestinationOffsetExact,
      boundByteLength,
      samePhaseDestinationWriteCount: samePhaseDestinationWrites.length,
      initializationOnly: false,
      executionProofs,
    };
  });
  const uniformWriteGroupMap = new Map();
  for (const grammar of uniformWriteGrammar) {
    const key = `${grammar.capturePhase}\u0000${grammar.bufferId}`;
    const group = uniformWriteGroupMap.get(key) ?? [];
    group.push(grammar);
    uniformWriteGroupMap.set(key, group);
  }
  const uniformWriteGroups = [...uniformWriteGroupMap].map(([groupId, groupRecords]) => {
    const [first] = groupRecords;
    const orderedDestinationRanges = groupRecords.map((record) => ({
      sequence: record.sequence,
      ...record.destinationRange,
    })).sort((left, right) => left.start - right.start || left.sequence - right.sequence);
    let furthestDestinationEnd = -1;
    let observedDestinationOverlapCount = 0;
    for (const range of orderedDestinationRanges) {
      if (range.start < furthestDestinationEnd) observedDestinationOverlapCount += 1;
      furthestDestinationEnd = Math.max(furthestDestinationEnd, range.end);
    }
    const writeSequencesStrictlyIncreasing = groupRecords.every(
      (record, index) => index === 0 || record.sequence > groupRecords[index - 1].sequence,
    );
    const sameCreationAndBinding = groupRecords.every((record) => (
      exactStructuredValue(record.creation, first.creation)
        && exactStructuredValue(record.binding, first.binding)
    ));
    const pass = groupRecords.length > 0
      && groupRecords.every((record) => record.pass)
      && new Set(groupRecords.map((record) => record.capturePhase)).size === 1
      && new Set(groupRecords.map((record) => record.bufferId)).size === 1
      && new Set(groupRecords.map((record) => record.category)).size === 1
      && sameCreationAndBinding
      && writeSequencesStrictlyIncreasing;
    return {
      pass,
      groupId,
      capturePhase: first.capturePhase,
      category: first.category,
      bufferId: first.bufferId,
      creation: first.creation,
      binding: first.binding,
      boundRange: first.boundRange,
      writeCount: groupRecords.length,
      writeSequences: groupRecords.map((record) => record.sequence),
      writeSequencesStrictlyIncreasing,
      sameCreationAndBinding,
      overlapPolicy: 'observed-not-prescribed',
      observedDestinationOverlapCount,
      orderedDestinationRanges,
      everyWriteExecuted: groupRecords.every(
        (record) => record.executionProofs.length === 1
          && record.executionProofs[0].pass,
      ),
      records: groupRecords,
    };
  });
  const uniformWriteGrammarExact = uniformWriteGrammar.every((record) => record.pass)
    && uniformWriteGroups.every((group) => group.pass)
    && uniformWriteGroups.reduce((count, group) => count + group.writeCount, 0)
      === uniformWriteGrammar.length;
  const rendererInitializationWriteCount = records.filter(
    (record) => record.category === 'renderer-initialization',
  ).length;
  const uniformOnlyCategories = new Set([
    'renderer-initialization',
    'production-render',
    'diagnostic-render',
  ]);
  const allowedLiveAttributeSemantics = (record) => {
    const match = record.capturePhase.match(/^phase0\/live\/(v99|v20)\/([AIF])$/);
    if (match === null) return [];
    const lane = match[2];
    return [
      'shared.bounds',
      'shared.objectBucket',
      'shared.bucketBase',
      'shared.bucketCapacity',
      'shared.cullOrder',
      'shared.visibleIds',
      'shared.overflow',
      `command.${lane}`,
    ];
  };
  const destinationPolicyExact = records.every((record) => {
    const owner = record.destinationOwner;
    const semantics = owner.semanticBindings.map((binding) => binding.semantic);
    const uniformExact = owner.destinationBindingClass === 'uniform-bind-group-buffer'
      && owner.uniformBindingProofs.length > 0
      && owner.uniformBindingProofs.every((proof) => proof.exact === true
        && proof.bufferId === record.bufferId
        && proof.bufferType === 'uniform'
        && proof.layoutMatchCount === 1
        && proof.coordinateMatchCount === 1);
    if (uniformOnlyCategories.has(record.category)) return uniformExact;
    if (record.category === 'live-compute') {
      const allowed = allowedLiveAttributeSemantics(record);
      return uniformExact || (owner.destinationBindingClass === 'known-attribute'
        && semantics.length > 0
        && semantics.every((semantic) => allowed.includes(semantic)));
    }
    if (record.category === 'scenario-matrix-realization') {
      return owner.destinationBindingClass === 'known-attribute'
        && semantics.includes('shared.matrix');
    }
    if (record.category === 'shared-resource-commitment') {
      return owner.destinationBindingClass === 'known-attribute'
        && semantics.includes('shared.visibleIds');
    }
    if (record.category === 'schedule-attribute-realization') {
      return owner.destinationBindingClass === 'known-attribute'
        && semantics.some((semantic) => semantic === 'geometry.A.bucketBase'
          || /^command\.[AIF]$/.test(semantic));
    }
    if (record.category === 'schedule-diagnostic-position-realization') {
      return owner.destinationBindingClass === 'known-attribute'
        && semantics.includes('diagnostic.address.featurePosition');
    }
    return false;
  });
  const expectedLiveAttributeWrites = [
    ['v99', 'I', []],
    ['v99', 'F', ['shared.visibleIds', 'shared.overflow']],
    ['v99', 'A', ['shared.visibleIds', 'shared.overflow']],
    ['v20', 'A', [
      'shared.bounds', 'shared.objectBucket', 'shared.bucketBase',
      'shared.bucketCapacity', 'shared.cullOrder', 'shared.visibleIds',
      'shared.overflow', 'command.A',
    ]],
    ['v20', 'F', ['shared.visibleIds', 'shared.overflow', 'command.F']],
    ['v20', 'I', ['shared.visibleIds', 'shared.overflow', 'command.I']],
  ];
  const liveAttributeWriteGrammar = expectedLiveAttributeWrites.map(([
    scenarioId, lane, expectedSemantics,
  ]) => {
    const capturePhase = `phase0/live/${scenarioId}/${lane}`;
    const phaseRecords = records.filter((record) => record.capturePhase === capturePhase);
    const attributeRecords = phaseRecords.filter(
      (record) => record.destinationOwner.destinationBindingClass === 'known-attribute',
    );
    const observedSemantics = attributeRecords.flatMap(
      (record) => record.destinationOwner.semanticBindings.map((binding) => binding.semantic),
    );
    return {
      pass: exactNumberArray(
        [...observedSemantics].sort(),
        [...expectedSemantics].sort(),
      ),
      scenarioId,
      lane,
      capturePhase,
      expectedSemantics,
      observedSemantics,
      attributeWriteCount: attributeRecords.length,
      uniformWriteCount: phaseRecords.length - attributeRecords.length,
      attributeRecords,
    };
  });
  const liveAttributeWriteGrammarExact = liveAttributeWriteGrammar.every(
    (record) => record.pass,
  );
  const genericUniformWriteCount = records.filter(
    (record) => uniformOnlyCategories.has(record.category),
  ).length;
  const specializedBindingsExact = specializedSources.every((source) => source.rawExact
    && source.records.every((record) => record.sourceSha256 === undefined
      || record.sourceSha256 === record.selectedSourceSha256))
    && exactNumberArray(specializedLedgerSequences, specializedSourceSequences)
    && exactStructuredValue(specializedCounts, expectedSpecializedCounts);
  const pass = rawRecords.length > 0
    && records.length === rawRecords.length
    && new Set(records.map((record) => record.sequence)).size === records.length
    && records.every((record) => Number.isInteger(record.sequence)
      && typeof record.capturePhase === 'string'
      && record.capturePhase.length > 0
      && typeof record.bufferId === 'string'
      && record.destinationOwner.exact === true
      && Number.isInteger(record.sourceElementByteSize)
      && record.sourceElementByteSize > 0
      && Number.isInteger(record.sourceElementCount)
      && record.selectedSourceByteOffset
        === record.selectedSourceElementOffset * record.sourceElementByteSize
      && record.selectedSourceByteLength
        === record.selectedSourceElementLength * record.sourceElementByteSize
      && /^[0-9a-f]{64}$/.test(record.selectedSourceSha256 ?? '')
      && record.category !== null)
    && partitions.reduce((count, partition) => count + partition.count, 0)
      === records.length
    && partitions.find(
      (partition) => partition.category === 'merged-geometry-realization',
    )?.count === 0
    && rendererInitializationWriteCount === 0
    && unclassified.length === 0
    && specializedBindingsExact
    && destinationPolicyExact
    && liveAttributeWriteGrammarExact
    && uniformWriteGrammarExact;
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-queue-write-buffer-ledger',
    pass,
    expectedSpecializedCounts,
    specializedCounts,
    specializedBindingsExact,
    destinationPolicyExact,
    liveAttributeWriteGrammarExact,
    liveAttributeWriteGrammar,
    mergedGeometryWriteCount: partitions.find(
      (partition) => partition.category === 'merged-geometry-realization',
    )?.count ?? null,
    rendererInitializationWriteCount,
    genericUniformWriteCount,
    uniformWriteGrammarExact,
    uniformWriteGrammar,
    uniformWriteGroupCount: uniformWriteGroups.length,
    uniformWriteGroupsExact: uniformWriteGroups.every((group) => group.pass),
    uniformWriteGroups,
    callCount: records.length,
    categoryOrder: [...QUEUE_WRITE_BUFFER_CATEGORY_ORDER],
    partitions,
    unclassifiedCount: unclassified.length,
    unclassified,
    specializedSources,
    records,
  };
}

function createBufferMapLifecycleEvidence(runtime, stagingTransferLedger) {
  const { evidence } = runtime.instrumentation;
  const buffers = evidence.resources.filter((resource) => resource.method === 'createBuffer');
  const useEvents = [];
  for (const write of evidence.queueWriteBuffers) {
    useEvents.push({ sequence: write.sequence, kind: 'queue.writeBuffer',
      bufferId: write.bufferId });
  }
  for (const resource of evidence.resources.filter(
    (record) => record.method === 'createBindGroup',
  )) {
    for (const entry of resource.entries ?? []) {
      if (entry.bufferId !== null) {
        useEvents.push({ sequence: resource.sequence, kind: 'createBindGroup-entry',
          bufferId: entry.bufferId });
      }
    }
  }
  for (const [traceKind, traces] of [
    ['render-bundle', evidence.renderBundleEncoders],
    ['render-pass', evidence.renderPassEncoders],
    ['compute-pass', evidence.computePassEncoders],
  ]) {
    for (const trace of traces) {
      for (const event of trace.events) {
        if (typeof event.bufferId === 'string') {
          useEvents.push({ sequence: event.sequence, kind: `${traceKind}/${event.method}`,
            bufferId: event.bufferId });
        }
      }
    }
  }
  for (const transfer of evidence.commandEncoderTransfers) {
    const ids = [
      transfer.sourceBufferId,
      transfer.destinationBufferId,
      transfer.source?.bufferId,
      transfer.destination?.bufferId,
    ].filter((value) => typeof value === 'string');
    for (const bufferId of new Set(ids)) {
      useEvents.push({ sequence: transfer.sequence, kind: transfer.method, bufferId });
    }
  }
  const stagingById = new Map(stagingTransferLedger.map(
    (record) => [record.stagingBufferId, record],
  ));
  const records = buffers.map((buffer) => {
    const mapEvents = evidence.bufferMapEvents.filter(
      (event) => event.resourceId === buffer.resourceId,
    );
    const firstUse = useEvents.filter((event) => event.bufferId === buffer.resourceId)
      .sort((left, right) => left.sequence - right.sequence)[0] ?? null;
    const staging = stagingById.get(buffer.resourceId) ?? null;
    const persistent = buffer.resourceClass === 'persistent-or-upload-buffer';
    const persistentMappedGrammar = persistent && buffer.mappedAtCreation === true
      && mapEvents.length === 2
      && mapEvents[0].method === 'getMappedRange'
      && mapEvents[0].offset === 0
      && (mapEvents[0].size === null || mapEvents[0].size === buffer.size)
      && mapEvents[1].method === 'unmap'
      && mapEvents.every((event) => event.capturePhase === buffer.capturePhase)
      && buffer.sequence < mapEvents[0].sequence
      && mapEvents[0].sequence < mapEvents[1].sequence
      && (firstUse === null || mapEvents[1].sequence < firstUse.sequence);
    const persistentUnmappedGrammar = persistent && buffer.mappedAtCreation === false
      && mapEvents.length === 0;
    const stagingGrammar = buffer.resourceClass === 'readback-staging'
      && staging?.pass === true
      && buffer.mappedAtCreation === false
      && mapEvents.length === 2
      && mapEvents[0].method === 'mapAsync'
      && mapEvents[1].method === 'getMappedRange'
      && mapEvents[0].sequence === staging.mapEvents[0].sequence
      && mapEvents[1].sequence === staging.mapEvents[1].sequence
      && staging.transfer.sequence < mapEvents[0].sequence
      && (firstUse === null || firstUse.sequence === staging.transfer.sequence);
    return {
      pass: persistent
        ? (Number(buffer.usage) & 0x0003) === 0
          && (persistentMappedGrammar || persistentUnmappedGrammar)
        : stagingGrammar,
      resourceId: buffer.resourceId,
      resourceClass: buffer.resourceClass,
      capturePhase: buffer.capturePhase,
      createSequence: buffer.sequence,
      size: buffer.size,
      usage: buffer.usage,
      mappedAtCreation: buffer.mappedAtCreation,
      mapEventCount: mapEvents.length,
      mapEvents,
      firstUse,
      persistentMappedGrammar,
      persistentUnmappedGrammar,
      stagingGrammar,
    };
  });
  const representedMapSequences = records.flatMap(
    (record) => record.mapEvents.map((event) => event.sequence),
  ).sort((left, right) => left - right);
  const rawMapSequences = evidence.bufferMapEvents.map(
    (event) => event.sequence,
  ).sort((left, right) => left - right);
  const unmatchedEvents = evidence.bufferMapEvents.filter((event) => (
    !buffers.some((buffer) => buffer.resourceId === event.resourceId)
  ));
  const persistentRecords = records.filter(
    (record) => record.resourceClass === 'persistent-or-upload-buffer',
  );
  const stagingRecords = records.filter(
    (record) => record.resourceClass === 'readback-staging',
  );
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-buffer-map-lifecycle',
    pass: records.length === buffers.length
      && buffers.length > EXPECTED_READBACK_STAGING_COUNT
      && new Set(buffers.map((buffer) => buffer.resourceId)).size === buffers.length
      && records.every((record) => record.pass)
      && stagingRecords.length === EXPECTED_READBACK_STAGING_COUNT
      && persistentRecords.length > 0
      && unmatchedEvents.length === 0
      && new Set(rawMapSequences).size === rawMapSequences.length
      && exactNumberArray(representedMapSequences, rawMapSequences),
    bufferCount: buffers.length,
    persistentBufferCount: persistentRecords.length,
    persistentMappedAtCreationCount: persistentRecords.filter(
      (record) => record.mappedAtCreation,
    ).length,
    stagingBufferCount: stagingRecords.length,
    rawMapEventCount: evidence.bufferMapEvents.length,
    representedMapEventCount: representedMapSequences.length,
    unmatchedEventCount: unmatchedEvents.length,
    unmatchedEvents,
    records,
  };
}

function createGpuByteWitnessEvidence(runtime) {
  const witnesses = Object.fromEntries(
    [...runtime.gpuByteWitnesses].sort(([left], [right]) => left.localeCompare(right)),
  );
  const references = runtime.gpuByteWitnessReferences.map((record) => ({ ...record }));
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-gpu-byte-witnesses',
    pass: references.length > 0
      && references.every((reference) => {
        const witness = witnesses[reference.witnessId];
        return witness?.sha256 === reference.sha256
          && witness.byteLength === reference.byteLength
          && witness.encoding === 'base64-exact-bytes';
      }),
    witnessCount: Object.keys(witnesses).length,
    referenceCount: references.length,
    witnesses,
    references,
  };
}

async function realizeScenarioMatrixAttribute(runtime, scenarioId) {
  const { renderer, strategy, instrumentation } = runtime;
  const manager = renderer._attributes;
  const attribute = strategy.sharedResources.attributes.matrix;
  requireCondition(manager?.constructor?.name === 'Attributes'
    && typeof manager.update === 'function'
    && typeof manager.get === 'function'
    && attribute?.array instanceof Float32Array,
  'Scenario matrix storage attribute cannot be explicitly realized.', {
      scenarioId,
      constructorName: manager?.constructor?.name ?? null,
      arrayType: attribute?.array?.constructor?.name ?? null,
    });
  const capturePhase = `phase0/scenario-load/${scenarioId}/matrix-realization`;
  const start = instrumentation.mark('phase0-scenario-matrix-realization-start', {
    scenarioId,
    attributeId: attribute.id,
    attributeVersion: attribute.version,
  });
  instrumentation.setCapturePhase(capturePhase);
  let beforeDataVersion;
  let afterDataVersion;
  let gpuBuffer;
  let updateComplete;
  try {
    beforeDataVersion = manager.get(attribute).version ?? null;
    manager.update(attribute, THREE_ATTRIBUTE_TYPE_STORAGE);
    gpuBuffer = renderer.backend.get(attribute)?.buffer ?? null;
    afterDataVersion = manager.get(attribute).version ?? null;
    updateComplete = instrumentation.mark('phase0-scenario-matrix-update-complete', {
      scenarioId,
      attributeId: attribute.id,
      attributeVersion: attribute.version,
      beforeDataVersion,
      afterDataVersion,
      gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
    });
    await renderer.backend.device.queue.onSubmittedWorkDone();
  } finally {
    instrumentation.setCapturePhase(null);
  }
  const queueComplete = instrumentation.mark('phase0-scenario-matrix-queue-complete', {
    scenarioId,
    attributeId: attribute.id,
  });
  const rawQueueWriteBuffers = instrumentation.evidence.queueWriteBuffers.filter(
    (record) => record.capturePhase === capturePhase
      && record.sequence > start.sequence
      && record.sequence < updateComplete.sequence,
  );
  const cpuSha256 = await sha256Bytes(attribute.array);
  const queueWriteBuffers = rawQueueWriteBuffers.map((write) => ({
    ...write,
    semantic: 'matrix',
    sourceSha256: cpuSha256,
  }));
  const expectedWriteCount = scenarioId === 'v99' ? 0 : 1;
  const queueWriteExact = scenarioId === 'v99'
    ? beforeDataVersion === null && queueWriteBuffers.length === 0
    : beforeDataVersion === attribute.version - 1
      && queueWriteBuffers.length === 1
      && queueWriteBuffers[0].capturePhase === capturePhase
      && queueWriteBuffers[0].bufferId === instrumentation.identify(gpuBuffer, 'buffer')
      && queueWriteBuffers[0].bufferOffset === 0
      && queueWriteBuffers[0].sourceType === 'Float32Array'
      && queueWriteBuffers[0].sourceByteLength
        === IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount * 16 * Float32Array.BYTES_PER_ELEMENT
      && queueWriteBuffers[0].dataOffset === 0
      && queueWriteBuffers[0].size === null
      && queueWriteBuffers[0].sequence > start.sequence
      && queueWriteBuffers[0].sequence < updateComplete.sequence;
  const pass = afterDataVersion === attribute.version
    && queueWriteExact
    && typeof instrumentation.identify(gpuBuffer, 'buffer') === 'string'
    && updateComplete.sequence < queueComplete.sequence;
  const result = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-scenario-matrix-realization',
    pass,
    scenarioId,
    capturePhase,
    attributeId: attribute.id,
    attributeVersion: attribute.version,
    attributeType: THREE_ATTRIBUTE_TYPE_STORAGE,
    beforeDataVersion,
    afterDataVersion,
    gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
    gpuBufferSize: gpuBuffer?.size ?? null,
    cpuSha256,
    startSequence: start.sequence,
    updateCompleteSequence: updateComplete.sequence,
    queueCompleteSequence: queueComplete.sequence,
    expectedQueueWriteBufferCount: expectedWriteCount,
    queueWriteBufferCount: queueWriteBuffers.length,
    queueWriteBuffers,
    queueWriteExact,
  };
  requireCondition(result.pass,
    `${scenarioId} scenario matrix realization is incomplete.`, result);
  runtime.scenarioMatrixRealizations.push(result);
  return result;
}

async function captureSharedGpuResourceCommitments(runtime, scenarioId) {
  const { renderer, strategy, instrumentation } = runtime;
  const capturePhase = `phase0/resources/${scenarioId}/shared-commitment`;
  const semantics = [
    'matrix',
    'bounds',
    'objectBucket',
    'bucketBase',
    'bucketCapacity',
    'cullOrder',
    'visibleIds',
    'overflow',
  ];
  const records = [];
  const frozenVisibleIds = strategy.frozenVisibleIds;
  requireCondition(frozenVisibleIds instanceof Uint32Array,
    `${scenarioId} has no explicit frozen visible-ID resource oracle.`);
  const manager = renderer._attributes;
  requireCondition(manager?.constructor?.name === 'Attributes'
    && typeof manager.update === 'function'
    && typeof manager.get === 'function',
  'Pinned Three renderer attribute manager is unavailable for shared-resource freeze.', {
      scenarioId,
      constructorName: manager?.constructor?.name ?? null,
    });
  const start = instrumentation.mark('phase0-shared-gpu-commitment-start', {
    scenarioId,
    capturePhase,
  });
  const managerCalls = [];
  const queueWriteBufferStartIndex = instrumentation.evidence.queueWriteBuffers.length;
  instrumentation.setCapturePhase(capturePhase);
  try {
    for (const semantic of semantics) {
      const attribute = strategy.sharedResources.attributes[semantic];
      const beforeDataVersion = manager.get(attribute).version ?? null;
      const before = instrumentation.mark('phase0-shared-attribute-update-start', {
        scenarioId,
        semantic,
        attributeId: attribute.id,
        attributeVersion: attribute.version,
        attributeType: THREE_ATTRIBUTE_TYPE_STORAGE,
        beforeDataVersion,
      });
      manager.update(attribute, THREE_ATTRIBUTE_TYPE_STORAGE);
      const gpuBuffer = renderer.backend.get(attribute)?.buffer ?? null;
      const afterDataVersion = manager.get(attribute).version ?? null;
      const after = instrumentation.mark('phase0-shared-attribute-update-complete', {
        scenarioId,
        semantic,
        attributeId: attribute.id,
        attributeVersion: attribute.version,
        attributeType: THREE_ATTRIBUTE_TYPE_STORAGE,
        afterDataVersion,
        gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
      });
      managerCalls.push({
        semantic,
        manager: 'renderer._attributes',
        managerConstructor: manager.constructor.name,
        attributeId: attribute.id,
        attributeVersion: attribute.version,
        attributeType: THREE_ATTRIBUTE_TYPE_STORAGE,
        beforeDataVersion,
        afterDataVersion,
        gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
        gpuBufferSize: gpuBuffer?.size ?? null,
        startSequence: before.sequence,
        completeSequence: after.sequence,
      });
    }
    await renderer.backend.device.queue.onSubmittedWorkDone();
    const queueComplete = instrumentation.mark(
      'phase0-shared-attribute-updates-queue-complete',
      { scenarioId },
    );
    const queueWriteBuffers = instrumentation.evidence.queueWriteBuffers
      .slice(queueWriteBufferStartIndex)
      .filter((record) => record.sequence < queueComplete.sequence);
    const visibleManagerCall = managerCalls.find(
      (record) => record.semantic === 'visibleIds',
    );
    const visibleIdsUpload = queueWriteBuffers.length === 1
      ? queueWriteBuffers[0]
      : null;
    const managerVersionsExact = managerCalls.every((record) => (
      record.afterDataVersion === record.attributeVersion
        && (record.semantic === 'visibleIds'
          ? record.beforeDataVersion === record.attributeVersion - 1
          : record.beforeDataVersion === record.attributeVersion)
    ));
    const visibleIdsUploadExact = visibleIdsUpload !== null
      && visibleIdsUpload.capturePhase === capturePhase
      && visibleIdsUpload.bufferId === visibleManagerCall?.gpuBufferId
      && visibleIdsUpload.bufferOffset === 0
      && visibleIdsUpload.sourceType === 'Uint32Array'
      && visibleIdsUpload.sourceByteLength
        === IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount * Uint32Array.BYTES_PER_ELEMENT
      && visibleIdsUpload.dataOffset === 0
      && visibleIdsUpload.size === null
      && visibleIdsUpload.sequence > visibleManagerCall.startSequence
      && visibleIdsUpload.sequence < visibleManagerCall.completeSequence
      && visibleManagerCall.completeSequence < queueComplete.sequence;
    for (const semantic of semantics) {
      const attribute = strategy.sharedResources.attributes[semantic];
      requireCondition(attribute && ArrayBuffer.isView(attribute.array),
        `Shared GPU resource ${semantic} is unavailable.`);
      const oracle = semantic === 'overflow'
        ? Uint32Array.of(0)
        : semantic === 'visibleIds'
          ? frozenVisibleIds
          : attribute.array;
      const oracleKind = semantic === 'overflow'
        ? 'protocol-zero-overflow'
        : semantic === 'visibleIds'
          ? 'frozen-cpu-canonical-visible-ids'
          : 'deterministic-production-attribute-array';
      requireCondition(oracle.constructor === attribute.array.constructor
        && oracle.byteLength === attribute.array.byteLength,
      `${scenarioId}/${semantic} resource oracle has the wrong representation.`, {
          oracleType: oracle.constructor.name,
          attributeType: attribute.array.constructor.name,
          oracleByteLength: oracle.byteLength,
          attributeByteLength: attribute.array.byteLength,
        });
      const readbackStart = instrumentation.mark(
        'phase0-shared-gpu-readback-start',
        { scenarioId, semantic, attributeId: attribute.id },
      );
      const gpu = new attribute.array.constructor(await renderer.getArrayBufferAsync(attribute));
      const readbackComplete = instrumentation.mark(
        'phase0-shared-gpu-readback-complete',
        { scenarioId, semantic, attributeId: attribute.id },
      );
      const cpuSourceBytes = bytesOf(attribute.array);
      const oracleBytes = bytesOf(oracle);
      const gpuBytes = bytesOf(gpu);
      const cpuSourceSha256 = await sha256Bytes(cpuSourceBytes);
      const oracleSha256 = await sha256Bytes(oracleBytes);
      const gpuSha256 = await sha256Bytes(gpuBytes);
      const cpuSourceMatchesOracle = cpuSourceBytes.length === oracleBytes.length
        && cpuSourceBytes.every((value, index) => value === oracleBytes[index]);
      const exact = gpuBytes.length === oracleBytes.length
        && gpuBytes.every((value, index) => value === oracleBytes[index]);
      const gpuBuffer = renderer.backend.get(attribute)?.buffer ?? null;
      const gpuBufferId = instrumentation.identify(gpuBuffer, 'buffer');
      const rawWitnessId = await registerGpuByteWitness(runtime, {
        bytes: gpuBytes,
        semantic: `shared/${scenarioId}/${semantic}`,
        gpuBufferId,
        capturePhase,
      });
      records.push({
        semantic,
        attributeId: attribute.id,
        attributeVersion: attribute.version,
        gpuBufferId,
        arrayType: oracle.constructor.name,
        byteLength: oracle.byteLength,
        oracleKind,
        cpuSha256: oracleSha256,
        oracleSha256,
        cpuSourceSha256,
        cpuSourceMatchesOracle,
        gpuSha256,
        rawWitnessId,
        readbackStartSequence: readbackStart.sequence,
        readbackCompleteSequence: readbackComplete.sequence,
        exact,
      });
    }
    const complete = instrumentation.mark('phase0-shared-gpu-commitment-complete', {
      scenarioId,
      queueCompleteSequence: queueComplete.sequence,
    });
    const annotatedQueueWriteBuffers = queueWriteBuffers.map((write) => {
      const source = records.find((record) => record.gpuBufferId === write.bufferId) ?? null;
      return {
        ...write,
        semantic: source?.semantic ?? null,
        sourceSha256: source?.cpuSourceSha256 ?? null,
        sourceWitnessId: source?.rawWitnessId ?? null,
      };
    });
    const annotatedVisibleIdsUpload = annotatedQueueWriteBuffers.find(
      (write) => write.bufferId === visibleManagerCall?.gpuBufferId,
    ) ?? null;
    const result = {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-shared-gpu-resource-commitments',
      pass: managerCalls.length === semantics.length
        && exactNumberArray(managerCalls.map((record) => record.semantic), semantics)
        && managerVersionsExact
        && managerCalls.every((record) => record.manager === 'renderer._attributes'
          && record.managerConstructor === 'Attributes'
          && record.attributeType === THREE_ATTRIBUTE_TYPE_STORAGE
          && typeof record.gpuBufferId === 'string'
          && record.startSequence < record.completeSequence
          && record.completeSequence < queueComplete.sequence)
        && records.length === semantics.length
        && records.every((record) => record.exact
          && record.cpuSha256 === record.gpuSha256
          && typeof record.rawWitnessId === 'string'
          && typeof record.gpuBufferId === 'string'
          && record.readbackStartSequence > queueComplete.sequence
          && record.readbackCompleteSequence > record.readbackStartSequence)
        && visibleIdsUploadExact
        && records.find((record) => record.semantic === 'visibleIds')?.gpuBufferId
          === managerCalls.find((record) => record.semantic === 'visibleIds')?.gpuBufferId,
      scenarioId,
      capturePhase,
      startSequence: start.sequence,
      queueCompleteSequence: queueComplete.sequence,
      completeSequence: complete.sequence,
      semantics,
      managerCalls,
      managerVersionsExact,
      queueWriteBufferCount: annotatedQueueWriteBuffers.length,
      queueWriteBuffers: annotatedQueueWriteBuffers,
      visibleIdsUpload: annotatedVisibleIdsUpload,
      visibleIdsUploadExact,
      records,
    };
    requireCondition(result.pass,
      `${scenarioId} shared GPU resource commitments failed.`, result);
    runtime.activeSharedGpuCommitment = result;
    return result;
  } finally {
    instrumentation.setCapturePhase(null);
  }
}

async function captureImmediateAifGeometryEvidence(runtime) {
  const { renderer, strategy, instrumentation, sourceGeometries, scenarios } = runtime;
  const capturePhase = 'phase0/resources/global/geometry-postflight';
  const sourceManifest = await fingerprintImmediateAifPhase0GeometryFixtures(
    sourceGeometries,
    IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier,
  );
  const scenarioManifests = {};
  for (const scenarioId of ['v99', 'v20']) {
    scenarioManifests[scenarioId] = await fingerprintFixedSubsetScenario(
      scenarios[scenarioId],
      IMMEDIATE_AIF_PHASE0_WORKLOAD.seed,
    );
  }

  const laneBindings = {};
  const allBoundVertexBufferIds = new Set();
  const allBoundIndexBufferIds = new Set();
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const context = strategy.getLaneValidationResources(lane);
    const found = findProductionBundle(
      renderer,
      instrumentation,
      runtime.productionTarget,
      runtime.camera,
      context,
    );
    const vertexEvents = found.trace.events
      .filter((event) => event.method === 'setVertexBuffer')
      .map((event) => structuredClone(event));
    const indexEvents = found.trace.events
      .filter((event) => event.method === 'setIndexBuffer')
      .map((event) => structuredClone(event));
    for (const event of vertexEvents) allBoundVertexBufferIds.add(event.bufferId);
    for (const event of indexEvents) allBoundIndexBufferIds.add(event.bufferId);
    laneBindings[lane] = {
      lane,
      geometryUuid: context.geometry.uuid,
      bundleGpuId: found.bundleGpuId,
      vertexEvents,
      indexEvents,
    };
  }

  const mergedGeometry = strategy.laneStates.A.geometry;
  const sources = [
    ...Object.keys(mergedGeometry.attributes).sort().map((name) => ({
      semantic: name,
      attribute: mergedGeometry.getAttribute(name),
      resourceKind: 'vertex-attribute',
    })),
    {
      semantic: 'index',
      attribute: mergedGeometry.index,
      resourceKind: 'index',
    },
  ];
  const gpuRecords = [];
  instrumentation.setCapturePhase(capturePhase);
  try {
    for (const { semantic, attribute, resourceKind } of sources) {
      requireCondition(attribute?.array && ArrayBuffer.isView(attribute.array),
        `Merged production geometry ${semantic} is not a typed-array attribute.`);
      const gpuBuffer = renderer.backend.get(attribute)?.buffer ?? null;
      const gpuBufferId = instrumentation.identify(gpuBuffer, 'buffer');
      const bound = resourceKind === 'index'
        ? allBoundIndexBufferIds.has(gpuBufferId)
        : allBoundVertexBufferIds.has(gpuBufferId);
      let gpuSha256 = null;
      let exact = false;
      let gpuByteLength = null;
      let rawWitnessId = null;
      if (gpuBuffer !== null) {
        const gpuBytes = bytesOf(await renderer.getArrayBufferAsync(
          attribute,
          null,
          0,
          attribute.array.byteLength,
        ));
        const cpuBytes = bytesOf(attribute.array);
        gpuByteLength = gpuBytes.byteLength;
        gpuSha256 = await sha256Bytes(gpuBytes);
        exact = gpuBytes.byteLength === cpuBytes.byteLength
          && gpuBytes.every((value, index) => value === cpuBytes[index]);
        rawWitnessId = await registerGpuByteWitness(runtime, {
          bytes: gpuBytes,
          semantic: `merged-geometry/${semantic}`,
          gpuBufferId,
          capturePhase,
        });
      }
      gpuRecords.push({
        semantic,
        resourceKind,
        attributeId: attribute.id,
        arrayType: attribute.array.constructor.name,
        itemSize: attribute.itemSize,
        count: attribute.count,
        normalized: attribute.normalized === true,
        usage: attribute.usage,
        gpuType: attribute.gpuType ?? null,
        byteLength: attribute.array.byteLength,
        cpuSha256: await sha256Bytes(attribute.array),
        gpuResident: gpuBuffer !== null,
        gpuBufferId,
        gpuBufferSize: gpuBuffer?.size ?? null,
        gpuByteLength,
        gpuSha256,
        rawWitnessId,
        exact,
        boundByProductionBundle: bound,
      });
    }
  } finally {
    instrumentation.setCapturePhase(null);
  }

  let vertexStart = 0;
  let firstIndex = 0;
  const bucketSpans = sourceManifest.geometries.map((geometry) => {
    const record = {
      bucket: geometry.bucket,
      family: geometry.family,
      vertexStart,
      vertexCount: geometry.attributes.position.count,
      firstIndex,
      indexCount: geometry.index.count,
      baseVertex: 0,
      sourceSha256: geometry.sha256,
    };
    vertexStart += record.vertexCount;
    firstIndex += record.indexCount;
    return record;
  });
  const knownGpuBufferIds = new Set(
    gpuRecords.filter((record) => record.gpuResident).map((record) => record.gpuBufferId),
  );
  const boundResourceCoverageExact = [...allBoundVertexBufferIds, ...allBoundIndexBufferIds]
    .every((bufferId) => knownGpuBufferIds.has(bufferId));
  const requiredGpuRecords = gpuRecords.filter((record) => record.boundByProductionBundle);
  const geometryRealization = runtime.mergedGeometryRealization;
  const realizationBindingExact = geometryRealization?.pass === true
    && geometryRealization.records.length === gpuRecords.length
    && gpuRecords.every((record) => {
      const realized = geometryRealization.records.find(
        (candidate) => candidate.semantic === record.semantic,
      );
      return realized?.attributeId === record.attributeId
        && realized.gpuBufferId === record.gpuBufferId
        && realized.cpuSha256 === record.cpuSha256;
    });
  const pass = strategy.exactWorkload?.pass === true
    && sourceManifest.schemaVersion === 2
    && sourceManifest.bucketCount === IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount
    && Object.values(scenarioManifests).every((manifest) => manifest.schemaVersion === 1)
    && bucketSpans.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount
    && gpuRecords.length > 0
    && gpuRecords.every((record) => record.gpuResident
      && record.exact
      && record.gpuSha256 === record.cpuSha256
      && typeof record.rawWitnessId === 'string')
    && requiredGpuRecords.length > 0
    && requiredGpuRecords.every((record) => record.exact
      && record.gpuSha256 === record.cpuSha256
      && typeof record.rawWitnessId === 'string')
    && gpuRecords.find((record) => record.semantic === 'index')?.boundByProductionBundle === true
    && boundResourceCoverageExact
    && realizationBindingExact
    && Object.values(laneBindings).every((binding) => (
      binding.vertexEvents.length > 0 && binding.indexEvents.length === 1
    ));
  const result = {
    schemaVersion: 2,
    kind: 'immediate-aif-phase0-geometry-resource-evidence',
    pass,
    capturePhase,
    tier: IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier,
    exactAcceptance: strategy.exactWorkload,
    sourceManifest,
    scenarioManifests,
    bucketSpans,
    mergedProduction: {
      geometryUuid: mergedGeometry.uuid,
      vertexCount: mergedGeometry.getAttribute('position').count,
      indexCount: mergedGeometry.index.count,
      gpuRecords,
    },
    laneBindings,
    boundResourceCoverageExact,
    realizationBindingExact,
    mergedGeometryRealization: geometryRealization,
  };
  requireCondition(result.pass,
    'Exact source/merged production geometry evidence failed.', result);
  return result;
}

function exactNumberArray(left, right) {
  return left?.length === right.length
    && right.every((value, index) => Object.is(left[index], value));
}

function textureExtentMatches(size, width, height) {
  if (Array.isArray(size)) return size[0] === width && size[1] === height;
  return size?.width === width && size?.height === height;
}

function exactJsonValue(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function exactStructuredValue(left, right) {
  if (typeof left !== 'object' || left === null
    || typeof right !== 'object' || right === null) {
    return Object.is(left, right);
  }
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left)
      && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => exactStructuredValue(value, right[index]));
  }
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  return exactNumberArray(leftKeys, rightKeys)
    && leftKeys.every((key) => exactStructuredValue(left[key], right[key]));
}

function logicalTextureState(texture) {
  if (texture == null) return null;
  return {
    format: texture.format,
    type: texture.type,
    colorSpace: texture.colorSpace,
    minFilter: texture.minFilter,
    magFilter: texture.magFilter,
    generateMipmaps: texture.generateMipmaps,
    flipY: texture.flipY,
    internalFormat: texture.internalFormat ?? null,
    anisotropy: texture.anisotropy,
    wrapS: texture.wrapS,
    wrapT: texture.wrapT,
    wrapR: texture.wrapR ?? null,
  };
}

function gpuTextureDescriptorState(record) {
  if (record == null) return null;
  return {
    label: record.label ?? null,
    size: Array.isArray(record.size)
      ? {
        width: Number(record.size[0]),
        height: Number(record.size[1] ?? 1),
        depthOrArrayLayers: Number(record.size[2] ?? 1),
      }
      : {
        width: Number(record.size?.width),
        height: Number(record.size?.height ?? 1),
        depthOrArrayLayers: Number(record.size?.depthOrArrayLayers ?? 1),
      },
    mipLevelCount: Number(record.mipLevelCount ?? 1),
    sampleCount: Number(record.sampleCount ?? 1),
    dimension: record.dimension ?? '2d',
    format: record.format ?? null,
    usage: Number(record.usage),
    viewFormats: Array.from(record.viewFormats ?? [], String),
    textureBindingViewDimension: record.textureBindingViewDimension ?? null,
  };
}

function captureTargetResourceEvidence(runtime, name, target, expected, purpose) {
  const { renderer, instrumentation } = runtime;
  const colorTexture = renderer.backend.get(target.texture)?.texture ?? null;
  const colorTextureId = instrumentation.identify(colorTexture, 'texture');
  const matchingPasses = instrumentation.evidence.renderPassEncoders.filter((trace) => (
    trace.colorAttachments?.some((attachment) => attachment.textureId === colorTextureId)
  ));
  const depthTextureIds = [...new Set(matchingPasses
    .map((trace) => trace.depthStencilAttachment?.textureId)
    .filter((value) => value !== null && value !== undefined))];
  const colorTextureCreations = instrumentation.evidence.resources.filter((record) => (
    record.method === 'createTexture' && record.resourceId === colorTextureId
  ));
  const depthTextureCreations = depthTextureIds.flatMap((textureId) => (
    instrumentation.evidence.resources.filter((record) => (
      record.method === 'createTexture' && record.resourceId === textureId
    ))
  ));
  const executedBundleIds = [...new Set(matchingPasses.flatMap((trace) => (
    trace.events
      .filter((event) => event.method === 'executeBundles')
      .flatMap((event) => event.bundleIds ?? [])
  )))];
  const executedPipelineIds = [...new Set(instrumentation.evidence.renderBundleEncoders
    .filter((trace) => executedBundleIds.includes(trace.bundleId))
    .flatMap((trace) => trace.events
      .filter((event) => event.method === 'setPipeline')
      .map((event) => event.pipelineId)))];
  const colorCreation = colorTextureCreations[0] ?? null;
  const depthCreation = depthTextureCreations[0] ?? null;
  const outputWitnessIds = [...runtime.outputWitnesses.values()]
    .filter((witness) => purpose.outputChannels.includes(witness.channel))
    .map((witness) => witness.witnessId)
    .sort();
  const logical = {
    width: target.width,
    height: target.height,
    depth: target.depth,
    count: target.textures.length,
    samples: target.samples,
    depthBuffer: target.depthBuffer,
    stencilBuffer: target.stencilBuffer,
    resolveDepthBuffer: target.resolveDepthBuffer,
    resolveStencilBuffer: target.resolveStencilBuffer,
    multiview: target.multiview,
    useArrayDepthTexture: target.useArrayDepthTexture,
    colorSpace: target.texture.colorSpace,
    colorTexture: logicalTextureState(target.texture),
    depthTexture: logicalTextureState(target.depthTexture),
  };
  const gpuColorDescriptor = gpuTextureDescriptorState(colorCreation);
  const gpuDepthDescriptor = gpuTextureDescriptorState(depthCreation);
  const pass = logical.width === expected.width
    && logical.height === expected.height
    && logical.depth === expected.depth
    && logical.count === expected.count
    && logical.samples === expected.samples
    && logical.depthBuffer === expected.depthBuffer
    && logical.stencilBuffer === expected.stencilBuffer
    && logical.resolveDepthBuffer === expected.resolveDepthBuffer
    && logical.resolveStencilBuffer === expected.resolveStencilBuffer
    && logical.multiview === expected.multiview
    && logical.useArrayDepthTexture === expected.useArrayDepthTexture
    && logical.colorSpace === expected.colorSpace
    && exactJsonValue(logical.colorTexture, expected.colorTexture)
    && exactJsonValue(logical.depthTexture, expected.depthTexture)
    && colorTextureId !== null
    && colorTextureCreations.length === 1
    && colorCreation.format === expected.colorFormat
    && textureExtentMatches(colorCreation.size, expected.width, expected.height)
    && exactJsonValue(gpuColorDescriptor, expected.gpuColorDescriptor)
    && matchingPasses.length > 0
    && (expected.depthBuffer
      ? depthTextureIds.length === 1
        && depthTextureCreations.length === 1
        && depthCreation.format === expected.depthFormat
        && textureExtentMatches(depthCreation.size, expected.width, expected.height)
        && exactJsonValue(gpuDepthDescriptor, expected.gpuDepthDescriptor)
      : depthTextureIds.length === 0
        && depthTextureCreations.length === 0
        && gpuDepthDescriptor === null);
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-target-resource-evidence',
    pass,
    name,
    purpose,
    expected: structuredClone(expected),
    logical,
    colorTextureId,
    depthTextureIds,
    colorTextureCreation: colorCreation,
    depthTextureCreation: depthCreation,
    gpuColorDescriptor,
    gpuDepthDescriptor,
    renderPassEncoderIds: matchingPasses.map((trace) => trace.encoderId),
    executedBundleIds,
    executedPipelineIds,
    outputWitnessIds,
  };
}

function productionMaterialProjection(material) {
  const mapNames = [
    'map',
    'alphaMap',
    'aoMap',
    'bumpMap',
    'normalMap',
    'displacementMap',
    'roughnessMap',
    'metalnessMap',
    'emissiveMap',
    'envMap',
    'lightMap',
  ];
  return {
    type: material.type,
    color: material.color.getHex(),
    roughness: material.roughness,
    metalness: material.metalness,
    emissive: material.emissive.getHex(),
    emissiveIntensity: material.emissiveIntensity,
    opacity: material.opacity,
    transparent: material.transparent,
    side: material.side,
    depthTest: material.depthTest,
    depthWrite: material.depthWrite,
    blending: material.blending,
    blendSrc: material.blendSrc,
    blendDst: material.blendDst,
    blendEquation: material.blendEquation,
    blendSrcAlpha: material.blendSrcAlpha,
    blendDstAlpha: material.blendDstAlpha,
    blendEquationAlpha: material.blendEquationAlpha,
    blendColor: material.blendColor.toArray(),
    blendAlpha: material.blendAlpha,
    premultipliedAlpha: material.premultipliedAlpha,
    colorWrite: material.colorWrite,
    depthFunc: material.depthFunc,
    vertexColors: material.vertexColors,
    toneMapped: material.toneMapped,
    wireframe: material.wireframe,
    alphaTest: material.alphaTest,
    alphaHash: material.alphaHash,
    alphaToCoverage: material.alphaToCoverage,
    forceSinglePass: material.forceSinglePass,
    polygonOffset: material.polygonOffset,
    polygonOffsetFactor: material.polygonOffsetFactor,
    polygonOffsetUnits: material.polygonOffsetUnits,
    stencilWrite: material.stencilWrite,
    stencilWriteMask: material.stencilWriteMask,
    stencilFunc: material.stencilFunc,
    stencilRef: material.stencilRef,
    stencilFuncMask: material.stencilFuncMask,
    stencilFail: material.stencilFail,
    stencilZFail: material.stencilZFail,
    stencilZPass: material.stencilZPass,
    fog: material.fog,
    lights: material.lights,
    flatShading: material.flatShading,
    dithering: material.dithering,
    precision: material.precision,
    bumpScale: material.bumpScale,
    normalScale: material.normalScale.toArray(),
    normalMapType: material.normalMapType,
    displacementScale: material.displacementScale,
    displacementBias: material.displacementBias,
    aoMapIntensity: material.aoMapIntensity,
    lightMapIntensity: material.lightMapIntensity,
    envMapIntensity: material.envMapIntensity,
    envMapRotation: material.envMapRotation.toArray(),
    clipIntersection: material.clipIntersection,
    clipShadows: material.clipShadows,
    shadowSide: material.shadowSide,
    clippingPlanes: material.clippingPlanes,
    wireframeLinewidth: material.wireframeLinewidth,
    wireframeLinecap: material.wireframeLinecap,
    wireframeLinejoin: material.wireframeLinejoin,
    maps: Object.fromEntries(mapNames.map((name) => [
      name,
      material[name] === null ? null : material[name]?.uuid ?? '__non-null-texture__',
    ])),
    visible: material.visible,
  };
}

function transformProjection(object) {
  return {
    type: object.type,
    name: object.name,
    position: object.position.toArray(),
    worldPosition: new Vector3().setFromMatrixPosition(object.matrixWorld).toArray(),
    quaternion: object.quaternion.toArray(),
    scale: object.scale.toArray(),
    matrix: object.matrix.toArray(),
    matrixWorld: object.matrixWorld.toArray(),
    matrixAutoUpdate: object.matrixAutoUpdate,
    matrixWorldAutoUpdate: object.matrixWorldAutoUpdate,
    visible: object.visible,
    layersMask: object.layers.mask,
    castShadow: object.castShadow,
    receiveShadow: object.receiveShadow,
    frustumCulled: object.frustumCulled,
    renderOrder: object.renderOrder,
  };
}

function productionLightProjection(productionScene) {
  productionScene.updateMatrixWorld(true);
  const hemisphere = productionScene.children.find((child) => child.isHemisphereLight);
  const directional = productionScene.children.find((child) => child.isDirectionalLight);
  if (hemisphere == null || directional == null || directional.target == null) {
    return { hemisphere: null, directional: null };
  }
  const hemisphereTransform = transformProjection(hemisphere);
  const directionalTransform = transformProjection(directional);
  const target = {
    ...transformProjection(directional.target),
    addedToProductionScene: directional.target.parent === productionScene,
  };
  const directionalWorldPosition = new Vector3().fromArray(
    directionalTransform.worldPosition,
  );
  const targetWorldPosition = new Vector3().fromArray(target.worldPosition);
  return {
    hemisphere: {
      ...hemisphereTransform,
      skyColor: hemisphere.color.getHex(),
      groundColor: hemisphere.groundColor.getHex(),
      intensity: hemisphere.intensity,
      effectiveNormalizedDirection: new Vector3()
        .fromArray(hemisphereTransform.worldPosition)
        .normalize()
        .toArray(),
    },
    directional: {
      ...directionalTransform,
      color: directional.color.getHex(),
      intensity: directional.intensity,
      targetDirection: targetWorldPosition.clone()
        .sub(directionalWorldPosition)
        .normalize()
        .toArray(),
      effectiveNormalizedDirection: directionalWorldPosition.clone()
        .sub(targetWorldPosition)
        .normalize()
        .toArray(),
      target,
    },
  };
}

function commonCameraProjection(camera) {
  camera.updateMatrixWorld(true);
  return {
    uuid: camera.uuid,
    id: camera.id,
    type: camera.type,
    fov: camera.fov,
    aspect: camera.aspect,
    near: camera.near,
    far: camera.far,
    zoom: camera.zoom,
    position: camera.position.toArray(),
    quaternion: camera.quaternion.toArray(),
    scale: camera.scale.toArray(),
    up: camera.up.toArray(),
    coordinateSystem: camera.coordinateSystem,
    reversedDepth: camera.reversedDepth,
    matrix: camera.matrix.toArray(),
    matrixWorld: camera.matrixWorld.toArray(),
    matrixWorldInverse: camera.matrixWorldInverse.toArray(),
    projectionMatrix: camera.projectionMatrix.toArray(),
    projectionMatrixInverse: camera.projectionMatrixInverse.toArray(),
    matrixAutoUpdate: camera.matrixAutoUpdate,
    matrixWorldAutoUpdate: camera.matrixWorldAutoUpdate,
  };
}

function commonTargetProjection(target) {
  return {
    uuid: target.uuid,
    width: target.width,
    height: target.height,
    depth: target.depth,
    count: target.textures.length,
    samples: target.samples,
    depthBuffer: target.depthBuffer,
    stencilBuffer: target.stencilBuffer,
    resolveDepthBuffer: target.resolveDepthBuffer,
    resolveStencilBuffer: target.resolveStencilBuffer,
    multiview: target.multiview,
    useArrayDepthTexture: target.useArrayDepthTexture,
    texture: {
      uuid: target.texture.uuid,
      id: target.texture.id,
      version: target.texture.version,
      ...logicalTextureState(target.texture),
    },
    depthTexture: target.depthTexture == null ? null : {
      uuid: target.depthTexture.uuid,
      id: target.depthTexture.id,
      version: target.depthTexture.version,
      ...logicalTextureState(target.depthTexture),
    },
  };
}

function canonicalCommonRenderStateProjection(runtime) {
  const { renderer, camera, strategy, productionScene, instrumentation } = runtime;
  productionScene.updateMatrixWorld(true);
  const directSceneChildren = productionScene.children.map((child) => ({
    uuid: child.uuid,
    id: child.id,
    type: child.type,
    name: child.name,
    visible: child.visible,
    matrix: child.matrix.toArray(),
    matrixWorld: child.matrixWorld.toArray(),
    matrixAutoUpdate: child.matrixAutoUpdate,
    matrixWorldAutoUpdate: child.matrixWorldAutoUpdate,
  }));
  return {
    renderer: {
      rendererConstructor: renderer.constructor.name,
      backendConstructor: renderer.backend.constructor.name,
      backendId: instrumentation.identify(renderer.backend, 'backend'),
      viewport: {
        width: renderer.domElement.width,
        height: renderer.domElement.height,
        devicePixelRatio: renderer.getPixelRatio(),
      },
      coordinateSystem: renderer.coordinateSystem,
      autoClear: renderer.autoClear,
      autoClearColor: renderer.autoClearColor,
      autoClearDepth: renderer.autoClearDepth,
      autoClearStencil: renderer.autoClearStencil,
      clearColor: renderer.getClearColor(new Color()).getHex(),
      clearAlpha: renderer.getClearAlpha(),
      clearDepth: renderer.getClearDepth(),
      clearStencil: renderer.getClearStencil(),
      sortObjects: renderer.sortObjects,
      toneMapping: renderer.toneMapping,
      toneMappingExposure: renderer.toneMappingExposure,
      outputColorSpace: renderer.outputColorSpace,
      reversedDepthBuffer: renderer.reversedDepthBuffer,
      trackTimestamp: renderer.backend.trackTimestamp,
      backendDeviceId: instrumentation.identify(renderer.backend.device, 'device'),
      backendParameterDeviceId: instrumentation.identify(
        renderer.backend.parameters?.device,
        'device',
      ),
      backendDeviceMatchesParameter:
        renderer.backend.device === renderer.backend.parameters?.device,
      activeRenderTargetUuid: renderer.getRenderTarget()?.uuid ?? null,
      activeCubeFace: renderer.getActiveCubeFace(),
      activeMipmapLevel: renderer.getActiveMipmapLevel(),
    },
    camera: commonCameraProjection(camera),
    materials: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => {
      const material = strategy.laneStates[lane].material;
      return [lane, {
        uuid: material.uuid,
        id: material.id,
        name: material.name,
        version: material.version,
        addressMode: material.userData.storageTransformAddressMode,
        content: productionMaterialProjection(material),
      }];
    })),
    lights: productionLightProjection(productionScene),
    scene: {
      uuid: productionScene.uuid,
      id: productionScene.id,
      type: productionScene.type,
      name: productionScene.name,
      background: productionScene.background.getHex(),
      matrix: productionScene.matrix.toArray(),
      matrixWorld: productionScene.matrixWorld.toArray(),
      matrixAutoUpdate: productionScene.matrixAutoUpdate,
      matrixWorldAutoUpdate: productionScene.matrixWorldAutoUpdate,
      directChildren: directSceneChildren,
      laneRoots: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => {
        const root = strategy.laneStates[lane].root;
        return [lane, {
          uuid: root.uuid,
          id: root.id,
          type: root.type,
          name: root.name,
          version: root.version,
          visible: root.visible,
          matrix: root.matrix.toArray(),
          matrixWorld: root.matrixWorld.toArray(),
          matrixAutoUpdate: root.matrixAutoUpdate,
          matrixWorldAutoUpdate: root.matrixWorldAutoUpdate,
          children: root.children.map((child) => ({
            uuid: child.uuid,
            id: child.id,
            type: child.type,
            name: child.name,
            visible: child.visible,
            geometryUuid: child.geometry?.uuid ?? null,
            materialUuid: child.material?.uuid ?? null,
            matrix: child.matrix.toArray(),
            matrixWorld: child.matrixWorld.toArray(),
            matrixAutoUpdate: child.matrixAutoUpdate,
            matrixWorldAutoUpdate: child.matrixWorldAutoUpdate,
          })),
        }];
      })),
      activeLane: strategy.activeLane,
    },
    targets: {
      production: commonTargetProjection(runtime.productionTarget),
      address: commonTargetProjection(runtime.addressDiagnostics.target),
      objectId: commonTargetProjection(runtime.objectIdDiagnostics.target),
    },
  };
}

async function captureCommonRenderStateCommitment(runtime, label) {
  const rawProjection = canonicalCommonRenderStateProjection(runtime);
  const projection = normalizeImmediateAifCommonRenderStateProjection(rawProjection);
  const visibleLanes = IMMEDIATE_AIF_PHASE0_LANES.filter(
    (lane) => rawProjection.scene.laneRoots[lane].visible,
  );
  const laneSelection = {
    activeLane: rawProjection.scene.activeLane,
    visibleLanes,
    exact: visibleLanes.length === 1
      && visibleLanes[0] === rawProjection.scene.activeLane,
  };
  const rawJson = JSON.stringify(rawProjection);
  const canonicalJson = JSON.stringify(projection);
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-common-render-state-commitment',
    label,
    encoding: 'json-utf8-property-order-v1',
    byteLength: new TextEncoder().encode(canonicalJson).byteLength,
    sha256: await sha256Text(canonicalJson),
    rawByteLength: new TextEncoder().encode(rawJson).byteLength,
    rawSha256: await sha256Text(rawJson),
    laneSelection,
    rawProjection,
    projection,
  };
}

function exactLightProjection(actual, expected) {
  return exactStructuredValue(actual, expected)
    && exactNumberArray(actual?.hemisphere?.effectiveNormalizedDirection,
      expected.hemisphere.effectiveNormalizedDirection)
    && exactNumberArray(actual?.directional?.targetDirection,
      expected.directional.targetDirection)
    && exactNumberArray(actual?.directional?.effectiveNormalizedDirection,
      expected.directional.effectiveNormalizedDirection);
}

async function captureImmediateAifCommonResourceEvidence(runtime, shaderEvidence) {
  const { renderer, camera, strategy, productionScene } = runtime;
  const expected = IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION;
  camera.updateMatrixWorld(true);
  const declaredTarget = new Vector3().fromArray(expected.camera.target);
  const targetDirection = declaredTarget.clone().sub(camera.position).normalize();
  const cameraState = {
    type: camera.type,
    fov: camera.fov,
    aspect: camera.aspect,
    near: camera.near,
    far: camera.far,
    zoom: camera.zoom,
    position: camera.position.toArray(),
    target: declaredTarget.toArray(),
    targetDistance: camera.position.distanceTo(declaredTarget),
    targetDirection: targetDirection.toArray(),
    worldDirection: camera.getWorldDirection(new Vector3()).toArray(),
    quaternion: camera.quaternion.toArray(),
    scale: camera.scale.toArray(),
    up: camera.up.toArray(),
    coordinateSystem: camera.coordinateSystem,
    reversedDepth: camera.reversedDepth,
    matrixWorld: camera.matrixWorld.toArray(),
    matrixWorldInverse: camera.matrixWorldInverse.toArray(),
    projectionMatrix: camera.projectionMatrix.toArray(),
    projectionMatrixInverse: camera.projectionMatrixInverse.toArray(),
  };
  cameraState.sha256 = await sha256Text(JSON.stringify(cameraState));
  const cameraPass = cameraState.type === expected.camera.type
    && cameraState.fov === expected.camera.fov
    && cameraState.aspect === expected.camera.aspect
    && cameraState.near === expected.camera.near
    && cameraState.far === expected.camera.far
    && cameraState.zoom === expected.camera.zoom
    && cameraState.coordinateSystem === expected.camera.coordinateSystem
    && cameraState.reversedDepth === expected.camera.reversedDepth
    && exactNumberArray(cameraState.position, expected.camera.position)
    && exactNumberArray(cameraState.target, expected.camera.target)
    && cameraState.targetDistance === expected.camera.targetDistance
    && exactNumberArray(cameraState.targetDirection, expected.camera.targetDirection)
    && exactNumberArray(cameraState.worldDirection, expected.camera.worldDirection)
    && exactNumberArray(cameraState.quaternion, expected.camera.quaternion)
    && exactNumberArray(cameraState.scale, expected.camera.scale)
    && exactNumberArray(cameraState.up, expected.camera.up)
    && exactNumberArray(cameraState.matrixWorld, expected.camera.matrixWorld)
    && exactNumberArray(cameraState.matrixWorldInverse, expected.camera.matrixWorldInverse)
    && exactNumberArray(cameraState.projectionMatrix, expected.camera.projectionMatrix)
    && exactNumberArray(
      cameraState.projectionMatrixInverse,
      expected.camera.projectionMatrixInverse,
    );

  const materials = {};
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const material = strategy.laneStates[lane].material;
    const canonical = productionMaterialProjection(material);
    const state = {
      lane,
      uuid: material.uuid,
      id: material.id,
      type: material.type,
      name: material.name,
      version: material.version,
      color: material.color.toArray(),
      colorHex: material.color.getHex(),
      roughness: material.roughness,
      metalness: material.metalness,
      opacity: material.opacity,
      transparent: material.transparent,
      side: material.side,
      depthTest: material.depthTest,
      depthWrite: material.depthWrite,
      blending: material.blending,
      vertexColors: material.vertexColors,
      toneMapped: material.toneMapped,
      wireframe: material.wireframe,
      alphaTest: material.alphaTest,
      forceSinglePass: material.forceSinglePass,
      visible: material.visible,
      canonical,
      addressMode: material.userData.storageTransformAddressMode,
      productionPipelineId: shaderEvidence.lanes[lane].pipelineId,
    };
    materials[lane] = {
      ...state,
      sha256: await sha256Text(JSON.stringify(state)),
      pass: exactJsonValue(canonical, expected.material)
        && state.addressMode === ADDRESS_MODE_BY_LANE[lane]
        && typeof state.productionPipelineId === 'string',
    };
  }

  const lights = productionLightProjection(productionScene);
  const lightsPass = exactLightProjection(lights, expected.lights);

  const rendererClearColor = renderer.getClearColor(new Color());
  const backendParameters = renderer.backend.parameters ?? {};
  const rendererState = {
    viewport: {
      width: renderer.domElement.width,
      height: renderer.domElement.height,
      devicePixelRatio: renderer.getPixelRatio(),
    },
    antialias: backendParameters.antialias === true,
    samples: renderer._samples,
    powerPreference: backendParameters.powerPreference ?? null,
    forceWebGL: backendParameters.forceWebGL === true,
    alpha: renderer.alpha,
    depth: renderer.depth,
    stencil: renderer.stencil,
    logarithmicDepthBuffer: renderer.logarithmicDepthBuffer,
    outputColorSpace: renderer.outputColorSpace,
    reversedDepthBuffer: renderer.reversedDepthBuffer,
    parameterReversedDepthBuffer: backendParameters.reversedDepthBuffer === true,
    trackTimestamp: renderer.backend.trackTimestamp,
    parameterTrackTimestamp: backendParameters.trackTimestamp === true,
    timestampQueryPoolsEmpty: Object.values(renderer.backend.timestampQueryPool ?? {})
      .every((pool) => pool === null),
    autoClear: renderer.autoClear,
    autoClearColor: renderer.autoClearColor,
    autoClearDepth: renderer.autoClearDepth,
    autoClearStencil: renderer.autoClearStencil,
    toneMapping: renderer.toneMapping,
    toneMappingExposure: renderer.toneMappingExposure,
    backendDeviceId: runtime.instrumentation.identify(
      renderer.backend.device,
      'device',
    ),
    backendParameterDeviceId: runtime.instrumentation.identify(
      backendParameters.device,
      'device',
    ),
    backendDeviceMatchesParameter: renderer.backend.device === backendParameters.device,
    sortObjects: renderer.sortObjects,
    clearColor: rendererClearColor.getHex(),
    clearAlpha: renderer.getClearAlpha(),
    sceneBackground: productionScene.background.getHex(),
  };
  const rendererPass = rendererState.viewport.width === expected.viewport.width
    && rendererState.viewport.height === expected.viewport.height
    && rendererState.viewport.devicePixelRatio === expected.viewport.devicePixelRatio
    && rendererState.antialias === expected.renderer.antialias
    && rendererState.samples === expected.renderer.samples
    && rendererState.powerPreference === expected.renderer.powerPreference
    && rendererState.forceWebGL === expected.renderer.forceWebGL
    && rendererState.alpha === expected.renderer.alpha
    && rendererState.depth === expected.renderer.depth
    && rendererState.stencil === expected.renderer.stencil
    && rendererState.logarithmicDepthBuffer === expected.renderer.logarithmicDepthBuffer
    && rendererState.outputColorSpace === expected.renderer.outputColorSpace
    && rendererState.reversedDepthBuffer === expected.renderer.reversedDepthBuffer
    && rendererState.parameterReversedDepthBuffer
      === expected.renderer.reversedDepthBuffer
    && rendererState.trackTimestamp === expected.renderer.trackTimestamp
    && rendererState.parameterTrackTimestamp === expected.renderer.trackTimestamp
    && rendererState.timestampQueryPoolsEmpty
    && rendererState.autoClear === expected.renderer.autoClear
    && rendererState.autoClearColor === expected.renderer.autoClearColor
    && rendererState.autoClearDepth === expected.renderer.autoClearDepth
    && rendererState.autoClearStencil === expected.renderer.autoClearStencil
    && rendererState.toneMapping === expected.renderer.toneMapping
    && rendererState.toneMappingExposure === expected.renderer.toneMappingExposure
    && rendererState.backendDeviceMatchesParameter
    && rendererState.sortObjects === expected.renderer.sortObjects
    && rendererState.clearColor === expected.clear.color
    && rendererState.clearAlpha === expected.clear.alpha
    && rendererState.sceneBackground === expected.clear.color;

  const targets = {
    production: captureTargetResourceEvidence(
      runtime,
      'production',
      runtime.productionTarget,
      expected.targets.production,
      { role: 'production-color-depth', outputChannels: ['color', 'depth'] },
    ),
    objectId: captureTargetResourceEvidence(
      runtime,
      'objectId',
      runtime.objectIdDiagnostics.target,
      expected.targets.objectId,
      { role: 'diagnostic-object-id', outputChannels: ['objectId'] },
    ),
    address: captureTargetResourceEvidence(
      runtime,
      'address',
      runtime.addressDiagnostics.target,
      expected.targets.address,
      { role: 'diagnostic-exact-address', outputChannels: [] },
    ),
  };
  targets.address.addressWitnessIds = [...runtime.addressWitnesses.keys()].sort();
  const pass = cameraPass
    && rendererPass
    && lightsPass
    && Object.values(materials).every((material) => material.pass)
    && Object.values(targets).every((target) => target.pass);
  const result = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-common-resource-evidence',
    pass,
    expected: structuredClone(expected),
    camera: { pass: cameraPass, ...cameraState },
    renderer: { pass: rendererPass, ...rendererState },
    lights: { pass: lightsPass, ...lights },
    materials,
    targets,
  };
  requireCondition(result.pass, 'Common camera/material/target state is not exact.', result);
  return result;
}

function creationEventInventory(instrumentation) {
  const evidence = instrumentation.evidence;
  return [
    ...evidence.resources.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: record.method,
      resourceId: record.resourceId,
      resourceClass: record.resourceClass ?? null,
      ...(record.method === 'createBuffer' ? {
        size: record.size,
        usage: record.usage,
        mappedAtCreation: record.mappedAtCreation,
      } : {}),
    })),
    ...evidence.renderPipelines.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: record.method,
      resourceId: record.pipelineId,
      resourceClass: 'render-pipeline',
    })),
    ...evidence.computePipelines.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: record.method,
      resourceId: record.pipelineId,
      resourceClass: 'compute-pipeline',
    })),
    ...evidence.shaderModules.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: 'createShaderModule',
      resourceId: record.moduleId,
      resourceClass: 'shader-module',
    })),
    ...evidence.bindGroupLayouts.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: 'createBindGroupLayout',
      resourceId: record.layoutId,
      resourceClass: 'bind-group-layout',
    })),
    ...evidence.pipelineLayouts.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: 'createPipelineLayout',
      resourceId: record.layoutId,
      resourceClass: 'pipeline-layout',
    })),
    ...evidence.textureViews.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: 'createTextureView',
      resourceId: record.viewId,
      resourceClass: 'texture-view',
      parentTextureId: record.textureId,
    })),
    ...evidence.renderBundleEncoders.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: 'createRenderBundleEncoder',
      resourceId: record.bundleId ?? record.encoderId,
      resourceClass: 'render-bundle',
      encoderId: record.encoderId,
      bundleId: record.bundleId,
      finishSequence: record.finishSequence ?? null,
      nativeFinishReturned: record.nativeFinishReturned,
    })),
  ].sort((left, right) => left.sequence - right.sequence);
}

function capturePersistentResourceInventory(runtime, label) {
  const marker = runtime.instrumentation.mark('phase0-persistent-resource-freeze', {
    label,
    scenarioId: runtime.strategy.activeScenarioId,
  });
  const records = creationEventInventory(runtime.instrumentation).filter((record) => (
    record.resourceClass !== 'readback-staging'
  ));
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-persistent-resource-freeze',
    pass: records.every((record) => Number.isInteger(record.sequence)
      && record.sequence < marker.sequence
      && typeof record.resourceId === 'string'),
    label,
    scenarioId: runtime.strategy.activeScenarioId,
    markerSequence: marker.sequence,
    resourceCount: records.length,
    records,
  };
}

function scenarioPersistentResourceDelta(scenarioId, preflight, postflight) {
  const beforePersistent = preflight.records.filter(
    (record) => record.resourceClass !== 'render-bundle',
  );
  const afterPersistent = postflight.records.filter(
    (record) => record.resourceClass !== 'render-bundle',
  );
  const beforeBundles = preflight.records.filter(
    (record) => record.resourceClass === 'render-bundle',
  );
  const afterBundles = postflight.records.filter(
    (record) => record.resourceClass === 'render-bundle',
  );
  const beforeBundleIds = new Set(beforeBundles.map((record) => record.bundleId));
  const retainedBundles = afterBundles.filter((record) => beforeBundleIds.has(record.bundleId));
  const addedBundles = afterBundles.filter((record) => !beforeBundleIds.has(record.bundleId));
  const scheduleIds = ['S1', 'S2', 'canonical'];
  const expectedDiagnosticPhases = scheduleIds.flatMap((scheduleId) => [
    `phase0/diagnostic-rerecord/${scenarioId}/${scheduleId}/I/address`,
    `phase0/diagnostic-rerecord/${scenarioId}/${scheduleId}/I/object-id`,
  ]);
  const diagnosticBundles = addedBundles.filter(
    (record) => expectedDiagnosticPhases.includes(record.capturePhase),
  );
  const productionBundles = addedBundles.filter((record) => (
    new RegExp(`^phase0/${scenarioId}/(?:S1|S2|canonical)/I/capture-[0-9]+/production$`)
      .test(record.capturePhase ?? '')
  ));
  const productionSchedules = productionBundles.map((record) => (
    record.capturePhase.split('/')[2]
  ));
  const exactRetainedBundles = retainedBundles.length === beforeBundles.length
    && beforeBundles.every((before) => {
      const after = retainedBundles.find((record) => record.bundleId === before.bundleId);
      return after !== undefined && exactStructuredValue(before, after);
    });
  const addedBundleTopologyExact = addedBundles.length === 9
    && addedBundles.every((record) => record.nativeFinishReturned === true
      && typeof record.bundleId === 'string'
      && Number.isInteger(record.finishSequence)
      && record.finishSequence > record.sequence)
    && diagnosticBundles.length === 6
    && exactNumberArray(
      diagnosticBundles.map((record) => record.capturePhase).sort(),
      [...expectedDiagnosticPhases].sort(),
    )
    && productionBundles.length === 3
    && exactNumberArray(productionSchedules, scheduleIds);
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-prescribed-persistent-resource-delta',
    pass: exactStructuredValue(beforePersistent, afterPersistent)
      && exactRetainedBundles
      && addedBundleTopologyExact,
    scenarioId,
    nonBundlePersistentExact: exactStructuredValue(beforePersistent, afterPersistent),
    retainedBundleCount: retainedBundles.length,
    expectedRetainedBundleCount: beforeBundles.length,
    exactRetainedBundles,
    expectedAddedBundleCount: 9,
    addedBundleCount: addedBundles.length,
    addedBundleTopologyExact,
    expectedDiagnosticPhases,
    diagnosticBundles,
    productionSchedules,
    productionBundles,
    addedBundles,
  };
}

function aBucketBaseCpuProjection(runtime, context) {
  const attribute = runtime.strategy.laneStates.A.geometry.getAttribute('bucketBase');
  const expectedBases = context.commandOracle.sourceBaseByDraw;
  let vertexCursor = 0;
  const spans = runtime.sourceGeometries.map((geometry, draw) => {
    const vertexCount = geometry.getAttribute('position').count;
    const start = vertexCursor;
    const end = start + vertexCount;
    const expectedBase = expectedBases[draw];
    const exact = attribute.array.subarray(start, end).every(
      (value) => value === expectedBase,
    );
    vertexCursor = end;
    return { draw, start, end, vertexCount, expectedBase, exact };
  });
  return {
    attribute,
    pass: vertexCursor === attribute.array.length && spans.every((span) => span.exact),
    spans,
  };
}

async function realizeMergedGeometryForFreeze(runtime) {
  requireCondition(runtime.mergedGeometryRealization === null,
    'Merged production geometry may be explicitly realized only once.',
    runtime.mergedGeometryRealization);
  const { renderer, strategy, instrumentation } = runtime;
  const manager = renderer._attributes;
  requireCondition(manager?.constructor?.name === 'Attributes'
    && typeof manager.update === 'function'
    && typeof manager.get === 'function',
  'Pinned Three renderer attribute manager is unavailable for geometry realization.', {
      constructorName: manager?.constructor?.name ?? null,
    });
  const geometry = strategy.laneStates.A.geometry;
  const expectedAttributeNames = ['bucketBase', 'normal', 'position', 'uv'];
  requireCondition(exactNumberArray(
    Object.keys(geometry.attributes).sort(),
    expectedAttributeNames,
  ) && geometry.index !== null,
  'Merged production geometry does not have the exact frozen attribute topology.', {
      observedAttributeNames: Object.keys(geometry.attributes).sort(),
      expectedAttributeNames,
      hasIndex: geometry.index !== null,
    });
  const calls = [
    ...expectedAttributeNames.map((semantic) => ({
      semantic,
      resourceKind: 'vertex-attribute',
      attribute: geometry.getAttribute(semantic),
      attributeType: THREE_ATTRIBUTE_TYPE_VERTEX,
    })),
    {
      semantic: 'index',
      resourceKind: 'index',
      attribute: geometry.index,
      attributeType: THREE_ATTRIBUTE_TYPE_INDEX,
    },
  ];
  const start = instrumentation.mark('phase0-merged-geometry-realization-start', {
    geometryUuid: geometry.uuid,
  });
  const records = [];
  const queueWriteBufferStartIndex = instrumentation.evidence.queueWriteBuffers.length;
  instrumentation.setCapturePhase('phase0/resources/merged-geometry-realization');
  try {
    for (const call of calls) {
      requireCondition(call.attribute?.array && ArrayBuffer.isView(call.attribute.array),
        `Merged production geometry ${call.semantic} is not a typed-array attribute.`);
      const beforeDataVersion = manager.get(call.attribute).version ?? null;
      const before = instrumentation.mark(
        'phase0-merged-geometry-attribute-update-start',
        {
          semantic: call.semantic,
          attributeId: call.attribute.id,
          attributeVersion: call.attribute.version,
          attributeType: call.attributeType,
          beforeDataVersion,
        },
      );
      manager.update(call.attribute, call.attributeType);
      const gpuBuffer = renderer.backend.get(call.attribute)?.buffer ?? null;
      const afterDataVersion = manager.get(call.attribute).version ?? null;
      const after = instrumentation.mark(
        'phase0-merged-geometry-attribute-update-complete',
        {
          semantic: call.semantic,
          attributeId: call.attribute.id,
          attributeVersion: call.attribute.version,
          attributeType: call.attributeType,
          afterDataVersion,
          gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
        },
      );
      records.push({
        semantic: call.semantic,
        resourceKind: call.resourceKind,
        manager: 'renderer._attributes',
        managerConstructor: manager.constructor.name,
        attributeId: call.attribute.id,
        attributeVersion: call.attribute.version,
        attributeType: call.attributeType,
        arrayType: call.attribute.array.constructor.name,
        byteLength: call.attribute.array.byteLength,
        cpuSha256: await sha256Bytes(call.attribute.array),
        beforeDataVersion,
        afterDataVersion,
        gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
        gpuBufferSize: gpuBuffer?.size ?? null,
        startSequence: before.sequence,
        completeSequence: after.sequence,
      });
    }
    await renderer.backend.device.queue.onSubmittedWorkDone();
  } finally {
    instrumentation.setCapturePhase(null);
  }
  const queueComplete = instrumentation.mark(
    'phase0-merged-geometry-realization-queue-complete',
    { geometryUuid: geometry.uuid },
  );
  const queueWriteBuffers = instrumentation.evidence.queueWriteBuffers
    .slice(queueWriteBufferStartIndex)
    .filter((record) => record.sequence < queueComplete.sequence);
  const managerVersionsExact = records.every((record) => (
    record.afterDataVersion === record.attributeVersion
      && (record.semantic === 'uv'
        ? record.beforeDataVersion === null
        : record.beforeDataVersion === record.attributeVersion)
  ));
  const pass = exactNumberArray(
    records.map((record) => record.semantic),
    [...expectedAttributeNames, 'index'],
  )
    && managerVersionsExact
    && queueWriteBuffers.length === 0
    && records.every((record, index) => record.manager === 'renderer._attributes'
      && record.managerConstructor === 'Attributes'
      && record.attributeType === (index < 4
        ? THREE_ATTRIBUTE_TYPE_VERTEX
        : THREE_ATTRIBUTE_TYPE_INDEX)
      && typeof record.gpuBufferId === 'string'
      && record.startSequence < record.completeSequence
      && record.completeSequence < queueComplete.sequence)
    && new Set(records.map((record) => record.gpuBufferId)).size === 5;
  const result = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-merged-geometry-realization',
    pass,
    frozen: true,
    capturePhase: 'phase0/resources/merged-geometry-realization',
    geometryUuid: geometry.uuid,
    startSequence: start.sequence,
    queueCompleteSequence: queueComplete.sequence,
    expectedAttributeNames,
    recordCount: records.length,
    records,
    managerVersionsExact,
    queueWriteBufferCount: queueWriteBuffers.length,
    queueWriteBuffers,
    frozenResourceIds: records.map((record) => record.gpuBufferId),
  };
  requireCondition(result.pass,
    'Merged production geometry realization is incomplete.', result);
  runtime.mergedGeometryRealization = result;
  return result;
}

async function realizeAddressDiagnosticPositions(runtime, {
  installationId,
  scenarioId,
  scheduleId,
  basePhase,
}) {
  const { renderer, strategy, instrumentation, addressDiagnostics, sourceGeometries } = runtime;
  const manager = renderer._attributes;
  requireCondition(addressDiagnostics !== null
    && manager?.constructor?.name === 'Attributes'
    && typeof manager.update === 'function'
    && typeof manager.get === 'function',
  'Address diagnostic position manager is unavailable.');
  const productionIndex = strategy.laneStates.A.geometry.index;
  const firstIndexes = strategy.sharedResources.firstIndexes;
  const canonicalBases = IMMEDIATE_AIF_PHASE0_SCHEDULES.canonical.sourceBaseByDraw;
  const scheduleBases = IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId].sourceBaseByDraw;
  const expectedCommon = createChallengePositions({
    sourceGeometries,
    productionIndex,
    firstIndexes,
    sourceBaseByDraw: canonicalBases,
    firstInstanceLane: false,
  }).positions;
  const expectedFeature = createChallengePositions({
    sourceGeometries,
    productionIndex,
    firstIndexes,
    sourceBaseByDraw: scheduleBases,
    firstInstanceLane: true,
  }).positions;
  const calls = [
    {
      semantic: 'commonPosition',
      attribute: addressDiagnostics.commonPosition,
      expected: expectedCommon,
    },
    {
      semantic: 'featurePosition',
      attribute: addressDiagnostics.featurePosition,
      expected: expectedFeature,
    },
  ];
  const capturePhase = `${basePhase}/diagnostic-position-realization`;
  const start = instrumentation.mark('phase0-address-position-realization-start', {
    installationId,
    scenarioId,
    scheduleId,
  });
  const queueWriteStartIndex = instrumentation.evidence.queueWriteBuffers.length;
  const records = [];
  instrumentation.setCapturePhase(capturePhase);
  try {
    for (const call of calls) {
      const cpuExact = call.attribute.array.length === call.expected.length
        && call.attribute.array.every((value, index) => Object.is(value, call.expected[index]));
      const beforeDataVersion = manager.get(call.attribute).version ?? null;
      const updateStart = instrumentation.mark('phase0-address-position-update-start', {
        installationId,
        semantic: call.semantic,
        attributeId: call.attribute.id,
        attributeVersion: call.attribute.version,
        beforeDataVersion,
      });
      manager.update(call.attribute, THREE_ATTRIBUTE_TYPE_VERTEX);
      const afterDataVersion = manager.get(call.attribute).version ?? null;
      const gpuBuffer = renderer.backend.get(call.attribute)?.buffer ?? null;
      const gpuBufferId = instrumentation.identify(gpuBuffer, 'buffer');
      const gpuBufferCreations = instrumentation.evidence.resources.filter(
        (resource) => resource.method === 'createBuffer'
          && resource.resourceId === gpuBufferId,
      );
      const gpuBufferCreation = gpuBufferCreations.length === 1
        ? gpuBufferCreations[0]
        : null;
      const updateComplete = instrumentation.mark('phase0-address-position-update-complete', {
        installationId,
        semantic: call.semantic,
        attributeId: call.attribute.id,
        attributeVersion: call.attribute.version,
        afterDataVersion,
        gpuBufferId,
      });
      const cpuSha256 = await sha256Bytes(call.attribute.array);
      const witnessId = await registerAddressPositionCpuWitness(
        runtime,
        call.attribute.array,
      );
      records.push({
        semantic: call.semantic,
        manager: 'renderer._attributes',
        managerConstructor: manager.constructor.name,
        attributeType: THREE_ATTRIBUTE_TYPE_VERTEX,
        attributeId: call.attribute.id,
        attributeVersion: call.attribute.version,
        beforeDataVersion,
        afterDataVersion,
        dirtyBeforeUpdate: beforeDataVersion !== null
          && beforeDataVersion !== call.attribute.version,
        arrayType: call.attribute.array.constructor.name,
        itemSize: call.attribute.itemSize,
        count: call.attribute.count,
        byteLength: call.attribute.array.byteLength,
        cpuExact,
        cpuSha256,
        witnessId,
        gpuBufferId,
        gpuBufferSize: gpuBuffer?.size ?? null,
        gpuBufferCreation: gpuBufferCreation === null ? null : {
          sequence: gpuBufferCreation.sequence,
          capturePhase: gpuBufferCreation.capturePhase,
          resourceId: gpuBufferCreation.resourceId,
          resourceClass: gpuBufferCreation.resourceClass,
          size: gpuBufferCreation.size,
          usage: gpuBufferCreation.usage,
          mappedAtCreation: gpuBufferCreation.mappedAtCreation,
        },
        updateStartSequence: updateStart.sequence,
        updateCompleteSequence: updateComplete.sequence,
      });
    }
    await renderer.backend.device.queue.onSubmittedWorkDone();
  } finally {
    instrumentation.setCapturePhase(null);
  }
  const queueComplete = instrumentation.mark('phase0-address-position-queue-complete', {
    installationId,
  });
  const rawQueueWriteBuffers = instrumentation.evidence.queueWriteBuffers
    .slice(queueWriteStartIndex)
    .filter((record) => record.sequence < queueComplete.sequence);
  const queueWriteBuffers = rawQueueWriteBuffers.map((write) => {
    const source = records.find((record) => record.gpuBufferId === write.bufferId) ?? null;
    return {
      ...write,
      semantic: source?.semantic ?? null,
      sourceSha256: source?.cpuSha256 ?? null,
      sourceWitnessId: source?.witnessId ?? null,
    };
  });
  const expectedWriteSemantics = scenarioId === 'v99'
      && runtime.scheduleInstallations.filter(
        (installation) => installation.scenarioId === scenarioId,
      ).length === 0
    ? []
    : ['featurePosition'];
  const observedWriteSemantics = queueWriteBuffers.map((write) => (
    records.find((record) => record.gpuBufferId === write.bufferId)?.semantic ?? null
  ));
  const queueWritesExact = exactNumberArray(
    observedWriteSemantics,
    expectedWriteSemantics,
  ) && queueWriteBuffers.every((write) => {
    const record = records.find((candidate) => candidate.gpuBufferId === write.bufferId);
    return record !== undefined
      && write.capturePhase === capturePhase
      && write.bufferOffset === 0
      && write.sourceType === 'Float32Array'
      && write.sourceByteLength === record.byteLength
      && write.dataOffset === 0
      && write.size === null
      && write.semantic === record.semantic
      && write.sourceSha256 === record.cpuSha256
      && write.sourceWitnessId === record.witnessId
      && write.sequence > record.updateStartSequence
      && write.sequence < record.updateCompleteSequence;
  });
  const common = records[0];
  const feature = records[1];
  const laneBindings = {
    A: { semantic: 'commonPosition', attributeId: common.attributeId,
      gpuBufferId: common.gpuBufferId },
    I: { semantic: 'commonPosition', attributeId: common.attributeId,
      gpuBufferId: common.gpuBufferId },
    F: { semantic: 'featurePosition', attributeId: feature.attributeId,
      gpuBufferId: feature.gpuBufferId },
  };
  const pass = records.length === 2
    && records.every((record) => record.manager === 'renderer._attributes'
      && record.managerConstructor === 'Attributes'
      && record.attributeType === THREE_ATTRIBUTE_TYPE_VERTEX
      && record.afterDataVersion === record.attributeVersion
      && record.cpuExact
      && typeof record.gpuBufferId === 'string'
      && record.gpuBufferSize === record.byteLength
      && record.gpuBufferCreation?.resourceId === record.gpuBufferId
      && record.gpuBufferCreation.resourceClass === 'persistent-or-upload-buffer'
      && record.gpuBufferCreation.size === record.byteLength
      && record.gpuBufferCreation.usage === (0x0020 | 0x0004 | 0x0008)
      && record.gpuBufferCreation.mappedAtCreation === true
      && record.updateStartSequence < record.updateCompleteSequence
      && record.updateCompleteSequence < queueComplete.sequence)
    && new Set(records.map((record) => record.attributeId)).size === 2
    && new Set(records.map((record) => record.gpuBufferId)).size === 2
    && common.dirtyBeforeUpdate === false
    && feature.dirtyBeforeUpdate === (expectedWriteSemantics.length === 1)
    && queueWritesExact
    && laneBindings.A.attributeId === laneBindings.I.attributeId
    && laneBindings.A.gpuBufferId === laneBindings.I.gpuBufferId
    && laneBindings.F.attributeId !== laneBindings.A.attributeId
    && laneBindings.F.gpuBufferId !== laneBindings.A.gpuBufferId;
  const result = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-address-position-realization',
    pass,
    installationId,
    scenarioId,
    scheduleId,
    capturePhase,
    startSequence: start.sequence,
    queueCompleteSequence: queueComplete.sequence,
    expectedWriteSemantics,
    observedWriteSemantics,
    queueWriteBufferCount: queueWriteBuffers.length,
    queueWriteBuffers,
    queueWritesExact,
    records,
    laneBindings,
  };
  requireCondition(result.pass,
    `Address diagnostic positions were not frozen for ${installationId}.`, result);
  return result;
}

async function realizeScheduleInstallation(runtime, reason) {
  const { renderer, strategy, instrumentation } = runtime;
  const scenarioId = strategy.activeScenarioId;
  const scheduleId = strategy.state.activeScheduleId;
  const ordinal = runtime.scheduleInstallations.length + 1;
  const scenarioOrdinal = runtime.scheduleInstallations.filter(
    (installation) => installation.scenarioId === scenarioId,
  ).length + 1;
  const installationId = `${scenarioId}/${ordinal}/${scheduleId}`;
  const basePhase = `phase0/schedule-install/${scenarioId}/${ordinal}-${scheduleId}`;
  const start = instrumentation.mark('phase0-schedule-install-start', {
    installationId,
    scenarioId,
    scheduleId,
    reason,
  });
  const manager = renderer._attributes;
  requireCondition(manager?.constructor?.name === 'Attributes'
    && typeof manager.update === 'function'
    && typeof manager.get === 'function',
  'Pinned Three renderer attribute manager is unavailable.', {
      constructorName: manager?.constructor?.name ?? null,
    });
  const aContext = strategy.getLaneValidationResources('A');
  const aBucket = aBucketBaseCpuProjection(runtime, aContext);
  requireCondition(aBucket.pass, 'A bucket-base CPU stream is not exact before realization.', {
    installationId,
    spans: aBucket.spans,
  });
  const calls = [
    {
      semantic: 'A.bucketBase',
      lane: 'A',
      attribute: aBucket.attribute,
      attributeType: THREE_ATTRIBUTE_TYPE_VERTEX,
    },
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
      semantic: `${lane}.indirectCommands`,
      lane,
      attribute: strategy.laneStates[lane].indirectAttribute,
      attributeType: THREE_ATTRIBUTE_TYPE_INDIRECT,
    })),
  ];
  const managerCalls = [];
  const installationQueueWriteStartIndex = instrumentation.evidence.queueWriteBuffers.length;
  instrumentation.setCapturePhase(`${basePhase}/attribute-realization`);
  try {
    for (const call of calls) {
      const beforeDataVersion = manager.get(call.attribute).version ?? null;
      const before = instrumentation.mark('phase0-attribute-manager-update-start', {
        installationId,
        semantic: call.semantic,
        attributeId: call.attribute.id,
        attributeVersion: call.attribute.version,
        attributeType: call.attributeType,
        beforeDataVersion,
      });
      manager.update(call.attribute, call.attributeType);
      const gpuBuffer = renderer.backend.get(call.attribute)?.buffer ?? null;
      const afterDataVersion = manager.get(call.attribute).version ?? null;
      const after = instrumentation.mark('phase0-attribute-manager-update-complete', {
        installationId,
        semantic: call.semantic,
        attributeId: call.attribute.id,
        attributeVersion: call.attribute.version,
        attributeType: call.attributeType,
        afterDataVersion,
        gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
      });
      managerCalls.push({
        semantic: call.semantic,
        lane: call.lane,
        manager: 'renderer._attributes',
        managerConstructor: manager.constructor.name,
        attributeId: call.attribute.id,
        attributeVersion: call.attribute.version,
        attributeType: call.attributeType,
        beforeDataVersion,
        afterDataVersion,
        dirtyBeforeUpdate: beforeDataVersion !== null
          && beforeDataVersion !== call.attribute.version,
        arrayType: call.attribute.array.constructor.name,
        byteLength: call.attribute.array.byteLength,
        cpuSha256: await sha256Bytes(call.attribute.array),
        gpuBufferId: instrumentation.identify(gpuBuffer, 'buffer'),
        gpuBufferSize: gpuBuffer?.size ?? null,
        startSequence: before.sequence,
        completeSequence: after.sequence,
      });
    }
    await renderer.backend.device.queue.onSubmittedWorkDone();
  } finally {
    instrumentation.setCapturePhase(null);
  }
  const queueComplete = instrumentation.mark('phase0-schedule-install-queue-complete', {
    installationId,
  });
  const rawQueueWriteBuffers = instrumentation.evidence.queueWriteBuffers
    .slice(installationQueueWriteStartIndex)
    .filter((record) => record.capturePhase === `${basePhase}/attribute-realization`
      && record.sequence < queueComplete.sequence);
  const observedDirtySemantics = managerCalls.filter(
    (record) => record.dirtyBeforeUpdate,
  ).map((record) => record.semantic);
  const expectedWriteSemantics = scenarioOrdinal === 1
    ? scenarioId === 'v99'
      ? []
      : ['A.bucketBase']
    : ['A.bucketBase', 'F.indirectCommands'];
  const queueWriteBuffers = rawQueueWriteBuffers.map((write) => {
    const source = managerCalls.find((record) => record.gpuBufferId === write.bufferId) ?? null;
    return {
      ...write,
      semantic: source?.semantic ?? null,
      sourceSha256: source?.cpuSha256 ?? null,
    };
  });
  const observedWriteSemantics = queueWriteBuffers.map((write) => (
    managerCalls.find((record) => record.gpuBufferId === write.bufferId)?.semantic ?? null
  ));
  const queueWritesExact = exactNumberArray(observedDirtySemantics, expectedWriteSemantics)
    && exactNumberArray(observedWriteSemantics, expectedWriteSemantics)
    && queueWriteBuffers.every((write) => {
      const managerCall = managerCalls.find(
        (record) => record.gpuBufferId === write.bufferId,
      );
      return managerCall !== undefined
        && write.bufferOffset === 0
        && write.sourceType === managerCall.arrayType
        && write.sourceByteLength === managerCall.byteLength
        && write.dataOffset === 0
        && write.size === null
        && write.semantic === managerCall.semantic
        && write.sourceSha256 === managerCall.cpuSha256
        && write.sequence > managerCall.startSequence
        && write.sequence < managerCall.completeSequence;
    });
  const diagnosticPositionRealization = await realizeAddressDiagnosticPositions(runtime, {
    installationId,
    scenarioId,
    scheduleId,
    basePhase,
  });
  const commands = {};
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const context = strategy.getLaneValidationResources(lane);
    const capturePhase = `${basePhase}/command-readback/${lane}`;
    instrumentation.setCapturePhase(capturePhase);
    const readbackStart = instrumentation.mark('phase0-command-preflight-readback-start', {
      installationId,
      lane,
      attributeId: context.indirectAttribute.id,
    });
    let gpu;
    try {
      gpu = new Uint32Array(await renderer.getArrayBufferAsync(context.indirectAttribute));
    } finally {
      instrumentation.setCapturePhase(null);
    }
    const readbackComplete = instrumentation.mark(
      'phase0-command-preflight-readback-complete',
      { installationId, lane },
    );
    const expected = context.commandOracle.postCullCommands;
    const gpuSha256 = await sha256Bytes(gpu);
    const expectedSha256 = await sha256Bytes(expected);
    commands[lane] = {
      lane,
      capturePhase,
      attributeId: context.indirectAttribute.id,
      attributeVersion: context.indirectAttribute.version,
      gpuBufferId: instrumentation.identify(
        renderer.backend.get(context.indirectAttribute)?.buffer ?? null,
        'buffer',
      ),
      byteLength: gpu.byteLength,
      gpuSha256,
      expectedSha256,
      readbackStartSequence: readbackStart.sequence,
      readbackCompleteSequence: readbackComplete.sequence,
      exact: gpu.byteLength === IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength
        && gpu.length === expected.length
        && gpu.every((value, index) => value === expected[index]),
      frozenWords: Object.freeze(Array.from(gpu)),
      expectedWords: Object.freeze(Array.from(expected)),
    };
  }
  const commonStatePreflight = await captureCommonRenderStateCommitment(
    runtime,
    `${installationId}/preflight`,
  );
  const complete = instrumentation.mark('phase0-schedule-install-complete', {
    installationId,
  });
  const frozenResourceIds = [
    managerCalls.find((call) => call.semantic === 'A.bucketBase')?.gpuBufferId,
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => commands[lane].gpuBufferId),
  ];
  const commandSetSha256 = await sha256Text(JSON.stringify(
    IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
      lane,
      gpuSha256: commands[lane].gpuSha256,
      frozenWords: commands[lane].frozenWords,
    })),
  ));
  const record = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-schedule-installation-preflight',
    pass: managerCalls.length === 4
      && exactNumberArray(
        managerCalls.map((call) => call.semantic),
        ['A.bucketBase', 'A.indirectCommands', 'I.indirectCommands', 'F.indirectCommands'],
      )
      && managerCalls.every((call, index) => typeof call.gpuBufferId === 'string'
        && call.startSequence < call.completeSequence
        && (index === 0
          ? call.attributeType === THREE_ATTRIBUTE_TYPE_VERTEX
          : call.attributeType === THREE_ATTRIBUTE_TYPE_INDIRECT))
      && queueWritesExact
      && Object.values(commands).every((command) => command.exact
        && command.gpuSha256 === command.expectedSha256
        && command.byteLength === IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength
        && command.readbackStartSequence > queueComplete.sequence
        && command.readbackCompleteSequence > command.readbackStartSequence)
      && diagnosticPositionRealization.pass
      && diagnosticPositionRealization.startSequence > queueComplete.sequence
      && diagnosticPositionRealization.queueCompleteSequence < complete.sequence
      && new Set(frozenResourceIds).size === 4,
    installationId,
    ordinal,
    scenarioOrdinal,
    scenarioId,
    scheduleId,
    reason,
    basePhase,
    startSequence: start.sequence,
    queueCompleteSequence: queueComplete.sequence,
    completeSequence: complete.sequence,
    managerCalls,
    observedDirtySemantics,
    expectedWriteSemantics,
    observedWriteSemantics,
    queueWriteBufferCount: queueWriteBuffers.length,
    queueWriteBuffers,
    queueWritesExact,
    diagnosticPositionRealization,
    aBucketBase: {
      attributeId: aBucket.attribute.id,
      attributeVersion: aBucket.attribute.version,
      byteLength: aBucket.attribute.array.byteLength,
      cpuSha256: await sha256Bytes(aBucket.attribute.array),
      exact: aBucket.pass,
      spans: aBucket.spans,
    },
    commands,
    commandSetSha256,
    frozen: true,
    frozenResourceIds,
    commonState: {
      pass: null,
      preflight: commonStatePreflight,
      postflight: null,
      exactProjection: null,
      exactSha256: null,
    },
    orderedChallenge: null,
  };
  requireCondition(record.pass, 'Schedule installation preflight failed.', record);
  runtime.scheduleInstallations.push(record);
  runtime.activeScheduleInstallation = record;
  return record;
}

async function finishScheduleInstallation(runtime, installation, label) {
  requireCondition(installation === runtime.activeScheduleInstallation,
    'Cannot postflight an inactive schedule installation.', {
      requested: installation?.installationId ?? null,
      active: runtime.activeScheduleInstallation?.installationId ?? null,
    });
  const postflight = await captureCommonRenderStateCommitment(runtime, label);
  const exactProjection = exactStructuredValue(
    postflight.projection,
    installation.commonState.preflight.projection,
  );
  const exactSha256 = postflight.sha256 === installation.commonState.preflight.sha256;
  installation.commonState = {
    pass: exactProjection
      && exactSha256
      && installation.commonState.preflight.laneSelection.exact
      && postflight.laneSelection.exact,
    preflight: installation.commonState.preflight,
    postflight,
    exactProjection,
    exactSha256,
  };
  installation.pass = installation.pass && installation.commonState.pass;
  requireCondition(installation.pass,
    `Common render state changed during ${installation.installationId}.`, installation.commonState);
  return installation.commonState;
}

async function resourceState(runtime, label) {
  const { renderer, strategy, instrumentation } = runtime;
  const attributes = strategy.sharedResources.attributes;
  return {
    label,
    rendererMemory: structuredClone(renderer.info.memory),
    instrumentedGpuResourceCount: instrumentation.evidence.resources.length,
    rootUuid: strategy.root.uuid,
    aBucketBaseVersion: strategy.laneStates.A.geometry.getAttribute('bucketBase').version,
    laneRoots: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [lane, {
      uuid: strategy.laneStates[lane].root.uuid,
      version: strategy.laneStates[lane].root.version,
      bundleRecordCount: strategy.laneStates[lane].bundleRecordCallbackCount,
    }])),
    geometries: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
      lane,
      strategy.laneStates[lane].geometry.uuid,
    ])),
    materials: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
      lane,
      strategy.laneStates[lane].material.uuid,
    ])),
    commandAttributes: Object.fromEntries(await Promise.all(
      IMMEDIATE_AIF_PHASE0_LANES.map(async (lane) => [
        lane,
        {
          id: strategy.laneStates[lane].indirectAttribute.id,
          version: strategy.laneStates[lane].indirectAttribute.version,
          sha256: await sha256Bytes(strategy.laneStates[lane].indirectAttribute.array),
        },
      ]),
    )),
    sharedAttributes: Object.fromEntries(await Promise.all(
      Object.entries(attributes).map(async ([name, attribute]) => [name, {
        id: attribute.id,
        version: attribute.version,
        byteLength: attribute.array.byteLength,
        sha256: await sha256Bytes(attribute.array),
      }]),
    )),
  };
}

async function scenarioLoadBoundaryState(runtime, label) {
  const state = await resourceState(runtime, label);
  const source = runtime.strategy.laneStates.I.indirectImmediateBases;
  return {
    ...state,
    activeScenarioId: runtime.strategy.activeScenarioId,
    activeLane: runtime.strategy.activeLane,
    immediateSource: {
      sourceId: runtime.instrumentation.identify(source, 'typed-array'),
      backingBufferId: runtime.instrumentation.identify(source?.buffer, 'array-buffer'),
      arrayType: source?.constructor?.name ?? null,
      length: source?.length ?? null,
      byteLength: source?.byteLength ?? null,
      sha256: source instanceof Uint32Array ? await sha256Bytes(source) : null,
      values: source instanceof Uint32Array ? Array.from(source) : null,
    },
  };
}

function scenarioLoadBoundaryDelta(before, after) {
  if (before === null) return null;
  const sharedAttributeVersionDeltas = Object.fromEntries(
    Object.keys(after.sharedAttributes).map((name) => [
      name,
      after.sharedAttributes[name].version - before.sharedAttributes[name].version,
    ]),
  );
  const commandAttributeVersionDeltas = Object.fromEntries(
    IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
      lane,
      after.commandAttributes[lane].version - before.commandAttributes[lane].version,
    ]),
  );
  const observed = {
    aBucketBaseVersion: after.aBucketBaseVersion - before.aBucketBaseVersion,
    sharedAttributeVersionDeltas,
    commandAttributeVersionDeltas,
    iRootVersion: after.laneRoots.I.version - before.laneRoots.I.version,
    iBundleRecordCount:
      after.laneRoots.I.bundleRecordCount - before.laneRoots.I.bundleRecordCount,
    immediateSourceIdentityStable:
      after.immediateSource.sourceId === before.immediateSource.sourceId
        && after.immediateSource.backingBufferId === before.immediateSource.backingBufferId,
  };
  const pass = observed.aBucketBaseVersion === 1
    && Object.values(sharedAttributeVersionDeltas).every((delta) => delta === 1)
    && Object.values(commandAttributeVersionDeltas).every((delta) => delta === 1)
    && observed.iRootVersion === 0
    && observed.iBundleRecordCount === 0
    && observed.immediateSourceIdentityStable;
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-v99-to-v20-load-boundary',
    pass,
    fromScenarioId: before.activeScenarioId,
    toScenarioId: after.activeScenarioId,
    before,
    after,
    observed,
  };
}

function stableResourceIdentity(left, right) {
  const identity = (value) => JSON.stringify({
    rendererMemory: value.rendererMemory,
    rootUuid: value.rootUuid,
    aBucketBaseVersion: value.aBucketBaseVersion,
    laneRoots: Object.fromEntries(Object.entries(value.laneRoots).map(
      ([lane, entry]) => [lane, { uuid: entry.uuid, version: entry.version,
        bundleRecordCount: entry.bundleRecordCount }],
    )),
    geometries: value.geometries,
    materials: value.materials,
    commandAttributes: value.commandAttributes,
    sharedAttributes: value.sharedAttributes,
  });
  return identity(left) === identity(right);
}

function prescribedTransitionDeltas(preflight, restored) {
  const observed = {
    aBucketBaseVersion: restored.aBucketBaseVersion - preflight.aBucketBaseVersion,
    fCommandVersion:
      restored.commandAttributes.F.version - preflight.commandAttributes.F.version,
    iRootVersion: restored.laneRoots.I.version - preflight.laneRoots.I.version,
    iBundleRecordCount:
      restored.laneRoots.I.bundleRecordCount - preflight.laneRoots.I.bundleRecordCount,
    aBundleRecordCount:
      restored.laneRoots.A.bundleRecordCount - preflight.laneRoots.A.bundleRecordCount,
    fBundleRecordCount:
      restored.laneRoots.F.bundleRecordCount - preflight.laneRoots.F.bundleRecordCount,
  };
  return {
    pass: observed.aBucketBaseVersion === 3
      && observed.fCommandVersion === 3
      && observed.iRootVersion === 3
      && observed.iBundleRecordCount === 3
      && observed.aBundleRecordCount === 0
      && observed.fBundleRecordCount === 0,
    expected: {
      aBucketBaseVersion: 3,
      fCommandVersion: 3,
      iRootVersion: 3,
      iBundleRecordCount: 3,
      aBundleRecordCount: 0,
      fBundleRecordCount: 0,
    },
    observed,
  };
}

function computeSemanticAttribute(strategy, lane, semantic) {
  if (semantic === 'indirectCommands') return strategy.laneStates[lane].indirectAttribute;
  return strategy.sharedResources.attributes[semantic] ?? null;
}

function bindComputeExecutionEvidence(runtime, computeLanes, scenarioId) {
  const { instrumentation, renderer, strategy } = runtime;
  const laneEvidence = {};
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const capturePhase = `phase0/live/${scenarioId}/${lane}`;
    const traces = instrumentation.evidence.computePassEncoders.filter(
      (trace) => trace.capturePhase === capturePhase,
    );
    requireCondition(traces.length === 1,
      `${lane} live compute execution trace is not unique.`, {
        capturePhase,
        traceCount: traces.length,
      });
    const trace = traces[0];
    const computeImmediateEvents = trace.events.filter(
      (event) => event.method === 'setImmediates',
    );
    const dynamicBindGroupEvents = trace.events.filter(
      (event) => event.method === 'setBindGroup'
        && (event.dynamicOffsets !== null
          || event.dynamicOffsetStart !== null
          || event.dynamicOffsetLength !== null
          || event.selectedDynamicOffsets !== null),
    );
    let activePipeline = null;
    const activeBindGroups = new Map();
    const dispatches = [];
    for (const event of trace.events) {
      if (event.method === 'setPipeline') activePipeline = event;
      if (event.method === 'setBindGroup') activeBindGroups.set(event.index, event);
      if (event.method === 'dispatchWorkgroups') {
        dispatches.push({
          event,
          pipelineId: activePipeline?.pipelineId ?? null,
          setPipeline: activePipeline === null ? null : structuredClone(activePipeline),
          bindGroups: [...activeBindGroups.values()].map((entry) => structuredClone(entry)),
        });
      }
    }
    requireCondition(dispatches.length === 2
      && !trace.events.some((event) => event.method === 'dispatchWorkgroupsIndirect'),
    `${lane} live compute pass must contain exactly reset and cull dispatches.`, trace);
    const phases = {};
    for (const [phaseIndex, phase] of ['reset', 'cull'].entries()) {
      const collector = computeLanes[lane].phases[phase];
      const dispatch = dispatches[phaseIndex];
      const expectedDispatch = phase === 'reset' ? [1, 1, 1] : [1_024, 1, 1];
      const pipelines = instrumentation.evidence.computePipelines.filter(
        (record) => record.pipelineId === dispatch.pipelineId,
      );
      requireCondition(pipelines.length === 1,
        `${lane}/${phase} compute pipeline descriptor is not unique.`, dispatch);
      const pipeline = pipelines[0];
      const modules = instrumentation.evidence.shaderModules.filter(
        (record) => record.moduleId === pipeline.computeModuleId,
      );
      const layouts = instrumentation.evidence.pipelineLayouts.filter(
        (record) => record.layoutId === pipeline.layoutId,
      );
      requireCondition(modules.length === 1 && layouts.length === 1,
        `${lane}/${phase} compute module/layout chain is not unique.`, pipeline);
      const module = modules[0];
      const layout = layouts[0];
      const bindGroupLayouts = layout.bindGroupLayoutIds.map((layoutId) => {
        const matches = instrumentation.evidence.bindGroupLayouts.filter(
          (record) => record.layoutId === layoutId,
        );
        requireCondition(matches.length === 1,
          `${lane}/${phase} bind-group layout ${layoutId} is not unique.`, matches);
        return matches[0];
      });
      const bindGroupChains = dispatch.bindGroups.map((bindGroupEvent) => {
        const bindGroups = instrumentation.evidence.resources.filter(
          (resource) => resource.method === 'createBindGroup'
            && resource.resourceId === bindGroupEvent.bindGroupId,
        );
        const bindGroup = bindGroups.length === 1 ? bindGroups[0] : null;
        const expectedLayoutId = layout.bindGroupLayoutIds[bindGroupEvent.index] ?? null;
        const bindGroupLayout = bindGroupLayouts.find(
          (candidate) => candidate.layoutId === expectedLayoutId,
        ) ?? null;
        const expectedBindings = collector.capture.bindings.filter(
          (binding) => binding.group === bindGroupEvent.index,
        );
        const entryBindings = bindGroup?.entries?.map((entry) => entry.binding) ?? [];
        const layoutBindings = bindGroupLayout?.entries?.map((entry) => entry.binding) ?? [];
        const expectedBindingIndices = expectedBindings.map((binding) => binding.binding);
        return {
          group: bindGroupEvent.index,
          setBindGroupSequence: bindGroupEvent.sequence,
          bindGroupId: bindGroupEvent.bindGroupId,
          bindGroupMatchCount: bindGroups.length,
          bindGroupCreateSequence: bindGroup?.sequence ?? null,
          bindGroupCapturePhase: bindGroup?.capturePhase ?? null,
          observedLayoutId: bindGroup?.layoutId ?? null,
          expectedLayoutId,
          layoutCreateSequence: bindGroupLayout?.sequence ?? null,
          dynamicOffsets: bindGroupEvent.dynamicOffsets,
          dynamicOffsetStart: bindGroupEvent.dynamicOffsetStart,
          dynamicOffsetLength: bindGroupEvent.dynamicOffsetLength,
          selectedDynamicOffsets: bindGroupEvent.selectedDynamicOffsets,
          entries: bindGroup?.entries ?? null,
          layoutEntries: bindGroupLayout?.entries ?? null,
          expectedBindings,
          entryBindings,
          layoutBindings,
          expectedBindingIndices,
          pass: bindGroups.length === 1
            && bindGroupLayout !== null
            && bindGroup.layoutId === expectedLayoutId
            && bindGroup.sequence < bindGroupEvent.sequence
            && bindGroupLayout.sequence < bindGroupEvent.sequence
            && bindGroupEvent.sequence < dispatch.event.sequence
            && bindGroupEvent.dynamicOffsets === null
            && bindGroupEvent.dynamicOffsetStart === null
            && bindGroupEvent.dynamicOffsetLength === null
            && bindGroupEvent.selectedDynamicOffsets === null
            && new Set(entryBindings).size === entryBindings.length
            && new Set(layoutBindings).size === layoutBindings.length
            && exactNumberArray([...entryBindings].sort((a, b) => a - b),
              [...expectedBindingIndices].sort((a, b) => a - b))
            && exactNumberArray([...layoutBindings].sort((a, b) => a - b),
              [...expectedBindingIndices].sort((a, b) => a - b)),
        };
      });
      const expectedGroupIndices = [...new Set(collector.capture.bindings.map(
        (binding) => binding.group,
      ))].sort((left, right) => left - right);
      const observedGroupIndices = dispatch.bindGroups.map(
        (event) => event.index,
      ).sort((left, right) => left - right);
      const bindGroupPartitionExact = exactNumberArray(
        observedGroupIndices,
        expectedGroupIndices,
      ) && exactNumberArray(
        observedGroupIndices,
        Array.from({ length: layout.bindGroupLayoutIds.length }, (_, index) => index),
      ) && bindGroupChains.every((chain) => chain.pass);
      const shaderBindingDeclarations = shaderDeclaredBindingCoordinates([
        { stage: 'compute', module },
      ]);
      const capturedBindingCoordinates = collector.capture.bindings.map((binding) => ({
        group: binding.group,
        binding: binding.binding,
      })).sort((left, right) => left.group - right.group || left.binding - right.binding);
      const declaredBindingCoordinates = shaderBindingDeclarations.coordinates.map(
        ({ group, binding }) => ({ group, binding }),
      );
      const shaderBindingTopologyExact = exactStructuredValue(
        declaredBindingCoordinates,
        capturedBindingCoordinates,
      );
      const shaderLayoutCoordinates = bindGroupLayouts.flatMap(
        (bindGroupLayout, group) => bindGroupLayout.entries.map(
          (entry) => ({ group, binding: entry.binding, entry }),
        ),
      );
      const shaderExpectedGroupIndices = [...new Set(
        shaderBindingDeclarations.coordinates.map((coordinate) => coordinate.group),
      )].sort((left, right) => left - right);
      const shaderExpectedContiguousGroups = Array.from(
        { length: shaderExpectedGroupIndices.length },
        (_, index) => index,
      );
      const shaderBindingSemantics = bindingLayoutSemanticsEvidence(
        shaderBindingDeclarations,
        shaderLayoutCoordinates,
      );
      const shaderLayoutBindingTopologyExact = shaderBindingTopologyExact
        && exactNumberArray(shaderExpectedGroupIndices, shaderExpectedContiguousGroups)
        && layout.bindGroupLayoutIds.length === shaderExpectedGroupIndices.length
        && bindGroupLayouts.length === shaderExpectedGroupIndices.length
        && exactNumberArray(
          shaderLayoutCoordinates.map(
            (coordinate) => `${coordinate.group}/${coordinate.binding}`,
          ).sort(),
          shaderBindingDeclarations.coordinates.map(
            (coordinate) => `${coordinate.group}/${coordinate.binding}`,
          ).sort(),
        )
        && shaderBindingSemantics.length
          === shaderBindingDeclarations.coordinates.length
        && shaderBindingSemantics.every((binding) => binding.pass);
      const boundBindings = collector.capture.bindings.map((binding) => {
        const bindGroupEvent = dispatch.bindGroups.find(
          (event) => event.index === binding.group,
        );
        const bindGroups = instrumentation.evidence.resources.filter(
          (resource) => resource.method === 'createBindGroup'
            && resource.resourceId === bindGroupEvent?.bindGroupId,
        );
        const bindGroup = bindGroups.length === 1 ? bindGroups[0] : null;
        const entries = bindGroup?.entries?.filter(
          (candidate) => candidate.binding === binding.binding,
        ) ?? [];
        const entry = entries.length === 1 ? entries[0] : null;
        const expectedBindGroupLayoutId = layout.bindGroupLayoutIds[binding.group] ?? null;
        const bindGroupLayoutMatches = instrumentation.evidence.bindGroupLayouts.filter(
          (candidate) => candidate.layoutId === expectedBindGroupLayoutId,
        );
        const bindGroupLayout = bindGroupLayoutMatches.length === 1
          ? bindGroupLayoutMatches[0]
          : null;
        const layoutEntries = bindGroupLayout?.entries?.filter(
          (candidate) => candidate.binding === binding.binding,
        ) ?? [];
        const layoutEntry = layoutEntries.length === 1 ? layoutEntries[0] : null;
        const expectedBufferType = binding.kind === 'uniform-buffer'
          ? 'uniform'
          : /read.?only/i.test(binding.access ?? '')
            ? 'read-only-storage'
            : 'storage';
        const attribute = binding.kind === 'storage-buffer'
          ? computeSemanticAttribute(strategy, lane, binding.semantic)
          : null;
        const expectedGpuBuffer = attribute === null
          ? null
          : renderer.backend.get(attribute)?.buffer ?? null;
        const expectedGpuBufferId = instrumentation.identify(expectedGpuBuffer, 'buffer');
        return {
          semantic: binding.semantic,
          kind: binding.kind,
          group: binding.group,
          binding: binding.binding,
          resourceAttributeId: binding.resourceId,
          expectedGpuBufferId,
          bindGroupId: bindGroupEvent?.bindGroupId ?? null,
          bindGroupSetSequence: bindGroupEvent?.sequence ?? null,
          bindGroupCreateSequence: bindGroup?.sequence ?? null,
          bindGroupMatchCount: bindGroups.length,
          bindGroupLayoutId: bindGroup?.layoutId ?? null,
          expectedBindGroupLayoutId,
          bindGroupLayoutMatchCount: bindGroupLayoutMatches.length,
          bindGroupEntryMatchCount: entries.length,
          layoutEntryMatchCount: layoutEntries.length,
          bindGroupEntry: entry,
          layoutEntry,
          expectedBufferType,
          observedGpuBufferId: entry?.bufferId ?? null,
          pass: bindGroupEvent !== undefined
            && bindGroups.length === 1
            && entries.length === 1
            && bindGroupLayoutMatches.length === 1
            && layoutEntries.length === 1
            && bindGroup.layoutId === expectedBindGroupLayoutId
            && (layoutEntry.buffer?.type ?? 'uniform') === expectedBufferType
            && layoutEntry.buffer.hasDynamicOffset === false
            && bindGroup.sequence < bindGroupEvent.sequence
            && bindGroupEvent.sequence < dispatch.event.sequence
            && (binding.kind !== 'storage-buffer'
              || (attribute !== null && entry.bufferId === expectedGpuBufferId)),
        };
      });
      const executionChronologyExact = module.sequence < pipeline.sequence
        && layout.sequence < pipeline.sequence
        && dispatch.setPipeline !== null
        && pipeline.sequence < dispatch.setPipeline.sequence
        && dispatch.setPipeline.sequence < dispatch.event.sequence
        && trace.sequence < dispatch.setPipeline.sequence
        && bindGroupChains.every((chain) => (
          chain.bindGroupCreateSequence < chain.setBindGroupSequence
            && chain.layoutCreateSequence < chain.setBindGroupSequence
            && chain.setBindGroupSequence < dispatch.event.sequence
        ));
      const record = {
        pass: module.code === collector.capture.computeShader
          && pipeline.computeEntryPoint === 'main'
          && (pipeline.immediateSize === null || pipeline.immediateSize === 0)
          && (pipeline.descriptor?.immediateSize === null
            || pipeline.descriptor?.immediateSize === 0)
          && layout.hasOwnImmediateSize === true
          && layout.immediateSize === 0
          && !/requires\s+immediate_address_space\s*;/.test(module.code)
          && !/var\s*<\s*immediate\s*>/.test(module.code)
          && JSON.stringify(dispatch.event.dispatch) === JSON.stringify(expectedDispatch)
          && computeImmediateEvents.length === 0
          && dynamicBindGroupEvents.length === 0
          && bindGroupPartitionExact
          && shaderLayoutBindingTopologyExact
          && executionChronologyExact
          && boundBindings.every((binding) => binding.pass),
        capturePhase,
        computePassEncoderId: trace.encoderId,
        commandEncoderId: trace.commandEncoderId,
        pipelineId: pipeline.pipelineId,
        pipelineCreationSequence: pipeline.sequence,
        pipelineCreationPhase: pipeline.capturePhase,
        pipelineLayoutId: pipeline.layoutId,
        pipelineLayoutBindGroupLayoutIds: layout.bindGroupLayoutIds,
        pipelineLayoutHasOwnImmediateSize: layout.hasOwnImmediateSize,
        pipelineLayoutImmediateSize: layout.immediateSize,
        pipelineLayoutImmediateSizeExact: layout.hasOwnImmediateSize === true
          && layout.immediateSize === 0,
        pipelineDescriptor: pipeline.descriptor,
        pipelineImmediateSize: pipeline.immediateSize,
        pipelineImmediateSizeExact: pipeline.immediateSize === null
          || pipeline.immediateSize === 0,
        pipelineLayout: layout,
        pipelineBindGroupLayouts: bindGroupLayouts,
        shaderModuleId: module.moduleId,
        shaderObservationNonce: module.observationNonce,
        shaderCreationOrdinal: module.creationOrdinal,
        shaderCreationSequence: module.sequence,
        shaderModuleCode: module.code,
        shaderModuleSha256: collector.rawSha256,
        shaderImmediateStateFree: !/requires\s+immediate_address_space\s*;/.test(module.code)
          && !/var\s*<\s*immediate\s*>/.test(module.code),
        entryPoint: pipeline.computeEntryPoint,
        setPipelineSequence: trace.events.find(
          (event) => event.method === 'setPipeline'
            && event.pipelineId === pipeline.pipelineId,
        )?.sequence ?? null,
        bindGroupPartitionExact,
        shaderBindingTopologyExact,
        shaderLayoutBindingTopologyExact,
        shaderBindingDeclarations,
        capturedBindingCoordinates,
        shaderLayoutCoordinates,
        shaderExpectedGroupIndices,
        shaderBindingSemantics,
        executionChronologyExact,
        expectedGroupIndices,
        observedGroupIndices,
        bindGroupChains,
        dispatchSequence: dispatch.event.sequence,
        dispatch: dispatch.event.dispatch,
        expectedDispatch,
        bindGroups: dispatch.bindGroups,
        boundBindings,
      };
      requireCondition(record.pass,
        `${lane}/${phase} actual compute execution chain failed.`, record);
      collector.boundExecutions ??= {};
      collector.boundExecutions[scenarioId] = record;
      phases[phase] = record;
    }
    laneEvidence[lane] = {
      pass: Object.values(phases).every((phase) => phase.pass)
        && computeImmediateEvents.length === 0
        && dynamicBindGroupEvents.length === 0,
      capturePhase,
      computePassEncoderId: trace.encoderId,
      setImmediatesCount: computeImmediateEvents.length,
      setImmediates: computeImmediateEvents.map((event) => structuredClone(event)),
      dynamicBindGroupCount: dynamicBindGroupEvents.length,
      dynamicBindGroups: dynamicBindGroupEvents.map((event) => structuredClone(event)),
      rawEvents: trace.events.map((event) => structuredClone(event)),
      phases,
    };
  }
  const pipelineIds = IMMEDIATE_AIF_PHASE0_LANES.flatMap(
    (lane) => ['reset', 'cull'].map((phase) => laneEvidence[lane].phases[phase].pipelineId),
  );
  const shaderModuleIds = IMMEDIATE_AIF_PHASE0_LANES.flatMap(
    (lane) => ['reset', 'cull'].map(
      (phase) => laneEvidence[lane].phases[phase].shaderModuleId,
    ),
  );
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-bound-compute-execution',
    pass: Object.values(laneEvidence).every((lane) => lane.pass)
      && new Set(pipelineIds).size === 6
      && new Set(shaderModuleIds).size === 6,
    scenarioId,
    exclusivePipelineCount: new Set(pipelineIds).size,
    exclusiveShaderModuleCount: new Set(shaderModuleIds).size,
    lanes: laneEvidence,
  };
}

async function productionShaderEvidence(runtime) {
  const { renderer, instrumentation, productionTarget, camera, strategy } = runtime;
  const lanes = {};
  const normalizationInputs = {};
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const context = strategy.getLaneValidationResources(lane);
    const bundle = findProductionBundle(
      renderer, instrumentation, productionTarget, camera, context,
    );
    const builder = bundle.renderObject.getNodeBuilderState();
    const pipeline = renderer.backend.get(bundle.renderObject.pipeline);
    const pipelineId = instrumentation.identify(pipeline.pipeline, 'render-pipeline');
    const boundPipelineRecords = instrumentation.evidence.renderPipelines.filter(
      (record) => record.pipelineId === pipelineId,
    );
    requireCondition(boundPipelineRecords.length === 1,
      `Lane ${lane} bound render pipeline descriptor is not unique.`, {
        pipelineId,
        recordCount: boundPipelineRecords.length,
      });
    const boundPipeline = boundPipelineRecords[0];
    const boundPipelineLayouts = instrumentation.evidence.pipelineLayouts.filter(
      (layout) => layout.layoutId === boundPipeline.layoutId,
    );
    const boundBindGroupLayouts = boundPipelineLayouts.length === 1
      ? boundPipelineLayouts[0].bindGroupLayoutIds.map((layoutId) => {
        const matches = instrumentation.evidence.bindGroupLayouts.filter(
          (layout) => layout.layoutId === layoutId,
        );
        requireCondition(matches.length === 1,
          `Lane ${lane} bind-group layout ${layoutId} is not unique.`, matches);
        return matches[0];
      })
      : [];
    const vertexModules = instrumentation.evidence.shaderModules.filter(
      (module) => module.moduleId === boundPipeline.vertexModuleId,
    );
    const fragmentModules = instrumentation.evidence.shaderModules.filter(
      (module) => module.moduleId === boundPipeline.fragmentModuleId,
    );
    requireCondition(vertexModules.length === 1
      && fragmentModules.length === 1
      && boundPipelineLayouts.length === 1
      && boundPipeline.descriptor !== null,
      `Lane ${lane} bound shader modules are not unique.`, {
        pipelineId,
        vertexModuleId: boundPipeline.vertexModuleId,
        fragmentModuleId: boundPipeline.fragmentModuleId,
        vertexModuleCount: vertexModules.length,
        fragmentModuleCount: fragmentModules.length,
      });
    const vertexModule = vertexModules[0];
    const fragmentModule = fragmentModules[0];
    const vertex = vertexModule.code;
    const fragment = fragmentModule.code;
    requireCondition(vertex === builder.vertexShader && fragment === builder.fragmentShader,
      `Lane ${lane} node-builder WGSL differs from the actual bound shader modules.`, {
        pipelineId,
        vertexModuleId: vertexModule.moduleId,
        fragmentModuleId: fragmentModule.moduleId,
      });
    const vertexInputs = runtimeVertexInputEvidence(renderer, bundle.renderObject);
    const storageBindings = runtimeStorageBindingEvidence(vertex, bundle.renderObject, {
      matrixAttribute: strategy.sharedResources.attributes.matrix,
      visibleIdsAttribute: strategy.sharedResources.attributes.visibleIds,
    });
    const visibleBinding = storageBindings.find((binding) => binding.semantic === 'visibleIds');
    const storagePattern = /((?:@(binding|group)\s*\(\s*\d+\s*\)\s*){2})var\s*<\s*storage\s*,\s*read\s*>\s*([A-Za-z_]\w*)\s*:/g;
    let visibleVariable = null;
    for (const match of vertex.matchAll(storagePattern)) {
      const coordinates = Object.fromEntries([...match[1].matchAll(
        /@(binding|group)\s*\(\s*(\d+)\s*\)/g,
      )].map((entry) => [entry[1], Number(entry[2])]));
      if (coordinates.group === visibleBinding?.group
        && coordinates.binding === visibleBinding?.binding) {
        visibleVariable = match[3];
      }
    }
    const escapedVisibleVariable = visibleVariable?.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const addressAccessPattern = escapedVisibleVariable == null
      ? null
      : new RegExp(`${escapedVisibleVariable}\\s*\\.\\s*value\\s*\\[([^\\]]+)\\]`, 'g');
    const addressExpressions = addressAccessPattern === null
      ? []
      : [...vertex.matchAll(addressAccessPattern)].map((match) => match[1].trim());
    const baseInput = vertexInputs.find(
      (input) => input.resourceId
        === strategy.laneStates.A.geometry.getAttribute('bucketBase')?.id,
    );
    const normalizeExpression = (value) => value.replace(/[\s()]/g, '');
    const expectedAddressExpression = lane === 'A'
      ? `${baseInput?.name}+instanceIndex`
      : lane === 'I'
        ? 'threeImmediateDrawBase+instanceIndex'
        : 'instanceIndex';
    const addressAssignment = {
      pass: visibleVariable !== null
        && addressExpressions.length === 1
        && normalizeExpression(addressExpressions[0]) === expectedAddressExpression,
      visibleStorageVariable: visibleVariable,
      observedExpressions: addressExpressions,
      expectedExpression: expectedAddressExpression,
    };
    const visibleIdsGpuBuffer = renderer.backend.get(
      strategy.sharedResources.attributes.visibleIds,
    )?.buffer ?? null;
    const semantic = {
      immediateRequirementCount:
        vertex.match(/requires\s+immediate_address_space\s*;/g)?.length ?? 0,
      immediateDeclarationCount:
        vertex.match(/var<immediate>\s+threeImmediateDrawBase\s*:\s*u32\s*;/g)?.length ?? 0,
      totalImmediateDeclarationCount:
        vertex.match(/var\s*<\s*immediate\s*>\s+[A-Za-z_]\w*\s*:/g)?.length ?? 0,
      immediateAddressUseCount:
        vertex.match(/threeImmediateDrawBase\s*\+\s*instanceIndex/g)?.length ?? 0,
      bucketBaseTokenCount: vertex.match(/\bbucketBase\b/g)?.length ?? 0,
      firstInstanceTokenCount: vertex.match(/\bfirstInstance\b/g)?.length ?? 0,
    };
    const expectedImmediate = lane === 'I';
    const semanticPass = expectedImmediate
      ? semantic.immediateRequirementCount === 1
        && semantic.immediateDeclarationCount === 1
        && semantic.totalImmediateDeclarationCount === 1
        && semantic.immediateAddressUseCount === 1
        && semantic.bucketBaseTokenCount === 0
      : semantic.immediateRequirementCount === 0
        && semantic.immediateDeclarationCount === 0
        && semantic.totalImmediateDeclarationCount === 0
        && semantic.immediateAddressUseCount === 0
        && (lane !== 'F' || semantic.bucketBaseTokenCount === 0);
    normalizationInputs[lane] = {
      vertexShader: vertex,
      fragmentShader: fragment,
      vertexInputs,
      storageBindings,
    };
    lanes[lane] = {
      pass: semanticPass
        && addressAssignment.pass
        && visibleBinding?.resourceId === strategy.sharedResources.attributes.visibleIds.id
        && pipeline.immediateSize === (lane === 'I' ? REQUIRED_IMMEDIATE_BYTES : 0),
      vertexShader: vertex,
      fragmentShader: fragment,
      vertexSha256: await sha256Text(vertex),
      fragmentSha256: await sha256Text(fragment),
      semantic,
      vertexInputs,
      storageBindings,
      visibleIdsBinding: {
        ...visibleBinding,
        generatedVariable: visibleVariable,
        attributeId: strategy.sharedResources.attributes.visibleIds.id,
        gpuBufferId: instrumentation.identify(visibleIdsGpuBuffer, 'buffer'),
      },
      addressAssignment,
      pipelineImmediateSize: pipeline.immediateSize ?? null,
      pipelineId,
      boundPipeline: {
        pipelineId,
        creationSequence: boundPipeline.sequence,
        creationPhase: boundPipeline.capturePhase,
        layoutId: boundPipeline.layoutId,
        vertexModuleId: boundPipeline.vertexModuleId,
        fragmentModuleId: boundPipeline.fragmentModuleId,
        descriptor: boundPipeline.descriptor,
        pipelineLayout: boundPipelineLayouts[0],
        bindGroupLayouts: boundBindGroupLayouts,
      },
      boundModules: {
        vertex: {
          moduleId: vertexModule.moduleId,
          observationNonce: vertexModule.observationNonce,
          creationOrdinal: vertexModule.creationOrdinal,
          creationSequence: vertexModule.sequence,
          creationPhase: vertexModule.capturePhase,
          code: vertex,
          sha256: await sha256Text(vertex),
          nodeBuilderExact: vertex === builder.vertexShader,
        },
        fragment: {
          moduleId: fragmentModule.moduleId,
          observationNonce: fragmentModule.observationNonce,
          creationOrdinal: fragmentModule.creationOrdinal,
          creationSequence: fragmentModule.sequence,
          creationPhase: fragmentModule.capturePhase,
          code: fragment,
          sha256: await sha256Text(fragment),
          nodeBuilderExact: fragment === builder.fragmentShader,
        },
      },
      bundleGpuId: bundle.bundleGpuId,
    };
  }
  const renderNormalization = await createImmediateAifRenderShaderEvidence(
    normalizationInputs,
  );
  const computeLanes = {};
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const capture = await collectLiveComputeLaneEvidence(
      renderer,
      strategy.sharedResources,
      strategy.laneStates[lane],
    );
    computeLanes[lane] = {
      ...capture,
      laneId: lane,
      addressMode: capture.laneId,
      phases: capture.phases,
    };
  }
  const computeExecutionScenarios = Object.fromEntries(['v99', 'v20'].map(
    (scenarioId) => [
      scenarioId,
      bindComputeExecutionEvidence(runtime, computeLanes, scenarioId),
    ],
  ));
  const computeExecution = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-bound-compute-execution-scenarios',
    pass: Object.values(computeExecutionScenarios).every((record) => record.pass),
    scenarioLoadSequence: ['v99', 'v20'],
    submissionCount: Object.values(computeExecutionScenarios).reduce(
      (count, scenario) => count + Object.keys(scenario.lanes).length,
      0,
    ),
    scenarios: computeExecutionScenarios,
  };
  const computePhases = {};
  for (const phase of ['reset', 'cull']) {
    const records = IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => computeLanes[lane].phases[phase],
    );
    const commandAccessAudits = records.map((record) => {
      const identifier = record.normalization.generatedVariableIdentifier;
      const escaped = identifier.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const source = record.capture.computeShader;
      const variableTokenCount = source.match(
        new RegExp(`\\b${escaped}\\b`, 'g'),
      )?.length ?? 0;
      const directInstanceCountAccesses = [...source.matchAll(new RegExp(
        `\\b${escaped}\\s*\\.\\s*value\\s*\\[[^\\]]+\\]`
          + '\\s*\\.\\s*instanceCount\\b',
        'g',
      ))].map((match) => match[0]);
      const expectedAtomicFunction = phase === 'reset' ? 'atomicStore' : 'atomicAdd';
      const atomicAccessCount = source.match(new RegExp(
        `\\b${expectedAtomicFunction}\\s*\\(\\s*&?\\s*${escaped}`
          + '\\s*\\.\\s*value\\s*\\[[^\\]]+\\]\\s*\\.\\s*instanceCount\\b',
        'g',
      ))?.length ?? 0;
      const audit = {
        pass: record.normalization.variableTokenCount === 2
          && variableTokenCount === 2
          && directInstanceCountAccesses.length === 1
          && atomicAccessCount === 1,
        generatedVariableIdentifier: identifier,
        declarationCount: 1,
        variableTokenCount,
        directInstanceCountAccessCount: directInstanceCountAccesses.length,
        directInstanceCountAccesses,
        expectedAtomicFunction,
        atomicAccessCount,
      };
      record.indirectCommandAccess = audit;
      record.pass = record.pass && audit.pass;
      return audit;
    });
    const normalizedShadersEqual = records.every(
      (record) => record.normalizedShader === records[0].normalizedShader,
    );
    const normalizedHashesEqual = records.every(
      (record) => record.normalizedSha256 === records[0].normalizedSha256,
    );
    const rawShadersPairwiseDifferent = new Set(
      records.map((record) => record.capture.computeShader),
    ).size === records.length;
    const generatedVariableIdentifiers = records.map(
      (record) => record.normalization.generatedVariableIdentifier,
    );
    const generatedVariablesPairwiseDistinct = new Set(
      generatedVariableIdentifiers,
    ).size === generatedVariableIdentifiers.length;
    const bindingShapes = records.map((record) => record.capture.bindings.map((binding) => {
      const { resourceId, ...shape } = binding;
      return shape;
    }));
    const bindingShapesEqual = bindingShapes.every(
      (shape) => JSON.stringify(shape) === JSON.stringify(bindingShapes[0]),
    );
    const storageBindingsBySemantic = Object.fromEntries(
      records[0].capture.bindings
        .filter((binding) => binding.kind === 'storage-buffer')
        .map((binding) => [binding.semantic, records.map((record) => (
          record.capture.bindings.find((candidate) => (
            candidate.kind === 'storage-buffer'
              && candidate.semantic === binding.semantic
          ))?.resourceId ?? null
        ))]),
    );
    const resourceIdentityExact = Object.entries(storageBindingsBySemantic).every(
      ([semantic, resourceIds]) => semantic === 'indirectCommands'
        ? new Set(resourceIds).size === IMMEDIATE_AIF_PHASE0_LANES.length
        : new Set(resourceIds).size === 1,
    );
    computePhases[phase] = {
      pass: records.every((record) => record.pass)
        && normalizedShadersEqual
        && normalizedHashesEqual
        && rawShadersPairwiseDifferent
        && generatedVariablesPairwiseDistinct
        && bindingShapesEqual
        && resourceIdentityExact
        && commandAccessAudits.every((audit) => audit.pass),
      normalizedShadersEqual,
      normalizedHashesEqual,
      commonNormalizedSha256: normalizedHashesEqual
        ? records[0].normalizedSha256
        : null,
      rawShadersPairwiseDifferent,
      generatedVariablesPairwiseDistinct,
      generatedVariableIdentifiers,
      bindingShapesEqual,
      resourceIdentityExact,
      storageBindingsBySemantic,
      commandAccessAudits,
    };
  }
  const computeNormalization = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-compute-shader-normalization',
    pass: Object.values(computePhases).every((phase) => phase.pass),
    phases: computePhases,
  };
  return {
    pass: IMMEDIATE_AIF_PHASE0_LANES.every((lane) => lanes[lane].pass)
      && renderNormalization.pass
      && IMMEDIATE_AIF_PHASE0_LANES.every((lane) => computeLanes[lane].pass)
      && computeNormalization.pass
      && computeExecution.pass,
    lanes,
    renderNormalization,
    computeLanes,
    computeNormalization,
    computeExecution,
  };
}

async function primeDiagnosticImmediateBundles(runtime, scheduleId) {
  const { strategy, addressDiagnostics, objectIdDiagnostics, camera, instrumentation } = runtime;
  objectIdDiagnostics.markImmediateForRerecord();
  const context = strategy.getLaneValidationResources('I');
  const basePhase = `phase0/diagnostic-rerecord/${strategy.activeScenarioId}/${scheduleId}/I`;
  try {
    instrumentation.setCapturePhase(`${basePhase}/address`);
    await primeDiagnosticLane({
      runtime,
      diagnostics: addressDiagnostics,
      diagnosticKind: 'address',
      lane: 'I',
      context,
      camera: addressDiagnostics.camera,
      capturePhase: `${basePhase}/address`,
    });
    instrumentation.setCapturePhase(`${basePhase}/object-id`);
    await primeDiagnosticLane({
      runtime,
      diagnostics: objectIdDiagnostics,
      diagnosticKind: 'object-id',
      camera,
      lane: 'I',
      context,
      capturePhase: `${basePhase}/object-id`,
    });
  } finally {
    instrumentation.setCapturePhase(null);
  }
}

async function transitionAndRerecord(runtime, scheduleId) {
  const { strategy, instrumentation } = runtime;
  let transition = strategy.transitionSchedule(scheduleId);
  requireCondition(transition.pass === true,
    `Schedule transition to ${scheduleId} failed.`, transition);
  runtime.addressDiagnostics.setSchedule(scheduleId);
  const scheduleInstallation = await realizeScheduleInstallation(
    runtime,
    `transition-${transition.fromScheduleId}-to-${scheduleId}`,
  );
  await primeDiagnosticImmediateBundles(runtime, scheduleId);
  let sourceDetachment = null;
  if (scheduleId === 'S1') {
    instrumentation.armImmediateBundleFinishHook((finish) => {
      const immediateSource = strategy.laneStates.I.indirectImmediateBases;
      const detachedSourceId = instrumentation.identify(immediateSource, 'typed-array');
      const detachedBackingBufferId = instrumentation.identify(
        immediateSource?.buffer,
        'array-buffer',
      );
      const expectedRecordedBases = IMMEDIATE_AIF_PHASE0_SCHEDULES.S1.sourceBaseByDraw;
      const setImmediatesSequences = finish.trace.events
        .filter((event) => event.method === 'setImmediates')
        .map((event) => event.sequence);
      requireCondition(finish.source === immediateSource
        && finish.sourceSnapshot?.pass === true
        && finish.sourceSnapshot.sourceId === detachedSourceId
        && finish.sourceSnapshot.backingBufferId === detachedBackingBufferId
        && finish.sourceSnapshot.callCount === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
        && finish.sourceSnapshot.recordedValues.every(
          (value, index) => value === expectedRecordedBases[index],
        )
        && setImmediatesSequences.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
        && setImmediatesSequences.every((sequence) => sequence < finish.finishSequence),
      'The finished S1 bundle is not bound to one exact immediate source.', finish);
      const detached = strategy.mutateAndDetachImmediateBaseSource(
        Uint32Array.from(IMMEDIATE_AIF_PHASE0_SCHEDULES.S2.sourceBaseByDraw),
      );
      const marker = instrumentation.mark('phase0-immediate-source-detached', {
        scenarioId: strategy.activeScenarioId,
        scheduleId,
        detachedSourceId,
        detachedBackingBufferId,
        cachedBundleGpuId: finish.bundleGpuId,
        bundleFinishSequence: finish.finishSequence,
      });
      sourceDetachment = {
        ...detached,
        recordedBases: Array.from(detached.recordedBases),
        mutatedBases: Array.from(detached.mutatedBases),
        detachedSourceId,
        detachedBackingBufferId,
        instrumentationSequence: marker.sequence,
        marker,
        cachedBundleGpuId: finish.bundleGpuId,
        bundleFinishSequence: finish.finishSequence,
        setImmediatesSequences,
        sourceSnapshot: finish.sourceSnapshot,
        finishBeforeDetach: finish.finishSequence < marker.sequence,
        allSetImmediatesBeforeFinish: setImmediatesSequences.every(
          (sequence) => sequence < finish.finishSequence,
        ),
        firstExecuteSequence: null,
        postDetachExecuteSequences: [],
      };
      return {
        pass: sourceDetachment.pass === true,
        detachedSourceId,
        detachedBackingBufferId,
        instrumentationSequence: marker.sequence,
        bundleFinishSequence: finish.finishSequence,
      };
    });
  }
  const rerecord = await strategy.rerecordImmediateBundle({
    renderLane: (context) => captureLaneCallback(runtime, context),
  });
  if (scheduleId === 'S1') {
    instrumentation.assertImmediateBundleFinishHookConsumed();
    requireCondition(sourceDetachment?.pass === true
      && sourceDetachment.detachedDuringImmediateRerecord === true
      && sourceDetachment.detachedByteLength === 0
      && sourceDetachment.detachedLength === 0
      && rerecord.callback?.bundle?.bundleGpuId === sourceDetachment.cachedBundleGpuId
      && rerecord.callback?.bundle?.sourceSnapshot?.sourceId
        === sourceDetachment.detachedSourceId
      && rerecord.callback?.bundle?.execution?.executeSequence
        > sourceDetachment.instrumentationSequence,
    'S1 finish-hook detachment did not precede the first bundle execution.', {
        sourceDetachment,
        rerecord,
      });
    sourceDetachment.firstExecuteSequence =
      rerecord.callback.bundle.execution.executeSequence;
  }
  requireCondition(rerecord.pass === true,
    `I bundle rerecord for ${scheduleId} failed.`, rerecord);
  if (scheduleId === 'S2') {
    const priorDetachment = runtime.immediateSourceDetachments.at(-1) ?? null;
    const replacementSource = strategy.laneStates.I.indirectImmediateBases;
    const replacementSourceId = instrumentation.identify(replacementSource, 'typed-array');
    const replacementBackingBufferId = instrumentation.identify(
      replacementSource?.buffer,
      'array-buffer',
    );
    const replacementSnapshot = rerecord.callback?.bundle?.sourceSnapshot ?? null;
    const replacementExact = priorDetachment !== null
      && transition.immediateSourceWasDetached === true
      && transition.immediateSourceReplaced === true
      && replacementSourceId !== priorDetachment.detachedSourceId
      && replacementBackingBufferId !== priorDetachment.detachedBackingBufferId
      && replacementSnapshot?.sourceId === replacementSourceId
      && replacementSnapshot?.backingBufferId === replacementBackingBufferId
      && replacementSnapshot?.recordedValues?.every(
        (value, index) => value === IMMEDIATE_AIF_PHASE0_SCHEDULES.S2.sourceBaseByDraw[index],
      );
    requireCondition(replacementExact,
      'S2 did not record from one distinct replacement immediate source.', {
        transition,
        priorDetachment,
        replacementSourceId,
        replacementBackingBufferId,
        replacementSnapshot,
      });
    transition = {
      ...transition,
      detachedSourceId: priorDetachment.detachedSourceId,
      detachedBackingBufferId: priorDetachment.detachedBackingBufferId,
      replacementSourceId,
      replacementBackingBufferId,
      replacementSourceExact: replacementExact,
      replacementSourceSnapshot: replacementSnapshot,
    };
  }
  return { transition, rerecord, sourceDetachment, scheduleInstallation };
}

function validateOrderedProductionChallenge(runtime, laneOrders, start, end, installation) {
  const { instrumentation, renderer, strategy } = runtime;
  const callbacks = laneOrders.results.flatMap(
    (order) => order.records.map((record) => record.callback),
  );
  const expectedLanes = IMMEDIATE_AIF_PHASE0_LANE_ORDERS.flatMap((order) => order);
  const observedLanes = callbacks.map((callback) => callback.lane);
  const productionPhases = callbacks.map((callback) => callback.phases.production);
  const productionPhaseSet = new Set(productionPhases);
  const renderPasses = instrumentation.evidence.renderPassEncoders.filter(
    (trace) => trace.sequence > start.sequence && trace.sequence < end.sequence,
  );
  const computePasses = instrumentation.evidence.computePassEncoders.filter(
    (trace) => trace.sequence > start.sequence && trace.sequence < end.sequence,
  );
  const diagnosticRenderPasses = renderPasses.filter(
    (trace) => /\/(?:address|object-id)$/.test(trace.capturePhase ?? ''),
  );
  const unexpectedRenderPasses = renderPasses.filter(
    (trace) => !productionPhaseSet.has(trace.capturePhase),
  );
  const phaseRenderPassCounts = Object.fromEntries(productionPhases.map((phase) => [
    phase,
    renderPasses.filter((trace) => trace.capturePhase === phase).length,
  ]));
  const expectedRenderPassPhaseOrder = productionPhases.flatMap((phase) => [phase, phase]);
  const observedRenderPassPhaseOrder = renderPasses.map((trace) => trace.capturePhase);
  const productionRenderPassTopology = productionPhases.map((phase, index) => {
    const traces = renderPasses.filter((trace) => trace.capturePhase === phase);
    const expectedBundleId = callbacks[index]?.bundle?.bundleGpuId ?? null;
    const clearTrace = traces[0] ?? null;
    const renderTrace = traces[1] ?? null;
    const drawLikeMethods = new Set([
      'draw',
      'drawIndexed',
      'drawIndirect',
      'drawIndexedIndirect',
    ]);
    const clearDrawLikeEvents = clearTrace?.events.filter(
      (event) => drawLikeMethods.has(event.method) || event.method === 'executeBundles',
    ) ?? [];
    const renderDrawLikeEvents = renderTrace?.events.filter(
      (event) => drawLikeMethods.has(event.method) || event.method === 'executeBundles',
    ) ?? [];
    const executeEvents = renderDrawLikeEvents.filter(
      (event) => event.method === 'executeBundles',
    );
    const pass = traces.length === 2
      && clearDrawLikeEvents.length === 0
      && clearTrace.events.at(-1)?.method === 'end'
      && renderDrawLikeEvents.length === 1
      && executeEvents.length === 1
      && exactNumberArray(executeEvents[0].bundleIds, [expectedBundleId])
      && renderTrace.events.at(-1)?.method === 'end';
    return {
      phase,
      pass,
      clearCommandEncoderId: clearTrace?.commandEncoderId ?? null,
      clearEncoderId: clearTrace?.encoderId ?? null,
      clearBeginSequence: clearTrace?.sequence ?? null,
      clearEndSequence: clearTrace?.events.at(-1)?.sequence ?? null,
      renderCommandEncoderId: renderTrace?.commandEncoderId ?? null,
      renderEncoderId: renderTrace?.encoderId ?? null,
      renderBeginSequence: renderTrace?.sequence ?? null,
      renderEndSequence: renderTrace?.events.at(-1)?.sequence ?? null,
      expectedBundleId,
      clearDrawLikeEvents,
      renderDrawLikeEvents,
    };
  });
  const productionRenderPassTopologyExact = productionRenderPassTopology.every(
    (record) => record.pass,
  );
  const productionExecuteSequences = callbacks.map(
    (callback) => callback.bundle.execution.executeSequence,
  );
  const executeSequencesStrictlyIncreasing = productionExecuteSequences.every(
    (sequence, index) => index === 0 || sequence > productionExecuteSequences[index - 1],
  );
  const creations = creationEventInventory(instrumentation).filter(
    (record) => record.sequence > start.sequence && record.sequence < end.sequence,
  );
  const transientReadbackBuffers = creations.filter((record) => (
    record.category === 'createBuffer' && record.resourceClass === 'readback-staging'
  ));
  const forbiddenCreations = creations.filter((record) => !(
    record.category === 'createBuffer' && record.resourceClass === 'readback-staging'
  ));
  const queueWriteBuffers = instrumentation.evidence.queueWriteBuffers.filter(
    (record) => record.sequence > start.sequence && record.sequence < end.sequence,
  );
  const queueWriteTextures = instrumentation.evidence.queueWriteTextures.filter(
    (record) => record.sequence > start.sequence && record.sequence < end.sequence,
  );
  const queueCopyExternalImagesToTexture =
    instrumentation.evidence.queueCopyExternalImagesToTexture.filter(
      (record) => record.sequence > start.sequence && record.sequence < end.sequence,
    );
  const textureDestructions = instrumentation.evidence.resourceDestructions.filter(
    (record) => record.resourceClass === 'persistent-texture'
      && record.sequence > start.sequence
      && record.sequence < end.sequence,
  );
  const frozenBufferWrites = queueWriteBuffers.filter(
    (record) => installation.frozenResourceIds.includes(record.bufferId),
  );
  const frozenBufferMaps = instrumentation.evidence.bufferMapEvents.filter(
    (record) => record.sequence > start.sequence
      && record.sequence < end.sequence
      && installation.frozenResourceIds.includes(record.resourceId),
  );
  const diagnosticPositionResourceIds = installation.diagnosticPositionRealization.records.map(
    (record) => record.gpuBufferId,
  );
  const installationImmutableResourceIds = [...new Set([
    ...installation.frozenResourceIds,
    ...diagnosticPositionResourceIds,
  ])];
  const installationBoundaryQueueWrites = instrumentation.evidence.queueWriteBuffers.filter(
    (record) => record.sequence > installation.completeSequence
      && record.sequence < start.sequence
      && installationImmutableResourceIds.includes(record.bufferId),
  );
  const installationBoundaryMaps = instrumentation.evidence.bufferMapEvents.filter(
    (record) => record.sequence > installation.completeSequence
      && record.sequence < start.sequence
      && installationImmutableResourceIds.includes(record.resourceId),
  );
  const immutableResources = [];
  const addImmutableAttribute = (semantic, attribute) => {
    const buffer = renderer.backend.get(attribute)?.buffer ?? null;
    const gpuBufferId = instrumentation.identify(buffer, 'buffer');
    requireCondition(typeof gpuBufferId === 'string',
      `Immutable challenge resource ${semantic} has no GPU buffer.`);
    immutableResources.push({ semantic, attributeId: attribute.id, gpuBufferId });
  };
  for (const [semantic, attribute] of Object.entries(strategy.sharedResources.attributes)) {
    addImmutableAttribute(`shared.${semantic}`, attribute);
  }
  addImmutableAttribute(
    'diagnostic.address.commonPosition',
    runtime.addressDiagnostics.commonPosition,
  );
  addImmutableAttribute(
    'diagnostic.address.featurePosition',
    runtime.addressDiagnostics.featurePosition,
  );
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const geometry = strategy.laneStates[lane].geometry;
    for (const [semantic, attribute] of Object.entries(geometry.attributes)) {
      addImmutableAttribute(`geometry.${lane}.${semantic}`, attribute);
    }
    addImmutableAttribute(`geometry.${lane}.index`, geometry.index);
    addImmutableAttribute(
      `geometry.${lane}.indirectCommands`,
      strategy.laneStates[lane].indirectAttribute,
    );
  }
  const immutableResourceIds = [...new Set(
    immutableResources.map((resource) => resource.gpuBufferId),
  )];
  const immutableTextureResources = [];
  const addImmutableTexture = (semantic, texture) => {
    if (texture === null) return;
    const gpuTexture = renderer.backend.get(texture)?.texture ?? null;
    const gpuTextureId = instrumentation.identify(gpuTexture, 'texture');
    requireCondition(typeof gpuTextureId === 'string',
      `Immutable challenge texture ${semantic} has no GPU texture.`);
    immutableTextureResources.push({
      semantic,
      textureUuid: texture.uuid,
      gpuTextureId,
    });
  };
  for (const [semantic, target] of [
    ['production', runtime.productionTarget],
    ['address', runtime.addressDiagnostics.target],
    ['objectId', runtime.objectIdDiagnostics.target],
  ]) {
    addImmutableTexture(`${semantic}.color`, target.texture);
    addImmutableTexture(`${semantic}.depth`, target.depthTexture);
  }
  const immutableTextureIds = [...new Set(
    immutableTextureResources.map((resource) => resource.gpuTextureId),
  )];
  const commandEncoderTransfers = instrumentation.evidence.commandEncoderTransfers.filter(
    (record) => record.sequence > start.sequence && record.sequence < end.sequence,
  );
  const destinationBufferId = (record) => {
    if (record.method === 'copyBufferToBuffer'
      || record.method === 'clearBuffer'
      || record.method === 'resolveQuerySet') {
      return record.destinationBufferId;
    }
    return record.method === 'copyTextureToBuffer'
      ? record.destination?.bufferId ?? null
      : null;
  };
  const destinationTextureId = (record) => (
    record.method === 'copyBufferToTexture' || record.method === 'copyTextureToTexture'
      ? record.destination?.textureId ?? null
      : null
  );
  const immutableBufferCommandWrites = commandEncoderTransfers.filter(
    (record) => immutableResourceIds.includes(destinationBufferId(record)),
  );
  const immutableTextureCommandWrites = commandEncoderTransfers.filter(
    (record) => immutableTextureIds.includes(destinationTextureId(record)),
  );
  const installationBoundaryCommandWrites =
    instrumentation.evidence.commandEncoderTransfers.filter(
      (record) => record.sequence > installation.completeSequence
        && record.sequence < start.sequence
        && installationImmutableResourceIds.includes(destinationBufferId(record)),
    );
  const prescribedInstallationWrites = instrumentation.evidence.queueWriteBuffers.filter(
    (record) => record.sequence > installation.startSequence
      && record.sequence < installation.completeSequence
      && installationImmutableResourceIds.includes(record.bufferId),
  );
  const installationFreezeBoundary = {
    pass: Number.isInteger(installation.completeSequence)
      && installation.completeSequence < start.sequence
      && installation.managerCalls.every(
        (record) => record.completeSequence < installation.queueCompleteSequence,
      )
      && installation.queueCompleteSequence < installation.completeSequence
      && prescribedInstallationWrites.every(
        (record) => record.sequence < installation.completeSequence,
      )
      && installationBoundaryQueueWrites.length === 0
      && installationBoundaryMaps.length === 0
      && installationBoundaryCommandWrites.length === 0,
    installationId: installation.installationId,
    installationStartSequence: installation.startSequence,
    installationQueueCompleteSequence: installation.queueCompleteSequence,
    installationCompleteSequence: installation.completeSequence,
    challengeStartSequence: start.sequence,
    frozenResourceIds: [...installation.frozenResourceIds],
    diagnosticPositionResourceIds,
    installationImmutableResourceIds,
    prescribedInstallationWrites,
    boundaryQueueWrites: installationBoundaryQueueWrites,
    boundaryMaps: installationBoundaryMaps,
    boundaryCommandWrites: installationBoundaryCommandWrites,
  };
  const stagingResourceIds = new Set(instrumentation.evidence.resources
    .filter((record) => record.resourceClass === 'readback-staging')
    .map((record) => record.resourceId));
  const productionTextureBySemantic = new Map(immutableTextureResources
    .filter((resource) => resource.semantic.startsWith('production.'))
    .map((resource) => [resource.semantic, resource]));
  const commandEncoders = instrumentation.evidence.commandEncoders.filter(
    (record) => record.sequence > start.sequence && record.sequence < end.sequence,
  );
  const queueSubmissions = instrumentation.evidence.queueSubmissions.filter(
    (record) => record.sequence > start.sequence && record.sequence < end.sequence,
  );
  const submittedCommandBufferIds = queueSubmissions.flatMap(
    (record) => record.commandBufferIds,
  );
  const encodedCommandBufferIds = commandEncoders.map((record) => record.commandBufferId);
  const commandSubmissionTopologyExact = commandEncoders.length === 72
    && queueSubmissions.length === 72
    && commandEncoders.every((record) => record.finishCallCount === 1
      && Number.isInteger(record.finishSequence)
      && typeof record.commandBufferId === 'string'
      && record.sequence < record.finishSequence
      && productionPhaseSet.has(record.capturePhase)
      && ((record.renderPassEncoderIds.length === 1
        && record.computePassEncoderIds.length === 0
        && record.transferSequences.length === 0)
        || (record.renderPassEncoderIds.length === 0
          && record.computePassEncoderIds.length === 0
          && record.transferSequences.length === 1)))
    && queueSubmissions.every((record) => record.commandBufferIds.length === 1
      && productionPhaseSet.has(record.capturePhase))
    && new Set(encodedCommandBufferIds).size === 72
    && new Set(submittedCommandBufferIds).size === 72
    && exactNumberArray(
      [...encodedCommandBufferIds].sort(),
      [...submittedCommandBufferIds].sort(),
    )
    && commandEncoders.every((record) => {
      const submission = queueSubmissions.find(
        (candidate) => candidate.commandBufferIds[0] === record.commandBufferId,
      );
      return submission?.sequence > record.finishSequence;
    });
  const expectedReadbackBufferSize = VIEWPORT.width * VIEWPORT.height * 4;
  const expectedReadbackBufferUsage = 0x0001 | 0x0008;
  const productionCommandTopology = productionPhases.map((phase, index) => {
    const renderTopology = productionRenderPassTopology[index];
    const phaseEncoders = commandEncoders.filter(
      (record) => record.capturePhase === phase,
    );
    const phaseSubmissions = queueSubmissions.filter(
      (record) => record.capturePhase === phase,
    );
    const phaseTransfers = commandEncoderTransfers.filter(
      (record) => record.capturePhase === phase,
    );
    const uniqueEncoder = (commandEncoderId) => {
      const matches = phaseEncoders.filter(
        (record) => record.commandEncoderId === commandEncoderId,
      );
      return matches.length === 1 ? matches[0] : null;
    };
    const uniqueSubmission = (commandBufferId) => {
      const matches = phaseSubmissions.filter((record) => (
        exactNumberArray(record.commandBufferIds, [commandBufferId])
      ));
      return matches.length === 1 ? matches[0] : null;
    };
    const renderLeg = (name, commandEncoderId, passEncoderId) => {
      const encoder = uniqueEncoder(commandEncoderId);
      const traceMatches = renderPasses.filter((trace) => (
        trace.capturePhase === phase && trace.encoderId === passEncoderId
      ));
      const trace = traceMatches.length === 1 ? traceMatches[0] : null;
      const submission = uniqueSubmission(encoder?.commandBufferId);
      const endEvents = trace?.events.filter((event) => event.method === 'end') ?? [];
      const executeEvents = trace?.events.filter(
        (event) => event.method === 'executeBundles',
      ) ?? [];
      const pass = encoder !== null
        && trace !== null
        && submission !== null
        && encoder.capturePhase === phase
        && exactNumberArray(encoder.renderPassEncoderIds, [passEncoderId])
        && encoder.computePassEncoderIds.length === 0
        && encoder.transferSequences.length === 0
        && encoder.finishCallCount === 1
        && encoder.sequence < trace.sequence
        && endEvents.length === 1
        && trace.events.at(-1)?.sequence === endEvents[0].sequence
        && endEvents[0].sequence < encoder.finishSequence
        && encoder.finishSequence < submission.sequence
        && (name === 'clear'
          ? executeEvents.length === 0
          : executeEvents.length === 1
            && executeEvents[0].sequence
              === callbacks[index]?.bundle?.execution?.executeSequence
            && exactNumberArray(
              executeEvents[0].bundleIds,
              [callbacks[index]?.bundle?.bundleGpuId],
            ));
      return {
        pass,
        commandEncoderId: encoder?.commandEncoderId ?? null,
        commandEncoderSequence: encoder?.sequence ?? null,
        renderPassEncoderId: trace?.encoderId ?? null,
        renderPassBeginSequence: trace?.sequence ?? null,
        renderPassEndSequence: endEvents.length === 1 ? endEvents[0].sequence : null,
        executeSequence: executeEvents.length === 1 ? executeEvents[0].sequence : null,
        commandBufferId: encoder?.commandBufferId ?? null,
        finishSequence: encoder?.finishSequence ?? null,
        submitSequence: submission?.sequence ?? null,
      };
    };
    const readbackLeg = (channel) => {
      const semantic = `production.${channel}`;
      const sourceTexture = productionTextureBySemantic.get(semantic) ?? null;
      const matches = phaseTransfers.filter((record) => (
        record.method === 'copyTextureToBuffer'
          && record.source?.textureId === sourceTexture?.gpuTextureId
      ));
      const transfer = matches.length === 1 ? matches[0] : null;
      const encoder = uniqueEncoder(transfer?.commandEncoderId);
      const submission = uniqueSubmission(encoder?.commandBufferId);
      const stagingMatches = transientReadbackBuffers.filter((record) => (
        record.resourceId === transfer?.destination?.bufferId
      ));
      const staging = stagingMatches.length === 1 ? stagingMatches[0] : null;
      const pass = sourceTexture !== null
        && matches.length === 1
        && encoder !== null
        && submission !== null
        && staging !== null
        && transfer.capturePhase === phase
        && transfer.source.mipLevel === 0
        && exactStructuredValue(transfer.source.origin, { x: 0, y: 0, z: 0 })
        && transfer.source.aspect === 'all'
        && transfer.destination.offset === 0
        && transfer.destination.bytesPerRow === VIEWPORT.width * 4
        && transfer.destination.rowsPerImage === null
        && exactStructuredValue(transfer.copySize, {
          width: VIEWPORT.width,
          height: VIEWPORT.height,
          depthOrArrayLayers: 1,
        })
        && staging.capturePhase === phase
        && staging.category === 'createBuffer'
        && staging.resourceClass === 'readback-staging'
        && staging.size === expectedReadbackBufferSize
        && staging.usage === expectedReadbackBufferUsage
        && staging.mappedAtCreation === false
        && stagingResourceIds.has(staging.resourceId)
        && encoder.capturePhase === phase
        && encoder.renderPassEncoderIds.length === 0
        && encoder.computePassEncoderIds.length === 0
        && exactNumberArray(encoder.transferSequences, [transfer.sequence])
        && encoder.finishCallCount === 1
        && staging.sequence < encoder.sequence
        && encoder.sequence < transfer.sequence
        && transfer.sequence < encoder.finishSequence
        && encoder.finishSequence < submission.sequence;
      return {
        pass,
        channel,
        sourceSemantic: semantic,
        sourceTextureId: transfer?.source?.textureId ?? null,
        sourceMipLevel: transfer?.source?.mipLevel ?? null,
        sourceOrigin: transfer?.source?.origin ?? null,
        sourceAspect: transfer?.source?.aspect ?? null,
        stagingBufferId: transfer?.destination?.bufferId ?? null,
        stagingCreationSequence: staging?.sequence ?? null,
        stagingSize: staging?.size ?? null,
        stagingUsage: staging?.usage ?? null,
        destinationOffset: transfer?.destination?.offset ?? null,
        destinationBytesPerRow: transfer?.destination?.bytesPerRow ?? null,
        destinationRowsPerImage: transfer?.destination?.rowsPerImage ?? null,
        copySize: transfer?.copySize ?? null,
        commandEncoderId: encoder?.commandEncoderId ?? null,
        commandEncoderSequence: encoder?.sequence ?? null,
        transferSequence: transfer?.sequence ?? null,
        commandBufferId: encoder?.commandBufferId ?? null,
        finishSequence: encoder?.finishSequence ?? null,
        submitSequence: submission?.sequence ?? null,
      };
    };
    const clear = renderLeg(
      'clear',
      renderTopology.clearCommandEncoderId,
      renderTopology.clearEncoderId,
    );
    const render = renderLeg(
      'render',
      renderTopology.renderCommandEncoderId,
      renderTopology.renderEncoderId,
    );
    const colorReadback = readbackLeg('color');
    const depthReadback = readbackLeg('depth');
    const chronologyExact = clear.submitSequence < render.commandEncoderSequence
      && render.submitSequence < colorReadback.stagingCreationSequence
      && colorReadback.submitSequence < depthReadback.stagingCreationSequence;
    const pass = renderTopology.pass
      && phaseEncoders.length === 4
      && phaseSubmissions.length === 4
      && phaseTransfers.length === 2
      && clear.pass
      && render.pass
      && colorReadback.pass
      && depthReadback.pass
      && colorReadback.sourceTextureId !== depthReadback.sourceTextureId
      && colorReadback.stagingBufferId !== depthReadback.stagingBufferId
      && chronologyExact;
    return {
      phase,
      pass,
      chronologyExact,
      clear,
      render,
      colorReadback,
      depthReadback,
    };
  });
  const commandReadbackTopologyExact = commandEncoderTransfers.length === 36
    && transientReadbackBuffers.length === 36
    && productionCommandTopology.every((record) => record.pass);
  const geometryRealization = runtime.mergedGeometryRealization;
  const geometryRealizationBeforeChallenge = geometryRealization?.pass === true
    && geometryRealization.queueCompleteSequence < start.sequence
    && geometryRealization.frozenResourceIds.every(
      (resourceId) => immutableResourceIds.includes(resourceId),
    );
  const sharedGpuCommitment = runtime.activeSharedGpuCommitment;
  const sharedGpuCommitmentBeforeChallenge = sharedGpuCommitment?.pass === true
    && sharedGpuCommitment.scenarioId === installation.scenarioId
    && sharedGpuCommitment.completeSequence < start.sequence
    && sharedGpuCommitment.records.every((record) => (
      immutableResources.some((resource) => (
        resource.semantic === `shared.${record.semantic}`
          && resource.attributeId === record.attributeId
          && resource.gpuBufferId === record.gpuBufferId
      ))
    ));
  const immutableBufferWrites = queueWriteBuffers.filter(
    (record) => immutableResourceIds.includes(record.bufferId),
  );
  const immutableBufferMaps = instrumentation.evidence.bufferMapEvents.filter(
    (record) => record.sequence > start.sequence
      && record.sequence < end.sequence
      && immutableResourceIds.includes(record.resourceId),
  );
  const pass = callbacks.length === 18
    && exactNumberArray(observedLanes, expectedLanes)
    && callbacks.every((callback) => callback.enrichmentState === 'production-only'
      && callback.pass === true
      && callback.scheduleInstallationId === installation.installationId)
    && renderPasses.length === 36
    && diagnosticRenderPasses.length === 0
    && unexpectedRenderPasses.length === 0
    && Object.values(phaseRenderPassCounts).every((count) => count === 2)
    && exactNumberArray(observedRenderPassPhaseOrder, expectedRenderPassPhaseOrder)
    && productionRenderPassTopologyExact
    && computePasses.length === 0
    && executeSequencesStrictlyIncreasing
    && forbiddenCreations.length === 0
    && queueWriteTextures.length === 0
    && queueCopyExternalImagesToTexture.length === 0
    && textureDestructions.length === 0
    && frozenBufferWrites.length === 0
    && frozenBufferMaps.length === 0
    && installationFreezeBoundary.pass
    && geometryRealizationBeforeChallenge
    && sharedGpuCommitmentBeforeChallenge
    && immutableBufferWrites.length === 0
    && immutableBufferMaps.length === 0
    && immutableBufferCommandWrites.length === 0
    && immutableTextureCommandWrites.length === 0
    && commandReadbackTopologyExact
    && commandSubmissionTopologyExact;
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-ordered-production-challenge',
    pass,
    installationId: installation.installationId,
    scenarioId: installation.scenarioId,
    scheduleId: installation.scheduleId,
    startSequence: start.sequence,
    endSequence: end.sequence,
    expectedLanes,
    observedLanes,
    callbackCount: callbacks.length,
    productionPhases,
    renderPassCount: renderPasses.length,
    phaseRenderPassCounts,
    expectedRenderPassPhaseOrder,
    observedRenderPassPhaseOrder,
    renderPassEncoderIds: renderPasses.map((trace) => trace.encoderId),
    productionRenderPassTopology,
    productionRenderPassTopologyExact,
    computePassCount: computePasses.length,
    computePasses,
    diagnosticRenderPassCount: diagnosticRenderPasses.length,
    unexpectedRenderPassCount: unexpectedRenderPasses.length,
    productionExecuteSequences,
    executeSequencesStrictlyIncreasing,
    creationEvents: creations,
    transientReadbackBufferCreationCount: transientReadbackBuffers.length,
    forbiddenCreationCount: forbiddenCreations.length,
    forbiddenCreations,
    queueWriteBufferCount: queueWriteBuffers.length,
    queueWriteBuffers,
    queueWriteTextureCount: queueWriteTextures.length,
    queueWriteTextures,
    queueCopyExternalImageToTextureCount: queueCopyExternalImagesToTexture.length,
    queueCopyExternalImagesToTexture,
    textureDestructionCount: textureDestructions.length,
    textureDestructions,
    frozenResourceIds: [...installation.frozenResourceIds],
    diagnosticPositionResourceIds,
    installationImmutableResourceIds,
    frozenBufferWriteCount: frozenBufferWrites.length,
    frozenBufferWrites,
    frozenBufferMapCount: frozenBufferMaps.length,
    frozenBufferMaps,
    installationFreezeBoundary,
    mergedGeometryRealizationQueueCompleteSequence:
      geometryRealization?.queueCompleteSequence ?? null,
    geometryRealizationBeforeChallenge,
    sharedGpuCommitmentCompleteSequence: sharedGpuCommitment?.completeSequence ?? null,
    sharedGpuCommitmentBeforeChallenge,
    immutableResources,
    immutableResourceIds,
    immutableBufferWriteCount: immutableBufferWrites.length,
    immutableBufferWrites,
    immutableBufferMapCount: immutableBufferMaps.length,
    immutableBufferMaps,
    immutableTextureResources,
    immutableTextureIds,
    commandEncoderTransferCount: commandEncoderTransfers.length,
    commandEncoderTransfers,
    immutableBufferCommandWriteCount: immutableBufferCommandWrites.length,
    immutableBufferCommandWrites,
    immutableTextureCommandWriteCount: immutableTextureCommandWrites.length,
    immutableTextureCommandWrites,
    expectedReadbackBufferSize,
    expectedReadbackBufferUsage,
    productionCommandTopology,
    productionCommandTopologyExact: productionCommandTopology.every(
      (record) => record.pass,
    ),
    commandReadbackTopologyExact,
    commandEncoderCount: commandEncoders.length,
    commandEncoders,
    queueSubmissionCount: queueSubmissions.length,
    queueSubmissions,
    commandSubmissionTopologyExact,
  };
}

async function runCoverageSnapshot(runtime, definition) {
  const { strategy } = runtime;
  let transition = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-schedule-transition',
    pass: true,
    changed: false,
    fromScheduleId: strategy.state.activeScheduleId,
    toScheduleId: definition.scheduleId,
  };
  let rerecord = null;
  let sourceDetachment = null;
  if (strategy.state.activeScheduleId !== definition.scheduleId) {
    ({ transition, rerecord, sourceDetachment } = await transitionAndRerecord(
      runtime,
      definition.scheduleId,
    ));
  }
  const installation = runtime.activeScheduleInstallation;
  requireCondition(installation?.scheduleId === definition.scheduleId,
    `${definition.label} has no matching schedule installation.`, installation);
  const persistentInventoryPreflight = capturePersistentResourceInventory(
    runtime,
    `${installation.installationId}/ordered-challenge-preflight`,
  );
  const challengeStart = runtime.instrumentation.mark(
    'phase0-ordered-production-challenge-start',
    {
      installationId: installation.installationId,
      scenarioId: strategy.activeScenarioId,
      scheduleId: definition.scheduleId,
      laneOrders: IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map((order) => [...order]),
    },
  );
  const laneOrders = await strategy.runAllLaneOrders({
    renderLane: (context) => captureProductionLaneCallback(runtime, context),
  });
  const challengeEnd = runtime.instrumentation.mark(
    'phase0-ordered-production-challenge-complete',
    {
      installationId: installation.installationId,
      scenarioId: strategy.activeScenarioId,
      scheduleId: definition.scheduleId,
    },
  );
  const persistentInventoryPostflight = capturePersistentResourceInventory(
    runtime,
    `${installation.installationId}/ordered-challenge-postflight`,
  );
  const persistentInventory = {
    pass: persistentInventoryPreflight.pass
      && persistentInventoryPostflight.pass
      && exactStructuredValue(
        persistentInventoryPreflight.records,
        persistentInventoryPostflight.records,
      ),
    preflight: persistentInventoryPreflight,
    postflight: persistentInventoryPostflight,
    exact: exactStructuredValue(
      persistentInventoryPreflight.records,
      persistentInventoryPostflight.records,
    ),
  };
  const productionChallenge = validateOrderedProductionChallenge(
    runtime,
    laneOrders,
    challengeStart,
    challengeEnd,
    installation,
  );
  productionChallenge.persistentInventory = persistentInventory;
  productionChallenge.pass = productionChallenge.pass && persistentInventory.pass;
  requireCondition(laneOrders.pass === true && productionChallenge.pass,
    `${definition.label} production-only ordering challenge failed.`, {
      laneOrders,
      productionChallenge,
    });
  for (const order of laneOrders.results) {
    for (const record of order.records) {
      await enrichLaneCallback(
        runtime,
        strategy.getLaneValidationResources(record.lane),
        record.callback,
      );
    }
  }
  const relational = snapshotRelationalEvidence(laneOrders);
  requireCondition(laneOrders.pass === true && relational.pass,
    `${definition.label} lane-order challenge failed.`, { laneOrders, relational });
  if (definition.scheduleId === 'S1') {
    const cachedExecutions = laneOrders.results.flatMap((result) => result.records)
      .filter((record) => record.lane === 'I')
      .map((record) => record.callback?.bundle)
      .filter(Boolean);
    const postDetachExecuteSequences = cachedExecutions.map(
      (bundle) => bundle.execution.executeSequence,
    );
    requireCondition(sourceDetachment !== null
      && cachedExecutions.length === IMMEDIATE_AIF_PHASE0_LANE_ORDERS.length
      && cachedExecutions.every((bundle) => (
        bundle.bundleGpuId === sourceDetachment.cachedBundleGpuId
          && bundle.sourceSnapshot?.sourceId === sourceDetachment.detachedSourceId
          && bundle.execution?.executeSequence > sourceDetachment.instrumentationSequence
      )),
    'Cached S1 order coverage did not execute strictly after source detachment.', {
        sourceDetachment,
        cachedExecutions,
      });
    sourceDetachment.postDetachExecuteSequences = postDetachExecuteSequences;
    sourceDetachment.temporalOrderExact = sourceDetachment.allSetImmediatesBeforeFinish
      && sourceDetachment.finishBeforeDetach
      && sourceDetachment.firstExecuteSequence > sourceDetachment.instrumentationSequence
      && postDetachExecuteSequences.every(
        (sequence) => sequence > sourceDetachment.instrumentationSequence,
      );
    runtime.immediateSourceDetachments.push(sourceDetachment);
  }
  installation.orderedChallenge = productionChallenge;
  const commonState = await finishScheduleInstallation(
    runtime,
    installation,
    `${installation.installationId}/postflight`,
  );
  return {
    label: definition.label,
    scheduleId: definition.scheduleId,
    pass: true,
    transition,
    rerecord,
    sourceDetachment,
    scheduleInstallationId: installation.installationId,
    productionChallenge,
    commonState,
    laneOrders,
    relational,
  };
}

function selectLaneForAudit(strategy, lane) {
  strategy.selectActiveLane(lane);
}

function laneCaptureRelationalEvidence(callbacks) {
  const syntheticOrders = {
    results: [{ records: callbacks.map((callback) => ({ callback })) }],
  };
  const fields = [
    ['color', (entry) => entry.output.color.sha256],
    ['depth', (entry) => entry.output.depth.sha256],
    ['objectId', (entry) => entry.output.objectId.sha256],
    ['address', (entry) => entry.address.sha256],
  ];
  const values = syntheticOrders.results.flatMap(
    (order) => order.records.map((record) => record.callback),
  );
  const hashes = Object.fromEntries(fields.map(([name, select]) => [
    name,
    [...new Set(values.map(select))],
  ]));
  return {
    pass: callbacks.length === 3
      && callbacks.every((callback) => callback.pass === true)
      && Object.values(hashes).every((entries) => entries.length === 1),
    callbackCount: callbacks.length,
    allCallbacksPass: callbacks.every((callback) => callback.pass === true),
    uniqueHashes: hashes,
  };
}

async function captureCanonicalAudit(runtime, label, transition, rerecord) {
  const laneCaptures = [];
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    selectLaneForAudit(runtime.strategy, lane);
    laneCaptures.push(await captureLaneCallback(
      runtime,
      runtime.strategy.getLaneValidationResources(lane),
    ));
  }
  const relational = laneCaptureRelationalEvidence(laneCaptures);
  requireCondition(relational.pass, `${label} canonical audit failed.`, {
    laneCaptures,
    relational,
  });
  return {
    label,
    scheduleId: 'canonical',
    pass: true,
    transition,
    rerecord,
    laneCaptures,
    relational,
  };
}

function validateScenarioRelations(snapshots) {
  const byLabel = Object.fromEntries(snapshots.map((snapshot) => [snapshot.label, snapshot]));
  const hashes = (snapshot) => snapshot.relational.uniqueHashes;
  const canonical = hashes(byLabel['canonical-preflight']);
  const restored = hashes(byLabel['canonical-restored']);
  const postflight = hashes(byLabel['canonical-postflight']);
  const s1 = hashes(byLabel.S1);
  const s2 = hashes(byLabel.S2);
  const canonicalRestored = ['color', 'depth', 'objectId', 'address'].every(
    (name) => canonical[name][0] === restored[name][0]
      && canonical[name][0] === postflight[name][0],
  );
  const sentinelDistinct = s1.address[0] !== canonical.address[0]
    && s2.address[0] !== canonical.address[0]
    && s1.address[0] !== s2.address[0];
  return {
    pass: canonicalRestored && sentinelDistinct,
    canonicalRestored,
    sentinelDistinct,
    hashes: { canonical, S1: s1, S2: s2, restored, postflight },
  };
}

async function runScenario(runtime, scenarioId, validationOrder) {
  const { strategy } = runtime;
  const loadBoundaryBefore = scenarioId === 'v20'
    ? await scenarioLoadBoundaryState(runtime, 'v99-to-v20/before-load')
    : null;
  const load = strategy.loadScenario(scenarioId);
  const loadBoundaryAfter = scenarioId === 'v20'
    ? await scenarioLoadBoundaryState(runtime, 'v99-to-v20/after-load')
    : null;
  const scenarioLoadBoundary = scenarioLoadBoundaryDelta(
    loadBoundaryBefore,
    loadBoundaryAfter,
  );
  if (scenarioLoadBoundary !== null) {
    requireCondition(scenarioLoadBoundary.pass
      && scenarioLoadBoundary.fromScenarioId === 'v99'
      && scenarioLoadBoundary.toScenarioId === 'v20',
    'The v99-to-v20 scenario load boundary is not exact.', scenarioLoadBoundary);
  }
  const matrixGpuRealization = await realizeScenarioMatrixAttribute(runtime, scenarioId);
  strategy.beginLiveValidation(validationOrder);
  const liveValidation = [];
  for (const lane of validationOrder) {
    runtime.instrumentation.setCapturePhase(`phase0/live/${scenarioId}/${lane}`);
    try {
      const validation = await strategy.submitAndValidateLane(lane);
      requireCondition(validation.pass === true,
        `${scenarioId} live validation failed for lane ${lane}.`, validation);
      liveValidation.push(compactLiveValidation(validation));
    } finally {
      runtime.instrumentation.setCapturePhase(null);
    }
  }
  const freeze = strategy.freezeValidatedVisibleIds();
  requireCondition(freeze.pass === true,
    `${scenarioId} survivor freeze failed.`, freeze);
  const gpuResourceCommitments = await captureSharedGpuResourceCommitments(
    runtime,
    scenarioId,
  );

  if (runtime.addressDiagnostics === null) {
    runtime.addressDiagnostics = createAddressDiagnostics(
      strategy,
      runtime.sourceGeometries,
      runtime.camera,
    );
    runtime.objectIdDiagnostics = createObjectIdDiagnostics(
      strategy,
      runtime.scenarios.v99,
    );
  }
  runtime.addressDiagnostics.setSchedule('canonical');

  let prime;
  if (scenarioId === 'v99') {
    prime = await strategy.primeBundles({
      order: ['A', 'F', 'I'],
      renderLane: async (context) => {
        const primePhase = `phase0/prime/${context.lane}`;
        runtime.instrumentation.setCapturePhase(primePhase);
        try {
          const output = await captureProductionOutput({
            renderer: runtime.renderer,
            scene: runtime.productionScene,
            camera: runtime.camera,
            target: runtime.productionTarget,
          });
          await registerProductionOutputWitnesses(
            runtime,
            output,
            context,
            primePhase,
            'bundle-prime',
          );
          const bundle = await captureBundleEvidence(
            runtime.renderer,
            runtime.instrumentation,
            runtime.productionTarget,
            runtime.camera,
            context,
            strategy,
            primePhase,
          );
          return {
            pass: bundle.pass && output.color.contentPass && output.depth.contentPass,
            lane: context.lane,
            output,
            bundle,
          };
        } finally {
          runtime.instrumentation.setCapturePhase(null);
        }
      },
    });
    requireCondition(prime.pass === true,
      'Production bundle priming failed.', prime);
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const context = strategy.getLaneValidationResources(lane);
      const addressPhase = `phase0/diagnostic-prime/${lane}/address`;
      runtime.instrumentation.setCapturePhase(addressPhase);
      await primeDiagnosticLane({
        runtime,
        diagnostics: runtime.addressDiagnostics,
        diagnosticKind: 'address',
        lane,
        context,
        camera: runtime.addressDiagnostics.camera,
        capturePhase: addressPhase,
      });
      const objectIdPhase = `phase0/diagnostic-prime/${lane}/object-id`;
      runtime.instrumentation.setCapturePhase(objectIdPhase);
      await primeDiagnosticLane({
        runtime,
        diagnostics: runtime.objectIdDiagnostics,
        diagnosticKind: 'object-id',
        camera: runtime.camera,
        lane,
        context,
        capturePhase: objectIdPhase,
      });
    }
    runtime.instrumentation.setCapturePhase(null);
  } else {
    prime = {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-bundle-prime-reuse',
      pass: strategy.state.phase === 'ready',
      reusedFromScenarioId: 'v99',
      bundleRecordCounts: strategy.diagnostics().bundleRecordCounts,
    };
  }

  if (runtime.mergedGeometryRealization === null) {
    await realizeMergedGeometryForFreeze(runtime);
  }
  strategy.selectActiveLane('A');
  const initialScheduleInstallation = await realizeScheduleInstallation(
    runtime,
    'scenario-load-canonical-after-primes',
  );

  const persistentResourcePreflight = capturePersistentResourceInventory(
    runtime,
    `${scenarioId}/pre-ordered-challenges`,
  );
  requireCondition(persistentResourcePreflight.pass,
    `${scenarioId} pre-challenge persistent resource inventory is malformed.`,
    persistentResourcePreflight);

  const resourcesPreflight = await resourceState(runtime, `${scenarioId}/preflight`);
  const snapshots = [];
  snapshots.push(await runCoverageSnapshot(runtime, IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN[0]));
  snapshots.push(await runCoverageSnapshot(runtime, IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN[1]));
  snapshots.push(await runCoverageSnapshot(runtime, IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN[2]));
  const restoredTransition = await transitionAndRerecord(runtime, 'canonical');
  const restoredSnapshot = await captureCanonicalAudit(
    runtime,
    'canonical-restored',
    restoredTransition.transition,
    restoredTransition.rerecord,
  );
  restoredSnapshot.scheduleInstallationId =
    restoredTransition.scheduleInstallation.installationId;
  snapshots.push(restoredSnapshot);
  const resourcesRestored = await resourceState(runtime, `${scenarioId}/restored`);
  const postflightSnapshot = await captureCanonicalAudit(
    runtime,
    'canonical-postflight',
    {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-schedule-transition',
      pass: true,
      changed: false,
      fromScheduleId: 'canonical',
      toScheduleId: 'canonical',
    },
    null,
  );
  postflightSnapshot.scheduleInstallationId =
    restoredTransition.scheduleInstallation.installationId;
  snapshots.push(postflightSnapshot);
  const restoredCommonState = await finishScheduleInstallation(
    runtime,
    restoredTransition.scheduleInstallation,
    `${restoredTransition.scheduleInstallation.installationId}/postflight`,
  );
  restoredSnapshot.commonState = restoredCommonState;
  postflightSnapshot.commonState = restoredCommonState;
  const relations = validateScenarioRelations(snapshots);
  requireCondition(relations.pass, `${scenarioId} schedule relations failed.`, relations);
  const resourcesPostflight = await resourceState(runtime, `${scenarioId}/postflight`);
  const resourcesStable = stableResourceIdentity(resourcesRestored, resourcesPostflight);
  const transitionDeltas = prescribedTransitionDeltas(
    resourcesPreflight,
    resourcesRestored,
  );
  requireCondition(resourcesStable,
    `${scenarioId} resources changed between preflight and postflight.`, {
      resourcesRestored,
      resourcesPostflight,
    });
  requireCondition(transitionDeltas.pass,
    `${scenarioId} transition version/count deltas are not exact.`, transitionDeltas);
  const visibleIdsAttribute = strategy.sharedResources.attributes.visibleIds;
  const visibleIdsCommitment = gpuResourceCommitments.records.find(
    (record) => record.semantic === 'visibleIds',
  );
  requireCondition(visibleIdsCommitment?.exact === true,
    `${scenarioId} visible-ID GPU commitment is absent.`, gpuResourceCommitments);
  const persistentResourcePostflight = capturePersistentResourceInventory(
    runtime,
    `${scenarioId}/post-ordered-challenges`,
  );
  const prescribedPersistentDelta = scenarioPersistentResourceDelta(
    scenarioId,
    persistentResourcePreflight,
    persistentResourcePostflight,
  );
  const persistentResourceFreeze = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-persistent-resource-freeze-comparison',
    pass: persistentResourcePreflight.pass
      && persistentResourcePostflight.pass
      && prescribedPersistentDelta.pass,
    scenarioId,
    preflight: persistentResourcePreflight,
    postflight: persistentResourcePostflight,
    prescribedPersistentDelta,
  };
  requireCondition(persistentResourceFreeze.pass,
    `${scenarioId} persistent resources changed across ordered challenges.`,
    persistentResourceFreeze);
  runtime.scenarioResourceFreezes.push(persistentResourceFreeze);
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-scenario-result',
    pass: true,
    scenarioId,
    validationOrder: [...validationOrder],
    load,
    matrixGpuRealization,
    scenarioLoadBoundary,
    liveValidation,
    freeze: {
      ...freeze,
      visibleIds: Array.from(freeze.visibleIds),
      visibleIdsSha256: visibleIdsCommitment.gpuSha256,
      visibleIdsCpuSha256: visibleIdsCommitment.cpuSha256,
      visibleIdsAttributeId: visibleIdsAttribute.id,
      visibleIdsAttributeVersion: visibleIdsAttribute.version,
      visibleIdsGpuBufferId: visibleIdsCommitment.gpuBufferId,
      gpuResourceCommitmentCapturePhase: gpuResourceCommitments.capturePhase,
      gpuResourceCommitmentCompleteSequence: gpuResourceCommitments.completeSequence,
    },
    gpuResourceCommitments,
    prime,
    persistentResourceFreeze,
    initialScheduleInstallationId: initialScheduleInstallation.installationId,
    scheduleInstallations: runtime.scheduleInstallations.filter(
      (installation) => installation.scenarioId === scenarioId,
    ),
    snapshots,
    relations,
    resources: {
      pass: resourcesStable,
      preflight: resourcesPreflight,
      restored: resourcesRestored,
      postflight: resourcesPostflight,
      transitionDeltas,
    },
  };
}

async function executePhase0(state) {
  const configuration = validateImmediateAifPhase0TargetConfiguration(
    RUNNER_INJECTED_TARGET_CONFIGURATION,
  );
  state.targetConfiguration = structuredClone(configuration);
  if (!globalThis.isSecureContext) unsupported('A secure context is required.');
  if (typeof navigator?.gpu !== 'object') unsupported('navigator.gpu is unavailable.');
  const wgsl = sortedFeatures(navigator.gpu.wgslLanguageFeatures);
  if (!wgsl.includes(FEATURE_IMMEDIATE_ADDRESS_SPACE)) {
    unsupported(`WGSL language feature ${FEATURE_IMMEDIATE_ADDRESS_SPACE} is unavailable.`);
  }
  const signature = runtimeSignatureEvidence();
  requireCondition(signature.revision === configuration.expectedRevision,
    `Runtime Three.js revision is not ${configuration.expectedRevision}.`, signature);
  requireCondition(signature.setIndirectHasImmediateBasesParameter
    && signature.setIndirectStoresImmediateBases
    && signature.backendHasSetImmediates
    && signature.backendHasIndexedIndirectAdjacencyMarker,
  'The served Three.js runtime lacks the immediate-data overlay.', signature);

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
    forceFallbackAdapter: false,
  });
  if (adapter === null) unsupported('WebGPU adapter request returned null.');
  const adapterInfo = await adapterInformation(adapter);
  if (adapterInfo?.isFallbackAdapter !== false) {
    unsupported('A known non-fallback adapter is required.', adapterInfo);
  }
  const adapterFeatures = sortedFeatures(adapter.features);
  if (!adapterFeatures.includes(FEATURE_INDIRECT_FIRST_INSTANCE)) {
    unsupported(`${FEATURE_INDIRECT_FIRST_INSTANCE} is unavailable on the adapter.`);
  }
  const adapterMaxImmediateSize = Number(adapter.limits?.maxImmediateSize);
  if (!Number.isFinite(adapterMaxImmediateSize)
    || adapterMaxImmediateSize < REQUIRED_IMMEDIATE_BYTES) {
    unsupported('Adapter maxImmediateSize is below four bytes.', adapterMaxImmediateSize);
  }
  const requestedDeviceFeatures = [FEATURE_INDIRECT_FIRST_INSTANCE];
  let device;
  try {
    device = await adapter.requestDevice({
      requiredFeatures: requestedDeviceFeatures,
      requiredLimits: { maxImmediateSize: REQUIRED_IMMEDIATE_BYTES },
    });
  } catch (error) {
    error.phase0Status = 'unsupported';
    throw error;
  }
  state.device = device;
  state.adapterInfo = adapterInfo;
  state.intentionalDeviceDestroy = false;
  state.errorObservationScope = {
    start: 'after-production-device-creation-before-three-renderer-construction',
    includesRendererInitialization: true,
    includesAllComputeBundleRenderAndReadbackOperations: true,
    excludesAdapterAndDeviceRequest: true,
  };
  device.addEventListener('uncapturederror', (event) => {
    event.preventDefault();
    state.gpuErrors.uncaptured.push(serializeGpuError(event.error));
  });
  state.deviceLostPromise = device.lost.then((information) => {
    const record = {
      reason: String(information?.reason ?? 'unknown').slice(0, 256),
      message: String(information?.message ?? '').slice(0, 4_096),
    };
    state.gpuErrors.finalDeviceLoss = record;
    if (!state.intentionalDeviceDestroy) {
      state.gpuErrors.unexpectedDeviceLosses.push(record);
    }
    return record;
  });
  for (const filter of IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS) {
    device.pushErrorScope(filter);
    state.pendingErrorScopeFilters.push(filter);
  }
  const instrumentation = installPhase0DeviceInstrumentation(
    device,
    configuration.observationNonce,
  );
  state.instrumentation = instrumentation;

  let renderer;
  instrumentation.setCapturePhase('phase0/renderer-init');
  try {
    renderer = new WebGPURenderer({
      device,
      antialias: false,
      forceWebGL: false,
      powerPreference: 'high-performance',
      reversedDepthBuffer: true,
      trackTimestamp: false,
    });
    state.renderer = renderer;
    renderer.setPixelRatio(VIEWPORT.devicePixelRatio);
    renderer.setSize(VIEWPORT.width, VIEWPORT.height, false);
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.setClearColor(0x030711, 1);
    renderer.sortObjects = false;
    document.getElementById('canvas-host')?.appendChild(renderer.domElement);
    await renderer.init();
    renderer.backend.trackTimestamp = false;
  } finally {
    instrumentation.setCapturePhase(null);
  }
  requireCondition(renderer.backend?.isWebGPUBackend === true,
    'Three.js did not initialize its WebGPU backend.');
  requireCondition(renderer.reversedDepthBuffer === true,
    'The Phase 0 renderer did not retain reversed depth.');

  const deviceFeatures = sortedFeatures(device.features);
  state.capabilities = {
    navigatorGpu: true,
    wgslLanguageFeatures: wgsl,
    wgslImmediateAddressSpace: true,
    adapterMaxImmediateSize,
    deviceMaxImmediateSize: Number(device.limits?.maxImmediateSize),
    renderPassSetImmediates: false,
    renderBundleSetImmediates: false,
    adapterFeatures,
    requestedDeviceFeatures,
    deviceFeatures,
    indirectFirstInstanceAdapter: adapterFeatures.includes(FEATURE_INDIRECT_FIRST_INSTANCE),
    indirectFirstInstanceDevice: deviceFeatures.includes(FEATURE_INDIRECT_FIRST_INSTANCE),
    rawExpectation: configuration.rawExpectation ?? null,
  };
  requireCondition(state.capabilities.deviceMaxImmediateSize >= REQUIRED_IMMEDIATE_BYTES
    && state.capabilities.indirectFirstInstanceDevice,
  'Production device features or immediate limit are incomplete.', state.capabilities);

  const camera = createCamera(renderer);
  const sourceGeometries = createIndexedGeometryFixtures(
    IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount,
    IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier,
  );
  state.sourceGeometries = sourceGeometries;
  const strategy = buildImmediateAifPhase0Strategy({
    renderer,
    camera,
    sourceGeometries,
  });
  state.strategy = strategy;
  const productionScene = createProductionScene(strategy);
  const productionTarget = createProductionTarget();
  state.productionScene = productionScene;
  state.productionTarget = productionTarget;
  const runtime = {
    renderer,
    camera,
    sourceGeometries,
    strategy,
    scenarios: strategy.scenarios,
    productionScene,
    productionTarget,
    instrumentation,
    addressDiagnostics: null,
    objectIdDiagnostics: null,
    immediateSourceSnapshot: null,
    immediateSourceDetachments: [],
    addressWitnesses: new Map(),
    addressWitnessRaw: new Map(),
    outputWitnesses: new Map(),
    outputWitnessRaw: new Map(),
    outputWitnessObservations: [],
    outputWitnessObservationIds: new Set(),
    gpuByteWitnesses: new Map(),
    gpuByteWitnessRaw: new Map(),
    gpuByteWitnessReferences: [],
    addressPositionWitnesses: new Map(),
    addressPositionWitnessRaw: new Map(),
    scheduleInstallations: [],
    activeScheduleInstallation: null,
    scenarioResourceFreezes: [],
    mergedGeometryRealization: null,
    activeSharedGpuCommitment: null,
    scenarioMatrixRealizations: [],
    captureSerial: 1,
  };
  state.runtime = runtime;

  const scenarios = [];
  scenarios.push(await runScenario(runtime, 'v99', ['I', 'F', 'A']));
  scenarios.push(await runScenario(runtime, 'v20', ['A', 'F', 'I']));
  const shaderEvidence = await productionShaderEvidence(runtime);
  requireCondition(shaderEvidence.pass, 'Production shader/pipeline semantic gates failed.', shaderEvidence);

  await device.queue.onSubmittedWorkDone();
  await Promise.resolve();
  state.capabilities.renderPassSetImmediates = instrumentation.evidence.renderPassEncoders
    .some((trace) => trace.callableSetImmediates);
  state.capabilities.renderBundleSetImmediates = instrumentation.evidence.renderBundleEncoders
    .some((trace) => trace.callableSetImmediates);
  requireCondition(state.capabilities.renderPassSetImmediates
    && state.capabilities.renderBundleSetImmediates,
  'Actual render encoders did not expose setImmediates.', state.capabilities);

  const shaderModules = await instrumentation.finalizeShaderEvidence();
  const shaderObservationIdentityExact = shaderModules.length > 0
    && shaderModules.every((module, index) => (
      module.observationNonce === configuration.observationNonce
        && module.creationOrdinal === index + 1
    ));
  requireCondition(shaderObservationIdentityExact,
    'Shader-module nonce/creation ordinals are incomplete.', shaderModules);
  const shaderCompilationErrors = shaderModules.flatMap((module) => (
    module.compilationMessages
      .filter((message) => message.type === 'error')
      .map((message) => ({ moduleId: module.moduleId, ...message }))
  ));
  requireCondition(shaderCompilationErrors.length === 0,
    'Shader compilation produced errors.', shaderCompilationErrors);
  const computeModules = shaderModules.filter((module) => /@compute\b/.test(module.code ?? ''));
  const computePipelineLayoutBindings = instrumentation.evidence.computePipelines.map(
    (pipeline) => {
      const matches = instrumentation.evidence.pipelineLayouts.filter(
        (layout) => layout.layoutId === pipeline.layoutId,
      );
      const layout = matches.length === 1 ? matches[0] : null;
      return {
        pipelineId: pipeline.pipelineId,
        pipelineLayoutId: pipeline.layoutId,
        layoutMatchCount: matches.length,
        layout,
        hasOwnImmediateSize: layout?.hasOwnImmediateSize ?? false,
        immediateSize: layout?.immediateSize ?? null,
        pass: matches.length === 1
          && layout.hasOwnImmediateSize === true
          && layout.immediateSize === 0,
      };
    },
  );
  const computePipelineLayouts = [...new Map(computePipelineLayoutBindings
    .filter((binding) => binding.layout !== null)
    .map((binding) => [binding.pipelineLayoutId, binding.layout])).values()];
  const computeImmediateState = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-compute-immediate-state-exclusion',
    pass: computeModules.length > 0
      && computeModules.every((module) => (
        !/requires\s+immediate_address_space\s*;/.test(module.code ?? '')
          && !/var\s*<\s*immediate\s*>/.test(module.code ?? '')
      ))
      && instrumentation.evidence.computePipelines.length > 0
      && instrumentation.evidence.computePipelines.every((pipeline) => (
        (pipeline.immediateSize === null || pipeline.immediateSize === 0)
          && (pipeline.descriptor?.immediateSize === null
            || pipeline.descriptor?.immediateSize === 0)
      ))
      && computePipelineLayoutBindings.length
        === instrumentation.evidence.computePipelines.length
      && computePipelineLayoutBindings.every((binding) => binding.pass),
    shaderModuleCount: computeModules.length,
    shaderModules: computeModules.map((module) => ({
      moduleId: module.moduleId,
      creationSequence: module.sequence,
      requiresImmediateAddressSpaceCount:
        module.code?.match(/requires\s+immediate_address_space\s*;/g)?.length ?? 0,
      immediateVariableDeclarationCount:
        module.code?.match(/var\s*<\s*immediate\s*>/g)?.length ?? 0,
    })),
    pipelineCount: instrumentation.evidence.computePipelines.length,
    pipelines: instrumentation.evidence.computePipelines.map((pipeline) => ({
      pipelineId: pipeline.pipelineId,
      creationSequence: pipeline.sequence,
      hasOwnImmediateSize: pipeline.hasOwnImmediateSize,
      immediateSize: pipeline.immediateSize,
      descriptorHasOwnImmediateSize: pipeline.descriptor?.hasOwnImmediateSize ?? false,
      descriptorImmediateSize: pipeline.descriptor?.immediateSize ?? null,
    })),
    pipelineLayoutBindingCount: computePipelineLayoutBindings.length,
    pipelineLayoutBindings: computePipelineLayoutBindings.map((binding) => ({
      pipelineId: binding.pipelineId,
      pipelineLayoutId: binding.pipelineLayoutId,
      layoutMatchCount: binding.layoutMatchCount,
      hasOwnImmediateSize: binding.hasOwnImmediateSize,
      immediateSize: binding.immediateSize,
      pass: binding.pass,
    })),
    pipelineLayoutCount: computePipelineLayouts.length,
    pipelineLayouts: computePipelineLayouts,
  };
  requireCondition(computeImmediateState.pass,
    'Compute pipelines or WGSL unexpectedly expose immediate state.',
    computeImmediateState);
  const commandComputeModules = computeModules.filter(
    (module) => /\bFixedSliceIndexedDraw\b/.test(module.code ?? ''),
  );
  const computeFirstInstanceAudits = commandComputeModules.map((module) => ({
    moduleId: module.moduleId,
    declarationCount: module.code.match(/\bfirstInstance\s*:/g)?.length ?? 0,
    executableAccessCount: module.code.match(/\.\s*firstInstance\b/g)?.length ?? 0,
  })).map((audit) => ({
    ...audit,
    pass: audit.declarationCount === 1 && audit.executableAccessCount === 0,
  }));
  const computeFirstInstanceFieldAudit = {
    pass: computeFirstInstanceAudits.length > 0
      && computeFirstInstanceAudits.every((audit) => audit.pass),
    declarationCount: computeFirstInstanceAudits.reduce(
      (sum, audit) => sum + audit.declarationCount,
      0,
    ),
    executableAccessCount: computeFirstInstanceAudits.reduce(
      (sum, audit) => sum + audit.executableAccessCount,
      0,
    ),
    modules: computeFirstInstanceAudits,
  };
  requireCondition(commandComputeModules.length > 0
    && computeFirstInstanceFieldAudit.pass,
  'Command-buffer compute shader evidence is absent or accesses firstInstance.', {
      computeModuleCount: computeModules.length,
      commandComputeModuleCount: commandComputeModules.length,
      computeImmediateState,
      computeFirstInstanceAudits,
      computeFirstInstanceFieldAudit,
    });

  const geometryFixtures = await captureImmediateAifGeometryEvidence(runtime);
  const commonResources = await captureImmediateAifCommonResourceEvidence(
    runtime,
    shaderEvidence,
  );
  const gpuByteWitnesses = createGpuByteWitnessEvidence(runtime);
  requireCondition(gpuByteWitnesses.pass,
    'Raw shared/geometry GPU byte witnesses are incomplete.', gpuByteWitnesses);
  const addressDiagnosticPositions = createAddressDiagnosticPositionEvidence(runtime);
  requireCondition(addressDiagnosticPositions.pass,
    'Address diagnostic position commitments are incomplete.',
    addressDiagnosticPositions);
  await instrumentation.finalizeQueueWriteBufferSources();
  const queueWriteBufferLedger = createQueueWriteBufferLedger(runtime, scenarios);
  requireCondition(queueWriteBufferLedger.pass,
    'The queue.writeBuffer trace is not exhaustively partitioned.',
    queueWriteBufferLedger);
  const allCallbacks = challengedScenarioCallbacks(scenarios);
  const callbackPreflightBindings = allCallbacks.map((callback, index) => ({
    callbackOrdinal: index + 1,
    phase: callback.phase,
    scenarioId: callback.scenarioId,
    scheduleId: callback.scheduleId,
    lane: callback.lane,
    installationId: callback.scheduleInstallationId,
    commonStatePass: callback.commonState?.pass === true,
    commandPreflightPass: callback.command?.preflightBinding?.pass === true,
    commandPreflightInstallationId:
      callback.command?.preflightBinding?.installationId ?? null,
  }));
  const scheduleInstallations = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-schedule-installations',
    pass: runtime.scheduleInstallations.length === 8
      && ['v99', 'v20'].every((scenarioId) => (
        runtime.scheduleInstallations.filter(
          (installation) => installation.scenarioId === scenarioId,
        ).length === 4
      ))
      && runtime.scheduleInstallations.every((installation) => (
        installation.pass === true
          && installation.managerCalls.length === 4
          && Object.values(installation.commands).every((command) => command.exact)
          && installation.commonState.pass === true
      ))
      && runtime.scheduleInstallations.filter(
        (installation) => installation.orderedChallenge !== null,
      ).length === 6
      && runtime.scheduleInstallations.filter(
        (installation) => installation.orderedChallenge !== null,
      ).every((installation) => installation.orderedChallenge.pass === true)
      && callbackPreflightBindings.length === 126
      && callbackPreflightBindings.every((binding) => (
        binding.commonStatePass
          && binding.commandPreflightPass
          && binding.installationId === binding.commandPreflightInstallationId
      )),
    expectedInstallationCount: 8,
    installationCount: runtime.scheduleInstallations.length,
    expectedCommandReadbackCount: 24,
    commandReadbackCount: runtime.scheduleInstallations.reduce(
      (count, installation) => count + Object.keys(installation.commands).length,
      0,
    ),
    expectedOrderedChallengeCount: 6,
    orderedChallengeCount: runtime.scheduleInstallations.filter(
      (installation) => installation.orderedChallenge !== null,
    ).length,
    scenarioOrder: runtime.scheduleInstallations.map(
      (installation) => installation.scenarioId,
    ),
    scheduleOrder: runtime.scheduleInstallations.map(
      (installation) => installation.scheduleId,
    ),
    records: runtime.scheduleInstallations,
    callbackPreflightBindings,
  };
  requireCondition(scheduleInstallations.pass,
    'Schedule installation/preflight evidence is incomplete.', scheduleInstallations);

  const errorScopeDrainage = await drainPageErrorScopes(state, 'normal-completion');
  const scopedErrors = errorScopeDrainage.scopedErrors;
  requireCondition(errorScopeDrainage.pass
    && state.gpuErrors.uncaptured.length === 0
    && state.gpuErrors.unexpectedDeviceLosses.length === 0
    && state.gpuErrors.pageErrors.length === 0
    && state.gpuErrors.unhandledRejections.length === 0,
  'Page or WebGPU errors occurred during Phase 0.', state.gpuErrors);

  const topology = {
    ...strategy.diagnostics(),
    lanes: [...IMMEDIATE_AIF_PHASE0_LANES],
    bucketCount: IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount,
    drawCount: IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount,
    addressDiagnostic: runtime.addressDiagnostics.topologyEvidence(),
    sourceGeometryCount: sourceGeometries.length,
    target: {
      width: productionTarget.width,
      height: productionTarget.height,
      samples: productionTarget.samples,
      depthBuffer: productionTarget.depthBuffer,
      reversedDepthBuffer: renderer.reversedDepthBuffer === true,
    },
  };
  requireCondition(topology.pass
    && topology.addressDiagnostic.pass
    && topology.activeRootCount === 1
    && topology.activeLane === 'F',
    'Phase 0 topology evidence failed.', topology);
  const finalRuntimeState = strategy.state;
  const runtimeCoverage = {
    complete: finalRuntimeState.runtimeCoverageComplete,
    scenarioLoadSequence: [...finalRuntimeState.scenarioLoadSequence],
    expectedCellCount: finalRuntimeState.expectedLaneOrderCoverageCount,
    cells: { ...finalRuntimeState.laneOrderCoverage },
  };
  requireCondition(runtimeCoverage.complete
    && runtimeCoverage.expectedCellCount === 36
    && Object.keys(runtimeCoverage.cells).length === 36
    && Object.values(runtimeCoverage.cells).every((count) => count === 1),
  'The exact 36-cell runtime coverage is incomplete.', runtimeCoverage);
  const expectedWitnessIds = ['v99', 'v20'].flatMap((scenarioId) => (
    Object.keys(IMMEDIATE_AIF_PHASE0_SCHEDULES).flatMap((scheduleId) => (
      IMMEDIATE_AIF_PHASE0_LANES.map((lane) => `${scenarioId}/${scheduleId}/${lane}`)
    ))
  ));
  const addressWitnessRecords = Object.fromEntries(runtime.addressWitnesses);
  const addressWitnesses = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-address-byte-witnesses',
    pass: runtime.addressWitnesses.size === expectedWitnessIds.length
      && expectedWitnessIds.every((witnessId) => (
        addressWitnessRecords[witnessId]?.witnessId === witnessId
          && addressWitnessRecords[witnessId]?.byteLength
            === IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.byteLength
          && typeof addressWitnessRecords[witnessId]?.bytesBase64 === 'string'
      )),
    witnessCount: runtime.addressWitnesses.size,
    expectedWitnessIds,
    witnesses: addressWitnessRecords,
  };
  requireCondition(addressWitnesses.pass,
    'The 18 unique scenario/schedule/lane address byte witnesses are incomplete.',
    addressWitnesses);
  const outputWitnesses = createOutputWitnessEvidence(runtime, scenarios);
  requireCondition(outputWitnesses.pass,
    'Production color/depth/object-ID raw output witnesses are incomplete.',
    outputWitnesses);
  const readbackStaging = instrumentation.evidence.resources.filter(
    (resource) => resource.resourceClass === 'readback-staging',
  );
  const declaredBufferReadbackSources = [];
  const addBufferReadbackSource = (semantic, attribute) => {
    declaredBufferReadbackSources.push({
      semantic,
      resourceKind: 'buffer',
      attributeId: attribute.id,
      resourceId: instrumentation.identify(
        renderer.backend.get(attribute)?.buffer ?? null,
        'buffer',
      ),
    });
  };
  for (const [semantic, attribute] of Object.entries(strategy.sharedResources.attributes)) {
    addBufferReadbackSource(`shared.${semantic}`, attribute);
  }
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    addBufferReadbackSource(`command.${lane}`, strategy.laneStates[lane].indirectAttribute);
  }
  const mergedReadbackGeometry = strategy.laneStates.A.geometry;
  for (const semantic of ['bucketBase', 'normal', 'position', 'uv']) {
    addBufferReadbackSource(
      `merged-geometry.${semantic}`,
      mergedReadbackGeometry.getAttribute(semantic),
    );
  }
  addBufferReadbackSource('merged-geometry.index', mergedReadbackGeometry.index);
  const declaredTextureReadbackSources = [];
  const addTextureReadbackSource = (semantic, texture) => {
    if (texture === null) return;
    declaredTextureReadbackSources.push({
      semantic,
      resourceKind: 'texture',
      textureUuid: texture.uuid,
      resourceId: instrumentation.identify(
        renderer.backend.get(texture)?.texture ?? null,
        'texture',
      ),
    });
  };
  for (const [semantic, target] of [
    ['production', productionTarget],
    ['address', runtime.addressDiagnostics.target],
    ['objectId', runtime.objectIdDiagnostics.target],
  ]) {
    addTextureReadbackSource(`${semantic}.color`, target.texture);
    addTextureReadbackSource(`${semantic}.depth`, target.depthTexture);
  }
  const readbackSourceInventory = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-readback-source-inventory',
    pass: declaredBufferReadbackSources.length === 16
      && declaredTextureReadbackSources.length === 5
      && [...declaredBufferReadbackSources, ...declaredTextureReadbackSources].every(
        (record) => typeof record.resourceId === 'string',
      )
      && new Set(declaredBufferReadbackSources.map(
        (record) => record.resourceId,
      )).size === 16
      && new Set(declaredTextureReadbackSources.map(
        (record) => record.resourceId,
      )).size === 5,
    bufferSourceCount: declaredBufferReadbackSources.length,
    textureSourceCount: declaredTextureReadbackSources.length,
    bufferSources: declaredBufferReadbackSources,
    textureSources: declaredTextureReadbackSources,
  };
  requireCondition(readbackSourceInventory.pass,
    'Readback sources are not exhaustively identified.', readbackSourceInventory);
  const diagnosticReadbackStaging = readbackStaging.filter(
    (resource) => /\/(?:address|object-id)$/.test(resource.capturePhase ?? ''),
  );
  const allCommandEncoderOperations = instrumentation.evidence.commandEncoderTransfers;
  const allCommandEncoders = instrumentation.evidence.commandEncoders;
  const allQueueSubmissions = instrumentation.evidence.queueSubmissions;
  const transferDestinationBufferId = (record) => (
    record.method === 'copyBufferToBuffer'
      ? record.destinationBufferId
      : record.method === 'copyTextureToBuffer'
        ? record.destination?.bufferId ?? null
        : null
  );
  const stagingTransferLedger = readbackStaging.map((resource) => {
    const transfers = allCommandEncoderOperations.filter(
      (record) => transferDestinationBufferId(record) === resource.resourceId,
    );
    const transfer = transfers.length === 1 ? transfers[0] : null;
    const encoders = allCommandEncoders.filter(
      (record) => record.commandEncoderId === transfer?.commandEncoderId,
    );
    const encoder = encoders.length === 1 ? encoders[0] : null;
    const submissions = allQueueSubmissions.filter((record) => (
      exactNumberArray(record.commandBufferIds, [encoder?.commandBufferId])
    ));
    const submission = submissions.length === 1 ? submissions[0] : null;
    const mapEvents = instrumentation.evidence.bufferMapEvents.filter(
      (record) => record.resourceId === resource.resourceId,
    );
    const mapAsync = mapEvents[0] ?? null;
    const getMappedRange = mapEvents[1] ?? null;
    const transferSourceId = transfer?.method === 'copyBufferToBuffer'
      ? transfer.sourceBufferId
      : transfer?.method === 'copyTextureToBuffer'
        ? transfer.source?.textureId ?? null
        : null;
    const sourceMatches = [
      ...declaredBufferReadbackSources,
      ...declaredTextureReadbackSources,
    ].filter((record) => record.resourceId === transferSourceId);
    const sourceBinding = sourceMatches.length === 1 ? sourceMatches[0] : null;
    const transferShapeExact = transfer?.method === 'copyBufferToBuffer'
      ? typeof transfer.sourceBufferId === 'string'
        && transfer.sourceOffset >= 0
        && transfer.destinationOffset === 0
        && transfer.size === resource.size
      : transfer?.method === 'copyTextureToBuffer'
        && typeof transfer.source?.textureId === 'string'
        && transfer.source.mipLevel >= 0
        && Number.isInteger(transfer.source.origin?.x)
        && Number.isInteger(transfer.source.origin?.y)
        && Number.isInteger(transfer.source.origin?.z)
        && transfer.source.aspect === 'all'
        && transfer.destination.offset === 0
        && Number.isInteger(transfer.destination.bytesPerRow)
        && transfer.destination.bytesPerRow > 0
        && transfer.destination.bytesPerRow % 256 === 0
        && transfer.destination.rowsPerImage === null
        && Number.isInteger(transfer.copySize?.width)
        && transfer.copySize.width > 0
        && Number.isInteger(transfer.copySize?.height)
        && transfer.copySize.height > 0
        && transfer.copySize.depthOrArrayLayers === 1
        && resource.size === (transfer.copySize.height - 1)
          * transfer.destination.bytesPerRow + transfer.copySize.width * 4;
    const expectedMappedSize = transfer?.method === 'copyBufferToBuffer'
      ? resource.size
      : null;
    const pass = transfers.length === 1
      && encoders.length === 1
      && submissions.length === 1
      && sourceBinding !== null
      && transferShapeExact
      && resource.usage === (0x0001 | 0x0008)
      && resource.mappedAtCreation === false
      && typeof resource.capturePhase === 'string'
      && resource.capturePhase.length > 0
      && transfer.capturePhase === resource.capturePhase
      && encoder.capturePhase === resource.capturePhase
      && submission.capturePhase === resource.capturePhase
      && encoder.renderPassEncoderIds.length === 0
      && encoder.computePassEncoderIds.length === 0
      && exactNumberArray(encoder.transferSequences, [transfer.sequence])
      && encoder.finishCallCount === 1
      && mapEvents.length === 2
      && mapAsync.method === 'mapAsync'
      && mapAsync.mode === 1
      && mapAsync.offset === 0
      && mapAsync.size === expectedMappedSize
      && getMappedRange.method === 'getMappedRange'
      && getMappedRange.offset === 0
      && getMappedRange.size === expectedMappedSize
      && resource.sequence < encoder.sequence
      && encoder.sequence < transfer.sequence
      && transfer.sequence < encoder.finishSequence
      && encoder.finishSequence < submission.sequence
      && submission.sequence < mapAsync.sequence
      && mapAsync.sequence < getMappedRange.sequence
      && getMappedRange.sequence < resource.destroySequence
      && resource.destroyCapturePhase === resource.capturePhase
      && resource.destroyCallCount === 1;
    return {
      pass,
      capturePhase: resource.capturePhase,
      stagingBufferId: resource.resourceId,
      stagingSize: resource.size,
      stagingUsage: resource.usage,
      stagingMappedAtCreation: resource.mappedAtCreation,
      createSequence: resource.sequence,
      transfer,
      sourceBinding,
      commandEncoder: encoder,
      submission,
      mapEvents,
      destroySequence: resource.destroySequence,
      destroyCapturePhase: resource.destroyCapturePhase,
      destroyCallCount: resource.destroyCallCount,
      transferShapeExact,
    };
  });
  const bufferMapLifecycle = createBufferMapLifecycleEvidence(
    runtime,
    stagingTransferLedger,
  );
  requireCondition(bufferMapLifecycle.pass,
    'GPUBuffer mapping is not exhaustively bound to creation and first use.',
    bufferMapLifecycle);
  const renderOnlyCommandEncoders = allCommandEncoders.filter((record) => (
    record.renderPassEncoderIds.length === 1
      && record.computePassEncoderIds.length === 0
      && record.transferSequences.length === 0
  ));
  const computeOnlyCommandEncoders = allCommandEncoders.filter((record) => (
    record.renderPassEncoderIds.length === 0
      && record.computePassEncoderIds.length === 1
      && record.transferSequences.length === 0
  ));
  const transferOnlyCommandEncoders = allCommandEncoders.filter((record) => (
    record.renderPassEncoderIds.length === 0
      && record.computePassEncoderIds.length === 0
      && record.transferSequences.length === 1
  ));
  const allSubmittedCommandBufferIds = allQueueSubmissions.flatMap(
    (record) => record.commandBufferIds,
  );
  const allEncodedCommandBufferIds = allCommandEncoders.map(
    (record) => record.commandBufferId,
  );
  const globalCommandLedgerPass = allCommandEncoderOperations.length === 825
    && allCommandEncoderOperations.every((record) => (
      record.method === 'copyBufferToBuffer' || record.method === 'copyTextureToBuffer'
    ))
    && new Set(allCommandEncoderOperations.map(transferDestinationBufferId)).size === 825
    && stagingTransferLedger.length === 825
    && stagingTransferLedger.every((record) => record.pass)
    && allCommandEncoders.length === 1_629
    && renderOnlyCommandEncoders.length === 798
    && computeOnlyCommandEncoders.length === 6
    && transferOnlyCommandEncoders.length === 825
    && allCommandEncoders.every((record) => typeof record.capturePhase === 'string'
      && record.capturePhase.length > 0
      && record.finishCallCount === 1
      && Number.isInteger(record.finishSequence)
      && record.sequence < record.finishSequence
      && typeof record.commandBufferId === 'string')
    && allQueueSubmissions.length === 1_629
    && allQueueSubmissions.every((record) => typeof record.capturePhase === 'string'
      && record.capturePhase.length > 0
      && record.commandBufferIds.length === 1)
    && new Set(allEncodedCommandBufferIds).size === 1_629
    && new Set(allSubmittedCommandBufferIds).size === 1_629
    && exactNumberArray(
      [...allEncodedCommandBufferIds].sort(),
      [...allSubmittedCommandBufferIds].sort(),
    )
    && allCommandEncoders.every((encoder) => {
      const matchingSubmissions = allQueueSubmissions.filter((submission) => (
        exactNumberArray(submission.commandBufferIds, [encoder.commandBufferId])
      ));
      return matchingSubmissions.length === 1
        && matchingSubmissions[0].sequence > encoder.finishSequence;
    })
    && instrumentation.evidence.queueWriteTextures.length === 0
    && instrumentation.evidence.queueCopyExternalImagesToTexture.length === 0
    && bufferMapLifecycle.pass;
  const stagingLifecycle = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-readback-staging-lifecycle',
    pass: readbackStaging.length === EXPECTED_READBACK_STAGING_COUNT
      && diagnosticReadbackStaging.length > 0
      && new Set(readbackStaging.map((resource) => resource.capturePhase)).size
        === EXPECTED_READBACK_PHASE_COUNT
      && readbackStaging.every((resource) => resource.destroyed === true
        && resource.destroyCallCount === 1
        && resource.destroySequence > resource.sequence
        && typeof resource.capturePhase === 'string'
        && resource.capturePhase.length > 0
        && resource.destroyCapturePhase === resource.capturePhase)
      && globalCommandLedgerPass,
    createdCount: readbackStaging.length,
    expectedCreatedCount: EXPECTED_READBACK_STAGING_COUNT,
    readbackPhaseCount: new Set(
      readbackStaging.map((resource) => resource.capturePhase),
    ).size,
    expectedReadbackPhaseCount: EXPECTED_READBACK_PHASE_COUNT,
    readbackPhases: [...new Set(
      readbackStaging.map((resource) => resource.capturePhase),
    )].sort(),
    diagnosticCreatedCount: diagnosticReadbackStaging.length,
    liveCount: readbackStaging.filter((resource) => resource.destroyed !== true).length,
    transferCount: allCommandEncoderOperations.length,
    expectedTransferCount: EXPECTED_READBACK_STAGING_COUNT,
    transferMethodCounts: Object.fromEntries([
      'copyBufferToBuffer',
      'copyTextureToBuffer',
      'clearBuffer',
      'copyBufferToTexture',
      'copyTextureToTexture',
      'resolveQuerySet',
      'writeTimestamp',
    ].map((method) => [
      method,
      allCommandEncoderOperations.filter((record) => record.method === method).length,
    ])),
    expectedCommandEncoderCount: 1_629,
    commandEncoderCount: allCommandEncoders.length,
    commandEncoderClassCounts: {
      renderOnly: renderOnlyCommandEncoders.length,
      computeOnly: computeOnlyCommandEncoders.length,
      transferOnly: transferOnlyCommandEncoders.length,
    },
    expectedQueueSubmissionCount: 1_629,
    queueSubmissionCount: allQueueSubmissions.length,
    globalQueueWriteTextureCount: instrumentation.evidence.queueWriteTextures.length,
    globalCopyExternalImageToTextureCount:
      instrumentation.evidence.queueCopyExternalImagesToTexture.length,
    commandLedgerPass: globalCommandLedgerPass,
    sourceInventory: readbackSourceInventory,
    transferLedger: stagingTransferLedger,
    resources: readbackStaging.map((resource) => ({
      resourceId: resource.resourceId,
      capturePhase: resource.capturePhase,
      createSequence: resource.sequence,
      destroySequence: resource.destroySequence,
      destroyCapturePhase: resource.destroyCapturePhase,
      destroyed: resource.destroyed,
      destroyCallCount: resource.destroyCallCount,
      size: resource.size,
      usage: resource.usage,
    })),
  };
  requireCondition(stagingLifecycle.pass,
    'Transient readback staging buffers were not destroyed exactly once.', stagingLifecycle);
  const renderPassTermination = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-render-pass-terminal-end-evidence',
    pass: instrumentation.evidence.renderPassEncoders.length > 0
      && instrumentation.evidence.renderPassEncoders.every((trace) => {
        const endEvents = trace.events.filter((event) => event.method === 'end');
        return endEvents.length === 1
          && trace.events.at(-1)?.sequence === endEvents[0].sequence;
      }),
    passCount: instrumentation.evidence.renderPassEncoders.length,
    records: instrumentation.evidence.renderPassEncoders.map((trace) => ({
      encoderId: trace.encoderId,
      capturePhase: trace.capturePhase,
      endCount: trace.events.filter((event) => event.method === 'end').length,
      endSequence: trace.events.find((event) => event.method === 'end')?.sequence ?? null,
      terminalSequence: trace.events.at(-1)?.sequence ?? null,
    })),
  };
  requireCondition(renderPassTermination.pass,
    'Every actual render pass must terminate with exactly one end().', renderPassTermination);
  const renderBundleBindGroupEvents = instrumentation.evidence.renderBundleEncoders
    .flatMap((trace) => trace.events.filter((event) => event.method === 'setBindGroup'));
  const renderPassBindGroupEvents = instrumentation.evidence.renderPassEncoders
    .flatMap((trace) => trace.events.filter((event) => event.method === 'setBindGroup'));
  const computePassBindGroupEvents = instrumentation.evidence.computePassEncoders
    .flatMap((trace) => trace.events.filter((event) => event.method === 'setBindGroup'));
  const allBindGroupEvents = [
    ...renderBundleBindGroupEvents,
    ...renderPassBindGroupEvents,
    ...computePassBindGroupEvents,
  ];
  const dynamicOffsetCalls = allBindGroupEvents.filter((event) => (
    event.dynamicOffsets !== null
      || event.dynamicOffsetStart !== null
      || event.dynamicOffsetLength !== null
      || event.selectedDynamicOffsets !== null
  ));
  const renderBundleImmediateTraces = instrumentation.evidence.renderBundleEncoders
    .map((trace) => ({
      encoderId: trace.encoderId,
      capturePhase: trace.capturePhase,
      bundleId: trace.bundleId,
      callCount: trace.events.filter((event) => event.method === 'setImmediates').length,
      calls: trace.events.filter((event) => event.method === 'setImmediates'),
    }));
  const renderPassImmediateCalls = instrumentation.evidence.renderPassEncoders
    .flatMap((trace) => trace.events.filter((event) => event.method === 'setImmediates'));
  const computePassImmediateCalls = instrumentation.evidence.computePassEncoders
    .flatMap((trace) => trace.events.filter((event) => event.method === 'setImmediates'));
  const occlusionQueryCalls = instrumentation.evidence.renderPassEncoders.flatMap(
    (trace) => trace.events.filter(
      (event) => event.method === 'beginOcclusionQuery'
        || event.method === 'endOcclusionQuery',
    ),
  );
  const encoderStateEvidence = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-encoder-state-exclusion-evidence',
    pass: instrumentation.evidence.renderBundleEncoders.length === 27
      && renderBundleImmediateTraces.filter((record) => record.callCount === 32).length === 21
      && renderBundleImmediateTraces.filter((record) => record.callCount === 0).length === 6
      && renderBundleImmediateTraces.every((record) => (
        record.callCount === 0
          || (record.callCount === IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount
            && record.calls.every((call) => call.dataSize === 1))
      ))
      && renderBundleImmediateTraces.reduce(
        (count, record) => count + record.callCount,
        0,
      ) === 672
      && renderPassImmediateCalls.length === 0
      && computePassImmediateCalls.length === 0
      && dynamicOffsetCalls.length === 0
      && occlusionQueryCalls.length === 0,
    expectedRenderBundleEncoderCount: 27,
    renderBundleEncoderCount: instrumentation.evidence.renderBundleEncoders.length,
    expectedImmediateRenderBundleCount: 21,
    immediateRenderBundleCount: renderBundleImmediateTraces.filter(
      (record) => record.callCount > 0,
    ).length,
    expectedRenderBundleSetImmediatesCount: 672,
    renderBundleSetImmediatesCount: renderBundleImmediateTraces.reduce(
      (count, record) => count + record.callCount,
      0,
    ),
    renderBundleImmediateTraces,
    renderPassSetImmediatesCount: renderPassImmediateCalls.length,
    renderPassSetImmediatesCalls: renderPassImmediateCalls,
    computePassSetImmediatesCount: computePassImmediateCalls.length,
    computePassSetImmediatesCalls: computePassImmediateCalls,
    setBindGroupCount: allBindGroupEvents.length,
    dynamicOffsetCallCount: dynamicOffsetCalls.length,
    dynamicOffsetCalls,
    occlusionQueryCallCount: occlusionQueryCalls.length,
    occlusionQueryCalls,
  };
  requireCondition(encoderStateEvidence.pass,
    'Phase 0 encoder state escaped the exact immediate/dynamic-offset/query grammar.',
    encoderStateEvidence);
  const querySets = instrumentation.evidence.resources.filter(
    (resource) => resource.method === 'createQuerySet',
  );
  const querySetEvidence = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-zero-query-set-evidence',
    pass: querySets.length === 0
      && instrumentation.evidence.renderPassEncoders.every((trace) => (
        trace.timestampWrites === null && trace.occlusionQuerySetId === null
      ))
      && instrumentation.evidence.computePassEncoders.every(
        (trace) => trace.timestampWrites === null,
      )
      && occlusionQueryCalls.length === 0
      && allCommandEncoderOperations.every(
        (record) => record.method !== 'resolveQuerySet'
          && record.method !== 'writeTimestamp',
      ),
    count: querySets.length,
    records: querySets,
  };
  requireCondition(querySetEvidence.pass,
    'Correctness-only Phase 0 must not allocate query sets.', querySetEvidence);
  const externalTextures = instrumentation.evidence.resources.filter(
    (resource) => resource.method === 'importExternalTexture',
  );
  const externalTextureEvidence = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-zero-external-texture-evidence',
    pass: externalTextures.length === 0,
    methodAvailable: typeof device.importExternalTexture === 'function',
    methodInstrumented: typeof device.importExternalTexture !== 'function'
      || instrumentation.evidence.patchedDeviceMethods.includes('importExternalTexture'),
    count: externalTextures.length,
    records: externalTextures,
  };
  externalTextureEvidence.pass = externalTextureEvidence.pass
    && externalTextureEvidence.methodInstrumented;
  requireCondition(externalTextureEvidence.pass,
    'Correctness-only Phase 0 must not import external textures.',
    externalTextureEvidence);
  const samplerEvidence = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-sampler-descriptor-evidence',
    pass: instrumentation.evidence.resources
      .filter((resource) => resource.method === 'createSampler')
      .every((resource) => resource.sampler !== null
        && typeof resource.sampler === 'object'),
    records: instrumentation.evidence.resources.filter(
      (resource) => resource.method === 'createSampler',
    ),
  };
  requireCondition(samplerEvidence.pass,
    'Sampler descriptor inventory is incomplete.', samplerEvidence);
  const persistentTextureRecords = instrumentation.evidence.resources.filter(
    (resource) => resource.method === 'createTexture',
  );
  const challengedTextureInventories = scenarios.flatMap((scenario) => (
    scenario.snapshots.slice(0, 3).map(
      (snapshot) => snapshot.productionChallenge.immutableTextureResources,
    )
  ));
  const declaredTargetTextures = challengedTextureInventories[0] ?? [];
  const declaredTargetTextureIds = declaredTargetTextures.map(
    (record) => record.gpuTextureId,
  );
  const targetTextureInventoryExact = declaredTargetTextures.length === 5
    && challengedTextureInventories.length === 6
    && challengedTextureInventories.every((records) => (
      exactStructuredValue(records, declaredTargetTextures)
    ))
    && new Set(declaredTargetTextureIds).size === 5
    && declaredTargetTextureIds.every((textureId) => persistentTextureRecords.some(
      (resource) => resource.resourceId === textureId,
    ));
  const internalTextureRecords = persistentTextureRecords.filter(
    (resource) => !declaredTargetTextureIds.includes(resource.resourceId),
  );
  const bindGroupViewIds = new Set(instrumentation.evidence.resources
    .filter((resource) => resource.method === 'createBindGroup')
    .flatMap((resource) => resource.entries ?? [])
    .map((entry) => entry.resourceId)
    .filter((resourceId) => typeof resourceId === 'string'));
  const attachmentViewIdsByTexture = new Map();
  for (const trace of instrumentation.evidence.renderPassEncoders) {
    for (const attachment of trace.colorAttachments ?? []) {
      if (typeof attachment?.textureId !== 'string'
        || typeof attachment?.viewId !== 'string') continue;
      const ids = attachmentViewIdsByTexture.get(attachment.textureId) ?? new Set();
      ids.add(attachment.viewId);
      attachmentViewIdsByTexture.set(attachment.textureId, ids);
    }
    const depth = trace.depthStencilAttachment;
    if (typeof depth?.textureId === 'string' && typeof depth?.viewId === 'string') {
      const ids = attachmentViewIdsByTexture.get(depth.textureId) ?? new Set();
      ids.add(depth.viewId);
      attachmentViewIdsByTexture.set(depth.textureId, ids);
    }
  }
  const textureInventory = persistentTextureRecords.map((resource) => {
    const views = instrumentation.evidence.textureViews.filter(
      (view) => view.textureId === resource.resourceId,
    );
    const boundViewIds = views.map((view) => view.viewId).filter(
      (viewId) => bindGroupViewIds.has(viewId),
    );
    const attachmentViewIds = views.map((view) => view.viewId).filter(
      (viewId) => attachmentViewIdsByTexture.get(resource.resourceId)?.has(viewId),
    );
    const classification = declaredTargetTextureIds.includes(resource.resourceId)
      ? 'declared-render-target'
      : 'renderer-internal';
    return {
      textureId: resource.resourceId,
      classification,
      createSequence: resource.sequence,
      capturePhase: resource.capturePhase,
      viewIds: views.map((view) => view.viewId),
      boundViewIds,
      attachmentViewIds,
      pass: views.length > 0
        && new Set(views.map((view) => view.viewId)).size === views.length
        && (classification === 'declared-render-target'
          ? attachmentViewIds.length === views.length && boundViewIds.length === 0
          : boundViewIds.length === views.length && attachmentViewIds.length === 0),
    };
  });
  const textureDestructionBeforeCleanup = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-texture-destruction-before-cleanup',
    pass: persistentTextureRecords.length >= 5
      && targetTextureInventoryExact
      && textureInventory.every((record) => record.pass)
      && persistentTextureRecords.every((resource) => resource.destroyed === false
        && resource.destroyCallCount === 0
        && resource.destroySequence === null
        && resource.destroyCapturePhase === null)
      && instrumentation.evidence.resourceDestructions.every(
        (record) => record.resourceClass !== 'persistent-texture',
      ),
    textureCount: persistentTextureRecords.length,
    textureIds: persistentTextureRecords.map((resource) => resource.resourceId),
    declaredTargetTextureCount: declaredTargetTextures.length,
    declaredTargetTextures,
    declaredTargetTextureIds,
    targetTextureInventoryExact,
    internalTextureCount: internalTextureRecords.length,
    internalTextureIds: internalTextureRecords.map((resource) => resource.resourceId),
    textureInventory,
    destructionRecords: instrumentation.evidence.resourceDestructions.filter(
      (record) => record.resourceClass === 'persistent-texture',
    ),
  };
  requireCondition(textureDestructionBeforeCleanup.pass,
    'Persistent GPU textures were destroyed before terminal cleanup.',
    textureDestructionBeforeCleanup);
  state.evidence = {
    plan: immediateAifPhase0PlanEvidence(),
    runtimeSignature: signature,
    topology,
    shaders: {
      ...shaderEvidence,
      modules: shaderModules,
      computeModuleCount: computeModules.length,
      commandComputeModuleCount: commandComputeModules.length,
      computeImmediateState,
      computeFirstInstanceAudits,
      computeFirstInstanceFieldAudit,
      compilationErrorCount: shaderCompilationErrors.length,
      observationIdentityExact: shaderObservationIdentityExact,
    },
    pipelines: {
      bindGroupLayouts: instrumentation.evidence.bindGroupLayouts,
      layouts: instrumentation.evidence.pipelineLayouts,
      render: instrumentation.evidence.renderPipelines,
      compute: instrumentation.evidence.computePipelines,
    },
    commands: {
      pass: scheduleInstallations.pass,
      drawCount: IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount,
      strideBytes: INDIRECT_WORDS * Uint32Array.BYTES_PER_ELEMENT,
      commandByteLength: IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength,
      byteLength: IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength,
      indirectOffsets: [...IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets],
      offsets: [...IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets],
      laneCommandAttributeIds: strategy.diagnostics().commandBufferIds,
      scheduleInstallations,
    },
    scenarios,
    runtimeCoverage,
    addressWitnesses,
    outputWitnesses,
    resources: {
      gpuCreations: instrumentation.evidence.resources,
      gpuDestructions: instrumentation.evidence.resourceDestructions,
      textureViews: instrumentation.evidence.textureViews,
      stagingLifecycle,
      bufferMapLifecycle,
      renderPassTermination,
      encoderState: encoderStateEvidence,
      geometryFixtures,
      mergedGeometryRealization: runtime.mergedGeometryRealization,
      gpuByteWitnesses,
      addressDiagnosticPositions,
      queueWriteBufferLedger,
      commonResources,
      scheduleInstallations,
      persistentResourceFreezes: {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-persistent-resource-freezes',
        pass: runtime.scenarioResourceFreezes.length === 2
          && runtime.scenarioResourceFreezes.every((freezeRecord) => freezeRecord.pass),
        records: runtime.scenarioResourceFreezes,
      },
      querySets: querySetEvidence,
      externalTextures: externalTextureEvidence,
      samplers: samplerEvidence,
      textureDestructionBeforeCleanup,
      immediateSourceDetachments: runtime.immediateSourceDetachments.map((entry) => ({
        ...entry,
        recordedBases: Array.from(entry.recordedBases),
        mutatedBases: Array.from(entry.mutatedBases),
      })),
    },
    lifecycle: {
      patchedDeviceMethods: instrumentation.evidence.patchedDeviceMethods,
      renderBundleEncoderCount: instrumentation.evidence.renderBundleEncoders.length,
      renderPassEncoderCount: instrumentation.evidence.renderPassEncoders.length,
      computePassEncoderCount: instrumentation.evidence.computePassEncoders.length,
      renderBundleTraces: instrumentation.evidence.renderBundleEncoders,
      renderPassTraces: instrumentation.evidence.renderPassEncoders,
      computePassTraces: instrumentation.evidence.computePassEncoders,
      instrumentationMarkers: instrumentation.evidence.markers,
      queueWriteBufferCalls: instrumentation.evidence.queueWriteBuffers,
      queueWriteTextureCalls: instrumentation.evidence.queueWriteTextures,
      queueCopyExternalImageToTextureCalls:
        instrumentation.evidence.queueCopyExternalImagesToTexture,
      queueSubmissions: instrumentation.evidence.queueSubmissions,
      bufferMapEvents: instrumentation.evidence.bufferMapEvents,
      commandEncoderTraces: instrumentation.evidence.commandEncoders,
      commandEncoderTransferCalls: instrumentation.evidence.commandEncoderTransfers,
      renderPassTermination,
      scopedErrors,
    },
  };
}

function gpuLifecycleFinalUseEvidence(instrumentation) {
  const { evidence } = instrumentation;
  const records = [];
  const add = (sequence, kind, id = null) => {
    if (Number.isInteger(sequence)) records.push({ sequence, kind, id });
  };
  for (const resource of evidence.resources) {
    add(resource.sequence, `resource/${resource.method}`, resource.resourceId);
  }
  for (const view of evidence.textureViews) add(view.sequence, 'texture/createView', view.viewId);
  for (const record of [
    ...evidence.shaderModules,
    ...evidence.bindGroupLayouts,
    ...evidence.pipelineLayouts,
    ...evidence.renderPipelines,
    ...evidence.computePipelines,
  ]) add(record.sequence, 'pipeline-resource', record.pipelineId ?? record.moduleId ?? null);
  for (const [kind, traces] of [
    ['render-bundle', evidence.renderBundleEncoders],
    ['render-pass', evidence.renderPassEncoders],
    ['compute-pass', evidence.computePassEncoders],
  ]) {
    for (const trace of traces) {
      add(trace.sequence, `${kind}/begin`, trace.encoderId);
      for (const event of trace.events) add(event.sequence, `${kind}/${event.method}`, trace.encoderId);
      add(trace.finishSequence, `${kind}/finish`, trace.bundleId ?? trace.encoderId);
    }
  }
  for (const write of evidence.queueWriteBuffers) {
    add(write.sequence, 'queue/writeBuffer', write.bufferId);
  }
  for (const write of evidence.queueWriteTextures) {
    add(write.sequence, 'queue/writeTexture', write.textureId);
  }
  for (const copy of evidence.queueCopyExternalImagesToTexture) {
    add(copy.sequence, 'queue/copyExternalImageToTexture', copy.destination?.textureId ?? null);
  }
  for (const map of evidence.bufferMapEvents) add(map.sequence, `buffer/${map.method}`, map.resourceId);
  for (const transfer of evidence.commandEncoderTransfers) {
    add(transfer.sequence, `command-encoder/${transfer.method}`, transfer.commandEncoderId);
  }
  for (const encoder of evidence.commandEncoders) {
    add(encoder.sequence, 'command-encoder/create', encoder.commandEncoderId);
    add(encoder.finishSequence, 'command-encoder/finish', encoder.commandBufferId);
  }
  for (const submission of evidence.queueSubmissions) {
    add(submission.sequence, 'queue/submit', submission.commandBufferIds?.[0] ?? null);
  }
  records.sort((left, right) => left.sequence - right.sequence);
  const uniqueRecords = [...new Map(records.map((record) => [record.sequence, record])).values()]
    .sort((left, right) => left.sequence - right.sequence);
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-global-final-gpu-use',
    pass: uniqueRecords.length > 0,
    eventCount: uniqueRecords.length,
    representedRecordCount: records.length,
    duplicateRepresentationCount: records.length - uniqueRecords.length,
    finalUseSequence: uniqueRecords.at(-1)?.sequence ?? null,
    finalUse: uniqueRecords.at(-1) ?? null,
  };
}

async function cleanup(state) {
  let complete = true;
  const errors = [];
  let errorScopeDrainage = state.errorScopeDrainage;
  if (errorScopeDrainage === null) {
    try {
      errorScopeDrainage = await drainPageErrorScopes(state, 'failure-teardown');
    } catch (error) {
      complete = false;
      errors.push({ label: 'error-scope-drainage', error: serializeImmediateAifPhase0Error(error) });
    }
  }
  if (errorScopeDrainage?.pass !== true) {
    complete = false;
    errors.push({ label: 'error-scope-drainage', error: {
      name: 'Error',
      message: 'WebGPU error scopes did not drain to three empty results.',
      detail: errorScopeDrainage,
    } });
  }
  const instrumentation = state.runtime?.instrumentation ?? null;
  const globalFinalUse = instrumentation === null
    ? null
    : gpuLifecycleFinalUseEvidence(instrumentation);
  const cleanupStart = instrumentation === null
    ? null
    : instrumentation.mark('phase0-cleanup-start', {
        globalFinalUseSequence: globalFinalUse.finalUseSequence,
      });
  if (globalFinalUse !== null && (!globalFinalUse.pass
    || globalFinalUse.finalUseSequence >= cleanupStart.sequence)) {
    complete = false;
    errors.push({ label: 'cleanup-boundary', error: {
      name: 'Error',
      message: 'Cleanup boundary does not follow the final recorded GPU use.',
    } });
  }
  const attempt = async (label, callback) => {
    if (instrumentation !== null) {
      instrumentation.setCapturePhase(`phase0/cleanup/${label}`);
    }
    try {
      await callback();
    } catch (error) {
      complete = false;
      errors.push({ label, error: serializeImmediateAifPhase0Error(error) });
    } finally {
      if (instrumentation !== null) instrumentation.setCapturePhase(null);
    }
  };
  await attempt('address-diagnostics', () => state.runtime?.addressDiagnostics?.dispose());
  await attempt('object-id-diagnostics', () => state.runtime?.objectIdDiagnostics?.dispose());
  await attempt('production-target', () => state.productionTarget?.dispose());
  await attempt('production-scene', () => state.productionScene?.clear());
  await attempt('strategy', () => {
    if (state.renderer && state.strategy) {
      disposeStrategyResources(state.renderer, state.strategy);
    }
  });
  await attempt('source-geometries', () => {
    for (const geometry of state.sourceGeometries ?? []) geometry.dispose();
  });
  await attempt('renderer', () => state.renderer?.dispose());
  let explicitPersistentTextureDestruction = null;
  await attempt('persistent-textures', () => {
    if (state.runtime?.instrumentation == null) return;
    explicitPersistentTextureDestruction =
      state.runtime.instrumentation.destroyUndestroyedPersistentTextures();
    requireCondition(explicitPersistentTextureDestruction.pass,
      'Persistent GPU textures were not explicitly destroyed exactly once.',
      explicitPersistentTextureDestruction);
  });
  let explicitPersistentBufferDestruction = null;
  await attempt('persistent-buffers', () => {
    if (state.runtime?.instrumentation == null) return;
    state.runtime.instrumentation.setCapturePhase('phase0/cleanup/persistent-buffers');
    try {
      explicitPersistentBufferDestruction =
        state.runtime.instrumentation.destroyUndestroyedPersistentBuffers();
      requireCondition(explicitPersistentBufferDestruction.pass,
        'Persistent GPU buffers were not explicitly destroyed exactly once.',
        explicitPersistentBufferDestruction);
    } finally {
      state.runtime.instrumentation.setCapturePhase(null);
    }
  });
  await attempt('device', async () => {
    if (state.device) {
      state.intentionalDeviceDestroy = true;
      state.device.destroy();
      const terminal = await state.deviceLostPromise;
      requireCondition(terminal?.reason === 'destroyed',
        'Intentional device destruction did not produce terminal destroyed loss.', terminal);
    }
  });
  const persistentBufferRecords = state.runtime?.instrumentation?.evidence?.resources
    ?.filter((resource) => resource.method === 'createBuffer'
      && resource.resourceClass !== 'readback-staging') ?? [];
  const terminalDeviceCoverage = state.intentionalDeviceDestroy === true
    && state.gpuErrors.finalDeviceLoss?.reason === 'destroyed';
  const persistentBuffers = persistentBufferRecords.map((resource) => {
    const explicitlyDestroyedExactlyOnce = resource.destroyed === true
      && resource.destroyCallCount === 1
      && resource.destroySequence > resource.sequence
      && resource.destroySequence > (globalFinalUse?.finalUseSequence ?? -1)
      && resource.destroySequence > (cleanupStart?.sequence ?? -1)
      && /^phase0\/cleanup\//.test(resource.destroyCapturePhase ?? '');
    return {
      resourceId: resource.resourceId,
      resourceClass: resource.resourceClass,
      creationPhase: resource.capturePhase,
      createSequence: resource.sequence,
      destroyCallCount: resource.destroyCallCount,
      destroySequence: resource.destroySequence,
      destroyCapturePhase: resource.destroyCapturePhase,
      afterGlobalFinalUse: resource.destroySequence > (globalFinalUse?.finalUseSequence ?? -1),
      disposition: explicitlyDestroyedExactlyOnce
        ? 'explicit-destroy-exactly-once'
        : 'unaccounted',
      pass: explicitlyDestroyedExactlyOnce,
    };
  });
  const persistentBufferCleanup = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-persistent-buffer-cleanup',
    pass: explicitPersistentBufferDestruction?.pass === true
      && persistentBuffers.length > 0
      && persistentBuffers.every((resource) => resource.pass)
      && terminalDeviceCoverage,
    terminalDeviceCoverage,
    resourceCount: persistentBuffers.length,
    explicitlyDestroyedCount: persistentBuffers.filter(
      (resource) => resource.disposition === 'explicit-destroy-exactly-once',
    ).length,
    explicitSweep: explicitPersistentBufferDestruction,
    records: persistentBuffers,
  };
  if (!persistentBufferCleanup.pass) complete = false;
  const persistentTextureRecords = state.runtime?.instrumentation?.evidence?.resources
    ?.filter((resource) => resource.method === 'createTexture') ?? [];
  const persistentTextures = persistentTextureRecords.map((resource) => {
    const explicitlyDestroyedExactlyOnce = resource.destroyed === true
      && resource.destroyCallCount === 1
      && resource.destroySequence > resource.sequence
      && resource.destroySequence > (globalFinalUse?.finalUseSequence ?? -1)
      && resource.destroySequence > (cleanupStart?.sequence ?? -1)
      && /^phase0\/cleanup\//.test(resource.destroyCapturePhase ?? '');
    return {
      resourceId: resource.resourceId,
      resourceClass: resource.resourceClass,
      creationPhase: resource.capturePhase,
      createSequence: resource.sequence,
      destroyCallCount: resource.destroyCallCount,
      destroySequence: resource.destroySequence,
      destroyCapturePhase: resource.destroyCapturePhase,
      classification: state.evidence?.resources?.textureDestructionBeforeCleanup
        ?.declaredTargetTextureIds?.includes(resource.resourceId)
        ? 'declared-render-target'
        : 'renderer-internal',
      afterGlobalFinalUse: resource.destroySequence > (globalFinalUse?.finalUseSequence ?? -1),
      disposition: explicitlyDestroyedExactlyOnce
        ? 'explicit-destroy-exactly-once'
        : 'unaccounted',
      pass: explicitlyDestroyedExactlyOnce,
    };
  });
  const persistentTextureCleanup = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-persistent-texture-cleanup',
    pass: explicitPersistentTextureDestruction?.pass === true
      && persistentTextures.length >= 5
      && persistentTextures.every((resource) => resource.pass)
      && persistentTextures.filter(
        (resource) => resource.classification === 'declared-render-target',
      ).length === 5
      && terminalDeviceCoverage,
    terminalDeviceCoverage,
    resourceCount: persistentTextures.length,
    explicitlyDestroyedCount: persistentTextures.filter(
      (resource) => resource.disposition === 'explicit-destroy-exactly-once',
    ).length,
    declaredRenderTargetCount: persistentTextures.filter(
      (resource) => resource.classification === 'declared-render-target',
    ).length,
    internalTextureCount: persistentTextures.filter(
      (resource) => resource.classification === 'renderer-internal',
    ).length,
    allowedDestroyCapturePhases: [...new Set(persistentTextures.map(
      (resource) => resource.destroyCapturePhase,
    ))].sort(),
    explicitSweep: explicitPersistentTextureDestruction,
    records: persistentTextures,
  };
  if (!persistentTextureCleanup.pass) complete = false;
  state.cleanup = {
    complete,
    errors,
    strategyRootDetached: state.strategy?.root?.parent === null,
    sourceGeometryCountDisposed: state.sourceGeometries?.length ?? 0,
    deviceDestroyIntentional: state.intentionalDeviceDestroy,
    terminalDeviceLoss: state.gpuErrors.finalDeviceLoss,
    cleanupStartSequence: cleanupStart?.sequence ?? null,
    errorScopeDrainage,
    globalFinalUse,
    persistentBufferCleanup,
    persistentTextureCleanup,
  };
}

async function attachImmediateAifObservationChains({ nonce, evidence }) {
  const chains = [];
  const addCallbackChains = async (callback, useId) => {
    requireCondition(callback?.pass === true
      && IMMEDIATE_AIF_PHASE0_LANES.includes(callback.lane),
    `Observation callback ${useId} is incomplete.`, callback);
    const shaderLane = evidence.shaders.lanes[callback.lane];
    requireCondition(shaderLane?.pass === true,
      `Observation callback ${useId} has no bound shader lane.`, shaderLane);
    const definitions = [
      {
        slot: 'production',
        kind: 'immediate-aif-phase0-production-observation-chain',
        chainEvidence: {
          modules: shaderLane.boundModules,
          pipeline: shaderLane.boundPipeline,
          bundle: callback.bundle,
          phases: callback.phases,
          commandGpuBufferId: callback.command.gpuBufferId,
          visibleIdsGpuBufferId: callback.address.visibleIdsGpuBufferId,
          outputs: {
            color: callback.output.color,
            depth: callback.output.depth,
          },
        },
      },
      {
        slot: 'address',
        kind: 'immediate-aif-phase0-address-observation-chain',
        chainEvidence: {
          producer: callback.address.producer,
          outputSha256: callback.address.sha256,
          phases: callback.phases,
        },
      },
      {
        slot: 'objectId',
        kind: 'immediate-aif-phase0-object-id-observation-chain',
        chainEvidence: {
          producer: callback.output.objectId.producer,
          outputSha256: callback.output.objectId.sha256,
          phases: callback.phases,
        },
      },
    ];
    callback.observationChains = {};
    for (const definition of definitions) {
      const chain = await createImmediateAifObservationChain({
        nonce,
        chainId: `${useId}/${definition.slot}`,
        kind: definition.kind,
        chainEvidence: definition.chainEvidence,
      });
      callback.observationChains[definition.slot] = chain;
      chains.push(chain);
    }
  };

  for (const scenario of evidence.scenarios) {
    for (const snapshot of scenario.snapshots) {
      const snapshotId = `${scenario.scenarioId}/${snapshot.label}`;
      if (snapshot.rerecord?.callback !== undefined) {
        await addCallbackChains(snapshot.rerecord.callback, `${snapshotId}/rerecord-I`);
      }
      if (snapshot.laneOrders !== undefined) {
        for (const [orderIndex, order] of snapshot.laneOrders.results.entries()) {
          const orderId = order.order.join('');
          for (const [position, record] of order.records.entries()) {
            await addCallbackChains(
              record.callback,
              `${snapshotId}/order-${orderIndex}-${orderId}/position-${position}-${record.lane}`,
            );
          }
        }
      } else {
        for (const callback of snapshot.laneCaptures) {
          await addCallbackChains(callback, `${snapshotId}/audit-${callback.lane}`);
        }
      }
    }
  }

  for (const scenarioId of ['v99', 'v20']) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      for (const phase of ['reset', 'cull']) {
        const collector = evidence.shaders.computeLanes[lane].phases[phase];
        const boundExecution = evidence.shaders.computeExecution
          .scenarios[scenarioId].lanes[lane].phases[phase];
        const chain = await createImmediateAifObservationChain({
          nonce,
          chainId: `compute/${scenarioId}/${lane}/${phase}`,
          kind: `immediate-aif-phase0-compute-${phase}-observation-chain`,
          chainEvidence: {
            collector: {
              rawSha256: collector.rawSha256,
              normalizedSha256: collector.normalizedSha256,
              normalization: collector.normalization,
            },
            boundExecution,
          },
        });
        boundExecution.observationChain = chain;
        chains.push(chain);
      }
    }
  }
  requireCondition(chains.length === 390
    && new Set(chains.map((chain) => chain.chainId)).size === chains.length,
  'The deterministic 390-chain observation traversal is incomplete.', {
      chainCount: chains.length,
      uniqueChainCount: new Set(chains.map((chain) => chain.chainId)).size,
    });
  return chains;
}

function recordUnexpectedCleanupFailure(state, cleanupFailure) {
  const priorCleanup = state.cleanup !== null && typeof state.cleanup === 'object'
    ? state.cleanup
    : {};
  const priorErrors = Array.isArray(priorCleanup.errors) ? priorCleanup.errors : [];
  state.cleanup = {
    ...priorCleanup,
    complete: false,
    errors: [
      ...priorErrors,
      { label: 'unexpected-cleanup-throw', error: cleanupFailure },
    ],
    unexpectedFailure: cleanupFailure,
  };
}

export async function runImmediateAifPhase0ExecutionLifecycle({
  state,
  execute,
  teardown,
  eventTarget,
  onError,
  onUnhandledRejection,
}) {
  eventTarget.addEventListener('error', onError);
  eventTarget.addEventListener('unhandledrejection', onUnhandledRejection);
  let failure = null;
  let cleanupFailure = null;
  let status = 'technical-canary-complete';
  try {
    try {
      await execute(state);
    } catch (error) {
      failure = serializeImmediateAifPhase0Error(error);
      status = error?.phase0Status === 'unsupported'
        ? 'technical-canary-unsupported'
        : 'technical-canary-failed';
    }
    try {
      await teardown(state);
    } catch (error) {
      cleanupFailure = serializeImmediateAifPhase0Error(error);
      recordUnexpectedCleanupFailure(state, cleanupFailure);
      if (failure === null) {
        status = 'technical-canary-failed';
        failure = {
          name: 'Error',
          message: 'Phase 0 cleanup threw unexpectedly.',
          stack: null,
          detail: cleanupFailure,
        };
      }
    }
  } finally {
    eventTarget.removeEventListener('error', onError);
    eventTarget.removeEventListener('unhandledrejection', onUnhandledRejection);
  }
  return { failure, cleanupFailure, status };
}

export async function runImmediateAifPhase0Page() {
  const state = {
    targetConfiguration: null,
    adapterInfo: null,
    capabilities: {
      navigatorGpu: typeof navigator?.gpu === 'object',
      wgslLanguageFeatures: [],
      wgslImmediateAddressSpace: false,
      adapterMaxImmediateSize: null,
      deviceMaxImmediateSize: null,
      renderPassSetImmediates: false,
      renderBundleSetImmediates: false,
      adapterFeatures: [],
      requestedDeviceFeatures: [],
      deviceFeatures: [],
      indirectFirstInstanceAdapter: false,
      indirectFirstInstanceDevice: false,
      rawExpectation: null,
    },
    errorObservationScope: null,
    evidence: null,
    gpuErrors: {
      scoped: null,
      scopeDrainage: null,
      uncaptured: [],
      unexpectedDeviceLosses: [],
      finalDeviceLoss: null,
      pageErrors: [],
      unhandledRejections: [],
    },
    cleanup: null,
    device: null,
    renderer: null,
    strategy: null,
    sourceGeometries: [],
    productionScene: null,
    productionTarget: null,
    runtime: null,
    intentionalDeviceDestroy: false,
    deviceLostPromise: null,
    pendingErrorScopeFilters: [],
    errorScopeDrainage: null,
  };
  const onError = (event) => {
    state.gpuErrors.pageErrors.push({
      message: String(event?.message ?? 'page error').slice(0, 4_096),
      filename: String(event?.filename ?? '').slice(0, 2_048),
      line: event?.lineno ?? null,
      column: event?.colno ?? null,
    });
  };
  const onUnhandledRejection = (event) => {
    state.gpuErrors.unhandledRejections.push(serializeImmediateAifPhase0Error(event?.reason));
  };
  let { failure, cleanupFailure, status } =
    await runImmediateAifPhase0ExecutionLifecycle({
      state,
      execute: executePhase0,
      teardown: cleanup,
      eventTarget: globalThis,
      onError,
      onUnhandledRejection,
    });
  if (status === 'technical-canary-complete' && state.cleanup?.complete !== true) {
    status = 'technical-canary-failed';
    failure = {
      name: 'Error',
      message: 'Phase 0 cleanup was incomplete.',
      stack: null,
      detail: state.cleanup,
    };
  }
  if (status === 'technical-canary-complete'
    && (state.gpuErrors.uncaptured.length !== 0
      || state.gpuErrors.unexpectedDeviceLosses.length !== 0
      || state.gpuErrors.pageErrors.length !== 0
      || state.gpuErrors.unhandledRejections.length !== 0)) {
    status = 'technical-canary-failed';
    failure = {
      name: 'Error',
      message: 'Page or GPU errors occurred during Phase 0 cleanup.',
      stack: null,
      detail: state.gpuErrors,
    };
  }
  if (state.evidence !== null) {
    state.evidence.lifecycle.cleanup = state.cleanup;
  }
  const result = {
    schemaVersion: 1,
    kind: IMMEDIATE_AIF_PHASE0_PAGE_KIND,
    status,
    executionMode: 'technical-canary',
    analysisEligible: false,
    efficacyAnalysisAllowed: false,
    numericalDecision: null,
    timingCaptured: false,
    target: state.targetConfiguration === null ? null : {
      schemaVersion: state.targetConfiguration.schemaVersion,
      key: state.targetConfiguration.key,
      sourceFamily: state.targetConfiguration.sourceFamily,
      expectedRevision: state.targetConfiguration.expectedRevision,
    },
    scope: state.targetConfiguration?.scope ?? 'unbound Phase 0 page',
    coverage: {
      implemented: [...(state.targetConfiguration?.implemented ?? [])],
      deferred: [...(state.targetConfiguration?.deferred ?? [])],
    },
    userAgent: navigator.userAgent,
    secureContext: globalThis.isSecureContext,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    adapterInfo: state.adapterInfo,
    errorObservationScope: state.errorObservationScope,
    capabilities: state.capabilities,
    evidence: state.evidence,
    gpuErrors: state.gpuErrors,
    cleanup: state.cleanup,
    cleanupFailure,
    failure,
  };
  if (result.status === 'technical-canary-complete') {
    try {
      const chains = await attachImmediateAifObservationChains({
        nonce: state.targetConfiguration.observationNonce,
        evidence: result.evidence,
      });
      result.evidence.observationChallenge = await createImmediateAifObservationChallenge({
        nonce: state.targetConfiguration.observationNonce,
        target: result.target,
        evidence: result.evidence,
        chains,
      });
    } catch (error) {
      result.status = 'technical-canary-failed';
      result.failure = serializeImmediateAifPhase0Error(error);
    }
  }
  validateImmediateAifPhase0PageResultShape(result);
  return result;
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  const published = { ready: false, result: null };
  window[IMMEDIATE_AIF_PHASE0_PAGE_GLOBAL] = published;
  const status = document.getElementById('status');
  void runImmediateAifPhase0Page().then((result) => {
    published.result = result;
    published.ready = true;
    if (status) {
      status.textContent = result.status === 'technical-canary-complete'
        ? 'Full timing-free A/I/F Phase 0 completed.'
        : `Phase 0 stopped: ${result.failure?.message ?? result.status}`;
    }
  }).catch((error) => {
    published.result = {
      schemaVersion: 1,
      kind: IMMEDIATE_AIF_PHASE0_PAGE_KIND,
      status: 'technical-canary-failed',
      executionMode: 'technical-canary',
      analysisEligible: false,
      efficacyAnalysisAllowed: false,
      numericalDecision: null,
      timingCaptured: false,
      target: null,
      scope: 'page publication failure',
      coverage: { implemented: [], deferred: [] },
      capabilities: null,
      evidence: null,
      gpuErrors: null,
      failure: serializeImmediateAifPhase0Error(error),
    };
    published.ready = true;
    if (status) status.textContent = `Phase 0 publication failed: ${error.message}`;
  });
}
