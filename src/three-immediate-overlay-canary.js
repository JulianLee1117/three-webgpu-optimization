import {
  BundleGroup,
  DoubleSide,
  Float32BufferAttribute,
  IndirectStorageBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  MeshBasicNodeMaterial,
  NoBlending,
  NoColorSpace,
  OrthographicCamera,
  RenderTarget,
  REVISION,
  RGBAFormat,
  Scene,
  StorageBufferAttribute,
  Uint32BufferAttribute,
  UnsignedByteType,
  WebGPUBackend,
  WebGPURenderer,
} from 'three/webgpu';
import {
  Fn,
  expression,
  float,
  instanceIndex,
  positionGeometry,
  storage,
  uint,
  varyingProperty,
  vec4,
} from 'three/tsl';

export const THREE_IMMEDIATE_OVERLAY_CANARY_KIND =
  'three-immediate-overlay-development-canary';
// WGSL reserves identifiers beginning with two underscores. Keep this symbol
// deliberately descriptive without entering the reserved namespace.
export const IMMEDIATE_DRAW_BASE_EXPRESSION = 'threeImmediateDrawBase';
export const WGSL_IMMEDIATE_FEATURE = 'immediate_address_space';
export const TARGET_WIDTH = 8;
export const TARGET_HEIGHT = 1;
export const DRAW_COUNT = 4;
export const INSTANCES_PER_DRAW = 2;
export const INDIRECT_WORD_COUNT = 5;

// The probe runner replaces this exact marker once while Vite transforms this
// module. A direct/unbound page load therefore fails closed instead of silently
// assuming either the installed r185.1 package or the pinned upstream-dev tree.
const RUNNER_INJECTED_TARGET_CONFIGURATION =
  /* THREE_IMMEDIATE_CANARY_TARGET_CONFIGURATION_V1 */ null;

export const CANONICAL_IMMEDIATE_BASES = Object.freeze([0, 2, 4, 6]);
export const MUTATED_IMMEDIATE_BASES = Object.freeze([1, 3, 5, 0]);
export const VISIBLE_IDS = Object.freeze([5, 0, 7, 2, 1, 6, 3, 4]);

const TRACKED_BUNDLE_METHODS = Object.freeze([
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
  'pushDebugGroup',
  'popDebugGroup',
  'insertDebugMarker',
]);

function targetConfiguration() {
  const configuration = RUNNER_INJECTED_TARGET_CONFIGURATION;
  requireCondition(
    configuration !== null
      && configuration?.schemaVersion === 1
      && typeof configuration?.key === 'string'
      && typeof configuration?.sourceFamily === 'string'
      && typeof configuration?.expectedRevision === 'string'
      && typeof configuration?.scope === 'string'
      && Array.isArray(configuration?.implemented)
      && Array.isArray(configuration?.deferred),
    'The canary target configuration was not injected by its bound runner.',
    configuration,
  );
  return configuration;
}

function requireCondition(condition, message, detail = null) {
  if (condition) return;
  const error = new Error(message);
  error.detail = detail;
  throw error;
}

function serializeError(error) {
  return {
    name: String(error?.name ?? 'Error').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
    stack: error?.stack == null ? null : String(error.stack).slice(0, 8_192),
    detail: error?.detail == null ? null : structuredClone(error.detail),
  };
}

function serializeGpuError(error) {
  return {
    name: String(error?.constructor?.name ?? error?.name ?? 'GPUError').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
  };
}

function exactArray(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function countMatches(value, pattern) {
  return value.match(pattern)?.length ?? 0;
}

function bytesOf(view) {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

function toHex(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

async function sha256Bytes(bytes) {
  return toHex(await crypto.subtle.digest('SHA-256', bytes));
}

async function sha256Text(value) {
  return sha256Bytes(new TextEncoder().encode(value));
}

function sortedFeatureNames(features) {
  if (features == null || typeof features[Symbol.iterator] !== 'function') return [];
  return [...features].map(String).sort();
}

async function adapterInformation(adapter) {
  let information = adapter?.info ?? null;
  if (information === null && typeof adapter?.requestAdapterInfo === 'function') {
    information = await adapter.requestAdapterInfo();
  }
  if (information === null || typeof information !== 'object') return null;
  const serialized = {};
  for (const key of [
    'vendor',
    'architecture',
    'device',
    'description',
    'backend',
    'type',
    'driver',
    'isFallbackAdapter',
  ]) {
    if (['string', 'boolean'].includes(typeof information[key])) {
      serialized[key] = information[key];
    }
  }
  return serialized;
}

function replaceCallable(target, name, replacementFactory) {
  const original = target?.[name];
  requireCondition(typeof original === 'function', `${name} is not callable.`);
  const replacement = replacementFactory(original.bind(target));
  Object.defineProperty(target, name, {
    configurable: true,
    writable: true,
    value: replacement,
  });
  requireCondition(target[name] === replacement, `Could not instrument ${name}.`);
}

function immediateCallSnapshot(args) {
  const [rangeOffset, data, suppliedDataOffset, suppliedDataSize] = args;
  const dataOffset = suppliedDataOffset ?? 0;
  const elementCount = ArrayBuffer.isView(data)
    ? data.length
    : data.byteLength;
  const dataSize = suppliedDataSize ?? (elementCount - dataOffset);
  let selectedValues = null;
  if (data instanceof Uint32Array) {
    selectedValues = Array.from(data.slice(dataOffset, dataOffset + dataSize));
  }
  return {
    rangeOffset,
    sourceType: data?.constructor?.name ?? null,
    sourceElementCount: elementCount,
    dataOffset,
    dataSize,
    selectedValues,
  };
}

function installDeviceInstrumentation(device, immediateBases) {
  const objectIds = new WeakMap();
  let nextObjectId = 1;
  let nextSequence = 1;
  const evidence = {
    shaderModules: [],
    pipelineLayouts: [],
    renderPipelines: [],
    bundleEncoders: [],
    renderPassExecutions: [],
    patchedDeviceMethods: [],
  };
  let capturePhase = null;

  const objectId = (object, prefix) => {
    if (object === null || (typeof object !== 'object' && typeof object !== 'function')) {
      return null;
    }
    if (!objectIds.has(object)) objectIds.set(object, `${prefix}-${nextObjectId ++}`);
    return objectIds.get(object);
  };

  replaceCallable(device, 'createShaderModule', (original) => (descriptor) => {
    const module = original(descriptor);
    evidence.shaderModules.push({
      moduleId: objectId(module, 'shader-module'),
      label: descriptor?.label ?? null,
      code: descriptor?.code ?? null,
    });
    return module;
  });
  evidence.patchedDeviceMethods.push('createShaderModule');

  replaceCallable(device, 'createPipelineLayout', (original) => (descriptor) => {
    const snapshot = {
      sequence: nextSequence ++,
      label: descriptor?.label ?? null,
      hasOwnImmediateSize: Object.hasOwn(descriptor, 'immediateSize'),
      immediateSize: descriptor?.immediateSize ?? null,
      bindGroupLayoutCount: descriptor?.bindGroupLayouts?.length ?? null,
    };
    const layout = original(descriptor);
    snapshot.layoutId = objectId(layout, 'pipeline-layout');
    evidence.pipelineLayouts.push(snapshot);
    return layout;
  });
  evidence.patchedDeviceMethods.push('createPipelineLayout');

  for (const methodName of ['createRenderPipeline', 'createRenderPipelineAsync']) {
    replaceCallable(device, methodName, (original) => (descriptor) => {
      const record = {
        sequence: nextSequence ++,
        method: methodName,
        label: descriptor?.label ?? null,
        layoutId: objectId(descriptor?.layout, 'pipeline-layout'),
        vertexModuleId: objectId(descriptor?.vertex?.module, 'shader-module'),
        fragmentModuleId: objectId(descriptor?.fragment?.module, 'shader-module'),
        pipelineId: null,
      };
      evidence.renderPipelines.push(record);
      const pipelineOrPromise = original(descriptor);
      if (methodName === 'createRenderPipeline') {
        record.pipelineId = objectId(pipelineOrPromise, 'render-pipeline');
      } else {
        void pipelineOrPromise.then(
          (pipeline) => { record.pipelineId = objectId(pipeline, 'render-pipeline'); },
          () => undefined,
        );
      }
      return pipelineOrPromise;
    });
    evidence.patchedDeviceMethods.push(methodName);
  }

  replaceCallable(device, 'createCommandEncoder', (original) => (descriptor) => {
    const commandEncoder = original(descriptor);
    const commandEncoderId = objectId(commandEncoder, 'command-encoder');
    replaceCallable(commandEncoder, 'beginRenderPass', (nativeBeginRenderPass) => (
      renderPassDescriptor
    ) => {
      const renderPass = nativeBeginRenderPass(renderPassDescriptor);
      const renderPassId = objectId(renderPass, 'render-pass');
      replaceCallable(renderPass, 'executeBundles', (nativeExecuteBundles) => (bundles) => {
        const bundleList = Array.from(bundles ?? []);
        const value = nativeExecuteBundles(bundles);
        evidence.renderPassExecutions.push({
          sequence: nextSequence ++,
          capturePhase,
          commandEncoderId,
          renderPassId,
          bundleCount: bundleList.length,
          bundleIds: bundleList.map((bundle) => objectId(bundle, 'render-bundle')),
        });
        return value;
      });
      return renderPass;
    });
    return commandEncoder;
  });
  evidence.patchedDeviceMethods.push('createCommandEncoder');

  replaceCallable(device, 'createRenderBundleEncoder', (original) => (descriptor) => {
    const encoder = original(descriptor);
    const trace = {
      encoderId: objectId(encoder, 'bundle-encoder'),
      descriptor: {
        label: descriptor?.label ?? null,
        colorFormats: descriptor?.colorFormats?.map(String) ?? [],
        depthStencilFormat: descriptor?.depthStencilFormat ?? null,
        sampleCount: descriptor?.sampleCount ?? 1,
      },
      callableSetImmediates: typeof encoder.setImmediates === 'function',
      events: [],
      nativeFinishReturned: false,
      bundleId: null,
      sourceMutation: null,
    };
    evidence.bundleEncoders.push(trace);

    for (const methodName of TRACKED_BUNDLE_METHODS) {
      if (typeof encoder[methodName] !== 'function') continue;
      replaceCallable(encoder, methodName, (nativeMethod) => (...args) => {
        const value = nativeMethod(...args);
        const event = {
          sequence: nextSequence ++,
          method: methodName,
        };
        if (methodName === 'setImmediates') {
          Object.assign(event, immediateCallSnapshot(args));
        } else if (methodName === 'setPipeline') {
          event.pipelineId = objectId(args[0], 'render-pipeline');
        } else if (methodName === 'drawIndexedIndirect'
          || methodName === 'drawIndirect') {
          event.bufferId = objectId(args[0], 'gpu-buffer');
          event.indirectOffset = args[1];
        }
        trace.events.push(event);
        return value;
      });
    }

    replaceCallable(encoder, 'finish', (nativeFinish) => (...args) => {
      const sourceBeforeFinishMutation = Array.from(immediateBases);
      const bundle = nativeFinish(...args);
      trace.nativeFinishReturned = true;
      trace.bundleId = objectId(bundle, 'render-bundle');
      trace.events.push({ sequence: nextSequence ++, method: 'finish' });

      const immediateCalls = trace.events.filter(
        (event) => event.method === 'setImmediates',
      );
      const indexedDraws = trace.events.filter(
        (event) => event.method === 'drawIndexedIndirect',
      );
      if (immediateCalls.length === DRAW_COUNT && indexedDraws.length === DRAW_COUNT) {
        immediateBases.set(MUTATED_IMMEDIATE_BASES);
        trace.sourceMutation = {
          phase: 'after-native-finish-returned-before-bundle-execution',
          before: sourceBeforeFinishMutation,
          after: Array.from(immediateBases),
        };
      }
      return bundle;
    });

    return encoder;
  });
  evidence.patchedDeviceMethods.push('createRenderBundleEncoder');

  Object.defineProperties(evidence, {
    identifyObject: {
      value: objectId,
    },
    setCapturePhase: {
      value: (phase) => { capturePhase = phase; },
    },
  });

  return evidence;
}

export function createIndirectCommands() {
  const commands = new Uint32Array(DRAW_COUNT * INDIRECT_WORD_COUNT);
  for (let draw = 0; draw < DRAW_COUNT; draw += 1) {
    const word = draw * INDIRECT_WORD_COUNT;
    commands[word] = 3;
    commands[word + 1] = INSTANCES_PER_DRAW;
    // The first index makes draw ordinal part of the output oracle. Immediate
    // values can no longer be permuted among otherwise interchangeable draws.
    commands[word + 2] = draw * 3;
    commands[word + 3] = 0;
    commands[word + 4] = 0;
  }
  return commands;
}

export function expectedOutputForBases(bases) {
  const output = new Uint8Array(TARGET_WIDTH * TARGET_HEIGHT * 4);
  for (let draw = 0; draw < bases.length; draw += 1) {
    const base = bases[draw];
    for (let localInstance = 0; localInstance < INSTANCES_PER_DRAW; localInstance += 1) {
      const address = base + localInstance;
      if (address >= VISIBLE_IDS.length) continue;
      const pixel = draw * INSTANCES_PER_DRAW + localInstance;
      if (pixel >= TARGET_WIDTH) continue;
      const byteOffset = pixel * 4;
      output[byteOffset] = VISIBLE_IDS[address] + 1;
      output[byteOffset + 1] = 0;
      output[byteOffset + 2] = 0;
      output[byteOffset + 3] = 255;
    }
  }
  return output;
}

function createGeometry() {
  const geometry = new InstancedBufferGeometry();
  geometry.name = 'three-immediate-overlay-canary-geometry';
  const positions = [];
  const indices = [];
  const halfTriangleWidth = (2 / TARGET_WIDTH) * 0.375;
  for (let draw = 0; draw < DRAW_COUNT; draw += 1) {
    const firstPixel = draw * INSTANCES_PER_DRAW;
    const centerX = ((firstPixel + 0.5) * 2 / TARGET_WIDTH) - 1;
    positions.push(
      centerX - halfTriangleWidth, -0.75, 0.5,
      centerX + halfTriangleWidth, -0.75, 0.5,
      centerX, 0.75, 0.5,
    );
    indices.push(draw * 3, draw * 3 + 1, draw * 3 + 2);
  }
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setIndex(new Uint32BufferAttribute(indices, 1));
  geometry.instanceCount = INSTANCES_PER_DRAW;
  return geometry;
}

function createAddressMaterial(visibleIdsAttribute, baseNode, label) {
  const visibleRead = storage(
    visibleIdsAttribute,
    'uint',
    VISIBLE_IDS.length,
  ).toReadOnly();
  const material = new MeshBasicNodeMaterial();
  material.name = label;
  material.depthWrite = false;
  material.depthTest = false;
  material.toneMapped = false;
  material.blending = NoBlending;
  material.transparent = false;
  material.premultipliedAlpha = false;
  material.fog = false;
  material.side = DoubleSide;
  material.vertexNode = Fn(() => {
    const address = baseNode.add(instanceIndex).toVar(`${label}Address`);
    const objectId = visibleRead.element(address).toVar(`${label}ObjectId`);
    varyingProperty('uint', 'vThreeImmediateObjectId').assign(objectId);
    const instanceX = float(instanceIndex).mul(2 / TARGET_WIDTH);
    return vec4(
      positionGeometry.x.add(instanceX),
      positionGeometry.y,
      positionGeometry.z,
      1,
    );
  })();
  material.fragmentNode = Fn(() => {
    const objectId = varyingProperty('uint', 'vThreeImmediateObjectId');
    const encoded = objectId.add(uint(1)).toVar('threeImmediateEncodedObjectId');
    return vec4(
      float(encoded.bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255),
      float(encoded.shiftRight(uint(16)).bitAnd(uint(0xff))).div(255),
      1,
    );
  })();
  return material;
}

function createCanaryResources(renderer, immediateBases, labelPrefix) {
  const commands = createIndirectCommands();
  const indirectOffsets = Array.from(
    { length: DRAW_COUNT },
    (_, index) => index * INDIRECT_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT,
  );
  const indirectAttribute = new IndirectStorageBufferAttribute(
    commands,
    INDIRECT_WORD_COUNT,
  );
  const visibleIdsAttribute = new StorageBufferAttribute(
    Uint32Array.from(VISIBLE_IDS),
    1,
  );
  const geometry = createGeometry();
  geometry.setIndirect(indirectAttribute, indirectOffsets, immediateBases);
  const controlMaterial = createAddressMaterial(
    visibleIdsAttribute,
    uint(0),
    `${labelPrefix}Control`,
  );
  const immediateMaterial = createAddressMaterial(
    visibleIdsAttribute,
    expression(IMMEDIATE_DRAW_BASE_EXPRESSION, 'uint'),
    `${labelPrefix}Immediate`,
  );
  const mesh = new Mesh(geometry, immediateMaterial);
  mesh.name = `${labelPrefix}-mesh`;
  mesh.frustumCulled = false;
  mesh.matrixAutoUpdate = false;
  mesh.matrixWorldAutoUpdate = false;
  const root = new BundleGroup();
  root.name = `${labelPrefix}-bundle-group`;
  root.matrixAutoUpdate = false;
  root.matrixWorldAutoUpdate = false;
  root.add(mesh);
  const scene = new Scene();
  scene.add(root);
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.coordinateSystem = renderer.coordinateSystem;
  camera.position.z = 1;
  camera.updateProjectionMatrix();
  const target = new RenderTarget(TARGET_WIDTH, TARGET_HEIGHT, {
    depthBuffer: false,
    stencilBuffer: false,
    samples: 0,
    format: RGBAFormat,
    type: UnsignedByteType,
    colorSpace: NoColorSpace,
  });
  return {
    commands,
    indirectOffsets,
    indirectAttribute,
    visibleIdsAttribute,
    geometry,
    controlMaterial,
    immediateMaterial,
    mesh,
    root,
    scene,
    camera,
    target,
  };
}

function disposeCanaryResources(resources) {
  resources?.target?.dispose();
  resources?.geometry?.dispose();
  resources?.controlMaterial?.dispose();
  resources?.immediateMaterial?.dispose();
  resources?.indirectAttribute?.dispose?.();
  resources?.visibleIdsAttribute?.dispose?.();
}

async function requestCanaryAdapterAndDevice() {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
    forceFallbackAdapter: false,
  });
  requireCondition(adapter !== null, 'WebGPU adapter request returned null.');
  const adapterInfo = await adapterInformation(adapter);
  requireCondition(adapterInfo?.isFallbackAdapter === false,
    'A known non-fallback adapter is required.', adapterInfo);
  const adapterMaxImmediateSize = Number(adapter.limits?.maxImmediateSize);
  requireCondition(Number.isFinite(adapterMaxImmediateSize)
    && adapterMaxImmediateSize >= 4,
  'The adapter exposes fewer than four immediate bytes.', adapterMaxImmediateSize);
  const adapterFeatures = sortedFeatureNames(adapter.features);
  const device = await adapter.requestDevice({
    requiredFeatures: adapterFeatures,
    requiredLimits: { maxImmediateSize: 4 },
  });
  requireCondition(Number(device.limits?.maxImmediateSize) >= 4,
    'The requested device exposes fewer than four immediate bytes.');
  return {
    adapter,
    adapterInfo,
    adapterFeatures,
    adapterMaxImmediateSize,
    device,
  };
}

function runtimeOverlaySignatureEvidence() {
  const setterSource = InstancedBufferGeometry.prototype.setIndirect.toString();
  const backendDrawSource = WebGPUBackend.prototype._draw.toString();
  return {
    revision: REVISION,
    setIndirectHasImmediateBasesParameter:
      /setIndirect\( indirect, indirectOffset = 0, indirectImmediateBases = null \)/
        .test(setterSource),
    setIndirectStoresImmediateBases:
      /this\.indirectImmediateBases = indirectImmediateBases/.test(setterSource),
    backendHasSetImmediates:
      /setImmediates\( 0, indirectImmediateBases, i, 1 \)/.test(backendDrawSource),
    backendHasIndexedIndirectAdjacencyMarker:
      /set one indexed-indirect base immediately before its draw/.test(backendDrawSource),
  };
}

function renderObjectEvidence(renderer, target, root, mesh, camera) {
  const renderContext = renderer?._renderContexts?.get(target, renderer._mrt);
  const renderBundle = renderer?._bundles?.get(root, camera, renderContext);
  const renderBundleData = renderer?.backend?.get(renderBundle);
  const renderObjects = renderBundleData?.renderObjects;
  requireCondition(
    Array.isArray(renderObjects)
      && renderObjects.length === 1
      && renderObjects[0]?.object === mesh
      && renderObjects[0]?.bundle === root,
    'The cached bundle did not contain exactly the expected mesh.',
  );
  const renderObject = renderObjects[0];
  const nodeBuilderState = renderObject.getNodeBuilderState();
  const pipelineData = renderer.backend.get(renderObject.pipeline);
  return {
    renderBundle,
    renderBundleData,
    renderObject,
    pipelineGpu: pipelineData.pipeline,
    vertexShader: nodeBuilderState.vertexShader,
    fragmentShader: nodeBuilderState.fragmentShader,
    summary: {
      renderObjectCount: renderObjects.length,
      meshIdentityExact: renderObject.object === mesh,
      bundleIdentityExact: renderObject.bundle === root,
      bundleVersion: renderBundleData.version,
      rootVersion: root.version,
      bundleVersionExact: renderBundleData.version === root.version,
      pipelineImmediateSize: pipelineData.immediateSize ?? null,
      immediateBasesIdentityExact:
        renderObject.getIndirectImmediateBases() === mesh.geometry.indirectImmediateBases,
    },
  };
}

function shaderEvidence(vertexShader, fragmentShader) {
  const requirementCount = countMatches(
    vertexShader,
    /requires\s+immediate_address_space\s*;/g,
  );
  const declarationCount = countMatches(
    vertexShader,
    /var<immediate>\s+threeImmediateDrawBase\s*:\s*u32\s*;/g,
  );
  const addressUseCount = countMatches(
    vertexShader,
    /threeImmediateDrawBase\s*\+\s*instanceIndex/g,
  );
  const bucketBaseCount = countMatches(vertexShader, /\bbucketBase\b/g);
  return {
    exact: requirementCount === 1
      && declarationCount === 1
      && addressUseCount === 1
      && bucketBaseCount === 0,
    requirementCount,
    declarationCount,
    addressUseCount,
    bucketBaseCount,
    addressExpression: `${IMMEDIATE_DRAW_BASE_EXPRESSION} + instanceIndex`,
    vertexShader,
    fragmentShader,
  };
}

function encoderEvidence(instrumentation, indirectOffsets, drawGpuBufferId) {
  const candidates = instrumentation.bundleEncoders.filter((trace) => (
    trace.events.filter((event) => event.method === 'drawIndexedIndirect').length
      === DRAW_COUNT
      && trace.events.filter((event) => event.method === 'setImmediates').length
        === DRAW_COUNT
  ));
  requireCondition(candidates.length === 1,
    'Expected exactly one four-draw indexed-indirect bundle trace.', {
      candidateCount: candidates.length,
      encoderCount: instrumentation.bundleEncoders.length,
    });
  const trace = candidates[0];
  const draws = trace.events.filter((event) => event.method === 'drawIndexedIndirect');
  const immediateCalls = trace.events.filter((event) => event.method === 'setImmediates');
  const adjacency = draws.map((draw) => {
    const index = trace.events.indexOf(draw);
    const previous = trace.events[index - 1] ?? null;
    return {
      drawSequence: draw.sequence,
      drawOffset: draw.indirectOffset,
      previousMethod: previous?.method ?? null,
      previousSequence: previous?.sequence ?? null,
      adjacent: previous?.method === 'setImmediates',
      immediate: previous?.method === 'setImmediates' ? {
        rangeOffset: previous.rangeOffset,
        sourceType: previous.sourceType,
        sourceElementCount: previous.sourceElementCount,
        dataOffset: previous.dataOffset,
        dataSize: previous.dataSize,
        selectedValues: previous.selectedValues,
      } : null,
    };
  });
  const exactCalls = immediateCalls.length === DRAW_COUNT
    && immediateCalls.every((call, index) => (
      call.rangeOffset === 0
        && call.sourceType === 'Uint32Array'
        && call.sourceElementCount === DRAW_COUNT
        && call.dataOffset === index
        && call.dataSize === 1
        && exactArray(call.selectedValues, [CANONICAL_IMMEDIATE_BASES[index]])
    ));
  const exactDraws = exactArray(
    draws.map((draw) => draw.indirectOffset),
    indirectOffsets,
  );
  const drawBufferIds = draws.map((draw) => draw.bufferId);
  const drawBufferIdentityExact = typeof drawGpuBufferId === 'string'
    && drawBufferIds.length === DRAW_COUNT
    && drawBufferIds.every((bufferId) => bufferId === drawGpuBufferId);
  return {
    exact: trace.callableSetImmediates
      && exactCalls
      && exactDraws
      && drawBufferIdentityExact
      && adjacency.every((entry) => entry.adjacent)
      && trace.nativeFinishReturned
      && trace.sourceMutation?.phase
        === 'after-native-finish-returned-before-bundle-execution'
      && exactArray(trace.sourceMutation?.before ?? [], CANONICAL_IMMEDIATE_BASES)
      && exactArray(trace.sourceMutation?.after ?? [], MUTATED_IMMEDIATE_BASES),
    exactCalls,
    exactDraws,
    drawGpuBufferId,
    drawBufferIds,
    drawBufferIdentityExact,
    adjacency,
    trace,
    patchedDeviceMethods: instrumentation.patchedDeviceMethods,
  };
}

function pipelineEvidence(instrumentation, {
  vertexShader,
  fragmentShader,
  pipelineGpu,
  bundleTrace,
}) {
  const shaderById = new Map(
    instrumentation.shaderModules.map((record) => [record.moduleId, record]),
  );
  const layoutById = new Map(
    instrumentation.pipelineLayouts.map((record) => [record.layoutId, record]),
  );
  const classified = instrumentation.renderPipelines.map((pipeline) => {
    const vertexShader = shaderById.get(pipeline.vertexModuleId)?.code ?? '';
    return {
      ...pipeline,
      vertexUsesImmediate: vertexShader.includes(IMMEDIATE_DRAW_BASE_EXPRESSION),
      vertexIsControl: vertexShader.includes('threeOverlayInstrumentedControlAddress'),
      layout: layoutById.get(pipeline.layoutId) ?? null,
    };
  });
  const immediate = classified.filter((pipeline) => pipeline.vertexUsesImmediate);
  const controls = classified.filter((pipeline) => pipeline.vertexIsControl);
  const exactImmediate = immediate.length === 1
    && immediate.every((pipeline) => (
      pipeline.layout?.hasOwnImmediateSize === true
      && pipeline.layout?.immediateSize === 4
    ));
  const exactControl = controls.length === 1
    && controls.every((pipeline) => (
      pipeline.layout?.hasOwnImmediateSize === true
      && pipeline.layout?.immediateSize === 0
    ));
  const immediateRecord = immediate[0] ?? null;
  const immediateVertexModule = shaderById.get(immediateRecord?.vertexModuleId) ?? null;
  const immediateFragmentModule = shaderById.get(immediateRecord?.fragmentModuleId) ?? null;
  const bundleSetPipelineEvents = bundleTrace?.events?.filter(
    (event) => event.method === 'setPipeline',
  ) ?? [];
  const actualPipelineId = instrumentation.identifyObject(
    pipelineGpu,
    'render-pipeline',
  );
  const moduleSourceBindingExact = immediateVertexModule?.code === vertexShader
    && immediateFragmentModule?.code === fragmentShader;
  const pipelineToBundleBindingExact = typeof actualPipelineId === 'string'
    && immediateRecord?.pipelineId === actualPipelineId
    && bundleSetPipelineEvents.length === 1
    && bundleSetPipelineEvents[0].pipelineId === actualPipelineId;
  return {
    exact: exactImmediate
      && exactControl
      && moduleSourceBindingExact
      && pipelineToBundleBindingExact,
    exactImmediate,
    exactControl,
    immediatePipelineCount: immediate.length,
    controlPipelineCount: controls.length,
    actualPipelineId,
    immediatePipelineId: immediateRecord?.pipelineId ?? null,
    immediateVertexModuleId: immediateRecord?.vertexModuleId ?? null,
    immediateFragmentModuleId: immediateRecord?.fragmentModuleId ?? null,
    bundleSetPipelineIds: bundleSetPipelineEvents.map((event) => event.pipelineId),
    moduleSourceBindingExact,
    pipelineToBundleBindingExact,
    classified,
    shaderModuleSnapshots: instrumentation.shaderModules,
    layoutSnapshots: instrumentation.pipelineLayouts,
  };
}

async function renderAndRead(
  renderer,
  scene,
  camera,
  target,
  instrumentation = null,
  capturePhase = null,
) {
  instrumentation?.setCapturePhase(capturePhase);
  try {
    renderer.setRenderTarget(target);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, false, false);
    renderer.autoClear = false;
    renderer.render(scene, camera);
    await renderer.backend.device.queue.onSubmittedWorkDone();
    const output = await renderer.readRenderTargetPixelsAsync(
      target,
      0,
      0,
      TARGET_WIDTH,
      TARGET_HEIGHT,
    );
    requireCondition(output instanceof Uint8Array,
      'Render-target readback must be Uint8Array.');
    return output;
  } finally {
    instrumentation?.setCapturePhase(null);
  }
}

async function runUninstrumentedSnapshotCanary(
  instrumentedDevice,
  expectedCanonical,
  expectedMutated,
) {
  let renderer = null;
  let device = null;
  let resources = null;
  let intentionalDeviceDestroy = false;
  const gpuErrors = {
    uncaptured: [],
    unexpectedDeviceLosses: [],
  };
  try {
    const acquired = await requestCanaryAdapterAndDevice();
    ({ device } = acquired);
    requireCondition(device !== instrumentedDevice,
      'The uninstrumented snapshot control must use a distinct GPUDevice.');
    device.addEventListener('uncapturederror', (event) => {
      event.preventDefault();
      gpuErrors.uncaptured.push(serializeGpuError(event.error));
    });
    void device.lost.then((information) => {
      if (!intentionalDeviceDestroy) {
        gpuErrors.unexpectedDeviceLosses.push({
          reason: String(information?.reason ?? 'unknown').slice(0, 256),
          message: String(information?.message ?? '').slice(0, 4_096),
        });
      }
    });

    renderer = new WebGPURenderer({
      device,
      antialias: false,
      forceWebGL: false,
      trackTimestamp: false,
    });
    renderer.setPixelRatio(1);
    renderer.setSize(TARGET_WIDTH, TARGET_HEIGHT, false);
    await renderer.init();
    renderer.backend.trackTimestamp = false;
    requireCondition(renderer.backend?.isWebGPUBackend === true,
      'The uninstrumented control did not initialize WebGPU.');

    const immediateBases = Uint32Array.from(CANONICAL_IMMEDIATE_BASES);
    resources = createCanaryResources(
      renderer,
      immediateBases,
      'threeOverlayUninstrumented',
    );
    const firstOutput = await renderAndRead(
      renderer,
      resources.scene,
      resources.camera,
      resources.target,
    );
    const firstBundle = renderObjectEvidence(
      renderer,
      resources.target,
      resources.root,
      resources.mesh,
      resources.camera,
    );
    const firstBundleGpu = firstBundle.renderBundleData.bundleGPU;
    const sourceBeforeMutation = Array.from(immediateBases);
    immediateBases.set(MUTATED_IMMEDIATE_BASES);
    const sourceAfterMutation = Array.from(immediateBases);
    const secondOutput = await renderAndRead(
      renderer,
      resources.scene,
      resources.camera,
      resources.target,
    );
    const secondBundle = renderObjectEvidence(
      renderer,
      resources.target,
      resources.root,
      resources.mesh,
      resources.camera,
    );
    const secondBundleGpu = secondBundle.renderBundleData.bundleGPU;

    await device.queue.onSubmittedWorkDone();
    await new Promise((resolve) => setTimeout(resolve, 0));
    const firstSha256 = await sha256Bytes(firstOutput);
    const secondSha256 = await sha256Bytes(secondOutput);
    const canonicalSha256 = await sha256Bytes(expectedCanonical);
    const evidence = {
      exact: device !== instrumentedDevice
        && exactArray(sourceBeforeMutation, CANONICAL_IMMEDIATE_BASES)
        && exactArray(sourceAfterMutation, MUTATED_IMMEDIATE_BASES)
        && exactArray(firstOutput, expectedCanonical)
        && exactArray(secondOutput, expectedCanonical)
        && !exactArray(secondOutput, expectedMutated)
        && firstBundle.renderBundle === secondBundle.renderBundle
        && firstBundleGpu === secondBundleGpu
        && firstBundle.summary.pipelineImmediateSize === 4
        && secondBundle.summary.pipelineImmediateSize === 4
        && gpuErrors.uncaptured.length === 0
        && gpuErrors.unexpectedDeviceLosses.length === 0,
      deviceInstrumentationInstalledByCanary: false,
      disposableRenderer: true,
      deviceDistinctFromInstrumentedRenderer: device !== instrumentedDevice,
      mutationPhase: 'after-first-render-executed-before-second-cached-render',
      sourceBeforeMutation,
      sourceAfterMutation,
      expectedCanonical: Array.from(expectedCanonical),
      expectedMutated: Array.from(expectedMutated),
      firstObserved: Array.from(firstOutput),
      secondObserved: Array.from(secondOutput),
      canonicalSha256,
      firstSha256,
      secondSha256,
      renderBundleIdentityStable:
        firstBundle.renderBundle === secondBundle.renderBundle,
      bundleGpuIdentityStable: firstBundleGpu === secondBundleGpu,
      firstBundle: firstBundle.summary,
      secondBundle: secondBundle.summary,
      adapterInfo: acquired.adapterInfo,
      adapterFeatures: acquired.adapterFeatures,
      adapterMaxImmediateSize: acquired.adapterMaxImmediateSize,
      deviceMaxImmediateSize: Number(device.limits?.maxImmediateSize),
      errorObservationScope: {
        start: 'after-device-creation-before-renderer-construction-and-initialization',
        includesRendererInitialization: true,
        includesBothCanaryRendersAndReadbacks: true,
        excludesAdapterAndDeviceRequest: true,
      },
      gpuErrors,
    };
    requireCondition(evidence.exact,
      'The separate uninstrumented snapshot control failed.', evidence);
    return evidence;
  } finally {
    intentionalDeviceDestroy = true;
    disposeCanaryResources(resources);
    renderer?.dispose();
    device?.destroy();
  }
}

async function executeCanary(state) {
  const configuration = targetConfiguration();
  state.targetConfiguration = structuredClone(configuration);
  requireCondition(globalThis.isSecureContext, 'A secure context is required.');
  requireCondition(typeof navigator?.gpu === 'object', 'navigator.gpu is unavailable.');
  state.capabilities.wgslLanguageFeatures = sortedFeatureNames(
    navigator.gpu.wgslLanguageFeatures,
  );
  state.capabilities.immediateAddressSpace =
    state.capabilities.wgslLanguageFeatures.includes(WGSL_IMMEDIATE_FEATURE);
  requireCondition(state.capabilities.immediateAddressSpace,
    `WGSL language feature ${WGSL_IMMEDIATE_FEATURE} is unavailable.`);

  const runtimeSignature = runtimeOverlaySignatureEvidence();
  state.runtimeOverlaySignature = runtimeSignature;
  requireCondition(runtimeSignature.revision === configuration.expectedRevision,
    `The runtime Three.js revision is not ${configuration.expectedRevision}.`, {
      configuration,
      runtimeSignature,
    });
  requireCondition(
    runtimeSignature.setIndirectHasImmediateBasesParameter
      && runtimeSignature.setIndirectStoresImmediateBases
      && runtimeSignature.backendHasSetImmediates
      && runtimeSignature.backendHasIndexedIndirectAdjacencyMarker,
    'The served Three.js runtime does not contain the generated overlay.',
    runtimeSignature,
  );

  const acquired = await requestCanaryAdapterAndDevice();
  const { device } = acquired;
  state.resources.device = device;
  state.adapterInfo = acquired.adapterInfo;
  state.capabilities.nonFallbackAdapter = acquired.adapterInfo?.isFallbackAdapter === false;
  state.capabilities.adapterFeatures = acquired.adapterFeatures;
  state.capabilities.adapterMaxImmediateSize = acquired.adapterMaxImmediateSize;
  state.errorObservationScope = {
    start: 'after-device-creation-before-renderer-construction-and-initialization',
    includesRendererInitialization: true,
    includesAllInstrumentedRendererRendersAndReadbacks: true,
    excludesAdapterAndDeviceRequest: true,
  };
  device.addEventListener('uncapturederror', (event) => {
    event.preventDefault();
    state.gpuErrors.uncaptured.push(serializeGpuError(event.error));
  });
  void device.lost.then((information) => {
    if (!state.intentionalDeviceDestroy) {
      state.gpuErrors.unexpectedDeviceLosses.push({
        reason: String(information?.reason ?? 'unknown').slice(0, 256),
        message: String(information?.message ?? '').slice(0, 4_096),
      });
    }
  });

  const renderer = new WebGPURenderer({
    device,
    antialias: false,
    forceWebGL: false,
    powerPreference: 'high-performance',
    trackTimestamp: false,
  });
  state.resources.renderer = renderer;
  renderer.setPixelRatio(1);
  renderer.setSize(TARGET_WIDTH, TARGET_HEIGHT, false);
  document.getElementById('canvas-host')?.appendChild(renderer.domElement);
  await renderer.init();
  renderer.backend.trackTimestamp = false;
  requireCondition(renderer.backend?.isWebGPUBackend === true,
    'Three.js did not initialize its WebGPU backend.');

  const { backend } = renderer;
  state.capabilities.deviceMaxImmediateSize = Number(device.limits?.maxImmediateSize);
  state.capabilities.deviceFeatures = sortedFeatureNames(device.features);
  state.capabilities.renderPassSetImmediates =
    typeof globalThis.GPURenderPassEncoder?.prototype?.setImmediates === 'function';
  requireCondition(Number.isFinite(state.capabilities.deviceMaxImmediateSize)
    && state.capabilities.deviceMaxImmediateSize >= 4,
  'The Three.js WebGPU device exposes fewer than four immediate bytes.');
  requireCondition(state.capabilities.renderPassSetImmediates,
    'GPURenderPassEncoder.setImmediates is unavailable.');

  const immediateBases = Uint32Array.from(CANONICAL_IMMEDIATE_BASES);
  const instrumentation = installDeviceInstrumentation(device, immediateBases);
  state.instrumentation = instrumentation;

  const resources = createCanaryResources(
    renderer,
    immediateBases,
    'threeOverlayInstrumented',
  );
  const {
    commands,
    indirectOffsets,
    indirectAttribute,
    visibleIdsAttribute,
    geometry,
    controlMaterial,
    immediateMaterial,
    mesh,
    root,
    scene,
    camera,
    target,
  } = resources;
  state.resources = {
    ...state.resources,
    geometry,
    controlMaterial,
    immediateMaterial,
    indirectAttribute,
    visibleIdsAttribute,
    target,
  };

  const expectedCanonical = expectedOutputForBases(CANONICAL_IMMEDIATE_BASES);
  const expectedMutated = expectedOutputForBases(MUTATED_IMMEDIATE_BASES);
  requireCondition(!exactArray(expectedCanonical, expectedMutated),
    'The mutation challenge must have a distinct output oracle.');

  const firstOutput = await renderAndRead(
    renderer,
    scene,
    camera,
    target,
    instrumentation,
    'instrumented-immediate-first',
  );
  requireCondition(exactArray(immediateBases, MUTATED_IMMEDIATE_BASES),
    'The bundle finish hook did not mutate the immediate source array.');
  const firstBundle = renderObjectEvidence(renderer, target, root, mesh, camera);
  const firstEncoderCount = instrumentation.bundleEncoders.length;
  const firstBundleGpu = firstBundle.renderBundleData.bundleGPU;

  const secondOutput = await renderAndRead(
    renderer,
    scene,
    camera,
    target,
    instrumentation,
    'instrumented-immediate-second',
  );
  const secondBundle = renderObjectEvidence(renderer, target, root, mesh, camera);
  const secondEncoderCount = instrumentation.bundleEncoders.length;
  const secondBundleGpu = secondBundle.renderBundleData.bundleGPU;

  const drawGpuBuffer = renderer.backend.get(indirectAttribute)?.buffer ?? null;
  const drawGpuBufferId = instrumentation.identifyObject(
    drawGpuBuffer,
    'gpu-buffer',
  );
  const readbackAttributeIdentityExact = geometry.getIndirect() === indirectAttribute;
  const gpuCommandBuffer = new Uint32Array(
    await renderer.getArrayBufferAsync(indirectAttribute),
  );
  const cpuWordsFour = Array.from(
    { length: DRAW_COUNT },
    (_, draw) => commands[draw * INDIRECT_WORD_COUNT + 4],
  );
  const gpuWordsFour = Array.from(
    { length: DRAW_COUNT },
    (_, draw) => gpuCommandBuffer[draw * INDIRECT_WORD_COUNT + 4],
  );
  const commandEvidence = {
    exact: exactArray(gpuCommandBuffer, commands)
      && cpuWordsFour.every((value) => value === 0)
      && gpuWordsFour.every((value) => value === 0)
      && readbackAttributeIdentityExact
      && typeof drawGpuBufferId === 'string',
    drawCount: DRAW_COUNT,
    strideBytes: INDIRECT_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT,
    indirectOffsets,
    cpuCommands: Array.from(commands),
    gpuCommands: Array.from(gpuCommandBuffer),
    cpuWordsFour,
    gpuWordsFour,
    readbackAttributeIdentityExact,
    drawGpuBufferId,
  };
  const observedShader = shaderEvidence(
    secondBundle.vertexShader,
    secondBundle.fragmentShader,
  );
  observedShader.vertexSha256 = await sha256Text(observedShader.vertexShader);
  observedShader.fragmentSha256 = await sha256Text(observedShader.fragmentShader);

  const immediateTopologyEvidence = {
    exact: root.isBundleGroup === true
      && root.static === true
      && root.children.length === 1
      && root.children[0] === mesh
      && geometry.indirectOffset.length === DRAW_COUNT
      && geometry.indirectImmediateBases === immediateBases,
    bundleGroupCount: 1,
    meshCount: root.children.length,
    indexed: geometry.index !== null,
    indirectOffsetCount: geometry.indirectOffset.length,
    indirectImmediateBaseCount: geometry.indirectImmediateBases.length,
    bucketBaseAttributePresent: geometry.getAttribute('bucketBase') !== undefined,
  };

  // Create a distinct non-immediate pipeline after the immediate pipeline.
  // This proves the following live control pipeline was created with zero
  // immediate bytes; it does not claim to observe the descriptor reset call.
  mesh.material = controlMaterial;
  geometry.setIndirect(indirectAttribute, indirectOffsets, null);
  root.needsUpdate = true;
  await renderAndRead(
    renderer,
    scene,
    camera,
    target,
    instrumentation,
    'instrumented-non-immediate-control',
  );
  const postImmediateControlEncoderCount = instrumentation.bundleEncoders.length;

  const observedEncoder = encoderEvidence(
    instrumentation,
    indirectOffsets,
    drawGpuBufferId,
  );
  commandEvidence.encoderDrawGpuBufferIdentityExact =
    observedEncoder.drawBufferIdentityExact;
  commandEvidence.encoderDrawGpuBufferIds = observedEncoder.drawBufferIds;
  commandEvidence.exact = commandEvidence.exact
    && observedEncoder.drawBufferIdentityExact;
  const observedPipeline = pipelineEvidence(instrumentation, {
    vertexShader: secondBundle.vertexShader,
    fragmentShader: secondBundle.fragmentShader,
    pipelineGpu: secondBundle.pipelineGpu,
    bundleTrace: observedEncoder.trace,
  });

  const canonicalSha256 = await sha256Bytes(expectedCanonical);
  const mutatedSha256 = await sha256Bytes(expectedMutated);
  const firstSha256 = await sha256Bytes(firstOutput);
  const secondSha256 = await sha256Bytes(secondOutput);
  const challengeDifferingPixelCount = Array.from(
    { length: TARGET_WIDTH },
    (_, pixel) => Number(
      expectedCanonical[pixel * 4] !== expectedMutated[pixel * 4],
    ),
  ).reduce((sum, value) => sum + value, 0);
  const outputEvidence = {
    exact: exactArray(firstOutput, expectedCanonical)
      && exactArray(secondOutput, expectedCanonical)
      && exactArray(firstOutput, secondOutput)
      && !exactArray(firstOutput, expectedMutated)
      && challengeDifferingPixelCount === TARGET_WIDTH,
    encoding: 'rgba8unorm-red-object-id-plus-one-at-draw-ordinal-instance-pixel',
    pixelMapping: 'pixel = drawOrdinal * instancesPerDraw + localInstance',
    challengeDifferingPixelCount,
    expectedCanonical: Array.from(expectedCanonical),
    expectedMutated: Array.from(expectedMutated),
    firstObserved: Array.from(firstOutput),
    secondObserved: Array.from(secondOutput),
    canonicalSha256,
    mutatedSha256,
    firstSha256,
    secondSha256,
    firstMismatchCount: firstOutput.reduce(
      (count, value, index) => count + Number(value !== expectedCanonical[index]),
      0,
    ),
    secondMismatchCount: secondOutput.reduce(
      (count, value, index) => count + Number(value !== expectedCanonical[index]),
      0,
    ),
  };
  const cachedBundleGpuId = instrumentation.identifyObject(
    firstBundleGpu,
    'render-bundle',
  );
  const firstExecutions = instrumentation.renderPassExecutions.filter(
    (execution) => execution.capturePhase === 'instrumented-immediate-first',
  );
  const secondExecutions = instrumentation.renderPassExecutions.filter(
    (execution) => execution.capturePhase === 'instrumented-immediate-second',
  );
  const renderPassExecutionExact = typeof cachedBundleGpuId === 'string'
    && firstExecutions.length === 1
    && secondExecutions.length === 1
    && exactArray(firstExecutions[0].bundleIds, [cachedBundleGpuId])
    && exactArray(secondExecutions[0].bundleIds, [cachedBundleGpuId])
    && observedEncoder.trace.bundleId === cachedBundleGpuId;
  const bundleEvidence = {
    exact: firstBundle.summary.renderObjectCount === 1
      && firstBundle.summary.bundleVersionExact
      && firstBundle.summary.pipelineImmediateSize === 4
      && firstBundle.summary.immediateBasesIdentityExact
      && firstBundle.renderBundle === secondBundle.renderBundle
      && firstBundleGpu === secondBundleGpu
      && firstEncoderCount === 1
      && secondEncoderCount === firstEncoderCount
      && postImmediateControlEncoderCount === secondEncoderCount + 1
      && renderPassExecutionExact
      && root.children.length === 1
      && root.children[0] === mesh,
    bundleGroupCount: 1,
    meshCount: root.children.length,
    rootStatic: root.static,
    first: firstBundle.summary,
    second: secondBundle.summary,
    renderBundleIdentityStable: firstBundle.renderBundle === secondBundle.renderBundle,
    bundleGpuIdentityStable: firstBundleGpu === secondBundleGpu,
    cachedBundleGpuId,
    finishBundleId: observedEncoder.trace.bundleId,
    renderPassExecutionExact,
    firstRenderPassExecutions: firstExecutions,
    secondRenderPassExecutions: secondExecutions,
    allRenderPassExecutions: instrumentation.renderPassExecutions,
    postImmediateControlEncoderCount,
    encoderCountAfterFirstRender: firstEncoderCount,
    encoderCountAfterSecondRender: secondEncoderCount,
    sourceBasesAfterFinish: Array.from(immediateBases),
  };

  const uninstrumentedSnapshot = await runUninstrumentedSnapshotCanary(
    device,
    expectedCanonical,
    expectedMutated,
  );

  state.capabilities.renderBundleSetImmediates =
    observedEncoder.trace.callableSetImmediates;
  state.evidence = {
    topology: immediateTopologyEvidence,
    shader: observedShader,
    pipeline: observedPipeline,
    commands: commandEvidence,
    encoder: observedEncoder,
    output: outputEvidence,
    bundle: bundleEvidence,
    uninstrumentedSnapshot,
  };

  requireCondition(state.evidence.topology.exact
    && state.evidence.topology.bucketBaseAttributePresent === false,
  'One-bundle/one-mesh topology evidence failed.', state.evidence.topology);
  requireCondition(observedShader.exact,
    'Generated WGSL immediate-data evidence failed.', observedShader);
  requireCondition(observedPipeline.exact,
    'Pipeline-layout immediateSize evidence failed.', observedPipeline);
  requireCondition(secondBundle.summary.pipelineImmediateSize === 4,
    'Three.js pipeline data did not retain immediateSize four.', secondBundle.summary);
  requireCondition(commandEvidence.exact,
    'Indirect command or zero-word-four evidence failed.', commandEvidence);
  requireCondition(observedEncoder.exact,
    'Bundle setImmediates ordering/value evidence failed.', observedEncoder);
  requireCondition(outputEvidence.exact,
    'Exact address/output or bundle snapshot evidence failed.', outputEvidence);
  requireCondition(bundleEvidence.exact,
    'Render-bundle identity/reuse evidence failed.', bundleEvidence);
  requireCondition(uninstrumentedSnapshot.exact,
    'Uninstrumented bundle snapshot evidence failed.', uninstrumentedSnapshot);

  await device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 0));
  requireCondition(state.gpuErrors.uncaptured.length === 0
    && state.gpuErrors.unexpectedDeviceLosses.length === 0,
  'WebGPU errors or unexpected device loss were observed.', state.gpuErrors);
}

function cleanup(state) {
  state.resources.target?.dispose();
  state.resources.geometry?.dispose();
  state.resources.controlMaterial?.dispose();
  state.resources.immediateMaterial?.dispose();
  state.resources.indirectAttribute?.dispose?.();
  state.resources.visibleIdsAttribute?.dispose?.();
  state.resources.renderer?.dispose();
  if (state.resources.device !== null && state.resources.device !== undefined) {
    state.intentionalDeviceDestroy = true;
    state.resources.device.destroy();
  }
}

export async function runThreeImmediateOverlayCanary() {
  const state = {
    targetConfiguration: null,
    capabilities: {
      wgslLanguageFeatures: [],
      immediateAddressSpace: false,
      nonFallbackAdapter: false,
      adapterMaxImmediateSize: null,
      adapterFeatures: [],
      deviceMaxImmediateSize: null,
      deviceFeatures: [],
      renderPassSetImmediates: false,
      renderBundleSetImmediates: false,
    },
    adapterInfo: null,
    errorObservationScope: null,
    runtimeOverlaySignature: null,
    instrumentation: null,
    evidence: null,
    gpuErrors: {
      uncaptured: [],
      unexpectedDeviceLosses: [],
    },
    resources: {
      renderer: null,
      device: null,
    },
    intentionalDeviceDestroy: false,
  };
  let failure = null;
  try {
    await executeCanary(state);
  } catch (error) {
    failure = serializeError(error);
  }

  const result = {
    schemaVersion: 1,
    kind: THREE_IMMEDIATE_OVERLAY_CANARY_KIND,
    status: failure === null
      ? 'development-checks-complete'
      : (failure.message.includes('unavailable') ? 'unsupported' : 'development-checks-failed'),
    executionMode: 'timing-free-development-correctness-canary',
    target: state.targetConfiguration === null ? null : {
      schemaVersion: state.targetConfiguration.schemaVersion,
      key: state.targetConfiguration.key,
      sourceFamily: state.targetConfiguration.sourceFamily,
      expectedRevision: state.targetConfiguration.expectedRevision,
    },
    scope: state.targetConfiguration?.scope ?? 'unbound canary target',
    analysisEligible: false,
    timingCaptured: false,
    efficacyEvaluated: false,
    fullPhaseZeroPass: false,
    fullPhaseZeroStatus: 'not-evaluated',
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
    runtimeOverlaySignature: state.runtimeOverlaySignature,
    evidence: state.evidence,
    gpuErrors: state.gpuErrors,
    failure,
  };

  cleanup(state);
  return result;
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  const published = { ready: false, result: null };
  window.__THREE_IMMEDIATE_OVERLAY_CANARY__ = published;
  const statusElement = document.getElementById('status');
  void runThreeImmediateOverlayCanary().then((result) => {
    published.result = result;
    published.ready = true;
    if (statusElement !== null) {
      statusElement.textContent = result.status === 'development-checks-complete'
        ? 'Minimal overlay integration checks completed; full Phase 0 remains unevaluated.'
        : `Overlay canary stopped: ${result.failure?.message ?? result.status}`;
    }
  }).catch((error) => {
    published.result = {
      schemaVersion: 1,
      kind: THREE_IMMEDIATE_OVERLAY_CANARY_KIND,
      status: 'development-checks-failed',
      executionMode: 'timing-free-development-correctness-canary',
      analysisEligible: false,
      timingCaptured: false,
      efficacyEvaluated: false,
      fullPhaseZeroPass: false,
      fullPhaseZeroStatus: 'not-evaluated',
      target: null,
      scope: 'unbound canary target',
      coverage: {
        implemented: [],
        deferred: [],
      },
      failure: serializeError(error),
    };
    published.ready = true;
    if (statusElement !== null) statusElement.textContent = 'Overlay canary failed closed.';
  });
}
