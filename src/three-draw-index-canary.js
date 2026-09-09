import {
  ArrayCamera,
  BatchedMesh,
  BufferGeometry,
  BundleGroup,
  DoubleSide,
  Float32BufferAttribute,
  getConsoleFunction,
  IndirectStorageBufferAttribute,
  Matrix4,
  Mesh,
  MeshBasicNodeMaterial,
  NoBlending,
  NoColorSpace,
  OrthographicCamera,
  RenderTarget,
  REVISION,
  RGBAFormat,
  Scene,
  Uint16BufferAttribute,
  UnsignedByteType,
  Vector4,
  WebGPUBackend,
  WebGPURenderer,
  WGSLNodeBuilder,
  setConsoleFunction,
} from 'three/webgpu';
import {
  batchIndirectIndex,
  drawIndex,
  float,
  uint,
  vec4,
} from 'three/tsl';
import {
  BATCH_ORDER,
  DRAW_COUNT,
  DRAW_INDEX_SYMBOL,
  INDIRECT_OFFSETS,
  INDIRECT_WORD_COUNT,
  RERECORD_INDIRECT_OFFSETS,
  TARGET_HEIGHT,
  TARGET_WIDTH,
  THREE_DRAW_INDEX_CANARY_KIND,
  WGSL_IMMEDIATE_FEATURE,
  createIndirectCommands,
  expectedBatchedControlOutput,
  expectedBatchedOutput,
  expectedDirectOutput,
  expectedIndirectOutput,
} from './three-draw-index-canary-contract.js';

export {
  BATCH_ORDER,
  DRAW_COUNT,
  DRAW_INDEX_SYMBOL,
  INDIRECT_OFFSETS,
  INDIRECT_WORD_COUNT,
  RERECORD_INDIRECT_OFFSETS,
  TARGET_HEIGHT,
  TARGET_WIDTH,
  THREE_DRAW_INDEX_CANARY_KIND,
  WGSL_IMMEDIATE_FEATURE,
  createIndirectCommands,
  expectedBatchedControlOutput,
  expectedBatchedOutput,
  expectedDirectOutput,
  expectedIndirectOutput,
};

const TRACKED_ENCODER_METHODS = Object.freeze([
  'setPipeline',
  'setBindGroup',
  'setIndexBuffer',
  'setVertexBuffer',
  'setImmediates',
  'draw',
  'drawIndexed',
  'drawIndirect',
  'drawIndexedIndirect',
  'executeBundles',
  'end',
]);

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

function exactArray(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function toHex(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

async function sha256Bytes(bytes) {
  return toHex(await crypto.subtle.digest('SHA-256', bytes));
}

async function outputEvidence(observed, expected, encoding) {
  const observedBytes = Uint8Array.from(observed);
  const expectedBytes = Uint8Array.from(expected);
  return {
    exact: exactArray(observedBytes, expectedBytes),
    encoding,
    expected: [...expectedBytes],
    observed: [...observedBytes],
    expectedSha256: await sha256Bytes(expectedBytes),
    observedSha256: await sha256Bytes(observedBytes),
    mismatchCount: observedBytes.reduce(
      (count, value, index) => count + Number(value !== expectedBytes[index]),
      0,
    ),
  };
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

function snapshotImmediateCall(args) {
  const [rangeOffset, data, suppliedDataOffset, suppliedDataSize] = args;
  const dataOffset = suppliedDataOffset ?? 0;
  const sourceElementCount = ArrayBuffer.isView(data) ? data.length : data.byteLength;
  const dataSize = suppliedDataSize ?? (sourceElementCount - dataOffset);
  return {
    rangeOffset,
    sourceType: data?.constructor?.name ?? null,
    sourceElementCount,
    dataOffset,
    dataSize,
    selectedValues: data instanceof Uint32Array
      ? Array.from(data.slice(dataOffset, dataOffset + dataSize))
      : null,
  };
}

function installDeviceInstrumentation(device) {
  const objectIds = new WeakMap();
  const shaderModuleObjects = [];
  const errorScopePromises = [];
  let nextObjectId = 1;
  let nextSequence = 1;
  let capturePhase = null;
  let errorScopeDepth = 0;
  const errorScopeFilters = [];
  const evidence = {
    shaderModules: [],
    pipelineLayouts: [],
    renderPipelines: [],
    renderPasses: [],
    renderBundles: [],
    errorScopes: [],
    patchedDeviceMethods: [],
  };

  const objectId = (object, prefix) => {
    if (object === null || (typeof object !== 'object' && typeof object !== 'function')) {
      return null;
    }
    if (!objectIds.has(object)) objectIds.set(object, `${prefix}-${nextObjectId++}`);
    return objectIds.get(object);
  };

  const instrumentEncoder = (encoder, type, descriptor) => {
    const trace = {
      encoderId: objectId(encoder, type === 'bundle' ? 'bundle-encoder' : 'render-pass'),
      type,
      creationPhase: capturePhase,
      descriptor: {
        label: descriptor?.label ?? null,
        colorFormatCount: descriptor?.colorFormats?.length
          ?? descriptor?.colorAttachments?.length
          ?? null,
        depthStencilFormat: descriptor?.depthStencilFormat ?? null,
        sampleCount: descriptor?.sampleCount ?? 1,
      },
      callableSetImmediates: typeof encoder.setImmediates === 'function',
      events: [],
      bundleId: null,
    };
    const collection = type === 'bundle' ? evidence.renderBundles : evidence.renderPasses;
    collection.push(trace);

    for (const methodName of TRACKED_ENCODER_METHODS) {
      if (typeof encoder[methodName] !== 'function') continue;
      replaceCallable(encoder, methodName, (original) => (...args) => {
        const event = {
          sequence: nextSequence++,
          phase: capturePhase,
          method: methodName,
        };
        if (methodName === 'setImmediates') {
          Object.assign(event, snapshotImmediateCall(args));
        } else if (methodName === 'setPipeline') {
          event.pipelineId = objectId(args[0], 'render-pipeline');
        } else if (methodName === 'draw' || methodName === 'drawIndexed') {
          event.arguments = args.map((value) => Number(value));
        } else if (methodName === 'drawIndirect'
          || methodName === 'drawIndexedIndirect') {
          event.bufferId = objectId(args[0], 'gpu-buffer');
          event.indirectOffset = Number(args[1]);
        } else if (methodName === 'executeBundles') {
          event.bundleIds = Array.from(args[0] ?? [], (bundle) => (
            objectId(bundle, 'render-bundle')
          ));
        }
        const value = original(...args);
        trace.events.push(event);
        return value;
      });
    }

    if (type === 'bundle') {
      replaceCallable(encoder, 'finish', (original) => (...args) => {
        const bundle = original(...args);
        trace.bundleId = objectId(bundle, 'render-bundle');
        trace.events.push({
          sequence: nextSequence++,
          phase: capturePhase,
          method: 'finish',
        });
        return bundle;
      });
    }
    return trace;
  };

  replaceCallable(device, 'createShaderModule', (original) => (descriptor) => {
    const module = original(descriptor);
    shaderModuleObjects.push(module);
    evidence.shaderModules.push({
      sequence: nextSequence++,
      creationPhase: capturePhase,
      moduleId: objectId(module, 'shader-module'),
      label: descriptor?.label ?? null,
      code: descriptor?.code ?? null,
    });
    return module;
  });
  evidence.patchedDeviceMethods.push('createShaderModule');

  replaceCallable(device, 'createPipelineLayout', (original) => (descriptor) => {
    const record = {
      sequence: nextSequence++,
      creationPhase: capturePhase,
      label: descriptor?.label ?? null,
      hasOwnImmediateSize: Object.hasOwn(descriptor, 'immediateSize'),
      immediateSize: descriptor?.immediateSize ?? null,
      bindGroupLayoutCount: descriptor?.bindGroupLayouts?.length ?? null,
      layoutId: null,
    };
    const layout = original(descriptor);
    record.layoutId = objectId(layout, 'pipeline-layout');
    evidence.pipelineLayouts.push(record);
    return layout;
  });
  evidence.patchedDeviceMethods.push('createPipelineLayout');

  for (const methodName of ['createRenderPipeline', 'createRenderPipelineAsync']) {
    if (typeof device[methodName] !== 'function') continue;
    replaceCallable(device, methodName, (original) => (descriptor) => {
      const record = {
        sequence: nextSequence++,
        creationPhase: capturePhase,
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
        void pipelineOrPromise.then((pipeline) => {
          record.pipelineId = objectId(pipeline, 'render-pipeline');
        }, () => undefined);
      }
      return pipelineOrPromise;
    });
    evidence.patchedDeviceMethods.push(methodName);
  }

  replaceCallable(device, 'createCommandEncoder', (original) => (descriptor) => {
    const commandEncoder = original(descriptor);
    replaceCallable(commandEncoder, 'beginRenderPass', (beginRenderPass) => (
      renderPassDescriptor
    ) => {
      const renderPass = beginRenderPass(renderPassDescriptor);
      instrumentEncoder(renderPass, 'pass', renderPassDescriptor);
      return renderPass;
    });
    return commandEncoder;
  });
  evidence.patchedDeviceMethods.push('createCommandEncoder');

  replaceCallable(device, 'createRenderBundleEncoder', (original) => (descriptor) => {
    const encoder = original(descriptor);
    instrumentEncoder(encoder, 'bundle', descriptor);
    return encoder;
  });
  evidence.patchedDeviceMethods.push('createRenderBundleEncoder');

  replaceCallable(device, 'pushErrorScope', (original) => (filter) => {
    const value = original(filter);
    errorScopeFilters.push(String(filter));
    errorScopeDepth = errorScopeFilters.length;
    evidence.errorScopes.push({
      sequence: nextSequence++,
      phase: capturePhase,
      operation: 'push',
      filter: String(filter),
      depthAfterCall: errorScopeDepth,
    });
    return value;
  });
  evidence.patchedDeviceMethods.push('pushErrorScope');

  replaceCallable(device, 'popErrorScope', (original) => () => {
    const depthBeforeCall = errorScopeDepth;
    const poppedFilter = errorScopeFilters.at(-1) ?? null;
    const promise = original();
    errorScopeFilters.pop();
    errorScopeDepth = errorScopeFilters.length;
    const record = {
      sequence: nextSequence++,
      phase: capturePhase,
      operation: 'pop',
      poppedFilter,
      depthBeforeCall,
      depthAfterCall: errorScopeDepth,
      resolvedError: null,
      rejected: null,
      settled: false,
    };
    evidence.errorScopes.push(record);
    const trackedPromise = promise.then((error) => {
      record.resolvedError = error === null ? null : serializeGpuError(error);
      record.settled = true;
      return error;
    }, (error) => {
      record.rejected = serializeError(error);
      record.settled = true;
      throw error;
    });
    errorScopePromises.push(trackedPromise);
    return trackedPromise;
  });
  evidence.patchedDeviceMethods.push('popErrorScope');

  Object.defineProperties(evidence, {
    identifyObject: { value: objectId },
    setCapturePhase: { value: (phase) => { capturePhase = phase; } },
    getErrorScopeDepth: { value: () => errorScopeDepth },
    collectShaderCompilationInfo: {
      value: async () => Promise.all(shaderModuleObjects.map(async (module) => {
        const moduleId = objectId(module, 'shader-module');
        if (typeof module.getCompilationInfo !== 'function') {
          return { moduleId, available: false, messages: [] };
        }
        const info = await module.getCompilationInfo();
        return {
          moduleId,
          available: true,
          messages: Array.from(info.messages ?? [], (message) => ({
            type: String(message.type),
            message: String(message.message).slice(0, 4_096),
            lineNum: Number(message.lineNum),
            linePos: Number(message.linePos),
            offset: Number(message.offset),
            length: Number(message.length),
          })),
        };
      })),
    },
    settlePendingErrorScopes: {
      value: async () => Promise.allSettled([...errorScopePromises]),
    },
  });
  return evidence;
}

function createCamera(width = TARGET_WIDTH, height = TARGET_HEIGHT) {
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 1;
  camera.viewport = new Vector4(0, 0, width, height);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return camera;
}

function createTarget(width = TARGET_WIDTH, height = TARGET_HEIGHT, depth = 1) {
  const target = new RenderTarget(width, height, {
    format: RGBAFormat,
    type: UnsignedByteType,
    colorSpace: NoColorSpace,
    depth,
    depthBuffer: depth > 1,
    useArrayDepthTexture: depth > 1,
    samples: 0,
  });
  target.texture.colorSpace = NoColorSpace;
  return target;
}

function configureMaterial(material, name) {
  material.name = name;
  material.depthWrite = false;
  material.depthTest = false;
  material.toneMapped = false;
  material.blending = NoBlending;
  material.transparent = false;
  material.premultipliedAlpha = false;
  material.fog = false;
  material.side = DoubleSide;
  material.forceSinglePass = true;
  return material;
}

function createDrawIndexMaterial(name, includeBatchIdentity = false) {
  const material = configureMaterial(new MeshBasicNodeMaterial(), name);
  const red = float(drawIndex).add(float(1)).div(255);
  const green = includeBatchIdentity
    ? float(batchIndirectIndex.add(uint(1))).div(255)
    : float(0);
  material.fragmentNode = vec4(red, green, 0, 1);
  return material;
}

function createBatchedControlMaterial(name) {
  const material = configureMaterial(new MeshBasicNodeMaterial(), name);
  material.fragmentNode = vec4(
    0,
    float(batchIndirectIndex.add(uint(1))).div(255),
    0,
    1,
  );
  return material;
}

function triangleAt(centerX, halfWidth) {
  return [
    centerX - halfWidth, -0.8, 0,
    centerX + halfWidth, -0.8, 0,
    centerX, 0.8, 0,
  ];
}

function createDirectGeometry() {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(triangleAt(0, 0.75), 3));
  geometry.setIndex(new Uint16BufferAttribute([0, 1, 2], 1));
  return geometry;
}

function createGridGeometry() {
  const positions = [];
  const indices = [];
  const halfPixel = 1 / TARGET_WIDTH;
  for (let index = 0; index < DRAW_COUNT; index += 1) {
    const centerX = ((index + 0.5) * 2 / TARGET_WIDTH) - 1;
    positions.push(...triangleAt(centerX, halfPixel * 0.75));
    indices.push(index * 3, index * 3 + 1, index * 3 + 2);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setIndex(new Uint16BufferAttribute(indices, 1));
  return geometry;
}

function createIndirectResources(material, { bundle = false } = {}) {
  const commands = createIndirectCommands();
  const indirect = new IndirectStorageBufferAttribute(commands, INDIRECT_WORD_COUNT);
  const geometry = createGridGeometry();
  geometry.setIndirect(indirect, [...INDIRECT_OFFSETS]);
  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  const scene = new Scene();
  let root = null;
  if (bundle) {
    root = new BundleGroup();
    root.name = 'DrawIndexCanaryBundle';
    root.static = true;
    root.add(mesh);
    scene.add(root);
  } else {
    scene.add(mesh);
  }
  return { commands, geometry, indirect, mesh, root, scene };
}

function createBatchedResources(explicitMaterial, controlMaterial) {
  const sourceGeometry = new BufferGeometry();
  sourceGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(triangleAt(0, (1 / TARGET_WIDTH) * 0.75), 3),
  );
  sourceGeometry.setIndex(new Uint16BufferAttribute([0, 1, 2], 1));
  const batchedMesh = new BatchedMesh(DRAW_COUNT, 3, 3, explicitMaterial);
  batchedMesh.name = 'DrawIndexCanaryBatchedMesh';
  batchedMesh.frustumCulled = false;
  batchedMesh.perObjectFrustumCulled = false;
  batchedMesh.sortObjects = true;
  const geometryId = batchedMesh.addGeometry(sourceGeometry);
  const matrix = new Matrix4();
  for (let instanceId = 0; instanceId < DRAW_COUNT; instanceId += 1) {
    const createdId = batchedMesh.addInstance(geometryId);
    requireCondition(createdId === instanceId, 'Unexpected BatchedMesh instance ID.');
    const centerX = ((instanceId + 0.5) * 2 / TARGET_WIDTH) - 1;
    batchedMesh.setMatrixAt(instanceId, matrix.makeTranslation(centerX, 0, 0));
  }
  const rank = new Map(BATCH_ORDER.map((instanceId, drawOrdinal) => [
    instanceId,
    drawOrdinal,
  ]));
  batchedMesh.setCustomSort((list) => {
    list.sort((left, right) => rank.get(left.index) - rank.get(right.index));
  });
  const scene = new Scene();
  scene.add(batchedMesh);
  return {
    batchedMesh,
    controlMaterial,
    explicitMaterial,
    scene,
    sourceGeometry,
  };
}

async function renderAndRead(renderer, scene, camera, target, instrumentation, phase) {
  instrumentation.setCapturePhase(phase);
  try {
    renderer.setRenderTarget(target);
    renderer.autoClear = true;
    renderer.render(scene, camera);
    await renderer.backend.device.queue.onSubmittedWorkDone();
    const output = await renderer.readRenderTargetPixelsAsync(
      target,
      0,
      0,
      target.width,
      target.height,
    );
    requireCondition(output instanceof Uint8Array, 'Expected Uint8Array render readback.');
    return output;
  } finally {
    instrumentation.setCapturePhase(null);
  }
}

async function renderAndReadLayers(
  renderer,
  scene,
  camera,
  target,
  instrumentation,
  phase,
) {
  instrumentation.setCapturePhase(phase);
  try {
    renderer.setRenderTarget(target);
    renderer.autoClear = true;
    renderer.render(scene, camera);
    await renderer.backend.device.queue.onSubmittedWorkDone();
    const layers = [];
    for (let layer = 0; layer < target.depth; layer += 1) {
      const output = await renderer.readRenderTargetPixelsAsync(
        target,
        0,
        0,
        target.width,
        target.height,
        0,
        layer,
      );
      requireCondition(output instanceof Uint8Array, 'Expected Uint8Array layer readback.');
      layers.push(output);
    }
    return layers;
  } finally {
    instrumentation.setCapturePhase(null);
  }
}

function relevantEvents(trace) {
  return trace.events.filter((event) => [
    'setPipeline',
    'setImmediates',
    'draw',
    'drawIndexed',
    'drawIndirect',
    'drawIndexedIndirect',
    'executeBundles',
    'finish',
  ].includes(event.method));
}

function tracesForPhase(instrumentation, phase) {
  const serializeTrace = (trace) => ({
    ...trace,
    events: relevantEvents(trace),
  });
  return {
    passes: instrumentation.renderPasses
      .filter((trace) => trace.creationPhase === phase
        || trace.events.some((event) => event.phase === phase))
      .map(serializeTrace),
    bundles: instrumentation.renderBundles
      .filter((trace) => trace.creationPhase === phase
        || trace.events.some((event) => event.phase === phase))
      .map(serializeTrace),
  };
}

function executedBundleIds(instrumentation, phase) {
  return instrumentation.renderPasses.flatMap((trace) => trace.events
    .filter((event) => event.phase === phase && event.method === 'executeBundles')
    .flatMap((event) => event.bundleIds));
}

function runtimeSignatureEvidence() {
  const setDrawIndexSource = WebGPUBackend.prototype._setDrawIndex?.toString() ?? '';
  const drawSource = WebGPUBackend.prototype._draw?.toString() ?? '';
  const getDrawIndexSource = WGSLNodeBuilder.prototype.getDrawIndex?.toString() ?? '';
  return {
    revision: REVISION,
    drawIndexNode: drawIndex?.isNode === true,
    backendHasDrawIndexSetter: /setImmediates\(\s*0,\s*this\._drawIndexData,\s*0,\s*1\s*\)/
      .test(setDrawIndexSource),
    backendDrawUsesPipelineMetadata: /pipelineData\.immediateSize/.test(drawSource),
    builderReturnsImmediateSymbol: /return\s+drawIndexName\s*;/.test(getDrawIndexSource),
    builderChecksDrawIndexAvailability: /isAvailable\(\s*'drawIndex'\s*\)/
      .test(getDrawIndexSource),
  };
}

function unsupported(state, stage, reason, detail = null) {
  state.preflight.capabilityDecision = 'unsupported';
  state.preflight.unsupportedStage = stage;
  state.preflight.unsupportedReason = reason;
  state.preflight.unsupportedDetail = detail;
  state.preflight.shaderModulesAtCapabilityDecision =
    state.resources.instrumentation?.shaderModules?.length ?? 0;
  return false;
}

async function runScenarios(state) {
  const {
    renderer,
    instrumentation,
  } = state.resources;
  const camera = createCamera();

  const directMaterial = createDrawIndexMaterial('DrawIndexCanaryDirectFragment');
  const directGeometry = createDirectGeometry();
  const directMesh = new Mesh(directGeometry, directMaterial);
  directMesh.frustumCulled = false;
  const directScene = new Scene();
  directScene.add(directMesh);
  const directTarget = createTarget(1, 1);
  state.resources.disposables.push(directTarget, directGeometry, directMaterial);
  const directObserved = await renderAndRead(
    renderer,
    directScene,
    createCamera(1, 1),
    directTarget,
    instrumentation,
    'direct-fragment',
  );
  const direct = {
    output: await outputEvidence(
      directObserved,
      expectedDirectOutput(),
      'r=drawIndex+1, g=0; direct draw must expose zero through a flat fragment varying',
    ),
    traces: tracesForPhase(instrumentation, 'direct-fragment'),
  };
  requireCondition(direct.output.exact, 'Direct fragment drawIndex output mismatch.', direct);

  const bundleMaterial = createDrawIndexMaterial('DrawIndexCanaryIndirectBundle');
  const bundleResources = createIndirectResources(bundleMaterial, { bundle: true });
  const bundleTarget = createTarget();
  state.resources.disposables.push(
    bundleTarget,
    bundleResources.geometry,
    bundleResources.indirect,
    bundleMaterial,
  );
  state.resources.indirectAttributes.push(bundleResources.indirect);
  const bundleFirstObserved = await renderAndRead(
    renderer,
    bundleResources.scene,
    camera,
    bundleTarget,
    instrumentation,
    'indirect-bundle-first',
  );
  const firstBundleIds = executedBundleIds(instrumentation, 'indirect-bundle-first');
  const bundleCachedObserved = await renderAndRead(
    renderer,
    bundleResources.scene,
    camera,
    bundleTarget,
    instrumentation,
    'indirect-bundle-cached',
  );
  const cachedBundleIds = executedBundleIds(instrumentation, 'indirect-bundle-cached');
  bundleResources.geometry.setIndirect(
    bundleResources.indirect,
    [...RERECORD_INDIRECT_OFFSETS],
  );
  bundleResources.root.needsUpdate = true;
  const bundleRerecordedObserved = await renderAndRead(
    renderer,
    bundleResources.scene,
    camera,
    bundleTarget,
    instrumentation,
    'indirect-bundle-rerecord',
  );
  const rerecordedBundleIds = executedBundleIds(
    instrumentation,
    'indirect-bundle-rerecord',
  );
  const indirectBundle = {
    commands: Array.from(bundleResources.commands),
    initialOffsets: [...INDIRECT_OFFSETS],
    rerecordOffsets: [...RERECORD_INDIRECT_OFFSETS],
    rootVersion: bundleResources.root.version,
    first: {
      output: await outputEvidence(
        bundleFirstObserved,
        expectedIndirectOutput(),
        'r=drawIndex+1 at the pixel selected by each indexed-indirect command',
      ),
      executedBundleIds: firstBundleIds,
      traces: tracesForPhase(instrumentation, 'indirect-bundle-first'),
    },
    cached: {
      output: await outputEvidence(
        bundleCachedObserved,
        expectedIndirectOutput(),
        'cached bundle must preserve the first recording exactly',
      ),
      executedBundleIds: cachedBundleIds,
      traces: tracesForPhase(instrumentation, 'indirect-bundle-cached'),
    },
    rerecorded: {
      output: await outputEvidence(
        bundleRerecordedObserved,
        expectedIndirectOutput(RERECORD_INDIRECT_OFFSETS),
        're-recorded reordered commands still receive compact ordinals 0..3',
      ),
      executedBundleIds: rerecordedBundleIds,
      traces: tracesForPhase(instrumentation, 'indirect-bundle-rerecord'),
    },
  };
  requireCondition(
    indirectBundle.first.output.exact
      && indirectBundle.cached.output.exact
      && indirectBundle.rerecorded.output.exact
      && firstBundleIds.length === 1
      && cachedBundleIds.length === 1
      && rerecordedBundleIds.length === 1
      && firstBundleIds[0] === cachedBundleIds[0]
      && rerecordedBundleIds[0] !== firstBundleIds[0]
      && indirectBundle.rootVersion === 1,
    'Indirect render-bundle cache/re-record evidence failed.',
    indirectBundle,
  );

  const explicitBatchMaterial = createDrawIndexMaterial(
    'DrawIndexCanaryBatchedExplicit',
    true,
  );
  const batchControlMaterial = createBatchedControlMaterial(
    'DrawIndexCanaryBatchedControl',
  );
  const batchResources = createBatchedResources(
    explicitBatchMaterial,
    batchControlMaterial,
  );
  const batchTarget = createTarget();
  state.resources.disposables.push(
    batchTarget,
    batchResources.sourceGeometry,
    explicitBatchMaterial,
    batchControlMaterial,
  );
  state.resources.batchedMeshes.push(batchResources.batchedMesh);
  const batchExplicitObserved = await renderAndRead(
    renderer,
    batchResources.scene,
    camera,
    batchTarget,
    instrumentation,
    'batched-explicit',
  );
  const observedBatchOrder = Array.from(
    batchResources.batchedMesh._indirectTexture.image.data.slice(0, DRAW_COUNT),
  );
  batchResources.batchedMesh.material = batchControlMaterial;
  const batchControlObserved = await renderAndRead(
    renderer,
    batchResources.scene,
    camera,
    batchTarget,
    instrumentation,
    'batched-control',
  );
  const batched = {
    requestedOrder: [...BATCH_ORDER],
    observedIndirectTextureOrder: observedBatchOrder,
    explicit: {
      output: await outputEvidence(
        batchExplicitObserved,
        expectedBatchedOutput(),
        'r=compact drawIndex+1, g=original batchIndirectIndex+1 at original-instance pixel',
      ),
      traces: tracesForPhase(instrumentation, 'batched-explicit'),
    },
    control: {
      output: await outputEvidence(
        batchControlObserved,
        expectedBatchedControlOutput(),
        'r=0, g=original batchIndirectIndex+1; default batching is immediate-free',
      ),
      traces: tracesForPhase(instrumentation, 'batched-control'),
    },
  };
  requireCondition(
    exactArray(observedBatchOrder, BATCH_ORDER)
      && batched.explicit.output.exact
      && batched.control.output.exact,
    'BatchedMesh drawIndex/batchIndirectIndex distinction failed.',
    batched,
  );

  const arrayMaterial = createDrawIndexMaterial('DrawIndexCanaryArrayCamera');
  const arrayResources = createIndirectResources(arrayMaterial);
  const arrayTarget = createTarget(TARGET_WIDTH, TARGET_HEIGHT, 2);
  const arrayCameras = [createCamera(), createCamera()];
  const arrayCamera = new ArrayCamera(arrayCameras);
  arrayCamera.updateMatrixWorld();
  state.resources.disposables.push(
    arrayTarget,
    arrayResources.geometry,
    arrayResources.indirect,
    arrayMaterial,
  );
  state.resources.indirectAttributes.push(arrayResources.indirect);
  const arrayObserved = await renderAndReadLayers(
    renderer,
    arrayResources.scene,
    arrayCamera,
    arrayTarget,
    instrumentation,
    'array-camera',
  );
  const arrayCameraEvidence = {
    cameraCount: arrayCameras.length,
    targetDepth: arrayTarget.depth,
    layers: await Promise.all(arrayObserved.map((observed, layer) => outputEvidence(
      observed,
      expectedIndirectOutput(),
      `array-camera layer ${layer}: r=drawIndex+1`,
    ))),
    traces: tracesForPhase(instrumentation, 'array-camera'),
  };
  requireCondition(
    arrayCameraEvidence.layers.length === 2
      && arrayCameraEvidence.layers.every((layer) => layer.exact),
    'ArrayCamera layer output failed.',
    arrayCameraEvidence,
  );

  state.scenarios = {
    directFragment: direct,
    indirectBundle,
    batched,
    arrayCamera: arrayCameraEvidence,
  };
}

async function executeCanary(state) {
  state.preflight.currentStage = 'secure-context';
  state.preflight.secureContext = globalThis.isSecureContext === true;
  if (!state.preflight.secureContext) {
    return unsupported(state, 'secure-context', 'A secure context is required.');
  }
  state.preflight.currentStage = 'navigator-gpu';
  if (typeof navigator?.gpu !== 'object') {
    return unsupported(state, 'navigator-gpu', 'navigator.gpu is unavailable.');
  }

  state.preflight.currentStage = 'wgsl-language-feature';
  state.capabilities.wgslLanguageFeatures = sortedFeatureNames(
    navigator.gpu.wgslLanguageFeatures,
  );
  state.capabilities.immediateAddressSpace =
    state.capabilities.wgslLanguageFeatures.includes(WGSL_IMMEDIATE_FEATURE);
  if (!state.capabilities.immediateAddressSpace) {
    return unsupported(
      state,
      'wgsl-language-feature',
      `${WGSL_IMMEDIATE_FEATURE} is unavailable.`,
    );
  }

  state.preflight.currentStage = 'adapter-request';
  state.preflight.adapterRequestAttempts = 1;
  state.preflight.requestAdapterArgumentsCount = 0;
  state.preflight.requestAdapterDescriptorSupplied = false;
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) {
    return unsupported(state, 'adapter-request', 'WebGPU adapter request returned null.');
  }
  state.preflight.adapterAcquisitions = 1;
  state.adapterInfo = await adapterInformation(adapter);
  state.capabilities.adapterFeatures = sortedFeatureNames(adapter.features);
  const adapterMaxImmediateSize = Number(adapter.limits?.maxImmediateSize);
  state.capabilities.adapterMaxImmediateSize = Number.isFinite(adapterMaxImmediateSize)
    ? adapterMaxImmediateSize
    : null;

  state.preflight.currentStage = 'device-request';
  state.preflight.deviceRequestAttempts = 1;
  state.preflight.requestDeviceArgumentsCount = 0;
  state.preflight.requestDeviceDescriptorSupplied = false;
  const device = await adapter.requestDevice();
  state.preflight.deviceAcquisitions = 1;
  state.resources.device = device;
  device.addEventListener('uncapturederror', (event) => {
    event.preventDefault();
    state.gpuErrors.uncaptured.push(serializeGpuError(event.error));
  });
  void device.lost.then((information) => {
    if (!state.resources.intentionalDeviceDestroy) {
      state.gpuErrors.unexpectedDeviceLosses.push({
        reason: String(information?.reason ?? 'unknown').slice(0, 256),
        message: String(information?.message ?? '').slice(0, 4_096),
      });
    }
  });
  const instrumentation = installDeviceInstrumentation(device);
  state.resources.instrumentation = instrumentation;
  state.instrumentation = instrumentation;
  state.preflight.instrumentationInstalledBeforeRenderer = true;
  state.capabilities.deviceFeatures = sortedFeatureNames(device.features);
  const deviceMaxImmediateSize = Number(device.limits?.maxImmediateSize);
  state.capabilities.deviceMaxImmediateSize = Number.isFinite(deviceMaxImmediateSize)
    ? deviceMaxImmediateSize
    : null;
  state.preflight.currentStage = 'device-limit';
  if (!Number.isInteger(state.capabilities.deviceMaxImmediateSize)
      || state.capabilities.deviceMaxImmediateSize < 4) {
    return unsupported(
      state,
      'device-limit',
      'The default device exposes fewer than four immediate bytes.',
      state.capabilities.deviceMaxImmediateSize,
    );
  }
  state.capabilities.renderPassSetImmediates =
    typeof globalThis.GPURenderPassEncoder?.prototype?.setImmediates === 'function';
  state.capabilities.renderBundleSetImmediates =
    typeof globalThis.GPURenderBundleEncoder?.prototype?.setImmediates === 'function';
  state.preflight.currentStage = 'encoder-surfaces';
  if (!state.capabilities.renderPassSetImmediates
      || !state.capabilities.renderBundleSetImmediates) {
    return unsupported(
      state,
      'encoder-surfaces',
      'The WebGPU render encoder immediate-data methods are unavailable.',
      {
        renderPassSetImmediates: state.capabilities.renderPassSetImmediates,
        renderBundleSetImmediates: state.capabilities.renderBundleSetImmediates,
      },
    );
  }

  state.preflight.capabilityDecision = 'supported';
  state.preflight.currentStage = 'renderer-and-shaders';
  state.preflight.shaderModulesAtCapabilityDecision = instrumentation.shaderModules.length;
  state.runtimeSignature = runtimeSignatureEvidence();
  requireCondition(
    state.runtimeSignature.revision === '186dev'
      && state.runtimeSignature.drawIndexNode
      && state.runtimeSignature.backendHasDrawIndexSetter
      && state.runtimeSignature.backendDrawUsesPipelineMetadata
      && state.runtimeSignature.builderReturnsImmediateSymbol
      && state.runtimeSignature.builderChecksDrawIndexAvailability,
    'The served runtime does not contain the pinned drawIndex overlay.',
    state.runtimeSignature,
  );

  state.resources.previousConsoleFunction = getConsoleFunction();
  setConsoleFunction((method, ...parameters) => {
    const record = {
      method: String(method),
      parameters: parameters.map((value) => String(value).slice(0, 4_096)),
    };
    state.threeConsole.push(record);
    if (method === 'error') state.threeConsoleErrors.push(record);
  });
  state.resources.consoleFunctionInstalled = true;

  const outerFilters = ['internal', 'out-of-memory', 'validation'];
  const successfulOuterFilters = [];
  let scenarioFailure = null;
  try {
    for (const filter of outerFilters) {
      device.pushErrorScope(filter);
      successfulOuterFilters.push(filter);
      state.errorObservation.outerFilters = [...successfulOuterFilters];
    }
    state.errorObservation.includesRendererInitialization = true;
    state.errorObservation.includesAllRendersAndReadbacks = true;

    const renderer = new WebGPURenderer({
      device,
      antialias: false,
      forceWebGL: false,
      trackTimestamp: false,
    });
    state.resources.renderer = renderer;
    state.preflight.rendererConstructed = true;
    renderer.onError = (info) => {
      state.threeErrors.push({
        api: String(info?.api ?? 'unknown').slice(0, 256),
        type: String(info?.type ?? 'unknown').slice(0, 256),
        message: String(info?.message ?? '').slice(0, 4_096),
      });
    };
    renderer.onDeviceLost = (info) => {
      if (!state.resources.intentionalDeviceDestroy) {
        state.threeDeviceLosses.push({
          api: String(info?.api ?? 'WebGPU').slice(0, 256),
          message: String(info?.message ?? '').slice(0, 4_096),
          reason: String(info?.reason ?? 'unknown').slice(0, 256),
        });
      }
    };
    renderer.setPixelRatio(1);
    renderer.setSize(TARGET_WIDTH, TARGET_HEIGHT, false);
    renderer.setClearColor(0x000000, 0);
    document.getElementById('canvas-host')?.appendChild(renderer.domElement);

    await renderer.init();
    renderer.backend.trackTimestamp = false;
    requireCondition(renderer.backend?.isWebGPUBackend === true,
      'Three.js did not initialize WebGPUBackend.');
    requireCondition(renderer.backend?._supportsImmediateData === true,
      'Three.js did not recognize immediate data support.');
    await runScenarios(state);
    state.shaderCompilation = await instrumentation.collectShaderCompilationInfo();
    requireCondition(
      state.shaderCompilation.every((record) => record.messages.every(
        (message) => message.type !== 'error',
      )),
      'A generated shader module reported a compilation error.',
      state.shaderCompilation,
    );
    await device.queue.onSubmittedWorkDone();
    await Promise.resolve();
    await Promise.resolve();
  } catch (error) {
    scenarioFailure = error;
  } finally {
    try {
      await disposeRendererResources(state, true);
      await device.queue.onSubmittedWorkDone();
      await instrumentation.settlePendingErrorScopes();
      await Promise.resolve();
    } catch (error) {
      if (scenarioFailure === null) scenarioFailure = error;
    }
    if (state.cleanupFailures.length > 0 && scenarioFailure === null) {
      scenarioFailure = new Error('Renderer resource teardown reported a failure.');
    }
    state.errorObservation.depthBeforeOuterPops = instrumentation.getErrorScopeDepth();
    for (const filter of [...successfulOuterFilters].reverse()) {
      try {
        const error = await device.popErrorScope();
        state.errorObservation.outerResults.push({
          filter,
          error: error === null ? null : serializeGpuError(error),
        });
      } catch (error) {
        state.errorObservation.outerResults.push({
          filter,
          popFailure: serializeError(error),
        });
      }
    }
    state.errorObservation.finalDepth = instrumentation.getErrorScopeDepth();
  }

  if (scenarioFailure !== null) throw scenarioFailure;
  requireCondition(
    state.errorObservation.depthBeforeOuterPops === outerFilters.length
      && state.errorObservation.finalDepth === 0
      && state.errorObservation.outerResults.length === outerFilters.length
      && state.errorObservation.outerResults.every((result) => result.error === null),
    'WebGPU error scopes were unbalanced or captured an error.',
    state.errorObservation,
  );
  requireCondition(
    state.threeErrors.length === 0
      && state.threeConsoleErrors.length === 0
      && state.threeDeviceLosses.length === 0
      && state.gpuErrors.uncaptured.length === 0
      && state.gpuErrors.unexpectedDeviceLosses.length === 0,
    'Three.js/WebGPU reported an error or unexpected device loss.',
    {
      threeErrors: state.threeErrors,
      threeConsoleErrors: state.threeConsoleErrors,
      threeDeviceLosses: state.threeDeviceLosses,
      gpuErrors: state.gpuErrors,
    },
  );
  await destroyDeviceAndAwaitLoss(state);
  requireCondition(
    state.cleanupFailures.length === 0
      && state.errorObservation.intentionalDeviceLoss?.reason === 'destroyed',
    'Intentional device destruction did not resolve as destroyed.',
    state.errorObservation.intentionalDeviceLoss,
  );
  return true;
}

async function attemptCleanup(state, operation, action) {
  try {
    await action();
    return true;
  } catch (error) {
    state.cleanupFailures.push({ operation, failure: serializeError(error) });
    return false;
  }
}

async function disposeRendererResources(state, withinScopes = false) {
  if (state.resources.rendererResourcesDisposalAttempted) return;
  const hasRendererResources = state.resources.renderer !== null
    || state.resources.batchedMeshes.length > 0
    || state.resources.disposables.length > 0;
  if (!hasRendererResources) return;
  state.resources.rendererResourcesDisposalAttempted = true;
  const failureCountBefore = state.cleanupFailures.length;
  for (const [index, batchedMesh] of state.resources.batchedMeshes.entries()) {
    await attemptCleanup(state, `batchedMesh[${index}].dispose`, async () => {
      await batchedMesh.dispose();
    });
  }
  for (const [index, disposable] of state.resources.disposables.entries()) {
    await attemptCleanup(state, `disposable[${index}].dispose`, async () => {
      await disposable?.dispose?.();
    });
  }
  for (const [index, indirect] of state.resources.indirectAttributes.entries()) {
    await attemptCleanup(state, `indirectAttribute[${index}].delete`, async () => {
      const deleted = state.resources.renderer?._attributes?.delete(indirect);
      requireCondition(
        deleted !== null && deleted !== undefined,
        'An indirect GPU attribute was not explicitly deleted during teardown.',
      );
      state.errorObservation.indirectGpuAttributesDeleted += 1;
    });
  }
  await attemptCleanup(state, 'renderer.dispose', async () => {
    await state.resources.renderer?.dispose();
  });
  const succeeded = state.cleanupFailures.length === failureCountBefore;
  state.resources.rendererResourcesDisposed = succeeded;
  state.errorObservation.resourcesDisposedWithinScopes = succeeded && withinScopes;
}

async function destroyDeviceAndAwaitLoss(state) {
  if (state.resources.device === null || state.resources.deviceDestroyed) return;
  state.resources.intentionalDeviceDestroy = true;
  const destroyed = await attemptCleanup(state, 'device.destroy', async () => {
    state.resources.device.destroy();
  });
  if (!destroyed) return;
  state.resources.deviceDestroyed = true;
  await attemptCleanup(state, 'device.lost', async () => {
    const intentionalLoss = await state.resources.device.lost;
    state.errorObservation.intentionalDeviceLoss = {
      reason: String(intentionalLoss?.reason ?? 'unknown'),
      message: String(intentionalLoss?.message ?? '').slice(0, 4_096),
    };
  });
}

async function cleanup(state) {
  await disposeRendererResources(state, false);
  if (state.resources.consoleFunctionInstalled) {
    await attemptCleanup(state, 'restore Three.js console function', async () => {
      setConsoleFunction(state.resources.previousConsoleFunction);
    });
    state.resources.consoleFunctionInstalled = false;
  }
  await destroyDeviceAndAwaitLoss(state);
}

export async function runThreeDrawIndexCanary() {
  const state = {
    preflight: {
      secureContext: false,
      currentStage: 'not-started',
      capabilityDecision: 'not-evaluated',
      unsupportedStage: null,
      unsupportedReason: null,
      unsupportedDetail: null,
      adapterRequestAttempts: 0,
      adapterAcquisitions: 0,
      requestAdapterArgumentsCount: null,
      requestAdapterDescriptorSupplied: null,
      deviceRequestAttempts: 0,
      deviceAcquisitions: 0,
      requestDeviceArgumentsCount: null,
      requestDeviceDescriptorSupplied: null,
      shaderModulesAtCapabilityDecision: null,
      instrumentationInstalledBeforeRenderer: false,
      rendererConstructed: false,
    },
    capabilities: {
      wgslLanguageFeatures: [],
      immediateAddressSpace: false,
      adapterFeatures: [],
      adapterMaxImmediateSize: null,
      deviceFeatures: [],
      deviceMaxImmediateSize: null,
    },
    adapterInfo: null,
    runtimeSignature: null,
    scenarios: null,
    instrumentation: null,
    shaderCompilation: [],
    errorObservation: {
      outerFilters: [],
      includesRendererInitialization: false,
      includesAllRendersAndReadbacks: false,
      depthBeforeOuterPops: null,
      outerResults: [],
      finalDepth: null,
      resourcesDisposedWithinScopes: false,
      indirectGpuAttributesDeleted: 0,
      intentionalDeviceLoss: null,
    },
    threeErrors: [],
    threeConsole: [],
    threeConsoleErrors: [],
    threeDeviceLosses: [],
    gpuErrors: {
      uncaptured: [],
      unexpectedDeviceLosses: [],
    },
    cleanupFailures: [],
    resources: {
      renderer: null,
      device: null,
      instrumentation: null,
      disposables: [],
      indirectAttributes: [],
      batchedMeshes: [],
      intentionalDeviceDestroy: false,
      previousConsoleFunction: null,
      consoleFunctionInstalled: false,
      rendererResourcesDisposed: false,
      rendererResourcesDisposalAttempted: false,
      deviceDestroyed: false,
    },
  };
  let failure = null;
  let completed = false;
  try {
    completed = await executeCanary(state);
  } catch (error) {
    failure = serializeError(error);
  }

  try {
    await cleanup(state);
  } catch (error) {
    if (failure === null) failure = serializeError(error);
  }
  if (state.cleanupFailures.length > 0 && failure === null) {
    failure = serializeError(new Error('Canary cleanup reported one or more failures.'));
  }

  const status = failure !== null
    ? 'failed'
    : (completed ? 'passed' : 'unsupported');
  const result = {
    schemaVersion: 1,
    kind: THREE_DRAW_INDEX_CANARY_KIND,
    status,
    executionMode: 'timing-free-one-shot-correctness',
    timingCaptured: false,
    benchmarkClaim: false,
    claimBoundary:
      'Correctness and integration on one standards-capable implementation; no performance or cross-device universality claim.',
    userAgent: navigator.userAgent,
    preflight: state.preflight,
    capabilities: state.capabilities,
    adapterInfo: state.adapterInfo,
    runtimeSignature: state.runtimeSignature,
    scenarios: state.scenarios,
    instrumentation: state.instrumentation,
    shaderCompilation: state.shaderCompilation,
    errorObservation: state.errorObservation,
    threeErrors: state.threeErrors,
    threeConsole: state.threeConsole,
    threeConsoleErrors: state.threeConsoleErrors,
    threeDeviceLosses: state.threeDeviceLosses,
    gpuErrors: state.gpuErrors,
    cleanupFailures: state.cleanupFailures,
    failure,
  };
  return result;
}

if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  const globalHandle = {
    ready: false,
    result: null,
  };
  globalThis.__THREE_DRAW_INDEX_CANARY__ = globalHandle;

  runThreeDrawIndexCanary().then((result) => {
    globalHandle.result = result;
    globalHandle.ready = true;
    const statusElement = document.getElementById('status');
    if (statusElement !== null) statusElement.textContent = `Canary status: ${result.status}`;
  }, (error) => {
    globalHandle.result = {
      schemaVersion: 1,
      kind: THREE_DRAW_INDEX_CANARY_KIND,
      status: 'failed',
      executionMode: 'timing-free-one-shot-correctness',
      timingCaptured: false,
      benchmarkClaim: false,
      failure: serializeError(error),
    };
    globalHandle.ready = true;
  });
}
