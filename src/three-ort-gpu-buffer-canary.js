import * as ort from '/__ort_webgpu__/ort.webgpu.bundle.min.mjs';
import * as THREE from '/__three_ort_bridge__/three.webgpu.js';
import {
  Fn,
  float,
  floatBitsToUint,
  instanceIndex,
  positionGeometry,
  storage,
  uint,
  uvec4,
  vec3,
  vec4,
} from '/__three_ort_bridge__/three.tsl.js';

import {
  ORT_CANARY_RUNS,
  ORT_GPU_BUFFER_CANARY_KIND,
  ORT_MARKER_NDC_SIZE,
  ORT_MODEL_BYTES,
  ORT_MODEL_SHA256,
  ORT_ORACLE_SENTINEL,
  ORT_ORACLE_SIZE,
  ORT_TARGET_ITEM_COUNT,
  ORT_TARGET_ITEM_SIZE,
  ORT_TARGET_LOGICAL_BYTES,
  ORT_TARGET_PHYSICAL_BYTES,
  analyzeOrtGpuBufferOracle,
  decodeOrtModelBytes,
  expectedOrtGpuBufferTuples,
} from './three-ort-gpu-buffer-canary-contract.js';

const ORT_WASM_PATH = '/__ort_webgpu__/ort-wasm-simd-threaded.asyncify.wasm';
const ORT_BUNDLE_PATH = '/__ort_webgpu__/ort.webgpu.bundle.min.mjs';
const THREE_BUNDLE_PATH = '/__three_ort_bridge__/three.webgpu.js';
const THREE_TSL_PATH = '/__three_ort_bridge__/three.tsl.js';

const state = {
  kind: ORT_GPU_BUFFER_CANARY_KIND,
  status: 'running',
  stage: 'bootstrap',
  error: null,
  model: {
    source: 'onnxruntime/test/testdata/mul_1.onnx',
    expectedBytes: ORT_MODEL_BYTES,
    actualBytes: null,
    expectedSha256: ORT_MODEL_SHA256,
    actualSha256: null,
    verified: false,
  },
  routes: {
    ortBundle: ORT_BUNDLE_PATH,
    ortWasm: ORT_WASM_PATH,
    threeBundle: THREE_BUNDLE_PATH,
    threeTsl: THREE_TSL_PATH,
  },
  environment: {
    navigatorGpu: navigator.gpu !== undefined,
    adapterAcquired: false,
    adapterInfo: null,
    deviceAcquired: false,
    deviceLost: null,
    uncapturedErrors: [],
    ortVersion: String(ort.env?.versions?.web ?? 'unknown'),
    ortProxy: false,
    ortThreads: 1,
    ortAdapterSame: null,
    ortDeviceSame: null,
    ortDeviceObservationError: null,
    ortExecutionProviderDeviceSupplied: null,
    ortExecutionProviderDeviceSame: null,
    instrumentation: {
      phaseHook: typeof window.__setThreeOrtGpuBufferPhase === 'function',
      targetHook: typeof window.__registerThreeOrtGpuBufferTarget === 'function',
      collector: typeof window.__collectThreeOrtGpuBufferInstrumentation === 'function',
      targetRegistrationResult: null,
    },
  },
  target: null,
  runs: [null, null],
  lifecycle: {
    phase: 'bootstrap',
    screenshotReady: false,
    continuationCalls: 0,
    firstThreeDisposed: false,
    targetReusedByOrt: false,
    targetReusedByThree: false,
    secondThreeDisposed: false,
    ortReleasedAfterThree: false,
    outputTensorDisposed: false,
    targetDestroyCountBeforeOwnerCleanup: null,
    targetDestroyCountAfterOwnerCleanup: null,
    ownerDestroyedTarget: false,
    cleanupErrors: [],
  },
};

Object.defineProperty(window, '__threeOrtGpuBufferCanary', { value: state });

const internals = {
  adapter: null,
  device: null,
  session: null,
  targetBuffer: null,
  outputTensor: null,
  activeProof: null,
  sessionReleased: false,
  targetDestroyed: false,
};

let resolveContinuation;
let continuationRequested = false;
const continuationSignal = new Promise((resolve) => {
  resolveContinuation = resolve;
});
let canaryPromise;

function serializeError(error) {
  return {
    name: String(error?.name ?? 'Error'),
    message: String(error?.message ?? error),
    stack: error?.stack == null ? null : String(error.stack),
  };
}

function serializeValidationError(error) {
  return error === null ? null : {
    name: String(error?.constructor?.name ?? error?.name ?? 'GPUError'),
    message: String(error?.message ?? error),
  };
}

function assertSerializableState() {
  JSON.parse(JSON.stringify(state));
}

function setStatus(status, message) {
  state.status = status;
  const statusElement = document.getElementById('status');
  if (statusElement) {
    statusElement.dataset.state = status;
    statusElement.textContent = message;
  }
  assertSerializableState();
}

function setProbePhase(phase) {
  state.stage = phase;
  state.lifecycle.phase = phase;
  if (typeof window.__setThreeOrtGpuBufferPhase === 'function') {
    window.__setThreeOrtGpuBufferPhase(phase);
  }
}

function serializableRegistrationResult(value) {
  if (value === null || typeof value === 'string' || typeof value === 'number'
    || typeof value === 'boolean') {
    return value;
  }
  return null;
}

function targetDestroyCountFromInstrumentation() {
  const instrumentation = window.__threeOrtGpuBufferInstrumentation;
  const targetBufferId = instrumentation?.target?.bufferId;
  if (!Number.isSafeInteger(targetBufferId) || !Array.isArray(instrumentation.events)) {
    return null;
  }
  return instrumentation.events.filter((event) => (
    event.method === 'destroyBuffer' && event.bufferId === targetBufferId
  )).length;
}

async function sha256Bytes(bytes) {
  const view = bytes instanceof Uint8Array
    ? bytes
    : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const exactBytes = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
  const digest = await crypto.subtle.digest('SHA-256', exactBytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

function adapterInfoRecord(adapter) {
  const info = adapter.info;
  if (!info) return null;
  return {
    vendor: String(info.vendor ?? ''),
    architecture: String(info.architecture ?? ''),
    device: String(info.device ?? ''),
    description: String(info.description ?? ''),
  };
}

function installDeviceErrorRecording(device) {
  device.addEventListener('uncapturederror', (event) => {
    state.environment.uncapturedErrors.push(serializeValidationError(event.error));
  });
  device.lost.then((info) => {
    state.environment.deviceLost = {
      reason: String(info.reason ?? 'unknown'),
      message: String(info.message ?? ''),
    };
  }).catch((error) => {
    state.environment.deviceLost = serializeError(error);
  });
}

function createMarkerGeometry() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([
    -0.5, -0.5, 0,
    0.5, -0.5, 0,
    0.5, 0.5, 0,
    -0.5, -0.5, 0,
    0.5, 0.5, 0,
    -0.5, 0.5, 0,
  ], 3));
  return geometry;
}

function createMarkerMesh(geometry, material) {
  const mesh = new THREE.InstancedMesh(geometry, material, 3);
  const identity = new THREE.Matrix4();
  for (let instance = 0; instance < 3; instance += 1) {
    mesh.setMatrixAt(instance, identity);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.frustumCulled = false;
  return mesh;
}

function createStorageDrivenMaterials(attribute) {
  const values = storage(attribute).toReadOnly();
  const positionMaterial = new THREE.NodeMaterial();
  positionMaterial.side = THREE.DoubleSide;
  positionMaterial.depthTest = false;
  positionMaterial.depthWrite = false;
  positionMaterial.blending = THREE.NoBlending;
  positionMaterial.toneMapped = false;
  positionMaterial.positionNode = Fn(() => {
    const base = instanceIndex.mul(uint(2));
    const x = values.element(base).toVar('ortPositionX');
    const y = values.element(base.add(uint(1))).toVar('ortPositionY');
    return vec3(
      positionGeometry.x.mul(float(ORT_MARKER_NDC_SIZE)).add(x),
      positionGeometry.y.mul(float(ORT_MARKER_NDC_SIZE)).add(y),
      0,
    );
  })();
  positionMaterial.fragmentNode = Fn(() => {
    const base = instanceIndex.mul(uint(2));
    const x = values.element(base).toVar('ortFragmentX');
    const y = values.element(base.add(uint(1))).toVar('ortFragmentY');
    return uvec4(
      floatBitsToUint(x),
      floatBitsToUint(y),
      uint(instanceIndex),
      uint(ORT_ORACLE_SENTINEL),
    );
  })();

  const visualValues = storage(attribute).toReadOnly();
  const visualMaterial = new THREE.NodeMaterial();
  visualMaterial.side = THREE.DoubleSide;
  visualMaterial.depthTest = false;
  visualMaterial.depthWrite = false;
  visualMaterial.blending = THREE.NoBlending;
  visualMaterial.toneMapped = false;
  visualMaterial.positionNode = Fn(() => {
    const base = instanceIndex.mul(uint(2));
    const x = visualValues.element(base).toVar('ortVisualX');
    const y = visualValues.element(base.add(uint(1))).toVar('ortVisualY');
    return vec3(
      positionGeometry.x.mul(float(ORT_MARKER_NDC_SIZE)).add(x),
      positionGeometry.y.mul(float(ORT_MARKER_NDC_SIZE)).add(y),
      0,
    );
  })();
  visualMaterial.fragmentNode = Fn(() => {
    const base = instanceIndex.mul(uint(2));
    const x = visualValues.element(base).toVar('ortVisualFragmentX');
    const y = visualValues.element(base.add(uint(1))).toVar('ortVisualFragmentY');
    const instance = float(instanceIndex);
    return vec4(
      x.add(1).mul(0.5),
      y.add(1).mul(0.5),
      instance.mul(0.24).add(0.38),
      1,
    );
  })();

  return { oracleMaterial: positionMaterial, visualMaterial };
}

function createCamera() {
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 1;
  return camera;
}

function createOracleTarget() {
  const target = new THREE.RenderTarget(ORT_ORACLE_SIZE.width, ORT_ORACLE_SIZE.height, {
    depthBuffer: false,
    format: THREE.RGBAIntegerFormat,
    magFilter: THREE.NearestFilter,
    minFilter: THREE.NearestFilter,
    samples: 0,
    type: THREE.UnsignedIntType,
  });
  target.texture.name = 'ort-three-borrowed-buffer-exact-rgba32uint-oracle';
  target.texture.colorSpace = THREE.NoColorSpace;
  target.texture.generateMipmaps = false;
  return target;
}

async function cleanupPartialThreeProof(proof) {
  if (proof.disposalAttempted) return;
  proof.disposalAttempted = true;
  const attempt = (operation) => {
    try {
      operation();
    } catch (error) {
      state.lifecycle.cleanupErrors.push(serializeError(error));
    }
  };

  if (proof.scene && proof.mesh) attempt(() => proof.scene.remove(proof.mesh));
  if (proof.attribute && proof.renderer._attributes?.has(proof.attribute)) {
    attempt(() => proof.renderer._attributes.delete(proof.attribute));
  }
  if (proof.geometry) attempt(() => proof.geometry.dispose());
  if (proof.oracleMaterial) attempt(() => proof.oracleMaterial.dispose());
  if (proof.visualMaterial) attempt(() => proof.visualMaterial.dispose());
  if (proof.oracleTarget) attempt(() => proof.oracleTarget.dispose());
  try {
    await internals.device.queue.onSubmittedWorkDone();
  } catch (error) {
    state.lifecycle.cleanupErrors.push(serializeError(error));
  }
  try {
    await proof.renderer.dispose();
  } catch (error) {
    state.lifecycle.cleanupErrors.push(serializeError(error));
  }
  attempt(() => proof.renderer.domElement.remove());
}

async function createThreeProof(runIndex) {
  const proof = {
    attribute: null,
    camera: null,
    disposalAttempted: false,
    geometry: null,
    mesh: null,
    oracleMaterial: null,
    oracleTarget: null,
    record: null,
    renderer: new THREE.WebGPURenderer({
      antialias: false,
      device: internals.device,
    }),
    scene: null,
    visualMaterial: null,
  };

  try {
    const { renderer } = proof;
    renderer.setPixelRatio(1);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    document.getElementById('canvas-host').replaceChildren(renderer.domElement);
    await renderer.init();

    proof.attribute = new THREE.GPUStorageBufferAttribute(
      internals.targetBuffer,
      internals.device,
      ORT_TARGET_ITEM_SIZE,
      ORT_TARGET_ITEM_COUNT,
    );
    proof.attribute.name = `ort-three-output-${runIndex + 1}`;
    const materials = createStorageDrivenMaterials(proof.attribute);
    proof.oracleMaterial = materials.oracleMaterial;
    proof.visualMaterial = materials.visualMaterial;
    proof.geometry = createMarkerGeometry();
    proof.mesh = createMarkerMesh(proof.geometry, proof.oracleMaterial);
    proof.scene = new THREE.Scene();
    proof.camera = createCamera();
    proof.oracleTarget = createOracleTarget();
    proof.scene.add(proof.mesh);

    renderer.setRenderTarget(proof.oracleTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.render(proof.scene, proof.camera);
    const words = await renderer.readRenderTargetPixelsAsync(
      proof.oracleTarget,
      0,
      0,
      ORT_ORACLE_SIZE.width,
      ORT_ORACLE_SIZE.height,
    );
    const oracle = analyzeOrtGpuBufferOracle(words, runIndex);
    const backendData = renderer.backend.get(proof.attribute);

    proof.mesh.material = proof.visualMaterial;
    renderer.setRenderTarget(null);
    renderer.setClearColor(0x07101f, 1);
    renderer.render(proof.scene, proof.camera);
    await internals.device.queue.onSubmittedWorkDone();

    proof.record = {
      rendererDeviceSame: renderer.backend.device === internals.device,
      backendIsWebGPU: renderer.backend.isWebGPUBackend === true,
      backendBufferSame: backendData.buffer === internals.targetBuffer,
      backendBindingOffset: Number(backendData.bindingOffset),
      backendBindingSize: Number(backendData.bindingSize),
      backendOwnsBuffer: backendData.ownsBuffer !== false,
      attribute: {
        isGpuStorageBufferAttribute: proof.attribute.isGPUStorageBufferAttribute === true,
        isStorageBufferAttribute: proof.attribute.isStorageBufferAttribute === true,
        itemSize: proof.attribute.itemSize,
        count: proof.attribute.count,
        byteOffset: proof.attribute.byteOffset,
        byteLength: proof.attribute.byteLength,
        storageOnly: true,
      },
      oracle: {
        ...oracle,
        typedArray: words.constructor.name,
        byteLength: words.byteLength,
        readbackSha256: await sha256Bytes(new Uint8Array(
          words.buffer,
          words.byteOffset,
          words.byteLength,
        )),
      },
      borrowedAttributeRelease: {
        rendererAttributesHadBefore: null,
        backendHadBefore: null,
        rendererAttributesHasAfter: null,
        backendHasAfter: null,
        targetDestroyCountBeforeRendererDispose: null,
        targetDestroyCountAfterRendererDispose: null,
      },
      disposed: false,
    };
    return proof;
  } catch (error) {
    await cleanupPartialThreeProof(proof);
    throw error;
  }
}

async function disposeThreeProof(proof, runRecord) {
  if (proof.disposalAttempted) return;
  proof.disposalAttempted = true;
  let failure = null;
  const rememberFailure = (error) => {
    if (failure === null) failure = error;
    else state.lifecycle.cleanupErrors.push(serializeError(error));
  };
  const attempt = (operation) => {
    try {
      operation();
    } catch (error) {
      rememberFailure(error);
    }
  };

  attempt(() => proof.scene.remove(proof.mesh));
  const release = runRecord.three.borrowedAttributeRelease;
  attempt(() => {
    release.rendererAttributesHadBefore = proof.renderer._attributes.has(proof.attribute);
    release.backendHadBefore = proof.renderer.backend.has(proof.attribute);
    if (release.rendererAttributesHadBefore !== true || release.backendHadBefore !== true) {
      throw new Error('Three did not cache the borrowed storage attribute before explicit release.');
    }
  });
  attempt(() => proof.renderer._attributes.delete(proof.attribute));
  attempt(() => {
    release.rendererAttributesHasAfter = proof.renderer._attributes.has(proof.attribute);
    release.backendHasAfter = proof.renderer.backend.has(proof.attribute);
    if (release.rendererAttributesHasAfter !== false || release.backendHasAfter !== false) {
      throw new Error('Three retained the borrowed storage attribute after explicit release.');
    }
  });
  release.targetDestroyCountBeforeRendererDispose = targetDestroyCountFromInstrumentation();
  if (release.targetDestroyCountBeforeRendererDispose !== null
    && release.targetDestroyCountBeforeRendererDispose !== 0) {
    rememberFailure(new Error(
      'Three destroyed the borrowed target while releasing its attribute cache.',
    ));
  }

  attempt(() => proof.geometry.dispose());
  attempt(() => proof.oracleMaterial.dispose());
  attempt(() => proof.visualMaterial.dispose());
  attempt(() => proof.oracleTarget.dispose());
  try {
    await internals.device.queue.onSubmittedWorkDone();
  } catch (error) {
    rememberFailure(error);
  }
  let rendererDisposed = false;
  try {
    await proof.renderer.dispose();
    rendererDisposed = true;
  } catch (error) {
    rememberFailure(error);
  }
  release.targetDestroyCountAfterRendererDispose = targetDestroyCountFromInstrumentation();
  if (release.targetDestroyCountAfterRendererDispose !== null
    && release.targetDestroyCountAfterRendererDispose !== 0) {
    rememberFailure(new Error('Three destroyed the borrowed target during renderer disposal.'));
  }
  attempt(() => proof.renderer.domElement.remove());
  runRecord.three.disposed = rendererDisposed;
  internals.activeProof = null;
  if (failure !== null) throw failure;
}

async function executeRun(runIndex) {
  const definition = ORT_CANARY_RUNS[runIndex];
  const runRecord = {
    runIndex,
    label: definition.label,
    input: [...definition.input],
    expectedOutput: [...definition.expectedOutput],
    expectedBits: [...definition.expectedBits],
    expectedTuples: expectedOrtGpuBufferTuples(runIndex),
    ort: null,
    three: null,
    validationScope: null,
  };
  state.runs[runIndex] = runRecord;

  internals.device.pushErrorScope('validation');
  let validationError = null;
  let thrownError = null;
  try {
    setProbePhase(`ort-run-${runIndex + 1}`);
    const inputTensor = new ort.Tensor(
      'float32',
      Float32Array.from(definition.input),
      [3, 2],
    );
    let results;
    try {
      results = await internals.session.run({ X: inputTensor }, { Y: internals.outputTensor });
    } finally {
      inputTensor.dispose();
    }

    runRecord.ort = {
      returnedSameTensor: results.Y === internals.outputTensor,
      returnedSameBuffer: results.Y.gpuBuffer === internals.targetBuffer,
      type: String(results.Y.type),
      dims: Array.from(results.Y.dims),
      usedPreallocatedFetch: true,
      calledGetData: false,
    };
    if (!runRecord.ort.returnedSameTensor || !runRecord.ort.returnedSameBuffer) {
      throw new Error(`ORT ${definition.label} did not return the preallocated GPU tensor unchanged.`);
    }

    setProbePhase(`three-proof-${runIndex + 1}`);
    const proof = await createThreeProof(runIndex);
    internals.activeProof = proof;
    runRecord.three = proof.record;

    if (proof.record.rendererDeviceSame !== true
      || proof.record.backendBufferSame !== true
      || proof.record.backendBindingOffset !== 0
      || proof.record.backendBindingSize !== ORT_TARGET_LOGICAL_BYTES
      || proof.record.backendOwnsBuffer !== false) {
      throw new Error(`Three ${definition.label} did not borrow the exact 24-byte target range.`);
    }
    if (proof.record.oracle.exact !== true) {
      throw new Error(`Three ${definition.label} rgba32uint bit-and-position oracle failed.`);
    }
  } catch (error) {
    thrownError = error;
  } finally {
    try {
      validationError = await internals.device.popErrorScope();
    } catch (error) {
      if (thrownError === null) thrownError = error;
    }
    runRecord.validationScope = serializeValidationError(validationError);
  }

  if (validationError !== null && thrownError === null) thrownError = validationError;
  if (thrownError !== null) throw thrownError;
  return runRecord;
}

async function releaseOrtAfterThree() {
  if (internals.session !== null && internals.sessionReleased === false) {
    setProbePhase('ort-cleanup');
    await internals.device.queue.onSubmittedWorkDone();
    await internals.session.release();
    internals.sessionReleased = true;
    state.lifecycle.ortReleasedAfterThree = true;
  }
}

function ownerDestroyTarget() {
  if (internals.targetBuffer !== null && internals.targetDestroyed === false) {
    state.lifecycle.targetDestroyCountBeforeOwnerCleanup =
      targetDestroyCountFromInstrumentation();
    if (state.lifecycle.targetDestroyCountBeforeOwnerCleanup !== null
      && state.lifecycle.targetDestroyCountBeforeOwnerCleanup !== 0) {
      throw new Error('The borrowed target was destroyed before explicit owner cleanup.');
    }
    setProbePhase('owner-cleanup');
    internals.targetBuffer.destroy();
    internals.targetDestroyed = true;
    state.lifecycle.ownerDestroyedTarget = true;
    state.lifecycle.targetDestroyCountAfterOwnerCleanup =
      targetDestroyCountFromInstrumentation();
    if (state.lifecycle.targetDestroyCountAfterOwnerCleanup !== null
      && state.lifecycle.targetDestroyCountAfterOwnerCleanup !== 1) {
      throw new Error('Explicit owner cleanup did not destroy the target exactly once.');
    }
  }
}

async function cleanupAfterFailure() {
  if (internals.activeProof !== null) {
    try {
      setProbePhase('three-dispose-failure');
      const runRecord = state.runs.find((record) => record?.three === internals.activeProof.record);
      if (runRecord) await disposeThreeProof(internals.activeProof, runRecord);
    } catch (error) {
      state.lifecycle.cleanupErrors.push(serializeError(error));
      internals.activeProof = null;
    }
  }
  try {
    await releaseOrtAfterThree();
  } catch (error) {
    state.lifecycle.cleanupErrors.push(serializeError(error));
  }
  try {
    ownerDestroyTarget();
  } catch (error) {
    state.lifecycle.cleanupErrors.push(serializeError(error));
  }
}

async function runCanary() {
  if (navigator.gpu === undefined) throw new Error('WebGPU is not available.');

  const modelBytes = decodeOrtModelBytes();
  state.model.actualBytes = modelBytes.byteLength;
  state.model.actualSha256 = await sha256Bytes(modelBytes);
  state.model.verified = state.model.actualBytes === ORT_MODEL_BYTES
    && state.model.actualSha256 === ORT_MODEL_SHA256;
  if (!state.model.verified) throw new Error('The embedded ONNX model failed its byte/hash pin.');

  internals.adapter = await navigator.gpu.requestAdapter();
  if (internals.adapter === null) throw new Error('WebGPU did not provide an adapter.');
  state.environment.adapterAcquired = true;
  state.environment.adapterInfo = adapterInfoRecord(internals.adapter);

  internals.device = await internals.adapter.requestDevice();
  state.environment.deviceAcquired = true;
  installDeviceErrorRecording(internals.device);

  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { wasm: ORT_WASM_PATH };
  ort.env.webgpu.adapter = internals.adapter;
  state.environment.ortAdapterSame = ort.env.webgpu.adapter === internals.adapter;

  const webgpuExecutionProvider = {
    name: 'webgpu',
    device: internals.device,
    validationMode: 'full',
  };
  state.environment.ortExecutionProviderDeviceSupplied =
    Object.hasOwn(webgpuExecutionProvider, 'device');
  state.environment.ortExecutionProviderDeviceSame =
    webgpuExecutionProvider.device === internals.device;
  if (state.environment.ortExecutionProviderDeviceSupplied !== true
    || state.environment.ortExecutionProviderDeviceSame !== true) {
    throw new Error('ORT WebGPU execution-provider options lost the explicit GPUDevice.');
  }
  internals.session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [webgpuExecutionProvider],
  });
  try {
    state.environment.ortDeviceSame = await ort.env.webgpu.device === internals.device;
  } catch (error) {
    state.environment.ortDeviceObservationError = serializeError(error);
  }

  internals.targetBuffer = internals.device.createBuffer({
    label: 'ort-three-preallocated-output',
    size: ORT_TARGET_PHYSICAL_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const registrationResult = typeof window.__registerThreeOrtGpuBufferTarget === 'function'
    ? window.__registerThreeOrtGpuBufferTarget(internals.targetBuffer, {
        logicalBytes: ORT_TARGET_LOGICAL_BYTES,
      })
    : null;
  state.environment.instrumentation.targetRegistrationResult =
    serializableRegistrationResult(registrationResult);

  internals.outputTensor = ort.Tensor.fromGpuBuffer(internals.targetBuffer, {
    dataType: 'float32',
    dims: [3, 2],
  });
  state.target = {
    label: 'ort-three-preallocated-output',
    physicalBytes: internals.targetBuffer.size,
    logicalBytes: ORT_TARGET_LOGICAL_BYTES,
    usage: internals.targetBuffer.usage,
    mapState: String(internals.targetBuffer.mapState),
    outputTensorType: String(internals.outputTensor.type),
    outputTensorDims: Array.from(internals.outputTensor.dims),
    outputTensorBufferSame: internals.outputTensor.gpuBuffer === internals.targetBuffer,
  };

  await executeRun(0);
  state.lifecycle.screenshotReady = true;
  setProbePhase('screenshot-ready');
  setStatus(
    'screenshot-ready',
    'RUN 1 EXACT: ORT GPU output is driving Three directly; awaiting lifecycle proof',
  );

  await continuationSignal;

  setProbePhase('three-dispose-1');
  await disposeThreeProof(internals.activeProof, state.runs[0]);
  state.lifecycle.firstThreeDisposed = true;

  await executeRun(1);
  state.lifecycle.targetReusedByOrt = state.runs[1].ort.returnedSameBuffer;
  state.lifecycle.targetReusedByThree = state.runs[1].three.backendBufferSame;

  setProbePhase('three-dispose-2');
  await disposeThreeProof(internals.activeProof, state.runs[1]);
  state.lifecycle.secondThreeDisposed = true;

  await releaseOrtAfterThree();
  ownerDestroyTarget();

  setStatus(
    'passed',
    'PASS: exact GPU bits + positions, same buffer reused after full Three teardown',
  );
}

function continueCanary() {
  state.lifecycle.continuationCalls += 1;
  if (state.status === 'failed' || state.status === 'passed') return canaryPromise;
  if (state.status !== 'screenshot-ready') {
    return Promise.reject(new Error(
      `Cannot continue ORT/Three canary while status is ${state.status}.`,
    ));
  }
  if (continuationRequested === false) {
    continuationRequested = true;
    setStatus('continuing', 'Disposing Three, reusing the same GPU buffer, and proving run 2...');
    resolveContinuation();
  }
  return canaryPromise;
}

Object.defineProperty(window, '__continueThreeOrtGpuBufferCanary', {
  value: continueCanary,
});

canaryPromise = runCanary().catch(async (error) => {
  state.error = serializeError(error);
  await cleanupAfterFailure();
  setStatus('failed', `FAIL: ${state.error.message}`);
  console.error(error);
});
