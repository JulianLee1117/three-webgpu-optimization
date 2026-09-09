import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  ORT_CANARY_RUNS,
  ORT_GPU_BUFFER_CANARY_KIND,
  ORT_MODEL_BYTES,
  ORT_MODEL_SHA256,
  ORT_ORACLE_SENTINEL,
  ORT_ORACLE_SIZE,
  ORT_TARGET_ITEM_COUNT,
  ORT_TARGET_LOGICAL_BYTES,
  ORT_TARGET_PHYSICAL_BYTES,
  analyzeOrtGpuBufferOracle,
  decodeOrtModelBytes,
  expectedOrtGpuBufferTuples,
  ndcToOraclePixel,
} from '../src/three-ort-gpu-buffer-canary-contract.js';
import {
  validateOrtGpuBufferEvidence,
  verifyOrtGpuBufferPageSource,
} from '../scripts/probe-three-ort-gpu-buffer.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const protocolPath = path.join(repoRoot, 'protocols', 'three-ort-gpu-buffer-canary-v1.json');
const pageModulePath = path.join(repoRoot, 'src', 'three-ort-gpu-buffer-canary.js');

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function floatBits(values) {
  const floats = Float32Array.from(values, Math.fround);
  return Array.from(new Uint32Array(floats.buffer));
}

function replaceOccurrence(source, needle, replacement, occurrence) {
  let index = -1;
  let searchFrom = 0;
  for (let current = 0; current <= occurrence; current += 1) {
    index = source.indexOf(needle, searchFrom);
    assert.notEqual(index, -1);
    searchFrom = index + needle.length;
  }
  return source.slice(0, index) + replacement + source.slice(index + needle.length);
}

function syntheticVertexShader(identifier) {
  return [
    `@group(0) @binding(0) var<storage, read> ${identifier}: array<f32>;`,
    '@vertex fn main() -> @builtin(position) vec4f {',
    `  let x = ${identifier}[0];`,
    `  let y = ${identifier}[1];`,
    '  return vec4f(x, y, 0.0, 1.0);',
    '}',
  ].join('\n');
}

function syntheticFragmentShader(identifier) {
  return [
    `@binding(0) @group(0) var<storage, read> ${identifier}: array<f32>;`,
    '@fragment fn main() -> @location(0) vec4u {',
    `  let x = bitcast<u32>(${identifier}[0]);`,
    `  let y = bitcast<u32>(${identifier}[1]);`,
    '  return vec4u(x, y, 0u, 1u);',
    '}',
  ].join('\n');
}

function syntheticOracle(runIndex, { corrupt = false, shiftFirstMarker = 0 } = {}) {
  const { width, height } = ORT_ORACLE_SIZE;
  const words = new Uint32Array(width * height * 4);
  const run = ORT_CANARY_RUNS[runIndex];
  const tuples = expectedOrtGpuBufferTuples(runIndex);

  for (let instance = 0; instance < 3; instance += 1) {
    const center = ndcToOraclePixel(
      run.expectedOutput[instance * 2],
      run.expectedOutput[instance * 2 + 1],
    );
    const centerX = center.x + (instance === 0 ? shiftFirstMarker : 0);
    const minX = Math.ceil(centerX - 11);
    const maxX = Math.floor(centerX + 11);
    const minY = Math.ceil(center.y - 11);
    const maxY = Math.floor(center.y + 11);
    const tuple = [...tuples[instance]];
    if (corrupt && instance === 1) tuple[0] ^= 1;

    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const offset = (y * width + x) * 4;
        words.set(tuple, offset);
      }
    }
  }

  return words;
}

async function readProtocol() {
  return JSON.parse(await readFile(protocolPath, 'utf8'));
}

function syntheticOracleRecord(protocol, runIndex) {
  const tuples = expectedOrtGpuBufferTuples(runIndex);
  return {
    typedArray: protocol.oracle.typedArray,
    width: protocol.oracle.width,
    height: protocol.oracle.height,
    byteLength: protocol.oracle.width * protocol.oracle.height * 16,
    minimumPixelsPerMarker: protocol.oracle.minimumPixelsPerTuple,
    maximumPixelsPerMarker: protocol.oracle.maximumPixelsPerTuple,
    spatialTolerancePixels: protocol.oracle.maximumCentroidErrorPixels,
    expectedPixelTuples: tuples,
    exact: true,
    unexpected: [],
    backgroundPixels: 64_084,
    spatialMismatches: [],
    tuples: tuples.map((tuple, instance) => ({
      instance,
      tuple,
      count: 484,
      sufficient: true,
      notOversized: true,
      spatialMismatch: false,
      centroidError: { x: 0, y: 0 },
    })),
    readbackSha256: (runIndex === 0 ? 'a' : 'b').repeat(64),
  };
}

function syntheticPageRun(protocol, runIndex) {
  const expected = protocol.runs[runIndex];
  return {
    runIndex,
    input: [...expected.input],
    expectedOutput: [...expected.expectedOutput],
    expectedBits: [...expected.expectedBits],
    expectedTuples: expectedOrtGpuBufferTuples(runIndex),
    ort: {
      returnedSameTensor: true,
      returnedSameBuffer: true,
      type: 'float32',
      dims: [...protocol.model.dims],
      usedPreallocatedFetch: true,
      calledGetData: false,
    },
    three: {
      rendererDeviceSame: true,
      backendIsWebGPU: true,
      backendBufferSame: true,
      backendBindingOffset: protocol.target.byteOffset,
      backendBindingSize: protocol.target.logicalBytes,
      backendOwnsBuffer: false,
      attribute: {
        isGpuStorageBufferAttribute: true,
        isStorageBufferAttribute: true,
        itemSize: protocol.target.itemSize,
        count: protocol.target.count,
        byteOffset: protocol.target.byteOffset,
        byteLength: protocol.target.logicalBytes,
        storageOnly: true,
      },
      oracle: syntheticOracleRecord(protocol, runIndex),
      borrowedAttributeRelease: {
        rendererAttributesHadBefore: true,
        backendHadBefore: true,
        rendererAttributesHasAfter: false,
        backendHasAfter: false,
        targetDestroyCountBeforeRendererDispose: 0,
        targetDestroyCountAfterRendererDispose: 0,
      },
      disposed: true,
    },
    validationScope: null,
  };
}

function bufferGroup(
  bindGroupId,
  deviceId,
  bufferId,
  { exactRange = false, layoutId = null } = {},
) {
  return {
    bindGroupId,
    deviceId,
    layoutId,
    entries: [{
      binding: 0,
      resource: {
        type: 'buffer',
        bufferId,
        hasOwnOffset: exactRange,
        hasOwnSize: exactRange,
        offset: exactRange ? 0 : null,
        size: exactRange ? ORT_TARGET_LOGICAL_BYTES : null,
      },
    }],
  };
}

function syntheticInstrumentation(protocol) {
  const targetId = 7;
  const deviceId = 1;
  const queueId = 601;
  const bindGroups = [
    bufferGroup(101, deviceId, targetId),
    bufferGroup(102, deviceId, targetId),
    bufferGroup(201, deviceId, targetId, { exactRange: true, layoutId: 701 }),
    bufferGroup(202, deviceId, targetId, { exactRange: true, layoutId: 702 }),
    bufferGroup(999, deviceId, 8),
  ];
  const events = [];
  const phaseTransitions = [];
  const commandEncoders = [];
  const commandBuffers = [];
  const passes = [];
  const pipelines = [];
  let sequence = 0;
  const phase = (name) => {
    const previous = phaseTransitions.at(-1)?.phase ?? 'bootstrap';
    sequence += 1;
    events.push({ method: 'phase', phase: name, sequence });
    phaseTransitions.push({ from: previous, phase: name, sequence });
  };
  const command = ({
    method,
    phaseName,
    passId,
    bindGroupId,
    commandEncoderId,
    commandBufferId,
    pipelineId,
    kind,
    pipelineDescriptor = null,
  }) => {
    commandEncoders.push({ commandEncoderId, deviceId, phase: phaseName });
    commandBuffers.push({ commandBufferId, commandEncoderId, deviceId, phase: phaseName });
    passes.push({ commandEncoderId, deviceId, kind: `${kind}-pass`, passId, phase: phaseName });
    pipelines.push({
      pipelineId,
      deviceId,
      kind,
      phase: phaseName,
      descriptor: pipelineDescriptor,
    });
    sequence += 1;
    events.push({
      method: 'setPipeline',
      phase: phaseName,
      sequence,
      passId,
      commandEncoderId,
      deviceId,
      pipelineId,
    });
    sequence += 1;
    events.push({
      method: 'setBindGroup',
      phase: phaseName,
      sequence,
      passId,
      commandEncoderId,
      deviceId,
      index: 0,
      bindGroupId,
      bufferIds: [targetId],
      dynamicOffsets: [],
      effectiveDynamicOffsets: [],
      dynamicOffsetsDataStart: null,
      dynamicOffsetsDataLength: null,
    });
    sequence += 1;
    events.push({
      method,
      phase: phaseName,
      sequence,
      passId,
      commandEncoderId,
      deviceId,
      pipelineId,
      bindGroupIds: [bindGroupId],
      boundBufferIds: [targetId],
      groups: [{
        index: 0,
        bindGroupId,
        bufferIds: [targetId],
      }],
    });
    sequence += 1;
    events.push({
      method: 'finishCommandEncoder',
      phase: phaseName,
      sequence,
      commandEncoderId,
      commandBufferId,
      deviceId,
    });
    sequence += 1;
    events.push({
      method: 'queueSubmit',
      phase: phaseName,
      sequence,
      deviceId,
      queueId,
      commandBufferIds: [commandBufferId],
    });
  };

  phase('ort-run-1');
  command({
    method: 'dispatchWorkgroups',
    phaseName: 'ort-run-1',
    passId: 11,
    bindGroupId: 101,
    commandEncoderId: 301,
    commandBufferId: 501,
    pipelineId: 401,
    kind: 'compute',
  });
  phase('three-proof-1');
  command({
    method: 'draw',
    phaseName: 'three-proof-1',
    passId: 21,
    bindGroupId: 201,
    commandEncoderId: 302,
    commandBufferId: 502,
    pipelineId: 402,
    kind: 'render',
    pipelineDescriptor: {
      layoutId: 801,
      vertex: { moduleId: 901 },
      fragment: { moduleId: 902 },
    },
  });
  phase('screenshot-ready');
  phase('three-dispose-1');
  phase('ort-run-2');
  command({
    method: 'dispatchWorkgroups',
    phaseName: 'ort-run-2',
    passId: 12,
    bindGroupId: 102,
    commandEncoderId: 303,
    commandBufferId: 503,
    pipelineId: 403,
    kind: 'compute',
  });
  phase('three-proof-2');
  command({
    method: 'draw',
    phaseName: 'three-proof-2',
    passId: 22,
    bindGroupId: 202,
    commandEncoderId: 304,
    commandBufferId: 504,
    pipelineId: 404,
    kind: 'render',
    pipelineDescriptor: {
      layoutId: 802,
      vertex: { moduleId: 903 },
      fragment: { moduleId: 904 },
    },
  });
  phase('three-dispose-2');
  phase('ort-cleanup');
  phase('owner-cleanup');
  sequence += 1;
  events.push({
    method: 'destroyBuffer',
    phase: 'owner-cleanup',
    sequence,
    bufferId: targetId,
  });

  return {
    kind: 'three-ort-gpu-buffer-instrumentation-v1',
    setupError: null,
    instrumentationFailures: [],
    target: {
      bufferId: targetId,
      deviceId,
      physicalBytes: protocol.target.physicalBytes,
      logicalBytes: protocol.target.logicalBytes,
    },
    acquisition: {
      adapterRequests: [{}],
      deviceRequests: [{}],
    },
    resources: {
      devices: [{ deviceId }],
      buffers: [{
        bufferId: targetId,
        deviceId,
        size: protocol.target.physicalBytes,
        usage: 0x8c,
        mapStateAtCreation: 'unmapped',
        descriptor: {
          size: protocol.target.physicalBytes,
          usage: 0x8c,
          mappedAtCreation: false,
        },
      }],
      bindGroups,
      bindGroupLayouts: [
        {
          bindGroupLayoutId: 701,
          deviceId,
          entries: [{
            binding: 0,
            buffer: { type: 'read-only-storage', hasDynamicOffset: false },
          }],
        },
        {
          bindGroupLayoutId: 702,
          deviceId,
          entries: [{
            binding: 0,
            buffer: { type: 'read-only-storage', hasDynamicOffset: false },
          }],
        },
      ],
      commandEncoders,
      commandBuffers,
      passes,
      pipelineLayouts: [
        { pipelineLayoutId: 801, deviceId, bindGroupLayoutIds: [701] },
        { pipelineLayoutId: 802, deviceId, bindGroupLayoutIds: [702] },
      ],
      pipelines,
      queues: [{ deviceId, queueId }],
      shaderModules: [
        { moduleId: 901, deviceId, phase: 'three-proof-1' },
        { moduleId: 902, deviceId, phase: 'three-proof-1' },
        { moduleId: 903, deviceId, phase: 'three-proof-2' },
        { moduleId: 904, deviceId, phase: 'three-proof-2' },
      ],
    },
    events,
    phaseTransitions,
    device: {
      errorScopes: {
        pushes: [{ filter: 'validation' }],
        pops: [{ settled: true, error: null, rejection: null }],
      },
      pipelinePromiseRejections: [],
      uncapturedErrors: [],
      unexpectedLosses: [],
    },
    shaderModules: [
      {
        moduleId: 901,
        phase: 'three-proof-1',
        code: syntheticVertexShader('runOneVertexValues'),
        messages: [],
      },
      {
        moduleId: 902,
        phase: 'three-proof-1',
        code: syntheticFragmentShader('runOneFragmentValues'),
        messages: [],
      },
      {
        moduleId: 903,
        phase: 'three-proof-2',
        code: syntheticVertexShader('runTwoVertexValues'),
        messages: [],
      },
      {
        moduleId: 904,
        phase: 'three-proof-2',
        code: syntheticFragmentShader('runTwoFragmentValues'),
        messages: [],
      },
    ],
    windowErrors: [],
    unhandledRejections: [],
  };
}

function syntheticEvidence(protocol) {
  const instrumentation = syntheticInstrumentation(protocol);
  return {
    page: {
      kind: ORT_GPU_BUFFER_CANARY_KIND,
      status: 'passed',
      error: null,
      model: {
        verified: true,
        actualBytes: protocol.model.bytes,
        actualSha256: protocol.model.sha256,
      },
      routes: {
        ortBundle: `${protocol.onnxRuntime.route}ort.webgpu.bundle.min.mjs`,
        ortWasm: `${protocol.onnxRuntime.route}ort-wasm-simd-threaded.asyncify.wasm`,
        threeBundle: `${protocol.build.route}three.webgpu.js`,
        threeTsl: `${protocol.build.route}three.tsl.js`,
      },
      environment: {
        navigatorGpu: true,
        adapterAcquired: true,
        deviceAcquired: true,
        ortProxy: protocol.onnxRuntime.proxy,
        ortThreads: protocol.onnxRuntime.numThreads,
        ortAdapterSame: true,
        ortExecutionProviderDeviceSupplied: true,
        ortExecutionProviderDeviceSame: true,
        ortDeviceSame: null,
        deviceLost: null,
        uncapturedErrors: [],
        instrumentation: {
          phaseHook: true,
          targetHook: true,
          collector: true,
          targetRegistrationResult: instrumentation.target.bufferId,
        },
      },
      target: {
        physicalBytes: protocol.target.physicalBytes,
        logicalBytes: protocol.target.logicalBytes,
        mapState: 'unmapped',
        outputTensorType: 'float32',
        outputTensorDims: [...protocol.model.dims],
        outputTensorBufferSame: true,
      },
      runs: [syntheticPageRun(protocol, 0), syntheticPageRun(protocol, 1)],
      lifecycle: {
        screenshotReady: true,
        continuationCalls: 1,
        firstThreeDisposed: true,
        secondThreeDisposed: true,
        targetReusedByOrt: true,
        targetReusedByThree: true,
        ortReleasedAfterThree: true,
        outputTensorDisposed: false,
        ownerDestroyedTarget: true,
        targetDestroyCountBeforeOwnerCleanup: 0,
        targetDestroyCountAfterOwnerCleanup: 1,
        cleanupErrors: [],
      },
    },
    instrumentation,
  };
}

function syntheticObservations(protocol) {
  const ortBundlePath = `${protocol.onnxRuntime.route}ort.webgpu.bundle.min.mjs`;
  const ortWasmPath = `${protocol.onnxRuntime.route}ort-wasm-simd-threaded.asyncify.wasm`;
  return {
    browserDisconnects: [],
    consoleErrors: [],
    consoleWarnings: [],
    httpErrors: [],
    pageCrashes: [],
    pageErrors: [],
    requestFailures: [],
    unexpectedRequests: [],
    requestedPaths: [
      ortBundlePath,
      ortWasmPath,
    ],
    serverRequests: [
      {
        path: ortBundlePath,
        status: 200,
        contentType: 'text/javascript; charset=utf-8',
        bytes: protocol.onnxRuntime.files['ort.webgpu.bundle.min.mjs'].bytes,
      },
      {
        path: ortWasmPath,
        status: 200,
        contentType: 'application/wasm',
        bytes: protocol.onnxRuntime.files['ort-wasm-simd-threaded.asyncify.wasm'].bytes,
      },
    ],
  };
}

function resequenceInstrumentation(instrumentation) {
  instrumentation.events.forEach((event, index) => {
    event.sequence = index + 1;
  });
  const phaseSequences = new Map(instrumentation.events
    .filter((event) => event.method === 'phase')
    .map((event) => [event.phase, event.sequence]));
  instrumentation.phaseTransitions.forEach((transition) => {
    transition.sequence = phaseSequences.get(transition.phase);
  });
}

function insertBeforeCommand(instrumentation, command, event) {
  const commandIndex = instrumentation.events.indexOf(command);
  assert.notEqual(commandIndex, -1);
  instrumentation.events.splice(commandIndex, 0, event);
  resequenceInstrumentation(instrumentation);
}

function removeEvent(instrumentation, predicate) {
  const index = instrumentation.events.findIndex(predicate);
  assert.notEqual(index, -1);
  instrumentation.events.splice(index, 1);
  resequenceInstrumentation(instrumentation);
}

function insertTargetCommand(instrumentation, {
  phase,
  method,
  kind,
  bindGroupId,
  passId,
  commandEncoderId,
  commandBufferId,
  pipelineId,
}) {
  const deviceId = instrumentation.target.deviceId;
  const queueId = instrumentation.resources.queues[0].queueId;
  instrumentation.resources.commandEncoders.push({ commandEncoderId, deviceId, phase });
  instrumentation.resources.commandBuffers.push({
    commandBufferId,
    commandEncoderId,
    deviceId,
    phase,
  });
  instrumentation.resources.passes.push({
    commandEncoderId,
    deviceId,
    kind: `${kind}-pass`,
    passId,
    phase,
  });
  instrumentation.resources.pipelines.push({ pipelineId, deviceId, kind, phase });

  const phaseStart = instrumentation.events.findIndex((event) => (
    event.method === 'phase' && event.phase === phase
  ));
  assert.notEqual(phaseStart, -1);
  let insertionIndex = instrumentation.events.findIndex((event, index) => (
    index > phaseStart && event.method === 'phase'
  ));
  if (insertionIndex === -1) insertionIndex = instrumentation.events.length;
  instrumentation.events.splice(insertionIndex, 0,
    {
      method: 'setPipeline',
      phase,
      passId,
      commandEncoderId,
      deviceId,
      pipelineId,
    },
    {
      method: 'setBindGroup',
      phase,
      passId,
      commandEncoderId,
      deviceId,
      index: 0,
      bindGroupId,
      bufferIds: [instrumentation.target.bufferId],
      dynamicOffsets: [],
      effectiveDynamicOffsets: [],
      dynamicOffsetsDataStart: null,
      dynamicOffsetsDataLength: null,
    },
    {
      method,
      phase,
      passId,
      commandEncoderId,
      deviceId,
      pipelineId,
      bindGroupIds: [bindGroupId],
      boundBufferIds: [instrumentation.target.bufferId],
      groups: [{
        index: 0,
        bindGroupId,
        bufferIds: [instrumentation.target.bufferId],
      }],
    },
    {
      method: 'finishCommandEncoder',
      phase,
      commandEncoderId,
      commandBufferId,
      deviceId,
    },
    {
      method: 'queueSubmit',
      phase,
      deviceId,
      queueId,
      commandBufferIds: [commandBufferId],
    });
  resequenceInstrumentation(instrumentation);
}

function assertRejected(protocol, evidence, reasonPattern) {
  const result = validateOrtGpuBufferEvidence(
    evidence,
    protocol,
    syntheticObservations(protocol),
  );
  assert.equal(result.verified, false);
  assert.match(result.reasons.join('\n'), reasonPattern);
}

test('ORT canary embeds the exact official 130-byte model', () => {
  const bytes = decodeOrtModelBytes();
  assert.equal(bytes.byteLength, ORT_MODEL_BYTES);
  assert.equal(sha256(bytes), ORT_MODEL_SHA256);
});

test('ORT canary vectors have deterministic float32 products and distinct second-run bits', () => {
  assert.equal(ORT_CANARY_RUNS.length, 2);
  for (const run of ORT_CANARY_RUNS) {
    const products = run.input.map((value, index) => (
      Math.fround(Math.fround(value) * Math.fround(index + 1))
    ));
    assert.deepEqual(products, run.expectedOutput);
    assert.deepEqual(floatBits(products), run.expectedBits);
  }
  assert.equal(
    ORT_CANARY_RUNS[0].expectedBits.every((value, index) => (
      value !== ORT_CANARY_RUNS[1].expectedBits[index]
    )),
    true,
  );
});

test('ORT canary exact integer oracle checks bits, coverage, and GPU-driven placement', () => {
  for (const run of ORT_CANARY_RUNS) {
    const analysis = analyzeOrtGpuBufferOracle(syntheticOracle(run.runIndex), run.runIndex);
    assert.equal(analysis.exact, true);
    assert.deepEqual(analysis.expectedPixelTuples, expectedOrtGpuBufferTuples(run.runIndex));
    assert.equal(analysis.unexpected.length, 0);
    assert.deepEqual(analysis.spatialMismatches, []);
    assert.equal(analysis.tuples.every((tuple) => tuple.count === 484), true);
  }

  const corrupt = analyzeOrtGpuBufferOracle(syntheticOracle(0, { corrupt: true }), 0);
  assert.equal(corrupt.exact, false);
  assert.equal(corrupt.unexpected.length, 1);

  const misplaced = analyzeOrtGpuBufferOracle(
    syntheticOracle(0, { shiftFirstMarker: 8 }),
    0,
  );
  assert.equal(misplaced.exact, false);
  assert.deepEqual(misplaced.spatialMismatches, [0]);
});

test('ORT GPU-buffer evidence validator accepts a minimal valid trace', async () => {
  const protocol = await readProtocol();
  const result = validateOrtGpuBufferEvidence(
    syntheticEvidence(protocol),
    protocol,
    syntheticObservations(protocol),
  );
  assert.equal(result.verified, true, result.reasons.join('\n'));
  assert.deepEqual(result.reasons, []);
});

test('ORT GPU-buffer evidence rejects a target group rebound away before a command', async (t) => {
  const protocol = await readProtocol();
  for (const scenario of [
    { label: 'ORT dispatch', phase: 'ort-run-1', method: 'dispatchWorkgroups' },
    { label: 'Three draw', phase: 'three-proof-1', method: 'draw' },
  ]) {
    await t.test(scenario.label, () => {
      const evidence = syntheticEvidence(protocol);
      const command = evidence.instrumentation.events.find((event) => (
        event.phase === scenario.phase && event.method === scenario.method
      ));
      insertBeforeCommand(evidence.instrumentation, command, {
        method: 'setBindGroup',
        phase: scenario.phase,
        passId: command.passId,
        commandEncoderId: command.commandEncoderId,
        deviceId: command.deviceId,
        index: 0,
        bindGroupId: 999,
        bufferIds: [8],
        dynamicOffsets: [],
        effectiveDynamicOffsets: [],
        dynamicOffsetsDataStart: null,
        dynamicOffsetsDataLength: null,
      });
      assertRejected(protocol, evidence, /target|bind|active/iu);
    });
  }
});

test('ORT GPU-buffer evidence requires command encoder finish and queue submission', async (t) => {
  const protocol = await readProtocol();
  await t.test('missing finish', () => {
    const evidence = syntheticEvidence(protocol);
    const command = evidence.instrumentation.events.find((event) => (
      event.phase === 'ort-run-1' && event.method === 'dispatchWorkgroups'
    ));
    removeEvent(evidence.instrumentation, (event) => (
      event.method === 'finishCommandEncoder'
        && event.commandEncoderId === command.commandEncoderId
    ));
    assertRejected(protocol, evidence, /finished and submitted/iu);
  });
  await t.test('missing submit', () => {
    const evidence = syntheticEvidence(protocol);
    const command = evidence.instrumentation.events.find((event) => (
      event.phase === 'ort-run-1' && event.method === 'dispatchWorkgroups'
    ));
    const finish = evidence.instrumentation.events.find((event) => (
      event.method === 'finishCommandEncoder'
        && event.commandEncoderId === command.commandEncoderId
    ));
    removeEvent(evidence.instrumentation, (event) => (
      event.method === 'queueSubmit'
        && event.commandBufferIds?.includes(finish.commandBufferId)
    ));
    assertRejected(protocol, evidence, /finished and submitted/iu);
  });
  await t.test('wrong submit device', () => {
    const evidence = syntheticEvidence(protocol);
    const submit = evidence.instrumentation.events.find((event) => (
      event.phase === 'ort-run-1' && event.method === 'queueSubmit'
    ));
    submit.deviceId = 2;
    assertRejected(protocol, evidence, /finished and submitted/iu);
  });
  await t.test('submit before finish', () => {
    const evidence = syntheticEvidence(protocol);
    const events = evidence.instrumentation.events;
    const finishIndex = events.findIndex((event) => (
      event.phase === 'ort-run-1' && event.method === 'finishCommandEncoder'
    ));
    const submitIndex = events.findIndex((event) => (
      event.phase === 'ort-run-1' && event.method === 'queueSubmit'
    ));
    const [submit] = events.splice(submitIndex, 1);
    events.splice(finishIndex, 0, submit);
    resequenceInstrumentation(evidence.instrumentation);
    assertRejected(protocol, evidence, /finished and submitted/iu);
  });
});

test('ORT GPU-buffer evidence rejects target-active commands outside allowed phases', async (t) => {
  const protocol = await readProtocol();
  await t.test('dispatch outside ORT run', () => {
    const evidence = syntheticEvidence(protocol);
    insertTargetCommand(evidence.instrumentation, {
      phase: 'three-proof-1',
      method: 'dispatchWorkgroups',
      kind: 'compute',
      bindGroupId: 101,
      passId: 31,
      commandEncoderId: 305,
      commandBufferId: 505,
      pipelineId: 405,
    });
    assertRejected(protocol, evidence, /outside its allowed phase/iu);
  });
  await t.test('draw outside Three proof', () => {
    const evidence = syntheticEvidence(protocol);
    insertTargetCommand(evidence.instrumentation, {
      phase: 'ort-run-1',
      method: 'draw',
      kind: 'render',
      bindGroupId: 201,
      passId: 32,
      commandEncoderId: 306,
      commandBufferId: 506,
      pipelineId: 406,
    });
    assertRejected(protocol, evidence, /outside its allowed phase/iu);
  });
});

test('ORT GPU-buffer evidence requires the correct active pipeline', async (t) => {
  const protocol = await readProtocol();
  await t.test('missing active pipeline', () => {
    const evidence = syntheticEvidence(protocol);
    const command = evidence.instrumentation.events.find((event) => (
      event.phase === 'ort-run-1' && event.method === 'dispatchWorkgroups'
    ));
    removeEvent(evidence.instrumentation, (event) => (
      event.method === 'setPipeline' && event.passId === command.passId
    ));
    assertRejected(protocol, evidence, /pipeline/iu);
  });
  await t.test('wrong active pipeline', () => {
    const evidence = syntheticEvidence(protocol);
    const setPipeline = evidence.instrumentation.events.find((event) => (
      event.phase === 'ort-run-1' && event.method === 'setPipeline'
    ));
    setPipeline.pipelineId = 499;
    evidence.instrumentation.resources.pipelines.push({
      pipelineId: 499,
      deviceId: evidence.instrumentation.target.deviceId,
      kind: 'compute',
      phase: 'ort-run-1',
    });
    assertRejected(protocol, evidence, /pipeline/iu);
  });
  await t.test('wrong pipeline kind', () => {
    const evidence = syntheticEvidence(protocol);
    const pipeline = evidence.instrumentation.resources.pipelines.find((candidate) => (
      candidate.pipelineId === 401
    ));
    pipeline.kind = 'render';
    assertRejected(protocol, evidence, /pipeline/iu);
  });
});

test('ORT GPU-buffer evidence rejects dynamic offsets on a Three target binding', async () => {
  const protocol = await readProtocol();
  const evidence = syntheticEvidence(protocol);
  const setGroup = evidence.instrumentation.events.find((event) => (
    event.phase === 'three-proof-1'
      && event.method === 'setBindGroup'
      && event.bufferIds.includes(evidence.instrumentation.target.bufferId)
  ));
  setGroup.dynamicOffsets = [4];
  setGroup.effectiveDynamicOffsets = [4];
  assertRejected(protocol, evidence, /dynamic offset/iu);
});

test('ORT GPU-buffer evidence requires a read-only Three storage witness', async (t) => {
  const protocol = await readProtocol();
  await t.test('writable bind-group layout', () => {
    const evidence = syntheticEvidence(protocol);
    const layout = evidence.instrumentation.resources.bindGroupLayouts.find((candidate) => (
      candidate.bindGroupLayoutId === 701
    ));
    layout.entries[0].buffer.type = 'storage';
    assertRejected(protocol, evidence, /read-only storage/iu);
  });
  await t.test('writable WGSL declaration in only one stage', () => {
    const evidence = syntheticEvidence(protocol);
    const module = evidence.instrumentation.shaderModules.find((candidate) => (
      candidate.moduleId === 901
    ));
    module.code = module.code.replace('var<storage, read>', 'var<storage, read_write>');
    assertRejected(protocol, evidence, /read-only storage/iu);
  });
  await t.test('writable target declaration followed by a read-only decoy', () => {
    const evidence = syntheticEvidence(protocol);
    const module = evidence.instrumentation.shaderModules.find((candidate) => (
      candidate.moduleId === 901
    ));
    module.code = [
      '@group(0) @binding(0)',
      'var<storage, read_write> targetValues: array<f32>;',
      'var<storage, read> decoyValues: array<f32>;',
    ].join(' ');
    assertRejected(protocol, evidence, /read-only storage/iu);
  });
  await t.test('target declaration present but unused', () => {
    const evidence = syntheticEvidence(protocol);
    const module = evidence.instrumentation.shaderModules.find((candidate) => (
      candidate.moduleId === 901
    ));
    module.code = '@group(0) @binding(0) var<storage, read> targetValues: array<f32>;';
    assertRejected(protocol, evidence, /read-only|post-declaration|both-stage/iu);
  });
  await t.test('target read only once beyond its declaration', () => {
    const evidence = syntheticEvidence(protocol);
    const module = evidence.instrumentation.shaderModules.find((candidate) => (
      candidate.moduleId === 901
    ));
    module.code = [
      '@group(0) @binding(0) var<storage, read> targetValues: array<f32>;',
      'fn oneRead() -> f32 { return targetValues[0]; }',
    ].join('\n');
    assertRejected(protocol, evidence, /read-only|post-declaration|both-stage/iu);
  });
  await t.test('target witness missing from only one stage', () => {
    const evidence = syntheticEvidence(protocol);
    const module = evidence.instrumentation.shaderModules.find((candidate) => (
      candidate.moduleId === 902
    ));
    module.code = [
      '@fragment fn main() -> @location(0) vec4u {',
      '  return vec4u(0u, 0u, 0u, 1u);',
      '}',
    ].join('\n');
    assertRejected(protocol, evidence, /read-only|post-declaration|both-stage/iu);
  });
});

test('ORT canary page source verifier enforces the GPU-only read-only contract', async (t) => {
  const source = await readFile(pageModulePath, 'utf8');
  const valid = verifyOrtGpuBufferPageSource(source);
  assert.equal(valid.verified, true, valid.reasons.join('\n'));

  for (const scenario of [
    {
      label: 'oracle position y disconnected from target storage',
      source: replaceOccurrence(
        source,
        'positionGeometry.y.mul(float(ORT_MARKER_NDC_SIZE)).add(y)',
        'positionGeometry.y.mul(float(ORT_MARKER_NDC_SIZE)).add(x)',
        0,
      ),
      reason: /exact oracle position/iu,
    },
    {
      label: 'oracle y bits disconnected from target storage',
      source: source.replace('floatBitsToUint(y)', 'floatBitsToUint(x)'),
      reason: /exact oracle fragment/iu,
    },
    {
      label: 'visual position y disconnected from target storage',
      source: replaceOccurrence(
        source,
        'positionGeometry.y.mul(float(ORT_MARKER_NDC_SIZE)).add(y)',
        'positionGeometry.y.mul(float(ORT_MARKER_NDC_SIZE)).add(x)',
        1,
      ),
      reason: /visual position/iu,
    },
    {
      label: 'visual y channel disconnected from target storage',
      source: source.replace('y.add(1).mul(0.5)', 'x.add(1).mul(0.5)'),
      reason: /visual fragment/iu,
    },
    {
      label: 'writable Three storage path',
      source: source.replace('.toReadOnly()', '.toReadWrite()'),
      reason: /read-only|writable/iu,
    },
    {
      label: 'target download',
      source: `${source}\nvoid outputTensor.getData();`,
      reason: /getData/iu,
    },
    {
      label: 'output tensor disposal',
      source: `${source}\noutputTensor.dispose();`,
      reason: /disposes/iu,
    },
  ]) {
    await t.test(scenario.label, () => {
      const result = verifyOrtGpuBufferPageSource(scenario.source);
      assert.equal(result.verified, false);
      assert.match(result.reasons.join('\n'), scenario.reason);
    });
  }
});

test('ORT GPU-buffer evidence rejects an incorrect Three logical binding range', async (t) => {
  const protocol = await readProtocol();
  for (const mutation of [
    { label: 'offset', property: 'offset', value: 4 },
    { label: 'size', property: 'size', value: ORT_TARGET_PHYSICAL_BYTES },
  ]) {
    await t.test(mutation.label, () => {
      const evidence = syntheticEvidence(protocol);
      const group = evidence.instrumentation.resources.bindGroups
        .find((candidate) => candidate.bindGroupId === 201);
      group.entries[0].resource[mutation.property] = mutation.value;
      assertRejected(protocol, evidence, /explicitly bind target range/iu);
    });
  }
});

test('ORT GPU-buffer evidence rejects every prohibited target operation', async (t) => {
  const protocol = await readProtocol();
  const cases = [
    {
      label: 'CPU write',
      event: { method: 'writeBuffer', bufferId: 7 },
      reason: /queue\.writeBuffer/iu,
    },
    {
      label: 'GPU copy',
      event: { method: 'copyBufferToBuffer', sourceBufferId: 7, destinationBufferId: 8 },
      reason: /GPU copy/iu,
    },
    {
      label: 'mapping',
      event: { method: 'mapAsync', bufferId: 7 },
      reason: /mapped/iu,
    },
    {
      label: 'clear',
      event: { method: 'clearBuffer', bufferId: 7 },
      reason: /cleared/iu,
    },
    {
      label: 'vertex input',
      event: { method: 'setVertexBuffer', bufferId: 7, slot: 0, offset: 0 },
      reason: /vertex or index input/iu,
    },
    {
      label: 'index input',
      event: { method: 'setIndexBuffer', bufferId: 7, format: 'uint32', offset: 0 },
      reason: /vertex or index input/iu,
    },
    {
      label: 'query resolve destination',
      event: {
        method: 'resolveQuerySet',
        querySetId: 71,
        firstQuery: 0,
        queryCount: 1,
        destinationBufferId: 7,
        destinationOffset: 0,
      },
      reason: /GPU copy or query resolve/iu,
    },
  ];
  for (const scenario of cases) {
    await t.test(scenario.label, () => {
      const evidence = syntheticEvidence(protocol);
      const ownerDestroy = evidence.instrumentation.events.find((event) => (
        event.method === 'destroyBuffer'
      ));
      insertBeforeCommand(evidence.instrumentation, ownerDestroy, {
        ...scenario.event,
        phase: 'three-proof-1',
      });
      assertRejected(protocol, evidence, scenario.reason);
    });
  }

  await t.test('indirect draw arguments', () => {
    const evidence = syntheticEvidence(protocol);
    const draw = evidence.instrumentation.events.find((event) => (
      event.phase === 'three-proof-1' && event.method === 'draw'
    ));
    draw.method = 'drawIndirect';
    draw.indirectBufferId = evidence.instrumentation.target.bufferId;
    draw.indirectOffset = 0;
    assertRejected(protocol, evidence, /indirect command/iu);
  });
});

test('ORT GPU-buffer evidence rejects early or repeated target destruction', async (t) => {
  const protocol = await readProtocol();
  await t.test('early destroy', () => {
    const evidence = syntheticEvidence(protocol);
    const destroy = evidence.instrumentation.events.find((event) => (
      event.method === 'destroyBuffer'
    ));
    destroy.phase = 'three-dispose-2';
    assertRejected(protocol, evidence, /destroyed exactly once by its owner/iu);
  });
  await t.test('more than once', () => {
    const evidence = syntheticEvidence(protocol);
    const destroy = evidence.instrumentation.events.find((event) => (
      event.method === 'destroyBuffer'
    ));
    insertBeforeCommand(evidence.instrumentation, destroy, {
      method: 'destroyBuffer',
      phase: 'owner-cleanup',
      bufferId: 7,
    });
    assertRejected(protocol, evidence, /destroyed exactly once by its owner/iu);
  });
});

test('ORT GPU-buffer evidence rejects target GPUDevice destruction or loss', async (t) => {
  const protocol = await readProtocol();
  await t.test('explicit target-device destroy', () => {
    const evidence = syntheticEvidence(protocol);
    const ownerDestroy = evidence.instrumentation.events.find((event) => (
      event.method === 'destroyBuffer'
    ));
    insertBeforeCommand(evidence.instrumentation, ownerDestroy, {
      method: 'destroyDevice',
      phase: 'owner-cleanup',
      deviceId: evidence.instrumentation.target.deviceId,
    });
    assertRejected(protocol, evidence, /target GPUDevice was destroyed/iu);
  });
  await t.test('unexpected target-device loss', () => {
    const evidence = syntheticEvidence(protocol);
    evidence.instrumentation.device.unexpectedLosses.push({
      deviceId: evidence.instrumentation.target.deviceId,
      message: 'synthetic target-device loss',
      phase: 'three-proof-2',
      reason: 'unknown',
    });
    assertRejected(protocol, evidence, /loss of the target GPUDevice/iu);
  });
});

test('ORT GPU-buffer evidence rejects commands issued on the wrong device', async (t) => {
  const protocol = await readProtocol();
  for (const scenario of [
    { label: 'ORT dispatch', phase: 'ort-run-1', method: 'dispatchWorkgroups' },
    { label: 'Three draw', phase: 'three-proof-1', method: 'draw' },
  ]) {
    await t.test(scenario.label, () => {
      const evidence = syntheticEvidence(protocol);
      const command = evidence.instrumentation.events.find((event) => (
        event.phase === scenario.phase && event.method === scenario.method
      ));
      command.deviceId = 2;
      assertRejected(protocol, evidence, /target|GPUDevice|binding/iu);
    });
  }
});

test('ORT GPU-buffer evidence rejects missing or reordered critical phases', async (t) => {
  const protocol = await readProtocol();
  await t.test('missing phase', () => {
    const evidence = syntheticEvidence(protocol);
    evidence.instrumentation.phaseTransitions = evidence.instrumentation.phaseTransitions
      .filter((transition) => transition.phase !== 'screenshot-ready');
    assertRejected(protocol, evidence, /phase order differs/iu);
  });
  await t.test('reordered phase', () => {
    const evidence = syntheticEvidence(protocol);
    const transitions = evidence.instrumentation.phaseTransitions;
    [transitions[3], transitions[4]] = [transitions[4], transitions[3]];
    assertRejected(protocol, evidence, /phase sequence|phase order differs/iu);
  });
});

test('ORT canary protocol pins the candidate, runtime, model, and bounded target range', async () => {
  const protocol = await readProtocol();
  assert.equal(protocol.schemaVersion, 1);
  assert.equal(protocol.candidate.commit, '31fdc388bc2d001f9a7d7ca27f52ca2c411b1cdc');
  assert.equal(protocol.candidate.tree, '8615edca00b47afe7fb32849182bd4a5ec3b687f');
  assert.equal(protocol.onnxRuntime.version, '1.29.0');
  assert.equal(protocol.model.bytes, ORT_MODEL_BYTES);
  assert.equal(protocol.model.sha256, ORT_MODEL_SHA256);
  assert.equal(protocol.target.physicalBytes, ORT_TARGET_PHYSICAL_BYTES);
  assert.equal(protocol.target.logicalBytes, ORT_TARGET_LOGICAL_BYTES);
  assert.equal(protocol.target.count, ORT_TARGET_ITEM_COUNT);
  assert.equal(protocol.oracle.width, ORT_ORACLE_SIZE.width);
  assert.equal(protocol.oracle.height, ORT_ORACLE_SIZE.height);
  assert.equal(protocol.oracle.sentinel, ORT_ORACLE_SENTINEL);
  assert.deepEqual(protocol.runs.map((run) => run.expectedBits), (
    ORT_CANARY_RUNS.map((run) => [...run.expectedBits])
  ));
  assert.match(ORT_GPU_BUFFER_CANARY_KIND, /borrowed-gpu-buffer/u);
  for (const file of Object.values(protocol.build.files)) {
    assert.equal(Number.isSafeInteger(file.bytes) && file.bytes > 0, true);
    assert.match(file.sha256, /^[a-f0-9]{64}$/u);
  }
  for (const file of Object.values(protocol.onnxRuntime.files)) {
    assert.equal(Number.isSafeInteger(file.bytes) && file.bytes > 0, true);
    assert.match(file.sha256, /^[a-f0-9]{64}$/u);
  }
});
