import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BATCH_ORDER,
  INDIRECT_OFFSETS,
  RERECORD_INDIRECT_OFFSETS,
  createIndirectCommands,
  expectedBatchedControlOutput,
  expectedBatchedOutput,
  expectedDirectOutput,
  expectedIndirectOutput,
} from '../src/three-draw-index-canary-contract.js';
import {
  findForbiddenThreeDrawIndexBrowserArguments,
  parseThreeDrawIndexArguments,
  validateThreeDrawIndexPageResult,
  validateThreeDrawIndexOverlayManifest,
  validateThreeDrawIndexProtocol,
  validateThreeDrawIndexReport,
} from '../scripts/probe-three-draw-index.mjs';
import { buildThreeDrawIndexOverlay } from '../scripts/build-three-draw-index-overlay.mjs';

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

const BOUND_CANARY_PATHS = [
  'scripts/probe-three-draw-index.mjs',
  'scripts/build-three-draw-index-overlay.mjs',
  'src/three-draw-index-canary.js',
  'src/three-draw-index-canary-contract.js',
  'three-draw-index-canary.html',
  'protocols/three-draw-index-canary-v1.json',
  'docs/WEBGPU_DRAW_INDEX_IMMEDIATES_PROTOCOL.md',
  'patches/three-webgpu-draw-index-immediates.patch',
  'patches/three-webgpu-draw-index-immediates.json',
  'package.json',
  'package-lock.json',
];

async function makeBoundSourceSnapshot() {
  const files = await Promise.all(BOUND_CANARY_PATHS.map(async (relativePath) => {
    const bytes = await readFile(relativePath);
    return { path: relativePath, byteLength: bytes.length, sha256: sha256(bytes) };
  }));
  return {
    schemaVersion: 1,
    files,
    aggregateSha256: sha256(Buffer.from(JSON.stringify(files), 'utf8')),
  };
}

async function makeServerStartFailureReport(protocol, manifest) {
  const protocolBytes = Buffer.from(`${JSON.stringify(protocol, null, 2)}\n`);
  const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  const boundSnapshot = await makeBoundSourceSnapshot();
  const gitSnapshot = {
    commit: '1'.repeat(40),
    tree: '2'.repeat(40),
    branch: 'experiment/webgpu-draw-index-immediates',
    statusPorcelain: '',
    clean: true,
    trackedFileCount: 42,
    trackedWorkingBytesSha256: '3'.repeat(64),
  };
  const executable = {
    path: protocol.browser.requiredExecutablePath.replaceAll('\\', '/'),
    byteLength: protocol.browser.requiredByteLength,
    sha256: protocol.browser.requiredSha256,
  };
  const overlayIdentity = {
    valid: true,
    reasons: [],
    fileCount: manifest.output.fileCount,
    totalBytes: manifest.output.totalBytes,
    inventorySha256: manifest.output.inventorySha256,
    manifestSha256: sha256(manifestBytes),
    pathCount: manifest.output.fileCount + 1,
  };
  const failure = { name: 'Error', message: 'synthetic server start failure', stack: null };
  return {
    schemaVersion: 1,
    kind: 'three-webgpu-draw-index-default-exposure-report',
    status: 'failed',
    executionMode: 'timing-free-one-shot-correctness',
    timingCaptured: false,
    benchmarkClaim: false,
    claimBoundary: protocol.claimBoundary,
    executionPolicy: {
      attemptCount: 1,
      retryCount: 0,
      browserLaunchCount: 1,
      deviceCountLimit: 1,
      customBrowserArguments: [],
      timingCaptured: false,
    },
    observedLifecycle: {
      serverStartAttempts: 1,
      serverInstancesCreated: 0,
      serverStartSuccesses: 0,
      browserLaunchAttempts: 0,
      browserLaunchSuccesses: 0,
      contextsCreated: 0,
      pagesCreated: 0,
      postflightChecksAttempted: 4,
      postflightChecksSucceeded: 4,
    },
    artifactPolicy: {
      reportPath: protocol.output.reportPath,
      sha256Path: protocol.output.sha256Path,
      exclusiveDirectory: true,
      overwrite: false,
    },
    protocol: {
      path: 'protocols/three-draw-index-canary-v1.json',
      byteLength: protocolBytes.length,
      sha256: sha256(protocolBytes),
      value: protocol,
    },
    sourceIdentity: {
      filesStart: boundSnapshot,
      filesEnd: structuredClone(boundSnapshot),
      filesMatch: true,
      gitStart: gitSnapshot,
      gitEnd: structuredClone(gitSnapshot),
      gitMatch: true,
    },
    overlay: {
      manifest,
      manifestSha256: sha256(manifestBytes),
      validation: validateThreeDrawIndexOverlayManifest(manifest),
      identityStart: overlayIdentity,
      identityEnd: structuredClone(overlayIdentity),
      identityMatch: true,
    },
    browser: {
      launchApi: 'playwright-core.chromium.launch',
      headless: true,
      customArguments: [],
      executableStart: executable,
      executableEnd: structuredClone(executable),
      identityMatchesStart: true,
      productVersion: null,
      effectiveArguments: null,
      forbiddenEffectiveArguments: null,
      contextClosedCleanly: false,
      closedCleanly: false,
    },
    server: {
      implementation: 'vite.createServer',
      configFile: false,
      host: '127.0.0.1',
      port: null,
      localOnly: true,
      isolationHeaders: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
      closedCleanly: false,
    },
    browserObservations: null,
    browserObservationsClean: false,
    servedSource: null,
    pageResult: null,
    pageValidation: null,
    runnerFailures: [{ stage: 'server-create', failure }],
    failure,
  };
}

function outputRecord(expected) {
  const hash = sha256(Buffer.from(expected));
  return {
    exact: true,
    encoding: 'synthetic test encoding',
    expected: [...expected],
    observed: [...expected],
    expectedSha256: hash,
    observedSha256: hash,
    mismatchCount: 0,
  };
}

let sequence = 1;

function setterEvent(phase, ordinal) {
  return {
    sequence: sequence++,
    phase,
    method: 'setImmediates',
    rangeOffset: 0,
    sourceType: 'Uint32Array',
    sourceElementCount: 1,
    dataOffset: 0,
    dataSize: 1,
    selectedValues: [ordinal],
  };
}

function immediateTrace({
  id,
  type,
  phase,
  drawMethod,
  offsets = null,
  bundleId = null,
}) {
  const events = [{
    sequence: sequence++,
    phase,
    method: 'setPipeline',
    pipelineId: 'pipeline-immediate',
  }];
  for (let ordinal = 0; ordinal < (offsets?.length ?? 1); ordinal += 1) {
    events.push(setterEvent(phase, ordinal));
    const draw = {
      sequence: sequence++,
      phase,
      method: drawMethod,
    };
    if (drawMethod === 'drawIndexedIndirect') {
      draw.bufferId = 'gpu-buffer-indirect';
      draw.indirectOffset = offsets[ordinal];
    } else {
      draw.arguments = offsets === null
        ? [3, 1, 0, 0, ordinal]
        : [3, 1, ordinal * 3, 0, ordinal];
    }
    events.push(draw);
  }
  if (type === 'bundle') {
    events.push({ sequence: sequence++, phase, method: 'finish' });
  }
  return {
    encoderId: id,
    type,
    creationPhase: phase,
    descriptor: {},
    callableSetImmediates: true,
    events,
    bundleId,
  };
}

function controlBatchTrace() {
  const phase = 'batched-control';
  const events = [{
    sequence: sequence++,
    phase,
    method: 'setPipeline',
    pipelineId: 'pipeline-control',
  }];
  for (let ordinal = 0; ordinal < 4; ordinal += 1) {
    events.push({
      sequence: sequence++,
      phase,
      method: 'drawIndexed',
      arguments: [3, 1, ordinal * 3, 0, ordinal],
    });
  }
  return {
    encoderId: 'pass-batched-control',
    type: 'pass',
    creationPhase: phase,
    descriptor: {},
    callableSetImmediates: true,
    events,
    bundleId: null,
  };
}

function executePass(phase, bundleId, suffix = '') {
  return {
    encoderId: `pass-execute-${phase}${suffix}`,
    type: 'pass',
    creationPhase: phase,
    descriptor: {},
    callableSetImmediates: true,
    events: [{
      sequence: sequence++,
      phase,
      method: 'executeBundles',
      bundleIds: [bundleId],
    }],
    bundleId: null,
  };
}

function makePageResult() {
  sequence = 1;
  const immediateVertex = [
    'requires immediate_address_space;',
    'var<immediate> nodeDrawIndex : u32;',
    'varyings.nodeVarying1 = nodeDrawIndex;',
  ].join('\n');
  const immediateFragment = [
    '@location( 0 ) @interpolate(flat, either) nodeVarying1 : u32',
    'let value = f32( nodeVarying1 );',
  ].join('\n');
  const direct = immediateTrace({
    id: 'pass-direct',
    type: 'pass',
    phase: 'direct-fragment',
    drawMethod: 'drawIndexed',
  });
  const firstBundle = immediateTrace({
    id: 'bundle-encoder-first',
    type: 'bundle',
    phase: 'indirect-bundle-first',
    drawMethod: 'drawIndexedIndirect',
    offsets: INDIRECT_OFFSETS,
    bundleId: 'bundle-first',
  });
  const rerecordedBundle = immediateTrace({
    id: 'bundle-encoder-rerecord',
    type: 'bundle',
    phase: 'indirect-bundle-rerecord',
    drawMethod: 'drawIndexedIndirect',
    offsets: RERECORD_INDIRECT_OFFSETS,
    bundleId: 'bundle-rerecorded',
  });
  const explicitBatch = immediateTrace({
    id: 'pass-batched-explicit',
    type: 'pass',
    phase: 'batched-explicit',
    drawMethod: 'drawIndexed',
    offsets: [0, 1, 2, 3],
  });
  const arrayBundle0 = immediateTrace({
    id: 'bundle-array-0',
    type: 'bundle',
    phase: 'array-camera',
    drawMethod: 'drawIndexedIndirect',
    offsets: INDIRECT_OFFSETS,
    bundleId: 'array-bundle-0',
  });
  const arrayBundle1 = immediateTrace({
    id: 'bundle-array-1',
    type: 'bundle',
    phase: 'array-camera',
    drawMethod: 'drawIndexedIndirect',
    offsets: INDIRECT_OFFSETS,
    bundleId: 'array-bundle-1',
  });
  const passes = [
    direct,
    executePass('indirect-bundle-first', 'bundle-first'),
    executePass('indirect-bundle-cached', 'bundle-first'),
    executePass('indirect-bundle-rerecord', 'bundle-rerecorded'),
    explicitBatch,
    controlBatchTrace(),
    executePass('array-camera', 'array-bundle-0', '-0'),
    executePass('array-camera', 'array-bundle-1', '-1'),
  ];
  const scopes = [
    {
      sequence: sequence++, operation: 'push', filter: 'internal', depthAfterCall: 1,
    },
    {
      sequence: sequence++, operation: 'push', filter: 'out-of-memory', depthAfterCall: 2,
    },
    {
      sequence: sequence++, operation: 'push', filter: 'validation', depthAfterCall: 3,
    },
    {
      sequence: sequence++, operation: 'pop', poppedFilter: 'validation',
      depthBeforeCall: 3, depthAfterCall: 2, settled: true,
      resolvedError: null, rejected: null,
    },
    {
      sequence: sequence++, operation: 'pop', poppedFilter: 'out-of-memory',
      depthBeforeCall: 2, depthAfterCall: 1, settled: true,
      resolvedError: null, rejected: null,
    },
    {
      sequence: sequence++, operation: 'pop', poppedFilter: 'internal',
      depthBeforeCall: 1, depthAfterCall: 0, settled: true,
      resolvedError: null, rejected: null,
    },
  ];
  const traces = (phase) => ({
    passes: passes.filter((trace) => trace.creationPhase === phase),
    bundles: [firstBundle, rerecordedBundle, arrayBundle0, arrayBundle1]
      .filter((trace) => trace.creationPhase === phase),
  });
  return {
    schemaVersion: 1,
    kind: 'three-webgpu-draw-index-correctness-canary',
    status: 'passed',
    executionMode: 'timing-free-one-shot-correctness',
    timingCaptured: false,
    benchmarkClaim: false,
    claimBoundary:
      'Correctness and integration on one standards-capable implementation; no performance or cross-device universality claim.',
    userAgent: 'test',
    preflight: {
      secureContext: true,
      currentStage: 'renderer-and-shaders',
      capabilityDecision: 'supported',
      unsupportedStage: null,
      unsupportedReason: null,
      unsupportedDetail: null,
      adapterRequestAttempts: 1,
      adapterAcquisitions: 1,
      requestAdapterArgumentsCount: 0,
      requestAdapterDescriptorSupplied: false,
      deviceRequestAttempts: 1,
      deviceAcquisitions: 1,
      requestDeviceArgumentsCount: 0,
      requestDeviceDescriptorSupplied: false,
      shaderModulesAtCapabilityDecision: 0,
      instrumentationInstalledBeforeRenderer: true,
      rendererConstructed: true,
    },
    capabilities: {
      wgslLanguageFeatures: ['immediate_address_space'],
      immediateAddressSpace: true,
      deviceMaxImmediateSize: 64,
      renderPassSetImmediates: true,
      renderBundleSetImmediates: true,
    },
    adapterInfo: { isFallbackAdapter: false },
    runtimeSignature: {
      revision: '186dev',
      drawIndexNode: true,
      backendHasDrawIndexSetter: true,
      backendDrawUsesPipelineMetadata: true,
      builderReturnsImmediateSymbol: true,
      builderChecksDrawIndexAvailability: true,
    },
    scenarios: {
      directFragment: {
        output: outputRecord(expectedDirectOutput()),
        traces: traces('direct-fragment'),
      },
      indirectBundle: {
        commands: Array.from(createIndirectCommands()),
        initialOffsets: [...INDIRECT_OFFSETS],
        rerecordOffsets: [...RERECORD_INDIRECT_OFFSETS],
        rootVersion: 1,
        first: {
          output: outputRecord(expectedIndirectOutput()),
          executedBundleIds: ['bundle-first'],
          traces: traces('indirect-bundle-first'),
        },
        cached: {
          output: outputRecord(expectedIndirectOutput()),
          executedBundleIds: ['bundle-first'],
          traces: traces('indirect-bundle-cached'),
        },
        rerecorded: {
          output: outputRecord(expectedIndirectOutput(RERECORD_INDIRECT_OFFSETS)),
          executedBundleIds: ['bundle-rerecorded'],
          traces: traces('indirect-bundle-rerecord'),
        },
      },
      batched: {
        requestedOrder: [...BATCH_ORDER],
        observedIndirectTextureOrder: [...BATCH_ORDER],
        explicit: {
          output: outputRecord(expectedBatchedOutput()),
          traces: traces('batched-explicit'),
        },
        control: {
          output: outputRecord(expectedBatchedControlOutput()),
          traces: traces('batched-control'),
        },
      },
      arrayCamera: {
        cameraCount: 2,
        targetDepth: 2,
        layers: [
          outputRecord(expectedIndirectOutput()),
          outputRecord(expectedIndirectOutput()),
        ],
        traces: traces('array-camera'),
      },
    },
    instrumentation: {
      shaderModules: [
        { moduleId: 'shader-vertex-immediate', code: immediateVertex },
        { moduleId: 'shader-fragment-immediate', code: immediateFragment },
        { moduleId: 'shader-vertex-control', code: 'control vertex' },
        { moduleId: 'shader-fragment-control', code: 'control fragment' },
      ],
      pipelineLayouts: [
        { layoutId: 'layout-immediate', hasOwnImmediateSize: true, immediateSize: 4 },
        { layoutId: 'layout-control', hasOwnImmediateSize: true, immediateSize: 0 },
      ],
      renderPipelines: [
        {
          pipelineId: 'pipeline-immediate',
          layoutId: 'layout-immediate',
          vertexModuleId: 'shader-vertex-immediate',
          fragmentModuleId: 'shader-fragment-immediate',
        },
        {
          pipelineId: 'pipeline-control',
          layoutId: 'layout-control',
          vertexModuleId: 'shader-vertex-control',
          fragmentModuleId: 'shader-fragment-control',
        },
      ],
      renderPasses: passes,
      renderBundles: [firstBundle, rerecordedBundle, arrayBundle0, arrayBundle1],
      errorScopes: scopes,
      patchedDeviceMethods: [],
    },
    shaderCompilation: [
      { moduleId: 'shader-vertex-immediate', available: true, messages: [] },
      { moduleId: 'shader-fragment-immediate', available: true, messages: [] },
      { moduleId: 'shader-vertex-control', available: true, messages: [] },
      { moduleId: 'shader-fragment-control', available: true, messages: [] },
    ],
    errorObservation: {
      outerFilters: ['internal', 'out-of-memory', 'validation'],
      includesRendererInitialization: true,
      includesAllRendersAndReadbacks: true,
      depthBeforeOuterPops: 3,
      outerResults: [
        { filter: 'validation', error: null },
        { filter: 'out-of-memory', error: null },
        { filter: 'internal', error: null },
      ],
      finalDepth: 0,
      resourcesDisposedWithinScopes: true,
      indirectGpuAttributesDeleted: 2,
      intentionalDeviceLoss: { reason: 'destroyed', message: 'destroyed' },
    },
    threeErrors: [],
    threeConsole: [],
    threeConsoleErrors: [],
    threeDeviceLosses: [],
    gpuErrors: { uncaptured: [], unexpectedDeviceLosses: [] },
    cleanupFailures: [],
    failure: null,
  };
}

test('nonidentity pixel oracles distinguish physical ordinal and original identity', () => {
  assert.deepEqual(INDIRECT_OFFSETS, [40, 0, 60, 20]);
  assert.deepEqual(expectedIndirectOutput(), [
    2, 0, 0, 255,
    4, 0, 0, 255,
    1, 0, 0, 255,
    3, 0, 0, 255,
  ]);
  assert.deepEqual(expectedBatchedOutput(), [
    2, 1, 0, 255,
    4, 2, 0, 255,
    1, 3, 0, 255,
    3, 4, 0, 255,
  ]);
  assert.notDeepEqual(expectedIndirectOutput(RERECORD_INDIRECT_OFFSETS),
    expectedIndirectOutput());
});

test('indirect command records leave firstInstance zero and use distinct geometry', () => {
  const commands = createIndirectCommands();
  assert.equal(commands.length, 20);
  for (let draw = 0; draw < 4; draw += 1) {
    assert.deepEqual(
      Array.from(commands.subarray(draw * 5, draw * 5 + 5)),
      [3, 1, draw * 3, 0, 0],
    );
  }
});

test('independent page validator reconstructs all outputs and command traces', () => {
  assert.deepEqual(validateThreeDrawIndexPageResult(makePageResult()), {
    valid: true,
    reasons: [],
  });
});

test('page validator rejects raw tampering despite page-authored exact flags', async (t) => {
  const cases = [
    ['direct byte', (value) => { value.scenarios.directFragment.output.observed[0] = 9; }],
    ['batch order', (value) => { value.scenarios.batched.observedIndirectTextureOrder[0] = 0; }],
    ['setter value', (value) => {
      value.instrumentation.renderBundles[0].events[3].selectedValues[0] = 9;
    }],
    ['setter adjacency', (value) => {
      value.instrumentation.renderBundles[0].events.splice(2, 0, {
        method: 'setBindGroup', phase: 'indirect-bundle-first',
      });
    }],
    ['layout size', (value) => {
      value.instrumentation.pipelineLayouts[0].immediateSize = 0;
    }],
    ['fragment direct immediate', (value) => {
      value.instrumentation.shaderModules[1].code += '\nnodeDrawIndex';
    }],
    ['cached identity', (value) => {
      value.instrumentation.renderPasses[2].events[0].bundleIds[0] = 'other';
    }],
    ['ArrayCamera continuation', (value) => {
      const second = value.instrumentation.renderBundles[3];
      second.events.find((event) => event.method === 'setImmediates').selectedValues[0] = 4;
    }],
    ['control setter', (value) => {
      value.instrumentation.renderPasses[5].events.splice(1, 0, setterEvent('batched-control', 0));
    }],
    ['scope error', (value) => {
      value.instrumentation.errorScopes[5].resolvedError = { name: 'GPUValidationError' };
    }],
    ['extra wrong-method draw', (value) => {
      value.instrumentation.renderPasses[0].events.push({
        method: 'draw', arguments: [0, 0, 0, 0], phase: 'direct-fragment',
      });
    }],
    ['extra control indirect draw', (value) => {
      value.instrumentation.renderPasses[5].events.push({
        method: 'drawIndexedIndirect', indirectOffset: 0, phase: 'batched-control',
      });
    }],
  ];
  for (const [name, mutate] of cases) {
    await t.test(name, () => {
      const value = makePageResult();
      mutate(value);
      assert.equal(validateThreeDrawIndexPageResult(value).valid, false);
    });
  }
});

function makeUnsupportedResult(stage) {
  const value = makePageResult();
  value.status = 'unsupported';
  value.preflight = {
    secureContext: stage !== 'secure-context',
    currentStage: stage,
    capabilityDecision: 'unsupported',
    unsupportedStage: stage,
    unsupportedReason: `unsupported at ${stage}`,
    unsupportedDetail: null,
    adapterRequestAttempts: 0,
    adapterAcquisitions: 0,
    requestAdapterArgumentsCount: null,
    requestAdapterDescriptorSupplied: null,
    deviceRequestAttempts: 0,
    deviceAcquisitions: 0,
    requestDeviceArgumentsCount: null,
    requestDeviceDescriptorSupplied: null,
    shaderModulesAtCapabilityDecision: 0,
    instrumentationInstalledBeforeRenderer: false,
    rendererConstructed: false,
  };
  value.capabilities = {
    wgslLanguageFeatures: [],
    immediateAddressSpace: false,
    adapterFeatures: [],
    adapterMaxImmediateSize: null,
    deviceFeatures: [],
    deviceMaxImmediateSize: null,
  };
  if (stage === 'adapter-request' || stage === 'device-limit'
      || stage === 'encoder-surfaces') {
    value.capabilities.wgslLanguageFeatures = ['immediate_address_space'];
    value.capabilities.immediateAddressSpace = true;
    value.preflight.adapterRequestAttempts = 1;
    value.preflight.requestAdapterArgumentsCount = 0;
    value.preflight.requestAdapterDescriptorSupplied = false;
  }
  if (stage === 'device-limit' || stage === 'encoder-surfaces') {
    value.preflight.adapterAcquisitions = 1;
    value.preflight.deviceRequestAttempts = 1;
    value.preflight.deviceAcquisitions = 1;
    value.preflight.requestDeviceArgumentsCount = 0;
    value.preflight.requestDeviceDescriptorSupplied = false;
    value.adapterInfo = { isFallbackAdapter: false };
    value.capabilities.deviceMaxImmediateSize = stage === 'device-limit' ? 3 : 64;
    value.capabilities.renderPassSetImmediates = true;
    value.capabilities.renderBundleSetImmediates = stage !== 'encoder-surfaces';
    value.preflight.instrumentationInstalledBeforeRenderer = true;
    value.instrumentation = {
      shaderModules: [],
      pipelineLayouts: [],
      renderPipelines: [],
      renderPasses: [],
      renderBundles: [],
      errorScopes: [],
      patchedDeviceMethods: ['createShaderModule'],
    };
  } else {
    value.adapterInfo = null;
    value.instrumentation = null;
  }
  value.runtimeSignature = null;
  value.scenarios = null;
  value.shaderCompilation = [];
  value.errorObservation = {
    outerFilters: [],
    includesRendererInitialization: false,
    includesAllRendersAndReadbacks: false,
    depthBeforeOuterPops: null,
    outerResults: [],
    finalDepth: null,
    resourcesDisposedWithinScopes: false,
    indirectGpuAttributesDeleted: 0,
    intentionalDeviceLoss: ['device-limit', 'encoder-surfaces'].includes(stage)
      ? { reason: 'destroyed', message: 'destroyed' }
      : null,
  };
  return value;
}

test('unsupported page results prove the exact capability-gate prefix', async (t) => {
  for (const stage of [
    'secure-context',
    'navigator-gpu',
    'wgsl-language-feature',
    'adapter-request',
    'device-limit',
    'encoder-surfaces',
  ]) {
    await t.test(stage, () => {
      assert.deepEqual(validateThreeDrawIndexPageResult(makeUnsupportedResult(stage)), {
        valid: true,
        reasons: [],
      });
    });
  }
  const shaderAfterUnsupported = makeUnsupportedResult('device-limit');
  shaderAfterUnsupported.shaderCompilation.push({ messages: [] });
  assert.equal(validateThreeDrawIndexPageResult(shaderAfterUnsupported).valid, false);
  const lossNotAwaited = makeUnsupportedResult('encoder-surfaces');
  lossNotAwaited.errorObservation.intentionalDeviceLoss = null;
  assert.equal(validateThreeDrawIndexPageResult(lossNotAwaited).valid, false);
  const crossedBoundary = makeUnsupportedResult('wgsl-language-feature');
  crossedBoundary.preflight.rendererConstructed = true;
  assert.equal(validateThreeDrawIndexPageResult(crossedBoundary).valid, false);
});

test('CLI accepts only one explicit browser path and no output override', () => {
  const browser = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  assert.deepEqual(parseThreeDrawIndexArguments(['--browser', browser]), {
    browserPath: path.resolve(browser),
  });
  assert.throws(() => parseThreeDrawIndexArguments([]), /--browser is required/u);
  assert.throws(
    () => parseThreeDrawIndexArguments(['--browser', browser, '--output', 'other.json']),
    /Unknown argument/u,
  );
});

test('frozen protocol is internally valid and contains no custom browser arguments', async () => {
  const protocol = JSON.parse(await readFile(
    'protocols/three-draw-index-canary-v1.json',
    'utf8',
  ));
  assert.deepEqual(validateThreeDrawIndexProtocol(protocol), { valid: true, reasons: [] });
  assert.deepEqual(protocol.browser.playwrightArguments, []);
  assert.equal(protocol.timingPolicy.captureTimings, false);
  assert.equal(protocol.output.overwrite, false);
  for (const mutate of [
    (value) => { value.command = 'different'; },
    (value) => { value.claimBoundary = 'performance'; },
    (value) => { value.browser.forbiddenArgumentPrefixes = []; },
    (value) => { value.page.timeoutMilliseconds = 1; },
    (value) => { value.capabilityOrder = []; },
    (value) => { value.assertions = []; },
    (value) => { value.output.encoding = 'different'; },
  ]) {
    const changed = structuredClone(protocol);
    mutate(changed);
    assert.equal(validateThreeDrawIndexProtocol(changed).valid, false);
  }
});

test('default-exposure argument check uses exact switch prefixes', async () => {
  const protocol = JSON.parse(await readFile(
    'protocols/three-draw-index-canary-v1.json',
    'utf8',
  ));
  assert.deepEqual(findForbiddenThreeDrawIndexBrowserArguments([
    '--disable-features=UseAngle',
    '--ordinary',
  ], protocol), []);
  assert.deepEqual(findForbiddenThreeDrawIndexBrowserArguments([
    '--ignore-gpu-blocklist',
    '--force-webgpu-compat=1',
  ], protocol), [
    '--ignore-gpu-blocklist',
    '--force-webgpu-compat=1',
  ]);
});

test('a self-asserted sparse failed report is rejected', () => {
  const fake = {
    schemaVersion: 1,
    kind: 'three-webgpu-draw-index-default-exposure-report',
    status: 'failed',
    executionMode: 'timing-free-one-shot-correctness',
    timingCaptured: false,
    benchmarkClaim: false,
    failure: { name: 'Error', message: 'synthetic' },
  };
  assert.equal(validateThreeDrawIndexReport(fake).valid, false);
});

test('canary source freezes timing-free, default-device, adversarial paths', async () => {
  const [pageSource, runnerSource] = await Promise.all([
    readFile('src/three-draw-index-canary.js', 'utf8'),
    readFile('scripts/probe-three-draw-index.mjs', 'utf8'),
  ]);
  for (const source of [pageSource, runnerSource]) {
    assert.doesNotMatch(source, /performance\.now\s*\(/u);
    assert.doesNotMatch(source, /trackTimestamp\s*:\s*true/u);
  }
  assert.match(pageSource, /adapter\.requestDevice\(\)/u);
  assert.match(pageSource, /navigator\.gpu\.requestAdapter\(\)/u);
  assert.doesNotMatch(pageSource, /requiredLimits/u);
  assert.match(pageSource, /batched-explicit/u);
  assert.match(pageSource, /batched-control/u);
  assert.match(pageSource, /array-camera/u);
  assert.match(runnerSource, /args:\s*\[\]/u);
  assert.match(runnerSource, /Browser\.getBrowserCommandLine/u);
  assert.match(runnerSource, /flag:\s*'wx'/u);
  assert.match(runnerSource, /chmod\(paths\.reportPath,\s*0o444\)/u);
});

test('exact overlay aliases bundle the browser canary without executing it', {
  timeout: 120_000,
}, async () => {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), 'draw-index-canary-build-'));
  const overlayRoot = path.join(temporaryRoot, 'overlay');
  try {
    const built = await buildThreeDrawIndexOverlay({ outputRoot: overlayRoot });
    assert.deepEqual(validateThreeDrawIndexOverlayManifest(built.manifest), {
      valid: true,
      reasons: [],
    });
    const alteredManifest = structuredClone(built.manifest);
    alteredManifest.output.totalBytes += 1;
    assert.equal(validateThreeDrawIndexOverlayManifest(alteredManifest).valid, false);
    const protocol = JSON.parse(await readFile(
      'protocols/three-draw-index-canary-v1.json',
      'utf8',
    ));
    const failedReport = await makeServerStartFailureReport(protocol, built.manifest);
    assert.deepEqual(validateThreeDrawIndexReport(failedReport), {
      valid: true,
      reasons: [],
    });
    const lifecycleForgery = structuredClone(failedReport);
    lifecycleForgery.observedLifecycle.serverStartAttempts = 0;
    assert.equal(validateThreeDrawIndexReport(lifecycleForgery).valid, false);
    const missingPostflight = structuredClone(failedReport);
    missingPostflight.sourceIdentity.filesEnd = null;
    missingPostflight.sourceIdentity.filesMatch = false;
    missingPostflight.observedLifecycle.postflightChecksSucceeded = 3;
    assert.equal(validateThreeDrawIndexReport(missingPostflight).valid, false);
    const { build } = await import('vite');
    const result = await build({
      root: process.cwd(),
      configFile: false,
      logLevel: 'silent',
      resolve: {
        alias: [
          { find: /^three$/u, replacement: path.join(overlayRoot, 'src', 'Three.js') },
          {
            find: /^three\/webgpu$/u,
            replacement: path.join(overlayRoot, 'src', 'Three.WebGPU.js'),
          },
          {
            find: /^three\/tsl$/u,
            replacement: path.join(overlayRoot, 'src', 'Three.TSL.js'),
          },
        ],
      },
      build: {
        write: false,
        minify: false,
        target: 'esnext',
        rollupOptions: { input: path.resolve('three-draw-index-canary.html') },
      },
    });
    const outputs = Array.isArray(result)
      ? result.flatMap((entry) => entry.output)
      : result.output;
    const source = outputs
      .filter((entry) => entry.type === 'chunk')
      .map((entry) => entry.code)
      .join('\n');
    assert.match(source, /requires immediate_address_space;/u);
    assert.match(source, /var<immediate> \$\{ drawIndexName \} : u32;/u);
    assert.match(source, /setImmediates/u);
    assert.match(source, /BatchedMesh/u);
    assert.match(source, /ArrayCamera/u);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});
