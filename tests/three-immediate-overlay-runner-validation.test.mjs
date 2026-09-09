import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  exactAliasEvidence,
  injectThreeImmediateOverlayCanaryTarget,
  pageConfigurationForThreeImmediateOverlayTarget,
  validateBoundSourceIdentity,
  validatePageResult,
  validateThreeImmediateOverlayManifest,
} from '../scripts/probe-three-immediate-overlay.mjs';
import {
  EXPECTED_TARGET_HASHES,
  EXPECTED_THREE_VERSION,
} from '../scripts/build-three-immediate-overlay.mjs';
import {
  EXPECTED_THREE_DEV_COMMIT,
  EXPECTED_THREE_DEV_PATCHED_HASHES,
  EXPECTED_THREE_DEV_REVISION,
  EXPECTED_THREE_DEV_TARGET_HASHES,
  EXPECTED_THREE_DEV_TREE,
  EXPECTED_THREE_DEV_VERSION,
} from '../scripts/build-three-immediate-dev-overlay.mjs';

const canonicalBases = [0, 2, 4, 6];
const mutatedBases = [1, 3, 5, 0];
const visibleIds = [5, 0, 7, 2, 1, 6, 3, 4];
const offsets = [0, 20, 40, 60];
const commandWords = [
  3, 2, 0, 0, 0,
  3, 2, 3, 0, 0,
  3, 2, 6, 0, 0,
  3, 2, 9, 0, 0,
];
function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function outputForBases(bases) {
  const output = new Uint8Array(8 * 4);
  for (let draw = 0; draw < bases.length; draw += 1) {
    const base = bases[draw];
    for (let instance = 0; instance < 2; instance += 1) {
      const address = base + instance;
      if (address >= visibleIds.length) continue;
      const pixel = draw * 2 + instance;
      output[pixel * 4] = visibleIds[address] + 1;
      output[pixel * 4 + 3] = 255;
    }
  }
  return [...output];
}

function makeEncoderEvents() {
  const events = [
    { sequence: 3, method: 'setPipeline', pipelineId: 'render-pipeline-5' },
    { sequence: 4, method: 'setBindGroup' },
    { sequence: 5, method: 'setIndexBuffer' },
    { sequence: 6, method: 'setVertexBuffer' },
  ];
  for (let index = 0; index < 4; index += 1) {
    events.push({
      sequence: 7 + index * 2,
      method: 'setImmediates',
      rangeOffset: 0,
      sourceType: 'Uint32Array',
      sourceElementCount: 4,
      dataOffset: index,
      dataSize: 1,
      selectedValues: [canonicalBases[index]],
    });
    events.push({
      sequence: 8 + index * 2,
      method: 'drawIndexedIndirect',
      bufferId: 'gpu-buffer-5',
      indirectOffset: offsets[index],
    });
  }
  events.push({ sequence: 15, method: 'finish' });
  return events;
}

function makePageResult(targetKey = 'r185') {
  const targetConfiguration = pageConfigurationForThreeImmediateOverlayTarget(targetKey);
  const vertexShader = `requires immediate_address_space;
var<immediate> threeImmediateDrawBase : u32;
fn address(instanceIndex : u32) -> u32 {
  return threeImmediateDrawBase + instanceIndex;
}`;
  const fragmentShader = '@fragment fn fragmentMain() -> @location(0) vec4f { return vec4f(); }';
  const immediateLayout = {
    sequence: 1,
    label: '',
    hasOwnImmediateSize: true,
    immediateSize: 4,
    bindGroupLayoutCount: 1,
    layoutId: 'pipeline-layout-4',
  };
  const controlLayout = {
    sequence: 16,
    label: '',
    hasOwnImmediateSize: true,
    immediateSize: 0,
    bindGroupLayoutCount: 1,
    layoutId: 'pipeline-layout-8',
  };
  const controlVertexShader = 'fn threeOverlayInstrumentedControlAddress() -> u32 { return 0u; }';
  const controlFragmentShader = '@fragment fn controlFragment() -> @location(0) vec4f { return vec4f(); }';
  const shaderModules = [
    { moduleId: 'shader-module-1', label: '', code: vertexShader },
    { moduleId: 'shader-module-2', label: '', code: fragmentShader },
    { moduleId: 'shader-module-6', label: '', code: controlVertexShader },
    { moduleId: 'shader-module-7', label: '', code: controlFragmentShader },
  ];
  const events = makeEncoderEvents();
  const adjacency = offsets.map((drawOffset, index) => ({
    drawSequence: 8 + index * 2,
    drawOffset,
    previousMethod: 'setImmediates',
    previousSequence: 7 + index * 2,
    adjacent: true,
    immediate: {
      rangeOffset: 0,
      sourceType: 'Uint32Array',
      sourceElementCount: 4,
      dataOffset: index,
      dataSize: 1,
      selectedValues: [canonicalBases[index]],
    },
  }));
  const canonical = outputForBases(canonicalBases);
  const mutated = outputForBases(mutatedBases);
  const bundleSummary = {
    renderObjectCount: 1,
    meshIdentityExact: true,
    bundleIdentityExact: true,
    bundleVersion: 0,
    rootVersion: 0,
    bundleVersionExact: true,
    pipelineImmediateSize: 4,
    immediateBasesIdentityExact: true,
  };
  return {
    schemaVersion: 1,
    kind: 'three-immediate-overlay-development-canary',
    status: 'development-checks-complete',
    executionMode: 'timing-free-development-correctness-canary',
    target: {
      schemaVersion: targetConfiguration.schemaVersion,
      key: targetConfiguration.key,
      sourceFamily: targetConfiguration.sourceFamily,
      expectedRevision: targetConfiguration.expectedRevision,
    },
    scope: targetConfiguration.scope,
    analysisEligible: false,
    timingCaptured: false,
    efficacyEvaluated: false,
    fullPhaseZeroPass: false,
    fullPhaseZeroStatus: 'not-evaluated',
    coverage: {
      implemented: [...targetConfiguration.implemented],
      deferred: [...targetConfiguration.deferred],
    },
    capabilities: {
      wgslLanguageFeatures: ['immediate_address_space'],
      immediateAddressSpace: true,
      nonFallbackAdapter: true,
      adapterFeatures: [],
      adapterMaxImmediateSize: 64,
      deviceMaxImmediateSize: 64,
      renderPassSetImmediates: true,
      renderBundleSetImmediates: true,
    },
    adapterInfo: {
      vendor: 'test-vendor',
      architecture: 'test-architecture',
      device: 'test-device',
      description: 'test adapter',
      isFallbackAdapter: false,
    },
    errorObservationScope: {
      start: 'after-device-creation-before-renderer-construction-and-initialization',
      includesRendererInitialization: true,
      includesAllInstrumentedRendererRendersAndReadbacks: true,
      excludesAdapterAndDeviceRequest: true,
    },
    runtimeOverlaySignature: {
      revision: targetConfiguration.expectedRevision,
      setIndirectHasImmediateBasesParameter: true,
      setIndirectStoresImmediateBases: true,
      backendHasSetImmediates: true,
      backendHasIndexedIndirectAdjacencyMarker: true,
    },
    evidence: {
      topology: {
        exact: true,
        bundleGroupCount: 1,
        meshCount: 1,
        indexed: true,
        indirectOffsetCount: 4,
        indirectImmediateBaseCount: 4,
        bucketBaseAttributePresent: false,
      },
      shader: {
        exact: true,
        requirementCount: 1,
        declarationCount: 1,
        addressUseCount: 1,
        bucketBaseCount: 0,
        addressExpression: 'threeImmediateDrawBase + instanceIndex',
        vertexShader,
        fragmentShader,
        vertexSha256: sha256(Buffer.from(vertexShader)),
        fragmentSha256: sha256(Buffer.from(fragmentShader)),
      },
      pipeline: {
        exact: true,
        exactImmediate: true,
        exactControl: true,
        immediatePipelineCount: 1,
        controlPipelineCount: 1,
        classified: [
          {
            sequence: 2,
            method: 'createRenderPipeline',
            pipelineId: 'render-pipeline-5',
            layoutId: immediateLayout.layoutId,
            vertexModuleId: 'shader-module-1',
            fragmentModuleId: 'shader-module-2',
            vertexUsesImmediate: true,
            vertexIsControl: false,
            layout: { ...immediateLayout },
          },
          {
            sequence: 17,
            method: 'createRenderPipeline',
            pipelineId: 'render-pipeline-9',
            layoutId: controlLayout.layoutId,
            vertexModuleId: 'shader-module-6',
            fragmentModuleId: 'shader-module-7',
            vertexUsesImmediate: false,
            vertexIsControl: true,
            layout: { ...controlLayout },
          },
        ],
        actualPipelineId: 'render-pipeline-5',
        immediatePipelineId: 'render-pipeline-5',
        immediateVertexModuleId: 'shader-module-1',
        immediateFragmentModuleId: 'shader-module-2',
        bundleSetPipelineIds: ['render-pipeline-5'],
        moduleSourceBindingExact: true,
        pipelineToBundleBindingExact: true,
        shaderModuleSnapshots: shaderModules,
        layoutSnapshots: [immediateLayout, controlLayout],
      },
      commands: {
        exact: true,
        drawCount: 4,
        strideBytes: 20,
        indirectOffsets: [...offsets],
        cpuCommands: [...commandWords],
        gpuCommands: [...commandWords],
        cpuWordsFour: [0, 0, 0, 0],
        gpuWordsFour: [0, 0, 0, 0],
        readbackAttributeIdentityExact: true,
        drawGpuBufferId: 'gpu-buffer-5',
        encoderDrawGpuBufferIdentityExact: true,
        encoderDrawGpuBufferIds: Array(4).fill('gpu-buffer-5'),
      },
      encoder: {
        exact: true,
        exactCalls: true,
        exactDraws: true,
        drawGpuBufferId: 'gpu-buffer-5',
        drawBufferIds: Array(4).fill('gpu-buffer-5'),
        drawBufferIdentityExact: true,
        patchedDeviceMethods: [
          'createShaderModule',
          'createPipelineLayout',
          'createRenderPipeline',
          'createRenderPipelineAsync',
          'createCommandEncoder',
          'createRenderBundleEncoder',
        ],
        adjacency,
        trace: {
          descriptor: {
            colorFormats: ['rgba8unorm'],
            depthStencilFormat: null,
            sampleCount: 1,
          },
          callableSetImmediates: true,
          bundleId: 'render-bundle-9',
          events,
          nativeFinishReturned: true,
          sourceMutation: {
            phase: 'after-native-finish-returned-before-bundle-execution',
            before: [...canonicalBases],
            after: [...mutatedBases],
          },
        },
      },
      output: {
        exact: true,
        encoding: 'rgba8unorm-red-object-id-plus-one-at-draw-ordinal-instance-pixel',
        pixelMapping: 'pixel = drawOrdinal * instancesPerDraw + localInstance',
        challengeDifferingPixelCount: 8,
        expectedCanonical: canonical,
        expectedMutated: mutated,
        firstObserved: [...canonical],
        secondObserved: [...canonical],
        canonicalSha256: sha256(Buffer.from(canonical)),
        mutatedSha256: sha256(Buffer.from(mutated)),
        firstSha256: sha256(Buffer.from(canonical)),
        secondSha256: sha256(Buffer.from(canonical)),
        firstMismatchCount: 0,
        secondMismatchCount: 0,
      },
      bundle: {
        exact: true,
        bundleGroupCount: 1,
        meshCount: 1,
        rootStatic: true,
        first: { ...bundleSummary },
        second: { ...bundleSummary },
        renderBundleIdentityStable: true,
        bundleGpuIdentityStable: true,
        cachedBundleGpuId: 'render-bundle-9',
        finishBundleId: 'render-bundle-9',
        renderPassExecutionExact: true,
        firstRenderPassExecutions: [{
          sequence: 16,
          capturePhase: 'instrumented-immediate-first',
          commandEncoderId: 'command-encoder-10',
          renderPassId: 'render-pass-11',
          bundleCount: 1,
          bundleIds: ['render-bundle-9'],
        }],
        secondRenderPassExecutions: [{
          sequence: 17,
          capturePhase: 'instrumented-immediate-second',
          commandEncoderId: 'command-encoder-12',
          renderPassId: 'render-pass-13',
          bundleCount: 1,
          bundleIds: ['render-bundle-9'],
        }],
        allRenderPassExecutions: [
          {
            sequence: 16,
            capturePhase: 'instrumented-immediate-first',
            commandEncoderId: 'command-encoder-10',
            renderPassId: 'render-pass-11',
            bundleCount: 1,
            bundleIds: ['render-bundle-9'],
          },
          {
            sequence: 17,
            capturePhase: 'instrumented-immediate-second',
            commandEncoderId: 'command-encoder-12',
            renderPassId: 'render-pass-13',
            bundleCount: 1,
            bundleIds: ['render-bundle-9'],
          },
          {
            sequence: 20,
            capturePhase: 'instrumented-non-immediate-control',
            commandEncoderId: 'command-encoder-18',
            renderPassId: 'render-pass-19',
            bundleCount: 1,
            bundleIds: ['render-bundle-17'],
          },
        ],
        postImmediateControlEncoderCount: 2,
        encoderCountAfterFirstRender: 1,
        encoderCountAfterSecondRender: 1,
        sourceBasesAfterFinish: [...mutatedBases],
      },
      uninstrumentedSnapshot: {
        exact: true,
        deviceInstrumentationInstalledByCanary: false,
        disposableRenderer: true,
        deviceDistinctFromInstrumentedRenderer: true,
        mutationPhase: 'after-first-render-executed-before-second-cached-render',
        sourceBeforeMutation: [...canonicalBases],
        sourceAfterMutation: [...mutatedBases],
        expectedCanonical: [...canonical],
        expectedMutated: [...mutated],
        firstObserved: [...canonical],
        secondObserved: [...canonical],
        canonicalSha256: sha256(Buffer.from(canonical)),
        firstSha256: sha256(Buffer.from(canonical)),
        secondSha256: sha256(Buffer.from(canonical)),
        renderBundleIdentityStable: true,
        bundleGpuIdentityStable: true,
        firstBundle: { ...bundleSummary },
        secondBundle: { ...bundleSummary },
        adapterInfo: { isFallbackAdapter: false },
        adapterFeatures: [],
        adapterMaxImmediateSize: 64,
        deviceMaxImmediateSize: 64,
        errorObservationScope: {
          start: 'after-device-creation-before-renderer-construction-and-initialization',
          includesRendererInitialization: true,
          includesBothCanaryRendersAndReadbacks: true,
          excludesAdapterAndDeviceRequest: true,
        },
        gpuErrors: {
          uncaptured: [],
          unexpectedDeviceLosses: [],
        },
      },
    },
    gpuErrors: {
      uncaptured: [],
      unexpectedDeviceLosses: [],
    },
    failure: null,
  };
}

test('runner independently reconstructs the complete minimal page evidence', () => {
  assert.deepEqual(validatePageResult(makePageResult()), {
    valid: true,
    reasons: [],
  });
});

test('the same reconstruction validates the distinct pinned upstream-dev target', () => {
  assert.deepEqual(
    validatePageResult(
      makePageResult(THREE_IMMEDIATE_OVERLAY_TARGET_DEV),
      THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
    ),
    { valid: true, reasons: [] },
  );

  const devAsR185 = makePageResult(THREE_IMMEDIATE_OVERLAY_TARGET_DEV);
  assert.equal(validatePageResult(devAsR185, 'r185').valid, false);
  const r185AsDev = makePageResult('r185');
  assert.equal(
    validatePageResult(r185AsDev, THREE_IMMEDIATE_OVERLAY_TARGET_DEV).valid,
    false,
  );
});

test('runner rejects raw tampering even when every page-authored exact flag stays true', async (t) => {
  const cases = [
    ['output byte', (result) => { result.evidence.output.firstObserved[0] ^= 1; }],
    ['GPU word four', (result) => { result.evidence.commands.gpuCommands[4] = 1; }],
    ['GPU draw ordinal', (result) => { result.evidence.commands.gpuCommands[7] = 0; }],
    ['draw/readback buffer identity', (result) => {
      result.evidence.commands.drawGpuBufferId = 'different-gpu-buffer';
    }],
    ['encoder value', (result) => {
      result.evidence.encoder.trace.events[6].selectedValues[0] = 9;
    }],
    ['draw adjacency', (result) => {
      result.evidence.encoder.trace.events.splice(7, 0, {
        sequence: 10,
        method: 'setViewport',
      });
    }],
    ['control immediate size', (result) => {
      result.evidence.pipeline.layoutSnapshots[1].immediateSize = 4;
    }],
    ['shader module to pipeline binding', (result) => {
      result.evidence.pipeline.shaderModuleSnapshots[0].code += '\n// tampered';
    }],
    ['pipeline to bundle binding', (result) => {
      result.evidence.encoder.trace.events[0].pipelineId = 'different-pipeline';
    }],
    ['render pass cached bundle identity', (result) => {
      result.evidence.bundle.secondRenderPassExecutions[0].bundleIds[0]
        = 'different-bundle';
    }],
    ['uninstrumented snapshot output', (result) => {
      result.evidence.uninstrumentedSnapshot.secondObserved[0] ^= 1;
    }],
    ['error observation scope', (result) => {
      result.errorObservationScope.includesRendererInitialization = false;
    }],
    ['coverage inventory', (result) => {
      result.coverage.implemented.push('invented-gate');
    }],
    ['shader text', (result) => {
      result.evidence.shader.vertexShader = result.evidence.shader.vertexShader
        .replace(' + instanceIndex', '');
    }],
  ];
  for (const [name, mutate] of cases) {
    await t.test(name, () => {
      const result = makePageResult();
      mutate(result);
      assert.equal(validatePageResult(result).valid, false);
    });
  }
});

function makeSourceIdentity(targetKey = 'r185') {
  const targetConfiguration = pageConfigurationForThreeImmediateOverlayTarget(targetKey);
  const files = [
    ['runner', 'scripts/probe-three-immediate-overlay.mjs', 'a'],
    ['page-module', 'src/three-immediate-overlay-canary.js', 'b'],
    ['entry-html', 'three-immediate-canary.html', 'c'],
  ].map(([role, path, contents]) => ({
    role,
    path,
    byteLength: 1,
    sha256: sha256(Buffer.from(contents)),
  }));
  const pageRecord = files.find((record) => record.role === 'page-module');
  pageRecord.targetInjectedSha256 = sha256(Buffer.from(`injected-${targetKey}`));
  const pageHash = pageRecord.sha256;
  const target = {
    schemaVersion: targetConfiguration.schemaVersion,
    key: targetConfiguration.key,
    sourceFamily: targetConfiguration.sourceFamily,
    expectedRevision: targetConfiguration.expectedRevision,
  };
  return {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    target,
    filesStart: structuredClone(files),
    filesEnd: structuredClone(files),
    filesMatch: true,
    pageTransformInput: {
      specifier: 'canary-page',
      transformInputSha256: pageHash,
      transformOutputSha256: pageRecord.targetInjectedSha256,
      transformCount: 1,
      injectionCount: 1,
      target: structuredClone(target),
      targetConfigurationSha256: sha256(Buffer.from(JSON.stringify(targetConfiguration))),
    },
    pageTransformMatches: true,
    exact: true,
  };
}

test('bound source validation rejects changed or non-served page bytes', () => {
  assert.deepEqual(validateBoundSourceIdentity(makeSourceIdentity()), {
    valid: true,
    reasons: [],
  });

  const changed = makeSourceIdentity();
  changed.filesEnd[1].sha256 = sha256(Buffer.from('changed'));
  assert.equal(validateBoundSourceIdentity(changed).valid, false);

  const unboundTransform = makeSourceIdentity();
  unboundTransform.pageTransformInput.transformInputSha256 = sha256(Buffer.from('other'));
  assert.equal(validateBoundSourceIdentity(unboundTransform).valid, false);

  const dev = makeSourceIdentity(THREE_IMMEDIATE_OVERLAY_TARGET_DEV);
  assert.equal(validateBoundSourceIdentity(dev, THREE_IMMEDIATE_OVERLAY_TARGET_DEV).valid, true);
  assert.equal(validateBoundSourceIdentity(dev, 'r185').valid, false);

  const wrongInjectedOutput = makeSourceIdentity();
  wrongInjectedOutput.pageTransformInput.transformOutputSha256 = sha256(Buffer.from('other'));
  assert.equal(validateBoundSourceIdentity(wrongInjectedOutput).valid, false);
});

test('served-source validation binds transitive Three runtime hashes and exact URLs', () => {
  const aliases = {
    'three/webgpu': 'C:/overlay/build/three.webgpu.js',
    'three/tsl': 'C:/overlay/build/three.tsl.js',
  };
  const expectedHashes = {
    'three/webgpu': sha256(Buffer.from('webgpu')),
    'three/tsl': sha256(Buffer.from('tsl')),
    'three/core-runtime': sha256(Buffer.from('core')),
  };
  const observed = new Map(Object.entries(expectedHashes).map(([specifier, hash]) => [
    specifier,
    {
      specifier,
      transformInputSha256: hash,
      transformCount: 1,
    },
  ]));
  const resources = [
    'http://127.0.0.1:5173/.generated/three-immediate-overlay/build/three.core.js',
    'http://127.0.0.1:5173/.generated/three-immediate-overlay/build/three.tsl.js',
    'http://127.0.0.1:5173/.generated/three-immediate-overlay/build/three.webgpu.js',
    'http://127.0.0.1:5173/src/three-immediate-overlay-canary.js',
  ];
  assert.equal(exactAliasEvidence(
    aliases,
    expectedHashes,
    observed,
    resources,
  ).exact, true);

  const wrongCore = new Map(observed);
  wrongCore.set('three/core-runtime', {
    specifier: 'three/core-runtime',
    transformInputSha256: sha256(Buffer.from('wrong')),
    transformCount: 1,
  });
  assert.equal(exactAliasEvidence(
    aliases,
    expectedHashes,
    wrongCore,
    resources,
  ).exact, false);

  assert.equal(exactAliasEvidence(
    aliases,
    expectedHashes,
    observed,
    [...resources, 'http://127.0.0.1:5173/other-three-runtime.js'],
  ).exact, false);

  const devResources = resources.map((url) => url.replace(
    '/.generated/three-immediate-overlay/',
    '/.generated/three-immediate-dev-overlay/',
  ));
  assert.equal(exactAliasEvidence(
    aliases,
    expectedHashes,
    observed,
    devResources,
    THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  ).exact, true);
  assert.equal(exactAliasEvidence(
    aliases,
    expectedHashes,
    observed,
    resources,
    THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  ).exact, false);
});

function commonManifestContract() {
  return {
    symbol: 'threeImmediateDrawBase',
    wgslRequirement: 'immediate_address_space',
    immediateSizeBytes: 4,
    geometryProperty: 'indirectImmediateBases',
    setter: 'setIndirect(indirect, indirectOffset, indirectImmediateBases)',
  };
}

function makeR185Manifest() {
  return {
    schemaVersion: 1,
    overlay: 'three-immediate-address-attribution',
    source: {
      package: 'three',
      version: EXPECTED_THREE_VERSION,
      targetSha256: { ...EXPECTED_TARGET_HASHES },
    },
    contract: commonManifestContract(),
    patchedSha256: Object.fromEntries(
      Object.keys(EXPECTED_TARGET_HASHES)
        .filter((relativePath) => relativePath !== 'package.json')
        .map((relativePath, index) => [
          relativePath,
          index.toString(16).padStart(64, '0'),
        ]),
    ),
  };
}

function makeDevManifest() {
  return {
    schemaVersion: 1,
    overlay: 'three-immediate-address-attribution-dev',
    source: {
      package: 'three',
      version: EXPECTED_THREE_DEV_VERSION,
      runtimeRevision: EXPECTED_THREE_DEV_REVISION,
      commit: EXPECTED_THREE_DEV_COMMIT,
      tree: EXPECTED_THREE_DEV_TREE,
      cleanTrackedTree: true,
      cleanFullWorktree: true,
      statusPorcelain: '',
      targetCanonicalSha256: { ...EXPECTED_THREE_DEV_TARGET_HASHES },
    },
    transform: {
      patchedTargetSha256: { ...EXPECTED_THREE_DEV_PATCHED_HASHES },
    },
    contract: {
      ...commonManifestContract(),
      servedEntrypoints: {
        'three/webgpu': 'build/three.webgpu.js',
        'three/tsl': 'build/three.tsl.js',
      },
      unpatchedUnexportedBuilds: ['build/three.webgpu.nodes.js'],
    },
  };
}

test('target-specific manifest validation pins package and upstream-dev source identities', () => {
  assert.deepEqual(validateThreeImmediateOverlayManifest(makeR185Manifest()), {
    valid: true,
    reasons: [],
  });
  assert.deepEqual(validateThreeImmediateOverlayManifest(
    makeDevManifest(),
    THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  ), {
    valid: true,
    reasons: [],
  });

  const wrongCommit = makeDevManifest();
  wrongCommit.source.commit = '0'.repeat(40);
  assert.equal(validateThreeImmediateOverlayManifest(
    wrongCommit,
    THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  ).valid, false);

  assert.equal(validateThreeImmediateOverlayManifest(
    makeR185Manifest(),
    THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  ).valid, false);
});

test('runner injection binds one shared page to exactly one source family and revision', async () => {
  const pageSource = await readFile('src/three-immediate-overlay-canary.js', 'utf8');
  const r185 = injectThreeImmediateOverlayCanaryTarget(pageSource, 'r185');
  const dev = injectThreeImmediateOverlayCanaryTarget(
    pageSource,
    THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  );
  assert.notEqual(r185, dev);
  assert.match(r185, /installed-three-r185\.1-package/);
  assert.match(r185, /"expectedRevision":"185"/);
  assert.match(dev, /pinned-upstream-three-dev-commit/);
  assert.match(dev, /"expectedRevision":"186dev"/);
  assert.doesNotMatch(r185, /THREE_IMMEDIATE_CANARY_TARGET_CONFIGURATION_V1/);
  assert.doesNotMatch(dev, /THREE_IMMEDIATE_CANARY_TARGET_CONFIGURATION_V1/);
  assert.throws(() => injectThreeImmediateOverlayCanaryTarget(r185, 'r185'));
});
