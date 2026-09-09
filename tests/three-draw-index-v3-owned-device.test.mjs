import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  createInitialCheckpoint,
  createPreCollectionCheckpoint,
  findForbiddenArguments,
  prepareExampleSource,
  summarizeEvidence,
  validateEvidence,
  validateLateState,
} from '../scripts/probe-three-draw-index-v3-owned-device.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const protocolPath = path.join(
  repoRoot,
  'protocols',
  'three-draw-index-v3-owned-device-v1.json',
);
const manifestPath = path.join(
  repoRoot,
  'patches',
  'three-webgpu-draw-index-immediates-v2.json',
);

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function makeEvidence() {
  const events = [];
  for (let index = 0; index < 8; index += 1) {
    const immediate = {
      method: 'setImmediates',
      immediateSize: 4,
      rangeOffset: 0,
      dataOffset: 0,
      dataSize: 1,
      sourceType: 'Uint32Array',
      values: [index],
    };
    events.push(immediate, {
      arguments: [36, 1, 0, 0, index],
      method: 'drawIndexed',
      immediateSize: 4,
      adjacentImmediate: true,
      immediate,
    });
  }
  for (let index = 0; index < 8; index += 1) {
    events.push({
      arguments: [36, 1, 0, 0, index],
      method: 'drawIndexed',
      immediateSize: 0,
      adjacentImmediate: false,
      immediate: null,
    });
  }
  return {
    acquisition: {
      adapterInfo: { isFallbackAdapter: false },
      adapterRequests: [{
        argumentsCount: 1,
        descriptor: { featureLevel: 'compatibility', xrCompatible: false },
      }],
      deviceRequests: [{
        argumentsCount: 1,
        descriptor: { requiredFeatures: [], requiredLimits: {} },
      }],
    },
    capabilities: {
      maxImmediateSize: 64,
      wgslLanguageFeatures: ['immediate_address_space'],
    },
    device: {
      callableSetImmediates: true,
      errorScopes: {
        pops: [0, 1, 2].map((id) => ({
          error: null,
          filter: 'validation',
          id,
          rejection: null,
          settled: true,
        })),
        pushes: [0, 1, 2].map((id) => ({ filter: 'validation', id })),
      },
      pipelinePromiseRejections: [],
      uncapturedErrors: [],
      unexpectedLoss: null,
    },
    pipelineLayouts: [
      { immediateSize: 4 },
      { immediateSize: 0 },
      { immediateSize: 0 },
    ],
    ready: {
      backendIsWebGPU: true,
      batchCounts: [8, 8],
      batchOrders: [
        [3, 4, 5, 6, 7, 0, 1, 2],
        [3, 4, 5, 6, 7, 0, 1, 2],
      ],
      capability: true,
      deviceLimit: 64,
      ownedDevice: true,
      phase: 3,
    },
    renderPasses: [{ events }, { events: [{
      arguments: [3, 1, 0, 0],
      method: 'draw',
      immediateSize: 0,
      adjacentImmediate: false,
      immediate: null,
    }] }],
    renderPipelines: [
      { immediateSize: 4 },
      { immediateSize: 0 },
      { immediateSize: 0 },
    ],
    setupError: null,
    snapshot: {
      collectionCalls: 1,
      compilationInfo: [
        { calls: 1, index: 0, instrumented: true, label: null, status: 'fulfilled' },
        { calls: 1, index: 1, instrumented: true, label: null, status: 'fulfilled' },
      ],
      progress: { stage: 'complete', status: 'complete' },
    },
    shaderModules: [{
      code: 'requires immediate_address_space;\nvar<immediate> nodeDrawIndex : u32;',
      label: null,
      messages: [],
    }, {
      code: 'ordinary vertex',
      label: null,
      messages: [],
    }],
    unhandledRejections: [],
    windowErrors: [],
  };
}

const cleanObservations = {
  browserDisconnects: [],
  consoleErrors: [],
  consoleWarnings: [],
  httpErrors: [],
  pageCrashes: [],
  pageErrors: [],
  requestFailures: [],
};

function makeLateState(evidence) {
  const { shaderModules, ...state } = structuredClone(evidence);
  return {
    ...state,
    shaderModuleCount: shaderModules.length,
    shaderModuleLabels: shaderModules.map((module) => module.label),
  };
}

test('v3 protocol binds the prior attempt, manifest, and frozen patch', async () => {
  const protocol = JSON.parse(await readFile(protocolPath, 'utf8'));
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const patchPath = path.join(repoRoot, ...protocol.patch.path.split('/'));
  const patch = await readFile(patchPath);
  const patchText = patch.toString('utf8');
  const headers = [...patchText.matchAll(/^diff --git /gmu)];

  assert.equal(patch.length, protocol.patch.bytes);
  assert.equal(sha256(patch), protocol.patch.sha256);
  assert.equal(manifest.patch.bytes, protocol.patch.bytes);
  assert.equal(manifest.patch.sha256, protocol.patch.sha256);
  assert.equal(headers.length, manifest.patch.files);
  assert.equal(patch.includes(13), false, 'patch uses canonical LF');
  assert.match(patchText, /new file mode 100644\nindex 0{40}\.\.[0-9a-f]{40}\n--- \/dev\/null\n\+\+\+ b\/examples\/webgpu_tsl_draw_index\.html/u);
  assert.match(patchText, /new file mode 100644\nindex 0{40}\.\.[0-9a-f]{40}\n--- \/dev\/null\n\+\+\+ b\/test\/unit\/src\/renderers\/webgpu\/WebGPUDrawIndex\.tests\.js/u);
  assert.match(
    patchText,
    /diff --git a\/examples\/screenshots\/webgpu_tsl_draw_index\.jpg b\/examples\/screenshots\/webgpu_tsl_draw_index\.jpg[\s\S]*GIT binary patch/u,
  );
  assert.match(
    patchText,
    /renderer\.backend\.isWebGPUBackend !== true \|\| renderer\.backend\.capabilities\.hasImmediateData\(\) === false/u,
  );
  assert.deepEqual(protocol.priorAttempt.report, {
    path: 'results/development/three-draw-index-v2-owned-device-canary/canary-994260a-v2/report.json',
    bytes: 5953,
    sha256: '866e5d557aa0c83c1d8e0fe27f1b32c4510de239d49832ea9fe061bccb963258',
  });
  assert.equal(
    protocol.priorAttempt.sidecar.sha256,
    '1d455559ad621e26d59847256d66c44887b529bdb99247c54cd745e62b84a2cc',
  );
  assert.equal(
    protocol.priorAttempt.screenshot.sha256,
    'c053908996302b34f416103a1bd0f5312fc5238f7e0e20ef4aa85205de2a27c7',
  );
  assert.equal(protocol.priorAttempt.expectedStatus, 'failed');
  assert.equal(
    protocol.priorAttempt.expectedFailure,
    'Final evidence snapshot exceeded 30000 ms.',
  );
  assert.equal(
    protocol.output.directory,
    'results/development/three-draw-index-v3-owned-device-canary/canary-994260a-v3',
  );
  assert.equal(protocol.output.checkpoint, 'initial-checkpoint.json');
  assert.equal(protocol.output.preCollectionCheckpoint, 'pre-collection-checkpoint.json');
  assert.notEqual(
    protocol.output.directory,
    path.dirname(protocol.priorAttempt.report.path),
  );
});

test('v3 evidence validator accepts the exact command, capability, and one-shot contract', async () => {
  const protocol = JSON.parse(await readFile(protocolPath, 'utf8'));
  const validation = validateEvidence(makeEvidence(), protocol, cleanObservations);

  assert.equal(validation.verified, true);
  assert.deepEqual(validation.reasons, []);
  assert.deepEqual(
    validation.events.immediateSetters.map((event) => event.values[0]),
    [0, 1, 2, 3, 4, 5, 6, 7],
  );
  assert.equal(validation.events.immediateDraws.length, 8);
  assert.equal(validation.events.ordinaryDraws.length, 8);
});

test('v3 evidence validator rejects semantic aliases, limit promotion, and repeat collection', async () => {
  const protocol = JSON.parse(await readFile(protocolPath, 'utf8'));
  const evidence = makeEvidence();
  evidence.renderPasses[0].events[2].values = [0];
  evidence.acquisition.deviceRequests[0].descriptor.requiredLimits.maxImmediateSize = 4;
  evidence.snapshot.compilationInfo[0].calls = 2;
  const validation = validateEvidence(evidence, protocol, cleanObservations);

  assert.equal(validation.verified, false);
  assert.equal(
    validation.reasons.some((reason) => reason.includes('exact ordinals')),
    true,
  );
  assert.equal(
    validation.reasons.some((reason) => reason.includes('promoted maxImmediateSize')),
    true,
  );
  assert.equal(
    validation.reasons.some((reason) => reason.includes('exactly once per module')),
    true,
  );
});

test('v3 evidence validator rejects fabricated order, device, and draw work', async () => {
  const protocol = JSON.parse(await readFile(protocolPath, 'utf8'));
  const evidence = makeEvidence();
  evidence.ready.batchOrders[0][0] = 7;
  evidence.ready.ownedDevice = false;
  evidence.device.errorScopes.pops[0].error = {
    message: 'synthetic validation error',
    name: 'GPUValidationError',
  };
  evidence.renderPasses[0].events[1].arguments[0] = 0;
  evidence.renderPasses[1].events[0].arguments[0] = 6;
  const validation = validateEvidence(evidence, protocol, cleanObservations);

  assert.equal(validation.verified, false);
  for (const fragment of [
    'Actual BatchedMesh command order',
    'retain the instrumented device',
    'validation scopes did not drain cleanly',
    'exact command or adjacent setter',
    'output-transform draw',
  ]) {
    assert.equal(
      validation.reasons.some((reason) => reason.includes(fragment)),
      true,
      `missing rejection for ${fragment}`,
    );
  }
});

test('browser argument policy rejects GPU-enabling overrides', async () => {
  const protocol = JSON.parse(await readFile(protocolPath, 'utf8'));
  assert.deepEqual(
    findForbiddenArguments(
      ['chrome.exe', '--headless', '--enable-unsafe-webgpu', '--use-angle=vulkan'],
      protocol.browser.forbiddenArgumentPrefixes,
    ),
    ['--enable-unsafe-webgpu', '--use-angle=vulkan'],
  );
  assert.deepEqual(
    findForbiddenArguments(
      ['chrome.exe', '--headless', '--disable-background-networking'],
      protocol.browser.forbiddenArgumentPrefixes,
    ),
    [],
  );
});

test('example preparation rewrites an exact source shape and fails closed', () => {
  const source = [
    '../build/three.webgpu.js',
    '../build/three.webgpu.js',
    '../build/three.tsl.js',
    '\t\t\t\trenderer.render( scene, camera );',
  ].join('\n');
  const prepared = prepareExampleSource(source, '/build');
  assert.equal(prepared.includes('../build/three.webgpu.js'), false);
  assert.equal(prepared.includes('../build/three.tsl.js'), false);
  assert.match(prepared, /window\.__drawIndexV3State\.ready/u);
  assert.throws(
    () => prepareExampleSource(source.replace('../build/three.tsl.js', ''), '/build'),
    /TSL import is ambiguous/u,
  );
});

test('late synchronous state must be identical to the initial state', () => {
  const evidence = makeEvidence();
  assert.deepEqual(validateLateState(evidence, null), {
    reasons: ['Late synchronous state is not an object.'],
    verified: false,
  });
  const cleanLateState = makeLateState(evidence);
  assert.deepEqual(validateLateState(evidence, cleanLateState), {
    reasons: [],
    verified: true,
  });

  const changedLateState = structuredClone(cleanLateState);
  changedLateState.device.uncapturedErrors.push({
    name: 'GPUValidationError',
    message: 'synthetic late error',
  });
  changedLateState.shaderModuleCount += 1;
  const validation = validateLateState(evidence, changedLateState);
  assert.equal(validation.verified, false);
  assert.equal(
    validation.reasons.some((reason) => reason.includes('count changed')),
    true,
  );
  assert.equal(
    validation.reasons.some((reason) => reason.includes('state changed')),
    true,
  );
});

test('initial checkpoint is immutable-by-copy and explicitly provisional', async () => {
  const evidence = makeEvidence();
  const initialValidation = { reasons: [], verified: true };
  const report = {
    claimBoundary: 'bounded claim',
    protocol: { path: 'protocol.json', sha256: 'protocol' },
    priorAttempt: { before: { status: 'failed' }, after: null },
    research: { before: { commit: 'research' } },
    source: { patch: { sha256: 'patch' } },
    build: { files: {} },
    browser: { productVersion: 'browser' },
    example: { dom: { order: 'order' } },
    observations: structuredClone(cleanObservations),
    evidence: {
      preCollection: makeLateState(evidence),
      initial: summarizeEvidence(evidence),
      late: null,
    },
  };
  const preCollectionCheckpoint = createPreCollectionCheckpoint(report);
  const checkpoint = createInitialCheckpoint(report, initialValidation);
  report.evidence.preCollection.ready.phase = 6;
  report.evidence.initial.ready.phase = 7;
  report.observations.consoleErrors.push('late mutation');

  assert.equal(
    preCollectionCheckpoint.kind,
    'three-webgpu-draw-index-v3-pre-collection-checkpoint',
  );
  assert.equal(preCollectionCheckpoint.status, 'pre-collection-not-final');
  assert.equal(preCollectionCheckpoint.evidence.ready.phase, 3);
  assert.equal(checkpoint.kind, 'three-webgpu-draw-index-v3-initial-checkpoint');
  assert.equal(checkpoint.status, 'checkpointed-not-final');
  assert.equal(checkpoint.evidence.ready.phase, 3);
  assert.deepEqual(checkpoint.observations.consoleErrors, []);
  assert.equal(checkpoint.pending.length, 4);
  assert.throws(
    () => createInitialCheckpoint({ evidence: { initial: null } }, initialValidation),
    /Initial evidence is unavailable/u,
  );
  assert.throws(
    () => createPreCollectionCheckpoint({ evidence: { preCollection: null } }),
    /Pre-collection state is unavailable/u,
  );
});

test('v3 source performs one compilation-info call and no final async snapshot', async () => {
  const source = await readFile(
    path.join(repoRoot, 'scripts', 'probe-three-draw-index-v3-owned-device.mjs'),
    'utf8',
  );
  assert.equal(source.split('.getCompilationInfo()').length - 1, 1);
  assert.equal(
    source.split('page.evaluate(() => window.__drawIndexV3CollectInitialEvidence())').length - 1,
    1,
  );
  assert.equal(source.includes('__drawIndexV2Snapshot'), false);
  assert.equal(source.includes('Final evidence snapshot'), false);
  assert.equal(source.includes("'.generated', 'three-draw-index-v3-live-build'"), true);
  assert.equal(source.includes("'.generated', 'three-draw-index-v2-live-build'"), false);
  assert.match(source, /const PAGE_STAGE_TIMEOUT_MS = 30_000/u);
  assert.match(source, /INITIAL_COLLECTION_TIMEOUT_MS/u);
  assert.match(source, /compilation\.calls \+= 1/u);
  assert.match(source, /compilation\.instrumented = replaceCallable/u);
  assert.equal(source.split('validatePriorAttempt(protocol.priorAttempt)').length - 1, 2);
  assert.match(source, /await mkdir\(runRoot\)/u);
  assert.match(source, /await mkdir\(targetRoot\)/u);
  assert.match(source, /page\.screenshot\(\{ fullPage: true \}\)/u);
  assert.match(source, /writeFile\(screenshotPath, screenshotBytes, \{ flag: 'wx' \}\)/u);
  assert.match(
    source,
    /writeFile\(\s*preCollectionCheckpointPath,\s*preCollectionCheckpointBytes,\s*\{ flag: 'wx' \}/u,
  );
  assert.match(source, /writeFile\(checkpointPath, checkpointBytes, \{ flag: 'wx' \}\)/u);
  assert.match(source, /await chmod\(checkpointPath, 0o444\)/u);
  assert.match(source, /checkpointAfter\.length === report\.checkpoint\.bytes/u);
  assert.match(source, /window\.__drawIndexV3LateState = \(\) => structuredClone/u);
  assert.match(source, /initial-checkpoint\.json|protocol\.output\.checkpoint/u);
  const postflightIndex = source.indexOf('const researchAfter = await gitSnapshot(repoRoot)');
  const lateStateIndex = source.indexOf(
    'page.evaluate(() => window.__drawIndexV3LateState())',
    postflightIndex,
  );
  assert.equal(lateStateIndex > postflightIndex, true);
  assert.equal(lateStateIndex < source.indexOf("report.status = 'passed'"), true);
});
