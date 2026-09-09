import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  findForbiddenArguments,
  prepareExampleSource,
  validateEvidence,
} from '../scripts/probe-three-draw-index-v2-owned-device.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const protocolPath = path.join(
  repoRoot,
  'protocols',
  'three-draw-index-v2-owned-device-v1.json',
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
    shaderModules: [{
      code: 'requires immediate_address_space;\nvar<immediate> nodeDrawIndex : u32;',
      messages: [],
    }, {
      code: 'ordinary vertex',
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

test('v2 protocol, manifest, and patch are mutually bound', async () => {
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
});

test('v2 evidence validator accepts the exact command and capability contract', async () => {
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

test('v2 evidence validator rejects semantic aliases and limit promotion', async () => {
  const protocol = JSON.parse(await readFile(protocolPath, 'utf8'));
  const evidence = makeEvidence();
  evidence.renderPasses[0].events[2].values = [0];
  evidence.acquisition.deviceRequests[0].descriptor.requiredLimits.maxImmediateSize = 4;
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
});

test('v2 evidence validator rejects fabricated order, device, and draw work', async () => {
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
  assert.match(prepared, /window\.__drawIndexV2State\.ready/u);
  assert.throws(
    () => prepareExampleSource(source.replace('../build/three.tsl.js', ''), '/build'),
    /TSL import is ambiguous/u,
  );
});
