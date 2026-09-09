import assert from 'node:assert/strict';
import { execFile as execFileCallback } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import {
  findForbiddenArguments,
  parsePrimitiveIndexArguments,
  validatePrimitiveIndexEvidence,
  verifyBuiltPrimitiveIndexCandidate,
} from '../scripts/probe-three-primitive-index.mjs';
import {
  PRIMITIVE_INDEX_CANARY_KIND,
  PRIMITIVE_INDEX_INDICES,
  PRIMITIVE_INDEX_MIN_PIXELS_PER_TUPLE,
  PRIMITIVE_INDEX_ORACLE_SIZE,
  PRIMITIVE_INDEX_SENTINEL,
  analyzePrimitiveIndexOracle,
  expectedPrimitiveIndexTuples,
} from '../src/three-primitive-index-canary-contract.js';

const execFile = promisify(execFileCallback);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const protocolPath = path.join(repoRoot, 'protocols', 'three-primitive-index-canary-v3.json');
const candidateRoot = path.join(
  repoRoot,
  '.generated',
  'three-primitive-index-current-dev-candidate',
);

const cleanObservations = Object.freeze({
  browserDisconnects: [],
  consoleErrors: [],
  consoleWarnings: [],
  httpErrors: [],
  pageCrashes: [],
  pageErrors: [],
  requestFailures: [],
});

async function protocol() {
  return JSON.parse(await readFile(protocolPath, 'utf8'));
}

function oracleWords(pixelsPerTuple = PRIMITIVE_INDEX_MIN_PIXELS_PER_TUPLE) {
  const { width, height } = PRIMITIVE_INDEX_ORACLE_SIZE;
  const words = new Uint32Array(width * height * 4);
  const halfWidth = width / 2;
  for (let instance = 0; instance < 2; instance += 1) {
    let localPixel = 0;
    for (const tuple of expectedPrimitiveIndexTuples().filter((value) => value[1] === instance)) {
      for (let count = 0; count < pixelsPerTuple; count += 1) {
        const x = (instance * halfWidth) + (localPixel % halfWidth);
        const y = Math.floor(localPixel / halfWidth);
        words.set(tuple, ((y * width) + x) * 4);
        localPixel += 1;
      }
    }
  }
  return words;
}

function makeEvidence(protocolValue) {
  const words = oracleWords();
  const oracle = analyzePrimitiveIndexOracle(words);
  const primitiveShader = `${protocolValue.wgsl.directive}\nstruct FragmentInput {\n  `
    + `${protocolValue.wgsl.builtin},\n};`;
  return {
    page: {
      kind: PRIMITIVE_INDEX_CANARY_KIND,
      status: 'passed',
      error: null,
      geometry: {
        attributes: ['position'],
        indexArrayType: 'Uint16Array',
        indexCount: 12,
        indexValues: [...PRIMITIVE_INDEX_INDICES],
        instanceCount: 2,
        positionCount: 5,
      },
      oracle: {
        ...oracle,
        byteLength: words.byteLength,
        expectedTuples: expectedPrimitiveIndexTuples(),
        readbackSha256: 'a'.repeat(64),
        typedArray: 'Uint32Array',
      },
      renderer: { backendIsWebGPU: true, devicePrimitiveIndex: true },
      validationScope: null,
    },
    instrumentation: {
      acquisition: {
        adapterFeatures: ['primitive-index'],
        adapterInfo: { isFallbackAdapter: false },
        adapterRequests: [{ argumentsCount: 1, descriptor: { featureLevel: 'compatibility' } }],
        deviceFeatures: ['primitive-index'],
        deviceRequests: [{
          argumentsCount: 1,
          descriptor: { requiredFeatures: ['primitive-index'], requiredLimits: {} },
        }],
      },
      device: {
        errorScopes: {
          pushes: [{ filter: 'validation', id: 0 }],
          pops: [{
            error: null,
            filter: 'validation',
            id: 0,
            rejection: null,
            settled: true,
          }],
        },
        pipelinePromiseRejections: [],
        uncapturedErrors: [],
        unexpectedLoss: null,
      },
      drawEncoders: [
        {
          kind: 'render-pass',
          events: [
            {
              method: 'setIndexBuffer',
              format: protocolValue.geometry.gpuIndexFormat,
              offset: 0,
              size: null,
            },
            { method: 'drawIndexed', arguments: [12, 2, 0, 0, 0] },
          ],
        },
        {
          kind: 'render-pass',
          events: [
            {
              method: 'setIndexBuffer',
              format: protocolValue.geometry.gpuIndexFormat,
              offset: 0,
              size: null,
            },
            { method: 'drawIndexed', arguments: [12, 2, 0, 0, 0] },
          ],
        },
      ],
      setupError: null,
      shaderModules: [
        { code: primitiveShader, label: 'oracle', messages: [] },
        { code: primitiveShader, label: 'visual', messages: [] },
        { code: 'ordinary output shader', label: 'output', messages: [] },
      ],
      textures: [{ format: 'rgba32uint', label: 'oracle', sampleCount: 1, usage: 17 }],
      wgslLanguageFeatures: [],
      windowErrors: [],
      unhandledRejections: [],
    },
  };
}

test('primitiveIndex contract accepts all eight exact u32 tuples with reset semantics', () => {
  const analysis = analyzePrimitiveIndexOracle(oracleWords());

  assert.equal(analysis.exact, true);
  assert.equal(analysis.resetProven, true);
  assert.equal(analysis.instanceSpatialMismatches, 0);
  assert.deepEqual(analysis.primitiveIdsByInstance, [[0, 1, 2, 3], [0, 1, 2, 3]]);
  assert.equal(analysis.tuples.length, 8);
  assert.equal(analysis.tuples.every((record) => record.sufficient), true);
  assert.equal(analysis.unexpected.length, 0);
  assert.equal(analysis.backgroundPixels > 0, true);
});

test('primitiveIndex contract rejects a wrong tuple and an undersized readback', () => {
  const words = oracleWords();
  words.set([99, 1, PRIMITIVE_INDEX_SENTINEL, 1], 0);
  const analysis = analyzePrimitiveIndexOracle(words);

  assert.equal(analysis.exact, false);
  assert.deepEqual(analysis.unexpected[0].tuple, [99, 1, PRIMITIVE_INDEX_SENTINEL, 1]);
  assert.throws(
    () => analyzePrimitiveIndexOracle(new Uint32Array(4)),
    /expected 230400/u,
  );
  assert.throws(() => analyzePrimitiveIndexOracle(new Uint8Array(4)), /Uint32Array/u);
});

test('primitiveIndex validator accepts feature negotiation, indexed draws, WGSL, and oracle', async () => {
  const protocolValue = await protocol();
  const result = validatePrimitiveIndexEvidence(
    makeEvidence(protocolValue),
    protocolValue,
    cleanObservations,
  );

  assert.equal(result.verified, true);
  assert.deepEqual(result.reasons, []);
  assert.equal(result.indexedDraws.length, 2);
  assert.equal(result.primitiveShaderModuleCount, 2);
});

test('primitiveIndex validator rejects absent negotiation and the wrong instance count', async () => {
  const protocolValue = await protocol();
  const evidence = makeEvidence(protocolValue);
  evidence.instrumentation.acquisition.deviceRequests[0].descriptor.requiredFeatures = [];
  evidence.instrumentation.drawEncoders[0].events[1].arguments[1] = 1;
  const result = validatePrimitiveIndexEvidence(evidence, protocolValue, cleanObservations);

  assert.equal(result.verified, false);
  assert.equal(result.reasons.some((reason) => reason.includes('requiredFeatures')), true);
  assert.equal(result.reasons.some((reason) => reason.includes('[12, 2, 0, 0, 0]')), true);
});

test('primitiveIndex validator binds the exact widened GPU index format', async () => {
  const protocolValue = await protocol();
  const evidence = makeEvidence(protocolValue);
  evidence.instrumentation.drawEncoders[0].events[0].format = 'uint16';
  const result = validatePrimitiveIndexEvidence(evidence, protocolValue, cleanObservations);

  assert.equal(result.verified, false);
  assert.equal(result.reasons.some((reason) => reason.includes('uint32')), true);
});

test('primitiveIndex validator rejects an ID attribute, missing directive, and GPU errors', async () => {
  const protocolValue = await protocol();
  const evidence = makeEvidence(protocolValue);
  evidence.page.geometry.attributes.push('primitiveId');
  evidence.instrumentation.shaderModules[0].code = 'fragment without directive';
  evidence.instrumentation.device.errorScopes.pops[0].error = {
    name: 'GPUValidationError',
    message: 'synthetic',
  };
  const result = validatePrimitiveIndexEvidence(evidence, protocolValue, cleanObservations);

  assert.equal(result.verified, false);
  assert.equal(result.reasons.some((reason) => reason.includes('primitive-ID')), true);
  assert.equal(result.reasons.some((reason) => reason.includes('fragment modules')), true);
  assert.equal(result.reasons.some((reason) => reason.includes('error scope')), true);
});

test('primitiveIndex build verifier requires feature, directive, builtin, and TSL export', () => {
  const inventory = Object.fromEntries([
    'three.core.js',
    'three.module.js',
    'three.tsl.js',
    'three.webgpu.js',
    'three.webgpu.nodes.js',
  ].map((filename) => [filename, { bytes: 1, sha256: 'a'.repeat(64) }]));
  const built = {
    inventory,
    modules: {
      'three.tsl.js': Buffer.from('export { primitiveIndex };'),
      'three.webgpu.js': Buffer.from("PrimitiveIndex: 'primitive-index'"),
      'three.webgpu.nodes.js': Buffer.from(
        "enableDirective( 'primitive_index' ); getBuiltin( 'primitive_index', 'primitiveIndex', 'u32' );",
      ),
    },
  };

  assert.equal(verifyBuiltPrimitiveIndexCandidate(built).verified, true);
  built.modules['three.tsl.js'] = Buffer.from('ordinary TSL');
  assert.equal(verifyBuiltPrimitiveIndexCandidate(built).verified, false);
});

test('primitiveIndex browser policy and CLI reject capability-forcing flags and loose invocations', () => {
  assert.deepEqual(
    findForbiddenArguments(
      ['chrome.exe', '--enable-dawn-features=primitive_index', '--headless'],
      ['--enable-dawn-features'],
    ),
    ['--enable-dawn-features=primitive_index'],
  );
  assert.deepEqual(parsePrimitiveIndexArguments(['--offline-preflight']), {
    mode: 'offline-preflight',
  });
  assert.throws(() => parsePrimitiveIndexArguments([]), /Expected --browser/u);
  assert.throws(
    () => parsePrimitiveIndexArguments([
      '--browser',
      'chrome.exe',
      '--expected-research-commit',
      'short',
    ]),
    /full Git object ID/u,
  );
});

test('primitiveIndex protocol binds a clean immutable candidate and explicit canary source', async () => {
  const protocolValue = await protocol();
  const [{ stdout: commit }, { stdout: tree }, { stdout: status }, html, module] =
    await Promise.all([
      execFile('git', ['-C', candidateRoot, 'rev-parse', 'HEAD'], { encoding: 'utf8' }),
      execFile('git', ['-C', candidateRoot, 'rev-parse', 'HEAD^{tree}'], { encoding: 'utf8' }),
      execFile(
        'git',
        ['-C', candidateRoot, 'status', '--porcelain=v1', '--untracked-files=all'],
        { encoding: 'utf8' },
      ),
      readFile(path.join(repoRoot, protocolValue.page.path)),
      readFile(path.join(repoRoot, protocolValue.page.module)),
    ]);
  const moduleText = module.toString('utf8');

  assert.equal(protocolValue.executionId, 'primitive-index-c788032-v3');
  assert.equal(protocolValue.geometry.gpuIndexFormat, 'uint32');
  assert.match(protocolValue.geometry.gpuIndexFormatRationale, /widens.+Uint32Array/u);
  assert.equal(commit.trim(), protocolValue.candidate.commit);
  assert.equal(tree.trim(), protocolValue.candidate.tree);
  assert.equal(status.trim(), '');
  const htmlText = html.toString('utf8');
  assert.match(htmlText, /No face-ID buffer/u);
  assert.match(
    htmlText,
    /"three\/webgpu"\s*:\s*"\/__three_primitive_index__\/three\.webgpu\.js"/u,
  );
  assert.equal([...htmlText].every((character) => character.codePointAt(0) < 128), true);
  assert.match(moduleText, /new THREE\.Uint16BufferAttribute\(PRIMITIVE_INDEX_INDICES, 1\)/u);
  assert.match(moduleText, /uvec4\(\s*primitiveIndex,\s*instanceIndex,/u);
  assert.match(moduleText, /new THREE\.InstancedMesh\(/u);
  assert.doesNotMatch(moduleText, /setAttribute\(\s*['"](?:primitive|face)/u);
  assert.equal(moduleText.includes('PASS: exact u32 tuples 0-3'), true);
  assert.equal([...moduleText].every((character) => character.codePointAt(0) < 128), true);
  assert.equal(moduleText.includes('\ufffd'), false);
});
