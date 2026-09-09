import { createHash, randomBytes } from 'node:crypto';
import {
  access,
  mkdir,
  readFile,
  stat,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

import {
  buildThreeImmediateOverlay,
  DEFAULT_OUTPUT_ROOT,
  EXPECTED_TARGET_HASHES,
  EXPECTED_THREE_VERSION,
} from './build-three-immediate-overlay.mjs';
import {
  buildThreeImmediateDevOverlay,
  DEFAULT_THREE_DEV_OUTPUT_ROOT,
  EXPECTED_THREE_DEV_COMMIT,
  EXPECTED_THREE_DEV_PATCHED_HASHES,
  EXPECTED_THREE_DEV_REVISION,
  EXPECTED_THREE_DEV_TARGET_HASHES,
  EXPECTED_THREE_DEV_TREE,
  EXPECTED_THREE_DEV_VERSION,
} from './build-three-immediate-dev-overlay.mjs';

export const THREE_IMMEDIATE_OVERLAY_RUNNER_KIND =
  'three-immediate-overlay-development-runner';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const THREE_IMMEDIATE_OVERLAY_TARGET_R185 = 'r185';
export const THREE_IMMEDIATE_OVERLAY_TARGET_DEV = 'dev';
const TARGET_CONFIGURATION_INJECTION_MARKER =
  '/* THREE_IMMEDIATE_CANARY_TARGET_CONFIGURATION_V1 */ null';
const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
});
const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const EXPECTED_R185_TSL_SHA256 =
  'c3844718a7becd4d2d10fef29a503e6a8789bd1636e2e62767f2cf00fa56f8c1';
const EXPECTED_DEV_TSL_SHA256 =
  '6b687980397e3c2d8084bee1494f1d324d968282deaa97ed752957685f9e5461';
const DRAW_COUNT = 4;
const INSTANCES_PER_DRAW = 2;
const INDIRECT_WORD_COUNT = 5;
const TARGET_WIDTH = 8;
const TARGET_HEIGHT = 1;
const CANONICAL_IMMEDIATE_BASES = Object.freeze([0, 2, 4, 6]);
const MUTATED_IMMEDIATE_BASES = Object.freeze([1, 3, 5, 0]);
const VISIBLE_IDS = Object.freeze([5, 0, 7, 2, 1, 6, 3, 4]);
const EXPECTED_INDIRECT_OFFSETS = Object.freeze([0, 20, 40, 60]);
const EXPECTED_COMMAND_WORDS = Object.freeze([
  3, 2, 0, 0, 0,
  3, 2, 3, 0, 0,
  3, 2, 6, 0, 0,
  3, 2, 9, 0, 0,
]);
const COMMON_IMPLEMENTED_GATES = Object.freeze([
  'wgsl-language-feature-and-device-limit',
  'one-static-bundle-group-one-mesh',
  'multiple-indexed-indirect-offsets',
  'uint32-immediate-bases-parallel-to-offsets',
  'exact-immediate-wgsl-directive-declaration-and-address-use',
  'actual-immediate-pipeline-layout-size-four',
  'actual-following-non-immediate-control-layout-size-zero',
  'pipeline-data-immediate-size-four',
  'cpu-and-gpu-indirect-commands-including-word-four-zero',
  'command-readback-bound-to-draw-gpu-buffer',
  'draw-ordinal-specific-noncommutative-rgba8-output-oracle',
  'set-immediates-immediately-adjacent-to-each-indirect-draw',
  'shader-module-pipeline-bundle-identity-chain',
  'render-bundle-source-array-snapshot-after-native-finish',
  'instrumented-cached-bundle-reuse-after-source-mutation',
  'render-pass-execute-bundles-bound-to-cached-bundle',
  'separate-uninstrumented-cached-bundle-snapshot-after-source-mutation',
  'zero-uncaptured-errors-and-unexpected-losses-observed-after-device-creation',
]);
const COMMON_DEFERRED_GATES = Object.freeze([
  'raw-webgpu-negative-validation-controls',
  'full-65536-object-32-draw-workload',
  'integrated-A-I-F-lanes-and-all-six-orders',
  'both-visibility-levels-and-second-sentinel-schedule',
  'full-browser-distribution-and-dependency-closure-attestation',
  'full-resource-identity-and-disposal-inventory',
  'complete-preflight-postflight-protocol-artifact-set',
]);

function frozenPageConfiguration({
  key,
  sourceFamily,
  expectedRevision,
  scope,
  implemented,
  deferred,
}) {
  return Object.freeze({
    schemaVersion: 1,
    key,
    sourceFamily,
    expectedRevision,
    scope,
    implemented: Object.freeze([...implemented]),
    deferred: Object.freeze([...deferred]),
  });
}

const R185_PAGE_CONFIGURATION = frozenPageConfiguration({
  key: THREE_IMMEDIATE_OVERLAY_TARGET_R185,
  sourceFamily: 'installed-three-r185.1-package',
  expectedRevision: '185',
  scope: 'minimal Three.js r185.1 generated-overlay integration only',
  implemented: [
    'generated-three-r185.1-runtime-signature',
    ...COMMON_IMPLEMENTED_GATES,
  ],
  deferred: [
    ...COMMON_DEFERRED_GATES.slice(0, 4),
    'current-upstream-three-development-commit-overlay',
    ...COMMON_DEFERRED_GATES.slice(4),
  ],
});

const DEV_PAGE_CONFIGURATION = frozenPageConfiguration({
  key: THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  sourceFamily: 'pinned-upstream-three-dev-commit',
  expectedRevision: EXPECTED_THREE_DEV_REVISION,
  scope: 'minimal pinned upstream Three.js dev generated-overlay integration only',
  implemented: [
    'pinned-upstream-three-dev-runtime-signature',
    'current-upstream-three-development-commit-overlay',
    ...COMMON_IMPLEMENTED_GATES,
  ],
  deferred: [...COMMON_DEFERRED_GATES],
});

const TARGETS = Object.freeze({
  [THREE_IMMEDIATE_OVERLAY_TARGET_R185]: Object.freeze({
    page: R185_PAGE_CONFIGURATION,
    version: EXPECTED_THREE_VERSION,
    outputRoot: DEFAULT_OUTPUT_ROOT,
    resultDirectory: 'three-immediate-overlay-canary',
    manifestFilename: 'THREE_IMMEDIATE_OVERLAY.json',
    overlayName: 'three-immediate-address-attribution',
    expectedTslSha256: EXPECTED_R185_TSL_SHA256,
    build: buildThreeImmediateOverlay,
  }),
  [THREE_IMMEDIATE_OVERLAY_TARGET_DEV]: Object.freeze({
    page: DEV_PAGE_CONFIGURATION,
    version: EXPECTED_THREE_DEV_VERSION,
    outputRoot: DEFAULT_THREE_DEV_OUTPUT_ROOT,
    resultDirectory: 'three-immediate-dev-overlay-canary',
    manifestFilename: 'THREE_IMMEDIATE_DEV_OVERLAY.json',
    overlayName: 'three-immediate-address-attribution-dev',
    expectedTslSha256: EXPECTED_DEV_TSL_SHA256,
    build: buildThreeImmediateDevOverlay,
  }),
});
const BOUND_SOURCE_FILES = Object.freeze([
  {
    role: 'runner',
    path: fileURLToPath(import.meta.url),
  },
  {
    role: 'page-module',
    path: path.join(repoRoot, 'src', 'three-immediate-overlay-canary.js'),
  },
  {
    role: 'entry-html',
    path: path.join(repoRoot, 'three-immediate-canary.html'),
  },
]);

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function resolveTarget(targetKey = THREE_IMMEDIATE_OVERLAY_TARGET_R185) {
  const target = TARGETS[targetKey];
  if (target === undefined) {
    throw new Error(
      `Unknown Three immediate overlay target ${JSON.stringify(targetKey)}; expected "r185" or "dev".`,
    );
  }
  return target;
}

function defaultResultPath(target) {
  return path.join(
    repoRoot,
    'results',
    'development',
    target.resultDirectory,
    `canary-${randomBytes(12).toString('hex')}`,
    'report.json',
  );
}

function targetIdentity(pageConfiguration) {
  return {
    schemaVersion: pageConfiguration.schemaVersion,
    key: pageConfiguration.key,
    sourceFamily: pageConfiguration.sourceFamily,
    expectedRevision: pageConfiguration.expectedRevision,
  };
}

function exactTargetIdentity(value, expected) {
  return value?.schemaVersion === expected.schemaVersion
    && value?.key === expected.key
    && value?.sourceFamily === expected.sourceFamily
    && value?.expectedRevision === expected.expectedRevision
    && Object.keys(value).length === Object.keys(expected).length;
}

export function pageConfigurationForThreeImmediateOverlayTarget(targetKey = 'r185') {
  return structuredClone(resolveTarget(targetKey).page);
}

export function injectThreeImmediateOverlayCanaryTarget(source, targetKey = 'r185') {
  const configuration = resolveTarget(targetKey).page;
  const markerCount = source.split(TARGET_CONFIGURATION_INJECTION_MARKER).length - 1;
  if (markerCount !== 1) {
    throw new Error(
      `Expected exactly one canary target injection marker; found ${markerCount}.`,
    );
  }
  return source.replace(
    TARGET_CONFIGURATION_INJECTION_MARKER,
    `Object.freeze(${JSON.stringify(configuration)})`,
  );
}

function exactArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function expectedOutputForBases(bases) {
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
      output[byteOffset + 3] = 255;
    }
  }
  return [...output];
}

function countMatches(value, pattern) {
  return typeof value === 'string' ? (value.match(pattern)?.length ?? 0) : 0;
}

async function collectBoundSourceFiles(targetKey) {
  return Promise.all(BOUND_SOURCE_FILES.map(async (record) => {
    const bytes = await readFile(record.path);
    const evidence = {
      role: record.role,
      path: path.relative(repoRoot, record.path).replaceAll('\\', '/'),
      byteLength: bytes.length,
      sha256: sha256(bytes),
    };
    if (record.role === 'page-module') {
      evidence.targetInjectedSha256 = sha256(Buffer.from(
        injectThreeImmediateOverlayCanaryTarget(bytes.toString('utf8'), targetKey),
        'utf8',
      ));
    }
    return evidence;
  }));
}

function boundSourceFilesMatch(left, right) {
  const expectedPaths = BOUND_SOURCE_FILES.map((record) => ({
    role: record.role,
    path: path.relative(repoRoot, record.path).replaceAll('\\', '/'),
  }));
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === BOUND_SOURCE_FILES.length
    && exactArray(
      left.map((record) => record.role),
      BOUND_SOURCE_FILES.map((record) => record.role),
    )
    && left.every((record, index) => record.path === expectedPaths[index].path)
    && JSON.stringify(left) === JSON.stringify(right)
    && left.every((record) => Number.isSafeInteger(record.byteLength)
      && record.byteLength > 0
      && SHA256_PATTERN.test(record.sha256))
    && left.filter((record) => record.role === 'page-module').every(
      (record) => SHA256_PATTERN.test(record.targetInjectedSha256),
    );
}

function exactPageTransformEvidence(record, pageStart, configuration) {
  const expectedIdentity = targetIdentity(configuration);
  return record?.specifier === 'canary-page'
    && record?.transformCount === 1
    && record?.transformInputSha256 === pageStart?.sha256
    && record?.transformOutputSha256 === pageStart?.targetInjectedSha256
    && record?.targetConfigurationSha256 === sha256(Buffer.from(
      JSON.stringify(configuration),
      'utf8',
    ))
    && record?.injectionCount === 1
    && exactTargetIdentity(record?.target, expectedIdentity);
}

function makeBoundSourceIdentity(start, end, pageTransformInput, targetKey) {
  const configuration = resolveTarget(targetKey).page;
  const pageStart = start?.find((record) => record.role === 'page-module') ?? null;
  const filesMatch = boundSourceFilesMatch(start, end);
  const pageTransformMatches = exactPageTransformEvidence(
    pageTransformInput,
    pageStart,
    configuration,
  );
  return {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    target: targetIdentity(configuration),
    filesStart: start,
    filesEnd: end,
    filesMatch,
    pageTransformInput,
    pageTransformMatches,
    exact: filesMatch && pageTransformMatches,
  };
}

export function validateBoundSourceIdentity(
  identity,
  targetKey = THREE_IMMEDIATE_OVERLAY_TARGET_R185,
) {
  const reasons = [];
  let configuration = null;
  try {
    configuration = resolveTarget(targetKey).page;
  } catch (error) {
    reasons.push(error.message);
  }
  const pageStart = identity?.filesStart?.find(
    (record) => record?.role === 'page-module',
  ) ?? null;
  const filesMatch = boundSourceFilesMatch(identity?.filesStart, identity?.filesEnd);
  const targetMatches = configuration !== null
    && exactTargetIdentity(identity?.target, targetIdentity(configuration));
  const pageTransformMatches = configuration !== null
    && exactPageTransformEvidence(
      identity?.pageTransformInput,
      pageStart,
      configuration,
    );
  if (identity?.schemaVersion !== 1 || identity?.hashAlgorithm !== 'sha256') {
    reasons.push('bound source identity header is invalid');
  }
  if (!filesMatch || identity?.filesMatch !== true) {
    reasons.push('runner/page source bytes changed or are malformed');
  }
  if (!targetMatches) {
    reasons.push('bound source target identity is invalid');
  }
  if (!pageTransformMatches || identity?.pageTransformMatches !== true) {
    reasons.push('served page module does not match its bound source hash');
  }
  if (identity?.exact !== true || !filesMatch || !targetMatches || !pageTransformMatches) {
    reasons.push('bound source identity is not exact');
  }
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function parseArguments(argv) {
  const options = {
    outputPath: null,
    browserPath: process.env.BROWSER_PATH ?? null,
    target: THREE_IMMEDIATE_OVERLAY_TARGET_R185,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--output') {
      const value = argv[++index];
      if (value === undefined) throw new Error('--output requires a path.');
      options.outputPath = path.resolve(value);
    } else if (argument === '--browser') {
      const value = argv[++index];
      if (value === undefined) throw new Error('--browser requires a path.');
      options.browserPath = path.resolve(value);
    } else if (argument === '--target') {
      const value = argv[++index];
      if (value === undefined) throw new Error('--target requires r185 or dev.');
      options.target = value;
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }
  resolveTarget(options.target);
  return options;
}

async function findChrome(explicitPath = null) {
  const programFiles = process.env.ProgramFiles ?? 'C:\\Program Files';
  const programFilesX86 = process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
  const localAppData = process.env.LOCALAPPDATA;
  const candidates = [
    explicitPath,
    path.join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFilesX86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    localAppData
      ? path.join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe')
      : null,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
  ].filter(Boolean);

  for (const candidate of [...new Set(candidates.map((value) => path.resolve(value)))]) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Continue through exact Chrome candidates.
    }
  }
  throw new Error('No Chrome executable found. Set BROWSER_PATH or pass --browser.');
}

function normalizedModuleId(id) {
  let value = id.split('?')[0];
  if (value.startsWith('/@fs/')) value = value.slice('/@fs/'.length);
  if (/^\/[A-Za-z]:\//.test(value)) value = value.slice(1);
  return path.normalize(path.resolve(value));
}

function sourceObserverPlugin(targets, observed, targetKey) {
  const configuration = resolveTarget(targetKey).page;
  const targetById = new Map(
    Object.entries(targets).map(([specifier, target]) => [
      path.normalize(path.resolve(target)),
      specifier,
    ]),
  );
  return {
    name: 'three-immediate-overlay-source-observer',
    enforce: 'post',
    transform(code, id) {
      const specifier = targetById.get(normalizedModuleId(id));
      if (specifier === undefined) return null;
      const transformedCode = specifier === 'canary-page'
        ? injectThreeImmediateOverlayCanaryTarget(code, targetKey)
        : code;
      const previous = observed.get(specifier);
      const record = {
        specifier,
        resolvedId: path.normalize(path.resolve(id.split('?')[0])),
        transformInputSha256: sha256(Buffer.from(code, 'utf8')),
        transformOutputSha256: sha256(Buffer.from(transformedCode, 'utf8')),
        transformCount: (previous?.transformCount ?? 0) + 1,
      };
      if (specifier === 'canary-page') {
        record.injectionCount = 1;
        record.target = targetIdentity(configuration);
        record.targetConfigurationSha256 = sha256(Buffer.from(
          JSON.stringify(configuration),
          'utf8',
        ));
      }
      observed.set(specifier, record);
      return specifier === 'canary-page'
        ? { code: transformedCode, map: null }
        : null;
    },
  };
}

function serializeBrowserError(error) {
  return {
    name: String(error?.name ?? 'Error').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
    stack: error?.stack == null ? null : String(error.stack).slice(0, 8_192),
  };
}

function installPageObservers(page) {
  const observations = {
    pageErrors: [],
    consoleErrors: [],
    requestFailures: [],
    httpErrors: [],
    crashes: [],
  };
  page.on('pageerror', (error) => {
    observations.pageErrors.push(serializeBrowserError(error));
  });
  page.on('console', (message) => {
    if (message.type() === 'error') {
      observations.consoleErrors.push({
        text: message.text().slice(0, 4_096),
        location: message.location(),
      });
    }
  });
  page.on('requestfailed', (request) => {
    observations.requestFailures.push({
      url: request.url(),
      method: request.method(),
      failure: request.failure(),
    });
  });
  page.on('response', (response) => {
    if (response.status() >= 400) {
      observations.httpErrors.push({
        url: response.url(),
        status: response.status(),
        statusText: response.statusText(),
      });
    }
  });
  page.on('crash', () => observations.crashes.push({ crashed: true }));
  return observations;
}

function pageObservationsAreClean(observations) {
  return Object.values(observations).every((values) => values.length === 0);
}

function addReasonUnless(condition, reasons, message) {
  if (!condition) reasons.push(message);
}

function validateRuntimeAndTopology(result, reasons, configuration) {
  const signature = result?.runtimeOverlaySignature;
  addReasonUnless(signature?.revision === configuration.expectedRevision
    && signature?.setIndirectHasImmediateBasesParameter === true
    && signature?.setIndirectStoresImmediateBases === true
    && signature?.backendHasSetImmediates === true
    && signature?.backendHasIndexedIndirectAdjacencyMarker === true,
  reasons, 'runtime overlay signature is not exact');

  const topology = result?.evidence?.topology;
  addReasonUnless(topology?.exact === true
    && topology?.bundleGroupCount === 1
    && topology?.meshCount === 1
    && topology?.indexed === true
    && topology?.indirectOffsetCount === DRAW_COUNT
    && topology?.indirectImmediateBaseCount === DRAW_COUNT
    && topology?.bucketBaseAttributePresent === false,
  reasons, 'one-bundle/one-mesh topology evidence is inconsistent');
}

function validateShaderEvidence(result, reasons) {
  const shader = result?.evidence?.shader;
  const vertexShader = shader?.vertexShader;
  const fragmentShader = shader?.fragmentShader;
  const requirementCount = countMatches(
    vertexShader,
    /requires\s+immediate_address_space\s*;/gu,
  );
  const declarationCount = countMatches(
    vertexShader,
    /var<immediate>\s+threeImmediateDrawBase\s*:\s*u32\s*;/gu,
  );
  const addressUseCount = countMatches(
    vertexShader,
    /threeImmediateDrawBase\s*\+\s*instanceIndex/gu,
  );
  const bucketBaseCount = countMatches(vertexShader, /\bbucketBase\b/gu);
  const hashesMatch = typeof vertexShader === 'string'
    && typeof fragmentShader === 'string'
    && fragmentShader.length > 0
    && shader?.vertexSha256 === sha256(Buffer.from(vertexShader, 'utf8'))
    && shader?.fragmentSha256 === sha256(Buffer.from(fragmentShader, 'utf8'));
  addReasonUnless(shader?.exact === true
    && requirementCount === 1
    && declarationCount === 1
    && addressUseCount === 1
    && bucketBaseCount === 0
    && shader?.requirementCount === requirementCount
    && shader?.declarationCount === declarationCount
    && shader?.addressUseCount === addressUseCount
    && shader?.bucketBaseCount === bucketBaseCount
    && shader?.addressExpression === 'threeImmediateDrawBase + instanceIndex'
    && hashesMatch,
  reasons, 'shader text/count/hash evidence is inconsistent');
}

function validLayoutSnapshot(layout, immediateSize) {
  return layout?.hasOwnImmediateSize === true
    && layout?.immediateSize === immediateSize
    && layout?.bindGroupLayoutCount === 1
    && typeof layout?.layoutId === 'string'
    && layout.layoutId.length > 0;
}

function validatePipelineEvidence(result, reasons) {
  const pipeline = result?.evidence?.pipeline;
  const classified = pipeline?.classified;
  const layouts = pipeline?.layoutSnapshots;
  const shaderModules = pipeline?.shaderModuleSnapshots;
  const immediate = Array.isArray(classified)
    ? classified.filter((record) => record?.vertexUsesImmediate === true)
    : [];
  const controls = Array.isArray(classified)
    ? classified.filter((record) => record?.vertexIsControl === true)
    : [];
  const immediateRecord = immediate[0];
  const controlRecord = controls[0];
  const orderIsFourThenZero = Array.isArray(layouts)
    && layouts.length === 2
    && validLayoutSnapshot(layouts[0], 4)
    && validLayoutSnapshot(layouts[1], 0)
    && Number.isSafeInteger(layouts[0].sequence)
    && Number.isSafeInteger(layouts[1].sequence)
    && layouts[0].sequence < layouts[1].sequence
    && layouts[0].layoutId !== layouts[1].layoutId;
  const classificationsAreBound = immediate.length === 1
    && controls.length === 1
    && classified.length === 2
    && immediateRecord?.vertexIsControl === false
    && controlRecord?.vertexUsesImmediate === false
    && validLayoutSnapshot(immediateRecord?.layout, 4)
    && validLayoutSnapshot(controlRecord?.layout, 0)
    && immediateRecord.layoutId === immediateRecord.layout.layoutId
    && controlRecord.layoutId === controlRecord.layout.layoutId
    && immediateRecord.layoutId === layouts?.[0]?.layoutId
    && controlRecord.layoutId === layouts?.[1]?.layoutId
    && immediateRecord.sequence < controlRecord.sequence
    && typeof immediateRecord.pipelineId === 'string'
    && typeof controlRecord.pipelineId === 'string'
    && immediateRecord.pipelineId !== controlRecord.pipelineId;
  const shaderById = new Map(
    Array.isArray(shaderModules)
      ? shaderModules.map((record) => [record?.moduleId, record])
      : [],
  );
  const classificationsMatchSources = Array.isArray(classified)
    && classified.every((record) => {
      const code = shaderById.get(record?.vertexModuleId)?.code ?? '';
      return record?.vertexUsesImmediate
          === code.includes('threeImmediateDrawBase')
        && record?.vertexIsControl
          === code.includes('threeOverlayInstrumentedControlAddress');
    });
  const moduleSourceBindingExact = immediate.length === 1
    && shaderById.get(immediateRecord?.vertexModuleId)?.code
      === result?.evidence?.shader?.vertexShader
    && shaderById.get(immediateRecord?.fragmentModuleId)?.code
      === result?.evidence?.shader?.fragmentShader;
  const pipelineToBundleBindingExact = immediate.length === 1
    && pipeline?.actualPipelineId === immediateRecord?.pipelineId
    && pipeline?.immediatePipelineId === immediateRecord?.pipelineId
    && pipeline?.immediateVertexModuleId === immediateRecord?.vertexModuleId
    && pipeline?.immediateFragmentModuleId === immediateRecord?.fragmentModuleId
    && exactArray(pipeline?.bundleSetPipelineIds, [immediateRecord?.pipelineId]);
  addReasonUnless(pipeline?.exact === true
    && pipeline?.exactImmediate === true
    && pipeline?.exactControl === true
    && pipeline?.immediatePipelineCount === 1
    && pipeline?.controlPipelineCount === 1
    && orderIsFourThenZero
    && classificationsAreBound
    && pipeline?.moduleSourceBindingExact === true
    && moduleSourceBindingExact
    && classificationsMatchSources
    && pipeline?.pipelineToBundleBindingExact === true
    && pipelineToBundleBindingExact,
  reasons, 'shader-module/pipeline/bundle chain or exact 4-then-0 layouts are inconsistent');
}

function validateCommandEvidence(result, reasons) {
  const commands = result?.evidence?.commands;
  const encoder = result?.evidence?.encoder;
  const expectedWordsFour = Array(DRAW_COUNT).fill(0);
  addReasonUnless(commands?.exact === true
    && commands?.drawCount === DRAW_COUNT
    && commands?.strideBytes === INDIRECT_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT
    && exactArray(commands?.indirectOffsets, EXPECTED_INDIRECT_OFFSETS)
    && exactArray(commands?.cpuCommands, EXPECTED_COMMAND_WORDS)
    && exactArray(commands?.gpuCommands, EXPECTED_COMMAND_WORDS)
    && exactArray(commands?.cpuWordsFour, expectedWordsFour)
    && exactArray(commands?.gpuWordsFour, expectedWordsFour)
    && commands?.readbackAttributeIdentityExact === true
    && typeof commands?.drawGpuBufferId === 'string'
    && commands.drawGpuBufferId.length > 0
    && commands?.encoderDrawGpuBufferIdentityExact === true
    && exactArray(commands?.encoderDrawGpuBufferIds, encoder?.drawBufferIds)
    && encoder?.drawGpuBufferId === commands.drawGpuBufferId
    && encoder?.drawBufferIdentityExact === true
    && Array.isArray(encoder?.drawBufferIds)
    && encoder.drawBufferIds.every((id) => id === commands.drawGpuBufferId),
  reasons, 'CPU/GPU indirect commands or word four do not match the fixed oracle');
}

function validateEncoderEvidence(result, reasons) {
  const encoder = result?.evidence?.encoder;
  const trace = encoder?.trace;
  const events = trace?.events;
  const expectedMethods = [
    'setPipeline',
    'setBindGroup',
    'setIndexBuffer',
    'setVertexBuffer',
    'setImmediates',
    'drawIndexedIndirect',
    'setImmediates',
    'drawIndexedIndirect',
    'setImmediates',
    'drawIndexedIndirect',
    'setImmediates',
    'drawIndexedIndirect',
    'finish',
  ];
  const methodsExact = Array.isArray(events)
    && exactArray(events.map((event) => event?.method), expectedMethods);
  const sequencesExact = Array.isArray(events)
    && events.every((event, index) => Number.isSafeInteger(event?.sequence)
      && (index === 0 || event.sequence === events[index - 1].sequence + 1));
  const immediateCalls = Array.isArray(events)
    ? events.filter((event) => event?.method === 'setImmediates')
    : [];
  const draws = Array.isArray(events)
    ? events.filter((event) => event?.method === 'drawIndexedIndirect')
    : [];
  const callsExact = immediateCalls.length === DRAW_COUNT
    && immediateCalls.every((call, index) => call?.rangeOffset === 0
      && call?.sourceType === 'Uint32Array'
      && call?.sourceElementCount === DRAW_COUNT
      && call?.dataOffset === index
      && call?.dataSize === 1
      && exactArray(call?.selectedValues, [CANONICAL_IMMEDIATE_BASES[index]]));
  const drawBufferIds = draws.map((draw) => draw?.bufferId);
  const expectedDrawGpuBufferId = result?.evidence?.commands?.drawGpuBufferId;
  const drawsExact = draws.length === DRAW_COUNT
    && exactArray(draws.map((draw) => draw?.indirectOffset), EXPECTED_INDIRECT_OFFSETS)
    && typeof drawBufferIds[0] === 'string'
    && drawBufferIds[0].length > 0
    && drawBufferIds.every((id) => id === drawBufferIds[0])
    && drawBufferIds.every((id) => id === expectedDrawGpuBufferId)
    && encoder?.drawGpuBufferId === expectedDrawGpuBufferId
    && encoder?.drawBufferIdentityExact === true
    && exactArray(encoder?.drawBufferIds, drawBufferIds);
  const adjacencyExact = draws.length === DRAW_COUNT
    && draws.every((draw, index) => {
      const position = events.indexOf(draw);
      const prior = events[position - 1];
      const reported = encoder?.adjacency?.[index];
      return prior === immediateCalls[index]
        && reported?.drawSequence === draw.sequence
        && reported?.drawOffset === EXPECTED_INDIRECT_OFFSETS[index]
        && reported?.previousMethod === 'setImmediates'
        && reported?.previousSequence === prior.sequence
        && reported?.adjacent === true
        && reported?.immediate?.rangeOffset === 0
        && reported?.immediate?.sourceType === 'Uint32Array'
        && reported?.immediate?.sourceElementCount === DRAW_COUNT
        && reported?.immediate?.dataOffset === index
        && reported?.immediate?.dataSize === 1
        && exactArray(
          reported?.immediate?.selectedValues,
          [CANONICAL_IMMEDIATE_BASES[index]],
        );
    })
    && encoder?.adjacency?.length === DRAW_COUNT;
  const mutationExact = trace?.nativeFinishReturned === true
    && trace?.sourceMutation?.phase
      === 'after-native-finish-returned-before-bundle-execution'
    && exactArray(trace?.sourceMutation?.before, CANONICAL_IMMEDIATE_BASES)
    && exactArray(trace?.sourceMutation?.after, MUTATED_IMMEDIATE_BASES);
  const identityChainExact = typeof events?.[0]?.pipelineId === 'string'
    && events[0].pipelineId === result?.evidence?.pipeline?.actualPipelineId
    && events[0].pipelineId === result?.evidence?.pipeline?.immediatePipelineId
    && typeof trace?.bundleId === 'string'
    && trace.bundleId === result?.evidence?.bundle?.cachedBundleGpuId
    && trace.bundleId === result?.evidence?.bundle?.finishBundleId;
  const patchedMethodsExact = exactArray(encoder?.patchedDeviceMethods, [
    'createShaderModule',
    'createPipelineLayout',
    'createRenderPipeline',
    'createRenderPipelineAsync',
    'createCommandEncoder',
    'createRenderBundleEncoder',
  ]);
  addReasonUnless(encoder?.exact === true
    && encoder?.exactCalls === true
    && encoder?.exactDraws === true
    && trace?.callableSetImmediates === true
    && trace?.descriptor?.sampleCount === 1
    && exactArray(trace?.descriptor?.colorFormats, ['rgba8unorm'])
    && trace?.descriptor?.depthStencilFormat === null
    && methodsExact
    && sequencesExact
    && callsExact
    && drawsExact
    && adjacencyExact
    && mutationExact
    && identityChainExact
    && patchedMethodsExact,
  reasons, 'encoder calls, draw adjacency, values, or post-finish mutation are inconsistent');
}

function validateOutputEvidence(result, reasons) {
  const output = result?.evidence?.output;
  const expectedCanonical = expectedOutputForBases(CANONICAL_IMMEDIATE_BASES);
  const expectedMutated = expectedOutputForBases(MUTATED_IMMEDIATE_BASES);
  const canonicalSha256 = sha256(Buffer.from(expectedCanonical));
  const mutatedSha256 = sha256(Buffer.from(expectedMutated));
  const first = output?.firstObserved;
  const second = output?.secondObserved;
  const firstSha256 = Array.isArray(first) ? sha256(Buffer.from(first)) : null;
  const secondSha256 = Array.isArray(second) ? sha256(Buffer.from(second)) : null;
  const firstMismatchCount = Array.isArray(first)
    ? first.reduce(
      (count, value, index) => count + Number(value !== expectedCanonical[index]),
      0,
    )
    : null;
  const secondMismatchCount = Array.isArray(second)
    ? second.reduce(
      (count, value, index) => count + Number(value !== expectedCanonical[index]),
      0,
    )
    : null;
  const challengeDifferingPixelCount = Array.from(
    { length: TARGET_WIDTH },
    (_, pixel) => Number(
      expectedCanonical[pixel * 4] !== expectedMutated[pixel * 4],
    ),
  ).reduce((sum, value) => sum + value, 0);
  addReasonUnless(output?.exact === true
    && output?.encoding
      === 'rgba8unorm-red-object-id-plus-one-at-draw-ordinal-instance-pixel'
    && output?.pixelMapping
      === 'pixel = drawOrdinal * instancesPerDraw + localInstance'
    && output?.challengeDifferingPixelCount === challengeDifferingPixelCount
    && challengeDifferingPixelCount === TARGET_WIDTH
    && exactArray(output?.expectedCanonical, expectedCanonical)
    && exactArray(output?.expectedMutated, expectedMutated)
    && exactArray(first, expectedCanonical)
    && exactArray(second, expectedCanonical)
    && !exactArray(first, expectedMutated)
    && output?.canonicalSha256 === canonicalSha256
    && output?.mutatedSha256 === mutatedSha256
    && output?.firstSha256 === firstSha256
    && output?.secondSha256 === secondSha256
    && output?.firstMismatchCount === firstMismatchCount
    && output?.secondMismatchCount === secondMismatchCount
    && firstMismatchCount === 0
    && secondMismatchCount === 0,
  reasons, 'render output bytes, hashes, or independently rebuilt oracle are inconsistent');
}

function validBundleSummary(summary) {
  return summary?.renderObjectCount === 1
    && summary?.meshIdentityExact === true
    && summary?.bundleIdentityExact === true
    && summary?.bundleVersion === 0
    && summary?.rootVersion === 0
    && summary?.bundleVersionExact === true
    && summary?.pipelineImmediateSize === 4
    && summary?.immediateBasesIdentityExact === true;
}

function validateBundleEvidence(result, reasons) {
  const bundle = result?.evidence?.bundle;
  const cachedBundleGpuId = bundle?.cachedBundleGpuId;
  const validExecution = (execution, phase) => (
    execution?.capturePhase === phase
      && Number.isSafeInteger(execution?.sequence)
      && typeof execution?.commandEncoderId === 'string'
      && typeof execution?.renderPassId === 'string'
      && execution?.bundleCount === 1
      && exactArray(execution?.bundleIds, [cachedBundleGpuId])
  );
  const firstExecutions = bundle?.firstRenderPassExecutions;
  const secondExecutions = bundle?.secondRenderPassExecutions;
  const allExecutions = bundle?.allRenderPassExecutions;
  const executionTraceExact = typeof cachedBundleGpuId === 'string'
    && cachedBundleGpuId.length > 0
    && bundle?.finishBundleId === cachedBundleGpuId
    && bundle?.renderPassExecutionExact === true
    && Array.isArray(firstExecutions)
    && firstExecutions.length === 1
    && validExecution(firstExecutions[0], 'instrumented-immediate-first')
    && Array.isArray(secondExecutions)
    && secondExecutions.length === 1
    && validExecution(secondExecutions[0], 'instrumented-immediate-second')
    && Array.isArray(allExecutions)
    && allExecutions.length === 3
    && validExecution(
      allExecutions.find(
        (execution) => execution?.capturePhase === 'instrumented-immediate-first',
      ),
      'instrumented-immediate-first',
    )
    && validExecution(
      allExecutions.find(
        (execution) => execution?.capturePhase === 'instrumented-immediate-second',
      ),
      'instrumented-immediate-second',
    );
  addReasonUnless(bundle?.exact === true
    && bundle?.bundleGroupCount === 1
    && bundle?.meshCount === 1
    && bundle?.rootStatic === true
    && validBundleSummary(bundle?.first)
    && validBundleSummary(bundle?.second)
    && bundle?.renderBundleIdentityStable === true
    && bundle?.bundleGpuIdentityStable === true
    && bundle?.encoderCountAfterFirstRender === 1
    && bundle?.encoderCountAfterSecondRender === 1
    && bundle?.postImmediateControlEncoderCount === 2
    && exactArray(bundle?.sourceBasesAfterFinish, MUTATED_IMMEDIATE_BASES)
    && executionTraceExact,
  reasons, 'bundle cache identity/reuse metadata is inconsistent');
}

function validateUninstrumentedSnapshot(result, reasons) {
  const snapshot = result?.evidence?.uninstrumentedSnapshot;
  const expectedCanonical = expectedOutputForBases(CANONICAL_IMMEDIATE_BASES);
  const expectedMutated = expectedOutputForBases(MUTATED_IMMEDIATE_BASES);
  const canonicalSha256 = sha256(Buffer.from(expectedCanonical));
  const firstSha256 = Array.isArray(snapshot?.firstObserved)
    ? sha256(Buffer.from(snapshot.firstObserved))
    : null;
  const secondSha256 = Array.isArray(snapshot?.secondObserved)
    ? sha256(Buffer.from(snapshot.secondObserved))
    : null;
  addReasonUnless(snapshot?.exact === true
    && snapshot?.deviceInstrumentationInstalledByCanary === false
    && snapshot?.disposableRenderer === true
    && snapshot?.deviceDistinctFromInstrumentedRenderer === true
    && snapshot?.mutationPhase
      === 'after-first-render-executed-before-second-cached-render'
    && exactArray(snapshot?.sourceBeforeMutation, CANONICAL_IMMEDIATE_BASES)
    && exactArray(snapshot?.sourceAfterMutation, MUTATED_IMMEDIATE_BASES)
    && exactArray(snapshot?.expectedCanonical, expectedCanonical)
    && exactArray(snapshot?.expectedMutated, expectedMutated)
    && exactArray(snapshot?.firstObserved, expectedCanonical)
    && exactArray(snapshot?.secondObserved, expectedCanonical)
    && !exactArray(snapshot?.secondObserved, expectedMutated)
    && snapshot?.canonicalSha256 === canonicalSha256
    && snapshot?.firstSha256 === firstSha256
    && snapshot?.secondSha256 === secondSha256
    && firstSha256 === canonicalSha256
    && secondSha256 === canonicalSha256
    && snapshot?.renderBundleIdentityStable === true
    && snapshot?.bundleGpuIdentityStable === true
    && validBundleSummary(snapshot?.firstBundle)
    && validBundleSummary(snapshot?.secondBundle)
    && snapshot?.adapterInfo?.isFallbackAdapter === false
    && Number(snapshot?.adapterMaxImmediateSize) >= 4
    && Number(snapshot?.deviceMaxImmediateSize) >= 4
    && snapshot?.errorObservationScope?.start
      === 'after-device-creation-before-renderer-construction-and-initialization'
    && snapshot?.errorObservationScope?.includesRendererInitialization === true
    && snapshot?.errorObservationScope?.includesBothCanaryRendersAndReadbacks === true
    && snapshot?.errorObservationScope?.excludesAdapterAndDeviceRequest === true
    && Array.isArray(snapshot?.gpuErrors?.uncaptured)
    && snapshot.gpuErrors.uncaptured.length === 0
    && Array.isArray(snapshot?.gpuErrors?.unexpectedDeviceLosses)
    && snapshot.gpuErrors.unexpectedDeviceLosses.length === 0,
  reasons, 'separate uninstrumented renderer snapshot evidence is inconsistent');
}

export function validatePageResult(
  result,
  targetKey = THREE_IMMEDIATE_OVERLAY_TARGET_R185,
) {
  const reasons = [];
  let configuration = null;
  try {
    configuration = resolveTarget(targetKey).page;
  } catch (error) {
    reasons.push(error.message);
  }
  if (result?.schemaVersion !== 1
      || result?.kind !== 'three-immediate-overlay-development-canary') {
    reasons.push('page result identity is invalid');
  }
  if (result?.status !== 'development-checks-complete') {
    reasons.push(`page status is ${result?.status ?? '<missing>'}`);
  }
  if (result?.analysisEligible !== false
      || result?.timingCaptured !== false
      || result?.efficacyEvaluated !== false
      || result?.fullPhaseZeroPass !== false
      || result?.fullPhaseZeroStatus !== 'not-evaluated') {
    reasons.push('development-only claim boundary is invalid');
  }
  if (configuration === null
      || !exactTargetIdentity(result?.target, targetIdentity(configuration))
      || result?.scope !== configuration.scope) {
    reasons.push('page target/source-family identity is not exact');
  }
  if (configuration === null
      || !exactArray(result?.coverage?.implemented, configuration.implemented)
      || !exactArray(result?.coverage?.deferred, configuration.deferred)) {
    reasons.push('coverage/deferred gate inventory is not the exact frozen inventory');
  }
  addReasonUnless(Array.isArray(result?.capabilities?.wgslLanguageFeatures)
    && result.capabilities.wgslLanguageFeatures.includes('immediate_address_space')
    && result.capabilities.immediateAddressSpace === true
    && result.capabilities.nonFallbackAdapter === true
    && Array.isArray(result.capabilities.adapterFeatures)
    && Number(result.capabilities.adapterMaxImmediateSize) >= 4
    && Number.isFinite(result.capabilities.deviceMaxImmediateSize)
    && result.capabilities.deviceMaxImmediateSize >= 4
    && result.capabilities.renderPassSetImmediates === true
    && result.capabilities.renderBundleSetImmediates === true,
  reasons, 'WebGPU immediate capability evidence is incomplete');
  addReasonUnless(result?.adapterInfo?.isFallbackAdapter === false,
    reasons, 'adapter identity is absent or does not prove non-fallback operation');
  addReasonUnless(result?.errorObservationScope?.start
      === 'after-device-creation-before-renderer-construction-and-initialization'
    && result.errorObservationScope.includesRendererInitialization === true
    && result.errorObservationScope
      .includesAllInstrumentedRendererRendersAndReadbacks === true
    && result.errorObservationScope.excludesAdapterAndDeviceRequest === true,
  reasons, 'WebGPU error observation scope is absent or overstated');
  if (configuration !== null) validateRuntimeAndTopology(result, reasons, configuration);
  validateShaderEvidence(result, reasons);
  validatePipelineEvidence(result, reasons);
  validateCommandEvidence(result, reasons);
  validateEncoderEvidence(result, reasons);
  validateOutputEvidence(result, reasons);
  validateBundleEvidence(result, reasons);
  validateUninstrumentedSnapshot(result, reasons);
  addReasonUnless(Array.isArray(result?.gpuErrors?.uncaptured)
    && result.gpuErrors.uncaptured.length === 0
    && Array.isArray(result?.gpuErrors?.unexpectedDeviceLosses)
    && result.gpuErrors.unexpectedDeviceLosses.length === 0,
  reasons, 'page reported WebGPU errors or unexpected device loss');
  addReasonUnless(result?.failure === null, reasons, 'page retained a failure');
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

async function executableEvidence(executablePath) {
  const metadata = await stat(executablePath);
  return {
    path: executablePath,
    byteLength: metadata.size,
    sha256: sha256(await readFile(executablePath)),
  };
}

export function exactAliasEvidence(
  aliasTargets,
  expectedHashes,
  observed,
  resourceUrls,
  targetKey = THREE_IMMEDIATE_OVERLAY_TARGET_R185,
) {
  const target = resolveTarget(targetKey);
  const modules = Object.fromEntries(
    Object.keys(expectedHashes).map((specifier) => [specifier, observed.get(specifier) ?? null]),
  );
  const resourceThreeUrls = resourceUrls.filter((url) => /three/i.test(url));
  const resourceThreePaths = resourceThreeUrls.map(
    (url) => new URL(url).pathname,
  ).sort();
  const overlayUrlRoot = `/${path.relative(repoRoot, target.outputRoot).replaceAll('\\', '/')}`;
  const expectedResourceThreePaths = [
    `${overlayUrlRoot}/build/three.core.js`,
    `${overlayUrlRoot}/build/three.tsl.js`,
    `${overlayUrlRoot}/build/three.webgpu.js`,
    '/src/three-immediate-overlay-canary.js',
  ].sort();
  const unexpectedNodeModulesUrls = resourceThreeUrls.filter(
    (url) => /\/node_modules\/three(?:\/|$)/i.test(new URL(url).pathname),
  );
  const exact = Object.values(modules).every((record) => record !== null)
    && Object.entries(modules).every(([specifier, record]) => (
      record?.transformInputSha256 === expectedHashes[specifier]
        && record?.transformCount === 1
    ))
    && unexpectedNodeModulesUrls.length === 0
    && exactArray(resourceThreePaths, expectedResourceThreePaths);
  return {
    exact,
    aliases: aliasTargets,
    observedModules: modules,
    expectedTransformInputSha256: expectedHashes,
    resourceThreeUrls,
    resourceThreePaths,
    expectedResourceThreePaths,
    unexpectedNodeModulesUrls,
  };
}

function exactStringRecord(value, expected) {
  return value !== null
    && typeof value === 'object'
    && !Array.isArray(value)
    && exactArray(Object.keys(value), Object.keys(expected))
    && Object.entries(expected).every(([key, expectedValue]) => (
      value[key] === expectedValue
    ));
}

export function validateThreeImmediateOverlayManifest(
  manifest,
  targetKey = THREE_IMMEDIATE_OVERLAY_TARGET_R185,
) {
  const reasons = [];
  let target = null;
  try {
    target = resolveTarget(targetKey);
  } catch (error) {
    reasons.push(error.message);
  }
  if (target === null) return { valid: false, reasons };

  addReasonUnless(
    manifest?.schemaVersion === 1
      && manifest?.overlay === target.overlayName
      && manifest?.source?.package === 'three'
      && manifest?.source?.version === target.version,
    reasons,
    'overlay manifest header/source family is not exact',
  );
  addReasonUnless(
    manifest?.contract?.symbol === 'threeImmediateDrawBase'
      && manifest?.contract?.wgslRequirement === 'immediate_address_space'
      && manifest?.contract?.immediateSizeBytes === 4
      && manifest?.contract?.geometryProperty === 'indirectImmediateBases'
      && manifest?.contract?.setter
        === 'setIndirect(indirect, indirectOffset, indirectImmediateBases)',
    reasons,
    'overlay manifest immediate-data contract is not exact',
  );

  if (targetKey === THREE_IMMEDIATE_OVERLAY_TARGET_R185) {
    addReasonUnless(
      exactStringRecord(manifest?.source?.targetSha256, EXPECTED_TARGET_HASHES),
      reasons,
      'r185.1 overlay source hashes are not the pinned inventory',
    );
    const expectedPatchedKeys = Object.keys(EXPECTED_TARGET_HASHES)
      .filter((relativePath) => relativePath !== 'package.json');
    addReasonUnless(
      exactArray(Object.keys(manifest?.patchedSha256 ?? {}), expectedPatchedKeys)
        && Object.values(manifest?.patchedSha256 ?? {}).every(
          (hash) => SHA256_PATTERN.test(hash),
        ),
      reasons,
      'r185.1 overlay patched hashes are malformed or incomplete',
    );
  } else {
    addReasonUnless(
      manifest?.source?.runtimeRevision === EXPECTED_THREE_DEV_REVISION
        && manifest?.source?.commit === EXPECTED_THREE_DEV_COMMIT
        && manifest?.source?.tree === EXPECTED_THREE_DEV_TREE
        && manifest?.source?.cleanTrackedTree === true
        && manifest?.source?.cleanFullWorktree === true
        && manifest?.source?.statusPorcelain === ''
        && exactStringRecord(
          manifest?.source?.targetCanonicalSha256,
          EXPECTED_THREE_DEV_TARGET_HASHES,
        ),
      reasons,
      'dev overlay source commit/tree/hash identity is not exact',
    );
    addReasonUnless(
      exactStringRecord(
        manifest?.transform?.patchedTargetSha256,
        EXPECTED_THREE_DEV_PATCHED_HASHES,
      ),
      reasons,
      'dev overlay patched hashes are not the pinned inventory',
    );
    addReasonUnless(
      manifest?.contract?.servedEntrypoints?.['three/webgpu']
        === 'build/three.webgpu.js'
        && manifest?.contract?.servedEntrypoints?.['three/tsl']
          === 'build/three.tsl.js'
        && exactArray(
          manifest?.contract?.unpatchedUnexportedBuilds,
          ['build/three.webgpu.nodes.js'],
        ),
      reasons,
      'dev overlay served-entrypoint boundary is not exact',
    );
  }

  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

export async function runThreeImmediateOverlayProbe({
  outputPath = null,
  browserPath = null,
  target: targetKey = THREE_IMMEDIATE_OVERLAY_TARGET_R185,
} = {}) {
  const target = resolveTarget(targetKey);
  const resolvedOutputPath = outputPath === null
    ? defaultResultPath(target)
    : path.resolve(outputPath);
  const envelope = {
    schemaVersion: 1,
    kind: THREE_IMMEDIATE_OVERLAY_RUNNER_KIND,
    status: 'development-runner-failed',
    executionMode: 'timing-free-development-correctness-canary',
    analysisEligible: false,
    timingCaptured: false,
    efficacyEvaluated: false,
    fullPhaseZeroPass: false,
    fullPhaseZeroStatus: 'not-evaluated',
    target: targetIdentity(target.page),
    threeVersion: target.version,
    sourceIdentity: null,
    overlay: null,
    servedSource: null,
    browser: null,
    browserObservations: null,
    pageResult: null,
    pageValidation: null,
    failure: null,
  };
  let server = null;
  let browser = null;
  let context = null;
  let observedModules = null;
  let sourceFilesStart = null;
  let sourceFilesEnd = null;

  try {
    sourceFilesStart = await collectBoundSourceFiles(targetKey);
    const build = await target.build();
    const overlayManifestPath = path.join(
      build.outputRoot,
      target.manifestFilename,
    );
    const overlayManifestBytes = await readFile(overlayManifestPath);
    const overlayManifest = JSON.parse(overlayManifestBytes.toString('utf8'));
    envelope.overlay = {
      root: build.outputRoot,
      manifestPath: overlayManifestPath,
      manifestSha256: sha256(overlayManifestBytes),
      manifest: overlayManifest,
      validation: validateThreeImmediateOverlayManifest(overlayManifest, targetKey),
    };

    if (!envelope.overlay.validation.valid) {
      throw new Error(
        `Generated overlay manifest validation failed: ${envelope.overlay.validation.reasons.join('; ')}`,
      );
    }

    const aliasTargets = {
      'three/webgpu': path.join(build.outputRoot, 'build', 'three.webgpu.js'),
      'three/tsl': path.join(build.outputRoot, 'build', 'three.tsl.js'),
    };
    const expectedAliasHashes = Object.fromEntries(await Promise.all(
      Object.entries(aliasTargets).map(async ([specifier, target]) => [
        specifier,
        sha256(await readFile(target)),
      ]),
    ));
    if (expectedAliasHashes['three/tsl'] !== target.expectedTslSha256) {
      throw new Error(
        `Generated three/tsl bytes do not match the pinned ${target.page.sourceFamily} hash.`,
      );
    }
    const patchedHashes = targetKey === THREE_IMMEDIATE_OVERLAY_TARGET_DEV
      ? overlayManifest.transform?.patchedTargetSha256
      : overlayManifest.patchedSha256;
    if (expectedAliasHashes['three/webgpu']
      !== patchedHashes?.['build/three.webgpu.js']) {
      throw new Error('Generated three/webgpu bytes do not match the overlay manifest.');
    }
    const coreRuntimePath = path.join(build.outputRoot, 'build', 'three.core.js');
    const coreRuntimeHash = sha256(await readFile(coreRuntimePath));
    if (coreRuntimeHash !== patchedHashes?.['build/three.core.js']) {
      throw new Error('Generated three.core runtime bytes do not match the overlay manifest.');
    }
    if (targetKey === THREE_IMMEDIATE_OVERLAY_TARGET_DEV) {
      const outputFiles = overlayManifest.output?.files;
      const exactOutputFile = (relativePath, expectedHash) => (
        Array.isArray(outputFiles)
          && outputFiles.filter((file) => file?.path === relativePath).length === 1
          && outputFiles.find((file) => file.path === relativePath)?.sha256 === expectedHash
      );
      if (!exactOutputFile('build/three.webgpu.js', expectedAliasHashes['three/webgpu'])
          || !exactOutputFile('build/three.tsl.js', expectedAliasHashes['three/tsl'])
          || !exactOutputFile('build/three.core.js', coreRuntimeHash)) {
        throw new Error('Generated dev runtime bytes do not match its output inventory.');
      }
    }
    const expectedRuntimeHashes = {
      ...expectedAliasHashes,
      'three/core-runtime': coreRuntimeHash,
    };
    observedModules = new Map();
    const observedSourceTargets = {
      ...aliasTargets,
      'three/core-runtime': coreRuntimePath,
      'canary-page': BOUND_SOURCE_FILES.find(
        (record) => record.role === 'page-module',
      ).path,
    };
    server = await createServer({
      root: repoRoot,
      configFile: false,
      appType: 'spa',
      logLevel: 'error',
      clearScreen: false,
      resolve: {
        alias: Object.entries(aliasTargets).map(([find, replacement]) => ({
          find: new RegExp(`^${find.replace('/', '\\/')}$`),
          replacement,
        })),
      },
      optimizeDeps: {
        noDiscovery: true,
        include: [],
      },
      plugins: [sourceObserverPlugin(observedSourceTargets, observedModules, targetKey)],
      server: {
        host: '127.0.0.1',
        port: 0,
        strictPort: false,
        headers: ISOLATION_HEADERS,
        fs: {
          strict: true,
          allow: [repoRoot, build.outputRoot],
        },
      },
    });
    await server.listen();
    const address = server.httpServer.address();
    if (address === null || typeof address === 'string') {
      throw new Error('Vite did not expose a numeric local port.');
    }
    const baseUrl = `http://127.0.0.1:${address.port}`;

    const executablePath = await findChrome(browserPath);
    envelope.browser = await executableEvidence(executablePath);
    browser = await chromium.launch({
      executablePath,
      headless: true,
      args: [...BROWSER_ARGS],
    });
    envelope.browser.productVersion = browser.version();
    envelope.browser.launchApi = 'playwright-core.chromium.launch';
    envelope.browser.arguments = [...BROWSER_ARGS];
    envelope.browser.disposableContext = true;

    context = await browser.newContext({
      viewport: { width: 320, height: 180 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    const browserObservations = installPageObservers(page);
    envelope.browserObservations = browserObservations;
    await page.goto(`${baseUrl}/three-immediate-canary.html`, {
      waitUntil: 'load',
      timeout: 120_000,
    });
    await page.waitForFunction(
      () => window.__THREE_IMMEDIATE_OVERLAY_CANARY__?.ready === true,
      null,
      { timeout: 120_000 },
    );
    envelope.pageResult = await page.evaluate(
      () => structuredClone(window.__THREE_IMMEDIATE_OVERLAY_CANARY__.result),
    );
    const resourceUrls = await page.evaluate(() => (
      performance.getEntriesByType('resource').map((entry) => entry.name).sort()
    ));
    sourceFilesEnd = await collectBoundSourceFiles(targetKey);
    envelope.sourceIdentity = makeBoundSourceIdentity(
      sourceFilesStart,
      sourceFilesEnd,
      observedModules.get('canary-page') ?? null,
      targetKey,
    );
    envelope.pageValidation = validatePageResult(envelope.pageResult, targetKey);
    envelope.servedSource = exactAliasEvidence(
      aliasTargets,
      expectedRuntimeHashes,
      observedModules,
      resourceUrls,
      targetKey,
    );

    if (!envelope.pageValidation.valid) {
      throw new Error(
        `Three.js overlay page validation failed: ${envelope.pageValidation.reasons.join('; ')}`,
      );
    }
    const sourceIdentityValidation = validateBoundSourceIdentity(
      envelope.sourceIdentity,
      targetKey,
    );
    if (!sourceIdentityValidation.valid) {
      throw new Error(
        `Runner/page source identity failed: ${sourceIdentityValidation.reasons.join('; ')}`,
      );
    }
    if (!envelope.servedSource.exact) {
      throw new Error('Vite served-source alias attestation failed.');
    }
    if (!pageObservationsAreClean(browserObservations)) {
      throw new Error('Browser/page observations contain errors.');
    }

    envelope.status = 'development-checks-complete';
  } catch (error) {
    envelope.failure = serializeBrowserError(error);
  } finally {
    await context?.close().catch(() => undefined);
    await browser?.close().catch(() => undefined);
    await server?.close().catch(() => undefined);
    if (sourceFilesStart !== null) {
      sourceFilesEnd = await collectBoundSourceFiles(targetKey).catch(() => null);
      envelope.sourceIdentity = makeBoundSourceIdentity(
        sourceFilesStart,
        sourceFilesEnd,
        observedModules?.get('canary-page') ?? null,
        targetKey,
      );
      const finalSourceValidation = validateBoundSourceIdentity(
        envelope.sourceIdentity,
        targetKey,
      );
      if (envelope.status === 'development-checks-complete'
        && !finalSourceValidation.valid) {
        envelope.status = 'development-runner-failed';
        envelope.failure = serializeBrowserError(new Error(
          `Runner/page source identity changed before teardown completed: ${finalSourceValidation.reasons.join('; ')}`,
        ));
      }
    }
    await mkdir(path.dirname(resolvedOutputPath), { recursive: true });
    await writeFile(resolvedOutputPath, `${JSON.stringify(envelope, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
  }

  return { outputPath: resolvedOutputPath, envelope };
}

function isDirectExecution() {
  return process.argv[1] !== undefined
    && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
}

if (isDirectExecution()) {
  const options = parseArguments(process.argv.slice(2));
  runThreeImmediateOverlayProbe(options).then(({ outputPath, envelope }) => {
    process.stdout.write(`${JSON.stringify({
      status: envelope.status,
      target: envelope.target,
      outputPath,
      pageStatus: envelope.pageResult?.status ?? null,
      servedSourceExact: envelope.servedSource?.exact ?? false,
      fullPhaseZeroStatus: envelope.fullPhaseZeroStatus,
      failure: envelope.failure,
    }, null, 2)}\n`);
    if (envelope.status !== 'development-checks-complete') process.exitCode = 1;
  }).catch((error) => {
    process.stderr.write(`${error.stack ?? error.message}\n`);
    process.exitCode = 1;
  });
}
