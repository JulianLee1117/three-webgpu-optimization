import { execFile } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import {
  access, chmod, lstat, mkdir, open, readFile, readdir, realpath, stat, writeFile,
} from 'node:fs/promises';
import { arch, platform, release, type as osType, version as osVersion } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { Euler, Matrix4, Quaternion, Vector3 } from 'three';

import {
  buildThreeImmediateOverlay, DEFAULT_OUTPUT_ROOT, DEFAULT_SOURCE_ROOT,
  EXPECTED_THREE_VERSION,
} from './build-three-immediate-overlay.mjs';
import {
  buildThreeImmediateDevOverlay, DEFAULT_THREE_DEV_OUTPUT_ROOT,
  DEFAULT_THREE_DEV_SOURCE_ROOT,
  EXPECTED_THREE_DEV_COMMIT, EXPECTED_THREE_DEV_REVISION,
  EXPECTED_THREE_DEV_TREE, EXPECTED_THREE_DEV_VERSION,
} from './build-three-immediate-dev-overlay.mjs';
import {
  THREE_IMMEDIATE_OVERLAY_TARGET_DEV, validateThreeImmediateOverlayManifest,
} from './probe-three-immediate-overlay.mjs';
import {
  assessDefaultExposureBrowserRecord, findForbiddenDefaultExposureArguments,
  rawWebGpuImmediatesArtifactModeIsReadOnly,
  runWebGpuImmediatesProbe,
  validateWebGpuImmediatesProbeReport, WEBGPU_IMMEDIATES_EXPOSURE_MODES,
} from './probe-webgpu-immediates.mjs';
import {
  collectSourceProvenance, sourceProvenanceMatches,
} from './source-provenance.mjs';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';

export const THREE_IMMEDIATE_PHASE0_RUNNER_KIND =
  'three-immediate-aif-phase0-default-exposure-runner';
export const THREE_IMMEDIATE_PHASE0_PAGE_KIND =
  'three-immediate-aif-phase0-page-result';
export const THREE_IMMEDIATE_PHASE0_TARGET_R185 = 'r185';
export const THREE_IMMEDIATE_PHASE0_TARGET_DEV = 'dev';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PAGE_PATH = path.join(repoRoot, 'src', 'phase0', 'immediate-aif-page.js');
const ENTRY_PATH = path.join(repoRoot, 'phase0-immediate-aif.html');
const PAGE_INJECTION_MARKER =
  '/* THREE_IMMEDIATE_PHASE0_TARGET_CONFIGURATION_V1 */ null';
const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const OBSERVATION_NONCE_PATTERN = /^[a-f0-9]{32}$/u;
const OBSERVATION_ENCODING = 'ordered-utf8-fields-u32be-byte-length-prefix-v1';
const OBSERVATION_DOMAIN = 'three-immediate-aif-phase0-observation-v1';
const OBSERVATION_CHAIN_DOMAIN = 'three-immediate-aif-phase0-chain-v1';
export const THREE_IMMEDIATE_PHASE0_ZERO_SIGN_POLICY =
  'json-persisted-zero-sign-canonicalized';
const OBSERVATION_FIELD_NAMES = Object.freeze([
  'domain', 'nonce', 'target', 'shaders', 'pipelines', 'commands', 'scenarios',
  'addressWitnesses', 'outputWitnesses', 'resources', 'chains', 'executionTraces',
]);
const OBSERVATION_CHAIN_FIELD_NAMES = Object.freeze([
  'domain', 'nonce', 'chainId', 'chainEvidence',
]);
const BROWSER_OPERATION_TIMEOUT_MS = 180_000;
const PAGE_READINESS_TIMEOUT_MS = 600_000;
const SERVER_OPERATION_TIMEOUT_MS = 120_000;
const STATIC_IDENTITY_TIMEOUT_MS = 30_000;
const DEFAULT_CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const NVIDIA_SMI_PATH = 'C:\\Windows\\System32\\nvidia-smi.exe';
const NVIDIA_SMI_ARGUMENTS = Object.freeze([
  '--query-gpu=index,uuid,name,driver_version,vbios_version,pci.bus_id',
  '--format=csv,noheader,nounits',
]);
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
});
const TARGET_WIDTH = 1280;
const TARGET_HEIGHT = 720;
const TARGET_BYTE_LENGTH = TARGET_WIDTH * TARGET_HEIGHT * 4;
const ADDRESS_WIDTH = 256;
const ADDRESS_HEIGHT = 256;
const ADDRESS_BYTE_LENGTH = ADDRESS_WIDTH * ADDRESS_HEIGHT * 4;
const VISIBILITY_IDS = Object.freeze(['v99', 'v20']);
const IMMEDIATE_AIF_UNUSED_ADDRESS = 0xffff_ffff;
const IMMEDIATE_AIF_PHASE0_LANES = Object.freeze(['A', 'I', 'F']);
const IMMEDIATE_AIF_PHASE0_LANE_ORDERS = deepFreeze([
  ['A', 'I', 'F'], ['A', 'F', 'I'], ['F', 'I', 'A'],
  ['I', 'A', 'F'], ['I', 'F', 'A'], ['F', 'A', 'I'],
]);
const IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS = Object.freeze(['canonical', 'S1', 'S2']);
const RUNNER_BUCKET_BASES = Object.freeze(Array.from({ length: 32 }, (_, index) => index * 2_048));
const RUNNER_BUCKET_COUNTS = Object.freeze(Array(32).fill(2_048));
const RUNNER_VISIBLE_COUNTS = deepFreeze({
  v99: [...Array(17).fill(2_028), ...Array(15).fill(2_027)],
  v20: [...Array(19).fill(410), ...Array(13).fill(409)],
});
const RUNNER_SOURCE_BUCKETS = deepFreeze({
  canonical: Array.from({ length: 32 }, (_, index) => index),
  S1: [16, ...Array.from({ length: 16 }, (_, index) => index), 18, 17,
    31, ...Array.from({ length: 12 }, (_, index) => index + 19)],
  S2: [...Array.from({ length: 16 }, (_, index) => index + 1), 0, 18, 17,
    ...Array.from({ length: 12 }, (_, index) => index + 20), 19],
});
const IMMEDIATE_AIF_PHASE0_SCHEDULES = deepFreeze(Object.fromEntries(
  IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.map((id) => [id, {
    id,
    sourceBucketByDraw: RUNNER_SOURCE_BUCKETS[id],
    sourceBaseByDraw: RUNNER_SOURCE_BUCKETS[id].map((bucket) => bucket * 2_048),
  }]),
));
const IMMEDIATE_AIF_PHASE0_WORKLOAD = deepFreeze({
  objectCount: 65_536,
  bucketCount: 32,
  geometryTier: 'medium',
  layout: 'baseline',
  seed: 0xb1ad_2026,
  bucketCapacity: 2_048,
  bucketCounts: RUNNER_BUCKET_COUNTS,
  bucketBases: RUNNER_BUCKET_BASES,
  drawCount: 32,
  indirectStrideUints: 5,
  indirectStrideBytes: 20,
  commandUint32Length: 160,
  commandByteLength: 640,
  indirectOffsets: Array.from({ length: 32 }, (_, index) => index * 20),
  visibilityLevels: [
    { id: 'v99', fraction: 0.99, expectedVisibleCount: 64_881,
      visibleCounts: RUNNER_VISIBLE_COUNTS.v99 },
    { id: 'v20', fraction: 0.2, expectedVisibleCount: 13_107,
      visibleCounts: RUNNER_VISIBLE_COUNTS.v20 },
  ],
});
const RUNNER_GEOMETRY_INDEX_COUNTS = Object.freeze([
  576, 1_320, 480, 1_200, 576, 1_320, 480, 1_200,
  576, 1_320, 480, 1_200, 576, 1_320, 480, 1_200,
  576, 1_320, 480, 1_200, 576, 1_320, 480, 1_200,
  576, 1_320, 480, 1_200, 576, 1_320, 480, 1_200,
]);
const RUNNER_GEOMETRY_VERTEX_COUNTS = Object.freeze([
  150, 273, 151, 231, 150, 273, 151, 231,
  150, 273, 151, 231, 150, 273, 151, 231,
  150, 273, 151, 231, 150, 273, 151, 231,
  150, 273, 151, 231, 150, 273, 151, 231,
]);
const RUNNER_GEOMETRY_FIRST_INDEXES = Object.freeze([
  0, 576, 1_896, 2_376, 3_576, 4_152, 5_472, 5_952,
  7_152, 7_728, 9_048, 9_528, 10_728, 11_304, 12_624, 13_104,
  14_304, 14_880, 16_200, 16_680, 17_880, 18_456, 19_776, 20_256,
  21_456, 22_032, 23_352, 23_832, 25_032, 25_608, 26_928, 27_408,
]);
const RUNNER_ADDRESS_TRIANGLE = Object.freeze([
  -0.375, -0.375,
  0.375, -0.375,
  0, 0.375,
]);
const PINNED_ADDRESS_POSITION_SHA256 = deepFreeze({
  common: '4da84079a813cac8707f686fd4dce636de7525479192e494e866feab5d8367b0',
  canonical: '447210cb0844d07d19d20cb486ae2af0bd3a6a944d2220caa252712eb1ffc1b4',
  S1: 'b334538455c3888bc97e2e894c2fd8286c5f1383d8a7428f4c2e0e134c11df57',
  S2: 'e6005ed29e674e29c6ee1637efb81dbc56c741699d17883b0aa2b85981cdc636',
});
const PINNED_GEOMETRY_SOURCE_SHA256 =
  '90c89e13cd232c29e335a42c25b669d1d7af0be33f1bfcdce83444f4d6b2268a';
const PINNED_GEOMETRY_RECORD_SHA256 = Object.freeze([
  '55675238bd1b5ae037e30c53dc93485cf93787ba6fff2da82e3836df7c6f92dc',
  'f392596e38e48de1978fc36dd5dfead7c1165656afd2d242e031128883a64690',
  '7efb77a917b2366fe68fc0555a5b6007386f5ecb53279323b6f39fb806b661bd',
  '5eacfa8bc9812a7d90af23290f14a778d2da9ad6b5cec8666f138c149bfabe75',
  '405b7e79503e31e8d6a5bfd12a45f103bfb1972a6875c2812b64eb3a148a79a3',
  '37fc5c7f3dce24378ce489257ff475c557fbd407a9ddbcf7ba2a8429b0e3bc68',
  '9f61c007ca03aefbd452ed53c0608b4c4314d347b802ff212f0626c15ce41ad5',
  'fb09ecefbe9e347505cb22e449f95e717117e5cfdb2188e71c8ede7b1338623d',
  '85d2ca8e529996b6d80c6f10db794bf675426bd72c0a5633bcc981e128062ba4',
  'bf940868c031307a3155ea018d2301a5d1812502e90372d3e553aebde20349f4',
  '627d446c223f65de9f8c18436b8bdfebdd05ef8f01d24dd8e5475ef3d3d6f48c',
  'ab9cc9240c4079466a4f04bcf1e7fbef117f381ffc3dde33f883f58dc15a4723',
  'c22e0f082e21d6533d4dfbd34f1f3c93c4207b94796211a883c16bbfa9286e03',
  '9b681bef8b09934c0d33d35a697ce272a95ae9012e1f8a45e7fc1c65c59d708c',
  '602e00fcc0280aab47c5b76790d025b3aaa63b33e51388419256eeca169e5a5a',
  '5cdcb309d13f38df7c82b79a54a53ac4787b0319d999ac81aab8c1fcb54d00c6',
  'c7c556a1472e45ce649d3d99b81c1f33fa143e319fda5ce540f871b8ceda701f',
  'c8ebcf3195530314571aa9c10eda5b121397e6abf4abc3d806d4fd3761c8a3d7',
  '3f39889d130c8859ba087342b1ff63aa5a266d771cfc58962913a9322e59a989',
  'ac9f925a9908563afebba8f3b32743dd775c869b437bcfc21dcc354c10a379b1',
  '6ee04dfe46d40ee732aa355924ee288cb1501426153a43ad821e58d5e6a15571',
  '43b902028e8299fee2157bdb34fd1febe8397c573772b5d5c171ff28856146b6',
  'f7c59fcdece4b5c239fef333893427f5a6d8230e46ab6f48ce97f6a1f207b7e8',
  '4a4a77618208bd35d8fa2e598c7d2fae5d22f76c681e8797a44c5cb8fdbc7028',
  '14ed94bb4ceae37231a49b96f332ab937606bf579772d702bf2fc08b5b552d33',
  'f946475a826a9203fd28a2a930c1f14441827a54be90e82d4f1eb5f7c98fe3d1',
  'e73710251d2e57519f2388fa1101db5f637d29a8629dbc8f0383962fe62950b1',
  '2bc85882fdb99ac0cc185f59ad30044ab76e4a84bf019df51414b1db73120835',
  'ad2d0086a6ba574a3c1925f0579f628d14793a0357d6f6a431332e52c6645d65',
  '10685b16fe2a8a407574ff1fd17f1f6346057e75386efc09a1b86964d6862dad',
  '3297b30060f92ab23c33986507628fbb74fee72b797b77e698a11f7bd4f17437',
  '8a6ce59939322a025baeb57174c782767408fc38a8e56322e11b9ab3a8ea7e64',
]);
const PINNED_SCENARIO_FINGERPRINTS = deepFreeze({
  v99: {
    sha256: '1394e9cf1b23435bf98dee8764f3b5e65619e377ebca552942b2c31fa27f19db',
    canonical: 'c7d73a0fa4d3a139d602e79406763e4c2fd5d93ea1f9bd623fde68e676809dcf',
    matrices: '3f4348a723b932592795fbfb50e4299efebba1bebff5eeb1037a8d013244e72d',
    bounds: 'd739331d88df61702dd654430661b827830dcffeb3b4338f21580cd107e77919',
    visibleCounts: '9bf7347057b9e7acab012ac33491d64933b9738507eb1f5e63a27552eb3b8aa4',
  },
  v20: {
    sha256: '2fdd12a2c5695a0f7dd451071469d6c3f9c8d0bfa9926e9db90d259ed2dc123a',
    canonical: '0b3d3bd34f066aa4b9a2b160246a10a1064d064db2e62b450eec5dadaa70961b',
    matrices: 'e61cbf0f7c4ffe3842d8f376e23c43944773264424ecd7c355cae30be4e8803c',
    bounds: '675eb73ff91515abcb86bec452ce2baaca1347b985d7e877bcd499aa4da33267',
    visibleCounts: '62bf7fe403ca3132776f6e36587c9bae494b0045c401330458cd06c702d8b83b',
  },
});
const PINNED_COMMON_SCENARIO_ARRAY_SHA256 = deepFreeze({
  bucketCounts: '75e7e5ee542781c51dda7e651d0f2624578827fc22a902f4a1cabf9997a9576d',
  bucketBases: '8873dc29d3a002e200ac115642b9bbea2ef52e6ec8dbf6c20668087ccfc4c3a9',
  objectBuckets: '35c74afc530ac4fc441ea369abf2080763e7398ddf45bacf885f7180eadf01c7',
  cullOrder: '4a35a59aabf394adb1d83cda6d3c2e799553e35ba7e4ee55537c8add209532a7',
});
const PINNED_MERGED_GEOMETRY = deepFreeze({
  vertexCount: 6_440,
  indexCount: 28_608,
  records: [
    ['bucketBase', 'vertex-attribute', 'Uint32Array', 1, 6_440, false, 35_044, 1_015,
      25_760, 'ccd6ea1e9cbb56aa531dd8663f65959b27e4b6c8417d243e06aed37778754771'],
    ['normal', 'vertex-attribute', 'Float32Array', 3, 6_440, false, 35_044, 1_015,
      77_280, '2a30d4b397b0a42ef709500eabb18e05fdaf9614a0ceb80719d52e22f56a559d'],
    ['position', 'vertex-attribute', 'Float32Array', 3, 6_440, false, 35_044, 1_015,
      77_280, 'b073217ba5db44f33762b3d6ef8563685addb3d733eb5adb98c6a72a1a349251'],
    ['uv', 'vertex-attribute', 'Float32Array', 2, 6_440, false, 35_044, 1_015,
      51_520, '72d458ef8d4fbb9cff919271ad57656c7f9523b985a70be9e7ae6b675e008586'],
    ['index', 'index', 'Uint32Array', 1, 28_608, false, 35_044, 1_015,
      114_432, '8679b053d45069dfdfeca39418b79759953eb1b168bdf877262ff800d97c8f57'],
  ].map(([semantic, resourceKind, arrayType, itemSize, count, normalized,
    usage, gpuType, byteLength, sha256Value]) => ({
    semantic, resourceKind, arrayType, itemSize, count, normalized,
    usage, gpuType, byteLength, sha256: sha256Value,
  })),
});
const PINNED_RENDER_CONFIGURATION_BYTE_LENGTH = 7_389;
const PINNED_RENDER_CONFIGURATION_SHA256 =
  'ef63b73377dbb5f65548ea06cf6a706861544e788478a2d6919c3c617ce20a64';
const PINNED_PRODUCTION_MATERIAL_LINEAR_COLOR = Object.freeze([
  0.13843161502267545, 0.38642943377667954, 0.8879231178794776,
]);
const PINNED_PRODUCTION_CLEAR_LINEAR_COLOR = Object.freeze([
  0.0009105809505882353, 0.0021246888847058823, 0.005605391621829108,
]);
const SNAPSHOT_LABELS = Object.freeze([
  'canonical-preflight', 'S1', 'S2', 'canonical-restored', 'canonical-postflight',
]);
const SNAPSHOT_SCHEDULES = Object.freeze([
  'canonical', 'S1', 'S2', 'canonical', 'canonical',
]);
const MISMATCH_KEYS = Object.freeze([
  'duplicate', 'missing', 'hidden', 'wrongBucket', 'outOfRange',
  'active', 'padding',
]);
const FORBIDDEN_EVIDENCE_KEY = /^(?:duration|durations|elapsed|timestamp|timestamps|timings?|frameTime|gpuTime|cpuTime|rank|ranking|delta|deltas|latency|throughput|fps)$/iu;
const ALLOWED_TIMING_BOUNDARY_KEYS = new Set([
  'timingCaptured', 'efficacyEvaluated', 'efficacyAnalysisAllowed',
]);
const IMPLEMENTED_GATES = Object.freeze([
  'default-exposure-raw-capability-on-exact-browser',
  'one-disposable-raw-browser-and-one-disposable-integration-browser',
  'exact-65536-object-32-draw-workload',
  'integrated-A-I-F-lanes',
  'all-six-lane-orders',
  'both-visibility-levels',
  'canonical-S1-S2-restored-postflight-schedules',
  'independent-command-membership-and-address-oracles',
  'raw-color-depth-object-id-and-address-sha256-commitments',
  'shader-module-pipeline-layout-bundle-command-identity-chain',
  'full-resource-and-lifecycle-inventory',
  'zero-WebGPU-errors-losses-page-errors-and-browser-crashes',
  'timing-free-correctness-only-boundary',
]);

function deepFreeze(value) {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) deepFreeze(nested);
    Object.freeze(value);
  }
  return value;
}

const RUNNER_NORMAL_ERROR_SCOPE_DRAINAGE = deepFreeze({
  schemaVersion: 1,
  kind: 'immediate-aif-phase0-error-scope-drainage',
  pass: true,
  applicable: true,
  trigger: 'normal-completion',
  pushOrder: ['validation', 'internal', 'out-of-memory'],
  expectedPopOrder: ['out-of-memory', 'internal', 'validation'],
  popOrder: ['out-of-memory', 'internal', 'validation'],
  records: [
    { filter: 'out-of-memory', settled: true, returnedError: null, popFailure: null },
    { filter: 'internal', settled: true, returnedError: null, popFailure: null },
    { filter: 'validation', settled: true, returnedError: null, popFailure: null },
  ],
  scopedErrors: {
    'out-of-memory': null,
    internal: null,
    validation: null,
  },
  scopedErrorsComplete: true,
  scopedErrorsEmpty: true,
});

const TARGETS = Object.freeze({
  r185: Object.freeze({
    key: 'r185', sourceFamily: 'installed-three-r185.1-package',
    expectedRevision: '185', version: EXPECTED_THREE_VERSION,
    scope: 'full timing-free Three.js r185.1 immediate-data A/I/F Phase 0 canary',
    outputRoot: DEFAULT_OUTPUT_ROOT, manifestFilename: 'THREE_IMMEDIATE_OVERLAY.json',
    resultDirectory: 'three-immediate-aif-phase0-r185', build: buildThreeImmediateOverlay,
  }),
  dev: Object.freeze({
    key: 'dev', sourceFamily: 'pinned-upstream-three-dev-commit',
    expectedRevision: EXPECTED_THREE_DEV_REVISION, version: EXPECTED_THREE_DEV_VERSION,
    scope: 'full timing-free pinned Three.js dev immediate-data A/I/F Phase 0 canary',
    outputRoot: DEFAULT_THREE_DEV_OUTPUT_ROOT,
    manifestFilename: 'THREE_IMMEDIATE_DEV_OVERLAY.json',
    resultDirectory: 'three-immediate-aif-phase0-dev', build: buildThreeImmediateDevOverlay,
  }),
});

const BOUND_SOURCE_PATHS = Object.freeze([
  ['runner', fileURLToPath(import.meta.url)],
  ['package-manifest', path.join(repoRoot, 'package.json')],
  ['package-lock', path.join(repoRoot, 'package-lock.json')],
  ['entry-html', ENTRY_PATH], ['page-module', PAGE_PATH],
  ['page-config', path.join(repoRoot, 'src', 'config.js')],
  ['page-contract', path.join(repoRoot, 'src', 'phase0', 'immediate-aif-page-contract.js')],
  ['phase0-plan', path.join(repoRoot, 'src', 'phase0', 'immediate-aif-plan.js')],
  ['phase0-workload-validation', path.join(repoRoot, 'src', 'phase0', 'immediate-aif-workload-validation.js')],
  ['phase0-runtime', path.join(repoRoot, 'src', 'strategies', 'immediate-aif-phase0.js')],
  ['fixed-slice-runtime', path.join(repoRoot, 'src', 'strategies', 'fixed-slice.js')],
  ['live-compute-capture', path.join(repoRoot, 'src', 'strategies', 'live-first-instance-crossover.js')],
  ['runtime-vertex-introspection', path.join(repoRoot, 'src', 'strategies', 'first-instance-crossover.js')],
  ['resource-cleanup', path.join(repoRoot, 'src', 'strategies', 'resources.js')],
  ['storage-transform-material', path.join(repoRoot, 'src', 'materials', 'storage-transform.js')],
  ['frustum-planes', path.join(repoRoot, 'src', 'culling', 'frustum-planes.js')],
  ['indexed-command-layout', path.join(repoRoot, 'src', 'culling', 'indexed-command-layout.js')],
  ['first-instance-crossover-plan', path.join(repoRoot, 'src', 'culling', 'first-instance-crossover.js')],
  ['benchmark-plan', path.join(repoRoot, 'src', 'benchmark', 'plan.js')],
  ['gpu-timestamp-runtime', path.join(repoRoot, 'src', 'benchmark', 'gpu-timestamps.js')],
  ['indexed-bucket-geometry', path.join(repoRoot, 'src', 'render', 'indexed-bucket-geometry.js')],
  ['membership-validator', path.join(repoRoot, 'src', 'validation', 'membership.js')],
  ['membership-digests', path.join(repoRoot, 'src', 'validation', 'membership-digests.js')],
  ['render-shader-normalizer', path.join(repoRoot, 'src', 'validation', 'first-instance-shader-evidence.js')],
  ['compute-shader-normalizer', path.join(repoRoot, 'src', 'validation', 'live-compute-shader-normalization.js')],
  ['address-challenge-oracle', path.join(repoRoot, 'src', 'validation', 'first-instance-address-challenge.js')],
  ['geometry-fixtures', path.join(repoRoot, 'src', 'scenes', 'geometry-fixtures.js')],
  ['geometry-fingerprints', path.join(repoRoot, 'src', 'scenes', 'geometry-fingerprints.js')],
  ['fixed-subsets', path.join(repoRoot, 'src', 'scenes', 'fixed-subsets.js')],
  ['random-generator', path.join(repoRoot, 'src', 'lib', 'random.js')],
  ['r185-overlay-builder', path.join(repoRoot, 'scripts', 'build-three-immediate-overlay.mjs')],
  ['dev-overlay-builder', path.join(repoRoot, 'scripts', 'build-three-immediate-dev-overlay.mjs')],
  ['overlay-manifest-validator', path.join(repoRoot, 'scripts', 'probe-three-immediate-overlay.mjs')],
  ['raw-capability-runner', path.join(repoRoot, 'scripts', 'probe-webgpu-immediates.mjs')],
  ['source-provenance', path.join(repoRoot, 'scripts', 'source-provenance.mjs')],
]);

const execFileAsync = promisify(execFile);

function resolveTarget(key) {
  const target = TARGETS[key];
  if (!target) throw new Error(`Unknown Phase 0 target ${JSON.stringify(key)}.`);
  return target;
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function sha256U32(values) {
  const bytes = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => bytes.writeUInt32LE(value >>> 0, index * 4));
  return sha256(bytes);
}

function sha256View(values) {
  if (!ArrayBuffer.isView(values)) throw new TypeError('Expected a typed-array view.');
  return sha256(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
}

export function commitThreeImmediatePhase0ObservationFields(fieldNames, fieldValues) {
  if (!Array.isArray(fieldNames) || !Array.isArray(fieldValues)
    || fieldNames.length === 0 || fieldNames.length !== fieldValues.length
    || !fieldNames.every((name) => typeof name === 'string' && name.length > 0)
    || !fieldValues.every((value) => typeof value === 'string')) {
    throw new TypeError('Observation fields must be matching nonempty string arrays.');
  }
  const encoded = fieldValues.map((value) => Buffer.from(value, 'utf8'));
  const framed = Buffer.alloc(encoded.reduce((sum, bytes) => sum + 4 + bytes.length, 0));
  let offset = 0;
  for (const bytes of encoded) {
    framed.writeUInt32BE(bytes.length, offset);
    offset += 4;
    bytes.copy(framed, offset);
    offset += bytes.length;
  }
  return {
    encoding: OBSERVATION_ENCODING,
    fieldNames: [...fieldNames],
    fieldCount: fieldNames.length,
    fieldCommitments: encoded.map((bytes, index) => ({
      name: fieldNames[index], byteLength: bytes.length, sha256: sha256(bytes),
    })),
    sha256: sha256(framed),
  };
}

function exactArray(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

export function equalThreeImmediatePhase0PersistedNumberArrays(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => (
      typeof value === 'number' && typeof right[index] === 'number'
        && value === 0 && right[index] === 0
        ? true
        : Object.is(value, right[index])
    ));
}

function exactJson(left, right) {
  try {
    const leftJson = JSON.stringify(left);
    const rightJson = JSON.stringify(right);
    return typeof leftJson === 'string' && typeof rightJson === 'string'
      && leftJson === rightJson;
  } catch {
    return false;
  }
}

function isStrictJsonData(value, ancestors = new Set()) {
  try {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') {
      return true;
    }
    if (typeof value === 'number') return Number.isFinite(value);
    if (typeof value !== 'object' || ancestors.has(value)) return false;
    const prototype = Object.getPrototypeOf(value);
    if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) {
      return false;
    }
    const keys = Reflect.ownKeys(value);
    if (keys.some((key) => typeof key !== 'string')) return false;
    if (Array.isArray(value)
      && (keys.some((key) => key !== 'length' && !/^(?:0|[1-9][0-9]*)$/u.test(key))
        || keys.filter((key) => key !== 'length').length !== value.length)) {
      return false;
    }
    ancestors.add(value);
    const exact = keys.every((key) => {
      if (Array.isArray(value) && key === 'length') return true;
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      return descriptor?.enumerable === true && 'value' in descriptor
        && isStrictJsonData(descriptor.value, ancestors);
    });
    ancestors.delete(value);
    return exact;
  } catch {
    return false;
  }
}

function addReasonUnless(condition, reasons, message) {
  if (!condition) reasons.push(message);
}

function serializeError(error) {
  const serialized = {
    name: String(error?.name ?? 'Error').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 8_192),
    stack: error?.stack == null ? null : String(error.stack).slice(0, 16_384),
  };
  if (typeof error?.pageStatus === 'string') {
    serialized.pageStatus = error.pageStatus.slice(0, 256);
  }
  if (isStrictJsonData(error?.pageFailure)
    && error.pageFailure !== null && typeof error.pageFailure === 'object') {
    serialized.pageFailure = structuredClone(error.pageFailure);
  }
  return serialized;
}

function normalizePath(value) {
  return value.replaceAll('\\', '/');
}

function withinRoot(filename, root) {
  const relative = path.relative(root, filename);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`)
    && !path.isAbsolute(relative);
}

function targetIdentity(configuration) {
  return {
    schemaVersion: 1, key: configuration.key,
    sourceFamily: configuration.sourceFamily,
    expectedRevision: configuration.expectedRevision,
  };
}

function rawCapabilityIdentity(rawReport) {
  const page = rawReport.pageResult;
  return deepFreeze({
    schemaVersion: 1,
    browserVersion: rawReport.browser.browserVersion,
    browserExecutableSha256: rawReport.browser.executableStart.sha256,
    adapterInfo: structuredClone(page.adapterInfo),
    wgslLanguageFeatures: [...page.capabilities.wgslLanguageFeatures],
    adapterFeatures: [...page.adapterFeatures],
    requestedDeviceFeatures: [...page.requestedDeviceFeatures],
    deviceFeatures: [...page.deviceFeatures],
    adapterMaxImmediateSize: page.capabilities.adapterMaxImmediateSize,
    deviceMaxImmediateSize: page.capabilities.deviceMaxImmediateSize,
    renderPassSetImmediates: page.capabilities.renderPassSetImmediates,
    renderBundleSetImmediates: page.capabilities.renderBundleSetImmediates,
    indirectFirstInstanceAdapter: page.capabilities.indirectFirstInstanceAdapter,
    indirectFirstInstanceDevice: page.capabilities.indirectFirstInstanceDevice,
  });
}

function configurationFor(targetKey, rawReport, observationNonce) {
  const target = resolveTarget(targetKey);
  if (!OBSERVATION_NONCE_PATTERN.test(observationNonce ?? '')) {
    throw new Error('Phase 0 observation nonce must be exactly 128 bits of lowercase hex.');
  }
  return deepFreeze({
    schemaVersion: 1, key: target.key, sourceFamily: target.sourceFamily,
    expectedRevision: target.expectedRevision, scope: target.scope,
    implemented: [...IMPLEMENTED_GATES], deferred: [],
    rawExpectation: rawCapabilityIdentity(rawReport),
    observationNonce,
  });
}

function injectedPageSource(source, configuration) {
  const count = source.split(PAGE_INJECTION_MARKER).length - 1;
  if (count !== 1) throw new Error(`Expected one Phase 0 injection marker; found ${count}.`);
  return source.replace(
    PAGE_INJECTION_MARKER,
    `Object.freeze(${JSON.stringify(configuration)})`,
  );
}

export function injectThreeImmediatePhase0Target(source, configuration) {
  return injectedPageSource(source, deepFreeze(structuredClone(configuration)));
}

function scanForbiddenEvidenceKeys(value, pointer = '$', findings = [], seen = new Set()) {
  if (value === null || typeof value !== 'object' || seen.has(value)) return findings;
  seen.add(value);
  for (const [key, nested] of Object.entries(value)) {
    const next = `${pointer}.${key}`;
    if (FORBIDDEN_EVIDENCE_KEY.test(key) && !ALLOWED_TIMING_BOUNDARY_KEYS.has(key)) {
      findings.push(next);
    }
    scanForbiddenEvidenceKeys(nested, next, findings, seen);
  }
  return findings;
}

export function findForbiddenPhase0PerformanceEvidence(value) {
  return scanForbiddenEvidenceKeys(value);
}

function runnerMulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function createRunnerLocalScenario(geometrySpheres, scenarioId) {
  const visibleCounts = Uint32Array.from(RUNNER_VISIBLE_COUNTS[scenarioId]);
  const expectedVisibleCount = visibleCounts.reduce((sum, count) => sum + count, 0);
  const expectedVisibleIds = new Uint32Array(expectedVisibleCount);
  const objectBuckets = new Uint32Array(65_536);
  const matrices = new Float32Array(65_536 * 16);
  const bounds = new Float32Array(65_536 * 4);
  const random = runnerMulberry32(0xb1ad_2026);
  const position = new Vector3();
  const scale = new Vector3();
  const quaternion = new Quaternion();
  const euler = new Euler();
  const matrix = new Matrix4();
  const worldCenter = new Vector3();
  let visibleCursor = 0;
  for (let bucket = 0; bucket < 32; bucket += 1) {
    const base = bucket * 2_048;
    const visibleCount = visibleCounts[bucket];
    const localSphere = geometrySpheres[bucket];
    for (let localIndex = 0; localIndex < 2_048; localIndex += 1) {
      const objectId = base + localIndex;
      const visible = localIndex < visibleCount;
      objectBuckets[objectId] = bucket;
      const randomX = random() * 2 - 1;
      const randomY = random() * 2 - 1;
      const inViewZ = (random() * 2 - 1) * 28;
      position.set(
        visible ? randomX * 30 : 10_000 + objectId * 0.01,
        randomY * 18,
        inViewZ,
      );
      euler.set(random() * Math.PI * 2, random() * Math.PI * 2, random() * Math.PI * 2);
      quaternion.setFromEuler(euler);
      scale.set(0.55 + random() * 0.9, 0.55 + random() * 0.9, 0.55 + random() * 0.9);
      matrix.compose(position, quaternion, scale);
      matrix.toArray(matrices, objectId * 16);
      worldCenter.copy(localSphere.center).applyMatrix4(matrix);
      const offset = objectId * 4;
      bounds[offset] = worldCenter.x;
      bounds[offset + 1] = worldCenter.y;
      bounds[offset + 2] = worldCenter.z;
      bounds[offset + 3] = localSphere.radius * matrix.getMaxScaleOnAxis();
      if (visible) expectedVisibleIds[visibleCursor++] = objectId;
    }
  }
  if (visibleCursor !== expectedVisibleCount) {
    throw new Error(`Runner-local ${scenarioId} visible-ID construction is incomplete.`);
  }
  return {
    scenarioId,
    objectCount: 65_536,
    bucketCount: 32,
    visibilityFraction: scenarioId === 'v99' ? 0.99 : 0.2,
    layout: 'baseline',
    seed: 0xb1ad_2026,
    geometryTier: 'medium',
    bucketCounts: Uint32Array.from(RUNNER_BUCKET_COUNTS),
    bucketBases: Uint32Array.from(RUNNER_BUCKET_BASES),
    visibleCounts,
    expectedVisibleCount,
    expectedVisibleIds,
    objectBuckets,
    matrices,
    bounds,
    cullOrder: Uint32Array.from({ length: 65_536 }, (_, index) => index),
  };
}

function createRunnerLocalCommandOracle(lane, scenario, scheduleId) {
  const sourceBaseByDraw = Uint32Array.from(
    RUNNER_SOURCE_BUCKETS[scheduleId], (bucket) => bucket * 2_048,
  );
  const initialCommands = new Uint32Array(160);
  const postCullCommands = new Uint32Array(160);
  for (let draw = 0; draw < 32; draw += 1) {
    const offset = draw * 5;
    for (const commands of [initialCommands, postCullCommands]) {
      commands[offset] = RUNNER_GEOMETRY_INDEX_COUNTS[draw];
      commands[offset + 2] = RUNNER_GEOMETRY_FIRST_INDEXES[draw];
      commands[offset + 3] = 0;
      commands[offset + 4] = lane === 'F' ? sourceBaseByDraw[draw] : 0;
    }
    postCullCommands[offset + 1] = scenario.visibleCounts[draw];
  }
  return { sourceBaseByDraw, initialCommands, postCullCommands };
}

function buildCpuOracles() {
  const geometries = createIndexedGeometryFixtures(32, 'medium');
  try {
    if (!exactArray(geometries.map((geometry) => geometry.index.count),
      RUNNER_GEOMETRY_INDEX_COUNTS)) {
      throw new Error('Generated medium geometry index counts differ from reviewed constants.');
    }
    const geometrySpheres = geometries.map((geometry) => geometry.boundingSphere.clone());
    const scenarios = Object.fromEntries(VISIBILITY_IDS.map(
      (scenarioId) => [scenarioId, createRunnerLocalScenario(geometrySpheres, scenarioId)],
    ));
    const result = {};
    for (const scenarioId of VISIBILITY_IDS) {
      const scenario = scenarios[scenarioId];
      const pins = PINNED_SCENARIO_FINGERPRINTS[scenarioId];
      if (sha256View(scenario.matrices) !== pins.matrices
        || sha256View(scenario.bounds) !== pins.bounds
        || sha256View(scenario.visibleCounts) !== pins.visibleCounts
        || sha256View(scenario.expectedVisibleIds) !== pins.canonical
        || sha256View(scenario.bucketCounts) !== PINNED_COMMON_SCENARIO_ARRAY_SHA256.bucketCounts
        || sha256View(scenario.bucketBases) !== PINNED_COMMON_SCENARIO_ARRAY_SHA256.bucketBases
        || sha256View(scenario.objectBuckets) !== PINNED_COMMON_SCENARIO_ARRAY_SHA256.objectBuckets
        || sha256View(scenario.cullOrder) !== PINNED_COMMON_SCENARIO_ARRAY_SHA256.cullOrder) {
        throw new Error(`Runner-local ${scenarioId} scenario differs from pinned reviewed bytes.`);
      }
      const expectedActiveIds = Array.from(scenario.expectedVisibleIds);
      const activeSet = new Set(expectedActiveIds);
      const canonicalPacking = new Array(65_536).fill(IMMEDIATE_AIF_UNUSED_ADDRESS);
      const cursors = new Array(32).fill(0);
      for (const objectId of expectedActiveIds) {
        const bucket = scenario.objectBuckets[objectId];
        canonicalPacking[scenario.bucketBases[bucket] + cursors[bucket]] = objectId;
        cursors[bucket] += 1;
      }
      const schedules = {};
      for (const scheduleId of IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS) {
        schedules[scheduleId] = {};
        for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
          const command = createRunnerLocalCommandOracle(lane, scenario, scheduleId);
          schedules[scheduleId][lane] = {
            initialCommandWords: [...command.initialCommands],
            initialCommandSha256: sha256U32(command.initialCommands),
            commandWords: [...command.postCullCommands],
            commandSha256: sha256U32(command.postCullCommands),
            sourceBaseByDraw: [...command.sourceBaseByDraw],
            sourceBaseSha256: sha256View(command.sourceBaseByDraw),
          };
        }
      }
      result[scenarioId] = {
        expectedVisibleCount: scenario.expectedVisibleCount,
        expectedActiveIds,
        activeSet,
        canonicalPacking,
        canonicalPackingSha256: sha256U32(canonicalPacking),
        bucketBases: [...scenario.bucketBases],
        bucketCounts: [...scenario.bucketCounts],
        visibleCounts: [...scenario.visibleCounts],
        objectBuckets: [...scenario.objectBuckets],
        sharedResources: Object.freeze(Object.fromEntries([
          ['matrix', scenario.matrices],
          ['bounds', scenario.bounds],
          ['objectBucket', scenario.objectBuckets],
          ['bucketBase', scenario.bucketBases],
          ['bucketCapacity', scenario.bucketCounts],
          ['cullOrder', scenario.cullOrder],
          ['visibleIds', Uint32Array.from(canonicalPacking)],
          ['overflow', Uint32Array.of(0)],
        ].map(([semantic, values]) => [semantic, Object.freeze({
          semantic,
          arrayType: values.constructor.name,
          byteLength: values.byteLength,
          expectedBytes: Uint8Array.from(
            new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
          ),
          sha256: sha256View(values),
          oracleKind: semantic === 'overflow' ? 'protocol-zero-overflow'
            : semantic === 'visibleIds' ? 'frozen-cpu-canonical-visible-ids'
              : 'deterministic-production-attribute-array',
          cpuSourceSha256: semantic === 'overflow'
            ? sha256View(Uint32Array.of(IMMEDIATE_AIF_UNUSED_ADDRESS))
            : sha256View(values),
          cpuSourceMatchesOracle: semantic !== 'overflow',
        })]))),
        schedules,
      };
    }
    return result;
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
}

let cachedCpuOracles;
function cpuOracles() {
  cachedCpuOracles ??= buildCpuOracles();
  return cachedCpuOracles;
}

function buildRunnerMergedGeometryByteOracles() {
  // createIndexedGeometryFixtures is used only as a deterministic byte source.
  // Every source record, the merged dimensions, and every resulting merged byte
  // digest are pinned below; changing that generator cannot redefine acceptance.
  const geometries = createIndexedGeometryFixtures(32, 'medium');
  try {
    const vertexCount = geometries.reduce(
      (sum, geometry) => sum + geometry.getAttribute('position').count, 0,
    );
    const indexCount = geometries.reduce(
      (sum, geometry) => sum + geometry.index.count, 0,
    );
    if (vertexCount !== PINNED_MERGED_GEOMETRY.vertexCount
      || indexCount !== PINNED_MERGED_GEOMETRY.indexCount) {
      throw new Error('Runner-local merged geometry dimensions differ from reviewed constants.');
    }
    const arrays = {};
    for (const name of ['normal', 'position', 'uv']) {
      const first = geometries[0].getAttribute(name);
      const values = new first.array.constructor(geometries.reduce(
        (sum, geometry) => sum + geometry.getAttribute(name).array.length, 0,
      ));
      let cursor = 0;
      for (const geometry of geometries) {
        const source = geometry.getAttribute(name).array;
        values.set(source, cursor);
        cursor += source.length;
      }
      arrays[name] = values;
    }
    arrays.bucketBase = new Uint32Array(vertexCount);
    let vertexCursor = 0;
    for (let bucket = 0; bucket < geometries.length; bucket += 1) {
      const count = geometries[bucket].getAttribute('position').count;
      arrays.bucketBase.fill(RUNNER_BUCKET_BASES[bucket], vertexCursor, vertexCursor + count);
      vertexCursor += count;
    }
    arrays.index = new Uint32Array(indexCount);
    let indexCursor = 0;
    vertexCursor = 0;
    for (const geometry of geometries) {
      for (const value of geometry.index.array) {
        arrays.index[indexCursor] = value + vertexCursor;
        indexCursor += 1;
      }
      vertexCursor += geometry.getAttribute('position').count;
    }
    const records = Object.fromEntries(PINNED_MERGED_GEOMETRY.records.map((record) => {
      const values = arrays[record.semantic];
      if (values?.constructor?.name !== record.arrayType
        || values.byteLength !== record.byteLength
        || sha256View(values) !== record.sha256) {
        throw new Error(
          `Runner-local merged ${record.semantic} bytes differ from reviewed constants.`,
        );
      }
      return [record.semantic, Uint8Array.from(
        new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      )];
    }));
    return records;
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
}

let cachedRunnerMergedGeometryByteOracles;
function runnerMergedGeometryByteOracles() {
  cachedRunnerMergedGeometryByteOracles ??= buildRunnerMergedGeometryByteOracles();
  return cachedRunnerMergedGeometryByteOracles;
}

function buildRunnerAddressPositionOracles() {
  const indexBytes = runnerMergedGeometryByteOracles().index;
  const index = new Uint32Array(
    indexBytes.buffer, indexBytes.byteOffset, indexBytes.byteLength / 4,
  );
  const spans = [];
  let vertexCursor = 0;
  for (let bucket = 0; bucket < RUNNER_GEOMETRY_VERTEX_COUNTS.length; bucket += 1) {
    const count = RUNNER_GEOMETRY_VERTEX_COUNTS[bucket];
    spans.push({ bucket, start: vertexCursor, end: vertexCursor + count });
    vertexCursor += count;
  }
  if (vertexCursor !== PINNED_MERGED_GEOMETRY.vertexCount) {
    throw new Error('Runner address-position vertex spans differ from reviewed constants.');
  }
  const build = (scheduleId, feature) => {
    const positions = new Float32Array(PINNED_MERGED_GEOMETRY.vertexCount * 3);
    const sourceBases = IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId].sourceBaseByDraw;
    for (const span of spans) {
      const targetBase = RUNNER_BUCKET_BASES[span.bucket];
      const z = feature ? targetBase - sourceBases[span.bucket] : targetBase;
      for (let vertex = span.start; vertex < span.end; vertex += 1) {
        positions[vertex * 3 + 2] = z;
      }
      const firstIndex = RUNNER_GEOMETRY_FIRST_INDEXES[span.bucket];
      const triangle = [index[firstIndex], index[firstIndex + 1], index[firstIndex + 2]];
      if (new Set(triangle).size !== 3
        || !triangle.every((vertex) => vertex >= span.start && vertex < span.end)) {
        throw new Error(`Runner address triangle ${span.bucket} is not independently addressable.`);
      }
      for (let corner = 0; corner < 3; corner += 1) {
        positions[triangle[corner] * 3] = RUNNER_ADDRESS_TRIANGLE[corner * 2];
        positions[triangle[corner] * 3 + 1] = RUNNER_ADDRESS_TRIANGLE[corner * 2 + 1];
      }
    }
    const hash = sha256View(positions);
    const expectedHash = feature
      ? PINNED_ADDRESS_POSITION_SHA256[scheduleId]
      : PINNED_ADDRESS_POSITION_SHA256.common;
    if (positions.byteLength !== 77_280 || hash !== expectedHash) {
      throw new Error(`Runner ${scheduleId}/${feature ? 'feature' : 'common'} address positions differ from reviewed bytes.`);
    }
    return positions;
  };
  return {
    common: build('canonical', false),
    feature: Object.fromEntries(IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.map(
      (scheduleId) => [scheduleId, build(scheduleId, true)],
    )),
  };
}

let cachedRunnerAddressPositionOracles;
function runnerAddressPositionOracles() {
  cachedRunnerAddressPositionOracles ??= buildRunnerAddressPositionOracles();
  return cachedRunnerAddressPositionOracles;
}

export function createThreeImmediatePhase0ExpectedCommandWords(
  scenarioId, scheduleId, lane, { initial = false } = {},
) {
  const record = cpuOracles()?.[scenarioId]?.schedules?.[scheduleId]?.[lane];
  if (record === undefined) throw new Error('Unknown Phase 0 command-oracle identity.');
  return Uint32Array.from(initial ? record.initialCommandWords : record.commandWords);
}

const GEOMETRY_FINGERPRINT_ORACLES = deepFreeze({
  source: {
    sha256: PINNED_GEOMETRY_SOURCE_SHA256,
    geometries: Array.from({ length: 32 }, (_, bucket) => ({
      bucket,
      family: bucket % 4,
      name: `controlled-family-${bucket % 4}-bucket-${bucket}`,
      attributes: {
        normal: { count: RUNNER_GEOMETRY_VERTEX_COUNTS[bucket] },
        position: { count: RUNNER_GEOMETRY_VERTEX_COUNTS[bucket] },
        uv: { count: RUNNER_GEOMETRY_VERTEX_COUNTS[bucket] },
      },
      index: { count: RUNNER_GEOMETRY_INDEX_COUNTS[bucket] },
      sha256: PINNED_GEOMETRY_RECORD_SHA256[bucket],
    })),
  },
  scenarios: PINNED_SCENARIO_FINGERPRINTS,
  merged: PINNED_MERGED_GEOMETRY,
});

function validateFrozenMembership(values, scenarioId) {
  const oracle = cpuOracles()[scenarioId];
  if (!Array.isArray(values) || values.length !== 65_536) return false;
  const seen = new Set();
  for (let bucket = 0; bucket < 32; bucket += 1) {
    const base = oracle.bucketBases[bucket];
    const visibleCount = oracle.visibleCounts[bucket];
    const count = oracle.bucketCounts[bucket];
    for (let slot = 0; slot < visibleCount; slot += 1) {
      const objectId = values[base + slot];
      if (!oracle.activeSet.has(objectId) || oracle.objectBuckets[objectId] !== bucket
        || seen.has(objectId)) return false;
      seen.add(objectId);
    }
    for (let slot = visibleCount; slot < count; slot += 1) {
      if (values[base + slot] !== IMMEDIATE_AIF_UNUSED_ADDRESS) return false;
    }
  }
  return seen.size === oracle.expectedVisibleCount
    && oracle.expectedActiveIds.every((objectId) => seen.has(objectId));
}

function expectedAddressBytes(frozenVisibleIds, scenarioId, scheduleId) {
  const oracle = cpuOracles()[scenarioId];
  const sourceBuckets = IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId].sourceBucketByDraw;
  const bytes = Buffer.alloc(ADDRESS_BYTE_LENGTH);
  for (let draw = 0; draw < 32; draw += 1) {
    const targetBase = oracle.bucketBases[draw];
    const sourceBase = oracle.bucketBases[sourceBuckets[draw]];
    for (let local = 0; local < oracle.visibleCounts[draw]; local += 1) {
      const encoded = frozenVisibleIds[sourceBase + local] + 1;
      const offset = (targetBase + local) * 4;
      bytes[offset] = encoded & 0xff;
      bytes[offset + 1] = (encoded >>> 8) & 0xff;
      bytes[offset + 2] = (encoded >>> 16) & 0xff;
      bytes[offset + 3] = 0xff;
    }
  }
  return bytes;
}

function expectedAddressHash(frozenVisibleIds, scenarioId, scheduleId) {
  return sha256(expectedAddressBytes(frozenVisibleIds, scenarioId, scheduleId));
}

function pinnedRenderConfigurationExact(configuration) {
  try {
    if (!isStrictJsonData(configuration)) return false;
    const json = JSON.stringify(configuration);
    if (typeof json !== 'string') return false;
    const bytes = Buffer.from(json);
    return bytes.byteLength === PINNED_RENDER_CONFIGURATION_BYTE_LENGTH
      && sha256(bytes) === PINNED_RENDER_CONFIGURATION_SHA256;
  } catch {
    return false;
  }
}

export function validateThreeImmediatePhase0PinnedRenderConfiguration(configuration) {
  return pinnedRenderConfigurationExact(configuration);
}

export function createThreeImmediatePhase0ExpectedAddressWitness(
  scenarioId, scheduleId,
) {
  const visibleIds = cpuOracles()?.[scenarioId]?.canonicalPacking;
  if (visibleIds === undefined || !IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.includes(scheduleId)) {
    throw new Error('Expected a frozen Phase 0 scenario and schedule identifier.');
  }
  return Uint8Array.from(expectedAddressBytes(visibleIds, scenarioId, scheduleId));
}

function validatePlanEvidence(plan, reasons) {
  addReasonUnless(exactArray(plan?.lanes, IMMEDIATE_AIF_PHASE0_LANES)
    && exactJson(plan?.laneOrders, IMMEDIATE_AIF_PHASE0_LANE_ORDERS)
    && exactArray(plan?.scheduleIds, IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS)
    && exactArray(plan?.snapshotPlan?.map((entry) => entry.label), SNAPSHOT_LABELS)
    && exactArray(plan?.snapshotPlan?.map((entry) => entry.scheduleId), SNAPSHOT_SCHEDULES)
    && plan?.workload?.objectCount === 65_536
    && plan?.workload?.bucketCount === 32
    && plan?.workload?.bucketCapacity === 2_048
    && plan?.workload?.drawCount === 32
    && plan?.workload?.seed === 0xb1ad_2026
    && plan?.workload?.geometryTier === 'medium'
    && plan?.workload?.layout === 'baseline'
    && exactArray(plan?.workload?.bucketCounts, RUNNER_BUCKET_COUNTS)
    && exactArray(plan?.workload?.bucketBases, RUNNER_BUCKET_BASES)
    && plan?.workload?.commandUint32Length === 160
    && plan?.workload?.commandByteLength === 640
    && exactArray(plan?.workload?.indirectOffsets,
      IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets)
    && exactJson(plan?.workload?.visibilityLevels,
      IMMEDIATE_AIF_PHASE0_WORKLOAD.visibilityLevels)
    && plan?.addressTarget?.width === 256 && plan?.addressTarget?.height === 256
    && plan?.addressTarget?.byteLength === 262_144
    && plan?.addressTarget?.encoding
      === 'rgb24-object-id-plus-one-transparent-zero-background'
    && pinnedRenderConfigurationExact(plan?.renderConfiguration),
  reasons, 'frozen Phase 0 plan evidence is not exact');
}

// The browser page produces a second WGSL audit through its own validation
// modules. These runner-owned parsers are intentionally independent: artifact
// acceptance never calls the page's shader normalizers.
const RUNNER_WGSL_IDENTIFIER = '[A-Za-z_][A-Za-z0-9_]*';
const RUNNER_GENERATED_BUFFER_IDENTIFIER = /^NodeBuffer_[0-9]+$/u;
const RUNNER_COMPUTE_CANONICAL_VARIABLE = 'LiveLaneIndirectCommands';
const RUNNER_COMPUTE_CANONICAL_STRUCT = `${RUNNER_COMPUTE_CANONICAL_VARIABLE}Struct`;

function runnerWgslRequire(condition, message) {
  if (!condition) throw new Error(message);
}

function runnerEscapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

function runnerCountMatches(source, pattern) {
  const flags = pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`;
  return [...source.matchAll(new RegExp(pattern.source, flags))].length;
}

function runnerCountIdentifier(source, identifier) {
  return source.match(new RegExp(`\\b${runnerEscapeRegExp(identifier)}\\b`, 'gu'))
    ?.length ?? 0;
}

function runnerReplaceIdentifier(source, identifier, replacement) {
  return source.replace(
    new RegExp(`\\b${runnerEscapeRegExp(identifier)}\\b`, 'gu'),
    replacement,
  );
}

function runnerExactStructBody(source, identifier) {
  const declarations = [...source.matchAll(new RegExp(
    `\\bstruct\\s+${runnerEscapeRegExp(identifier)}\\s*\\{([^{}]*)\\}\\s*;`,
    'gu',
  ))];
  runnerWgslRequire(declarations.length === 1,
    `${identifier} must have exactly one non-nested WGSL struct declaration`);
  return declarations[0][1].replace(/\s+/gu, '').replace(/,$/u, '');
}

function runnerCommentIdentifierCount(source, identifier) {
  const comments = source.match(/\/\/[^\r\n]*|\/\*[\s\S]*?\*\//gu) ?? [];
  return comments.reduce(
    (count, comment) => count + runnerCountIdentifier(comment, identifier), 0,
  );
}

export function normalizeThreeImmediatePhase0ComputeShader(source, bindings) {
  runnerWgslRequire(typeof source === 'string' && source.length > 0,
    'a nonempty raw compute WGSL source is required');
  runnerWgslRequire(Array.isArray(bindings), 'runtime compute bindings must be an array');
  const candidates = bindings.filter((binding) => binding?.semantic === 'indirectCommands');
  runnerWgslRequire(candidates.length === 1 && candidates[0]?.kind === 'storage-buffer',
    'exactly one indirectCommands storage binding is required');
  const metadata = candidates[0];
  runnerWgslRequire(metadata.access === 'readWrite'
    && metadata.attributeType === 'IndirectStorageBufferAttribute'
    && metadata.arrayType === 'Uint32Array'
    && metadata.itemSize === 5 && metadata.count === 32 && metadata.byteLength === 640,
  'indirectCommands runtime binding shape is not the frozen Phase 0 shape');
  runnerWgslRequire(Number.isSafeInteger(metadata.group) && metadata.group >= 0
    && Number.isSafeInteger(metadata.binding) && metadata.binding >= 0,
  'indirectCommands coordinates are invalid');
  const declarationPattern = new RegExp(
    `@binding\\s*\\(\\s*${metadata.binding}\\s*\\)\\s*`
      + `@group\\s*\\(\\s*${metadata.group}\\s*\\)\\s*`
      + `var\\s*<\\s*storage\\s*,\\s*read_write\\s*>\\s*`
      + `(${RUNNER_WGSL_IDENTIFIER})\\s*:\\s*(${RUNNER_WGSL_IDENTIFIER})\\s*;`,
    'gu',
  );
  const declarations = [...source.matchAll(declarationPattern)];
  runnerWgslRequire(declarations.length === 1,
    'indirectCommands coordinate does not resolve exactly one storage declaration');
  const variableIdentifier = declarations[0][1];
  const structIdentifier = declarations[0][2];
  runnerWgslRequire(RUNNER_GENERATED_BUFFER_IDENTIFIER.test(variableIdentifier)
    && structIdentifier === `${variableIdentifier}Struct`,
  'indirectCommands generated identifiers are malformed');
  runnerWgslRequire(!source.includes(RUNNER_COMPUTE_CANONICAL_VARIABLE)
    && !source.includes(RUNNER_COMPUTE_CANONICAL_STRUCT),
  'raw compute source contains a reserved runner canonical identifier');
  const wrapperBody = runnerExactStructBody(source, structIdentifier);
  const wrapper = wrapperBody.match(new RegExp(
    `^value:array<(${RUNNER_WGSL_IDENTIFIER})>$`, 'u',
  ));
  runnerWgslRequire(wrapper !== null && wrapper[1] === 'FixedSliceIndexedDraw',
    'indirectCommands wrapper is not array<FixedSliceIndexedDraw>');
  const commandStructIdentifier = wrapper[1];
  runnerWgslRequire(runnerExactStructBody(source, commandStructIdentifier)
    === 'indexCount:u32,instanceCount:atomic<u32>,firstIndex:u32,baseVertex:i32,firstInstance:u32',
  'indexed-indirect command struct is not exact');
  const variableTokenCount = runnerCountIdentifier(source, variableIdentifier);
  const structTokenCount = runnerCountIdentifier(source, structIdentifier);
  const commentIdentifierTokenCount = runnerCommentIdentifierCount(source, variableIdentifier)
    + runnerCommentIdentifierCount(source, structIdentifier);
  runnerWgslRequire(variableTokenCount === 2 && structTokenCount === 2
    && commentIdentifierTokenCount === 0,
  'indirectCommands generated token inventory is malformed');
  const normalizedShader = runnerReplaceIdentifier(
    runnerReplaceIdentifier(source, structIdentifier, RUNNER_COMPUTE_CANONICAL_STRUCT),
    variableIdentifier,
    RUNNER_COMPUTE_CANONICAL_VARIABLE,
  );
  return {
    normalizedShader,
    audit: {
      schemaVersion: 1,
      kind: 'live-indirect-command-wgsl-identifier-normalization',
      pass: true,
      semantic: 'indirectCommands',
      group: metadata.group,
      binding: metadata.binding,
      generatedVariableIdentifier: variableIdentifier,
      generatedStructIdentifier: structIdentifier,
      canonicalVariableIdentifier: RUNNER_COMPUTE_CANONICAL_VARIABLE,
      canonicalStructIdentifier: RUNNER_COMPUTE_CANONICAL_STRUCT,
      commandStructIdentifier,
      commandStructFields: [
        'indexCount:u32',
        'instanceCount:atomic<u32>',
        'firstIndex:u32',
        'baseVertex:i32',
        'firstInstance:u32',
      ],
      variableTokenCount,
      structTokenCount,
      commentIdentifierTokenCount,
    },
  };
}

const RUNNER_RENDER_SEMANTICS = Object.freeze(['matrix', 'visibleIds']);
const RUNNER_RENDER_CANONICAL_VARIABLES = deepFreeze({
  matrix: '__FIRST_INSTANCE_MATRIX_BUFFER__',
  visibleIds: '__FIRST_INSTANCE_VISIBLE_IDS_BUFFER__',
});
const RUNNER_RENDER_CANONICAL_STRUCTS = deepFreeze({
  matrix: '__FIRST_INSTANCE_MATRIX_BUFFER_STRUCT__',
  visibleIds: '__FIRST_INSTANCE_VISIBLE_IDS_BUFFER_STRUCT__',
});
const RUNNER_RENDER_ELEMENT_TYPES = deepFreeze({
  matrix: 'mat4x4<f32>', visibleIds: 'u32',
});
const RUNNER_RENDER_ELEMENT_BYTES = deepFreeze({ matrix: 64, visibleIds: 4 });
const RUNNER_VERTEX_INPUTS = deepFreeze({
  A: [
    { kind: 'builtin', builtin: 'instance_index', name: 'instanceIndex', wgslType: 'u32' },
    { kind: 'location', location: 0, name: 'position', wgslType: 'vec3<f32>' },
    { kind: 'location', location: 1, name: 'bucketBase', wgslType: 'u32' },
    { kind: 'location', location: 2, name: 'normal', wgslType: 'vec3<f32>' },
  ],
  feature: [
    { kind: 'builtin', builtin: 'instance_index', name: 'instanceIndex', wgslType: 'u32' },
    { kind: 'location', location: 0, name: 'position', wgslType: 'vec3<f32>' },
    { kind: 'location', location: 1, name: 'normal', wgslType: 'vec3<f32>' },
  ],
});
const RUNNER_RUNTIME_VERTEX_INPUTS = deepFreeze({
  A: [
    {
      name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
      arrayType: 'Float32Array', itemSize: 3, normalized: false,
    },
    {
      name: 'bucketBase', shaderLocation: 1, format: 'uint32', stepMode: 'vertex',
      arrayType: 'Uint32Array', itemSize: 1, normalized: false,
    },
    {
      name: 'normal', shaderLocation: 2, format: 'float32x3', stepMode: 'vertex',
      arrayType: 'Float32Array', itemSize: 3, normalized: false,
    },
  ],
  feature: [
    {
      name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
      arrayType: 'Float32Array', itemSize: 3, normalized: false,
    },
    {
      name: 'normal', shaderLocation: 1, format: 'float32x3', stepMode: 'vertex',
      arrayType: 'Float32Array', itemSize: 3, normalized: false,
    },
  ],
});

function runnerNormalizedWgslType(value) {
  return typeof value === 'string' ? value.replace(/\s+/gu, '') : null;
}

function runnerFindMatchingDelimiter(source, start, open, close) {
  let depth = 0;
  for (let index = start; index < source.length; index += 1) {
    if (source[index] === open) depth += 1;
    else if (source[index] === close) {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function runnerParseVertexInputs(source) {
  const entries = [...source.matchAll(/@vertex\s+fn\s+main\s*\(/gu)];
  runnerWgslRequire(entries.length === 1, 'vertex WGSL must have exactly one main entry point');
  const open = source.indexOf('(', entries[0].index);
  const close = runnerFindMatchingDelimiter(source, open, '(', ')');
  runnerWgslRequire(close > open, 'vertex WGSL main parameters are malformed');
  const parameters = source.slice(open + 1, close);
  const pattern = /@(builtin|location)\s*\(\s*([^)]+?)\s*\)\s*([A-Za-z_]\w*)\s*:\s*([^,\n)]+)/gu;
  const parsed = [];
  for (const match of parameters.matchAll(pattern)) {
    const type = runnerNormalizedWgslType(match[4]);
    if (match[1] === 'builtin') {
      parsed.push({ kind: 'builtin', builtin: match[2].trim(), name: match[3], wgslType: type });
    } else {
      runnerWgslRequire(/^\d+$/u.test(match[2].trim()), 'vertex location is not an integer');
      parsed.push({
        kind: 'location', location: Number(match[2].trim()), name: match[3], wgslType: type,
      });
    }
  }
  runnerWgslRequire(parameters.replace(pattern, '').replaceAll(',', '').trim() === '',
    'vertex WGSL contains an unsupported main input');
  return parsed;
}

function runnerResourceIdentityExact(value) {
  return (typeof value === 'string' && value.length > 0)
    || (typeof value === 'number' && Number.isFinite(value));
}

function runnerValidateRuntimeVertexInputs(laneMode, inputs) {
  const expected = RUNNER_RUNTIME_VERTEX_INPUTS[laneMode];
  runnerWgslRequire(Array.isArray(inputs) && inputs.length === expected.length,
    `${laneMode} runtime vertex input count is not exact`);
  const sanitized = inputs.map((input, index) => {
    const fixed = expected[index];
    runnerWgslRequire(input && typeof input === 'object'
      && Object.entries(fixed).every(([key, value]) => input[key] === value)
      && Number.isSafeInteger(input.count) && input.count > 0
      && runnerResourceIdentityExact(input.resourceId),
    `${laneMode} runtime vertex input ${index} is malformed`);
    return { ...fixed, count: input.count, resourceId: input.resourceId };
  });
  runnerWgslRequire(sanitized.every((input) => input.count === sanitized[0].count),
    `${laneMode} runtime vertex input counts differ`);
  return sanitized;
}

function runnerValidateStorageMetadata(laneId, bindings) {
  runnerWgslRequire(Array.isArray(bindings) && bindings.length === 2,
    `${laneId} must expose exactly two vertex storage bindings`);
  const bySemantic = new Map();
  for (const binding of bindings) {
    runnerWgslRequire(binding && typeof binding === 'object'
      && RUNNER_RENDER_SEMANTICS.includes(binding.semantic)
      && !bySemantic.has(binding.semantic)
      && Number.isSafeInteger(binding.group) && binding.group >= 0
      && Number.isSafeInteger(binding.binding) && binding.binding >= 0
      && binding.access === 'read'
      && (binding.visibility === 'vertex' || binding.visibility === 1)
      && runnerNormalizedWgslType(binding.elementType)
        === RUNNER_RENDER_ELEMENT_TYPES[binding.semantic]
      && Number.isSafeInteger(binding.count) && binding.count > 0
      && binding.byteLength === binding.count * RUNNER_RENDER_ELEMENT_BYTES[binding.semantic]
      && runnerResourceIdentityExact(binding.resourceId),
    `${laneId} storage binding metadata is malformed`);
    bySemantic.set(binding.semantic, {
      semantic: binding.semantic,
      group: binding.group,
      binding: binding.binding,
      access: 'read',
      visibility: 'vertex',
      elementType: RUNNER_RENDER_ELEMENT_TYPES[binding.semantic],
      count: binding.count,
      byteLength: binding.byteLength,
      resourceId: binding.resourceId,
    });
  }
  runnerWgslRequire(RUNNER_RENDER_SEMANTICS.every((semantic) => bySemantic.has(semantic)),
    `${laneId} storage semantic inventory is incomplete`);
  const matrix = bySemantic.get('matrix');
  const visible = bySemantic.get('visibleIds');
  runnerWgslRequire(matrix.group === visible.group && matrix.binding < visible.binding
    && matrix.count === visible.count,
  `${laneId} matrix/visibleIds storage relationship is malformed`);
  return bySemantic;
}

function runnerParseStorageDeclarations(source) {
  const pattern = /((?:@(binding|group)\s*\(\s*\d+\s*\)\s*){2})var\s*<\s*storage\s*,\s*(read|read_write)\s*>\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)\s*;/gu;
  const declarations = [];
  for (const match of source.matchAll(pattern)) {
    const coordinates = {};
    for (const annotation of match[1].matchAll(/@(binding|group)\s*\(\s*(\d+)\s*\)/gu)) {
      runnerWgslRequire(coordinates[annotation[1]] === undefined,
        'storage declaration repeats a coordinate annotation');
      coordinates[annotation[1]] = Number(annotation[2]);
    }
    runnerWgslRequire(Number.isSafeInteger(coordinates.group)
      && Number.isSafeInteger(coordinates.binding),
    'storage declaration omits a coordinate');
    declarations.push({
      group: coordinates.group,
      binding: coordinates.binding,
      access: match[3],
      variableName: match[4],
      structName: match[5],
    });
  }
  return declarations;
}

function runnerStorageElementType(source, structName) {
  const pattern = new RegExp(
    `struct\\s+${runnerEscapeRegExp(structName)}\\s*\\{([\\s\\S]*?)\\}\\s*;`, 'gu',
  );
  const matches = [...source.matchAll(pattern)];
  runnerWgslRequire(matches.length === 1,
    `storage struct ${structName} is not declared exactly once`);
  const field = /^\s*value\s*:\s*array\s*<([\s\S]+)>\s*,?\s*$/u.exec(matches[0][1]);
  runnerWgslRequire(field !== null, `${structName} is not a single array wrapper`);
  return runnerNormalizedWgslType(field[1]);
}

function runnerMapStorageIdentifiers(laneId, source, metadata) {
  const declarations = runnerParseStorageDeclarations(source);
  runnerWgslRequire(declarations.length === metadata.size,
    `${laneId} storage declaration count differs from runtime metadata`);
  const byCoordinate = new Map();
  for (const declaration of declarations) {
    const key = `${declaration.group}:${declaration.binding}`;
    runnerWgslRequire(!byCoordinate.has(key), `${laneId} duplicates storage coordinate ${key}`);
    byCoordinate.set(key, declaration);
  }
  let normalized = source;
  const mappings = [];
  for (const semantic of RUNNER_RENDER_SEMANTICS) {
    const expected = metadata.get(semantic);
    const declaration = byCoordinate.get(`${expected.group}:${expected.binding}`);
    runnerWgslRequire(declaration !== undefined && declaration.access === expected.access,
      `${laneId} ${semantic} WGSL binding does not match metadata`);
    runnerWgslRequire(RUNNER_GENERATED_BUFFER_IDENTIFIER.test(declaration.variableName)
      && declaration.structName === `${declaration.variableName}Struct`,
    `${laneId} ${semantic} generated identifier is malformed`);
    const elementType = runnerStorageElementType(source, declaration.structName);
    runnerWgslRequire(elementType === expected.elementType
      && runnerCountIdentifier(source, declaration.variableName) === 2
      && runnerCountIdentifier(source, declaration.structName) === 2
      && runnerCommentIdentifierCount(source, declaration.variableName) === 0
      && runnerCommentIdentifierCount(source, declaration.structName) === 0,
    `${laneId} ${semantic} storage source inventory is malformed`);
    mappings.push({
      semantic,
      group: expected.group,
      binding: expected.binding,
      access: expected.access,
      elementType,
      variableName: declaration.variableName,
      structName: declaration.structName,
    });
    normalized = runnerReplaceIdentifier(
      runnerReplaceIdentifier(
        normalized, declaration.structName, RUNNER_RENDER_CANONICAL_STRUCTS[semantic],
      ),
      declaration.variableName,
      RUNNER_RENDER_CANONICAL_VARIABLES[semantic],
    );
  }
  return { normalized, mappings };
}

function runnerReplaceUnique(source, pattern, replacement, message) {
  runnerWgslRequire(runnerCountMatches(source, pattern) === 1, message);
  return source.replace(pattern, replacement);
}

function runnerNormalizeAddress(laneMode, source) {
  const matrix = runnerEscapeRegExp(RUNNER_RENDER_CANONICAL_VARIABLES.matrix);
  const visible = runnerEscapeRegExp(RUNNER_RENDER_CANONICAL_VARIABLES.visibleIds);
  const expectedIndex = laneMode === 'A'
    ? '\\(\\s*bucketBase\\s*\\+\\s*instanceIndex\\s*\\)'
    : 'instanceIndex';
  const full = new RegExp(
    `${matrix}\\s*\\.\\s*value\\s*\\[\\s*${visible}`
      + `\\s*\\.\\s*value\\s*\\[\\s*${expectedIndex}\\s*\\]\\s*\\]`,
    'gu',
  );
  runnerWgslRequire(runnerCountMatches(source, full) === 1,
    `${laneMode} matrix/visibleIds address expression is not exact`);
  return runnerReplaceUnique(
    source,
    new RegExp(
      `(${visible}\\s*\\.\\s*value\\s*\\[\\s*)${expectedIndex}(\\s*\\])`, 'gu',
    ),
    '$1__FIRST_INSTANCE_ADDRESS_INDEX__$2',
    `${laneMode} visibleIds address expression is not unique`,
  );
}

function runnerNormalizeImmediateContrast(source) {
  const requirement = /requires\s+immediate_address_space\s*;/gu;
  const declaration = /var\s*<\s*immediate\s*>\s+threeImmediateDrawBase\s*:\s*u32\s*;/gu;
  const injection = /requires\s+immediate_address_space\s*;\r?\n\r?\n\/\/ immediate data\r?\nvar\s*<\s*immediate\s*>\s+threeImmediateDrawBase\s*:\s*u32\s*;\r?\n/gu;
  const parenthesized = /\(\s*threeImmediateDrawBase\s*\+\s*instanceIndex\s*\)/gu;
  const bare = /threeImmediateDrawBase\s*\+\s*instanceIndex/gu;
  const requirementCount = runnerCountMatches(source, requirement);
  const declarationCount = runnerCountMatches(source, declaration);
  const totalImmediateDeclarationCount = runnerCountMatches(
    source, /var\s*<\s*immediate\s*>\s+[A-Za-z_]\w*\s*:/gu,
  );
  const injectionCount = runnerCountMatches(source, injection);
  const parenthesizedAddressCount = runnerCountMatches(source, parenthesized);
  const addressExpressionCount = runnerCountMatches(source, bare);
  const variableTokenCount = runnerCountIdentifier(source, 'threeImmediateDrawBase');
  runnerWgslRequire(requirementCount === 1 && declarationCount === 1
    && totalImmediateDeclarationCount === 1 && injectionCount === 1
    && addressExpressionCount === 1 && variableTokenCount === 2,
  'I immediate-address-space declaration/use inventory is not exact');
  let normalized = source.replace(injection, '');
  if (parenthesizedAddressCount === 1) {
    normalized = normalized.replace(parenthesized, 'instanceIndex');
  } else {
    runnerWgslRequire(parenthesizedAddressCount === 0,
      'I has multiple parenthesized immediate address expressions');
    normalized = normalized.replace(bare, 'instanceIndex');
  }
  runnerWgslRequire(runnerCountIdentifier(normalized, 'threeImmediateDrawBase') === 0
    && runnerCountIdentifier(normalized, 'immediate_address_space') === 0,
  'I normalization left immediate-address-space tokens');
  return {
    normalized,
    audit: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-immediate-wgsl-normalization',
      pass: true,
      requirementCount,
      declarationCount,
      totalImmediateDeclarationCount,
      injectionCount,
      addressExpressionCount,
      parenthesizedAddressCount,
      variableTokenCount,
      replacement: 'instanceIndex',
    },
  };
}

function runnerAuditRenderLane(laneId, laneMode, originalLane, source) {
  runnerWgslRequire(originalLane && typeof originalLane === 'object'
    && typeof source === 'string' && source.length > 0
    && typeof originalLane.fragmentShader === 'string' && originalLane.fragmentShader.length > 0,
  `${laneId} render shader source is absent`);
  const instanceIndex = runnerCountIdentifier(source, 'instanceIndex');
  const bucketBase = runnerCountIdentifier(source, 'bucketBase');
  runnerWgslRequire(instanceIndex === 2 && bucketBase === (laneMode === 'A' ? 2 : 0),
    `${laneId} address identifier occurrence counts are not exact`);
  runnerWgslRequire(exactJson(runnerParseVertexInputs(source), RUNNER_VERTEX_INPUTS[laneMode]),
    `${laneId} WGSL vertex inputs are not the frozen shape`);
  const runtimeVertexInputs = runnerValidateRuntimeVertexInputs(
    laneMode, originalLane.vertexInputs,
  );
  const storageMetadata = runnerValidateStorageMetadata(laneId, originalLane.storageBindings);
  const mapped = runnerMapStorageIdentifiers(laneId, source, storageMetadata);
  let normalizedVertexShader = runnerNormalizeAddress(laneMode, mapped.normalized);
  if (laneMode === 'A') {
    normalizedVertexShader = runnerReplaceUnique(
      normalizedVertexShader,
      /@location\s*\(\s*1\s*\)\s*bucketBase\s*:\s*u32\s*,\s*/gu,
      '',
      'A bucketBase entry input is not unique',
    );
    normalizedVertexShader = runnerReplaceUnique(
      normalizedVertexShader,
      /@location\s*\(\s*2\s*\)\s*normal\s*:\s*vec3\s*<\s*f32\s*>/gu,
      (match) => match.replace(/\(\s*2\s*\)/u, '( 1 )'),
      'A normal input location is not uniquely shiftable',
    );
  }
  runnerWgslRequire(runnerCountIdentifier(normalizedVertexShader, 'bucketBase') === 0
    && runnerCountIdentifier(normalizedVertexShader, 'instanceIndex') === 1,
  `${laneId} normalization changed an unapproved input/address token`);
  return {
    laneId,
    source,
    fragmentShader: originalLane.fragmentShader,
    normalizedVertexShader,
    runtimeVertexInputs,
    storageBindings: Object.fromEntries(
      [...storageMetadata.entries()].map(([semantic, binding]) => [semantic, binding]),
    ),
    semanticMappings: mapped.mappings,
    occurrenceCounts: { instanceIndex, bucketBase },
  };
}

function runnerValidateCrossLaneRenderIdentity(portable, feature) {
  for (const name of ['position', 'normal']) {
    const left = portable.runtimeVertexInputs.find((input) => input.name === name);
    const right = feature.runtimeVertexInputs.find((input) => input.name === name);
    runnerWgslRequire(left.resourceId === right.resourceId && left.count === right.count,
      `${name} runtime resource differs across render lanes`);
  }
  const bucket = portable.runtimeVertexInputs.find((input) => input.name === 'bucketBase');
  runnerWgslRequire(bucket.count === portable.runtimeVertexInputs[0].count,
    'A bucketBase input is not expanded to the common vertex count');
  for (const semantic of RUNNER_RENDER_SEMANTICS) {
    const left = portable.storageBindings[semantic];
    const right = feature.storageBindings[semantic];
    runnerWgslRequire([
      'group', 'binding', 'access', 'visibility', 'elementType', 'count', 'byteLength',
      'resourceId',
    ].every((property) => left[property] === right[property]),
    `${semantic} storage resource differs across render lanes`);
  }
}

export function normalizeThreeImmediatePhase0RenderShaders(lanes = {}) {
  const reasons = [];
  const audited = {};
  let immediate = null;
  try {
    runnerWgslRequire(lanes && typeof lanes === 'object' && !Array.isArray(lanes),
      'render lanes must be an object');
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      runnerWgslRequire(lanes[lane] && typeof lanes[lane] === 'object',
        `render lane ${lane} is absent`);
    }
    runnerWgslRequire(runnerCountIdentifier(lanes.A.vertexShader, 'immediate_address_space') === 0
      && runnerCountIdentifier(lanes.A.vertexShader, 'threeImmediateDrawBase') === 0
      && runnerCountMatches(lanes.A.vertexShader, /var\s*<\s*immediate\s*>/gu) === 0,
    'A unexpectedly contains immediate-address-space code');
    runnerWgslRequire(runnerCountIdentifier(lanes.F.vertexShader, 'immediate_address_space') === 0
      && runnerCountIdentifier(lanes.F.vertexShader, 'threeImmediateDrawBase') === 0
      && runnerCountMatches(lanes.F.vertexShader, /var\s*<\s*immediate\s*>/gu) === 0,
    'F unexpectedly contains immediate-address-space code');
    audited.A = runnerAuditRenderLane('A', 'A', lanes.A, lanes.A.vertexShader);
    immediate = runnerNormalizeImmediateContrast(lanes.I.vertexShader);
    audited.I = runnerAuditRenderLane('I', 'feature', lanes.I, immediate.normalized);
    audited.F = runnerAuditRenderLane('F', 'feature', lanes.F, lanes.F.vertexShader);
    runnerValidateCrossLaneRenderIdentity(audited.A, audited.I);
    runnerValidateCrossLaneRenderIdentity(audited.A, audited.F);
  } catch (error) {
    reasons.push(error.message);
  }
  const records = {};
  if (IMMEDIATE_AIF_PHASE0_LANES.every((lane) => audited[lane] !== undefined)) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      records[lane] = {
        laneId: lane,
        addressMode: lane === 'A' ? 'bucket-base'
          : lane === 'I' ? 'immediate-base' : 'indirect-first-instance',
        rawVertexShader: lanes[lane].vertexShader,
        rawFragmentShader: lanes[lane].fragmentShader,
        normalizedVertexShader: audited[lane].normalizedVertexShader,
        vertexInputs: audited[lane].runtimeVertexInputs,
        storageBindings: audited[lane].storageBindings,
        semanticMappings: audited[lane].semanticMappings,
        occurrenceCounts: audited[lane].occurrenceCounts,
        immediateNormalization: lane === 'I' ? immediate.audit : null,
      };
    }
    const normalizedVertexEqual = IMMEDIATE_AIF_PHASE0_LANES.every(
      (lane) => records[lane].normalizedVertexShader === records.A.normalizedVertexShader,
    );
    const rawFragmentEqual = IMMEDIATE_AIF_PHASE0_LANES.every(
      (lane) => records[lane].rawFragmentShader === records.A.rawFragmentShader,
    );
    const rawVertexPairwiseDifferent = new Set(IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => records[lane].rawVertexShader,
    )).size === 3;
    if (!normalizedVertexEqual) {
      reasons.push('normalized render WGSL differs outside approved transports');
    }
    if (!rawFragmentEqual) reasons.push('render fragment WGSL is not byte-identical');
    if (!rawVertexPairwiseDifferent) {
      reasons.push('raw render WGSL does not expose three distinct transports');
    }
    return {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-render-shader-normalization',
      pass: reasons.length === 0,
      reasons,
      lanes: records,
      comparison: { normalizedVertexEqual, rawFragmentEqual, rawVertexPairwiseDifferent },
    };
  }
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-render-shader-normalization',
    pass: false,
    reasons,
    lanes: records,
    comparison: null,
  };
}

const RUNNER_DIAGNOSTIC_VERTEX_INPUTS = deepFreeze({
  A: [
    { kind: 'builtin', builtin: 'instance_index', name: 'instanceIndex', wgslType: 'u32' },
    { kind: 'location', location: 0, name: 'position', wgslType: 'vec3<f32>' },
    { kind: 'location', location: 1, name: 'bucketBase', wgslType: 'u32' },
  ],
  I: [
    { kind: 'builtin', builtin: 'instance_index', name: 'instanceIndex', wgslType: 'u32' },
    { kind: 'location', location: 0, name: 'position', wgslType: 'vec3<f32>' },
  ],
  F: [
    { kind: 'builtin', builtin: 'instance_index', name: 'instanceIndex', wgslType: 'u32' },
    { kind: 'location', location: 0, name: 'position', wgslType: 'vec3<f32>' },
  ],
});

function runnerDiagnosticStorageMetadata(purpose, lane, bindings) {
  const semantics = purpose === 'address' ? ['visibleIds'] : ['matrix', 'visibleIds'];
  runnerWgslRequire(Array.isArray(bindings) && bindings.length === semantics.length,
    `${purpose}/${lane} diagnostic storage inventory is not exact`);
  const bySemantic = new Map();
  for (const binding of bindings) {
    runnerWgslRequire(binding && typeof binding === 'object'
      && semantics.includes(binding.semantic) && !bySemantic.has(binding.semantic)
      && Number.isSafeInteger(binding.group) && binding.group >= 0
      && Number.isSafeInteger(binding.binding) && binding.binding >= 0
      && binding.access === 'read'
      && (binding.visibility === 'vertex' || binding.visibility === 1)
      && runnerNormalizedWgslType(binding.elementType)
        === RUNNER_RENDER_ELEMENT_TYPES[binding.semantic]
      && binding.count === IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount
      && binding.byteLength
        === IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount
          * RUNNER_RENDER_ELEMENT_BYTES[binding.semantic]
      && runnerResourceIdentityExact(binding.resourceId),
    `${purpose}/${lane} diagnostic ${binding?.semantic ?? '<unknown>'} metadata is invalid`);
    bySemantic.set(binding.semantic, binding);
  }
  runnerWgslRequire(exactArray(bindings.map((binding) => binding.semantic), semantics)
    && semantics.every((semantic) => bySemantic.has(semantic)),
  `${purpose}/${lane} diagnostic storage semantics/order is invalid`);
  return { semantics, bySemantic };
}

function runnerNormalizeDiagnosticStorageIdentifiers(
  purpose, lane, source, semantics, metadata,
) {
  const declarations = runnerParseStorageDeclarations(source);
  runnerWgslRequire(declarations.length === semantics.length,
    `${purpose}/${lane} diagnostic WGSL storage declaration count is invalid`);
  const byCoordinate = new Map();
  for (const declaration of declarations) {
    const key = `${declaration.group}:${declaration.binding}`;
    runnerWgslRequire(!byCoordinate.has(key),
      `${purpose}/${lane} duplicates diagnostic storage coordinate ${key}`);
    byCoordinate.set(key, declaration);
  }
  let normalized = source;
  const mappings = [];
  for (const semantic of semantics) {
    const expected = metadata.get(semantic);
    const declaration = byCoordinate.get(`${expected.group}:${expected.binding}`);
    runnerWgslRequire(declaration !== undefined && declaration.access === 'read'
      && RUNNER_GENERATED_BUFFER_IDENTIFIER.test(declaration.variableName)
      && declaration.structName === `${declaration.variableName}Struct`
      && runnerStorageElementType(source, declaration.structName)
        === RUNNER_RENDER_ELEMENT_TYPES[semantic]
      && runnerCountIdentifier(source, declaration.variableName) === 2
      && runnerCountIdentifier(source, declaration.structName) === 2
      && runnerCommentIdentifierCount(source, declaration.variableName) === 0
      && runnerCommentIdentifierCount(source, declaration.structName) === 0,
    `${purpose}/${lane} diagnostic ${semantic} raw binding/source is invalid`);
    mappings.push({
      semantic,
      group: expected.group,
      binding: expected.binding,
      resourceId: expected.resourceId,
      variableName: declaration.variableName,
      structName: declaration.structName,
    });
    normalized = runnerReplaceIdentifier(
      runnerReplaceIdentifier(
        normalized, declaration.structName, RUNNER_RENDER_CANONICAL_STRUCTS[semantic],
      ),
      declaration.variableName, RUNNER_RENDER_CANONICAL_VARIABLES[semantic],
    );
  }
  return { normalized, mappings };
}

function runnerNormalizeDiagnosticAddressExpression(purpose, lane, source) {
  const visible = runnerEscapeRegExp(RUNNER_RENDER_CANONICAL_VARIABLES.visibleIds);
  const addressVariable = purpose === 'address'
    ? 'phase0SourceAddress' : 'phase0ObjectAddress';
  const objectVariable = purpose === 'address'
    ? 'phase0AddressObjectId' : 'phase0ObjectId';
  const escapedAddressVariable = runnerEscapeRegExp(addressVariable);
  const escapedObjectVariable = runnerEscapeRegExp(objectVariable);
  const laneIndex = lane === 'A'
    ? '\\(\\s*bucketBase\\s*\\+\\s*instanceIndex\\s*\\)'
    : 'instanceIndex';
  const addressDeclaration = new RegExp(
    `var\\s*<\\s*private\\s*>\\s+${escapedAddressVariable}\\s*:\\s*u32\\s*;`, 'gu',
  );
  const objectDeclaration = new RegExp(
    `var\\s*<\\s*private\\s*>\\s+${escapedObjectVariable}\\s*:\\s*u32\\s*;`, 'gu',
  );
  const addressAssignment = new RegExp(
    `(${escapedAddressVariable}\\s*=\\s*)${laneIndex}(\\s*;)`, 'gu',
  );
  const visibleLookup = new RegExp(
    `${escapedObjectVariable}\\s*=\\s*${visible}`
      + `\\s*\\.\\s*value\\s*\\[\\s*${escapedAddressVariable}\\s*\\]\\s*;`,
    'gu',
  );
  runnerWgslRequire(runnerCountMatches(source, addressDeclaration) === 1
    && runnerCountMatches(source, objectDeclaration) === 1
    && runnerCountIdentifier(source, addressVariable) === 3
    && runnerCountIdentifier(source, objectVariable) === (purpose === 'address' ? 3 : 4)
    && runnerCountMatches(source, addressAssignment) === 1
    && runnerCountMatches(source, visibleLookup) === 1,
  `${purpose}/${lane} named address/visible-ID transport is not exact`);
  if (purpose === 'object-id') {
    const matrix = runnerEscapeRegExp(RUNNER_RENDER_CANONICAL_VARIABLES.matrix);
    const matrixLookup = new RegExp(
      `${matrix}\\s*\\.\\s*value\\s*\\[\\s*${escapedObjectVariable}\\s*\\]`,
      'gu',
    );
    runnerWgslRequire(runnerCountMatches(source, matrixLookup) === 1,
      `${purpose}/${lane} matrix/visible-ID lookup chain is not exact`);
  }
  return source.replace(
    addressAssignment,
    '$1__PHASE0_DIAGNOSTIC_ADDRESS_INDEX__$2',
  );
}

/**
 * Independently normalizes the bound raw address/object-ID diagnostic WGSL.
 * It admits only the one A bucket-base input/expression and the one I immediate
 * declaration/expression; generated storage IDs are mapped by binding coordinates.
 */
export function normalizeThreeImmediatePhase0DiagnosticShaders(
  lanes = {}, { purpose } = {},
) {
  const reasons = [];
  const records = {};
  try {
    runnerWgslRequire(['address', 'object-id'].includes(purpose),
      'diagnostic shader purpose is invalid');
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const producer = lanes?.[lane];
      runnerWgslRequire(producer && typeof producer === 'object'
        && typeof producer.vertexShader === 'string' && producer.vertexShader.length > 0
        && typeof producer.fragmentShader === 'string' && producer.fragmentShader.length > 0,
      `${purpose}/${lane} diagnostic raw shader source is absent`);
      let source = producer.vertexShader;
      let immediateAudit = null;
      if (lane === 'I') {
        const immediate = runnerNormalizeImmediateContrast(source);
        source = immediate.normalized;
        immediateAudit = immediate.audit;
      } else {
        runnerWgslRequire(runnerCountIdentifier(source, 'immediate_address_space') === 0
          && runnerCountIdentifier(source, 'threeImmediateDrawBase') === 0
          && runnerCountMatches(source, /var\s*<\s*immediate\s*>/gu) === 0,
        `${purpose}/${lane} unexpectedly contains immediate-address-space code`);
      }
      runnerWgslRequire((source.match(/\.\s*firstInstance\b/gu)?.length ?? 0) === 0
        && exactJson(runnerParseVertexInputs(source), RUNNER_DIAGNOSTIC_VERTEX_INPUTS[lane]),
      `${purpose}/${lane} diagnostic raw vertex input/firstInstance shape is invalid`);
      const { semantics, bySemantic } = runnerDiagnosticStorageMetadata(
        purpose, lane, producer.storageBindings,
      );
      const mapped = runnerNormalizeDiagnosticStorageIdentifiers(
        purpose, lane, source, semantics, bySemantic,
      );
      let normalizedVertexShader = runnerNormalizeDiagnosticAddressExpression(
        purpose, lane, mapped.normalized,
      );
      if (lane === 'A') {
        normalizedVertexShader = runnerReplaceUnique(
          normalizedVertexShader,
          /,\s*@location\s*\(\s*1\s*\)\s*bucketBase\s*:\s*u32\s*/gu,
          ' ',
          `${purpose}/A bucketBase input is not unique`,
        );
      }
      runnerWgslRequire(runnerCountIdentifier(normalizedVertexShader, 'bucketBase') === 0
        && runnerCountIdentifier(normalizedVertexShader, 'threeImmediateDrawBase') === 0
        && runnerCountIdentifier(normalizedVertexShader, 'immediate_address_space') === 0
        && runnerCountIdentifier(normalizedVertexShader,
          '__PHASE0_DIAGNOSTIC_ADDRESS_INDEX__') === 1,
      `${purpose}/${lane} diagnostic normalization left an unapproved transport`);
      records[lane] = {
        rawVertexShader: producer.vertexShader,
        rawFragmentShader: producer.fragmentShader,
        rawVertexSha256: sha256(Buffer.from(producer.vertexShader)),
        rawFragmentSha256: sha256(Buffer.from(producer.fragmentShader)),
        normalizedVertexShader,
        normalizedVertexSha256: sha256(Buffer.from(normalizedVertexShader)),
        semanticMappings: mapped.mappings,
        immediateNormalization: immediateAudit,
      };
    }
  } catch (error) {
    reasons.push(error.message);
  }
  const complete = IMMEDIATE_AIF_PHASE0_LANES.every((lane) => records[lane] !== undefined);
  const normalizedVertexEqual = complete && IMMEDIATE_AIF_PHASE0_LANES.every(
    (lane) => records[lane].normalizedVertexShader === records.A.normalizedVertexShader,
  );
  const rawFragmentEqual = complete && IMMEDIATE_AIF_PHASE0_LANES.every(
    (lane) => records[lane].rawFragmentShader === records.A.rawFragmentShader,
  );
  const rawVertexPairwiseDifferent = complete && new Set(IMMEDIATE_AIF_PHASE0_LANES.map(
    (lane) => records[lane].rawVertexShader,
  )).size === 3;
  if (complete && !normalizedVertexEqual) {
    reasons.push(`${purpose} normalized diagnostic vertex WGSL differs outside approved transports`);
  }
  if (complete && !rawFragmentEqual) {
    reasons.push(`${purpose} diagnostic fragment WGSL is not byte-identical`);
  }
  if (complete && !rawVertexPairwiseDifferent) {
    reasons.push(`${purpose} diagnostic raw vertex WGSL does not expose three transports`);
  }
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-diagnostic-shader-normalization',
    purpose,
    pass: reasons.length === 0 && complete,
    reasons,
    lanes: records,
    comparison: complete ? {
      normalizedVertexEqual,
      rawFragmentEqual,
      rawVertexPairwiseDifferent,
      commonNormalizedVertexSha256: records.A.normalizedVertexSha256,
      commonRawFragmentSha256: records.A.rawFragmentSha256,
    } : null,
  };
}

function bindingWithoutResourceId(binding) {
  const { resourceId, ...shape } = binding;
  return shape;
}

function float32Words(...words) {
  return Array.from(new Float32Array(Uint32Array.from(words).buffer));
}

const EXPECTED_COMPUTE_BINDINGS = deepFreeze({
  reset: [
    ['indirectCommands', 'storage-buffer', 'readWrite', 640,
      'IndirectStorageBufferAttribute', 'Uint32Array', 5, 32],
    ['overflow', 'storage-buffer', 'readWrite', 16,
      'StorageBufferAttribute', 'Uint32Array', 1, 1],
    ['uniforms', 'uniform-buffer', null, 16, null, null, null, null],
  ],
  cull: [
    ['cullOrder', 'storage-buffer', 'readOnly', 262_144,
      'StorageBufferAttribute', 'Uint32Array', 1, 65_536],
    ['bounds', 'storage-buffer', 'readOnly', 1_048_576,
      'StorageBufferAttribute', 'Float32Array', 4, 65_536],
    ['uniforms', 'uniform-buffer', null, 112, null, null, null, null],
    ['objectBucket', 'storage-buffer', 'readOnly', 262_144,
      'StorageBufferAttribute', 'Uint32Array', 1, 65_536],
    ['indirectCommands', 'storage-buffer', 'readWrite', 640,
      'IndirectStorageBufferAttribute', 'Uint32Array', 5, 32],
    ['bucketCapacity', 'storage-buffer', 'readOnly', 128,
      'StorageBufferAttribute', 'Uint32Array', 1, 32],
    ['visibleIds', 'storage-buffer', 'readWrite', 262_144,
      'StorageBufferAttribute', 'Uint32Array', 1, 65_536],
    ['bucketBase', 'storage-buffer', 'readOnly', 128,
      'StorageBufferAttribute', 'Uint32Array', 1, 32],
    ['overflow', 'storage-buffer', 'readWrite', 16,
      'StorageBufferAttribute', 'Uint32Array', 1, 1],
  ],
});

function expectedComputeExecution(phase) {
  const count = phase === 'reset' ? 32 : 65_536;
  return {
    count,
    workgroupSize: [64, 1, 1],
    invocationsPerWorkgroup: 64,
    derivedDispatchSize: [phase === 'reset' ? 1 : 1_024, 1, 1],
    runtimeDispatchSize: [phase === 'reset' ? 1 : 1_024, 1, 1],
    runtimeMatchesDerived: true,
  };
}

function validateComputeBindingInventory(bindings, phase) {
  if (!Array.isArray(bindings)
    || bindings.length !== EXPECTED_COMPUTE_BINDINGS[phase].length) return false;
  const coordinates = new Set();
  for (let index = 0; index < bindings.length; index += 1) {
    const binding = bindings[index];
    const [semantic, kind, accessMode, byteLength,
      attributeType, arrayType, itemSize, count] = EXPECTED_COMPUTE_BINDINGS[phase][index];
    const coordinate = `${binding?.group}:${binding?.binding}`;
    if (binding?.group !== 0 || binding?.binding !== index || coordinates.has(coordinate)
      || binding?.semantic !== semantic || binding?.kind !== kind
      || binding?.access !== accessMode || binding?.visibility !== (kind === 'storage-buffer' ? 4 : 7)
      || binding?.byteLength !== byteLength || binding?.attributeType !== attributeType
      || binding?.arrayType !== arrayType || binding?.itemSize !== itemSize
      || binding?.count !== count) return false;
    coordinates.add(coordinate);
    if (kind === 'storage-buffer') {
      if (!Number.isSafeInteger(binding.resourceId) || binding.resourceId < 0
        || binding.uniformValues !== null) return false;
    } else if (binding.resourceId !== null
      || !Array.isArray(binding.uniformValues)
      || binding.uniformValues.length !== byteLength / 4
      || binding.uniformValues.some((value) => !Number.isFinite(value))) return false;
  }
  const uniforms = bindings.find((binding) => binding.semantic === 'uniforms').uniformValues;
  const countWords = float32Words(phase === 'reset' ? 32 : 65_536, 0, 0, 0);
  return phase === 'reset'
    ? exactArray(uniforms, countWords)
    : exactArray(uniforms.slice(-4), countWords);
}

function validateComputeNormalization(shaders, reasons) {
  const computeLanes = shaders?.computeLanes;
  const reconstructions = { reset: [], cull: [] };
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const laneRecord = computeLanes?.[lane];
    addReasonUnless(laneRecord?.schemaVersion === 1
      && laneRecord?.kind === 'live-first-instance-standalone-compute-shader-evidence'
      && laneRecord?.pass === true && laneRecord?.laneId === lane
      && laneRecord?.fixedWorkloadExact === true
      && laneRecord?.maxStorageBindingCount === 8
      && exactJson(laneRecord?.dispatchDimensions,
        { reset: [1, 1, 1], cull: [1_024, 1, 1] }),
      reasons, `${lane} compute evidence is absent or failed`);
    for (const phase of ['reset', 'cull']) {
      const record = laneRecord?.phases?.[phase];
      try {
        const reconstructed = normalizeThreeImmediatePhase0ComputeShader(
          record?.capture?.computeShader,
          record?.capture?.bindings,
        );
        const hash = sha256(Buffer.from(reconstructed.normalizedShader));
        const rawHash = sha256(Buffer.from(record.capture.computeShader));
        const expectedExecution = expectedComputeExecution(phase);
        const expectedFixed = {
          count: expectedExecution.count,
          workgroupSize: expectedExecution.workgroupSize,
          derivedDispatchSize: expectedExecution.derivedDispatchSize,
        };
        const fieldDeclarations = record.capture.computeShader.match(
          /\bfirstInstance\s*:/gu,
        )?.length ?? 0;
        const fieldAccesses = record.capture.computeShader.match(
          /\.\s*firstInstance\b/gu,
        )?.length ?? 0;
        const generatedVariable = runnerEscapeRegExp(
          reconstructed.audit.generatedVariableIdentifier,
        );
        const allowedIndex = phase === 'reset' ? 'instanceIndex' : 'objectBucket';
        const commandMutation = phase === 'reset'
          ? new RegExp(
              `atomicStore\\s*\\(\\s*&\\s*${generatedVariable}`
                + `\\s*\\.\\s*value\\s*\\[\\s*${allowedIndex}\\s*\\]`
                + '\\s*\\.\\s*instanceCount\\s*,\\s*0u\\s*\\)',
              'gu',
            )
          : new RegExp(
              `atomicAdd\\s*\\(\\s*&\\s*${generatedVariable}`
                + `\\s*\\.\\s*value\\s*\\[\\s*${allowedIndex}\\s*\\]`
                + '\\s*\\.\\s*instanceCount\\s*,\\s*1u\\s*\\)',
              'gu',
            );
        const generatedValueUses = runnerCountMatches(
          record.capture.computeShader,
          new RegExp(`${generatedVariable}\\s*\\.\\s*value\\s*\\[`, 'gu'),
        );
        const commandMutationCount = runnerCountMatches(
          record.capture.computeShader, commandMutation,
        );
        reconstructions[phase].push({
          lane,
          normalizedShader: reconstructed.normalizedShader,
          normalizedSha256: hash,
          rawShader: record.capture.computeShader,
          rawSha256: rawHash,
          audit: reconstructed.audit,
          bindings: record.capture.bindings,
        });
        addReasonUnless(record?.pass === true
          && record?.normalizedShader === reconstructed.normalizedShader
          && record?.normalizedSha256 === hash
          && exactJson(record?.normalization, reconstructed.audit)
          && record?.rawSha256 === rawHash
          && record?.wordFour?.pass === true
          && record?.wordFour?.declarationCount === fieldDeclarations
          && record?.wordFour?.executableAccessCount === fieldAccesses
          && fieldDeclarations === 1 && fieldAccesses === 0
          && generatedValueUses === 1 && commandMutationCount === 1
          && record?.fixedWorkloadExact === true
          && record?.storageBindingsKnown === true
          && record?.workgroupDeclaration === '@workgroup_size( 64, 1, 1 )'
          && exactJson(record?.fixedExpectation, expectedFixed)
          && exactJson(record?.capture?.execution, expectedExecution)
          && validateComputeBindingInventory(record?.capture?.bindings, phase),
        reasons, `${lane}/${phase} compute normalization is not exact`);
      } catch (error) {
        reasons.push(`${lane}/${phase} compute normalization failed: ${error.message}`);
      }
    }
  }
  for (const phase of ['reset', 'cull']) {
    const records = reconstructions[phase];
    const normalizedEqual = records.length === 3
      && new Set(records.map((record) => record.normalizedShader)).size === 1
      && new Set(records.map((record) => record.normalizedSha256)).size === 1;
    const rawPairwiseDifferent = records.length === 3
      && new Set(records.map((record) => record.rawShader)).size === 3;
    const generatedVariableIdentifiers = records.map(
      (record) => record.audit.generatedVariableIdentifier,
    );
    const generatedStructIdentifiers = records.map(
      (record) => record.audit.generatedStructIdentifier,
    );
    const generatedDistinct = records.length === 3
      && new Set(generatedVariableIdentifiers).size === 3
      && new Set(generatedStructIdentifiers).size === 3;
    const bindingShapes = records.map(
      (record) => record.bindings.map(bindingWithoutResourceId),
    );
    const bindingShapesEqual = bindingShapes.length === 3
      && bindingShapes.every((shape) => exactJson(shape, bindingShapes[0]));
    const firstBindings = records[0]?.bindings ?? [];
    const storageSemantics = firstBindings
      .filter((binding) => binding?.kind === 'storage-buffer')
      .map((binding) => binding.semantic);
    const uniqueSemantics = new Set(storageSemantics).size === storageSemantics.length;
    const storageBindingsBySemantic = Object.fromEntries(storageSemantics.map((semantic) => [
      semantic,
      records.map((record) => record.bindings.find(
        (binding) => binding?.kind === 'storage-buffer'
          && binding?.semantic === semantic,
      )?.resourceId ?? null),
    ]));
    const resourceIdentityExact = records.length === 3 && uniqueSemantics
      && Object.entries(storageBindingsBySemantic).every(([semantic, ids]) => (
        ids.every((id) => Number.isSafeInteger(id) && id >= 0)
        && (semantic === 'indirectCommands'
          ? new Set(ids).size === 3 : new Set(ids).size === 1)
      ));
    const expectedSummary = {
      pass: normalizedEqual && rawPairwiseDifferent && generatedDistinct
        && bindingShapesEqual && resourceIdentityExact,
      normalizedShadersEqual: normalizedEqual,
      normalizedHashesEqual: normalizedEqual,
      commonNormalizedSha256: normalizedEqual ? records[0]?.normalizedSha256 : null,
      rawShadersPairwiseDifferent: rawPairwiseDifferent,
      generatedVariablesPairwiseDistinct: generatedDistinct,
      generatedVariableIdentifiers,
      bindingShapesEqual,
      resourceIdentityExact,
      storageBindingsBySemantic,
    };
    addReasonUnless(expectedSummary.pass
      && exactJson(shaders?.computeNormalization?.phases?.[phase], expectedSummary),
    reasons, `${phase} compute WGSL/bindings differ outside lane-local commands`);
  }
  addReasonUnless(shaders?.computeNormalization?.schemaVersion === 1
    && shaders?.computeNormalization?.kind
      === 'immediate-aif-phase0-compute-shader-normalization'
    && shaders?.computeNormalization?.pass === true,
  reasons, 'aggregate three-lane compute normalization is absent or failed');
}

function validateShaderEvidence(shaders, reasons) {
  const lanes = shaders?.lanes;
  const text = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
    lane, lanes?.[lane]?.vertexShader ?? '',
  ]));
  const immediateDeclarationCounts = Object.fromEntries(
    IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
      lane, text[lane].match(/\bvar\s*<\s*immediate\s*>/gu)?.length ?? 0,
    ]),
  );
  addReasonUnless(shaders?.pass === true
    && /bucketBase/u.test(text.A)
    && /bucketBase\s*\+\s*instanceIndex|instanceIndex\s*\+\s*bucketBase/u.test(text.A)
    && immediateDeclarationCounts.A === 0
    && /requires\s+immediate_address_space\s*;/u.test(text.I)
    && /var\s*<\s*immediate\s*>\s*threeImmediateDrawBase\s*:\s*u32\s*;/u.test(text.I)
    && immediateDeclarationCounts.I === 1
    && /threeImmediateDrawBase\s*\+\s*instanceIndex/u.test(text.I)
    && !/bucketBase/u.test(text.I)
    && immediateDeclarationCounts.F === 0
    && !/bucketBase/u.test(text.F),
  reasons, 'raw shader evidence does not prove exact A/I/F address expressions');
  const visibleAttributeIds = new Set();
  const visibleGpuBufferIds = new Set();
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const record = lanes?.[lane];
    const binding = record?.visibleIdsBinding;
    const assignment = record?.addressAssignment;
    const generated = binding?.generatedVariable;
    const escaped = typeof generated === 'string'
      ? generated.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&') : null;
    const matches = escaped === null ? [] : [...record.vertexShader.matchAll(
      new RegExp(`${escaped}\\s*\\.\\s*value\\s*\\[([^\\]]+)\\]`, 'gu'),
    )].map((match) => match[1].replace(/[\s()]/gu, ''));
    const expected = lane === 'I' ? 'threeImmediateDrawBase+instanceIndex'
      : lane === 'F' ? 'instanceIndex' : assignment?.expectedExpression;
    addReasonUnless(record?.pass === true && assignment?.pass === true
      && binding?.semantic === 'visibleIds'
      && typeof binding?.group === 'number' && typeof binding?.binding === 'number'
      && typeof binding?.attributeId === 'number'
      && typeof binding?.gpuBufferId === 'string'
      && matches.length === 1 && matches[0] === expected
      && exactArray(assignment?.observedExpressions?.map(
        (value) => value.replace(/[\s()]/gu, ''),
      ), [expected])
      && (lane === 'A'
        ? Array.isArray(record?.vertexInputs)
          && record.vertexInputs.some((input) => input?.name === expected.split('+')[0])
        : !record?.vertexInputs?.some((input) => /bucketBase/u.test(input?.name ?? ''))),
    reasons, `lane ${lane} visible-ID binding/address assignment is not exact`);
    visibleAttributeIds.add(binding?.attributeId);
    visibleGpuBufferIds.add(binding?.gpuBufferId);
  }
  addReasonUnless(visibleAttributeIds.size === 1 && visibleGpuBufferIds.size === 1,
    reasons, 'A/I/F shaders do not bind one shared visible-ID GPU resource');
  const normalizationInputs = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
    lane,
    {
      vertexShader: lanes?.[lane]?.vertexShader,
      fragmentShader: lanes?.[lane]?.fragmentShader,
      vertexInputs: lanes?.[lane]?.vertexInputs,
      storageBindings: lanes?.[lane]?.storageBindings,
    },
  ]));
  const reconstructedRender = normalizeThreeImmediatePhase0RenderShaders(normalizationInputs);
  addReasonUnless(reconstructedRender.pass === true
    && exactArray(reconstructedRender.reasons, []),
  reasons, `independent A/I/F render normalization failed: ${reconstructedRender.reasons.join('; ')}`);
  const renderHashes = [];
  const fragmentHashes = [];
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const reconstructed = reconstructedRender?.lanes?.[lane];
    const retained = shaders?.renderNormalization?.lanes?.[lane];
    const normalizedSha256 = typeof reconstructed?.normalizedVertexShader === 'string'
      ? sha256(Buffer.from(reconstructed.normalizedVertexShader)) : null;
    const rawVertexSha256 = typeof reconstructed?.rawVertexShader === 'string'
      ? sha256(Buffer.from(reconstructed.rawVertexShader)) : null;
    const rawFragmentSha256 = typeof reconstructed?.rawFragmentShader === 'string'
      ? sha256(Buffer.from(reconstructed.rawFragmentShader)) : null;
    renderHashes.push(normalizedSha256);
    fragmentHashes.push(rawFragmentSha256);
    addReasonUnless(retained?.laneId === lane
      && retained?.rawVertexShader === reconstructed?.rawVertexShader
      && retained?.rawFragmentShader === reconstructed?.rawFragmentShader
      && retained?.normalizedVertexShader === reconstructed?.normalizedVertexShader
      && retained?.rawVertexSha256 === rawVertexSha256
      && retained?.rawFragmentSha256 === rawFragmentSha256
      && retained?.normalizedSha256 === normalizedSha256
      && retained?.normalizedVertexSha256 === normalizedSha256
      && exactJson(retained?.vertexInputs, reconstructed?.vertexInputs)
      && exactJson(retained?.storageBindings, reconstructed?.storageBindings)
      && exactJson(retained?.semanticMappings, reconstructed?.semanticMappings)
      && exactJson(retained?.occurrenceCounts, reconstructed?.occurrenceCounts)
      && exactJson(retained?.immediateNormalization,
        reconstructed?.immediateNormalization)
      && retained?.byteLengths?.rawVertex
        === Buffer.byteLength(reconstructed?.rawVertexShader ?? '')
      && retained?.byteLengths?.rawFragment
        === Buffer.byteLength(reconstructed?.rawFragmentShader ?? '')
      && retained?.byteLengths?.normalizedVertex
        === Buffer.byteLength(reconstructed?.normalizedVertexShader ?? ''),
    reasons, `${lane} retained render normalization differs from reconstruction`);
  }
  addReasonUnless(renderHashes.length === 3 && new Set(renderHashes).size === 1
    && new Set(fragmentHashes).size === 1
    && shaders?.renderNormalization?.schemaVersion === 1
    && shaders?.renderNormalization?.kind
      === 'immediate-aif-phase0-render-shader-normalization'
    && shaders?.renderNormalization?.pass === true
    && exactArray(shaders?.renderNormalization?.reasons, [])
    && exactJson(shaders?.renderNormalization?.comparison, {
      ...reconstructedRender.comparison,
      normalizedVertexSha256Equal: true,
      rawFragmentSha256Equal: true,
    })
    && shaders.renderNormalization?.commonVertexSha256 === renderHashes[0]
    && shaders.renderNormalization?.commonFragmentSha256 === fragmentHashes[0],
  reasons, 'bound A/I/F render shaders do not normalize to exact common bytes');
  const computeModules = shaders?.modules?.filter(
    (module) => /@compute\b/u.test(module?.code ?? ''),
  ) ?? [];
  const commandModules = computeModules.filter(
    (module) => /struct\s+FixedSliceIndexedDraw\b/u.test(module.code),
  );
  const computeAuditExact = commandModules.length > 0 && commandModules.every((module) => {
    const declarations = [...module.code.matchAll(
      /struct\s+FixedSliceIndexedDraw\s*\{([^{}]*)\}\s*;/gu,
    )];
    const body = declarations[0]?.[1]?.replace(/\s+/gu, '').replace(/,$/u, '');
    return declarations.length === 1
      && body === 'indexCount:u32,instanceCount:atomic<u32>,firstIndex:u32,baseVertex:i32,firstInstance:u32'
      && (module.code.match(/\.\s*firstInstance\b/gu)?.length ?? 0) === 0;
  });
  const aggregateAudit = shaders?.computeFirstInstanceFieldAudit;
  const auditArray = shaders?.computeFirstInstanceAudits;
  addReasonUnless(Array.isArray(shaders?.modules) && shaders.modules.length > 0
    && shaders?.computeModuleCount === computeModules.length
    && shaders?.commandComputeModuleCount === commandModules.length
    && computeModules.length === 6 && commandModules.length === 6
    && computeAuditExact && Array.isArray(auditArray)
    && auditArray.length === commandModules.length
    && auditArray.every((audit) => audit?.pass === true
      && audit?.declarationCount === 1 && audit?.executableAccessCount === 0)
    && aggregateAudit?.pass === true
    && aggregateAudit?.declarationCount === commandModules.length
    && aggregateAudit?.executableAccessCount === 0
    && shaders?.compilationErrorCount === 0
    && shaders.modules.every((module) => typeof module?.moduleId === 'string'
      && typeof module?.code === 'string' && module.sha256 === sha256(Buffer.from(module.code))
      && Array.isArray(module?.compilationMessages)
      && module.compilationMessages.every((message) => message?.type !== 'error')),
  reasons, 'shader-module compilation/source commitments are incomplete');
  addReasonUnless(computeModules.length === 6
    && computeModules.every((module) => (
      runnerCountIdentifier(module?.code ?? '', 'immediate_address_space') === 0
        && runnerCountMatches(module?.code ?? '', /var\s*<\s*immediate\s*>/gu) === 0
        && runnerCountIdentifier(module?.code ?? '', 'threeImmediateDrawBase') === 0
    )), reasons, 'compute shader modules unexpectedly expose immediate-address state');
  validateComputeNormalization(shaders, reasons);
}

function expectedPipelineVertexBuffers(vertexInputs) {
  if (!Array.isArray(vertexInputs)) return null;
  return vertexInputs.map((input) => ({
    arrayStride: input.itemSize * (input.arrayType === 'Uint32Array' ? 4 : 4),
    stepMode: input.stepMode,
    attributes: [{
      format: input.format,
      offset: 0,
      shaderLocation: input.shaderLocation,
    }],
  }));
}

/**
 * Diagnostic materials intentionally consume only position (plus A's one
 * bucket-base input).  This is independent of the production material, which
 * also consumes normals.  Keep this predicate runner-owned so a retained page
 * boolean cannot make a production vertex layout look like a diagnostic one.
 */
export function validateThreeImmediatePhase0DiagnosticVertexInputs(
  producer,
  { purpose, lane, mergedPosition = null, mergedBucketBase = null } = {},
) {
  const reasons = [];
  const inputs = producer?.vertexInputs;
  const expectedInputs = producer?.expectedVertexInputs;
  const pipelineInputs = producer?.pipelineVertexInputs;
  const expectedCount = lane === 'A' ? 2 : 1;
  const addressAttributeFirst = lane === 'A' && purpose === 'address';
  const expectedPipelineInputs = lane === 'A'
    ? addressAttributeFirst ? [
        { shaderLocation: 0, format: 'uint32', stepMode: 'vertex' },
        { shaderLocation: 1, format: 'float32x3', stepMode: 'vertex' },
      ] : [
        { shaderLocation: 0, format: 'float32x3', stepMode: 'vertex' },
        { shaderLocation: 1, format: 'uint32', stepMode: 'vertex' },
      ]
    : [{ shaderLocation: 0, format: 'float32x3', stepMode: 'vertex' }];
  // Address/A references bucketBase before position, while object-id/A's
  // position assignment builds position first. Three preserves those distinct
  // TSL construction orders as vertex locations/slots. I/F use only position.
  const positionIndex = addressAttributeFirst ? 1 : 0;
  const bucketBaseIndex = lane !== 'A' ? null : addressAttributeFirst ? 0 : 1;
  const position = inputs?.[positionIndex];
  const bucketBase = bucketBaseIndex === null ? undefined : inputs?.[bucketBaseIndex];
  const vertexEvents = producer?.rawEvents?.filter(
    (event) => event?.method === 'setVertexBuffer',
  ) ?? [];
  addReasonUnless(['address', 'object-id'].includes(purpose)
    && IMMEDIATE_AIF_PHASE0_LANES.includes(lane)
    && Array.isArray(inputs) && inputs.length === expectedCount
    && Array.isArray(expectedInputs) && exactJson(expectedInputs, inputs)
    && producer?.vertexInputsExact === true
    && Array.isArray(pipelineInputs)
    && exactJson(pipelineInputs, expectedPipelineInputs)
    && producer?.pipelineVertexInputsExact === true
    && position?.name === 'position'
    && position?.shaderLocation === positionIndex && position?.format === 'float32x3'
    && position?.stepMode === 'vertex' && position?.arrayType === 'Float32Array'
    && position?.itemSize === 3 && position?.normalized === false
    && position?.count === PINNED_MERGED_GEOMETRY.vertexCount
    && Number.isSafeInteger(position?.resourceId) && position.resourceId >= 0
    && (lane !== 'A' || (bucketBase?.name === 'bucketBase'
      && bucketBase?.shaderLocation === bucketBaseIndex && bucketBase?.format === 'uint32'
      && bucketBase?.stepMode === 'vertex' && bucketBase?.arrayType === 'Uint32Array'
      && bucketBase?.itemSize === 1 && bucketBase?.normalized === false
      && bucketBase?.count === PINNED_MERGED_GEOMETRY.vertexCount
      && Number.isSafeInteger(bucketBase?.resourceId) && bucketBase.resourceId >= 0))
    && vertexEvents.length === expectedCount
    && vertexEvents.every((event, index) => event?.slot === index
      && typeof event?.bufferId === 'string' && event.bufferId.length > 0
      && event?.offset === 0 && event?.size === null)
    && (mergedPosition === null || (position.resourceId === mergedPosition.attributeId
      && vertexEvents[positionIndex]?.bufferId === mergedPosition.gpuBufferId))
    && (mergedBucketBase === null || lane !== 'A'
      || (bucketBase.resourceId === mergedBucketBase.attributeId
        && vertexEvents[bucketBaseIndex]?.bufferId === mergedBucketBase.gpuBufferId)),
  reasons, `${purpose ?? '<unknown>'}/${lane ?? '<unknown>'} diagnostic vertex inputs/bindings are not exact`);
  return {
    valid: reasons.length === 0,
    reasons,
    positionResourceId: position?.resourceId ?? null,
    positionGpuBufferId: vertexEvents[positionIndex]?.bufferId ?? null,
    bucketBaseResourceId: bucketBase?.resourceId ?? null,
    bucketBaseGpuBufferId: bucketBaseIndex === null
      ? null : vertexEvents[bucketBaseIndex]?.bufferId ?? null,
  };
}

function exactDepth32ReversedState(value) {
  return exactJson(value, {
    format: 'depth32float',
    depthWriteEnabled: true,
    depthCompare: 'greater-equal',
    stencilFront: null,
    stencilBack: null,
    stencilReadMask: 0xffff_ffff,
    stencilWriteMask: 0xffff_ffff,
    depthBias: 0,
    depthBiasSlopeScale: 0,
    depthBiasClamp: 0,
  });
}

export function validateThreeImmediatePhase0PipelineLayoutImmediateState(
  layout, { lane = null, compute = false } = {},
) {
  const reasons = [];
  const expectedSize = compute ? 0 : lane === 'I' ? 4 : 0;
  addReasonUnless((compute || IMMEDIATE_AIF_PHASE0_LANES.includes(lane))
    && layout !== null && typeof layout === 'object'
    && typeof layout?.layoutId === 'string' && layout.layoutId.length > 0
    && layout?.hasOwnImmediateSize === true
    && layout?.immediateSize === expectedSize,
  reasons, 'pipeline-layout immediate-address-space size is not exact');
  return { valid: reasons.length === 0, reasons, expectedSize };
}

/**
 * Validates the fully materialized GPU render-pipeline descriptor snapshot.
 * Module/layout identities are intentionally treated as opaque nonempty IDs;
 * their actual graph linkage is checked by the calling artifact validator.
 */
export function validateThreeImmediatePhase0RenderPipelineDescriptor(
  descriptor,
  { purpose, lane, vertexInputs } = {},
) {
  const reasons = [];
  const diagnostic = purpose === 'address' || purpose === 'object-id';
  addReasonUnless(['production', 'address', 'object-id'].includes(purpose),
    reasons, 'render pipeline purpose is unknown');
  addReasonUnless(IMMEDIATE_AIF_PHASE0_LANES.includes(lane),
    reasons, 'render pipeline lane is unknown');
  addReasonUnless(descriptor !== null && typeof descriptor === 'object'
    && typeof descriptor?.label === 'string' && descriptor.label.length > 0
    && descriptor?.hasOwnImmediateSize === false && descriptor?.immediateSize === null
    && typeof descriptor?.layoutId === 'string' && descriptor.layoutId.length > 0,
  reasons, 'render pipeline descriptor envelope/default fields are not exact');
  addReasonUnless(typeof descriptor?.vertex?.moduleId === 'string'
    && descriptor.vertex.moduleId.length > 0
    && descriptor?.vertex?.entryPoint === 'main'
    && descriptor?.vertex?.constants === null
    && exactJson(descriptor?.vertex?.buffers, expectedPipelineVertexBuffers(vertexInputs)),
  reasons, 'render pipeline vertex stage/layout is not exact');
  addReasonUnless(typeof descriptor?.fragment?.moduleId === 'string'
    && descriptor.fragment.moduleId.length > 0
    && descriptor?.fragment?.entryPoint === 'main'
    && descriptor?.fragment?.constants === null
    && exactJson(descriptor?.fragment?.targets, [{
      format: 'rgba8unorm', writeMask: 15, blend: null,
    }]),
  reasons, 'render pipeline fragment stage/target is not exact');
  addReasonUnless(exactJson(descriptor?.primitive, {
    topology: 'triangle-list',
    stripIndexFormat: null,
    frontFace: 'ccw',
    cullMode: diagnostic ? 'none' : 'back',
    unclippedDepth: false,
  }), reasons, 'render pipeline primitive state is not exact');
  addReasonUnless(purpose === 'address'
    ? descriptor?.depthStencil === null
    : exactDepth32ReversedState(descriptor?.depthStencil),
  reasons, 'render pipeline depth/stencil state is not exact');
  addReasonUnless(exactJson(descriptor?.multisample, {
    count: 1,
    mask: 0xffff_ffff,
    alphaToCoverageEnabled: false,
  }), reasons, 'render pipeline multisample state is not exact');
  return { valid: reasons.length === 0, reasons };
}

function stripThreeImmediatePhase0WgslComments(source) {
  return String(source ?? '')
    .replace(/\/\*[\s\S]*?\*\//gu, '')
    .replace(/\/\/[^\r\n]*/gu, '');
}

function parseThreeImmediatePhase0WgslBindingType(addressSpace, rawType) {
  const address = String(addressSpace ?? '').replace(/\s+/gu, '').toLowerCase();
  const type = String(rawType ?? '').replace(/\s+/gu, '').toLowerCase();
  if (address === 'uniform') {
    return { buffer: { type: 'uniform', hasDynamicOffset: false, minBindingSize: 0 },
      sampler: null, texture: null, storageTexture: null };
  }
  if (address === 'storage,read') {
    return { buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 0 },
      sampler: null, texture: null, storageTexture: null };
  }
  if (address === 'storage,read_write' || address === 'storage,write') {
    return { buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 0 },
      sampler: null, texture: null, storageTexture: null };
  }
  if (address !== '') return null;
  if (type === 'sampler' || type === 'sampler_comparison') {
    return { buffer: null,
      sampler: { type: type === 'sampler_comparison' ? 'comparison' : 'filtering' },
      texture: null, storageTexture: null };
  }
  const storage = type.match(
    /^texture_storage_(1d|2d|2d_array|3d)<([^,>]+),(read|write|read_write)>$/u,
  );
  if (storage !== null) {
    const access = { read: 'read-only', write: 'write-only', read_write: 'read-write' }[
      storage[3]
    ];
    return { buffer: null, sampler: null, texture: null,
      storageTexture: { access, format: storage[2], viewDimension: storage[1].replace('_', '-') } };
  }
  const depth = type.match(
    /^texture_depth_(2d|2d_array|cube|cube_array)$/u,
  );
  if (depth !== null) {
    return { buffer: null, sampler: null,
      texture: { sampleType: 'depth', viewDimension: depth[1].replace('_', '-'),
        multisampled: false }, storageTexture: null };
  }
  if (type === 'texture_depth_multisampled_2d') {
    return { buffer: null, sampler: null,
      texture: { sampleType: 'depth', viewDimension: '2d', multisampled: true },
      storageTexture: null };
  }
  const sampled = type.match(
    /^texture_(?:(multisampled)_)?(1d|2d|2d_array|cube|cube_array|3d)<(f32|i32|u32)>$/u,
  );
  if (sampled !== null) {
    const sampleType = { f32: 'float', i32: 'sint', u32: 'uint' }[sampled[3]];
    return { buffer: null, sampler: null,
      texture: { sampleType, viewDimension: sampled[2].replace('_', '-'),
        multisampled: sampled[1] === 'multisampled' }, storageTexture: null };
  }
  return null;
}

function collectThreeImmediatePhase0WgslBindings(stages) {
  const reasons = [];
  const byCoordinate = new Map();
  for (const { source, visibility, label } of stages) {
    const text = stripThreeImmediatePhase0WgslComments(source);
    const declarationPattern = /((?:@(binding|group)\s*\(\s*\d+\s*\)\s*){2})var\s*(?:<\s*([^>]+?)\s*>)?\s+([A-Za-z_]\w*)\s*:\s*([^;]+);/gu;
    const matches = [...text.matchAll(declarationPattern)];
    const bindingTokenCount = [...text.matchAll(/@binding\s*\(/gu)].length;
    const groupTokenCount = [...text.matchAll(/@group\s*\(/gu)].length;
    addReasonUnless(bindingTokenCount === matches.length && groupTokenCount === matches.length,
      reasons, `${label} WGSL contains an unparsed resource binding declaration`);
    const stageCoordinates = new Set();
    for (const match of matches) {
      const coordinates = Object.fromEntries([...match[1].matchAll(
        /@(binding|group)\s*\(\s*(\d+)\s*\)/gu,
      )].map((entry) => [entry[1], Number(entry[2])]));
      const resource = parseThreeImmediatePhase0WgslBindingType(match[3], match[5]);
      const key = `${coordinates.group}/${coordinates.binding}`;
      addReasonUnless(Number.isSafeInteger(coordinates.group) && coordinates.group >= 0
        && Number.isSafeInteger(coordinates.binding) && coordinates.binding >= 0
        && resource !== null && !stageCoordinates.has(key),
      reasons, `${label} WGSL resource ${key} is duplicated or has an unsupported type`);
      stageCoordinates.add(key);
      if (resource === null) continue;
      const prior = byCoordinate.get(key);
      if (prior === undefined) {
        byCoordinate.set(key, {
          group: coordinates.group,
          binding: coordinates.binding,
          visibility,
          ...resource,
        });
      } else {
        const priorResource = { buffer: prior.buffer, sampler: prior.sampler,
          texture: prior.texture, storageTexture: prior.storageTexture };
        addReasonUnless(exactJson(priorResource, resource), reasons,
          `${label} WGSL changes the resource type at ${key}`);
        prior.visibility |= visibility;
      }
    }
  }
  const bindings = [...byCoordinate.values()].sort(
    (left, right) => left.group - right.group || left.binding - right.binding,
  );
  // Three's NodeUniformsGroup intentionally exposes uniform buffers to all
  // three shader stages, even when a particular generated module is the only
  // stage that currently declares the group. Other resource visibility is the
  // exact union of the bound WGSL stages.
  for (const binding of bindings) {
    if (binding?.buffer?.type === 'uniform') binding.visibility = 7;
  }
  return { valid: reasons.length === 0, reasons, bindings };
}

function normalizeThreeImmediatePhase0BindGroupLayoutEntry(entry) {
  const kinds = ['buffer', 'sampler', 'texture', 'storageTexture'].filter(
    (kind) => entry?.[kind] !== null && entry?.[kind] !== undefined,
  );
  if (kinds.length !== 1) return null;
  const normalized = {
    binding: entry?.binding,
    visibility: entry?.visibility,
    buffer: null,
    sampler: null,
    texture: null,
    storageTexture: null,
  };
  if (kinds[0] === 'buffer') {
    normalized.buffer = {
      type: entry.buffer?.type ?? 'uniform',
      hasDynamicOffset: entry.buffer?.hasDynamicOffset ?? false,
      minBindingSize: Number(entry.buffer?.minBindingSize ?? 0),
    };
  } else if (kinds[0] === 'sampler') {
    normalized.sampler = { type: entry.sampler?.type ?? 'filtering' };
  } else if (kinds[0] === 'texture') {
    normalized.texture = {
      sampleType: entry.texture?.sampleType ?? 'float',
      viewDimension: entry.texture?.viewDimension ?? '2d',
      multisampled: entry.texture?.multisampled ?? false,
    };
  } else {
    normalized.storageTexture = {
      access: entry.storageTexture?.access ?? 'write-only',
      format: entry.storageTexture?.format ?? null,
      viewDimension: entry.storageTexture?.viewDimension ?? '2d',
    };
  }
  return normalized;
}

export function validateThreeImmediatePhase0ShaderDeclaredPipelineLayouts({
  pipelines = {}, shaders = {}, resources = {}, lifecycle = {},
} = {}) {
  const reasons = [];
  const modules = shaders?.modules ?? [];
  const layouts = pipelines?.layouts ?? [];
  const bindGroupLayouts = pipelines?.bindGroupLayouts ?? [];
  const gpuCreations = resources?.gpuCreations ?? [];
  const textureViews = resources?.textureViews ?? [];
  const allPipelines = [...(pipelines?.render ?? []), ...(pipelines?.compute ?? [])];
  for (const pipeline of allPipelines) {
    const render = typeof pipeline?.vertexModuleId === 'string';
    const stageDefinitions = render
      ? [[pipeline.vertexModuleId, 1, 'vertex'], [pipeline.fragmentModuleId, 2, 'fragment']]
      : [[pipeline?.computeModuleId, 4, 'compute']];
    const stageModules = stageDefinitions.map(([moduleId, visibility, stage]) => {
      const matches = modules.filter((module) => module?.moduleId === moduleId);
      addReasonUnless(matches.length === 1, reasons,
        `${pipeline?.pipelineId ?? '<unknown>'} ${stage} module is not unique`);
      return { source: matches[0]?.code, visibility,
        label: `${pipeline?.pipelineId ?? '<unknown>'}/${stage}` };
    });
    const declarationAudit = collectThreeImmediatePhase0WgslBindings(stageModules);
    reasons.push(...declarationAudit.reasons);
    const layoutMatches = layouts.filter((layout) => layout?.layoutId === pipeline?.layoutId);
    const layout = layoutMatches[0];
    const maxGroup = declarationAudit.bindings.reduce(
      (maximum, binding) => Math.max(maximum, binding.group), -1,
    );
    const expectedGroupCount = maxGroup + 1;
    addReasonUnless(layoutMatches.length === 1
      && layout?.bindGroupLayoutIds?.length === expectedGroupCount
      && Number.isSafeInteger(layout?.sequence) && layout.sequence < pipeline?.sequence,
    reasons, `${pipeline?.pipelineId ?? '<unknown>'} pipeline layout has extra/missing groups`);
    const expectedNonemptyGroups = [];
    for (let group = 0; group < expectedGroupCount; group += 1) {
      const expectedEntries = declarationAudit.bindings.filter(
        (binding) => binding.group === group,
      ).map(({ group: _group, ...entry }) => entry);
      if (expectedEntries.length > 0) expectedNonemptyGroups.push(group);
      const bglId = layout?.bindGroupLayoutIds?.[group];
      const bglMatches = bindGroupLayouts.filter((candidate) => candidate?.layoutId === bglId);
      const bgl = bglMatches[0];
      const observedEntries = (bgl?.entries ?? []).map(
        normalizeThreeImmediatePhase0BindGroupLayoutEntry,
      ).sort((left, right) => (left?.binding ?? -1) - (right?.binding ?? -1));
      addReasonUnless(bglMatches.length === 1
        && Number.isSafeInteger(bgl?.sequence) && bgl.sequence < layout?.sequence
        && observedEntries.every((entry) => entry !== null)
        && exactJson(observedEntries, expectedEntries),
      reasons, `${pipeline?.pipelineId ?? '<unknown>'} group ${group} BGL differs from bound WGSL`);
    }
    addReasonUnless(expectedNonemptyGroups.length === expectedGroupCount
      && exactArray(expectedNonemptyGroups,
        Array.from({ length: expectedGroupCount }, (_, index) => index)),
    reasons, `${pipeline?.pipelineId ?? '<unknown>'} shader binding groups are not contiguous/nonempty`);
    if (!render) continue;
    const traces = (lifecycle?.renderBundleTraces ?? []).filter((trace) => (
      trace?.events?.some((event) => event?.method === 'setPipeline'
        && event?.pipelineId === pipeline?.pipelineId)
    ));
    addReasonUnless(traces.length > 0, reasons,
      `${pipeline?.pipelineId ?? '<unknown>'} has no executed/retained render bundle`);
    for (const trace of traces) {
      const pipelineEvents = trace?.events?.filter(
        (event) => event?.method === 'setPipeline',
      ) ?? [];
      const groupEvents = trace?.events?.filter(
        (event) => event?.method === 'setBindGroup',
      ) ?? [];
      addReasonUnless(pipelineEvents.length === 1
        && pipelineEvents[0]?.pipelineId === pipeline?.pipelineId
        && exactArray(groupEvents.map((event) => event?.index), expectedNonemptyGroups)
        && Number.isSafeInteger(trace?.finishSequence),
      reasons, `${pipeline?.pipelineId ?? '<unknown>'} bundle bind-group partition is not shader-exact`);
      for (const event of groupEvents) {
        const bglId = layout?.bindGroupLayoutIds?.[event?.index];
        const bgl = bindGroupLayouts.find((candidate) => candidate?.layoutId === bglId);
        const bindGroupMatches = gpuCreations.filter(
          (creation) => creation?.method === 'createBindGroup'
            && creation?.resourceId === event?.bindGroupId,
        );
        const bindGroup = bindGroupMatches[0];
        const expectedEntries = (bgl?.entries ?? []).map(
          normalizeThreeImmediatePhase0BindGroupLayoutEntry,
        ).sort((left, right) => (left?.binding ?? -1) - (right?.binding ?? -1));
        const observedEntries = [...(bindGroup?.entries ?? [])].sort(
          (left, right) => left?.binding - right?.binding,
        );
        let resourcesExact = observedEntries.length === expectedEntries.length;
        for (let index = 0; resourcesExact && index < expectedEntries.length; index += 1) {
          const expected = expectedEntries[index];
          const observed = observedEntries[index];
          resourcesExact &&= observed?.binding === expected?.binding;
          if (expected?.buffer !== null) {
            const bufferMatches = gpuCreations.filter(
              (creation) => creation?.method === 'createBuffer'
                && creation?.resourceId === observed?.bufferId,
            );
            resourcesExact &&= typeof observed?.bufferId === 'string'
              && observed?.resourceId === null
              && observed?.bufferOffset === 0
              && bufferMatches.length === 1
              && (observed?.bufferSize === null
                || observed?.bufferSize === bufferMatches[0]?.size)
              && bufferMatches[0]?.sequence < bindGroup?.sequence;
          } else {
            resourcesExact &&= observed?.bufferId === null
              && typeof observed?.resourceId === 'string';
            if (expected?.sampler !== null) {
              const samplerMatches = gpuCreations.filter(
                (creation) => creation?.method === 'createSampler'
                  && creation?.resourceId === observed.resourceId,
              );
              resourcesExact &&= samplerMatches.length === 1
                && samplerMatches[0]?.sequence < bindGroup?.sequence;
            } else {
              const viewMatches = textureViews.filter(
                (view) => view?.viewId === observed.resourceId,
              );
              resourcesExact &&= viewMatches.length === 1
                && viewMatches[0]?.sequence < bindGroup?.sequence;
            }
          }
        }
        addReasonUnless(bindGroupMatches.length === 1
          && bindGroup?.layoutId === bglId
          && bgl?.sequence < bindGroup?.sequence
          && bindGroup.sequence < event?.sequence
          && event.sequence < trace?.finishSequence
          && event?.dynamicOffsets === null
          && event?.dynamicOffsetStart === null
          && event?.dynamicOffsetLength === null
          && event?.selectedDynamicOffsets === null
          && resourcesExact,
        reasons, `${pipeline?.pipelineId ?? '<unknown>'} bundle group ${event?.index} resource chain is not exact`);
      }
    }
  }
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function validateThreeImmediatePhase0RetainedPipelineBindingTopology(
  topology, { pipelineId, rawEvents, pipelines, resources } = {},
) {
  const pipelineMatches = pipelines?.render?.filter(
    (record) => record?.pipelineId === pipelineId,
  ) ?? [];
  const pipeline = pipelineMatches[0];
  const layoutMatches = pipelines?.layouts?.filter(
    (record) => record?.layoutId === pipeline?.layoutId,
  ) ?? [];
  const layout = layoutMatches[0];
  const bindGroupLayouts = layout?.bindGroupLayoutIds?.map((layoutId) => (
    pipelines?.bindGroupLayouts?.filter((record) => record?.layoutId === layoutId) ?? []
  )) ?? [];
  const expectedGroups = Array.from(
    { length: layout?.bindGroupLayoutIds?.length ?? 0 }, (_, index) => index,
  );
  const expectedLayoutCoordinates = bindGroupLayouts.flatMap((matches, group) => (
    matches.length === 1
      ? (matches[0]?.entries ?? []).map((entry) => ({ group, binding: entry?.binding, entry }))
      : []
  ));
  const coordinateKeys = (records) => records?.map(
    (record) => `${record?.group}/${record?.binding}`,
  ).sort() ?? [];
  const expectedCoordinateKeys = coordinateKeys(expectedLayoutCoordinates);
  const rawBindGroups = rawEvents?.filter((event) => event?.method === 'setBindGroup') ?? [];
  const rawPipelineEvents = rawEvents?.filter((event) => event?.method === 'setPipeline') ?? [];
  const shaderModuleIds = new Set([pipeline?.vertexModuleId, pipeline?.fragmentModuleId]);
  const topologyDeclarationsExact = Array.isArray(topology?.shaderDeclarations)
    && topology.shaderDeclarations.length >= expectedCoordinateKeys.length
    && topology.shaderDeclarations.every((declaration) => (
      ['vertex', 'fragment'].includes(declaration?.stage)
        && shaderModuleIds.has(declaration?.moduleId)
        && typeof declaration?.variable === 'string' && declaration.variable.length > 0
        && (declaration?.addressSpace === null
          || typeof declaration?.addressSpace === 'string')
        && typeof declaration?.declaredType === 'string'
        && declaration.declaredType.length > 0
        && Number.isSafeInteger(declaration?.group) && declaration.group >= 0
        && Number.isSafeInteger(declaration?.binding) && declaration.binding >= 0
    ));
  const boundGroupsExact = Array.isArray(topology?.boundGroups)
    && topology.boundGroups.length === expectedGroups.length
    && topology.boundGroups.every((record, index) => {
      const eventMatches = rawBindGroups.filter((event) => event?.index === index);
      const event = eventMatches[0];
      const bindGroupMatches = resources?.gpuCreations?.filter(
        (creation) => creation?.method === 'createBindGroup'
          && creation?.resourceId === event?.bindGroupId,
      ) ?? [];
      const expectedBgl = bindGroupLayouts[index]?.[0];
      return record?.pass === true && record?.group === index
        && record?.eventMatchCount === 1
        && exactJson(record?.setBindGroupEvent, event)
        && record?.bindGroupMatchCount === 1
        && exactJson(record?.bindGroup, bindGroupMatches[0])
        && record?.expectedLayoutId === layout?.bindGroupLayoutIds?.[index]
        && exactJson(record?.bindGroupLayout, expectedBgl)
        && exactArray(record?.expectedBindings,
          expectedBgl?.entries?.map((entry) => entry?.binding).sort((a, b) => a - b))
        && exactArray(record?.layoutBindings, record?.expectedBindings)
        && exactArray(record?.entryBindings, record?.expectedBindings);
    });
  return pipelineMatches.length === 1 && layoutMatches.length === 1
    && bindGroupLayouts.length === expectedGroups.length
    && bindGroupLayouts.every((matches) => matches.length === 1)
    && topology?.schemaVersion === 1
    && topology?.kind === 'immediate-aif-phase0-pipeline-binding-topology'
    && topology?.pass === true
    && topology?.pipelineId === pipelineId
    && topology?.pipelineLayoutId === layout?.layoutId
    && topologyDeclarationsExact
    && exactJson(topology?.layoutCoordinates, expectedLayoutCoordinates)
    && exactArray(coordinateKeys(topology?.shaderCoordinates), expectedCoordinateKeys)
    && Array.isArray(topology?.bindingSemantics)
    && topology.bindingSemantics.length === expectedCoordinateKeys.length
    && topology.bindingSemantics.every((record) => record?.pass === true)
    && exactArray(coordinateKeys(topology.bindingSemantics), expectedCoordinateKeys)
    && exactArray(topology?.expectedGroupIndices, expectedGroups)
    && exactArray(topology?.observedGroupIndices, expectedGroups)
    && exactArray(rawBindGroups.map((event) => event?.index).sort((a, b) => a - b),
      expectedGroups)
    && exactJson(topology?.pipelineEvents, rawPipelineEvents)
    && rawPipelineEvents.length === 1
    && rawPipelineEvents[0]?.pipelineId === pipelineId
    && boundGroupsExact;
}

function validatePipelineEvidence(pipelines, shaders, reasons) {
  addReasonUnless(Array.isArray(pipelines?.layouts)
    && Array.isArray(pipelines?.render) && Array.isArray(pipelines?.compute)
    && new Set(pipelines.layouts.map((record) => record?.layoutId)).size
      === pipelines.layouts.length
    && new Set(pipelines.render.map((record) => record?.pipelineId)).size
      === pipelines.render.length
    && new Set(pipelines.compute.map((record) => record?.pipelineId)).size
      === pipelines.compute.length,
  reasons, 'GPU pipeline/layout identity inventories are malformed or duplicated');
  addReasonUnless(pipelines?.compute?.length === 6
    && pipelines.compute.every((record) => record?.hasOwnImmediateSize === false
      && record?.immediateSize === null
      && record?.descriptor?.hasOwnImmediateSize === false
      && record?.descriptor?.immediateSize === null),
  reasons, 'compute pipeline descriptors unexpectedly expose immediate state');
  const normalizedDescriptors = [];
  const bindGroupLayoutShapes = [];
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const shader = shaders?.lanes?.[lane];
    const size = lane === 'I' ? 4 : 0;
    const matches = pipelines?.render?.filter(
      (record) => record?.pipelineId === shader?.pipelineId,
    ) ?? [];
    const record = matches[0];
    const layout = pipelines?.layouts?.find(
      (candidate) => candidate?.layoutId === record?.layoutId,
    );
    const vertexModule = shaders?.modules?.find(
      (module) => module?.moduleId === record?.vertexModuleId,
    );
    const fragmentModule = shaders?.modules?.find(
      (module) => module?.moduleId === record?.fragmentModuleId,
    );
    const boundBindGroupLayouts = layout?.bindGroupLayoutIds?.map((layoutId) => (
      pipelines?.bindGroupLayouts?.filter((candidate) => candidate?.layoutId === layoutId)
    ));
    const descriptor = record?.descriptor;
    const flattenedVertexLayouts = descriptor?.vertex?.buffers?.flatMap((buffer) => (
      buffer?.attributes?.map((attribute) => ({
        shaderLocation: attribute?.shaderLocation,
        format: attribute?.format,
        stepMode: buffer?.stepMode,
      })) ?? []
    ));
    const expectedVertexLayouts = shader?.vertexInputs?.map((input) => ({
      shaderLocation: input?.shaderLocation,
      format: input?.format,
      stepMode: input?.stepMode,
    }));
    const descriptorAudit = validateThreeImmediatePhase0RenderPipelineDescriptor(
      descriptor, { purpose: 'production', lane, vertexInputs: shader?.vertexInputs },
    );
    const immediateLayoutAudit = validateThreeImmediatePhase0PipelineLayoutImmediateState(
      layout, { lane },
    );
    addReasonUnless(matches.length === 1 && shader?.pipelineImmediateSize === size
      && immediateLayoutAudit.valid
      && vertexModule?.code === shader?.vertexShader
      && fragmentModule?.code === shader?.fragmentShader
      && shader?.vertexSha256 === sha256(Buffer.from(shader?.vertexShader ?? ''))
      && shader?.fragmentSha256 === sha256(Buffer.from(shader?.fragmentShader ?? ''))
      && exactJson(shader?.boundPipeline, {
        pipelineId: record?.pipelineId,
        layoutId: record?.layoutId,
        vertexModuleId: record?.vertexModuleId,
        fragmentModuleId: record?.fragmentModuleId,
      })
      && shader?.boundModules?.vertex?.moduleId === vertexModule?.moduleId
      && shader?.boundModules?.vertex?.code === vertexModule?.code
      && shader?.boundModules?.vertex?.sha256 === vertexModule?.sha256
      && shader?.boundModules?.vertex?.nodeBuilderExact === true
      && shader?.boundModules?.fragment?.moduleId === fragmentModule?.moduleId
      && shader?.boundModules?.fragment?.code === fragmentModule?.code
      && shader?.boundModules?.fragment?.sha256 === fragmentModule?.sha256
      && shader?.boundModules?.fragment?.nodeBuilderExact === true
      && shader?.boundPipeline?.creationSequence === record?.sequence
      && shader?.boundPipeline?.creationPhase === record?.capturePhase
      && shader?.boundModules?.vertex?.creationSequence === vertexModule?.sequence
      && shader?.boundModules?.fragment?.creationSequence === fragmentModule?.sequence
      && shader?.boundModules?.vertex?.creationPhase === vertexModule?.capturePhase
      && shader?.boundModules?.fragment?.creationPhase === fragmentModule?.capturePhase
      && exactJson(shader?.boundPipeline?.descriptor, descriptor)
      && exactJson(shader?.boundPipeline?.pipelineLayout, layout)
      && Array.isArray(boundBindGroupLayouts)
      && boundBindGroupLayouts.every((matchesForLayout) => matchesForLayout.length === 1)
      && exactJson(shader?.boundPipeline?.bindGroupLayouts,
        boundBindGroupLayouts?.map((matchesForLayout) => matchesForLayout[0]))
      && descriptor?.layoutId === record?.layoutId
      && descriptor?.vertex?.moduleId === record?.vertexModuleId
      && descriptor?.fragment?.moduleId === record?.fragmentModuleId
      && descriptor?.vertex?.entryPoint === record?.vertexEntryPoint
      && descriptor?.fragment?.entryPoint === record?.fragmentEntryPoint
      && exactJson(flattenedVertexLayouts, expectedVertexLayouts)
      && descriptorAudit.valid
      && typeof shader?.bundleGpuId === 'string',
    reasons, `lane ${lane} pipeline/layout chain is not exact: ${descriptorAudit.reasons.join('; ')}`);
    const normalized = structuredClone(descriptor ?? {});
    if (normalized.vertex) normalized.vertex.moduleId = '<vertex-module>';
    if (normalized.fragment) normalized.fragment.moduleId = '<fragment-module>';
    normalized.layoutId = '<pipeline-layout>';
    if (lane === 'A' && Array.isArray(normalized?.vertex?.buffers)) {
      const baseInput = shader?.vertexInputs?.find((input) => (
        input?.name === shader?.addressAssignment?.expectedExpression?.split('+')[0]
      ));
      addReasonUnless(baseInput?.format === 'uint32' && baseInput?.stepMode === 'vertex'
        && baseInput?.arrayType === 'Uint32Array' && baseInput?.itemSize === 1
        && baseInput?.count === PINNED_MERGED_GEOMETRY.vertexCount,
      reasons, 'A production bucketBase vertex input is not the frozen merged-vertex attribute');
      normalized.vertex.buffers = normalized.vertex.buffers.flatMap((buffer) => {
        const attributes = (buffer?.attributes ?? []).filter(
          (attribute) => attribute?.shaderLocation !== baseInput?.shaderLocation,
        ).map((attribute) => ({
          ...attribute,
          shaderLocation: attribute.shaderLocation > baseInput?.shaderLocation
            ? attribute.shaderLocation - 1 : attribute.shaderLocation,
        }));
        return attributes.length === 0 ? [] : [{ ...buffer, attributes }];
      });
    }
    normalizedDescriptors.push(normalized);
    bindGroupLayoutShapes.push(boundBindGroupLayouts?.map((matchesForLayout) => (
      matchesForLayout[0]?.entries
    )) ?? []);
  }
  addReasonUnless(normalizedDescriptors.length === 3
    && normalizedDescriptors.slice(1).every(
      (descriptor) => exactJson(descriptor, normalizedDescriptors[0]),
    )
    && bindGroupLayoutShapes.slice(1).every(
      (shape) => exactJson(shape, bindGroupLayoutShapes[0]),
    ),
  reasons, 'A/I/F production pipeline descriptors differ beyond the one A base input');
  addReasonUnless(Array.isArray(pipelines?.compute) && pipelines.compute.length > 0,
    reasons, 'compute pipeline evidence is absent');
  const computeModules = (shaders?.modules ?? []).filter(
    (module) => /@compute\b/u.test(module?.code ?? ''),
  );
  const pipelineLayoutBindings = (pipelines?.compute ?? []).map((pipeline) => {
    const matches = (pipelines?.layouts ?? []).filter(
      (layout) => layout?.layoutId === pipeline?.layoutId,
    );
    const layout = matches.length === 1 ? matches[0] : null;
    return {
      pipelineId: pipeline?.pipelineId,
      pipelineLayoutId: pipeline?.layoutId,
      layoutMatchCount: matches.length,
      hasOwnImmediateSize: layout?.hasOwnImmediateSize ?? false,
      immediateSize: layout?.immediateSize ?? null,
      pass: matches.length === 1
        && layout?.hasOwnImmediateSize === true && layout?.immediateSize === 0,
    };
  });
  const computePipelineLayouts = [...new Map((pipelines?.compute ?? []).flatMap(
    (pipeline) => {
      const matches = (pipelines?.layouts ?? []).filter(
        (layout) => layout?.layoutId === pipeline?.layoutId,
      );
      return matches.length === 1 ? [[pipeline.layoutId, matches[0]]] : [];
    },
  )).values()];
  const expectedComputeImmediateState = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-compute-immediate-state-exclusion',
    pass: true,
    shaderModuleCount: computeModules.length,
    shaderModules: computeModules.map((module) => ({
      moduleId: module?.moduleId,
      creationSequence: module?.sequence,
      requiresImmediateAddressSpaceCount:
        module?.code?.match(/requires\s+immediate_address_space\s*;/gu)?.length ?? 0,
      immediateVariableDeclarationCount:
        module?.code?.match(/var\s*<\s*immediate\s*>/gu)?.length ?? 0,
    })),
    pipelineCount: pipelines?.compute?.length,
    pipelines: pipelines?.compute?.map((pipeline) => ({
      pipelineId: pipeline?.pipelineId,
      creationSequence: pipeline?.sequence,
      hasOwnImmediateSize: pipeline?.hasOwnImmediateSize,
      immediateSize: pipeline?.immediateSize,
      descriptorHasOwnImmediateSize: pipeline?.descriptor?.hasOwnImmediateSize ?? false,
      descriptorImmediateSize: pipeline?.descriptor?.immediateSize ?? null,
    })),
    pipelineLayoutBindingCount: pipelineLayoutBindings.length,
    pipelineLayoutBindings,
    pipelineLayoutCount: computePipelineLayouts.length,
    pipelineLayouts: computePipelineLayouts,
  };
  addReasonUnless(computeModules.length === 6
    && computeModules.every((module) => !/requires\s+immediate_address_space\s*;/u.test(
      module?.code ?? '',
    ) && !/var\s*<\s*immediate\s*>/u.test(module?.code ?? ''))
    && pipelineLayoutBindings.length === 6
    && pipelineLayoutBindings.every((binding) => binding.pass)
    && exactJson(shaders?.computeImmediateState, expectedComputeImmediateState),
  reasons, 'compute immediate-state exclusion/layout evidence is not independently exact');
}

function validateComputeExecution(evidence, reasons) {
  const shaders = evidence?.shaders;
  const pipelines = evidence?.pipelines;
  const execution = shaders?.computeExecution;
  const resources = evidence?.resources?.gpuCreations;
  const computePassTraces = evidence?.lifecycle?.computePassTraces;
  addReasonUnless(execution?.schemaVersion === 1
    && execution?.kind === 'immediate-aif-phase0-bound-compute-execution-scenarios'
    && execution?.pass === true
    && exactArray(execution?.scenarioLoadSequence, VISIBILITY_IDS)
    && execution?.submissionCount === 6,
  reasons, 'aggregate bound compute execution evidence is invalid');
  const identities = {};
  for (const scenarioId of VISIBILITY_IDS) {
    const scenario = execution?.scenarios?.[scenarioId];
    addReasonUnless(scenario?.schemaVersion === 1
      && scenario?.kind === 'immediate-aif-phase0-bound-compute-execution'
      && scenario?.pass === true && scenario?.scenarioId === scenarioId
      && scenario?.exclusivePipelineCount === 6
      && scenario?.exclusiveShaderModuleCount === 6
      && exactArray(Object.keys(scenario?.lanes ?? {}), IMMEDIATE_AIF_PHASE0_LANES),
    reasons, `${scenarioId} bound compute execution envelope is invalid`);
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const laneRecord = scenario?.lanes?.[lane];
      const expectedCapturePhase = `phase0/live/${scenarioId}/${lane}`;
      const traceMatches = Array.isArray(computePassTraces)
        ? computePassTraces.filter((trace) => trace?.encoderId === laneRecord?.computePassEncoderId
          && trace?.capturePhase === expectedCapturePhase) : [];
      const trace = traceMatches[0];
      const events = laneRecord?.rawEvents;
      const dispatches = Array.isArray(events)
        ? events.filter((event) => event?.method === 'dispatchWorkgroups') : [];
      const indirectDispatches = Array.isArray(events)
        ? events.filter((event) => event?.method === 'dispatchWorkgroupsIndirect') : [];
      const ends = Array.isArray(events)
        ? events.filter((event) => event?.method === 'end') : [];
      addReasonUnless(laneRecord?.pass === true
        && laneRecord?.capturePhase === expectedCapturePhase
        && typeof laneRecord?.computePassEncoderId === 'string'
        && traceMatches.length === 1 && exactJson(trace?.events, events)
        && trace?.commandEncoderId === laneRecord?.phases?.reset?.commandEncoderId
        && trace?.commandEncoderId === laneRecord?.phases?.cull?.commandEncoderId
        && Array.isArray(events) && events.length > 0
        && events.every((event, index) => Number.isSafeInteger(event?.sequence)
          && (index === 0 || event.sequence > events[index - 1].sequence))
        && events.every((event) => ['setPipeline', 'setBindGroup',
          'dispatchWorkgroups', 'end'].includes(event?.method))
        && dispatches.length === 2 && indirectDispatches.length === 0
        && ends.length === 1 && ends[0] === events.at(-1),
      reasons, `${scenarioId}/${lane} compute-pass trace is not exact`);
      for (const [phaseIndex, phase] of ['reset', 'cull'].entries()) {
        const record = laneRecord?.phases?.[phase];
        const collector = shaders?.computeLanes?.[lane]?.phases?.[phase];
        const expectedDispatch = phase === 'reset' ? [1, 1, 1] : [1_024, 1, 1];
        const moduleMatches = shaders?.modules?.filter(
          (candidate) => candidate?.moduleId === record?.shaderModuleId,
        ) ?? [];
        const pipelineMatches = pipelines?.compute?.filter(
          (candidate) => candidate?.pipelineId === record?.pipelineId,
        ) ?? [];
        const layoutMatches = pipelines?.layouts?.filter(
          (candidate) => candidate?.layoutId === record?.pipelineLayoutId,
        ) ?? [];
        const pipeline = pipelineMatches[0];
        const layout = layoutMatches[0];
        const dispatch = dispatches[phaseIndex];
        const priorPipelineEvents = events?.slice(0, events.indexOf(dispatch))
          .filter((event) => event?.method === 'setPipeline') ?? [];
        const activePipelineEvent = priorPipelineEvents.at(-1);
        const bindGroupEvents = record?.bindGroups;
        const expectedBindings = collector?.capture?.bindings ?? [];
        const requiredGroups = [...new Set(expectedBindings.map(
          (binding) => binding?.group,
        ))].sort((left, right) => left - right);
        const activePipelineIndex = events?.indexOf(activePipelineEvent) ?? -1;
        const dispatchIndex = events?.indexOf(dispatch) ?? -1;
        const intervalBindGroupEvents = activePipelineIndex >= 0 && dispatchIndex > activePipelineIndex
          ? events.slice(activePipelineIndex + 1, dispatchIndex).filter(
              (event) => event?.method === 'setBindGroup',
            )
          : [];
        const expectedShaderLayoutCoordinates = (layout?.bindGroupLayoutIds ?? []).flatMap(
          (layoutId, group) => {
            const matches = pipelines?.bindGroupLayouts?.filter(
              (candidate) => candidate?.layoutId === layoutId,
            ) ?? [];
            return matches.length === 1 ? (matches[0]?.entries ?? []).map(
              (entry) => ({ group, binding: entry?.binding, entry }),
            ) : [];
          },
        );
        const expectedCapturedBindingCoordinates = [...expectedBindings].map((binding) => ({
          group: binding?.group,
          binding: binding?.binding,
        })).sort((left, right) => left.group - right.group || left.binding - right.binding);
        let rawBindingLayoutExact = Array.isArray(bindGroupEvents)
          && exactJson(bindGroupEvents, intervalBindGroupEvents)
          && Array.isArray(layout?.bindGroupLayoutIds)
          && layout.bindGroupLayoutIds.length === requiredGroups.length
          && exactArray(requiredGroups,
            Array.from({ length: layout.bindGroupLayoutIds.length }, (_, index) => index))
          && bindGroupEvents.length === requiredGroups.length
          && exactArray(bindGroupEvents.map((event) => event?.index), requiredGroups)
          && new Set(bindGroupEvents.map((event) => event?.index)).size
            === requiredGroups.length;
        for (const group of requiredGroups) {
          const groupBindings = expectedBindings.filter(
            (binding) => binding?.group === group,
          );
          const event = bindGroupEvents?.find((candidate) => candidate?.index === group);
          const bindGroupMatches = Array.isArray(resources)
            ? resources.filter((resource) => resource?.method === 'createBindGroup'
              && resource?.resourceId === event?.bindGroupId) : [];
          const bindGroup = bindGroupMatches[0];
          const expectedBindGroupLayoutId = layout?.bindGroupLayoutIds?.[group];
          const bindGroupLayoutMatches = pipelines?.bindGroupLayouts?.filter(
            (candidate) => candidate?.layoutId === expectedBindGroupLayoutId,
          ) ?? [];
          const bindGroupLayout = bindGroupLayoutMatches[0];
          rawBindingLayoutExact &&= bindGroupMatches.length === 1
            && bindGroupLayoutMatches.length === 1
            && bindGroup?.layoutId === expectedBindGroupLayoutId
            && Number.isSafeInteger(bindGroupLayout?.sequence)
            && Number.isSafeInteger(layout?.sequence)
            && Number.isSafeInteger(pipeline?.sequence)
            && Number.isSafeInteger(bindGroup?.sequence)
            && bindGroupLayout.sequence < layout.sequence
            && layout.sequence < pipeline.sequence
            && bindGroupLayout.sequence < bindGroup.sequence
            && bindGroup.sequence < event?.sequence
            && event.sequence < dispatch?.sequence
            && event?.dynamicOffsets === null
            && event?.dynamicOffsetStart === null
            && event?.dynamicOffsetLength === null
            && event?.selectedDynamicOffsets === null
            && exactArray(bindGroup?.entries?.map((entry) => entry?.binding),
              groupBindings.map((binding) => binding.binding))
            && exactArray(bindGroupLayout?.entries?.map((entry) => entry?.binding),
              groupBindings.map((binding) => binding.binding));
          for (const binding of groupBindings) {
            const entries = bindGroup?.entries?.filter(
              (candidate) => candidate?.binding === binding?.binding,
            ) ?? [];
            const layoutEntries = bindGroupLayout?.entries?.filter(
              (candidate) => candidate?.binding === binding?.binding,
            ) ?? [];
            const entry = entries[0];
            const layoutEntry = layoutEntries[0];
            const expectedBufferType = binding?.kind === 'uniform-buffer'
              ? 'uniform'
              : binding?.access === 'readOnly' ? 'read-only-storage' : 'storage';
            const bufferCreations = resources?.filter(
              (candidate) => candidate?.method === 'createBuffer'
                && candidate?.resourceId === entry?.bufferId,
            ) ?? [];
            const buffer = bufferCreations[0];
            rawBindingLayoutExact &&= entries.length === 1 && layoutEntries.length === 1
              && layoutEntry?.binding === binding.binding
              && layoutEntry?.visibility === binding.visibility
              && (layoutEntry?.buffer?.type ?? 'uniform') === expectedBufferType
              && layoutEntry?.buffer?.hasDynamicOffset === false
              && layoutEntry?.buffer?.minBindingSize === 0
              && layoutEntry?.sampler === null && layoutEntry?.texture === null
              && layoutEntry?.storageTexture === null
              && bufferCreations.length === 1
              && buffer?.sequence < bindGroup.sequence
              && entry?.bufferOffset === 0 && entry?.bufferSize === null
              && (binding?.kind === 'storage-buffer'
                ? buffer?.size === Math.max(16, binding?.byteLength)
                : buffer?.size === binding?.byteLength);
          }
        }
        let bindingExact = Array.isArray(collector?.capture?.bindings)
          && Array.isArray(record?.boundBindings)
          && record.boundBindings.length === collector.capture.bindings.length;
        for (let index = 0; bindingExact && index < record.boundBindings.length; index += 1) {
          const bound = record.boundBindings[index];
          const binding = collector.capture.bindings[index];
          const bindGroupEvent = bindGroupEvents?.find(
            (event) => event?.index === binding?.group,
          );
          const bindGroupMatches = Array.isArray(resources)
            ? resources.filter((resource) => resource?.method === 'createBindGroup'
              && resource?.resourceId === bindGroupEvent?.bindGroupId) : [];
          const bindGroup = bindGroupMatches[0];
          const entry = bindGroup?.entries?.find(
            (candidate) => candidate?.binding === binding?.binding,
          );
          bindingExact &&= bound?.pass === true
            && bound?.semantic === binding?.semantic && bound?.kind === binding?.kind
            && bound?.group === binding?.group && bound?.binding === binding?.binding
            && bound?.resourceAttributeId === binding?.resourceId
            && bound?.bindGroupId === bindGroupEvent?.bindGroupId
            && bindGroupMatches.length === 1
            && bound?.bindGroupLayoutId === bindGroup?.layoutId
            && entry?.bufferId === bound?.observedGpuBufferId
            && typeof bound?.observedGpuBufferId === 'string'
            && (binding?.kind !== 'storage-buffer'
              || (bound?.observedGpuBufferId === bound?.expectedGpuBufferId
                && typeof bound?.expectedGpuBufferId === 'string'));
        }
        const identityKey = `${lane}/${phase}`;
        identities[identityKey] ??= {
          pipelineId: record?.pipelineId,
          layoutId: record?.pipelineLayoutId,
          moduleId: record?.shaderModuleId,
        };
        addReasonUnless(record?.pass === true
          && record?.capturePhase === expectedCapturePhase
          && record?.computePassEncoderId === laneRecord?.computePassEncoderId
          && record?.commandEncoderId === trace?.commandEncoderId
          && moduleMatches.length === 1
          && moduleMatches[0]?.code === collector?.capture?.computeShader
          && record?.shaderModuleCode === collector?.capture?.computeShader
          && record?.shaderModuleSha256 === sha256(Buffer.from(
            collector?.capture?.computeShader ?? '',
          ))
          && moduleMatches[0]?.sha256 === record?.shaderModuleSha256
          && pipelineMatches.length === 1 && layoutMatches.length === 1
          && validateThreeImmediatePhase0PipelineLayoutImmediateState(
            layout, { compute: true },
          ).valid
          && pipeline?.computeModuleId === record?.shaderModuleId
          && pipeline?.layoutId === record?.pipelineLayoutId
          && pipeline?.computeEntryPoint === 'main' && record?.entryPoint === 'main'
          && exactJson(record?.pipelineLayoutBindGroupLayoutIds,
            layout?.bindGroupLayoutIds)
          && record?.bindGroupPartitionExact === true
          && record?.shaderBindingTopologyExact === true
          && record?.shaderLayoutBindingTopologyExact === true
          && exactJson(record?.capturedBindingCoordinates,
            expectedCapturedBindingCoordinates)
          && exactJson(record?.shaderLayoutCoordinates,
            expectedShaderLayoutCoordinates)
          && exactArray(record?.shaderExpectedGroupIndices, requiredGroups)
          && Array.isArray(record?.shaderBindingSemantics)
          && record.shaderBindingSemantics.length === expectedBindings.length
          && record.shaderBindingSemantics.every((binding) => binding?.pass === true)
          && exactArray(record.shaderBindingSemantics.map(
            (binding) => `${binding?.group}/${binding?.binding}`,
          ).sort(), expectedBindings.map(
            (binding) => `${binding?.group}/${binding?.binding}`,
          ).sort())
          && record?.executionChronologyExact === true
          && exactArray(record?.expectedGroupIndices, requiredGroups)
          && exactArray(record?.observedGroupIndices, requiredGroups)
          && activePipelineEvent?.pipelineId === record?.pipelineId
          && record?.setPipelineSequence === activePipelineEvent?.sequence
          && record?.dispatchSequence === dispatch?.sequence
          && exactArray(record?.dispatch, expectedDispatch)
          && exactArray(record?.expectedDispatch, expectedDispatch)
          && Array.isArray(bindGroupEvents)
          && bindGroupEvents.every((event) => event?.method === 'setBindGroup'
            && event?.sequence > activePipelineEvent?.sequence
            && event?.sequence < dispatch?.sequence
            && events.some((candidate) => exactJson(candidate, event)))
          && bindingExact
          && rawBindingLayoutExact
          && exactJson(collector?.boundExecutions?.[scenarioId], record)
          && exactJson(identities[identityKey], {
            pipelineId: record?.pipelineId,
            layoutId: record?.pipelineLayoutId,
            moduleId: record?.shaderModuleId,
          }),
        reasons, `${scenarioId}/${lane}/${phase} module-to-dispatch chain is invalid`);
      }
    }
  }
  addReasonUnless(Object.keys(identities).length === 6
    && new Set(Object.values(identities).map((record) => record.pipelineId)).size === 6
    && new Set(Object.values(identities).map((record) => record.moduleId)).size === 6,
  reasons, 'compute reset/cull execution did not use six exclusive lane pipelines/modules');
}

function validatePipelineCreationCache(evidence, reasons) {
  const shaders = evidence?.shaders;
  const pipelines = evidence?.pipelines;
  const scenarios = evidence?.scenarios ?? [];
  const production = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
    lane, shaders?.lanes?.[lane]?.pipelineId,
  ]));
  const diagnostics = { address: {}, objectId: {} };
  const diagnosticProducers = { address: {}, objectId: {} };
  const mergedRecords = evidence?.resources?.geometryFixtures
    ?.mergedProduction?.gpuRecords ?? [];
  const mergedPosition = mergedRecords.find((record) => record?.semantic === 'position');
  const mergedBucketBase = mergedRecords.find((record) => record?.semantic === 'bucketBase');
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const callbacks = scenarios.flatMap(scenarioCallbacks)
      .filter((callback) => callback?.lane === lane);
    const addressProducers = callbacks.map((callback) => callback?.address?.producer);
    const objectProducers = callbacks.map((callback) => callback?.output?.objectId?.producer);
    const addressIds = new Set(addressProducers.map((producer) => producer?.pipelineId));
    const objectIds = new Set(objectProducers.map((producer) => producer?.pipelineId));
    addReasonUnless(addressIds.size === 1 && objectIds.size === 1,
      reasons, `${lane} diagnostic pipelines were not cached globally`);
    [diagnostics.address[lane]] = addressIds;
    [diagnostics.objectId[lane]] = objectIds;
    diagnosticProducers.address[lane] = addressProducers;
    diagnosticProducers.objectId[lane] = objectProducers;
    for (const [kind, producers] of [
      ['address', addressProducers], ['object-id', objectProducers],
    ]) {
      const first = producers[0];
      const mergedPositionForPurpose = kind === 'object-id' ? mergedPosition : null;
      addReasonUnless(producers.length === 42 && producers.every((producer) => {
        const inputAudit = validateThreeImmediatePhase0DiagnosticVertexInputs(producer, {
          purpose: kind, lane,
          mergedPosition: mergedPositionForPurpose,
          mergedBucketBase,
        });
        return inputAudit.valid
          && exactJson(producer?.vertexInputs, first?.vertexInputs)
          && exactJson(producer?.expectedVertexInputs, first?.expectedVertexInputs)
          && exactJson(producer?.pipelineVertexInputs, first?.pipelineVertexInputs)
          && producer?.pipelineId === first?.pipelineId;
      }), reasons, `${kind}/${lane} diagnostic vertex input/buffer evidence is not cached and exact`);
    }
  }

  for (const [purpose, key] of [['address', 'address'], ['object-id', 'objectId']]) {
    const normalization = normalizeThreeImmediatePhase0DiagnosticShaders(
      Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
        lane, diagnosticProducers[key]?.[lane]?.[0],
      ])),
      { purpose },
    );
    addReasonUnless(normalization.pass, reasons,
      `${purpose} diagnostic WGSL normalization failed: ${normalization.reasons.join('; ')}`);
  }

  const diagnosticInputIdentity = (kind, lane) => {
    const producer = diagnosticProducers[kind]?.[lane]?.[0];
    return validateThreeImmediatePhase0DiagnosticVertexInputs(producer, {
      purpose: kind === 'objectId' ? 'object-id' : 'address', lane,
      mergedPosition: kind === 'objectId' ? mergedPosition : null,
      mergedBucketBase,
    });
  };
  const addressInputIdentities = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
    (lane) => [lane, diagnosticInputIdentity('address', lane)],
  ));
  const objectInputIdentities = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
    (lane) => [lane, diagnosticInputIdentity('objectId', lane)],
  ));
  addReasonUnless(mergedPosition && mergedBucketBase
    && Object.values(addressInputIdentities).every((audit) => audit.valid)
    && Object.values(objectInputIdentities).every((audit) => audit.valid)
    && addressInputIdentities.A.positionResourceId
      === addressInputIdentities.I.positionResourceId
    && addressInputIdentities.A.positionGpuBufferId
      === addressInputIdentities.I.positionGpuBufferId
    && addressInputIdentities.F.positionResourceId
      !== addressInputIdentities.A.positionResourceId
    && addressInputIdentities.F.positionGpuBufferId
      !== addressInputIdentities.A.positionGpuBufferId
    && !Object.values(addressInputIdentities).some(
      (audit) => audit.positionResourceId === mergedPosition.attributeId
        || audit.positionGpuBufferId === mergedPosition.gpuBufferId,
    )
    && Object.values(objectInputIdentities).every(
      (audit) => audit.positionResourceId === mergedPosition.attributeId
        && audit.positionGpuBufferId === mergedPosition.gpuBufferId,
    )
    && addressInputIdentities.A.bucketBaseResourceId === mergedBucketBase.attributeId
    && addressInputIdentities.A.bucketBaseGpuBufferId === mergedBucketBase.gpuBufferId
    && objectInputIdentities.A.bucketBaseResourceId === mergedBucketBase.attributeId
    && objectInputIdentities.A.bucketBaseGpuBufferId === mergedBucketBase.gpuBufferId,
  reasons, 'diagnostic address/object-ID vertex resources do not have the exact independent topology');
  const renderPurposes = [
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
      id: production[lane], phase: `phase0/prime/${lane}`, purpose: `production/${lane}`,
    })),
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
      id: diagnostics.address[lane], phase: `phase0/diagnostic-prime/${lane}/address`,
      purpose: `address/${lane}`,
    })),
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
      id: diagnostics.objectId[lane], phase: `phase0/diagnostic-prime/${lane}/object-id`,
      purpose: `object-id/${lane}`,
    })),
  ];
  const computePurposes = IMMEDIATE_AIF_PHASE0_LANES.flatMap((lane) => (
    ['reset', 'cull'].map((phase) => ({
      id: shaders?.computeExecution?.scenarios?.v99?.lanes?.[lane]?.phases?.[phase]
        ?.pipelineId,
      phase: `phase0/live/v99/${lane}`,
      purpose: `compute/${lane}/${phase}`,
    }))
  ));
  const allPurposes = [...renderPurposes, ...computePurposes];
  const expectedIds = allPurposes.map((purpose) => purpose.id);
  addReasonUnless(expectedIds.every((id) => typeof id === 'string')
    && new Set(expectedIds).size === 15
    && pipelines?.render?.length === 9 && pipelines?.compute?.length === 6
    && new Set([
      ...(pipelines?.render ?? []).map((record) => record?.pipelineId),
      ...(pipelines?.compute ?? []).map((record) => record?.pipelineId),
    ]).size === 15,
  reasons, 'pipeline inventory is not exactly nine render plus six compute purposes');
  for (const purpose of allPurposes) {
    const inventory = purpose.purpose.startsWith('compute/')
      ? pipelines?.compute : pipelines?.render;
    const matches = inventory?.filter((record) => record?.pipelineId === purpose.id) ?? [];
    addReasonUnless(matches.length === 1 && matches[0]?.capturePhase === purpose.phase,
      reasons, `${purpose.purpose} pipeline was not uniquely created in ${purpose.phase}`);
  }
  const normalizeDiagnosticDescriptor = (descriptor, lane) => {
    const normalized = structuredClone(descriptor ?? {});
    normalized.layoutId = '<pipeline-layout>';
    if (normalized.vertex) normalized.vertex.moduleId = '<vertex-module>';
    if (normalized.fragment) normalized.fragment.moduleId = '<fragment-module>';
    if (lane === 'A' && Array.isArray(normalized?.vertex?.buffers)) {
      const candidates = normalized.vertex.buffers.flatMap((buffer, bufferIndex) => (
        (buffer?.attributes ?? []).map((attribute) => ({ buffer, bufferIndex, attribute }))
      )).filter(({ buffer, attribute }) => buffer?.stepMode === 'vertex'
        && attribute?.format === 'uint32');
      addReasonUnless(candidates.length === 1,
        reasons, 'diagnostic A descriptor does not have one uint32 base input');
      const removedLocation = candidates[0]?.attribute?.shaderLocation;
      normalized.vertex.buffers = normalized.vertex.buffers.flatMap((buffer) => {
        const attributes = (buffer?.attributes ?? []).filter(
          (attribute) => attribute?.shaderLocation !== removedLocation,
        ).map((attribute) => ({
          ...attribute,
          shaderLocation: attribute.shaderLocation > removedLocation
            ? attribute.shaderLocation - 1 : attribute.shaderLocation,
        }));
        return attributes.length === 0 ? [] : [{ ...buffer, attributes }];
      });
    }
    return normalized;
  };
  for (const [kind, key] of [['address', 'address'], ['object-id', 'objectId']]) {
    const descriptors = IMMEDIATE_AIF_PHASE0_LANES.map((lane) => {
      const pipelineId = diagnostics[key][lane];
      const descriptor = pipelines?.render?.find(
        (record) => record?.pipelineId === pipelineId,
      )?.descriptor;
      const descriptorAudit = validateThreeImmediatePhase0RenderPipelineDescriptor(
        descriptor, {
          purpose: kind,
          lane,
          vertexInputs: diagnosticProducers[key]?.[lane]?.[0]?.vertexInputs,
        },
      );
      addReasonUnless(descriptorAudit.valid,
        reasons, `${kind}/${lane} actual diagnostic pipeline state is invalid: ${descriptorAudit.reasons.join('; ')}`);
      return normalizeDiagnosticDescriptor(descriptor, lane);
    });
    addReasonUnless(descriptors.slice(1).every(
      (descriptor) => exactJson(descriptor, descriptors[0]),
    ), reasons, `${kind} A/I/F descriptors differ beyond A's one base input`);
  }
  const allPipelines = [...(pipelines?.render ?? []), ...(pipelines?.compute ?? [])];
  const usedModuleIds = new Set(allPipelines.flatMap((pipeline) => [
    pipeline?.vertexModuleId, pipeline?.fragmentModuleId, pipeline?.computeModuleId,
  ]).filter((value) => value !== null && value !== undefined));
  addReasonUnless(Array.isArray(shaders?.modules)
    && usedModuleIds.size === shaders.modules.length
    && shaders.modules.every((module) => usedModuleIds.has(module?.moduleId)),
  reasons, 'shader-module inventory includes unused or unrecorded pipeline modules');
  for (const module of shaders?.modules ?? []) {
    const consumers = allPipelines.filter((pipeline) => [
      pipeline?.vertexModuleId, pipeline?.fragmentModuleId, pipeline?.computeModuleId,
    ].includes(module?.moduleId));
    const firstConsumer = [...consumers].sort(
      (left, right) => left.sequence - right.sequence,
    )[0];
    addReasonUnless(consumers.length > 0
      && module?.capturePhase === firstConsumer?.capturePhase
      && module?.sequence < firstConsumer?.sequence,
    reasons, `shader module ${module?.moduleId} was not created for its first bound pipeline`);
  }
  const usedLayoutIds = new Set(allPipelines.map((pipeline) => pipeline?.layoutId));
  addReasonUnless(pipelines?.layouts?.every((layout) => usedLayoutIds.has(layout?.layoutId))
    && usedLayoutIds.size === pipelines?.layouts?.length,
  reasons, 'pipeline-layout inventory includes unused or missing layouts');
  for (const layout of pipelines?.layouts ?? []) {
    const consumers = allPipelines.filter((pipeline) => pipeline?.layoutId === layout?.layoutId);
    const firstConsumer = [...consumers].sort(
      (left, right) => left.sequence - right.sequence,
    )[0];
    addReasonUnless(consumers.length > 0
      && layout?.capturePhase === firstConsumer?.capturePhase
      && layout?.sequence < firstConsumer?.sequence,
    reasons, `pipeline layout ${layout?.layoutId} was not created for its first pipeline`);
  }
  const usedBindGroupLayoutIds = new Set((pipelines?.layouts ?? []).flatMap(
    (layout) => layout?.bindGroupLayoutIds ?? [],
  ));
  addReasonUnless(pipelines?.bindGroupLayouts?.every(
    (layout) => usedBindGroupLayoutIds.has(layout?.layoutId),
  ) && usedBindGroupLayoutIds.size === pipelines?.bindGroupLayouts?.length,
  reasons, 'bind-group-layout inventory includes unused or missing layouts');
  for (const layout of pipelines?.bindGroupLayouts ?? []) {
    const layoutConsumers = (pipelines?.layouts ?? []).filter(
      (pipelineLayout) => pipelineLayout?.bindGroupLayoutIds?.includes(layout?.layoutId),
    );
    const firstPipeline = allPipelines.filter((pipeline) => (
      layoutConsumers.some((pipelineLayout) => pipelineLayout.layoutId === pipeline.layoutId)
    )).sort((left, right) => left.sequence - right.sequence)[0];
    addReasonUnless(layoutConsumers.length > 0 && firstPipeline !== undefined
      && layout?.capturePhase === firstPipeline.capturePhase
      && layout?.sequence < firstPipeline.sequence,
    reasons, `bind-group layout ${layout?.layoutId} was not created for its first pipeline`);
  }
  const allowedCreationPhases = new Set(allPurposes.map((purpose) => purpose.phase));
  addReasonUnless([
    ...(shaders?.modules ?? []), ...(pipelines?.bindGroupLayouts ?? []),
    ...(pipelines?.layouts ?? []), ...allPipelines,
  ].every((record) => allowedCreationPhases.has(record?.capturePhase)),
  reasons, 'shader/layout/pipeline compilation occurred outside frozen prime/first-live phases');
  const shaderLayoutAudit = validateThreeImmediatePhase0ShaderDeclaredPipelineLayouts({
    pipelines,
    shaders,
    resources: evidence?.resources,
    lifecycle: evidence?.lifecycle,
  });
  addReasonUnless(shaderLayoutAudit.valid, reasons,
    `shader-declared pipeline-layout closure is invalid: ${shaderLayoutAudit.reasons.join('; ')}`);
}

function validateCapabilities(result, configuration, rawReport, reasons) {
  const value = result?.capabilities;
  const raw = rawReport?.pageResult;
  addReasonUnless(exactJson(value?.rawExpectation,
    configuration.rawExpectation),
  reasons, 'page did not echo the injected raw capability identity');
  addReasonUnless(result?.adapterInfo?.isFallbackAdapter === false
    && exactJson(result?.adapterInfo, raw?.adapterInfo)
    && value?.navigatorGpu === true
    && exactArray(value?.wgslLanguageFeatures, raw?.capabilities?.wgslLanguageFeatures)
    && value?.wgslImmediateAddressSpace === true
    && value?.adapterMaxImmediateSize === raw?.capabilities?.adapterMaxImmediateSize
    && value?.deviceMaxImmediateSize === raw?.capabilities?.deviceMaxImmediateSize
    && value?.renderPassSetImmediates === raw?.capabilities?.renderPassSetImmediates
    && value?.renderBundleSetImmediates === raw?.capabilities?.renderBundleSetImmediates
    && value?.indirectFirstInstanceAdapter === true
    && value?.indirectFirstInstanceDevice === true
    && exactArray(value?.adapterFeatures, raw?.adapterFeatures)
    && exactArray(value?.requestedDeviceFeatures, raw?.requestedDeviceFeatures)
    && exactArray(value?.deviceFeatures, raw?.deviceFeatures)
    && !value.requestedDeviceFeatures.includes('immediate')
    && !value.requestedDeviceFeatures.includes('immediate_address_space'),
  reasons, 'integration capability/device identity does not exactly match raw canary');
}

function traceEventOffset(event) {
  return event?.indirectOffset ?? event?.offset ?? null;
}

function textureSizeExact(size, width, height) {
  return (Array.isArray(size) && size[0] === width && size[1] === height
      && (size.length < 3 || size[2] === 1))
    || (size !== null && typeof size === 'object'
      && size.width === width && size.height === height
      && (size.depthOrArrayLayers === undefined || size.depthOrArrayLayers === 1));
}

function validateRenderExecution(execution, target, {
  capturePhase, bundleGpuId, resources, lifecycle, requireDepth, label,
}, reasons) {
  const traceMatches = lifecycle?.renderPassTraces?.filter(
    (trace) => trace?.encoderId === execution?.renderPassEncoderId
      && trace?.capturePhase === capturePhase,
  ) ?? [];
  const trace = traceMatches[0];
  const executeEvents = trace?.events?.filter(
    (event) => event?.method === 'executeBundles',
  ) ?? [];
  const forbidden = trace?.events?.filter((event) => [
    'draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect',
    'dispatchWorkgroups', 'dispatchWorkgroupsIndirect', 'setImmediates',
  ].includes(event?.method)) ?? [];
  const textureMatches = resources?.gpuCreations?.filter(
    (resource) => resource?.method === 'createTexture'
      && resource?.resourceId === target?.gpuTextureId,
  ) ?? [];
  const texture = textureMatches[0];
  const colorAttachment = trace?.colorAttachments?.[0];
  const viewMatches = resources?.textureViews?.filter(
    (view) => view?.viewId === colorAttachment?.viewId
      && view?.textureId === target?.gpuTextureId,
  ) ?? [];
  const depthAttachment = trace?.depthStencilAttachment;
  const depthTextureId = target?.depthGpuTextureId
    ?? depthAttachment?.textureId ?? null;
  const depthTextureMatches = resources?.gpuCreations?.filter(
    (resource) => resource?.method === 'createTexture'
      && resource?.resourceId === depthTextureId,
  ) ?? [];
  addReasonUnless(execution?.pass === true
    && traceMatches.length === 1
    && Array.isArray(trace?.events) && trace.events.length >= 1
    && executeEvents.length === 1 && forbidden.length === 0
    && executeEvents[0]?.sequence === execution?.executeSequence
    && exactArray(executeEvents[0]?.bundleIds, [bundleGpuId])
    && execution?.cachedBundleId === bundleGpuId
    && exactArray(execution?.bundleIds, [bundleGpuId])
    && exactJson(execution?.colorAttachments, trace?.colorAttachments)
    && exactJson(execution?.depthStencilAttachment, depthAttachment)
    && execution?.colorTargetExact === true
    && trace?.colorAttachments?.length === 1
    && colorAttachment?.textureId === target?.gpuTextureId
    && (target?.colorAttachmentViewId === undefined
      || target?.colorAttachmentViewId === colorAttachment?.viewId),
  reasons, `${label} executeBundles/render-pass trace is invalid`);
  addReasonUnless(textureMatches.length === 1
    && texture?.format === 'rgba8unorm'
    && textureSizeExact(texture?.size, target?.width, target?.height)
    && viewMatches.length === 1,
  reasons, `${label} color target is not the declared GPU texture/view`);
  if (requireDepth) {
    addReasonUnless(depthAttachment !== null
      && typeof depthAttachment?.viewId === 'string'
      && typeof depthTextureId === 'string'
      && depthTextureId !== target?.gpuTextureId
      && depthTextureMatches.length === 1
      && depthTextureMatches[0]?.format === 'depth32float'
      && textureSizeExact(depthTextureMatches[0]?.size, target?.width, target?.height)
      && (execution?.depthTargetExact === undefined
        || execution.depthTargetExact === true),
    reasons, `${label} depth attachment is not the declared depth texture`);
  } else {
    addReasonUnless(depthAttachment === null,
      reasons, `${label} unexpectedly uses a depth attachment`);
  }
  return trace;
}

function bundleEventGrammarExact(events, lane) {
  if (!Array.isArray(events) || events.length === 0 || events.at(-1)?.method !== 'finish') {
    return false;
  }
  const suffixStart = events.findIndex((event) => event?.method === (
    lane === 'I' ? 'setImmediates' : 'drawIndexedIndirect'
  ));
  if (suffixStart < 0) return false;
  const setup = events.slice(0, suffixStart);
  const suffix = events.slice(suffixStart, -1);
  const setupMethods = new Set([
    'setPipeline', 'setBindGroup', 'setIndexBuffer', 'setVertexBuffer',
  ]);
  if (!setup.every((event) => setupMethods.has(event?.method))) return false;
  const pipelines = setup.filter((event) => event.method === 'setPipeline');
  const bindGroups = setup.filter((event) => event.method === 'setBindGroup');
  const indexes = setup.filter((event) => event.method === 'setIndexBuffer');
  const vertices = setup.filter((event) => event.method === 'setVertexBuffer');
  if (pipelines.length !== 1 || indexes.length !== 1
    || bindGroups.length === 0 || vertices.length === 0
    || new Set(bindGroups.map((event) => event.index)).size !== bindGroups.length
    || new Set(vertices.map((event) => event.slot)).size !== vertices.length
    || !bindGroups.every((event) => Number.isSafeInteger(event.index) && event.index >= 0
      && typeof event.bindGroupId === 'string'
      && (event.dynamicOffsets === null || Array.isArray(event.dynamicOffsets)))
    || !vertices.every((event) => Number.isSafeInteger(event.slot) && event.slot >= 0
      && typeof event.bufferId === 'string' && event.offset === 0 && event.size === null)
    || typeof indexes[0].bufferId !== 'string'
    || !['uint16', 'uint32'].includes(indexes[0].indexFormat)
    || indexes[0].offset !== 0 || indexes[0].size !== null) return false;
  if (lane === 'I') {
    return suffix.length === 64 && suffix.every((event, index) => (
      event.method === (index % 2 === 0 ? 'setImmediates' : 'drawIndexedIndirect')
    ));
  }
  return suffix.length === 32
    && suffix.every((event) => event.method === 'drawIndexedIndirect');
}

export function validateThreeImmediatePhase0BundleEventGrammar(events, { lane } = {}) {
  const reasons = [];
  addReasonUnless(IMMEDIATE_AIF_PHASE0_LANES.includes(lane)
    && bundleEventGrammarExact(events, lane),
  reasons, `${String(lane)} render-bundle event grammar is not exact`);
  return { valid: reasons.length === 0, reasons };
}

function validateTrace(bundle, lane, expectedBases, expectedBufferId,
  expectedPipelineId, reasons, label, expectedSourceId = null) {
  const events = bundle?.rawEvents;
  const draws = Array.isArray(events)
    ? events.filter((event) => event?.method === 'drawIndexedIndirect') : [];
  const otherDraws = Array.isArray(events)
    ? events.filter((event) => ['draw', 'drawIndexed', 'drawIndirect',
      'dispatchWorkgroups', 'dispatchWorkgroupsIndirect'].includes(event?.method)) : [];
  const immediates = Array.isArray(events)
    ? events.filter((event) => event?.method === 'setImmediates') : [];
  const sourceSnapshot = bundle?.sourceSnapshot;
  let exact = Array.isArray(events) && events.length > 0
    && events.every((event, index) => Number.isSafeInteger(event?.sequence)
      && (index === 0 || event.sequence > events[index - 1].sequence))
    && otherDraws.length === 0 && draws.length === 32
    && exactArray(draws.map(traceEventOffset), IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets)
    && draws.every((event) => event?.bufferId === expectedBufferId)
    && immediates.length === (lane === 'I' ? 32 : 0)
    && bundleEventGrammarExact(events, lane);
  if (exact && lane === 'I') {
    draws.forEach((draw, index) => {
      const prior = events[events.indexOf(draw) - 1];
      if (prior?.method !== 'setImmediates' || prior?.rangeOffset !== 0
        || prior?.sourceType !== 'Uint32Array' || prior?.sourceElementCount !== 32
        || prior?.sourceByteOffset !== 0 || prior?.sourceByteLength !== 128
        || prior?.dataOffset !== index || prior?.dataSize !== 1
        || prior?.sourceId !== sourceSnapshot?.sourceId
        || prior?.backingBufferId !== sourceSnapshot?.backingBufferId
        || !exactArray(prior?.selectedValues, [expectedBases[index]])) exact = false;
    });
    exact &&= sourceSnapshot?.pass === true
      && sourceSnapshot?.callCount === 32
      && typeof sourceSnapshot?.sourceId === 'string'
      && typeof sourceSnapshot?.backingBufferId === 'string'
      && sourceSnapshot?.sourceType === 'Uint32Array'
      && sourceSnapshot?.sourceElementCount === 32
      && sourceSnapshot?.sourceByteOffset === 0
      && sourceSnapshot?.sourceByteLength === 128
      && exactArray(sourceSnapshot?.dataOffsets, Array.from({ length: 32 }, (_, i) => i))
      && exactArray(sourceSnapshot?.dataSizes, new Array(32).fill(1))
      && exactArray(sourceSnapshot?.recordedValues, expectedBases)
      && (expectedSourceId === null || sourceSnapshot.sourceId === expectedSourceId);
  } else if (lane !== 'I') {
    exact &&= sourceSnapshot === null;
  }
  const setPipelines = Array.isArray(events)
    ? events.filter((event) => event?.method === 'setPipeline') : [];
  const finishEvents = Array.isArray(events)
    ? events.filter((event) => event?.method === 'finish') : [];
  addReasonUnless(bundle?.drawExact === true && bundle?.immediateExact === true
    && bundle?.adjacent === true && bundle?.renderObjectCount === 1
    && bundle?.pipelineImmediateSize === (lane === 'I' ? 4 : 0)
    && bundle?.sourceSnapshotExact === true
    && bundle?.pipelineId === expectedPipelineId
    && setPipelines.length === 1 && setPipelines[0].pipelineId === expectedPipelineId
    && finishEvents.length === 1
    && finishEvents[0] === events.at(-1)
    && bundle?.finishSequence === finishEvents[0]?.sequence
    && exact,
  reasons, `${label} bundle trace is not exact`);
  return {
    pipelineId: setPipelines[0]?.pipelineId ?? null,
    sourceId: sourceSnapshot?.sourceId ?? null,
    backingBufferId: sourceSnapshot?.backingBufferId ?? null,
    finishSequence: finishEvents[0]?.sequence ?? null,
  };
}

function validateDiagnosticProducer(producer, {
  kind, lane, scenarioId, scheduleId, callback, frozen, pipelines, shaders,
  resources, lifecycle, outputSha256, outputByteLength, expectedImmediateSourceId, label,
}, reasons) {
  const expectedPhase = callback?.phases?.[kind === 'address' ? 'address' : 'objectId'];
  const expectedBases = cpuOracles()[scenarioId].schedules[scheduleId][lane].sourceBaseByDraw;
  const events = producer?.rawEvents;
  const draws = Array.isArray(events)
    ? events.filter((event) => event?.method === 'drawIndexedIndirect') : [];
  const forbiddenDraws = Array.isArray(events)
    ? events.filter((event) => ['draw', 'drawIndexed', 'drawIndirect'].includes(event?.method))
    : [];
  const immediates = Array.isArray(events)
    ? events.filter((event) => event?.method === 'setImmediates') : [];
  const setPipelines = Array.isArray(events)
    ? events.filter((event) => event?.method === 'setPipeline') : [];
  const finishes = Array.isArray(events)
    ? events.filter((event) => event?.method === 'finish') : [];
  const pipelineMatches = pipelines?.render?.filter(
    (record) => record?.pipelineId === producer?.pipelineId,
  ) ?? [];
  const pipeline = pipelineMatches[0];
  const layouts = pipelines?.layouts?.filter(
    (record) => record?.layoutId === producer?.pipelineLayoutId,
  ) ?? [];
  const vertexModules = shaders?.modules?.filter(
    (module) => module?.moduleId === producer?.vertexModuleId,
  ) ?? [];
  const fragmentModules = shaders?.modules?.filter(
    (module) => module?.moduleId === producer?.fragmentModuleId,
  ) ?? [];
  const expectedStorageSemantics = kind === 'address'
    ? ['visibleIds'] : ['matrix', 'visibleIds'];
  const storageSemantics = producer?.storageBindings?.map(
    (binding) => binding?.semantic,
  );
  const storageExact = exactArray(storageSemantics, expectedStorageSemantics)
    && Array.isArray(producer?.boundStorage)
    && producer.boundStorage.length === expectedStorageSemantics.length
    && producer.boundStorage.every((binding, index) => {
      const source = producer.storageBindings[index];
      return binding?.pass === true && binding?.semantic === source?.semantic
        && binding?.group === source?.group && binding?.binding === source?.binding
        && binding?.resourceId === source?.resourceId
        && binding?.observedGpuBufferId === binding?.expectedGpuBufferId
        && typeof binding?.observedGpuBufferId === 'string'
        && (binding.semantic !== 'visibleIds'
          || (binding.resourceId === frozen.attributeId
            && binding.observedGpuBufferId === frozen.gpuBufferId));
    });
  const diagnosticStorageAudit = validateThreeImmediatePhase0DiagnosticStorageBindings(
    producer,
    { kind, scenarioId, resources, pipelines },
  );
  let immediateExact = immediates.length === (lane === 'I' ? 32 : 0);
  if (immediateExact && lane === 'I') {
    immediateExact = producer?.sourceSnapshot?.pass === true
      && producer.sourceSnapshot?.sourceId === expectedImmediateSourceId
      && exactArray(producer.sourceSnapshot?.recordedValues, expectedBases)
      && draws.every((draw, index) => {
        const prior = events[events.indexOf(draw) - 1];
        return prior?.method === 'setImmediates'
          && prior?.sourceId === expectedImmediateSourceId
          && prior?.backingBufferId === producer.sourceSnapshot.backingBufferId
          && prior?.dataOffset === index && prior?.dataSize === 1
          && exactArray(prior?.selectedValues, [expectedBases[index]]);
      });
  } else if (lane !== 'I') immediateExact &&= producer?.sourceSnapshot === null;
  const target = producer?.target;
  const installationMatches = resources?.scheduleInstallations?.records?.filter(
    (record) => record?.installationId === callback?.scheduleInstallationId,
  ) ?? [];
  const installation = installationMatches[0];
  const targetExact = kind === 'address'
    ? target?.width === 256 && target?.height === 256 && target?.depthBuffer === false
    : target?.width === TARGET_WIDTH && target?.height === TARGET_HEIGHT
      && target?.depthBuffer === true;
  const renderBundleMatches = lifecycle?.renderBundleTraces?.filter(
    (trace) => trace?.bundleId === producer?.bundleGpuId,
  ) ?? [];
  const mergedRecords = resources?.geometryFixtures?.mergedProduction?.gpuRecords ?? [];
  const mergedIndex = mergedRecords.find((record) => record?.semantic === 'index');
  const indexEvents = Array.isArray(events)
    ? events.filter((event) => event?.method === 'setIndexBuffer') : [];
  const addressRealizations = kind === 'address'
    ? (resources?.addressDiagnosticPositions?.records ?? []).filter(
        (record) => record?.installationId === callback?.scheduleInstallationId,
      )
    : [];
  const addressRealization = addressRealizations[0];
  const addressLaneBinding = addressRealization?.laneBindings?.[lane];
  const addressPositionRecord = addressRealization?.records?.find(
    (record) => record?.semantic === addressLaneBinding?.semantic,
  );
  const diagnosticVertexAudit = validateThreeImmediatePhase0DiagnosticVertexInputs(
    producer,
    {
      purpose: kind, lane,
      mergedPosition: kind === 'object-id'
        ? mergedRecords.find((record) => record?.semantic === 'position')
        : {
            attributeId: addressPositionRecord?.attributeId,
            gpuBufferId: addressPositionRecord?.gpuBufferId,
          },
      mergedBucketBase: mergedRecords.find((record) => record?.semantic === 'bucketBase'),
    },
  );
  const descriptorAudit = validateThreeImmediatePhase0RenderPipelineDescriptor(
    pipeline?.descriptor,
    {
      purpose: kind === 'address' ? 'address' : 'object-id',
      lane,
      vertexInputs: producer?.vertexInputs,
    },
  );
  validateRenderExecution(producer?.execution, target, {
    capturePhase: expectedPhase,
    bundleGpuId: producer?.bundleGpuId,
    resources,
    lifecycle,
    requireDepth: kind === 'object-id',
    label: `${label}/${kind}`,
  }, reasons);
  addReasonUnless(producer?.schemaVersion === 1
    && producer?.kind === `immediate-aif-phase0-${kind}-producer`
    && producer?.pass === true && producer?.diagnosticKind === kind
    && producer?.lane === lane && producer?.capturePhase === expectedPhase
    && pipelineMatches.length === 1 && layouts.length === 1
    && validateThreeImmediatePhase0PipelineLayoutImmediateState(
      layouts[0], { lane },
    ).valid
    && pipeline?.layoutId === producer.pipelineLayoutId
    && pipeline?.vertexModuleId === producer.vertexModuleId
    && pipeline?.fragmentModuleId === producer.fragmentModuleId
    && exactJson(producer?.pipelineLayoutBindGroupLayoutIds,
      layouts[0]?.bindGroupLayoutIds)
    && exactJson(producer?.pipelineDescriptor, pipeline?.descriptor)
    && exactJson(producer?.pipelineLayout, layouts[0])
    && pipeline?.descriptor?.layoutId === producer?.pipelineLayoutId
    && pipeline?.descriptor?.vertex?.moduleId === producer?.vertexModuleId
    && pipeline?.descriptor?.fragment?.moduleId === producer?.fragmentModuleId
    && pipeline?.descriptor?.vertex?.entryPoint === pipeline?.vertexEntryPoint
    && pipeline?.descriptor?.fragment?.entryPoint === pipeline?.fragmentEntryPoint
    && descriptorAudit.valid && diagnosticVertexAudit.valid
    && validateThreeImmediatePhase0RetainedPipelineBindingTopology(
      producer?.pipelineBindingTopology,
      { pipelineId: producer?.pipelineId, rawEvents: producer?.rawEvents,
        pipelines, resources },
    )
    && mergedIndex?.arrayType === 'Uint32Array'
    && mergedIndex?.itemSize === 1
    && mergedIndex?.count === PINNED_MERGED_GEOMETRY.indexCount
    && indexEvents.length === 1
    && indexEvents[0]?.bufferId === mergedIndex?.gpuBufferId
    && indexEvents[0]?.indexFormat === 'uint32'
    && indexEvents[0]?.offset === 0 && indexEvents[0]?.size === null
    && (kind !== 'address' || (addressRealizations.length === 1
      && addressRealization?.scenarioId === scenarioId
      && addressRealization?.scheduleId === scheduleId
      && addressLaneBinding?.attributeId === addressPositionRecord?.attributeId
      && addressLaneBinding?.gpuBufferId === addressPositionRecord?.gpuBufferId
      && diagnosticVertexAudit.positionResourceId === addressPositionRecord?.attributeId
      && diagnosticVertexAudit.positionGpuBufferId === addressPositionRecord?.gpuBufferId))
    && vertexModules.length === 1 && fragmentModules.length === 1
    && vertexModules[0]?.code === producer.vertexShader
    && fragmentModules[0]?.code === producer.fragmentShader
    && producer?.vertexSha256 === sha256(Buffer.from(producer?.vertexShader ?? ''))
    && producer?.fragmentSha256 === sha256(Buffer.from(producer?.fragmentShader ?? ''))
    && producer?.pipelineImmediateSize === (lane === 'I' ? 4 : 0)
    && setPipelines.length === 1 && setPipelines[0]?.pipelineId === producer.pipelineId
    && forbiddenDraws.length === 0 && draws.length === 32
    && exactArray(draws.map(traceEventOffset), IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets)
    && draws.every((draw) => draw?.bufferId === callback?.command?.gpuBufferId)
    && producer?.commandAttributeId === callback?.command?.attributeId
    && producer?.commandGpuBufferId === callback?.command?.gpuBufferId
    && producer?.visibleIdsAttributeId === frozen.attributeId
    && producer?.visibleIdsGpuBufferId === frozen.gpuBufferId
    && producer?.drawCount === 32 && producer?.drawExact === true
    && producer?.setImmediatesCount === (lane === 'I' ? 32 : 0)
    && producer?.immediateExact === true && producer?.adjacent === true
    && immediateExact && storageExact && diagnosticStorageAudit.valid
    && bundleEventGrammarExact(events, lane)
    && Array.isArray(events) && events.every((event, index) => (
      Number.isSafeInteger(event?.sequence)
        && (index === 0 || event.sequence > events[index - 1].sequence)
    ))
    && finishes.length === 1 && finishes[0] === events.at(-1)
    && producer?.finishSequence === finishes[0]?.sequence
    && producer?.traceSha256 === sha256(Buffer.from(JSON.stringify(events)))
    && renderBundleMatches.length === 1
    && exactJson(renderBundleMatches[0]?.events, events)
    && renderBundleMatches[0]?.finishSequence === producer?.finishSequence
    && renderBundleMatches[0]?.nativeFinishReturned === true
    && producer?.execution?.pass === true
    && producer.execution?.cachedBundleId === producer.bundleGpuId
    && exactArray(producer.execution?.bundleIds, [producer.bundleGpuId])
    && Number.isSafeInteger(producer.execution?.executeSequence)
    && installationMatches.length === 1
    && installation?.scenarioId === scenarioId
    && installation?.scheduleId === scheduleId
    && Number.isSafeInteger(installation?.completeSequence)
    && producer.execution.executeSequence > installation.completeSequence
    && typeof producer.execution?.renderPassEncoderId === 'string'
    && typeof producer?.rootUuid === 'string' && typeof producer?.bundleGpuId === 'string'
    && target?.format === 'rgba8unorm' && target?.samples === 0
    && typeof target?.renderTargetUuid === 'string'
    && typeof target?.textureUuid === 'string' && typeof target?.gpuTextureId === 'string'
    && targetExact
    && producer?.outputSha256 === outputSha256
    && producer?.outputByteLength === outputByteLength,
  reasons, `${label} actual ${kind} module/pipeline/bundle/output chain is invalid: ${[
    ...descriptorAudit.reasons, ...diagnosticVertexAudit.reasons,
    ...diagnosticStorageAudit.reasons,
  ].join('; ')}`);
  return {
    bundleGpuId: producer?.bundleGpuId,
    pipelineId: producer?.pipelineId,
    sourceId: producer?.sourceSnapshot?.sourceId ?? null,
  };
}

export function validateThreeImmediatePhase0DiagnosticStorageBindings(
  producer,
  { kind, scenarioId, resources, pipelines } = {},
) {
  const reasons = [];
  const expectedSemantics = kind === 'address'
    ? ['visibleIds'] : kind === 'object-id' ? ['matrix', 'visibleIds'] : [];
  const bindings = producer?.storageBindings;
  const commitments = resources?.gpuResourceCommitments?.records;
  const events = producer?.rawEvents ?? [];
  const bindGroupEvents = events.filter((event) => event?.method === 'setBindGroup');
  const firstDraw = events.find((event) => event?.method === 'drawIndexedIndirect');
  const pipelineMatches = pipelines?.render?.filter(
    (record) => record?.pipelineId === producer?.pipelineId,
  ) ?? [];
  const pipeline = pipelineMatches[0];
  const pipelineLayoutMatches = pipelines?.layouts?.filter(
    (record) => record?.layoutId === pipeline?.layoutId,
  ) ?? [];
  const pipelineLayout = pipelineLayoutMatches[0];
  addReasonUnless(expectedSemantics.length > 0
    && VISIBILITY_IDS.includes(scenarioId)
    && Array.isArray(bindings) && Array.isArray(commitments)
    && exactArray(bindings.map((binding) => binding?.semantic), expectedSemantics)
    && pipelineMatches.length === 1 && pipelineLayoutMatches.length === 1
    && producer?.pipelineLayoutId === pipelineLayout?.layoutId
    && Array.isArray(pipelineLayout?.bindGroupLayoutIds)
    && Number.isSafeInteger(firstDraw?.sequence),
  reasons, `${kind ?? '<unknown>'} diagnostic storage envelope is invalid`);
  for (const binding of bindings ?? []) {
    const commitmentMatches = (commitments ?? []).filter(
      (record) => record?.semantic === binding?.semantic,
    );
    const commitment = commitmentMatches[0];
    const groupEvents = bindGroupEvents.filter(
      (event) => event?.index === binding?.group,
    );
    const bindGroupEvent = groupEvents[0];
    const bindGroupMatches = resources?.gpuCreations?.filter(
      (record) => record?.method === 'createBindGroup'
        && record?.resourceId === bindGroupEvent?.bindGroupId,
    ) ?? [];
    const bindGroup = bindGroupMatches[0];
    const entries = bindGroup?.entries?.filter(
      (entry) => entry?.binding === binding?.binding,
    ) ?? [];
    const layoutId = pipelineLayout?.bindGroupLayoutIds?.[binding?.group];
    const layoutMatches = pipelines?.bindGroupLayouts?.filter(
      (record) => record?.layoutId === layoutId,
    ) ?? [];
    const layoutEntries = layoutMatches[0]?.entries?.filter(
      (entry) => entry?.binding === binding?.binding,
    ) ?? [];
    const bufferMatches = resources?.gpuCreations?.filter(
      (record) => record?.method === 'createBuffer'
        && record?.resourceId === commitment?.gpuBufferId,
    ) ?? [];
    const retainedBound = producer?.boundStorage?.filter(
      (record) => record?.semantic === binding?.semantic,
    ) ?? [];
    addReasonUnless(commitmentMatches.length === 1
      && binding?.resourceId === commitment?.attributeId
      && binding?.count === IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount
      && binding?.byteLength === commitment?.byteLength
      && Number.isSafeInteger(binding?.group) && binding.group >= 0
      && Number.isSafeInteger(binding?.binding) && binding.binding >= 0
      && groupEvents.length === 1
      && bindGroupEvent?.dynamicOffsets === null
      && bindGroupEvent?.dynamicOffsetStart === null
      && bindGroupEvent?.dynamicOffsetLength === null
      && bindGroupMatches.length === 1
      && bindGroup?.layoutId === layoutId
      && Number.isSafeInteger(bindGroup?.sequence)
      && bindGroup.sequence < bindGroupEvent?.sequence
      && bindGroupEvent.sequence < firstDraw?.sequence
      && bindGroupEvent.sequence < producer?.finishSequence
      && entries.length === 1
      && entries[0]?.bufferId === commitment?.gpuBufferId
      && entries[0]?.bufferOffset === 0
      && (entries[0]?.bufferSize === null
        || entries[0]?.bufferSize === commitment?.byteLength)
      && layoutMatches.length === 1 && layoutEntries.length === 1
      && layoutMatches[0]?.sequence < bindGroup.sequence
      && layoutEntries[0]?.visibility === 1
      && layoutEntries[0]?.buffer?.type === 'read-only-storage'
      && layoutEntries[0]?.buffer?.hasDynamicOffset === false
      && layoutEntries[0]?.buffer?.minBindingSize === 0
      && bufferMatches.length === 1
      && bufferMatches[0]?.resourceClass === 'persistent-or-upload-buffer'
      && bufferMatches[0]?.size === commitment?.byteLength
      && bufferMatches[0]?.usage === 172
      && bufferMatches[0]?.sequence < bindGroup.sequence
      && retainedBound.length === 1
      && retainedBound[0]?.pass === true
      && retainedBound[0]?.group === binding.group
      && retainedBound[0]?.binding === binding.binding
      && retainedBound[0]?.resourceId === commitment?.attributeId
      && retainedBound[0]?.bindGroupId === bindGroup?.resourceId
      && retainedBound[0]?.bindGroupLayoutId === layoutId
      && retainedBound[0]?.expectedGpuBufferId === commitment?.gpuBufferId
      && retainedBound[0]?.observedGpuBufferId === commitment?.gpuBufferId,
    reasons, `${scenarioId ?? '<unknown>'}/${kind}/${binding?.semantic ?? '<unknown>'} diagnostic storage binding is not the committed GPU resource`);
  }
  return { valid: reasons.length === 0, reasons };
}

export function validateThreeImmediatePhase0ProductionStorageBindings(
  bundle,
  { lane, scenarioId, shaders, resources, pipelines } = {},
) {
  const reasons = [];
  const bindings = shaders?.lanes?.[lane]?.storageBindings;
  const commitments = resources?.gpuResourceCommitments?.records;
  const events = bundle?.rawEvents ?? [];
  const bindGroupEvents = events.filter((event) => event?.method === 'setBindGroup');
  const expectedSemantics = ['matrix', 'visibleIds'];
  const pipeline = pipelines?.render?.find(
    (record) => record?.pipelineId === bundle?.pipelineId,
  );
  const pipelineLayout = pipelines?.layouts?.find(
    (record) => record?.layoutId === pipeline?.layoutId,
  );
  addReasonUnless(IMMEDIATE_AIF_PHASE0_LANES.includes(lane)
    && VISIBILITY_IDS.includes(scenarioId)
    && Array.isArray(bindings) && Array.isArray(commitments)
    && exactArray(bindings.map((binding) => binding?.semantic), expectedSemantics)
    && new Set(bindings.map((binding) => `${binding?.group}/${binding?.binding}`)).size
      === bindings.length
    && pipeline?.pipelineId === bundle?.pipelineId
    && Array.isArray(pipelineLayout?.bindGroupLayoutIds),
  reasons, 'production storage-binding envelope is not exact');
  for (const binding of bindings ?? []) {
    const commitmentMatches = (commitments ?? []).filter(
      (record) => record?.semantic === binding?.semantic,
    );
    const groupEvents = bindGroupEvents.filter(
      (event) => event?.index === binding?.group,
    );
    const bindGroupEvent = groupEvents[0];
    const bindGroupMatches = resources?.gpuCreations?.filter(
      (record) => record?.method === 'createBindGroup'
        && record?.resourceId === bindGroupEvent?.bindGroupId,
    ) ?? [];
    const bindGroup = bindGroupMatches[0];
    const entries = bindGroup?.entries?.filter(
      (entry) => entry?.binding === binding?.binding,
    ) ?? [];
    const commitment = commitmentMatches[0];
    const layoutId = pipelineLayout?.bindGroupLayoutIds?.[binding?.group];
    const layoutMatches = pipelines?.bindGroupLayouts?.filter(
      (record) => record?.layoutId === layoutId,
    ) ?? [];
    const layoutEntries = layoutMatches[0]?.entries?.filter(
      (entry) => entry?.binding === binding?.binding,
    ) ?? [];
    const bufferMatches = resources?.gpuCreations?.filter(
      (record) => record?.method === 'createBuffer'
        && record?.resourceId === commitment?.gpuBufferId,
    ) ?? [];
    addReasonUnless(commitmentMatches.length === 1
      && Number.isSafeInteger(binding?.group) && binding.group >= 0
      && Number.isSafeInteger(binding?.binding) && binding.binding >= 0
      && binding?.resourceId === commitment?.attributeId
      && typeof commitment?.gpuBufferId === 'string'
      && groupEvents.length === 1
      && bindGroupEvent?.dynamicOffsets === null
      && bindGroupEvent?.dynamicOffsetStart === null
      && bindGroupEvent?.dynamicOffsetLength === null
      && bindGroupEvent?.selectedDynamicOffsets === null
      && bindGroupMatches.length === 1
      && bindGroup?.layoutId === layoutId
      && Number.isSafeInteger(bindGroup?.sequence)
      && bindGroup.sequence < bindGroupEvent?.sequence
      && bindGroupEvent.sequence < bundle?.finishSequence
      && entries.length === 1
      && entries[0]?.bufferId === commitment.gpuBufferId
      && entries[0]?.bufferOffset === 0
      && (entries[0]?.bufferSize === null
        || entries[0]?.bufferSize === commitment?.byteLength)
      && layoutMatches.length === 1 && layoutEntries.length === 1
      && layoutMatches[0]?.sequence < bindGroup.sequence
      && layoutEntries[0]?.visibility === 1
      && layoutEntries[0]?.buffer?.type === 'read-only-storage'
      && layoutEntries[0]?.buffer?.hasDynamicOffset === false
      && layoutEntries[0]?.buffer?.minBindingSize === 0
      && bufferMatches.length === 1
      && bufferMatches[0]?.size === commitment?.byteLength
      && bufferMatches[0]?.usage === 172
      && bufferMatches[0]?.sequence < bindGroup.sequence,
    reasons, `${scenarioId}/${lane} ${binding?.semantic ?? '<unknown>'} production binding is not the committed GPU buffer`);
  }
  return { valid: reasons.length === 0, reasons };
}

function validateCallback(callback, context, reasons) {
  const {
    scenarioId, scheduleId, lane, label, frozen, pipelineIds, pipelines, shaders,
    resources, lifecycle,
    expectedImmediateSourceId = null,
  } = context;
  const oracle = cpuOracles()[scenarioId].schedules[scheduleId][lane];
  const expectedAddress = expectedAddressHash(frozen.visibleIds, scenarioId, scheduleId);
  const command = callback?.command;
  const address = callback?.address;
  const output = callback?.output;
  const bundle = callback?.bundle;
  addReasonUnless(callback?.schemaVersion === 1
    && callback?.kind === 'immediate-aif-phase0-render-callback'
    && callback?.pass === true && callback?.lane === lane
    && callback?.scenarioId === scenarioId && callback?.scheduleId === scheduleId
    && typeof callback?.phase === 'string'
    && exactJson(callback?.phases, {
      production: `${callback?.phase}/production`,
      command: `${callback?.phase}/command-readback`,
      address: `${callback?.phase}/address`,
      objectId: `${callback?.phase}/object-id`,
      visibleIds: `${callback?.phase}/visible-id-readback`,
    }),
  reasons, `${label} callback identity/pass is invalid`);
  addReasonUnless(command?.pass === true && command?.byteLength === 640
    && exactArray(command?.gpuWords, oracle.commandWords)
    && exactArray(command?.expectedWords, oracle.commandWords)
    && command?.sha256 === oracle.commandSha256
    && command?.expectedSha256 === oracle.commandSha256
    && command.sha256 === sha256U32(command.gpuWords)
    && exactArray(command?.offsets, IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets)
    && exactArray(command?.wordsFour,
      Array.from({ length: 32 }, (_, draw) => oracle.commandWords[draw * 5 + 4]))
    && typeof command?.gpuBufferId === 'string' && command.gpuBufferId.length > 0
    && Number.isSafeInteger(command?.attributeId) && command.attributeId >= 0
    && Number.isSafeInteger(command?.version) && command.version >= 0,
  reasons, `${label} command readback differs from independent oracle`);
  addReasonUnless(address?.pass === true && address?.format === 'rgba8unorm'
    && address?.encoding
      === 'rgb24-object-id-plus-one-transparent-zero-background'
    && address?.width === ADDRESS_WIDTH && address?.height === ADDRESS_HEIGHT
    && address?.byteLength === ADDRESS_BYTE_LENGTH
    && address?.activeCount === cpuOracles()[scenarioId].expectedVisibleCount
    && address?.paddingCount
      === 65_536 - cpuOracles()[scenarioId].expectedVisibleCount
    && address?.exactExpectedBytes === true
    && address?.sha256 === expectedAddress && address?.expectedSha256 === expectedAddress
    && address?.visibleIdsAttributeId === frozen.attributeId
    && address?.visibleIdsGpuBufferId === frozen.gpuBufferId
    && address?.visibleIdsSha256 === frozen.sha256
    && address?.visibleIdsCpuSha256 === frozen.sha256
    && address?.visibleIdsExact === true
    && address?.witnessId === `${scenarioId}/${scheduleId}/${lane}`
    && MISMATCH_KEYS.every((key) => address?.mismatchCounts?.[key] === 0),
  reasons, `${label} address target/frozen-survivor binding differs from oracle`);
  addReasonUnless(output?.width === TARGET_WIDTH && output?.height === TARGET_HEIGHT
    && output?.color?.format === 'rgba8unorm'
    && output.color.arrayType === 'Uint8Array'
    && output.color.byteLength === TARGET_BYTE_LENGTH
    && SHA256_PATTERN.test(output.color.sha256)
    && output?.depth?.format === 'depth32float'
    && typeof output.depth.arrayType === 'string' && output.depth.arrayType.length > 0
    && output.depth.byteLength === TARGET_BYTE_LENGTH
    && SHA256_PATTERN.test(output.depth.sha256)
    && output?.objectId?.format === 'rgba8unorm-object-id-plus-one'
    && output.objectId.arrayType === 'Uint8Array'
    && output.objectId.byteLength === TARGET_BYTE_LENGTH
    && SHA256_PATTERN.test(output.objectId.sha256)
    && output.objectId?.validation?.pass === true
    && Number.isSafeInteger(output.objectId.validation.coveredPixelCount)
    && output.objectId.validation.coveredPixelCount > 0
    && Number.isSafeInteger(output.objectId.validation.backgroundPixelCount)
    && output.objectId.validation.coveredPixelCount
      + output.objectId.validation.backgroundPixelCount === TARGET_WIDTH * TARGET_HEIGHT
    && output.objectId.validation.outOfRangePixelCount === 0
    && output.objectId.validation.hiddenPixelCount === 0,
  reasons, `${label} output target commitments are malformed`);
  addReasonUnless(typeof bundle?.rootUuid === 'string'
    && typeof bundle?.bundleGpuId === 'string'
    && Number.isSafeInteger(bundle?.version) && Number.isSafeInteger(bundle?.recordCount)
    && bundle?.drawCount === 32
    && bundle?.setImmediatesCount === (lane === 'I' ? 32 : 0)
    && SHA256_PATTERN.test(bundle?.traceSha256)
    && bundle.traceSha256 === sha256(Buffer.from(JSON.stringify(bundle.rawEvents)))
    && bundle?.execution?.pass === true
    && bundle.execution?.cachedBundleId === bundle.bundleGpuId
    && exactArray(bundle.execution?.bundleIds, [bundle.bundleGpuId])
    && typeof bundle.execution?.renderPassEncoderId === 'string'
    && Number.isSafeInteger(bundle.execution?.executeSequence),
  reasons, `${label} bundle identity/trace commitment is malformed`);
  const productionBundleMatches = lifecycle?.renderBundleTraces?.filter(
    (record) => record?.bundleId === bundle?.bundleGpuId,
  ) ?? [];
  const productionTopologyExact = validateThreeImmediatePhase0RetainedPipelineBindingTopology(
    bundle?.pipelineBindingTopology,
    { pipelineId: bundle?.pipelineId, rawEvents: bundle?.rawEvents,
      pipelines, resources },
  );
  addReasonUnless(productionBundleMatches.length === 1
    && exactJson(productionBundleMatches[0]?.events, bundle?.rawEvents)
    && productionBundleMatches[0]?.finishSequence === bundle?.finishSequence
    && productionBundleMatches[0]?.nativeFinishReturned === true
    && bundle?.target?.width === TARGET_WIDTH
    && bundle?.target?.height === TARGET_HEIGHT
    && bundle?.target?.samples === 0
    && typeof bundle?.target?.renderTargetUuid === 'string'
    && typeof bundle?.target?.textureUuid === 'string'
    && typeof bundle?.target?.gpuTextureId === 'string'
    && typeof bundle?.target?.depthTextureUuid === 'string'
    && typeof bundle?.target?.depthGpuTextureId === 'string'
    && productionTopologyExact,
  reasons, `${label} production render-bundle/target identity is invalid`);
  validateRenderExecution(bundle?.execution, bundle?.target, {
    capturePhase: callback?.phases?.production,
    bundleGpuId: bundle?.bundleGpuId,
    resources,
    lifecycle,
    requireDepth: true,
    label: `${label}/production`,
  }, reasons);
  const trace = validateTrace(bundle, lane, oracle.sourceBaseByDraw,
    command?.gpuBufferId, pipelineIds?.[lane], reasons, label,
    expectedImmediateSourceId);
  const productionStorageAudit = validateThreeImmediatePhase0ProductionStorageBindings(
    bundle,
    {
      lane,
      scenarioId,
      shaders,
      resources,
      pipelines,
    },
  );
  addReasonUnless(productionStorageAudit.valid,
    reasons, `${label} production storage-resource chain is invalid: ${productionStorageAudit.reasons.join('; ')}`);
  const addressProducer = validateDiagnosticProducer(address?.producer, {
    kind: 'address', lane, scenarioId, scheduleId, callback, frozen,
    pipelines, shaders, resources, lifecycle, outputSha256: address?.sha256,
    outputByteLength: address?.byteLength,
    expectedImmediateSourceId: trace.sourceId,
    label,
  }, reasons);
  const objectIdProducer = validateDiagnosticProducer(output?.objectId?.producer, {
    kind: 'object-id', lane, scenarioId, scheduleId, callback, frozen,
    pipelines, shaders, resources, lifecycle, outputSha256: output?.objectId?.sha256,
    outputByteLength: output?.objectId?.byteLength,
    expectedImmediateSourceId: trace.sourceId,
    label,
  }, reasons);
  addReasonUnless(callback?.producerChainsDistinct === true
    && new Set([bundle?.bundleGpuId, addressProducer.bundleGpuId,
      objectIdProducer.bundleGpuId]).size === 3
    && new Set([bundle?.pipelineId, addressProducer.pipelineId,
      objectIdProducer.pipelineId]).size === 3,
  reasons, `${label} production/address/object-ID producer chains are not distinct`);
  return {
    targetHashes: output ? {
      color: output.color?.sha256, depth: output.depth?.sha256,
      objectId: output.objectId?.sha256,
      address: address?.sha256,
    } : null,
    commandBufferId: command?.gpuBufferId,
    rootUuid: bundle?.rootUuid,
    bundleGpuId: bundle?.bundleGpuId,
    pipelineId: trace.pipelineId,
    immediateSourceId: trace.sourceId,
    immediateBackingBufferId: trace.backingBufferId,
    finishSequence: trace.finishSequence,
    executeSequence: bundle?.execution?.executeSequence,
  };
}

export function validateThreeImmediatePhase0AddressWitnesses(addressWitnesses) {
  const reasons = [];
  const expectedIds = VISIBILITY_IDS.flatMap((scenarioId) => (
    IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.flatMap((scheduleId) => (
      IMMEDIATE_AIF_PHASE0_LANES.map((lane) => `${scenarioId}/${scheduleId}/${lane}`)
    ))
  ));
  addReasonUnless(addressWitnesses?.schemaVersion === 1
    && addressWitnesses?.kind === 'immediate-aif-phase0-address-byte-witnesses'
    && addressWitnesses?.pass === true
    && addressWitnesses?.witnessCount === expectedIds.length
    && exactArray(addressWitnesses?.expectedWitnessIds, expectedIds)
    && exactArray(Object.keys(addressWitnesses?.witnesses ?? {}), expectedIds),
  reasons, 'address witness envelope/order is not exact');
  for (const witnessId of expectedIds) {
    const [scenarioId, scheduleId, lane] = witnessId.split('/');
    const witness = addressWitnesses?.witnesses?.[witnessId];
    const expected = expectedAddressBytes(
      cpuOracles()[scenarioId].canonicalPacking, scenarioId, scheduleId,
    );
    let bytes = null;
    try {
      if (!canonicalBase64Syntax(witness?.bytesBase64)) {
        throw new Error('non-canonical base64 syntax');
      }
      bytes = Buffer.from(witness.bytesBase64, 'base64');
      if (bytes.toString('base64') !== witness.bytesBase64) {
        throw new Error('non-canonical base64 encoding');
      }
    } catch (error) {
      reasons.push(`${witnessId} address witness cannot be decoded: ${error.message}`);
    }
    addReasonUnless(witness?.schemaVersion === 1
      && witness?.kind === 'immediate-aif-phase0-address-byte-witness'
      && witness?.witnessId === witnessId
      && witness?.scenarioId === scenarioId && witness?.scheduleId === scheduleId
      && witness?.lane === lane && witness?.encoding === 'base64-rgba8unorm'
      && witness?.byteLength === ADDRESS_BYTE_LENGTH
      && bytes?.byteLength === ADDRESS_BYTE_LENGTH
      && bytes?.equals(expected)
      && witness?.sha256 === sha256(expected)
      && witness?.expectedSha256 === sha256(expected)
      && (bytes === null || witness.sha256 === sha256(bytes)),
    reasons, `${witnessId} raw address bytes differ from the independent oracle`);
  }
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

export function validateThreeImmediatePhase0AddressDiagnosticPositions(
  evidence,
  { lifecycle = {}, gpuCreations = [], scheduleInstallations = [] } = {},
) {
  const reasons = [];
  const witnesses = evidence?.witnesses ?? {};
  const witnessIds = Object.keys(witnesses);
  const expectedInstallations = VISIBILITY_IDS.flatMap((scenarioId, scenarioIndex) => (
    ['canonical', 'S1', 'S2', 'canonical'].map((scheduleId, localIndex) => ({
      scenarioId,
      scheduleId,
      ordinal: scenarioIndex * 4 + localIndex + 1,
      installationId: `${scenarioId}/${scenarioIndex * 4 + localIndex + 1}/${scheduleId}`,
    }))
  ));
  const oracleBytes = new Map();
  const registerOracle = (values) => {
    const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
    const hash = sha256(bytes);
    oracleBytes.set(`sha256-${hash}-${bytes.byteLength}`, bytes);
  };
  const oracles = runnerAddressPositionOracles();
  registerOracle(oracles.common);
  for (const values of Object.values(oracles.feature)) registerOracle(values);
  addReasonUnless(evidence?.schemaVersion === 1
    && evidence?.kind === 'immediate-aif-phase0-address-diagnostic-position-evidence'
    && evidence?.pass === true
    && evidence?.expectedRecordCount === 8 && evidence?.recordCount === 8
    && Array.isArray(evidence?.records) && evidence.records.length === 8
    && evidence?.witnessCount === 4 && witnessIds.length === 4
    && exactArray(witnessIds, [...witnessIds].sort((left, right) => left.localeCompare(right)))
    && exactArray(witnessIds, [...oracleBytes.keys()].sort((left, right) => left.localeCompare(right))),
  reasons, 'address diagnostic position evidence envelope/order/count is invalid');
  for (const witnessId of witnessIds) {
    const witness = witnesses[witnessId];
    const bytes = decodeCanonicalBase64(
      witness?.bytesBase64, `address-position/${witnessId}`, reasons,
    );
    const expected = oracleBytes.get(witnessId);
    addReasonUnless(witness?.schemaVersion === 1
      && witness?.kind === 'immediate-aif-phase0-address-position-cpu-byte-witness'
      && witness?.witnessId === witnessId
      && witness?.encoding === 'base64-exact-bytes'
      && witness?.byteLength === 77_280
      && witness?.sha256 === sha256(bytes ?? Buffer.alloc(0))
      && expected !== undefined && bytes?.equals(expected),
    reasons, `${witnessId} address-position raw bytes differ from runner oracle`);
  }
  const creations = gpuCreations.filter((record) => record?.method === 'createBuffer');
  const identity = {};
  for (let index = 0; index < expectedInstallations.length; index += 1) {
    const expectedInstallation = expectedInstallations[index];
    const record = evidence?.records?.[index];
    const capturePhase = `phase0/schedule-install/${expectedInstallation.scenarioId}`
      + `/${expectedInstallation.ordinal}-${expectedInstallation.scheduleId}`
      + '/diagnostic-position-realization';
    const expectedWriteSemantics = index === 0 ? [] : ['featurePosition'];
    const installationMatches = scheduleInstallations.filter(
      (candidate) => candidate?.installationId === expectedInstallation.installationId,
    );
    const installation = installationMatches[0];
    const positionValues = {
      commonPosition: oracles.common,
      featurePosition: oracles.feature[expectedInstallation.scheduleId],
    };
    const expectedRecords = ['commonPosition', 'featurePosition'];
    addReasonUnless(record?.schemaVersion === 1
      && record?.kind === 'immediate-aif-phase0-address-position-realization'
      && record?.pass === true
      && record?.installationId === expectedInstallation.installationId
      && record?.scenarioId === expectedInstallation.scenarioId
      && record?.scheduleId === expectedInstallation.scheduleId
      && record?.capturePhase === capturePhase
      && Number.isSafeInteger(record?.startSequence)
      && Number.isSafeInteger(record?.queueCompleteSequence)
      && record.startSequence < record.queueCompleteSequence
      && exactArray(record?.expectedWriteSemantics, expectedWriteSemantics)
      && exactArray(record?.observedWriteSemantics, expectedWriteSemantics)
      && record?.queueWriteBufferCount === expectedWriteSemantics.length
      && record?.queueWritesExact === true
      && Array.isArray(record?.records) && record.records.length === 2
      && exactArray(record.records.map((entry) => entry?.semantic), expectedRecords),
    reasons, `${expectedInstallation.installationId} address-position realization envelope is invalid`);
    addReasonUnless(installationMatches.length === 1
      && exactJson(installation?.diagnosticPositionRealization, record)
      && Number.isSafeInteger(installation?.startSequence)
      && Number.isSafeInteger(installation?.queueCompleteSequence)
      && Number.isSafeInteger(installation?.completeSequence)
      && installation.startSequence < installation.queueCompleteSequence
      && installation.queueCompleteSequence < record?.startSequence
      && record?.queueCompleteSequence < installation.completeSequence
      && IMMEDIATE_AIF_PHASE0_LANES.every((lane) => (
        Number.isSafeInteger(installation?.commands?.[lane]?.readbackStartSequence)
          && Number.isSafeInteger(installation?.commands?.[lane]?.readbackCompleteSequence)
          && record.queueCompleteSequence
            < installation.commands[lane].readbackStartSequence
          && installation.commands[lane].readbackStartSequence
            < installation.commands[lane].readbackCompleteSequence
          && installation.commands[lane].readbackCompleteSequence
            < installation.completeSequence
      )),
    reasons, `${expectedInstallation.installationId} realization is not the exact pre-readback installation record`);
    for (const semantic of expectedRecords) {
      const value = record?.records?.find((entry) => entry?.semantic === semantic);
      const values = positionValues[semantic];
      const hash = sha256View(values);
      const witnessId = `sha256-${hash}-${values.byteLength}`;
      identity[semantic] ??= {
        attributeId: value?.attributeId,
        gpuBufferId: value?.gpuBufferId,
      };
      addReasonUnless(value?.semantic === semantic
        && value?.manager === 'renderer._attributes'
        && value?.managerConstructor === 'Attributes'
        && value?.attributeType === 1
        && Number.isSafeInteger(value?.attributeId) && value.attributeId >= 0
        && Number.isSafeInteger(value?.attributeVersion) && value.attributeVersion >= 0
        && value?.afterDataVersion === value.attributeVersion
        && value?.arrayType === 'Float32Array' && value?.itemSize === 3
        && value?.count === PINNED_MERGED_GEOMETRY.vertexCount
        && value?.byteLength === 77_280 && value?.cpuExact === true
        && value?.cpuSha256 === hash && value?.witnessId === witnessId
        && typeof value?.gpuBufferId === 'string' && value.gpuBufferId.length > 0
        && value?.gpuBufferSize === 77_280
        && Number.isSafeInteger(value?.updateStartSequence)
        && Number.isSafeInteger(value?.updateCompleteSequence)
        && record.startSequence < value.updateStartSequence
        && value.updateStartSequence < value.updateCompleteSequence
        && value.updateCompleteSequence < record.queueCompleteSequence
        && value?.attributeId === identity[semantic].attributeId
        && value?.gpuBufferId === identity[semantic].gpuBufferId
        && (semantic === 'commonPosition' || index === 0
          ? value?.beforeDataVersion === value.attributeVersion
            && value?.dirtyBeforeUpdate === false
          : value?.beforeDataVersion === value.attributeVersion - 1
            && value?.dirtyBeforeUpdate === true),
      reasons, `${expectedInstallation.installationId}/${semantic} is not the exact CPU/manager/GPU commitment`);
      const expectedCreationPhase = semantic === 'commonPosition'
        ? 'phase0/diagnostic-prime/A/address'
        : 'phase0/diagnostic-prime/F/address';
      const bufferCreations = creations.filter(
        (creation) => creation?.resourceId === value?.gpuBufferId,
      );
      const bufferCreation = bufferCreations[0];
      addReasonUnless(bufferCreations.length === 1
        && bufferCreation?.method === 'createBuffer'
        && bufferCreation?.resourceClass === 'persistent-or-upload-buffer'
        && bufferCreation?.capturePhase === expectedCreationPhase
        && bufferCreation?.size === 77_280 && bufferCreation?.usage === 44
        && bufferCreation?.mappedAtCreation === true
        && bufferCreation?.sequence < record?.startSequence
        && exactJson(value?.gpuBufferCreation, {
          sequence: bufferCreation?.sequence,
          capturePhase: bufferCreation?.capturePhase,
          resourceId: bufferCreation?.resourceId,
          resourceClass: bufferCreation?.resourceClass,
          size: bufferCreation?.size,
          usage: bufferCreation?.usage,
          mappedAtCreation: bufferCreation?.mappedAtCreation,
        }),
      reasons, `${expectedInstallation.installationId}/${semantic} creation is not the primed persistent vertex buffer`);
    }
    const common = record?.records?.[0];
    const feature = record?.records?.[1];
    addReasonUnless(exactJson(record?.laneBindings, {
      A: { semantic: 'commonPosition', attributeId: common?.attributeId,
        gpuBufferId: common?.gpuBufferId },
      I: { semantic: 'commonPosition', attributeId: common?.attributeId,
        gpuBufferId: common?.gpuBufferId },
      F: { semantic: 'featurePosition', attributeId: feature?.attributeId,
        gpuBufferId: feature?.gpuBufferId },
    }) && common?.attributeId !== feature?.attributeId
      && common?.gpuBufferId !== feature?.gpuBufferId,
    reasons, `${expectedInstallation.installationId} address lane binding topology is invalid`);
    const phaseWrites = (lifecycle?.queueWriteBufferCalls ?? []).filter(
      (write) => write?.capturePhase === capturePhase,
    );
    addReasonUnless(exactJson(record?.queueWriteBuffers, phaseWrites)
      && phaseWrites.length === expectedWriteSemantics.length
      && phaseWrites.every((write) => write?.bufferId === feature?.gpuBufferId
        && write?.bufferOffset === 0 && write?.sourceType === 'Float32Array'
        && write?.sourceByteLength === 77_280 && write?.dataOffset === 0
        && write?.size === null
        && write.sequence > feature?.updateStartSequence
        && write.sequence < feature?.updateCompleteSequence),
    reasons, `${expectedInstallation.installationId} address-position queue upload is invalid`);
    const phaseCreations = creations.filter((creation) => creation?.capturePhase === capturePhase);
    addReasonUnless(phaseCreations.length === 0,
    reasons, `${expectedInstallation.installationId} address-position GPU creation/reuse is invalid`);
  }
  addReasonUnless(identity.commonPosition?.attributeId !== identity.featurePosition?.attributeId
    && identity.commonPosition?.gpuBufferId !== identity.featurePosition?.gpuBufferId,
  reasons, 'address diagnostic common/feature identities are not stable and distinct');
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function canonicalBase64Syntax(value) {
  if (typeof value !== 'string' || value.length % 4 !== 0) return false;
  let padding = 0;
  if (value.endsWith('==')) padding = 2;
  else if (value.endsWith('=')) padding = 1;
  const payloadLength = value.length - padding;
  for (let index = 0; index < payloadLength; index += 1) {
    const code = value.charCodeAt(index);
    const allowed = (code >= 65 && code <= 90) || (code >= 97 && code <= 122)
      || (code >= 48 && code <= 57) || code === 43 || code === 47;
    if (!allowed) return false;
  }
  for (let index = payloadLength; index < value.length; index += 1) {
    if (value.charCodeAt(index) !== 61) return false;
  }
  return padding === 0 || payloadLength % 4 === (padding === 2 ? 2 : 3);
}

function decodeCanonicalBase64(value, label, reasons) {
  if (!canonicalBase64Syntax(value)) {
    reasons.push(`${label} does not use canonical base64 syntax`);
    return null;
  }
  const bytes = Buffer.from(value, 'base64');
  if (bytes.toString('base64') !== value) {
    reasons.push(`${label} does not use canonical base64 encoding`);
    return null;
  }
  return bytes;
}

function independentlyDecodeObjectIdWitness(bytes, scenarioId) {
  const active = cpuOracles()[scenarioId].activeSet;
  const counts = new Map();
  let coveredPixelCount = 0;
  let backgroundPixelCount = 0;
  let outOfRangePixelCount = 0;
  let hiddenPixelCount = 0;
  for (let offset = 0; offset < bytes.length; offset += 4) {
    const encoded = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
    if (encoded === 0) {
      backgroundPixelCount += 1;
      continue;
    }
    coveredPixelCount += 1;
    const objectId = encoded - 1;
    if (objectId >= IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount) outOfRangePixelCount += 1;
    else if (!active.has(objectId)) hiddenPixelCount += 1;
    else counts.set(objectId, (counts.get(objectId) ?? 0) + 1);
  }
  const observedIdCounts = [...counts].sort(([left], [right]) => left - right)
    .map(([objectId, pixelCount]) => ({ objectId, pixelCount }));
  return {
    pass: coveredPixelCount > 0 && outOfRangePixelCount === 0 && hiddenPixelCount === 0,
    coveredPixelCount, backgroundPixelCount, outOfRangePixelCount, hiddenPixelCount,
    uniqueObservedIdCount: observedIdCounts.length,
    observedObjectIds: observedIdCounts.map((record) => record.objectId),
    observedIdCounts,
  };
}

function strictObjectIdAlphaEncoding(bytes) {
  let covered = 0;
  for (let offset = 0; offset < bytes.length; offset += 4) {
    const encoded = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
    if (encoded === 0) {
      if (bytes[offset] !== 0 || bytes[offset + 1] !== 0
        || bytes[offset + 2] !== 0 || bytes[offset + 3] !== 0) return false;
    } else {
      covered += 1;
      if (bytes[offset + 3] !== 255) return false;
    }
  }
  return covered > 0;
}

function productionOutputsCoverObjectIds(color, depth, objectId) {
  if (!Buffer.isBuffer(color) || !Buffer.isBuffer(depth) || !Buffer.isBuffer(objectId)
    || color.length !== TARGET_BYTE_LENGTH || depth.length !== TARGET_BYTE_LENGTH
    || objectId.length !== TARGET_BYTE_LENGTH) return false;
  let clearColor = null;
  let coveredColorDiffers = false;
  let coveredNonClearDepth = false;
  for (let offset = 0; offset < objectId.length; offset += 4) {
    const encoded = objectId[offset]
      | (objectId[offset + 1] << 8) | (objectId[offset + 2] << 16);
    if (encoded === 0 && clearColor === null) clearColor = color.subarray(offset, offset + 4);
  }
  if (clearColor === null) return false;
  for (let offset = 0; offset < objectId.length; offset += 4) {
    const encoded = objectId[offset]
      | (objectId[offset + 1] << 8) | (objectId[offset + 2] << 16);
    if (encoded === 0) continue;
    coveredColorDiffers ||= color[offset] !== clearColor[0]
      || color[offset + 1] !== clearColor[1]
      || color[offset + 2] !== clearColor[2]
      || color[offset + 3] !== clearColor[3];
    const depthValue = depth.readFloatLE(offset);
    coveredNonClearDepth ||= Number.isFinite(depthValue) && depthValue > 0 && depthValue <= 1;
  }
  return coveredColorDiffers && coveredNonClearDepth;
}

function callbackOutputReferences(scenarios) {
  const channels = ['color', 'depth', 'objectId'];
  const callbacks = (scenarios ?? []).flatMap((scenario) => scenarioCallbacks(scenario));
  return {
    callbacks,
    references: callbacks.flatMap((callback, callbackOrdinal) => channels.map((channel) => {
      const output = channel === 'objectId' ? callback?.output?.objectId : callback?.output?.[channel];
      return {
        callbackOrdinal: callbackOrdinal + 1,
        phase: callback?.phase,
        scenarioId: callback?.scenarioId,
        scheduleId: callback?.scheduleId,
        lane: callback?.lane,
        channel,
        witnessId: output?.witnessId ?? null,
        sha256: output?.sha256 ?? null,
        byteLength: output?.byteLength ?? null,
      };
    })),
  };
}

export function validateThreeImmediatePhase0OutputWitnesses(outputWitnesses, scenarios) {
  const reasons = [];
  const channels = ['color', 'depth', 'objectId'];
  const { callbacks, references } = callbackOutputReferences(scenarios);
  const witnesses = outputWitnesses?.witnesses;
  const witnessIds = Object.keys(witnesses ?? {});
  addReasonUnless(outputWitnesses?.schemaVersion === 1
    && outputWitnesses?.kind === 'immediate-aif-phase0-output-byte-witnesses'
    && outputWitnesses?.pass === true
    && outputWitnesses?.encoding === 'base64-exact-bytes'
    && exactArray(outputWitnesses?.channels, channels)
    && outputWitnesses?.expectedChallengedCallbackCount === 126
    && outputWitnesses?.challengedCallbackCount === 126 && callbacks.length === 126
    && outputWitnesses?.expectedObservationCount === 384
    && outputWitnesses?.observationCount === 384
    && outputWitnesses?.witnessCount === witnessIds.length
    && exactArray(witnessIds, [...witnessIds].sort((left, right) => left.localeCompare(right)))
    && exactJson(outputWitnesses?.callbackReferences, references),
  reasons, 'output witness envelope/count/order or callback mapping is invalid');

  const decodedBytes = new Map();
  const colorStatistics = new Map();
  const depthStatistics = new Map();
  for (const witnessId of witnessIds) {
    const witness = witnesses[witnessId];
    const bytes = decodeCanonicalBase64(witness?.bytesBase64, witnessId, reasons);
    const expectedMetadata = {
      color: { format: 'rgba8unorm', arrayType: 'Uint8Array' },
      depth: { format: 'depth32float', arrayType: 'Float32Array' },
      objectId: { format: 'rgba8unorm-object-id-plus-one', arrayType: 'Uint8Array' },
    }[witness?.channel];
    addReasonUnless(expectedMetadata !== undefined
      && witness?.schemaVersion === 1
      && witness?.kind === 'immediate-aif-phase0-output-byte-witness'
      && witness?.witnessId === witnessId
      && witnessId === `${witness.channel}/${witness.sha256}`
      && witness?.format === expectedMetadata?.format
      && witness?.arrayType === expectedMetadata?.arrayType
      && witness?.byteLength === TARGET_BYTE_LENGTH
      && bytes?.byteLength === TARGET_BYTE_LENGTH
      && witness?.sha256 === (bytes === null ? null : sha256(bytes))
      && witness?.encoding === 'base64-exact-bytes'
      && (witness.channel === 'objectId' ? witness.decoded !== null : witness.decoded === null),
    reasons, `${witnessId} content-addressed raw output bytes/metadata are invalid`);
    if (bytes === null) continue;
    decodedBytes.set(witnessId, bytes);
    if (witness.channel === 'color') {
      const baselineRgba = Array.from(bytes.subarray(0, 4));
      let nonClearPixelCount = 0;
      for (let offset = 0; offset < bytes.length; offset += 4) {
        if (bytes[offset] !== baselineRgba[0] || bytes[offset + 1] !== baselineRgba[1]
          || bytes[offset + 2] !== baselineRgba[2]
          || bytes[offset + 3] !== baselineRgba[3]) nonClearPixelCount += 1;
      }
      colorStatistics.set(witnessId, { baselineRgba, nonClearPixelCount });
      addReasonUnless(bytes.some((value) => value !== 0)
        && bytes.some((value, index) => index % 4 === 3 && value !== 0)
        && nonClearPixelCount > 0,
      reasons, `${witnessId} color output contains no rendered coverage`);
    } else if (witness.channel === 'depth') {
      let nonzero = false;
      let zero = false;
      let finiteUnit = true;
      let zeroDepthCount = 0;
      let nonzeroFiniteDepthCount = 0;
      for (let offset = 0; offset < bytes.length; offset += 4) {
        const value = bytes.readFloatLE(offset);
        finiteUnit &&= Number.isFinite(value) && value >= 0 && value <= 1;
        nonzero ||= value > 0;
        zero ||= Object.is(value, 0);
        if (value === 0) zeroDepthCount += 1;
        else if (Number.isFinite(value)) nonzeroFiniteDepthCount += 1;
      }
      depthStatistics.set(witnessId, { zeroDepthCount, nonzeroFiniteDepthCount });
      addReasonUnless(finiteUnit && nonzero && zero,
        reasons, `${witnessId} depth32float bytes are not a finite rendered depth field`);
    }
  }

  const objectIdDecodings = new Map();
  for (const reference of references) {
    const witness = witnesses?.[reference.witnessId];
    addReasonUnless(witness?.channel === reference.channel
      && witness?.sha256 === reference.sha256
      && witness?.byteLength === reference.byteLength,
    reasons, `${reference.phase}/${reference.channel} does not map to its raw byte witness`);
    if (reference.channel === 'objectId' && decodedBytes.has(reference.witnessId)) {
      const cacheKey = `${reference.witnessId}/${reference.scenarioId}`;
      if (!objectIdDecodings.has(cacheKey)) {
        objectIdDecodings.set(cacheKey, independentlyDecodeObjectIdWitness(
          decodedBytes.get(reference.witnessId), reference.scenarioId,
        ));
      }
      const decoded = objectIdDecodings.get(cacheKey);
      addReasonUnless(decoded.pass === true
        && strictObjectIdAlphaEncoding(decodedBytes.get(reference.witnessId))
        && exactJson(witness.decoded, decoded),
        reasons, `${reference.witnessId} object-ID decoded validation is not independently exact`);
    }
    const callback = callbacks[reference.callbackOrdinal - 1];
    const output = reference.channel === 'objectId'
      ? callback?.output?.objectId : callback?.output?.[reference.channel];
    if (reference.channel === 'color') {
      const statistics = colorStatistics.get(reference.witnessId);
      addReasonUnless(output?.contentPass === true
        && exactArray(output?.baselineRgba, statistics?.baselineRgba)
        && output?.nonClearPixelCount === statistics?.nonClearPixelCount,
      reasons, `${reference.phase}/color content statistics do not match raw bytes`);
    } else if (reference.channel === 'depth') {
      const statistics = depthStatistics.get(reference.witnessId);
      addReasonUnless(output?.contentPass === true
        && output?.zeroDepthCount === statistics?.zeroDepthCount
        && output?.nonzeroFiniteDepthCount === statistics?.nonzeroFiniteDepthCount,
      reasons, `${reference.phase}/depth content statistics do not match raw bytes`);
    }
  }

  const expectedBaselineKeys = VISIBILITY_IDS.flatMap((scenarioId) => (
    IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.map((scheduleId) => `${scenarioId}/${scheduleId}`)
  ));
  addReasonUnless(exactArray(Object.keys(outputWitnesses?.objectIdBaselines ?? {}),
    expectedBaselineKeys), reasons, 'object-ID baseline key order is invalid');
  for (const groupId of expectedBaselineKeys) {
    const [scenarioId, scheduleId] = groupId.split('/');
    const laneWitnessIds = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
      lane,
      [...new Set(references.filter((record) => record.channel === 'objectId'
        && record.scenarioId === scenarioId && record.scheduleId === scheduleId
        && record.lane === lane).map((record) => record.witnessId))],
    ]));
    const baselineWitnessId = laneWitnessIds.A.length === 1 ? laneWitnessIds.A[0] : null;
    const expected = {
      scenarioId, scheduleId, baselineLane: 'A', baselineWitnessId, laneWitnessIds,
      exact: baselineWitnessId !== null && IMMEDIATE_AIF_PHASE0_LANES.every(
        (lane) => laneWitnessIds[lane].length === 1
          && laneWitnessIds[lane][0] === baselineWitnessId,
      ),
    };
    addReasonUnless(expected.exact && exactJson(outputWitnesses?.objectIdBaselines?.[groupId], expected),
      reasons, `${groupId} raw A/I/F object-ID baseline equality is invalid`);
  }

  const expectedObservations = [];
  const appendObservation = (output, context, capturePhase, observationKind, channel) => {
    expectedObservations.push({
      ordinal: expectedObservations.length + 1,
      observationId: `${capturePhase}/${channel}`,
      observationKind, capturePhase,
      scenarioId: context.scenarioId, scheduleId: context.scheduleId, lane: context.lane,
      channel, witnessId: output?.witnessId ?? null,
      sha256: output?.sha256 ?? null, byteLength: output?.byteLength ?? null,
    });
  };
  const primeRecords = scenarios?.[0]?.prime?.records;
  if (Array.isArray(primeRecords)) {
    for (const record of primeRecords) {
      const context = { scenarioId: 'v99', scheduleId: 'canonical', lane: record?.lane };
      for (const channel of ['color', 'depth']) {
        appendObservation(record?.callback?.output?.[channel], context,
          `phase0/prime/${record?.lane}`, 'bundle-prime', channel);
      }
      const primeColor = record?.callback?.output?.color;
      const primeDepth = record?.callback?.output?.depth;
      const colorStats = colorStatistics.get(primeColor?.witnessId);
      const depthStats = depthStatistics.get(primeDepth?.witnessId);
      addReasonUnless(record?.callback?.pass === true
        && primeColor?.contentPass === true
        && exactArray(primeColor?.baselineRgba, colorStats?.baselineRgba)
        && primeColor?.nonClearPixelCount === colorStats?.nonClearPixelCount
        && primeDepth?.contentPass === true
        && primeDepth?.zeroDepthCount === depthStats?.zeroDepthCount
        && primeDepth?.nonzeroFiniteDepthCount === depthStats?.nonzeroFiniteDepthCount,
      reasons, `v99 production-prime ${record?.lane} statistics do not match raw bytes`);
    }
  }
  for (const callback of callbacks) {
    for (const channel of ['color', 'depth']) {
      appendObservation(callback?.output?.[channel], callback,
        callback?.phases?.production, 'challenged-render-callback', channel);
    }
    appendObservation(callback?.output?.objectId, callback,
      callback?.phases?.objectId, 'challenged-render-callback', 'objectId');
  }
  addReasonUnless(exactJson(outputWitnesses?.observations, expectedObservations)
    && new Set(expectedObservations.map((record) => record.observationId)).size === 384
    && new Set(expectedObservations.map((record) => record.witnessId)).size === witnessIds.length
    && expectedObservations.every((record) => witnesses?.[record.witnessId]?.sha256 === record.sha256),
  reasons, 'raw output observation order, producer phase, or witness closure is invalid');
  for (const channel of ['color', 'depth']) {
    const challengedBaseline = references.find((record) => record.scenarioId === 'v99'
      && record.scheduleId === 'canonical' && record.lane === 'A'
      && record.channel === channel)?.witnessId;
    const primeWitnessIds = expectedObservations.filter(
      (record) => record.observationKind === 'bundle-prime' && record.channel === channel,
    ).map((record) => record.witnessId);
    addReasonUnless(primeWitnessIds.length === 3
      && challengedBaseline !== undefined
      && primeWitnessIds.every((witnessId) => witnessId === challengedBaseline),
    reasons, `v99 production-prime ${channel} bytes differ from canonical A/I/F output`);
  }

  for (const scenarioId of VISIBILITY_IDS) {
    const relevant = references.filter((reference) => reference.scenarioId === scenarioId);
    for (const scheduleId of IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS) {
      for (const channel of channels) {
        const ids = new Set(relevant.filter((record) => record.scheduleId === scheduleId
          && record.channel === channel).map((record) => record.witnessId));
        addReasonUnless(ids.size === 1,
          reasons, `${scenarioId}/${scheduleId}/${channel} raw callbacks are not byte-identical`);
      }
      const witnessIdFor = (channel) => relevant.find(
        (record) => record.scheduleId === scheduleId && record.channel === channel,
      )?.witnessId;
      addReasonUnless(productionOutputsCoverObjectIds(
        decodedBytes.get(witnessIdFor('color')),
        decodedBytes.get(witnessIdFor('depth')),
        decodedBytes.get(witnessIdFor('objectId')),
      ), reasons, `${scenarioId}/${scheduleId} production color/depth are all-clear at object coverage`);
    }
    for (const channel of channels) {
      const canonical = new Set(relevant.filter((record) => record.scheduleId === 'canonical'
        && record.channel === channel).map((record) => record.witnessId));
      addReasonUnless(canonical.size === 1,
        reasons, `${scenarioId} canonical restored/postflight ${channel} raw bytes changed`);
    }
  }
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function jsonObservationField(value, label) {
  const encoded = JSON.stringify(value);
  if (typeof encoded !== 'string') throw new TypeError(`${label} is not JSON-serializable.`);
  return encoded;
}

function expectedObservationChain(nonce, chainId, kind, chainEvidence) {
  return {
    schemaVersion: 1,
    kind,
    chainId,
    ...commitThreeImmediatePhase0ObservationFields(
      OBSERVATION_CHAIN_FIELD_NAMES,
      [OBSERVATION_CHAIN_DOMAIN, nonce, chainId,
        jsonObservationField(chainEvidence, `${chainId} chainEvidence`)],
    ),
  };
}

export function validateThreeImmediatePhase0ObservationChallenge(result, observationNonce) {
  const reasons = [];
  const evidence = result?.evidence;
  const challenge = evidence?.observationChallenge;
  const expectedChains = [];
  const validateCallbackChains = (callback, useId) => {
    const lane = callback?.lane;
    const shaderLane = evidence?.shaders?.lanes?.[lane];
    const definitions = [
      ['production', 'immediate-aif-phase0-production-observation-chain', {
        modules: shaderLane?.boundModules,
        pipeline: shaderLane?.boundPipeline,
        bundle: callback?.bundle,
        phases: callback?.phases,
        commandGpuBufferId: callback?.command?.gpuBufferId,
        visibleIdsGpuBufferId: callback?.address?.visibleIdsGpuBufferId,
        outputs: { color: callback?.output?.color, depth: callback?.output?.depth },
      }],
      ['address', 'immediate-aif-phase0-address-observation-chain', {
        producer: callback?.address?.producer,
        outputSha256: callback?.address?.sha256,
        phases: callback?.phases,
      }],
      ['objectId', 'immediate-aif-phase0-object-id-observation-chain', {
        producer: callback?.output?.objectId?.producer,
        outputSha256: callback?.output?.objectId?.sha256,
        phases: callback?.phases,
      }],
    ];
    addReasonUnless(callback?.pass === true
      && IMMEDIATE_AIF_PHASE0_LANES.includes(lane)
      && exactArray(Object.keys(callback?.observationChains ?? {}),
        definitions.map(([slot]) => slot)),
    reasons, `${useId} observation-chain container is invalid`);
    for (const [slot, kind, chainEvidence] of definitions) {
      const expected = expectedObservationChain(
        observationNonce, `${useId}/${slot}`, kind, chainEvidence,
      );
      expectedChains.push(expected);
      addReasonUnless(exactJson(callback?.observationChains?.[slot], expected),
        reasons, `${useId}/${slot} challenged chain commitment is invalid`);
    }
  };
  addReasonUnless(OBSERVATION_NONCE_PATTERN.test(observationNonce ?? '')
    && challenge?.nonce === observationNonce && challenge?.nonceEntropyBits === 128,
  reasons, 'observation challenge nonce is absent, malformed, or not the injected nonce');
  const scenarios = evidence?.scenarios;
  if (!Array.isArray(scenarios)) {
    reasons.push('observation challenge has no scenario traversal');
  } else {
    for (const scenario of scenarios) {
      for (const snapshot of scenario?.snapshots ?? []) {
        const snapshotId = `${scenario?.scenarioId}/${snapshot?.label}`;
        if (snapshot?.rerecord?.callback !== undefined) {
          validateCallbackChains(snapshot.rerecord.callback, `${snapshotId}/rerecord-I`);
        }
        if (snapshot?.laneOrders !== undefined) {
          for (const [orderIndex, order] of (snapshot.laneOrders?.results ?? []).entries()) {
            const orderId = order?.order?.join('') ?? '<invalid>';
            for (const [position, record] of (order?.records ?? []).entries()) {
              validateCallbackChains(record?.callback,
                `${snapshotId}/order-${orderIndex}-${orderId}/position-${position}-${record?.lane}`);
            }
          }
        } else {
          for (const callback of snapshot?.laneCaptures ?? []) {
            validateCallbackChains(callback, `${snapshotId}/audit-${callback?.lane}`);
          }
        }
      }
    }
  }
  for (const scenarioId of VISIBILITY_IDS) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      for (const phase of ['reset', 'cull']) {
        const collector = evidence?.shaders?.computeLanes?.[lane]?.phases?.[phase];
        const boundExecution = evidence?.shaders?.computeExecution
          ?.scenarios?.[scenarioId]?.lanes?.[lane]?.phases?.[phase];
        const { observationChain: retainedChain, ...unchallengedBoundExecution } =
          boundExecution ?? {};
        const chainId = `compute/${scenarioId}/${lane}/${phase}`;
        const expected = expectedObservationChain(
          observationNonce,
          chainId,
          `immediate-aif-phase0-compute-${phase}-observation-chain`,
          {
            collector: {
              rawSha256: collector?.rawSha256,
              normalizedSha256: collector?.normalizedSha256,
              normalization: collector?.normalization,
            },
            boundExecution: unchallengedBoundExecution,
          },
        );
        expectedChains.push(expected);
        addReasonUnless(exactJson(retainedChain, expected),
          reasons, `${chainId} challenged chain commitment is invalid`);
      }
    }
  }
  addReasonUnless(expectedChains.length === 390
    && new Set(expectedChains.map((chain) => chain.chainId)).size === 390
    && exactJson(challenge?.chains, expectedChains)
    && challenge?.chainCount === 390,
  reasons, 'observation challenge does not contain the exact ordered 390-use chain set');
  const modules = evidence?.shaders?.modules;
  addReasonUnless(Array.isArray(modules) && modules.length > 0
    && evidence?.shaders?.observationIdentityExact === true
    && modules.every((module, index) => module?.observationNonce === observationNonce
      && module?.creationOrdinal === index + 1
      && Number.isSafeInteger(module?.sequence)
      && (index === 0 || module.sequence > modules[index - 1].sequence)),
  reasons, 'shader-module creation records are not individually nonce-bound and ordered');
  let expectedRoot = null;
  try {
    const executionTraces = {
      renderBundles: evidence?.lifecycle?.renderBundleTraces,
      renderPasses: evidence?.lifecycle?.renderPassTraces,
      computePasses: evidence?.lifecycle?.computePassTraces,
      markers: evidence?.lifecycle?.instrumentationMarkers,
    };
    expectedRoot = {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-observation-challenge',
      nonce: observationNonce,
      nonceEntropyBits: 128,
      ...commitThreeImmediatePhase0ObservationFields(
        OBSERVATION_FIELD_NAMES,
        [
          OBSERVATION_DOMAIN,
          observationNonce,
          jsonObservationField(result?.target, 'target'),
          jsonObservationField(evidence?.shaders, 'shaders'),
          jsonObservationField(evidence?.pipelines, 'pipelines'),
          jsonObservationField(evidence?.commands, 'commands'),
          jsonObservationField(evidence?.scenarios, 'scenarios'),
          jsonObservationField(evidence?.addressWitnesses, 'addressWitnesses'),
          jsonObservationField(evidence?.outputWitnesses, 'outputWitnesses'),
          jsonObservationField(evidence?.resources, 'resources'),
          jsonObservationField(expectedChains, 'chains'),
          jsonObservationField(executionTraces, 'executionTraces'),
        ],
      ),
      chains: expectedChains,
      chainCount: expectedChains.length,
    };
  } catch (error) {
    reasons.push(`observation root reconstruction failed: ${error.message}`);
  }
  addReasonUnless(expectedRoot !== null && exactJson(challenge, expectedRoot),
    reasons, 'observation challenge root/field commitments are not exact');
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function validateLiveAndFreeze(scenario, scenarioId, reasons) {
  const live = scenario?.liveValidation;
  const expectedOrder = scenarioId === 'v99' ? ['I', 'F', 'A'] : ['A', 'F', 'I'];
  const records = live;
  addReasonUnless(exactArray(scenario?.validationOrder, expectedOrder)
    && Array.isArray(records) && records.length === 3
    && exactArray(records.map((record) => record?.lane), expectedOrder),
  reasons, `${scenarioId} live-validation envelope is invalid`);
  if (Array.isArray(records)) {
    for (let index = 0; index < 3; index += 1) {
      const lane = expectedOrder[index];
      const record = records[index];
      const values = record?.snapshot?.visibleIds;
      addReasonUnless(record?.lane === lane && record?.pass === true
        && record?.scheduleId === 'canonical'
        && validateFrozenMembership(values, scenarioId)
        && record?.correctness?.membership?.pass === true
        && record?.correctness?.membershipDigests?.pass === true
        && record?.correctness?.commandValidation?.pass === true
        && record?.correctness?.survivorSha256 === sha256U32(values)
        && exactArray(record?.snapshot?.commands,
          cpuOracles()[scenarioId].schedules.canonical[lane].commandWords)
        && record?.correctness?.commandSha256
          === cpuOracles()[scenarioId].schedules.canonical[lane].commandSha256
        && record?.snapshot?.overflow === 0,
      reasons, `${scenarioId} live lane ${lane} membership/commands are invalid`);
    }
  }
  const freeze = scenario?.freeze;
  const freezeValues = freeze?.visibleIds;
  addReasonUnless(freeze?.pass === true && freeze?.scenarioId === scenarioId
    && freeze?.source === 'deterministic-cpu-canonical-packing'
    && freeze?.sourceLane === null
    && freeze?.allLiveMembershipValidationsPass === true
    && freeze?.canonicalPackingExact === true
    && freeze?.valueCount === 65_536 && freeze?.byteLength === 262_144
    && exactArray(freezeValues, cpuOracles()[scenarioId].canonicalPacking)
    && freeze?.visibleIdsSha256 === sha256U32(freezeValues)
    && freeze.visibleIdsSha256 === cpuOracles()[scenarioId].canonicalPackingSha256
    && freeze?.visibleIdsCpuSha256 === freeze.visibleIdsSha256
    && Number.isSafeInteger(freeze?.visibleIdsAttributeId)
    && freeze.visibleIdsAttributeId >= 0
    && typeof freeze?.visibleIdsGpuBufferId === 'string'
    && freeze.visibleIdsGpuBufferId.length > 0,
  reasons, `${scenarioId} frozen survivor membership/identity is invalid`);
  return {
    visibleIds: freezeValues,
    sha256: freeze?.visibleIdsSha256,
    gpuBufferId: freeze?.visibleIdsGpuBufferId,
    attributeId: freeze?.visibleIdsAttributeId,
  };
}

function expectedCoverageCells() {
  return Object.fromEntries(VISIBILITY_IDS.flatMap((scenarioId) => (
    IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.flatMap((scheduleId) => (
      IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map(
        (order) => [`${scenarioId}/${scheduleId}/${order.join('')}`, 1],
      )
    ))
  )));
}

function validateRuntimeCoverage(coverage, reasons) {
  addReasonUnless(coverage?.complete === true
    && exactArray(coverage?.scenarioLoadSequence, VISIBILITY_IDS)
    && coverage?.expectedCellCount === 36
    && exactJson(coverage?.cells, expectedCoverageCells()),
  reasons, 'runtime coverage is not exactly 36 once-only cells with final canonical restore');
}

function validateScheduleTransition(snapshot, scheduleId, previousScheduleId,
  reasons, label) {
  const transition = snapshot?.transition;
  const changed = scheduleId !== previousScheduleId;
  addReasonUnless(transition?.schemaVersion === 1
    && transition?.kind === 'immediate-aif-phase0-schedule-transition'
    && transition?.pass === true
    && transition?.changed === changed
    && transition?.fromScheduleId === previousScheduleId
    && transition?.toScheduleId === scheduleId,
  reasons, `${label} schedule transition is invalid`);
  if (changed) {
    const before = transition?.before;
    const after = transition?.after;
    const expectedReplacement = scheduleId === 'S2';
    addReasonUnless(transition?.immediateRerecordRequired === true
      && transition?.immediateSourceWasDetached === expectedReplacement
      && transition?.immediateSourceReplaced === expectedReplacement
      && after?.aAttributeVersion === before?.aAttributeVersion + 1
      && after?.iBundleVersion === before?.iBundleVersion + 1
      && after?.fCommandVersion === before?.fCommandVersion + 1
      && exactJson(after?.bundleCounts, before?.bundleCounts),
    reasons, `${label} resource transition mutations are not exact`);
  }
}

function validateRenderPassDescriptorsAndEvents(evidence, reasons) {
  const lifecycle = evidence?.lifecycle;
  const resources = evidence?.resources;
  const common = resources?.commonResources;
  const traces = lifecycle?.renderPassTraces;
  if (!Array.isArray(traces)) {
    reasons.push('render-pass trace inventory is absent');
    return;
  }
  const phaseTargets = new Map();
  const addPhase = (phase, targetName) => {
    if (typeof phase !== 'string' || phase.length === 0 || phaseTargets.has(phase)) {
      reasons.push(`render phase is absent or duplicated: ${String(phase)}`);
    } else phaseTargets.set(phase, targetName);
  };
  for (const entry of expectedPhase0CallbackTraversal(evidence?.scenarios, reasons)) {
    addPhase(`${entry.phase}/production`, 'production');
    addPhase(`${entry.phase}/address`, 'address');
    addPhase(`${entry.phase}/object-id`, 'objectId');
  }
  for (const lane of ['A', 'F', 'I']) addPhase(`phase0/prime/${lane}`, 'production');
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    addPhase(`phase0/diagnostic-prime/${lane}/address`, 'address');
    addPhase(`phase0/diagnostic-prime/${lane}/object-id`, 'objectId');
  }
  for (const scenarioId of VISIBILITY_IDS) {
    for (const scheduleId of ['S1', 'S2', 'canonical']) {
      addPhase(`phase0/diagnostic-rerecord/${scenarioId}/${scheduleId}/I/address`, 'address');
      addPhase(`phase0/diagnostic-rerecord/${scenarioId}/${scheduleId}/I/object-id`, 'objectId');
    }
  }
  const byPhase = new Map();
  for (const trace of traces) {
    const entries = byPhase.get(trace?.capturePhase) ?? [];
    entries.push(trace);
    byPhase.set(trace?.capturePhase, entries);
  }
  addReasonUnless(phaseTargets.size === 399 && traces.length === 798
    && byPhase.size === phaseTargets.size
    && [...byPhase.keys()].every((phase) => phaseTargets.has(phase)),
  reasons, 'render passes include missing, extra, duplicated, or unclassified phases');

  const clearByTarget = {
    production: {
      r: PINNED_PRODUCTION_CLEAR_LINEAR_COLOR[0],
      g: PINNED_PRODUCTION_CLEAR_LINEAR_COLOR[1],
      b: PINNED_PRODUCTION_CLEAR_LINEAR_COLOR[2],
      a: 1,
    },
    objectId: { r: 0, g: 0, b: 0, a: 0 },
    address: { r: 0, g: 0, b: 0, a: 0 },
  };
  const defaultUnusedClear = { r: 0, g: 0, b: 0, a: 1 };
  const views = resources?.textureViews ?? [];
  const pinnedConfiguration = pinnedRenderConfigurationExact(common?.expected)
    ? common.expected : null;
  for (const [phase, targetName] of phaseTargets) {
    const pair = (byPhase.get(phase) ?? []).sort((left, right) => left.sequence - right.sequence);
    const [clearTrace, renderTrace] = pair;
    const target = common?.targets?.[targetName];
    const targetExpected = pinnedConfiguration?.targets?.[targetName];
    const validateAttachment = (trace, role, loadOp, clearValue) => {
      const attachment = trace?.colorAttachments?.[0];
      const matchingViews = views.filter((view) => view?.viewId === attachment?.viewId
        && view?.textureId === target?.colorTextureId);
      const descriptorDefaults = validateThreeImmediatePhase0RenderPassDescriptorDefaults(
        trace, { role },
      );
      return descriptorDefaults.valid
        && trace?.occlusionQuerySetId === null && trace?.timestampWrites === null
        && trace?.colorAttachments?.length === 1
        && attachment?.textureId === target?.colorTextureId
        && typeof attachment?.viewId === 'string' && matchingViews.length === 1
        && attachment?.resolveTargetViewId === null
        && attachment?.resolveTargetTextureId === null
        && attachment?.depthSlice === null
        && attachment?.loadOp === loadOp && attachment?.storeOp === 'store'
        && exactJson(attachment?.clearValue, clearValue);
    };
    const validateDepth = (trace, loadOp) => {
      const depth = trace?.depthStencilAttachment;
      if (targetExpected?.depthBuffer !== true) return depth === null;
      const expectedDepthId = target?.depthTextureIds?.[0];
      const matchingViews = views.filter((view) => view?.viewId === depth?.viewId
        && view?.textureId === expectedDepthId);
      const descriptorDefaults = validateThreeImmediatePhase0DepthStencilAttachmentDefaults(
        depth, { loadOp },
      );
      return descriptorDefaults.valid
        && depth?.textureId === expectedDepthId
        && typeof depth?.viewId === 'string' && matchingViews.length === 1;
    };
    const clearMethods = clearTrace?.events?.map((event) => event?.method);
    const expectedRenderMethods = targetName === 'address'
      ? ['setViewport', 'executeBundles', 'end'] : ['executeBundles', 'end'];
    const renderMethods = renderTrace?.events?.map((event) => event?.method);
    const viewport = renderTrace?.events?.find((event) => event?.method === 'setViewport');
    const execute = renderTrace?.events?.find((event) => event?.method === 'executeBundles');
    addReasonUnless(pair.length === 2
      && Number.isSafeInteger(clearTrace?.sequence)
      && Number.isSafeInteger(renderTrace?.sequence)
      && clearTrace.sequence < renderTrace.sequence
      && clearTrace?.commandEncoderId !== renderTrace?.commandEncoderId
      && validateAttachment(clearTrace, 'clear', 'clear', clearByTarget[targetName])
      && validateDepth(clearTrace, 'clear')
      && validateAttachment(renderTrace, 'render', 'load', defaultUnusedClear)
      && validateDepth(renderTrace, 'load')
      && exactArray(clearMethods, ['end'])
      && exactArray(renderMethods, expectedRenderMethods)
      && (targetName !== 'address' || exactJson(viewport?.viewport, {
        x: 0, y: 0, width: targetExpected.width, height: targetExpected.height,
        minDepth: 0, maxDepth: 1,
      }))
      && Array.isArray(execute?.bundleIds) && execute.bundleIds.length === 1
      && ![clearTrace, renderTrace].some((trace) => trace?.events?.some((event) => [
        'setScissorRect', 'setBlendConstant', 'setStencilReference',
        'draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect', 'setImmediates',
      ].includes(event?.method)))
      && clearTrace?.events?.[0]?.sequence > clearTrace.sequence
      && renderTrace?.events?.[0]?.sequence > renderTrace.sequence
      && clearTrace?.events?.at(-1)?.method === 'end'
      && renderTrace?.events?.at(-1)?.method === 'end',
    reasons, `${phase} clear/render pass descriptors or terminal event grammar are invalid`);
  }
  const terminationRecords = traces.map((trace) => ({
    encoderId: trace?.encoderId,
    capturePhase: trace?.capturePhase,
    endCount: trace?.events?.filter((event) => event?.method === 'end').length,
    endSequence: trace?.events?.find((event) => event?.method === 'end')?.sequence ?? null,
    terminalSequence: trace?.events?.at(-1)?.sequence ?? null,
  }));
  addReasonUnless(resources?.renderPassTermination?.schemaVersion === 1
    && resources?.renderPassTermination?.kind
      === 'immediate-aif-phase0-render-pass-terminal-end-evidence'
    && resources?.renderPassTermination?.pass === true
    && resources?.renderPassTermination?.passCount === traces.length
    && exactJson(resources?.renderPassTermination?.records, terminationRecords),
  reasons, 'render-pass terminal end evidence does not reconstruct from raw traces');
}

export function validateThreeImmediatePhase0RenderPassDescriptorDefaults(
  trace, { role } = {},
) {
  const reasons = [];
  const exact = role === 'clear'
    ? trace?.label === null && trace?.maxDrawCount === null
    : role === 'render'
      ? trace?.label === '' && trace?.maxDrawCount === 50_000_000
      : false;
  addReasonUnless(exact, reasons,
    `${String(role)} render-pass descriptor label/maxDrawCount is not exact`);
  return { valid: reasons.length === 0, reasons };
}

export function validateThreeImmediatePhase0DepthStencilAttachmentDefaults(
  attachment, { loadOp } = {},
) {
  const reasons = [];
  const exact = (loadOp === 'clear' || loadOp === 'load')
    && attachment?.depthLoadOp === loadOp
    && attachment?.depthStoreOp === 'store'
    && attachment?.depthClearValue === 0
    && attachment?.depthReadOnly === false
    && attachment?.stencilLoadOp === null
    && attachment?.stencilStoreOp === null
    && attachment?.stencilClearValue === 0
    && attachment?.stencilReadOnly === false;
  addReasonUnless(exact, reasons,
    'non-stencil depth attachment does not preserve the exact Three descriptor defaults');
  return { valid: reasons.length === 0, reasons };
}

function validateDetachment(detachment, scenarioId, reasons) {
  addReasonUnless(detachment?.schemaVersion === 1
    && detachment?.kind === 'immediate-aif-phase0-post-finish-source-detachment'
    && detachment?.pass === true && detachment?.scenarioId === scenarioId
    && detachment?.scheduleId === 'S1'
    && detachment?.detachedDuringImmediateRerecord === true
    && exactArray(detachment?.recordedBases,
      IMMEDIATE_AIF_PHASE0_SCHEDULES.S1.sourceBaseByDraw)
    && Array.isArray(detachment?.mutatedBases)
    && exactArray(detachment.mutatedBases,
      IMMEDIATE_AIF_PHASE0_SCHEDULES.S2.sourceBaseByDraw)
    && detachment?.detachedByteLength === 0
    && detachment?.detachedLength === 0
    && Number.isSafeInteger(detachment?.instrumentationSequence)
    && detachment?.marker?.sequence === detachment.instrumentationSequence
    && detachment.marker?.kind === 'phase0-immediate-source-detached'
    && detachment.marker?.detail?.scenarioId === scenarioId
    && detachment.marker?.detail?.scheduleId === 'S1'
    && detachment.marker?.detail?.detachedSourceId === detachment.detachedSourceId
    && detachment.marker?.detail?.detachedBackingBufferId
      === detachment.detachedBackingBufferId
    && detachment.marker?.detail?.cachedBundleGpuId === detachment.cachedBundleGpuId
    && detachment.marker?.detail?.bundleFinishSequence === detachment.bundleFinishSequence,
  reasons, `${scenarioId} I immediate source was not mutated and detached after finish`);
}

function validateImmediateRerecord(rerecord, {
  scenarioId, scheduleId, frozen, identities, pipelineIds, pipelines, shaders,
  resources, lifecycle, label,
}, reasons) {
  const before = rerecord?.before;
  const after = rerecord?.after;
  addReasonUnless(rerecord?.schemaVersion === 1
    && rerecord?.kind === 'immediate-aif-phase0-immediate-bundle-rerecord'
    && rerecord?.pass === true && rerecord?.lane === 'I'
    && rerecord?.scheduleId === scheduleId
    && rerecord?.sameBundleIdentity === true
    && rerecord?.sameBundleVersion === true
    && after?.I === before?.I + 1
    && after?.A === before?.A && after?.F === before?.F,
  reasons, `${label} I bundle re-record counters/identity are invalid`);
  const summary = validateCallback(rerecord?.callback, {
    scenarioId, scheduleId, lane: 'I', frozen, pipelineIds, pipelines, shaders,
    resources, lifecycle,
    label: `${label}/rerecord-I`,
  }, reasons);
  addReasonUnless(identities.commandBufferIds.I === summary.commandBufferId
    && identities.rootUuids.I === summary.rootUuid,
  reasons, `${label} I re-record changed its persistent command/root identity`);
  return summary;
}

function validateOrderResults(laneOrders, scenarioId, scheduleId, frozen,
  identities, pipelineIds, pipelines, shaders, resources, lifecycle, reasons, label,
  expectedImmediateSourceId = null) {
  addReasonUnless(laneOrders?.pass === true
    && Array.isArray(laneOrders?.results)
    && exactJson(laneOrders.results.map((result) => result?.order),
      IMMEDIATE_AIF_PHASE0_LANE_ORDERS),
  reasons, `${label} six-order coverage is invalid`);
  const summaries = [];
  if (!Array.isArray(laneOrders?.results)) return summaries;
  for (let orderIndex = 0; orderIndex < 6; orderIndex += 1) {
    const order = IMMEDIATE_AIF_PHASE0_LANE_ORDERS[orderIndex];
    const orderResult = laneOrders.results[orderIndex];
    const orderLabel = `${label}/${order.join('')}`;
    addReasonUnless(orderResult?.pass === true
      && orderResult?.scenarioId === scenarioId
      && orderResult?.scheduleId === scheduleId
      && exactArray(orderResult?.order, order)
      && Array.isArray(orderResult?.records)
      && exactArray(orderResult.records.map((record) => record?.lane), order),
    reasons, `${orderLabel} record is invalid`);
    if (!Array.isArray(orderResult?.records)) continue;
    for (let laneIndex = 0; laneIndex < 3; laneIndex += 1) {
      const lane = order[laneIndex];
      const record = orderResult.records[laneIndex];
      addReasonUnless(record?.lane === lane && record?.pass === true
        && record?.bundleStable === true,
      reasons, `${orderLabel}/${lane} runtime record failed`);
      const summary = validateCallback(record?.callback, {
        scenarioId, scheduleId, lane, frozen, pipelineIds, pipelines, shaders,
        resources, lifecycle,
        expectedImmediateSourceId: lane === 'I' ? expectedImmediateSourceId : null,
        label: `${orderLabel}/${lane}`,
      }, reasons);
      summaries.push({ lane, ...summary });
      identities.commandBufferIds[lane] ??= summary.commandBufferId;
      identities.rootUuids[lane] ??= summary.rootUuid;
      addReasonUnless(identities.commandBufferIds[lane] === summary.commandBufferId
        && identities.rootUuids[lane] === summary.rootUuid,
      reasons, `${scenarioId} lane ${lane} resource identity changed`);
    }
  }
  addReasonUnless(summaries.length === 18
    && summaries.every((summary) => exactJson(summary.targetHashes,
      summaries[0]?.targetHashes)),
  reasons, `${label} A/I/F raw targets are not byte-identical in all six orders`);
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const laneSummaries = summaries.filter((summary) => summary.lane === lane);
    addReasonUnless(laneSummaries.length === 6
      && laneSummaries.every((summary) => summary.bundleGpuId
        === laneSummaries[0].bundleGpuId)
      && laneSummaries.every((summary) => summary.pipelineId
        === laneSummaries[0].pipelineId),
    reasons, `${label} lane ${lane} actual GPU bundle/pipeline identity is unstable`);
  }
  addReasonUnless(new Set(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => (
    summaries.find((summary) => summary.lane === lane)?.bundleGpuId
  ))).size === 3,
  reasons, `${label} A/I/F actual GPU bundle identities are not distinct`);
  return summaries;
}

function validateLaneCaptures(captures, scenarioId, scheduleId, frozen,
  identities, pipelineIds, pipelines, shaders, resources, lifecycle, reasons, label,
  expectedImmediateSourceId = null) {
  addReasonUnless(Array.isArray(captures) && captures.length === 3
    && exactArray(captures.map((record) => record?.lane), IMMEDIATE_AIF_PHASE0_LANES),
  reasons, `${label} lane captures are not exactly A/I/F`);
  if (!Array.isArray(captures)) return [];
  const summaries = captures.map((record, index) => {
    const lane = IMMEDIATE_AIF_PHASE0_LANES[index];
    addReasonUnless(record?.pass === true && record?.lane === lane,
      reasons, `${label}/${lane} capture failed`);
    const summary = validateCallback(record?.callback ?? record, {
      scenarioId, scheduleId, lane, frozen, pipelineIds, pipelines, shaders,
      resources, lifecycle,
      expectedImmediateSourceId: lane === 'I' ? expectedImmediateSourceId : null,
      label: `${label}/${lane}`,
    }, reasons);
    addReasonUnless(identities.commandBufferIds[lane] === summary.commandBufferId
      && identities.rootUuids[lane] === summary.rootUuid,
    reasons, `${label}/${lane} resource identity changed`);
    return { lane, ...summary };
  });
  addReasonUnless(new Set(summaries.map((summary) => summary.bundleGpuId)).size === 3,
    reasons, `${label} A/I/F actual GPU bundle identities are not distinct`);
  addReasonUnless(summaries.length === 3
    && summaries.every((summary) => exactJson(summary.targetHashes,
      summaries[0]?.targetHashes)),
  reasons, `${label} A/I/F raw targets are not byte-identical`);
  return summaries;
}

function validateScenarioEvidence(
  scenarios, shaders, pipelines, resources, lifecycle, reasons,
) {
  addReasonUnless(Array.isArray(scenarios)
    && exactArray(scenarios.map((scenario) => scenario?.scenarioId), VISIBILITY_IDS),
  reasons, 'scenario array is not exactly v99 then v20');
  if (!Array.isArray(scenarios)) return;
  const pipelineIds = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
    (lane) => [lane, shaders?.lanes?.[lane]?.pipelineId],
  ));
  const identities = { commandBufferIds: {}, rootUuids: {} };
  const allSummaries = [];
  let priorScenarioFinalImmediate = null;
  for (let scenarioIndex = 0; scenarioIndex < 2; scenarioIndex += 1) {
    const scenarioId = VISIBILITY_IDS[scenarioIndex];
    const scenario = scenarios[scenarioIndex];
    if (scenario?.scenarioId !== scenarioId) continue;
    const scenarioResources = {
      ...resources,
      gpuResourceCommitments: scenario?.gpuResourceCommitments,
    };
    const expectedLevel = IMMEDIATE_AIF_PHASE0_WORKLOAD.visibilityLevels.find(
      (level) => level.id === scenarioId,
    );
    addReasonUnless(scenario?.schemaVersion === 1
      && scenario?.kind === 'immediate-aif-phase0-scenario-result'
      && scenario?.pass === true
      && scenario?.load?.schemaVersion === 1
      && scenario?.load?.kind === 'immediate-aif-phase0-scenario-load'
      && scenario?.load?.scenarioId === scenarioId
      && scenario?.load?.visibilityFraction === expectedLevel?.fraction
      && scenario?.load?.expectedVisibleCount === expectedLevel?.expectedVisibleCount,
    reasons, `${scenarioId} scenario/load envelope is invalid`);
    const frozen = validateLiveAndFreeze(scenario, scenarioId, reasons);
    addReasonUnless(Array.isArray(scenario?.snapshots)
      && exactArray(scenario.snapshots.map((snapshot) => snapshot?.label), SNAPSHOT_LABELS)
      && exactArray(scenario.snapshots.map((snapshot) => snapshot?.scheduleId),
        SNAPSHOT_SCHEDULES),
    reasons, `${scenarioId} snapshot order is incomplete`);
    if (!Array.isArray(scenario?.snapshots)) continue;
    const scheduleHashes = {};
    const immediateSchedule = {};
    let scenarioDetachment = null;
    let previousSchedule = 'canonical';
    for (let snapshotIndex = 0; snapshotIndex < 5; snapshotIndex += 1) {
      const snapshot = scenario.snapshots[snapshotIndex];
      const scheduleId = SNAPSHOT_SCHEDULES[snapshotIndex];
      const label = `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]}`;
      validateScheduleTransition(snapshot, scheduleId, previousSchedule, reasons, label);
      previousSchedule = scheduleId;
      let rerecordSummary = null;
      if (scheduleId !== 'canonical' || snapshotIndex === 3) {
        rerecordSummary = validateImmediateRerecord(snapshot?.rerecord, {
          scenarioId, scheduleId, frozen, identities, pipelineIds, pipelines, shaders,
          resources: scenarioResources, lifecycle, label,
        }, reasons);
      } else {
        addReasonUnless(snapshot?.rerecord === null,
          reasons, `${label} has an unexpected I bundle re-record`);
      }
      let summaries;
      if (snapshotIndex < 3) {
        addReasonUnless(snapshot?.laneCaptures === undefined,
          reasons, `${label} unexpectedly duplicates lane captures`);
        summaries = validateOrderResults(snapshot?.laneOrders, scenarioId,
          scheduleId, frozen, identities, pipelineIds, pipelines, shaders,
          scenarioResources, lifecycle, reasons, label,
          rerecordSummary?.immediateSourceId ?? null);
      } else {
        addReasonUnless(snapshot?.laneOrders === undefined,
          reasons, `${label} illegally repeated runtime coverage cells`);
        summaries = validateLaneCaptures(snapshot?.laneCaptures, scenarioId,
          scheduleId, frozen, identities, pipelineIds, pipelines, shaders,
          scenarioResources, lifecycle, reasons, label,
          rerecordSummary?.immediateSourceId
            ?? immediateSchedule[SNAPSHOT_LABELS[snapshotIndex - 1]]?.immediateSourceId
            ?? null);
      }
      allSummaries.push(...summaries);
      const immediate = summaries.find((summary) => summary.lane === 'I') ?? null;
      immediateSchedule[SNAPSHOT_LABELS[snapshotIndex]] = immediate;
      if (rerecordSummary !== null) {
        addReasonUnless(immediate !== null
          && rerecordSummary.bundleGpuId === immediate.bundleGpuId
          && rerecordSummary.immediateSourceId === immediate.immediateSourceId
          && rerecordSummary.immediateBackingBufferId
            === immediate.immediateBackingBufferId
          && exactJson(rerecordSummary.targetHashes, immediate.targetHashes),
        reasons, `${label} cached I bundle/output differs from its re-record callback`);
      }
      if (snapshotIndex === 1) {
        const detachment = snapshot?.sourceDetachment;
        scenarioDetachment = detachment;
        validateDetachment(detachment, scenarioId, reasons);
        const setSequences = detachment?.setImmediatesSequences;
        addReasonUnless(rerecordSummary !== null
          && detachment?.temporalOrderExact === true
          && detachment?.allSetImmediatesBeforeFinish === true
          && detachment?.finishBeforeDetach === true
          && detachment?.cachedBundleGpuId === rerecordSummary.bundleGpuId
          && detachment?.detachedSourceId === rerecordSummary.immediateSourceId
          && detachment?.detachedBackingBufferId
            === rerecordSummary.immediateBackingBufferId
          && detachment?.bundleFinishSequence === rerecordSummary.finishSequence
          && Array.isArray(setSequences) && setSequences.length === 32
          && setSequences.every((sequence, index) => Number.isSafeInteger(sequence)
            && (index === 0 || sequence > setSequences[index - 1])
            && sequence < rerecordSummary.finishSequence)
          && rerecordSummary.finishSequence < detachment?.instrumentationSequence
          && detachment.instrumentationSequence < rerecordSummary.executeSequence
          && detachment?.firstExecuteSequence === rerecordSummary.executeSequence
          && exactArray(detachment?.postDetachExecuteSequences,
            summaries.filter((summary) => summary.lane === 'I')
              .map((summary) => summary.executeSequence))
          && detachment.postDetachExecuteSequences.every(
            (sequence) => sequence > detachment.instrumentationSequence,
          ),
        reasons, `${label} finish/detach/cached-execution temporal chain is invalid`);
      } else {
        addReasonUnless(snapshot?.sourceDetachment === null,
          reasons, `${label} has an unexpected immediate-source detachment`);
      }
      if (snapshotIndex === 2) {
        const transition = snapshot?.transition;
        addReasonUnless(transition?.replacementSourceExact === true
          && transition?.detachedSourceId === scenarioDetachment?.detachedSourceId
          && transition?.detachedBackingBufferId
            === scenarioDetachment?.detachedBackingBufferId
          && transition?.replacementSourceId === immediate?.immediateSourceId
          && transition?.replacementBackingBufferId
            === immediate?.immediateBackingBufferId
          && transition?.replacementSourceId !== transition?.detachedSourceId
          && transition?.replacementBackingBufferId
            !== transition?.detachedBackingBufferId
          && transition?.replacementSourceSnapshot?.sourceId
            === transition?.replacementSourceId
          && transition?.replacementSourceSnapshot?.backingBufferId
            === transition?.replacementBackingBufferId
          && exactArray(transition?.replacementSourceSnapshot?.recordedValues,
            IMMEDIATE_AIF_PHASE0_SCHEDULES.S2.sourceBaseByDraw),
        reasons, `${label} detached S1 source was not replaced exactly for S2`);
      }
      const hashes = summaries[0]?.targetHashes ?? null;
      if (snapshotIndex === 0) scheduleHashes.canonicalPreflight = hashes;
      if (snapshotIndex === 1) scheduleHashes.S1 = hashes;
      if (snapshotIndex === 2) scheduleHashes.S2 = hashes;
      if (snapshotIndex === 3) scheduleHashes.canonicalRestored = hashes;
      if (snapshotIndex === 4) scheduleHashes.canonicalPostflight = hashes;
    }
    addReasonUnless(exactJson(scheduleHashes.canonicalPreflight,
      scheduleHashes.canonicalRestored)
      && exactJson(scheduleHashes.canonicalPreflight,
        scheduleHashes.canonicalPostflight),
    reasons, `${scenarioId} canonical output did not restore exactly`);
    addReasonUnless(scheduleHashes.S1?.address
      !== scheduleHashes.canonicalPreflight?.address
      && scheduleHashes.S2?.address !== scheduleHashes.canonicalPreflight?.address
      && scheduleHashes.S1?.address !== scheduleHashes.S2?.address,
    reasons, `${scenarioId} sentinel schedules did not produce three distinct address oracles`);
    const preflightI = immediateSchedule['canonical-preflight'];
    const s1I = immediateSchedule.S1;
    const s2I = immediateSchedule.S2;
    const restoredI = immediateSchedule['canonical-restored'];
    const postflightI = immediateSchedule['canonical-postflight'];
    addReasonUnless(preflightI !== null && s1I !== null && s2I !== null
      && restoredI !== null && postflightI !== null
      && new Set([preflightI.bundleGpuId, s1I.bundleGpuId,
        s2I.bundleGpuId, restoredI.bundleGpuId]).size === 4
      && postflightI.bundleGpuId === restoredI.bundleGpuId
      && preflightI.immediateSourceId === s1I.immediateSourceId
      && preflightI.immediateBackingBufferId === s1I.immediateBackingBufferId
      && s2I.immediateSourceId !== s1I.immediateSourceId
      && s2I.immediateBackingBufferId !== s1I.immediateBackingBufferId
      && restoredI.immediateSourceId === s2I.immediateSourceId
      && restoredI.immediateBackingBufferId === s2I.immediateBackingBufferId
      && postflightI.immediateSourceId === restoredI.immediateSourceId
      && postflightI.immediateBackingBufferId === restoredI.immediateBackingBufferId,
    reasons, `${scenarioId} I cached-bundle/source replacement chain is invalid`);
    if (scenarioIndex === 1) {
      addReasonUnless(priorScenarioFinalImmediate?.bundleGpuId === preflightI?.bundleGpuId
        && priorScenarioFinalImmediate?.immediateSourceId === preflightI?.immediateSourceId
        && priorScenarioFinalImmediate?.immediateBackingBufferId
          === preflightI?.immediateBackingBufferId,
      reasons, 'v20 did not begin from the exact restored v99 cached I bundle/source');
    }
    priorScenarioFinalImmediate = postflightI;
  }
  for (const lane of ['A', 'F']) {
    const records = allSummaries.filter((summary) => summary.lane === lane);
    addReasonUnless(records.length === 40
      && records.every((record) => record.bundleGpuId === records[0]?.bundleGpuId)
      && records.every((record) => record.bundleGpuId === shaders?.lanes?.[lane]?.bundleGpuId)
      && records.every((record) => record.pipelineId === pipelineIds[lane]),
    reasons, `lane ${lane} production bundle/pipeline was not globally stable`);
  }
  addReasonUnless(pipelineIds.I === allSummaries.find(
    (summary) => summary.lane === 'I',
  )?.pipelineId,
  reasons, 'lane I production pipeline is not bound to callback traces');
}

function validateTopology(topology, reasons) {
  addReasonUnless(topology?.pass === true
    && topology?.configuredDrawCommands === 32
    && topology?.configuredRenderObjectsPerLane === 1
    && topology?.configuredComputeDispatchesPerLane === 2
    && topology?.sourceGeometryCount === 32
    && topology?.target?.width === TARGET_WIDTH
    && topology?.target?.height === TARGET_HEIGHT
    && topology?.target?.samples === 0
    && topology?.target?.depthBuffer === true
    && topology?.target?.reversedDepthBuffer === true
    && topology?.activeRootCount === 1 && topology?.activeLane === 'F'
    && topology?.frozenVisibleIds === true
    && topology?.state?.phase === 'ready'
    && topology.state?.scenarioId === 'v20'
    && topology.state?.activeScheduleId === 'canonical'
    && topology.state?.immediateRecordedScheduleId === 'canonical'
    && topology.state?.immediateRerecordRequired === false
    && topology.state?.runtimeCoverageComplete === true
    && topology.state?.failure === null
    && exactArray(topology?.lanes, IMMEDIATE_AIF_PHASE0_LANES)
    && topology?.bucketCount === 32 && topology?.drawCount === 32
    && exactJson(topology?.bucketBaseVertexStreams, { A: true, I: false, F: false })
    && new Set(Object.values(topology?.commandBufferIds ?? {})).size === 3
    && Object.values(topology?.commandBufferByteLengths ?? {})
      .every((byteLength) => byteLength === 640)
    && new Set(Object.values(topology?.geometryUuids ?? {})).size === 3
    && topology?.addressDiagnostic?.pass === true
    && topology.addressDiagnostic?.productionBucketBaseStreamReused === true
    && topology.addressDiagnostic?.addedBucketBaseStreams === 0
    && exactJson(topology.addressDiagnostic?.bucketBaseAttributes,
      { A: topology.addressDiagnostic?.bucketBaseAttributes?.A, I: null, F: null })
    && typeof topology.addressDiagnostic?.bucketBaseAttributes?.A === 'number'
    && topology.addressDiagnostic?.indexSharedAcrossLanes === true,
  reasons, 'A/I/F runtime topology is not exact');
}

function scenarioCallbacks(scenario) {
  const callbacks = [];
  for (const snapshot of scenario?.snapshots ?? []) {
    if (snapshot?.rerecord?.callback !== undefined) callbacks.push(snapshot.rerecord.callback);
    if (snapshot?.laneOrders !== undefined) {
      for (const order of snapshot.laneOrders?.results ?? []) {
        callbacks.push(...(order?.records ?? []).map((record) => record?.callback));
      }
    } else callbacks.push(...(snapshot?.laneCaptures ?? []));
  }
  return callbacks;
}

/**
 * Reconstructs the only permitted 126-callback traversal without consulting
 * callback-authored phase strings or challenge-authored phase lists.
 */
function expectedPhase0CallbackTraversal(scenarios, reasons = null) {
  const entries = [];
  let serial = 1;
  const report = (condition, message) => {
    if (!condition && Array.isArray(reasons)) reasons.push(message);
  };
  for (let scenarioIndex = 0; scenarioIndex < VISIBILITY_IDS.length; scenarioIndex += 1) {
    const scenarioId = VISIBILITY_IDS[scenarioIndex];
    const scenario = scenarios?.[scenarioIndex];
    report(scenario?.scenarioId === scenarioId,
      `callback traversal scenario ${scenarioIndex} is not ${scenarioId}`);
    for (let snapshotIndex = 0; snapshotIndex < SNAPSHOT_LABELS.length; snapshotIndex += 1) {
      const snapshot = scenario?.snapshots?.[snapshotIndex];
      const scheduleId = SNAPSHOT_SCHEDULES[snapshotIndex];
      const installationOrdinal = scenarioIndex * 4 + Math.min(snapshotIndex + 1, 4);
      const installationId = `${scenarioId}/${installationOrdinal}/${scheduleId}`;
      const push = (callback, lane, kind, orderIndex = null, position = null) => {
        const phase = `phase0/${scenarioId}/${scheduleId}/${lane}/capture-${serial}`;
        entries.push({
          callback, scenario, snapshot, scenarioId, scenarioIndex,
          snapshotIndex, snapshotLabel: SNAPSHOT_LABELS[snapshotIndex], scheduleId,
          installationOrdinal, installationId, lane, kind, orderIndex, position,
          serial, phase,
        });
        report(callback !== null && typeof callback === 'object',
          `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]}/${kind} callback is absent`);
        report(callback?.lane === lane,
          `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]}/${kind} lane is not ${lane}`);
        report(callback?.phase === phase,
          `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]}/${kind} phase is not ${phase}`);
        report(callback?.scheduleInstallationId === installationId,
          `${phase} is not bound to installation ${installationId}`);
        serial += 1;
      };
      const expectsRerecord = snapshotIndex === 1
        || snapshotIndex === 2 || snapshotIndex === 3;
      report(expectsRerecord
        ? snapshot?.rerecord?.callback !== undefined && snapshot?.rerecord?.callback !== null
        : snapshot?.rerecord === null,
      `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]} has an invalid re-record presence`);
      if (expectsRerecord) push(snapshot?.rerecord?.callback, 'I', 'rerecord');
      if (snapshotIndex < 3) {
        const results = snapshot?.laneOrders?.results;
        report(Array.isArray(results) && results.length === 6,
          `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]} lacks six ordered results`);
        for (let orderIndex = 0; orderIndex < 6; orderIndex += 1) {
          const order = IMMEDIATE_AIF_PHASE0_LANE_ORDERS[orderIndex];
          const result = results?.[orderIndex];
          report(exactArray(result?.order, order),
            `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]}/order-${orderIndex} differs`);
          for (let position = 0; position < 3; position += 1) {
            const record = result?.records?.[position];
            push(record?.callback, order[position], 'order', orderIndex, position);
          }
        }
      } else {
        const captures = snapshot?.laneCaptures;
        report(Array.isArray(captures) && captures.length === 3,
          `${scenarioId}/${SNAPSHOT_LABELS[snapshotIndex]} lacks three audit lanes`);
        for (let position = 0; position < 3; position += 1) {
          const record = captures?.[position];
          push(record?.callback ?? record, IMMEDIATE_AIF_PHASE0_LANES[position],
            'audit', null, position);
        }
      }
    }
  }
  report(entries.length === 126 && serial === 127,
    'callback traversal is not exactly 126 globally serialized callbacks');
  return entries;
}

function expectedScheduleBucketBaseWords(scheduleId) {
  const words = new Uint32Array(PINNED_MERGED_GEOMETRY.vertexCount);
  let cursor = 0;
  const spans = RUNNER_GEOMETRY_VERTEX_COUNTS.map((vertexCount, draw) => {
    const start = cursor;
    const end = start + vertexCount;
    const expectedBase = IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId].sourceBaseByDraw[draw];
    words.fill(expectedBase, start, end);
    cursor = end;
    return { draw, start, end, vertexCount, expectedBase, exact: true };
  });
  if (cursor !== words.length) throw new Error('Pinned merged vertex spans are inconsistent.');
  return { words, spans, sha256: sha256(Buffer.from(words.buffer)) };
}

function normalizeRunnerCommonRenderState(rawProjection) {
  if (!isPlainJsonObject(rawProjection) || !isPlainJsonObject(rawProjection.scene)
    || !isPlainJsonObject(rawProjection.scene.laneRoots)
    || !Array.isArray(rawProjection.scene.directChildren)) return null;
  const projection = structuredClone(rawProjection);
  const laneUuids = new Set();
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const root = projection.scene.laneRoots[lane];
    if (!isPlainJsonObject(root) || typeof root.uuid !== 'string'
      || typeof root.visible !== 'boolean') return null;
    laneUuids.add(root.uuid);
    root.visible = '__selected-lane-dependent__';
  }
  projection.scene.activeLane = '__selected-lane__';
  for (const child of projection.scene.directChildren) {
    if (laneUuids.has(child?.uuid)) child.visible = '__selected-lane-dependent__';
  }
  return projection;
}

function commonCommitmentExact(
  commitment, expectedLabel, expectedLane = null, expectedConfiguration = null,
) {
  const raw = commitment?.rawProjection;
  const normalized = normalizeRunnerCommonRenderState(raw);
  if (normalized === null || !pinnedRenderConfigurationExact(expectedConfiguration)) {
    return false;
  }
  const rawBytes = Buffer.from(JSON.stringify(raw));
  const normalizedBytes = Buffer.from(JSON.stringify(normalized));
  const expected = expectedConfiguration;
  const materialModes = { A: 'bucket-base', I: 'immediate-base', F: 'indirect-first-instance' };
  const renderer = raw?.renderer;
  const camera = raw?.camera;
  const scene = raw?.scene;
  const visibleLanes = IMMEDIATE_AIF_PHASE0_LANES.filter(
    (lane) => scene?.laneRoots?.[lane]?.visible === true,
  );
  const rendererExact = exactJson({
    viewport: renderer?.viewport,
    autoClear: renderer?.autoClear,
    autoClearColor: renderer?.autoClearColor,
    autoClearDepth: renderer?.autoClearDepth,
    autoClearStencil: renderer?.autoClearStencil,
    clearColor: renderer?.clearColor,
    clearAlpha: renderer?.clearAlpha,
    sortObjects: renderer?.sortObjects,
    toneMapping: renderer?.toneMapping,
    toneMappingExposure: renderer?.toneMappingExposure,
    outputColorSpace: renderer?.outputColorSpace,
    reversedDepthBuffer: renderer?.reversedDepthBuffer,
    trackTimestamp: renderer?.trackTimestamp,
  }, {
    viewport: expected.viewport,
    autoClear: expected.renderer.autoClear,
    autoClearColor: expected.renderer.autoClearColor,
    autoClearDepth: expected.renderer.autoClearDepth,
    autoClearStencil: expected.renderer.autoClearStencil,
    clearColor: expected.clear.color,
    clearAlpha: expected.clear.alpha,
    sortObjects: expected.renderer.sortObjects,
    toneMapping: expected.renderer.toneMapping,
    toneMappingExposure: expected.renderer.toneMappingExposure,
    outputColorSpace: expected.renderer.outputColorSpace,
    reversedDepthBuffer: expected.renderer.reversedDepthBuffer,
    trackTimestamp: expected.renderer.trackTimestamp,
  }) && typeof renderer?.backendDeviceId === 'string'
    && renderer.backendDeviceId === renderer?.backendParameterDeviceId
    && renderer?.backendDeviceMatchesParameter === true;
  const cameraExact = camera?.type === expected.camera.type
    && camera?.fov === expected.camera.fov && camera?.aspect === expected.camera.aspect
    && camera?.near === expected.camera.near && camera?.far === expected.camera.far
    && camera?.zoom === expected.camera.zoom
    && equalThreeImmediatePhase0PersistedNumberArrays(
      camera?.position, expected.camera.position,
    )
    && equalThreeImmediatePhase0PersistedNumberArrays(
      camera?.quaternion, expected.camera.quaternion,
    )
    && equalThreeImmediatePhase0PersistedNumberArrays(camera?.scale, expected.camera.scale)
    && equalThreeImmediatePhase0PersistedNumberArrays(camera?.up, expected.camera.up)
    && camera?.coordinateSystem === expected.camera.coordinateSystem
    && equalThreeImmediatePhase0PersistedNumberArrays(
      camera?.matrixWorld, expected.camera.matrixWorld,
    )
    && equalThreeImmediatePhase0PersistedNumberArrays(
      camera?.matrixWorldInverse, expected.camera.matrixWorldInverse,
    )
    && equalThreeImmediatePhase0PersistedNumberArrays(
      camera?.projectionMatrix, expected.camera.projectionMatrix,
    )
    && equalThreeImmediatePhase0PersistedNumberArrays(
      camera?.projectionMatrixInverse, expected.camera.projectionMatrixInverse,
    )
    && typeof camera?.uuid === 'string' && Number.isSafeInteger(camera?.id);
  const materialsExact = IMMEDIATE_AIF_PHASE0_LANES.every((lane) => {
    const material = raw?.materials?.[lane];
    return typeof material?.uuid === 'string' && Number.isSafeInteger(material?.id)
      && material?.name === '' && Number.isSafeInteger(material?.version)
      && material.version >= 0 && material?.addressMode === materialModes[lane]
      && exactJson(material?.content, expected.material);
  }) && new Set(IMMEDIATE_AIF_PHASE0_LANES.map(
    (lane) => raw?.materials?.[lane]?.uuid,
  )).size === 3;
  const sceneExact = scene?.type === 'Scene' && scene?.name === 'phase0-production-scene'
    && scene?.background === expected.clear.color
    && exactArray(scene?.matrix, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    && exactArray(scene?.matrixWorld,
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    && scene?.matrixAutoUpdate === false && scene?.matrixWorldAutoUpdate === false
    && IMMEDIATE_AIF_PHASE0_LANES.includes(scene?.activeLane)
    && visibleLanes.length === 1 && visibleLanes[0] === scene.activeLane;
  return commitment?.schemaVersion === 1
    && commitment?.kind === 'immediate-aif-phase0-common-render-state-commitment'
    && commitment?.label === expectedLabel
    && commitment?.encoding === 'json-utf8-property-order-v1'
    && commitment?.rawByteLength === rawBytes.byteLength
    && commitment?.rawSha256 === sha256(rawBytes)
    && commitment?.byteLength === normalizedBytes.byteLength
    && commitment?.sha256 === sha256(normalizedBytes)
    && exactJson(commitment?.projection, normalized)
    && commitment?.laneSelection?.activeLane === scene.activeLane
    && exactArray(commitment?.laneSelection?.visibleLanes, visibleLanes)
    && commitment?.laneSelection?.exact === true
    && (expectedLane === null || scene.activeLane === expectedLane)
    && rendererExact && cameraExact && materialsExact
    && exactJson(raw?.lights, expected.lights) && sceneExact;
}

function phase0CreationEventInventory(evidence) {
  const resources = evidence?.resources?.gpuCreations ?? [];
  const pipelines = evidence?.pipelines ?? {};
  const shaders = evidence?.shaders ?? {};
  const lifecycle = evidence?.lifecycle ?? {};
  return [
    ...resources.map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: record?.method,
      resourceId: record?.resourceId,
      resourceClass: record?.resourceClass ?? null,
      ...(record?.method === 'createBuffer' ? {
        size: record?.size,
        usage: record?.usage,
        mappedAtCreation: record?.mappedAtCreation,
      } : {}),
    })),
    ...(pipelines?.render ?? []).map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: record?.method,
      resourceId: record?.pipelineId,
      resourceClass: 'render-pipeline',
    })),
    ...(pipelines?.compute ?? []).map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: record?.method,
      resourceId: record?.pipelineId,
      resourceClass: 'compute-pipeline',
    })),
    ...(shaders?.modules ?? []).map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: 'createShaderModule',
      resourceId: record?.moduleId,
      resourceClass: 'shader-module',
    })),
    ...(pipelines?.bindGroupLayouts ?? []).map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: 'createBindGroupLayout',
      resourceId: record?.layoutId,
      resourceClass: 'bind-group-layout',
    })),
    ...(pipelines?.layouts ?? []).map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: 'createPipelineLayout',
      resourceId: record?.layoutId,
      resourceClass: 'pipeline-layout',
    })),
    ...(evidence?.resources?.textureViews ?? []).map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: 'createTextureView',
      resourceId: record?.viewId,
      resourceClass: 'texture-view',
      parentTextureId: record?.textureId,
    })),
    ...(lifecycle?.renderBundleTraces ?? []).map((record) => ({
      sequence: record?.sequence,
      capturePhase: record?.capturePhase,
      category: 'createRenderBundleEncoder',
      resourceId: record?.bundleId ?? record?.encoderId,
      resourceClass: 'render-bundle',
      encoderId: record?.encoderId,
      bundleId: record?.bundleId,
      finishSequence: record?.finishSequence ?? null,
      nativeFinishReturned: record?.nativeFinishReturned,
    })),
  ].sort((left, right) => left.sequence - right.sequence);
}

function markerExact(markers, sequence, kind, detail) {
  const matches = (markers ?? []).filter((marker) => marker?.sequence === sequence);
  return matches.length === 1 && matches[0]?.kind === kind
    && exactJson(matches[0]?.detail, detail);
}

function validateScheduleInstallationRecord(
  installation, expected, evidence, traversal, creationInventory, reasons,
) {
  const { scenarioId, scheduleId, ordinal, reason } = expected;
  const installationId = `${scenarioId}/${ordinal}/${scheduleId}`;
  const basePhase = `phase0/schedule-install/${scenarioId}/${ordinal}-${scheduleId}`;
  const oracle = cpuOracles()[scenarioId].schedules[scheduleId];
  const base = expectedScheduleBucketBaseWords(scheduleId);
  const markers = evidence?.lifecycle?.instrumentationMarkers;
  const managerSemantics = [
    ['A.bucketBase', 'A', 1, base.words.byteLength, base.sha256],
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
      `${lane}.indirectCommands`, lane, 4, 640, oracle[lane].commandSha256,
    ]),
  ];
  const managerCalls = installation?.managerCalls;
  const callsExact = Array.isArray(managerCalls) && managerCalls.length === 4
    && managerCalls.every((call, index) => {
      const [semantic, lane, attributeType, byteLength] = managerSemantics[index];
      return call?.semantic === semantic && call?.lane === lane
        && call?.manager === 'renderer._attributes'
        && call?.managerConstructor === 'Attributes'
        && call?.attributeType === attributeType
        && Number.isSafeInteger(call?.attributeId) && call.attributeId >= 0
        && Number.isSafeInteger(call?.attributeVersion) && call.attributeVersion >= 0
        && (call?.beforeDataVersion === null
          || (Number.isSafeInteger(call.beforeDataVersion) && call.beforeDataVersion >= -1))
        && Number.isSafeInteger(call?.afterDataVersion)
        && call.afterDataVersion === call.attributeVersion
        && typeof call?.gpuBufferId === 'string' && call.gpuBufferId.length > 0
        && call?.gpuBufferSize === byteLength
        && Number.isSafeInteger(call?.startSequence)
        && Number.isSafeInteger(call?.completeSequence)
        && call.startSequence < call.completeSequence
        && markerExact(markers, call.startSequence,
          'phase0-attribute-manager-update-start', {
            installationId, semantic, attributeId: call.attributeId,
            attributeVersion: call.attributeVersion, attributeType,
            beforeDataVersion: call.beforeDataVersion,
          })
        && markerExact(markers, call.completeSequence,
          'phase0-attribute-manager-update-complete', {
            installationId, semantic, attributeId: call.attributeId,
            attributeVersion: call.attributeVersion, attributeType,
            afterDataVersion: call.afterDataVersion, gpuBufferId: call.gpuBufferId,
          });
    });
  const commands = installation?.commands;
  const commandsExact = exactArray(Object.keys(commands ?? {}), IMMEDIATE_AIF_PHASE0_LANES)
    && IMMEDIATE_AIF_PHASE0_LANES.every((lane) => {
      const command = commands?.[lane];
      const expectedCommand = oracle[lane];
      const call = managerCalls?.find((candidate) => (
        candidate?.semantic === `${lane}.indirectCommands`
      ));
      const capturePhase = `${basePhase}/command-readback/${lane}`;
      return command?.lane === lane && command?.capturePhase === capturePhase
        && command?.attributeId === call?.attributeId
        && command?.attributeVersion === call?.attributeVersion
        && command?.gpuBufferId === call?.gpuBufferId
        && command?.byteLength === 640
        && command?.gpuSha256 === expectedCommand.commandSha256
        && command?.expectedSha256 === expectedCommand.commandSha256
        && exactArray(command?.frozenWords, expectedCommand.commandWords)
        && exactArray(command?.expectedWords, expectedCommand.commandWords)
        && command?.exact === true
        && Number.isSafeInteger(command?.readbackStartSequence)
        && Number.isSafeInteger(command?.readbackCompleteSequence)
        && command.readbackStartSequence > installation?.queueCompleteSequence
        && command.readbackCompleteSequence > command.readbackStartSequence
        && markerExact(markers, command.readbackStartSequence,
          'phase0-command-preflight-readback-start', {
            installationId, lane, attributeId: command.attributeId,
          })
        && markerExact(markers, command.readbackCompleteSequence,
          'phase0-command-preflight-readback-complete', { installationId, lane });
    });
  const aCall = managerCalls?.[0];
  const expectedFrozenIds = [
    aCall?.gpuBufferId,
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => commands?.[lane]?.gpuBufferId),
  ];
  const commandSetProjection = IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
    lane,
    gpuSha256: commands?.[lane]?.gpuSha256,
    frozenWords: commands?.[lane]?.frozenWords,
  }));
  const preflightLane = ordinal % 4 === 1 ? 'A' : 'I';
  const postflightLane = ordinal % 4 === 0 ? 'F' : 'I';
  const common = installation?.commonState;
  const expectedRenderConfiguration = evidence?.resources?.commonResources?.expected;
  const commonExact = common?.pass === true
    && common?.exactProjection === true && common?.exactSha256 === true
    && commonCommitmentExact(common?.preflight, `${installationId}/preflight`, preflightLane,
      expectedRenderConfiguration)
    && commonCommitmentExact(common?.postflight, `${installationId}/postflight`, postflightLane,
      expectedRenderConfiguration)
    && exactJson(common?.preflight?.projection, common?.postflight?.projection)
    && common?.preflight?.sha256 === common?.postflight?.sha256;
  const callbacks = traversal.filter((entry) => entry.installationId === installationId);
  const callbackBindingsExact = callbacks.every(({ callback, lane, phase }) => {
    const command = installation?.commands?.[lane];
    const binding = callback?.command?.preflightBinding;
    const commonState = callback?.commonState;
    return callback?.enrichmentState === 'complete'
      && callback?.schedulePreflightSha256 === common?.preflight?.sha256
      && binding?.pass === true && binding?.installationId === installationId
      && binding?.installationOrdinal === ordinal && binding?.scenarioId === scenarioId
      && binding?.scheduleId === scheduleId
      && binding?.preflightSha256 === command?.gpuSha256
      && binding?.preflightByteLength === 640 && binding?.frozenWordCount === 160
      && commonState?.pass === true
      && commonState?.preflightSha256 === common?.preflight?.sha256
      && commonState?.beforeSha256 === common?.preflight?.sha256
      && commonState?.afterSha256 === common?.preflight?.sha256
      && SHA256_PATTERN.test(commonState?.beforeRawSha256 ?? '')
      && commonState?.afterRawSha256 === commonState?.beforeRawSha256
      && exactJson(commonState?.beforeLaneSelection, {
        activeLane: lane, visibleLanes: [lane], exact: true,
      })
      && exactJson(commonState?.afterLaneSelection, {
        activeLane: lane, visibleLanes: [lane], exact: true,
      })
      && callback?.phase === phase;
  });
  addReasonUnless(installation?.schemaVersion === 1
    && installation?.kind === 'immediate-aif-phase0-schedule-installation-preflight'
    && installation?.pass === true && installation?.frozen === true
    && installation?.installationId === installationId
    && installation?.ordinal === ordinal && installation?.scenarioId === scenarioId
    && installation?.scheduleId === scheduleId && installation?.reason === reason
    && installation?.basePhase === basePhase
    && Number.isSafeInteger(installation?.startSequence)
    && Number.isSafeInteger(installation?.queueCompleteSequence)
    && Number.isSafeInteger(installation?.completeSequence)
    && installation.startSequence < managerCalls?.[0]?.startSequence
    && managerCalls?.every((call, index) => index === 0
      || call.startSequence > managerCalls[index - 1].completeSequence)
    && managerCalls?.at(-1)?.completeSequence < installation.queueCompleteSequence
    && installation.queueCompleteSequence
      < Math.min(...IMMEDIATE_AIF_PHASE0_LANES.map(
        (lane) => commands?.[lane]?.readbackStartSequence ?? Infinity,
      ))
    && Math.max(...IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => commands?.[lane]?.readbackCompleteSequence ?? -Infinity,
    )) < installation.completeSequence
    && markerExact(markers, installation.startSequence, 'phase0-schedule-install-start', {
      installationId, scenarioId, scheduleId, reason,
    })
    && markerExact(markers, installation.queueCompleteSequence,
      'phase0-schedule-install-queue-complete', { installationId })
    && markerExact(markers, installation.completeSequence,
      'phase0-schedule-install-complete', { installationId })
    && callsExact && commandsExact
    && installation?.aBucketBase?.attributeId === aCall?.attributeId
    && installation?.aBucketBase?.attributeVersion === aCall?.attributeVersion
    && installation?.aBucketBase?.byteLength === base.words.byteLength
    && installation?.aBucketBase?.cpuSha256 === base.sha256
    && installation?.aBucketBase?.exact === true
    && exactJson(installation?.aBucketBase?.spans, base.spans)
    && installation?.commandSetSha256
      === sha256(Buffer.from(JSON.stringify(commandSetProjection)))
    && exactArray(installation?.frozenResourceIds, expectedFrozenIds)
    && new Set(expectedFrozenIds).size === 4
    && commonExact && callbackBindingsExact,
  reasons, `${installationId} schedule installation/preflight is not independently exact`);
  return { installationId, callbacks, expectedFrozenIds, commonExact };
}

function validatePersistentInventorySnapshot(
  snapshot, expectedLabel, scenarioId, creationInventory, markers,
) {
  const expectedRecords = creationInventory.filter((record) => (
    record.resourceClass !== 'readback-staging'
      && record.sequence < (snapshot?.markerSequence ?? -Infinity)
  ));
  return snapshot?.schemaVersion === 1
    && snapshot?.kind === 'immediate-aif-phase0-persistent-resource-freeze'
    && snapshot?.pass === true && snapshot?.label === expectedLabel
    && snapshot?.scenarioId === scenarioId
    && Number.isSafeInteger(snapshot?.markerSequence)
    && markerExact(markers, snapshot.markerSequence, 'phase0-persistent-resource-freeze', {
      label: expectedLabel, scenarioId,
    })
    && snapshot?.resourceCount === expectedRecords.length
    && exactJson(snapshot?.records, expectedRecords);
}

export function validateThreeImmediatePhase0ChallengeWriteClosure(challenge, {
  queueWrites = [], queueTextureWrites = [], queueExternalCopies = [], maps = [],
  frozenIds = [], immutableIds = [],
} = {}) {
  const reasons = [];
  const frozenWrites = queueWrites.filter((record) => frozenIds.includes(record?.bufferId));
  const frozenMaps = maps.filter((record) => frozenIds.includes(record?.resourceId));
  const immutableWrites = queueWrites.filter(
    (record) => immutableIds.includes(record?.bufferId),
  );
  const immutableMaps = maps.filter((record) => immutableIds.includes(record?.resourceId));
  addReasonUnless(challenge?.queueWriteBufferCount === queueWrites.length
    && exactJson(challenge?.queueWriteBuffers, queueWrites)
    && challenge?.queueWriteTextureCount === 0
    && queueTextureWrites.length === 0
    && exactJson(challenge?.queueWriteTextures, [])
    && challenge?.queueCopyExternalImageToTextureCount === 0
    && queueExternalCopies.length === 0
    && exactJson(challenge?.queueCopyExternalImagesToTexture, [])
    && challenge?.frozenBufferWriteCount === 0
    && exactJson(challenge?.frozenBufferWrites, frozenWrites) && frozenWrites.length === 0
    && challenge?.frozenBufferMapCount === 0
    && exactJson(challenge?.frozenBufferMaps, frozenMaps) && frozenMaps.length === 0
    && challenge?.immutableBufferWriteCount === 0
    && exactJson(challenge?.immutableBufferWrites, immutableWrites)
    && immutableWrites.length === 0
    && challenge?.immutableBufferMapCount === 0
    && exactJson(challenge?.immutableBufferMaps, immutableMaps)
    && immutableMaps.length === 0,
  reasons, 'ordered challenge mutated a frozen/immutable resource or texture');
  return { valid: reasons.length === 0, reasons };
}

export function validateThreeImmediatePhase0ProductionReadbackTransfers({
  phases = [], transfers = [], stagingCreations = [], gpuCreations = [],
  colorTextureId = null, depthTextureId = null,
} = {}) {
  const reasons = [];
  const stagingIds = new Set(stagingCreations.map((record) => record?.resourceId));
  const expectedSourceIds = [colorTextureId, depthTextureId];
  addReasonUnless(phases.length > 0
    && typeof colorTextureId === 'string' && typeof depthTextureId === 'string'
    && colorTextureId !== depthTextureId
    && transfers.length === phases.length * 2
    && stagingCreations.length === phases.length * 2
    && new Set(stagingCreations.map((record) => record?.resourceId)).size
      === stagingCreations.length
    && phases.every((phase) => {
      const phaseTransfers = transfers.filter((record) => record?.capturePhase === phase);
      const phaseStaging = stagingCreations.filter((record) => record?.capturePhase === phase);
      const phaseGpuCreations = gpuCreations.filter((record) => (
        phaseStaging.some((staging) => staging.resourceId === record?.resourceId)
      ));
      return phaseTransfers.length === 2 && phaseStaging.length === 2
        && phaseGpuCreations.length === 2
        && phaseStaging.every((record) => record?.category === 'createBuffer'
          && record?.resourceClass === 'readback-staging')
        && phaseGpuCreations.every((record) => record?.method === 'createBuffer'
          && record?.resourceClass === 'readback-staging'
          && record?.size === TARGET_BYTE_LENGTH && record?.usage === 9)
        && phaseGpuCreations.every((record, index) => (
          Number.isSafeInteger(record?.sequence)
            && record.sequence < phaseTransfers[index]?.sequence
        ))
        && exactArray(phaseTransfers.map((record) => record?.source?.textureId),
          expectedSourceIds)
        && phaseTransfers.every((record) => record?.method === 'copyTextureToBuffer'
          && record?.source?.mipLevel === 0
          && exactJson(record?.source?.origin, { x: 0, y: 0, z: 0 })
          && record?.source?.aspect === 'all'
          && stagingIds.has(record?.destination?.bufferId)
          && record?.destination?.offset === 0
          && record?.destination?.bytesPerRow === 5_120
          && record?.destination?.rowsPerImage === null
          && exactJson(record?.copySize, {
            width: 1_280, height: 720, depthOrArrayLayers: 1,
          }))
        && exactArray(phaseTransfers.map((record) => record?.destination?.bufferId),
          phaseStaging.map((record) => record?.resourceId));
    }),
  reasons, 'production texture readback copy topology is not exact');
  return { valid: reasons.length === 0, reasons };
}

export function validateThreeImmediatePhase0CommandSubmissionClosure({
  phases = [], commandEncoders = [], queueSubmissions = [], renderPasses = [], transfers = [],
} = {}) {
  const reasons = [];
  const submittedIds = queueSubmissions.flatMap((record) => record?.commandBufferIds ?? []);
  addReasonUnless(phases.length > 0
    && commandEncoders.length === phases.length * 4
    && queueSubmissions.length === phases.length * 4
    && new Set(commandEncoders.map((record) => record?.commandEncoderId)).size
      === commandEncoders.length
    && new Set(commandEncoders.map((record) => record?.commandBufferId)).size
      === commandEncoders.length
    && new Set(submittedIds).size === commandEncoders.length
    && phases.every((phase) => {
      const encoders = commandEncoders.filter(
        (record) => record?.capturePhase === phase,
      ).sort((left, right) => left.sequence - right.sequence);
      const submissions = queueSubmissions.filter(
        (record) => record?.capturePhase === phase,
      ).sort((left, right) => left.sequence - right.sequence);
      const passes = renderPasses.filter((record) => record?.capturePhase === phase);
      const phaseTransfers = transfers.filter((record) => record?.capturePhase === phase);
      return encoders.length === 4 && submissions.length === 4
        && passes.length === 2 && phaseTransfers.length === 2
        && encoders.every((encoder, index) => encoder?.finishCallCount === 1
          && encoder?.finishDescriptor === null
          && Number.isSafeInteger(encoder?.sequence)
          && Number.isSafeInteger(encoder?.finishSequence)
          && encoder.sequence < encoder.finishSequence
          && typeof encoder?.commandBufferId === 'string'
          && exactArray(submissions[index]?.commandBufferIds, [encoder.commandBufferId])
          && submissions[index]?.sequence > encoder.finishSequence
          && (index === 3 || submissions[index].sequence < encoders[index + 1].sequence))
        && exactArray(encoders[0]?.renderPassEncoderIds, [passes[0]?.encoderId])
        && exactArray(encoders[1]?.renderPassEncoderIds, [passes[1]?.encoderId])
        && exactArray(encoders[2]?.renderPassEncoderIds, [])
        && exactArray(encoders[3]?.renderPassEncoderIds, [])
        && encoders.every((encoder) => exactArray(encoder?.computePassEncoderIds, []))
        && exactArray(encoders[0]?.transferSequences, [])
        && exactArray(encoders[1]?.transferSequences, [])
        && exactArray(encoders[2]?.transferSequences, [phaseTransfers[0]?.sequence])
        && exactArray(encoders[3]?.transferSequences, [phaseTransfers[1]?.sequence]);
    })
    && exactArray([...submittedIds].sort(),
      commandEncoders.map((record) => record.commandBufferId).sort()),
  reasons, 'command-encoder finish to singleton queue-submit closure is not exact');
  return { valid: reasons.length === 0, reasons };
}

/**
 * Closes the complete command-buffer ledger.  Every finished command buffer is
 * a singleton submission containing exactly one render pass, one compute pass,
 * or one readback copy.  In particular, this catches work encoded before a
 * challenged interval and submitted inside it, as well as unused/piggybacked
 * command buffers that phase-local pass counts cannot see.
 */
export function validateThreeImmediatePhase0GlobalCommandClosure({
  commandEncoders = [],
  renderPasses = [],
  computePasses = [],
  transfers = [],
  queueSubmissions = [],
  stagingCreations = [],
  bufferMapEvents = [],
  bufferDestructions = [],
  bufferCreations = [],
  textureCreations = [],
  queueWriteTextures = [],
  queueExternalCopies = [],
  expectedRenderPassCount = 798,
  expectedComputePassCount = 6,
  expectedTransferCount = 825,
} = {}) {
  const reasons = [];
  const expectedEncoderCount = expectedRenderPassCount
    + expectedComputePassCount + expectedTransferCount;
  const encoderIds = commandEncoders.map((record) => record?.commandEncoderId);
  const commandBufferIds = commandEncoders.map((record) => record?.commandBufferId);
  const submittedIds = queueSubmissions.flatMap(
    (record) => record?.commandBufferIds ?? [],
  );
  const stagingIds = new Set(stagingCreations.map((record) => record?.resourceId));
  const transferDestinationId = (record) => (
    record?.method === 'copyBufferToBuffer'
      ? record?.destinationBufferId
      : record?.method === 'copyTextureToBuffer'
        ? record?.destination?.bufferId
        : null
  );
  const sequencesStrict = (records) => records.every((record, index) => (
    Number.isSafeInteger(record?.sequence)
      && (index === 0 || record.sequence > records[index - 1].sequence)
  ));

  addReasonUnless(commandEncoders.length === expectedEncoderCount
    && renderPasses.length === expectedRenderPassCount
    && computePasses.length === expectedComputePassCount
    && transfers.length === expectedTransferCount
    && stagingCreations.length === expectedTransferCount
    && queueSubmissions.length === expectedEncoderCount
    && queueWriteTextures.length === 0 && queueExternalCopies.length === 0,
  reasons, 'global command/readback counts or forbidden queue texture operations differ');
  addReasonUnless(sequencesStrict(commandEncoders) && sequencesStrict(renderPasses)
    && sequencesStrict(computePasses) && sequencesStrict(transfers)
    && sequencesStrict(queueSubmissions)
    && new Set(encoderIds).size === commandEncoders.length
    && new Set(commandBufferIds).size === commandEncoders.length
    && new Set(submittedIds).size === queueSubmissions.length
    && submittedIds.length === queueSubmissions.length
    && exactArray([...commandBufferIds].sort(), [...submittedIds].sort()),
  reasons, 'global command encoder/command buffer/submission identities are not bijective');

  addReasonUnless(transfers.every((record) => (
    (record?.method === 'copyBufferToBuffer'
      && bufferCreations.filter((buffer) => buffer?.resourceId === record?.sourceBufferId)
        .length === 1
      && stagingIds.has(record?.destinationBufferId)
      && Number.isSafeInteger(record?.sourceOffset) && record.sourceOffset >= 0
      && record?.destinationOffset === 0
      && Number.isSafeInteger(record?.size) && record.size > 0
      && record.sourceOffset + record.size <= bufferCreations.find(
        (buffer) => buffer?.resourceId === record.sourceBufferId,
      )?.size)
    || (record?.method === 'copyTextureToBuffer'
      && textureCreations.filter(
        (texture) => texture?.resourceId === record?.source?.textureId,
      ).length === 1
      && stagingIds.has(record?.destination?.bufferId)
      && record?.source?.mipLevel === 0
      && exactJson(record?.source?.origin, { x: 0, y: 0, z: 0 })
      && record?.source?.aspect === 'all'
      && record?.destination?.offset === 0
      && Number.isSafeInteger(record?.destination?.bytesPerRow)
      && record.destination.bytesPerRow > 0
      && record.destination.bytesPerRow % 256 === 0
      && record.destination.bytesPerRow >= record?.copySize?.width * 4
      && record?.destination?.rowsPerImage === null
      && Number.isSafeInteger(record?.copySize?.width) && record.copySize.width > 0
      && Number.isSafeInteger(record?.copySize?.height) && record.copySize.height > 0
      && record?.copySize?.depthOrArrayLayers === 1
      && ['rgba8unorm', 'depth32float'].includes(textureCreations.find(
        (texture) => texture?.resourceId === record.source.textureId,
      )?.format)
      && textureSizeExact(textureCreations.find(
        (texture) => texture?.resourceId === record.source.textureId,
      )?.size, record.copySize.width, record.copySize.height))
  )), reasons, 'global transfer ledger contains a forbidden or malformed operation');
  addReasonUnless(new Set(transfers.map(transferDestinationId)).size === transfers.length
    && transfers.every((record) => stagingIds.has(transferDestinationId(record)))
    && exactArray([...transfers.map(transferDestinationId)].sort(), [...stagingIds].sort()),
  reasons, 'readback transfers and staging destinations are not one-to-one');

  for (const encoder of commandEncoders) {
    const encoderTransfers = transfers.filter(
      (record) => record?.commandEncoderId === encoder?.commandEncoderId,
    );
    const encoderRenderPasses = renderPasses.filter(
      (record) => record?.commandEncoderId === encoder?.commandEncoderId,
    );
    const encoderComputePasses = computePasses.filter(
      (record) => record?.commandEncoderId === encoder?.commandEncoderId,
    );
    const categoryCounts = [
      encoderRenderPasses.length, encoderComputePasses.length, encoderTransfers.length,
    ];
    const submissions = queueSubmissions.filter(
      (record) => record?.commandBufferIds?.includes(encoder?.commandBufferId),
    );
    const childRecords = [
      ...encoderRenderPasses, ...encoderComputePasses, ...encoderTransfers,
    ];
    addReasonUnless(typeof encoder?.capturePhase === 'string'
      && encoder.capturePhase.length > 0
      && Number.isSafeInteger(encoder?.sequence)
      && typeof encoder?.commandEncoderId === 'string'
      && encoder?.finishCallCount === 1 && encoder?.finishDescriptor === null
      && Number.isSafeInteger(encoder?.finishSequence)
      && encoder.finishSequence > encoder.sequence
      && typeof encoder?.commandBufferId === 'string'
      && categoryCounts.filter((count) => count === 1).length === 1
      && categoryCounts.filter((count) => count === 0).length === 2
      && exactArray(encoder?.renderPassEncoderIds,
        encoderRenderPasses.map((record) => record.encoderId))
      && exactArray(encoder?.computePassEncoderIds,
        encoderComputePasses.map((record) => record.encoderId))
      && exactArray(encoder?.transferSequences,
        encoderTransfers.map((record) => record.sequence))
      && childRecords.every((record) => record?.capturePhase === encoder.capturePhase
        && record.sequence > encoder.sequence && record.sequence < encoder.finishSequence)
      && submissions.length === 1
      && submissions[0]?.capturePhase === encoder.capturePhase
      && exactArray(submissions[0]?.commandBufferIds, [encoder.commandBufferId])
      && Number.isSafeInteger(submissions[0]?.sequence)
      && submissions[0].sequence > encoder.finishSequence,
    reasons, `command encoder ${encoder?.commandEncoderId ?? '<unknown>'} is not one exact submitted category`);
  }

  for (const staging of stagingCreations) {
    const destinationTransfers = transfers.filter(
      (record) => transferDestinationId(record) === staging?.resourceId,
    );
    const transfer = destinationTransfers[0];
    const encoder = commandEncoders.find(
      (record) => record?.commandEncoderId === transfer?.commandEncoderId,
    );
    const submission = queueSubmissions.find(
      (record) => exactArray(record?.commandBufferIds, [encoder?.commandBufferId]),
    );
    const maps = bufferMapEvents.filter(
      (record) => record?.resourceId === staging?.resourceId,
    );
    const destructions = bufferDestructions.filter(
      (record) => record?.resourceId === staging?.resourceId,
    );
    const transferSize = transfer?.method === 'copyBufferToBuffer'
      ? transfer?.size
      : ((transfer?.copySize?.height - 1) * transfer?.destination?.bytesPerRow)
        + transfer?.copySize?.width * 4;
    addReasonUnless(staging?.method === 'createBuffer'
      && staging?.resourceClass === 'readback-staging'
      && typeof staging?.capturePhase === 'string' && staging.capturePhase.length > 0
      && staging?.usage === 9 && staging?.mappedAtCreation === false
      && Number.isSafeInteger(staging?.size) && staging.size > 0
      && destinationTransfers.length === 1 && transfer?.capturePhase === staging.capturePhase
      && transferSize === staging.size && staging.sequence < transfer.sequence
      && encoder?.finishSequence > transfer.sequence
      && submission?.sequence > encoder?.finishSequence
      && maps.length === 2
      && exactArray(maps.map((record) => record?.method), ['mapAsync', 'getMappedRange'])
      && maps.every((record) => record?.capturePhase === staging.capturePhase
        && record?.resourceClass === 'readback-staging'
        && record?.offset === 0
        && (record?.size === null || record.size === staging.size))
      && maps[0]?.mode === 1 && maps[1]?.mode === null
      && submission.sequence < maps[0].sequence && maps[0].sequence < maps[1].sequence
      && destructions.length === 1
      && destructions[0]?.capturePhase === staging.capturePhase
      && destructions[0]?.method === 'destroy' && destructions[0]?.callCount === 1
      && maps[1].sequence < destructions[0].sequence
      && staging?.destroyed === true && staging?.destroyCallCount === 1
      && staging?.destroySequence === destructions[0].sequence
      && staging?.destroyCapturePhase === staging.capturePhase,
    reasons, `staging buffer ${staging?.resourceId ?? '<unknown>'} lacks an exact copy/map/destroy chain`);
  }

  return { valid: reasons.length === 0, reasons };
}

/**
 * Reconstructs every Phase 0 readback source from protocol-owned semantics and
 * joins it to the raw copy command.  A staging buffer with a well-formed copy
 * is insufficient: the copy must originate from the exact committed command,
 * shared, geometry, or target resource for that phase.
 */
export function validateThreeImmediatePhase0ReadbackSourceBijection(evidence, {
  stagingCreations = [], transfers = [], transferLedger = [],
} = {}) {
  const reasons = [];
  const resources = evidence?.resources;
  const scenarios = evidence?.scenarios ?? [];
  const firstScenario = scenarios.find((scenario) => scenario?.scenarioId === 'v99');
  const secondScenario = scenarios.find((scenario) => scenario?.scenarioId === 'v20');
  const firstCommitments = firstScenario?.gpuResourceCommitments?.records ?? [];
  const secondCommitments = secondScenario?.gpuResourceCommitments?.records ?? [];
  const shared = Object.fromEntries(PHASE0_SHARED_RESOURCE_SEMANTICS.map((semantic) => {
    const first = firstCommitments.filter((record) => record?.semantic === semantic);
    const second = secondCommitments.filter((record) => record?.semantic === semantic);
    addReasonUnless(first.length === 1 && second.length === 1
      && first[0]?.attributeId === second[0]?.attributeId
      && first[0]?.gpuBufferId === second[0]?.gpuBufferId
      && first[0]?.byteLength === second[0]?.byteLength,
    reasons, `shared ${semantic} readback source is not stable across scenarios`);
    return [semantic, first[0] ?? null];
  }));
  const installations = resources?.scheduleInstallations?.records ?? [];
  const commands = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => {
    const records = installations.map((installation) => installation?.commands?.[lane]);
    const first = records[0];
    addReasonUnless(records.length === 8 && records.every((record) => (
      record?.attributeId === first?.attributeId
        && record?.gpuBufferId === first?.gpuBufferId
        && record?.byteLength === 640
    )), reasons, `${lane} command readback source is not stable across installations`);
    return [lane, first ?? null];
  }));
  const mergedRecords = resources?.geometryFixtures?.mergedProduction?.gpuRecords ?? [];
  const merged = Object.fromEntries(['bucketBase', 'normal', 'position', 'uv', 'index']
    .map((semantic) => {
      const matches = mergedRecords.filter((record) => record?.semantic === semantic);
      addReasonUnless(matches.length === 1, reasons,
        `merged ${semantic} readback source is absent or duplicated`);
      return [semantic, matches[0] ?? null];
    }));
  const declaredTargets = resources?.textureDestructionBeforeCleanup
    ?.declaredTargetTextures ?? [];
  const target = Object.fromEntries([
    'production.color', 'production.depth', 'address.color',
    'objectId.color', 'objectId.depth',
  ].map((semantic) => {
    const matches = declaredTargets.filter((record) => record?.semantic === semantic);
    addReasonUnless(matches.length === 1, reasons,
      `${semantic} readback texture is absent or duplicated`);
    return [semantic, matches[0] ?? null];
  }));
  const bufferSources = [
    ...PHASE0_SHARED_RESOURCE_SEMANTICS.map((semantic) => ({
      semantic: `shared.${semantic}`,
      resourceKind: 'buffer',
      attributeId: shared[semantic]?.attributeId,
      resourceId: shared[semantic]?.gpuBufferId,
    })),
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
      semantic: `command.${lane}`,
      resourceKind: 'buffer',
      attributeId: commands[lane]?.attributeId,
      resourceId: commands[lane]?.gpuBufferId,
    })),
    ...['bucketBase', 'normal', 'position', 'uv', 'index'].map((semantic) => ({
      semantic: `merged-geometry.${semantic}`,
      resourceKind: 'buffer',
      attributeId: merged[semantic]?.attributeId,
      resourceId: merged[semantic]?.gpuBufferId,
    })),
  ];
  const textureSources = [
    'production.color', 'production.depth', 'address.color',
    'objectId.color', 'objectId.depth',
  ].map((semantic) => ({
    semantic,
    resourceKind: 'texture',
    textureUuid: target[semantic]?.textureUuid,
    resourceId: target[semantic]?.gpuTextureId,
  }));
  const sourceInventory = resources?.stagingLifecycle?.sourceInventory;
  addReasonUnless(sourceInventory?.schemaVersion === 1
    && sourceInventory?.kind === 'immediate-aif-phase0-readback-source-inventory'
    && sourceInventory?.pass === true
    && sourceInventory?.bufferSourceCount === 16
    && sourceInventory?.textureSourceCount === 5
    && exactJson(sourceInventory?.bufferSources, bufferSources)
    && exactJson(sourceInventory?.textureSources, textureSources)
    && new Set(bufferSources.map((record) => record.resourceId)).size === 16
    && new Set(textureSources.map((record) => record.resourceId)).size === 5,
  reasons, 'page readback source inventory differs from independent resource identities');

  const expected = [];
  const addBuffer = (phase, semantic, record, byteLength) => expected.push({
    phase, semantic, kind: 'buffer', resourceId: record?.gpuBufferId,
    byteLength,
    sourceBinding: bufferSources.find((candidate) => candidate.semantic === semantic),
  });
  const addTexture = (phase, semantic) => {
    const address = semantic === 'address.color';
    const width = address ? 256 : TARGET_WIDTH;
    const height = address ? 256 : TARGET_HEIGHT;
    expected.push({
      phase, semantic, kind: 'texture', resourceId: target[semantic]?.gpuTextureId,
      byteLength: width * height * 4, width, height, bytesPerRow: width * 4,
      sourceBinding: textureSources.find((candidate) => candidate.semantic === semantic),
    });
  };
  for (const entry of expectedPhase0CallbackTraversal(scenarios, reasons)) {
    const callback = entry.callback;
    addTexture(`${entry.phase}/production`, 'production.color');
    addTexture(`${entry.phase}/production`, 'production.depth');
    addBuffer(`${entry.phase}/command-readback`, `command.${entry.lane}`,
      commands[entry.lane], 640);
    addTexture(`${entry.phase}/address`, 'address.color');
    addTexture(`${entry.phase}/object-id`, 'objectId.color');
    addBuffer(`${entry.phase}/visible-id-readback`, 'shared.visibleIds',
      shared.visibleIds, ADDRESS_BYTE_LENGTH);
    addReasonUnless(callback?.phases?.production === `${entry.phase}/production`, reasons,
      `${entry.phase} callback phase cannot be used for readback reconstruction`);
  }
  for (const scenarioId of VISIBILITY_IDS) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const phase = `phase0/live/${scenarioId}/${lane}`;
      addBuffer(phase, 'shared.overflow', shared.overflow, 4);
      addBuffer(phase, `command.${lane}`, commands[lane], 640);
      addBuffer(phase, 'shared.visibleIds', shared.visibleIds, ADDRESS_BYTE_LENGTH);
    }
    const phase = `phase0/resources/${scenarioId}/shared-commitment`;
    for (const semantic of PHASE0_SHARED_RESOURCE_SEMANTICS) {
      addBuffer(phase, `shared.${semantic}`, shared[semantic], shared[semantic]?.byteLength);
    }
  }
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    addTexture(`phase0/prime/${lane}`, 'production.color');
    addTexture(`phase0/prime/${lane}`, 'production.depth');
  }
  for (const semantic of ['bucketBase', 'normal', 'position', 'uv', 'index']) {
    addBuffer('phase0/resources/global/geometry-postflight',
      `merged-geometry.${semantic}`, merged[semantic], merged[semantic]?.byteLength);
  }
  for (const [index, installation] of installations.entries()) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      addBuffer(`${installation?.basePhase}/command-readback/${lane}`,
        `command.${lane}`, commands[lane], 640);
    }
    addReasonUnless(installation?.ordinal === index + 1, reasons,
      'schedule installation order changed during readback reconstruction');
  }

  const transferSourceId = (record) => record?.method === 'copyBufferToBuffer'
    ? record?.sourceBufferId : record?.method === 'copyTextureToBuffer'
      ? record?.source?.textureId : null;
  const transferDestinationId = (record) => record?.method === 'copyBufferToBuffer'
    ? record?.destinationBufferId : record?.method === 'copyTextureToBuffer'
      ? record?.destination?.bufferId : null;
  const transferByKey = new Map();
  for (const record of transfers) {
    const key = `${record?.capturePhase}\0${transferSourceId(record)}`;
    const bucket = transferByKey.get(key) ?? [];
    bucket.push(record);
    transferByKey.set(key, bucket);
  }
  const stagingById = new Map(stagingCreations.map(
    (record) => [record?.resourceId, record],
  ));
  const ledgerByStaging = new Map(transferLedger.map(
    (record) => [record?.stagingBufferId, record],
  ));
  const usedSequences = new Set();
  for (const spec of expected) {
    const matches = transferByKey.get(`${spec.phase}\0${spec.resourceId}`) ?? [];
    const transfer = matches.find((record) => !usedSequences.has(record?.sequence));
    if (transfer) usedSequences.add(transfer.sequence);
    const staging = stagingById.get(transferDestinationId(transfer));
    const ledger = ledgerByStaging.get(staging?.resourceId);
    const shapeExact = spec.kind === 'buffer'
      ? transfer?.method === 'copyBufferToBuffer'
        && transfer?.sourceOffset === 0 && transfer?.destinationOffset === 0
        && transfer?.size === spec.byteLength
      : transfer?.method === 'copyTextureToBuffer'
        && transfer?.source?.mipLevel === 0
        && exactJson(transfer?.source?.origin, { x: 0, y: 0, z: 0 })
        && transfer?.source?.aspect === 'all'
        && transfer?.destination?.offset === 0
        && transfer?.destination?.bytesPerRow === spec.bytesPerRow
        && transfer?.destination?.rowsPerImage === null
        && exactJson(transfer?.copySize, {
          width: spec.width, height: spec.height, depthOrArrayLayers: 1,
        });
    addReasonUnless(matches.length === 1 && transfer != null && shapeExact
      && staging?.capturePhase === spec.phase && staging?.size === spec.byteLength
      && staging?.usage === 9 && staging?.mappedAtCreation === false
      && ledger?.capturePhase === spec.phase
      && exactJson(ledger?.transfer, transfer)
      && exactJson(ledger?.sourceBinding, spec.sourceBinding),
    reasons, `${spec.phase}/${spec.semantic} readback source/copy/staging join is not exact`);
  }
  addReasonUnless(expected.length === 825 && transfers.length === 825
    && stagingCreations.length === 825 && transferLedger.length === 825
    && usedSequences.size === 825
    && transfers.every((record) => usedSequences.has(record?.sequence)),
  reasons, '825 readback transfers are not an exhaustive phase/source bijection');
  return { valid: reasons.length === 0, reasons, expectedCount: expected.length };
}

export function validateThreeImmediatePhase0DeviceInstrumentationClosure({
  patchedMethods = [], gpuCreations = [], queueWriteTextures = [],
  queueExternalCopies = [],
} = {}) {
  const reasons = [];
  const expectedPatchedMethods = [
    'createShaderModule', 'createPipelineLayout', 'createBindGroupLayout',
    'createRenderPipeline', 'createRenderPipelineAsync',
    'createComputePipeline', 'createComputePipelineAsync',
    'createBuffer', 'createTexture', 'createBindGroup', 'createSampler', 'createQuerySet',
    'importExternalTexture',
    'queue.writeBuffer', 'queue.writeTexture',
    'queue.copyExternalImageToTexture', 'queue.submit',
    'createRenderBundleEncoder', 'createCommandEncoder',
  ];
  addReasonUnless(exactArray(patchedMethods, expectedPatchedMethods),
    reasons, 'GPU device/queue instrumentation method inventory is not exact');
  addReasonUnless(gpuCreations.filter(
    (record) => record?.method === 'importExternalTexture',
  ).length === 0, reasons, 'GPUDevice.importExternalTexture call was recorded');
  addReasonUnless(queueWriteTextures.length === 0,
    reasons, 'GPUQueue.writeTexture call was recorded');
  addReasonUnless(queueExternalCopies.length === 0,
    reasons, 'GPUQueue.copyExternalImageToTexture call was recorded');
  return { valid: reasons.length === 0, reasons, expectedPatchedMethods };
}

function validateOrderedProductionChallengeRecord(
  challenge, installation, expected, evidence, traversal, creationInventory, reasons,
) {
  const { scenarioId, scheduleId, snapshotIndex, installationId } = expected;
  const callbacks = traversal.filter((entry) => entry.scenarioId === scenarioId
    && entry.snapshotIndex === snapshotIndex && entry.kind === 'order');
  const phases = callbacks.map((entry) => entry.phase);
  const expectedLanes = IMMEDIATE_AIF_PHASE0_LANE_ORDERS.flatMap((order) => order);
  const markers = evidence?.lifecycle?.instrumentationMarkers ?? [];
  const renderPasses = (evidence?.lifecycle?.renderPassTraces ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const computePasses = (evidence?.lifecycle?.computePassTraces ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const expectedPassPhases = phases.flatMap((phase) => [
    `${phase}/production`, `${phase}/production`,
  ]);
  const byPhase = Object.fromEntries(phases.map((phase) => [
    `${phase}/production`,
    renderPasses.filter((record) => record?.capturePhase === `${phase}/production`).length,
  ]));
  const creationEvents = creationInventory.filter((record) => (
    record.sequence > challenge?.startSequence && record.sequence < challenge?.endSequence
  ));
  const stagingCreations = creationEvents.filter((record) => (
    record.category === 'createBuffer' && record.resourceClass === 'readback-staging'
  ));
  const forbiddenCreations = creationEvents.filter((record) => !(
    record.category === 'createBuffer' && record.resourceClass === 'readback-staging'
  ));
  const queueWrites = (evidence?.lifecycle?.queueWriteBufferCalls ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const queueTextureWrites = (evidence?.lifecycle?.queueWriteTextureCalls ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const queueExternalCopies = (
    evidence?.lifecycle?.queueCopyExternalImageToTextureCalls ?? []
  ).filter((record) => record?.sequence > challenge?.startSequence
    && record?.sequence < challenge?.endSequence);
  const maps = (evidence?.lifecycle?.bufferMapEvents ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const sharedIds = (evidence?.scenarios ?? []).find(
    (scenario) => scenario?.scenarioId === scenarioId,
  )?.gpuResourceCommitments?.records?.map((record) => record?.gpuBufferId) ?? [];
  const geometryIds = evidence?.resources?.geometryFixtures
    ?.mergedProduction?.gpuRecords?.map((record) => record?.gpuBufferId) ?? [];
  const commandIds = IMMEDIATE_AIF_PHASE0_LANES.map(
    (lane) => installation?.commands?.[lane]?.gpuBufferId,
  );
  const expectedImmutableIds = [...new Set([...sharedIds, ...geometryIds, ...commandIds])];
  const scenarioCommitments = (evidence?.scenarios ?? []).find(
    (scenario) => scenario?.scenarioId === scenarioId,
  )?.gpuResourceCommitments?.records ?? [];
  const geometryRecords = evidence?.resources?.geometryFixtures
    ?.mergedProduction?.gpuRecords ?? [];
  const expectedImmutableResources = [
    ...scenarioCommitments.map((record) => ({
      semantic: `shared.${record?.semantic}`,
      attributeId: record?.attributeId,
      gpuBufferId: record?.gpuBufferId,
    })),
    ...IMMEDIATE_AIF_PHASE0_LANES.flatMap((lane) => [
      ...geometryRecords.filter((record) => (
        record?.semantic !== 'index' && (lane === 'A' || record?.semantic !== 'bucketBase')
      )).map((record) => ({
        semantic: `geometry.${lane}.${record?.semantic}`,
        attributeId: record?.attributeId,
        gpuBufferId: record?.gpuBufferId,
      })),
      ...geometryRecords.filter((record) => record?.semantic === 'index').map((record) => ({
        semantic: `geometry.${lane}.index`,
        attributeId: record?.attributeId,
        gpuBufferId: record?.gpuBufferId,
      })),
      {
        semantic: `geometry.${lane}.indirectCommands`,
        attributeId: installation?.commands?.[lane]?.attributeId,
        gpuBufferId: installation?.commands?.[lane]?.gpuBufferId,
      },
    ]),
  ];
  const immutableResourceSort = (left, right) => left.semantic.localeCompare(
    right.semantic, 'en', { sensitivity: 'variant' },
  );
  const rawTargets = installation?.commonState?.preflight?.rawProjection?.targets ?? {};
  const commonTargets = evidence?.resources?.commonResources?.targets ?? {};
  const expectedImmutableTextureResources = [
    ['production.color', rawTargets?.production?.texture?.uuid,
      commonTargets?.production?.colorTextureId],
    ['production.depth', rawTargets?.production?.depthTexture?.uuid,
      commonTargets?.production?.depthTextureIds?.[0]],
    ['address.color', rawTargets?.address?.texture?.uuid,
      commonTargets?.address?.colorTextureId],
    ['objectId.color', rawTargets?.objectId?.texture?.uuid,
      commonTargets?.objectId?.colorTextureId],
    ['objectId.depth', rawTargets?.objectId?.depthTexture?.uuid,
      commonTargets?.objectId?.depthTextureIds?.[0]],
  ].map(([semantic, textureUuid, gpuTextureId]) => ({
    semantic, textureUuid, gpuTextureId,
  }));
  const expectedImmutableTextureIds = expectedImmutableTextureResources.map(
    (record) => record.gpuTextureId,
  );
  const commandTransfers = (evidence?.lifecycle?.commandEncoderTransferCalls ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const commandEncoders = (evidence?.lifecycle?.commandEncoderTraces ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const queueSubmissions = (evidence?.lifecycle?.queueSubmissions ?? []).filter(
    (record) => record?.sequence > challenge?.startSequence
      && record?.sequence < challenge?.endSequence,
  );
  const productionTextureIds = [
    commonTargets?.production?.colorTextureId,
    commonTargets?.production?.depthTextureIds?.[0],
  ];
  const destinationBufferId = (record) => {
    if (record?.method === 'copyBufferToBuffer' || record?.method === 'clearBuffer') {
      return record?.destinationBufferId;
    }
    return record?.method === 'copyTextureToBuffer' ? record?.destination?.bufferId : null;
  };
  const destinationTextureId = (record) => (
    record?.method === 'copyBufferToTexture' || record?.method === 'copyTextureToTexture'
      ? record?.destination?.textureId : null
  );
  const immutableCommandBufferWrites = commandTransfers.filter(
    (record) => expectedImmutableIds.includes(destinationBufferId(record)),
  );
  const immutableCommandTextureWrites = commandTransfers.filter(
    (record) => expectedImmutableTextureIds.includes(destinationTextureId(record)),
  );
  const postInstallFrozenWrites = (evidence?.lifecycle?.queueWriteBufferCalls ?? [])
    .filter((record) => record?.sequence > installation?.completeSequence
      && record?.sequence < challenge?.startSequence
      && installation?.frozenResourceIds?.includes(record?.bufferId));
  const postInstallFrozenMaps = (evidence?.lifecycle?.bufferMapEvents ?? [])
    .filter((record) => record?.sequence > installation?.completeSequence
      && record?.sequence < challenge?.startSequence
      && installation?.frozenResourceIds?.includes(record?.resourceId));
  const postInstallFrozenCommandWrites = (
    evidence?.lifecycle?.commandEncoderTransferCalls ?? []
  ).filter((record) => record?.sequence > installation?.completeSequence
    && record?.sequence < challenge?.startSequence
    && installation?.frozenResourceIds?.includes(destinationBufferId(record)));
  const expectedProductionTopology = phases.map((phase, index) => {
    const phaseName = `${phase}/production`;
    const traces = renderPasses.filter((record) => record?.capturePhase === phaseName);
    const expectedBundleId = callbacks[index]?.callback?.bundle?.bundleGpuId;
    const clearTrace = traces[0] ?? null;
    const renderTrace = traces[1] ?? null;
    const drawLike = (trace) => (trace?.events ?? []).filter((event) => [
      'draw', 'drawIndexed', 'drawIndirect', 'drawIndexedIndirect', 'executeBundles',
    ].includes(event?.method));
    return {
      phase: phaseName,
      pass: traces.length === 2
        && drawLike(clearTrace).length === 0
        && clearTrace?.events?.at(-1)?.method === 'end'
        && drawLike(renderTrace).length === 1
        && drawLike(renderTrace)[0]?.method === 'executeBundles'
        && exactArray(drawLike(renderTrace)[0]?.bundleIds, [expectedBundleId])
        && renderTrace?.events?.at(-1)?.method === 'end',
      clearEncoderId: clearTrace?.encoderId ?? null,
      renderEncoderId: renderTrace?.encoderId ?? null,
      expectedBundleId,
      clearDrawLikeEvents: drawLike(clearTrace),
      renderDrawLikeEvents: drawLike(renderTrace),
    };
  });
  const commandReadbackExact = validateThreeImmediatePhase0ProductionReadbackTransfers({
    phases: phases.map((phase) => `${phase}/production`),
    transfers: commandTransfers,
    stagingCreations,
    gpuCreations: evidence?.resources?.gpuCreations ?? [],
    colorTextureId: productionTextureIds[0],
    depthTextureId: productionTextureIds[1],
  }).valid;
  const commandSubmissionExact = validateThreeImmediatePhase0CommandSubmissionClosure({
    phases: phases.map((phase) => `${phase}/production`),
    commandEncoders,
    queueSubmissions,
    renderPasses,
    transfers: commandTransfers,
  }).valid;
  const frozenWrites = queueWrites.filter(
    (record) => installation?.frozenResourceIds?.includes(record?.bufferId),
  );
  const frozenMaps = maps.filter(
    (record) => installation?.frozenResourceIds?.includes(record?.resourceId),
  );
  const immutableWrites = queueWrites.filter(
    (record) => expectedImmutableIds.includes(record?.bufferId),
  );
  const immutableMaps = maps.filter(
    (record) => expectedImmutableIds.includes(record?.resourceId),
  );
  const writeClosure = validateThreeImmediatePhase0ChallengeWriteClosure(challenge, {
    queueWrites,
    queueTextureWrites,
    queueExternalCopies,
    maps,
    frozenIds: installation?.frozenResourceIds ?? [],
    immutableIds: expectedImmutableIds,
  });
  const startDetail = {
    installationId, scenarioId, scheduleId,
    laneOrders: IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  };
  const preflightInventory = challenge?.persistentInventory?.preflight;
  const postflightInventory = challenge?.persistentInventory?.postflight;
  const preflightLabel = `${installationId}/ordered-challenge-preflight`;
  const postflightLabel = `${installationId}/ordered-challenge-postflight`;
  const producerBundlesPrecreated = callbacks.every(({ callback }) => (
    [callback?.bundle, callback?.address?.producer, callback?.output?.objectId?.producer]
      .every((producer) => {
        const matches = (evidence?.lifecycle?.renderBundleTraces ?? []).filter(
          (trace) => trace?.bundleId === producer?.bundleGpuId,
        );
        return matches.length === 1
          && matches[0]?.sequence < challenge?.startSequence
          && matches[0]?.finishSequence < challenge?.startSequence;
      })
  ));
  const productionExecuteSequences = callbacks.map(
    ({ callback }) => callback?.bundle?.execution?.executeSequence,
  );
  const challengeExact = challenge?.schemaVersion === 1
    && challenge?.kind === 'immediate-aif-phase0-ordered-production-challenge'
    && challenge?.pass === true && challenge?.installationId === installationId
    && challenge?.scenarioId === scenarioId && challenge?.scheduleId === scheduleId
    && Number.isSafeInteger(challenge?.startSequence)
    && Number.isSafeInteger(challenge?.endSequence)
    && challenge.startSequence < challenge.endSequence
    && markerExact(markers, challenge.startSequence,
      'phase0-ordered-production-challenge-start', startDetail)
    && markerExact(markers, challenge.endSequence,
      'phase0-ordered-production-challenge-complete', {
        installationId, scenarioId, scheduleId,
      })
    && challenge?.callbackCount === 18
    && exactArray(challenge?.expectedLanes, expectedLanes)
    && exactArray(challenge?.observedLanes, expectedLanes)
    && exactArray(challenge?.productionPhases, phases.map((phase) => `${phase}/production`))
    && challenge?.renderPassCount === 36
    && exactJson(challenge?.phaseRenderPassCounts, byPhase)
    && exactArray(challenge?.expectedRenderPassPhaseOrder, expectedPassPhases)
    && exactArray(challenge?.observedRenderPassPhaseOrder, expectedPassPhases)
    && exactArray(renderPasses.map((record) => record?.capturePhase), expectedPassPhases)
    && exactArray(challenge?.renderPassEncoderIds,
      renderPasses.map((record) => record?.encoderId))
    && exactJson(challenge?.productionRenderPassTopology, expectedProductionTopology)
    && challenge?.productionRenderPassTopologyExact === true
    && expectedProductionTopology.every((record) => record.pass)
    && challenge?.computePassCount === 0
    && exactJson(challenge?.computePasses, computePasses) && computePasses.length === 0
    && challenge?.diagnosticRenderPassCount === 0
    && challenge?.unexpectedRenderPassCount === 0
    && exactArray(challenge?.productionExecuteSequences, productionExecuteSequences)
    && challenge?.executeSequencesStrictlyIncreasing === true
    && productionExecuteSequences.every((sequence, index) => Number.isSafeInteger(sequence)
      && sequence > challenge.startSequence && sequence < challenge.endSequence
      && (index === 0 || sequence > productionExecuteSequences[index - 1]))
    && exactJson(challenge?.creationEvents, creationEvents)
    && challenge?.transientReadbackBufferCreationCount === 36
    && stagingCreations.length === 36
    && phases.every((phase) => stagingCreations.filter(
      (record) => record?.capturePhase === `${phase}/production`
        && record?.size === TARGET_BYTE_LENGTH,
    ).length === 2)
    && challenge?.forbiddenCreationCount === 0
    && exactJson(challenge?.forbiddenCreations, forbiddenCreations)
    && forbiddenCreations.length === 0
    && challenge?.queueWriteBufferCount === queueWrites.length
    && exactJson(challenge?.queueWriteBuffers, queueWrites)
    && challenge?.queueWriteTextureCount === queueTextureWrites.length
    && exactJson(challenge?.queueWriteTextures, queueTextureWrites)
    && challenge?.queueWriteTextureCount === 0 && queueTextureWrites.length === 0
    && challenge?.queueCopyExternalImageToTextureCount === 0
    && exactJson(challenge?.queueCopyExternalImagesToTexture, queueExternalCopies)
    && queueExternalCopies.length === 0
    && exactArray(challenge?.frozenResourceIds, installation?.frozenResourceIds)
    && challenge?.frozenBufferWriteCount === 0
    && exactJson(challenge?.frozenBufferWrites, frozenWrites) && frozenWrites.length === 0
    && challenge?.frozenBufferMapCount === 0
    && exactJson(challenge?.frozenBufferMaps, frozenMaps) && frozenMaps.length === 0
    && new Set(challenge?.immutableResourceIds ?? []).size === expectedImmutableIds.length
    && expectedImmutableIds.every((id) => challenge?.immutableResourceIds?.includes(id))
    && challenge?.immutableBufferWriteCount === 0
    && exactJson(challenge?.immutableBufferWrites, immutableWrites)
    && immutableWrites.length === 0
    && challenge?.immutableBufferMapCount === 0
    && exactJson(challenge?.immutableBufferMaps, immutableMaps)
    && immutableMaps.length === 0
    && writeClosure.valid
    && Array.isArray(challenge?.immutableResources)
    && challenge.immutableResources.length === 24
    && expectedImmutableResources.length === 24
    && exactJson([...challenge.immutableResources].sort(immutableResourceSort),
      [...expectedImmutableResources].sort(immutableResourceSort))
    && exactJson(challenge?.immutableTextureResources,
      expectedImmutableTextureResources)
    && exactArray(challenge?.immutableTextureIds, expectedImmutableTextureIds)
    && new Set(expectedImmutableTextureIds).size === 5
    && challenge?.commandEncoderTransferCount === 36
    && exactJson(challenge?.commandEncoderTransfers, commandTransfers)
    && challenge?.immutableBufferCommandWriteCount === 0
    && exactJson(challenge?.immutableBufferCommandWrites, immutableCommandBufferWrites)
    && immutableCommandBufferWrites.length === 0
    && challenge?.immutableTextureCommandWriteCount === 0
    && exactJson(challenge?.immutableTextureCommandWrites, immutableCommandTextureWrites)
    && immutableCommandTextureWrites.length === 0
    && Number.isSafeInteger(installation?.completeSequence)
    && installation.completeSequence < challenge.startSequence
    && postInstallFrozenWrites.length === 0 && postInstallFrozenMaps.length === 0
    && postInstallFrozenCommandWrites.length === 0
    && challenge?.commandReadbackTopologyExact === true && commandReadbackExact
    && challenge?.commandEncoderCount === 72
    && exactJson(challenge?.commandEncoders, commandEncoders)
    && challenge?.queueSubmissionCount === 72
    && exactJson(challenge?.queueSubmissions, queueSubmissions)
    && challenge?.commandSubmissionTopologyExact === true && commandSubmissionExact
    && producerBundlesPrecreated
    && validatePersistentInventorySnapshot(preflightInventory, preflightLabel,
      scenarioId, creationInventory, markers)
    && validatePersistentInventorySnapshot(postflightInventory, postflightLabel,
      scenarioId, creationInventory, markers)
    && challenge?.persistentInventory?.pass === true
    && challenge?.persistentInventory?.exact === true
    && exactJson(preflightInventory?.records, postflightInventory?.records);
  addReasonUnless(challengeExact, reasons,
    `${installationId} ordered production challenge is not independently exact`);
  return challengeExact;
}

function validatePhase0ScheduleInstallations(evidence, reasons) {
  const scenarios = evidence?.scenarios;
  const traversal = expectedPhase0CallbackTraversal(scenarios, reasons);
  const aggregate = evidence?.resources?.scheduleInstallations;
  const commandAggregate = evidence?.commands?.scheduleInstallations;
  const expectedRecords = [
    ['v99', 'canonical', 1, 'scenario-load-canonical-after-primes'],
    ['v99', 'S1', 2, 'transition-canonical-to-S1'],
    ['v99', 'S2', 3, 'transition-S1-to-S2'],
    ['v99', 'canonical', 4, 'transition-S2-to-canonical'],
    ['v20', 'canonical', 5, 'scenario-load-canonical-after-primes'],
    ['v20', 'S1', 6, 'transition-canonical-to-S1'],
    ['v20', 'S2', 7, 'transition-S1-to-S2'],
    ['v20', 'canonical', 8, 'transition-S2-to-canonical'],
  ].map(([scenarioId, scheduleId, ordinal, reason]) => ({
    scenarioId, scheduleId, ordinal, reason,
    installationId: `${scenarioId}/${ordinal}/${scheduleId}`,
  }));
  const creationInventory = phase0CreationEventInventory(evidence);
  const records = aggregate?.records;
  const retainedByScenarios = (scenarios ?? []).flatMap(
    (scenario) => scenario?.scheduleInstallations ?? [],
  );
  addReasonUnless(aggregate?.schemaVersion === 1
    && aggregate?.kind === 'immediate-aif-phase0-schedule-installations'
    && aggregate?.pass === true && aggregate?.expectedInstallationCount === 8
    && aggregate?.installationCount === 8
    && aggregate?.expectedCommandReadbackCount === 24
    && aggregate?.commandReadbackCount === 24
    && aggregate?.expectedOrderedChallengeCount === 6
    && aggregate?.orderedChallengeCount === 6
    && exactArray(aggregate?.scenarioOrder, expectedRecords.map((record) => record.scenarioId))
    && exactArray(aggregate?.scheduleOrder, expectedRecords.map((record) => record.scheduleId))
    && Array.isArray(records) && records.length === 8
    && exactJson(commandAggregate, aggregate)
    && exactJson(retainedByScenarios, records),
  reasons, 'aggregate schedule installation inventory is not exact');
  for (let index = 0; index < expectedRecords.length; index += 1) {
    const expected = expectedRecords[index];
    const installation = records?.[index];
    validateScheduleInstallationRecord(
      installation, expected, evidence, traversal, creationInventory, reasons,
    );
    const snapshotIndex = expected.ordinal % 4 === 0 ? 3 : (expected.ordinal - 1) % 4;
    const scenario = scenarios?.[expected.scenarioId === 'v99' ? 0 : 1];
    const snapshot = scenario?.snapshots?.[snapshotIndex];
    addReasonUnless(snapshot?.scheduleInstallationId === expected.installationId,
      reasons, `${expected.installationId} is not bound to its snapshot`);
    if (expected.ordinal % 4 !== 0) {
      addReasonUnless(exactJson(installation?.orderedChallenge,
        snapshot?.productionChallenge), reasons,
      `${expected.installationId} challenge copies diverge`);
      validateOrderedProductionChallengeRecord(
        installation?.orderedChallenge, installation,
        { ...expected, snapshotIndex }, evidence, traversal, creationInventory, reasons,
      );
    } else {
      addReasonUnless(installation?.orderedChallenge === null
        && snapshot?.productionChallenge === undefined,
      reasons, `${expected.installationId} canonical audit has an ordered challenge`);
    }
  }
  const expectedBindings = traversal.map((entry, index) => ({
    callbackOrdinal: index + 1,
    phase: entry.phase,
    scenarioId: entry.scenarioId,
    scheduleId: entry.scheduleId,
    lane: entry.lane,
    installationId: entry.installationId,
    commonStatePass: true,
    commandPreflightPass: true,
    commandPreflightInstallationId: entry.installationId,
  }));
  addReasonUnless(traversal.length === 126
    && exactJson(aggregate?.callbackPreflightBindings, expectedBindings),
  reasons, '126 callback-to-schedule-preflight bindings are not exact');
}

function expectedExactWorkloadGeometryAcceptance() {
  let firstIndex = 0;
  const records = GEOMETRY_FINGERPRINT_ORACLES.source.geometries.map((geometry) => {
    const record = {
      bucket: geometry.bucket,
      family: geometry.family,
      name: geometry.name,
      attributeNames: Object.keys(geometry.attributes).sort(),
      vertexCount: geometry.attributes.position.count,
      indexCount: geometry.index.count,
      firstIndex,
      baseVertex: 0,
    };
    firstIndex += record.indexCount;
    return record;
  });
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-exact-workload-validation',
    pass: true,
    geometry: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-exact-medium-geometry-validation',
      pass: true,
      tier: 'medium',
      bucketCount: 32,
      totalIndexCount: firstIndex,
      records,
    },
  };
}

function recomputeJsonRecordSha256(value) {
  if (!isPlainJsonObject(value) || typeof value.sha256 !== 'string') return null;
  const { sha256: ignored, ...payload } = value;
  return sha256(Buffer.from(JSON.stringify(payload)));
}

function validatePinnedGeometrySourceManifest(manifest) {
  if (manifest?.schemaVersion !== 2
    || manifest?.kind !== 'immediate-aif-phase0-geometry-fixture-manifest'
    || manifest?.generator !== 'createIndexedGeometryFixtures'
    || manifest?.tier !== 'medium' || manifest?.bucketCount !== 32
    || manifest?.sha256 !== PINNED_GEOMETRY_SOURCE_SHA256
    || recomputeJsonRecordSha256(manifest) !== PINNED_GEOMETRY_SOURCE_SHA256
    || !Array.isArray(manifest?.geometries) || manifest.geometries.length !== 32) {
    return false;
  }
  return manifest.geometries.every((record, bucket) => (
    record?.bucket === bucket && record?.family === bucket % 4
    && record?.name === `controlled-family-${bucket % 4}-bucket-${bucket}`
    && exactArray(Object.keys(record?.attributes ?? {}), ['normal', 'position', 'uv'])
    && record?.attributes?.position?.count === RUNNER_GEOMETRY_VERTEX_COUNTS[bucket]
    && record?.attributes?.normal?.count === RUNNER_GEOMETRY_VERTEX_COUNTS[bucket]
    && record?.attributes?.uv?.count === RUNNER_GEOMETRY_VERTEX_COUNTS[bucket]
    && record?.index?.count === RUNNER_GEOMETRY_INDEX_COUNTS[bucket]
    && record?.sha256 === PINNED_GEOMETRY_RECORD_SHA256[bucket]
    && recomputeJsonRecordSha256(record) === PINNED_GEOMETRY_RECORD_SHA256[bucket]
  ));
}

function expectedScenarioArrayFingerprints(scenarioId) {
  const pins = PINNED_SCENARIO_FINGERPRINTS[scenarioId];
  const expectedVisibleCount = scenarioId === 'v99' ? 64_881 : 13_107;
  return {
    bucketCounts: { arrayType: 'Uint32Array', length: 32,
      sha256: PINNED_COMMON_SCENARIO_ARRAY_SHA256.bucketCounts },
    bucketBases: { arrayType: 'Uint32Array', length: 32,
      sha256: PINNED_COMMON_SCENARIO_ARRAY_SHA256.bucketBases },
    visibleCounts: { arrayType: 'Uint32Array', length: 32, sha256: pins.visibleCounts },
    objectBuckets: { arrayType: 'Uint32Array', length: 65_536,
      sha256: PINNED_COMMON_SCENARIO_ARRAY_SHA256.objectBuckets },
    matrices: { arrayType: 'Float32Array', length: 1_048_576, sha256: pins.matrices },
    bounds: { arrayType: 'Float32Array', length: 262_144, sha256: pins.bounds },
    expectedVisibleIds: {
      arrayType: 'Uint32Array', length: expectedVisibleCount, sha256: pins.canonical,
    },
    cullOrder: { arrayType: 'Uint32Array', length: 65_536,
      sha256: PINNED_COMMON_SCENARIO_ARRAY_SHA256.cullOrder },
  };
}

function validatePinnedScenarioManifest(manifest, scenarioId) {
  const pins = PINNED_SCENARIO_FINGERPRINTS[scenarioId];
  const fraction = scenarioId === 'v99' ? 0.99 : 0.2;
  const count = scenarioId === 'v99' ? 64_881 : 13_107;
  return manifest?.schemaVersion === 1
    && manifest?.generator === 'createFixedSubsetScenario'
    && manifest?.seed === 0xb1ad_2026
    && manifest?.objectCount === 65_536 && manifest?.bucketCount === 32
    && manifest?.visibilityFraction === fraction && manifest?.layout === 'baseline'
    && manifest?.depthBinRange === null && manifest?.expectedVisibleCount === count
    && manifest?.expectedVisibleIdsCanonicalSha256 === pins.canonical
    && exactJson(manifest?.arrays, expectedScenarioArrayFingerprints(scenarioId))
    && manifest?.sha256 === pins.sha256
    && recomputeJsonRecordSha256(manifest) === pins.sha256;
}

export function validateThreeImmediatePhase0PinnedGeometryManifests(
  sourceManifest, scenarioManifests,
) {
  const reasons = [];
  addReasonUnless(validatePinnedGeometrySourceManifest(sourceManifest), reasons,
    'source geometry manifest differs from pinned reviewed bytes');
  addReasonUnless(exactArray(Object.keys(scenarioManifests ?? {}), VISIBILITY_IDS)
    && VISIBILITY_IDS.every((scenarioId) => validatePinnedScenarioManifest(
      scenarioManifests?.[scenarioId], scenarioId,
    )), reasons, 'scenario manifests differ from pinned reviewed bytes');
  return { valid: reasons.length === 0, reasons };
}

function validateMergedGeometryRealization(evidence, reasons) {
  const realization = evidence?.resources?.mergedGeometryRealization;
  const geometry = evidence?.resources?.geometryFixtures;
  const gpuRecords = geometry?.mergedProduction?.gpuRecords ?? [];
  const expectedSemantics = PINNED_MERGED_GEOMETRY.records.map((record) => record.semantic);
  const markers = evidence?.lifecycle?.instrumentationMarkers ?? [];
  const capturePhase = 'phase0/resources/merged-geometry-realization';
  const writes = (evidence?.lifecycle?.queueWriteBufferCalls ?? []).filter(
    (record) => record?.capturePhase === capturePhase
      && record?.sequence > realization?.startSequence
      && record?.sequence < realization?.queueCompleteSequence,
  );
  const creations = (evidence?.resources?.gpuCreations ?? []).filter(
    (record) => record?.capturePhase === capturePhase
      && record?.sequence > realization?.startSequence
      && record?.sequence < realization?.queueCompleteSequence,
  );
  const maps = (evidence?.lifecycle?.bufferMapEvents ?? []).filter(
    (record) => record?.capturePhase === capturePhase
      && record?.sequence > realization?.startSequence
      && record?.sequence < realization?.queueCompleteSequence,
  );
  const firstChallengeSequence = Math.min(...(evidence?.scenarios ?? []).flatMap(
    (scenario) => (scenario?.snapshots ?? []).slice(0, 3)
      .map((snapshot) => snapshot?.productionChallenge?.startSequence ?? Infinity),
  ));
  const recordsExact = Array.isArray(realization?.records)
    && realization.records.length === expectedSemantics.length
    && realization.records.every((record, index) => {
      const expected = PINNED_MERGED_GEOMETRY.records[index];
      const committed = gpuRecords.find((candidate) => candidate?.semantic === expected.semantic);
      const expectedType = expected.semantic === 'index' ? 2 : 1;
      const expectedBefore = expected.semantic === 'uv' ? null : record?.attributeVersion;
      return record?.semantic === expected.semantic
        && record?.resourceKind === expected.resourceKind
        && record?.manager === 'renderer._attributes'
        && record?.managerConstructor === 'Attributes'
        && record?.attributeId === committed?.attributeId
        && Number.isSafeInteger(record?.attributeVersion) && record.attributeVersion >= 0
        && record?.attributeType === expectedType
        && record?.arrayType === expected.arrayType
        && record?.byteLength === expected.byteLength
        && record?.cpuSha256 === expected.sha256
        && record?.beforeDataVersion === expectedBefore
        && record?.afterDataVersion === record.attributeVersion
        && record?.gpuBufferId === committed?.gpuBufferId
        && record?.gpuBufferSize === expected.byteLength
        && Number.isSafeInteger(record?.startSequence)
        && Number.isSafeInteger(record?.completeSequence)
        && record.startSequence < record.completeSequence
        && markerExact(markers, record.startSequence,
          'phase0-merged-geometry-attribute-update-start', {
            semantic: expected.semantic,
            attributeId: record.attributeId,
            attributeVersion: record.attributeVersion,
            attributeType: expectedType,
            beforeDataVersion: record.beforeDataVersion,
          })
        && markerExact(markers, record.completeSequence,
          'phase0-merged-geometry-attribute-update-complete', {
            semantic: expected.semantic,
            attributeId: record.attributeId,
            attributeVersion: record.attributeVersion,
            attributeType: expectedType,
            afterDataVersion: record.afterDataVersion,
            gpuBufferId: record.gpuBufferId,
          });
    });
  const uv = realization?.records?.find((record) => record?.semantic === 'uv');
  const uvCreations = creations.filter((record) => record?.resourceId === uv?.gpuBufferId);
  addReasonUnless(realization?.schemaVersion === 1
    && realization?.kind === 'immediate-aif-phase0-merged-geometry-realization'
    && realization?.pass === true && realization?.frozen === true
    && realization?.capturePhase === capturePhase
    && realization?.geometryUuid === geometry?.mergedProduction?.geometryUuid
    && Number.isSafeInteger(realization?.startSequence)
    && Number.isSafeInteger(realization?.queueCompleteSequence)
    && realization.startSequence < realization.records?.[0]?.startSequence
    && realization.records?.every((record, index) => index === 0
      || record.startSequence > realization.records[index - 1].completeSequence)
    && realization.records?.at(-1)?.completeSequence < realization.queueCompleteSequence
    && realization.queueCompleteSequence < firstChallengeSequence
    && markerExact(markers, realization.startSequence,
      'phase0-merged-geometry-realization-start', {
        geometryUuid: realization.geometryUuid,
      })
    && markerExact(markers, realization.queueCompleteSequence,
      'phase0-merged-geometry-realization-queue-complete', {
        geometryUuid: realization.geometryUuid,
      })
    && exactArray(realization?.expectedAttributeNames, expectedSemantics.slice(0, 4))
    && realization?.recordCount === 5 && recordsExact
    && realization?.managerVersionsExact === true
    && realization?.queueWriteBufferCount === 0
    && exactJson(realization?.queueWriteBuffers, []) && writes.length === 0
    && exactArray(realization?.frozenResourceIds,
      realization.records?.map((record) => record.gpuBufferId))
    && new Set(realization?.frozenResourceIds ?? []).size === 5
    && creations.length === 1 && uvCreations.length === 1
    && uvCreations[0]?.method === 'createBuffer'
    && uvCreations[0]?.resourceClass === 'persistent-or-upload-buffer'
    && uvCreations[0]?.size === PINNED_MERGED_GEOMETRY.records.find(
      (record) => record.semantic === 'uv',
    ).byteLength
    && uvCreations[0]?.usage === 140 && uvCreations[0]?.mappedAtCreation === true
    && maps.length === 2
    && exactArray(maps.map((record) => record?.method), ['getMappedRange', 'unmap'])
    && maps.every((record) => record?.resourceId === uv?.gpuBufferId)
    && geometry?.realizationBindingExact === true
    && exactJson(geometry?.mergedGeometryRealization, realization),
  reasons, 'merged geometry was not physically realized/frozen before challenges');
}

function validateGeometryResourceEvidence(evidence, reasons) {
  validateMergedGeometryRealization(evidence, reasons);
  const geometry = evidence?.resources?.geometryFixtures;
  const expectedAcceptance = expectedExactWorkloadGeometryAcceptance();
  const expectedSpans = GEOMETRY_FINGERPRINT_ORACLES.source.geometries.map((record) => {
    const prior = GEOMETRY_FINGERPRINT_ORACLES.source.geometries
      .filter((candidate) => candidate.bucket < record.bucket);
    return {
      bucket: record.bucket, family: record.family,
      vertexStart: prior.reduce((sum, candidate) => sum + candidate.attributes.position.count, 0),
      vertexCount: record.attributes.position.count,
      firstIndex: prior.reduce((sum, candidate) => sum + candidate.index.count, 0),
      indexCount: record.index.count,
      baseVertex: 0,
      sourceSha256: record.sha256,
    };
  });
  addReasonUnless(geometry?.schemaVersion === 2
    && geometry?.kind === 'immediate-aif-phase0-geometry-resource-evidence'
    && geometry?.pass === true
    && geometry?.capturePhase === 'phase0/resources/global/geometry-postflight'
    && geometry?.tier === 'medium'
    && exactJson(geometry?.exactAcceptance, expectedAcceptance)
    && validatePinnedGeometrySourceManifest(geometry?.sourceManifest)
    && exactArray(Object.keys(geometry?.scenarioManifests ?? {}), VISIBILITY_IDS)
    && VISIBILITY_IDS.every((scenarioId) => validatePinnedScenarioManifest(
      geometry.scenarioManifests[scenarioId], scenarioId,
    ))
    && exactJson(geometry?.bucketSpans, expectedSpans)
    && geometry?.boundResourceCoverageExact === true,
  reasons, 'exact medium source/scenario geometry fingerprints are invalid');
  const observedRecords = geometry?.mergedProduction?.gpuRecords;
  const expectedRecords = GEOMETRY_FINGERPRINT_ORACLES.merged.records;
  addReasonUnless(geometry?.mergedProduction?.vertexCount
      === GEOMETRY_FINGERPRINT_ORACLES.merged.vertexCount
    && geometry?.mergedProduction?.indexCount
      === GEOMETRY_FINGERPRINT_ORACLES.merged.indexCount
    && Array.isArray(observedRecords)
    && exactArray(observedRecords.map((record) => record?.semantic),
      expectedRecords.map((record) => record.semantic)),
  reasons, 'merged production geometry dimensions/attribute order are invalid');
  const boundIds = new Set();
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const binding = geometry?.laneBindings?.[lane];
    const traces = evidence?.lifecycle?.renderBundleTraces?.filter(
      (trace) => trace?.bundleId === binding?.bundleGpuId,
    ) ?? [];
    const expectedVertexEvents = traces[0]?.events?.filter(
      (event) => event?.method === 'setVertexBuffer',
    ) ?? [];
    const expectedIndexEvents = traces[0]?.events?.filter(
      (event) => event?.method === 'setIndexBuffer',
    ) ?? [];
    const shaderInputs = evidence?.shaders?.lanes?.[lane]?.vertexInputs ?? [];
    const expectedVertexBufferIds = shaderInputs.map((input) => (
      observedRecords?.find((record) => record?.attributeId === input?.resourceId)
        ?.gpuBufferId
    ));
    addReasonUnless(binding?.lane === lane
      && binding?.geometryUuid === evidence?.scenarios?.[0]?.resources?.postflight?.geometries?.[lane]
      && traces.length === 1
      && exactJson(binding?.vertexEvents, expectedVertexEvents)
      && exactJson(binding?.indexEvents, expectedIndexEvents)
      && binding.vertexEvents.length === (lane === 'A' ? 3 : 2)
      && exactArray(binding.vertexEvents.map((event) => event?.bufferId),
        expectedVertexBufferIds)
      && binding.indexEvents.length === 1
      && binding.vertexEvents.every((event, index) => event?.slot === index
        && event?.offset === 0 && event?.size === null)
      && binding.indexEvents[0]?.offset === 0 && binding.indexEvents[0]?.size === null
      && binding.indexEvents[0]?.indexFormat === 'uint32',
    reasons, `${lane} production bundle is not bound to the exact merged indexed geometry`);
    for (const event of [...(binding?.vertexEvents ?? []), ...(binding?.indexEvents ?? [])]) {
      boundIds.add(event?.bufferId);
    }
  }
  const allA = geometry?.laneBindings?.A?.vertexEvents?.map((event) => event.bufferId) ?? [];
  const commonI = geometry?.laneBindings?.I?.vertexEvents?.map((event) => event.bufferId);
  const commonF = geometry?.laneBindings?.F?.vertexEvents?.map((event) => event.bufferId);
  const bucketBaseBufferId = observedRecords?.find(
    (record) => record?.semantic === 'bucketBase',
  )?.gpuBufferId;
  addReasonUnless(exactArray(commonI, commonF)
    && allA.length === commonI?.length + 1
    && commonI.every((bufferId) => allA.includes(bufferId))
    && allA.filter((bufferId) => !commonI.includes(bufferId)).length === 1
    && allA.find((bufferId) => !commonI.includes(bufferId)) === bucketBaseBufferId
    && geometry?.laneBindings?.A?.indexEvents?.[0]?.bufferId
      === geometry?.laneBindings?.I?.indexEvents?.[0]?.bufferId
    && geometry?.laneBindings?.I?.indexEvents?.[0]?.bufferId
      === geometry?.laneBindings?.F?.indexEvents?.[0]?.bufferId,
  reasons, 'A/I/F production lanes do not share exactly the common vertex/index payload');
  for (let index = 0; index < expectedRecords.length; index += 1) {
    const expected = expectedRecords[index];
    const observed = observedRecords?.[index];
    const stable = observed === undefined ? null : {
      semantic: observed.semantic, resourceKind: observed.resourceKind,
      arrayType: observed.arrayType, itemSize: observed.itemSize, count: observed.count,
      normalized: observed.normalized, usage: observed.usage, gpuType: observed.gpuType,
      byteLength: observed.byteLength, sha256: observed.cpuSha256,
    };
    const creations = evidence?.resources?.gpuCreations?.filter(
      (record) => record?.method === 'createBuffer'
        && record?.resourceId === observed?.gpuBufferId,
    ) ?? [];
    const expectedBoundByProduction = expected.semantic !== 'uv';
    addReasonUnless(exactJson(stable, expected)
      && observed?.gpuResident === true
      && typeof observed?.gpuBufferId === 'string'
      && observed?.gpuBufferSize === expected.byteLength
      && observed?.gpuByteLength === expected.byteLength
      && observed?.gpuSha256 === expected.sha256
      && observed?.rawWitnessId === `sha256-${expected.sha256}-${expected.byteLength}`
      && observed?.exact === true
      && observed?.boundByProductionBundle === boundIds.has(observed?.gpuBufferId)
      && observed.boundByProductionBundle === expectedBoundByProduction
      && creations.length === 1
      && creations[0]?.size === expected.byteLength
      && creations[0]?.usage === (expected.resourceKind === 'index' ? 28 : 140),
    reasons, `merged production ${expected.semantic} CPU/GPU/bundle commitment is invalid`);
  }
}

function expectedCameraState(expectedConfiguration) {
  // The candidate configuration is accepted only after its complete canonical
  // bytes match the runner-owned reviewed SHA-256/length commitment. Values are
  // copied rather than regenerated with the same Three.js math used by the page.
  if (!pinnedRenderConfigurationExact(expectedConfiguration)) return null;
  const camera = expectedConfiguration.camera;
  const state = {
    type: camera.type,
    fov: camera.fov,
    aspect: camera.aspect,
    near: camera.near,
    far: camera.far,
    zoom: camera.zoom,
    position: [...camera.position],
    target: [...camera.target],
    targetDistance: camera.targetDistance,
    targetDirection: [...camera.targetDirection],
    worldDirection: [...camera.worldDirection],
    quaternion: [...camera.quaternion],
    scale: [...camera.scale],
    up: [...camera.up],
    coordinateSystem: camera.coordinateSystem,
    reversedDepth: camera.reversedDepth,
    matrixWorld: [...camera.matrixWorld],
    matrixWorldInverse: [...camera.matrixWorldInverse],
    projectionMatrix: [...camera.projectionMatrix],
    projectionMatrixInverse: [...camera.projectionMatrixInverse],
  };
  return { ...state, sha256: sha256(Buffer.from(JSON.stringify(state))) };
}

function validateCommonResourceEvidence(evidence, reasons) {
  const common = evidence?.resources?.commonResources;
  const expected = pinnedRenderConfigurationExact(common?.expected)
    ? common.expected : null;
  addReasonUnless(common?.schemaVersion === 1
    && common?.kind === 'immediate-aif-phase0-common-resource-evidence'
    && common?.pass === true && expected !== null
    && common?.camera?.pass === true
    && exactJson((({ pass, ...state }) => state)(common?.camera ?? {}),
      expectedCameraState(expected)),
  reasons, 'camera matrices/projection/viewport configuration are not exact');
  if (expected === null) {
    reasons.push('common render configuration differs from the runner-owned byte commitment');
    return;
  }
  const expectedColor = PINNED_PRODUCTION_MATERIAL_LINEAR_COLOR;
  const materialStates = [];
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const material = common?.materials?.[lane];
    const { pass, sha256: retainedHash, ...state } = material ?? {};
    const stable = {
      type: state.type, name: state.name, version: state.version,
      color: state.color, colorHex: state.colorHex,
      roughness: state.roughness, metalness: state.metalness,
      opacity: state.opacity, transparent: state.transparent,
      side: state.side, depthTest: state.depthTest, depthWrite: state.depthWrite,
      blending: state.blending, vertexColors: state.vertexColors,
      toneMapped: state.toneMapped, wireframe: state.wireframe,
      alphaTest: state.alphaTest, forceSinglePass: state.forceSinglePass,
      visible: state.visible,
    };
    materialStates.push(stable);
    addReasonUnless(pass === true && state.lane === lane
      && typeof state.uuid === 'string' && state.uuid.length > 0
      && Number.isSafeInteger(state.id) && state.id >= 0
      && retainedHash === sha256(Buffer.from(JSON.stringify(state)))
      && exactJson(state.canonical, expected.material)
      && stable.type === expected.material.type && stable.name === ''
      && Number.isSafeInteger(stable.version) && stable.version >= 0
      && exactArray(stable.color, expectedColor) && stable.colorHex === expected.material.color
      && stable.roughness === expected.material.roughness
      && stable.metalness === expected.material.metalness
      && stable.opacity === expected.material.opacity
      && stable.transparent === expected.material.transparent
      && stable.side === expected.material.side
      && stable.depthTest === expected.material.depthTest
      && stable.depthWrite === expected.material.depthWrite
      && stable.blending === expected.material.blending
      && stable.vertexColors === expected.material.vertexColors
      && stable.toneMapped === expected.material.toneMapped
      && stable.wireframe === expected.material.wireframe
      && stable.alphaTest === expected.material.alphaTest
      && stable.forceSinglePass === expected.material.forceSinglePass
      && stable.visible === expected.material.visible
      && state.addressMode === ({ A: 'bucket-base', I: 'immediate-base',
        F: 'indirect-first-instance' })[lane]
      && state.productionPipelineId === evidence?.shaders?.lanes?.[lane]?.pipelineId,
    reasons, `${lane} production material state/hash is invalid`);
  }
  addReasonUnless(new Set(Object.values(common?.materials ?? {}).map((value) => value?.uuid)).size === 3
    && new Set(Object.values(common?.materials ?? {}).map((value) => value?.id)).size === 3
    && materialStates.every((state) => exactJson(state, materialStates[0])),
  reasons, 'A/I/F materials do not differ solely by addressing node/resource identity');
  const { pass: lightsPass, ...retainedLights } = common?.lights ?? {};
  addReasonUnless(lightsPass === true
    && exactJson(retainedLights, expected.lights)
    && exactArray(retainedLights?.hemisphere?.effectiveNormalizedDirection,
      expected.lights.hemisphere.effectiveNormalizedDirection)
    && exactArray(retainedLights?.directional?.targetDirection,
      expected.lights.directional.targetDirection)
    && exactArray(retainedLights?.directional?.effectiveNormalizedDirection,
      expected.lights.directional.effectiveNormalizedDirection)
    && exactArray(retainedLights?.directional?.target?.worldPosition,
      expected.lights.directional.target.worldPosition),
  reasons, 'production light transforms, target, and derived directions are not exact');
  addReasonUnless(common?.renderer?.pass === true
    && exactJson(common.renderer?.viewport, expected.viewport)
    && common.renderer?.antialias === expected.renderer.antialias
    && common.renderer?.samples === expected.renderer.samples
    && common.renderer?.powerPreference === expected.renderer.powerPreference
    && common.renderer?.forceWebGL === expected.renderer.forceWebGL
    && common.renderer?.alpha === expected.renderer.alpha
    && common.renderer?.depth === expected.renderer.depth
    && common.renderer?.stencil === expected.renderer.stencil
    && common.renderer?.logarithmicDepthBuffer
      === expected.renderer.logarithmicDepthBuffer
    && common.renderer?.outputColorSpace === expected.renderer.outputColorSpace
    && common.renderer?.reversedDepthBuffer === expected.renderer.reversedDepthBuffer
    && common.renderer?.parameterReversedDepthBuffer
      === expected.renderer.reversedDepthBuffer
    && common.renderer?.trackTimestamp === expected.renderer.trackTimestamp
    && common.renderer?.parameterTrackTimestamp === expected.renderer.trackTimestamp
    && common.renderer?.timestampQueryPoolsEmpty === true
    && common.renderer?.autoClear === expected.renderer.autoClear
    && common.renderer?.autoClearColor === expected.renderer.autoClearColor
    && common.renderer?.autoClearDepth === expected.renderer.autoClearDepth
    && common.renderer?.autoClearStencil === expected.renderer.autoClearStencil
    && common.renderer?.toneMapping === expected.renderer.toneMapping
    && common.renderer?.toneMappingExposure === expected.renderer.toneMappingExposure
    && typeof common.renderer?.backendDeviceId === 'string'
    && common.renderer.backendDeviceId.length > 0
    && common.renderer?.backendParameterDeviceId === common.renderer.backendDeviceId
    && common.renderer?.backendDeviceMatchesParameter === true
    && common.renderer?.sortObjects === expected.renderer.sortObjects
    && common.renderer?.clearColor === expected.clear.color
    && common.renderer?.clearAlpha === expected.clear.alpha
    && common.renderer?.sceneBackground === expected.clear.color,
  reasons, 'renderer clear/color-space/reversed-depth state is invalid');

  const outputWitnessIds = evidence?.outputWitnesses?.witnesses === undefined ? []
    : Object.keys(evidence.outputWitnesses.witnesses);
  const addressWitnessIds = Object.keys(evidence?.addressWitnesses?.witnesses ?? {}).sort();
  for (const name of ['production', 'objectId', 'address']) {
    const target = common?.targets?.[name];
    const targetExpected = expected.targets[name];
    const purpose = name === 'production'
      ? { role: 'production-color-depth', outputChannels: ['color', 'depth'] }
      : name === 'objectId'
        ? { role: 'diagnostic-object-id', outputChannels: ['objectId'] }
        : { role: 'diagnostic-exact-address', outputChannels: [] };
    const expectedWitnessIds = name === 'address' ? [] : outputWitnessIds.filter(
      (id) => purpose.outputChannels.includes(id.split('/')[0]),
    ).sort();
    const colorCreationMatches = evidence?.resources?.gpuCreations?.filter(
      (record) => record?.method === 'createTexture'
        && record?.resourceId === target?.colorTextureId,
    ) ?? [];
    const depthCreationMatches = (target?.depthTextureIds ?? []).flatMap((id) => (
      evidence?.resources?.gpuCreations?.filter((record) => record?.method === 'createTexture'
        && record?.resourceId === id) ?? []
    ));
    const passMatches = evidence?.lifecycle?.renderPassTraces?.filter((trace) => (
      trace?.colorAttachments?.some((attachment) => attachment?.textureId === target?.colorTextureId)
    )) ?? [];
    const executedBundleIds = [...new Set(passMatches.flatMap((trace) => (
      trace?.events?.filter((event) => event?.method === 'executeBundles')
        .flatMap((event) => event?.bundleIds ?? []) ?? []
    )))];
    const executedPipelineIds = [...new Set((evidence?.lifecycle?.renderBundleTraces ?? [])
      .filter((trace) => executedBundleIds.includes(trace?.bundleId))
      .flatMap((trace) => trace?.events?.filter((event) => event?.method === 'setPipeline')
        .map((event) => event?.pipelineId) ?? []))];
    const expectedLogical = {
      width: targetExpected.width,
      height: targetExpected.height,
      depth: targetExpected.depth,
      count: targetExpected.count,
      samples: targetExpected.samples,
      depthBuffer: targetExpected.depthBuffer,
      stencilBuffer: targetExpected.stencilBuffer,
      resolveDepthBuffer: targetExpected.resolveDepthBuffer,
      resolveStencilBuffer: targetExpected.resolveStencilBuffer,
      multiview: targetExpected.multiview,
      useArrayDepthTexture: targetExpected.useArrayDepthTexture,
      colorSpace: targetExpected.colorSpace,
      colorTexture: targetExpected.colorTexture,
      depthTexture: targetExpected.depthTexture,
    };
    addReasonUnless(target?.schemaVersion === 1
      && target?.kind === 'immediate-aif-phase0-target-resource-evidence'
      && target?.pass === true && target?.name === name
      && exactJson(target?.purpose, purpose) && exactJson(target?.expected, targetExpected)
      && exactJson(target?.logical, expectedLogical)
      && colorCreationMatches.length === 1
      && exactJson(target?.colorTextureCreation, colorCreationMatches[0])
      && exactJson(target?.gpuColorDescriptor, targetExpected.gpuColorDescriptor)
      && colorCreationMatches[0]?.format === targetExpected.colorFormat
      && colorCreationMatches[0]?.dimension === '2d'
      && colorCreationMatches[0]?.mipLevelCount === 1
      && colorCreationMatches[0]?.sampleCount === 1
      && colorCreationMatches[0]?.usage === 23
      && textureSizeExact(colorCreationMatches[0]?.size, targetExpected.width, targetExpected.height)
      && (targetExpected.depthBuffer
        ? target?.depthTextureIds?.length === 1
          && depthCreationMatches.length === 1
          && exactJson(target?.depthTextureCreation, depthCreationMatches[0])
          && exactJson(target?.gpuDepthDescriptor, targetExpected.gpuDepthDescriptor)
          && depthCreationMatches[0]?.format === targetExpected.depthFormat
          && depthCreationMatches[0]?.dimension === '2d'
          && depthCreationMatches[0]?.mipLevelCount === 1
          && depthCreationMatches[0]?.sampleCount === 1
          && depthCreationMatches[0]?.usage === 23
          && textureSizeExact(depthCreationMatches[0]?.size,
            targetExpected.width, targetExpected.height)
        : target?.depthTextureIds?.length === 0 && target?.depthTextureCreation === null
          && target?.gpuDepthDescriptor === null)
      && exactArray(target?.renderPassEncoderIds, passMatches.map((trace) => trace.encoderId))
      && passMatches.length > 0
      && passMatches.every((trace) => trace?.timestampWrites === null
        && trace?.occlusionQuerySetId === null)
      && exactArray(target?.executedBundleIds, executedBundleIds)
      && exactArray(target?.executedPipelineIds, executedPipelineIds)
      && executedBundleIds.length > 0 && executedPipelineIds.length > 0
      && exactArray(target?.outputWitnessIds, expectedWitnessIds)
      && (name !== 'address' || exactArray(target?.addressWitnessIds, addressWitnessIds)),
    reasons, `${name} render target/texture/pass/witness chain is invalid`);
  }
}

export function validateThreeImmediatePhase0GpuByteWitnesses(evidence) {
  const reasons = [];
  const container = evidence?.resources?.gpuByteWitnesses;
  const expectedReferences = [];
  for (const scenarioId of VISIBILITY_IDS) {
    const scenario = evidence?.scenarios?.find(
      (candidate) => candidate?.scenarioId === scenarioId,
    );
    const bySemantic = Object.fromEntries(
      (scenario?.gpuResourceCommitments?.records ?? []).map(
        (record) => [record?.semantic, record],
      ),
    );
    for (const semantic of PHASE0_SHARED_RESOURCE_SEMANTICS) {
      const oracle = cpuOracles()[scenarioId].sharedResources[semantic];
      const bytes = Buffer.from(oracle.expectedBytes);
      const hash = sha256(bytes);
      expectedReferences.push({
        witnessId: `sha256-${hash}-${bytes.byteLength}`,
        semantic: `shared/${scenarioId}/${semantic}`,
        gpuBufferId: bySemantic[semantic]?.gpuBufferId,
        capturePhase: `phase0/resources/${scenarioId}/shared-commitment`,
        byteLength: bytes.byteLength,
        sha256: hash,
        bytes,
        retainedWitnessId: bySemantic[semantic]?.rawWitnessId,
      });
    }
  }
  const geometryRecords = evidence?.resources?.geometryFixtures
    ?.mergedProduction?.gpuRecords ?? [];
  const geometryBySemantic = Object.fromEntries(
    geometryRecords.map((record) => [record?.semantic, record]),
  );
  for (const record of PINNED_MERGED_GEOMETRY.records) {
    const bytes = Buffer.from(runnerMergedGeometryByteOracles()[record.semantic]);
    const hash = sha256(bytes);
    expectedReferences.push({
      witnessId: `sha256-${hash}-${bytes.byteLength}`,
      semantic: `merged-geometry/${record.semantic}`,
      gpuBufferId: geometryBySemantic[record.semantic]?.gpuBufferId,
      capturePhase: 'phase0/resources/global/geometry-postflight',
      byteLength: bytes.byteLength,
      sha256: hash,
      bytes,
      retainedWitnessId: geometryBySemantic[record.semantic]?.rawWitnessId,
    });
  }
  const expectedIds = [...new Set(expectedReferences.map((record) => record.witnessId))]
    .sort((left, right) => left.localeCompare(right));
  addReasonUnless(container?.schemaVersion === 1
    && container?.kind === 'immediate-aif-phase0-gpu-byte-witnesses'
    && container?.pass === true
    && container?.referenceCount === expectedReferences.length
    && container?.witnessCount === expectedIds.length
    && exactArray(Object.keys(container?.witnesses ?? {}), expectedIds)
    && Array.isArray(container?.references)
    && container.references.length === expectedReferences.length,
  reasons, 'GPU byte witness envelope/order/count is not exact');
  for (let index = 0; index < expectedReferences.length; index += 1) {
    const expected = expectedReferences[index];
    const reference = container?.references?.[index];
    const witness = container?.witnesses?.[expected.witnessId];
    const decoded = decodeCanonicalBase64(
      witness?.bytesBase64, `${expected.semantic} GPU byte witness`, reasons,
    );
    addReasonUnless(expected.retainedWitnessId === expected.witnessId
      && exactJson(reference, {
        witnessId: expected.witnessId,
        semantic: expected.semantic,
        gpuBufferId: expected.gpuBufferId,
        capturePhase: expected.capturePhase,
        byteLength: expected.byteLength,
        sha256: expected.sha256,
      })
      && witness?.schemaVersion === 1
      && witness?.kind === 'immediate-aif-phase0-gpu-byte-witness'
      && witness?.witnessId === expected.witnessId
      && witness?.encoding === 'base64-exact-bytes'
      && witness?.byteLength === expected.byteLength
      && witness?.sha256 === expected.sha256
      && decoded?.byteLength === expected.byteLength
      && decoded?.equals(expected.bytes)
      && (decoded === null || sha256(decoded) === expected.sha256),
    reasons, `${expected.semantic} raw GPU bytes differ from the independent oracle`);
  }
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

const RUNNER_QUEUE_WRITE_BUFFER_CATEGORY_ORDER = Object.freeze([
  'renderer-initialization',
  'scenario-matrix-realization',
  'live-compute',
  'shared-resource-commitment',
  'merged-geometry-realization',
  'schedule-attribute-realization',
  'schedule-diagnostic-position-realization',
  'production-render',
  'diagnostic-render',
]);

function runnerQueueWriteBufferCategory(capturePhase) {
  if (capturePhase === 'phase0/renderer-init') return 'renderer-initialization';
  if (/^phase0\/scenario-load\/(?:v99|v20)\/matrix-realization$/u.test(capturePhase)) {
    return 'scenario-matrix-realization';
  }
  if (/^phase0\/live\/(?:v99|v20)\/[AIF]$/u.test(capturePhase)) return 'live-compute';
  if (/^phase0\/resources\/(?:v99|v20)\/shared-commitment$/u.test(capturePhase)) {
    return 'shared-resource-commitment';
  }
  if (capturePhase === 'phase0/resources/merged-geometry-realization') {
    return 'merged-geometry-realization';
  }
  if (/^phase0\/schedule-install\/(?:v99|v20)\/\d+-(?:canonical|S1|S2)\/attribute-realization$/u.test(capturePhase)) {
    return 'schedule-attribute-realization';
  }
  if (/^phase0\/schedule-install\/(?:v99|v20)\/\d+-(?:canonical|S1|S2)\/diagnostic-position-realization$/u.test(capturePhase)) {
    return 'schedule-diagnostic-position-realization';
  }
  if (/^phase0\/prime\/[AIF]$/u.test(capturePhase)
    || /^phase0\/(?:v99|v20)\/(?:canonical|S1|S2)\/[AIF]\/capture-\d+\/production$/u.test(capturePhase)) {
    return 'production-render';
  }
  if (/^phase0\/diagnostic-(?:prime|rerecord)\//u.test(capturePhase)
    || /^phase0\/(?:v99|v20)\/(?:canonical|S1|S2)\/[AIF]\/capture-\d+\/(?:address|object-id)$/u.test(capturePhase)) {
    return 'diagnostic-render';
  }
  return null;
}

function runnerQueueWriteBufferProjection(record) {
  return {
    sequence: record?.sequence,
    capturePhase: record?.capturePhase,
    bufferId: record?.bufferId,
    bufferOffset: record?.bufferOffset,
    sourceId: record?.sourceId,
    sourceBackingBufferId: record?.sourceBackingBufferId,
    sourceType: record?.sourceType,
    sourceByteLength: record?.sourceByteLength,
    sourceByteOffset: record?.sourceByteOffset,
    sourceElementByteSize: record?.sourceElementByteSize,
    sourceElementCount: record?.sourceElementCount,
    dataOffset: record?.dataOffset,
    size: record?.size,
    selectedSourceElementOffset: record?.selectedSourceElementOffset,
    selectedSourceElementLength: record?.selectedSourceElementLength,
    selectedSourceByteOffset: record?.selectedSourceByteOffset,
    selectedSourceByteLength: record?.selectedSourceByteLength,
    selectedSourceSha256: record?.selectedSourceSha256,
  };
}

function runnerUniformNullDynamicOffsets(event) {
  return event?.dynamicOffsets === null
    && event?.dynamicOffsetStart === null
    && event?.dynamicOffsetLength === null
    && event?.selectedDynamicOffsets === null;
}

function runnerUniformActiveBindGroupUseWitness(
  events, bindGroupId, writeSequence, usePredicate,
) {
  const activeSlots = new Map();
  for (const event of events ?? []) {
    if (event?.method === 'setBindGroup') {
      activeSlots.set(event.index, event);
      continue;
    }
    if (!usePredicate(event) || event?.sequence <= writeSequence) continue;
    const activeMatches = [...activeSlots].filter(([, active]) => (
      active?.bindGroupId === bindGroupId
        && Number.isSafeInteger(active?.sequence)
        && active.sequence < event.sequence
        && runnerUniformNullDynamicOffsets(active)
    ));
    if (activeMatches.length !== 1) continue;
    const [[bindGroupIndex, active]] = activeMatches;
    return {
      useEvent: event,
      bindGroupIndex,
      activeSetBindGroupSequence: active.sequence,
      activeTargetBindGroupMatchCount: activeMatches.length,
    };
  }
  return null;
}

function runnerUniformCommandCompletionProof(
  trace, useSequence, useKind, lifecycle, extra = {},
) {
  const commandEncoders = (lifecycle?.commandEncoderTraces ?? []).filter(
    (encoder) => encoder?.commandEncoderId === trace?.commandEncoderId,
  );
  const commandEncoder = commandEncoders.length === 1 ? commandEncoders[0] : null;
  const submissions = commandEncoder === null ? []
    : (lifecycle?.queueSubmissions ?? []).filter((submission) => (
      exactArray(submission?.commandBufferIds, [commandEncoder?.commandBufferId])
    ));
  const submission = submissions.length === 1 ? submissions[0] : null;
  return {
    useKind,
    useSequence,
    passEncoderId: trace?.encoderId,
    commandEncoderId: trace?.commandEncoderId,
    commandEncoderMatchCount: commandEncoders.length,
    commandBufferId: commandEncoder?.commandBufferId ?? null,
    commandEncoderFinishSequence: commandEncoder?.finishSequence ?? null,
    submissionMatchCount: submissions.length,
    submitSequence: submission?.sequence ?? null,
    ...extra,
    pass: commandEncoders.length === 1
      && submissions.length === 1
      && trace?.capturePhase === submission?.capturePhase
      && useSequence < commandEncoder?.finishSequence
      && commandEncoder.finishSequence < submission?.sequence,
  };
}

function runnerUniformExecutionProofs(bindGroupId, capturePhase, writeSequence, lifecycle) {
  const proofs = [];
  for (const trace of (lifecycle?.computePassTraces ?? []).filter(
    (candidate) => candidate?.capturePhase === capturePhase,
  )) {
    const witness = runnerUniformActiveBindGroupUseWitness(
      trace?.events,
      bindGroupId,
      writeSequence,
      (event) => event?.method === 'dispatchWorkgroups'
        || event?.method === 'dispatchWorkgroupsIndirect',
    );
    if (witness !== null) {
      proofs.push(runnerUniformCommandCompletionProof(
        trace, witness.useEvent.sequence, 'compute-dispatch', lifecycle, {
          setBindGroupSequence: witness.activeSetBindGroupSequence,
          bindGroupIndex: witness.bindGroupIndex,
          activeSetBindGroupSequence: witness.activeSetBindGroupSequence,
          activeTargetBindGroupMatchCount: witness.activeTargetBindGroupMatchCount,
          bindGroupId,
        },
      ));
    }
  }
  for (const trace of (lifecycle?.renderPassTraces ?? []).filter(
    (candidate) => candidate?.capturePhase === capturePhase,
  )) {
    const directWitness = runnerUniformActiveBindGroupUseWitness(
      trace?.events,
      bindGroupId,
      writeSequence,
      (event) => /^draw/u.test(event?.method ?? ''),
    );
    let directProof = null;
    if (directWitness !== null) {
      directProof = runnerUniformCommandCompletionProof(
        trace, directWitness.useEvent.sequence, 'render-pass-draw', lifecycle, {
          setBindGroupSequence: directWitness.activeSetBindGroupSequence,
          bindGroupIndex: directWitness.bindGroupIndex,
          activeSetBindGroupSequence: directWitness.activeSetBindGroupSequence,
          activeTargetBindGroupMatchCount: directWitness.activeTargetBindGroupMatchCount,
          bindGroupId,
        },
      );
    }
    let bundleProof = null;
    for (const execute of trace?.events?.filter(
      (event) => event?.method === 'executeBundles' && writeSequence < event?.sequence,
    ) ?? []) {
      for (const bundleId of execute?.bundleIds ?? []) {
        const bundleMatches = (lifecycle?.renderBundleTraces ?? []).filter(
          (bundle) => bundle?.bundleId === bundleId,
        );
        const bundle = bundleMatches.length === 1 ? bundleMatches[0] : null;
        const bundleWitness = bundle === null ? null
          : runnerUniformActiveBindGroupUseWitness(
            bundle?.events,
            bindGroupId,
            Number.NEGATIVE_INFINITY,
            (event) => /^draw/u.test(event?.method ?? ''),
          );
        if (bundle === null || bundleWitness === null) continue;
        const proof = runnerUniformCommandCompletionProof(
          trace, execute.sequence, 'executed-render-bundle', lifecycle, {
            bindGroupId,
            bindGroupIndex: bundleWitness.bindGroupIndex,
            activeSetBindGroupSequence: bundleWitness.activeSetBindGroupSequence,
            activeTargetBindGroupMatchCount: bundleWitness.activeTargetBindGroupMatchCount,
            bundleId,
            bundleMatchCount: bundleMatches.length,
            bundleSetBindGroupSequence: bundleWitness.activeSetBindGroupSequence,
            bundleDrawSequence: bundleWitness.useEvent.sequence,
            bundleFinishSequence: bundle.finishSequence,
          },
        );
        proof.pass = proof.pass
          && bundleMatches.length === 1
          && bundleWitness.activeSetBindGroupSequence < bundleWitness.useEvent.sequence
          && bundle.finishSequence < execute.sequence;
        bundleProof = proof;
        break;
      }
      if (bundleProof !== null) break;
    }
    const selectedTraceProof = [directProof, bundleProof]
      .filter((proof) => proof !== null)
      .sort((left, right) => left.useSequence - right.useSequence)[0] ?? null;
    if (selectedTraceProof !== null) proofs.push(selectedTraceProof);
  }
  return proofs;
}

export function reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
  records,
  { bufferCreations = [], lifecycle = {} } = {},
) {
  const uniformWriteGrammar = (records ?? []).filter(
    (record) => record?.destinationOwner?.destinationBindingClass
      === 'uniform-bind-group-buffer',
  ).map((record) => {
    const creations = bufferCreations.filter(
      (candidate) => candidate?.resourceId === record?.bufferId,
    );
    const creation = creations.length === 1 ? creations[0] : null;
    const bindingProofs = record?.destinationOwner?.uniformBindingProofs ?? [];
    const binding = bindingProofs.length === 1 ? bindingProofs[0] : null;
    const executionProofs = binding === null ? [] : runnerUniformExecutionProofs(
      binding.bindGroupId,
      record.capturePhase,
      record.sequence,
      lifecycle,
    );
    const samePhaseDestinationWrites = (records ?? []).filter(
      (candidate) => candidate?.capturePhase === record?.capturePhase
        && candidate?.bufferId === record?.bufferId,
    );
    const boundStart = binding?.bufferOffset ?? null;
    const boundByteLength = binding === null || creation === null
      ? null : binding.bufferSize ?? creation.size - binding.bufferOffset;
    const boundEnd = boundStart === null || boundByteLength === null
      ? null : boundStart + boundByteLength;
    const sourceRange = {
      start: record?.selectedSourceByteOffset,
      byteLength: record?.selectedSourceByteLength,
      end: record?.selectedSourceByteOffset + record?.selectedSourceByteLength,
    };
    const destinationRange = {
      start: record?.bufferOffset,
      byteLength: record?.selectedSourceByteLength,
      end: record?.bufferOffset + record?.selectedSourceByteLength,
    };
    const rangePositive = Number.isSafeInteger(sourceRange.byteLength)
      && sourceRange.byteLength > 0
      && destinationRange.byteLength === sourceRange.byteLength;
    const rangeAligned = [
      sourceRange.start,
      sourceRange.byteLength,
      destinationRange.start,
      destinationRange.byteLength,
    ].every((value) => Number.isSafeInteger(value) && value % 4 === 0);
    const sourceRangeWithinData = Number.isSafeInteger(record?.sourceByteLength)
      && sourceRange.start >= 0
      && sourceRange.end <= record.sourceByteLength;
    const destinationRangeWithinCreation = Number.isSafeInteger(creation?.size)
      && destinationRange.start >= 0
      && destinationRange.end <= creation.size;
    const destinationRangeWithinBinding = Number.isSafeInteger(boundStart)
      && Number.isSafeInteger(boundByteLength)
      && boundByteLength > 0
      && destinationRange.start >= boundStart
      && destinationRange.end <= boundEnd;
    const sourceDestinationOffsetExact = Number.isSafeInteger(boundStart)
      && sourceRange.start === destinationRange.start - boundStart;
    return {
      pass: creations.length === 1
        && creation?.usage === 72
        && binding !== null
        && binding.hasDynamicOffset === false
        && record.category !== 'renderer-initialization'
        && rangePositive
        && rangeAligned
        && sourceRangeWithinData
        && destinationRangeWithinCreation
        && destinationRangeWithinBinding
        && sourceDestinationOffsetExact
        && executionProofs.length === 1
        && executionProofs[0].pass,
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: record.category,
      bufferId: record.bufferId,
      creationMatchCount: creations.length,
      creation,
      binding,
      alignmentBytes: 4,
      sourceRange,
      destinationRange,
      boundRange: { start: boundStart, byteLength: boundByteLength, end: boundEnd },
      rangePositive,
      rangeAligned,
      sourceRangeWithinData,
      destinationRangeWithinCreation,
      destinationRangeWithinBinding,
      sourceDestinationOffsetExact,
      boundByteLength,
      samePhaseDestinationWriteCount: samePhaseDestinationWrites.length,
      initializationOnly: false,
      executionProofs,
    };
  });
  const uniformWriteGroupMap = new Map();
  for (const grammar of uniformWriteGrammar) {
    const groupId = `${grammar.capturePhase}\u0000${grammar.bufferId}`;
    const group = uniformWriteGroupMap.get(groupId) ?? [];
    group.push(grammar);
    uniformWriteGroupMap.set(groupId, group);
  }
  const uniformWriteGroups = [...uniformWriteGroupMap].map(([groupId, groupRecords]) => {
    const [first] = groupRecords;
    const orderedDestinationRanges = groupRecords.map((record) => ({
      sequence: record.sequence,
      ...record.destinationRange,
    })).sort((left, right) => left.start - right.start || left.sequence - right.sequence);
    let furthestDestinationEnd = -1;
    let observedDestinationOverlapCount = 0;
    for (const range of orderedDestinationRanges) {
      if (range.start < furthestDestinationEnd) observedDestinationOverlapCount += 1;
      furthestDestinationEnd = Math.max(furthestDestinationEnd, range.end);
    }
    const writeSequencesStrictlyIncreasing = groupRecords.every(
      (record, index) => index === 0 || record.sequence > groupRecords[index - 1].sequence,
    );
    const sameCreationAndBinding = groupRecords.every((record) => (
      exactJson(record.creation, first.creation) && exactJson(record.binding, first.binding)
    ));
    const pass = groupRecords.length > 0
      && groupRecords.every((record) => record.pass)
      && new Set(groupRecords.map((record) => record.capturePhase)).size === 1
      && new Set(groupRecords.map((record) => record.bufferId)).size === 1
      && new Set(groupRecords.map((record) => record.category)).size === 1
      && sameCreationAndBinding
      && writeSequencesStrictlyIncreasing;
    return {
      pass,
      groupId,
      capturePhase: first.capturePhase,
      category: first.category,
      bufferId: first.bufferId,
      creation: first.creation,
      binding: first.binding,
      boundRange: first.boundRange,
      writeCount: groupRecords.length,
      writeSequences: groupRecords.map((record) => record.sequence),
      writeSequencesStrictlyIncreasing,
      sameCreationAndBinding,
      overlapPolicy: 'observed-not-prescribed',
      observedDestinationOverlapCount,
      orderedDestinationRanges,
      everyWriteExecuted: groupRecords.every(
        (record) => record.executionProofs.length === 1
          && record.executionProofs[0].pass,
      ),
      records: groupRecords,
    };
  });
  const valid = uniformWriteGrammar.every((record) => record.pass)
    && uniformWriteGroups.every((group) => group.pass)
    && uniformWriteGroups.reduce((count, group) => count + group.writeCount, 0)
      === uniformWriteGrammar.length;
  return {
    valid,
    expected: {
      uniformWriteGrammarExact: true,
      uniformWriteGrammar,
      uniformWriteGroupCount: uniformWriteGroups.length,
      uniformWriteGroupsExact: true,
      uniformWriteGroups,
    },
  };
}

export function validateThreeImmediatePhase0UniformQueueWriteEvidence(
  retained,
  { records = [], bufferCreations = [], lifecycle = {} } = {},
) {
  const reasons = [];
  const reconstructed = reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
    records, { bufferCreations, lifecycle },
  );
  addReasonUnless(reconstructed.valid,
    reasons, 'uniform queue writes do not have exact ranges and consuming executions');
  addReasonUnless(exactJson(retained, reconstructed.expected),
    reasons, 'retained uniform queue-write evidence differs from raw reconstruction');
  return { valid: reasons.length === 0, reasons, expected: reconstructed.expected };
}

export function validateThreeImmediatePhase0QueueWriteBufferLedger(
  ledger,
  { lifecycle = {}, scenarios = [], resources = {}, pipelines = {} } = {},
) {
  const reasons = [];
  const rawRecords = lifecycle?.queueWriteBufferCalls ?? [];
  const bufferCreations = resources?.gpuCreations?.filter(
    (record) => record?.method === 'createBuffer',
  ) ?? [];
  const semanticBindings = [];
  const addSemanticBinding = (semantic, attributeId, bufferId) => {
    semanticBindings.push({ semantic, attributeId, bufferId });
  };
  const firstScenarioCommitments = scenarios?.[0]?.gpuResourceCommitments?.records ?? [];
  for (const semantic of PHASE0_SHARED_RESOURCE_SEMANTICS) {
    const record = firstScenarioCommitments.find(
      (candidate) => candidate?.semantic === semantic,
    );
    addSemanticBinding(`shared.${semantic}`, record?.attributeId, record?.gpuBufferId);
  }
  const installations = resources?.scheduleInstallations?.records ?? [];
  const mergedRecords = resources?.geometryFixtures?.mergedProduction?.gpuRecords ?? [];
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const command = installations?.[0]?.commands?.[lane];
    addSemanticBinding(`command.${lane}`, command?.attributeId, command?.gpuBufferId);
    for (const semantic of ['position', 'normal', 'uv', ...(lane === 'A' ? ['bucketBase'] : [])]) {
      const record = mergedRecords.find((candidate) => candidate?.semantic === semantic);
      addSemanticBinding(`geometry.${lane}.${semantic}`,
        record?.attributeId, record?.gpuBufferId);
    }
    const index = mergedRecords.find((candidate) => candidate?.semantic === 'index');
    addSemanticBinding(`geometry.${lane}.index`, index?.attributeId, index?.gpuBufferId);
  }
  const firstAddressRealization = resources?.addressDiagnosticPositions?.records?.[0];
  for (const semantic of ['commonPosition', 'featurePosition']) {
    const record = firstAddressRealization?.records?.find(
      (candidate) => candidate?.semantic === semantic,
    );
    addSemanticBinding(`diagnostic.address.${semantic}`,
      record?.attributeId, record?.gpuBufferId);
  }
  const bindGroups = resources?.gpuCreations?.filter(
    (record) => record?.method === 'createBindGroup',
  ) ?? [];
  const retainedBindGroupLayouts = Array.isArray(pipelines?.bindGroupLayouts)
    ? pipelines.bindGroupLayouts : [];
  const uniformBindingProofs = (bufferId) => {
    const occurrences = bindGroups.flatMap((bindGroup) => (
      (bindGroup?.entries ?? []).filter((entry) => entry?.bufferId === bufferId)
        .map((entry) => ({ bindGroup, entry }))
    ));
    if (occurrences.length !== 1) return [];
    return occurrences.flatMap(({ bindGroup, entry }) => {
        const layoutMatches = retainedBindGroupLayouts.filter(
          (layout) => layout?.layoutId === bindGroup?.layoutId,
        );
        const coordinateMatches = layoutMatches.flatMap((layout) => (
          (layout?.entries ?? []).filter(
            (candidate) => candidate?.binding === entry?.binding,
          )
        ));
        const effectiveBufferType = coordinateMatches[0]?.buffer === null
          || coordinateMatches[0]?.buffer === undefined
          ? null : coordinateMatches[0].buffer.type ?? 'uniform';
        if (layoutMatches.length !== 1 || coordinateMatches.length !== 1
          || effectiveBufferType !== 'uniform') return [];
        return [{
          exact: true,
          bindGroupId: bindGroup.resourceId,
          bindGroupCreateSequence: bindGroup.sequence,
          bindGroupCapturePhase: bindGroup.capturePhase,
          bindGroupLayoutId: bindGroup.layoutId,
          bindGroupLayoutCreateSequence: layoutMatches[0].sequence,
          binding: entry.binding,
          bufferId: entry.bufferId,
          bufferOffset: entry.bufferOffset,
          bufferSize: entry.bufferSize,
          bufferRawType: coordinateMatches[0]?.buffer?.type ?? null,
          bufferType: effectiveBufferType,
          hasDynamicOffset: coordinateMatches[0].buffer.hasDynamicOffset,
          minBindingSize: coordinateMatches[0].buffer.minBindingSize,
          layoutMatchCount: layoutMatches.length,
          coordinateMatchCount: coordinateMatches.length,
        }];
      });
  };
  const records = rawRecords.map((record) => ({
    category: runnerQueueWriteBufferCategory(record?.capturePhase),
    ...runnerQueueWriteBufferProjection(record),
    destinationOwner: (() => {
      const matches = bufferCreations.filter(
        (creation) => creation?.resourceId === record?.bufferId,
      );
      const creation = matches.length === 1 ? matches[0] : null;
      const matchedSemanticBindings = semanticBindings.filter(
        (binding) => binding.bufferId === record?.bufferId,
      ).map(({ semantic, attributeId }) => ({ semantic, attributeId }));
      const matchedUniformBindingProofs = uniformBindingProofs(record?.bufferId);
      const destinationBindingClass = matchedSemanticBindings.length > 0
        ? 'known-attribute'
        : matchedUniformBindingProofs.length > 0
          ? 'uniform-bind-group-buffer'
          : 'unclassified-persistent-buffer';
      return {
        exact: creation !== null && destinationBindingClass !== 'unclassified-persistent-buffer',
        resourceId: record?.bufferId,
        resourceClass: creation?.resourceClass ?? null,
        createSequence: creation?.sequence ?? null,
        createCapturePhase: creation?.capturePhase ?? null,
        size: creation?.size ?? null,
        usage: creation?.usage ?? null,
        mappedAtCreation: creation?.mappedAtCreation ?? null,
        destinationBindingClass,
        semanticBindings: matchedSemanticBindings,
        uniformBindingProofs: matchedUniformBindingProofs,
      };
    })(),
  }));
  const partitions = RUNNER_QUEUE_WRITE_BUFFER_CATEGORY_ORDER.map((category) => ({
    category,
    count: records.filter((record) => record.category === category).length,
    records: records.filter((record) => record.category === category),
  }));
  const unclassified = records.filter((record) => record.category === null);
  const specializedSources = [
    ...scenarios.map((scenario) => ({
      ownerKind: 'scenario-matrix-realization',
      ownerId: scenario?.scenarioId,
      capturePhase: scenario?.matrixGpuRealization?.capturePhase,
      records: scenario?.matrixGpuRealization?.queueWriteBuffers ?? [],
    })),
    ...scenarios.map((scenario) => ({
      ownerKind: 'shared-resource-commitment',
      ownerId: scenario?.scenarioId,
      capturePhase: scenario?.gpuResourceCommitments?.capturePhase,
      records: scenario?.gpuResourceCommitments?.queueWriteBuffers ?? [],
    })),
    ...installations.map((installation) => ({
      ownerKind: 'schedule-attribute-realization',
      ownerId: installation?.installationId,
      capturePhase: `${installation?.basePhase}/attribute-realization`,
      records: installation?.queueWriteBuffers ?? [],
    })),
    ...installations.map((installation) => ({
      ownerKind: 'schedule-diagnostic-position-realization',
      ownerId: installation?.installationId,
      capturePhase: installation?.diagnosticPositionRealization?.capturePhase,
      records: installation?.diagnosticPositionRealization?.queueWriteBuffers ?? [],
    })),
  ].map((source) => {
    const sequences = source.records.map((record) => record?.sequence);
    const rawMatches = rawRecords.filter((record) => sequences.includes(record?.sequence));
    const rawExact = rawMatches.length === source.records.length
      && source.records.every((record) => rawMatches.some((raw) => (
        exactJson(runnerQueueWriteBufferProjection(raw),
          runnerQueueWriteBufferProjection(record))
      )));
    return {
      ownerKind: source.ownerKind,
      ownerId: source.ownerId,
      capturePhase: source.capturePhase,
      recordCount: source.records.length,
      sequences,
      records: source.records,
      rawExact,
    };
  });
  const specializedCategories = [
    'scenario-matrix-realization',
    'shared-resource-commitment',
    'schedule-attribute-realization',
    'schedule-diagnostic-position-realization',
  ];
  const specializedLedgerSequences = records.filter(
    (record) => specializedCategories.includes(record.category),
  ).map((record) => record.sequence).sort((left, right) => left - right);
  const specializedSourceSequences = specializedSources.flatMap(
    (source) => source.sequences,
  ).sort((left, right) => left - right);
  const specializedCounts = Object.fromEntries(specializedCategories.map(
    (category) => [category, records.filter((record) => record.category === category).length],
  ));
  const expectedSpecializedCounts = {
    'scenario-matrix-realization': 1,
    'shared-resource-commitment': 2,
    'schedule-attribute-realization': 13,
    'schedule-diagnostic-position-realization': 7,
  };
  const uniformReconstruction = reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
    records, { bufferCreations, lifecycle },
  );
  const {
    uniformWriteGrammar,
    uniformWriteGroupCount,
    uniformWriteGroupsExact,
    uniformWriteGroups,
  } = uniformReconstruction.expected;
  const uniformWriteGrammarExact = uniformReconstruction.valid;
  const retainedUniformEvidence = {
    uniformWriteGrammarExact: ledger?.uniformWriteGrammarExact,
    uniformWriteGrammar: ledger?.uniformWriteGrammar,
    uniformWriteGroupCount: ledger?.uniformWriteGroupCount,
    uniformWriteGroupsExact: ledger?.uniformWriteGroupsExact,
    uniformWriteGroups: ledger?.uniformWriteGroups,
  };
  const retainedUniformEvidenceExact = exactJson(
    retainedUniformEvidence, uniformReconstruction.expected,
  );
  const uniformOnlyCategories = new Set([
    'renderer-initialization', 'production-render', 'diagnostic-render',
  ]);
  const allowedLiveAttributeSemantics = (record) => {
    const match = record?.capturePhase?.match(/^phase0\/live\/(v99|v20)\/([AIF])$/u);
    if (match === null) return [];
    return [
      'shared.bounds', 'shared.objectBucket', 'shared.bucketBase',
      'shared.bucketCapacity', 'shared.cullOrder', 'shared.visibleIds',
      'shared.overflow', `command.${match[2]}`,
    ];
  };
  const destinationPolicyExact = records.every((record) => {
    const owner = record.destinationOwner;
    const semantics = owner?.semanticBindings?.map((binding) => binding.semantic) ?? [];
    const uniformExact = owner?.destinationBindingClass === 'uniform-bind-group-buffer'
      && owner?.uniformBindingProofs?.length > 0
      && owner.uniformBindingProofs.every((proof) => proof?.exact === true
        && proof?.bufferId === record.bufferId && proof?.bufferType === 'uniform'
        && proof?.hasDynamicOffset === false && proof?.minBindingSize === 0
        && proof?.layoutMatchCount === 1 && proof?.coordinateMatchCount === 1);
    if (uniformOnlyCategories.has(record.category)) return uniformExact;
    if (record.category === 'live-compute') {
      const allowed = allowedLiveAttributeSemantics(record);
      return uniformExact || (owner?.destinationBindingClass === 'known-attribute'
        && semantics.length > 0 && semantics.every((semantic) => allowed.includes(semantic)));
    }
    if (record.category === 'scenario-matrix-realization') {
      return owner?.destinationBindingClass === 'known-attribute'
        && semantics.includes('shared.matrix');
    }
    if (record.category === 'shared-resource-commitment') {
      return owner?.destinationBindingClass === 'known-attribute'
        && semantics.includes('shared.visibleIds');
    }
    if (record.category === 'schedule-attribute-realization') {
      return owner?.destinationBindingClass === 'known-attribute'
        && semantics.some((semantic) => semantic === 'geometry.A.bucketBase'
          || /^command\.[AIF]$/u.test(semantic));
    }
    if (record.category === 'schedule-diagnostic-position-realization') {
      return owner?.destinationBindingClass === 'known-attribute'
        && semantics.includes('diagnostic.address.featurePosition');
    }
    return false;
  });
  const computeUses = (lifecycle?.computePassTraces ?? []).flatMap((trace) => (
    (trace?.events ?? []).filter((event) => event?.method === 'setBindGroup').map(
      (event) => ({ trace, event }),
    )
  ));
  const bindGroupContainsBuffer = (bindGroupId, bufferId) => bindGroups.some(
    (group) => group?.resourceId === bindGroupId
      && group?.entries?.some((entry) => entry?.bufferId === bufferId),
  );
  const exactNullDynamicOffsets = (event) => event?.dynamicOffsets === null
    && event?.dynamicOffsetStart === null
    && event?.dynamicOffsetLength === null
    && event?.selectedDynamicOffsets === null;
  const allRawBindGroupOccurrencesForBuffer = (bufferId) => bindGroups.flatMap(
    (bindGroup) => (bindGroup?.entries ?? []).filter(
      (entry) => entry?.bufferId === bufferId,
    ).map((entry) => ({ bindGroup, entry })),
  );
  const uniformBufferBindingClosureExact = (record) => {
    const creationMatches = bufferCreations.filter(
      (candidate) => candidate?.resourceId === record?.bufferId,
    );
    const creation = creationMatches[0];
    const occurrences = allRawBindGroupOccurrencesForBuffer(record?.bufferId);
    return creationMatches.length === 1
      && creation?.usage === 72
      && creation?.mappedAtCreation === false
      && record?.destinationOwner?.destinationBindingClass === 'uniform-bind-group-buffer'
      && record.destinationOwner.semanticBindings.length === 0
      && occurrences.length === 1
      && record.destinationOwner.uniformBindingProofs.length === occurrences.length
      && occurrences.every(({ bindGroup, entry }) => {
        const layoutMatches = retainedBindGroupLayouts.filter(
          (layout) => layout?.layoutId === bindGroup?.layoutId,
        );
        const coordinateMatches = layoutMatches.flatMap((layout) => (
          (layout?.entries ?? []).filter(
            (candidate) => candidate?.binding === entry?.binding,
          )
        ));
        return layoutMatches.length === 1
          && coordinateMatches.length === 1
          && layoutMatches[0]?.sequence < bindGroup?.sequence
          && Number.isSafeInteger(entry?.bufferOffset) && entry.bufferOffset >= 0
          && entry.bufferOffset % 4 === 0
          && (entry?.bufferSize === null
            ? entry.bufferOffset < creation?.size
            : Number.isSafeInteger(entry.bufferSize) && entry.bufferSize > 0
              && entry.bufferSize % 4 === 0
              && entry.bufferOffset + entry.bufferSize <= creation?.size)
          && (coordinateMatches[0]?.buffer?.type ?? 'uniform') === 'uniform'
          && coordinateMatches[0]?.buffer?.hasDynamicOffset === false
          && coordinateMatches[0]?.buffer?.minBindingSize === 0;
      });
  };
  const uniformBindingClosureExact = records.filter(
    (record) => record.destinationOwner.destinationBindingClass
      === 'uniform-bind-group-buffer',
  ).every(uniformBufferBindingClosureExact);
  const liveAttributeExecutionExact = records.every((record) => {
    if (record.category !== 'live-compute'
      || record.destinationOwner.destinationBindingClass !== 'known-attribute') return true;
    const creation = bufferCreations.find(
      (candidate) => candidate?.resourceId === record.bufferId,
    );
    const fullWrite = record.bufferOffset === 0
      && record.selectedSourceByteLength === creation?.size
      && record.selectedSourceByteOffset === 0
      && record.dataOffset === 0
      && record.size === null
      && records.filter((candidate) => candidate.capturePhase === record.capturePhase
        && candidate.bufferId === record.bufferId).length === 1;
    if (!fullWrite) return false;
    return fullWrite && computeUses.some(({ trace, event }) => (
      trace?.capturePhase === record.capturePhase
        && event?.sequence > record.sequence
        && exactNullDynamicOffsets(event)
        && bindGroupContainsBuffer(event?.bindGroupId, record.bufferId)
    ));
  });
  const expectedLiveAttributeWrites = [
    ['v99', 'I', []],
    ['v99', 'F', ['shared.visibleIds', 'shared.overflow']],
    ['v99', 'A', ['shared.visibleIds', 'shared.overflow']],
    ['v20', 'A', [
      'shared.bounds', 'shared.objectBucket', 'shared.bucketBase',
      'shared.bucketCapacity', 'shared.cullOrder', 'shared.visibleIds',
      'shared.overflow', 'command.A',
    ]],
    ['v20', 'F', ['shared.visibleIds', 'shared.overflow', 'command.F']],
    ['v20', 'I', ['shared.visibleIds', 'shared.overflow', 'command.I']],
  ];
  const liveAttributeWriteGrammar = expectedLiveAttributeWrites.map(([
    scenarioId, lane, expectedSemantics,
  ]) => {
    const capturePhase = `phase0/live/${scenarioId}/${lane}`;
    const phaseRecords = records.filter((record) => record.capturePhase === capturePhase);
    const attributeRecords = phaseRecords.filter(
      (record) => record.destinationOwner.destinationBindingClass === 'known-attribute',
    );
    const observedSemantics = attributeRecords.flatMap(
      (record) => record.destinationOwner.semanticBindings.map((binding) => binding.semantic),
    );
    return {
      pass: exactArray([...observedSemantics].sort(), [...expectedSemantics].sort()),
      scenarioId,
      lane,
      capturePhase,
      expectedSemantics,
      observedSemantics,
      attributeWriteCount: attributeRecords.length,
      uniformWriteCount: phaseRecords.length - attributeRecords.length,
      attributeRecords,
    };
  });
  const liveAttributeWriteGrammarExact = liveAttributeWriteGrammar.every(
    (record) => record.pass,
  );
  const mergedGeometryWriteCount = records.filter(
    (record) => record.category === 'merged-geometry-realization',
  ).length;
  const rendererInitializationWriteCount = records.filter(
    (record) => record.category === 'renderer-initialization',
  ).length;
  const genericUniformWriteCount = records.filter(
    (record) => uniformOnlyCategories.has(record.category),
  ).length;
  const specializedBindingsExact = specializedSources.every((source) => source.rawExact
    && source.records.every((record) => record?.sourceSha256 === undefined
      || record.sourceSha256 === record.selectedSourceSha256))
    && exactArray(specializedLedgerSequences, specializedSourceSequences)
    && exactJson(specializedCounts, expectedSpecializedCounts);
  const bytesPerElement = {
    Int8Array: 1, Uint8Array: 1, Uint8ClampedArray: 1,
    Int16Array: 2, Uint16Array: 2,
    Int32Array: 4, Uint32Array: 4, Float32Array: 4,
    BigInt64Array: 8, BigUint64Array: 8, Float64Array: 8,
    ArrayBuffer: 1, SharedArrayBuffer: 1, DataView: 1,
  };
  const physicalWritesExact = records.every((record, index) => {
    const matches = bufferCreations.filter(
      (creation) => creation?.resourceId === record?.bufferId,
    );
    const creation = matches[0];
    const expectedElementByteSize = bytesPerElement[record?.sourceType];
    const expectedElementLength = record?.size === null
      ? record?.sourceElementCount - record?.dataOffset : record?.size;
    const effectiveSize = expectedElementLength * expectedElementByteSize;
    return Number.isSafeInteger(record?.sequence)
      && (index === 0 || record.sequence > records[index - 1].sequence)
      && typeof record?.capturePhase === 'string' && record.capturePhase.length > 0
      && record?.category !== null
      && matches.length === 1
      && creation?.resourceClass === 'persistent-or-upload-buffer'
      && creation?.sequence < record.sequence
      && (Number(creation?.usage) & 0x0008) !== 0
      && (Number(creation?.usage) & 0x0003) === 0
      && record?.destinationOwner?.exact === true
      && record.destinationOwner.resourceId === creation.resourceId
      && record.destinationOwner.resourceClass === creation.resourceClass
      && record.destinationOwner.createSequence === creation.sequence
      && record.destinationOwner.createCapturePhase === creation.capturePhase
      && record.destinationOwner.size === creation.size
      && record.destinationOwner.usage === creation.usage
      && record.destinationOwner.mappedAtCreation === creation.mappedAtCreation
      && typeof record?.sourceId === 'string' && record.sourceId.length > 0
      && typeof record?.sourceBackingBufferId === 'string'
      && record.sourceBackingBufferId.length > 0
      && Number.isSafeInteger(record?.bufferOffset) && record.bufferOffset >= 0
      && Number.isSafeInteger(record?.sourceByteLength) && record.sourceByteLength > 0
      && Number.isSafeInteger(record?.sourceByteOffset) && record.sourceByteOffset >= 0
      && Number.isSafeInteger(expectedElementByteSize)
      && record?.sourceElementByteSize === expectedElementByteSize
      && Number.isSafeInteger(record?.sourceElementCount)
      && record.sourceElementCount * expectedElementByteSize === record.sourceByteLength
      && Number.isSafeInteger(record?.dataOffset) && record.dataOffset >= 0
      && (record?.size === null || (Number.isSafeInteger(record.size) && record.size > 0))
      && record?.selectedSourceElementOffset === record.dataOffset
      && record?.selectedSourceElementLength === expectedElementLength
      && record?.selectedSourceByteOffset === record.dataOffset * expectedElementByteSize
      && record?.selectedSourceByteLength === effectiveSize
      && /^[0-9a-f]{64}$/u.test(record?.selectedSourceSha256 ?? '')
      && Number.isSafeInteger(effectiveSize) && effectiveSize > 0
      && record.dataOffset + expectedElementLength <= record.sourceElementCount
      && record.bufferOffset + effectiveSize <= creation.size
      && record.bufferOffset % 4 === 0
      && record.selectedSourceByteOffset % 4 === 0 && effectiveSize % 4 === 0;
  });
  const expectedLedger = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-queue-write-buffer-ledger',
    pass: true,
    expectedSpecializedCounts,
    specializedCounts,
    specializedBindingsExact: true,
    destinationPolicyExact: true,
    liveAttributeWriteGrammarExact: true,
    liveAttributeWriteGrammar,
    mergedGeometryWriteCount: 0,
    rendererInitializationWriteCount: 0,
    genericUniformWriteCount,
    uniformWriteGrammarExact: true,
    uniformWriteGrammar,
    uniformWriteGroupCount,
    uniformWriteGroupsExact,
    uniformWriteGroups,
    callCount: records.length,
    categoryOrder: [...RUNNER_QUEUE_WRITE_BUFFER_CATEGORY_ORDER],
    partitions,
    unclassifiedCount: 0,
    unclassified: [],
    specializedSources,
    records,
  };
  addReasonUnless(Array.isArray(rawRecords) && rawRecords.length > 0
    && new Set(records.map((record) => record.sequence)).size === records.length
    && partitions.reduce((count, partition) => count + partition.count, 0) === records.length
    && unclassified.length === 0
    && specializedSources.length === 20
    && semanticBindings.every((binding) => Number.isSafeInteger(binding.attributeId)
      && typeof binding.bufferId === 'string' && binding.bufferId.length > 0)
    && specializedBindingsExact
    && destinationPolicyExact
    && liveAttributeWriteGrammarExact
    && mergedGeometryWriteCount === 0
    && rendererInitializationWriteCount === 0
    && uniformWriteGrammarExact
    && retainedUniformEvidenceExact
    && uniformBindingClosureExact
    && liveAttributeExecutionExact
    && physicalWritesExact
    && exactJson(ledger, expectedLedger),
  reasons, 'queue.writeBuffer ledger is not the exact independent raw partition');
  return { valid: reasons.length === 0, reasons, expected: expectedLedger };
}

export function validateThreeImmediatePhase0BufferMapLifecycle(
  retained,
  {
    lifecycle = {}, resources = {}, stagingTransferLedger = [],
  } = {},
) {
  const reasons = [];
  const buffers = resources?.gpuCreations?.filter(
    (record) => record?.method === 'createBuffer',
  ) ?? [];
  const rawMapEvents = lifecycle?.bufferMapEvents ?? [];
  const useEvents = [];
  for (const write of lifecycle?.queueWriteBufferCalls ?? []) {
    useEvents.push({ sequence: write?.sequence, kind: 'queue.writeBuffer',
      bufferId: write?.bufferId });
  }
  for (const resource of resources?.gpuCreations?.filter(
    (record) => record?.method === 'createBindGroup',
  ) ?? []) {
    for (const entry of resource?.entries ?? []) {
      if (typeof entry?.bufferId === 'string') {
        useEvents.push({ sequence: resource.sequence, kind: 'createBindGroup-entry',
          bufferId: entry.bufferId });
      }
    }
  }
  for (const [traceKind, traces] of [
    ['render-bundle', lifecycle?.renderBundleTraces ?? []],
    ['render-pass', lifecycle?.renderPassTraces ?? []],
    ['compute-pass', lifecycle?.computePassTraces ?? []],
  ]) {
    for (const trace of traces) {
      for (const event of trace?.events ?? []) {
        if (typeof event?.bufferId === 'string') {
          useEvents.push({ sequence: event.sequence, kind: `${traceKind}/${event.method}`,
            bufferId: event.bufferId });
        }
      }
    }
  }
  for (const transfer of lifecycle?.commandEncoderTransferCalls ?? []) {
    const ids = [
      transfer?.sourceBufferId, transfer?.destinationBufferId,
      transfer?.source?.bufferId, transfer?.destination?.bufferId,
    ].filter((value) => typeof value === 'string');
    for (const bufferId of new Set(ids)) {
      useEvents.push({ sequence: transfer.sequence, kind: transfer.method, bufferId });
    }
  }
  const stagingById = new Map(stagingTransferLedger.map(
    (record) => [record?.stagingBufferId, record],
  ));
  const records = buffers.map((buffer) => {
    const mapEvents = rawMapEvents.filter(
      (event) => event?.resourceId === buffer?.resourceId,
    );
    const firstUse = useEvents.filter((event) => event.bufferId === buffer?.resourceId)
      .sort((left, right) => left.sequence - right.sequence)[0] ?? null;
    const staging = stagingById.get(buffer?.resourceId) ?? null;
    const persistent = buffer?.resourceClass === 'persistent-or-upload-buffer';
    const persistentMappedGrammar = persistent && buffer?.mappedAtCreation === true
      && mapEvents.length === 2
      && mapEvents[0]?.method === 'getMappedRange'
      && mapEvents[0]?.offset === 0
      && (mapEvents[0]?.size === null || mapEvents[0]?.size === buffer?.size)
      && mapEvents[1]?.method === 'unmap'
      && mapEvents.every((event) => event?.capturePhase === buffer?.capturePhase)
      && buffer.sequence < mapEvents[0].sequence
      && mapEvents[0].sequence < mapEvents[1].sequence
      && (firstUse === null || mapEvents[1].sequence < firstUse.sequence);
    const persistentUnmappedGrammar = persistent && buffer?.mappedAtCreation === false
      && mapEvents.length === 0;
    const stagingGrammar = buffer?.resourceClass === 'readback-staging'
      && staging?.pass === true
      && buffer?.mappedAtCreation === false
      && mapEvents.length === 2
      && mapEvents[0]?.method === 'mapAsync' && mapEvents[0]?.mode === 1
      && mapEvents[0]?.offset === 0
      && mapEvents[1]?.method === 'getMappedRange' && mapEvents[1]?.offset === 0
      && mapEvents[0]?.sequence === staging?.mapEvents?.[0]?.sequence
      && mapEvents[1]?.sequence === staging?.mapEvents?.[1]?.sequence
      && staging?.transfer?.sequence < mapEvents[0].sequence
      && (firstUse === null || firstUse.sequence === staging?.transfer?.sequence);
    return {
      pass: persistent
        ? (Number(buffer?.usage) & 0x0003) === 0
          && (persistentMappedGrammar || persistentUnmappedGrammar)
        : stagingGrammar,
      resourceId: buffer?.resourceId,
      resourceClass: buffer?.resourceClass,
      capturePhase: buffer?.capturePhase,
      createSequence: buffer?.sequence,
      size: buffer?.size,
      usage: buffer?.usage,
      mappedAtCreation: buffer?.mappedAtCreation,
      mapEventCount: mapEvents.length,
      mapEvents,
      firstUse,
      persistentMappedGrammar,
      persistentUnmappedGrammar,
      stagingGrammar,
    };
  });
  const representedMapSequences = records.flatMap(
    (record) => record.mapEvents.map((event) => event.sequence),
  ).sort((left, right) => left - right);
  const rawMapSequences = rawMapEvents.map((event) => event?.sequence)
    .sort((left, right) => left - right);
  const unmatchedEvents = rawMapEvents.filter((event) => (
    !buffers.some((buffer) => buffer?.resourceId === event?.resourceId)
  ));
  const persistentRecords = records.filter(
    (record) => record.resourceClass === 'persistent-or-upload-buffer',
  );
  const stagingRecords = records.filter(
    (record) => record.resourceClass === 'readback-staging',
  );
  const expected = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-buffer-map-lifecycle',
    pass: true,
    bufferCount: buffers.length,
    persistentBufferCount: persistentRecords.length,
    persistentMappedAtCreationCount: persistentRecords.filter(
      (record) => record.mappedAtCreation,
    ).length,
    stagingBufferCount: stagingRecords.length,
    rawMapEventCount: rawMapEvents.length,
    representedMapEventCount: representedMapSequences.length,
    unmatchedEventCount: 0,
    unmatchedEvents: [],
    records,
  };
  addReasonUnless(buffers.length > 825
    && new Set(buffers.map((buffer) => buffer?.resourceId)).size === buffers.length
    && records.every((record) => record.pass)
    && stagingRecords.length === 825 && persistentRecords.length > 0
    && unmatchedEvents.length === 0
    && new Set(rawMapSequences).size === rawMapSequences.length
    && exactArray(representedMapSequences, rawMapSequences)
    && exactJson(retained, expected),
  reasons, 'buffer map lifecycle is not the exact independent creation/use partition');
  return { valid: reasons.length === 0, reasons, evidence: expected };
}

export function validateThreeImmediatePhase0EncoderState(
  retained,
  { renderBundles = [], renderPasses = [], computePasses = [] } = {},
) {
  const reasons = [];
  const renderBundleBindGroups = renderBundles.flatMap(
    (trace) => (trace?.events ?? []).filter((event) => event?.method === 'setBindGroup'),
  );
  const renderPassBindGroups = renderPasses.flatMap(
    (trace) => (trace?.events ?? []).filter((event) => event?.method === 'setBindGroup'),
  );
  const computePassBindGroups = computePasses.flatMap(
    (trace) => (trace?.events ?? []).filter((event) => event?.method === 'setBindGroup'),
  );
  const allBindGroups = [
    ...renderBundleBindGroups, ...renderPassBindGroups, ...computePassBindGroups,
  ];
  const dynamicOffsetCalls = allBindGroups.filter((event) => (
    event?.dynamicOffsets !== null
      || event?.dynamicOffsetStart !== null
      || event?.dynamicOffsetLength !== null
      || event?.selectedDynamicOffsets !== null
  ));
  const renderBundleImmediateTraces = renderBundles.map((trace) => {
    const calls = (trace?.events ?? []).filter((event) => event?.method === 'setImmediates');
    return {
      encoderId: trace?.encoderId,
      capturePhase: trace?.capturePhase,
      bundleId: trace?.bundleId,
      callCount: calls.length,
      calls,
    };
  });
  const renderPassImmediateCalls = renderPasses.flatMap(
    (trace) => (trace?.events ?? []).filter((event) => event?.method === 'setImmediates'),
  );
  const computePassImmediateCalls = computePasses.flatMap(
    (trace) => (trace?.events ?? []).filter((event) => event?.method === 'setImmediates'),
  );
  const occlusionQueryCalls = renderPasses.flatMap(
    (trace) => (trace?.events ?? []).filter((event) => (
      event?.method === 'beginOcclusionQuery' || event?.method === 'endOcclusionQuery'
    )),
  );
  const immediateBundleCount = renderBundleImmediateTraces.filter(
    (record) => record.callCount > 0,
  ).length;
  const immediateCallCount = renderBundleImmediateTraces.reduce(
    (count, record) => count + record.callCount, 0,
  );
  const expected = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-encoder-state-exclusion-evidence',
    pass: true,
    expectedRenderBundleEncoderCount: 27,
    renderBundleEncoderCount: renderBundles.length,
    expectedImmediateRenderBundleCount: 21,
    immediateRenderBundleCount: immediateBundleCount,
    expectedRenderBundleSetImmediatesCount: 672,
    renderBundleSetImmediatesCount: immediateCallCount,
    renderBundleImmediateTraces,
    renderPassSetImmediatesCount: renderPassImmediateCalls.length,
    renderPassSetImmediatesCalls: renderPassImmediateCalls,
    computePassSetImmediatesCount: computePassImmediateCalls.length,
    computePassSetImmediatesCalls: computePassImmediateCalls,
    setBindGroupCount: allBindGroups.length,
    dynamicOffsetCallCount: dynamicOffsetCalls.length,
    dynamicOffsetCalls,
    occlusionQueryCallCount: occlusionQueryCalls.length,
    occlusionQueryCalls,
  };
  addReasonUnless(renderBundles.length === 27
    && immediateBundleCount === 21 && immediateCallCount === 672
    && renderBundleImmediateTraces.every((record) => record.callCount === 0
      || (record.callCount === 32
        && record.calls.every((call) => call?.dataSize === 1)))
    && renderPassImmediateCalls.length === 0
    && computePassImmediateCalls.length === 0
    && dynamicOffsetCalls.length === 0
    && occlusionQueryCalls.length === 0
    && exactJson(retained, expected),
  reasons, 'global encoder immediate/dynamic-offset/query state is not exact');
  return { valid: reasons.length === 0, reasons };
}

function validateGpuResourceCommitments(evidence, reasons) {
  const expectedSemantics = [
    'matrix', 'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity',
    'cullOrder', 'visibleIds', 'overflow',
  ];
  const scenarioRecords = {};
  for (const scenarioId of VISIBILITY_IDS) {
    const scenario = evidence?.scenarios?.find(
      (candidate) => candidate?.scenarioId === scenarioId,
    );
    const commitment = scenario?.gpuResourceCommitments;
    const expectedResources = cpuOracles()[scenarioId].sharedResources;
    const capturePhase = `phase0/resources/${scenarioId}/shared-commitment`;
    const markers = evidence?.lifecycle?.instrumentationMarkers ?? [];
    const firstChallengeStart = scenario?.snapshots?.[0]?.productionChallenge?.startSequence;
    addReasonUnless(commitment?.schemaVersion === 1
      && commitment?.kind === 'immediate-aif-phase0-shared-gpu-resource-commitments'
      && commitment?.pass === true && commitment?.scenarioId === scenarioId
      && commitment?.capturePhase === capturePhase
      && exactArray(commitment?.semantics, expectedSemantics)
      && Array.isArray(commitment?.records)
      && exactArray(commitment.records.map((record) => record?.semantic), expectedSemantics)
      && commitment?.managerVersionsExact === true
      && commitment?.queueWriteBufferCount === 1
      && Number.isSafeInteger(commitment?.startSequence)
      && Number.isSafeInteger(commitment?.queueCompleteSequence)
      && Number.isSafeInteger(commitment?.completeSequence)
      && commitment.startSequence < commitment.queueCompleteSequence
      && commitment.queueCompleteSequence < commitment.completeSequence
      && commitment.completeSequence < firstChallengeStart
      && markerExact(markers, commitment.startSequence,
        'phase0-shared-gpu-commitment-start', { scenarioId, capturePhase })
      && markerExact(markers, commitment.queueCompleteSequence,
        'phase0-shared-attribute-updates-queue-complete', { scenarioId })
      && markerExact(markers, commitment.completeSequence,
        'phase0-shared-gpu-commitment-complete', {
          scenarioId, queueCompleteSequence: commitment.queueCompleteSequence,
        })
      && Array.isArray(commitment?.managerCalls)
      && commitment.managerCalls.length === 8,
    reasons, `${scenarioId} shared GPU resource commitment envelope is invalid`);
    scenarioRecords[scenarioId] = Object.fromEntries(
      (commitment?.records ?? []).map((record) => [record?.semantic, record]),
    );
    for (const semantic of expectedSemantics) {
      const expected = expectedResources[semantic];
      const record = scenarioRecords[scenarioId][semantic];
      const managerCall = commitment?.managerCalls?.find(
        (candidate) => candidate?.semantic === semantic,
      );
      const state = scenario?.resources?.postflight?.sharedAttributes?.[semantic];
      const creationMatches = evidence?.resources?.gpuCreations?.filter(
        (creation) => creation?.method === 'createBuffer'
          && creation?.resourceId === record?.gpuBufferId,
      ) ?? [];
      addReasonUnless(record?.semantic === semantic && record?.exact === true
        && Number.isSafeInteger(record?.attributeId) && record.attributeId >= 0
        && Number.isSafeInteger(record?.attributeVersion) && record.attributeVersion >= 0
        && typeof record?.gpuBufferId === 'string' && record.gpuBufferId.length > 0
        && record?.arrayType === expected?.arrayType
        && record?.byteLength === expected?.byteLength
        && record?.oracleKind === expected?.oracleKind
        && record?.cpuSha256 === expected?.sha256
        && record?.oracleSha256 === expected?.sha256
        && record?.cpuSourceSha256 === expected?.cpuSourceSha256
        && record?.cpuSourceMatchesOracle === expected?.cpuSourceMatchesOracle
        && record?.gpuSha256 === expected?.sha256
        && record?.rawWitnessId
          === `sha256-${expected?.sha256}-${expected?.byteLength}`
        && Number.isSafeInteger(record?.readbackStartSequence)
        && Number.isSafeInteger(record?.readbackCompleteSequence)
        && record.readbackStartSequence > commitment?.queueCompleteSequence
        && record.readbackCompleteSequence > record.readbackStartSequence
        && record.readbackCompleteSequence < commitment?.completeSequence
        && markerExact(markers, record.readbackStartSequence,
          'phase0-shared-gpu-readback-start', {
            scenarioId, semantic, attributeId: record.attributeId,
          })
        && markerExact(markers, record.readbackCompleteSequence,
          'phase0-shared-gpu-readback-complete', {
            scenarioId, semantic, attributeId: record.attributeId,
          })
        && managerCall?.manager === 'renderer._attributes'
        && managerCall?.managerConstructor === 'Attributes'
        && managerCall?.attributeId === record.attributeId
        && managerCall?.attributeVersion === record.attributeVersion
        && managerCall?.attributeType === 3
        && managerCall?.gpuBufferId === record.gpuBufferId
        && managerCall?.gpuBufferSize === expected?.byteLength
        && Number.isSafeInteger(managerCall?.startSequence)
        && Number.isSafeInteger(managerCall?.completeSequence)
        && managerCall.startSequence < managerCall.completeSequence
        && managerCall.completeSequence < commitment?.queueCompleteSequence
        && managerCall?.afterDataVersion === record.attributeVersion
        && managerCall?.beforeDataVersion === (semantic === 'visibleIds'
          ? record.attributeVersion - 1 : record.attributeVersion)
        && markerExact(markers, managerCall.startSequence,
          'phase0-shared-attribute-update-start', {
            scenarioId, semantic, attributeId: record.attributeId,
            attributeVersion: record.attributeVersion, attributeType: 3,
            beforeDataVersion: managerCall.beforeDataVersion,
          })
        && markerExact(markers, managerCall.completeSequence,
          'phase0-shared-attribute-update-complete', {
            scenarioId, semantic, attributeId: record.attributeId,
            attributeVersion: record.attributeVersion, attributeType: 3,
            afterDataVersion: managerCall.afterDataVersion,
            gpuBufferId: record.gpuBufferId,
          })
        && state?.id === record?.attributeId && state?.byteLength === expected?.byteLength
        && state?.sha256 === expected?.cpuSourceSha256
        && creationMatches.length === 1
        && creationMatches[0]?.size === expected?.byteLength
        && creationMatches[0]?.usage === 172
        && creationMatches[0]?.resourceClass === 'persistent-or-upload-buffer',
      reasons, `${scenarioId}/${semantic} CPU-to-GPU bytes/identity are not exact`);
    }
    const commitmentWrites = (evidence?.lifecycle?.queueWriteBufferCalls ?? []).filter(
      (write) => write?.capturePhase === capturePhase
        && write?.sequence > commitment?.startSequence
        && write?.sequence < commitment?.queueCompleteSequence,
    );
    const expectedWriteSemantics = ['visibleIds'];
    addReasonUnless(commitmentWrites.length === expectedWriteSemantics.length
      && expectedWriteSemantics.every((semantic, index) => {
        const record = scenarioRecords[scenarioId][semantic];
        const manager = commitment?.managerCalls?.find(
          (candidate) => candidate?.semantic === semantic,
        );
        const write = commitmentWrites[index];
        return write?.bufferId === record?.gpuBufferId
          && write?.bufferOffset === 0
          && write?.sourceType === record?.arrayType
          && write?.sourceByteLength === record?.byteLength
          && write?.dataOffset === 0 && write?.size === null
          && write?.sequence > manager?.startSequence
          && write?.sequence < manager?.completeSequence;
      })
      && exactJson(commitment?.queueWriteBuffers, commitmentWrites)
      && exactJson(commitment?.visibleIdsUpload, commitmentWrites[0])
      && commitment?.visibleIdsUploadExact === true,
    reasons, `${scenarioId} dirty shared resources were not exactly uploaded before commitment`);
    const frozen = scenario?.freeze;
    const visible = scenarioRecords[scenarioId].visibleIds;
    addReasonUnless(visible?.attributeId === frozen?.visibleIdsAttributeId
      && visible?.gpuBufferId === frozen?.visibleIdsGpuBufferId
      && visible?.attributeVersion === frozen?.visibleIdsAttributeVersion
      && visible?.gpuSha256 === frozen?.visibleIdsSha256
      && frozen?.gpuResourceCommitmentCapturePhase === capturePhase
      && frozen?.gpuResourceCommitmentCompleteSequence === commitment?.completeSequence,
    reasons, `${scenarioId} frozen visible-ID GPU commitment is not the render resource`);

    const matrix = scenarioRecords[scenarioId].matrix;
    const realization = scenario?.matrixGpuRealization;
    const realizationPhase = `phase0/scenario-load/${scenarioId}/matrix-realization`;
    const realizationWrites = (evidence?.lifecycle?.queueWriteBufferCalls ?? []).filter(
      (record) => record?.capturePhase === realizationPhase
        && record?.sequence > realization?.startSequence
        && record?.sequence < realization?.updateCompleteSequence,
    );
    const scenarioComputeTraces = (evidence?.lifecycle?.computePassTraces ?? []).filter(
      (trace) => trace?.capturePhase?.startsWith(`phase0/live/${scenarioId}/`),
    );
    const expectedRealizationWriteCount = scenarioId === 'v99' ? 0 : 1;
    const realizationCreations = (evidence?.resources?.gpuCreations ?? []).filter(
      (record) => record?.capturePhase === realizationPhase
        && record?.sequence > realization?.startSequence
        && record?.sequence < realization?.updateCompleteSequence,
    );
    const realizationMaps = (evidence?.lifecycle?.bufferMapEvents ?? []).filter(
      (record) => record?.capturePhase === realizationPhase
        && record?.sequence > realization?.startSequence
        && record?.sequence < realization?.updateCompleteSequence,
    );
    const realizationWriteExact = scenarioId === 'v99'
      ? realizationWrites.length === 0
      : realizationWrites.length === 1
        && realizationWrites[0]?.bufferId === matrix?.gpuBufferId
        && realizationWrites[0]?.bufferOffset === 0
        && realizationWrites[0]?.sourceType === 'Float32Array'
        && realizationWrites[0]?.sourceByteLength === matrix?.byteLength
        && realizationWrites[0]?.dataOffset === 0
        && realizationWrites[0]?.size === null;
    addReasonUnless(realization?.schemaVersion === 1
      && realization?.kind === 'immediate-aif-phase0-scenario-matrix-realization'
      && realization?.pass === true && realization?.scenarioId === scenarioId
      && realization?.capturePhase === realizationPhase
      && realization?.attributeId === matrix?.attributeId
      && realization?.attributeVersion === matrix?.attributeVersion
      && realization?.attributeType === 3
      && realization?.beforeDataVersion === (scenarioId === 'v99'
        ? null : realization.attributeVersion - 1)
      && realization?.afterDataVersion === realization?.attributeVersion
      && realization?.gpuBufferId === matrix?.gpuBufferId
      && realization?.gpuBufferSize === matrix?.byteLength
      && realization?.cpuSha256 === expectedResources.matrix.sha256
      && Number.isSafeInteger(realization?.startSequence)
      && Number.isSafeInteger(realization?.updateCompleteSequence)
      && Number.isSafeInteger(realization?.queueCompleteSequence)
      && realization.startSequence < realization.updateCompleteSequence
      && realization.updateCompleteSequence < realization.queueCompleteSequence
      && realization.queueCompleteSequence < commitment.startSequence
      && scenarioComputeTraces.length === 3
      && scenarioComputeTraces.every(
        (trace) => trace?.sequence > realization.queueCompleteSequence
          && trace?.events?.at(-1)?.sequence < commitment.startSequence,
      )
      && markerExact(markers, realization.startSequence,
        'phase0-scenario-matrix-realization-start', {
          scenarioId, attributeId: realization.attributeId,
          attributeVersion: realization.attributeVersion,
        })
      && markerExact(markers, realization.updateCompleteSequence,
        'phase0-scenario-matrix-update-complete', {
          scenarioId, attributeId: realization.attributeId,
          attributeVersion: realization.attributeVersion,
          beforeDataVersion: realization.beforeDataVersion,
          afterDataVersion: realization.afterDataVersion,
          gpuBufferId: realization.gpuBufferId,
        })
      && markerExact(markers, realization.queueCompleteSequence,
        'phase0-scenario-matrix-queue-complete', {
          scenarioId, attributeId: realization.attributeId,
        })
      && realization?.expectedQueueWriteBufferCount === expectedRealizationWriteCount
      && realization?.queueWriteBufferCount === expectedRealizationWriteCount
      && exactJson(realization?.queueWriteBuffers, realizationWrites)
      && realization?.queueWriteExact === true && realizationWriteExact
      && (scenarioId === 'v99'
        ? realizationCreations.length === 1
          && realizationCreations[0]?.method === 'createBuffer'
          && realizationCreations[0]?.resourceClass === 'persistent-or-upload-buffer'
          && realizationCreations[0]?.resourceId === matrix?.gpuBufferId
          && realizationCreations[0]?.size === matrix?.byteLength
          && realizationCreations[0]?.usage === 172
          && realizationCreations[0]?.mappedAtCreation === true
          && realizationMaps.length === 2
          && exactArray(realizationMaps.map((record) => record?.method),
            ['getMappedRange', 'unmap'])
          && realizationMaps.every((record) => record?.resourceId === matrix?.gpuBufferId)
        : realizationCreations.length === 0 && realizationMaps.length === 0),
    reasons, `${scenarioId} scenario matrix GPU realization is not exact`);
  }
  for (const semantic of expectedSemantics) {
    addReasonUnless(scenarioRecords.v99?.[semantic]?.attributeId
        === scenarioRecords.v20?.[semantic]?.attributeId
      && scenarioRecords.v99?.[semantic]?.gpuBufferId
        === scenarioRecords.v20?.[semantic]?.gpuBufferId,
    reasons, `${semantic} GPU resource identity changed between visibility scenarios`);
  }
  const globalCommandIds = {};
  for (const scenario of evidence?.scenarios ?? []) {
    const scenarioId = scenario?.scenarioId;
    const shared = scenarioRecords[scenarioId];
    for (const callback of scenarioCallbacks(scenario)) {
      const lane = callback?.lane;
      globalCommandIds[lane] ??= callback?.command?.gpuBufferId;
      addReasonUnless(callback?.command?.gpuBufferId === globalCommandIds[lane]
        && callback?.address?.visibleIdsGpuBufferId === shared?.visibleIds?.gpuBufferId
        && callback?.address?.producer?.visibleIdsGpuBufferId === shared?.visibleIds?.gpuBufferId
        && callback?.output?.objectId?.producer?.visibleIdsGpuBufferId
          === shared?.visibleIds?.gpuBufferId
        && callback?.output?.objectId?.producer?.boundStorage?.find(
          (binding) => binding?.semantic === 'matrix',
        )?.observedGpuBufferId === shared?.matrix?.gpuBufferId,
      reasons, `${scenarioId}/${lane} callback is not bound to the committed GPU resources`);
    }
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      for (const phase of ['reset', 'cull']) {
        const bound = evidence?.shaders?.computeExecution
          ?.scenarios?.[scenarioId]?.lanes?.[lane]?.phases?.[phase];
        for (const binding of bound?.boundBindings ?? []) {
          if (binding?.kind !== 'storage-buffer') continue;
          const expectedGpuBufferId = binding.semantic === 'indirectCommands'
            ? globalCommandIds[lane] : shared?.[binding.semantic]?.gpuBufferId;
          addReasonUnless(binding?.observedGpuBufferId === expectedGpuBufferId
            && binding?.expectedGpuBufferId === expectedGpuBufferId,
          reasons, `${scenarioId}/${lane}/${phase}/${binding?.semantic} is not the committed GPU buffer`);
        }
      }
    }
  }
  addReasonUnless(new Set(Object.values(globalCommandIds)).size === 3
    && exactJson(evidence?.commands?.laneCommandAttributeIds,
      Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
        lane,
        evidence?.scenarios?.[0]?.resources?.postflight?.commandAttributes?.[lane]?.id,
      ]))),
  reasons, 'three private command buffer/attribute identities are not exact');
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const creations = evidence?.resources?.gpuCreations?.filter(
      (record) => record?.method === 'createBuffer'
        && record?.resourceId === globalCommandIds[lane],
    ) ?? [];
    addReasonUnless(creations.length === 1
      && creations[0]?.size === 640
      && creations[0]?.usage === 396
      && creations[0]?.resourceClass === 'persistent-or-upload-buffer',
    reasons, `${lane} indirect/storage command GPU buffer descriptor is invalid`);
  }
}

const PHASE0_SHARED_RESOURCE_SEMANTICS = Object.freeze([
  'matrix', 'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity', 'cullOrder',
  'visibleIds', 'overflow',
]);

function immediateSourceFromCallback(callback) {
  const calls = callback?.bundle?.rawEvents?.filter(
    (event) => event?.method === 'setImmediates',
  ) ?? [];
  if (calls.length !== 32) return null;
  const sourceIds = [...new Set(calls.map((event) => event?.sourceId))];
  const backingBufferIds = [...new Set(calls.map((event) => event?.backingBufferId))];
  return sourceIds.length === 1 && backingBufferIds.length === 1 ? {
    sourceId: sourceIds[0], backingBufferId: backingBufferIds[0],
  } : null;
}

function validateScenarioLoadBoundary(scenarios, reasons) {
  const v99 = scenarios?.[0];
  const v20 = scenarios?.[1];
  const boundary = v20?.scenarioLoadBoundary;
  addReasonUnless(v99?.scenarioLoadBoundary === null,
    reasons, 'v99 has an unexpected scenario-load boundary');
  if (boundary === null || typeof boundary !== 'object') {
    reasons.push('v99-to-v20 scenario-load boundary is absent');
    return;
  }
  const before = boundary.before;
  const after = boundary.after;
  const v99Postflight = v99?.resources?.postflight;
  const v20Preflight = v20?.resources?.preflight;
  const expectedSharedDeltas = Object.fromEntries(
    PHASE0_SHARED_RESOURCE_SEMANTICS.map((semantic) => [semantic, 1]),
  );
  const expectedCommandDeltas = Object.fromEntries(
    IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [lane, 1]),
  );
  const observed = {
    aBucketBaseVersion: after?.aBucketBaseVersion - before?.aBucketBaseVersion,
    sharedAttributeVersionDeltas: Object.fromEntries(
      PHASE0_SHARED_RESOURCE_SEMANTICS.map((semantic) => [semantic,
        after?.sharedAttributes?.[semantic]?.version
          - before?.sharedAttributes?.[semantic]?.version,
      ]),
    ),
    commandAttributeVersionDeltas: Object.fromEntries(
      IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [lane,
        after?.commandAttributes?.[lane]?.version
          - before?.commandAttributes?.[lane]?.version,
      ]),
    ),
    iRootVersion: after?.laneRoots?.I?.version - before?.laneRoots?.I?.version,
    iBundleRecordCount:
      after?.laneRoots?.I?.bundleRecordCount - before?.laneRoots?.I?.bundleRecordCount,
    immediateSourceIdentityStable:
      after?.immediateSource?.sourceId === before?.immediateSource?.sourceId
        && after?.immediateSource?.backingBufferId
          === before?.immediateSource?.backingBufferId,
  };
  addReasonUnless(boundary?.schemaVersion === 1
    && boundary?.kind === 'immediate-aif-phase0-v99-to-v20-load-boundary'
    && boundary?.pass === true
    && boundary?.fromScenarioId === 'v99' && boundary?.toScenarioId === 'v20'
    && exactJson(boundary?.observed, observed)
    && exactJson(observed, {
      aBucketBaseVersion: 1,
      sharedAttributeVersionDeltas: expectedSharedDeltas,
      commandAttributeVersionDeltas: expectedCommandDeltas,
      iRootVersion: 0,
      iBundleRecordCount: 0,
      immediateSourceIdentityStable: true,
    }),
  reasons, 'v99-to-v20 load delta is not independently exact');

  const baseStateKeys = [
    'label', 'rendererMemory', 'instrumentedGpuResourceCount', 'rootUuid',
    'aBucketBaseVersion', 'laneRoots', 'geometries', 'materials',
    'commandAttributes', 'sharedAttributes', 'activeScenarioId', 'activeLane',
    'immediateSource',
  ];
  const exactLaneMap = (value) => exactArray(Object.keys(value ?? {}),
    IMMEDIATE_AIF_PHASE0_LANES);
  const exactSharedMap = (value) => exactArray(Object.keys(value ?? {}),
    PHASE0_SHARED_RESOURCE_SEMANTICS);
  const structurallyExact = (state) => state !== null && typeof state === 'object'
    && exactArray(Object.keys(state), baseStateKeys)
    && typeof state.rootUuid === 'string' && state.rootUuid.length > 0
    && Number.isSafeInteger(state.aBucketBaseVersion)
    && Number.isSafeInteger(state.instrumentedGpuResourceCount)
    && exactLaneMap(state.laneRoots) && exactLaneMap(state.geometries)
    && exactLaneMap(state.materials) && exactLaneMap(state.commandAttributes)
    && exactSharedMap(state.sharedAttributes)
    && IMMEDIATE_AIF_PHASE0_LANES.every((lane) => (
      typeof state.laneRoots[lane]?.uuid === 'string'
        && Number.isSafeInteger(state.laneRoots[lane]?.version)
        && Number.isSafeInteger(state.laneRoots[lane]?.bundleRecordCount)
        && typeof state.geometries[lane] === 'string'
        && typeof state.materials[lane] === 'string'
        && Number.isSafeInteger(state.commandAttributes[lane]?.id)
        && Number.isSafeInteger(state.commandAttributes[lane]?.version)
        && SHA256_PATTERN.test(state.commandAttributes[lane]?.sha256 ?? '')
    ))
    && PHASE0_SHARED_RESOURCE_SEMANTICS.every((semantic) => (
      Number.isSafeInteger(state.sharedAttributes[semantic]?.id)
        && Number.isSafeInteger(state.sharedAttributes[semantic]?.version)
        && Number.isSafeInteger(state.sharedAttributes[semantic]?.byteLength)
        && SHA256_PATTERN.test(state.sharedAttributes[semantic]?.sha256 ?? '')
    ));
  addReasonUnless(structurallyExact(before) && structurallyExact(after)
    && before.label === 'v99-to-v20/before-load'
    && after.label === 'v99-to-v20/after-load'
    && before.activeScenarioId === 'v99' && before.activeLane === 'F'
    && after.activeScenarioId === 'v20' && after.activeLane === null,
  reasons, 'scenario-load boundary state envelopes are malformed');

  const persistentProjection = (state) => ({
    rendererMemory: state?.rendererMemory,
    rootUuid: state?.rootUuid,
    aBucketBaseVersion: state?.aBucketBaseVersion,
    laneRoots: state?.laneRoots,
    geometries: state?.geometries,
    materials: state?.materials,
    commandAttributes: state?.commandAttributes,
    sharedAttributes: state?.sharedAttributes,
  });
  addReasonUnless(exactJson(persistentProjection(before),
    persistentProjection(v99Postflight))
    && before?.instrumentedGpuResourceCount
      === v99Postflight?.instrumentedGpuResourceCount + 8
    && after?.instrumentedGpuResourceCount === before?.instrumentedGpuResourceCount
    && exactJson(after?.rendererMemory, before?.rendererMemory)
    && after?.rootUuid === before?.rootUuid
    && exactJson(after?.geometries, before?.geometries)
    && exactJson(after?.materials, before?.materials),
  reasons, 'load boundary is not joined to v99 postflight persistent resources');

  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const initial = cpuOracles().v20.schedules.canonical[lane];
    addReasonUnless(after?.laneRoots?.[lane]?.uuid === before?.laneRoots?.[lane]?.uuid
      && after?.laneRoots?.[lane]?.version === before?.laneRoots?.[lane]?.version
      && after?.laneRoots?.[lane]?.bundleRecordCount
        === before?.laneRoots?.[lane]?.bundleRecordCount
      && after?.commandAttributes?.[lane]?.id === before?.commandAttributes?.[lane]?.id
      && after?.commandAttributes?.[lane]?.version
        === before?.commandAttributes?.[lane]?.version + 1
      && after?.commandAttributes?.[lane]?.sha256 === initial.initialCommandSha256,
    reasons, `v20 load did not exactly replace ${lane} canonical command staging bytes`);
  }
  const sentinelVisibleSha256 = sha256U32(
    new Array(IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount).fill(IMMEDIATE_AIF_UNUSED_ADDRESS),
  );
  for (const semantic of PHASE0_SHARED_RESOURCE_SEMANTICS) {
    const expected = cpuOracles().v20.sharedResources[semantic];
    const expectedAfterSha = semantic === 'visibleIds'
      ? sentinelVisibleSha256 : expected.cpuSourceSha256;
    addReasonUnless(after?.sharedAttributes?.[semantic]?.id
        === before?.sharedAttributes?.[semantic]?.id
      && after?.sharedAttributes?.[semantic]?.version
        === before?.sharedAttributes?.[semantic]?.version + 1
      && after?.sharedAttributes?.[semantic]?.byteLength === expected.byteLength
      && after?.sharedAttributes?.[semantic]?.sha256 === expectedAfterSha,
    reasons, `v20 load did not exactly replace shared ${semantic} staging bytes`);
  }

  const sourceEvidence = (state, scenarioId) => {
    const source = state?.immediateSource;
    const expected = cpuOracles()[scenarioId].schedules.canonical.I;
    return source?.arrayType === 'Uint32Array'
      && source?.length === 32 && source?.byteLength === 128
      && typeof source?.sourceId === 'string' && source.sourceId.length > 0
      && typeof source?.backingBufferId === 'string' && source.backingBufferId.length > 0
      && exactArray(source?.values, expected.sourceBaseByDraw)
      && source?.sha256 === expected.sourceBaseSha256;
  };
  const v99FinalI = immediateSourceFromCallback(
    v99?.snapshots?.[4]?.laneCaptures?.find((callback) => callback?.lane === 'I'),
  );
  const v20FirstI = immediateSourceFromCallback(
    v20?.snapshots?.[0]?.laneOrders?.results?.[0]?.records
      ?.find((record) => record?.lane === 'I')?.callback,
  );
  addReasonUnless(sourceEvidence(before, 'v99') && sourceEvidence(after, 'v20')
    && observed.immediateSourceIdentityStable
    && before?.immediateSource?.sourceId === v99FinalI?.sourceId
    && before?.immediateSource?.backingBufferId === v99FinalI?.backingBufferId
    && after?.immediateSource?.sourceId === v20FirstI?.sourceId
    && after?.immediateSource?.backingBufferId === v20FirstI?.backingBufferId,
  reasons, 'v99-to-v20 immediate source identity/content chain is invalid');

  const preflightFromAfterDeltas = {
    matrix: 0, bounds: 0, objectBucket: 0, bucketBase: 0,
    bucketCapacity: 0, cullOrder: 0, visibleIds: 4, overflow: 3,
  };
  addReasonUnless(v20Preflight?.instrumentedGpuResourceCount
      === after?.instrumentedGpuResourceCount + 9
    && v20Preflight?.rootUuid === after?.rootUuid
    && v20Preflight?.aBucketBaseVersion === after?.aBucketBaseVersion
    && exactJson(v20Preflight?.rendererMemory, after?.rendererMemory)
    && exactJson(v20Preflight?.geometries, after?.geometries)
    && exactJson(v20Preflight?.materials, after?.materials)
    && IMMEDIATE_AIF_PHASE0_LANES.every((lane) => (
      exactJson(v20Preflight?.laneRoots?.[lane], after?.laneRoots?.[lane])
        && v20Preflight?.commandAttributes?.[lane]?.id
          === after?.commandAttributes?.[lane]?.id
        && v20Preflight?.commandAttributes?.[lane]?.version
          === after?.commandAttributes?.[lane]?.version + 1
        && v20Preflight?.commandAttributes?.[lane]?.sha256
          === cpuOracles().v20.schedules.canonical[lane].initialCommandSha256
    ))
    && PHASE0_SHARED_RESOURCE_SEMANTICS.every((semantic) => (
      v20Preflight?.sharedAttributes?.[semantic]?.id
          === after?.sharedAttributes?.[semantic]?.id
        && v20Preflight?.sharedAttributes?.[semantic]?.version
          === after?.sharedAttributes?.[semantic]?.version
            + preflightFromAfterDeltas[semantic]
        && v20Preflight?.sharedAttributes?.[semantic]?.byteLength
          === cpuOracles().v20.sharedResources[semantic].byteLength
        && v20Preflight?.sharedAttributes?.[semantic]?.sha256
          === cpuOracles().v20.sharedResources[semantic].cpuSourceSha256
    )),
  reasons, 'v20 post-load live/freeze accounting is not exact at preflight');
}

function runnerGlobalFinalGpuUseEvidence(evidence) {
  const resources = evidence?.resources ?? {};
  const lifecycle = evidence?.lifecycle ?? {};
  const records = [];
  const add = (sequence, kind, id = null) => {
    if (Number.isSafeInteger(sequence)) records.push({ sequence, kind, id });
  };
  for (const resource of resources?.gpuCreations ?? []) {
    add(resource?.sequence, `resource/${resource?.method}`, resource?.resourceId);
  }
  for (const view of resources?.textureViews ?? []) {
    add(view?.sequence, 'texture/createView', view?.viewId);
  }
  for (const record of [
    ...(evidence?.shaders?.modules ?? []),
    ...(evidence?.pipelines?.bindGroupLayouts ?? []),
    ...(evidence?.pipelines?.layouts ?? []),
    ...(evidence?.pipelines?.render ?? []),
    ...(evidence?.pipelines?.compute ?? []),
  ]) {
    add(record?.sequence, 'pipeline-resource', record?.pipelineId ?? record?.moduleId ?? null);
  }
  for (const [kind, traces] of [
    ['render-bundle', lifecycle?.renderBundleTraces ?? []],
    ['render-pass', lifecycle?.renderPassTraces ?? []],
    ['compute-pass', lifecycle?.computePassTraces ?? []],
  ]) {
    for (const trace of traces) {
      add(trace?.sequence, `${kind}/begin`, trace?.encoderId);
      for (const event of trace?.events ?? []) {
        add(event?.sequence, `${kind}/${event?.method}`, trace?.encoderId);
      }
      add(trace?.finishSequence, `${kind}/finish`, trace?.bundleId ?? trace?.encoderId);
    }
  }
  for (const write of lifecycle?.queueWriteBufferCalls ?? []) {
    add(write?.sequence, 'queue/writeBuffer', write?.bufferId);
  }
  for (const write of lifecycle?.queueWriteTextureCalls ?? []) {
    add(write?.sequence, 'queue/writeTexture', write?.textureId);
  }
  for (const copy of lifecycle?.queueCopyExternalImageToTextureCalls ?? []) {
    add(copy?.sequence, 'queue/copyExternalImageToTexture', copy?.destination?.textureId ?? null);
  }
  for (const map of lifecycle?.bufferMapEvents ?? []) {
    add(map?.sequence, `buffer/${map?.method}`, map?.resourceId);
  }
  for (const transfer of lifecycle?.commandEncoderTransferCalls ?? []) {
    add(transfer?.sequence, `command-encoder/${transfer?.method}`, transfer?.commandEncoderId);
  }
  for (const encoder of lifecycle?.commandEncoderTraces ?? []) {
    add(encoder?.sequence, 'command-encoder/create', encoder?.commandEncoderId);
    add(encoder?.finishSequence, 'command-encoder/finish', encoder?.commandBufferId);
  }
  for (const submission of lifecycle?.queueSubmissions ?? []) {
    add(submission?.sequence, 'queue/submit', submission?.commandBufferIds?.[0] ?? null);
  }
  records.sort((left, right) => left.sequence - right.sequence);
  const uniqueRecords = [...new Map(records.map((record) => [record.sequence, record])).values()]
    .sort((left, right) => left.sequence - right.sequence);
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-global-final-gpu-use',
    pass: uniqueRecords.length > 0,
    eventCount: uniqueRecords.length,
    representedRecordCount: records.length,
    duplicateRepresentationCount: records.length - uniqueRecords.length,
    finalUseSequence: uniqueRecords.at(-1)?.sequence ?? null,
    finalUse: uniqueRecords.at(-1) ?? null,
  };
}

export function validateThreeImmediatePhase0ErrorScopeDrainage(result) {
  const reasons = [];
  const drainage = result?.gpuErrors?.scopeDrainage;
  const scopedErrors = RUNNER_NORMAL_ERROR_SCOPE_DRAINAGE.scopedErrors;
  const lifecycle = result?.evidence?.lifecycle;
  const cleanup = result?.cleanup;
  addReasonUnless(exactJson(drainage, RUNNER_NORMAL_ERROR_SCOPE_DRAINAGE)
    && exactJson(result?.gpuErrors?.scoped, scopedErrors)
    && exactJson(lifecycle?.scopedErrors, scopedErrors)
    && exactJson(cleanup?.errorScopeDrainage, RUNNER_NORMAL_ERROR_SCOPE_DRAINAGE)
    && exactJson(lifecycle?.cleanup?.errorScopeDrainage,
      RUNNER_NORMAL_ERROR_SCOPE_DRAINAGE)
    && exactJson(cleanup?.errorScopeDrainage, drainage)
    && exactJson(lifecycle?.cleanup?.errorScopeDrainage, drainage),
  reasons, 'normal-completion GPU error-scope drainage is absent, inexact, or inconsistent');
  return { valid: reasons.length === 0, reasons };
}

function validateResourcesAndLifecycle(evidence, reasons) {
  const resources = evidence?.resources;
  const lifecycle = evidence?.lifecycle;
  const scenarios = evidence?.scenarios ?? [];
  const scenarioResources = scenarios.map((scenario) => scenario?.resources);
  const persistentIdentity = (value) => ({
    rendererMemory: value?.rendererMemory,
    rootUuid: value?.rootUuid,
    aBucketBaseVersion: value?.aBucketBaseVersion,
    laneRoots: value?.laneRoots,
    geometries: value?.geometries,
    materials: value?.materials,
    commandAttributes: value?.commandAttributes,
    sharedAttributes: value?.sharedAttributes,
  });
  const expectedDeltas = {
    aBucketBaseVersion: 3, fCommandVersion: 3, iRootVersion: 3,
    iBundleRecordCount: 3, aBundleRecordCount: 0, fBundleRecordCount: 0,
  };
  const globalIdentity = { rootUuid: null, geometries: null, materials: null,
    commandAttributeIds: null, sharedAttributeIds: null };
  validateScenarioLoadBoundary(scenarios, reasons);
  for (let scenarioIndex = 0; scenarioIndex < scenarios.length; scenarioIndex += 1) {
    const scenarioId = VISIBILITY_IDS[scenarioIndex];
    const record = scenarioResources[scenarioIndex];
    const preflight = record?.preflight;
    const restored = record?.restored;
    const postflight = record?.postflight;
    const observed = {
      aBucketBaseVersion: restored?.aBucketBaseVersion - preflight?.aBucketBaseVersion,
      fCommandVersion:
        restored?.commandAttributes?.F?.version - preflight?.commandAttributes?.F?.version,
      iRootVersion: restored?.laneRoots?.I?.version - preflight?.laneRoots?.I?.version,
      iBundleRecordCount:
        restored?.laneRoots?.I?.bundleRecordCount
          - preflight?.laneRoots?.I?.bundleRecordCount,
      aBundleRecordCount:
        restored?.laneRoots?.A?.bundleRecordCount
          - preflight?.laneRoots?.A?.bundleRecordCount,
      fBundleRecordCount:
        restored?.laneRoots?.F?.bundleRecordCount
          - preflight?.laneRoots?.F?.bundleRecordCount,
    };
    const commandIds = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => [lane, preflight?.commandAttributes?.[lane]?.id],
    ));
    const sharedIds = Object.fromEntries(Object.entries(preflight?.sharedAttributes ?? {})
      .map(([semantic, value]) => [semantic, value?.id]));
    const expectedShared = cpuOracles()[scenarioId].sharedResources;
    addReasonUnless(record?.pass === true
      && preflight?.label === `${scenarioId}/preflight`
      && restored?.label === `${scenarioId}/restored`
      && postflight?.label === `${scenarioId}/postflight`
      && preflight?.rootUuid === restored?.rootUuid
      && restored?.rootUuid === postflight?.rootUuid
      && exactJson(preflight?.geometries, restored?.geometries)
      && exactJson(restored?.geometries, postflight?.geometries)
      && exactJson(preflight?.materials, restored?.materials)
      && exactJson(restored?.materials, postflight?.materials)
      && exactArray(Object.keys(preflight?.laneRoots ?? {}), IMMEDIATE_AIF_PHASE0_LANES)
      && exactArray(Object.keys(preflight?.geometries ?? {}), IMMEDIATE_AIF_PHASE0_LANES)
      && exactArray(Object.keys(preflight?.materials ?? {}), IMMEDIATE_AIF_PHASE0_LANES)
      && exactArray(Object.keys(preflight?.commandAttributes ?? {}),
        IMMEDIATE_AIF_PHASE0_LANES)
      && exactArray(Object.keys(preflight?.sharedAttributes ?? {}),
        PHASE0_SHARED_RESOURCE_SEMANTICS)
      && exactJson(persistentIdentity(restored), persistentIdentity(postflight))
      && Number.isSafeInteger(preflight?.instrumentedGpuResourceCount)
      && Number.isSafeInteger(restored?.instrumentedGpuResourceCount)
      && Number.isSafeInteger(postflight?.instrumentedGpuResourceCount)
      && preflight.instrumentedGpuResourceCount <= restored.instrumentedGpuResourceCount
      && restored.instrumentedGpuResourceCount <= postflight.instrumentedGpuResourceCount
      && exactJson(observed, expectedDeltas)
      && exactJson(record?.transitionDeltas, {
        pass: true, expected: expectedDeltas, observed,
      })
      && restored?.laneRoots?.A?.version === preflight?.laneRoots?.A?.version
      && restored?.laneRoots?.F?.version === preflight?.laneRoots?.F?.version
      && restored?.laneRoots?.A?.bundleRecordCount
        === preflight?.laneRoots?.A?.bundleRecordCount
      && restored?.laneRoots?.F?.bundleRecordCount
        === preflight?.laneRoots?.F?.bundleRecordCount
      && restored?.commandAttributes?.A?.version === preflight?.commandAttributes?.A?.version
      && restored?.commandAttributes?.I?.version === preflight?.commandAttributes?.I?.version
      && exactJson(restored?.rendererMemory, preflight?.rendererMemory)
      && IMMEDIATE_AIF_PHASE0_LANES.every((lane) => (
        preflight?.laneRoots?.[lane]?.uuid === restored?.laneRoots?.[lane]?.uuid
          && preflight?.geometries?.[lane] === restored?.geometries?.[lane]
          && preflight?.materials?.[lane] === restored?.materials?.[lane]
      )),
    reasons, `${scenarioId} preflight-to-restored resource deltas/identities are invalid`);
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const expectedCommand = cpuOracles()[scenarioId].schedules.canonical[lane];
      addReasonUnless(preflight?.commandAttributes?.[lane]?.id === commandIds[lane]
        && preflight?.commandAttributes?.[lane]?.sha256
          === expectedCommand.initialCommandSha256,
      reasons, `${scenarioId}/preflight/${lane} initial command staging bytes are invalid`);
      const expectedRestoredHash = lane === 'F'
        ? expectedCommand.commandSha256 : expectedCommand.initialCommandSha256;
      for (const state of [restored, postflight]) {
        addReasonUnless(state?.commandAttributes?.[lane]?.id === commandIds[lane]
          && state?.commandAttributes?.[lane]?.sha256 === expectedRestoredHash,
        reasons, `${scenarioId}/${state?.label}/${lane} restored command staging bytes are invalid`);
      }
    }
    for (const [semantic, expected] of Object.entries(expectedShared)) {
      for (const state of [preflight, restored, postflight]) {
        addReasonUnless(state?.sharedAttributes?.[semantic]?.id === sharedIds[semantic]
          && state?.sharedAttributes?.[semantic]?.byteLength === expected.byteLength
          && state?.sharedAttributes?.[semantic]?.sha256 === expected.cpuSourceSha256,
        reasons, `${scenarioId}/${state?.label}/${semantic} CPU resource state is invalid`);
      }
      addReasonUnless(restored?.sharedAttributes?.[semantic]?.version
        === preflight?.sharedAttributes?.[semantic]?.version,
      reasons, `${scenarioId}/${semantic} shared attribute version changed during schedules`);
    }
    if (scenarioIndex === 0) {
      Object.assign(globalIdentity, {
        rootUuid: preflight?.rootUuid,
        geometries: preflight?.geometries,
        materials: preflight?.materials,
        commandAttributeIds: commandIds,
        sharedAttributeIds: sharedIds,
      });
    } else {
      addReasonUnless(preflight?.rootUuid === globalIdentity.rootUuid
        && exactJson(preflight?.geometries, globalIdentity.geometries)
        && exactJson(preflight?.materials, globalIdentity.materials)
        && exactJson(commandIds, globalIdentity.commandAttributeIds)
        && exactJson(sharedIds, globalIdentity.sharedAttributeIds),
      reasons, 'persistent resource identities changed across visibility scenarios');
    }
  }

  const creationInventory = phase0CreationEventInventory(evidence);
  const markers = lifecycle?.instrumentationMarkers ?? [];
  const freezeAggregate = resources?.persistentResourceFreezes;
  addReasonUnless(freezeAggregate?.schemaVersion === 1
    && freezeAggregate?.kind === 'immediate-aif-phase0-persistent-resource-freezes'
    && freezeAggregate?.pass === true
    && Array.isArray(freezeAggregate?.records) && freezeAggregate.records.length === 2
    && exactJson(freezeAggregate.records,
      scenarios.map((scenario) => scenario?.persistentResourceFreeze)),
  reasons, 'aggregate persistent resource freeze inventory is invalid');
  for (let scenarioIndex = 0; scenarioIndex < scenarios.length; scenarioIndex += 1) {
    const scenarioId = VISIBILITY_IDS[scenarioIndex];
    const freeze = scenarios[scenarioIndex]?.persistentResourceFreeze;
    const before = freeze?.preflight;
    const after = freeze?.postflight;
    const beforePersistent = before?.records?.filter(
      (record) => record?.resourceClass !== 'render-bundle',
    ) ?? [];
    const afterPersistent = after?.records?.filter(
      (record) => record?.resourceClass !== 'render-bundle',
    ) ?? [];
    const beforeBundles = before?.records?.filter(
      (record) => record?.resourceClass === 'render-bundle',
    ) ?? [];
    const afterBundles = after?.records?.filter(
      (record) => record?.resourceClass === 'render-bundle',
    ) ?? [];
    const beforeIds = new Set(beforeBundles.map((record) => record?.bundleId));
    const retained = afterBundles.filter((record) => beforeIds.has(record?.bundleId));
    const added = afterBundles.filter((record) => !beforeIds.has(record?.bundleId));
    const diagnosticPhases = ['S1', 'S2', 'canonical'].flatMap((scheduleId) => [
      `phase0/diagnostic-rerecord/${scenarioId}/${scheduleId}/I/address`,
      `phase0/diagnostic-rerecord/${scenarioId}/${scheduleId}/I/object-id`,
    ]);
    const expectedProductionPhases = expectedPhase0CallbackTraversal(scenarios)
      .filter((entry) => entry.scenarioId === scenarioId && entry.kind === 'rerecord')
      .map((entry) => `${entry.phase}/production`);
    const diagnosticBundles = added.filter(
      (record) => diagnosticPhases.includes(record?.capturePhase),
    );
    const productionBundles = added.filter(
      (record) => expectedProductionPhases.includes(record?.capturePhase),
    );
    const exactRetained = retained.length === beforeBundles.length
      && beforeBundles.every((record) => exactJson(record,
        retained.find((candidate) => candidate?.bundleId === record?.bundleId)));
    const delta = freeze?.prescribedPersistentDelta;
    const deltaExact = delta?.schemaVersion === 1
      && delta?.kind === 'immediate-aif-phase0-prescribed-persistent-resource-delta'
      && delta?.pass === true && delta?.scenarioId === scenarioId
      && delta?.nonBundlePersistentExact === true
      && exactJson(beforePersistent, afterPersistent)
      && delta?.retainedBundleCount === retained.length
      && delta?.expectedRetainedBundleCount === beforeBundles.length
      && delta?.exactRetainedBundles === exactRetained && exactRetained
      && delta?.expectedAddedBundleCount === 9 && delta?.addedBundleCount === 9
      && delta?.addedBundleTopologyExact === true && added.length === 9
      && exactArray(delta?.expectedDiagnosticPhases, diagnosticPhases)
      && exactJson(delta?.diagnosticBundles, diagnosticBundles)
      && diagnosticBundles.length === 6
      && exactArray(delta?.productionSchedules, ['S1', 'S2', 'canonical'])
      && exactJson(delta?.productionBundles, productionBundles)
      && productionBundles.length === 3
      && exactJson(delta?.addedBundles, added)
      && added.every((record) => record?.nativeFinishReturned === true
        && typeof record?.bundleId === 'string'
        && Number.isSafeInteger(record?.finishSequence)
        && record.finishSequence > record.sequence);
    const immutableIds = [
      ...(scenarios[scenarioIndex]?.gpuResourceCommitments?.records ?? [])
        .map((record) => record?.gpuBufferId),
      ...(resources?.geometryFixtures?.mergedProduction?.gpuRecords ?? [])
        .filter((record) => record?.semantic !== 'bucketBase')
        .map((record) => record?.gpuBufferId),
    ];
    const scenarioWrites = (lifecycle?.queueWriteBufferCalls ?? []).filter(
      (record) => record?.sequence > before?.markerSequence
        && record?.sequence < after?.markerSequence
        && immutableIds.includes(record?.bufferId),
    );
    const scenarioMaps = (lifecycle?.bufferMapEvents ?? []).filter(
      (record) => record?.sequence > before?.markerSequence
        && record?.sequence < after?.markerSequence
        && immutableIds.includes(record?.resourceId),
    );
    addReasonUnless(freeze?.schemaVersion === 1
      && freeze?.kind === 'immediate-aif-phase0-persistent-resource-freeze-comparison'
      && freeze?.pass === true && freeze?.scenarioId === scenarioId
      && validatePersistentInventorySnapshot(before,
        `${scenarioId}/pre-ordered-challenges`, scenarioId, creationInventory, markers)
      && validatePersistentInventorySnapshot(after,
        `${scenarioId}/post-ordered-challenges`, scenarioId, creationInventory, markers)
      && deltaExact && new Set(immutableIds).size === 12
      && scenarioWrites.length === 0 && scenarioMaps.length === 0,
    reasons, `${scenarioId} persistent creation delta or immutable scenario window is invalid`);
  }

  const creations = resources?.gpuCreations;
  const destructions = resources?.gpuDestructions;
  const staging = resources?.stagingLifecycle;
  const addressPositionAudit = validateThreeImmediatePhase0AddressDiagnosticPositions(
    resources?.addressDiagnosticPositions,
    {
      lifecycle,
      gpuCreations: Array.isArray(creations) ? creations : [],
      scheduleInstallations: resources?.scheduleInstallations?.records ?? [],
    },
  );
  addReasonUnless(addressPositionAudit.valid, reasons,
    `address diagnostic position realization is invalid: ${addressPositionAudit.reasons.join('; ')}`);
  const queueWriteAudit = validateThreeImmediatePhase0QueueWriteBufferLedger(
    resources?.queueWriteBufferLedger,
    { lifecycle, scenarios, resources, pipelines: evidence?.pipelines },
  );
  addReasonUnless(queueWriteAudit.valid, reasons,
    `queue.writeBuffer partition is invalid: ${queueWriteAudit.reasons.join('; ')}`);
  const stagingCreations = Array.isArray(creations)
    ? creations.filter((record) => record?.resourceClass === 'readback-staging') : [];
  const diagnosticStaging = stagingCreations.filter(
    (record) => /\/(?:address|object-id)$/u.test(record?.capturePhase ?? ''),
  );
  const stagingProjection = stagingCreations.map((resource) => ({
    resourceId: resource.resourceId,
    capturePhase: resource.capturePhase,
    createSequence: resource.sequence,
    destroySequence: resource.destroySequence,
    destroyCapturePhase: resource.destroyCapturePhase,
    destroyed: resource.destroyed,
    destroyCallCount: resource.destroyCallCount,
    size: resource.size,
    usage: resource.usage,
  }));
  addReasonUnless(Array.isArray(creations) && creations.length > 0
    && creations.every((record, index) => typeof record?.resourceId === 'string'
      && Number.isSafeInteger(record?.sequence)
      && (index === 0 || record.sequence > creations[index - 1].sequence)
      && ['createBuffer', 'createTexture', 'createBindGroup', 'createSampler',
        'createQuerySet', 'importExternalTexture']
        .includes(record?.method))
    && new Set(creations.map((record) => record.resourceId)).size === creations.length
    && Array.isArray(destructions)
    && new Set(destructions.map((record) => record?.sequence)).size === destructions.length
    && destructions.every((destruction) => {
      const matches = creations.filter((creation) => (
        creation?.resourceId === destruction?.resourceId
      ));
      return matches.length === 1
        && ['createBuffer', 'createTexture'].includes(matches[0]?.method)
        && destruction?.method === 'destroy'
        && destruction?.resourceClass === matches[0]?.resourceClass
        && destruction?.sequence === matches[0]?.destroySequence
        && destruction?.capturePhase === matches[0]?.destroyCapturePhase
        && destruction?.callCount === matches[0]?.destroyCallCount
        && destruction.sequence > matches[0].sequence;
    })
    && staging?.schemaVersion === 1
    && staging?.kind === 'immediate-aif-phase0-readback-staging-lifecycle'
    && staging?.pass === true
    && staging?.createdCount === stagingCreations.length
    && staging?.diagnosticCreatedCount === diagnosticStaging.length
    && staging?.liveCount === 0
    && exactJson(staging?.resources, stagingProjection)
    && stagingCreations.length > 0
    && stagingCreations.every((resource) => resource?.method === 'createBuffer'
      && typeof resource?.capturePhase === 'string' && resource.capturePhase.length > 0
      && resource?.destroyCapturePhase === resource.capturePhase
      && resource?.destroyed === true && resource?.destroyCallCount === 1
      && Number.isSafeInteger(resource?.destroySequence)
      && resource.destroySequence > resource.sequence
      && destructions.filter((entry) => entry?.resourceId === resource.resourceId).length === 1),
  reasons, 'GPU creation/destruction and transient staging inventory is invalid');

  const bufferCreations = creations?.filter((record) => record?.method === 'createBuffer') ?? [];
  const globalFinalUse = runnerGlobalFinalGpuUseEvidence(evidence);
  const globalFinalUseSequence = globalFinalUse.finalUseSequence ?? 0;
  const cleanupStartSequence = lifecycle?.cleanup?.cleanupStartSequence;
  const persistentBufferCreations = bufferCreations.filter(
    (record) => record?.resourceClass === 'persistent-or-upload-buffer',
  );
  const persistentCleanup = lifecycle?.cleanup?.persistentBufferCleanup;
  const expectedPersistentCleanupRecords = persistentBufferCreations.map((resource) => ({
    resourceId: resource.resourceId,
    resourceClass: resource.resourceClass,
    creationPhase: resource.capturePhase,
    createSequence: resource.sequence,
    destroyCallCount: resource.destroyCallCount,
    destroySequence: resource.destroySequence,
    destroyCapturePhase: resource.destroyCapturePhase,
    afterGlobalFinalUse: resource.destroySequence > globalFinalUseSequence,
    disposition: 'explicit-destroy-exactly-once',
    pass: true,
  }));
  const expectedSweepRecords = persistentBufferCreations.map((resource) => ({
    resourceId: resource.resourceId,
    attempted: resource.destroyCapturePhase === 'phase0/cleanup/persistent-buffers',
    error: null,
    destroyCallCount: resource.destroyCallCount,
    destroyed: resource.destroyed,
    destroySequence: resource.destroySequence,
    destroyCapturePhase: resource.destroyCapturePhase,
  }));
  const destroyedBufferIds = destructions?.filter((record) => (
    record?.resourceClass === 'readback-staging'
      || record?.resourceClass === 'persistent-or-upload-buffer'
  )).map((record) => record?.resourceId) ?? [];
  addReasonUnless(persistentBufferCreations.length > 0
    && bufferCreations.every((resource) => resource?.destroyed === true
      && resource?.destroyCallCount === 1
      && Number.isSafeInteger(resource?.destroySequence)
      && resource.destroySequence > resource.sequence)
    && destroyedBufferIds.length === bufferCreations.length
    && new Set(destroyedBufferIds).size === destroyedBufferIds.length
    && exactArray([...destroyedBufferIds].sort(),
      bufferCreations.map((resource) => resource.resourceId).sort())
    && persistentBufferCreations.every((resource) => (
      /^phase0\/cleanup\//u.test(resource?.destroyCapturePhase ?? '')
        && resource.destroySequence > globalFinalUseSequence
        && Number.isSafeInteger(cleanupStartSequence)
        && resource.destroySequence > cleanupStartSequence
    ))
    && persistentCleanup?.schemaVersion === 1
    && persistentCleanup?.kind === 'immediate-aif-phase0-persistent-buffer-cleanup'
    && persistentCleanup?.pass === true
    && persistentCleanup?.terminalDeviceCoverage === true
    && persistentCleanup?.resourceCount === persistentBufferCreations.length
    && persistentCleanup?.explicitlyDestroyedCount === persistentBufferCreations.length
    && exactJson(persistentCleanup?.records, expectedPersistentCleanupRecords)
    && persistentCleanup?.explicitSweep?.pass === true
    && exactJson(persistentCleanup?.explicitSweep?.records, expectedSweepRecords),
  reasons, 'persistent GPU buffer destruction/terminal-device closure is invalid');

  const textureCreationsForCleanup = creations?.filter(
    (record) => record?.method === 'createTexture',
  ) ?? [];
  const textureDestructions = destructions?.filter(
    (record) => record?.resourceClass === 'persistent-texture',
  ) ?? [];
  const expectedTextureCleanupRecords = textureCreationsForCleanup.map((resource) => ({
    resourceId: resource.resourceId,
    resourceClass: resource.resourceClass,
    creationPhase: resource.capturePhase,
    createSequence: resource.sequence,
    destroyCallCount: resource.destroyCallCount,
    destroySequence: resource.destroySequence,
    destroyCapturePhase: resource.destroyCapturePhase,
    classification: resources?.textureDestructionBeforeCleanup?.declaredTargetTextureIds
      ?.includes(resource.resourceId)
      ? 'declared-render-target' : 'renderer-internal',
    afterGlobalFinalUse: resource.destroySequence > globalFinalUseSequence,
    disposition: 'explicit-destroy-exactly-once',
    pass: true,
  }));
  const expectedTextureSweepRecords = textureCreationsForCleanup.map((resource) => ({
    resourceId: resource.resourceId,
    attempted: resource.destroyCapturePhase === 'phase0/cleanup/persistent-textures',
    error: null,
    destroyCallCount: resource.destroyCallCount,
    destroyed: resource.destroyed,
    destroySequence: resource.destroySequence,
    destroyCapturePhase: resource.destroyCapturePhase,
  }));
  const persistentTextureCleanup = lifecycle?.cleanup?.persistentTextureCleanup;
  const preCleanupTextures = resources?.textureDestructionBeforeCleanup;
  const challengeTextureInventories = scenarios.flatMap((scenario) => (
    (scenario?.snapshots ?? []).slice(0, 3).map(
      (snapshot) => snapshot?.productionChallenge?.immutableTextureResources,
    )
  ));
  const declaredTargetTextures = challengeTextureInventories[0] ?? [];
  const declaredTargetTextureIds = declaredTargetTextures.map(
    (record) => record?.gpuTextureId,
  );
  const expectedTargetSemantics = [
    'production.color', 'production.depth', 'address.color',
    'objectId.color', 'objectId.depth',
  ];
  const bindGroupTextureViewIds = new Set((creations ?? [])
    .filter((record) => record?.method === 'createBindGroup')
    .flatMap((record) => record?.entries ?? [])
    .filter((entry) => entry?.bufferId === null && typeof entry?.resourceId === 'string')
    .map((entry) => entry.resourceId));
  const attachmentUses = (lifecycle?.renderPassTraces ?? []).flatMap((trace) => [
    ...(trace?.colorAttachments ?? []).flatMap((attachment) => [
      { viewId: attachment?.viewId, textureId: attachment?.textureId, class: 'color' },
      ...(attachment?.resolveTargetViewId == null ? [] : [{
        viewId: attachment.resolveTargetViewId,
        textureId: attachment.resolveTargetTextureId,
        class: 'color-resolve',
      }]),
    ]),
    ...(trace?.depthStencilAttachment == null ? [] : [{
      viewId: trace.depthStencilAttachment.viewId,
      textureId: trace.depthStencilAttachment.textureId,
      class: 'depth-stencil',
    }]),
  ].filter((record) => typeof record.viewId === 'string'));
  const targetTextureIdSet = new Set(declaredTargetTextureIds);
  const declaredTargetById = new Map(declaredTargetTextures.map(
    (record) => [record?.gpuTextureId, record],
  ));
  const commonTargetTextureIds = new Map([
    ['production.color', resources?.commonResources?.targets?.production?.colorTextureId],
    ['production.depth', resources?.commonResources?.targets?.production?.depthTextureIds?.[0]],
    ['address.color', resources?.commonResources?.targets?.address?.colorTextureId],
    ['objectId.color', resources?.commonResources?.targets?.objectId?.colorTextureId],
    ['objectId.depth', resources?.commonResources?.targets?.objectId?.depthTextureIds?.[0]],
  ]);
  const expectedTextureInventory = textureCreationsForCleanup.map((resource) => {
    const views = (resources?.textureViews ?? []).filter(
      (view) => view?.textureId === resource.resourceId,
    );
    const boundViewIds = views.map((view) => view.viewId).filter(
      (viewId) => bindGroupTextureViewIds.has(viewId),
    );
    const attachmentViewIds = views.map((view) => view.viewId).filter(
      (viewId) => attachmentUses.some((use) => use.viewId === viewId),
    );
    const classification = targetTextureIdSet.has(resource.resourceId)
      ? 'declared-render-target' : 'renderer-internal';
    const targetSemantic = declaredTargetById.get(resource.resourceId)?.semantic ?? null;
    const expectedAttachmentClass = targetSemantic?.endsWith('.depth')
      ? 'depth-stencil' : 'color';
    const classExact = classification === 'declared-render-target'
      ? views.length > 0
        && boundViewIds.length === 0
        && attachmentViewIds.length === views.length
        && commonTargetTextureIds.get(targetSemantic) === resource.resourceId
        && views.every((view) => attachmentUses.some(
          (use) => use.viewId === view.viewId && use.textureId === resource.resourceId
            && use.class === expectedAttachmentClass,
        ))
      : views.length > 0
        && attachmentViewIds.length === 0
        && boundViewIds.length === views.length;
    return {
      textureId: resource.resourceId,
      classification,
      createSequence: resource.sequence,
      capturePhase: resource.capturePhase,
      viewIds: views.map((view) => view.viewId),
      boundViewIds,
      attachmentViewIds,
      pass: classExact,
    };
  });
  const expectedInternalTextureIds = textureCreationsForCleanup
    .map((record) => record.resourceId)
    .filter((resourceId) => !targetTextureIdSet.has(resourceId));
  const expectedAllowedTextureDestroyCapturePhases = [...new Set(
    textureCreationsForCleanup.map((resource) => resource.destroyCapturePhase),
  )].sort();
  addReasonUnless(preCleanupTextures?.schemaVersion === 1
    && preCleanupTextures?.kind
      === 'immediate-aif-phase0-texture-destruction-before-cleanup'
    && preCleanupTextures?.pass === true
    && preCleanupTextures?.textureCount
      === textureCreationsForCleanup.length
    && exactArray(preCleanupTextures?.textureIds,
      textureCreationsForCleanup.map((resource) => resource.resourceId))
    && exactJson(preCleanupTextures?.destructionRecords, [])
    && textureCreationsForCleanup.length >= 5
    && challengeTextureInventories.length === 6
    && challengeTextureInventories.every(
      (records) => exactJson(records, declaredTargetTextures),
    )
    && exactArray(declaredTargetTextures.map((record) => record?.semantic),
      expectedTargetSemantics)
    && declaredTargetTextureIds.length === 5
    && new Set(declaredTargetTextureIds).size === 5
    && declaredTargetTextureIds.every((textureId) => textureCreationsForCleanup.some(
      (record) => record.resourceId === textureId,
    ))
    && preCleanupTextures?.declaredTargetTextureCount === 5
    && exactJson(preCleanupTextures?.declaredTargetTextures, declaredTargetTextures)
    && exactArray(preCleanupTextures?.declaredTargetTextureIds, declaredTargetTextureIds)
    && preCleanupTextures?.targetTextureInventoryExact === true
    && preCleanupTextures?.internalTextureCount === expectedInternalTextureIds.length
    && exactArray(preCleanupTextures?.internalTextureIds, expectedInternalTextureIds)
    && exactJson(preCleanupTextures?.textureInventory, expectedTextureInventory)
    && expectedTextureInventory.every((record) => record.pass)
    && textureDestructions.length === textureCreationsForCleanup.length
    && new Set(textureDestructions.map((record) => record?.resourceId)).size
      === textureCreationsForCleanup.length
    && textureCreationsForCleanup.every((resource) => resource?.resourceClass
      === 'persistent-texture'
      && resource?.destroyed === true && resource?.destroyCallCount === 1
      && Number.isSafeInteger(resource?.destroySequence)
      && resource.destroySequence > resource.sequence
      && resource.destroySequence > globalFinalUseSequence
      && Number.isSafeInteger(cleanupStartSequence)
      && resource.destroySequence > cleanupStartSequence
      && /^phase0\/cleanup\/(?:address-diagnostics|object-id-diagnostics|production-target|renderer|persistent-textures)$/u
        .test(resource?.destroyCapturePhase ?? '')
      && textureDestructions.filter(
        (record) => record?.resourceId === resource.resourceId
          && record?.sequence === resource.destroySequence
          && record?.capturePhase === resource.destroyCapturePhase
          && record?.callCount === 1,
      ).length === 1)
    && persistentTextureCleanup?.schemaVersion === 1
    && persistentTextureCleanup?.kind === 'immediate-aif-phase0-persistent-texture-cleanup'
    && persistentTextureCleanup?.pass === true
    && persistentTextureCleanup?.terminalDeviceCoverage === true
    && persistentTextureCleanup?.resourceCount === textureCreationsForCleanup.length
    && persistentTextureCleanup?.explicitlyDestroyedCount === textureCreationsForCleanup.length
    && persistentTextureCleanup?.declaredRenderTargetCount === 5
    && persistentTextureCleanup?.internalTextureCount === expectedInternalTextureIds.length
    && exactArray(persistentTextureCleanup?.allowedDestroyCapturePhases,
      expectedAllowedTextureDestroyCapturePhases)
    && exactJson(persistentTextureCleanup?.explicitSweep?.records,
      expectedTextureSweepRecords)
    && persistentTextureCleanup?.explicitSweep?.pass === true
    && exactJson(persistentTextureCleanup?.records, expectedTextureCleanupRecords),
  reasons, 'persistent GPU texture creation/use/destruction lifecycle is not exact');

  const byCapturePhase = new Map();
  for (const resource of stagingCreations) {
    const records = byCapturePhase.get(resource.capturePhase) ?? [];
    records.push(resource);
    byCapturePhase.set(resource.capturePhase, records);
  }
  const declaredReadbackPhases = new Set();
  const requireReadbacks = (phase, expectedSizes, label) => {
    declaredReadbackPhases.add(phase);
    const observedSizes = (byCapturePhase.get(phase) ?? [])
      .map((resource) => resource?.size).sort((left, right) => left - right);
    const sortedExpected = [...expectedSizes].sort((left, right) => left - right);
    addReasonUnless(exactArray(observedSizes, sortedExpected), reasons,
      `${label} staging sizes/count differ from ${JSON.stringify(sortedExpected)}`);
  };
  const traversal = expectedPhase0CallbackTraversal(scenarios, reasons);
  for (const entry of traversal) {
      const callback = entry.callback;
      const expectedPhases = {
        production: `${entry.phase}/production`,
        command: `${entry.phase}/command-readback`,
        address: `${entry.phase}/address`,
        objectId: `${entry.phase}/object-id`,
        visibleIds: `${entry.phase}/visible-id-readback`,
      };
      addReasonUnless(exactJson(callback?.phases, expectedPhases), reasons,
        `${entry.phase} retained subphases are not exact`);
      requireReadbacks(expectedPhases.production,
        [TARGET_BYTE_LENGTH, TARGET_BYTE_LENGTH],
        `${entry.phase}/production`);
      requireReadbacks(expectedPhases.command, [640], `${entry.phase}/command`);
      requireReadbacks(expectedPhases.address, [ADDRESS_BYTE_LENGTH],
        `${entry.phase}/address`);
      requireReadbacks(expectedPhases.objectId, [TARGET_BYTE_LENGTH],
        `${entry.phase}/object-id`);
      requireReadbacks(expectedPhases.visibleIds, [ADDRESS_BYTE_LENGTH],
        `${entry.phase}/visible-ids`);
  }
  for (const scenario of scenarios) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const phase = `phase0/live/${scenario?.scenarioId}/${lane}`;
      requireReadbacks(phase, [4, 640, ADDRESS_BYTE_LENGTH], phase);
    }
    const commitmentPhase = `phase0/resources/${scenario?.scenarioId}/shared-commitment`;
    requireReadbacks(commitmentPhase,
      Object.values(cpuOracles()[scenario?.scenarioId].sharedResources)
        .map((expected) => expected.byteLength), commitmentPhase);
  }
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    requireReadbacks(`phase0/prime/${lane}`,
      [TARGET_BYTE_LENGTH, TARGET_BYTE_LENGTH], `phase0/prime/${lane}`);
  }
  requireReadbacks('phase0/resources/global/geometry-postflight',
    GEOMETRY_FINGERPRINT_ORACLES.merged.records.map((record) => record.byteLength),
    'phase0/resources/global/geometry-postflight');
  for (const scenarioIndex of [0, 1]) {
    for (let localOrdinal = 1; localOrdinal <= 4; localOrdinal += 1) {
      const scenarioId = VISIBILITY_IDS[scenarioIndex];
      const ordinal = scenarioIndex * 4 + localOrdinal;
      const scheduleId = ['canonical', 'S1', 'S2', 'canonical'][localOrdinal - 1];
      const basePhase = `phase0/schedule-install/${scenarioId}/${ordinal}-${scheduleId}`;
      for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
        requireReadbacks(`${basePhase}/command-readback/${lane}`, [640],
          `${basePhase}/${lane}`);
      }
    }
  }
  addReasonUnless(stagingCreations.length === 825
    && diagnosticStaging.length === 252
    && staging?.expectedCreatedCount === 825
    && staging?.readbackPhaseCount === 666
    && staging?.expectedReadbackPhaseCount === 666
    && exactArray(staging?.readbackPhases, [...declaredReadbackPhases].sort())
    && byCapturePhase.size === declaredReadbackPhases.size
    && [...byCapturePhase.keys()].every((phase) => declaredReadbackPhases.has(phase)),
  reasons, 'staging buffers include missing, extra, or unclassified readback phases');

  const detachments = resources?.immediateSourceDetachments;
  addReasonUnless(Array.isArray(detachments) && detachments.length === 2
    && detachments.every((record, index) => {
      const scenario = scenarios[index];
      const snapshotRecord = scenario?.snapshots?.[1]?.sourceDetachment;
      return record?.scenarioId === VISIBILITY_IDS[index]
        && exactJson(record, snapshotRecord);
    }),
  reasons, 'global immediate-source detachment inventory is not the two scenario records');

  const querySetCreations = creations?.filter((record) => record?.method === 'createQuerySet') ?? [];
  addReasonUnless(resources?.querySets?.schemaVersion === 1
    && resources?.querySets?.kind === 'immediate-aif-phase0-zero-query-set-evidence'
    && resources?.querySets?.pass === true && resources.querySets.count === 0
    && exactJson(resources.querySets.records, []) && querySetCreations.length === 0
    && (lifecycle?.renderPassTraces ?? []).every((trace) => (
      trace?.timestampWrites === null && trace?.occlusionQuerySetId === null
    ))
    && (lifecycle?.computePassTraces ?? []).every(
      (trace) => trace?.timestampWrites === null,
    ),
  reasons, 'query sets, occlusion queries, or timestamp writes crossed the timing-free boundary');

  const importedExternalTextures = creations?.filter(
    (record) => record?.method === 'importExternalTexture',
  ) ?? [];
  addReasonUnless(importedExternalTextures.length === 0
    && resources?.externalTextures?.schemaVersion === 1
    && resources?.externalTextures?.kind
      === 'immediate-aif-phase0-zero-external-texture-evidence'
    && resources.externalTextures?.pass === true
    && resources.externalTextures?.methodAvailable === true
    && resources.externalTextures?.methodInstrumented === true
    && resources.externalTextures?.count === 0
    && exactJson(resources.externalTextures?.records, []),
  reasons, 'GPUDevice.importExternalTexture availability/instrumentation/zero-use is invalid');

  const samplerCreations = creations?.filter((record) => record?.method === 'createSampler') ?? [];
  const samplerIds = new Set(samplerCreations.map((record) => record.resourceId));
  const allowedAddressModes = new Set(['clamp-to-edge', 'repeat', 'mirror-repeat']);
  const allowedFilters = new Set(['nearest', 'linear']);
  const allowedCompare = new Set([
    null, 'never', 'less', 'equal', 'less-equal', 'greater', 'not-equal',
    'greater-equal', 'always',
  ]);
  const boundNonBufferIds = new Set((creations ?? [])
    .filter((record) => record?.method === 'createBindGroup')
    .flatMap((record) => record?.entries ?? [])
    .filter((entry) => entry?.bufferId === null && typeof entry?.resourceId === 'string')
    .map((entry) => entry.resourceId));
  addReasonUnless(resources?.samplers?.schemaVersion === 1
    && resources?.samplers?.kind === 'immediate-aif-phase0-sampler-descriptor-evidence'
    && resources?.samplers?.pass === true
    && exactJson(resources?.samplers?.records, samplerCreations)
    && samplerIds.size === samplerCreations.length
    && new Set(samplerCreations.map((record) => JSON.stringify(record.sampler))).size
      === samplerCreations.length
    && samplerCreations.every((record) => {
      const sampler = record?.sampler;
      return typeof record?.capturePhase === 'string'
        && !/\/(?:production|command-readback|address|object-id|visible-id-readback)$/u
          .test(record.capturePhase)
        && exactArray(Object.keys(sampler ?? {}), [
          'addressModeU', 'addressModeV', 'addressModeW', 'magFilter', 'minFilter',
          'mipmapFilter', 'lodMinClamp', 'lodMaxClamp', 'compare', 'maxAnisotropy',
        ])
        && allowedAddressModes.has(sampler.addressModeU)
        && allowedAddressModes.has(sampler.addressModeV)
        && allowedAddressModes.has(sampler.addressModeW)
        && allowedFilters.has(sampler.magFilter)
        && allowedFilters.has(sampler.minFilter)
        && allowedFilters.has(sampler.mipmapFilter)
        && sampler.lodMinClamp === 0 && sampler.lodMaxClamp === 32
        && allowedCompare.has(sampler.compare)
        && Number.isSafeInteger(sampler.maxAnisotropy)
        && sampler.maxAnisotropy >= 1
        && (sampler.maxAnisotropy === 1
          || (sampler.magFilter === 'linear' && sampler.minFilter === 'linear'
            && sampler.mipmapFilter === 'linear'))
        && boundNonBufferIds.has(record.resourceId);
    }),
  reasons, 'sampler descriptor/cache/resource-to-bind-group inventory is invalid');

  const textureViews = resources?.textureViews ?? [];
  const textureCreations = creations?.filter(
    (record) => record?.method === 'createTexture',
  ) ?? [];
  const usedTextureViewIds = new Set([
    ...(lifecycle?.renderPassTraces ?? []).flatMap((trace) => [
      ...(trace?.colorAttachments ?? []).flatMap((attachment) => [
        attachment?.viewId, attachment?.resolveTargetViewId,
      ]),
      trace?.depthStencilAttachment?.viewId,
    ]),
    ...(creations ?? []).filter((record) => record?.method === 'createBindGroup')
      .flatMap((record) => record?.entries ?? [])
      .filter((entry) => entry?.bufferId === null)
      .map((entry) => entry?.resourceId),
  ].filter((value) => typeof value === 'string'));
  addReasonUnless(Array.isArray(textureViews) && textureViews.length > 0
    && new Set(textureViews.map((record) => record?.viewId)).size === textureViews.length
    && textureViews.every((record) => {
      const parents = textureCreations.filter(
        (texture) => texture?.resourceId === record?.textureId,
      );
      const descriptor = record?.descriptor;
      return Number.isSafeInteger(record?.sequence)
        && typeof record?.viewId === 'string' && record.viewId.length > 0
        && parents.length === 1 && parents[0].sequence < record.sequence
        && (descriptor === null || (isPlainJsonObject(descriptor)
          && exactArray(Object.keys(descriptor), [
            'label', 'format', 'dimension', 'aspect', 'baseMipLevel', 'mipLevelCount',
            'baseArrayLayer', 'arrayLayerCount',
          ])
          && descriptor.aspect === 'all'
          && descriptor.baseMipLevel === 0 && descriptor.baseArrayLayer === 0))
        && usedTextureViewIds.has(record.viewId);
    }),
  reasons, 'GPU texture-view creation/parent/use closure is invalid');

  const commandEncoders = lifecycle?.commandEncoderTraces ?? [];
  const commandTransfers = lifecycle?.commandEncoderTransferCalls ?? [];
  const queueSubmissions = lifecycle?.queueSubmissions ?? [];
  const queueExternalCopies = lifecycle?.queueCopyExternalImageToTextureCalls ?? [];
  const bufferIds = new Set((creations ?? []).filter(
    (record) => record?.method === 'createBuffer',
  ).map((record) => record.resourceId));
  const textureIds = new Set((creations ?? []).filter(
    (record) => record?.method === 'createTexture',
  ).map((record) => record.resourceId));
  const transferResourceIdsExact = (record) => {
    if (record?.method === 'copyBufferToBuffer') {
      return bufferIds.has(record.sourceBufferId) && bufferIds.has(record.destinationBufferId)
        && Number.isSafeInteger(record.sourceOffset) && record.sourceOffset >= 0
        && Number.isSafeInteger(record.destinationOffset) && record.destinationOffset >= 0
        && Number.isSafeInteger(record.size) && record.size > 0;
    }
    if (record?.method === 'clearBuffer') {
      return bufferIds.has(record.destinationBufferId)
        && Number.isSafeInteger(record.destinationOffset) && record.destinationOffset >= 0
        && (record.size === null || (Number.isSafeInteger(record.size) && record.size > 0));
    }
    if (record?.method === 'copyBufferToTexture') {
      return bufferIds.has(record?.source?.bufferId)
        && textureIds.has(record?.destination?.textureId);
    }
    if (record?.method === 'copyTextureToBuffer') {
      return textureIds.has(record?.source?.textureId)
        && bufferIds.has(record?.destination?.bufferId);
    }
    if (record?.method === 'copyTextureToTexture') {
      return textureIds.has(record?.source?.textureId)
        && textureIds.has(record?.destination?.textureId);
    }
    return false;
  };
  const submittedIds = queueSubmissions.flatMap((record) => record?.commandBufferIds ?? []);
  addReasonUnless(Array.isArray(commandEncoders) && commandEncoders.length > 0
    && Array.isArray(commandTransfers) && Array.isArray(queueSubmissions)
    && Array.isArray(queueExternalCopies) && queueExternalCopies.length === 0
    && new Set(commandEncoders.map((record) => record?.commandEncoderId)).size
      === commandEncoders.length
    && new Set(commandEncoders.map((record) => record?.commandBufferId)).size
      === commandEncoders.length
    && commandTransfers.every((record) => Number.isSafeInteger(record?.sequence)
      && typeof record?.commandEncoderId === 'string'
      && typeof record?.capturePhase === 'string' && record.capturePhase.length > 0
      && transferResourceIdsExact(record))
    && commandEncoders.every((encoder) => {
      const transfers = commandTransfers.filter(
        (record) => record?.commandEncoderId === encoder?.commandEncoderId,
      );
      const renderPasses = lifecycle?.renderPassTraces?.filter(
        (trace) => trace?.commandEncoderId === encoder?.commandEncoderId,
      ) ?? [];
      const computePasses = lifecycle?.computePassTraces?.filter(
        (trace) => trace?.commandEncoderId === encoder?.commandEncoderId,
      ) ?? [];
      const submissions = queueSubmissions.filter(
        (record) => record?.commandBufferIds?.includes(encoder?.commandBufferId),
      );
      return Number.isSafeInteger(encoder?.sequence)
        && typeof encoder?.capturePhase === 'string' && encoder.capturePhase.length > 0
        && typeof encoder?.commandEncoderId === 'string'
        && encoder?.finishCallCount === 1
        && Number.isSafeInteger(encoder?.finishSequence)
        && encoder.finishSequence > encoder.sequence
        && typeof encoder?.commandBufferId === 'string'
        && encoder?.finishDescriptor === null
        && exactArray(encoder?.renderPassEncoderIds,
          renderPasses.map((trace) => trace.encoderId))
        && exactArray(encoder?.computePassEncoderIds,
          computePasses.map((trace) => trace.encoderId))
        && exactArray(encoder?.transferSequences,
          transfers.map((record) => record.sequence))
        && renderPasses.length + computePasses.length + transfers.length > 0
        && [...renderPasses, ...computePasses, ...transfers].every(
          (record) => record.sequence > encoder.sequence
            && record.sequence < encoder.finishSequence
            && record.capturePhase === encoder.capturePhase,
        )
        && submissions.length === 1
        && submissions[0]?.capturePhase === encoder.capturePhase
        && exactArray(submissions[0]?.commandBufferIds, [encoder.commandBufferId])
        && submissions[0]?.sequence > encoder.finishSequence;
    })
    && queueSubmissions.length === commandEncoders.length
    && queueSubmissions.every((record) => Number.isSafeInteger(record?.sequence)
      && typeof record?.capturePhase === 'string' && record.capturePhase.length > 0
      && Array.isArray(record?.commandBufferIds) && record.commandBufferIds.length === 1)
    && submittedIds.length === commandEncoders.length
    && new Set(submittedIds).size === submittedIds.length
    && exactArray([...submittedIds].sort(),
      commandEncoders.map((record) => record.commandBufferId).sort()),
  reasons, 'command encoder/transfer/finish/queue-submit execution closure is invalid');

  const globalCommandAudit = validateThreeImmediatePhase0GlobalCommandClosure({
    commandEncoders,
    renderPasses: lifecycle?.renderPassTraces ?? [],
    computePasses: lifecycle?.computePassTraces ?? [],
    transfers: commandTransfers,
    queueSubmissions,
    stagingCreations,
    bufferMapEvents: lifecycle?.bufferMapEvents ?? [],
    bufferDestructions: destructions ?? [],
    bufferCreations: creations?.filter((record) => record?.method === 'createBuffer') ?? [],
    textureCreations: creations?.filter((record) => record?.method === 'createTexture') ?? [],
    queueWriteTextures: lifecycle?.queueWriteTextureCalls ?? [],
    queueExternalCopies,
  });
  addReasonUnless(globalCommandAudit.valid,
    reasons, `global command/readback execution closure is invalid: ${globalCommandAudit.reasons.join('; ')}`);

  const stagingTransferDestinationId = (record) => (
    record?.method === 'copyBufferToBuffer'
      ? record?.destinationBufferId
      : record?.method === 'copyTextureToBuffer'
        ? record?.destination?.bufferId ?? null
        : null
  );
  const reconstructedTransferLedger = stagingCreations.map((resource) => {
    const transfer = commandTransfers.find(
      (record) => stagingTransferDestinationId(record) === resource.resourceId,
    ) ?? null;
    const encoder = commandEncoders.find(
      (record) => record?.commandEncoderId === transfer?.commandEncoderId,
    ) ?? null;
    const submission = queueSubmissions.find(
      (record) => exactArray(record?.commandBufferIds, [encoder?.commandBufferId]),
    ) ?? null;
    const mapEvents = (lifecycle?.bufferMapEvents ?? []).filter(
      (record) => record?.resourceId === resource.resourceId,
    );
    const retainedLedgerRecord = (staging?.transferLedger ?? []).find(
      (record) => record?.stagingBufferId === resource.resourceId,
    );
    return {
      pass: true,
      capturePhase: resource.capturePhase,
      stagingBufferId: resource.resourceId,
      stagingSize: resource.size,
      stagingUsage: resource.usage,
      stagingMappedAtCreation: resource.mappedAtCreation,
      createSequence: resource.sequence,
      transfer,
      sourceBinding: retainedLedgerRecord?.sourceBinding ?? null,
      commandEncoder: encoder,
      submission,
      mapEvents,
      destroySequence: resource.destroySequence,
      destroyCapturePhase: resource.destroyCapturePhase,
      destroyCallCount: resource.destroyCallCount,
      transferShapeExact: true,
    };
  });
  const transferMethods = [
    'copyBufferToBuffer', 'copyTextureToBuffer', 'clearBuffer',
    'copyBufferToTexture', 'copyTextureToTexture', 'resolveQuerySet', 'writeTimestamp',
  ];
  const reconstructedTransferMethodCounts = Object.fromEntries(transferMethods.map(
    (method) => [method, commandTransfers.filter((record) => record?.method === method).length],
  ));
  const bufferMapAudit = validateThreeImmediatePhase0BufferMapLifecycle(
    resources?.bufferMapLifecycle,
    { lifecycle, resources, stagingTransferLedger: reconstructedTransferLedger },
  );
  addReasonUnless(bufferMapAudit.valid, reasons,
    `buffer map lifecycle is invalid: ${bufferMapAudit.reasons.join('; ')}`);
  const readbackSourceAudit = validateThreeImmediatePhase0ReadbackSourceBijection(
    evidence,
    {
      stagingCreations,
      transfers: commandTransfers,
      transferLedger: reconstructedTransferLedger,
    },
  );
  addReasonUnless(readbackSourceAudit.valid, reasons,
    `readback source bijection is invalid: ${readbackSourceAudit.reasons.join('; ')}`);
  addReasonUnless(staging?.transferCount === commandTransfers.length
    && staging?.expectedTransferCount === 825
    && exactJson(staging?.transferMethodCounts, reconstructedTransferMethodCounts)
    && reconstructedTransferMethodCounts.clearBuffer === 0
    && reconstructedTransferMethodCounts.copyBufferToTexture === 0
    && reconstructedTransferMethodCounts.copyTextureToTexture === 0
    && reconstructedTransferMethodCounts.resolveQuerySet === 0
    && reconstructedTransferMethodCounts.writeTimestamp === 0
    && staging?.expectedCommandEncoderCount === 1_629
    && staging?.commandEncoderCount === commandEncoders.length
    && exactJson(staging?.commandEncoderClassCounts, {
      renderOnly: 798, computeOnly: 6, transferOnly: 825,
    })
    && staging?.expectedQueueSubmissionCount === 1_629
    && staging?.queueSubmissionCount === queueSubmissions.length
    && staging?.globalQueueWriteTextureCount === 0
    && staging?.globalCopyExternalImageToTextureCount === 0
    && staging?.commandLedgerPass === true
    && exactJson(staging?.transferLedger, reconstructedTransferLedger),
  reasons, 'page staging command ledger does not equal the independent raw reconstruction');

  const deviceInstrumentationAudit = validateThreeImmediatePhase0DeviceInstrumentationClosure({
    patchedMethods: lifecycle?.patchedDeviceMethods,
    gpuCreations: creations ?? [],
    queueWriteTextures: lifecycle?.queueWriteTextureCalls ?? [],
    queueExternalCopies,
  });
  const traceArrays = [
    lifecycle?.renderBundleTraces, lifecycle?.renderPassTraces,
    lifecycle?.computePassTraces,
  ];
  const encoderStateAudit = validateThreeImmediatePhase0EncoderState(
    resources?.encoderState,
    {
      renderBundles: lifecycle?.renderBundleTraces ?? [],
      renderPasses: lifecycle?.renderPassTraces ?? [],
      computePasses: lifecycle?.computePassTraces ?? [],
    },
  );
  const allTraceIds = traceArrays.flatMap((records) => (
    Array.isArray(records) ? records.map((record) => record?.encoderId) : []
  ));
  addReasonUnless(deviceInstrumentationAudit.valid && encoderStateAudit.valid
    && lifecycle?.renderBundleEncoderCount === lifecycle?.renderBundleTraces?.length
    && lifecycle?.renderPassEncoderCount === lifecycle?.renderPassTraces?.length
    && lifecycle?.computePassEncoderCount === lifecycle?.computePassTraces?.length
    && lifecycle?.computePassEncoderCount === 6
    && traceArrays.every(Array.isArray)
    && allTraceIds.every((id) => typeof id === 'string')
    && new Set(allTraceIds).size === allTraceIds.length
    && traceArrays.every((records) => records.every((trace) => (
      Number.isSafeInteger(trace?.sequence)
        && Array.isArray(trace?.events)
        && trace.events.every((event, index) => Number.isSafeInteger(event?.sequence)
          && (index === 0 || event.sequence > trace.events[index - 1].sequence))
    )))
    && Array.isArray(lifecycle?.instrumentationMarkers)
    && lifecycle.instrumentationMarkers.length === 255
    && lifecycle.instrumentationMarkers.every((marker, index) => (
      Number.isSafeInteger(marker?.sequence)
        && (index === 0
          || marker.sequence > lifecycle.instrumentationMarkers[index - 1].sequence)
    ))
    && exactJson(Object.fromEntries([
      ['phase0-persistent-resource-freeze', 16],
      ['phase0-merged-geometry-realization-start', 1],
      ['phase0-merged-geometry-attribute-update-start', 5],
      ['phase0-merged-geometry-attribute-update-complete', 5],
      ['phase0-merged-geometry-realization-queue-complete', 1],
      ['phase0-schedule-install-start', 8],
      ['phase0-attribute-manager-update-start', 32],
      ['phase0-attribute-manager-update-complete', 32],
      ['phase0-schedule-install-queue-complete', 8],
      ['phase0-command-preflight-readback-start', 24],
      ['phase0-command-preflight-readback-complete', 24],
      ['phase0-schedule-install-complete', 8],
      ['phase0-shared-gpu-commitment-start', 2],
      ['phase0-scenario-matrix-realization-start', 2],
      ['phase0-scenario-matrix-update-complete', 2],
      ['phase0-scenario-matrix-queue-complete', 2],
      ['phase0-shared-attribute-update-start', 16],
      ['phase0-shared-attribute-update-complete', 16],
      ['phase0-shared-attribute-updates-queue-complete', 2],
      ['phase0-shared-gpu-readback-start', 16],
      ['phase0-shared-gpu-readback-complete', 16],
      ['phase0-shared-gpu-commitment-complete', 2],
      ['phase0-immediate-source-detached', 2],
      ['phase0-ordered-production-challenge-start', 6],
      ['phase0-ordered-production-challenge-complete', 6],
      ['phase0-cleanup-start', 1],
    ].map(([kind, count]) => [kind,
      lifecycle.instrumentationMarkers.filter((marker) => marker?.kind === kind).length,
    ])), Object.fromEntries([
      ['phase0-persistent-resource-freeze', 16],
      ['phase0-merged-geometry-realization-start', 1],
      ['phase0-merged-geometry-attribute-update-start', 5],
      ['phase0-merged-geometry-attribute-update-complete', 5],
      ['phase0-merged-geometry-realization-queue-complete', 1],
      ['phase0-schedule-install-start', 8],
      ['phase0-attribute-manager-update-start', 32],
      ['phase0-attribute-manager-update-complete', 32],
      ['phase0-schedule-install-queue-complete', 8],
      ['phase0-command-preflight-readback-start', 24],
      ['phase0-command-preflight-readback-complete', 24],
      ['phase0-schedule-install-complete', 8],
      ['phase0-shared-gpu-commitment-start', 2],
      ['phase0-scenario-matrix-realization-start', 2],
      ['phase0-scenario-matrix-update-complete', 2],
      ['phase0-scenario-matrix-queue-complete', 2],
      ['phase0-shared-attribute-update-start', 16],
      ['phase0-shared-attribute-update-complete', 16],
      ['phase0-shared-attribute-updates-queue-complete', 2],
      ['phase0-shared-gpu-readback-start', 16],
      ['phase0-shared-gpu-readback-complete', 16],
      ['phase0-shared-gpu-commitment-complete', 2],
      ['phase0-immediate-source-detached', 2],
      ['phase0-ordered-production-challenge-start', 6],
      ['phase0-ordered-production-challenge-complete', 6],
      ['phase0-cleanup-start', 1],
    ]))
    && exactJson(lifecycle?.cleanup?.globalFinalUse, globalFinalUse)
    && Number.isSafeInteger(lifecycle?.cleanup?.cleanupStartSequence)
    && lifecycle.cleanup.cleanupStartSequence > globalFinalUseSequence
    && markerExact(lifecycle.instrumentationMarkers,
      lifecycle.cleanup.cleanupStartSequence, 'phase0-cleanup-start', {
        globalFinalUseSequence,
      })
    && lifecycle.instrumentationMarkers.find(
      (marker) => marker?.sequence === lifecycle.cleanup.cleanupStartSequence,
    )?.capturePhase === null
    && exactJson(lifecycle?.scopedErrors,
      RUNNER_NORMAL_ERROR_SCOPE_DRAINAGE.scopedErrors)
    && exactJson(lifecycle?.cleanup?.errorScopeDrainage,
      RUNNER_NORMAL_ERROR_SCOPE_DRAINAGE)
    && lifecycle?.cleanup?.complete === true
    && lifecycle.cleanup?.strategyRootDetached === true
    && lifecycle.cleanup?.sourceGeometryCountDisposed === 32
    && lifecycle.cleanup?.deviceDestroyIntentional === true
    && lifecycle.cleanup?.terminalDeviceLoss?.reason === 'destroyed'
    && Array.isArray(lifecycle.cleanup?.errors) && lifecycle.cleanup.errors.length === 0,
  reasons, `renderer/resource lifecycle is incomplete: ${encoderStateAudit.reasons.join('; ')}`);
}

function validateThreeImmediatePhase0PageResultUnchecked(result, {
  targetKey = 'r185', rawReport, observationNonce,
} = {}) {
  const reasons = [];
  let configuration;
  try { configuration = configurationFor(targetKey, rawReport, observationNonce); } catch (error) {
    reasons.push(`cannot construct target/raw configuration: ${error.message}`);
  }
  addReasonUnless(result?.schemaVersion === 1
    && result?.kind === THREE_IMMEDIATE_PHASE0_PAGE_KIND
    && result?.status === 'technical-canary-complete'
    && result?.executionMode === 'technical-canary'
    && result?.analysisEligible === false && result?.efficacyAnalysisAllowed === false
    && result?.numericalDecision === null && result?.timingCaptured === false
    && result?.cleanupFailure === null
    && result?.failure === null
    && result?.secureContext === true && result?.crossOriginIsolated === true
    && typeof result?.userAgent === 'string' && result.userAgent.length > 0
    && result?.userAgent === rawReport?.pageResult?.userAgent
    && exactJson(result?.errorObservationScope, {
      start: 'after-production-device-creation-before-three-renderer-construction',
      includesRendererInitialization: true,
      includesAllComputeBundleRenderAndReadbackOperations: true,
      excludesAdapterAndDeviceRequest: true,
    }),
  reasons, 'page identity/status or correctness-only boundary is invalid');
  if (configuration) {
    addReasonUnless(exactJson(result?.target, targetIdentity(configuration))
      && result?.scope === configuration.scope
      && exactArray(result?.coverage?.implemented, configuration.implemented)
      && exactArray(result?.coverage?.deferred, configuration.deferred),
    reasons, 'page target identity/coverage is not exact');
    validateCapabilities(result, configuration, rawReport, reasons);
  }
  const forbidden = findForbiddenPhase0PerformanceEvidence(result);
  if (forbidden.length) reasons.push(`page retained performance keys: ${forbidden.join(', ')}`);
  validatePlanEvidence(result?.evidence?.plan, reasons);
  addReasonUnless(result?.evidence?.runtimeSignature?.revision
      === configuration?.expectedRevision
    && result?.evidence?.runtimeSignature?.setIndirectHasImmediateBasesParameter === true
    && result?.evidence?.runtimeSignature?.backendHasSetImmediates === true,
  reasons, 'patched Three runtime signature is not exact');
  validateTopology(result?.evidence?.topology, reasons);
  validateShaderEvidence(result?.evidence?.shaders, reasons);
  validatePipelineEvidence(result?.evidence?.pipelines, result?.evidence?.shaders, reasons);
  validatePipelineCreationCache(result?.evidence, reasons);
  validateComputeExecution(result?.evidence, reasons);
  addReasonUnless(result?.evidence?.commands?.commandByteLength === 640
    && exactArray(result?.evidence?.commands?.indirectOffsets,
      IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets),
  reasons, 'global indirect-command evidence is incomplete');
  validatePhase0ScheduleInstallations(result?.evidence, reasons);
  validateScenarioEvidence(
    result?.evidence?.scenarios,
    result?.evidence?.shaders,
    result?.evidence?.pipelines,
    result?.evidence?.resources,
    result?.evidence?.lifecycle,
    reasons,
  );
  validateRuntimeCoverage(result?.evidence?.runtimeCoverage, reasons);
  reasons.push(...validateThreeImmediatePhase0AddressWitnesses(
    result?.evidence?.addressWitnesses,
  ).reasons);
  reasons.push(...validateThreeImmediatePhase0OutputWitnesses(
    result?.evidence?.outputWitnesses,
    result?.evidence?.scenarios,
  ).reasons);
  reasons.push(...validateThreeImmediatePhase0ObservationChallenge(
    result, observationNonce,
  ).reasons);
  reasons.push(...validateThreeImmediatePhase0GpuByteWitnesses(
    result?.evidence,
  ).reasons);
  validateGpuResourceCommitments(result?.evidence, reasons);
  validateGeometryResourceEvidence(result?.evidence, reasons);
  validateCommonResourceEvidence(result?.evidence, reasons);
  validateRenderPassDescriptorsAndEvents(result?.evidence, reasons);
  validateResourcesAndLifecycle(result?.evidence, reasons);
  reasons.push(...validateThreeImmediatePhase0ErrorScopeDrainage(result).reasons);
  addReasonUnless(Array.isArray(result?.gpuErrors?.uncaptured)
    && result.gpuErrors.uncaptured.length === 0
    && Array.isArray(result?.gpuErrors?.unexpectedDeviceLosses)
    && result.gpuErrors.unexpectedDeviceLosses.length === 0
    && Array.isArray(result?.gpuErrors?.pageErrors)
    && result.gpuErrors.pageErrors.length === 0
    && Array.isArray(result?.gpuErrors?.unhandledRejections)
    && result.gpuErrors.unhandledRejections.length === 0
    && result?.gpuErrors?.finalDeviceLoss?.reason === 'destroyed'
    && exactJson(result?.gpuErrors?.finalDeviceLoss,
      result?.cleanup?.terminalDeviceLoss)
    && exactJson(result?.cleanup, result?.evidence?.lifecycle?.cleanup),
  reasons, 'production device errors/loss lifecycle is invalid');
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function phase0PageFailureReason(result) {
  const status = typeof result?.status === 'string'
    ? result.status.slice(0, 256) : '<missing-status>';
  const failure = result?.failure;
  if (isPlainJsonObject(failure)
    && typeof failure.name === 'string' && failure.name.length > 0
    && typeof failure.message === 'string' && failure.message.length > 0) {
    return `page reported ${status}: ${failure.name.slice(0, 256)}: ${
      failure.message.slice(0, 8_192)}`;
  }
  return `page reported ${status} without one well-formed failure record`;
}

function createPhase0PageValidationError(result, validation) {
  const error = new Error(
    `Phase 0 page failed independent validation: ${validation.reasons.join('; ')}`,
  );
  error.name = 'Phase0PageValidationError';
  error.pageStatus = typeof result?.status === 'string' ? result.status : '<missing-status>';
  if (isPlainJsonObject(result?.failure) && isStrictJsonData(result.failure)) {
    error.pageFailure = structuredClone(result.failure);
  }
  return error;
}

function validateThreeImmediatePhase0PageResultPublicUnchecked(result, options = {}) {
  if (!isPlainJsonObject(result) || !isStrictJsonData(result)) {
    return {
      valid: false,
      reasons: ['Phase 0 page result must be one non-null strict JSON object'],
    };
  }
  if (result.status !== 'technical-canary-complete') {
    return {
      valid: false,
      reasons: [
        'page identity/status or correctness-only boundary is invalid',
        phase0PageFailureReason(result),
      ],
    };
  }
  if (result.cleanupFailure !== null) {
    return {
      valid: false,
      reasons: ['completed Phase 0 page result retained a cleanup failure'],
    };
  }
  if (!isPlainJsonObject(result.evidence)) {
    return {
      valid: false,
      reasons: ['Phase 0 completed page result has absent or non-object evidence'],
    };
  }
  try {
    return validateThreeImmediatePhase0PageResultUnchecked(result, options);
  } catch (error) {
    const failure = serializeError(error);
    return {
      valid: false,
      reasons: [`Phase 0 page validation failed closed: ${failure.name}: ${failure.message}`],
    };
  }
}

export function validateThreeImmediatePhase0PageResult(result, options = {}) {
  try {
    return validateThreeImmediatePhase0PageResultPublicUnchecked(result, options);
  } catch (error) {
    const failure = serializeError(error);
    return {
      valid: false,
      reasons: [`Phase 0 page validation failed closed: ${failure.name}: ${failure.message}`],
    };
  }
}

async function collectBoundSources(configuration) {
  const records = [];
  for (const [role, filename] of BOUND_SOURCE_PATHS) {
    const bytes = await readFile(filename);
    const record = {
      role, path: normalizePath(path.relative(repoRoot, filename)),
      byteLength: bytes.length, sha256: sha256(bytes),
    };
    if (role === 'page-module' && configuration !== null) {
      record.injectedSha256 = sha256(Buffer.from(
        injectedPageSource(bytes.toString('utf8'), configuration),
      ));
    }
    records.push(record);
  }
  return records;
}

function boundSourcesExact(start, end) {
  return Array.isArray(start) && Array.isArray(end)
    && start.length === BOUND_SOURCE_PATHS.length && exactJson(start, end)
    && start.every((record, index) => record?.role === BOUND_SOURCE_PATHS[index][0]
      && record?.path === normalizePath(path.relative(repoRoot, BOUND_SOURCE_PATHS[index][1]))
      && Number.isSafeInteger(record?.byteLength) && record.byteLength > 0
      && SHA256_PATTERN.test(record?.sha256)
      && (record.role !== 'page-module' || SHA256_PATTERN.test(record.injectedSha256)));
}

function unconfiguredBoundSourcesExact(start, end) {
  return Array.isArray(start) && Array.isArray(end)
    && start.length === BOUND_SOURCE_PATHS.length && exactJson(start, end)
    && start.every((record, index) => record?.role === BOUND_SOURCE_PATHS[index][0]
      && record?.path === normalizePath(path.relative(repoRoot, BOUND_SOURCE_PATHS[index][1]))
      && Number.isSafeInteger(record?.byteLength) && record.byteLength > 0
      && SHA256_PATTERN.test(record?.sha256)
      && record.injectedSha256 === undefined);
}

function configuredBoundMatchesUnconfigured(configured, unconfigured) {
  if (!Array.isArray(configured) || !Array.isArray(unconfigured)
    || configured.length !== unconfigured.length) return false;
  return configured.every((record, index) => {
    const baseline = unconfigured[index];
    return record?.role === baseline?.role && record?.path === baseline?.path
      && record?.byteLength === baseline?.byteLength && record?.sha256 === baseline?.sha256
      && (record.role === 'page-module'
        ? SHA256_PATTERN.test(record?.injectedSha256 ?? '')
        : record?.injectedSha256 === undefined);
  });
}

function executableIdentityExact(left, right) {
  return left?.path === right?.path && left?.byteLength === right?.byteLength
    && left?.sha256 === right?.sha256 && SHA256_PATTERN.test(left?.sha256);
}

function chromeExecutableIdentityExact(left, right) {
  return executableIdentityExact(left, right)
    && left?.canonicalPath === normalizePath(DEFAULT_CHROME_PATH)
    && right?.canonicalPath === normalizePath(DEFAULT_CHROME_PATH)
    && left?.path === left.canonicalPath && right?.path === right.canonicalPath
    && left?.requestedPath === left.canonicalPath
    && right?.requestedPath === right.canonicalPath
    && left?.byteLength === EXPECTED_CHROME_BYTE_LENGTH
    && left?.sha256 === EXPECTED_CHROME_SHA256
    && exactJson(left?.version, right?.version)
    && left?.version?.fileVersion === EXPECTED_CHROME_PRODUCT_VERSION
    && left?.version?.productVersion === EXPECTED_CHROME_PRODUCT_VERSION
    && /^chrome\.exe$/iu.test(left?.version?.originalFilename ?? '')
    && /^google llc$/iu.test(left?.version?.companyName ?? '')
    && /google chrome/iu.test(left?.version?.productName ?? '');
}

function validStaticEnvironmentIdentity(record) {
  const telemetry = record?.gpuStaticTelemetryIdentity;
  const device = telemetry?.devices?.[0];
  return record?.schemaVersion === 1
    && record?.os?.platform === 'win32'
    && record?.os?.type === 'Windows_NT'
    && typeof record?.os?.release === 'string' && record.os.release.length > 0
    && typeof record?.os?.version === 'string' && record.os.version.length > 0
    && record?.os?.arch === 'x64'
    && telemetry?.command === normalizePath(NVIDIA_SMI_PATH)
    && exactArray(telemetry?.arguments, NVIDIA_SMI_ARGUMENTS)
    && telemetry?.executable?.path === normalizePath(NVIDIA_SMI_PATH)
    && Number.isSafeInteger(telemetry?.executable?.byteLength)
    && telemetry.executable.byteLength > 0
    && SHA256_PATTERN.test(telemetry?.executable?.sha256 ?? '')
    && telemetry?.devices?.length === 1
    && device?.index === 0
    && /^GPU-[0-9a-f-]+$/iu.test(device?.uuid ?? '')
    && typeof device?.name === 'string' && device.name.length > 0
    && /^\d+(?:\.\d+)+$/u.test(device?.driverVersion ?? '')
    && typeof device?.vbiosVersion === 'string' && device.vbiosVersion.length > 0
    && /^[0-9a-f:.]+$/iu.test(device?.pciBusId ?? '');
}

function browserDistributionExact(left, right) {
  return left?.schemaVersion === 1 && right?.schemaVersion === 1
    && left?.root === right?.root && left?.fileCount === right?.fileCount
    && left?.totalBytes === right?.totalBytes
    && left?.inventorySha256 === right?.inventorySha256
    && exactJson(left?.files, right?.files);
}

function validDirectoryInventory(inventory) {
  const files = inventory?.files;
  return inventory?.schemaVersion === 1
    && typeof inventory?.root === 'string' && path.isAbsolute(inventory.root)
    && Array.isArray(files) && files.length > 0
    && inventory.fileCount === files.length
    && inventory.totalBytes === files.reduce((sum, file) => sum + file.byteLength, 0)
    && exactArray(files.map((file) => file.path),
      files.map((file) => file.path).sort((left, right) => left.localeCompare(right, 'en')))
    && new Set(files.map((file) => file.path)).size === files.length
    && files.every((file) => typeof file.path === 'string' && file.path.length > 0
      && !path.isAbsolute(file.path) && !file.path.split('/').includes('..')
      && Number.isSafeInteger(file.byteLength) && file.byteLength >= 0
      && SHA256_PATTERN.test(file.sha256 ?? ''))
    && inventory.inventorySha256 === sha256(Buffer.from(JSON.stringify(files)));
}

function resourceUrlToServedPath(url, source) {
  try {
    const parsed = new URL(url);
    if (parsed.origin !== source.serverOrigin || parsed.hash !== '' || parsed.search !== '') {
      return null;
    }
    const pathname = decodeURIComponent(parsed.pathname);
    if (pathname === '/@vite/client') {
      const matches = source.loadedModules.filter(
        (module) => /repo:\/node_modules\/vite\/dist\/client\/client\.mjs$/u.test(module.servedPath),
      );
      return matches.length === 1 ? matches[0].servedPath : null;
    }
    if (pathname.startsWith('/@fs/')) {
      let filename = pathname.slice('/@fs/'.length);
      if (/^\/[A-Za-z]:\//u.test(filename)) filename = filename.slice(1);
      return servedPathFor(path.resolve(filename), source.overlayStart?.root);
    }
    const candidate = path.resolve(repoRoot, pathname.slice(1));
    return servedPathFor(candidate, source.overlayStart?.root);
  } catch {
    return null;
  }
}

function validateLoadedSourceClosure(source, reasons) {
  const resources = source?.loadedResources;
  const urls = resources?.map((record) => record?.url);
  addReasonUnless(Array.isArray(resources) && resources.length > 0
    && exactArray(source?.loadedResourceUrls, urls)
    && exactArray(urls, [...urls].sort((left, right) => left.localeCompare(right, 'en')))
    && new Set(urls).size === urls.length
    && resources.every((record) => {
      try {
        const parsed = new URL(record.url);
        return parsed.origin === source.serverOrigin
          && parsed.protocol === 'http:' && parsed.hostname === '127.0.0.1'
          && parsed.hash === '' && parsed.search === ''
          && record.status === 200 && record.fromServiceWorker === false
          && ['document', 'script'].includes(record.resourceType)
          && typeof record.contentType === 'string' && record.contentType.length > 0
          && Number.isSafeInteger(record.byteLength) && record.byteLength > 0
          && SHA256_PATTERN.test(record.sha256 ?? '');
      } catch { return false; }
    }),
  reasons, 'loaded resources include an external, cached, failed, or unexpected request');
  const scriptPaths = (resources ?? []).filter((record) => record.resourceType === 'script')
    .map((record) => resourceUrlToServedPath(record.url, source));
  const modulePaths = source?.loadedModules?.map((record) => record.servedPath) ?? [];
  addReasonUnless(scriptPaths.every((value) => typeof value === 'string')
    && exactArray([...scriptPaths].sort(), [...modulePaths].sort())
    && new Set(scriptPaths).size === scriptPaths.length
    && source.loadedModules.every((module) => {
      const response = resources.find((record) => resourceUrlToServedPath(record.url, source)
        === module.servedPath);
      return Array.isArray(module.servedResponses)
        && module.servedResponses.length === 1
        && exactJson(module.servedResponses[0], response === undefined ? null : {
          url: response.url, byteLength: response.byteLength, sha256: response.sha256,
        });
    })
    && resources.filter((record) => record.resourceType === 'document').length === 1,
  reasons, 'loaded JavaScript responses and transformed-module closure are not bijective');
}

export function validateThreeImmediatePhase0LoadedSourceClosure(source) {
  const reasons = [];
  validateLoadedSourceClosure(source, reasons);
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function dependencyClosureFromLoadedModules(loadedModules, aliases) {
  const modules = Array.isArray(loadedModules) ? loadedModules : [];
  const aliasValues = Object.values(aliases ?? {});
  const duplicateModuleCount = modules.filter((record) => record?.transformCount !== 1).length;
  const unpatchedThreeModuleCount = modules.filter(
    (record) => record?.servedPath?.startsWith('repo:/node_modules/three/'),
  ).length;
  const externalModuleCount = modules.filter(
    (record) => record?.servedPath?.startsWith('external:/'),
  ).length;
  const aliasModulesPresent = aliasValues.length === 4 && aliasValues.every((alias) => (
    modules.some((record) => record.servedPath === alias.servedPath
      && record.inputSha256 === alias.sha256)
  ));
  return {
    exact: duplicateModuleCount === 0 && unpatchedThreeModuleCount === 0
      && externalModuleCount === 0 && aliasModulesPresent,
    loadedModuleCount: modules.length,
    duplicateModuleCount,
    unpatchedThreeModuleCount,
    externalModuleCount,
    aliasModulesPresent,
  };
}

export function reconstructThreeImmediatePhase0DependencyClosure(loadedModules, aliases) {
  return dependencyClosureFromLoadedModules(loadedModules, aliases);
}

function loadedModulesBoundToFrozenInventories(source) {
  const lookup = (inventory, relativePath) => inventory?.files?.find(
    (record) => record.path === relativePath,
  );
  return Array.isArray(source?.loadedModules) && source.loadedModules.length > 0
    && source.loadedModules.every((module) => {
      let frozen;
      if (module?.servedPath?.startsWith('overlay:/')) {
        frozen = lookup(source.overlayStart, module.servedPath.slice('overlay:/'.length));
      } else if (module?.servedPath?.startsWith('repo:/node_modules/')) {
        if (module.servedPath.startsWith('repo:/node_modules/.vite/')) return false;
        frozen = lookup(
          source.freezeStart?.installedDependencies,
          module.servedPath.slice('repo:/node_modules/'.length),
        );
      } else if (module?.servedPath?.startsWith('repo:/')) {
        frozen = lookup(
          source.freezeStart?.trackedSources, module.servedPath.slice('repo:/'.length),
        );
      } else return false;
      return frozen?.byteLength > 0 && frozen.byteLength === module?.inputByteLength
        && frozen.sha256 === module?.inputSha256;
    });
}

function diagnosticsClean(value) {
  return Array.isArray(value?.events) && value.events.length === 0
    && value?.overflowCount === 0;
}

function validateSourceIdentity(source, configuration, manifest, reasons) {
  addReasonUnless(boundSourcesExact(source?.boundStart, source?.boundEnd)
    && source?.boundExact === true,
  reasons, 'bound runner/page/runtime sources changed or are malformed');
  const pageRecord = source?.boundStart?.find((record) => record?.role === 'page-module');
  const pageModule = source?.loadedModules?.find(
    (record) => record?.servedPath === 'repo:/src/phase0/immediate-aif-page.js',
  );
  addReasonUnless(pageModule?.transformCount === 1
    && pageModule?.inputSha256 === pageRecord?.sha256
    && pageModule?.outputSha256 === pageRecord?.injectedSha256
    && pageModule?.injected === true
    && source?.configurationSha256 === sha256(Buffer.from(JSON.stringify(configuration))),
  reasons, 'served Phase 0 page is not the injected bound source');
  const expectedAliasKeys = ['three', 'three/core-runtime', 'three/tsl', 'three/webgpu'];
  addReasonUnless(exactArray(Object.keys(source?.aliases ?? {}).sort(), expectedAliasKeys)
    && exactJson(source?.dependencyClosure,
      dependencyClosureFromLoadedModules(source?.loadedModules, source?.aliases))
    && source?.dependencyClosure?.exact === true
    && Array.isArray(source?.loadedModules)
    && source.loadedModules.every((record) => Number.isSafeInteger(record?.inputByteLength)
      && record.inputByteLength > 0 && Number.isSafeInteger(record?.outputByteLength)
      && record.outputByteLength > 0 && SHA256_PATTERN.test(record?.inputSha256)
      && SHA256_PATTERN.test(record?.outputSha256) && record.transformCount === 1)
    && loadedModulesBoundToFrozenInventories(source),
  reasons, 'served dependency closure is not exact');
  const freezeStart = source?.freezeStart;
  const freezeEnd = source?.freezeEnd;
  addReasonUnless(source?.freezeExact === true
    && provenanceProjectionMatches(freezeStart?.provenance, freezeEnd?.provenance)
    && unconfiguredBoundSourcesExact(
      freezeStart?.unconfiguredBound, freezeEnd?.unconfiguredBound,
    )
    && validDirectoryInventory(freezeStart?.trackedSources)
    && validDirectoryInventory(freezeEnd?.trackedSources)
    && directoryInventoriesExact(freezeStart?.trackedSources, freezeEnd?.trackedSources)
    && exactArray(freezeStart?.installedDependencyExclusions, ['.vite'])
    && exactArray(freezeEnd?.installedDependencyExclusions, ['.vite'])
    && validDirectoryInventory(freezeStart?.installedDependencies)
    && validDirectoryInventory(freezeEnd?.installedDependencies)
    && directoryInventoriesExact(
      freezeStart?.installedDependencies, freezeEnd?.installedDependencies,
    )
    && validDirectoryInventory(freezeStart?.targetSource)
    && validDirectoryInventory(freezeEnd?.targetSource)
    && directoryInventoriesExact(freezeStart?.targetSource, freezeEnd?.targetSource),
  reasons, 'clean Git/tracked/package-lock/full dependency source freeze is invalid');
  addReasonUnless(configuredBoundMatchesUnconfigured(
    source?.boundStart, freezeStart?.unconfiguredBound,
  ) && configuredBoundMatchesUnconfigured(
    source?.boundEnd, freezeEnd?.unconfiguredBound,
  ), reasons, 'configured bound-source capture does not match the frozen source inputs');
  addReasonUnless(source?.overlayExact === true
    && validDirectoryInventory(source?.overlayStart)
    && validDirectoryInventory(source?.overlayEnd)
    && directoryInventoriesExact(source?.overlayStart, source?.overlayEnd)
    && source?.overlayIntegration?.exact === true
    && exactJson(source.overlayIntegration, buildOverlayIntegrationEvidence(
      freezeStart?.targetSource, source.overlayStart, manifest, configuration.key,
    )),
  reasons, 'overlay start/end inventory is invalid');
  const expectedServerConfiguration = serverConfigurationEvidence(
    resolveTarget(configuration.key).outputRoot, configuration.observationNonce,
  );
  addReasonUnless(exactJson(source?.serverConfiguration, expectedServerConfiguration)
    && typeof source?.serverOrigin === 'string'
    && /^http:\/\/127\.0\.0\.1:\d+$/u.test(source.serverOrigin),
  reasons, 'Vite alias/header/cache/service-worker configuration is not exact');
  validateLoadedSourceClosure(source, reasons);
  for (const specifier of ['three', 'three/webgpu', 'three/tsl', 'three/core-runtime']) {
    const alias = source?.aliases?.[specifier];
    addReasonUnless(typeof alias?.servedPath === 'string'
      && alias.servedPath.startsWith('overlay:/') && SHA256_PATTERN.test(alias?.sha256)
      && source.loadedModules?.some((record) => record.servedPath === alias.servedPath
        && record.inputSha256 === alias.sha256),
    reasons, `served ${specifier} is not bound to the generated overlay`);
  }
}

function validateThreeImmediatePhase0ReportUnchecked(report) {
  const reasons = [];
  const targetKey = report?.target?.key;
  let configuration;
  try {
    configuration = configurationFor(
      targetKey, report?.raw?.report, report?.observationNonce,
    );
  } catch (error) {
    reasons.push(`report target/raw identity is invalid: ${error.message}`);
  }
  addReasonUnless(report?.schemaVersion === 1
    && report?.kind === THREE_IMMEDIATE_PHASE0_RUNNER_KIND
    && report?.status === 'technical-canary-pass'
    && report?.executionMode === 'technical-canary'
    && report?.analysisEligible === false && report?.efficacyAnalysisAllowed === false
    && report?.numericalDecision === null && report?.timingCaptured === false
    && report?.efficacyEvaluated === false && report?.fullPhase0Pass === true
    && report?.failure === null
    && OBSERVATION_NONCE_PATTERN.test(report?.observationNonce ?? ''),
  reasons, 'runner identity/status or correctness-only boundary is invalid');
  addReasonUnless(report?.executionPolicy?.attemptCount === 1
    && report.executionPolicy.rawAttemptCount === 1
    && report.executionPolicy.integrationAttemptCount === 1
    && report.executionPolicy.retryCount === 0
    && report.executionPolicy.replacementAllowed === false
    && report.executionPolicy.developerFlagFallbackAllowed === false
    && exactArray(report.executionPolicy.launchArguments, [])
    && report.executionPolicy.timingCaptured === false
    && report.executionPolicy.efficacyEvaluated === false,
  reasons, 'one-shot no-fallback policy is invalid');
  let rawValidation = { pass: false, reasons: ['raw report unavailable'] };
  try { rawValidation = validateWebGpuImmediatesProbeReport(report?.raw?.report); } catch (error) {
    rawValidation = { pass: false, reasons: [error.message] };
  }
  addReasonUnless(rawValidation.pass === true
    && report?.raw?.report?.exposureMode === 'default-web-platform'
    && report?.raw?.artifact?.relativePath?.endsWith('/report.json')
    && Number.isSafeInteger(report?.raw?.artifact?.byteLength)
    && SHA256_PATTERN.test(report?.raw?.artifact?.sha256),
  reasons, `embedded raw canary is invalid: ${rawValidation.reasons.join('; ')}`);
  addReasonUnless(validateRawCorrectnessWitness(
    report?.raw?.correctnessWitness, report?.raw?.report,
  ), reasons, 'raw WebGPU address/output witnesses are not exact');
  const browser = report?.integration?.browser;
  const exposure = assessDefaultExposureBrowserRecord(browser);
  addReasonUnless(exposure.pass === true
    && browser?.launchApi === 'playwright-core.chromium.launch'
    && browser?.headless === true && browser?.persistentContext === false
    && browser?.profilePolicy === 'fresh-playwright-temporary-profile'
    && exactArray(browser?.launchArguments, [])
    && browser?.singleContext === true && browser?.singlePage === true
    && browser?.closedCleanly === true
    && browser?.cacheDisabledProtocol === true
    && browser?.serviceWorkers === 'block'
    && browser?.serviceWorkerRegistrationCount === 0
    && browser?.browserVersion === report?.raw?.report?.browser?.browserVersion
    && browserGpuMatchesStaticIdentity(
      browser?.gpuSystemInfo, report?.environmentIdentity?.start,
    )
    && chromeExecutableIdentityExact(browser?.executableStart, browser?.executableEnd)
    && executableIdentityExact(browser?.executableStart,
      report?.raw?.report?.browser?.executableStart)
    && validDirectoryInventory(browser?.distributionStart)
    && validDirectoryInventory(browser?.distributionEnd)
    && browserDistributionExact(browser?.distributionStart, browser?.distributionEnd)
    && browser?.userAgent === report?.integration?.pageResult?.userAgent
    && browser.userAgent === report?.raw?.report?.pageResult?.userAgent,
  reasons, `integration browser/default exposure is invalid: ${exposure.reasons.join('; ')}`);
  addReasonUnless(report?.integration?.server?.localOnly === true
    && report.integration.server.closedCleanly === true
    && report.integration.server.origin === report?.sourceIdentity?.serverOrigin
    && report.integration.server.actualSocket?.address === '127.0.0.1'
    && report.integration.server.actualSocket?.family === 'IPv4'
    && Number.isSafeInteger(report.integration.server.actualSocket?.port)
    && report.integration.server.actualSocket.port > 0
    && report.integration.server.origin
      === `http://127.0.0.1:${report.integration.server.actualSocket.port}`
    && report.integration.server.responseCount
      === report?.sourceIdentity?.loadedResources?.length
    && validViteEffectiveConfiguration(
      report.integration.server.effectiveConfiguration,
      report.sourceIdentity.serverConfiguration,
    )
    && exactJson(
      (({
        origin, actualSocket, responseCount, effectiveConfiguration,
        closedCleanly, ...configuration
      }) => configuration)(
        report?.integration?.server ?? {},
      ),
      report?.sourceIdentity?.serverConfiguration,
    )
    && diagnosticsClean(report?.integration?.diagnostics),
  reasons, 'server/browser diagnostics are not clean');
  const shutdown = report?.integration?.shutdown;
  addReasonUnless(shutdown?.schemaVersion === 1 && shutdown?.complete === true
    && ['context', 'browser', 'server'].every((name) => {
      const record = shutdown?.[name];
      return record?.present === true && record?.attempted === true
        && record?.succeeded === true && record?.error === null;
    }),
  reasons, 'integration context/browser/server cleanup was not fully attempted and successful');
  const environment = report?.environmentIdentity;
  addReasonUnless(environment?.exact === true
    && exactJson(environment?.start, environment?.end)
    && validStaticEnvironmentIdentity(environment?.start)
    && adapterMatchesStaticGpu(
      report?.integration?.pageResult?.adapterInfo, environment?.start,
      report?.integration?.browser?.gpuSystemInfo,
    ),
  reasons, 'static Windows/NVIDIA environment identity is invalid or does not match WebGPU');
  if (configuration) {
    addReasonUnless(exactJson(report?.target, targetIdentity(configuration)),
      reasons, 'report target identity is invalid');
    const targetForManifest = targetKey === 'dev'
      ? THREE_IMMEDIATE_OVERLAY_TARGET_DEV : 'r185';
    addReasonUnless(report?.overlay?.validation?.valid === true
      && validateThreeImmediateOverlayManifest(
        report?.overlay?.manifest, targetForManifest,
      ).valid === true
      && report?.overlay?.manifestSha256
        === sha256(Buffer.from(`${JSON.stringify(report.overlay.manifest, null, 2)}\n`)),
    reasons, 'overlay manifest/source pins are invalid');
    validateSourceIdentity(
      report?.sourceIdentity, configuration, report?.overlay?.manifest, reasons,
    );
  }
  const pageValidation = validateThreeImmediatePhase0PageResult(
    report?.integration?.pageResult, {
      targetKey,
      rawReport: report?.raw?.report,
      observationNonce: report?.observationNonce,
    },
  );
  addReasonUnless(pageValidation.valid === true
    && exactJson(report?.integration?.pageValidation, pageValidation),
  reasons, `page evidence failed reconstruction: ${pageValidation.reasons.join('; ')}`);
  const forbidden = findForbiddenPhase0PerformanceEvidence(report);
  if (forbidden.length) reasons.push(`report retained performance keys: ${forbidden.join(', ')}`);
  return { valid: reasons.length === 0, reasons: [...new Set(reasons)] };
}

function phase0RunnerFailureReason(report) {
  const failure = report?.failure;
  if (isPlainJsonObject(failure)
    && typeof failure.name === 'string' && failure.name.length > 0
    && typeof failure.message === 'string' && failure.message.length > 0) {
    return `runner recorded ${failure.name.slice(0, 256)}: ${failure.message.slice(0, 8_192)}`;
  }
  return 'runner did not retain one well-formed failure record';
}

function validateThreeImmediatePhase0ReportPublicUnchecked(report) {
  if (!isPlainJsonObject(report) || !isStrictJsonData(report)) {
    return {
      valid: false,
      reasons: ['Phase 0 report must be one non-null strict JSON object'],
    };
  }
  const pageResult = report?.integration?.pageResult;
  const pageHasInvalidEnvelope = !isPlainJsonObject(pageResult)
    || !isStrictJsonData(pageResult)
    || pageResult.status !== 'technical-canary-complete'
    || pageResult.cleanupFailure !== null
    || !isPlainJsonObject(pageResult.evidence);
  if (report.status !== 'technical-canary-pass' || pageHasInvalidEnvelope) {
    const pageValidation = validateThreeImmediatePhase0PageResult(pageResult, {
      targetKey: report?.target?.key,
      rawReport: report?.raw?.report,
      observationNonce: report?.observationNonce,
    });
    const reasons = ['runner identity/status or correctness-only boundary is invalid'];
    reasons.push(`page evidence failed reconstruction: ${pageValidation.reasons.join('; ')}`);
    if (report.status !== 'technical-canary-pass') {
      reasons.push(phase0RunnerFailureReason(report));
    }
    return { valid: false, reasons: [...new Set(reasons)] };
  }
  try {
    return validateThreeImmediatePhase0ReportUnchecked(report);
  } catch (error) {
    const failure = serializeError(error);
    return {
      valid: false,
      reasons: [`Phase 0 report validation failed closed: ${failure.name}: ${failure.message}`],
    };
  }
}

export function validateThreeImmediatePhase0Report(report) {
  try {
    return validateThreeImmediatePhase0ReportPublicUnchecked(report);
  } catch (error) {
    const failure = serializeError(error);
    return {
      valid: false,
      reasons: [`Phase 0 report validation failed closed: ${failure.name}: ${failure.message}`],
    };
  }
}

async function withDeadline(promise, label, milliseconds = BROWSER_OPERATION_TIMEOUT_MS) {
  let handle;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        handle = setTimeout(
          () => reject(new Error(`${label} exceeded its fixed deadline.`)),
          milliseconds,
        );
      }),
    ]);
  } finally {
    clearTimeout(handle);
  }
}

function isPlainJsonObject(value) {
  try {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
      && Object.getPrototypeOf(value) === Object.prototype;
  } catch {
    return false;
  }
}

function sameFilesystemObject(left, right) {
  return left?.isFile() === true && right?.isFile() === true
    && left.size === right.size
    && (left.dev === 0 || right.dev === 0 || left.dev === right.dev)
    && (left.ino === 0 || right.ino === 0 || left.ino === right.ino);
}

async function assertUnaliasedPathWithin(filename, allowedRoot) {
  const resolvedRoot = path.resolve(allowedRoot);
  const resolved = path.resolve(filename);
  if (!withinRoot(resolved, resolvedRoot)) {
    throw new Error(`Artifact path escapes its allowed root: ${resolved}`);
  }
  const components = path.relative(resolvedRoot, resolved).split(path.sep);
  for (let index = 0, current = resolvedRoot; index <= components.length; index += 1) {
    if (index > 0) current = path.join(current, components[index - 1]);
    const metadata = await lstat(current);
    if (metadata.isSymbolicLink()) {
      throw new Error(`Artifact path contains a symbolic link or junction: ${current}`);
    }
    const canonical = await realpath(current);
    if (path.resolve(canonical).toLowerCase() !== path.resolve(current).toLowerCase()) {
      throw new Error(`Artifact path contains an aliased ancestor: ${current}`);
    }
  }
}

async function readRegularFileSecurely(filename, allowedRoot) {
  const resolved = path.resolve(filename);
  await assertUnaliasedPathWithin(resolved, allowedRoot);
  const before = await lstat(resolved);
  if (!before.isFile() || before.isSymbolicLink()) {
    throw new Error(`Expected one regular non-symlink file: ${resolved}`);
  }
  const handle = await open(resolved, 'r');
  let handleMetadata;
  let bytes;
  try {
    handleMetadata = await handle.stat();
    bytes = await handle.readFile();
  } finally {
    await handle.close();
  }
  const after = await lstat(resolved);
  await assertUnaliasedPathWithin(resolved, allowedRoot);
  if (!sameFilesystemObject(before, handleMetadata)
    || !sameFilesystemObject(handleMetadata, after)
    || bytes.length !== after.size) {
    throw new Error(`File identity changed while it was being read: ${resolved}`);
  }
  return { resolved, bytes, metadata: after };
}

async function fileIdentity(filename) {
  const [metadata, bytes] = await Promise.all([stat(filename), readFile(filename)]);
  return {
    path: normalizePath(path.resolve(filename)),
    byteLength: metadata.size,
    sha256: sha256(bytes),
  };
}

async function inventoryDirectory(root, { excludeDirectories = [] } = {}) {
  const resolvedRoot = path.resolve(root);
  const excluded = new Set(excludeDirectories);
  const files = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name, 'en'));
    for (const entry of entries) {
      if (entry.isDirectory() && excluded.has(entry.name)) continue;
      const filename = path.join(directory, entry.name);
      const metadata = await lstat(filename);
      if (metadata.isSymbolicLink()) {
        throw new Error(`Browser distribution contains a symbolic link: ${filename}`);
      }
      if (metadata.isDirectory()) await visit(filename);
      else if (metadata.isFile()) {
        const bytes = await readFile(filename);
        files.push({
          path: normalizePath(path.relative(resolvedRoot, filename)),
          byteLength: bytes.length,
          sha256: sha256(bytes),
        });
      } else throw new Error(`Unsupported browser distribution entry: ${filename}`);
    }
  }
  await visit(resolvedRoot);
  files.sort((left, right) => left.path.localeCompare(right.path, 'en'));
  return {
    schemaVersion: 1,
    root: normalizePath(resolvedRoot),
    fileCount: files.length,
    totalBytes: files.reduce((sum, file) => sum + file.byteLength, 0),
    inventorySha256: sha256(Buffer.from(JSON.stringify(files))),
    files,
  };
}

async function captureTrackedSourceInventory() {
  const { stdout } = await execFileAsync('git', ['ls-files', '-z'], {
    cwd: repoRoot, encoding: 'buffer', windowsHide: true,
    maxBuffer: 8 * 1024 * 1024,
    timeout: STATIC_IDENTITY_TIMEOUT_MS, killSignal: 'SIGKILL',
  });
  const filenames = Buffer.from(stdout).toString('utf8').split('\0').filter(Boolean).sort(
    (left, right) => left.localeCompare(right, 'en'),
  );
  const files = [];
  for (const relativePath of filenames) {
    const resolved = path.resolve(repoRoot, relativePath);
    if (!withinRoot(resolved, repoRoot)) {
      throw new Error(`Tracked path escapes repository: ${relativePath}`);
    }
    const bytes = await readFile(resolved);
    files.push({
      path: normalizePath(relativePath), byteLength: bytes.length, sha256: sha256(bytes),
    });
  }
  return {
    schemaVersion: 1,
    root: normalizePath(repoRoot),
    fileCount: files.length,
    totalBytes: files.reduce((sum, file) => sum + file.byteLength, 0),
    inventorySha256: sha256(Buffer.from(JSON.stringify(files))),
    files,
  };
}

async function captureEffectiveArguments(browser) {
  const session = await withDeadline(
    browser.newBrowserCDPSession(), 'browser CDP session creation',
  );
  try {
    const response = await withDeadline(
      session.send('Browser.getBrowserCommandLine'), 'Browser.getBrowserCommandLine',
    );
    if (!Array.isArray(response?.arguments)
      || response.arguments.some((argument) => typeof argument !== 'string')) {
      throw new Error('Browser.getBrowserCommandLine returned malformed arguments.');
    }
    const record = {
      api: 'Browser.getBrowserCommandLine', available: true,
      arguments: [...response.arguments],
      forbiddenArguments: findForbiddenDefaultExposureArguments(response.arguments),
    };
    if (record.forbiddenArguments.length !== 0) {
      throw new Error(`Effective Chrome argv contains forbidden capability switches: ${record.forbiddenArguments.join(', ')}`);
    }
    return record;
  } finally {
    await withDeadline(session.detach(), 'browser CDP detach');
  }
}

async function captureBrowserGpuSystemInfo(browser) {
  const session = await withDeadline(
    browser.newBrowserCDPSession(), 'browser GPU CDP session creation',
  );
  try {
    const response = await withDeadline(
      session.send('SystemInfo.getInfo'), 'SystemInfo.getInfo',
    );
    if (!response?.gpu || !Array.isArray(response.gpu.devices)
      || response.gpu.devices.length === 0) {
      throw new Error('SystemInfo.getInfo returned no GPU devices.');
    }
    return {
      schemaVersion: 1,
      kind: 'chrome-cdp-static-gpu-system-info',
      modelName: response.modelName ?? '',
      modelVersion: response.modelVersion ?? '',
      gpu: structuredClone(response.gpu),
    };
  } finally {
    await withDeadline(session.detach(), 'browser GPU CDP detach');
  }
}

function createDiagnostics() {
  return { events: [], overflowCount: 0, shutdownStarted: false };
}

function attachDiagnostics(page, diagnostics, loadedResources) {
  const append = (event) => {
    if (diagnostics.shutdownStarted) return;
    if (diagnostics.events.length < 256) diagnostics.events.push(event);
    else diagnostics.overflowCount += 1;
  };
  page.on('pageerror', (error) => append({ type: 'pageerror', error: serializeError(error) }));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      append({ type: 'console-error', text: message.text().slice(0, 8_192) });
    }
  });
  page.on('requestfailed', (request) => append({
    type: 'request-failed', method: request.method(),
    url: request.url().slice(0, 4_096), failure: request.failure(),
  }));
  page.on('response', (response) => {
    const request = response.request();
    const task = withDeadline(
      response.body(), `response-body capture for ${response.url()}`,
    ).then((body) => {
      loadedResources.records.push({
        url: response.url(), status: response.status(),
        resourceType: request.resourceType(),
        fromServiceWorker: response.fromServiceWorker(),
        contentType: response.headers()['content-type'] ?? '',
        byteLength: body.length,
        sha256: sha256(body),
      });
    }).catch((error) => append({
      type: 'response-body-capture-failed', url: response.url(),
      error: serializeError(error),
    }));
    loadedResources.pending.add(task);
    void task.then(() => loadedResources.pending.delete(task));
    if (response.status() >= 400) {
      append({ type: 'http-error', status: response.status(), url: response.url() });
    }
  });
  page.on('crash', () => append({ type: 'page-crash' }));
}

function normalizedModuleId(id) {
  if (id.startsWith('\0') || id.startsWith('virtual:')) return null;
  let value = id.split('?')[0];
  if (value.startsWith('/@fs/')) value = value.slice('/@fs/'.length);
  if (/^\/[A-Za-z]:\//u.test(value)) value = value.slice(1);
  return path.normalize(path.resolve(value));
}

function servedPathFor(filename, overlayRoot) {
  if (filename === overlayRoot) return 'overlay:/';
  if (withinRoot(filename, overlayRoot)) {
    return `overlay:/${normalizePath(path.relative(overlayRoot, filename))}`;
  }
  if (filename === repoRoot) return 'repo:/';
  if (withinRoot(filename, repoRoot)) {
    return `repo:/${normalizePath(path.relative(repoRoot, filename))}`;
  }
  return `external:/${normalizePath(filename)}`;
}

function sourceObserverPlugin({ configuration, overlayRoot, records }) {
  return {
    name: 'three-immediate-phase0-source-closure',
    enforce: 'post',
    transform(code, id) {
      const filename = normalizedModuleId(id);
      if (filename === null) return null;
      const servedPath = servedPathFor(filename, overlayRoot);
      const isPage = filename === path.normalize(PAGE_PATH);
      const transformed = isPage ? injectedPageSource(code, configuration) : code;
      const prior = records.get(servedPath);
      records.set(servedPath, {
        servedPath,
        inputByteLength: Buffer.byteLength(code),
        inputSha256: sha256(Buffer.from(code)),
        outputByteLength: Buffer.byteLength(transformed),
        outputSha256: sha256(Buffer.from(transformed)),
        transformCount: (prior?.transformCount ?? 0) + 1,
        injected: isPage,
      });
      return isPage ? { code: transformed, map: null } : null;
    },
  };
}

async function buildSourceEvidence({
  overlayRoot, configuration, aliases, records, loadedResources, boundStart, boundEnd,
  serverOrigin, serverConfiguration,
}) {
  const responseRecords = [...loadedResources.records].sort(
    (left, right) => left.url.localeCompare(right.url, 'en'),
  );
  const loadedModules = [...records.values()].sort(
    (left, right) => left.servedPath.localeCompare(right.servedPath, 'en'),
  );
  const aliasEvidence = {};
  for (const [specifier, filename] of Object.entries(aliases)) {
    aliasEvidence[specifier] = {
      servedPath: servedPathFor(filename, overlayRoot),
      sha256: sha256(await readFile(filename)),
    };
  }
  const corePath = path.join(overlayRoot, 'build', 'three.core.js');
  aliasEvidence['three/core-runtime'] = {
    servedPath: servedPathFor(corePath, overlayRoot),
    sha256: sha256(await readFile(corePath)),
  };
  const sourceForMapping = {
    serverOrigin, loadedModules, overlayStart: { root: normalizePath(path.resolve(overlayRoot)) },
  };
  for (const module of loadedModules) {
    const responses = responseRecords.filter((record) => record.resourceType === 'script'
      && resourceUrlToServedPath(record.url, sourceForMapping) === module.servedPath);
    module.servedResponses = responses.map((record) => ({
      url: record.url, byteLength: record.byteLength, sha256: record.sha256,
    }));
  }
  return {
    schemaVersion: 1,
    configurationSha256: sha256(Buffer.from(JSON.stringify(configuration))),
    boundStart, boundEnd, boundExact: boundSourcesExact(boundStart, boundEnd),
    aliases: aliasEvidence, loadedModules,
    serverOrigin,
    serverConfiguration,
    loadedResources: responseRecords,
    loadedResourceUrls: responseRecords.map((record) => record.url),
    dependencyClosure: dependencyClosureFromLoadedModules(loadedModules, aliasEvidence),
  };
}

async function writeExclusiveReport(report, target) {
  const parent = path.join(repoRoot, 'results', 'development', target.resultDirectory);
  await mkdir(parent, { recursive: true });
  const directory = path.join(parent, `phase0-${randomBytes(16).toString('hex')}`);
  await mkdir(directory);
  const reportPath = path.join(directory, 'report.json');
  const bytes = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
  await writeFile(reportPath, bytes, { flag: 'wx' });
  const reread = await readRegularFileSecurely(reportPath, repoRoot);
  if (!reread.bytes.equals(bytes)) {
    throw new Error('Just-written Phase 0 report failed its file-handle reread.');
  }
  await chmod(reportPath, 0o444);
  const immutableReread = await readRegularFileSecurely(reportPath, repoRoot);
  if (!immutableReread.bytes.equals(bytes) || (immutableReread.metadata.mode & 0o222) !== 0) {
    throw new Error('Just-written Phase 0 report is not exact and read-only after publication.');
  }
  return {
    reportPath,
    relativePath: normalizePath(path.relative(repoRoot, reportPath)),
    byteLength: bytes.length,
    sha256: sha256(bytes),
  };
}

export async function verifyThreeImmediatePhase0Artifact(reportPath) {
  const resolved = path.resolve(reportPath);
  const reasons = [];
  let bytes = Buffer.alloc(0);
  let metadata = null;
  try {
    ({ bytes, metadata } = await readRegularFileSecurely(resolved, repoRoot));
    if ((metadata.mode & 0o222) !== 0) {
      reasons.push('Phase 0 report is not immutable/read-only');
    }
    const entries = await readdir(path.dirname(resolved), { withFileTypes: true });
    if (entries.length !== 1 || entries[0].name !== 'report.json'
      || !entries[0].isFile() || entries[0].isSymbolicLink()) {
      reasons.push('Phase 0 artifact directory has missing or extra entries');
    }
  } catch (error) {
    reasons.push(`Phase 0 report file identity is invalid: ${error.message}`);
  }
  let report = null;
  try { report = JSON.parse(bytes.toString('utf8')); } catch (error) {
    reasons.push(`Phase 0 report JSON is invalid: ${error.message}`);
  }
  if (!isPlainJsonObject(report)) {
    reasons.push('Phase 0 report JSON root must be one non-null plain object');
  } else {
    try {
      const target = resolveTarget(report?.target?.key);
      const expectedParent = path.resolve(
        repoRoot, 'results', 'development', target.resultDirectory,
      );
      const artifactDirectory = path.dirname(resolved);
      if (path.dirname(artifactDirectory).toLowerCase() !== expectedParent.toLowerCase()
        || !/^phase0-[a-f0-9]{32}$/u.test(path.basename(artifactDirectory))
        || path.basename(resolved) !== 'report.json') {
        reasons.push('Phase 0 report is outside its exact target-specific result tree');
      }
    } catch (error) {
      reasons.push(`Phase 0 artifact target path is invalid: ${error.message}`);
    }
    const canonicalBytes = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
    if (!bytes.equals(canonicalBytes)) {
      reasons.push('Phase 0 report is not the exact canonical pretty-JSON byte encoding');
    }
    reasons.push(...validateThreeImmediatePhase0Report(report).reasons);
    const relative = report?.raw?.artifact?.relativePath;
    if (typeof relative !== 'string') reasons.push('Raw artifact path is absent');
    else {
      const rawPath = path.resolve(repoRoot, relative);
      if (!withinRoot(rawPath, repoRoot)) reasons.push('Raw artifact path escapes repository');
      else {
        try {
          const rawDirectory = path.dirname(rawPath);
          const rawParent = path.resolve(
            repoRoot, 'results', 'development',
            'webgpu-immediates-default-exposure-canary',
          );
          if (path.dirname(rawDirectory).toLowerCase() !== rawParent.toLowerCase()
            || !/^canary-[a-f0-9]{24}$/u.test(path.basename(rawDirectory))
            || path.basename(rawPath) !== 'report.json') {
            reasons.push('Referenced raw artifact is outside its canonical result tree');
          }
          const { bytes: rawBytes, metadata: rawMetadata } =
            await readRegularFileSecurely(rawPath, repoRoot);
          const rawEntries = await readdir(rawDirectory, { withFileTypes: true });
          if (!rawWebGpuImmediatesArtifactModeIsReadOnly(rawMetadata)
            || rawBytes.length !== report.raw.artifact.byteLength
            || sha256(rawBytes) !== report.raw.artifact.sha256
            || rawEntries.length !== 1 || rawEntries[0].name !== 'report.json'
            || !rawEntries[0].isFile() || rawEntries[0].isSymbolicLink()
            || !exactJson(JSON.parse(rawBytes.toString('utf8')), report.raw.report)) {
            reasons.push('Referenced raw artifact is missing, aliased, or changed');
          }
        } catch (error) {
          reasons.push(`Referenced raw artifact cannot be verified: ${error.message}`);
        }
      }
    }
    try {
      const target = resolveTarget(report?.target?.key);
      const [currentSource, currentOverlay, currentEnvironment, currentBrowser,
        currentBrowserDistribution] =
        await Promise.all([
          captureSourceFreezeEnd(target),
          inventoryDirectory(target.outputRoot),
          captureStaticEnvironmentIdentity(),
          exactChromeExecutableIdentity(DEFAULT_CHROME_PATH),
          inventoryDirectory(path.dirname(DEFAULT_CHROME_PATH)),
        ]);
      if (!exactJson(currentSource, report?.sourceIdentity?.freezeEnd)) {
        reasons.push('Current Git/dependency/target-source closure differs from the artifact');
      }
      if (!exactJson(currentOverlay, report?.sourceIdentity?.overlayEnd)) {
        reasons.push('Current generated overlay bytes differ from the artifact');
      }
      if (!exactJson(currentEnvironment, report?.environmentIdentity?.end)) {
        reasons.push('Current OS/NVIDIA static identity differs from the artifact');
      }
      if (!exactJson(currentBrowser, report?.integration?.browser?.executableEnd)) {
        reasons.push('Current canonical Chrome executable identity differs from the artifact');
      }
      if (!exactJson(
        currentBrowserDistribution, report?.integration?.browser?.distributionEnd,
      )) {
        reasons.push('Current canonical Chrome distribution differs from the artifact');
      }
      const configuration = configurationFor(
        report?.target?.key, report?.raw?.report, report?.observationNonce,
      );
      const currentBound = await collectBoundSources(configuration);
      if (!exactJson(currentBound, report?.sourceIdentity?.boundEnd)) {
        reasons.push('Current configured runner/page/proof source hashes differ from the artifact');
      }
    } catch (error) {
      reasons.push(`Current source/environment closure cannot be verified: ${error.message}`);
    }
  }
  return {
    valid: reasons.length === 0, reasons: [...new Set(reasons)],
    artifact: {
      path: normalizePath(resolved), byteLength: bytes.length, sha256: sha256(bytes),
      regularFile: metadata?.isFile() === true && metadata?.isSymbolicLink() === false,
    },
    report,
  };
}

// Frozen on 2026-09-03 before either full Phase 0 target was launched. The
// earlier minimal canaries used Chrome 152.0.7977.66 and remain a separate,
// historical browser stratum.
export const THREE_IMMEDIATE_PHASE0_EXPECTED_CHROME_IDENTITY = Object.freeze({
  canonicalPath: DEFAULT_CHROME_PATH,
  productVersion: '152.0.7977.82',
  byteLength: 4_461_720,
  sha256: '8cd23aec3a30479b9b8db2063e70526c88a7ec2c99cd744603eb26ab598733ed',
});
const EXPECTED_CHROME_PRODUCT_VERSION =
  THREE_IMMEDIATE_PHASE0_EXPECTED_CHROME_IDENTITY.productVersion;
const EXPECTED_CHROME_BYTE_LENGTH =
  THREE_IMMEDIATE_PHASE0_EXPECTED_CHROME_IDENTITY.byteLength;
const EXPECTED_CHROME_SHA256 = THREE_IMMEDIATE_PHASE0_EXPECTED_CHROME_IDENTITY.sha256;

export function parseThreeImmediatePhase0Arguments(arguments_) {
  if (!Array.isArray(arguments_)) throw new TypeError('CLI arguments must be an array.');
  const options = {
    target: null,
    browser: DEFAULT_CHROME_PATH,
    verify: null,
    help: false,
  };
  const seen = new Set();
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === '--help' || argument === '-h') {
      options.help = true;
      continue;
    }
    if (!['--target', '--browser', '--verify'].includes(argument)) {
      throw new Error(`Unknown Phase 0 argument ${JSON.stringify(argument)}.`);
    }
    const value = arguments_[index + 1];
    if (typeof value !== 'string' || value.length === 0 || value.startsWith('--')) {
      throw new Error(`${argument} requires one value.`);
    }
    index += 1;
    const key = argument.slice(2);
    if (seen.has(key)) throw new Error(`${argument} was repeated.`);
    seen.add(key);
    options[key] = value;
  }
  if (options.help) return options;
  if (options.verify !== null) {
    if (arguments_.includes('--browser')) {
      throw new Error('--verify cannot be combined with --browser.');
    }
    if (options.target !== null && !Object.hasOwn(TARGETS, options.target)) {
      throw new Error(`Unknown Phase 0 target ${JSON.stringify(options.target)}.`);
    }
    return options;
  }
  if (options.target === null) throw new Error('--target r185|dev is required.');
  resolveTarget(options.target);
  return options;
}

export function threeImmediatePhase0Help() {
  return [
    'Three.js WebGPU immediate-data Phase 0 correctness runner',
    '',
    'Usage:',
    '  node scripts/probe-three-immediate-phase0.mjs --target r185|dev [--browser PATH]',
    '  node scripts/probe-three-immediate-phase0.mjs [--target r185|dev] --verify REPORT.json',
    '  node scripts/probe-three-immediate-phase0.mjs --help',
    '',
    `Default browser: ${DEFAULT_CHROME_PATH}`,
    '--browser is accepted only when it resolves to that exact pinned path and byte identity.',
    'The run uses one raw and one integration launch, no GPU-enabling flags, no retry,',
    'and captures correctness/identity evidence only. --verify never launches a browser.',
  ].join('\n');
}

function projectedProvenance(value) {
  if (value?.status !== 'available') return value;
  const { capturedAt, captureWindowStartedAt, captureAttempts, ...withoutCaptureTimes } = value;
  return withoutCaptureTimes;
}

function provenanceProjectionMatches(left, right) {
  return exactJson(left, right)
    && left?.schemaVersion === 1 && left?.status === 'available'
    && left?.captureStable === true && left?.dirty === false
    && /^[a-f0-9]{40}$/u.test(left?.commit ?? '')
    && /^[a-f0-9]{40}$/u.test(left?.tree ?? '')
    && SHA256_PATTERN.test(left?.trackedFilesSha256 ?? '')
    && left?.packageLockTracked === true
    && SHA256_PATTERN.test(left?.packageLockSha256 ?? '');
}

async function chromeVersionIdentity(filename) {
  const escaped = filename.replaceAll("'", "''");
  const script = `$value=(Get-Item -LiteralPath '${escaped}').VersionInfo; `
    + `[Console]::Out.Write(($value | Select-Object FileVersion,ProductVersion,OriginalFilename,CompanyName,ProductName | ConvertTo-Json -Compress))`;
  const { stdout } = await execFileAsync(
    'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe',
    ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
    {
      encoding: 'utf8', windowsHide: true, maxBuffer: 64 * 1024,
      timeout: STATIC_IDENTITY_TIMEOUT_MS, killSignal: 'SIGKILL',
    },
  );
  const record = JSON.parse(stdout);
  return {
    fileVersion: String(record.FileVersion ?? ''),
    productVersion: String(record.ProductVersion ?? ''),
    originalFilename: String(record.OriginalFilename ?? ''),
    companyName: String(record.CompanyName ?? ''),
    productName: String(record.ProductName ?? ''),
  };
}

async function exactChromeExecutableIdentity(requestedPath) {
  const requested = path.resolve(requestedPath);
  await access(requested);
  const metadata = await lstat(requested);
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    throw new Error(`Chrome path is not one regular non-symlink file: ${requested}`);
  }
  const canonical = await realpath(requested);
  if (path.resolve(canonical).toLowerCase() !== requested.toLowerCase()) {
    throw new Error(`Chrome path resolves through an alias: ${requested} -> ${canonical}`);
  }
  const [identity, version] = await Promise.all([
    fileIdentity(requested), chromeVersionIdentity(requested),
  ]);
  if (normalizePath(canonical).toLowerCase() !== normalizePath(DEFAULT_CHROME_PATH).toLowerCase()
    || identity.byteLength !== EXPECTED_CHROME_BYTE_LENGTH
    || identity.sha256 !== EXPECTED_CHROME_SHA256
    || version.productVersion !== EXPECTED_CHROME_PRODUCT_VERSION
    || version.fileVersion !== EXPECTED_CHROME_PRODUCT_VERSION
    || version.originalFilename.toLowerCase() !== 'chrome.exe'
    || !/^google llc$/iu.test(version.companyName)) {
    throw new Error(`Chrome executable version identity is outside the frozen protocol: ${JSON.stringify(version)}`);
  }
  return {
    ...identity,
    requestedPath: normalizePath(requested),
    canonicalPath: normalizePath(canonical),
    version,
  };
}

function parseNvidiaIdentity(stdout) {
  const rows = stdout.trim().split(/\r?\n/u).filter(Boolean).map((line) => {
    const columns = line.split(',').map((value) => value.trim());
    if (columns.length !== 6) throw new Error(`Malformed nvidia-smi identity row: ${line}`);
    const [index, uuid, name, driverVersion, vbiosVersion, pciBusId] = columns;
    if (!/^\d+$/u.test(index) || !/^GPU-[0-9a-f-]+$/iu.test(uuid)
      || !name || !/^\d+(?:\.\d+)+$/u.test(driverVersion)
      || !vbiosVersion || !/^[0-9a-f:.]+$/iu.test(pciBusId)) {
      throw new Error(`Invalid nvidia-smi static identity row: ${line}`);
    }
    return { index: Number(index), uuid, name, driverVersion, vbiosVersion, pciBusId };
  });
  if (rows.length === 0 || new Set(rows.map((row) => row.index)).size !== rows.length
    || new Set(rows.map((row) => row.uuid)).size !== rows.length) {
    throw new Error('nvidia-smi did not return a unique nonempty GPU identity set.');
  }
  return rows.sort((left, right) => left.index - right.index);
}

async function captureStaticEnvironmentIdentity() {
  const smiIdentity = await fileIdentity(NVIDIA_SMI_PATH);
  const { stdout, stderr } = await execFileAsync(NVIDIA_SMI_PATH, NVIDIA_SMI_ARGUMENTS, {
    encoding: 'utf8', windowsHide: true, maxBuffer: 256 * 1024,
    timeout: STATIC_IDENTITY_TIMEOUT_MS, killSignal: 'SIGKILL',
  });
  if (stderr.trim() !== '') throw new Error(`nvidia-smi emitted stderr: ${stderr.trim()}`);
  return {
    schemaVersion: 1,
    os: { platform: platform(), type: osType(), release: release(), version: osVersion(), arch: arch() },
    gpuStaticTelemetryIdentity: {
      command: normalizePath(NVIDIA_SMI_PATH), arguments: [...NVIDIA_SMI_ARGUMENTS],
      executable: smiIdentity, devices: parseNvidiaIdentity(stdout),
    },
  };
}

function directoryInventoriesExact(start, end) {
  return start?.schemaVersion === 1 && exactJson(start, end)
    && typeof start?.root === 'string' && start.fileCount > 0 && start.totalBytes > 0
    && SHA256_PATTERN.test(start.inventorySha256 ?? '')
    && Array.isArray(start.files) && start.files.length === start.fileCount
    && start.files.every((file) => typeof file?.path === 'string' && file.path.length > 0
      && Number.isSafeInteger(file.byteLength) && file.byteLength >= 0
      && SHA256_PATTERN.test(file.sha256 ?? ''));
}

function rawCorrectnessWitness(rawReport) {
  const direct = rawReport?.pageResult?.canaries?.directRenderPass;
  const bundle = rawReport?.pageResult?.canaries?.renderBundleSnapshot;
  const records = [
    {
      witnessId: 'raw/direct-render-pass', kind: 'direct-render-pass-address-output',
      addresses: direct?.addressSequence, expected: direct?.expectedU32Sequence,
      observed: direct?.observedU32Sequence,
    },
    {
      witnessId: 'raw/render-bundle-snapshot', kind: 'render-bundle-snapshot-address-output',
      addresses: bundle?.addressSequenceAtEncoding, expected: bundle?.expectedU32Sequence,
      observed: bundle?.observedU32Sequence,
      mutatedSourceWouldProduce: bundle?.mutatedSourceWouldProduceU32Sequence,
    },
  ].map((record) => ({
    ...record,
    sha256: sha256(Buffer.from(JSON.stringify(record))),
  }));
  return {
    schemaVersion: 1, kind: 'raw-webgpu-immediate-address-output-witnesses',
    records,
    sha256: sha256(Buffer.from(JSON.stringify(records))),
  };
}

function validateRawCorrectnessWitness(witness, rawReport) {
  return exactJson(witness, rawCorrectnessWitness(rawReport))
    && witness.records.every((record) => exactArray(record.addresses,
      record.witnessId.endsWith('direct-render-pass')
        ? [2, 1, 5] : [1, 4, 2])
      && exactArray(record.observed, record.expected))
    && witness.records[1].observed.every(
      (value, index) => value !== witness.records[1].mutatedSourceWouldProduce[index],
    );
}

async function verifyJustWrittenRawArtifact(raw) {
  const relativePath = raw?.artifact?.relativePath;
  if (typeof relativePath !== 'string') return false;
  const filename = path.resolve(repoRoot, relativePath);
  if (!withinRoot(filename, repoRoot)) return false;
  const expectedParent = path.resolve(
    repoRoot, 'results', 'development', 'webgpu-immediates-default-exposure-canary',
  );
  const directory = path.dirname(filename);
  if (path.dirname(directory).toLowerCase() !== expectedParent.toLowerCase()
    || !/^canary-[a-f0-9]{24}$/u.test(path.basename(directory))
    || path.basename(filename) !== 'report.json') return false;
  let bytes;
  let metadata;
  try {
    ({ bytes, metadata } = await readRegularFileSecurely(filename, repoRoot));
    const siblings = await readdir(directory, { withFileTypes: true });
    if (siblings.length !== 1 || siblings[0].name !== 'report.json'
      || !siblings[0].isFile() || siblings[0].isSymbolicLink()) return false;
  } catch { return false; }
  if (!rawWebGpuImmediatesArtifactModeIsReadOnly(metadata)
    || bytes.length !== raw.artifact.byteLength
    || sha256(bytes) !== raw.artifact.sha256) {
    return false;
  }
  let parsed;
  try { parsed = JSON.parse(bytes.toString('utf8')); } catch { return false; }
  return exactJson(parsed, raw.report);
}

function inventoryByPath(inventory) {
  return new Map((inventory?.files ?? []).map((file) => [file.path, file]));
}

function buildOverlayIntegrationEvidence(sourceInventory, overlayInventory, manifest, targetKey) {
  const sourceFiles = inventoryByPath(sourceInventory);
  const overlayFiles = inventoryByPath(overlayInventory);
  const manifestFilename = targetKey === THREE_IMMEDIATE_PHASE0_TARGET_DEV
    ? 'THREE_IMMEDIATE_DEV_OVERLAY.json' : 'THREE_IMMEDIATE_OVERLAY.json';
  const patched = targetKey === THREE_IMMEDIATE_PHASE0_TARGET_DEV
    ? manifest?.transform?.patchedTargetSha256 : manifest?.patchedSha256;
  const outputPayload = [...overlayFiles.keys()].filter((filename) => filename !== manifestFilename);
  const expectedPayload = targetKey === THREE_IMMEDIATE_PHASE0_TARGET_DEV
    ? (manifest?.output?.files ?? []).map((file) => file.path)
    : [...sourceFiles.keys()];
  const records = outputPayload.map((relativePath) => {
    const source = sourceFiles.get(relativePath);
    const overlay = overlayFiles.get(relativePath);
    const expectedPatchedSha256 = patched?.[relativePath] ?? null;
    return {
      path: relativePath,
      classification: expectedPatchedSha256 === null ? 'byte-identical' : 'manifest-patched',
      sourceByteLength: source?.byteLength ?? null,
      sourceSha256: source?.sha256 ?? null,
      overlayByteLength: overlay?.byteLength ?? null,
      overlaySha256: overlay?.sha256 ?? null,
      expectedPatchedSha256,
      exact: source !== undefined && (expectedPatchedSha256 === null
        ? source.byteLength === overlay.byteLength && source.sha256 === overlay.sha256
        : overlay.sha256 === expectedPatchedSha256),
    };
  });
  const recordPaths = records.map((record) => record.path);
  return {
    schemaVersion: 1,
    kind: 'three-immediate-overlay-source-integration-diff',
    targetKey,
    manifestFilename,
    expectedPayloadPathsSha256: sha256(Buffer.from(JSON.stringify([...expectedPayload].sort()))),
    outputPayloadPathsSha256: sha256(Buffer.from(JSON.stringify([...recordPaths].sort()))),
    records,
    exact: exactArray([...recordPaths].sort(), [...expectedPayload].sort())
      && records.every((record) => record.exact)
      && Object.entries(patched ?? {}).every(([relativePath, hash]) => (
        records.filter((record) => record.path === relativePath).length === 1
          && records.find((record) => record.path === relativePath)?.overlaySha256 === hash
      )),
  };
}

async function captureSourceFreezeStart(target) {
  const [provenance, unconfiguredBound, trackedSources,
    installedDependencies, targetSource] = await Promise.all([
    collectSourceProvenance(repoRoot),
    collectBoundSources(null),
    captureTrackedSourceInventory(),
    inventoryDirectory(path.join(repoRoot, 'node_modules'), {
      excludeDirectories: ['.vite'],
    }),
    inventoryDirectory(
      target.key === THREE_IMMEDIATE_PHASE0_TARGET_DEV
        ? DEFAULT_THREE_DEV_SOURCE_ROOT : DEFAULT_SOURCE_ROOT,
      { excludeDirectories: ['.git'] },
    ),
  ]);
  const projected = projectedProvenance(provenance);
  if (projected?.dirty !== false) {
    throw new Error('Phase 0 requires one clean committed repository tree.');
  }
  return {
    provenance: projected, unconfiguredBound, trackedSources,
    installedDependencies,
    installedDependencyExclusions: ['.vite'],
    targetSource,
  };
}

async function captureSourceFreezeEnd(target) {
  return captureSourceFreezeStart(target);
}

function sourceFreezeExact(start, end) {
  return provenanceProjectionMatches(start?.provenance, end?.provenance)
    && unconfiguredBoundSourcesExact(start?.unconfiguredBound, end?.unconfiguredBound)
    && directoryInventoriesExact(start?.trackedSources, end?.trackedSources)
    && exactArray(start?.installedDependencyExclusions, ['.vite'])
    && exactArray(end?.installedDependencyExclusions, ['.vite'])
    && directoryInventoriesExact(start?.installedDependencies, end?.installedDependencies)
    && directoryInventoriesExact(start?.targetSource, end?.targetSource);
}

function normalizeGpuIdentityText(value) {
  return String(value ?? '').toLowerCase().replace(/[^a-z0-9]+/gu, ' ').trim();
}

function driverDigitsMatch(left, right) {
  const leftDigits = String(left ?? '').replace(/\D/gu, '');
  const rightDigits = String(right ?? '').replace(/\D/gu, '');
  return leftDigits.length >= 4 && rightDigits.length >= 4
    && (leftDigits.endsWith(rightDigits) || rightDigits.endsWith(leftDigits));
}

export function browserGpuMatchesStaticIdentity(gpuSystemInfo, environment) {
  const devices = environment?.gpuStaticTelemetryIdentity?.devices;
  const browserDevices = gpuSystemInfo?.gpu?.devices;
  if (!Array.isArray(devices) || devices.length !== 1
    || gpuSystemInfo?.schemaVersion !== 1
    || gpuSystemInfo?.kind !== 'chrome-cdp-static-gpu-system-info'
    || !Array.isArray(browserDevices) || browserDevices.length !== 1
    || browserDevices.some((browserDevice) => /swiftshader|software|llvmpipe|basic render/iu.test(
      `${browserDevice?.vendorString ?? ''} ${browserDevice?.deviceString ?? ''}`,
    ))) {
    return false;
  }
  const device = devices[0];
  const normalizedName = normalizeGpuIdentityText(device.name);
  const numericModelTokens = normalizedName.split(' ').filter((token) => /\d/u.test(token));
  const matches = browserDevices.filter((browserDevice) => {
    const vendor = normalizeGpuIdentityText(browserDevice?.vendorString);
    const description = normalizeGpuIdentityText(browserDevice?.deviceString);
    const vendorId = Number(browserDevice?.vendorId);
    return (vendor.includes('nvidia') || vendorId === 0x10de)
      && description === normalizedName
      && numericModelTokens.length > 0
      && numericModelTokens.every((token) => description.split(' ').includes(token))
      && driverDigitsMatch(browserDevice?.driverVersion, device.driverVersion);
  });
  return matches.length === 1
    && normalizeGpuIdentityText(matches[0].deviceString).includes('5070')
    && !normalizeGpuIdentityText(matches[0].deviceString).includes('5080');
}

export function adapterMatchesStaticGpu(adapterInfo, environment, gpuSystemInfo) {
  const devices = environment?.gpuStaticTelemetryIdentity?.devices;
  if (!Array.isArray(devices) || devices.length !== 1
    || adapterInfo?.isFallbackAdapter !== false
    || !browserGpuMatchesStaticIdentity(gpuSystemInfo, environment)) {
    return false;
  }
  const device = devices[0];
  const vendor = normalizeGpuIdentityText(adapterInfo?.vendor);
  const description = normalizeGpuIdentityText(adapterInfo?.description);
  const adapterDevice = normalizeGpuIdentityText(adapterInfo?.device);
  const normalizedName = normalizeGpuIdentityText(device.name).replace(/^nvidia\s+/u, '');
  const vendorExact = vendor.includes('nvidia') || /^0x10de$/iu.test(adapterInfo?.vendor ?? '');
  const numericModelTokens = normalizedName.split(' ').filter((token) => /\d/u.test(token));
  const descriptiveIdentity = `${description} ${adapterDevice}`.trim();
  const descriptiveModelExact = description.length > 0
    && (description === normalizedName
      || description === `nvidia ${normalizedName}`)
    && numericModelTokens.length > 0
    && numericModelTokens.every((token) => descriptiveIdentity.split(' ').includes(token));
  const privacyReducedExact = description === '' && adapterDevice === ''
    && normalizeGpuIdentityText(adapterInfo?.architecture) === 'blackwell';
  const driver = adapterInfo?.driver;
  const driverExactWhenExposed = driver === undefined || driver === null || driver === ''
    || driverDigitsMatch(driver, device.driverVersion);
  return vendorExact && (descriptiveModelExact || privacyReducedExact)
    && driverExactWhenExposed;
}

function phase0AliasConfiguration(overlayRoot) {
  const targets = {
    three: path.join(overlayRoot, 'build', 'three.module.js'),
    'three/webgpu': path.join(overlayRoot, 'build', 'three.webgpu.js'),
    'three/tsl': path.join(overlayRoot, 'build', 'three.tsl.js'),
  };
  const alias = [
    { find: /^three\/webgpu$/u, replacement: targets['three/webgpu'] },
    { find: /^three\/tsl$/u, replacement: targets['three/tsl'] },
    { find: /^three$/u, replacement: targets.three },
    {
      find: /^three\/addons\/(.*)$/u,
      replacement: `${normalizePath(path.join(overlayRoot, 'examples', 'jsm'))}/$1`,
    },
  ];
  return { targets, alias };
}

function serverConfigurationEvidence(overlayRoot, observationNonce) {
  const nonce = OBSERVATION_NONCE_PATTERN.test(observationNonce ?? '')
    ? observationNonce : 'uninitialized';
  return {
    implementation: 'vite.createServer',
    root: 'repo:/',
    configFile: false,
    appType: 'mpa',
    host: '127.0.0.1',
    dynamicPort: true,
    strictPort: false,
    localOnly: true,
    logLevel: 'error',
    clearScreen: false,
    hmr: false,
    isolationHeaders: { ...ISOLATION_HEADERS },
    cacheControl: 'no-store, max-age=0',
    optimizeDependencies: { noDiscovery: true, include: [] },
    browserCacheDisabled: true,
    serviceWorkers: 'block',
    cacheDirectory: normalizePath(path.relative(
      repoRoot, path.join(repoRoot, '.generated', 'phase0-vite-cache', nonce),
    )),
    filesystem: {
      strict: true,
      allow: ['repo:/', 'overlay:/'],
    },
    socketBinding: {
      address: '127.0.0.1', family: 'IPv4', dynamicPort: true,
    },
    aliasOrder: [
      '^three/webgpu$', '^three/tsl$', '^three$', '^three/addons/(.*)$',
    ],
    overlayRoot: normalizePath(path.resolve(overlayRoot)),
    sourceClosurePlugin: 'three-immediate-phase0-source-closure',
  };
}

function viteEffectiveConfigurationEvidence(server, overlayRoot) {
  const configuration = server?.config;
  const allowed = configuration?.server?.fs?.allow ?? [];
  const normalizedAllowed = allowed.map((value) => normalizePath(path.resolve(value)));
  return {
    root: normalizePath(path.resolve(configuration?.root ?? '')),
    cacheDir: normalizePath(path.resolve(configuration?.cacheDir ?? '')),
    configFile: configuration?.configFile ?? false,
    appType: configuration?.appType,
    host: configuration?.server?.host,
    port: configuration?.server?.port,
    strictPort: configuration?.server?.strictPort,
    hmr: configuration?.server?.hmr,
    fsStrict: configuration?.server?.fs?.strict,
    fsAllow: normalizedAllowed,
    optimizeNoDiscovery: configuration?.optimizeDeps?.noDiscovery,
    optimizeInclude: configuration?.optimizeDeps?.include ?? [],
    headers: configuration?.server?.headers,
    expected: {
      root: normalizePath(repoRoot),
      overlayRoot: normalizePath(path.resolve(overlayRoot)),
    },
  };
}

function validViteEffectiveConfiguration(record, requested) {
  const cacheDir = normalizePath(path.resolve(repoRoot, requested?.cacheDirectory ?? ''));
  return record?.root === normalizePath(repoRoot)
    && record?.cacheDir === cacheDir
    && record?.configFile === false && record?.appType === 'mpa'
    && record?.host === '127.0.0.1' && record?.port === 0
    && record?.strictPort === false && record?.hmr === false
    && record?.fsStrict === true
    && Array.isArray(record?.fsAllow)
    && record.fsAllow.includes(normalizePath(repoRoot))
    && record.fsAllow.includes(requested?.overlayRoot)
    && record?.optimizeNoDiscovery === true
    && exactArray(record?.optimizeInclude, [])
    && exactJson(record?.headers, {
      ...ISOLATION_HEADERS, 'Cache-Control': 'no-store, max-age=0',
    })
    && record?.expected?.root === normalizePath(repoRoot)
    && record?.expected?.overlayRoot === requested?.overlayRoot;
}

export async function closeThreeImmediatePhase0IntegrationState(state) {
  state.diagnostics.shutdownStarted = true;
  const errors = [];
  const closeOne = async (name, property, close, timeout = BROWSER_OPERATION_TIMEOUT_MS) => {
    const resource = state[property];
    const record = state.shutdownRecord[name];
    if (resource === null || record.attempted) return;
    record.present = true;
    record.attempted = true;
    try {
      await withDeadline(close(resource), `integration ${name} close`, timeout);
      record.succeeded = true;
    } catch (error) {
      record.error = serializeError(error);
      errors.push(error);
    } finally {
      state[property] = null;
    }
  };
  await closeOne('context', 'context', (context) => context.close());
  await closeOne('browser', 'browser', async (browser) => {
    await browser.close();
    state.browserRecord.closedCleanly = browser.isConnected() === false;
    if (!state.browserRecord.closedCleanly) {
      throw new Error('Integration browser remained connected after close.');
    }
  });
  await closeOne('server', 'server', async (server) => {
    await server.close();
    state.serverRecord.closedCleanly = true;
  }, SERVER_OPERATION_TIMEOUT_MS);
  state.shutdownRecord.complete = ['context', 'browser', 'server'].every((name) => {
    const record = state.shutdownRecord[name];
    return record.present === false || record.attempted === true;
  });
  if (errors.length > 0) {
    throw new AggregateError(errors, 'One or more integration cleanup operations failed.');
  }
}

export async function runThreeImmediatePhase0({ target: targetKey, browser: browserPath } = {}) {
  const target = resolveTarget(targetKey);
  const requestedBrowser = browserPath || DEFAULT_CHROME_PATH;
  const state = {
    browser: null, context: null, server: null,
    diagnostics: createDiagnostics(),
    shutdownRecord: {
      schemaVersion: 1,
      context: { present: false, attempted: false, succeeded: false, error: null },
      browser: { present: false, attempted: false, succeeded: false, error: null },
      server: { present: false, attempted: false, succeeded: false, error: null },
      complete: false,
    },
    attemptRecord: { raw: 0, integration: 0 },
    browserRecord: {
      launchApi: 'playwright-core.chromium.launch', headless: true,
      persistentContext: false, profilePolicy: 'fresh-playwright-temporary-profile',
      exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM,
      launchArguments: [], effectiveArgumentsCapture: null,
      browserVersion: null, userAgent: null, gpuSystemInfo: null,
      cacheDisabledProtocol: false, serviceWorkers: 'block',
      serviceWorkerRegistrationCount: null,
      executableStart: null, executableEnd: null,
      distributionStart: null, distributionEnd: null,
      singleContext: false, singlePage: false, closedCleanly: false,
    },
    serverRecord: {
      ...serverConfigurationEvidence(target.outputRoot, null),
      origin: null, closedCleanly: false,
    },
  };
  let sourceFreezeStart = null;
  let sourceFreezeEnd = null;
  let environmentStart = null;
  let environmentEnd = null;
  let configuredBoundStart = null;
  let configuredBoundEnd = null;
  let overlayInventoryStart = null;
  let overlayInventoryEnd = null;
  let overlayBuild = null;
  let overlayManifestBytes = null;
  let configuration = null;
  let observationNonce = null;
  let raw = null;
  let pageResult = null;
  let pageValidation = null;
  let sourceEvidence = null;
  let failure = null;
  let aliases = null;
  let serverConfiguration = null;
  let baseUrl = null;
  const moduleRecords = new Map();
  const loadedResources = { records: [], pending: new Set() };

  try {
    const executableStart = await exactChromeExecutableIdentity(requestedBrowser);
    process.stdout.write(`${JSON.stringify({
      phase: 'browser-identity-before-launch',
      browserPath: executableStart.canonicalPath,
      productVersion: executableStart.version.productVersion,
    })}\n`);
    state.browserRecord.executableStart = executableStart;
    [sourceFreezeStart, environmentStart, state.browserRecord.distributionStart] =
      await Promise.all([
        captureSourceFreezeStart(target),
        captureStaticEnvironmentIdentity(),
        inventoryDirectory(path.dirname(executableStart.canonicalPath)),
      ]);
    observationNonce = randomBytes(16).toString('hex');

    const priorBrowserPath = process.env.BROWSER_PATH;
    process.env.BROWSER_PATH = executableStart.canonicalPath;
    try {
      state.attemptRecord.raw += 1;
      raw = await runWebGpuImmediatesProbe({
        exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM,
        cacheDirectory: path.join(
          repoRoot, '.generated', 'phase0-raw-vite-cache', observationNonce,
        ),
        emitSummary: false,
        setProcessExitCode: false,
      });
    } finally {
      if (priorBrowserPath === undefined) delete process.env.BROWSER_PATH;
      else process.env.BROWSER_PATH = priorBrowserPath;
    }
    const rawValidation = validateWebGpuImmediatesProbeReport(raw?.report);
    if (!rawValidation.pass || raw?.report?.status !== 'pass') {
      throw new Error(`Raw default-exposure canary failed: ${rawValidation.reasons.join('; ')}`);
    }
    if (!await verifyJustWrittenRawArtifact(raw)) {
      throw new Error('Just-written raw default-exposure artifact failed exact reread verification.');
    }
    if (!executableIdentityExact(executableStart, raw.report.browser.executableStart)) {
      throw new Error('Raw canary did not use the exact predeclared Chrome executable.');
    }
    configuration = configurationFor(targetKey, raw.report, observationNonce);

    overlayBuild = await target.build();
    const overlayManifestPath = path.join(overlayBuild.outputRoot, target.manifestFilename);
    [overlayManifestBytes, overlayInventoryStart] = await Promise.all([
      readFile(overlayManifestPath), inventoryDirectory(overlayBuild.outputRoot),
    ]);
    const manifestOnDisk = JSON.parse(overlayManifestBytes.toString('utf8'));
    if (!exactJson(manifestOnDisk, overlayBuild.manifest)) {
      throw new Error('Overlay builder return value differs from its on-disk manifest.');
    }
    const targetForManifest = targetKey === THREE_IMMEDIATE_PHASE0_TARGET_DEV
      ? THREE_IMMEDIATE_OVERLAY_TARGET_DEV : 'r185';
    const overlayValidation = validateThreeImmediateOverlayManifest(
      overlayBuild.manifest, targetForManifest,
    );
    if (!overlayValidation.valid) {
      throw new Error(`Generated overlay is invalid: ${overlayValidation.reasons.join('; ')}`);
    }
    configuredBoundStart = await collectBoundSources(configuration);

    const aliasConfiguration = phase0AliasConfiguration(overlayBuild.outputRoot);
    aliases = aliasConfiguration.targets;
    const { alias } = aliasConfiguration;
    serverConfiguration = serverConfigurationEvidence(
      overlayBuild.outputRoot, observationNonce,
    );
    state.serverRecord = {
      ...serverConfiguration,
      origin: null, actualSocket: null, responseCount: null,
      effectiveConfiguration: null, closedCleanly: false,
    };
    state.server = await createServer({
      root: repoRoot,
      configFile: false,
      appType: 'mpa',
      logLevel: 'error',
      clearScreen: false,
      cacheDir: path.join(repoRoot, serverConfiguration.cacheDirectory),
      resolve: { alias, dedupe: [] },
      optimizeDeps: { noDiscovery: true, include: [] },
      plugins: [sourceObserverPlugin({
        configuration, overlayRoot: overlayBuild.outputRoot, records: moduleRecords,
      })],
      server: {
        host: '127.0.0.1', port: 0, strictPort: false, hmr: false,
        headers: { ...ISOLATION_HEADERS, 'Cache-Control': 'no-store, max-age=0' },
        fs: { strict: true, allow: [repoRoot, overlayBuild.outputRoot] },
      },
    });
    state.shutdownRecord.server.present = true;
    state.serverRecord.effectiveConfiguration = viteEffectiveConfigurationEvidence(
      state.server, overlayBuild.outputRoot,
    );
    if (!validViteEffectiveConfiguration(
      state.serverRecord.effectiveConfiguration, serverConfiguration,
    )) {
      throw new Error('Resolved Vite configuration differs from the frozen local-only configuration.');
    }
    await withDeadline(state.server.listen(), 'integration Vite server listen',
      SERVER_OPERATION_TIMEOUT_MS);
    const address = state.server.httpServer?.address();
    if (address === null || address === undefined || typeof address === 'string') {
      throw new Error('Vite did not expose one numeric loopback port.');
    }
    baseUrl = `http://127.0.0.1:${address.port}`;
    state.serverRecord.origin = baseUrl;
    state.serverRecord.actualSocket = {
      address: address.address,
      family: address.family,
      port: address.port,
    };

    state.attemptRecord.integration += 1;
    state.browser = await chromium.launch({
      executablePath: executableStart.canonicalPath,
      headless: true,
      args: [],
      timeout: BROWSER_OPERATION_TIMEOUT_MS,
    });
    state.shutdownRecord.browser.present = true;
    state.browserRecord.browserVersion = state.browser.version();
    [state.browserRecord.effectiveArgumentsCapture, state.browserRecord.gpuSystemInfo] =
      await Promise.all([
        captureEffectiveArguments(state.browser),
        captureBrowserGpuSystemInfo(state.browser),
      ]);
    const exposure = assessDefaultExposureBrowserRecord(state.browserRecord);
    if (!exposure.pass) {
      throw new Error(`Integration browser has non-default exposure: ${exposure.reasons.join('; ')}`);
    }
    if (state.browser.contexts().length !== 0) {
      throw new Error('Fresh integration browser unexpectedly has a context.');
    }
    state.context = await withDeadline(state.browser.newContext({
      viewport: { width: TARGET_WIDTH, height: TARGET_HEIGHT },
      deviceScaleFactor: 1,
      serviceWorkers: 'block',
    }), 'integration browser context creation');
    state.shutdownRecord.context.present = true;
    state.browserRecord.singleContext = state.browser.contexts().length === 1;
    const page = await withDeadline(state.context.newPage(), 'integration page creation');
    state.browserRecord.singlePage = state.context.pages().length === 1;
    if (!state.browserRecord.singleContext || !state.browserRecord.singlePage) {
      throw new Error('Integration browser did not retain exactly one context/page.');
    }
    attachDiagnostics(page, state.diagnostics, loadedResources);
    const cdp = await withDeadline(
      state.context.newCDPSession(page), 'integration page CDP session creation',
    );
    await withDeadline(cdp.send('Network.enable'), 'integration Network.enable');
    await withDeadline(
      cdp.send('Network.setCacheDisabled', { cacheDisabled: true }),
      'integration Network.setCacheDisabled',
    );
    state.browserRecord.cacheDisabledProtocol = true;
    await page.goto(`${baseUrl}/phase0-immediate-aif.html`, {
      waitUntil: 'domcontentloaded', timeout: BROWSER_OPERATION_TIMEOUT_MS,
    });
    const preexistingRegistrations = await withDeadline(page.evaluate(
      () => navigator.serviceWorker?.getRegistrations().then((items) => items.length) ?? 0,
    ), 'service-worker registration query');
    state.browserRecord.serviceWorkerRegistrationCount = preexistingRegistrations;
    if (preexistingRegistrations !== 0) throw new Error('A service worker registration is active.');
    await page.waitForFunction(
      () => window.__THREE_IMMEDIATE_AIF_PHASE0__?.ready === true,
      null,
      { timeout: PAGE_READINESS_TIMEOUT_MS },
    );
    pageResult = await withDeadline(page.evaluate(
      () => structuredClone(window.__THREE_IMMEDIATE_AIF_PHASE0__.result),
    ), 'large Phase 0 result extraction', PAGE_READINESS_TIMEOUT_MS);
    state.browserRecord.userAgent = await withDeadline(
      page.evaluate(() => navigator.userAgent), 'integration user-agent extraction',
    );
    pageValidation = validateThreeImmediatePhase0PageResult(pageResult, {
      targetKey, rawReport: raw.report, observationNonce,
    });
    if (!pageValidation.valid) {
      throw createPhase0PageValidationError(pageResult, pageValidation);
    }
    await withDeadline(
      Promise.all([...loadedResources.pending]),
      'loaded response-body closure capture',
    );
    state.serverRecord.responseCount = loadedResources.records.length;
    if (!diagnosticsClean(state.diagnostics)) {
      throw new Error('Integration browser emitted a page/network/crash diagnostic.');
    }
    await withDeadline(cdp.detach(), 'integration page CDP detach');
    await closeThreeImmediatePhase0IntegrationState(state);

    [configuredBoundEnd, overlayInventoryEnd, sourceFreezeEnd, environmentEnd,
      state.browserRecord.distributionEnd, state.browserRecord.executableEnd] = await Promise.all([
      collectBoundSources(configuration),
      inventoryDirectory(overlayBuild.outputRoot),
      captureSourceFreezeEnd(target),
      captureStaticEnvironmentIdentity(),
      inventoryDirectory(path.dirname(executableStart.canonicalPath)),
      exactChromeExecutableIdentity(executableStart.canonicalPath),
    ]);
    if (!sourceFreezeExact(sourceFreezeStart, sourceFreezeEnd)
      || !directoryInventoriesExact(overlayInventoryStart, overlayInventoryEnd)
      || !exactJson(environmentStart, environmentEnd)
      || !directoryInventoriesExact(
        state.browserRecord.distributionStart, state.browserRecord.distributionEnd,
      )
      || !exactJson(state.browserRecord.executableStart, state.browserRecord.executableEnd)) {
      throw new Error('Source, dependency, overlay, OS/GPU, or Chrome identity changed during Phase 0.');
    }
    sourceEvidence = await buildSourceEvidence({
      overlayRoot: overlayBuild.outputRoot,
      configuration,
      aliases,
      records: moduleRecords,
      loadedResources,
      boundStart: configuredBoundStart,
      boundEnd: configuredBoundEnd,
      serverOrigin: baseUrl,
      serverConfiguration,
    });
    sourceEvidence.freezeStart = sourceFreezeStart;
    sourceEvidence.freezeEnd = sourceFreezeEnd;
    sourceEvidence.freezeExact = true;
    sourceEvidence.overlayStart = overlayInventoryStart;
    sourceEvidence.overlayEnd = overlayInventoryEnd;
    sourceEvidence.overlayExact = true;
    sourceEvidence.overlayIntegration = buildOverlayIntegrationEvidence(
      sourceFreezeStart.targetSource,
      overlayInventoryStart,
      overlayBuild.manifest,
      targetKey,
    );
    if (!sourceEvidence.overlayIntegration.exact) {
      throw new Error('Overlay output is not the exact declared source-to-patch integration diff.');
    }
    if (!adapterMatchesStaticGpu(
      pageResult.adapterInfo, environmentStart, state.browserRecord.gpuSystemInfo,
    )) {
      throw new Error('Page WebGPU adapter does not match the static NVIDIA identity.');
    }
  } catch (error) {
    failure = serializeError(error);
  } finally {
    if (loadedResources.pending.size > 0) {
      await withDeadline(
        Promise.all([...loadedResources.pending]),
        'failure-path response-body closure capture',
      ).catch((error) => {
        failure ??= serializeError(error);
        failure.responseBodyCapture = serializeError(error);
      });
    }
    state.serverRecord.responseCount ??= loadedResources.records.length;
    await closeThreeImmediatePhase0IntegrationState(state).catch((error) => {
      if (failure === null) failure = serializeError(error);
      else failure.cleanup = ['context', 'browser', 'server']
        .map((name) => state.shutdownRecord[name].error)
        .filter((value) => value !== null);
    });
    const endCaptureFailures = [];
    const capture = async (label, action, assign) => {
      try { assign(await action()); } catch (error) {
        endCaptureFailures.push({ label, error: serializeError(error) });
      }
    };
    const captures = [];
    if (configuredBoundStart !== null && configuration !== null && configuredBoundEnd === null) {
      captures.push(capture('configured-bound-end', () => collectBoundSources(configuration),
        (value) => { configuredBoundEnd = value; }));
    }
    if (overlayInventoryStart !== null && overlayBuild !== null && overlayInventoryEnd === null) {
      captures.push(capture('overlay-inventory-end',
        () => inventoryDirectory(overlayBuild.outputRoot),
        (value) => { overlayInventoryEnd = value; }));
    }
    if (sourceFreezeStart !== null && sourceFreezeEnd === null) {
      captures.push(capture('source-freeze-end', () => captureSourceFreezeEnd(target),
        (value) => { sourceFreezeEnd = value; }));
    }
    if (environmentStart !== null && environmentEnd === null) {
      captures.push(capture('environment-end', captureStaticEnvironmentIdentity,
        (value) => { environmentEnd = value; }));
    }
    if (state.browserRecord.distributionStart !== null
      && state.browserRecord.distributionEnd === null) {
      captures.push(capture('browser-distribution-end',
        () => inventoryDirectory(path.dirname(state.browserRecord.executableStart.canonicalPath)),
        (value) => { state.browserRecord.distributionEnd = value; }));
    }
    if (state.browserRecord.executableStart !== null
      && state.browserRecord.executableEnd === null) {
      captures.push(capture('browser-executable-end',
        () => exactChromeExecutableIdentity(state.browserRecord.executableStart.canonicalPath),
        (value) => { state.browserRecord.executableEnd = value; }));
    }
    await Promise.all(captures);
    if (sourceEvidence === null && overlayBuild !== null && configuration !== null
      && aliases !== null && serverConfiguration !== null && baseUrl !== null
      && configuredBoundStart !== null && configuredBoundEnd !== null) {
      await capture('source-evidence-end', async () => {
        const evidence = await buildSourceEvidence({
          overlayRoot: overlayBuild.outputRoot, configuration, aliases,
          records: moduleRecords, loadedResources,
          boundStart: configuredBoundStart, boundEnd: configuredBoundEnd,
          serverOrigin: baseUrl, serverConfiguration,
        });
        evidence.freezeStart = sourceFreezeStart;
        evidence.freezeEnd = sourceFreezeEnd;
        evidence.freezeExact = sourceFreezeExact(sourceFreezeStart, sourceFreezeEnd);
        evidence.overlayStart = overlayInventoryStart;
        evidence.overlayEnd = overlayInventoryEnd;
        evidence.overlayExact = directoryInventoriesExact(
          overlayInventoryStart, overlayInventoryEnd,
        );
        evidence.overlayIntegration = buildOverlayIntegrationEvidence(
          sourceFreezeStart?.targetSource, overlayInventoryStart,
          overlayBuild.manifest, targetKey,
        );
        return evidence;
      }, (value) => { sourceEvidence = value; });
    }
    if (endCaptureFailures.length > 0) {
      if (failure === null) {
        failure = {
          name: 'Phase0EndCaptureError',
          message: 'One or more required end-of-run identities could not be captured.',
          stack: null,
        };
      }
      failure.endCapture = endCaptureFailures;
    }
  }

  const unsupported = raw?.report?.pageResult?.status === 'unsupported'
    || pageResult?.status === 'technical-canary-unsupported';
  const report = {
    schemaVersion: 1,
    kind: THREE_IMMEDIATE_PHASE0_RUNNER_KIND,
    status: failure === null ? 'technical-canary-pass'
      : unsupported ? 'technical-canary-unsupported' : 'technical-canary-failed',
    executionMode: 'technical-canary',
    analysisEligible: false,
    efficacyAnalysisAllowed: false,
    numericalDecision: null,
    timingCaptured: false,
    efficacyEvaluated: false,
    fullPhase0Pass: failure === null,
    target: configuration === null ? targetIdentity(target) : targetIdentity(configuration),
    observationNonce,
    executionPolicy: {
      attemptCount: 1,
      rawAttemptCount: state.attemptRecord.raw,
      integrationAttemptCount: state.attemptRecord.integration,
      retryCount: 0, replacementAllowed: false,
      developerFlagFallbackAllowed: false, launchArguments: [],
      timingCaptured: false, efficacyEvaluated: false,
    },
    raw: raw === null ? null : {
      report: raw.report,
      artifact: {
        relativePath: raw.artifact.relativePath,
        byteLength: raw.artifact.byteLength,
        sha256: raw.artifact.sha256,
      },
      correctnessWitness: rawCorrectnessWitness(raw.report),
    },
    overlay: overlayBuild === null ? null : {
      manifest: overlayBuild.manifest,
      manifestSha256: overlayManifestBytes === null ? null : sha256(overlayManifestBytes),
      validation: validateThreeImmediateOverlayManifest(
        overlayBuild.manifest,
        targetKey === THREE_IMMEDIATE_PHASE0_TARGET_DEV
          ? THREE_IMMEDIATE_OVERLAY_TARGET_DEV : 'r185',
      ),
    },
    sourceIdentity: sourceEvidence,
    environmentIdentity: {
      start: environmentStart,
      end: environmentEnd,
      exact: environmentStart !== null && exactJson(environmentStart, environmentEnd),
    },
    integration: {
      browser: state.browserRecord,
      server: state.serverRecord,
      diagnostics: { events: state.diagnostics.events, overflowCount: state.diagnostics.overflowCount },
      shutdown: state.shutdownRecord,
      pageResult,
      pageValidation,
    },
    failure,
  };
  if (failure === null) {
    const validation = validateThreeImmediatePhase0Report(report);
    if (!validation.valid) {
      report.status = 'technical-canary-failed';
      report.fullPhase0Pass = false;
      report.failure = {
        name: 'Phase0IndependentValidationError',
        message: 'Completed browser result failed independent artifact reconstruction.',
        stack: null,
        detail: validation,
      };
    }
  }
  const artifact = await writeExclusiveReport(report, target);
  if (report.status === 'technical-canary-pass') {
    const postWriteVerification = await verifyThreeImmediatePhase0Artifact(
      artifact.reportPath,
    );
    if (!postWriteVerification.valid
      || postWriteVerification.artifact.byteLength !== artifact.byteLength
      || postWriteVerification.artifact.sha256 !== artifact.sha256
      || !exactJson(postWriteVerification.report, report)) {
      throw new Error(`Just-written Phase 0 artifact failed independent reread: ${
        postWriteVerification.reasons.join('; ')}`);
    }
  }
  return { report: deepFreeze(report), artifact };
}

async function phase0CliMain(arguments_) {
  const options = parseThreeImmediatePhase0Arguments(arguments_);
  if (options.help) {
    process.stdout.write(`${threeImmediatePhase0Help()}\n`);
    return;
  }
  if (options.verify !== null) {
    const verification = await verifyThreeImmediatePhase0Artifact(options.verify);
    if (options.target !== null && verification.report?.target?.key !== options.target) {
      verification.valid = false;
      verification.reasons = [...new Set([
        ...verification.reasons,
        `Artifact target does not match explicit --target ${options.target}.`,
      ])];
    }
    process.stdout.write(`${JSON.stringify({
      valid: verification.valid,
      artifact: verification.artifact,
      reasons: verification.reasons,
    }, null, 2)}\n`);
    if (!verification.valid) process.exitCode = 1;
    return;
  }
  const { report, artifact } = await runThreeImmediatePhase0(options);
  process.stdout.write(`${JSON.stringify({
    status: report.status, target: report.target, artifact, failure: report.failure,
  }, null, 2)}\n`);
  if (report.status !== 'technical-canary-pass') process.exitCode = 1;
}

const invokedUrl = process.argv[1] === undefined
  ? null : pathToFileURL(path.resolve(process.argv[1])).href;
if (invokedUrl === import.meta.url) {
  phase0CliMain(process.argv.slice(2)).catch((error) => {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
