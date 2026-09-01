import { createHash } from 'node:crypto';
import {
  lstat,
  readFile,
  readdir,
  stat,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { summarizeFrozenCrossoverRows } from './frozen-crossover-summary.mjs';
import { summarizeFirstInstanceCrossoverRows } from './first-instance-crossover-summary.mjs';
import {
  summarizeLiveFirstInstanceCrossoverRows,
} from './live-first-instance-crossover-summary.mjs';
import {
  FROZEN_CROSSOVER_BLOCK_SIZE,
  FROZEN_CROSSOVER_MEASURED_BLOCKS,
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  FROZEN_CROSSOVER_PATTERNS,
  FROZEN_CROSSOVER_WARMUP_BLOCKS,
  FROZEN_CROSSOVER_WARMUP_FRAMES,
  frozenCrossoverFrame,
} from '../src/benchmark/frozen-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS,
  FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_CROSSOVER_PATTERNS,
  FIRST_INSTANCE_CROSSOVER_WARMUP_BLOCKS,
  FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
} from '../src/benchmark/first-instance-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceLiveCrossoverHistoryCounts,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS,
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS,
  FIRST_INSTANCE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS,
  FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS,
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS,
  FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS,
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
  FROZEN_DEPTH_CROSSOVER_ORIENTATION_OFFSETS,
  FROZEN_DEPTH_CROSSOVER_REPETITIONS,
  FROZEN_DEPTH_CROSSOVER_STORAGE_ORDERS,
} from '../src/benchmark/plan.js';
import {
  physicalBinSequenceIdentity,
  renderParityIdentity,
  validateDepthOrderingCompletionInvariant,
  validateExactValidation,
  validateFrozenCrossoverCompletionInvariant,
  validateFrozenCrossoverRenderParity,
  validateGeometryFixtureManifest,
  validateScenarioManifest,
} from '../scripts/evidence-validation.mjs';
import {
  firstInstanceCrossoverScheduleSha256,
  firstInstanceRenderParityIdentity,
  firstInstanceValidationSemanticSha256,
  validateFirstInstanceCrossoverRenderParity,
  validateFirstInstanceCrossoverValidation,
  validateFirstInstanceTrialEvidence,
} from '../scripts/first-instance-evidence-validation.mjs';
import {
  liveFirstInstanceCrossoverScheduleSha256,
  liveFirstInstanceRenderParityIdentity,
  liveFirstInstanceValidationSemanticSha256,
  validateLiveFirstInstanceCrossoverRenderParity,
  validateLiveFirstInstanceCrossoverValidation,
  validateLiveFirstInstanceForcedFeatureOffGate,
  validateLiveFirstInstanceTrialEvidence,
} from '../scripts/live-first-instance-evidence-validation.mjs';
import {
  NVIDIA_QUERY_FIELDS,
  TELEMETRY_CSV_FIELDS,
  compareComputeProcessIdentitySets,
  createNvidiaTelemetryCoverageAudit,
  summarizeTelemetryRows,
} from '../scripts/nvidia-telemetry.mjs';
import {
  executionDependencyClosuresMatch,
  validateCandidateViteRuntimeAudit,
} from '../scripts/execution-dependency-closure.mjs';
import {
  createLiveFirstInstanceEnvironmentAudit,
} from '../scripts/live-first-instance-environment-audit.mjs';
import {
  LIVE_PRIVATE_BUNDLE_LABEL,
  LIVE_PUBLIC_BUNDLE_LABEL,
  LIVE_PUBLIC_EVIDENCE_POLICY_ID,
  LIVE_SANITIZER_IMPLEMENTATION_PATHS,
  resolveLiveSanitizerImplementationAtCommit,
  validateLivePublicBundleProvenanceShape,
} from '../scripts/live-evidence-sanitizer-policy.mjs';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const REQUIRED_RUN_ARTIFACTS = Object.freeze([
  'frames.csv',
  'metadata.json',
  'trial-summaries.json',
  'validation-artifacts.json',
  'workload-manifests.json',
  'gpu-telemetry-summary.json',
]);
const LIVE_FIRST_INSTANCE_REQUIRED_RUN_ARTIFACTS = Object.freeze([
  ...REQUIRED_RUN_ARTIFACTS,
  'forced-feature-off-evidence.json',
]);

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const ALLOWED_EVIDENCE_STATUSES = new Set(['development', 'candidate']);
const FIXED_SLICE_REPRESENTATION_MATRIX = 'fixed-slice-representation';
const FIXED_SLICE_REPRESENTATION_MODES = Object.freeze([
  'fixed-slice-per-bucket',
  'fixed-slice',
]);
const FIXED_SLICE_REPRESENTATION_VISIBILITIES = Object.freeze([0.2, 0.8, 0.99]);
const FIXED_SLICE_REPRESENTATION_REPETITIONS = 6;
const FIXED_SLICE_REPRESENTATION_WARMUP_FRAMES = 300;
const FIXED_SLICE_REPRESENTATION_MEASURED_FRAMES = 240;
const FIXED_SLICE_REPRESENTATION_OBJECT_COUNTS = Object.freeze([4_096, 16_384, 65_536]);
const FIXED_SLICE_REPRESENTATION_BUCKET_COUNTS = Object.freeze([1, 4, 32, 128]);
const FIXED_SLICE_REPRESENTATION_ORDERING =
  'six-repetition-balanced-ab-ba-with-rotated-visibility-order';
const DEPTH_ORDERING_MATRIX = 'depth-ordering';
const DEPTH_ORDERING_MODES = Object.freeze([
  'fixed-slice',
  'fixed-slice-depth-front-to-back',
  'fixed-slice-depth-reverse',
]);
const DEPTH_ORDERING_LAYOUTS = Object.freeze([
  'high-overlap',
  'low-overlap',
]);
const DEPTH_ORDERING_VISIBILITIES = Object.freeze([0.99]);
const DEPTH_ORDERING_REPETITIONS = 6;
const DEPTH_ORDERING_WARMUP_FRAMES = 300;
const DEPTH_ORDERING_MEASURED_FRAMES = 240;
const DEPTH_ORDERING_OBJECT_COUNT = 65_536;
const DEPTH_ORDERING_BUCKET_COUNT = 32;
const DEPTH_ORDERING_BIN_COUNT = 8;
const DEPTH_ORDERING_ORDERING =
  'all-six-mode-permutations-with-alternating-high-low-layout-order';
const DEPTH_ORDERING_RENDER_PARITY =
  'same-snapshot exact validation plus two stable offscreen captures of rgba8 color, depth32float, and encoded object ID';
const DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS = 8;
const DEPTH_ORDERING_PARITY_WIDTH = 1280;
const DEPTH_ORDERING_PARITY_HEIGHT = 720;
const DEPTH_ORDERING_PARITY_BYTE_LENGTH =
  DEPTH_ORDERING_PARITY_WIDTH * DEPTH_ORDERING_PARITY_HEIGHT * 4;
const DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS = 6;
const DEPTH_ORDERING_MINIMUM_DIRECTIONALLY_CONSISTENT_PAIRS = 5;
const DEPTH_ORDERING_MATERIAL_PERCENT = 10;
const DEPTH_ORDERING_MATERIAL_ABSOLUTE_MS = 0.10;
const DEPTH_ORDERING_LOW_OVERLAP_MAX_REGRESSION_PERCENT = 5;
const DEPTH_ORDERING_MODE_ORDERS = Object.freeze([
  Object.freeze([
    DEPTH_ORDERING_MODES[0],
    DEPTH_ORDERING_MODES[1],
    DEPTH_ORDERING_MODES[2],
  ]),
  Object.freeze([
    DEPTH_ORDERING_MODES[1],
    DEPTH_ORDERING_MODES[2],
    DEPTH_ORDERING_MODES[0],
  ]),
  Object.freeze([
    DEPTH_ORDERING_MODES[2],
    DEPTH_ORDERING_MODES[0],
    DEPTH_ORDERING_MODES[1],
  ]),
  Object.freeze([
    DEPTH_ORDERING_MODES[2],
    DEPTH_ORDERING_MODES[1],
    DEPTH_ORDERING_MODES[0],
  ]),
  Object.freeze([
    DEPTH_ORDERING_MODES[1],
    DEPTH_ORDERING_MODES[0],
    DEPTH_ORDERING_MODES[2],
  ]),
  Object.freeze([
    DEPTH_ORDERING_MODES[0],
    DEPTH_ORDERING_MODES[2],
    DEPTH_ORDERING_MODES[1],
  ]),
]);
const FROZEN_DEPTH_CROSSOVER_MATRIX = 'depth-ordering-render-only';
const FROZEN_DEPTH_CROSSOVER_LAYOUTS = DEPTH_ORDERING_LAYOUTS;
const FROZEN_DEPTH_CROSSOVER_VISIBILITIES = DEPTH_ORDERING_VISIBILITIES;
const FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT = DEPTH_ORDERING_OBJECT_COUNT;
const FROZEN_DEPTH_CROSSOVER_BUCKET_COUNT = DEPTH_ORDERING_BUCKET_COUNT;
const FROZEN_DEPTH_CROSSOVER_BIN_COUNT = DEPTH_ORDERING_BIN_COUNT;
const FROZEN_DEPTH_CROSSOVER_MAXIMUM_TIMESTAMP_QUANTUM_NS = 10_000;
const FROZEN_DEPTH_CROSSOVER_PATTERNS_AS_STRINGS = Object.freeze(
  FROZEN_CROSSOVER_PATTERNS.map((_, index) => (
    index === 0 ? 'FRFRRFRF' : 'RFRFFRFR'
  )),
);
const FROZEN_DEPTH_CROSSOVER_VALIDATION_KIND =
  'frozen-depth-crossover-exact-paired-snapshots';
const FROZEN_DEPTH_CROSSOVER_ORDERING =
  'twelve-repetition-paired-eight-frame-frozen-crossover-with-balanced-layout-storage-base-and-starting-orientation';
const FROZEN_DEPTH_CROSSOVER_RENDER_PARITY =
  'preflight, timing-start, and postflight paired-lane exact validation plus two stable offscreen captures per lane of rgba8 color, depth32float, and encoded object ID';
const FIRST_INSTANCE_CROSSOVER_MATRIX = 'first-instance-render-only';
const FIRST_INSTANCE_CROSSOVER_LAYOUTS = Object.freeze(['baseline']);
const FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT = 65_536;
const FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT = 32;
const FIRST_INSTANCE_CROSSOVER_MAXIMUM_TIMESTAMP_QUANTUM_NS = 10_000;
const FIRST_INSTANCE_CROSSOVER_VALIDATION_KIND =
  'first-instance-crossover-exact-paired-snapshots';
const FIRST_INSTANCE_CROSSOVER_PATTERNS_AS_STRINGS = Object.freeze(
  FIRST_INSTANCE_CROSSOVER_PATTERNS.map((pattern) => pattern.map(
    (laneId) => (laneId === FIRST_INSTANCE_CROSSOVER_LANES[0] ? 'P' : 'F'),
  ).join('')),
);
const FIRST_INSTANCE_CROSSOVER_ORDERING =
  'twelve-repetition-two-visibility-eight-frame-crossover-with-pairwise-balanced-command-segment-visibility-order-and-starting-orientation';
const FIRST_INSTANCE_CROSSOVER_RENDER_PARITY =
  'preflight, timing-start, and postflight paired portable/feature exact validation plus two stable offscreen captures per lane of rgba8 color, depth32float, and encoded object ID';
const LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX = 'first-instance-live';
const LIVE_FIRST_INSTANCE_CROSSOVER_LAYOUTS = Object.freeze(['baseline']);
const LIVE_FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT = 65_536;
const LIVE_FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT = 32;
const LIVE_FIRST_INSTANCE_CROSSOVER_MAXIMUM_TIMESTAMP_QUANTUM_NS = 1_000;
const LIVE_FIRST_INSTANCE_CROSSOVER_VALIDATION_KIND =
  'first-instance-live-crossover-exact-paired-snapshots';
const LIVE_FIRST_INSTANCE_CROSSOVER_PATTERNS_AS_STRINGS = Object.freeze(
  FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS.map((pattern) => pattern.map(
    (laneId) => (laneId === FIRST_INSTANCE_LIVE_CROSSOVER_LANES[0] ? 'P' : 'F'),
  ).join('')),
);
const LIVE_FIRST_INSTANCE_CROSSOVER_ORDERING =
  'twelve-repetition-two-visibility-cyclic-eight-frame-live-crossover-with-exact-transition-history-balance-and-pairwise-balanced-lane-physical-order-visibility-order-and-orientation';
const LIVE_FIRST_INSTANCE_CROSSOVER_MEASURED_HISTORY =
  firstInstanceLiveCrossoverHistoryCounts(FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES, 0);
const LIVE_FIRST_INSTANCE_CROSSOVER_RENDER_PARITY =
  'preflight, timing-start, and postflight serialized live-compute portable/feature exact validation plus two stable offscreen captures per lane of rgba8 color, depth32float, and encoded object ID';
const FIRST_INSTANCE_CROSSOVER_BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);
const FIRST_INSTANCE_CROSSOVER_ADAPTER_INFO_FIELDS = Object.freeze([
  'vendor',
  'architecture',
  'device',
  'description',
  'backend',
  'type',
  'driver',
  'isFallbackAdapter',
]);
const FIRST_INSTANCE_CROSSOVER_TIMING_FIELDS = Object.freeze({
  cpuCommonUpdateMs: Object.freeze([
    'cpuCommonUpdateP50Ms',
    'cpuCommonUpdateP95Ms',
  ]),
  cpuFrameBodyMs: Object.freeze(['cpuFrameBodyP50Ms', 'cpuFrameBodyP95Ms']),
  cpuSubmitTotalMs: Object.freeze(['cpuSubmitP50Ms', 'cpuSubmitP95Ms']),
  gpuRenderMs: Object.freeze(['gpuRenderP50Ms', 'gpuRenderP95Ms']),
  gpuPassTotalMs: Object.freeze(['gpuPassTotalP50Ms', 'gpuPassTotalP95Ms']),
});
const FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA = Object.freeze([
  ...Object.values(FIRST_INSTANCE_CROSSOVER_TIMING_FIELDS).flat(),
  'gpuComputeP50Ms',
  'gpuComputeP95Ms',
]);
const PROVENANCE_STABILITY_FIELDS = Object.freeze([
  'commit',
  'tree',
  'ref',
  'dirty',
  'stagedChanges',
  'unstagedChanges',
  'untrackedFiles',
  'porcelainEntryCount',
  'porcelainByteCount',
  'porcelainSha256',
  'trackedFileCount',
  'trackedFilesSha256',
  'packageLockTracked',
  'packageLockSha256',
]);

const REQUIRED_COLUMNS = Object.freeze([
  'modeId',
  'targetVisibilityFraction',
  'gpuPassTotalMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuFrameBodyMs',
  'cpuSubmitTotalMs',
]);

const METRICS = Object.freeze([
  'gpuPassTotalMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuFrameBodyMs',
  'cpuSubmitTotalMs',
  'accountedCpuSubmitPlusGpuPassMs',
]);

const REPETITION_COLUMNS = Object.freeze([
  'repetitionIndex',
  'repetition',
  'suiteRepeat',
  'repeat',
  'trialId',
  'runId',
]);

const FROZEN_CROSSOVER_REQUIRED_COLUMNS = Object.freeze([
  'runId',
  'trialId',
  'planIndex',
  'repetitionIndex',
  'modeOrderPosition',
  'visibilityOrderPosition',
  'frameIndex',
  'phaseFrameIndex',
  'modeId',
  'targetVisibilityFraction',
  'scenarioLayout',
  'layoutOrderPosition',
  'plannedModeOrder',
  'plannedVisibilityOrder',
  'plannedLayoutOrder',
  'protocolWarmupFrames',
  'protocolMeasuredFrames',
  'objectCount',
  'bucketCount',
  'expectedVisibleCount',
  'depthBinRangeNear',
  'depthBinRangeFar',
  'plannedLaneStorageOrder',
  'superblockOrientationOffset',
  'plannedScheduleSha256',
  'crossoverBlockIndex',
  'withinBlockPosition',
  'crossoverPattern',
  'crossoverPatternIndex',
  'laneId',
  'laneBase',
  'frontLaneBase',
  'reverseLaneBase',
  'selectorWriteSerial',
  'renderCallSerial',
  'gpuFrameId',
  'gpuRenderTimestampUidCount',
  'expectedRenderTimestampUidCount',
  'validationKind',
  'validationPass',
  'timestampAvailable',
  'usesCompute',
  'configuredDrawCommands',
  'configuredRenderObjects',
  'configuredComputeDispatches',
  'configuredComputeSubmissions',
  'configuredSubmittedInstances',
  'bundleRecordCallbackCountAtTimingStart',
  'gpuPassTotalMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'cpuComputeSubmitMs',
  'selectorWriteSerialAtTimingStart',
  'renderCallSerialAtTimingStart',
  'computeCallSerialAtTimingStart',
  'totalPipelineCacheEntriesAtTimingStart',
  'computePipelineCacheEntriesAtTimingStart',
  'bundleGroupUuidAtTimingStart',
  'meshUuidAtTimingStart',
  'geometryUuidAtTimingStart',
  'materialUuidAtTimingStart',
  'matrixAttributeIdAtTimingStart',
  'visibleIdsAttributeIdAtTimingStart',
  'indirectAttributeIdAtTimingStart',
  'selectorChallengeAttributeIdAtTimingStart',
  'bundleGroupVersionAtTimingStart',
  'matrixAttributeVersionAtTimingStart',
  'visibleIdsAttributeVersionAtTimingStart',
  'indirectAttributeVersionAtTimingStart',
  'selectorUniformUuidAtTimingStart',
  'renderTargetTextureUuidAtTimingStart',
  'renderTargetWidthAtTimingStart',
  'renderTargetHeightAtTimingStart',
  'renderTargetSamplesAtTimingStart',
  'renderTargetDepthBufferAtTimingStart',
  'cameraViewFnv64AtTimingStart',
  'cameraProjectionFnv64AtTimingStart',
]);

const FROZEN_CROSSOVER_ROW_IDENTITY_FIELDS = Object.freeze([
  'bundleGroupUuidAtTimingStart',
  'meshUuidAtTimingStart',
  'geometryUuidAtTimingStart',
  'materialUuidAtTimingStart',
  'matrixAttributeIdAtTimingStart',
  'visibleIdsAttributeIdAtTimingStart',
  'indirectAttributeIdAtTimingStart',
  'selectorChallengeAttributeIdAtTimingStart',
  'bundleGroupVersionAtTimingStart',
  'matrixAttributeVersionAtTimingStart',
  'visibleIdsAttributeVersionAtTimingStart',
  'indirectAttributeVersionAtTimingStart',
  'selectorUniformUuidAtTimingStart',
  'renderTargetTextureUuidAtTimingStart',
  'renderTargetWidthAtTimingStart',
  'renderTargetHeightAtTimingStart',
  'renderTargetSamplesAtTimingStart',
  'renderTargetDepthBufferAtTimingStart',
  'cameraViewFnv64AtTimingStart',
  'cameraProjectionFnv64AtTimingStart',
]);

const FIRST_INSTANCE_CROSSOVER_REQUIRED_COLUMNS = Object.freeze([
  'schemaVersion',
  'runId',
  'trialId',
  'planIndex',
  'repetitionIndex',
  'modeOrderPosition',
  'visibilityOrderPosition',
  'layoutOrderPosition',
  'plannedModeOrder',
  'plannedVisibilityOrder',
  'plannedLayoutOrder',
  'frameIndex',
  'phaseFrameIndex',
  'modeId',
  'objectCount',
  'bucketCount',
  'targetVisibilityFraction',
  'scenarioLayout',
  'expectedVisibleCount',
  'protocolWarmupFrames',
  'protocolMeasuredFrames',
  'validationKind',
  'validationPass',
  'usesCompute',
  'configuredDrawCommands',
  'configuredRenderObjects',
  'configuredComputeDispatches',
  'configuredComputeSubmissions',
  'configuredSubmittedInstances',
  'timestampAvailable',
  'expectedRenderTimestampUidCount',
  'plannedLaneCommandSegmentOrder',
  'plannedCommandSegments',
  'plannedScheduleSha256',
  'superblockOrientationOffset',
  'lifecycleCommitmentAtTimingStart',
  'rootUuidAtTimingStart',
  'rootVersionAtTimingStart',
  'bundleUuidsAtTimingStart',
  'bundleVersionsAtTimingStart',
  'meshUuidsAtTimingStart',
  'geometryUuidsAtTimingStart',
  'materialUuidsAtTimingStart',
  'commonAttributeIdsAtTimingStart',
  'commonAttributeVersionsAtTimingStart',
  'indexAttributeIdAtTimingStart',
  'indexAttributeVersionAtTimingStart',
  'bucketBaseAttributeIdAtTimingStart',
  'bucketBaseAttributeVersionAtTimingStart',
  'shaderPortableVertexSha256AtTimingStart',
  'shaderFeatureVertexSha256AtTimingStart',
  'shaderNormalizedVertexSha256AtTimingStart',
  'shaderFragmentSha256AtTimingStart',
  'matrixAttributeIdAtTimingStart',
  'matrixAttributeVersionAtTimingStart',
  'visibleIdsAttributeIdAtTimingStart',
  'visibleIdsAttributeVersionAtTimingStart',
  'indirectAttributeIdAtTimingStart',
  'indirectAttributeVersionAtTimingStart',
  'computeCallSerialAtTimingStart',
  'selectorWriteSerialAtTimingStart',
  'strategySelectionSerialAtTimingStart',
  'renderCallSerialAtTimingStart',
  'renderTargetTextureUuidAtTimingStart',
  'renderTargetWidthAtTimingStart',
  'renderTargetHeightAtTimingStart',
  'renderTargetSamplesAtTimingStart',
  'renderTargetDepthBufferAtTimingStart',
  'cameraViewFnv64AtTimingStart',
  'cameraProjectionFnv64AtTimingStart',
  'totalPipelineCacheEntriesAtTimingStart',
  'computePipelineCacheEntriesAtTimingStart',
  'crossoverBlockIndex',
  'withinBlockPosition',
  'crossoverPattern',
  'crossoverPatternIndex',
  'laneId',
  'commandSegmentIndex',
  'commandRecordBase',
  'commandByteBase',
  'selectorWriteSerial',
  'strategySelectionSerial',
  'renderCallSerial',
  'gpuFrameId',
  'gpuRenderTimestampUidCount',
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuSubmitTotalMs',
  'cpuFrameBodyMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'gpuPassTotalMs',
]);

const FIRST_INSTANCE_CROSSOVER_SAFE_INTEGER_COLUMNS = Object.freeze([
  'schemaVersion',
  'planIndex',
  'repetitionIndex',
  'modeOrderPosition',
  'visibilityOrderPosition',
  'layoutOrderPosition',
  'frameIndex',
  'phaseFrameIndex',
  'objectCount',
  'bucketCount',
  'expectedVisibleCount',
  'protocolWarmupFrames',
  'protocolMeasuredFrames',
  'configuredDrawCommands',
  'configuredRenderObjects',
  'configuredComputeDispatches',
  'configuredComputeSubmissions',
  'configuredSubmittedInstances',
  'expectedRenderTimestampUidCount',
  'superblockOrientationOffset',
  'indexAttributeIdAtTimingStart',
  'indexAttributeVersionAtTimingStart',
  'bucketBaseAttributeIdAtTimingStart',
  'bucketBaseAttributeVersionAtTimingStart',
  'matrixAttributeIdAtTimingStart',
  'matrixAttributeVersionAtTimingStart',
  'visibleIdsAttributeIdAtTimingStart',
  'visibleIdsAttributeVersionAtTimingStart',
  'indirectAttributeIdAtTimingStart',
  'indirectAttributeVersionAtTimingStart',
  'computeCallSerialAtTimingStart',
  'selectorWriteSerialAtTimingStart',
  'strategySelectionSerialAtTimingStart',
  'renderCallSerialAtTimingStart',
  'renderTargetWidthAtTimingStart',
  'renderTargetHeightAtTimingStart',
  'renderTargetSamplesAtTimingStart',
  'totalPipelineCacheEntriesAtTimingStart',
  'computePipelineCacheEntriesAtTimingStart',
  'crossoverBlockIndex',
  'withinBlockPosition',
  'crossoverPatternIndex',
  'commandSegmentIndex',
  'commandRecordBase',
  'commandByteBase',
  'selectorWriteSerial',
  'strategySelectionSerial',
  'renderCallSerial',
  'gpuFrameId',
  'gpuRenderTimestampUidCount',
]);

const FIRST_INSTANCE_CROSSOVER_NULLABLE_SAFE_INTEGER_COLUMNS = Object.freeze([
  'rootVersionAtTimingStart',
]);

const FIRST_INSTANCE_CROSSOVER_NUMBER_COLUMNS = Object.freeze([
  'targetVisibilityFraction',
  'cpuCommonUpdateMs',
  'cpuRenderSubmitMs',
  'cpuSubmitTotalMs',
  'cpuFrameBodyMs',
  'gpuRenderMs',
  'gpuPassTotalMs',
]);

const LIVE_FIRST_INSTANCE_CROSSOVER_REQUIRED_COLUMNS = Object.freeze([
  'schemaVersion',
  'runId',
  'trialId',
  'planIndex',
  'repetitionIndex',
  'modeOrderPosition',
  'visibilityOrderPosition',
  'layoutOrderPosition',
  'plannedModeOrder',
  'plannedVisibilityOrder',
  'plannedLayoutOrder',
  'frameIndex',
  'phaseFrameIndex',
  'modeId',
  'objectCount',
  'bucketCount',
  'targetVisibilityFraction',
  'scenarioLayout',
  'expectedVisibleCount',
  'protocolWarmupFrames',
  'protocolMeasuredFrames',
  'validationKind',
  'validationPass',
  'usesCompute',
  'configuredDrawCommands',
  'configuredRenderObjects',
  'configuredComputeDispatches',
  'configuredComputeSubmissions',
  'configuredSubmittedInstances',
  'timestampAvailable',
  'strictTimestampUidAttribution',
  'expectedComputeTimestampUidCount',
  'expectedRenderTimestampUidCount',
  'plannedLanePhysicalOrder',
  'lanePhysicalOrder',
  'plannedScheduleSha256',
  'superblockOrientationOffset',
  'lifecycleCommitmentAtTimingStart',
  'commandBufferCommitmentsAtTimingStart',
  'portableCommandBufferIdAtTimingStart',
  'featureCommandBufferIdAtTimingStart',
  'selectorWriteSerialAtTimingStart',
  'strategySelectionSerialAtTimingStart',
  'strategyComputeCallSerialAtTimingStart',
  'strategyPrepareSerialAtTimingStart',
  'computeCallSerialAtTimingStart',
  'renderCallSerialAtTimingStart',
  'renderTargetTextureUuidAtTimingStart',
  'renderTargetWidthAtTimingStart',
  'renderTargetHeightAtTimingStart',
  'renderTargetSamplesAtTimingStart',
  'renderTargetDepthBufferAtTimingStart',
  'cameraViewFnv64AtTimingStart',
  'cameraProjectionFnv64AtTimingStart',
  'totalPipelineCacheEntriesAtTimingStart',
  'computePipelineCacheEntriesAtTimingStart',
  'computeProgramEntriesAtTimingStart',
  'rendererMemoryAtTimingStart',
  'viewportStateAtTimingStart',
  'timestampPoolStaticCommitmentAtTimingStart',
  'webgpuUncapturedErrorCountAtTimingStart',
  'crossoverBlockIndex',
  'withinBlockPosition',
  'crossoverPattern',
  'crossoverPatternIndex',
  'previousPreviousLaneId',
  'previousLaneId',
  'laneId',
  'commandBufferId',
  'submittedComputeLaneId',
  'computeTimestampContextId',
  'computeGroupIdentity',
  'computeTimestampRegistrationSerial',
  'computeTimestampBackendIdentity',
  'computeTimestampBackendWrapperIdentity',
  'submittedComputeNodeIds',
  'commandSegmentIndex',
  'commandRecordBase',
  'commandByteBase',
  'commandByteOffset',
  'commandBufferRecordCount',
  'commandBufferByteLength',
  'selectorWriteSerial',
  'strategySelectionSerial',
  'strategyComputeCallSerial',
  'computeCallSerial',
  'computeFrameCallIndex',
  'renderCallSerial',
  'renderFrameCallIndex',
  'gpuFrameId',
  'gpuComputeTimestampUidCount',
  'gpuComputeTimestampUids',
  'gpuComputeTimestampRecords',
  'gpuComputeTimestampDurationValid',
  'gpuRenderTimestampUidCount',
  'gpuRenderTimestampUids',
  'gpuRenderTimestampRecords',
  'gpuRenderTimestampDurationValid',
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuSubmitTotalMs',
  'cpuFrameBodyMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'gpuPassTotalMs',
]);

const LIVE_FIRST_INSTANCE_CROSSOVER_SAFE_INTEGER_COLUMNS = Object.freeze([
  'schemaVersion',
  'planIndex',
  'repetitionIndex',
  'modeOrderPosition',
  'visibilityOrderPosition',
  'layoutOrderPosition',
  'frameIndex',
  'phaseFrameIndex',
  'objectCount',
  'bucketCount',
  'expectedVisibleCount',
  'protocolWarmupFrames',
  'protocolMeasuredFrames',
  'configuredDrawCommands',
  'configuredRenderObjects',
  'configuredComputeDispatches',
  'configuredComputeSubmissions',
  'configuredSubmittedInstances',
  'expectedComputeTimestampUidCount',
  'expectedRenderTimestampUidCount',
  'superblockOrientationOffset',
  'portableCommandBufferIdAtTimingStart',
  'featureCommandBufferIdAtTimingStart',
  'selectorWriteSerialAtTimingStart',
  'strategySelectionSerialAtTimingStart',
  'strategyComputeCallSerialAtTimingStart',
  'strategyPrepareSerialAtTimingStart',
  'computeCallSerialAtTimingStart',
  'renderCallSerialAtTimingStart',
  'renderTargetWidthAtTimingStart',
  'renderTargetHeightAtTimingStart',
  'renderTargetSamplesAtTimingStart',
  'totalPipelineCacheEntriesAtTimingStart',
  'computePipelineCacheEntriesAtTimingStart',
  'computeProgramEntriesAtTimingStart',
  'webgpuUncapturedErrorCountAtTimingStart',
  'crossoverBlockIndex',
  'withinBlockPosition',
  'crossoverPatternIndex',
  'commandBufferId',
  'computeTimestampContextId',
  'computeGroupIdentity',
  'computeTimestampRegistrationSerial',
  'computeTimestampBackendIdentity',
  'computeTimestampBackendWrapperIdentity',
  'commandSegmentIndex',
  'commandRecordBase',
  'commandByteBase',
  'commandByteOffset',
  'commandBufferRecordCount',
  'commandBufferByteLength',
  'selectorWriteSerial',
  'strategySelectionSerial',
  'strategyComputeCallSerial',
  'computeCallSerial',
  'computeFrameCallIndex',
  'renderCallSerial',
  'renderFrameCallIndex',
  'gpuFrameId',
  'gpuComputeTimestampUidCount',
  'gpuRenderTimestampUidCount',
]);

const LIVE_FIRST_INSTANCE_CROSSOVER_NUMBER_COLUMNS = Object.freeze([
  'targetVisibilityFraction',
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuSubmitTotalMs',
  'cpuFrameBodyMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'gpuPassTotalMs',
]);

const LIVE_FIRST_INSTANCE_CROSSOVER_TIMING_FIELDS = FIRST_INSTANCE_CROSSOVER_TIMING_FIELDS;
const LIVE_FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA = FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA;
const LIVE_FIRST_INSTANCE_CROSSOVER_TIMESTAMP_SUMMARY_SCHEMA = Object.freeze([
  'accepted',
  'available',
  'rowCount',
  'warmupRowCount',
  'missingWarmupRenderFrames',
  'invalidWarmupRenderTimestampUidCountFrames',
  'invalidWarmupRenderTimestampDurationFrames',
  'missingWarmupComputeFrames',
  'invalidWarmupComputeTimestampUidCountFrames',
  'invalidWarmupComputeTimestampDurationFrames',
  'missingRenderFrames',
  'missingComputeFrames',
  'expectedRenderTimestampUidCount',
  'invalidRenderTimestampUidCountFrames',
  'expectedComputeTimestampUidCount',
  'invalidComputeTimestampUidCountFrames',
  'invalidComputeTimestampDurationFrames',
  'invalidRenderTimestampDurationFrames',
  'renderTimestampPoolQualityValid',
  'computeTimestampPoolQualityValid',
  'warmupRenderTimestampPoolQualityValid',
  'warmupComputeTimestampPoolQualityValid',
  'warmupTimestampFrameCountValid',
  'measurementTimestampFrameCountValid',
  'timestampResolutions',
  'timestampPhases',
  'quantumNs',
  'classification',
]);

function csvError(message, line) {
  const suffix = line === undefined ? '' : ` at CSV line ${line}`;
  return new Error(`${message}${suffix}`);
}

/**
 * Parse RFC-4180-style CSV, including quoted commas, newlines, and doubled quotes.
 */
export function parseCsv(text, { allowEmptyRecords = false } = {}) {
  if (typeof text !== 'string') throw new TypeError('CSV input must be a string.');

  const source = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  let closedQuote = false;
  let line = 1;

  const pushField = () => {
    row.push(field);
    field = '';
    closedQuote = false;
  };
  const pushRow = () => {
    pushField();
    rows.push(row);
    row = [];
  };

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];

    if (quoted) {
      if (character === '"') {
        if (source[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
          closedQuote = true;
        }
      } else {
        field += character;
        if (character === '\n') line += 1;
        if (character === '\r' && source[index + 1] !== '\n') line += 1;
      }
      continue;
    }

    if (closedQuote) {
      if (character === ',') {
        pushField();
      } else if (character === '\n' || character === '\r') {
        if (character === '\r' && source[index + 1] === '\n') index += 1;
        pushRow();
        line += 1;
      } else {
        throw csvError('Unexpected character after a closing quote', line);
      }
      continue;
    }

    if (character === '"') {
      if (field.length !== 0) throw csvError('Quote encountered inside an unquoted field', line);
      quoted = true;
    } else if (character === ',') {
      pushField();
    } else if (character === '\n' || character === '\r') {
      if (character === '\r' && source[index + 1] === '\n') index += 1;
      pushRow();
      line += 1;
    } else {
      field += character;
    }
  }

  if (quoted) throw csvError('Unterminated quoted field', line);
  if (closedQuote || field.length > 0 || row.length > 0) pushRow();

  const nonBlankRows = rows.filter((values) => !(values.length === 1 && values[0] === ''));
  if (nonBlankRows.length === 0) throw new Error('CSV input is empty.');

  const headers = nonBlankRows[0].map((header) => header.trim());
  if (headers.some((header) => header.length === 0)) {
    throw new Error('CSV header contains an empty column name.');
  }
  if (new Set(headers).size !== headers.length) {
    throw new Error('CSV header contains duplicate column names.');
  }

  const records = nonBlankRows.slice(1).map((values, recordIndex) => {
    if (values.length !== headers.length) {
      throw new Error(
        `CSV record ${recordIndex + 2} has ${values.length} fields; expected ${headers.length}.`,
      );
    }
    return Object.fromEntries(headers.map((header, columnIndex) => [header, values[columnIndex]]));
  });

  if (records.length === 0 && !allowEmptyRecords) {
    throw new Error('CSV input contains a header but no frame rows.');
  }
  return { headers, records };
}

function requireColumns(headers) {
  const available = new Set(headers);
  const missing = REQUIRED_COLUMNS.filter((column) => !available.has(column));
  if (missing.length > 0) throw new Error(`CSV is missing required columns: ${missing.join(', ')}.`);
}

function finiteNumber(value, field, recordNumber, { minimum = -Infinity, maximum = Infinity } = {}) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Record ${recordNumber} has no ${field}.`);
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`Record ${recordNumber} has non-finite ${field}: ${JSON.stringify(value)}.`);
  }
  if (number < minimum || number > maximum) {
    throw new Error(
      `Record ${recordNumber} has out-of-range ${field}: ${number}; expected ${minimum}..${maximum}.`,
    );
  }
  return number;
}

function optionalBoolean(value, field, recordNumber) {
  if (value === undefined || value.trim() === '') return null;
  const normalized = value.trim().toLowerCase();
  if (normalized === 'true' || normalized === '1') return true;
  if (normalized === 'false' || normalized === '0') return false;
  throw new Error(`Record ${recordNumber} has invalid ${field}: ${JSON.stringify(value)}.`);
}

function optionalNonnegativeInteger(value, field, recordNumber) {
  if (value === undefined || value.trim() === '') return null;
  const number = Number(value);
  if (!Number.isInteger(number) || number < 0) {
    throw new Error(`Record ${recordNumber} has invalid ${field}: ${JSON.stringify(value)}.`);
  }
  return number;
}

function optionalNonemptyString(value, field, recordNumber) {
  if (value === undefined || value.trim() === '') return null;
  const normalized = value.trim();
  if (normalized.length === 0) {
    throw new Error(`Record ${recordNumber} has invalid ${field}: ${JSON.stringify(value)}.`);
  }
  return normalized;
}

function normalizedRepetition(value, recordNumber) {
  const text = value.trim();
  if (text.length === 0) throw new Error(`Record ${recordNumber} has no repetition identifier.`);
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : text;
}

function repetitionKey(value) {
  return `${typeof value}:${String(value)}`;
}

function nearestRank(values, fraction = 0.5) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(sorted.length * fraction) - 1),
  );
  return sorted[index];
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[midpoint - 1] + sorted[midpoint]) / 2
    : sorted[midpoint];
}

function compareRepetitions(left, right) {
  if (typeof left === 'number' && typeof right === 'number') return left - right;
  return String(left).localeCompare(String(right), undefined, { numeric: true });
}

function parseFrameRecords(parsed) {
  requireColumns(parsed.headers);
  const headerSet = new Set(parsed.headers);
  const repetitionColumn = REPETITION_COLUMNS.find((column) => headerSet.has(column)) ?? null;
  const hasModeOrderPosition = headerSet.has('modeOrderPosition');
  const hasPlannedModeOrder = headerSet.has('plannedModeOrder');
  const hasScenarioLayout = headerSet.has('scenarioLayout');
  const hasLayout = headerSet.has('layout');

  const frames = parsed.records.map((record, index) => {
    const recordNumber = index + 2;
    const modeId = record.modeId.trim();
    if (modeId.length === 0) throw new Error(`Record ${recordNumber} has no modeId.`);

    const targetVisibilityFraction = finiteNumber(
      record.targetVisibilityFraction,
      'targetVisibilityFraction',
      recordNumber,
      { minimum: 0, maximum: 1 },
    );
    const usesComputeValue = optionalBoolean(record.usesCompute, 'usesCompute', recordNumber);
    const usesCompute = usesComputeValue ?? modeId !== 'draw-all';
    const computeText = record.gpuComputeMs.trim();
    let gpuComputeMs;
    if (computeText === '') {
      if (usesCompute) {
        throw new Error(`Record ${recordNumber} is a compute mode but has no gpuComputeMs.`);
      }
      gpuComputeMs = 0;
    } else {
      gpuComputeMs = finiteNumber(computeText, 'gpuComputeMs', recordNumber, { minimum: 0 });
    }
    const cpuComputeText = record.cpuComputeSubmitMs.trim();
    let cpuComputeSubmitMs;
    if (cpuComputeText === '') {
      if (usesCompute) {
        throw new Error(`Record ${recordNumber} is a compute mode but has no cpuComputeSubmitMs.`);
      }
      cpuComputeSubmitMs = 0;
    } else {
      cpuComputeSubmitMs = finiteNumber(
        cpuComputeText,
        'cpuComputeSubmitMs',
        recordNumber,
        { minimum: 0 },
      );
    }

    const repetition = repetitionColumn === null
      ? 1
      : normalizedRepetition(record[repetitionColumn], recordNumber);
    const scenarioLayout = hasScenarioLayout
      ? optionalNonemptyString(record.scenarioLayout, 'scenarioLayout', recordNumber)
      : null;
    const layout = hasLayout
      ? optionalNonemptyString(record.layout, 'layout', recordNumber)
      : null;
    if (scenarioLayout !== null && layout !== null && scenarioLayout !== layout) {
      throw new Error(`Record ${recordNumber} has inconsistent scenarioLayout and layout values.`);
    }

    const gpuPassTotalMs = finiteNumber(
      record.gpuPassTotalMs,
      'gpuPassTotalMs',
      recordNumber,
      { minimum: 0 },
    );
    const cpuSubmitTotalMs = finiteNumber(
      record.cpuSubmitTotalMs,
      'cpuSubmitTotalMs',
      recordNumber,
      { minimum: 0 },
    );

    return {
      modeId,
      targetVisibilityFraction,
      layout: scenarioLayout ?? layout,
      repetition,
      modeOrderPosition: hasModeOrderPosition
        ? optionalNonnegativeInteger(record.modeOrderPosition, 'modeOrderPosition', recordNumber)
        : null,
      plannedModeOrder: hasPlannedModeOrder
        ? optionalNonemptyString(record.plannedModeOrder, 'plannedModeOrder', recordNumber)
        : null,
      usesCompute,
      gpuPassTotalMs,
      gpuComputeMs,
      gpuRenderMs: finiteNumber(record.gpuRenderMs, 'gpuRenderMs', recordNumber, { minimum: 0 }),
      cpuCommonUpdateMs: finiteNumber(
        record.cpuCommonUpdateMs,
        'cpuCommonUpdateMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuComputeSubmitMs,
      cpuRenderSubmitMs: finiteNumber(
        record.cpuRenderSubmitMs,
        'cpuRenderSubmitMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuFrameBodyMs: finiteNumber(
        record.cpuFrameBodyMs,
        'cpuFrameBodyMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuSubmitTotalMs,
      accountedCpuSubmitPlusGpuPassMs: cpuSubmitTotalMs + gpuPassTotalMs,
    };
  });

  return { frames, repetitionColumn };
}

function frozenCrossoverScheduleRecord(orientationOffset) {
  const phase = (frameCount) => Array.from({ length: frameCount }, (_, phaseFrameIndex) => {
    const scheduled = frozenCrossoverFrame(phaseFrameIndex, orientationOffset);
    return {
      phaseFrameIndex,
      crossoverBlockIndex: scheduled.crossoverBlockIndex,
      withinBlockPosition: scheduled.withinBlockPosition,
      patternIndex: scheduled.patternIndex,
      pattern: scheduled.pattern,
      laneId: scheduled.laneId,
    };
  });
  return {
    schemaVersion: 1,
    kind: 'frozen-depth-crossover-frame-schedule',
    blockSize: FROZEN_CROSSOVER_BLOCK_SIZE,
    warmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
    orientationOffset,
    warmup: phase(FROZEN_CROSSOVER_WARMUP_FRAMES),
    measured: phase(FROZEN_CROSSOVER_MEASURED_FRAMES),
  };
}

function frozenCrossoverScheduleSha256(orientationOffset) {
  return sha256Json(frozenCrossoverScheduleRecord(orientationOffset));
}

function csvInteger(value, field, recordNumber, { minimum = -Infinity, maximum = Infinity } = {}) {
  const number = finiteNumber(value, field, recordNumber, { minimum, maximum });
  if (!Number.isInteger(number)) {
    throw new Error(`Record ${recordNumber} has non-integer ${field}: ${JSON.stringify(value)}.`);
  }
  return number;
}

function isFrozenCrossoverCsv(parsed) {
  const headers = new Set(parsed.headers);
  return parsed.records.some((record) => record.modeId === FROZEN_DEPTH_CROSSOVER_MODE)
    || [
      'crossoverBlockIndex',
      'plannedLaneStorageOrder',
      'plannedScheduleSha256',
      'superblockOrientationOffset',
    ].some((header) => headers.has(header));
}

function parseFrozenCrossoverRecords(parsed) {
  const headers = new Set(parsed.headers);
  const missing = FROZEN_CROSSOVER_REQUIRED_COLUMNS.filter((column) => !headers.has(column));
  if (missing.length > 0) {
    throw new Error(`Frozen crossover CSV is missing required columns: ${missing.join(', ')}.`);
  }
  if (parsed.records.some((record) => record.modeId !== FROZEN_DEPTH_CROSSOVER_MODE)) {
    throw new Error('Frozen crossover CSV cannot mix frozen and non-frozen modes.');
  }

  const expectedScheduleSha256 = [0, 1].map(frozenCrossoverScheduleSha256);
  const seenGpuFrameIds = new Set();
  const normalized = parsed.records.map((record, index) => {
    const recordNumber = index + 2;
    const integer = (field, limits = {}) => csvInteger(
      record[field],
      field,
      recordNumber,
      limits,
    );
    const repetitionIndex = integer('repetitionIndex', {
      minimum: 0,
      maximum: FROZEN_DEPTH_CROSSOVER_REPETITIONS - 1,
    });
    const frameIndex = integer('frameIndex', {
      minimum: 0,
      maximum: FROZEN_CROSSOVER_MEASURED_FRAMES - 1,
    });
    const phaseFrameIndex = integer('phaseFrameIndex', {
      minimum: 0,
      maximum: FROZEN_CROSSOVER_MEASURED_FRAMES - 1,
    });
    const layoutOrderPosition = integer('layoutOrderPosition', { minimum: 0, maximum: 1 });
    const orientationOffset = integer('superblockOrientationOffset', {
      minimum: 0,
      maximum: 1,
    });
    const gpuFrameId = integer('gpuFrameId', { minimum: 0 });
    if (seenGpuFrameIds.has(gpuFrameId)) {
      throw new Error(`Record ${recordNumber} duplicates gpuFrameId ${gpuFrameId}.`);
    }
    seenGpuFrameIds.add(gpuFrameId);

    const gpuRenderMs = finiteNumber(
      record.gpuRenderMs,
      'gpuRenderMs',
      recordNumber,
      { minimum: Number.MIN_VALUE },
    );
    const gpuPassTotalMs = finiteNumber(
      record.gpuPassTotalMs,
      'gpuPassTotalMs',
      recordNumber,
      { minimum: Number.MIN_VALUE },
    );
    if (gpuPassTotalMs !== gpuRenderMs) {
      throw new Error(`Record ${recordNumber} gpuPassTotalMs must equal gpuRenderMs.`);
    }
    if (record.gpuComputeMs !== '' || record.cpuComputeSubmitMs !== '') {
      throw new Error(`Record ${recordNumber} contains an unexpected compute duration.`);
    }
    if (record.usesCompute !== 'false') {
      throw new Error(`Record ${recordNumber} must record usesCompute=false.`);
    }
    if (record.validationPass !== 'true' || record.timestampAvailable !== 'true') {
      throw new Error(`Record ${recordNumber} lacks accepted validation or GPU timestamps.`);
    }
    if (record.validationKind !== FROZEN_DEPTH_CROSSOVER_VALIDATION_KIND) {
      throw new Error(`Record ${recordNumber} has the wrong frozen validation kind.`);
    }
    const expectedVisibleCount = Math.round(
      FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT * FROZEN_DEPTH_CROSSOVER_VISIBILITIES[0],
    );
    if (finiteNumber(
      record.targetVisibilityFraction,
      'targetVisibilityFraction',
      recordNumber,
    ) !== FROZEN_DEPTH_CROSSOVER_VISIBILITIES[0]
      || integer('modeOrderPosition') !== 0
      || integer('visibilityOrderPosition') !== 0
      || record.plannedModeOrder !== FROZEN_DEPTH_CROSSOVER_MODE
      || record.plannedVisibilityOrder !== String(FROZEN_DEPTH_CROSSOVER_VISIBILITIES[0])
      || integer('protocolWarmupFrames') !== FROZEN_CROSSOVER_WARMUP_FRAMES
      || integer('protocolMeasuredFrames') !== FROZEN_CROSSOVER_MEASURED_FRAMES
      || integer('objectCount') !== FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT
      || integer('bucketCount') !== FROZEN_DEPTH_CROSSOVER_BUCKET_COUNT
      || integer('expectedVisibleCount') !== expectedVisibleCount
      || integer('configuredDrawCommands') !== FROZEN_DEPTH_CROSSOVER_BUCKET_COUNT
      || integer('configuredRenderObjects') !== 1
      || integer('configuredComputeDispatches') !== 0
      || integer('configuredComputeSubmissions') !== 0
      || integer('configuredSubmittedInstances') !== expectedVisibleCount
      || integer('bundleRecordCallbackCountAtTimingStart') !== 1) {
      throw new Error(`Record ${recordNumber} has the wrong frozen workload or render-only shape.`);
    }
    if (integer('gpuRenderTimestampUidCount') !== 1
      || integer('expectedRenderTimestampUidCount') !== 1) {
      throw new Error(`Record ${recordNumber} must contain exactly one render timestamp UID.`);
    }
    if (record.plannedScheduleSha256 !== expectedScheduleSha256[orientationOffset]) {
      throw new Error(`Record ${recordNumber} has the wrong frozen schedule commitment.`);
    }
    const stringIdentityFields = [
      'bundleGroupUuidAtTimingStart',
      'meshUuidAtTimingStart',
      'geometryUuidAtTimingStart',
      'materialUuidAtTimingStart',
      'selectorUniformUuidAtTimingStart',
      'renderTargetTextureUuidAtTimingStart',
    ];
    if (stringIdentityFields.some((field) => record[field].trim() === '')
      || !/^[0-9a-f]{16}$/.test(record.cameraViewFnv64AtTimingStart)
      || !/^[0-9a-f]{16}$/.test(record.cameraProjectionFnv64AtTimingStart)
      || integer('renderTargetWidthAtTimingStart') !== 1280
      || integer('renderTargetHeightAtTimingStart') !== 720
      || integer('renderTargetSamplesAtTimingStart') !== 0
      || record.renderTargetDepthBufferAtTimingStart !== 'true') {
      throw new Error(`Record ${recordNumber} has an invalid frozen resource, target, or camera identity.`);
    }
    const selectorWriteSerialAtTimingStart = integer(
      'selectorWriteSerialAtTimingStart',
      { minimum: 0 },
    );
    const renderCallSerialAtTimingStart = integer(
      'renderCallSerialAtTimingStart',
      { minimum: 0 },
    );
    const computeCallSerialAtTimingStart = integer(
      'computeCallSerialAtTimingStart',
      { minimum: 0 },
    );
    const totalPipelineCacheEntriesAtTimingStart = integer(
      'totalPipelineCacheEntriesAtTimingStart',
      { minimum: 0 },
    );
    const computePipelineCacheEntriesAtTimingStart = integer(
      'computePipelineCacheEntriesAtTimingStart',
      { minimum: 0 },
    );
    if (computePipelineCacheEntriesAtTimingStart
      > totalPipelineCacheEntriesAtTimingStart) {
      throw new Error(`Record ${recordNumber} has impossible frozen pipeline-cache counts.`);
    }
    const numericIdentity = Object.fromEntries([
      'matrixAttributeIdAtTimingStart',
      'visibleIdsAttributeIdAtTimingStart',
      'indirectAttributeIdAtTimingStart',
      'selectorChallengeAttributeIdAtTimingStart',
      'bundleGroupVersionAtTimingStart',
      'matrixAttributeVersionAtTimingStart',
      'visibleIdsAttributeVersionAtTimingStart',
      'indirectAttributeVersionAtTimingStart',
    ].map((field) => [field, integer(field, { minimum: 0 })]));

    return {
      runId: record.runId,
      trialId: record.trialId,
      planIndex: integer('planIndex', { minimum: 0, maximum: 23 }),
      repetitionIndex,
      frameIndex,
      phaseFrameIndex,
      layout: record.scenarioLayout,
      layoutOrderPosition,
      plannedLayoutOrder: record.plannedLayoutOrder,
      laneStorageOrder: record.plannedLaneStorageOrder,
      frontLaneBase: integer('frontLaneBase', {
        minimum: 0,
        maximum: FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT,
      }),
      reverseLaneBase: integer('reverseLaneBase', {
        minimum: 0,
        maximum: FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT,
      }),
      superblockOrientationOffset: orientationOffset,
      plannedScheduleSha256: record.plannedScheduleSha256,
      crossoverBlockIndex: integer('crossoverBlockIndex', {
        minimum: 0,
        maximum: FROZEN_CROSSOVER_MEASURED_BLOCKS - 1,
      }),
      withinBlockPosition: integer('withinBlockPosition', {
        minimum: 0,
        maximum: FROZEN_CROSSOVER_BLOCK_SIZE - 1,
      }),
      crossoverPattern: record.crossoverPattern,
      crossoverPatternIndex: integer('crossoverPatternIndex', { minimum: 0, maximum: 1 }),
      laneId: record.laneId,
      laneBase: integer('laneBase', {
        minimum: 0,
        maximum: FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT,
      }),
      selectorWriteSerial: integer('selectorWriteSerial', { minimum: 1 }),
      renderCallSerial: integer('renderCallSerial', { minimum: 1 }),
      gpuFrameId,
      gpuRenderMs,
      selectorWriteSerialAtTimingStart,
      renderCallSerialAtTimingStart,
      computeCallSerialAtTimingStart,
      totalPipelineCacheEntriesAtTimingStart,
      computePipelineCacheEntriesAtTimingStart,
      bundleGroupUuidAtTimingStart: record.bundleGroupUuidAtTimingStart,
      meshUuidAtTimingStart: record.meshUuidAtTimingStart,
      geometryUuidAtTimingStart: record.geometryUuidAtTimingStart,
      materialUuidAtTimingStart: record.materialUuidAtTimingStart,
      ...numericIdentity,
      selectorUniformUuidAtTimingStart: record.selectorUniformUuidAtTimingStart,
      renderTargetTextureUuidAtTimingStart: record.renderTargetTextureUuidAtTimingStart,
      renderTargetWidthAtTimingStart: 1280,
      renderTargetHeightAtTimingStart: 720,
      renderTargetSamplesAtTimingStart: 0,
      renderTargetDepthBufferAtTimingStart: true,
      cameraViewFnv64AtTimingStart: record.cameraViewFnv64AtTimingStart,
      cameraProjectionFnv64AtTimingStart: record.cameraProjectionFnv64AtTimingStart,
    };
  });

  const runIds = new Set(normalized.map((row) => row.runId));
  if (runIds.size !== 1 || [...runIds][0] === '') {
    throw new Error('Frozen crossover CSV must contain exactly one nonempty runId.');
  }
  const byTrial = new Map();
  for (const row of normalized) {
    let rows = byTrial.get(row.trialId);
    if (rows === undefined) {
      rows = [];
      byTrial.set(row.trialId, rows);
    }
    rows.push(row);
  }
  for (const [trialId, rows] of byTrial) {
    if (trialId === '') throw new Error('Frozen crossover CSV contains an empty trialId.');
    const first = rows[0];
    const expectedLayoutOrder = first.repetitionIndex % 2 === 0
      ? FROZEN_DEPTH_CROSSOVER_LAYOUTS
      : [...FROZEN_DEPTH_CROSSOVER_LAYOUTS].reverse();
    const expectedStorageOrder = FROZEN_DEPTH_CROSSOVER_STORAGE_ORDERS[first.repetitionIndex];
    const expectedFrontLaneBase = expectedStorageOrder.indexOf(
      FROZEN_DEPTH_CROSSOVER_LANES[0],
    ) * FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT;
    if (first.planIndex !== first.repetitionIndex * 2 + first.layoutOrderPosition
      || first.layout !== expectedLayoutOrder[first.layoutOrderPosition]
      || first.plannedLayoutOrder !== expectedLayoutOrder.join('|')
      || first.laneStorageOrder !== expectedStorageOrder.join('|')
      || first.frontLaneBase !== expectedFrontLaneBase
      || first.reverseLaneBase
        !== (expectedFrontLaneBase === 0 ? FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT : 0)
      || first.superblockOrientationOffset
        !== FROZEN_DEPTH_CROSSOVER_ORIENTATION_OFFSETS[first.repetitionIndex]) {
      throw new Error(`Frozen crossover trial ${JSON.stringify(trialId)} differs from the committed plan.`);
    }
    for (const [frameIndex, row] of rows.entries()) {
      const scheduled = frozenCrossoverFrame(
        frameIndex,
        first.superblockOrientationOffset,
      );
      if (row.planIndex !== first.planIndex
        || row.repetitionIndex !== first.repetitionIndex
        || row.layout !== first.layout
        || row.layoutOrderPosition !== first.layoutOrderPosition
        || row.plannedLayoutOrder !== first.plannedLayoutOrder
        || row.laneStorageOrder !== first.laneStorageOrder
        || row.frontLaneBase !== first.frontLaneBase
        || row.reverseLaneBase !== first.reverseLaneBase
        || row.superblockOrientationOffset !== first.superblockOrientationOffset
        || row.plannedScheduleSha256 !== first.plannedScheduleSha256
        || row.selectorWriteSerialAtTimingStart
          !== first.selectorWriteSerialAtTimingStart
        || row.renderCallSerialAtTimingStart !== first.renderCallSerialAtTimingStart
        || row.computeCallSerialAtTimingStart !== first.computeCallSerialAtTimingStart
        || row.totalPipelineCacheEntriesAtTimingStart
          !== first.totalPipelineCacheEntriesAtTimingStart
        || row.computePipelineCacheEntriesAtTimingStart
          !== first.computePipelineCacheEntriesAtTimingStart
        || FROZEN_CROSSOVER_ROW_IDENTITY_FIELDS.some(
          (field) => row[field] !== first[field],
        )
        || row.frameIndex !== frameIndex
        || row.phaseFrameIndex !== frameIndex
        || row.crossoverPatternIndex !== scheduled.patternIndex
        || row.selectorWriteSerial !== row.selectorWriteSerialAtTimingStart
          + FROZEN_CROSSOVER_WARMUP_FRAMES + frameIndex + 1
        || row.renderCallSerial !== row.renderCallSerialAtTimingStart
          + FROZEN_CROSSOVER_WARMUP_FRAMES + frameIndex + 1) {
        throw new Error(`Frozen crossover trial ${JSON.stringify(trialId)} has inconsistent frame audit fields.`);
      }
    }
  }
  return normalized;
}

function isLiveFirstInstanceCrossoverCsv(parsed) {
  const headers = new Set(parsed.headers);
  return parsed.records.some(
    (record) => record.modeId === FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  )
    || headers.has('plannedLanePhysicalOrder')
    || headers.has('portableCommandBufferIdAtTimingStart')
    || headers.has('gpuComputeTimestampUidCount');
}

function requireCanonicalJsonObjectColumn(record, field, recordNumber) {
  const raw = record[field];
  if (typeof raw !== 'string' || raw.trim() === '') {
    throw new Error(`Record ${recordNumber} has no ${field}.`);
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`Record ${recordNumber} has invalid JSON in ${field}.`);
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)
    || JSON.stringify(parsed) !== raw) {
    throw new Error(
      `Record ${recordNumber} has a non-canonical JSON object in ${field}.`,
    );
  }
}

function requireCanonicalJsonArrayColumn(record, field, recordNumber) {
  const raw = record[field];
  if (typeof raw !== 'string' || raw.trim() === '') {
    throw new Error(`Record ${recordNumber} has no ${field}.`);
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`Record ${recordNumber} has invalid JSON in ${field}.`);
  }
  if (!Array.isArray(parsed) || JSON.stringify(parsed) !== raw) {
    throw new Error(
      `Record ${recordNumber} has a non-canonical JSON array in ${field}.`,
    );
  }
}

function validateParsedLiveTimestampRecord(record, {
  type,
  frameId,
  callIndex,
  contextId,
  durationMs,
  recordNumber,
}) {
  const label = `Record ${recordNumber} ${type} timestamp`;
  const keys = ['uid', 'type', 'callIndex', 'contextId', 'frameId', 'durationMs'];
  if (record === null || typeof record !== 'object' || Array.isArray(record)
    || JSON.stringify(Object.keys(record).sort()) !== JSON.stringify([...keys].sort())) {
    throw new Error(`${label} has fields outside the exact UID-record schema.`);
  }
  if (record.type !== type
    || !Number.isSafeInteger(record.callIndex) || record.callIndex <= 0
    || record.callIndex !== callIndex
    || !Number.isSafeInteger(record.contextId) || record.contextId < 0
    || (contextId !== undefined && record.contextId !== contextId)
    || record.frameId !== frameId
    || !Number.isFinite(record.durationMs) || record.durationMs <= 0
    || !Object.is(record.durationMs, durationMs)) {
    throw new Error(`${label} contextId/type/call/frame/duration binding failed.`);
  }
  const prefix = type === 'compute' ? 'c' : 'r';
  const expectedUid = `${prefix}:${record.callIndex}:${record.contextId}:f${record.frameId}`;
  if (record.uid !== expectedUid) {
    throw new Error(`${label} UID is ${JSON.stringify(record.uid)}; expected ${expectedUid}.`);
  }
}

function validateParsedLiveTimestampRow(row, recordNumber) {
  if (row.strictTimestampUidAttribution !== true
    || row.gpuComputeTimestampDurationValid !== true
    || row.gpuRenderTimestampDurationValid !== true
    || row.submittedComputeLaneId !== row.laneId
    || !Number.isSafeInteger(row.computeTimestampContextId)
    || row.computeTimestampContextId <= 0
    || row.computeFrameCallIndex !== 1
    || row.renderFrameCallIndex !== 1) {
    throw new Error(`Record ${recordNumber} lacks exact live timestamp attribution.`);
  }
  for (const field of [
    'computeGroupIdentity', 'computeTimestampRegistrationSerial',
    'computeTimestampBackendIdentity', 'computeTimestampBackendWrapperIdentity',
  ]) {
    if (!Number.isSafeInteger(row[field]) || row[field] <= 0) {
      throw new Error(`Record ${recordNumber} has invalid ${field}.`);
    }
  }
  const nodeIds = JSON.parse(row.submittedComputeNodeIds);
  if (nodeIds.length !== 2 || nodeIds.some(
    (value) => !((typeof value === 'string' && value.length > 0)
      || (Number.isSafeInteger(value) && value >= 0)),
  )) throw new Error(`Record ${recordNumber} has invalid submitted compute-node IDs.`);
  for (const type of ['compute', 'render']) {
    const title = type === 'compute' ? 'Compute' : 'Render';
    const uids = JSON.parse(row[`gpu${title}TimestampUids`]);
    const records = JSON.parse(row[`gpu${title}TimestampRecords`]);
    if (uids.length !== 1 || records.length !== 1 || uids[0] !== records[0]?.uid) {
      throw new Error(`Record ${recordNumber} ${type} timestamp UID join is not exact.`);
    }
    validateParsedLiveTimestampRecord(records[0], {
      type,
      frameId: row.gpuFrameId,
      callIndex: row[`${type}FrameCallIndex`],
      contextId: type === 'compute' ? row.computeTimestampContextId : undefined,
      durationMs: row[`gpu${title}Ms`],
      recordNumber,
    });
  }
}

function parseLiveFirstInstanceCrossoverRecords(parsed) {
  const headers = new Set(parsed.headers);
  const missing = LIVE_FIRST_INSTANCE_CROSSOVER_REQUIRED_COLUMNS.filter(
    (column) => !headers.has(column),
  );
  if (missing.length > 0) {
    throw new Error(
      `Live first-instance crossover CSV is missing required columns: ${missing.join(', ')}.`,
    );
  }
  if (parsed.records.some(
    (record) => record.modeId !== FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  )) {
    throw new Error(
      'Live first-instance crossover CSV cannot mix crossover and non-crossover modes.',
    );
  }

  return parsed.records.map((record, index) => {
    const recordNumber = index + 2;
    const row = { ...record };
    for (const field of LIVE_FIRST_INSTANCE_CROSSOVER_SAFE_INTEGER_COLUMNS) {
      const value = Number(record[field]);
      if (record[field]?.trim() === '' || !Number.isSafeInteger(value) || value < 0) {
        throw new Error(
          `Record ${recordNumber} has invalid safe integer ${field}: ${JSON.stringify(record[field])}.`,
        );
      }
      row[field] = value;
    }
    for (const field of LIVE_FIRST_INSTANCE_CROSSOVER_NUMBER_COLUMNS) {
      row[field] = finiteNumber(record[field], field, recordNumber, { minimum: 0 });
    }
    for (const field of [
      'validationPass',
      'usesCompute',
      'timestampAvailable',
      'strictTimestampUidAttribution',
      'gpuComputeTimestampDurationValid',
      'gpuRenderTimestampDurationValid',
    ]) {
      const value = optionalBoolean(record[field], field, recordNumber);
      if (value === null) throw new Error(`Record ${recordNumber} has no ${field}.`);
      row[field] = value;
    }
    const depthBuffer = optionalBoolean(
      record.renderTargetDepthBufferAtTimingStart,
      'renderTargetDepthBufferAtTimingStart',
      recordNumber,
    );
    if (depthBuffer === null) {
      throw new Error(
        `Record ${recordNumber} has no renderTargetDepthBufferAtTimingStart.`,
      );
    }
    row.renderTargetDepthBufferAtTimingStart = depthBuffer;
    for (const field of [
      'runId',
      'trialId',
      'plannedModeOrder',
      'plannedVisibilityOrder',
      'plannedLayoutOrder',
      'scenarioLayout',
      'validationKind',
      'plannedLanePhysicalOrder',
      'lanePhysicalOrder',
      'plannedScheduleSha256',
      'lifecycleCommitmentAtTimingStart',
      'commandBufferCommitmentsAtTimingStart',
      'rendererMemoryAtTimingStart',
      'viewportStateAtTimingStart',
      'renderTargetTextureUuidAtTimingStart',
      'cameraViewFnv64AtTimingStart',
      'cameraProjectionFnv64AtTimingStart',
      'timestampPoolStaticCommitmentAtTimingStart',
      'crossoverPattern',
      'previousPreviousLaneId',
      'previousLaneId',
      'laneId',
      'submittedComputeLaneId',
      'submittedComputeNodeIds',
      'gpuComputeTimestampUids',
      'gpuComputeTimestampRecords',
      'gpuRenderTimestampUids',
      'gpuRenderTimestampRecords',
    ]) {
      if (typeof record[field] !== 'string' || record[field].trim() === '') {
        throw new Error(`Record ${recordNumber} has no ${field}.`);
      }
    }
    for (const field of [
      'rendererMemoryAtTimingStart',
      'viewportStateAtTimingStart',
    ]) requireCanonicalJsonObjectColumn(record, field, recordNumber);
    for (const field of [
      'submittedComputeNodeIds',
      'gpuComputeTimestampUids',
      'gpuComputeTimestampRecords',
      'gpuRenderTimestampUids',
      'gpuRenderTimestampRecords',
    ]) requireCanonicalJsonArrayColumn(record, field, recordNumber);
    validateParsedLiveTimestampRow(row, recordNumber);
    return row;
  });
}

function isFirstInstanceCrossoverCsv(parsed) {
  const headers = new Set(parsed.headers);
  return !isLiveFirstInstanceCrossoverCsv(parsed)
    && (parsed.records.some((record) => record.modeId === FIRST_INSTANCE_CROSSOVER_MODE)
      || headers.has('plannedLaneCommandSegmentOrder')
      || headers.has('commandSegmentIndex')
      || headers.has('strategySelectionSerial'));
}

function parseFirstInstanceCrossoverRecords(parsed) {
  const headers = new Set(parsed.headers);
  const missing = FIRST_INSTANCE_CROSSOVER_REQUIRED_COLUMNS.filter(
    (column) => !headers.has(column),
  );
  if (missing.length > 0) {
    throw new Error(
      `First-instance crossover CSV is missing required columns: ${missing.join(', ')}.`,
    );
  }
  if (parsed.records.some((record) => record.modeId !== FIRST_INSTANCE_CROSSOVER_MODE)) {
    throw new Error('First-instance crossover CSV cannot mix crossover and non-crossover modes.');
  }

  return parsed.records.map((record, index) => {
    const recordNumber = index + 2;
    const row = { ...record };
    for (const field of FIRST_INSTANCE_CROSSOVER_SAFE_INTEGER_COLUMNS) {
      const value = Number(record[field]);
      if (record[field]?.trim() === '' || !Number.isSafeInteger(value) || value < 0) {
        throw new Error(
          `Record ${recordNumber} has invalid safe integer ${field}: ${JSON.stringify(record[field])}.`,
        );
      }
      row[field] = value;
    }
    for (const field of FIRST_INSTANCE_CROSSOVER_NULLABLE_SAFE_INTEGER_COLUMNS) {
      if (record[field] === '') {
        row[field] = null;
        continue;
      }
      const value = Number(record[field]);
      if (!Number.isSafeInteger(value) || value < 0) {
        throw new Error(
          `Record ${recordNumber} has invalid nullable safe integer ${field}: ${JSON.stringify(record[field])}.`,
        );
      }
      row[field] = value;
    }
    for (const field of FIRST_INSTANCE_CROSSOVER_NUMBER_COLUMNS) {
      row[field] = finiteNumber(record[field], field, recordNumber, { minimum: 0 });
    }
    for (const field of ['validationPass', 'usesCompute', 'timestampAvailable']) {
      const value = optionalBoolean(record[field], field, recordNumber);
      if (value === null) throw new Error(`Record ${recordNumber} has no ${field}.`);
      row[field] = value;
    }
    const depthBuffer = optionalBoolean(
      record.renderTargetDepthBufferAtTimingStart,
      'renderTargetDepthBufferAtTimingStart',
      recordNumber,
    );
    if (depthBuffer === null) {
      throw new Error(
        `Record ${recordNumber} has no renderTargetDepthBufferAtTimingStart.`,
      );
    }
    row.renderTargetDepthBufferAtTimingStart = depthBuffer;
    for (const field of ['cpuComputeSubmitMs', 'gpuComputeMs']) {
      if (record[field] !== '') {
        throw new Error(`Record ${recordNumber} contains an unexpected ${field}.`);
      }
      row[field] = null;
    }
    for (const field of [
      'runId',
      'trialId',
      'plannedModeOrder',
      'plannedVisibilityOrder',
      'plannedLayoutOrder',
      'scenarioLayout',
      'validationKind',
      'plannedLaneCommandSegmentOrder',
      'plannedCommandSegments',
      'plannedScheduleSha256',
      'lifecycleCommitmentAtTimingStart',
      'renderTargetTextureUuidAtTimingStart',
      'cameraViewFnv64AtTimingStart',
      'cameraProjectionFnv64AtTimingStart',
      'crossoverPattern',
      'laneId',
    ]) {
      if (typeof record[field] !== 'string' || record[field].trim() === '') {
        throw new Error(`Record ${recordNumber} has no ${field}.`);
      }
    }
    return row;
  });
}

function metricP50(frames) {
  return Object.fromEntries(METRICS.map((metric) => [
    metric,
    nearestRank(frames.map((frame) => frame[metric])),
  ]));
}

function groupFrames(frames) {
  const groups = new Map();
  for (const frame of frames) {
    const groupKey = `${frame.modeId}\u0000${frame.targetVisibilityFraction}\u0000${frame.layout ?? ''}`;
    let group = groups.get(groupKey);
    if (group === undefined) {
      group = {
        modeId: frame.modeId,
        targetVisibilityFraction: frame.targetVisibilityFraction,
        layout: frame.layout,
        frames: [],
        trials: new Map(),
      };
      groups.set(groupKey, group);
    }
    group.frames.push(frame);

    const key = repetitionKey(frame.repetition);
    let trial = group.trials.get(key);
    if (trial === undefined) {
      trial = {
        repetition: frame.repetition,
        usesCompute: frame.usesCompute,
        modeOrderPosition: frame.modeOrderPosition,
        plannedModeOrder: frame.plannedModeOrder,
        frames: [],
      };
      group.trials.set(key, trial);
    } else if (trial.usesCompute !== frame.usesCompute) {
      throw new Error(
        `Mode ${frame.modeId}, visibility ${frame.targetVisibilityFraction}, layout ${frame.layout ?? '(unspecified)'}, repetition ${frame.repetition} mixes usesCompute values.`,
      );
    } else if (trial.modeOrderPosition !== frame.modeOrderPosition
      || trial.plannedModeOrder !== frame.plannedModeOrder) {
      throw new Error(
        `Mode ${frame.modeId}, visibility ${frame.targetVisibilityFraction}, layout ${frame.layout ?? '(unspecified)'}, repetition ${frame.repetition} mixes order audit values.`,
      );
    }
    trial.frames.push(frame);
  }

  return [...groups.values()]
    .sort((left, right) => (
      left.modeId.localeCompare(right.modeId)
      || left.targetVisibilityFraction - right.targetVisibilityFraction
      || String(left.layout ?? '').localeCompare(String(right.layout ?? ''))
    ))
    .map((group) => {
      const perTrialP50 = [...group.trials.values()]
        .sort((left, right) => compareRepetitions(left.repetition, right.repetition))
        .map((trial) => ({
          repetition: trial.repetition,
          modeOrderPosition: trial.modeOrderPosition,
          plannedModeOrder: trial.plannedModeOrder,
          nFrames: trial.frames.length,
          p50: metricP50(trial.frames),
        }));
      const medianAcrossTrials = Object.fromEntries(METRICS.map((metric) => [
        metric,
        median(perTrialP50.map((trial) => trial.p50[metric])),
      ]));

      return {
        modeId: group.modeId,
        targetVisibilityFraction: group.targetVisibilityFraction,
        ...(group.layout === null ? {} : { layout: group.layout }),
        nTrials: perTrialP50.length,
        nFrames: group.frames.length,
        perTrialP50,
        medianAcrossTrials,
      };
    });
}

function metricDelta(fixedValue, baselineValue) {
  const absoluteMs = fixedValue - baselineValue;
  let percent = null;
  if (baselineValue !== 0) percent = (absoluteMs / baselineValue) * 100;
  else if (fixedValue === 0) percent = 0;
  return { absoluteMs, percent };
}

function pairedDelta(fixedP50, baselineP50) {
  return Object.fromEntries(METRICS.map((metric) => [
    metric,
    metricDelta(fixedP50[metric], baselineP50[metric]),
  ]));
}

function summarizePairedDeltas(pairs) {
  return Object.fromEntries(METRICS.map((metric) => {
    const absoluteValues = pairs.map((pair) => pair.delta[metric].absoluteMs);
    const percentValues = pairs
      .map((pair) => pair.delta[metric].percent)
      .filter(Number.isFinite);
    return [metric, {
      absoluteMs: median(absoluteValues),
      percent: median(percentValues),
    }];
  }));
}

function resolvedModeOrderPosition(trial, modeId) {
  let position = trial.modeOrderPosition;
  if (trial.plannedModeOrder !== null) {
    const order = trial.plannedModeOrder.split('|');
    const plannedPosition = order.indexOf(modeId);
    if (plannedPosition < 0 || new Set(order).size !== order.length) {
      throw new Error(
        `Mode ${modeId}, repetition ${trial.repetition} has an invalid plannedModeOrder.`,
      );
    }
    if (position !== null && position !== plannedPosition) {
      throw new Error(
        `Mode ${modeId}, repetition ${trial.repetition} has inconsistent order audit values.`,
      );
    }
    position = plannedPosition;
  }
  return position;
}

function pairedOrderAudit(leftTrial, rightTrial, leftModeId, rightModeId) {
  if (leftTrial.plannedModeOrder !== null
    && rightTrial.plannedModeOrder !== null
    && leftTrial.plannedModeOrder !== rightTrial.plannedModeOrder) {
    throw new Error(
      `Modes ${leftModeId} and ${rightModeId}, repetition ${leftTrial.repetition} have different plannedModeOrder values.`,
    );
  }
  const leftModeOrderPosition = resolvedModeOrderPosition(leftTrial, leftModeId);
  const rightModeOrderPosition = resolvedModeOrderPosition(rightTrial, rightModeId);
  if (leftModeOrderPosition === null || rightModeOrderPosition === null) {
    return {
      status: 'unavailable',
      orderStratum: null,
      leftModeOrderPosition,
      rightModeOrderPosition,
      plannedModeOrder: leftTrial.plannedModeOrder ?? rightTrial.plannedModeOrder,
    };
  }
  if (leftModeOrderPosition === rightModeOrderPosition) {
    throw new Error(
      `Modes ${leftModeId} and ${rightModeId}, repetition ${leftTrial.repetition} claim the same mode-order position.`,
    );
  }
  return {
    status: 'classified',
    orderStratum: leftModeOrderPosition < rightModeOrderPosition ? 'left-first' : 'right-first',
    leftModeOrderPosition,
    rightModeOrderPosition,
    plannedModeOrder: leftTrial.plannedModeOrder ?? rightTrial.plannedModeOrder,
  };
}

function summarizeOrderStratification(pairs) {
  const classified = pairs.filter((pair) => pair.orderAudit.status === 'classified');
  const strata = ['left-first', 'right-first'].map((orderStratum) => {
    const stratumPairs = classified.filter((pair) => pair.orderAudit.orderStratum === orderStratum);
    return {
      orderStratum,
      nPairs: stratumPairs.length,
      medianPairedDelta: summarizePairedDeltas(stratumPairs),
    };
  });
  return {
    status: pairs.length === 0 || classified.length === 0
      ? 'unavailable'
      : classified.length === pairs.length
        ? 'complete'
        : 'partial',
    classifiedPairs: classified.length,
    unclassifiedPairs: pairs.length - classified.length,
    strata,
  };
}

function pairedContrasts(groups, leftModeId, rightModeId) {
  const leftGroups = groups.filter((group) => group.modeId === leftModeId);
  const rightByCell = new Map(
    groups
      .filter((group) => group.modeId === rightModeId)
      .map((group) => [
        JSON.stringify([group.targetVisibilityFraction, group.layout]),
        group,
      ]),
  );

  return leftGroups
    .map((leftGroup) => {
      const rightGroup = rightByCell.get(JSON.stringify([
        leftGroup.targetVisibilityFraction,
        leftGroup.layout,
      ]));
      if (rightGroup === undefined) return null;
      const rightTrials = new Map(
        rightGroup.perTrialP50.map((trial) => [repetitionKey(trial.repetition), trial]),
      );
      const leftTrials = new Map(
        leftGroup.perTrialP50.map((trial) => [repetitionKey(trial.repetition), trial]),
      );
      const pairs = leftGroup.perTrialP50
        .map((leftTrial) => {
          const rightTrial = rightTrials.get(repetitionKey(leftTrial.repetition));
          if (rightTrial === undefined) return null;
          const orderAudit = pairedOrderAudit(
            leftTrial,
            rightTrial,
            leftModeId,
            rightModeId,
          );
          return {
            repetition: leftTrial.repetition,
            leftP50: leftTrial.p50,
            rightP50: rightTrial.p50,
            delta: pairedDelta(leftTrial.p50, rightTrial.p50),
            orderAudit,
          };
        })
        .filter(Boolean);
      const unmatchedLeftRepetitions = leftGroup.perTrialP50
        .filter((trial) => !rightTrials.has(repetitionKey(trial.repetition)))
        .map((trial) => trial.repetition);
      const unmatchedRightRepetitions = rightGroup.perTrialP50
        .filter((trial) => !leftTrials.has(repetitionKey(trial.repetition)))
        .map((trial) => trial.repetition);

      return {
        targetVisibilityFraction: leftGroup.targetVisibilityFraction,
        ...(leftGroup.layout === undefined ? {} : { layout: leftGroup.layout }),
        leftModeId,
        rightModeId,
        nPairs: pairs.length,
        pairs,
        medianPairedDelta: summarizePairedDeltas(pairs),
        orderStratification: summarizeOrderStratification(pairs),
        unmatchedLeftRepetitions,
        unmatchedRightRepetitions,
      };
    })
    .filter(Boolean)
    .sort((left, right) => (
      left.targetVisibilityFraction - right.targetVisibilityFraction
      || String(left.layout ?? '').localeCompare(String(right.layout ?? ''))
    ));
}

function depthContrastForLayout(contrasts, layout) {
  return contrasts.find((contrast) => contrast.layout === layout) ?? null;
}

function materialImprovementGate(contrast, metric) {
  const observed = contrast?.medianPairedDelta?.[metric] ?? null;
  const complete = contrast?.nPairs === DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS
    && Number.isFinite(observed?.absoluteMs)
    && Number.isFinite(observed?.percent);
  return {
    status: complete ? 'evaluated' : 'incomplete',
    pass: complete && (
      observed.absoluteMs <= -DEPTH_ORDERING_MATERIAL_ABSOLUTE_MS
      || observed.percent <= -DEPTH_ORDERING_MATERIAL_PERCENT
    ),
    metric,
    expectedPairs: DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS,
    observedPairs: contrast?.nPairs ?? 0,
    threshold: {
      rule: 'median paired delta must be at most either threshold',
      absoluteMs: -DEPTH_ORDERING_MATERIAL_ABSOLUTE_MS,
      percent: -DEPTH_ORDERING_MATERIAL_PERCENT,
    },
    observed,
  };
}

function noMaterialDifferenceGate(contrast, metric) {
  const observed = contrast?.medianPairedDelta?.[metric] ?? null;
  const complete = contrast?.nPairs === DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS
    && Number.isFinite(observed?.absoluteMs)
    && Number.isFinite(observed?.percent);
  return {
    status: complete ? 'evaluated' : 'incomplete',
    pass: complete
      && Math.abs(observed.absoluteMs) < DEPTH_ORDERING_MATERIAL_ABSOLUTE_MS
      && Math.abs(observed.percent) < DEPTH_ORDERING_MATERIAL_PERCENT,
    metric,
    expectedPairs: DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS,
    observedPairs: contrast?.nPairs ?? 0,
    equivalenceBounds: {
      rule: 'absolute median paired delta must remain strictly inside both bounds',
      absoluteMs: DEPTH_ORDERING_MATERIAL_ABSOLUTE_MS,
      percent: DEPTH_ORDERING_MATERIAL_PERCENT,
    },
    observed,
  };
}

function maximumRegressionGate(contrast, metric) {
  const observed = contrast?.medianPairedDelta?.[metric] ?? null;
  const complete = contrast?.nPairs === DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS
    && Number.isFinite(observed?.percent);
  return {
    status: complete ? 'evaluated' : 'incomplete',
    pass: complete
      && observed.percent <= DEPTH_ORDERING_LOW_OVERLAP_MAX_REGRESSION_PERCENT,
    metric,
    expectedPairs: DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS,
    observedPairs: contrast?.nPairs ?? 0,
    maximumRegressionPercent: DEPTH_ORDERING_LOW_OVERLAP_MAX_REGRESSION_PERCENT,
    observed,
  };
}

function absolutePositionStrata(pairs, field) {
  return [0, 1, 2].map((position) => {
    const selected = pairs.filter((pair) => pair.orderAudit[field] === position);
    const medianGpuRenderDeltaMs = median(selected.map(
      (pair) => pair.delta.gpuRenderMs.absoluteMs,
    ));
    return {
      position,
      expectedPairs: 2,
      observedPairs: selected.length,
      medianGpuRenderDeltaMs,
      pass: selected.length === 2 && medianGpuRenderDeltaMs < 0,
    };
  });
}

function relativeOrderStrata(pairs) {
  return ['left-first', 'right-first'].map((orderStratum) => {
    const selected = pairs.filter(
      (pair) => pair.orderAudit.orderStratum === orderStratum,
    );
    const medianGpuRenderDeltaMs = median(selected.map(
      (pair) => pair.delta.gpuRenderMs.absoluteMs,
    ));
    return {
      orderStratum,
      expectedPairs: 3,
      observedPairs: selected.length,
      medianGpuRenderDeltaMs,
      pass: selected.length === 3 && medianGpuRenderDeltaMs < 0,
    };
  });
}

function directionStabilityGate(contrast) {
  const pairs = contrast?.pairs ?? [];
  const finitePairs = pairs.filter(
    (pair) => Number.isFinite(pair.delta.gpuRenderMs.absoluteMs),
  );
  const frontFasterPairs = finitePairs.filter(
    (pair) => pair.delta.gpuRenderMs.absoluteMs < 0,
  ).length;
  const tiedPairs = finitePairs.filter(
    (pair) => pair.delta.gpuRenderMs.absoluteMs === 0,
  ).length;
  const frontPositionStrata = absolutePositionStrata(
    finitePairs,
    'leftModeOrderPosition',
  );
  const reversePositionStrata = absolutePositionStrata(
    finitePairs,
    'rightModeOrderPosition',
  );
  const relativeStrata = relativeOrderStrata(finitePairs);
  const complete = finitePairs.length === DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS;
  return {
    status: complete ? 'evaluated' : 'incomplete',
    pass: complete
      && frontFasterPairs >= DEPTH_ORDERING_MINIMUM_DIRECTIONALLY_CONSISTENT_PAIRS
      && frontPositionStrata.every((stratum) => stratum.pass)
      && reversePositionStrata.every((stratum) => stratum.pass)
      && relativeStrata.every((stratum) => stratum.pass),
    metric: 'gpuRenderMs',
    expectedPairs: DEPTH_ORDERING_EXPECTED_PAIRED_REPETITIONS,
    observedPairs: finitePairs.length,
    minimumFrontFasterPairs:
      DEPTH_ORDERING_MINIMUM_DIRECTIONALLY_CONSISTENT_PAIRS,
    frontFasterPairs,
    reverseFasterPairs: finitePairs.length - frontFasterPairs - tiedPairs,
    tiedPairs,
    positionRule:
      'each absolute mode position must contain two pairs with a negative median delta',
    relativeOrderRule:
      'each relative-order stratum must contain three pairs with a negative median delta',
    frontPositionStrata,
    reversePositionStrata,
    relativeOrderStrata: relativeStrata,
  };
}

function evaluateDepthOrderingGates(orderingContrasts, contextualComparisons) {
  if (orderingContrasts.length === 0 && contextualComparisons.frontToBack.length === 0) {
    return null;
  }
  const highOrdering = depthContrastForLayout(orderingContrasts, 'high-overlap');
  const lowOrdering = depthContrastForLayout(orderingContrasts, 'low-overlap');
  const highFrontVsAtomic = depthContrastForLayout(
    contextualComparisons.frontToBack,
    'high-overlap',
  );
  const lowFrontVsAtomic = depthContrastForLayout(
    contextualComparisons.frontToBack,
    'low-overlap',
  );
  const gates = {
    highOverlapOrderingBenefit: materialImprovementGate(
      highOrdering,
      'gpuRenderMs',
    ),
    highOverlapDirectionStability: directionStabilityGate(highOrdering),
    lowOverlapNegativeControl: noMaterialDifferenceGate(
      lowOrdering,
      'gpuRenderMs',
    ),
    highOverlapFrontVsAtomicWholeMechanism: materialImprovementGate(
      highFrontVsAtomic,
      'gpuPassTotalMs',
    ),
    lowOverlapFrontVsAtomicWholeMechanism: maximumRegressionGate(
      lowFrontVsAtomic,
      'gpuPassTotalMs',
    ),
  };
  const failed = Object.entries(gates)
    .filter(([, gate]) => gate.pass !== true)
    .map(([name]) => name);
  return {
    status: Object.values(gates).every((gate) => gate.status === 'evaluated')
      ? 'evaluated'
      : 'incomplete',
    pass: failed.length === 0,
    deltaConvention: 'front-to-back minus comparator; negative values favor front-to-back',
    failedGates: failed,
    gates,
  };
}

function comparisonsAgainst(groups, baselineModeId) {
  return pairedContrasts(groups, 'fixed-slice', baselineModeId).map((contrast) => ({
    targetVisibilityFraction: contrast.targetVisibilityFraction,
    ...(contrast.layout === undefined ? {} : { layout: contrast.layout }),
    baselineModeId,
    nPairs: contrast.nPairs,
    pairs: contrast.pairs.map((pair) => ({
      repetition: pair.repetition,
      fixedSliceP50: pair.leftP50,
      baselineP50: pair.rightP50,
      delta: pair.delta,
    })),
    medianPairedDelta: contrast.medianPairedDelta,
    unmatchedFixedSliceRepetitions: contrast.unmatchedLeftRepetitions,
    unmatchedBaselineRepetitions: contrast.unmatchedRightRepetitions,
  }));
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function failVerification(message) {
  throw new Error(`Run verification failed: ${message}`);
}

function requireRecord(value, label) {
  if (!isRecord(value)) failVerification(`${label} must be an object.`);
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) failVerification(`${label} must be an array.`);
  return value;
}

function requireNonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    failVerification(`${label} must be a non-empty string.`);
  }
  return value;
}

function requireInteger(value, label, { minimum = 0 } = {}) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    failVerification(`${label} must be a safe integer greater than or equal to ${minimum}.`);
  }
  return value;
}

function requireFiniteNumber(value, label, { minimum = -Infinity } = {}) {
  if (!Number.isFinite(value) || value < minimum) {
    failVerification(`${label} must be a finite number greater than or equal to ${minimum}.`);
  }
  return value;
}

function requireSha256(value, label) {
  if (!SHA256_PATTERN.test(value ?? '')) {
    failVerification(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

function requireIsoTimestamp(value, label) {
  requireNonemptyString(value, label);
  if (!Number.isFinite(Date.parse(value))) failVerification(`${label} must be an ISO timestamp.`);
  return value;
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sha256Json(value) {
  return sha256Bytes(JSON.stringify(value));
}

function parseJsonArtifact(contents, name) {
  try {
    return JSON.parse(contents.toString('utf8'));
  } catch (error) {
    failVerification(
      `${name} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function sortedUniqueStrings(value, label) {
  const values = requireArray(value, label).map((entry, index) => (
    requireNonemptyString(entry, `${label}[${index}]`)
  ));
  if (new Set(values).size !== values.length) failVerification(`${label} contains duplicates.`);
  return [...values].sort();
}

function requireSafeArtifactName(name, label) {
  requireNonemptyString(name, label);
  if (name === '.' || name === '..' || name.includes('/') || name.includes('\\')) {
    failVerification(`${label} is not a safe run-directory filename.`);
  }
  return name;
}

async function readOptionalRegularFile(filename, label) {
  try {
    const fileStat = await lstat(filename);
    if (!fileStat.isFile() || fileStat.isSymbolicLink()) {
      failVerification(`${label} must be a regular non-symbolic-link file.`);
    }
    return await readFile(filename);
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

async function verifyCommittedSanitizerImplementation(
  repositoryRoot,
  candidateSource,
  files,
) {
  const expectedPaths = files.map((entry) => entry.repositoryPath);
  if (JSON.stringify(expectedPaths) !== JSON.stringify(LIVE_SANITIZER_IMPLEMENTATION_PATHS)) {
    failVerification('public bundle sanitizer implementation closure has the wrong paths.');
  }
  let committed;
  try {
    committed = await resolveLiveSanitizerImplementationAtCommit(
      repositoryRoot,
      candidateSource,
    );
  } catch (error) {
    failVerification(
      `public bundle candidate Git-object verification failed: ${
        error instanceof Error ? error.message : String(error)
      }.`,
    );
  }
  if (JSON.stringify(committed.files) !== JSON.stringify(files)) {
    failVerification(
      'public bundle sanitizer implementation hashes do not match the recorded candidate commit.',
    );
  }
}

async function loadVerifiedArtifactContents(runDirectory, repositoryRoot) {
  const manifestPath = path.join(runDirectory, 'artifact-manifest.json');
  const manifestBytes = await readOptionalRegularFile(
    manifestPath,
    'artifact-manifest.json',
  );
  if (manifestBytes === null) failVerification('artifact-manifest.json is missing.');
  const manifest = requireRecord(
    parseJsonArtifact(manifestBytes, 'artifact-manifest.json'),
    'artifact-manifest.json',
  );
  const publicDerived = Object.hasOwn(manifest, 'bundleProvenance');
  requireExactRecordKeys(
    manifest,
    publicDerived
      ? [
        'schemaVersion',
        'runId',
        'hashAlgorithm',
        'requiredFiles',
        'optionalFiles',
        'files',
        'bundleProvenance',
      ]
      : [
        'schemaVersion',
        'runId',
        'hashAlgorithm',
        'requiredFiles',
        'optionalFiles',
        'files',
      ],
    'artifact-manifest.json',
  );
  if (manifest.schemaVersion !== 2) {
    failVerification(`artifact-manifest.json schemaVersion must be 2; received ${JSON.stringify(manifest.schemaVersion)}.`);
  }
  requireNonemptyString(manifest.runId, 'artifact-manifest.json runId');
  if (manifest.hashAlgorithm !== 'sha256') {
    failVerification('artifact-manifest.json hashAlgorithm must be "sha256".');
  }

  const requiredNames = sortedUniqueStrings(
    manifest.requiredFiles,
    'artifact-manifest.json requiredFiles',
  );
  const allowedRequiredSets = [
    REQUIRED_RUN_ARTIFACTS,
    LIVE_FIRST_INSTANCE_REQUIRED_RUN_ARTIFACTS,
  ].map((names) => [...names].sort());
  if (!allowedRequiredSets.some((expectedNames) => (
    requiredNames.length === expectedNames.length
      && requiredNames.every((name, index) => name === expectedNames[index])
  ))) {
    failVerification(
      'artifact-manifest.json requiredFiles do not match a supported benchmark matrix.',
    );
  }

  const optionalRecords = requireArray(
    manifest.optionalFiles,
    'artifact-manifest.json optionalFiles',
  );
  const optionalByName = new Map();
  for (const [index, optional] of optionalRecords.entries()) {
    requireExactRecordKeys(
      optional,
      ['name', 'present', 'evidenceAvailable', 'absenceReason'],
      `artifact-manifest.json optionalFiles[${index}]`,
    );
    const name = requireSafeArtifactName(
      optional.name,
      `artifact-manifest.json optionalFiles[${index}].name`,
    );
    if (optionalByName.has(name) || requiredNames.includes(name)) {
      failVerification(`artifact-manifest.json declares ${JSON.stringify(name)} more than once.`);
    }
    if (typeof optional.present !== 'boolean') {
      failVerification(`artifact-manifest.json optional file ${JSON.stringify(name)} lacks a boolean present flag.`);
    }
    if (typeof optional.evidenceAvailable !== 'boolean') {
      failVerification(
        `artifact-manifest.json optional file ${JSON.stringify(name)} lacks a boolean evidenceAvailable flag.`,
      );
    }
    if (optional.evidenceAvailable && !optional.present) {
      failVerification(
        `artifact-manifest.json optional file ${JSON.stringify(name)} cannot provide absent evidence.`,
      );
    }
    if (!optional.evidenceAvailable) {
      requireNonemptyString(
        optional.absenceReason,
        `artifact-manifest.json optional file ${JSON.stringify(name)} absence reason`,
      );
    }
    optionalByName.set(name, optional);
  }

  const entries = requireArray(manifest.files, 'artifact-manifest.json files');
  const entriesByName = new Map();
  for (const [index, entry] of entries.entries()) {
    requireExactRecordKeys(
      entry,
      ['name', 'role', 'required', 'present', 'bytes', 'sha256', 'absenceReason'],
      `artifact-manifest.json files[${index}]`,
    );
    const name = requireSafeArtifactName(entry.name, `artifact-manifest.json files[${index}].name`);
    if (entriesByName.has(name)) {
      failVerification(`artifact-manifest.json has duplicate file entry ${JSON.stringify(name)}.`);
    }
    requireNonemptyString(entry.role, `artifact-manifest.json file ${JSON.stringify(name)} role`);
    if (typeof entry.required !== 'boolean' || typeof entry.present !== 'boolean') {
      failVerification(
        `artifact-manifest.json file ${JSON.stringify(name)} must declare boolean required and present flags.`,
      );
    }
    const declaredRequired = requiredNames.includes(name);
    const declaredOptional = optionalByName.has(name);
    if (!declaredRequired && !declaredOptional) {
      failVerification(`artifact-manifest.json file ${JSON.stringify(name)} is not declared required or optional.`);
    }
    if (entry.required !== declaredRequired) {
      failVerification(`artifact-manifest.json file ${JSON.stringify(name)} has an inconsistent required flag.`);
    }
    if (declaredRequired && !entry.present) {
      failVerification(`required artifact ${name} is marked absent.`);
    }
    if (declaredOptional && entry.present !== optionalByName.get(name).present) {
      failVerification(`optional artifact ${name} has inconsistent present flags.`);
    }
    if (!entry.present) {
      requireNonemptyString(
        entry.absenceReason,
        `artifact-manifest.json file ${JSON.stringify(name)} absenceReason`,
      );
    }
    entriesByName.set(name, entry);
  }

  const declaredNames = [...requiredNames, ...optionalByName.keys()].sort();
  const entryNames = [...entriesByName.keys()].sort();
  if (declaredNames.length !== entryNames.length
    || declaredNames.some((name, index) => name !== entryNames[index])) {
    failVerification('artifact-manifest.json files do not exactly cover its required and optional declarations.');
  }

  const contentsByName = new Map();
  for (const [name, entry] of entriesByName) {
    const contents = await readOptionalRegularFile(
      path.join(runDirectory, name),
      `artifact ${name}`,
    );
    if (!entry.present) {
      if (contents !== null) {
        failVerification(`artifact ${name} exists but is marked absent in artifact-manifest.json.`);
      }
      continue;
    }
    if (contents === null) failVerification(`artifact ${name} is missing.`);
    requireInteger(entry.bytes, `artifact-manifest.json file ${JSON.stringify(name)} bytes`);
    requireSha256(entry.sha256, `artifact-manifest.json file ${JSON.stringify(name)} sha256`);
    if (contents.length !== entry.bytes) {
      failVerification(
        `artifact ${name} byte size is ${contents.length}; manifest records ${entry.bytes}.`,
      );
    }
    const digest = sha256Bytes(contents);
    if (digest !== entry.sha256) {
      failVerification(`artifact ${name} SHA-256 does not match artifact-manifest.json.`);
    }
    contentsByName.set(name, contents);
  }

  const expectedDirectoryNames = [
    'artifact-manifest.json',
    ...contentsByName.keys(),
  ].sort();
  const actualDirectoryNames = (await readdir(runDirectory)).sort();
  if (JSON.stringify(actualDirectoryNames) !== JSON.stringify(expectedDirectoryNames)) {
    failVerification(
      'run directory contains undeclared or non-artifact entries outside artifact-manifest.json.',
    );
  }

  let bundleIntegrity;
  if (publicDerived) {
    const provenanceReasons = validateLivePublicBundleProvenanceShape(
      manifest.bundleProvenance,
    );
    if (provenanceReasons.length > 0) {
      failVerification(`public bundle provenance is invalid: ${provenanceReasons.join('; ')}.`);
    }
    await verifyCommittedSanitizerImplementation(
      repositoryRoot,
      manifest.bundleProvenance.candidateSource,
      manifest.bundleProvenance.sanitizer.files,
    );
    bundleIntegrity = {
      schemaVersion: 1,
      status: 'consistent',
      bundleLabel: LIVE_PUBLIC_BUNDLE_LABEL,
      sourceBundleLabel: LIVE_PRIVATE_BUNDLE_LABEL,
      policyId: LIVE_PUBLIC_EVIDENCE_POLICY_ID,
      artifactManifestSha256: sha256Bytes(manifestBytes),
      privateArtifactManifestSha256:
        manifest.bundleProvenance.privateArtifactManifest.sha256,
      sanitizerImplementationKind:
        manifest.bundleProvenance.sanitizer.implementationKind,
      sanitizerImplementation: structuredClone(
        manifest.bundleProvenance.sanitizer.files,
      ),
      candidateCommitObjectVerified: true,
      candidateTreeObjectVerified: true,
      sanitizerImplementationCandidateCommitMatch: true,
      candidateSource: structuredClone(manifest.bundleProvenance.candidateSource),
      candidateMetadataBindingVerified: null,
      redactionPolicyValidated: true,
      declaredArtifactsIntegrityVerified: true,
      undeclaredDirectoryEntriesRejected: true,
      privateSourceRelationshipVerified: false,
      authenticityVerified: false,
    };
  } else {
    bundleIntegrity = {
      schemaVersion: 1,
      status: 'consistent',
      bundleLabel: LIVE_PRIVATE_BUNDLE_LABEL,
      sourceBundleLabel: null,
      policyId: null,
      artifactManifestSha256: sha256Bytes(manifestBytes),
      privateArtifactManifestSha256: null,
      sanitizerImplementationKind: null,
      sanitizerImplementation: null,
      candidateCommitObjectVerified: null,
      candidateTreeObjectVerified: null,
      sanitizerImplementationCandidateCommitMatch: null,
      candidateSource: null,
      candidateMetadataBindingVerified: null,
      redactionPolicyValidated: null,
      declaredArtifactsIntegrityVerified: true,
      undeclaredDirectoryEntriesRejected: true,
      privateSourceRelationshipVerified: null,
      authenticityVerified: false,
    };
  }

  return {
    manifest,
    contentsByName,
    requiredNames,
    bundleIntegrity,
  };
}

function validateCandidateProvenance(metadata) {
  if (metadata.evidenceStatus !== 'candidate') return;
  const provenance = requireRecord(metadata.sourceProvenance, 'metadata sourceProvenance');
  if (provenance.stable !== true) {
    failVerification('candidate evidence does not declare stable source provenance.');
  }
  const start = requireRecord(provenance.start, 'metadata sourceProvenance.start');
  const end = requireRecord(provenance.end, 'metadata sourceProvenance.end');
  for (const [label, record] of [['start', start], ['end', end]]) {
    if (record.status !== 'available') {
      failVerification(`candidate source provenance ${label} capture is unavailable.`);
    }
    if (record.dirty !== false
      || record.stagedChanges !== 0
      || record.unstagedChanges !== 0
      || record.untrackedFiles !== 0) {
      failVerification(`candidate source provenance ${label} capture is not clean.`);
    }
    if (record.packageLockTracked !== true) {
      failVerification(`candidate source provenance ${label} capture lacks a tracked package lock.`);
    }
    if (!/^[0-9a-f]{40,64}$/.test(record.commit ?? '')
      || !/^[0-9a-f]{40,64}$/.test(record.tree ?? '')) {
      failVerification(`candidate source provenance ${label} capture lacks Git object IDs.`);
    }
    if (record.ref !== null && (typeof record.ref !== 'string' || record.ref === '')) {
      failVerification(`candidate source provenance ${label} capture has an invalid ref.`);
    }
    for (const field of [
      'stagedChanges',
      'unstagedChanges',
      'untrackedFiles',
      'porcelainEntryCount',
      'porcelainByteCount',
      'trackedFileCount',
    ]) {
      requireInteger(record[field], `candidate provenance ${label} ${field}`);
    }
    requireSha256(record.porcelainSha256, `candidate provenance ${label} porcelainSha256`);
    requireSha256(record.trackedFilesSha256, `candidate provenance ${label} trackedFilesSha256`);
    requireSha256(record.packageLockSha256, `candidate provenance ${label} packageLockSha256`);
  }
  for (const field of PROVENANCE_STABILITY_FIELDS) {
    if (start[field] !== end[field]) {
      failVerification(`candidate source provenance changed at field ${field}.`);
    }
  }
}

export function validateLiveCandidateReservationAndDependencies(metadata) {
  if (metadata.protocol?.matrixKind !== LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX
    || metadata.evidenceStatus !== 'candidate') return;
  const reservation = requireExactRecordKeys(
    metadata.candidateSeriesReservation,
    [
      'schemaVersion',
      'kind',
      'seriesId',
      'reservationEventSha256',
      'attemptOrdinal',
      'matrixOrdinal',
      'sourceCommit',
      'sourceTree',
      'executionDependencyClosureSha256',
    ],
    'metadata candidateSeriesReservation',
  );
  if (reservation.schemaVersion !== 1
    || reservation.kind !== 'first-instance-live-candidate-series-reservation') {
    failVerification('live candidate reservation has an unsupported identity.');
  }
  requireNonemptyString(reservation.seriesId, 'candidate reservation seriesId');
  requireSha256(
    reservation.reservationEventSha256,
    'candidate reservation reservationEventSha256',
  );
  requireInteger(reservation.attemptOrdinal, 'candidate reservation attemptOrdinal', {
    minimum: 1,
  });
  requireInteger(reservation.matrixOrdinal, 'candidate reservation matrixOrdinal', {
    minimum: 1,
  });
  if (reservation.matrixOrdinal > 2) {
    failVerification('candidate reservation matrixOrdinal must be 1 or 2.');
  }
  if (!/^[0-9a-f]{40,64}$/.test(reservation.sourceCommit ?? '')
    || !/^[0-9a-f]{40,64}$/.test(reservation.sourceTree ?? '')) {
    failVerification('candidate reservation lacks Git source object IDs.');
  }
  requireSha256(
    reservation.executionDependencyClosureSha256,
    'candidate reservation executionDependencyClosureSha256',
  );
  const envelope = requireExactRecordKeys(
    metadata.executionDependencyClosure,
    ['start', 'end', 'stable'],
    'metadata executionDependencyClosure',
  );
  if (envelope.stable !== true) {
    failVerification('candidate execution-dependency closure is not stable.');
  }
  const validateClosure = (value, label) => {
    const closure = requireExactRecordKeys(
      value,
      [
        'schemaVersion',
        'kind',
        'root',
        'format',
        'hashAlgorithm',
        'exclusions',
        'fileCount',
        'totalBytes',
        'sha256',
      ],
      label,
    );
    if (closure.schemaVersion !== 1
      || closure.kind !== 'installed-execution-dependency-closure'
      || closure.root !== 'node_modules'
      || closure.format !== 'node-modules-sorted-path-size-content-sha256-v1'
      || closure.hashAlgorithm !== 'sha256'
      || JSON.stringify(closure.exclusions) !== JSON.stringify(['.bin/**', '.vite*/**'])) {
      failVerification(`${label} has an unsupported closure definition.`);
    }
    requireInteger(closure.fileCount, `${label}.fileCount`, { minimum: 1 });
    requireInteger(closure.totalBytes, `${label}.totalBytes`, { minimum: 1 });
    requireSha256(closure.sha256, `${label}.sha256`);
    return closure;
  };
  const start = validateClosure(envelope.start, 'executionDependencyClosure.start');
  const end = validateClosure(envelope.end, 'executionDependencyClosure.end');
  if (!executionDependencyClosuresMatch(start, end)) {
    failVerification('candidate execution-dependency start/end closures differ.');
  }
  const provenanceStart = metadata.sourceProvenance.start;
  if (reservation.sourceCommit !== provenanceStart.commit
    || reservation.sourceTree !== provenanceStart.tree
    || reservation.executionDependencyClosureSha256 !== start.sha256) {
    failVerification('candidate reservation differs from executed source/dependency bytes.');
  }
}

export async function validateLiveCandidateViteRuntime(metadata, repositoryRoot) {
  if (metadata.protocol?.matrixKind !== LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX
    || metadata.evidenceStatus !== 'candidate') return;
  const validation = await validateCandidateViteRuntimeAudit(
    metadata.candidateViteRuntimeAudit,
    repositoryRoot,
  );
  if (!validation.pass) {
    failVerification(
      `candidate Vite runtime audit is invalid: ${validation.reasons.join('; ')}.`,
    );
  }
}

function validatePublicBundleCandidateBinding(bundleIntegrity, metadata) {
  if (bundleIntegrity.bundleLabel !== LIVE_PUBLIC_BUNDLE_LABEL) return;
  if (metadata.evidenceStatus !== 'candidate'
    || metadata.protocol?.matrixKind !== LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    failVerification(
      'public-derived live evidence must be a first-instance-live candidate run.',
    );
  }
  const candidate = bundleIntegrity.candidateSource;
  const start = metadata.sourceProvenance?.start;
  const end = metadata.sourceProvenance?.end;
  if (candidate.commit !== start?.commit
    || candidate.commit !== end?.commit
    || candidate.tree !== start?.tree
    || candidate.tree !== end?.tree
    || candidate.trackedFilesSha256 !== start?.trackedFilesSha256
    || candidate.trackedFilesSha256 !== end?.trackedFilesSha256
    || candidate.packageLockSha256 !== start?.packageLockSha256
    || candidate.packageLockSha256 !== end?.packageLockSha256
    || start?.dirty !== false
    || end?.dirty !== false
    || metadata.sourceProvenance?.stable !== true) {
    failVerification(
      'public bundle candidateSource does not match clean, stable run source provenance.',
    );
  }
  bundleIntegrity.candidateMetadataBindingVerified = true;
}

function requireExactRecordKeys(value, expectedKeys, label) {
  const record = requireRecord(value, label);
  const keys = Object.keys(record);
  if (keys.length !== expectedKeys.length
    || expectedKeys.some((key) => !Object.hasOwn(record, key))) {
    failVerification(`${label} has an unexpected schema.`);
  }
  return record;
}

function requireNullableNonemptyString(value, label) {
  if (value !== null) requireNonemptyString(value, label);
  return value;
}

function validateTelemetryNumericSummary(value, label) {
  if (value === null) return;
  const summary = requireExactRecordKeys(
    value,
    ['minimum', 'median', 'maximum'],
    label,
  );
  for (const field of ['minimum', 'median', 'maximum']) {
    requireFiniteNumber(summary[field], `${label}.${field}`);
  }
  if (summary.minimum > summary.median || summary.median > summary.maximum) {
    failVerification(`${label} is not ordered minimum/median/maximum.`);
  }
}

function validateTelemetryCounts(value, sampleCount, label) {
  const counts = requireRecord(value, label);
  let total = 0;
  for (const [key, count] of Object.entries(counts)) {
    requireNonemptyString(key, `${label} key`);
    requireInteger(count, `${label}.${key}`, { minimum: 1 });
    total += count;
  }
  if (!Number.isSafeInteger(total) || total !== sampleCount) {
    failVerification(`${label} does not sum to the GPU sample count.`);
  }
}

function validateTelemetrySummarySchema(summary, status) {
  requireExactRecordKeys(
    summary,
    ['sampleCount', 'gpuCount', 'firstObservedAtIso', 'lastObservedAtIso', 'gpus'],
    'gpu-telemetry-summary.json summary',
  );
  requireInteger(summary.sampleCount, 'gpu-telemetry-summary.json summary.sampleCount');
  requireInteger(summary.gpuCount, 'gpu-telemetry-summary.json summary.gpuCount');
  const gpus = requireArray(summary.gpus, 'gpu-telemetry-summary.json summary.gpus');
  if (summary.gpuCount !== gpus.length) {
    failVerification('gpu-telemetry-summary.json summary.gpuCount differs from gpus.length.');
  }
  if (summary.sampleCount === 0) {
    if (summary.gpuCount !== 0
      || summary.firstObservedAtIso !== null
      || summary.lastObservedAtIso !== null) {
      failVerification('empty GPU telemetry summary has non-empty observations.');
    }
  } else {
    requireIsoTimestamp(
      summary.firstObservedAtIso,
      'gpu-telemetry-summary.json summary.firstObservedAtIso',
    );
    requireIsoTimestamp(
      summary.lastObservedAtIso,
      'gpu-telemetry-summary.json summary.lastObservedAtIso',
    );
    if (Date.parse(summary.lastObservedAtIso) < Date.parse(summary.firstObservedAtIso)) {
      failVerification('GPU telemetry summary observations are time-reversed.');
    }
  }
  if ((status === 'available' || status === 'interrupted' || status === 'recorded-not-written')
      !== (summary.sampleCount > 0)) {
    failVerification('GPU telemetry status and sample count are inconsistent.');
  }
  const numericFields = [
    'graphicsClockMHz',
    'memoryClockMHz',
    'gpuUtilizationPercent',
    'memoryUtilizationPercent',
    'memoryUsedMiB',
    'memoryTotalMiB',
    'temperatureC',
    'powerDrawW',
  ];
  let totalSamples = 0;
  for (const [index, gpu] of gpus.entries()) {
    const label = `gpu-telemetry-summary.json summary.gpus[${index}]`;
    requireExactRecordKeys(gpu, [
      'gpuIndex',
      'gpuName',
      'gpuUuid',
      'sampleCount',
      'firstObservedAtIso',
      'lastObservedAtIso',
      'maximumSampleGapMs',
      'phaseSampleCounts',
      'pstateSampleCounts',
      ...numericFields,
    ], label);
    if (gpu.gpuIndex !== null) requireInteger(gpu.gpuIndex, `${label}.gpuIndex`);
    requireNullableNonemptyString(gpu.gpuName, `${label}.gpuName`);
    requireNullableNonemptyString(gpu.gpuUuid, `${label}.gpuUuid`);
    requireInteger(gpu.sampleCount, `${label}.sampleCount`, { minimum: 1 });
    totalSamples += gpu.sampleCount;
    requireIsoTimestamp(gpu.firstObservedAtIso, `${label}.firstObservedAtIso`);
    requireIsoTimestamp(gpu.lastObservedAtIso, `${label}.lastObservedAtIso`);
    if (Date.parse(gpu.lastObservedAtIso) < Date.parse(gpu.firstObservedAtIso)) {
      failVerification(`${label} observations are time-reversed.`);
    }
    if (gpu.maximumSampleGapMs !== null) {
      requireFiniteNumber(gpu.maximumSampleGapMs, `${label}.maximumSampleGapMs`, {
        minimum: 0,
      });
    }
    validateTelemetryCounts(gpu.phaseSampleCounts, gpu.sampleCount, `${label}.phaseSampleCounts`);
    validateTelemetryCounts(gpu.pstateSampleCounts, gpu.sampleCount, `${label}.pstateSampleCounts`);
    for (const field of numericFields) {
      validateTelemetryNumericSummary(gpu[field], `${label}.${field}`);
    }
  }
  if (totalSamples !== summary.sampleCount) {
    failVerification('GPU telemetry per-device samples do not sum to summary.sampleCount.');
  }
}

function validateComputeProcessSnapshot(snapshot, expectedLabel, label) {
  if (snapshot === null) return;
  requireExactRecordKeys(
    snapshot,
    [
      'label',
      'status',
      'capturedAtIso',
      'runElapsedMs',
      'reason',
      'rawNonemptyLineCount',
      'parsedRecordCount',
      'malformedLineCount',
      'stdoutByteCount',
      'stdoutTruncated',
      'stderrByteCount',
      'processes',
    ],
    label,
  );
  if (snapshot.label !== expectedLabel
    || (snapshot.status !== 'available' && snapshot.status !== 'unavailable')) {
    failVerification(`${label} has an invalid label or status.`);
  }
  requireIsoTimestamp(snapshot.capturedAtIso, `${label}.capturedAtIso`);
  requireFiniteNumber(snapshot.runElapsedMs, `${label}.runElapsedMs`, { minimum: 0 });
  for (const field of [
    'rawNonemptyLineCount',
    'parsedRecordCount',
    'malformedLineCount',
    'stdoutByteCount',
    'stderrByteCount',
  ]) {
    requireInteger(snapshot[field], `${label}.${field}`);
  }
  if (typeof snapshot.stdoutTruncated !== 'boolean') {
    failVerification(`${label}.stdoutTruncated must be boolean.`);
  }
  if (snapshot.parsedRecordCount + snapshot.malformedLineCount
      !== snapshot.rawNonemptyLineCount
    || snapshot.stdoutTruncated !== (snapshot.stdoutByteCount > 1_000_000)
    || (snapshot.parsedRecordCount > 0 && snapshot.stdoutByteCount === 0)) {
    failVerification(`${label} has contradictory parser byte/record diagnostics.`);
  }
  const processes = requireArray(snapshot.processes, `${label}.processes`);
  if (snapshot.status === 'available') {
    if (snapshot.reason !== null) {
      failVerification(`${label} available snapshot has a reason.`);
    }
    if (snapshot.stdoutTruncated !== false
      || snapshot.stderrByteCount !== 0
      || snapshot.malformedLineCount !== 0
      || snapshot.parsedRecordCount !== snapshot.rawNonemptyLineCount
      || snapshot.parsedRecordCount !== processes.length) {
      failVerification(`${label} available snapshot has invalid parser diagnostics.`);
    }
  } else {
    requireNonemptyString(snapshot.reason, `${label}.reason`);
    if (processes.length !== 0) {
      failVerification(`${label} unavailable snapshot contains processes.`);
    }
  }
  for (const [index, processRecord] of processes.entries()) {
    const processLabel = `${label}.processes[${index}]`;
    requireExactRecordKeys(
      processRecord,
      ['gpuUuid', 'pid', 'processName', 'usedMemoryMiB'],
      processLabel,
    );
    requireNonemptyString(processRecord.gpuUuid, `${processLabel}.gpuUuid`);
    requireInteger(processRecord.pid, `${processLabel}.pid`, { minimum: 1 });
    requireNonemptyString(processRecord.processName, `${processLabel}.processName`);
    if (processRecord.usedMemoryMiB !== null) {
      requireFiniteNumber(processRecord.usedMemoryMiB, `${processLabel}.usedMemoryMiB`, {
        minimum: 0,
      });
    }
  }
}

function validateTelemetryCoverageAudit(audit, summary, sampling) {
  requireExactRecordKeys(audit, [
    'schemaVersion',
    'kind',
    'requestedIntervalMs',
    'livenessToleranceMultiplier',
    'livenessToleranceMs',
    'sampleGroupingGapMs',
    'collectorStartedRunElapsedMs',
    'collectorStopRequestedRunElapsedMs',
    'activeDurationMs',
    'sampleCount',
    'sampleCycleCount',
    'gpuIdentities',
    'constantGpuIdentitySet',
    'initialMaximumGapMs',
    'internalMaximumGapMs',
    'finalMaximumGapMs',
    'pass',
    'failureCodes',
    'reasons',
  ], 'gpu-telemetry-summary.json coverageAudit');
  if (audit.schemaVersion !== 1
    || audit.kind !== 'nvidia-telemetry-collector-coverage'
    || audit.requestedIntervalMs !== 250
    || audit.livenessToleranceMultiplier !== 8
    || audit.livenessToleranceMs !== 2_000
    || audit.sampleGroupingGapMs !== 125) {
    failVerification('gpu telemetry coverage audit changes the frozen liveness contract.');
  }
  for (const field of [
    'collectorStartedRunElapsedMs',
    'collectorStopRequestedRunElapsedMs',
    'activeDurationMs',
    'initialMaximumGapMs',
    'internalMaximumGapMs',
    'finalMaximumGapMs',
  ]) {
    if (audit[field] !== null) {
      requireFiniteNumber(audit[field], `gpu telemetry coverageAudit.${field}`, {
        minimum: 0,
      });
    }
  }
  for (const field of ['sampleCount', 'sampleCycleCount']) {
    requireInteger(audit[field], `gpu telemetry coverageAudit.${field}`);
  }
  if (audit.sampleCycleCount > audit.sampleCount) {
    failVerification('gpu telemetry coverage audit has more cycles than samples.');
  }
  if (audit.sampleCount !== summary.sampleCount
    || audit.collectorStartedRunElapsedMs !== sampling.collectorStartedRunElapsedMs
    || audit.collectorStopRequestedRunElapsedMs !== sampling.collectorStopRequestedRunElapsedMs) {
    failVerification('gpu telemetry coverage audit is not bound to sampling bounds and rows.');
  }
  const identities = requireArray(
    audit.gpuIdentities,
    'gpu telemetry coverageAudit.gpuIdentities',
  );
  const identityKeys = [];
  for (const [index, identity] of identities.entries()) {
    const label = `gpu telemetry coverageAudit.gpuIdentities[${index}]`;
    requireExactRecordKeys(identity, ['gpuIndex', 'gpuName', 'gpuUuid'], label);
    requireInteger(identity.gpuIndex, `${label}.gpuIndex`);
    requireNonemptyString(identity.gpuName, `${label}.gpuName`);
    requireNonemptyString(identity.gpuUuid, `${label}.gpuUuid`);
    identityKeys.push(JSON.stringify([identity.gpuIndex, identity.gpuName, identity.gpuUuid]));
  }
  if (new Set(identityKeys).size !== identityKeys.length
    || JSON.stringify([...identityKeys].sort()) !== JSON.stringify(identityKeys)) {
    failVerification('gpu telemetry coverage identities are duplicated or unsorted.');
  }
  if (typeof audit.constantGpuIdentitySet !== 'boolean' || typeof audit.pass !== 'boolean') {
    failVerification('gpu telemetry coverage disposition fields must be boolean.');
  }
  const boundsValid = audit.collectorStartedRunElapsedMs !== null
    && audit.collectorStopRequestedRunElapsedMs !== null
    && audit.collectorStopRequestedRunElapsedMs >= audit.collectorStartedRunElapsedMs;
  const expectedActiveDuration = boundsValid
    ? audit.collectorStopRequestedRunElapsedMs - audit.collectorStartedRunElapsedMs
    : null;
  if (audit.activeDurationMs !== expectedActiveDuration) {
    failVerification('gpu telemetry coverage active duration contradicts collector bounds.');
  }
  if (audit.constantGpuIdentitySet
    && (audit.sampleCycleCount === 0 || identityKeys.length === 0)) {
    failVerification('gpu telemetry coverage claims a constant empty identity set.');
  }
  for (const field of ['failureCodes', 'reasons']) {
    const values = requireArray(audit[field], `gpu telemetry coverageAudit.${field}`);
    if (values.some((value) => typeof value !== 'string' || value === '')
      || new Set(values).size !== values.length) {
      failVerification(`gpu telemetry coverageAudit.${field} is invalid.`);
    }
  }
  if (audit.pass !== (audit.failureCodes.length === 0)
    || (audit.pass && audit.reasons.length !== 0)
    || (!audit.pass && audit.reasons.length === 0)) {
    failVerification('gpu telemetry coverage audit disposition is internally inconsistent.');
  }
}

function nullableTelemetryCsvValue(value, parser, label) {
  if (value === '') return null;
  return parser(value, label);
}

function parseTelemetryCsv(contents, runId) {
  const parsed = parseCsv(contents.toString('utf8'), { allowEmptyRecords: true });
  if (!orderedValuesMatch(parsed.headers, TELEMETRY_CSV_FIELDS)) {
    failVerification('gpu-telemetry.csv headers differ from the Nvidia telemetry schema.');
  }
  const integer = (value, label) => {
    const result = Number(value);
    if (!Number.isSafeInteger(result) || result < 0) {
      failVerification(`${label} must be a nonnegative safe integer.`);
    }
    return result;
  };
  const number = (value, label) => {
    const result = Number(value);
    if (!Number.isFinite(result)) failVerification(`${label} must be finite.`);
    return result;
  };
  return parsed.records.map((record, index) => {
    const label = `gpu-telemetry.csv record ${index + 2}`;
    requireIsoTimestamp(record.observedAtIso, `${label}.observedAtIso`);
    if (record.runId !== runId) failVerification(`${label}.runId differs from metadata.`);
    requireNonemptyString(record.phase, `${label}.phase`);
    return {
      observedAtIso: record.observedAtIso,
      runElapsedMs: number(record.runElapsedMs, `${label}.runElapsedMs`),
      runId: record.runId,
      trialId: record.trialId === '' ? null : requireNonemptyString(record.trialId, `${label}.trialId`),
      planIndex: nullableTelemetryCsvValue(record.planIndex, integer, `${label}.planIndex`),
      repetitionIndex: nullableTelemetryCsvValue(
        record.repetitionIndex,
        integer,
        `${label}.repetitionIndex`,
      ),
      modeId: record.modeId === '' ? null : requireNonemptyString(record.modeId, `${label}.modeId`),
      visibilityFraction: nullableTelemetryCsvValue(
        record.visibilityFraction,
        number,
        `${label}.visibilityFraction`,
      ),
      layout: record.layout === '' ? null : requireNonemptyString(record.layout, `${label}.layout`),
      phase: record.phase,
      gpuIndex: nullableTelemetryCsvValue(record.gpuIndex, integer, `${label}.gpuIndex`),
      gpuName: record.gpuName === '' ? null : requireNonemptyString(record.gpuName, `${label}.gpuName`),
      gpuUuid: record.gpuUuid === '' ? null : requireNonemptyString(record.gpuUuid, `${label}.gpuUuid`),
      pstate: record.pstate === '' ? null : requireNonemptyString(record.pstate, `${label}.pstate`),
      ...Object.fromEntries([
        'graphicsClockMHz',
        'memoryClockMHz',
        'gpuUtilizationPercent',
        'memoryUtilizationPercent',
        'memoryUsedMiB',
        'memoryTotalMiB',
        'temperatureC',
        'powerDrawW',
      ].map((field) => [field, nullableTelemetryCsvValue(
        record[field],
        number,
        `${label}.${field}`,
      )])),
    };
  });
}

function validateNvidiaTelemetryReport(report, metadata, manifest, contentsByName) {
  requireExactRecordKeys(report, [
    'provider',
    'status',
    'reason',
    'command',
    'sampling',
    'summary',
    'coverageAudit',
    'computeProcesses',
    'acceptanceBoundary',
  ], 'gpu-telemetry-summary.json');
  const allowedStatuses = new Set([
    'available',
    'unavailable',
    'interrupted',
    'recorded-not-written',
  ]);
  if (report.provider !== 'nvidia-smi' || !allowedStatuses.has(report.status)) {
    failVerification('gpu-telemetry-summary.json has an invalid provider or terminal status.');
  }
  requireNonemptyString(report.command, 'gpu-telemetry-summary.json command');
  if (report.status === 'available') {
    if (report.reason !== null) failVerification('available GPU telemetry has a reason.');
  } else {
    requireNonemptyString(report.reason, 'gpu-telemetry-summary.json reason');
  }
  if (report.status === 'recorded-not-written'
    && report.reason !== 'telemetry-output-write-failed') {
    failVerification('recorded-not-written GPU telemetry has the wrong reason.');
  }
  const sampling = requireExactRecordKeys(report.sampling, [
    'processModel',
    'requestedIntervalMs',
    'queryFields',
    'outputFile',
    'collectorStartedRunElapsedMs',
    'collectorStopRequestedRunElapsedMs',
    'malformedLineCount',
    'stderrByteCount',
    'exit',
  ], 'gpu-telemetry-summary.json sampling');
  if (sampling.processModel !== 'one-long-lived-process'
    || sampling.requestedIntervalMs !== 250
    || !orderedValuesMatch(sampling.queryFields, NVIDIA_QUERY_FIELDS)
    || sampling.outputFile !== 'gpu-telemetry.csv') {
    failVerification('gpu-telemetry-summary.json sampling contract differs from the runner.');
  }
  requireInteger(sampling.malformedLineCount, 'gpu telemetry malformedLineCount');
  requireInteger(sampling.stderrByteCount, 'gpu telemetry stderrByteCount');
  for (const field of [
    'collectorStartedRunElapsedMs',
    'collectorStopRequestedRunElapsedMs',
  ]) {
    if (sampling[field] !== null) {
      requireFiniteNumber(sampling[field], `gpu telemetry sampling.${field}`, {
        minimum: 0,
      });
    }
  }
  if (sampling.collectorStartedRunElapsedMs !== null
    && sampling.collectorStartedRunElapsedMs > metadata.elapsedMs) {
    failVerification('gpu telemetry collector start exceeds the recorded run duration.');
  }
  if (sampling.collectorStopRequestedRunElapsedMs !== null
    && sampling.collectorStopRequestedRunElapsedMs > metadata.elapsedMs) {
    failVerification('gpu telemetry collector stop request exceeds the recorded run duration.');
  }
  if (sampling.exit !== null) {
    requireExactRecordKeys(sampling.exit, ['code', 'signal'], 'gpu telemetry sampling.exit');
    if (sampling.exit.code !== null) {
      requireInteger(sampling.exit.code, 'gpu telemetry sampling.exit.code');
    }
    requireNullableNonemptyString(sampling.exit.signal, 'gpu telemetry sampling.exit.signal');
    if (sampling.exit.code === null && sampling.exit.signal === null) {
      failVerification('gpu telemetry sampling.exit lacks both code and signal.');
    }
  }
  validateTelemetrySummarySchema(report.summary, report.status);
  validateTelemetryCoverageAudit(report.coverageAudit, report.summary, sampling);
  const computeProcesses = requireExactRecordKeys(
    report.computeProcesses,
    ['pre', 'post'],
    'gpu-telemetry-summary.json computeProcesses',
  );
  validateComputeProcessSnapshot(
    computeProcesses.pre,
    'pre-run',
    'gpu-telemetry-summary.json computeProcesses.pre',
  );
  validateComputeProcessSnapshot(
    computeProcesses.post,
    'post-run',
    'gpu-telemetry-summary.json computeProcesses.post',
  );
  const acceptance = requireExactRecordKeys(report.acceptanceBoundary, [
    'affectsTechnicalRunAcceptance',
    'candidateEnvironmentReviewRequired',
    'automaticPstateRejectionThreshold',
  ], 'gpu-telemetry-summary.json acceptanceBoundary');
  if (acceptance.affectsTechnicalRunAcceptance !== false
    || acceptance.candidateEnvironmentReviewRequired !== true
    || acceptance.automaticPstateRejectionThreshold !== null) {
    failVerification('gpu-telemetry-summary.json changes the telemetry acceptance boundary.');
  }

  if (manifest.optionalFiles.length !== 1
    || manifest.optionalFiles[0]?.name !== 'gpu-telemetry.csv') {
    failVerification(
      'first-instance artifact manifest must declare exactly gpu-telemetry.csv as optional.',
    );
  }
  const optional = requireRecord(
    manifest.optionalFiles[0],
    'artifact-manifest.json gpu-telemetry.csv optional declaration',
  );
  const file = requireRecord(
    manifest.files.find((record) => record.name === 'gpu-telemetry.csv'),
    'artifact-manifest.json gpu-telemetry.csv file entry',
  );
  const present = contentsByName.has('gpu-telemetry.csv');
  const evidenceAvailable = report.status === 'available' && present;
  const absenceReason = report.reason ?? `telemetry status: ${report.status}`;
  if (optional.present !== present
    || file.present !== present
    || optional.evidenceAvailable !== evidenceAvailable
    || optional.absenceReason !== (evidenceAvailable ? null : absenceReason)
    || file.absenceReason !== (present ? null : absenceReason)) {
    failVerification(
      'gpu-telemetry.csv manifest presence/evidence fields are incoherent with telemetry status.',
    );
  }
  if ((report.status === 'available' || report.status === 'interrupted') && !present) {
    failVerification(`GPU telemetry status ${report.status} requires gpu-telemetry.csv.`);
  }
  if (report.status === 'recorded-not-written' && present) {
    failVerification('recorded-not-written GPU telemetry cannot include gpu-telemetry.csv.');
  }
  if (report.status === 'unavailable'
    && !present
    && report.reason !== 'telemetry-output-write-failed') {
    failVerification('unavailable telemetry without gpu-telemetry.csv has an incoherent reason.');
  }
  if (present) {
    const rows = parseTelemetryCsv(contentsByName.get('gpu-telemetry.csv'), metadata.runId);
    const recomputed = summarizeTelemetryRows(rows);
    if (JSON.stringify(recomputed) !== JSON.stringify(report.summary)) {
      failVerification('gpu-telemetry.csv does not reconstruct gpu-telemetry-summary.json.');
    }
    const recomputedCoverage = createNvidiaTelemetryCoverageAudit(rows, {
      collectorStartedRunElapsedMs: sampling.collectorStartedRunElapsedMs,
      collectorStopRequestedRunElapsedMs: sampling.collectorStopRequestedRunElapsedMs,
      requestedIntervalMs: sampling.requestedIntervalMs,
    });
    if (JSON.stringify(recomputedCoverage) !== JSON.stringify(report.coverageAudit)) {
      failVerification('gpu-telemetry.csv does not reconstruct telemetry coverage audit.');
    }
  }
}

export function validateLiveFirstInstanceEnvironmentAudit(report, metadata) {
  const comparison = compareComputeProcessIdentitySets(
    report.computeProcesses?.pre,
    report.computeProcesses?.post,
  );
  const audit = requireExactRecordKeys(
    metadata.liveFirstInstanceEnvironmentAudit,
    [
      'schemaVersion',
      'kind',
      'telemetryStatus',
      'telemetryMalformedLineCount',
      'telemetryStderrByteCount',
      'telemetrySampleCount',
      'telemetryCoveragePass',
      'adapterTelemetryAssociation',
      'computeProcessIdentityComparison',
      'candidateEnvironmentGate',
      'overallEvidenceDecision',
    ],
    'metadata.json liveFirstInstanceEnvironmentAudit',
  );
  const expectedAudit = createLiveFirstInstanceEnvironmentAudit({
    evidenceStatus: metadata.evidenceStatus,
    telemetryReport: report,
    adapterInfo: metadata.environment?.benchmarkPage?.adapterInfo,
    computeProcessIdentityComparison: comparison,
    preregisteredNumericalDecision:
      metadata.liveFirstInstanceAnalysisAudit?.preregisteredNumericalDecision ?? null,
  });
  if (JSON.stringify(audit) !== JSON.stringify(expectedAudit)) {
    failVerification(
      'metadata.json liveFirstInstanceEnvironmentAudit does not reconstruct from telemetry evidence.',
    );
  }
  if (metadata.evidenceStatus === 'candidate'
    && expectedAudit.candidateEnvironmentGate.retryable === true) {
    failVerification(
      'first-instance-live completed candidate contains a retryable telemetry collector failure.',
    );
  }
  return {
    adapterTelemetryAssociation: audit.adapterTelemetryAssociation,
    candidateEnvironmentGate: audit.candidateEnvironmentGate,
    overallEvidenceDecision: audit.overallEvidenceDecision,
  };
}

function validateLiveFirstInstanceEnvironmentEvidence(
  report,
  metadata,
  contentsByName,
) {
  validateLiveFirstInstanceEnvironmentAudit(report, metadata);

  const gateLink = requireExactRecordKeys(
    metadata.liveForcedFeatureOffGate,
    ['artifact', 'accepted', 'sha256'],
    'metadata.json liveForcedFeatureOffGate',
  );
  if (gateLink.artifact !== 'forced-feature-off-evidence.json'
    || gateLink.accepted !== true) {
    failVerification(
      'metadata.json does not accept the required forced-feature-off deployment gate.',
    );
  }
  const gateArtifact = requireRecord(
    parseJsonArtifact(
      contentsByName.get('forced-feature-off-evidence.json'),
      'forced-feature-off-evidence.json',
    ),
    'forced-feature-off-evidence.json',
  );
  requireSha256(gateLink.sha256, 'metadata.json liveForcedFeatureOffGate.sha256');
  if (sha256Json(gateArtifact) !== gateLink.sha256) {
    failVerification(
      'forced-feature-off-evidence.json does not match its metadata SHA-256.',
    );
  }
  if (gateArtifact.accepted !== true
    || requireArray(
      gateArtifact.rejectionReasons,
      'forced-feature-off-evidence.json rejectionReasons',
    ).length !== 0
    || requireArray(
      gateArtifact.pageErrors,
      'forced-feature-off-evidence.json pageErrors',
    ).length !== 0
    || requireArray(
      gateArtifact.webgpuUncapturedErrors,
      'forced-feature-off-evidence.json webgpuUncapturedErrors',
    ).length !== 0) {
    failVerification('forced-feature-off-evidence.json is not cleanly accepted.');
  }
  const gateReasons = validateLiveFirstInstanceForcedFeatureOffGate(
    gateArtifact.evidence,
  );
  if (gateReasons.length !== 0) {
    failVerification(
      `forced-feature-off-evidence.json payload failed: ${gateReasons.join('; ')}.`,
    );
  }
}

function orderedValuesMatch(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function exactScalarRecordMatch(actual, expected) {
  if (actual === null || typeof actual !== 'object' || Array.isArray(actual)) return false;
  const actualKeys = Object.keys(actual).sort();
  const expectedKeys = Object.keys(expected).sort();
  return orderedValuesMatch(actualKeys, expectedKeys)
    && expectedKeys.every((key) => actual[key] === expected[key]);
}

export function validateProtocolMatrix(protocol) {
  requireRecord(protocol, 'metadata.json protocol');
  if (protocol.matrixKind === FIXED_SLICE_REPRESENTATION_MATRIX) {
    if (!orderedValuesMatch(protocol.modes, FIXED_SLICE_REPRESENTATION_MODES)) {
      failVerification(
        'fixed-slice-representation protocol modes must be exactly fixed-slice-per-bucket, fixed-slice.',
      );
    }
    if (protocol.repetitions !== FIXED_SLICE_REPRESENTATION_REPETITIONS) {
      failVerification('fixed-slice-representation protocol must use exactly six repetitions.');
    }
    if (!orderedValuesMatch(
      protocol.visibilityLevels,
      FIXED_SLICE_REPRESENTATION_VISIBILITIES,
    )) {
      failVerification(
        'fixed-slice-representation visibility levels must be exactly 0.2, 0.8, 0.99.',
      );
    }
    if (protocol.heterogeneousComparator !== null) {
      failVerification('fixed-slice-representation heterogeneousComparator must be null.');
    }
    if (protocol.ordering !== FIXED_SLICE_REPRESENTATION_ORDERING) {
      failVerification(
        `fixed-slice-representation ordering must be ${JSON.stringify(FIXED_SLICE_REPRESENTATION_ORDERING)}.`,
      );
    }
    if (protocol.warmupFrames !== FIXED_SLICE_REPRESENTATION_WARMUP_FRAMES
      || protocol.measuredFrames !== FIXED_SLICE_REPRESENTATION_MEASURED_FRAMES) {
      failVerification(
        'fixed-slice-representation protocol must use 300 warmup and 240 measured frames.',
      );
    }
    if (!FIXED_SLICE_REPRESENTATION_OBJECT_COUNTS.includes(protocol.objectCount)
      || !FIXED_SLICE_REPRESENTATION_BUCKET_COUNTS.includes(protocol.bucketCount)) {
      failVerification('fixed-slice-representation protocol has an unsupported workload size.');
    }
    if (protocol.matrix
      !== `${FIXED_SLICE_REPRESENTATION_MATRIX}-o${protocol.objectCount}-b${protocol.bucketCount}`) {
      failVerification('fixed-slice-representation matrix identifier does not match its workload.');
    }
    const expectedScaleRole = protocol.bucketCount === 1
      ? 'negative-control-equal-mesh-render-object-count'
      : 'primary-one-versus-b-mesh-render-object-representation-ablation';
    if (protocol.representationScaleRole !== expectedScaleRole) {
      failVerification('fixed-slice-representation scale role does not match its bucket count.');
    }
    return;
  }

  if (protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    if (!orderedValuesMatch(protocol.modes, [FIRST_INSTANCE_LIVE_CROSSOVER_MODE])
      || !orderedValuesMatch(protocol.layouts, LIVE_FIRST_INSTANCE_CROSSOVER_LAYOUTS)
      || !orderedValuesMatch(
        protocol.visibilityLevels,
        FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS,
      )) {
      failVerification(
        'first-instance-live protocol has the wrong mode, layout, or visibility levels.',
      );
    }
    if (protocol.repetitions !== FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS
      || protocol.warmupFrames !== FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES
      || protocol.measuredFrames !== FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES) {
      failVerification(
        'first-instance-live protocol has the wrong repetitions or frame counts.',
      );
    }
    if (protocol.objectCount !== LIVE_FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT
      || protocol.bucketCount !== LIVE_FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT
      || protocol.depthBinCount !== null) {
      failVerification(
        'first-instance-live protocol must use 65536 objects, 32 buckets, and no depth bins.',
      );
    }
    if (!orderedValuesMatch(
      protocol.allowedObjectCounts,
      FIXED_SLICE_REPRESENTATION_OBJECT_COUNTS,
    )
      || !orderedValuesMatch(
        protocol.allowedBucketCounts,
        FIXED_SLICE_REPRESENTATION_BUCKET_COUNTS,
      )
      || !orderedValuesMatch(
        protocol.allowedHeterogeneousComparators,
        ['coalesced-v11', 'historical-v10'],
      )) {
      failVerification(
        'first-instance-live protocol changes the runner workload/comparator domain.',
      );
    }
    if (protocol.matrix
      !== `${LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX}-o${LIVE_FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT}-b${LIVE_FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT}`) {
      failVerification(
        'first-instance-live matrix identifier does not match its exact workload.',
      );
    }
    if (protocol.reversedDepthBuffer !== true
      || protocol.minimumStorageBuffersPerShaderStage !== 8
      || protocol.heterogeneousComparator !== null
      || protocol.representationScaleRole !== null
      || protocol.threeBlocksScheduling !== null) {
      failVerification(
        'first-instance-live protocol has the wrong renderer, storage, or comparator contract.',
      );
    }
    if (protocol.maximumCpuTimerQuantumMs !== 0.01
      || protocol.ordering !== LIVE_FIRST_INSTANCE_CROSSOVER_ORDERING
      || protocol.renderParity !== LIVE_FIRST_INSTANCE_CROSSOVER_RENDER_PARITY) {
      failVerification(
        'first-instance-live protocol has the wrong timer, ordering, or parity contract.',
      );
    }
    const crossover = requireRecord(
      protocol.firstInstanceLiveCrossover,
      'metadata.json protocol.firstInstanceLiveCrossover',
    );
    const crossoverKeys = [
      'requiredFeature',
      'lanes',
      'blockSize',
      'warmupBlocks',
      'measuredBlocks',
      'patterns',
      'scheduleDesign',
      'expectedMeasuredRowsPerLane',
      'expectedMeasuredTransitionCounts',
      'expectedMeasuredHistoryTripleCounts',
      'expectedComputeCallsPerFrame',
      'expectedRenderCallsPerFrame',
      'expectedComputeTimestampUidCount',
      'expectedRenderTimestampUidCount',
      'commandBuffers',
      'commandBufferByteOffset',
      'commandRecordsPerBuffer',
      'scheduleSha256ByOrientation',
    ];
    if (Object.keys(crossover).length !== crossoverKeys.length
      || crossoverKeys.some((key) => !Object.hasOwn(crossover, key))) {
      failVerification(
        'first-instance-live firstInstanceLiveCrossover has an unexpected schema.',
      );
    }
    if (crossover.requiredFeature !== 'indirect-first-instance'
      || !orderedValuesMatch(crossover.lanes, FIRST_INSTANCE_LIVE_CROSSOVER_LANES)
      || crossover.blockSize !== FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE
      || crossover.warmupBlocks !== FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS
      || crossover.measuredBlocks !== FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS
      || !orderedValuesMatch(
        crossover.patterns,
        LIVE_FIRST_INSTANCE_CROSSOVER_PATTERNS_AS_STRINGS,
      )
      || crossover.scheduleDesign
        !== 'cyclic-binary-de-bruijn-order-three-with-complementary-orientation'
      || crossover.expectedMeasuredRowsPerLane
        !== FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES / 2
      || !exactScalarRecordMatch(
        crossover.expectedMeasuredTransitionCounts,
        LIVE_FIRST_INSTANCE_CROSSOVER_MEASURED_HISTORY.transitionCounts,
      )
      || !exactScalarRecordMatch(
        crossover.expectedMeasuredHistoryTripleCounts,
        LIVE_FIRST_INSTANCE_CROSSOVER_MEASURED_HISTORY.historyTripleCounts,
      )
      || crossover.expectedComputeCallsPerFrame !== 1
      || crossover.expectedRenderCallsPerFrame !== 1
      || crossover.expectedComputeTimestampUidCount !== 1
      || crossover.expectedRenderTimestampUidCount !== 1
      || crossover.commandBuffers !== FIRST_INSTANCE_LIVE_CROSSOVER_LANES.length
      || crossover.commandBufferByteOffset !== 0
      || crossover.commandRecordsPerBuffer !== LIVE_FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT) {
      failVerification(
        'first-instance-live crossover constants differ from the preregistration.',
      );
    }
    const scheduleDigests = requireRecord(
      crossover.scheduleSha256ByOrientation,
      'metadata.json protocol.firstInstanceLiveCrossover.scheduleSha256ByOrientation',
    );
    if (Object.keys(scheduleDigests).length !== 2
      || scheduleDigests['0'] !== liveFirstInstanceCrossoverScheduleSha256(0)
      || scheduleDigests['1'] !== liveFirstInstanceCrossoverScheduleSha256(1)) {
      failVerification(
        'first-instance-live schedule commitments are incomplete or inconsistent.',
      );
    }
    return;
  }

  if (protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX) {
    if (!orderedValuesMatch(protocol.modes, [FIRST_INSTANCE_CROSSOVER_MODE])
      || !orderedValuesMatch(protocol.layouts, FIRST_INSTANCE_CROSSOVER_LAYOUTS)
      || !orderedValuesMatch(
        protocol.visibilityLevels,
        FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS,
      )) {
      failVerification(
        'first-instance-render-only protocol has the wrong mode, layout, or visibility levels.',
      );
    }
    if (protocol.repetitions !== FIRST_INSTANCE_CROSSOVER_REPETITIONS
      || protocol.warmupFrames !== FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES
      || protocol.measuredFrames !== FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES) {
      failVerification(
        'first-instance-render-only protocol has the wrong repetitions or frame counts.',
      );
    }
    if (protocol.objectCount !== FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT
      || protocol.bucketCount !== FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT
      || protocol.depthBinCount !== null) {
      failVerification(
        'first-instance-render-only protocol must use 65536 objects, 32 buckets, and no depth bins.',
      );
    }
    if (!orderedValuesMatch(
      protocol.allowedObjectCounts,
      FIXED_SLICE_REPRESENTATION_OBJECT_COUNTS,
    )
      || !orderedValuesMatch(
        protocol.allowedBucketCounts,
        FIXED_SLICE_REPRESENTATION_BUCKET_COUNTS,
      )
      || !orderedValuesMatch(
        protocol.allowedHeterogeneousComparators,
        ['coalesced-v11', 'historical-v10'],
      )) {
      failVerification(
        'first-instance-render-only protocol changes the runner workload/comparator domain.',
      );
    }
    if (protocol.matrix
      !== `${FIRST_INSTANCE_CROSSOVER_MATRIX}-o${FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT}-b${FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT}`) {
      failVerification(
        'first-instance-render-only matrix identifier does not match its exact workload.',
      );
    }
    if (protocol.reversedDepthBuffer !== true
      || protocol.minimumStorageBuffersPerShaderStage !== null
      || protocol.heterogeneousComparator !== null
      || protocol.representationScaleRole !== null
      || protocol.threeBlocksScheduling !== null) {
      failVerification(
        'first-instance-render-only protocol has the wrong renderer or comparator contract.',
      );
    }
    if (protocol.maximumCpuTimerQuantumMs !== 0.01
      || protocol.ordering !== FIRST_INSTANCE_CROSSOVER_ORDERING
      || protocol.renderParity !== FIRST_INSTANCE_CROSSOVER_RENDER_PARITY) {
      failVerification(
        'first-instance-render-only protocol has the wrong timer, ordering, or parity contract.',
      );
    }
    const crossover = requireRecord(
      protocol.firstInstanceCrossover,
      'metadata.json protocol.firstInstanceCrossover',
    );
    const crossoverKeys = [
      'requiredFeature',
      'lanes',
      'blockSize',
      'warmupBlocks',
      'measuredBlocks',
      'patterns',
      'expectedMeasuredRowsPerLane',
      'expectedRenderCallsPerFrame',
      'expectedRenderTimestampUidCount',
      'expectedComputeTimestampsPerFrame',
      'commandSegments',
      'commandRecordsPerSegment',
      'scheduleSha256ByOrientation',
    ];
    if (Object.keys(crossover).length !== crossoverKeys.length
      || crossoverKeys.some((key) => !Object.hasOwn(crossover, key))) {
      failVerification(
        'first-instance-render-only firstInstanceCrossover has an unexpected schema.',
      );
    }
    if (crossover.requiredFeature !== 'indirect-first-instance'
      || !orderedValuesMatch(crossover.lanes, FIRST_INSTANCE_CROSSOVER_LANES)
      || crossover.blockSize !== FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE
      || crossover.warmupBlocks !== FIRST_INSTANCE_CROSSOVER_WARMUP_BLOCKS
      || crossover.measuredBlocks !== FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS
      || !orderedValuesMatch(
        crossover.patterns,
        FIRST_INSTANCE_CROSSOVER_PATTERNS_AS_STRINGS,
      )
      || crossover.expectedMeasuredRowsPerLane
        !== FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES / 2
      || crossover.expectedRenderCallsPerFrame !== 1
      || crossover.expectedRenderTimestampUidCount !== 1
      || crossover.expectedComputeTimestampsPerFrame !== 0
      || crossover.commandSegments !== FIRST_INSTANCE_CROSSOVER_LANES.length
      || crossover.commandRecordsPerSegment !== FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT) {
      failVerification(
        'first-instance-render-only crossover constants differ from the preregistration.',
      );
    }
    const scheduleDigests = requireRecord(
      crossover.scheduleSha256ByOrientation,
      'metadata.json protocol.firstInstanceCrossover.scheduleSha256ByOrientation',
    );
    if (Object.keys(scheduleDigests).length !== 2
      || scheduleDigests['0'] !== firstInstanceCrossoverScheduleSha256(0)
      || scheduleDigests['1'] !== firstInstanceCrossoverScheduleSha256(1)) {
      failVerification(
        'first-instance-render-only schedule commitments are incomplete or inconsistent.',
      );
    }
    return;
  }

  if (protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
    if (!orderedValuesMatch(protocol.modes, [FROZEN_DEPTH_CROSSOVER_MODE])) {
      failVerification('depth-ordering-render-only protocol must use exactly the frozen crossover mode.');
    }
    if (!orderedValuesMatch(protocol.layouts, FROZEN_DEPTH_CROSSOVER_LAYOUTS)
      || !orderedValuesMatch(
        protocol.visibilityLevels,
        FROZEN_DEPTH_CROSSOVER_VISIBILITIES,
      )) {
      failVerification('depth-ordering-render-only protocol has the wrong layouts or visibility level.');
    }
    if (protocol.repetitions !== FROZEN_DEPTH_CROSSOVER_REPETITIONS
      || protocol.warmupFrames !== FROZEN_CROSSOVER_WARMUP_FRAMES
      || protocol.measuredFrames !== FROZEN_CROSSOVER_MEASURED_FRAMES) {
      failVerification('depth-ordering-render-only protocol has the wrong repetitions or frame counts.');
    }
    if (protocol.objectCount !== FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT
      || protocol.bucketCount !== FROZEN_DEPTH_CROSSOVER_BUCKET_COUNT
      || protocol.depthBinCount !== FROZEN_DEPTH_CROSSOVER_BIN_COUNT) {
      failVerification('depth-ordering-render-only protocol must use 65536 objects, 32 buckets, and eight depth bins.');
    }
    if (protocol.matrix
      !== `${FROZEN_DEPTH_CROSSOVER_MATRIX}-o${FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT}-b${FROZEN_DEPTH_CROSSOVER_BUCKET_COUNT}`) {
      failVerification('depth-ordering-render-only matrix identifier does not match its exact workload.');
    }
    if (protocol.reversedDepthBuffer !== true
      || protocol.minimumStorageBuffersPerShaderStage
        !== DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS
      || protocol.heterogeneousComparator !== null
      || protocol.representationScaleRole !== null) {
      failVerification('depth-ordering-render-only protocol has the wrong renderer or comparator contract.');
    }
    if (protocol.ordering !== FROZEN_DEPTH_CROSSOVER_ORDERING
      || protocol.renderParity !== FROZEN_DEPTH_CROSSOVER_RENDER_PARITY) {
      failVerification('depth-ordering-render-only protocol has the wrong ordering or parity contract.');
    }
    const frozen = requireRecord(
      protocol.frozenCrossover,
      'metadata.json protocol.frozenCrossover',
    );
    const frozenKeys = [
      'lanes',
      'blockSize',
      'warmupBlocks',
      'measuredBlocks',
      'patterns',
      'expectedMeasuredRowsPerLane',
      'expectedRenderCallsPerFrame',
      'expectedRenderTimestampUidCount',
      'expectedComputeTimestampsPerFrame',
      'survivorBufferSegments',
      'survivorSegmentLength',
      'legalLaneBases',
      'scheduleSha256ByOrientation',
    ];
    if (Object.keys(frozen).length !== frozenKeys.length
      || frozenKeys.some((key) => !Object.hasOwn(frozen, key))) {
      failVerification('depth-ordering-render-only frozenCrossover has an unexpected schema.');
    }
    if (!orderedValuesMatch(frozen.lanes, FROZEN_DEPTH_CROSSOVER_LANES)
      || frozen.blockSize !== FROZEN_CROSSOVER_BLOCK_SIZE
      || frozen.warmupBlocks !== FROZEN_CROSSOVER_WARMUP_BLOCKS
      || frozen.measuredBlocks !== FROZEN_CROSSOVER_MEASURED_BLOCKS
      || !orderedValuesMatch(frozen.patterns, FROZEN_DEPTH_CROSSOVER_PATTERNS_AS_STRINGS)
      || frozen.expectedMeasuredRowsPerLane !== FROZEN_CROSSOVER_MEASURED_FRAMES / 2
      || frozen.expectedRenderCallsPerFrame !== 1
      || frozen.expectedRenderTimestampUidCount !== 1
      || frozen.expectedComputeTimestampsPerFrame !== 0
      || frozen.survivorBufferSegments !== 2
      || frozen.survivorSegmentLength !== FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT
      || !orderedValuesMatch(
        frozen.legalLaneBases,
        [0, FROZEN_DEPTH_CROSSOVER_OBJECT_COUNT],
      )) {
      failVerification('depth-ordering-render-only frozenCrossover constants differ from the preregistration.');
    }
    const scheduleDigests = requireRecord(
      frozen.scheduleSha256ByOrientation,
      'metadata.json protocol.frozenCrossover.scheduleSha256ByOrientation',
    );
    if (Object.keys(scheduleDigests).length !== 2
      || scheduleDigests['0'] !== frozenCrossoverScheduleSha256(0)
      || scheduleDigests['1'] !== frozenCrossoverScheduleSha256(1)) {
      failVerification('depth-ordering-render-only schedule commitments are incomplete or inconsistent.');
    }
    return;
  }

  if (protocol.matrixKind !== DEPTH_ORDERING_MATRIX) return;
  if (!orderedValuesMatch(protocol.modes, DEPTH_ORDERING_MODES)) {
    failVerification(
      'depth-ordering protocol modes must be exactly fixed-slice, fixed-slice-depth-front-to-back, fixed-slice-depth-reverse.',
    );
  }
  if (!orderedValuesMatch(protocol.layouts, DEPTH_ORDERING_LAYOUTS)) {
    failVerification(
      'depth-ordering layouts must be exactly high-overlap, low-overlap.',
    );
  }
  if (!orderedValuesMatch(protocol.visibilityLevels, DEPTH_ORDERING_VISIBILITIES)) {
    failVerification('depth-ordering visibility levels must be exactly 0.99.');
  }
  if (protocol.repetitions !== DEPTH_ORDERING_REPETITIONS) {
    failVerification('depth-ordering protocol must use exactly six repetitions.');
  }
  if (protocol.warmupFrames !== DEPTH_ORDERING_WARMUP_FRAMES
    || protocol.measuredFrames !== DEPTH_ORDERING_MEASURED_FRAMES) {
    failVerification('depth-ordering protocol must use 300 warmup and 240 measured frames.');
  }
  if (protocol.objectCount !== DEPTH_ORDERING_OBJECT_COUNT
    || protocol.bucketCount !== DEPTH_ORDERING_BUCKET_COUNT) {
    failVerification('depth-ordering protocol must use exactly 65536 objects and 32 buckets.');
  }
  if (protocol.matrix
    !== `${DEPTH_ORDERING_MATRIX}-o${DEPTH_ORDERING_OBJECT_COUNT}-b${DEPTH_ORDERING_BUCKET_COUNT}`) {
    failVerification('depth-ordering matrix identifier does not match its exact workload.');
  }
  if (protocol.depthBinCount !== DEPTH_ORDERING_BIN_COUNT) {
    failVerification('depth-ordering protocol must use exactly eight depth bins.');
  }
  if (protocol.reversedDepthBuffer !== true) {
    failVerification('depth-ordering protocol must require reversedDepthBuffer.');
  }
  if (protocol.minimumStorageBuffersPerShaderStage
    !== DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS) {
    failVerification('depth-ordering protocol must require exactly eight storage buffers per shader stage.');
  }
  if (protocol.heterogeneousComparator !== null) {
    failVerification('depth-ordering heterogeneousComparator must be null.');
  }
  if (protocol.representationScaleRole !== null) {
    failVerification('depth-ordering representationScaleRole must be null.');
  }
  if (protocol.ordering !== DEPTH_ORDERING_ORDERING) {
    failVerification(
      `depth-ordering ordering must be ${JSON.stringify(DEPTH_ORDERING_ORDERING)}.`,
    );
  }
  if (protocol.renderParity !== DEPTH_ORDERING_RENDER_PARITY) {
    failVerification(
      `depth-ordering renderParity must be ${JSON.stringify(DEPTH_ORDERING_RENDER_PARITY)}.`,
    );
  }
}

function requireMetadataCompleteness(metadata, manifest) {
  requireRecord(metadata, 'metadata.json');
  if (metadata.schemaVersion !== 2) {
    failVerification(`metadata.json schemaVersion must be 2; received ${JSON.stringify(metadata.schemaVersion)}.`);
  }
  const runId = requireNonemptyString(metadata.runId, 'metadata.json runId');
  if (manifest.runId !== runId) {
    failVerification('artifact-manifest.json and metadata.json runId values differ.');
  }
  if (metadata.status !== 'complete') {
    failVerification(`metadata.json status must be "complete"; received ${JSON.stringify(metadata.status)}.`);
  }
  if (metadata.error !== null) failVerification('metadata.json contains a run error.');
  const pageErrors = requireArray(metadata.pageErrors, 'metadata.json pageErrors');
  if (pageErrors.length !== 0) failVerification('metadata.json contains page errors.');
  if (!ALLOWED_EVIDENCE_STATUSES.has(metadata.evidenceStatus)) {
    failVerification(`metadata.json has unsupported evidenceStatus ${JSON.stringify(metadata.evidenceStatus)}.`);
  }
  const startedAt = requireIsoTimestamp(metadata.startedAt, 'metadata.json startedAt');
  const completedAt = requireIsoTimestamp(metadata.completedAt, 'metadata.json completedAt');
  if (Date.parse(completedAt) < Date.parse(startedAt)) {
    failVerification('metadata.json completedAt precedes startedAt.');
  }
  requireFiniteNumber(metadata.elapsedMs, 'metadata.json elapsedMs', { minimum: 0 });
  requireRecord(metadata.environment, 'metadata.json environment');
  const sourceProvenance = requireRecord(
    metadata.sourceProvenance,
    'metadata.json sourceProvenance',
  );
  requireRecord(sourceProvenance.start, 'metadata.json sourceProvenance.start');
  requireRecord(sourceProvenance.end, 'metadata.json sourceProvenance.end');
  if (sourceProvenance.stable !== true
    && sourceProvenance.stable !== false
    && sourceProvenance.stable !== null) {
    failVerification('metadata.json sourceProvenance.stable must be true, false, or null.');
  }
  requireRecord(metadata.workload, 'metadata.json workload');
  const protocol = requireRecord(metadata.protocol, 'metadata.json protocol');
  requireNonemptyString(protocol.matrix, 'metadata.json protocol.matrix');
  requireInteger(protocol.objectCount, 'metadata.json protocol.objectCount', { minimum: 1 });
  requireInteger(protocol.bucketCount, 'metadata.json protocol.bucketCount', { minimum: 1 });
  requireInteger(protocol.repetitions, 'metadata.json protocol.repetitions', { minimum: 1 });
  requireInteger(protocol.warmupFrames, 'metadata.json protocol.warmupFrames', { minimum: 1 });
  requireInteger(protocol.measuredFrames, 'metadata.json protocol.measuredFrames', { minimum: 1 });
  const modes = requireArray(protocol.modes, 'metadata.json protocol.modes');
  if (modes.length === 0 || modes.some((mode) => typeof mode !== 'string' || mode === '')) {
    failVerification('metadata.json protocol.modes must contain mode identifiers.');
  }
  const visibilityLevels = requireArray(
    protocol.visibilityLevels,
    'metadata.json protocol.visibilityLevels',
  );
  if (visibilityLevels.length === 0 || visibilityLevels.some((value) => !Number.isFinite(value))) {
    failVerification('metadata.json protocol.visibilityLevels must contain finite values.');
  }
  validateProtocolMatrix(protocol);
  if (protocol.matrixKind === DEPTH_ORDERING_MATRIX
    || protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX
    || protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
    || protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    const benchmarkPage = requireRecord(
      metadata.environment.benchmarkPage,
      'metadata.json environment.benchmarkPage',
    );
    if ((protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX
      || protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
      || protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX)
      && benchmarkPage.reversedDepth !== true) {
      failVerification(
        `${protocol.matrixKind} metadata does not prove camera reversed-depth operation.`,
      );
    }
    if (benchmarkPage.rendererReversedDepthBuffer !== true) {
      failVerification('depth-ordering metadata does not prove renderer reversed-depth operation.');
    }
    if (protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
      || protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
      const isLive = protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX;
      const matrixLabel = isLive ? 'first-instance-live' : 'first-instance-render-only';
      const browser = requireRecord(
        metadata.environment.browser,
        'metadata.json environment.browser',
      );
      const browserKeys = ['executable', 'version', 'headless', 'args', 'viewport'];
      if (Object.keys(browser).length !== browserKeys.length
        || browserKeys.some((key) => !Object.hasOwn(browser, key))) {
        failVerification(
          `${matrixLabel} browser metadata has an unexpected schema.`,
        );
      }
      requireNonemptyString(
        browser.executable,
        'metadata.json environment.browser.executable',
      );
      requireNonemptyString(
        browser.version,
        'metadata.json environment.browser.version',
      );
      if (browser.headless !== true
        || !orderedValuesMatch(browser.args, FIRST_INSTANCE_CROSSOVER_BROWSER_ARGS)) {
        failVerification(
          `${matrixLabel} browser launch identity differs from the runner.`,
        );
      }
      const browserViewport = requireRecord(
        browser.viewport,
        'metadata.json environment.browser.viewport',
      );
      if (Object.keys(browserViewport).length !== 3
        || browserViewport.width !== 1280
        || browserViewport.height !== 900
        || browserViewport.deviceScaleFactor !== 1) {
        failVerification(
          `${matrixLabel} browser viewport differs from the runner.`,
        );
      }
      if (benchmarkPage.indirectFirstInstanceAvailable !== true) {
        failVerification(
          `${matrixLabel} metadata does not prove indirect-first-instance availability.`,
        );
      }
      if (benchmarkPage.timestampAvailable !== true) {
        failVerification(
          `${matrixLabel} metadata does not prove GPU timestamp availability.`,
        );
      }
      if (benchmarkPage.crossOriginIsolated !== true) {
        failVerification(
          `${matrixLabel} metadata does not prove cross-origin isolation.`,
        );
      }
      if (!Number.isFinite(benchmarkPage.performanceNowQuantumMs)
        || benchmarkPage.performanceNowQuantumMs <= 0
        || benchmarkPage.performanceNowQuantumMs > protocol.maximumCpuTimerQuantumMs) {
        failVerification(
          `${matrixLabel} metadata does not prove the preregistered CPU timer quantum.`,
        );
      }
      if (benchmarkPage.threeRevision !== '185') {
        failVerification(
          `${matrixLabel} metadata does not prove the pinned Three.js revision.`,
        );
      }
      const pageViewport = requireRecord(
        benchmarkPage.viewport,
        'metadata.json environment.benchmarkPage.viewport',
      );
      if (Object.keys(pageViewport).length !== 3
        || pageViewport.width !== 1280
        || pageViewport.height !== 720
        || pageViewport.devicePixelRatio !== 1) {
        failVerification(
          `${matrixLabel} page viewport differs from the pinned render target.`,
        );
      }
      if (benchmarkPage.rendererBackend !== 'WebGPUBackend'
        || benchmarkPage.coordinateSystem !== 2001) {
        failVerification(
          `${matrixLabel} metadata does not prove the pinned WebGPU renderer identity.`,
        );
      }
      requireNonemptyString(
        benchmarkPage.userAgent,
        'metadata.json environment.benchmarkPage.userAgent',
      );
      const adapterInfo = requireRecord(
        benchmarkPage.adapterInfo,
        'metadata.json environment.benchmarkPage.adapterInfo',
      );
      if (Object.keys(adapterInfo).length !== FIRST_INSTANCE_CROSSOVER_ADAPTER_INFO_FIELDS.length
        || FIRST_INSTANCE_CROSSOVER_ADAPTER_INFO_FIELDS.some(
          (field) => !Object.hasOwn(adapterInfo, field),
        )) {
        failVerification(
          `${matrixLabel} adapterInfo has an unexpected schema.`,
        );
      }
      for (const field of FIRST_INSTANCE_CROSSOVER_ADAPTER_INFO_FIELDS.slice(0, -1)) {
        if (adapterInfo[field] !== null
          && (typeof adapterInfo[field] !== 'string' || adapterInfo[field].trim() === '')) {
          failVerification(
            `${matrixLabel} adapterInfo.${field} must be null or a non-empty string.`,
          );
        }
      }
      if (adapterInfo.isFallbackAdapter !== null
        && typeof adapterInfo.isFallbackAdapter !== 'boolean') {
        failVerification(
          `${matrixLabel} adapterInfo.isFallbackAdapter must be null or boolean.`,
        );
      }
      if (!['vendor', 'architecture', 'device', 'description'].some(
        (field) => typeof adapterInfo[field] === 'string',
      )) {
        failVerification(
          `${matrixLabel} adapterInfo lacks a non-empty adapter identity.`,
        );
      }
      const expectedBackend = `${adapterInfo.description ?? adapterInfo.device ?? 'WebGPU'} · ${adapterInfo.backend ?? 'unknown backend'}`;
      if (metadata.environment.backend !== expectedBackend) {
        failVerification(
          `${matrixLabel} runner backend identity differs from page adapterInfo.`,
        );
      }
      if (isLive) {
        requireInteger(
          benchmarkPage.maxStorageBuffersPerShaderStage,
          'metadata.json environment.benchmarkPage.maxStorageBuffersPerShaderStage',
        );
        if (benchmarkPage.maxStorageBuffersPerShaderStage < 8) {
          failVerification(
            'first-instance-live metadata reports fewer than eight storage buffers per shader stage.',
          );
        }
        for (const field of [
          'webgpuUncapturedErrorCount',
          'webgpuValidationErrorCount',
          'webgpuDeviceLossCount',
        ]) {
          if (benchmarkPage[field] !== 0) {
            failVerification(`first-instance-live benchmarkPage.${field} must be zero.`);
          }
        }
        const benchmarkPageAtEnd = requireRecord(
          metadata.environment.benchmarkPageAtEnd,
          'metadata.json environment.benchmarkPageAtEnd',
        );
        if (JSON.stringify(benchmarkPageAtEnd) !== JSON.stringify(benchmarkPage)) {
          failVerification('first-instance-live page environment changed during the matrix.');
        }
        const uncapturedErrors = requireArray(
          metadata.webgpuUncapturedErrors,
          'metadata.json webgpuUncapturedErrors',
        );
        if (uncapturedErrors.length !== 0) {
          failVerification('first-instance-live metadata contains uncaptured WebGPU errors.');
        }
      }
    } else {
      requireInteger(
        benchmarkPage.maxStorageBuffersPerShaderStage,
        'metadata.json environment.benchmarkPage.maxStorageBuffersPerShaderStage',
      );
      if (benchmarkPage.maxStorageBuffersPerShaderStage
        < DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS) {
        failVerification('depth-ordering metadata reports fewer than eight storage buffers per shader stage.');
      }
    }
  }
  const plan = requireArray(metadata.plan, 'metadata.json plan');
  if (plan.length === 0) failVerification('metadata.json plan is empty.');
  requireInteger(metadata.expectedTrialCount, 'metadata.json expectedTrialCount', { minimum: 1 });
  requireInteger(metadata.completedTrialCount, 'metadata.json completedTrialCount');
  requireInteger(metadata.acceptedTrialCount, 'metadata.json acceptedTrialCount');
  requireInteger(metadata.frameRowCount, 'metadata.json frameRowCount', { minimum: 1 });
  requireInteger(metadata.validationArtifactCount, 'metadata.json validationArtifactCount', { minimum: 1 });
  requireArray(metadata.validationArtifactSha256, 'metadata.json validationArtifactSha256');
  if (metadata.expectedTrialCount !== plan.length) {
    failVerification('metadata.json expectedTrialCount does not equal plan length.');
  }
  const layoutCount = protocol.matrixKind === DEPTH_ORDERING_MATRIX
    || protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX
    || protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
    || protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX
    ? requireArray(protocol.layouts, 'metadata.json protocol.layouts').length
    : 1;
  const protocolTrialCount = protocol.repetitions
    * modes.length
    * visibilityLevels.length
    * layoutCount;
  if (metadata.expectedTrialCount !== protocolTrialCount) {
    failVerification('metadata.json expectedTrialCount is inconsistent with the protocol matrix.');
  }
  if (metadata.completedTrialCount !== metadata.expectedTrialCount
    || metadata.acceptedTrialCount !== metadata.expectedTrialCount) {
    failVerification('metadata.json does not report every expected trial as completed and accepted.');
  }
  validateCandidateProvenance(metadata);
  validateLiveCandidateReservationAndDependencies(metadata);
  return { plan, protocol };
}

function requireMatchingIdentity(actual, expected, label) {
  const fields = [
    'trialId',
    'planIndex',
    'repetitionIndex',
    'modeId',
    'visibilityFraction',
  ];
  if (Object.hasOwn(expected, 'layout')) fields.push('layout');
  for (const field of fields) {
    if (actual[field] !== expected[field]) {
      failVerification(`${label} ${field} does not match metadata.json plan.`);
    }
  }
}

function requireExactPermutation(value, expectedValues, label) {
  const order = requireArray(value, label);
  if (order.length !== expectedValues.length
    || new Set(order).size !== order.length
    || expectedValues.some((expected) => !order.includes(expected))) {
    failVerification(`${label} must be an exact permutation of the protocol values.`);
  }
  return order;
}

export function validateBenchmarkPlan(plan, metadata) {
  validateProtocolMatrix(metadata.protocol);
  const isDepthOrdering = metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX;
  const isFrozenCrossover = metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX;
  const isFirstInstanceCrossover =
    metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX;
  const isLiveFirstInstanceCrossover =
    metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX;
  const hasLayouts = isDepthOrdering || isFrozenCrossover || isFirstInstanceCrossover
    || isLiveFirstInstanceCrossover;
  const byTrialId = new Map();
  const byPlanIndex = new Map();
  const matrixCells = new Set();
  const repetitionOrders = new Map();
  const modes = metadata.protocol.modes;
  const visibilityLevels = metadata.protocol.visibilityLevels;
  const layouts = hasLayouts ? metadata.protocol.layouts : null;
  for (const [arrayIndex, item] of plan.entries()) {
    requireRecord(item, `metadata.json plan[${arrayIndex}]`);
    const trialId = requireNonemptyString(item.trialId, `metadata.json plan[${arrayIndex}].trialId`);
    requireInteger(item.planIndex, `metadata.json plan[${arrayIndex}].planIndex`);
    requireInteger(item.repetitionIndex, `metadata.json plan[${arrayIndex}].repetitionIndex`);
    requireNonemptyString(item.modeId, `metadata.json plan[${arrayIndex}].modeId`);
    const modeOrder = requireExactPermutation(
      item.modeOrder,
      modes,
      `metadata.json plan[${arrayIndex}].modeOrder`,
    );
    requireInteger(
      item.modeOrderPosition,
      `metadata.json plan[${arrayIndex}].modeOrderPosition`,
    );
    if (modeOrder[item.modeOrderPosition] !== item.modeId) {
      failVerification(`metadata.json plan[${arrayIndex}] mode position is inconsistent.`);
    }
    requireFiniteNumber(
      item.visibilityFraction,
      `metadata.json plan[${arrayIndex}].visibilityFraction`,
      { minimum: 0 },
    );
    const visibilityOrder = requireExactPermutation(
      item.visibilityOrder,
      visibilityLevels,
      `metadata.json plan[${arrayIndex}].visibilityOrder`,
    );
    requireInteger(
      item.visibilityOrderPosition,
      `metadata.json plan[${arrayIndex}].visibilityOrderPosition`,
    );
    if (visibilityOrder[item.visibilityOrderPosition] !== item.visibilityFraction) {
      failVerification(`metadata.json plan[${arrayIndex}] visibility position is inconsistent.`);
    }
    let layoutOrder = null;
    if (hasLayouts) {
      requireNonemptyString(item.layout, `metadata.json plan[${arrayIndex}].layout`);
      layoutOrder = requireExactPermutation(
        item.layoutOrder,
        layouts,
        `metadata.json plan[${arrayIndex}].layoutOrder`,
      );
      requireInteger(
        item.layoutOrderPosition,
        `metadata.json plan[${arrayIndex}].layoutOrderPosition`,
      );
      if (layoutOrder[item.layoutOrderPosition] !== item.layout) {
        failVerification(`metadata.json plan[${arrayIndex}] layout position is inconsistent.`);
      }
    }
    if (isFrozenCrossover) {
      const laneStorageOrder = requireExactPermutation(
        item.laneStorageOrder,
        FROZEN_DEPTH_CROSSOVER_LANES,
        `metadata.json plan[${arrayIndex}].laneStorageOrder`,
      );
      requireInteger(
        item.superblockOrientationOffset,
        `metadata.json plan[${arrayIndex}].superblockOrientationOffset`,
        { minimum: 0, maximum: 1 },
      );
      if (!orderedValuesMatch(
        laneStorageOrder,
        FROZEN_DEPTH_CROSSOVER_STORAGE_ORDERS[item.repetitionIndex],
      ) || item.superblockOrientationOffset
        !== FROZEN_DEPTH_CROSSOVER_ORIENTATION_OFFSETS[item.repetitionIndex]) {
        failVerification(`metadata.json plan[${arrayIndex}] changes a preregistered frozen crossover factor.`);
      }
    }
    if (isFirstInstanceCrossover) {
      const laneCommandSegmentOrder = requireExactPermutation(
        item.laneCommandSegmentOrder,
        FIRST_INSTANCE_CROSSOVER_LANES,
        `metadata.json plan[${arrayIndex}].laneCommandSegmentOrder`,
      );
      requireInteger(
        item.superblockOrientationOffset,
        `metadata.json plan[${arrayIndex}].superblockOrientationOffset`,
      );
      if (item.superblockOrientationOffset !== 0
        && item.superblockOrientationOffset !== 1) {
        failVerification(
          `metadata.json plan[${arrayIndex}].superblockOrientationOffset must be 0 or 1.`,
        );
      }
      if (!orderedValuesMatch(
        laneCommandSegmentOrder,
        FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS[item.repetitionIndex],
      ) || item.superblockOrientationOffset
        !== FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS[item.repetitionIndex]) {
        failVerification(
          `metadata.json plan[${arrayIndex}] changes a preregistered first-instance crossover factor.`,
        );
      }
    }
    if (isLiveFirstInstanceCrossover) {
      const lanePhysicalOrder = requireExactPermutation(
        item.lanePhysicalOrder,
        FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
        `metadata.json plan[${arrayIndex}].lanePhysicalOrder`,
      );
      requireInteger(
        item.superblockOrientationOffset,
        `metadata.json plan[${arrayIndex}].superblockOrientationOffset`,
        { minimum: 0, maximum: 1 },
      );
      if (!orderedValuesMatch(
        lanePhysicalOrder,
        FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS[item.repetitionIndex],
      ) || item.superblockOrientationOffset
        !== FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS[item.repetitionIndex]) {
        failVerification(
          `metadata.json plan[${arrayIndex}] changes a preregistered live first-instance crossover factor.`,
        );
      }
    }
    if (item.runId !== metadata.runId) failVerification(`metadata.json plan[${arrayIndex}] has the wrong runId.`);
    requireInteger(item.objectCount, `metadata.json plan[${arrayIndex}].objectCount`, { minimum: 1 });
    requireInteger(item.bucketCount, `metadata.json plan[${arrayIndex}].bucketCount`, { minimum: 1 });
    if (!metadata.protocol.modes.includes(item.modeId)
      || !metadata.protocol.visibilityLevels.includes(item.visibilityFraction)
      || (hasLayouts && !layouts.includes(item.layout))
      || item.repetitionIndex >= metadata.protocol.repetitions
      || item.objectCount !== metadata.protocol.objectCount
      || item.bucketCount !== metadata.protocol.bucketCount) {
      failVerification(`metadata.json plan[${arrayIndex}] is inconsistent with the protocol.`);
    }
    if (item.planIndex !== arrayIndex) {
      failVerification('metadata.json plan indexes are not contiguous and ordered from zero.');
    }
    if ((isFirstInstanceCrossover || isLiveFirstInstanceCrossover)
      && trialId !== `${metadata.runId}-t${String(arrayIndex + 1).padStart(2, '0')}`) {
      failVerification(
        `metadata.json plan[${arrayIndex}] changes the exact first-instance trial identity.`,
      );
    }
    if (byTrialId.has(trialId) || byPlanIndex.has(item.planIndex)) {
      failVerification('metadata.json plan has duplicate trial identities.');
    }
    const cellKey = JSON.stringify(hasLayouts
      ? [item.repetitionIndex, item.layout, item.modeId, item.visibilityFraction]
      : [item.repetitionIndex, item.modeId, item.visibilityFraction]);
    if (matrixCells.has(cellKey)) {
      failVerification(hasLayouts
        ? 'metadata.json plan duplicates a repetition/layout/mode/visibility cell.'
        : 'metadata.json plan duplicates a repetition/mode/visibility cell.');
    }
    matrixCells.add(cellKey);

    const orderRecord = repetitionOrders.get(item.repetitionIndex);
    const modeOrderSignature = JSON.stringify(modeOrder);
    const visibilityOrderSignature = JSON.stringify(visibilityOrder);
    const layoutOrderSignature = hasLayouts ? JSON.stringify(layoutOrder) : null;
    if (orderRecord === undefined) {
      repetitionOrders.set(item.repetitionIndex, {
        modeOrder: [...modeOrder],
        visibilityOrder: [...visibilityOrder],
        layoutOrder: layoutOrder === null ? null : [...layoutOrder],
        modeOrderSignature,
        visibilityOrderSignature,
        layoutOrderSignature,
      });
    } else if (orderRecord.modeOrderSignature !== modeOrderSignature
      || orderRecord.visibilityOrderSignature !== visibilityOrderSignature
      || orderRecord.layoutOrderSignature !== layoutOrderSignature) {
      failVerification('metadata.json plan changes an order within one repetition.');
    }
    byTrialId.set(trialId, item);
    byPlanIndex.set(item.planIndex, item);
  }

  for (let repetition = 0; repetition < metadata.protocol.repetitions; repetition += 1) {
    if (!repetitionOrders.has(repetition)) {
      failVerification(`metadata.json plan omits repetition ${repetition}.`);
    }
    const expectedLayouts = hasLayouts ? layouts : [null];
    for (const layout of expectedLayouts) {
      for (const mode of modes) {
        for (const visibility of visibilityLevels) {
          const cellKey = JSON.stringify(hasLayouts
            ? [repetition, layout, mode, visibility]
            : [repetition, mode, visibility]);
          if (!matrixCells.has(cellKey)) {
            failVerification(hasLayouts
              ? 'metadata.json plan omits a repetition/layout/mode/visibility cell.'
              : 'metadata.json plan omits a repetition/mode/visibility cell.');
          }
        }
      }
    }
  }

  let executionIndex = 0;
  for (let repetition = 0; repetition < metadata.protocol.repetitions; repetition += 1) {
    const orderRecord = repetitionOrders.get(repetition);
    if (metadata.protocol.matrixKind === FIXED_SLICE_REPRESENTATION_MATRIX) {
      const expectedModeOrder = repetition % 2 === 0
        ? [...FIXED_SLICE_REPRESENTATION_MODES]
        : [...FIXED_SLICE_REPRESENTATION_MODES].reverse();
      const visibilityOffset = repetition % FIXED_SLICE_REPRESENTATION_VISIBILITIES.length;
      const expectedVisibilityOrder = [
        ...FIXED_SLICE_REPRESENTATION_VISIBILITIES.slice(visibilityOffset),
        ...FIXED_SLICE_REPRESENTATION_VISIBILITIES.slice(0, visibilityOffset),
      ];
      if (!orderedValuesMatch(orderRecord.modeOrder, expectedModeOrder)) {
        failVerification(
          'fixed-slice-representation mode orders must alternate AB/BA by repetition.',
        );
      }
      if (!orderedValuesMatch(orderRecord.visibilityOrder, expectedVisibilityOrder)) {
        failVerification(
          'fixed-slice-representation visibility orders must rotate by repetition.',
        );
      }
    }
    if (isFrozenCrossover) {
      const expectedLayoutOrder = repetition % 2 === 0
        ? FROZEN_DEPTH_CROSSOVER_LAYOUTS
        : [...FROZEN_DEPTH_CROSSOVER_LAYOUTS].reverse();
      if (!orderedValuesMatch(orderRecord.modeOrder, [FROZEN_DEPTH_CROSSOVER_MODE])
        || !orderedValuesMatch(
          orderRecord.visibilityOrder,
          FROZEN_DEPTH_CROSSOVER_VISIBILITIES,
        )
        || !orderedValuesMatch(orderRecord.layoutOrder, expectedLayoutOrder)) {
        failVerification('depth-ordering-render-only plan changes its exact mode, visibility, or layout order.');
      }
      for (let layoutPosition = 0;
        layoutPosition < expectedLayoutOrder.length;
        layoutPosition += 1) {
        const item = plan[executionIndex];
        if (item.repetitionIndex !== repetition
          || item.layoutOrderPosition !== layoutPosition
          || item.layout !== expectedLayoutOrder[layoutPosition]
          || item.visibilityOrderPosition !== 0
          || item.visibilityFraction !== FROZEN_DEPTH_CROSSOVER_VISIBILITIES[0]
          || item.modeOrderPosition !== 0
          || item.modeId !== FROZEN_DEPTH_CROSSOVER_MODE) {
          failVerification('metadata.json frozen crossover plan execution order is not repetition-contiguous and layout-paired.');
        }
        executionIndex += 1;
      }
      continue;
    }
    if (isFirstInstanceCrossover) {
      const expectedVisibilityOrder = repetition % 2 === 0
        ? [...FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS]
        : [...FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS].reverse();
      if (!orderedValuesMatch(orderRecord.modeOrder, [FIRST_INSTANCE_CROSSOVER_MODE])
        || !orderedValuesMatch(orderRecord.visibilityOrder, expectedVisibilityOrder)
        || !orderedValuesMatch(orderRecord.layoutOrder, FIRST_INSTANCE_CROSSOVER_LAYOUTS)) {
        failVerification(
          'first-instance-render-only plan changes its exact mode, visibility, or layout order.',
        );
      }
      for (let visibilityPosition = 0;
        visibilityPosition < expectedVisibilityOrder.length;
        visibilityPosition += 1) {
        const item = plan[executionIndex];
        if (item.repetitionIndex !== repetition
          || item.planIndex !== executionIndex
          || item.layoutOrderPosition !== 0
          || item.layout !== FIRST_INSTANCE_CROSSOVER_LAYOUTS[0]
          || item.visibilityOrderPosition !== visibilityPosition
          || item.visibilityFraction !== expectedVisibilityOrder[visibilityPosition]
          || item.modeOrderPosition !== 0
          || item.modeId !== FIRST_INSTANCE_CROSSOVER_MODE
          || !orderedValuesMatch(
            item.laneCommandSegmentOrder,
            FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS[repetition],
          )
          || item.superblockOrientationOffset
            !== FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS[repetition]) {
          failVerification(
            'metadata.json first-instance crossover plan execution order is not repetition-contiguous and visibility-paired.',
          );
        }
        executionIndex += 1;
      }
      continue;
    }
    if (isLiveFirstInstanceCrossover) {
      const expectedVisibilityOrder = repetition % 2 === 0
        ? [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS]
        : [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS].reverse();
      if (!orderedValuesMatch(orderRecord.modeOrder, [FIRST_INSTANCE_LIVE_CROSSOVER_MODE])
        || !orderedValuesMatch(orderRecord.visibilityOrder, expectedVisibilityOrder)
        || !orderedValuesMatch(
          orderRecord.layoutOrder,
          LIVE_FIRST_INSTANCE_CROSSOVER_LAYOUTS,
        )) {
        failVerification(
          'first-instance-live plan changes its exact mode, visibility, or layout order.',
        );
      }
      for (let visibilityPosition = 0;
        visibilityPosition < expectedVisibilityOrder.length;
        visibilityPosition += 1) {
        const item = plan[executionIndex];
        if (item.repetitionIndex !== repetition
          || item.planIndex !== executionIndex
          || item.layoutOrderPosition !== 0
          || item.layout !== LIVE_FIRST_INSTANCE_CROSSOVER_LAYOUTS[0]
          || item.visibilityOrderPosition !== visibilityPosition
          || item.visibilityFraction !== expectedVisibilityOrder[visibilityPosition]
          || item.modeOrderPosition !== 0
          || item.modeId !== FIRST_INSTANCE_LIVE_CROSSOVER_MODE
          || !orderedValuesMatch(
            item.lanePhysicalOrder,
            FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS[repetition],
          )
          || item.superblockOrientationOffset
            !== FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS[repetition]) {
          failVerification(
            'metadata.json live first-instance crossover plan execution order is not repetition-contiguous and visibility-paired.',
          );
        }
        executionIndex += 1;
      }
      continue;
    }
    if (isDepthOrdering) {
      const expectedModeOrder = DEPTH_ORDERING_MODE_ORDERS[repetition];
      const expectedLayoutOrder = repetition % 2 === 0
        ? DEPTH_ORDERING_LAYOUTS
        : [...DEPTH_ORDERING_LAYOUTS].reverse();
      if (!orderedValuesMatch(orderRecord.modeOrder, expectedModeOrder)) {
        failVerification(
          'depth-ordering mode orders must use the exact six balanced permutations.',
        );
      }
      if (!orderedValuesMatch(orderRecord.layoutOrder, expectedLayoutOrder)) {
        failVerification(
          'depth-ordering layout orders must alternate high/low and low/high by repetition.',
        );
      }
      if (!orderedValuesMatch(orderRecord.visibilityOrder, DEPTH_ORDERING_VISIBILITIES)) {
        failVerification('depth-ordering visibility order must be exactly 0.99.');
      }
      for (let layoutPosition = 0;
        layoutPosition < orderRecord.layoutOrder.length;
        layoutPosition += 1) {
        for (let modePosition = 0;
          modePosition < orderRecord.modeOrder.length;
          modePosition += 1) {
          const item = plan[executionIndex];
          if (item.repetitionIndex !== repetition
            || item.layoutOrderPosition !== layoutPosition
            || item.layout !== orderRecord.layoutOrder[layoutPosition]
            || item.visibilityOrderPosition !== 0
            || item.visibilityFraction !== DEPTH_ORDERING_VISIBILITIES[0]
            || item.modeOrderPosition !== modePosition
            || item.modeId !== orderRecord.modeOrder[modePosition]) {
            failVerification(
              'metadata.json depth-ordering plan execution order must be repetition-contiguous with layout outer and mode inner.',
            );
          }
          executionIndex += 1;
        }
      }
      continue;
    }
    for (let visibilityPosition = 0;
      visibilityPosition < orderRecord.visibilityOrder.length;
      visibilityPosition += 1) {
      for (let modePosition = 0;
        modePosition < orderRecord.modeOrder.length;
        modePosition += 1) {
        const item = plan[executionIndex];
        if (item.repetitionIndex !== repetition
          || item.visibilityOrderPosition !== visibilityPosition
          || item.visibilityFraction !== orderRecord.visibilityOrder[visibilityPosition]
          || item.modeOrderPosition !== modePosition
          || item.modeId !== orderRecord.modeOrder[modePosition]) {
          failVerification(
            'metadata.json plan execution order must be repetition-contiguous with visibility outer and mode inner.',
          );
        }
        executionIndex += 1;
      }
    }
  }

  if (metadata.protocol.repetitions % modes.length === 0) {
    const expectedPerPosition = metadata.protocol.repetitions / modes.length;
    for (const mode of modes) {
      for (let position = 0; position < modes.length; position += 1) {
        const count = [...repetitionOrders.values()]
          .filter((record) => record.modeOrder[position] === mode).length;
        if (count !== expectedPerPosition) {
          failVerification('metadata.json mode ordering is not position-balanced.');
        }
      }
    }
  }
  if (metadata.protocol.repetitions % visibilityLevels.length === 0) {
    const expectedPerPosition = metadata.protocol.repetitions / visibilityLevels.length;
    for (const visibility of visibilityLevels) {
      for (let position = 0; position < visibilityLevels.length; position += 1) {
        const count = [...repetitionOrders.values()]
          .filter((record) => record.visibilityOrder[position] === visibility).length;
        if (count !== expectedPerPosition) {
          failVerification('metadata.json visibility ordering is not position-balanced.');
        }
      }
    }
  }
  if (hasLayouts && metadata.protocol.repetitions % layouts.length === 0) {
    const expectedPerPosition = metadata.protocol.repetitions / layouts.length;
    for (const layout of layouts) {
      for (let position = 0; position < layouts.length; position += 1) {
        const count = [...repetitionOrders.values()]
          .filter((record) => record.layoutOrder[position] === layout).length;
        if (count !== expectedPerPosition) {
          failVerification('metadata.json layout ordering is not position-balanced.');
        }
      }
    }
  }
  return { byTrialId, byPlanIndex };
}

function validateTrialSummaries(trialSummaries, metadata, planIndex) {
  requireArray(trialSummaries, 'trial-summaries.json');
  if (trialSummaries.length !== metadata.expectedTrialCount) {
    failVerification('trial-summaries.json count does not equal metadata expectedTrialCount.');
  }
  const byTrialId = new Map();
  for (const [index, summary] of trialSummaries.entries()) {
    requireRecord(summary, `trial-summaries.json[${index}]`);
    const trialId = requireNonemptyString(summary.trialId, `trial-summaries.json[${index}].trialId`);
    if (byTrialId.has(trialId)) failVerification('trial-summaries.json has duplicate trial IDs.');
    const planned = planIndex.byTrialId.get(trialId);
    if (planned === undefined) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} is absent from the plan.`);
    }
    if (planned.planIndex !== index) {
      failVerification('trial-summaries.json is not in exact plan-index order.');
    }
    requireMatchingIdentity(summary, planned, `trial-summaries.json trial ${JSON.stringify(trialId)}`);
    if (summary.objectCount !== planned.objectCount || summary.bucketCount !== planned.bucketCount) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} workload size differs from the plan.`);
    }
    if (summary.accepted !== true) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} was not accepted.`);
    }
    const rejectionReasons = requireArray(
      summary.rejectionReasons,
      `trial-summaries.json trial ${JSON.stringify(trialId)} rejectionReasons`,
    );
    if (rejectionReasons.length !== 0) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} has rejection reasons.`);
    }
    requireIsoTimestamp(
      summary.startedAt,
      `trial-summaries.json trial ${JSON.stringify(trialId)} startedAt`,
    );
    requireIsoTimestamp(
      summary.completedAt,
      `trial-summaries.json trial ${JSON.stringify(trialId)} completedAt`,
    );
    requireFiniteNumber(
      summary.elapsedMs,
      `trial-summaries.json trial ${JSON.stringify(trialId)} elapsedMs`,
      { minimum: 0 },
    );
    const validation = requireRecord(
      summary.validation,
      `trial-summaries.json trial ${JSON.stringify(trialId)} validation`,
    );
    if (validation.pass !== true) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} lacks passing validation.`);
    }
    requireSha256(
      validation.artifactSha256,
      `trial-summaries.json trial ${JSON.stringify(trialId)} validation artifactSha256`,
    );
    const timestamps = requireRecord(
      summary.timestamps,
      `trial-summaries.json trial ${JSON.stringify(trialId)} timestamps`,
    );
    if (timestamps.accepted !== true
      || timestamps.available !== true
      || timestamps.rowCount !== metadata.protocol.measuredFrames
      || timestamps.missingRenderFrames !== 0
      || timestamps.missingComputeFrames !== 0) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} lacks complete accepted timestamps.`);
    }
    if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX
      || metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX
      || metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
      || metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
      requireRecord(
        summary.completionInvariant,
        `trial-summaries.json trial ${JSON.stringify(trialId)} completionInvariant`,
      );
    }
    if (metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
      const plannedScheduleSha256 = metadata.protocol.frozenCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)];
      if (validation.kind !== FROZEN_DEPTH_CROSSOVER_VALIDATION_KIND
        || !orderedValuesMatch(summary.laneStorageOrder, planned.laneStorageOrder)
        || summary.superblockOrientationOffset !== planned.superblockOrientationOffset
        || summary.plannedScheduleSha256 !== plannedScheduleSha256) {
        failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} changes its frozen plan commitments.`);
      }
      if (timestamps.expectedRenderTimestampUidCount !== 1
        || timestamps.invalidRenderTimestampUidCountFrames !== 0
        || !Number.isFinite(timestamps.quantumNs)
        || timestamps.quantumNs <= 0
        || timestamps.quantumNs > FROZEN_DEPTH_CROSSOVER_MAXIMUM_TIMESTAMP_QUANTUM_NS) {
        failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} violates the frozen timestamp contract.`);
      }
      const selected = requireRecord(
        summary.selectedConfig,
        `trial-summaries.json trial ${JSON.stringify(trialId)} selectedConfig`,
      );
      if (selected.strategyId !== planned.modeId
        || selected.objectCount !== planned.objectCount
        || selected.bucketCount !== planned.bucketCount
        || selected.visibilityFraction !== planned.visibilityFraction
        || selected.layout !== planned.layout
        || !orderedValuesMatch(selected.laneStorageOrder, planned.laneStorageOrder)
        || selected.superblockOrientationOffset !== planned.superblockOrientationOffset) {
        failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} selectedConfig differs from its frozen plan.`);
      }
    }
    if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX) {
      const plannedScheduleSha256 = metadata.protocol.firstInstanceCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)];
      const semanticSha256 = requireSha256(
        validation.firstInstanceSemanticSha256,
        `trial-summaries.json trial ${JSON.stringify(trialId)} validation firstInstanceSemanticSha256`,
      );
      if (validation.kind !== FIRST_INSTANCE_CROSSOVER_VALIDATION_KIND
        || !orderedValuesMatch(summary.modeOrder, planned.modeOrder)
        || !orderedValuesMatch(summary.visibilityOrder, planned.visibilityOrder)
        || !orderedValuesMatch(summary.layoutOrder, planned.layoutOrder)
        || !orderedValuesMatch(
          summary.laneCommandSegmentOrder,
          planned.laneCommandSegmentOrder,
        )
        || summary.superblockOrientationOffset !== planned.superblockOrientationOffset
        || summary.plannedScheduleSha256 !== plannedScheduleSha256) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} changes its first-instance plan commitments.`,
        );
      }
      if (timestamps.expectedRenderTimestampUidCount !== 1
        || timestamps.invalidRenderTimestampUidCountFrames !== 0
        || timestamps.classification !== 'fine'
        || !Number.isFinite(timestamps.quantumNs)
        || timestamps.quantumNs <= 0
        || timestamps.quantumNs > FIRST_INSTANCE_CROSSOVER_MAXIMUM_TIMESTAMP_QUANTUM_NS) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} violates the first-instance timestamp contract.`,
        );
      }
      const selected = requireRecord(
        summary.selectedConfig,
        `trial-summaries.json trial ${JSON.stringify(trialId)} selectedConfig`,
      );
      if (selected.strategyId !== planned.modeId
        || selected.objectCount !== planned.objectCount
        || selected.bucketCount !== planned.bucketCount
        || selected.visibilityFraction !== planned.visibilityFraction
        || selected.layout !== planned.layout
        || !orderedValuesMatch(
          selected.laneCommandSegmentOrder,
          planned.laneCommandSegmentOrder,
        )
        || selected.superblockOrientationOffset !== planned.superblockOrientationOffset) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} selectedConfig differs from its first-instance plan.`,
        );
      }
      const timing = requireRecord(
        summary.timing,
        `trial-summaries.json trial ${JSON.stringify(trialId)} timing`,
      );
      const timingKeys = Object.keys(timing);
      if (timingKeys.length !== FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA.length
        || FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA.some(
          (key) => !Object.hasOwn(timing, key),
        )) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} timing has an unexpected schema.`,
        );
      }
      for (const key of Object.values(FIRST_INSTANCE_CROSSOVER_TIMING_FIELDS).flat()) {
        requireFiniteNumber(
          timing[key],
          `trial-summaries.json trial ${JSON.stringify(trialId)} timing ${key}`,
          { minimum: 0 },
        );
      }
      if (timing.gpuComputeP50Ms !== null || timing.gpuComputeP95Ms !== null) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} timing contains compute durations in the render-only crossover.`,
        );
      }
      if (semanticSha256 !== validation.firstInstanceSemanticSha256) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} has an invalid first-instance semantic commitment.`,
        );
      }
    }
    if (metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
      requireExactRecordKeys(
        timestamps,
        LIVE_FIRST_INSTANCE_CROSSOVER_TIMESTAMP_SUMMARY_SCHEMA,
        `trial-summaries.json trial ${JSON.stringify(trialId)} timestamps`,
      );
      const plannedScheduleSha256 = metadata.protocol.firstInstanceLiveCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)];
      const semanticSha256 = requireSha256(
        validation.liveFirstInstanceSemanticSha256,
        `trial-summaries.json trial ${JSON.stringify(trialId)} validation liveFirstInstanceSemanticSha256`,
      );
      if (validation.kind !== LIVE_FIRST_INSTANCE_CROSSOVER_VALIDATION_KIND
        || !orderedValuesMatch(summary.modeOrder, planned.modeOrder)
        || !orderedValuesMatch(summary.visibilityOrder, planned.visibilityOrder)
        || !orderedValuesMatch(summary.layoutOrder, planned.layoutOrder)
        || !orderedValuesMatch(summary.lanePhysicalOrder, planned.lanePhysicalOrder)
        || summary.superblockOrientationOffset !== planned.superblockOrientationOffset
        || summary.plannedScheduleSha256 !== plannedScheduleSha256) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} changes its live first-instance plan commitments.`,
        );
      }
      if (timestamps.expectedComputeTimestampUidCount !== 1
        || timestamps.invalidComputeTimestampUidCountFrames !== 0
        || timestamps.invalidComputeTimestampDurationFrames !== 0
        || timestamps.expectedRenderTimestampUidCount !== 1
        || timestamps.invalidRenderTimestampUidCountFrames !== 0
        || timestamps.invalidRenderTimestampDurationFrames !== 0
        || timestamps.warmupRowCount !== metadata.protocol.warmupFrames
        || timestamps.missingWarmupComputeFrames !== 0
        || timestamps.missingWarmupRenderFrames !== 0
        || timestamps.invalidWarmupComputeTimestampUidCountFrames !== 0
        || timestamps.invalidWarmupRenderTimestampUidCountFrames !== 0
        || timestamps.invalidWarmupComputeTimestampDurationFrames !== 0
        || timestamps.invalidWarmupRenderTimestampDurationFrames !== 0
        || timestamps.renderTimestampPoolQualityValid !== true
        || timestamps.computeTimestampPoolQualityValid !== true
        || timestamps.warmupRenderTimestampPoolQualityValid !== true
        || timestamps.warmupComputeTimestampPoolQualityValid !== true
        || timestamps.warmupTimestampFrameCountValid !== true
        || timestamps.measurementTimestampFrameCountValid !== true
        || timestamps.classification !== 'fine'
        || !Number.isFinite(timestamps.quantumNs)
        || timestamps.quantumNs <= 0
        || timestamps.quantumNs > LIVE_FIRST_INSTANCE_CROSSOVER_MAXIMUM_TIMESTAMP_QUANTUM_NS) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} violates the live first-instance timestamp contract.`,
        );
      }
      const selected = requireRecord(
        summary.selectedConfig,
        `trial-summaries.json trial ${JSON.stringify(trialId)} selectedConfig`,
      );
      if (selected.strategyId !== planned.modeId
        || selected.objectCount !== planned.objectCount
        || selected.bucketCount !== planned.bucketCount
        || selected.visibilityFraction !== planned.visibilityFraction
        || selected.layout !== planned.layout
        || !orderedValuesMatch(selected.lanePhysicalOrder, planned.lanePhysicalOrder)
        || selected.superblockOrientationOffset !== planned.superblockOrientationOffset) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} selectedConfig differs from its live first-instance plan.`,
        );
      }
      const timing = requireRecord(
        summary.timing,
        `trial-summaries.json trial ${JSON.stringify(trialId)} timing`,
      );
      const timingKeys = Object.keys(timing);
      if (timingKeys.length !== LIVE_FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA.length
        || LIVE_FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA.some(
          (key) => !Object.hasOwn(timing, key),
        )) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} timing has an unexpected schema.`,
        );
      }
      for (const key of LIVE_FIRST_INSTANCE_CROSSOVER_TIMING_SCHEMA) {
        requireFiniteNumber(
          timing[key],
          `trial-summaries.json trial ${JSON.stringify(trialId)} timing ${key}`,
          { minimum: 0 },
        );
      }
      if (semanticSha256 !== validation.liveFirstInstanceSemanticSha256) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} has an invalid live first-instance semantic commitment.`,
        );
      }
    }
    byTrialId.set(trialId, summary);
  }
  return byTrialId;
}

function bodyWithoutSha256(record) {
  return Object.fromEntries(Object.entries(record).filter(([key]) => key !== 'sha256'));
}

function validateRenderParityCapture(parity, label, {
  planned,
  geometryManifest,
  scenarioManifest,
}) {
  requireRecord(parity, label);
  if (parity.schemaVersion !== 1
    || parity.kind !== 'fixed-camera-offscreen-exact-render-parity') {
    failVerification(`${label} has an unsupported schema or kind.`);
  }
  if (parity.pass !== true || parity.captures !== 2) {
    failVerification(`${label} is not passing evidence from exactly two captures.`);
  }
  if (parity.width !== DEPTH_ORDERING_PARITY_WIDTH
    || parity.height !== DEPTH_ORDERING_PARITY_HEIGHT) {
    failVerification(`${label} does not use the fixed 1280x720 viewport.`);
  }
  if (parity.reversedDepthBuffer !== true) {
    failVerification(`${label} did not use the required reversed depth buffer.`);
  }
  requireRecord(parity.material, `${label}.material`);
  const stability = requireRecord(parity.stability, `${label}.stability`);
  if (stability.pass !== true) failVerification(`${label}.stability did not pass.`);
  const first = requireRecord(stability.first, `${label}.stability.first`);
  const firstCapture = requireRecord(
    stability.firstCapture,
    `${label}.stability.firstCapture`,
  );
  const channelRequirements = {
    color: { format: 'rgba8unorm', arrayType: 'Uint8Array' },
    depth: { format: 'depth32float', arrayType: 'Float32Array' },
    objectId: { format: 'rgba8unorm-object-id-plus-one', arrayType: 'Uint8Array' },
  };
  for (const [name, expected] of Object.entries(channelRequirements)) {
    const channel = requireRecord(parity[name], `${label}.${name}`);
    if (channel.format !== expected.format || channel.arrayType !== expected.arrayType) {
      failVerification(`${label}.${name} has an unsupported format or array type.`);
    }
    if (channel.byteLength !== DEPTH_ORDERING_PARITY_BYTE_LENGTH) {
      failVerification(`${label}.${name} has an invalid byte length.`);
    }
    requireSha256(channel.sha256, `${label}.${name}.sha256`);
    if (first[`${name}Sha256`] !== channel.sha256) {
      failVerification(`${label}.${name} changed between its two captures.`);
    }
    if (JSON.stringify(firstCapture[name]) !== JSON.stringify(channel)) {
      failVerification(`${label}.${name} first-capture record is inconsistent.`);
    }
  }
  const objectIdValidation = requireRecord(
    parity.objectIdValidation,
    `${label}.objectIdValidation`,
  );
  if (objectIdValidation.pass !== true
    || objectIdValidation.encoding !== 'rgb24-object-id-plus-one-zero-background'
    || !Number.isInteger(objectIdValidation.coveredPixels)
    || objectIdValidation.coveredPixels <= 0
    || !Number.isInteger(objectIdValidation.backgroundPixels)
    || objectIdValidation.backgroundPixels < 0
    || objectIdValidation.coveredPixels + objectIdValidation.backgroundPixels
      !== DEPTH_ORDERING_PARITY_WIDTH * DEPTH_ORDERING_PARITY_HEIGHT
    || objectIdValidation.outOfRangePixels !== 0
    || objectIdValidation.nonVisiblePixels !== 0) {
    failVerification(`${label}.objectIdValidation did not prove a valid encoded ID domain.`);
  }
  const snapshotCheck = validateExactValidation(parity.snapshotValidation, {
    modeId: planned.modeId,
    objectCount: planned.objectCount,
    bucketCount: planned.bucketCount,
    expectedVisibleCount: scenarioManifest?.expectedVisibleCount,
    expectedVisibleIdsCanonicalSha256:
      scenarioManifest?.expectedVisibleIdsCanonicalSha256,
    geometryManifest,
    scenarioManifest,
  });
  if (snapshotCheck.rejectionReasons.length !== 0) {
    failVerification(
      `${label} same-snapshot validation failed: ${snapshotCheck.rejectionReasons.join('; ')}.`,
    );
  }
  return sha256Json({
    width: parity.width,
    height: parity.height,
    reversedDepthBuffer: parity.reversedDepthBuffer,
    material: parity.material,
    color: parity.color,
    depth: parity.depth,
    objectId: parity.objectId,
    objectIdValidation: parity.objectIdValidation,
    membershipSha256:
      parity.snapshotValidation?.membershipDigests?.actual?.sha256 ?? null,
  });
}

function validateEvidenceCapture(capture, label) {
  requireRecord(capture, label);
  requireIsoTimestamp(capture.capturedAt, `${label}.capturedAt`);
  if (capture.accepted !== true) failVerification(`${label} was not accepted.`);
  const rejectionReasons = requireArray(capture.rejectionReasons, `${label}.rejectionReasons`);
  if (rejectionReasons.length !== 0) failVerification(`${label} has rejection reasons.`);
  const workload = requireRecord(capture.workload, `${label}.workload`);
  requireInteger(workload.scenarioSeed, `${label}.workload.scenarioSeed`);
  requireSha256(workload.geometryFixtureSha256, `${label}.workload.geometryFixtureSha256`);
  requireSha256(workload.scenarioSha256, `${label}.workload.scenarioSha256`);
  const validation = requireRecord(capture.validation, `${label}.validation`);
  requireSha256(validation.payloadSha256, `${label}.validation.payloadSha256`);
  requireSha256(validation.semanticSha256, `${label}.validation.semanticSha256`);
  requireRecord(validation.payload, `${label}.validation.payload`);
  if (sha256Json(validation.payload) !== validation.payloadSha256) {
    failVerification(`${label}.validation payload SHA-256 is inconsistent.`);
  }
  return {
    scenarioSeed: workload.scenarioSeed,
    geometrySha256: workload.geometryFixtureSha256,
    scenarioSha256: workload.scenarioSha256,
    payloadSha256: validation.payloadSha256,
    semanticSha256: validation.semanticSha256,
    renderParity: capture.renderParity ?? null,
  };
}

function validateWorkloadManifests(catalog, metadata) {
  requireRecord(catalog, 'workload-manifests.json');
  if (catalog.schemaVersion !== 1 || catalog.hashAlgorithm !== 'sha256') {
    failVerification('workload-manifests.json has an unsupported schema or hash algorithm.');
  }
  const geometries = requireRecord(
    catalog.geometryFixturesBySha256,
    'workload-manifests.json geometryFixturesBySha256',
  );
  const scenarios = requireRecord(
    catalog.scenariosBySha256,
    'workload-manifests.json scenariosBySha256',
  );
  const invalid = requireArray(
    catalog.invalidObservations,
    'workload-manifests.json invalidObservations',
  );
  if (invalid.length !== 0) {
    failVerification('workload-manifests.json contains invalid workload observations.');
  }
  for (const [digest, record] of Object.entries(geometries)) {
    requireSha256(digest, 'workload-manifests.json geometry key');
    requireRecord(record, `workload-manifests.json geometry ${digest}`);
    if (record.sha256 !== digest) {
      failVerification(`workload-manifests.json geometry ${digest} has an inconsistent digest.`);
    }
  }
  for (const [digest, record] of Object.entries(scenarios)) {
    requireSha256(digest, 'workload-manifests.json scenario key');
    requireRecord(record, `workload-manifests.json scenario ${digest}`);
    if (record.sha256 !== digest) {
      failVerification(`workload-manifests.json scenario ${digest} has an inconsistent digest.`);
    }
  }
  if (metadata.workload.manifestArtifact !== 'workload-manifests.json') {
    failVerification('metadata workload does not link workload-manifests.json.');
  }
  const metadataGeometrySha256 = requireSha256(
    metadata.workload.geometryFixtureSha256,
    'metadata.json workload.geometryFixtureSha256',
  );
  if (geometries[metadataGeometrySha256] === undefined) {
    failVerification('metadata geometry digest is absent from workload-manifests.json.');
  }
  if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
    || metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    const isLive = metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX;
    const firstInstanceVisibilities = isLive
      ? FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS
      : FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS;
    const firstInstanceObjectCount = isLive
      ? LIVE_FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT
      : FIRST_INSTANCE_CROSSOVER_OBJECT_COUNT;
    const firstInstanceBucketCount = isLive
      ? LIVE_FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT
      : FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT;
    if (metadata.workload.scenarioGenerator !== 'createFixedSubsetScenario') {
      failVerification('first-instance workload uses an unexpected scenario generator.');
    }
    const scenarioSeedValue = requireInteger(
      metadata.workload.scenarioSeed,
      'metadata.json workload.scenarioSeed',
    );
    if (Object.keys(geometries).length !== 1) {
      failVerification(
        'first-instance workload catalog must contain exactly one geometry fixture manifest.',
      );
    }
    const geometryReasons = validateGeometryFixtureManifest(
      geometries[metadataGeometrySha256],
      { bucketCount: firstInstanceBucketCount, tier: 'medium' },
    );
    if (geometryReasons.length !== 0) {
      failVerification(
        `first-instance geometry fixture manifest failed: ${geometryReasons.join('; ')}.`,
      );
    }
    const visibilityLinks = requireRecord(
      metadata.workload.scenarioSha256ByVisibility,
      'metadata.json workload.scenarioSha256ByVisibility',
    );
    const cellLinks = requireRecord(
      metadata.workload.scenarioSha256ByCell,
      'metadata.json workload.scenarioSha256ByCell',
    );
    const parityLinks = requireRecord(
      metadata.workload.renderParitySha256ByCell,
      'metadata.json workload.renderParitySha256ByCell',
    );
    const visibilityKeys = firstInstanceVisibilities.map(String);
    const cellKeys = visibilityKeys.map((visibility) => `baseline|${visibility}`);
    const hasExactKeys = (record, expected) => {
      const actual = Object.keys(record);
      return actual.length === expected.length
        && expected.every((key) => Object.hasOwn(record, key));
    };
    if (!hasExactKeys(visibilityLinks, visibilityKeys)
      || !hasExactKeys(cellLinks, cellKeys)) {
      failVerification(
        'first-instance scenario links must exactly cover both preregistered visibility cells.',
      );
    }
    if (!hasExactKeys(parityLinks, cellKeys)) {
      failVerification(
        'first-instance render-parity links must exactly cover both preregistered visibility cells.',
      );
    }
    if (metadata.workload.physicalBinSequenceSha256ByPair !== null) {
      failVerification(
        'first-instance physicalBinSequenceSha256ByPair must be null.',
      );
    }
    const linkedScenarioDigests = new Set();
    for (const visibility of firstInstanceVisibilities) {
      const visibilityKey = String(visibility);
      const cellKey = `baseline|${visibilityKey}`;
      const digest = requireSha256(
        visibilityLinks[visibilityKey],
        `metadata first-instance scenario digest for visibility ${visibilityKey}`,
      );
      if (cellLinks[cellKey] !== digest) {
        failVerification(
          `metadata first-instance scenario aliases differ for ${cellKey}.`,
        );
      }
      const manifest = scenarios[digest];
      if (manifest === undefined) {
        failVerification(
          `metadata first-instance scenario digest for ${cellKey} is absent from the catalog.`,
        );
      }
      const scenarioReasons = validateScenarioManifest(manifest, {
        objectCount: firstInstanceObjectCount,
        bucketCount: firstInstanceBucketCount,
        visibilityFraction: visibility,
        seed: scenarioSeedValue,
        layout: 'baseline',
      });
      if (scenarioReasons.length !== 0) {
        failVerification(
          `first-instance scenario manifest ${JSON.stringify(cellKey)} failed: ${scenarioReasons.join('; ')}.`,
        );
      }
      requireSha256(
        parityLinks[cellKey],
        `metadata first-instance render-parity digest for ${cellKey}`,
      );
      linkedScenarioDigests.add(digest);
    }
    if (Object.keys(scenarios).length !== linkedScenarioDigests.size
      || Object.keys(scenarios).some((digest) => !linkedScenarioDigests.has(digest))) {
      failVerification(
        'first-instance workload catalog contains an unlinked or duplicate scenario manifest.',
      );
    }
    return { geometries, scenarios };
  }
  if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX
    || metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
    if (metadata.workload.scenarioSha256ByVisibility !== null) {
      failVerification('depth-ordering workload scenarioSha256ByVisibility must be null.');
    }
    const scenarioLinks = requireRecord(
      metadata.workload.scenarioSha256ByCell,
      'metadata.json workload.scenarioSha256ByCell',
    );
    const expectedKeys = metadata.protocol.layouts.flatMap((layout) => (
      metadata.protocol.visibilityLevels.map((visibility) => `${layout}|${visibility}`)
    ));
    const actualKeys = Object.keys(scenarioLinks);
    if (actualKeys.length !== expectedKeys.length
      || expectedKeys.some((key) => !Object.hasOwn(scenarioLinks, key))) {
      failVerification('depth-ordering workload scenarioSha256ByCell must exactly cover every layout/visibility cell.');
    }
    for (const key of expectedKeys) {
      const digest = scenarioLinks[key];
      requireSha256(digest, `metadata scenario digest for cell ${key}`);
      if (scenarios[digest] === undefined) {
        failVerification(`metadata scenario digest for cell ${key} is absent from the catalog.`);
      }
      const separator = key.lastIndexOf('|');
      const layout = key.slice(0, separator);
      const visibility = Number(key.slice(separator + 1));
      if (scenarios[digest].layout !== layout
        || scenarios[digest].visibilityFraction !== visibility) {
        failVerification(`metadata scenario digest for cell ${key} links the wrong scenario manifest.`);
      }
      requireInteger(
        scenarios[digest].expectedVisibleCount,
        `workload scenario ${digest} expectedVisibleCount`,
        { minimum: 1 },
      );
      requireSha256(
        scenarios[digest].expectedVisibleIdsCanonicalSha256,
        `workload scenario ${digest} expectedVisibleIdsCanonicalSha256`,
      );
      const depthBinRange = requireRecord(
        scenarios[digest].depthBinRange,
        `workload scenario ${digest} depthBinRange`,
      );
      requireFiniteNumber(
        depthBinRange.near,
        `workload scenario ${digest} depthBinRange.near`,
      );
      requireFiniteNumber(
        depthBinRange.far,
        `workload scenario ${digest} depthBinRange.far`,
      );
      if (depthBinRange.far <= depthBinRange.near) {
        failVerification(`workload scenario ${digest} depthBinRange must be increasing.`);
      }
    }
    const renderParityLinks = requireRecord(
      metadata.workload.renderParitySha256ByCell,
      'metadata.json workload.renderParitySha256ByCell',
    );
    const renderParityKeys = Object.keys(renderParityLinks);
    if (renderParityKeys.length !== expectedKeys.length
      || expectedKeys.some((key) => !Object.hasOwn(renderParityLinks, key))) {
      failVerification('depth-ordering workload renderParitySha256ByCell must exactly cover every layout/visibility cell.');
    }
    for (const key of expectedKeys) {
      requireSha256(renderParityLinks[key], `metadata render-parity digest for cell ${key}`);
    }
    const physicalSequenceLinks = requireRecord(
      metadata.workload.physicalBinSequenceSha256ByPair,
      'metadata.json workload.physicalBinSequenceSha256ByPair',
    );
    const expectedSequenceKeys = [];
    for (let repetition = 0; repetition < metadata.protocol.repetitions; repetition += 1) {
      for (const layout of metadata.protocol.layouts) {
        for (const visibility of metadata.protocol.visibilityLevels) {
          const scenarioSha256 = scenarioLinks[`${layout}|${visibility}`];
          expectedSequenceKeys.push(
            [repetition, layout, visibility, scenarioSha256].join('|'),
          );
        }
      }
    }
    const physicalSequenceKeys = Object.keys(physicalSequenceLinks);
    if (physicalSequenceKeys.length !== expectedSequenceKeys.length
      || expectedSequenceKeys.some((key) => !Object.hasOwn(physicalSequenceLinks, key))) {
      failVerification('depth-ordering workload physicalBinSequenceSha256ByPair must exactly cover every repetition/layout pair.');
    }
    for (const key of expectedSequenceKeys) {
      requireSha256(
        physicalSequenceLinks[key],
        `metadata physical-bin sequence digest for pair ${key}`,
      );
    }
  } else {
    const scenarioLinks = requireRecord(
      metadata.workload.scenarioSha256ByVisibility,
      'metadata.json workload.scenarioSha256ByVisibility',
    );
    for (const [visibility, digest] of Object.entries(scenarioLinks)) {
      requireSha256(digest, `metadata scenario digest for visibility ${visibility}`);
      if (scenarios[digest] === undefined) {
        failVerification(`metadata scenario digest for visibility ${visibility} is absent from the catalog.`);
      }
    }
  }
  return { geometries, scenarios };
}

function scenarioDigestForTrial(metadata, trial) {
  if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX
    || metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
    return metadata.workload.scenarioSha256ByCell?.[
      `${trial.layout}|${trial.visibilityFraction}`
    ];
  }
  return metadata.workload.scenarioSha256ByVisibility?.[String(trial.visibilityFraction)];
}

function renderParityDigestForTrial(metadata, trial) {
  return metadata.workload.renderParitySha256ByCell?.[
    `${trial.layout}|${trial.visibilityFraction}`
  ];
}

async function validateValidationArtifacts(
  validationArtifacts,
  metadata,
  planIndex,
  summariesByTrialId,
  workloadCatalog,
  firstInstanceRowsByTrial = null,
) {
  requireArray(validationArtifacts, 'validation-artifacts.json');
  if (validationArtifacts.length !== metadata.validationArtifactCount
    || validationArtifacts.length !== metadata.expectedTrialCount) {
    failVerification('validation-artifacts.json count is inconsistent with metadata trial counts.');
  }
  if (metadata.validationArtifactSha256.length !== validationArtifacts.length) {
    failVerification('metadata validationArtifactSha256 count is inconsistent.');
  }

  const seenTrialIds = new Set();
  const physicalBinSequencePairs = new Map();
  for (const [index, artifact] of validationArtifacts.entries()) {
    requireRecord(artifact, `validation-artifacts.json[${index}]`);
    if (artifact.schemaVersion !== 2) {
      failVerification(`validation-artifacts.json[${index}] schemaVersion must be 2.`);
    }
    const trialId = requireNonemptyString(
      artifact.trialId,
      `validation-artifacts.json[${index}].trialId`,
    );
    if (seenTrialIds.has(trialId)) failVerification('validation-artifacts.json has duplicate trial IDs.');
    seenTrialIds.add(trialId);
    const planned = planIndex.byTrialId.get(trialId);
    if (planned === undefined) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} is absent from the plan.`);
    }
    if (planned.planIndex !== index) {
      failVerification('validation-artifacts.json is not in exact plan-index order.');
    }
    requireMatchingIdentity(artifact, planned, `validation artifact ${JSON.stringify(trialId)}`);
    if (artifact.objectCount !== planned.objectCount || artifact.bucketCount !== planned.bucketCount) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} workload size differs from the plan.`);
    }
    if (artifact.status !== 'accepted') {
      failVerification(`validation artifact ${JSON.stringify(trialId)} status is not "accepted".`);
    }
    const rejectionReasons = requireArray(
      artifact.rejectionReasons,
      `validation artifact ${JSON.stringify(trialId)} rejectionReasons`,
    );
    if (rejectionReasons.length !== 0) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} has rejection reasons.`);
    }
    const selectedConfig = requireRecord(
      artifact.selectedConfig,
      `validation artifact ${JSON.stringify(trialId)} selectedConfig`,
    );
    if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX
      && (selectedConfig.strategyId !== planned.modeId
        || selectedConfig.objectCount !== planned.objectCount
        || selectedConfig.bucketCount !== planned.bucketCount
        || selectedConfig.visibilityFraction !== planned.visibilityFraction
        || selectedConfig.layout !== planned.layout)) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} selectedConfig differs from the planned depth-ordering cell.`);
    }
    if (metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX
      && (selectedConfig.strategyId !== planned.modeId
        || selectedConfig.objectCount !== planned.objectCount
        || selectedConfig.bucketCount !== planned.bucketCount
        || selectedConfig.visibilityFraction !== planned.visibilityFraction
        || selectedConfig.layout !== planned.layout
        || !orderedValuesMatch(selectedConfig.laneStorageOrder, planned.laneStorageOrder)
        || selectedConfig.superblockOrientationOffset
          !== planned.superblockOrientationOffset)) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} selectedConfig differs from the frozen plan.`);
    }
    if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
      && (selectedConfig.strategyId !== planned.modeId
        || selectedConfig.objectCount !== planned.objectCount
        || selectedConfig.bucketCount !== planned.bucketCount
        || selectedConfig.visibilityFraction !== planned.visibilityFraction
        || selectedConfig.layout !== planned.layout
        || !orderedValuesMatch(
          selectedConfig.laneCommandSegmentOrder,
          planned.laneCommandSegmentOrder,
        )
        || selectedConfig.superblockOrientationOffset
          !== planned.superblockOrientationOffset)) {
      failVerification(
        `validation artifact ${JSON.stringify(trialId)} selectedConfig differs from the first-instance plan.`,
      );
    }
    if (metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX
      && (selectedConfig.strategyId !== planned.modeId
        || selectedConfig.objectCount !== planned.objectCount
        || selectedConfig.bucketCount !== planned.bucketCount
        || selectedConfig.visibilityFraction !== planned.visibilityFraction
        || selectedConfig.layout !== planned.layout
        || !orderedValuesMatch(
          selectedConfig.lanePhysicalOrder,
          planned.lanePhysicalOrder,
        )
        || selectedConfig.superblockOrientationOffset
          !== planned.superblockOrientationOffset)) {
      failVerification(
        `validation artifact ${JSON.stringify(trialId)} selectedConfig differs from the live first-instance plan.`,
      );
    }
    if (metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
      const plannedScheduleSha256 = metadata.protocol.frozenCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)];
      if (!orderedValuesMatch(artifact.laneStorageOrder, planned.laneStorageOrder)
        || artifact.superblockOrientationOffset !== planned.superblockOrientationOffset
        || artifact.plannedScheduleSha256 !== plannedScheduleSha256) {
        failVerification(`validation artifact ${JSON.stringify(trialId)} changes its frozen plan commitments.`);
      }
    }
    if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX) {
      const plannedScheduleSha256 = metadata.protocol.firstInstanceCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)];
      if (!orderedValuesMatch(
        artifact.laneCommandSegmentOrder,
        planned.laneCommandSegmentOrder,
      )
        || artifact.superblockOrientationOffset !== planned.superblockOrientationOffset
        || artifact.plannedScheduleSha256 !== plannedScheduleSha256) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changes its first-instance plan commitments.`,
        );
      }
    }
    if (metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
      const plannedScheduleSha256 = metadata.protocol.firstInstanceLiveCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)];
      if (!orderedValuesMatch(artifact.lanePhysicalOrder, planned.lanePhysicalOrder)
        || artifact.superblockOrientationOffset !== planned.superblockOrientationOffset
        || artifact.plannedScheduleSha256 !== plannedScheduleSha256) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changes its live first-instance plan commitments.`,
        );
      }
    }
    requireSha256(artifact.sha256, `validation artifact ${JSON.stringify(trialId)} sha256`);
    if (sha256Json(bodyWithoutSha256(artifact)) !== artifact.sha256) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} record SHA-256 is inconsistent.`);
    }
    if (metadata.validationArtifactSha256[index] !== artifact.sha256) {
      failVerification(`metadata validation digest at index ${index} does not match its artifact.`);
    }
    const summary = summariesByTrialId.get(trialId);
    if (summary?.validation?.artifactSha256 !== artifact.sha256) {
      failVerification(`trial summary and validation artifact digests differ for ${JSON.stringify(trialId)}.`);
    }

    const captures = ['pre', 'timingStart', 'post'].map((name) => validateEvidenceCapture(
      artifact[name],
      `validation artifact ${JSON.stringify(trialId)} ${name}`,
    ));
    for (const field of [
      'scenarioSeed',
      'geometrySha256',
      'scenarioSha256',
      'semanticSha256',
    ]) {
      if (captures.some((capture) => capture[field] !== captures[0][field])) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changed ${field} across captures.`,
        );
      }
    }
    if (captures[0].scenarioSeed !== metadata.workload.scenarioSeed) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} scenario seed differs from metadata.`);
    }
    if (captures[0].geometrySha256 !== metadata.workload.geometryFixtureSha256) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} geometry digest differs from metadata.`);
    }
    const expectedScenarioSha256 = scenarioDigestForTrial(metadata, artifact);
    if (captures[0].scenarioSha256 !== expectedScenarioSha256) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} scenario digest differs from metadata.`);
    }
    const geometryManifest = workloadCatalog.geometries[captures[0].geometrySha256];
    const scenarioManifest = workloadCatalog.scenarios[captures[0].scenarioSha256];
    if (geometryManifest === undefined || scenarioManifest === undefined) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} links an unknown workload manifest.`);
    }
    if (metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
      const parityDigests = [];
      for (const captureName of ['pre', 'timingStart', 'post']) {
        const capture = artifact[captureName];
        const exactCheck = validateExactValidation(capture.validation.payload, {
          modeId: planned.modeId,
          objectCount: planned.objectCount,
          bucketCount: planned.bucketCount,
          expectedVisibleCount: scenarioManifest.expectedVisibleCount,
          expectedVisibleIdsCanonicalSha256:
            scenarioManifest.expectedVisibleIdsCanonicalSha256,
          geometryManifest,
          scenarioManifest,
          laneStorageOrder: planned.laneStorageOrder,
        });
        if (exactCheck.rejectionReasons.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} frozen payload failed: ${exactCheck.rejectionReasons.join('; ')}.`,
          );
        }
        if (exactCheck.semanticSha256 !== capture.validation.semanticSha256) {
          failVerification(`validation artifact ${JSON.stringify(trialId)} ${captureName} frozen semantic SHA-256 is inconsistent.`);
        }
        const parityReasons = validateFrozenCrossoverRenderParity(
          capture.renderParity,
          { spec: planned, geometryManifest, scenarioManifest },
        );
        if (parityReasons.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} frozen render parity failed: ${parityReasons.join('; ')}.`,
          );
        }
        if (capture.renderParity.snapshotValidation?.physicalBinSequenceSha256
          !== capture.validation.payload.physicalBinSequenceSha256) {
          failVerification(`validation artifact ${JSON.stringify(trialId)} ${captureName} changed its physical-bin sequence between parity and validation.`);
        }
        if (sha256Json(capture.renderParity.snapshotValidation)
          !== capture.validation.payloadSha256) {
          failVerification(`validation artifact ${JSON.stringify(trialId)} ${captureName} parity snapshot differs from its exact validation payload.`);
        }
        parityDigests.push(renderParityIdentity(capture.renderParity));
      }
      const expectedParitySha256 = renderParityDigestForTrial(metadata, artifact);
      if (parityDigests.some((digest) => digest !== expectedParitySha256)) {
        failVerification(`validation artifact ${JSON.stringify(trialId)} frozen render-parity digest differs from its layout cell.`);
      }
      const completionReasons = validateFrozenCrossoverCompletionInvariant(
        summary.completionInvariant,
        {
          objectCount: planned.objectCount,
          validation: artifact.timingStart.validation.payload,
        },
      );
      if (completionReasons.length !== 0) {
        failVerification(
          `trial summary ${JSON.stringify(trialId)} frozen completion invariant failed: ${completionReasons.join('; ')}.`,
        );
      }
      const pairKey = [
        artifact.repetitionIndex,
        artifact.layout,
        artifact.visibilityFraction,
        captures[0].scenarioSha256,
      ].join('|');
      const sequenceSha256 = artifact.pre.validation.payload.physicalBinSequenceSha256;
      if (physicalBinSequencePairs.has(pairKey)) {
        failVerification(`frozen physical-bin sequence pair ${JSON.stringify(pairKey)} is duplicated.`);
      }
      physicalBinSequencePairs.set(pairKey, { sha256: sequenceSha256, modeIds: null });
    }
    if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX) {
      const phaseSemanticDigests = [];
      const phaseParityDigests = [];
      const environment = metadata.environment.benchmarkPage;
      for (const captureName of ['pre', 'timingStart', 'post']) {
        const capture = artifact[captureName];
        const validationReasons = await validateFirstInstanceCrossoverValidation(
          capture.validation.payload,
          { spec: planned, environment },
        );
        if (validationReasons.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} first-instance payload failed: ${validationReasons.join('; ')}.`,
          );
        }
        const semanticSha256 = firstInstanceValidationSemanticSha256(
          capture.validation.payload,
        );
        if (capture.validation.semanticSha256 !== semanticSha256) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} first-instance semantic SHA-256 is inconsistent.`,
          );
        }
        const parityReasons = validateFirstInstanceCrossoverRenderParity(
          capture.renderParity,
          { spec: planned, validation: capture.validation.payload },
        );
        if (parityReasons.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} first-instance render parity failed: ${parityReasons.join('; ')}.`,
          );
        }
        const paritySha256 = firstInstanceRenderParityIdentity(capture.renderParity);
        if (capture.renderParitySemanticSha256 !== paritySha256
          || capture.renderParityOutputSha256 !== paritySha256) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} first-instance render-parity SHA-256 is inconsistent.`,
          );
        }
        if (capture.validation.payload?.membership?.expectedCount
            !== scenarioManifest.expectedVisibleCount
          || capture.validation.payload?.membershipDigests?.expected?.sha256
            !== scenarioManifest.expectedVisibleIdsCanonicalSha256
          || capture.validation.payload?.membershipDigests?.actual?.sha256
            !== scenarioManifest.expectedVisibleIdsCanonicalSha256) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} first-instance membership differs from the signed scenario manifest.`,
          );
        }
        phaseSemanticDigests.push(semanticSha256);
        phaseParityDigests.push(paritySha256);
      }
      if (phaseSemanticDigests.some((digest) => digest !== phaseSemanticDigests[0])) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changes first-instance validation semantics across phases.`,
        );
      }
      if (phaseParityDigests.some((digest) => digest !== phaseParityDigests[0])) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changes first-instance render output across phases.`,
        );
      }
      const expectedParitySha256 = renderParityDigestForTrial(metadata, artifact);
      if (phaseParityDigests[0] !== expectedParitySha256) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} first-instance render parity differs from its visibility cell.`,
        );
      }
      const rows = firstInstanceRowsByTrial?.get(trialId);
      if (!Array.isArray(rows)
        || rows.length !== FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} lacks its exact retained row set.`,
        );
      }
      const pageSummary = {
        accepted: summary.timestamps.accepted,
        timestampAvailable: summary.timestamps.available,
        rowCount: summary.timestamps.rowCount,
        missingRenderFrames: summary.timestamps.missingRenderFrames,
        invalidRenderTimestampUidCountFrames:
          summary.timestamps.invalidRenderTimestampUidCountFrames,
        expectedRenderTimestampUidCount:
          summary.timestamps.expectedRenderTimestampUidCount,
        missingComputeFrames: summary.timestamps.missingComputeFrames,
        classification: summary.timestamps.classification,
        quantumNs: summary.timestamps.quantumNs,
        completionInvariant: summary.completionInvariant,
      };
      const evidence = await validateFirstInstanceTrialEvidence({
        spec: planned,
        environment,
        validation: artifact.timingStart.validation.payload,
        renderParity: artifact.timingStart.renderParity,
        rows,
        summary: pageSummary,
        protocol: {
          schemaVersion: 2,
          warmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
          measuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
          plannedScheduleSha256: artifact.plannedScheduleSha256,
        },
      });
      const persisted = requireRecord(
        artifact.firstInstanceTrialEvidence,
        `validation artifact ${JSON.stringify(trialId)} firstInstanceTrialEvidence`,
      );
      const persistedKeys = Object.keys(persisted);
      if (persistedKeys.length !== 3
        || !['pass', 'rejectionReasons', 'semanticSha256'].every(
          (key) => Object.hasOwn(persisted, key),
        )
        || persisted.pass !== true
        || !Array.isArray(persisted.rejectionReasons)
        || persisted.rejectionReasons.length !== 0
        || persisted.semanticSha256 !== phaseSemanticDigests[0]
        || persisted.semanticSha256 !== summary.validation.firstInstanceSemanticSha256
        || JSON.stringify(persisted) !== JSON.stringify(evidence)
        || evidence.pass !== true
        || evidence.rejectionReasons.length !== 0) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} persisted first-instance trial evidence is absent or inconsistent.`,
        );
      }
    }
    if (metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
      const phaseSemanticDigests = [];
      const phaseParityDigests = [];
      const environment = metadata.environment.benchmarkPage;
      for (const captureName of ['pre', 'timingStart', 'post']) {
        const capture = artifact[captureName];
        const uncapturedErrors = requireArray(
          capture.webgpuUncapturedErrors,
          `validation artifact ${JSON.stringify(trialId)} ${captureName} webgpuUncapturedErrors`,
        );
        if (uncapturedErrors.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} contains uncaptured WebGPU errors.`,
          );
        }
        const validationReasons = await validateLiveFirstInstanceCrossoverValidation(
          capture.validation.payload,
          { spec: planned, environment, scenarioManifest, geometryManifest },
        );
        if (validationReasons.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} live first-instance payload failed: ${validationReasons.join('; ')}.`,
          );
        }
        const semanticSha256 = liveFirstInstanceValidationSemanticSha256(
          capture.validation.payload,
        );
        if (capture.validation.semanticSha256 !== semanticSha256) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} live first-instance semantic SHA-256 is inconsistent.`,
          );
        }
        const parityReasons = validateLiveFirstInstanceCrossoverRenderParity(
          capture.renderParity,
          {
            spec: planned,
            validation: capture.validation.payload,
            scenarioManifest,
          },
        );
        if (parityReasons.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} live first-instance render parity failed: ${parityReasons.join('; ')}.`,
          );
        }
        const paritySha256 = liveFirstInstanceRenderParityIdentity(capture.renderParity);
        if (capture.renderParitySemanticSha256 !== paritySha256
          || capture.renderParityOutputSha256 !== paritySha256) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} live first-instance render-parity SHA-256 is inconsistent.`,
          );
        }
        phaseSemanticDigests.push(semanticSha256);
        phaseParityDigests.push(paritySha256);
      }
      if (phaseSemanticDigests.some((digest) => digest !== phaseSemanticDigests[0])) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changes live first-instance validation semantics across phases.`,
        );
      }
      if (phaseParityDigests.some((digest) => digest !== phaseParityDigests[0])) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changes live first-instance render output across phases.`,
        );
      }
      const expectedParitySha256 = renderParityDigestForTrial(metadata, artifact);
      if (phaseParityDigests[0] !== expectedParitySha256) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} live first-instance render parity differs from its visibility cell.`,
        );
      }
      const rows = firstInstanceRowsByTrial?.get(trialId);
      if (!Array.isArray(rows)
        || rows.length !== FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} lacks its exact live retained row set.`,
        );
      }
      const pageSummary = {
        accepted: summary.timestamps.accepted,
        timestampAvailable: summary.timestamps.available,
        rowCount: summary.timestamps.rowCount,
        warmupRowCount: summary.timestamps.warmupRowCount,
        missingWarmupRenderFrames: summary.timestamps.missingWarmupRenderFrames,
        invalidWarmupRenderTimestampUidCountFrames:
          summary.timestamps.invalidWarmupRenderTimestampUidCountFrames,
        invalidWarmupRenderTimestampDurationFrames:
          summary.timestamps.invalidWarmupRenderTimestampDurationFrames,
        missingWarmupComputeFrames: summary.timestamps.missingWarmupComputeFrames,
        invalidWarmupComputeTimestampUidCountFrames:
          summary.timestamps.invalidWarmupComputeTimestampUidCountFrames,
        invalidWarmupComputeTimestampDurationFrames:
          summary.timestamps.invalidWarmupComputeTimestampDurationFrames,
        missingRenderFrames: summary.timestamps.missingRenderFrames,
        missingComputeFrames: summary.timestamps.missingComputeFrames,
        expectedComputeTimestampUidCount:
          summary.timestamps.expectedComputeTimestampUidCount,
        invalidComputeTimestampUidCountFrames:
          summary.timestamps.invalidComputeTimestampUidCountFrames,
        invalidComputeTimestampDurationFrames:
          summary.timestamps.invalidComputeTimestampDurationFrames,
        expectedRenderTimestampUidCount:
          summary.timestamps.expectedRenderTimestampUidCount,
        invalidRenderTimestampUidCountFrames:
          summary.timestamps.invalidRenderTimestampUidCountFrames,
        invalidRenderTimestampDurationFrames:
          summary.timestamps.invalidRenderTimestampDurationFrames,
        renderTimestampPoolQualityValid:
          summary.timestamps.renderTimestampPoolQualityValid,
        computeTimestampPoolQualityValid:
          summary.timestamps.computeTimestampPoolQualityValid,
        warmupRenderTimestampPoolQualityValid:
          summary.timestamps.warmupRenderTimestampPoolQualityValid,
        warmupComputeTimestampPoolQualityValid:
          summary.timestamps.warmupComputeTimestampPoolQualityValid,
        warmupTimestampFrameCountValid:
          summary.timestamps.warmupTimestampFrameCountValid,
        measurementTimestampFrameCountValid:
          summary.timestamps.measurementTimestampFrameCountValid,
        timestampResolutions: summary.timestamps.timestampResolutions,
        timestampPhases: summary.timestamps.timestampPhases,
        classification: summary.timestamps.classification,
        quantumNs: summary.timestamps.quantumNs,
        completionInvariant: summary.completionInvariant,
      };
      const evidence = await validateLiveFirstInstanceTrialEvidence({
        spec: planned,
        environment,
        preflightValidation: artifact.pre.validation.payload,
        preflightRenderParity: artifact.pre.renderParity,
        validation: artifact.timingStart.validation.payload,
        renderParity: artifact.timingStart.renderParity,
        postflightValidation: artifact.post.validation.payload,
        postflightRenderParity: artifact.post.renderParity,
        shaderObservationChallenges: artifact.shaderObservationChallenges,
        rows,
        summary: pageSummary,
        protocol: {
          schemaVersion: 2,
          warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
          measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
          plannedScheduleSha256: artifact.plannedScheduleSha256,
        },
        scenarioManifest,
        geometryManifest,
      });
      const persisted = requireRecord(
        artifact.liveFirstInstanceTrialEvidence,
        `validation artifact ${JSON.stringify(trialId)} liveFirstInstanceTrialEvidence`,
      );
      const persistedKeys = Object.keys(persisted);
      if (persistedKeys.length !== 4
        || !['pass', 'rejectionReasons', 'semanticSha256', 'historyBalance'].every(
          (key) => Object.hasOwn(persisted, key),
        )
        || persisted.pass !== true
        || !Array.isArray(persisted.rejectionReasons)
        || persisted.rejectionReasons.length !== 0
        || persisted.semanticSha256 !== phaseSemanticDigests[0]
        || persisted.semanticSha256 !== summary.validation.liveFirstInstanceSemanticSha256
        || JSON.stringify(persisted) !== JSON.stringify(evidence)
        || evidence.pass !== true
        || evidence.rejectionReasons.length !== 0) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} persisted live first-instance trial evidence is absent or inconsistent.`,
        );
      }
    }
    if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX) {
      for (const captureName of ['pre', 'timingStart', 'post']) {
        const capture = artifact[captureName];
        const exactCheck = validateExactValidation(capture.validation.payload, {
          modeId: planned.modeId,
          objectCount: planned.objectCount,
          bucketCount: planned.bucketCount,
          expectedVisibleCount: scenarioManifest.expectedVisibleCount,
          expectedVisibleIdsCanonicalSha256:
            scenarioManifest.expectedVisibleIdsCanonicalSha256,
          geometryManifest,
          scenarioManifest,
        });
        if (exactCheck.rejectionReasons.length !== 0) {
          failVerification(
            `validation artifact ${JSON.stringify(trialId)} ${captureName} exact payload failed: ${exactCheck.rejectionReasons.join('; ')}.`,
          );
        }
        if (exactCheck.semanticSha256 !== capture.validation.semanticSha256) {
          failVerification(`validation artifact ${JSON.stringify(trialId)} ${captureName} semantic SHA-256 is inconsistent.`);
        }
      }
      const completionReasons = validateDepthOrderingCompletionInvariant(
        summary.completionInvariant,
        {
          modeId: planned.modeId,
          objectCount: planned.objectCount,
          bucketCount: planned.bucketCount,
          validation: artifact.timingStart.validation.payload,
          scenarioManifest,
        },
      );
      if (completionReasons.length !== 0) {
        failVerification(
          `trial summary ${JSON.stringify(trialId)} completion invariant failed: ${completionReasons.join('; ')}.`,
        );
      }
      if (captures[0].renderParity === null) {
        failVerification(`validation artifact ${JSON.stringify(trialId)} lacks preflight render-parity evidence.`);
      }
      const expectedRenderParitySha256 = renderParityDigestForTrial(metadata, artifact);
      for (const [captureIndex, capture] of captures.entries()) {
        if (capture.renderParity === null) continue;
        const paritySha256 = validateRenderParityCapture(
          capture.renderParity,
          `validation artifact ${JSON.stringify(trialId)} render parity capture ${captureIndex}`,
          { planned, geometryManifest, scenarioManifest },
        );
        if (paritySha256 !== expectedRenderParitySha256) {
          failVerification(`validation artifact ${JSON.stringify(trialId)} render-parity digest differs from its layout cell.`);
        }
      }
      if (artifact.modeId === 'fixed-slice-depth-front-to-back'
        || artifact.modeId === 'fixed-slice-depth-reverse') {
        const preflightSequenceSha256 = physicalBinSequenceIdentity(
          artifact.pre?.validation?.payload,
        );
        const paritySequenceSha256 = physicalBinSequenceIdentity(
          artifact.pre?.renderParity?.snapshotValidation,
        );
        if (preflightSequenceSha256 === null || paritySequenceSha256 === null) {
          failVerification(`validation artifact ${JSON.stringify(trialId)} lacks a physical-bin sequence commitment.`);
        }
        if (preflightSequenceSha256 !== paritySequenceSha256) {
          failVerification(`validation artifact ${JSON.stringify(trialId)} changed its physical-bin sequence between parity and preflight snapshots.`);
        }
        const pairKey = [
          artifact.repetitionIndex,
          artifact.layout,
          artifact.visibilityFraction,
          captures[0].scenarioSha256,
        ].join('|');
        let pair = physicalBinSequencePairs.get(pairKey);
        if (pair === undefined) {
          pair = { sha256: preflightSequenceSha256, modeIds: new Set() };
          physicalBinSequencePairs.set(pairKey, pair);
        } else if (pair.sha256 !== preflightSequenceSha256) {
          failVerification(`ordered pair ${JSON.stringify(pairKey)} has unequal traversal-normalized physical-bin sequences.`);
        }
        if (pair.modeIds.has(artifact.modeId)) {
          failVerification(`ordered pair ${JSON.stringify(pairKey)} duplicates mode ${JSON.stringify(artifact.modeId)}.`);
        }
        pair.modeIds.add(artifact.modeId);
      }
    }
    if (artifact.modeId !== 'three-blocks-historical'
      && metadata.protocol.matrixKind !== FIRST_INSTANCE_CROSSOVER_MATRIX
      && metadata.protocol.matrixKind !== LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX
      && captures.some((capture) => capture.payloadSha256 !== captures[0].payloadSha256)) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} changed its exact payload.`);
    }
  }
  if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX) {
    const orderedModes = [
      'fixed-slice-depth-front-to-back',
      'fixed-slice-depth-reverse',
    ];
    const expectedPairCount = metadata.protocol.repetitions
      * metadata.protocol.layouts.length
      * metadata.protocol.visibilityLevels.length;
    if (physicalBinSequencePairs.size !== expectedPairCount
      || [...physicalBinSequencePairs.values()].some(
        (pair) => orderedModes.some((modeId) => !pair.modeIds.has(modeId))
          || pair.modeIds.size !== orderedModes.length,
      )) {
      failVerification('physical-bin sequence commitments do not cover every ordered-mode pair.');
    }
    for (const [key, pair] of physicalBinSequencePairs) {
      if (metadata.workload.physicalBinSequenceSha256ByPair?.[key] !== pair.sha256) {
        failVerification(`physical-bin sequence pair ${JSON.stringify(key)} differs from metadata.`);
      }
    }
  }
  if (metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
    const expectedPairCount = metadata.protocol.repetitions
      * metadata.protocol.layouts.length
      * metadata.protocol.visibilityLevels.length;
    if (physicalBinSequencePairs.size !== expectedPairCount) {
      failVerification('frozen physical-bin sequence commitments do not cover every trial.');
    }
    for (const [key, pair] of physicalBinSequencePairs) {
      if (metadata.workload.physicalBinSequenceSha256ByPair?.[key] !== pair.sha256) {
        failVerification(`frozen physical-bin sequence pair ${JSON.stringify(key)} differs from metadata.`);
      }
    }
  }
}

function exactCsvInteger(value, label) {
  if (typeof value !== 'string' || value.trim() === '') failVerification(`${label} is missing.`);
  const number = Number(value);
  if (!Number.isInteger(number)) failVerification(`${label} must be an integer.`);
  return number;
}

function exactCsvNumber(value, label) {
  if (typeof value !== 'string' || value.trim() === '') failVerification(`${label} is missing.`);
  const number = Number(value);
  if (!Number.isFinite(number)) failVerification(`${label} must be finite.`);
  return number;
}

function exactCsvNullableInteger(value, label) {
  if (typeof value !== 'string') failVerification(`${label} is missing.`);
  if (value.trim() === '') return null;
  return exactCsvInteger(value, label);
}

function depthFrameShape(modeId, bucketCount) {
  const depthBinned = modeId === 'fixed-slice-depth-front-to-back'
    || modeId === 'fixed-slice-depth-reverse';
  return {
    validationKind: depthBinned
      ? `${modeId}-exact-membership-and-depth-order`
      : 'fixed-slice-exact-membership',
    configuredDrawCommands: bucketCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: depthBinned ? 4 : 2,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    bundleRecordCallbackCountAtTimingStart: 1,
  };
}

function validateVerifiedFrozenFrames(
  parsed,
  metadata,
  planIndex,
  summariesByTrialId,
  workloadCatalog,
) {
  let rows;
  try {
    rows = parseFrozenCrossoverRecords(parsed);
  } catch (error) {
    failVerification(error instanceof Error ? error.message : String(error));
  }
  if (rows.length !== metadata.frameRowCount
    || metadata.frameRowCount !== metadata.expectedTrialCount * metadata.protocol.measuredFrames) {
    failVerification('frozen frames.csv row count is inconsistent with metadata.');
  }
  const rowCountByTrial = new Map();
  for (const [index, row] of rows.entries()) {
    const record = parsed.records[index];
    const label = `frames.csv record ${index + 2}`;
    const planned = planIndex.byTrialId.get(row.trialId);
    const summary = summariesByTrialId.get(row.trialId);
    if (row.runId !== metadata.runId
      || planned === undefined
      || summary === undefined) {
      failVerification(`${label} has an unknown run or trial identity.`);
    }
    if (row.planIndex !== planned.planIndex
      || row.repetitionIndex !== planned.repetitionIndex
      || row.layout !== planned.layout
      || row.layoutOrderPosition !== planned.layoutOrderPosition
      || row.laneStorageOrder !== planned.laneStorageOrder.join('|')
      || row.superblockOrientationOffset !== planned.superblockOrientationOffset
      || row.plannedScheduleSha256
        !== metadata.protocol.frozenCrossover.scheduleSha256ByOrientation[
          String(planned.superblockOrientationOffset)
        ]) {
      failVerification(`${label} differs from its frozen plan or schedule commitment.`);
    }
    const completion = summary.completionInvariant;
    if (row.selectorWriteSerialAtTimingStart
        !== completion.selectorWriteSerialAtTimingStart
      || row.renderCallSerialAtTimingStart !== completion.renderCallSerialAtTimingStart
      || row.computeCallSerialAtTimingStart !== completion.computeCallSerialAtTimingStart
      || row.totalPipelineCacheEntriesAtTimingStart
        !== completion.totalPipelineCacheEntriesAtTimingStart
      || row.computePipelineCacheEntriesAtTimingStart
        !== completion.computePipelineCacheEntriesAtTimingStart
      || FROZEN_CROSSOVER_ROW_IDENTITY_FIELDS.some(
        (field) => row[field] !== completion[field],
      )) {
      failVerification(`${label} differs from its frozen timing-start lifecycle commitment.`);
    }
    const scenarioSha256 = scenarioDigestForTrial(metadata, planned);
    const scenarioManifest = workloadCatalog.scenarios[scenarioSha256];
    if (!scenarioManifest
      || exactCsvNumber(record.depthBinRangeNear, `${label} depthBinRangeNear`)
        !== scenarioManifest.depthBinRange.near
      || exactCsvNumber(record.depthBinRangeFar, `${label} depthBinRangeFar`)
        !== scenarioManifest.depthBinRange.far
      || exactCsvInteger(record.expectedVisibleCount, `${label} expectedVisibleCount`)
        !== scenarioManifest.expectedVisibleCount) {
      failVerification(`${label} differs from its frozen scenario manifest.`);
    }
    rowCountByTrial.set(row.trialId, (rowCountByTrial.get(row.trialId) ?? 0) + 1);
  }
  if (rowCountByTrial.size !== planIndex.byTrialId.size) {
    failVerification('frozen frames.csv does not cover every planned trial.');
  }
  for (const trialId of planIndex.byTrialId.keys()) {
    if (rowCountByTrial.get(trialId) !== FROZEN_CROSSOVER_MEASURED_FRAMES) {
      failVerification(`frozen frames.csv has an incomplete trial ${JSON.stringify(trialId)}.`);
    }
  }
}

function liveTimestampPoolStaticCommitment(timestampPools) {
  return Object.fromEntries(['render', 'compute'].map((type) => {
    const pool = timestampPools?.[type];
    return [type, pool ? {
      poolIdentity: pool.poolIdentity,
      querySetIdentity: pool.querySetIdentity,
      resolveBufferIdentity: pool.resolveBufferIdentity,
      resultBufferIdentity: pool.resultBufferIdentity,
      maxQueries: pool.maxQueries,
      isDisposed: pool.isDisposed,
    } : null];
  }));
}

function validateVerifiedLiveFirstInstanceFrames(
  parsed,
  metadata,
  planIndex,
  summariesByTrialId,
  workloadCatalog,
) {
  let rows;
  try {
    rows = parseLiveFirstInstanceCrossoverRecords(parsed);
  } catch (error) {
    failVerification(error instanceof Error ? error.message : String(error));
  }
  const expectedTrialCount = FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS
    * FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.length;
  const expectedRows = expectedTrialCount
    * FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES;
  if (metadata.expectedTrialCount !== expectedTrialCount
    || metadata.completedTrialCount !== expectedTrialCount
    || metadata.acceptedTrialCount !== expectedTrialCount
    || metadata.validationArtifactCount !== expectedTrialCount
    || metadata.frameRowCount !== expectedRows
    || rows.length !== expectedRows) {
    failVerification(
      'first-instance-live run must contain exactly 24 accepted trials and 11,520 retained rows.',
    );
  }
  let analysis;
  try {
    analysis = summarizeLiveFirstInstanceCrossoverRows(rows);
  } catch (error) {
    failVerification(error instanceof Error ? error.message : String(error));
  }
  const audit = requireRecord(
    metadata.liveFirstInstanceAnalysisAudit,
    'metadata.json liveFirstInstanceAnalysisAudit',
  );
  const expectedAudit = {
    schemaVersion: analysis.schemaVersion,
    kind: analysis.kind,
    nTrials: analysis.nTrials,
    nRows: analysis.nRows,
    preregisteredDecision: analysis.preregisteredDecision,
    preregisteredNumericalDecision: analysis.preregisteredNumericalDecision,
    sha256: sha256Json(analysis),
  };
  if (JSON.stringify(audit) !== JSON.stringify(expectedAudit)) {
    failVerification(
      'metadata.json liveFirstInstanceAnalysisAudit does not exactly match the reconstructed analysis.',
    );
  }

  const rowsByTrial = new Map();
  for (const [index, row] of rows.entries()) {
    const label = `frames.csv record ${index + 2}`;
    const planned = planIndex.byTrialId.get(row.trialId);
    const summary = summariesByTrialId.get(row.trialId);
    if (row.runId !== metadata.runId || planned === undefined || summary === undefined) {
      failVerification(`${label} has an unknown run or trial identity.`);
    }
    const scenarioSha256 = scenarioDigestForTrial(metadata, planned);
    const scenario = workloadCatalog.scenarios[scenarioSha256];
    if (scenario === undefined
      || row.expectedVisibleCount !== scenario.expectedVisibleCount) {
      failVerification(`${label} differs from its signed live first-instance scenario.`);
    }
    if (row.planIndex !== planned.planIndex
      || row.repetitionIndex !== planned.repetitionIndex
      || row.modeId !== planned.modeId
      || row.modeOrderPosition !== planned.modeOrderPosition
      || row.visibilityOrderPosition !== planned.visibilityOrderPosition
      || row.layoutOrderPosition !== planned.layoutOrderPosition
      || row.targetVisibilityFraction !== planned.visibilityFraction
      || row.scenarioLayout !== planned.layout
      || row.plannedModeOrder !== planned.modeOrder.join('|')
      || row.plannedVisibilityOrder !== planned.visibilityOrder.join('|')
      || row.plannedLayoutOrder !== planned.layoutOrder.join('|')
      || row.plannedLanePhysicalOrder !== planned.lanePhysicalOrder.join('|')
      || row.lanePhysicalOrder !== planned.lanePhysicalOrder.join('|')
      || row.superblockOrientationOffset !== planned.superblockOrientationOffset
      || row.plannedScheduleSha256 !== metadata.protocol.firstInstanceLiveCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)]) {
      failVerification(`${label} differs from its live first-instance plan commitment.`);
    }
    const completion = summary.completionInvariant;
    const commandCommitments = completion.commandBufferCommitments;
    if (row.lifecycleCommitmentAtTimingStart
        !== completion.lifecycleCommitmentAtTimingStart
      || row.commandBufferCommitmentsAtTimingStart
        !== JSON.stringify(completion.commandBufferCommitments)
      || row.portableCommandBufferIdAtTimingStart
        !== commandCommitments?.[FIRST_INSTANCE_LIVE_CROSSOVER_LANES[0]]?.attributeId
      || row.featureCommandBufferIdAtTimingStart
        !== commandCommitments?.[FIRST_INSTANCE_LIVE_CROSSOVER_LANES[1]]?.attributeId
      || row.selectorWriteSerialAtTimingStart
        !== completion.selectorWriteSerialAtTimingStart
      || row.strategySelectionSerialAtTimingStart
        !== completion.strategySelectionSerialAtTimingStart
      || row.strategyComputeCallSerialAtTimingStart
        !== completion.strategyComputeCallSerialAtTimingStart
      || row.strategyPrepareSerialAtTimingStart
        !== completion.strategyPrepareSerialAtTimingStart
      || row.computeCallSerialAtTimingStart !== completion.computeCallSerialAtTimingStart
      || row.renderCallSerialAtTimingStart !== completion.renderCallSerialAtTimingStart
      || row.renderTargetTextureUuidAtTimingStart
        !== completion.renderTargetTextureUuidAtTimingStart
      || row.renderTargetWidthAtTimingStart !== completion.renderTargetWidthAtTimingStart
      || row.renderTargetHeightAtTimingStart !== completion.renderTargetHeightAtTimingStart
      || row.renderTargetSamplesAtTimingStart !== completion.renderTargetSamplesAtTimingStart
      || row.renderTargetDepthBufferAtTimingStart
        !== completion.renderTargetDepthBufferAtTimingStart
      || row.cameraViewFnv64AtTimingStart !== completion.cameraViewFnv64AtTimingStart
      || row.cameraProjectionFnv64AtTimingStart
        !== completion.cameraProjectionFnv64AtTimingStart
      || row.totalPipelineCacheEntriesAtTimingStart
        !== completion.totalPipelineCacheEntriesAtTimingStart
      || row.computePipelineCacheEntriesAtTimingStart
        !== completion.computePipelineCacheEntriesAtTimingStart
      || row.webgpuUncapturedErrorCountAtTimingStart
        !== completion.webgpuUncapturedErrorCountAtTimingStart
      || row.webgpuUncapturedErrorCountAtTimingStart !== 0
      || row.timestampPoolStaticCommitmentAtTimingStart !== JSON.stringify(
        liveTimestampPoolStaticCommitment(completion.timestampPoolsAtTimingStart),
      )) {
      failVerification(
        `${label} differs from its live first-instance timing-start lifecycle commitment.`,
      );
    }
    const serialOffset = FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES
      + row.frameIndex + 1;
    if (row.strategyComputeCallSerial
      !== completion.strategyComputeCallSerialAtTimingStart + serialOffset) {
      failVerification(`${label} changes the live strategy-compute serial stream.`);
    }
    if (row.commandSegmentIndex !== 0
      || row.commandRecordBase !== 0
      || row.commandByteBase !== 0
      || row.commandByteOffset !== 0
      || row.commandBufferRecordCount !== LIVE_FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT
      || row.commandBufferByteLength !== LIVE_FIRST_INSTANCE_CROSSOVER_BUCKET_COUNT * 20) {
      failVerification(`${label} changes the zero-offset live command-buffer shape.`);
    }
    let trialRows = rowsByTrial.get(row.trialId);
    if (trialRows === undefined) {
      trialRows = [];
      rowsByTrial.set(row.trialId, trialRows);
    }
    trialRows.push(row);
  }
  if (rowsByTrial.size !== expectedTrialCount
    || [...planIndex.byTrialId.keys()].some(
      (trialId) => rowsByTrial.get(trialId)?.length
        !== FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    )) {
    failVerification(
      'first-instance-live frames.csv does not contain exactly 480 rows for every planned trial.',
    );
  }
  for (const [trialId, trialRows] of rowsByTrial) {
    const timing = summariesByTrialId.get(trialId).timing;
    for (const [rowField, [p50Field, p95Field]] of Object.entries(
      LIVE_FIRST_INSTANCE_CROSSOVER_TIMING_FIELDS,
    )) {
      const values = trialRows.map((row) => row[rowField]);
      const expectedP50 = nearestRank(values, 0.5);
      const expectedP95 = nearestRank(values, 0.95);
      if (timing[p50Field] !== expectedP50 || timing[p95Field] !== expectedP95) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} timing ${p50Field}/${p95Field} differs from its retained rows.`,
        );
      }
    }
  }
  return { rows, rowsByTrial, analysis };
}

function validateVerifiedFirstInstanceFrames(
  parsed,
  metadata,
  planIndex,
  summariesByTrialId,
  workloadCatalog,
) {
  let rows;
  try {
    rows = parseFirstInstanceCrossoverRecords(parsed);
  } catch (error) {
    failVerification(error instanceof Error ? error.message : String(error));
  }
  const expectedRows = FIRST_INSTANCE_CROSSOVER_REPETITIONS
    * FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.length
    * FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES;
  if (metadata.expectedTrialCount
      !== FIRST_INSTANCE_CROSSOVER_REPETITIONS
        * FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.length
    || metadata.completedTrialCount !== metadata.expectedTrialCount
    || metadata.acceptedTrialCount !== metadata.expectedTrialCount
    || metadata.validationArtifactCount !== metadata.expectedTrialCount
    || metadata.frameRowCount !== expectedRows
    || rows.length !== expectedRows) {
    failVerification(
      'first-instance run must contain exactly 24 accepted trials and 11,520 retained rows.',
    );
  }
  let analysis;
  try {
    analysis = summarizeFirstInstanceCrossoverRows(rows);
  } catch (error) {
    failVerification(error instanceof Error ? error.message : String(error));
  }
  const audit = requireRecord(
    metadata.firstInstanceAnalysisAudit,
    'metadata.json firstInstanceAnalysisAudit',
  );
  const expectedAudit = {
    schemaVersion: analysis.schemaVersion,
    kind: analysis.kind,
    deltaConvention: analysis.deltaConvention,
    nTrials: analysis.nTrials,
    nRows: analysis.nRows,
    sha256: sha256Json(analysis),
  };
  if (Object.keys(audit).length !== Object.keys(expectedAudit).length
    || Object.entries(expectedAudit).some(([key, value]) => audit[key] !== value)) {
    failVerification(
      'metadata.json firstInstanceAnalysisAudit does not exactly match the reconstructed analysis.',
    );
  }
  const rowsByTrial = new Map();
  for (const [index, row] of rows.entries()) {
    const label = `frames.csv record ${index + 2}`;
    const planned = planIndex.byTrialId.get(row.trialId);
    const summary = summariesByTrialId.get(row.trialId);
    if (row.runId !== metadata.runId || planned === undefined || summary === undefined) {
      failVerification(`${label} has an unknown run or trial identity.`);
    }
    const scenarioSha256 = scenarioDigestForTrial(metadata, planned);
    const scenario = workloadCatalog.scenarios[scenarioSha256];
    if (scenario === undefined
      || row.expectedVisibleCount !== scenario.expectedVisibleCount) {
      failVerification(`${label} differs from its signed first-instance scenario.`);
    }
    if (row.planIndex !== planned.planIndex
      || row.repetitionIndex !== planned.repetitionIndex
      || row.modeId !== planned.modeId
      || row.modeOrderPosition !== planned.modeOrderPosition
      || row.visibilityOrderPosition !== planned.visibilityOrderPosition
      || row.layoutOrderPosition !== planned.layoutOrderPosition
      || row.targetVisibilityFraction !== planned.visibilityFraction
      || row.scenarioLayout !== planned.layout
      || row.plannedModeOrder !== planned.modeOrder.join('|')
      || row.plannedVisibilityOrder !== planned.visibilityOrder.join('|')
      || row.plannedLayoutOrder !== planned.layoutOrder.join('|')
      || row.plannedLaneCommandSegmentOrder
        !== planned.laneCommandSegmentOrder.join('|')
      || row.superblockOrientationOffset !== planned.superblockOrientationOffset
      || row.plannedScheduleSha256 !== metadata.protocol.firstInstanceCrossover
        .scheduleSha256ByOrientation[String(planned.superblockOrientationOffset)]) {
      failVerification(`${label} differs from its first-instance plan commitment.`);
    }
    let trialRows = rowsByTrial.get(row.trialId);
    if (trialRows === undefined) {
      trialRows = [];
      rowsByTrial.set(row.trialId, trialRows);
    }
    trialRows.push(row);
  }
  if (rowsByTrial.size !== metadata.expectedTrialCount
    || [...planIndex.byTrialId.keys()].some(
      (trialId) => rowsByTrial.get(trialId)?.length
        !== FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
    )) {
    failVerification(
      'first-instance frames.csv does not contain exactly 480 rows for every planned trial.',
    );
  }
  for (const [trialId, trialRows] of rowsByTrial) {
    const timing = summariesByTrialId.get(trialId).timing;
    for (const [rowField, [p50Field, p95Field]] of Object.entries(
      FIRST_INSTANCE_CROSSOVER_TIMING_FIELDS,
    )) {
      const values = trialRows.map((row) => row[rowField]);
      const expectedP50 = nearestRank(values, 0.5);
      const expectedP95 = nearestRank(values, 0.95);
      if (timing[p50Field] !== expectedP50 || timing[p95Field] !== expectedP95) {
        failVerification(
          `trial-summaries.json trial ${JSON.stringify(trialId)} timing ${p50Field}/${p95Field} differs from its retained rows.`,
        );
      }
    }
  }
  return { rows, rowsByTrial, analysis };
}

function validateVerifiedFrames(
  parsed,
  metadata,
  planIndex,
  summariesByTrialId,
  workloadCatalog,
) {
  if (metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    return validateVerifiedLiveFirstInstanceFrames(
      parsed,
      metadata,
      planIndex,
      summariesByTrialId,
      workloadCatalog,
    );
  }
  if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX) {
    return validateVerifiedFirstInstanceFrames(
      parsed,
      metadata,
      planIndex,
      summariesByTrialId,
      workloadCatalog,
    );
  }
  if (metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
    validateVerifiedFrozenFrames(
      parsed,
      metadata,
      planIndex,
      summariesByTrialId,
      workloadCatalog,
    );
    return null;
  }
  const requiredAuditColumns = [
    'runId',
    'trialId',
    'planIndex',
    'repetitionIndex',
    'frameIndex',
    'objectCount',
    'bucketCount',
    'validationPass',
    'timestampAvailable',
  ];
  if (metadata.protocol.matrixKind === FIXED_SLICE_REPRESENTATION_MATRIX
    || metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX) {
    requiredAuditColumns.push(
      'modeOrderPosition',
      'visibilityOrderPosition',
      'plannedModeOrder',
      'plannedVisibilityOrder',
    );
  }
  if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX) {
    requiredAuditColumns.push(
      'scenarioLayout',
      'layoutOrderPosition',
      'plannedLayoutOrder',
      'protocolWarmupFrames',
      'protocolMeasuredFrames',
      'depthBinRangeNear',
      'depthBinRangeFar',
      'expectedVisibleCount',
      'validationKind',
      'usesCompute',
      'configuredDrawCommands',
      'configuredRenderObjects',
      'configuredComputeDispatches',
      'configuredComputeSubmissions',
      'configuredSubmittedInstances',
      'bundleRecordCallbackCountAtTimingStart',
    );
  }
  const headers = new Set(parsed.headers);
  const missing = requiredAuditColumns.filter((column) => !headers.has(column));
  if (missing.length > 0) {
    failVerification(`frames.csv lacks audit columns: ${missing.join(', ')}.`);
  }
  if (parsed.records.length !== metadata.frameRowCount) {
    failVerification('frames.csv row count does not equal metadata frameRowCount.');
  }
  const expectedFrameRows = metadata.expectedTrialCount * metadata.protocol.measuredFrames;
  if (metadata.frameRowCount !== expectedFrameRows) {
    failVerification('metadata frameRowCount does not equal trials multiplied by measured frames.');
  }

  const frameIndexesByTrial = new Map();
  for (const [index, record] of parsed.records.entries()) {
    const label = `frames.csv record ${index + 2}`;
    if (record.runId !== metadata.runId) failVerification(`${label} has the wrong runId.`);
    const planned = planIndex.byTrialId.get(record.trialId);
    if (planned === undefined || !summariesByTrialId.has(record.trialId)) {
      failVerification(`${label} has an unknown trialId.`);
    }
    if (exactCsvInteger(record.planIndex, `${label} planIndex`) !== planned.planIndex
      || exactCsvInteger(record.repetitionIndex, `${label} repetitionIndex`) !== planned.repetitionIndex
      || record.modeId !== planned.modeId
      || exactCsvNumber(record.targetVisibilityFraction, `${label} targetVisibilityFraction`)
        !== planned.visibilityFraction
      || exactCsvInteger(record.objectCount, `${label} objectCount`) !== metadata.protocol.objectCount
      || exactCsvInteger(record.bucketCount, `${label} bucketCount`) !== metadata.protocol.bucketCount) {
      failVerification(`${label} does not match its planned trial.`);
    }
    if ((metadata.protocol.matrixKind === FIXED_SLICE_REPRESENTATION_MATRIX
      || metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX)
      && (exactCsvInteger(record.modeOrderPosition, `${label} modeOrderPosition`)
          !== planned.modeOrderPosition
        || exactCsvInteger(record.visibilityOrderPosition, `${label} visibilityOrderPosition`)
          !== planned.visibilityOrderPosition
        || record.plannedModeOrder !== planned.modeOrder.join('|')
        || record.plannedVisibilityOrder !== planned.visibilityOrder.join('|'))) {
      failVerification(`${label} order audit fields do not match its planned trial.`);
    }
    if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX
      && (record.scenarioLayout !== planned.layout
        || exactCsvInteger(record.layoutOrderPosition, `${label} layoutOrderPosition`)
          !== planned.layoutOrderPosition
        || record.plannedLayoutOrder !== planned.layoutOrder.join('|'))) {
      failVerification(`${label} layout audit fields do not match its planned trial.`);
    }
    if (metadata.protocol.matrixKind === DEPTH_ORDERING_MATRIX) {
      const scenarioSha256 = scenarioDigestForTrial(metadata, planned);
      const scenarioManifest = workloadCatalog.scenarios[scenarioSha256];
      const shape = depthFrameShape(planned.modeId, planned.bucketCount);
      if (!scenarioManifest) {
        failVerification(`${label} cannot resolve its scenario manifest.`);
      }
      if (exactCsvInteger(record.protocolWarmupFrames, `${label} protocolWarmupFrames`)
          !== metadata.protocol.warmupFrames
        || exactCsvInteger(record.protocolMeasuredFrames, `${label} protocolMeasuredFrames`)
          !== metadata.protocol.measuredFrames
        || exactCsvNumber(record.depthBinRangeNear, `${label} depthBinRangeNear`)
          !== scenarioManifest.depthBinRange.near
        || exactCsvNumber(record.depthBinRangeFar, `${label} depthBinRangeFar`)
          !== scenarioManifest.depthBinRange.far
        || exactCsvInteger(record.expectedVisibleCount, `${label} expectedVisibleCount`)
          !== scenarioManifest.expectedVisibleCount
        || record.validationKind !== shape.validationKind
        || record.usesCompute !== 'true'
        || exactCsvInteger(record.configuredDrawCommands, `${label} configuredDrawCommands`)
          !== shape.configuredDrawCommands
        || exactCsvInteger(record.configuredRenderObjects, `${label} configuredRenderObjects`)
          !== shape.configuredRenderObjects
        || exactCsvInteger(
          record.configuredComputeDispatches,
          `${label} configuredComputeDispatches`,
        ) !== shape.configuredComputeDispatches
        || exactCsvInteger(
          record.configuredComputeSubmissions,
          `${label} configuredComputeSubmissions`,
        ) !== shape.configuredComputeSubmissions
        || exactCsvNullableInteger(
          record.configuredSubmittedInstances,
          `${label} configuredSubmittedInstances`,
        ) !== shape.configuredSubmittedInstances
        || exactCsvNullableInteger(
          record.bundleRecordCallbackCountAtTimingStart,
          `${label} bundleRecordCallbackCountAtTimingStart`,
        ) !== shape.bundleRecordCallbackCountAtTimingStart) {
        failVerification(`${label} depth protocol audit fields do not match its planned trial.`);
      }
    }
    if (record.validationPass !== 'true' || record.timestampAvailable !== 'true') {
      failVerification(`${label} lacks accepted validation or GPU timestamps.`);
    }
    const frameIndex = exactCsvInteger(record.frameIndex, `${label} frameIndex`);
    let indexes = frameIndexesByTrial.get(record.trialId);
    if (indexes === undefined) {
      indexes = new Set();
      frameIndexesByTrial.set(record.trialId, indexes);
    }
    if (indexes.has(frameIndex)) failVerification(`${label} duplicates a frame index.`);
    indexes.add(frameIndex);
  }
  for (const trialId of planIndex.byTrialId.keys()) {
    const indexes = frameIndexesByTrial.get(trialId);
    if (indexes?.size !== metadata.protocol.measuredFrames) {
      failVerification(`frames.csv has an incomplete row set for trial ${JSON.stringify(trialId)}.`);
    }
    for (let index = 0; index < metadata.protocol.measuredFrames; index += 1) {
      if (!indexes.has(index)) {
        failVerification(`frames.csv trial ${JSON.stringify(trialId)} has non-contiguous frame indexes.`);
      }
    }
  }
  return null;
}

export async function verifyRunDirectory(
  runDirectory,
  { repositoryRoot = PROJECT_ROOT } = {},
) {
  const absoluteDirectory = path.resolve(runDirectory);
  const candidateRepositoryRoot = path.resolve(repositoryRoot);
  const inputStat = await lstat(absoluteDirectory);
  if (!inputStat.isDirectory() || inputStat.isSymbolicLink()) {
    failVerification('run-directory input must be a non-symbolic-link directory.');
  }
  const {
    manifest,
    contentsByName,
    requiredNames,
    bundleIntegrity,
  } = await loadVerifiedArtifactContents(absoluteDirectory, candidateRepositoryRoot);
  const metadata = parseJsonArtifact(contentsByName.get('metadata.json'), 'metadata.json');
  const trialSummaries = parseJsonArtifact(
    contentsByName.get('trial-summaries.json'),
    'trial-summaries.json',
  );
  const validationArtifacts = parseJsonArtifact(
    contentsByName.get('validation-artifacts.json'),
    'validation-artifacts.json',
  );
  const workloadManifests = parseJsonArtifact(
    contentsByName.get('workload-manifests.json'),
    'workload-manifests.json',
  );
  const telemetrySummary = requireRecord(
    parseJsonArtifact(
      contentsByName.get('gpu-telemetry-summary.json'),
      'gpu-telemetry-summary.json',
    ),
    'gpu-telemetry-summary.json',
  );
  const { plan } = requireMetadataCompleteness(metadata, manifest);
  await validateLiveCandidateViteRuntime(metadata, candidateRepositoryRoot);
  validatePublicBundleCandidateBinding(bundleIntegrity, metadata);
  const expectedRequiredNames = [...(
    metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX
      ? LIVE_FIRST_INSTANCE_REQUIRED_RUN_ARTIFACTS
      : REQUIRED_RUN_ARTIFACTS
  )].sort();
  if (requiredNames.length !== expectedRequiredNames.length
    || requiredNames.some((name, index) => name !== expectedRequiredNames[index])) {
    failVerification(
      'artifact-manifest.json requiredFiles do not match metadata protocol.matrixKind.',
    );
  }
  if (JSON.stringify(metadata.environment.gpuTelemetry) !== JSON.stringify(telemetrySummary)) {
    failVerification('metadata GPU telemetry summary differs from gpu-telemetry-summary.json.');
  }
  if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX
    || metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    validateNvidiaTelemetryReport(
      telemetrySummary,
      metadata,
      manifest,
      contentsByName,
    );
  }
  if (metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    validateLiveFirstInstanceEnvironmentEvidence(
      telemetrySummary,
      metadata,
      contentsByName,
    );
  }
  const planIndex = validateBenchmarkPlan(plan, metadata);
  const summariesByTrialId = validateTrialSummaries(
    trialSummaries,
    metadata,
    planIndex,
  );
  const workloadCatalog = validateWorkloadManifests(workloadManifests, metadata);
  const csvText = contentsByName.get('frames.csv').toString('utf8');
  const parsed = parseCsv(csvText);
  if (metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX) {
    try {
      parseLiveFirstInstanceCrossoverRecords(parsed);
    } catch (error) {
      failVerification(error instanceof Error ? error.message : String(error));
    }
  } else if (metadata.protocol.matrixKind === FIRST_INSTANCE_CROSSOVER_MATRIX) {
    try {
      parseFirstInstanceCrossoverRecords(parsed);
    } catch (error) {
      failVerification(error instanceof Error ? error.message : String(error));
    }
  } else if (metadata.protocol.matrixKind === FROZEN_DEPTH_CROSSOVER_MATRIX) {
    try {
      parseFrozenCrossoverRecords(parsed);
    } catch (error) {
      failVerification(error instanceof Error ? error.message : String(error));
    }
  } else {
    parseFrameRecords(parsed);
  }
  const verifiedFrames = validateVerifiedFrames(
    parsed,
    metadata,
    planIndex,
    summariesByTrialId,
    workloadCatalog,
  );
  await validateValidationArtifacts(
    validationArtifacts,
    metadata,
    planIndex,
    summariesByTrialId,
    workloadCatalog,
    verifiedFrames?.rowsByTrial ?? null,
  );

  return {
    csvText,
    bundleIntegrity,
    liveFirstInstanceEvidenceDecision:
      metadata.protocol.matrixKind === LIVE_FIRST_INSTANCE_CROSSOVER_MATRIX
        ? {
          adapterTelemetryAssociation:
            metadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation,
          candidateEnvironmentGate:
            metadata.liveFirstInstanceEnvironmentAudit.candidateEnvironmentGate,
          overallEvidenceDecision:
            metadata.liveFirstInstanceEnvironmentAudit.overallEvidenceDecision,
        }
        : null,
    artifactVerification: {
      status: 'consistent',
      scope: 'artifact-integrity-and-schema-only',
      authenticityVerified: false,
      inputKind: 'run-directory',
      runId: metadata.runId,
      evidenceStatus: metadata.evidenceStatus,
      manifestSchemaVersion: manifest.schemaVersion,
      verifiedArtifactCount: [...contentsByName.keys()].length,
      requiredArtifactCount: expectedRequiredNames.length,
      completedTrialCount: metadata.completedTrialCount,
      acceptedTrialCount: metadata.acceptedTrialCount,
      sourceProvenanceStable: metadata.sourceProvenance.stable,
    },
  };
}

export function summarizeCsv(text) {
  const parsed = parseCsv(text);
  if (isLiveFirstInstanceCrossoverCsv(parsed)) {
    return {
      ...summarizeLiveFirstInstanceCrossoverRows(
        parseLiveFirstInstanceCrossoverRecords(parsed),
      ),
      artifactVerification: {
        status: 'unverified',
        scope: 'artifact-integrity-and-schema-only',
        authenticityVerified: false,
        inputKind: 'raw-csv-content',
        evidenceStatus: null,
        reason: 'Raw CSV is not bound to a consistent run artifact manifest.',
      },
    };
  }
  if (isFirstInstanceCrossoverCsv(parsed)) {
    return {
      ...summarizeFirstInstanceCrossoverRows(
        parseFirstInstanceCrossoverRecords(parsed),
      ),
      artifactVerification: {
        status: 'unverified',
        scope: 'artifact-integrity-and-schema-only',
        authenticityVerified: false,
        inputKind: 'raw-csv-content',
        evidenceStatus: null,
        reason: 'Raw CSV is not bound to a consistent run artifact manifest.',
      },
    };
  }
  if (isFrozenCrossoverCsv(parsed)) {
    return {
      ...summarizeFrozenCrossoverRows(parseFrozenCrossoverRecords(parsed)),
      artifactVerification: {
        status: 'unverified',
        scope: 'artifact-integrity-and-schema-only',
        authenticityVerified: false,
        inputKind: 'raw-csv-content',
        evidenceStatus: null,
        reason: 'Raw CSV is not bound to a consistent run artifact manifest.',
      },
    };
  }
  const { frames, repetitionColumn } = parseFrameRecords(parsed);
  const groups = groupFrames(frames);
  const depthFrontToBackVsReverse = pairedContrasts(
    groups,
    'fixed-slice-depth-front-to-back',
    'fixed-slice-depth-reverse',
  );
  const depthFrontToBackVsAtomicFixedSlice = pairedContrasts(
    groups,
    'fixed-slice-depth-front-to-back',
    'fixed-slice',
  );
  const depthReverseVsAtomicFixedSlice = pairedContrasts(
    groups,
    'fixed-slice-depth-reverse',
    'fixed-slice',
  );
  return {
    schemaVersion: 2,
    artifactVerification: {
      status: 'unverified',
      scope: 'artifact-integrity-and-schema-only',
      authenticityVerified: false,
      inputKind: 'raw-csv-content',
      evidenceStatus: null,
      reason: 'Raw CSV is not bound to a consistent run artifact manifest.',
    },
    percentileMethod: {
      frameP50: 'nearest-rank',
      acrossTrials: 'arithmetic midpoint for even sample counts',
    },
    repetitionColumn: repetitionColumn ?? '(implicit single trial)',
    deltaConvention: 'left mode minus right mode; negative values mean the left mode is faster',
    nFrames: frames.length,
    groups,
    comparisons: {
      fixedSliceVsDrawAll: comparisonsAgainst(groups, 'draw-all'),
      fixedSliceVsThreeBlocksCurrent: comparisonsAgainst(groups, 'three-blocks-current'),
      fixedSliceVsThreeBlocksCoalesced: comparisonsAgainst(groups, 'three-blocks-coalesced'),
      fixedSliceVsThreeBlocksHistorical: comparisonsAgainst(groups, 'three-blocks-historical'),
    },
    causalContrasts: {
      mergedFixedSliceVsPerBucketRepresentation: pairedContrasts(
        groups,
        'fixed-slice',
        'fixed-slice-per-bucket',
      ),
      depthFrontToBackVsReverse,
    },
    contextualWholeMechanismComparisons: {
      depthFrontToBackVsAtomicFixedSlice,
      depthReverseVsAtomicFixedSlice,
    },
    preregisteredGates: {
      depthOrdering: evaluateDepthOrderingGates(
        depthFrontToBackVsReverse,
        {
          frontToBack: depthFrontToBackVsAtomicFixedSlice,
          reverse: depthReverseVsAtomicFixedSlice,
        },
      ),
    },
  };
}

export async function summarizeInput(inputPath) {
  const absoluteInput = path.resolve(inputPath);
  const inputStat = await stat(absoluteInput);
  if (inputStat.isDirectory()) {
    const verified = await verifyRunDirectory(absoluteInput);
    return {
      ...summarizeCsv(verified.csvText),
      bundleIntegrity: verified.bundleIntegrity,
      ...(verified.liveFirstInstanceEvidenceDecision === null
        ? {}
        : {
          liveFirstInstanceEvidenceDecision: verified.liveFirstInstanceEvidenceDecision,
        }),
      artifactVerification: verified.artifactVerification,
    };
  }
  if (!inputStat.isFile()) {
    throw new Error('Analyzer input must be a run directory or a raw frames.csv file.');
  }
  const csv = await readFile(absoluteInput, 'utf8');
  return {
    ...summarizeCsv(csv),
    artifactVerification: {
      status: 'unverified',
      scope: 'artifact-integrity-and-schema-only',
      authenticityVerified: false,
      inputKind: 'raw-frames-csv',
      evidenceStatus: null,
      reason: 'Standalone CSV input is not bound to a consistent run artifact manifest.',
    },
  };
}

async function main() {
  const arguments_ = process.argv.slice(2);
  if (arguments_.length !== 1) {
    throw new Error('Usage: node analysis/summarize.mjs <run-directory-or-frames.csv>');
  }
  const summary = await summarizeInput(arguments_[0]);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

const invokedUrl = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;

if (import.meta.url === invokedUrl) {
  main().catch((error) => {
    process.stderr.write(`summarize: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
