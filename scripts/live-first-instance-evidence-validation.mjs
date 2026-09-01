import { createHash } from 'node:crypto';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_LIVE_CROSSOVER_HISTORY_TRIPLE_KEYS,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS,
  FIRST_INSTANCE_LIVE_CROSSOVER_TRANSITION_KEYS,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceLiveCrossoverHistoryCounts,
  firstInstanceLiveCrossoverFrame,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS,
  FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS,
} from '../src/benchmark/plan.js';
import { INDEXED_INDIRECT_STRIDE_BYTES } from '../src/culling/indexed-command-layout.js';
import { createFirstInstanceShaderEvidence } from '../src/validation/first-instance-shader-evidence.js';
import {
  normalizeLiveIndirectCommandComputeShader,
} from '../src/validation/live-compute-shader-normalization.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const FNV64_PATTERN = /^[0-9a-f]{16}$/;
const PARITY_WIDTH = 1280;
const PARITY_HEIGHT = 720;
const PARITY_PIXEL_COUNT = PARITY_WIDTH * PARITY_HEIGHT;
const TIMED_FRAME_COUNT = FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES
  + FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES;
const LIVE_SHADER_OBSERVATION_SEQUENCE = Object.freeze([
  Object.freeze({ phase: 'preflight', role: 'render-parity' }),
  Object.freeze({ phase: 'preflight', role: 'main-validation' }),
  Object.freeze({ phase: 'timing-start', role: 'render-parity' }),
  Object.freeze({ phase: 'timing-start', role: 'main-validation' }),
  Object.freeze({ phase: 'postflight', role: 'render-parity' }),
  Object.freeze({ phase: 'postflight', role: 'main-validation' }),
]);
const SHADER_OBSERVATION_REQUEST_KEYS = Object.freeze([
  'schemaVersion', 'kind', 'origin', 'runId', 'trialId', 'planIndex',
  'repetitionIndex', 'phase', 'role', 'captureOrdinal', 'challengeNonce',
]);
const SHADER_OBSERVATION_COUNTER_KEYS = Object.freeze([
  'laneSelectionSerial', 'computeCallSerial', 'prepareSerial',
]);
const SHADER_OBSERVATION_KEYS = Object.freeze([
  'schemaVersion', 'kind', 'serial', 'captureMode', 'request',
  'executionCountersAtStart', 'executionCountersAtEnd', 'executionCountersStable',
  'resourcesPreinitialized', 'resourceIdentitiesAtStart', 'resourceIdentitiesAtEnd',
  'resourceIdentitiesStable', 'inspectionStateBefore', 'inspectionStateAfter',
  'inspectionStateStable', 'semanticSha256', 'observationSha256',
]);
const TOTAL_EQUALITY_TOLERANCE_MS = 1e-9;
const RENDERER_MEMORY_FIELDS = Object.freeze([
  'attributes',
  'attributesSize',
  'geometries',
  'indexAttributes',
  'indexAttributesSize',
  'indirectStorageAttributes',
  'indirectStorageAttributesSize',
  'programs',
  'programsSize',
  'storageAttributes',
  'storageAttributesSize',
  'uniformBuffers',
  'uniformBuffersSize',
]);

export const LIVE_FIRST_INSTANCE_VALIDATION_KIND =
  'first-instance-live-crossover-exact-paired-snapshots';

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isNonnegativeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function isPositiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function isIdentity(value) {
  return (typeof value === 'string' && value.length > 0) || isNonnegativeInteger(value);
}

function requireCondition(condition, message, reasons) {
  if (!condition) reasons.push(message);
}

function requireEqual(actual, expected, label, reasons) {
  if (!Object.is(actual, expected)) {
    reasons.push(`${label} is ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}`);
  }
}

function requireSha256(value, label, reasons) {
  if (!SHA256_PATTERN.test(value ?? '')) reasons.push(`${label} is not a lowercase SHA-256 digest`);
}

function requireFNV64(value, label, reasons) {
  if (!FNV64_PATTERN.test(value ?? '')) reasons.push(`${label} is not a lowercase FNV-1a-64 digest`);
}

function requireExactArray(actual, expected, label, reasons) {
  if (!Array.isArray(actual) || actual.length !== expected.length) {
    reasons.push(`${label} is not the exact expected array`);
    return false;
  }
  for (let index = 0; index < expected.length; index += 1) {
    requireEqual(actual[index], expected[index], `${label}[${index}]`, reasons);
  }
  return true;
}

function requireExactObjectKeys(actual, expected, label, reasons) {
  if (!isRecord(actual)) {
    reasons.push(`${label} is not an object`);
    return false;
  }
  return requireExactArray(Object.keys(actual).sort(), [...expected].sort(), `${label} keys`, reasons);
}

function uniqueReasons(reasons) {
  return [...new Set(reasons)];
}

function sha256Json(value) {
  return createHash('sha256').update(JSON.stringify(value) ?? 'undefined').digest('hex');
}

function fnv1a64Text(value) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function staticLifecycle(lifecycle) {
  if (!isRecord(lifecycle)) return null;
  const clone = structuredClone(lifecycle);
  for (const field of [
    'activeLane', 'activeCommandBufferId', 'residentPreparedLane',
    'laneSelectionSerial', 'computeCallSerial', 'prepareSerial',
  ]) delete clone[field];
  return clone;
}

function stripAddressOrderingCommitments(challenge) {
  if (!isRecord(challenge)) return challenge;
  const clone = structuredClone(challenge);
  for (const field of Object.keys(clone)) {
    if (field === 'sha256'
      || field === 'expectedSha256'
      || field === 'survivorSha256'
      || /^raw.*(?:sha256|digest|bytes|order|ordering|commitment)$/i.test(field)) {
      delete clone[field];
    }
  }
  return clone;
}

function semanticValidation(validation) {
  if (!isRecord(validation)) return validation;
  const clone = structuredClone(validation);
  for (const field of [
    'prepareSerialStart', 'prepareSerialEnd',
    'computeCallSerialStart', 'computeCallSerialEnd',
    'residentPreparedLane',
  ]) delete clone[field];
  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    if (isRecord(clone.lanes?.[lane])) {
      for (const field of [
        'survivorSha256', 'prepareSerial', 'laneSelectionSerial',
        'computeCallSerial', 'residentAfterValidation',
      ]) delete clone.lanes[lane][field];
    }
    if (isRecord(clone.hooks?.[lane])) {
      const hook = clone.hooks[lane];
      if (isRecord(hook.addressChallenge)) {
        hook.addressChallenge = stripAddressOrderingCommitments(hook.addressChallenge);
      }
      if (isRecord(hook.address)) {
        hook.address = stripAddressOrderingCommitments(hook.address);
      }
      delete hook.external;
    }
    if (isRecord(clone.addressChallenges?.lanes?.[lane])) {
      clone.addressChallenges.lanes[lane] = stripAddressOrderingCommitments(
        clone.addressChallenges.lanes[lane],
      );
    }
  }
  if (isRecord(clone.shaderEvidence)) delete clone.shaderEvidence.observation;
  clone.lifecycle = staticLifecycle(clone.lifecycle);
  return clone;
}

export function liveFirstInstanceValidationSemanticSha256(validation) {
  return sha256Json(semanticValidation(validation));
}

function schedulePhase(frameCount, orientationOffset, { cyclicHistory }) {
  let actualPreviousPreviousLaneId = null;
  let actualPreviousLaneId = null;
  return Array.from({ length: frameCount }, (_, phaseFrameIndex) => {
    const frame = firstInstanceLiveCrossoverFrame(phaseFrameIndex, orientationOffset);
    const record = {
      phaseFrameIndex,
      crossoverBlockIndex: frame.crossoverBlockIndex,
      withinBlockPosition: frame.withinBlockPosition,
      patternIndex: frame.patternIndex,
      pattern: frame.pattern,
      previousPreviousLaneId: cyclicHistory
        ? frame.previousPreviousLaneId
        : actualPreviousPreviousLaneId,
      previousLaneId: cyclicHistory ? frame.previousLaneId : actualPreviousLaneId,
      laneId: frame.laneId,
    };
    actualPreviousPreviousLaneId = actualPreviousLaneId;
    actualPreviousLaneId = frame.laneId;
    return record;
  });
}

export function liveFirstInstanceCrossoverScheduleSha256(orientationOffset) {
  return sha256Json({
    schemaVersion: 1,
    kind: 'indirect-first-instance-live-crossover-frame-schedule',
    blockSize: FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
    warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    orientationOffset,
    patterns: FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS,
    historyBoundary:
      'warmup starts with null history; measurement history continues from warmup tail',
    measuredHistoryBalance: firstInstanceLiveCrossoverHistoryCounts(
      FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
      orientationOffset,
    ),
    warmup: schedulePhase(
      FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
      orientationOffset,
      { cyclicHistory: false },
    ),
    measured: schedulePhase(
      FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
      orientationOffset,
      { cyclicHistory: true },
    ),
  });
}

function validateSpec(spec, environment, reasons) {
  if (!isRecord(spec)) {
    reasons.push('live first-instance trial spec is missing');
    return;
  }
  requireEqual(spec.modeId, FIRST_INSTANCE_LIVE_CROSSOVER_MODE, 'spec modeId', reasons);
  requireEqual(spec.objectCount, 65_536, 'spec objectCount', reasons);
  requireEqual(spec.bucketCount, 32, 'spec bucketCount', reasons);
  requireCondition(isNonnegativeInteger(spec.planIndex), 'spec planIndex is invalid', reasons);
  requireCondition(
    isNonnegativeInteger(spec.repetitionIndex)
      && spec.repetitionIndex < FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
    'spec repetitionIndex is invalid',
    reasons,
  );
  requireCondition(typeof spec.trialId === 'string' && spec.trialId.length > 0,
    'spec trialId is missing', reasons);
  requireExactArray(spec.modeOrder, [FIRST_INSTANCE_LIVE_CROSSOVER_MODE],
    'spec mode order', reasons);
  requireEqual(spec.modeOrderPosition, 0, 'spec modeOrderPosition', reasons);
  requireCondition(FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.includes(
    spec.visibilityFraction,
  ), 'spec visibilityFraction is not preregistered', reasons);
  const expectedVisibilityOrder = spec.repetitionIndex % 2 === 0
    ? [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS]
    : [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS].reverse();
  requireExactArray(spec.visibilityOrder, expectedVisibilityOrder,
    'spec preregistered visibility order', reasons);
  requireCondition(spec.visibilityOrderPosition === 0 || spec.visibilityOrderPosition === 1,
    'spec visibilityOrderPosition is invalid', reasons);
  requireEqual(spec.visibilityOrder?.[spec.visibilityOrderPosition], spec.visibilityFraction,
    'spec visibility order position', reasons);
  requireEqual(spec.planIndex, spec.repetitionIndex * 2 + spec.visibilityOrderPosition,
    'spec preregistered planIndex', reasons);
  requireEqual(spec.layout, 'baseline', 'spec layout', reasons);
  requireExactArray(spec.layoutOrder, ['baseline'], 'spec layout order', reasons);
  requireEqual(spec.layoutOrderPosition, 0, 'spec layoutOrderPosition', reasons);
  requireExactArray(
    spec.lanePhysicalOrder,
    FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS[spec.repetitionIndex] ?? [],
    'spec preregistered lane physical order',
    reasons,
  );
  requireEqual(
    spec.superblockOrientationOffset,
    FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS[spec.repetitionIndex],
    'spec preregistered orientation offset',
    reasons,
  );
  requireEqual(environment?.indirectFirstInstanceAvailable, true,
    'environment indirectFirstInstanceAvailable', reasons);
  requireEqual(environment?.reversedDepth, true, 'environment reversedDepth', reasons);
  requireEqual(environment?.rendererReversedDepthBuffer, true,
    'environment rendererReversedDepthBuffer', reasons);
  requireCondition(Number.isInteger(environment?.maxStorageBuffersPerShaderStage)
    && environment.maxStorageBuffersPerShaderStage >= 8,
  'environment maxStorageBuffersPerShaderStage is below eight', reasons);
  requireEqual(environment?.webgpuUncapturedErrorCount, 0,
    'environment webgpuUncapturedErrorCount', reasons);
}

function validateCommandBuffer(commitment, lane, spec, reasons) {
  const label = `${lane} command buffer`;
  if (!isRecord(commitment)) {
    reasons.push(`${label} commitment is missing`);
    return;
  }
  requireEqual(commitment.lane, lane, `${label} lane`, reasons);
  requireCondition(isIdentity(commitment.attributeId), `${label} identity is missing`, reasons);
  requireCondition(isNonnegativeInteger(commitment.attributeVersion),
    `${label} attributeVersion is invalid`, reasons);
  requireEqual(commitment.byteOffset, 0, `${label} byteOffset`, reasons);
  requireEqual(commitment.byteLength,
    spec.bucketCount * INDEXED_INDIRECT_STRIDE_BYTES, `${label} byteLength`, reasons);
  requireEqual(commitment.recordCount, spec.bucketCount, `${label} recordCount`, reasons);
  requireEqual(commitment.drawCommandCount, spec.bucketCount,
    `${label} drawCommandCount`, reasons);
  requireEqual(commitment.firstOffset, 0, `${label} firstOffset`, reasons);
  requireExactArray(
    commitment.allOffsets,
    Array.from({ length: spec.bucketCount }, (_, bucket) => (
      bucket * INDEXED_INDIRECT_STRIDE_BYTES
    )),
    `${label} allOffsets`,
    reasons,
  );
}

function validateMembership(lane, spec, scenarioManifest, reasons) {
  const label = `${lane?.lane ?? 'unknown'} lane`;
  const expectedCount = scenarioManifest?.expectedVisibleCount;
  const expectedDigest = scenarioManifest?.expectedVisibleIdsCanonicalSha256;
  requireEqual(lane?.membership?.pass, true, `${label} membership pass`, reasons);
  requireEqual(lane?.membership?.expectedCount, expectedCount,
    `${label} membership expected count`, reasons);
  requireEqual(lane?.membership?.listedCount, expectedCount,
    `${label} membership listed count`, reasons);
  for (const field of [
    'duplicateIds', 'outOfRangeIds', 'wrongBucketIds', 'listedHiddenIds',
    'missingVisibleIds', 'overflow', 'errors',
  ]) requireEqual(lane?.membership?.[field], 0, `${label} membership ${field}`, reasons);
  const digests = lane?.membershipDigests;
  requireEqual(digests?.schemaVersion, 1, `${label} digest schemaVersion`, reasons);
  requireEqual(digests?.hashAlgorithm, 'sha256', `${label} digest algorithm`, reasons);
  requireEqual(digests?.encoding, 'sorted-uint32-little-endian',
    `${label} digest encoding`, reasons);
  requireEqual(digests?.pass, true, `${label} digest pass`, reasons);
  requireEqual(digests?.invalidExpectedIds, 0, `${label} invalid expected IDs`, reasons);
  requireEqual(digests?.truncatedActualIds, 0, `${label} truncated actual IDs`, reasons);
  for (const side of ['expected', 'actual']) {
    requireEqual(digests?.[side]?.count, expectedCount, `${label} ${side} count`, reasons);
    requireEqual(digests?.[side]?.sha256, expectedDigest, `${label} ${side} digest`, reasons);
    requireSha256(digests?.[side]?.sha256, `${label} ${side} digest`, reasons);
  }
  requireEqual(digests?.perBucket?.length, spec.bucketCount,
    `${label} per-bucket digest count`, reasons);
  if (Array.isArray(digests?.perBucket)) {
    let count = 0;
    for (let bucket = 0; bucket < spec.bucketCount; bucket += 1) {
      const record = digests.perBucket[bucket];
      requireEqual(record?.bucket, bucket, `${label} bucket ${bucket} index`, reasons);
      requireEqual(record?.match, true, `${label} bucket ${bucket} digest match`, reasons);
      requireEqual(record?.actual?.count, record?.expected?.count,
        `${label} bucket ${bucket} count`, reasons);
      requireEqual(record?.actual?.declaredCount, record?.expected?.count,
        `${label} bucket ${bucket} declared count`, reasons);
      requireEqual(record?.actual?.sha256, record?.expected?.sha256,
        `${label} bucket ${bucket} digest`, reasons);
      requireSha256(record?.expected?.sha256, `${label} bucket ${bucket} digest`, reasons);
      if (isNonnegativeInteger(record?.expected?.count)) count += record.expected.count;
    }
    requireEqual(count, expectedCount, `${label} aggregate bucket count`, reasons);
  }
}

function validateCommands(laneRecord, laneId, spec, geometryManifest, reasons) {
  const command = laneRecord?.commandValidation;
  const label = `${laneId} lane command`;
  requireEqual(command?.pass, true, `${label} pass`, reasons);
  requireCondition(Array.isArray(command?.errors) && command.errors.length === 0,
    `${label} errors is not empty`, reasons);
  requireEqual(command?.commandCount, spec.bucketCount, `${label} count`, reasons);
  requireEqual(command?.records?.length, spec.bucketCount, `${label} record count`, reasons);
  let firstIndex = 0;
  let totalInstances = 0;
  if (Array.isArray(command?.records)) {
    for (let bucket = 0; bucket < spec.bucketCount; bucket += 1) {
      const record = command.records[bucket];
      const expectedCount = laneRecord?.membershipDigests?.perBucket?.[bucket]?.expected?.count;
      const expectedFirstInstance = laneId === PORTABLE
        ? 0
        : bucket * Math.ceil(spec.objectCount / spec.bucketCount);
      const indexCount = geometryManifest?.geometries?.[bucket]?.index?.count;
      requireEqual(record?.bucket, bucket, `${label} ${bucket} bucket`, reasons);
      for (const side of ['expected', 'actual']) {
        requireEqual(record?.[side]?.indexCount, indexCount,
          `${label} ${bucket} ${side} indexCount`, reasons);
        requireEqual(record?.[side]?.instanceCount, expectedCount,
          `${label} ${bucket} ${side} instanceCount`, reasons);
        requireEqual(record?.[side]?.firstIndex, firstIndex,
          `${label} ${bucket} ${side} firstIndex`, reasons);
        requireEqual(record?.[side]?.baseVertex, 0,
          `${label} ${bucket} ${side} baseVertex`, reasons);
        requireEqual(record?.[side]?.firstInstance, expectedFirstInstance,
          `${label} ${bucket} ${side} firstInstance`, reasons);
      }
      if (isNonnegativeInteger(record?.actual?.instanceCount)) {
        totalInstances += record.actual.instanceCount;
      }
      if (isNonnegativeInteger(indexCount)) firstIndex += indexCount;
    }
  }
  requireEqual(command?.totalInstanceCount, scenarioVisibleCount(laneRecord),
    `${label} total instance count`, reasons);
  requireEqual(totalInstances, scenarioVisibleCount(laneRecord),
    `${label} reconstructed instance count`, reasons);
}

function scenarioVisibleCount(laneRecord) {
  return laneRecord?.membership?.expectedCount;
}

function validateGeometry(geometry, reasons) {
  requireEqual(geometry?.pass, true, 'live geometry pass', reasons);
  requireExactArray(geometry?.portableAttributeNames,
    ['bucketBase', 'normal', 'position', 'uv'], 'portable geometry attributes', reasons);
  requireExactArray(geometry?.featureAttributeNames,
    ['normal', 'position', 'uv'], 'feature geometry attributes', reasons);
  requireExactArray(geometry?.portableOnlyAttributeNames,
    ['bucketBase'], 'portable-only geometry attributes', reasons);
  requireEqual(geometry?.sharedIndex, true, 'geometry shared index', reasons);
  requireEqual(geometry?.noInstanceSteppedAttributes, true,
    'geometry no instance-stepped attributes', reasons);
  requireExactArray(Object.keys(geometry?.commonAttributesShared ?? {}).sort(),
    ['normal', 'position', 'uv'], 'geometry shared attribute names', reasons);
  for (const name of ['normal', 'position', 'uv']) {
    requireEqual(geometry?.commonAttributesShared?.[name], true,
      `geometry shared ${name}`, reasons);
  }
}

function validateComputeShaderEvidence(evidence, reasons) {
  requireEqual(evidence?.schemaVersion, 1, 'compute shader evidence schemaVersion', reasons);
  requireEqual(evidence?.kind, 'live-first-instance-compute-shader-evidence',
    'compute shader evidence kind', reasons);
  requireEqual(evidence?.pass, true, 'compute shader evidence pass', reasons);
  requireEqual(evidence?.dispatchDimensionsEqual, true,
    'compute dispatch-dimension equality', reasons);
  requireEqual(evidence?.fixedWorkloadExact, true,
    'compute fixed-workload commitment', reasons);
  requireExactArray(evidence?.dispatchDimensions?.reset, [1, 1, 1],
    'compute reset dispatch dimensions', reasons);
  requireExactArray(evidence?.dispatchDimensions?.cull, [1024, 1, 1],
    'compute cull dispatch dimensions', reasons);
  requireEqual(evidence?.maxStorageBindingCount, 8,
    'compute maximum storage-binding count', reasons);
  const expectedSemantics = {
    reset: ['indirectCommands', 'overflow'],
    cull: [
      'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity', 'cullOrder',
      'visibleIds', 'overflow', 'indirectCommands',
    ],
  };
  const expectedExecution = {
    reset: { count: 32, dispatch: [1, 1, 1] },
    cull: { count: 65_536, dispatch: [1024, 1, 1] },
  };
  let reconstructedMaxStorageBindingCount = 0;
  for (const phaseName of ['reset', 'cull']) {
    const phase = evidence?.phases?.[phaseName];
    const label = `compute ${phaseName}`;
    requireEqual(phase?.pass, true, `${label} pass`, reasons);
    requireEqual(phase?.rawWgslEqual, false, `${label} raw WGSL inequality`, reasons);
    requireEqual(phase?.normalizedWgslEqual, true,
      `${label} normalized WGSL equality`, reasons);
    requireEqual(phase?.rawDifferenceRestrictedToLaneLocalIndirectBinding, true,
      `${label} restricted raw WGSL difference`, reasons);
    requireEqual(phase?.executionEqual, true, `${label} execution equality`, reasons);
    requireEqual(phase?.fixedWorkloadExact, true,
      `${label} fixed-workload commitment`, reasons);
    requireEqual(
      JSON.stringify(phase?.fixedExpectation),
      JSON.stringify({
        count: expectedExecution[phaseName].count,
        workgroupSize: [64, 1, 1],
        derivedDispatchSize: expectedExecution[phaseName].dispatch,
      }),
      `${label} fixed expectation`,
      reasons,
    );
    requireCondition(typeof phase?.workgroupDeclaration === 'string'
      && /@workgroup_size\s*\(\s*64(?:\s*,\s*1\s*,\s*1)?\s*\)/.test(
        phase.workgroupDeclaration,
      ), `${label} workgroup declaration is not pinned at 64`, reasons);
    requireEqual(phase?.bindings?.pass, true, `${label} binding pass`, reasons);
    requireEqual(phase?.bindings?.shapeEqual, true, `${label} binding shape`, reasons);
    requireEqual(phase?.bindings?.uniformValuesEqual, true,
      `${label} uniform values`, reasons);
    const comparisons = phase?.bindings?.resourceComparisons;
    requireCondition(Array.isArray(comparisons), `${label} resource comparisons are missing`, reasons);
    if (Array.isArray(comparisons)) {
      const semantics = comparisons.map((record) => record.semantic).sort();
      requireExactArray(semantics, [...expectedSemantics[phaseName]].sort(),
        `${label} storage semantics`, reasons);
      for (const record of comparisons) {
        requireEqual(record?.pass, true, `${label} ${record?.semantic} resource comparison`, reasons);
        requireEqual(record?.shouldShare, record?.semantic !== 'indirectCommands',
          `${label} ${record?.semantic} sharing contract`, reasons);
        requireCondition(isIdentity(record?.portableResourceId),
          `${label} ${record?.semantic} portable identity is missing`, reasons);
        requireCondition(isIdentity(record?.featureResourceId),
          `${label} ${record?.semantic} feature identity is missing`, reasons);
        requireEqual(
          record?.portableResourceId === record?.featureResourceId,
          record?.semantic !== 'indirectCommands',
          `${label} ${record?.semantic} resource identity`,
          reasons,
        );
      }
    }
    const reconstructed = {};
    const capturedBindings = {};
    for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
      const capture = phase?.lanes?.[lane];
      capturedBindings[lane] = capture?.bindings;
      requireCondition(Array.isArray(capturedBindings[lane]),
        `${label} ${lane} runtime bindings are missing`, reasons);
      if (Array.isArray(capturedBindings[lane])) {
        reconstructedMaxStorageBindingCount = Math.max(
          reconstructedMaxStorageBindingCount,
          capturedBindings[lane].filter((binding) => binding?.kind === 'storage-buffer').length,
        );
        requireExactArray(
          capturedBindings[lane]
            .filter((binding) => binding?.kind === 'storage-buffer')
            .map((binding) => binding.semantic)
            .sort(),
          [...expectedSemantics[phaseName]].sort(),
          `${label} ${lane} captured storage semantics`,
          reasons,
        );
      }
      const execution = capture?.execution;
      requireEqual(execution?.count, expectedExecution[phaseName].count,
        `${label} ${lane} invocation count`, reasons);
      requireExactArray(execution?.workgroupSize, [64, 1, 1],
        `${label} ${lane} workgroup size`, reasons);
      requireEqual(execution?.invocationsPerWorkgroup, 64,
        `${label} ${lane} invocations per workgroup`, reasons);
      requireExactArray(execution?.derivedDispatchSize, expectedExecution[phaseName].dispatch,
        `${label} ${lane} derived dispatch`, reasons);
      requireExactArray(execution?.runtimeDispatchSize, expectedExecution[phaseName].dispatch,
        `${label} ${lane} runtime dispatch`, reasons);
      requireEqual(execution?.runtimeMatchesDerived, true,
        `${label} ${lane} runtime dispatch match`, reasons);
      requireEqual(phase?.wordFour?.[lane]?.pass, true,
        `${label} ${lane} command-word-four audit`, reasons);
      requireEqual(phase?.wordFour?.[lane]?.declarationCount, 1,
        `${label} ${lane} command-word-four declaration count`, reasons);
      requireEqual(phase?.wordFour?.[lane]?.executableAccessCount, 0,
        `${label} ${lane} command-word-four access count`, reasons);
      requireCondition(typeof phase?.lanes?.[lane]?.computeShader === 'string'
        && phase.lanes[lane].computeShader.length > 0,
      `${label} ${lane} raw compute WGSL is missing`, reasons);
      if (typeof capture?.computeShader === 'string') {
        requireEqual(
          phase?.wordFour?.[lane]?.declarationCount,
          capture.computeShader.match(/\bfirstInstance\s*:/g)?.length ?? 0,
          `${label} ${lane} reconstructed command-word-four declaration count`,
          reasons,
        );
        requireEqual(
          phase?.wordFour?.[lane]?.executableAccessCount,
          capture.computeShader.match(/\.\s*firstInstance\b/g)?.length ?? 0,
          `${label} ${lane} reconstructed command-word-four access count`,
          reasons,
        );
        requireEqual(
          phase?.workgroupDeclaration,
          capture.computeShader.match(/@workgroup_size\s*\([^)]*\)/)?.[0] ?? null,
          `${label} ${lane} reconstructed workgroup declaration`,
          reasons,
        );
      }
      try {
        reconstructed[lane] = normalizeLiveIndirectCommandComputeShader(
          phase?.lanes?.[lane]?.computeShader,
          phase?.lanes?.[lane]?.bindings,
        );
        requireEqual(
          JSON.stringify(phase?.normalization?.[lane]),
          JSON.stringify(reconstructed[lane].audit),
          `${label} ${lane} normalization audit`,
          reasons,
        );
        requireEqual(
          phase?.sourceDigests?.[lane]?.rawSha256,
          createHash('sha256').update(phase.lanes[lane].computeShader).digest('hex'),
          `${label} ${lane} raw WGSL SHA-256`,
          reasons,
        );
        requireEqual(
          phase?.sourceDigests?.[lane]?.normalizedSha256,
          createHash('sha256').update(reconstructed[lane].normalizedShader).digest('hex'),
          `${label} ${lane} normalized WGSL SHA-256`,
          reasons,
        );
      } catch (error) {
        reasons.push(`${label} ${lane} normalization failed: ${error.message}`);
      }
    }
    if (Array.isArray(capturedBindings[PORTABLE])
      && Array.isArray(capturedBindings[FEATURE])) {
      const shape = (binding) => {
        const { resourceId, uniformValues, ...rest } = binding;
        return rest;
      };
      requireEqual(
        JSON.stringify(capturedBindings[PORTABLE].map(shape)),
        JSON.stringify(capturedBindings[FEATURE].map(shape)),
        `${label} independently reconstructed binding shape`,
        reasons,
      );
      requireEqual(
        JSON.stringify(capturedBindings[PORTABLE].map((binding) => binding.uniformValues)),
        JSON.stringify(capturedBindings[FEATURE].map((binding) => binding.uniformValues)),
        `${label} independently reconstructed uniform values`,
        reasons,
      );
      for (const record of Array.isArray(comparisons) ? comparisons : []) {
        const portableBinding = capturedBindings[PORTABLE].find((binding) => (
          binding.group === record.group && binding.binding === record.binding
        ));
        const featureBinding = capturedBindings[FEATURE].find((binding) => (
          binding.group === record.group && binding.binding === record.binding
        ));
        requireEqual(portableBinding?.semantic, record.semantic,
          `${label} ${record.semantic} captured portable coordinate`, reasons);
        requireEqual(featureBinding?.semantic, record.semantic,
          `${label} ${record.semantic} captured feature coordinate`, reasons);
        requireEqual(portableBinding?.resourceId, record.portableResourceId,
          `${label} ${record.semantic} captured portable identity`, reasons);
        requireEqual(featureBinding?.resourceId, record.featureResourceId,
          `${label} ${record.semantic} captured feature identity`, reasons);
      }
    }
    requireCondition(
      phase?.lanes?.[PORTABLE]?.computeShader !== phase?.lanes?.[FEATURE]?.computeShader,
      `${label} raw compute WGSL must retain the distinct lane-local symbol`,
      reasons,
    );
    if (reconstructed[PORTABLE] && reconstructed[FEATURE]) {
      requireEqual(
        reconstructed[PORTABLE].normalizedShader,
        reconstructed[FEATURE].normalizedShader,
        `${label} independently normalized compute WGSL`,
        reasons,
      );
      requireEqual(
        phase?.sourceDigests?.[PORTABLE]?.normalizedSha256,
        phase?.sourceDigests?.[FEATURE]?.normalizedSha256,
        `${label} normalized WGSL SHA-256 equality`,
        reasons,
      );
      requireCondition(
        reconstructed[PORTABLE].audit.generatedVariableIdentifier
          !== reconstructed[FEATURE].audit.generatedVariableIdentifier,
        `${label} lane-local generated identifiers are not distinct`,
        reasons,
      );
    }
  }
  requireEqual(evidence?.maxStorageBindingCount, reconstructedMaxStorageBindingCount,
    'reconstructed compute maximum storage-binding count', reasons);
}

export function validateLiveComputeShaderEvidence(evidence) {
  const reasons = [];
  validateComputeShaderEvidence(evidence, reasons);
  return uniqueReasons(reasons);
}

async function validateRenderShaderEvidence(evidence, reasons) {
  requireEqual(evidence?.schemaVersion, 1, 'render shader evidence schemaVersion', reasons);
  requireEqual(evidence?.kind, 'indirect-first-instance-shader-evidence',
    'render shader evidence kind', reasons);
  requireEqual(evidence?.pass, true, 'render shader evidence pass', reasons);
  requireCondition(Array.isArray(evidence?.reasons) && evidence.reasons.length === 0,
    'render shader evidence reasons is not empty', reasons);
  for (const [field, expected] of Object.entries({
    rawVertexDifferent: true,
    rawFragmentEqual: true,
    normalizedVertexEqual: true,
    normalizedVertexSha256Equal: true,
  })) requireEqual(evidence?.comparison?.[field], expected,
    `render shader comparison ${field}`, reasons);

  const source = evidence?.rawSources;
  if (!isRecord(source)) {
    reasons.push('render raw shader sources are missing');
    return;
  }
  const input = {};
  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    input[lane] = {
      vertexShader: source?.[lane]?.vertexShader,
      fragmentShader: source?.[lane]?.fragmentShader,
      vertexInputs: evidence?.[lane]?.vertexInputs,
      storageBindings: Object.values(evidence?.[lane]?.storageBindings ?? {}),
    };
  }
  const reconstructed = await createFirstInstanceShaderEvidence(input);
  requireEqual(reconstructed.pass, true, 'reconstructed render shader evidence pass', reasons);
  const capturedWithoutSources = structuredClone(evidence);
  delete capturedWithoutSources.rawSources;
  requireEqual(sha256Json(capturedWithoutSources), sha256Json(reconstructed),
    'reconstructed render shader evidence identity', reasons);
}

function validateShaderInspectionState(state, label, reasons) {
  if (!isRecord(state)) {
    reasons.push(`${label} renderer inspection state is missing`);
    return;
  }
  requireCondition(isNonnegativeInteger(state.totalPipelineCacheEntries),
    `${label} total pipeline-cache count is invalid`, reasons);
  requireExactArray(Object.keys(state.programEntries ?? {}).sort(),
    ['compute', 'fragment', 'vertex'], `${label} program-entry fields`, reasons);
  for (const stage of ['vertex', 'fragment', 'compute']) {
    requireCondition(isNonnegativeInteger(state.programEntries?.[stage]),
      `${label} ${stage} program count is invalid`, reasons);
  }
  requireExactArray(Object.keys(state.memory ?? {}).sort(),
    [...RENDERER_MEMORY_FIELDS].sort(), `${label} memory-counter fields`, reasons);
  for (const field of RENDERER_MEMORY_FIELDS) {
    requireCondition(isNonnegativeInteger(state.memory?.[field]),
      `${label} memory counter ${field} is invalid`, reasons);
  }
}

function validateShaderResourceIdentities(resources, label, reasons) {
  if (!isRecord(resources)) {
    reasons.push(`${label} resource identities are missing`);
    return;
  }
  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    const render = resources.render?.[lane];
    for (const field of [
      'renderObjectIdentity', 'nodeBuilderStateIdentity',
      'bindingArrayIdentity', 'pipelineIdentity',
    ]) requireCondition(isPositiveInteger(render?.[field]),
      `${label} ${lane} render ${field} is invalid`, reasons);
    for (const phase of ['reset', 'cull']) {
      const compute = resources.compute?.[lane]?.[phase];
      requireCondition(isIdentity(compute?.computeNodeId),
        `${label} ${lane} ${phase} compute-node identity is invalid`, reasons);
      requireCondition(isNonnegativeInteger(compute?.computeNodeVersion),
        `${label} ${lane} ${phase} compute-node version is invalid`, reasons);
      for (const field of [
        'nodeBuilderStateIdentity', 'bindingArrayIdentity', 'pipelineIdentity',
      ]) requireCondition(isPositiveInteger(compute?.[field]),
        `${label} ${lane} ${phase} ${field} is invalid`, reasons);
    }
  }
}

function validateShaderObservationRequest(request, spec, label, reasons) {
  requireExactObjectKeys(request, SHADER_OBSERVATION_REQUEST_KEYS, label, reasons);
  requireEqual(request?.schemaVersion, 1, `${label} schemaVersion`, reasons);
  requireEqual(request?.kind, 'live-first-instance-shader-observation-challenge',
    `${label} kind`, reasons);
  requireEqual(request?.origin, 'node-runner', `${label} origin`, reasons);
  requireEqual(request?.runId, spec?.runId, `${label} runId`, reasons);
  requireEqual(request?.trialId, spec?.trialId, `${label} trialId`, reasons);
  requireEqual(request?.planIndex, spec?.planIndex, `${label} planIndex`, reasons);
  requireEqual(request?.repetitionIndex, spec?.repetitionIndex,
    `${label} repetitionIndex`, reasons);
  requireCondition(LIVE_SHADER_OBSERVATION_SEQUENCE.some(
    (entry) => entry.phase === request?.phase && entry.role === request?.role,
  ), `${label} phase/role is unsupported`, reasons);
  requireCondition(isPositiveInteger(request?.captureOrdinal),
    `${label} captureOrdinal is invalid`, reasons);
  requireSha256(request?.challengeNonce, `${label} challenge nonce`, reasons);
}

function validateShaderObservationCounters(counters, label, reasons) {
  requireExactObjectKeys(counters, SHADER_OBSERVATION_COUNTER_KEYS, label, reasons);
  for (const field of SHADER_OBSERVATION_COUNTER_KEYS) {
    requireCondition(isNonnegativeInteger(counters?.[field]),
      `${label} ${field} is invalid`, reasons);
  }
}

function findResourceComparison(evidence, phase, semantic) {
  const records = evidence?.compute?.phases?.[phase]?.bindings?.resourceComparisons;
  return Array.isArray(records)
    ? records.find((record) => record?.semantic === semantic)
    : undefined;
}

function validateShaderLifecycleJoins(evidence, validation, reasons) {
  const lifecycle = validation?.lifecycle;
  const observation = evidence?.observation;
  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    for (const [phaseIndex, phase] of ['reset', 'cull'].entries()) {
      for (const boundary of ['resourceIdentitiesAtStart', 'resourceIdentitiesAtEnd']) {
        requireEqual(
          observation?.[boundary]?.compute?.[lane]?.[phase]?.computeNodeId,
          lifecycle?.computeNodeIds?.[lane]?.[phaseIndex],
          `live shader ${boundary} ${lane} ${phase} compute-node lifecycle identity`,
          reasons,
        );
      }
    }
    for (const semantic of ['matrix', 'visibleIds']) {
      requireEqual(
        evidence?.render?.[lane]?.storageBindings?.[semantic]?.resourceId,
        lifecycle?.sharedAttributeIds?.[semantic],
        `live render ${lane} ${semantic} shared-resource lifecycle identity`,
        reasons,
      );
    }
    for (const semantic of ['position', 'normal']) {
      const input = evidence?.render?.[lane]?.vertexInputs?.find(
        (record) => record?.name === semantic,
      );
      requireEqual(input?.resourceId, lifecycle?.commonVertexAttributeIds?.[semantic],
        `live render ${lane} ${semantic} vertex-resource lifecycle identity`, reasons);
    }
    if (lane === PORTABLE) {
      const bucketBase = evidence?.render?.[lane]?.vertexInputs?.find(
        (record) => record?.name === 'bucketBase',
      );
      requireEqual(bucketBase?.resourceId, lifecycle?.bucketBaseAttributeId,
        'live render portable bucketBase lifecycle identity', reasons);
    }
  }
  for (const phase of ['reset', 'cull']) {
    const records = evidence?.compute?.phases?.[phase]?.bindings?.resourceComparisons;
    if (!Array.isArray(records)) continue;
    for (const record of records) {
      if (record?.semantic === 'indirectCommands') {
        requireEqual(record?.portableResourceId,
          lifecycle?.commandBufferCommitments?.[PORTABLE]?.attributeId,
          `compute ${phase} portable command-buffer lifecycle identity`, reasons);
        requireEqual(record?.featureResourceId,
          lifecycle?.commandBufferCommitments?.[FEATURE]?.attributeId,
          `compute ${phase} feature command-buffer lifecycle identity`, reasons);
      } else {
        requireEqual(record?.portableResourceId,
          lifecycle?.sharedAttributeIds?.[record?.semantic],
          `compute ${phase} ${record?.semantic} portable shared-resource lifecycle identity`,
          reasons);
        requireEqual(record?.featureResourceId,
          lifecycle?.sharedAttributeIds?.[record?.semantic],
          `compute ${phase} ${record?.semantic} feature shared-resource lifecycle identity`,
          reasons);
      }
    }
  }
  for (const semantic of ['overflow', 'indirectCommands']) {
    const reset = findResourceComparison(evidence, 'reset', semantic);
    const cull = findResourceComparison(evidence, 'cull', semantic);
    requireEqual(reset?.portableResourceId, cull?.portableResourceId,
      `compute ${semantic} portable reset/cull lifecycle identity`, reasons);
    requireEqual(reset?.featureResourceId, cull?.featureResourceId,
      `compute ${semantic} feature reset/cull lifecycle identity`, reasons);
  }
}

async function validateShaderEvidence(evidence, validation, spec, reasons) {
  requireEqual(evidence?.schemaVersion, 1, 'live shader evidence schemaVersion', reasons);
  requireEqual(evidence?.kind, 'live-first-instance-compute-render-shader-evidence',
    'live shader evidence kind', reasons);
  requireEqual(evidence?.pass, true, 'live shader evidence pass', reasons);
  await validateRenderShaderEvidence(evidence?.render, reasons);
  validateComputeShaderEvidence(evidence?.compute, reasons);
  const observation = evidence?.observation;
  requireExactObjectKeys(observation, SHADER_OBSERVATION_KEYS,
    'live shader observation', reasons);
  requireEqual(observation?.schemaVersion, 1,
    'live shader observation schemaVersion', reasons);
  requireEqual(observation?.kind, 'live-first-instance-fresh-shader-runtime-observation',
    'live shader observation kind', reasons);
  requireCondition(isPositiveInteger(observation?.serial),
    'live shader observation serial is invalid', reasons);
  requireEqual(observation?.captureMode, 'live-r185-cache-inspection',
    'live shader observation capture mode', reasons);
  validateShaderObservationRequest(
    observation?.request,
    spec,
    'live shader observation request',
    reasons,
  );
  requireEqual(observation?.serial, observation?.request?.captureOrdinal,
    'live shader observation serial/request ordinal', reasons);
  validateShaderObservationCounters(
    observation?.executionCountersAtStart,
    'live shader observation counters at start',
    reasons,
  );
  validateShaderObservationCounters(
    observation?.executionCountersAtEnd,
    'live shader observation counters at end',
    reasons,
  );
  requireEqual(observation?.executionCountersStable, true,
    'live shader observation execution-counter stability', reasons);
  requireEqual(sha256Json(observation?.executionCountersAtStart),
    sha256Json(observation?.executionCountersAtEnd),
    'live shader observation execution counters', reasons);
  requireEqual(observation?.executionCountersAtStart?.prepareSerial,
    validation?.prepareSerialStart,
    'live shader observation/validation prepare counter', reasons);
  requireEqual(observation?.executionCountersAtStart?.computeCallSerial,
    validation?.computeCallSerialStart,
    'live shader observation/validation compute counter', reasons);
  requireEqual(observation?.executionCountersAtStart?.laneSelectionSerial + 1,
    validation?.lanes?.[PORTABLE]?.laneSelectionSerial,
    'live shader observation/portable lane-selection counter', reasons);
  requireEqual(observation?.executionCountersAtStart?.laneSelectionSerial + 2,
    validation?.lanes?.[FEATURE]?.laneSelectionSerial,
    'live shader observation/feature lane-selection counter', reasons);
  requireEqual(observation?.resourcesPreinitialized, true,
    'live shader observation preinitialized resources', reasons);
  requireEqual(observation?.resourceIdentitiesStable, true,
    'live shader observation resource identity stability', reasons);
  requireEqual(observation?.inspectionStateStable, true,
    'live shader observation inspection-state stability', reasons);
  validateShaderResourceIdentities(
    observation?.resourceIdentitiesAtStart,
    'live shader observation start',
    reasons,
  );
  validateShaderResourceIdentities(
    observation?.resourceIdentitiesAtEnd,
    'live shader observation end',
    reasons,
  );
  requireEqual(
    sha256Json(observation?.resourceIdentitiesAtStart),
    sha256Json(observation?.resourceIdentitiesAtEnd),
    'live shader observation resource identities',
    reasons,
  );
  validateShaderInspectionState(
    observation?.inspectionStateBefore,
    'live shader observation before',
    reasons,
  );
  validateShaderInspectionState(
    observation?.inspectionStateAfter,
    'live shader observation after',
    reasons,
  );
  requireEqual(
    sha256Json(observation?.inspectionStateBefore),
    sha256Json(observation?.inspectionStateAfter),
    'live shader observation renderer inspection state',
    reasons,
  );
  const semanticEvidence = structuredClone(evidence);
  delete semanticEvidence.observation;
  requireSha256(observation?.semanticSha256,
    'live shader observation semantic digest', reasons);
  requireEqual(observation?.semanticSha256, sha256Json(semanticEvidence),
    'live shader observation semantic digest recomputation', reasons);
  const observationCore = {
    schemaVersion: observation?.schemaVersion,
    kind: observation?.kind,
    serial: observation?.serial,
    captureMode: observation?.captureMode,
    request: observation?.request,
    executionCountersAtStart: observation?.executionCountersAtStart,
    executionCountersAtEnd: observation?.executionCountersAtEnd,
    executionCountersStable: observation?.executionCountersStable,
    resourcesPreinitialized: observation?.resourcesPreinitialized,
    resourceIdentitiesAtStart: observation?.resourceIdentitiesAtStart,
    resourceIdentitiesAtEnd: observation?.resourceIdentitiesAtEnd,
    resourceIdentitiesStable: observation?.resourceIdentitiesStable,
    inspectionStateBefore: observation?.inspectionStateBefore,
    inspectionStateAfter: observation?.inspectionStateAfter,
    inspectionStateStable: observation?.inspectionStateStable,
    semanticSha256: observation?.semanticSha256,
  };
  requireSha256(observation?.observationSha256,
    'live shader observation digest', reasons);
  requireEqual(observation?.observationSha256, sha256Json(observationCore),
    'live shader observation digest recomputation', reasons);
  validateShaderLifecycleJoins(evidence, validation, reasons);
}

function validateLifecycle(lifecycle, spec, validation, reasons) {
  if (!isRecord(lifecycle)) {
    reasons.push('live first-instance lifecycle evidence is missing');
    return;
  }
  requireEqual(lifecycle.kind, 'live-first-instance-crossover-static-resource-lifecycle',
    'live lifecycle kind', reasons);
  requireEqual(lifecycle.lanesPrimed, true, 'live lifecycle lanesPrimed', reasons);
  for (const [field, expected] of [
    ['lanePhysicalOrder', spec.lanePhysicalOrder],
    ['laneConstructionOrder', spec.lanePhysicalOrder],
    ['firstComputeUseOrder', spec.lanePhysicalOrder],
    ['renderPipelinePrimeOrder', spec.lanePhysicalOrder],
  ]) requireExactArray(lifecycle[field], expected, `live lifecycle ${field}`, reasons);
  requireCondition(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.includes(lifecycle.activeLane),
    'live lifecycle activeLane is unsupported', reasons);
  requireCondition(lifecycle.residentPreparedLane === null
    || FIRST_INSTANCE_LIVE_CROSSOVER_LANES.includes(lifecycle.residentPreparedLane),
  'live lifecycle residentPreparedLane is unsupported', reasons);
  for (const field of ['laneSelectionSerial', 'computeCallSerial', 'prepareSerial']) {
    requireCondition(isNonnegativeInteger(lifecycle[field]),
      `live lifecycle ${field} is invalid`, reasons);
  }
  requireEqual(lifecycle.allBundlesStatic, true, 'live lifecycle allBundlesStatic', reasons);
  requireEqual(lifecycle.commandBuffersDistinct, true,
    'live lifecycle commandBuffersDistinct', reasons);
  requireEqual(lifecycle.commandBuffersZeroOffset, true,
    'live lifecycle commandBuffersZeroOffset', reasons);
  requireEqual(lifecycle.geometry?.pass, true, 'live lifecycle geometry pass', reasons);
  requireEqual(lifecycle.shaderEvidencePass, true,
    'live lifecycle shader evidence pass', reasons);
  requireEqual(lifecycle.configuredComputeDispatches, 2,
    'live lifecycle configured compute dispatches', reasons);
  requireEqual(lifecycle.configuredComputeSubmissions, 1,
    'live lifecycle configured compute submissions', reasons);
  requireCondition(isIdentity(lifecycle.rootUuid), 'live lifecycle root identity is missing', reasons);
  requireCondition(lifecycle.rootVersion === null || isNonnegativeInteger(lifecycle.rootVersion),
    'live lifecycle root version is invalid', reasons);
  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    requireEqual(lifecycle.bundleStaticFlags?.[lane], true,
      `${lane} lifecycle static bundle`, reasons);
    requireEqual(lifecycle.bundleRecordCounts?.[lane], 1,
      `${lane} lifecycle bundle record count`, reasons);
    for (const field of ['bundleUuids', 'meshUuids', 'geometryUuids', 'materialUuids']) {
      requireCondition(isIdentity(lifecycle[field]?.[lane]),
        `${lane} lifecycle ${field} identity is missing`, reasons);
    }
    requireCondition(lifecycle.bundleVersions?.[lane] === null
      || isNonnegativeInteger(lifecycle.bundleVersions?.[lane]),
    `${lane} lifecycle bundle version is invalid`, reasons);
    requireCondition(isNonnegativeInteger(lifecycle.materialVersions?.[lane]),
      `${lane} lifecycle material version is invalid`, reasons);
    requireCondition(Array.isArray(lifecycle.computeNodeIds?.[lane])
      && lifecycle.computeNodeIds[lane].length === 2
      && lifecycle.computeNodeIds[lane].every(isIdentity),
    `${lane} lifecycle compute-node identities are invalid`, reasons);
    requireCondition(isPositiveInteger(lifecycle.computeTimestampContextIds?.[lane]),
      `${lane} lifecycle compute timestamp context is invalid`, reasons);
    const registration = lifecycle.computeTimestampRegistrations?.[lane];
    requireExactObjectKeys(registration, [
      'schemaVersion', 'kind', 'laneId', 'timestampContextId', 'registrationSerial',
      'backendIdentity', 'backendWrapperIdentity', 'computeGroupIdentity', 'computeNodeIds',
    ], `${lane} lifecycle compute timestamp registration`, reasons);
    requireEqual(registration?.schemaVersion, 1,
      `${lane} compute timestamp registration schemaVersion`, reasons);
    requireEqual(registration?.kind,
      'live-first-instance-compute-timestamp-group-registration',
      `${lane} compute timestamp registration kind`, reasons);
    requireEqual(registration?.laneId, lane,
      `${lane} compute timestamp registration lane`, reasons);
    requireEqual(registration?.timestampContextId,
      lifecycle.computeTimestampContextIds?.[lane],
      `${lane} compute timestamp registration context`, reasons);
    for (const field of [
      'registrationSerial', 'backendIdentity', 'backendWrapperIdentity',
      'computeGroupIdentity',
    ]) requireCondition(isPositiveInteger(registration?.[field]),
      `${lane} compute timestamp registration ${field} is invalid`, reasons);
    requireExactArray(registration?.computeNodeIds, lifecycle.computeNodeIds?.[lane] ?? [],
      `${lane} compute timestamp registration node IDs`, reasons);
    validateCommandBuffer(lifecycle.commandBufferCommitments?.[lane], lane, spec, reasons);
    requireEqual(sha256Json(lifecycle.commandBufferCommitments?.[lane]),
      sha256Json(validation?.commandBufferCommitments?.[lane]),
      `${lane} lifecycle command-buffer commitment`, reasons);
    requireCondition(isNonnegativeInteger(lifecycle.indirectAttributeVersions?.[lane]),
      `${lane} lifecycle indirect attribute version is invalid`, reasons);
    requireEqual(lifecycle.indirectAttributeVersions?.[lane],
      lifecycle.commandBufferCommitments?.[lane]?.attributeVersion,
      `${lane} lifecycle indirect attribute version commitment`, reasons);
  }
  for (const field of ['bundleUuids', 'meshUuids', 'geometryUuids', 'materialUuids']) {
    requireCondition(lifecycle[field]?.[PORTABLE] !== lifecycle[field]?.[FEATURE],
      `live lifecycle ${field} lanes are not distinct`, reasons);
  }
  requireCondition(lifecycle.commandBufferCommitments?.[PORTABLE]?.attributeId
    !== lifecycle.commandBufferCommitments?.[FEATURE]?.attributeId,
  'live lifecycle command-buffer identities are aliased', reasons);
  requireCondition(lifecycle.computeTimestampContextIds?.[PORTABLE]
    !== lifecycle.computeTimestampContextIds?.[FEATURE],
  'live lifecycle compute timestamp contexts are aliased', reasons);
  requireCondition(lifecycle.computeTimestampRegistrations?.[PORTABLE]?.computeGroupIdentity
    !== lifecycle.computeTimestampRegistrations?.[FEATURE]?.computeGroupIdentity,
  'live lifecycle compute timestamp groups are aliased', reasons);
  requireEqual(lifecycle.computeTimestampRegistrations?.[PORTABLE]?.backendIdentity,
    lifecycle.computeTimestampRegistrations?.[FEATURE]?.backendIdentity,
    'live lifecycle compute timestamp backend identity', reasons);
  requireEqual(lifecycle.computeTimestampRegistrations?.[PORTABLE]?.backendWrapperIdentity,
    lifecycle.computeTimestampRegistrations?.[FEATURE]?.backendWrapperIdentity,
    'live lifecycle compute timestamp wrapper identity', reasons);
  requireEqual(lifecycle.computeTimestampRegistrations?.[FEATURE]?.registrationSerial,
    lifecycle.computeTimestampRegistrations?.[PORTABLE]?.registrationSerial + 1,
    'live lifecycle compute timestamp registration sequence', reasons);
  requireEqual(lifecycle.activeCommandBufferId,
    lifecycle.commandBufferCommitments?.[lifecycle.activeLane]?.attributeId,
    'live lifecycle active command-buffer identity', reasons);
  const sharedNames = [
    'matrix', 'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity',
    'cullOrder', 'visibleIds', 'overflow',
  ];
  requireExactArray(Object.keys(lifecycle.sharedAttributeIds ?? {}).sort(),
    [...sharedNames].sort(), 'live lifecycle shared attribute IDs', reasons);
  requireExactArray(Object.keys(lifecycle.sharedAttributeVersions ?? {}).sort(),
    [...sharedNames].sort(), 'live lifecycle shared attribute versions', reasons);
  for (const name of sharedNames) {
    requireCondition(isIdentity(lifecycle.sharedAttributeIds?.[name]),
      `live lifecycle ${name} identity is missing`, reasons);
    requireCondition(isNonnegativeInteger(lifecycle.sharedAttributeVersions?.[name]),
      `live lifecycle ${name} version is invalid`, reasons);
  }
  requireExactArray(Object.keys(lifecycle.commonVertexAttributeIds ?? {}).sort(),
    ['normal', 'position', 'uv'], 'live lifecycle common vertex attribute IDs', reasons);
  requireExactArray(Object.keys(lifecycle.commonVertexAttributeVersions ?? {}).sort(),
    ['normal', 'position', 'uv'],
    'live lifecycle common vertex attribute versions', reasons);
  for (const name of ['normal', 'position', 'uv']) {
    requireCondition(isIdentity(lifecycle.commonVertexAttributeIds?.[name]),
      `live lifecycle ${name} vertex identity is missing`, reasons);
    requireCondition(isNonnegativeInteger(lifecycle.commonVertexAttributeVersions?.[name]),
      `live lifecycle ${name} vertex version is invalid`, reasons);
  }
  for (const field of [
    'indexAttributeId', 'bucketBaseAttributeId',
  ]) requireCondition(isIdentity(lifecycle[field]), `live lifecycle ${field} is missing`, reasons);
  for (const field of [
    'indexAttributeVersion', 'bucketBaseAttributeVersion',
  ]) requireCondition(isNonnegativeInteger(lifecycle[field]),
    `live lifecycle ${field} is invalid`, reasons);
}

function validateAddressHook(hook, lane, validation, spec, reasons) {
  const label = `${lane} serialized validation hook`;
  if (!isRecord(hook)) {
    reasons.push(`${label} is missing`);
    return;
  }
  requireEqual(hook.pass, true, `${label} pass`, reasons);
  requireEqual(hook.lane ?? lane, lane, `${label} lane`, reasons);
  const challenge = hook.addressChallenge ?? hook.address;
  if (!isRecord(challenge)) {
    reasons.push(`${label} address challenge is missing`);
    return;
  }
  requireEqual(challenge.pass, true, `${label} address pass`, reasons);
  requireEqual(challenge.lane, lane, `${label} address lane`, reasons);
  requireEqual(challenge.commandByteOffset ?? challenge.commandBuffer?.byteOffset, 0,
    `${label} command byte offset`, reasons);
  const commandId = challenge.commandBufferId
    ?? challenge.commandBuffer?.attributeId
    ?? challenge.commandAttributeId;
  requireEqual(commandId, validation?.commandBufferCommitments?.[lane]?.attributeId,
    `${label} command-buffer identity`, reasons);
  if (isRecord(challenge.commandBuffer)) {
    requireEqual(sha256Json(challenge.commandBuffer),
      sha256Json(validation?.commandBufferCommitments?.[lane]),
      `${label} command-buffer commitment`, reasons);
  }
  requireEqual(challenge.exactExpectedBytes, true,
    `${label} exact expected address bytes`, reasons);
  for (const field of [
    'activeMismatchCount', 'paddingMismatchCount', 'targetPaddingMismatchCount',
    'outOfRangeIds',
  ]) {
    if (field in challenge) requireEqual(challenge[field], 0, `${label} ${field}`, reasons);
  }
  requireEqual(challenge.activeAddressCount,
    validation?.lanes?.[lane]?.membership?.expectedCount,
    `${label} active address count`, reasons);
  requireEqual(challenge.paddingAddressCount,
    spec.objectCount - validation?.lanes?.[lane]?.membership?.expectedCount,
    `${label} padding address count`, reasons);
  requireSha256(challenge.sha256, `${label} address digest`, reasons);
  requireEqual(challenge.sha256, challenge.expectedSha256,
    `${label} address expected digest`, reasons);
}

export async function validateLiveFirstInstanceCrossoverValidation(validation, {
  spec,
  environment,
  scenarioManifest,
  geometryManifest,
} = {}) {
  const reasons = [];
  validateSpec(spec, environment, reasons);
  if (!isRecord(validation)) {
    return uniqueReasons([...reasons, 'live first-instance exact validation is missing']);
  }
  requireEqual(validation.schemaVersion, 1, 'live validation schemaVersion', reasons);
  requireEqual(validation.kind, LIVE_FIRST_INSTANCE_VALIDATION_KIND,
    'live validation kind', reasons);
  requireEqual(validation.pass, true, 'live validation pass', reasons);
  requireExactArray(validation.validationOrder, [PORTABLE, FEATURE],
    'live serialized validation order', reasons);
  requireCondition(isNonnegativeInteger(validation.prepareSerialStart),
    'live validation prepareSerialStart is invalid', reasons);
  requireEqual(validation.prepareSerialEnd, validation.prepareSerialStart + 2,
    'live validation prepare serial delta', reasons);
  requireCondition(isNonnegativeInteger(validation.computeCallSerialStart),
    'live validation computeCallSerialStart is invalid', reasons);
  requireEqual(validation.computeCallSerialEnd, validation.computeCallSerialStart + 2,
    'live validation compute-call serial delta', reasons);
  requireEqual(validation.exactlyOneComputePerLane, true,
    'live validation exactlyOneComputePerLane', reasons);
  requireEqual(validation.commandCoresEqual, true,
    'live validation commandCoresEqual', reasons);
  requireEqual(validation.canonicalMembershipEqual, true,
    'live validation canonicalMembershipEqual', reasons);
  requireEqual(validation.rawSurvivorOrderRequiredEqual, false,
    'live validation raw survivor equality requirement', reasons);
  requireEqual(validation.residentPreparedLane, FEATURE,
    'live validation resident prepared lane', reasons);

  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    const laneRecord = validation.lanes?.[lane];
    requireEqual(laneRecord?.kind, 'live-first-instance-lane-snapshot-validation',
      `${lane} lane validation kind`, reasons);
    requireEqual(laneRecord?.pass, true, `${lane} lane validation pass`, reasons);
    requireEqual(laneRecord?.lane, lane, `${lane} lane validation lane`, reasons);
    requireEqual(laneRecord?.overflow, 0, `${lane} lane overflow`, reasons);
    requireSha256(laneRecord?.commandSha256, `${lane} lane command digest`, reasons);
    requireSha256(laneRecord?.survivorSha256, `${lane} lane survivor digest`, reasons);
    requireCondition(isNonnegativeInteger(laneRecord?.prepareSerial),
      `${lane} lane prepare serial is invalid`, reasons);
    requireCondition(isNonnegativeInteger(laneRecord?.laneSelectionSerial),
      `${lane} lane selection serial is invalid`, reasons);
    requireCondition(isNonnegativeInteger(laneRecord?.computeCallSerial),
      `${lane} lane compute-call serial is invalid`, reasons);
    requireEqual(laneRecord?.residentAfterValidation, true,
      `${lane} lane residentAfterValidation`, reasons);
    validateCommandBuffer(validation.commandBufferCommitments?.[lane], lane, spec, reasons);
    validateCommandBuffer(laneRecord?.commandBuffer, lane, spec, reasons);
    requireEqual(sha256Json(laneRecord?.commandBuffer),
      sha256Json(validation.commandBufferCommitments?.[lane]),
      `${lane} lane command-buffer identity`, reasons);
    validateMembership(laneRecord, spec, scenarioManifest, reasons);
    validateCommands(laneRecord, lane, spec, geometryManifest, reasons);
    validateAddressHook(validation.hooks?.[lane], lane, validation, spec, reasons);
  }
  requireEqual(validation.lanes?.[PORTABLE]?.prepareSerial,
    validation.prepareSerialStart + 1, 'portable lane prepare serial', reasons);
  requireEqual(validation.lanes?.[FEATURE]?.prepareSerial,
    validation.prepareSerialStart + 2, 'feature lane prepare serial', reasons);
  requireEqual(validation.lanes?.[PORTABLE]?.computeCallSerial,
    validation.computeCallSerialStart + 1, 'portable lane compute-call serial', reasons);
  requireEqual(validation.lanes?.[FEATURE]?.computeCallSerial,
    validation.computeCallSerialStart + 2, 'feature lane compute-call serial', reasons);
  requireCondition(validation.commandBufferCommitments?.[PORTABLE]?.attributeId
    !== validation.commandBufferCommitments?.[FEATURE]?.attributeId,
  'live validation command-buffer identities are aliased', reasons);
  requireEqual(validation.lanes?.[PORTABLE]?.membershipDigests?.actual?.sha256,
    validation.lanes?.[FEATURE]?.membershipDigests?.actual?.sha256,
    'live validation cross-lane membership digest', reasons);
  validateGeometry(validation.geometry, reasons);
  await validateShaderEvidence(validation.shaderEvidence, validation, spec, reasons);
  validateLifecycle(validation.lifecycle, spec, validation, reasons);
  return uniqueReasons(reasons);
}

function parityLaneIdentity(lane) {
  const production = lane?.productionBundleOutput;
  return {
    width: lane?.width,
    height: lane?.height,
    reversedDepthBuffer: lane?.reversedDepthBuffer,
    material: lane?.material,
    color: lane?.color,
    depth: lane?.depth,
    objectId: lane?.objectId,
    objectIdValidation: lane?.objectIdValidation,
    productionBundleOutput: {
      schemaVersion: production?.schemaVersion,
      kind: production?.kind,
      pass: production?.pass,
      laneId: production?.laneId,
      target: {
        width: production?.target?.width,
        height: production?.target?.height,
        samples: production?.target?.samples,
        depthBuffer: production?.target?.depthBuffer,
      },
      captures: production?.captures,
      color: production?.color,
      directDiagnosticColor: production?.directDiagnosticColor,
      directDiagnosticExact: production?.directDiagnosticExact,
      resourcesStable: production?.resourcesStable,
      bundleRecordedExactlyOnce: production?.bundleRecordedExactlyOnce,
      executionExact: production?.executionExact,
      stability: production?.stability,
      renderCallDeltas: [
        production?.executionBetween?.rendererRenderCallSerial
          - production?.executionBefore?.rendererRenderCallSerial,
        production?.executionAfter?.rendererRenderCallSerial
          - production?.executionBetween?.rendererRenderCallSerial,
      ],
    },
  };
}

export function liveFirstInstanceRenderParityIdentity(parity) {
  return sha256Json({
    schemaVersion: parity?.schemaVersion,
    kind: parity?.kind,
    laneIds: parity?.laneIds,
    crossLaneExact: parity?.crossLaneExact,
    lanes: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
      (lane) => [lane, parityLaneIdentity(parity?.lanes?.[lane])],
    )),
    membership: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
      (lane) => [lane, {
        count: parity?.snapshotValidation?.lanes?.[lane]
          ?.membershipDigests?.actual?.count ?? null,
        sha256: parity?.snapshotValidation?.lanes?.[lane]
          ?.membershipDigests?.actual?.sha256 ?? null,
      }],
    )),
  });
}

function validateParityLane(lane, label, reasons) {
  requireExactObjectKeys(lane, [
    'schemaVersion', 'kind', 'pass', 'width', 'height', 'captures', 'material',
    'color', 'depth', 'objectId', 'objectIdValidation', 'reversedDepthBuffer',
    'stability', 'productionBundleOutput',
  ], label, reasons);
  requireEqual(lane?.schemaVersion, 1, `${label} schemaVersion`, reasons);
  requireEqual(lane?.kind, 'fixed-camera-offscreen-exact-render-parity',
    `${label} kind`, reasons);
  requireEqual(lane?.pass, true, `${label} pass`, reasons);
  requireEqual(lane?.width, PARITY_WIDTH, `${label} width`, reasons);
  requireEqual(lane?.height, PARITY_HEIGHT, `${label} height`, reasons);
  requireEqual(lane?.captures, 2, `${label} captures`, reasons);
  requireEqual(lane?.reversedDepthBuffer, true, `${label} reversedDepthBuffer`, reasons);
  requireEqual(lane?.stability?.pass, true, `${label} stability pass`, reasons);
  const channels = {
    color: { format: 'rgba8unorm', arrayType: 'Uint8Array' },
    depth: { format: 'depth32float', arrayType: 'Float32Array' },
    objectId: { format: 'rgba8unorm-object-id-plus-one', arrayType: 'Uint8Array' },
  };
  for (const [name, schema] of Object.entries(channels)) {
    requireEqual(lane?.[name]?.format, schema.format, `${label} ${name} format`, reasons);
    requireEqual(lane?.[name]?.arrayType, schema.arrayType,
      `${label} ${name} array type`, reasons);
    requireEqual(lane?.[name]?.byteLength, PARITY_PIXEL_COUNT * 4,
      `${label} ${name} byteLength`, reasons);
    requireSha256(lane?.[name]?.sha256, `${label} ${name} digest`, reasons);
    requireEqual(lane?.stability?.firstCapture?.[name]?.sha256,
      lane?.[name]?.sha256, `${label} stable ${name} digest`, reasons);
    requireEqual(lane?.stability?.first?.[`${name}Sha256`],
      lane?.[name]?.sha256, `${label} first ${name} digest`, reasons);
  }
  requireEqual(lane?.objectIdValidation?.pass, true,
    `${label} object-ID validation pass`, reasons);
  requireEqual(lane?.objectIdValidation?.encoding,
    'rgb24-object-id-plus-one-zero-background', `${label} object-ID encoding`, reasons);
  requireCondition(isPositiveInteger(lane?.objectIdValidation?.coveredPixels),
    `${label} object-ID coverage is empty`, reasons);
  requireCondition(isNonnegativeInteger(lane?.objectIdValidation?.backgroundPixels),
    `${label} object-ID background count is invalid`, reasons);
  requireEqual(lane?.objectIdValidation?.outOfRangePixels, 0,
    `${label} out-of-range pixels`, reasons);
  requireEqual(lane?.objectIdValidation?.nonVisiblePixels, 0,
    `${label} non-visible pixels`, reasons);
  requireEqual(lane?.objectIdValidation?.coveredPixels
    + lane?.objectIdValidation?.backgroundPixels, PARITY_PIXEL_COUNT,
  `${label} classified pixel count`, reasons);
}

const PRODUCTION_OUTPUT_KEYS = Object.freeze([
  'schemaVersion', 'kind', 'pass', 'laneId', 'target', 'captures', 'color',
  'directDiagnosticColor', 'directDiagnosticExact', 'resourcesStable',
  'bundleRecordedExactlyOnce', 'executionExact', 'executionBefore',
  'executionBetween', 'executionAfter', 'commitmentBefore', 'commitmentBetween',
  'commitmentAfter', 'stability',
]);
const TIMED_BUNDLE_COMMITMENT_KEYS = Object.freeze([
  'schemaVersion', 'kind', 'laneId', 'bundleUuid', 'bundleVersion', 'meshUuid',
  'geometryUuid', 'materialUuid', 'materialVersion', 'bundleRecordCallbackCount',
  'renderResourceIdentities', 'inspectionState',
]);
const PRODUCTION_EXECUTION_COUNTER_KEYS = Object.freeze([
  'laneSelectionSerial', 'strategyComputeCallSerial', 'strategyPrepareSerial',
  'rendererComputeCallSerial', 'rendererRenderCallSerial',
]);

function validateExactColorReadback(record, label, reasons) {
  requireExactObjectKeys(record, ['format', 'arrayType', 'byteLength', 'sha256'],
    label, reasons);
  requireEqual(record?.format, 'rgba8unorm', `${label} format`, reasons);
  requireEqual(record?.arrayType, 'Uint8Array', `${label} array type`, reasons);
  requireEqual(record?.byteLength, PARITY_PIXEL_COUNT * 4,
    `${label} byteLength`, reasons);
  requireSha256(record?.sha256, `${label} digest`, reasons);
}

function validateProductionExecutionCounters(counters, label, reasons) {
  requireExactObjectKeys(counters, PRODUCTION_EXECUTION_COUNTER_KEYS, label, reasons);
  for (const field of PRODUCTION_EXECUTION_COUNTER_KEYS) {
    requireCondition(isNonnegativeInteger(counters?.[field]),
      `${label} ${field} is invalid`, reasons);
  }
}

function validateTimedBundleCommitment(commitment, lane, validation, label, reasons) {
  requireExactObjectKeys(commitment, TIMED_BUNDLE_COMMITMENT_KEYS, label, reasons);
  requireEqual(commitment?.schemaVersion, 1, `${label} schemaVersion`, reasons);
  requireEqual(commitment?.kind, 'live-first-instance-timed-bundle-commitment',
    `${label} kind`, reasons);
  requireEqual(commitment?.laneId, lane, `${label} laneId`, reasons);
  const lifecycle = validation?.lifecycle;
  for (const [field, expected] of [
    ['bundleUuid', lifecycle?.bundleUuids?.[lane]],
    ['bundleVersion', lifecycle?.bundleVersions?.[lane]],
    ['meshUuid', lifecycle?.meshUuids?.[lane]],
    ['geometryUuid', lifecycle?.geometryUuids?.[lane]],
    ['materialUuid', lifecycle?.materialUuids?.[lane]],
    ['materialVersion', lifecycle?.materialVersions?.[lane]],
  ]) requireEqual(commitment?.[field], expected, `${label} ${field}`, reasons);
  requireEqual(commitment?.bundleRecordCallbackCount, 1,
    `${label} bundle record count`, reasons);
  const renderIdentities = commitment?.renderResourceIdentities;
  requireExactObjectKeys(renderIdentities, [
    'renderObjectIdentity', 'nodeBuilderStateIdentity',
    'bindingArrayIdentity', 'pipelineIdentity',
  ], `${label} render-resource identities`, reasons);
  for (const field of [
    'renderObjectIdentity', 'nodeBuilderStateIdentity',
    'bindingArrayIdentity', 'pipelineIdentity',
  ]) requireCondition(isPositiveInteger(renderIdentities?.[field]),
    `${label} ${field} is invalid`, reasons);
  requireEqual(
    sha256Json(renderIdentities),
    sha256Json(validation?.shaderEvidence?.observation?.resourceIdentitiesAtStart?.render?.[lane]),
    `${label} shader-observation render-resource identity`,
    reasons,
  );
  validateShaderInspectionState(commitment?.inspectionState, label, reasons);
}

function validateProductionBundleOutput(output, lane, parityLane, validation, reasons) {
  const label = `${lane} production bundle output`;
  requireExactObjectKeys(output, PRODUCTION_OUTPUT_KEYS, label, reasons);
  requireEqual(output?.schemaVersion, 1, `${label} schemaVersion`, reasons);
  requireEqual(output?.kind, 'live-first-instance-production-bundle-output',
    `${label} kind`, reasons);
  requireEqual(output?.pass, true, `${label} pass`, reasons);
  requireEqual(output?.laneId, lane, `${label} laneId`, reasons);
  requireExactObjectKeys(output?.target,
    ['textureUuid', 'width', 'height', 'samples', 'depthBuffer'],
    `${label} target`, reasons);
  requireCondition(isIdentity(output?.target?.textureUuid),
    `${label} target texture identity is missing`, reasons);
  requireEqual(output?.target?.width, PARITY_WIDTH, `${label} target width`, reasons);
  requireEqual(output?.target?.height, PARITY_HEIGHT, `${label} target height`, reasons);
  requireEqual(output?.target?.samples, 0, `${label} target samples`, reasons);
  requireEqual(output?.target?.depthBuffer, true, `${label} target depthBuffer`, reasons);
  requireEqual(output?.captures, 2, `${label} captures`, reasons);
  for (const field of [
    'directDiagnosticExact', 'resourcesStable', 'bundleRecordedExactlyOnce', 'executionExact',
  ]) requireEqual(output?.[field], true, `${label} ${field}`, reasons);
  validateExactColorReadback(output?.color, `${label} color`, reasons);
  validateExactColorReadback(
    output?.directDiagnosticColor,
    `${label} direct diagnostic color`,
    reasons,
  );
  requireEqual(sha256Json(output?.color), sha256Json(output?.directDiagnosticColor),
    `${label} direct diagnostic color equality`, reasons);
  requireEqual(sha256Json(output?.color), sha256Json(parityLane?.color),
    `${label} parity color equality`, reasons);
  requireExactObjectKeys(output?.stability,
    ['pass', 'firstCapture', 'secondCapture'], `${label} stability`, reasons);
  requireEqual(output?.stability?.pass, true, `${label} stability pass`, reasons);
  for (const field of ['firstCapture', 'secondCapture']) {
    validateExactColorReadback(output?.stability?.[field],
      `${label} ${field}`, reasons);
    requireEqual(sha256Json(output?.stability?.[field]), sha256Json(output?.color),
      `${label} ${field} equality`, reasons);
  }
  for (const field of ['commitmentBefore', 'commitmentBetween', 'commitmentAfter']) {
    validateTimedBundleCommitment(output?.[field], lane, validation,
      `${label} ${field}`, reasons);
  }
  requireEqual(sha256Json(output?.commitmentBefore), sha256Json(output?.commitmentBetween),
    `${label} before/between commitment`, reasons);
  requireEqual(sha256Json(output?.commitmentBefore), sha256Json(output?.commitmentAfter),
    `${label} before/after commitment`, reasons);
  for (const field of ['executionBefore', 'executionBetween', 'executionAfter']) {
    validateProductionExecutionCounters(output?.[field], `${label} ${field}`, reasons);
  }
  const before = output?.executionBefore;
  const between = output?.executionBetween;
  const after = output?.executionAfter;
  requireEqual(between?.rendererRenderCallSerial, before?.rendererRenderCallSerial + 1,
    `${label} first production render`, reasons);
  requireEqual(after?.rendererRenderCallSerial, between?.rendererRenderCallSerial + 1,
    `${label} second production render`, reasons);
  for (const field of [
    'laneSelectionSerial', 'strategyComputeCallSerial', 'strategyPrepareSerial',
    'rendererComputeCallSerial',
  ]) {
    requireEqual(between?.[field], before?.[field],
      `${label} first capture ${field}`, reasons);
    requireEqual(after?.[field], before?.[field],
      `${label} second capture ${field}`, reasons);
  }
}

export function validateLiveFirstInstanceCrossoverRenderParity(parity, {
  spec,
  validation,
  scenarioManifest,
} = {}) {
  const reasons = [];
  if (!isRecord(parity)) return ['live first-instance render-parity evidence is missing'];
  requireExactObjectKeys(parity, [
    'schemaVersion', 'kind', 'pass', 'laneIds', 'crossLaneExact',
    'crossLaneProductionExact', 'lanes', 'snapshotValidation',
  ], 'live render parity', reasons);
  requireEqual(spec?.modeId, FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
    'live render parity spec modeId', reasons);
  requireEqual(parity.schemaVersion, 1, 'live render parity schemaVersion', reasons);
  requireEqual(parity.kind, 'first-instance-live-crossover-exact-render-parity',
    'live render parity kind', reasons);
  requireEqual(parity.pass, true, 'live render parity pass', reasons);
  requireExactArray(parity.laneIds, FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
    'live render parity lanes', reasons);
  requireEqual(parity.crossLaneExact, true, 'live render parity crossLaneExact', reasons);
  requireEqual(parity.crossLaneProductionExact, true,
    'live render parity crossLaneProductionExact', reasons);
  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    validateParityLane(parity.lanes?.[lane], `${lane} live render parity`, reasons);
    validateProductionBundleOutput(
      parity.lanes?.[lane]?.productionBundleOutput,
      lane,
      parity.lanes?.[lane],
      validation,
      reasons,
    );
    requireEqual(
      parity.snapshotValidation?.lanes?.[lane]?.membershipDigests?.actual?.sha256,
      scenarioManifest?.expectedVisibleIdsCanonicalSha256,
      `${lane} render-parity membership digest`,
      reasons,
    );
  }
  const portable = parity.lanes?.[PORTABLE];
  const feature = parity.lanes?.[FEATURE];
  for (const field of ['material', 'color', 'depth', 'objectId', 'objectIdValidation']) {
    requireEqual(sha256Json(portable?.[field]), sha256Json(feature?.[field]),
      `live render parity cross-lane ${field}`, reasons);
  }
  requireEqual(portable?.reversedDepthBuffer, feature?.reversedDepthBuffer,
    'live render parity cross-lane reversedDepthBuffer', reasons);
  requireEqual(
    sha256Json(portable?.productionBundleOutput?.color),
    sha256Json(feature?.productionBundleOutput?.color),
    'live render parity cross-lane production color',
    reasons,
  );
  requireEqual(
    portable?.productionBundleOutput?.target?.textureUuid,
    feature?.productionBundleOutput?.target?.textureUuid,
    'live render parity cross-lane production target',
    reasons,
  );
  requireEqual(parity.snapshotValidation?.pass, true,
    'live render parity snapshot validation pass', reasons);
  requireEqual(parity.snapshotValidation?.kind, LIVE_FIRST_INSTANCE_VALIDATION_KIND,
    'live render parity snapshot validation kind', reasons);
  requireEqual(liveFirstInstanceValidationSemanticSha256(parity.snapshotValidation),
    liveFirstInstanceValidationSemanticSha256(validation),
    'live render parity snapshot validation semantics', reasons);
  return uniqueReasons(reasons);
}

function validateTimestampPoolStatic(pool, label, reasons) {
  if (!isRecord(pool)) {
    reasons.push(`${label} timestamp-pool evidence is missing`);
    return;
  }
  for (const field of [
    'poolIdentity', 'querySetIdentity', 'resolveBufferIdentity', 'resultBufferIdentity',
  ]) requireCondition(isIdentity(pool[field]), `${label} ${field} is missing`, reasons);
  requireEqual(pool.maxQueries, 2048, `${label} maxQueries`, reasons);
  requireEqual(pool.isDisposed, false, `${label} isDisposed`, reasons);
}

const TIMESTAMP_POOL_STATIC_FIELDS = Object.freeze([
  'poolIdentity', 'querySetIdentity', 'resolveBufferIdentity',
  'resultBufferIdentity', 'maxQueries', 'isDisposed',
]);

function validateTimestampPoolState(pool, label, {
  expectedFrameCount,
} = {}, reasons) {
  validateTimestampPoolStatic(pool, label, reasons);
  requireEqual(pool?.currentQueryIndex, 0, `${label} currentQueryIndex`, reasons);
  requireEqual(pool?.queryOffsetCount, 0, `${label} queryOffsetCount`, reasons);
  requireCondition(Array.isArray(pool?.queryOffsetUids)
    && pool.queryOffsetUids.length === 0, `${label} queryOffsetUids is not empty`, reasons);
  requireEqual(pool?.frameCount, expectedFrameCount, `${label} frameCount`, reasons);
  requireCondition(Array.isArray(pool?.frames)
    && pool.frames.length === expectedFrameCount
    && new Set(pool.frames).size === expectedFrameCount,
  `${label} resolved frame accounting is invalid`, reasons);
  requireCondition(isPositiveInteger(pool?.timestampUidCount),
    `${label} timestampUidCount is invalid`, reasons);
  requireCondition(Array.isArray(pool?.timestampUids)
    && pool.timestampUids.length === pool.timestampUidCount
    && new Set(pool.timestampUids).size === pool.timestampUidCount,
  `${label} resolved UID accounting is invalid`, reasons);
  requireEqual(pool?.pendingResolve, false, `${label} pendingResolve`, reasons);
  requireEqual(pool?.isDisposed, false, `${label} isDisposed`, reasons);
  requireEqual(pool?.resultBufferMapState, 'unmapped', `${label} map state`, reasons);
}

function validateRendererMemoryCounters(memory, label, reasons) {
  requireExactArray(Object.keys(memory ?? {}).sort(), [...RENDERER_MEMORY_FIELDS].sort(),
    `${label} renderer-memory fields`, reasons);
  for (const field of RENDERER_MEMORY_FIELDS) {
    requireCondition(isNonnegativeInteger(memory?.[field]),
      `${label} renderer-memory ${field} is invalid`, reasons);
  }
}

function validatePinnedRectangle(value, label, reasons) {
  requireEqual(value?.x, 0, `${label} x`, reasons);
  requireEqual(value?.y, 0, `${label} y`, reasons);
  requireEqual(value?.width, PARITY_WIDTH, `${label} width`, reasons);
  requireEqual(value?.height, PARITY_HEIGHT, `${label} height`, reasons);
}

function validatePinnedViewport(value, label, reasons) {
  validatePinnedRectangle(value, label, reasons);
  requireEqual(value?.minDepth, 0, `${label} minDepth`, reasons);
  requireEqual(value?.maxDepth, 1, `${label} maxDepth`, reasons);
}

function validatePinnedViewportState(state, label, renderTargetTextureUuid, reasons) {
  requireEqual(state?.schemaVersion, 1, `${label} schemaVersion`, reasons);
  requireEqual(state?.kind, 'three-r185-live-first-instance-viewport-state',
    `${label} kind`, reasons);
  validatePinnedViewport(state?.renderer?.viewport, `${label} renderer viewport`, reasons);
  validatePinnedRectangle(state?.renderer?.scissor, `${label} renderer scissor`, reasons);
  requireEqual(state?.renderer?.scissorTest, false,
    `${label} renderer scissorTest`, reasons);
  requireEqual(state?.renderer?.activeRenderTargetTextureUuid, null,
    `${label} active render target`, reasons);
  requireEqual(state?.renderTarget?.textureUuid, renderTargetTextureUuid,
    `${label} render-target identity`, reasons);
  requireEqual(state?.renderTarget?.width, PARITY_WIDTH,
    `${label} render-target width`, reasons);
  requireEqual(state?.renderTarget?.height, PARITY_HEIGHT,
    `${label} render-target height`, reasons);
  validatePinnedViewport(state?.renderTarget?.viewport,
    `${label} render-target viewport`, reasons);
  validatePinnedRectangle(state?.renderTarget?.scissor,
    `${label} render-target scissor`, reasons);
  requireEqual(state?.renderTarget?.scissorTest, false,
    `${label} render-target scissorTest`, reasons);
}

export function validateLiveFirstInstanceCompletionInvariant(invariant, {
  spec,
  validation,
} = {}) {
  const reasons = [];
  if (!isRecord(invariant)) return ['live first-instance completion invariant is missing'];
  requireEqual(invariant.pass, true, 'live completion invariant pass', reasons);
  requireEqual(invariant.kind, 'first-instance-live-crossover-static-resource-invariant',
    'live completion invariant kind', reasons);
  for (const [field, expected] of Object.entries({
    lanesPrimed: true,
    allBundlesStatic: true,
    commandBuffersDistinct: true,
    commandBuffersZeroOffset: true,
    commandBuffersExact: true,
    lanePhysicalOrderExact: true,
    lifecycleExact: true,
    timestampPreprimeExact: true,
    timestampPoolsExact: true,
    computeProgramEntriesExact: true,
    rendererMemoryExact: true,
    viewportStateExact: true,
  })) requireEqual(invariant[field], expected, `live completion ${field}`, reasons);
  requireExactArray(invariant.observedLanePhysicalOrder, spec?.lanePhysicalOrder ?? [],
    'live completion observed lane physical order', reasons);
  requireEqual(invariant.plannedLanePhysicalOrder, spec?.lanePhysicalOrder?.join('|'),
    'live completion planned lane physical order', reasons);
  for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    requireEqual(invariant.bundleStaticFlags?.[lane], true,
      `live completion ${lane} bundle static`, reasons);
    requireEqual(invariant.bundleRecordCounts?.[lane], 1,
      `live completion ${lane} bundle record count`, reasons);
    validateCommandBuffer(invariant.commandBufferCommitments?.[lane], lane, spec, reasons);
    requireEqual(sha256Json(invariant.commandBufferCommitments?.[lane]),
      sha256Json(validation?.commandBufferCommitments?.[lane]),
      `live completion ${lane} command-buffer commitment`, reasons);
  }
  requireEqual(sha256Json(invariant.staticLifecycleAtTimingStart),
    sha256Json(invariant.staticLifecycleAtTimingEnd),
    'live completion static lifecycle equality', reasons);
  requireEqual(sha256Json(invariant.staticLifecycleAtTimingStart),
    sha256Json(staticLifecycle(validation?.lifecycle)),
    'live completion lifecycle versus validation', reasons);
  requireFNV64(invariant.lifecycleCommitmentAtTimingStart,
    'live completion lifecycle start commitment', reasons);
  requireEqual(invariant.lifecycleCommitmentAtTimingStart,
    invariant.lifecycleCommitmentAtTimingEnd,
    'live completion lifecycle commitment stability', reasons);
  requireEqual(invariant.lifecycleCommitmentAtTimingStart,
    fnv1a64Text(JSON.stringify(invariant.staticLifecycleAtTimingStart)),
    'live completion lifecycle commitment recomputation', reasons);
  const serials = [
    ['selectorWriteSerial', 'selectorWritesDuringTiming'],
    ['strategySelectionSerial', 'strategySelectionsDuringTiming'],
    ['computeCallSerial', 'computeCallsDuringTiming'],
    ['renderCallSerial', 'renderCallsDuringTiming'],
  ];
  for (const [prefix, countField] of serials) {
    requireCondition(isNonnegativeInteger(invariant[`${prefix}AtTimingStart`]),
      `live completion ${prefix} start is invalid`, reasons);
    requireCondition(isNonnegativeInteger(invariant[`${prefix}AtTimingEnd`]),
      `live completion ${prefix} end is invalid`, reasons);
    requireEqual(invariant[countField], TIMED_FRAME_COUNT,
      `live completion ${countField}`, reasons);
    requireEqual(invariant[`${prefix}AtTimingEnd`]
      - invariant[`${prefix}AtTimingStart`], TIMED_FRAME_COUNT,
    `live completion ${prefix} serial delta`, reasons);
  }
  requireEqual(invariant.expectedTimedFrameCount, TIMED_FRAME_COUNT,
    'live completion expectedTimedFrameCount', reasons);
  for (const field of [
    'renderTargetTextureUuid', 'renderTargetWidth', 'renderTargetHeight',
    'renderTargetSamples', 'renderTargetDepthBuffer', 'cameraViewFnv64',
    'cameraProjectionFnv64', 'totalPipelineCacheEntries',
    'computePipelineCacheEntries', 'computeProgramEntries',
  ]) requireEqual(invariant[`${field}AtTimingStart`], invariant[`${field}AtTimingEnd`],
    `live completion stable ${field}`, reasons);
  requireCondition(isIdentity(invariant.renderTargetTextureUuidAtTimingStart),
    'live completion render-target identity is missing', reasons);
  requireEqual(invariant.renderTargetWidthAtTimingStart, PARITY_WIDTH,
    'live completion render-target width', reasons);
  requireEqual(invariant.renderTargetHeightAtTimingStart, PARITY_HEIGHT,
    'live completion render-target height', reasons);
  requireEqual(invariant.renderTargetSamplesAtTimingStart, 0,
    'live completion render-target samples', reasons);
  requireEqual(invariant.renderTargetDepthBufferAtTimingStart, true,
    'live completion render-target depth buffer', reasons);
  requireFNV64(invariant.cameraViewFnv64AtTimingStart,
    'live completion camera view digest', reasons);
  requireFNV64(invariant.cameraProjectionFnv64AtTimingStart,
    'live completion camera projection digest', reasons);
  requireCondition(isNonnegativeInteger(invariant.totalPipelineCacheEntriesAtTimingStart),
    'live completion total pipeline-cache count is invalid', reasons);
  requireCondition(isNonnegativeInteger(invariant.computePipelineCacheEntriesAtTimingStart),
    'live completion compute pipeline-cache count is invalid', reasons);
  requireCondition(isNonnegativeInteger(invariant.computeProgramEntriesAtTimingStart),
    'live completion compute program count is invalid', reasons);
  validateRendererMemoryCounters(
    invariant.rendererMemoryAtTimingStart,
    'live completion timing-start',
    reasons,
  );
  validateRendererMemoryCounters(
    invariant.rendererMemoryAtTimingEnd,
    'live completion timing-end',
    reasons,
  );
  requireEqual(sha256Json(invariant.rendererMemoryAtTimingStart),
    sha256Json(invariant.rendererMemoryAtTimingEnd),
    'live completion stable renderer memory', reasons);
  validatePinnedViewportState(
    invariant.viewportStateAtTimingStart,
    'live completion timing-start viewport state',
    invariant.renderTargetTextureUuidAtTimingStart,
    reasons,
  );
  validatePinnedViewportState(
    invariant.viewportStateAtTimingEnd,
    'live completion timing-end viewport state',
    invariant.renderTargetTextureUuidAtTimingStart,
    reasons,
  );
  requireEqual(sha256Json(invariant.viewportStateAtTimingStart),
    sha256Json(invariant.viewportStateAtTimingEnd),
    'live completion stable viewport state', reasons);

  requireEqual(invariant.timestampPreprime?.schemaVersion, 1,
    'live completion timestamp preprime schemaVersion', reasons);
  requireEqual(invariant.timestampPreprime?.kind, 'three-r185-timestamp-pool-preprime',
    'live completion timestamp preprime kind', reasons);
  for (const type of ['compute', 'render']) {
    requireEqual(invariant.timestampPreprime?.addedTimestampUidCount?.[type], 1,
      `live completion preprime ${type} added UID count`, reasons);
    requireEqual(invariant.timestampPreprime?.after?.[type]?.currentQueryIndex, 0,
      `live completion preprime ${type} reset index`, reasons);
    requireEqual(invariant.timestampPreprime?.after?.[type]?.queryOffsetCount, 0,
      `live completion preprime ${type} query-offset count`, reasons);
    requireEqual(invariant.timestampPreprime?.after?.[type]?.frameCount, 1,
      `live completion preprime ${type} frame count`, reasons);
    requireCondition(isPositiveInteger(
      invariant.timestampPreprime?.after?.[type]?.timestampUidCount,
    ), `live completion preprime ${type} resolved UID count is invalid`, reasons);
    requireEqual(invariant.timestampPreprime?.after?.[type]?.pendingResolve, false,
      `live completion preprime ${type} pendingResolve`, reasons);
    requireEqual(invariant.timestampPreprime?.after?.[type]?.isDisposed, false,
      `live completion preprime ${type} isDisposed`, reasons);
    requireEqual(invariant.timestampPreprime?.after?.[type]?.resultBufferMapState,
      'unmapped', `live completion preprime ${type} map state`, reasons);
    const startPool = invariant.timestampPoolsAtTimingStart?.[type];
    const endPool = invariant.timestampPoolsAtTimingEnd?.[type];
    requireEqual(sha256Json(invariant.timestampPreprime?.after?.[type]),
      sha256Json(startPool),
      `live completion preprime/timing-start ${type} timestamp pool identity`,
      reasons);
    validateTimestampPoolState(startPool, `live completion timing-start ${type}`, {
      expectedFrameCount: 1,
    }, reasons);
    validateTimestampPoolState(endPool, `live completion timing-end ${type}`, {
      expectedFrameCount: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    }, reasons);
    for (const field of TIMESTAMP_POOL_STATIC_FIELDS) {
      requireEqual(startPool?.[field], endPool?.[field],
        `live completion stable ${type} timestamp pool ${field}`, reasons);
    }
    requireEqual(endPool?.timestampUidCount - startPool?.timestampUidCount,
      TIMED_FRAME_COUNT, `live completion ${type} timestamp UID delta`, reasons);
  }
  return uniqueReasons(reasons);
}

function parseOrder(value) {
  if (Array.isArray(value)) return [...value];
  return typeof value === 'string' ? value.split('|') : null;
}

function parseNumberOrder(value) {
  const values = parseOrder(value);
  return values?.map(Number) ?? null;
}

const TIMESTAMP_RECORD_KEYS = Object.freeze([
  'uid', 'type', 'callIndex', 'contextId', 'frameId', 'durationMs',
]);
const WARMUP_EVENT_KEYS = Object.freeze([
  'schemaVersion', 'kind', 'phase', 'frameIndex', 'warmupFrameIndex',
  'phaseFrameIndex', 'crossoverBlockIndex', 'withinBlockPosition',
  'crossoverPattern', 'crossoverPatternIndex', 'previousPreviousLaneId',
  'previousLaneId', 'laneId', 'commandBufferId', 'submittedComputeLaneId',
  'computeTimestampContextId', 'computeGroupIdentity',
  'computeTimestampRegistrationSerial', 'computeTimestampBackendIdentity',
  'computeTimestampBackendWrapperIdentity', 'submittedComputeNodeIds', 'gpuFrameId',
  'selectorWriteSerial', 'strategySelectionSerial', 'strategyComputeCallSerial',
  'computeCallSerial', 'computeFrameCallIndex', 'renderCallSerial',
  'renderFrameCallIndex', 'gpuComputeMs', 'gpuComputeTimestampUids',
  'gpuComputeTimestampRecords', 'gpuRenderMs', 'gpuRenderTimestampUids',
  'gpuRenderTimestampRecords', 'gpuPassTotalMs',
]);

function parseCanonicalJsonArray(value, label, reasons) {
  if (typeof value !== 'string') {
    reasons.push(`${label} is not canonical JSON text`);
    return [];
  }
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed) || JSON.stringify(parsed) !== value) {
      reasons.push(`${label} is not a canonical JSON array`);
      return [];
    }
    return parsed;
  } catch {
    reasons.push(`${label} is not valid JSON`);
    return [];
  }
}

function validateTimestampRecord(record, {
  type,
  frameId,
  callIndex,
  contextId = undefined,
  durationMs,
  label,
}, reasons) {
  requireExactObjectKeys(record, TIMESTAMP_RECORD_KEYS, label, reasons);
  requireEqual(record?.type, type, `${label} type`, reasons);
  requireCondition(isPositiveInteger(record?.callIndex),
    `${label} callIndex is invalid`, reasons);
  requireEqual(record?.callIndex, callIndex, `${label} callIndex`, reasons);
  requireCondition(isNonnegativeInteger(record?.contextId),
    `${label} contextId is invalid`, reasons);
  if (contextId !== undefined) {
    requireEqual(record?.contextId, contextId, `${label} contextId`, reasons);
  }
  requireEqual(record?.frameId, frameId, `${label} frameId`, reasons);
  requireCondition(Number.isFinite(record?.durationMs) && record.durationMs > 0,
    `${label} duration is not strictly positive`, reasons);
  requireEqual(record?.durationMs, durationMs, `${label} duration`, reasons);
  const prefix = type === 'compute' ? 'c' : 'r';
  requireEqual(
    record?.uid,
    `${prefix}:${record?.callIndex}:${record?.contextId}:f${record?.frameId}`,
    `${label} exact UID`,
    reasons,
  );
}

function validateTimestampPhasePool(pool, {
  type,
  frameCount,
  expectedFrames,
  expectedRecords,
  label,
}, reasons) {
  requireExactObjectKeys(pool, ['type', 'included', 'frames', 'uidRecords', 'resolution'],
    label, reasons);
  requireEqual(pool?.type, type, `${label} type`, reasons);
  requireEqual(pool?.included, true, `${label} included`, reasons);
  requireExactArray(pool?.frames, expectedFrames, `${label} frames`, reasons);
  requireEqual(sha256Json(pool?.uidRecords), sha256Json(expectedRecords),
    `${label} UID records`, reasons);
  const resolution = pool?.resolution;
  requireExactObjectKeys(resolution, [
    'quantumNs', 'classification', 'recordCount',
    'positiveDurationCount', 'nonpositiveDurationCount',
  ], `${label} resolution`, reasons);
  requireEqual(resolution?.recordCount, frameCount, `${label} record count`, reasons);
  requireEqual(resolution?.positiveDurationCount, frameCount,
    `${label} positive-duration count`, reasons);
  requireEqual(resolution?.nonpositiveDurationCount, 0,
    `${label} nonpositive-duration count`, reasons);
  requireEqual(resolution?.classification, 'fine', `${label} classification`, reasons);
  requireCondition(Number.isFinite(resolution?.quantumNs)
    && resolution.quantumNs > 0
    && resolution.quantumNs <= 1_000,
  `${label} timestamp quantum exceeds 1,000 ns or is invalid`, reasons);
  const durationNanoseconds = Array.isArray(expectedRecords)
    ? expectedRecords.map((record) => Math.round(record?.durationMs * 1e6))
    : [];
  const gcd = (left, right) => {
    let a = Math.abs(left);
    let b = Math.abs(right);
    while (b !== 0) [a, b] = [b, a % b];
    return a;
  };
  const reconstructedQuantumNs = durationNanoseconds.length === 0
    || durationNanoseconds.some((value) => !Number.isSafeInteger(value) || value <= 0)
    ? null
    : durationNanoseconds.reduce(gcd);
  requireEqual(resolution?.quantumNs, reconstructedQuantumNs,
    `${label} reconstructed timestamp quantum`, reasons);
}

function validateTimestampPhase(phase, {
  phaseName,
  frameCount,
  expectedFrames,
  expectedRecordsByType,
}, reasons) {
  requireExactObjectKeys(phase, [
    'schemaVersion', 'kind', 'phase', 'includedTypes', 'strictUidGrammar', 'pools',
  ], `live ${phaseName} timestamp phase`, reasons);
  requireEqual(phase?.schemaVersion, 1,
    `live ${phaseName} timestamp phase schemaVersion`, reasons);
  requireEqual(phase?.kind, 'three-r185-timestamp-phase-result',
    `live ${phaseName} timestamp phase kind`, reasons);
  requireEqual(phase?.phase, phaseName, `live ${phaseName} timestamp phase name`, reasons);
  requireExactArray(phase?.includedTypes, ['render', 'compute'],
    `live ${phaseName} timestamp types`, reasons);
  requireEqual(phase?.strictUidGrammar, true,
    `live ${phaseName} strict timestamp UID grammar`, reasons);
  requireExactObjectKeys(phase?.pools, ['render', 'compute'],
    `live ${phaseName} timestamp pools`, reasons);
  for (const type of ['render', 'compute']) {
    validateTimestampPhasePool(phase?.pools?.[type], {
      type,
      frameCount,
      expectedFrames,
      expectedRecords: expectedRecordsByType[type],
      label: `live ${phaseName} ${type} timestamp pool`,
    }, reasons);
  }
}

function validateComputeSubmissionEvidence(event, lane, validation, label, reasons) {
  const registration = validation?.lifecycle?.computeTimestampRegistrations?.[lane];
  requireEqual(event?.submittedComputeLaneId, lane,
    `${label} submitted compute lane`, reasons);
  requireEqual(event?.computeTimestampContextId, registration?.timestampContextId,
    `${label} compute timestamp context`, reasons);
  requireEqual(event?.computeGroupIdentity, registration?.computeGroupIdentity,
    `${label} compute group identity`, reasons);
  requireEqual(event?.computeTimestampRegistrationSerial, registration?.registrationSerial,
    `${label} compute timestamp registration serial`, reasons);
  requireEqual(event?.computeTimestampBackendIdentity, registration?.backendIdentity,
    `${label} compute timestamp backend identity`, reasons);
  requireEqual(event?.computeTimestampBackendWrapperIdentity,
    registration?.backendWrapperIdentity,
    `${label} compute timestamp wrapper identity`, reasons);
  const nodeIds = Array.isArray(event?.submittedComputeNodeIds)
    ? event.submittedComputeNodeIds
    : parseCanonicalJsonArray(event?.submittedComputeNodeIds,
      `${label} submitted compute-node IDs`, reasons);
  requireExactArray(nodeIds, registration?.computeNodeIds ?? [],
    `${label} submitted compute-node IDs`, reasons);
}

function validateCompactWarmupAudit(audit, {
  spec,
  summary,
  invariant,
  validation,
  measuredRows,
}, reasons) {
  const label = 'live compact warmup audit';
  requireExactObjectKeys(audit, [
    'schemaVersion', 'kind', 'pass', 'expectedFrameCount', 'actualStartupHistory',
    'eventsExact', 'timestampPhaseExact', 'postWarmupStateExact',
    'postWarmupState', 'events',
  ], label, reasons);
  requireEqual(audit?.schemaVersion, 1, `${label} schemaVersion`, reasons);
  requireEqual(audit?.kind, 'live-first-instance-compact-warmup-schedule-audit',
    `${label} kind`, reasons);
  requireEqual(audit?.pass, true, `${label} pass`, reasons);
  requireEqual(audit?.expectedFrameCount, FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    `${label} expected frame count`, reasons);
  for (const field of ['eventsExact', 'timestampPhaseExact', 'postWarmupStateExact']) {
    requireEqual(audit?.[field], true, `${label} ${field}`, reasons);
  }
  requireExactObjectKeys(audit?.actualStartupHistory,
    ['previousPreviousLaneId', 'previousLaneId'], `${label} startup history`, reasons);
  requireEqual(audit?.actualStartupHistory?.previousPreviousLaneId, null,
    `${label} startup previous-previous lane`, reasons);
  requireEqual(audit?.actualStartupHistory?.previousLaneId, null,
    `${label} startup previous lane`, reasons);
  requireCondition(Array.isArray(audit?.events), `${label} events are missing`, reasons);
  const events = Array.isArray(audit?.events) ? audit.events : [];
  requireEqual(events.length, FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    `${label} event count`, reasons);
  const computeRecords = [];
  const renderRecords = [];
  for (let index = 0; index < events.length; index += 1) {
    const event = events[index];
    const eventLabel = `${label} event ${index}`;
    requireExactObjectKeys(event, WARMUP_EVENT_KEYS, eventLabel, reasons);
    requireEqual(event?.schemaVersion, 1, `${eventLabel} schemaVersion`, reasons);
    requireEqual(event?.kind, 'live-first-instance-warmup-frame-event',
      `${eventLabel} kind`, reasons);
    requireEqual(event?.phase, 'warmup', `${eventLabel} phase`, reasons);
    requireEqual(event?.frameIndex, index, `${eventLabel} frameIndex`, reasons);
    requireEqual(event?.warmupFrameIndex, index, `${eventLabel} warmupFrameIndex`, reasons);
    requireEqual(event?.phaseFrameIndex, index, `${eventLabel} phaseFrameIndex`, reasons);
    const scheduled = firstInstanceLiveCrossoverFrame(
      index,
      spec?.superblockOrientationOffset,
    );
    for (const field of ['crossoverBlockIndex', 'withinBlockPosition', 'laneId']) {
      requireEqual(event?.[field], scheduled[field], `${eventLabel} ${field}`, reasons);
    }
    requireEqual(event?.crossoverPattern, scheduled.pattern,
      `${eventLabel} crossoverPattern`, reasons);
    requireEqual(event?.crossoverPatternIndex, scheduled.patternIndex,
      `${eventLabel} crossoverPatternIndex`, reasons);
    requireEqual(event?.previousLaneId, index === 0 ? null : events[index - 1]?.laneId,
      `${eventLabel} actual previous lane`, reasons);
    requireEqual(event?.previousPreviousLaneId,
      index < 2 ? null : events[index - 2]?.laneId,
      `${eventLabel} actual previous-previous lane`, reasons);
    if (index >= 2) {
      requireEqual(event?.previousLaneId, scheduled.previousLaneId,
        `${eventLabel} cyclic previous lane`, reasons);
      requireEqual(event?.previousPreviousLaneId, scheduled.previousPreviousLaneId,
        `${eventLabel} cyclic previous-previous lane`, reasons);
    }
    requireEqual(event?.commandBufferId,
      validation?.commandBufferCommitments?.[scheduled.laneId]?.attributeId,
      `${eventLabel} command-buffer identity`, reasons);
    validateComputeSubmissionEvidence(event, scheduled.laneId, validation, eventLabel, reasons);
    const ordinal = index + 1;
    for (const [field, base] of [
      ['selectorWriteSerial', invariant?.selectorWriteSerialAtTimingStart],
      ['strategySelectionSerial', invariant?.strategySelectionSerialAtTimingStart],
      ['strategyComputeCallSerial', invariant?.strategyComputeCallSerialAtTimingStart],
      ['computeCallSerial', invariant?.computeCallSerialAtTimingStart],
      ['renderCallSerial', invariant?.renderCallSerialAtTimingStart],
    ]) requireEqual(event?.[field], base + ordinal, `${eventLabel} ${field}`, reasons);
    requireEqual(event?.computeFrameCallIndex, 1,
      `${eventLabel} compute frame-call index`, reasons);
    requireEqual(event?.renderFrameCallIndex, 1,
      `${eventLabel} render frame-call index`, reasons);
    requireCondition(isNonnegativeInteger(event?.gpuFrameId),
      `${eventLabel} GPU frame is invalid`, reasons);
    if (index > 0) requireEqual(event?.gpuFrameId, events[index - 1]?.gpuFrameId + 1,
      `${eventLabel} GPU frame sequence`, reasons);
    requireCondition(Number.isFinite(event?.gpuComputeMs) && event.gpuComputeMs > 0,
      `${eventLabel} compute duration is invalid`, reasons);
    requireCondition(Number.isFinite(event?.gpuRenderMs) && event.gpuRenderMs > 0,
      `${eventLabel} render duration is invalid`, reasons);
    requireEqual(event?.gpuPassTotalMs, event?.gpuComputeMs + event?.gpuRenderMs,
      `${eventLabel} pass total`, reasons);
    requireExactArray(event?.gpuComputeTimestampUids,
      [event?.gpuComputeTimestampRecords?.[0]?.uid],
      `${eventLabel} compute timestamp UIDs`, reasons);
    requireExactArray(event?.gpuRenderTimestampUids,
      [event?.gpuRenderTimestampRecords?.[0]?.uid],
      `${eventLabel} render timestamp UIDs`, reasons);
    requireEqual(event?.gpuComputeTimestampRecords?.length, 1,
      `${eventLabel} compute timestamp record count`, reasons);
    requireEqual(event?.gpuRenderTimestampRecords?.length, 1,
      `${eventLabel} render timestamp record count`, reasons);
    validateTimestampRecord(event?.gpuComputeTimestampRecords?.[0], {
      type: 'compute',
      frameId: event?.gpuFrameId,
      callIndex: event?.computeFrameCallIndex,
      contextId: event?.computeTimestampContextId,
      durationMs: event?.gpuComputeMs,
      label: `${eventLabel} compute timestamp`,
    }, reasons);
    validateTimestampRecord(event?.gpuRenderTimestampRecords?.[0], {
      type: 'render',
      frameId: event?.gpuFrameId,
      callIndex: event?.renderFrameCallIndex,
      durationMs: event?.gpuRenderMs,
      label: `${eventLabel} render timestamp`,
    }, reasons);
    if (event?.gpuComputeTimestampRecords?.[0]) {
      computeRecords.push(event.gpuComputeTimestampRecords[0]);
    }
    if (event?.gpuRenderTimestampRecords?.[0]) {
      renderRecords.push(event.gpuRenderTimestampRecords[0]);
    }
  }
  const tail = audit?.postWarmupState;
  requireExactObjectKeys(tail, [
    'schemaVersion', 'kind', 'previousPreviousLaneId', 'previousLaneId',
    'lastWarmupGpuFrameId', 'selectorWriteSerial', 'strategySelectionSerial',
    'strategyComputeCallSerial', 'computeCallSerial', 'renderCallSerial',
  ], `${label} post-warmup state`, reasons);
  requireEqual(tail?.schemaVersion, 1, `${label} post-warmup schemaVersion`, reasons);
  requireEqual(tail?.kind, 'live-first-instance-post-warmup-state',
    `${label} post-warmup kind`, reasons);
  const last = events.at(-1);
  const penultimate = events.at(-2);
  for (const [field, expected] of [
    ['previousPreviousLaneId', penultimate?.laneId],
    ['previousLaneId', last?.laneId],
    ['lastWarmupGpuFrameId', last?.gpuFrameId],
    ['selectorWriteSerial', last?.selectorWriteSerial],
    ['strategySelectionSerial', last?.strategySelectionSerial],
    ['strategyComputeCallSerial', last?.strategyComputeCallSerial],
    ['computeCallSerial', last?.computeCallSerial],
    ['renderCallSerial', last?.renderCallSerial],
  ]) requireEqual(tail?.[field], expected, `${label} post-warmup ${field}`, reasons);
  const firstMeasured = measuredRows?.[0];
  requireEqual(firstMeasured?.previousPreviousLaneId, tail?.previousPreviousLaneId,
    `${label} measurement-boundary previous-previous lane`, reasons);
  requireEqual(firstMeasured?.previousLaneId, tail?.previousLaneId,
    `${label} measurement-boundary previous lane`, reasons);
  requireEqual(firstMeasured?.gpuFrameId, tail?.lastWarmupGpuFrameId + 1,
    `${label} measurement-boundary GPU frame`, reasons);
  validateTimestampPhase(summary?.timestampPhases?.warmup, {
    phaseName: 'warmup',
    frameCount: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    expectedFrames: events.map((event) => event.gpuFrameId),
    expectedRecordsByType: { compute: computeRecords, render: renderRecords },
  }, reasons);
}

function allRowsEqual(rows, field, expected, reasons) {
  for (let index = 0; index < rows.length; index += 1) {
    requireEqual(rows[index]?.[field], expected, `row ${index} ${field}`, reasons);
  }
}

function observedLiveFirstInstanceHistoryBalance(rows) {
  const transitionCounts = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_TRANSITION_KEYS.map((key) => [key, 0]),
  );
  const historyTripleCounts = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_HISTORY_TRIPLE_KEYS.map((key) => [key, 0]),
  );
  for (const row of Array.isArray(rows) ? rows : []) {
    const transitionKey = `${row?.previousLaneId}->${row?.laneId}`;
    const historyTripleKey =
      `${row?.previousPreviousLaneId}->${row?.previousLaneId}->${row?.laneId}`;
    if (Object.hasOwn(transitionCounts, transitionKey)) transitionCounts[transitionKey] += 1;
    if (Object.hasOwn(historyTripleCounts, historyTripleKey)) {
      historyTripleCounts[historyTripleKey] += 1;
    }
  }
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-crossover-history-balance',
    frameCount: Array.isArray(rows) ? rows.length : 0,
    expectedTransitionCountPerCell:
      FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES / 4,
    transitionCounts,
    expectedHistoryTripleCountPerCell:
      FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES / 8,
    historyTripleCounts,
  };
}

export function validateLiveFirstInstanceCrossoverRows({
  spec,
  rows,
  summary,
  validation,
  protocol,
} = {}) {
  const reasons = [];
  const scheduleSha256 = liveFirstInstanceCrossoverScheduleSha256(
    spec?.superblockOrientationOffset,
  );
  requireEqual(protocol?.schemaVersion, 2, 'live protocol schemaVersion', reasons);
  requireEqual(protocol?.warmupFrames, FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    'live protocol warmupFrames', reasons);
  requireEqual(protocol?.measuredFrames, FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    'live protocol measuredFrames', reasons);
  requireEqual(protocol?.plannedScheduleSha256, scheduleSha256,
    'live protocol schedule digest', reasons);
  requireSha256(protocol?.plannedScheduleSha256, 'live protocol schedule digest', reasons);
  requireCondition(Array.isArray(rows), 'live retained rows are missing', reasons);
  if (!Array.isArray(rows)) return uniqueReasons(reasons);
  requireEqual(rows.length, FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    'live retained row count', reasons);
  requireEqual(summary?.accepted, true, 'live timing summary accepted', reasons);
  requireEqual(summary?.timestampAvailable, true, 'live timestamp availability', reasons);
  requireEqual(summary?.rowCount, FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    'live timing summary rowCount', reasons);
  requireEqual(summary?.warmupRowCount, FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    'live timing summary warmupRowCount', reasons);
  for (const field of [
    'missingWarmupComputeFrames', 'missingWarmupRenderFrames',
    'invalidWarmupComputeTimestampUidCountFrames',
    'invalidWarmupRenderTimestampUidCountFrames',
    'invalidWarmupComputeTimestampDurationFrames',
    'invalidWarmupRenderTimestampDurationFrames',
  ]) requireEqual(summary?.[field], 0, `live ${field}`, reasons);
  requireEqual(summary?.missingComputeFrames, 0, 'live missing compute frames', reasons);
  requireEqual(summary?.missingRenderFrames, 0, 'live missing render frames', reasons);
  requireEqual(summary?.expectedComputeTimestampUidCount, 1,
    'live expected compute timestamp UID count', reasons);
  requireEqual(summary?.expectedRenderTimestampUidCount, 1,
    'live expected render timestamp UID count', reasons);
  requireEqual(summary?.invalidComputeTimestampUidCountFrames, 0,
    'live invalid compute timestamp UID frames', reasons);
  requireEqual(summary?.invalidRenderTimestampUidCountFrames, 0,
    'live invalid render timestamp UID frames', reasons);
  requireEqual(summary?.invalidComputeTimestampDurationFrames, 0,
    'live invalid compute timestamp duration frames', reasons);
  requireEqual(summary?.invalidRenderTimestampDurationFrames, 0,
    'live invalid render timestamp duration frames', reasons);
  for (const field of [
    'renderTimestampPoolQualityValid', 'computeTimestampPoolQualityValid',
    'warmupRenderTimestampPoolQualityValid', 'warmupComputeTimestampPoolQualityValid',
    'warmupTimestampFrameCountValid', 'measurementTimestampFrameCountValid',
  ]) requireEqual(summary?.[field], true, `live ${field}`, reasons);
  requireEqual(summary?.classification, 'fine', 'live timestamp classification', reasons);
  requireCondition(Number.isFinite(summary?.quantumNs)
    && summary.quantumNs > 0
    && summary.quantumNs <= 1_000,
  'live timestamp quantum exceeds 1,000 ns or is invalid', reasons);

  const invariant = summary?.completionInvariant;
  reasons.push(...validateLiveFirstInstanceCompletionInvariant(invariant, {
    spec,
    validation,
  }));
  const bases = {
    selectorWriteSerial: invariant?.selectorWriteSerialAtTimingStart,
    strategySelectionSerial: invariant?.strategySelectionSerialAtTimingStart,
    computeCallSerial: invariant?.computeCallSerialAtTimingStart,
    renderCallSerial: invariant?.renderCallSerialAtTimingStart,
  };
  const commandIds = Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
    (lane) => [lane, validation?.commandBufferCommitments?.[lane]?.attributeId],
  ));
  const seenGpuFrames = new Set();
  const measurementRecords = { compute: [], render: [] };
  validateCompactWarmupAudit(invariant?.warmupScheduleAudit, {
    spec,
    summary,
    invariant,
    validation,
    measuredRows: rows,
  }, reasons);
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const scheduled = firstInstanceLiveCrossoverFrame(
      index,
      spec?.superblockOrientationOffset,
    );
    const ordinal = FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES + index + 1;
    requireEqual(row?.frameIndex, index, `row ${index} frameIndex`, reasons);
    requireEqual(row?.phaseFrameIndex, index, `row ${index} phaseFrameIndex`, reasons);
    requireEqual(row?.crossoverBlockIndex, scheduled.crossoverBlockIndex,
      `row ${index} crossoverBlockIndex`, reasons);
    requireEqual(row?.withinBlockPosition, scheduled.withinBlockPosition,
      `row ${index} withinBlockPosition`, reasons);
    requireEqual(row?.crossoverPattern, scheduled.pattern,
      `row ${index} crossoverPattern`, reasons);
    requireEqual(row?.crossoverPatternIndex, scheduled.patternIndex,
      `row ${index} crossoverPatternIndex`, reasons);
    requireEqual(row?.previousPreviousLaneId, scheduled.previousPreviousLaneId,
      `row ${index} previousPreviousLaneId`, reasons);
    requireEqual(row?.previousLaneId, scheduled.previousLaneId,
      `row ${index} previousLaneId`, reasons);
    requireEqual(row?.laneId, scheduled.laneId, `row ${index} laneId`, reasons);
    requireEqual(row?.commandBufferId, commandIds[scheduled.laneId],
      `row ${index} commandBufferId`, reasons);
    validateComputeSubmissionEvidence(row, scheduled.laneId, validation,
      `row ${index}`, reasons);
    requireEqual(row?.portableCommandBufferIdAtTimingStart, commandIds[PORTABLE],
      `row ${index} portable command-buffer identity`, reasons);
    requireEqual(row?.featureCommandBufferIdAtTimingStart, commandIds[FEATURE],
      `row ${index} feature command-buffer identity`, reasons);
    requireEqual(row?.commandBufferCommitmentsAtTimingStart,
      JSON.stringify(validation?.commandBufferCommitments),
      `row ${index} timing-start command-buffer commitments`, reasons);
    requireEqual(row?.computeProgramEntriesAtTimingStart,
      invariant?.computeProgramEntriesAtTimingStart,
      `row ${index} timing-start compute-program count`, reasons);
    requireEqual(row?.rendererMemoryAtTimingStart,
      JSON.stringify(invariant?.rendererMemoryAtTimingStart),
      `row ${index} timing-start renderer memory`, reasons);
    requireEqual(row?.viewportStateAtTimingStart,
      JSON.stringify(invariant?.viewportStateAtTimingStart),
      `row ${index} timing-start viewport state`, reasons);
    requireCondition(commandIds[PORTABLE] !== commandIds[FEATURE],
      `row ${index} command-buffer identities are aliased`, reasons);
    for (const field of ['commandSegmentIndex', 'commandRecordBase', 'commandByteBase']) {
      requireEqual(row?.[field], 0, `row ${index} ${field}`, reasons);
    }
    for (const [field, base] of Object.entries(bases)) {
      requireEqual(row?.[`${field}AtTimingStart`], base,
        `row ${index} ${field} start`, reasons);
      requireEqual(row?.[field], base + ordinal, `row ${index} ${field}`, reasons);
      if (index > 0) requireEqual(row?.[field], rows[index - 1]?.[field] + 1,
        `row ${index} ${field} sequence`, reasons);
    }
    requireCondition(isNonnegativeInteger(row?.gpuFrameId),
      `row ${index} gpuFrameId is invalid`, reasons);
    requireCondition(!seenGpuFrames.has(row?.gpuFrameId),
      `row ${index} gpuFrameId is duplicated`, reasons);
    seenGpuFrames.add(row?.gpuFrameId);
    if (index > 0) requireEqual(row?.gpuFrameId, rows[index - 1]?.gpuFrameId + 1,
      `row ${index} gpuFrameId sequence`, reasons);
    requireEqual(row?.gpuComputeTimestampUidCount, 1,
      `row ${index} compute timestamp UID count`, reasons);
    requireEqual(row?.expectedComputeTimestampUidCount, 1,
      `row ${index} expected compute timestamp UID count`, reasons);
    requireEqual(row?.gpuRenderTimestampUidCount, 1,
      `row ${index} render timestamp UID count`, reasons);
    requireEqual(row?.expectedRenderTimestampUidCount, 1,
      `row ${index} expected render timestamp UID count`, reasons);
    requireCondition(Number.isFinite(row?.gpuComputeMs) && row.gpuComputeMs > 0,
      `row ${index} gpuComputeMs is invalid`, reasons);
    requireCondition(Number.isFinite(row?.gpuRenderMs) && row.gpuRenderMs > 0,
      `row ${index} gpuRenderMs is invalid`, reasons);
    requireCondition(Number.isFinite(row?.gpuPassTotalMs) && row.gpuPassTotalMs > 0,
      `row ${index} gpuPassTotalMs is invalid`, reasons);
    requireEqual(row?.strictTimestampUidAttribution, true,
      `row ${index} strict timestamp UID attribution`, reasons);
    requireEqual(row?.gpuComputeTimestampDurationValid, true,
      `row ${index} compute timestamp duration validity`, reasons);
    requireEqual(row?.gpuRenderTimestampDurationValid, true,
      `row ${index} render timestamp duration validity`, reasons);
    requireEqual(row?.computeFrameCallIndex, 1,
      `row ${index} compute frame-call index`, reasons);
    requireEqual(row?.renderFrameCallIndex, 1,
      `row ${index} render frame-call index`, reasons);
    const computeUids = parseCanonicalJsonArray(
      row?.gpuComputeTimestampUids,
      `row ${index} compute timestamp UIDs`,
      reasons,
    );
    const computeRecords = parseCanonicalJsonArray(
      row?.gpuComputeTimestampRecords,
      `row ${index} compute timestamp records`,
      reasons,
    );
    const renderUids = parseCanonicalJsonArray(
      row?.gpuRenderTimestampUids,
      `row ${index} render timestamp UIDs`,
      reasons,
    );
    const renderRecords = parseCanonicalJsonArray(
      row?.gpuRenderTimestampRecords,
      `row ${index} render timestamp records`,
      reasons,
    );
    requireEqual(computeRecords.length, 1,
      `row ${index} compute timestamp record count`, reasons);
    requireEqual(renderRecords.length, 1,
      `row ${index} render timestamp record count`, reasons);
    requireExactArray(computeUids, [computeRecords[0]?.uid],
      `row ${index} compute timestamp UID join`, reasons);
    requireExactArray(renderUids, [renderRecords[0]?.uid],
      `row ${index} render timestamp UID join`, reasons);
    validateTimestampRecord(computeRecords[0], {
      type: 'compute',
      frameId: row?.gpuFrameId,
      callIndex: row?.computeFrameCallIndex,
      contextId: row?.computeTimestampContextId,
      durationMs: row?.gpuComputeMs,
      label: `row ${index} compute timestamp`,
    }, reasons);
    validateTimestampRecord(renderRecords[0], {
      type: 'render',
      frameId: row?.gpuFrameId,
      callIndex: row?.renderFrameCallIndex,
      durationMs: row?.gpuRenderMs,
      label: `row ${index} render timestamp`,
    }, reasons);
    if (computeRecords[0]) measurementRecords.compute.push(computeRecords[0]);
    if (renderRecords[0]) measurementRecords.render.push(renderRecords[0]);
    requireCondition(Number.isFinite(row?.gpuComputeMs + row?.gpuRenderMs)
      && Math.abs(row.gpuPassTotalMs - (row.gpuComputeMs + row.gpuRenderMs))
        <= TOTAL_EQUALITY_TOLERANCE_MS,
    `row ${index} gpuPassTotalMs is not compute plus render`, reasons);
    requireEqual(row?.usesCompute, true, `row ${index} usesCompute`, reasons);
    requireEqual(row?.configuredDrawCommands, spec?.bucketCount,
      `row ${index} configuredDrawCommands`, reasons);
    requireEqual(row?.configuredRenderObjects, 1,
      `row ${index} configuredRenderObjects`, reasons);
    requireEqual(row?.configuredComputeDispatches, 2,
      `row ${index} configuredComputeDispatches`, reasons);
    requireEqual(row?.configuredComputeSubmissions, 1,
      `row ${index} configuredComputeSubmissions`, reasons);
    requireEqual(row?.configuredSubmittedInstances,
      validation?.lanes?.[PORTABLE]?.membership?.expectedCount,
      `row ${index} configuredSubmittedInstances`, reasons);
    requireEqual(row?.expectedVisibleCount,
      validation?.lanes?.[PORTABLE]?.membership?.expectedCount,
      `row ${index} expectedVisibleCount`, reasons);
    requireEqual(row?.validationPass, true, `row ${index} validationPass`, reasons);
    requireEqual(row?.validationKind, LIVE_FIRST_INSTANCE_VALIDATION_KIND,
      `row ${index} validationKind`, reasons);
    requireEqual(row?.timestampAvailable, true,
      `row ${index} timestampAvailable`, reasons);
    requireExactArray(parseOrder(row?.plannedLanePhysicalOrder),
      spec?.lanePhysicalOrder ?? [], `row ${index} planned lane physical order`, reasons);
    if (row?.lanePhysicalOrder !== undefined && row.lanePhysicalOrder !== null) {
      requireExactArray(parseOrder(row.lanePhysicalOrder), spec?.lanePhysicalOrder ?? [],
        `row ${index} lane physical order`, reasons);
    }
    requireExactArray(parseNumberOrder(row?.plannedVisibilityOrder),
      spec?.visibilityOrder ?? [], `row ${index} planned visibility order`, reasons);
  }
  requireExactObjectKeys(summary?.timestampPhases,
    ['schemaVersion', 'kind', 'warmup', 'measurement'],
    'live timestamp phases', reasons);
  requireEqual(summary?.timestampPhases?.schemaVersion, 1,
    'live timestamp phases schemaVersion', reasons);
  requireEqual(summary?.timestampPhases?.kind, 'three-r185-timestamp-phase-results',
    'live timestamp phases kind', reasons);
  validateTimestampPhase(summary?.timestampPhases?.measurement, {
    phaseName: 'measurement',
    frameCount: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    expectedFrames: rows.map((row) => row.gpuFrameId),
    expectedRecordsByType: measurementRecords,
  }, reasons);
  for (const type of ['render', 'compute']) {
    requireEqual(
      sha256Json(summary?.timestampResolutions?.[type]),
      sha256Json(summary?.timestampPhases?.measurement?.pools?.[type]?.resolution),
      `live ${type} timestamp resolution commitment`,
      reasons,
    );
  }
  const runId = spec?.runId ?? rows[0]?.runId;
  const common = {
    schemaVersion: 2,
    runId,
    trialId: spec?.trialId,
    planIndex: spec?.planIndex,
    repetitionIndex: spec?.repetitionIndex,
    modeOrderPosition: spec?.modeOrderPosition,
    visibilityOrderPosition: spec?.visibilityOrderPosition,
    layoutOrderPosition: spec?.layoutOrderPosition,
    plannedModeOrder: spec?.modeOrder?.join('|'),
    plannedLayoutOrder: spec?.layoutOrder?.join('|'),
    modeId: FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
    objectCount: spec?.objectCount,
    bucketCount: spec?.bucketCount,
    targetVisibilityFraction: spec?.visibilityFraction,
    scenarioLayout: spec?.layout,
    plannedLanePhysicalOrder: spec?.lanePhysicalOrder?.join('|'),
    superblockOrientationOffset: spec?.superblockOrientationOffset,
    plannedScheduleSha256: scheduleSha256,
    protocolWarmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    protocolMeasuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    validationKind: LIVE_FIRST_INSTANCE_VALIDATION_KIND,
    validationPass: true,
    timestampAvailable: true,
    strictTimestampUidAttribution: true,
    expectedComputeTimestampUidCount: 1,
    expectedRenderTimestampUidCount: 1,
  };
  for (const [field, expected] of Object.entries(common)) {
    allRowsEqual(rows, field, expected, reasons);
  }
  return uniqueReasons(reasons);
}

export function validateLiveFirstInstanceShaderObservationSequence(validations, {
  spec,
  shaderObservationChallenges,
} = {}) {
  const reasons = [];
  if (!Array.isArray(validations) || validations.length !== LIVE_SHADER_OBSERVATION_SEQUENCE.length) {
    return ['live shader observation sequence must contain exactly six parity/main captures'];
  }
  if (!Array.isArray(shaderObservationChallenges)
    || shaderObservationChallenges.length !== LIVE_SHADER_OBSERVATION_SEQUENCE.length) {
    return ['live shader observation challenge record must contain exactly six entries'];
  }
  const observations = validations.map((validation) => validation?.shaderEvidence?.observation);
  const serials = observations.map((observation) => observation?.serial);
  for (let index = 0; index < observations.length; index += 1) {
    const expected = LIVE_SHADER_OBSERVATION_SEQUENCE[index];
    const challenge = shaderObservationChallenges[index];
    validateShaderObservationRequest(
      challenge,
      spec,
      `live shader challenge ${index}`,
      reasons,
    );
    requireEqual(challenge?.phase, expected.phase,
      `live shader challenge ${index} phase`, reasons);
    requireEqual(challenge?.role, expected.role,
      `live shader challenge ${index} role`, reasons);
    requireEqual(challenge?.captureOrdinal, index + 1,
      `live shader challenge ${index} ordinal`, reasons);
    requireCondition(isPositiveInteger(serials[index]),
      `live shader observation ${index} serial is invalid`, reasons);
    requireEqual(serials[index], index + 1,
      `live shader observation ${index} exact serial`, reasons);
    requireEqual(sha256Json(observations[index]?.request), sha256Json(challenge),
      `live shader observation ${index} runner challenge`, reasons);
    requireSha256(observations[index]?.semanticSha256,
      `live shader observation ${index} semantic digest`, reasons);
    requireSha256(observations[index]?.observationSha256,
      `live shader observation ${index} observation digest`, reasons);
    if (index > 0) {
      requireEqual(observations[index]?.semanticSha256, observations[0]?.semanticSha256,
        `live shader observation ${index} semantic stability`, reasons);
      requireEqual(
        sha256Json(observations[index]?.resourceIdentitiesAtStart),
        sha256Json(observations[0]?.resourceIdentitiesAtStart),
        `live shader observation ${index} cross-phase resource identities`,
        reasons,
      );
    }
  }
  requireCondition(new Set(serials).size === observations.length,
    'live shader observation serials are not unique', reasons);
  requireCondition(new Set(observations.map(
    (observation) => observation?.observationSha256,
  )).size === observations.length,
  'live shader observation digests are repeated cached copies', reasons);
  requireCondition(new Set(shaderObservationChallenges.map(
    (challenge) => challenge?.challengeNonce,
  )).size === shaderObservationChallenges.length,
  'live shader observation challenge nonces are not unique', reasons);
  for (const [field, expected] of Object.entries({
    laneSelectionSerial: 7,
    computeCallSerial: 3,
    prepareSerial: 0,
  })) {
    requireEqual(observations[0]?.executionCountersAtStart?.[field], expected,
      `live shader observation initial ${field}`, reasons);
  }
  const transitions = [
    { laneSelectionSerial: 3, computeCallSerial: 2, prepareSerial: 2 },
    { laneSelectionSerial: 2, computeCallSerial: 2, prepareSerial: 2 },
    { laneSelectionSerial: 3, computeCallSerial: 2, prepareSerial: 2 },
    {
      laneSelectionSerial: TIMED_FRAME_COUNT + 2,
      computeCallSerial: TIMED_FRAME_COUNT + 2,
      prepareSerial: 2,
    },
    { laneSelectionSerial: 3, computeCallSerial: 2, prepareSerial: 2 },
  ];
  for (let index = 0; index < transitions.length; index += 1) {
    const current = observations[index]?.executionCountersAtStart;
    const next = observations[index + 1]?.executionCountersAtStart;
    for (const field of SHADER_OBSERVATION_COUNTER_KEYS) {
      requireEqual(next?.[field], current?.[field] + transitions[index][field],
        `live shader observation ${index}/${index + 1} ${field} transition`, reasons);
    }
  }
  return uniqueReasons(reasons);
}

export async function validateLiveFirstInstanceTrialEvidence({
  spec,
  environment,
  preflightValidation,
  preflightRenderParity,
  validation,
  renderParity,
  postflightValidation,
  postflightRenderParity,
  shaderObservationChallenges,
  rows,
  summary,
  protocol,
  scenarioManifest,
  geometryManifest,
} = {}) {
  const reasons = [];
  for (const [phase, phaseValidation, phaseParity] of [
    ['preflight', preflightValidation, preflightRenderParity],
    ['timing-start', validation, renderParity],
    ['postflight', postflightValidation, postflightRenderParity],
  ]) {
    reasons.push(...(await validateLiveFirstInstanceCrossoverValidation(phaseValidation, {
      spec,
      environment,
      scenarioManifest,
      geometryManifest,
    })).map((reason) => `${phase}: ${reason}`));
    reasons.push(...validateLiveFirstInstanceCrossoverRenderParity(phaseParity, {
      spec,
      validation: phaseValidation,
      scenarioManifest,
    }).map((reason) => `${phase}: ${reason}`));
  }
  const semantic = liveFirstInstanceValidationSemanticSha256(validation);
  requireEqual(liveFirstInstanceValidationSemanticSha256(preflightValidation), semantic,
    'preflight/timing-start live validation semantics', reasons);
  requireEqual(liveFirstInstanceValidationSemanticSha256(postflightValidation), semantic,
    'postflight/timing-start live validation semantics', reasons);
  const timedTargetTextureUuid = summary?.completionInvariant
    ?.renderTargetTextureUuidAtTimingStart;
  for (const [phase, phaseParity] of [
    ['preflight', preflightRenderParity],
    ['timing-start', renderParity],
    ['postflight', postflightRenderParity],
  ]) {
    for (const lane of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
      requireEqual(
        phaseParity?.lanes?.[lane]?.productionBundleOutput?.target?.textureUuid,
        timedTargetTextureUuid,
        `${phase} ${lane} production target versus timed target`,
        reasons,
      );
    }
  }
  reasons.push(...validateLiveFirstInstanceShaderObservationSequence([
    preflightRenderParity?.snapshotValidation,
    preflightValidation,
    renderParity?.snapshotValidation,
    validation,
    postflightRenderParity?.snapshotValidation,
    postflightValidation,
  ], { spec, shaderObservationChallenges }));
  const parityIdentity = liveFirstInstanceRenderParityIdentity(renderParity);
  requireEqual(liveFirstInstanceRenderParityIdentity(preflightRenderParity), parityIdentity,
    'preflight/timing-start live render output', reasons);
  requireEqual(liveFirstInstanceRenderParityIdentity(postflightRenderParity), parityIdentity,
    'postflight/timing-start live render output', reasons);
  reasons.push(...validateLiveFirstInstanceCrossoverRows({
    spec,
    rows,
    summary,
    validation,
    protocol,
  }));
  const rejectionReasons = uniqueReasons(reasons);
  return {
    pass: rejectionReasons.length === 0,
    rejectionReasons,
    semanticSha256: validation ? semantic : null,
    historyBalance: observedLiveFirstInstanceHistoryBalance(rows),
  };
}

export function validateLiveFirstInstanceForcedFeatureOffGate(gate) {
  const reasons = [];
  if (!isRecord(gate)) return ['forced-feature-off deployment evidence is missing'];
  requireEqual(gate.schemaVersion, 1, 'forced-feature-off schemaVersion', reasons);
  requireEqual(gate.kind, 'first-instance-live-forced-feature-off-deployment-gate',
    'forced-feature-off kind', reasons);
  requireEqual(gate.pass, true, 'forced-feature-off pass', reasons);
  requireEqual(gate.passBeforeDisposal, true, 'forced-feature-off pre-disposal pass', reasons);
  requireEqual(gate.actualFeatureAvailable, true,
    'forced-feature-off actual feature availability', reasons);
  requireEqual(gate.forcedFeatureAvailable, false,
    'forced-feature-off availability override', reasons);
  requireEqual(gate.separateDisposableRendererRequired, true,
    'forced-feature-off disposable renderer requirement', reasons);
  requireEqual(gate.timingContaminationBoundary, 'caller-owned-disposable-renderer-device',
    'forced-feature-off contamination boundary', reasons);
  requireEqual(gate.selection?.lane, PORTABLE,
    'forced-feature-off selected lane', reasons);
  requireEqual(gate.selection?.strategyId, 'fixed-slice',
    'forced-feature-off selected strategy', reasons);
  requireEqual(gate.selection?.featureAvailable, false,
    'forced-feature-off selected feature availability', reasons);
  requireEqual(gate.construction?.pass, true, 'forced-feature-off construction pass', reasons);
  requireEqual(gate.construction?.constructedLane, PORTABLE,
    'forced-feature-off constructed lane', reasons);
  requireEqual(gate.construction?.featureLaneConstructed, false,
    'forced-feature-off feature construction', reasons);
  requireEqual(gate.construction?.portableBucketBasePresent, true,
    'forced-feature-off portable bucket base', reasons);
  requireEqual(gate.construction?.geometryModes?.length, 1,
    'forced-feature-off geometry-mode count', reasons);
  requireEqual(gate.construction?.geometryModes?.[0], 'bucket-base',
    'forced-feature-off geometry mode', reasons);
  requireEqual(gate.construction?.configuredDrawCommands, 32,
    'forced-feature-off configured draw commands', reasons);
  requireEqual(gate.construction?.configuredComputeDispatches, 2,
    'forced-feature-off configured compute dispatches', reasons);
  requireEqual(gate.construction?.configuredComputeSubmissions, 1,
    'forced-feature-off configured compute submissions', reasons);
  requireEqual(gate.commands?.pass, true, 'forced-feature-off command pass', reasons);
  requireEqual(gate.commands?.commandRecordCount, 32,
    'forced-feature-off command record count', reasons);
  requireEqual(gate.commands?.drawCommandCount, 32,
    'forced-feature-off draw command count', reasons);
  requireEqual(gate.commands?.commandByteOffset, 0,
    'forced-feature-off command byte offset', reasons);
  requireEqual(gate.commands?.fifthCommandWordsAllZero, true,
    'forced-feature-off fifth command words', reasons);
  requireEqual(gate.commands?.nonzeroFirstInstanceCount, 0,
    'forced-feature-off nonzero firstInstance count', reasons);
  requireCondition(Array.isArray(gate.commands?.firstInstanceWords)
    && gate.commands.firstInstanceWords.length === 32
    && gate.commands.firstInstanceWords.every((value) => value === 0),
  'forced-feature-off firstInstance words are not exactly 32 zeros', reasons);
  requireEqual(gate.correctness?.pass, true,
    'forced-feature-off correctness pass', reasons);
  requireEqual(gate.correctness?.membership?.pass, true,
    'forced-feature-off membership pass', reasons);
  requireEqual(gate.correctness?.membershipDigests?.pass, true,
    'forced-feature-off membership digest pass', reasons);
  requireEqual(gate.correctness?.commandValidation?.pass, true,
    'forced-feature-off command validation pass', reasons);
  requireEqual(gate.correctness?.overflow, 0, 'forced-feature-off overflow', reasons);
  requireEqual(gate.address?.pass, true, 'forced-feature-off address pass', reasons);
  requireEqual(gate.output?.pass, true, 'forced-feature-off output pass', reasons);
  requireEqual(gate.disposal?.pass, true, 'forced-feature-off disposal pass', reasons);
  requireEqual(gate.disposal?.attempted, true,
    'forced-feature-off disposal attempted', reasons);
  requireEqual(gate.disposal?.rootDetached, true,
    'forced-feature-off root detached', reasons);
  requireEqual(gate.disposal?.indirectDetached, true,
    'forced-feature-off indirect detached', reasons);
  return uniqueReasons(reasons);
}

export const LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT = TIMED_FRAME_COUNT;
