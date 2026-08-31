import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  summarizeCsv,
  summarizeInput,
  validateBenchmarkPlan,
  validateProtocolMatrix,
} from '../analysis/summarize.mjs';
import { frozenCrossoverFrame } from '../src/benchmark/frozen-crossover-schedule.js';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  buildFrozenDepthCrossoverPlan,
} from '../src/benchmark/plan.js';
import {
  frozenCrossoverScheduleSha256,
  renderParityIdentity,
  validateExactValidation,
} from '../scripts/evidence-validation.mjs';

const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const VISIBILITY = 0.99;
const EXPECTED_VISIBLE_COUNT = Math.round(OBJECT_COUNT * VISIBILITY);
const LOGICAL_LANES = Object.freeze(['front-to-back', 'reverse']);
const [FRONT_ID] = FROZEN_DEPTH_CROSSOVER_LANES;
const GEOMETRY_SHA256 = '1'.repeat(64);
const HIGH_SCENARIO_SHA256 = '2'.repeat(64);
const LOW_SCENARIO_SHA256 = '3'.repeat(64);
const EXPECTED_VISIBLE_IDS_SHA256 = '7'.repeat(64);
const PHYSICAL_SEQUENCE_SHA256 = '8'.repeat(64);
const DEPTH_RANGE = Object.freeze({ near: 90, far: 190 });
const GEOMETRY_INDEX_COUNTS = Object.freeze(
  Array.from({ length: BUCKET_COUNT }, (_, bucket) => 12 + bucket * 3),
);

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sha256Json(value) {
  return sha256(JSON.stringify(value));
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

function protocol(overrides = {}) {
  return {
    matrix: 'depth-ordering-render-only-o65536-b32',
    matrixKind: 'depth-ordering-render-only',
    representationScaleRole: null,
    layouts: ['high-overlap', 'low-overlap'],
    depthBinCount: 8,
    reversedDepthBuffer: true,
    minimumStorageBuffersPerShaderStage: 8,
    renderParity: 'preflight, timing-start, and postflight paired-lane exact validation plus two stable offscreen captures per lane of rgba8 color, depth32float, and encoded object ID',
    frozenCrossover: {
      lanes: [...FROZEN_DEPTH_CROSSOVER_LANES],
      blockSize: 8,
      warmupBlocks: 40,
      measuredBlocks: 60,
      patterns: ['FRFRRFRF', 'RFRFFRFR'],
      expectedMeasuredRowsPerLane: 240,
      expectedRenderCallsPerFrame: 1,
      expectedRenderTimestampUidCount: 1,
      expectedComputeTimestampsPerFrame: 0,
      survivorBufferSegments: 2,
      survivorSegmentLength: OBJECT_COUNT,
      legalLaneBases: [0, OBJECT_COUNT],
      scheduleSha256ByOrientation: {
        0: frozenCrossoverScheduleSha256(0),
        1: frozenCrossoverScheduleSha256(1),
      },
    },
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    allowedObjectCounts: [4096, 16_384, OBJECT_COUNT],
    allowedBucketCounts: [1, 4, 32, 128],
    allowedHeterogeneousComparators: ['coalesced-v11', 'historical-v10'],
    heterogeneousComparator: null,
    modes: ['fixed-slice-depth-frozen-crossover'],
    visibilityLevels: [VISIBILITY],
    repetitions: 12,
    warmupFrames: 320,
    measuredFrames: 480,
    maximumCpuTimerQuantumMs: 0.005,
    ordering: 'twelve-repetition-paired-eight-frame-frozen-crossover-with-balanced-layout-storage-base-and-starting-orientation',
    threeBlocksScheduling: null,
    ...overrides,
  };
}

function plan(runId) {
  return buildFrozenDepthCrossoverPlan({
    runId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  }).map((trial) => ({ ...trial, runId }));
}

function visibleCounts() {
  const base = Math.floor(EXPECTED_VISIBLE_COUNT / BUCKET_COUNT);
  const counts = new Array(BUCKET_COUNT).fill(base);
  for (let bucket = 0;
    bucket < EXPECTED_VISIBLE_COUNT - base * BUCKET_COUNT;
    bucket += 1) counts[bucket] += 1;
  return counts;
}

function membership(counts) {
  return {
    pass: true,
    expectedCount: EXPECTED_VISIBLE_COUNT,
    listedCount: EXPECTED_VISIBLE_COUNT,
    duplicateIds: 0,
    outOfRangeIds: 0,
    wrongBucketIds: 0,
    listedHiddenIds: 0,
    missingVisibleIds: 0,
    overflow: 0,
    errors: 0,
  };
}

function membershipDigests(counts) {
  return {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    encoding: 'sorted-uint32-little-endian',
    pass: true,
    invalidExpectedIds: 0,
    truncatedActualIds: 0,
    expected: { count: EXPECTED_VISIBLE_COUNT, sha256: EXPECTED_VISIBLE_IDS_SHA256 },
    actual: { count: EXPECTED_VISIBLE_COUNT, sha256: EXPECTED_VISIBLE_IDS_SHA256 },
    perBucket: counts.map((count, bucket) => {
      const digest = ((bucket % 15) + 1).toString(16).repeat(64);
      return {
        bucket,
        match: true,
        expected: { count, sha256: digest },
        actual: { count, declaredCount: count, sha256: digest },
      };
    }),
  };
}

function commandValidation(counts) {
  let firstIndex = 0;
  const records = counts.map((count, bucket) => {
    const command = {
      indexCount: GEOMETRY_INDEX_COUNTS[bucket],
      instanceCount: count,
      firstIndex,
      baseVertex: 0,
      firstInstance: 0,
    };
    firstIndex += GEOMETRY_INDEX_COUNTS[bucket];
    return { bucket, actual: { ...command }, expected: { ...command } };
  });
  return {
    pass: true,
    errors: [],
    commandCount: BUCKET_COUNT,
    totalInstanceCount: EXPECTED_VISIBLE_COUNT,
    records,
  };
}

function depthBins(order, counts) {
  const traversal = order === 'front-to-back'
    ? [0, 1, 2, 3, 4, 5, 6, 7]
    : [7, 6, 5, 4, 3, 2, 1, 0];
  const expectedCounts = new Array(BUCKET_COUNT * 8).fill(0);
  const starts = new Array(BUCKET_COUNT * 8).fill(0);
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    expectedCounts[bucket * 8] = counts[bucket];
    let cursor = 0;
    for (const physicalBin of traversal) {
      starts[bucket * 8 + physicalBin] = cursor;
      cursor += expectedCounts[bucket * 8 + physicalBin];
    }
  }
  return {
    pass: true,
    errors: [],
    binCount: 8,
    order,
    traversal,
    expectedCounts,
    actualCounts: [...expectedCounts],
    expectedStarts: starts,
    actualStarts: [...starts],
    expectedBucketTotals: [...counts],
    commandCounts: [...counts],
    physicalBinSequenceCommitment: {
      schemaVersion: 1,
      hashAlgorithm: 'sha256',
      encoding: 'bucket-major-physical-bin-major-tagged-uint32-little-endian',
      bucketCount: BUCKET_COUNT,
      binCount: 8,
      recordCount: BUCKET_COUNT * 8,
      survivorCount: EXPECTED_VISIBLE_COUNT,
      sha256: PHYSICAL_SEQUENCE_SHA256,
    },
  };
}

function frozenValidation(trial) {
  const counts = visibleCounts();
  const frontAtZero = trial.laneStorageOrder[0] === FRONT_ID;
  const laneStorageOrder = frontAtZero ? 'front-to-back' : 'reverse';
  const laneOffsets = frontAtZero
    ? { 'front-to-back': 0, reverse: OBJECT_COUNT }
    : { 'front-to-back': OBJECT_COUNT, reverse: 0 };
  const identity = {
    bundleGroupUuid: `bundle-${trial.planIndex}`,
    meshUuid: `mesh-${trial.planIndex}`,
    geometryUuid: `geometry-${trial.planIndex}`,
    materialUuid: `material-${trial.planIndex}`,
    matrixAttributeId: trial.planIndex * 10 + 1,
    visibleIdsAttributeId: trial.planIndex * 10 + 2,
    indirectAttributeId: trial.planIndex * 10 + 3,
    selectorChallengeAttributeId: trial.planIndex * 10 + 4,
    bundleGroupVersion: 0,
    matrixAttributeVersion: 0,
    visibleIdsAttributeVersion: 0,
    indirectAttributeVersion: 0,
    selectorUniformUuid: `selector-${trial.planIndex}`,
  };
  const representation = {
    kind: 'single-render-object-frozen-depth-crossover',
    laneStorageOrder,
    laneOffsets: { ...laneOffsets },
    activeLane: 'front-to-back',
    activeVisibleIdOffset: laneOffsets['front-to-back'],
    visibleIdsCount: OBJECT_COUNT * 2,
    visibleIdSegmentLength: OBJECT_COUNT,
    depthBinCount: 8,
    bundleGroupStatic: true,
    bundleRecordCallbackCount: 1,
    meshCount: 1,
    geometryIdentityCount: 1,
    materialIdentityCount: 1,
    commandCount: BUCKET_COUNT,
    zeroFirstInstanceCount: BUCKET_COUNT,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    diagnosticSelectorDispatchesPerValidation: 2,
    ...identity,
  };
  const lifecycle = {
    kind: 'frozen-depth-crossover-static-bundle-lifecycle',
    laneStorageOrder,
    laneOffsets: { ...laneOffsets },
    activeLane: 'front-to-back',
    activeVisibleIdOffset: laneOffsets['front-to-back'],
    bundleGroupStatic: true,
    bundleRecordCallbackCount: 1,
    meshCount: 1,
    geometryIdentityCount: 1,
    materialIdentityCount: 1,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    ...identity,
  };
  const commands = commandValidation(counts);
  const lanes = Object.fromEntries(LOGICAL_LANES.map((logicalLane, laneIndex) => {
    const segmentSha256 = (laneIndex === 0 ? 'b' : 'c').repeat(64);
    return [logicalLane, {
      pass: true,
      order: logicalLane,
      storageOffset: laneOffsets[logicalLane],
      traversal: logicalLane === 'front-to-back'
        ? [0, 1, 2, 3, 4, 5, 6, 7]
        : [7, 6, 5, 4, 3, 2, 1, 0],
      membership: membership(counts),
      membershipDigests: membershipDigests(counts),
      depthBins: depthBins(logicalLane, counts),
      storageSegmentSha256: segmentSha256,
      paddingSentinelCount: OBJECT_COUNT - EXPECTED_VISIBLE_COUNT,
      paddingCorruptionCount: 0,
    }];
  }));
  const selectorChallenges = Object.fromEntries(LOGICAL_LANES.map((logicalLane) => [
    logicalLane,
    {
      pass: true,
      kind: 'gpu-selector-address-challenge',
      lane: logicalLane,
      storageOffset: laneOffsets[logicalLane],
      elementCount: OBJECT_COUNT,
      sha256: lanes[logicalLane].storageSegmentSha256,
      expectedSha256: lanes[logicalLane].storageSegmentSha256,
    },
  ]));
  return {
    pass: true,
    kind: 'frozen-depth-crossover-exact-paired-snapshots',
    expectedIdsMatchScenario: true,
    visibleIdsByteLength: OBJECT_COUNT * 2 * 4,
    visibleIdsSha256: (frontAtZero ? 'd' : 'e').repeat(64),
    expectedVisibleIdsSha256: (frontAtZero ? 'd' : 'e').repeat(64),
    visibleIdsExactPackingMatch: true,
    commandSha256: 'a'.repeat(64),
    laneStorageOrder,
    laneOffsets,
    activeLane: 'front-to-back',
    activeVisibleIdOffset: laneOffsets['front-to-back'],
    physicalBinSequenceCommitmentsEqual: true,
    rawLaneSequencesDiffer: true,
    physicalBinSequenceSha256: PHYSICAL_SEQUENCE_SHA256,
    commandValidation: commands,
    lanes,
    selectorChallenges,
    representation,
    lifecycle,
  };
}

function parityLane(layout) {
  const channelSha256 = (layout === 'high-overlap' ? '5' : '6').repeat(64);
  const channels = {
    color: {
      format: 'rgba8unorm',
      arrayType: 'Uint8Array',
      byteLength: 1280 * 720 * 4,
      sha256: channelSha256,
    },
    depth: {
      format: 'depth32float',
      arrayType: 'Float32Array',
      byteLength: 1280 * 720 * 4,
      sha256: channelSha256,
    },
    objectId: {
      format: 'rgba8unorm-object-id-plus-one',
      arrayType: 'Uint8Array',
      byteLength: 1280 * 720 * 4,
      sha256: channelSha256,
    },
  };
  return {
    schemaVersion: 1,
    kind: 'fixed-camera-offscreen-exact-render-parity',
    pass: true,
    width: 1280,
    height: 720,
    captures: 2,
    material: { type: 'MeshStandardNodeMaterial', color: 0x88aaff },
    ...channels,
    objectIdValidation: {
      pass: true,
      encoding: 'rgb24-object-id-plus-one-zero-background',
      backgroundPixels: 1280 * 720 - 10_000,
      coveredPixels: 10_000,
      outOfRangePixels: 0,
      nonVisiblePixels: 0,
    },
    reversedDepthBuffer: true,
    stability: {
      pass: true,
      firstCapture: structuredClone(channels),
      first: {
        colorSha256: channelSha256,
        depthSha256: channelSha256,
        objectIdSha256: channelSha256,
      },
    },
  };
}

function frozenParity(trial, validation) {
  const lane = parityLane(trial.layout);
  return {
    schemaVersion: 1,
    kind: 'frozen-depth-crossover-exact-render-parity',
    pass: true,
    laneIds: [...FROZEN_DEPTH_CROSSOVER_LANES],
    crossLaneExact: true,
    lanes: Object.fromEntries(FROZEN_DEPTH_CROSSOVER_LANES.map((laneId) => [
      laneId,
      structuredClone(lane),
    ])),
    snapshotValidation: structuredClone(validation),
  };
}

function completionInvariant(validation) {
  const lifecycle = validation.lifecycle;
  const invariant = {
    pass: true,
    kind: 'frozen-depth-crossover-static-bundle-invariant',
    bundleGroupStatic: true,
    bundleRecordCallbackCountAtTimingStart: 1,
    bundleRecordCallbackCountAtTimingEnd: 1,
    meshCount: 1,
    geometryIdentityCount: 1,
    materialIdentityCount: 1,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    frontLaneBase: validation.laneOffsets['front-to-back'],
    reverseLaneBase: validation.laneOffsets.reverse,
  };
  for (const field of [
    'bundleGroupUuid',
    'meshUuid',
    'geometryUuid',
    'materialUuid',
    'matrixAttributeId',
    'visibleIdsAttributeId',
    'indirectAttributeId',
    'selectorChallengeAttributeId',
    'bundleGroupVersion',
    'matrixAttributeVersion',
    'visibleIdsAttributeVersion',
    'indirectAttributeVersion',
    'selectorUniformUuid',
  ]) {
    invariant[`${field}AtTimingStart`] = lifecycle[field];
    invariant[`${field}AtTimingEnd`] = lifecycle[field];
  }
  const fixedPairs = {
    renderTargetTextureUuid: 'target-texture',
    renderTargetWidth: 1280,
    renderTargetHeight: 720,
    renderTargetSamples: 0,
    renderTargetDepthBuffer: true,
    cameraViewFnv64: '1'.repeat(16),
    cameraProjectionFnv64: '2'.repeat(16),
  };
  for (const [field, value] of Object.entries(fixedPairs)) {
    invariant[`${field}AtTimingStart`] = value;
    invariant[`${field}AtTimingEnd`] = value;
  }
  return {
    ...invariant,
    selectorWriteSerialAtTimingStart: 0,
    selectorWriteSerialAtTimingEnd: 800,
    selectorWritesDuringTiming: 800,
    renderCallSerialAtTimingStart: 0,
    renderCallSerialAtTimingEnd: 800,
    renderCallsDuringTiming: 800,
    computeCallSerialAtTimingStart: 6,
    computeCallSerialAtTimingEnd: 6,
    computeCallsDuringTiming: 0,
    totalPipelineCacheEntriesAtTimingStart: 5,
    totalPipelineCacheEntriesAtTimingEnd: 5,
    computePipelineCacheEntriesAtTimingStart: 1,
    computePipelineCacheEntriesAtTimingEnd: 1,
    expectedTimedFrameCount: 800,
  };
}

const FRAME_HEADERS = Object.freeze([
  'runId', 'trialId', 'planIndex', 'repetitionIndex', 'modeOrderPosition',
  'visibilityOrderPosition', 'layoutOrderPosition', 'plannedModeOrder',
  'plannedVisibilityOrder', 'plannedLayoutOrder', 'protocolWarmupFrames',
  'protocolMeasuredFrames', 'plannedLaneStorageOrder',
  'superblockOrientationOffset', 'plannedScheduleSha256', 'frameIndex',
  'phaseFrameIndex', 'crossoverBlockIndex', 'withinBlockPosition',
  'crossoverPattern', 'crossoverPatternIndex', 'laneId', 'laneBase',
  'selectorWriteSerial', 'renderCallSerial', 'gpuFrameId', 'modeId',
  'targetVisibilityFraction', 'scenarioLayout', 'depthBinRangeNear',
  'depthBinRangeFar', 'expectedVisibleCount', 'objectCount', 'bucketCount',
  'validationKind', 'validationPass', 'timestampAvailable', 'usesCompute',
  'configuredDrawCommands', 'configuredRenderObjects',
  'configuredComputeDispatches', 'configuredComputeSubmissions',
  'configuredSubmittedInstances', 'bundleRecordCallbackCountAtTimingStart',
  'frontLaneBase', 'reverseLaneBase', 'expectedRenderTimestampUidCount',
  'gpuRenderTimestampUidCount', 'gpuPassTotalMs', 'gpuComputeMs',
  'gpuRenderMs', 'cpuCommonUpdateMs', 'cpuComputeSubmitMs',
  'cpuRenderSubmitMs', 'cpuFrameBodyMs', 'cpuSubmitTotalMs',
  'selectorWriteSerialAtTimingStart', 'renderCallSerialAtTimingStart',
  'computeCallSerialAtTimingStart', 'totalPipelineCacheEntriesAtTimingStart',
  'computePipelineCacheEntriesAtTimingStart', 'bundleGroupUuidAtTimingStart',
  'meshUuidAtTimingStart', 'geometryUuidAtTimingStart',
  'materialUuidAtTimingStart', 'matrixAttributeIdAtTimingStart',
  'visibleIdsAttributeIdAtTimingStart', 'indirectAttributeIdAtTimingStart',
  'selectorChallengeAttributeIdAtTimingStart', 'bundleGroupVersionAtTimingStart',
  'matrixAttributeVersionAtTimingStart', 'visibleIdsAttributeVersionAtTimingStart',
  'indirectAttributeVersionAtTimingStart', 'selectorUniformUuidAtTimingStart',
  'renderTargetTextureUuidAtTimingStart', 'renderTargetWidthAtTimingStart',
  'renderTargetHeightAtTimingStart', 'renderTargetSamplesAtTimingStart',
  'renderTargetDepthBufferAtTimingStart', 'cameraViewFnv64AtTimingStart',
  'cameraProjectionFnv64AtTimingStart',
]);

function frameRecords(runId, benchmarkPlan) {
  let gpuFrameId = 10_000;
  return benchmarkPlan.flatMap((trial) => {
    const frontBase = trial.laneStorageOrder.indexOf(FRONT_ID) * OBJECT_COUNT;
    const reverseBase = frontBase === 0 ? OBJECT_COUNT : 0;
    const validation = frozenValidation(trial);
    const lifecycle = validation.lifecycle;
    return Array.from({ length: 480 }, (_, frameIndex) => {
      const scheduled = frozenCrossoverFrame(
        frameIndex,
        trial.superblockOrientationOffset,
      );
      const gpuRenderMs = scheduled.laneId === FRONT_ID
        && trial.layout === 'high-overlap' ? 0.8 : 1;
      gpuFrameId += 1;
      return {
        runId,
        trialId: trial.trialId,
        planIndex: trial.planIndex,
        repetitionIndex: trial.repetitionIndex,
        modeOrderPosition: 0,
        visibilityOrderPosition: 0,
        layoutOrderPosition: trial.layoutOrderPosition,
        plannedModeOrder: trial.modeOrder.join('|'),
        plannedVisibilityOrder: trial.visibilityOrder.join('|'),
        plannedLayoutOrder: trial.layoutOrder.join('|'),
        protocolWarmupFrames: 320,
        protocolMeasuredFrames: 480,
        plannedLaneStorageOrder: trial.laneStorageOrder.join('|'),
        superblockOrientationOffset: trial.superblockOrientationOffset,
        plannedScheduleSha256: frozenCrossoverScheduleSha256(
          trial.superblockOrientationOffset,
        ),
        frameIndex,
        phaseFrameIndex: frameIndex,
        crossoverBlockIndex: scheduled.crossoverBlockIndex,
        withinBlockPosition: scheduled.withinBlockPosition,
        crossoverPattern: scheduled.pattern,
        crossoverPatternIndex: scheduled.patternIndex,
        laneId: scheduled.laneId,
        laneBase: scheduled.laneId === FRONT_ID ? frontBase : reverseBase,
        selectorWriteSerial: 321 + frameIndex,
        renderCallSerial: 321 + frameIndex,
        gpuFrameId,
        modeId: trial.modeId,
        targetVisibilityFraction: trial.visibilityFraction,
        scenarioLayout: trial.layout,
        depthBinRangeNear: DEPTH_RANGE.near,
        depthBinRangeFar: DEPTH_RANGE.far,
        expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
        objectCount: OBJECT_COUNT,
        bucketCount: BUCKET_COUNT,
        validationKind: 'frozen-depth-crossover-exact-paired-snapshots',
        validationPass: true,
        timestampAvailable: true,
        usesCompute: false,
        configuredDrawCommands: BUCKET_COUNT,
        configuredRenderObjects: 1,
        configuredComputeDispatches: 0,
        configuredComputeSubmissions: 0,
        configuredSubmittedInstances: EXPECTED_VISIBLE_COUNT,
        bundleRecordCallbackCountAtTimingStart: 1,
        frontLaneBase: frontBase,
        reverseLaneBase: reverseBase,
        expectedRenderTimestampUidCount: 1,
        gpuRenderTimestampUidCount: 1,
        gpuPassTotalMs: gpuRenderMs,
        gpuComputeMs: '',
        gpuRenderMs,
        cpuCommonUpdateMs: 0.01,
        cpuComputeSubmitMs: '',
        cpuRenderSubmitMs: 0.02,
        cpuFrameBodyMs: 0.04,
        cpuSubmitTotalMs: 0.02,
        selectorWriteSerialAtTimingStart: 0,
        renderCallSerialAtTimingStart: 0,
        computeCallSerialAtTimingStart: 6,
        totalPipelineCacheEntriesAtTimingStart: 5,
        computePipelineCacheEntriesAtTimingStart: 1,
        bundleGroupUuidAtTimingStart: lifecycle.bundleGroupUuid,
        meshUuidAtTimingStart: lifecycle.meshUuid,
        geometryUuidAtTimingStart: lifecycle.geometryUuid,
        materialUuidAtTimingStart: lifecycle.materialUuid,
        matrixAttributeIdAtTimingStart: lifecycle.matrixAttributeId,
        visibleIdsAttributeIdAtTimingStart: lifecycle.visibleIdsAttributeId,
        indirectAttributeIdAtTimingStart: lifecycle.indirectAttributeId,
        selectorChallengeAttributeIdAtTimingStart:
          lifecycle.selectorChallengeAttributeId,
        bundleGroupVersionAtTimingStart: lifecycle.bundleGroupVersion,
        matrixAttributeVersionAtTimingStart: lifecycle.matrixAttributeVersion,
        visibleIdsAttributeVersionAtTimingStart: lifecycle.visibleIdsAttributeVersion,
        indirectAttributeVersionAtTimingStart: lifecycle.indirectAttributeVersion,
        selectorUniformUuidAtTimingStart: lifecycle.selectorUniformUuid,
        renderTargetTextureUuidAtTimingStart: 'target-texture',
        renderTargetWidthAtTimingStart: 1280,
        renderTargetHeightAtTimingStart: 720,
        renderTargetSamplesAtTimingStart: 0,
        renderTargetDepthBufferAtTimingStart: true,
        cameraViewFnv64AtTimingStart: '1'.repeat(16),
        cameraProjectionFnv64AtTimingStart: '2'.repeat(16),
      };
    });
  });
}

function recordsToCsv(records) {
  return [
    FRAME_HEADERS.join(','),
    ...records.map((record) => FRAME_HEADERS.map((field) => record[field] ?? '').join(',')),
  ].join('\n');
}

function scenarioSha256(layout) {
  return layout === 'high-overlap' ? HIGH_SCENARIO_SHA256 : LOW_SCENARIO_SHA256;
}

async function createRunDirectory(t, mutate = () => undefined) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'three-webgpu-frozen-analysis-'));
  t.after(async () => rm(directory, { recursive: true, force: true }));
  const runId = 'depth-ordering-render-only-o65536-b32-test';
  const benchmarkPlan = plan(runId);
  const frames = frameRecords(runId, benchmarkPlan);
  const framesCsv = recordsToCsv(frames);
  const validationArtifacts = benchmarkPlan.map((trial) => {
    const payload = frozenValidation(trial);
    const exact = validateExactValidation(payload, {
      modeId: trial.modeId,
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
      expectedVisibleIdsCanonicalSha256: EXPECTED_VISIBLE_IDS_SHA256,
      geometryManifest: {
        geometries: GEOMETRY_INDEX_COUNTS.map((count) => ({ index: { count } })),
      },
      scenarioManifest: {
        layout: trial.layout,
        visibilityFraction: VISIBILITY,
        depthBinRange: { ...DEPTH_RANGE },
        expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
        expectedVisibleIdsCanonicalSha256: EXPECTED_VISIBLE_IDS_SHA256,
      },
      laneStorageOrder: trial.laneStorageOrder,
    });
    assert.deepEqual(exact.rejectionReasons, []);
    const parity = frozenParity(trial, payload);
    const capture = {
      capturedAt: '2026-08-31T20:00:01.000Z',
      workload: {
        scenarioSeed: 123,
        geometryFixtureSha256: GEOMETRY_SHA256,
        scenarioSha256: scenarioSha256(trial.layout),
      },
      validation: {
        payloadSha256: sha256Json(payload),
        semanticSha256: exact.semanticSha256,
        payload,
      },
      renderParity: parity,
      accepted: true,
      rejectionReasons: [],
    };
    const artifact = {
      schemaVersion: 2,
      trialId: trial.trialId,
      planIndex: trial.planIndex,
      repetitionIndex: trial.repetitionIndex,
      modeId: trial.modeId,
      visibilityFraction: trial.visibilityFraction,
      layout: trial.layout,
      objectCount: trial.objectCount,
      bucketCount: trial.bucketCount,
      laneStorageOrder: [...trial.laneStorageOrder],
      superblockOrientationOffset: trial.superblockOrientationOffset,
      plannedScheduleSha256: frozenCrossoverScheduleSha256(
        trial.superblockOrientationOffset,
      ),
      selectedConfig: {
        strategyId: trial.modeId,
        objectCount: trial.objectCount,
        bucketCount: trial.bucketCount,
        visibilityFraction: trial.visibilityFraction,
        layout: trial.layout,
        laneStorageOrder: [...trial.laneStorageOrder],
        superblockOrientationOffset: trial.superblockOrientationOffset,
      },
      status: 'accepted',
      rejectionReasons: [],
      pre: structuredClone(capture),
      timingStart: structuredClone(capture),
      post: structuredClone(capture),
    };
    artifact.sha256 = sha256Json(artifact);
    return artifact;
  });
  const trialSummaries = benchmarkPlan.map((trial, index) => ({
    ...trial,
    plannedScheduleSha256: frozenCrossoverScheduleSha256(
      trial.superblockOrientationOffset,
    ),
    selectedConfig: structuredClone(validationArtifacts[index].selectedConfig),
    startedAt: '2026-08-31T20:00:00.000Z',
    completedAt: '2026-08-31T20:00:02.000Z',
    elapsedMs: 2_000,
    validation: {
      pass: true,
      kind: 'frozen-depth-crossover-exact-paired-snapshots',
      artifactSha256: validationArtifacts[index].sha256,
    },
    timestamps: {
      accepted: true,
      available: true,
      rowCount: 480,
      missingRenderFrames: 0,
      missingComputeFrames: 0,
      expectedRenderTimestampUidCount: 1,
      invalidRenderTimestampUidCountFrames: 0,
      quantumNs: 1_000,
      classification: 'fine',
    },
    completionInvariant: completionInvariant(
      validationArtifacts[index].timingStart.validation.payload,
    ),
    accepted: true,
    rejectionReasons: [],
  }));
  const parityByLayout = Object.fromEntries(['high-overlap', 'low-overlap'].map((layout) => {
    const trial = benchmarkPlan.find((entry) => entry.layout === layout);
    const payload = validationArtifacts[trial.planIndex].pre.validation.payload;
    return [`${layout}|${VISIBILITY}`, renderParityIdentity(frozenParity(trial, payload))];
  }));
  const telemetrySummary = { status: 'unavailable', reason: 'test fixture' };
  const metadata = {
    schemaVersion: 2,
    runId,
    status: 'complete',
    startedAt: '2026-08-31T20:00:00.000Z',
    completedAt: '2026-08-31T20:00:02.000Z',
    elapsedMs: 2_000,
    environment: {
      benchmarkPage: {
        reversedDepth: true,
        rendererReversedDepthBuffer: true,
        maxStorageBuffersPerShaderStage: 8,
      },
      gpuTelemetry: telemetrySummary,
    },
    evidenceStatus: 'development',
    sourceProvenance: {
      start: { status: 'unavailable' },
      end: { status: 'unavailable' },
      stable: null,
    },
    workload: {
      scenarioGenerator: 'createFixedSubsetScenario',
      scenarioSeed: 123,
      manifestArtifact: 'workload-manifests.json',
      geometryFixtureSha256: GEOMETRY_SHA256,
      scenarioSha256ByVisibility: null,
      scenarioSha256ByCell: {
        [`high-overlap|${VISIBILITY}`]: HIGH_SCENARIO_SHA256,
        [`low-overlap|${VISIBILITY}`]: LOW_SCENARIO_SHA256,
      },
      renderParitySha256ByCell: parityByLayout,
      physicalBinSequenceSha256ByPair: Object.fromEntries(benchmarkPlan.map((trial) => [
        [
          trial.repetitionIndex,
          trial.layout,
          trial.visibilityFraction,
          scenarioSha256(trial.layout),
        ].join('|'),
        PHYSICAL_SEQUENCE_SHA256,
      ])),
    },
    protocol: protocol(),
    plan: benchmarkPlan,
    expectedTrialCount: benchmarkPlan.length,
    completedTrialCount: benchmarkPlan.length,
    acceptedTrialCount: benchmarkPlan.length,
    frameRowCount: frames.length,
    validationArtifactCount: validationArtifacts.length,
    validationArtifactSha256: validationArtifacts.map((artifact) => artifact.sha256),
    pageErrors: [],
    error: null,
  };
  const workloadManifests = {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    geometryFixturesBySha256: {
      [GEOMETRY_SHA256]: {
        sha256: GEOMETRY_SHA256,
        geometries: GEOMETRY_INDEX_COUNTS.map((count) => ({ index: { count } })),
      },
    },
    scenariosBySha256: Object.fromEntries([
      ['high-overlap', HIGH_SCENARIO_SHA256],
      ['low-overlap', LOW_SCENARIO_SHA256],
    ].map(([layout, digest]) => [digest, {
      sha256: digest,
      layout,
      visibilityFraction: VISIBILITY,
      depthBinRange: { ...DEPTH_RANGE },
      expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
      expectedVisibleIdsCanonicalSha256: EXPECTED_VISIBLE_IDS_SHA256,
    }])),
    invalidObservations: [],
  };
  const fixture = {
    metadata,
    trialSummaries,
    validationArtifacts,
    workloadManifests,
    telemetrySummary,
    framesCsv,
  };
  mutate(fixture);
  for (const [index, artifact] of fixture.validationArtifacts.entries()) {
    delete artifact.sha256;
    artifact.sha256 = sha256Json(artifact);
    fixture.trialSummaries[index].validation.artifactSha256 = artifact.sha256;
  }
  fixture.metadata.validationArtifactSha256 = fixture.validationArtifacts.map(
    (artifact) => artifact.sha256,
  );
  const artifacts = new Map([
    ['frames.csv', Buffer.from(fixture.framesCsv)],
    ['metadata.json', jsonBytes(fixture.metadata)],
    ['trial-summaries.json', jsonBytes(fixture.trialSummaries)],
    ['validation-artifacts.json', jsonBytes(fixture.validationArtifacts)],
    ['workload-manifests.json', jsonBytes(fixture.workloadManifests)],
    ['gpu-telemetry-summary.json', jsonBytes(fixture.telemetrySummary)],
  ]);
  await Promise.all([...artifacts].map(([name, contents]) => (
    writeFile(path.join(directory, name), contents)
  )));
  const manifest = {
    schemaVersion: 2,
    runId,
    hashAlgorithm: 'sha256',
    requiredFiles: [...artifacts.keys()],
    optionalFiles: [{
      name: 'gpu-telemetry.csv',
      present: false,
      evidenceAvailable: false,
      absenceReason: 'test fixture has no device telemetry',
    }],
    files: [
      ...[...artifacts].map(([name, contents]) => ({
        name,
        role: `test ${name}`,
        required: true,
        present: true,
        bytes: contents.length,
        sha256: sha256(contents),
        absenceReason: null,
      })),
      {
        name: 'gpu-telemetry.csv',
        role: 'optional device telemetry samples',
        required: false,
        present: false,
        bytes: null,
        sha256: null,
        absenceReason: 'test fixture has no device telemetry',
      },
    ],
  };
  await writeFile(path.join(directory, 'artifact-manifest.json'), jsonBytes(manifest));
  return directory;
}

test('raw frozen crossover CSV routes to the strict crossover analyzer', () => {
  const benchmarkPlan = plan('frozen-raw');
  const summary = summarizeCsv(recordsToCsv(frameRecords('frozen-raw', benchmarkPlan)));
  assert.equal(summary.kind, 'frozen-depth-crossover-summary');
  assert.equal(summary.nRows, 11_520);
  assert.equal(summary.nTrials, 24);
  assert.equal(summary.preregisteredDecision.pass, true);
  assert.equal(summary.artifactVerification.status, 'unverified');
});

test('raw frozen crossover integration rejects schedule and render-only tampering', async (t) => {
  const cases = [
    ['schedule digest', (rows) => { rows[0].plannedScheduleSha256 = '0'.repeat(64); }, /schedule commitment/],
    ['timestamp multiplicity', (rows) => { rows[0].gpuRenderTimestampUidCount = 2; }, /one render timestamp UID/],
    ['compute duration', (rows) => { rows[0].gpuComputeMs = 0.1; }, /unexpected compute duration/],
    ['selector sequence', (rows) => { rows[1].selectorWriteSerial += 1; }, /frame audit fields/],
  ];
  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const rows = frameRecords('frozen-adversarial', plan('frozen-adversarial'));
      mutate(rows);
      assert.throws(() => summarizeCsv(recordsToCsv(rows)), pattern);
    });
  }
});

test('frozen protocol and plan validators bind the exact preregistered design', () => {
  const validProtocol = protocol();
  assert.doesNotThrow(() => validateProtocolMatrix(validProtocol));
  const validPlan = plan('frozen-plan');
  assert.doesNotThrow(() => validateBenchmarkPlan(validPlan, {
    runId: 'frozen-plan',
    protocol: validProtocol,
  }));

  const changedSchedule = structuredClone(validProtocol);
  changedSchedule.frozenCrossover.scheduleSha256ByOrientation[0] = 'f'.repeat(64);
  assert.throws(() => validateProtocolMatrix(changedSchedule), /schedule commitments/);

  const changedFactor = structuredClone(validPlan);
  changedFactor[0].superblockOrientationOffset = 1;
  assert.throws(
    () => validateBenchmarkPlan(changedFactor, {
      runId: 'frozen-plan',
      protocol: validProtocol,
    }),
    /preregistered frozen crossover factor/,
  );
});

test('verified frozen run binds artifacts, lifecycle, parity, and numeric gates', async (t) => {
  const directory = await createRunDirectory(t);
  const summary = await summarizeInput(directory);
  assert.equal(summary.artifactVerification.status, 'consistent');
  assert.equal(summary.artifactVerification.completedTrialCount, 24);
  assert.equal(summary.kind, 'frozen-depth-crossover-summary');
  assert.equal(summary.preregisteredDecision.pass, true);
});

test('verified frozen run rejects lifecycle, parity, and frame tampering after rehash', async (t) => {
  const cases = [
    [
      'protocol schedule commitment',
      ({ metadata }) => {
        metadata.protocol.frozenCrossover.scheduleSha256ByOrientation[0] = 'f'.repeat(64);
      },
      /schedule commitments/,
    ],
    [
      'plan crossover factor',
      ({ metadata }) => {
        metadata.plan[0].superblockOrientationOffset = 1;
      },
      /preregistered frozen crossover factor/,
    ],
    [
      'camera reversed-depth environment',
      ({ metadata }) => {
        metadata.environment.benchmarkPage.reversedDepth = false;
      },
      /does not prove camera reversed-depth operation/,
    ],
    [
      'resource identity drift',
      ({ trialSummaries }) => {
        trialSummaries[0].completionInvariant.meshUuidAtTimingEnd = 'different-mesh';
      },
      /stable meshUuid/,
    ],
    [
      'missing postflight parity',
      ({ validationArtifacts }) => {
        validationArtifacts[0].post.renderParity = null;
      },
      /post frozen render parity failed/,
    ],
    [
      'wrong planned lane base',
      (fixture) => {
        const rows = fixture.framesCsv.split('\n').map((row) => row.split(','));
        const laneBaseColumn = rows[0].indexOf('laneBase');
        rows[1][laneBaseColumn] = '1';
        fixture.framesCsv = rows.map((row) => row.join(',')).join('\n');
      },
      /laneBase/,
    ],
    [
      'row lifecycle identity drift',
      (fixture) => {
        const rows = fixture.framesCsv.split('\n').map((row) => row.split(','));
        const identityColumn = rows[0].indexOf('meshUuidAtTimingStart');
        rows[1][identityColumn] = 'tampered-mesh';
        fixture.framesCsv = rows.map((row) => row.join(',')).join('\n');
      },
      /inconsistent frame audit fields|timing-start lifecycle commitment/,
    ],
  ];
  for (const [name, mutate, pattern] of cases) {
    await t.test(name, async (subtest) => {
      const directory = await createRunDirectory(subtest, mutate);
      await assert.rejects(summarizeInput(directory), pattern);
    });
  }
});
