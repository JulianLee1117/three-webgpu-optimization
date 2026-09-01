import assert from 'node:assert/strict';
import test from 'node:test';
import {
  LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT,
  liveFirstInstanceCrossoverScheduleSha256,
  liveFirstInstanceValidationSemanticSha256,
  validateLiveFirstInstanceCrossoverRows,
  validateLiveFirstInstanceForcedFeatureOffGate,
  validateLiveFirstInstanceShaderObservationSequence,
} from '../scripts/live-first-instance-evidence-validation.mjs';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceLiveCrossoverFrame,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
} from '../src/benchmark/plan.js';

const PORTABLE = 'portable';
const FEATURE = 'feature';

function fnv1a64Text(value) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function commandBuffer(lane, attributeId) {
  return {
    lane,
    attributeId,
    attributeVersion: 0,
    byteOffset: 0,
    byteLength: 32 * 20,
    recordCount: 32,
    drawCommandCount: 32,
    firstOffset: 0,
    allOffsets: Array.from({ length: 32 }, (_, bucket) => bucket * 20),
  };
}

function rendererMemory() {
  return {
    attributes: 4,
    attributesSize: 4_096,
    geometries: 2,
    indexAttributes: 1,
    indexAttributesSize: 1_024,
    indirectStorageAttributes: 2,
    indirectStorageAttributesSize: 1_280,
    programs: 6,
    programsSize: 12_288,
    storageAttributes: 8,
    storageAttributesSize: 65_536,
    uniformBuffers: 3,
    uniformBuffersSize: 768,
  };
}

function viewportState() {
  const rectangle = { x: 0, y: 0, width: 1280, height: 720 };
  const viewport = { ...rectangle, minDepth: 0, maxDepth: 1 };
  return {
    schemaVersion: 1,
    kind: 'three-r185-live-first-instance-viewport-state',
    renderer: {
      viewport: structuredClone(viewport),
      scissor: structuredClone(rectangle),
      scissorTest: false,
      activeRenderTargetTextureUuid: null,
    },
    renderTarget: {
      textureUuid: 'target',
      width: 1280,
      height: 720,
      viewport: structuredClone(viewport),
      scissor: structuredClone(rectangle),
      scissorTest: false,
    },
  };
}

function shiftMeasurementGpuFrames(artifact, offset) {
  const shiftRecord = (record) => {
    const frameId = record.frameId + offset;
    return {
      ...record,
      uid: record.uid.replace(/:f\d+$/, `:f${frameId}`),
      frameId,
    };
  };
  for (const row of artifact.rows) {
    row.gpuFrameId += offset;
    for (const type of ['Compute', 'Render']) {
      const recordsField = `gpu${type}TimestampRecords`;
      const uidsField = `gpu${type}TimestampUids`;
      const records = JSON.parse(row[recordsField]).map(shiftRecord);
      row[recordsField] = JSON.stringify(records);
      row[uidsField] = JSON.stringify(records.map((record) => record.uid));
    }
  }
  for (const type of ['compute', 'render']) {
    const pool = artifact.summary.timestampPhases.measurement.pools[type];
    pool.frames = pool.frames.map((frameId) => frameId + offset);
    pool.uidRecords = pool.uidRecords.map(shiftRecord);
  }
  return artifact;
}

function pool(identity) {
  return {
    poolIdentity: identity,
    querySetIdentity: identity + 1,
    resolveBufferIdentity: identity + 2,
    resultBufferIdentity: identity + 3,
    maxQueries: 2048,
    isDisposed: false,
  };
}

function timingPool(identity, frameCount, timestampUidCount) {
  return {
    ...pool(identity),
    currentQueryIndex: 0,
    queryOffsetCount: 0,
    queryOffsetUids: [],
    frameCount,
    frames: Array.from({ length: frameCount }, (_, index) => index),
    timestampUidCount,
    timestampUids: Array.from(
      { length: timestampUidCount },
      (_, index) => `uid-${identity}-${index}`,
    ),
    pendingResolve: false,
    resultBufferMapState: 'unmapped',
  };
}

function createRowEvidence() {
  const spec = {
    runId: 'live-run',
    trialId: 'live-run-t01',
    planIndex: 0,
    repetitionIndex: 0,
    modeId: FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
    modeOrder: [FIRST_INSTANCE_LIVE_CROSSOVER_MODE],
    modeOrderPosition: 0,
    visibilityFraction: 0.99,
    visibilityOrder: [0.99, 0.2],
    visibilityOrderPosition: 0,
    layout: 'baseline',
    layoutOrder: ['baseline'],
    layoutOrderPosition: 0,
    lanePhysicalOrder: [PORTABLE, FEATURE],
    superblockOrientationOffset: 0,
    objectCount: 65_536,
    bucketCount: 32,
  };
  const commandBuffers = {
    [PORTABLE]: commandBuffer(PORTABLE, 101),
    [FEATURE]: commandBuffer(FEATURE, 202),
  };
  const computeTimestampContextIds = { [PORTABLE]: 41, [FEATURE]: 42 };
  const computeNodeIds = { [PORTABLE]: [301, 302], [FEATURE]: [401, 402] };
  const computeTimestampRegistrations = Object.fromEntries(
    [PORTABLE, FEATURE].map((lane, index) => [lane, {
      schemaVersion: 1,
      kind: 'live-first-instance-compute-timestamp-group-registration',
      laneId: lane,
      timestampContextId: computeTimestampContextIds[lane],
      registrationSerial: 71 + index,
      backendIdentity: 81,
      backendWrapperIdentity: 82,
      computeGroupIdentity: 91 + index,
      computeNodeIds: computeNodeIds[lane],
    }]),
  );
  const dynamicLifecycle = {
    kind: 'fixture-lifecycle',
    activeLane: FEATURE,
    activeCommandBufferId: 202,
    residentPreparedLane: FEATURE,
    laneSelectionSerial: 20,
    computeCallSerial: 30,
    prepareSerial: 6,
    stableIdentity: 'stable',
    computeNodeIds,
    computeTimestampContextIds,
    computeTimestampRegistrations,
  };
  const staticLifecycle = structuredClone(dynamicLifecycle);
  for (const field of [
    'activeLane', 'activeCommandBufferId', 'residentPreparedLane',
    'laneSelectionSerial', 'computeCallSerial', 'prepareSerial',
  ]) delete staticLifecycle[field];
  const lifecycleCommitment = fnv1a64Text(JSON.stringify(staticLifecycle));
  const validation = {
    kind: 'first-instance-live-crossover-exact-paired-snapshots',
    commandBufferCommitments: commandBuffers,
    lifecycle: dynamicLifecycle,
    lanes: {
      [PORTABLE]: { membership: { expectedCount: 64_881 } },
      [FEATURE]: { membership: { expectedCount: 64_881 } },
    },
  };
  const serialStarts = {
    selectorWriteSerial: 1_000,
    strategySelectionSerial: 2_000,
    strategyComputeCallSerial: 5_000,
    computeCallSerial: 3_000,
    renderCallSerial: 4_000,
  };
  const timestampPoolAtTimingStart = {
    render: timingPool(10, 1, 1),
    compute: timingPool(20, 1, 1),
  };
  const completionInvariant = {
    pass: true,
    kind: 'first-instance-live-crossover-static-resource-invariant',
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
    plannedLanePhysicalOrder: 'portable|feature',
    observedLanePhysicalOrder: [PORTABLE, FEATURE],
    bundleStaticFlags: { [PORTABLE]: true, [FEATURE]: true },
    bundleRecordCounts: { [PORTABLE]: 1, [FEATURE]: 1 },
    commandBufferCommitments: structuredClone(commandBuffers),
    staticLifecycleAtTimingStart: structuredClone(staticLifecycle),
    staticLifecycleAtTimingEnd: structuredClone(staticLifecycle),
    lifecycleCommitmentAtTimingStart: lifecycleCommitment,
    lifecycleCommitmentAtTimingEnd: lifecycleCommitment,
    expectedTimedFrameCount: LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT,
    renderTargetTextureUuidAtTimingStart: 'target',
    renderTargetTextureUuidAtTimingEnd: 'target',
    renderTargetWidthAtTimingStart: 1280,
    renderTargetWidthAtTimingEnd: 1280,
    renderTargetHeightAtTimingStart: 720,
    renderTargetHeightAtTimingEnd: 720,
    renderTargetSamplesAtTimingStart: 0,
    renderTargetSamplesAtTimingEnd: 0,
    renderTargetDepthBufferAtTimingStart: true,
    renderTargetDepthBufferAtTimingEnd: true,
    cameraViewFnv64AtTimingStart: '0123456789abcdef',
    cameraViewFnv64AtTimingEnd: '0123456789abcdef',
    cameraProjectionFnv64AtTimingStart: 'fedcba9876543210',
    cameraProjectionFnv64AtTimingEnd: 'fedcba9876543210',
    totalPipelineCacheEntriesAtTimingStart: 12,
    totalPipelineCacheEntriesAtTimingEnd: 12,
    computePipelineCacheEntriesAtTimingStart: 2,
    computePipelineCacheEntriesAtTimingEnd: 2,
    computeProgramEntriesAtTimingStart: 2,
    computeProgramEntriesAtTimingEnd: 2,
    rendererMemoryAtTimingStart: rendererMemory(),
    rendererMemoryAtTimingEnd: rendererMemory(),
    viewportStateAtTimingStart: viewportState(),
    viewportStateAtTimingEnd: viewportState(),
    timestampPreprime: {
      schemaVersion: 1,
      kind: 'three-r185-timestamp-pool-preprime',
      addedTimestampUidCount: { render: 1, compute: 1 },
      after: {
        render: structuredClone(timestampPoolAtTimingStart.render),
        compute: structuredClone(timestampPoolAtTimingStart.compute),
      },
    },
    timestampPoolsAtTimingStart: timestampPoolAtTimingStart,
    timestampPoolsAtTimingEnd: {
      render: timingPool(10, 480, 801),
      compute: timingPool(20, 480, 801),
    },
  };
  for (const [prefix, start] of Object.entries(serialStarts)) {
    completionInvariant[`${prefix}AtTimingStart`] = start;
    completionInvariant[`${prefix}AtTimingEnd`] = start
      + LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT;
    const countField = prefix === 'selectorWriteSerial'
      ? 'selectorWritesDuringTiming'
      : prefix === 'strategySelectionSerial'
        ? 'strategySelectionsDuringTiming'
        : prefix === 'strategyComputeCallSerial'
          ? 'strategyComputeCallsDuringTiming'
          : prefix === 'computeCallSerial'
            ? 'computeCallsDuringTiming'
            : 'renderCallsDuringTiming';
    completionInvariant[countField] = LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT;
  }
  const scheduleSha256 = liveFirstInstanceCrossoverScheduleSha256(0);
  const timestampRecord = (type, frameId, lane, durationMs) => {
    const callIndex = 1;
    const contextId = type === 'compute' ? computeTimestampContextIds[lane] : 501;
    const prefix = type === 'compute' ? 'c' : 'r';
    return {
      uid: `${prefix}:${callIndex}:${contextId}:f${frameId}`,
      type,
      callIndex,
      contextId,
      frameId,
      durationMs,
    };
  };
  const durationFor = (type, frameIndex) => (
    type === 'compute'
      ? (frameIndex % 2 === 0 ? 0.03 : 0.031)
      : (frameIndex % 2 === 0 ? 0.2 : 0.201)
  );
  const rows = Array.from(
    { length: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES },
    (_, frameIndex) => {
      const scheduled = firstInstanceLiveCrossoverFrame(frameIndex, 0);
      const ordinal = FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES + frameIndex + 1;
      const gpuFrameId = 10_000 + frameIndex;
      const gpuComputeMs = durationFor('compute', frameIndex);
      const gpuRenderMs = durationFor('render', frameIndex);
      const computeRecord = timestampRecord(
        'compute',
        gpuFrameId,
        scheduled.laneId,
        gpuComputeMs,
      );
      const renderRecord = timestampRecord(
        'render',
        gpuFrameId,
        scheduled.laneId,
        gpuRenderMs,
      );
      const registration = computeTimestampRegistrations[scheduled.laneId];
      return {
        schemaVersion: 2,
        runId: spec.runId,
        trialId: spec.trialId,
        planIndex: 0,
        repetitionIndex: 0,
        frameIndex,
        phaseFrameIndex: frameIndex,
        modeId: spec.modeId,
        modeOrderPosition: 0,
        plannedModeOrder: spec.modeId,
        targetVisibilityFraction: 0.99,
        plannedVisibilityOrder: '0.99|0.2',
        visibilityOrderPosition: 0,
        plannedLanePhysicalOrder: 'portable|feature',
        superblockOrientationOffset: 0,
        scenarioLayout: 'baseline',
        layoutOrderPosition: 0,
        plannedLayoutOrder: 'baseline',
        protocolWarmupFrames: 320,
        protocolMeasuredFrames: 480,
        plannedScheduleSha256: scheduleSha256,
        objectCount: 65_536,
        bucketCount: 32,
        expectedVisibleCount: 64_881,
        usesCompute: true,
        configuredDrawCommands: 32,
        configuredRenderObjects: 1,
        configuredComputeDispatches: 2,
        configuredComputeSubmissions: 1,
        configuredSubmittedInstances: 64_881,
        validationPass: true,
        validationKind: validation.kind,
        timestampAvailable: true,
        strictTimestampUidAttribution: true,
        gpuComputeTimestampUidCount: 1,
        gpuComputeTimestampUids: JSON.stringify([computeRecord.uid]),
        gpuComputeTimestampRecords: JSON.stringify([computeRecord]),
        gpuComputeTimestampDurationValid: true,
        expectedComputeTimestampUidCount: 1,
        gpuRenderTimestampUidCount: 1,
        gpuRenderTimestampUids: JSON.stringify([renderRecord.uid]),
        gpuRenderTimestampRecords: JSON.stringify([renderRecord]),
        gpuRenderTimestampDurationValid: true,
        expectedRenderTimestampUidCount: 1,
        gpuComputeMs,
        gpuRenderMs,
        gpuPassTotalMs: gpuComputeMs + gpuRenderMs,
        crossoverBlockIndex: scheduled.crossoverBlockIndex,
        withinBlockPosition: scheduled.withinBlockPosition,
        crossoverPattern: scheduled.pattern,
        crossoverPatternIndex: scheduled.patternIndex,
        previousPreviousLaneId: scheduled.previousPreviousLaneId,
        previousLaneId: scheduled.previousLaneId,
        laneId: scheduled.laneId,
        commandBufferId: commandBuffers[scheduled.laneId].attributeId,
        submittedComputeLaneId: scheduled.laneId,
        computeTimestampContextId: registration.timestampContextId,
        computeGroupIdentity: registration.computeGroupIdentity,
        computeTimestampRegistrationSerial: registration.registrationSerial,
        computeTimestampBackendIdentity: registration.backendIdentity,
        computeTimestampBackendWrapperIdentity: registration.backendWrapperIdentity,
        submittedComputeNodeIds: JSON.stringify(registration.computeNodeIds),
        portableCommandBufferIdAtTimingStart: commandBuffers[PORTABLE].attributeId,
        featureCommandBufferIdAtTimingStart: commandBuffers[FEATURE].attributeId,
        commandBufferCommitmentsAtTimingStart: JSON.stringify(commandBuffers),
        computeProgramEntriesAtTimingStart:
          completionInvariant.computeProgramEntriesAtTimingStart,
        rendererMemoryAtTimingStart: JSON.stringify(
          completionInvariant.rendererMemoryAtTimingStart,
        ),
        viewportStateAtTimingStart: JSON.stringify(
          completionInvariant.viewportStateAtTimingStart,
        ),
        commandSegmentIndex: 0,
        commandRecordBase: 0,
        commandByteBase: 0,
        gpuFrameId,
        computeFrameCallIndex: 1,
        renderFrameCallIndex: 1,
        ...Object.fromEntries(Object.entries(serialStarts).flatMap(([prefix, start]) => [
          [`${prefix}AtTimingStart`, start],
          [prefix, start + ordinal],
        ])),
      };
    },
  );
  const warmupEvents = Array.from(
    { length: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES },
    (_, frameIndex) => {
      const scheduled = firstInstanceLiveCrossoverFrame(frameIndex, 0);
      const gpuFrameId = 10_000 - FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES + frameIndex;
      const gpuComputeMs = durationFor('compute', frameIndex);
      const gpuRenderMs = durationFor('render', frameIndex);
      const computeRecord = timestampRecord(
        'compute', gpuFrameId, scheduled.laneId, gpuComputeMs,
      );
      const renderRecord = timestampRecord(
        'render', gpuFrameId, scheduled.laneId, gpuRenderMs,
      );
      const registration = computeTimestampRegistrations[scheduled.laneId];
      const ordinal = frameIndex + 1;
      return {
        schemaVersion: 1,
        kind: 'live-first-instance-warmup-frame-event',
        phase: 'warmup',
        frameIndex,
        warmupFrameIndex: frameIndex,
        phaseFrameIndex: frameIndex,
        crossoverBlockIndex: scheduled.crossoverBlockIndex,
        withinBlockPosition: scheduled.withinBlockPosition,
        crossoverPattern: scheduled.pattern,
        crossoverPatternIndex: scheduled.patternIndex,
        previousPreviousLaneId: frameIndex < 2
          ? null
          : firstInstanceLiveCrossoverFrame(frameIndex - 2, 0).laneId,
        previousLaneId: frameIndex === 0
          ? null
          : firstInstanceLiveCrossoverFrame(frameIndex - 1, 0).laneId,
        laneId: scheduled.laneId,
        commandBufferId: commandBuffers[scheduled.laneId].attributeId,
        submittedComputeLaneId: scheduled.laneId,
        computeTimestampContextId: registration.timestampContextId,
        computeGroupIdentity: registration.computeGroupIdentity,
        computeTimestampRegistrationSerial: registration.registrationSerial,
        computeTimestampBackendIdentity: registration.backendIdentity,
        computeTimestampBackendWrapperIdentity: registration.backendWrapperIdentity,
        submittedComputeNodeIds: structuredClone(registration.computeNodeIds),
        gpuFrameId,
        selectorWriteSerial: serialStarts.selectorWriteSerial + ordinal,
        strategySelectionSerial: serialStarts.strategySelectionSerial + ordinal,
        strategyComputeCallSerial: serialStarts.strategyComputeCallSerial + ordinal,
        computeCallSerial: serialStarts.computeCallSerial + ordinal,
        computeFrameCallIndex: 1,
        renderCallSerial: serialStarts.renderCallSerial + ordinal,
        renderFrameCallIndex: 1,
        gpuComputeMs,
        gpuComputeTimestampUids: [computeRecord.uid],
        gpuComputeTimestampRecords: [computeRecord],
        gpuRenderMs,
        gpuRenderTimestampUids: [renderRecord.uid],
        gpuRenderTimestampRecords: [renderRecord],
        gpuPassTotalMs: gpuComputeMs + gpuRenderMs,
      };
    },
  );
  const lastWarmup = warmupEvents.at(-1);
  const penultimateWarmup = warmupEvents.at(-2);
  completionInvariant.warmupScheduleAudit = {
    schemaVersion: 1,
    kind: 'live-first-instance-compact-warmup-schedule-audit',
    pass: true,
    expectedFrameCount: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    actualStartupHistory: { previousPreviousLaneId: null, previousLaneId: null },
    eventsExact: true,
    timestampPhaseExact: true,
    postWarmupStateExact: true,
    postWarmupState: {
      schemaVersion: 1,
      kind: 'live-first-instance-post-warmup-state',
      previousPreviousLaneId: penultimateWarmup.laneId,
      previousLaneId: lastWarmup.laneId,
      lastWarmupGpuFrameId: lastWarmup.gpuFrameId,
      selectorWriteSerial: lastWarmup.selectorWriteSerial,
      strategySelectionSerial: lastWarmup.strategySelectionSerial,
      strategyComputeCallSerial: lastWarmup.strategyComputeCallSerial,
      computeCallSerial: lastWarmup.computeCallSerial,
      renderCallSerial: lastWarmup.renderCallSerial,
    },
    events: warmupEvents,
  };
  const timestampPhase = (phase, events) => ({
    schemaVersion: 1,
    kind: 'three-r185-timestamp-phase-result',
    phase,
    includedTypes: ['render', 'compute'],
    strictUidGrammar: true,
    pools: Object.fromEntries(['render', 'compute'].map((type) => [type, {
      type,
      included: true,
      frames: events.map((event) => event.gpuFrameId),
      uidRecords: events.flatMap((event) => (
        type === 'compute'
          ? event.gpuComputeTimestampRecords
          : event.gpuRenderTimestampRecords
      )),
      resolution: {
        quantumNs: 1_000,
        classification: 'fine',
        recordCount: events.length,
        positiveDurationCount: events.length,
        nonpositiveDurationCount: 0,
      },
    }])),
  });
  const measurementEvents = rows.map((row) => ({
    gpuFrameId: row.gpuFrameId,
    gpuComputeTimestampRecords: JSON.parse(row.gpuComputeTimestampRecords),
    gpuRenderTimestampRecords: JSON.parse(row.gpuRenderTimestampRecords),
  }));
  const summary = {
    accepted: true,
    timestampAvailable: true,
    rowCount: 480,
    warmupRowCount: 320,
    missingWarmupComputeFrames: 0,
    missingWarmupRenderFrames: 0,
    invalidWarmupComputeTimestampUidCountFrames: 0,
    invalidWarmupRenderTimestampUidCountFrames: 0,
    invalidWarmupComputeTimestampDurationFrames: 0,
    invalidWarmupRenderTimestampDurationFrames: 0,
    missingComputeFrames: 0,
    missingRenderFrames: 0,
    expectedComputeTimestampUidCount: 1,
    expectedRenderTimestampUidCount: 1,
    invalidComputeTimestampUidCountFrames: 0,
    invalidRenderTimestampUidCountFrames: 0,
    invalidComputeTimestampDurationFrames: 0,
    invalidRenderTimestampDurationFrames: 0,
    renderTimestampPoolQualityValid: true,
    computeTimestampPoolQualityValid: true,
    warmupRenderTimestampPoolQualityValid: true,
    warmupComputeTimestampPoolQualityValid: true,
    warmupTimestampFrameCountValid: true,
    measurementTimestampFrameCountValid: true,
    classification: 'fine',
    quantumNs: 1_000,
    timestampResolutions: {
      render: {
        quantumNs: 1_000,
        classification: 'fine',
        recordCount: 480,
        positiveDurationCount: 480,
        nonpositiveDurationCount: 0,
      },
      compute: {
        quantumNs: 1_000,
        classification: 'fine',
        recordCount: 480,
        positiveDurationCount: 480,
        nonpositiveDurationCount: 0,
      },
    },
    timestampPhases: {
      schemaVersion: 1,
      kind: 'three-r185-timestamp-phase-results',
      warmup: timestampPhase('warmup', warmupEvents),
      measurement: timestampPhase('measurement', measurementEvents),
    },
    completionInvariant,
  };
  const protocol = {
    schemaVersion: 2,
    warmupFrames: 320,
    measuredFrames: 480,
    plannedScheduleSha256: scheduleSha256,
  };
  return { spec, rows, summary, validation, protocol };
}

test('live schedule commitments bind both orientations', () => {
  assert.match(liveFirstInstanceCrossoverScheduleSha256(0), /^[0-9a-f]{64}$/);
  assert.match(liveFirstInstanceCrossoverScheduleSha256(1), /^[0-9a-f]{64}$/);
  assert.notEqual(
    liveFirstInstanceCrossoverScheduleSha256(0),
    liveFirstInstanceCrossoverScheduleSha256(1),
  );
});

test('semantic validation ignores legal survivor order and serial movement only', () => {
  const validation = {
    stable: 'kept',
    prepareSerialStart: 1,
    prepareSerialEnd: 3,
    lanes: {
      [PORTABLE]: { survivorSha256: 'a'.repeat(64), computeCallSerial: 4 },
      [FEATURE]: { survivorSha256: 'b'.repeat(64), computeCallSerial: 5 },
    },
    hooks: {
      [PORTABLE]: {
        addressChallenge: {
          sha256: '1'.repeat(64),
          expectedSha256: '1'.repeat(64),
          survivorSha256: '2'.repeat(64),
          exactExpectedBytes: true,
        },
        external: { pass: true, output: { sha256: '3'.repeat(64) } },
      },
      [FEATURE]: {
        addressChallenge: {
          sha256: '4'.repeat(64),
          expectedSha256: '4'.repeat(64),
          survivorSha256: '5'.repeat(64),
          exactExpectedBytes: true,
        },
        external: null,
      },
    },
    addressChallenges: {
      lanes: {
        [PORTABLE]: {
          sha256: '1'.repeat(64),
          expectedSha256: '1'.repeat(64),
          survivorSha256: '2'.repeat(64),
          exactExpectedBytes: true,
        },
        [FEATURE]: {
          sha256: '4'.repeat(64),
          expectedSha256: '4'.repeat(64),
          survivorSha256: '5'.repeat(64),
          exactExpectedBytes: true,
        },
      },
    },
    shaderEvidence: {
      stable: true,
      observation: { serial: 1, observationSha256: '8'.repeat(64) },
    },
    lifecycle: { activeLane: FEATURE, computeCallSerial: 5, stable: true },
  };
  const changedDynamic = structuredClone(validation);
  changedDynamic.prepareSerialStart = 20;
  changedDynamic.prepareSerialEnd = 22;
  changedDynamic.lanes[PORTABLE].survivorSha256 = 'c'.repeat(64);
  changedDynamic.lifecycle.activeLane = PORTABLE;
  changedDynamic.hooks[PORTABLE].addressChallenge.sha256 = '6'.repeat(64);
  changedDynamic.hooks[PORTABLE].addressChallenge.expectedSha256 = '6'.repeat(64);
  changedDynamic.hooks[PORTABLE].addressChallenge.survivorSha256 = '7'.repeat(64);
  changedDynamic.hooks[PORTABLE].external = null;
  changedDynamic.addressChallenges.lanes[PORTABLE].sha256 = '6'.repeat(64);
  changedDynamic.addressChallenges.lanes[PORTABLE].expectedSha256 = '6'.repeat(64);
  changedDynamic.addressChallenges.lanes[PORTABLE].survivorSha256 = '7'.repeat(64);
  changedDynamic.shaderEvidence.observation = {
    serial: 99,
    observationSha256: '9'.repeat(64),
  };
  assert.equal(
    liveFirstInstanceValidationSemanticSha256(validation),
    liveFirstInstanceValidationSemanticSha256(changedDynamic),
  );
  changedDynamic.stable = 'changed';
  assert.notEqual(
    liveFirstInstanceValidationSemanticSha256(validation),
    liveFirstInstanceValidationSemanticSha256(changedDynamic),
  );
});

test('fresh shader observation sequence binds six runner challenges and exact counters', () => {
  const spec = {
    runId: 'live-run',
    trialId: 'live-run-t01',
    planIndex: 0,
    repetitionIndex: 0,
  };
  const phaseRoles = [
    ['preflight', 'render-parity'],
    ['preflight', 'main-validation'],
    ['timing-start', 'render-parity'],
    ['timing-start', 'main-validation'],
    ['postflight', 'render-parity'],
    ['postflight', 'main-validation'],
  ];
  const challenges = phaseRoles.map(([phase, role], index) => ({
    schemaVersion: 1,
    kind: 'live-first-instance-shader-observation-challenge',
    origin: 'node-runner',
    runId: spec.runId,
    trialId: spec.trialId,
    planIndex: spec.planIndex,
    repetitionIndex: spec.repetitionIndex,
    phase,
    role,
    captureOrdinal: index + 1,
    challengeNonce: (index + 1).toString(16).repeat(64),
  }));
  const counterStarts = [
    [7, 3, 0],
    [10, 5, 2],
    [12, 7, 4],
    [15, 9, 6],
    [15 + LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT + 2,
      9 + LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT + 2, 8],
    [18 + LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT + 2,
      11 + LIVE_FIRST_INSTANCE_TIMED_FRAME_COUNT + 2, 10],
  ];
  const resources = { render: { portable: { id: 1 }, feature: { id: 2 } } };
  const fresh = challenges.map((request, index) => ({
    shaderEvidence: {
      observation: {
        serial: index + 1,
        request,
        executionCountersAtStart: {
          laneSelectionSerial: counterStarts[index][0],
          computeCallSerial: counterStarts[index][1],
          prepareSerial: counterStarts[index][2],
        },
        resourceIdentitiesAtStart: structuredClone(resources),
        semanticSha256: 'a'.repeat(64),
        observationSha256: (index + 1).toString(16).repeat(64),
      },
    },
  }));
  assert.deepEqual(validateLiveFirstInstanceShaderObservationSequence(
    fresh,
    { spec, shaderObservationChallenges: challenges },
  ), []);

  const stale = structuredClone(fresh);
  stale[1].shaderEvidence.observation = structuredClone(stale[0].shaderEvidence.observation);
  const staleReasons = validateLiveFirstInstanceShaderObservationSequence(
    stale,
    { spec, shaderObservationChallenges: challenges },
  );
  assert.ok(staleReasons.some((reason) => reason.includes('runner challenge')));
  assert.ok(staleReasons.some((reason) => reason.includes('repeated cached copies')));

  const changedResource = structuredClone(fresh);
  changedResource[4].shaderEvidence.observation.resourceIdentitiesAtStart.render.portable.id = 99;
  assert.ok(validateLiveFirstInstanceShaderObservationSequence(
    changedResource,
    { spec, shaderObservationChallenges: challenges },
  ).some((reason) => reason.includes('cross-phase resource identities')));

  const skippedTimingWork = structuredClone(fresh);
  skippedTimingWork[4].shaderEvidence.observation.executionCountersAtStart.computeCallSerial -= 1;
  assert.ok(validateLiveFirstInstanceShaderObservationSequence(
    skippedTimingWork,
    { spec, shaderObservationChallenges: challenges },
  ).some((reason) => reason.includes('computeCallSerial transition')));
});

test('forced-feature-off evidence requires portable-only construction and full gates', () => {
  const gate = {
    schemaVersion: 1,
    kind: 'first-instance-live-forced-feature-off-deployment-gate',
    pass: true,
    passBeforeDisposal: true,
    actualFeatureAvailable: true,
    forcedFeatureAvailable: false,
    separateDisposableRendererRequired: true,
    timingContaminationBoundary: 'caller-owned-disposable-renderer-device',
    selection: { lane: PORTABLE, strategyId: 'fixed-slice', featureAvailable: false },
    construction: {
      pass: true,
      constructedLane: PORTABLE,
      featureLaneConstructed: false,
      portableBucketBasePresent: true,
      geometryModes: ['bucket-base'],
      configuredDrawCommands: 32,
      configuredComputeDispatches: 2,
      configuredComputeSubmissions: 1,
    },
    commands: {
      pass: true,
      commandRecordCount: 32,
      drawCommandCount: 32,
      commandByteOffset: 0,
      fifthCommandWordsAllZero: true,
      nonzeroFirstInstanceCount: 0,
      firstInstanceWords: Array(32).fill(0),
    },
    correctness: {
      pass: true,
      membership: { pass: true },
      membershipDigests: { pass: true },
      commandValidation: { pass: true },
      overflow: 0,
    },
    address: { pass: true },
    output: { pass: true },
    disposal: { pass: true, attempted: true, rootDetached: true, indirectDetached: true },
  };
  assert.deepEqual(validateLiveFirstInstanceForcedFeatureOffGate(gate), []);
  gate.commands.firstInstanceWords[7] = 1;
  assert.ok(validateLiveFirstInstanceForcedFeatureOffGate(gate).some(
    (reason) => reason.includes('32 zeros'),
  ));
});

test('live rows require exact compute/render timestamps, serials, and one-microsecond quantum', () => {
  const artifact = createRowEvidence();
  assert.deepEqual(validateLiveFirstInstanceCrossoverRows(artifact), []);

  const coarse = structuredClone(artifact);
  coarse.summary.quantumNs = 1_001;
  assert.ok(validateLiveFirstInstanceCrossoverRows(coarse).some(
    (reason) => reason.includes('1,000 ns'),
  ));

  const malformed = structuredClone(artifact);
  malformed.rows[9].gpuComputeTimestampUidCount = 0;
  malformed.rows[10].computeCallSerial += 1;
  malformed.rows[11].commandByteBase = 20;
  const reasons = validateLiveFirstInstanceCrossoverRows(malformed);
  assert.ok(reasons.some((reason) => reason.includes('compute timestamp UID count')));
  assert.ok(reasons.some((reason) => reason.includes('computeCallSerial')));
  assert.ok(reasons.some((reason) => reason.includes('commandByteBase')));

  const zeroCompute = structuredClone(artifact);
  zeroCompute.rows[0].gpuComputeMs = 0;
  zeroCompute.rows[0].gpuPassTotalMs = zeroCompute.rows[0].gpuRenderMs;
  assert.ok(validateLiveFirstInstanceCrossoverRows(zeroCompute).some(
    (reason) => reason.includes('gpuComputeMs is invalid'),
  ));

  const changedHistory = structuredClone(artifact);
  changedHistory.rows[0].previousLaneId = PORTABLE;
  assert.ok(validateLiveFirstInstanceCrossoverRows(changedHistory).some(
    (reason) => reason.includes('previousLaneId'),
  ));

  const changedWarmupStartup = structuredClone(artifact);
  changedWarmupStartup.summary.completionInvariant.warmupScheduleAudit
    .events[0].previousLaneId = FEATURE;
  assert.ok(validateLiveFirstInstanceCrossoverRows(changedWarmupStartup).some(
    (reason) => reason.includes('actual previous lane'),
  ));

  const changedWarmupContext = structuredClone(artifact);
  const warmupEvent = changedWarmupContext.summary.completionInvariant
    .warmupScheduleAudit.events[0];
  warmupEvent.gpuComputeTimestampRecords[0].contextId += 1;
  warmupEvent.gpuComputeTimestampRecords[0].uid =
    `c:1:${warmupEvent.gpuComputeTimestampRecords[0].contextId}:f${warmupEvent.gpuFrameId}`;
  warmupEvent.gpuComputeTimestampUids[0] = warmupEvent.gpuComputeTimestampRecords[0].uid;
  assert.ok(validateLiveFirstInstanceCrossoverRows(changedWarmupContext).some(
    (reason) => reason.includes('compute timestamp contextId'),
  ));

  const coarseWarmupComputePool = structuredClone(artifact);
  coarseWarmupComputePool.summary.timestampPhases.warmup
    .pools.compute.resolution.quantumNs = 1_001;
  assert.ok(validateLiveFirstInstanceCrossoverRows(coarseWarmupComputePool).some(
    (reason) => reason.includes('warmup compute timestamp pool'),
  ));
});

test('live warmup boundary permits idle global RAFs during timestamp resolution', () => {
  // Three r185 advances renderer.info.frame on every RAF, including RAFs where
  // the controller is resolving warmup timestamps and submits no timed work.
  // Trial 3 of the preserved candidate attempt observed exactly this +1 gap.
  const oneIdleRaf = shiftMeasurementGpuFrames(createRowEvidence(), 1);
  assert.deepEqual(validateLiveFirstInstanceCrossoverRows(oneIdleRaf), []);

  const changedStrategyComputeSerial = structuredClone(oneIdleRaf);
  changedStrategyComputeSerial.rows[0].strategyComputeCallSerial += 1;
  assert.ok(validateLiveFirstInstanceCrossoverRows(changedStrategyComputeSerial).some(
    (reason) => reason.includes('strategyComputeCallSerial'),
  ));

  const equalBoundary = shiftMeasurementGpuFrames(createRowEvidence(), -1);
  assert.ok(validateLiveFirstInstanceCrossoverRows(equalBoundary).some(
    (reason) => reason.includes(
      'measurement-boundary GPU frame does not advance after warmup',
    ),
  ));

  const backwardBoundary = shiftMeasurementGpuFrames(createRowEvidence(), -2);
  assert.ok(validateLiveFirstInstanceCrossoverRows(backwardBoundary).some(
    (reason) => reason.includes(
      'measurement-boundary GPU frame does not advance after warmup',
    ),
  ));
});

test('live completion reconstructs the strategy compute-call serial interval', () => {
  const changedEnd = createRowEvidence();
  changedEnd.summary.completionInvariant.strategyComputeCallSerialAtTimingEnd += 1;
  assert.ok(validateLiveFirstInstanceCrossoverRows(changedEnd).some(
    (reason) => reason.includes('strategyComputeCallSerial serial delta'),
  ));

  const changedCount = createRowEvidence();
  changedCount.summary.completionInvariant.strategyComputeCallsDuringTiming -= 1;
  assert.ok(validateLiveFirstInstanceCrossoverRows(changedCount).some(
    (reason) => reason.includes('strategyComputeCallsDuringTiming'),
  ));
});

test('live completion rejects command-buffer version mutation', () => {
  const artifact = createRowEvidence();
  artifact.summary.completionInvariant.commandBufferCommitments[PORTABLE]
    .attributeVersion += 1;
  const reasons = validateLiveFirstInstanceCrossoverRows(artifact);
  assert.ok(reasons.some((reason) => reason.includes('command-buffer commitment')));
});

test('live completion rejects viewport-only mutation', () => {
  const artifact = createRowEvidence();
  artifact.summary.completionInvariant.viewportStateAtTimingEnd.renderTarget.viewport.width -= 1;
  const reasons = validateLiveFirstInstanceCrossoverRows(artifact);
  assert.ok(reasons.some((reason) => reason.includes('viewport')));
});

test('live completion rejects compute-program and renderer-memory mutation', () => {
  const programMutation = createRowEvidence();
  programMutation.summary.completionInvariant.computeProgramEntriesAtTimingEnd += 1;
  assert.ok(validateLiveFirstInstanceCrossoverRows(programMutation).some(
    (reason) => reason.includes('computeProgramEntries'),
  ));

  const memoryMutation = createRowEvidence();
  memoryMutation.summary.completionInvariant.rendererMemoryAtTimingEnd.programs += 1;
  assert.ok(validateLiveFirstInstanceCrossoverRows(memoryMutation).some(
    (reason) => reason.includes('renderer memory'),
  ));
});
