import assert from 'node:assert/strict';
import test from 'node:test';
import {
  summarizeCsv,
  validateBenchmarkPlan,
  validateProtocolMatrix,
} from '../analysis/summarize.mjs';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS,
  FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS,
  firstInstanceLiveCrossoverHistoryCounts,
  firstInstanceLiveCrossoverFrame,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS,
  buildFirstInstanceLiveCrossoverPlan,
} from '../src/benchmark/plan.js';
import {
  liveFirstInstanceCrossoverScheduleSha256,
} from '../scripts/live-first-instance-evidence-validation.mjs';

const RUN_ID = 'first-instance-live-run-verification';
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const [PORTABLE, FEATURE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;

function protocol() {
  return {
    matrix: 'first-instance-live-o65536-b32',
    matrixKind: 'first-instance-live',
    representationScaleRole: null,
    layouts: ['baseline'],
    depthBinCount: null,
    reversedDepthBuffer: true,
    minimumStorageBuffersPerShaderStage: 8,
    renderParity:
      'preflight, timing-start, and postflight serialized live-compute portable/feature exact validation plus two stable offscreen captures per lane of rgba8 color, depth32float, and encoded object ID',
    firstInstanceLiveCrossover: {
      requiredFeature: 'indirect-first-instance',
      lanes: [...FIRST_INSTANCE_LIVE_CROSSOVER_LANES],
      blockSize: FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
      warmupBlocks: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS,
      measuredBlocks: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS,
      patterns: FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS.map((pattern) => pattern.map(
        (laneId) => (laneId === PORTABLE ? 'P' : 'F'),
      ).join('')),
      scheduleDesign: 'cyclic-binary-de-bruijn-order-three-with-complementary-orientation',
      expectedMeasuredRowsPerLane: 240,
      expectedMeasuredTransitionCounts: {
        ...firstInstanceLiveCrossoverHistoryCounts(480, 0).transitionCounts,
      },
      expectedMeasuredHistoryTripleCounts: {
        ...firstInstanceLiveCrossoverHistoryCounts(480, 0).historyTripleCounts,
      },
      expectedComputeCallsPerFrame: 1,
      expectedRenderCallsPerFrame: 1,
      expectedComputeTimestampUidCount: 1,
      expectedRenderTimestampUidCount: 1,
      commandBuffers: 2,
      commandBufferByteOffset: 0,
      commandRecordsPerBuffer: BUCKET_COUNT,
      scheduleSha256ByOrientation: {
        0: liveFirstInstanceCrossoverScheduleSha256(0),
        1: liveFirstInstanceCrossoverScheduleSha256(1),
      },
    },
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    allowedObjectCounts: [4_096, 16_384, 65_536],
    allowedBucketCounts: [1, 4, 32, 128],
    allowedHeterogeneousComparators: ['coalesced-v11', 'historical-v10'],
    heterogeneousComparator: null,
    modes: [FIRST_INSTANCE_LIVE_CROSSOVER_MODE],
    visibilityLevels: [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS],
    repetitions: FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
    warmupFrames: 320,
    measuredFrames: 480,
    maximumCpuTimerQuantumMs: 0.01,
    ordering:
      'twelve-repetition-two-visibility-cyclic-eight-frame-live-crossover-with-exact-transition-history-balance-and-pairwise-balanced-lane-physical-order-visibility-order-and-orientation',
    threeBlocksScheduling: null,
  };
}

function rows() {
  const plan = buildFirstInstanceLiveCrossoverPlan({
    runId: RUN_ID,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  });
  const result = [];
  for (const trial of plan) {
    const selectorStart = trial.planIndex * 10_000;
    const selectionStart = selectorStart + 100;
    const computeStart = selectorStart + 200;
    const renderStart = selectorStart + 300;
    const commandIds = Object.fromEntries(trial.lanePhysicalOrder.map(
      (laneId, index) => [laneId, 1_000 + trial.planIndex * 10 + index],
    ));
    const timestampRegistrations = Object.fromEntries(
      FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((laneId, laneIndex) => {
        const timestampContextId = trial.planIndex * 2 + laneIndex + 1;
        return [laneId, {
          timestampContextId,
          computeGroupIdentity: 10_000 + timestampContextId,
          registrationSerial: timestampContextId,
          backendIdentity: 20_001,
          backendWrapperIdentity: 20_002,
          computeNodeIds: [
            30_000 + timestampContextId * 2,
            30_000 + timestampContextId * 2 + 1,
          ],
        }];
      }),
    );
    const expectedVisibleCount = Math.round(OBJECT_COUNT * trial.visibilityFraction);
    const rendererMemoryAtTimingStart = JSON.stringify({
      attributes: 8,
      attributesSize: 1024,
      geometries: 2,
      indexAttributes: 1,
      indexAttributesSize: 256,
      indirectStorageAttributes: 2,
      indirectStorageAttributesSize: 1280,
      programs: 6,
      programsSize: 2048,
      storageAttributes: 8,
      storageAttributesSize: 4096,
      uniformBuffers: 2,
      uniformBuffersSize: 256,
    });
    const viewportStateAtTimingStart = JSON.stringify({
      schemaVersion: 1,
      kind: 'three-r185-live-first-instance-viewport-state',
      renderer: {
        viewport: { x: 0, y: 0, width: 1280, height: 720, minDepth: 0, maxDepth: 1 },
        scissor: { x: 0, y: 0, width: 1280, height: 720 },
        scissorTest: false,
        activeRenderTargetTextureUuid: null,
      },
      renderTarget: {
        textureUuid: `target-${trial.planIndex}`,
        width: 1280,
        height: 720,
        viewport: { x: 0, y: 0, width: 1280, height: 720, minDepth: 0, maxDepth: 1 },
        scissor: { x: 0, y: 0, width: 1280, height: 720 },
        scissorTest: false,
      },
    });
    for (let frameIndex = 0; frameIndex < 480; frameIndex += 1) {
      const scheduled = firstInstanceLiveCrossoverFrame(
        frameIndex,
        trial.superblockOrientationOffset,
      );
      const feature = scheduled.laneId === FEATURE;
      const gpuComputeMs = 0.2;
      const gpuRenderMs = feature && trial.visibilityFraction === 0.99 ? 0.64 : 0.8;
      const serialOffset = 320 + frameIndex + 1;
      const gpuFrameId = trial.planIndex * 1_000 + frameIndex;
      const registration = timestampRegistrations[scheduled.laneId];
      const computeFrameCallIndex = 1;
      const renderFrameCallIndex = 1;
      const computeTimestampUid =
        `c:${computeFrameCallIndex}:${registration.timestampContextId}:f${gpuFrameId}`;
      const renderTimestampUid = `r:${renderFrameCallIndex}:0:f${gpuFrameId}`;
      result.push({
        schemaVersion: 2,
        runId: RUN_ID,
        trialId: trial.trialId,
        planIndex: trial.planIndex,
        repetitionIndex: trial.repetitionIndex,
        modeOrderPosition: 0,
        visibilityOrderPosition: trial.visibilityOrderPosition,
        layoutOrderPosition: 0,
        plannedModeOrder: FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
        plannedVisibilityOrder: trial.visibilityOrder.join('|'),
        plannedLayoutOrder: 'baseline',
        frameIndex,
        phaseFrameIndex: frameIndex,
        modeId: FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
        objectCount: OBJECT_COUNT,
        bucketCount: BUCKET_COUNT,
        targetVisibilityFraction: trial.visibilityFraction,
        scenarioLayout: 'baseline',
        expectedVisibleCount,
        protocolWarmupFrames: 320,
        protocolMeasuredFrames: 480,
        validationKind: 'first-instance-live-crossover-exact-paired-snapshots',
        validationPass: true,
        usesCompute: true,
        configuredDrawCommands: BUCKET_COUNT,
        configuredRenderObjects: 1,
        configuredComputeDispatches: 2,
        configuredComputeSubmissions: 1,
        configuredSubmittedInstances: expectedVisibleCount,
        timestampAvailable: true,
        strictTimestampUidAttribution: true,
        expectedComputeTimestampUidCount: 1,
        expectedRenderTimestampUidCount: 1,
        plannedLanePhysicalOrder: trial.lanePhysicalOrder.join('|'),
        lanePhysicalOrder: trial.lanePhysicalOrder.join('|'),
        plannedScheduleSha256: liveFirstInstanceCrossoverScheduleSha256(
          trial.superblockOrientationOffset,
        ),
        superblockOrientationOffset: trial.superblockOrientationOffset,
        lifecycleCommitmentAtTimingStart: '0000000000000001',
        commandBufferCommitmentsAtTimingStart: '{}',
        portableCommandBufferIdAtTimingStart: commandIds[PORTABLE],
        featureCommandBufferIdAtTimingStart: commandIds[FEATURE],
        selectorWriteSerialAtTimingStart: selectorStart,
        strategySelectionSerialAtTimingStart: selectionStart,
        strategyComputeCallSerialAtTimingStart: computeStart,
        strategyPrepareSerialAtTimingStart: 1,
        computeCallSerialAtTimingStart: computeStart,
        renderCallSerialAtTimingStart: renderStart,
        renderTargetTextureUuidAtTimingStart: `target-${trial.planIndex}`,
        renderTargetWidthAtTimingStart: 1280,
        renderTargetHeightAtTimingStart: 720,
        renderTargetSamplesAtTimingStart: 0,
        renderTargetDepthBufferAtTimingStart: true,
        cameraViewFnv64AtTimingStart: '0000000000000002',
        cameraProjectionFnv64AtTimingStart: '0000000000000003',
        totalPipelineCacheEntriesAtTimingStart: 4,
        computePipelineCacheEntriesAtTimingStart: 2,
        computeProgramEntriesAtTimingStart: 2,
        rendererMemoryAtTimingStart,
        viewportStateAtTimingStart,
        timestampPoolStaticCommitmentAtTimingStart: '{}',
        webgpuUncapturedErrorCountAtTimingStart: 0,
        crossoverBlockIndex: scheduled.crossoverBlockIndex,
        withinBlockPosition: scheduled.withinBlockPosition,
        crossoverPattern: scheduled.pattern,
        crossoverPatternIndex: scheduled.patternIndex,
        previousPreviousLaneId: scheduled.previousPreviousLaneId,
        previousLaneId: scheduled.previousLaneId,
        laneId: scheduled.laneId,
        commandBufferId: commandIds[scheduled.laneId],
        submittedComputeLaneId: scheduled.laneId,
        computeTimestampContextId: registration.timestampContextId,
        computeGroupIdentity: registration.computeGroupIdentity,
        computeTimestampRegistrationSerial: registration.registrationSerial,
        computeTimestampBackendIdentity: registration.backendIdentity,
        computeTimestampBackendWrapperIdentity: registration.backendWrapperIdentity,
        submittedComputeNodeIds: JSON.stringify(registration.computeNodeIds),
        commandSegmentIndex: 0,
        commandRecordBase: 0,
        commandByteBase: 0,
        commandByteOffset: 0,
        commandBufferRecordCount: BUCKET_COUNT,
        commandBufferByteLength: BUCKET_COUNT * 20,
        selectorWriteSerial: selectorStart + serialOffset,
        strategySelectionSerial: selectionStart + serialOffset,
        strategyComputeCallSerial: computeStart + serialOffset,
        computeCallSerial: computeStart + serialOffset,
        computeFrameCallIndex,
        renderCallSerial: renderStart + serialOffset,
        renderFrameCallIndex,
        gpuFrameId,
        gpuComputeTimestampUidCount: 1,
        gpuComputeTimestampUids: JSON.stringify([computeTimestampUid]),
        gpuComputeTimestampRecords: JSON.stringify([{
          uid: computeTimestampUid,
          type: 'compute',
          callIndex: computeFrameCallIndex,
          contextId: registration.timestampContextId,
          frameId: gpuFrameId,
          durationMs: gpuComputeMs,
        }]),
        gpuComputeTimestampDurationValid: true,
        gpuRenderTimestampUidCount: 1,
        gpuRenderTimestampUids: JSON.stringify([renderTimestampUid]),
        gpuRenderTimestampRecords: JSON.stringify([{
          uid: renderTimestampUid,
          type: 'render',
          callIndex: renderFrameCallIndex,
          contextId: 0,
          frameId: gpuFrameId,
          durationMs: gpuRenderMs,
        }]),
        gpuRenderTimestampDurationValid: true,
        cpuCommonUpdateMs: 0.01,
        cpuComputeSubmitMs: 0.02,
        cpuRenderSubmitMs: 0.03,
        cpuSubmitTotalMs: 0.05,
        cpuFrameBodyMs: 0.06,
        gpuComputeMs,
        gpuRenderMs,
        gpuPassTotalMs: gpuComputeMs + gpuRenderMs,
      });
    }
  }
  return result;
}

function csv(records) {
  const headers = Object.keys(records[0]);
  const encode = (value) => {
    const text = String(value);
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [
    headers.join(','),
    ...records.map((record) => headers.map((header) => encode(record[header])).join(',')),
  ].join('\n');
}

test('live protocol and plan verification bind the exact preregistered matrix', () => {
  const exactProtocol = protocol();
  assert.doesNotThrow(() => validateProtocolMatrix(exactProtocol));
  const plan = buildFirstInstanceLiveCrossoverPlan({
    runId: RUN_ID,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  }).map((trial) => ({ ...trial, runId: RUN_ID }));
  assert.doesNotThrow(() => validateBenchmarkPlan(plan, {
    runId: RUN_ID,
    protocol: exactProtocol,
  }));

  const changed = structuredClone(plan);
  changed[0].lanePhysicalOrder.reverse();
  assert.throws(
    () => validateBenchmarkPlan(changed, { runId: RUN_ID, protocol: exactProtocol }),
    /preregistered live first-instance crossover factor/,
  );
});

test('raw live CSV routes through strict parsing and the live analyzer', () => {
  const records = rows();
  const summary = summarizeCsv(csv(records));
  assert.equal(summary.kind, 'indirect-first-instance-live-crossover-summary');
  assert.equal(summary.nTrials, 24);
  assert.equal(summary.nRows, 11_520);
  assert.equal(summary.artifactVerification.status, 'unverified');

  const missingComputeUid = records.map((record) => ({ ...record }));
  for (const record of missingComputeUid) delete record.gpuComputeTimestampUidCount;
  assert.throws(
    () => summarizeCsv(csv(missingComputeUid)),
    /missing required columns: gpuComputeTimestampUidCount/,
  );

  const missingRendererState = records.map((record) => ({ ...record }));
  for (const record of missingRendererState) {
    delete record.computeProgramEntriesAtTimingStart;
    delete record.rendererMemoryAtTimingStart;
    delete record.viewportStateAtTimingStart;
  }
  assert.throws(
    () => summarizeCsv(csv(missingRendererState)),
    /missing required columns: computeProgramEntriesAtTimingStart, rendererMemoryAtTimingStart, viewportStateAtTimingStart/,
  );

  const malformedRendererState = records.map((record) => ({ ...record }));
  malformedRendererState[0].rendererMemoryAtTimingStart = '{invalid-json';
  assert.throws(
    () => summarizeCsv(csv(malformedRendererState)),
    /invalid JSON in rendererMemoryAtTimingStart/,
  );

  const wrongComputeTimestampContext = records.map((record) => ({ ...record }));
  const firstComputeTimestampRecord = JSON.parse(
    wrongComputeTimestampContext[0].gpuComputeTimestampRecords,
  );
  firstComputeTimestampRecord[0].contextId += 1;
  wrongComputeTimestampContext[0].gpuComputeTimestampRecords = JSON.stringify(
    firstComputeTimestampRecord,
  );
  assert.throws(
    () => summarizeCsv(csv(wrongComputeTimestampContext)),
    /compute timestamp.*(?:contextId|exact UID)|contextId.*compute timestamp/i,
  );
});
