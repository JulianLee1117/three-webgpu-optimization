import assert from 'node:assert/strict';
import test from 'node:test';

import {
  STANDALONE_EXECUTION_MODES,
  STANDALONE_FAILURE_DIAGNOSTIC_LIMITS,
  STANDALONE_FULL_RESULT_ROOT_NAME,
  STANDALONE_FULL_RUN_ID_STEM,
  STANDALONE_POST_DISCONNECT_DELAY_MS,
  boundStandaloneFailureDiagnosticValue,
  captureStandaloneReadinessSnapshot,
  createStandaloneBrowserObservationRecorder,
  createStandaloneAuditContext,
  createStandaloneShaderObservationChallenges,
  createStandaloneTerminalFailureEnvelope,
  createStandaloneTrialArtifact,
  guardStandaloneBrowserOperation,
  selectStandaloneDeploymentExecution,
  standaloneFullEnvironmentGatesPassed,
  validateStandaloneBrowserLifecycleChain,
  validateStandaloneSessionTimestampContinuity,
  validateStandaloneTimingRows,
} from '../scripts/run-first-instance-standalone-deployment.mjs';
import {
  buildFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';

const RUN_ID = 'standalone-runner-unit';
const PLAN_SHA256 = 'a'.repeat(64);
const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
const canonicalTrial = plan.trials[0];

test('corrective full capture uses an isolated v2 result namespace', () => {
  assert.equal(STANDALONE_FULL_RUN_ID_STEM, 'first-instance-standalone-deployment-v2');
  assert.equal(STANDALONE_FULL_RESULT_ROOT_NAME, 'candidate-standalone-deployment-v2');
});

function timestampUid(type, frame) {
  const prefix = type === 'compute' ? 'c' : 'r';
  const contextId = type === 'compute' ? 2 : 0;
  return `${prefix}:1:${contextId}:f${frame}`;
}

function timestampPhase(startFrame, frameCount) {
  const frames = Array.from({ length: frameCount }, (_, index) => startFrame + index);
  const pool = (type) => ({
    included: true,
    frames: [...frames],
    uidRecords: frames.map((frame) => ({ uid: timestampUid(type, frame) })),
    resolution: { quantumNs: 1 },
  });
  return { pools: { compute: pool('compute'), render: pool('render') } };
}

function timestampPool(type, {
  frames,
  timestampUids,
  queryOffsetUids = [],
  currentQueryIndex = 0,
}) {
  const identityBase = type === 'render' ? 10 : 20;
  return {
    poolIdentity: identityBase,
    querySetIdentity: identityBase + 1,
    resolveBufferIdentity: identityBase + 2,
    resultBufferIdentity: identityBase + 3,
    maxQueries: 2_048,
    currentQueryIndex,
    queryOffsetCount: queryOffsetUids.length,
    queryOffsetUids,
    frameCount: frames.length,
    frames,
    timestampUidCount: timestampUids.length,
    timestampUids: [...timestampUids].sort(),
    pendingResolve: false,
    isDisposed: false,
    resultBufferMapState: 'unmapped',
  };
}

function timestampDiagnostics({
  tracking,
  frames,
  timestampUidsByType,
  queryOffsetUidsByType = { render: [], compute: [] },
  currentQueryIndex = 0,
}) {
  return {
    schemaVersion: 1,
    kind: 'three-r185-timestamp-pool-diagnostics',
    backendTrackingEnabled: tracking,
    render: timestampPool('render', {
      frames,
      timestampUids: timestampUidsByType.render,
      queryOffsetUids: queryOffsetUidsByType.render,
      currentQueryIndex,
    }),
    compute: timestampPool('compute', {
      frames,
      timestampUids: timestampUidsByType.compute,
      queryOffsetUids: queryOffsetUidsByType.compute,
      currentQueryIndex,
    }),
  };
}

function acceptedSummary() {
  const warmup = timestampPhase(10, 320);
  const measurement = timestampPhase(330, 480);
  const allFrames = [...warmup.pools.render.frames, ...measurement.pools.render.frames];
  const timedUids = Object.fromEntries(['render', 'compute'].map((type) => [
    type,
    [
      ...warmup.pools[type].uidRecords,
      ...measurement.pools[type].uidRecords,
    ].map(({ uid }) => uid),
  ]));
  const primedUids = {
    render: ['r:2:0:f1'],
    compute: ['c:2:2:f1'],
  };
  const startPools = timestampDiagnostics({
    tracking: false,
    frames: [1],
    timestampUidsByType: primedUids,
  });
  const beforePools = timestampDiagnostics({
    tracking: true,
    frames: [1],
    timestampUidsByType: primedUids,
    queryOffsetUidsByType: timedUids,
    currentQueryIndex: 1_600,
  });
  const afterPools = timestampDiagnostics({
    tracking: true,
    frames: allFrames,
    timestampUidsByType: {
      render: [...primedUids.render, ...timedUids.render],
      compute: [...primedUids.compute, ...timedUids.compute],
    },
  });
  return {
    rowCount: 480,
    warmupRowCount: 320,
    missingWarmupRenderFrames: 0,
    invalidWarmupRenderTimestampUidCountFrames: 0,
    invalidWarmupRenderTimestampDurationFrames: 0,
    missingWarmupComputeFrames: 0,
    invalidWarmupComputeTimestampUidCountFrames: 0,
    invalidWarmupComputeTimestampDurationFrames: 0,
    missingRenderFrames: 0,
    invalidRenderTimestampUidCountFrames: 0,
    invalidRenderTimestampDurationFrames: 0,
    missingComputeFrames: 0,
    invalidComputeTimestampUidCountFrames: 0,
    invalidComputeTimestampDurationFrames: 0,
    expectedRenderTimestampUidCount: 1,
    expectedComputeTimestampUidCount: 1,
    timestampAvailable: true,
    accepted: true,
    completionInvariant: {
      pass: true,
      timestampPoolsAtTimingStart: startPools,
      timestampPoolsAtTimingEnd: afterPools,
      timestampPhaseBoundaryExact: true,
      timestampResolvedFrameIntervalExact: true,
      timestampResolutionTopologyExact: true,
    },
    renderTimestampPoolQualityValid: true,
    computeTimestampPoolQualityValid: true,
    warmupRenderTimestampPoolQualityValid: true,
    warmupComputeTimestampPoolQualityValid: true,
    warmupTimestampFrameCountValid: true,
    measurementTimestampFrameCountValid: true,
    quantumNs: 1,
    timestampPhases: {
      warmup,
      measurement,
    },
    timestampResolutionTopology: {
      schemaVersion: 1,
      kind: 'three-r185-timestamp-resolution-topology',
      mode: 'single-post-measurement',
      warmupBoundaryResolveBatchCount: 0,
      postMeasurementResolveBatchCount: 1,
      queriesPerTimestampUid: 2,
      requiredQueriesPerType: 1_600,
      resolvedFrameCountByType: { render: 800, compute: 800 },
      firstGpuFrameId: 10,
      lastGpuFrameId: 809,
      intervalContiguous: true,
      poolsAtStart: startPools,
      poolsBeforePostMeasurementResolve: beforePools,
      poolsAfterPostMeasurementResolve: afterPools,
    },
  };
}

function acceptedRows(trial = canonicalTrial) {
  const cpuComputeSubmitMs = 0.1;
  const cpuRenderSubmitMs = 0.2;
  return Array.from({ length: 480 }, (_, index) => {
    const gpuFrameId = 330 + index;
    const computeUid = timestampUid('compute', gpuFrameId);
    const renderUid = timestampUid('render', gpuFrameId);
    return {
      ...trial,
      runId: RUN_ID,
      harnessContextSchemaVersion: 2,
      frameIndex: index,
      phaseFrameIndex: index,
      gpuFrameId,
      measuredBlockIndex: Math.floor(index / 8),
      withinBlockPosition: index % 8,
      laneId: trial.assignedLaneId,
      submittedComputeLaneId: trial.assignedLaneId,
      targetVisibilityFraction: trial.visibilityFraction,
      commandSegmentIndex: 0,
      commandRecordBase: 0,
      commandByteBase: 0,
      commandByteOffset: 0,
      commandBufferRecordCount: 32,
      configuredDrawCommands: 32,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 2,
      configuredComputeSubmissions: 1,
      objectCount: 65_536,
      bucketCount: 32,
      gpuComputeMs: 0.4,
      gpuRenderMs: 0.6,
      gpuPassTotalMs: 1,
      cpuCommonUpdateMs: 0.05,
      cpuComputeSubmitMs,
      cpuRenderSubmitMs,
      cpuSubmitTotalMs: cpuComputeSubmitMs + cpuRenderSubmitMs,
      cpuFrameBodyMs: 0.4,
      gpuComputeTimestampUidCount: 1,
      gpuRenderTimestampUidCount: 1,
      gpuComputeTimestampDurationValid: true,
      gpuRenderTimestampDurationValid: true,
      gpuComputeTimestampUids: JSON.stringify([computeUid]),
      gpuRenderTimestampUids: JSON.stringify([renderUid]),
      gpuComputeTimestampRecords: JSON.stringify([{
        uid: computeUid,
        durationMs: 0.4,
      }]),
      gpuRenderTimestampRecords: JSON.stringify([{
        uid: renderUid,
        durationMs: 0.6,
      }]),
    };
  });
}

test('failure diagnostics retain fatal browser observations under declared caps', () => {
  let clock = 0;
  const recorder = createStandaloneBrowserObservationRecorder({
    now: () => `2026-01-01T00:00:${String(clock).padStart(2, '0')}.000Z`,
    elapsed: () => clock++,
  });
  const oversizedStack = 'x'.repeat(
    STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticStringCodeUnits + 100,
  );
  for (let index = 0;
    index < STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.observationRecordsPerKind + 4;
    index += 1) {
    recorder.record('pageerror', {
      name: 'Error',
      message: `failure-${index}`,
      stack: oversizedStack,
    }, { fatal: true });
  }
  recorder.record('requestfailed', {
    url: 'http://127.0.0.1/module.js',
    failure: { errorText: 'net::ERR_FAILED' },
  }, { fatal: true });
  recorder.record('console-error', { text: 'module failed' }, { fatal: true });
  recorder.record('http-error-response', {
    url: 'http://127.0.0.1/missing.js',
    status: 404,
  }, { fatal: true });
  recorder.record('page-crash', {}, { fatal: true });
  recorder.record('page-close', { shutdownIntent: null }, { fatal: true });
  recorder.record('browser-disconnected', {
    shutdownIntent: null,
  }, { fatal: true });
  recorder.record('browser-disconnected', {
    shutdownIntent: 'failure-cleanup',
  }, { fatal: true, expected: true });

  const report = recorder.report();
  assert.equal(report.totalObserved,
    STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.observationRecordsPerKind + 11);
  assert.equal(report.observedCounts.pageerror,
    STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.observationRecordsPerKind + 4);
  assert.equal(report.retainedCounts.pageerror,
    STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.observationRecordsPerKind);
  assert.equal(report.droppedCounts.pageerror, 4);
  assert.equal(report.recordsByKind.requestfailed.length, 1);
  assert.equal(report.recordsByKind['console-error'].length, 1);
  assert.equal(report.recordsByKind['http-error-response'].length, 1);
  assert.equal(report.recordsByKind['page-crash'].length, 1);
  assert.equal(report.recordsByKind['page-close'].length, 1);
  assert.equal(report.recordsByKind['browser-disconnected'][0].expected, false);
  assert.equal(report.recordsByKind['browser-disconnected'][1].expected, true);
  assert.equal(report.firstFatalObservation.kind, 'pageerror');
  assert.match(report.recordsByKind.pageerror[0].detail.stack, /diagnostic text omitted/);
  assert.ok(report.recordsByKind.pageerror[0].detail.stack.length
    <= STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticStringCodeUnits);

  const cyclic = { values: Array.from({ length: 40 }, (_, index) => index) };
  cyclic.self = cyclic;
  const bounded = boundStandaloneFailureDiagnosticValue(cyclic);
  assert.match(bounded.self, /circular/);
  assert.equal(bounded.values.length,
    STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticArrayItems + 1);
});

test('fatal page observation wins a pending browser operation without waiting for timeout', async () => {
  const recorder = createStandaloneBrowserObservationRecorder();
  const guarded = guardStandaloneBrowserOperation(
    () => new Promise(() => {}),
    recorder,
    'readiness',
  );
  recorder.record('pageerror', {
    name: 'Error',
    message: 'module boot failed',
  }, { fatal: true });
  await assert.rejects(guarded, (error) => {
    assert.equal(error.name, 'StandaloneBrowserObservationError');
    assert.equal(error.code, 'FIRST_INSTANCE_STANDALONE_BROWSER_OBSERVATION');
    assert.match(error.message, /pageerror/);
    return true;
  });

  const expectedRecorder = createStandaloneBrowserObservationRecorder();
  expectedRecorder.record('browser-disconnected', {
    shutdownIntent: 'normal-close',
  }, { fatal: true, expected: true });
  assert.equal(await guardStandaloneBrowserOperation(
    () => Promise.resolve('complete'),
    expectedRecorder,
    'normal close',
  ), 'complete');
  assert.equal(expectedRecorder.report().firstFatalObservation, null);

  const alreadyFatalRecorder = createStandaloneBrowserObservationRecorder();
  alreadyFatalRecorder.record('requestfailed', {
    url: 'http://127.0.0.1/module.js',
  }, { fatal: true });
  let operationStarted = false;
  await assert.rejects(guardStandaloneBrowserOperation(
    () => {
      operationStarted = true;
      return Promise.resolve();
    },
    alreadyFatalRecorder,
    'post-failure operation',
  ), /requestfailed/);
  assert.equal(operationStarted, false);
});

test('readiness snapshot has a global payload budget and a capture deadline', async () => {
  const captured = await captureStandaloneReadinessSnapshot({
    isClosed: () => false,
    evaluate: async () => ({
      schemaVersion: 1,
      kind: 'first-instance-standalone-readiness-snapshot-value',
      documentReadyState: 'interactive',
      benchmarkGlobalPresent: false,
    }),
  });
  assert.equal(captured.captureStatus, 'captured');
  assert.equal(captured.value.documentReadyState, 'interactive');
  assert.match(captured.valueSha256, /^[0-9a-f]{64}$/);
  assert.ok(captured.observedJsonByteLength <= captured.maximumJsonBytes);

  const huge = Object.fromEntries(Array.from(
    { length: STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticObjectKeys },
    (_, index) => [
      `field-${index}`,
      Array.from(
        { length: STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticArrayItems },
        () => 'x'.repeat(STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticStringCodeUnits),
      ),
    ],
  ));
  const globallyBounded = await captureStandaloneReadinessSnapshot({
    isClosed: () => false,
    evaluate: async () => huge,
  });
  assert.equal(globallyBounded.captureStatus, 'captured');
  assert.notEqual(globallyBounded.value, null);
  assert.ok(globallyBounded.observedJsonByteLength <= globallyBounded.maximumJsonBytes);
  assert.match(JSON.stringify(globallyBounded.value), /diagnostic budget exhausted/);
  assert.match(globallyBounded.valueSha256, /^[0-9a-f]{64}$/);

  const deadline = await captureStandaloneReadinessSnapshot({
    isClosed: () => false,
    evaluate: () => new Promise(() => {}),
  }, { timeoutMs: 1 });
  assert.equal(deadline.captureStatus, 'deadline-exceeded');
  assert.equal(deadline.value, null);

  const unavailable = await captureStandaloneReadinessSnapshot({
    isClosed: () => true,
  });
  assert.equal(unavailable.captureStatus, 'page-unavailable');
});

test('terminal failure envelope is versioned independently from successful artifacts', () => {
  const error = new TypeError('readiness failed');
  error.stack = 's'.repeat(
    STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticStringCodeUnits + 100,
  );
  const diagnostics = {
    schemaVersion: 1,
    kind: 'first-instance-standalone-active-browser-failure-diagnostics',
    observations: { firstFatalObservation: { kind: 'pageerror' } },
  };
  const failure = createStandaloneTerminalFailureEnvelope({
    error,
    activeBrowserFailureDiagnostics: diagnostics,
  });
  assert.equal(failure.schemaVersion, 2);
  assert.equal(failure.kind, 'first-instance-standalone-deployment-capture-failure');
  assert.equal(failure.signal, null);
  assert.equal(failure.error.name, 'TypeError');
  assert.equal(failure.error.message, 'readiness failed');
  assert.match(failure.error.stack, /code units omitted/);
  assert.ok(failure.error.stack.length
    <= STANDALONE_FAILURE_DIAGNOSTIC_LIMITS.diagnosticStringCodeUnits);
  assert.deepEqual(failure.activeBrowserFailureDiagnostics, diagnostics);

  const interruption = createStandaloneTerminalFailureEnvelope({
    terminationSignal: 'SIGINT',
    error,
    activeBrowserFailureDiagnostics: diagnostics,
  });
  assert.equal(interruption.schemaVersion, 2);
  assert.equal(
    interruption.kind,
    'first-instance-standalone-deployment-capture-interruption',
  );
  assert.equal(interruption.signal, 'SIGINT');
});

test('execution selection preserves the full plan and freezes a two-session P/F smoke prefix', () => {
  const full = selectStandaloneDeploymentExecution(plan, STANDALONE_EXECUTION_MODES.FULL);
  assert.equal(full.analysisEligible, true);
  assert.equal(full.matrices.length, 2);
  assert.equal(full.sessions.length, 96);
  assert.equal(full.trials.length, 192);

  const smoke = selectStandaloneDeploymentExecution(plan, STANDALONE_EXECUTION_MODES.SMOKE);
  assert.equal(smoke.analysisEligible, false);
  assert.equal(smoke.matrices.length, 1);
  assert.deepEqual(smoke.sessions.map(({ assignedLaneId }) => assignedLaneId), [
    'portable',
    'feature',
  ]);
  assert.equal(smoke.trials.length, 4);
  assert.deepEqual(smoke.trials.map(({ planIndex }) => planIndex), [0, 1, 2, 3]);
  assert.deepEqual(smoke.trials.map(({ visibilityExposure }) => visibilityExposure), [
    'first',
    'second',
    'first',
    'second',
  ]);
  assert.throws(
    () => selectStandaloneDeploymentExecution(plan, 'ad-hoc'),
    /Unknown standalone deployment execution mode/,
  );
});

test('audit context and trial artifact retain every canonical analyzer identity field', () => {
  const context = createStandaloneAuditContext({
    runId: RUN_ID,
    canonicalTrial,
    executionMode: STANDALONE_EXECUTION_MODES.SMOKE,
    planSha256: PLAN_SHA256,
    browserRecord: {
      browserInstanceSerial: 2,
      sessionNamespace: `${canonicalTrial.sessionId}:browser-2`,
      profilePolicy: 'fresh-playwright-temporary-profile-per-process',
    },
  });
  for (const [field, value] of Object.entries(canonicalTrial)) {
    assert.deepEqual(context[field], value, `context field ${field}`);
  }
  assert.equal(context.schemaVersion, 1);
  const simulatedPageRow = { ...context, harnessContextSchemaVersion: 2 };
  for (const [field, value] of Object.entries(canonicalTrial)) {
    assert.deepEqual(simulatedPageRow[field], value, `row field ${field}`);
  }

  const rows = acceptedRows();
  const artifact = createStandaloneTrialArtifact({
    canonicalTrial,
    rows,
    executionMode: STANDALONE_EXECUTION_MODES.SMOKE,
    planSha256: PLAN_SHA256,
    body: { capturedAt: '2026-01-01T00:00:00.000Z' },
  });
  for (const [field, value] of Object.entries(canonicalTrial)) {
    assert.deepEqual(artifact[field], value, `artifact field ${field}`);
  }
  assert.equal(artifact.rows, rows);
  assert.deepEqual(artifact.canonicalTrial, canonicalTrial);
  assert.equal(artifact.analysisEligible, false);
  assert.throws(() => createStandaloneTrialArtifact({
    canonicalTrial,
    rows,
    executionMode: STANDALONE_EXECUTION_MODES.SMOKE,
    planSha256: PLAN_SHA256,
    body: { trialId: 'replacement' },
  }), /may not replace reserved field trialId/);
});

test('shader challenges cover ordinals 1-12 across the reused two-trial session', () => {
  const usedNonces = new Set();
  let nonce = 0;
  const nonceFactory = () => `${(++nonce).toString(16).padStart(64, '0')}`;
  const first = createStandaloneShaderObservationChallenges({
    canonicalTrial: plan.trials[0],
    runId: RUN_ID,
    firstCaptureOrdinal: 1,
    usedNonces,
    nonceFactory,
  });
  const second = createStandaloneShaderObservationChallenges({
    canonicalTrial: plan.trials[1],
    runId: RUN_ID,
    firstCaptureOrdinal: 7,
    usedNonces,
    nonceFactory,
  });
  assert.deepEqual([...first, ...second].map(({ captureOrdinal }) => captureOrdinal),
    Array.from({ length: 12 }, (_, index) => index + 1));
  assert.equal(usedNonces.size, 12);
  assert.ok([...first, ...second].every(
    ({ kind }) => kind === 'live-first-instance-standalone-shader-observation-challenge',
  ));
});

test('timing-row validation accepts the frozen 480-row shape and fails closed on drift', () => {
  const rows = acceptedRows();
  const summary = acceptedSummary();
  assert.deepEqual(validateStandaloneTimingRows(rows, canonicalTrial, summary).reasons, []);
  assert.equal(validateStandaloneTimingRows(rows, canonicalTrial, summary).pass, true);

  const changed = structuredClone(rows);
  changed[17].assignedLaneId = 'feature';
  changed[18].gpuPassTotalMs = 2;
  const invalid = validateStandaloneTimingRows(changed, canonicalTrial, summary);
  assert.equal(invalid.pass, false);
  assert.ok(invalid.reasons.some((reason) => reason.includes('canonical context differs')));
  assert.ok(invalid.reasons.some((reason) => reason.includes('GPU duration is invalid')));

  const timestampCorruption = structuredClone(rows);
  timestampCorruption[19].gpuRenderTimestampRecords = '[]';
  const invalidTimestamp = validateStandaloneTimingRows(
    timestampCorruption,
    canonicalTrial,
    summary,
  );
  assert.equal(invalidTimestamp.pass, false);
  assert.ok(invalidTimestamp.reasons.some(
    (reason) => reason.includes('timestamp attribution is invalid'),
  ));

  const boundaryGap = acceptedSummary();
  for (const type of ['render', 'compute']) {
    boundaryGap.timestampPhases.measurement.pools[type].frames =
      boundaryGap.timestampPhases.measurement.pools[type].frames.map((frame) => frame + 1);
  }
  const invalidBoundary = validateStandaloneTimingRows(rows, canonicalTrial, boundaryGap);
  assert.equal(invalidBoundary.pass, false);
  assert.ok(invalidBoundary.reasons.some((reason) => reason.includes('continuous GPU-frame')));

  const divergentTypes = acceptedSummary();
  divergentTypes.timestampPhases.measurement.pools.compute.frames[0] += 1;
  const invalidTypes = validateStandaloneTimingRows(rows, canonicalTrial, divergentTypes);
  assert.equal(invalidTypes.pass, false);
  assert.ok(invalidTypes.reasons.some((reason) => reason.includes('compute/render')));

  const wrongTopology = acceptedSummary();
  wrongTopology.timestampResolutionTopology.postMeasurementResolveBatchCount = 2;
  const invalidTopology = validateStandaloneTimingRows(rows, canonicalTrial, wrongTopology);
  assert.equal(invalidTopology.pass, false);
  assert.ok(invalidTopology.reasons.some((reason) => reason.includes('one post-measurement')));

  const insufficientCapacity = acceptedSummary();
  insufficientCapacity.timestampResolutionTopology.poolsAtStart.render.maxQueries = 1_599;
  insufficientCapacity.completionInvariant.timestampPoolsAtTimingStart.render.maxQueries = 1_599;
  const invalidCapacity = validateStandaloneTimingRows(
    rows,
    canonicalTrial,
    insufficientCapacity,
  );
  assert.equal(invalidCapacity.pass, false);
  assert.ok(invalidCapacity.reasons.some((reason) => reason.includes('render timestamp pool')));

  const missingCapacity = acceptedSummary();
  delete missingCapacity.timestampResolutionTopology.poolsAtStart.render.maxQueries;
  const invalidMissingCapacity = validateStandaloneTimingRows(
    rows,
    canonicalTrial,
    missingCapacity,
  );
  assert.equal(invalidMissingCapacity.pass, false);
  assert.ok(invalidMissingCapacity.reasons.some(
    (reason) => reason.includes('render timestamp pool'),
  ));

  const detachedResolvedUid = acceptedSummary();
  detachedResolvedUid.timestampResolutionTopology
    .poolsAfterPostMeasurementResolve.compute.timestampUids[0] = 'detached';
  const invalidResolvedUid = validateStandaloneTimingRows(
    rows,
    canonicalTrial,
    detachedResolvedUid,
  );
  assert.equal(invalidResolvedUid.pass, false);
  assert.ok(invalidResolvedUid.reasons.some(
    (reason) => reason.includes('compute timestamp pool'),
  ));
});

test('reused session timestamp pools join trial one to a clean trial-two start', () => {
  const firstSummary = acceptedSummary();
  const secondStart = structuredClone(
    firstSummary.timestampResolutionTopology.poolsAfterPostMeasurementResolve,
  );
  secondStart.backendTrackingEnabled = false;
  const records = [
    {
      sessionId: 'session-1',
      visibilityOrderPosition: 0,
      timestampPoolsAtStart: firstSummary.timestampResolutionTopology.poolsAtStart,
      timestampPoolsAtEnd:
        firstSummary.timestampResolutionTopology.poolsAfterPostMeasurementResolve,
    },
    {
      sessionId: 'session-1',
      visibilityOrderPosition: 1,
      timestampPoolsAtStart: secondStart,
      timestampPoolsAtEnd:
        firstSummary.timestampResolutionTopology.poolsAfterPostMeasurementResolve,
    },
  ];
  assert.deepEqual(validateStandaloneSessionTimestampContinuity(records), {
    schemaVersion: 1,
    kind: 'first-instance-standalone-runner-session-timestamp-continuity-validation',
    pass: true,
    reasons: [],
  });

  const detached = structuredClone(records);
  detached[1].timestampPoolsAtStart.compute.timestampUids.pop();
  detached[1].timestampPoolsAtStart.compute.timestampUidCount -= 1;
  const invalidHistory = validateStandaloneSessionTimestampContinuity(detached);
  assert.equal(invalidHistory.pass, false);
  assert.ok(invalidHistory.reasons.some(
    (reason) => reason.includes('compute timestamp history transition'),
  ));

  const trackingStillEnabled = structuredClone(records);
  trackingStillEnabled[1].timestampPoolsAtStart.backendTrackingEnabled = true;
  const invalidTracking = validateStandaloneSessionTimestampContinuity(trackingStillEnabled);
  assert.equal(invalidTracking.pass, false);
  assert.ok(invalidTracking.reasons.some(
    (reason) => reason.includes('tracking did not reset'),
  ));

  const dirtyButEqual = structuredClone(records);
  dirtyButEqual[0].timestampPoolsAtEnd.render.currentQueryIndex = 2;
  dirtyButEqual[1].timestampPoolsAtStart.render.currentQueryIndex = 2;
  const invalidCleanState = validateStandaloneSessionTimestampContinuity(dirtyButEqual);
  assert.equal(invalidCleanState.pass, false);
  assert.ok(invalidCleanState.reasons.some(
    (reason) => reason.includes('render timestamp history did not enter trial two cleanly'),
  ));

  assert.equal(validateStandaloneSessionTimestampContinuity(records.slice(0, 1)).pass, false);
});

test('browser lifecycle validation requires fresh profiles and every two-second gap', () => {
  const records = Array.from({ length: 3 }, (_, index) => ({
    browserInstanceSerial: index + 1,
    launchApi: 'chromium.launch',
    persistentContext: false,
    profilePolicy: 'fresh-playwright-temporary-profile-per-process',
    contextCountBeforeClose: 1,
    pageCountBeforeClose: 1,
    disconnectedEventCount: 1,
    disconnectedAt: `2026-01-01T00:00:0${index}.000Z`,
    disconnectedRunElapsedMs: index * 3_000 + 1_000,
    priorBrowserInstanceSerial: index === 0 ? null : index,
    priorBrowserDisconnectedAt: index === 0
      ? null
      : `2026-01-01T00:00:0${index - 1}.000Z`,
    previousDisconnectToLaunchGapMs: index === 0 ? null : 2_001,
    postDisconnectDelay: {
      requestedMs: STANDALONE_POST_DISCONNECT_DELAY_MS,
      elapsedMs: 2_001,
    },
  }));
  assert.equal(validateStandaloneBrowserLifecycleChain(
    records,
    { requireTerminalDelay: true },
  ).pass, true);

  const early = structuredClone(records);
  early[1].previousDisconnectToLaunchGapMs = 1_999.9;
  assert.equal(validateStandaloneBrowserLifecycleChain(
    early,
    { requireTerminalDelay: true },
  ).pass, false);
  const reusedProfile = structuredClone(records);
  reusedProfile[2].persistentContext = true;
  assert.equal(validateStandaloneBrowserLifecycleChain(reusedProfile).pass, false);
});

test('completed environment gate failures remain explicit full-capture outcomes', () => {
  const passingMatrix = (matrixOrdinal) => ({
    matrixOrdinal,
    identity: { environmentIdentityExact: true },
    telemetry: { eligibility: { pass: true, reasons: [] } },
  });
  const matrices = [passingMatrix(1), passingMatrix(2)];
  assert.equal(standaloneFullEnvironmentGatesPassed(
    { environmentIdentityExact: true, environmentDrift: [] },
    matrices,
  ), true);

  assert.equal(standaloneFullEnvironmentGatesPassed(
    { environmentIdentityExact: false, environmentDrift: [{ label: 'between matrices' }] },
    matrices,
  ), false);

  const matrixDrift = structuredClone(matrices);
  matrixDrift[0].identity.environmentIdentityExact = false;
  assert.equal(standaloneFullEnvironmentGatesPassed(
    { environmentIdentityExact: true, environmentDrift: [] },
    matrixDrift,
  ), false);

  const telemetryFailure = structuredClone(matrices);
  telemetryFailure[1].telemetry.eligibility = {
    pass: false,
    reasons: ['external GPU compute-process identity set changed during matrix'],
  };
  assert.equal(standaloneFullEnvironmentGatesPassed(
    { environmentIdentityExact: true, environmentDrift: [] },
    telemetryFailure,
  ), false);

  assert.throws(() => standaloneFullEnvironmentGatesPassed(
    { environmentIdentityExact: true, environmentDrift: [] },
    matrices.slice(0, 1),
  ), RangeError);
  assert.throws(() => standaloneFullEnvironmentGatesPassed(null, matrices), TypeError);
  assert.throws(() => standaloneFullEnvironmentGatesPassed({}, null), TypeError);
  assert.throws(() => standaloneFullEnvironmentGatesPassed(
    { environmentIdentityExact: true },
    [{}, passingMatrix(2)],
  ), TypeError);
});
