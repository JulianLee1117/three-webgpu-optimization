import assert from 'node:assert/strict';
import test from 'node:test';

import {
  STANDALONE_EXECUTION_MODES,
  STANDALONE_POST_DISCONNECT_DELAY_MS,
  createStandaloneAuditContext,
  createStandaloneShaderObservationChallenges,
  createStandaloneTrialArtifact,
  selectStandaloneDeploymentExecution,
  standaloneFullEnvironmentGatesPassed,
  validateStandaloneBrowserLifecycleChain,
  validateStandaloneTimingRows,
} from '../scripts/run-first-instance-standalone-deployment.mjs';
import {
  buildFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';

const RUN_ID = 'standalone-runner-unit';
const PLAN_SHA256 = 'a'.repeat(64);
const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
const canonicalTrial = plan.trials[0];

function timestampPhase(frameCount) {
  const pool = (type) => ({
    included: true,
    frames: Array.from({ length: frameCount }, (_, index) => index),
    uidRecords: Array.from({ length: frameCount }, (_, index) => ({
      uid: `${type}:${index}`,
    })),
    resolution: { quantumNs: 1 },
  });
  return { pools: { compute: pool('compute'), render: pool('render') } };
}

function acceptedSummary() {
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
    completionInvariant: { pass: true },
    renderTimestampPoolQualityValid: true,
    computeTimestampPoolQualityValid: true,
    warmupRenderTimestampPoolQualityValid: true,
    warmupComputeTimestampPoolQualityValid: true,
    warmupTimestampFrameCountValid: true,
    measurementTimestampFrameCountValid: true,
    quantumNs: 1,
    timestampPhases: {
      warmup: timestampPhase(320),
      measurement: timestampPhase(480),
    },
  };
}

function acceptedRows(trial = canonicalTrial) {
  const cpuComputeSubmitMs = 0.1;
  const cpuRenderSubmitMs = 0.2;
  return Array.from({ length: 480 }, (_, index) => {
    const computeUid = `c:1:0:f${index + 1}`;
    const renderUid = `r:1:0:f${index + 1}`;
    return {
      ...trial,
      runId: RUN_ID,
      harnessContextSchemaVersion: 2,
      frameIndex: index,
      phaseFrameIndex: index,
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
