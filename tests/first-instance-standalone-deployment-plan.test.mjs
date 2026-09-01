import assert from 'node:assert/strict';
import test from 'node:test';

import {
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_LANE_ORDERS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_QUARTET_CODES,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_MATRIX,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_MATRIX,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_MASKS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
  buildFirstInstanceStandaloneDeploymentPlan,
  validateFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';

const RUN_ID = 'first-instance-standalone-deployment-test';

function countBy(rows, field) {
  return Object.fromEntries([...new Set(rows.map((row) => row[field]))].map(
    (value) => [value, rows.filter((row) => row[field] === value).length],
  ));
}

test('standalone deployment constants freeze the exact two-matrix design', () => {
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
    'first-instance-standalone-deployment-candidate');
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT, 2);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX, 12);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_MATRIX, 48);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_MATRIX, 96);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT, 96);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT, 192);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES, 320);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES, 480);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE, 8);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS, 60);
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS, [0.99, 0.2]);
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_LANE_ORDERS, {
    A: ['portable', 'feature', 'feature', 'portable'],
    B: ['feature', 'portable', 'portable', 'feature'],
  });
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_MASKS, {
    X: ['H', 'H', 'L', 'L'],
    Y: ['L', 'L', 'H', 'H'],
  });
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .highVisibilityGpuPassTotal.maximumDeltaMs, -0.10);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .highVisibilityGpuPassTotal.maximumDeltaPercent, -5);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .highVisibilityGpuPassTotal.minimumNegativeQuartets, 10);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .highVisibilityGpuRender.maximumDeltaMs, -0.10);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .highVisibilityGpuRender.maximumDeltaPercent, -5);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .lowVisibilityGpuPassTotal.strictUpperDeltaMs, 0.02);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .lowVisibilityGpuPassTotal.strictUpperDeltaPercent, 5);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .pairedHighMinusLowGpuPassTotal.maximumDeltaMs, -0.05);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .nuisanceInteraction.strictMaximumAbsoluteDeltaMs, 0.10);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .nuisanceInteraction.strictMaximumAbsoluteDeltaPercentagePoints, 5);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .conditionBlindDrift.strictMaximumAbsoluteDeltaMs, 0.10);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .conditionBlindDrift.strictMaximumAbsoluteDeltaPercent, 5);
  assert.equal(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS
    .maximumTimestampQuantumNs, 1_000);
  assert.equal(Object.isFrozen(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS), true);
  assert.equal(Object.isFrozen(
    FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS.highVisibilityGpuPassTotal,
  ), true);
});

test('matrix two uses the exact reverse quartet ordering of matrix one', () => {
  const expectedMatrixOne = [
    'AX', 'AY', 'BX', 'BY',
    'AX', 'AY', 'BX', 'BY',
    'AX', 'AY', 'BX', 'BY',
  ];
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_QUARTET_CODES[0],
    expectedMatrixOne);
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_QUARTET_CODES[1],
    [...expectedMatrixOne].reverse());

  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  assert.deepEqual(plan.matrices[0].quartetCodes, expectedMatrixOne);
  assert.deepEqual(plan.matrices[1].quartetCodes, [...expectedMatrixOne].reverse());
});

test('every matrix has exact quartet, lane, visibility, session, and trial balance', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  assert.equal(plan.matrices.length, 2);
  assert.equal(plan.quartets.length, 24);
  assert.equal(plan.sessions.length, 96);
  assert.equal(plan.trials.length, 192);

  for (let matrixIndex = 0; matrixIndex < 2; matrixIndex += 1) {
    const quartets = plan.quartets.filter((row) => row.matrixIndex === matrixIndex);
    const sessions = plan.sessions.filter((row) => row.matrixIndex === matrixIndex);
    const trials = plan.trials.filter((row) => row.matrixIndex === matrixIndex);
    assert.equal(quartets.length, 12);
    assert.equal(sessions.length, 48);
    assert.equal(trials.length, 96);
    assert.deepEqual(countBy(quartets, 'quartetCode'), { AX: 3, AY: 3, BX: 3, BY: 3 });
    assert.deepEqual(countBy(quartets, 'laneOrderId'), { A: 6, B: 6 });
    assert.deepEqual(countBy(quartets, 'visibilityMaskId'), { X: 6, Y: 6 });
    assert.deepEqual(countBy(sessions, 'assignedLaneId'), { portable: 24, feature: 24 });
    assert.deepEqual(countBy(sessions, 'visibilityOrderId'), { H: 24, L: 24 });
    assert.deepEqual(countBy(trials, 'assignedLaneId'), { portable: 48, feature: 48 });
    assert.equal(trials.filter((trial) => trial.visibilityFraction === 0.99).length, 48);
    assert.equal(trials.filter((trial) => trial.visibilityFraction === 0.2).length, 48);

    for (const quartet of quartets) {
      const quartetSessions = sessions.filter((row) => row.quartetId === quartet.quartetId);
      assert.equal(quartetSessions.length, 4);
      assert.deepEqual(countBy(quartetSessions, 'assignedLaneId'), {
        portable: 2,
        feature: 2,
      });
      for (const lane of ['portable', 'feature']) {
        const laneSessions = quartetSessions.filter((row) => row.assignedLaneId === lane);
        assert.deepEqual(countBy(laneSessions, 'visibilityOrderId'), { H: 1, L: 1 });
      }
    }
  }
});

test('every session is a unique fresh-browser assignment with one reused selected lane', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  assert.equal(new Set(plan.sessions.map(({ sessionId }) => sessionId)).size, 96);
  assert.equal(new Set(plan.trials.map(({ trialId }) => trialId)).size, 192);
  assert.deepEqual(plan.trials.map(({ planIndex }) => planIndex),
    Array.from({ length: 192 }, (_, index) => index));

  for (const session of plan.sessions) {
    assert.equal(session.freshBrowserProcessRequired, true);
    assert.equal(session.freshPageContextRendererAdapterDeviceRequired, true);
    assert.equal(session.previousBrowserDisconnectRequired, true);
    assert.equal(session.selectedLaneConstructionCount, 1);
    assert.equal(session.absentLaneConstructionCount, 0);
    assert.equal(session.reuseSelectedLaneAcrossVisibilityTrialsRequired, true);
    assert.equal(session.trialIds.length, 2);
    const trials = plan.trials.filter((trial) => trial.sessionId === session.sessionId);
    assert.equal(trials.length, 2);
    assert.deepEqual(trials.map(({ trialId }) => trialId), session.trialIds);
    assert.ok(trials.every(({ assignedLaneId }) => assignedLaneId === session.assignedLaneId));
    assert.ok(trials.every(({ absentLaneId }) => absentLaneId === session.absentLaneId));
    assert.deepEqual(trials.map(({ visibilityFraction }) => visibilityFraction),
      session.visibilityOrder);
    assert.deepEqual(trials.map(({ visibilityExposure }) => visibilityExposure),
      ['first', 'second']);
    assert.ok(trials.every(({ warmupFrames }) => warmupFrames === 320));
    assert.ok(trials.every(({ measuredFrames }) => measuredFrames === 480));
    assert.ok(trials.every(({ measuredBlockSize }) => measuredBlockSize === 8));
    assert.ok(trials.every(({ measuredBlockCount }) => measuredBlockCount === 60));
  }
});

test('plan construction is deterministic, deeply immutable, and rejects invalid input', () => {
  const first = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const second = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  assert.deepEqual(first, second);
  assert.equal(validateFirstInstanceStandaloneDeploymentPlan(first, { runId: RUN_ID }), true);
  assert.equal(Object.isFrozen(first), true);
  assert.ok(first.matrices.every(Object.isFrozen));
  assert.ok(first.quartets.every(Object.isFrozen));
  assert.ok(first.sessions.every(Object.isFrozen));
  assert.ok(first.trials.every(Object.isFrozen));
  assert.ok(first.sessions.every((session) => Object.isFrozen(session.visibilityOrder)));
  assert.ok(first.trials.every((trial) => Object.isFrozen(trial.visibilityOrder)));

  assert.throws(() => buildFirstInstanceStandaloneDeploymentPlan(), /must be an object/);
  assert.throws(
    () => buildFirstInstanceStandaloneDeploymentPlan({ runId: '' }),
    /nonempty string/,
  );
  assert.throws(
    () => buildFirstInstanceStandaloneDeploymentPlan({ runId: '../escape' }),
    /portable identifier/,
  );
  assert.throws(
    () => buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID, matrixCount: 1 }),
    /accepts exactly the runId option/,
  );
  assert.throws(
    () => validateFirstInstanceStandaloneDeploymentPlan(null, { runId: RUN_ID }),
    /must be an object/,
  );

  const changedLane = structuredClone(first);
  changedLane.sessions[0].assignedLaneId = 'feature';
  assert.throws(
    () => validateFirstInstanceStandaloneDeploymentPlan(changedLane, { runId: RUN_ID }),
    /differs at plan\.sessions\[0\]\.assignedLaneId/,
  );

  const extraField = structuredClone(first);
  extraField.trials[0].unregistered = true;
  assert.throws(
    () => validateFirstInstanceStandaloneDeploymentPlan(extraField, { runId: RUN_ID }),
    /differs at plan\.trials\[0\] keys/,
  );
  assert.throws(
    () => validateFirstInstanceStandaloneDeploymentPlan(first, { runId: 'different-run' }),
    /differs at plan\.runId/,
  );
});
