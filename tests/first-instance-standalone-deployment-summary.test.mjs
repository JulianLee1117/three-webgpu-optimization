import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_CPU_TIMING_FIELDS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS,
  summarizeFirstInstanceStandaloneDeployment,
  summarizeFirstInstanceStandaloneDeploymentTrialRecord,
} from '../analysis/first-instance-standalone-deployment-summary.mjs';
import {
  buildFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';

const RUN_ID = 'standalone-analyzer-test';

function closeTo(actual, expected, tolerance = 1e-10) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} is not within ${tolerance} of ${expected}`,
  );
}

function timingFor(trial, frameIndex, {
  highFeatureEffect = ({ matrixIndex }) => (matrixIndex === 0 ? -0.2 : -0.2),
  lowFeatureEffect = () => 0,
  blockRenderOffset = () => 0,
} = {}) {
  const isFeature = trial.assignedLaneId === 'feature';
  const effect = !isFeature
    ? 0
    : trial.visibilityFraction === 0.99
      ? highFeatureEffect(trial)
      : lowFeatureEffect(trial);
  const gpuComputeMs = 0.4;
  const gpuRenderMs = 1.6 + effect + blockRenderOffset({
    trial,
    frameIndex,
    blockIndex: Math.floor(frameIndex / 8),
  });
  const cpuComputeSubmitMs = 0.04;
  const cpuRenderSubmitMs = 0.08;
  return {
    gpuPassTotalMs: gpuComputeMs + gpuRenderMs,
    gpuRenderMs,
    gpuComputeMs,
    cpuCommonUpdateMs: 0.02,
    cpuComputeSubmitMs,
    cpuRenderSubmitMs,
    cpuSubmitTotalMs: cpuComputeSubmitMs + cpuRenderSubmitMs,
    cpuFrameBodyMs: 0.2,
  };
}

function createTrialRecord(plan, trial, timingOptions) {
  const rows = Array.from({ length: 480 }, (_, frameIndex) => ({
    ...trial,
    runId: plan.runId,
    frameIndex,
    phaseFrameIndex: frameIndex,
    ...timingFor(trial, frameIndex, timingOptions),
  }));
  return { ...trial, rows };
}

function createRecords(plan, timingOptions) {
  return plan.trials.map((trial) => createTrialRecord(plan, trial, timingOptions));
}

function cloneOneRecord(record) {
  return {
    ...record,
    visibilityOrder: [...record.visibilityOrder],
    rows: record.rows.map((row) => ({
      ...row,
      visibilityOrder: [...row.visibilityOrder],
    })),
  };
}

test('standalone analyzer exports the explicit identity and timing contract', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  assert.deepEqual(
    FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS,
    Object.keys(plan.trials[0]),
  );
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS, [
    'gpuPassTotalMs',
    'gpuRenderMs',
    'gpuComputeMs',
  ]);
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_CPU_TIMING_FIELDS, [
    'cpuCommonUpdateMs',
    'cpuComputeSubmitMs',
    'cpuRenderSubmitMs',
    'cpuSubmitTotalMs',
    'cpuFrameBodyMs',
  ]);
  assert.deepEqual(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS, [
    ...FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS,
    ...FIRST_INSTANCE_STANDALONE_DEPLOYMENT_CPU_TIMING_FIELDS,
  ]);
});

test('single-trial reduction uses the conventional median of 60 eight-frame means', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const record = createTrialRecord(plan, plan.trials[0], {
    blockRenderOffset: ({ blockIndex }) => blockIndex,
  });
  const trial = summarizeFirstInstanceStandaloneDeploymentTrialRecord(plan, record);

  assert.equal(trial.nRows, 480);
  assert.equal(trial.nBlocks, 60);
  assert.deepEqual(
    trial.blocks.map((block) => [block.startFrameIndex, block.endFrameIndex]),
    Array.from({ length: 60 }, (_, index) => [index * 8, index * 8 + 7]),
  );
  closeTo(trial.estimates.gpuRenderMs, 1.6 + 29.5);
  closeTo(trial.estimates.gpuComputeMs, 0.4);
  closeTo(trial.estimates.gpuPassTotalMs, 2 + 29.5);
  closeTo(trial.drift.gpuRenderMs.deltaMs, 45);
});

test('candidate analysis keeps matrices separate and evaluates every frozen numerical gate', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const records = createRecords(plan);
  const passing = summarizeFirstInstanceStandaloneDeployment(plan, records);

  assert.equal(passing.kind, 'first-instance-standalone-deployment-summary');
  assert.equal(passing.analysisMode, 'candidate');
  assert.equal(passing.nTrialRecords, 192);
  assert.equal(passing.nRows, 92_160);
  assert.equal(passing.trials.length, 192);
  assert.equal(passing.sessions.length, 96);
  assert.equal(passing.quartets.length, 24);
  assert.equal(passing.matrices.length, 2);
  assert.equal(passing.protocol.inferentialUnit, 'quartet');
  assert.equal(passing.protocol.framesSessionsAndTrialsAreInferentialUnits, false);

  for (const matrix of passing.matrices) {
    const high = matrix.estimates['0.99'];
    const low = matrix.estimates['0.2'];
    closeTo(high.gpuPassTotalMs.median.deltaMs, -0.2);
    closeTo(high.gpuPassTotalMs.median.deltaPercent, -10);
    assert.equal(high.gpuPassTotalMs.negativeCount, 12);
    closeTo(high.gpuRenderMs.median.deltaMs, -0.2);
    closeTo(high.gpuRenderMs.median.deltaPercent, -12.5);
    assert.equal(high.gpuRenderMs.negativeCount, 12);
    closeTo(low.gpuPassTotalMs.median.deltaMs, 0);
    closeTo(matrix.pairedHighMinusLow.median.deltaMs, -0.2);
    for (const factor of Object.values(matrix.highVisibilityFactors)) {
      assert.equal(factor.strata.every((stratum) => stratum.median.deltaMs < 0), true);
      closeTo(factor.interaction.deltaMs, 0);
      closeTo(factor.interaction.deltaPercentagePoints, 0);
    }
    closeTo(
      matrix.drift.conditionBlindSessionSequence.overall.gpuPassTotalMs.median.deltaMs,
      0,
    );
    closeTo(
      matrix.drift.conditionBlindWithinTrial.overall.gpuPassTotalMs.median.deltaMs,
      0,
    );
  }
  assert.equal(passing.preregisteredNumericalDecision.pass, true);
  assert.equal(
    passing.preregisteredNumericalDecision.numericalVerdict,
    'preregistered-numerical-gates-met',
  );
  assert.equal(passing.preregisteredNumericalDecision.standaloneProtocolVerdict, null);
  assert.equal(
    passing.preregisteredNumericalDecision.matrixDecisions.every(
      (decision) => decision.pass,
    ),
    true,
  );

  for (const record of records) {
    if (record.matrixIndex !== 1
      || record.visibilityFraction !== 0.99
      || record.assignedLaneId !== 'feature') continue;
    for (const row of record.rows) {
      row.gpuRenderMs = 1.55;
      row.gpuPassTotalMs = row.gpuComputeMs + row.gpuRenderMs;
    }
  }
  const split = summarizeFirstInstanceStandaloneDeployment(plan, records);
  assert.equal(split.preregisteredNumericalDecision.pass, false);
  assert.equal(
    split.preregisteredNumericalDecision.standaloneProtocolVerdict,
    'standalone-confirmation-not-met',
  );
  assert.deepEqual(
    split.preregisteredNumericalDecision.matrixDecisions.map((decision) => decision.pass),
    [true, false],
  );
  assert.equal(
    split.preregisteredNumericalDecision.matrixDecisions[1].failedGates.includes(
      'highVisibilityGpuPassTotal',
    ),
    true,
  );
  assert.equal(
    split.preregisteredNumericalDecision.matrixDecisions[1].failedGates.includes(
      'highVisibilityGpuRender',
    ),
    true,
  );

  for (const record of records) {
    if (record.assignedLaneId !== 'feature') continue;
    if (record.matrixIndex === 1 && record.visibilityFraction === 0.99) {
      for (const row of record.rows) {
        row.gpuRenderMs = 1.475;
        row.gpuPassTotalMs = row.gpuComputeMs + row.gpuRenderMs;
      }
    }
    if (record.visibilityFraction === 0.2) {
      for (const row of record.rows) {
        row.gpuRenderMs = 1.62;
        row.gpuPassTotalMs = row.gpuComputeMs + row.gpuRenderMs;
      }
    }
  }
  const boundaries = summarizeFirstInstanceStandaloneDeployment(plan, records);
  const secondDecision = boundaries.preregisteredNumericalDecision.matrixDecisions[1];
  assert.equal(secondDecision.gates.highVisibilityGpuPassTotal.pass, true);
  closeTo(
    secondDecision.gates.highVisibilityGpuPassTotal.observed.median.deltaMs,
    -0.125,
  );
  closeTo(
    secondDecision.gates.highVisibilityGpuPassTotal.observed.median.deltaPercent,
    -6.25,
  );
  for (const matrixDecision of boundaries.preregisteredNumericalDecision.matrixDecisions) {
    assert.equal(matrixDecision.gates.lowVisibilityGpuPassTotal.pass, false);
    closeTo(
      matrixDecision.gates.lowVisibilityGpuPassTotal.observed.median.deltaMs,
      0.02,
    );
  }
});

test('high-visibility endpoint direction gates accept exactly 10 of 12 and reject 9', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const summary = summarizeFirstInstanceStandaloneDeployment(
    plan,
    createRecords(plan, {
      highFeatureEffect: ({ matrixIndex, quartetIndex }) => {
        const negativeQuartets = matrixIndex === 0 ? 10 : 9;
        return quartetIndex < negativeQuartets ? -0.2 : 0;
      },
    }),
  );
  const [tenNegative, nineNegative] =
    summary.preregisteredNumericalDecision.matrixDecisions;

  for (const gateName of ['highVisibilityGpuPassTotal', 'highVisibilityGpuRender']) {
    const atMinimum = tenNegative.gates[gateName];
    assert.equal(atMinimum.observed.negativeCount, 10);
    assert.equal(atMinimum.pass, true);
    assert.ok(atMinimum.observed.median.deltaMs <= atMinimum.thresholds.maximumDeltaMs);
    assert.ok(
      atMinimum.observed.median.deltaPercent <= atMinimum.thresholds.maximumDeltaPercent,
    );

    const belowMinimum = nineNegative.gates[gateName];
    assert.equal(belowMinimum.observed.negativeCount, 9);
    assert.equal(belowMinimum.pass, false);
    assert.ok(belowMinimum.observed.median.deltaMs <= belowMinimum.thresholds.maximumDeltaMs);
    assert.ok(
      belowMinimum.observed.median.deltaPercent <= belowMinimum.thresholds.maximumDeltaPercent,
    );
  }
});

test('nuisance gates enforce strict interaction bounds and strict level negativity', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const summary = summarizeFirstInstanceStandaloneDeployment(
    plan,
    createRecords(plan, {
      highFeatureEffect: ({ matrixIndex, quartetCode }) => {
        const laneOrder = quartetCode[0];
        if (matrixIndex === 0) return laneOrder === 'A' ? -0.25 : -0.149999;
        return laneOrder === 'A' ? 0 : -0.05;
      },
    }),
  );
  const [interactionBoundaryDecision, levelBoundaryDecision] =
    summary.preregisteredNumericalDecision.matrixDecisions;
  const interactionBoundary = interactionBoundaryDecision.gates
    .highVisibilityNuisanceFactors.factors.sequence;
  assert.equal(interactionBoundary.levelsNegative, true);
  assert.equal(interactionBoundary.interactionWithinBounds, false);
  assert.equal(interactionBoundary.pass, false);
  closeTo(Math.abs(interactionBoundary.observed.interaction.deltaMs), 0.100001);
  closeTo(
    Math.abs(interactionBoundary.observed.interaction.deltaPercentagePoints),
    5.00005,
  );

  const levelBoundary = levelBoundaryDecision.gates
    .highVisibilityNuisanceFactors.factors.sequence;
  assert.equal(levelBoundary.observed.strata[0].level, 'A');
  closeTo(levelBoundary.observed.strata[0].median.deltaMs, 0);
  assert.equal(levelBoundary.levelsNegative, false);
  assert.equal(levelBoundary.interactionWithinBounds, true);
  assert.equal(levelBoundary.pass, false);
});

test('condition-blind drift gates straddle the strict 0.10 ms boundary', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const summary = summarizeFirstInstanceStandaloneDeployment(
    plan,
    createRecords(plan, {
      blockRenderOffset: ({ trial, blockIndex }) => {
        const drift = trial.matrixIndex === 0 ? 0.099999 : 0.100001;
        return 10
          + (trial.sessionPosition >= 2 ? drift : 0)
          + (blockIndex >= 45 ? drift : 0);
      },
    }),
  );

  for (const [matrixIndex, decision] of
    summary.preregisteredNumericalDecision.matrixDecisions.entries()) {
    const expectedDrift = matrixIndex === 0 ? 0.099999 : 0.100001;
    const expectedPass = matrixIndex === 0;
    for (const family of [
      decision.gates.conditionBlindSessionSequenceDrift,
      decision.gates.conditionBlindWithinTrialDrift,
    ]) {
      assert.equal(family.pass, expectedPass);
      closeTo(Math.abs(family.overall.observed.deltaMs), expectedDrift);
      assert.equal(family.overall.pass, expectedPass);
      for (const visibilityGate of Object.values(family.visibilities)) {
        closeTo(Math.abs(visibilityGate.observed.deltaMs), expectedDrift);
        assert.equal(visibilityGate.pass, expectedPass);
      }
    }
  }
});

test('smoke and partial modes are structurally excluded from decisions', () => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const first = createTrialRecord(plan, plan.trials[0]);
  const smokeRecords = plan.trials.slice(0, 4).map(
    (trial) => createTrialRecord(plan, trial),
  );
  const smoke = summarizeFirstInstanceStandaloneDeployment(plan, smokeRecords, {
    mode: 'smoke',
  });
  assert.equal(smoke.coverage.complete, false);
  assert.equal(smoke.decisionEligibility.eligible, false);
  assert.equal(smoke.preregisteredNumericalDecision.status, 'excluded');
  assert.equal(smoke.preregisteredNumericalDecision.pass, null);
  assert.equal(smoke.preregisteredNumericalDecision.matrixDecisions.length, 0);
  assert.equal(smoke.sessions.length, 0);
  assert.equal(smoke.trials.length, 4);

  assert.throws(
    () => summarizeFirstInstanceStandaloneDeployment(plan, [first], { mode: 'smoke' }),
    /exactly frozen plan indices 0, 1, 2, 3/,
  );
  assert.throws(
    () => summarizeFirstInstanceStandaloneDeployment(
      plan,
      [
        ...smokeRecords.slice(0, 3),
        createTrialRecord(plan, plan.trials[4]),
      ],
      { mode: 'smoke' },
    ),
    /exactly frozen plan indices 0, 1, 2, 3/,
  );

  const last = createTrialRecord(plan, plan.trials.at(-1));
  const partial = summarizeFirstInstanceStandaloneDeployment(plan, [first, last], {
    mode: 'partial',
  });
  assert.equal(partial.coverage.presentTrialCount, 2);
  assert.equal(partial.coverage.missingPlanIndices.length, 190);
  assert.equal(partial.preregisteredDecision.eligible, false);

  assert.throws(
    () => summarizeFirstInstanceStandaloneDeployment(
      plan,
      createRecords(plan),
      { mode: 'smoke' },
    ),
    /exactly frozen plan indices 0, 1, 2, 3/,
  );
});

test('standalone analyzer fails closed on counts, order, identity, and timing corruption', async (t) => {
  const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });
  const pristine = createTrialRecord(plan, plan.trials[0]);
  const cases = [
    [
      'missing measured row',
      (record) => record.rows.pop(),
      /expected exactly 480/,
    ],
    [
      'extra measured row',
      (record) => record.rows.push({ ...record.rows.at(-1), frameIndex: 480, phaseFrameIndex: 480 }),
      /expected exactly 480/,
    ],
    [
      'reordered measured rows',
      (record) => { [record.rows[0], record.rows[1]] = [record.rows[1], record.rows[0]]; },
      /frameIndex/,
    ],
    [
      'wrong row trial identity',
      (record) => { record.rows[0].sessionId = 'wrong-session'; },
      /sessionId/,
    ],
    [
      'missing row timing field',
      (record) => { delete record.rows[0].cpuFrameBodyMs; },
      /cpuFrameBodyMs is required/,
    ],
    [
      'non-finite GPU timing',
      (record) => { record.rows[0].gpuRenderMs = Number.NaN; },
      /gpuRenderMs must be a finite number/,
    ],
    [
      'non-finite CPU timing',
      (record) => { record.rows[0].cpuCommonUpdateMs = Number.POSITIVE_INFINITY; },
      /cpuCommonUpdateMs must be a finite number/,
    ],
    [
      'GPU total mismatch',
      (record) => { record.rows[0].gpuPassTotalMs += 0.001; },
      /must equal gpuComputeMs \+ gpuRenderMs/,
    ],
    [
      'CPU total mismatch',
      (record) => { record.rows[0].cpuSubmitTotalMs += 0.001; },
      /must equal cpuComputeSubmitMs \+ cpuRenderSubmitMs/,
    ],
  ];
  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const record = cloneOneRecord(pristine);
      mutate(record);
      assert.throws(
        () => summarizeFirstInstanceStandaloneDeployment(
          plan,
          [record],
          { mode: 'partial' },
        ),
        pattern,
      );
    });
  }

  assert.throws(
    () => summarizeFirstInstanceStandaloneDeployment(plan, [pristine]),
    /expected exactly 192/,
  );
  assert.throws(
    () => summarizeFirstInstanceStandaloneDeployment(
      plan,
      Array.from({ length: 193 }, () => pristine),
      { mode: 'partial' },
    ),
    /only 192/,
  );
  const second = createTrialRecord(plan, plan.trials[1]);
  assert.throws(
    () => summarizeFirstInstanceStandaloneDeployment(
      plan,
      [second, pristine],
      { mode: 'partial' },
    ),
    /strictly ordered|frozen plan/,
  );
});
