import assert from 'node:assert/strict';
import test from 'node:test';
import {
  summarizeLiveFirstInstanceCrossoverRows,
  summarizeLiveFirstInstanceTrialRows,
} from '../analysis/live-first-instance-crossover-summary.mjs';
import {
  firstInstanceLiveCrossoverFrame,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  buildFirstInstanceLiveCrossoverPlan,
} from '../src/benchmark/plan.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;

function closeTo(actual, expected, tolerance = 1e-10) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} is not within ${tolerance} of ${expected}`,
  );
}

function createRows({
  renderBaseline = () => 0.8,
  computeBaseline = () => 0.2,
  renderEffect = ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.16 : 0),
  computeEffect = () => 0,
} = {}) {
  const runId = 'live-first-instance-analysis';
  const plan = buildFirstInstanceLiveCrossoverPlan({
    runId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  });
  const rows = [];
  for (const trial of plan) {
    const selectorStart = trial.planIndex * 10_000;
    const strategyStart = trial.planIndex * 10_000 + 100;
    const computeStart = trial.planIndex * 10_000 + 200;
    const renderStart = trial.planIndex * 10_000 + 300;
    const expectedVisibleCount = Math.round(OBJECT_COUNT * trial.visibilityFraction);
    const commandIds = {};
    for (const [position, laneId] of trial.lanePhysicalOrder.entries()) {
      commandIds[laneId] = 1_000 + trial.planIndex * 10 + position;
    }
    for (let frameIndex = 0; frameIndex < 480; frameIndex += 1) {
      const scheduled = firstInstanceLiveCrossoverFrame(
        frameIndex,
        trial.superblockOrientationOffset,
      );
      const context = {
        trial,
        visibilityFraction: trial.visibilityFraction,
        repetitionIndex: trial.repetitionIndex,
        visibilityOrderPosition: trial.visibilityOrderPosition,
        portablePhysicalOrderPosition: trial.lanePhysicalOrder.indexOf(PORTABLE),
        orientationOffset: trial.superblockOrientationOffset,
        blockIndex: scheduled.crossoverBlockIndex,
        frameIndex,
        previousPreviousLaneId: scheduled.previousPreviousLaneId,
        previousLaneId: scheduled.previousLaneId,
        laneId: scheduled.laneId,
      };
      const isFeature = scheduled.laneId === FEATURE;
      const gpuComputeMs = computeBaseline(context) + (isFeature ? computeEffect(context) : 0);
      const gpuRenderMs = renderBaseline(context) + (isFeature ? renderEffect(context) : 0);
      const serialOffset = 320 + frameIndex + 1;
      rows.push({
        schemaVersion: 2,
        runId,
        trialId: trial.trialId,
        planIndex: trial.planIndex,
        repetitionIndex: trial.repetitionIndex,
        frameIndex,
        phaseFrameIndex: frameIndex,
        modeId: trial.modeId,
        modeOrderPosition: 0,
        plannedModeOrder: trial.modeId,
        targetVisibilityFraction: trial.visibilityFraction,
        plannedVisibilityOrder: trial.visibilityOrder.join('|'),
        visibilityOrderPosition: trial.visibilityOrderPosition,
        plannedLanePhysicalOrder: trial.lanePhysicalOrder.join('|'),
        superblockOrientationOffset: trial.superblockOrientationOffset,
        scenarioLayout: 'baseline',
        layoutOrderPosition: 0,
        plannedLayoutOrder: 'baseline',
        protocolWarmupFrames: 320,
        protocolMeasuredFrames: 480,
        objectCount: OBJECT_COUNT,
        bucketCount: BUCKET_COUNT,
        expectedVisibleCount,
        usesCompute: true,
        configuredDrawCommands: BUCKET_COUNT,
        configuredRenderObjects: 1,
        configuredComputeDispatches: 2,
        configuredComputeSubmissions: 1,
        configuredSubmittedInstances: expectedVisibleCount,
        validationPass: true,
        validationKind: 'first-instance-live-crossover-exact-paired-snapshots',
        timestampAvailable: true,
        gpuComputeTimestampUidCount: 1,
        expectedComputeTimestampUidCount: 1,
        gpuRenderTimestampUidCount: 1,
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
        commandBufferId: commandIds[scheduled.laneId],
        portableCommandBufferIdAtTimingStart: commandIds[PORTABLE],
        featureCommandBufferIdAtTimingStart: commandIds[FEATURE],
        commandRecordBase: 0,
        commandByteBase: 0,
        commandSegmentIndex: 0,
        selectorWriteSerialAtTimingStart: selectorStart,
        strategySelectionSerialAtTimingStart: strategyStart,
        computeCallSerialAtTimingStart: computeStart,
        renderCallSerialAtTimingStart: renderStart,
        selectorWriteSerial: selectorStart + serialOffset,
        strategySelectionSerial: strategyStart + serialOffset,
        computeCallSerial: computeStart + serialOffset,
        renderCallSerial: renderStart + serialOffset,
        gpuFrameId: trial.planIndex * 1_000 + frameIndex,
      });
    }
  }
  return rows;
}

test('live analyzer reconstructs exact component estimators and passes the frozen gates', () => {
  const summary = summarizeLiveFirstInstanceCrossoverRows(createRows());

  assert.equal(summary.kind, 'indirect-first-instance-live-crossover-summary');
  assert.equal(summary.primaryEndpoint, 'gpuPassTotalMs = gpuComputeMs + gpuRenderMs');
  assert.equal(summary.nRows, 11_520);
  assert.equal(summary.nTrials, 24);
  assert.equal(summary.trials.every((trial) => (
    trial.nRows === 480
    && trial.nBlocks === 60
    && trial.blocks.every((block) => (
      Number.isFinite(block.metrics.gpuPassTotal.deltaMs)
      && Number.isFinite(block.metrics.gpuRender.deltaMs)
      && Number.isFinite(block.metrics.gpuCompute.deltaMs)
      && block.pattern === (
        trial.superblockOrientationOffset === 0 ? 'PPPFPFFF' : 'FFFPFPPP'
      )
      && Object.values(block.historyBalance.transitionCounts).every((count) => count === 2)
      && Object.values(block.historyBalance.historyTripleCounts).every((count) => count === 1)
    ))
    && Object.values(trial.historyBalance.transitionCounts).every((count) => count === 120)
    && Object.values(trial.historyBalance.historyTripleCounts).every((count) => count === 60)
  )), true);

  const high = summary.contrasts.visibilities['0.99'];
  const low = summary.contrasts.visibilities['0.2'];
  closeTo(high.gpuPassTotal.median.deltaMs, -0.16);
  closeTo(high.gpuPassTotal.median.deltaPercent, -16);
  assert.equal(high.gpuPassTotal.negativeCount, 12);
  closeTo(high.gpuRender.median.deltaMs, -0.16);
  closeTo(high.gpuRender.median.deltaPercent, -20);
  closeTo(high.gpuCompute.median.deltaMs, 0);
  closeTo(low.gpuPassTotal.median.deltaMs, 0);
  closeTo(summary.contrasts.pairedHighMinusLow.median.deltaMs, -0.16);
  closeTo(summary.contrasts.drift.overall.median.deltaMs, 0);

  const decision = summary.preregisteredNumericalDecision;
  assert.equal(decision.pass, true);
  assert.deepEqual(decision.failedGates, []);
  assert.equal(decision.gates.highVisibilityTotalMaterial.pass, true);
  assert.equal(decision.gates.highVisibilityTotalDirection.pass, true);
  assert.equal(decision.gates.highVisibilityRender.pass, true);
  assert.equal(decision.gates.highVisibilityTotalStrata.pass, true);
  assert.equal(decision.gates.lowVisibilityTotalRegression.pass, true);
  assert.equal(decision.gates.pairedHighMinusLowTotal.pass, true);
  assert.equal(decision.gates.drift.pass, true);
});

test('single-trial helper returns the existing internal 480-row trial summary', () => {
  const rows = createRows();
  const runId = rows[0].runId;
  const expectedTrial = buildFirstInstanceLiveCrossoverPlan({
    runId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  })[0];
  const single = summarizeLiveFirstInstanceTrialRows(
    rows.slice(0, 480),
    expectedTrial,
    runId,
  );
  const matrix = summarizeLiveFirstInstanceCrossoverRows(rows);

  assert.deepEqual(single, matrix.trials[0]);
  assert.throws(
    () => summarizeLiveFirstInstanceTrialRows(rows.slice(0, 479), expectedTrial, runId),
    /expected exactly 480/,
  );
  assert.throws(
    () => summarizeLiveFirstInstanceTrialRows(
      rows.slice(0, 480),
      expectedTrial,
      'different-run',
    ),
    /must all use runId/,
  );
});

test('live estimator standardizes predecessor lanes and retains directed carryover diagnostics', () => {
  const summary = summarizeLiveFirstInstanceCrossoverRows(createRows({
    renderBaseline: ({ previousLaneId }) => (previousLaneId === PORTABLE ? 0.8 : 0.9),
    renderEffect: ({ previousLaneId, visibilityFraction }) => (
      visibilityFraction === 0.99
        ? previousLaneId === PORTABLE ? -0.16 : -0.14
        : 0
    ),
  }));
  const highTrial = summary.trials.find((trial) => trial.visibilityFraction === 0.99);
  closeTo(highTrial.estimates.gpuRender.deltaMs, -0.15);
  closeTo(
    highTrial.estimates.gpuRender.previousLaneStrata[PORTABLE].deltaMs,
    -0.16,
  );
  closeTo(
    highTrial.estimates.gpuRender.previousLaneStrata[FEATURE].deltaMs,
    -0.14,
  );
  const carryover = summary.preregisteredDecision.gates.highVisibilityCarryover;
  assert.equal(carryover.levelsNegative, true);
  assert.equal(carryover.interactionWithinBounds, true);
  assert.equal(carryover.pass, true);
});

test('high-visibility carryover gate rejects an aggregate win confined to one predecessor', () => {
  const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
    renderEffect: ({ previousLaneId, visibilityFraction }) => (
      visibilityFraction === 0.99
        ? previousLaneId === PORTABLE ? -0.3 : 0.01
        : 0
    ),
  })).preregisteredDecision;
  assert.equal(decision.gates.highVisibilityTotalMaterial.pass, true);
  assert.equal(decision.gates.highVisibilityCarryover.levelsNegative, false);
  assert.equal(decision.gates.highVisibilityCarryover.pass, false);
  assert.equal(decision.pass, false);
});

test('live analyzer rejects row, schedule, resource, timestamp, and serial tampering', async (t) => {
  const pristine = createRows();
  const cases = [
    ['missing row', (rows) => rows.pop(), /expected exactly 11520/],
    ['wrong row schema', (rows) => { rows[0].schemaVersion = 9; }, /schemaVersion/],
    [
      'reordered rows',
      (rows) => { [rows[0], rows[1]] = [rows[1], rows[0]]; },
      /frameIndex|phaseFrameIndex/,
    ],
    ['wrong pattern', (rows) => { rows[0].crossoverPattern = 'FFFPFPPP'; }, /crossoverPattern/],
    [
      'wrong previous lane',
      (rows) => {
        rows[0].previousLaneId = rows[0].previousLaneId === PORTABLE ? FEATURE : PORTABLE;
      },
      /previousLaneId/,
    ],
    [
      'wrong two-frame history',
      (rows) => {
        rows[0].previousPreviousLaneId = rows[0].previousPreviousLaneId === PORTABLE
          ? FEATURE
          : PORTABLE;
      },
      /previousPreviousLaneId/,
    ],
    ['wrong lane', (rows) => { rows[0].laneId = FEATURE; }, /laneId/],
    [
      'wrong physical order',
      (rows) => { rows[0].plannedLanePhysicalOrder = 'feature|portable'; },
      /lanePhysicalOrder differs from the committed plan/,
    ],
    [
      'wrong selected command buffer',
      (rows) => { rows[0].commandBufferId = rows[0].featureCommandBufferIdAtTimingStart; },
      /commandBufferId/,
    ],
    [
      'aliased command buffers',
      (rows) => {
        rows[0].featureCommandBufferIdAtTimingStart =
          rows[0].portableCommandBufferIdAtTimingStart;
      },
      /identities must be distinct/,
    ],
    ['nonzero command offset', (rows) => { rows[0].commandByteBase = 20; }, /commandByteBase/],
    [
      'missing selector event',
      (rows) => { rows[7].selectorWriteSerial -= 1; },
      /selectorWriteSerial/,
    ],
    [
      'missing strategy selection event',
      (rows) => { rows[7].strategySelectionSerial -= 1; },
      /strategySelectionSerial/,
    ],
    ['extra compute event', (rows) => { rows[9].computeCallSerial += 1; }, /computeCallSerial/],
    ['extra render event', (rows) => { rows[9].renderCallSerial += 1; }, /renderCallSerial/],
    [
      'unsafe serial',
      (rows) => { rows[0].computeCallSerialAtTimingStart = 2 ** 53; },
      /safe integer/,
    ],
    [
      'duplicate GPU frame',
      (rows) => { rows[1].gpuFrameId = rows[0].gpuFrameId; },
      /duplicates gpuFrameId/,
    ],
    [
      'global compute chronology',
      (rows) => {
        for (let index = 480; index < 960; index += 1) {
          rows[index].computeCallSerialAtTimingStart -= 10_000;
          rows[index].computeCallSerial -= 10_000;
        }
      },
      /global compute-call chronology/,
    ],
    [
      'global render chronology',
      (rows) => {
        for (let index = 480; index < 960; index += 1) {
          rows[index].renderCallSerialAtTimingStart -= 10_000;
          rows[index].renderCallSerial -= 10_000;
        }
      },
      /global render-call chronology/,
    ],
    [
      'wrong visibility order',
      (rows) => { rows[0].plannedVisibilityOrder = '0.2|0.99'; },
      /visibilityOrder differs from the committed plan/,
    ],
    [
      'missing compute timestamp UID',
      (rows) => { rows[0].gpuComputeTimestampUidCount = 0; },
      /gpuComputeTimestampUidCount/,
    ],
    [
      'missing render timestamp UID',
      (rows) => { rows[0].gpuRenderTimestampUidCount = 0; },
      /gpuRenderTimestampUidCount/,
    ],
    ['non-finite compute', (rows) => { rows[0].gpuComputeMs = NaN; }, /gpuComputeMs/],
    [
      'inconsistent pass total',
      (rows) => { rows[0].gpuPassTotalMs += 1e-8; },
      /differs from gpuComputeMs \+ gpuRenderMs/,
    ],
    ['wrong dispatch count', (rows) => { rows[0].configuredComputeDispatches = 3; }, /configuredComputeDispatches/],
    ['wrong submission count', (rows) => { rows[0].configuredComputeSubmissions = 2; }, /configuredComputeSubmissions/],
    ['failed validation', (rows) => { rows[0].validationPass = false; }, /validationPass/],
    [
      'overflowed derived block arithmetic',
      (rows) => {
        for (const row of rows) {
          if (row.frameIndex < 8) {
            row.gpuComputeMs = 0;
            row.gpuRenderMs = Number.MAX_VALUE;
            row.gpuPassTotalMs = Number.MAX_VALUE;
          }
        }
      },
      /finite number|non-finite/,
    ],
  ];

  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const rows = structuredClone(pristine);
      mutate(rows);
      assert.throws(() => summarizeLiveFirstInstanceCrossoverRows(rows), pattern);
    });
  }
});

test('primary and render material gates enforce their independent conjunctions', async (t) => {
  await t.test('GPU-pass percentage alone is insufficient', () => {
    const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
      renderEffect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.12 : 0),
      computeEffect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? 0.06 : 0),
    })).preregisteredDecision;
    assert.equal(decision.gates.highVisibilityRender.pass, true);
    assert.equal(
      decision.gates.highVisibilityTotalMaterial.observed.deltaPercent <= -5,
      true,
    );
    assert.equal(decision.gates.highVisibilityTotalMaterial.pass, false);
  });

  await t.test('GPU-pass milliseconds alone are insufficient', () => {
    const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
      renderBaseline: () => 0.8,
      computeBaseline: () => 2.2,
      renderEffect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.2 : 0),
      computeEffect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? 0.09 : 0),
    })).preregisteredDecision;
    assert.equal(decision.gates.highVisibilityRender.pass, true);
    assert.equal(decision.gates.highVisibilityTotalMaterial.observed.deltaMs <= -0.1, true);
    assert.equal(decision.gates.highVisibilityTotalMaterial.pass, false);
  });

  await t.test('render is a mandatory secondary endpoint', () => {
    const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
      renderEffect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.05 : 0),
      computeEffect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.11 : 0),
    })).preregisteredDecision;
    assert.equal(decision.gates.highVisibilityTotalMaterial.pass, true);
    assert.equal(decision.gates.highVisibilityTotalDirection.pass, true);
    assert.equal(decision.gates.highVisibilityRender.pass, false);
    assert.equal(decision.pass, false);
  });
});

test('direction gate requires ten negative repetition-level GPU-pass estimates', () => {
  const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
    renderEffect: ({ visibilityFraction, repetitionIndex }) => {
      if (visibilityFraction !== 0.99) return 0;
      return repetitionIndex < 9 ? -0.2 : 0.05;
    },
  })).preregisteredDecision;
  assert.equal(decision.gates.highVisibilityTotalMaterial.pass, true);
  assert.equal(decision.gates.highVisibilityTotalDirection.observedNegative, 9);
  assert.equal(decision.gates.highVisibilityTotalDirection.pass, false);
});

test('total nuisance gates reject physical-order, orientation, visibility, and half interactions', async (t) => {
  const cases = [
    ['lanePhysicalOrder', ({ portablePhysicalOrderPosition }) => portablePhysicalOrderPosition],
    ['startingOrientation', ({ orientationOffset }) => orientationOffset],
    ['visibilityOrderPosition', ({ visibilityOrderPosition }) => visibilityOrderPosition],
  ];
  for (const [factor, level] of cases) {
    await t.test(factor, () => {
      const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
        renderEffect: (context) => {
          if (context.visibilityFraction !== 0.99) return 0;
          return level(context) === 0 ? -0.25 : -0.05;
        },
      })).preregisteredDecision;
      const gate = decision.gates.highVisibilityTotalStrata.factors[factor];
      assert.equal(gate.levelsNegative, true);
      assert.equal(gate.interactionWithinBounds, false);
      assert.equal(gate.pass, false);
    });
  }

  await t.test('measurementHalf', () => {
    const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
      renderEffect: ({ visibilityFraction, blockIndex }) => {
        if (visibilityFraction !== 0.99) return 0;
        return blockIndex < 30 ? -0.25 : -0.05;
      },
    })).preregisteredDecision;
    const gate = decision.gates.highVisibilityTotalStrata.factors.measurementHalf;
    assert.equal(gate.levelsNegative, true);
    assert.equal(gate.interactionWithinBounds, false);
    assert.equal(gate.pass, false);
  });
});

test('low-dose, paired-dose, and condition-blind drift gates fail closed', async (t) => {
  await t.test('low-dose ten-of-twelve count', () => {
    const gate = summarizeLiveFirstInstanceCrossoverRows(createRows({
      renderEffect: ({ visibilityFraction, repetitionIndex }) => {
        if (visibilityFraction === 0.99) return -0.16;
        return repetitionIndex < 9 ? 0 : 0.06;
      },
    })).preregisteredDecision.gates.lowVisibilityTotalRegression;
    assert.equal(gate.medianBelowBothBounds, true);
    assert.deepEqual(gate.counts, {
      belowMsUpperBound: 9,
      belowPercentUpperBound: 9,
    });
    assert.equal(gate.pass, false);
  });

  await t.test('paired high-minus-low separation', () => {
    const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
      renderEffect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.11 : -0.07),
    })).preregisteredDecision;
    assert.equal(decision.gates.highVisibilityTotalMaterial.pass, true);
    assert.equal(decision.gates.lowVisibilityTotalRegression.pass, true);
    assert.equal(decision.gates.pairedHighMinusLowTotal.pass, false);
  });

  await t.test('condition-blind first-versus-last-quarter drift', () => {
    const decision = summarizeLiveFirstInstanceCrossoverRows(createRows({
      renderBaseline: ({ blockIndex }) => {
        if (blockIndex < 15) return 0.8;
        if (blockIndex >= 45) return 1;
        return 0.9;
      },
    })).preregisteredDecision;
    assert.equal(decision.gates.drift.pass, false);
    assert.equal(decision.gates.drift.overall.pass, false);
    assert.equal(decision.gates.drift.visibilities['0.99'].pass, false);
    assert.equal(decision.gates.drift.visibilities['0.2'].pass, false);
  });
});

test('zero compute timestamps are rejected instead of reducing the primary endpoint to render-only', () => {
  assert.throws(
    () => summarizeLiveFirstInstanceCrossoverRows(createRows({
      computeBaseline: () => 0,
    })),
    /gpuComputeMs must be a finite number > 0/,
  );
});

test('the analyzer refuses a conflicting live mode-order alias', () => {
  const rows = createRows();
  rows[0].modeOrder = [FIRST_INSTANCE_LIVE_CROSSOVER_MODE];
  rows[0].plannedModeOrder = 'wrong-live-mode';
  assert.throws(
    () => summarizeLiveFirstInstanceCrossoverRows(rows),
    /plannedModeOrder/,
  );
});
