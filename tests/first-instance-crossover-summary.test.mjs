import assert from 'node:assert/strict';
import test from 'node:test';
import {
  summarizeFirstInstanceCrossoverRows,
} from '../analysis/first-instance-crossover-summary.mjs';
import { firstInstanceCrossoverFrame } from '../src/benchmark/first-instance-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  buildFirstInstanceCrossoverPlan,
} from '../src/benchmark/plan.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_CROSSOVER_LANES;
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const RECORD_BYTES = 20;

function closeTo(actual, expected, tolerance = 1e-10) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} is not within ${tolerance} of ${expected}`,
  );
}

function createRows({
  effect = ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.2 : 0),
  baseline = () => 1,
} = {}) {
  const plan = buildFirstInstanceCrossoverPlan({
    runId: 'first-instance-analysis',
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  });
  const rows = [];
  for (const trial of plan) {
    const selectorStart = trial.planIndex * 10_000;
    const strategyStart = trial.planIndex * 10_000 + 100;
    const renderStart = trial.planIndex * 10_000 + 200;
    const expectedVisibleCount = Math.round(OBJECT_COUNT * trial.visibilityFraction);
    const portableSegmentIndex = trial.laneCommandSegmentOrder.indexOf(PORTABLE);
    for (let frameIndex = 0; frameIndex < 480; frameIndex += 1) {
      const scheduled = firstInstanceCrossoverFrame(
        frameIndex,
        trial.superblockOrientationOffset,
      );
      const context = {
        trial,
        visibilityFraction: trial.visibilityFraction,
        repetitionIndex: trial.repetitionIndex,
        visibilityOrderPosition: trial.visibilityOrderPosition,
        portableSegmentIndex,
        orientationOffset: trial.superblockOrientationOffset,
        blockIndex: scheduled.crossoverBlockIndex,
        frameIndex,
      };
      const portableDuration = baseline(context);
      const laneEffect = effect(context);
      const gpuRenderMs = scheduled.laneId === FEATURE
        ? portableDuration + laneEffect
        : portableDuration;
      const commandSegmentIndex = trial.laneCommandSegmentOrder.indexOf(scheduled.laneId);
      const serialOffset = 320 + frameIndex + 1;
      rows.push({
        schemaVersion: 2,
        runId: 'first-instance-analysis',
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
        plannedLaneCommandSegmentOrder: trial.laneCommandSegmentOrder.join('|'),
        superblockOrientationOffset: trial.superblockOrientationOffset,
        scenarioLayout: 'baseline',
        layoutOrderPosition: 0,
        plannedLayoutOrder: 'baseline',
        protocolWarmupFrames: 320,
        protocolMeasuredFrames: 480,
        objectCount: OBJECT_COUNT,
        bucketCount: BUCKET_COUNT,
        expectedVisibleCount,
        usesCompute: false,
        configuredDrawCommands: BUCKET_COUNT,
        configuredRenderObjects: 1,
        configuredComputeDispatches: 0,
        configuredComputeSubmissions: 0,
        configuredSubmittedInstances: expectedVisibleCount,
        validationPass: true,
        validationKind: 'first-instance-crossover-exact-paired-snapshots',
        timestampAvailable: true,
        gpuRenderTimestampUidCount: 1,
        expectedRenderTimestampUidCount: 1,
        gpuComputeMs: null,
        cpuComputeSubmitMs: null,
        gpuRenderMs,
        gpuPassTotalMs: gpuRenderMs,
        crossoverBlockIndex: scheduled.crossoverBlockIndex,
        withinBlockPosition: scheduled.withinBlockPosition,
        crossoverPattern: scheduled.pattern,
        crossoverPatternIndex: scheduled.patternIndex,
        laneId: scheduled.laneId,
        commandSegmentIndex,
        commandRecordBase: commandSegmentIndex * BUCKET_COUNT,
        commandByteBase: commandSegmentIndex * BUCKET_COUNT * RECORD_BYTES,
        selectorWriteSerialAtTimingStart: selectorStart,
        strategySelectionSerialAtTimingStart: strategyStart,
        renderCallSerialAtTimingStart: renderStart,
        selectorWriteSerial: selectorStart + serialOffset,
        strategySelectionSerial: strategyStart + serialOffset,
        renderCallSerial: renderStart + serialOffset,
        gpuFrameId: trial.planIndex * 1_000 + frameIndex,
      });
    }
  }
  return rows;
}

test('first-instance analyzer reconstructs all 24 trials, 11,520 rows, and block estimates', () => {
  const summary = summarizeFirstInstanceCrossoverRows(createRows());

  assert.equal(summary.kind, 'indirect-first-instance-crossover-summary');
  assert.equal(summary.nRows, 11_520);
  assert.equal(summary.nTrials, 24);
  assert.equal(summary.trials.every((trial) => (
    trial.nRows === 480
    && trial.nBlocks === 60
    && trial.blocks.every((block) => (
      Number.isFinite(block.deltaMs)
      && Number.isFinite(block.deltaPercent)
      && block.pattern === (
        (block.crossoverBlockIndex + trial.superblockOrientationOffset) % 2 === 0
          ? 'PFPFFPFP'
          : 'FPFPPFPF'
      )
    ))
  )), true);

  const high = summary.contrasts.visibilities['0.99'];
  const low = summary.contrasts.visibilities['0.2'];
  closeTo(high.median.deltaMs, -0.2);
  closeTo(high.median.deltaPercent, -20);
  assert.equal(high.negativeCount, 12);
  closeTo(low.median.deltaMs, 0);
  closeTo(summary.contrasts.pairedHighMinusLow.median.deltaMs, -0.2);
  closeTo(summary.contrasts.drift.overall.median.deltaMs, 0);

  const decision = summary.preregisteredNumericalDecision;
  assert.equal(decision.status, 'evaluated');
  assert.equal(decision.pass, true);
  assert.deepEqual(decision.failedGates, []);
  assert.equal(decision.gates.highVisibilityMaterial.pass, true);
  assert.equal(decision.gates.highVisibilityDirection.pass, true);
  assert.equal(decision.gates.highVisibilityDirection.observedNegative, 12);
  assert.equal(decision.gates.highVisibilityStrata.pass, true);
  assert.equal(
    Object.values(decision.gates.highVisibilityStrata.factors).every(
      (gate) => gate.pass && gate.strata.length === 2,
    ),
    true,
  );
  assert.equal(decision.gates.lowVisibilityRegression.pass, true);
  assert.deepEqual(decision.gates.lowVisibilityRegression.counts, {
    belowMsUpperBound: 12,
    belowPercentUpperBound: 12,
  });
  assert.equal(decision.gates.pairedHighMinusLow.pass, true);
  assert.equal(decision.gates.drift.pass, true);
});

test('first-instance analyzer rejects schedule, segment, event, and matrix tampering', async (t) => {
  const pristine = createRows();
  const cases = [
    ['missing row', (rows) => rows.pop(), /expected exactly 11520/],
    ['wrong row schema', (rows) => { rows[0].schemaVersion = 999; }, /schemaVersion/],
    [
      'reordered row',
      (rows) => { [rows[0], rows[1]] = [rows[1], rows[0]]; },
      /frameIndex|phaseFrameIndex/,
    ],
    ['wrong pattern', (rows) => { rows[0].crossoverPattern = 'FPFPPFPF'; }, /crossoverPattern/],
    ['wrong lane', (rows) => { rows[0].laneId = FEATURE; }, /laneId/],
    [
      'wrong command segment',
      (rows) => { rows[0].commandSegmentIndex = 1 - rows[0].commandSegmentIndex; },
      /commandSegmentIndex/,
    ],
    [
      'wrong command byte base',
      (rows) => { rows[0].commandByteBase += RECORD_BYTES; },
      /commandByteBase/,
    ],
    [
      'missing selection event',
      (rows) => { rows[7].strategySelectionSerial -= 1; },
      /strategySelectionSerial/,
    ],
    [
      'unsafe selection serial',
      (rows) => { rows[0].selectorWriteSerialAtTimingStart = 2 ** 53; },
      /safe integer/,
    ],
    [
      'extra render event',
      (rows) => { rows[9].renderCallSerial += 1; },
      /renderCallSerial/,
    ],
    [
      'duplicate GPU frame',
      (rows) => { rows[1].gpuFrameId = rows[0].gpuFrameId; },
      /duplicates gpuFrameId/,
    ],
    [
      'reordered trial GPU chronology',
      (rows) => {
        for (let index = 0; index < 480; index += 1) {
          rows[index].gpuFrameId += 1_000;
          rows[index + 480].gpuFrameId -= 1_000;
        }
      },
      /global GPU frame chronology/,
    ],
    [
      'reordered trial render-call chronology',
      (rows) => {
        for (let index = 0; index < 480; index += 1) {
          rows[index].renderCallSerialAtTimingStart += 10_000;
          rows[index].renderCallSerial += 10_000;
          rows[index + 480].renderCallSerialAtTimingStart -= 10_000;
          rows[index + 480].renderCallSerial -= 10_000;
        }
      },
      /global render-call chronology/,
    ],
    [
      'wrong command order',
      (rows) => { rows[0].plannedLaneCommandSegmentOrder = 'feature|portable'; },
      /laneCommandSegmentOrder differs from the committed plan/,
    ],
    [
      'conflicting mode-order alias',
      (rows) => {
        rows[0].modeOrder = [FIRST_INSTANCE_CROSSOVER_MODE];
        rows[0].plannedModeOrder = 'corrupt';
      },
      /plannedModeOrder/,
    ],
    [
      'conflicting layout-order alias',
      (rows) => {
        rows[0].layoutOrder = ['baseline'];
        rows[0].plannedLayoutOrder = 'corrupt';
      },
      /plannedLayoutOrder/,
    ],
    [
      'wrong plan index',
      (rows) => { rows[480].planIndex = 0; },
      /not contiguous in committed plan-index order/,
    ],
    [
      'wrong visibility order',
      (rows) => { rows[0].plannedVisibilityOrder = '0.2|0.99'; },
      /visibilityOrder differs from the committed plan/,
    ],
    [
      'unexpected compute duration',
      (rows) => { rows[0].gpuComputeMs = 0; },
      /gpuComputeMs/,
    ],
    [
      'overflowed derived block arithmetic',
      (rows) => {
        for (const row of rows) {
          if (row.frameIndex >= 160 && row.frameIndex < 168) {
            row.gpuRenderMs = Number.MAX_VALUE;
            row.gpuPassTotalMs = Number.MAX_VALUE;
          }
        }
      },
      /non-finite/,
    ],
    [
      'missing timestamp UID',
      (rows) => { rows[0].gpuRenderTimestampUidCount = 0; },
      /gpuRenderTimestampUidCount/,
    ],
  ];

  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const rows = structuredClone(pristine);
      mutate(rows);
      assert.throws(() => summarizeFirstInstanceCrossoverRows(rows), pattern);
    });
  }
});

test('99% material gate requires both preregistered thresholds', async (t) => {
  await t.test('percent alone is insufficient', () => {
    const decision = summarizeFirstInstanceCrossoverRows(createRows({
      effect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.06 : 0),
    })).preregisteredNumericalDecision;
    assert.equal(decision.gates.highVisibilityMaterial.observed.deltaPercent <= -5, true);
    assert.equal(decision.gates.highVisibilityMaterial.pass, false);
  });

  await t.test('milliseconds alone are insufficient', () => {
    const decision = summarizeFirstInstanceCrossoverRows(createRows({
      effect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.11 : 0),
      baseline: () => 3,
    })).preregisteredNumericalDecision;
    assert.equal(decision.gates.highVisibilityMaterial.observed.deltaMs <= -0.1, true);
    assert.equal(decision.gates.highVisibilityMaterial.pass, false);
  });
});

test('99% direction gate rejects fewer than ten negative repetition estimates', () => {
  const decision = summarizeFirstInstanceCrossoverRows(createRows({
    effect: ({ visibilityFraction, repetitionIndex }) => {
      if (visibilityFraction !== 0.99) return 0;
      return repetitionIndex < 9 ? -0.2 : 0.01;
    },
  })).preregisteredNumericalDecision;
  assert.equal(decision.gates.highVisibilityMaterial.pass, true);
  assert.equal(decision.gates.highVisibilityDirection.observedNegative, 9);
  assert.equal(decision.gates.highVisibilityDirection.pass, false);
});

test('99% stratum gates reject physical, orientation, order, and half interactions', async (t) => {
  const cases = [
    ['physicalCommandSegment', ({ portableSegmentIndex }) => portableSegmentIndex],
    ['startingOrientation', ({ orientationOffset }) => orientationOffset],
    ['visibilityOrderPosition', ({ visibilityOrderPosition }) => visibilityOrderPosition],
  ];
  for (const [factor, level] of cases) {
    await t.test(factor, () => {
      const decision = summarizeFirstInstanceCrossoverRows(createRows({
        effect: (context) => {
          if (context.visibilityFraction !== 0.99) return 0;
          return level(context) === 0 ? -0.25 : -0.05;
        },
      })).preregisteredNumericalDecision;
      const gate = decision.gates.highVisibilityStrata.factors[factor];
      assert.equal(gate.levelsNegative, true);
      assert.equal(gate.interactionWithinBounds, false);
      assert.equal(gate.pass, false);
    });
  }

  await t.test('measurementHalf', () => {
    const decision = summarizeFirstInstanceCrossoverRows(createRows({
      effect: ({ visibilityFraction, blockIndex }) => {
        if (visibilityFraction !== 0.99) return 0;
        return blockIndex < 30 ? -0.25 : -0.05;
      },
    })).preregisteredNumericalDecision;
    const gate = decision.gates.highVisibilityStrata.factors.measurementHalf;
    assert.equal(gate.levelsNegative, true);
    assert.equal(gate.interactionWithinBounds, false);
    assert.equal(gate.pass, false);
  });
});

test('20% regression gate enforces medians and ten-of-twelve upper-bound counts', async (t) => {
  await t.test('median regression', () => {
    const gate = summarizeFirstInstanceCrossoverRows(createRows({
      effect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.2 : 0.03),
    })).preregisteredNumericalDecision.gates.lowVisibilityRegression;
    assert.equal(gate.medianBelowBothBounds, false);
    assert.equal(gate.pass, false);
  });

  await t.test('insufficient repetition count despite passing medians', () => {
    const gate = summarizeFirstInstanceCrossoverRows(createRows({
      effect: ({ visibilityFraction, repetitionIndex }) => {
        if (visibilityFraction === 0.99) return -0.2;
        return repetitionIndex < 9 ? 0 : 0.06;
      },
    })).preregisteredNumericalDecision.gates.lowVisibilityRegression;
    assert.equal(gate.medianBelowBothBounds, true);
    assert.equal(gate.counts.belowMsUpperBound, 9);
    assert.equal(gate.counts.belowPercentUpperBound, 9);
    assert.equal(gate.pass, false);
  });
});

test('paired 99%-minus-20% and condition-blind drift gates fail closed', async (t) => {
  await t.test('insufficient paired separation', () => {
    const decision = summarizeFirstInstanceCrossoverRows(createRows({
      effect: ({ visibilityFraction }) => (visibilityFraction === 0.99 ? -0.11 : -0.07),
    })).preregisteredNumericalDecision;
    assert.equal(decision.gates.highVisibilityMaterial.pass, true);
    assert.equal(decision.gates.lowVisibilityRegression.pass, true);
    assert.equal(decision.gates.pairedHighMinusLow.pass, false);
  });

  await t.test('condition-blind drift overall and within each visibility', () => {
    const decision = summarizeFirstInstanceCrossoverRows(createRows({
      baseline: ({ blockIndex }) => {
        if (blockIndex < 15) return 1;
        if (blockIndex >= 45) return 1.2;
        return 1.1;
      },
    })).preregisteredNumericalDecision;
    assert.equal(decision.gates.drift.pass, false);
    assert.equal(decision.gates.drift.overall.pass, false);
    assert.equal(decision.gates.drift.visibilities['0.99'].pass, false);
    assert.equal(decision.gates.drift.visibilities['0.2'].pass, false);
  });
});
