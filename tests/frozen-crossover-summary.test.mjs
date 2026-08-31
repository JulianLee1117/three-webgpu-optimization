import assert from 'node:assert/strict';
import test from 'node:test';
import { summarizeFrozenCrossoverRows } from '../analysis/frozen-crossover-summary.mjs';
import { frozenCrossoverFrame } from '../src/benchmark/frozen-crossover-schedule.js';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  buildFrozenDepthCrossoverPlan,
} from '../src/benchmark/plan.js';

const [FRONT_TO_BACK] = FROZEN_DEPTH_CROSSOVER_LANES;
const OBJECT_COUNT = 65_536;

function closeTo(actual, expected, tolerance = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} is not within ${tolerance} of ${expected}`,
  );
}

function createRows({
  effect = ({ layout }) => (layout === 'high-overlap' ? -0.2 : 0),
  baseline = () => 1,
} = {}) {
  const plan = buildFrozenDepthCrossoverPlan({
    runId: 'frozen-analysis',
    objectCount: OBJECT_COUNT,
    bucketCount: 32,
  });
  const rows = [];
  for (const trial of plan) {
    const frontLaneBase = trial.laneStorageOrder.indexOf(FRONT_TO_BACK) * OBJECT_COUNT;
    for (let frameIndex = 0; frameIndex < 480; frameIndex += 1) {
      const scheduled = frozenCrossoverFrame(
        frameIndex,
        trial.superblockOrientationOffset,
      );
      const reverseDuration = baseline({
        trial,
        frameIndex,
        blockIndex: scheduled.crossoverBlockIndex,
      });
      const laneEffect = effect({
        trial,
        layout: trial.layout,
        repetitionIndex: trial.repetitionIndex,
        frontLaneBase,
        frameIndex,
        blockIndex: scheduled.crossoverBlockIndex,
      });
      const mappingEvidence = trial.repetitionIndex % 2 === 0
        ? { laneStorageOrder: trial.laneStorageOrder.join('|') }
        : { frontLaneBase };
      rows.push({
        trialId: trial.trialId,
        repetitionIndex: trial.repetitionIndex,
        layout: trial.layout,
        layoutOrderPosition: trial.layoutOrderPosition,
        ...mappingEvidence,
        superblockOrientationOffset: trial.superblockOrientationOffset,
        crossoverBlockIndex: scheduled.crossoverBlockIndex,
        withinBlockPosition: scheduled.withinBlockPosition,
        crossoverPattern: scheduled.pattern,
        laneId: scheduled.laneId,
        laneBase: scheduled.laneId === FRONT_TO_BACK
          ? frontLaneBase
          : (frontLaneBase === 0 ? OBJECT_COUNT : 0),
        gpuRenderMs: scheduled.laneId === FRONT_TO_BACK
          ? reverseDuration + laneEffect
          : reverseDuration,
      });
    }
  }
  return rows;
}

function changeRepetitionStorageMapping(rows, repetitionIndex) {
  for (const row of rows.filter((entry) => entry.repetitionIndex === repetitionIndex)) {
    const oldFrontBase = row.laneStorageOrder?.startsWith(FRONT_TO_BACK)
      ? 0
      : row.frontLaneBase;
    const newFrontBase = oldFrontBase === 0 ? OBJECT_COUNT : 0;
    if (row.laneStorageOrder !== undefined) {
      row.laneStorageOrder = row.laneStorageOrder.split('|').reverse().join('|');
    } else {
      row.frontLaneBase = newFrontBase;
    }
    row.laneBase = row.laneId === FRONT_TO_BACK
      ? newFrontBase
      : (newFrontBase === 0 ? OBJECT_COUNT : 0);
  }
}

test('frozen crossover analyzer accepts the exact matrix and evaluates every gate', () => {
  const summary = summarizeFrozenCrossoverRows(createRows());

  assert.equal(summary.kind, 'frozen-depth-crossover-summary');
  assert.equal(summary.nRows, 11_520);
  assert.equal(summary.nTrials, 24);
  assert.equal(summary.trials.every((trial) => (
    trial.nRows === 480
    && trial.nBlocks === 60
    && trial.blocks.every((block) => Number.isFinite(block.deltaPercent))
  )), true);

  const high = summary.contrasts.layouts['high-overlap'];
  const low = summary.contrasts.layouts['low-overlap'];
  closeTo(high.median.deltaMs, -0.2);
  closeTo(high.median.deltaPercent, -20);
  assert.equal(high.negativeCount, 12);
  closeTo(low.median.deltaMs, 0);
  closeTo(summary.contrasts.pairedHighMinusLow.median.deltaMs, -0.2);
  closeTo(summary.contrasts.drift.overall.median.deltaMs, 0);

  const decision = summary.preregisteredDecision;
  assert.equal(decision.status, 'evaluated');
  assert.equal(decision.pass, true);
  assert.deepEqual(decision.failedGates, []);
  assert.equal(decision.gates.highOverlapMaterial.pass, true);
  assert.equal(decision.gates.highOverlapDirection.pass, true);
  assert.equal(decision.gates.highOverlapDirection.observedNegative, 12);
  assert.equal(decision.gates.highOverlapStrata.pass, true);
  assert.equal(
    Object.values(decision.gates.highOverlapStrata.factors).every(
      (gate) => gate.strata.length === 2 && gate.pass,
    ),
    true,
  );
  assert.equal(decision.gates.lowOverlapEquivalence.pass, true);
  assert.deepEqual(decision.gates.lowOverlapEquivalence.counts, {
    msAboveLower: 12,
    msBelowUpper: 12,
    percentAboveLower: 12,
    percentBelowUpper: 12,
  });
  assert.equal(decision.gates.pairedHighMinusLow.pass, true);
  assert.equal(decision.gates.drift.pass, true);
});
test('frozen crossover analyzer rejects schedule, ordering, and lane/base tampering', async (t) => {
  const cases = [
    [
      'missing row',
      (rows) => rows.pop(),
      /expected exactly 11520/,
    ],
    [
      'reordered row',
      (rows) => { [rows[0], rows[1]] = [rows[1], rows[0]]; },
      /withinBlockPosition/,
    ],
    [
      'wrong pattern',
      (rows) => { rows[0].crossoverPattern = 'RFRFFRFR'; },
      /crossoverPattern/,
    ],
    [
      'wrong lane',
      (rows) => { rows[0].laneId = FROZEN_DEPTH_CROSSOVER_LANES[1]; },
      /laneId/,
    ],
    [
      'wrong lane base',
      (rows) => { rows[0].laneBase = 123; },
      /laneBase/,
    ],
    [
      'changed mapping',
      (rows) => { rows[1].laneStorageOrder = rows[1].laneStorageOrder.split('|').reverse().join('|'); },
      /changes the trial lane\/base mapping/,
    ],
  ];

  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const rows = createRows();
      mutate(rows);
      assert.throws(() => summarizeFrozenCrossoverRows(rows), pattern);
    });
  }
});

test('frozen crossover analyzer rejects an unbalanced committed factor matrix', () => {
  const rows = createRows();
  changeRepetitionStorageMapping(rows, 0);
  assert.throws(
    () => summarizeFrozenCrossoverRows(rows),
    /frontLaneBase=.*expected 6/,
  );
});

test('frozen crossover timing gates fail closed under adversarial effects', async (t) => {
  await t.test('wrong direction', () => {
    const result = summarizeFrozenCrossoverRows(createRows({
      effect: ({ layout }) => (layout === 'high-overlap' ? 0.02 : 0),
    })).preregisteredDecision;
    assert.equal(result.pass, false);
    assert.equal(result.gates.highOverlapMaterial.pass, false);
    assert.equal(result.gates.highOverlapDirection.pass, false);
  });

  await t.test('physical-base interaction', () => {
    const result = summarizeFrozenCrossoverRows(createRows({
      effect: ({ layout, frontLaneBase }) => {
        if (layout === 'low-overlap') return 0;
        return frontLaneBase === 0 ? -0.25 : -0.05;
      },
    })).preregisteredDecision;
    assert.equal(result.gates.highOverlapMaterial.pass, true);
    assert.equal(result.gates.highOverlapDirection.pass, true);
    assert.equal(result.gates.highOverlapStrata.pass, false);
    assert.equal(
      result.gates.highOverlapStrata.factors.physicalBase.interactionWithinBounds,
      false,
    );
  });

  await t.test('low-overlap non-equivalence', () => {
    const result = summarizeFrozenCrossoverRows(createRows({
      effect: ({ layout }) => (layout === 'high-overlap' ? -0.2 : 0.2),
    })).preregisteredDecision;
    assert.equal(result.gates.lowOverlapEquivalence.pass, false);
  });

  await t.test('insufficient paired high-minus-low separation', () => {
    const result = summarizeFrozenCrossoverRows(createRows({
      effect: ({ layout }) => (layout === 'high-overlap' ? -0.11 : -0.02),
    })).preregisteredDecision;
    assert.equal(result.gates.highOverlapMaterial.pass, true);
    assert.equal(result.gates.lowOverlapEquivalence.pass, true);
    assert.equal(result.gates.pairedHighMinusLow.pass, false);
  });

  await t.test('condition-blind drift', () => {
    const result = summarizeFrozenCrossoverRows(createRows({
      baseline: ({ blockIndex }) => {
        if (blockIndex < 15) return 1;
        if (blockIndex >= 45) return 1.2;
        return 1.1;
      },
    })).preregisteredDecision;
    assert.equal(result.gates.drift.pass, false);
    assert.equal(result.gates.drift.overall.pass, false);
    assert.equal(result.gates.drift.layouts['high-overlap'].pass, false);
    assert.equal(result.gates.drift.layouts['low-overlap'].pass, false);
  });
});
