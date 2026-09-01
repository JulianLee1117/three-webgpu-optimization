import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceLiveCrossoverHistoryCounts,
  firstInstanceLiveCrossoverFrame,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS,
  FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS,
  buildFirstInstanceLiveCrossoverPlan,
} from '../src/benchmark/plan.js';

const PLAN_INPUT = Object.freeze({
  runId: 'first-instance-live',
  objectCount: 65_536,
  bucketCount: 32,
});

function binaryFactorCounts(rows, left, right) {
  const counts = new Map([
    ['0|0', 0],
    ['0|1', 0],
    ['1|0', 0],
    ['1|1', 0],
  ]);
  for (const row of rows) {
    const key = `${left(row)}|${right(row)}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.values()].sort((left, right) => left - right);
}

test('live first-instance constants freeze the preregistered matrix', () => {
  assert.equal(
    FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
    'indirect-first-instance-live-crossover',
  );
  assert.deepEqual(FIRST_INSTANCE_LIVE_CROSSOVER_LANES, ['portable', 'feature']);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS, 12);
  assert.deepEqual(FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS, [0.99, 0.2]);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS.length, 12);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS.length, 12);
  assert.equal(Object.isFrozen(FIRST_INSTANCE_LIVE_CROSSOVER_LANES), true);
  assert.equal(Object.isFrozen(FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS), true);
  assert.equal(Object.isFrozen(FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS), true);
  assert.ok(FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS.every(Object.isFrozen));
  assert.equal(Object.isFrozen(FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS), true);
});

test('live first-instance plan has 24 exact paired trials', () => {
  const plan = buildFirstInstanceLiveCrossoverPlan(PLAN_INPUT);
  assert.equal(plan.length, 24);
  assert.equal(new Set(plan.map((trial) => trial.trialId)).size, 24);
  assert.deepEqual(
    plan.map((trial) => trial.trialId),
    Array.from(
      { length: 24 },
      (_, index) => `first-instance-live-t${String(index + 1).padStart(2, '0')}`,
    ),
  );

  for (let repetitionIndex = 0;
    repetitionIndex < FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS;
    repetitionIndex += 1) {
    const trials = plan.filter((trial) => trial.repetitionIndex === repetitionIndex);
    const expectedVisibilityOrder = repetitionIndex % 2 === 0
      ? [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS]
      : [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS].reverse();
    const expectedPhysicalOrder = FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS[
      repetitionIndex
    ];
    assert.equal(trials.length, 2);
    assert.deepEqual(trials.map((trial) => trial.visibilityFraction), expectedVisibilityOrder);
    assert.deepEqual(trials.map((trial) => trial.visibilityOrderPosition), [0, 1]);
    for (const trial of trials) {
      assert.equal(trial.modeId, FIRST_INSTANCE_LIVE_CROSSOVER_MODE);
      assert.deepEqual(trial.modeOrder, [FIRST_INSTANCE_LIVE_CROSSOVER_MODE]);
      assert.equal(trial.modeOrderPosition, 0);
      assert.deepEqual(trial.visibilityOrder, expectedVisibilityOrder);
      assert.deepEqual(trial.lanePhysicalOrder, expectedPhysicalOrder);
      assert.notEqual(trial.lanePhysicalOrder, expectedPhysicalOrder);
      assert.equal(
        trial.superblockOrientationOffset,
        FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS[repetitionIndex],
      );
      assert.equal(trial.layout, 'baseline');
      assert.deepEqual(trial.layoutOrder, ['baseline']);
      assert.equal(trial.layoutOrderPosition, 0);
      assert.equal(trial.objectCount, PLAN_INPUT.objectCount);
      assert.equal(trial.bucketCount, PLAN_INPUT.bucketCount);
    }
  }
  for (const visibilityFraction of FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS) {
    assert.equal(
      plan.filter((trial) => trial.visibilityFraction === visibilityFraction).length,
      FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
    );
  }
});

test('live physical order, orientation, and high-visibility position are pairwise balanced', () => {
  const plan = buildFirstInstanceLiveCrossoverPlan(PLAN_INPUT);
  const repetitions = Array.from(
    { length: FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS },
    (_, repetitionIndex) => {
      const trials = plan.filter((trial) => trial.repetitionIndex === repetitionIndex);
      return {
        physicalOrder: trials[0].lanePhysicalOrder[0] === 'portable' ? 0 : 1,
        orientation: trials[0].superblockOrientationOffset,
        visibilityPosition: trials.find(
          (trial) => trial.visibilityFraction === 0.99,
        ).visibilityOrderPosition,
      };
    },
  );
  const exactPairCounts = [3, 3, 3, 3];
  assert.deepEqual(
    binaryFactorCounts(repetitions, (row) => row.physicalOrder, (row) => row.orientation),
    exactPairCounts,
  );
  assert.deepEqual(
    binaryFactorCounts(
      repetitions,
      (row) => row.physicalOrder,
      (row) => row.visibilityPosition,
    ),
    exactPairCounts,
  );
  assert.deepEqual(
    binaryFactorCounts(repetitions, (row) => row.orientation, (row) => row.visibilityPosition),
    exactPairCounts,
  );
});

test('live first-instance plan rejects changed or correlated factors', () => {
  assert.throws(
    () => buildFirstInstanceLiveCrossoverPlan({
      ...PLAN_INPUT,
      visibilityLevels: [0.2, 0.99],
    }),
    /visibility levels must be exactly \[0\.99, 0\.2\]/,
  );
  assert.throws(
    () => buildFirstInstanceLiveCrossoverPlan({
      ...PLAN_INPUT,
      lanePhysicalOrders: FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS.slice(1),
    }),
    /requires 12 lane physical orders/,
  );
  const invalidPermutation = FIRST_INSTANCE_LIVE_CROSSOVER_LANE_PHYSICAL_ORDERS.map(
    (order) => [...order],
  );
  invalidPermutation[4] = ['portable', 'portable'];
  assert.throws(
    () => buildFirstInstanceLiveCrossoverPlan({
      ...PLAN_INPUT,
      lanePhysicalOrders: invalidPermutation,
    }),
    /not an exact lane permutation/,
  );
  assert.throws(
    () => buildFirstInstanceLiveCrossoverPlan({
      ...PLAN_INPUT,
      orientationOffsets: FIRST_INSTANCE_LIVE_CROSSOVER_ORIENTATION_OFFSETS.slice(1),
    }),
    /requires 12 binary orientation offsets/,
  );

  const correlatedOrders = Array.from(
    { length: FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS },
    (_, repetitionIndex) => (repetitionIndex % 2 === 0
      ? [...FIRST_INSTANCE_LIVE_CROSSOVER_LANES]
      : [...FIRST_INSTANCE_LIVE_CROSSOVER_LANES].reverse()),
  );
  const correlatedOrientations = Array.from(
    { length: FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS },
    (_, repetitionIndex) => repetitionIndex % 2,
  );
  assert.throws(
    () => buildFirstInstanceLiveCrossoverPlan({
      ...PLAN_INPUT,
      lanePhysicalOrders: correlatedOrders,
      orientationOffsets: correlatedOrientations,
    }),
    /not pairwise balanced/,
  );
});

test('live first-instance schedule is complementary, balanced, and pool-bounded', () => {
  const [portable, feature] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE, 8);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS, 40);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS, 60);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES, 320);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES, 480);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES * 2, 640);
  assert.equal(FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES * 2, 960);
  assert.ok(FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES * 2 < 2_048);
  assert.deepEqual(
    FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS.map((pattern) => pattern.map(
      (lane) => (lane === portable ? 'P' : 'F'),
    ).join('')),
    ['PPPFPFFF', 'FFFPFPPP'],
  );
  for (const pattern of FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS) {
    assert.equal(pattern.filter((lane) => lane === portable).length, 4);
    assert.equal(pattern.filter((lane) => lane === feature).length, 4);
  }
  for (let frameIndex = 0; frameIndex < 16; frameIndex += 1) {
    assert.equal(firstInstanceLiveCrossoverFrame(frameIndex, 0).patternIndex, 0);
    assert.equal(firstInstanceLiveCrossoverFrame(frameIndex, 1).patternIndex, 1);
    assert.notEqual(
      firstInstanceLiveCrossoverFrame(frameIndex, 0).laneId,
      firstInstanceLiveCrossoverFrame(frameIndex, 1).laneId,
    );
  }
  const measured = Array.from(
    { length: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES },
    (_, frameIndex) => firstInstanceLiveCrossoverFrame(frameIndex, 0),
  );
  assert.equal(measured.filter((frame) => frame.laneId === portable).length, 240);
  assert.equal(measured.filter((frame) => frame.laneId === feature).length, 240);
  for (const orientationOffset of [0, 1]) {
    const block = firstInstanceLiveCrossoverHistoryCounts(8, orientationOffset);
    assert.deepEqual(Object.values(block.transitionCounts), [2, 2, 2, 2]);
    assert.deepEqual(Object.values(block.historyTripleCounts), Array(8).fill(1));
    const trial = firstInstanceLiveCrossoverHistoryCounts(
      FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
      orientationOffset,
    );
    assert.deepEqual(Object.values(trial.transitionCounts), [120, 120, 120, 120]);
    assert.deepEqual(Object.values(trial.historyTripleCounts), Array(8).fill(60));
    const firstMeasured = firstInstanceLiveCrossoverFrame(0, orientationOffset);
    const warmupPenultimate = firstInstanceLiveCrossoverFrame(
      FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES - 2,
      orientationOffset,
    );
    const warmupLast = firstInstanceLiveCrossoverFrame(
      FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES - 1,
      orientationOffset,
    );
    assert.equal(firstMeasured.previousPreviousLaneId, warmupPenultimate.laneId);
    assert.equal(firstMeasured.previousLaneId, warmupLast.laneId);
  }
  assert.equal(measured.at(-1).crossoverBlockIndex, 59);
  assert.throws(() => firstInstanceLiveCrossoverFrame(-1), /nonnegative integer/);
  assert.throws(() => firstInstanceLiveCrossoverFrame(0, 2), /zero or one/);
});
