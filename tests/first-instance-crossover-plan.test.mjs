import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS,
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS,
  FIRST_INSTANCE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS,
  buildFirstInstanceCrossoverPlan,
} from '../src/benchmark/plan.js';

const PLAN_INPUT = Object.freeze({
  runId: 'first-instance',
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
  return [...counts.values()].sort((a, b) => a - b);
}

test('first-instance crossover constants freeze the preregistered 12-repetition matrix', () => {
  assert.equal(FIRST_INSTANCE_CROSSOVER_MODE, 'indirect-first-instance-frozen-crossover');
  assert.deepEqual(FIRST_INSTANCE_CROSSOVER_LANES, ['portable', 'feature']);
  assert.equal(FIRST_INSTANCE_CROSSOVER_REPETITIONS, 12);
  assert.deepEqual(FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS, [0.99, 0.2]);
  assert.equal(FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS.length, 12);
  assert.equal(FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS.length, 12);

  assert.equal(Object.isFrozen(FIRST_INSTANCE_CROSSOVER_LANES), true);
  assert.equal(Object.isFrozen(FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS), true);
  assert.equal(Object.isFrozen(FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS), true);
  assert.ok(FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS.every(Object.isFrozen));
  assert.equal(Object.isFrozen(FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS), true);
});

test('first-instance crossover plan has exactly two visibility trials per repetition', () => {
  const plan = buildFirstInstanceCrossoverPlan(PLAN_INPUT);

  assert.equal(plan.length, 24);
  assert.equal(new Set(plan.map((trial) => trial.trialId)).size, 24);
  assert.deepEqual(
    plan.map((trial) => trial.trialId),
    Array.from(
      { length: 24 },
      (_, index) => `first-instance-t${String(index + 1).padStart(2, '0')}`,
    ),
  );

  for (let repetitionIndex = 0;
    repetitionIndex < FIRST_INSTANCE_CROSSOVER_REPETITIONS;
    repetitionIndex += 1) {
    const trials = plan.filter((trial) => trial.repetitionIndex === repetitionIndex);
    const expectedVisibilityOrder = repetitionIndex % 2 === 0
      ? [...FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS]
      : [...FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS].reverse();
    const expectedCommandOrder = FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS[
      repetitionIndex
    ];

    assert.equal(trials.length, 2);
    assert.deepEqual(trials.map((trial) => trial.visibilityFraction), expectedVisibilityOrder);
    assert.deepEqual(trials.map((trial) => trial.visibilityOrderPosition), [0, 1]);
    for (const trial of trials) {
      assert.equal(trial.modeId, FIRST_INSTANCE_CROSSOVER_MODE);
      assert.deepEqual(trial.modeOrder, [FIRST_INSTANCE_CROSSOVER_MODE]);
      assert.equal(trial.modeOrderPosition, 0);
      assert.deepEqual(trial.visibilityOrder, expectedVisibilityOrder);
      assert.deepEqual(trial.laneCommandSegmentOrder, expectedCommandOrder);
      assert.notEqual(trial.laneCommandSegmentOrder, expectedCommandOrder);
      assert.equal(
        trial.superblockOrientationOffset,
        FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS[repetitionIndex],
      );
      assert.equal(trial.layout, 'baseline');
      assert.deepEqual(trial.layoutOrder, ['baseline']);
      assert.equal(trial.layoutOrderPosition, 0);
      assert.equal(trial.objectCount, PLAN_INPUT.objectCount);
      assert.equal(trial.bucketCount, PLAN_INPUT.bucketCount);
    }
  }

  for (const visibilityFraction of FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS) {
    assert.equal(
      plan.filter((trial) => trial.visibilityFraction === visibilityFraction).length,
      FIRST_INSTANCE_CROSSOVER_REPETITIONS,
    );
  }
});

test('command order, starting orientation, and visibility-order position are pairwise balanced', () => {
  const plan = buildFirstInstanceCrossoverPlan(PLAN_INPUT);
  const repetitions = Array.from(
    { length: FIRST_INSTANCE_CROSSOVER_REPETITIONS },
    (_, repetitionIndex) => {
      const trials = plan.filter((trial) => trial.repetitionIndex === repetitionIndex);
      const highVisibilityTrial = trials.find((trial) => trial.visibilityFraction === 0.99);
      return {
        commandOrder: trials[0].laneCommandSegmentOrder[0] === 'portable' ? 0 : 1,
        orientation: trials[0].superblockOrientationOffset,
        visibilityPosition: highVisibilityTrial.visibilityOrderPosition,
      };
    },
  );

  const exactPairCounts = [3, 3, 3, 3];
  assert.deepEqual(
    binaryFactorCounts(repetitions, (row) => row.commandOrder, (row) => row.orientation),
    exactPairCounts,
  );
  assert.deepEqual(
    binaryFactorCounts(repetitions, (row) => row.commandOrder, (row) => row.visibilityPosition),
    exactPairCounts,
  );
  assert.deepEqual(
    binaryFactorCounts(repetitions, (row) => row.orientation, (row) => row.visibilityPosition),
    exactPairCounts,
  );
});

test('first-instance crossover rejects changes to its exact factors or pairwise balance', () => {
  assert.throws(
    () => buildFirstInstanceCrossoverPlan({
      ...PLAN_INPUT,
      visibilityLevels: [0.2, 0.99],
    }),
    /visibility levels must be exactly \[0\.99, 0\.2\]/,
  );
  assert.throws(
    () => buildFirstInstanceCrossoverPlan({
      ...PLAN_INPUT,
      commandSegmentOrders: FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS.slice(1),
    }),
    /requires 12 command-segment orders/,
  );

  const invalidPermutation = FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS.map(
    (order) => [...order],
  );
  invalidPermutation[4] = ['portable', 'portable'];
  assert.throws(
    () => buildFirstInstanceCrossoverPlan({
      ...PLAN_INPUT,
      commandSegmentOrders: invalidPermutation,
    }),
    /not an exact lane permutation/,
  );
  assert.throws(
    () => buildFirstInstanceCrossoverPlan({
      ...PLAN_INPUT,
      orientationOffsets: FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS.slice(1),
    }),
    /requires 12 binary orientation offsets/,
  );
  assert.throws(
    () => buildFirstInstanceCrossoverPlan({
      ...PLAN_INPUT,
      orientationOffsets: [
        ...FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS.slice(0, -1),
        2,
      ],
    }),
    /requires 12 binary orientation offsets/,
  );

  const correlatedOrders = Array.from(
    { length: FIRST_INSTANCE_CROSSOVER_REPETITIONS },
    (_, repetitionIndex) => (repetitionIndex % 2 === 0
      ? [...FIRST_INSTANCE_CROSSOVER_LANES]
      : [...FIRST_INSTANCE_CROSSOVER_LANES].reverse()),
  );
  const correlatedOrientations = Array.from(
    { length: FIRST_INSTANCE_CROSSOVER_REPETITIONS },
    (_, repetitionIndex) => repetitionIndex % 2,
  );
  assert.throws(
    () => buildFirstInstanceCrossoverPlan({
      ...PLAN_INPUT,
      commandSegmentOrders: correlatedOrders,
      orientationOffsets: correlatedOrientations,
    }),
    /not pairwise balanced/,
  );
});
