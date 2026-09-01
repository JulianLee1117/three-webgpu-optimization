import assert from 'node:assert/strict';
import test from 'node:test';

import {
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELLS,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_VISIBILITY_FRACTION,
  buildFirstInstanceLiveOrderFactorialPlan,
} from '../src/benchmark/first-instance-live-order-factorial-plan.js';

const RUN_ID = 'first-instance-live-order-factorial';

function factorPairCounts(rows, leftKey, rightKey) {
  const counts = new Map([
    ['0|0', 0],
    ['0|1', 0],
    ['1|0', 0],
    ['1|1', 0],
  ]);
  for (const row of rows) {
    const key = `${row.factorLevels[leftKey]}|${row.factorLevels[rightKey]}`;
    counts.set(key, counts.get(key) + 1);
  }
  return [...counts.values()];
}

test('order-factorial constants freeze all 16 C/K/R/T cells and two permutations', () => {
  assert.equal(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND,
    'first-instance-live-order-factorial-development');
  assert.equal(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_VISIBILITY_FRACTION, 0.99);
  assert.equal(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT, 2);
  assert.equal(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT, 16);
  assert.equal(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION, 32);
  assert.equal(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT, 64);
  assert.deepEqual(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS, ['C', 'K', 'R', 'T']);
  assert.deepEqual(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS, [
    [0, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8],
    [5, 7, 10, 1, 8, 3, 14, 13, 12, 15, 2, 9, 0, 11, 6, 4],
  ]);

  assert.equal(Object.isFrozen(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS), true);
  assert.equal(Object.isFrozen(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELLS), true);
  assert.ok(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELLS.every(Object.isFrozen));
  assert.equal(Object.isFrozen(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS), true);
  assert.ok(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS.every(Object.isFrozen));
  assert.equal(new Set(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELLS.map(
    ({ factorialCellId }) => factorialCellId,
  )).size, 16);
  assert.deepEqual(
    FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELLS.map(({ factorialCellId }) => factorialCellId),
    Array.from({ length: 16 }, (_, index) => [
      `C${(index >> 3) & 1}`,
      `K${(index >> 2) & 1}`,
      `R${(index >> 1) & 1}`,
      `T${index & 1}`,
    ].join('')),
  );
  for (const permutation of FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS) {
    assert.equal(permutation.length, 16);
    assert.deepEqual([...permutation].sort((left, right) => left - right),
      Array.from({ length: 16 }, (_, index) => index));
  }
});

test('order-factorial plan is exactly two frozen permutations followed by reverse', () => {
  const plan = buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID });
  assert.equal(plan.length, 64);
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(new Set(plan.map(({ trialId }) => trialId)).size, 64);
  assert.deepEqual(plan.map(({ planIndex }) => planIndex),
    Array.from({ length: 64 }, (_, index) => index));

  for (let sessionIndex = 0; sessionIndex < 2; sessionIndex += 1) {
    const session = plan.filter((trial) => trial.sessionIndex === sessionIndex);
    const permutation = FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS[
      sessionIndex
    ];
    assert.equal(session.length, 32);
    assert.deepEqual(
      session.map(({ factorialCellIndex }) => factorialCellIndex),
      [...permutation, ...[...permutation].reverse()],
    );
    assert.deepEqual(
      session.map(({ sessionTrialIndex }) => sessionTrialIndex),
      Array.from({ length: 32 }, (_, index) => index),
    );
    assert.ok(session.every(({ visibilityFraction }) => visibilityFraction === 0.99));
    assert.deepEqual(
      session.slice(0, 16).map(({ permutationDirection }) => permutationDirection),
      Array(16).fill('forward'),
    );
    assert.deepEqual(
      session.slice(16).map(({ permutationDirection }) => permutationDirection),
      Array(16).fill('reverse'),
    );
  }
});

test('every cell has exact count, reverse-position balance, and balanced orientations', () => {
  const plan = buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID });
  for (let sessionIndex = 0; sessionIndex < 2; sessionIndex += 1) {
    const session = plan.filter((trial) => trial.sessionIndex === sessionIndex);
    assert.equal(session.filter(({ superblockOrientationOffset }) => (
      superblockOrientationOffset === 0
    )).length, 16);
    assert.equal(session.filter(({ superblockOrientationOffset }) => (
      superblockOrientationOffset === 1
    )).length, 16);
    for (const blockIndex of [0, 1]) {
      const block = session.filter((trial) => trial.permutationBlockIndex === blockIndex);
      assert.equal(block.filter(({ superblockOrientationOffset }) => (
        superblockOrientationOffset === 0
      )).length, 8);
      assert.equal(block.filter(({ superblockOrientationOffset }) => (
        superblockOrientationOffset === 1
      )).length, 8);
    }

    for (let cellIndex = 0; cellIndex < 16; cellIndex += 1) {
      const occurrences = session.filter(
        ({ factorialCellIndex }) => factorialCellIndex === cellIndex,
      );
      assert.equal(occurrences.length, 2);
      assert.deepEqual(
        occurrences.map(({ superblockOrientationOffset }) => (
          superblockOrientationOffset
        )).sort(),
        [0, 1],
      );
      assert.equal(
        occurrences[0].permutationPosition + occurrences[1].permutationPosition,
        15,
      );
      assert.equal(
        occurrences[0].sessionTrialIndex + occurrences[1].sessionTrialIndex,
        31,
      );
    }
  }

  for (let cellIndex = 0; cellIndex < 16; cellIndex += 1) {
    assert.equal(plan.filter(({ factorialCellIndex }) => (
      factorialCellIndex === cellIndex
    )).length, 4);
  }
});

test('factor levels and pairwise cells remain exact within each session and orientation', () => {
  const plan = buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID });
  for (let sessionIndex = 0; sessionIndex < 2; sessionIndex += 1) {
    const session = plan.filter((trial) => trial.sessionIndex === sessionIndex);
    for (const factorKey of FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS) {
      for (const level of [0, 1]) {
        const levelRows = session.filter(({ factorLevels }) => factorLevels[factorKey] === level);
        assert.equal(levelRows.length, 16);
        assert.equal(levelRows.filter(({ superblockOrientationOffset }) => (
          superblockOrientationOffset === 0
        )).length, 8);
        assert.equal(levelRows.filter(({ superblockOrientationOffset }) => (
          superblockOrientationOffset === 1
        )).length, 8);
      }
    }
    for (let leftIndex = 0;
      leftIndex < FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS.length;
      leftIndex += 1) {
      for (let rightIndex = leftIndex + 1;
        rightIndex < FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS.length;
        rightIndex += 1) {
        assert.deepEqual(factorPairCounts(
          session,
          FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS[leftIndex],
          FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS[rightIndex],
        ), [8, 8, 8, 8]);
      }
    }
  }
});

test('plan construction is deterministic, deeply frozen, and has no schedule override', () => {
  const first = buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID });
  const second = buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID });
  assert.deepEqual(first, second);
  for (const trial of first) {
    assert.equal(Object.isFrozen(trial), true);
    assert.equal(Object.isFrozen(trial.factorLevels), true);
    assert.equal(Object.isFrozen(trial.laneConstructionOrder), true);
    assert.equal(Object.isFrozen(trial.firstComputeUseOrder), true);
    assert.equal(Object.isFrozen(trial.renderPipelinePrimeOrder), true);
  }
  assert.throws(() => buildFirstInstanceLiveOrderFactorialPlan(), /must be an object/);
  assert.throws(
    () => buildFirstInstanceLiveOrderFactorialPlan({ runId: '' }),
    /nonempty string/,
  );
  assert.throws(
    () => buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID, orientation: 1 }),
    /accepts exactly the runId option/,
  );
});
