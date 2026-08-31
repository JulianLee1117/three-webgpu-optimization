import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BENCHMARK_VISIBILITY_LEVELS,
  FIXED_SLICE_REPRESENTATION_MODES,
  assertBalancedModeOrders,
  buildBenchmarkPlan,
  createRepresentationModeOrders,
  rotateValues,
} from '../src/benchmark/plan.js';

test('fixed-slice causal plan is the complete preregistered 36-trial AB/BA design', () => {
  const modeOrders = createRepresentationModeOrders();
  assertBalancedModeOrders(FIXED_SLICE_REPRESENTATION_MODES, modeOrders);
  const plan = buildBenchmarkPlan({
    runId: 'causal',
    modeOrders,
    visibilityLevels: BENCHMARK_VISIBILITY_LEVELS,
    objectCount: 16_384,
    bucketCount: 32,
  });

  assert.equal(plan.length, 36);
  assert.equal(new Set(plan.map((trial) => trial.trialId)).size, 36);
  for (let repetition = 0; repetition < 6; repetition += 1) {
    const trials = plan.filter((trial) => trial.repetitionIndex === repetition);
    const expectedModeOrder = repetition % 2 === 0
      ? [...FIXED_SLICE_REPRESENTATION_MODES]
      : [...FIXED_SLICE_REPRESENTATION_MODES].reverse();
    const expectedVisibilityOrder = rotateValues(BENCHMARK_VISIBILITY_LEVELS, repetition);
    assert.equal(trials.length, 6);
    assert.deepEqual(trials[0].modeOrder, expectedModeOrder);
    assert.deepEqual(trials[0].visibilityOrder, expectedVisibilityOrder);
    for (let visibilityPosition = 0; visibilityPosition < 3; visibilityPosition += 1) {
      const pair = trials.slice(visibilityPosition * 2, visibilityPosition * 2 + 2);
      assert.deepEqual(pair.map((trial) => trial.modeId), expectedModeOrder);
      assert.deepEqual(pair.map((trial) => trial.modeOrderPosition), [0, 1]);
      assert.ok(pair.every((trial) => (
        trial.visibilityFraction === expectedVisibilityOrder[visibilityPosition]
        && trial.visibilityOrderPosition === visibilityPosition
      )));
    }
  }

  for (const visibility of BENCHMARK_VISIBILITY_LEVELS) {
    for (const mode of FIXED_SLICE_REPRESENTATION_MODES) {
      assert.equal(
        plan.filter((trial) => (
          trial.visibilityFraction === visibility && trial.modeId === mode
        )).length,
        6,
      );
    }
  }
});
