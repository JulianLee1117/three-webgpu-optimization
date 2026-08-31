import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BENCHMARK_VISIBILITY_LEVELS,
  DEPTH_ORDERING_LAYOUTS,
  DEPTH_ORDERING_MODES,
  DEPTH_ORDERING_VISIBILITY,
  FIXED_SLICE_REPRESENTATION_MODES,
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
  FROZEN_DEPTH_CROSSOVER_ORIENTATION_OFFSETS,
  FROZEN_DEPTH_CROSSOVER_REPETITIONS,
  assertBalancedModeOrders,
  buildBenchmarkPlan,
  buildDepthOrderingPlan,
  buildFrozenDepthCrossoverPlan,
  createEcosystemModeOrders,
  createRepresentationModeOrders,
  rotateValues,
} from '../src/benchmark/plan.js';
import {
  FROZEN_CROSSOVER_MEASURED_BLOCKS,
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  FROZEN_CROSSOVER_PATTERNS,
  FROZEN_CROSSOVER_WARMUP_BLOCKS,
  FROZEN_CROSSOVER_WARMUP_FRAMES,
  frozenCrossoverFrame,
} from '../src/benchmark/frozen-crossover-schedule.js';

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

test('depth-ordering plan balances all mode positions and alternates overlap layout order', () => {
  const modeOrders = createEcosystemModeOrders(DEPTH_ORDERING_MODES);
  assertBalancedModeOrders(DEPTH_ORDERING_MODES, modeOrders);
  const plan = buildDepthOrderingPlan({
    runId: 'depth',
    modeOrders,
    objectCount: 65_536,
    bucketCount: 32,
  });

  assert.equal(plan.length, 36);
  assert.equal(new Set(plan.map((trial) => trial.trialId)).size, 36);
  for (let repetitionIndex = 0; repetitionIndex < 6; repetitionIndex += 1) {
    const trials = plan.filter((trial) => trial.repetitionIndex === repetitionIndex);
    const expectedLayoutOrder = repetitionIndex % 2 === 0
      ? [...DEPTH_ORDERING_LAYOUTS]
      : [...DEPTH_ORDERING_LAYOUTS].reverse();
    assert.equal(trials.length, 6);
    assert.deepEqual(trials[0].layoutOrder, expectedLayoutOrder);
    assert.ok(trials.every((trial) => (
      trial.visibilityFraction === DEPTH_ORDERING_VISIBILITY
      && trial.visibilityOrderPosition === 0
    )));
    for (let layoutPosition = 0; layoutPosition < 2; layoutPosition += 1) {
      const triplet = trials.slice(layoutPosition * 3, layoutPosition * 3 + 3);
      assert.deepEqual(triplet.map((trial) => trial.modeId), modeOrders[repetitionIndex]);
      assert.deepEqual(triplet.map((trial) => trial.modeOrderPosition), [0, 1, 2]);
      assert.ok(triplet.every((trial) => (
        trial.layout === expectedLayoutOrder[layoutPosition]
        && trial.layoutOrderPosition === layoutPosition
      )));
    }
  }

  for (const layout of DEPTH_ORDERING_LAYOUTS) {
    for (const mode of DEPTH_ORDERING_MODES) {
      const cell = plan.filter((trial) => trial.layout === layout && trial.modeId === mode);
      assert.equal(cell.length, 6);
      for (let modePosition = 0; modePosition < 3; modePosition += 1) {
        assert.equal(cell.filter((trial) => trial.modeOrderPosition === modePosition).length, 2);
      }
    }
  }
});

test('frozen depth crossover plan orthogonally balances layout position and lane storage order', () => {
  const plan = buildFrozenDepthCrossoverPlan({
    runId: 'frozen-depth',
    objectCount: 65_536,
    bucketCount: 32,
  });

  assert.equal(plan.length, FROZEN_DEPTH_CROSSOVER_REPETITIONS * 2);
  assert.equal(new Set(plan.map((trial) => trial.trialId)).size, plan.length);
  for (const trial of plan) {
    assert.equal(trial.modeId, FROZEN_DEPTH_CROSSOVER_MODE);
    assert.deepEqual(trial.modeOrder, [FROZEN_DEPTH_CROSSOVER_MODE]);
    assert.equal(trial.modeOrderPosition, 0);
    assert.deepEqual([...trial.laneStorageOrder].sort(), [...FROZEN_DEPTH_CROSSOVER_LANES].sort());
    assert.equal(trial.visibilityFraction, DEPTH_ORDERING_VISIBILITY);
    assert.equal(
      trial.superblockOrientationOffset,
      FROZEN_DEPTH_CROSSOVER_ORIENTATION_OFFSETS[trial.repetitionIndex],
    );
  }

  for (const layout of DEPTH_ORDERING_LAYOUTS) {
    const cell = plan.filter((trial) => trial.layout === layout);
    assert.equal(cell.length, FROZEN_DEPTH_CROSSOVER_REPETITIONS);
    for (let layoutPosition = 0; layoutPosition < 2; layoutPosition += 1) {
      const positionCell = cell.filter(
        (trial) => trial.layoutOrderPosition === layoutPosition,
      );
      assert.equal(positionCell.length, FROZEN_DEPTH_CROSSOVER_REPETITIONS / 2);
      for (const firstLane of FROZEN_DEPTH_CROSSOVER_LANES) {
        assert.equal(
          positionCell.filter((trial) => trial.laneStorageOrder[0] === firstLane).length,
          FROZEN_DEPTH_CROSSOVER_REPETITIONS / 4,
        );
      }
    }
  }

  const factorialCells = new Map();
  for (let repetition = 0; repetition < FROZEN_DEPTH_CROSSOVER_REPETITIONS; repetition += 1) {
    const trial = plan.find((entry) => entry.repetitionIndex === repetition);
    const storageFirst = trial.laneStorageOrder[0] === FROZEN_DEPTH_CROSSOVER_LANES[0] ? 0 : 1;
    const key = `${storageFirst}|${trial.superblockOrientationOffset}`;
    factorialCells.set(key, (factorialCells.get(key) ?? 0) + 1);
  }
  assert.deepEqual([...factorialCells.values()].sort(), [3, 3, 3, 3]);
});

test('frozen crossover schedule balances lanes, block position, and pattern', () => {
  assert.equal(FROZEN_CROSSOVER_WARMUP_BLOCKS, 40);
  assert.equal(FROZEN_CROSSOVER_WARMUP_FRAMES, 320);
  assert.equal(FROZEN_CROSSOVER_MEASURED_BLOCKS, 60);
  assert.equal(FROZEN_CROSSOVER_MEASURED_FRAMES, 480);
  assert.deepEqual(FROZEN_CROSSOVER_PATTERNS, [
    [
      FROZEN_DEPTH_CROSSOVER_LANES[0], FROZEN_DEPTH_CROSSOVER_LANES[1],
      FROZEN_DEPTH_CROSSOVER_LANES[0], FROZEN_DEPTH_CROSSOVER_LANES[1],
      FROZEN_DEPTH_CROSSOVER_LANES[1], FROZEN_DEPTH_CROSSOVER_LANES[0],
      FROZEN_DEPTH_CROSSOVER_LANES[1], FROZEN_DEPTH_CROSSOVER_LANES[0],
    ],
    [
      FROZEN_DEPTH_CROSSOVER_LANES[1], FROZEN_DEPTH_CROSSOVER_LANES[0],
      FROZEN_DEPTH_CROSSOVER_LANES[1], FROZEN_DEPTH_CROSSOVER_LANES[0],
      FROZEN_DEPTH_CROSSOVER_LANES[0], FROZEN_DEPTH_CROSSOVER_LANES[1],
      FROZEN_DEPTH_CROSSOVER_LANES[0], FROZEN_DEPTH_CROSSOVER_LANES[1],
    ],
  ]);

  const frames = Array.from(
    { length: FROZEN_CROSSOVER_MEASURED_FRAMES },
    (_, frameIndex) => frozenCrossoverFrame(frameIndex),
  );
  for (const laneId of FROZEN_DEPTH_CROSSOVER_LANES) {
    assert.equal(frames.filter((frame) => frame.laneId === laneId).length, 240);
    for (let position = 0; position < 8; position += 1) {
      assert.equal(
        frames.filter(
          (frame) => frame.laneId === laneId && frame.withinBlockPosition === position,
        ).length,
        30,
      );
    }
  }
  assert.equal(frames.filter((frame) => frame.pattern === 'FRFRRFRF').length, 240);
  assert.equal(frames.filter((frame) => frame.pattern === 'RFRFFRFR').length, 240);
  assert.deepEqual(
    Array.from({ length: 8 }, (_, index) => frozenCrossoverFrame(index, 1).laneId),
    FROZEN_CROSSOVER_PATTERNS[1],
  );
});
