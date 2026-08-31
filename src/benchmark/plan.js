export const FIXED_SLICE_REPRESENTATION_MODES = Object.freeze([
  'fixed-slice-per-bucket',
  'fixed-slice',
]);

export const BENCHMARK_VISIBILITY_LEVELS = Object.freeze([0.2, 0.8, 0.99]);

export const DEPTH_ORDERING_MODES = Object.freeze([
  'fixed-slice',
  'fixed-slice-depth-front-to-back',
  'fixed-slice-depth-reverse',
]);

export const DEPTH_ORDERING_LAYOUTS = Object.freeze([
  'high-overlap',
  'low-overlap',
]);

export const DEPTH_ORDERING_VISIBILITY = 0.99;

export const FROZEN_DEPTH_CROSSOVER_MODE = 'fixed-slice-depth-frozen-crossover';

export const FROZEN_DEPTH_CROSSOVER_LANES = Object.freeze([
  'fixed-slice-depth-front-to-back',
  'fixed-slice-depth-reverse',
]);

export const FROZEN_DEPTH_CROSSOVER_REPETITIONS = 12;

export const FROZEN_DEPTH_CROSSOVER_STORAGE_ORDERS = Object.freeze([
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES].reverse()),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES].reverse()),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES].reverse()),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES].reverse()),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES].reverse()),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES].reverse()),
]);

export const FROZEN_DEPTH_CROSSOVER_ORIENTATION_OFFSETS = Object.freeze([
  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
]);

export function createRepresentationModeOrders(modes = FIXED_SLICE_REPRESENTATION_MODES) {
  if (modes.length !== 2 || new Set(modes).size !== 2) {
    throw new Error('The fixed-slice representation design requires exactly two distinct modes.');
  }
  const forward = Object.freeze([...modes]);
  const reverse = Object.freeze([...modes].reverse());
  return Object.freeze([forward, reverse, forward, reverse, forward, reverse]);
}

export function createEcosystemModeOrders(modes) {
  if (modes.length !== 3 || new Set(modes).size !== 3) {
    throw new Error('The ecosystem design requires exactly three distinct modes.');
  }
  return Object.freeze([
    Object.freeze([modes[0], modes[1], modes[2]]),
    Object.freeze([modes[1], modes[2], modes[0]]),
    Object.freeze([modes[2], modes[0], modes[1]]),
    Object.freeze([modes[2], modes[1], modes[0]]),
    Object.freeze([modes[1], modes[0], modes[2]]),
    Object.freeze([modes[0], modes[2], modes[1]]),
  ]);
}

export function assertBalancedModeOrders(modes, orders) {
  if (new Set(modes).size !== modes.length || orders.length % modes.length !== 0) {
    throw new Error('Benchmark modes and order count cannot form a balanced design.');
  }
  const expectedPerPosition = orders.length / modes.length;
  for (const [orderIndex, order] of orders.entries()) {
    if (order.length !== modes.length
      || new Set(order).size !== modes.length
      || modes.some((mode) => !order.includes(mode))) {
      throw new Error(`Benchmark mode order ${orderIndex} is not an exact permutation.`);
    }
  }
  for (const mode of modes) {
    for (let position = 0; position < modes.length; position += 1) {
      const count = orders.filter((order) => order[position] === mode).length;
      if (count !== expectedPerPosition) {
        throw new Error(`Benchmark mode ${mode} is not position-balanced.`);
      }
    }
  }
}

export function rotateValues(values, offset) {
  const normalized = offset % values.length;
  return [...values.slice(normalized), ...values.slice(0, normalized)];
}

export function buildBenchmarkPlan({
  runId,
  modeOrders,
  visibilityLevels,
  objectCount,
  bucketCount,
}) {
  const plan = [];
  for (let repetitionIndex = 0; repetitionIndex < modeOrders.length; repetitionIndex += 1) {
    const modeOrder = [...modeOrders[repetitionIndex]];
    const visibilityOrder = rotateValues(visibilityLevels, repetitionIndex);
    for (let visibilityOrderPosition = 0;
      visibilityOrderPosition < visibilityOrder.length;
      visibilityOrderPosition += 1) {
      const visibilityFraction = visibilityOrder[visibilityOrderPosition];
      for (let modeOrderPosition = 0;
        modeOrderPosition < modeOrder.length;
        modeOrderPosition += 1) {
        const modeId = modeOrder[modeOrderPosition];
        const planIndex = plan.length;
        plan.push({
          trialId: `${runId}-t${String(planIndex + 1).padStart(2, '0')}`,
          planIndex,
          repetitionIndex,
          modeId,
          modeOrder: [...modeOrder],
          modeOrderPosition,
          visibilityFraction,
          visibilityOrder: [...visibilityOrder],
          visibilityOrderPosition,
          layout: 'baseline',
          layoutOrder: ['baseline'],
          layoutOrderPosition: 0,
          objectCount,
          bucketCount,
        });
      }
    }
  }
  return plan;
}

export function buildDepthOrderingPlan({
  runId,
  modeOrders,
  objectCount,
  bucketCount,
  layouts = DEPTH_ORDERING_LAYOUTS,
  visibilityFraction = DEPTH_ORDERING_VISIBILITY,
}) {
  if (layouts.length !== 2 || new Set(layouts).size !== 2) {
    throw new Error('The depth-ordering design requires exactly two distinct layouts.');
  }
  if (!Number.isFinite(visibilityFraction)
    || visibilityFraction <= 0
    || visibilityFraction > 1) {
    throw new RangeError('Depth-ordering visibilityFraction must be in (0, 1].');
  }

  const plan = [];
  for (let repetitionIndex = 0; repetitionIndex < modeOrders.length; repetitionIndex += 1) {
    const modeOrder = [...modeOrders[repetitionIndex]];
    const layoutOrder = repetitionIndex % 2 === 0
      ? [...layouts]
      : [...layouts].reverse();
    for (let layoutOrderPosition = 0;
      layoutOrderPosition < layoutOrder.length;
      layoutOrderPosition += 1) {
      const layout = layoutOrder[layoutOrderPosition];
      for (let modeOrderPosition = 0;
        modeOrderPosition < modeOrder.length;
        modeOrderPosition += 1) {
        const modeId = modeOrder[modeOrderPosition];
        const planIndex = plan.length;
        plan.push({
          trialId: `${runId}-t${String(planIndex + 1).padStart(2, '0')}`,
          planIndex,
          repetitionIndex,
          modeId,
          modeOrder: [...modeOrder],
          modeOrderPosition,
          visibilityFraction,
          visibilityOrder: [visibilityFraction],
          visibilityOrderPosition: 0,
          layout,
          layoutOrder: [...layoutOrder],
          layoutOrderPosition,
          objectCount,
          bucketCount,
        });
      }
    }
  }
  return plan;
}

export function buildFrozenDepthCrossoverPlan({
  runId,
  objectCount,
  bucketCount,
  layouts = DEPTH_ORDERING_LAYOUTS,
  visibilityFraction = DEPTH_ORDERING_VISIBILITY,
  storageOrders = FROZEN_DEPTH_CROSSOVER_STORAGE_ORDERS,
  orientationOffsets = FROZEN_DEPTH_CROSSOVER_ORIENTATION_OFFSETS,
}) {
  if (layouts.length !== 2 || new Set(layouts).size !== 2) {
    throw new Error('The frozen depth crossover requires exactly two distinct layouts.');
  }
  if (storageOrders.length !== FROZEN_DEPTH_CROSSOVER_REPETITIONS) {
    throw new Error(
      `The frozen depth crossover requires ${FROZEN_DEPTH_CROSSOVER_REPETITIONS} storage orders.`,
    );
  }
  if (orientationOffsets.length !== FROZEN_DEPTH_CROSSOVER_REPETITIONS
    || orientationOffsets.some((value) => value !== 0 && value !== 1)) {
    throw new Error(
      `The frozen depth crossover requires ${FROZEN_DEPTH_CROSSOVER_REPETITIONS} binary orientation offsets.`,
    );
  }
  if (!Number.isFinite(visibilityFraction)
    || visibilityFraction <= 0
    || visibilityFraction > 1) {
    throw new RangeError('Frozen depth crossover visibilityFraction must be in (0, 1].');
  }

  const plan = [];
  for (let repetitionIndex = 0;
    repetitionIndex < FROZEN_DEPTH_CROSSOVER_REPETITIONS;
    repetitionIndex += 1) {
    const layoutOrder = repetitionIndex % 2 === 0
      ? [...layouts]
      : [...layouts].reverse();
    const laneStorageOrder = [...storageOrders[repetitionIndex]];
    if (laneStorageOrder.length !== FROZEN_DEPTH_CROSSOVER_LANES.length
      || new Set(laneStorageOrder).size !== FROZEN_DEPTH_CROSSOVER_LANES.length
      || FROZEN_DEPTH_CROSSOVER_LANES.some((lane) => !laneStorageOrder.includes(lane))) {
      throw new Error(`Frozen depth crossover storage order ${repetitionIndex} is invalid.`);
    }
    for (let layoutOrderPosition = 0;
      layoutOrderPosition < layoutOrder.length;
      layoutOrderPosition += 1) {
      const layout = layoutOrder[layoutOrderPosition];
      const planIndex = plan.length;
      plan.push({
        trialId: `${runId}-t${String(planIndex + 1).padStart(2, '0')}`,
        planIndex,
        repetitionIndex,
        modeId: FROZEN_DEPTH_CROSSOVER_MODE,
        modeOrder: [FROZEN_DEPTH_CROSSOVER_MODE],
        modeOrderPosition: 0,
        visibilityFraction,
        visibilityOrder: [visibilityFraction],
        visibilityOrderPosition: 0,
        layout,
        layoutOrder: [...layoutOrder],
        layoutOrderPosition,
        laneStorageOrder,
        superblockOrientationOffset: orientationOffsets[repetitionIndex],
        objectCount,
        bucketCount,
      });
    }
  }
  return plan;
}
