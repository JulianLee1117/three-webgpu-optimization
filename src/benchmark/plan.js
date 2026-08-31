export const FIXED_SLICE_REPRESENTATION_MODES = Object.freeze([
  'fixed-slice-per-bucket',
  'fixed-slice',
]);

export const BENCHMARK_VISIBILITY_LEVELS = Object.freeze([0.2, 0.8, 0.99]);

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
          objectCount,
          bucketCount,
        });
      }
    }
  }
  return plan;
}
