import { FIRST_INSTANCE_LIVE_CROSSOVER_LANES } from './plan.js';

export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND =
  'first-instance-live-order-factorial-development';
export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_VISIBILITY_FRACTION = 0.99;
export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT = 2;
export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT = 16;
export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION = 32;
export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT = 64;
export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS = Object.freeze([
  'C',
  'K',
  'R',
  'T',
]);

const [PORTABLE, FEATURE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;
const PORTABLE_FIRST = Object.freeze([PORTABLE, FEATURE]);
const FEATURE_FIRST = Object.freeze([FEATURE, PORTABLE]);

// Each session executes its frozen 16-cell permutation and then its exact
// reverse. The two occurrences of every cell consequently occupy positions
// p and 31 - p within the session. These arrays are data, not generated from
// runtime state, and the builder exposes no schedule override.
export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS = Object.freeze([
  Object.freeze([0, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8]),
  Object.freeze([5, 7, 10, 1, 8, 3, 14, 13, 12, 15, 2, 9, 0, 11, 6, 4]),
]);

function orderForLevel(level) {
  return level === 0 ? PORTABLE_FIRST : FEATURE_FIRST;
}

function createCell(cellIndex) {
  const factorLevels = Object.freeze({
    C: (cellIndex >> 3) & 1,
    K: (cellIndex >> 2) & 1,
    R: (cellIndex >> 1) & 1,
    T: cellIndex & 1,
  });
  return Object.freeze({
    factorialCellIndex: cellIndex,
    factorialCellId: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS.map(
      (key) => `${key}${factorLevels[key]}`,
    ).join(''),
    factorLevels,
    laneConstructionOrder: orderForLevel(factorLevels.C),
    firstComputeUseOrder: orderForLevel(factorLevels.K),
    renderPipelinePrimeOrder: orderForLevel(factorLevels.R),
    timestampPreprimeLaneId: factorLevels.T === 0 ? PORTABLE : FEATURE,
  });
}

export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELLS = Object.freeze(
  Array.from(
    { length: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT },
    (_, cellIndex) => createCell(cellIndex),
  ),
);

function validateOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('Live order-factorial plan options must be an object.');
  }
  const keys = Object.keys(options);
  if (keys.length !== 1 || keys[0] !== 'runId') {
    throw new Error('Live order-factorial plan accepts exactly the runId option.');
  }
  if (typeof options.runId !== 'string' || options.runId.trim() === '') {
    throw new TypeError('Live order-factorial runId must be a nonempty string.');
  }
}

function orientationFor(cell, sessionIndex, blockIndex) {
  const { C, K, R, T } = cell.factorLevels;
  return C ^ K ^ R ^ T ^ sessionIndex ^ blockIndex;
}

export function buildFirstInstanceLiveOrderFactorialPlan(options) {
  validateOptions(options);
  const plan = [];

  for (let sessionIndex = 0;
    sessionIndex < FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT;
    sessionIndex += 1) {
    const permutation = FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_PERMUTATIONS[
      sessionIndex
    ];
    const blocks = [permutation, [...permutation].reverse()];

    for (let blockIndex = 0; blockIndex < blocks.length; blockIndex += 1) {
      const block = blocks[blockIndex];
      for (let blockPosition = 0; blockPosition < block.length; blockPosition += 1) {
        const factorialCellIndex = block[blockPosition];
        const cell = FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELLS[factorialCellIndex];
        const sessionTrialIndex = (
          blockIndex * FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT
        ) + blockPosition;
        const planIndex = (
          sessionIndex * FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION
        ) + sessionTrialIndex;

        plan.push(Object.freeze({
          schemaVersion: 1,
          kind: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND,
          trialId: `${options.runId}-s${sessionIndex + 1}-t${String(
            sessionTrialIndex + 1,
          ).padStart(2, '0')}`,
          planIndex,
          sessionIndex,
          sessionOrdinal: sessionIndex + 1,
          sessionTrialIndex,
          sessionTrialOrdinal: sessionTrialIndex + 1,
          permutationBlockIndex: blockIndex,
          permutationDirection: blockIndex === 0 ? 'forward' : 'reverse',
          permutationPosition: blockPosition,
          sessionCellOccurrenceIndex: blockIndex,
          factorialCellIndex,
          factorialCellId: cell.factorialCellId,
          factorLevels: cell.factorLevels,
          laneConstructionOrder: cell.laneConstructionOrder,
          firstComputeUseOrder: cell.firstComputeUseOrder,
          renderPipelinePrimeOrder: cell.renderPipelinePrimeOrder,
          timestampPreprimeLaneId: cell.timestampPreprimeLaneId,
          visibilityFraction: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_VISIBILITY_FRACTION,
          superblockOrientationOffset: orientationFor(cell, sessionIndex, blockIndex),
        }));
      }
    }
  }

  if (plan.length !== FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT) {
    throw new Error('Live order-factorial plan did not produce its frozen trial count.');
  }
  return Object.freeze(plan);
}
