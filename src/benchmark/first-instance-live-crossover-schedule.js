import { FIRST_INSTANCE_LIVE_CROSSOVER_LANES } from './plan.js';

export const FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE = 8;
export const FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS = 40;
export const FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS = 60;
export const FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES =
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_BLOCKS * FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE;
export const FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES =
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS * FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE;

const [PORTABLE, FEATURE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;

export const FIRST_INSTANCE_LIVE_CROSSOVER_TRANSITION_KEYS = Object.freeze([
  `${PORTABLE}->${PORTABLE}`,
  `${PORTABLE}->${FEATURE}`,
  `${FEATURE}->${PORTABLE}`,
  `${FEATURE}->${FEATURE}`,
]);

export const FIRST_INSTANCE_LIVE_CROSSOVER_HISTORY_TRIPLE_KEYS = Object.freeze(
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES.flatMap((previousPreviousLaneId) => (
    FIRST_INSTANCE_LIVE_CROSSOVER_LANES.flatMap((previousLaneId) => (
      FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => `${previousPreviousLaneId}->${previousLaneId}->${laneId}`,
      )
    ))
  )),
);

export const FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS = Object.freeze([
  Object.freeze([
    PORTABLE, PORTABLE, PORTABLE, FEATURE,
    PORTABLE, FEATURE, FEATURE, FEATURE,
  ]),
  Object.freeze([
    FEATURE, FEATURE, FEATURE, PORTABLE,
    FEATURE, PORTABLE, PORTABLE, PORTABLE,
  ]),
]);

const PATTERN_LABELS = Object.freeze(['PPPFPFFF', 'FFFPFPPP']);

export function firstInstanceLiveCrossoverFrame(frameIndex, orientationOffset = 0) {
  if (!Number.isInteger(frameIndex) || frameIndex < 0) {
    throw new RangeError(
      'Live first-instance crossover frameIndex must be a nonnegative integer.',
    );
  }
  if (orientationOffset !== 0 && orientationOffset !== 1) {
    throw new RangeError(
      'Live first-instance crossover orientationOffset must be zero or one.',
    );
  }
  const crossoverBlockIndex = Math.floor(
    frameIndex / FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
  );
  const withinBlockPosition = frameIndex % FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE;
  const patternIndex = orientationOffset;
  const pattern = FIRST_INSTANCE_LIVE_CROSSOVER_PATTERNS[patternIndex];
  const previousWithinBlockPosition = (
    withinBlockPosition + FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE - 1
  ) % FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE;
  const previousPreviousWithinBlockPosition = (
    withinBlockPosition + FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE - 2
  ) % FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE;
  return Object.freeze({
    crossoverBlockIndex,
    withinBlockPosition,
    patternIndex,
    pattern: PATTERN_LABELS[patternIndex],
    previousPreviousLaneId: pattern[previousPreviousWithinBlockPosition],
    previousLaneId: pattern[previousWithinBlockPosition],
    laneId: pattern[withinBlockPosition],
  });
}

export function firstInstanceLiveCrossoverHistoryCounts(frameCount, orientationOffset = 0) {
  if (!Number.isInteger(frameCount) || frameCount <= 0) {
    throw new RangeError(
      'Live first-instance crossover history frameCount must be a positive integer.',
    );
  }
  const transitionCounts = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_TRANSITION_KEYS.map((key) => [key, 0]),
  );
  const historyTripleCounts = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_HISTORY_TRIPLE_KEYS.map((key) => [key, 0]),
  );
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const frame = firstInstanceLiveCrossoverFrame(frameIndex, orientationOffset);
    transitionCounts[`${frame.previousLaneId}->${frame.laneId}`] += 1;
    historyTripleCounts[
      `${frame.previousPreviousLaneId}->${frame.previousLaneId}->${frame.laneId}`
    ] += 1;
  }
  return Object.freeze({
    schemaVersion: 1,
    kind: 'live-first-instance-crossover-history-balance',
    frameCount,
    orientationOffset,
    transitionCounts: Object.freeze(transitionCounts),
    historyTripleCounts: Object.freeze(historyTripleCounts),
  });
}
