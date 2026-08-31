import { FROZEN_DEPTH_CROSSOVER_LANES } from './plan.js';

export const FROZEN_CROSSOVER_BLOCK_SIZE = 8;
export const FROZEN_CROSSOVER_WARMUP_BLOCKS = 40;
export const FROZEN_CROSSOVER_MEASURED_BLOCKS = 60;
export const FROZEN_CROSSOVER_WARMUP_FRAMES =
  FROZEN_CROSSOVER_WARMUP_BLOCKS * FROZEN_CROSSOVER_BLOCK_SIZE;
export const FROZEN_CROSSOVER_MEASURED_FRAMES =
  FROZEN_CROSSOVER_MEASURED_BLOCKS * FROZEN_CROSSOVER_BLOCK_SIZE;

const [FRONT_TO_BACK, REVERSE] = FROZEN_DEPTH_CROSSOVER_LANES;

export const FROZEN_CROSSOVER_PATTERNS = Object.freeze([
  Object.freeze([
    FRONT_TO_BACK, REVERSE, FRONT_TO_BACK, REVERSE,
    REVERSE, FRONT_TO_BACK, REVERSE, FRONT_TO_BACK,
  ]),
  Object.freeze([
    REVERSE, FRONT_TO_BACK, REVERSE, FRONT_TO_BACK,
    FRONT_TO_BACK, REVERSE, FRONT_TO_BACK, REVERSE,
  ]),
]);

export function frozenCrossoverFrame(frameIndex, orientationOffset = 0) {
  if (!Number.isInteger(frameIndex) || frameIndex < 0) {
    throw new RangeError('Frozen crossover frameIndex must be a nonnegative integer.');
  }
  if (orientationOffset !== 0 && orientationOffset !== 1) {
    throw new RangeError('Frozen crossover orientationOffset must be zero or one.');
  }
  const crossoverBlockIndex = Math.floor(frameIndex / FROZEN_CROSSOVER_BLOCK_SIZE);
  const withinBlockPosition = frameIndex % FROZEN_CROSSOVER_BLOCK_SIZE;
  const patternIndex = (crossoverBlockIndex + orientationOffset)
    % FROZEN_CROSSOVER_PATTERNS.length;
  const pattern = FROZEN_CROSSOVER_PATTERNS[patternIndex];
  return Object.freeze({
    crossoverBlockIndex,
    withinBlockPosition,
    patternIndex,
    pattern: patternIndex === 0 ? 'FRFRRFRF' : 'RFRFFRFR',
    laneId: pattern[withinBlockPosition],
  });
}
