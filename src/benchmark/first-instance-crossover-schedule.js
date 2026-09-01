import { FIRST_INSTANCE_CROSSOVER_LANES } from './plan.js';

export const FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE = 8;
export const FIRST_INSTANCE_CROSSOVER_WARMUP_BLOCKS = 40;
export const FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS = 60;
export const FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES =
  FIRST_INSTANCE_CROSSOVER_WARMUP_BLOCKS * FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE;
export const FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES =
  FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS * FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE;

const [PORTABLE, FEATURE] = FIRST_INSTANCE_CROSSOVER_LANES;

export const FIRST_INSTANCE_CROSSOVER_PATTERNS = Object.freeze([
  Object.freeze([
    PORTABLE, FEATURE, PORTABLE, FEATURE,
    FEATURE, PORTABLE, FEATURE, PORTABLE,
  ]),
  Object.freeze([
    FEATURE, PORTABLE, FEATURE, PORTABLE,
    PORTABLE, FEATURE, PORTABLE, FEATURE,
  ]),
]);

export function firstInstanceCrossoverFrame(frameIndex, orientationOffset = 0) {
  if (!Number.isInteger(frameIndex) || frameIndex < 0) {
    throw new RangeError('First-instance crossover frameIndex must be a nonnegative integer.');
  }
  if (orientationOffset !== 0 && orientationOffset !== 1) {
    throw new RangeError('First-instance crossover orientationOffset must be zero or one.');
  }
  const crossoverBlockIndex = Math.floor(
    frameIndex / FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE,
  );
  const withinBlockPosition = frameIndex % FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE;
  const patternIndex = (crossoverBlockIndex + orientationOffset)
    % FIRST_INSTANCE_CROSSOVER_PATTERNS.length;
  const pattern = FIRST_INSTANCE_CROSSOVER_PATTERNS[patternIndex];
  return Object.freeze({
    crossoverBlockIndex,
    withinBlockPosition,
    patternIndex,
    pattern: patternIndex === 0 ? 'PFPFFPFP' : 'FPFPPFPF',
    laneId: pattern[withinBlockPosition],
  });
}
