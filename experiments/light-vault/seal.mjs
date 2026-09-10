import {sourceFromPhase, propagate, transfer, intensity} from './wave.mjs';
import {OPTICS, aperture, matchImage} from './targets.mjs';

const FORMAT = 'light-vault-seal-v1';
export const MAX_SEAL_BYTES = 3_500_000;
const COUNT = OPTICS.n * OPTICS.n, PHASE_LIMIT = Math.PI + 1e-5;
const FIXED = Object.freeze({
  n: OPTICS.n, pitch: OPTICS.pitch, wavelength: OPTICS.wavelength,
  apertureRadius: OPTICS.apertureRadius,
  distances: Object.freeze([...OPTICS.distances]), range: Object.freeze([...OPTICS.range]),
});
const numericArray = value => Array.isArray(value) || value instanceof Float32Array || value instanceof Float64Array;

function validatePhase(phase) {
  if (!numericArray(phase) || phase.length !== COUNT) throw Error('Seal needs exactly 65,536 phase values.');
  for (const value of phase)
    if (!Number.isFinite(value) || Math.abs(value) > PHASE_LIMIT) throw Error('Seal phase must be finite radians between −π and π.');
}

function validateReference(reference) {
  if (!Array.isArray(reference) || reference.length !== COUNT) throw Error('Seal reference must contain exactly 65,536 intensity bytes.');
  let ink = 0, bright = 0;
  for (const value of reference) {
    if (!Number.isInteger(value) || value < 0 || value > 255) throw Error('Seal reference intensities must be integers from 0 through 255.');
    ink += value / 255;
    if (value >= 64) bright++;
  }
  if (ink < 20 || bright < 20) throw Error('Seal references need a bolder mark.');
}

function quantizeReference(target, index) {
  const amplitude = target?.amplitude ?? target;
  if (!numericArray(amplitude) || amplitude.length !== COUNT) throw Error('Seal needs two complete target amplitude grids.');
  if (target?.distance !== undefined && target.distance !== FIXED.distances[index]) throw Error('Seal target depth does not match the fixed optics.');
  let maximum = 0;
  for (const value of amplitude) {
    if (!Number.isFinite(value) || value < 0 || value > 1e100) throw Error('Seal target amplitudes must be finite and nonnegative.');
    maximum = Math.max(maximum, value);
  }
  if (maximum === 0) throw Error('Seal references need a bolder mark.');
  // Squared amplitude normalized by its peak; division first avoids needless overflow.
  const reference = Array.from(amplitude, value => Math.round(255 * (value / maximum) ** 2));
  validateReference(reference);
  return reference;
}

function validateSize(text) {
  if (typeof text !== 'string') throw Error('Seal must be JSON text.');
  if (text.length > MAX_SEAL_BYTES || new TextEncoder().encode(text).byteLength > MAX_SEAL_BYTES)
    throw Error('Seal exceeds the 3.5 MB file limit.');
}

/** Reference masks are verification metadata; only phase drives the forward wave. */
export function serializeSeal(phase, targets) {
  validatePhase(phase);
  if (!Array.isArray(targets) || targets.length !== 2) throw Error('Seal needs exactly two image references.');
  const text = JSON.stringify({
    format: FORMAT, ...FIXED,
    phaseRadians: Array.from(Float32Array.from(phase)),
    references: targets.map(quantizeReference),
  });
  validateSize(text);
  return text;
}

/** Parses and physically verifies a fixed-optics seal before any scene state is changed. */
export function parseSeal(text) {
  validateSize(text);
  let seal;
  try { seal = JSON.parse(text); } catch { throw Error('Seal is not valid JSON.'); }
  if (!seal || typeof seal !== 'object' || Array.isArray(seal) || seal.format !== FORMAT)
    throw Error('Unsupported seal format.');
  for (const key of ['n', 'pitch', 'wavelength', 'apertureRadius'])
    if (seal[key] !== FIXED[key]) throw Error(`Seal ${key} does not match the fixed optics.`);
  for (const key of ['distances', 'range'])
    if (!Array.isArray(seal[key]) || seal[key].length !== FIXED[key].length || seal[key].some((value, i) => value !== FIXED[key][i]))
      throw Error(`Seal ${key} does not match the fixed optics.`);
  validatePhase(seal.phaseRadians);
  if (!Array.isArray(seal.references) || seal.references.length !== 2) throw Error('Seal needs exactly two image references.');
  seal.references.forEach(validateReference);

  const phase = Float64Array.from(seal.phaseRadians);
  const targets = seal.references.map((reference, index) => ({
    distance: FIXED.distances[index], amplitude: Float64Array.from(reference, value => Math.sqrt(value / 255)),
  }));
  // No reference pixels participate in this propagation. They are compared only afterwards.
  const field = sourceFromPhase(phase, aperture());
  const quality = FIXED.distances.map(distance => {
    const pixels = intensity(propagate(field, FIXED.n, transfer(FIXED.n, FIXED.pitch, FIXED.wavelength, distance)));
    return matchImage(pixels, targets);
  });
  if (!quality.every((result, index) => result.clear && result.best === index))
    throw Error('This phase plate does not produce two clear, distinct reference images at the design depths.');
  return {phase, targets, quality};
}
