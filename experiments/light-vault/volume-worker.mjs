import {fft2, transfer, sourceFromPhase} from './wave.mjs';
import {OPTICS, aperture} from './targets.mjs';

const SLICES = 32, SIDE = 32, BLOCK = 4, MAX_POINTS = SLICES * SIDE * SIDE;
const BUDGET_MS = 4000;

/** Coarse scalar-wave samples, visualized as weak scattering; no target images are read. */
export function createVolumePreview(phase) {
  const started = performance.now();
  if (!(phase instanceof Float64Array) || phase.length !== OPTICS.n ** 2)
    throw Error('Volume preview requires one bounded Float64 phase grid.');

  const {n, pitch, wavelength} = OPTICS;
  const spectrum = fft2(sourceFromPhase(phase, aperture()), n);
  const field = new Float64Array(spectrum.length);
  const positions = new Float32Array(MAX_POINTS * 3), colors = new Float32Array(MAX_POINTS * 3);
  const ends = new Uint32Array(SLICES);
  let points = 0;

  for (let slice = 0; slice < SLICES; slice++) {
    if (performance.now() - started > BUDGET_MS) throw Error('Volume preview exceeded its four-second CPU budget.');
    const distance = .004 + slice / (SLICES - 1) * .034;
    const h = transfer(n, pitch, wavelength, distance);
    // Reuse the one source spectrum and one inverse-FFT workspace for every depth.
    for (let i = 0; i < field.length; i += 2) {
      field[i] = spectrum[i] * h[i] - spectrum[i + 1] * h[i + 1];
      field[i + 1] = spectrum[i] * h[i + 1] + spectrum[i + 1] * h[i];
    }
    fft2(field, n, true);

    for (let row = 0; row < SIDE; row++) for (let col = 0; col < SIDE; col++) {
      let sum = 0;
      for (let dy = 0; dy < BLOCK; dy++) for (let dx = 0; dx < BLOCK; dx++) {
        const i = 2 * ((64 + row * BLOCK + dy) * n + 64 + col * BLOCK + dx);
        sum += field[i] ** 2 + field[i + 1] ** 2;
      }
      const mean = sum / (BLOCK * BLOCK);
      if (mean <= .20) continue;
      const at = points * 3, gain = Math.min(1, mean * .14);
      positions[at] = -1.7 + (distance - .004) / .034 * 4.9;
      positions[at + 1] = 1.65 + (64 - (row * BLOCK + 2)) / 128 * 2.3;
      positions[at + 2] = ((col * BLOCK + 2) - 64) / 128 * 2.3;
      colors[at] = gain; colors[at + 1] = gain * .62; colors[at + 2] = gain * .19;
      points++;
    }
    ends[slice] = points;
  }

  const milliseconds = performance.now() - started;
  if (milliseconds > BUDGET_MS) throw Error('Volume preview exceeded its four-second CPU budget.');
  return {positions: positions.slice(0, points * 3), colors: colors.slice(0, points * 3), ends, milliseconds};
}

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = ({data}) => {
    const id = data?.id;
    try {
      const volume = createVolumePreview(data?.phase);
      self.postMessage({type: 'result', id, ...volume}, [volume.positions.buffer, volume.colors.buffer, volume.ends.buffer]);
    } catch (error) {
      self.postMessage({type: 'error', id, message: error.message ?? String(error)});
    }
  };
}
