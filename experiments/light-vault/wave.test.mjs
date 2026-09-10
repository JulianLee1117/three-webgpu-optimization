import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fft2, transfer, propagate, intensity, sourceFromPhase, solveHologram} from './wave.mjs';

const norm = a => a.reduce((sum, v) => sum + v * v, 0);
function relativeError(a, b) {
  let error = 0;
  for (let i = 0; i < a.length; i++) error += (a[i] - b[i]) ** 2;
  return Math.sqrt(error / Math.max(norm(b), 1e-30));
}
function fixture(n) {
  return Float64Array.from({length: 2 * n * n}, (_, i) => Math.sin(i * 1.718) + .3 * Math.cos(i * .137));
}
// Deliberately direct O(N^4) oracle: no FFT, butterfly, or production frequency code.
function directDFT(input, n, inverse = false) {
  const out = new Float64Array(input.length), sign = inverse ? 1 : -1;
  for (let ky = 0; ky < n; ky++) for (let kx = 0; kx < n; kx++) {
    const to = 2 * (ky * n + kx);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const from = 2 * (y * n + x), angle = sign * 2 * Math.PI * (kx * x + ky * y) / n;
      out[to] += input[from] * Math.cos(angle) - input[from + 1] * Math.sin(angle);
      out[to + 1] += input[from] * Math.sin(angle) + input[from + 1] * Math.cos(angle);
    }
  }
  if (inverse) for (let i = 0; i < out.length; i++) out[i] /= n * n;
  return out;
}

test('8 by 8 FFT agrees with an independent direct complex DFT in both directions', () => {
  const input = fixture(8);
  for (const inverse of [false, true])
    assert.ok(relativeError(fft2(input.slice(), 8, inverse), directDFT(input, 8, inverse)) < 1e-13);
});

test('FFT round trip and Parseval normalization preserve complex field energy', () => {
  const n = 64, input = fixture(n), transformed = fft2(input.slice(), n);
  assert.ok(Math.abs(norm(transformed) / (n * n * norm(input)) - 1) < 2e-13);
  assert.ok(relativeError(fft2(transformed, n, true), input) < 2e-13);
});

test('a sampled plane wave acquires the analytic angular-spectrum phase', () => {
  const n = 32, pitch = 10e-6, wavelength = 532e-9, distance = .03, qx = 3, qy = -2;
  const input = new Float64Array(2 * n * n), expected = new Float64Array(input.length);
  // Physical longitudinal wave number, independently evaluated rather than calling transfer().
  const k = 2 * Math.PI / wavelength, kx = 2 * Math.PI * qx / (n * pitch), ky = 2 * Math.PI * qy / (n * pitch);
  const delta = distance * (Math.sqrt(k * k - kx * kx - ky * ky) - k);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = 2 * (y * n + x), a = 2 * Math.PI * (qx * x + qy * y) / n;
    input[i] = Math.cos(a); input[i + 1] = Math.sin(a);
    expected[i] = Math.cos(a + delta); expected[i + 1] = Math.sin(a + delta);
  }
  assert.ok(relativeError(propagate(input, n, transfer(n, pitch, wavelength, distance)), expected) < 1e-9);
});

test('forward and conjugate propagation are inverses when no spectral samples are discarded', () => {
  const n = 64, input = fixture(n), h = transfer(n, 10e-6, 532e-9, .03);
  const forward = propagate(input, n, h), reverse = propagate(forward, n, h, true);
  assert.ok(relativeError(reverse, input) < 3e-13);
  assert.ok(Math.abs(norm(forward) / norm(input) - 1) < 3e-13);
});

test('an evanescent-frequency mask is an adjoint projection, not an invertible propagator', () => {
  const n = 8, h = transfer(n, 100e-9, 532e-9, .001), input = fixture(n);
  assert.ok(Array.from(h).some((v, i) => i % 2 === 0 && v === 0 && h[i + 1] === 0));
  const projected = propagate(propagate(input, n, h), n, h, true);
  assert.ok(norm(projected) < norm(input));
  assert.ok(relativeError(projected, input) > .1);
});

test('grid and optical bounds reject invalid sizes and units before allocating propagation buffers', () => {
  for (const n of [NaN, Infinity, 0, 7, 9, 8.5, 1024, 2 ** 30])
    assert.throws(() => transfer(n, 10e-6, 532e-9, .03), /Grid/);
  for (const [pitch, wavelength, distance] of [[0, 532e-9, .03], [NaN, 532e-9, .03], [1e-300, 532e-9, .03], [10e-6, Infinity, .03], [10e-6, -1, .03], [10e-6, 532e-9, NaN], [10e-6, 532e-9, .501]])
    assert.throws(() => transfer(8, pitch, wavelength, distance), /optical units/);
});

test('invalid complex fields fail before FFT mutation and invalid transfer functions cannot propagate', () => {
  const field = fixture(8), h = transfer(8, 10e-6, 532e-9, .03);
  const invalid = field.slice(); invalid[25] = NaN; const before = invalid.slice();
  assert.throws(() => fft2(invalid, 8), /sample/);
  assert.deepEqual(invalid, before);
  assert.throws(() => fft2(new Float32Array(field), 8), /type/);
  assert.throws(() => fft2(field.slice(), 8, 'inverse'), /boolean/);
  assert.throws(() => propagate(field, 8, h.subarray(2)), /length/);
  const invalidTransfer = h.slice(); invalidTransfer[11] = Infinity;
  assert.throws(() => propagate(field, 8, invalidTransfer), /sample/);
  const activeTransfer = h.slice(); activeTransfer[0] = 2;
  assert.throws(() => propagate(field, 8, activeTransfer), /passive/);
  assert.throws(() => propagate(field, 8, h, 1), /boolean/);
  assert.throws(() => intensity(new Float64Array(127)), /Grid/);
  assert.throws(() => intensity(invalid), /sample/);
});

test('source arrays require a matching bounded grid, finite phase and nonnegative aperture', () => {
  const phase = new Float64Array(64), aperture = new Float64Array(64).fill(1);
  assert.throws(() => sourceFromPhase(phase, aperture.subarray(1)), /length/);
  const badPhase = phase.slice(); badPhase[0] = NaN;
  assert.throws(() => sourceFromPhase(badPhase, aperture), /phase sample/);
  const badAperture = aperture.slice(); badAperture[0] = -1;
  assert.throws(() => sourceFromPhase(phase, badAperture), /aperture sample/);
  badAperture[0] = Infinity;
  assert.throws(() => sourceFromPhase(phase, badAperture), /aperture sample/);
  assert.throws(() => sourceFromPhase(new Float64Array(513 ** 2), new Float64Array(0)), /Grid/);
});

test('solver rejects malformed targets and unbounded work before fitting', () => {
  const options = {n: 8, apertureRadius: 3, iterations: 1}, amplitude = new Float64Array(64).fill(1);
  const target = {amplitude, distance: .012};
  for (const targets of [null, [], [target, target, target, target], [null], [{}]])
    assert.throws(() => solveHologram(targets, options));
  for (const value of [NaN, Infinity, -1, 1e200]) {
    const bad = amplitude.slice(); bad[25] = value;
    assert.throws(() => solveHologram([{amplitude: bad, distance: .012}], options), /target amplitude sample/);
  }
  assert.throws(() => solveHologram([{amplitude: new Float64Array(64), distance: .012}], options), /target energy/);
  assert.throws(() => solveHologram([{amplitude, distance: NaN}], options), /optical units/);
  for (const iterations of [0, -1, 121, NaN, Infinity, 1.5])
    assert.throws(() => solveHologram([target], {...options, iterations}), /Iterations/);
  for (const apertureRadius of [0, .5, 4.1, NaN, Infinity])
    assert.throws(() => solveHologram([target], {...options, apertureRadius}), /radius/);
  for (const seed of [-1, NaN, 1.5, 2 ** 32])
    assert.throws(() => solveHologram([target], {...options, seed}), /Seed/);
  assert.throws(() => solveHologram([target], {...options, onProgress: true}), /onProgress/);
  assert.throws(() => solveHologram([target], {...options, n: 2 ** 30}), /Grid/);
});

test('bounded fitting preserves the fixed aperture and deterministic finite phase', () => {
  const target = {amplitude: new Float64Array(64).fill(1), distance: 0};
  const options = {n: 8, apertureRadius: 3, iterations: 2, seed: 731};
  const solution = solveHologram([target], options), repeated = solveHologram([target], options);
  assert.deepEqual(solution.phase, repeated.phase);
  assert.ok(solution.phase.every(Number.isFinite));
  assert.ok(solution.history.every(entry => Number.isFinite(entry.loss)));
  const pixels = intensity(sourceFromPhase(solution.phase, solution.aperture));
  assert.ok(relativeError(pixels, solution.aperture) < 1e-15);
  assert.equal(solution.energy, solution.aperture.reduce((sum, a) => sum + a * a, 0));
});

test('committed phase plate remains stable when the empty padding doubles at fixed physical sampling', async () => {
  const solved = JSON.parse(await readFile(new URL('./assets/initial-phase.json', import.meta.url), 'utf8'));
  const bytes = await readFile(new URL('./assets/initial-phase.bin', import.meta.url));
  const {n, pitch, wavelength} = solved;
  assert.equal(n, 256, 'This bounded canary only compares 256 with 512, without refitting the phase.');
  assert.equal(bytes.byteLength, n * n * 4, 'Committed phase is one little-endian Float32 per source pixel.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const phase = Float64Array.from({length: n * n}, (_, i) => view.getFloat32(i * 4, true));
  const aperture = Float64Array.from({length: n * n}, (_, i) => Math.hypot(i % n + .5 - n / 2, Math.floor(i / n) + .5 - n / 2) < solved.apertureRadius ? 1 : 0);
  const started = performance.now(), larger = n * 2, offset = n / 2;
  const field = sourceFromPhase(phase, aperture);
  const padded = new Float64Array(2 * larger * larger);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const a = 2 * (y * n + x), b = 2 * ((y + offset) * larger + x + offset);
    padded[b] = field[a]; padded[b + 1] = field[a + 1];
  }
  const sourceEnergy = norm(field), reports = [];
  for (const distance of [.012, .03]) {
    const original = propagate(field, n, transfer(n, pitch, wavelength, distance));
    const expanded = propagate(padded, larger, transfer(larger, pitch, wavelength, distance));
    const crop = new Float64Array(field.length), originalIntensity = intensity(original), expandedIntensity = intensity(expanded);
    let outsideOriginalWindow = 0, expandedGuard = 0, originalGuard = 0;
    for (let y = 0; y < larger; y++) for (let x = 0; x < larger; x++) {
      const i = y * larger + x, inOriginal = x >= offset && x < offset + n && y >= offset && y < offset + n;
      if (inOriginal) {
        const to = 2 * ((y - offset) * n + x - offset);
        crop[to] = expanded[2 * i]; crop[to + 1] = expanded[2 * i + 1];
      } else outsideOriginalWindow += expandedIntensity[i];
      if (x < 32 || y < 32 || x >= larger - 32 || y >= larger - 32) expandedGuard += expandedIntensity[i];
    }
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
      if (x < 16 || y < 16 || x >= n - 16 || y >= n - 16) originalGuard += originalIntensity[y * n + x];
    reports.push({distance,
      centralComplexRelativeL2: relativeError(original, crop),
      centralIntensityRelativeL2: relativeError(originalIntensity, intensity(crop)),
      originalOuter16EnergyFraction: originalGuard / sourceEnergy,
      expandedOuter32EnergyFraction: expandedGuard / sourceEnergy,
      expandedEnergyOutsideOriginalWindowFraction: outsideOriginalWindow / sourceEnergy,
      expandedEnergyRelativeError: Math.abs(norm(expanded) / sourceEnergy - 1)});
  }
  const report = {source: 'assets/initial-phase.bin', physicalPitch: pitch, originalN: n, paddedN: larger, sourceEnergy, milliseconds: performance.now() - started, planes: reports};
  console.log('PADDING_REPORT ' + JSON.stringify(report));
  assert.ok(report.milliseconds < 5000, 'The padding check exceeded its five-second CPU budget.');
  for (const p of reports) {
    assert.ok(p.centralIntensityRelativeL2 < .05, `Padding changes central intensity by ${p.centralIntensityRelativeL2} at ${p.distance} m`);
    assert.ok(p.expandedOuter32EnergyFraction < .01, 'Expanded window still carries material edge energy.');
  }
});
