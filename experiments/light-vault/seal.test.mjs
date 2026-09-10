import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {serializeSeal, parseSeal, MAX_SEAL_BYTES} from './seal.mjs';
import {OPTICS, presetAmplitude} from './targets.mjs';

const bytes = await readFile(new URL('./assets/initial-phase.bin', import.meta.url));
const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
const phase = Float64Array.from({length: OPTICS.n ** 2}, (_, i) => view.getFloat32(i * 4, true));
const targets = ['heart', 'letterA'].map((kind, i) => ({amplitude: presetAmplitude(kind), distance: OPTICS.distances[i]}));
const original = serializeSeal(phase, targets);
const edited = mutate => {const seal = JSON.parse(original); mutate(seal); return JSON.stringify(seal);};

test('committed phase round-trips with byte references and retains two verified images', () => {
  const raw = JSON.parse(original), loaded = parseSeal(original);
  assert.equal(raw.format, 'light-vault-seal-v1');
  assert.ok(new TextEncoder().encode(original).byteLength < MAX_SEAL_BYTES);
  assert.deepEqual(loaded.phase, phase);
  assert.ok(loaded.quality.every((q, i) => q.clear && q.best === i));
  assert.ok(loaded.quality[0].scores[0] > .90 && loaded.quality[1].scores[1] > .94);
  for (let k = 0; k < 2; k++) {
    let peak = 0;
    for (let i = 0; i < phase.length; i++) {
      const expected = Math.round(targets[k].amplitude[i] ** 2 * 255);
      assert.equal(raw.references[k][i], expected);
      assert.equal(loaded.targets[k].amplitude[i], Math.sqrt(expected / 255));
      peak = Math.max(peak, raw.references[k][i]);
    }
    assert.equal(peak, 255);
    assert.equal(loaded.targets[k].distance, OPTICS.distances[k]);
  }
  assert.equal(serializeSeal(loaded.phase, loaded.targets), original);
});

test('quantization uses normalized intensity and does not alter input phase or amplitude', () => {
  const precise = phase.slice(); precise[0] += 1e-9;
  const sourceBefore = precise.slice(), scaled = targets.map(t => ({...t, amplitude: Float64Array.from(t.amplitude, value => value * 7)}));
  const encoded = JSON.parse(serializeSeal(precise, scaled));
  assert.equal(encoded.phaseRadians[0], Math.fround(precise[0]));
  assert.deepEqual(precise, sourceBefore);
  assert.deepEqual(encoded.references, JSON.parse(original).references);
  assert.equal(scaled[0].amplitude[10000], targets[0].amplitude[10000] * 7);
});

test('malformed JSON, unsupported formats and oversized UTF-8 inputs are rejected', () => {
  for (const text of ['{', 'null', '[]', '"seal"', '{}']) assert.throws(() => parseSeal(text), /JSON|format/);
  assert.throws(() => parseSeal(null), /JSON text/);
  assert.throws(() => parseSeal(' '.repeat(MAX_SEAL_BYTES + 1)), /3.5 MB/);
  const unicode = '"' + 'é'.repeat(MAX_SEAL_BYTES / 2 + 1) + '"';
  assert.ok(unicode.length < MAX_SEAL_BYTES);
  assert.throws(() => parseSeal(unicode), /3.5 MB/);
  assert.throws(() => parseSeal(edited(s => {s.format = 'light-vault-seal-v2';})), /format/);
});

test('every optical constant and depth bound must exactly match the fixed physical model', () => {
  for (const key of ['n', 'pitch', 'wavelength', 'apertureRadius'])
    assert.throws(() => parseSeal(edited(s => {s[key] *= 2;})), /fixed optics/);
  for (const key of ['distances', 'range']) {
    assert.throws(() => parseSeal(edited(s => {s[key][0] += .001;})), /fixed optics/);
    assert.throws(() => parseSeal(edited(s => {s[key].push(.04);})), /fixed optics/);
  }
});

test('phase dimensions, type, finiteness and radian range are checked before simulation', () => {
  for (const value of [null, '0', 4, -4])
    assert.throws(() => parseSeal(edited(s => {s.phaseRadians[3] = value;})), /phase/);
  assert.throws(() => parseSeal(edited(s => {s.phaseRadians.pop();})), /phase/);
  const invalid = phase.slice(); invalid[3] = NaN;
  assert.throws(() => serializeSeal(invalid, targets), /phase/);
  invalid[3] = Infinity;
  assert.throws(() => serializeSeal(invalid, targets), /phase/);
});

test('references require exactly two complete byte grids with enough ink', () => {
  assert.throws(() => parseSeal(edited(s => {s.references.pop();})), /two image/);
  assert.throws(() => parseSeal(edited(s => {s.references[0].pop();})), /65,536/);
  for (const value of [-1, 256, .5, null, '255'])
    assert.throws(() => parseSeal(edited(s => {s.references[0][0] = value;})), /integers/);
  assert.throws(() => parseSeal(edited(s => {s.references[0].fill(0);})), /bolder/);
  assert.throws(() => parseSeal(edited(s => {s.references[0].fill(1);})), /bolder/);
  assert.throws(() => serializeSeal(phase, [targets[0]]), /two image/);
  const badTarget = targets[0].amplitude.slice(); badTarget[0] = -1;
  assert.throws(() => serializeSeal(phase, [{amplitude: badTarget}, targets[1]]), /nonnegative/);
});

test('real propagation rejects a valid-sized replacement phase, duplicate and unrelated references', () => {
  assert.throws(() => parseSeal(edited(s => {s.phaseRadians.fill(0);})), /does not produce/);
  assert.throws(() => parseSeal(edited(s => {s.references[1] = s.references[0].slice();})), /distinct/);
  assert.throws(() => parseSeal(edited(s => {
    s.references = [0, 1].map(side => Array.from({length: phase.length}, (_, i) => {
      const x = i % OPTICS.n, y = Math.floor(i / OPTICS.n);
      return y < 32 && x >= side * 128 && x < side * 128 + 32 ? 255 : 0;
    }));
  })), /does not produce/);
  assert.throws(() => parseSeal(edited(s => {s.references.reverse();})), /does not produce/);
});
