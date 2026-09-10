import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createVolumePreview} from './volume-worker.mjs';
import {sourceFromPhase, propagate, transfer, intensity} from './wave.mjs';
import {OPTICS, aperture} from './targets.mjs';

test('committed plate produces a bounded, physically sampled volume preview', async t => {
  const raw = await readFile(new URL('./assets/initial-phase.bin', import.meta.url));
  assert.equal(raw.byteLength, OPTICS.n ** 2 * 4);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const phase = Float64Array.from({length: OPTICS.n ** 2}, (_, i) => view.getFloat32(i * 4, true));
  const before = phase.slice();
  // One complete preview shared by the checks; no fitting, browser, or GPU work.
  const result = createVolumePreview(phase);

  await t.test('output is finite, bounded and leaves the source phase unchanged', () => {
    assert.deepEqual(phase, before);
    assert.ok(result.positions instanceof Float32Array);
    assert.ok(result.colors instanceof Float32Array);
    assert.ok(result.ends instanceof Uint32Array);
    assert.equal(result.ends.length, 32);
    assert.equal(result.positions.length, result.colors.length);
    assert.equal(result.positions.length, result.ends[31] * 3);
    assert.ok(result.ends[31] > 0 && result.ends[31] <= 32_768);
    assert.ok(result.positions.every(Number.isFinite));
    assert.ok(result.colors.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
    assert.ok(Number.isFinite(result.milliseconds) && result.milliseconds < 4000);
  });

  await t.test('cumulative ends partition points into ordered depth slices for receiver clipping', () => {
    let begin = 0;
    for (let slice = 0; slice < 32; slice++) {
      const end = result.ends[slice], expectedX = -1.7 + slice / 31 * 4.9;
      assert.ok(end >= begin);
      assert.ok(end - begin <= 32 * 32);
      for (let point = begin; point < end; point++) {
        const x = result.positions[point * 3];
        assert.ok(Math.abs(x - expectedX) < 2e-7, `Point ${point} belongs to the wrong depth slice.`);
        if (point > 0) assert.ok(x >= result.positions[(point - 1) * 3]);
      }
      // A draw range ending here must contain no point from a later slice.
      if (slice < 31 && end < result.ends[31])
        assert.ok(result.positions[end * 3] > expectedX + 1e-4);
      begin = end;
    }
  });

  await t.test('first and last slices match independent full-field propagation and 4 by 4 means', () => {
    const field = sourceFromPhase(phase, aperture());
    let checkedPoints = 0;
    for (const slice of [0, 31]) {
      const distance = .004 + slice / 31 * .034;
      // Public propagation owns its own source FFT; this does not use the worker's spectrum or workspace.
      const pixels = intensity(propagate(field, OPTICS.n, transfer(OPTICS.n, OPTICS.pitch, OPTICS.wavelength, distance)));
      let point = slice === 0 ? 0 : result.ends[slice - 1];
      for (let row = 0; row < 32; row++) for (let col = 0; col < 32; col++) {
        let sum = 0;
        for (let dy = 0; dy < 4; dy++) for (let dx = 0; dx < 4; dx++)
          sum += pixels[(64 + row * 4 + dy) * OPTICS.n + 64 + col * 4 + dx];
        const mean = sum / 16;
        if (mean <= .20) continue;
        assert.ok(point < result.ends[slice], 'The independently propagated slice contains an unreported point.');
        const expectedPosition = [-1.7 + slice / 31 * 4.9, 1.65 + (64 - (row * 4 + 2)) / 128 * 2.3, ((col * 4 + 2) - 64) / 128 * 2.3];
        const gain = Math.min(1, mean * .14), expectedColor = [gain, gain * .62, gain * .19];
        for (let channel = 0; channel < 3; channel++) {
          assert.ok(Math.abs(result.positions[point * 3 + channel] - expectedPosition[channel]) < 2e-7);
          assert.ok(Math.abs(result.colors[point * 3 + channel] - expectedColor[channel]) < 4e-8);
        }
        point++; checkedPoints++;
      }
      assert.equal(point, result.ends[slice], 'The preview slice contains a point absent from independent propagation.');
    }
    assert.ok(checkedPoints > 0);
    t.diagnostic(`Preview: ${result.ends[31]} points in ${result.milliseconds.toFixed(1)} ms; ${checkedPoints} endpoint points independently verified.`);
  });

  await t.test('malformed phase input is rejected before volume computation', () => {
    assert.throws(() => createVolumePreview(new Float64Array(10)), /phase grid/);
    assert.throws(() => createVolumePreview(null), /phase grid/);
  });
});
