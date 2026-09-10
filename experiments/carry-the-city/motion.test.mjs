import test from 'node:test';
import assert from 'node:assert/strict';
import { captureClip, captureMarkerLoop, interpolateClipPoint, samplePoint, sampleFrame } from './motion.mjs';

const near = (actual, expected, tolerance = 1e-8) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected} by more than ${tolerance}`);
};
const vectorNear = (a, b, tolerance) => a.forEach((x, c) => near(x, b[c], tolerance));
const dot = (a, b) => a.reduce((sum, x, c) => sum + x * b[c], 0);

test('capture stores the authored source positions and declares its provenance', () => {
  const clip = captureClip({ kind: 'swim' });
  assert.equal(clip.data.length, 64 * 17 * 3);
  assert.equal(clip.sourceInfo.inferred, false);
  assert.equal(clip.sourceInfo.type, 'authored-marker-samples');
  for (let frame = 0; frame < clip.frames; frame += 7) {
    for (let marker = 0; marker < clip.markers; marker++) {
      const u = 2 * marker / (clip.markers - 1) - 1;
      const gain = 0.3 + 0.7 * (u + 1) / 2;
      const angle = 2 * Math.PI * frame / clip.frames + 2.5 * u;
      vectorNear(interpolateClipPoint(clip, frame * clip.duration / clip.frames, u),
        [u, 0.04 * Math.sin(angle) * gain, 0.16 * Math.sin(angle) * gain], 3e-8);
    }
  }
});

test('a third authored source produces analytic rigid translation without source-specific branches', () => {
  const clip = captureMarkerLoop({
    id: 'independent-test-donor', duration: 3, frames: 12, markers: 9,
    axis: 'x', extent: 4, scale: [4, 2, 6.25],
    sample: (u) => [u, 0.25, -0.125],
  });
  // This is neither source used by the game. Arbitrary height and width points
  // must all inherit the same displacement, rather than a built-in animation.
  for (const p of [[-4, 0, -1.8], [0.123, 2.4, 1.3], [4, -0.8, 0], [7, 1, 2]]) {
    const expected = [p[0], p[1] + 0.5, p[2] - 0.78125];
    vectorNear(samplePoint(clip, 0.321, p), expected, 1e-9);
    const frame = sampleFrame(clip, 1.233, p);
    vectorNear(frame.position, expected, 1e-9);
    vectorNear(frame.tangentX, [1, 0, 0]);
    vectorNear(frame.tangentZ, [0, 0, 1]);
    vectorNear(frame.normal, [0, 1, 0]);
    vectorNear(frame.velocity, [0, 0, 0]);
  }
});

test('rendered points change when captured source data changes', () => {
  const clip = captureClip({ kind: 'swim' });
  const time = 0.371;
  const p = [0.4, 0, -0.2];
  const before = samplePoint(clip, time, p);
  for (let offset = 2; offset < clip.data.length; offset += 3) clip.data[offset] += 0.125;
  const after = samplePoint(clip, time, p);
  vectorNear(after, [before[0], before[1], before[2] + 0.78125], 2e-5);
});

test('both clips loop at negative and positive times without a position discontinuity', () => {
  for (const kind of ['swim', 'fly']) {
    const clip = captureClip({ kind });
    for (const time of [-6.133, -0.01, 0, 0.133, 2.033]) {
      for (const p of [[-3.7, 0, -1.4], [0, 1.4, 0.75], [3.9, 0, 1.8]]) {
        vectorNear(samplePoint(clip, time, p), samplePoint(clip, time + 3 * clip.duration, p), 2e-8);
      }
    }
    const p = [2.2, 0.4, 1.3];
    vectorNear(samplePoint(clip, -1e-7, p), samplePoint(clip, 1e-7, p), 1e-5);
  }
});

test('zero strength is exactly the rest pose, including the support frame and velocity', () => {
  for (const kind of ['swim', 'fly']) {
    const clip = captureClip({ kind });
    const point = [2.213, 0.71, -1.229];
    assert.deepEqual(samplePoint(clip, 2.931, point, { strength: 0 }), point);
    const f = sampleFrame(clip, 2.931, point, { strength: 0 });
    vectorNear(f.position, point);
    vectorNear(f.tangentX, [1, 0, 0]);
    vectorNear(f.tangentZ, [0, 0, 1]);
    vectorNear(f.normal, [0, 1, 0]);
    vectorNear(f.velocity, [0, 0, 0]);
  }
});

test('every sampled road pose has an upward orthonormal frame across the full supported footprint', () => {
  // In particular, the final marker interval is checked: linear extrapolation
  // there produced a curvature spike and folded the inner edge of the street.
  for (const kind of ['swim', 'fly']) {
    const clip = captureClip({ kind });
    for (let frame = 0; frame < 64; frame++) {
      for (let xi = 0; xi <= 32; xi++) {
        for (let zi = 0; zi <= 12; zi++) {
          const p = [-4 + xi * 0.25, 0, -1.8 + zi * 0.3];
          const f = sampleFrame(clip, frame * clip.duration / 64, p);
          const location = `${kind} frame=${frame} point=${p}`;
          assert.ok(f.normal[1] > 0.6, `Surface folds: ${location}`);
          near(Math.hypot(...f.normal), 1, 1e-9);
          near(Math.hypot(...f.tangentX), 1, 1e-9);
          near(Math.hypot(...f.tangentZ), 1, 1e-9);
          near(dot(f.normal, f.tangentX), 0, 1e-9);
          near(dot(f.normal, f.tangentZ), 0, 1e-9);
          near(dot(f.tangentX, f.tangentZ), 0, 1e-9);
          assert.ok([...f.position, ...f.velocity].every(Number.isFinite), location);
        }
      }
    }
  }
});

test('reported material velocity predicts a separately sampled nearby pose', () => {
  for (const kind of ['swim', 'fly']) {
    const clip = captureClip({ kind });
    const p = [1.371, 0.25, 0.93];
    const time = 0.7313;
    const delta = 1e-4;
    const f = sampleFrame(clip, time, p);
    const next = samplePoint(clip, time + delta, p);
    vectorNear(next, f.position.map((x, c) => x + delta * f.velocity[c]), 2e-7);
  }
});

test('moth core stays level and its outer sections rotate their height with the wing', () => {
  const clip = captureClip({ kind: 'fly' });
  const time = clip.duration / 4;
  vectorNear(samplePoint(clip, time, [0, 0, 0]), [0, 0, 0], 1e-8);
  const root = samplePoint(clip, time, [0, 0, 1.4]);
  const top = samplePoint(clip, time, [0, 0.5, 1.4]);
  const f = sampleFrame(clip, time, [0, 0, 1.4]);
  vectorNear(top, root.map((x, c) => x + 0.5 * f.normal[c]), 1e-5);
  assert.ok(root[1] > 0.5);
  assert.ok(top[2] < root[2], 'Raised wing rotates height toward its center');
});

test('typed outputs and point objects are supported while invalid workloads fail explicitly', () => {
  const clip = captureClip();
  const output = new Float32Array(3);
  assert.equal(samplePoint(clip, 0, { x: 1, y: 0, z: 0 }, { out: output }), output);
  assert.ok(output.every(Number.isFinite));
  assert.throws(() => captureClip({ kind: 'unimplemented' }), /Unknown authored/);
  assert.throws(() => captureClip({ frames: 1000000 }), /bounded/);
  assert.throws(() => samplePoint(clip, NaN, [0, 0, 0]), /Time/);
  assert.throws(() => samplePoint(clip, 0, [NaN, 0, 0]), /finite/);
  assert.throws(() => samplePoint(clip, 0, [0, 0, 0], { strength: 2 }), /Strength/);
  assert.throws(() => sampleFrame(clip, 0, [0, 0, 0], { epsilon: 0 }), /epsilon/);
  assert.throws(() => interpolateClipPoint(clip, 0, Infinity), /coordinate/);
  clip.data.fill(NaN);
  assert.throws(() => samplePoint(clip, 0, [0, 0, 0]), /non-finite/);
});
