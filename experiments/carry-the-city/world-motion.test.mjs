import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { captureClip } from './motion.mjs';
import { createMotionRig } from './rig.mjs';
import { createWorldMotionHistory } from './world-motion.mjs';

const near = (a, b, tolerance = 1e-8) => a.forEach((v, c) => {
  assert.ok(Math.abs(v - b[c]) <= tolerance, `${a} differs from ${b} at component ${c}`);
});
const anchor = [-1.5, 0, 1.6];
function fixture() {
  const town = { root: new THREE.Group(), surfaces: [], attachments: [] };
  const clips = { swim: captureClip({ kind: 'swim' }), fly: captureClip({ kind: 'fly' }) };
  return { town, clips, rig: createMotionRig(town, clips), history: createWorldMotionHistory() };
}

test('translation is inherited once and an explicit kick does not accumulate', () => {
  const history = createWorldMotionHistory();
  history.record('cargo', [1, 2, 3], 1);
  history.record('cargo', [1.1, 2.2, 2.9], 1.05);
  const first = history.release('cargo', { kick: [0, 1.8, 0] });
  near(first.inheritedVelocity, [2, 4, -2]);
  near(first.velocity, [2, 5.8, -2]);
  assert.deepEqual(history.release('cargo', { kick: [0, 1.8, 0] }), first);
  near(history.release('cargo').velocity, [2, 4, -2]);
});

test('steering around a stationary root gives an off-center anchor angular motion', () => {
  const { town, rig, history } = fixture();
  // Freeze the donor animation so all observed displacement comes from yaw.
  const local = rig.field(anchor, 0.4, 0);
  const angle = 0.12, dt = 0.02;
  history.record('cargo', rig.worldPoint(anchor, 0.4, 0), 1);
  town.root.rotation.y = angle;
  history.record('cargo', rig.worldPoint(anchor, 0.4, 0), 1 + dt);
  const rotated = [Math.cos(angle) * local[0] + Math.sin(angle) * local[2], local[1], -Math.sin(angle) * local[0] + Math.cos(angle) * local[2]];
  near(history.release('cargo').velocity, rotated.map((v, c) => (v - local[c]) / dt));
  assert.ok(Math.hypot(...history.release('cargo').velocity) > 5);
});

test('a donor blend change contributes motion even when its animation time is frozen', () => {
  const { rig, history } = fixture();
  const t = 0.4, dt = 0.02;
  const previous = rig.worldPoint(anchor, t, 0).toArray();
  history.record('cargo', previous, 1);
  const current = rig.worldPoint(anchor, t, 0.12).toArray();
  history.record('cargo', current, 1 + dt);
  near(history.release('cargo').velocity, current.map((v, c) => (v - previous[c]) / dt));
  assert.ok(Math.hypot(...history.release('cargo').velocity) > 0.5);
});

test('actual rig integration combines root translation, turning, animation and changing blend', () => {
  const { town, rig, history } = fixture();
  const t0 = 0.731, t1 = t0 + 1 / 60;
  town.root.position.set(2, 0.4, 5);
  town.root.rotation.y = 0.9;
  const previous = rig.worldPoint(anchor, t0, 0.2).toArray();
  history.record('cargo', previous, t0);
  town.root.position.set(2.04, 0.46, 4.97);
  town.root.rotation.y = 0.94;
  const current = rig.worldPoint(anchor, t1, 0.25).toArray();
  history.record('cargo', current, t1);
  const release = history.release('cargo', { kick: [0, 1.8, 0] });
  const expected = current.map((v, c) => (v - previous[c]) / (t1 - t0));
  near(release.inheritedVelocity, expected);
  near(release.velocity, [expected[0], expected[1] + 1.8, expected[2]]);
  near(release.position, current);

  // The old approach applies the current matrix/blend to the old source time.
  // Even after adding the exact root translation it omits turn/blend movement.
  const reconstructed = rig.worldPoint(anchor, t0, 0.25).toArray();
  const wrong = current.map((v, c) => (v - reconstructed[c] + [0.04, 0.06, -0.03][c]) / (t1 - t0));
  assert.ok(Math.hypot(...wrong.map((v, c) => v - expected[c])) > 0.5);
});

test('stored positions survive source-data edits and later mutation of Three vectors', () => {
  const { clips, rig, history } = fixture();
  const initial = rig.worldPoint(anchor, 0.25, 0);
  const expectedInitial = initial.toArray();
  history.record('cargo', initial, 1);
  initial.set(900, 800, 700);
  for (let i = 2; i < clips.swim.data.length; i += 3) clips.swim.data[i] += 0.125;
  const changed = rig.worldPoint(anchor, 0.25, 0).toArray();
  history.record('cargo', changed, 1.02);
  near(history.release('cargo').velocity, changed.map((v, c) => (v - expectedInitial[c]) / 0.02), 1e-7);
  assert.ok(history.release('cargo').velocity[2] > 30);
  const release = history.release('cargo');
  release.position.fill(1000);
  release.velocity.fill(1000);
  near(history.release('cargo').position, changed);
});

test('first frame, restart, same-time edits, stale history and teleports cannot create spurious velocities', () => {
  const history = createWorldMotionHistory();
  const check = (reason) => {
    const p = history.release('cargo', { kick: [0, 1, 0] });
    assert.equal(p.hasVelocity, false);
    assert.equal(p.reason, reason);
    assert.equal(p.deltaTime, 0);
    near(p.inheritedVelocity, [0, 0, 0]);
    near(p.velocity, [0, 1, 0]);
  };
  history.record('cargo', [0, 0, 0], 0);
  check('first-sample');
  history.record('cargo', [1, 0, 0], 0);
  check('non-increasing-time');
  history.record('cargo', [100, 0, 0], 1);
  check('stale-history');
  history.record('cargo', [0, 0, 0], 0);
  check('non-increasing-time');
  history.record('cargo', [1000, 0, 0], 0.01, { discontinuous: true });
  check('discontinuous');
  history.record('cargo', [1000.01, 0, 0], 0.02);
  assert.equal(history.release('cargo').hasVelocity, true);
  near(history.release('cargo').velocity, [1, 0, 0]);
  history.clear();
  assert.equal(history.size, 0);
  assert.throws(() => history.release('cargo'), /not been recorded/);
});

test('bounded independent anchors retain only their own movement', () => {
  const history = createWorldMotionHistory({ maxPoints: 2 });
  history.record('a', [0, 0, 0], 0);
  history.record('b', [0, 0, 0], 0);
  for (let i = 1; i <= 500; i++) {
    history.record('a', [i, 0, 0], i * 0.01);
    history.record('b', [0, -i, 0], i * 0.01);
  }
  assert.equal(history.size, 2);
  near(history.release('a').velocity, [100, 0, 0]);
  near(history.release('b').velocity, [0, -100, 0]);
  assert.throws(() => history.record('c', [0, 0, 0], 5), /limit/);
  assert.equal(history.forget('a'), true);
  history.record('c', [0, 0, 0], 5);
  assert.equal(history.size, 2);
});

test('invalid data fails before changing a valid recorded anchor', () => {
  const history = createWorldMotionHistory();
  history.record('cargo', new Float32Array([1, 2, 3]), 0);
  const before = history.release('cargo');
  assert.throws(() => history.record('cargo', [Infinity, 0, 0], 1), /finite/);
  assert.throws(() => history.record('cargo', [0, 0, 0], NaN), /Time/);
  assert.throws(() => history.record('', [0, 0, 0], 1), /id/);
  assert.throws(() => history.release('cargo', { kick: [0, NaN, 0] }), /kick/);
  assert.throws(() => createWorldMotionHistory({ maxPoints: 10000 }), /maxPoints/);
  assert.throws(() => createWorldMotionHistory({ maxDeltaTime: 0 }), /maxDeltaTime/);
  assert.deepEqual(history.release('cargo'), before);
});
