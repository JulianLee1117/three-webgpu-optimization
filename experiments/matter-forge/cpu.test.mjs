import test from 'node:test';
import assert from 'node:assert/strict';
import { makeState, regularParticles, mergeStates, normalizeConfig, stepCPU, transmute, particleTotals, determinant } from './cpu.mjs';
test('empty grid is a finite no-op', () => {
  const result = stepCPU(normalizeConfig(), makeState({ x: [] }), { retainGrid: true });
  assert.equal(result.diagnostics.gridMass, 0); assert.equal(result.state.x.length, 0);
  assert.ok(result.grid.velocity.every(x => x === 0));
});
test('quadratic APIC mass/momentum and affine velocity reproduction', () => {
  const s = regularParticles({ min: [7, 7, 7], max: [11, 11, 11], mode: 0 });
  const A = [.1, .03, -.02, .04, -.08, .01, -.03, .02, .05], b = [.2, -.1, .3];
  for (let p = 0; p < s.ids.length; p++) {
    s.C.set(A, 9 * p);
    for (let d = 0; d < 3; d++) s.v[3 * p + d] = b[d] + A.slice(3 * d, 3 * d + 3).reduce((sum, a, k) => sum + a * s.x[3 * p + k], 0);
  }
  const before = makeState(s), { state, diagnostics } = stepCPU(normalizeConfig({ gravity: [0, 0, 0], bulk: 0 }), s);
  assert.ok(Math.abs(diagnostics.massError) < 1e-11);
  assert.ok(diagnostics.transferMomentumError.every(x => Math.abs(x) < 1e-11));
  for (let i = 0; i < s.v.length; i++) assert.ok(Math.abs(state.v[i] - s.v[i]) < 1e-11);
  for (let i = 0; i < s.C.length; i++) assert.ok(Math.abs(state.C[i] - s.C[i]) < 1e-11);
  assert.deepEqual(s.x, before.x); assert.deepEqual(s.F, before.F);
});
test('stress-free solid stays at rest without gravity', () => {
  const initial = regularParticles({ min: [7, 7, 7], max: [9, 9, 9] });
  const { state, diagnostics } = stepCPU(normalizeConfig({ gravity: [0, 0, 0] }), initial);
  assert.deepEqual(state.x, initial.x); assert.deepEqual(state.F, initial.F); assert.equal(diagnostics.minJ, 1);
});
test('uniform gravity has the declared momentum impulse away from boundaries', () => {
  const initial = regularParticles({ min: [7, 7, 7], max: [9, 9, 9], mode: 0 });
  const config = normalizeConfig({ bulk: 0 }), result = stepCPU(config, initial);
  assert.ok(Math.abs(particleTotals(result.state).momentum[1] - particleTotals(initial).mass * config.dt * config.gravity[1]) < 1e-11);
  assert.ok(result.diagnostics.momentumLedgerError.every(x => Math.abs(x) < 1e-11));
});
test('transmutation preserves identity, coordinates and momentum and creates a new rest reference', () => {
  const input = regularParticles({ min: [7, 7, 7], max: [8, 8, 8] });
  input.v.fill(.4); input.F[0] = 1.2;
  const liquid = transmute(input, [0, 1], 0), solid = transmute(liquid, [0, 1], 1);
  for (const key of ['ids', 'x', 'v', 'C', 'mass', 'volume']) assert.deepEqual(solid[key], input[key]);
  assert.equal(liquid.F[0], 1.2); assert.equal(solid.F[0], 1); assert.equal(determinant(solid.F), 1);
  assert.equal(input.F[0], 1.2); assert.equal(solid.materialEdits.length, 2);
});
test('invalid deformation and out-of-domain particles fail explicitly', () => {
  const input = regularParticles({ min: [7, 7, 7], max: [8, 8, 8] }); input.F[0] = -1;
  assert.throws(() => stepCPU(normalizeConfig(), input), /determinant/);
  assert.throws(() => stepCPU(normalizeConfig(), makeState({ x: [.1, 8, 8] })), /stencil/);
});
test('named anchors only clamp nodes carrying solid material', () => {
  const solid = regularParticles({ min: [7, 7, 7], max: [8, 8, 8] }); solid.v.fill(.2);
  const config = normalizeConfig({ gravity: [0, 0, 0], bulk: 0, anchors: [{ name: 'test-clamp', min: [5, 5, 5], max: [10, 10, 10] }] });
  const a = stepCPU(config, solid), b = stepCPU(config, transmute(solid, null, 0));
  assert.ok(a.diagnostics.anchoredNodes > 0); assert.equal(a.state.v.reduce((x, y) => x + y, 0), 0);
  assert.equal(b.diagnostics.anchoredNodes, 0); assert.ok(b.state.v.every(x => Math.abs(x - .2) < 1e-12));
  assert.equal(mergeStates([solid, solid]).ids.length, solid.ids.length * 2);
});
