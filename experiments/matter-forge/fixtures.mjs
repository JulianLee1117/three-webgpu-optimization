import { normalizeConfig, regularParticles, mergeStates, transmute } from './cpu.mjs';

// Exact first CPU screen fixture. Every call returns independent arrays.
export function beamFixture({ mode = 'solid' } = {}) {
  if (mode !== 'solid' && mode !== 'liquid') throw new RangeError('mode must be solid or liquid');
  const protocol = { version: 1, steps: 480, recordEvery: 30, dt: 1 / 240, N: 24, seed: 'regular-lattice-no-randomness',
    beam: { min: [6, 8, 10], max: [18, 11, 13], spacing: .5, mass: .125, volume: .125, mode: 1 },
    load: { min: [11, 13, 10.5], max: [13, 15, 12.5], spacing: .5, mass: .125, volume: .125, mode: 1 },
    qualityGate: 'Both lanes finish finite; solid final load COM y>=8.0 and >=2.0 above liquid control, solid load speed<=2.5, max relative mass error<1e-10.',
    scope: 'A constrained elastic beam with real particle load; no mesh import, fracture, calibrated ice, or arbitrary crafted bridge claim. Shared-grid numerical contact, no hidden bridge collider.' };
  const config = normalizeConfig({ N: protocol.N, dt: protocol.dt, gravity: [0, -9.81, 0], bulk: 120, gamma: 5, mu: 80, lambda: 120,
    colliders: [{ name: 'left-bank', min: [3, 2, 8], max: [8, 8, 15] }, { name: 'right-bank', min: [16, 2, 8], max: [21, 8, 15] }],
    anchors: [{ name: 'left-solid-clamp', min: [6, 8.1, 10], max: [8.1, 11.1, 13] }, { name: 'right-solid-clamp', min: [15.9, 8.1, 10], max: [18, 11.1, 13] }] });
  const beam = regularParticles(protocol.beam), load = regularParticles(protocol.load), original = mergeStates([beam, load]);
  const beamCount = beam.ids.length, loadCount = load.ids.length;
  const state = mode === 'solid' ? original : transmute(original, Array.from({ length: beamCount }, (_, i) => i), 0);
  return { config, state, beamCount, loadCount, protocol };
}
