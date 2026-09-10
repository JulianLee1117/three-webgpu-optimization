import { normalizeConfig, regularParticles, mergeStates } from './cpu.mjs';

/** A prospective grid-scale casting fixture, not a continuous collision model.
 * Boxes constrain integer grid nodes inside inclusive bounds. The grille bars
 * occupy nodes x8..9 and16..17, y12, z9..14. Kernel-mediated contact is sticky
 * and has grid-scale thickness. The removable mold floor occupies node y7;
 * explicit banks remain at y<=8. No bridge collider remains after release.
 */
export function castingFixture({ bars = 2 } = {}) {
  if (bars !== 2) throw new RangeError('Only the prospectively frozen two-bar fixture is defined');
  const protocol = { version: 1, bars, N: 24, dt: 1 / 240, castingSteps: 960, releasedSteps: 960, recordEvery: 60,
    totalExecutedSteps: 2880, seed: 'regular-lattice-no-randomness',
    body: { min: [6, 13, 10], max: [18, 16, 13], spacing: .5, mode: 0, mass: .125, volume: .125 },
    load: { min: [11, 19, 10.5], max: [13, 21, 12.5], spacing: .5, mode: 1, mass: .125, volume: .125 },
    gate: { belowGrilleY: 11.5, fractionBelowGrilleGreaterThan: .75,
      centralRegion: { min: [9, 7.5, 9.5], max: [15, 11.5, 13.5] }, minimumCentralBodyParticles: 32,
      solidFinalLoadYAtLeast: 8, solidAboveLiquidByAtLeast: 2, maxRelativeMassError: 1e-10 },
    scope: 'One two-bar candidate, no settings search. All body particles start liquid; load starts held by an explicit collider. At4s both branches remove grille, mold and load holder; one changes body to solid, the other stays liquid. Named banks and end clamps remain. No repositioning or later particle spawn.' };
  const banks = [{ name: 'left-bank', min: [3, 2, 8], max: [8, 8, 15] }, { name: 'right-bank', min: [16, 2, 8], max: [21, 8, 15] }];
  const anchors = [{ name: 'left-solid-clamp', min: [6, 8.1, 10], max: [8.1, 11.1, 13] }, { name: 'right-solid-clamp', min: [15.9, 8.1, 10], max: [18, 11.1, 13] }];
  const removable = [
    { name: 'grille-left-bar', min: [8, 11.5, 9], max: [9, 12.5, 14] },
    { name: 'grille-right-bar', min: [16, 11.5, 9], max: [17, 12.5, 14] },
    { name: 'mold-floor', min: [5, 6.5, 8.5], max: [19, 7.5, 14.5] },
    { name: 'mold-front-wall', min: [5, 6.5, 8.5], max: [19, 11.5, 9.5] },
    { name: 'mold-back-wall', min: [5, 6.5, 13.5], max: [19, 11.5, 14.5] },
    { name: 'mold-left-end', min: [5, 6.5, 8.5], max: [6, 11.5, 14.5] },
    { name: 'mold-right-end', min: [18, 6.5, 8.5], max: [19, 11.5, 14.5] },
    { name: 'load-holder', min: [10, 18, 9.5], max: [14, 22, 13.5] }
  ];
  const material = { N: protocol.N, dt: protocol.dt, gravity: [0, -9.81, 0], bulk: 120, gamma: 5, mu: 80, lambda: 120, anchors };
  const body = regularParticles(protocol.body), load = regularParticles(protocol.load);
  return { castingConfig: normalizeConfig({ ...material, colliders: [...banks, ...removable] }),
    releasedConfig: normalizeConfig({ ...material, colliders: banks }),
    state: mergeStates([body, load]), bodyCount: body.ids.length, loadCount: load.ids.length, protocol };
}
