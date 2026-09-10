import { castingFixture } from './casting-fixture.mjs';
import { normalizeConfig, regularParticles, mergeStates } from './cpu.mjs';

/** Prospectively declared engineering revision of the failed v1 casting fixture.
 * Floor now constrains integer y8; narrow grille bars constrain x10 and14,y12.
 * Added source volume and stiffness are explicit changes, not a repeated v1 test.
 * Stiff elastic constants are arbitrary simulation units, not calibrated steel.
 */
export function castingFixtureV2() {
  const base = castingFixture(), protocol = structuredClone(base.protocol);
  protocol.version = 2; protocol.body.max[1] = 17;
  protocol.predecessorSHA256 = '35405523337d7bc0a84b6dd1552f0c8475fef5d20d458cdcae7be3a992de1b0e';
  protocol.changes = ['mold-floor integer node8 instead of7', 'grille integer x10/x14 instead of x8..9/x16..17', 'body volume144 instead of108', 'mu400/lambda600 instead of80/120'];
  protocol.material = { mu: 400, lambda: 600, label: 'stiff elastic, uncalibrated', explicitWaveSpeedEstimate: Math.sqrt(600 + 2 * 400), gridCFLAtRestDensity1: Math.sqrt(1400) / 240 };
  protocol.computeBudgetMs = 12000;
  protocol.gate.minimumParticlesInEachSupport = 24;
  protocol.gate.centralBins = { xEdges: [8, 10, 12, 14, 16], minY: 8.1, maxY: 11.1, minZ: 10, maxZ: 13, minimumParticlesPerBin: 24 };
  protocol.scope = 'One v2 engineering candidate. A raised mold, relocated bars, additional liquid volume and stiffer elastic material correct the diagnosed v1 support mismatch. No parameter sweep or force controller. Support and central occupancy gates are necessary checks, not a continuum connectivity proof.';
  const colliders = base.castingConfig.colliders.map(box => structuredClone(box));
  const floor = colliders.find(box => box.name === 'mold-floor'); floor.min[1] = 7.5; floor.max[1] = 8.5;
  const left = colliders.find(box => box.name === 'grille-left-bar'); left.min[0] = 10; left.max[0] = 10.5;
  const right = colliders.find(box => box.name === 'grille-right-bar'); right.min[0] = 13.5; right.max[0] = 14;
  const body = regularParticles(protocol.body), load = regularParticles(protocol.load);
  return { castingConfig: normalizeConfig({ ...base.castingConfig, mu: 400, lambda: 600, colliders }),
    releasedConfig: normalizeConfig({ ...base.releasedConfig, mu: 400, lambda: 600 }),
    state: mergeStates([body, load]), bodyCount: body.ids.length, loadCount: load.ids.length, protocol };
}
