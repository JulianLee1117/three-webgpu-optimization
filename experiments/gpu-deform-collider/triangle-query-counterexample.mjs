/**
 * CPU-only reproducer for three-mesh-bvh 0.9.15's WGSL closestPointToTriangle.
 * Pinned source: 8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab
 * https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/src/webgpu/tsl/fns.js
 *
 * Browser imports are side-effect free. Run `node <this-file> --self-test` for
 * an independent Three.Triangle oracle and a bounded deterministic CPU sweep.
 * This reproducer uses doubles; its minimal counterexample uses exactly
 * representable integers/halves throughout the upstream calculation.
 */

const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const clamp = (x) => Math.max(0, Math.min(1, x));
const distanceSq = (a, b) => dot(sub(a, b), sub(a, b));
const interpolate = (a, b, c, barycoord) => a.map((v, i) => v * barycoord[0] + b[i] * barycoord[1] + c[i] * barycoord[2]);

function projectedBarycoord(p, a, b, c) {
  const v10 = sub(b, a), v02 = sub(a, c);
  const nor = cross(v10, v02), q = cross(nor, sub(p, a));
  const d = 1 / dot(nor, nor);
  const u = d * dot(q, v02), v = d * dot(q, v10);
  return [1 - u - v, u, v];
}

/** Literal transcription of the pinned WGSL primitive, including its bug. */
export function upstreamClosestPointTriangle(p, a, b, c) {
  const v10 = sub(b, a), v21 = sub(c, b), v02 = sub(a, c);
  const p0 = sub(p, a), p1 = sub(p, b), p2 = sub(p, c);
  const nor = cross(v10, v02), q = cross(nor, p0);
  const d = 1 / dot(nor, nor);
  let u = d * dot(q, v02), v = d * dot(q, v10), w = 1 - u - v;
  if (u < 0) {
    w = clamp(dot(p2, v02) / dot(v02, v02));
    u = 0;
    v = 1 - w;
  } else if (v < 0) {
    u = clamp(dot(p0, v10) / dot(v10, v10));
    v = 0;
    w = 1 - u;
  } else if (w < 0) {
    v = clamp(dot(p1, v21) / dot(v21, v21));
    w = 0;
    u = 1 - v;
  }
  const barycoord = [w, u, v];
  const point = interpolate(a, b, c, barycoord);
  return {point, barycoord, distanceSq: distanceSq(p, point)};
}

/** Standard plane-interior / nearest-of-three-segments reference alternative. */
export function edgeMinimumClosestPointTriangle(p, a, b, c) {
  const vertices = [a, b, c];
  const projected = projectedBarycoord(p, a, b, c);
  if (projected.every((v) => Number.isFinite(v) && v >= 0)) {
    const point = interpolate(a, b, c, projected);
    return {point, barycoord: projected, distanceSq: distanceSq(p, point)};
  }
  let best;
  for (const [i, j] of [[0, 1], [1, 2], [2, 0]]) {
    const edge = sub(vertices[j], vertices[i]);
    const lengthSq = dot(edge, edge);
    const t = lengthSq > 0 ? clamp(dot(sub(p, vertices[i]), edge) / lengthSq) : 0;
    const barycoord = [0, 0, 0];
    barycoord[i] = 1 - t;
    barycoord[j] = t;
    const point = interpolate(a, b, c, barycoord);
    const candidate = {point, barycoord, distanceSq: distanceSq(p, point)};
    if (!best || candidate.distanceSq < best.distanceSq) best = candidate;
  }
  return best;
}

export const knownCounterexample = {
  name: 'obtuse-corner-chooses-wrong-edge',
  a: [0, 0, 0], b: [2, 0, 0], c: [-1, 1, 0], p: [1, -2, 0],
  expectedPoint: [1, 0, 0], expectedBarycoord: [0.5, 0.5, 0], expectedDistanceSq: 4,
  upstreamPoint: [0, 0, 0], upstreamDistanceSq: 5,
  projectedBarycoord: [3.5, -0.5, -2],
};

const permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
const vertices = [knownCounterexample.a, knownCounterexample.b, knownCounterexample.c];
/** Same geometry, all six vertex orders, inside and outside queries. */
export const fixtures = ['outside', 'inside', 'outside-off-plane', 'inside-off-plane'].flatMap((kind) => {
  const outside = kind.startsWith('outside'), offPlane = kind.endsWith('off-plane');
  const p = outside ? [1, -2, offPlane ? 0.5 : 0] : [0.25, 0.25, offPlane ? 0.5 : 0];
  const expectedPoint = outside ? [1, 0, 0] : [0.25, 0.25, 0];
  const expectedOriginalBarycoord = outside ? [0.5, 0.5, 0] : [0.5, 0.25, 0.25];
  return permutations.map((order) => ({
    name: `${kind}-${order.join('')}`,
    a: [...vertices[order[0]]], b: [...vertices[order[1]]], c: [...vertices[order[2]]],
    p: [...p], expectedPoint: [...expectedPoint],
    expectedBarycoord: order.map((index) => expectedOriginalBarycoord[index]),
    expectedDistanceSq: (outside ? 4 : 0) + (offPlane ? 0.25 : 0),
  }));
});

async function selfTest() {
  const {default: assert} = await import('node:assert/strict');
  const {Triangle, Vector3} = await import('three');
  const {readFile} = await import('node:fs/promises');
  const {createHash} = await import('node:crypto');
  const upstreamSource = await readFile(new URL('../../node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js', import.meta.url), 'utf8');
  assert.match(upstreamSource, /if \( u < 0\.0 \)/);
  assert.match(upstreamSource, /else if \( v < 0\.0 \)/);
  assert.match(upstreamSource, /else if \( w < 0\.0 \)/);
  const oracle = (p, a, b, c) => {
    const triangle = new Triangle(...[a, b, c].map((v) => new Vector3(...v)));
    const point = triangle.closestPointToPoint(new Vector3(...p), new Vector3()).toArray();
    return {point, distanceSq: distanceSq(p, point)};
  };
  const close = (actual, expected, tolerance = 1e-10) => {
    assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} differs from ${expected}`);
  };
  const failedFixtureNames = [];
  for (const fixture of fixtures) {
    const args = [fixture.p, fixture.a, fixture.b, fixture.c];
    const old = upstreamClosestPointTriangle(...args), fixed = edgeMinimumClosestPointTriangle(...args), reference = oracle(...args);
    close(reference.distanceSq, fixture.expectedDistanceSq);
    close(fixed.distanceSq, reference.distanceSq);
    fixed.point.forEach((v, i) => close(v, fixture.expectedPoint[i]));
    fixed.barycoord.forEach((v, i) => close(v, fixture.expectedBarycoord[i]));
    if (Math.abs(old.distanceSq - reference.distanceSq) > 1e-10) failedFixtureNames.push(fixture.name);
  }
  const known = knownCounterexample;
  const bad = upstreamClosestPointTriangle(known.p, known.a, known.b, known.c);
  assert.deepEqual(bad.point, known.upstreamPoint);
  assert.equal(bad.distanceSq, known.upstreamDistanceSq);
  assert.ok(failedFixtureNames.length > 0 && failedFixtureNames.length < fixtures.length);

  let state = 17431;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  const vector = () => [random() * 4 - 2, random() * 4 - 2, random() * 4 - 2];
  let cases = 0, upstreamMismatches = 0, maxFixedDistanceError = 0;
  while (cases < 10000) {
    const a = vector(), b = vector(), c = vector(), p = vector();
    const normal = cross(sub(b, a), sub(c, a));
    if (dot(normal, normal) < 1e-8) continue;
    const reference = oracle(p, a, b, c);
    const fixed = edgeMinimumClosestPointTriangle(p, a, b, c);
    const old = upstreamClosestPointTriangle(p, a, b, c);
    close(fixed.distanceSq, reference.distanceSq);
    fixed.point.forEach((v, i) => close(v, reference.point[i]));
    maxFixedDistanceError = Math.max(maxFixedDistanceError, Math.abs(fixed.distanceSq - reference.distanceSq));
    if (Math.abs(old.distanceSq - reference.distanceSq) > 1e-9) upstreamMismatches++;
    cases++;
  }
  assert.ok(upstreamMismatches > 0);
  console.log(JSON.stringify({
    passed: true, knownCounterexample, actualUpstreamResult: bad,
    sourceSha256: createHash('sha256').update(upstreamSource).digest('hex'),
    fixtureCount: fixtures.length, failedUpstreamFixtureNames: failedFixtureNames,
    deterministicSweep: {cases, upstreamMismatches, correctedMismatches: 0, maxFixedDistanceError},
    caveat: 'CPU geometric correctness reproducer, not a GPU float-accuracy or performance claim.',
  }, null, 2));
}

if (typeof process !== 'undefined' && process.versions?.node && process.argv.includes('--self-test')) {
  const {pathToFileURL} = await import('node:url');
  if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await selfTest();
}
