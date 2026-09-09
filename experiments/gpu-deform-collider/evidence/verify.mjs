import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {gunzipSync} from 'node:zlib';

const base = new URL('./', import.meta.url);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const read = name => {
  assert.match(name, /^[a-zA-Z0-9._-]+$/, 'Only files in this evidence directory are permitted');
  return readFile(new URL(name, base));
};
const pinned = [
  '3121e489aea3c620ed5f49d3ce73b76f577642499e39c3cc413eddd37c05a4f7',
  '75c1b1c87a57dad99a2ff59d8137e56b64f403bd528d2754747925d15a265ce0',
];
const manifestBytes = await read('manifest.json');
assert.equal(sha(manifestBytes), (await read('manifest.sha256')).toString().trim());
const manifest = JSON.parse(manifestBytes);
assert.equal(manifest.format, 'triangle-query-portable-evidence-v1');
for (const entry of manifest.files) {
  const bytes = await read(entry.file);
  assert.equal(bytes.length, entry.bytes, entry.file);
  assert.equal(sha(bytes), entry.sha256, entry.file);
}

const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const scale = (a, k) => a.map(v => v * k);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const d2 = (a, b) => dot(sub(a, b), sub(a, b));
// Independent double-precision reference: nearest of all edge segments and
// the plane projection when it lies inside the triangle. No AD or GPU code.
function closest(p, a, b, c) {
  const segment = (x, y) => {
    const v = sub(y, x), length = dot(v, v);
    return add(x, scale(v, length ? Math.max(0, Math.min(1, dot(sub(p, x), v) / length)) : 0));
  };
  const candidates = [segment(a, b), segment(b, c), segment(c, a)];
  const ab = sub(b, a), ac = sub(c, a), n = cross(ab, ac), n2 = dot(n, n);
  assert.ok(n2 > 0, 'This corpus only claims nondegenerate coverage');
  const q = sub(p, scale(n, dot(sub(p, a), n) / n2));
  const v = dot(cross(sub(q, a), ac), n) / n2;
  const w = dot(cross(ab, sub(q, a)), n) / n2;
  if (v >= 0 && w >= 0 && v + w <= 1) candidates.push(q);
  return candidates.reduce((best, x) => d2(p, x) < d2(p, best) ? x : best);
}

const reports = [], summaries = [];
assert.equal(manifest.reports.length, 2);
for (const [index, entry] of manifest.reports.entries()) {
  const raw = gunzipSync(await read(entry.archive), {maxOutputLength: 2_000_000});
  assert.equal(raw.length, entry.rawBytes);
  assert.equal(sha(raw), pinned[index]);
  assert.equal(entry.rawSha256, pinned[index]);
  assert.deepEqual(entry.sanitization.changes, []);
  const report = JSON.parse(raw), probe = report.triangleProbe, t = probe.tolerances;
  assert.equal(report.kind, 'upstream-triangle-probe');
  assert.deepEqual(report.errors, []); assert.deepEqual(probe.errors, []);
  assert.deepEqual(report.changedSources, []);
  assert.equal(probe.caseCount, 200); assert.equal(probe.cases.length, 200);
  assert.equal(probe.rawGpuOutput.length, 1600);
  assert.equal(probe.randomCount, 128); assert.equal(probe.analyticCaseCount, 72);
  assert.equal(probe.randomSeed, 1831565813); assert.equal(probe.cpuOracleMismatchCount, 0);
  const groups = {}; let mismatches = 0;
  for (const [i, c] of probe.cases.entries()) {
    assert.deepEqual(probe.rawGpuOutput.slice(8*i, 8*i+3), c.gpu.barycoord);
    assert.equal(probe.rawGpuOutput[8*i+3], 0);
    assert.deepEqual(probe.rawGpuOutput.slice(8*i+4, 8*i+7), c.gpu.closestPoint);
    assert.equal(probe.rawGpuOutput[8*i+7], c.gpu.distanceSq);
    const oracle = closest(c.p, c.a, c.b, c.c);
    assert.ok(Math.sqrt(d2(oracle, c.expected.closestPoint)) <= 1e-10, c.id);
    assert.ok(Math.abs(d2(c.p, oracle) - c.expected.distanceSq) <= 1e-10, c.id);
    assert.equal(c.cpuAgreement, true);
    assert.ok(Math.sqrt(d2(c.threeCpu.closestPoint, c.expected.closestPoint)) <= t.cpuPointAbsolute);
    assert.ok(Math.abs(c.threeCpu.distanceSq - c.expected.distanceSq) <= t.cpuDistanceSqAbsolute);
    const pointError = Math.sqrt(d2(c.gpu.closestPoint, c.expected.closestPoint));
    const distanceError = Math.abs(c.gpu.distanceSq - c.expected.distanceSq);
    const finite = [...c.gpu.closestPoint, ...c.gpu.barycoord, c.gpu.distanceSq].every(Number.isFinite);
    const baryValid = Math.abs(c.gpu.barycoord.reduce((a,b) => a+b, 0)-1) <= t.barycentricSumAbsolute
      && c.gpu.barycoord.every(v => v >= t.barycentricLowerBound);
    const bad = !finite || !baryValid || pointError > t.pointAbsolute
      || distanceError > t.distanceSqAbsolute + t.distanceSqRelative*c.expected.distanceSq;
    assert.equal(c.mismatch, bad, c.id); mismatches += Number(bad);
    groups[c.group] ??= {count: 0, mismatches: 0};
    groups[c.group].count++; groups[c.group].mismatches += Number(bad);
  }
  assert.equal(mismatches, index === 0 ? 10 : 0);
  assert.equal(probe.mismatchCount, mismatches);
  assert.deepEqual(groups, entry.groups);
  reports.push(report); summaries.push({variant: entry.variant, cases: 200, mismatches});
}
const [original, corrected] = reports;
const inputs = r => r.triangleProbe.cases.map(({id, group, permutation, p, a, b, c, expected, threeCpu}) =>
  ({id, group, permutation, p, a, b, c, expected, threeCpu}));
assert.deepEqual(inputs(original), inputs(corrected));
assert.deepEqual(original.triangleProbe.tolerances, corrected.triangleProbe.tolerances);
assert.deepEqual(original.sources, corrected.sources);
assert.equal(Object.keys(original.sources).length, 18);
assert.deepEqual(Object.keys(original.servedSources).sort(), Object.keys(corrected.servedSources).sort());
assert.equal(Object.keys(original.servedSources).length, 67);
assert.deepEqual(Object.keys(original.servedSources).filter(k => original.servedSources[k] !== corrected.servedSources[k]),
  ['/node_modules/three-mesh-bvh/src/webgpu/tsl/fns.js']);
const thin = original.triangleProbe.cases.find(c => c.id === 'thin-obtuse-0.001/012');
console.log(JSON.stringify({verified: true, gpuRerun: false, reports: summaries,
  thinCaseDistanceOverestimate: Math.sqrt(thin.gpu.distanceSq / thin.expected.distanceSq),
  independentCpuReferenceCases: 400, servedSourceDifferences: 1}, null, 2));
