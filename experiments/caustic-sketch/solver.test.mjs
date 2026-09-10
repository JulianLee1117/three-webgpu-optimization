import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { deflateSync } from 'node:zlib';
import { silhouette, solve, solveSamples, makeRays, forward, targetSamples, targetSamplesFromMask, photonImage } from './solver.mjs';
import { sampleHeight, sampleSurface, independentSnellTrace } from './optics.mjs';
import { processRequest } from './worker.mjs';

const kinds = ['heart', 'ring', 'letterA', 'star', 'two-dot'];
const near = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b}`);
function lens(grid = 9) { return { grid, height: new Float64Array(grid * grid), ior: 1.49, distance: 10, extent: 1, baseZ: -0.5 }; }

/** Independent validation: interpolate height directly from four corners,
 * estimate its normal by geometric finite differences, then use vector Snell.
 * It does not call the solver's ray interpolation, forward map or derivatives.
 */
function independentTrace(surface, resolution = 144) {
  const at = (x, y) => {
    const gx = (x + 1) * (surface.grid - 1) / 2, gy = (y + 1) * (surface.grid - 1) / 2;
    const ix = Math.min(surface.grid - 2, Math.max(0, Math.floor(gx))), iy = Math.min(surface.grid - 2, Math.max(0, Math.floor(gy)));
    const a = gx - ix, b = gy - iy, h = surface.height;
    const low = h[iy * surface.grid + ix] * (1 - a) + h[iy * surface.grid + ix + 1] * a;
    const high = h[(iy + 1) * surface.grid + ix] * (1 - a) + h[(iy + 1) * surface.grid + ix + 1] * a;
    return low * (1 - b) + high * b;
  };
  const hits = new Float64Array(resolution * resolution * 2), epsilon = 1e-5;
  let maxSlope = 0, lost = 0;
  for (let row = 0; row < resolution; row++) for (let col = 0; col < resolution; col++) {
    // Off-grid positions prevent reusing the optimization rays.
    const x = -1 + 2 * (col + 0.37) / resolution, y = -1 + 2 * (row + 0.63) / resolution;
    const height = at(x, y);
    const hx = (at(x + epsilon, y) - at(x - epsilon, y)) / (2 * epsilon);
    const hy = (at(x, y + epsilon) - at(x, y - epsilon)) / (2 * epsilon);
    maxSlope = Math.max(maxSlope, Math.hypot(hx, hy));
    const normalLength = Math.sqrt(hx * hx + hy * hy + 1), normal = [hx / normalLength, hy / normalLength, -1 / normalLength];
    const cosine = -normal[2], transmittedCosineSquared = 1 - surface.ior * surface.ior * (1 - cosine * cosine);
    if (transmittedCosineSquared < 0) { lost++; continue; }
    const correction = surface.ior * cosine - Math.sqrt(transmittedCosineSquared);
    const direction = [correction * normal[0], correction * normal[1], surface.ior + correction * normal[2]];
    near(Math.hypot(...direction), 1, 2e-12);
    const distance = (surface.distance - height) / direction[2], i = (row * resolution + col) * 2;
    hits[i] = x + distance * direction[0]; hits[i + 1] = y + distance * direction[1];
  }
  return { hits, maxSlope, lost };
}

test('flat slab leaves the normally incident parallel beam unchanged', () => {
  const surface = lens(), rays = makeRays(16, surface.grid), actual = forward(surface, rays).hits;
  rays.forEach((r, i) => { near(actual[2 * i], r.x); near(actual[2 * i + 1], r.y); });
});

test('a planar wedge agrees with the trigonometric refraction angle', () => {
  const surface = lens(), slope = 0.15;
  for (let y = 0; y < surface.grid; y++) for (let x = 0; x < surface.grid; x++) surface.height[y * surface.grid + x] = slope * (-1 + 2 * x / (surface.grid - 1));
  const angle = Math.asin(surface.ior * Math.sin(Math.atan(slope))) - Math.atan(slope);
  const rays = makeRays(16, surface.grid), actual = forward(surface, rays).hits;
  rays.forEach((r, i) => { near(actual[2 * i], r.x + (surface.distance - slope * r.x) * Math.tan(angle)); near(actual[2 * i + 1], r.y); });
});

test('analytic height derivatives agree with numerical control-point perturbations', () => {
  const surface = lens();
  for (let y = 0; y < surface.grid; y++) for (let x = 0; x < surface.grid; x++) surface.height[y * surface.grid + x] = 0.008 * Math.sin(x * 0.63) * Math.cos(y * 0.8);
  const rays = makeRays(16, surface.grid), base = forward(surface, rays, true), epsilon = 1e-7;
  for (const ri of [0, 57, 104, 219, 255]) for (let corner = 0; corner < 4; corner++) {
    const r = rays[ri], index = r.indices[corner], old = surface.height[index];
    surface.height[index] = old + epsilon; const plus = forward(surface, [r]).hits;
    surface.height[index] = old - epsilon; const minus = forward(surface, [r]).hits;
    surface.height[index] = old;
    for (let c = 0; c < 2; c++) {
      const j = ri * 6 + c * 3;
      const analytic = base.jacobian[j] * r.weight[corner] + base.jacobian[j + 1] * r.dx[corner] + base.jacobian[j + 2] * r.dy[corner];
      near(analytic, (plus[c] - minus[c]) / (2 * epsilon), 3e-7);
    }
  }
});

test('the independent geometric-normal vector tracer agrees on a nonplanar surface', () => {
  const surface = lens();
  for (let y = 0; y < surface.grid; y++) for (let x = 0; x < surface.grid; x++) surface.height[y * surface.grid + x] = 0.03 * Math.sin(x * 0.43) * Math.cos(y * 0.3);
  const independent = independentTrace(surface, 48), expected = forward(surface, makeRays(48, surface.grid, 0.37, 0.63));
  let maximum = 0;
  independent.hits.forEach((v, i) => { maximum = Math.max(maximum, Math.abs(v - expected.hits[i])); });
  assert.ok(maximum < 1e-8, `Independent ray disagreement ${maximum}`);
  assert.equal(independent.lost, 0);
});

test('photon deposition accounts for every incoming ray, including escaped flux', () => {
  const hits = new Float64Array([0, 0, 0.2, 0.7, 1.15, -1.15, -5, 5]);
  const image = photonImage(hits, 48, 1.15);
  near(image.pixels.reduce((sum, v) => sum + v, 0) + image.escapedFraction * 4, 4, 1e-10);
});

test('all five deterministic fits improve and preserve finite physical surfaces', () => {
  for (const kind of kinds) {
    const result = solve(kind), reference = independentTrace(result.lens, 96);
    assert.ok(result.finalLoss < result.initialLoss / 20, `${kind} did not improve`);
    assert.equal(reference.lost, 0, `${kind} lost rays to total internal reflection`);
    assert.ok(reference.maxSlope < 0.7, `${kind} has excessive slope`);
    assert.ok(result.lens.height.every((v) => Number.isFinite(v) && v > result.lens.baseZ && v < result.lens.distance));
    assert.ok(result.milliseconds < 5000, `${kind} exceeded the interactive canary budget`);
  }
});

test('an unnamed target distribution is optimized without the silhouette presets', () => {
  const n = 16, target = new Float64Array(n * n * 2), angle = 0.42;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const u = (2 * (i + 0.5) / n - 1) * 0.72, v = (2 * (j + 0.5) / n - 1) * 0.19;
    target[2 * (j * n + i)] = u * Math.cos(angle) - v * Math.sin(angle);
    target[2 * (j * n + i) + 1] = u * Math.sin(angle) + v * Math.cos(angle);
  }
  const result = solveSamples(target, { resolution: n, grid: 9, iterations: 240 });
  assert.ok(result.finalLoss < result.initialLoss / 50);
  assert.equal(independentTrace(result.lens, 48).lost, 0);
});

test('invalid inputs and excessive workloads fail explicitly', () => {
  assert.throws(() => solve('heart', { resolution: 10000 }), /bound/);
  assert.throws(() => solve('heart', { iterations: 100000 }), /Iterations/);
  assert.throws(() => solve('heart', { learningRate: NaN }), /Learning rate/);
  assert.throws(() => solve('heart', { smoothness: -1 }), /Smoothness/);
  assert.throws(() => targetSamples('heart', 1024, -1), /Background/);
  assert.throws(() => solveSamples(new Float64Array([0, 0])), /Target/);
  assert.throws(() => makeRays(32, 17, -1), /offsets/);
  assert.throws(() => photonImage(new Float64Array([0, 0]), 1000000), /bounded/);
  const surface = lens(); surface.height[2] = NaN;
  assert.throws(() => forward(surface, makeRays(16, surface.grid)), /finite/);
});

test('inverse-CDF mask sampling preserves canvas orientation and grayscale mass', () => {
  const top = new Uint8Array([255, 255, 0, 0]);
  const first = targetSamplesFromMask(top, 2, 2, 1024, { background: 0 });
  for (let i = 1; i < first.length; i += 2) assert.ok(first[i] > 0, 'Canvas top row must become positive analytical y');
  assert.deepEqual(first, targetSamplesFromMask(top, 2, 2, 1024, { background: 0 }));
  const weighted = targetSamplesFromMask(new Float32Array([0, 0.5, 1]), 3, 1, 3000, { background: 0 }), counts = [0, 0, 0];
  for (let i = 0; i < weighted.length; i += 2) counts[Math.min(2, Math.floor((weighted[i] + 1) * 1.5))]++;
  assert.deepEqual(counts, [0, 1000, 2000]);
});

test('a sparse maximum-size mask has bounded sampling and malformed masks fail', () => {
  const mask = new Uint8Array(512 * 512); mask[0] = 255;
  const points = targetSamplesFromMask(mask, 512, 512, 1024, { background: 0 });
  for (let i = 0; i < points.length; i += 2) { assert.ok(points[i] < -0.996); assert.ok(points[i + 1] > 0.996); }
  assert.throws(() => targetSamplesFromMask(new Uint8Array(4), 2, 3, 32), /one-channel/);
  assert.throws(() => targetSamplesFromMask(new Uint8Array(4), 2, 2, 32, { background: 0 }), /no light/);
  assert.throws(() => targetSamplesFromMask(new Float32Array([1, 2]), 2, 1, 32), /density/);
  assert.throws(() => targetSamplesFromMask(new Uint8Array(4), 1024, 1, 32), /dimensions/);
});

test('progress reports copied current heightfields, including initial and final states', () => {
  const options = { resolution: 16, grid: 9, iterations: 120 };
  const snapshots = [], reference = solve('ring', options);
  const result = solve('ring', { ...options, onProgress: ({ iteration, height, loss }) => {
    snapshots.push({ iteration, height: height.slice(), loss });
    height.fill(9999); // Public progress state must not alias the live solver.
  } });
  assert.deepEqual(snapshots.map((v) => v.iteration), [0, 50, 100, 120]);
  assert.ok(snapshots[0].height.every((v) => v === 0));
  assert.deepEqual(result.lens.height, reference.lens.height);
  assert.deepEqual(snapshots.at(-1).height, result.lens.height);
  near(snapshots.at(-1).loss, result.finalLoss, 1e-14);
});

test('reusable independent optics agrees with the original validator and receiver-distance scaling', () => {
  const surface = lens(), slope = 0.15;
  for (let y = 0; y < surface.grid; y++) for (let x = 0; x < surface.grid; x++) surface.height[y * surface.grid + x] = slope * (-1 + 2 * x / (surface.grid - 1));
  near(sampleHeight(surface, 1, -1), slope);
  const sample = sampleSurface(surface, 0.5, 0.2);
  near(sample.height, 0.5 * slope);
  near(sample.gradient[0], slope, 1e-10);
  near(sample.normal[0], -slope / Math.hypot(slope, 1), 1e-10);
  assert.ok(sample.normal[2] > 0);
  const current = independentSnellTrace(surface, { resolution: 48, fresnel: false });
  const old = independentTrace(surface, 48), farther = independentSnellTrace(surface, { resolution: 48, receiverDistance: 15, fresnel: false });
  current.hits.forEach((v, i) => near(v, old.hits[i], 2e-10));
  const angle = Math.asin(surface.ior * Math.sin(Math.atan(slope))) - Math.atan(slope);
  for (let i = 0; i < current.hits.length; i += 2) { near(farther.hits[i] - current.hits[i], 5 * Math.tan(angle), 1e-9); near(farther.hits[i + 1], current.hits[i + 1]); }
});

test('Fresnel transmission and deposited/escaped photon energy are accounted for', () => {
  const surface = lens(); surface.ior = 1.5;
  const trace = independentSnellTrace(surface, { resolution: 48 });
  const expected = (1 - 0.04) ** 2;
  trace.weights.forEach((v) => near(v, expected, 1e-12));
  near(trace.energy.incident, 4);
  near(trace.energy.transmitted, 4 * expected, 1e-10);
  near(trace.energy.reflected, 4 * (1 - expected), 1e-10);
  near(trace.energy.conservationError, 0, 1e-10);
  const image = photonImage(trace.hits, 48, 0.7, trace.weights);
  near(image.pixels.reduce((s, v) => s + v, 0) + image.escapedWeight, trace.weights.reduce((s, v) => s + v, 0), 1e-8);
  assert.ok(image.escapedFraction > 0.4);
});

test('total internal reflection is explicit and reflected rays cannot deposit ghost light', () => {
  const surface = lens(); surface.ior = 1.5; surface.baseZ = -2;
  for (let y = 0; y < surface.grid; y++) for (let x = 0; x < surface.grid; x++) surface.height[y * surface.grid + x] = -1 + 2 * x / (surface.grid - 1);
  const trace = independentSnellTrace(surface, { resolution: 16 });
  assert.equal(trace.lost, 256);
  assert.ok(trace.hits.every(Number.isNaN));
  assert.ok(trace.weights.every((v) => v === 0));
  near(trace.energy.transmitted, 0);
  near(trace.energy.reflected, 4);
  assert.ok(photonImage(trace.hits, 16, 1.15, trace.weights).pixels.every((v) => v === 0));
});

test('worker protocol transfers progress copies without detaching live solver state', () => {
  const messages = [], mask = new Uint8Array(32 * 32);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) mask[y * 32 + x] = silhouette('heart', 2 * (x + 0.5) / 32 - 1, 1 - 2 * (y + 0.5) / 32) ? 255 : 0;
  processRequest({ id: 'test-mask', mask, width: 32, height: 32, resolution: 16, grid: 9, iterations: 120 }, (value, transfer) => messages.push(structuredClone(value, { transfer })));
  assert.deepEqual(messages.filter((v) => v.type === 'progress').map((v) => v.iteration), [0, 50, 100, 120]);
  const result = messages.at(-1);
  assert.equal(result.type, 'result'); assert.equal(result.id, 'test-mask');
  assert.ok(result.lens.height instanceof Float64Array);
  assert.equal(result.lens.height.length, 81);
  assert.ok(result.finalLoss < result.initialLoss / 20);
  assert.equal(independentSnellTrace(result.lens, { resolution: 48 }).lost, 0);
});

function metrics(kind, hits, image) {
  let inside = 0;
  for (let i = 0; i < hits.length; i += 2) inside += silhouette(kind, hits[i], hits[i + 1]) ? 1 : 0;
  let overlap = 0, union = 0, targetCells = 0, predictedCells = 0;
  const count = hits.length / 2, pixelArea = (2 * image.halfExtent / image.resolution) ** 2;
  for (let y = 0; y < image.resolution; y++) for (let x = 0; x < image.resolution; x++) {
    const isTarget = silhouette(kind, (2 * (x + 0.5) / image.resolution - 1) * image.halfExtent, (2 * (y + 0.5) / image.resolution - 1) * image.halfExtent);
    // A fixed threshold of one times the incident irradiance, chosen without
    // optimizing it against each shape. Images contain all traced photons.
    const prediction = image.pixels[y * image.resolution + x] * 4 / (count * pixelArea) > 1;
    targetCells += +isTarget; predictedCells += +prediction;
    overlap += +(isTarget && prediction); union += +(isTarget || prediction);
  }
  return { fractionInsideSilhouette: inside / count, brightRegionIoU: overlap / union, targetCells, predictedCells, escapedFraction: image.escapedFraction };
}

// Minimal PNG encoder for numerical evidence, using Node's built-in zlib only.
function png(width, height, rgba) {
  const crc = (data) => { let value = 0xffffffff; for (const byte of data) { value ^= byte; for (let i = 0; i < 8; i++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0); } return (value ^ 0xffffffff) >>> 0; };
  const chunk = (name, data) => { const type = Buffer.from(name), length = Buffer.alloc(4), check = Buffer.alloc(4); length.writeUInt32BE(data.length); check.writeUInt32BE(crc(Buffer.concat([type, data]))); return Buffer.concat([length, type, data, check]); };
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 6;
  const scan = Buffer.alloc(height * (1 + width * 4));
  for (let y = 0; y < height; y++) rgba.copy(scan, y * (1 + width * 4) + 1, y * width * 4, (y + 1) * width * 4);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', deflateSync(scan)), chunk('IEND', Buffer.alloc(0))]);
}

function contactSheet(rows) {
  const tile = 192, border = 12, width = tile * 5 + border * 6, height = tile * 3 + border * 4;
  const pixels = Buffer.alloc(width * height * 4, 255);
  for (let row = 0; row < 3; row++) for (let col = 0; col < 5; col++) {
    const item = rows[row * 5 + col];
    for (let y = 0; y < tile; y++) for (let x = 0; x < tile; x++) {
      let intensity;
      if (row === 0) intensity = silhouette(kinds[col], (2 * (x + 0.5) / tile - 1) * 1.15, (1 - 2 * (y + 0.5) / tile) * 1.15) ? 1 : 0.04;
      else {
        const px = Math.min(item.image.resolution - 1, Math.floor(x / tile * item.image.resolution));
        const py = Math.min(item.image.resolution - 1, Math.floor((tile - 1 - y) / tile * item.image.resolution));
        const area = (2 * item.image.halfExtent / item.image.resolution) ** 2;
        const irradiance = item.image.pixels[py * item.image.resolution + px] * 4 / (item.photonCount * area);
        intensity = 1 - Math.exp(-irradiance * 0.5);
      }
      const i = ((border + row * (tile + border) + y) * width + border + col * (tile + border) + x) * 4;
      const value = Math.round(clamp(intensity, 0, 1) * 255);
      pixels[i] = pixels[i + 1] = pixels[i + 2] = value;
    }
  }
  return png(width, height, pixels);
}
const clamp = (x, low, high) => Math.max(low, Math.min(high, x));

if (process.argv.includes('--report') || process.argv.includes('--mask-report')) {
  const maskPipeline = process.argv.includes('--mask-report');
  const runId = new Date().toISOString().replaceAll(':', '-');
  const out = resolve('results/development/caustic-sketch-cpu', runId);
  mkdirSync(out, { recursive: true });
  const cases = [], rows = new Array(5).fill(null);
  for (const resolution of [32, 48]) for (const kind of kinds) {
    let result;
    if (maskPipeline) {
      const mask = new Uint8Array(128 * 128);
      for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) mask[y * 128 + x] = silhouette(kind, 2 * (x + 0.5) / 128 - 1, 1 - 2 * (y + 0.5) / 128) ? 255 : 0;
      processRequest({ id: kind, mask, width: 128, height: 128, resolution, grid: resolution / 2 + 1, iterations: 400 }, (value, transfer) => {
        const message = structuredClone(value, { transfer });
        if (message.type === 'result') result = message;
      });
    } else result = solve(kind, { resolution, grid: resolution / 2 + 1, iterations: resolution === 32 ? 400 : 600, learningRate: resolution === 32 ? 0.0009 : 0.0007 });
    const reference = maskPipeline ? independentSnellTrace(result.lens, { resolution: 144 }) : independentTrace(result.lens, 144);
    const image = photonImage(reference.hits, 48, 1.15, reference.weights ?? null);
    const direct = forward(result.lens, makeRays(144, result.lens.grid, 0.37, 0.63));
    let maximumDisagreement = 0;
    reference.hits.forEach((v, i) => { maximumDisagreement = Math.max(maximumDisagreement, Math.abs(v - direct.hits[i])); });
    const flat = maskPipeline ? independentSnellTrace(lens(result.lens.grid), { resolution: 144 }) : independentTrace(lens(result.lens.grid), 144);
    const baseline = metrics(kind, flat.hits, photonImage(flat.hits, 48, 1.15, flat.weights ?? null));
    const entry = {
      kind, ...result, lens: { ...result.lens, height: Array.from(result.lens.height) }, target: undefined,
      independent: { ...metrics(kind, reference.hits, image), rayCount: reference.hits.length / 2, maximumDisagreement, maxSlope: reference.maxSlope, lost: reference.lost, energy: reference.energy },
      baseline,
    };
    cases.push(entry);
    rows.push({ image, photonCount: reference.hits.length / 2 });
    console.log(JSON.stringify({ kind, resolution, ms: result.milliseconds, initialLoss: result.initialLoss, finalLoss: result.finalLoss, independent: entry.independent, baseline }));
  }
  const report = {
    runId, maskPipeline, method: 'Original projected Adam on a bilinear heightfield; exact single-exit Snell forward model; 6 sliced distribution directions per update',
    targetGeneration: maskPipeline ? '128×128 byte masks; finite inverse CDF; transferable worker protocol; 400 iterations at both resolutions' : 'Procedural silhouette acceptance of deterministic Halton samples',
    provenance: 'Targets are optimization samples only. Final evaluation derives solely from exported heightfield data.',
    physicalModel: { incoming: 'Uniform vertical parallel rays; normal incidence at flat entry face', ior: 1.49, receiverZ: 10, lensBaseZ: -0.5, absorption: 'ignored', Fresnel: maskPipeline ? 'Unpolarized dielectric entry and exit reflection; internal multiple bounces untraced' : 'ignored', diffraction: 'ignored', spectralDispersion: 'ignored' },
    limitations: ['Small fixed parallel illumination; no manufacturing claim', 'Grid surface has piecewise bilinear normals', 'No GPU or browser latency measurement', 'Recognition is assessed separately from scalar distribution loss'],
    sources: Object.fromEntries(['solver.mjs', 'solver.test.mjs', 'optics.mjs', 'worker.mjs'].map((file) => [file, createHash('sha256').update(readFileSync(new URL(file, import.meta.url))).digest('hex')])),
    contactSheet: { file: 'contact-sheet.png', columns: kinds, rows: ['target masks', '32×32 optimization rays; 17×17 height controls; 144×144 independent evaluation rays', '48×48 optimization rays; 25×25 height controls; 144×144 independent evaluation rays'], exposure: 'Fixed: 1-exp(-.5 * relative irradiance) for both output rows; no target image in outputs' },
    cases,
  };
  writeFileSync(resolve(out, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  writeFileSync(resolve(out, 'contact-sheet.png'), contactSheet(rows));
  console.log(`REPORT ${out}`);
}
