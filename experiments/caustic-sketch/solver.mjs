/**
 * Original bounded CPU experiment in inverse caustic design.
 * A vertical parallel beam enters a flat face normally, travels inside glass,
 * and exits a bilinear heightfield. The receiver is the plane z = distance.
 * Target samples are used only in the sliced distribution loss. Forward images
 * are reconstructed solely from the heightfield, Snell's law and photon hits.
 */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export function silhouette(kind, x, y) {
  if (kind === 'heart') {
    x /= 0.72; y = (y - 0.05) / 0.72;
    return (x * x + y * y - 0.75) ** 3 - x * x * y * y * y <= 0;
  }
  if (kind === 'ring') { const r = Math.hypot(x, y); return r > 0.37 && r < 0.7; }
  if (kind === 'letterA') {
    const ax = Math.abs(x), half = 0.62 * (0.76 - y) / 1.52;
    return y >= -0.76 && y <= 0.76 && ((ax <= half && ax >= Math.max(0, half - 0.19)) || (Math.abs(y + 0.13) < 0.095 && ax < half));
  }
  if (kind === 'star') {
    const points = Array.from({ length: 10 }, (_, i) => {
      const a = Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? 0.32 : 0.77;
      return [Math.cos(a) * r, Math.sin(a) * r];
    });
    let inside = false;
    for (let i = 0, j = 9; i < 10; j = i++) {
      const [xi, yi] = points[i], [xj, yj] = points[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  if (kind === 'two-dot') return Math.hypot(x - 0.43, y) < 0.25 || Math.hypot(x + 0.43, y) < 0.25;
  throw new RangeError(`Unknown target ${kind}`);
}

function radicalInverse(index, base) {
  let result = 0, weight = 1 / base;
  while (index) { result += (index % base) * weight; index = Math.floor(index / base); weight /= base; }
  return result;
}

export function targetSamples(kind, count, background = 0.04) {
  if (!Number.isInteger(count) || count < 16 || count > 16384) throw new RangeError('Target sample count must be in [16, 16384]');
  if (!Number.isFinite(background) || background < 0 || background > 0.5) throw new RangeError('Background density must be in [0, .5]');
  const points = new Float64Array(count * 2);
  let found = 0;
  for (let candidate = 1; candidate <= 200000 && found < count; candidate++) {
    const x = 2 * radicalInverse(candidate, 2) - 1, y = 2 * radicalInverse(candidate, 3) - 1;
    if (silhouette(kind, x, y) || radicalInverse(candidate, 5) < background) {
      points[2 * found] = x; points[2 * found + 1] = y; found++;
    }
  }
  if (found !== count) throw new Error('Bounded target sampler did not fill');
  return points;
}

/**
 * Sample a drawn one-channel density mask with a finite inverse-CDF algorithm.
 * Byte masks use [0,255]; floating masks use [0,1]. Canvas row zero is the top,
 * so the returned analytical y coordinate decreases as the row index increases.
 * A uniform density floor makes dark regions dim rather than requiring an
 * infinitely high-contrast, discontinuous ray transport map.
 */
export function targetSamplesFromMask(data, width, height, count, { background = 0.04 } = {}) {
  if (![width, height].every((v) => Number.isInteger(v) && v >= 1 && v <= 512)) throw new RangeError('Mask dimensions must be in [1,512]');
  if (!Number.isInteger(count) || count < 16 || count > 16384) throw new RangeError('Target sample count must be in [16,16384]');
  if (!Number.isFinite(background) || background < 0 || background > 0.5) throw new RangeError('Background density must be in [0,.5]');
  const bytes = data instanceof Uint8Array || data instanceof Uint8ClampedArray;
  if ((!bytes && !(data instanceof Float32Array) && !(data instanceof Float64Array)) || data.length !== width * height) throw new TypeError('Mask must be a one-channel byte or normalized floating array');
  const cdf = new Float64Array(data.length), divisor = bytes ? 255 : 1;
  let total = 0;
  for (let i = 0; i < data.length; i++) {
    const density = data[i] / divisor;
    if (!Number.isFinite(density) || density < 0 || density > 1) throw new RangeError('Mask density is outside [0,1]');
    total += background + (1 - background) * density;
    cdf[i] = total;
  }
  if (!(total > 0)) throw new RangeError('Mask has no light density');
  const points = new Float64Array(count * 2);
  for (let i = 0; i < count; i++) {
    const quantile = (i + 0.5) / count * total;
    let low = 0, high = cdf.length - 1;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (cdf[middle] < quantile) low = middle + 1;
      else high = middle;
    }
    const col = low % width, row = Math.floor(low / width);
    const x = (col + radicalInverse(i + 1, 2)) / width;
    const y = (row + radicalInverse(i + 1, 3)) / height;
    points[2 * i] = 2 * x - 1;
    points[2 * i + 1] = 1 - 2 * y;
  }
  return points;
}

export function makeRays(resolution, grid, offsetX = 0.5, offsetY = 0.5) {
  if (!Number.isInteger(resolution) || resolution < 8 || resolution > 192 || !Number.isInteger(grid) || grid < 3 || grid > 49) throw new RangeError('Ray or lens grid exceeds canary bound');
  if (![offsetX, offsetY].every((v) => Number.isFinite(v) && v > 0 && v < 1)) throw new RangeError('Ray offsets must lie inside their pixel cells');
  const count = resolution * resolution, rays = new Array(count), spacing = 2 / (grid - 1);
  for (let j = 0; j < resolution; j++) for (let i = 0; i < resolution; i++) {
    const x = -1 + 2 * (i + offsetX) / resolution, y = -1 + 2 * (j + offsetY) / resolution;
    const gx = (x + 1) / spacing, gy = (y + 1) / spacing;
    const ix = Math.min(grid - 2, Math.floor(gx)), iy = Math.min(grid - 2, Math.floor(gy));
    const tx = gx - ix, ty = gy - iy;
    rays[j * resolution + i] = {
      x, y, indices: [iy * grid + ix, iy * grid + ix + 1, (iy + 1) * grid + ix, (iy + 1) * grid + ix + 1],
      weight: [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty],
      dx: [-(1 - ty) / spacing, (1 - ty) / spacing, -ty / spacing, ty / spacing],
      dy: [-(1 - tx) / spacing, -tx / spacing, (1 - tx) / spacing, tx / spacing],
    };
  }
  return rays;
}

function sampleRay(height, ray) {
  let h = 0, px = 0, py = 0;
  for (let c = 0; c < 4; c++) { const v = height[ray.indices[c]]; h += v * ray.weight[c]; px += v * ray.dx[c]; py += v * ray.dy[c]; }
  return [h, px, py];
}

export function forward(lens, rays, derivatives = false) {
  const { height, ior, distance } = lens;
  if (!(height instanceof Float64Array) || height.length !== lens.grid * lens.grid || !height.every(Number.isFinite)
    || !Number.isFinite(ior) || ior < 1 || ior > 2 || !Number.isFinite(distance) || distance <= 0 || distance > 100) throw new RangeError('Invalid finite optical surface');
  const hits = new Float64Array(rays.length * 2), jacobian = derivatives ? new Float64Array(rays.length * 6) : null;
  let maxSlope = 0;
  for (let index = 0; index < rays.length; index++) {
    const ray = rays[index], [h, px, py] = sampleRay(height, ray), s = px * px + py * py, k = ior * ior - 1;
    maxSlope = Math.max(maxSlope, Math.sqrt(s));
    if (k * s >= 1) throw new Error('Lens produced total internal reflection');
    const q = Math.sqrt(1 - k * s), a = ior - q, b = ior * s + q, beta = a / b;
    const length = distance - h;
    if (!(length > 0)) throw new Error('Lens crossed receiver');
    hits[2 * index] = ray.x + length * beta * px;
    hits[2 * index + 1] = ray.y + length * beta * py;
    if (derivatives) {
      const dq = -k / (2 * q), dbeta = (-dq * b - a * (ior + dq)) / (b * b);
      const base = index * 6;
      jacobian[base] = -beta * px;
      jacobian[base + 1] = length * (beta + 2 * dbeta * px * px);
      jacobian[base + 2] = length * 2 * dbeta * px * py;
      jacobian[base + 3] = -beta * py;
      jacobian[base + 4] = length * 2 * dbeta * px * py;
      jacobian[base + 5] = length * (beta + 2 * dbeta * py * py);
    }
  }
  return { hits, jacobian, maxSlope };
}

export function makeLoss(target, directions = 24) {
  const count = target.length / 2;
  return Array.from({ length: directions }, (_, i) => {
    const angle = (i + 0.5) / directions * Math.PI, x = Math.cos(angle), y = Math.sin(angle);
    const projected = Float64Array.from({ length: count }, (_, p) => x * target[p * 2] + y * target[p * 2 + 1]);
    projected.sort();
    return { x, y, projected };
  });
}

function lossGradient(hits, targets, offset = 0, batch = targets.length) {
  const count = hits.length / 2, gradient = new Float64Array(hits.length), projected = new Float64Array(count), order = new Uint32Array(count);
  let loss = 0;
  for (let direction = 0; direction < batch; direction++) {
    const target = targets[(offset + direction * 7) % targets.length];
    for (let p = 0; p < count; p++) { projected[p] = hits[p * 2] * target.x + hits[p * 2 + 1] * target.y; order[p] = p; }
    order.sort((a, b) => projected[a] - projected[b]);
    for (let p = 0; p < count; p++) {
      const index = order[p], delta = projected[index] - target.projected[p];
      loss += delta * delta / (count * batch);
      gradient[index * 2] += 2 * delta * target.x / (count * batch);
      gradient[index * 2 + 1] += 2 * delta * target.y / (count * batch);
    }
  }
  return { loss, gradient };
}

function projectHeight(height, grid) {
  const maximumDifference = 0.45 * 2 / (grid - 1);
  for (let pass = 0; pass < 4; pass++) for (let y = 0; y < grid; y++) for (let x = 0; x < grid; x++) {
    const a = y * grid + x;
    for (const b of [x + 1 < grid ? a + 1 : -1, y + 1 < grid ? a + grid : -1]) {
      if (b < 0) continue;
      const excess = height[b] - height[a] - clamp(height[b] - height[a], -maximumDifference, maximumDifference);
      height[a] += excess / 2; height[b] -= excess / 2;
    }
  }
  let mean = height.reduce((sum, v) => sum + v, 0) / height.length;
  for (let i = 0; i < height.length; i++) height[i] = clamp(height[i] - mean, -0.35, 0.35);
}

export function solve(kind, { resolution = 32, grid = 17, iterations = 400, learningRate = 0.0009, background = 0.04, smoothness = 0.00001, onProgress } = {}) {
  const started = performance.now();
  makeRays(resolution, grid); // Validate bounds before allocating target samples.
  const target = targetSamples(kind, resolution * resolution, background);
  const result = solveSamples(target, { resolution, grid, iterations, learningRate, background, smoothness, onProgress });
  result.pipelineMilliseconds = performance.now() - started;
  return result;
}

/** The solver also accepts an arbitrary sampled target distribution. No target
 * label or silhouette function is used by this optimization path. Coordinates
 * cover the receiver's [-1,1] square, matching the source aperture footprint.
 */
export function solveSamples(target, { resolution = 32, grid = 17, iterations = 400, learningRate = 0.0009, background = null, smoothness = 0.00001, onProgress } = {}) {
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 1200) throw new RangeError('Iterations must be in [1, 1200]');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 0.01) throw new RangeError('Learning rate must be in (0, .01]');
  if (!Number.isFinite(smoothness) || smoothness < 0 || smoothness > 0.1) throw new RangeError('Smoothness must be in [0, .1]');
  if (onProgress !== undefined && typeof onProgress !== 'function') throw new TypeError('onProgress must be a function');
  const start = performance.now(), rays = makeRays(resolution, grid);
  if (!(target instanceof Float64Array) || target.length !== rays.length * 2 || !target.every((v) => Number.isFinite(v) && Math.abs(v) <= 1)) throw new RangeError('Target must contain one finite normalized xy sample per ray');
  target = target.slice();
  const targets = makeLoss(target);
  const lens = { grid, height: new Float64Array(grid * grid), ior: 1.49, distance: 10, extent: 1, baseZ: -0.5 };
  const first = forward(lens, rays), initialLoss = lossGradient(first.hits, targets).loss;
  const firstMoment = new Float64Array(lens.height.length), secondMoment = new Float64Array(lens.height.length), history = [];
  onProgress?.({ iteration: 0, height: lens.height.slice(), loss: initialLoss });
  for (let iteration = 0; iteration < iterations; iteration++) {
    if (performance.now() - start > 15000) throw new Error('Solver exceeded 15 second CPU bound');
    const { hits, jacobian } = forward(lens, rays, true), { loss, gradient } = lossGradient(hits, targets, iteration % targets.length, 6);
    const dh = new Float64Array(lens.height.length);
    for (let i = 0; i < rays.length; i++) {
      const ray = rays[i], gx = gradient[i * 2], gy = gradient[i * 2 + 1], j = i * 6;
      const h = gx * jacobian[j] + gy * jacobian[j + 3], px = gx * jacobian[j + 1] + gy * jacobian[j + 4], py = gx * jacobian[j + 2] + gy * jacobian[j + 5];
      for (let c = 0; c < 4; c++) dh[ray.indices[c]] += h * ray.weight[c] + px * ray.dx[c] + py * ray.dy[c];
    }
    // A curvature penalty discourages high-frequency overfitting to photon rays.
    const spacing = 2 / (grid - 1), scale = smoothness / ((grid - 2) ** 2 * spacing ** 4);
    for (let y = 1; y < grid - 1; y++) for (let x = 1; x < grid - 1; x++) {
      const p = y * grid + x, lap = lens.height[p - 1] + lens.height[p + 1] + lens.height[p - grid] + lens.height[p + grid] - 4 * lens.height[p];
      const d = 2 * scale * lap;
      dh[p - 1] += d; dh[p + 1] += d; dh[p - grid] += d; dh[p + grid] += d; dh[p] -= 4 * d;
    }
    const rate = learningRate * (0.2 + 0.8 * (1 - iteration / iterations));
    for (let p = 0; p < dh.length; p++) {
      firstMoment[p] = 0.9 * firstMoment[p] + 0.1 * dh[p];
      secondMoment[p] = 0.999 * secondMoment[p] + 0.001 * dh[p] * dh[p];
      const m = firstMoment[p] / (1 - 0.9 ** (iteration + 1)), v = secondMoment[p] / (1 - 0.999 ** (iteration + 1));
      lens.height[p] -= rate * m / (Math.sqrt(v) + 1e-9);
    }
    projectHeight(lens.height, grid);
    if (iteration % 50 === 0 || iteration === iterations - 1) history.push({ iteration, sampledLoss: loss });
    if (onProgress && ((iteration + 1) % 50 === 0 || iteration === iterations - 1)) {
      const progressLoss = lossGradient(forward(lens, rays).hits, targets).loss;
      onProgress({ iteration: iteration + 1, height: lens.height.slice(), loss: progressLoss });
    }
  }
  const final = forward(lens, rays), finalLoss = lossGradient(final.hits, targets).loss;
  return { lens, target, initialLoss, finalLoss, history, maxSlope: final.maxSlope, milliseconds: performance.now() - start, parameters: { resolution, grid, iterations, learningRate, background, smoothness } };
}

export function photonImage(hits, resolution = 48, halfExtent = 1.15, weights = null) {
  if (!Number.isInteger(resolution) || resolution < 8 || resolution > 256 || !Number.isFinite(halfExtent) || halfExtent <= 0 || halfExtent > 10) throw new RangeError('Photon image exceeds bounded dimensions');
  if (!(hits instanceof Float64Array) || hits.length % 2 || hits.length > 192 * 192 * 2) throw new RangeError('Photon hit buffer exceeds bounds');
  if (weights !== null && (!(weights instanceof Float64Array) || weights.length !== hits.length / 2 || !weights.every((v) => Number.isFinite(v) && v >= 0 && v <= 1))) throw new RangeError('Photon weights must be finite transmission fractions');
  const pixels = new Float64Array(resolution * resolution);
  let escaped = 0, totalWeight = 0;
  for (let i = 0; i < hits.length; i += 2) {
    const transmission = weights ? weights[i / 2] : 1;
    totalWeight += transmission;
    if (transmission === 0) continue;
    if (!Number.isFinite(hits[i]) || !Number.isFinite(hits[i + 1])) throw new RangeError('A transmitted photon has a non-finite hit');
    const gx = (hits[i] / (2 * halfExtent) + 0.5) * resolution - 0.5, gy = (hits[i + 1] / (2 * halfExtent) + 0.5) * resolution - 0.5;
    const ix = Math.floor(gx), iy = Math.floor(gy), tx = gx - ix, ty = gy - iy;
    for (let y = 0; y <= 1; y++) for (let x = 0; x <= 1; x++) {
      const weight = (x ? tx : 1 - tx) * (y ? ty : 1 - ty) * transmission;
      if (ix + x < 0 || ix + x >= resolution || iy + y < 0 || iy + y >= resolution) escaped += weight;
      else pixels[(iy + y) * resolution + ix + x] += weight;
    }
  }
  return { pixels, escapedFraction: totalWeight ? escaped / totalWeight : 0, escapedWeight: escaped, totalWeight, resolution, halfExtent };
}
