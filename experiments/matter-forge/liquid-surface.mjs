import { Color, MeshStandardNodeMaterial } from 'three/webgpu';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';

/** Display-only reconstruction of actual fluid particle positions. No blur,
 * particle motion, collision geometry, or new physical state is introduced.
 *
 * r186 addBall contributes max(S/(1e-6+(r/N)^2)-subtract,0). Solve S and
 * subtract so one particle has iso radius0.45 and zero field beyond radius0.50
 * in simulation grid units. Overlaps add, so this is not volume-preserving.
 * A marching-cubes vertex lies on a sampled edge: at the default0.5 spacing
 * it is at most1.0 grid unit from some field-contributing particle. Sampling
 * can merge droplets and clips the outer field layer; it is not contact repair.
 *
 * Optional colors are flat per-body-particle LINEAR RGB, indexed like state.x.
 * Normalize the addBall palette by its own smooth weight to avoid making
 * overlapping particles artificially brighter. The material base is white;
 * the default teal, or supplied asset colors, come from vertex colors.
 */
export function createLiquidSurface({ N = 24, resolution = 48, maxPolyCount = 20000 } = {}) {
  if (!Number.isInteger(N) || N < 8 || N > 32) throw new RangeError('Liquid surface N must be8..32');
  if (!Number.isInteger(resolution) || resolution < 2 * N || resolution > 64) throw new RangeError('Liquid surface resolution must be at least2*N and at most64');
  if (!Number.isInteger(maxPolyCount) || maxPolyCount < 1 || maxPolyCount > 20000) throw new RangeError('Liquid surface triangle budget must be1..20000');
  const epsilon = 1e-6, isolation = 80, isolatedRadius = .45, supportRadius = .5;
  const subtract = isolation * (epsilon + (isolatedRadius / N) ** 2) / ((supportRadius ** 2 - isolatedRadius ** 2) / N ** 2);
  const strength = subtract * (epsilon + (supportRadius / N) ** 2);
  const material = new MeshStandardNodeMaterial({ color: 0xffffff, metalness: .28, roughness: .26, vertexColors: true });
  const mesh = new MarchingCubes(resolution, material, false, true, maxPolyCount);
  mesh.name = 'MatterForge display-only liquid isosurface';
  mesh.isolation = isolation; mesh.scale.setScalar(N / 2); mesh.position.setScalar(N / 2);
  // Upstream's radius1 sphere does not contain the corners of its [-1,1]^3
  // domain. Keep a conservative fixed bound, avoiding stale dynamic culling.
  mesh.geometry.boundingSphere.radius = Math.sqrt(3);
  mesh.geometry.setDrawRange(0, 0); mesh.visible = false;
  const paletteWeights = new Float32Array(resolution ** 3), color = new Color(), teal = new Color(0x20cfb5);
  const stats = { updates: 0, disposed: false, fluidParticles: 0, triangles: 0, geometryOverflow: false,
    resolution, maxPolyCount, gridSpacing: N / resolution, isolation, subtract, strength, isolatedRadius, supportRadius,
    maximumInterpolationReach: supportRadius + N / resolution, clippedKernelParticles: 0, addonUpperSlabCompleted: true,
    elapsedMs: 0, maximumElapsedMs: 0, worldBounds: null, particleBounds: null, maximumAxisExpansion: 0 };

  // r186 addBall floors its upper integer bound and iterates strictly below it.
  // With a small kernel this omits positive samples on its uppermost slab and
  // biases the surface toward negative axes. Complete ONLY those omitted
  // samples with the same published addon formula, keeping particle positions
  // unchanged. Also accumulate the identical palette weight for normalization.
  function addPaletteWeight(x, y, z) {
    const radius = resolution * Math.sqrt(strength / subtract), radiusSquared = radius * radius;
    const cx = x * resolution, cy = y * resolution, cz = z * resolution;
    const minX = Math.max(1, Math.floor(cx - radius)), maxX = Math.min(resolution - 1, Math.floor(cx + radius));
    const minY = Math.max(1, Math.floor(cy - radius)), maxY = Math.min(resolution - 1, Math.floor(cy + radius));
    const minZ = Math.max(1, Math.floor(cz - radius)), maxZ = Math.min(resolution - 1, Math.floor(cz + radius));
    for (let k = minZ; k <= Math.min(maxZ, resolution - 2); k++) for (let j = minY; j <= Math.min(maxY, resolution - 2); j++) for (let i = minX; i <= Math.min(maxX, resolution - 2); i++) {
      const distanceSquared = (i / resolution - x) ** 2 + (j / resolution - y) ** 2 + (k / resolution - z) ** 2;
      const value = strength / (epsilon + distanceSquared) - subtract;
      if (value <= 0) continue;
      const ratio = Math.sqrt(((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) / radiusSquared);
      const index = i + resolution * (j + resolution * k), weight = 1 - ratio ** 3 * (ratio * (ratio * 6 - 15) + 10);
      paletteWeights[index] += weight;
      if (i >= maxX || j >= maxY || k >= maxZ) {
        mesh.field[index] += value;
        mesh.palette[3 * index] += color.r * weight;
        mesh.palette[3 * index + 1] += color.g * weight;
        mesh.palette[3 * index + 2] += color.b * weight;
      }
    }
  }
  function update(state, bodyCount, colors = null) {
    if (stats.disposed) throw new Error('Liquid surface is disposed');
    if (!Number.isInteger(bodyCount) || bodyCount < 0 || bodyCount > 4096 || !state?.x || state.x.length < bodyCount * 3 || !state?.mode || state.mode.length < bodyCount) throw new RangeError('Liquid surface expects at most4096 body particles');
    if (colors != null && colors.length < bodyCount * 3) throw new RangeError('Liquid colors must contain linear RGB for every body particle');
    const start = performance.now();
    mesh.visible = false; mesh.geometry.setDrawRange(0, 0); mesh.reset(); paletteWeights.fill(0);
    let count = 0, clipped = 0;
    const particleMin = [Infinity, Infinity, Infinity], particleMax = [-Infinity, -Infinity, -Infinity];
    for (let p = 0; p < bodyCount; p++) {
      if (state.mode[p] !== 0 && state.mode[p] !== 1) throw new RangeError('Unknown particle material mode');
      if (state.mode[p] !== 0) continue;
      const x = state.x[3 * p], y = state.x[3 * p + 1], z = state.x[3 * p + 2];
      if (![x, y, z].every(v => Number.isFinite(v) && v >= 0 && v <= N)) throw new RangeError('Liquid particle lies outside finite field domain');
      const coordinates = [x, y, z];
      if (coordinates.some(v => v - supportRadius < N / resolution || v + supportRadius > N - 2 * N / resolution)) clipped++;
      for (let d = 0; d < 3; d++) { particleMin[d] = Math.min(particleMin[d], coordinates[d]); particleMax[d] = Math.max(particleMax[d], coordinates[d]); }
      if (colors) {
        const r = colors[3 * p], g = colors[3 * p + 1], b = colors[3 * p + 2];
        if (![r, g, b].every(v => Number.isFinite(v) && v >= 0 && v <= 1)) throw new RangeError('Liquid colors must be linear RGB in[0,1]');
        color.setRGB(r, g, b);
      } else color.copy(teal);
      mesh.addBall(x / N, y / N, z / N, strength, subtract, color);
      addPaletteWeight(x / N, y / N, z / N); count++;
    }
    for (let i = 0; i < paletteWeights.length; i++) if (paletteWeights[i] > 0) {
      for (let d = 0; d < 3; d++) mesh.palette[3 * i + d] /= paletteWeights[i];
    }
    if (count) mesh.update(); else mesh.count = 0;
    const triangles = mesh.count / 3, overflow = triangles > maxPolyCount;
    if (overflow) mesh.geometry.setDrawRange(0, 0); // Never expose an out-of-range GPU draw.
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    if (!overflow) for (let vertex = 0; vertex < mesh.count; vertex++) for (let d = 0; d < 3; d++) {
      const value = mesh.positionArray[3 * vertex + d] * N / 2 + N / 2;
      if (!Number.isFinite(value)) throw new Error('Nonfinite marching-cubes output');
      min[d] = Math.min(min[d], value); max[d] = Math.max(max[d], value);
    }
    const axisExpansion = count && mesh.count && !overflow ? Math.max(0, ...min.map((v, d) => particleMin[d] - v), ...max.map((v, d) => v - particleMax[d])) : 0;
    const elapsed = performance.now() - start;
    Object.assign(stats, { updates: stats.updates + 1, fluidParticles: count, triangles, geometryOverflow: overflow,
      clippedKernelParticles: clipped, elapsedMs: elapsed, maximumElapsedMs: Math.max(stats.maximumElapsedMs, elapsed),
      worldBounds: mesh.count && !overflow ? { min, max } : null, particleBounds: count ? { min: particleMin, max: particleMax } : null,
      maximumAxisExpansion: axisExpansion });
    mesh.visible = count > 0 && mesh.count > 0 && !overflow;
    return stats;
  }
  function dispose() {
    if (stats.disposed) return;
    stats.disposed = true; mesh.visible = false; mesh.removeFromParent(); mesh.geometry.dispose(); material.dispose();
  }
  return { mesh, update, dispose, stats };
}
