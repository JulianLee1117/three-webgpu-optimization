/**
 * Independent forward validation and browser preview of the solved heightfield.
 * This module does not import the optimization forward map or its derivatives.
 * The beam remains parallel to +z and enters the flat base normally. Only the
 * receiver plane's distance may change; rotating lights/lenses is unsupported.
 */

function validateSurface(surface) {
  if (!surface || !Number.isInteger(surface.grid) || surface.grid < 3 || surface.grid > 49
    || (!(surface.height instanceof Float64Array) && !(surface.height instanceof Float32Array))
    || surface.height.length !== surface.grid * surface.grid
    || !Number.isFinite(surface.extent ?? 1) || (surface.extent ?? 1) <= 0 || (surface.extent ?? 1) > 10) throw new RangeError('Invalid bounded heightfield');
}

function heightAt(surface, x, y) {
  const extent = surface.extent ?? 1;
  const gx = (x / extent + 1) * (surface.grid - 1) / 2, gy = (y / extent + 1) * (surface.grid - 1) / 2;
  const ix = Math.min(surface.grid - 2, Math.max(0, Math.floor(gx))), iy = Math.min(surface.grid - 2, Math.max(0, Math.floor(gy)));
  const a = gx - ix, b = gy - iy, h = surface.height;
  const low = h[iy * surface.grid + ix] * (1 - a) + h[iy * surface.grid + ix + 1] * a;
  const high = h[(iy + 1) * surface.grid + ix] * (1 - a) + h[(iy + 1) * surface.grid + ix + 1] * a;
  const value = low * (1 - b) + high * b;
  if (!Number.isFinite(value)) throw new RangeError('Non-finite heightfield sample');
  return value;
}

function validatePoint(surface, x, y) {
  const extent = surface.extent ?? 1;
  if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > extent + 1e-10 || Math.abs(y) > extent + 1e-10) throw new RangeError('Point lies outside the lens aperture');
}

/** x/y are analytical aperture coordinates; returned height is analytical z. */
export function sampleHeight(surface, x, y) {
  validateSurface(surface);
  validatePoint(surface, x, y);
  return heightAt(surface, x, y);
}

function surfaceAt(surface, x, y, epsilon) {
  const extent = surface.extent ?? 1;
  const left = Math.max(-extent, x - epsilon), right = Math.min(extent, x + epsilon);
  const bottom = Math.max(-extent, y - epsilon), top = Math.min(extent, y + epsilon);
  const hx = (heightAt(surface, right, y) - heightAt(surface, left, y)) / (right - left);
  const hy = (heightAt(surface, x, top) - heightAt(surface, x, bottom)) / (top - bottom);
  const length = Math.hypot(hx, hy, 1);
  return { height: heightAt(surface, x, y), gradient: [hx, hy], normal: [-hx / length, -hy / length, 1 / length] };
}

/** Outward EXIT normal, pointing toward the +z receiver for a flat slab. */
export function sampleSurface(surface, x, y, { epsilon = (surface.extent ?? 1) * 1e-5 } = {}) {
  validateSurface(surface);
  validatePoint(surface, x, y);
  if (!Number.isFinite(epsilon) || epsilon <= 0 || epsilon > (surface.extent ?? 1) * 0.01) throw new RangeError('Invalid normal difference step');
  return surfaceAt(surface, x, y, epsilon);
}

/**
 * Uniform rays are stratified across the whole aperture. Their outgoing
 * directions follow the standard vector refraction construction, using normals
 * found by finite differences of the actual bilinear surface.
 *
 * weights are transmission FRACTIONS per photon. energy assumes one unit of
 * incoming irradiance per unit aperture area; reflected light is accounted for
 * but is not traced through additional internal bounces. No absorption or wave
 * optics is modeled. TIR rays have zero weight and NaN hit coordinates.
 */
export function independentSnellTrace(surface, {
  resolution = 144,
  receiverDistance = surface.distance,
  offsetX = 0.37,
  offsetY = 0.63,
  fresnel = true,
} = {}) {
  validateSurface(surface);
  if (!Number.isInteger(resolution) || resolution < 8 || resolution > 192) throw new RangeError('Trace resolution must be in [8,192]');
  if (![offsetX, offsetY].every((v) => Number.isFinite(v) && v > 0 && v < 1)) throw new RangeError('Ray offsets must lie inside their pixel cells');
  if (!Number.isFinite(receiverDistance) || receiverDistance <= 0 || receiverDistance > 100) throw new RangeError('Receiver distance must be in (0,100]');
  if (!Number.isFinite(surface.ior) || surface.ior < 1 || surface.ior > 2) throw new RangeError('Refractive index must be in [1,2]');
  if (typeof fresnel !== 'boolean') throw new TypeError('fresnel must be a boolean');
  if (!surface.height.every((v) => Number.isFinite(v) && v < receiverDistance && (surface.baseZ === undefined || v > surface.baseZ))) throw new RangeError('Lens surface is non-finite or intersects its base/receiver');
  const extent = surface.extent ?? 1, count = resolution * resolution;
  const hits = new Float64Array(count * 2), weights = new Float64Array(count), epsilon = extent * 1e-5;
  const entryReflection = fresnel ? ((surface.ior - 1) / (surface.ior + 1)) ** 2 : 0;
  let maxSlope = 0, lost = 0, transmittedFractionSum = 0, reflectedFractionSum = 0;
  for (let row = 0; row < resolution; row++) for (let col = 0; col < resolution; col++) {
    const x = extent * (-1 + 2 * (col + offsetX) / resolution), y = extent * (-1 + 2 * (row + offsetY) / resolution);
    const sampled = surfaceAt(surface, x, y, epsilon), index = row * resolution + col;
    maxSlope = Math.max(maxSlope, Math.hypot(...sampled.gradient));
    // Refraction's interface normal points against the incident +z ray.
    const normal = sampled.normal.map((v) => -v), cosine = -normal[2];
    const transmittedCosineSquared = 1 - surface.ior * surface.ior * (1 - cosine * cosine);
    if (transmittedCosineSquared < 0) {
      hits[2 * index] = hits[2 * index + 1] = NaN;
      weights[index] = 0; reflectedFractionSum += 1; lost++;
      continue;
    }
    const transmittedCosine = Math.sqrt(transmittedCosineSquared), correction = surface.ior * cosine - transmittedCosine;
    const direction = [correction * normal[0], correction * normal[1], surface.ior + correction * normal[2]];
    if (!(direction[2] > 0)) throw new Error('Transmitted ray does not reach the supported forward receiver');
    const length = (receiverDistance - sampled.height) / direction[2];
    hits[2 * index] = x + length * direction[0];
    hits[2 * index + 1] = y + length * direction[1];
    let exitReflection = 0;
    if (fresnel) {
      const rs = (surface.ior * cosine - transmittedCosine) / (surface.ior * cosine + transmittedCosine);
      const rp = (cosine - surface.ior * transmittedCosine) / (cosine + surface.ior * transmittedCosine);
      exitReflection = (rs * rs + rp * rp) / 2;
    }
    const transmission = (1 - entryReflection) * (1 - exitReflection);
    weights[index] = transmission;
    transmittedFractionSum += transmission;
    reflectedFractionSum += 1 - transmission;
  }
  const incident = 4 * extent * extent, photonFlux = incident / count;
  const transmitted = transmittedFractionSum * photonFlux, reflected = reflectedFractionSum * photonFlux;
  return {
    hits, weights, maxSlope, lost, resolution, receiverDistance, fresnel,
    energy: { incident, transmitted, reflected, conservationError: incident - transmitted - reflected },
  };
}
