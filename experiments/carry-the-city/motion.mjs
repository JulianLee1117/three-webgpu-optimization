/**
 * Sampled motion for Carry the City. No renderer, physics engine or model call.
 *
 * Clips contain normalized source marker POSITIONS, not a target animation.
 * The generic evaluator sweeps a target cross-section along their sampled spine.
 * The two small authored sources below are deliberately explicit about provenance.
 */

const TAU = Math.PI * 2;
const UP = [0, 1, 0];
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = (a) => {
  const length = Math.hypot(...a);
  if (!(length > 1e-10) || !Number.isFinite(length)) throw new RangeError('Degenerate sampled motion frame');
  return mul(a, 1 / length);
};

function vector(value, label = 'point') {
  const p = Array.isArray(value) || ArrayBuffer.isView(value) ? value : [value?.x, value?.y, value?.z];
  if (p.length !== 3 || !p.every(Number.isFinite)) throw new TypeError(`${label} must have three finite coordinates`);
  return p;
}

function validateClip(clip) {
  if (!clip || !['x', 'z'].includes(clip.axis) || !(clip.duration > 0) || !Number.isFinite(clip.duration)
    || !(clip.extent > 0) || !Number.isFinite(clip.extent)
    || !Number.isInteger(clip.frames) || clip.frames < 4 || clip.frames > 512
    || !Number.isInteger(clip.markers) || clip.markers < 4 || clip.markers > 513
    || !(clip.data instanceof Float32Array) || clip.data.length !== clip.frames * clip.markers * 3
    || !(clip.referencePositions instanceof Float32Array) || clip.referencePositions.length !== clip.markers * 3) {
    throw new TypeError('Invalid bounded marker clip');
  }
  vector(clip.scale, 'clip.scale');
  if (clip.scale.some((v) => v <= 0)) throw new RangeError('Clip scale must be positive');
}

function settings(time, strength) {
  if (!Number.isFinite(time)) throw new TypeError('Time must be finite seconds');
  if (!Number.isFinite(strength) || strength < 0 || strength > 1) throw new RangeError('Strength must be between zero and one');
}

/**
 * Capture a bounded loop from authored source markers. There is no duplicated
 * final frame: interpolation wraps the final sample back to the first sample.
 * Source u is evenly spaced in [-1, 1]; sample(u, time) returns local position.
 */
export function captureMarkerLoop({ id, duration, frames = 64, markers = 17, axis, extent, scale, sample }) {
  if (typeof sample !== 'function') throw new TypeError('A source marker sampler is required');
  if (!Number.isInteger(frames) || frames < 4 || frames > 512 || !Number.isInteger(markers) || markers < 4 || markers > 513) {
    throw new RangeError('Capture is bounded to 4–512 frames and 4–513 markers');
  }
  const clip = {
    id: String(id), duration, frames, markers, axis, extent,
    scale: Array.from(vector(scale, 'scale')),
    data: new Float32Array(frames * markers * 3),
    referencePositions: new Float32Array(markers * 3),
    sourceInfo: { type: 'authored-marker-samples', inferred: false, sampling: 'uniform-loop', coordinateRange: [-1, 1] },
  };
  validateClip(clip);
  const component = axis === 'x' ? 0 : 2;
  for (let marker = 0; marker < markers; marker++) {
    const u = 2 * marker / (markers - 1) - 1;
    clip.referencePositions[marker * 3 + component] = u;
    for (let frame = 0; frame < frames; frame++) {
      const position = vector(sample(u, frame * duration / frames), 'source marker');
      const offset = (frame * markers + marker) * 3;
      clip.data.set(position, offset);
      if (!clip.data.subarray(offset, offset + 3).every(Number.isFinite)) throw new RangeError('Source position exceeds Float32 range');
    }
  }
  return clip;
}

/** Make one of the two visible, authored donors used by the demonstration. */
export function captureClip({ kind = 'swim', frames = 64, markers = 17 } = {}) {
  if (kind === 'swim') {
    const duration = 2.4;
    return captureMarkerLoop({
      id: 'koi-swim', duration, frames, markers, axis: 'x', extent: 4, scale: [4, 2, 6.25],
      sample: (u, time) => {
        const phase = time / duration * TAU;
        const gain = 0.3 + 0.7 * (u + 1) / 2;
        // Longer spatial wavelength keeps the city's 3.6-unit-wide sweep
        // from folding at the inside of a bend. Tail amplitude remains 1 unit.
        return [u, 0.04 * Math.sin(phase + u * 2.5) * gain, 0.16 * Math.sin(phase + u * 2.5) * gain];
      },
    });
  }
  if (kind === 'fly') {
    const duration = 2;
    return captureMarkerLoop({
      id: 'moth-flight', duration, frames, markers, axis: 'z', extent: 1.8, scale: [4, 1.8, 1.8],
      sample: (u, time) => {
        const core = 0.28;
        const wing = Math.max(0, Math.abs(u) - core);
        const angle = 0.85 * Math.sin(time / duration * TAU);
        return [0, wing * Math.sin(angle), Math.sign(u) * (Math.min(Math.abs(u), core) + wing * Math.cos(angle))];
      },
    });
  }
  throw new RangeError(`Unknown authored motion source: ${kind}`);
}

// Cubic interpolation preserves a straight rest spine exactly. Quadratic
// extrapolation of endpoint neighbors retains curvature without an artificial
// spike in the last marker interval.
function cubicAt(read, count, coordinate) {
  const x = clamp(coordinate, 0, count - 1);
  const i = Math.min(count - 2, Math.floor(x));
  const t = x - i;
  const p1 = read(i), p2 = read(i + 1);
  const p0 = i > 0 ? read(i - 1) : 3 * p1 - 3 * p2 + read(i + 2);
  const p3 = i + 2 < count ? read(i + 2) : 3 * p2 - 3 * p1 + read(i - 1);
  return p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)));
}

function sampled(clip, time, u, reference, out) {
  const coordinate = (clamp(u, -1, 1) + 1) * 0.5 * (clip.markers - 1);
  const cycle = ((time % clip.duration) + clip.duration) % clip.duration / clip.duration * clip.frames;
  const frame = Math.floor(cycle), next = (frame + 1) % clip.frames, blend = cycle - frame;
  for (let c = 0; c < 3; c++) {
    out[c] = cubicAt((marker) => {
      if (reference) return clip.referencePositions[marker * 3 + c];
      const a = clip.data[(frame * clip.markers + marker) * 3 + c];
      const b = clip.data[(next * clip.markers + marker) * 3 + c];
      return a + (b - a) * blend;
    }, clip.markers, coordinate);
  }
  if (!out.every(Number.isFinite)) throw new RangeError('Sampled source contains non-finite coordinates');
  return out;
}

/** Sample the SOURCE in normalized coordinates, independent of target scale. */
export function interpolateClipPoint(clip, time, u, out = [0, 0, 0]) {
  validateClip(clip);
  settings(time, 1);
  if (!Number.isFinite(u)) throw new TypeError('Source coordinate must be finite');
  return sampled(clip, time, u, false, out);
}

function center(clip, time, coordinate, strength) {
  const u = coordinate / clip.extent;
  const source = sampled(clip, time, u, false, [0, 0, 0]);
  const rest = sampled(clip, time, u, true, [0, 0, 0]);
  const p = source.map((x, c) => (x - rest[c]) * clip.scale[c] * strength);
  p[clip.axis === 'x' ? 0 : 2] += coordinate;
  return p;
}

function spineFrame(clip, time, coordinate, strength) {
  const h = clip.extent * 1e-4;
  // One-sided differences at the domain ends retain the sampled end tangent.
  const a = clamp(coordinate - h, -clip.extent, clip.extent);
  const b = clamp(coordinate + h, -clip.extent, clip.extent);
  const ca = center(clip, time, a, strength), cb = center(clip, time, b, strength);
  const tangent = normalize(cb.map((x, c) => x - ca[c]));
  if (Math.abs(dot(tangent, UP)) > 0.98) throw new RangeError('Sampled spine is too vertical for the supported sweep frame');
  if (clip.axis === 'x') {
    const tangentZ = normalize(cross(tangent, UP));
    return { tangentX: tangent, tangentZ, normal: normalize(cross(tangentZ, tangent)) };
  }
  const tangentX = normalize(cross(UP, tangent));
  return { tangentX, tangentZ: tangent, normal: normalize(cross(tangent, tangentX)) };
}

function pointUnchecked(clip, time, p, strength, out) {
  if (strength === 0) {
    for (let c = 0; c < 3; c++) out[c] = p[c];
    return out;
  }
  const axis = clip.axis === 'x' ? 0 : 2;
  const coordinate = clamp(p[axis], -clip.extent, clip.extent);
  const basis = spineFrame(clip, time, coordinate, strength);
  const along = clip.axis === 'x' ? basis.tangentX : basis.tangentZ;
  const across = clip.axis === 'x' ? basis.tangentZ : basis.tangentX;
  const q = add(add(center(clip, time, coordinate, strength), mul(along, p[axis] - coordinate)),
    add(mul(across, p[axis === 0 ? 2 : 0]), mul(basis.normal, p[1])));
  for (let c = 0; c < 3; c++) out[c] = q[c];
  return out;
}

/**
 * Deform a target-space point. Times are seconds; strength is [0,1]. Outside
 * the captured axis domain the endpoint's frame extends rigidly. y is height
 * above the local surface, so rendered meshes and attachments share a mapping.
 */
export function samplePoint(clip, time, point, { strength = 1, out = [0, 0, 0] } = {}) {
  validateClip(clip);
  settings(time, strength);
  return pointUnchecked(clip, time, vector(point), strength, out);
}

/**
 * A local surface frame and material-point velocity from the SAME point map.
 * tangentX and tangentZ are orthonormal; normal points up for the supported
 * authored clips. This is kinematic deformation, not a force or thrust solver.
 */
export function sampleFrame(clip, time, point, { strength = 1, epsilon = 1e-3 } = {}) {
  validateClip(clip);
  settings(time, strength);
  const p = Array.from(vector(point));
  if (!Number.isFinite(epsilon) || epsilon < 1e-6 || epsilon > 0.02) throw new RangeError('Frame epsilon must be in [1e-6, .02]');
  const derivative = (axis) => {
    const a = p.slice(), b = p.slice();
    a[axis] -= epsilon;
    b[axis] += epsilon;
    const pa = pointUnchecked(clip, time, a, strength, [0, 0, 0]);
    const pb = pointUnchecked(clip, time, b, strength, [0, 0, 0]);
    return pb.map((x, c) => (x - pa[c]) / (2 * epsilon));
  };
  const dx = derivative(0), dz = derivative(2);
  const tangentX = normalize(dx);
  const normal = normalize(cross(dz, dx));
  const tangentZ = normalize(cross(tangentX, normal));
  const before = pointUnchecked(clip, time - epsilon, p, strength, [0, 0, 0]);
  const after = pointUnchecked(clip, time + epsilon, p, strength, [0, 0, 0]);
  return {
    position: pointUnchecked(clip, time, p, strength, [0, 0, 0]),
    tangentX, tangentZ, normal,
    velocity: after.map((x, c) => (x - before[c]) / (2 * epsilon)),
  };
}

export const evaluateFrame = sampleFrame;
