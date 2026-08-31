import {
  Euler,
  Matrix4,
  Quaternion,
  Vector3,
} from 'three';
import { mulberry32 } from '../lib/random.js';

export const SCENARIO_LAYOUTS = Object.freeze([
  'baseline',
  'low-overlap',
  'high-overlap',
]);

const LAYOUT_EXTENTS = Object.freeze({
  baseline: Object.freeze({ x: 30, y: 18 }),
  'low-overlap': Object.freeze({ x: 75, y: 40 }),
  'high-overlap': Object.freeze({ x: 1.5, y: 0.9 }),
});

function assertPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer.`);
  }
}

export function allocateBalancedCounts(total, bucketCount) {
  assertPositiveInteger(total, 'total');
  assertPositiveInteger(bucketCount, 'bucketCount');
  if (bucketCount > total) throw new RangeError('bucketCount cannot exceed total.');

  const counts = new Uint32Array(bucketCount);
  const quotient = Math.floor(total / bucketCount);
  const remainder = total % bucketCount;
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    counts[bucket] = quotient + (bucket < remainder ? 1 : 0);
  }
  return counts;
}

export function prefixBases(counts) {
  const bases = new Uint32Array(counts.length);
  let cursor = 0;
  for (let index = 0; index < counts.length; index += 1) {
    bases[index] = cursor;
    cursor += counts[index];
  }
  return bases;
}

export function allocateVisibleCounts(counts, fraction) {
  if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1) {
    throw new RangeError('fraction must be between 0 and 1.');
  }

  const total = counts.reduce((sum, count) => sum + count, 0);
  const target = Math.round(total * fraction);
  const visible = new Uint32Array(counts.length);
  const remainders = [];
  let assigned = 0;

  for (let bucket = 0; bucket < counts.length; bucket += 1) {
    const exact = counts[bucket] * fraction;
    const floor = Math.floor(exact);
    visible[bucket] = floor;
    assigned += floor;
    remainders.push({ bucket, remainder: exact - floor });
  }

  remainders.sort((left, right) => right.remainder - left.remainder || left.bucket - right.bucket);
  for (let index = 0; index < target - assigned; index += 1) {
    visible[remainders[index].bucket] += 1;
  }
  return visible;
}

function requireGeometrySphere(sphere, bucket) {
  if (!sphere || !sphere.center || !Number.isFinite(sphere.radius) || sphere.radius <= 0) {
    throw new TypeError(`geometrySpheres[${bucket}] must be a finite positive sphere.`);
  }
}

/**
 * Creates nested visibility subsets without changing the projection matrix.
 * IDs that remain visible between two fractions retain identical transforms.
 */
export function createFixedSubsetScenario({
  objectCount,
  bucketCount,
  visibilityFraction,
  geometrySpheres,
  seed,
  layout = 'baseline',
}) {
  assertPositiveInteger(objectCount, 'objectCount');
  assertPositiveInteger(bucketCount, 'bucketCount');
  if (geometrySpheres.length !== bucketCount) {
    throw new RangeError('geometrySpheres length must equal bucketCount.');
  }
  if (!SCENARIO_LAYOUTS.includes(layout)) {
    throw new RangeError(`layout must be one of: ${SCENARIO_LAYOUTS.join(', ')}.`);
  }

  const layoutExtents = LAYOUT_EXTENTS[layout];

  const bucketCounts = allocateBalancedCounts(objectCount, bucketCount);
  const bucketBases = prefixBases(bucketCounts);
  const visibleCounts = allocateVisibleCounts(bucketCounts, visibilityFraction);
  const expectedVisibleCount = visibleCounts.reduce((sum, count) => sum + count, 0);
  const expectedVisibleIds = new Uint32Array(expectedVisibleCount);
  const objectBuckets = new Uint32Array(objectCount);
  const matrices = new Float32Array(objectCount * 16);
  const bounds = new Float32Array(objectCount * 4);
  const random = mulberry32(seed);

  const position = new Vector3();
  const scale = new Vector3();
  const quaternion = new Quaternion();
  const euler = new Euler();
  const matrix = new Matrix4();
  const worldCenter = new Vector3();
  let visibleCursor = 0;

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const localSphere = geometrySpheres[bucket];
    requireGeometrySphere(localSphere, bucket);
    const base = bucketBases[bucket];
    const count = bucketCounts[bucket];
    const visibleCount = visibleCounts[bucket];

    for (let localIndex = 0; localIndex < count; localIndex += 1) {
      const objectId = base + localIndex;
      const isVisible = localIndex < visibleCount;
      objectBuckets[objectId] = bucket;

      const randomX = random() * 2 - 1;
      const randomY = random() * 2 - 1;
      const inViewX = randomX * layoutExtents.x;
      const inViewY = randomY * layoutExtents.y;
      const hiddenY = randomY * LAYOUT_EXTENTS.baseline.y;
      const inViewZ = (random() * 2 - 1) * 28;
      position.set(
        isVisible ? inViewX : 10_000 + objectId * 0.01,
        isVisible ? inViewY : hiddenY,
        inViewZ,
      );

      euler.set(
        random() * Math.PI * 2,
        random() * Math.PI * 2,
        random() * Math.PI * 2,
      );
      quaternion.setFromEuler(euler);
      scale.set(
        0.55 + random() * 0.9,
        0.55 + random() * 0.9,
        0.55 + random() * 0.9,
      );
      matrix.compose(position, quaternion, scale);
      matrix.toArray(matrices, objectId * 16);

      worldCenter.copy(localSphere.center).applyMatrix4(matrix);
      const worldRadius = localSphere.radius * matrix.getMaxScaleOnAxis();
      const boundOffset = objectId * 4;
      bounds[boundOffset] = worldCenter.x;
      bounds[boundOffset + 1] = worldCenter.y;
      bounds[boundOffset + 2] = worldCenter.z;
      bounds[boundOffset + 3] = worldRadius;

      if (isVisible) expectedVisibleIds[visibleCursor++] = objectId;
    }
  }

  return {
    objectCount,
    bucketCount,
    visibilityFraction,
    layout,
    bucketCounts,
    bucketBases,
    visibleCounts,
    objectBuckets,
    matrices,
    bounds,
    expectedVisibleIds,
    expectedVisibleCount,
    cullOrder: Uint32Array.from({ length: objectCount }, (_, index) => index),
  };
}
