import { sha256CanonicalUint32 } from '../validation/membership-digests.js';

export const DEPTH_BIN_COUNT = 8;
export const HIDDEN_DEPTH_BIN = DEPTH_BIN_COUNT;
export const DEPTH_ORDER_FRONT_TO_BACK = 'front-to-back';
export const DEPTH_ORDER_REVERSE = 'reverse';

export const DEPTH_ORDERS = Object.freeze([
  DEPTH_ORDER_FRONT_TO_BACK,
  DEPTH_ORDER_REVERSE,
]);

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer.`);
  }
}

function requireDepthOrder(order) {
  if (!DEPTH_ORDERS.includes(order)) {
    throw new RangeError(`depth order must be one of: ${DEPTH_ORDERS.join(', ')}.`);
  }
}

export function validateDepthRange(range) {
  if (!range || !Number.isFinite(range.near) || !Number.isFinite(range.far)) {
    throw new TypeError('depth range must provide finite near and far values.');
  }
  if (range.far <= range.near) {
    throw new RangeError('depth range far must be greater than near.');
  }
  return { near: range.near, far: range.far };
}

export function createDepthBinTraversal(order) {
  requireDepthOrder(order);
  return Uint32Array.from(
    { length: DEPTH_BIN_COUNT },
    (_, index) => (
      order === DEPTH_ORDER_FRONT_TO_BACK
        ? index
        : DEPTH_BIN_COUNT - 1 - index
    ),
  );
}

export function depthBinForViewDepth(viewDepth, range) {
  if (!Number.isFinite(viewDepth)) throw new TypeError('viewDepth must be finite.');
  const { near, far } = validateDepthRange(range);
  const normalized = Math.min(1, Math.max(0, (viewDepth - near) / (far - near)));
  return Math.min(DEPTH_BIN_COUNT - 1, Math.floor(normalized * DEPTH_BIN_COUNT));
}

/**
 * Computes the offset of each physical depth bin within each geometry bucket.
 * Counts are bucket-major: bucket * DEPTH_BIN_COUNT + physicalDepthBin.
 */
export function createOrderedDepthBinLayout(binCounts, order) {
  if (!(binCounts instanceof Uint32Array)) {
    throw new TypeError('binCounts must be a Uint32Array.');
  }
  if (binCounts.length === 0 || binCounts.length % DEPTH_BIN_COUNT !== 0) {
    throw new RangeError(`binCounts length must be a positive multiple of ${DEPTH_BIN_COUNT}.`);
  }
  const traversal = createDepthBinTraversal(order);
  const bucketCount = binCounts.length / DEPTH_BIN_COUNT;
  const starts = new Uint32Array(binCounts.length);
  const totals = new Uint32Array(bucketCount);

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    let cursor = 0;
    for (const physicalBin of traversal) {
      const index = bucket * DEPTH_BIN_COUNT + physicalBin;
      starts[index] = cursor;
      cursor += binCounts[index];
      if (cursor > 0xffff_ffff) {
        throw new RangeError(`bucket ${bucket} depth-bin counts exceed uint32 capacity.`);
      }
    }
    totals[bucket] = cursor;
  }

  return { bucketCount, traversal, starts, totals };
}

/**
 * Commits to the unsorted object-ID sequence inside every physical depth bin.
 * Bin blocks are normalized to bucket-major physical-bin order, so front and
 * reverse traversal snapshots have the same commitment only when their
 * otherwise unconstrained atomic scatter order is also identical.
 */
export async function createPhysicalDepthBinSequenceCommitment({
  actualIds,
  binCounts,
  binStarts,
  bucketBases,
  bucketCapacities,
}) {
  for (const [label, value] of Object.entries({
    actualIds,
    binCounts,
    binStarts,
    bucketBases,
    bucketCapacities,
  })) {
    if (!(value instanceof Uint32Array)) {
      throw new TypeError(`${label} must be a Uint32Array.`);
    }
  }
  if (bucketBases.length === 0
    || bucketCapacities.length !== bucketBases.length
    || binCounts.length !== bucketBases.length * DEPTH_BIN_COUNT
    || binStarts.length !== binCounts.length) {
    throw new RangeError('Depth-bin sequence arrays have inconsistent lengths.');
  }

  let visibleCount = 0;
  for (const count of binCounts) {
    visibleCount += count;
    if (!Number.isSafeInteger(visibleCount)) {
      throw new RangeError('Depth-bin sequence length exceeds safe integer capacity.');
    }
  }
  const headerWords = 3;
  const recordWords = binCounts.length * 3;
  const canonical = new Uint32Array(headerWords + recordWords + visibleCount);
  canonical[0] = bucketBases.length;
  canonical[1] = DEPTH_BIN_COUNT;
  canonical[2] = visibleCount;
  let cursor = headerWords;

  for (let bucket = 0; bucket < bucketBases.length; bucket += 1) {
    for (let physicalBin = 0; physicalBin < DEPTH_BIN_COUNT; physicalBin += 1) {
      const binIndex = bucket * DEPTH_BIN_COUNT + physicalBin;
      const count = binCounts[binIndex];
      const start = bucketBases[bucket] + binStarts[binIndex];
      const end = start + count;
      const bucketEnd = bucketBases[bucket] + bucketCapacities[bucket];
      if (!Number.isSafeInteger(end)
        || !Number.isSafeInteger(bucketEnd)
        || end > bucketEnd
        || end > actualIds.length) {
        throw new RangeError(`bucket ${bucket} bin ${physicalBin} sequence exceeds readback capacity.`);
      }
      canonical[cursor] = bucket;
      canonical[cursor + 1] = physicalBin;
      canonical[cursor + 2] = count;
      cursor += 3;
      canonical.set(actualIds.subarray(start, end), cursor);
      cursor += count;
    }
  }
  if (cursor !== canonical.length) {
    throw new Error('Depth-bin sequence commitment length is inconsistent.');
  }

  return {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    encoding: 'bucket-major-physical-bin-major-tagged-uint32-little-endian',
    bucketCount: bucketBases.length,
    binCount: DEPTH_BIN_COUNT,
    recordCount: binCounts.length,
    survivorCount: visibleCount,
    sha256: await sha256CanonicalUint32(canonical),
  };
}

/**
 * Builds an independent CPU expectation for the GPU's sphere-nearest view-depth
 * classifier. The nearest-surface key is center view depth minus world radius.
 * Matrix elements use Three.js' column-major Matrix4 representation.
 */
export function createExpectedObjectDepthBins({
  bounds,
  objectCount,
  expectedIds,
  viewMatrixElements,
  depthRange,
}) {
  requirePositiveInteger(objectCount, 'objectCount');
  if (!(bounds instanceof Float32Array) || bounds.length !== objectCount * 4) {
    throw new RangeError('bounds must contain one Float32 vec4 per object.');
  }
  if (!(expectedIds instanceof Uint32Array)) {
    throw new TypeError('expectedIds must be a Uint32Array.');
  }
  if (!viewMatrixElements || viewMatrixElements.length !== 16) {
    throw new RangeError('viewMatrixElements must contain 16 values.');
  }
  const range = validateDepthRange(depthRange);
  const expectedBins = new Uint32Array(objectCount);
  expectedBins.fill(HIDDEN_DEPTH_BIN);
  const seen = new Uint8Array(objectCount);

  for (const objectId of expectedIds) {
    if (objectId >= objectCount) {
      throw new RangeError(`expected object ID ${objectId} is out of range.`);
    }
    if (seen[objectId]) throw new RangeError(`expected object ID ${objectId} is duplicated.`);
    seen[objectId] = 1;
    const offset = objectId * 4;
    const x = bounds[offset];
    const y = bounds[offset + 1];
    const z = bounds[offset + 2];
    const viewZ = viewMatrixElements[2] * x
      + viewMatrixElements[6] * y
      + viewMatrixElements[10] * z
      + viewMatrixElements[14];
    const nearestViewDepth = -viewZ - bounds[offset + 3];
    expectedBins[objectId] = depthBinForViewDepth(nearestViewDepth, range);
  }
  return expectedBins;
}

/**
 * Validates classifier output, per-bin counts/prefixes, and survivor block order.
 * Atomic ordering inside a bin is intentionally unconstrained.
 */
export function validateDepthBinReadback({
  actualIds,
  objectBins,
  binCounts,
  binStarts,
  commandCounts,
  expectedObjectBins,
  objectBuckets,
  bucketBases,
  bucketCapacities,
  order,
}) {
  const arrays = {
    actualIds,
    objectBins,
    binCounts,
    binStarts,
    commandCounts,
    expectedObjectBins,
    objectBuckets,
    bucketBases,
    bucketCapacities,
  };
  for (const [label, value] of Object.entries(arrays)) {
    if (!(value instanceof Uint32Array)) {
      throw new TypeError(`${label} must be a Uint32Array.`);
    }
  }
  const objectCount = objectBins.length;
  const bucketCount = bucketCapacities.length;
  if (objectCount === 0 || actualIds.length < objectCount) {
    throw new RangeError('actualIds must have capacity for every object.');
  }
  if (expectedObjectBins.length !== objectCount || objectBuckets.length !== objectCount) {
    throw new RangeError('object-bin arrays must match objectBins length.');
  }
  if (bucketBases.length !== bucketCount || commandCounts.length !== bucketCount) {
    throw new RangeError('bucket arrays must have equal lengths.');
  }
  if (binCounts.length !== bucketCount * DEPTH_BIN_COUNT
    || binStarts.length !== binCounts.length) {
    throw new RangeError('bin arrays must contain eight entries per bucket.');
  }

  const expectedCounts = new Uint32Array(binCounts.length);
  const errors = [];
  for (let objectId = 0; objectId < objectCount; objectId += 1) {
    const expectedBin = expectedObjectBins[objectId];
    const actualBin = objectBins[objectId];
    if (actualBin !== expectedBin) {
      errors.push(`object ${objectId}: classifier bin ${actualBin}, expected ${expectedBin}`);
    }
    if (expectedBin < DEPTH_BIN_COUNT) {
      const bucket = objectBuckets[objectId];
      if (bucket >= bucketCount) {
        errors.push(`object ${objectId}: bucket ${bucket} is out of range`);
      } else {
        expectedCounts[bucket * DEPTH_BIN_COUNT + expectedBin] += 1;
      }
    } else if (expectedBin !== HIDDEN_DEPTH_BIN) {
      errors.push(`object ${objectId}: invalid expected bin ${expectedBin}`);
    }
  }

  for (let index = 0; index < binCounts.length; index += 1) {
    if (binCounts[index] !== expectedCounts[index]) {
      const bucket = Math.floor(index / DEPTH_BIN_COUNT);
      const bin = index % DEPTH_BIN_COUNT;
      errors.push(
        `bucket ${bucket} bin ${bin}: count ${binCounts[index]}, expected ${expectedCounts[index]}`,
      );
    }
  }

  const expectedLayout = createOrderedDepthBinLayout(expectedCounts, order);
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    if (commandCounts[bucket] !== expectedLayout.totals[bucket]) {
      errors.push(
        `bucket ${bucket}: command count ${commandCounts[bucket]}, expected ${expectedLayout.totals[bucket]}`,
      );
    }
    if (expectedLayout.totals[bucket] > bucketCapacities[bucket]) {
      errors.push(
        `bucket ${bucket}: ${expectedLayout.totals[bucket]} survivors exceed capacity ${bucketCapacities[bucket]}`,
      );
    }
  }

  for (let index = 0; index < binStarts.length; index += 1) {
    if (binStarts[index] !== expectedLayout.starts[index]) {
      const bucket = Math.floor(index / DEPTH_BIN_COUNT);
      const bin = index % DEPTH_BIN_COUNT;
      errors.push(
        `bucket ${bucket} bin ${bin}: start ${binStarts[index]}, expected ${expectedLayout.starts[index]}`,
      );
    }
  }

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const bucketBase = bucketBases[bucket];
    for (const physicalBin of expectedLayout.traversal) {
      const binIndex = bucket * DEPTH_BIN_COUNT + physicalBin;
      const start = bucketBase + expectedLayout.starts[binIndex];
      const end = start + expectedCounts[binIndex];
      if (end > bucketBase + bucketCapacities[bucket] || end > actualIds.length) {
        errors.push(`bucket ${bucket} bin ${physicalBin}: survivor block exceeds capacity`);
        continue;
      }
      for (let cursor = start; cursor < end; cursor += 1) {
        const objectId = actualIds[cursor];
        if (objectId >= objectCount) {
          errors.push(`bucket ${bucket} bin ${physicalBin}: object ID ${objectId} is out of range`);
        } else if (objectBuckets[objectId] !== bucket) {
          errors.push(`bucket ${bucket} bin ${physicalBin}: object ${objectId} belongs to another bucket`);
        } else if (expectedObjectBins[objectId] !== physicalBin) {
          errors.push(
            `bucket ${bucket} bin ${physicalBin}: object ${objectId} belongs to bin ${expectedObjectBins[objectId]}`,
          );
        }
      }
    }
  }

  return {
    pass: errors.length === 0,
    errors,
    binCount: DEPTH_BIN_COUNT,
    order,
    traversal: Array.from(expectedLayout.traversal),
    expectedCounts: Array.from(expectedCounts),
    actualCounts: Array.from(binCounts),
    expectedStarts: Array.from(expectedLayout.starts),
    actualStarts: Array.from(binStarts),
    expectedBucketTotals: Array.from(expectedLayout.totals),
    commandCounts: Array.from(commandCounts),
  };
}
