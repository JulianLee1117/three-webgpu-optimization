import {
  DEPTH_BIN_COUNT,
  DEPTH_ORDER_FRONT_TO_BACK,
  DEPTH_ORDER_REVERSE,
  createExpectedObjectDepthBins,
  createOrderedDepthBinLayout,
  validateDepthRange,
} from './depth-bin-layout.js';

export const FROZEN_DEPTH_CROSSOVER_LANES = Object.freeze([
  DEPTH_ORDER_FRONT_TO_BACK,
  DEPTH_ORDER_REVERSE,
]);

export const FROZEN_DEPTH_UNUSED_OBJECT_ID = 0xffff_ffff;

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer.`);
  }
}

function requireUint32Array(value, expectedLength, label) {
  if (!(value instanceof Uint32Array) || value.length !== expectedLength) {
    throw new RangeError(`${label} must be a Uint32Array of length ${expectedLength}.`);
  }
}

export function validateFrozenDepthLane(value, label = 'lane') {
  if (!FROZEN_DEPTH_CROSSOVER_LANES.includes(value)) {
    throw new RangeError(
      `${label} must be one of: ${FROZEN_DEPTH_CROSSOVER_LANES.join(', ')}.`,
    );
  }
  return value;
}

function validateScenarioShape(scenario) {
  requirePositiveInteger(scenario?.objectCount, 'scenario.objectCount');
  requirePositiveInteger(scenario?.bucketCount, 'scenario.bucketCount');
  if (scenario.objectCount > 0x7fff_ffff) {
    throw new RangeError('scenario.objectCount is too large for two uint32 lane segments.');
  }

  requireUint32Array(
    scenario.bucketBases,
    scenario.bucketCount,
    'scenario.bucketBases',
  );
  requireUint32Array(
    scenario.bucketCounts,
    scenario.bucketCount,
    'scenario.bucketCounts',
  );
  requireUint32Array(
    scenario.visibleCounts,
    scenario.bucketCount,
    'scenario.visibleCounts',
  );
  requireUint32Array(
    scenario.objectBuckets,
    scenario.objectCount,
    'scenario.objectBuckets',
  );
  if (!(scenario.expectedVisibleIds instanceof Uint32Array)) {
    throw new TypeError('scenario.expectedVisibleIds must be a Uint32Array.');
  }
  if (!(scenario.bounds instanceof Float32Array)
    || scenario.bounds.length !== scenario.objectCount * 4) {
    throw new RangeError('scenario.bounds must contain one Float32 vec4 per object.');
  }

  let objectCursor = 0;
  let expectedVisibleCount = 0;
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    if (scenario.bucketBases[bucket] !== objectCursor) {
      throw new RangeError('Frozen crossover requires contiguous bucket-major objects.');
    }
    const bucketCount = scenario.bucketCounts[bucket];
    if (scenario.visibleCounts[bucket] > bucketCount) {
      throw new RangeError(`scenario.visibleCounts[${bucket}] exceeds bucket capacity.`);
    }
    const bucketEnd = objectCursor + bucketCount;
    if (!Number.isSafeInteger(bucketEnd) || bucketEnd > scenario.objectCount) {
      throw new RangeError(`scenario.bucketCounts[${bucket}] exceeds object capacity.`);
    }
    for (let objectId = objectCursor; objectId < bucketEnd; objectId += 1) {
      if (scenario.objectBuckets[objectId] !== bucket) {
        throw new RangeError(
          `scenario.objectBuckets[${objectId}] does not match its contiguous bucket.`,
        );
      }
    }
    objectCursor = bucketEnd;
    expectedVisibleCount += scenario.visibleCounts[bucket];
  }
  if (objectCursor !== scenario.objectCount) {
    throw new RangeError('scenario.bucketCounts must sum to scenario.objectCount.');
  }
  if (expectedVisibleCount !== scenario.expectedVisibleIds.length
    || (Number.isInteger(scenario.expectedVisibleCount)
      && scenario.expectedVisibleCount !== expectedVisibleCount)) {
    throw new RangeError('scenario visible counts do not match expectedVisibleIds.');
  }

  const seen = new Uint8Array(scenario.objectCount);
  const actualVisibleCounts = new Uint32Array(scenario.bucketCount);
  let previousObjectId = -1;
  for (const objectId of scenario.expectedVisibleIds) {
    if (objectId >= scenario.objectCount) {
      throw new RangeError(`Expected visible object ID ${objectId} is out of range.`);
    }
    if (seen[objectId]) {
      throw new RangeError(`Expected visible object ID ${objectId} is duplicated.`);
    }
    if (objectId <= previousObjectId) {
      throw new RangeError('scenario.expectedVisibleIds must be strictly ascending.');
    }
    seen[objectId] = 1;
    previousObjectId = objectId;
    actualVisibleCounts[scenario.objectBuckets[objectId]] += 1;
  }
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    if (actualVisibleCounts[bucket] !== scenario.visibleCounts[bucket]) {
      throw new RangeError(
        `scenario.expectedVisibleIds count for bucket ${bucket} does not match visibleCounts.`,
      );
    }
  }
}

function validateViewMatrixElements(viewMatrixElements) {
  if (!viewMatrixElements || viewMatrixElements.length !== 16) {
    throw new RangeError('viewMatrixElements must contain 16 values.');
  }
  for (const value of viewMatrixElements) {
    if (!Number.isFinite(value)) {
      throw new TypeError('viewMatrixElements must contain only finite values.');
    }
  }
  return Float32Array.from(viewMatrixElements);
}

function createLaneRecord(order, storageOffset, layout) {
  return Object.freeze({
    order,
    storageOffset,
    traversal: layout.traversal,
    binStarts: layout.starts,
    bucketTotals: layout.totals,
  });
}

/**
 * Builds two immutable logical render lists in one storage allocation. Each
 * bucket retains its fixed slice; only the eight physical-bin blocks reverse.
 * Object order inside a physical bin follows expectedVisibleIds exactly.
 */
export function createFrozenDepthCrossoverPacking({
  scenario,
  viewMatrixElements,
  laneStorageOrder = DEPTH_ORDER_FRONT_TO_BACK,
}) {
  validateScenarioShape(scenario);
  validateFrozenDepthLane(laneStorageOrder, 'laneStorageOrder');
  const frozenViewMatrixElements = validateViewMatrixElements(viewMatrixElements);
  const depthBinRange = validateDepthRange(scenario.depthBinRange);
  const objectBins = createExpectedObjectDepthBins({
    bounds: scenario.bounds,
    objectCount: scenario.objectCount,
    expectedIds: scenario.expectedVisibleIds,
    viewMatrixElements: frozenViewMatrixElements,
    depthRange: depthBinRange,
  });
  const binCounts = new Uint32Array(scenario.bucketCount * DEPTH_BIN_COUNT);
  const idsByPhysicalBin = Array.from(
    { length: binCounts.length },
    () => [],
  );

  for (const objectId of scenario.expectedVisibleIds) {
    const bucket = scenario.objectBuckets[objectId];
    const physicalBin = objectBins[objectId];
    const binIndex = bucket * DEPTH_BIN_COUNT + physicalBin;
    binCounts[binIndex] += 1;
    idsByPhysicalBin[binIndex].push(objectId);
  }

  const frontLayout = createOrderedDepthBinLayout(
    binCounts,
    DEPTH_ORDER_FRONT_TO_BACK,
  );
  const reverseLayout = createOrderedDepthBinLayout(
    binCounts,
    DEPTH_ORDER_REVERSE,
  );
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    const expected = scenario.visibleCounts[bucket];
    if (frontLayout.totals[bucket] !== expected
      || reverseLayout.totals[bucket] !== expected) {
      throw new Error(`Frozen crossover bucket ${bucket} total is inconsistent.`);
    }
  }

  const laneOffsets = Object.freeze({
    [DEPTH_ORDER_FRONT_TO_BACK]: laneStorageOrder === DEPTH_ORDER_FRONT_TO_BACK
      ? 0
      : scenario.objectCount,
    [DEPTH_ORDER_REVERSE]: laneStorageOrder === DEPTH_ORDER_REVERSE
      ? 0
      : scenario.objectCount,
  });
  const visibleIds = new Uint32Array(scenario.objectCount * 2);
  visibleIds.fill(FROZEN_DEPTH_UNUSED_OBJECT_ID);
  const layouts = {
    [DEPTH_ORDER_FRONT_TO_BACK]: frontLayout,
    [DEPTH_ORDER_REVERSE]: reverseLayout,
  };

  for (const order of FROZEN_DEPTH_CROSSOVER_LANES) {
    const layout = layouts[order];
    const laneOffset = laneOffsets[order];
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const bucketBase = scenario.bucketBases[bucket];
      for (let physicalBin = 0; physicalBin < DEPTH_BIN_COUNT; physicalBin += 1) {
        const binIndex = bucket * DEPTH_BIN_COUNT + physicalBin;
        visibleIds.set(
          idsByPhysicalBin[binIndex],
          laneOffset + bucketBase + layout.starts[binIndex],
        );
      }
    }
  }

  return Object.freeze({
    schemaVersion: 1,
    kind: 'cpu-frozen-depth-crossover-packing',
    objectCount: scenario.objectCount,
    bucketCount: scenario.bucketCount,
    expectedVisibleCount: scenario.expectedVisibleIds.length,
    depthBinCount: DEPTH_BIN_COUNT,
    depthBinRange: Object.freeze({ ...depthBinRange }),
    viewMatrixElements: frozenViewMatrixElements,
    laneStorageOrder,
    laneOffsets,
    visibleIds,
    objectBins,
    binCounts,
    lanes: Object.freeze({
      [DEPTH_ORDER_FRONT_TO_BACK]: createLaneRecord(
        DEPTH_ORDER_FRONT_TO_BACK,
        laneOffsets[DEPTH_ORDER_FRONT_TO_BACK],
        frontLayout,
      ),
      [DEPTH_ORDER_REVERSE]: createLaneRecord(
        DEPTH_ORDER_REVERSE,
        laneOffsets[DEPTH_ORDER_REVERSE],
        reverseLayout,
      ),
    }),
  });
}

export function getFrozenDepthLaneSegment(packing, lane) {
  validateFrozenDepthLane(lane);
  if (!(packing?.visibleIds instanceof Uint32Array)
    || !Number.isInteger(packing.objectCount)
    || packing.visibleIds.length !== packing.objectCount * 2) {
    throw new TypeError('packing does not contain two complete uint32 lane segments.');
  }
  const offset = packing.laneOffsets?.[lane];
  if (offset !== 0 && offset !== packing.objectCount) {
    throw new RangeError(`packing lane ${lane} has an invalid storage offset.`);
  }
  return packing.visibleIds.subarray(offset, offset + packing.objectCount);
}
