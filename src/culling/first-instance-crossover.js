import {
  INDEXED_INDIRECT_STRIDE_BYTES,
  INDEXED_INDIRECT_STRIDE_UINTS,
  createIndexedIndirectCommands,
} from './indexed-command-layout.js';
import { validateIndexedCommands } from '../validation/membership.js';
import { sha256CanonicalUint32 } from '../validation/membership-digests.js';

export const FIRST_INSTANCE_COMMAND_LANES = Object.freeze([
  'portable',
  'feature',
]);

export const FIRST_INSTANCE_UNUSED_OBJECT_ID = 0xffff_ffff;

const [PORTABLE_LANE, FEATURE_LANE] = FIRST_INSTANCE_COMMAND_LANES;

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

function exactUint32(left, right) {
  return left instanceof Uint32Array
    && right instanceof Uint32Array
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function validateScenarioShape(scenario) {
  requirePositiveInteger(scenario?.objectCount, 'scenario.objectCount');
  requirePositiveInteger(scenario?.bucketCount, 'scenario.bucketCount');
  if (scenario.objectCount >= FIRST_INSTANCE_UNUSED_OBJECT_ID) {
    throw new RangeError('scenario.objectCount collides with the unused-object sentinel.');
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

  let objectCursor = 0;
  let expectedVisibleCount = 0;
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    if (scenario.bucketBases[bucket] !== objectCursor) {
      throw new RangeError('First-instance crossover requires contiguous fixed bucket slices.');
    }
    const capacity = scenario.bucketCounts[bucket];
    const visibleCount = scenario.visibleCounts[bucket];
    if (capacity === 0) {
      throw new RangeError(`scenario.bucketCounts[${bucket}] must be positive.`);
    }
    if (visibleCount > capacity) {
      throw new RangeError(`scenario.visibleCounts[${bucket}] exceeds bucket capacity.`);
    }
    const bucketEnd = objectCursor + capacity;
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
    expectedVisibleCount += visibleCount;
  }
  if (objectCursor !== scenario.objectCount) {
    throw new RangeError('scenario.bucketCounts must sum to scenario.objectCount.');
  }
  if (expectedVisibleCount !== scenario.expectedVisibleIds.length
    || (Number.isInteger(scenario.expectedVisibleCount)
      && scenario.expectedVisibleCount !== expectedVisibleCount)) {
    throw new RangeError('scenario visible counts do not match expectedVisibleIds.');
  }

  const actualVisibleCounts = new Uint32Array(scenario.bucketCount);
  let previousObjectId = -1;
  for (const objectId of scenario.expectedVisibleIds) {
    if (objectId >= scenario.objectCount) {
      throw new RangeError(`Expected visible object ID ${objectId} is out of range.`);
    }
    if (objectId <= previousObjectId) {
      throw new RangeError('scenario.expectedVisibleIds must be strictly ascending.');
    }
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

function validateSourceGeometries(sourceGeometries, bucketCount) {
  if (!Array.isArray(sourceGeometries) || sourceGeometries.length !== bucketCount) {
    throw new RangeError('sourceGeometries length must equal scenario.bucketCount.');
  }
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    const count = sourceGeometries[bucket]?.index?.count;
    if (!Number.isInteger(count) || count <= 0 || count > 0xffff_ffff) {
      throw new Error(`Geometry bucket ${bucket} must have a nonempty uint32-sized index.`);
    }
  }
}

function resolveFirstIndexes(sourceGeometries, firstIndexes) {
  if (firstIndexes !== null && firstIndexes !== undefined) {
    requireUint32Array(firstIndexes, sourceGeometries.length, 'firstIndexes');
    return firstIndexes.slice();
  }
  const resolved = new Uint32Array(sourceGeometries.length);
  let cursor = 0;
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    if (cursor > 0xffff_ffff) {
      throw new RangeError('Cumulative firstIndex exceeds uint32 capacity.');
    }
    resolved[bucket] = cursor;
    cursor += sourceGeometries[bucket].index.count;
    if (!Number.isSafeInteger(cursor)) {
      throw new RangeError('Cumulative index count exceeds JavaScript safe integer capacity.');
    }
  }
  return resolved;
}

export function validateFirstInstanceCommandSegmentOrder(
  value,
  label = 'laneCommandSegmentOrder',
) {
  if (!Array.isArray(value)
    || value.length !== FIRST_INSTANCE_COMMAND_LANES.length
    || new Set(value).size !== FIRST_INSTANCE_COMMAND_LANES.length
    || FIRST_INSTANCE_COMMAND_LANES.some((lane) => !value.includes(lane))) {
    throw new RangeError(
      `${label} must be the exact portable/feature lane permutation.`,
    );
  }
  return value;
}

export function validateFirstInstanceCommandLane(value, label = 'lane') {
  if (!FIRST_INSTANCE_COMMAND_LANES.includes(value)) {
    throw new RangeError(`${label} must be portable or feature.`);
  }
  return value;
}

function createVisibleIdPacking(scenario) {
  const visibleIds = new Uint32Array(scenario.objectCount);
  visibleIds.fill(FIRST_INSTANCE_UNUSED_OBJECT_ID);
  const cursors = new Uint32Array(scenario.bucketCount);
  for (const objectId of scenario.expectedVisibleIds) {
    const bucket = scenario.objectBuckets[objectId];
    const cursor = cursors[bucket];
    if (cursor >= scenario.visibleCounts[bucket]) {
      throw new Error(`Visible packing overflowed bucket ${bucket}.`);
    }
    visibleIds[scenario.bucketBases[bucket] + cursor] = objectId;
    cursors[bucket] += 1;
  }
  if (!exactUint32(cursors, scenario.visibleCounts)) {
    throw new Error('Frozen visible packing did not fill every declared survivor slot.');
  }
  return visibleIds;
}

function copyLaneRecord(record) {
  return Object.freeze({
    laneId: record.laneId,
    commandSegmentIndex: record.commandSegmentIndex,
    commandRecordBase: record.commandRecordBase,
    commandByteBase: record.commandByteBase,
    offsets: record.offsets,
  });
}

export function createFirstInstanceCrossoverPacking({
  scenario,
  sourceGeometries,
  firstIndexes = null,
  laneCommandSegmentOrder = FIRST_INSTANCE_COMMAND_LANES,
}) {
  validateScenarioShape(scenario);
  validateSourceGeometries(sourceGeometries, scenario.bucketCount);
  validateFirstInstanceCommandSegmentOrder(laneCommandSegmentOrder);
  const resolvedFirstIndexes = resolveFirstIndexes(sourceGeometries, firstIndexes);
  const visibleIds = createVisibleIdPacking(scenario);

  const logicalCommands = {
    [PORTABLE_LANE]: createIndexedIndirectCommands(
      sourceGeometries,
      scenario.bucketCounts,
      scenario.visibleCounts,
      resolvedFirstIndexes,
    ),
    [FEATURE_LANE]: createIndexedIndirectCommands(
      sourceGeometries,
      scenario.bucketCounts,
      scenario.visibleCounts,
      resolvedFirstIndexes,
      scenario.bucketBases,
    ),
  };
  const recordsPerLane = logicalCommands[PORTABLE_LANE].recordCount;
  if (logicalCommands[FEATURE_LANE].recordCount !== recordsPerLane) {
    throw new Error('First-instance command lanes have different record capacities.');
  }
  const uintsPerLane = recordsPerLane * INDEXED_INDIRECT_STRIDE_UINTS;
  const bytesPerLane = recordsPerLane * INDEXED_INDIRECT_STRIDE_BYTES;
  const commands = new Uint32Array(uintsPerLane * FIRST_INSTANCE_COMMAND_LANES.length);
  const lanes = {};

  for (let segmentIndex = 0;
    segmentIndex < laneCommandSegmentOrder.length;
    segmentIndex += 1) {
    const laneId = laneCommandSegmentOrder[segmentIndex];
    const commandRecordBase = segmentIndex * recordsPerLane;
    const commandByteBase = segmentIndex * bytesPerLane;
    commands.set(
      logicalCommands[laneId].commands,
      commandRecordBase * INDEXED_INDIRECT_STRIDE_UINTS,
    );
    lanes[laneId] = copyLaneRecord({
      laneId,
      commandSegmentIndex: segmentIndex,
      commandRecordBase,
      commandByteBase,
      offsets: Uint32Array.from(
        logicalCommands[laneId].offsets,
        (offset) => commandByteBase + offset,
      ),
    });
  }

  return Object.freeze({
    schemaVersion: 1,
    kind: 'cpu-frozen-first-instance-crossover-packing',
    objectCount: scenario.objectCount,
    bucketCount: scenario.bucketCount,
    expectedVisibleCount: scenario.expectedVisibleIds.length,
    bucketBases: scenario.bucketBases.slice(),
    bucketCounts: scenario.bucketCounts.slice(),
    visibleCounts: scenario.visibleCounts.slice(),
    firstIndexes: resolvedFirstIndexes,
    visibleIds,
    commands,
    recordsPerLane,
    commandSegmentByteLength: bytesPerLane,
    laneCommandSegmentOrder: Object.freeze([...laneCommandSegmentOrder]),
    lanes: Object.freeze(lanes),
  });
}

function validatePackingShape(packing) {
  if (!packing || packing.kind !== 'cpu-frozen-first-instance-crossover-packing') {
    throw new TypeError('packing is not a first-instance crossover packing.');
  }
  requirePositiveInteger(packing.objectCount, 'packing.objectCount');
  requirePositiveInteger(packing.bucketCount, 'packing.bucketCount');
  requirePositiveInteger(packing.recordsPerLane, 'packing.recordsPerLane');
  requireUint32Array(packing.visibleIds, packing.objectCount, 'packing.visibleIds');
  requireUint32Array(
    packing.commands,
    packing.recordsPerLane
      * FIRST_INSTANCE_COMMAND_LANES.length
      * INDEXED_INDIRECT_STRIDE_UINTS,
    'packing.commands',
  );
  requireUint32Array(packing.bucketBases, packing.bucketCount, 'packing.bucketBases');
  requireUint32Array(packing.bucketCounts, packing.bucketCount, 'packing.bucketCounts');
  requireUint32Array(packing.visibleCounts, packing.bucketCount, 'packing.visibleCounts');
  requireUint32Array(packing.firstIndexes, packing.bucketCount, 'packing.firstIndexes');
  validateFirstInstanceCommandSegmentOrder(packing.laneCommandSegmentOrder);
  for (const lane of FIRST_INSTANCE_COMMAND_LANES) {
    const laneRecord = packing.lanes?.[lane];
    if (!laneRecord) throw new TypeError(`packing lacks the ${lane} lane record.`);
    requireUint32Array(laneRecord.offsets, packing.bucketCount, `${lane}.offsets`);
  }
}

export function getFirstInstanceLaneCommandSegment(packing, lane) {
  validatePackingShape(packing);
  validateFirstInstanceCommandLane(lane);
  const start = packing.lanes[lane].commandRecordBase * INDEXED_INDIRECT_STRIDE_UINTS;
  return packing.commands.subarray(
    start,
    start + packing.recordsPerLane * INDEXED_INDIRECT_STRIDE_UINTS,
  );
}

export function getFirstInstanceLaneDrawCommands(packing, lane) {
  return getFirstInstanceLaneCommandSegment(packing, lane).subarray(
    0,
    packing.bucketCount * INDEXED_INDIRECT_STRIDE_UINTS,
  );
}

function commandCore(commands, bucketCount) {
  const core = new Uint32Array(bucketCount * (INDEXED_INDIRECT_STRIDE_UINTS - 1));
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const sourceBase = bucket * INDEXED_INDIRECT_STRIDE_UINTS;
    core.set(
      commands.subarray(sourceBase, sourceBase + INDEXED_INDIRECT_STRIDE_UINTS - 1),
      bucket * (INDEXED_INDIRECT_STRIDE_UINTS - 1),
    );
  }
  return core;
}

function pairCanonical(portable, feature, bucketCount) {
  const wordsPerPair = 1 + INDEXED_INDIRECT_STRIDE_UINTS * 2;
  const canonical = new Uint32Array(bucketCount * wordsPerPair);
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const targetBase = bucket * wordsPerPair;
    const sourceBase = bucket * INDEXED_INDIRECT_STRIDE_UINTS;
    canonical[targetBase] = bucket;
    canonical.set(
      portable.subarray(sourceBase, sourceBase + INDEXED_INDIRECT_STRIDE_UINTS),
      targetBase + 1,
    );
    canonical.set(
      feature.subarray(sourceBase, sourceBase + INDEXED_INDIRECT_STRIDE_UINTS),
      targetBase + 1 + INDEXED_INDIRECT_STRIDE_UINTS,
    );
  }
  return canonical;
}

function paddingCanonical(packing) {
  let paddingCount = 0;
  for (let bucket = 0; bucket < packing.bucketCount; bucket += 1) {
    paddingCount += packing.bucketCounts[bucket] - packing.visibleCounts[bucket];
  }
  const canonical = new Uint32Array(paddingCount * 2);
  let cursor = 0;
  for (let bucket = 0; bucket < packing.bucketCount; bucket += 1) {
    const start = packing.bucketBases[bucket] + packing.visibleCounts[bucket];
    const end = packing.bucketBases[bucket] + packing.bucketCounts[bucket];
    for (let address = start; address < end; address += 1) {
      canonical[cursor] = address;
      canonical[cursor + 1] = packing.visibleIds[address];
      cursor += 2;
    }
  }
  return canonical;
}

export async function createFirstInstanceCrossoverCommitments(packing) {
  validatePackingShape(packing);
  const portable = getFirstInstanceLaneDrawCommands(packing, PORTABLE_LANE);
  const feature = getFirstInstanceLaneDrawCommands(packing, FEATURE_LANE);
  const portableCore = commandCore(portable, packing.bucketCount);
  const featureCore = commandCore(feature, packing.bucketCount);
  const pairWords = pairCanonical(portable, feature, packing.bucketCount);
  const paddingWords = paddingCanonical(packing);
  const perBucket = await Promise.all(Array.from(
    { length: packing.bucketCount },
    async (_, bucket) => {
      const wordsPerPair = 1 + INDEXED_INDIRECT_STRIDE_UINTS * 2;
      const pair = pairWords.subarray(bucket * wordsPerPair, (bucket + 1) * wordsPerPair);
      const portableBase = bucket * INDEXED_INDIRECT_STRIDE_UINTS;
      const featureBase = bucket * INDEXED_INDIRECT_STRIDE_UINTS;
      const coreEqual = portable.subarray(portableBase, portableBase + 4)
        .every((value, index) => value === feature[featureBase + index]);
      return {
        bucket,
        coreEqual,
        portableFirstInstance: portable[portableBase + 4],
        featureFirstInstance: feature[featureBase + 4],
        expectedFeatureFirstInstance: packing.bucketBases[bucket],
        sha256: await sha256CanonicalUint32(pair),
      };
    },
  ));
  const [
    visibleIdsSha256,
    physicalCommandsSha256,
    portableCommandsSha256,
    featureCommandsSha256,
    portableCoreSha256,
    featureCoreSha256,
    logicalPairSha256,
    paddingSha256,
  ] = await Promise.all([
    sha256CanonicalUint32(packing.visibleIds),
    sha256CanonicalUint32(packing.commands),
    sha256CanonicalUint32(portable),
    sha256CanonicalUint32(feature),
    sha256CanonicalUint32(portableCore),
    sha256CanonicalUint32(featureCore),
    sha256CanonicalUint32(pairWords),
    sha256CanonicalUint32(paddingWords),
  ]);
  return {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    encoding: 'uint32-little-endian',
    visibleIdsSha256,
    physicalCommandsSha256,
    logicalPairSha256,
    paddingSha256,
    commandCoresEqual: exactUint32(portableCore, featureCore)
      && portableCoreSha256 === featureCoreSha256,
    lanes: {
      [PORTABLE_LANE]: {
        commandsSha256: portableCommandsSha256,
        coreSha256: portableCoreSha256,
      },
      [FEATURE_LANE]: {
        commandsSha256: featureCommandsSha256,
        coreSha256: featureCoreSha256,
      },
    },
    pairs: perBucket,
  };
}

function inspectPadding(packing) {
  let paddingCount = 0;
  let paddingSentinelCount = 0;
  let activeSentinelCount = 0;
  const corruptPaddingAddresses = [];
  for (let bucket = 0; bucket < packing.bucketCount; bucket += 1) {
    const base = packing.bucketBases[bucket];
    const activeEnd = base + packing.visibleCounts[bucket];
    const capacityEnd = base + packing.bucketCounts[bucket];
    for (let address = base; address < activeEnd; address += 1) {
      if (packing.visibleIds[address] === FIRST_INSTANCE_UNUSED_OBJECT_ID) {
        activeSentinelCount += 1;
      }
    }
    for (let address = activeEnd; address < capacityEnd; address += 1) {
      paddingCount += 1;
      if (packing.visibleIds[address] === FIRST_INSTANCE_UNUSED_OBJECT_ID) {
        paddingSentinelCount += 1;
      } else {
        corruptPaddingAddresses.push(address);
      }
    }
  }
  return {
    pass: activeSentinelCount === 0 && paddingSentinelCount === paddingCount,
    sentinel: FIRST_INSTANCE_UNUSED_OBJECT_ID,
    paddingCount,
    paddingSentinelCount,
    activeSentinelCount,
    corruptPaddingAddresses,
  };
}

export async function validateFirstInstanceCrossoverPacking({
  packing,
  scenario,
  sourceGeometries,
  firstIndexes = null,
  expectedLaneCommandSegmentOrder = null,
}) {
  validatePackingShape(packing);
  validateScenarioShape(scenario);
  validateSourceGeometries(sourceGeometries, scenario.bucketCount);
  const committedOrder = expectedLaneCommandSegmentOrder === null
    ? packing.laneCommandSegmentOrder
    : expectedLaneCommandSegmentOrder;
  validateFirstInstanceCommandSegmentOrder(committedOrder, 'expectedLaneCommandSegmentOrder');
  const expected = createFirstInstanceCrossoverPacking({
    scenario,
    sourceGeometries,
    firstIndexes,
    laneCommandSegmentOrder: committedOrder,
  });

  const metadataErrors = [];
  for (const field of [
    'objectCount',
    'bucketCount',
    'expectedVisibleCount',
    'recordsPerLane',
    'commandSegmentByteLength',
  ]) {
    if (packing[field] !== expected[field]) metadataErrors.push(field);
  }
  if (packing.laneCommandSegmentOrder.some(
    (lane, index) => lane !== expected.laneCommandSegmentOrder[index],
  )) {
    metadataErrors.push('laneCommandSegmentOrder');
  }
  for (const field of ['bucketBases', 'bucketCounts', 'visibleCounts', 'firstIndexes']) {
    if (!exactUint32(packing[field], expected[field])) metadataErrors.push(field);
  }
  for (const lane of FIRST_INSTANCE_COMMAND_LANES) {
    const actualLane = packing.lanes[lane];
    const expectedLane = expected.lanes[lane];
    for (const field of [
      'commandSegmentIndex',
      'commandRecordBase',
      'commandByteBase',
    ]) {
      if (actualLane[field] !== expectedLane[field]) metadataErrors.push(`${lane}.${field}`);
    }
    if (!exactUint32(actualLane.offsets, expectedLane.offsets)) {
      metadataErrors.push(`${lane}.offsets`);
    }
  }

  const visibleIdsExact = exactUint32(packing.visibleIds, expected.visibleIds);
  const physicalCommandsExact = exactUint32(packing.commands, expected.commands);
  const padding = inspectPadding(packing);
  const commandValidations = {
    [PORTABLE_LANE]: validateIndexedCommands({
      commands: getFirstInstanceLaneCommandSegment(packing, PORTABLE_LANE),
      geometries: sourceGeometries,
      expectedCounts: scenario.visibleCounts,
      expectedFirstIndexes: expected.firstIndexes,
    }),
    [FEATURE_LANE]: validateIndexedCommands({
      commands: getFirstInstanceLaneCommandSegment(packing, FEATURE_LANE),
      geometries: sourceGeometries,
      expectedCounts: scenario.visibleCounts,
      expectedFirstIndexes: expected.firstIndexes,
      expectedFirstInstances: scenario.bucketBases,
    }),
  };
  const commitments = await createFirstInstanceCrossoverCommitments(packing);
  const pairPass = commitments.commandCoresEqual
    && commitments.pairs.every((pair) => (
      pair.coreEqual
      && pair.portableFirstInstance === 0
      && pair.featureFirstInstance === pair.expectedFeatureFirstInstance
    ));
  const pass = metadataErrors.length === 0
    && visibleIdsExact
    && physicalCommandsExact
    && padding.pass
    && FIRST_INSTANCE_COMMAND_LANES.every((lane) => commandValidations[lane].pass)
    && pairPass;

  return {
    schemaVersion: 1,
    kind: 'first-instance-crossover-exact-frozen-packing-validation',
    pass,
    metadata: {
      pass: metadataErrors.length === 0,
      errors: metadataErrors,
      laneCommandSegmentOrder: [...packing.laneCommandSegmentOrder],
      recordsPerLane: packing.recordsPerLane,
      commandSegmentByteLength: packing.commandSegmentByteLength,
    },
    visibleIds: {
      pass: visibleIdsExact,
      exactExpectedPacking: visibleIdsExact,
      elementCount: packing.visibleIds.length,
      sha256: commitments.visibleIdsSha256,
    },
    padding,
    commands: {
      pass: physicalCommandsExact
        && FIRST_INSTANCE_COMMAND_LANES.every((lane) => commandValidations[lane].pass)
        && pairPass,
      exactExpectedPhysicalPacking: physicalCommandsExact,
      pairPass,
      lanes: commandValidations,
    },
    commitments,
  };
}
