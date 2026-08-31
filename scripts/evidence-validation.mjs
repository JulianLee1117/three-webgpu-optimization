import { createHash } from 'node:crypto';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
} from '../src/benchmark/plan.js';
import {
  FROZEN_CROSSOVER_BLOCK_SIZE,
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  FROZEN_CROSSOVER_PATTERNS,
  FROZEN_CROSSOVER_WARMUP_FRAMES,
  frozenCrossoverFrame,
} from '../src/benchmark/frozen-crossover-schedule.js';

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const GEOMETRY_ATTRIBUTE_NAMES = Object.freeze(['normal', 'position', 'uv']);
const DEPTH_BIN_COUNT = 8;
const RENDER_PARITY_WIDTH = 1280;
const RENDER_PARITY_HEIGHT = 720;
const RENDER_PARITY_PIXEL_COUNT = RENDER_PARITY_WIDTH * RENDER_PARITY_HEIGHT;
const RENDER_PARITY_BYTE_LENGTH = RENDER_PARITY_PIXEL_COUNT * 4;
const DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS = 8;
const FROZEN_LANE_ORDER_BY_ID = Object.freeze({
  'fixed-slice-depth-front-to-back': 'front-to-back',
  'fixed-slice-depth-reverse': 'reverse',
});
const FROZEN_LOGICAL_LANES = Object.freeze(['front-to-back', 'reverse']);
const FROZEN_EXPECTED_MEASURED_ROWS_PER_LANE =
  FROZEN_CROSSOVER_MEASURED_FRAMES / FROZEN_LOGICAL_LANES.length;
const DEPTH_MODE_PROTOCOLS = Object.freeze({
  'fixed-slice-depth-front-to-back': Object.freeze({
    order: 'front-to-back',
    traversal: Object.freeze([0, 1, 2, 3, 4, 5, 6, 7]),
    reverseOrderUniformValue: false,
  }),
  'fixed-slice-depth-reverse': Object.freeze({
    order: 'reverse',
    traversal: Object.freeze([7, 6, 5, 4, 3, 2, 1, 0]),
    reverseOrderUniformValue: true,
  }),
});
const SCENARIO_ARRAY_SCHEMA = Object.freeze({
  bucketCounts: Object.freeze({ arrayType: 'Uint32Array', length: ({ bucketCount }) => bucketCount }),
  bucketBases: Object.freeze({ arrayType: 'Uint32Array', length: ({ bucketCount }) => bucketCount }),
  visibleCounts: Object.freeze({ arrayType: 'Uint32Array', length: ({ bucketCount }) => bucketCount }),
  objectBuckets: Object.freeze({ arrayType: 'Uint32Array', length: ({ objectCount }) => objectCount }),
  matrices: Object.freeze({ arrayType: 'Float32Array', length: ({ objectCount }) => objectCount * 16 }),
  bounds: Object.freeze({ arrayType: 'Float32Array', length: ({ objectCount }) => objectCount * 4 }),
  expectedVisibleIds: Object.freeze({
    arrayType: 'Uint32Array',
    length: ({ expectedVisibleCount }) => expectedVisibleCount,
  }),
  cullOrder: Object.freeze({ arrayType: 'Uint32Array', length: ({ objectCount }) => objectCount }),
});

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, expectedKeys, label, reasons) {
  if (!isRecord(value)) {
    reasons.push(`${label} is not an object`);
    return false;
  }
  const actualKeys = Object.keys(value).sort();
  const sortedExpected = [...expectedKeys].sort();
  if (actualKeys.length !== sortedExpected.length
    || actualKeys.some((key, index) => key !== sortedExpected[index])) {
    reasons.push(`${label} has an unexpected schema`);
    return false;
  }
  return true;
}

function isNonnegativeInteger(value) {
  return Number.isInteger(value) && value >= 0;
}

function isPositiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function isFiniteVector(value, length) {
  return Array.isArray(value) && value.length === length && value.every(Number.isFinite);
}

function requireSha256(value, label, reasons) {
  if (!SHA256_PATTERN.test(value ?? '')) reasons.push(`${label} is not a lowercase SHA-256 digest`);
}

function requireEqual(actual, expected, label, reasons) {
  if (actual !== expected) reasons.push(`${label} is ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}`);
}

function requireExactArray(actual, expected, label, reasons) {
  if (!Array.isArray(actual) || actual.length !== expected.length) {
    reasons.push(`${label} has the wrong length`);
    return false;
  }
  for (let index = 0; index < expected.length; index += 1) {
    requireEqual(actual[index], expected[index], `${label}[${index}]`, reasons);
  }
  return true;
}

function depthModeProtocol(modeId) {
  return DEPTH_MODE_PROTOCOLS[modeId] ?? null;
}

function frozenLogicalLane(laneId) {
  return FROZEN_LANE_ORDER_BY_ID[laneId] ?? null;
}

function frozenLaneModeId(logicalLane) {
  return FROZEN_DEPTH_CROSSOVER_LANES.find(
    (laneId) => frozenLogicalLane(laneId) === logicalLane,
  ) ?? null;
}

export function sha256Json(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}

function frozenSchedulePhase(frameCount, orientationOffset) {
  return Array.from({ length: frameCount }, (_, phaseFrameIndex) => {
    const frame = frozenCrossoverFrame(phaseFrameIndex, orientationOffset);
    return {
      phaseFrameIndex,
      crossoverBlockIndex: frame.crossoverBlockIndex,
      withinBlockPosition: frame.withinBlockPosition,
      patternIndex: frame.patternIndex,
      pattern: frame.pattern,
      laneId: frame.laneId,
    };
  });
}

export function frozenCrossoverScheduleSha256(orientationOffset) {
  return sha256Json({
    schemaVersion: 1,
    kind: 'frozen-depth-crossover-frame-schedule',
    blockSize: FROZEN_CROSSOVER_BLOCK_SIZE,
    warmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
    orientationOffset,
    warmup: frozenSchedulePhase(FROZEN_CROSSOVER_WARMUP_FRAMES, orientationOffset),
    measured: frozenSchedulePhase(FROZEN_CROSSOVER_MEASURED_FRAMES, orientationOffset),
  });
}

function canonicalAttribute(attribute) {
  return {
    arrayType: attribute?.arrayType,
    count: attribute?.count,
    itemSize: attribute?.itemSize,
    normalized: attribute?.normalized,
    sha256: attribute?.sha256,
  };
}

function validateGeometryAttribute(attribute, label, {
  arrayTypes,
  itemSize,
  expectedCount = null,
}, reasons) {
  if (!exactKeys(
    attribute,
    ['arrayType', 'count', 'itemSize', 'normalized', 'sha256'],
    label,
    reasons,
  )) return;
  if (!arrayTypes.includes(attribute.arrayType)) {
    reasons.push(`${label}.arrayType is unsupported`);
  }
  if (!isPositiveInteger(attribute.count)) reasons.push(`${label}.count is not a positive integer`);
  requireEqual(attribute.itemSize, itemSize, `${label}.itemSize`, reasons);
  requireEqual(attribute.normalized, false, `${label}.normalized`, reasons);
  if (expectedCount !== null) requireEqual(attribute.count, expectedCount, `${label}.count`, reasons);
  requireSha256(attribute.sha256, `${label}.sha256`, reasons);
}

function canonicalGeometryRecord(record) {
  return {
    bucket: record.bucket,
    family: record.family,
    name: record.name,
    attributes: Object.fromEntries(GEOMETRY_ATTRIBUTE_NAMES.map((name) => (
      [name, canonicalAttribute(record.attributes[name])]
    ))),
    index: canonicalAttribute(record.index),
    drawRange: {
      start: record.drawRange.start,
      count: record.drawRange.count,
    },
    boundingBox: {
      min: record.boundingBox.min,
      max: record.boundingBox.max,
    },
    boundingSphere: {
      center: record.boundingSphere.center,
      radius: record.boundingSphere.radius,
    },
  };
}

export function validateGeometryFixtureManifest(manifest, {
  bucketCount,
  tier = 'medium',
} = {}) {
  const reasons = [];
  const rootKeys = ['schemaVersion', 'generator', 'tier', 'bucketCount', 'geometries', 'sha256'];
  if (!exactKeys(manifest, rootKeys, 'geometry manifest', reasons)) return reasons;
  requireEqual(manifest.schemaVersion, 1, 'geometry manifest schemaVersion', reasons);
  requireEqual(manifest.generator, 'createIndexedGeometryFixtures', 'geometry manifest generator', reasons);
  requireEqual(manifest.tier, tier, 'geometry manifest tier', reasons);
  requireEqual(manifest.bucketCount, bucketCount, 'geometry manifest bucketCount', reasons);
  requireSha256(manifest.sha256, 'geometry manifest sha256', reasons);
  if (!Array.isArray(manifest.geometries)) {
    reasons.push('geometry manifest geometries is not an array');
    return reasons;
  }
  requireEqual(manifest.geometries.length, bucketCount, 'geometry manifest geometry count', reasons);

  const canonicalRecords = [];
  for (let bucket = 0; bucket < manifest.geometries.length; bucket += 1) {
    const record = manifest.geometries[bucket];
    const label = `geometry manifest bucket ${bucket}`;
    if (!exactKeys(record, [
      'bucket',
      'family',
      'name',
      'attributes',
      'index',
      'drawRange',
      'boundingBox',
      'boundingSphere',
      'sha256',
    ], label, reasons)) continue;
    requireEqual(record.bucket, bucket, `${label}.bucket`, reasons);
    requireEqual(record.family, bucket % 4, `${label}.family`, reasons);
    if (typeof record.name !== 'string' || record.name.length === 0) {
      reasons.push(`${label}.name is missing`);
    }
    if (!exactKeys(record.attributes, GEOMETRY_ATTRIBUTE_NAMES, `${label}.attributes`, reasons)) {
      continue;
    }
    const positionCount = record.attributes.position?.count ?? null;
    validateGeometryAttribute(record.attributes.normal, `${label}.attributes.normal`, {
      arrayTypes: ['Float32Array'], itemSize: 3, expectedCount: positionCount,
    }, reasons);
    validateGeometryAttribute(record.attributes.position, `${label}.attributes.position`, {
      arrayTypes: ['Float32Array'], itemSize: 3,
    }, reasons);
    validateGeometryAttribute(record.attributes.uv, `${label}.attributes.uv`, {
      arrayTypes: ['Float32Array'], itemSize: 2, expectedCount: positionCount,
    }, reasons);
    validateGeometryAttribute(record.index, `${label}.index`, {
      arrayTypes: ['Uint16Array', 'Uint32Array'], itemSize: 1,
    }, reasons);
    if (isPositiveInteger(record.index?.count) && record.index.count % 3 !== 0) {
      reasons.push(`${label}.index.count is not triangle aligned`);
    }
    if (!exactKeys(record.drawRange, ['start', 'count'], `${label}.drawRange`, reasons)) continue;
    requireEqual(record.drawRange.start, 0, `${label}.drawRange.start`, reasons);
    requireEqual(record.drawRange.count, 'Infinity', `${label}.drawRange.count`, reasons);
    if (!exactKeys(record.boundingBox, ['min', 'max'], `${label}.boundingBox`, reasons)) continue;
    if (!isFiniteVector(record.boundingBox.min, 3)) reasons.push(`${label}.boundingBox.min is malformed`);
    if (!isFiniteVector(record.boundingBox.max, 3)) reasons.push(`${label}.boundingBox.max is malformed`);
    if (!exactKeys(record.boundingSphere, ['center', 'radius'], `${label}.boundingSphere`, reasons)) continue;
    if (!isFiniteVector(record.boundingSphere.center, 3)) reasons.push(`${label}.boundingSphere.center is malformed`);
    if (!Number.isFinite(record.boundingSphere.radius) || record.boundingSphere.radius <= 0) {
      reasons.push(`${label}.boundingSphere.radius is not finite and positive`);
    }
    requireSha256(record.sha256, `${label}.sha256`, reasons);

    const canonical = canonicalGeometryRecord(record);
    canonicalRecords.push({ ...canonical, sha256: record.sha256 });
    const recomputed = sha256Json(canonical);
    if (record.sha256 !== recomputed) reasons.push(`${label}.sha256 does not match its record`);
  }

  if (canonicalRecords.length === manifest.geometries.length) {
    if (new Set(canonicalRecords.map((record) => record.sha256)).size !== canonicalRecords.length) {
      reasons.push('geometry manifest contains duplicate fixture digests');
    }
    const canonicalManifest = {
      schemaVersion: manifest.schemaVersion,
      generator: manifest.generator,
      tier: manifest.tier,
      bucketCount: manifest.bucketCount,
      geometries: canonicalRecords,
    };
    if (manifest.sha256 !== sha256Json(canonicalManifest)) {
      reasons.push('geometry manifest sha256 does not match its nested records');
    }
  }
  return [...new Set(reasons)];
}

function canonicalScenarioArray(record) {
  return {
    arrayType: record.arrayType,
    length: record.length,
    sha256: record.sha256,
  };
}

export function validateScenarioManifest(manifest, {
  objectCount,
  bucketCount,
  visibilityFraction,
  seed,
  layout = 'baseline',
} = {}) {
  const reasons = [];
  const hasLayoutExtension = isRecord(manifest)
    && (Object.hasOwn(manifest, 'layout') || Object.hasOwn(manifest, 'depthBinRange'));
  const rootKeys = [
    'schemaVersion',
    'generator',
    'seed',
    'objectCount',
    'bucketCount',
    'visibilityFraction',
    'expectedVisibleCount',
    'expectedVisibleIdsCanonicalSha256',
    'arrays',
    'sha256',
  ];
  if (hasLayoutExtension) rootKeys.splice(6, 0, 'layout', 'depthBinRange');
  if (!exactKeys(manifest, rootKeys, 'scenario manifest', reasons)) return reasons;
  requireEqual(manifest.schemaVersion, 1, 'scenario manifest schemaVersion', reasons);
  requireEqual(manifest.generator, 'createFixedSubsetScenario', 'scenario manifest generator', reasons);
  requireEqual(manifest.seed, seed, 'scenario manifest seed', reasons);
  requireEqual(manifest.objectCount, objectCount, 'scenario manifest objectCount', reasons);
  requireEqual(manifest.bucketCount, bucketCount, 'scenario manifest bucketCount', reasons);
  requireEqual(
    manifest.visibilityFraction,
    visibilityFraction,
    'scenario manifest visibilityFraction',
    reasons,
  );
  if (hasLayoutExtension) {
    if (!['baseline', 'low-overlap', 'high-overlap'].includes(manifest.layout)) {
      reasons.push('scenario manifest layout is unsupported');
    }
    requireEqual(manifest.layout, layout, 'scenario manifest layout', reasons);
    if (manifest.depthBinRange !== null) {
      if (exactKeys(
        manifest.depthBinRange,
        ['near', 'far'],
        'scenario manifest depthBinRange',
        reasons,
      )) {
        if (!Number.isFinite(manifest.depthBinRange.near)
          || !Number.isFinite(manifest.depthBinRange.far)
          || manifest.depthBinRange.far <= manifest.depthBinRange.near) {
          reasons.push('scenario manifest depthBinRange is not a finite increasing range');
        }
      }
    }
  }
  const expectedVisibleCount = Math.round(objectCount * visibilityFraction);
  requireEqual(
    manifest.expectedVisibleCount,
    expectedVisibleCount,
    'scenario manifest expectedVisibleCount',
    reasons,
  );
  requireSha256(
    manifest.expectedVisibleIdsCanonicalSha256,
    'scenario manifest expectedVisibleIdsCanonicalSha256',
    reasons,
  );
  requireSha256(manifest.sha256, 'scenario manifest sha256', reasons);
  const arrayNames = Object.keys(SCENARIO_ARRAY_SCHEMA);
  if (!exactKeys(manifest.arrays, arrayNames, 'scenario manifest arrays', reasons)) return reasons;

  const canonicalArrays = {};
  const dimensions = { objectCount, bucketCount, expectedVisibleCount };
  for (const name of arrayNames) {
    const record = manifest.arrays[name];
    const label = `scenario manifest arrays.${name}`;
    if (!exactKeys(record, ['arrayType', 'length', 'sha256'], label, reasons)) continue;
    const schema = SCENARIO_ARRAY_SCHEMA[name];
    requireEqual(record.arrayType, schema.arrayType, `${label}.arrayType`, reasons);
    requireEqual(record.length, schema.length(dimensions), `${label}.length`, reasons);
    requireSha256(record.sha256, `${label}.sha256`, reasons);
    canonicalArrays[name] = canonicalScenarioArray(record);
  }
  requireEqual(
    manifest.arrays.expectedVisibleIds?.sha256,
    manifest.expectedVisibleIdsCanonicalSha256,
    'scenario expected-visible canonical/array digest',
    reasons,
  );

  if (Object.keys(canonicalArrays).length === arrayNames.length) {
    const canonicalManifest = {
      schemaVersion: manifest.schemaVersion,
      generator: manifest.generator,
      seed: manifest.seed,
      objectCount: manifest.objectCount,
      bucketCount: manifest.bucketCount,
      visibilityFraction: manifest.visibilityFraction,
      ...(hasLayoutExtension ? {
        layout: manifest.layout,
        depthBinRange: manifest.depthBinRange,
      } : {}),
      expectedVisibleCount: manifest.expectedVisibleCount,
      expectedVisibleIdsCanonicalSha256: manifest.expectedVisibleIdsCanonicalSha256,
      arrays: canonicalArrays,
    };
    if (manifest.sha256 !== sha256Json(canonicalManifest)) {
      reasons.push('scenario manifest sha256 does not match its nested records');
    }
  }
  return [...new Set(reasons)];
}

function expectedStrategyShape(modeId, bucketCount, objectCount) {
  const depthProtocol = depthModeProtocol(modeId);
  const shared = {
    kind: modeId === 'draw-all'
      ? 'draw-all-reference'
      : depthProtocol
        ? `${modeId}-exact-membership-and-depth-order`
        : `${modeId}-exact-membership`,
    configuredDrawCommands: bucketCount,
    configuredSubmittedInstances: modeId === 'draw-all' ? objectCount : null,
  };
  const shapes = {
    'draw-all': {
      compute: false,
      configuredRenderObjects: bucketCount,
      configuredComputeDispatches: 0,
      configuredComputeSubmissions: 0,
    },
    'fixed-slice': {
      compute: true,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 2,
      configuredComputeSubmissions: 1,
    },
    'fixed-slice-per-bucket': {
      compute: true,
      configuredRenderObjects: bucketCount,
      configuredComputeDispatches: 2,
      configuredComputeSubmissions: 1,
    },
    'fixed-slice-depth-front-to-back': {
      compute: true,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 4,
      configuredComputeSubmissions: 1,
    },
    'fixed-slice-depth-reverse': {
      compute: true,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 4,
      configuredComputeSubmissions: 1,
    },
    [FROZEN_DEPTH_CROSSOVER_MODE]: {
      compute: false,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 0,
      configuredComputeSubmissions: 0,
    },
    'three-blocks-current': {
      compute: true,
      configuredRenderObjects: bucketCount,
      configuredComputeDispatches: bucketCount * 4,
      configuredComputeSubmissions: bucketCount,
    },
    'three-blocks-coalesced': {
      compute: true,
      configuredRenderObjects: bucketCount,
      configuredComputeDispatches: bucketCount * 4,
      configuredComputeSubmissions: 1,
    },
    'three-blocks-historical': {
      compute: true,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 9,
      configuredComputeSubmissions: 1,
    },
  };
  return shapes[modeId] ? { ...shared, ...shapes[modeId] } : null;
}

function commandSemanticRecord(validation, modeId) {
  const command = validation.commandValidation;
  if (!command) return null;
  return {
    commandCount: command.commandCount,
    total: command.totalInstanceCount ?? command.totalCommandSurvivors ?? null,
    survivorReadbackLength: command.survivorReadbackLength ?? null,
    records: command.records.map((record) => ({
      bucket: record?.bucket,
      actual: {
        indexCount: record?.actual?.indexCount,
        instanceCount: record?.actual?.instanceCount,
        firstIndex: record?.actual?.firstIndex,
        baseVertex: record?.actual?.baseVertex,
        firstInstance: modeId === 'three-blocks-historical' ? null : record?.actual?.firstInstance,
      },
      expected: {
        indexCount: record?.expected?.indexCount,
        instanceCount: record?.expected?.instanceCount,
        firstIndex: record?.expected?.firstIndex,
        baseVertex: record?.expected?.baseVertex,
        firstInstance: record?.expected?.firstInstance,
      },
    })),
  };
}

function validateDepthBinEvidence(depthBins, {
  modeId,
  bucketCount,
  expectedVisibleCount,
  membershipDigests,
  commandValidation,
}, reasons) {
  const protocol = depthModeProtocol(modeId);
  if (!protocol) return;
  if (!exactKeys(depthBins, [
    'pass',
    'errors',
    'binCount',
    'order',
    'traversal',
    'expectedCounts',
    'actualCounts',
    'expectedStarts',
    'actualStarts',
    'expectedBucketTotals',
    'commandCounts',
    'physicalBinSequenceCommitment',
  ], 'depth-bin validation', reasons)) return;

  requireEqual(depthBins.pass, true, 'depth-bin validation pass', reasons);
  if (!Array.isArray(depthBins.errors) || depthBins.errors.length !== 0) {
    reasons.push('depth-bin validation errors is not empty');
  }
  requireEqual(depthBins.binCount, DEPTH_BIN_COUNT, 'depth-bin count', reasons);
  requireEqual(depthBins.order, protocol.order, 'depth-bin order', reasons);
  requireExactArray(depthBins.traversal, protocol.traversal, 'depth-bin traversal', reasons);

  const sequence = depthBins.physicalBinSequenceCommitment;
  if (exactKeys(sequence, [
    'schemaVersion',
    'hashAlgorithm',
    'encoding',
    'bucketCount',
    'binCount',
    'recordCount',
    'survivorCount',
    'sha256',
  ], 'physical-bin sequence commitment', reasons)) {
    requireEqual(sequence.schemaVersion, 1, 'physical-bin sequence schemaVersion', reasons);
    requireEqual(sequence.hashAlgorithm, 'sha256', 'physical-bin sequence hash algorithm', reasons);
    requireEqual(
      sequence.encoding,
      'bucket-major-physical-bin-major-tagged-uint32-little-endian',
      'physical-bin sequence encoding',
      reasons,
    );
    requireEqual(sequence.bucketCount, bucketCount, 'physical-bin sequence bucket count', reasons);
    requireEqual(sequence.binCount, DEPTH_BIN_COUNT, 'physical-bin sequence bin count', reasons);
    requireEqual(
      sequence.recordCount,
      bucketCount * DEPTH_BIN_COUNT,
      'physical-bin sequence record count',
      reasons,
    );
    requireEqual(
      sequence.survivorCount,
      expectedVisibleCount,
      'physical-bin sequence survivor count',
      reasons,
    );
    requireSha256(sequence.sha256, 'physical-bin sequence sha256', reasons);
  }

  const binRecordCount = bucketCount * DEPTH_BIN_COUNT;
  const arrayFields = [
    ['expectedCounts', binRecordCount],
    ['actualCounts', binRecordCount],
    ['expectedStarts', binRecordCount],
    ['actualStarts', binRecordCount],
    ['expectedBucketTotals', bucketCount],
    ['commandCounts', bucketCount],
  ];
  let arraysWellFormed = true;
  for (const [field, expectedLength] of arrayFields) {
    const value = depthBins[field];
    if (!Array.isArray(value) || value.length !== expectedLength) {
      reasons.push(`depth-bin ${field} has the wrong length`);
      arraysWellFormed = false;
      continue;
    }
    for (let index = 0; index < value.length; index += 1) {
      if (!isNonnegativeInteger(value[index]) || value[index] > 0xffff_ffff) {
        reasons.push(`depth-bin ${field}[${index}] is not a uint32`);
        arraysWellFormed = false;
      }
    }
  }
  if (!arraysWellFormed) return;

  let aggregateCount = 0;
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    let cursor = 0;
    for (const physicalBin of protocol.traversal) {
      const binIndex = bucket * DEPTH_BIN_COUNT + physicalBin;
      requireEqual(
        depthBins.actualCounts[binIndex],
        depthBins.expectedCounts[binIndex],
        `depth-bin bucket ${bucket} bin ${physicalBin} actual count`,
        reasons,
      );
      requireEqual(
        depthBins.expectedStarts[binIndex],
        cursor,
        `depth-bin bucket ${bucket} bin ${physicalBin} expected start`,
        reasons,
      );
      requireEqual(
        depthBins.actualStarts[binIndex],
        cursor,
        `depth-bin bucket ${bucket} bin ${physicalBin} actual start`,
        reasons,
      );
      cursor += depthBins.expectedCounts[binIndex];
      if (cursor > 0xffff_ffff) {
        reasons.push(`depth-bin bucket ${bucket} count exceeds uint32 capacity`);
      }
    }
    requireEqual(
      depthBins.expectedBucketTotals[bucket],
      cursor,
      `depth-bin bucket ${bucket} recomputed total`,
      reasons,
    );
    requireEqual(
      depthBins.commandCounts[bucket],
      cursor,
      `depth-bin bucket ${bucket} command total`,
      reasons,
    );
    requireEqual(
      cursor,
      membershipDigests?.perBucket?.[bucket]?.expected?.count,
      `depth-bin bucket ${bucket} membership total`,
      reasons,
    );
    requireEqual(
      cursor,
      commandValidation?.records?.[bucket]?.actual?.instanceCount,
      `depth-bin bucket ${bucket} native-command total`,
      reasons,
    );
    aggregateCount += cursor;
  }
  requireEqual(aggregateCount, expectedVisibleCount, 'depth-bin aggregate count', reasons);
}

export function physicalBinSequenceIdentity(validation) {
  const sha256 = validation?.kind === 'frozen-depth-crossover-exact-paired-snapshots'
    ? validation?.physicalBinSequenceSha256
    : validation?.depthBins?.physicalBinSequenceCommitment?.sha256;
  return typeof sha256 === 'string' && /^[0-9a-f]{64}$/.test(sha256)
    ? sha256
    : null;
}

function validateDepthRepresentation(representation, {
  modeId,
  objectCount,
  bucketCount,
  scenarioManifest,
}, reasons) {
  const protocol = depthModeProtocol(modeId);
  if (!protocol) return;
  if (!exactKeys(representation, [
    'kind',
    'depthBinCount',
    'depthOrder',
    'binTraversal',
    'depthBinRange',
    'reverseOrderUniformValue',
    'bundleRecordCallbackCount',
    'meshCount',
    'geometryIdentityCount',
    'materialIdentityCount',
    'commandCount',
    'zeroFirstInstanceCount',
    'computeDispatchCount',
    'computeDispatchWorkItems',
  ], 'depth-binned representation', reasons)) return;

  requireEqual(
    representation.kind,
    'single-merged-geometry-depth-binned-fixed-slice',
    'depth-binned representation kind',
    reasons,
  );
  requireEqual(representation.depthBinCount, DEPTH_BIN_COUNT, 'representation depth-bin count', reasons);
  requireEqual(representation.depthOrder, protocol.order, 'representation depth order', reasons);
  requireExactArray(
    representation.binTraversal,
    protocol.traversal,
    'representation bin traversal',
    reasons,
  );
  requireEqual(
    representation.reverseOrderUniformValue,
    protocol.reverseOrderUniformValue,
    'representation reverse-order uniform',
    reasons,
  );

  const signedRange = scenarioManifest?.depthBinRange;
  if (!isRecord(scenarioManifest)) {
    reasons.push('depth-binned validation scenario manifest is missing');
  } else if (!isRecord(signedRange)) {
    reasons.push('depth-binned scenario manifest depthBinRange is missing');
  }
  if (exactKeys(
    representation.depthBinRange,
    ['near', 'far'],
    'depth-binned representation depthBinRange',
    reasons,
  )) {
    if (!Number.isFinite(representation.depthBinRange.near)
      || !Number.isFinite(representation.depthBinRange.far)
      || representation.depthBinRange.far <= representation.depthBinRange.near) {
      reasons.push('depth-binned representation depthBinRange is not a finite increasing range');
    }
    requireEqual(
      representation.depthBinRange.near,
      signedRange?.near,
      'representation depth range near versus scenario manifest',
      reasons,
    );
    requireEqual(
      representation.depthBinRange.far,
      signedRange?.far,
      'representation depth range far versus scenario manifest',
      reasons,
    );
  }

  requireEqual(representation.bundleRecordCallbackCount, 1, 'depth-binned bundle-record callback count', reasons);
  requireEqual(representation.meshCount, 1, 'depth-binned mesh count', reasons);
  requireEqual(representation.geometryIdentityCount, 1, 'depth-binned geometry identity count', reasons);
  requireEqual(representation.materialIdentityCount, 1, 'depth-binned material identity count', reasons);
  requireEqual(representation.commandCount, bucketCount, 'depth-binned command count', reasons);
  requireEqual(
    representation.zeroFirstInstanceCount,
    bucketCount,
    'depth-binned zero-first-instance count',
    reasons,
  );
  requireEqual(representation.computeDispatchCount, 4, 'depth-binned compute dispatch count', reasons);
  requireExactArray(
    representation.computeDispatchWorkItems,
    [bucketCount * DEPTH_BIN_COUNT, objectCount, bucketCount, bucketCount],
    'depth-binned compute dispatch work items',
    reasons,
  );
}

function validateFrozenLaneOffsets(laneOffsets, laneStorageOrder, objectCount, label, reasons) {
  if (!exactKeys(laneOffsets, FROZEN_LOGICAL_LANES, label, reasons)) return;
  const otherLane = FROZEN_LOGICAL_LANES.find((lane) => lane !== laneStorageOrder);
  requireEqual(laneOffsets[laneStorageOrder], 0, `${label}.${laneStorageOrder}`, reasons);
  requireEqual(laneOffsets[otherLane], objectCount, `${label}.${otherLane}`, reasons);
}

function validateFrozenIdentityDiagnostics(record, label, reasons) {
  for (const field of [
    'bundleGroupUuid',
    'meshUuid',
    'geometryUuid',
    'materialUuid',
    'selectorUniformUuid',
  ]) {
    if (typeof record?.[field] !== 'string' || record[field].length === 0) {
      reasons.push(`${label}.${field} is not a nonempty identity string`);
    }
  }
  for (const field of [
    'matrixAttributeId',
    'visibleIdsAttributeId',
    'indirectAttributeId',
    'selectorChallengeAttributeId',
    'bundleGroupVersion',
    'matrixAttributeVersion',
    'visibleIdsAttributeVersion',
    'indirectAttributeVersion',
  ]) {
    if (!isNonnegativeInteger(record?.[field])) {
      reasons.push(`${label}.${field} is not a nonnegative integer`);
    }
  }
}

function validateFrozenRepresentation(validation, {
  objectCount,
  bucketCount,
}, reasons) {
  const representation = validation.representation;
  const lifecycle = validation.lifecycle;
  const sharedIdentityFields = [
    'bundleGroupUuid',
    'meshUuid',
    'geometryUuid',
    'materialUuid',
    'matrixAttributeId',
    'visibleIdsAttributeId',
    'indirectAttributeId',
    'selectorChallengeAttributeId',
    'bundleGroupVersion',
    'matrixAttributeVersion',
    'visibleIdsAttributeVersion',
    'indirectAttributeVersion',
    'selectorUniformUuid',
  ];
  const representationKeys = [
    'kind',
    'laneStorageOrder',
    'laneOffsets',
    'activeLane',
    'activeVisibleIdOffset',
    'visibleIdsCount',
    'visibleIdSegmentLength',
    'depthBinCount',
    'bundleGroupStatic',
    'bundleRecordCallbackCount',
    'meshCount',
    'geometryIdentityCount',
    'materialIdentityCount',
    'commandCount',
    'zeroFirstInstanceCount',
    'configuredComputeDispatches',
    'configuredComputeSubmissions',
    'diagnosticSelectorDispatchesPerValidation',
    ...sharedIdentityFields,
  ];
  const lifecycleKeys = [
    'kind',
    'laneStorageOrder',
    'laneOffsets',
    'activeLane',
    'activeVisibleIdOffset',
    'bundleGroupStatic',
    'bundleRecordCallbackCount',
    'meshCount',
    'geometryIdentityCount',
    'materialIdentityCount',
    'configuredComputeDispatches',
    'configuredComputeSubmissions',
    ...sharedIdentityFields,
  ];
  if (!exactKeys(
    representation,
    representationKeys,
    'frozen crossover representation',
    reasons,
  )) return;
  if (!exactKeys(
    lifecycle,
    lifecycleKeys,
    'frozen crossover lifecycle',
    reasons,
  )) return;

  requireEqual(
    representation.kind,
    'single-render-object-frozen-depth-crossover',
    'frozen crossover representation kind',
    reasons,
  );
  requireEqual(
    lifecycle.kind,
    'frozen-depth-crossover-static-bundle-lifecycle',
    'frozen crossover lifecycle kind',
    reasons,
  );
  for (const record of [representation, lifecycle]) {
    requireEqual(
      record.laneStorageOrder,
      validation.laneStorageOrder,
      `${record === representation ? 'representation' : 'lifecycle'} laneStorageOrder`,
      reasons,
    );
    validateFrozenLaneOffsets(
      record.laneOffsets,
      validation.laneStorageOrder,
      objectCount,
      `${record === representation ? 'representation' : 'lifecycle'} lane offsets`,
      reasons,
    );
    requireEqual(record.activeLane, validation.activeLane, 'frozen active lane', reasons);
    requireEqual(
      record.activeVisibleIdOffset,
      validation.activeVisibleIdOffset,
      'frozen active visible-ID offset',
      reasons,
    );
    requireEqual(record.bundleGroupStatic, true, 'frozen static BundleGroup', reasons);
    requireEqual(record.bundleRecordCallbackCount, 1, 'frozen bundle-record callback count', reasons);
    requireEqual(record.meshCount, 1, 'frozen mesh count', reasons);
    requireEqual(record.geometryIdentityCount, 1, 'frozen geometry identity count', reasons);
    requireEqual(record.materialIdentityCount, 1, 'frozen material identity count', reasons);
    requireEqual(record.configuredComputeDispatches, 0, 'frozen configured compute dispatches', reasons);
    requireEqual(record.configuredComputeSubmissions, 0, 'frozen configured compute submissions', reasons);
    validateFrozenIdentityDiagnostics(
      record,
      record === representation ? 'frozen representation' : 'frozen lifecycle',
      reasons,
    );
  }
  for (const field of sharedIdentityFields) {
    requireEqual(
      lifecycle[field],
      representation[field],
      `frozen lifecycle ${field} versus representation`,
      reasons,
    );
  }
  requireEqual(representation.visibleIdsCount, objectCount * 2, 'frozen visible-ID count', reasons);
  requireEqual(
    representation.visibleIdSegmentLength,
    objectCount,
    'frozen visible-ID segment length',
    reasons,
  );
  requireEqual(representation.depthBinCount, DEPTH_BIN_COUNT, 'frozen depth-bin count', reasons);
  requireEqual(representation.commandCount, bucketCount, 'frozen command count', reasons);
  requireEqual(
    representation.zeroFirstInstanceCount,
    bucketCount,
    'frozen zero-first-instance count',
    reasons,
  );
  requireEqual(
    representation.diagnosticSelectorDispatchesPerValidation,
    2,
    'frozen diagnostic selector dispatch count',
    reasons,
  );
}

export function validateFrozenCrossoverValidation(validation, {
  objectCount,
  bucketCount,
  expectedVisibleCount,
  expectedVisibleIdsCanonicalSha256,
  geometryManifest,
  scenarioManifest = null,
  laneStorageOrder = null,
} = {}) {
  const reasons = [];
  if (!isRecord(validation)) {
    return { rejectionReasons: ['frozen crossover validation is not an object'], semanticSha256: null };
  }
  exactKeys(validation, [
    'pass',
    'kind',
    'expectedIdsMatchScenario',
    'visibleIdsByteLength',
    'visibleIdsSha256',
    'expectedVisibleIdsSha256',
    'visibleIdsExactPackingMatch',
    'commandSha256',
    'laneStorageOrder',
    'laneOffsets',
    'activeLane',
    'activeVisibleIdOffset',
    'physicalBinSequenceCommitmentsEqual',
    'rawLaneSequencesDiffer',
    'physicalBinSequenceSha256',
    'commandValidation',
    'lanes',
    'selectorChallenges',
    'representation',
    'lifecycle',
  ], 'frozen crossover exact validation', reasons);
  requireEqual(validation.pass, true, 'frozen crossover exact validation pass', reasons);
  requireEqual(
    validation.kind,
    'frozen-depth-crossover-exact-paired-snapshots',
    'frozen crossover exact validation kind',
    reasons,
  );
  requireEqual(validation.expectedIdsMatchScenario, true, 'frozen expected-ID scenario match', reasons);
  if (!isRecord(scenarioManifest)) {
    reasons.push('frozen crossover scenario manifest is missing');
  } else {
    requireEqual(
      scenarioManifest.expectedVisibleCount,
      expectedVisibleCount,
      'frozen crossover scenario expected visible count',
      reasons,
    );
    requireEqual(
      scenarioManifest.expectedVisibleIdsCanonicalSha256,
      expectedVisibleIdsCanonicalSha256,
      'frozen crossover scenario expected visible-ID digest',
      reasons,
    );
    if (!isRecord(scenarioManifest.depthBinRange)
      || !Number.isFinite(scenarioManifest.depthBinRange.near)
      || !Number.isFinite(scenarioManifest.depthBinRange.far)
      || scenarioManifest.depthBinRange.far <= scenarioManifest.depthBinRange.near) {
      reasons.push('frozen crossover scenario depthBinRange is missing or invalid');
    }
  }
  requireEqual(
    validation.visibleIdsByteLength,
    objectCount * 2 * Uint32Array.BYTES_PER_ELEMENT,
    'frozen visible-ID byte length',
    reasons,
  );
  requireSha256(validation.visibleIdsSha256, 'frozen visible-ID sha256', reasons);
  requireSha256(validation.expectedVisibleIdsSha256, 'frozen expected visible-ID sha256', reasons);
  requireEqual(
    validation.visibleIdsSha256,
    validation.expectedVisibleIdsSha256,
    'frozen visible-ID packing digest',
    reasons,
  );
  requireEqual(validation.visibleIdsExactPackingMatch, true, 'frozen exact packing match', reasons);
  requireSha256(validation.commandSha256, 'frozen command sha256', reasons);
  requireEqual(
    FROZEN_LOGICAL_LANES.includes(validation.laneStorageOrder),
    true,
    'frozen laneStorageOrder support',
    reasons,
  );
  if (laneStorageOrder !== null) {
    const expectedStorageOrder = Array.isArray(laneStorageOrder)
      ? frozenLogicalLane(laneStorageOrder[0])
      : laneStorageOrder;
    requireEqual(
      validation.laneStorageOrder,
      expectedStorageOrder,
      'frozen laneStorageOrder versus plan',
      reasons,
    );
  }
  validateFrozenLaneOffsets(
    validation.laneOffsets,
    validation.laneStorageOrder,
    objectCount,
    'frozen lane offsets',
    reasons,
  );
  if (!FROZEN_LOGICAL_LANES.includes(validation.activeLane)) {
    reasons.push('frozen active lane is unsupported');
  } else {
    requireEqual(
      validation.activeVisibleIdOffset,
      validation.laneOffsets?.[validation.activeLane],
      'frozen active lane offset',
      reasons,
    );
  }
  requireEqual(
    validation.physicalBinSequenceCommitmentsEqual,
    true,
    'frozen physical-bin sequence commitment equality',
    reasons,
  );
  requireEqual(validation.rawLaneSequencesDiffer, true, 'frozen raw lane sequence difference', reasons);
  requireSha256(
    validation.physicalBinSequenceSha256,
    'frozen physical-bin sequence sha256',
    reasons,
  );

  if (!exactKeys(validation.lanes, FROZEN_LOGICAL_LANES, 'frozen lane validations', reasons)) {
    // exactKeys records the structural rejection.
  } else {
    for (const logicalLane of FROZEN_LOGICAL_LANES) {
      const lane = validation.lanes[logicalLane];
      const laneModeId = frozenLaneModeId(logicalLane);
      const label = `frozen ${logicalLane} lane`;
      if (!exactKeys(lane, [
        'pass',
        'order',
        'storageOffset',
        'traversal',
        'membership',
        'membershipDigests',
        'depthBins',
        'storageSegmentSha256',
        'paddingSentinelCount',
        'paddingCorruptionCount',
      ], label, reasons)) continue;
      requireEqual(lane.pass, true, `${label} pass`, reasons);
      requireEqual(lane.order, logicalLane, `${label} order`, reasons);
      requireEqual(
        lane.storageOffset,
        validation.laneOffsets?.[logicalLane],
        `${label} storage offset`,
        reasons,
      );
      requireExactArray(
        lane.traversal,
        DEPTH_MODE_PROTOCOLS[laneModeId].traversal,
        `${label} traversal`,
        reasons,
      );
      requireSha256(lane.storageSegmentSha256, `${label} storage segment sha256`, reasons);
      requireEqual(
        lane.paddingSentinelCount,
        objectCount - expectedVisibleCount,
        `${label} padding sentinel count`,
        reasons,
      );
      requireEqual(lane.paddingCorruptionCount, 0, `${label} padding corruption count`, reasons);

      const membershipCheck = validateExactValidation({
        pass: lane.pass,
        kind: 'fixed-slice-exact-membership',
        membership: lane.membership,
        membershipDigests: lane.membershipDigests,
        commandValidation: validation.commandValidation,
        overflow: lane.membership?.overflow,
      }, {
        modeId: 'fixed-slice',
        objectCount,
        bucketCount,
        expectedVisibleCount,
        expectedVisibleIdsCanonicalSha256,
        geometryManifest,
        scenarioManifest,
      });
      reasons.push(...membershipCheck.rejectionReasons.map((reason) => `${label}: ${reason}`));
      validateDepthBinEvidence(lane.depthBins, {
        modeId: laneModeId,
        bucketCount,
        expectedVisibleCount,
        membershipDigests: lane.membershipDigests,
        commandValidation: validation.commandValidation,
      }, reasons);
      requireEqual(
        lane.depthBins?.physicalBinSequenceCommitment?.sha256,
        validation.physicalBinSequenceSha256,
        `${label} physical-bin sequence commitment`,
        reasons,
      );
    }
    requireEqual(
      validation.lanes['front-to-back']?.storageSegmentSha256
        === validation.lanes.reverse?.storageSegmentSha256,
      false,
      'frozen raw lane segment digest equality',
      reasons,
    );
  }

  if (!exactKeys(
    validation.selectorChallenges,
    FROZEN_LOGICAL_LANES,
    'frozen selector challenges',
    reasons,
  )) {
    // exactKeys records the structural rejection.
  } else {
    for (const logicalLane of FROZEN_LOGICAL_LANES) {
      const challenge = validation.selectorChallenges[logicalLane];
      const label = `frozen ${logicalLane} selector challenge`;
      if (!exactKeys(challenge, [
        'pass',
        'kind',
        'lane',
        'storageOffset',
        'elementCount',
        'sha256',
        'expectedSha256',
      ], label, reasons)) continue;
      requireEqual(challenge.pass, true, `${label} pass`, reasons);
      requireEqual(challenge.kind, 'gpu-selector-address-challenge', `${label} kind`, reasons);
      requireEqual(challenge.lane, logicalLane, `${label} lane`, reasons);
      requireEqual(
        challenge.storageOffset,
        validation.laneOffsets?.[logicalLane],
        `${label} storage offset`,
        reasons,
      );
      requireEqual(challenge.elementCount, objectCount, `${label} element count`, reasons);
      requireSha256(challenge.sha256, `${label} sha256`, reasons);
      requireEqual(challenge.sha256, challenge.expectedSha256, `${label} expected digest`, reasons);
      requireEqual(
        challenge.sha256,
        validation.lanes?.[logicalLane]?.storageSegmentSha256,
        `${label} lane segment digest`,
        reasons,
      );
    }
  }
  validateFrozenRepresentation(validation, { objectCount, bucketCount }, reasons);

  return {
    rejectionReasons: [...new Set(reasons)],
    semanticSha256: sha256Json(validation),
  };
}

export function validateExactValidation(validation, {
  modeId,
  objectCount,
  bucketCount,
  expectedVisibleCount,
  expectedVisibleIdsCanonicalSha256,
  geometryManifest,
  scenarioManifest = null,
  laneStorageOrder = null,
} = {}) {
  if (modeId === FROZEN_DEPTH_CROSSOVER_MODE) {
    return validateFrozenCrossoverValidation(validation, {
      objectCount,
      bucketCount,
      expectedVisibleCount,
      expectedVisibleIdsCanonicalSha256,
      geometryManifest,
      scenarioManifest,
      laneStorageOrder,
    });
  }
  const reasons = [];
  const shape = expectedStrategyShape(modeId, bucketCount, null);
  if (!shape) {
    return {
      rejectionReasons: [`unsupported modeId ${JSON.stringify(modeId)}`],
      semanticSha256: null,
    };
  }
  if (!isRecord(validation)) {
    return { rejectionReasons: ['exact validation is not an object'], semanticSha256: null };
  }
  requireEqual(validation.pass, true, 'exact validation pass', reasons);
  requireEqual(validation.kind, shape.kind, 'exact validation kind', reasons);
  if (modeId === 'draw-all') {
    requireEqual(
      validation.expectedVisibleCount,
      expectedVisibleCount,
      'draw-all expectedVisibleCount',
      reasons,
    );
    return { rejectionReasons: [...new Set(reasons)], semanticSha256: sha256Json({
      kind: validation.kind,
      pass: validation.pass,
      expectedVisibleCount: validation.expectedVisibleCount,
    }) };
  }

  const membership = validation.membership;
  if (!isRecord(membership)) {
    reasons.push('membership diagnostics are missing');
  } else {
    requireEqual(membership.pass, true, 'membership pass', reasons);
    requireEqual(membership.expectedCount, expectedVisibleCount, 'membership expectedCount', reasons);
    requireEqual(membership.listedCount, expectedVisibleCount, 'membership listedCount', reasons);
    for (const name of [
      'duplicateIds',
      'outOfRangeIds',
      'wrongBucketIds',
      'listedHiddenIds',
      'missingVisibleIds',
      'overflow',
      'errors',
    ]) requireEqual(membership[name], 0, `membership ${name}`, reasons);
  }

  const digests = validation.membershipDigests;
  if (!isRecord(digests)) {
    reasons.push('membership digest evidence is missing');
  } else {
    requireEqual(digests.schemaVersion, 1, 'membership digest schemaVersion', reasons);
    requireEqual(digests.hashAlgorithm, 'sha256', 'membership digest hashAlgorithm', reasons);
    requireEqual(digests.encoding, 'sorted-uint32-little-endian', 'membership digest encoding', reasons);
    requireEqual(digests.pass, true, 'membership digest pass', reasons);
    requireEqual(digests.invalidExpectedIds, 0, 'membership digest invalidExpectedIds', reasons);
    requireEqual(digests.truncatedActualIds, 0, 'membership digest truncatedActualIds', reasons);
    for (const side of ['expected', 'actual']) {
      if (!isRecord(digests[side])) {
        reasons.push(`membership digest ${side} is missing`);
        continue;
      }
      requireEqual(digests[side].count, expectedVisibleCount, `membership digest ${side}.count`, reasons);
      requireSha256(digests[side].sha256, `membership digest ${side}.sha256`, reasons);
      requireEqual(
        digests[side].sha256,
        expectedVisibleIdsCanonicalSha256,
        `membership digest ${side}.sha256`,
        reasons,
      );
    }
    if (!Array.isArray(digests.perBucket) || digests.perBucket.length !== bucketCount) {
      reasons.push('membership digest perBucket has the wrong length');
    } else {
      let aggregateBucketCount = 0;
      for (let bucket = 0; bucket < bucketCount; bucket += 1) {
        const record = digests.perBucket[bucket];
        const label = `membership digest bucket ${bucket}`;
        if (!isRecord(record)) {
          reasons.push(`${label} is missing`);
          continue;
        }
        requireEqual(record.bucket, bucket, `${label}.bucket`, reasons);
        requireEqual(record.match, true, `${label}.match`, reasons);
        if (!isRecord(record.expected) || !isRecord(record.actual)) {
          reasons.push(`${label} expected or actual digest is missing`);
          continue;
        }
        if (!isNonnegativeInteger(record.expected.count)) reasons.push(`${label}.expected.count is invalid`);
        requireEqual(record.actual.count, record.expected.count, `${label}.actual.count`, reasons);
        requireEqual(record.actual.declaredCount, record.expected.count, `${label}.actual.declaredCount`, reasons);
        requireSha256(record.expected.sha256, `${label}.expected.sha256`, reasons);
        requireEqual(record.actual.sha256, record.expected.sha256, `${label}.actual.sha256`, reasons);
        aggregateBucketCount += Number.isInteger(record.expected.count) ? record.expected.count : 0;
      }
      requireEqual(aggregateBucketCount, expectedVisibleCount, 'membership digest bucket-count sum', reasons);
    }
  }

  const command = validation.commandValidation;
  if (!isRecord(command)) {
    reasons.push('native command validation is missing');
  } else {
    requireEqual(command.pass, true, 'native command validation pass', reasons);
    if (!Array.isArray(command.errors) || command.errors.length !== 0) {
      reasons.push('native command validation errors is not empty');
    }
    requireEqual(command.commandCount, bucketCount, 'native command count', reasons);
    if (!Array.isArray(command.records) || command.records.length !== bucketCount) {
      reasons.push('native command records have the wrong length');
    } else {
      let totalInstances = 0;
      let cumulativeFirstIndex = 0;
      const historicalRanges = [];
      for (let bucket = 0; bucket < bucketCount; bucket += 1) {
        const record = command.records[bucket];
        const geometry = geometryManifest?.geometries?.[bucket];
        const label = `native command bucket ${bucket}`;
        if (!isRecord(record) || !isRecord(record.actual) || !isRecord(record.expected)) {
          reasons.push(`${label} record is malformed`);
          continue;
        }
        requireEqual(record.bucket, bucket, `${label}.bucket`, reasons);
        const expectedFirstIndex = modeId === 'fixed-slice'
          || modeId === 'fixed-slice-per-bucket'
          || modeId === 'fixed-slice-depth-front-to-back'
          || modeId === 'fixed-slice-depth-reverse'
          || modeId === 'three-blocks-historical'
          ? cumulativeFirstIndex
          : 0;
        requireEqual(record.expected.indexCount, geometry?.index?.count, `${label}.expected.indexCount`, reasons);
        requireEqual(record.actual.indexCount, record.expected.indexCount, `${label}.actual.indexCount`, reasons);
        if (!isNonnegativeInteger(record.expected.instanceCount)) {
          reasons.push(`${label}.expected.instanceCount is invalid`);
        }
        requireEqual(record.actual.instanceCount, record.expected.instanceCount, `${label}.actual.instanceCount`, reasons);
        requireEqual(
          record.actual.instanceCount,
          digests?.perBucket?.[bucket]?.expected?.count,
          `${label}.instanceCount versus membership digest`,
          reasons,
        );
        requireEqual(record.expected.firstIndex, expectedFirstIndex, `${label}.expected.firstIndex`, reasons);
        requireEqual(record.actual.firstIndex, expectedFirstIndex, `${label}.actual.firstIndex`, reasons);
        requireEqual(record.expected.baseVertex, 0, `${label}.expected.baseVertex`, reasons);
        requireEqual(record.actual.baseVertex, 0, `${label}.actual.baseVertex`, reasons);
        if (modeId === 'three-blocks-historical') {
          requireEqual(record.expected.firstInstance, null, `${label}.expected.firstInstance`, reasons);
          if (!isNonnegativeInteger(record.actual.firstInstance)) {
            reasons.push(`${label}.actual.firstInstance is invalid`);
          } else if (isNonnegativeInteger(record.actual.instanceCount)) {
            historicalRanges.push({
              bucket,
              start: record.actual.firstInstance,
              end: record.actual.firstInstance + record.actual.instanceCount,
            });
          }
        } else {
          requireEqual(record.expected.firstInstance, 0, `${label}.expected.firstInstance`, reasons);
          requireEqual(record.actual.firstInstance, 0, `${label}.actual.firstInstance`, reasons);
        }
        totalInstances += isNonnegativeInteger(record.actual.instanceCount) ? record.actual.instanceCount : 0;
        cumulativeFirstIndex += geometry?.index?.count ?? 0;
      }
      requireEqual(totalInstances, expectedVisibleCount, 'native command instance-count sum', reasons);
      if (modeId === 'three-blocks-historical') {
        historicalRanges.sort((left, right) => left.start - right.start || left.bucket - right.bucket);
        let cursor = 0;
        for (const range of historicalRanges) {
          requireEqual(range.start, cursor, `historical survivor range ${range.bucket} start`, reasons);
          cursor = Math.max(cursor, range.end);
        }
        requireEqual(cursor, expectedVisibleCount, 'historical survivor range coverage', reasons);
      }
    }
    const reportedTotal = command.totalInstanceCount ?? command.totalCommandSurvivors;
    requireEqual(reportedTotal, expectedVisibleCount, 'native command reported survivor total', reasons);
    if (modeId === 'three-blocks-historical') {
      requireEqual(
        command.survivorReadbackLength,
        expectedVisibleCount,
        'historical survivor readback length',
        reasons,
      );
    }
  }
  if ('readbackErrors' in validation
    && (!Array.isArray(validation.readbackErrors) || validation.readbackErrors.length !== 0)) {
    reasons.push('validation readbackErrors is not empty');
  }
  if (modeId === 'fixed-slice'
    || modeId === 'fixed-slice-per-bucket'
    || depthModeProtocol(modeId)) {
    requireEqual(validation.overflow, 0, `${modeId} overflow`, reasons);
  }
  if (modeId === 'fixed-slice-per-bucket') {
    const representation = validation.representation;
    if (!exactKeys(representation, [
      'kind',
      'bundleRecordCallbackCount',
      'geometryIdentityCount',
      'materialIdentityCount',
      'meshCount',
      'geometryInstanceCount',
    ], 'fixed-slice-per-bucket representation', reasons)) {
      // exactKeys records the structural rejection.
    } else {
      requireEqual(
        representation.kind,
        'shared-merged-geometry-per-bucket-render-objects',
        'fixed-slice-per-bucket representation kind',
        reasons,
      );
      requireEqual(
        representation.bundleRecordCallbackCount,
        bucketCount,
        'fixed-slice-per-bucket bundle-record callback count',
        reasons,
      );
      requireEqual(
        representation.geometryIdentityCount,
        1,
        'fixed-slice-per-bucket geometry identity count',
        reasons,
      );
      requireEqual(
        representation.materialIdentityCount,
        1,
        'fixed-slice-per-bucket material identity count',
        reasons,
      );
      requireEqual(
        representation.meshCount,
        bucketCount,
        'fixed-slice-per-bucket mesh count',
        reasons,
      );
      requireEqual(
        representation.geometryInstanceCount,
        Math.ceil(objectCount / bucketCount),
        'fixed-slice-per-bucket geometry instanceCount',
        reasons,
      );
    }
  }
  if (depthModeProtocol(modeId)) {
    validateDepthBinEvidence(validation.depthBins, {
      modeId,
      bucketCount,
      expectedVisibleCount,
      membershipDigests: digests,
      commandValidation: command,
    }, reasons);
    validateDepthRepresentation(validation.representation, {
      modeId,
      objectCount,
      bucketCount,
      scenarioManifest,
    }, reasons);
  }

  const semanticRecord = {
    kind: validation.kind,
    pass: validation.pass,
    membership,
    membershipDigests: digests,
    commands: isRecord(command) && Array.isArray(command.records)
      ? commandSemanticRecord(validation, modeId)
      : null,
    overflow: validation.overflow ?? null,
    readbackErrors: validation.readbackErrors ?? null,
    ...(depthModeProtocol(modeId) ? { depthBins: validation.depthBins ?? null } : {}),
    representation: validation.representation ?? null,
  };
  return {
    rejectionReasons: [...new Set(reasons)],
    semanticSha256: sha256Json(semanticRecord),
  };
}

function validateRenderParityChannel(record, label, expected, reasons) {
  if (!exactKeys(
    record,
    ['format', 'arrayType', 'byteLength', 'sha256'],
    label,
    reasons,
  )) return;
  requireEqual(record.format, expected.format, `${label}.format`, reasons);
  requireEqual(record.arrayType, expected.arrayType, `${label}.arrayType`, reasons);
  requireEqual(record.byteLength, RENDER_PARITY_BYTE_LENGTH, `${label}.byteLength`, reasons);
  requireSha256(record.sha256, `${label}.sha256`, reasons);
}

/**
 * Validate the depth-ordering matrix's exact offscreen parity capture. The
 * embedded exact validation is deliberately checked with the same fixture and
 * scenario manifests as the surrounding trial so a valid-looking capture
 * cannot be transplanted from another workload.
 */
export function validateRenderParity(parity, {
  spec,
  geometryManifest,
  scenarioManifest,
  skipSnapshotValidation = false,
} = {}) {
  if (spec?.modeId === FROZEN_DEPTH_CROSSOVER_MODE && !skipSnapshotValidation) {
    return validateFrozenCrossoverRenderParity(parity, {
      spec,
      geometryManifest,
      scenarioManifest,
    });
  }
  const reasons = [];
  if (!isRecord(parity)) return ['exact render-parity evidence is missing'];

  requireEqual(parity.schemaVersion, 1, 'exact render-parity schemaVersion', reasons);
  requireEqual(
    parity.kind,
    'fixed-camera-offscreen-exact-render-parity',
    'exact render-parity kind',
    reasons,
  );
  requireEqual(parity.pass, true, 'exact render-parity pass', reasons);
  requireEqual(parity.captures, 2, 'exact render-parity capture count', reasons);
  requireEqual(parity.width, RENDER_PARITY_WIDTH, 'exact render-parity width', reasons);
  requireEqual(parity.height, RENDER_PARITY_HEIGHT, 'exact render-parity height', reasons);
  requireEqual(
    parity.reversedDepthBuffer,
    true,
    'exact render-parity reversedDepthBuffer',
    reasons,
  );
  if (!isRecord(parity.material)) reasons.push('exact render-parity material record is missing');

  const stability = parity.stability;
  if (!isRecord(stability)) {
    reasons.push('exact render-parity stability record is missing');
  } else {
    requireEqual(stability.pass, true, 'exact render-parity stability pass', reasons);
    if (!exactKeys(
      stability.first,
      ['colorSha256', 'depthSha256', 'objectIdSha256'],
      'exact render-parity stability.first',
      reasons,
    )) {
      // exactKeys records the structural rejection.
    }
    if (!exactKeys(
      stability.firstCapture,
      ['color', 'depth', 'objectId'],
      'exact render-parity stability.firstCapture',
      reasons,
    )) {
      // exactKeys records the structural rejection.
    }
  }

  const channels = Object.freeze({
    color: Object.freeze({ format: 'rgba8unorm', arrayType: 'Uint8Array' }),
    depth: Object.freeze({ format: 'depth32float', arrayType: 'Float32Array' }),
    objectId: Object.freeze({
      format: 'rgba8unorm-object-id-plus-one',
      arrayType: 'Uint8Array',
    }),
  });
  for (const [name, expected] of Object.entries(channels)) {
    const label = `exact render-parity ${name}`;
    validateRenderParityChannel(parity[name], label, expected, reasons);

    const firstCapture = stability?.firstCapture?.[name];
    validateRenderParityChannel(
      firstCapture,
      `exact render-parity stability.firstCapture.${name}`,
      expected,
      reasons,
    );
    if (isRecord(parity[name]) && isRecord(firstCapture)) {
      for (const field of ['format', 'arrayType', 'byteLength', 'sha256']) {
        requireEqual(
          firstCapture[field],
          parity[name][field],
          `exact render-parity ${name} first-capture ${field}`,
          reasons,
        );
      }
    }
    requireEqual(
      stability?.first?.[`${name}Sha256`],
      parity[name]?.sha256,
      `exact render-parity ${name} stability digest`,
      reasons,
    );
  }

  const objectIdValidation = parity.objectIdValidation;
  if (exactKeys(objectIdValidation, [
    'pass',
    'encoding',
    'backgroundPixels',
    'coveredPixels',
    'outOfRangePixels',
    'nonVisiblePixels',
  ], 'exact render-parity objectIdValidation', reasons)) {
    requireEqual(objectIdValidation.pass, true, 'object-ID validation pass', reasons);
    requireEqual(
      objectIdValidation.encoding,
      'rgb24-object-id-plus-one-zero-background',
      'object-ID validation encoding',
      reasons,
    );
    if (!isNonnegativeInteger(objectIdValidation.backgroundPixels)) {
      reasons.push('object-ID validation backgroundPixels is not a nonnegative integer');
    }
    if (!isPositiveInteger(objectIdValidation.coveredPixels)) {
      reasons.push('object-ID validation coveredPixels is not a positive integer');
    }
    requireEqual(objectIdValidation.outOfRangePixels, 0, 'object-ID out-of-range pixels', reasons);
    requireEqual(objectIdValidation.nonVisiblePixels, 0, 'object-ID non-visible pixels', reasons);
    if (isNonnegativeInteger(objectIdValidation.backgroundPixels)
      && isNonnegativeInteger(objectIdValidation.coveredPixels)) {
      requireEqual(
        objectIdValidation.backgroundPixels + objectIdValidation.coveredPixels,
        RENDER_PARITY_PIXEL_COUNT,
        'object-ID classified pixel total',
        reasons,
      );
    }
  }

  if (!skipSnapshotValidation) {
    const snapshotCheck = validateExactValidation(parity.snapshotValidation, {
      modeId: spec?.modeId,
      objectCount: spec?.objectCount,
      bucketCount: spec?.bucketCount,
      expectedVisibleCount: scenarioManifest?.expectedVisibleCount,
      expectedVisibleIdsCanonicalSha256:
        scenarioManifest?.expectedVisibleIdsCanonicalSha256,
      geometryManifest,
      scenarioManifest,
      laneStorageOrder: spec?.laneStorageOrder ?? null,
    });
    reasons.push(...snapshotCheck.rejectionReasons.map(
      (reason) => `render-parity snapshot: ${reason}`,
    ));
  }

  return [...new Set(reasons)];
}

export function validateFrozenCrossoverRenderParity(parity, {
  spec,
  geometryManifest,
  scenarioManifest,
} = {}) {
  const reasons = [];
  if (!isRecord(parity)) return ['frozen crossover render-parity evidence is missing'];
  exactKeys(parity, [
    'schemaVersion',
    'kind',
    'pass',
    'laneIds',
    'crossLaneExact',
    'lanes',
    'snapshotValidation',
  ], 'frozen crossover render parity', reasons);
  requireEqual(parity.schemaVersion, 1, 'frozen crossover render-parity schemaVersion', reasons);
  requireEqual(
    parity.kind,
    'frozen-depth-crossover-exact-render-parity',
    'frozen crossover render-parity kind',
    reasons,
  );
  requireEqual(parity.pass, true, 'frozen crossover render-parity pass', reasons);
  requireExactArray(
    parity.laneIds,
    FROZEN_DEPTH_CROSSOVER_LANES,
    'frozen crossover parity lane IDs',
    reasons,
  );
  requireEqual(parity.crossLaneExact, true, 'frozen crossover cross-lane exactness', reasons);
  if (exactKeys(
    parity.lanes,
    FROZEN_DEPTH_CROSSOVER_LANES,
    'frozen crossover parity lanes',
    reasons,
  )) {
    for (const laneId of FROZEN_DEPTH_CROSSOVER_LANES) {
      const lane = parity.lanes[laneId];
      const label = `frozen crossover parity ${laneId}`;
      exactKeys(lane, [
        'schemaVersion',
        'kind',
        'pass',
        'width',
        'height',
        'captures',
        'material',
        'color',
        'depth',
        'objectId',
        'objectIdValidation',
        'reversedDepthBuffer',
        'stability',
      ], label, reasons);
      reasons.push(...validateRenderParity(lane, {
        spec,
        geometryManifest,
        scenarioManifest,
        skipSnapshotValidation: true,
      }).map((reason) => `${label}: ${reason}`));
    }
    const front = parity.lanes[FROZEN_DEPTH_CROSSOVER_LANES[0]];
    const reverse = parity.lanes[FROZEN_DEPTH_CROSSOVER_LANES[1]];
    for (const field of ['material', 'color', 'depth', 'objectId', 'objectIdValidation']) {
      requireEqual(
        sha256Json(reverse?.[field]),
        sha256Json(front?.[field]),
        `frozen crossover parity cross-lane ${field}`,
        reasons,
      );
    }
    requireEqual(
      reverse?.reversedDepthBuffer,
      front?.reversedDepthBuffer,
      'frozen crossover parity cross-lane reversedDepthBuffer',
      reasons,
    );
  }

  const snapshotCheck = validateExactValidation(parity.snapshotValidation, {
    modeId: FROZEN_DEPTH_CROSSOVER_MODE,
    objectCount: spec?.objectCount,
    bucketCount: spec?.bucketCount,
    expectedVisibleCount: scenarioManifest?.expectedVisibleCount,
    expectedVisibleIdsCanonicalSha256:
      scenarioManifest?.expectedVisibleIdsCanonicalSha256,
    geometryManifest,
    scenarioManifest,
    laneStorageOrder: spec?.laneStorageOrder ?? null,
  });
  reasons.push(...snapshotCheck.rejectionReasons.map(
    (reason) => `frozen crossover render-parity snapshot: ${reason}`,
  ));
  return [...new Set(reasons)];
}

export function renderParityIdentity(parity) {
  if (parity?.kind === 'frozen-depth-crossover-exact-render-parity') {
    return sha256Json({
      schemaVersion: parity.schemaVersion,
      kind: parity.kind,
      laneIds: parity.laneIds,
      crossLaneExact: parity.crossLaneExact,
      lanes: Object.fromEntries(FROZEN_DEPTH_CROSSOVER_LANES.map((laneId) => {
        const lane = parity?.lanes?.[laneId];
        return [laneId, {
          width: lane?.width,
          height: lane?.height,
          reversedDepthBuffer: lane?.reversedDepthBuffer,
          material: lane?.material,
          color: lane?.color,
          depth: lane?.depth,
          objectId: lane?.objectId,
          objectIdValidation: lane?.objectIdValidation,
        }];
      })),
      physicalBinSequenceSha256:
        parity?.snapshotValidation?.physicalBinSequenceSha256 ?? null,
      membershipSha256: Object.fromEntries(FROZEN_LOGICAL_LANES.map((lane) => [
        lane,
        parity?.snapshotValidation?.lanes?.[lane]?.membershipDigests?.actual?.sha256 ?? null,
      ])),
    });
  }
  return sha256Json({
    width: parity?.width,
    height: parity?.height,
    reversedDepthBuffer: parity?.reversedDepthBuffer,
    material: parity?.material,
    color: parity?.color,
    depth: parity?.depth,
    objectId: parity?.objectId,
    objectIdValidation: parity?.objectIdValidation,
    membershipSha256:
      parity?.snapshotValidation?.membershipDigests?.actual?.sha256 ?? null,
  });
}

/**
 * Bind depth-ordering protocol metadata to the renderer facts recorded by the
 * page. Other benchmark matrices intentionally retain their legacy contract.
 */
export function validateBenchmarkProtocolEnvironment(protocol, pageEnvironment) {
  const frozenCrossover = protocol?.matrixKind === 'depth-ordering-render-only';
  if (protocol?.matrixKind !== 'depth-ordering' && !frozenCrossover) return [];

  const reasons = [];
  requireEqual(
    protocol.reversedDepthBuffer,
    true,
    'depth-ordering protocol reversedDepthBuffer',
    reasons,
  );
  requireEqual(
    protocol.minimumStorageBuffersPerShaderStage,
    DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS,
    'depth-ordering protocol minimumStorageBuffersPerShaderStage',
    reasons,
  );
  if (!isRecord(pageEnvironment)) {
    reasons.push('depth-ordering benchmark page environment is missing');
  } else {
    if (frozenCrossover) {
      requireEqual(
        pageEnvironment.reversedDepth,
        true,
        'depth-ordering-render-only page reversedDepth',
        reasons,
      );
    }
    requireEqual(
      pageEnvironment.rendererReversedDepthBuffer,
      true,
      'depth-ordering page rendererReversedDepthBuffer',
      reasons,
    );
    if (!Number.isInteger(pageEnvironment.maxStorageBuffersPerShaderStage)
      || pageEnvironment.maxStorageBuffersPerShaderStage
        < DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS) {
      reasons.push(
        `depth-ordering page maxStorageBuffersPerShaderStage must be an integer >= ${DEPTH_ORDERING_MINIMUM_STORAGE_BUFFERS}`,
      );
    }
  }
  return [...new Set(reasons)];
}

function finiteNonnegative(value) {
  return Number.isFinite(value) && value >= 0;
}

function validateDepthCompletionInvariant(invariant, {
  modeId,
  objectCount,
  bucketCount,
  validation,
  scenarioManifest,
}, reasons) {
  const protocol = depthModeProtocol(modeId);
  if (!protocol) return;
  if (!exactKeys(invariant, [
    'pass',
    'kind',
    'depthBinCount',
    'depthOrder',
    'binTraversal',
    'depthBinRange',
    'reverseOrderUniformValue',
    'bundleRecordCallbackCountAtTimingStart',
    'bundleRecordCallbackCountAtTimingEnd',
    'meshCount',
    'geometryIdentityCount',
    'materialIdentityCount',
    'commandCount',
    'zeroFirstInstanceCount',
    'computeDispatchCount',
    'computeDispatchWorkItems',
  ], 'depth-binned completion invariant', reasons)) return;

  const representation = validation?.representation;
  requireEqual(invariant.pass, true, 'depth-binned completion invariant pass', reasons);
  requireEqual(
    invariant.kind,
    'depth-binned-static-bundle-invariant',
    'depth-binned completion invariant kind',
    reasons,
  );
  requireEqual(invariant.depthBinCount, DEPTH_BIN_COUNT, 'completion depth-bin count', reasons);
  requireEqual(invariant.depthOrder, protocol.order, 'completion depth order', reasons);
  requireExactArray(invariant.binTraversal, protocol.traversal, 'completion bin traversal', reasons);
  requireEqual(
    invariant.reverseOrderUniformValue,
    protocol.reverseOrderUniformValue,
    'completion reverse-order uniform',
    reasons,
  );
  if (exactKeys(
    invariant.depthBinRange,
    ['near', 'far'],
    'depth-binned completion depthBinRange',
    reasons,
  )) {
    requireEqual(
      invariant.depthBinRange.near,
      scenarioManifest?.depthBinRange?.near,
      'completion depth range near versus scenario manifest',
      reasons,
    );
    requireEqual(
      invariant.depthBinRange.far,
      scenarioManifest?.depthBinRange?.far,
      'completion depth range far versus scenario manifest',
      reasons,
    );
    requireEqual(
      invariant.depthBinRange.near,
      representation?.depthBinRange?.near,
      'completion depth range near versus validation',
      reasons,
    );
    requireEqual(
      invariant.depthBinRange.far,
      representation?.depthBinRange?.far,
      'completion depth range far versus validation',
      reasons,
    );
  }
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingStart,
    1,
    'depth-binned timing-start bundle-record callback count',
    reasons,
  );
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingStart,
    representation?.bundleRecordCallbackCount,
    'depth-binned timing-start versus validation bundle-record callback count',
    reasons,
  );
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingEnd,
    invariant.bundleRecordCallbackCountAtTimingStart,
    'depth-binned timing-end bundle-record callback count',
    reasons,
  );
  for (const field of ['meshCount', 'geometryIdentityCount', 'materialIdentityCount']) {
    requireEqual(invariant[field], 1, `completion ${field}`, reasons);
    requireEqual(invariant[field], representation?.[field], `completion ${field} versus validation`, reasons);
  }
  requireEqual(invariant.commandCount, bucketCount, 'completion command count', reasons);
  requireEqual(
    invariant.commandCount,
    representation?.commandCount,
    'completion command count versus validation',
    reasons,
  );
  requireEqual(
    invariant.zeroFirstInstanceCount,
    bucketCount,
    'completion zero-first-instance count',
    reasons,
  );
  requireEqual(
    invariant.zeroFirstInstanceCount,
    representation?.zeroFirstInstanceCount,
    'completion zero-first-instance count versus validation',
    reasons,
  );
  requireEqual(invariant.computeDispatchCount, 4, 'completion compute dispatch count', reasons);
  requireEqual(
    invariant.computeDispatchCount,
    representation?.computeDispatchCount,
    'completion compute dispatch count versus validation',
    reasons,
  );
  const expectedWorkItems = [
    bucketCount * DEPTH_BIN_COUNT,
    objectCount,
    bucketCount,
    bucketCount,
  ];
  requireExactArray(
    invariant.computeDispatchWorkItems,
    expectedWorkItems,
    'completion compute dispatch work items',
    reasons,
  );
  requireExactArray(
    invariant.computeDispatchWorkItems,
    representation?.computeDispatchWorkItems ?? [],
    'completion versus validation compute dispatch work items',
    reasons,
  );
}

function validateAtomicFixedSliceCompletionInvariant(invariant, reasons) {
  if (!exactKeys(invariant, [
    'pass',
    'kind',
    'bundleGroupStatic',
    'bundleRecordCallbackCountAtTimingStart',
    'bundleRecordCallbackCountAtTimingEnd',
    'meshCount',
    'geometryIdentityCount',
    'materialIdentityCount',
  ], 'atomic fixed-slice completion invariant', reasons)) return;

  requireEqual(invariant.pass, true, 'atomic fixed-slice completion invariant pass', reasons);
  requireEqual(
    invariant.kind,
    'atomic-fixed-slice-static-bundle-invariant',
    'atomic fixed-slice completion invariant kind',
    reasons,
  );
  requireEqual(invariant.bundleGroupStatic, true, 'atomic fixed-slice static BundleGroup', reasons);
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingStart,
    1,
    'atomic fixed-slice timing-start bundle-record callback count',
    reasons,
  );
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingEnd,
    invariant.bundleRecordCallbackCountAtTimingStart,
    'atomic fixed-slice timing-end bundle-record callback count',
    reasons,
  );
  requireEqual(invariant.meshCount, 1, 'atomic fixed-slice completion meshCount', reasons);
  requireEqual(
    invariant.geometryIdentityCount,
    1,
    'atomic fixed-slice completion geometryIdentityCount',
    reasons,
  );
  requireEqual(
    invariant.materialIdentityCount,
    1,
    'atomic fixed-slice completion materialIdentityCount',
    reasons,
  );
}

export function validateFrozenCrossoverCompletionInvariant(invariant, {
  objectCount,
  validation,
} = {}) {
  const reasons = [];
  const pairedIdentityFields = [
    'bundleGroupUuid',
    'meshUuid',
    'geometryUuid',
    'materialUuid',
    'matrixAttributeId',
    'visibleIdsAttributeId',
    'indirectAttributeId',
    'selectorChallengeAttributeId',
    'bundleGroupVersion',
    'matrixAttributeVersion',
    'visibleIdsAttributeVersion',
    'indirectAttributeVersion',
    'selectorUniformUuid',
    'renderTargetTextureUuid',
    'renderTargetWidth',
    'renderTargetHeight',
    'renderTargetSamples',
    'renderTargetDepthBuffer',
    'cameraViewFnv64',
    'cameraProjectionFnv64',
  ];
  const exactFields = [
    'pass',
    'kind',
    'bundleGroupStatic',
    'bundleRecordCallbackCountAtTimingStart',
    'bundleRecordCallbackCountAtTimingEnd',
    'meshCount',
    'geometryIdentityCount',
    'materialIdentityCount',
    'configuredComputeDispatches',
    'configuredComputeSubmissions',
    'frontLaneBase',
    'reverseLaneBase',
    ...pairedIdentityFields.flatMap((field) => [
      `${field}AtTimingStart`,
      `${field}AtTimingEnd`,
    ]),
    'selectorWriteSerialAtTimingStart',
    'selectorWriteSerialAtTimingEnd',
    'selectorWritesDuringTiming',
    'renderCallSerialAtTimingStart',
    'renderCallSerialAtTimingEnd',
    'renderCallsDuringTiming',
    'computeCallSerialAtTimingStart',
    'computeCallSerialAtTimingEnd',
    'computeCallsDuringTiming',
    'totalPipelineCacheEntriesAtTimingStart',
    'totalPipelineCacheEntriesAtTimingEnd',
    'computePipelineCacheEntriesAtTimingStart',
    'computePipelineCacheEntriesAtTimingEnd',
    'expectedTimedFrameCount',
  ];
  if (!exactKeys(
    invariant,
    exactFields,
    'frozen crossover completion invariant',
    reasons,
  )) return reasons;

  const expectedTimedFrameCount = FROZEN_CROSSOVER_WARMUP_FRAMES
    + FROZEN_CROSSOVER_MEASURED_FRAMES;
  const lifecycle = validation?.lifecycle;
  requireEqual(invariant.pass, true, 'frozen crossover completion invariant pass', reasons);
  requireEqual(
    invariant.kind,
    'frozen-depth-crossover-static-bundle-invariant',
    'frozen crossover completion invariant kind',
    reasons,
  );
  requireEqual(invariant.bundleGroupStatic, true, 'frozen completion static BundleGroup', reasons);
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingStart,
    1,
    'frozen completion timing-start bundle-record callback count',
    reasons,
  );
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingEnd,
    invariant.bundleRecordCallbackCountAtTimingStart,
    'frozen completion timing-end bundle-record callback count',
    reasons,
  );
  requireEqual(
    invariant.bundleRecordCallbackCountAtTimingStart,
    lifecycle?.bundleRecordCallbackCount,
    'frozen completion versus validation bundle-record callback count',
    reasons,
  );
  for (const field of ['meshCount', 'geometryIdentityCount', 'materialIdentityCount']) {
    requireEqual(invariant[field], 1, `frozen completion ${field}`, reasons);
    requireEqual(invariant[field], lifecycle?.[field], `frozen completion ${field} versus validation`, reasons);
  }
  requireEqual(invariant.configuredComputeDispatches, 0, 'frozen completion compute dispatches', reasons);
  requireEqual(invariant.configuredComputeSubmissions, 0, 'frozen completion compute submissions', reasons);
  requireEqual(
    invariant.configuredComputeDispatches,
    lifecycle?.configuredComputeDispatches,
    'frozen completion compute dispatches versus validation',
    reasons,
  );
  requireEqual(
    invariant.configuredComputeSubmissions,
    lifecycle?.configuredComputeSubmissions,
    'frozen completion compute submissions versus validation',
    reasons,
  );
  requireEqual(
    invariant.frontLaneBase,
    validation?.laneOffsets?.['front-to-back'],
    'frozen completion front lane base',
    reasons,
  );
  requireEqual(
    invariant.reverseLaneBase,
    validation?.laneOffsets?.reverse,
    'frozen completion reverse lane base',
    reasons,
  );
  if (![0, objectCount].includes(invariant.frontLaneBase)
    || ![0, objectCount].includes(invariant.reverseLaneBase)
    || invariant.frontLaneBase === invariant.reverseLaneBase) {
    reasons.push('frozen completion lane bases are not the exact two legal segment bases');
  }

  const lifecycleIdentityFields = pairedIdentityFields.filter(
    (field) => !field.startsWith('renderTarget')
      && !field.startsWith('camera'),
  );
  for (const field of pairedIdentityFields) {
    const startField = `${field}AtTimingStart`;
    const endField = `${field}AtTimingEnd`;
    requireEqual(invariant[endField], invariant[startField], `frozen completion stable ${field}`, reasons);
    if (lifecycleIdentityFields.includes(field)) {
      requireEqual(
        invariant[startField],
        lifecycle?.[field],
        `frozen completion ${field} versus validation`,
        reasons,
      );
    }
  }
  for (const field of [
    'bundleGroupUuid',
    'meshUuid',
    'geometryUuid',
    'materialUuid',
    'selectorUniformUuid',
    'renderTargetTextureUuid',
  ]) {
    const value = invariant[`${field}AtTimingStart`];
    if (typeof value !== 'string' || value.length === 0) {
      reasons.push(`frozen completion ${field} is not a nonempty identity string`);
    }
  }
  for (const field of ['cameraViewFnv64', 'cameraProjectionFnv64']) {
    if (!/^[0-9a-f]{16}$/.test(invariant[`${field}AtTimingStart`] ?? '')) {
      reasons.push(`frozen completion ${field} is not an FNV-1a-64 digest`);
    }
  }
  for (const field of [
    'matrixAttributeId',
    'visibleIdsAttributeId',
    'indirectAttributeId',
    'selectorChallengeAttributeId',
    'bundleGroupVersion',
    'matrixAttributeVersion',
    'visibleIdsAttributeVersion',
    'indirectAttributeVersion',
  ]) {
    if (!isNonnegativeInteger(invariant[`${field}AtTimingStart`])) {
      reasons.push(`frozen completion ${field} is not a nonnegative integer`);
    }
  }
  requireEqual(invariant.renderTargetWidthAtTimingStart, 1280, 'frozen render-target width', reasons);
  requireEqual(invariant.renderTargetHeightAtTimingStart, 720, 'frozen render-target height', reasons);
  requireEqual(invariant.renderTargetSamplesAtTimingStart, 0, 'frozen render-target samples', reasons);
  requireEqual(
    invariant.renderTargetDepthBufferAtTimingStart,
    true,
    'frozen render-target depthBuffer',
    reasons,
  );

  for (const field of [
    'selectorWriteSerialAtTimingStart',
    'selectorWriteSerialAtTimingEnd',
    'renderCallSerialAtTimingStart',
    'renderCallSerialAtTimingEnd',
    'computeCallSerialAtTimingStart',
    'computeCallSerialAtTimingEnd',
    'totalPipelineCacheEntriesAtTimingStart',
    'totalPipelineCacheEntriesAtTimingEnd',
    'computePipelineCacheEntriesAtTimingStart',
    'computePipelineCacheEntriesAtTimingEnd',
  ]) {
    if (!isNonnegativeInteger(invariant[field])) {
      reasons.push(`frozen completion ${field} is not a nonnegative integer`);
    }
  }
  requireEqual(
    invariant.expectedTimedFrameCount,
    expectedTimedFrameCount,
    'frozen completion expected timed frame count',
    reasons,
  );
  requireEqual(
    invariant.selectorWritesDuringTiming,
    expectedTimedFrameCount,
    'frozen completion selector writes during timing',
    reasons,
  );
  requireEqual(
    invariant.selectorWriteSerialAtTimingEnd - invariant.selectorWriteSerialAtTimingStart,
    expectedTimedFrameCount,
    'frozen completion selector serial delta',
    reasons,
  );
  requireEqual(
    invariant.renderCallsDuringTiming,
    expectedTimedFrameCount,
    'frozen completion render calls during timing',
    reasons,
  );
  requireEqual(
    invariant.renderCallSerialAtTimingEnd - invariant.renderCallSerialAtTimingStart,
    expectedTimedFrameCount,
    'frozen completion render-call serial delta',
    reasons,
  );
  requireEqual(invariant.computeCallsDuringTiming, 0, 'frozen completion compute calls', reasons);
  requireEqual(
    invariant.computeCallSerialAtTimingEnd,
    invariant.computeCallSerialAtTimingStart,
    'frozen completion compute-call serial stability',
    reasons,
  );
  requireEqual(
    invariant.totalPipelineCacheEntriesAtTimingEnd,
    invariant.totalPipelineCacheEntriesAtTimingStart,
    'frozen completion total pipeline-cache stability',
    reasons,
  );
  requireEqual(
    invariant.computePipelineCacheEntriesAtTimingEnd,
    invariant.computePipelineCacheEntriesAtTimingStart,
    'frozen completion compute pipeline-cache stability',
    reasons,
  );
  if (isNonnegativeInteger(invariant.totalPipelineCacheEntriesAtTimingStart)
    && isNonnegativeInteger(invariant.computePipelineCacheEntriesAtTimingStart)
    && invariant.computePipelineCacheEntriesAtTimingStart
      > invariant.totalPipelineCacheEntriesAtTimingStart) {
    reasons.push('frozen completion compute pipeline-cache entries exceed total entries');
  }
  return [...new Set(reasons)];
}

export function validateDepthOrderingCompletionInvariant(invariant, {
  modeId,
  objectCount,
  bucketCount,
  validation,
  scenarioManifest,
} = {}) {
  const reasons = [];
  if (modeId === FROZEN_DEPTH_CROSSOVER_MODE) {
    return validateFrozenCrossoverCompletionInvariant(invariant, {
      objectCount,
      validation,
    });
  }
  if (depthModeProtocol(modeId)) {
    validateDepthCompletionInvariant(invariant, {
      modeId,
      objectCount,
      bucketCount,
      validation,
      scenarioManifest,
    }, reasons);
  } else if (modeId === 'fixed-slice') {
    validateAtomicFixedSliceCompletionInvariant(invariant, reasons);
  } else {
    reasons.push(`unsupported depth-ordering completion mode ${JSON.stringify(modeId)}`);
  }
  return [...new Set(reasons)];
}

export function validateFrozenCrossoverTrialRows(spec, rows, pageSummary, validation, {
  warmupFrames,
  measuredFrames,
  plannedScheduleSha256 = null,
} = {}) {
  const reasons = [];
  if (spec?.modeId !== FROZEN_DEPTH_CROSSOVER_MODE) {
    return [`frozen crossover row validation received modeId ${JSON.stringify(spec?.modeId)}`];
  }
  requireEqual(
    warmupFrames,
    FROZEN_CROSSOVER_WARMUP_FRAMES,
    'frozen crossover protocol warmup frame count',
    reasons,
  );
  requireEqual(
    measuredFrames,
    FROZEN_CROSSOVER_MEASURED_FRAMES,
    'frozen crossover protocol measured frame count',
    reasons,
  );
  if (!Array.isArray(rows)) return ['frozen crossover timed rows are not an array'];
  requireEqual(
    rows.length,
    FROZEN_CROSSOVER_MEASURED_FRAMES,
    'frozen crossover measured row count',
    reasons,
  );
  if (!Array.isArray(spec.laneStorageOrder)
    || spec.laneStorageOrder.length !== FROZEN_DEPTH_CROSSOVER_LANES.length
    || new Set(spec.laneStorageOrder).size !== FROZEN_DEPTH_CROSSOVER_LANES.length
    || FROZEN_DEPTH_CROSSOVER_LANES.some((lane) => !spec.laneStorageOrder.includes(lane))) {
    reasons.push('frozen crossover planned lane storage order is not the exact lane pair');
  }
  if (spec.superblockOrientationOffset !== 0 && spec.superblockOrientationOffset !== 1) {
    reasons.push('frozen crossover superblock orientation offset is not zero or one');
  }
  const expectedStorageOrder = frozenLogicalLane(spec.laneStorageOrder?.[0]);
  requireEqual(
    validation?.laneStorageOrder,
    expectedStorageOrder,
    'frozen crossover validation laneStorageOrder versus plan',
    reasons,
  );
  const scheduleSha256 = (spec.superblockOrientationOffset === 0
    || spec.superblockOrientationOffset === 1)
    ? frozenCrossoverScheduleSha256(spec.superblockOrientationOffset)
    : null;
  if (plannedScheduleSha256 === null) {
    reasons.push('frozen crossover planned schedule sha256 is missing');
  } else {
    requireEqual(
      plannedScheduleSha256,
      scheduleSha256,
      'frozen crossover planned schedule sha256',
      reasons,
    );
  }
  requireEqual(
    pageSummary?.expectedRenderTimestampUidCount,
    1,
    'frozen crossover summary expected render timestamp UID count',
    reasons,
  );
  requireEqual(
    pageSummary?.invalidRenderTimestampUidCountFrames,
    0,
    'frozen crossover summary invalid render timestamp UID-count frames',
    reasons,
  );

  const invariant = pageSummary?.completionInvariant;
  reasons.push(...validateFrozenCrossoverCompletionInvariant(invariant, {
    objectCount: spec.objectCount,
    validation,
  }));
  const laneCounts = Object.fromEntries(
    FROZEN_DEPTH_CROSSOVER_LANES.map((laneId) => [laneId, 0]),
  );
  const completionIdentityFields = [
    'bundleGroupUuid',
    'meshUuid',
    'geometryUuid',
    'materialUuid',
    'matrixAttributeId',
    'visibleIdsAttributeId',
    'indirectAttributeId',
    'selectorChallengeAttributeId',
    'bundleGroupVersion',
    'matrixAttributeVersion',
    'visibleIdsAttributeVersion',
    'indirectAttributeVersion',
    'selectorUniformUuid',
    'renderTargetTextureUuid',
    'renderTargetWidth',
    'renderTargetHeight',
    'renderTargetSamples',
    'renderTargetDepthBuffer',
    'cameraViewFnv64',
    'cameraProjectionFnv64',
  ];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const prefix = `frozen frame ${index}`;
    if (!isRecord(row)) continue;
    if (spec.superblockOrientationOffset !== 0 && spec.superblockOrientationOffset !== 1) continue;
    const scheduled = frozenCrossoverFrame(index, spec.superblockOrientationOffset);
    const logicalLane = frozenLogicalLane(scheduled.laneId);
    laneCounts[scheduled.laneId] += 1;
    const expectedFields = {
      expectedRenderTimestampUidCount: 1,
      gpuRenderTimestampUidCount: 1,
      plannedLaneStorageOrder: spec.laneStorageOrder?.join('|'),
      plannedScheduleSha256: scheduleSha256,
      superblockOrientationOffset: spec.superblockOrientationOffset,
      frontLaneBase: validation?.laneOffsets?.['front-to-back'],
      reverseLaneBase: validation?.laneOffsets?.reverse,
      phaseFrameIndex: index,
      crossoverBlockIndex: scheduled.crossoverBlockIndex,
      withinBlockPosition: scheduled.withinBlockPosition,
      crossoverPattern: scheduled.pattern,
      crossoverPatternIndex: scheduled.patternIndex,
      laneId: scheduled.laneId,
      laneBase: validation?.laneOffsets?.[logicalLane],
      selectorWriteSerialAtTimingStart: invariant?.selectorWriteSerialAtTimingStart,
      selectorWriteSerial:
        invariant?.selectorWriteSerialAtTimingStart
          + FROZEN_CROSSOVER_WARMUP_FRAMES + index + 1,
      renderCallSerialAtTimingStart: invariant?.renderCallSerialAtTimingStart,
      renderCallSerial:
        invariant?.renderCallSerialAtTimingStart
          + FROZEN_CROSSOVER_WARMUP_FRAMES + index + 1,
      computeCallSerialAtTimingStart: invariant?.computeCallSerialAtTimingStart,
      totalPipelineCacheEntriesAtTimingStart:
        invariant?.totalPipelineCacheEntriesAtTimingStart,
      computePipelineCacheEntriesAtTimingStart:
        invariant?.computePipelineCacheEntriesAtTimingStart,
    };
    for (const field of completionIdentityFields) {
      expectedFields[`${field}AtTimingStart`] = invariant?.[`${field}AtTimingStart`];
    }
    for (const [field, expected] of Object.entries(expectedFields)) {
      if (!(field in row)) reasons.push(`${prefix} is missing ${field}`);
      else requireEqual(row[field], expected, `${prefix} ${field}`, reasons);
    }
  }
  for (const laneId of FROZEN_DEPTH_CROSSOVER_LANES) {
    requireEqual(
      laneCounts[laneId],
      FROZEN_EXPECTED_MEASURED_ROWS_PER_LANE,
      `frozen crossover measured rows for ${laneId}`,
      reasons,
    );
  }
  requireEqual(
    FROZEN_CROSSOVER_PATTERNS.length,
    2,
    'frozen crossover schedule pattern count',
    reasons,
  );
  return [...new Set(reasons)];
}

export function validateTrialRows(spec, rows, pageSummary, validation, scenarioManifest, {
  schemaVersion,
  warmupFrames,
  measuredFrames,
  plannedScheduleSha256 = null,
} = {}) {
  const reasons = [];
  const shape = expectedStrategyShape(spec.modeId, spec.bucketCount, spec.objectCount);
  const depthProtocol = depthModeProtocol(spec.modeId);
  const frozenCrossoverTrial = spec.modeId === FROZEN_DEPTH_CROSSOVER_MODE;
  const depthOrderingTrial = depthProtocol !== null
    || ['high-overlap', 'low-overlap'].includes(spec.layout);
  if (!shape) return [`unsupported modeId ${JSON.stringify(spec.modeId)}`];
  if (!Array.isArray(rows)) return ['timed rows are not an array'];
  if (rows.length !== measuredFrames) {
    reasons.push(`expected ${measuredFrames} rows, received ${rows.length}`);
  }
  if (pageSummary?.accepted !== true) reasons.push('page timing summary was rejected');
  if (pageSummary?.timestampAvailable !== true) reasons.push('GPU timestamps were unavailable');
  requireEqual(pageSummary?.rowCount, measuredFrames, 'page summary rowCount', reasons);
  requireEqual(pageSummary?.missingRenderFrames, 0, 'page summary missingRenderFrames', reasons);
  requireEqual(pageSummary?.missingComputeFrames, 0, 'page summary missingComputeFrames', reasons);
  if (!Number.isFinite(pageSummary?.quantumNs) || pageSummary.quantumNs <= 0) {
    reasons.push('page summary timestamp quantum is missing or invalid');
  }
  if (typeof pageSummary?.classification !== 'string' || pageSummary.classification.length === 0) {
    reasons.push('page summary timestamp classification is missing');
  }
  if (frozenCrossoverTrial) {
    reasons.push(...validateFrozenCrossoverTrialRows(spec, rows, pageSummary, validation, {
      warmupFrames,
      measuredFrames,
      plannedScheduleSha256,
    }));
  }
  if (depthOrderingTrial) {
    if (!['high-overlap', 'low-overlap'].includes(spec.layout)) {
      reasons.push('depth-ordering spec layout is unsupported');
    }
    if (!Array.isArray(spec.layoutOrder) || spec.layoutOrder.length !== 2
      || new Set(spec.layoutOrder).size !== 2
      || !spec.layoutOrder.includes('high-overlap')
      || !spec.layoutOrder.includes('low-overlap')) {
      reasons.push('depth-ordering spec layoutOrder is not the two-layout audit pair');
    }
    if (!Number.isInteger(spec.layoutOrderPosition)
      || spec.layoutOrderPosition < 0
      || spec.layoutOrderPosition >= (spec.layoutOrder?.length ?? 0)) {
      reasons.push('depth-ordering spec layoutOrderPosition is invalid');
    } else {
      requireEqual(
        spec.layoutOrder[spec.layoutOrderPosition],
        spec.layout,
        'depth-ordering spec selected layout position',
        reasons,
      );
    }
    requireEqual(
      scenarioManifest?.layout,
      spec.layout,
      'depth-ordering scenario layout versus plan',
      reasons,
    );
    if (!isRecord(scenarioManifest?.depthBinRange)
      || !Number.isFinite(scenarioManifest.depthBinRange.near)
      || !Number.isFinite(scenarioManifest.depthBinRange.far)
      || scenarioManifest.depthBinRange.far <= scenarioManifest.depthBinRange.near) {
      reasons.push('depth-ordering scenario manifest depthBinRange is missing or invalid');
    }
    if (frozenCrossoverTrial) {
      // The frozen path validates its stronger completion contract above.
    } else if (depthProtocol) {
      reasons.push(...validateDepthOrderingCompletionInvariant(
        pageSummary?.completionInvariant,
        {
        modeId: spec.modeId,
        objectCount: spec.objectCount,
        bucketCount: spec.bucketCount,
        validation,
        scenarioManifest,
        },
      ));
    } else if (spec.modeId === 'fixed-slice') {
      reasons.push(...validateDepthOrderingCompletionInvariant(
        pageSummary?.completionInvariant,
        {
          modeId: spec.modeId,
          objectCount: spec.objectCount,
          bucketCount: spec.bucketCount,
          validation,
          scenarioManifest,
        },
      ));
    }
  }
  if (spec.modeId === 'fixed-slice-per-bucket') {
    const invariant = pageSummary?.completionInvariant;
    if (!isRecord(invariant)) {
      reasons.push('fixed-slice-per-bucket completion invariant is missing');
    } else {
      requireEqual(invariant.pass, true, 'fixed-slice-per-bucket completion invariant pass', reasons);
      requireEqual(
        invariant.kind,
        'fixed-slice-per-bucket-static-bundle-invariant',
        'fixed-slice-per-bucket completion invariant kind',
        reasons,
      );
      requireEqual(
        invariant.bundleRecordCallbackCountAtTimingStart,
        validation?.representation?.bundleRecordCallbackCount,
        'fixed-slice-per-bucket timing-start bundle-record callback count',
        reasons,
      );
      requireEqual(
        invariant.bundleRecordCallbackCountAtTimingEnd,
        invariant.bundleRecordCallbackCountAtTimingStart,
        'fixed-slice-per-bucket timing-end bundle-record callback count',
        reasons,
      );
      requireEqual(invariant.geometryIdentityCount, 1, 'completion geometry identity count', reasons);
      requireEqual(invariant.materialIdentityCount, 1, 'completion material identity count', reasons);
      requireEqual(invariant.meshCount, spec.bucketCount, 'completion mesh count', reasons);
      requireEqual(
        invariant.geometryInstanceCount,
        Math.ceil(spec.objectCount / spec.bucketCount),
        'completion geometry instanceCount',
        reasons,
      );
    }
  }

  const frameIds = new Set();
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const prefix = `frame ${index}`;
    if (!isRecord(row)) {
      reasons.push(`${prefix} is not an object`);
      continue;
    }
    const expectedFields = {
      schemaVersion,
      runId: spec.runId,
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      modeOrderPosition: spec.modeOrderPosition,
      visibilityOrderPosition: spec.visibilityOrderPosition,
      plannedModeOrder: spec.modeOrder.join('|'),
      plannedVisibilityOrder: spec.visibilityOrder.join('|'),
      protocolWarmupFrames: warmupFrames,
      protocolMeasuredFrames: measuredFrames,
      modeId: spec.modeId,
      objectCount: spec.objectCount,
      bucketCount: spec.bucketCount,
      targetVisibilityFraction: spec.visibilityFraction,
      expectedVisibleCount: scenarioManifest?.expectedVisibleCount,
      validationKind: validation?.kind,
      validationPass: true,
      usesCompute: shape.compute,
      configuredDrawCommands: shape.configuredDrawCommands,
      configuredRenderObjects: shape.configuredRenderObjects,
      configuredComputeDispatches: shape.configuredComputeDispatches,
      configuredComputeSubmissions: shape.configuredComputeSubmissions,
      configuredSubmittedInstances: frozenCrossoverTrial
        ? scenarioManifest?.expectedVisibleCount
        : shape.configuredSubmittedInstances,
      bundleRecordCallbackCountAtTimingStart: frozenCrossoverTrial
        ? 1
        : depthProtocol
        ? 1
        : depthOrderingTrial && spec.modeId === 'fixed-slice'
          ? 1
          : spec.modeId === 'fixed-slice-per-bucket'
            ? validation?.representation?.bundleRecordCallbackCount
            : null,
      timestampAvailable: true,
      frameIndex: index,
      ...(depthOrderingTrial ? {
        layoutOrderPosition: spec.layoutOrderPosition,
        plannedLayoutOrder: spec.layoutOrder.join('|'),
        scenarioLayout: spec.layout,
        depthBinRangeNear: scenarioManifest?.depthBinRange?.near,
        depthBinRangeFar: scenarioManifest?.depthBinRange?.far,
      } : {}),
    };
    for (const [field, expected] of Object.entries(expectedFields)) {
      if (!(field in row)) {
        reasons.push(`${prefix} is missing ${field}`);
      } else if (row[field] !== expected) {
        reasons.push(`${prefix} ${field} differs from the plan`);
      }
    }
    if (!Number.isInteger(row.gpuFrameId) || row.gpuFrameId < 0) {
      reasons.push(`${prefix} has no nonnegative integer GPU frame ID`);
    } else if (frameIds.has(row.gpuFrameId)) {
      reasons.push(`${prefix} duplicates GPU frame ID ${row.gpuFrameId}`);
    } else {
      frameIds.add(row.gpuFrameId);
    }
    for (const field of [
      'cpuCommonUpdateMs',
      'cpuRenderSubmitMs',
      'cpuSubmitTotalMs',
      'cpuFrameBodyMs',
      'gpuRenderMs',
      'gpuPassTotalMs',
    ]) {
      if (!finiteNonnegative(row[field])) reasons.push(`${prefix} ${field} is missing or invalid`);
    }
    if (shape.compute) {
      if (!finiteNonnegative(row.cpuComputeSubmitMs)) {
        reasons.push(`${prefix} cpuComputeSubmitMs is missing or invalid`);
      }
      if (!finiteNonnegative(row.gpuComputeMs)) reasons.push(`${prefix} gpuComputeMs is missing or invalid`);
    } else {
      requireEqual(row.cpuComputeSubmitMs, null, `${prefix} cpuComputeSubmitMs`, reasons);
      requireEqual(row.gpuComputeMs, null, `${prefix} gpuComputeMs`, reasons);
    }
    if (finiteNonnegative(row.cpuRenderSubmitMs)
      && (shape.compute ? finiteNonnegative(row.cpuComputeSubmitMs) : true)
      && Math.abs(
        row.cpuSubmitTotalMs - (row.cpuRenderSubmitMs + (row.cpuComputeSubmitMs ?? 0)),
      ) > 1e-9) {
      reasons.push(`${prefix} CPU submit total does not equal its component sum`);
    }
    if (finiteNonnegative(row.gpuRenderMs)
      && (shape.compute ? finiteNonnegative(row.gpuComputeMs) : true)
      && Math.abs(row.gpuPassTotalMs - (row.gpuRenderMs + (row.gpuComputeMs ?? 0))) > 1e-9) {
      reasons.push(`${prefix} GPU pass total does not equal its component sum`);
    }
  }
  return [...new Set(reasons)];
}
