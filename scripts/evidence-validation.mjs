import { createHash } from 'node:crypto';

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const GEOMETRY_ATTRIBUTE_NAMES = Object.freeze(['normal', 'position', 'uv']);
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

export function sha256Json(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
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
} = {}) {
  const reasons = [];
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
  const compute = modeId !== 'draw-all';
  return {
    kind: modeId === 'draw-all' ? 'draw-all-reference' : `${modeId}-exact-membership`,
    compute,
    configuredDrawCommands: bucketCount,
    configuredRenderObjects: modeId === 'fixed-slice' || modeId === 'three-blocks-historical'
      ? 1
      : bucketCount,
    configuredComputeDispatches: modeId === 'draw-all'
      ? 0
      : modeId === 'fixed-slice'
        ? 2
        : modeId === 'three-blocks-historical'
          ? 9
          : bucketCount * 4,
    configuredComputeSubmissions: modeId === 'draw-all'
      ? 0
      : modeId === 'fixed-slice'
        || modeId === 'three-blocks-coalesced'
        || modeId === 'three-blocks-historical'
        ? 1
        : bucketCount,
    configuredSubmittedInstances: modeId === 'draw-all' ? objectCount : null,
  };
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

export function validateExactValidation(validation, {
  modeId,
  bucketCount,
  expectedVisibleCount,
  expectedVisibleIdsCanonicalSha256,
  geometryManifest,
} = {}) {
  const reasons = [];
  const shape = expectedStrategyShape(modeId, bucketCount, null);
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
        const expectedFirstIndex = modeId === 'fixed-slice' || modeId === 'three-blocks-historical'
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
  if (modeId === 'fixed-slice') requireEqual(validation.overflow, 0, 'fixed-slice overflow', reasons);

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
  };
  return {
    rejectionReasons: [...new Set(reasons)],
    semanticSha256: sha256Json(semanticRecord),
  };
}

function finiteNonnegative(value) {
  return Number.isFinite(value) && value >= 0;
}

export function validateTrialRows(spec, rows, pageSummary, validation, scenarioManifest, {
  schemaVersion,
  warmupFrames,
  measuredFrames,
} = {}) {
  const reasons = [];
  const shape = expectedStrategyShape(spec.modeId, spec.bucketCount, spec.objectCount);
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
      configuredSubmittedInstances: shape.configuredSubmittedInstances,
      timestampAvailable: true,
      frameIndex: index,
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
