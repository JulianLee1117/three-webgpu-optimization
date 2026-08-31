import assert from 'node:assert/strict';
import test from 'node:test';
import {
  frozenCrossoverScheduleSha256,
  physicalBinSequenceIdentity,
  renderParityIdentity,
  validateExactValidation,
  validateFrozenCrossoverCompletionInvariant,
  validateFrozenCrossoverRenderParity,
  validateRenderParity,
  validateTrialRows,
} from '../scripts/evidence-validation.mjs';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
} from '../src/benchmark/plan.js';
import {
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  FROZEN_CROSSOVER_WARMUP_FRAMES,
  frozenCrossoverFrame,
} from '../src/benchmark/frozen-crossover-schedule.js';

const OBJECT_COUNT = 16;
const BUCKET_COUNT = 2;
const VISIBLE_COUNT = 6;
const GLOBAL_DIGEST = 'a'.repeat(64);
const DEPTH_RANGE = Object.freeze({ near: 90, far: 190 });
const PARITY_BYTE_LENGTH = 1280 * 720 * 4;
const LOGICAL_LANES = Object.freeze(['front-to-back', 'reverse']);
const LOGICAL_BY_MODE = Object.freeze({
  [FROZEN_DEPTH_CROSSOVER_LANES[0]]: LOGICAL_LANES[0],
  [FROZEN_DEPTH_CROSSOVER_LANES[1]]: LOGICAL_LANES[1],
});

function traversalFor(order) {
  return order === 'front-to-back'
    ? [0, 1, 2, 3, 4, 5, 6, 7]
    : [7, 6, 5, 4, 3, 2, 1, 0];
}

function membershipDigests() {
  return {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    encoding: 'sorted-uint32-little-endian',
    pass: true,
    invalidExpectedIds: 0,
    truncatedActualIds: 0,
    expected: { count: VISIBLE_COUNT, sha256: GLOBAL_DIGEST },
    actual: { count: VISIBLE_COUNT, sha256: GLOBAL_DIGEST },
    perBucket: [0, 1].map((bucket) => ({
      bucket,
      match: true,
      expected: { count: 3, sha256: String(bucket + 1).repeat(64) },
      actual: {
        count: 3,
        declaredCount: 3,
        sha256: String(bucket + 1).repeat(64),
      },
    })),
  };
}

function commandValidation() {
  let firstIndex = 0;
  const records = [12, 18].map((indexCount, bucket) => {
    const command = {
      indexCount,
      instanceCount: 3,
      firstIndex,
      baseVertex: 0,
      firstInstance: 0,
    };
    firstIndex += indexCount;
    return { bucket, actual: { ...command }, expected: { ...command } };
  });
  return {
    pass: true,
    errors: [],
    commandCount: BUCKET_COUNT,
    totalInstanceCount: VISIBLE_COUNT,
    records,
  };
}

function depthBins(order) {
  const counts = [
    1, 1, 0, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 1, 0, 0,
  ];
  const traversal = traversalFor(order);
  const starts = new Array(BUCKET_COUNT * 8).fill(0);
  const totals = new Array(BUCKET_COUNT).fill(0);
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    let cursor = 0;
    for (const physicalBin of traversal) {
      const index = bucket * 8 + physicalBin;
      starts[index] = cursor;
      cursor += counts[index];
    }
    totals[bucket] = cursor;
  }
  return {
    pass: true,
    errors: [],
    binCount: 8,
    order,
    traversal,
    expectedCounts: [...counts],
    actualCounts: [...counts],
    expectedStarts: [...starts],
    actualStarts: [...starts],
    expectedBucketTotals: [...totals],
    commandCounts: [...totals],
    physicalBinSequenceCommitment: {
      schemaVersion: 1,
      hashAlgorithm: 'sha256',
      encoding: 'bucket-major-physical-bin-major-tagged-uint32-little-endian',
      bucketCount: BUCKET_COUNT,
      binCount: 8,
      recordCount: BUCKET_COUNT * 8,
      survivorCount: VISIBLE_COUNT,
      sha256: '9'.repeat(64),
    },
  };
}

const SHARED_IDENTITIES = Object.freeze({
  bundleGroupUuid: 'bundle-uuid',
  meshUuid: 'mesh-uuid',
  geometryUuid: 'geometry-uuid',
  materialUuid: 'material-uuid',
  matrixAttributeId: 11,
  visibleIdsAttributeId: 12,
  indirectAttributeId: 13,
  selectorChallengeAttributeId: 14,
  bundleGroupVersion: 0,
  matrixAttributeVersion: 0,
  visibleIdsAttributeVersion: 0,
  indirectAttributeVersion: 0,
  selectorUniformUuid: 'selector-uniform-uuid',
});

function frozenDiagnostics(laneStorageOrder, laneOffsets, activeLane) {
  const shared = {
    laneStorageOrder,
    laneOffsets: { ...laneOffsets },
    activeLane,
    activeVisibleIdOffset: laneOffsets[activeLane],
    bundleGroupStatic: true,
    bundleRecordCallbackCount: 1,
    meshCount: 1,
    geometryIdentityCount: 1,
    materialIdentityCount: 1,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    ...SHARED_IDENTITIES,
  };
  return {
    representation: {
      kind: 'single-render-object-frozen-depth-crossover',
      ...shared,
      visibleIdsCount: OBJECT_COUNT * 2,
      visibleIdSegmentLength: OBJECT_COUNT,
      depthBinCount: 8,
      commandCount: BUCKET_COUNT,
      zeroFirstInstanceCount: BUCKET_COUNT,
      diagnosticSelectorDispatchesPerValidation: 2,
    },
    lifecycle: {
      kind: 'frozen-depth-crossover-static-bundle-lifecycle',
      ...shared,
    },
  };
}

function createFrozenValidation(storageOrder = 'front-to-back') {
  const laneOffsets = storageOrder === 'front-to-back'
    ? { 'front-to-back': 0, reverse: OBJECT_COUNT }
    : { 'front-to-back': OBJECT_COUNT, reverse: 0 };
  const activeLane = 'front-to-back';
  const lanes = Object.fromEntries(LOGICAL_LANES.map((order, index) => [order, {
    pass: true,
    order,
    storageOffset: laneOffsets[order],
    traversal: traversalFor(order),
    membership: {
      pass: true,
      expectedCount: VISIBLE_COUNT,
      listedCount: VISIBLE_COUNT,
      duplicateIds: 0,
      outOfRangeIds: 0,
      wrongBucketIds: 0,
      listedHiddenIds: 0,
      missingVisibleIds: 0,
      overflow: 0,
      errors: 0,
    },
    membershipDigests: membershipDigests(),
    depthBins: depthBins(order),
    storageSegmentSha256: String(index + 2).repeat(64),
    paddingSentinelCount: OBJECT_COUNT - VISIBLE_COUNT,
    paddingCorruptionCount: 0,
  }]));
  const selectorChallenges = Object.fromEntries(LOGICAL_LANES.map((order) => [order, {
    pass: true,
    kind: 'gpu-selector-address-challenge',
    lane: order,
    storageOffset: laneOffsets[order],
    elementCount: OBJECT_COUNT,
    sha256: lanes[order].storageSegmentSha256,
    expectedSha256: lanes[order].storageSegmentSha256,
  }]));
  const diagnostics = frozenDiagnostics(storageOrder, laneOffsets, activeLane);
  return {
    pass: true,
    kind: 'frozen-depth-crossover-exact-paired-snapshots',
    expectedIdsMatchScenario: true,
    visibleIdsByteLength: OBJECT_COUNT * 2 * 4,
    visibleIdsSha256: 'd'.repeat(64),
    expectedVisibleIdsSha256: 'd'.repeat(64),
    visibleIdsExactPackingMatch: true,
    commandSha256: 'e'.repeat(64),
    laneStorageOrder: storageOrder,
    laneOffsets,
    activeLane,
    activeVisibleIdOffset: laneOffsets[activeLane],
    physicalBinSequenceCommitmentsEqual: true,
    rawLaneSequencesDiffer: true,
    physicalBinSequenceSha256: '9'.repeat(64),
    commandValidation: commandValidation(),
    lanes,
    selectorChallenges,
    ...diagnostics,
  };
}

function frozenSpec(storageOrder = [...FROZEN_DEPTH_CROSSOVER_LANES], orientation = 0) {
  return {
    runId: 'frozen-run',
    trialId: 'frozen-trial',
    planIndex: 0,
    repetitionIndex: 0,
    modeId: FROZEN_DEPTH_CROSSOVER_MODE,
    modeOrder: [FROZEN_DEPTH_CROSSOVER_MODE],
    modeOrderPosition: 0,
    visibilityFraction: 0.99,
    visibilityOrder: [0.99],
    visibilityOrderPosition: 0,
    layout: 'high-overlap',
    layoutOrder: ['high-overlap', 'low-overlap'],
    layoutOrderPosition: 0,
    laneStorageOrder: [...storageOrder],
    superblockOrientationOffset: orientation,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  };
}

function exactOptions(spec = frozenSpec()) {
  return {
    modeId: FROZEN_DEPTH_CROSSOVER_MODE,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    expectedVisibleCount: VISIBLE_COUNT,
    expectedVisibleIdsCanonicalSha256: GLOBAL_DIGEST,
    geometryManifest: {
      geometries: [
        { index: { count: 12 } },
        { index: { count: 18 } },
      ],
    },
    scenarioManifest: {
      expectedVisibleCount: VISIBLE_COUNT,
      expectedVisibleIdsCanonicalSha256: GLOBAL_DIGEST,
      layout: spec.layout,
      depthBinRange: { ...DEPTH_RANGE },
    },
    laneStorageOrder: spec.laneStorageOrder,
  };
}

function parityChannel(format, arrayType, sha256) {
  return { format, arrayType, byteLength: PARITY_BYTE_LENGTH, sha256 };
}

function parityLane() {
  const color = parityChannel('rgba8unorm', 'Uint8Array', '4'.repeat(64));
  const depth = parityChannel('depth32float', 'Float32Array', '5'.repeat(64));
  const objectId = parityChannel(
    'rgba8unorm-object-id-plus-one',
    'Uint8Array',
    '6'.repeat(64),
  );
  return {
    schemaVersion: 1,
    kind: 'fixed-camera-offscreen-exact-render-parity',
    pass: true,
    width: 1280,
    height: 720,
    captures: 2,
    material: { type: 'MeshStandardNodeMaterial' },
    color,
    depth,
    objectId,
    objectIdValidation: {
      pass: true,
      encoding: 'rgb24-object-id-plus-one-zero-background',
      backgroundPixels: 900_000,
      coveredPixels: 21_600,
      outOfRangePixels: 0,
      nonVisiblePixels: 0,
    },
    reversedDepthBuffer: true,
    stability: {
      pass: true,
      firstCapture: {
        color: { ...color },
        depth: { ...depth },
        objectId: { ...objectId },
      },
      first: {
        colorSha256: color.sha256,
        depthSha256: depth.sha256,
        objectIdSha256: objectId.sha256,
      },
    },
  };
}

function frozenParity(validation = createFrozenValidation()) {
  return {
    schemaVersion: 1,
    kind: 'frozen-depth-crossover-exact-render-parity',
    pass: true,
    laneIds: [...FROZEN_DEPTH_CROSSOVER_LANES],
    crossLaneExact: true,
    lanes: Object.fromEntries(
      FROZEN_DEPTH_CROSSOVER_LANES.map((laneId) => [laneId, parityLane()]),
    ),
    snapshotValidation: validation,
  };
}

const COMPLETION_ONLY_IDENTITIES = Object.freeze({
  renderTargetTextureUuid: 'render-target-texture-uuid',
  renderTargetWidth: 1280,
  renderTargetHeight: 720,
  renderTargetSamples: 0,
  renderTargetDepthBuffer: true,
  cameraViewFnv64: '0123456789abcdef',
  cameraProjectionFnv64: 'fedcba9876543210',
});

function completionInvariant(validation) {
  const invariant = {
    pass: true,
    kind: 'frozen-depth-crossover-static-bundle-invariant',
    bundleGroupStatic: true,
    bundleRecordCallbackCountAtTimingStart: 1,
    bundleRecordCallbackCountAtTimingEnd: 1,
    meshCount: 1,
    geometryIdentityCount: 1,
    materialIdentityCount: 1,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    frontLaneBase: validation.laneOffsets['front-to-back'],
    reverseLaneBase: validation.laneOffsets.reverse,
    selectorWriteSerialAtTimingStart: 10,
    selectorWriteSerialAtTimingEnd: 810,
    selectorWritesDuringTiming: 800,
    renderCallSerialAtTimingStart: 100,
    renderCallSerialAtTimingEnd: 900,
    renderCallsDuringTiming: 800,
    computeCallSerialAtTimingStart: 5,
    computeCallSerialAtTimingEnd: 5,
    computeCallsDuringTiming: 0,
    totalPipelineCacheEntriesAtTimingStart: 12,
    totalPipelineCacheEntriesAtTimingEnd: 12,
    computePipelineCacheEntriesAtTimingStart: 2,
    computePipelineCacheEntriesAtTimingEnd: 2,
    expectedTimedFrameCount: 800,
  };
  for (const [field, value] of Object.entries({
    ...SHARED_IDENTITIES,
    ...COMPLETION_ONLY_IDENTITIES,
  })) {
    invariant[`${field}AtTimingStart`] = value;
    invariant[`${field}AtTimingEnd`] = value;
  }
  return invariant;
}

function trialArtifacts() {
  const spec = frozenSpec();
  const validation = createFrozenValidation();
  const invariant = completionInvariant(validation);
  const common = {
    schemaVersion: 2,
    runId: spec.runId,
    trialId: spec.trialId,
    planIndex: spec.planIndex,
    repetitionIndex: spec.repetitionIndex,
    modeOrderPosition: spec.modeOrderPosition,
    visibilityOrderPosition: spec.visibilityOrderPosition,
    plannedModeOrder: spec.modeOrder.join('|'),
    plannedVisibilityOrder: spec.visibilityOrder.join('|'),
    protocolWarmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
    protocolMeasuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
    modeId: spec.modeId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    targetVisibilityFraction: spec.visibilityFraction,
    expectedVisibleCount: VISIBLE_COUNT,
    validationKind: validation.kind,
    validationPass: true,
    usesCompute: false,
    configuredDrawCommands: BUCKET_COUNT,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    configuredSubmittedInstances: VISIBLE_COUNT,
    bundleRecordCallbackCountAtTimingStart: 1,
    timestampAvailable: true,
    layoutOrderPosition: spec.layoutOrderPosition,
    plannedLayoutOrder: spec.layoutOrder.join('|'),
    scenarioLayout: spec.layout,
    depthBinRangeNear: DEPTH_RANGE.near,
    depthBinRangeFar: DEPTH_RANGE.far,
    expectedRenderTimestampUidCount: 1,
    plannedLaneStorageOrder: spec.laneStorageOrder.join('|'),
    plannedScheduleSha256: frozenCrossoverScheduleSha256(spec.superblockOrientationOffset),
    superblockOrientationOffset: spec.superblockOrientationOffset,
    frontLaneBase: validation.laneOffsets['front-to-back'],
    reverseLaneBase: validation.laneOffsets.reverse,
    selectorWriteSerialAtTimingStart: invariant.selectorWriteSerialAtTimingStart,
    renderCallSerialAtTimingStart: invariant.renderCallSerialAtTimingStart,
    computeCallSerialAtTimingStart: invariant.computeCallSerialAtTimingStart,
    totalPipelineCacheEntriesAtTimingStart:
      invariant.totalPipelineCacheEntriesAtTimingStart,
    computePipelineCacheEntriesAtTimingStart:
      invariant.computePipelineCacheEntriesAtTimingStart,
  };
  for (const field of Object.keys({ ...SHARED_IDENTITIES, ...COMPLETION_ONLY_IDENTITIES })) {
    common[`${field}AtTimingStart`] = invariant[`${field}AtTimingStart`];
  }
  const rows = Array.from({ length: FROZEN_CROSSOVER_MEASURED_FRAMES }, (_, index) => {
    const scheduled = frozenCrossoverFrame(index, spec.superblockOrientationOffset);
    return {
      ...common,
      frameIndex: index,
      phaseFrameIndex: index,
      crossoverBlockIndex: scheduled.crossoverBlockIndex,
      withinBlockPosition: scheduled.withinBlockPosition,
      crossoverPattern: scheduled.pattern,
      crossoverPatternIndex: scheduled.patternIndex,
      laneId: scheduled.laneId,
      laneBase: validation.laneOffsets[LOGICAL_BY_MODE[scheduled.laneId]],
      selectorWriteSerial:
        invariant.selectorWriteSerialAtTimingStart
          + FROZEN_CROSSOVER_WARMUP_FRAMES + index + 1,
      renderCallSerial:
        invariant.renderCallSerialAtTimingStart
          + FROZEN_CROSSOVER_WARMUP_FRAMES + index + 1,
      gpuFrameId: 1_000 + index,
      cpuCommonUpdateMs: 0.01,
      cpuComputeSubmitMs: null,
      cpuRenderSubmitMs: 0.1,
      cpuSubmitTotalMs: 0.1,
      cpuFrameBodyMs: 0.12,
      gpuComputeMs: null,
      gpuRenderMs: 0.2,
      gpuRenderTimestampUidCount: 1,
      gpuPassTotalMs: 0.2,
    };
  });
  const summary = {
    accepted: true,
    timestampAvailable: true,
    rowCount: FROZEN_CROSSOVER_MEASURED_FRAMES,
    missingRenderFrames: 0,
    invalidRenderTimestampUidCountFrames: 0,
    expectedRenderTimestampUidCount: 1,
    missingComputeFrames: 0,
    quantumNs: 32,
    classification: 'hardware-like',
    completionInvariant: invariant,
  };
  const scenarioManifest = {
    expectedVisibleCount: VISIBLE_COUNT,
    expectedVisibleIdsCanonicalSha256: GLOBAL_DIGEST,
    layout: spec.layout,
    depthBinRange: { ...DEPTH_RANGE },
  };
  return { spec, validation, rows, summary, scenarioManifest };
}

test('paired frozen validation binds both exact lanes, offsets, selector challenges, and identities', () => {
  const spec = frozenSpec();
  const validation = createFrozenValidation();
  const result = validateExactValidation(validation, exactOptions(spec));
  assert.deepEqual(result.rejectionReasons, []);
  assert.match(result.semanticSha256, /^[0-9a-f]{64}$/);
  assert.equal(physicalBinSequenceIdentity(validation), '9'.repeat(64));
  const reverseStorageSpec = frozenSpec([...FROZEN_DEPTH_CROSSOVER_LANES].reverse());
  assert.deepEqual(validateExactValidation(
    createFrozenValidation('reverse'),
    exactOptions(reverseStorageSpec),
  ).rejectionReasons, []);

  const cases = [
    ['laneStorageOrder versus plan', (value) => { value.laneStorageOrder = 'reverse'; }],
    ['padding corruption count', (value) => { value.lanes.reverse.paddingCorruptionCount = 1; }],
    ['physical-bin sequence commitment', (value) => {
      value.lanes.reverse.depthBins.physicalBinSequenceCommitment.sha256 = '8'.repeat(64);
    }],
    ['lane segment digest', (value) => { value.selectorChallenges.reverse.sha256 = '7'.repeat(64); }],
    ['versus representation', (value) => { value.lifecycle.meshUuid = 'replacement-mesh'; }],
  ];
  for (const [message, mutate] of cases) {
    const tampered = structuredClone(validation);
    mutate(tampered);
    assert.match(
      validateExactValidation(tampered, exactOptions(spec)).rejectionReasons.join('; '),
      new RegExp(message),
      message,
    );
  }
});

test('paired frozen parity requires exact channels in both lanes and the paired snapshot', () => {
  const spec = frozenSpec();
  const parity = frozenParity();
  const context = {
    spec,
    geometryManifest: exactOptions(spec).geometryManifest,
    scenarioManifest: exactOptions(spec).scenarioManifest,
  };
  assert.deepEqual(validateFrozenCrossoverRenderParity(parity, context), []);
  assert.deepEqual(validateRenderParity(parity, context), []);
  const identity = renderParityIdentity(parity);

  const unpaired = { ...parityLane(), snapshotValidation: parity.snapshotValidation };
  assert.match(
    validateRenderParity(unpaired, context).join('; '),
    /frozen crossover render-parity kind|parity lanes/,
  );

  const changedLane = structuredClone(parity);
  changedLane.lanes[FROZEN_DEPTH_CROSSOVER_LANES[1]].depth.sha256 = '7'.repeat(64);
  assert.match(
    validateRenderParity(changedLane, context).join('; '),
    /cross-lane depth/,
  );
  assert.notEqual(renderParityIdentity(changedLane), identity);

  const changedSnapshot = structuredClone(parity);
  changedSnapshot.snapshotValidation.selectorChallenges.reverse.storageOffset = 0;
  assert.match(
    validateRenderParity(changedSnapshot, context).join('; '),
    /render-parity snapshot: frozen reverse selector challenge storage offset/,
  );
});

test('frozen crossover accepts only the exact 480-row schedule, serials, and one render UID', () => {
  const artifacts = trialArtifacts();
  const protocol = {
    schemaVersion: 2,
    warmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
    plannedScheduleSha256: frozenCrossoverScheduleSha256(0),
  };
  const validate = (value) => validateTrialRows(
    value.spec,
    value.rows,
    value.summary,
    value.validation,
    value.scenarioManifest,
    protocol,
  );
  assert.deepEqual(validate(artifacts), []);

  const cases = [
    ['laneId', (value) => {
      value.rows[17].laneId = value.rows[17].laneId === FROZEN_DEPTH_CROSSOVER_LANES[0]
        ? FROZEN_DEPTH_CROSSOVER_LANES[1]
        : FROZEN_DEPTH_CROSSOVER_LANES[0];
    }],
    ['selectorWriteSerial', (value) => { value.rows[200].selectorWriteSerial += 1; }],
    ['gpuRenderTimestampUidCount', (value) => { value.rows[479].gpuRenderTimestampUidCount = 2; }],
    ['plannedScheduleSha256', (value) => { value.rows[3].plannedScheduleSha256 = '0'.repeat(64); }],
    ['measured row count', (value) => { value.rows.pop(); value.summary.rowCount -= 1; }],
  ];
  for (const [message, mutate] of cases) {
    const tampered = structuredClone(artifacts);
    mutate(tampered);
    assert.match(validate(tampered).join('; '), new RegExp(message), message);
  }

  assert.match(validateTrialRows(
    artifacts.spec,
    artifacts.rows,
    artifacts.summary,
    artifacts.validation,
    artifacts.scenarioManifest,
    { ...protocol, plannedScheduleSha256: '0'.repeat(64) },
  ).join('; '), /planned schedule sha256/);
  assert.match(validateTrialRows(
    artifacts.spec,
    artifacts.rows,
    artifacts.summary,
    artifacts.validation,
    artifacts.scenarioManifest,
    {
      schemaVersion: 2,
      warmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
      measuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
    },
  ).join('; '), /planned schedule sha256 is missing/);
});

test('frozen completion rejects resource, target, camera, compute, and cache drift', () => {
  const validation = createFrozenValidation();
  const invariant = completionInvariant(validation);
  assert.deepEqual(validateFrozenCrossoverCompletionInvariant(invariant, {
    objectCount: OBJECT_COUNT,
    validation,
  }), []);

  const cases = [
    ['stable meshUuid', (value) => { value.meshUuidAtTimingEnd = 'new-mesh'; }],
    ['render-target width', (value) => {
      value.renderTargetWidthAtTimingStart = 1279;
      value.renderTargetWidthAtTimingEnd = 1279;
    }],
    ['cameraViewFnv64', (value) => { value.cameraViewFnv64AtTimingEnd = '0'.repeat(16); }],
    ['compute-call serial stability', (value) => { value.computeCallSerialAtTimingEnd += 1; }],
    ['total pipeline-cache stability', (value) => {
      value.totalPipelineCacheEntriesAtTimingEnd += 1;
    }],
  ];
  for (const [message, mutate] of cases) {
    const tampered = structuredClone(invariant);
    mutate(tampered);
    assert.match(
      validateFrozenCrossoverCompletionInvariant(tampered, {
        objectCount: OBJECT_COUNT,
        validation,
      }).join('; '),
      new RegExp(message),
      message,
    );
  }
});
