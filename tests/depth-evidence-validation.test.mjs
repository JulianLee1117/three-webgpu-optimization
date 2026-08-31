import assert from 'node:assert/strict';
import test from 'node:test';
import {
  renderParityIdentity,
  sha256Json,
  validateBenchmarkProtocolEnvironment,
  validateExactValidation,
  validateRenderParity,
  validateScenarioManifest,
  validateTrialRows,
} from '../scripts/evidence-validation.mjs';

const FRONT_MODE = 'fixed-slice-depth-front-to-back';
const REVERSE_MODE = 'fixed-slice-depth-reverse';
const OBJECT_COUNT = 16;
const BUCKET_COUNT = 2;
const VISIBLE_COUNT = 6;
const GLOBAL_DIGEST = 'a'.repeat(64);
const DEPTH_RANGE = Object.freeze({ near: 90, far: 190 });
const PARITY_BYTE_LENGTH = 1280 * 720 * 4;

function traversalFor(modeId) {
  return modeId === FRONT_MODE
    ? [0, 1, 2, 3, 4, 5, 6, 7]
    : [7, 6, 5, 4, 3, 2, 1, 0];
}

function orderedDepthEvidence(modeId, expectedCounts = [
  1, 1, 0, 1, 0, 0, 0, 0,
  0, 1, 1, 0, 0, 1, 0, 0,
]) {
  const traversal = traversalFor(modeId);
  const starts = new Array(BUCKET_COUNT * 8).fill(0);
  const totals = new Array(BUCKET_COUNT).fill(0);
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    let cursor = 0;
    for (const physicalBin of traversal) {
      const index = bucket * 8 + physicalBin;
      starts[index] = cursor;
      cursor += expectedCounts[index];
    }
    totals[bucket] = cursor;
  }
  return {
    pass: true,
    errors: [],
    binCount: 8,
    order: modeId === FRONT_MODE ? 'front-to-back' : 'reverse',
    traversal,
    expectedCounts: [...expectedCounts],
    actualCounts: [...expectedCounts],
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
    perBucket: [0, 1].map((bucket) => {
      const sha256 = String(bucket + 1).repeat(64);
      return {
        bucket,
        match: true,
        expected: { count: 3, sha256 },
        actual: { count: 3, declaredCount: 3, sha256 },
      };
    }),
  };
}

function depthRepresentation(modeId) {
  const traversal = traversalFor(modeId);
  return {
    kind: 'single-merged-geometry-depth-binned-fixed-slice',
    depthBinCount: 8,
    depthOrder: modeId === FRONT_MODE ? 'front-to-back' : 'reverse',
    binTraversal: traversal,
    depthBinRange: { ...DEPTH_RANGE },
    reverseOrderUniformValue: modeId === REVERSE_MODE,
    bundleRecordCallbackCount: 1,
    meshCount: 1,
    geometryIdentityCount: 1,
    materialIdentityCount: 1,
    commandCount: BUCKET_COUNT,
    zeroFirstInstanceCount: BUCKET_COUNT,
    computeDispatchCount: 4,
    computeDispatchWorkItems: [BUCKET_COUNT * 8, OBJECT_COUNT, BUCKET_COUNT, BUCKET_COUNT],
  };
}

function createDepthValidation(modeId, expectedCounts) {
  const records = [
    {
      bucket: 0,
      actual: { indexCount: 12, instanceCount: 3, firstIndex: 0, baseVertex: 0, firstInstance: 0 },
      expected: { indexCount: 12, instanceCount: 3, firstIndex: 0, baseVertex: 0, firstInstance: 0 },
    },
    {
      bucket: 1,
      actual: { indexCount: 18, instanceCount: 3, firstIndex: 12, baseVertex: 0, firstInstance: 0 },
      expected: { indexCount: 18, instanceCount: 3, firstIndex: 12, baseVertex: 0, firstInstance: 0 },
    },
  ];
  return {
    pass: true,
    kind: `${modeId}-exact-membership-and-depth-order`,
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
    commandValidation: {
      pass: true,
      errors: [],
      commandCount: BUCKET_COUNT,
      totalInstanceCount: VISIBLE_COUNT,
      records,
    },
    depthBins: orderedDepthEvidence(modeId, expectedCounts),
    overflow: 0,
    representation: depthRepresentation(modeId),
  };
}

function exactOptions() {
  return {
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
      layout: 'high-overlap',
      depthBinRange: { ...DEPTH_RANGE },
    },
  };
}

for (const modeId of [FRONT_MODE, REVERSE_MODE]) {
  test(`Node accepts exact ${modeId} membership, commands, bins, and representation`, () => {
    const validation = createDepthValidation(modeId);
    const result = validateExactValidation(validation, { ...exactOptions(), modeId });
    assert.deepEqual(result.rejectionReasons, []);
    assert.match(result.semanticSha256, /^[0-9a-f]{64}$/);
  });
}

test('depth exact validation rejects command, bin-layout, overflow, range, and lifecycle tampering', () => {
  const options = { ...exactOptions(), modeId: FRONT_MODE };
  const validation = createDepthValidation(FRONT_MODE);
  const cases = [
    ['exact validation kind', (value) => { value.kind = `${FRONT_MODE}-exact-membership`; }],
    ['actual.firstIndex', (value) => { value.commandValidation.records[1].actual.firstIndex = 0; }],
    [`${FRONT_MODE} overflow`, (value) => { value.overflow = 1; }],
    ['actual start', (value) => { value.depthBins.actualStarts[1] += 1; }],
    ['recomputed total', (value) => { value.depthBins.expectedBucketTotals[0] += 1; }],
    ['depth-bin traversal', (value) => { value.depthBins.traversal.reverse(); }],
    ['range near versus scenario manifest', (value) => { value.representation.depthBinRange.near += 1; }],
    ['compute dispatch work items', (value) => { value.representation.computeDispatchWorkItems[0] -= 1; }],
    ['bundle-record callback count', (value) => { value.representation.bundleRecordCallbackCount = 0; }],
    ['physical-bin sequence survivor count', (value) => {
      value.depthBins.physicalBinSequenceCommitment.survivorCount -= 1;
    }],
    ['physical-bin sequence sha256', (value) => {
      value.depthBins.physicalBinSequenceCommitment.sha256 = 'not-a-sha';
    }],
    ['unexpected schema', (value) => { value.depthBins.unvalidatedField = true; }],
  ];
  for (const [expectedMessage, tamper] of cases) {
    const tampered = structuredClone(validation);
    tamper(tampered);
    assert.match(
      validateExactValidation(tampered, options).rejectionReasons.join('; '),
      new RegExp(expectedMessage.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
      expectedMessage,
    );
  }

  const missingManifest = validateExactValidation(validation, {
    ...options,
    scenarioManifest: null,
  });
  assert.match(missingManifest.rejectionReasons.join('; '), /scenario manifest is missing/);
});

test('depth-bin evidence participates in the exact semantic digest', () => {
  const options = { ...exactOptions(), modeId: FRONT_MODE };
  const first = validateExactValidation(createDepthValidation(FRONT_MODE), options);
  const redistributedCounts = [
    0, 2, 0, 1, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 1, 0, 0,
  ];
  const second = validateExactValidation(
    createDepthValidation(FRONT_MODE, redistributedCounts),
    options,
  );
  assert.deepEqual(first.rejectionReasons, []);
  assert.deepEqual(second.rejectionReasons, []);
  assert.notEqual(first.semanticSha256, second.semanticSha256);

  const changedSequence = createDepthValidation(FRONT_MODE);
  changedSequence.depthBins.physicalBinSequenceCommitment.sha256 = '8'.repeat(64);
  const third = validateExactValidation(changedSequence, options);
  assert.deepEqual(third.rejectionReasons, []);
  assert.notEqual(first.semanticSha256, third.semanticSha256);
});

function parityChannel(format, arrayType, sha256) {
  return {
    format,
    arrayType,
    byteLength: PARITY_BYTE_LENGTH,
    sha256,
  };
}

function createRenderParity(modeId = FRONT_MODE) {
  const color = parityChannel('rgba8unorm', 'Uint8Array', 'b'.repeat(64));
  const depth = parityChannel('depth32float', 'Float32Array', 'c'.repeat(64));
  const objectId = parityChannel(
    'rgba8unorm-object-id-plus-one',
    'Uint8Array',
    'd'.repeat(64),
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
    snapshotValidation: createDepthValidation(modeId),
  };
}

function parityContext(modeId = FRONT_MODE) {
  const options = exactOptions();
  return {
    spec: {
      modeId,
      objectCount: options.objectCount,
      bucketCount: options.bucketCount,
    },
    geometryManifest: options.geometryManifest,
    scenarioManifest: options.scenarioManifest,
  };
}

test('render parity requires exact reversed-depth channels and same-snapshot validation', () => {
  const parity = createRenderParity();
  const context = parityContext();
  assert.deepEqual(validateRenderParity(parity, context), []);

  const cases = [
    ['reversedDepthBuffer', (value) => { value.reversedDepthBuffer = false; }],
    ['color.byteLength', (value) => { value.color.byteLength -= 4; }],
    [
      'stability.firstCapture.depth.byteLength',
      (value) => { value.stability.firstCapture.depth.byteLength -= 4; },
    ],
    ['object-ID validation pass', (value) => { value.objectIdValidation.pass = false; }],
    ['object-ID validation encoding', (value) => { value.objectIdValidation.encoding = 'rgb24'; }],
    [
      'coveredPixels is not a positive integer',
      (value) => {
        value.objectIdValidation.backgroundPixels = 1280 * 720;
        value.objectIdValidation.coveredPixels = 0;
      },
    ],
    ['object-ID out-of-range pixels', (value) => { value.objectIdValidation.outOfRangePixels = 1; }],
    ['object-ID non-visible pixels', (value) => { value.objectIdValidation.nonVisiblePixels = 1; }],
    ['classified pixel total', (value) => { value.objectIdValidation.backgroundPixels -= 1; }],
    [
      'render-parity snapshot: native command bucket 0.actual.indexCount',
      (value) => { value.snapshotValidation.commandValidation.records[0].actual.indexCount += 1; },
    ],
  ];
  for (const [expectedMessage, tamper] of cases) {
    const tampered = structuredClone(parity);
    tamper(tampered);
    assert.match(
      validateRenderParity(tampered, context).join('; '),
      new RegExp(expectedMessage.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
      expectedMessage,
    );
  }
});

test('render-parity snapshot is bound to the surrounding scenario and geometry manifests', () => {
  const parity = createRenderParity();

  const wrongScenario = structuredClone(parityContext());
  wrongScenario.scenarioManifest.expectedVisibleIdsCanonicalSha256 = 'e'.repeat(64);
  assert.match(
    validateRenderParity(parity, wrongScenario).join('; '),
    /render-parity snapshot: membership digest expected\.sha256/,
  );

  const wrongGeometry = structuredClone(parityContext());
  wrongGeometry.geometryManifest.geometries[0].index.count += 3;
  assert.match(
    validateRenderParity(parity, wrongGeometry).join('; '),
    /render-parity snapshot: native command bucket 0\.expected\.indexCount/,
  );
});

test('render-parity identity binds reversed depth, object-ID audit, and snapshot membership', () => {
  const parity = createRenderParity();
  const original = renderParityIdentity(parity);
  for (const tamper of [
    (value) => { value.reversedDepthBuffer = false; },
    (value) => { value.objectIdValidation.coveredPixels += 1; },
    (value) => { value.snapshotValidation.membershipDigests.actual.sha256 = 'e'.repeat(64); },
  ]) {
    const changed = structuredClone(parity);
    tamper(changed);
    assert.notEqual(renderParityIdentity(changed), original);
  }
});

test('depth protocol metadata is bound to reversed-depth and storage-limit page evidence', () => {
  const protocol = {
    matrixKind: 'depth-ordering',
    reversedDepthBuffer: true,
    minimumStorageBuffersPerShaderStage: 8,
  };
  const pageEnvironment = {
    reversedDepth: true,
    rendererReversedDepthBuffer: true,
    maxStorageBuffersPerShaderStage: 8,
  };
  assert.deepEqual(
    validateBenchmarkProtocolEnvironment(protocol, pageEnvironment),
    [],
  );
  assert.deepEqual(
    validateBenchmarkProtocolEnvironment({
      ...protocol,
      matrixKind: 'depth-ordering-render-only',
    }, pageEnvironment),
    [],
  );

  for (const [expectedMessage, mutate] of [
    ['protocol reversedDepthBuffer', (p) => { p.reversedDepthBuffer = false; }],
    [
      'protocol minimumStorageBuffersPerShaderStage',
      (p) => { p.minimumStorageBuffersPerShaderStage = 7; },
    ],
    [
      'page rendererReversedDepthBuffer',
      (_p, environment) => { environment.rendererReversedDepthBuffer = false; },
    ],
    [
      'page maxStorageBuffersPerShaderStage',
      (_p, environment) => { environment.maxStorageBuffersPerShaderStage = 7; },
    ],
  ]) {
    const changedProtocol = structuredClone(protocol);
    const changedEnvironment = structuredClone(pageEnvironment);
    mutate(changedProtocol, changedEnvironment);
    assert.match(
      validateBenchmarkProtocolEnvironment(changedProtocol, changedEnvironment).join('; '),
      new RegExp(expectedMessage),
    );
  }

  const frozenProtocol = {
    ...protocol,
    matrixKind: 'depth-ordering-render-only',
  };
  assert.match(
    validateBenchmarkProtocolEnvironment(frozenProtocol, {
      ...pageEnvironment,
      rendererReversedDepthBuffer: false,
    }).join('; '),
    /page rendererReversedDepthBuffer/,
  );
  assert.match(
    validateBenchmarkProtocolEnvironment(frozenProtocol, {
      ...pageEnvironment,
      reversedDepth: false,
    }).join('; '),
    /page reversedDepth/,
  );
  assert.match(
    validateBenchmarkProtocolEnvironment({
      ...frozenProtocol,
      minimumStorageBuffersPerShaderStage: 7,
    }, pageEnvironment).join('; '),
    /protocol minimumStorageBuffersPerShaderStage/,
  );

  assert.deepEqual(validateBenchmarkProtocolEnvironment({ matrixKind: 'ecosystem' }, {}), []);
  assert.deepEqual(validateBenchmarkProtocolEnvironment({
    matrixKind: 'fixed-slice-representation',
  }, null), []);
});

function depthTrialArtifacts(modeId = FRONT_MODE) {
  const spec = {
    runId: 'depth-run',
    trialId: 'depth-trial',
    planIndex: 0,
    repetitionIndex: 0,
    modeId,
    modeOrder: ['fixed-slice', FRONT_MODE, REVERSE_MODE],
    modeOrderPosition: modeId === FRONT_MODE ? 1 : 0,
    visibilityFraction: 0.99,
    visibilityOrder: [0.99],
    visibilityOrderPosition: 0,
    layout: 'high-overlap',
    layoutOrder: ['high-overlap', 'low-overlap'],
    layoutOrderPosition: 0,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  };
  const representation = depthRepresentation(modeId);
  const validation = {
    kind: `${modeId}-exact-membership-and-depth-order`,
    representation,
  };
  const row = {
    schemaVersion: 2,
    runId: spec.runId,
    trialId: spec.trialId,
    planIndex: spec.planIndex,
    repetitionIndex: spec.repetitionIndex,
    modeOrderPosition: spec.modeOrderPosition,
    visibilityOrderPosition: spec.visibilityOrderPosition,
    layoutOrderPosition: spec.layoutOrderPosition,
    plannedModeOrder: spec.modeOrder.join('|'),
    plannedVisibilityOrder: spec.visibilityOrder.join('|'),
    plannedLayoutOrder: spec.layoutOrder.join('|'),
    protocolWarmupFrames: 300,
    protocolMeasuredFrames: 1,
    modeId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    targetVisibilityFraction: 0.99,
    scenarioLayout: spec.layout,
    depthBinRangeNear: DEPTH_RANGE.near,
    depthBinRangeFar: DEPTH_RANGE.far,
    expectedVisibleCount: VISIBLE_COUNT,
    validationKind: validation.kind,
    validationPass: true,
    usesCompute: true,
    configuredDrawCommands: BUCKET_COUNT,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 4,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    bundleRecordCallbackCountAtTimingStart: 1,
    timestampAvailable: true,
    frameIndex: 0,
    gpuFrameId: 10,
    cpuCommonUpdateMs: 0.01,
    cpuComputeSubmitMs: 0.03,
    cpuRenderSubmitMs: 0.1,
    cpuSubmitTotalMs: 0.13,
    cpuFrameBodyMs: 0.14,
    gpuComputeMs: 0.02,
    gpuRenderMs: 0.2,
    gpuPassTotalMs: 0.22,
  };
  const summary = {
    accepted: true,
    timestampAvailable: true,
    rowCount: 1,
    missingRenderFrames: 0,
    missingComputeFrames: 0,
    quantumNs: 32,
    classification: 'hardware-like',
    completionInvariant: {
      pass: true,
      kind: 'depth-binned-static-bundle-invariant',
      depthBinCount: 8,
      depthOrder: representation.depthOrder,
      binTraversal: [...representation.binTraversal],
      depthBinRange: { ...DEPTH_RANGE },
      reverseOrderUniformValue: representation.reverseOrderUniformValue,
      bundleRecordCallbackCountAtTimingStart: 1,
      bundleRecordCallbackCountAtTimingEnd: 1,
      meshCount: 1,
      geometryIdentityCount: 1,
      materialIdentityCount: 1,
      commandCount: BUCKET_COUNT,
      zeroFirstInstanceCount: BUCKET_COUNT,
      computeDispatchCount: 4,
      computeDispatchWorkItems: [...representation.computeDispatchWorkItems],
    },
  };
  const scenarioManifest = {
    expectedVisibleCount: VISIBLE_COUNT,
    layout: spec.layout,
    depthBinRange: { ...DEPTH_RANGE },
  };
  return { spec, validation, row, summary, scenarioManifest };
}

test('depth-ordering rows require layout/range audit, configured schedule, and completion invariant', () => {
  const artifacts = depthTrialArtifacts();
  const protocol = { schemaVersion: 2, warmupFrames: 300, measuredFrames: 1 };
  const validate = ({ spec, row, summary, validation, scenarioManifest }) => validateTrialRows(
    spec,
    [row],
    summary,
    validation,
    scenarioManifest,
    protocol,
  );
  assert.deepEqual(validate(artifacts), []);

  for (const field of [
    'scenarioLayout',
    'depthBinRangeNear',
    'depthBinRangeFar',
    'layoutOrderPosition',
    'plannedLayoutOrder',
    'configuredComputeDispatches',
    'configuredComputeSubmissions',
    'configuredRenderObjects',
  ]) {
    const tampered = structuredClone(artifacts);
    delete tampered.row[field];
    assert.match(validate(tampered).join('; '), new RegExp(`missing ${field}`));
  }

  const noCompletion = structuredClone(artifacts);
  delete noCompletion.summary.completionInvariant;
  assert.match(validate(noCompletion).join('; '), /completion invariant is not an object/);

  const rerecorded = structuredClone(artifacts);
  rerecorded.summary.completionInvariant.bundleRecordCallbackCountAtTimingEnd = 2;
  assert.match(validate(rerecorded).join('; '), /timing-end bundle-record callback count/);

  const wrongLayoutAudit = structuredClone(artifacts);
  wrongLayoutAudit.spec.layoutOrderPosition = 1;
  wrongLayoutAudit.row.layoutOrderPosition = 1;
  assert.match(validate(wrongLayoutAudit).join('; '), /selected layout position/);
});

test('atomic fixed-slice depth rows require a stable one-time bundle recording', () => {
  const artifacts = depthTrialArtifacts(FRONT_MODE);
  artifacts.spec.modeId = 'fixed-slice';
  artifacts.spec.modeOrderPosition = 0;
  Object.assign(artifacts.row, {
    modeId: 'fixed-slice',
    modeOrderPosition: 0,
    validationKind: 'fixed-slice-exact-membership',
    configuredComputeDispatches: 2,
    bundleRecordCallbackCountAtTimingStart: 1,
  });
  artifacts.validation = { kind: 'fixed-slice-exact-membership' };
  artifacts.summary.completionInvariant = {
    pass: true,
    kind: 'atomic-fixed-slice-static-bundle-invariant',
    bundleGroupStatic: true,
    bundleRecordCallbackCountAtTimingStart: 1,
    bundleRecordCallbackCountAtTimingEnd: 1,
    meshCount: 1,
    geometryIdentityCount: 1,
    materialIdentityCount: 1,
  };
  const protocol = { schemaVersion: 2, warmupFrames: 300, measuredFrames: 1 };
  assert.deepEqual(validateTrialRows(
    artifacts.spec,
    [artifacts.row],
    artifacts.summary,
    artifacts.validation,
    artifacts.scenarioManifest,
    protocol,
  ), []);

  const rerecorded = structuredClone(artifacts);
  rerecorded.summary.completionInvariant.bundleRecordCallbackCountAtTimingEnd = 2;
  assert.match(validateTrialRows(
    rerecorded.spec,
    [rerecorded.row],
    rerecorded.summary,
    rerecorded.validation,
    rerecorded.scenarioManifest,
    protocol,
  ).join('; '), /atomic fixed-slice timing-end bundle-record callback count/);

  delete artifacts.row.scenarioLayout;
  assert.match(validateTrialRows(
    artifacts.spec,
    [artifacts.row],
    artifacts.summary,
    artifacts.validation,
    artifacts.scenarioManifest,
    protocol,
  ).join('; '), /missing scenarioLayout/);
});

test('legacy scenario manifests remain valid without layout extensions', () => {
  const arrayDigest = 'f'.repeat(64);
  const arrays = {
    bucketCounts: { arrayType: 'Uint32Array', length: 2, sha256: arrayDigest },
    bucketBases: { arrayType: 'Uint32Array', length: 2, sha256: arrayDigest },
    visibleCounts: { arrayType: 'Uint32Array', length: 2, sha256: arrayDigest },
    objectBuckets: { arrayType: 'Uint32Array', length: 4, sha256: arrayDigest },
    matrices: { arrayType: 'Float32Array', length: 64, sha256: arrayDigest },
    bounds: { arrayType: 'Float32Array', length: 16, sha256: arrayDigest },
    expectedVisibleIds: { arrayType: 'Uint32Array', length: 2, sha256: arrayDigest },
    cullOrder: { arrayType: 'Uint32Array', length: 4, sha256: arrayDigest },
  };
  const canonical = {
    schemaVersion: 1,
    generator: 'createFixedSubsetScenario',
    seed: 7,
    objectCount: 4,
    bucketCount: 2,
    visibilityFraction: 0.5,
    expectedVisibleCount: 2,
    expectedVisibleIdsCanonicalSha256: arrayDigest,
    arrays,
  };
  const manifest = { ...canonical, sha256: sha256Json(canonical) };
  assert.deepEqual(validateScenarioManifest(manifest, {
    objectCount: 4,
    bucketCount: 2,
    visibilityFraction: 0.5,
    seed: 7,
  }), []);
});
