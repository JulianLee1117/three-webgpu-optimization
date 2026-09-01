import { access, mkdir } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import {
  FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceCrossoverFrame,
} from '../src/benchmark/first-instance-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  buildFirstInstanceCrossoverPlan,
} from '../src/benchmark/plan.js';
import { INDEXED_INDIRECT_STRIDE_BYTES } from '../src/culling/indexed-command-layout.js';
import {
  firstInstanceCrossoverScheduleSha256,
  validateFirstInstanceTrialEvidence,
} from './first-instance-evidence-validation.mjs';

const OBJECT_COUNT = 4_096;
const BUCKET_COUNT = 32;
const VISIBILITY_FRACTION = 0.99;
const LAYOUT = 'baseline';
const RECORDS_PER_LANE = Math.max(2, BUCKET_COUNT);
const COMMAND_SEGMENT_BYTE_LENGTH = RECORDS_PER_LANE * INDEXED_INDIRECT_STRIDE_BYTES;
const TIMED_FRAME_COUNT = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES
  + FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES;
const [PORTABLE, FEATURE] = FIRST_INSTANCE_CROSSOVER_LANES;
const SMOKE_RUN_ID = 'first-instance-crossover-smoke';
const SMOKE_SPEC = Object.freeze({
  ...buildFirstInstanceCrossoverPlan({
    runId: SMOKE_RUN_ID,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  })[0],
  runId: SMOKE_RUN_ID,
});

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const failureDirectory = path.join(projectRoot, '.test-output');
const browserCandidates = [
  process.env.BROWSER_PATH,
  path.join(
    process.env.ProgramFiles ?? 'C:\\Program Files',
    'Google',
    'Chrome',
    'Application',
    'chrome.exe',
  ),
  path.join(
    process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)',
    'Google',
    'Chrome',
    'Application',
    'chrome.exe',
  ),
  path.join(
    process.env.ProgramFiles ?? 'C:\\Program Files',
    'Microsoft',
    'Edge',
    'Application',
    'msedge.exe',
  ),
  process.env.LOCALAPPDATA
    && path.join(
      process.env.LOCALAPPDATA,
      'Google',
      'Chrome',
      'Application',
      'chrome.exe',
    ),
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

async function findBrowser() {
  for (const candidate of [...new Set(browserCandidates)]) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next installed browser.
    }
  }
  throw new Error('No installed Chrome, Chromium, or Edge executable found. Set BROWSER_PATH.');
}

function fail(message, evidence = null) {
  const detail = evidence === null ? '' : `: ${JSON.stringify(evidence)}`;
  throw new Error(`${message}${detail}`);
}

function requireCondition(condition, message, evidence = null) {
  if (!condition) fail(message, evidence);
}

function isSha256(value) {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
}

function isIdentity(value) {
  return (typeof value === 'string' && value.length > 0)
    || (Number.isInteger(value) && value >= 0);
}

function expectedCommandSegments(order) {
  return Object.fromEntries(order.map((laneId, segmentIndex) => [laneId, {
    index: segmentIndex,
    recordBase: segmentIndex * RECORDS_PER_LANE,
    byteBase: segmentIndex * COMMAND_SEGMENT_BYTE_LENGTH,
  }]));
}

function assertExactSegment(actual, expected, label) {
  requireCondition(
    actual?.index === expected.index
      && actual.recordBase === expected.recordBase
      && actual.byteBase === expected.byteBase,
    `${label} command segment is incorrect`,
    { actual, expected },
  );
}

function assertShaderEvidence(shaderEvidence) {
  requireCondition(
    shaderEvidence?.pass === true
      && shaderEvidence.kind === 'indirect-first-instance-shader-evidence'
      && Array.isArray(shaderEvidence.reasons)
      && shaderEvidence.reasons.length === 0
      && shaderEvidence.comparison?.rawVertexDifferent === true
      && shaderEvidence.comparison?.rawFragmentEqual === true
      && shaderEvidence.comparison?.normalizedVertexEqual === true
      && shaderEvidence.comparison?.normalizedVertexSha256Equal === true,
    'First-instance shader evidence failed',
    shaderEvidence,
  );

  const portable = shaderEvidence[PORTABLE];
  const feature = shaderEvidence[FEATURE];
  requireCondition(
    isSha256(portable?.raw?.vertex?.sha256)
      && isSha256(feature?.raw?.vertex?.sha256)
      && portable.raw.vertex.sha256 !== feature.raw.vertex.sha256
      && isSha256(portable.raw.fragment.sha256)
      && portable.raw.fragment.sha256 === feature.raw.fragment.sha256
      && isSha256(portable.normalizedVertex.sha256)
      && portable.normalizedVertex.sha256 === feature.normalizedVertex.sha256
      && portable.occurrenceCounts?.instanceIndex === 2
      && portable.occurrenceCounts?.bucketBase === 2
      && feature.occurrenceCounts?.instanceIndex === 2
      && feature.occurrenceCounts?.bucketBase === 0,
    'First-instance shader hashes or address occurrence counts are invalid',
    { portable, feature },
  );

  requireCondition(
    portable.vertexInputs?.map((input) => input.name).join('|')
      === 'position|bucketBase|normal'
      && feature.vertexInputs?.map((input) => input.name).join('|')
        === 'position|normal'
      && portable.vertexInputs.every((input) => input.stepMode === 'vertex')
      && feature.vertexInputs.every((input) => input.stepMode === 'vertex'),
    'Runtime vertex-input evidence does not isolate bucketBase',
    { portable: portable.vertexInputs, feature: feature.vertexInputs },
  );

  for (const semantic of ['matrix', 'visibleIds']) {
    const portableBinding = portable.storageBindings?.[semantic];
    const featureBinding = feature.storageBindings?.[semantic];
    requireCondition(
      portableBinding?.semantic === semantic
        && featureBinding?.semantic === semantic
        && portableBinding.resourceId === featureBinding.resourceId
        && portableBinding.group === featureBinding.group
        && portableBinding.binding === featureBinding.binding
        && portableBinding.access === 'read'
        && featureBinding.access === 'read'
        && portableBinding.visibility === 'vertex'
        && featureBinding.visibility === 'vertex',
      `Shader storage evidence for ${semantic} is not shared and vertex-read-only`,
      { portableBinding, featureBinding },
    );
  }

  requireCondition(
    typeof shaderEvidence.rawSources?.[PORTABLE]?.vertexShader === 'string'
      && shaderEvidence.rawSources[PORTABLE].vertexShader.includes('bucketBase')
      && typeof shaderEvidence.rawSources?.[FEATURE]?.vertexShader === 'string'
      && !shaderEvidence.rawSources[FEATURE].vertexShader.includes('bucketBase')
      && shaderEvidence.rawSources[PORTABLE].fragmentShader
        === shaderEvidence.rawSources[FEATURE].fragmentShader,
    'Raw WGSL does not expose the exact portable/feature contrast',
  );
}

function assertGeometryEvidence(geometry) {
  requireCondition(
    geometry?.pass === true
      && geometry.kind === 'shared-first-instance-geometry-evidence'
      && geometry.noInstanceSteppedAttributes === true
      && geometry.bucketBaseMismatchCount === 0
      && geometry.sharedIndex?.sameObject === true
      && geometry.bucketBase?.arrayType === 'Uint32Array'
      && geometry.bucketBase?.itemSize === 1
      && geometry.bucketBase?.isInstancedBufferAttribute === false
      && geometry.portableAttributeNames?.join('|') === 'bucketBase|normal|position|uv'
      && geometry.featureAttributeNames?.join('|') === 'normal|position|uv',
    'Shared geometry evidence failed',
    geometry,
  );
  requireCondition(
    geometry.sharedIndex.portable.id === geometry.sharedIndex.feature.id
      && geometry.sharedIndex.portable.sha256 === geometry.sharedIndex.feature.sha256
      && Object.keys(geometry.commonAttributes ?? {}).sort().join('|') === 'normal|position|uv',
    'Shared geometry index or attribute coverage is invalid',
    geometry,
  );
  for (const [name, record] of Object.entries(geometry.commonAttributes)) {
    requireCondition(
      record.sameObject === true
        && record.portable.id === record.feature.id
        && record.portable.version === record.feature.version
        && record.portable.sha256 === record.feature.sha256
        && record.portable.isInstancedBufferAttribute === false
        && record.feature.isInstancedBufferAttribute === false,
      `Common geometry attribute ${name} is not exact and shared`,
      record,
    );
  }
}

function assertLifecycle(lifecycle, order, segments) {
  requireCondition(
    lifecycle?.kind === 'first-instance-crossover-static-resource-lifecycle'
      && lifecycle.bundlesPrimed === true
      && lifecycle.allBundlesStatic === true
      && lifecycle.bundleCount === 2
      && lifecycle.meshCount === 2
      && lifecycle.activeRenderObjectCount === 1
      && lifecycle.laneCommandSegmentOrder?.join('|') === order.join('|')
      && lifecycle.configuredComputeDispatches === 0
      && lifecycle.configuredComputeSubmissions === 0
      && lifecycle.shaderEvidence?.pass === true,
    'First-instance lifecycle evidence failed',
    lifecycle,
  );
  for (const laneId of FIRST_INSTANCE_CROSSOVER_LANES) {
    assertExactSegment(lifecycle.commandSegments?.[laneId], segments[laneId], `${laneId} lifecycle`);
    requireCondition(
      lifecycle.bundleStaticFlags?.[laneId] === true
        && lifecycle.bundleRecordCounts?.[laneId] === 1
        && isIdentity(lifecycle.bundleUuids?.[laneId])
        && isIdentity(lifecycle.meshUuids?.[laneId])
        && isIdentity(lifecycle.geometryUuids?.[laneId])
        && isIdentity(lifecycle.materialUuids?.[laneId])
        && Number.isInteger(lifecycle.materialVersions?.[laneId])
        && lifecycle.materialVersions[laneId] >= 0,
      `${laneId} static bundle lifecycle is invalid`,
      lifecycle,
    );
  }
  requireCondition(
    lifecycle.bundleUuids[PORTABLE] !== lifecycle.bundleUuids[FEATURE]
      && lifecycle.meshUuids[PORTABLE] !== lifecycle.meshUuids[FEATURE]
      && lifecycle.geometryUuids[PORTABLE] !== lifecycle.geometryUuids[FEATURE]
      && lifecycle.materialUuids[PORTABLE] !== lifecycle.materialUuids[FEATURE]
      && isIdentity(lifecycle.indexAttributeId)
      && isIdentity(lifecycle.bucketBaseAttributeId)
      && isIdentity(lifecycle.matrixAttributeId)
      && isIdentity(lifecycle.visibleIdsAttributeId)
      && isIdentity(lifecycle.indirectAttributeId),
    'Lifecycle identities do not bind the two distinct lanes and shared resources',
    lifecycle,
  );
}

function assertPacking(validation, order, segments) {
  const packing = validation?.frozenPacking;
  const expectedVisibleCount = validation?.membership?.expectedCount;
  requireCondition(
    packing?.pass === true
      && packing.kind === 'first-instance-crossover-exact-frozen-packing-validation'
      && packing.metadata?.pass === true
      && packing.metadata.errors?.length === 0
      && packing.metadata.laneCommandSegmentOrder?.join('|') === order.join('|')
      && packing.metadata.recordsPerLane === RECORDS_PER_LANE
      && packing.metadata.commandSegmentByteLength === COMMAND_SEGMENT_BYTE_LENGTH
      && packing.visibleIds?.pass === true
      && packing.visibleIds.exactExpectedPacking === true
      && packing.visibleIds.elementCount === OBJECT_COUNT
      && packing.padding?.pass === true
      && packing.padding.activeSentinelCount === 0
      && packing.padding.paddingSentinelCount === packing.padding.paddingCount
      && packing.padding.corruptPaddingAddresses?.length === 0
      && packing.commands?.pass === true
      && packing.commands.exactExpectedPhysicalPacking === true
      && packing.commands.pairPass === true
      && packing.commitments?.commandCoresEqual === true
      && packing.commitments.pairs?.length === BUCKET_COUNT,
    'Frozen first-instance packing validation failed',
    packing,
  );
  requireCondition(
    isSha256(packing.commitments.visibleIdsSha256)
      && isSha256(packing.commitments.physicalCommandsSha256)
      && isSha256(packing.commitments.logicalPairSha256)
      && isSha256(packing.commitments.paddingSha256),
    'Frozen packing commitments are missing',
    packing.commitments,
  );
  for (const laneId of FIRST_INSTANCE_CROSSOVER_LANES) {
    const laneCommands = packing.commands.lanes?.[laneId];
    requireCondition(
      laneCommands?.pass === true
        && laneCommands.commandCount === BUCKET_COUNT
        && laneCommands.totalInstanceCount === expectedVisibleCount
        && laneCommands.errors?.length === 0,
      `${laneId} indirect commands are invalid`,
      laneCommands,
    );
  }
  for (const pair of packing.commitments.pairs) {
    requireCondition(
      pair.coreEqual === true
        && pair.portableFirstInstance === 0
        && pair.featureFirstInstance === pair.expectedFeatureFirstInstance,
      `Bucket ${pair.bucket} command pair differs outside firstInstance`,
      pair,
    );
  }
  for (const laneId of FIRST_INSTANCE_CROSSOVER_LANES) {
    const challengeSegment = validation.addressChallenges?.lanes?.[laneId]?.commandSegment;
    assertExactSegment(challengeSegment, segments[laneId], `${laneId} address challenge`);
  }
}

function assertAddressChallenges(validation, segments) {
  const addressChallenges = validation?.addressChallenges;
  const expectedVisibleCount = validation?.membership?.expectedCount;
  requireCondition(
    addressChallenges?.pass === true && addressChallenges.byteIdentical === true,
    'Cross-lane all-address challenge failed',
    addressChallenges,
  );
  const geometry = addressChallenges.geometry;
  requireCondition(
    geometry?.pass === true
      && geometry.kind === 'fragment-address-challenge-geometry-evidence'
      && geometry.topology === 'triangle-list'
      && geometry.pixelLocalCoordinates === true
      && geometry.target?.width === 256
      && geometry.target?.height === Math.ceil(OBJECT_COUNT / 256)
      && geometry.target?.pixelCount === OBJECT_COUNT
      && Number.isInteger(geometry.indexCount)
      && geometry.indexCount >= BUCKET_COUNT * 3
      && geometry.indexCount % 3 === 0
      && geometry.addressedTriangleCount === BUCKET_COUNT
      && geometry.degenerateTriangleCount
        === geometry.indexCount / 3 - BUCKET_COUNT
      && geometry.addressedTrianglesPerSubmittedInstance === 1
      && geometry.positionMismatchCount === 0
      && geometry.bucketBaseMismatchCount === 0
      && geometry.indexMismatchCount === 0
      && geometry.attributesExact === true
      && geometry.sharedPayloadExact === true
      && geometry.indirectIdentityExact === true
      && geometry.laneOffsetsExact?.[PORTABLE] === true
      && geometry.laneOffsetsExact?.[FEATURE] === true,
    'Fragment address challenge geometry is not exact',
    geometry,
  );
  for (const laneId of FIRST_INSTANCE_CROSSOVER_LANES) {
    const challenge = addressChallenges.lanes?.[laneId];
    requireCondition(
      challenge?.pass === true
        && challenge.kind === 'render-target-all-address-challenge'
        && challenge.outputStage === 'fragment'
        && challenge.addressTransport === 'vertex-address-to-rgba8-pixel'
        && challenge.encoding === 'rgb24-object-id-plus-one-transparent-zero-background'
        && challenge.lane === laneId
        && challenge.target?.pass === true
        && challenge.target?.width === 256
        && challenge.target?.height === Math.ceil(OBJECT_COUNT / 256)
        && challenge.target?.pixelCount === OBJECT_COUNT
        && challenge.target?.configuredFormat === 'RGBAFormat/UnsignedByteType'
        && challenge.target?.backendFormat === 'rgba8unorm'
        && challenge.target?.readbackArrayType === 'Uint8Array'
        && challenge.target?.bytesPerPixel === 4
        && challenge.target?.bytesPerRow === 1_024
        && challenge.target?.rowAlignmentBytes === 256
        && challenge.target?.origin === 'top-left'
        && challenge.target?.sampleLocation === 'integer-plus-half'
        && challenge.target?.samples === 0
        && challenge.target?.depthBuffer === false
        && challenge.target?.stencilBuffer === false
        && challenge.target?.scissorTest === false
        && challenge.target?.viewportExact === true
        && challenge.target?.scissorExact === true
        && challenge.target?.colorSpace === 'none'
        && challenge.target?.flipY === false
        && challenge.target?.generateMipmaps === false
        && challenge.reset?.pass === true
        && challenge.reset.pixelCount === OBJECT_COUNT
        && challenge.reset.addressCount === OBJECT_COUNT
        && challenge.reset.byteLength === OBJECT_COUNT * 4
        && challenge.reset.sha256 === challenge.reset.expectedSha256
        && isSha256(challenge.reset.sha256)
        && challenge.pixelCount === OBJECT_COUNT
        && challenge.addressCount === OBJECT_COUNT
        && challenge.byteLength === OBJECT_COUNT * 4
        && challenge.sha256 === challenge.expectedSha256
        && isSha256(challenge.sha256)
        && challenge.exactExpectedBytes === true
        && challenge.activeAddressCount === expectedVisibleCount
        && challenge.paddingAddressCount === OBJECT_COUNT - expectedVisibleCount
        && challenge.targetPaddingPixelCount === 0
        && challenge.activeMismatchCount === 0
        && challenge.paddingMismatchCount === 0
        && challenge.targetPaddingMismatchCount === 0
        && challenge.coverage?.allBucketsActive === true
        && challenge.coverage?.nonzeroBucketBaseCount === BUCKET_COUNT - 1
        && challenge.coverage?.activeRowCount > 1
        && challenge.coverage?.nonzeroEncodedChannelPixelCounts?.red > 0
        && challenge.coverage?.nonzeroEncodedChannelPixelCounts?.green > 0
        && challenge.coverage?.alphaEncodedPixelCount === expectedVisibleCount
        && typeof challenge.shader?.rawSources?.vertexShader === 'string'
        && typeof challenge.shader?.rawSources?.fragmentShader === 'string'
        && challenge.shader?.storageBindings?.length === 1
        && challenge.shader.storageBindings[0]?.semantic === 'visibleIds'
        && challenge.shader.storageBindings[0]?.access === 'read'
        && challenge.shader.storageBindings[0]?.elementType === 'u32'
        && challenge.shader?.vertexInputs?.every((input) => input.stepMode === 'vertex'),
      `${laneId} reset/address challenge failed`,
      challenge,
    );
    assertExactSegment(challenge.commandSegment, segments[laneId], `${laneId} challenge`);
  }
  requireCondition(
    addressChallenges.lanes[PORTABLE].sha256 === addressChallenges.lanes[FEATURE].sha256
      && addressChallenges.lanes[PORTABLE].reset.sha256
        === addressChallenges.lanes[FEATURE].reset.sha256
      && isSha256(addressChallenges.lanes[PORTABLE].shader?.vertexSha256)
      && isSha256(addressChallenges.lanes[FEATURE].shader?.vertexSha256)
      && addressChallenges.lanes[PORTABLE].shader.vertexSha256
        !== addressChallenges.lanes[FEATURE].shader.vertexSha256
      && addressChallenges.lanes[PORTABLE].shader.fragmentSha256
        === addressChallenges.lanes[FEATURE].shader.fragmentSha256
      && isSha256(addressChallenges.lanes[PORTABLE].shader.fragmentSha256)
      && addressChallenges.lanes[PORTABLE].shader.vertexLines.some(
        (line) => line.includes('( bucketBase + instanceIndex )'),
      )
      && !addressChallenges.lanes[FEATURE].shader.vertexLines.some(
        (line) => line.includes('bucketBase'),
      )
      && addressChallenges.lanes[FEATURE].shader.vertexLines.some(
        (line) => line.includes('featureChallengeAddress = instanceIndex'),
      )
      && FIRST_INSTANCE_CROSSOVER_LANES.every((laneId) => (
        addressChallenges.lanes[laneId].shader.vertexLines.filter(
          (line) => line.includes('var<storage, read>'),
        ).length === 1
        && !addressChallenges.lanes[laneId].shader.vertexLines.some(
          (line) => /atomic|read_write/.test(line),
        )
        && !addressChallenges.lanes[laneId].shader.fragmentLines.some(
          (line) => /atomic|var<storage/.test(line),
        )
      )),
    'Portable and feature address/reset commitments differ',
    addressChallenges,
  );
}

function assertValidation(validation, order) {
  const segments = expectedCommandSegments(order);
  const validationSummary = {
    pass: validation?.pass,
    kind: validation?.kind,
    expectedIdsMatchScenario: validation?.expectedIdsMatchScenario,
    frozenPackingPass: validation?.frozenPacking?.pass,
    membership: validation?.membership,
    membershipDigestsPass: validation?.membershipDigests?.pass,
    geometryPass: validation?.geometry?.pass,
    addressChallengesPass: validation?.addressChallenges?.pass,
    portableAddressPass: validation?.addressChallenges?.lanes?.[PORTABLE]?.pass,
    featureAddressPass: validation?.addressChallenges?.lanes?.[FEATURE]?.pass,
    portableResetPass: validation?.addressChallenges?.lanes?.[PORTABLE]?.reset?.pass,
    featureResetPass: validation?.addressChallenges?.lanes?.[FEATURE]?.reset?.pass,
    portableAddress: validation?.addressChallenges?.lanes?.[PORTABLE] ? {
      sha256: validation.addressChallenges.lanes[PORTABLE].sha256,
      expectedSha256: validation.addressChallenges.lanes[PORTABLE].expectedSha256,
      activeAddressCount: validation.addressChallenges.lanes[PORTABLE].activeAddressCount,
      paddingAddressCount: validation.addressChallenges.lanes[PORTABLE].paddingAddressCount,
      activeMismatchCount: validation.addressChallenges.lanes[PORTABLE].activeMismatchCount,
      paddingMismatchCount: validation.addressChallenges.lanes[PORTABLE].paddingMismatchCount,
      targetPaddingMismatchCount:
        validation.addressChallenges.lanes[PORTABLE].targetPaddingMismatchCount,
      shader: validation.addressChallenges.lanes[PORTABLE].shader,
    } : null,
    featureAddress: validation?.addressChallenges?.lanes?.[FEATURE] ? {
      sha256: validation.addressChallenges.lanes[FEATURE].sha256,
      expectedSha256: validation.addressChallenges.lanes[FEATURE].expectedSha256,
      activeAddressCount: validation.addressChallenges.lanes[FEATURE].activeAddressCount,
      paddingAddressCount: validation.addressChallenges.lanes[FEATURE].paddingAddressCount,
      activeMismatchCount: validation.addressChallenges.lanes[FEATURE].activeMismatchCount,
      paddingMismatchCount: validation.addressChallenges.lanes[FEATURE].paddingMismatchCount,
      targetPaddingMismatchCount:
        validation.addressChallenges.lanes[FEATURE].targetPaddingMismatchCount,
      shader: validation.addressChallenges.lanes[FEATURE].shader,
    } : null,
    addressChallengeGeometry: validation?.addressChallenges?.geometry,
    shaderPass: validation?.shaderEvidence?.pass,
    shaderReasons: validation?.shaderEvidence?.reasons,
    lifecycle: validation?.lifecycle,
  };
  requireCondition(
    validation?.pass === true
      && validation.kind === 'first-instance-crossover-exact-paired-snapshots'
      && validation.expectedIdsMatchScenario === true
      && validation.membership?.pass === true
      && validation.membership.errors === 0
      && validation.membership.expectedCount === validation.membership.listedCount
      && validation.membershipDigests?.pass === true
      && validation.membershipDigests.invalidExpectedIds === 0
      && validation.membershipDigests.truncatedActualIds === 0,
    'First-instance exact paired validation failed',
    validationSummary,
  );
  assertPacking(validation, order, segments);
  assertGeometryEvidence(validation.geometry);
  assertAddressChallenges(validation, segments);
  assertShaderEvidence(validation.shaderEvidence);
  assertLifecycle(validation.lifecycle, order, segments);
  return segments;
}

function stableValidation(validation) {
  const stable = structuredClone(validation);
  if (stable?.lifecycle) {
    delete stable.lifecycle.activeLane;
    delete stable.lifecycle.laneSelectionSerial;
  }
  return stable;
}

function assertStableValidation(left, right, message) {
  requireCondition(
    JSON.stringify(stableValidation(left)) === JSON.stringify(stableValidation(right)),
    message,
  );
}

function parityIdentity(parity) {
  return Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((laneId) => [
    laneId,
    Object.fromEntries(['color', 'depth', 'objectId'].map((channel) => [
      channel,
      parity?.lanes?.[laneId]?.[channel]?.sha256 ?? null,
    ])),
  ]));
}

function assertParity(parity, order) {
  requireCondition(
    parity?.pass === true
      && parity.kind === 'first-instance-crossover-exact-render-parity'
      && parity.crossLaneExact === true
      && parity.laneIds?.join('|') === FIRST_INSTANCE_CROSSOVER_LANES.join('|')
      && parity.snapshotValidation?.pass === true,
    'First-instance cross-lane parity failed',
    parity,
  );
  assertValidation(parity.snapshotValidation, order);
  for (const laneId of FIRST_INSTANCE_CROSSOVER_LANES) {
    const lane = parity.lanes?.[laneId];
    requireCondition(
      lane?.pass === true
        && lane.kind === 'fixed-camera-offscreen-exact-render-parity'
        && lane.width === 1_280
        && lane.height === 720
        && lane.captures === 2
        && lane.stability?.pass === true
        && lane.objectIdValidation?.pass === true
        && lane.objectIdValidation.coveredPixels > 0
        && lane.objectIdValidation.outOfRangePixels === 0
        && lane.objectIdValidation.nonVisiblePixels === 0
        && lane.reversedDepthBuffer === true
        && ['color', 'depth', 'objectId'].every((channel) => (
          isSha256(lane[channel]?.sha256)
            && lane[channel].sha256 === lane.stability.firstCapture?.[channel]?.sha256
        )),
      `${laneId} exact render parity capture failed`,
      lane,
    );
  }
  requireCondition(
    JSON.stringify(parityIdentity(parity)[PORTABLE])
      === JSON.stringify(parityIdentity(parity)[FEATURE]),
    'Portable and feature exact output commitments differ',
    parityIdentity(parity),
  );
}

function assertSelectedConfig(selectedConfig, order, orientationOffset) {
  requireCondition(
    selectedConfig?.strategyId === FIRST_INSTANCE_CROSSOVER_MODE
      && selectedConfig.objectCount === OBJECT_COUNT
      && selectedConfig.bucketCount === BUCKET_COUNT
      && selectedConfig.visibilityFraction === VISIBILITY_FRACTION
      && selectedConfig.layout === LAYOUT
      && selectedConfig.laneCommandSegmentOrder?.join('|') === order.join('|')
      && selectedConfig.superblockOrientationOffset === orientationOffset,
    'Page selected configuration differs from the requested crossover cell',
    selectedConfig,
  );
}

async function configure(page, { laneCommandSegmentOrder, orientationOffset }) {
  return page.evaluate(async (configuration) => {
    const bench = window.__WEBGPU_BENCH__;
    bench.configureFirstInstanceCrossover({
      laneCommandSegmentOrder: configuration.laneCommandSegmentOrder,
      superblockOrientationOffset: configuration.orientationOffset,
    });
    document.querySelector('#objects').value = String(configuration.objectCount);
    document.querySelector('#buckets').value = String(configuration.bucketCount);
    document.querySelector('#visibility').value = String(configuration.visibilityFraction);
    document.querySelector('#layout').value = configuration.layout;
    document.querySelector('#strategy').value = configuration.mode;
    await bench.rebuild();
    const validation = await bench.validate();
    if (validation?.pass !== true) {
      return {
        selectedConfig: bench.selectedConfig(),
        validation,
        parity: null,
        repeatedValidation: null,
      };
    }
    const parity = await bench.captureRenderParity();
    const repeatedValidation = await bench.validate();
    return {
      selectedConfig: bench.selectedConfig(),
      validation,
      parity,
      repeatedValidation,
    };
  }, {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: VISIBILITY_FRACTION,
    layout: LAYOUT,
    laneCommandSegmentOrder,
    orientationOffset,
    mode: FIRST_INSTANCE_CROSSOVER_MODE,
  });
}

function assertCompletionInvariant(invariant, order, segments) {
  requireCondition(
    invariant?.pass === true
      && invariant.kind === 'first-instance-crossover-static-resource-invariant'
      && invariant.bundlesPrimed === true
      && invariant.shaderEvidencePass === true
      && invariant.configuredCommandOrderExact === true
      && invariant.commandSegmentsExact === true
      && invariant.lifecycleExact === true
      && invariant.lifecycleCommitmentAtTimingStart
        === invariant.lifecycleCommitmentAtTimingEnd
      && invariant.selectorWritesDuringTiming === TIMED_FRAME_COUNT
      && invariant.strategySelectionsDuringTiming === TIMED_FRAME_COUNT
      && invariant.renderCallsDuringTiming === TIMED_FRAME_COUNT
      && invariant.computeCallsDuringTiming === 0
      && invariant.expectedTimedFrameCount === TIMED_FRAME_COUNT
      && invariant.cameraViewFnv64AtTimingStart === invariant.cameraViewFnv64AtTimingEnd
      && invariant.cameraProjectionFnv64AtTimingStart
        === invariant.cameraProjectionFnv64AtTimingEnd
      && invariant.renderTargetTextureUuidAtTimingStart
        === invariant.renderTargetTextureUuidAtTimingEnd
      && invariant.renderTargetWidthAtTimingStart === invariant.renderTargetWidthAtTimingEnd
      && invariant.renderTargetHeightAtTimingStart === invariant.renderTargetHeightAtTimingEnd
      && invariant.renderTargetSamplesAtTimingStart === invariant.renderTargetSamplesAtTimingEnd
      && invariant.renderTargetDepthBufferAtTimingStart
        === invariant.renderTargetDepthBufferAtTimingEnd
      && invariant.totalPipelineCacheEntriesAtTimingStart
        === invariant.totalPipelineCacheEntriesAtTimingEnd
      && invariant.computePipelineCacheEntriesAtTimingStart
        === invariant.computePipelineCacheEntriesAtTimingEnd,
    'First-instance completion invariant failed',
    invariant,
  );
  requireCondition(
    invariant.bundleRecordCounts?.[PORTABLE] === 1
      && invariant.bundleRecordCounts?.[FEATURE] === 1
      && invariant.plannedLaneCommandSegmentOrder === order.join('|')
      && invariant.observedLaneCommandSegmentOrder === order.join('|')
      && invariant.representation?.kind === 'frozen-first-instance-addressing-crossover'
      && invariant.representation.activeRenderObjectCount === 1
      && invariant.representation.geometryIdentityCount === 2
      && invariant.representation.materialIdentityCount === 2
      && invariant.representation.commonIndexIdentityCount === 1
      && invariant.representation.commandRecordCount === RECORDS_PER_LANE * 2
      && invariant.representation.visibleIdsCount === OBJECT_COUNT,
    'Completion representation or bundle counts are invalid',
    invariant,
  );
  for (const laneId of FIRST_INSTANCE_CROSSOVER_LANES) {
    assertExactSegment(
      invariant.observedCommandSegments?.[laneId],
      segments[laneId],
      `${laneId} completion`,
    );
  }
  requireCondition(
    JSON.stringify(invariant.staticLifecycleAtTimingStart)
      === JSON.stringify(invariant.staticLifecycleAtTimingEnd),
    'Static lifecycle snapshot changed during timing',
    invariant,
  );
  assertLifecycle(invariant.staticLifecycleAtTimingEnd, order, segments);
}

function assertTiming(timing, order, orientationOffset) {
  const segments = expectedCommandSegments(order);
  requireCondition(
    timing?.phase === 'complete'
      && timing.error === null
      && timing.summary?.accepted === true
      && timing.rows?.length === FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES
      && timing.summary.rowCount === FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES
      && timing.summary.missingRenderFrames === 0
      && timing.summary.invalidRenderTimestampUidCountFrames === 0
      && timing.summary.expectedRenderTimestampUidCount === 1
      && timing.summary.missingComputeFrames === 0
      && Number.isFinite(timing.summary.quantumNs)
      && timing.summary.quantumNs <= 10_000,
    'First-instance timed crossover failed',
    timing,
  );
  assertCompletionInvariant(timing.summary.completionInvariant, order, segments);

  const seenGpuFrameIds = new Set();
  for (let index = 0; index < timing.rows.length; index += 1) {
    const row = timing.rows[index];
    const scheduled = firstInstanceCrossoverFrame(index, orientationOffset);
    const segment = segments[scheduled.laneId];
    const expectedTimedOrdinal = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES + index + 1;
    const rowValid = row.frameIndex === index
      && row.phaseFrameIndex === index
      && row.modeId === FIRST_INSTANCE_CROSSOVER_MODE
      && row.objectCount === OBJECT_COUNT
      && row.bucketCount === BUCKET_COUNT
      && row.targetVisibilityFraction === VISIBILITY_FRACTION
      && row.scenarioLayout === LAYOUT
      && row.plannedLaneCommandSegmentOrder === order.join('|')
      && row.superblockOrientationOffset === orientationOffset
      && row.crossoverBlockIndex === scheduled.crossoverBlockIndex
      && row.withinBlockPosition === scheduled.withinBlockPosition
      && row.crossoverPattern === scheduled.pattern
      && row.crossoverPatternIndex === scheduled.patternIndex
      && row.laneId === scheduled.laneId
      && row.commandSegmentIndex === segment.index
      && row.commandRecordBase === segment.recordBase
      && row.commandByteBase === segment.byteBase
      && row.selectorWriteSerial
        === row.selectorWriteSerialAtTimingStart + expectedTimedOrdinal
      && row.strategySelectionSerial
        === row.strategySelectionSerialAtTimingStart + expectedTimedOrdinal
      && row.renderCallSerial === row.renderCallSerialAtTimingStart + expectedTimedOrdinal
      && Number.isInteger(row.gpuFrameId)
      && !seenGpuFrameIds.has(row.gpuFrameId)
      && row.gpuRenderTimestampUidCount === 1
      && Number.isFinite(row.gpuRenderMs)
      && row.gpuRenderMs >= 0
      && row.gpuComputeMs === null
      && row.gpuPassTotalMs === row.gpuRenderMs
      && row.cpuComputeSubmitMs === null
      && row.usesCompute === false
      && row.configuredDrawCommands === BUCKET_COUNT
      && row.configuredRenderObjects === 1
      && row.configuredComputeDispatches === 0
      && row.configuredComputeSubmissions === 0
      && row.configuredSubmittedInstances === row.expectedVisibleCount;
    if (!rowValid) fail(`First-instance crossover row ${index} violates the exact schedule`, row);
    seenGpuFrameIds.add(row.gpuFrameId);
    if (index > 0) {
      const previous = timing.rows[index - 1];
      requireCondition(
        row.selectorWriteSerial === previous.selectorWriteSerial + 1
          && row.strategySelectionSerial === previous.strategySelectionSerial + 1
          && row.renderCallSerial === previous.renderCallSerial + 1
          && row.gpuFrameId === previous.gpuFrameId + 1,
        `First-instance serials are not contiguous at row ${index}`,
        { previous, row },
      );
    }
  }
}

await mkdir(failureDirectory, { recursive: true });
const executablePath = await findBrowser();
const server = await createServer({
  root: projectRoot,
  configFile: false,
  resolve: { dedupe: ['three'] },
  server: {
    host: '127.0.0.1',
    port: 0,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  logLevel: 'error',
});
let browser;
let page;
const capturedBrowserErrors = [];
try {
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  if (!url) throw new Error('Vite did not expose a local URL.');
  browser = await chromium.launch({
    executablePath,
    headless: true,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-webgpu-developer-features',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 1_280, height: 900 },
    deviceScaleFactor: 1,
  });
  page = await context.newPage();
  const errors = capturedBrowserErrors;
  page.on('pageerror', (error) => errors.push(error.stack ?? error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__WEBGPU_BENCH__?.ready === true,
    null,
    { timeout: 120_000 },
  );
  const pageEnvironment = await page.evaluate(() => window.__WEBGPU_BENCH__.environment);
  requireCondition(
    pageEnvironment.indirectFirstInstanceAvailable === true
      && pageEnvironment.timestampAvailable === true
      && pageEnvironment.reversedDepth === true
      && pageEnvironment.rendererReversedDepthBuffer === true,
    'First-instance crossover WebGPU feature/timestamp/depth environment is invalid',
    pageEnvironment,
  );

  const forwardOrder = [...FIRST_INSTANCE_CROSSOVER_LANES];
  const pre = await configure(page, {
    laneCommandSegmentOrder: forwardOrder,
    orientationOffset: 0,
  });
  assertSelectedConfig(pre.selectedConfig, forwardOrder, 0);
  assertValidation(pre.validation, forwardOrder);
  assertParity(pre.parity, forwardOrder);
  assertValidation(pre.repeatedValidation, forwardOrder);
  assertStableValidation(
    pre.validation,
    pre.parity.snapshotValidation,
    'Forward validation changed before parity capture.',
  );
  assertStableValidation(
    pre.validation,
    pre.repeatedValidation,
    'Forward validation changed before timing.',
  );

  const auditContext = {
    runId: SMOKE_SPEC.runId,
    trialId: SMOKE_SPEC.trialId,
    planIndex: SMOKE_SPEC.planIndex,
    repetitionIndex: SMOKE_SPEC.repetitionIndex,
    modeOrderPosition: SMOKE_SPEC.modeOrderPosition,
    visibilityOrderPosition: SMOKE_SPEC.visibilityOrderPosition,
    layoutOrderPosition: SMOKE_SPEC.layoutOrderPosition,
    plannedModeOrder: SMOKE_SPEC.modeOrder.join('|'),
    plannedVisibilityOrder: SMOKE_SPEC.visibilityOrder.join('|'),
    plannedLayoutOrder: SMOKE_SPEC.layoutOrder.join('|'),
    protocolWarmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
    protocolMeasuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
    plannedLaneCommandSegmentOrder: SMOKE_SPEC.laneCommandSegmentOrder.join('|'),
    superblockOrientationOffset: SMOKE_SPEC.superblockOrientationOffset,
    plannedScheduleSha256: firstInstanceCrossoverScheduleSha256(
      SMOKE_SPEC.superblockOrientationOffset,
    ),
    firstInstanceCrossoverSmoke: true,
  };
  await page.evaluate(async (context) => {
    await window.__WEBGPU_BENCH__.startTrial(context);
  }, auditContext);
  await page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: 180_000 },
  );
  const timing = await page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__.phase,
    error: window.__WEBGPU_BENCH__.trialError,
    rows: window.__WEBGPU_BENCH__.rows,
    summary: JSON.parse(document.querySelector('#details').textContent),
  }));
  assertTiming(timing, forwardOrder, 0);

  const strictTrialEvidence = await validateFirstInstanceTrialEvidence({
    spec: SMOKE_SPEC,
    environment: pageEnvironment,
    validation: pre.repeatedValidation,
    renderParity: pre.parity,
    rows: timing.rows,
    summary: timing.summary,
    protocol: {
      schemaVersion: 2,
      warmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
      measuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
      plannedScheduleSha256: auditContext.plannedScheduleSha256,
    },
  });
  requireCondition(
    strictTrialEvidence.pass === true,
    'Strict first-instance trial evidence validator rejected the real WebGPU capture',
    {
      ...strictTrialEvidence,
      diagnosticFragmentShader: pre.repeatedValidation.addressChallenges
        ?.lanes?.[PORTABLE]?.shader?.rawSources?.fragmentShader,
    },
  );

  const post = await page.evaluate(async () => ({
    validation: await window.__WEBGPU_BENCH__.validate(),
    parity: await window.__WEBGPU_BENCH__.captureRenderParity(),
  }));
  assertValidation(post.validation, forwardOrder);
  assertParity(post.parity, forwardOrder);
  assertStableValidation(
    pre.validation,
    post.validation,
    'Frozen first-instance validation changed after timing.',
  );
  assertStableValidation(
    post.validation,
    post.parity.snapshotValidation,
    'Postflight validation changed before parity capture.',
  );
  requireCondition(
    JSON.stringify(parityIdentity(pre.parity)) === JSON.stringify(parityIdentity(post.parity)),
    'Exact rendered commitments changed after timing',
  );

  const reversedOrder = [...FIRST_INSTANCE_CROSSOVER_LANES].reverse();
  const reversed = await configure(page, {
    laneCommandSegmentOrder: reversedOrder,
    orientationOffset: 1,
  });
  assertSelectedConfig(reversed.selectedConfig, reversedOrder, 1);
  assertValidation(reversed.validation, reversedOrder);
  assertParity(reversed.parity, reversedOrder);
  assertValidation(reversed.repeatedValidation, reversedOrder);
  assertStableValidation(
    reversed.validation,
    reversed.parity.snapshotValidation,
    'Reversed validation changed before parity capture.',
  );
  assertStableValidation(
    reversed.validation,
    reversed.repeatedValidation,
    'Reversed validation was not repeatable.',
  );

  const forwardCommitments = pre.validation.frozenPacking.commitments;
  const reversedCommitments = reversed.validation.frozenPacking.commitments;
  requireCondition(
    forwardCommitments.physicalCommandsSha256
      !== reversedCommitments.physicalCommandsSha256
      && forwardCommitments.logicalPairSha256 === reversedCommitments.logicalPairSha256
      && forwardCommitments.visibleIdsSha256 === reversedCommitments.visibleIdsSha256
      && FIRST_INSTANCE_CROSSOVER_LANES.every((laneId) => (
        forwardCommitments.lanes[laneId].commandsSha256
          === reversedCommitments.lanes[laneId].commandsSha256
      )),
    'Physical segment reversal changed logical commands or frozen survivors',
    { forwardCommitments, reversedCommitments },
  );
  requireCondition(
    JSON.stringify(parityIdentity(pre.parity))
      === JSON.stringify(parityIdentity(reversed.parity)),
    'Physical command-segment reversal changed exact rendered output',
  );
  if (errors.length > 0) throw new Error(errors.join('\n'));

  process.stdout.write(`${JSON.stringify({
    browser: { executable: path.basename(executablePath), version: browser.version() },
    backend: await page.locator('#backend').textContent(),
    indirectFirstInstanceAvailable: pageEnvironment.indirectFirstInstanceAvailable,
    workload: {
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      visibilityFraction: VISIBILITY_FRACTION,
      layout: LAYOUT,
    },
    timing: {
      rowCount: timing.rows.length,
      timestampQuantumNs: timing.summary.quantumNs,
      completionInvariant: timing.summary.completionInvariant,
    },
    commandSegmentOrders: [
      pre.selectedConfig.laneCommandSegmentOrder,
      reversed.selectedConfig.laneCommandSegmentOrder,
    ],
    exactParity: true,
    strictTrialEvidencePass: strictTrialEvidence.pass,
  }, null, 2)}\n`);
} catch (error) {
  if (page) {
    await page.screenshot({
      path: path.join(failureDirectory, 'first-instance-crossover-smoke-failure.png'),
      fullPage: true,
    });
  }
  if (capturedBrowserErrors.length > 0) {
    error.message = `${error.message}\nBrowser errors:\n${capturedBrowserErrors.join('\n')}`;
  }
  throw error;
} finally {
  await browser?.close();
  await server.close();
}
