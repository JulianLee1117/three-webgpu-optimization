import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  summarizeCsv,
  summarizeInput,
  validateBenchmarkPlan,
  validateProtocolMatrix,
} from '../analysis/summarize.mjs';
import {
  buildDepthOrderingPlan,
  createEcosystemModeOrders,
  DEPTH_ORDERING_LAYOUTS,
  DEPTH_ORDERING_MODES,
  DEPTH_ORDERING_VISIBILITY,
} from '../src/benchmark/plan.js';
import { validateExactValidation } from '../scripts/evidence-validation.mjs';

const GEOMETRY_SHA256 = '1'.repeat(64);
const HIGH_SCENARIO_SHA256 = '2'.repeat(64);
const LOW_SCENARIO_SHA256 = '3'.repeat(64);
const EXPECTED_VISIBLE_IDS_SHA256 = '7'.repeat(64);
const EXPECTED_VISIBLE_COUNT = Math.round(65_536 * 0.99);
const DEPTH_RANGE = Object.freeze({ near: 90, far: 190 });
const GEOMETRY_INDEX_COUNTS = Object.freeze(
  Array.from({ length: 32 }, (_, bucket) => 12 + bucket * 3),
);

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

function depthProtocol(overrides = {}) {
  return {
    matrix: 'depth-ordering-o65536-b32',
    matrixKind: 'depth-ordering',
    representationScaleRole: null,
    layouts: [...DEPTH_ORDERING_LAYOUTS],
    depthBinCount: 8,
    reversedDepthBuffer: true,
    minimumStorageBuffersPerShaderStage: 8,
    objectCount: 65_536,
    bucketCount: 32,
    heterogeneousComparator: null,
    modes: [...DEPTH_ORDERING_MODES],
    visibilityLevels: [DEPTH_ORDERING_VISIBILITY],
    repetitions: 6,
    warmupFrames: 300,
    measuredFrames: 240,
    ordering: 'all-six-mode-permutations-with-alternating-high-low-layout-order',
    renderParity: 'same-snapshot exact validation plus two stable offscreen captures of rgba8 color, depth32float, and encoded object ID',
    ...overrides,
  };
}

function depthPlan(runId = 'depth-run') {
  return buildDepthOrderingPlan({
    runId,
    modeOrders: createEcosystemModeOrders(DEPTH_ORDERING_MODES),
    objectCount: 65_536,
    bucketCount: 32,
  }).map((trial) => ({ ...trial, runId }));
}

function perBucketVisibleCounts() {
  const base = Math.floor((65_536 / 32) * 0.99);
  const counts = new Array(32).fill(base);
  for (let bucket = 0; bucket < EXPECTED_VISIBLE_COUNT - base * 32; bucket += 1) {
    counts[bucket] += 1;
  }
  return counts;
}

function exactValidationFor(modeId) {
  const counts = perBucketVisibleCounts();
  const membershipDigests = {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    encoding: 'sorted-uint32-little-endian',
    pass: true,
    invalidExpectedIds: 0,
    truncatedActualIds: 0,
    expected: { count: EXPECTED_VISIBLE_COUNT, sha256: EXPECTED_VISIBLE_IDS_SHA256 },
    actual: { count: EXPECTED_VISIBLE_COUNT, sha256: EXPECTED_VISIBLE_IDS_SHA256 },
    perBucket: counts.map((count, bucket) => {
      const digest = ((bucket % 15) + 1).toString(16).repeat(64);
      return {
        bucket,
        match: true,
        expected: { count, sha256: digest },
        actual: { count, declaredCount: count, sha256: digest },
      };
    }),
  };
  let firstIndex = 0;
  const records = counts.map((count, bucket) => {
    const command = {
      indexCount: GEOMETRY_INDEX_COUNTS[bucket],
      instanceCount: count,
      firstIndex,
      baseVertex: 0,
      firstInstance: 0,
    };
    firstIndex += GEOMETRY_INDEX_COUNTS[bucket];
    return {
      bucket,
      actual: { ...command },
      expected: { ...command },
    };
  });
  const validation = {
    pass: true,
    kind: modeId === 'fixed-slice'
      ? 'fixed-slice-exact-membership'
      : `${modeId}-exact-membership-and-depth-order`,
    membership: {
      pass: true,
      expectedCount: EXPECTED_VISIBLE_COUNT,
      listedCount: EXPECTED_VISIBLE_COUNT,
      duplicateIds: 0,
      outOfRangeIds: 0,
      wrongBucketIds: 0,
      listedHiddenIds: 0,
      missingVisibleIds: 0,
      overflow: 0,
      errors: 0,
    },
    membershipDigests,
    commandValidation: {
      pass: true,
      errors: [],
      commandCount: 32,
      totalInstanceCount: EXPECTED_VISIBLE_COUNT,
      records,
    },
    overflow: 0,
  };
  if (modeId !== 'fixed-slice') {
    const reverse = modeId === 'fixed-slice-depth-reverse';
    const traversal = reverse
      ? [7, 6, 5, 4, 3, 2, 1, 0]
      : [0, 1, 2, 3, 4, 5, 6, 7];
    const expectedCounts = new Array(32 * 8).fill(0);
    const starts = new Array(32 * 8).fill(0);
    for (let bucket = 0; bucket < 32; bucket += 1) {
      expectedCounts[bucket * 8] = counts[bucket];
      let cursor = 0;
      for (const physicalBin of traversal) {
        const binIndex = bucket * 8 + physicalBin;
        starts[binIndex] = cursor;
        cursor += expectedCounts[binIndex];
      }
    }
    validation.depthBins = {
      pass: true,
      errors: [],
      binCount: 8,
      order: reverse ? 'reverse' : 'front-to-back',
      traversal,
      expectedCounts,
      actualCounts: [...expectedCounts],
      expectedStarts: starts,
      actualStarts: [...starts],
      expectedBucketTotals: [...counts],
      commandCounts: [...counts],
      physicalBinSequenceCommitment: {
        schemaVersion: 1,
        hashAlgorithm: 'sha256',
        encoding: 'bucket-major-physical-bin-major-tagged-uint32-little-endian',
        bucketCount: 32,
        binCount: 8,
        recordCount: 32 * 8,
        survivorCount: EXPECTED_VISIBLE_COUNT,
        sha256: '8'.repeat(64),
      },
    };
    validation.representation = {
      kind: 'single-merged-geometry-depth-binned-fixed-slice',
      depthBinCount: 8,
      depthOrder: reverse ? 'reverse' : 'front-to-back',
      binTraversal: [...traversal],
      depthBinRange: { ...DEPTH_RANGE },
      reverseOrderUniformValue: reverse,
      bundleRecordCallbackCount: 1,
      meshCount: 1,
      geometryIdentityCount: 1,
      materialIdentityCount: 1,
      commandCount: 32,
      zeroFirstInstanceCount: 32,
      computeDispatchCount: 4,
      computeDispatchWorkItems: [32 * 8, 65_536, 32, 32],
    };
  }
  return validation;
}

function checkDepthValidationFixture(payload, modeId, layout) {
  return validateExactValidation(payload, {
    modeId,
    objectCount: 65_536,
    bucketCount: 32,
    expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
    expectedVisibleIdsCanonicalSha256: EXPECTED_VISIBLE_IDS_SHA256,
    geometryManifest: {
      geometries: GEOMETRY_INDEX_COUNTS.map((count) => ({ index: { count } })),
    },
    scenarioManifest: {
      layout,
      visibilityFraction: DEPTH_ORDERING_VISIBILITY,
      depthBinRange: { ...DEPTH_RANGE },
      expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
      expectedVisibleIdsCanonicalSha256: EXPECTED_VISIBLE_IDS_SHA256,
    },
  });
}

function renderParityFor(layout, modeId = 'fixed-slice') {
  const digestDigit = layout === 'high-overlap' ? '5' : '6';
  const channelSha256 = digestDigit.repeat(64);
  const channels = {
    color: {
      format: 'rgba8unorm',
      arrayType: 'Uint8Array',
      byteLength: 1280 * 720 * 4,
      sha256: channelSha256,
    },
    depth: {
      format: 'depth32float',
      arrayType: 'Float32Array',
      byteLength: 1280 * 720 * 4,
      sha256: channelSha256,
    },
    objectId: {
      format: 'rgba8unorm-object-id-plus-one',
      arrayType: 'Uint8Array',
      byteLength: 1280 * 720 * 4,
      sha256: channelSha256,
    },
  };
  return {
    schemaVersion: 1,
    kind: 'fixed-camera-offscreen-exact-render-parity',
    pass: true,
    width: 1280,
    height: 720,
    captures: 2,
    reversedDepthBuffer: true,
    material: { type: 'MeshStandardNodeMaterial', color: 0x88aaff },
    ...channels,
    objectIdValidation: {
      pass: true,
      encoding: 'rgb24-object-id-plus-one-zero-background',
      backgroundPixels: 1280 * 720 - 10_000,
      coveredPixels: 10_000,
      outOfRangePixels: 0,
      nonVisiblePixels: 0,
    },
    snapshotValidation: exactValidationFor(modeId),
    stability: {
      pass: true,
      firstCapture: structuredClone(channels),
      first: {
        colorSha256: channelSha256,
        depthSha256: channelSha256,
        objectIdSha256: channelSha256,
      },
    },
  };
}

function renderParityIdentity(parity) {
  return sha256(JSON.stringify({
    width: parity.width,
    height: parity.height,
    reversedDepthBuffer: parity.reversedDepthBuffer,
    material: parity.material,
    color: parity.color,
    depth: parity.depth,
    objectId: parity.objectId,
    objectIdValidation: parity.objectIdValidation,
    membershipSha256: parity.snapshotValidation.membershipDigests.actual.sha256,
  }));
}

function metricRow({ modeId, layout, repetitionIndex, value }) {
  return [
    modeId,
    '0.99',
    layout,
    String(repetitionIndex),
    'true',
    String(value),
    '1',
    String(value - 1),
    '0.2',
    '0.3',
    '0.4',
    '1.2',
    '0.7',
  ].join(',');
}

function depthGateCsv({ unstableHighOverlap = false, materialLowOverlap = false } = {}) {
  const header = [
    'modeId',
    'targetVisibilityFraction',
    'scenarioLayout',
    'repetitionIndex',
    'modeOrderPosition',
    'plannedModeOrder',
    'usesCompute',
    'gpuPassTotalMs',
    'gpuComputeMs',
    'gpuRenderMs',
    'cpuCommonUpdateMs',
    'cpuComputeSubmitMs',
    'cpuRenderSubmitMs',
    'cpuFrameBodyMs',
    'cpuSubmitTotalMs',
  ].join(',');
  const orders = createEcosystemModeOrders(DEPTH_ORDERING_MODES);
  const rows = [];
  for (const [repetitionIndex, modeOrder] of orders.entries()) {
    for (const layout of DEPTH_ORDERING_LAYOUTS) {
      for (const [modeOrderPosition, modeId] of modeOrder.entries()) {
        let gpuPassTotalMs;
        if (layout === 'high-overlap') {
          if (modeId === 'fixed-slice') gpuPassTotalMs = 12;
          else if (modeId === 'fixed-slice-depth-reverse') gpuPassTotalMs = 10;
          else gpuPassTotalMs = unstableHighOverlap && repetitionIndex >= 4 ? 10.2 : 8;
        } else {
          if (modeId === 'fixed-slice') gpuPassTotalMs = 10;
          else if (modeId === 'fixed-slice-depth-front-to-back') gpuPassTotalMs = 10.2;
          else gpuPassTotalMs = materialLowOverlap ? 10.5 : 10.22;
        }
        rows.push([
          modeId,
          0.99,
          layout,
          repetitionIndex,
          modeOrderPosition,
          modeOrder.join('|'),
          'true',
          gpuPassTotalMs,
          1,
          gpuPassTotalMs - 1,
          0.2,
          0.3,
          0.4,
          1.2,
          0.7,
        ].join(','));
      }
    }
  }
  return [header, ...rows].join('\n');
}

async function createDepthRunFixture(t, mutate = () => undefined) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'three-webgpu-depth-analysis-'));
  t.after(async () => {
    await rm(directory, { recursive: true, force: true });
  });

  const runId = 'depth-ordering-o65536-b32-test';
  const protocol = depthProtocol();
  const plan = depthPlan(runId);
  const scenarioDigest = (layout) => (
    layout === 'high-overlap' ? HIGH_SCENARIO_SHA256 : LOW_SCENARIO_SHA256
  );
  const captureFor = (trial, { includeRenderParity = false } = {}) => {
    const payload = exactValidationFor(trial.modeId);
    const exactCheck = checkDepthValidationFixture(payload, trial.modeId, trial.layout);
    if (exactCheck.rejectionReasons.length !== 0) {
      throw new Error(`Invalid depth fixture: ${exactCheck.rejectionReasons.join('; ')}`);
    }
    return {
      capturedAt: '2026-08-31T20:00:01.000Z',
      workload: {
        scenarioSeed: 123,
        geometryFixtureSha256: GEOMETRY_SHA256,
        scenarioSha256: scenarioDigest(trial.layout),
      },
      validation: {
        payloadSha256: sha256(JSON.stringify(payload)),
        semanticSha256: exactCheck.semanticSha256,
        payload,
      },
      renderParity: includeRenderParity
        ? renderParityFor(trial.layout, trial.modeId)
        : null,
      accepted: true,
      rejectionReasons: [],
    };
  };

  const validationArtifacts = plan.map((trial) => {
    const artifact = {
      schemaVersion: 2,
      trialId: trial.trialId,
      planIndex: trial.planIndex,
      repetitionIndex: trial.repetitionIndex,
      modeId: trial.modeId,
      visibilityFraction: trial.visibilityFraction,
      layout: trial.layout,
      objectCount: trial.objectCount,
      bucketCount: trial.bucketCount,
      selectedConfig: {
        strategyId: trial.modeId,
        objectCount: trial.objectCount,
        bucketCount: trial.bucketCount,
        visibilityFraction: trial.visibilityFraction,
        layout: trial.layout,
      },
      status: 'accepted',
      rejectionReasons: [],
      pre: captureFor(trial, { includeRenderParity: true }),
      timingStart: captureFor(trial),
      post: captureFor(trial),
    };
    artifact.sha256 = sha256(JSON.stringify(artifact));
    return artifact;
  });

  const trialSummaries = plan.map((trial, index) => ({
    ...trial,
    selectedConfig: validationArtifacts[index].selectedConfig,
    startedAt: '2026-08-31T20:00:00.000Z',
    completedAt: '2026-08-31T20:00:02.000Z',
    elapsedMs: 2_000,
    validation: { pass: true, artifactSha256: validationArtifacts[index].sha256 },
    timestamps: {
      accepted: true,
      available: true,
      rowCount: protocol.measuredFrames,
      missingRenderFrames: 0,
      missingComputeFrames: 0,
    },
    completionInvariant: trial.modeId === 'fixed-slice'
      ? {
        pass: true,
        kind: 'atomic-fixed-slice-static-bundle-invariant',
        bundleGroupStatic: true,
        bundleRecordCallbackCountAtTimingStart: 1,
        bundleRecordCallbackCountAtTimingEnd: 1,
        meshCount: 1,
        geometryIdentityCount: 1,
        materialIdentityCount: 1,
      }
      : (() => {
        const representation = validationArtifacts[index]
          .timingStart.validation.payload.representation;
        return {
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
          commandCount: 32,
          zeroFirstInstanceCount: 32,
          computeDispatchCount: 4,
          computeDispatchWorkItems: [...representation.computeDispatchWorkItems],
        };
      })(),
    accepted: true,
    rejectionReasons: [],
  }));

  const frameHeader = [
    'runId',
    'trialId',
    'planIndex',
    'repetitionIndex',
    'modeOrderPosition',
    'visibilityOrderPosition',
    'layoutOrderPosition',
    'plannedModeOrder',
    'plannedVisibilityOrder',
    'plannedLayoutOrder',
    'protocolWarmupFrames',
    'protocolMeasuredFrames',
    'frameIndex',
    'modeId',
    'targetVisibilityFraction',
    'scenarioLayout',
    'depthBinRangeNear',
    'depthBinRangeFar',
    'expectedVisibleCount',
    'objectCount',
    'bucketCount',
    'validationKind',
    'validationPass',
    'timestampAvailable',
    'usesCompute',
    'configuredDrawCommands',
    'configuredRenderObjects',
    'configuredComputeDispatches',
    'configuredComputeSubmissions',
    'configuredSubmittedInstances',
    'bundleRecordCallbackCountAtTimingStart',
    'gpuPassTotalMs',
    'gpuComputeMs',
    'gpuRenderMs',
    'cpuCommonUpdateMs',
    'cpuComputeSubmitMs',
    'cpuRenderSubmitMs',
    'cpuFrameBodyMs',
    'cpuSubmitTotalMs',
  ].join(',');
  const frameRows = plan.flatMap((trial) => Array.from(
    { length: protocol.measuredFrames },
    (_, frameIndex) => [
      runId,
      trial.trialId,
      trial.planIndex,
      trial.repetitionIndex,
      trial.modeOrderPosition,
      trial.visibilityOrderPosition,
      trial.layoutOrderPosition,
      trial.modeOrder.join('|'),
      trial.visibilityOrder.join('|'),
      trial.layoutOrder.join('|'),
      protocol.warmupFrames,
      protocol.measuredFrames,
      frameIndex,
      trial.modeId,
      trial.visibilityFraction,
      trial.layout,
      DEPTH_RANGE.near,
      DEPTH_RANGE.far,
      EXPECTED_VISIBLE_COUNT,
      trial.objectCount,
      trial.bucketCount,
      trial.modeId === 'fixed-slice'
        ? 'fixed-slice-exact-membership'
        : `${trial.modeId}-exact-membership-and-depth-order`,
      'true',
      'true',
      'true',
      32,
      1,
      trial.modeId === 'fixed-slice' ? 2 : 4,
      1,
      '',
      1,
      trial.modeId === 'fixed-slice-depth-front-to-back' ? 4 : 6,
      1,
      trial.modeId === 'fixed-slice-depth-front-to-back' ? 3 : 5,
      0.2,
      0.3,
      0.4,
      1.2,
      0.7,
    ].join(','),
  ));

  const telemetrySummary = { status: 'unavailable', reason: 'test fixture' };
  const metadata = {
    schemaVersion: 2,
    runId,
    status: 'complete',
    startedAt: '2026-08-31T20:00:00.000Z',
    completedAt: '2026-08-31T20:00:02.000Z',
    elapsedMs: 2_000,
    environment: {
      benchmarkPage: {
        rendererReversedDepthBuffer: true,
        maxStorageBuffersPerShaderStage: 8,
      },
      gpuTelemetry: telemetrySummary,
    },
    evidenceStatus: 'development',
    sourceProvenance: {
      start: { status: 'unavailable' },
      end: { status: 'unavailable' },
      stable: null,
    },
    workload: {
      scenarioGenerator: 'createFixedSubsetScenario',
      scenarioSeed: 123,
      manifestArtifact: 'workload-manifests.json',
      geometryFixtureSha256: GEOMETRY_SHA256,
      scenarioSha256ByVisibility: null,
      scenarioSha256ByCell: {
        'high-overlap|0.99': HIGH_SCENARIO_SHA256,
        'low-overlap|0.99': LOW_SCENARIO_SHA256,
      },
      renderParitySha256ByCell: {
        'high-overlap|0.99': renderParityIdentity(renderParityFor('high-overlap')),
        'low-overlap|0.99': renderParityIdentity(renderParityFor('low-overlap')),
      },
      physicalBinSequenceSha256ByPair: Object.fromEntries(
        Array.from({ length: 6 }, (_, repetitionIndex) => (
          DEPTH_ORDERING_LAYOUTS.map((layout) => [
            [
              repetitionIndex,
              layout,
              DEPTH_ORDERING_VISIBILITY,
              scenarioDigest(layout),
            ].join('|'),
            '8'.repeat(64),
          ])
        )).flat(),
      ),
    },
    protocol,
    plan,
    expectedTrialCount: plan.length,
    completedTrialCount: plan.length,
    acceptedTrialCount: plan.length,
    frameRowCount: frameRows.length,
    validationArtifactCount: validationArtifacts.length,
    validationArtifactSha256: validationArtifacts.map((artifact) => artifact.sha256),
    pageErrors: [],
    error: null,
  };
  const workloadManifests = {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    geometryFixturesBySha256: {
      [GEOMETRY_SHA256]: {
        sha256: GEOMETRY_SHA256,
        geometries: GEOMETRY_INDEX_COUNTS.map((count) => ({ index: { count } })),
      },
    },
    scenariosBySha256: {
      [HIGH_SCENARIO_SHA256]: {
        sha256: HIGH_SCENARIO_SHA256,
        layout: 'high-overlap',
        visibilityFraction: 0.99,
        depthBinRange: { ...DEPTH_RANGE },
        expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
        expectedVisibleIdsCanonicalSha256: EXPECTED_VISIBLE_IDS_SHA256,
      },
      [LOW_SCENARIO_SHA256]: {
        sha256: LOW_SCENARIO_SHA256,
        layout: 'low-overlap',
        visibilityFraction: 0.99,
        depthBinRange: { ...DEPTH_RANGE },
        expectedVisibleCount: EXPECTED_VISIBLE_COUNT,
        expectedVisibleIdsCanonicalSha256: EXPECTED_VISIBLE_IDS_SHA256,
      },
    },
    invalidObservations: [],
  };
  const fixture = {
    metadata,
    trialSummaries,
    validationArtifacts,
    workloadManifests,
    telemetrySummary,
    framesCsv: [frameHeader, ...frameRows].join('\n'),
  };
  mutate(fixture);
  for (const [index, artifact] of fixture.validationArtifacts.entries()) {
    delete artifact.sha256;
    artifact.sha256 = sha256(JSON.stringify(artifact));
    fixture.trialSummaries[index].validation.artifactSha256 = artifact.sha256;
  }
  fixture.metadata.validationArtifactSha256 = fixture.validationArtifacts.map(
    (artifact) => artifact.sha256,
  );

  const artifacts = new Map([
    ['frames.csv', Buffer.from(fixture.framesCsv)],
    ['metadata.json', jsonBytes(fixture.metadata)],
    ['trial-summaries.json', jsonBytes(fixture.trialSummaries)],
    ['validation-artifacts.json', jsonBytes(fixture.validationArtifacts)],
    ['workload-manifests.json', jsonBytes(fixture.workloadManifests)],
    ['gpu-telemetry-summary.json', jsonBytes(fixture.telemetrySummary)],
  ]);
  await Promise.all([...artifacts].map(([name, contents]) => (
    writeFile(path.join(directory, name), contents)
  )));
  const manifest = {
    schemaVersion: 2,
    runId,
    hashAlgorithm: 'sha256',
    requiredFiles: [...artifacts.keys()],
    optionalFiles: [{
      name: 'gpu-telemetry.csv',
      present: false,
      evidenceAvailable: false,
      absenceReason: 'test fixture has no device telemetry',
    }],
    files: [
      ...[...artifacts].map(([name, contents]) => ({
        name,
        role: `test ${name}`,
        required: true,
        present: true,
        bytes: contents.length,
        sha256: sha256(contents),
        absenceReason: null,
      })),
      {
        name: 'gpu-telemetry.csv',
        role: 'optional device telemetry samples',
        required: false,
        present: false,
        bytes: null,
        sha256: null,
        absenceReason: 'test fixture has no device telemetry',
      },
    ],
  };
  await writeFile(path.join(directory, 'artifact-manifest.json'), jsonBytes(manifest));
  return directory;
}

test('depth-ordering protocol requires the exact causal matrix', () => {
  const valid = depthProtocol();
  assert.doesNotThrow(() => validateProtocolMatrix(valid));
  for (const [mutate, pattern] of [
    [(protocol) => { protocol.modes.reverse(); }, /protocol modes must be exactly/],
    [(protocol) => { protocol.layouts.reverse(); }, /layouts must be exactly/],
    [(protocol) => { protocol.visibilityLevels = [0.8]; }, /visibility levels must be exactly/],
    [(protocol) => { protocol.repetitions = 3; }, /exactly six repetitions/],
    [(protocol) => { protocol.objectCount = 16_384; }, /65536 objects and 32 buckets/],
    [(protocol) => { protocol.depthBinCount = 4; }, /exactly eight depth bins/],
    [(protocol) => { protocol.reversedDepthBuffer = false; }, /require reversedDepthBuffer/],
    [(protocol) => { protocol.minimumStorageBuffersPerShaderStage = 7; }, /exactly eight storage buffers/],
    [(protocol) => { protocol.matrix = 'depth-ordering-o65536-b4'; }, /matrix identifier/],
    [(protocol) => { protocol.ordering = 'balanced'; }, /depth-ordering ordering must be/],
    [(protocol) => { protocol.renderParity = null; }, /depth-ordering renderParity must be/],
  ]) {
    const invalid = structuredClone(valid);
    mutate(invalid);
    assert.throws(() => validateProtocolMatrix(invalid), pattern);
  }
});

test('depth-ordering plan requires all 36 cells in exact layout-outer, mode-inner order', () => {
  const protocol = depthProtocol();
  const metadata = { runId: 'depth-run', protocol };
  const valid = depthPlan();
  assert.equal(valid.length, 36);
  assert.equal(validateBenchmarkPlan(valid, metadata).byTrialId.size, 36);

  const reordered = structuredClone(valid);
  [reordered[0], reordered[1]] = [reordered[1], reordered[0]];
  reordered[0].planIndex = 0;
  reordered[1].planIndex = 1;
  assert.throws(
    () => validateBenchmarkPlan(reordered, metadata),
    /layout outer and mode inner/,
  );

  const wrongAlternation = structuredClone(valid);
  for (const trial of wrongAlternation.filter((entry) => entry.repetitionIndex === 1)) {
    trial.layoutOrder = [...DEPTH_ORDERING_LAYOUTS];
    trial.layoutOrderPosition = trial.layoutOrder.indexOf(trial.layout);
  }
  assert.throws(
    () => validateBenchmarkPlan(wrongAlternation, metadata),
    /layout orders must alternate/,
  );
});

test('depth analysis keeps high- and low-overlap cells separate in every paired contrast', () => {
  const header = [
    'modeId',
    'targetVisibilityFraction',
    'scenarioLayout',
    'repetitionIndex',
    'usesCompute',
    'gpuPassTotalMs',
    'gpuComputeMs',
    'gpuRenderMs',
    'cpuCommonUpdateMs',
    'cpuComputeSubmitMs',
    'cpuRenderSubmitMs',
    'cpuFrameBodyMs',
    'cpuSubmitTotalMs',
  ].join(',');
  const rows = [];
  for (let repetitionIndex = 0; repetitionIndex < 2; repetitionIndex += 1) {
    for (const layout of DEPTH_ORDERING_LAYOUTS) {
      const offset = layout === 'high-overlap' ? 0 : 10;
      rows.push(metricRow({
        modeId: 'fixed-slice',
        layout,
        repetitionIndex,
        value: 8 + offset,
      }));
      rows.push(metricRow({
        modeId: 'fixed-slice-depth-front-to-back',
        layout,
        repetitionIndex,
        value: 4 + offset,
      }));
      rows.push(metricRow({
        modeId: 'fixed-slice-depth-reverse',
        layout,
        repetitionIndex,
        value: 6 + offset,
      }));
    }
  }

  const summary = summarizeCsv([header, ...rows].join('\n'));
  assert.equal(summary.groups.length, 6);
  assert.equal(summary.groups.every((group) => group.nTrials === 2), true);
  const contrastSets = [
    summary.causalContrasts.depthFrontToBackVsReverse,
    summary.contextualWholeMechanismComparisons.depthFrontToBackVsAtomicFixedSlice,
    summary.contextualWholeMechanismComparisons.depthReverseVsAtomicFixedSlice,
  ];
  for (const contrasts of contrastSets) {
    assert.deepEqual(contrasts.map((contrast) => contrast.layout), [
      'high-overlap',
      'low-overlap',
    ]);
    assert.equal(contrasts.every((contrast) => contrast.nPairs === 2), true);
    assert.deepEqual(contrasts.flatMap((contrast) => (
      contrast.pairs.map((pair) => pair.repetition)
    )), [0, 1, 0, 1]);
  }
  assert.equal(
    summary.causalContrasts.depthFrontToBackVsReverse[0]
      .medianPairedDelta.gpuPassTotalMs.absoluteMs,
    -2,
  );
  assert.equal(
    summary.contextualWholeMechanismComparisons
      .depthFrontToBackVsAtomicFixedSlice[1]
      .medianPairedDelta.gpuPassTotalMs.absoluteMs,
    -4,
  );
  assert.equal(
    Object.hasOwn(
      summary.causalContrasts,
      'depthFrontToBackVsAtomicFixedSlice',
    ),
    false,
  );
});

test('depth analysis evaluates every numeric preregistered timing gate', () => {
  const summary = summarizeCsv(depthGateCsv());
  const decision = summary.preregisteredGates.depthOrdering;
  assert.equal(decision.status, 'evaluated');
  assert.equal(decision.pass, true);
  assert.deepEqual(decision.failedGates, []);

  const gates = decision.gates;
  assert.equal(gates.highOverlapOrderingBenefit.pass, true);
  assert.deepEqual(gates.highOverlapOrderingBenefit.threshold, {
    rule: 'median paired delta must be at most either threshold',
    absoluteMs: -0.1,
    percent: -10,
  });
  assert.equal(gates.highOverlapDirectionStability.pass, true);
  assert.equal(gates.highOverlapDirectionStability.frontFasterPairs, 6);
  assert.equal(gates.highOverlapDirectionStability.minimumFrontFasterPairs, 5);
  assert.equal(
    gates.highOverlapDirectionStability.frontPositionStrata.every(
      (stratum) => stratum.observedPairs === 2 && stratum.pass,
    ),
    true,
  );
  assert.equal(
    gates.highOverlapDirectionStability.relativeOrderStrata.every(
      (stratum) => stratum.observedPairs === 3 && stratum.pass,
    ),
    true,
  );
  assert.equal(gates.lowOverlapNegativeControl.pass, true);
  assert.deepEqual(gates.lowOverlapNegativeControl.equivalenceBounds, {
    rule: 'absolute median paired delta must remain strictly inside both bounds',
    absoluteMs: 0.1,
    percent: 10,
  });
  assert.equal(gates.highOverlapFrontVsAtomicWholeMechanism.pass, true);
  assert.equal(gates.lowOverlapFrontVsAtomicWholeMechanism.pass, true);
  assert.equal(
    gates.lowOverlapFrontVsAtomicWholeMechanism.maximumRegressionPercent,
    5,
  );
});

test('depth timing gates reject unstable direction and a material low-overlap difference', () => {
  const unstable = summarizeCsv(depthGateCsv({ unstableHighOverlap: true }))
    .preregisteredGates.depthOrdering;
  assert.equal(unstable.gates.highOverlapOrderingBenefit.pass, true);
  assert.equal(unstable.gates.highOverlapDirectionStability.frontFasterPairs, 4);
  assert.equal(unstable.gates.highOverlapDirectionStability.pass, false);
  assert.equal(unstable.pass, false);

  const materialControl = summarizeCsv(depthGateCsv({ materialLowOverlap: true }))
    .preregisteredGates.depthOrdering;
  assert.equal(materialControl.gates.lowOverlapNegativeControl.pass, false);
  assert.equal(materialControl.failedGates.includes('lowOverlapNegativeControl'), true);
  assert.equal(materialControl.pass, false);
});

test('verified depth run binds artifacts, frames, and scenario digests to layout cells', async (t) => {
  const directory = await createDepthRunFixture(t);
  const summary = await summarizeInput(directory);
  assert.equal(summary.artifactVerification.status, 'consistent');
  assert.equal(summary.nFrames, 36 * 240);
  assert.equal(summary.groups.length, 6);
  assert.equal(summary.causalContrasts.depthFrontToBackVsReverse.length, 2);
});

test('verified depth run rejects unequal within-bin sequences in a paired repetition', async (t) => {
  const directory = await createDepthRunFixture(t, ({ validationArtifacts }) => {
    const artifact = validationArtifacts.find((entry) => (
      entry.repetitionIndex === 0
      && entry.layout === 'high-overlap'
      && entry.modeId === 'fixed-slice-depth-reverse'
    ));
    for (const name of ['pre', 'timingStart', 'post']) {
      const capture = artifact[name];
      capture.validation.payload.depthBins.physicalBinSequenceCommitment.sha256 = '9'.repeat(64);
      capture.validation.payloadSha256 = sha256(JSON.stringify(capture.validation.payload));
      capture.validation.semanticSha256 = checkDepthValidationFixture(
        capture.validation.payload,
        artifact.modeId,
        artifact.layout,
      ).semanticSha256;
    }
    artifact.pre.renderParity.snapshotValidation
      .depthBins.physicalBinSequenceCommitment.sha256 = '9'.repeat(64);
  });
  await assert.rejects(
    summarizeInput(directory),
    /unequal traversal-normalized physical-bin sequences/,
  );
});

test('verified depth run independently revalidates payload semantics and lifecycle', async (t) => {
  const invalidPayload = await createDepthRunFixture(t, ({ validationArtifacts }) => {
    const artifact = validationArtifacts[0];
    for (const name of ['pre', 'timingStart', 'post']) {
      const capture = artifact[name];
      capture.validation.payload.overflow = 1;
      capture.validation.payloadSha256 = sha256(JSON.stringify(capture.validation.payload));
      capture.validation.semanticSha256 = checkDepthValidationFixture(
        capture.validation.payload,
        artifact.modeId,
        artifact.layout,
      ).semanticSha256;
    }
  });
  await assert.rejects(
    summarizeInput(invalidPayload),
    /exact payload failed: fixed-slice overflow/,
  );

  const invalidLifecycle = await createDepthRunFixture(t, ({ trialSummaries }) => {
    trialSummaries[0].completionInvariant.bundleRecordCallbackCountAtTimingEnd = 2;
  });
  await assert.rejects(
    summarizeInput(invalidLifecycle),
    /completion invariant failed: atomic fixed-slice timing-end bundle-record callback count/,
  );
});

test('verified depth run rejects a scenario digest linked to the wrong layout manifest', async (t) => {
  const directory = await createDepthRunFixture(t, ({ metadata }) => {
    metadata.workload.scenarioSha256ByCell['high-overlap|0.99'] = LOW_SCENARIO_SHA256;
  });
  await assert.rejects(
    summarizeInput(directory),
    /cell high-overlap\|0\.99 links the wrong scenario manifest/,
  );
});

test('verified depth run requires exact render-parity digest coverage for both layouts', async (t) => {
  const directory = await createDepthRunFixture(t, ({ metadata }) => {
    delete metadata.workload.renderParitySha256ByCell['low-overlap|0.99'];
  });
  await assert.rejects(
    summarizeInput(directory),
    /renderParitySha256ByCell must exactly cover every layout\/visibility cell/,
  );
});

test('verified depth run binds physical-bin sequence pairs into metadata', async (t) => {
  const missing = await createDepthRunFixture(t, ({ metadata }) => {
    delete metadata.workload.physicalBinSequenceSha256ByPair[
      `0|high-overlap|0.99|${HIGH_SCENARIO_SHA256}`
    ];
  });
  await assert.rejects(
    summarizeInput(missing),
    /must exactly cover every repetition\/layout pair/,
  );

  const disagreement = await createDepthRunFixture(t, ({ metadata }) => {
    metadata.workload.physicalBinSequenceSha256ByPair[
      `0|high-overlap|0.99|${HIGH_SCENARIO_SHA256}`
    ] = '9'.repeat(64);
  });
  await assert.rejects(
    summarizeInput(disagreement),
    /differs from metadata/,
  );
});

test('verified depth run requires reversed depth and the baseline storage-binding limit', async (t) => {
  for (const [name, mutate, pattern] of [
    [
      'renderer reversed depth',
      ({ metadata }) => { metadata.environment.benchmarkPage.rendererReversedDepthBuffer = false; },
      /does not prove renderer reversed-depth operation/,
    ],
    [
      'storage binding limit',
      ({ metadata }) => { metadata.environment.benchmarkPage.maxStorageBuffersPerShaderStage = 7; },
      /fewer than eight storage buffers/,
    ],
  ]) {
    await t.test(name, async (subtest) => {
      const directory = await createDepthRunFixture(subtest, mutate);
      await assert.rejects(summarizeInput(directory), pattern);
    });
  }
});

test('verified depth run rejects incomplete or inconsistent exact render parity', async (t) => {
  const cases = [
    [
      'capture reversed depth',
      (parity) => { parity.reversedDepthBuffer = false; },
      /did not use the required reversed depth buffer/,
    ],
    [
      'exact byte length',
      (parity) => { parity.depth.byteLength -= 4; },
      /depth has an invalid byte length/,
    ],
    [
      'full first capture',
      (parity) => { parity.stability.firstCapture.color.sha256 = '8'.repeat(64); },
      /color first-capture record is inconsistent/,
    ],
    [
      'object-ID domain',
      (parity) => { parity.objectIdValidation.outOfRangePixels = 1; },
      /objectIdValidation did not prove a valid encoded ID domain/,
    ],
    [
      'same-snapshot canonical membership',
      (parity) => {
        parity.snapshotValidation.membershipDigests.actual.sha256 = '8'.repeat(64);
      },
      /same-snapshot validation failed: membership digest actual\.sha256/,
    ],
  ];
  for (const [name, mutateParity, pattern] of cases) {
    await t.test(name, async (subtest) => {
      const directory = await createDepthRunFixture(subtest, ({ validationArtifacts }) => {
        mutateParity(validationArtifacts[0].pre.renderParity);
      });
      await assert.rejects(summarizeInput(directory), pattern);
    });
  }
});

test('verified depth run cannot reuse one layout render-parity identity for the other', async (t) => {
  const directory = await createDepthRunFixture(t, ({ metadata }) => {
    metadata.workload.renderParitySha256ByCell['low-overlap|0.99'] =
      metadata.workload.renderParitySha256ByCell['high-overlap|0.99'];
  });
  await assert.rejects(
    summarizeInput(directory),
    /render-parity digest differs from its layout cell/,
  );
});

test('verified depth frames preserve the configured compute schedule', async (t) => {
  const directory = await createDepthRunFixture(t, (fixture) => {
    const rows = fixture.framesCsv.split('\n').map((row) => row.split(','));
    const dispatchColumn = rows[0].indexOf('configuredComputeDispatches');
    rows[1][dispatchColumn] = '99';
    fixture.framesCsv = rows.map((row) => row.join(',')).join('\n');
  });
  await assert.rejects(
    summarizeInput(directory),
    /depth protocol audit fields do not match its planned trial/,
  );
});
