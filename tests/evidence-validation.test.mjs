import assert from 'node:assert/strict';
import test from 'node:test';
import {
  validateExactValidation,
  validateGeometryFixtureManifest,
  validateScenarioManifest,
  validateTrialRows,
} from '../scripts/evidence-validation.mjs';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  fingerprintFixedSubsetScenario,
  fingerprintGeometryFixtures,
} from '../src/scenes/geometry-fingerprints.js';
import { createFixedSubsetScenario } from '../src/scenes/fixed-subsets.js';
import { createMembershipDigestEvidence } from '../src/validation/membership-digests.js';

const SEED = 0xb1ad_2026;

test('malformed browser evidence is rejected without crashing the Node gate', () => {
  assert.match(
    validateGeometryFixtureManifest(null, { bucketCount: 4 }).join('; '),
    /not an object/,
  );
  assert.match(
    validateScenarioManifest(null, {
      objectCount: 4_096,
      bucketCount: 4,
      visibilityFraction: 0.2,
      seed: SEED,
    }).join('; '),
    /not an object/,
  );
  assert.match(
    validateExactValidation(null, {
      modeId: 'fixed-slice',
      bucketCount: 4,
      expectedVisibleCount: 819,
      expectedVisibleIdsCanonicalSha256: '0'.repeat(64),
      geometryManifest: null,
    }).rejectionReasons.join('; '),
    /not an object/,
  );
});

async function fixtureEvidence({ bucketCount = 4, objectCount = 4_096, visibilityFraction = 0.2 } = {}) {
  const geometries = createIndexedGeometryFixtures(bucketCount, 'medium');
  const scenario = createFixedSubsetScenario({
    objectCount,
    bucketCount,
    visibilityFraction,
    geometrySpheres: geometries.map((geometry) => geometry.boundingSphere.clone()),
    seed: SEED,
  });
  const [geometryManifest, scenarioManifest] = await Promise.all([
    fingerprintGeometryFixtures(geometries, 'medium'),
    fingerprintFixedSubsetScenario(scenario, SEED),
  ]);
  return { geometries, scenario, geometryManifest, scenarioManifest };
}

test('Node validates and recomputes nested geometry and scenario manifests', async () => {
  const evidence = await fixtureEvidence();
  try {
    assert.deepEqual(validateGeometryFixtureManifest(evidence.geometryManifest, {
      bucketCount: 4,
      tier: 'medium',
    }), []);
    assert.deepEqual(validateScenarioManifest(evidence.scenarioManifest, {
      objectCount: 4_096,
      bucketCount: 4,
      visibilityFraction: 0.2,
      seed: SEED,
    }), []);

    const tamperedGeometry = structuredClone(evidence.geometryManifest);
    tamperedGeometry.geometries[0].index.count += 3;
    assert.match(
      validateGeometryFixtureManifest(tamperedGeometry, { bucketCount: 4 }) .join('; '),
      /does not match its record|nested records/,
    );

    const tamperedScenario = structuredClone(evidence.scenarioManifest);
    tamperedScenario.arrays.matrices.length -= 16;
    assert.match(
      validateScenarioManifest(tamperedScenario, {
        objectCount: 4_096,
        bucketCount: 4,
        visibilityFraction: 0.2,
        seed: SEED,
      }).join('; '),
      /matrices\.length|nested records/,
    );
    const disconnectedMembership = structuredClone(evidence.scenarioManifest);
    disconnectedMembership.expectedVisibleIdsCanonicalSha256 = '0'.repeat(64);
    assert.match(
      validateScenarioManifest(disconnectedMembership, {
        objectCount: 4_096,
        bucketCount: 4,
        visibilityFraction: 0.2,
        seed: SEED,
      }).join('; '),
      /canonical\/array digest|nested records/,
    );
  } finally {
    evidence.geometries.forEach((geometry) => geometry.dispose());
  }
});

test('compute validation requires survivor digests bound to the scenario manifest', async () => {
  const evidence = await fixtureEvidence();
  try {
    const { scenario, geometryManifest, scenarioManifest } = evidence;
    const actualIds = new Uint32Array(scenario.objectCount);
    let expectedCursor = 0;
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const count = scenario.visibleCounts[bucket];
      actualIds.set(
        scenario.expectedVisibleIds.subarray(expectedCursor, expectedCursor + count),
        scenario.bucketBases[bucket],
      );
      expectedCursor += count;
    }
    const membershipDigests = await createMembershipDigestEvidence({
      expectedIds: scenario.expectedVisibleIds,
      actualIds,
      actualCounts: scenario.visibleCounts,
      objectBuckets: scenario.objectBuckets,
      bucketBases: scenario.bucketBases,
      capacities: scenario.bucketCounts,
    });
    let firstIndex = 0;
    const records = geometryManifest.geometries.map((geometry, bucket) => {
      const command = {
        indexCount: geometry.index.count,
        instanceCount: scenario.visibleCounts[bucket],
        firstIndex,
        baseVertex: 0,
        firstInstance: 0,
      };
      firstIndex += geometry.index.count;
      return { bucket, actual: { ...command }, expected: { ...command } };
    });
    const validation = {
      pass: true,
      kind: 'fixed-slice-exact-membership',
      membership: {
        pass: true,
        expectedCount: scenario.expectedVisibleCount,
        listedCount: scenario.expectedVisibleCount,
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
        commandCount: scenario.bucketCount,
        totalInstanceCount: scenario.expectedVisibleCount,
        records,
      },
      overflow: 0,
    };
    const options = {
      modeId: 'fixed-slice',
      objectCount: scenario.objectCount,
      bucketCount: scenario.bucketCount,
      expectedVisibleCount: scenario.expectedVisibleCount,
      expectedVisibleIdsCanonicalSha256: scenarioManifest.expectedVisibleIdsCanonicalSha256,
      geometryManifest,
    };
    assert.deepEqual(validateExactValidation(validation, options).rejectionReasons, []);
    const perBucketValidation = structuredClone(validation);
    perBucketValidation.kind = 'fixed-slice-per-bucket-exact-membership';
    perBucketValidation.representation = {
      kind: 'shared-merged-geometry-per-bucket-render-objects',
      bundleRecordCallbackCount: scenario.bucketCount,
      geometryIdentityCount: 1,
      materialIdentityCount: 1,
      meshCount: scenario.bucketCount,
      geometryInstanceCount: Math.ceil(scenario.objectCount / scenario.bucketCount),
    };
    const perBucketOptions = { ...options, modeId: 'fixed-slice-per-bucket' };
    assert.deepEqual(
      validateExactValidation(perBucketValidation, perBucketOptions).rejectionReasons,
      [],
    );
    const badRepresentation = structuredClone(perBucketValidation);
    badRepresentation.representation.geometryIdentityCount = scenario.bucketCount;
    assert.match(
      validateExactValidation(badRepresentation, perBucketOptions).rejectionReasons.join('; '),
      /geometry identity count/,
    );
    const badFirstIndex = structuredClone(perBucketValidation);
    badFirstIndex.commandValidation.records[1].actual.firstIndex = 0;
    assert.match(
      validateExactValidation(badFirstIndex, perBucketOptions).rejectionReasons.join('; '),
      /actual\.firstIndex/,
    );
    const badOverflow = structuredClone(perBucketValidation);
    badOverflow.overflow = 1;
    assert.match(
      validateExactValidation(badOverflow, perBucketOptions).rejectionReasons.join('; '),
      /fixed-slice-per-bucket overflow/,
    );
    const tampered = structuredClone(validation);
    tampered.membershipDigests.actual.sha256 = '0'.repeat(64);
    assert.match(
      validateExactValidation(tampered, options).rejectionReasons.join('; '),
      /membership digest actual\.sha256/,
    );
  } finally {
    evidence.geometries.forEach((geometry) => geometry.dispose());
  }
});

test('row audit validation rejects omitted protocol and configured-work fields', () => {
  const spec = {
    runId: 'run',
    trialId: 'trial',
    planIndex: 0,
    repetitionIndex: 0,
    modeId: 'draw-all',
    modeOrder: ['draw-all', 'three-blocks-coalesced', 'fixed-slice'],
    modeOrderPosition: 0,
    visibilityFraction: 0.2,
    visibilityOrder: [0.2, 0.8, 0.99],
    visibilityOrderPosition: 0,
    objectCount: 4_096,
    bucketCount: 4,
  };
  const row = {
    schemaVersion: 2,
    runId: 'run',
    trialId: 'trial',
    planIndex: 0,
    repetitionIndex: 0,
    modeOrderPosition: 0,
    visibilityOrderPosition: 0,
    plannedModeOrder: spec.modeOrder.join('|'),
    plannedVisibilityOrder: spec.visibilityOrder.join('|'),
    protocolWarmupFrames: 300,
    protocolMeasuredFrames: 1,
    modeId: 'draw-all',
    objectCount: 4_096,
    bucketCount: 4,
    targetVisibilityFraction: 0.2,
    expectedVisibleCount: 819,
    validationKind: 'draw-all-reference',
    validationPass: true,
    usesCompute: false,
    configuredDrawCommands: 4,
    configuredRenderObjects: 4,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    configuredSubmittedInstances: 4_096,
    bundleRecordCallbackCountAtTimingStart: null,
    timestampAvailable: true,
    frameIndex: 0,
    gpuFrameId: 10,
    cpuCommonUpdateMs: 0.01,
    cpuComputeSubmitMs: null,
    cpuRenderSubmitMs: 0.1,
    cpuSubmitTotalMs: 0.1,
    cpuFrameBodyMs: 0.12,
    gpuComputeMs: null,
    gpuRenderMs: 0.2,
    gpuPassTotalMs: 0.2,
  };
  const summary = {
    accepted: true,
    timestampAvailable: true,
    rowCount: 1,
    missingRenderFrames: 0,
    missingComputeFrames: 0,
    quantumNs: 32,
    classification: 'hardware-like',
  };
  const validation = { kind: 'draw-all-reference' };
  const scenario = { expectedVisibleCount: 819 };
  const protocol = { schemaVersion: 2, warmupFrames: 300, measuredFrames: 1 };
  assert.deepEqual(validateTrialRows(spec, [row], summary, validation, scenario, protocol), []);
  const omitted = structuredClone(row);
  delete omitted.configuredDrawCommands;
  assert.match(
    validateTrialRows(spec, [omitted], summary, validation, scenario, protocol).join('; '),
    /missing configuredDrawCommands/,
  );
});

test('per-bucket fixed-slice rows require the causal schedule and stable bundle recording', () => {
  const spec = {
    runId: 'run',
    trialId: 'trial',
    planIndex: 0,
    repetitionIndex: 0,
    modeId: 'fixed-slice-per-bucket',
    modeOrder: ['fixed-slice-per-bucket', 'fixed-slice'],
    modeOrderPosition: 0,
    visibilityFraction: 0.2,
    visibilityOrder: [0.2, 0.8, 0.99],
    visibilityOrderPosition: 0,
    objectCount: 4_096,
    bucketCount: 4,
  };
  const validation = {
    kind: 'fixed-slice-per-bucket-exact-membership',
    representation: {
      bundleRecordCallbackCount: 4,
    },
  };
  const row = {
    schemaVersion: 2,
    runId: 'run',
    trialId: 'trial',
    planIndex: 0,
    repetitionIndex: 0,
    modeOrderPosition: 0,
    visibilityOrderPosition: 0,
    plannedModeOrder: spec.modeOrder.join('|'),
    plannedVisibilityOrder: spec.visibilityOrder.join('|'),
    protocolWarmupFrames: 300,
    protocolMeasuredFrames: 1,
    modeId: spec.modeId,
    objectCount: spec.objectCount,
    bucketCount: spec.bucketCount,
    targetVisibilityFraction: spec.visibilityFraction,
    expectedVisibleCount: 819,
    validationKind: validation.kind,
    validationPass: true,
    usesCompute: true,
    configuredDrawCommands: 4,
    configuredRenderObjects: 4,
    configuredComputeDispatches: 2,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    bundleRecordCallbackCountAtTimingStart: 4,
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
      kind: 'fixed-slice-per-bucket-static-bundle-invariant',
      bundleRecordCallbackCountAtTimingStart: 4,
      bundleRecordCallbackCountAtTimingEnd: 4,
      geometryIdentityCount: 1,
      materialIdentityCount: 1,
      meshCount: 4,
      geometryInstanceCount: 1_024,
    },
  };
  const scenario = { expectedVisibleCount: 819 };
  const protocol = { schemaVersion: 2, warmupFrames: 300, measuredFrames: 1 };
  assert.deepEqual(validateTrialRows(spec, [row], summary, validation, scenario, protocol), []);

  for (const [field, badValue] of [
    ['configuredRenderObjects', 1],
    ['configuredComputeDispatches', 16],
    ['configuredComputeSubmissions', 4],
    ['bundleRecordCallbackCountAtTimingStart', 8],
  ]) {
    const tampered = { ...row, [field]: badValue };
    assert.match(
      validateTrialRows(spec, [tampered], summary, validation, scenario, protocol).join('; '),
      new RegExp(field),
    );
  }
  const rebuiltSummary = structuredClone(summary);
  rebuiltSummary.completionInvariant.bundleRecordCallbackCountAtTimingEnd = 8;
  assert.match(
    validateTrialRows(spec, [row], rebuiltSummary, validation, scenario, protocol).join('; '),
    /timing-end bundle-record callback count/,
  );
  assert.match(
    validateTrialRows(
      { ...spec, modeId: 'fixed-slice-typo' },
      [row],
      summary,
      validation,
      scenario,
      protocol,
    ).join('; '),
    /unsupported modeId/,
  );
  assert.match(
    validateExactValidation({}, {
      modeId: 'fixed-slice-typo',
      objectCount: 4_096,
      bucketCount: 4,
    }).rejectionReasons.join('; '),
    /unsupported modeId/,
  );
});
