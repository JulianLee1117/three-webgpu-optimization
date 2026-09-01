import assert from 'node:assert/strict';
import test from 'node:test';
import {
  IndirectStorageBufferAttribute,
  StorageBufferAttribute,
} from 'three/webgpu';
import {
  createFirstInstanceAddressChallengeOracle,
  createFirstInstanceAddressChallengeExpected,
  createFragmentAddressChallengeGeometry,
} from '../src/validation/first-instance-address-challenge.js';
import { STORAGE_TRANSFORM_ADDRESS_MODES } from '../src/materials/storage-transform.js';

const scenario = Object.freeze({
  objectCount: 5,
  bucketCount: 2,
  bucketBases: Uint32Array.of(0, 2),
  bucketCounts: Uint32Array.of(2, 3),
});

test('lane-local address expectation encodes active IDs and leaves all padding zero', () => {
  const expected = createFirstInstanceAddressChallengeExpected({
    scenario,
    visibleIds: Uint32Array.of(4, 99, 2, 0, 77),
    activeCounts: Uint32Array.of(1, 2),
  });
  assert.equal(expected.activeAddressCount, 3);
  assert.equal(expected.paddingAddressCount, 2);
  assert.equal(expected.targetPaddingPixelCount, 251);
  assert.equal(expected.outOfRangeIds, 0);
  assert.deepEqual([...expected.bytes.slice(0, 20)], [
    5, 0, 0, 255,
    0, 0, 0, 0,
    3, 0, 0, 255,
    1, 0, 0, 255,
    0, 0, 0, 0,
  ]);
  assert.equal(expected.bytes.slice(20).every((value) => value === 0), true);
});

test('lane-local expectation rejects command counts beyond fixed capacity', () => {
  assert.throws(
    () => createFirstInstanceAddressChallengeExpected({
      scenario,
      visibleIds: new Uint32Array(5),
      activeCounts: Uint32Array.of(3, 0),
    }),
    /exceeds its capacity/,
  );
});

test('lane-local expectation records an out-of-range active survivor as a hard failure input', () => {
  const expected = createFirstInstanceAddressChallengeExpected({
    scenario,
    visibleIds: Uint32Array.of(5, 0, 0, 0, 0),
    activeCounts: Uint32Array.of(1, 0),
  });
  assert.equal(expected.outOfRangeIds, 1);
  assert.deepEqual([...expected.bytes.slice(0, 4)], [0, 0, 0, 0]);
});

function featureOnlyFixture() {
  const sourceGeometries = [
    { index: { count: 3 } },
    { index: { count: 6 } },
  ];
  const firstIndexes = Uint32Array.of(0, 3);
  const visibleIdsAttribute = new StorageBufferAttribute(new Uint32Array(5), 1);
  const commands = Uint32Array.of(
    3, 0, 0, 0, 0,
    6, 0, 3, 0, 2,
  );
  const indirectAttribute = new IndirectStorageBufferAttribute(commands, 5);
  const indirectOffsets = Uint32Array.of(0, 20);
  const productionGeometry = createFragmentAddressChallengeGeometry({
    sourceGeometries,
    bucketCounts: scenario.bucketCounts,
    firstIndexes,
    addressMode: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
    name: 'feature-production-geometry-fixture',
  });
  productionGeometry.setIndirect(indirectAttribute, Array.from(indirectOffsets));
  const feature = {
    addressMode: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
    productionGeometry,
    visibleIdsAttribute,
    indirectAttribute,
    indirectOffsets,
    commandBuffer: {
      lane: 'feature',
      attributeId: indirectAttribute.id,
      byteOffset: 0,
      byteLength: commands.byteLength,
      recordCount: scenario.bucketCount,
      drawCommandCount: scenario.bucketCount,
      firstOffset: indirectOffsets[0],
      allOffsets: Array.from(indirectOffsets),
    },
    commandSegment: { index: 0, recordBase: 0, byteBase: 0 },
  };
  return {
    sourceGeometries,
    firstIndexes,
    visibleIdsAttribute,
    indirectAttribute,
    productionGeometry,
    feature,
  };
}

function disposeFeatureOnlyFixture(fixture, oracle = null) {
  oracle?.dispose();
  oracle?.geometries.feature.dispose();
  oracle?.materials.feature.dispose();
  fixture.productionGeometry.dispose();
  fixture.indirectAttribute.dispose();
  fixture.visibleIdsAttribute.dispose();
}

test('feature-only oracle constructs only feature geometry/material and binds production resources', () => {
  const fixture = featureOnlyFixture();
  let oracle = null;
  try {
    oracle = createFirstInstanceAddressChallengeOracle({
      scenario,
      sourceGeometries: fixture.sourceGeometries,
      firstIndexes: fixture.firstIndexes,
      visibleIdsAttribute: fixture.visibleIdsAttribute,
      laneIds: ['feature'],
      laneDefinitions: { feature: fixture.feature },
      namePrefix: 'feature-only-address-challenge',
    });
    assert.deepEqual(Object.keys(oracle.geometries), ['feature']);
    assert.deepEqual(Object.keys(oracle.materials), ['feature']);
    assert.equal(oracle.geometries.portable, undefined);
    assert.equal(oracle.materials.portable, undefined);
    assert.equal(oracle.geometries.feature.getAttribute('bucketBase'), undefined);
    assert.equal(fixture.productionGeometry.getAttribute('bucketBase'), undefined);
    assert.equal(oracle.geometries.feature.indirect, fixture.indirectAttribute);
    assert.equal(oracle.geometryEvidence.pass, true);
    assert.equal(oracle.geometryEvidence.featureOnly, true);
    assert.equal(oracle.geometryEvidence.bucketBaseAttributeAbsent, true);
    assert.equal(oracle.geometryEvidence.productionBucketBaseAttributeAbsent, true);
    assert.equal(oracle.geometryEvidence.productionVisibleIdsIdentityExact, true);
    assert.equal(oracle.geometryEvidence.productionCommandResourceExact, true);
    assert.equal(oracle.geometryEvidence.indirectIdentityExact, true);
  } finally {
    disposeFeatureOnlyFixture(fixture, oracle);
  }
});

test('feature-only oracle rejects lane mislabeling and any portable definition/reference', () => {
  const fixture = featureOnlyFixture();
  const foreignVisibleIdsAttribute = new StorageBufferAttribute(5, 1);
  try {
    assert.throws(
      () => createFirstInstanceAddressChallengeOracle({
        scenario,
        sourceGeometries: fixture.sourceGeometries,
        firstIndexes: fixture.firstIndexes,
        visibleIdsAttribute: fixture.visibleIdsAttribute,
        laneIds: ['feature'],
        laneDefinitions: {
          feature: {
            ...fixture.feature,
            addressMode: STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
          },
        },
      }),
      /mislabeled/,
    );
    assert.throws(
      () => createFirstInstanceAddressChallengeOracle({
        scenario,
        sourceGeometries: fixture.sourceGeometries,
        firstIndexes: fixture.firstIndexes,
        visibleIdsAttribute: fixture.visibleIdsAttribute,
        laneIds: ['feature'],
        laneDefinitions: {
          feature: fixture.feature,
          portable: fixture.feature,
        },
      }),
      /exactly the selected lanes/,
    );
    assert.throws(
      () => createFirstInstanceAddressChallengeOracle({
        scenario,
        sourceGeometries: fixture.sourceGeometries,
        firstIndexes: fixture.firstIndexes,
        visibleIdsAttribute: fixture.visibleIdsAttribute,
        laneIds: ['feature'],
        laneDefinitions: {
          feature: { ...fixture.feature, visibleIdsAttribute: foreignVisibleIdsAttribute },
        },
      }),
      /production visibleIds attribute/,
    );
    assert.throws(
      () => createFirstInstanceAddressChallengeOracle({
        scenario,
        sourceGeometries: fixture.sourceGeometries,
        firstIndexes: fixture.firstIndexes,
        visibleIdsAttribute: fixture.visibleIdsAttribute,
        laneIds: ['feature'],
        laneDefinitions: {
          feature: {
            ...fixture.feature,
            commandBuffer: {
              ...fixture.feature.commandBuffer,
              attributeId: fixture.indirectAttribute.id + 1,
            },
          },
        },
      }),
      /geometry failed its exact construction audit/,
    );
  } finally {
    foreignVisibleIdsAttribute.dispose();
    disposeFeatureOnlyFixture(fixture);
  }
});
