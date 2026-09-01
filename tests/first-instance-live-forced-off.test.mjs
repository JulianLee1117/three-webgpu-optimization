import assert from 'node:assert/strict';
import test from 'node:test';
import {
  PerspectiveCamera,
  WebGPUCoordinateSystem,
} from 'three/webgpu';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  runFirstInstanceLiveForcedFeatureOffGate,
} from '../src/validation/first-instance-live-forced-off.js';
import {
  createFirstInstanceAddressChallengeOracle,
} from '../src/validation/first-instance-address-challenge.js';

const PORTABLE = 'portable';

function scenarioFixture() {
  const objectCount = 4;
  const matrices = new Float32Array(objectCount * 16);
  for (let objectId = 0; objectId < objectCount; objectId += 1) {
    const base = objectId * 16;
    matrices[base] = 1;
    matrices[base + 5] = 1;
    matrices[base + 10] = 1;
    matrices[base + 15] = 1;
  }
  return {
    objectCount,
    bucketCount: 2,
    matrices,
    bounds: new Float32Array(objectCount * 4),
    objectBuckets: Uint32Array.of(0, 0, 1, 1),
    bucketBases: Uint32Array.of(0, 2),
    bucketCounts: Uint32Array.of(2, 2),
    visibleCounts: Uint32Array.of(0, 0),
    cullOrder: Uint32Array.of(0, 1, 2, 3),
    expectedVisibleIds: new Uint32Array(0),
    expectedVisibleCount: 0,
  };
}

test('forced-off helper selects only portable, validates, runs hooks, and disposes', async () => {
  const scenario = scenarioFixture();
  const sourceGeometries = createIndexedGeometryFixtures(scenario.bucketCount, 'low');
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.z = 10;
  camera.updateProjectionMatrix();
  const events = [];
  const deletedAttributes = [];
  const renderer = {
    coordinateSystem: WebGPUCoordinateSystem,
    hasFeature(name) {
      return name === 'indirect-first-instance';
    },
    compute(nodes) {
      events.push(`compute:${nodes.length}`);
    },
    async getArrayBufferAsync(attribute) {
      return attribute.array.slice().buffer;
    },
    _attributes: {
      delete(attribute) {
        deletedAttributes.push(attribute);
      },
    },
  };
  try {
    const evidence = await runFirstInstanceLiveForcedFeatureOffGate({
      scenario,
      sourceGeometries,
      renderer,
      camera,
      addressChallenge({ strategy, resources, snapshot, correctness }) {
        events.push('address');
        assert.equal(strategy.id, 'fixed-slice');
        assert.equal(resources.addressMode, 'bucket-base');
        assert.equal(resources.commandByteOffset, 0);
        assert.ok(snapshot.visibleIds instanceof Uint32Array);
        assert.ok(snapshot.commands instanceof Uint32Array);
        assert.ok(correctness.actualCounts instanceof Uint32Array);
        assert.equal(resources.visibleIdsAttribute, strategy.sharedResources.attributes.visibleIds);
        assert.equal(resources.indirectAttribute, strategy.laneState.indirectAttribute);
        const oracle = createFirstInstanceAddressChallengeOracle({
          scenario,
          sourceGeometries,
          firstIndexes: resources.firstIndexes,
          visibleIdsAttribute: resources.visibleIdsAttribute,
          laneIds: [PORTABLE],
          laneDefinitions: {
            [PORTABLE]: {
              addressMode: resources.addressMode,
              productionGeometry: resources.geometry,
              indirectAttribute: resources.indirectAttribute,
              indirectOffsets: resources.commandOffsets,
              commandBuffer: strategy.laneState.commandBufferCommitment(),
              commandSegment: { index: 0, recordBase: 0, byteBase: 0 },
            },
          },
          namePrefix: 'forced-off-portable-address-challenge',
        });
        assert.deepEqual(Object.keys(oracle.geometries), [PORTABLE]);
        assert.deepEqual(Object.keys(oracle.materials), [PORTABLE]);
        assert.equal(oracle.geometries[PORTABLE].getAttribute('bucketBase') !== undefined, true);
        assert.equal(oracle.geometryEvidence.portableOnly, true);
        oracle.dispose();
        oracle.geometries[PORTABLE].setIndirect(null);
        oracle.geometries[PORTABLE].dispose();
        oracle.materials[PORTABLE].dispose();
        return { kind: 'test-address', pass: true };
      },
      captureOutput({ strategy }) {
        events.push('output');
        assert.equal(strategy.laneState.lane, 'portable');
        return { kind: 'test-output', pass: true };
      },
    });
    assert.equal(evidence.pass, true, JSON.stringify(evidence, null, 2));
    assert.equal(
      evidence.kind,
      'first-instance-live-forced-feature-off-deployment-gate',
    );
    assert.equal(evidence.actualFeatureAvailable, true);
    assert.equal(evidence.forcedFeatureAvailable, false);
    assert.deepEqual(evidence.selection, {
      lane: 'portable',
      strategyId: 'fixed-slice',
      featureAvailable: false,
    });
    assert.equal(evidence.construction.featureLaneConstructed, false);
    assert.equal(evidence.construction.portableBucketBasePresent, true);
    assert.equal(evidence.commands.fifthCommandWordsAllZero, true);
    assert.equal(evidence.correctness.pass, true);
    assert.equal(evidence.disposal.pass, true);
    assert.deepEqual(events, ['compute:2', 'address', 'output']);
    assert.equal(deletedAttributes.length, 9);
  } finally {
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});

test('forced-off helper fails closed when address/output gates are not supplied', async () => {
  const scenario = scenarioFixture();
  const sourceGeometries = createIndexedGeometryFixtures(scenario.bucketCount, 'low');
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.updateProjectionMatrix();
  const renderer = {
    coordinateSystem: WebGPUCoordinateSystem,
    hasFeature: () => true,
    compute() {},
    async getArrayBufferAsync(attribute) {
      return attribute.array.slice().buffer;
    },
    _attributes: { delete() {} },
  };
  try {
    const evidence = await runFirstInstanceLiveForcedFeatureOffGate({
      scenario,
      sourceGeometries,
      renderer,
      camera,
    });
    assert.equal(evidence.pass, false);
    assert.equal(evidence.passBeforeDisposal, false);
    assert.equal(evidence.address.available, false);
    assert.equal(evidence.output.available, false);
    assert.equal(evidence.disposal.pass, true);
  } finally {
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});
