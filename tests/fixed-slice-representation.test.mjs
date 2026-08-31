import assert from 'node:assert/strict';
import test from 'node:test';
import { INDEXED_INDIRECT_STRIDE_BYTES } from '../src/culling/indexed-command-layout.js';
import { createFixedSubsetScenario } from '../src/scenes/fixed-subsets.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  buildFixedSlicePerBucketStrategy,
  buildFixedSliceStrategy,
} from '../src/strategies/fixed-slice.js';
import { disposeStrategyResources } from '../src/strategies/resources.js';

function exactBytes(attribute) {
  return new Uint8Array(
    attribute.array.buffer,
    attribute.array.byteOffset,
    attribute.array.byteLength,
  );
}

function disposeStrategy(strategy) {
  disposeStrategyResources({ _attributes: { delete() {} } }, strategy);
}

for (const bucketCount of [1, 4, 32, 128]) {
  test(`fixed-slice control holds compute schedule, commands, and geometry payload constant at B=${bucketCount}`, () => {
    const sourceGeometries = createIndexedGeometryFixtures(bucketCount, 'medium');
    const objectCount = Math.max(512, bucketCount * 4);
    const scenario = createFixedSubsetScenario({
      objectCount,
      bucketCount,
      visibilityFraction: 0.2,
      geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
      seed: 0xb1ad_2026,
    });
    let merged;
    let perBucket;
    try {
      merged = buildFixedSliceStrategy({ scenario, sourceGeometries });
      perBucket = buildFixedSlicePerBucketStrategy({ scenario, sourceGeometries });

      assert.equal(merged.configuredRenderObjects, 1);
      assert.equal(perBucket.configuredRenderObjects, bucketCount);
      for (const strategy of [merged, perBucket]) {
        assert.equal(strategy.usesCompute, true);
        assert.equal(strategy.configuredDrawCommands, bucketCount);
        assert.equal(strategy.configuredComputeDispatches, 2);
        assert.equal(strategy.configuredComputeSubmissions, 1);
        assert.equal(strategy.configuredSubmittedInstances, null);
        assert.equal(strategy.computeNodes.length, 2);
        assert.equal(strategy.geometries.length, 1);
        assert.equal(strategy.materials.length, 1);
        assert.equal(strategy.root.static, true);
        assert.equal(strategy.root.matrixAutoUpdate, false);
        assert.equal(strategy.root.matrixWorldAutoUpdate, false);
        assert.ok(strategy.root.children.every((mesh) => (
          mesh.frustumCulled === false
          && mesh.matrixAutoUpdate === false
          && mesh.matrixWorldAutoUpdate === false
        )));
      }

      const mergedGeometry = merged.geometries[0];
      const perBucketGeometry = perBucket.geometries[0];
      assert.equal(merged.root.children.length, 1);
      assert.equal(perBucket.root.children.length, bucketCount);
      assert.equal(new Set(perBucket.root.children).size, bucketCount);
      assert.deepEqual(
        new Set(perBucket.root.children.map((mesh) => mesh.geometry)),
        new Set([perBucketGeometry]),
      );
      assert.deepEqual(
        new Set(perBucket.root.children.map((mesh) => mesh.material)),
        new Set([perBucket.materials[0]]),
      );
      assert.equal(perBucketGeometry.instanceCount, mergedGeometry.instanceCount);
      assert.equal(perBucketGeometry.instanceCount, Math.ceil(objectCount / bucketCount));
      assert.deepEqual(Object.keys(perBucketGeometry.attributes), Object.keys(mergedGeometry.attributes));
      for (const name of Object.keys(mergedGeometry.attributes)) {
        assert.deepEqual(
          exactBytes(perBucketGeometry.getAttribute(name)),
          exactBytes(mergedGeometry.getAttribute(name)),
          `${name} differs at B=${bucketCount}`,
        );
      }
      assert.deepEqual(exactBytes(perBucketGeometry.index), exactBytes(mergedGeometry.index));
      assert.deepEqual(
        exactBytes(perBucketGeometry.indirect),
        exactBytes(mergedGeometry.indirect),
      );
      assert.equal(perBucket.storageAttributes.length, merged.storageAttributes.length);
      for (let index = 0; index < merged.storageAttributes.length; index += 1) {
        const mergedAttribute = merged.storageAttributes[index];
        const perBucketAttribute = perBucket.storageAttributes[index];
        assert.equal(perBucketAttribute.constructor, mergedAttribute.constructor);
        assert.equal(perBucketAttribute.itemSize, mergedAttribute.itemSize);
        assert.equal(perBucketAttribute.count, mergedAttribute.count);
        assert.deepEqual(exactBytes(perBucketAttribute), exactBytes(mergedAttribute));
      }
      assert.equal(perBucket.materials[0].type, merged.materials[0].type);
      assert.equal(perBucket.materials[0].color.getHex(), merged.materials[0].color.getHex());
      assert.equal(perBucket.materials[0].roughness, merged.materials[0].roughness);
      assert.equal(perBucket.materials[0].metalness, merged.materials[0].metalness);
      assert.deepEqual(
        mergedGeometry.indirectOffset,
        Array.from({ length: bucketCount }, (_, bucket) => (
          bucket * INDEXED_INDIRECT_STRIDE_BYTES
        )),
      );

      const before = perBucket.diagnostics();
      assert.deepEqual(before, {
        kind: 'shared-merged-geometry-per-bucket-render-objects',
        bundleRecordCallbackCount: 0,
        geometryIdentityCount: 1,
        materialIdentityCount: 1,
        meshCount: bucketCount,
        geometryInstanceCount: Math.ceil(objectCount / bucketCount),
      });
      for (let bucket = 0; bucket < bucketCount; bucket += 1) {
        perBucket.root.children[bucket].onBeforeRender({ _currentRenderBundle: {} });
        assert.equal(
          perBucketGeometry.indirectOffset,
          bucket * INDEXED_INDIRECT_STRIDE_BYTES,
        );
      }
      assert.equal(perBucket.diagnostics().bundleRecordCallbackCount, bucketCount);
      perBucket.root.children[0].onBeforeRender({ _currentRenderBundle: null });
      assert.equal(perBucket.diagnostics().bundleRecordCallbackCount, bucketCount);
    } finally {
      if (merged) disposeStrategy(merged);
      if (perBucket) disposeStrategy(perBucket);
      sourceGeometries.forEach((geometry) => geometry.dispose());
    }
  });
}
