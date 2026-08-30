import assert from 'node:assert/strict';
import test from 'node:test';
import {
  fingerprintFixedSubsetScenario,
  fingerprintGeometryFixtures,
} from '../src/scenes/geometry-fingerprints.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import { createFixedSubsetScenario } from '../src/scenes/fixed-subsets.js';

test('geometry manifests are deterministic, unique, and sensitive to vertex changes', async () => {
  const first = createIndexedGeometryFixtures(8, 'medium');
  const second = createIndexedGeometryFixtures(8, 'medium');
  try {
    const firstManifest = await fingerprintGeometryFixtures(first, 'medium');
    const secondManifest = await fingerprintGeometryFixtures(second, 'medium');
    assert.deepEqual(secondManifest, firstManifest);
    assert.equal(new Set(firstManifest.geometries.map((record) => record.sha256)).size, 8);

    const position = second[0].getAttribute('position');
    position.setX(0, position.getX(0) + 0.001);
    const changed = await fingerprintGeometryFixtures(second, 'medium');
    assert.notEqual(changed.geometries[0].sha256, firstManifest.geometries[0].sha256);
    assert.notEqual(changed.sha256, firstManifest.sha256);
  } finally {
    first.forEach((geometry) => geometry.dispose());
    second.forEach((geometry) => geometry.dispose());
  }
});

test('scenario manifests bind the seed, transforms, visibility, and bucket layout', async () => {
  const geometries = createIndexedGeometryFixtures(4, 'low');
  try {
    const options = {
      objectCount: 4_096,
      bucketCount: 4,
      visibilityFraction: 0.2,
      geometrySpheres: geometries.map((geometry) => geometry.boundingSphere.clone()),
      seed: 0xb1ad_2026,
    };
    const first = createFixedSubsetScenario(options);
    const second = createFixedSubsetScenario(options);
    const firstManifest = await fingerprintFixedSubsetScenario(first, options.seed);
    const secondManifest = await fingerprintFixedSubsetScenario(second, options.seed);
    assert.deepEqual(secondManifest, firstManifest);

    second.matrices[0] += 0.001;
    const changed = await fingerprintFixedSubsetScenario(second, options.seed);
    assert.notEqual(changed.sha256, firstManifest.sha256);
    assert.notEqual(changed.arrays.matrices.sha256, firstManifest.arrays.matrices.sha256);
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
});
