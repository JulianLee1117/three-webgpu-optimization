import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { PerspectiveCamera, Vector3, WebGPUCoordinateSystem } from 'three';
import { CAMERA } from '../src/config.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  allocateBalancedCounts,
  allocateVisibleCounts,
  createFixedSubsetScenario,
  prefixBases,
} from '../src/scenes/fixed-subsets.js';
import { cpuVisibleIds } from '../src/validation/membership.js';

function positionHash(geometry) {
  const { array } = geometry.getAttribute('position');
  return createHash('sha256')
    .update(new Uint8Array(array.buffer, array.byteOffset, array.byteLength))
    .digest('hex');
}

test('balanced counts and bases preserve the total', () => {
  const counts = allocateBalancedCounts(16_384, 32);
  const bases = prefixBases(counts);
  assert.equal(counts.reduce((sum, value) => sum + value, 0), 16_384);
  assert.equal(bases[31] + counts[31], 16_384);
  assert.ok(Math.max(...counts) - Math.min(...counts) <= 1);
});

test('visible counts hit the rounded global target', () => {
  const counts = allocateBalancedCounts(65_537, 32);
  for (const fraction of [0.2, 0.8, 0.99]) {
    const visible = allocateVisibleCounts(counts, fraction);
    assert.equal(visible.reduce((sum, value) => sum + value, 0), Math.round(65_537 * fraction));
  }
});

test('128 bucket fixtures are deterministic, unique, and topology matched by family', () => {
  const first = createIndexedGeometryFixtures(128, 'medium');
  const second = createIndexedGeometryFixtures(128, 'medium');
  try {
    const firstHashes = first.map(positionHash);
    const secondHashes = second.map(positionHash);
    assert.deepEqual(firstHashes, secondHashes);
    assert.equal(new Set(firstHashes).size, 128);

    for (let family = 0; family < 4; family += 1) {
      const familyGeometries = first.filter((_, bucket) => bucket % 4 === family);
      const reference = familyGeometries[0];
      assert.ok(familyGeometries.every(
        (geometry) => geometry.index.count === reference.index.count
          && geometry.getAttribute('position').count === reference.getAttribute('position').count,
      ));
      const radii = familyGeometries.map((geometry) => geometry.boundingSphere.radius);
      assert.ok(Math.max(...radii) / Math.min(...radii) < 1.02);
    }
  } finally {
    first.forEach((geometry) => geometry.dispose());
    second.forEach((geometry) => geometry.dispose());
  }
});

test('fixed subsets retain transforms and match independent CPU frustum visibility', () => {
  const geometries = createIndexedGeometryFixtures(4, 'low');
  const geometrySpheres = geometries.map((geometry) => geometry.boundingSphere.clone());
  const common = {
    objectCount: 4096,
    bucketCount: 4,
    geometrySpheres,
    seed: 0xb1ad_2026,
  };
  const low = createFixedSubsetScenario({ ...common, visibilityFraction: 0.2 });
  const high = createFixedSubsetScenario({ ...common, visibilityFraction: 0.8 });

  const camera = new PerspectiveCamera(CAMERA.fov, CAMERA.aspect, CAMERA.near, CAMERA.far);
  camera.position.fromArray(CAMERA.position);
  camera.lookAt(new Vector3().fromArray(CAMERA.target));
  camera.updateProjectionMatrix();

  assert.deepEqual(
    [...cpuVisibleIds(low, camera, WebGPUCoordinateSystem)],
    [...low.expectedVisibleIds],
  );
  assert.deepEqual(
    [...cpuVisibleIds(high, camera, WebGPUCoordinateSystem)],
    [...high.expectedVisibleIds],
  );

  const lowIds = new Set(low.expectedVisibleIds);
  for (const id of lowIds) {
    const matrixOffset = id * 16;
    assert.deepEqual(
      [...low.matrices.slice(matrixOffset, matrixOffset + 16)],
      [...high.matrices.slice(matrixOffset, matrixOffset + 16)],
    );
  }

  geometries.forEach((geometry) => geometry.dispose());
});
