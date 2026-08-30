import { Frustum, Matrix4, Sphere, Vector3 } from 'three';
import { INDEXED_INDIRECT_STRIDE_UINTS } from '../culling/indexed-command-layout.js';

export function cpuVisibleIds(scenario, camera, coordinateSystem, reversedDepth = false) {
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  const projectionView = new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const frustum = new Frustum().setFromProjectionMatrix(projectionView, coordinateSystem, reversedDepth);
  const center = new Vector3();
  const sphere = new Sphere();
  const visible = [];

  for (let objectId = 0; objectId < scenario.objectCount; objectId += 1) {
    const offset = objectId * 4;
    center.set(scenario.bounds[offset], scenario.bounds[offset + 1], scenario.bounds[offset + 2]);
    sphere.center.copy(center);
    sphere.radius = scenario.bounds[offset + 3];
    if (frustum.intersectsSphere(sphere)) visible.push(objectId);
  }
  return Uint32Array.from(visible);
}

export function compareMembership({
  expectedIds,
  actualIds,
  actualCounts,
  objectBuckets,
  bucketBases,
  capacities,
  objectCount,
}) {
  const expected = new Uint8Array(objectCount);
  for (const id of expectedIds) expected[id] = 1;
  const seen = new Uint8Array(objectCount);
  const listed = [];
  let duplicateIds = 0;
  let outOfRangeIds = 0;
  let wrongBucketIds = 0;
  let listedHiddenIds = 0;
  let overflow = 0;

  for (let bucket = 0; bucket < capacities.length; bucket += 1) {
    const count = actualCounts[bucket];
    const capacity = capacities[bucket];
    if (count > capacity) overflow += count - capacity;
    const safeCount = Math.min(count, capacity);
    const base = bucketBases[bucket];
    for (let slot = 0; slot < safeCount; slot += 1) {
      const id = actualIds[base + slot];
      listed.push(id);
      if (id >= objectCount) {
        outOfRangeIds += 1;
        continue;
      }
      if (seen[id]) duplicateIds += 1;
      seen[id] = 1;
      if (objectBuckets[id] !== bucket) wrongBucketIds += 1;
      if (!expected[id]) listedHiddenIds += 1;
    }
  }

  let missingVisibleIds = 0;
  for (let id = 0; id < objectCount; id += 1) {
    if (expected[id] && !seen[id]) missingVisibleIds += 1;
  }

  const errors = duplicateIds + outOfRangeIds + wrongBucketIds + listedHiddenIds + missingVisibleIds + overflow;
  return {
    pass: errors === 0,
    expectedCount: expectedIds.length,
    listedCount: listed.length,
    duplicateIds,
    outOfRangeIds,
    wrongBucketIds,
    listedHiddenIds,
    missingVisibleIds,
    overflow,
    errors,
  };
}

export function validateIndexedCommands({
  commands,
  geometries,
  expectedCounts,
  expectedFirstIndexes = null,
}) {
  const signed = new Int32Array(commands.buffer, commands.byteOffset, commands.length);
  const errors = [];
  const records = [];
  let totalInstanceCount = 0;
  for (let bucket = 0; bucket < geometries.length; bucket += 1) {
    const base = bucket * INDEXED_INDIRECT_STRIDE_UINTS;
    const expectedFirstIndex = expectedFirstIndexes
      ? expectedFirstIndexes[bucket]
      : geometries[bucket].drawRange.start > 0 ? geometries[bucket].drawRange.start : 0;
    const actual = {
      indexCount: commands[base] ?? null,
      instanceCount: commands[base + 1] ?? null,
      firstIndex: commands[base + 2] ?? null,
      baseVertex: signed[base + 3] ?? null,
      firstInstance: commands[base + 4] ?? null,
    };
    const expected = {
      indexCount: geometries[bucket].index.count,
      instanceCount: expectedCounts[bucket],
      firstIndex: expectedFirstIndex,
      baseVertex: 0,
      firstInstance: 0,
    };
    records.push({ bucket, actual, expected });
    if (Number.isInteger(actual.instanceCount)) totalInstanceCount += actual.instanceCount;
    if (actual.indexCount !== expected.indexCount) errors.push(`bucket ${bucket}: indexCount`);
    if (actual.instanceCount !== expected.instanceCount) errors.push(`bucket ${bucket}: instanceCount`);
    if (actual.firstIndex !== expected.firstIndex) errors.push(`bucket ${bucket}: firstIndex`);
    if (actual.baseVertex !== expected.baseVertex) errors.push(`bucket ${bucket}: baseVertex`);
    if (actual.firstInstance !== expected.firstInstance) errors.push(`bucket ${bucket}: firstInstance`);
  }
  return {
    pass: errors.length === 0,
    errors,
    commandCount: records.length,
    totalInstanceCount,
    records,
  };
}
