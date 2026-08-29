import assert from 'node:assert/strict';
import test from 'node:test';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import { decodeHistoricalIndirectResults } from '../src/validation/historical-indirect.js';
import { compareMembership } from '../src/validation/membership.js';

function createHistoricalCommands(geometries, counts, firstInstances) {
  const commands = new Uint32Array(geometries.length * 5);
  let firstIndex = 0;
  for (let bucket = 0; bucket < geometries.length; bucket += 1) {
    const base = bucket * 5;
    commands[base] = geometries[bucket].index.count;
    commands[base + 1] = counts[bucket];
    commands[base + 2] = firstIndex;
    commands[base + 3] = 0;
    commands[base + 4] = firstInstances[bucket];
    firstIndex += geometries[bucket].index.count;
  }
  return commands;
}

test('historical indirect ranges remap nonzero firstInstance survivors by geometry', () => {
  const geometries = createIndexedGeometryFixtures(3, 'low');
  const expectedCounts = Uint32Array.of(2, 1, 3);
  const bucketBases = Uint32Array.of(0, 2, 3);
  const capacities = Uint32Array.of(2, 1, 3);
  const commands = createHistoricalCommands(
    geometries,
    expectedCounts,
    Uint32Array.of(4, 0, 1),
  );
  const survivorIds = Uint32Array.of(2, 3, 4, 5, 0, 1);

  const decoded = decodeHistoricalIndirectResults({
    commands,
    survivorIds,
    geometries,
    expectedCounts,
    bucketBases,
    capacities,
    objectCount: 6,
  });
  assert.equal(decoded.commandValidation.pass, true);
  assert.deepEqual([...decoded.actualCounts], [2, 1, 3]);
  assert.deepEqual([...decoded.actualIds], [0, 1, 2, 3, 4, 5]);

  const membership = compareMembership({
    expectedIds: Uint32Array.of(0, 1, 2, 3, 4, 5),
    actualIds: decoded.actualIds,
    actualCounts: decoded.actualCounts,
    objectBuckets: Uint32Array.of(0, 0, 1, 2, 2, 2),
    bucketBases,
    capacities,
    objectCount: 6,
  });
  assert.equal(membership.pass, true);

  geometries.forEach((geometry) => geometry.dispose());
});

test('historical command validation rejects overlapping survivor ranges', () => {
  const geometries = createIndexedGeometryFixtures(3, 'low');
  const expectedCounts = Uint32Array.of(2, 1, 3);
  const decoded = decodeHistoricalIndirectResults({
    commands: createHistoricalCommands(
      geometries,
      expectedCounts,
      Uint32Array.of(3, 0, 1),
    ),
    survivorIds: Uint32Array.of(2, 3, 4, 0, 1, 5),
    geometries,
    expectedCounts,
    bucketBases: Uint32Array.of(0, 2, 3),
    capacities: Uint32Array.of(2, 1, 3),
    objectCount: 6,
  });

  assert.equal(decoded.commandValidation.pass, false);
  assert.ok(decoded.commandValidation.errors.some((error) => error.includes('overlaps')));
  assert.ok(decoded.commandValidation.errors.some((error) => error.includes('unclaimed')));

  geometries.forEach((geometry) => geometry.dispose());
});
