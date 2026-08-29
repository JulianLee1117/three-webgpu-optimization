import assert from 'node:assert/strict';
import test from 'node:test';
import { createIndexedIndirectCommands, INDEXED_INDIRECT_STRIDE_BYTES } from '../src/culling/indexed-command-layout.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import { validateIndexedCommands } from '../src/validation/membership.js';

test('indexed commands use native five-word records and zero firstInstance', () => {
  const geometries = createIndexedGeometryFixtures(4, 'low');
  const capacities = Uint32Array.of(8, 7, 6, 5);
  const counts = Uint32Array.of(4, 3, 2, 1);
  const { commands, offsets } = createIndexedIndirectCommands(geometries, capacities, counts);

  assert.equal(commands.length, geometries.length * 5);
  assert.deepEqual([...offsets], [0, 1, 2, 3].map((index) => index * INDEXED_INDIRECT_STRIDE_BYTES));
  assert.deepEqual([...commands.filter((_, index) => index % 5 === 4)], [0, 0, 0, 0]);
  assert.equal(validateIndexedCommands({ commands, geometries, expectedCounts: counts }).pass, true);

  geometries.forEach((geometry) => geometry.dispose());
});

test('one-bucket commands retain an indexable struct-array binding', () => {
  const geometries = createIndexedGeometryFixtures(1, 'low');
  const { commands, recordCount, offsets } = createIndexedIndirectCommands(
    geometries,
    Uint32Array.of(16),
    Uint32Array.of(7),
  );
  assert.equal(recordCount, 2);
  assert.equal(commands.length, 10);
  assert.deepEqual([...offsets], [0]);
  assert.equal(validateIndexedCommands({
    commands,
    geometries,
    expectedCounts: Uint32Array.of(7),
  }).pass, true);
  geometries.forEach((geometry) => geometry.dispose());
});
