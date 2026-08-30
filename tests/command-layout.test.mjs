import assert from 'node:assert/strict';
import test from 'node:test';
import { createIndexedIndirectCommands, INDEXED_INDIRECT_STRIDE_BYTES } from '../src/culling/indexed-command-layout.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import { createMergedIndexedBucketGeometry } from '../src/render/indexed-bucket-geometry.js';
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

function createBucketLayout(bucketCount) {
  const instanceCounts = Uint32Array.from(
    { length: bucketCount },
    (_, bucket) => ((bucket * 7) % 23) + 1,
  );
  const bucketBases = new Uint32Array(bucketCount);
  let base = 0;
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    bucketBases[bucket] = base;
    base += instanceCounts[bucket];
  }
  return { bucketBases, instanceCounts };
}

function exactAttributeBytes(attribute, componentOffset = 0, componentCount = attribute.count * attribute.itemSize) {
  return new Uint8Array(
    attribute.array.buffer,
    attribute.array.byteOffset + componentOffset * attribute.array.BYTES_PER_ELEMENT,
    componentCount * attribute.array.BYTES_PER_ELEMENT,
  );
}

function assertMergedFixedSliceLayout(bucketCount) {
  const sources = createIndexedGeometryFixtures(bucketCount, 'medium');
  const { bucketBases, instanceCounts } = createBucketLayout(bucketCount);
  let geometry;
  try {
    const merged = createMergedIndexedBucketGeometry(sources, bucketBases, instanceCounts);
    geometry = merged.geometry;
    const sourceAttributeNames = Object.keys(sources[0].attributes).sort();
    const expectedAttributeNames = [...sourceAttributeNames, 'bucketBase'].sort();

    assert.equal(geometry.isInstancedBufferGeometry, true);
    assert.equal(geometry.instanceCount, Math.max(...instanceCounts));
    assert.deepEqual(Object.keys(geometry.attributes).sort(), expectedAttributeNames);
    assert.ok(merged.firstIndexes instanceof Uint32Array);

    for (const source of sources) {
      assert.deepEqual(Object.keys(source.attributes).sort(), sourceAttributeNames);
      assert.equal(source.getAttribute('bucketBase'), undefined);
    }

    for (const attributeName of sourceAttributeNames) {
      const mergedAttribute = geometry.getAttribute(attributeName);
      const firstSourceAttribute = sources[0].getAttribute(attributeName);
      const expectedCount = sources.reduce(
        (total, source) => total + source.getAttribute(attributeName).count,
        0,
      );

      assert.equal(mergedAttribute.itemSize, firstSourceAttribute.itemSize);
      assert.equal(mergedAttribute.normalized, firstSourceAttribute.normalized);
      assert.equal(mergedAttribute.gpuType, firstSourceAttribute.gpuType);
      assert.equal(mergedAttribute.array.constructor, firstSourceAttribute.array.constructor);
      assert.equal(mergedAttribute.count, expectedCount);

      let componentCursor = 0;
      for (let bucket = 0; bucket < bucketCount; bucket += 1) {
        const sourceAttribute = sources[bucket].getAttribute(attributeName);
        const componentCount = sourceAttribute.count * sourceAttribute.itemSize;
        assert.equal(sourceAttribute.itemSize, mergedAttribute.itemSize);
        assert.equal(sourceAttribute.normalized, mergedAttribute.normalized);
        assert.equal(sourceAttribute.gpuType, mergedAttribute.gpuType);
        assert.equal(sourceAttribute.array.constructor, mergedAttribute.array.constructor);
        assert.deepEqual(
          exactAttributeBytes(mergedAttribute, componentCursor, componentCount),
          exactAttributeBytes(sourceAttribute),
          `${attributeName} bytes differ in bucket ${bucket} for B=${bucketCount}`,
        );
        componentCursor += componentCount;
      }
      assert.equal(componentCursor, mergedAttribute.count * mergedAttribute.itemSize);
    }

    const bucketBaseAttribute = geometry.getAttribute('bucketBase');
    assert.ok(bucketBaseAttribute.array instanceof Uint32Array);
    assert.equal(bucketBaseAttribute.itemSize, 1);
    assert.equal(bucketBaseAttribute.normalized, false);
    assert.equal(bucketBaseAttribute.count, geometry.getAttribute('position').count);

    let vertexCursor = 0;
    let indexCursor = 0;
    const expectedFirstIndexes = new Uint32Array(bucketCount);
    for (let bucket = 0; bucket < bucketCount; bucket += 1) {
      const source = sources[bucket];
      const vertexCount = source.getAttribute('position').count;
      expectedFirstIndexes[bucket] = indexCursor;

      assert.deepEqual(
        bucketBaseAttribute.array.subarray(vertexCursor, vertexCursor + vertexCount),
        new Uint32Array(vertexCount).fill(bucketBases[bucket]),
        `bucketBase span differs in bucket ${bucket} for B=${bucketCount}`,
      );
      for (let index = 0; index < source.index.count; index += 1) {
        assert.equal(
          geometry.index.getX(indexCursor + index),
          source.index.getX(index) + vertexCursor,
          `rebased index differs in bucket ${bucket}, index ${index}, for B=${bucketCount}`,
        );
      }

      vertexCursor += vertexCount;
      indexCursor += source.index.count;
    }

    assert.deepEqual(merged.firstIndexes, expectedFirstIndexes);
    assert.equal(vertexCursor, geometry.getAttribute('position').count);
    assert.equal(indexCursor, geometry.index.count);
    if (bucketCount === 128) {
      assert.ok(
        merged.firstIndexes.some((firstIndex) => firstIndex > 0xffff),
        'B=128 must exercise cumulative firstIndex values above the uint16 range',
      );
    }

    const commandLayout = createIndexedIndirectCommands(
      sources,
      instanceCounts,
      instanceCounts,
      merged.firstIndexes,
    );
    const validation = validateIndexedCommands({
      commands: commandLayout.commands,
      geometries: sources,
      expectedCounts: instanceCounts,
      expectedFirstIndexes: expectedFirstIndexes,
    });
    assert.equal(validation.pass, true);
    assert.equal(validation.commandCount, bucketCount);
    assert.equal(
      validation.totalInstanceCount,
      instanceCounts.reduce((total, count) => total + count, 0),
    );
  } finally {
    geometry?.dispose();
    sources.forEach((source) => source.dispose());
  }
}

for (const bucketCount of [1, 4, 32, 128]) {
  test(`merged fixed-slice geometry preserves exact source layout at B=${bucketCount}`, () => {
    assertMergedFixedSliceLayout(bucketCount);
  });
}
