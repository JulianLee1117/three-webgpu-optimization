import assert from 'node:assert/strict';
import test from 'node:test';
import { InstancedBufferAttribute } from 'three/webgpu';
import {
  FIRST_INSTANCE_COMMAND_LANES,
  FIRST_INSTANCE_UNUSED_OBJECT_ID,
  createFirstInstanceCrossoverCommitments,
  createFirstInstanceCrossoverPacking,
  getFirstInstanceLaneCommandSegment,
  getFirstInstanceLaneDrawCommands,
  validateFirstInstanceCommandSegmentOrder,
  validateFirstInstanceCrossoverPacking,
} from '../src/culling/first-instance-crossover.js';
import {
  FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS,
  FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_CROSSOVER_PATTERNS,
  FIRST_INSTANCE_CROSSOVER_WARMUP_BLOCKS,
  FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceCrossoverFrame,
} from '../src/benchmark/first-instance-crossover-schedule.js';
import { FIRST_INSTANCE_CROSSOVER_LANES } from '../src/benchmark/plan.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  createMergedIndexedBucketGeometry,
  createSharedGeometryShell,
} from '../src/render/indexed-bucket-geometry.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_COMMAND_LANES;

function scenarioFixture() {
  return {
    objectCount: 9,
    bucketCount: 3,
    bucketBases: Uint32Array.of(0, 3, 5),
    bucketCounts: Uint32Array.of(3, 2, 4),
    visibleCounts: Uint32Array.of(2, 1, 3),
    objectBuckets: Uint32Array.of(0, 0, 0, 1, 1, 2, 2, 2, 2),
    expectedVisibleIds: Uint32Array.of(0, 2, 3, 5, 7, 8),
    expectedVisibleCount: 6,
  };
}

async function withGeometries(callback, bucketCount = 3) {
  const geometries = createIndexedGeometryFixtures(bucketCount, 'low');
  try {
    return await callback(geometries);
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
}

test('frozen first-instance packing creates exact visible slices and paired commands', async () => {
  await withGeometries(async (sourceGeometries) => {
    const scenario = scenarioFixture();
    const packing = createFirstInstanceCrossoverPacking({ scenario, sourceGeometries });

    assert.equal(packing.kind, 'cpu-frozen-first-instance-crossover-packing');
    assert.deepEqual(packing.laneCommandSegmentOrder, [PORTABLE, FEATURE]);
    assert.deepEqual(
      packing.visibleIds,
      Uint32Array.of(
        0, 2, FIRST_INSTANCE_UNUSED_OBJECT_ID,
        3, FIRST_INSTANCE_UNUSED_OBJECT_ID,
        5, 7, 8, FIRST_INSTANCE_UNUSED_OBJECT_ID,
      ),
    );
    assert.equal(packing.recordsPerLane, 3);
    assert.equal(packing.commandSegmentByteLength, 60);
    assert.equal(packing.lanes[PORTABLE].commandRecordBase, 0);
    assert.equal(packing.lanes[FEATURE].commandRecordBase, 3);
    assert.deepEqual(packing.lanes[PORTABLE].offsets, Uint32Array.of(0, 20, 40));
    assert.deepEqual(packing.lanes[FEATURE].offsets, Uint32Array.of(60, 80, 100));

    const portable = getFirstInstanceLaneDrawCommands(packing, PORTABLE);
    const feature = getFirstInstanceLaneDrawCommands(packing, FEATURE);
    assert.deepEqual(
      [...portable.filter((_, index) => index % 5 === 4)],
      [0, 0, 0],
    );
    assert.deepEqual(
      [...feature.filter((_, index) => index % 5 === 4)],
      [...scenario.bucketBases],
    );
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const base = bucket * 5;
      assert.deepEqual(
        portable.subarray(base, base + 4),
        feature.subarray(base, base + 4),
      );
    }

    const validation = await validateFirstInstanceCrossoverPacking({
      packing,
      scenario,
      sourceGeometries,
      expectedLaneCommandSegmentOrder: [PORTABLE, FEATURE],
    });
    assert.equal(validation.pass, true, JSON.stringify(validation, null, 2));
    assert.equal(validation.visibleIds.pass, true);
    assert.equal(validation.padding.pass, true);
    assert.equal(validation.padding.paddingCount, 3);
    assert.equal(validation.padding.paddingSentinelCount, 3);
    assert.equal(validation.commands.pairPass, true);
    assert.equal(validation.commitments.commandCoresEqual, true);
    assert.match(validation.commitments.logicalPairSha256, /^[0-9a-f]{64}$/);
    assert.match(validation.commitments.paddingSha256, /^[0-9a-f]{64}$/);
    assert.equal(validation.commitments.pairs.length, scenario.bucketCount);
    assert.equal(validation.commitments.pairs.every((pair) => pair.coreEqual), true);
  });
});

test('command-segment counterbalancing changes only physical placement', async () => {
  await withGeometries(async (sourceGeometries) => {
    const scenario = scenarioFixture();
    const forward = createFirstInstanceCrossoverPacking({
      scenario,
      sourceGeometries,
      laneCommandSegmentOrder: [PORTABLE, FEATURE],
    });
    const reverse = createFirstInstanceCrossoverPacking({
      scenario,
      sourceGeometries,
      laneCommandSegmentOrder: [FEATURE, PORTABLE],
    });
    assert.equal(forward.lanes[PORTABLE].commandRecordBase, 0);
    assert.equal(reverse.lanes[PORTABLE].commandRecordBase, reverse.recordsPerLane);
    assert.equal(forward.lanes[FEATURE].commandRecordBase, forward.recordsPerLane);
    assert.equal(reverse.lanes[FEATURE].commandRecordBase, 0);
    assert.deepEqual(forward.visibleIds, reverse.visibleIds);
    for (const lane of FIRST_INSTANCE_COMMAND_LANES) {
      assert.deepEqual(
        getFirstInstanceLaneCommandSegment(forward, lane),
        getFirstInstanceLaneCommandSegment(reverse, lane),
      );
    }
    assert.notDeepEqual(forward.commands, reverse.commands);

    const [forwardCommitments, reverseCommitments] = await Promise.all([
      createFirstInstanceCrossoverCommitments(forward),
      createFirstInstanceCrossoverCommitments(reverse),
    ]);
    assert.notEqual(
      forwardCommitments.physicalCommandsSha256,
      reverseCommitments.physicalCommandsSha256,
    );
    assert.equal(
      forwardCommitments.logicalPairSha256,
      reverseCommitments.logicalPairSha256,
    );
    assert.equal(
      forwardCommitments.lanes[PORTABLE].commandsSha256,
      reverseCommitments.lanes[PORTABLE].commandsSha256,
    );
    assert.equal(
      forwardCommitments.lanes[FEATURE].commandsSha256,
      reverseCommitments.lanes[FEATURE].commandsSha256,
    );

    const accepted = await validateFirstInstanceCrossoverPacking({
      packing: reverse,
      scenario,
      sourceGeometries,
      expectedLaneCommandSegmentOrder: [FEATURE, PORTABLE],
    });
    assert.equal(accepted.pass, true);
    const rejected = await validateFirstInstanceCrossoverPacking({
      packing: reverse,
      scenario,
      sourceGeometries,
      expectedLaneCommandSegmentOrder: [PORTABLE, FEATURE],
    });
    assert.equal(rejected.pass, false);
    assert.equal(rejected.metadata.pass, false);
    assert.ok(rejected.metadata.errors.includes('laneCommandSegmentOrder'));
  });
});

test('one-bucket paired packing preserves padded struct-array segments', async () => {
  await withGeometries(async (sourceGeometries) => {
    const scenario = {
      objectCount: 4,
      bucketCount: 1,
      bucketBases: Uint32Array.of(0),
      bucketCounts: Uint32Array.of(4),
      visibleCounts: Uint32Array.of(3),
      objectBuckets: Uint32Array.of(0, 0, 0, 0),
      expectedVisibleIds: Uint32Array.of(0, 1, 3),
      expectedVisibleCount: 3,
    };
    const packing = createFirstInstanceCrossoverPacking({ scenario, sourceGeometries });
    assert.equal(packing.recordsPerLane, 2);
    assert.equal(packing.commands.length, 20);
    assert.equal(packing.commandSegmentByteLength, 40);
    assert.deepEqual(packing.lanes[PORTABLE].offsets, Uint32Array.of(0));
    assert.deepEqual(packing.lanes[FEATURE].offsets, Uint32Array.of(40));
    assert.deepEqual(
      getFirstInstanceLaneCommandSegment(packing, PORTABLE).subarray(5),
      new Uint32Array(5),
    );
    assert.deepEqual(
      getFirstInstanceLaneCommandSegment(packing, FEATURE).subarray(5),
      new Uint32Array(5),
    );
    assert.equal((await validateFirstInstanceCrossoverPacking({
      packing,
      scenario,
      sourceGeometries,
    })).pass, true);
  }, 1);
});

test('frozen packing validation rejects survivor, padding, and command tampering', async (t) => {
  await withGeometries(async (sourceGeometries) => {
    await t.test('active survivor', async () => {
      const scenario = scenarioFixture();
      const packing = createFirstInstanceCrossoverPacking({ scenario, sourceGeometries });
      packing.visibleIds[0] = 1;
      const validation = await validateFirstInstanceCrossoverPacking({
        packing,
        scenario,
        sourceGeometries,
      });
      assert.equal(validation.pass, false);
      assert.equal(validation.visibleIds.pass, false);
    });

    await t.test('padding sentinel', async () => {
      const scenario = scenarioFixture();
      const packing = createFirstInstanceCrossoverPacking({ scenario, sourceGeometries });
      packing.visibleIds[2] = 1;
      const validation = await validateFirstInstanceCrossoverPacking({
        packing,
        scenario,
        sourceGeometries,
      });
      assert.equal(validation.pass, false);
      assert.equal(validation.padding.pass, false);
      assert.deepEqual(validation.padding.corruptPaddingAddresses, [2]);
    });

    await t.test('feature firstInstance', async () => {
      const scenario = scenarioFixture();
      const packing = createFirstInstanceCrossoverPacking({ scenario, sourceGeometries });
      const featureBase = packing.lanes[FEATURE].commandRecordBase * 5;
      packing.commands[featureBase + 5 + 4] = 0;
      const validation = await validateFirstInstanceCrossoverPacking({
        packing,
        scenario,
        sourceGeometries,
      });
      assert.equal(validation.pass, false);
      assert.equal(validation.commands.lanes[FEATURE].pass, false);
      assert.equal(validation.commands.pairPass, false);
    });

    await t.test('paired core', async () => {
      const scenario = scenarioFixture();
      const packing = createFirstInstanceCrossoverPacking({ scenario, sourceGeometries });
      const featureBase = packing.lanes[FEATURE].commandRecordBase * 5;
      packing.commands[featureBase] += 1;
      const validation = await validateFirstInstanceCrossoverPacking({
        packing,
        scenario,
        sourceGeometries,
      });
      assert.equal(validation.pass, false);
      assert.equal(validation.commitments.commandCoresEqual, false);
      assert.equal(validation.commands.pairPass, false);
    });
  });
});

test('packing construction rejects malformed fixed-slice inputs', async () => {
  await withGeometries((sourceGeometries) => {
    const duplicate = scenarioFixture();
    duplicate.expectedVisibleIds = Uint32Array.of(0, 2, 2, 5, 7, 8);
    assert.throws(
      () => createFirstInstanceCrossoverPacking({ scenario: duplicate, sourceGeometries }),
      /strictly ascending/,
    );

    const wrongBucket = scenarioFixture();
    wrongBucket.objectBuckets[0] = 1;
    assert.throws(
      () => createFirstInstanceCrossoverPacking({ scenario: wrongBucket, sourceGeometries }),
      /does not match its contiguous bucket/,
    );

    assert.throws(
      () => createFirstInstanceCrossoverPacking({
        scenario: scenarioFixture(),
        sourceGeometries,
        laneCommandSegmentOrder: [PORTABLE, PORTABLE],
      }),
      /exact portable\/feature lane permutation/,
    );
    assert.throws(
      () => validateFirstInstanceCommandSegmentOrder('portable|feature'),
      /exact portable\/feature lane permutation/,
    );
  });
});

test('first-instance schedule is complementary, balanced, and query-pool bounded', () => {
  assert.deepEqual(FIRST_INSTANCE_COMMAND_LANES, FIRST_INSTANCE_CROSSOVER_LANES);
  assert.equal(FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE, 8);
  assert.equal(FIRST_INSTANCE_CROSSOVER_WARMUP_BLOCKS, 40);
  assert.equal(FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS, 60);
  assert.equal(FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES, 320);
  assert.equal(FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES, 480);
  assert.equal(FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES * 2, 640);
  assert.equal(FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES * 2, 960);
  assert.ok(FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES * 2 < 2_048);
  assert.deepEqual(
    FIRST_INSTANCE_CROSSOVER_PATTERNS.map((pattern) => pattern.map(
      (lane) => (lane === PORTABLE ? 'P' : 'F'),
    ).join('')),
    ['PFPFFPFP', 'FPFPPFPF'],
  );
  for (const pattern of FIRST_INSTANCE_CROSSOVER_PATTERNS) {
    assert.equal(pattern.filter((lane) => lane === PORTABLE).length, 4);
    assert.equal(pattern.filter((lane) => lane === FEATURE).length, 4);
  }
  for (let frameIndex = 0; frameIndex < 16; frameIndex += 1) {
    const forward = firstInstanceCrossoverFrame(frameIndex, 0);
    const reverse = firstInstanceCrossoverFrame(frameIndex, 1);
    assert.notEqual(forward.laneId, reverse.laneId);
  }
  const measured = Array.from(
    { length: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES },
    (_, frameIndex) => firstInstanceCrossoverFrame(frameIndex, 0),
  );
  assert.equal(measured.filter((frame) => frame.laneId === PORTABLE).length, 240);
  assert.equal(measured.filter((frame) => frame.laneId === FEATURE).length, 240);
  assert.equal(measured.at(-1).crossoverBlockIndex, 59);
  assert.throws(() => firstInstanceCrossoverFrame(-1), /nonnegative integer/);
  assert.throws(() => firstInstanceCrossoverFrame(0, 2), /zero or one/);
});

test('shared geometry shell reuses exact common payload and rejects instanced attributes', () => {
  const sources = createIndexedGeometryFixtures(4, 'medium');
  let portable;
  let shell;
  try {
    const merged = createMergedIndexedBucketGeometry(
      sources,
      Uint32Array.of(0, 4, 8, 12),
      Uint32Array.of(4, 4, 4, 4),
    );
    portable = merged.geometry;
    portable.computeBoundingBox();
    portable.computeBoundingSphere();
    shell = createSharedGeometryShell(portable, { omitAttributes: ['bucketBase'] });

    assert.notEqual(shell, portable);
    assert.equal(shell.index, portable.index);
    assert.equal(shell.getAttribute('bucketBase'), undefined);
    for (const name of Object.keys(portable.attributes).filter((name) => name !== 'bucketBase')) {
      assert.equal(shell.getAttribute(name), portable.getAttribute(name), name);
    }
    assert.equal(shell.instanceCount, portable.instanceCount);
    assert.deepEqual(shell.drawRange, portable.drawRange);
    assert.deepEqual(shell.groups, portable.groups);
    assert.deepEqual(shell.boundingBox, portable.boundingBox);
    assert.deepEqual(shell.boundingSphere, portable.boundingSphere);
    assert.notEqual(shell.boundingBox, portable.boundingBox);
    assert.notEqual(shell.boundingSphere, portable.boundingSphere);
    assert.equal(shell.indirect, null);

    assert.throws(
      () => createSharedGeometryShell(portable, { omitAttributes: ['missing'] }),
      /Cannot omit missing geometry attribute/,
    );
    assert.throws(
      () => createSharedGeometryShell(portable, { omitAttributes: ['bucketBase', 'bucketBase'] }),
      /duplicate names/,
    );
    assert.throws(
      () => createSharedGeometryShell(portable, { omitAttributes: 'bucketBase' }),
      /must be an array/,
    );

    portable.setAttribute(
      'unsafeInstance',
      new InstancedBufferAttribute(new Float32Array(portable.instanceCount), 1),
    );
    assert.throws(
      () => createSharedGeometryShell(portable, { omitAttributes: ['bucketBase'] }),
      /instanced attribute unsafeInstance/,
    );
    portable.deleteAttribute('unsafeInstance');
  } finally {
    shell?.dispose();
    portable?.dispose();
    sources.forEach((geometry) => geometry.dispose());
  }
});
