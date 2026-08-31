import assert from 'node:assert/strict';
import test from 'node:test';
import {
  DEPTH_BIN_COUNT,
  DEPTH_ORDER_FRONT_TO_BACK,
  DEPTH_ORDER_REVERSE,
  HIDDEN_DEPTH_BIN,
  createDepthBinTraversal,
  createExpectedObjectDepthBins,
  createOrderedDepthBinLayout,
  createPhysicalDepthBinSequenceCommitment,
  depthBinForViewDepth,
  validateDepthBinReadback,
  validateDepthRange,
} from '../src/culling/depth-bin-layout.js';
import { INDEXED_INDIRECT_STRIDE_BYTES } from '../src/culling/indexed-command-layout.js';
import { createFixedSubsetScenario } from '../src/scenes/fixed-subsets.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  buildDepthBinnedFrontToBackStrategy,
  buildDepthBinnedReverseStrategy,
} from '../src/strategies/depth-binned-fixed-slice.js';
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

test('depth-bin traversal and boundary classification are explicit', () => {
  assert.deepEqual(
    Array.from(createDepthBinTraversal(DEPTH_ORDER_FRONT_TO_BACK)),
    [0, 1, 2, 3, 4, 5, 6, 7],
  );
  assert.deepEqual(
    Array.from(createDepthBinTraversal(DEPTH_ORDER_REVERSE)),
    [7, 6, 5, 4, 3, 2, 1, 0],
  );
  assert.throws(() => createDepthBinTraversal('unordered'), /depth order/);

  const range = { near: 10, far: 90 };
  assert.deepEqual(validateDepthRange(range), range);
  assert.equal(depthBinForViewDepth(-100, range), 0);
  assert.equal(depthBinForViewDepth(10, range), 0);
  assert.equal(depthBinForViewDepth(19.999, range), 0);
  assert.equal(depthBinForViewDepth(20, range), 1);
  assert.equal(depthBinForViewDepth(89.999, range), 7);
  assert.equal(depthBinForViewDepth(90, range), 7);
  assert.equal(depthBinForViewDepth(1_000, range), 7);
  assert.throws(() => validateDepthRange({ near: 2, far: 2 }), /greater than/);
});

test('ordered depth-bin layout reverses blocks without changing counts or totals', () => {
  const counts = Uint32Array.from([
    1, 2, 0, 3, 0, 0, 1, 2,
    0, 1, 1, 0, 2, 1, 0, 1,
  ]);
  const front = createOrderedDepthBinLayout(counts, DEPTH_ORDER_FRONT_TO_BACK);
  const reverse = createOrderedDepthBinLayout(counts, DEPTH_ORDER_REVERSE);

  assert.deepEqual(Array.from(front.totals), [9, 6]);
  assert.deepEqual(Array.from(reverse.totals), [9, 6]);
  assert.deepEqual(Array.from(front.starts.subarray(0, 8)), [0, 1, 3, 3, 6, 6, 6, 7]);
  assert.deepEqual(Array.from(reverse.starts.subarray(0, 8)), [8, 6, 6, 3, 3, 3, 2, 0]);
  assert.deepEqual(Array.from(front.traversal), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(Array.from(reverse.traversal), [7, 6, 5, 4, 3, 2, 1, 0]);
});

test('physical-bin sequence commitments normalize traversal and retain within-bin order', async () => {
  const front = syntheticDepthReadback(DEPTH_ORDER_FRONT_TO_BACK);
  const reverse = syntheticDepthReadback(DEPTH_ORDER_REVERSE);
  const inputs = (readback) => ({
    actualIds: readback.actualIds,
    binCounts: readback.binCounts,
    binStarts: readback.binStarts,
    bucketBases: readback.bucketBases,
    bucketCapacities: readback.bucketCapacities,
  });
  const [frontCommitment, reverseCommitment] = await Promise.all([
    createPhysicalDepthBinSequenceCommitment(inputs(front)),
    createPhysicalDepthBinSequenceCommitment(inputs(reverse)),
  ]);
  assert.deepEqual(frontCommitment, reverseCommitment);
  assert.equal(frontCommitment.survivorCount, 6);
  assert.equal(frontCommitment.recordCount, 16);
  assert.match(frontCommitment.sha256, /^[0-9a-f]{64}$/);

  const reordered = structuredClone(reverse);
  const start = reordered.bucketBases[1]
    + reordered.binStarts[DEPTH_BIN_COUNT + 1];
  [reordered.actualIds[start], reordered.actualIds[start + 1]] = [
    reordered.actualIds[start + 1],
    reordered.actualIds[start],
  ];
  const reorderedCommitment = await createPhysicalDepthBinSequenceCommitment(inputs(reordered));
  assert.notEqual(reorderedCommitment.sha256, frontCommitment.sha256);

  const invalid = inputs(front);
  invalid.bucketCapacities = Uint32Array.from([1, 4]);
  await assert.rejects(
    createPhysicalDepthBinSequenceCommitment(invalid),
    /sequence exceeds readback capacity/,
  );
});

test('CPU expected bins use sphere-nearest view depth and hidden sentinel', () => {
  const bounds = Float32Array.from([
    0, 0, -20, 2,
    0, 0, -50, 5,
    0, 0, -80, 1,
  ]);
  const bins = createExpectedObjectDepthBins({
    bounds,
    objectCount: 3,
    expectedIds: Uint32Array.from([0, 2]),
    viewMatrixElements: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ],
    depthRange: { near: 10, far: 90 },
  });

  // Nearest keys are 18 and 79, not the sphere-center depths 20 and 80.
  assert.deepEqual(Array.from(bins), [0, HIDDEN_DEPTH_BIN, 6]);
});

function syntheticDepthReadback(order) {
  const objectBins = Uint32Array.from([0, 2, 8, 7, 1, 6, 8, 1]);
  const objectBuckets = Uint32Array.from([0, 0, 0, 0, 1, 1, 1, 1]);
  const binCounts = new Uint32Array(2 * DEPTH_BIN_COUNT);
  for (let objectId = 0; objectId < objectBins.length; objectId += 1) {
    if (objectBins[objectId] < DEPTH_BIN_COUNT) {
      binCounts[objectBuckets[objectId] * DEPTH_BIN_COUNT + objectBins[objectId]] += 1;
    }
  }
  const layout = createOrderedDepthBinLayout(binCounts, order);
  const actualIds = new Uint32Array(8);
  const perBin = new Map([
    ['0:0', [0]],
    ['0:2', [1]],
    ['0:7', [3]],
    ['1:1', [4, 7]],
    ['1:6', [5]],
  ]);
  for (let bucket = 0; bucket < 2; bucket += 1) {
    for (const bin of layout.traversal) {
      const ids = perBin.get(`${bucket}:${bin}`) ?? [];
      actualIds.set(ids, bucket * 4 + layout.starts[bucket * DEPTH_BIN_COUNT + bin]);
    }
  }
  return {
    actualIds,
    objectBins,
    binCounts,
    binStarts: layout.starts,
    commandCounts: layout.totals,
    expectedObjectBins: objectBins.slice(),
    objectBuckets,
    bucketBases: Uint32Array.from([0, 4]),
    bucketCapacities: Uint32Array.from([4, 4]),
    order,
  };
}

for (const order of [DEPTH_ORDER_FRONT_TO_BACK, DEPTH_ORDER_REVERSE]) {
  test(`depth-bin readback validates exact ${order} blocks while ignoring within-bin order`, () => {
    const readback = syntheticDepthReadback(order);
    const result = validateDepthBinReadback(readback);
    assert.equal(result.pass, true, result.errors.join('\n'));
    assert.equal(result.binCount, 8);
    assert.equal(result.order, order);

    const changedWithinBin = structuredClone(readback);
    const bucketOneBinOneStart = 4 + changedWithinBin.binStarts[DEPTH_BIN_COUNT + 1];
    [
      changedWithinBin.actualIds[bucketOneBinOneStart],
      changedWithinBin.actualIds[bucketOneBinOneStart + 1],
    ] = [
      changedWithinBin.actualIds[bucketOneBinOneStart + 1],
      changedWithinBin.actualIds[bucketOneBinOneStart],
    ];
    assert.equal(validateDepthBinReadback(changedWithinBin).pass, true);

    const wrongBlock = structuredClone(readback);
    [wrongBlock.actualIds[0], wrongBlock.actualIds[2]] = [
      wrongBlock.actualIds[2],
      wrongBlock.actualIds[0],
    ];
    const rejected = validateDepthBinReadback(wrongBlock);
    assert.equal(rejected.pass, false);
    assert.ok(rejected.errors.some((error) => error.includes('belongs to bin')));
  });
}

for (const bucketCount of [1, 4, 32]) {
  test(`depth-order pair preserves one-object four-dispatch workload at B=${bucketCount}`, () => {
    const sourceGeometries = createIndexedGeometryFixtures(bucketCount, 'medium');
    const objectCount = Math.max(512, bucketCount * 4);
    const scenario = createFixedSubsetScenario({
      objectCount,
      bucketCount,
      visibilityFraction: 0.8,
      geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
      seed: 0xb1ad_2026,
    });
    scenario.depthBinRange = { near: 90, far: 190 };
    let front;
    let reverse;
    try {
      front = buildDepthBinnedFrontToBackStrategy({ scenario, sourceGeometries });
      reverse = buildDepthBinnedReverseStrategy({ scenario, sourceGeometries });
      assert.equal(front.id, 'fixed-slice-depth-front-to-back');
      assert.equal(reverse.id, 'fixed-slice-depth-reverse');

      for (const strategy of [front, reverse]) {
        assert.equal(strategy.usesCompute, true);
        assert.equal(strategy.configuredDrawCommands, bucketCount);
        assert.equal(strategy.configuredRenderObjects, 1);
        assert.equal(strategy.configuredComputeDispatches, 4);
        assert.equal(strategy.configuredComputeSubmissions, 1);
        assert.equal(strategy.computeNodes.length, 4);
        assert.ok(strategy.computeNodes.every((node) => (
          node.count === null && node.countNode === null
        )));
        assert.deepEqual(
          strategy.computeNodes.map((node) => node.dispatchSize),
          [
            [Math.ceil((bucketCount * DEPTH_BIN_COUNT) / 64), 1, 1],
            [Math.ceil(objectCount / 64), 1, 1],
            [Math.ceil(bucketCount / 64), 1, 1],
            [Math.ceil(bucketCount / 64), 1, 1],
          ],
        );
        assert.equal(strategy.geometries.length, 1);
        assert.equal(strategy.materials.length, 1);
        assert.equal(strategy.root.children.length, 1);
        assert.equal(strategy.root.static, true);
        assert.equal(strategy.root.matrixAutoUpdate, false);
        assert.equal(strategy.root.matrixWorldAutoUpdate, false);
        assert.equal(strategy.root.children[0].frustumCulled, false);
        assert.equal(strategy.root.children[0].matrixAutoUpdate, false);
        assert.equal(strategy.root.children[0].matrixWorldAutoUpdate, false);
        assert.equal(strategy.root.children[0].geometry, strategy.geometries[0]);
        assert.equal(strategy.root.children[0].material, strategy.materials[0]);
        assert.equal(strategy.storageAttributes.length, 11);
        const binRecordsAttribute = strategy.storageAttributes[8];
        assert.equal(binRecordsAttribute.itemSize, 4);
        assert.equal(binRecordsAttribute.count, bucketCount * DEPTH_BIN_COUNT);
        assert.equal(binRecordsAttribute.array.length, bucketCount * DEPTH_BIN_COUNT * 4);
        assert.deepEqual(
          strategy.geometries[0].indirectOffset,
          Array.from(
            { length: bucketCount },
            (_, bucket) => bucket * INDEXED_INDIRECT_STRIDE_BYTES,
          ),
        );
        for (let bucket = 0; bucket < bucketCount; bucket += 1) {
          assert.equal(strategy.geometries[0].indirect.array[bucket * 5 + 4], 0);
        }
        assert.equal(strategy.diagnostics().bundleRecordCallbackCount, 0);
        strategy.root.children[0].onBeforeRender({ _currentRenderBundle: {} });
        assert.equal(strategy.diagnostics().bundleRecordCallbackCount, 1);
        strategy.root.children[0].onBeforeRender({ _currentRenderBundle: null });
        assert.equal(strategy.diagnostics().bundleRecordCallbackCount, 1);
      }

      assert.deepEqual(
        exactBytes(front.geometries[0].index),
        exactBytes(reverse.geometries[0].index),
      );
      assert.deepEqual(
        exactBytes(front.geometries[0].indirect),
        exactBytes(reverse.geometries[0].indirect),
      );
      assert.equal(front.storageAttributes.length, reverse.storageAttributes.length);
      for (let index = 0; index < front.storageAttributes.length; index += 1) {
        assert.equal(front.storageAttributes[index].constructor, reverse.storageAttributes[index].constructor);
        assert.equal(front.storageAttributes[index].itemSize, reverse.storageAttributes[index].itemSize);
        assert.deepEqual(
          exactBytes(front.storageAttributes[index]),
          exactBytes(reverse.storageAttributes[index]),
        );
      }

      const frontDiagnostics = front.diagnostics();
      const reverseDiagnostics = reverse.diagnostics();
      assert.deepEqual(frontDiagnostics, {
        kind: 'single-merged-geometry-depth-binned-fixed-slice',
        depthBinCount: 8,
        depthOrder: 'front-to-back',
        binTraversal: [0, 1, 2, 3, 4, 5, 6, 7],
        depthBinRange: { near: 90, far: 190 },
        reverseOrderUniformValue: false,
        bundleRecordCallbackCount: 1,
        meshCount: 1,
        geometryIdentityCount: 1,
        materialIdentityCount: 1,
        commandCount: bucketCount,
        zeroFirstInstanceCount: bucketCount,
        computeDispatchCount: 4,
        computeDispatchWorkItems: [bucketCount * 8, objectCount, bucketCount, bucketCount],
      });
      assert.deepEqual(reverseDiagnostics, {
        ...frontDiagnostics,
        depthOrder: 'reverse',
        binTraversal: [7, 6, 5, 4, 3, 2, 1, 0],
        reverseOrderUniformValue: true,
      });
    } finally {
      if (front) disposeStrategy(front);
      if (reverse) disposeStrategy(reverse);
      sourceGeometries.forEach((geometry) => geometry.dispose());
    }
  });
}

test('depth-binned strategy rejects a missing or degenerate preregistered range', () => {
  const sourceGeometries = createIndexedGeometryFixtures(1, 'medium');
  const scenario = createFixedSubsetScenario({
    objectCount: 64,
    bucketCount: 1,
    visibilityFraction: 0.8,
    geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
    seed: 7,
  });
  try {
    assert.throws(
      () => buildDepthBinnedFrontToBackStrategy({ scenario, sourceGeometries }),
      /depth range/,
    );
    scenario.depthBinRange = { near: 12, far: 12 };
    assert.throws(
      () => buildDepthBinnedReverseStrategy({ scenario, sourceGeometries }),
      /greater than/,
    );
  } finally {
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});
