import assert from 'node:assert/strict';
import test from 'node:test';
import { PerspectiveCamera } from 'three';
import {
  DEPTH_BIN_COUNT,
  DEPTH_ORDER_FRONT_TO_BACK,
  DEPTH_ORDER_REVERSE,
  createPhysicalDepthBinSequenceCommitment,
} from '../src/culling/depth-bin-layout.js';
import {
  FROZEN_DEPTH_UNUSED_OBJECT_ID,
  createFrozenDepthCrossoverPacking,
  getFrozenDepthLaneSegment,
} from '../src/culling/frozen-depth-crossover.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  FROZEN_DEPTH_CROSSOVER_STRATEGY_ID,
  buildFrozenDepthCrossoverStrategy,
} from '../src/strategies/frozen-depth-crossover.js';
import { disposeStrategyResources } from '../src/strategies/resources.js';
import { compareMembership } from '../src/validation/membership.js';
import { resolveRenderParityResources } from '../src/validation/render-parity.js';

const OBJECT_COUNT = 8;
const BUCKET_COUNT = 2;

function identityMatrices(count) {
  const matrices = new Float32Array(count * 16);
  for (let objectId = 0; objectId < count; objectId += 1) {
    const offset = objectId * 16;
    matrices[offset] = 1;
    matrices[offset + 5] = 1;
    matrices[offset + 10] = 1;
    matrices[offset + 15] = 1;
  }
  return matrices;
}

function syntheticScenario() {
  // Visible nearest-surface depths classify as:
  // bucket 0 => object 0/bin 0, 1/bin 2, 3/bin 7
  // bucket 1 => object 4/bin 1, 5/bin 6, 7/bin 1
  const nearestDepths = [4, 24, 34, 74, 14, 64, 44, 16];
  const bounds = new Float32Array(OBJECT_COUNT * 4);
  for (let objectId = 0; objectId < OBJECT_COUNT; objectId += 1) {
    bounds[objectId * 4 + 2] = -(nearestDepths[objectId] + 1);
    bounds[objectId * 4 + 3] = 1;
  }
  return {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    bucketCounts: Uint32Array.from([4, 4]),
    bucketBases: Uint32Array.from([0, 4]),
    visibleCounts: Uint32Array.from([3, 3]),
    objectBuckets: Uint32Array.from([0, 0, 0, 0, 1, 1, 1, 1]),
    matrices: identityMatrices(OBJECT_COUNT),
    bounds,
    expectedVisibleIds: Uint32Array.from([0, 1, 3, 4, 5, 7]),
    expectedVisibleCount: 6,
    depthBinRange: { near: 0, far: 80 },
  };
}

const IDENTITY_VIEW = Object.freeze([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function laneCommitment(packing, scenario, lane) {
  return createPhysicalDepthBinSequenceCommitment({
    actualIds: getFrozenDepthLaneSegment(packing, lane),
    binCounts: packing.binCounts,
    binStarts: packing.lanes[lane].binStarts,
    bucketBases: scenario.bucketBases,
    bucketCapacities: scenario.bucketCounts,
  });
}

test('CPU frozen crossover reverses only physical-bin blocks and normalizes commitments', async () => {
  const scenario = syntheticScenario();
  const packing = createFrozenDepthCrossoverPacking({
    scenario,
    viewMatrixElements: IDENTITY_VIEW,
  });
  assert.equal(packing.laneStorageOrder, DEPTH_ORDER_FRONT_TO_BACK);
  assert.deepEqual(packing.laneOffsets, {
    [DEPTH_ORDER_FRONT_TO_BACK]: 0,
    [DEPTH_ORDER_REVERSE]: OBJECT_COUNT,
  });
  assert.deepEqual(
    Array.from(getFrozenDepthLaneSegment(packing, DEPTH_ORDER_FRONT_TO_BACK)),
    [0, 1, 3, FROZEN_DEPTH_UNUSED_OBJECT_ID, 4, 7, 5, FROZEN_DEPTH_UNUSED_OBJECT_ID],
  );
  assert.deepEqual(
    Array.from(getFrozenDepthLaneSegment(packing, DEPTH_ORDER_REVERSE)),
    [3, 1, 0, FROZEN_DEPTH_UNUSED_OBJECT_ID, 5, 4, 7, FROZEN_DEPTH_UNUSED_OBJECT_ID],
  );
  assert.deepEqual(
    Array.from(packing.lanes[DEPTH_ORDER_FRONT_TO_BACK].traversal),
    [0, 1, 2, 3, 4, 5, 6, 7],
  );
  assert.deepEqual(
    Array.from(packing.lanes[DEPTH_ORDER_REVERSE].traversal),
    [7, 6, 5, 4, 3, 2, 1, 0],
  );
  assert.equal(packing.binCounts.length, BUCKET_COUNT * DEPTH_BIN_COUNT);

  const [frontCommitment, reverseCommitment] = await Promise.all([
    laneCommitment(packing, scenario, DEPTH_ORDER_FRONT_TO_BACK),
    laneCommitment(packing, scenario, DEPTH_ORDER_REVERSE),
  ]);
  assert.deepEqual(frontCommitment, reverseCommitment);

  for (const lane of [DEPTH_ORDER_FRONT_TO_BACK, DEPTH_ORDER_REVERSE]) {
    const membership = compareMembership({
      expectedIds: scenario.expectedVisibleIds,
      actualIds: getFrozenDepthLaneSegment(packing, lane),
      actualCounts: scenario.visibleCounts,
      objectBuckets: scenario.objectBuckets,
      bucketBases: scenario.bucketBases,
      capacities: scenario.bucketCounts,
      objectCount: scenario.objectCount,
    });
    assert.equal(membership.pass, true);
  }

  const wrongBucket = getFrozenDepthLaneSegment(packing, DEPTH_ORDER_FRONT_TO_BACK).slice();
  [wrongBucket[0], wrongBucket[4]] = [wrongBucket[4], wrongBucket[0]];
  assert.equal(compareMembership({
    expectedIds: scenario.expectedVisibleIds,
    actualIds: wrongBucket,
    actualCounts: scenario.visibleCounts,
    objectBuckets: scenario.objectBuckets,
    bucketBases: scenario.bucketBases,
    capacities: scenario.bucketCounts,
    objectCount: scenario.objectCount,
  }).pass, false);

  const changedWithinBin = getFrozenDepthLaneSegment(packing, DEPTH_ORDER_REVERSE).slice();
  const binOneIndex = DEPTH_BIN_COUNT + 1;
  const binOneStart = scenario.bucketBases[1]
    + packing.lanes[DEPTH_ORDER_REVERSE].binStarts[binOneIndex];
  [changedWithinBin[binOneStart], changedWithinBin[binOneStart + 1]] = [
    changedWithinBin[binOneStart + 1],
    changedWithinBin[binOneStart],
  ];
  const changedCommitment = await createPhysicalDepthBinSequenceCommitment({
    actualIds: changedWithinBin,
    binCounts: packing.binCounts,
    binStarts: packing.lanes[DEPTH_ORDER_REVERSE].binStarts,
    bucketBases: scenario.bucketBases,
    bucketCapacities: scenario.bucketCounts,
  });
  assert.notEqual(changedCommitment.sha256, frontCommitment.sha256);
});

test('laneStorageOrder swaps physical segments without changing either logical lane', () => {
  const scenario = syntheticScenario();
  const frontFirst = createFrozenDepthCrossoverPacking({
    scenario,
    viewMatrixElements: IDENTITY_VIEW,
  });
  const reverseFirst = createFrozenDepthCrossoverPacking({
    scenario,
    viewMatrixElements: IDENTITY_VIEW,
    laneStorageOrder: DEPTH_ORDER_REVERSE,
  });
  assert.deepEqual(reverseFirst.laneOffsets, {
    [DEPTH_ORDER_FRONT_TO_BACK]: OBJECT_COUNT,
    [DEPTH_ORDER_REVERSE]: 0,
  });
  for (const lane of [DEPTH_ORDER_FRONT_TO_BACK, DEPTH_ORDER_REVERSE]) {
    assert.deepEqual(
      getFrozenDepthLaneSegment(reverseFirst, lane),
      getFrozenDepthLaneSegment(frontFirst, lane),
    );
  }
  assert.deepEqual(
    reverseFirst.visibleIds.subarray(0, OBJECT_COUNT),
    getFrozenDepthLaneSegment(frontFirst, DEPTH_ORDER_REVERSE),
  );
});

test('CPU packing rejects adversarial scenario membership and storage inputs', () => {
  const duplicate = syntheticScenario();
  duplicate.expectedVisibleIds = Uint32Array.from([0, 1, 1, 4, 5, 7]);
  assert.throws(
    () => createFrozenDepthCrossoverPacking({
      scenario: duplicate,
      viewMatrixElements: IDENTITY_VIEW,
    }),
    /duplicated|strictly ascending/,
  );

  const wrongBucket = syntheticScenario();
  wrongBucket.objectBuckets[0] = 1;
  assert.throws(
    () => createFrozenDepthCrossoverPacking({
      scenario: wrongBucket,
      viewMatrixElements: IDENTITY_VIEW,
    }),
    /does not match its contiguous bucket/,
  );

  const wrongVisibleCount = syntheticScenario();
  wrongVisibleCount.visibleCounts[0] = 2;
  assert.throws(
    () => createFrozenDepthCrossoverPacking({
      scenario: wrongVisibleCount,
      viewMatrixElements: IDENTITY_VIEW,
    }),
    /visible counts do not match/,
  );

  assert.throws(
    () => createFrozenDepthCrossoverPacking({
      scenario: syntheticScenario(),
      viewMatrixElements: IDENTITY_VIEW,
      laneStorageOrder: 'allocation-dependent',
    }),
    /laneStorageOrder/,
  );
});

function readbackRenderer(strategy) {
  let challengedLane = null;
  return {
    compute() {
      challengedLane = strategy.activeLane;
    },
    getArrayBufferAsync(attribute) {
      if (attribute === strategy.storageAttributes[2]) {
        const visibleIds = strategy.storageAttributes[1].array;
        const start = strategy.laneOffsets[challengedLane];
        const lane = visibleIds.subarray(start, start + OBJECT_COUNT);
        return Promise.resolve(lane.buffer.slice(
          lane.byteOffset,
          lane.byteOffset + lane.byteLength,
        ));
      }
      const { array } = attribute;
      return Promise.resolve(array.buffer.slice(
        array.byteOffset,
        array.byteOffset + array.byteLength,
      ));
    },
  };
}

test('frozen crossover strategy shares one render object and validates both offset-selected lanes', async () => {
  const scenario = syntheticScenario();
  const sourceGeometries = createIndexedGeometryFixtures(BUCKET_COUNT, 'medium');
  const camera = new PerspectiveCamera(60, 16 / 9, 0.1, 1_000);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  let strategy;
  try {
    strategy = buildFrozenDepthCrossoverStrategy({
      scenario,
      sourceGeometries,
      camera,
      laneStorageOrder: DEPTH_ORDER_REVERSE,
    });
    assert.equal(strategy.id, FROZEN_DEPTH_CROSSOVER_STRATEGY_ID);
    assert.equal(strategy.usesCompute, false);
    assert.equal(strategy.configuredComputeDispatches, 0);
    assert.equal(strategy.configuredComputeSubmissions, 0);
    assert.equal(strategy.computeNodes.length, 1);
    assert.equal(strategy.configuredDrawCommands, BUCKET_COUNT);
    assert.equal(strategy.configuredRenderObjects, 1);
    assert.equal(strategy.root.children.length, 1);
    assert.equal(strategy.root.static, true);
    assert.equal(strategy.geometries.length, 1);
    assert.equal(strategy.materials.length, 1);
    assert.equal(strategy.storageAttributes.length, 4);
    assert.throws(() => strategy.submitCompute(), /no compute submission/);

    const materialIdentity = strategy.materials[0];
    assert.equal(strategy.activeLane, DEPTH_ORDER_FRONT_TO_BACK);
    assert.equal(strategy.parityResources.visibleIdsCount, OBJECT_COUNT * 2);
    assert.equal(strategy.parityResources.visibleIdOffset, OBJECT_COUNT);
    assert.equal(strategy.setActiveLane(DEPTH_ORDER_REVERSE), 0);
    assert.equal(strategy.activeLane, DEPTH_ORDER_REVERSE);
    assert.equal(strategy.parityResources.visibleIdOffset, 0);
    assert.equal(strategy.materials[0], materialIdentity);
    assert.throws(() => strategy.setActiveLane('sideways'), /lane must be one of/);

    const normalizedParity = resolveRenderParityResources(strategy);
    assert.equal(normalizedParity.visibleIdsCount, OBJECT_COUNT * 2);
    assert.equal(normalizedParity.visibleIdOffset, 0);
    strategy.setActiveLane(DEPTH_ORDER_FRONT_TO_BACK);
    assert.equal(resolveRenderParityResources(strategy).visibleIdOffset, OBJECT_COUNT);

    const invalidParityStrategy = {
      id: 'invalid-offset',
      geometries: strategy.geometries,
      materials: strategy.materials,
      parityResources: {
        matrixAttribute: strategy.parityResources.matrixAttribute,
        visibleIdsAttribute: strategy.parityResources.visibleIdsAttribute,
        objectCount: OBJECT_COUNT,
        visibleIdsCount: OBJECT_COUNT,
        visibleIdOffset: OBJECT_COUNT,
      },
    };
    assert.throws(
      () => resolveRenderParityResources(invalidParityStrategy),
      /invalid parity visibleIdOffset/,
    );

    assert.equal(strategy.diagnostics().bundleRecordCallbackCount, 0);
    strategy.root.children[0].onBeforeRender({ _currentRenderBundle: {} });
    assert.equal(strategy.diagnostics().bundleRecordCallbackCount, 1);
    strategy.root.children[0].onBeforeRender({ _currentRenderBundle: null });
    assert.equal(strategy.diagnostics().bundleRecordCallbackCount, 1);

    strategy.update(camera);
    const validation = await strategy.validate(
      readbackRenderer(strategy),
      scenario.expectedVisibleIds,
    );
    assert.equal(validation.pass, true, JSON.stringify(validation, null, 2));
    assert.equal(validation.kind, 'frozen-depth-crossover-exact-paired-snapshots');
    assert.equal(validation.laneStorageOrder, DEPTH_ORDER_REVERSE);
    assert.deepEqual(validation.laneOffsets, {
      [DEPTH_ORDER_FRONT_TO_BACK]: OBJECT_COUNT,
      [DEPTH_ORDER_REVERSE]: 0,
    });
    assert.equal(validation.physicalBinSequenceCommitmentsEqual, true);
    assert.match(validation.physicalBinSequenceSha256, /^[0-9a-f]{64}$/);
    assert.equal(validation.commandValidation.pass, true);
    assert.equal(validation.representation.meshCount, 1);
    assert.equal(validation.representation.materialIdentityCount, 1);
    assert.equal(validation.representation.configuredComputeDispatches, 0);
    assert.equal(validation.lifecycle.bundleRecordCallbackCount, 1);
    for (const field of ['bundleGroupUuid', 'meshUuid', 'geometryUuid', 'materialUuid']) {
      assert.match(validation.representation[field], /^[0-9a-f-]{36}$/i);
      assert.equal(validation.lifecycle[field], validation.representation[field]);
    }
    for (const field of [
      'matrixAttributeId',
      'visibleIdsAttributeId',
      'indirectAttributeId',
      'selectorChallengeAttributeId',
    ]) {
      assert.equal(Number.isInteger(validation.representation[field]), true);
      assert.equal(validation.lifecycle[field], validation.representation[field]);
    }
    for (const lane of [DEPTH_ORDER_FRONT_TO_BACK, DEPTH_ORDER_REVERSE]) {
      assert.equal(validation.lanes[lane].pass, true);
      assert.equal(validation.lanes[lane].storageOffset, validation.laneOffsets[lane]);
      assert.equal(validation.lanes[lane].membership.pass, true);
      assert.equal(validation.lanes[lane].depthBins.pass, true);
    }

    const wrongExpected = scenario.expectedVisibleIds.slice();
    [wrongExpected[0], wrongExpected[1]] = [wrongExpected[1], wrongExpected[0]];
    const rejected = await strategy.validate(readbackRenderer(strategy), wrongExpected);
    assert.equal(rejected.pass, false);
    assert.equal(rejected.expectedIdsMatchScenario, false);

    camera.position.z = 1;
    assert.throws(() => strategy.update(camera), /camera changed after packing/);

    const deleted = [];
    disposeStrategyResources({
      _attributes: { delete(attribute) { deleted.push(attribute); } },
    }, strategy);
    strategy = null;
    assert.equal(new Set(deleted).size, 4);
  } finally {
    if (strategy) {
      disposeStrategyResources({ _attributes: { delete() {} } }, strategy);
    }
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});
