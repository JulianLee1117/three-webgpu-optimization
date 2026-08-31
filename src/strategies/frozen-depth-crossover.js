import {
  BundleGroup,
  IndirectStorageBufferAttribute,
  Mesh,
  StorageBufferAttribute,
} from 'three/webgpu';
import {
  Fn,
  If,
  instanceIndex,
  storage,
  uint,
  uniform,
} from 'three/tsl';
import {
  DEPTH_BIN_COUNT,
  DEPTH_ORDER_FRONT_TO_BACK,
  DEPTH_ORDER_REVERSE,
  createPhysicalDepthBinSequenceCommitment,
  validateDepthBinReadback,
} from '../culling/depth-bin-layout.js';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_UNUSED_OBJECT_ID,
  createFrozenDepthCrossoverPacking,
  getFrozenDepthLaneSegment,
  validateFrozenDepthLane,
} from '../culling/frozen-depth-crossover.js';
import { createIndexedIndirectCommands } from '../culling/indexed-command-layout.js';
import { createStorageTransformMaterial } from '../materials/storage-transform.js';
import { createMergedIndexedBucketGeometry } from '../render/indexed-bucket-geometry.js';
import { compareMembership, validateIndexedCommands } from '../validation/membership.js';
import {
  createMembershipDigestEvidence,
  sha256CanonicalUint32,
} from '../validation/membership-digests.js';

export const FROZEN_DEPTH_CROSSOVER_STRATEGY_ID = 'fixed-slice-depth-frozen-crossover';

function asUint32(buffer) {
  if (buffer instanceof ArrayBuffer) return new Uint32Array(buffer);
  if (ArrayBuffer.isView(buffer)) {
    return new Uint32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  }
  throw new TypeError('GPU readback must be an ArrayBuffer or ArrayBuffer view.');
}

function freezeStaticTransform(object) {
  object.matrixAutoUpdate = false;
  object.matrixWorldAutoUpdate = false;
}

function exactUint32Sequence(left, right) {
  return left instanceof Uint32Array
    && right instanceof Uint32Array
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function exactFloat32Sequence(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function snapshotCamera(camera) {
  if (!camera?.matrixWorldInverse?.elements || !camera?.projectionMatrix?.elements) {
    throw new TypeError('Frozen depth crossover requires a camera with view and projection matrices.');
  }
  camera.updateMatrixWorld();
  return {
    view: Float32Array.from(camera.matrixWorldInverse.elements),
    projection: Float32Array.from(camera.projectionMatrix.elements),
  };
}

function validateInputs(scenario, sourceGeometries) {
  if (!Array.isArray(sourceGeometries)
    || sourceGeometries.length !== scenario?.bucketCount) {
    throw new RangeError('sourceGeometries length must equal scenario.bucketCount.');
  }
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    if (!sourceGeometries[bucket]?.index) {
      throw new Error(`Geometry bucket ${bucket} must be indexed.`);
    }
  }
  if (!(scenario?.matrices instanceof Float32Array)
    || scenario.matrices.length !== scenario.objectCount * 16) {
    throw new RangeError('scenario.matrices must contain one Float32 mat4 per object.');
  }
}

function copyLaneOffsets(laneOffsets) {
  return {
    [DEPTH_ORDER_FRONT_TO_BACK]: laneOffsets[DEPTH_ORDER_FRONT_TO_BACK],
    [DEPTH_ORDER_REVERSE]: laneOffsets[DEPTH_ORDER_REVERSE],
  };
}

export function buildFrozenDepthCrossoverStrategy({
  scenario,
  sourceGeometries,
  camera,
  laneStorageOrder = DEPTH_ORDER_FRONT_TO_BACK,
}) {
  validateInputs(scenario, sourceGeometries);
  validateFrozenDepthLane(laneStorageOrder, 'laneStorageOrder');
  const frozenCamera = snapshotCamera(camera);
  const packing = createFrozenDepthCrossoverPacking({
    scenario,
    viewMatrixElements: frozenCamera.view,
    laneStorageOrder,
  });
  const { geometry, firstIndexes } = createMergedIndexedBucketGeometry(
    sourceGeometries,
    scenario.bucketBases,
    scenario.bucketCounts,
  );
  const commandLayout = createIndexedIndirectCommands(
    sourceGeometries,
    scenario.bucketCounts,
    scenario.visibleCounts,
    firstIndexes,
  );
  const matrixAttribute = new StorageBufferAttribute(scenario.matrices, 16);
  const visibleIdsAttribute = new StorageBufferAttribute(packing.visibleIds, 1);
  const selectorChallengeAttribute = new StorageBufferAttribute(
    new Uint32Array(scenario.objectCount),
    1,
  );
  const indirectAttribute = new IndirectStorageBufferAttribute(commandLayout.commands, 5);
  let activeLane = DEPTH_ORDER_FRONT_TO_BACK;
  const visibleIdOffsetUniform = uniform(packing.laneOffsets[activeLane], 'uint');
  const frozenVisibleIdsRead = storage(
    visibleIdsAttribute,
    'uint',
    packing.visibleIds.length,
  ).toReadOnly();
  const selectorChallengeWrite = storage(
    selectorChallengeAttribute,
    'uint',
    scenario.objectCount,
  );
  const selectorChallengeNode = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.objectCount)), () => {
      selectorChallengeWrite.element(instanceIndex).assign(
        frozenVisibleIdsRead.element(visibleIdOffsetUniform.add(instanceIndex)),
      );
    });
  })().compute([Math.ceil(scenario.objectCount / 64), 1, 1]);
  const material = createStorageTransformMaterial({
    matrixAttribute,
    objectCount: scenario.objectCount,
    visibleIdsAttribute,
    visibleIdsCount: packing.visibleIds.length,
    visibleIdOffsetNode: visibleIdOffsetUniform,
  });
  geometry.setIndirect(indirectAttribute, Array.from(commandLayout.offsets));
  const mesh = new Mesh(geometry, material);
  freezeStaticTransform(mesh);
  mesh.frustumCulled = false;
  let bundleRecordCallbackCount = 0;
  mesh.onBeforeRender = (activeRenderer) => {
    if (activeRenderer?._currentRenderBundle !== null
      && activeRenderer?._currentRenderBundle !== undefined) {
      bundleRecordCallbackCount += 1;
    }
  };
  const root = new BundleGroup();
  freezeStaticTransform(root);
  root.name = `${FROZEN_DEPTH_CROSSOVER_STRATEGY_ID}-merged-indexed-indirect-bundle`;
  root.add(mesh);

  const laneOffsets = Object.freeze(copyLaneOffsets(packing.laneOffsets));
  const parityResources = {
    matrixAttribute,
    visibleIdsAttribute,
    visibleIdsCount: packing.visibleIds.length,
    objectCount: scenario.objectCount,
    get visibleIdOffset() {
      return visibleIdOffsetUniform.value;
    },
  };
  Object.freeze(parityResources);

  const diagnostics = () => ({
    kind: 'single-render-object-frozen-depth-crossover',
    laneStorageOrder,
    laneOffsets: copyLaneOffsets(laneOffsets),
    activeLane,
    activeVisibleIdOffset: visibleIdOffsetUniform.value,
    visibleIdsCount: packing.visibleIds.length,
    visibleIdSegmentLength: scenario.objectCount,
    depthBinCount: DEPTH_BIN_COUNT,
    bundleGroupStatic: root.static === true,
    bundleRecordCallbackCount,
    meshCount: root.children.length,
    geometryIdentityCount: new Set(root.children.map((child) => child.geometry)).size,
    materialIdentityCount: new Set(root.children.map((child) => child.material)).size,
    commandCount: scenario.bucketCount,
    zeroFirstInstanceCount: Array.from(
      { length: scenario.bucketCount },
      (_, bucket) => commandLayout.commands[bucket * 5 + 4],
    ).filter((value) => value === 0).length,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    diagnosticSelectorDispatchesPerValidation: 2,
    bundleGroupUuid: root.uuid,
    meshUuid: mesh.uuid,
    geometryUuid: geometry.uuid,
    materialUuid: material.uuid,
    matrixAttributeId: matrixAttribute.id,
    visibleIdsAttributeId: visibleIdsAttribute.id,
    indirectAttributeId: indirectAttribute.id,
    selectorChallengeAttributeId: selectorChallengeAttribute.id,
    bundleGroupVersion: root.version ?? null,
    matrixAttributeVersion: matrixAttribute.version,
    visibleIdsAttributeVersion: visibleIdsAttribute.version,
    indirectAttributeVersion: indirectAttribute.version,
    selectorUniformUuid: visibleIdOffsetUniform.uuid ?? null,
  });

  const lifecycleDiagnostics = () => ({
    kind: 'frozen-depth-crossover-static-bundle-lifecycle',
    laneStorageOrder,
    laneOffsets: copyLaneOffsets(laneOffsets),
    activeLane,
    activeVisibleIdOffset: visibleIdOffsetUniform.value,
    bundleGroupStatic: root.static === true,
    bundleRecordCallbackCount,
    meshCount: root.children.length,
    geometryIdentityCount: new Set(root.children.map((child) => child.geometry)).size,
    materialIdentityCount: new Set(root.children.map((child) => child.material)).size,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    bundleGroupUuid: root.uuid,
    meshUuid: mesh.uuid,
    geometryUuid: geometry.uuid,
    materialUuid: material.uuid,
    matrixAttributeId: matrixAttribute.id,
    visibleIdsAttributeId: visibleIdsAttribute.id,
    indirectAttributeId: indirectAttribute.id,
    selectorChallengeAttributeId: selectorChallengeAttribute.id,
    bundleGroupVersion: root.version ?? null,
    matrixAttributeVersion: matrixAttribute.version,
    visibleIdsAttributeVersion: visibleIdsAttribute.version,
    indirectAttributeVersion: indirectAttribute.version,
    selectorUniformUuid: visibleIdOffsetUniform.uuid ?? null,
  });

  function setActiveLane(lane) {
    validateFrozenDepthLane(lane);
    activeLane = lane;
    visibleIdOffsetUniform.value = laneOffsets[lane];
    return visibleIdOffsetUniform.value;
  }

  async function challengeSelector(renderer, lane) {
    validateFrozenDepthLane(lane);
    const previousLane = activeLane;
    try {
      setActiveLane(lane);
      renderer.compute(selectorChallengeNode);
      const challengeBuffer = asUint32(
        await renderer.getArrayBufferAsync(selectorChallengeAttribute),
      );
      const expected = getFrozenDepthLaneSegment(packing, lane);
      const [sha256, expectedSha256] = await Promise.all([
        sha256CanonicalUint32(challengeBuffer),
        sha256CanonicalUint32(expected),
      ]);
      return {
        pass: exactUint32Sequence(challengeBuffer, expected) && sha256 === expectedSha256,
        kind: 'gpu-selector-address-challenge',
        lane,
        storageOffset: laneOffsets[lane],
        elementCount: challengeBuffer.length,
        sha256,
        expectedSha256,
      };
    } finally {
      setActiveLane(previousLane);
    }
  }

  return {
    id: FROZEN_DEPTH_CROSSOVER_STRATEGY_ID,
    root,
    geometries: [geometry],
    materials: [material],
    parityResources,
    storageAttributes: [
      matrixAttribute,
      visibleIdsAttribute,
      selectorChallengeAttribute,
      indirectAttribute,
    ],
    computeNodes: [selectorChallengeNode],
    usesCompute: false,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    configuredSubmittedInstances: scenario.expectedVisibleCount,
    laneStorageOrder,
    laneOffsets,
    get activeLane() {
      return activeLane;
    },
    setActiveLane,
    challengeSelector,
    diagnostics,
    lifecycleDiagnostics,
    update(nextCamera) {
      const current = snapshotCamera(nextCamera);
      if (!exactFloat32Sequence(current.view, frozenCamera.view)
        || !exactFloat32Sequence(current.projection, frozenCamera.projection)) {
        throw new Error('Frozen depth crossover camera changed after packing.');
      }
    },
    submitCompute() {
      throw new Error('Frozen depth crossover has no compute submission.');
    },
    async validate(renderer, expectedIds) {
      if (!(expectedIds instanceof Uint32Array)) {
        throw new TypeError('expectedIds must be a Uint32Array.');
      }
      const [commandBuffer, visibleBuffer] = await Promise.all([
        renderer.getArrayBufferAsync(indirectAttribute),
        renderer.getArrayBufferAsync(visibleIdsAttribute),
      ]);
      const commands = asUint32(commandBuffer);
      const visibleIds = asUint32(visibleBuffer);
      if (visibleIds.length !== scenario.objectCount * 2) {
        throw new RangeError('Frozen crossover visible-ID readback has the wrong length.');
      }
      const actualCounts = new Uint32Array(scenario.bucketCount);
      for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
        actualCounts[bucket] = commands[bucket * 5 + 1];
      }
      const commandValidation = validateIndexedCommands({
        commands,
        geometries: sourceGeometries,
        expectedCounts: scenario.visibleCounts,
        expectedFirstIndexes: firstIndexes,
      });
      const [visibleIdsSha256, expectedVisibleIdsSha256, commandSha256] = await Promise.all([
        sha256CanonicalUint32(visibleIds),
        sha256CanonicalUint32(packing.visibleIds),
        sha256CanonicalUint32(commands),
      ]);
      const visibleIdsExactPackingMatch = exactUint32Sequence(
        visibleIds,
        packing.visibleIds,
      );
      const laneValidations = {};
      const selectorChallenges = {};
      for (const lane of FROZEN_DEPTH_CROSSOVER_LANES) {
        const offset = laneOffsets[lane];
        const laneIds = visibleIds.subarray(offset, offset + scenario.objectCount);
        const lanePacking = packing.lanes[lane];
        const membership = compareMembership({
          expectedIds,
          actualIds: laneIds,
          actualCounts,
          objectBuckets: scenario.objectBuckets,
          bucketBases: scenario.bucketBases,
          capacities: scenario.bucketCounts,
          objectCount: scenario.objectCount,
        });
        const membershipDigests = await createMembershipDigestEvidence({
          expectedIds,
          actualIds: laneIds,
          actualCounts,
          objectBuckets: scenario.objectBuckets,
          bucketBases: scenario.bucketBases,
          capacities: scenario.bucketCounts,
        });
        const depthBins = validateDepthBinReadback({
          actualIds: laneIds,
          objectBins: packing.objectBins,
          binCounts: packing.binCounts,
          binStarts: lanePacking.binStarts,
          commandCounts: actualCounts,
          expectedObjectBins: packing.objectBins,
          objectBuckets: scenario.objectBuckets,
          bucketBases: scenario.bucketBases,
          bucketCapacities: scenario.bucketCounts,
          order: lane,
        });
        depthBins.physicalBinSequenceCommitment = await createPhysicalDepthBinSequenceCommitment({
          actualIds: laneIds,
          binCounts: packing.binCounts,
          binStarts: lanePacking.binStarts,
          bucketBases: scenario.bucketBases,
          bucketCapacities: scenario.bucketCounts,
        });
        const storageSegmentSha256 = await sha256CanonicalUint32(laneIds);
        let paddingSentinelCount = 0;
        let paddingCorruptionCount = 0;
        for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
          const paddingStart = scenario.bucketBases[bucket] + actualCounts[bucket];
          const paddingEnd = scenario.bucketBases[bucket] + scenario.bucketCounts[bucket];
          for (let index = paddingStart; index < paddingEnd; index += 1) {
            paddingSentinelCount += 1;
            if (laneIds[index] !== FROZEN_DEPTH_UNUSED_OBJECT_ID) paddingCorruptionCount += 1;
          }
        }
        laneValidations[lane] = {
          pass: membership.pass && membershipDigests.pass && depthBins.pass,
          order: lane,
          storageOffset: offset,
          traversal: Array.from(lanePacking.traversal),
          membership,
          membershipDigests,
          depthBins,
          storageSegmentSha256,
          paddingSentinelCount,
          paddingCorruptionCount,
        };
        selectorChallenges[lane] = await challengeSelector(renderer, lane);
      }

      const commitmentsEqual = laneValidations[DEPTH_ORDER_FRONT_TO_BACK]
        .depthBins.physicalBinSequenceCommitment.sha256
        === laneValidations[DEPTH_ORDER_REVERSE]
          .depthBins.physicalBinSequenceCommitment.sha256;
      const expectedIdsMatchScenario = exactUint32Sequence(
        expectedIds,
        scenario.expectedVisibleIds,
      );
      const rawLaneSequencesDiffer = laneValidations[DEPTH_ORDER_FRONT_TO_BACK]
        .storageSegmentSha256
        !== laneValidations[DEPTH_ORDER_REVERSE].storageSegmentSha256;
      const representation = diagnostics();
      const lifecycle = lifecycleDiagnostics();
      const representationPass = representation.bundleGroupStatic === true
        && representation.bundleRecordCallbackCount === 1
        && representation.meshCount === 1
        && representation.geometryIdentityCount === 1
        && representation.materialIdentityCount === 1
        && representation.commandCount === scenario.bucketCount
        && representation.zeroFirstInstanceCount === scenario.bucketCount
        && representation.visibleIdsCount === scenario.objectCount * 2
        && representation.visibleIdSegmentLength === scenario.objectCount
        && representation.configuredComputeDispatches === 0
        && representation.configuredComputeSubmissions === 0;
      const lifecyclePass = lifecycle.bundleGroupStatic === true
        && lifecycle.bundleRecordCallbackCount === 1
        && lifecycle.configuredComputeDispatches === 0
        && lifecycle.configuredComputeSubmissions === 0;

      return {
        pass: expectedIdsMatchScenario
          && visibleIdsExactPackingMatch
          && visibleIdsSha256 === expectedVisibleIdsSha256
          && commandValidation.pass
          && commitmentsEqual
          && rawLaneSequencesDiffer
          && FROZEN_DEPTH_CROSSOVER_LANES.every((lane) => laneValidations[lane].pass)
          && FROZEN_DEPTH_CROSSOVER_LANES.every(
            (lane) => laneValidations[lane].paddingCorruptionCount === 0,
          )
          && FROZEN_DEPTH_CROSSOVER_LANES.every((lane) => selectorChallenges[lane].pass)
          && representationPass
          && lifecyclePass,
        kind: 'frozen-depth-crossover-exact-paired-snapshots',
        expectedIdsMatchScenario,
        visibleIdsByteLength: visibleIds.byteLength,
        visibleIdsSha256,
        expectedVisibleIdsSha256,
        visibleIdsExactPackingMatch,
        commandSha256,
        laneStorageOrder,
        laneOffsets: copyLaneOffsets(laneOffsets),
        activeLane,
        activeVisibleIdOffset: visibleIdOffsetUniform.value,
        physicalBinSequenceCommitmentsEqual: commitmentsEqual,
        rawLaneSequencesDiffer,
        physicalBinSequenceSha256: commitmentsEqual
          ? laneValidations[DEPTH_ORDER_FRONT_TO_BACK]
            .depthBins.physicalBinSequenceCommitment.sha256
          : null,
        commandValidation,
        lanes: laneValidations,
        selectorChallenges,
        representation,
        lifecycle,
      };
    },
  };
}
