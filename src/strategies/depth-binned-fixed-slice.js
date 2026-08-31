import {
  BundleGroup,
  IndirectStorageBufferAttribute,
  Matrix4,
  Mesh,
  StorageBufferAttribute,
} from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicLoad,
  atomicStore,
  dot,
  float,
  instanceIndex,
  storage,
  struct,
  uint,
  uniform,
  vec4,
} from 'three/tsl';
import {
  DEPTH_BIN_COUNT,
  DEPTH_ORDER_FRONT_TO_BACK,
  DEPTH_ORDER_REVERSE,
  createDepthBinTraversal,
  createExpectedObjectDepthBins,
  createPhysicalDepthBinSequenceCommitment,
  validateDepthBinReadback,
  validateDepthRange,
} from '../culling/depth-bin-layout.js';
import { createFrustumPlaneState, updateFrustumPlaneState } from '../culling/frustum-planes.js';
import { createIndexedIndirectCommands } from '../culling/indexed-command-layout.js';
import { createStorageTransformMaterial } from '../materials/storage-transform.js';
import { createMergedIndexedBucketGeometry } from '../render/indexed-bucket-geometry.js';
import { compareMembership, validateIndexedCommands } from '../validation/membership.js';
import { createMembershipDigestEvidence } from '../validation/membership-digests.js';

const COMPUTE_WORKGROUP_SIZE = 64;

function dispatchSizeForWorkItems(workItemCount) {
  return [Math.ceil(workItemCount / COMPUTE_WORKGROUP_SIZE), 1, 1];
}

function sphereInsideNode(sphere, planeUniforms) {
  let inside = dot(planeUniforms[0].xyz, sphere.xyz)
    .add(planeUniforms[0].w)
    .greaterThanEqual(sphere.w.mul(float(-1)));
  for (let plane = 1; plane < planeUniforms.length; plane += 1) {
    inside = inside.and(
      dot(planeUniforms[plane].xyz, sphere.xyz)
        .add(planeUniforms[plane].w)
        .greaterThanEqual(sphere.w.mul(float(-1))),
    );
  }
  return inside;
}

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

function validateInputs(scenario, sourceGeometries) {
  if (!Number.isInteger(scenario?.objectCount) || scenario.objectCount <= 0) {
    throw new RangeError('Depth-binned fixed-slice requires a positive objectCount.');
  }
  if (!Number.isInteger(scenario.bucketCount) || scenario.bucketCount <= 0) {
    throw new RangeError('Depth-binned fixed-slice requires a positive bucketCount.');
  }
  if (!Array.isArray(sourceGeometries)
    || sourceGeometries.length !== scenario.bucketCount) {
    throw new RangeError('sourceGeometries length must equal scenario.bucketCount.');
  }
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    if (!sourceGeometries[bucket]?.index) {
      throw new Error(`Geometry bucket ${bucket} must be indexed.`);
    }
  }
  const objectArrays = [
    ['objectBuckets', scenario.objectBuckets, scenario.objectCount],
    ['cullOrder', scenario.cullOrder, scenario.objectCount],
    ['matrices', scenario.matrices, scenario.objectCount * 16],
    ['bounds', scenario.bounds, scenario.objectCount * 4],
  ];
  for (const [label, value, expectedLength] of objectArrays) {
    if (!value || value.length !== expectedLength) {
      throw new RangeError(`scenario.${label} has an invalid length.`);
    }
  }
  const bucketArrays = [
    ['bucketBases', scenario.bucketBases],
    ['bucketCounts', scenario.bucketCounts],
    ['visibleCounts', scenario.visibleCounts],
  ];
  for (const [label, value] of bucketArrays) {
    if (!(value instanceof Uint32Array) || value.length !== scenario.bucketCount) {
      throw new RangeError(`scenario.${label} must contain one uint32 per bucket.`);
    }
  }
  let cursor = 0;
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    if (scenario.bucketBases[bucket] !== cursor) {
      throw new RangeError('Depth-binned fixed-slice requires contiguous bucket-major objects.');
    }
    cursor += scenario.bucketCounts[bucket];
  }
  if (cursor !== scenario.objectCount) {
    throw new RangeError('scenario.bucketCounts must sum to scenario.objectCount.');
  }
  return validateDepthRange(scenario.depthBinRange);
}

function buildDepthBinnedFixedSliceStrategy(
  { scenario, sourceGeometries },
  { id, order },
) {
  const depthRange = validateInputs(scenario, sourceGeometries);
  const reverseOrder = order === DEPTH_ORDER_REVERSE;
  const traversal = createDepthBinTraversal(order);
  const binRecordCount = scenario.bucketCount * DEPTH_BIN_COUNT;
  if (!Number.isSafeInteger(binRecordCount)) {
    throw new RangeError('Depth-bin record count exceeds JavaScript safe integer capacity.');
  }

  const { geometry, firstIndexes } = createMergedIndexedBucketGeometry(
    sourceGeometries,
    scenario.bucketBases,
    scenario.bucketCounts,
  );
  const commandLayout = createIndexedIndirectCommands(
    sourceGeometries,
    scenario.bucketCounts,
    null,
    firstIndexes,
  );
  const matrixAttribute = new StorageBufferAttribute(scenario.matrices, 16);
  const boundsAttribute = new StorageBufferAttribute(scenario.bounds, 4);
  const bucketAttribute = new StorageBufferAttribute(scenario.objectBuckets, 1);
  const baseAttribute = new StorageBufferAttribute(scenario.bucketBases, 1);
  const capacityAttribute = new StorageBufferAttribute(commandLayout.capacities, 1);
  const cullOrderAttribute = new StorageBufferAttribute(scenario.cullOrder, 1);
  const visibleIdsAttribute = new StorageBufferAttribute(new Uint32Array(scenario.objectCount), 1);
  const objectBinsAttribute = new StorageBufferAttribute(new Uint32Array(scenario.objectCount), 1);
  // One interleaved binding keeps scatter at WebGPU's portable per-stage limit
  // of eight storage buffers. The explicit fourth word keeps the CPU attribute
  // and WGSL struct-array strides identical under Three.js r185.
  const binRecordsAttribute = new StorageBufferAttribute(
    new Uint32Array(binRecordCount * 4),
    4,
  );
  const overflowAttribute = new StorageBufferAttribute(new Uint32Array(1), 1);
  const indirectAttribute = new IndirectStorageBufferAttribute(commandLayout.commands, 5);
  const planeState = createFrustumPlaneState();
  const viewMatrixState = new Matrix4();
  const viewMatrixUniform = uniform(viewMatrixState);
  const depthNearUniform = uniform(depthRange.near);
  const inverseDepthSpanUniform = uniform(1 / (depthRange.far - depthRange.near));
  // Both modes compile the same prefix shader. Only this uniform value differs.
  const reverseOrderUniform = uniform(reverseOrder);
  let validationViewMatrixElements = null;

  const boundsRead = storage(boundsAttribute, 'vec4', scenario.objectCount).toReadOnly();
  const bucketRead = storage(bucketAttribute, 'uint', scenario.objectCount).toReadOnly();
  const baseRead = storage(baseAttribute, 'uint', scenario.bucketCount).toReadOnly();
  const capacityRead = storage(capacityAttribute, 'uint', scenario.bucketCount).toReadOnly();
  const orderRead = storage(cullOrderAttribute, 'uint', scenario.objectCount).toReadOnly();
  const visibleWrite = storage(visibleIdsAttribute, 'uint', scenario.objectCount);
  const objectBinsWrite = storage(objectBinsAttribute, 'uint', scenario.objectCount);
  const objectBinsRead = storage(objectBinsAttribute, 'uint', scenario.objectCount).toReadOnly();
  const binRecordStruct = struct({
    count: { type: 'uint', atomic: true },
    writeOffset: { type: 'uint', atomic: true },
    start: 'uint',
    padding: 'uint',
  }, 'DepthBinnedFixedSliceRecord');
  const binRecords = storage(binRecordsAttribute, binRecordStruct, binRecordCount);
  const overflowAtomic = storage(overflowAttribute, 'uint', 1).toAtomic();
  const drawStruct = struct({
    indexCount: 'uint',
    instanceCount: { type: 'uint', atomic: true },
    firstIndex: 'uint',
    baseVertex: 'int',
    firstInstance: 'uint',
  }, 'DepthBinnedFixedSliceIndexedDraw');
  const drawStorage = storage(indirectAttribute, drawStruct, commandLayout.recordCount);

  const reset = Fn(() => {
    If(instanceIndex.lessThan(uint(binRecordCount)), () => {
      const binRecord = binRecords.element(instanceIndex);
      atomicStore(binRecord.get('count'), uint(0));
      atomicStore(binRecord.get('writeOffset'), uint(0));
      binRecord.get('start').assign(uint(0));
      binRecord.get('padding').assign(uint(0));
    });
    If(instanceIndex.lessThan(uint(scenario.bucketCount)), () => {
      atomicStore(drawStorage.element(instanceIndex).get('instanceCount'), uint(0));
    });
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(overflowAtomic.element(uint(0)), uint(0));
    });
  })().compute(dispatchSizeForWorkItems(binRecordCount));

  const classifyAndCount = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.objectCount)), () => {
      const objectId = orderRead.element(instanceIndex).toVar('depthCullObjectId');
      objectBinsWrite.element(objectId).assign(uint(DEPTH_BIN_COUNT));
      const sphere = boundsRead.element(objectId).toVar('depthCullWorldSphere');
      If(sphereInsideNode(sphere, planeState.uniforms), () => {
        const nearestViewDepth = viewMatrixUniform
          .mul(vec4(sphere.xyz, float(1)))
          .z
          .negate()
          .sub(sphere.w)
          .toVar('nearestViewDepth');
        const normalizedDepth = nearestViewDepth
          .sub(depthNearUniform)
          .mul(inverseDepthSpanUniform)
          .clamp(float(0), float(1))
          .toVar('normalizedDepth');
        const physicalBin = uint(normalizedDepth.mul(float(DEPTH_BIN_COUNT)))
          .min(uint(DEPTH_BIN_COUNT - 1))
          .toVar('physicalDepthBin');
        objectBinsWrite.element(objectId).assign(physicalBin);
        const bucket = bucketRead.element(objectId).toVar('depthCullBucket');
        const binIndex = bucket
          .mul(uint(DEPTH_BIN_COUNT))
          .add(physicalBin)
          .toVar('depthCountIndex');
        const priorCount = atomicAdd(binRecords.element(binIndex).get('count'), uint(1))
          .toVar('priorDepthBinCount');
        If(priorCount.greaterThanEqual(capacityRead.element(bucket)), () => {
          atomicStore(overflowAtomic.element(uint(0)), uint(1));
        });
      });
    });
  })().compute(dispatchSizeForWorkItems(scenario.objectCount));

  const orderedPrefix = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.bucketCount)), () => {
      const cursor = uint(0).toVar('orderedDepthCursor');
      for (let traversalIndex = 0; traversalIndex < DEPTH_BIN_COUNT; traversalIndex += 1) {
        const physicalBin = reverseOrderUniform.select(
          uint(DEPTH_BIN_COUNT - 1 - traversalIndex),
          uint(traversalIndex),
        ).toVar(`orderedPhysicalBin${traversalIndex}`);
        const binIndex = instanceIndex
          .mul(uint(DEPTH_BIN_COUNT))
          .add(physicalBin)
          .toVar(`orderedBinIndex${traversalIndex}`);
        const binRecord = binRecords.element(binIndex);
        binRecord.get('start').assign(cursor);
        cursor.addAssign(atomicLoad(binRecord.get('count')));
      }
      atomicStore(drawStorage.element(instanceIndex).get('instanceCount'), cursor);
      If(cursor.greaterThan(capacityRead.element(instanceIndex)), () => {
        atomicStore(overflowAtomic.element(uint(0)), uint(1));
      });
    });
  })().compute(dispatchSizeForWorkItems(scenario.bucketCount));

  const scatter = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.bucketCount)), () => {
      const bucket = instanceIndex.toVar('depthScatterBucket');
      const bucketBase = baseRead.element(bucket).toVar('depthScatterBucketBase');
      const bucketCapacity = capacityRead.element(bucket).toVar('depthScatterBucketCapacity');
      const bucketEnd = bucketBase.add(bucketCapacity).toVar('depthScatterBucketEnd');
      // One invocation owns each bucket and visits its contiguous object IDs in
      // ascending order. Atomic offsets retain the same storage semantics while
      // becoming deterministic because no other invocation writes this bucket.
      Loop({
        start: uint(0),
        end: bucketCapacity,
        type: 'uint',
        condition: '<',
      }, ({ i }) => {
        const objectId = bucketBase.add(i).toVar('depthScatterObjectId');
        const physicalBin = objectBinsRead.element(objectId).toVar('scatterPhysicalDepthBin');
        If(physicalBin.lessThan(uint(DEPTH_BIN_COUNT)), () => {
          const binIndex = bucket
            .mul(uint(DEPTH_BIN_COUNT))
            .add(physicalBin)
            .toVar('depthScatterBinIndex');
          const binRecord = binRecords.element(binIndex);
          const slot = atomicAdd(binRecord.get('writeOffset'), uint(1))
            .toVar('depthBinWriteSlot');
          const destination = bucketBase
            .add(binRecord.get('start'))
            .add(slot)
            .toVar('depthScatterDestination');
          If(
            slot.lessThan(atomicLoad(binRecord.get('count')))
              .and(destination.lessThan(bucketEnd)),
            () => {
              visibleWrite.element(destination).assign(objectId);
            },
          ).Else(() => {
            atomicStore(overflowAtomic.element(uint(0)), uint(1));
          });
        });
      });
    });
  })().compute(dispatchSizeForWorkItems(scenario.bucketCount));

  const material = createStorageTransformMaterial({
    matrixAttribute,
    objectCount: scenario.objectCount,
    visibleIdsAttribute,
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
  root.name = `${id}-merged-indexed-indirect-bundle`;
  root.add(mesh);

  const computeDispatchWorkItems = Object.freeze([
    binRecordCount,
    scenario.objectCount,
    scenario.bucketCount,
    scenario.bucketCount,
  ]);
  const diagnostics = () => ({
    kind: 'single-merged-geometry-depth-binned-fixed-slice',
    depthBinCount: DEPTH_BIN_COUNT,
    depthOrder: order,
    binTraversal: Array.from(traversal),
    depthBinRange: { ...depthRange },
    reverseOrderUniformValue: reverseOrder,
    bundleRecordCallbackCount,
    meshCount: root.children.length,
    geometryIdentityCount: new Set(root.children.map((child) => child.geometry)).size,
    materialIdentityCount: new Set(root.children.map((child) => child.material)).size,
    commandCount: scenario.bucketCount,
    zeroFirstInstanceCount: Array.from(
      { length: scenario.bucketCount },
      (_, bucket) => commandLayout.commands[bucket * 5 + 4],
    ).filter((value) => value === 0).length,
    computeDispatchCount: 4,
    computeDispatchWorkItems: [...computeDispatchWorkItems],
  });

  return {
    id,
    root,
    geometries: [geometry],
    materials: [material],
    parityResources: Object.freeze({
      matrixAttribute,
      visibleIdsAttribute,
      objectCount: scenario.objectCount,
    }),
    storageAttributes: [
      matrixAttribute,
      boundsAttribute,
      bucketAttribute,
      baseAttribute,
      capacityAttribute,
      cullOrderAttribute,
      visibleIdsAttribute,
      objectBinsAttribute,
      binRecordsAttribute,
      overflowAttribute,
      indirectAttribute,
    ],
    computeNodes: [reset, classifyAndCount, orderedPrefix, scatter],
    usesCompute: true,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 4,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    diagnostics,
    update(camera, renderer) {
      updateFrustumPlaneState(planeState, camera, renderer);
      viewMatrixState.copy(camera.matrixWorldInverse);
      validationViewMatrixElements = Float32Array.from(viewMatrixState.elements);
    },
    submitCompute(renderer) {
      renderer.compute([reset, classifyAndCount, orderedPrefix, scatter]);
    },
    async validate(renderer, expectedIds) {
      if (!validationViewMatrixElements) {
        throw new Error('Depth-binned strategy must be updated with a camera before validation.');
      }
      const [
        commandBuffer,
        visibleBuffer,
        objectBinsBuffer,
        binRecordsBuffer,
        overflowBuffer,
      ] = await Promise.all([
        renderer.getArrayBufferAsync(indirectAttribute),
        renderer.getArrayBufferAsync(visibleIdsAttribute),
        renderer.getArrayBufferAsync(objectBinsAttribute),
        renderer.getArrayBufferAsync(binRecordsAttribute),
        renderer.getArrayBufferAsync(overflowAttribute),
      ]);
      const commands = asUint32(commandBuffer);
      const visibleIds = asUint32(visibleBuffer);
      const objectBins = asUint32(objectBinsBuffer);
      const interleavedBinRecords = asUint32(binRecordsBuffer);
      const binCounts = new Uint32Array(binRecordCount);
      const binStarts = new Uint32Array(binRecordCount);
      for (let index = 0; index < binRecordCount; index += 1) {
        binCounts[index] = interleavedBinRecords[index * 4];
        binStarts[index] = interleavedBinRecords[index * 4 + 2];
      }
      const overflow = asUint32(overflowBuffer)[0];
      const actualCounts = new Uint32Array(scenario.bucketCount);
      for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
        actualCounts[bucket] = commands[bucket * 5 + 1];
      }
      const membership = compareMembership({
        expectedIds,
        actualIds: visibleIds,
        actualCounts,
        objectBuckets: scenario.objectBuckets,
        bucketBases: scenario.bucketBases,
        capacities: scenario.bucketCounts,
        objectCount: scenario.objectCount,
      });
      const commandValidation = validateIndexedCommands({
        commands,
        geometries: sourceGeometries,
        expectedCounts: scenario.visibleCounts,
        expectedFirstIndexes: firstIndexes,
      });
      const membershipDigests = await createMembershipDigestEvidence({
        expectedIds,
        actualIds: visibleIds,
        actualCounts,
        objectBuckets: scenario.objectBuckets,
        bucketBases: scenario.bucketBases,
        capacities: scenario.bucketCounts,
      });
      const expectedObjectBins = createExpectedObjectDepthBins({
        bounds: scenario.bounds,
        objectCount: scenario.objectCount,
        expectedIds,
        viewMatrixElements: validationViewMatrixElements,
        depthRange,
      });
      const depthBins = validateDepthBinReadback({
        actualIds: visibleIds,
        objectBins,
        binCounts,
        binStarts,
        commandCounts: actualCounts,
        expectedObjectBins,
        objectBuckets: scenario.objectBuckets,
        bucketBases: scenario.bucketBases,
        bucketCapacities: scenario.bucketCounts,
        order,
      });
      depthBins.physicalBinSequenceCommitment = await createPhysicalDepthBinSequenceCommitment({
        actualIds: visibleIds,
        binCounts,
        binStarts,
        bucketBases: scenario.bucketBases,
        bucketCapacities: scenario.bucketCounts,
      });
      const representation = diagnostics();
      const representationPass = representation.meshCount === 1
        && representation.geometryIdentityCount === 1
        && representation.materialIdentityCount === 1
        && representation.bundleRecordCallbackCount === 1
        && representation.commandCount === scenario.bucketCount
        && representation.zeroFirstInstanceCount === scenario.bucketCount
        && representation.computeDispatchCount === 4;

      return {
        pass: membership.pass
          && membershipDigests.pass
          && commandValidation.pass
          && depthBins.pass
          && overflow === 0
          && representationPass,
        kind: `${id}-exact-membership-and-depth-order`,
        membership,
        membershipDigests,
        commandValidation,
        depthBins,
        overflow,
        representation,
      };
    },
  };
}

export function buildDepthBinnedFrontToBackStrategy(options) {
  return buildDepthBinnedFixedSliceStrategy(options, {
    id: 'fixed-slice-depth-front-to-back',
    order: DEPTH_ORDER_FRONT_TO_BACK,
  });
}

export function buildDepthBinnedReverseStrategy(options) {
  return buildDepthBinnedFixedSliceStrategy(options, {
    id: 'fixed-slice-depth-reverse',
    order: DEPTH_ORDER_REVERSE,
  });
}
