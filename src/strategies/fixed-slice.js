import {
  BundleGroup,
  IndirectStorageBufferAttribute,
  Mesh,
  StorageBufferAttribute,
} from 'three/webgpu';
import {
  Fn,
  If,
  atomicAdd,
  atomicStore,
  dot,
  float,
  instanceIndex,
  storage,
  struct,
  uint,
} from 'three/tsl';
import { createFrustumPlaneState, updateFrustumPlaneState } from '../culling/frustum-planes.js';
import { createIndexedIndirectCommands } from '../culling/indexed-command-layout.js';
import { createStorageTransformMaterial } from '../materials/storage-transform.js';
import { createMergedIndexedBucketGeometry } from '../render/indexed-bucket-geometry.js';
import { compareMembership, validateIndexedCommands } from '../validation/membership.js';
import { createMembershipDigestEvidence } from '../validation/membership-digests.js';

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
  return new Uint32Array(buffer instanceof ArrayBuffer ? buffer : buffer.buffer);
}

export function buildFixedSliceStrategy({ scenario, sourceGeometries }) {
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
  const overflowAttribute = new StorageBufferAttribute(new Uint32Array(1), 1);
  const indirectAttribute = new IndirectStorageBufferAttribute(commandLayout.commands, 5);
  const planeState = createFrustumPlaneState();

  const boundsRead = storage(boundsAttribute, 'vec4', scenario.objectCount).toReadOnly();
  const bucketRead = storage(bucketAttribute, 'uint', scenario.objectCount).toReadOnly();
  const baseRead = storage(baseAttribute, 'uint', scenario.bucketCount).toReadOnly();
  const capacityRead = storage(capacityAttribute, 'uint', scenario.bucketCount).toReadOnly();
  const orderRead = storage(cullOrderAttribute, 'uint', scenario.objectCount).toReadOnly();
  const visibleWrite = storage(visibleIdsAttribute, 'uint', scenario.objectCount);
  const overflowAtomic = storage(overflowAttribute, 'uint', 1).toAtomic();
  const drawStruct = struct({
    indexCount: 'uint',
    instanceCount: { type: 'uint', atomic: true },
    firstIndex: 'uint',
    baseVertex: 'int',
    firstInstance: 'uint',
  }, 'FixedSliceIndexedDraw');
  const drawStorage = storage(indirectAttribute, drawStruct, commandLayout.recordCount);

  const reset = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.bucketCount)), () => {
      atomicStore(drawStorage.element(instanceIndex).get('instanceCount'), uint(0));
      If(instanceIndex.equal(uint(0)), () => {
        atomicStore(overflowAtomic.element(uint(0)), uint(0));
      });
    });
  })().compute(scenario.bucketCount);

  const cull = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.objectCount)), () => {
      const objectId = orderRead.element(instanceIndex).toVar('cullObjectId');
      const sphere = boundsRead.element(objectId).toVar('worldSphere');
      If(sphereInsideNode(sphere, planeState.uniforms), () => {
        const bucket = bucketRead.element(objectId).toVar('objectBucket');
        const draw = drawStorage.element(bucket);
        const slot = atomicAdd(draw.get('instanceCount'), uint(1)).toVar('visibleSlot');
        If(slot.lessThan(capacityRead.element(bucket)), () => {
          visibleWrite.element(baseRead.element(bucket).add(slot)).assign(objectId);
        }).Else(() => {
          atomicStore(overflowAtomic.element(uint(0)), uint(1));
        });
      });
    });
  })().compute(scenario.objectCount);

  const material = createStorageTransformMaterial({
    matrixAttribute,
    objectCount: scenario.objectCount,
    visibleIdsAttribute,
  });
  const root = new BundleGroup();
  root.name = 'fixed-slice-merged-indexed-indirect-bundle';
  geometry.setIndirect(indirectAttribute, Array.from(commandLayout.offsets));
  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  root.add(mesh);

  return {
    id: 'fixed-slice',
    root,
    geometries: [geometry],
    materials: [material],
    storageAttributes: [
      matrixAttribute,
      boundsAttribute,
      bucketAttribute,
      baseAttribute,
      capacityAttribute,
      cullOrderAttribute,
      visibleIdsAttribute,
      overflowAttribute,
      indirectAttribute,
    ],
    computeNodes: [reset, cull],
    usesCompute: true,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 2,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    update(camera, renderer) {
      updateFrustumPlaneState(planeState, camera, renderer);
    },
    submitCompute(renderer) {
      renderer.compute([reset, cull]);
    },
    async validate(renderer, expectedIds) {
      const [commandBuffer, visibleBuffer, overflowBuffer] = await Promise.all([
        renderer.getArrayBufferAsync(indirectAttribute),
        renderer.getArrayBufferAsync(visibleIdsAttribute),
        renderer.getArrayBufferAsync(overflowAttribute),
      ]);
      const commands = asUint32(commandBuffer);
      const visibleIds = asUint32(visibleBuffer);
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
      return {
        pass: membership.pass
          && membershipDigests.pass
          && commandValidation.pass
          && overflow === 0,
        kind: 'fixed-slice-exact-membership',
        membership,
        membershipDigests,
        commandValidation,
        overflow,
      };
    },
  };
}
