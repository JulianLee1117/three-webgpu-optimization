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
import {
  STORAGE_TRANSFORM_ADDRESS_MODES,
  createStorageTransformMaterial,
  validateStorageTransformAddressMode,
} from '../materials/storage-transform.js';
import {
  createMergedIndexedBucketGeometry,
  createSharedGeometryShell,
} from '../render/indexed-bucket-geometry.js';
import { compareMembership, validateIndexedCommands } from '../validation/membership.js';
import {
  createMembershipDigestEvidence,
  sha256CanonicalUint32,
} from '../validation/membership-digests.js';

export const FIXED_SLICE_LANES = Object.freeze(['portable', 'feature']);

const [PORTABLE_LANE, FEATURE_LANE] = FIXED_SLICE_LANES;

export const FIXED_SLICE_ADDRESS_MODE_BY_LANE = Object.freeze({
  [PORTABLE_LANE]: STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  [FEATURE_LANE]: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
});

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

function validateSharedInputs({ scenario, sourceGeometries }) {
  if (!Number.isInteger(scenario?.objectCount) || scenario.objectCount <= 0) {
    throw new RangeError('scenario.objectCount must be a positive integer.');
  }
  if (!Number.isInteger(scenario?.bucketCount) || scenario.bucketCount <= 0) {
    throw new RangeError('scenario.bucketCount must be a positive integer.');
  }
  if (!Array.isArray(sourceGeometries)
    || sourceGeometries.length !== scenario.bucketCount) {
    throw new RangeError('sourceGeometries length must equal scenario.bucketCount.');
  }
  const typedInputs = [
    ['matrices', Float32Array, scenario.objectCount * 16],
    ['bounds', Float32Array, scenario.objectCount * 4],
    ['objectBuckets', Uint32Array, scenario.objectCount],
    ['bucketBases', Uint32Array, scenario.bucketCount],
    ['bucketCounts', Uint32Array, scenario.bucketCount],
    ['cullOrder', Uint32Array, scenario.objectCount],
  ];
  for (const [name, Type, length] of typedInputs) {
    if (!(scenario[name] instanceof Type) || scenario[name].length !== length) {
      throw new RangeError(`scenario.${name} must be a ${Type.name} of length ${length}.`);
    }
  }
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    if (!sourceGeometries[bucket]?.index) {
      throw new Error(`Geometry bucket ${bucket} must be indexed.`);
    }
  }
}

function validateAddressModes(addressModes) {
  if (!Array.isArray(addressModes) || addressModes.length === 0) {
    throw new RangeError('addressModes must be a nonempty array.');
  }
  for (const addressMode of addressModes) validateStorageTransformAddressMode(addressMode);
  if (new Set(addressModes).size !== addressModes.length) {
    throw new RangeError('addressModes must not contain duplicates.');
  }
}

export function validateFixedSliceLane(lane, label = 'lane') {
  if (!FIXED_SLICE_LANES.includes(lane)) {
    throw new RangeError(`${label} must be portable or feature.`);
  }
  return lane;
}

export function validateFixedSliceLanePhysicalOrder(value, label = 'lanePhysicalOrder') {
  if (!Array.isArray(value)
    || value.length !== FIXED_SLICE_LANES.length
    || new Set(value).size !== FIXED_SLICE_LANES.length
    || FIXED_SLICE_LANES.some((lane) => !value.includes(lane))) {
    throw new RangeError(`${label} must be the exact portable/feature lane permutation.`);
  }
  return value;
}

/** Allocates all common cull resources and the common merged geometry payload. */
export function createFixedSliceSharedResources(
  { scenario, sourceGeometries },
  { addressModes = [STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE] } = {},
) {
  validateSharedInputs({ scenario, sourceGeometries });
  validateAddressModes(addressModes);

  const includesPortable = addressModes.includes(STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE);
  const merged = createMergedIndexedBucketGeometry(
    sourceGeometries,
    scenario.bucketBases,
    scenario.bucketCounts,
    { includeBucketBase: includesPortable },
  );
  const geometriesByAddressMode = {};
  if (includesPortable) {
    merged.geometry.name = 'fixed-slice-portable-merged-indexed-fixtures';
    geometriesByAddressMode[STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE] = merged.geometry;
  }
  if (addressModes.includes(STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE)) {
    if (includesPortable) {
      const featureGeometry = createSharedGeometryShell(merged.geometry, {
        omitAttributes: ['bucketBase'],
      });
      featureGeometry.name = 'fixed-slice-feature-merged-indexed-fixtures';
      geometriesByAddressMode[STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE]
        = featureGeometry;
    } else {
      merged.geometry.name = 'fixed-slice-feature-merged-indexed-fixtures';
      geometriesByAddressMode[STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE]
        = merged.geometry;
    }
  }

  const commandTemplate = createIndexedIndirectCommands(
    sourceGeometries,
    scenario.bucketCounts,
    null,
    merged.firstIndexes,
  );
  const attributes = Object.freeze({
    matrix: new StorageBufferAttribute(scenario.matrices, 16),
    bounds: new StorageBufferAttribute(scenario.bounds, 4),
    objectBucket: new StorageBufferAttribute(scenario.objectBuckets, 1),
    bucketBase: new StorageBufferAttribute(scenario.bucketBases, 1),
    bucketCapacity: new StorageBufferAttribute(commandTemplate.capacities, 1),
    cullOrder: new StorageBufferAttribute(scenario.cullOrder, 1),
    visibleIds: new StorageBufferAttribute(new Uint32Array(scenario.objectCount), 1),
    overflow: new StorageBufferAttribute(new Uint32Array(1), 1),
  });
  const planeState = createFrustumPlaneState();
  const storageNodes = Object.freeze({
    boundsRead: storage(attributes.bounds, 'vec4', scenario.objectCount).toReadOnly(),
    bucketRead: storage(attributes.objectBucket, 'uint', scenario.objectCount).toReadOnly(),
    baseRead: storage(attributes.bucketBase, 'uint', scenario.bucketCount).toReadOnly(),
    capacityRead: storage(attributes.bucketCapacity, 'uint', scenario.bucketCount).toReadOnly(),
    orderRead: storage(attributes.cullOrder, 'uint', scenario.objectCount).toReadOnly(),
    visibleWrite: storage(attributes.visibleIds, 'uint', scenario.objectCount),
    overflowAtomic: storage(attributes.overflow, 'uint', 1).toAtomic(),
  });
  const drawStruct = struct({
    indexCount: 'uint',
    instanceCount: { type: 'uint', atomic: true },
    firstIndex: 'uint',
    baseVertex: 'int',
    firstInstance: 'uint',
  }, 'FixedSliceIndexedDraw');

  return {
    kind: 'fixed-slice-shared-resources',
    scenario,
    sourceGeometries,
    firstIndexes: merged.firstIndexes,
    commandCapacities: commandTemplate.capacities,
    commandRecordCount: commandTemplate.recordCount,
    geometriesByAddressMode,
    ownedGeometries: Object.values(geometriesByAddressMode),
    attributes,
    storageAttributes: Object.values(attributes),
    storageNodes,
    drawStruct,
    planeState,
  };
}

function createFixedSliceComputeNodes(shared, indirectAttribute, commandRecordCount) {
  const { scenario, storageNodes, planeState, drawStruct } = shared;
  const drawStorage = storage(indirectAttribute, drawStruct, commandRecordCount);
  const reset = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.bucketCount)), () => {
      atomicStore(drawStorage.element(instanceIndex).get('instanceCount'), uint(0));
      If(instanceIndex.equal(uint(0)), () => {
        atomicStore(storageNodes.overflowAtomic.element(uint(0)), uint(0));
      });
    });
  })().compute(scenario.bucketCount);

  const cull = Fn(() => {
    If(instanceIndex.lessThan(uint(scenario.objectCount)), () => {
      const objectId = storageNodes.orderRead.element(instanceIndex).toVar('cullObjectId');
      const sphere = storageNodes.boundsRead.element(objectId).toVar('worldSphere');
      If(sphereInsideNode(sphere, planeState.uniforms), () => {
        const bucket = storageNodes.bucketRead.element(objectId).toVar('objectBucket');
        const draw = drawStorage.element(bucket);
        const slot = atomicAdd(draw.get('instanceCount'), uint(1)).toVar('visibleSlot');
        If(slot.lessThan(storageNodes.capacityRead.element(bucket)), () => {
          storageNodes.visibleWrite
            .element(storageNodes.baseRead.element(bucket).add(slot))
            .assign(objectId);
        }).Else(() => {
          atomicStore(storageNodes.overflowAtomic.element(uint(0)), uint(1));
        });
      });
    });
  })().compute(scenario.objectCount);
  return [reset, cull];
}

/** Creates the only lane-specific state used by paired and standalone paths. */
export function createFixedSliceLane(
  shared,
  {
    lane = PORTABLE_LANE,
    id = lane === FEATURE_LANE
      ? 'fixed-slice-indirect-first-instance'
      : 'fixed-slice',
    perBucketRenderObjects = false,
  } = {},
) {
  validateFixedSliceLane(lane);
  if (!shared || shared.kind !== 'fixed-slice-shared-resources') {
    throw new TypeError('createFixedSliceLane requires fixed-slice shared resources.');
  }
  const { scenario, sourceGeometries, firstIndexes } = shared;
  const addressMode = FIXED_SLICE_ADDRESS_MODE_BY_LANE[lane];
  const geometry = shared.geometriesByAddressMode[addressMode];
  if (!geometry) {
    throw new Error(`Shared fixed-slice resources do not include the ${lane} geometry.`);
  }
  const indirectFirstInstance = lane === FEATURE_LANE;
  const commandLayout = createIndexedIndirectCommands(
    sourceGeometries,
    scenario.bucketCounts,
    null,
    firstIndexes,
    indirectFirstInstance ? scenario.bucketBases : null,
  );
  const indirectAttribute = new IndirectStorageBufferAttribute(commandLayout.commands, 5);
  const computeNodes = createFixedSliceComputeNodes(
    shared,
    indirectAttribute,
    commandLayout.recordCount,
  );
  const material = createStorageTransformMaterial({
    matrixAttribute: shared.attributes.matrix,
    objectCount: scenario.objectCount,
    visibleIdsAttribute: shared.attributes.visibleIds,
    addressMode,
  });
  const root = new BundleGroup();
  freezeStaticTransform(root);
  root.name = perBucketRenderObjects
    ? 'fixed-slice-per-bucket-merged-indexed-indirect-bundle'
    : `${id}-merged-indexed-indirect-bundle`;

  let bundleRecordCallbackCount = 0;
  const meshes = [];
  if (perBucketRenderObjects) {
    geometry.setIndirect(indirectAttribute, 0);
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const indirectOffset = commandLayout.offsets[bucket];
      const mesh = new Mesh(geometry, material);
      freezeStaticTransform(mesh);
      mesh.frustumCulled = false;
      mesh.onBeforeRender = (activeRenderer) => {
        geometry.setIndirect(indirectAttribute, indirectOffset);
        if (activeRenderer?._currentRenderBundle !== null
          && activeRenderer?._currentRenderBundle !== undefined) {
          bundleRecordCallbackCount += 1;
        }
      };
      meshes.push(mesh);
      root.add(mesh);
    }
  } else {
    geometry.setIndirect(indirectAttribute, Array.from(commandLayout.offsets));
    const mesh = new Mesh(geometry, material);
    freezeStaticTransform(mesh);
    mesh.frustumCulled = false;
    mesh.onBeforeRender = (activeRenderer) => {
      if (activeRenderer?._currentRenderBundle !== null
        && activeRenderer?._currentRenderBundle !== undefined) {
        bundleRecordCallbackCount += 1;
      }
    };
    meshes.push(mesh);
    root.add(mesh);
  }

  return {
    kind: 'fixed-slice-lane',
    id,
    lane,
    addressMode,
    root,
    bundle: root,
    meshes,
    geometry,
    material,
    indirectAttribute,
    commandLayout,
    computeNodes,
    get bundleRecordCallbackCount() {
      return bundleRecordCallbackCount;
    },
    commandBufferCommitment() {
      return {
        lane,
        attributeId: indirectAttribute.id,
        attributeVersion: indirectAttribute.version,
        byteOffset: 0,
        byteLength: commandLayout.commands.byteLength,
        recordCount: commandLayout.recordCount,
        drawCommandCount: scenario.bucketCount,
        firstOffset: commandLayout.offsets[0],
        allOffsets: Array.from(commandLayout.offsets),
      };
    },
  };
}

function expectedCountsForIds(scenario, expectedIds) {
  if (!(expectedIds instanceof Uint32Array)) {
    throw new TypeError('expectedIds must be a Uint32Array.');
  }
  const counts = new Uint32Array(scenario.bucketCount);
  for (const objectId of expectedIds) {
    if (objectId >= scenario.objectCount) {
      throw new RangeError(`Expected visible object ID ${objectId} is out of range.`);
    }
    counts[scenario.objectBuckets[objectId]] += 1;
  }
  return counts;
}

export async function readFixedSliceLaneSnapshot(renderer, shared, lane) {
  if (typeof renderer?.getArrayBufferAsync !== 'function') {
    throw new TypeError('Fixed-slice validation requires renderer.getArrayBufferAsync().');
  }
  const [commandBuffer, visibleBuffer, overflowBuffer] = await Promise.all([
    renderer.getArrayBufferAsync(lane.indirectAttribute),
    renderer.getArrayBufferAsync(shared.attributes.visibleIds),
    renderer.getArrayBufferAsync(shared.attributes.overflow),
  ]);
  return {
    commands: asUint32(commandBuffer).slice(),
    visibleIds: asUint32(visibleBuffer).slice(),
    overflow: asUint32(overflowBuffer)[0],
  };
}

export async function validateFixedSliceLaneSnapshot({
  shared,
  lane,
  expectedIds,
  snapshot,
}) {
  if (!shared || shared.kind !== 'fixed-slice-shared-resources') {
    throw new TypeError('validateFixedSliceLaneSnapshot requires shared resources.');
  }
  if (!lane || lane.kind !== 'fixed-slice-lane') {
    throw new TypeError('validateFixedSliceLaneSnapshot requires a fixed-slice lane.');
  }
  if (!(snapshot?.commands instanceof Uint32Array)
    || !(snapshot.visibleIds instanceof Uint32Array)) {
    throw new TypeError('Fixed-slice validation snapshot must contain Uint32Array data.');
  }
  const { scenario, sourceGeometries, firstIndexes } = shared;
  const expectedCounts = expectedCountsForIds(scenario, expectedIds);
  const actualCounts = new Uint32Array(scenario.bucketCount);
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    actualCounts[bucket] = snapshot.commands[bucket * 5 + 1];
  }
  const membership = compareMembership({
    expectedIds,
    actualIds: snapshot.visibleIds,
    actualCounts,
    objectBuckets: scenario.objectBuckets,
    bucketBases: scenario.bucketBases,
    capacities: scenario.bucketCounts,
    objectCount: scenario.objectCount,
  });
  const commandValidation = validateIndexedCommands({
    commands: snapshot.commands,
    geometries: sourceGeometries,
    expectedCounts,
    expectedFirstIndexes: firstIndexes,
    expectedFirstInstances: lane.lane === FEATURE_LANE ? scenario.bucketBases : null,
  });
  const membershipDigests = await createMembershipDigestEvidence({
    expectedIds,
    actualIds: snapshot.visibleIds,
    actualCounts,
    objectBuckets: scenario.objectBuckets,
    bucketBases: scenario.bucketBases,
    capacities: scenario.bucketCounts,
  });
  const [commandSha256, survivorSha256] = await Promise.all([
    sha256CanonicalUint32(snapshot.commands),
    sha256CanonicalUint32(snapshot.visibleIds),
  ]);
  return {
    pass: membership.pass
      && membershipDigests.pass
      && commandValidation.pass
      && snapshot.overflow === 0,
    kind: `${lane.id}-exact-membership`,
    lane: lane.lane,
    membership,
    membershipDigests,
    commandValidation,
    overflow: snapshot.overflow,
    actualCounts,
    commandSha256,
    survivorSha256,
  };
}

function buildFixedSliceRepresentation(options, {
  id,
  lane = PORTABLE_LANE,
  perBucketRenderObjects,
}) {
  const addressMode = FIXED_SLICE_ADDRESS_MODE_BY_LANE[lane];
  const shared = createFixedSliceSharedResources(options, { addressModes: [addressMode] });
  const laneState = createFixedSliceLane(shared, {
    id,
    lane,
    perBucketRenderObjects,
  });
  const { scenario } = shared;

  const diagnostics = () => {
    if (perBucketRenderObjects) return {
      kind: 'shared-merged-geometry-per-bucket-render-objects',
      bundleRecordCallbackCount: laneState.bundleRecordCallbackCount,
      geometryIdentityCount: new Set(laneState.meshes.map((child) => child.geometry)).size,
      materialIdentityCount: new Set(laneState.meshes.map((child) => child.material)).size,
      meshCount: laneState.meshes.length,
      geometryInstanceCount: laneState.geometry.instanceCount,
    };
    if (lane === FEATURE_LANE) return {
      kind: 'single-merged-geometry-indirect-first-instance',
      addressMode,
      hasBucketBaseAttribute: laneState.geometry.getAttribute('bucketBase') !== undefined,
      nonzeroFirstInstanceCount: Array.from(
        { length: scenario.bucketCount },
        (_, bucket) => laneState.commandLayout.commands[bucket * 5 + 4],
      ).filter((value) => value !== 0).length,
      bundleRecordCallbackCount: laneState.bundleRecordCallbackCount,
      geometryIdentityCount: 1,
      materialIdentityCount: 1,
      meshCount: laneState.meshes.length,
      geometryInstanceCount: laneState.geometry.instanceCount,
    };
    return null;
  };

  const lifecycleDiagnostics = () => (perBucketRenderObjects
    ? null
    : {
      kind: 'single-merged-geometry-atomic-fixed-slice-lifecycle',
      bundleGroupStatic: laneState.root.static === true,
      bundleRecordCallbackCount: laneState.bundleRecordCallbackCount,
      geometryIdentityCount: new Set(laneState.meshes.map((child) => child.geometry)).size,
      materialIdentityCount: new Set(laneState.meshes.map((child) => child.material)).size,
      meshCount: laneState.meshes.length,
    });

  return {
    id,
    root: laneState.root,
    geometries: shared.ownedGeometries,
    materials: [laneState.material],
    parityResources: Object.freeze({
      matrixAttribute: shared.attributes.matrix,
      visibleIdsAttribute: shared.attributes.visibleIds,
      objectCount: scenario.objectCount,
      addressMode,
    }),
    storageAttributes: [...shared.storageAttributes, laneState.indirectAttribute],
    computeNodes: laneState.computeNodes,
    usesCompute: true,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: perBucketRenderObjects ? scenario.bucketCount : 1,
    configuredComputeDispatches: 2,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    sharedResources: shared,
    laneState,
    diagnostics,
    lifecycleDiagnostics,
    update(camera, renderer) {
      updateFrustumPlaneState(shared.planeState, camera, renderer);
    },
    submitCompute(renderer) {
      renderer.compute(laneState.computeNodes);
    },
    async validate(renderer, expectedIds) {
      const snapshot = await readFixedSliceLaneSnapshot(renderer, shared, laneState);
      const validation = await validateFixedSliceLaneSnapshot({
        shared,
        lane: laneState,
        expectedIds,
        snapshot,
      });
      const representation = diagnostics();
      const representationPass = !perBucketRenderObjects || (
        representation?.kind === 'shared-merged-geometry-per-bucket-render-objects'
        && representation.bundleRecordCallbackCount === scenario.bucketCount
        && representation.geometryIdentityCount === 1
        && representation.materialIdentityCount === 1
        && representation.meshCount === scenario.bucketCount
        && representation.geometryInstanceCount
          === Math.ceil(scenario.objectCount / scenario.bucketCount)
      );
      return {
        ...validation,
        pass: validation.pass && representationPass,
        representation,
      };
    },
  };
}

export function buildFixedSliceStrategy(options) {
  return buildFixedSliceRepresentation(options, {
    id: 'fixed-slice',
    lane: PORTABLE_LANE,
    perBucketRenderObjects: false,
  });
}

export function buildFixedSlicePerBucketStrategy(options) {
  return buildFixedSliceRepresentation(options, {
    id: 'fixed-slice-per-bucket',
    lane: PORTABLE_LANE,
    perBucketRenderObjects: true,
  });
}

export function buildIndirectFirstInstanceStrategy(options) {
  if (options?.renderer?.hasFeature?.('indirect-first-instance') !== true) {
    throw new Error(
      'fixed-slice-indirect-first-instance requires the indirect-first-instance WebGPU feature.',
    );
  }
  return buildFixedSliceRepresentation(options, {
    id: 'fixed-slice-indirect-first-instance',
    lane: FEATURE_LANE,
    perBucketRenderObjects: false,
  });
}

/** Selects one standalone lane without constructing the other lane. */
export function selectFixedSliceDeployment({ renderer, featureAvailable } = {}) {
  if (featureAvailable !== undefined && typeof featureAvailable !== 'boolean') {
    throw new TypeError('featureAvailable must be a boolean when supplied.');
  }
  const available = featureAvailable
    ?? (renderer?.hasFeature?.('indirect-first-instance') === true);
  return Object.freeze({
    lane: available ? FEATURE_LANE : PORTABLE_LANE,
    strategyId: available
      ? 'fixed-slice-indirect-first-instance'
      : 'fixed-slice',
    featureAvailable: available,
  });
}

export function buildFixedSliceDeploymentStrategy(options, { featureAvailable } = {}) {
  const selection = selectFixedSliceDeployment({
    renderer: options?.renderer,
    featureAvailable,
  });
  return selection.lane === FEATURE_LANE
    ? buildIndirectFirstInstanceStrategy(options)
    : buildFixedSliceStrategy(options);
}
