import {
  Color,
  MeshStandardNodeMaterial,
} from 'three/webgpu';
import {
  Fn,
  attribute,
  instanceIndex,
  normalLocal,
  positionGeometry,
  storage,
  transformNormal,
  vec4,
} from 'three/tsl';

export const STORAGE_TRANSFORM_ADDRESS_MODES = Object.freeze({
  BUCKET_BASE: 'bucket-base',
  INDIRECT_FIRST_INSTANCE: 'indirect-first-instance',
});

const STORAGE_TRANSFORM_ADDRESS_MODE_VALUES = Object.freeze(
  Object.values(STORAGE_TRANSFORM_ADDRESS_MODES),
);

export function validateStorageTransformAddressMode(addressMode) {
  if (!STORAGE_TRANSFORM_ADDRESS_MODE_VALUES.includes(addressMode)) {
    throw new RangeError(
      `addressMode must be one of: ${STORAGE_TRANSFORM_ADDRESS_MODE_VALUES.join(', ')}.`,
    );
  }
  return addressMode;
}

export function createVisibleIdAddressNode({
  addressMode = STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  visibleIdOffsetNode = null,
} = {}) {
  validateStorageTransformAddressMode(addressMode);
  const localIndex = addressMode === STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE
    ? attribute('bucketBase', 'uint').add(instanceIndex)
    : instanceIndex;
  return visibleIdOffsetNode === null
    ? localIndex
    : visibleIdOffsetNode.add(localIndex);
}

export function createStorageTransformMaterial({
  matrixAttribute,
  objectCount,
  visibleIdsAttribute = null,
  visibleIdsCount = objectCount,
  visibleIdOffsetNode = null,
  addressMode = STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  color = 0x68a7f2,
}) {
  validateStorageTransformAddressMode(addressMode);
  const matrixRead = storage(matrixAttribute, 'mat4', objectCount).toReadOnly();
  if (visibleIdsAttribute
    && (!Number.isInteger(visibleIdsCount)
      || visibleIdsCount < objectCount
      || visibleIdsCount > visibleIdsAttribute.count)) {
    throw new RangeError(
      'visibleIdsCount must cover objectCount without exceeding the storage attribute.',
    );
  }
  if (!visibleIdsAttribute && visibleIdOffsetNode !== null) {
    throw new TypeError('visibleIdOffsetNode requires visibleIdsAttribute.');
  }
  const visibleRead = visibleIdsAttribute
    ? storage(visibleIdsAttribute, 'uint', visibleIdsCount).toReadOnly()
    : null;

  const material = new MeshStandardNodeMaterial();
  material.color = new Color(color);
  material.roughness = 0.68;
  material.metalness = 0.08;
  material.userData.storageTransformAddressMode = addressMode;
  material.positionNode = Fn(() => {
    const sliceIndex = createVisibleIdAddressNode({
      addressMode,
      visibleIdOffsetNode,
    });
    const objectId = visibleRead ? visibleRead.element(sliceIndex) : sliceIndex;
    const objectMatrix = matrixRead.element(objectId).toVar('objectMatrix');
    normalLocal.assign(transformNormal(normalLocal, objectMatrix));
    return objectMatrix.mul(vec4(positionGeometry, 1)).xyz;
  })();

  return material;
}
