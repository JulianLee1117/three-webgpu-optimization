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

export function createStorageTransformMaterial({
  matrixAttribute,
  objectCount,
  visibleIdsAttribute = null,
  visibleIdsCount = objectCount,
  visibleIdOffsetNode = null,
  color = 0x68a7f2,
}) {
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
  material.positionNode = Fn(() => {
    const sliceBase = attribute('bucketBase', 'uint');
    const sliceIndex = visibleIdOffsetNode === null
      ? sliceBase.add(instanceIndex)
      : visibleIdOffsetNode.add(sliceBase).add(instanceIndex);
    const objectId = visibleRead ? visibleRead.element(sliceIndex) : sliceIndex;
    const objectMatrix = matrixRead.element(objectId).toVar('objectMatrix');
    normalLocal.assign(transformNormal(normalLocal, objectMatrix));
    return objectMatrix.mul(vec4(positionGeometry, 1)).xyz;
  })();

  return material;
}
