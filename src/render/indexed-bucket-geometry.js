import {
  InstancedBufferGeometry,
  Uint32BufferAttribute,
} from 'three/webgpu';

export function createIndexedBucketGeometry(source, bucketBase, instanceCount) {
  if (!source.index) throw new Error('The controlled benchmark requires indexed geometry.');
  const geometry = new InstancedBufferGeometry();
  geometry.copy(source);
  geometry.instanceCount = instanceCount;
  const vertexCount = geometry.getAttribute('position').count;
  geometry.setAttribute(
    'bucketBase',
    new Uint32BufferAttribute(new Uint32Array(vertexCount).fill(bucketBase), 1),
  );
  return geometry;
}
