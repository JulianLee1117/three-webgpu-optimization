import {
  InstancedBufferGeometry,
  Uint32BufferAttribute,
} from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

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

export function createMergedIndexedBucketGeometry(sources, bucketBases, instanceCounts) {
  if (sources.length === 0
    || bucketBases.length !== sources.length
    || instanceCounts.length !== sources.length) {
    throw new RangeError('Merged indexed buckets require matched, nonempty inputs.');
  }

  const bucketGeometries = sources.map((source, bucket) => createIndexedBucketGeometry(
    source,
    bucketBases[bucket],
    instanceCounts[bucket],
  ));
  const firstIndexes = new Uint32Array(sources.length);
  let indexCursor = 0;
  for (let bucket = 0; bucket < sources.length; bucket += 1) {
    firstIndexes[bucket] = indexCursor;
    indexCursor += sources[bucket].index.count;
  }

  let mergedSource;
  try {
    mergedSource = mergeGeometries(bucketGeometries, false);
    if (!mergedSource?.index) throw new Error('Indexed fixture merge failed.');
    const geometry = new InstancedBufferGeometry().copy(mergedSource);
    geometry.name = 'fixed-slice-merged-indexed-fixtures';
    geometry.instanceCount = Math.max(...instanceCounts);
    return { geometry, firstIndexes };
  } finally {
    bucketGeometries.forEach((geometry) => geometry.dispose());
    mergedSource?.dispose();
  }
}
