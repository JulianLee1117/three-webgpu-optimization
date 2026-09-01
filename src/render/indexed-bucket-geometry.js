import {
  InstancedBufferGeometry,
  Uint32BufferAttribute,
} from 'three/webgpu';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

export function createIndexedBucketGeometry(
  source,
  bucketBase,
  instanceCount,
  { includeBucketBase = true } = {},
) {
  if (!source.index) throw new Error('The controlled benchmark requires indexed geometry.');
  if (typeof includeBucketBase !== 'boolean') {
    throw new TypeError('includeBucketBase must be a boolean.');
  }
  const geometry = new InstancedBufferGeometry();
  geometry.copy(source);
  geometry.instanceCount = instanceCount;
  if (includeBucketBase) {
    const vertexCount = geometry.getAttribute('position').count;
    geometry.setAttribute(
      'bucketBase',
      new Uint32BufferAttribute(new Uint32Array(vertexCount).fill(bucketBase), 1),
    );
  }
  return geometry;
}

export function createMergedIndexedBucketGeometry(
  sources,
  bucketBases,
  instanceCounts,
  { includeBucketBase = true } = {},
) {
  if (sources.length === 0
    || bucketBases.length !== sources.length
    || instanceCounts.length !== sources.length) {
    throw new RangeError('Merged indexed buckets require matched, nonempty inputs.');
  }
  if (typeof includeBucketBase !== 'boolean') {
    throw new TypeError('includeBucketBase must be a boolean.');
  }

  const bucketGeometries = sources.map((source, bucket) => createIndexedBucketGeometry(
    source,
    bucketBases[bucket],
    instanceCounts[bucket],
    { includeBucketBase },
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

function isInstancedAttribute(attribute) {
  return attribute?.isInstancedBufferAttribute === true
    || attribute?.data?.isInstancedInterleavedBuffer === true
    || Number.isInteger(attribute?.meshPerAttribute);
}

/**
 * Creates a second geometry shell over the exact same indexed vertex payload.
 * Attribute omission changes only the shell's vertex layout; retained index,
 * vertex, and morph attributes preserve object identity with the source.
 */
export function createSharedGeometryShell(
  source,
  { omitAttributes = [] } = {},
) {
  if (!source?.isBufferGeometry || !source.index) {
    throw new TypeError('Shared geometry shells require an indexed BufferGeometry.');
  }
  if (!Array.isArray(omitAttributes)
    || omitAttributes.some((name) => typeof name !== 'string' || name.length === 0)) {
    throw new TypeError('omitAttributes must be an array of nonempty attribute names.');
  }
  const omitted = new Set(omitAttributes);
  if (omitted.size !== omitAttributes.length) {
    throw new RangeError('omitAttributes must not contain duplicate names.');
  }

  const attributeNames = Object.keys(source.attributes);
  for (const name of omitted) {
    if (!attributeNames.includes(name)) {
      throw new RangeError(`Cannot omit missing geometry attribute ${name}.`);
    }
  }
  for (const name of attributeNames) {
    if (isInstancedAttribute(source.getAttribute(name))) {
      throw new Error(
        `Shared first-instance geometry cannot retain or omit instanced attribute ${name}.`,
      );
    }
  }
  for (const [name, attributes] of Object.entries(source.morphAttributes)) {
    for (const attribute of attributes) {
      if (isInstancedAttribute(attribute)) {
        throw new Error(
          `Shared first-instance geometry cannot use instanced morph attribute ${name}.`,
        );
      }
    }
  }

  const shell = new InstancedBufferGeometry();
  shell.name = source.name ? `${source.name}-shared-shell` : 'shared-indexed-geometry-shell';
  shell.setIndex(source.index);
  for (const name of attributeNames) {
    if (!omitted.has(name)) shell.setAttribute(name, source.getAttribute(name));
  }
  shell.morphAttributes = Object.fromEntries(
    Object.entries(source.morphAttributes).map(([name, attributes]) => [name, [...attributes]]),
  );
  shell.morphTargetsRelative = source.morphTargetsRelative;
  shell.instanceCount = source.instanceCount;
  shell.setDrawRange(source.drawRange.start, source.drawRange.count);
  for (const group of source.groups) {
    shell.addGroup(group.start, group.count, group.materialIndex);
  }
  shell.boundingBox = source.boundingBox?.clone() ?? null;
  shell.boundingSphere = source.boundingSphere?.clone() ?? null;
  return shell;
}
