export const PINNED_V11_COMPUTE_NODE_FIELDS = Object.freeze([
  'initAll',
  'clearArgs',
  'clearVis',
  'selectPack',
  'capInstanceCount',
  'fillSortPairs',
  'applySortedPairs',
]);

export const PINNED_V11_STORAGE_ATTRIBUTE_FIELDS = Object.freeze([
  'outIdSSBO',
  'outVisSSBO',
  'indirect',
  'sortKeysIA',
  'sortValuesIA',
]);

export function retainPinnedV11ComputeNodes(culler, bucket) {
  const byField = new Map();
  for (const field of PINNED_V11_COMPUTE_NODE_FIELDS) {
    const node = culler?.[field];
    if (node?.isComputeNode !== true || typeof node.dispose !== 'function') {
      throw new TypeError(
        `three-blocks@0.11.0 requires disposable ComputeNode ${field} at bucket ${bucket}.`,
      );
    }
    byField.set(field, node);
  }
  return {
    byField,
    nodes: [...byField.values()],
  };
}

export function disposeRetainedComputeNodes(retainedCollections) {
  const uniqueNodes = new Set();
  for (const collection of retainedCollections) {
    for (const node of collection ?? []) uniqueNodes.add(node);
  }
  for (const node of uniqueNodes) node.dispose();
  return uniqueNodes.size;
}

export function retainPinnedV11StorageAttributes(culler, bucket) {
  const attributes = PINNED_V11_STORAGE_ATTRIBUTE_FIELDS.map((field) => {
    const value = culler?.[field];
    const attribute = value?.isBufferAttribute === true
      ? value
      : value?.value?.isBufferAttribute === true
        ? value.value
        : value?.attribute?.isBufferAttribute === true
          ? value.attribute
          : null;
    if (attribute?.isBufferAttribute !== true) {
      throw new TypeError(
        `three-blocks@0.11.0 requires BufferAttribute ${field} at bucket ${bucket}.`,
      );
    }
    return attribute;
  });
  if (culler.boundingSpheresSSBO?.isBufferAttribute === true) {
    attributes.push(culler.boundingSpheresSSBO);
  }
  return attributes;
}
