export const PINNED_V11_COMPUTE_NODE_FIELDS = Object.freeze([
  'initAll',
  'clearArgs',
  'clearVis',
  'selectPack',
  'capInstanceCount',
  'fillSortPairs',
  'applySortedPairs',
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
