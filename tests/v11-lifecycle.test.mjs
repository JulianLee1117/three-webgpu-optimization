import assert from 'node:assert/strict';
import test from 'node:test';
import {
  PINNED_V11_COMPUTE_NODE_FIELDS,
  PINNED_V11_STORAGE_ATTRIBUTE_FIELDS,
  disposeRetainedComputeNodes,
  retainPinnedV11ComputeNodes,
  retainPinnedV11StorageAttributes,
} from '../src/strategies/pinned-v11-compute-lifecycle.js';

function fakeComputeNode() {
  return {
    isComputeNode: true,
    disposeCalls: 0,
    dispose() { this.disposeCalls += 1; },
  };
}

test('pinned v0.11 lifecycle retains every guarded compute phase', () => {
  const culler = Object.fromEntries(
    PINNED_V11_COMPUTE_NODE_FIELDS.map((field) => [field, fakeComputeNode()]),
  );
  const retained = retainPinnedV11ComputeNodes(culler, 3);
  assert.deepEqual([...retained.byField.keys()], [...PINNED_V11_COMPUTE_NODE_FIELDS]);
  assert.equal(retained.nodes.length, PINNED_V11_COMPUTE_NODE_FIELDS.length);
});

test('pinned v0.11 lifecycle fails closed when the runtime surface changes', () => {
  const culler = Object.fromEntries(
    PINNED_V11_COMPUTE_NODE_FIELDS.map((field) => [field, fakeComputeNode()]),
  );
  culler.selectPack = { isComputeNode: false, dispose() {} };
  assert.throws(
    () => retainPinnedV11ComputeNodes(culler, 7),
    /disposable ComputeNode selectPack at bucket 7/,
  );
});

test('retained package compute nodes are deduplicated and disposed exactly once', () => {
  const shared = fakeComputeNode();
  const first = [shared, fakeComputeNode()];
  const second = [shared, fakeComputeNode()];
  assert.equal(disposeRetainedComputeNodes([first, second]), 3);
  for (const node of new Set([...first, ...second])) assert.equal(node.disposeCalls, 1);
});

test('pinned v0.11 lifecycle retains every package-owned storage attribute', () => {
  const culler = Object.fromEntries(PINNED_V11_STORAGE_ATTRIBUTE_FIELDS.map((field) => (
    [field, { isBufferAttribute: true, field }]
  )));
  culler.sortKeysIA = { value: culler.sortKeysIA };
  culler.sortValuesIA = { attribute: culler.sortValuesIA };
  const attributes = retainPinnedV11StorageAttributes(culler, 3);
  assert.deepEqual(attributes.map((attribute) => attribute.field), PINNED_V11_STORAGE_ATTRIBUTE_FIELDS);
});

test('pinned v0.11 storage lifecycle fails closed when the runtime surface changes', () => {
  const culler = Object.fromEntries(PINNED_V11_STORAGE_ATTRIBUTE_FIELDS.map((field) => (
    [field, { isBufferAttribute: true, field }]
  )));
  culler.outVisSSBO = null;
  assert.throws(
    () => retainPinnedV11StorageAttributes(culler, 8),
    /requires BufferAttribute outVisSSBO at bucket 8/,
  );
});
