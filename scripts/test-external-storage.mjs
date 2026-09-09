import assert from 'node:assert/strict';
import test from 'node:test';

// Run the candidate's actual QUnit cases with Node's test runner, without a GPU.
globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
  INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
globalThis.QUnit = { module: (_name, body) => body(), test: (name, body) => test(name, () => body(assert)) };
await import('../.local-research/three-ort-gpu-buffer-current-dev-candidate/test/unit/src/renderers/webgpu/ExternalStorageBufferAttribute.tests.js');
