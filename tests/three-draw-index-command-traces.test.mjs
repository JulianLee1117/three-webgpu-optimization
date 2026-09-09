import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { after, before, test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { buildThreeDrawIndexOverlay } from '../scripts/build-three-draw-index-overlay.mjs';

let temporaryRoot;
let WebGPUBackend;

function pipeline(value) {
  return { method: 'setPipeline', value };
}

function indexBuffer(buffer, format) {
  return { method: 'setIndexBuffer', buffer, format };
}

function immediate(value) {
  return {
    method: 'setImmediates',
    rangeOffset: 0,
    sourceType: 'Uint32Array',
    sourceElementCount: 1,
    dataOffset: 0,
    dataSize: 1,
    selectedValues: [value],
  };
}

function draw(method, ...args) {
  return { method, args };
}

function createEncoder({ supportsImmediates = true } = {}) {
  const events = [];
  const encoder = {
    events,
    setPipeline(value) {
      events.push(pipeline(value));
    },
    setIndexBuffer(buffer, format) {
      events.push(indexBuffer(buffer, format));
    },
    setVertexBuffer(...args) {
      events.push(draw('setVertexBuffer', ...args));
    },
    setBindGroup(...args) {
      events.push(draw('setBindGroup', ...args));
    },
    draw(...args) {
      events.push(draw('draw', ...args));
    },
    drawIndexed(...args) {
      events.push(draw('drawIndexed', ...args));
    },
    drawIndirect(...args) {
      events.push(draw('drawIndirect', ...args));
    },
    drawIndexedIndirect(...args) {
      events.push(draw('drawIndexedIndirect', ...args));
    },
  };
  if (supportsImmediates) {
    encoder.setImmediates = (
      rangeOffset,
      data,
      dataOffset = 0,
      dataSize = data.length - dataOffset,
    ) => {
      events.push({
        method: 'setImmediates',
        rangeOffset,
        sourceType: data.constructor.name,
        sourceElementCount: data.length,
        dataOffset,
        dataSize,
        // WebGPU copies the selected bytes during this call. The recorder must
        // do the same instead of retaining Three's reused typed array.
        selectedValues: Array.from(data.slice(dataOffset, dataOffset + dataSize)),
      });
    };
  }
  return encoder;
}

function freshCurrentSets() {
  return {
    pipeline: null,
    bindingGroups: [],
    attributes: [],
    index: null,
  };
}

function runDraw(backend, {
  encoder = createEncoder(),
  currentSets = freshCurrentSets(),
  immediateSize = 4,
  includeImmediateSize = true,
  pipelineGPU = immediateSize === 4 ? 'P4' : 'P0',
  indexed = false,
  uint32Index = false,
  indirectOffset = null,
  batched = null,
  vertexCount = 12,
  instanceCount = 1,
  firstVertex = 5,
} = {}) {
  const index = indexed
    ? { array: uint32Index ? new Uint32Array(0) : new Uint16Array(0) }
    : null;
  const indirect = indirectOffset === null ? null : {};
  if (index !== null) backend.set(index, { buffer: 'IB' });
  if (indirect !== null) backend.set(indirect, { buffer: 'B' });

  const object = batched === null
    ? {}
    : {
      isBatchedMesh: true,
      _multiDrawStarts: batched.starts,
      _multiDrawCounts: batched.counts,
      _multiDrawCount: batched.drawCount,
      _multiDrawBytesPerElement: batched.bytesPerElement ?? 1,
    };
  const infoUpdates = [];
  const renderObject = {
    object,
    material: {},
    context: { stencil: false },
    getIndex: () => index,
    getIndirect: () => indirect,
    getIndirectOffset: () => indirectOffset,
  };
  const pipelineData = { pipeline: pipelineGPU };
  if (includeImmediateSize) pipelineData.immediateSize = immediateSize;
  backend._draw(
    renderObject,
    {
      update(updatedObject, count, instances) {
        assert.equal(updatedObject, object);
        infoUpdates.push([count, instances]);
      },
    },
    {},
    pipelineData,
    [],
    [],
    { vertexCount, instanceCount, firstVertex },
    encoder,
    currentSets,
  );
  return { encoder, currentSets, infoUpdates };
}

before(async () => {
  temporaryRoot = await mkdtemp(path.join(os.tmpdir(), 'three-draw-index-traces-'));
  const overlayRoot = path.join(temporaryRoot, 'overlay');
  await buildThreeDrawIndexOverlay({ outputRoot: overlayRoot });
  ({ default: WebGPUBackend } = await import(pathToFileURL(
    path.join(overlayRoot, 'src', 'renderers', 'webgpu', 'WebGPUBackend.js'),
  )));
}, { timeout: 120_000 });

after(async () => {
  if (temporaryRoot !== undefined) {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test('direct indexed and non-indexed draws set ordinal zero immediately before draw', () => {
  const backend = new WebGPUBackend();
  const indexed = runDraw(backend, {
    indexed: true,
    vertexCount: 36,
    instanceCount: 7,
    firstVertex: 3,
  });
  assert.deepEqual(indexed.encoder.events, [
    pipeline('P4'),
    indexBuffer('IB', 'uint16'),
    immediate(0),
    draw('drawIndexed', 36, 7, 3, 0, 0),
  ]);
  assert.deepEqual(indexed.infoUpdates, [[36, 7]]);

  const nonIndexed = runDraw(backend, {
    vertexCount: 12,
    instanceCount: 7,
    firstVertex: 5,
  });
  assert.deepEqual(nonIndexed.encoder.events, [
    pipeline('P4'),
    immediate(0),
    draw('draw', 12, 7, 5, 0),
  ]);
  assert.deepEqual(nonIndexed.infoUpdates, [[12, 7]]);
});

test('BatchedMesh uses compact ordinals and preserves firstInstance', () => {
  const backend = new WebGPUBackend();
  const indexed = runDraw(backend, {
    indexed: true,
    batched: {
      starts: [0, 12, 28],
      counts: [6, 0, 9],
      drawCount: 3,
      bytesPerElement: 4,
    },
  });
  assert.deepEqual(indexed.encoder.events, [
    pipeline('P4'),
    indexBuffer('IB', 'uint16'),
    immediate(0),
    draw('drawIndexed', 6, 1, 0, 0, 0),
    immediate(1),
    draw('drawIndexed', 0, 1, 3, 0, 1),
    immediate(2),
    draw('drawIndexed', 9, 1, 7, 0, 2),
  ]);
  assert.deepEqual(indexed.infoUpdates, [[6, 1], [0, 1], [9, 1]]);

  const nonIndexed = runDraw(backend, {
    batched: {
      starts: [5, 0, 21],
      counts: [3, 0, 4],
      drawCount: 3,
    },
  });
  assert.deepEqual(nonIndexed.encoder.events, [
    pipeline('P4'),
    immediate(0),
    draw('draw', 3, 1, 5, 0),
    immediate(1),
    draw('draw', 0, 1, 0, 1),
    immediate(2),
    draw('draw', 4, 1, 21, 2),
  ]);
  assert.deepEqual(nonIndexed.infoUpdates, [[3, 1], [0, 1], [4, 1]]);

  const defaultPipeline = runDraw(backend, {
    immediateSize: 0,
    pipelineGPU: 'P0',
    batched: {
      starts: [4, 12],
      counts: [3, 6],
      drawCount: 2,
    },
  });
  assert.deepEqual(defaultPipeline.encoder.events, [
    pipeline('P0'),
    draw('draw', 3, 1, 4, 0),
    draw('draw', 6, 1, 12, 1),
  ]);
});

test('array-indirect ordinal follows array position, not byte offset', () => {
  const backend = new WebGPUBackend();
  const offsets = [40, 0, 40, 20];
  const indexed = runDraw(backend, {
    indexed: true,
    indirectOffset: offsets,
    vertexCount: 36,
    instanceCount: 2,
  });
  assert.deepEqual(indexed.encoder.events, [
    pipeline('P4'),
    indexBuffer('IB', 'uint16'),
    immediate(0),
    draw('drawIndexedIndirect', 'B', 40),
    immediate(1),
    draw('drawIndexedIndirect', 'B', 0),
    immediate(2),
    draw('drawIndexedIndirect', 'B', 40),
    immediate(3),
    draw('drawIndexedIndirect', 'B', 20),
  ]);

  const nonIndexed = runDraw(backend, { indirectOffset: offsets });
  assert.deepEqual(nonIndexed.encoder.events, [
    pipeline('P4'),
    immediate(0),
    draw('drawIndirect', 'B', 40),
    immediate(1),
    draw('drawIndirect', 'B', 0),
    immediate(2),
    draw('drawIndirect', 'B', 40),
    immediate(3),
    draw('drawIndirect', 'B', 20),
  ]);
});

test('scalar indirect and consecutive logical lists each begin at zero', () => {
  const backend = new WebGPUBackend();
  const encoder = createEncoder();
  const currentSets = freshCurrentSets();
  runDraw(backend, { encoder, currentSets, indirectOffset: [40, 20] });
  runDraw(backend, { encoder, currentSets, indirectOffset: 64 });
  runDraw(backend, { encoder, currentSets, indirectOffset: [12, 4] });
  assert.deepEqual(
    encoder.events.filter((event) => event.method === 'setImmediates')
      .map((event) => event.selectedValues[0]),
    [0, 1, 0, 0, 1],
  );
  assert.equal(
    encoder.events.filter((event) => event.method === 'setPipeline').length,
    1,
  );
});

test('ordinary pipelines issue no setters and pipeline switches do not cache values', () => {
  const backend = new WebGPUBackend();
  const encoder = createEncoder();
  const currentSets = freshCurrentSets();
  runDraw(backend, { encoder, currentSets, immediateSize: 4, pipelineGPU: 'P4' });
  runDraw(backend, { encoder, currentSets, immediateSize: 0, pipelineGPU: 'P0' });
  runDraw(backend, { encoder, currentSets, immediateSize: 4, pipelineGPU: 'P4' });
  assert.deepEqual(encoder.events, [
    pipeline('P4'),
    immediate(0),
    draw('draw', 12, 1, 5, 0),
    pipeline('P0'),
    draw('draw', 12, 1, 5, 0),
    pipeline('P4'),
    immediate(0),
    draw('draw', 12, 1, 5, 0),
  ]);
});

test('zero-count physical draws still get values while empty lists do not', () => {
  const backend = new WebGPUBackend();
  const direct = runDraw(backend, { vertexCount: 0 });
  assert.deepEqual(direct.encoder.events, [
    pipeline('P4'),
    immediate(0),
    draw('draw', 0, 1, 5, 0),
  ]);

  const emptyBatch = runDraw(backend, {
    batched: { starts: [], counts: [], drawCount: 0 },
  });
  assert.deepEqual(emptyBatch.encoder.events, [pipeline('P4')]);
  assert.deepEqual(emptyBatch.infoUpdates, []);

  const emptyIndirect = runDraw(backend, { indirectOffset: [] });
  assert.deepEqual(emptyIndirect.encoder.events, [pipeline('P4')]);
  assert.equal(
    emptyIndirect.encoder.events.some((event) => event.method === 'setImmediates'),
    false,
  );
});

test('missing or invalid immediate support fails before encoder mutation', () => {
  const backend = new WebGPUBackend();
  const missing = createEncoder({ supportsImmediates: false });
  assert.throws(
    () => runDraw(backend, { encoder: missing }),
    /requires unsupported immediate data/,
  );
  assert.deepEqual(missing.events, []);

  const nonCallable = createEncoder();
  nonCallable.setImmediates = 17;
  assert.throws(
    () => runDraw(backend, { encoder: nonCallable }),
    /requires unsupported immediate data/,
  );
  assert.deepEqual(nonCallable.events, []);

  const invalidSize = createEncoder();
  assert.throws(
    () => runDraw(backend, { encoder: invalidSize, immediateSize: 8 }),
    /invalid immediate data metadata/,
  );
  assert.deepEqual(invalidSize.events, []);

  const missingMetadata = createEncoder();
  assert.throws(
    () => runDraw(backend, {
      encoder: missingMetadata,
      includeImmediateSize: false,
    }),
    /invalid immediate data metadata/,
  );
  assert.deepEqual(missingMetadata.events, []);

  const nullMetadata = createEncoder();
  assert.throws(
    () => runDraw(backend, {
      encoder: nullMetadata,
      immediateSize: null,
      pipelineGPU: 'P-null',
    }),
    /invalid immediate data metadata/,
  );
  assert.deepEqual(nullMetadata.events, []);

  const ordinary = createEncoder({ supportsImmediates: false });
  runDraw(backend, { encoder: ordinary, immediateSize: 0, pipelineGPU: 'P0' });
  assert.deepEqual(ordinary.events, [
    pipeline('P0'),
    draw('draw', 12, 1, 5, 0),
  ]);
});

test('render-bundle recording uses call-time snapshots from the reused array', () => {
  const backend = new WebGPUBackend();
  const bundleEncoder = createEncoder();
  runDraw(backend, {
    encoder: bundleEncoder,
    indirectOffset: [0, 20, 40],
  });
  backend._drawIndexData[0] = 999;
  assert.deepEqual(
    bundleEncoder.events.filter((event) => event.method === 'setImmediates')
      .map((event) => event.selectedValues[0]),
    [0, 1, 2],
  );
  const terminal = bundleEncoder.events.filter((event) => (
    event.method === 'setImmediates' || event.method.startsWith('draw')
  ));
  for (let i = 0; i < terminal.length; i += 2) {
    assert.equal(terminal[i].method, 'setImmediates');
    assert.match(terminal[i + 1].method, /^draw/);
  }
});

test('public draw selects ArrayCamera bundle encoders and resets each list', () => {
  const backend = new WebGPUBackend();
  backend.renderer = { getPixelRatio: () => 1 };
  const encoders = [createEncoder(), createEncoder()];
  const context = {
    renderTarget: null,
    stencil: false,
  };
  backend.set(context, {
    bundleEncoders: encoders,
    bundleSets: [freshCurrentSets(), freshCurrentSets()],
  });

  const pipelineKey = {};
  backend.set(pipelineKey, { pipeline: 'P4', immediateSize: 4 });
  const cameraIndex = {};
  const camera = {
    isArrayCamera: true,
    cameras: [
      { layers: {}, viewport: null },
      { layers: {}, viewport: null },
    ],
  };
  backend.set(camera, { indexesGPU: ['C0', 'C1'] });
  const indirect = {};
  backend.set(indirect, { buffer: 'B' });

  const object = { layers: { test: () => true } };
  const renderObject = {
    object,
    material: {},
    context,
    pipeline: pipelineKey,
    camera,
    getDrawParameters: () => ({
      vertexCount: 12,
      instanceCount: 1,
      firstVertex: 0,
    }),
    getBindings: () => [cameraIndex],
    getVertexBuffers: () => [],
    getBindingGroup: (name) => {
      assert.equal(name, 'cameraIndex');
      return cameraIndex;
    },
    getIndex: () => null,
    getIndirect: () => indirect,
    getIndirectOffset: () => [12, 24],
  };
  const infoUpdates = [];
  backend.draw(renderObject, {
    update(updatedObject, count, instances) {
      assert.equal(updatedObject, object);
      infoUpdates.push([count, instances]);
    },
  });

  for (let i = 0; i < encoders.length; i += 1) {
    assert.deepEqual(encoders[i].events, [
      draw('setBindGroup', 0, `C${i}`),
      pipeline('P4'),
      immediate(0),
      draw('drawIndirect', 'B', 12),
      immediate(1),
      draw('drawIndirect', 'B', 24),
    ]);
  }
  assert.deepEqual(infoUpdates, [
    [12, 1],
    [12, 1],
  ]);
});

test('uint32 indexed draws preserve the existing index format', () => {
  const backend = new WebGPUBackend();
  const result = runDraw(backend, { indexed: true, uint32Index: true });
  assert.deepEqual(result.encoder.events.slice(0, 2), [
    pipeline('P4'),
    indexBuffer('IB', 'uint32'),
  ]);
});
