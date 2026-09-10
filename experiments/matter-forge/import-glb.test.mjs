import test from 'node:test';
import assert from 'node:assert/strict';
import { BoxGeometry } from 'three/webgpu';
import { importGLBMatter, validateGLBMatter } from './import-glb.mjs';

function cubeDocument() {
  const geometry = new BoxGeometry(2, 2, 2), positions = geometry.attributes.position.array, indices = geometry.index.array;
  const binary = new Uint8Array(positions.byteLength + indices.byteLength);
  binary.set(new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength));
  binary.set(new Uint8Array(indices.buffer, indices.byteOffset, indices.byteLength), positions.byteLength);
  const json = { asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, translation: [4, 2, -3] }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [.25, .5, .75, 1] } }],
    buffers: [{ byteLength: binary.length }], bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }, { buffer: 0, byteOffset: positions.byteLength, byteLength: indices.byteLength }],
    accessors: [{ bufferView: 0, componentType: 5126, type: 'VEC3', count: positions.length / 3, min: [-1, -1, -1], max: [1, 1, 1] },
      { bufferView: 1, componentType: 5123, type: 'SCALAR', count: indices.length }] };
  geometry.dispose(); return { json, binary };
}
function encode({ json, binary }) {
  const text = new TextEncoder().encode(JSON.stringify(json)), jsonBytes = Math.ceil(text.length / 4) * 4, binaryBytes = Math.ceil(binary.length / 4) * 4;
  const bytes = new Uint8Array(12 + 8 + jsonBytes + 8 + binaryBytes), view = new DataView(bytes.buffer);
  view.setUint32(0, 0x46546c67, true); view.setUint32(4, 2, true); view.setUint32(8, bytes.length, true);
  view.setUint32(12, jsonBytes, true); view.setUint32(16, 0x4e4f534a, true); bytes.fill(32, 20, 20 + jsonBytes); bytes.set(text, 20);
  view.setUint32(20 + jsonBytes, binaryBytes, true); view.setUint32(24 + jsonBytes, 0x004e4942, true); bytes.set(binary, 28 + jsonBytes);
  return bytes.buffer;
}
const fileOf = document => new File([encode(document)], 'closed-cube.glb', { type: 'model/gltf-binary' });

test('embedded GLB cube becomes a closed volume with exact base color and owned cleanup', async () => {
  const document = cubeDocument(), file = fileOf(document), original = new Uint8Array(await file.arrayBuffer());
  const result = await importGLBMatter(file);
  assert.equal(result.pointCount, 216); assert.equal(result.masses.reduce((a, b) => a + b, 0), 27);
  assert.deepEqual(result.colors.slice(0, 3), new Float32Array([.25, .5, .75]));
  assert.equal(result.source.definitionTriangles, 12); assert.equal(result.source.externalRequests, 0);
  assert.equal(result.source.resourcesDisposed, true); assert.ok(result.source.disposedCounts.geometries >= 1);
  assert.ok(result.source.disposedCounts.materials >= 1); assert.match(result.source.sha256, /^[a-f0-9]{64}$/);
  assert.deepEqual(new Uint8Array(await file.arrayBuffer()), original);
});

test('embedded texture bytes are never decoded or fetched; base color remains explicit', async () => {
  const document = cubeDocument();
  // This is deliberately not a PNG; successful import proves image decoding is skipped.
  document.json.images = [{ bufferView: 0, mimeType: 'image/png' }]; document.json.textures = [{ source: 0 }];
  document.json.materials[0].pbrMetallicRoughness.baseColorTexture = { index: 0 };
  const result = await importGLBMatter(fileOf(document));
  assert.equal(result.pointCount, 216); assert.equal(result.source.embeddedImages, 1);
  assert.equal(result.source.texturesDecoded, 0); assert.equal(result.source.externalRequests, 0);
  assert.equal(result.source.disposedCounts.textures, 0);
});

test('URI resources, decoding extensions, animations and skinning fail before loading', async () => {
  for (const mutate of [
    d => { d.json.buffers[0].uri = 'https://example.invalid/model.bin'; },
    d => { d.json.buffers[0].uri = 'data:application/octet-stream;base64,AAAA'; },
    d => { d.json.images = [{ uri: 'https://example.invalid/image.png' }]; },
    d => { d.json.extensionsRequired = ['KHR_draco_mesh_compression']; },
    d => { d.json.bufferViews[0].extensions = { EXT_meshopt_compression: {} }; },
    d => { d.json.nodes[0].extensions = { EXT_mesh_gpu_instancing: {} }; },
    d => { d.json.animations = [{}]; },
    d => { d.json.nodes[0].skin = 0; },
  ]) { const document = cubeDocument(); mutate(document); await assert.rejects(importGLBMatter(fileOf(document)), /importGLBMatter:/); }
});

test('GLB container and decoded/instanced geometry limits are enforced before loader allocation', () => {
  const original = encode(cubeDocument());
  for (const offset of [0, 4, 8, 12, 16]) { const bytes = original.slice(0); new DataView(bytes).setUint32(offset, 0, true); assert.throws(() => validateGLBMatter(bytes), /importGLBMatter:/); }
  for (const mutate of [
    d => { d.json.accessors[0].count = 1000000000; },
    d => { d.json.bufferViews[0].byteLength = 1000000000; },
    d => { d.json.accessors[0].sparse = {}; },
    d => { d.json.nodes = Array.from({ length: 65 }, () => ({ mesh: 0 })); d.json.scenes[0].nodes = d.json.nodes.map((_, i) => i); },
    d => { d.json.nodes[0].children = [0]; },
  ]) { const document = cubeDocument(); mutate(document); assert.throws(() => validateGLBMatter(encode(document)), /importGLBMatter:/); }
});

test('open and invalid-index meshes are rejected and every resolved geometry/material is disposed', async () => {
  for (const mutate of [d => { d.json.accessors[1].count -= 3; }, d => { new DataView(d.binary.buffer).setUint16(d.json.bufferViews[1].byteOffset, 999, true); }]) {
    const document = cubeDocument(); mutate(document);
    await assert.rejects(importGLBMatter(fileOf(document)), error => {
      assert.match(error.message, /voxelizeMeshes:/); assert.equal(error.importMetadata.resourcesDisposed, true);
      assert.ok(error.importMetadata.disposedCounts.geometries >= 1); assert.ok(error.importMetadata.disposedCounts.materials >= 1); return true;
    });
  }
});

test('file and particle caps reject before expensive work', async () => {
  await assert.rejects(importGLBMatter({ name: 'huge.glb', size: 5 * 1024 * 1024 + 1, arrayBuffer() { throw Error('Must not read'); } }), /at most 5 MiB/);
  await assert.rejects(importGLBMatter(new File(['not glb'], 'wrong.gltf')), /local .glb/);
  await assert.rejects(importGLBMatter(fileOf(cubeDocument()), { maxParticles: 5000 }), /particle cap/);
  await assert.rejects(importGLBMatter(fileOf(cubeDocument()), { maxParticles: 32 }), /particle cap/);
});
