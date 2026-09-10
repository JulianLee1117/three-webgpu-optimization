import test from 'node:test';
import assert from 'node:assert/strict';
import { BoxGeometry, SphereGeometry, Mesh, MeshBasicMaterial, Group, BufferGeometry, Float32BufferAttribute } from 'three';
import { voxelizeMeshes } from './voxelize.mjs';

const material = color => new MeshBasicMaterial({ color });
const box = (size = 2, color = 0xffffff) => new Mesh(new BoxGeometry(size, size, size), material(color));
const countByAxis = (result, axis) => new Set(Array.from(result.points).filter((_, i) => i % 3 === axis)).size;

test('closed indexed cube: exact center count, mass, linear color and no source mutation', () => {
  const mesh = box(2, 0xff0000), original = mesh.geometry.attributes.position.array.slice();
  const result = voxelizeMeshes([mesh]);
  assert.equal(result.pointCount, 64); assert.equal(result.diagnostics.representedVolume, 8);
  assert.equal(result.masses.reduce((a, b) => a + b, 0), 8);
  assert.deepEqual(result.colors.slice(0, 3), new Float32Array([1, 0, 0]));
  assert.deepEqual(mesh.geometry.attributes.position.array, original);
  assert.deepEqual([0, 1, 2].map(a => countByAxis(result, a)), [4, 4, 4]);
  assert.equal(result.diagnostics.closedComponentCount, 1);
});

test('nonindexed UV-seamed cube has identical interior lattice', () => {
  const a = box(), b = box(); b.geometry = b.geometry.toNonIndexed();
  assert.deepEqual(voxelizeMeshes([a]).points, voxelizeMeshes([b]).points);
});

test('sphere volume converges to known solid volume without sampling only its surface', () => {
  const mesh = new Mesh(new SphereGeometry(1, 24, 16), material(0xffffff));
  const result = voxelizeMeshes([mesh], { spacing: 0.2 });
  assert.ok(Math.abs(result.diagnostics.representedVolume - 4 * Math.PI / 3) < 0.16);
  for (let i = 0; i < result.points.length; i += 3) assert.ok(Math.hypot(...result.points.slice(i, i + 3)) < 1.000001);
  assert.ok(result.diagnostics.representedVolume > 4);
});

test('parent world transform, nonuniform scale and mirrored orientation are honored', () => {
  const group = new Group(), mesh = box(); group.add(mesh);
  group.position.set(4, 3, 2); mesh.scale.set(-2, 1, 0.5); mesh.rotation.y = Math.PI / 2;
  const result = voxelizeMeshes([mesh]);
  assert.equal(result.pointCount, 64); assert.equal(result.diagnostics.representedVolume, 8);
  assert.deepEqual([0, 1, 2].map(a => countByAxis(result, a)), [2, 4, 8]);
  const center = [0, 1, 2].map(k => result.points.reduce((s, v, i) => s + (i % 3 === k ? v : 0), 0) / result.pointCount);
  assert.deepEqual(center, [4, 3, 2]);
});

test('targetBounds applies one uniform fit to all meshes, preserving placement', () => {
  const a = box(); a.position.x = -2;
  const b = box(); b.position.x = 2;
  const result = voxelizeMeshes([a, b], { spacing: 0.25, targetBounds: { min: [2, 2, 2], max: [8, 8, 8] } });
  assert.equal(result.diagnostics.worldToSimulation.scale, 1);
  assert.deepEqual(result.diagnostics.worldToSimulation.translation, [5, 5, 5]);
  assert.equal(result.pointCount, 1024);
});

test('overlapping closed meshes form a union with no duplicate particles or mass', () => {
  const a = box(2, 0xff0000), b = box(2, 0x0000ff); b.position.x = 1;
  const result = voxelizeMeshes([a, b]);
  assert.equal(result.pointCount, 96); assert.equal(result.diagnostics.representedVolume, 12);
  assert.equal(result.diagnostics.overlapCenters, 32);
  const unique = new Set(); for (let i = 0; i < result.points.length; i += 3) unique.add(result.points.slice(i, i + 3).join(','));
  assert.equal(unique.size, result.pointCount);
  assert.equal(voxelizeMeshes([a, a]).pointCount, 64);
});

test('closed disconnected components in one BufferGeometry are unioned', () => {
  const one = new BoxGeometry(2, 2, 2).toNonIndexed().attributes.position.array;
  const data = new Float32Array(one.length * 2); data.set(one);
  for (let i = 0; i < one.length; i++) data[one.length + i] = one[i] + (i % 3 === 0 ? 3 : 0);
  const geometry = new BufferGeometry(); geometry.setAttribute('position', new Float32BufferAttribute(data, 3));
  const result = voxelizeMeshes([new Mesh(geometry, material(0xffffff))]);
  assert.equal(result.pointCount, 128); assert.equal(result.diagnostics.closedComponentCount, 2);
});

test('material groups carry deterministic source provenance; texture omission is explicit', () => {
  const mesh = box(); mesh.material = [0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0xff00ff, 0x00ffff].map(material);
  mesh.material[0].map = {}; mesh.material[1].vertexColors = true;
  const result = voxelizeMeshes([mesh]);
  for (let i = 0; i < result.pointCount; i++) {
    const c = mesh.material[result.materialProvenance.materialIndices[i]].color;
    assert.deepEqual(result.colors.slice(i * 3, i * 3 + 3), new Float32Array([c.r, c.g, c.b]));
  }
  assert.equal(result.diagnostics.sources[0].materials[0].textureIgnored, true);
  assert.equal(result.diagnostics.sources[0].materials[1].vertexColorsIgnored, true);
});

test('open, inconsistent, unsupported and invalid inputs reject without guessing a volume', () => {
  const open = box(); open.geometry.setIndex(Array.from(open.geometry.index.array).slice(3));
  assert.throws(() => voxelizeMeshes([open]), /open or nonmanifold/);
  const flipped = box(), index = Array.from(flipped.geometry.index.array); [index[0], index[1]] = [index[1], index[0]]; flipped.geometry.setIndex(index);
  assert.throws(() => voxelizeMeshes([flipped]), /inconsistent triangle orientation/);
  const bad = box(); bad.geometry.attributes.position.setX(0, NaN);
  assert.throws(() => voxelizeMeshes([bad]), /nonfinite vertex/);
  const skinned = box(); skinned.isSkinnedMesh = true;
  assert.throws(() => voxelizeMeshes([skinned]), /baking/);
  const morph = box(); morph.geometry.morphAttributes.position = [morph.geometry.attributes.position];
  assert.throws(() => voxelizeMeshes([morph]), /morph geometry/);
  const shaderDeformed = box(); shaderDeformed.material.positionNode = {};
  assert.throws(() => voxelizeMeshes([shaderDeformed]), /shader-displaced/);
  assert.throws(() => voxelizeMeshes([box()], { maxParticles: 10 }), /particle cap/);
  assert.throws(() => voxelizeMeshes([box()], { spacing: 0.001 }), /lattice cap/);
});

test('coincident boundary centers are omitted explicitly; deterministic repeats agree', () => {
  const mesh = box();
  const a = voxelizeMeshes([mesh], { spacing: 0.8 }), b = voxelizeMeshes([mesh], { spacing: 0.8 });
  assert.equal(a.pointCount, 8);
  assert.ok(a.diagnostics.boundaryCentersOmitted > 0);
  assert.deepEqual(a.points, b.points); assert.deepEqual(a.materialProvenance, b.materialProvenance);
});
