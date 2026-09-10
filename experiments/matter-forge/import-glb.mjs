import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { voxelizeMeshes } from './voxelize.mjs';

const LIMITS = Object.freeze({ bytes: 5 * 1024 * 1024, jsonBytes: 1024 * 1024, triangles: 4096, primitives: 64, nodes: 256, accessorValues: 262144 });
const COMPONENT_BYTES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
// These extensions require no external decoder or additional geometry runtime.
// Material appearance is reduced to its base color by the existing voxelizer.
const SAFE_EXTENSIONS = new Set(['KHR_materials_unlit', 'KHR_mesh_quantization', 'KHR_texture_transform',
  'KHR_materials_clearcoat', 'KHR_materials_transmission', 'KHR_materials_volume', 'KHR_materials_ior',
  'KHR_materials_specular', 'KHR_materials_sheen', 'KHR_materials_iridescence', 'KHR_materials_emissive_strength',
  'KHR_materials_anisotropy', 'KHR_materials_dispersion']);
const fail = message => { throw Error('importGLBMatter: ' + message); };
const integer = (value, min, max, label) => {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail('Invalid ' + label);
  return value;
};
const list = (json, key, cap) => {
  const value = json[key] ?? [];
  if (!Array.isArray(value) || value.length > cap) fail(`${key} exceeds its bounded import scope`);
  return value;
};

/** CPU validation precedes GLTFLoader, including allocation and instance caps. */
export function validateGLBMatter(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 28 || buffer.byteLength > LIMITS.bytes) fail('GLB must be between 28 bytes and 5 MiB');
  const data = new DataView(buffer);
  if (data.getUint32(0, true) !== 0x46546c67 || data.getUint32(4, true) !== 2 || data.getUint32(8, true) !== buffer.byteLength) fail('Expected exact GLB v2 header and declared file length');
  let json, binaryBytes = 0, offset = 12, chunks = 0;
  while (offset < buffer.byteLength) {
    if (offset + 8 > buffer.byteLength) fail('Truncated GLB chunk header');
    const length = data.getUint32(offset, true), type = data.getUint32(offset + 4, true); offset += 8;
    if (length % 4 || offset + length > buffer.byteLength) fail('Invalid GLB chunk length or alignment');
    if (chunks === 0) {
      if (type !== 0x4e4f534a || length > LIMITS.jsonBytes) fail('First chunk must be bounded JSON');
      try { json = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(buffer, offset, length))); }
      catch { fail('Invalid UTF-8 GLB JSON'); }
    } else if (chunks === 1 && type === 0x004e4942) binaryBytes = length;
    else fail('Only one JSON chunk and one embedded BIN chunk are supported');
    offset += length; chunks++;
  }
  if (!json || typeof json !== 'object' || Array.isArray(json) || json.asset?.version !== '2.0' || (json.asset.minVersion && json.asset.minVersion !== '2.0')) fail('Expected glTF 2.0 metadata');
  const buffers = list(json, 'buffers', 1), views = list(json, 'bufferViews', 1024), accessors = list(json, 'accessors', 1024);
  const meshes = list(json, 'meshes', 64), nodes = list(json, 'nodes', LIMITS.nodes), scenes = list(json, 'scenes', 16);
  const images = list(json, 'images', 64), textures = list(json, 'textures', 64), materials = list(json, 'materials', 64);
  if (!buffers.length || !binaryBytes || !meshes.length || !scenes.length) fail('A scene with embedded mesh buffers is required');
  if (list(json, 'animations', 64).length || list(json, 'skins', 64).length) fail('Bake animation and skinning into a static closed mesh first');
  for (const item of [...buffers, ...images]) if (!item || typeof item !== 'object' || item.uri !== undefined) fail('URI buffers and images are unsupported; use embedded GLB data');
  integer(buffers[0].byteLength, 1, binaryBytes, 'embedded buffer length');
  if (binaryBytes - buffers[0].byteLength > 3) fail('BIN padding exceeds three bytes');
  for (const field of ['extensionsUsed', 'extensionsRequired']) for (const name of list(json, field, 64)) if (!SAFE_EXTENSIONS.has(name)) fail(`Unsupported extension ${String(name)}; no decoder is installed`);
  const objects = [json]; let objectCount = 0;
  while (objects.length) {
    const object = objects.pop(); if (++objectCount > 50000) fail('JSON object count exceeds import cap');
    if (object.extensions) for (const name of Object.keys(object.extensions)) if (!SAFE_EXTENSIONS.has(name)) fail('Unsupported extension ' + name);
    for (const value of Object.values(object)) if (value && typeof value === 'object') objects.push(value);
  }
  views.forEach(view => {
    if (view.buffer !== 0) fail('Every bufferView must use embedded buffer 0');
    const start = integer(view.byteOffset ?? 0, 0, buffers[0].byteLength, 'bufferView offset');
    integer(view.byteLength, 1, buffers[0].byteLength - start, 'bufferView length');
    if (view.byteStride !== undefined && (integer(view.byteStride, 4, 252, 'byteStride') % 4)) fail('byteStride must be four-byte aligned');
  });
  let accessorValues = 0;
  accessors.forEach(accessor => {
    if (accessor.sparse) fail('Sparse accessors require explicit baking first');
    const count = integer(accessor.count, 1, 12288, 'accessor count'), width = COMPONENTS[accessor.type], bytes = COMPONENT_BYTES[accessor.componentType];
    if (!width || !bytes) fail('Unsupported accessor format');
    accessorValues += count * width; if (accessorValues > LIMITS.accessorValues) fail('Decoded accessor allocation cap exceeded');
    const view = views[integer(accessor.bufferView, 0, views.length - 1, 'accessor bufferView')];
    const start = integer(accessor.byteOffset ?? 0, 0, view.byteLength, 'accessor offset'), stride = view.byteStride ?? width * bytes;
    if (start % bytes || (view.byteOffset ?? 0) % bytes || stride < width * bytes || start + (count - 1) * stride + width * bytes > view.byteLength) fail('Accessor does not fit its bufferView');
  });
  const meshSizes = []; let definitionTriangles = 0, definitionPrimitives = 0;
  meshes.forEach(mesh => {
    if (mesh.weights !== undefined || !Array.isArray(mesh.primitives) || !mesh.primitives.length) fail('Static triangle primitives are required');
    let triangles = 0;
    for (const primitive of mesh.primitives) {
      if ((primitive.mode ?? 4) !== 4 || primitive.targets?.length) fail('Only static TRIANGLES are supported; bake morphs first');
      const attributes = primitive.attributes;
      if (!attributes || Object.keys(attributes).length > 16 || attributes.JOINTS_0 !== undefined || attributes.WEIGHTS_0 !== undefined) fail('Unsupported vertex attributes or skinning');
      const position = accessors[integer(attributes.POSITION, 0, accessors.length - 1, 'POSITION accessor')];
      if (position.type !== 'VEC3') fail('POSITION must contain VEC3 values');
      for (const index of Object.values(attributes)) if (accessors[integer(index, 0, accessors.length - 1, 'attribute accessor')].count !== position.count) fail('Vertex attribute counts must match POSITION');
      let count = position.count;
      if (primitive.indices !== undefined) {
        const indices = accessors[integer(primitive.indices, 0, accessors.length - 1, 'index accessor')];
        if (indices.type !== 'SCALAR' || ![5121, 5123, 5125].includes(indices.componentType)) fail('Indices must be unsigned scalar integers');
        count = indices.count;
      }
      if (count % 3 || count < 12) fail('A closed triangular primitive needs at least four triangles');
      if (primitive.material !== undefined) integer(primitive.material, 0, materials.length - 1, 'material index');
      triangles += count / 3; definitionPrimitives++;
    }
    definitionTriangles += triangles; meshSizes.push({ triangles, primitives: mesh.primitives.length });
  });
  if (definitionTriangles > LIMITS.triangles || definitionPrimitives > LIMITS.primitives) fail('Mesh definitions exceed 4096 triangles or 64 primitives');
  images.forEach(image => integer(image.bufferView, 0, views.length - 1, 'embedded image bufferView'));
  textures.forEach(texture => integer(texture.source, 0, images.length - 1, 'texture source'));
  nodes.forEach(node => {
    if (node.skin !== undefined || node.weights !== undefined) fail('Bake skinned or morphed nodes first');
    if (node.mesh !== undefined) integer(node.mesh, 0, meshes.length - 1, 'node mesh');
    for (const [field, length] of [['matrix', 16], ['translation', 3], ['rotation', 4], ['scale', 3]]) {
      if (node[field] !== undefined && (!Array.isArray(node[field]) || node[field].length !== length || !node[field].every(Number.isFinite))) fail('Invalid node ' + field);
    }
    for (const child of list(node, 'children', LIMITS.nodes)) integer(child, 0, nodes.length - 1, 'child node');
  });
  let instanceTriangles = 0, instancePrimitives = 0;
  for (const scene of scenes) {
    const stack = list(scene, 'nodes', LIMITS.nodes).map(index => ({ index, depth: 0 })), visited = new Set();
    while (stack.length) {
      const { index, depth } = stack.pop(); integer(index, 0, nodes.length - 1, 'scene node');
      if (visited.has(index) || depth > 64) fail('Cyclic, shared-child, or excessively deep scene graph');
      visited.add(index); const node = nodes[index];
      if (node.mesh !== undefined) { instanceTriangles += meshSizes[node.mesh].triangles; instancePrimitives += meshSizes[node.mesh].primitives; }
      if (instanceTriangles > LIMITS.triangles || instancePrimitives > LIMITS.primitives) fail('Instanced scene geometry exceeds 4096 triangles or 64 primitives');
      for (const child of node.children ?? []) stack.push({ index: child, depth: depth + 1 });
    }
  }
  integer(json.scene ?? 0, 0, scenes.length - 1, 'default scene');
  return { json, limits: LIMITS, definitionTriangles, definitionPrimitives, instanceTriangles, instancePrimitives, embeddedImages: images.length, accessorValues };
}

/** Local File/Blob -> closed-volume particles. No renderer/device is created. */
export async function importGLBMatter(file, { targetBounds = { min: [6, 13, 10], max: [18, 17, 13] }, spacing = .5, maxParticles = 4096 } = {}) {
  if (!file || typeof file.arrayBuffer !== 'function' || !Number.isSafeInteger(file.size) || file.size < 28 || file.size > LIMITS.bytes || !/\.glb$/i.test(file.name ?? '')) fail('Choose a local .glb file of at most 5 MiB');
  integer(maxParticles, 1, 4096, 'particle cap');
  const bytes = await file.arrayBuffer();
  if (bytes.byteLength !== file.size) fail('File size changed while reading');
  const validation = validateGLBMatter(bytes);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const source = { name: String(file.name).split(/[\\/]/).at(-1), bytes: bytes.byteLength,
    sha256: Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join(''), threeRevision: THREE.REVISION,
    ...Object.fromEntries(['definitionTriangles', 'definitionPrimitives', 'instanceTriangles', 'instancePrimitives', 'embeddedImages', 'accessorValues'].map(key => [key, validation[key]])),
    texturesDecoded: 0, colorSource: 'Linear material base color only; image textures, vertex colors, opacity and advanced appearance are not sampled.',
    volumeAssumption: 'Uniformly fitted closed-mesh union sampled at voxel centers; mass equals represented volume at density 1. Bridge outcomes are not guaranteed.',
    externalRequests: 0, inputModified: false, resourcesDisposed: false };
  const geometries = new Set(), materials = new Set(), textures = new Set(), pending = new Set();
  const trackMaterial = material => { if (!material) return; materials.add(material); for (const value of Object.values(material)) if (value?.isTexture) textures.add(value); };
  const trackTree = root => root?.traverse(object => { if (object.geometry) geometries.add(object.geometry); for (const material of Array.isArray(object.material) ? object.material : [object.material]) trackMaterial(material); });
  const manager = new THREE.LoadingManager();
  manager.setURLModifier(() => { source.externalRequests++; fail('Unexpected resource request was blocked'); });
  let parser, result;
  const loader = new GLTFLoader(manager);
  loader.register(value => {
    parser = value;
    return { name: 'MATTER_FORGE_BASE_COLOR_ONLY',
      // No image decoder, object URL, texture fetch, or large compressed-image allocation.
      loadTexture() { return Promise.resolve(null); },
      loadMaterial(index) { const promise = parser.loadMaterial(index).then(material => { trackMaterial(material); return material; }); pending.add(promise); return promise; },
      loadMesh(index) { const promise = parser.loadMesh(index).then(mesh => { trackTree(mesh); return mesh; }); pending.add(promise); return promise; } };
  });
  try {
    result = await new Promise((resolve, reject) => loader.parse(bytes, '', resolve, reject));
    for (const scene of result.scenes) trackTree(scene);
    const meshes = []; result.scene.updateMatrixWorld(true); result.scene.traverse(object => { if (object.isMesh) meshes.push(object); });
    if (!meshes.length) fail('The default scene contains no mesh');
    const voxels = voxelizeMeshes(meshes, { spacing, maxParticles, targetBounds });
    return { ...voxels, source };
  } catch (error) { error.importMetadata = source; throw error; }
  finally {
    // GLTFLoader can reject one branch while another geometry is still resolving.
    // Its r185/r186 primitive cache lets us own successful partial allocations too.
    await Promise.allSettled(pending);
    const cached = await Promise.allSettled(Object.values(parser?.primitiveCache ?? {}).map(entry => entry.promise));
    for (const entry of cached) if (entry.status === 'fulfilled' && entry.value?.isBufferGeometry) geometries.add(entry.value);
    for (const object of parser?.associations?.keys() ?? []) { if (object.isMaterial) trackMaterial(object); if (object.isTexture) textures.add(object); if (object.isObject3D) trackTree(object); }
    for (const scene of result?.scenes ?? []) trackTree(scene);
    geometries.forEach(geometry => geometry.dispose()); materials.forEach(material => material.dispose()); textures.forEach(texture => { texture.dispose(); texture.source?.data?.close?.(); });
    source.disposedCounts = { geometries: geometries.size, materials: materials.size, textures: textures.size };
    source.resourcesDisposed = true;
  }
}
