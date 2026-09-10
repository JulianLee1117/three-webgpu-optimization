/**
 * Deterministic voxel-center sampling of the union of closed Mesh geometries.
 * No Three import: source/build runtime realms cannot be mixed by this module.
 * Validates welded edge topology; ray winding ambiguity at sampled points fails.
 * This is not a global self-intersection certificate or a watertight repair tool.
 */
const DIRECTIONS = [[1, 0.3713906763541037, 0.529177210903], [0.217137, 1, 0.613271], [0.433113, 0.293117, 1], [1, -0.712371, 0.193177]].map(v => {
  const n = Math.hypot(...v); return v.map(x => x / n);
});
const LIMITS = Object.freeze({ meshes: 64, triangles: 4096, pointTests: 200000, triangleTests: 20000000, particles: 8192, milliseconds: 3000 });
const now = () => globalThis.performance?.now() ?? Date.now();
const fail = message => { throw new Error(`voxelizeMeshes: ${message}`); };
const finite3 = v => v?.length === 3 && Array.from(v).every(Number.isFinite);
const emptyBounds = () => ({ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
function extend(bounds, p) { for (let k = 0; k < 3; k++) { bounds.min[k] = Math.min(bounds.min[k], p[k]); bounds.max[k] = Math.max(bounds.max[k], p[k]); } }
const inBounds = (p, b, eps) => p.every((x, k) => x >= b.min[k] - eps && x <= b.max[k] + eps);
const determinant4Affine = m => m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);

function readWorldVertex(attribute, index, matrix) {
  if (!Number.isInteger(index) || index < 0 || index >= attribute.count) fail('invalid geometry index');
  const x = attribute.getX(index), y = attribute.getY(index), z = attribute.getZ(index);
  if (![x, y, z].every(Number.isFinite)) fail('nonfinite vertex');
  const p = [matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12], matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13], matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]];
  if (!finite3(p)) fail('nonfinite transformed vertex');
  return p;
}

function sourceMaterial(mesh, faceOffset) {
  let materialIndex = 0;
  if (Array.isArray(mesh.material)) {
    const matches = (mesh.geometry.groups ?? []).filter(g => faceOffset >= g.start && faceOffset + 2 < g.start + g.count);
    if (matches.length > 1) fail('overlapping geometry material groups');
    materialIndex = matches[0]?.materialIndex ?? 0;
  }
  const material = Array.isArray(mesh.material) ? mesh.material[materialIndex] : mesh.material;
  if (!material) fail(`missing material ${materialIndex}`);
  if (material.positionNode || material.vertexNode || material.displacementMap || material.clippingPlanes?.length) fail('shader-displaced or clipped geometry requires explicit baking first');
  const color = material.color ? [material.color.r, material.color.g, material.color.b] : [1, 1, 1];
  if (!finite3(color) || color.some(v => v < 0)) fail('invalid linear material color');
  return { materialIndex, color, textureIgnored: Boolean(material.map), vertexColorsIgnored: Boolean(material.vertexColors) };
}

function prepareRawMeshes(meshes, diagnostics) {
  const raw = [], worldBounds = emptyBounds();
  let triangleCount = 0;
  meshes.forEach((mesh, meshIndex) => {
    if (!mesh?.geometry || !mesh.isMesh) fail(`input ${meshIndex} is not a Mesh`);
    if (mesh.isSkinnedMesh || mesh.isInstancedMesh || mesh.isBatchedMesh) fail('skinned, instanced and batched meshes require explicit baking first');
    const geometry = mesh.geometry;
    if (Object.values(geometry.morphAttributes ?? {}).some(a => a?.length)) fail('morph geometry requires explicit baking first');
    const position = geometry.getAttribute?.('position') ?? geometry.attributes?.position;
    if (!position || position.itemSize < 3 || typeof position.getX !== 'function') fail('position BufferAttribute is required');
    const index = geometry.getIndex?.() ?? geometry.index;
    const count = index ? index.count : position.count;
    if (!Number.isInteger(count) || count < 12 || count % 3) fail('triangle index/vertex count is invalid for a closed solid');
    const drawRange = geometry.drawRange;
    if (drawRange && (drawRange.start !== 0 || (Number.isFinite(drawRange.count) && drawRange.count < count))) fail('partial drawRange must be baked to a closed geometry first');
    triangleCount += count / 3;
    if (triangleCount > LIMITS.triangles) fail(`triangle cap ${LIMITS.triangles} exceeded`);
    mesh.updateWorldMatrix?.(true, false);
    const matrix = mesh.matrixWorld?.elements;
    if (!matrix || matrix.length !== 16 || !Array.from(matrix).every(Number.isFinite)) fail('finite matrixWorld is required');
    if (Math.abs(matrix[3]) + Math.abs(matrix[7]) + Math.abs(matrix[11]) > 1e-12 || Math.abs(matrix[15] - 1) > 1e-12 || determinant4Affine(matrix) === 0) fail('matrixWorld must be nonsingular and affine');
    const vertices = new Map(), triangles = [], materials = new Map();
    const vertex = i => {
      if (!vertices.has(i)) { const p = readWorldVertex(position, i, matrix); vertices.set(i, p); extend(worldBounds, p); }
      return vertices.get(i);
    };
    for (let f = 0; f < count; f += 3) {
      const ids = [0, 1, 2].map(k => index ? index.getX(f + k) : f + k);
      const material = sourceMaterial(mesh, f);
      materials.set(material.materialIndex, material);
      triangles.push({ vertices: ids.map(vertex), meshIndex, triangleIndex: f / 3, ...material });
    }
    diagnostics.sources.push({ meshIndex, name: String(mesh.name ?? ''), triangleCount: count / 3, materials: [...materials.values()] });
    raw.push({ meshIndex, triangles, vertices: [...vertices.values()] });
  });
  diagnostics.triangleCount = triangleCount;
  return { raw, worldBounds };
}

function createWelder(tolerance) {
  const buckets = new Map(), vertices = [];
  return p => {
    const q = p.map(v => Math.floor(v / tolerance));
    for (let z = -1; z <= 1; z++) for (let y = -1; y <= 1; y++) for (let x = -1; x <= 1; x++) {
      const bucket = buckets.get(`${q[0] + x},${q[1] + y},${q[2] + z}`);
      if (bucket) for (const id of bucket) {
        const v = vertices[id];
        if (Math.hypot(p[0] - v[0], p[1] - v[1], p[2] - v[2]) <= tolerance) return id;
      }
    }
    const id = vertices.length, key = q.join(',');
    vertices.push(p); if (!buckets.has(key)) buckets.set(key, []); buckets.get(key).push(id);
    return id;
  };
}

function prepareComponents(rawMeshes, transform, diagnostics) {
  const all = [];
  for (const mesh of rawMeshes) {
    for (const p of mesh.vertices) for (let k = 0; k < 3; k++) p[k] = p[k] * transform.scale + transform.translation[k];
    const bounds = emptyBounds(); for (const p of mesh.vertices) extend(bounds, p);
    const diagonal = Math.hypot(...bounds.max.map((v, k) => v - bounds.min[k]));
    if (!(diagonal > 0)) fail('zero-size mesh');
    const weldTolerance = Math.max(diagonal * 1e-7, 1e-10), epsilon = Math.max(diagonal * 1e-9, 1e-10);
    const weld = createWelder(weldTolerance), edges = new Map(), parent = mesh.triangles.map((_, i) => i);
    const find = i => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
    for (let t = 0; t < mesh.triangles.length; t++) {
      const triangle = mesh.triangles[t], [a, b, c] = triangle.vertices, ids = triangle.vertices.map(weld);
      const e1 = b.map((v, k) => v - a[k]), e2 = c.map((v, k) => v - a[k]);
      const cross = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const area2 = Math.hypot(...cross);
      if (new Set(ids).size !== 3 || area2 <= diagonal * diagonal * 1e-14) fail(`degenerate or weld-collapsed triangle in mesh ${mesh.meshIndex}`);
      triangle.a = a; triangle.e1 = e1; triangle.e2 = e2;
      triangle.coefficients = DIRECTIONS.map(d => {
        const p = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
        const det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
        return Math.abs(det) <= area2 * 1e-12 ? null : { p, inverse: 1 / det, orientation: -Math.sign(det) };
      });
      for (let k = 0; k < 3; k++) {
        const u = ids[k], v = ids[(k + 1) % 3], key = u < v ? `${u},${v}` : `${v},${u}`;
        if (!edges.has(key)) edges.set(key, []); edges.get(key).push({ t, direction: u < v ? 1 : -1 });
      }
    }
    for (const entries of edges.values()) {
      if (entries.length !== 2) fail(`mesh ${mesh.meshIndex} is open or nonmanifold after welding (edge count ${entries.length})`);
      if (entries[0].direction === entries[1].direction) fail(`mesh ${mesh.meshIndex} has inconsistent triangle orientation`);
      parent[find(entries[1].t)] = find(entries[0].t);
    }
    const components = new Map();
    mesh.triangles.forEach((triangle, i) => {
      const id = find(i);
      if (!components.has(id)) components.set(id, { triangles: [], bounds: emptyBounds(), epsilon, meshIndex: mesh.meshIndex });
      const component = components.get(id); component.triangles.push(triangle);
      for (const p of triangle.vertices) extend(component.bounds, p);
    });
    diagnostics.sources[mesh.meshIndex].closedComponentCount = components.size;
    diagnostics.sources[mesh.meshIndex].weldTolerance = weldTolerance;
    for (const component of components.values()) { component.componentIndex = all.length; all.push(component); }
  }
  diagnostics.closedComponentCount = all.length;
  return all;
}

function castRay(point, component, directionIndex, diagnostics) {
  const d = DIRECTIONS[directionIndex], hits = [], eps = component.epsilon;
  let edgeHit = false;
  for (const triangle of component.triangles) {
    if (++diagnostics.triangleTests > LIMITS.triangleTests) fail(`triangle-test cap ${LIMITS.triangleTests} exceeded`);
    const co = triangle.coefficients[directionIndex]; if (!co) continue;
    const x = point[0] - triangle.a[0], y = point[1] - triangle.a[1], z = point[2] - triangle.a[2];
    const u = (x * co.p[0] + y * co.p[1] + z * co.p[2]) * co.inverse;
    if (u < -1e-9 || u > 1 + 1e-9) continue;
    const qx = y * triangle.e1[2] - z * triangle.e1[1], qy = z * triangle.e1[0] - x * triangle.e1[2], qz = x * triangle.e1[1] - y * triangle.e1[0];
    const v = (d[0] * qx + d[1] * qy + d[2] * qz) * co.inverse;
    if (v < -1e-9 || u + v > 1 + 1e-9) continue;
    const t = (triangle.e2[0] * qx + triangle.e2[1] * qy + triangle.e2[2] * qz) * co.inverse;
    if (Math.abs(t) <= eps) return { boundary: true };
    if (t < 0) continue;
    if (Math.min(u, v, 1 - u - v) <= 1e-9) edgeHit = true;
    hits.push({ t, triangle, orientation: co.orientation });
  }
  hits.sort((a, b) => a.t - b.t);
  // Coincident edge/vertex hits are grouped; such rays are retried, not used as evidence.
  const unique = [];
  for (const hit of hits) {
    const previous = unique.at(-1);
    if (previous && Math.abs(previous.t - hit.t) <= eps) { edgeHit = true; diagnostics.deduplicatedHits++; continue; }
    unique.push(hit);
  }
  if (edgeHit) return { retry: true };
  const winding = unique.reduce((s, h) => s + h.orientation, 0);
  if (Math.abs(winding) > 1 || Math.abs(winding) !== unique.length % 2) fail(`ambiguous/self-overlapping winding in component ${component.componentIndex}`);
  return { inside: winding !== 0, exit: unique[0]?.triangle };
}

function classifyPoint(point, component, diagnostics) {
  if (!inBounds(point, component.bounds, component.epsilon)) return { inside: false };
  if (++diagnostics.pointTests > LIMITS.pointTests) fail(`point-test cap ${LIMITS.pointTests} exceeded`);
  let first;
  for (let i = 0; i < DIRECTIONS.length; i++) {
    const result = castRay(point, component, i, diagnostics);
    if (result.boundary) return result;
    if (result.retry) { diagnostics.rayRetries++; continue; }
    if (!first) first = result;
    else {
      if (first.inside !== result.inside) fail(`direction-dependent volume classification in component ${component.componentIndex}`);
      return first;
    }
  }
  fail(`ambiguous ray/edge classification in component ${component.componentIndex}`);
}

/**
 * targetBounds, if supplied, is {min:[x,y,z],max:[x,y,z]}. All input meshes
 * receive ONE uniform fit transform, preserving their relative placement.
 * Lattice centers are aligned to the resulting union AABB's minimum.
 * Material color is sampled from the first valid ray's exit triangle, in linear
 * RGB. Texture maps/vertex colors are not sampled and are reported explicitly.
 */
export function voxelizeMeshes(meshes, { spacing = 0.5, maxParticles = 8192, targetBounds } = {}) {
  const started = now();
  if (!Array.isArray(meshes) || meshes.length === 0 || meshes.length > LIMITS.meshes) fail(`provide 1..${LIMITS.meshes} meshes`);
  if (!Number.isFinite(spacing) || spacing <= 0 || !Number.isInteger(maxParticles) || maxParticles < 1 || maxParticles > LIMITS.particles) fail('invalid spacing or particle cap');
  const diagnostics = { kind: 'closed-mesh-center-voxelization-v1', sources: [], pointTests: 0, triangleTests: 0, rayRetries: 0, deduplicatedHits: 0, boundaryCentersOmitted: 0, overlapCenters: 0,
    limits: { ...LIMITS }, boundaryPolicy: 'Omit centers on the outer boundary; another component containing a center takes precedence.',
    volumeSemantics: 'Union of individually closed connected components. Cavities represented as separately nested shells are filled by this explicit union policy.',
    topologyValidation: 'Welded manifold edge closure/orientation, nondegenerate triangles, and consistent sampled ray winding. Global triangle self-intersection is not certified.',
    colorSemantics: 'Linear material color of a deterministic first-owner exit triangle; material maps and vertex colors are not sampled.' };
  const { raw, worldBounds } = prepareRawMeshes(meshes, diagnostics);
  const transform = { scale: 1, translation: [0, 0, 0] };
  if (targetBounds) {
    if (!finite3(targetBounds.min) || !finite3(targetBounds.max) || targetBounds.max.some((v, k) => v <= targetBounds.min[k])) fail('targetBounds needs finite strictly increasing min/max');
    const extents = worldBounds.max.map((v, k) => v - worldBounds.min[k]);
    if (extents.some(v => !(v > 0))) fail('input has no three-dimensional extent');
    transform.scale = Math.min(...extents.map((v, k) => (targetBounds.max[k] - targetBounds.min[k]) / v));
    transform.translation = worldBounds.min.map((v, k) => (targetBounds.min[k] + targetBounds.max[k]) / 2 - transform.scale * (v + worldBounds.max[k]) / 2);
  }
  const components = prepareComponents(raw, transform, diagnostics), bounds = emptyBounds();
  for (const component of components) { extend(bounds, component.bounds.min); extend(bounds, component.bounds.max); }
  const counts = bounds.max.map((v, k) => Math.ceil((v - bounds.min[k]) / spacing));
  const candidateCount = counts[0] * counts[1] * counts[2];
  if (!Number.isSafeInteger(candidateCount) || candidateCount > LIMITS.pointTests) fail(`candidate lattice cap ${LIMITS.pointTests} exceeded`);
  const points = [], colors = [], meshIndices = [], materialIndices = [], triangleIndices = [], componentIndices = [];
  let visited = 0;
  for (let z = 0; z < counts[2]; z++) for (let y = 0; y < counts[1]; y++) for (let x = 0; x < counts[0]; x++) {
    if ((visited++ & 63) === 0 && now() - started > LIMITS.milliseconds) fail(`CPU budget ${LIMITS.milliseconds}ms exceeded`);
    const p = [bounds.min[0] + (x + 0.5) * spacing, bounds.min[1] + (y + 0.5) * spacing, bounds.min[2] + (z + 0.5) * spacing];
    let owner, ownerComponent, containing = 0, boundary = false;
    for (const component of components) {
      const result = classifyPoint(p, component, diagnostics);
      boundary ||= Boolean(result.boundary);
      if (result.inside) { containing++; if (!owner) { owner = result.exit; ownerComponent = component; } }
    }
    if (!owner) { if (boundary) diagnostics.boundaryCentersOmitted++; continue; }
    if (points.length / 3 === maxParticles) fail(`particle cap ${maxParticles} exceeded; increase spacing`);
    if (p.some(v => !Number.isFinite(Math.fround(v)))) fail('positions exceed Float32 range');
    if (containing > 1) diagnostics.overlapCenters++;
    points.push(...p); colors.push(...owner.color); meshIndices.push(owner.meshIndex); materialIndices.push(owner.materialIndex); triangleIndices.push(owner.triangleIndex); componentIndices.push(ownerComponent.componentIndex);
  }
  if (points.length === 0) fail('no interior voxel centers at this spacing');
  const pointCount = points.length / 3, volumePerPoint = spacing ** 3;
  if (!Number.isFinite(Math.fround(volumePerPoint)) || Math.fround(volumePerPoint) <= 0) fail('voxel volume exceeds supported Float32 range');
  diagnostics.candidateCount = candidateCount; diagnostics.pointCount = pointCount;
  diagnostics.worldBounds = worldBounds; diagnostics.sampleBounds = bounds; diagnostics.worldToSimulation = transform;
  diagnostics.spacing = spacing; diagnostics.volumePerPoint = volumePerPoint; diagnostics.representedVolume = pointCount * volumePerPoint;
  diagnostics.elapsedMs = now() - started;
  return { points: new Float32Array(points), colors: new Float32Array(colors), masses: new Float32Array(pointCount).fill(volumePerPoint), volumes: new Float32Array(pointCount).fill(volumePerPoint),
    pointCount, spacing, materialProvenance: { meshIndices: new Uint32Array(meshIndices), materialIndices: new Uint32Array(materialIndices), triangleIndices: new Uint32Array(triangleIndices), componentIndices: new Uint32Array(componentIndices) }, diagnostics };
}
