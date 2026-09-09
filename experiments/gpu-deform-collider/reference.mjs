/**
 * Independent CPU oracle for the GPU-deformed collider experiment.
 *
 * No Three.js or production collision/refit code is imported. Vertex and query
 * input arrays contain xyz in Float32Array vec4 records; w is ignored. Indices
 * are a flat Uint32Array, three entries per triangle, in the PACKED BVH order.
 * Arithmetic here is JavaScript double precision over actual GPU-readback data.
 * Distances are unsigned: triangle winding does not establish inside/outside.
 *
 * Packed-node interpretation is pinned to three-mesh-bvh commit
 * 8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab, src/webgpu/tsl/structs.js and
 * src/webgpu/utils/packBVHBufferUtils.js. This oracle supports one identity
 * transform slot, including multiple cluster subtrees. The caller must verify
 * that transform slot 0 is actually identity; node words alone cannot prove it.
 *
 * Run the CPU-only known-answer checks: node reference.mjs --self-test
 */
const NONE = 0xffffffff;
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const xyz = (array, vertex) => Array.from(array.subarray(vertex * 4, vertex * 4 + 3));

function finitePoint(point, name) {
  if (!point || point.length < 3 || ![point[0], point[1], point[2]].every(Number.isFinite)) {
    throw new TypeError(`${name} must contain three finite coordinates`);
  }
}

function meshInputs(vertices, indices) {
  if (!(vertices instanceof Float32Array) || vertices.length % 4) throw new TypeError('vertices must be a Float32Array of vec4 records');
  if (!(indices instanceof Uint32Array) || indices.length % 3) throw new TypeError('indices must be a Uint32Array of triangle triples');
  for (let v = 0; v < vertices.length / 4; v++) finitePoint(vertices.subarray(v * 4, v * 4 + 3), `vertex ${v}`);
  for (const index of indices) if (index >= vertices.length / 4) throw new RangeError(`vertex index ${index} is out of range`);
}

function result(point, a, b, c, weights) {
  const closestPoint = [0, 1, 2].map(k => a[k] * weights[0] + b[k] * weights[1] + c[k] * weights[2]);
  const difference = sub(point, closestPoint);
  return { closestPoint, distanceSq: dot(difference, difference), barycoord: weights };
}

function edgeResult(point, a, b, c, from, to) {
  const corners = [a, b, c];
  const edge = sub(corners[to], corners[from]);
  const lengthSq = dot(edge, edge);
  const fraction = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, dot(sub(point, corners[from]), edge) / lengthSq));
  const weights = [0, 0, 0];
  weights[from] = 1 - fraction;
  weights[to] = fraction;
  return result(point, a, b, c, weights);
}

function degenerateResult(point, a, b, c) {
  const candidates = [edgeResult(point, a, b, c, 0, 1), edgeResult(point, a, b, c, 1, 2), edgeResult(point, a, b, c, 2, 0)];
  return candidates.reduce((best, candidate) => candidate.distanceSq < best.distanceSq ? candidate : best);
}

/**
 * Ericson's Voronoi-region triangle test, independently implemented here.
 * Collapsed/collinear triangles reduce to the closest of their three segments.
 * At machine-scale near-degeneracy (altitude roughly < 8 double eps * edge),
 * the same segment fallback avoids unstable barycentric division. This is a
 * numerical convention, not a claim of exact arithmetic for arbitrary inputs.
 */
export function closestPointTriangle(point, a, b, c) {
  [point, a, b, c].forEach((p, i) => finitePoint(p, ['point', 'a', 'b', 'c'][i]));
  const ab = sub(b, a), ac = sub(c, a), bc = sub(c, b);
  const normal = cross(ab, ac);
  const longestSq = Math.max(dot(ab, ab), dot(ac, ac), dot(bc, bc));
  if (dot(normal, normal) <= 64 * Number.EPSILON ** 2 * longestSq ** 2) return degenerateResult(point, a, b, c);
  const ap = sub(point, a);
  const d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return result(point, a, b, c, [1, 0, 0]);
  const bp = sub(point, b);
  const d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return result(point, a, b, c, [0, 1, 0]);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return result(point, a, b, c, [1 - v, v, 0]);
  }
  const cp = sub(point, c);
  const d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return result(point, a, b, c, [0, 0, 1]);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return result(point, a, b, c, [1 - w, 0, w]);
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return result(point, a, b, c, [0, 1 - w, w]);
  }
  const denominator = va + vb + vc;
  if (!(denominator > 0)) return degenerateResult(point, a, b, c);
  const v = vb / denominator, w = vc / denominator;
  return result(point, a, b, c, [1 - v - w, v, w]);
}

/**
 * Exhaustive queries; no BVH, normal-side heuristic, cutoff, or vertex-only
 * approximation. Exact ties keep the lowest packed triangle ID. Other equally
 * close triangles may be valid GPU results: compare point/distance and verify
 * the returned triangle rather than requiring IDs to agree at shared edges.
 * Empty meshes return ID 0xffffffff, distance Infinity, and NaN point/barycoord.
 */
export function bruteForceQueries(vertices, indices, queries) {
  meshInputs(vertices, indices);
  if (!(queries instanceof Float32Array) || queries.length % 4) throw new TypeError('queries must be a Float32Array of vec4 records');
  const queryCount = queries.length / 4, triangleCount = indices.length / 3;
  const closestPoints = new Float64Array(queryCount * 4).fill(NaN);
  const distanceSq = new Float64Array(queryCount).fill(Infinity);
  const triangleIDs = new Uint32Array(queryCount).fill(NONE);
  const barycoords = new Float64Array(queryCount * 3).fill(NaN);
  const triangles = Array.from({ length: triangleCount }, (_, t) => [xyz(vertices, indices[t * 3]), xyz(vertices, indices[t * 3 + 1]), xyz(vertices, indices[t * 3 + 2])]);
  for (let q = 0; q < queryCount; q++) {
    const point = xyz(queries, q);
    finitePoint(point, `query ${q}`);
    for (let t = 0; t < triangleCount; t++) {
      const candidate = closestPointTriangle(point, ...triangles[t]);
      if (candidate.distanceSq < distanceSq[q]) {
        distanceSq[q] = candidate.distanceSq;
        triangleIDs[q] = t;
        closestPoints.set(candidate.closestPoint, q * 4);
        barycoords.set(candidate.barycoord, q * 3);
      }
    }
    closestPoints[q * 4 + 3] = distanceSq[q];
  }
  return { queryCount, triangleCount, closestPoints, distanceSq, triangleIDs, barycoords };
}

/** Compare ALL packed metadata words, including currently unreachable nodes. */
export function validateMetadataUnchanged(beforeU32, afterU32) {
  const errors = [];
  if (!(beforeU32 instanceof Uint32Array) || !(afterU32 instanceof Uint32Array)) throw new TypeError('node buffers must be Uint32Array views');
  if (beforeU32.length !== afterU32.length || beforeU32.length % 8) errors.push('node buffer length mismatch or incomplete 8-word record');
  let mismatches = 0;
  for (let word = 0; word + 7 < Math.min(beforeU32.length, afterU32.length); word += 8) {
    for (const field of [6, 7]) if (beforeU32[word + field] !== afterU32[word + field]) {
      mismatches++;
      if (errors.length < 64) errors.push(`node ${word / 8} metadata word ${field} changed`);
    }
  }
  return { ok: errors.length === 0, errors, mismatches };
}

/**
 * Validate bounds independently from the refit procedure. Descendant triangle
 * extents are accumulated from actual vertices, then tested against EVERY
 * ancestor's six float bounds. Child-box containment is also checked, but can
 * be diagnostic-only for independently mapped/padded interval bounds by setting
 * requireChildContainment:false. This NEVER disables descendant-triangle tests.
 * Triangle leaves must be decoded before TLAS leaves because 0xffff also has
 * top byte 0xff. Internal children are node+1 and node+word6; TLAS word6 is an
 * absolute node index. Mesh word6 is a triangle offset, not an index offset.
 *
 * Returns errors (first 64), total errorCount, node/leaf/triangle counts and
 * unreferencedTriangleCount. Unreferenced index-buffer triangles are reported,
 * not automatically rejected (the input can contain unused packed capacity).
 * Default absolute tolerance is zero: GPU min/max over the same f32 positions
 * should enclose them exactly. This routine checks identity transform slot 0
 * tags; caller must independently check the transform buffer itself.
 */
export function validateBounds(nodeU32, vertices, indices, { root = 0, tolerance = 0, requireChildContainment = true } = {}) {
  meshInputs(vertices, indices);
  if (!(nodeU32 instanceof Uint32Array) || nodeU32.length % 8) throw new TypeError('nodes must be a Uint32Array of 8-word records');
  if (!Number.isFinite(tolerance) || tolerance < 0) throw new RangeError('tolerance must be finite and nonnegative');
  if (typeof requireChildContainment !== 'boolean') throw new TypeError('requireChildContainment must be boolean');
  const floats = new Float32Array(nodeU32.buffer, nodeU32.byteOffset, nodeU32.length);
  const errors = [], childContainmentErrors = [], active = new Set(), visited = new Set(), memo = new Map(), triangles = new Set();
  let errorCount = 0, childContainmentErrorCount = 0, leafCount = 0, tlasLeafCount = 0, maxDepth = 0;
  const error = text => { errorCount++; if (errors.length < 64) errors.push(text); };
  const childError = text => {
    childContainmentErrorCount++;
    if (childContainmentErrors.length < 64) childContainmentErrors.push(text);
    if (requireChildContainment) error(text);
  };
  const empty = () => [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  const merge = (a, b) => { for (let k = 0; k < 3; k++) { a[k] = Math.min(a[k], b[k]); a[k + 3] = Math.max(a[k + 3], b[k + 3]); } return a; };
  function contains(bounds, actual, label, report = error) {
    if (actual[0] > actual[3]) return;
    for (let k = 0; k < 3; k++) if (actual[k] < bounds[k] - tolerance || actual[k + 3] > bounds[k + 3] + tolerance) report(`${label} axis ${k} escapes bounds`);
  }
  function visit(node, depth) {
    if (!Number.isSafeInteger(node) || node < 0 || node >= nodeU32.length / 8) { error(`invalid node index ${node}`); return empty(); }
    if (active.has(node)) { error(`cycle at node ${node}`); return empty(); }
    if (memo.has(node)) return memo.get(node);
    // Deliberately fail malformed extreme topology before exhausting JS stack.
    if (depth > 1024) { error('tree depth exceeds oracle limit 1024'); return empty(); }
    active.add(node); visited.add(node); maxDepth = Math.max(maxDepth, depth);
    const offset = node * 8, info = nodeU32[offset + 7], pointer = nodeU32[offset + 6];
    const bounds = Array.from(floats.subarray(offset, offset + 6));
    if (!bounds.every(Number.isFinite)) error(`node ${node} has nonfinite bounds`);
    const actual = empty();
    const children = [];
    if ((info >>> 16) === 0xffff) {
      leafCount++;
      const count = info & 0xffff;
      if (pointer + count > indices.length / 3) error(`node ${node} triangle range is out of bounds`);
      else for (let t = pointer; t < pointer + count; t++) {
        triangles.add(t);
        for (let corner = 0; corner < 3; corner++) {
          const vertex = indices[t * 3 + corner] * 4;
          for (let k = 0; k < 3; k++) { actual[k] = Math.min(actual[k], vertices[vertex + k]); actual[k + 3] = Math.max(actual[k + 3], vertices[vertex + k]); }
        }
      }
    } else if ((info >>> 24) === 0xff) {
      tlasLeafCount++;
      if ((info & 0x00ffffff) !== 0) error(`node ${node} uses unsupported transform slot ${info & 0x00ffffff}`);
      children.push(pointer);
    } else {
      if (info > 2) error(`node ${node} has invalid split-axis metadata ${info}`);
      if (pointer < 2) error(`node ${node} has invalid relative right-child offset ${pointer}`);
      children.push(node + 1, node + pointer);
    }
    for (const child of children) {
      merge(actual, visit(child, depth + 1));
      if (Number.isSafeInteger(child) && child >= 0 && child < nodeU32.length / 8) contains(bounds, Array.from(floats.subarray(child * 8, child * 8 + 6)), `node ${node} child ${child}`, childError);
    }
    if (actual[0] <= actual[3]) {
      if ([0, 1, 2].some(k => bounds[k] > bounds[k + 3])) error(`nonempty node ${node} has inverted bounds`);
      contains(bounds, actual, `node ${node} descendant triangles`);
    }
    active.delete(node); memo.set(node, actual);
    return actual;
  }
  visit(root, 1);
  return { ok: errorCount === 0, errors, errorCount, requireChildContainment, childContainmentErrors, childContainmentErrorCount, nodeCount: nodeU32.length / 8, visitedNodeCount: visited.size, leafCount, tlasLeafCount, maxDepth, referencedTriangleCount: triangles.size, unreferencedTriangleCount: indices.length / 3 - triangles.size };
}

export function runSelfTests() {
  // Keep this oracle browser-importable; assertions are only for these tests.
  const assert = {
    ok(value, message = 'self-test assertion failed') { if (!value) throw new Error(message); },
    equal(actual, expected) { if (!Object.is(actual, expected)) throw new Error(`self-test: ${actual} != ${expected}`); },
  };
  let checks = 0;
  const near = (got, expected, tolerance = 1e-12) => { checks++; assert.ok(Math.abs(got - expected) <= tolerance, `${got} != ${expected}`); };
  const pointNear = (got, expected) => expected.forEach((value, i) => near(got[i], value));
  const a = [0, 0, 0], b = [2, 0, 0], c = [0, 2, 0];
  for (const [point, expected, distance] of [
    [[0.5, 0.5, 3], [0.5, 0.5, 0], 9], // interior
    [[1, -1, 0], [1, 0, 0], 1], // AB
    [[-1, 1, 0], [0, 1, 0], 1], // AC
    [[2, 2, 0], [1, 1, 0], 2], // BC
    [[-1, -1, 0], a, 2], [[3, -1, 0], b, 2], [[-1, 3, 0], c, 2],
    [[0.5, 0.5, -3], [0.5, 0.5, 0], 9], // opposite normal side, still unsigned
  ]) {
    const r = closestPointTriangle(point, a, b, c);
    pointNear(r.closestPoint, expected); near(r.distanceSq, distance); near(r.barycoord.reduce((x, y) => x + y), 1);
    near(closestPointTriangle(point, a, c, b).distanceSq, distance); // reversed winding
  }
  pointNear(closestPointTriangle([1, 2, 0], a, [1, 0, 0], b).closestPoint, [1, 0, 0]);
  pointNear(closestPointTriangle([2, 3, 4], a, a, a).closestPoint, a);
  pointNear(closestPointTriangle([1, 1, 0], a, a, b).closestPoint, [1, 0, 0]);
  const vertices = new Float32Array([0, 0, 0, 1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1]);
  const indices = new Uint32Array([0, 1, 2, 0, 1, 3]);
  const brute = bruteForceQueries(vertices, indices, new Float32Array([0.5, 0.5, 3, 0, 1, 0, 0, 0]));
  near(brute.distanceSq[0], 1.5); // closest at top vertex of vertical triangle
  assert.equal(brute.triangleIDs[1], 0); checks++;
  // Closed box: interior points still have positive distance to the surface.
  const box = new Float32Array([-1,-1,-1,1, 1,-1,-1,1, 1,1,-1,1, -1,1,-1,1, -1,-1,1,1, 1,-1,1,1, 1,1,1,1, -1,1,1,1]);
  const boxIndices = new Uint32Array([0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0]);
  const boxAnswers = bruteForceQueries(box, boxIndices, new Float32Array([0,0,0,0, 2,0,0,0, 2,2,2,0]));
  [1, 1, 3].forEach((value, i) => near(boxAnswers.distanceSq[i], value));
  // A coarse sphere-like surface must be tested as triangles, not as a sphere.
  const octahedron = new Float32Array([1,0,0,1, -1,0,0,1, 0,1,0,1, 0,-1,0,1, 0,0,1,1, 0,0,-1,1]);
  const octaIndices = new Uint32Array([0,2,4,2,1,4,1,3,4,3,0,4,2,0,5,1,2,5,3,1,5,0,3,5]);
  const octaAnswers = bruteForceQueries(octahedron, octaIndices, new Float32Array([0,0,0,0, 2,0,0,0]));
  near(octaAnswers.distanceSq[0], 1 / 3); near(octaAnswers.distanceSq[1], 1);
  const nodes = new Uint32Array(4 * 8), f32 = new Float32Array(nodes.buffer);
  for (let n = 0; n < 4; n++) f32.set([0, 0, 0, 2, 2, 2], n * 8);
  nodes[6] = 1; nodes[7] = 0xff000000; // absolute TLAS -> cluster root
  nodes[14] = 2; nodes[15] = 0; // relative internal -> left2,right3
  nodes[22] = 0; nodes[23] = 0xffff0001;
  nodes[30] = 1; nodes[31] = 0xffff0001;
  assert.equal(validateBounds(nodes, vertices, indices).ok, true); checks++;
  for (const node of [0, 1, 2, 3]) { // stale leaf, cluster parent, or TLAS must fail
    const stale = nodes.slice(); new Float32Array(stale.buffer)[node * 8 + 3] = 0.5;
    assert.equal(validateBounds(stale, vertices, indices).ok, false); checks++;
    // Disabling padded-box nesting must still catch a too-tight interior ancestor.
    assert.equal(validateBounds(stale, vertices, indices, { requireChildContainment: false }).ok, false); checks++;
  }
  const looseChild = nodes.slice(); new Float32Array(looseChild.buffer)[2 * 8 + 3] = 3;
  assert.equal(validateBounds(looseChild, vertices, indices).ok, false); checks++;
  const intervalValidation = validateBounds(looseChild, vertices, indices, { requireChildContainment: false });
  assert.equal(intervalValidation.ok, true); assert.ok(intervalValidation.childContainmentErrorCount > 0); checks += 2;
  const changed = nodes.slice(); changed[22] = 1;
  assert.equal(validateMetadataUnchanged(nodes, changed).ok, false); checks++;
  const boundsOnly = nodes.slice(); new Float32Array(boundsOnly.buffer)[0] = -1;
  assert.equal(validateMetadataUnchanged(nodes, boundsOnly).ok, true); checks++;
  const cycle = nodes.slice(); cycle[6] = 0;
  assert.equal(validateBounds(cycle, vertices, indices).ok, false); checks++;
  const otherTransform = nodes.slice(); otherTransform[7] = 0xff000001;
  assert.equal(validateBounds(otherTransform, vertices, indices).ok, false); checks++;
  const padded = new Uint32Array(nodes.length + 8); padded.set(nodes, 8);
  assert.equal(validateBounds(padded.subarray(8), vertices, indices).ok, true); checks++;
  const emptyAnswers = bruteForceQueries(new Float32Array(), new Uint32Array(), new Float32Array([0,0,0,0]));
  assert.equal(emptyAnswers.triangleIDs[0], NONE); assert.equal(emptyAnswers.distanceSq[0], Infinity); checks += 2;
  return { status: 'passed', checks, kind: 'gpu-deform-collider-cpu-reference', packedSourceRevision: '8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab', signedDistance: false };
}

if (typeof process !== 'undefined' && process.versions?.node && process.argv[1]) {
  const ownPath = decodeURIComponent(new URL(import.meta.url).pathname).replace(/^\/(?=[A-Za-z]:\/)/, '');
  const executedPath = process.argv[1].replaceAll('\\', '/');
  if (ownPath === executedPath || (process.platform === 'win32' && ownPath.toLowerCase() === executedPath.toLowerCase())) {
    if (!process.argv.includes('--self-test')) throw new Error('Use --self-test; this module performs no GPU work.');
    console.log(JSON.stringify(runSelfTests(), null, 2));
  }
}
