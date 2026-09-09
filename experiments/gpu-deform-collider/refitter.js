// Fixed-topology refit for three-mesh-bvh 0.9.15 / commit
// 8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab. This is the standard
// leaf-bounds / bottom-up-union algorithm, not a new BVH construction method.
// Layout source: src/webgpu/tsl/structs.js and utils/packBVHBufferUtils.js.

const WORDS_PER_NODE = 8;
const MAX_TRIANGLES = 4096;
const MAX_NODES = MAX_TRIANGLES * 4;
const MAX_DEPTH = 32;
const COPY_CHILD = 0xffffffff;
const WORKGROUP_SIZE = 64;

function requireUint32(array, name) {
  if (!(array instanceof Uint32Array)) throw Error(`${name} must be a Uint32Array.`);
}

/**
 * Parse the immutable upstream packing ONCE, on the CPU. No position values are
 * read. Node zero is the TLAS root. TLAS leaves point to absolute BLAS node IDs;
 * internal right children use relative node offsets. All transform slots must
 * be zero. Multiple clusters belonging to that one mesh are supported.
 *
 * Returns leafNodes = [{node, offset, count}] and deepest-first levels, each an
 * array of {node, left, right, kind}; kind is 'union' or 'copy' (a TLAS leaf).
 * The graph must be a single, fully reachable tree and cover every packed
 * triangle exactly once. Empty trees, shared BLAS instances, malformed packing,
 * more than 4096 triangles, and paths deeper than 32 nodes are rejected.
 */
export function buildRefitPlan({nodeArray, indexArray, vertexCount, vertexStrideFloats = 4}) {
  requireUint32(nodeArray, 'nodeArray');
  requireUint32(indexArray, 'indexArray');
  if (vertexStrideFloats !== 4) throw Error('Only a single vec4f position attribute is supported.');
  if (!Number.isInteger(vertexCount) || vertexCount < 1 || vertexCount > MAX_TRIANGLES * 3) {
    throw Error('vertexCount must be between 1 and 12288.');
  }
  if (!nodeArray.length || nodeArray.length % WORDS_PER_NODE) throw Error('Invalid 32-byte node packing.');
  const nodeCount = nodeArray.length / WORDS_PER_NODE;
  if (nodeCount > MAX_NODES) throw Error('Node count exceeds the bounded canary limit.');
  const triangleCount = indexArray.length / 3;
  if (!Number.isInteger(triangleCount) || triangleCount < 1 || triangleCount > MAX_TRIANGLES) {
    throw Error('The packed index buffer must contain 1–4096 complete triangles.');
  }
  for (const vertex of indexArray) if (vertex >= vertexCount) throw Error('Triangle index exceeds vertexCount.');

  const visited = new Uint8Array(nodeCount);
  const covered = new Uint8Array(triangleCount);
  const leafNodes = [];
  const parentsByDepth = [];
  let maxDepth = 0;
  let tlasLeafCount = 0;

  function visit(node, depth, region) {
    if (!Number.isInteger(node) || node < 0 || node >= nodeCount) throw Error('Child node is outside the packed buffer.');
    if (depth > MAX_DEPTH) throw Error('BVH depth exceeds 32 nodes.');
    if (visited[node]) throw Error('Cyclic or shared node references are unsupported.');
    visited[node] = 1;
    maxDepth = Math.max(maxDepth, depth);
    const word = node * WORDS_PER_NODE;
    const data = nodeArray[word + 6];
    const tag = nodeArray[word + 7];
    const high = tag >>> 16;
    if (high === 0) {
      if (tag > 2 || data < 2) throw Error('Invalid internal node split axis or right-child offset.');
      const left = node + 1, right = node + data;
      (parentsByDepth[depth] ??= []).push({node, left, right, kind: 'union'});
      visit(left, depth + 1, region);
      visit(right, depth + 1, region);
    } else if (region === 'tlas') {
      if (tag !== 0xff000000) throw Error('Only TLAS leaves for transform slot zero are supported.');
      tlasLeafCount++;
      (parentsByDepth[depth] ??= []).push({node, left: data, right: COPY_CHILD, kind: 'copy'});
      visit(data, depth + 1, 'blas');
    } else {
      if (high !== 0xffff) throw Error('Invalid BLAS triangle-leaf tag.');
      const count = tag & 0xffff;
      if (!count || data + count > triangleCount) throw Error('Invalid triangle leaf range.');
      for (let triangle = data; triangle < data + count; triangle++) {
        if (covered[triangle]) throw Error('Triangle ranges overlap.');
        covered[triangle] = 1;
      }
      leafNodes.push({node, offset: data, count});
    }
  }
  visit(0, 1, 'tlas');
  if (!tlasLeafCount || visited.some(value => value === 0)) throw Error('The node buffer contains unreachable nodes or no TLAS leaves.');
  if (covered.some(value => value === 0)) throw Error('The leaves do not cover every packed triangle.');
  const levels = parentsByDepth.filter(Boolean).reverse();
  // Freeze the CPU schedule: callers can inspect it without changing dispatches.
  for (const leaf of leafNodes) Object.freeze(leaf);
  for (const level of levels) { for (const job of level) Object.freeze(job); Object.freeze(level); }
  return Object.freeze({leafNodes: Object.freeze(leafNodes), levels: Object.freeze(levels), nodeCount, triangleCount, vertexCount, maxDepth, tlasLeafCount});
}

function validateIdentityTransforms(transformArray) {
  requireUint32(transformArray, 'transformArray');
  // Upstream pads the transform buffer to at least two 36-word entries.
  if (transformArray.length !== 72) throw Error('Expected one transform plus the upstream zero padding entry.');
  const floats = new Float32Array(transformArray.buffer, transformArray.byteOffset, transformArray.length);
  for (let matrix = 0; matrix < 2; matrix++) for (let i = 0; i < 16; i++) {
    if (floats[matrix * 16 + i] !== (i % 5 === 0 ? 1 : 0)) throw Error('Only identity world and inverse-world transforms are supported.');
  }
  if (transformArray[32] !== 1) throw Error('The sole mesh must be visible.');
  for (let i = 33; i < 72; i++) if (transformArray[i] !== 0) throw Error('Additional transforms or nonzero transform padding are unsupported.');
}

const leafShader = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> triangles: array<u32>;
@group(0) @binding(2) var<storage, read> vertices: array<vec4f>;
@group(0) @binding(3) var<storage, read> jobs: array<vec4u>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3u) {
  if (invocation.x >= arrayLength(&jobs)) { return; }
  let job = jobs[invocation.x];
  var lower = vec3f(3.402823e38);
  var upper = vec3f(-3.402823e38);
  for (var t = job.y; t < job.y + job.z; t++) {
    for (var corner = 0u; corner < 3u; corner++) {
      let point = vertices[triangles[3u * t + corner]].xyz;
      lower = min(lower, point);
      upper = max(upper, point);
    }
  }
  // The final two words contain traversal metadata: never write them.
  for (var axis = 0u; axis < 3u; axis++) {
    nodes[8u * job.x + axis] = bitcast<u32>(lower[axis]);
    nodes[8u * job.x + 3u + axis] = bitcast<u32>(upper[axis]);
  }
}`;

const parentShader = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> jobs: array<vec4u>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3u) {
  if (invocation.x >= arrayLength(&jobs)) { return; }
  let job = jobs[invocation.x];
  for (var axis = 0u; axis < 3u; axis++) {
    var lower = bitcast<f32>(nodes[8u * job.y + axis]);
    var upper = bitcast<f32>(nodes[8u * job.y + 3u + axis]);
    if (job.z != 0xffffffffu) {
      lower = min(lower, bitcast<f32>(nodes[8u * job.z + axis]));
      upper = max(upper, bitcast<f32>(nodes[8u * job.z + 3u + axis]));
    }
    nodes[8u * job.x + axis] = bitcast<u32>(lower);
    nodes[8u * job.x + 3u + axis] = bitcast<u32>(upper);
  }
}`;

/**
 * Borrow already-created native upstream GPU buffers. The caller supplies one
 * Mesh, identity TLAS root/mesh transforms, one vec4f position attribute, and
 * fixed topology. `transformArray` is the upstream Uint32Array CPU snapshot.
 * All CPU snapshots must correspond to the borrowed GPU buffers at construction.
 *
 * Deformation must write FINITE xyz positions into vertexBuffer before encode().
 * The fourth float is ignored. Submit deformation, encode(), then queries in that
 * order on the same device queue. encode() only records commands; it does not
 * submit, allocate, map, read back, or touch CPU position arrays.
 *
 * Do not subsequently call BVHComputeData.update()/updateTransforms(), set the
 * borrowed attributes' needsUpdate, mutate transforms/indices, or replace their
 * buffers. Those operations invalidate this fixed schedule or overwrite GPU
 * bounds. Recreate the refitter after any topology/buffer change.
 */
export class GPUBVHRefitter {
  constructor({device, nodeBuffer, indexBuffer, vertexBuffer, nodeArray, indexArray,
    vertexCount, vertexStrideFloats = 4, transformArray}) {
    const plan = buildRefitPlan({nodeArray, indexArray, vertexCount, vertexStrideFloats});
    validateIdentityTransforms(transformArray);
    if (!device || typeof device.createComputePipeline !== 'function') throw Error('Provide an initialized GPUDevice.');
    const borrowed = [nodeBuffer, indexBuffer, vertexBuffer];
    if (new Set(borrowed).size !== 3) throw Error('Node, index, and vertex buffers must be distinct.');
    for (const [i, [name, bytes]] of [['nodeBuffer', nodeArray.byteLength], ['indexBuffer', indexArray.byteLength], ['vertexBuffer', vertexCount * 16]].entries()) {
      const buffer = borrowed[i];
      if (!buffer || buffer.size < bytes || (buffer.usage & GPUBufferUsage.STORAGE) === 0 || buffer.mapState !== 'unmapped') {
        throw Error(`${name} must be an unmapped STORAGE buffer with at least ${bytes} bytes.`);
      }
    }
    if (device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE || device.limits.maxStorageBuffersPerShaderStage < 4) {
      throw Error('Device limits cannot support the bounded refit kernels.');
    }
    this.plan = plan;
    this._owned = [];
    this._disposed = false;
    this._dispatches = [];
    const jobsBuffer = (jobs, label) => {
      const buffer = device.createBuffer({label, size: jobs.length * 16, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true});
      this._owned.push(buffer);
      const words = new Uint32Array(buffer.getMappedRange());
      for (let i = 0; i < jobs.length; i++) words.set(jobs[i], i * 4);
      buffer.unmap();
      return buffer;
    };
    try {
      const leaves = jobsBuffer(plan.leafNodes.map(({node, offset, count}) => [node, offset, count, 0]), 'refit-leaf-schedule');
      const leafPipeline = device.createComputePipeline({label: 'refit-leaf-bounds', layout: 'auto', compute: {
        module: device.createShaderModule({label: 'refit-leaf-bounds', code: leafShader}), entryPoint: 'main',
      }});
      const parentPipeline = device.createComputePipeline({label: 'refit-parent-bounds', layout: 'auto', compute: {
        module: device.createShaderModule({label: 'refit-parent-bounds', code: parentShader}), entryPoint: 'main',
      }});
      const leafBindings = device.createBindGroup({layout: leafPipeline.getBindGroupLayout(0), entries: [
        {binding: 0, resource: {buffer: nodeBuffer, size: nodeArray.byteLength}},
        {binding: 1, resource: {buffer: indexBuffer, size: indexArray.byteLength}},
        {binding: 2, resource: {buffer: vertexBuffer, size: vertexCount * 16}},
        {binding: 3, resource: {buffer: leaves}},
      ]});
      this._dispatches.push({pipeline: leafPipeline, bindings: leafBindings, count: plan.leafNodes.length});
      for (const [level, jobs] of plan.levels.entries()) {
        const buffer = jobsBuffer(jobs.map(({node, left, right}) => [node, left, right, 0]), `refit-parent-schedule-${level}`);
        const bindings = device.createBindGroup({layout: parentPipeline.getBindGroupLayout(0), entries: [
          {binding: 0, resource: {buffer: nodeBuffer, size: nodeArray.byteLength}},
          {binding: 1, resource: {buffer}},
        ]});
        this._dispatches.push({pipeline: parentPipeline, bindings, count: jobs.length});
      }
      this.dispatchCount = this._dispatches.length;
    } catch (error) {
      this.dispose();
      throw error;
    }
  }

  encode(encoder) {
    if (this._disposed) throw Error('The refitter has been disposed.');
    if (!encoder || typeof encoder.beginComputePass !== 'function') throw Error('Provide an open GPUCommandEncoder.');
    // A separate pass per level establishes ordering between dependent bounds.
    for (const {pipeline, bindings, count} of this._dispatches) {
      const pass = encoder.beginComputePass({label: 'gpu-bvh-refit'});
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindings);
      pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP_SIZE));
      pass.end();
    }
    return this.dispatchCount;
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    for (const buffer of this._owned) buffer.destroy();
    this._owned.length = 0;
    this._dispatches.length = 0;
  }
}

if (typeof process !== 'undefined' && process.argv?.includes('--self-test')) {
  const assert = (await import('node:assert/strict')).default;
  // Two TLAS cluster leaves, one internal BLAS and one singleton BLAS.
  const nodeArray = new Uint32Array(7 * 8);
  const metadata = [[2, 0], [3, 0xff000000], [6, 0xff000000], [2, 1], [0, 0xffff0001], [1, 0xffff0001], [2, 0xffff0001]];
  metadata.forEach((pair, i) => nodeArray.set(pair, i * 8 + 6));
  const indexArray = Uint32Array.of(0, 1, 2, 2, 3, 0, 3, 4, 0);
  const inputs = {nodeArray, indexArray, vertexCount: 5};
  const before = nodeArray.slice();
  const plan = buildRefitPlan(inputs);
  assert.equal(plan.triangleCount, 3);
  assert.equal(plan.maxDepth, 4);
  assert.deepEqual(plan.leafNodes.map(job => job.node), [4, 5, 6]);
  assert.deepEqual(plan.levels.map(level => level.map(job => job.node)), [[3], [1, 2], [0]]);
  assert.deepEqual(nodeArray, before);
  for (const mutate of [
    nodes => { nodes[1 * 8 + 7] = 0xff000001; }, // extra transform
    nodes => { nodes[2 * 8 + 6] = 3; }, // shared subtree
    nodes => { nodes[3 * 8 + 6] = 99; }, // invalid child
    nodes => { nodes[5 * 8 + 6] = 0; }, // overlapping triangles
    nodes => { nodes[6 * 8 + 7] = 0xffff0000; }, // empty leaf
  ]) {
    const nodes = nodeArray.slice(); mutate(nodes);
    assert.throws(() => buildRefitPlan({...inputs, nodeArray: nodes}));
  }
  assert.throws(() => buildRefitPlan({...inputs, vertexCount: 4}));
  assert.throws(() => buildRefitPlan({...inputs, vertexStrideFloats: 8}));
  const transforms = new Uint32Array(72), floats = new Float32Array(transforms.buffer);
  for (let offset = 0; offset <= 16; offset += 16) for (let i = 0; i < 16; i += 5) floats[offset + i] = 1;
  transforms[32] = 1;
  validateIdentityTransforms(transforms);
  floats[12] = 1;
  assert.throws(() => validateIdentityTransforms(transforms));
  console.log('Refit plan CPU self-test passed (no GPU execution).');
}
