# Prior art and falsifiable scope

Audited 2026-09-09. This is a source audit and proposed experiment, not a passed benchmark. No GPU run, dependency installation or model download was performed for this audit.

The potentially useful contribution is a small adapter that refits the **current upstream `BVHComputeData` packed hierarchy** from GPU-authored vertex positions while retaining its existing query functions and rendering from those same positions. GPU BVH refitting, GPU skinning, GPU collision queries, and GPU-only Three.js geometry are already established. This experiment must not claim any of those as a first, or a new BVH algorithm.

## Current upstream capability and remaining gap

The inspected `gkjohnson/three-mesh-bvh` revision is [`8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab`](https://github.com/gkjohnson/three-mesh-bvh/commit/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab), committed 2026-09-09 at 12:56:22 UTC. Pin the actual dependency used by the experiment separately: this audit does not imply that every npm release contains this revision.

`BVHComputeData` already provides clustered TLAS/BLAS packing and GPU query construction through TSL, including closest-point queries and custom shapecasts. Its API is explicitly unstable and requires Three.js r185 or newer. Thus, adding GPU scene queries alone would repeat existing functionality. See the [pinned WebGPU API](https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/WEBGPU_API.md#L63).

The data update path remains materially different from GPU-authored deformation:

- [`update()` lines 219–228](https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/src/webgpu/BVHComputeData.js#L219) leaves geometry, skinning, and morph refit as TODO work. Lines 247–254 construct a new clustered hierarchy and dispose previous buffers; lines 367–430 pack CPU arrays into new storage attributes.
- [`updateTransforms()` lines 455–476](https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/src/webgpu/BVHComputeData.js#L455) updates the clustered top-level bounds and object transforms. It does not refit bottom-level triangles from a compute-written vertex buffer.
- [`appendGeometryData()` lines 338–389](https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/src/webgpu/utils/packBVHBufferUtils.js#L338) reads geometry attributes on the CPU, applies bone transforms there when applicable, and writes the packed attribute array.

The maintainer's open [issue #884](https://github.com/gkjohnson/three-mesh-bvh/issues/884), opened 2026-06-15, explicitly lists fast geometry/skinned-mesh updates and BLAS refitting as high-priority work. This is evidence of an acknowledged integration need, not evidence that nobody has implemented GPU refitting elsewhere.

## Strongest overlapping implementation: open PR #853

Jure Triglav's [draft PR #853](https://github.com/gkjohnson/three-mesh-bvh/pull/853), opened 2026-02-21, was still open at audit. Its inspected head is [`8bfafc6ff5c5b878bd0c9b25548a6d40bd5329ba`](https://github.com/jure/three-mesh-bvh/tree/8bfafc6ff5c5b878bd0c9b25548a6d40bd5329ba); GitHub reports its latest PR update as 2026-08-01.

This PR already includes an H-PLOC-like GPU BVH2 builder, refitting, input directly from GPU buffers, and full GPU skinning and exploding-mesh examples. Its `GPUMeshBVH` exposes [refit operations](https://github.com/jure/three-mesh-bvh/blob/8bfafc6ff5c5b878bd0c9b25548a6d40bd5329ba/src/gpu/GPUMeshBVH.js#L340) and [external position-buffer input](https://github.com/jure/three-mesh-bvh/blob/8bfafc6ff5c5b878bd0c9b25548a6d40bd5329ba/src/gpu/GPUMeshBVH.js#L512). See its [skinned-mesh example](https://github.com/jure/three-mesh-bvh/blob/8bfafc6ff5c5b878bd0c9b25548a6d40bd5329ba/example/webgpu_skinnedMesh.js).

The maintainer's February 27 discussion distinguishes its node layout from the CPU-compatible layout subsequently used by `BVHComputeData`, and identifies future interoperability as desirable. The present opportunity is therefore narrower: preserving the current upstream packed format and unchanged upstream closest-point traversal for an existing GPU-authored mesh. The draft is a credible alternative implementation, not something that can be ignored because it is unmerged. Its displayed early timings are explicitly corrected in the PR discussion as submission-only; do not copy them into a comparison.

The upstream path tracer also retains GPU BVH generation/refit in its [remaining-features issue #777](https://github.com/gkjohnson/three-gpu-pathtracer/issues/777). A broad browser-wide novelty claim remains unsupported.

## Other directions screened out

These sources prevent recycling neighboring capabilities as the finding:

- [Navcat](https://github.com/isaac-mason/navcat/tree/bc9d3c3f372a9a94cde9c8c2382baa35c1ebd25f) (revision dated 2026-07-09) includes dynamic navmeshes, crowds, flow fields, and Three.js helpers. Its [dynamic-navmesh example](https://github.com/isaac-mason/navcat/blob/bc9d3c3f372a9a94cde9c8c2382baa35c1ebd25f/examples/src/example-dynamic-navmesh.ts) combines Three WebGPU rendering with Rapier and tile rebuild scheduling. [Recast Navigation JS](https://github.com/isaac-mason/recast-navigation-js) already exposes tiled generation and dynamic-obstacle tile caches.
- [Rapier snapshots](https://rapier.rs/docs/user_guides/templates/serialization/) and [Jolt's rollback documentation](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md#rolling-back-a-simulation) already support state restoration; [JoltPhysics.js](https://jrouwe.github.io/JoltPhysics.js/) includes a snapshot demo. [Official MuJoCo WASM](https://github.com/google-deepmind/mujoco/blob/deb60af526451d588958e03833aea0459528ae1e/wasm/README.md) and [MuJoCo state APIs](https://mujoco.readthedocs.io/en/latest/programming/simulation.html#state) cover browser physics and explicit simulation state.
- [SunaEngine](https://github.com/DARIENBATHALTER/sunaengine/tree/a23cfe8cb64874e94aeaf01d141daacd323d4d42) (2026-08-16) already demonstrates integer WebGPU particle simulation with replay verification and rewind. Those are the author's documented capabilities; this audit did not independently run its tests.
- [Taichi.js](https://github.com/AmesingFlank/taichi.js/tree/3579e317c8a0af0f42fb311dbc886aed5c514594) supplies JavaScript-to-WebGPU computation; [DiffTaichi](https://github.com/taichi-dev/difftaichi) establishes differentiable physical simulation and controller optimization. These are different projects; do not infer that all Python Taichi autodiff facilities exist in Taichi.js.

The requested recent “polyengine” comparison could not be identified unambiguously from that name. The [PolyEngine website](https://poly-engine.github.io/) found by search is not sufficient evidence about a current GPU-skinning collision implementation. Treat that item as **unverified**, not as a negative prior-art result.

## Proposed minimal test and claim boundaries

Start with fixed topology and one identity-transformed mesh. Build the upstream hierarchy once, compute vertex deformation into one buffer, refit its packed BLAS and clustered TLAS bounds, then run the unchanged upstream GPU closest-point function and render from the same vertex data. Keep topology, indices, transforms, and query kernels fixed across comparison lanes. Initialization may inspect CPU metadata; steady-state deformation and refit must not read vertices back to the CPU.

Proposed correctness gates, declared before timing:

1. For selected frames, compare upstream-query results with a small brute-force GPU closest-triangle control and an independent CPU control reading the **actual GPU vertices**. Use predeclared absolute/relative tolerances for distances and points; triangle IDs may legitimately differ at equal-distance ties.
2. Include rest, large deformation outside the original bounds, asymmetric deformation, and reversal. A deliberately stale-bounds lane must fail a designed fixture. If it does not, the fixture cannot establish that refitting matters.
3. Verify every packed parent contains its children and every leaf contains its current triangles, including the clustered TLAS. Retain enough raw validation outputs and source hashes to reproduce the assertion. Do not equate a pleasing picture with numerical correctness.
4. Assert shared vertex-buffer identity and count readback/upload bytes per lane. Separate validation-only readbacks from steady-state work. Retain failures and WebGPU validation errors.

The strongest fair same-API comparator is GPU deformation followed by actual vertex readback, CPU refit of the existing topology, and upload of updated packed bounds, followed by the same GPU query/render path. Reuse allocations and packed storage in this comparator; rebuilding everything each frame would exaggerate the benefit. A matched CPU-authored deformation lane can show when this adapter is unnecessary, but cannot replace the comparator for GPU-only input. GPU brute force is a correctness control and a viable small-mesh alternative.

PR #853 remains the strongest alternative GPU implementation. Without running it under matched geometry, query counts, completion boundaries, and quality checks, report only improvement over the measured CPU bridge and compatibility with current upstream queries. Do not claim fastest GPU refit or superiority to existing GPU BVH implementations.

Measure completed end-to-end deformation→refit→query work, GPU timestamps when available, CPU blocking time, and transferred bytes separately. Use paired lane order and predeclared geometry/query scales; include warmup, dispatches, synchronization, and any intentional pauses consistently. Submission time is not GPU execution time. Exclude rendering only if it is identical and the exclusion is explicit. Keep the first screen bounded to modest geometry, fixed iteration counts, and automatic stop/disposal; do not escalate load merely to manufacture a large ratio.

Passing these gates would justify a reusable adapter/demo claim: **GPU-authored deformation can feed the current upstream Three.js BVH query representation without a per-frame CPU vertex round trip.** It would not establish full collision physics, continuous collision detection, arbitrary topology edits, arbitrary skinned assets, multi-object correctness beyond tested fixtures, stable public API compatibility, device portability, or algorithmic novelty. A supported performance claim additionally requires a repeatable benefit against the fair comparator at a useful workload; a null result must remain visible.
