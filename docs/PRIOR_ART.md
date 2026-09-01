# Prior art and positioning

GPU frustum culling, survivor compaction, indirect drawing, and retained command submission are established rendering techniques. This project does not claim to have invented them.

The narrower question is how those mechanisms behave through Three.js WebGPU APIs, how existing ecosystem implementations compare under equivalent workloads, and whether a fixed-ownership specialization or scheduling change produces a material, reproducible difference.

## Direct Three.js ecosystem work

- Three.js r185 [completed `InstancedMesh` support inside render bundles](https://github.com/mrdoob/three.js/pull/33839),
  including geometry updates during bundle replay. The experiments here build
  on that renderer capability and isolate compute-written indexed-indirect
  addressing and fixed-ownership layout choices; they do not claim the render-
  bundle integration itself.
- [Three Blocks instance culling](https://threejs-blocks.com/docs/blocks/instance-culling) provides GPU instance culling and compacted survivors for WebGPU.
- [Three Blocks `ComputeInstanceCulling`](https://threejs-blocks.com/docs/api/ComputeInstanceCulling) exposes indirect arguments, survivor data, optional sorting, diagnostic readback, and explicit camera/update methods.
- The [Three Blocks changelog](https://threejs-blocks.com/changelog) records the removal of its earlier indirect-batching implementation in version 0.11.0.
- The published `three-blocks@0.10.0` indirect-batching entry exposes `IndirectBatchedMesh`. Its public `enableInternalCulling(renderer)` and `updateInternalCulling(camera)` methods provide the historical heterogeneous-geometry comparison; the underlying `ComputeBatchCulling` class is not part of that public entry point.
- [InstancedMesh2](https://agargaro.github.io/instanced-mesh/) provides per-instance culling and sorting in a WebGL-focused implementation and belongs in a separate renderer lane.
- [Three.ez BatchedMesh extensions](https://github.com/agargaro/batched-mesh-extensions) provide CPU/BVH culling options, including a WebGPU build.

The published Three Blocks package uses the PolyForm Noncommercial 1.0.0 license. Current and historical comparisons use version-pinned packages as dependencies and begin from their published entry points. The coalesced probe and historical diagnostic readback access guarded, undocumented runtime fields and are identified separately below. No Three Blocks implementation source is copied or adapted into this repository, and package use remains subject to its license. The license URL and required attribution are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Three Blocks public baseline

The public baseline targets `three-blocks@0.11.0` with Three.js 0.185.1. At B geometry buckets it constructs B independent mesh-bound `ComputeInstanceCulling` instances, each with its own indexed geometry clone, node material, local matrix storage, and `InstancedMesh`.

The mesh-bound constructor installs the package's material/geometry integration and an automatic pre-render callback. The benchmark retains the integration but restores the previous callback, supplies camera uniforms explicitly, and calls the public `update()` method before cached bundle replay. This prevents duplicate work and exposes scheduling cost. It characterizes a best-controlled public use of the package, not the CPU behavior of its stock automatic hook.

After one-time initialization, each culler schedules four compute nodes in one `renderer.compute()` call. The 32-bucket baseline therefore performs 128 dispatches through 32 explicit Three.js calls per frame. Native commands and bucket-local survivor IDs are read through public diagnostic methods and validated against the independent global object-ID reference.

## Guarded coalesced scheduling probe

The coalesced lane uses exactly the same package cullers, options, scene inputs, output buffers, rendering path, and validation as the public baseline. It first initializes every culler through public `update()` calls. Steady-state frames then flatten the four runtime compute nodes from each culler into one stable array and submit that array in one `renderer.compute()` call.

The runtime node properties used for this step are not exposed by the stable `three-blocks/instance-culling` facade. The adapter pins the package version and requires every value to identify itself as a `ComputeNode`; it fails if the runtime shape changes. This is an undocumented research probe, not a Three Blocks feature, compatibility promise, or recommended integration.

A difference between the public and coalesced lanes is evidence about submission topology under this pinned implementation, including renderer-call and compute-pass/command-buffer grouping. It must not be attributed to different package nodes or dispatch work, because both lanes execute the same 128 package dispatches at 32 buckets. It also must not be described as stock package performance.

## Historical Three Blocks indirect batching

The historical lane pins the published `three-blocks@0.10.0` package and uses its stable `IndirectBatchedMesh` facade. Geometry registration, instance registration, matrix updates, culling enablement, steady-state `updateInternalCulling(camera)` calls, rendering, and primary mesh/culler disposal follow that public path. One mesh owns the heterogeneous population and records one indexed indirect draw per geometry into a cached bundle.

After its one-time initialization, the package schedules nine dispatches in one explicit Three.js compute call per frame. Its general pipeline performs block counting and prefix work, constructs geometry-specific survivor ranges, writes multi-indirect commands, and redistributes survivors by geometry. That makes it the direct historical ecosystem baseline for the fixed-ownership specialization, not merely another scheduling variant of the v0.11 cullers.

Version 0.10 has no public heterogeneous readback method. Untimed validation therefore uses guarded methods on the package-owned runtime culler to read all indirect commands and survivor IDs. The adapter pins the package version and fails when that diagnostic shape changes; this diagnostic access is not described as part of the public Three Blocks API. The harness likewise retains guarded, version-pinned storage attributes and compute nodes only for untimed post-trial cleanup, preventing stale Three.js r185 cache entries from accumulating across repetitions. That lifecycle cleanup is not part of the package's public facade and does not change the measured execution path.

The historical allocator writes nonzero `firstInstance` values for most geometry commands. Consequently, the lane requires the optional WebGPU `indirect-first-instance` feature and is explicitly unavailable when the adapter does not expose it. Fixed-slice's zero-`firstInstance` layout remains the portable design under comparison.

## Fixed-slice specialization

Fixed-slice uses established primitives in a narrower ownership model. Each geometry has a permanent survivor capacity and command slot. One dispatch resets all counts; one dispatch culls objects and appends global IDs directly into the owning geometry's slice. The immutable fixture ranges are concatenated into one indexed buffer with per-vertex bucket bases, allowing one mesh to retain all indirect offsets. Every command remains portable with `firstInstance` equal to zero and replays through a cached `BundleGroup`.

At 32 buckets, fixed-slice uses two dispatches in one explicit Three.js compute call and one render object containing 32 indexed indirect offsets. Unlike either one-submission package comparator, it changes compaction layout, geometry organization, material/vertex-transform wiring, and dispatch work; compared with v0.11 it also changes render-object count. Compute and render durations must be separated, and a render-pass difference cannot be assigned to compaction alone. A claim that fixed-slice is differentiated therefore requires direct, validated evidence against both the coalesced v0.11 probe and compatible historical v0.10 baseline, as well as against draw all; a win over the 32-call public schedule alone would establish only a scheduling opportunity.

## Indirect `firstInstance` specialization

Using nonzero indirect `firstInstance` values is established WebGPU practice and is already present in prior libraries, including the historical Three Blocks path above. This project does not claim the command-field technique as new. The narrower specialization removes a redundant per-vertex bucket-base input from the fixed-slice representation: the portable shader reads `visibleIds[bucketBase + instanceIndex]`, while the feature-gated shader reads `visibleIds[instanceIndex]` and receives the same base through the indirect command.

The controlled contribution is the causal measurement and Three.js integration boundary. The render-only crossover binds the two address paths to the same survivor list, common geometry payload, storage resources, material output, native draws, and balanced within-block schedule. Both aggregate estimates show a material-sized reduction on one device, but one of the two matrices fails preregistered nuisance-interaction bounds, so the declared confirmatory criterion is not met. The result therefore motivates a live compute-plus-render evaluation; it does not establish novelty of the primitive, a confirmed first-device result, cross-device generality, or a total-GPU deployment claim.

## Three.js, WebGPU, and browser constraints

- [`BufferGeometry.setIndirect`](https://threejs.org/docs/pages/BufferGeometry.html) allows application-provided indirect buffers and offsets.
- [`BundleGroup`](https://threejs.org/docs/pages/BundleGroup.html) exposes WebGPU render bundles through Three.js.
- Portable WebGPU indirect commands require `firstInstance` to remain zero unless the optional `indirect-first-instance` feature is available. The historical v0.10 lane requires the feature; other implemented indirect lanes retain zero.
- The core WebGPU API and Three.js r185 path used here do not provide a counted multi-draw operation equivalent to native `drawIndirectCount` workflows.
- The [Three.js r185 WebGPU timestamp-query pool](https://github.com/mrdoob/three.js/blob/r185/src/renderers/webgpu/utils/WebGPUTimestampQueryPool.js) is fixed at 2,048 queries per timestamp type. Each timed submission uses two queries. This is a Three.js version-specific measurement constraint, not a WebGPU limit.
- Per-frame timestamp joining reads version-pinned r185 backend pool metadata after the public resolution call. That instrumentation is not a stable Three.js timing API.
- In r185, `Renderer.compute()` passes an array-valued compute group to the
  backend timestamp-identifier path. The array has neither the single-node
  `isComputeNode` marker nor an `id`, so the default identifier is
  render-prefixed and contains an undefined context ID even though the WebGPU
  backend allocates its queries from the compute pool. The live crossover
  registers each exact compute-group identity with a unique lifecycle-bound
  context ID through a version-pinned wrapper, retains the resulting UID and
  duration pair for every frame, and rejects any type, context, frame, or lane
  mismatch. This attribution wrapper is measurement instrumentation, not a
  stable Three.js API.
- In the [r185 backend timestamp-identifier path](https://github.com/mrdoob/three.js/blob/r185/src/renderers/common/Backend.js#L474-L510),
  multiple array-valued `Renderer.compute()` calls within one frame receive a
  colliding identifier. Query slots are consumed for every call, but the
  identifier map retains only the last call's offset. A one-call lane avoids
  that collision but still lacks operation and lane attribution without the
  registration above; a valid multi-call aggregate needs unique per-call
  instrumentation or a Three.js change.
- High-resolution JavaScript CPU timing depends on browser security and precision policy. The harness requests cross-origin isolation with COOP/COEP headers, records the actual isolation state, and measures the observed `performance.now()` increment instead of assuming a resolution. The smoke gate and focused runner enforce the isolation and precision threshold.

The query-pool limit matters directly at 32 buckets: the public lane would request 19,200 queries during the unresolved 300-frame warmup and 15,360 during the 240-frame measurement block. Each one-submission lane requests 600 and 480 queries respectively, with resolution between phases. Capacity chunking alone would not repair the public lane's timestamp-identifier collision. The standard GPU-timestamp matrix therefore does not substitute coalesced or historical timings for the v0.11 public baseline; it reports each selected comparator under its own identity and keeps the public lane as a separate API/scheduling reference.

## Claim policy

Potential contributions are classified as one of:

- **established**: already demonstrated in prior systems or libraries;
- **replicated**: independently reproduced under a documented configuration;
- **differentiated**: directly compared and shown to have a material implementation distinction;
- **unverified**: plausible but not yet tested against the relevant baseline;
- **falsified**: contradicted by controlled evidence.

Any reported performance result must identify the exact lane, package and Three.js versions, object and bucket counts, visibility, compute dispatch and submission counts, device/backend, clock precision, absolute timing, validation status, and known limits. Evidence from one device or scene is not generalized to WebGPU as a whole.
