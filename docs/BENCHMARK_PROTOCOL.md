# Benchmark protocol

## Research question

Under which combinations of visibility, instance count, geometry cost, geometry diversity, camera motion, and device class does GPU visibility compaction improve a Three.js WebGPU workload after compute and CPU submission overhead are included?

## Implemented 32-bucket comparison

The active heterogeneous experiment uses 32 indexed geometry buckets. "Compute submissions" below are explicit Three.js `renderer.compute()` calls; "dispatches" are the compute nodes scheduled by those calls. Counts describe steady-state measured frames after one-time initialization.

| Lane | Compute dispatches | Compute submissions | Render objects | Purpose |
| --- | ---: | ---: | ---: | --- |
| Bundled draw all | 0 | 0 | 32 | No-culling retained-rendering reference |
| Three Blocks public baseline | 128 | 32 | 32 | Supported public `culler.update()` scheduling, one culler per bucket |
| Three Blocks coalesced probe | 128 | 1 | 32 | Same package cullers and dispatch work with a guarded, pinned scheduling change |
| Three Blocks historical baseline | 9 | 1 | 1 | Published v0.10 heterogeneous indirect-batching execution path |
| Fixed-slice | 2 | 1 | 1 | Shared reset-and-compact implementation with merged immutable fixture storage |
| Fixed-slice per-bucket control | 2 | 1 | 32 | Same fixed-slice compute and draw payload retained by one mesh/render object per bucket |

These are strategy-declared configuration values that the runner checks against each lane's expected schedule. They are not runtime counters observed from the GPU command stream.

The standard 240-frame GPU-timestamp matrix compares bundled draw all, one explicitly selected heterogeneous package comparator, and fixed-slice. `BENCHMARK_HETEROGENEOUS_COMPARATOR` accepts `coalesced-v11` (the default) or `historical-v10`; the resulting modes remain exactly draw all, the selected comparator, and fixed-slice. Historical v0.10 runs require `indirect-first-instance` and are rejected at startup when the page reports that the feature is unavailable. The 32-submission public v0.11 lane remains the authoritative public-API correctness and scheduling baseline, but it is excluded from that timestamp window because it exceeds Three.js r185's timestamp-query capacity and its per-call timestamp identifiers collide. It must not be silently replaced by the coalesced probe when describing package behavior.

Single-geometry trials continue to use the public Three Blocks v0.11 lane directly. The heterogeneous comparator selector applies to 4-, 32-, and 128-bucket runs. Four buckets provide the merge-aware control with exactly one asset from each topology family. Stock Three.js `BatchedMesh`, 512-bucket scaling, moving cameras, and production assets remain subsequent comparison stages rather than evidence already supplied by the 32-bucket matrix.

A separate `fixed-slice-representation` matrix isolates retained representation topology. It compares only fixed-slice per-bucket against fixed-slice, uses six alternating AB/BA repetitions, rotates the three visibility levels by repetition, and places each within-visibility pair adjacently. The runner and analyzer both require the exact 36-cell plan. One-bucket runs are equal-count negative controls; the scaling interpretation applies only when B is greater than one.

The separate `depth-ordering` matrix compares atomic fixed-slice compaction with eight-bin front-to-back and reverse traversal under high- and low-overlap layouts. Its fixed 65,536-object, 32-bucket design, reversed-depth requirement, exact offscreen parity gate, balanced 36-trial order, and decision thresholds are specified in [Coarse depth-ordering protocol](DEPTH_ORDERING_PROTOCOL.md).

## Controlled scene

The focused 32-bucket cells hold constant:

- Three.js 0.185.1 and pinned package versions;
- 32 independently allocated, unique indexed fixtures drawn from four procedural topology families: box, sphere, cylinder, and torus;
- a deterministic, small non-affine vertex deformation per bucket, with family topology and comparable bounds preserved;
- bucket-major, static, full translation/rotation/scale matrices;
- object counts selected from 4,096, 16,384, and 65,536;
- target visible fractions of approximately 20%, 80%, and 99%;
- a deterministic seed, fixed perspective projection, 1280 by 720 viewport, and device pixel ratio 1;
- matched PBR parameters and lighting;
- cached `BundleGroup` rendering for all retained-draw lanes;
- zero Three Blocks v0.11 frustum padding and disabled sorting; the historical v0.10 lane retains its published default padding of 0.2 in XY and at the far plane, with accepted and rejected placements separated far enough to preserve the predetermined membership.

Visibility changes through deterministic object placement rather than field-of-view changes, avoiding projected-coverage changes caused only by a different projection.

## Three Blocks public baseline

The baseline uses the published `three-blocks@0.11.0` mesh-bound `ComputeInstanceCulling` constructor once per bucket. Every bucket receives a dedicated indexed geometry clone, unique node material, local `StorageInstancedBufferAttribute`, `InstancedMesh`, and culler. Matrices are the corresponding bucket slice of the shared scenario, and meshes remain at identity transforms.

Construction installs the package's geometry/material integration and an automatic `onBeforeRender` update. The benchmark retains the integration but restores the mesh's prior hook, then explicitly calls the public `setCameraUniforms(camera)` and `update()` methods before rendering. This avoids duplicate culling, works with cached bundle replay, and makes submission cost observable.

Three Blocks v0.11.0 disposes its buffers but does not release the package-created Three.js compute nodes. To keep repeated trials from accumulating stale pipeline-cache entries, the harness retains a version-pinned, guarded list of those nodes and disposes them after the culler is disposed. This untimed lifecycle cleanup does not alter construction, validation, or steady-state execution, but it is an internal compatibility measure rather than part of the public package facade.

The first untimed update performs each culler's lazy initialization. With sorting disabled, every later culler update schedules four nodes: clear indirect arguments, clear visibility, select/pack survivors, and cap the count. Those nodes run in one `renderer.compute()` call per culler. For 32 buckets, the public baseline therefore performs 128 dispatches through 32 calls per frame.

This is the best-controlled public-API package integration. It is not a measurement of the package's stock automatic render hook, so no CPU-overhead claim about that stock hook may be inferred from this lane.

## Three Blocks coalesced scheduling probe

The coalesced probe reuses the same per-bucket construction, options, buffers, rendering, validation, and disposal as the public baseline. Its first untimed submit invokes every public `culler.update()` once, ensuring package initialization and an initial valid result occur through supported APIs.

Steady-state submits use one stable flattened array containing each culler's four runtime compute nodes, in bucket order, and pass that array to the construction renderer in one `renderer.compute()` call. Every value is required to report `isComputeNode === true`; otherwise construction fails.

Those node properties are not part of the stable Three Blocks facade. This lane is therefore a guarded, undocumented probe pinned to 0.11.0, not a supported package feature or a replacement for the public baseline. It answers a narrow question: what changes when identical package nodes and dispatches are grouped into one Three.js compute call rather than 32?

## Three Blocks historical heterogeneous baseline

The historical baseline pins `three-blocks@0.10.0` and imports `IndirectBatchedMesh` from its published `three-blocks/indirect-batching` entry. It merges the controlled indexed fixtures into one package mesh, assigns every object to its fixture through the stable `addGeometry()` and `addInstance()` methods, uploads the shared full TRS matrices, enables internal culling, and invokes the public `updateInternalCulling(camera)` method before cached bundle replay.

After one untimed initialization dispatch, each steady-state update schedules nine compute nodes in one `renderer.compute()` call: clear block counts; cull and count by block; prefix block counts; clear geometry counts; scatter linear survivors and count by geometry; clear the geometry allocation head; allocate survivor ranges and write multi-indirect commands; clear geometry scatter heads; and scatter survivors by geometry. The schedule is independent of bucket count.

Version 0.10 does not expose heterogeneous command and survivor readback through its public entry point. The correctness gate therefore accesses the pinned runtime culler only for untimed diagnostics and fails if the expected readback surface is absent. Construction, scene mutation, culling update, rendering, and primary mesh/culler disposal continue through the published `IndirectBatchedMesh` path. The harness also retains a guarded list of package-owned storage attributes and compute nodes so Three.js r185 can release their cache entries after each trial; that untimed lifecycle cleanup is version-specific and not a public Three Blocks capability. These distinctions must accompany any result from the lane.

Its indexed commands use GPU-allocated, generally nonzero `firstInstance` offsets. The lane is valid only when `renderer.hasFeature('indirect-first-instance')` reports support. Unsupported adapters produce an explicit smoke skip and cannot run the historical focused matrix; they do not receive a substituted result.

## Fixed-slice design

Fixed-slice assigns each geometry a permanent survivor range sized to its object capacity. One reset dispatch clears all 32 indirect instance counts and overflow state. One cull/compact dispatch tests every object and appends its global object ID directly into its bucket's range. The immutable indexed fixtures are concatenated once into nonoverlapping vertex/index ranges with a per-vertex bucket base. One mesh owns the merged buffer and the array of 32 indirect offsets; each command retains zero `firstInstance`, and the mesh is replayed from one cached `BundleGroup`.

Both nodes are passed in one `renderer.compute()` call. Fixed-slice therefore changes more than scheduling: compared with the coalesced Three Blocks probe, it also changes buffer layout, compaction work, dispatch count, render-object count, mesh type, and material/vertex-transform wiring. GPU compute and render durations must be reported separately, and a render-pass difference cannot be attributed solely to compaction.

## Fixed-slice representation ablation

The per-bucket control is constructed from the same scenario and reproduces the fixed-slice merged attribute and index bytes, storage payloads, five-word indirect command buffer, material parameters, two-dispatch compute schedule, one compute submission, and B native indexed draws. Both lanes freeze their identity object transforms. Fixed-slice retains one mesh whose geometry carries an array of B indirect offsets. The control retains B meshes that share exactly one geometry and one material; while the static `BundleGroup` is recorded, each mesh selects its scalar indirect offset.

Timing is refused unless the initial render records exactly B callbacks, the shared-geometry/material and mesh-count diagnostics match the preregistered representation, and those callback counts remain unchanged through warmup and measurement. This proves cached-bundle reuse rather than per-frame bundle reconstruction. The estimand is the full one-versus-B retained mesh/render-object topology, including scene traversal, bindings, and encoded render-bundle representation. It is not described as a private Three.js `RenderObject` scan in isolation.

## Comparison interpretation

- Public Three Blocks versus coalesced Three Blocks isolates submission topology, including renderer calls and compute-pass/command-buffer grouping, while retaining the same four package nodes and dispatches per bucket.
- Coalesced Three Blocks versus fixed-slice compares two one-call schedules but includes different kernels, storage layouts, materials, and mesh representations.
- Historical Three Blocks versus fixed-slice directly compares two heterogeneous one-call designs, but still changes package algorithms, command layout, compaction storage, mesh representation, and material integration.
- Draw all versus either culling lane measures net compute-plus-render behavior at a specified visibility fraction.
- Single-geometry Three Blocks versus fixed-slice provides the closest package/kernel parity check without heterogeneous submission scaling.
- Fixed-slice versus the per-bucket fixed-slice control isolates the cost of one versus B retained mesh/render-object topology while holding the logical compute and indexed-draw workload constant.

No one comparison alone attributes a difference to every layer of the system.

## Correctness gate

Timing begins only after the currently implemented lane-specific checks pass, and the same semantic checks run again outside the timed window after every trial. Draw all submits every object and records a reference marker plus the scenario's expected-visible count; the survivor and indirect-command checks below apply to compute lanes unless a lane is named explicitly:

- predetermined visibility agrees with an independent CPU frustum test;
- native five-word indexed indirect commands have the expected index count, instance count, first index, and base vertex;
- current and fixed-slice commands retain zero `firstInstance`; historical commands use in-range, disjoint survivor intervals whose union exactly covers the compacted prefix;
- compacted IDs exactly match the expected global object-ID membership;
- no ID is duplicate, hidden, assigned to the wrong bucket, or out of range;
- Three Blocks survivor readback length agrees with the command-declared total; fixed-slice validates each command-declared prefix within its capacity-sized readback;
- fixed-slice reports no in-capacity overflow;
- the per-bucket fixed-slice control reports one geometry identity, one material identity, B meshes, merged maximum instance capacity, exactly B callbacks during the initial bundle recording, and no bundle rebuild during timing;
- the browser reports no page or console error during the validation path.

Three Blocks v0.11 returns bucket-local survivor IDs, which the adapter maps to global IDs using `bucketBases` before applying the common membership check. Historical v0.10 retains stable global instance IDs; the adapter slices its flat survivor buffer using each command's `firstInstance` and reconstructs the common bucket-major validation layout. Aggregate and per-bucket SHA-256 commitments over sorted survivor IDs bind both validations to the scenario's expected-membership commitment. Historical allocation offsets and survivor ordering may vary, so its pre/post comparison ignores allocation placement only after command semantics and those canonical membership digests agree. Diagnostic readbacks and workload fingerprinting are outside timed frames.

For the fixed-camera 4,096-object, 20%-visibility scene, the automated browser smoke captures native PNGs and requires zero-tolerance decoded-RGBA screenshot agreement among draw all, fixed-slice, and the per-bucket control at 4, 32, and 128 buckets; both fixed-slice representations are checked again after timed replay at 32 and 128 buckets. PNG byte equality is reported separately as a diagnostic. This is static color-output smoke coverage, not an independent reference image or candidate-run artifact. The depth-ordering matrix adds exact depth32float and object-ID evidence for its fixed scenes; deterministic moving-camera, explicit bundle-rebuild, and mutation validation remain open.

## Timing environment and precision

The Vite server sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` so the benchmark can enter a cross-origin-isolated context. The page records the actual `crossOriginIsolated` value and empirically samples the smallest positive `performance.now()` increment. High-resolution CPU results require isolation and an observed increment no greater than 0.01 ms. The automated browser smoke and focused runner both enforce that threshold; the focused runner fails before its first trial if isolation or adequate timer precision is absent.

CPU measurements use `performance.now()` around common updates, compute submission calls, render submission, and the full frame body. They describe JavaScript-side elapsed time, not GPU execution or presentation latency.

GPU compute and render durations come from separate Three.js timestamp pools. The harness calls the public timestamp-resolution method, then joins frame durations through version-pinned r185 backend pool metadata. That per-frame join is benchmark instrumentation, not a stable Three.js API. A trial is rejected if timestamp support is unavailable or any measured frame lacks its expected compute or render duration. The harness also records an observed duration quantum; this diagnostic is distinct from the CPU timer increment and is not presented as end-to-end frame latency.

There is an additional r185 attribution constraint. `Renderer.compute(array)` passes the array into the backend timestamp-identifier path, which does not identify it as a `ComputeNode`. Multiple newly allocated array calls in one frame consequently receive the same identifier; the pool consumes query pairs for every call but its identifier-to-offset map retains only the last offset. The 32-call public Three Blocks lane would therefore report only its last compute pass for a frame, not the aggregate of all 32. The coalesced, historical, and fixed-slice lanes make one array-valued compute call per frame and do not collide with another compute call in that frame.

### Three.js r185 timestamp-pool limit

The r185 WebGPU backend allocates 2,048 queries for each timestamp type. A timed compute submission consumes two queries, one at each boundary. Queries remain unresolved throughout each 300-frame warmup or 240-frame measurement block; the harness resolves and clears the warmup block before measurement begins.

- One compute submission per frame uses 600 queries during warmup and 480 during measurement, so each block fits independently.
- The 32-bucket public Three Blocks lane would use 19,200 queries during warmup and 15,360 during measurement.

Neither public-lane block can fit in the r185 compute timestamp pool: at 32 submissions per frame, one unresolved block is limited to 32 complete frames. Shorter blocks alone do not fix the timestamp-identifier collision described above. A valid public-lane GPU protocol needs both bounded query resolution and unique per-call timestamp aggregation, or a Three.js change that provides it. Accordingly, the standard 32-bucket GPU-timestamp matrix uses either the one-submission coalesced probe or the one-submission historical baseline alongside fixed-slice. The capacity and identifier behavior are Three.js r185 implementation constraints, not WebGPU limits or performance results.

## Environment telemetry and evidence status

The focused runner defaults to `development` evidence status. `BENCHMARK_EVIDENCE_STATUS` accepts exactly `development` or `candidate`; candidate status must be requested explicitly after the run environment has been reviewed.

When `nvidia-smi` is available, one long-lived process samples all visible NVIDIA GPUs at a requested 250 ms interval for the run. `gpu-telemetry.csv` records wall-clock and monotonic run time; current trial, plan position, mode, visibility, and phase; GPU identity; performance state; graphics and memory clocks; GPU and memory utilization; memory occupancy; temperature; and power. The runner also records sanitized pre-run and post-run compute-process snapshots that retain only process basenames. Sampling is performed outside the browser and no telemetry call is made from a measured frame.

Telemetry absence does not fail a technically valid benchmark, which keeps the harness portable to non-NVIDIA systems. It does leave the environment unverified unless equivalent device-specific evidence is supplied. Promotion to candidate evidence requires a manual review that establishes an otherwise idle device and checks for unexplained, mode- or order-correlated discontinuities in clocks, utilization, memory, temperature, power, or device state during measured phases. There is no automatic P-state rejection threshold: boost and power-state behavior are device-, driver-, and workload-dependent, so any rejection boundary must be supported by evidence for the tested system rather than invented by the harness.

At run start and after teardown, the runner records the full Git commit and tree, exact porcelain digest and counts, tracked-file-list digest, a SHA-256 commitment over every tracked working-tree file, and the raw `package-lock.json` SHA-256. Each capture brackets and repeats its Git-state and working-byte reads, retrying rather than accepting a mixed snapshot. Candidate status requires clean source at startup. Any run that begins with available provenance is rejected unless the same commit, ref, porcelain state, tracked list, working bytes, and lock are proven stable through teardown. Candidate runs should begin from `npm ci`; the lock commitment records the intended dependency graph but cannot prove that installed package contents were not edited locally.

The page freshly fingerprints every source fixture attribute, index, draw range, and bound before and after timing. It also fingerprints the generated matrices, bounds, bucket layout, cull order, and expected membership. Node validates the full schemas, dimensions, nested record hashes, seed, visibility, and array lengths rather than trusting only the browser's aggregate digest. Full manifests are deduplicated by digest in the crash-safe `workload-manifests.json`; each trial capture records its pre/start/post digest references. Full validation diagnostics and every actual/expected native indirect-command record are stored in the crash-safe `validation-artifacts.json`, linked by digest from the trial summary rather than duplicated across frame rows. Separate structural tests prove that merged fixed-slice storage preserves those fixture bytes, metadata, rebased indices, and command offsets.

The final atomically written `artifact-manifest.json` is the finalization marker. It requires frame data, metadata, trial summaries, validation artifacts, workload manifests, and a telemetry-availability summary, recording each file's byte length and SHA-256. A usable run additionally requires `metadata.status` to equal `complete` and analyzer verification to succeed. Device-sample telemetry remains optional; its presence and evidence availability, or the reason for absence, are explicit in the manifest.

## Recorded measurements

Each accepted timed run records:

- CPU common-update, compute-submission, render-submission, total-submission, and frame-body distributions;
- GPU compute-pass, render-pass, and summed-pass distributions;
- configured compute dispatch and submission counts;
- configured draw-command and render-object counts;
- visible fraction and expected survivor count;
- timestamp availability, missing-frame counts, and observed quantum;
- renderer, browser, device, viewport, isolation, and CPU timer metadata;
- validation kind and deterministic configuration identifiers;
- Git/source/lock provenance, geometry and scenario manifests, and full validation artifacts.

Summed pass durations are not described as presentation latency. Queue gaps, browser scheduling, and compositor work require separate measurement.

## Statistical protocol

- Warm 300 frames, resolve warmup timestamps, then measure 240 frames.
- Use six repetitions of each focused cell.
- Rotate all mode permutations and rotate visibility order by repetition.
- For the representation ablation, alternate AB/BA mode order exactly three times each and keep each within-visibility pair adjacent.
- Retain frame-level samples, not only per-trial summaries.
- Report medians, tail distributions, absolute milliseconds, and frame-budget context.
- Treat raw result files as immutable inputs to versioned analysis.
- Require a stable, otherwise idle device for evidence used in inference.
- Reproduce material conclusions on a second GPU family.

## Initial decision gates

On the first discrete-GPU test system, the fixed-slice direction remains interesting if it satisfies all of the following in an otherwise equivalent workload:

- at approximately 20% visibility, at least a 40% GPU-pass reduction versus bundled draw all;
- at approximately 80% visibility, at least a 10% reduction in a meaningful geometry workload;
- near full visibility, no more than a 5% regression;
- zero correctness failures;
- a material benefit versus the selected one-submission package comparator, not merely versus the multi-call public schedule.

For the comparator condition, the preregistered threshold is at least 10% in GPU pass time or 0.10 ms in a disclosed CPU-submit-plus-GPU-pass accounting metric at low visibility, without a material regression in either component. That sum is an accounting metric, not frame latency. Both the coalesced v0.11 probe and compatible historical v0.10 baseline must be reported before claiming differentiation from the existing ecosystem. Failure to clear the threshold favors a scheduling contribution, benchmark result, or documented negative result over a new culling implementation. These are falsification gates for this research direction, not universal WebGPU guidance.
