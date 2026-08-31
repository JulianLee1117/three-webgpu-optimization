# Three.js WebGPU Optimization

Evidence-driven research into GPU visibility, indirect rendering, and retained command submission for Three.js WebGPU.

## Objective

This project investigates when GPU-driven visibility compaction reduces the net measured CPU-submission and timestamped GPU-pass cost in Three.js. The work focuses on GPU frustum culling, compacted instance IDs, compute-written indexed indirect commands, and cached `BundleGroup` rendering across heterogeneous geometry. Presentation latency remains outside the current measurement boundary.

The goal is a reproducible boundary, not a universal speedup claim: identify the workloads where an optimization wins, the crossover where its overhead exceeds its benefit, and the Three.js or WebGPU constraints that explain the result.

## Current 32-bucket experiment

The current controlled experiment uses Three.js 0.185.1, 32 unique indexed geometry assets drawn from four procedural topology families. Each bucket receives a deterministic, small non-affine vertex deformation that preserves its family's topology and comparable bounds. The scene otherwise holds static full TRS instance matrices, a fixed projection, matched PBR parameters, and exact survivor/command validation. A bundled draw-all mode provides the no-culling reference. Four GPU-culling lanes separate historical package behavior, current package integration, scheduling, and algorithm design:

| Lane | Steady-state schedule at 32 buckets | Interpretation |
| --- | ---: | --- |
| Three Blocks public baseline | 128 dispatches in 32 `renderer.compute()` calls | One pinned `three-blocks@0.11.0` culler and render object per bucket, invoked through public `setCameraUniforms()` and `update()` methods |
| Three Blocks coalesced probe | 128 dispatches in one `renderer.compute()` call | The same per-bucket cullers, render objects, and GPU work, but a guarded, version-pinned array of undocumented runtime compute nodes |
| Three Blocks historical baseline | 9 dispatches in one `renderer.compute()` call | The published `three-blocks@0.10.0` heterogeneous `IndirectBatchedMesh` execution path in one render object |
| Fixed-slice | 2 dispatches in one `renderer.compute()` call | One preallocated survivor slice per geometry, with all indexed fixture ranges and indirect offsets retained by one render object |
| Fixed-slice per-bucket control | 2 dispatches in one `renderer.compute()` call | The same merged geometry, material, storage payloads, indirect commands, and native draws as fixed-slice, retained by one mesh/render object per bucket |

Here, "submission" means one explicit Three.js `renderer.compute()` call. The public and coalesced Three Blocks v0.11 lanes use the same per-bucket resources, compute nodes, dispatches, render objects, and validation; their distinction is submission topology, including renderer-call and compute-pass/command-buffer grouping. The historical v0.10 lane and fixed-slice each retain B indexed indirect commands in one render object, but remain different package algorithms and mesh representations. Comparisons between either package lane and fixed-slice therefore change compaction layout, kernel count, material wiring, and geometry organization, so compute and render timings must be reported separately.

The schedule counts are strategy-declared metadata checked by the runner against the expected lane configuration; they are not runtime command-stream counters. A four-bucket control uses exactly one asset from each topology family, making it the family-count and merge-aware control between the single-geometry and higher-diversity cells.

The coalesced lane is a diagnostic probe, not a supported Three Blocks capability. It is pinned to the installed package version, checks every accessed runtime value for `isComputeNode`, and fails rather than silently changing behavior if that internal surface moves.

The historical lane constructs and updates `IndirectBatchedMesh` through its published v0.10 entry point. Exact command and survivor readback uses a guarded, version-pinned diagnostic surface because that release has no public heterogeneous readback method. Its generated commands use nonzero `firstInstance` values, so the lane is available only when the adapter exposes WebGPU's optional `indirect-first-instance` feature.

## Coarse depth-ordering experiment

The completed matrix tested whether per-bucket coarse GPU-generated front-to-back bin-block traversal reduces render cost under heavy overdraw. It compares atomic fixed-slice compaction with two eight-bin lanes: front-to-back and reverse traversal. The ordered pair retains the same four dispatches, shader operations, merged geometry, material, static bundle, 32 native indirect draws, and one compute submission; only one traversal uniform differs. Bucket draw order remains fixed.

The fixed candidate cell uses 65,536 objects, 32 buckets, 99% visibility, a high-overlap layout, and a low-overlap negative control. Three.js reversed depth is enabled for every lane so dense-scene depth ties do not make output depend on draw order. Candidate timing is gated by exact survivor, command, bin-layout, traversal-normalized within-bin sequence, bundle-lifecycle, full-frame color, depth, and object-ID evidence. The preregistered design and falsification thresholds are in [Coarse depth-ordering protocol](docs/DEPTH_ORDERING_PROTOCOL.md).

## Frozen render-order crossover

The active experiment isolates the remaining render-order question from compaction cost. Before timing, it constructs exact front-to-back and reverse survivor lists in the two halves of one immutable storage buffer. Both lanes then use the same merged geometry, indirect commands, matrix buffer, material, shader pipeline, mesh, and static bundle; a uint base offset is the only lane-varying shader input. There is no compute submission in the timed workload.

Each paired trial alternates the two lanes inside complementary eight-frame superblocks and renders once per animation frame to a fixed offscreen target. Twelve repetitions cover both high- and low-overlap layouts while counterbalancing physical buffer placement, starting superblock orientation, and layout position. Full buffer readback, an untimed selector-address challenge, exact color/depth/object-ID parity, timestamp identity, and static-resource invariants gate every result. The estimand, schedule, replication rule, and decision thresholds are fixed in the [Frozen render-order crossover protocol](docs/FROZEN_DEPTH_CROSSOVER_PROTOCOL.md).

## Current status

The package baselines, scheduling probe, fixed-slice lane, and fixed-slice representation control are integrated with lane-specific correctness gates; every compute lane requires exact survivor membership and native indexed-command validation. The representation control additionally requires one shared geometry and material, exactly B retained meshes, exactly B bundle-record callbacks before timing, no bundle rebuild during timing, and exact decoded-pixel parity.

Two independent single-device candidate runs now show that the fixed-ownership specialization materially reduces timestamped GPU-pass cost against the published Three Blocks 0.10 heterogeneous path at 32 buckets. Two further candidate runs show a low- and medium-visibility GPU-pass reduction against the guarded Three Blocks 0.11 coalesced diagnostic, with a near-full-visibility crossover. These are scoped results for one RTX 5070 Ti / D3D12 system, not a cross-device conclusion.

Two coarse depth-ordering candidates each completed all 36 trials and passed the correctness, lifecycle, provenance, and artifact gates, but neither passed the preregistered performance decision. The first measured front-to-back render time 4.1% slower than reverse; the replication measured it 10.6% faster but failed the absolute-position stability gate. The deterministic bucket-serial scatter increased total GPU-pass time by 49.1-55.1% in the high-overlap layout and 58.7-66.5% in the low-overlap control. The frozen render-order crossover now tests the possible early-depth signal without including survivor generation or compaction in the timed contrast. Exact completed comparisons and limits are reported in [Candidate results](docs/CANDIDATE_RESULTS.md).

The core techniques, including GPU frustum culling, survivor compaction, indirect drawing, and retained command submission, are established prior art. The research question is whether a narrower Three.js integration or fixed-ownership specialization produces a material, reproducible difference.

## Run locally

Node.js 22.12 or newer and an installed Chrome, Chromium, or Edge build with WebGPU support are required. Set `BROWSER_PATH` when the browser is not in a standard location.

```sh
npm ci
npm test
npm run build
npm run smoke:browser
```

The focused benchmark defaults to 16,384 objects and 32 geometry buckets:

```sh
npm run benchmark:focused
```

Raw runs are written under the ignored `results/runs/` directory. Each run records the Git commit, porcelain-state digest and counts, SHA-256 commitments for the tracked working tree and dependency lock, deterministic geometry and scenario manifests, untimed preflight/timing-start/post-trial lane-specific validation payloads, frame data, and optional GPU telemetry. The runner accepts exactly `development` or `candidate` evidence status; candidate evidence requires a clean Git tree, and every run that begins with available provenance requires it to remain unchanged through teardown. Core files are hash-bound by a final artifact manifest, while optional telemetry absence is explicit rather than silently omitted. `BENCHMARK_OBJECT_COUNT` accepts `4096`, `16384`, or `65536`; `BENCHMARK_BUCKET_COUNT` accepts `1`, `4`, `32`, or `128`. For 4-, 32-, and 128-bucket runs, `BENCHMARK_HETEROGENEOUS_COMPARATOR` accepts `coalesced-v11` (the default) or `historical-v10`. The historical choice fails at startup when `indirect-first-instance` is unavailable. Single-bucket runs continue to use the public Three Blocks v0.11 lane. For example, on macOS or Linux:

```sh
BENCHMARK_HETEROGENEOUS_COMPARATOR=historical-v10 npm run benchmark:focused
```

On PowerShell:

```powershell
$env:BENCHMARK_HETEROGENEOUS_COMPARATOR = 'historical-v10'
npm run benchmark:focused
```

The separate fixed-slice representation ablation runs six balanced AB/BA repetitions at all three visibility levels:

```sh
BENCHMARK_MATRIX=fixed-slice-representation npm run benchmark:focused
```

```powershell
$env:BENCHMARK_MATRIX = 'fixed-slice-representation'
npm run benchmark:focused
```

This matrix changes the retained mesh/render-object topology from B to one while holding the compute schedule, merged geometry payload, storage payload bytes, indirect commands, material parameters, and B native draws constant. A one-bucket run is labeled as an equal-count negative control rather than render-object scaling evidence.

The depth-ordering matrix has a fixed 65,536-object, 32-bucket design:

```sh
BENCHMARK_MATRIX=depth-ordering npm run benchmark:focused
```

```powershell
$env:BENCHMARK_MATRIX = 'depth-ordering'
npm run benchmark:focused
```

The frozen render-order crossover uses the same fixed scene size and runs 24 paired trials:

```sh
BENCHMARK_MATRIX=depth-ordering-render-only npm run benchmark:focused
```

```powershell
$env:BENCHMARK_MATRIX = 'depth-ordering-render-only'
npm run benchmark:focused
```

A completed run can be summarized with:

```sh
npm run analyze -- results/runs/<run-id>
```

Directory analysis verifies the artifact manifest, run acceptance, source provenance, workload links, and cross-file counts before reporting statistics. For the coarse depth matrix it evaluates the preregistered numeric gates, keeps front-to-back versus reverse under causal contrasts, and labels comparisons against atomic fixed-slice as contextual whole-mechanism comparisons. For the frozen crossover it reconstructs every superblock, rejects schedule or base-mapping deviations, and evaluates the preregistered repetition-level, control, interaction, and drift gates. A standalone `frames.csv` remains accepted for exploratory analysis but is labeled unverified.

The benchmark requires a WebGPU-capable device and records the actual adapter, backend, browser, timestamp support, and timer precision with each run. The browser smoke additionally requires zero-tolerance decoded-RGBA screenshot agreement among draw all, fixed-slice, and the per-bucket representation control at 4, 32, and 128 buckets, reports PNG byte equality as a diagnostic, verifies 32- and 128-bucket timed replay, and checks repeated strategy teardown against the renderer's resource and cache baseline. Results from a busy or changing device should be retained as development evidence rather than used for performance claims.

## Primary hypothesis

Scenes with immutable geometry ownership may not need a general prefix-allocation-and-redistribution pipeline. A fixed-slice specialization can preallocate one survivor range per geometry, reset its counters, append visible IDs directly into those ranges, keep indirect `firstInstance` values at zero, concatenate immutable fixture ranges into one indexed buffer, and replay their indirect offsets through one cached render object.

That design trades mutation flexibility for two compute dispatches and a portable command layout. The experiment tests whether this specialization remains useful after CPU submission cost, GPU compute cost, render cost, visibility, and correctness are measured together.

## Measurement boundaries

The local server sends COOP and COEP headers to request a cross-origin-isolated context. The harness records both `crossOriginIsolated` and the smallest observed positive `performance.now()` increment; CPU timings are not treated as high-resolution evidence unless isolation succeeds and the measured increment is at most 0.01 ms. The browser smoke and focused runner enforce this threshold.

GPU pass timing uses Three.js timestamp queries and records the observed timestamp quantum separately from CPU clock precision. Per-frame timestamp joining reads version-pinned r185 backend pool metadata after public timestamp resolution; it is instrumentation for this harness, not a stable Three.js API. Three.js r185 allocates 2,048 queries per render or compute timestamp pool, and each timed compute submission consumes two queries. A 32-bucket public-package lane would require 19,200 compute queries during the unresolved 300-frame warmup and 15,360 during the 240-frame measurement block. In addition, r185 does not assign distinct timestamp identifiers to multiple array-valued compute calls in one frame, so only the last such call would remain in its per-frame map. The public v0.11 lane is therefore excluded from the standard GPU-timestamp matrix until both limits are addressed. The one-submission coalesced, historical, and fixed-slice lanes are unaffected by the identifier collision and require 600 warmup queries and 480 measurement queries, with the pool resolved between phases. These are measurement-system constraints, not evidence that one GPU algorithm is faster.

## Evidence standard

An accepted performance result must include:

- pinned Three.js, package, browser, operating-system, and device details;
- equivalent scene inputs and rendering conditions across compared modes;
- exact survivor membership and native indexed-command validation before and after timing;
- GPU compute and render timestamps separated from CPU update and submission time;
- recorded CPU-clock and GPU-timestamp precision;
- frame-level samples, rotated mode order, repeated runs, and machine-readable output;
- negative and near-parity results alongside wins;
- a second materially different GPU family before generalizing beyond the tested system.

The detailed procedure is in [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md). Ecosystem context and claim boundaries are in [docs/PRIOR_ART.md](docs/PRIOR_ART.md). Dependency licensing and required attribution are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Scope

The harness exposes bundled draw all, Three Blocks v0.11 public and coalesced scheduling, the historical Three Blocks v0.10 indirect-batching path, fixed-slice compaction, and a per-bucket fixed-slice representation control. At 4, 32, or 128 buckets, the ecosystem matrix contains draw all, one explicitly selected one-submission package comparator, and fixed-slice. The separate representation matrix compares only the B-object and one-object fixed-slice lanes. The multi-submission public v0.11 lane remains a separate correctness and scheduling reference. Stock Three.js `BatchedMesh`, linear-depth and object-ID output checks, dynamic cameras, production assets, and additional GPU families remain separate comparison stages.

This repository is an experiment harness and does not expose a supported library API.

## License

No license is currently granted for the original source in this repository. Third-party dependencies remain subject to their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
