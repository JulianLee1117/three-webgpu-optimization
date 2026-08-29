# Three.js WebGPU Optimization

Evidence-driven research into GPU visibility, indirect rendering, and retained command submission for Three.js WebGPU.

## Objective

This project investigates when GPU-driven visibility compaction reduces the net measured CPU-submission and timestamped GPU-pass cost in Three.js. The work focuses on GPU frustum culling, compacted instance IDs, compute-written indexed indirect commands, and cached `BundleGroup` rendering across heterogeneous geometry. Presentation latency remains outside the current measurement boundary.

The goal is a reproducible boundary, not a universal speedup claim: identify the workloads where an optimization wins, the crossover where its overhead exceeds its benefit, and the Three.js or WebGPU constraints that explain the result.

## Current 32-bucket experiment

The current controlled experiment uses Three.js 0.185.1, 32 unique indexed geometry assets drawn from four procedural topology families. Each bucket receives a deterministic, small non-affine vertex deformation that preserves its family's topology and comparable bounds. The scene otherwise holds static full TRS instance matrices, a fixed projection, matched PBR parameters, and exact survivor/command validation. A bundled draw-all mode provides the no-culling reference. Four GPU-culling lanes separate historical package behavior, current package integration, scheduling, and algorithm design:

| Lane | Steady-state schedule at 32 buckets | Interpretation |
| --- | ---: | --- |
| Three Blocks public baseline | 128 dispatches in 32 `renderer.compute()` calls | One pinned `three-blocks@0.11.0` culler per bucket, invoked through public `setCameraUniforms()` and `update()` methods |
| Three Blocks coalesced probe | 128 dispatches in one `renderer.compute()` call | The same cullers and GPU work, but a guarded, version-pinned array of undocumented runtime compute nodes |
| Three Blocks historical baseline | 9 dispatches in one `renderer.compute()` call | The published `three-blocks@0.10.0` heterogeneous `IndirectBatchedMesh` execution path |
| Fixed-slice | 2 dispatches in one `renderer.compute()` call | An independent reset-and-compact design with one preallocated survivor slice per geometry |

Here, "submission" means one explicit Three.js `renderer.compute()` call. The public and coalesced Three Blocks v0.11 lanes use the same per-bucket resources, compute nodes, dispatches, and validation; their distinction is submission topology, including renderer-call and compute-pass/command-buffer grouping. The historical v0.10 lane is a separate package algorithm and mesh representation. Comparisons between either one-submission package lane and fixed-slice therefore change compaction layout, kernel count, material wiring, and mesh representation, so compute and render timings must be reported separately.

The schedule counts are strategy-declared metadata checked by the runner against the expected lane configuration; they are not runtime command-stream counters. A four-bucket control uses exactly one asset from each topology family, making it the family-count and merge-aware control between the single-geometry and higher-diversity cells.

The coalesced lane is a diagnostic probe, not a supported Three Blocks capability. It is pinned to the installed package version, checks every accessed runtime value for `isComputeNode`, and fails rather than silently changing behavior if that internal surface moves.

The historical lane constructs and updates `IndirectBatchedMesh` through its published v0.10 entry point. Exact command and survivor readback uses a guarded, version-pinned diagnostic surface because that release has no public heterogeneous readback method. Its generated commands use nonzero `firstInstance` values, so the lane is available only when the adapter exposes WebGPU's optional `indirect-first-instance` feature.

## Current status

The package baselines, scheduling probe, fixed-slice lane, and their exact correctness gates are integrated. This README does not assert a 32-bucket performance conclusion; timing evidence must pass the environment, timestamp-completeness, repetition, and cross-device requirements below before it supports a broader claim.

The core techniques, including GPU frustum culling, survivor compaction, indirect drawing, and retained command submission, are established prior art. The research question is whether a narrower Three.js integration or fixed-ownership specialization produces a material, reproducible difference.

## Run locally

Node.js 22.12 or newer and an installed Chrome, Chromium, or Edge build with WebGPU support are required. Set `BROWSER_PATH` when the browser is not in a standard location.

```sh
npm install
npm test
npm run build
npm run smoke:browser
```

The focused benchmark defaults to 16,384 objects and 32 geometry buckets:

```sh
npm run benchmark:focused
```

Raw runs are written under the ignored `results/runs/` directory. `BENCHMARK_OBJECT_COUNT` accepts `4096`, `16384`, or `65536`; `BENCHMARK_BUCKET_COUNT` accepts `1`, `4`, `32`, or `128`. For 4-, 32-, and 128-bucket runs, `BENCHMARK_HETEROGENEOUS_COMPARATOR` accepts `coalesced-v11` (the default) or `historical-v10`. The historical choice fails at startup when `indirect-first-instance` is unavailable. Single-bucket runs continue to use the public Three Blocks v0.11 lane. For example, on macOS or Linux:

```sh
BENCHMARK_HETEROGENEOUS_COMPARATOR=historical-v10 npm run benchmark:focused
```

On PowerShell:

```powershell
$env:BENCHMARK_HETEROGENEOUS_COMPARATOR = 'historical-v10'
npm run benchmark:focused
```

A completed run can be summarized with:

```sh
npm run analyze -- results/runs/<run-id>/frames.csv
```

The benchmark requires a WebGPU-capable device and records the actual adapter, backend, browser, timestamp support, and timer precision with each run. Results from a busy or changing device should be retained as development evidence rather than used for performance claims.

## Primary hypothesis

Scenes with immutable geometry ownership may not need a general prefix-allocation-and-redistribution pipeline. A fixed-slice specialization can preallocate one survivor range per geometry, reset its counters, append visible IDs directly into those ranges, keep indirect `firstInstance` values at zero, and replay separate indexed geometry commands from a cached render bundle.

That design trades mutation flexibility for two compute dispatches and a portable command layout. The experiment tests whether this specialization remains useful after CPU submission cost, GPU compute cost, render cost, visibility, and correctness are measured together.

## Measurement boundaries

The local server sends COOP and COEP headers to request a cross-origin-isolated context. The harness records both `crossOriginIsolated` and the smallest observed positive `performance.now()` increment; CPU timings are not treated as high-resolution evidence unless isolation succeeds and the measured increment is at most 0.01 ms. The browser smoke and focused runner enforce this threshold.

GPU pass timing uses Three.js timestamp queries and records the observed timestamp quantum separately from CPU clock precision. Per-frame timestamp joining reads version-pinned r185 backend pool metadata after public timestamp resolution; it is instrumentation for this harness, not a stable Three.js API. Three.js r185 allocates 2,048 queries per render or compute timestamp pool, and each timed compute submission consumes two queries. A 32-bucket public-package lane would require 19,200 compute queries during the unresolved 300-frame warmup and 15,360 during the 240-frame measurement block. In addition, r185 does not assign distinct timestamp identifiers to multiple array-valued compute calls in one frame, so only the last such call would remain in its per-frame map. The public v0.11 lane is therefore excluded from the standard GPU-timestamp matrix until both limits are addressed. The one-submission coalesced, historical, and fixed-slice lanes are unaffected by the identifier collision and require 600 warmup queries and 480 measurement queries, with the pool resolved between phases. These are measurement-system constraints, not evidence that one GPU algorithm is faster.

## Evidence standard

An accepted performance result must include:

- pinned Three.js, package, browser, operating-system, and device details;
- equivalent scene inputs and rendering conditions across compared modes;
- exact survivor membership and native indexed-command validation before timing;
- GPU compute and render timestamps separated from CPU update and submission time;
- recorded CPU-clock and GPU-timestamp precision;
- frame-level samples, rotated mode order, repeated runs, and machine-readable output;
- negative and near-parity results alongside wins;
- a second materially different GPU family before generalizing beyond the tested system.

The detailed procedure is in [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md). Ecosystem context and claim boundaries are in [docs/PRIOR_ART.md](docs/PRIOR_ART.md). Dependency licensing and required attribution are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Scope

The harness exposes bundled draw all, Three Blocks v0.11 public and coalesced scheduling, the historical Three Blocks v0.10 indirect-batching path, and fixed-slice compaction. At 4, 32, or 128 buckets, the focused timed matrix contains draw all, one explicitly selected one-submission package comparator, and fixed-slice; the multi-submission public v0.11 lane remains a separate correctness and scheduling reference. Stock Three.js `BatchedMesh`, dynamic cameras, production assets, and additional GPU families remain separate comparison stages.

This repository is an experiment harness and does not expose a supported library API.

## License

No license is currently granted for the original source in this repository. Third-party dependencies remain subject to their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
