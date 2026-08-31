# Candidate results: fixed-ownership compaction at 32 geometry buckets

## Status and scope

These results are candidate evidence from one machine. They establish a reproducible result for the tested RTX 5070 Ti / D3D12 configuration; they do not establish a general WebGPU result.

All headline runs used source commit `abbc5629cb5ce44bafd2ad0ced91fbbb07d6e8f2`, Three.js 0.185.1, Chrome 151.0.7922.174, a 1280 x 720 WebGPU viewport, 16,384 static objects, and 32 indexed geometry buckets. The fixtures contain positions, normals, UVs, full static TRS matrices, and matched `MeshStandardNodeMaterial` parameters. They are controlled procedural assets rather than production content.

Each run contains six position-balanced repetitions at 20%, 80%, and 99% target visibility. Each trial uses 300 untimed warm-up frames followed by 240 measured frames. The values below are median paired deltas of per-trial p50s. The sign convention is fixed-slice minus comparator, so negative values favor fixed-slice.

Every ecosystem run reported in the two tables below completed all 54 trials and 12,960 measured frames, passed exact survivor-membership and native indexed-command validation, retained stable source provenance, and passed artifact-manifest verification. GPU telemetry remained available throughout. GPU pass is the sum of timestamped compute and render durations; it is not presentation or end-to-end queue latency. Accounted CPU+GPU is the harness sum of synchronous CPU submission and timestamped GPU pass, not an assertion that the two execute serially.

## Published Three Blocks 0.10 heterogeneous path

The historical comparison uses the published `IndirectBatchedMesh` entry point from Three Blocks 0.10.0. Both it and fixed-slice retain the heterogeneous population in one render object. The historical path performs nine compute dispatches and uses nonzero `firstInstance`; fixed-slice performs two dispatches and keeps every `firstInstance` at zero.

| Visibility | GPU pass A / B | GPU compute A / B | GPU render A / B | Accounted CPU+GPU A / B |
| ---: | ---: | ---: | ---: | ---: |
| 20% | -31.5% / -33.1% | -89.6% / -89.7% | -12.4% / -13.2% | -10.5% / -22.9% |
| 80% | -12.7% / -12.1% | -89.3% / -89.4% | -5.6% / -5.6% | -11.5% / -11.0% |
| 99% | -13.7% / -17.9% | -89.4% / -89.6% | -8.6% / -12.9% | -12.8% / -17.1% |

The compute reduction is the clearest mechanism result. The fixed-ownership layout avoids the historical block-count, prefix, command-construction, and redistribution pipeline when geometry ownership is immutable. Render-only time also favored fixed-slice in both replications, so the total GPU-pass reduction was not produced by compute timing alone.

The historical implementation was removed from Three Blocks 0.11 and requires WebGPU's optional `indirect-first-instance` feature. This result therefore characterizes an existing published implementation; it is not a comparison with the current supported package facade.

## Three Blocks 0.11 coalesced diagnostic

The current-package diagnostic constructs the same 32 public Three Blocks 0.11 cullers and their 128 compute nodes as the public per-bucket integration, but submits those nodes in one guarded, version-pinned `renderer.compute()` call. This avoids the Three.js r185 timestamp-identifier collision for multiple array-valued compute calls. It is an undocumented research probe, not a supported Three Blocks capability or a measurement of the stock 32-call schedule.

| Visibility | GPU pass A / B | GPU compute A / B | GPU render A / B | Accounted CPU+GPU A / B |
| ---: | ---: | ---: | ---: | ---: |
| 20% | -52.8% / -54.6% | -97.6% / -97.6% | +38.2% / +38.7% | -50.1% / -57.7% |
| 80% | -14.4% / -10.5% | -97.6% / -97.6% | +30.3% / +37.5% | -33.6% / -27.0% |
| 99% | +0.1% / -5.6% | -97.6% / -96.6% | +43.5% / +35.2% | -17.8% / -24.5% |

This comparison has a different mechanism split. Fixed-slice's two-dispatch compute stage is substantially cheaper, while its merged-geometry render path is slower than the package's per-bucket render path. Compute savings dominate at low and medium visibility. Near full visibility, GPU pass reaches the crossover: one run was effectively tied and the other favored fixed-slice modestly. The accounted metric still favored fixed-slice because its compute submission and retained-object topology also reduce synchronous CPU work.

## Draw-all reference

Across the four ecosystem candidate runs, fixed-slice reduced median paired GPU-pass time relative to bundled draw-all by 77.2-80.3% at 20% visibility and 20.4-26.4% at 80% visibility. At 99% visibility the measured reduction ranged from 2.1% to 12.7%, indicating a much weaker, more environment-sensitive regime.

The draw-all comparison establishes the visibility-compaction crossover. It does not distinguish fixed-slice from existing GPU-culling implementations; the package comparisons above provide that control.

## Retained representation ablation

The representation matrix holds the compute schedule, merged indexed geometry, material, storage payloads, indirect commands, and B native draws constant. It changes only whether those commands are retained by one mesh/render object or by B meshes/render objects in one cached bundle.

| Bucket count | Accepted run-level replications | CPU render-submission delta across visibility cells | Interpretation |
| ---: | ---: | ---: | --- |
| 1 | 1 | approximately 0 to +0.0025 ms | Equal-count negative control |
| 4 | 1 clean | -0.0025 to -0.0100 ms | Small effect near the 0.005 ms CPU timer quantum |
| 32 | 2 | -0.0300 to -0.0475 ms | Reproduced one-versus-32 topology cost |
| 128 | 2 | -0.1275 to -0.1900 ms | Reproduced high-topology scaling |

The one-object representation reduced CPU render submission by 29.3-38.8% at 32 buckets and 60.9-69.4% at 128 buckets. GPU-pass deltas did not scale monotonically with bucket count, so the ablation supports a CPU retained-topology conclusion, not an independent GPU optimization claim.

One earlier four-bucket artifact is excluded from this table because device memory moved between approximately 7.2 and 14.1 GiB during the run. The clean repeat held device memory between 7.1 and 7.3 GiB. This exclusion does not affect the two 32-bucket or two 128-bucket replications.

## Evidence identifiers

| Comparison | Replication | Run identifier |
| --- | --- | --- |
| Three Blocks 0.10 historical | A | `ecosystem-o16384-b32-2026-08-31T08-31-28.754Z` |
| Three Blocks 0.10 historical | B | `ecosystem-o16384-b32-2026-08-31T20-05-45.150Z` |
| Three Blocks 0.11 coalesced diagnostic | A | `ecosystem-o16384-b32-2026-08-31T20-09-51.773Z` |
| Three Blocks 0.11 coalesced diagnostic | B | `ecosystem-o16384-b32-2026-08-31T20-13-07.746Z` |

Generated run directories remain ignored source artifacts. Each identifier above names a manifest-bound local directory containing frame-level CSV data, metadata, trial summaries, validation payloads, workload manifests, GPU telemetry, and SHA-256 commitments. The analyzer rejects incomplete trials, changed source provenance, mismatched workload links, and altered required artifacts.

## What the evidence establishes

- Fixed geometry ownership can materially simplify GPU visibility compaction in the tested Three.js WebGPU workload.
- The two-dispatch fixed-slice compute stage is reproducibly cheaper than both the published nine-dispatch historical heterogeneous path and the 128-dispatch current-package diagnostic.
- One retained mesh/render object removes a CPU submission cost that grows with bucket count.
- The current-package comparison has a real crossover: fixed-slice wins through compute efficiency while its render-only path is slower.

## What remains open

- A second materially different GPU family is required before generalizing the crossover or percentage reductions.
- The current-package result is a diagnostic comparison, not stock public-API timing.
- Production assets, textures, dynamic cameras, moving objects, shadows, transparency, skinning, and morph targets are outside this result.
- The benchmark does not measure presentation latency or queue overlap.
- Atomic survivor order may affect early-depth efficiency in overlapping opaque scenes.

The next GPU-side experiment should isolate the last point. A controlled depth-ordering matrix should preserve exact visible membership, geometry, transforms, material, and draw count while comparing front-to-back and reverse coarse depth bins in high-overlap and low-overlap scenes. It should proceed only with exact color, depth, object-ID, survivor, and command parity, and it should be rejected if its extra compute cost does not improve summed GPU pass time.
