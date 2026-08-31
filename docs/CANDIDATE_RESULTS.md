# Candidate results

## Status and scope

These results are candidate evidence from one machine. They establish a reproducible result for the tested RTX 5070 Ti / D3D12 configuration; they do not establish a general WebGPU result.

The fixed-ownership ecosystem runs used source commit `abbc5629cb5ce44bafd2ad0ced91fbbb07d6e8f2`, Three.js 0.185.1, Chrome 151.0.7922.174, a 1280 x 720 WebGPU viewport, 16,384 static objects, and 32 indexed geometry buckets. The fixtures contain positions, normals, UVs, full static TRS matrices, and matched `MeshStandardNodeMaterial` parameters. They are controlled procedural assets rather than production content.

Each fixed-ownership ecosystem run contains six position-balanced repetitions at 20%, 80%, and 99% target visibility. Each trial uses 300 untimed warm-up frames followed by 240 measured frames. The values below are median paired deltas of per-trial p50s. The sign convention is fixed-slice minus comparator, so negative values favor fixed-slice.

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

## Coarse depth-ordering result

The depth-ordering matrix used source commit `9637ad2552834a79d7b2f34386c516967d73614b`, the same RTX 5070 Ti / D3D12 system and browser, 65,536 objects, 32 buckets, 99% visibility, and eight physical depth bins. The two candidates were consecutive same-session replications on one device, not independent hardware replications. Front-to-back and reverse traversal used the same four kernels, shader operations, resources, static bundle, material, geometry, and 32 native indirect draws; only the traversal uniform differed. Atomic fixed-slice was a contextual whole-mechanism comparator.

Each candidate completed all 36 trials and 8,640 measured frames. Both passed exact survivor, command, bin-layout, traversal-normalized sequence, full-frame color/depth/object-ID, static-bundle lifecycle, source-provenance, and artifact-integrity checks. The table reports median paired deltas of per-trial p50s. Negative front-to-back-minus-comparator values favor front-to-back.

| Replication | High-overlap render: front-to-back minus reverse | Front-to-back wins | Low-overlap render control | High-overlap total GPU: front-to-back minus atomic | Low-overlap total GPU: front-to-back minus atomic | Preregistered result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | +0.094 ms (+4.1%) | 2 / 6 | +0.049 ms (+4.1%) | +1.063 ms (+55.1%) | +1.155 ms (+58.7%) | Fail |
| B | -0.237 ms (-10.6%) | 5 / 6 | +0.050 ms (+2.1%) | +0.912 ms (+49.1%) | +1.216 ms (+66.5%) | Fail |

Replication A failed the material high-overlap benefit and direction-stability gates. Replication B passed the aggregate high-overlap benefit and five-of-six win gates, but failed position stability: the two pairs with front-to-back in absolute position zero had a +0.128 ms median render delta. Both low-overlap controls remained inside the preregistered 0.10 ms and 10% equivalence bounds.

The replications therefore do not establish a stable render-ordering benefit. Their high-overlap estimates changed sign, and the second result remained sensitive to trial position. This is consistent with a possible early-depth-rejection effect confounded by temporal or order drift, but it is not evidence sufficient to attribute or size that effect.

The deterministic fourth kernel deliberately assigns one scatter owner per bucket so independently generated front-to-back and reverse survivor sequences can be proven identical. That diagnostic design costs approximately 0.84-0.86 ms of compute per frame, compared with approximately 0.004 ms for atomic fixed-slice, and makes the complete ordered mechanism materially slower. It is an attribution instrument, not a deployable optimization.

The next experiment should remove compaction from the causal contrast: generate and validate frozen front-to-back and reverse survivor lists before timing, then alternate the two render orders within trials or short balanced blocks. It should retain the same bundle, draw, material, geometry, exact-output, and object-ID gates. A scalable parallel scan or radix compaction path is warranted only if that render-only signal is stable.

## Evidence identifiers

| Comparison | Replication | Run identifier |
| --- | --- | --- |
| Three Blocks 0.10 historical | A | `ecosystem-o16384-b32-2026-08-31T08-31-28.754Z` |
| Three Blocks 0.10 historical | B | `ecosystem-o16384-b32-2026-08-31T20-05-45.150Z` |
| Three Blocks 0.11 coalesced diagnostic | A | `ecosystem-o16384-b32-2026-08-31T20-09-51.773Z` |
| Three Blocks 0.11 coalesced diagnostic | B | `ecosystem-o16384-b32-2026-08-31T20-13-07.746Z` |
| Coarse depth ordering | A | `depth-ordering-o65536-b32-2026-08-31T22-21-43.907Z` |
| Coarse depth ordering | B | `depth-ordering-o65536-b32-2026-08-31T22-24-21.618Z` |

Generated run directories remain ignored source artifacts. Each identifier above names a manifest-bound local directory containing frame-level CSV data, metadata, trial summaries, validation payloads, workload manifests, GPU telemetry, and SHA-256 commitments. The analyzer rejects incomplete trials, changed source provenance, mismatched workload links, and altered required artifacts.

## What the evidence establishes

- Fixed geometry ownership can materially simplify GPU visibility compaction in the tested Three.js WebGPU workload.
- The two-dispatch fixed-slice compute stage is reproducibly cheaper than both the published nine-dispatch historical heterogeneous path and the 128-dispatch current-package diagnostic.
- One retained mesh/render object removes a CPU submission cost that grows with bucket count.
- The current-package comparison has a real crossover: fixed-slice wins through compute efficiency while its render-only path is slower.
- The tested bucket-serial depth-ordering mechanism is not a total-GPU optimization, and its render-only contrast was not stable enough to support an ordering claim.

## What remains open

- A second materially different GPU family is required before generalizing the crossover or percentage reductions.
- The current-package result is a diagnostic comparison, not stock public-API timing.
- Production assets, textures, dynamic cameras, moving objects, shadows, transparency, skinning, and morph targets are outside this result.
- The benchmark does not measure presentation latency or queue overlap.
- Whether coarse front-to-back traversal improves render-only time after within-trial temporal drift is controlled.

The next GPU-side experiment should isolate the last point with frozen, prevalidated survivor buffers and tightly interleaved render-order measurements. Only a stable render benefit would justify implementing and timing a scalable GPU ordering pipeline.
