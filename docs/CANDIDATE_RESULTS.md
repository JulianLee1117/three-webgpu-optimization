# Candidate results

## Status and scope

These results are candidate evidence from one machine and are governed by experiment-specific decision rules. Some comparisons below meet their declared single-device replication rules; neither completed indirect-`firstInstance` study meets its two-matrix confirmation rule. The first standalone deployment capture is incomplete and analysis-ineligible, so it adds no efficacy estimate or decision. None establishes a general WebGPU result.

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

The follow-up below removes compaction from the causal contrast: it generates and validates frozen front-to-back and reverse survivor lists before timing, then alternates the two render orders within complementary eight-frame blocks. It retains the same bundle, draw, material, geometry, exact-output, and object-ID gates. The design was fixed before measurement in the [Frozen render-order crossover protocol](FROZEN_DEPTH_CROSSOVER_PROTOCOL.md).

## Frozen render-order crossover result

The frozen crossover used source commit `deda556564e5f60ab6d461e68ad38a9d9a88fa1b`, the same RTX 5070 Ti / D3D12 system and browser, 65,536 objects, 32 buckets, 99% visibility, and the same high- and low-overlap layouts. Front-to-back and reverse survivor sequences occupied the two halves of one immutable storage buffer. Both lanes shared the matrix buffer, geometry, indirect commands, material, mesh, pipeline, static bundle, and 32 native draws; a base-offset uniform was the only timed lane difference. No compute submission occurred during warmup or measurement.

Each candidate completed all 24 trials and 11,520 measured render rows. Both passed the full frozen-buffer readback, selector-address challenge, indirect-command, exact color/depth/object-ID, timestamp-identity, static-resource, lifecycle, reversed-depth, source-provenance, and artifact-integrity gates. GPU telemetry contained no malformed samples or material gaps, retained the same process set before and after each run, sustained the performance clock state during measurement, and showed no thermal-throttling pattern. Negative values below favor front-to-back.

| Replication | High-overlap render: front-to-back minus reverse | Front-to-back wins | Low-overlap control | Paired high-minus-low effect | Condition-blind pooled drift | Preregistered result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | +0.000886 ms (+0.0486%) | 5 / 12 | -0.003126 ms (-0.1529%) | +0.001570 ms (+0.0760%) | -0.006626 ms (-0.3534%) | Fail |
| B | +0.001014 ms (+0.0500%) | 5 / 12 | -0.002614 ms (-0.1295%) | +0.006910 ms (+0.3068%) | -0.002090 ms (-0.1014%) | Fail |

Both replications passed the low-overlap equivalence and drift gates. Both failed the material high-overlap threshold, the 10-of-12 direction rule, the requirement for a negative high-overlap median in every counterbalanced stratum, and the paired high-minus-low threshold. The high-overlap point estimates were approximately one microsecond and had the opposite sign from the proposed benefit.

The tightly interleaved render-only experiment therefore finds no material front-to-back benefit for this frozen opaque workload. The coarse candidate's apparent -10.6% result does not survive removal of between-trial temporal and ordering effects. On this evidence, a scalable GPU scan, radix, or prefix-compaction pipeline for coarse depth ordering is not warranted. This conclusion is specific to the tested scene and does not weaken the independently reproduced fixed-ownership compaction result.

## Indirect `firstInstance` render crossover result

The indirect-addressing crossover used source commit `cd461b58a9bd84a46904fa610469f8b1f92de7d7`, Three.js 0.185.1, Chrome 151.0.7922.174, and the same RTX 5070 Ti / D3D12 system. It fixed 65,536 objects, 32 medium indexed geometry buckets, 99% and 20% visibility cells, a 1280 x 720 target, reversed depth, and one immutable bucket-sliced survivor list. WebGPU reported `indirect-first-instance`; both runs observed a 32 ns GPU timestamp quantum.

The portable lane retained a per-vertex uint bucket base and zero `firstInstance`. The feature lane removed that vertex input and addition and placed the exact bucket-slice base in the fifth word of each indexed indirect command. Both lanes shared the common index, position, normal, and UV attributes; matrix and survivor storage; material output; 32 native draws; and fixed camera and target. Each candidate completed all 24 trials and 11,520 retained rows with zero timed compute submissions.

Both candidates passed full survivor and indirect-buffer readback, raw WGSL and runtime binding inspection, the all-address RGBA8 oracle, exact color/depth/object-ID parity, static-bundle and resource lifecycle checks, source provenance, artifact reconstruction, and telemetry coherence. The same GPU process set was present before and after each run. Across the complete runs, telemetry recorded P1 for 374 of 377 and 375 of 378 samples; median temperature was 49 C and 51 C, with maxima of 54 C and 56 C. Negative feature-minus-portable values favor indirect `firstInstance`.

| Replication | 99% render delta | Feature wins | 20% render delta | Paired 99%-minus-20% | Condition-blind pooled drift | Preregistered result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | -0.345978 ms (-17.56%) | 12 / 12 | -0.060206 ms (-15.47%) | -0.292290 ms | +0.000007 ms (+0.001%) | Fail |
| B | -0.337666 ms (-17.12%) | 12 / 12 | -0.055124 ms (-14.45%) | -0.295466 ms | +0.001189 ms (+0.126%) | Pass |

Replication A passed the material-effect, direction, low-visibility, paired-dose, and drift gates. It failed the predeclared high-visibility nuisance-interaction bounds. Physical command placement differed by 0.119074 ms and 7.56% between strata, exceeding the strict 0.10 ms and 5% limits; visibility-order position differed by 5.99%, exceeding the percent limit. Both levels of every stratum still favored the feature lane: the physical-placement medians were -0.377092 ms and -0.258018 ms, and the visibility-order medians were -0.282196 ms and -0.377092 ms. Replication B kept every interaction within both bounds and passed all numerical gates.

The required two-matrix confirmatory decision is therefore not met, and the failed first matrix is not replaced or reinterpreted. At the same time, the aggregate high-visibility direction and magnitude recurred closely: both session medians were approximately -17%, and all 24 high-visibility repetition estimates favored the feature lane. This is a material-sized same-device render-stage signal for the tested address-path contrast, but the protocol does not confirm a stable first-device result, a general WebGPU optimization, or a deployable timestamped GPU-pass win.

The next decision experiment retained the zero-`firstInstance` fallback and compared normal live compute-plus-render operation. It required exact output and command parity and a reproduced timestamped GPU-pass improvement of at least 0.10 ms and 5%, as fixed before these candidate measurements. Its outcome follows.

## Live indirect `firstInstance` compute-plus-render result

The live crossover used source commit `aa577d7c3725469ed04005b236009e81a4a764d9`, annotated anchor tag `first-instance-live-candidate-7b343849f9dc3e8a`, and study key `7b343849f9dc3e8adcf13722eac937df06ce9841bfd9f532446d1e9f9f77aab4`. It used Three.js 0.185.1, Chrome 151.0.7922.174, the same RTX 5070 Ti / D3D12 system, 65,536 objects, 32 indexed geometry buckets, 99% and 20% visibility cells, a 1280 x 720 timed target inside a 1280 x 900 browser viewport, and reversed depth.

The portable and feature lanes retained the render-only study's exact addressing contrast while running the normal culling compute and render phases. Each matrix completed all 24 trials and retained 11,520 measured rows. Both passed exact survivor, indirect-command, shader, runtime-binding, timestamp-identity, color/depth/object-ID, resource-lifecycle, source-provenance, artifact, process-identity, telemetry-coverage, and adapter-to-GPU association checks. There were no retries or invalid attempts. Negative feature-minus-portable values favor indirect `firstInstance`.

| Matrix | 99% timestamped GPU-pass delta | 99% render delta | Feature wins | 20% timestamped GPU-pass delta | Lane-physical-order interaction | Preregistered result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | -0.248552 ms (-11.50%) | -0.248516 ms (-11.52%) | 12 / 12 | -0.042942 ms (-10.88%) | +0.141034 ms (+8.004 percentage points) | Fail |
| B | -0.257194 ms (-12.49%) | -0.257112 ms (-12.52%) | 12 / 12 | -0.045258 ms (-11.50%) | -0.060440 ms (-2.026 percentage points) | Pass |

Matrix A passed the aggregate material-effect, direction, render, low-visibility, paired-dose, carryover, drift, and three of four high-visibility nuisance-factor gates. It failed only the lane-physical-order interaction bound: the two order-stratum medians were -0.305812 ms (-16.04%) and -0.164778 ms (-8.03%), so both levels favored the feature lane but their separation exceeded the predeclared 0.10 ms and 5 percentage-point limits. Matrix B reversed the interaction direction and kept it within both limits. Construction order, first compute use, and render-pipeline priming are bundled in this factor, so the evidence does not attribute the session-dependent interaction to one cause.

The timestamped GPU-pass effect was almost entirely render time; the high-visibility compute deltas were -0.000028 ms in Matrix A and +0.000002 ms in Matrix B. The aggregate direction and magnitude therefore reproduce the earlier addressing-path signal under normal culling, but the strict conjunctive pair decision remains `confirmation-not-met`: both matrices were required to pass independently. The failed first matrix is retained and is not replaced or reinterpreted. This is strong same-device evidence of a material render-driven effect, not confirmation of a deployable optimization or a general Three.js/WebGPU result.

The deterministic public package `first-device-live-7b343849f9dc3e8a.tar.br` contains the sanitized pair and receipt, is 4,486,242 bytes, and has SHA-256 `5cb66f15b53c5360785a063fd2a46d82cd1041a97827cfd354e47bb589467b99`. Its archive, embedded receipt, and public-run commitments pass the strict package verifier while preserving the `confirmation-not-met` decision.

## Setup-order development diagnostic

The bounded follow-up used source commit
`2d9acf5cc12ad018c92e217c5bd92c2ee01662c4` and the same 65,536-object,
32-bucket live workload. It separated four exposures that had been bundled by
the candidate's lane-physical-order factor: resource construction (`C`), first
compute use (`K`), render-pipeline and bundle priming (`R`), and timestamp-pool
prepriming (`T`). Two fresh browser/device sessions each executed one frozen
16-cell permutation followed by its exact reverse. The complete diagnostic
contains 64 trials, 30,720 retained measured rows, and no retry, replacement,
or outlier deletion.

Every trial passed the exact output, survivor, command, addressing, shader,
timestamp, and resource-lifecycle gates. The independent verifier reconstructed
all 64 trial summaries and all 15 factorial contrasts from 72 declared
artifacts, checked 384 fresh shader-observation challenges, and confirmed exact
source, installed-dependency, and served-module identity. Its decision remains
`diagnostic-only-no-candidate-pass`. Negative values below favor the feature
lane.

| Scope | Timestamped GPU-pass delta | Render delta | Compute delta | Negative session-cell responses |
| --- | ---: | ---: | ---: | ---: |
| Session 1 | -0.295791 ms (-14.6050%) | -0.296116 ms (-14.6597%) | +0.000007 ms (+0.1684%) | 16 / 16 |
| Session 2 | -0.271902 ms (-13.8958%) | -0.271938 ms (-13.9307%) | -0.000003 ms (-0.0637%) | 16 / 16 |
| Equal-session pooled cells | -0.283847 ms (-14.2504%) | -0.284027 ms (-14.2952%) | +0.000002 ms (+0.0524%) | 16 / 16 in both sessions |

Render priming was the only large, session-concordant main effect. With
portable primed first (`R=0`), the pooled feature-minus-portable response was
-0.324272 ms; with feature primed first (`R=1`), it was -0.243422 ms. The
level-one-minus-level-zero `R` contrast was therefore +0.080850 ms and +3.371
percentage points. It was +0.079379 ms in session 1 and +0.082321 ms in session
2, and remained positive in every session by forward/reverse block,
previous-lane stratum, and timing-half slice. Construction and first-compute-use
main effects were approximately -0.0021 and -0.0012 ms and changed sign between
sessions. The timestamp-preprime contrast was -0.0096 ms but was close to zero
in session 2.

The recorded renderer-program-byte transitions did not simply align with the
render-prime pattern. Each session had the same three timing-start states:
renderer program bytes stepped from 38,311 for positions 0-1, to 38,415 for
positions 2-15, and to 38,519 for positions 16-31. Pipeline counts,
compute-program counts, and all other recorded resource fields remained fixed,
and every individual trial held its start state through completion. The `R`
contrast retained the same direction
and similar magnitude in both reverse blocks, where the cache state was fixed.
Occurrence-level history and drift nevertheless remain aliased with the frozen
schedule and cannot be assigned a causal interpretation.

A contemporaneous Windows GPU-engine sample attributed approximately 13-15%
3D utilization to the development environment. That observation is outside the
manifest-bound evidence and further limits the timing values to mechanism
diagnosis. The run does not amend either completed candidate matrix, establish
that `R` caused the earlier interaction, or support a deployment or cross-device
claim. It does justify continuing with a standalone topology in which exactly
one lane is constructed, resident, and primed per fresh browser/device session;
that design removes render-prime order from the comparison rather than treating
it as a nuisance to average away.

## Standalone deployment v1 capture failure

The first full standalone capture used clean source commit
`52ec935e44bcaf147e1c9a68f1a79f03c30c0144` and retained its failed artifact at:

```text
results/candidate-standalone-deployment/first-instance-standalone-deployment-2026-09-03T07-29-33-833Z-959a6905/
```

It stopped while opening Matrix 1 session 16 after committing 15 of 96 sessions
and 30 of 192 trials. No matrix completed. `failure.json` has SHA-256
`838424d99ed429831ba28468a6e7f5d396fa6ca074e2f601b01bca54ca7ae1d1`;
the directory contains 98 files totaling 12,966,684 bytes, and all 96 referenced
artifact descriptors independently matched their recorded lengths and hashes.
The terminal error was a 120-second page-readiness timeout. The record is
explicitly `analysisEligible: false`; the partial trials have not been and must
not be summarized for portable-versus-feature efficacy.

The timeout was not an independently logged browser/device loss,
telemetry-collector failure, or artifact-I/O failure, so the frozen retry rule
did not authorize another v1 launch. There was no retry, resume, or replacement.
The retained failure therefore neither passes nor fails the standalone
two-matrix efficacy decision: no such decision exists for v1.

Windows recorded Tcpip Event ID 4231, record 18340, at
`2026-09-03T07:32:53.263605Z`, approximately 25 ms after the failed page was
created. That warning reported failure to allocate an ephemeral port from the
global TCP port space. Its timing and the readiness-failure shape make transient
host-wide port exhaustion the strongest supported cause. The attribution is not
conclusive because v1 did not persist page console, failed-request, HTTP-error,
crash, or bounded readiness diagnostics, and Event 4231 does not identify the
responsible process. This finding is infrastructure evidence only.

A corrected v2 series requires a new clean committed source identity, terminal
failure/interruption schema v2, and the separate one-gate-plus-32-session
[startup-endurance qualification](INDIRECT_FIRST_INSTANCE_STANDALONE_STARTUP_ENDURANCE_PROTOCOL.md).
That one-attempt qualification rejects an Event 4231 in the prior 10 minutes
and requires at least `max(4096, ceil(25% of the configured range))` free unique
dynamic local ports before and after each of its 33 browser lifecycles. It
performs no trial timing
or efficacy analysis. Only the exact qualified source and dependency closure
may write a single corrected candidate under
`results/candidate-standalone-deployment-v2/first-instance-standalone-deployment-v2-.../`.
The original workload, schedule, estimators, thresholds, correctness rules, and
two-complete-matrix requirement remain unchanged.

## Standalone deployment v2 result

The exact corrected source first passed its one-shot, analysis-ineligible
startup-endurance qualification. The full v2 candidate then completed from
clean source commit `3ff0b6f4674c062f7dd9e83c0618a604d6626b4d` with unchanged
installed dependency closure
`2de960326f2740a20e836b5e4228c3aa0ed0aba58e670856706b73abd0c6d388`.
The immutable candidate identifier is:

```text
first-instance-standalone-deployment-v2-2026-09-03T17-56-15-996Z-3b0cae96
```

It contains all 192 trials, 96 measured sessions, two matrices, 92,160 retained
rows, and 98 non-overlapping fresh browser processes including the forced-off
gates. The captured environment was Three.js r185 / npm 0.185.1,
HeadlessChrome 152, an RTX 5070 Ti through D3D12 driver 32.0.16.1656, and a
32 ns timestamp quantum. Negative values favor indirect `firstInstance`.

| Endpoint | Matrix 1 | Matrix 2 |
| --- | ---: | ---: |
| 99% timestamped GPU pass | +0.0672225 ms (+3.55123%); 4/12 negative | +0.1435670 ms (+7.90525%); 3/12 negative |
| 99% render | +0.0671855 ms (+3.55720%); 4/12 negative | +0.1435655 ms (+7.92373%); 3/12 negative |
| 20% timestamped GPU pass | +0.0059525 ms (+1.61142%) | +0.0181605 ms (+5.08588%) |
| Paired 99%-minus-20% | +0.0615680 ms (+2.02039 pp) | +0.1248765 ms (+2.68676 pp) |
| Preregistered result | Fail | Fail |

Both matrices failed the high-visibility GPU-pass, render, low-visibility,
paired-dose, and high-visibility nuisance-factor gates. Both passed every
condition-blind session-sequence and within-trial drift gate. Their median 99%
compute responses were only +0.0000005 ms and +0.0000035 ms, so the observed
contrast is render-driven.

Independent verification returned `status: consistent`,
`technicalGatePass: true`, and
`decision: standalone-confirmation-not-met`. Every correctness, exact-output,
shader, resource-lifecycle, timestamp-continuity, source, dependency,
served-module, browser-lifecycle, telemetry, and artifact gate passed. The
manifest SHA-256 is
`acd70bd77330edc8d4cb2a29e88333a8dc7a9fa02971d315a7b0deaabc6b855f`.
All 598 declared artifacts matched their lengths and stored/logical hashes; the
directory has 599 files and 80,289,683 stored bytes in total. The verifier's
`authenticityVerified: false` correctly means internal consistency, not an
external signature or independent-origin attestation.

The fresh-session responses are sharply multimodal. At 99% visibility, 40/48
feature sessions cluster at 1.959178-1.961420 ms while 8/48 cluster at
1.503918-1.504968 ms. For portable, 39/48 cluster at
1.815644-1.817742 ms while 9/48 cluster at 1.968966-1.969780 ms. These modes
persist into the same browser's 20% trial despite essentially equal measured GPU
clocks. The preregistered direction and nuisance gates correctly reject a
positive claim under this mixture. The result is not evidence that
`firstInstance` is inherently slower; it is evidence that the proposed
deployment benefit does not survive the tested process-state modes.

The earlier positive live candidates used Chrome 151, whereas this candidate
used HeadlessChrome 152. The contemporaneous v2 lane contrast is valid, but the
sign reversal relative to the earlier candidates cannot be causally assigned to
standalone topology alone. No matrix will be retried or replaced. The next
mechanism study adds a zero-`firstInstance`, no-bucket-vertex-stream WebGPU
immediate-data lane to distinguish address sources.

## Evidence identifiers

| Comparison | Replication | Run identifier |
| --- | --- | --- |
| Three Blocks 0.10 historical | A | `ecosystem-o16384-b32-2026-08-31T08-31-28.754Z` |
| Three Blocks 0.10 historical | B | `ecosystem-o16384-b32-2026-08-31T20-05-45.150Z` |
| Three Blocks 0.11 coalesced diagnostic | A | `ecosystem-o16384-b32-2026-08-31T20-09-51.773Z` |
| Three Blocks 0.11 coalesced diagnostic | B | `ecosystem-o16384-b32-2026-08-31T20-13-07.746Z` |
| Coarse depth ordering | A | `depth-ordering-o65536-b32-2026-08-31T22-21-43.907Z` |
| Coarse depth ordering | B | `depth-ordering-o65536-b32-2026-08-31T22-24-21.618Z` |
| Frozen render-order crossover | A | `depth-ordering-render-only-o65536-b32-2026-09-01T02-24-19.996Z` |
| Frozen render-order crossover | B | `depth-ordering-render-only-o65536-b32-2026-09-01T02-27-00.288Z` |
| Indirect `firstInstance` crossover | A | `first-instance-render-only-o65536-b32-2026-09-01T05-42-36.298Z` |
| Indirect `firstInstance` crossover | B | `first-instance-render-only-o65536-b32-2026-09-01T05-44-58.948Z` |
| Live indirect `firstInstance` | A | `first-instance-live-o65536-b32-2026-09-01T16-47-53.050Z` |
| Live indirect `firstInstance` | B | `first-instance-live-o65536-b32-2026-09-01T16-50-07.577Z` |
| Setup-order factorial | Development diagnostic | `first-instance-live-order-factorial-2026-09-01T18-25-27-021Z-ace6b5c5` |
| Standalone deployment v1 | Failed, analysis-ineligible capture | `first-instance-standalone-deployment-2026-09-03T07-29-33-833Z-959a6905` |
| Standalone deployment v2 | Completed; confirmation not met | `first-instance-standalone-deployment-v2-2026-09-03T17-56-15-996Z-3b0cae96` |

Generated run directories remain ignored source artifacts and are not part of the tracked repository. Each completed identifier above names a manifest-bound local directory containing frame-level data, metadata, trial summaries, validation payloads, workload manifests, and SHA-256 commitments. Candidate directories also retain their required GPU telemetry; the setup-order development diagnostic does not. The failed standalone v1 identifier instead names a `failure.json`-bound partial directory with partial telemetry and no completion manifest. Its artifact descriptors were checked directly, but it is excluded from all candidate analysis. The completed v2 directory passed its independent verifier with 598 declared artifacts and zero inventory, length, hash, decompression, parse, or semantic mismatch. The analyzers reject incomplete trials, changed source provenance, mismatched workload links, and altered required artifacts; this establishes internal artifact consistency, not independent authenticity.

## What the evidence establishes

- Fixed geometry ownership can materially simplify GPU visibility compaction in the tested Three.js WebGPU workload.
- The two-dispatch fixed-slice compute stage is reproducibly cheaper than both the published nine-dispatch historical heterogeneous path and the 128-dispatch current-package diagnostic.
- One retained mesh/render object removes a CPU submission cost that grows with bucket count.
- The current-package comparison has a real crossover: fixed-slice wins through compute efficiency while its render-only path is slower.
- The tested bucket-serial depth-ordering mechanism is not a total-GPU optimization, and the frozen crossover finds no material render-order benefit after controlling within-trial time and physical-buffer placement.
- Two same-device matrices observed approximately 17% lower timestamped render-pass time with indirect `firstInstance` addressing in the tested high-visibility fixed-slice path, but this remains an unconfirmed signal because one matrix failed nuisance-interaction bounds and the required two-matrix decision was not met.
- Normal live culling preserved that addressing-path direction: two further matrices observed approximately 11.5-12.5% lower high-visibility timestamped GPU-pass time, almost entirely in render, but one matrix again failed a strict order-interaction bound and the pair did not confirm.
- The setup-order development diagnostic preserved a negative feature-minus-portable response in every factorial cell and localized a repeatable render-priming-order sensitivity. This is diagnostic evidence, not a candidate or deployment result.
- The standalone v1 capture establishes only a preserved, internally checked infrastructure failure. Its partial trials establish no portable-versus-feature effect.
- The corrected standalone v2 comparison completed with every technical gate passing, but both matrices numerically favored the portable lane at their medians and the preregistered confirmation was not met.
- Standalone session estimates occupy discrete, lane-associated performance modes that are not explained by the captured GPU-clock telemetry. This invalidates a simple inherent-speed interpretation in either direction and identifies a process-local render-pipeline mechanism for follow-up.

## What remains open

- A second materially different GPU family is required before generalizing the crossover or percentage reductions.
- The current-package result is a diagnostic comparison, not stock public-API timing.
- Production assets, textures, dynamic cameras, moving objects, shadows, transparency, skinning, and morph targets are outside this result.
- The benchmark does not measure presentation latency or queue overlap.
- A different scene, material cost, resolution, depth distribution, or GPU may have a different render-order crossover; that would require a new preregistered experiment rather than extending this null result.
- The cause of the discrete fresh-browser performance modes remains unobserved. The earlier positive runs used Chrome 151 and v2 used HeadlessChrome 152, so browser change and deployment topology cannot be separated retrospectively.
- WebGPU immediate data now permits a three-way address-source control: per-vertex base, immediate base with zero `firstInstance`, and indirect `firstInstance`. Three.js does not yet expose that path.

The frozen render-order crossover closes the ordering branch for the current fixture. Across the live, setup-order, and standalone work, the indirect-addressing contrast remains render-driven while changing sign under materially different execution conditions. No additional A/F rescue run is warranted. The next mechanism branch is an A/I/F immediate-data attribution study with exact address and output parity. Exact occlusion headroom remains a later branch if that study does not yield a stable, upstreamable result.
