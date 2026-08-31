# Frozen depth-order render crossover protocol

## Research question

Does coarse per-bucket front-to-back instance traversal reduce Three.js WebGPU render-pass time in the high-overlap scene when survivor generation, compute work, draw commands, geometry, material, render-object topology, and bundle recording are removed from the timed contrast?

This experiment follows the inconclusive coarse depth-ordering candidates. It is a new preregistered matrix, not a post hoc reinterpretation of those runs. A pass would establish only a render-order signal for a frozen opaque scene. It would not establish that a GPU ordering pipeline improves total GPU time.

## Fixed workload and attribution object

- Three.js 0.185.1 and the repository's pinned dependency lock;
- 65,536 objects, 32 indexed geometry buckets, and 99% target visibility;
- the existing high-overlap layout and low-overlap negative control;
- a 1280 by 720 viewport, device pixel ratio 1, fixed camera and lighting, no antialiasing, and reversed depth;
- eight physical bins using sphere-nearest view depth, `-viewZ - radius`;
- one immutable `Uint32` survivor buffer containing exactly `2 * O` elements;
- one matrix buffer, merged indexed geometry, indirect-command buffer, material, shader pipeline, mesh, and static `BundleGroup`;
- exactly 32 native indexed indirect draws and one render call per animation frame.

The survivor buffer contains one complete front-to-back lane and one complete reverse lane. The assignment of those lanes to physical base zero and physical base `O` is counterbalanced across repetitions. Both use identical ascending object-ID order inside each physical bucket/bin. A uint `laneBase`, whose value is exactly zero or `O`, is the only lane-varying shader input. The vertex-stage index is:

```text
visibleIds[laneBase + bucketBase + instanceIndex]
```

The CPU constructs and uploads both lists before timing. That work is an untimed attribution instrument, not a proposed production implementation. No compute pass is part of the timed workload, and no survivor, command, matrix, geometry, or material write is permitted from warmup start through postflight. The address-selection challenge is a diagnostic exception: with timestamp tracking disabled, it copies the selected immutable lane into a dedicated readback buffer at preflight, timing start, and postflight. It does not write any render input and is excluded from warmup and measurement.

## Frozen-buffer and output gates

Timing is refused unless all checks pass:

- full GPU readback of the `2 * O` buffer has the exact byte length and expected SHA-256 commitment;
- each physical half exactly matches an independent bucket/bin reconstruction, including per-bucket slice bounds, visible counts, ordered object-ID sequence, and unused slice padding;
- both halves have exact expected membership and identical traversal-normalized physical-bin sequence commitments;
- the 32 shared five-word indexed indirect commands contain the expected index count, instance count, cumulative first index, zero base vertex, and zero `firstInstance`;
- the lane-to-base mapping is one of the two preregistered assignments and matches the plan;
- an untimed address-selection challenge proves that both legal `laneBase` values select their committed physical segment through the same indexing contract used by the render material;
- two stable offscreen captures for each lane have exact full-frame RGBA8 color, depth32float, and RGB24 object-ID commitments, nonzero coverage, and no non-visible or out-of-range ID;
- the two lanes have identical color, depth, and object-ID commitments;
- the same exact validation and parity commitments recur after timing;
- the renderer reports no page, console, or WebGPU validation error.

The single static bundle must be recorded exactly once before timing and never rebuilt. Geometry, material, mesh, indirect buffer, survivor buffer, matrix buffer, pipeline/binding identities, and camera/viewport facts must remain unchanged. Validation records the complete buffer and semantic commitments at preflight, timing start, and postflight. Diagnostic selector submissions and readbacks occur only while timestamp tracking is disabled; the lifecycle gate requires exactly zero compute submissions during the 800 warmup and measured render frames.

## Crossover schedule

One render call occurs in each animation frame. Multiple renders in one animation frame are prohibited because Three.js r185 assigns them the same frame number in timestamp identifiers. Each eight-frame superblock uses one of two complementary patterns:

```text
FRFR | RFRF
RFRF | FRFR
```

`F` denotes front-to-back and `R` reverse. Each lane occupies four positions with the same mean temporal position inside every superblock. The two orientations alternate, with a preregistered starting offset. Warmup contains 40 superblocks, or 320 render frames. Measurement contains 60 superblocks, or 480 retained rows: 240 per lane and exactly 30 superblocks of each orientation.

Before every render, including repeated lane values, the harness must perform one audited selector write. Each row records the superblock index, orientation, within-superblock position, planned lane, physical base, setter serial, render serial, and unique GPU frame ID. The schedule is hash-bound before timing. Analysis rejects a missing, duplicate, reordered, or extra setter/render event, any base other than zero or `O`, any row that disagrees with the planned lane-to-base mapping, or any unexpected compute duration.

Three.js r185 uses two timestamp queries per render pass. Warmup therefore consumes 640 queries and measurement 960, each below the pinned 2,048-query render-pool capacity. The pool is resolved and discarded after warmup and resolved for collection after measurement. Every retained frame must have one unique render timestamp and no compute timestamp. GPU timestamp quantum must be no greater than 10,000 ns. Compilation, readback, parity capture, and address challenges occur with timestamp tracking disabled.

## Trial plan and estimator

The matrix contains 12 repetitions and both layouts, for exactly 24 paired trials and 11,520 retained render rows. Layout order, front/reverse physical-buffer assignment, and starting superblock orientation are pairwise balanced by the committed plan. All 24 trials must complete; there is no efficacy or futility stopping.

For each superblock, analysis computes:

```text
mean(front-to-back GPU render time) - mean(reverse GPU render time)
```

The trial estimate is the median of its 60 superblock deltas. Percent deltas use the corresponding reverse mean as denominator before the trial median. Frames and superblocks are repeated observations, not independent replicates; the 12 repetition-level trial estimates are the inferential units for each layout. No frame or block is deleted as an outlier.

## Preregistered decision gates

Negative values favor front-to-back. A first-device render-order result passes only if every gate below passes:

- the high-overlap median across the 12 trial estimates is at most -0.10 ms or at most -10%;
- at least 10 of 12 high-overlap trial estimates are strictly negative;
- the high-overlap median is negative in both physical-base strata, both starting-orientation strata, both layout-order positions, and both measurement halves;
- each high-overlap stratum interaction remains strictly inside both 0.10 ms and 10%;
- the low-overlap median remains strictly inside both -0.10 to +0.10 ms and -10% to +10%;
- for each low-overlap absolute and percentage bound, at least 10 of 12 trial estimates lie above the lower bound and at least 10 of 12 lie below the upper bound;
- the high-overlap minus low-overlap median paired effect is at most -0.10 ms;
- condition-blind first-versus-last-quarter pooled render drift has an absolute median strictly below both 0.10 ms and 5%, including within each layout stratum;
- every schedule, timestamp, frozen-buffer, exact-output, lifecycle, environment, telemetry, source-provenance, and artifact-integrity gate passes.

Ten same-direction results out of 12 have exact one-sided sign probability 0.0193 under a fair null. The low-overlap bound checks use the same 10-of-12 rule on both sides as a sign-based equivalence requirement. These discrete rules are fixed before timing and are not replaced with frame-level significance tests.

## Stopping and replication

A trial may be replaced only for a prespecified technical failure detected without inspecting performance deltas. Its failed artifacts remain preserved, and the complete two-layout repetition is restarted. Directional or near-threshold timing is not a technical failure.

Exactly two full candidate matrices are required in separate benchmark sessions. A render-order claim on the first device requires both matrices independently to pass every primary, control, stability, and evidence gate without changing the protocol between runs. Only then should work begin on a scalable parallel GPU scan, radix, or prefix-compaction implementation and a separate total-GPU evaluation.
