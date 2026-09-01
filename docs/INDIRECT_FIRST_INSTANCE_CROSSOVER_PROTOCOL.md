# Frozen indirect `firstInstance` render crossover protocol

## Research question

Within the repository's fixed-slice heterogeneous representation, does moving
each bucket's survivor-list base from a per-vertex `uint` attribute into the
fifth word of its indexed indirect command materially reduce Three.js WebGPU
render-pass time?

This protocol follows a development-only sequential screen that justified a
more rigorous experiment. That screen is not candidate evidence: it rebuilt
each lane separately, reran atomic compaction, did not preserve raw frame rows,
and was exposed to between-trial device drift. The crossover below freezes the
render inputs and alternates the two address paths inside balanced eight-frame
blocks.

The optimization is a feature-gated integration specialization, not a claim
that `firstInstance` is new. WGSL defines `@builtin(instance_index)` to begin at
the draw's `firstInstance`; WebGPU permits nonzero indirect values only when the
optional `indirect-first-instance` feature is enabled; and prior engines and
libraries use this facility. The research contribution under test is the
measured causal value of eliminating the redundant vertex input in this
Three.js fixed-ownership representation.

## Fixed workload and causal contrast

The candidate workload fixes:

- Three.js 0.185.1 and the committed dependency lock;
- 65,536 objects and 32 medium indexed geometry buckets;
- the deterministic baseline layout at 99% visibility and a 20% lower-dose
  safety/dose-response cell;
- the repository seed, fixed camera and lighting, 1280 by 720 target, device
  pixel ratio 1, no antialiasing, sample count 1, and reversed depth;
- one immutable bucket-sliced survivor buffer in ascending object-ID order;
- one immutable matrix buffer and one shared 64-record indirect buffer;
- exactly 32 indexed indirect draws, one selected visible render object and
  pre-recorded static bundle per timed frame, and one top-level render call.

The portable lane uses:

```text
visibleIds[bucketBase + instanceIndex]
command.firstInstance = 0
```

The feature lane uses:

```text
visibleIds[instanceIndex]
command.firstInstance = bucketBase
```

The two geometry shells share the exact same index, position, normal, and UV
`BufferAttribute` objects. Only the portable shell has `bucketBase`. The two
materials have identical visible properties and node logic outside the address
expression. Separate vertex pipelines, meshes, and static bundles are an
unavoidable consequence of the different vertex layout and address graph.

The indirect allocation contains one 32-command segment per lane. For every
bucket, words zero through three must be pairwise identical: index count,
visible instance count, cumulative first index, and zero base vertex. Word four
must be zero in the portable segment and the exact bucket-slice base in the
feature segment. Which lane occupies the first physical command segment is
counterbalanced across repetitions. The feature lane is refused unless
`renderer.hasFeature('indirect-first-instance')` is true. The portable path
remains the fallback.

No instance-stepped geometry attribute is permitted. A future addition of one
requires a new audit because `firstInstance` offsets instance-stepped inputs as
well as the WGSL builtin.

## Frozen inputs and all-address challenge

The CPU constructs the canonical survivor list and both command segments before
timing. This is an attribution instrument, not the proposed live culling path.
No compute dispatch or mutation of a render input is permitted during warmup or
measurement.

Full-frame parity alone cannot prove every survivor address because occluded
objects may contribute no final pixel. Each preflight, timing-start, and
postflight validation therefore runs an untimed fragment-stage address
challenge for both lanes. Mutable storage is not permitted in the WGSL vertex
stage, and the pinned Three.js renderer exposes render-stage storage bindings
as read-only. The challenge therefore uses a normal RGBA8 render attachment as
its proof channel rather than unsupported render-stage storage writes.

The challenge reuses the actual survivor buffer, physical indirect allocation,
lane-specific command offsets, and production address expressions. Its
dedicated indexed geometry preserves every production `indexCount` and
`firstIndex` span. The first triangle in each bucket is expressed in pixel-local
coordinates; every remaining triangle in that bucket's command span is
degenerate. The vertex shader maps the lane's computed address to one pixel in
a `256 x ceil(objectCount / 256)` single-sample target. That triangle contains
the addressed pixel center and no neighboring pixel center. The portable
diagnostic shell carries the exact bucket base across every synthetic vertex,
while the feature shell omits that input. The fragment shader writes the
survivor ID plus one as RGB24, leaving transparent all-zero bytes for unused
addresses. The attachment is explicitly `rgba8unorm`, has no color-space
conversion, depth, stencil, multisampling, scissor, blending, or fog, and uses
a 256-pixel width so each 1,024-byte row is already WebGPU-copy aligned. The
entire tightly packed `Uint8Array` target is read back.

The challenge passes only if every active fixed-slice address contains the
exact expected ID plus one, every unused slice address remains zero, both lanes
have byte-identical output, and the complete commitments repeat at all three
validation points. Before each lane draw, a manual transparent-black color
clear and readback must match the exact all-zero reset image, preventing a stale
target from passing. The diagnostic draw disables automatic clearing and
restores the renderer target, clear color, clear alpha, and auto-clear state.
Diagnostic clears, renders, readbacks, and shader validation occur
with timestamp tracking disabled and are excluded from warmup and measurement.

## Shader, geometry, and output gates

Timing is refused unless all checks pass:

- full survivor and indirect-buffer readbacks have the exact byte lengths and
  expected SHA-256 commitments;
- the survivor buffer exactly matches the independent bucket-slice
  reconstruction, including visible order, counts, bounds, and padding;
- all 64 indirect records and every paired word satisfy the declared command
  contract;
- the shared index/position/normal/UV attributes have identical object IDs,
  versions, formats, counts, and byte commitments in both geometry shells;
- the portable shell has one `uint` `bucketBase` vertex attribute with the
  exact per-bucket values, while the feature shell has no such attribute and
  neither shell has an instance-stepped input;
- raw WGSL is captured after both pipelines are primed: fragment shader bytes
  are identical, vertex shader bytes differ, the portable vertex input and
  addition are present, and the feature address is the direct
  `instance_index` path with no `bucketBase` declaration or reference;
- two stable captures per lane have exact full-frame RGBA8, depth32float, and
  RGB24 object-ID commitments, nonzero coverage, and no non-visible or
  out-of-range ID;
- portable and feature color, depth, and object-ID commitments are identical;
- the RGBA8 all-address challenge and its exact synthetic-geometry
  audit pass for both lanes;
- the same validation, shader, and output commitments recur after timing;
- the renderer reports no page, console, or WebGPU validation error.

Both static bundles must be recorded exactly once before timing and never
rebuilt. The parent, bundle, mesh, geometry, material, common vertex/index
attributes, matrix buffer, survivor buffer, indirect buffer, camera, target,
and viewport are identity- and version-bound at timing start and end. Three.js
r185 exposes no stable public pipeline or bind-group identity, so the audit
binds their inspectable proxies: raw and normalized WGSL, storage coordinates
and resources, and unchanged renderer pipeline-cache cardinality. The only
per-frame state change is the audited selection of which already-recorded
bundle is visible.

## Crossover schedule

`P` denotes portable bucket-base addressing and `F` the feature-gated indirect
`firstInstance` lane. Each eight-frame superblock uses one of two complementary
patterns:

```text
PFPF | FPFP
FPFP | PFPF
```

Each lane occupies four positions with the same mean temporal position inside
every block. The two orientations alternate, with a committed starting offset.
Warmup contains 40 blocks, or 320 render frames. Measurement contains 60
blocks, or 480 retained rows: 240 per lane and 30 blocks of each orientation.

Before every render, including repeated lane values, the harness performs one
audited lane selection. Each row records the block, orientation, position,
planned lane, physical command segment, selection serial, render serial, and
unique GPU frame ID. Analysis rejects a missing, duplicate, reordered, or extra
selection/render event; a lane/segment mismatch; any compute duration; or any
unexpected state mutation.

Three.js r185 uses two timestamp queries per render pass. Warmup consumes 640
queries and measurement 960, each below the pinned 2,048-query render-pool
capacity. Every retained frame must have exactly one render timestamp UID and
no compute timestamp. Timestamp quantum must be no greater than 10,000 ns.

## Trial plan and estimator

The matrix contains 12 repetitions and both visibility levels, for exactly 24
paired trials and 11,520 retained rows. Visibility order, physical command-
segment order, and starting schedule orientation are pairwise balanced by the
committed plan.

For every measured block, analysis computes:

```text
mean(feature GPU render time) - mean(portable GPU render time)
```

The trial estimate is the conventional median of its 60 block deltas; an even
median is the arithmetic mean of the two central values. Percentage deltas use
the corresponding portable block mean as denominator before taking the trial
median. Frames and blocks are repeated observations, not independent
replicates. The 12 repetition-level estimates are the inferential units for
each visibility cell. No frame or block is removed as an outlier.

## Preregistered decision gates

Negative values favor indirect `firstInstance` addressing. A first-device
render-only result passes only if every gate below passes:

- the 99% median across 12 trial estimates is at most -0.10 ms and at most -5%;
- at least 10 of 12 99% trial estimates are strictly negative;
- the 99% median is negative in both physical command-segment strata, both
  starting-orientation strata, both visibility-order positions, and both
  measurement halves;
- each 99% stratum interaction remains strictly inside both 0.10 ms and 5%;
- the 20% median regression is below both +0.02 ms and +5%, with at least 10 of
  12 estimates below each upper bound;
- the paired 99%-minus-20% median effect is at most -0.05 ms;
- condition-blind first-versus-last-quarter pooled render drift has an absolute
  median strictly below both 0.10 ms and 5%, including within each visibility
  stratum;
- every schedule, timestamp, frozen-buffer, address, shader, exact-output,
  lifecycle, environment, telemetry, source-provenance, and artifact-integrity
  gate passes.

Ten same-direction results out of 12 have exact one-sided sign probability
0.0193 under a fair null. The numerical thresholds define a practically
material effect; they are not replaced with frame-level significance tests.

## Stopping, replication, and deployment scope

A trial may be replaced only for a prespecified technical failure detected
without inspecting performance deltas. Its failed artifacts remain preserved,
and the complete two-visibility repetition is restarted. Directional or near-
threshold timing is not a technical failure.

Exactly two full candidate matrices are required in separate benchmark
sessions from the same frozen source. A first-device render-only claim requires
both matrices independently to pass every gate without changing the protocol.

The feature should be described as deployable only after a separate normal
compute-plus-render evaluation also reproduces a total-GPU improvement of at
least 0.10 ms and 5%, preserves exact output, and retains the zero-
`firstInstance` fallback. Until then, a passing crossover establishes a scoped
render-stage mechanism on one GPU/driver, not a general WebGPU result.
