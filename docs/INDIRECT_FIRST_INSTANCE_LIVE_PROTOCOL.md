# Live indirect `firstInstance` compute-plus-render crossover protocol

## Research question

Within the repository's normal fixed-slice GPU-culling path, does moving each
bucket's survivor-list base from a per-vertex `uint` attribute into the fifth
word of its indexed indirect command reduce GPU-pass total, defined as the sum
of timestamped compute-pass and render-pass durations, while preserving the
portable fallback and exact rendered output?

The preceding frozen render-only experiment found similar aggregate effects in
two same-device sessions, but one session failed preregistered nuisance-factor
bounds. It therefore did not meet its two-matrix confirmation rule. This live
experiment is a new decision test, not a continuation or replacement of that
failed matrix. Its primary endpoint includes both the normal two-dispatch cull
and the render pass.

The WebGPU command field and WGSL builtin are established features. This study
does not claim invention of `firstInstance`. It tests the narrower engineering
claim that a feature-gated Three.js fixed-ownership specialization can remove a
material amount of timestamped pass time while continuously submitting live
culling for a fixed workload. It does not test dynamic-camera or dynamic-object
behavior.

## Fixed workload

Candidate timing fixes:

- Three.js 0.185.1 and the committed dependency lock;
- 65,536 static objects and 32 medium indexed geometry buckets;
- the deterministic baseline layout at 99% visibility and a 20% lower-dose
  safety cell;
- the repository seed, camera, lighting, material values, 1280 by 720 target,
  device pixel ratio 1, no antialiasing, sample count 1, and reversed depth;
- one reset dispatch over 32 command records followed by one cull dispatch
  over 65,536 objects in one `renderer.compute([reset, cull])` call;
- 32 indexed indirect draws in one selected static bundle and one top-level
  render call per frame.

Camera parameters and matrix bit patterns, object transforms, visibility mask,
geometry, material, target, and viewport must remain unchanged during warmup
and measurement. The normal frustum update may idempotently rewrite camera and
plane state only when their resulting commitments remain identical.

## Paired live representation

The live crossover owns one shared set of read-only object inputs and writable
cull outputs:

- matrix, bounds, object-bucket, bucket-base, bucket-capacity, and cull-order
  storage;
- one survivor-list storage buffer;
- one overflow flag;
- common index, position, normal, and UV attributes.

Each lane has its own natural, zero-offset indirect-command buffer and its own
pre-recorded render bundle. Separate command buffers avoid the physical segment
offset implicated by the render-only nuisance result and match standalone
use, where the command allocation begins at record zero. A locked binary
`lanePhysicalOrder` factor controls lane-specific command-buffer, compute-node,
material, mesh, and bundle construction; first compute use; and render-pipeline
priming. This factor is counterbalanced across repetitions. Common geometry and
shared-resource construction occurs before either lane and is outside the
factor.

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

Only the portable geometry contains the per-vertex `bucketBase` attribute. No
instance-stepped geometry attribute is permitted. The feature lane is refused
unless `renderer.hasFeature('indirect-first-instance')` is true.

An untimed forced-feature-off gate must exercise the same deployment-selection
factory with feature availability overridden to false. It must select only the
portable builder, construct no feature lane, retain zero in every fifth command
word, and pass the same command, membership, address, and output validation.
This gate runs in a separate disposable browser renderer/device before the
fresh candidate session so it cannot allocate or prime the candidate's portable
resources first.

Both lane-specific reset/cull node pairs must be produced by the same factory
used by the standalone strategies. They bind the same shared inputs, survivor
output, overflow output, frustum planes, dispatch dimensions, and storage
types. Only the indirect-command resource identity differs. The compute graph
does not read `firstInstance`; reset changes only `instanceCount` and overflow,
and cull atomically changes only `instanceCount`, survivor slots, and overflow.

For every scheduled frame, including consecutive occurrences of the same
lane, the timed body is exactly:

```text
select one already-built lane
update the shared frustum uniforms
renderer.compute([selected reset, selected cull])
render the selected pre-recorded bundle
```

The unselected lane is not computed or rendered. Outside the selected lane's
normal reset/cull submission, warmup and measurement forbid a benchmark-
authored copy or write to storage, indirect, or vertex resources. They also
forbid a finalize or third compute dispatch, readback, diagnostic render,
dual-lane compute, timed-pipeline compilation, bundle recording, or persistent
resource allocation in the frame body. Normal renderer-managed frustum-uniform
uploads and transient command encoders are part of both lanes. Phase-boundary
timestamp resolution is permitted after warmup and after measurement, outside
all timed frame bodies.

## Compute, command, and membership gates

Candidate timing is refused unless untimed preflight proves:

- raw reset and cull WGSL are retained for both lanes; for each phase, the raw
  sources must differ because Three.js r185 assigns a distinct generated symbol
  to each lane-local command buffer, then become byte-identical after replacing
  only the variable and struct identifiers resolved from the `indirectCommands`
  storage-binding coordinate with fixed canonical identifiers;
- workgroup sizes, dispatch dimensions, storage-binding types and coordinates,
  and uniform layouts are identical; for the fixed workload, the default
  64-thread workgroup produces one reset workgroup and 1,024 cull workgroups;
- runtime storage-resource identities are shared for every binding except the
  two indirect-command buffers; automatically generated count uniforms may be
  lane-local but must have identical values and layouts;
- each command buffer contains exactly 32 five-word indexed records at byte
  offset zero;
- after its lane's live compute, every command's index count, instance count,
  first index, and base vertex exactly match the independent expectation;
- portable word four is zero and feature word four is the exact bucket base;
- raw WGSL and a structural field-access audit show no executable read or write
  of command word four; its declaration inside the common command struct is
  permitted;
- overflow is zero and every visible object appears exactly once in its
  correct bucket slice, with no out-of-range ID or active padding value;
- canonical per-bucket membership digests match the independent CPU frustum
  reference and match across lanes.

Atomic append order is not part of the algorithm's semantics and may differ
between valid dispatches. Raw survivor bytes are therefore validated in full
against their own counts and canonical membership, but cross-lane byte order is
not required to match. No sorting or copy is added to force an order. The
primary estimand is consequently a paired comparison of the two complete live
feature configurations, including naturally occurring variation in atomic compaction order;
it is not a pure render-stage causal estimate. The separately reported render
component is a secondary live-configuration decision and coherence endpoint,
interpreted alongside the earlier frozen-input experiment; it is not a pure
address-path estimate either.

At each validation point the harness serializes portable compute, its survivor
and command readbacks, its address oracle, and its production captures before
running the same sequence for feature. The second compute is never allowed to
overwrite the shared survivor buffer before the first snapshot is complete.
This validation runs before timing, at timing start, and after timing. Static
command fields, canonical membership, compute source, binding shape, workload,
and exact production-output commitments must remain stable at all three points;
raw survivor-order and address-image commitments may change with legal atomic
ordering.

## Address and rendered-output gates

Each validation point runs a generalized untimed RGBA8 all-address challenge
against each lane's freshly computed survivor snapshot and actual command
buffer. Every active fixed-slice address must encode that snapshot's exact
survivor ID plus one, and every unused slice address must remain transparent
zero. The oracle must use the production survivor resource, lane-specific
command resource, command offset zero, and production address expression.

Because valid atomic ordering may differ, the two address images are not
required to be byte-identical. Their canonical bucket memberships must be
identical. The production render is stricter: two stable captures per lane must
have exact full-frame RGBA8, depth32float, and RGB24 object-ID commitments,
nonzero coverage, and no non-visible or out-of-range ID. Color, depth, and
object-ID commitments must be identical across lanes and stable across all
validation points.

Raw render WGSL and runtime bindings must prove the intended contrast: equal
fragment shaders; a portable vertex input and bucket-base addition; a feature
vertex path that directly uses `instance_index` with no `bucketBase`
declaration or reference; common matrix and survivor storage resources; and no
unintended vertex-layout difference. Shader normalization may remove only the
portable `bucketBase` location and the resulting renumbering of otherwise
identical common inputs.

Both timed compute pipelines and both timed render pipelines are primed in
`lanePhysicalOrder` before timing. Parent, bundle, mesh, geometry, material,
common attributes, shared storage, lane command buffers, camera, target, and
viewport identities and versions are bound at timing start and end. Renderer
pipeline-cache cardinality, bundle-record counts, and inspectable binding
commitments are captured after timing-start diagnostics and compared
immediately after measurement, before postflight diagnostics. These are
inspectable lifecycle proxies, not proof about driver-internal compilation.
The paired candidate renderer constructs lane-specific resources in
`lanePhysicalOrder`, primes first compute and render use in that order, and only
then runs the fixed portable-then-feature serialized validation sequence.

## Crossover schedule

`P` denotes portable bucket-base addressing and `F` denotes feature-gated
indirect `firstInstance` addressing. Each eight-frame block uses one of two
complementary patterns:

```text
PFPF | FPFP
FPFP | PFPF
```

The orientations alternate, with a committed starting offset. Warmup contains
40 blocks, or 320 frames. Measurement contains 60 blocks, or 480 retained
rows: 240 per lane and 30 blocks of each orientation.

Every row records the block, orientation, within-block position, planned lane,
lane construction order, command-buffer identity commitment, lane-selection
serial, compute-call serial, render-call serial, and unique GPU frame ID. A
trial is rejected for a missing, duplicate, reordered, or extra selection,
compute, or render event.

Three.js r185 uses two timestamp queries for each compute submission and two
for each render pass. Warmup therefore consumes 640 queries in each pool and
measurement consumes 960 in each pool, below the pinned 2,048-query capacity.
Every retained frame must have exactly one compute timestamp UID and one render
timestamp UID. Timestamp quantum must be no greater than 1,000 ns.
Timestamp tracking is disabled during construction, priming, every validation,
address challenge, production capture, and readback. The compute and render
pool capacities, start/end indices, resolved UID sets, and phase resets are
committed so the query accounting can be reconstructed. The environment must
also expose at least eight storage buffers per shader stage, the cull kernel's
pinned requirement.

Because Three.js r185 creates timestamp query sets and their persistent
resolve/result buffers lazily, the harness pre-primes them before the trial. It
enables tracking for one untimed selected-lane compute and render, resolves both
pools, disables tracking, then requires both pool indices to be zero, UID
offsets empty, and pool identities/capacities stable. Those priming UIDs are
excluded from warmup and measurement accounting.

## Trial plan and primary estimator

One matrix contains 12 repetitions and both visibility levels, for exactly 24
paired trials and 11,520 retained rows. Visibility order, lane
`lanePhysicalOrder`, and starting schedule orientation are pairwise
balanced by the committed plan.

For every measured block, analysis computes:

```text
mean(feature GPU pass total) - mean(portable GPU pass total)
```

where each row's GPU pass total is its timestamped compute duration plus its
timestamped render duration. The trial estimate is the conventional median of
its 60 block deltas; an even median is the arithmetic mean of the two central
values. Percentage deltas use the corresponding portable block mean as the
denominator before taking the trial median.

The identical block and trial estimator is applied to render duration as a
secondary confirmatory endpoint and to compute duration as a descriptive
diagnostic. Percentage denominators are the corresponding portable component
means. Row-wise total must equal compute plus render exactly within 1e-9 ms.
Frames and blocks are repeated observations, not independent replicates. The 12
repetition estimates are the inferential units for each visibility cell. No
frame or block is removed as an outlier.

For every binary nuisance factor, each level summary is the median of the six
repetition-level estimates assigned to that level. Its interaction is level
one's median minus level zero's median; the gate applies to the absolute value
in milliseconds and percentage points. Measurement halves are formed within
each trial by applying the same block estimator separately to blocks 0 through
29 and 30 through 59, then taking the median across the 12 first-half or
second-half repetition estimates. Their interaction is second minus first.

The dose contrast pairs the 99% and 20% trial estimates with the same
repetition index and subtracts low-visibility feature-minus-portable from the
high-visibility feature-minus-portable value. Its summary is the median of the
12 paired differences. The millisecond value is the decision endpoint;
percentage-point differences are reported descriptively.

Condition-blind drift uses all lanes together. Within each trial, it is the
mean GPU pass total of the final 15 measured blocks minus the mean of the first
15 blocks; its percentage uses the first-quarter mean as denominator. Overall
and visibility-specific drift summaries are the medians of those trial-level
values. No lane label or feature-minus-portable delta enters the drift
calculation.

## Preregistered decision gates

Negative values favor indirect `firstInstance`. A first-device live result
passes only if every gate below passes independently in each of two candidate
matrices:

- the 99% median GPU-pass-total delta across 12 trials is at most -0.10 ms and at
  most -5%;
- at least 10 of 12 99% GPU-pass-total trial estimates are strictly negative;
- the 99% median render delta is at most -0.10 ms and at most -5%, with at
  least 10 of 12 render estimates strictly negative;
- the 99% GPU-pass-total median is negative in both `lanePhysicalOrder` strata, both
  starting-orientation strata, both visibility-order positions, and both
  measurement halves;
- each 99% GPU-pass-total nuisance interaction remains strictly inside both
  0.10 ms and 5 percentage points;
- the 20% median GPU-pass-total regression is below both +0.02 ms and +5%, with at
  least 10 of 12 estimates below each upper bound;
- the paired 99%-minus-20% median GPU-pass-total effect is at most -0.05 ms;
- condition-blind first-versus-last-quarter pooled GPU-pass-total drift has an
  absolute median strictly below both 0.10 ms and 5%, including within each
  visibility stratum;
- every schedule, timestamp, compute-identity, command, membership, address,
  shader, exact-output, lifecycle, environment, telemetry, source-provenance,
  and artifact-integrity gate passes.

Compute timing is reported but has no independent percentage gate because its
expected duration is close to timer resolution and any real compute difference
already contributes to the primary GPU-pass-total endpoint. Structural source,
binding, dispatch, and event equality establishes that the lanes perform the
same culling work.

GPU-pass total excludes renderer-managed uniform-upload time, command-encoding
and queue gaps, pass overlap, presentation, browser composition, and end-to-end
latency. CPU submission work is recorded separately where the harness exposes
it. The result must not abbreviate this endpoint as unrestricted "total GPU
work."

Ten same-direction results out of 12 have exact one-sided sign probability
0.0193 under a fair null. The numerical thresholds define a practically
material effect; they are not replaced with frame-level significance tests.

## Environment and telemetry boundary

Every candidate requires the same machine, adapter identity, backend, driver,
browser build, Three.js revision, feature availability, viewport, reversed-
depth state, and cross-origin-isolation state across all trials and both
matrices. GPU timestamp support, a quantum no greater than 1,000 ns, and the
declared storage-buffer limit are mandatory. A fresh session means a new
benchmark-runner invocation with a new browser process, renderer, adapter/device
request, and page lifecycle; it does not mean a different machine or driver.

The committed Nvidia telemetry collector samples at 250 ms and must finish with
status `available`, zero malformed records, and a reconstructable CSV/summary.
The compute-process identity set must be the same in the pre-run and post-run
snapshots, using the exact sorted tuple `(gpuUuid, pid, sanitized processName)`;
reported memory use is not part of identity. A pre/post mismatch is a
non-replaceable environment-gate failure for that otherwise completed matrix,
not a collector failure. Other resident GPU processes are allowed when the set
is stable and are disclosed. Temperature, utilization, clock, power-state,
memory, and power values are contextual: none has a post hoc rejection
threshold. Device drift is governed only by the condition-blind numerical drift
gate above. A collector failure is handled by the matrix-level transient rule
below; observed telemetry values or the presence of a stable process are not.

The page and runner must record zero page errors, console errors, uncaptured
WebGPU errors, and WebGPU validation errors from construction through teardown.
An ordinary shader, binding, command, or validation error is an implementation
failure requiring a source fix and two new candidate matrices. Only an actual
device/browser loss follows the matrix-level transient rule below.

## Stopping, replication, and claim scope

The protocol is committed before candidate timing. The implementation and all
correctness checks are then frozen in a separate clean source commit. Any
development or smoke timing is excluded from candidate evidence.

Before candidate timing, implementation validation established that the two
required lane-local command buffers receive distinct generated `NodeBuffer_N`
symbols in Three.js r185 WGSL. The compute-source gate was therefore amended to
the coordinate-resolved, single-identifier normalization specified above. This
pre-candidate amendment changes no workload, timing endpoint, schedule,
threshold, or decision rule; raw sources remain mandatory evidence.

Exactly two full candidate matrices are run in separate browser/device sessions
from that same frozen commit. Matrix two is run even if matrix one fails. A
trial-local timestamp failure may restart only its complete two-visibility
repetition in the same session, and only when an independently logged runner or
device interruption identifies the cause before delta inspection.

Browser/device loss, telemetry-collector failure, or run-level artifact
incompleteness/corruption invalidates the entire attempted matrix. Its artifacts
are preserved and the matrix restarts in a fresh session. Infrastructure-failed
attempts do not count toward the exactly two valid candidate matrices. Shader,
binding, command, membership, address, output, lifecycle, schedule, source,
ordinary WebGPU validation, or post-hoc-verification failure is an
implementation/evidence failure, not a retry condition; correcting one requires
a new frozen commit and restarts both matrices. Performance, drift, telemetry
values or process-set mismatches, nuisance interactions, atomic ordering, and
directional or near-threshold outcomes are never replaced.

If either matrix fails any decision gate, the first-device live confirmation is
not met and both estimates are reported unchanged. If both pass, the result
supports enabling the feature-gated lane for the tested Three.js, browser, GPU,
driver, and fixed workload while retaining the portable fallback. It does not
test dynamic scenes, establish a general WebGPU optimization, or retroactively
convert the frozen render-only non-pass into a pass. A materially different GPU
family or backend is required before making a broader claim.

Before candidate timing, a committed deterministic sanitizer must define an
exact JSON-path/CSV-column redaction allowlist and fixed replacement values for
unrelated machine-local identifiers; no unlisted field may be changed. It
derives a public evidence bundle from the private original, then computes a new
SHA-256 manifest over the derived bytes and records the sanitizer source hash.
The private original and its distinct manifest remain unchanged. Both bundles
are labeled explicitly, and the public manifest never purports to authenticate
bytes removed after hashing.
