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
  storage-binding coordinate with fixed canonical identifiers; no access,
  coordinate, payload type, field, literal, statement, comment, whitespace, or
  unrelated identifier may change;
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
identical. The direct diagnostic render is stricter: two stable captures per
lane must have exact full-frame RGBA8, depth32float, and RGB24 object-ID
commitments, nonzero coverage, and no non-visible or out-of-range ID. Color,
depth, and object-ID commitments must be identical across lanes and stable
across all validation points.

That fresh diagnostic mesh is the membership oracle, but it does not by itself
prove execution of the opaque timed bundle. While the same lane remains active,
the harness also renders the existing benchmark scene twice through its actual
pre-recorded `BundleGroup` into the exact timed target and reads back RGBA8.
Both production captures must be byte-identical and equal the diagnostic color
capture. Renderer call serials must advance by exactly one for each capture,
with no compute, selection, or snapshot-preparation event. Before, between, and
after commitments must retain the same bundle GPU object, sole render object,
mesh, geometry, material, bindings, target, pipeline/cache state, and exactly
one bundle-record callback. Timestamp tracking remains disabled throughout.

Raw render WGSL and runtime bindings must prove the intended contrast: equal
fragment shaders; a portable vertex input and bucket-base addition; a feature
vertex path that directly uses `instance_index` with no `bucketBase`
declaration or reference; common matrix and survivor storage resources; and no
unintended vertex-layout difference. Shader normalization may remove only the
portable `bucketBase` location and the resulting renumbering of otherwise
identical common inputs.

Each of the three evidence points retains two independent runtime inspections:
the validation attached to the parity snapshot and the final validation for
that point. The runner issues the exact six-entry phase/role sequence, a
consecutive capture ordinal, and a unique phase-bound challenge before each
retained inspection. Those records are committed outside the page-produced
payload and included in the observation digest. Verification requires exact
challenge equality, the fixed sequence, consecutive capture counters, distinct
observation digests, stable semantic evidence, and stable inspected resource
identities across all six captures. Re-labeling or re-hashing one cached
observation therefore cannot satisfy the retained evidence contract.

The observed reset/cull compute-node identities must equal the corresponding
lifecycle node identities. Every observed shared compute binding must equal its
named lifecycle storage attribute, and each lane-local `indirectCommands`
binding must equal that lane's lifecycle command-buffer identity. These joins
are checked within every observation and across all six captures.

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
indirect `firstInstance` addressing. Each trial selects one of two
complementary cyclic eight-frame patterns and repeats that pattern for every
warmup and measured block:

```text
PPPFPFFF
FFFPFPPP
```

The selected orientation is fixed within a trial and counterbalanced across
repetitions. Each cyclic block, including the actual predecessor from the
previous block, contains each of `PP`, `PF`, `FP`, and `FF` exactly twice. It
also contains each of the eight possible `(t-2, t-1, t)` lane triples exactly
once. Warmup contains 40 complete cycles, or 320 frames, so measurement starts
with the exact cyclic two-frame history. Measurement contains 60 complete
cycles, or 480 retained rows: 240 per lane, 120 of each first-order transition,
and 60 of each two-frame-history/current triple.

Every row records the block, orientation, within-block position, planned lane,
the actual previous two timed lanes, lane construction order, command-buffer
identity commitment, lane-selection serial, compute-call serial,
render-call serial, and unique GPU frame ID. The actual history is carried
across the warmup/measurement boundary and must equal the cyclic schedule. A
trial is rejected for a missing, duplicate, reordered, or extra selection,
compute, render, transition, or history event.

Three.js r185 uses two timestamp queries for each compute submission and two
for each render pass. Warmup therefore consumes 640 queries in each pool and
measurement consumes 960 in each pool, below the pinned 2,048-query capacity.
The r185 timestamp helper otherwise treats an array-valued compute group as a
render context with an undefined ID. Before first use, the harness therefore
registers each exact lane compute-array identity with a distinct positive
context ID through a version-pinned backend wrapper. The registration binds the
backend, wrapper, array, lane, reset/cull node IDs, and registration serial.
Every submission returns that actual registration record rather than inferring
the submitted lane from the schedule.

Every warmup and retained frame must have exactly one raw compute UID-duration
record and one raw render UID-duration record. Strict parsing requires the r185
forms `c:<frame-call>:<registered-context>:f<frame>` and
`r:<frame-call>:<render-context>:f<frame>` in their corresponding pools. The
frame, call index, compute context, submitted group, lane, and row must join
exactly; both durations must be strictly positive. Compute and render timestamp
quality is established independently in both warmup and measurement, and each
pool's observed quantum must be no greater than 1,000 ns.

The 320 warmup submissions are retained as compact event evidence rather than
timing rows. The audit starts with physically accurate `null`/`null` lane
history, reconstructs every scheduled lane, command buffer, submitted compute
group, node set, serial, GPU frame, and raw UID record, and commits the final
two executed lanes and serials at the post-warmup boundary. Measurement frame
zero must join that executed tail; no conceptual cyclic predecessor is accepted
for the first two warmup frames.
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
`lanePhysicalOrder`, and cyclic schedule orientation are pairwise
balanced by the committed plan.

For every measured block and each previous-lane stratum `s` in `{P, F}`, analysis
computes:

```text
delta_s = mean(feature GPU pass total | previous lane = s)
        - mean(portable GPU pass total | previous lane = s)

block delta = (delta_P + delta_F) / 2
```

Each transition cell contains exactly two rows per block. The standardized
portable and feature block means give each predecessor stratum weight one half;
their difference equals the block delta. Each row's GPU pass total is its
timestamped compute duration plus its timestamped render duration. The trial
estimate is the conventional median of its 60 stratified block deltas; an even
median is the arithmetic mean of the two central values. Percentage deltas use
the standardized portable block mean as the denominator before taking the
trial median. The exact two-frame-history balance is also reconstructed as a
schedule gate, while the preregistered primary estimator stratifies on the immediately
previous lane.

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
  cyclic-orientation strata, both visibility-order positions, and both
  measurement halves;
- the predecessor-stratified 99% GPU-pass-total median is negative after both
  portable and feature predecessors, and the feature-predecessor minus
  portable-predecessor interaction remains strictly inside both 0.10 ms and
  5 percentage points;
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
Candidate evidence requires exactly one telemetry GPU. The page-reported
adapter vendor must normalize exactly to `nvidia`, and its normalized adapter
description must equal the sole telemetry GPU name under the frozen ASCII
trim/collapse/lowercase policy. The audit records separate adapter, telemetry,
and association commitments; the post-hoc verifier reconstructs them. This
fails closed when a hybrid or multi-GPU topology cannot unambiguously associate
the WebGPU workload with the sampled device.
Collector liveness is a structural infrastructure gate, not a telemetry-value
gate. The collector records its active start and stop-request monotonic times.
For every observed GPU, the first-sample delay, maximum internal sample gap,
and final-sample staleness must each be no greater than eight requested
intervals (2,000 ms). The same GPU identity set must cover that active window.
The post-hoc verifier reconstructs those facts from `gpu-telemetry.csv`; it
never rejects temperature, utilization, clocks, power state, memory, or power.

Candidate evidence also binds the WebGPU page adapter to the telemetry device
instead of treating those records as independent environment descriptions.
`gpu-telemetry-summary.json` must contain exactly one GPU, the page adapter
vendor must normalize exactly to `nvidia`, and the normalized page adapter
description must equal that GPU's normalized `gpuName`. Identity text is valid
only when it contains ASCII horizontal tabs or printable ASCII characters;
normalization trims ASCII spaces/tabs, collapses each remaining run to one
space, and folds ASCII `A`-`Z` to lowercase. It performs no Unicode,
punctuation, vendor-prefix, substring, or fuzzy normalization. The environment
audit records the normalized identities and deterministic SHA-256 commitments,
and the verifier reconstructs the association from the page and CSV-bound
telemetry evidence. Zero GPUs, multiple GPUs, a non-NVIDIA vendor, or an exact
name mismatch is a non-replaceable environment-gate failure. The association
contains no GPU UUID or process identity, so public derivation preserves it
unchanged and revalidates it after UUID and process pseudonymization.

Each compute-process query records its stdout byte count, truncation status,
non-empty row count, parsed-record count, malformed-row count, and stderr byte
count. A genuinely blank, successful query is the only valid empty process
set. Truncation, stderr, or any malformed identity row is a retryable collector
failure; it cannot collapse into an apparently stable empty set.

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

Before candidate reservation and at runner start and teardown, the harness
computes the same deterministic installed-dependency closure over every regular
file under `node_modules`,
excluding only generated `.bin/**` shims and `.vite*/**` caches. The closure
binds sorted relative path, byte count, and file SHA-256; symbolic links or
unsupported entries outside those exclusions fail closed. Its digest, file
count, and total bytes are fixed in the series source identity, recorded at
runner start and end, and must remain equal across both matrices. This binds
the installed Three.js, both statically imported Three Blocks packages, Vite,
Playwright, and installed transitive dependencies in addition to the tracked
source and package lock.

Generated Vite optimizer caches are excluded from that installed closure only
because candidate serving makes them non-executable. The candidate server
uses Vite's custom-app mode, serves the exact tracked `index.html` itself, and
rejects all Vite-client and virtual-module requests under `/@vite/` or `/@id/`.
It disables optimizer discovery, has no explicit optimizer entries, and
receives a unique, initially absent cache directory outside the project for
each matrix. A fail-closed middleware and transform audit also reject any
decoded request, module identifier, source path, or transformed reference
under an excluded `.vite*` directory. The cache must remain empty.

The audit commits the entry HTML source and exact 2xx response, then records the
sorted project-relative source path, source byte count and SHA-256, and every
exact successful browser-response SHA-256 for each requested project or
dependency JavaScript/CSS module. Query variants are canonicalized to one
source while retaining distinct response variants and reconciled counts;
redirects, errors, incomplete responses, or unmapped executable requests fail
closed. The benchmark entry point and live strategy module are mandatory. The
verifier independently rehashes every retained source and reconstructs the
runtime-audit aggregate and request/response counts after teardown. Captured
transformed-response commitments must be byte-identical across the two
matrices. Thus neither a reusable untracked prebundle nor a randomized Vite
client can substitute for the tracked source or installed dependency bytes
bound above.

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
symbols and derived struct symbols in Three.js r185 WGSL. The compute-source
gate in protocol commit `129972e` was therefore clarified, before any candidate
matrix, as the coordinate-resolved, single-binding/two-identifier alpha-renaming
specified above. This pre-candidate amendment changes no workload, timing
endpoint, schedule, threshold, or decision rule; raw sources remain mandatory
evidence.

Exactly two full candidate matrices are run in separate browser/device sessions
from that same frozen commit. Matrix two is run even if matrix one fails. A
trial-local timestamp failure may restart only its complete two-visibility
repetition in the same session, and only when an independently logged runner or
device interruption identifies the cause before delta inspection.

Before any attempt, a separate initialize-only command claims the study in the
one root candidate registry. Its study key is the canonical SHA-256 of the
experiment ID, Git tree, tracked-file digest, and package-lock digest; omitting
the commit prevents an empty commit over the same tree from resetting the
series. The claim binds the exact commit and full installed-dependency identity,
so a closure change under that study key rejects. The derived basename is
`first-device-live-<first-16-study-key-hex>`. A hash-linked materialization event
then binds the one series ID and its opening-event digest. Root locking and exact
inventory reject aliases, sibling copies, missing registries, duplicate source
claims, deleted materialized series, and concurrent initialization.
Verification also enforces claim time ≤ series-opening time ≤
materialization time across the registry and series chains.

Initialization emits a deterministic annotated-tag name, target, and canonical
message containing the claim, series-opening, and materialization digests. That
exact tag is created and published before timing. Every attempt requires the
matching annotated tag locally; the pair verifier revalidates it and reports
its message digest. Remote availability is verified operationally before the
first run. Deleting and rewriting both ignored local registry and series bytes
remains an inherent local limitation. The disclosed chains establish internally
consistent reported chronology; external pre-timing chronology depends on a
remote observer having fetched or retained the tag-object ID or ref before
timing. These commitments establish integrity, not authorship.

Every candidate attempt is launched by the committed series orchestrator. It
opens a hash-chained JSONL ledger for one clean source/dependency identity and
syncs an `attempt-reserved` event before starting the child runner. The runner
records the exact series ID, reservation-event SHA-256, attempt and matrix
ordinals, source commit and tree, and installed-dependency digest as structured
metadata. Finalization binds that reservation, child disposition, independently
derived closed classification, exact run-directory identity, and a recursive
content commitment for the preserved attempt. The pair verifier rejects a
broken chain, a missing or surplus attempt/run directory, a transplanted run,
post-finalization byte changes, overlapping sessions, a third valid matrix, or
any cross-matrix source, browser, backend, page-state, adapter/driver, physical
GPU, protocol, or workload mismatch. Ambiguous interrupted-child recovery
fails closed rather than starting an overlapping session.

A completed matrix advances its slot whether its numerical decision passes or
fails and whether its stable-process environment gate passes or fails. A
retryable infrastructure attempt retains its matrix ordinal and stops the
orchestrator for an explicit fresh-session invocation. A non-retryable
implementation/evidence failure terminates that source series. The registry and
ledger hash chains provide internally reproducible chronology and byte
commitments, while the matching current local tag binds the reported selected
opening boundary. External preexistence of that boundary depends on the retained
remote observation described above; none is an author signature or establishes
external authorship.
The exact event schemas, classifications, commands, and pair identity are
specified in
[`FIRST_INSTANCE_LIVE_CANDIDATE_LEDGER.md`](FIRST_INSTANCE_LIVE_CANDIDATE_LEDGER.md).

Browser/device loss, telemetry-collector failure, or an artifact-persistence
failure with exact infrastructure provenance invalidates the entire attempted
matrix. Artifact provenance is closed to a pre-metadata recorded spawn failure,
operating-system termination, or Windows abnormal status, plus the frozen
runner's allowlisted filesystem-I/O marker. When failed metadata exists, the
marker's I/O code must also occur in that metadata's error record. Artifact
absence or corruption by itself is not retry evidence. In particular, a
completed matrix cannot be replaced by deleting or corrupting its manifest,
and readable failed shader or validation metadata cannot borrow a later I/O
marker even if its manifest is incomplete.
Its artifacts are preserved and the matrix restarts in a fresh session.
Infrastructure-failed attempts do not count toward the exactly two valid
candidate matrices. More than one runner output inside a single reservation is
surplus evidence, not retryable incompleteness. A valid candidate additionally
requires a zero child exit with no signal. Shader,
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

The frozen allowlist, replacement templates, rejection rules, invocation, and
manifest-provenance boundary are specified in
[`PUBLIC_EVIDENCE_SANITIZER.md`](PUBLIC_EVIDENCE_SANITIZER.md) and implemented by
the committed `scripts/sanitize-live-evidence.mjs` source.
