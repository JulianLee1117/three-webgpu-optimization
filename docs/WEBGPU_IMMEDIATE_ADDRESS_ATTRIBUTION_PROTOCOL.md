# WebGPU immediate-data address attribution protocol

## Status and claim boundary

This document freezes a correctness-first investigation of three ways to supply
the bucket-slice base used by the existing Three.js/WebGPU indirect renderer.
It is written before an immediate-data integration or any timing run is used to
choose an implementation.

The work has two phases:

0. an analysis-ineligible technical canary; and
1. a compact, descriptive mechanism diagnostic with two complementary
   topologies.

Neither phase is a deployment candidate. Neither can produce `confirmed`,
`faster`, `regressed`, or an efficacy pass. Candidate thresholds from the
earlier indirect-`firstInstance` studies may be drawn as historical reference
lines only; they are not gates here. A later candidate requires a new protocol
and fresh data.

Protocol clarification (2026-09-03, before any full Phase 0 or timing run):
the existing fixed-slice implementation initializes indirect word four on the
CPU and the live reset/cull kernels update only `instanceCount`, survivor IDs,
and overflow. The full canary therefore requires alpha-normalized compute WGSL
to be exact across A/I/F and proves separately that compute never accesses or
writes `firstInstance`. An earlier draft incorrectly allowed a lane-specific
compute write to word four; no retained execution used that wording.

The separation is important because the completed standalone study exposed a
hidden execution-state problem. At 99% visibility, 40 of 48 feature sessions
clustered near 1.959--1.961 ms and eight near 1.504--1.505 ms, while 39 of 48
portable sessions clustered near 1.816--1.818 ms and nine near 1.969--1.970 ms.
The same state persisted into each browser's 20% trial despite nearly identical
sampled graphics clocks. That record contains no retained variable that
predicts which mode a fresh process entered. It therefore supports the factual
readout `persistent discrete process modes; predictor unknown`, not an
attribution to clocks, shaders, `firstInstance`, or another unobserved cause.

There is also a historical confound. Earlier positive live measurements used
Chrome 151 and a multi-lane, same-browser topology; the negative standalone
measurement used Chrome 152 and one lane per fresh browser. Browser version and
topology changed together. No comparison in this protocol may use those old
runs to estimate either effect.

## Fixed address lanes and contrasts

`instanceIndex` below is WGSL `@builtin(instance_index)`. `bucketBase` is the
canonical start of one bucket's slice in the immutable `visibleIds` storage
array.

| Lane | Vertex address expression | Indirect word four | Bucket-base vertex stream | Immediate data |
| --- | --- | --- | --- | --- |
| `A` -- attribute | `visibleIds[bucketBaseAttribute + instanceIndex]` | zero | one `uint32` value per vertex | none |
| `I` -- immediate | `visibleIds[threeImmediateDrawBase + instanceIndex]` | zero | absent | one `uint32` base set before every indirect draw |
| `F` -- first instance | `visibleIds[instanceIndex]` | exact `bucketBase` | absent | none |

The immediate shader declares exactly one statically used value:

```wgsl
requires immediate_address_space;
var<immediate> threeImmediateDrawBase : u32;
```

For each I bundle recording, the integration uses exactly one 32-element
`Uint32Array` named `indirectImmediateBases`, parallel one-to-one with the
indirect offsets; it does not create a view per draw. For draw index `i`, it
records
`setImmediates(0, indirectImmediateBases, i, 1)` immediately before the
corresponding `drawIndexedIndirect()`. Here `i` and the final `1` are the
element offset and element count for a typed-array source. The I pipeline's
explicit pipeline layout has `immediateSize: 4`; A and F have an immediate size
of zero. No adapter feature is requested for immediates. Support is established
through the WGSL language feature and device limit described below.

The three descriptive contrasts are always named and signed as follows:

```text
I - A  immediate base minus vertex-attribute base
F - I  indirect firstInstance minus immediate base
F - A  indirect firstInstance minus vertex-attribute base
```

Negative duration favors the first named lane. `I - A` includes removal of the
vertex stream and shader input plus the immediate command/access path. `F - I`
includes the built-in instance-address path versus an immediate load and add.
`F - A` is the full historical representation contrast. None is described as
a single-instruction benchmark.

## Common workload and renderer boundary

Both phases retain the earlier fixed workload unless the plan fails closed
before any execution:

- 65,536 object transforms and bounds that are immutable within each
  visibility-level challenge. The prescribed `v99` to `v20` transition is the
  only scenario-content replacement and occurs before the second challenge;
- 32 medium indexed geometry buckets in the baseline layout;
- deterministic 99% and 20% visible subsets from scenario seed `2980913190`;
- the exact existing geometry, scenario, survivor-membership, and command
  commitments;
- a 1280 by 720 offscreen target, device-pixel ratio one, reversed depth, and
  the same camera, material output, clear values, and draw order;
- one render object and 32 `drawIndexedIndirect()` calls per lane; and
- one fixed-size bucket slice per draw, with identical words zero through three
  in all three lanes.

The technical canary validates live compute output. Atomic compaction does not
have a portable invocation-order guarantee, so each A/I/F dispatch gates exact
membership, per-bucket counts, commands, padding, and overflow independently;
it does not require three separately dispatched survivor arrays to have the
same byte order. After all three pass, the harness uploads and reads back one
deterministic CPU-canonical survivor packing used by every address oracle and
render challenge. Before diagnostic timing, that common validated packing and
the three command arrays are frozen. Timed
frames perform no culling dispatch, readback, buffer write, resource rebuild,
pipeline compilation, bundle recording, or opposite-purpose diagnostic render.
The I bundle's already-recorded sequence of per-draw immediate values is fixed
across warmup and measurement.
The primary descriptive endpoint is timestamped GPU render duration. Compute
and one-time bundle-recording costs are outside that endpoint and may be
reported only as untimed engineering observations.

The renderer seam is the WebGPU backend's loop over indirect byte offsets. The
I lane sets its base on the actual `GPURenderBundleEncoder` immediately before
each indirect call in that loop. A mesh-level `onBeforeRender` callback is not
an acceptable implementation: it runs once for the retained mesh and cannot
supply 32 bases without restoring a 32-object CPU topology.

## Source, dependency, and browser freeze

No experiment may edit, patch, replace, or generate a file beneath
`node_modules`. The hash of the installed dependency closure is captured before
and after each phase and must remain exact.

Immediate support is developed as repository-owned source or as a clean,
separately pinned Three.js source worktree imported explicitly by the harness.
The served-module audit must prove that the intended source was loaded. An
in-memory monkey patch with no auditable source file, DevTools live edit, import
map that resolves accidentally to the installed package, or post-launch source
mutation invalidates the run.

Phase 0 is run against both of these source families before either is eligible
for Phase 1:

- the exact installed r185 family used by the prior result, with the package,
  lockfile, tarball integrity, and repository-owned integration patch pinned;
- one exact upstream Three.js development commit, with its commit, tree, and
  integration diff pinned. Commit
  `994260a7a59abe466de8e59db1ecc9e30352751d` is the observed pre-integration
  reference, not a moving `dev` alias.

The Phase 1 plan names one of those exact source targets. Results from different
Three.js commits are separate runs and are never pooled.

Every plan and final manifest records:

- Git commit, tree, ref, dirty state, sorted tracked-file digest, lockfile
  digest, and integration-diff digest;
- installed dependency closure file count, byte count, and digest before and
  after execution;
- Vite configuration and the sorted source/transformed digest of every served
  runtime module;
- browser executable canonical path, product/file version, byte length, and
  SHA-256, plus a sorted browser-distribution closure digest;
- Playwright package/version, launch API, complete launch arguments, user agent,
  and fresh-profile policy;
- adapter information, backend, driver, device features and limits, WGSL
  language features, timestamp quantum when timing is enabled, and viewport;
- operating-system build and GPU telemetry identity.

An exact browser binary is one experimental stratum. Chrome 151 and Chrome 152
may be compared only by repeating the complete frozen phase with the same
source, hardware, workload, and topology. Their results remain side by side;
they are not pooled and cannot repair the historical browser/topology confound.

The full Phase-0 execution stratum was frozen on 2026-09-03, before either
source target was launched, to the canonical Chrome executable version
`152.0.7977.82`, byte length `4,461,720`, and SHA-256
`8cd23aec3a30479b9b8db2063e70526c88a7ec2c99cd744603eb26ab598733ed`.
The earlier minimal canaries on `152.0.7977.66` remain historical evidence and
do not substitute for the fresh raw capability canary embedded in each full
Phase-0 target run.

## Phase 0: correctness-only technical canary

### Eligibility boundary

Phase 0 uses disposable browsers/devices and has:

```text
executionMode: technical-canary
analysisEligible: false
efficacyAnalysisAllowed: false
numericalDecision: null
```

GPU timestamps and benchmark timers are disabled. The canary may retain only
operation completion, validation, and correctness evidence. It must not retain
per-lane durations, rank lanes, calculate deltas, or print an efficacy-oriented
summary. A capability absence is `unsupported`; a malformed integration is
`technical-canary-failed`. Neither is a performance result.

### Raw capability canary

A raw-WebGPU canary runs before the Three.js integration canary, in a separate
disposable browser/device. It must establish all of the following:

1. `navigator.gpu` and a non-fallback target adapter are available.
2. `navigator.gpu.wgslLanguageFeatures` contains
   `immediate_address_space`.
3. `device.limits.maxImmediateSize >= 4`.
4. Actual render-pass and render-bundle encoders expose a callable
   `setImmediates`; prototype-name inspection alone is insufficient.
5. A shader containing the exact `requires` directive and one immediate `u32`
   compiles without messages of error severity.
6. An explicit pipeline layout with `immediateSize: 4` creates a valid pipeline.
7. Direct-pass and render-bundle draws each consume at least three distinct,
   non-monotonic immediate values and produce the exact integer address oracle.
8. Scoped negative controls for an unset required slot, a misaligned range, and
   a range beyond `maxImmediateSize` are rejected rather than silently drawn.
   The exact exception/error representation is recorded because Chrome
   151--152 changed `setImmediates()` failure reporting; a scoped expected error
   must not become an uncaptured device error.
9. The F lane's independent requirement, `indirect-first-instance`, is present
   and requestable on the same adapter. Absence blocks the three-lane phase but
   does not make the A/I capability result false.

The raw canary uses no Three.js code and is destroyed before the integration
canary. This prevents expected validation errors or raw pipeline state from
entering later evidence.

### Integrated A/I/F canary

The integration canary creates A, I, and F under one disposable Three.js
renderer. It runs serialized correctness challenges in all six lane orders:

```text
A I F    A F I    F I A    I A F    I F A    F A I
```

Each listed order is a sequence of completed, cleared render-pass captures; it
tests contamination through persistent renderer, cache, and bundle state, not
same-render-pass encoder-state inheritance. The raw negative control below owns
the same-pass leakage test. Order changes correctness exposure only; no duration
is recorded. Each lane must pass all gates below at both visibility levels and
with both frozen non-monotonic sentinel base schedules, S1 and S2, whose
expected address outputs differ from the production schedule and each other.

#### Capability and canary gates

- The raw canary passed on the exact browser binary used by the integration
  canary.
- Page-level WGSL features, device limits, adapter identity, and requested
  device features match the raw canary exactly.
- No fallback, compatibility-mode substitution, WebGL renderer, or software
  adapter is accepted.
- The immediate language feature is not confused with a requested
  `GPUFeatureName`; the device request remains valid without inventing an
  `immediate` adapter feature.
- All scoped error controls are complete before the production device is
  created, and the production device has zero uncaptured errors or losses.

#### Shader gates

- I contains the exact `requires immediate_address_space` directive, the exact
  declaration `var<immediate> threeImmediateDrawBase : u32;`, and the
  expression `threeImmediateDrawBase + instanceIndex` feeding the survivor
  lookup. This is its one statically used immediate variable.
- A contains the bucket-base vertex input and its addition; it has no immediate
  declaration or use.
- F contains neither base source nor base addition and indexes `visibleIds`
  directly with `instanceIndex`.
- I and F expose no `bucketBase` vertex input. A alone consumes the declared
  integer attribute location.
- Vertex sources normalize to one digest after exactly these approved
  transforms: replace the three frozen address expressions and their required
  declarations; remove A's sole `bucketBase` vertex entry while compacting the
  following generated normal-input location; and canonicalize only the
  Three.js-generated storage-structure and storage-variable identifiers for
  the same matrix and visible-ID bindings. Fragment sources are exact. Compute
  sources normalize to one digest after replacing only Three.js-generated
  lane-local command-buffer identifiers. Each resolved command-buffer variable
  occurs exactly twice: once in its unique storage-binding declaration and once
  in executable code. That sole executable occurrence is the complete direct
  chain `value[allowedIndex].instanceCount`, used by the reset kernel only as
  the address of `atomicStore(..., 0u)` and by the cull kernel only as the
  address of `atomicAdd(..., 1u)`. Pointer aliases, helper forwarding,
  whole-structure access, other members, and any additional occurrence fail.
  Thus none may access or write `firstInstance`, whose lane-specific values are
  initialized before the live cull. Every transform count and before/after
  source is retained. Compute modules contain neither an
  `immediate_address_space` requirement nor a `var<immediate>` declaration;
  compute pipeline descriptors have no own `immediateSize`, every compute
  pipeline's uniquely matched pipeline-layout descriptor owns
  `immediateSize: 0`, and compute pass traces contain no `setImmediates()`
  call.
- Raw WGSL, normalized WGSL, entry points, compilation messages, shader-module
  identities, and challenged observation nonces are retained. A cache label or
  source string supplied by the harness is not accepted as proof of the shader
  passed to `createShaderModule()`.
- The address and object-ID diagnostics repeat the same proof independently.
  For each diagnostic purpose the offline runner starts from the captured shader
  modules, locates the exact `visibleIds` storage binding, and accepts only the
  lane expressions in the fixed A/I/F table. It then applies a separately
  enumerated purpose-local alpha normalization and requires one normalized
  vertex digest and one normalized fragment digest across A/I/F. Production
  shader equality cannot substitute for either diagnostic proof.

The runner generates one unpredictable 128-bit observation nonce, encoded as
exactly 32 lowercase hexadecimal characters, after the source freeze and
injects it with the target configuration. Every
actual `createShaderModule()` observation retains that nonce, its creation
ordinal, raw WGSL bytes, and module identity. For each challenged render or
compute use, the page commits to the nonce, ordinal, raw WGSL, module identity,
layout identity, pipeline identity, and bundle or dispatch identity in an
unambiguous length-delimited SHA-256 record. The offline runner requires the
injected nonce, independently rebuilds every commitment from the captured
descriptor and execution chain, and rejects a nonce echo without that chain.
The nonce associates observations within this run; it is not an authenticity
claim, external timestamp, workload input, or permitted shader difference.

#### Pipeline-layout and pipeline gates

- The descriptor actually supplied to `createPipelineLayout()` is captured.
  I has `immediateSize: 4`; A and F have zero. The I render pipeline retains
  that exact layout identity through every draw and bundle recording.
- All bind-group layouts, target/depth formats, sample counts, primitive state,
  vertex-buffer layouts other than A's one base attribute, and material state
  are exact.
- For every challenged render and compute pipeline, the offline runner parses
  every raw WGSL `@group`/`@binding` declaration and reconstructs the exact
  contiguous pipeline-layout group set, bind-group-layout entry partition,
  created bind groups, and `setBindGroup()` indices. Extra empty groups, extra
  entries, missing entries, duplicate coordinates, dynamic offsets, and
  descriptor-class mismatches fail. WebGPU descriptor defaults are normalized
  before comparison: in particular an omitted buffer binding `type` is
  effective `uniform`. The pinned Three.js `NodeUniformsGroup` construction
  exposes uniform entries to the exact all-stage mask `7`; other resource
  visibility equals the union of the stages that declare the coordinate. Both
  the page and runner retain the raw descriptor and this effective semantic
  interpretation.
- Every challenged production and diagnostic draw reconstructs the binding
  chain from the recorded `setBindGroup()` call through the exact created bind
  group and its buffer entries. The matrix and `visibleIds` entries must resolve
  to the committed scenario buffers with exact offsets and extents. The actual
  vertex-buffer layouts, slots, native buffer identities, and merged index
  buffer are likewise reconstructed; a shader declaration without the matching
  bound resource is insufficient.
- Pipeline descriptors and their transitive shader/layout commitments are
  hashed. The production path resolves to exactly one cached production
  pipeline per lane. The separately identified address and object-ID
  diagnostic paths each have one inventoried pipeline per lane. All are primed
  before the ordered snapshot challenges, and none compiles late during a
  correctness snapshot or diagnostic timing interval.
- Pipeline creation and compilation scopes contain no validation or internal
  errors. A pipeline whose layout silently infers zero immediate bytes fails.

#### Command and indirect-word gates

- Each production lane records exactly one render object and 32
  indexed-indirect draws at offsets `0, 20, ... 620` bytes. A and F retain one
  globally stable cached production bundle. I retains exactly one cached
  production bundle at a time: it is stable within a schedule and is replaced
  only by the three prescribed schedule re-records. Address and object-ID
  diagnostic render objects and bundles are separate, explicitly identified,
  precreated, primed, and inventoried.
- I records exactly 32
  `setImmediates(0, indirectImmediateBases, i, 1)` calls, one for each `i` from
  zero through 31, immediately adjacent to and before the matching indirect
  draw. Each copies element `i` into immediate byte range zero through three;
  the source values equal the active schedule's exact 32-element
  `sourceBaseByDraw` permutation of the canonical bucket bases. A and F record
  none.
- Indirect words zero through three are bit-exact across A, I, and F. Word four
  is zero in every A and I command and equals the active schedule's exact
  `sourceBaseByDraw` value in F. There is exactly one zero value; its draw is
  zero in the canonical schedule and is determined by the frozen permutation
  in S1 and S2. The other 31 values are nonzero.
- The command-buffer byte length, usage flags, attribute identities, versions,
  offsets, and complete preflight/postflight readbacks are retained. Each
  schedule installation is an audited, untimed transition followed by its own
  command freeze and preflight readback. There is zero overflow, padding
  mutation, or command write inside that schedule's six-order challenge.
- The complete production-device command stream is closed globally, not only
  inside challenge intervals. The frozen Phase-0 plan contains exactly 798
  render passes, six compute passes, and 825 readback transfers, hence 1,629
  command encoders, finished command buffers, and singleton submissions. Every
  encoder contains exactly one of those three operation classes. Every transfer
  is an exact `copyBufferToBuffer()` or `copyTextureToBuffer()` into one unique
  825-member staging-buffer inventory, followed in order by submit, map, mapped
  range, and one destroy. Unknown, unused, multiply submitted, piggybacked, or
  pre-encoded command buffers fail the canary.
- The 825 transfers are also reconstructed as a phase-and-channel-indexed
  expected-source table and matched bijectively to the raw transfer ledger.
  Every row fixes the claimed producing buffer or texture identity, source
  offset or texture origin/mip/aspect/extent/format, destination staging-buffer
  identity and offset, and byte extent. This includes production color/depth,
  address and object-ID diagnostics, visible-ID and command snapshots, the
  `v99` prime reads, and the live/shared/geometry/schedule reads. Matching only
  a phase label, output bytes, or staging size is insufficient evidence that a
  witness came from the declared producer.
- Every production-device `queue.writeBuffer()` call is assigned exactly once
  to a predeclared semantic window: scenario matrix realization, canonical
  visible-ID upload, schedule-install attribute realization, diagnostic
  position realization, or a render/compute uniform update whose buffer is
  proven by the challenged binding chain. The disjoint union of these
  semantic-window record assignments must equal the raw global call ledger: no
  call may be claimed by two windows and no call may remain unexplained. Each
  call independently fixes destination identity, source type, converted source
  and destination byte ranges, 4-byte alignment and bounds, selected-source
  SHA-256, phase, and ordering. Multiple valid uniform calls may target
  overlapping destination byte ranges; those overlaps are retained as
  observations but are not an acceptance predicate. Every render or compute
  uniform write must have exactly one proof witness whose challenged consuming
  event is later in the same capture phase--either a direct draw or dispatch,
  or the render-pass `executeBundles()` that submits a previously recorded
  bundle. Other uses of the same binding may exist. The evidence must replay
  bind-group state at the witnessed draw or dispatch and reconstruct the
  applicable `setBindGroup()`, draw or dispatch, bundle finish when applicable,
  containing command-encoder finish, and singleton submission chain. For a
  cached bundle, its recorded bind, draw, and bundle finish may predate the
  uniform write; the challenged `executeBundles()`, containing encoder finish,
  and submission must follow it.
  `dataOffset` and `size` are converted from typed-array elements (but from
  bytes for `ArrayBuffer` and `DataView`) before byte-range validation. Global
  `queue.writeTexture()`, `copyExternalImageToTexture()`, external-texture
  import, and command-copy destinations outside the 825 staging buffers are
  zero.
- Buffer mapping is closed with the same global, disjoint accounting. A
  readback staging buffer has exactly its submitted `MAP_READ` chain. A
  persistent buffer created mapped has exactly one initial mapped-range access
  and unmap in its creation phase before first GPU use; a persistent buffer not
  created mapped has no map event. Persistent buffers expose neither
  `MAP_READ` nor `MAP_WRITE` usage, and the union of these per-creation grammars
  must equal the raw map-event ledger. Diagnostic position buffers join the
  applicable post-install immutability interval as well as this global map
  proof.

#### Address and output gates

- An independent CPU oracle reconstructs the visible IDs and expected address
  of every submitted instance in every bucket.
- GPU address output covers every active address plus fixed padding sentinels.
  It has zero duplicate, missing, hidden, wrong-bucket, out-of-range, active,
  and padding mismatches. The exact 256-by-256 target has 65,536 pixels for
  exactly 65,536 logical addresses, so there is no separate target-padding
  domain and no `targetPadding` acceptance gate.
- The address diagnostic owns exactly two position streams: one common A/I
  stream and one F stream whose `z` component compensates the active
  `firstInstance` schedule only for target placement. Across the eight frozen
  schedule installations, exact CPU bytes and hashes are retained, the Three.js
  attribute-manager upload window and native vertex-buffer identity are
  recorded, and the bound diagnostic draws supply the GPU-use witness. A third
  stream, lane-private A/I stream, wrong schedule encoding, late upload, or
  uncommitted vertex buffer fails.
- The address target is `rgba8unorm` in row-major logical-address order:
  logical address `a` occupies bytes `4*a` through `4*a+3`. An active pixel
  encodes the object ID fetched from the expected scheduled source address,
  plus one, as little-endian RGB24 with alpha 255. A padding logical address
  is four zero bytes. It does not encode the numeric source address itself.
- The deterministic frozen survivor-ID buffer is byte-identical when read
  through A, I, and F, and the lanes produce identical color, depth, and
  object-ID targets. The three earlier live-cull snapshots need exact
  membership but may have different within-bucket order. Target comparison
  uses raw bytes and exact SHA-256 commitments, not screenshot tolerance.
  Every distinct color, depth, and object-ID byte sequence is retained once as
  a content-addressed raw witness; every capture references one witness by
  type, byte length, and digest. The offline runner decodes and rehashes those
  witnesses and reconstructs A/I/F byte equality without trusting a
  page-authored equality flag. The A object-ID bytes are the frozen
  per-scenario/per-schedule baseline for I and F; the separate full-domain
  address oracle, command readbacks, and geometry commitments establish the
  submitted address coverage because occlusion means a final object-ID image
  need not contain every submitted object.
- Preflight, sentinel challenge, restored-production, and postflight snapshots
  all pass. The canonical, S1, and S2 address hashes are pairwise distinct.
  Color, depth, and object-ID hashes need not differ across schedules, but A,
  I, and F remain exact within each schedule. Restoring the production bases
  must restore every original hash.

#### Resource and lifecycle gates

- A owns exactly one bucket-base vertex stream. I and F own none. I creates no
  replacement uniform/storage buffer, bind group, hidden vertex stream, or
  per-draw render object for its base.
- Each I bundle recording uses one 32-element CPU `Uint32Array` as its
  `setImmediates()` source. It is allowed and is not counted as a GPU resource;
  no per-draw typed-array view is allocated. Within each visibility-level
  snapshot sequence, the snapshot challenge performs exactly one audited
  post-`finish()` detach of that source and creates exactly one replacement for
  the next recording. All other schedule changes mutate the current whole
  source in place. The final conjunctive evidence requires exactly two detach
  events and two replacement events overall--one of each in v99 and v20--and
  binds each event to that scenario's S1-finish/S2-recording chain; runtime
  schedule coverage alone is not evidence of this lifecycle gate.
- Common matrix, bounds, visible-ID, geometry, material, target, camera, and
  compute resources have exact content commitments. Lane-private indirect
  buffers have equal shape and differ only in word four as declared.
- Within each visibility-level sequence, from preflight through restored
  production, identities and contents remain fixed except for the prescribed
  three schedule transitions: A's bucket-base attribute version advances by
  three, F's command-attribute version advances by three, and I's root version
  and bundle-record count each advance by three; A and F record no replacement
  bundle. The schedule-transition-attributed advances sum to six for each of
  those four I/A/F counters over the two visibility challenges. The intervening
  prescribed `v99` to `v20` scenario load is inventoried separately: its
  content replacements and any resulting attribute-version advances must match
  an independently reconstructed scenario-load delta rather than being folded
  into the six-transition total. At each restored-production point those
  resources contain the canonical bytes. Resource IDs,
  pipeline-cache entries, common resources, and renderer memory counters are
  then exact from restored production through postflight, with no further
  version or bundle change.
- The integration performs no production-resource rebuild, shader replacement,
  disposal, or absent production-resource construction inside a challenge.
  Diagnostic readback may use transient staging buffers only when every such
  buffer is labeled by capture phase, has an exact descriptor and creation and
  destruction record, is excluded from the persistent-resource comparison, and
  leaves no live staging buffer after that readback. Disposal detaches every
  created root and destroys every persistent buffer afterward.
- Persistent texture creation and every derived texture view are inventoried
  with parent identity before the applicable freeze. The five declared target
  textures and their views are pinned independently. Every other created
  texture is admitted only as a pipeline-bound Three.js default: it has at
  least one recorded child view and every such view appears in an actual
  captured bind-group entry. Every creation and view belongs to exactly one of
  those two classes, target views appear only as the declared attachments,
  pipeline-default views appear only in bind groups, and every view is used
  only in its allowed class. Texture
  writes, external copies/imports, late views, and challenge-time texture
  destruction are zero. Cleanup must destroy every persistent texture and
  buffer exactly once after its final recorded use. Final use includes queue
  writes and maps, bundle/render/compute events and finishes, readback
  transfers and their owning encoder finishes, and the corresponding singleton
  submission; the runner joins each destruction back to its unique creation
  rather than trusting aggregate counts.
- Page errors, WebGPU validation/internal errors, unexpected device losses,
  promise rejections, and browser crashes are all zero. The final intentional
  `device.destroy()` must produce exactly one terminal loss record whose reason
  is `destroyed`; it is cleanup evidence, not an unexpected loss.

Runtime orchestration booleans such as callback `pass` and
`runtimeCoverageComplete` are scheduling assertions only. They satisfy none of
the shader, pipeline, raw-output, resource, or lifecycle gates by themselves;
the offline runner reconstructs every conjunct from retained bytes,
descriptors, identities, and challenged execution chains.

### Render-bundle snapshot semantics

Immediate state is encoder state, not an external buffer. The production bundle
therefore records the I value before each draw; it never expects a value set on
the outer render pass to flow into a bundle.

The disposable canary proves these semantics explicitly:

1. Record and finish an I bundle from a 32-value non-monotonic base array.
2. Mutate and detach the JavaScript source view after `finish()`, then execute
   the already-finished bundle. Output must retain the values present at each
   `setImmediates()` call.
3. Increment the bundle version and record a new bundle from a distinct valid
   base permutation. Its output must change to the new oracle.
4. Restore canonical bases, record once more, and recover the canonical output
   hashes.
5. Execute serialized, cleared captures of I, A, and F in each order. Each
   bundle's result must be invariant to the preceding completed capture;
   recorded I data must not contaminate persistent renderer, cache, or bundle
   state used by another lane. This ordered integration challenge does not
   claim same-render-pass inheritance coverage.
6. In the raw disposable device only, execute a bundle, restore every direct
   draw state required after `executeBundles()` (including pipeline, index and
   vertex buffers, and bind groups), and attempt a direct I draw while omitting
   only `setImmediates()`. The expected scoped validation failure establishes
   that bundle execution did not leave a usable outer-pass immediate slot. This
   is the same-render-pass leakage control. The production path always sets
   every required slot.

The evidence records the value sequence, call/draw adjacency, encoder kind,
bundle identity/version, source-view mutation, and all output hashes. Merely
observing the final color image is insufficient.

### Phase-0 stop rule

Every gate is conjunctive. If any target/browser combination is unsupported,
the result is `technical-canary-unsupported`. If a supported combination fails
any gate, it is `technical-canary-failed`. The failure artifact and all prior
artifacts remain immutable. A source fix uses a new run ID and repeats the
entire canary; it does not overwrite or resume the old attempt.

Only a complete pass can authorize Phase 1 for the exact same source,
dependency closure, browser binary, adapter/backend, and integration digest.
It does not authorize an efficacy claim.

## Phase 1: compact attribution diagnostic

Phase 1 has two deliberately different estimands. The interleaved topology asks
what changes among A/I/F inside one established browser/device state. The
lane-alone topology asks whether fresh processes enter persistent performance
modes and whether lane contrasts survive that deployment-like variation. Their
rows are never pooled, and agreement is not converted into a candidate pass.

All diagnostic artifacts state:

```text
executionMode: mechanism-diagnostic
analysisEligible: true
analysisScope: descriptive-mechanism-only
candidateDecisionAllowed: false
```

There is one fixed attempt per diagnostic. No frame, block, session, triad, or
mode is deleted, winsorized, replaced, or rerun because of its value.

### Diagnostic 1: tightly interleaved same-browser superblocks

There are exactly two fresh browser/device sessions, one fixed at 99%
visibility and one fixed at 20%, in that order. All three validated lanes are
resident in each session. Each owns a static bundle recorded before warmup;
only the selected bundle is executed on a frame. No resource, indirect command,
or immediate value is changed during warmup or measurement.

One cyclic 18-frame superblock contains all six lane permutations in this exact
order:

```text
A I F | A F I | F I A | I A F | I F A | F A I
```

Each lane appears six times, twice in each absolute triad position. Across the
cyclic sequence, including the final `I -> A` boundary into the next block,
each of the six directed unlike-lane transitions occurs exactly three times.
There are no same-lane transitions.

Each session executes eight unretained conditioning superblocks followed by 32
retained superblocks: 144 warmup frames and 576 measured frames. Measurement
starts at a superblock boundary. Every frame clears and renders the same target
and receives one fine-resolution GPU render timestamp. Timestamp resolve and
readback happen only after the complete measured sequence.

For superblock `b`, lane response is the arithmetic mean of its six frame
durations. The retained descriptive contrasts are:

```text
d[b,I-A] = mean(b,I) - mean(b,A)
d[b,F-I] = mean(b,F) - mean(b,I)
d[b,F-A] = mean(b,F) - mean(b,A)
```

Report all 32 values, their conventional median, range, quartiles, negative
count, and results split by the lane's two absolute positions and six immediate
predecessor exposures. Frames are repeated observations, not independent
replicates. The two visibility sessions are reported separately; their
difference is descriptive and is not a dose-effect test.

This topology minimizes elapsed-time, process-mode, and device-state separation
between lanes. It necessarily uses co-resident pipelines, resources, and
bundles, so it does not estimate cold start, lane-alone deployment, bundle
recording CPU cost, or the probability of entering a process mode.

### Diagnostic 2: fresh-browser lane-alone order triads

The second diagnostic launches exactly 18 browser/device/profile sessions.
Each browser constructs, primes, and times exactly one lane; code capable of
constructing either absent lane is not invoked after boot. A browser is fully
closed and its disconnect observed before the next launch. The server may
remain alive. A fixed two-second post-disconnect interval and the existing TCP
port-health gate apply before every next launch.

The 18 browsers form six order-balanced triads. `H` means visibility order
`[0.99, 0.2]`; `L` means `[0.2, 0.99]`. The lane and visibility orders are:

| Triad | Browser lane order | Visibility orders by position |
| ---: | --- | --- |
| 1 | `A I F` | `H H L` |
| 2 | `A F I` | `L L H` |
| 3 | `F I A` | `H L H` |
| 4 | `I A F` | `L L H` |
| 5 | `I F A` | `H H L` |
| 6 | `F A I` | `L H L` |

Every lane occurs six times, twice in every launch position, three times with
each visibility order, and once with each visibility order in every
lane-by-position cell. The schedule is immutable and is not reordered in
response to telemetry or an observed mode.

Each visibility trial uses 320 warmup frames and 480 retained render-only
frames partitioned into 60 consecutive eight-frame blocks. The session response
is the conventional median of the 60 block means. A triad contrast subtracts
the two applicable session responses in the fixed `I-A`, `F-I`, or `F-A`
direction. Report all six triad contrasts and their conventional median by
visibility, lane launch position, and visibility exposure. With six triads,
these are descriptive process-level observations, not population inference.

Fresh-process evidence additionally retains the pre-response predictor set:

- browser, renderer, adapter/device, and GPU-process launch ordinals and OS
  process identities where the platform exposes them;
- pipeline/layout/shader/bundle hashes and creation/first-use serials;
- Vite module hashes and cache-entry/memory snapshots;
- pre-session and per-phase GPU clock, P-state, temperature, power, utilization,
  and competing-compute-process sets; and
- visibility order, lane launch position, elapsed time, and prior disconnect
  gap.

These fields are a fixed screen, not a license to select a favorable subgroup.
The earlier standalone artifacts did not retain a predictor of their discrete
mode, so any new association is exploratory until reproduced prospectively.

For each lane, sort the six 99% session responses and publish all five adjacent
gaps. A predeclared `candidate discrete separation` flag is set only if the
largest gap is at least 0.05 ms and at least five times the second-largest gap.
The partition is then carried unchanged to the paired 20% responses. It is
called `persistent across visibility` only if those two groups remain
non-overlapping at 20% in the same direction. The flag is a compact diagnostic
description informed by the already observed modes. It is not a significance
test, an efficacy gate, or proof that no mode exists when the flag is absent.

### Shared timing and evidence gates

Before and after every measured interval, the applicable Phase-0 shader,
pipeline, command, indirect-word, address, output, resource, and bundle
commitments must remain exact. All measured rows must have one positive render
duration with one uniquely attributable timestamp UID and a quantum no greater
than 1,000 ns. The render target, viewport, camera, clear state, bundle identity,
command buffers, and GPU resource versions are fixed.

Telemetry coverage, constant adapter/GPU association, source/dependency/runtime
identity, page/browser lifecycle, zero competing-process-set change, zero
WebGPU errors or losses, and exact artifact inventory are mandatory technical
gates. Failure makes that diagnostic `invalid`; it does not make a lane fast or
slow. Partial data are preserved but receive no contrast summary.

Raw artifacts are committed incrementally with compressed and logical byte
lengths and SHA-256 hashes. The final manifest declares every file; an
independent verifier rejects missing, extra, duplicate, aliased, symlinked, or
hash-mismatched paths and rebuilds the frozen plan and all summaries. The
self-declared manifest establishes internal consistency, not authorship or an
external timestamp.

## Falsification and interpretation rules

The following rules are fixed before Phase 0 implementation:

- If I cannot pass the raw direct-pass and render-bundle canaries, immediate
  attribution is technically unsupported on that exact browser; do not fall
  back silently to A or a buffer-backed imitation.
- If I requires a bucket-base vertex stream, replacement GPU buffer/bind group,
  one render object per bucket, outer-pass state, or post-record mutation of a
  finished bundle, the proposed Three.js integration is falsified.
- If words zero through three, the deterministic frozen survivor packing,
  shader-normalized common code, or output bytes differ unexpectedly, the
  A/I/F contrast is invalid. Different legal order in the earlier independent
  atomic-compaction snapshots is not itself a mismatch.
- If the same-browser `I-A` contrast descriptively carries most of `F-A` while
  `F-I` is much smaller, the evidence points toward removal of the vertex-base
  path rather than uniquely toward indirect `firstInstance`.
- If `I-A` is much smaller than `F-I`, the evidence points toward the built-in
  first-instance/address-add path. If I is distinct from both A and F,
  immediate data has its own path-dependent effect. Exact distributions must
  accompany these descriptions; this diagnostic has no frozen equivalence or
  component-dominance margin, so `equivalent` and `accounts for` are forbidden
  as formal conclusions.
- If tightly interleaved contrasts are stable but fresh-browser triads split
  into persistent modes, process topology is a material moderator and a future
  candidate must block, predict, or randomize over it. Averaging the modes away
  is not mechanism attribution.
- If both topologies show materially different directions, neither replaces
  the other. Report `topology-dependent attribution unresolved` and stop before
  a candidate.
- If both topologies are directionally concordant and every technical gate
  passes, the only allowed conclusion is that a separately preregistered
  candidate may be designed. Phase 1 itself still has no efficacy pass.
- Failure to reproduce a discrete-mode flag in 18 browsers does not prove that
  the modes disappeared. Conversely, observing a predictor association in this
  small fixed screen does not establish its cause.
- No Chrome 151-versus-152 conclusion is allowed without complete same-topology
  runs against exact hashed binaries. The earlier live and standalone records
  remain jointly confounded.

All null, negative, heterogeneous, and technically failed outcomes are retained
unchanged. There is no optional stopping, post hoc lane definition, adaptive
schedule, retry for an unfavorable value, or promotion of a canary into a
performance result.

## External technical references

- [Chrome 149--150: WebGPU immediates](https://developer.chrome.com/blog/new-in-webgpu-149-150)
- [Current WebGPU specification](https://gpuweb.github.io/gpuweb/)
- [Current WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/)
- [Three.js issue #33576: WebGPU immediate-data integration](https://github.com/mrdoob/three.js/issues/33576)
