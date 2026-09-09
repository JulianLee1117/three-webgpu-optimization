# WebGPU immediate-data results

## Current status: the full r185 Phase 0 gate did not pass

The sole full Three.js r185 Phase 0 attempt ended
`technical-canary-failed`. It is retained as a failed one-shot, has not been
rerun or replaced, and does not authorize the development-source run or either
Phase 1 diagnostic. No timing, lane delta, or efficacy result was captured.

The nested raw WebGPU canary passed before the integrated workload began. On
the exact tested Chrome installation it independently established default
exposure and correct render-pass and render-bundle immediate-data behavior.
That narrow capability result remains valid; it is not a substitute for the
full A/I/F gate.

The integration then stopped at the first `address/A` diagnostic. The
diagnostic oracle expected `position` at vertex location 0 and `bucketBase` at
location 1. The pinned Three.js sources deterministically generate and bind
the reverse order for that graph: `bucketBase` at location 0 and `position` at
location 1. The retained record shows that only the two oracle comparisons
failed. The bound inputs, 32 indirect draws, immediate calls, storage/index
bindings, pipeline topology, bundle execution, and target linkage all matched.
This is evidence of a harness false rejection, not evidence that the integrated
immediate-data path passed: later A diagnostics, all F/I diagnostics, final
readbacks, and the complete Phase 0 evidence were never reached.

A second harness defect masked that page failure in the top-level `failure`
field: the runner attempted to hash absent page evidence and threw
`Buffer.from(undefined)`. Post-attempt code repairs make the oracle explicit
and the validators total and fail-closed, but they do not retroactively change
the immutable result. The early page exit also preceded normal error-scope
drainage. The artifact records no uncaptured GPU error or unexpected device
loss, but it cannot prove that the scoped validation, internal, and
out-of-memory error queues were empty.

## Immutable full-run execution record

```text
Source commit:  d02d71afa39597d6912cf590bf20dbb78a07b18c
Source tree:    3fb66ceb51da7f719ae42544a46aaea615bcc4cd
Target:         Three.js 0.185.1 / revision 185
Browser file:   Chrome 152.0.7977.82
Browser SHA-256: 8cd23aec3a30479b9b8db2063e70526c88a7ec2c99cd744603eb26ab598733ed
Attempt count:  1
Retry count:    0
Launch flags:   none
```

Top-level failed artifact:

```text
results/development/three-immediate-aif-phase0-r185/phase0-7c802640a9cb1e081fcaa4c3eed74cda/report.json
```

It is read-only, contains 4,284,628 bytes, and has SHA-256
`04c8f013aae5bc4701fb07bf07356032fb3b2bb19986419c5e38bea7480a1173`.
Outer context, browser, and server shutdown all succeeded.

Nested raw capability artifact:

```text
results/development/webgpu-immediates-default-exposure-canary/canary-8087a18b4db3827e49617493/report.json
```

It is read-only, contains 12,177 bytes, and has SHA-256
`99e90c3a9e68757783796830ce70473e157c218a86a4c095fd89f61935fb30bf`.
The raw page passed exact direct outputs, bundle snapshot semantics after source
mutation and detachment, and all three scoped negative controls. It observed
`immediate_address_space`, `maxImmediateSize = 64`, callable render-pass and
render-bundle `setImmediates()`, and `indirect-first-instance`, with zero
uncaptured errors or unexpected device losses. The final raw-device loss was
the intentional device destruction during cleanup.

## Earlier minimal-canary environment

```text
Three.js targets: 0.185.1 / revision 185; pinned upstream revision 186dev
Browser file:     Chrome 152.0.7977.66
Browser SHA-256:  e942c7328883032badb7f136d317ef9142c992f8b65e68ba395f777f12484f38
Raw-canary path:  WebGPU / D3D12 / NVIDIA GeForce RTX 5070 Ti
Immediate limit:  64 bytes
WGSL feature:     immediate_address_space
Required feature: indirect-first-instance
```

The generated overlay is rooted at
`.generated/three-immediate-overlay/`. Its manifest SHA-256 for the passing
integration run is
`b077924a319e4c0d2616c0d4b3923fd4a16a80aa8b11bd2ea8697976553b3438`.
The generator verifies the exact installed Three.js version and the SHA-256 of
every transformed source/build input before copying. It never edits
`node_modules`.

Both current integration canaries reported a non-fallback RTX 5070 Ti/D3D12
adapter. The no-flags raw canary exposed only NVIDIA/Blackwell and non-fallback
identity fields. The minimal integration requests all adapter features while
the raw canary requests only its required feature, so exact raw/integration
device-feature equality remains a full Phase 0 gate.

## Default-exposure raw WebGPU canary

Passing artifact:

```text
results/development/webgpu-immediates-default-exposure-canary/canary-58aa26301a6588348c916296/report.json
```

Report SHA-256:
`0c8339b63f1b3e0dfaef285d3abfba7f68cca5a31f2d0d9076feea6962aa17a3`.

This one-attempt canary supplied an empty custom browser-argument array. It
captured Chrome's effective command line through CDP and found none of the
forbidden unsafe/developer, Dawn feature-forcing, adapter-forcing, or GPU
blocklist-bypass switches. There was no retry or developer-flag fallback.

On the exact Chrome 152 headless/Windows/NVIDIA configuration, the page exposed
`immediate_address_space`, a 64-byte immediate limit, callable render-pass and
render-bundle `setImmediates()`, and the separately required
`indirect-first-instance` feature. The direct, bundle-snapshot, and three
strict scoped `GPUValidationError` controls all passed. This establishes
default exposure for this exact configuration, not for every browser, OS, or
adapter.

## Developer-enabled raw WebGPU canary

Passing artifact:

```text
results/development/webgpu-immediates-canary/canary-27c5498ba1d42f2fbbffe550/report.json
```

Report SHA-256:
`9fe2d84f74b2def5eb4d1b6dfa75e3acfd7701d5429e7e1b7890978dd1726a7d`.

This earlier retained run is a developer-enabled capability result. Its exact launch
arguments include `--enable-unsafe-webgpu` and
`--enable-webgpu-developer-features`; it therefore does not prove that the same
browser binary exposes immediates by default. The separate no-enabling-flags
artifact above supplies that narrower result for the exact tested
configuration; this older artifact cannot substitute for it elsewhere.

The canary established:

- `navigator.gpu.wgslLanguageFeatures` includes
  `immediate_address_space`;
- the device exposes `maxImmediateSize = 64`;
- render-pass and render-bundle encoders both expose callable
  `setImmediates()`;
- each encoder consumed three distinct, non-monotonic immediate values and
  produced the exact expected `r32uint` address outputs;
- changing and detaching the source array after `GPURenderBundleEncoder.finish()`
  did not alter the recorded values; and
- an unset slot after `executeBundles()`, a byte offset of two, and a range at
  byte 64 plus four were each rejected by a scoped `GPUValidationError`.

There were zero uncaptured WebGPU errors and zero unexpected device losses.
The report explicitly records `fullPhase0Pass: false` because the integrated
Three.js canary is a separate gate.

## Minimal Three.js integration canary

Passing artifact:

```text
results/development/three-immediate-overlay-canary/canary-1142abadd0219450badbe3e2/report.json
```

Report SHA-256:
`7e73875af68745b769f646ac1a2f616e6a7e6d2c42109dfae99563df10f41e0c`.

The served-module observer proved that `three/webgpu` and `three/tsl` resolved
to the generated overlay and that no Three.js module was loaded from
`node_modules`. The live renderer then proved all of the following:

- one static `BundleGroup`, one mesh, one indexed geometry, four indirect byte
  offsets (`0`, `20`, `40`, `60`), and draw-specific `firstIndex` values
  (`0`, `3`, `6`, `9`);
- exactly four calls shaped as
  `setImmediates(0, immediateBases, i, 1)`, each immediately before its matching
  `drawIndexedIndirect()`;
- an exact vertex shader containing one
  `requires immediate_address_space;`, one
  `var<immediate> threeImmediateDrawBase : u32;`, and one
  `threeImmediateDrawBase + instanceIndex` address;
- the actual shader modules were bound to the actual immediate pipeline, that
  pipeline was bound by the recorded bundle, and its layout received
  `immediateSize: 4`; a following distinct control pipeline's observed layout
  received zero, proving the two supplied descriptors were `4` then `0` without
  overclaiming which internal reset path produced the second value;
- all command words, including four zero word-four values, matched in CPU
  source and GPU readback, and the readback attribute resolved to the same
  native GPU buffer used by every recorded indirect draw;
- the first and cached second render matched a draw-ordinal-specific exact
  8-pixel RGBA8 address oracle; every output pixel differs under the post-finish
  mutated bases, and exhaustive unit checks reject every base swap, omission,
  and valid single-value mutation;
- both render-pass executions named the exact native bundle returned by the
  single recorded bundle encoder; and
- mutating the four-element immediate source from `[0, 2, 4, 6]` to
  `[1, 3, 5, 0]` immediately after native bundle `finish()` did not change the
  cached bundle output. A second disposable renderer/device with no canary
  instrumentation independently reproduced the same snapshot behavior after
  mutation following its first completed render.

The Node runner independently reconstructs the raw output, complete commands,
encoder adjacency and values, pipeline/layout linkage, cached-bundle execution,
coverage inventory, and source hashes rather than trusting page-authored pass
flags. There were zero page errors, request failures, HTTP errors, browser
crashes, uncaptured WebGPU errors, or unexpected device losses in the declared
scope, which begins immediately after device creation and before renderer
initialization. Adapter/device request errors are outside that listener scope.
The canary intentionally reports `fullPhaseZeroStatus: not-evaluated` because
it is a minimal integration test, not the full protocol workload.

## Pinned upstream development integration canary

Passing artifact:

```text
results/development/three-immediate-dev-overlay-canary/canary-64c8010da08f052a2e320697/report.json
```

Report SHA-256:
`42618608ac4c07ee48bdfb219a52404f0a8c8a4e4478cc50bde8494851d2484e`.

The second target uses upstream Three.js commit
`994260a7a59abe466de8e59db1ecc9e30352751d`, tree
`7f3d4a048ea19bef1a488b0e2e243e47f6b30002`, runtime revision `186dev`.
Its clean tracked payload is materialized from canonical Git index bytes rather
than the checkout's CRLF-converted files. The generated overlay contains 1,244
runtime files and has inventory SHA-256
`d49fa1774e1ed50a5f3908332356cd794123245a714772e255bdef08df3a50c9`;
the canary-run manifest SHA-256 is
`f984730e905ccda2bd9baa298ec3efca2e3b1499912601c2191388b190f36576`.

The same hardened page and independent runner passed every minimal integration
gate listed above against this pinned development source. Served `three/webgpu`,
`three/tsl`, and the transitive core runtime matched their generated hashes;
no Three.js request resolved to `node_modules`. This closes the minimal
source-compatibility question only. The complete A/I/F workload is still
required on both source families for full Phase 0.

## Development failures retained as design constraints

The first live integration attempt used `__threeImmediateDrawBase`. Chrome
correctly rejected it because WGSL identifiers cannot begin with two
underscores. The overlay contract now uses the legal identifier
`threeImmediateDrawBase`, and focused tests pin that spelling.

A later development attempt initially treated both the immediate bundle and
the non-immediate descriptor-reset control as candidate immediate traces. The
evidence selector now requires both four indirect draws and four immediate
calls. Another check compared a live backend data object after the control
bundle had replaced its `bundleGPU`; the passing version snapshots the native
bundle identity before that intentional replacement. Neither failure produced
timing data or an efficacy result.

An earlier technically successful artifact,
`canary-7bd2c572b5035091b3c0dd91`, used a commutative output layout: permuting
the four canonical bases among draws still covered the same eight pixels. Its
runner also trusted several page-authored exactness booleans. It is retained as
development history but superseded, not used as the durable minimal-canary
result. The current draw-ordinal oracle and independent runner validation close
those false-positive surfaces.

## What remains before any performance claim

The complete workload has been implemented, but its first target did not clear
the technical gate. Continuing this exact attribution sequence would require a
separately versioned, preregistered remediation attempt after offline tests,
followed by a complete pass on r185 and then the pinned development source.
Only those passes could authorize the already frozen descriptive Phase 1
schedule. The failed run cannot be resumed, overwritten, or promoted.

The retained positive result is narrower: WebGPU immediate data can be inserted
at Three.js's per-indirect-draw backend seam and recorded into a cached render
bundle on the tested implementation. It does not show full A/I/F correctness,
a speedup, cross-device portability, or deployment benefit.
