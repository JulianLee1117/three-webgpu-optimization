# Standalone indirect `firstInstance` deployment protocol

## Purpose and scope

The completed live candidate pair observed a repeatable, render-dominated
feature-minus-portable reduction, but one matrix exceeded its preregistered
lane-physical-order interaction bound. The subsequent setup-order diagnostic
kept the response negative in all 16 factorial cells in each of two sessions.
It also observed a session-concordant render-priming contrast of approximately
+0.081 ms. Those results justify removing opposite-lane construction and
priming from the next comparison; they do not amend the completed candidate
decision.

This protocol tests steady-state deployment topology. Every measured browser
and device constructs, primes, validates, and times exactly one selected lane:

- `P`: the portable bucket-base vertex-input path with zero `firstInstance`;
- `F`: the feature-gated path with the bucket base in the indexed indirect
  command and no bucket-base vertex input.

It does not measure cold startup, browser presentation, end-to-end latency, or
driver-cache state outside the browser. A passing result supports a deployment
choice only for the fixed Three.js, browser, backend, GPU, driver, and workload.
Hardware generality requires a materially different GPU or backend replication.

## Fixed workload and selected-lane lifecycle

The workload remains fixed at 65,536 static objects, 32 medium indexed geometry
buckets, baseline layout, a 1280 by 720 offscreen target, device pixel ratio 1,
sample count 1, reversed depth, one reset dispatch, one cull dispatch, and 32
indexed indirect draws in one static bundle. The two visibility cells are the
same deterministic nested 99% and 20% subsets used by the live candidate.

A session is one new Chromium process, context, page, renderer, adapter request,
and device request. The preceding browser must be observed disconnected before
the next browser is launched. Sessions may not overlap. A fixed two-second
post-disconnect interval separates launches. A matrix may keep its server alive,
but no measured browser, page, renderer, adapter, or device crosses a session
boundary. Each launch uses a new Playwright-managed temporary browser profile;
no Dawn disk cache or profile directory is reused. Driver-internal cache state
can still persist across sessions, so the experiment is steady-state after its
declared untimed prime, not a cold-start measurement, and the quartet order is
the control for sequential carryover.

Each session selects one lane through the production deployment factory and
constructs and primes that lane exactly once. It must construct no resource,
compute graph, shader observation, render pipeline, material, geometry, command
buffer, bundle, or address oracle for the absent lane. A portable session uses
the explicit unavailable-feature selection; a feature session additionally
requires the real renderer feature to be present.

The assigned lane and visibility order are supplied by a strict initial-page
boot query before the page's first rebuild. The page records exactly one rebuild
and one constructed strategy, rejects a second rebuild, and may not first create
the interactive default strategy. This keeps unrelated lane construction and
undeclared pipeline-cache warming outside every measured and forced-off browser.

The selected lane remains resident for both visibility trials. The two exact
scenario snapshots are computed before timing. At the untimed visibility
boundary, the harness updates only the existing common matrix and bounds
payloads and the corresponding CPU expectation. Buffer sizes and identities,
object and bucket assignments, geometry, material, indirect-command buffer
identity and static fields, compute nodes, bundle, target, camera, pipeline
state, and selected-lane identities must remain unchanged. The normal compute
pass may change only command instance counts and the survivor payload. The
harness then repeats full validation and warmup. A
pipeline, bundle, command-buffer, or selected-lane reconstruction at this
boundary invalidates the session.

This lifecycle removes opposite-lane construction and priming while balancing
ordinary same-lane first-versus-second visibility exposure.

## Frozen two-matrix plan

The binary quartet factors are:

```text
A = P F F P       X = H H L L
B = F P P F       Y = L L H H
```

`H` means that a session runs 99% visibility first and 20% second. `L` means
20% first and 99% second. Consequently, every quartet assigns each lane to one
`H` session and one `L` session. Its early and late adjacent pairs also compare
portable and feature under the same visibility order.

The exact quartet sequences are:

```text
matrix 1: AX AY BX BY  AX AY BX BY  AX AY BX BY
matrix 2: BY BX AY AX  BY BX AY AX  BY BX AY AX
```

Matrix 2 is the exact reverse of Matrix 1. Each matrix therefore contains:

- 12 quartets, with every `AX`, `AY`, `BX`, and `BY` combination three times;
- six `PFFP` and six `FPPF` lane assignments;
- 48 fresh browser/device sessions, 24 per lane;
- two visibility trials per session and 96 trials overall;
- 48 trials at each visibility and 48 trials per lane.

The full frozen study contains two matrices, 24 quartets, 96 measured sessions,
and 192 visibility trials. Matrix 2 runs regardless of Matrix 1's numerical or
stable-environment outcome. The matrices are evaluated independently and are
never pooled to rescue a failed matrix.

## Timed body and trial estimator

Each visibility trial performs exactly 320 warmup frames followed by 480
retained measured frames. Measurement is partitioned into 60 consecutive
eight-frame blocks. Every frame performs only:

```text
update the fixed frustum state
submit the selected reset and cull nodes once
render the selected pre-recorded bundle once
```

Warmup and measurement forbid an opposite-lane action, resource reconstruction,
readback, diagnostic render, extra dispatch, or benchmark-authored storage or
indirect-buffer write. Timestamp resolution and correctness readback occur only
at declared untimed boundaries. Every compute and render duration must be
strictly positive, join its exact frame and submitted resource identities, and
come from a timestamp pool whose observed quantum is at most 1,000 ns.

For session `s`, visibility `v`, and metric `m`, the absolute session response is:

```text
a[s,v,m] = median over 60 blocks(mean of the block's 8 measured frames)
```

For quartet `q`, the feature-minus-portable response is:

```text
d[q,v,m] = mean(the two feature session responses)
         - mean(the two portable session responses)

percent[q,v,m] = 100 * d[q,v,m]
               / mean(the two portable session responses)
```

The matrix response is the conventional median of its 12 quartet responses;
percentage responses are aggregated by taking the median of the 12 quartet
percentages. GPU-pass total is exactly timestamped compute plus timestamped
render. It excludes CPU encoding and submission, queue gaps, presentation,
composition, and end-to-end latency. Render is a secondary confirmatory
endpoint. Compute and exposed CPU submission timing are descriptive.

The paired dose endpoint is the median of the 12 values:

```text
d[q,99%,gpu-pass-total] - d[q,20%,gpu-pass-total]
```

Quartets are the inferential units. Frames, blocks, sessions, and the two trials
inside a session are repeated observations. No frame, block, session, quartet,
or matrix is removed, filtered, winsorized, or replaced because of its measured
value. The 10-of-12 direction rule is a frozen robustness gate, not a calibrated
population significance statement; sequential same-machine quartets may be
autocorrelated.

## Preregistered numerical decision

Negative values favor indirect `firstInstance`. Each matrix must independently
pass every gate below:

- the 99% median GPU-pass-total response is at most -0.10 ms and at most -5%,
  with at least 10 of 12 quartet responses strictly negative;
- the 99% median render response independently meets the same -0.10 ms, -5%,
  and 10-of-12 gates;
- the 20% median GPU-pass-total response is strictly below +0.02 ms and +5%,
  with at least 10 of 12 quartet responses below each upper bound;
- the median paired 99%-minus-20% GPU-pass-total response is at most -0.05 ms;
- every required stratum, drift, correctness, environment, lifecycle, source,
  telemetry, and artifact gate passes.

At 99% visibility, GPU-pass-total responses must remain negative in both levels
of quartet lane order (`A`/`B`), visibility exposure (first/second), adjacent
pair position (early/late), and matrix half (first six/last six quartets). Each
level-one-minus-level-zero interaction must be strictly inside both 0.10 ms and
5 percentage points. The exact ordered interactions are `B - A`, `second -
first`, `late - early`, and `last - first`, respectively. For lane order and
matrix half, each level estimate is the conventional median of the complete
high-visibility quartet effects assigned to that level. For visibility
exposure and adjacent-pair position, each quartet first supplies one
feature-minus-portable contrast from the single feature and portable session in
that level, and the level estimate is the conventional median of those 12
within-quartet contrasts. Percentage level estimates apply the same median to
the corresponding quartet percentages; the percentage-point interaction is
the level-one percentage median minus the level-zero percentage median, not a
percentage computed from the millisecond interaction. Both millisecond level
medians must be strictly negative.

Within each quartet, the condition-blind mean of sessions three and four minus
the mean of sessions one and two contains one portable and one feature session
on each side. Its matrix median must remain strictly inside 0.10 ms and 5%,
overall and by visibility. Within each trial, final-15-block minus
first-15-block condition-blind drift is summarized across quartets and must
remain strictly inside the same bounds overall and by visibility. Specifically,
the first and final values for a trial are the arithmetic means of that trial's
first 15 and final 15 eight-frame block means. Trial drift is final minus first,
and trial percentage drift is `100 * (final - first) / first`. Within a quartet,
the millisecond drift is the equal-weight arithmetic mean of the eight trial
drifts overall, or the four trial drifts at one visibility. Percentage drift is
separately the equal-weight arithmetic mean of those same trials' percentage
drifts; it is not recomputed from the aggregate millisecond drift and aggregate
first value. The matrix values gated against the strict bounds are the
conventional medians of the 12 quartet aggregates, separately for milliseconds
and percentages, overall and at each visibility. Same-lane outer and inner
session drift is reported as an additional stability diagnostic.

Both matrices must pass independently for `standalone-confirmed`. Otherwise the
decision is `standalone-confirmation-not-met`, with both matrix estimates and
all failed gates retained unchanged.

## Fail-closed standalone correctness

Each session uses a single-lane address oracle. The oracle validates the actual
production survivor and command buffers against an independent CPU frustum and
address reference without compiling or constructing the absent lane. At
preflight, timing start, and postflight it must establish:

- exact command fields, including zero portable word four or exact feature
  bucket bases;
- zero overflow and exact canonical survivor membership per bucket;
- exact active and unused address behavior for the selected expression;
- exact RGBA8 color, depth32float, and RGB24 object-ID output;
- two identical captures through the actual timed static bundle;
- stable selected-lane resources, bundle-record count, target, pipeline state,
  viewport, and timestamp registrations;
- no uncaptured WebGPU error, device loss, page error, or unexpected event.

Lane-local shader inspection must prove the portable bucket-base input and
addition or the feature path's direct `instance_index` use and absent
bucket-base input. Lane-local compute inspection must prove exact reset/cull
structure, dispatch dimensions, storage roles, and no executable access to
command word four.

The post-hoc verifier compares records across separate sessions rather than
creating both lanes in a page. It requires equal normalized vertex WGSL after
only the approved address and input-location normalization, byte-identical
fragment semantics, equal normalized reset/cull WGSL after only allowed
lane-local identifier normalization, equal binding roles and payload types,
and exact workload and rendered-output commitments by visibility. Runtime GPU
resource identities are required to be stable within their own session.
Numeric identity tokens are scoped to one fresh JavaScript realm and may repeat
after a browser restart; separate closed lifecycle records, rather than numeric
inequality, establish the cross-session boundary.

An allocation ledger must show exactly one selected lane, one indirect command
buffer, two compute nodes, one production material, mesh, and bundle, and zero
absent-lane constructions. Loading repository source that contains both factory
branches is not itself a second GPU lane; invoking, compiling, or allocating the
absent branch is forbidden.

## Evidence, environment, and stopping rules

The source commit, clean worktree, installed dependency closure, browser build
and arguments, Three.js revision, backend, adapter and driver, viewport,
reversed-depth state, feature availability, scenario inputs, and served-module
bytes are fixed before the first candidate reservation and must remain exact
through both matrices.

One forced-feature-off gate runs in a separate disposable browser before each
matrix. It must use the production deployment factory, construct only the
portable lane, retain zero command word four, and pass the same lane-local
correctness gates. That browser must disconnect before the first measured
session.

The pre-candidate smoke is an exact excluded prefix: the first portable and
feature sessions of matrix 1, with both ordered visibility trials in each. It
therefore exercises the fresh-process boundary, the sole visibility switch,
shader-observation ordinals 1 through 12, and cross-lane offline comparison,
but its four trials cannot enter either matrix estimate or decision. Smoke-mode
analysis accepts exactly frozen plan indices 0, 1, 2, and 3 in that order and
rejects every other subset. Partial-mode analysis may accept any nonempty
strictly plan-ordered subset, but is likewise excluded from every decision.

Evidence is written incrementally after every trial using bounded
Brotli-compressed JSON. The manifest commits compressed and logical JSON bytes,
the complete frozen plan, source and dependency identity, session lifecycle,
selected-lane allocation ledger, cache and renderer state, workload and shader
evidence, raw timing rows, telemetry, and reconstructed summaries. The
independent verifier rejects undeclared files, path aliases, symlinks, missing
or surplus sessions or trials, a changed plan, a cross-session identity leak,
and any byte, semantic, estimator, gate, or inventory mismatch.

The Nvidia telemetry collector samples throughout each matrix at 250 ms. It
must have valid identity, adapter association, liveness, and coverage. Quiescent
process snapshots are taken before the first measured session and after each
browser disconnect; the external process identity set must remain stable. GPU
temperature, utilization, clocks, power state, memory, and power are retained
as context and receive no post-hoc value threshold.

The GPU need not have zero resident processes. Active competing games, video,
wallpapers, renders, or GPU-compute workloads are paused for the complete run;
ordinary inactive resident processes are permitted. Once a matrix reservation
begins, an observed process-set change is an environment failure for that
matrix, not a reason to replace its measurements.

Only independently logged browser/device loss, telemetry-collector failure, or
artifact I/O failure is retryable, and retrying restarts the complete matrix in
fresh sessions. Performance, drift, process-set changes, validation failures,
directional outcomes, and near-threshold results are never retried or replaced.
An implementation or evidence correction requires a new clean commit and a new
two-matrix series. Partial and failed artifacts remain preserved.

## Expected execution cost

The complete pair contains 153,600 warmup-plus-measured frames and 92,160
retained measurement rows. Including the two forced-feature-off gates, it uses
98 sequential browser launches, with only one measured browser alive at a time.
At a 60 Hz browser frame cadence, the timed-frame schedule alone requires
approximately 43 minutes, and the fixed process gaps add approximately three
minutes. Browser startup, correctness capture, readback, compression, and
teardown make 50 to 75 minutes a practical pre-verification planning range;
the excluded smoke records the observed host cadence before candidate timing.
Independent verification and packaging are expected to add approximately 5 to
15 minutes. Retained compressed evidence is expected to occupy roughly 40 to
120 MB; logical decompressed JSON is expected to be approximately 1.5 to 3 GB.
Reserving 4 GB of working storage is sufficient for the frozen design.
