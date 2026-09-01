# Live indirect `firstInstance` setup-order diagnostic

## Purpose

The completed first-device candidate pair found a repeatable, render-dominated
feature-minus-portable reduction, but it did not meet its preregistered
confirmation rule. One matrix exceeded the nuisance bound for the bundled
`lanePhysicalOrder` factor, while the second matrix showed a smaller interaction
in the opposite direction. The candidate result remains
`confirmation-not-met` and is not replaced or reinterpreted by this diagnostic.

The bundled factor controlled four distinct setup exposures. This development
experiment separates them to determine whether the measured response depends
on:

- `C`: lane-specific resource construction order;
- `K`: first compute-pipeline use order;
- `R`: render-pipeline and bundle priming order;
- `T`: the lane used to pre-prime the timestamp pools.

Each binary factor uses portable-first/portable for level zero and
feature-first/feature for level one. The diagnostic is descriptive. It has no
candidate pass state and cannot establish deployability, cross-device
generality, end-to-end frame improvement, or a general Three.js/WebGPU
optimization claim.

## Fixed workload and timed estimator

The workload, dual-resident lane implementations, correctness gates, timed
frame body, crossover schedule, and estimator remain those in
[`INDIRECT_FIRST_INSTANCE_LIVE_PROTOCOL.md`](INDIRECT_FIRST_INSTANCE_LIVE_PROTOCOL.md),
with these fixed restrictions:

- 65,536 objects, 32 geometry buckets, baseline layout, and 99% visibility;
- 320 warmup frames and 480 retained measured frames per trial;
- the same 1280 by 720 offscreen target and one compute plus one render
  submission per frame;
- no lower-visibility decision cell;
- no artificial timestamp-resolution delay;
- no frame, block, trial, or session removal, replacement, or outlier filter.

For total GPU-pass, render, and compute duration, each trial retains the existing
previous-lane-stratified median of 60 eight-frame block deltas. Negative values
favor the feature lane. Total GPU-pass still means only timestamped compute plus
timestamped render duration.

## Setup topology

Construction follows `C`. Priming then uses a staged topology so `K` and `R`
can vary independently:

```text
construct both lanes in C order
perform first compute use for both lanes in K order
prime render pipeline and bundle use for both lanes in R order
pre-prime timestamp pools with lane T
```

The staged topology differs from the candidate's interleaved
compute-then-render sequence. Absolute diagnostic timing is therefore not
pooled with either candidate matrix. Every artifact records the planned and
observed construction, compute-first-use, render-prime, and timestamp-preprime
choices. A mismatch invalidates the trial.

Each trial builds a fresh strategy, but all 32 trials in one session reuse that
session's renderer and device. This diagnoses order sensitivity under
session-resident driver and renderer caches; it is not a cold-device or
standalone-deployment first-use experiment.

Shader capture, serialized correctness validation, render parity, timestamp
registration, and timed-lane selection retain their fixed canonical or
scheduled orders. They are held constant rather than promoted to additional
post hoc factors.

## Frozen factorial plan

The plan is the full `2^4` set of 16 `C/K/R/T` cells. There are exactly two
fresh browser/device sessions. Within each session, one committed 16-cell
permutation is followed by its exact reverse, producing 32 trials per session
and 64 trials overall.

Consequently every cell:

- occurs twice per session;
- occupies positions `p` and `31 - p`;
- appears once in each forward/reverse block;
- receives both crossover-schedule orientations within each session.

All factor levels and all pairwise cells are balanced within session and
orientation. Session one must close its browser process before session two is
launched. The server may remain alive, but the page, context, renderer, adapter,
device, and browser process may not cross the session boundary.

The retained lifecycle proves that Playwright observed the sole page and
context close and the browser connection disconnect before the next launch. It
does not retain an operating-system PID attestation for every browser helper
process.

Reverse-position pairing balances an additive linear position drift. It does
not prove the absence of nonlinear thermal/clock drift, driver-cache history,
or carryover from the preceding trial. The second permutation and separate
session expose some of that sensitivity, but session disagreement remains a
reason to stop rather than a quantity to average away.

## Evidence boundary

Every trial must pass the existing strict preflight, timing-start, completion,
and postflight gates, including exact output parity, survivor membership,
addressing, shader structure, static-resource lifecycle, timestamp shape,
schedule, and zero WebGPU/page errors. The forced-feature-off deployment gate
runs once in a separate disposable browser before either diagnostic session.

Raw rows and retained evidence are written per trial to Brotli-compressed
artifacts as the run proceeds. A manifest commits the compressed and
uncompressed bytes. Failed or partial runs remain diagnostic evidence and are
not silently retried. A source correction requires a new run identifier and a
new complete 64-trial plan.

The independent verifier rejects undeclared files and path aliases, rechecks
every JSON and Brotli byte commitment, reruns the forced-off and strict
per-trial evidence gates, reconstructs all trial estimates and factorial
contrasts, and validates the served-module audit:

```text
npm run verify:first-instance-live-order -- <run-directory>
```

This establishes internal artifact consistency. The development manifest is
self-declared and is not an authorship or external-timestamp signature.

The installed source/dependency runtime closure, retained browser identity
(executable basename, launch arguments, and user agent), backend, adapter,
feature availability, viewport, workload manifests, and relevant environment
identity must remain exact across all trials and both sessions.
The machine does not need to have zero resident GPU processes, but active
competing GPU workloads should be paused for the full sub-millisecond timing
run. Ordinary inactive resident processes are permitted. The within-session
reversal and second fresh session expose residual shared-device drift; they are
not substitutes for avoiding sustained competing work, and interpretation
remains limited to the observed machine state.

## Factorial summaries and interpretation

The two reverse-position occurrences are averaged into one response for each
session and factorial cell. For each metric, all 15 standard factorial
contrasts are reported separately by session and on the equally weighted
pooled cell means. The sign convention is:

```text
effect = mean(response at contrast level +1)
       - mean(response at contrast level -1)
```

This yields four main effects, six two-factor interactions, four three-factor
interactions, and one four-factor interaction. Percentage-point contrasts are
reported alongside milliseconds. Frames and blocks are repeated observations;
they are not treated as independent samples. With only two sessions, the
factorial table is a mechanism diagnostic, not an inferential population
model.

The next engineering decision uses the complete pattern:

- If the feature reduction remains negative across the factorial cells and
  setup contrasts are small and session-concordant, proceed to a standalone
  fresh-session deployment-topology comparison.
- If a material response localizes to one or more setup exposures, remove or
  control that mechanism before any new candidate claim. The prior candidate
  pair remains unchanged.
- If cell signs or factor effects reverse substantially between sessions, treat
  dual-resident initialization state as unresolved and do not use this harness
  for a performance claim. Continue the addressing specialization, if at all,
  with a standalone fresh-session comparison.
- If the feature advantage does not survive the cells, stop investing in this
  representation path and redirect the research program to a different source
  of logical work reduction, beginning with the fixed-view contributor oracle.

Thresholds from the candidate protocol may be shown as reference lines, but
they are not converted into a diagnostic pass/fail rule and do not amend the
completed preregistration.
