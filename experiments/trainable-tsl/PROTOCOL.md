# Trainable native TSL: bounded protocol v1

Finalized prospectively on 2026-09-09 before the first GPU probe. Freeze this
protocol, fixture definitions and source hashes with every report. Later changes
require a new report cohort and an explicit amendment; do not replace failed
reports.

Amendment before the final four-cell cohort: the initial 2026-09-09T20-08-22.166Z
neural/11 canary passed its numerical checks but failed the strict browser-error
gate because `/favicon.ico` returned 404. The server now answers that route with
204. Independent CPU review also found that unused designated inputs escaped
finite-value/width checks; registration now validates those inputs eagerly.
Both corrections retain the original report and require fresh source hashes.
Review of the same canary also found that the export/reload check mutated the
CPU array used as a reset image, so the second lane inherited learned weights.
The immutable reset image now has separate storage, and every fit reads back
and checks its actual GPU starting weights against the declared initial vector.
That canary's second-lane comparison is invalid. The fixture definitions,
optimizer settings, sample counts and numerical gates are unchanged. The
original canary is not promoted to a passed final-cohort run.

No outcome is asserted here. The scope and prior-art boundary are in
[PRIOR_ART.md](PRIOR_ART.md).

## Hypothesis and fixtures

A supported native Three TSL forward expression can acquire automatically
generated reverse-mode parameter gradients, learn from a small fixed set of
height observations on WebGPU, and render/export learned weights through the
same forward function. The proposed benefit is eliminating a separately authored
backward model. This is not full-renderer differentiation or a new optimizer.

Use the two current `fields.js` programs, which supersede the earlier rough
24/65-parameter concept:

- `islands`: nine movable Gaussian bumps plus a bias, 28 scalar parameters.
  Amplitudes and center coordinates are trainable; the shared width is fixed.
- `neural`: a two-input, twelve-unit sine hidden layer and scalar linear output,
  49 scalar parameters including biases. This is a small neural height field.

Only the forward programs are authored. Changing graph kind must not select a
manually derived backward formula. Inputs occupy the square `[-1,1]^2`. Each
fixture uses 128 seeded random `(x,z,height)` observations and the independent
analytic target function declared in `fields.js`. Targets need not be exactly
representable by the fitted family. Neither optimizer receives target parameters
or uses the held-out set during training.

The matrix is exactly four cells: `neural` and `islands`, each with seeds 11 and
29, with both methods in every cell. Seed 11 runs AD then FD; seed 29 reverses
that order. There are no seed-47 cells or extra seed-11 repetitions. A failed
canary may motivate a source correction, but any resulting run belongs to a
distinct cohort, with the failure retained. These four cells do not represent a
wider population of scenes. Stop on correctness or device failure rather than
continuing an invalid matrix.

## Correctness gates before accepting a fit

1. CPU checks exercise supported operator rules, shared DAG paths, repeated
   variables and swizzles, scalar/vector broadcasting, and explicit rejection of
   unsupported or mutable nodes. The compiler may accept designated opaque
   inputs, whose declared independence is the caller's responsibility. Its
   bounded support list and graph limits must be recorded from `autograd.js`.
2. Compare GPU forward predictions with the separately written double-precision
   JavaScript `numericField` formula. Maximum absolute error must be below
   `2e-5` on every training sample, at the initial weights and at each method's
   final learned weights.
3. Read all per-sample parameter gradients from the actual GPU-generated
   graph. Compare each against central differences of squared residual using the
   independent double-precision formula, step `1e-5`. Require finite values and
   `abs(error) <= 2e-4 + 2e-3 * abs(reference)` for every component. This checks
   3,584 components for islands and 6,272 for neural at each checked state, not
   only an aggregate. Check both initial and each method's final learned state.
4. Validate the GPU finite-difference control against that same independent CPU
   oracle. Its step is `0.001`; require
   `abs(error) <= 3e-3 + 5e-3 * abs(reference)` for every component. This looser
   gate acknowledges f32 subtraction. Record both methods' actual maximum
   errors and failure counts. Do not interpret the two tolerances as identical
   numerical accuracy or silently tune the step after observing outcomes. Run
   this check at the same initial and final states as the AD check; retain the
   raw GPU AD, GPU FD and CPU reference gradient vectors.
5. Before training, independently calculate the first Adam update on the CPU
   from the mean of the actual GPU AD partials. Compare every resulting GPU
   state component: weight absolute error at most `5e-4`, first/second moments
   and padding absolute error at most `2e-6`, all finite. Record the mean
   gradients, expected state, actual state and failures. This isolates reduction
   and optimizer-state correctness from the separately checked gradient formula.
   Restore initial weights and zero moments afterward. This also warms the
   update pipeline before either measured training call.
6. Check final weights and metrics for finiteness. JSON export/reload must
   preserve every weight exactly and reproduce the final GPU AD gradient array
   exactly. This reload gate compares gradients, not rendered images or forward
   predictions; forward predictions have their separate initial/final gate.
   Rendering must call the same authored `field` function with learned weights.

The double-precision finite-difference oracle is independent of the AD traversal,
but is still numerical validation rather than a proof. Initial/final checks do
not validate every intermediate optimizer state or all possible native TSL
programs. First-step Adam validation does not independently validate all later
bias-correction states.

## Fair GPU control

Compare generated reverse-mode AD with central finite differences of the **same
native TSL forward graph on the GPU**, not with a CPU-only implementation. Both
receive byte-identical samples and starting weights, zeroed Adam moments, the
same loss reduction, update kernel, learning rate and iteration schedule.

Use mean squared height residual, learning rate `0.025`, Adam beta1 `0.9`, beta2
`0.999`, epsilon `1e-8`, and clamp resulting weights to `[-8,8]`, as in the current
engine. Both methods get at most 160 completed updates on the same fixed batch
of 128 samples. Finite differences use central perturbations `+/-0.001` for every
parameter, with no parameter-specific tuning. This is an equal-update comparison,
not equal arithmetic work. Record completed updates, gradient invocations,
forward evaluations and actual dispatches for each method.

The present layout computes a per-sample gradient array and shares the same
GPU reduction/Adam update. AD has 128 gradient invocations per step; finite
differences have `128 * parameterCount`, each evaluating two perturbed forwards.
Invocation counts are not a measured speedup. Do not call either path fused
end-to-end, compare it with unmeasured Slang/jax-js, or claim that this layout is
optimal. No full training tensors may be read back each iteration for CPU
optimization; startup validation and final parameter export are separate reads.

## Utility and reporting

The held-out set is the fixed offset 31-by-31 grid in `fields.js`, independent
of the seeded training locations. Report initial and final training RMSE,
held-out RMSE, held-out maximum absolute height error and completion status for
every field/seed/method. The capability gate is **at least 80% held-out RMSE
reduction**, `final <= 0.2 * initial`, for each accepted fixture, not merely an
80% training-loss or pooled-average improvement. A partial or timed-out fit
cannot pass the 160-update gate. Report failures even if another seed looks good.

An AD fit passing this gate demonstrates trainability of the selected native
graphs. Whether it beats finite differences is a separate empirical result.
Comparable final quality can still establish the integration capability; equal
updates alone cannot establish efficiency. Compare per-cell outcomes and retain
both methods' results. A clean visual convergence is supporting evidence only.

Current observations are noiseless synthetic heights, with broad coverage and
no occlusion. Do not claim reconstruction from photographs, robustness to noisy
sensors, recovery from extremely sparse constraints, extrapolation outside the
sampled domain, or physically meaningful terrain. Noise, clustered/sparser
samples are separate untested conditions unless prospectively added with their
own cohorts and gates. The UI can paint custom example heights; those runs are
marked `customSamples=true` and have no known held-out surface or utility gate.
Painting, fitting, exporting and stopping are interaction checks, not evidence of
general quality for arbitrary painted examples. No pretrained model or training
dataset download is needed.

## Execution limits and evidence

Use Windows, ordinary WebGPU and installed Three `0.185.1`; no new GPU features,
custom device limits, large models or training dependencies. Only one GPU task
may execute at a time. A probe has an external 45-second watchdog. Each
interactive training call has a 20-second wall limit and 160-update cap. Yield
between small update groups, stop on page hiding or user cancellation, and await
in-flight work before another trainer owns the device. Probe completion, failure
and cancellation release buffers, materials, geometries, renderer and device.
The interactive UI may retain the completed scene and learned weights allocated
for inspection while the GPU is idle; it has no ongoing render/training loop.
Restore, page hiding, cancellation and failure release that retained trainer.
Do not automatically retry device loss, increase the workload, or bury a timeout.

Record source/protocol SHA-256 hashes, exact Three version, adapter/browser,
fixture and seed, initial/sample/final arrays, compiler graph statistics,
gradient-error summaries, validation failures, completed/requested updates,
WebGPU errors, cleanup status and timing boundaries in an immutable timestamped
report. Preserve generated shaders or their hashes to establish what executed.
Retain partial/failed reports, and exclude changed-source cells from a cohort.

Current `completedWallMs` is CPU-observed completed training including queue
synchronization, progress callbacks and requested preview renders. It excludes
earlier compilation/reset and later final export. It is not active GPU time.
Compare only runs with equal preview/readback behavior and record method order.
Initial gradient and first-Adam-state validation warm both gradient paths and
the shared update; queue waits and JavaScript yields remain included. Report
remaining warmup/order limitations, not a general performance superiority claim.

## Entry commands

From the repository root, run these four probes sequentially, inspecting each
report before starting the next. Each probe compares both methods:

```powershell
node scripts/run-trainable-tsl.mjs --field=neural --seed=11
node scripts/run-trainable-tsl.mjs --field=islands --seed=11
node scripts/run-trainable-tsl.mjs --field=neural --seed=29
node scripts/run-trainable-tsl.mjs --field=islands --seed=29
```

UI checks are separate from the four-cell numerical matrix. The current UI uses
its own default initialization; do not treat a runner seed label as a separate
UI training replicate:

```powershell
node scripts/run-trainable-tsl.mjs --ui --field=neural
node scripts/run-trainable-tsl.mjs --ui --field=islands
```

For manual inspection, `node scripts/run-trainable-tsl.mjs --serve` starts a
loopback server on port 5192. Loading the page does not allocate a GPU trainer;
training begins on user action. The commands above are verified against runner
argument handling, not executed by the protocol author.
