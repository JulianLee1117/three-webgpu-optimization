# Editable Three.js scene fitting

A working bounded prototype fits structural dimensions in an ordinary Three.js
scene and exports the resulting editable parameter values. The research aim is a
useful adapter for editable, including AI-generated, scene factories. The current
fixtures are three hand-authored families, not arbitrary generated code or unknown
objects recovered from photographs.

**September 9 result: the GPU least-squares candidate did not improve on the
strongest existing solver in any of nine tested cells, on either view.** Its GPU
calculation passed the recorded correctness checks. This is a functioning tool
with a negative solver comparison, not a new optimization algorithm, a measured
speedup or a demonstrated substantial research advance.

## What the prototype does

The toy rover, geometric robot and pavilion each expose eight bounded dimensions.
Every evaluation updates existing positions/scales; topology, material colors,
lights and cameras remain fixed. The scene stays ordinary editable Three geometry.
The target is rendered from a predeclared fixture vector. Solvers receive the same
seeded initial vector, normalized bounds and the image oracle, without the target
dimensions. The displayed target and saved evidence contain those dimensions for
reproduction, not as optimizer input.

Training uses RGB mean-square error at 128x128 from one camera. A second camera
evaluates the initial and final scenes without providing training feedback.
The GUI shows reference, initial and fitted geometry, offers the declared seeds,
and lets the user inspect the unseen camera and edit all eight sliders after a
fit. Saved parameter JSON reflects those edits and applies to the corresponding
scene factory; it does not reconstruct an arbitrary scene hierarchy.

The [V1 protocol](PROTOCOL.md) compares random search, coordinate pattern search
and a budget-truncated Powell-style direction search with bounded Brent lines.
The [V2 protocol](PROTOCOL-V2.md) retains those solvers and adds central finite
differences, GPU normal equations and bounded Levenberg-Marquardt. These are
standard numerical methods. The latter estimates derivatives through rendered
perturbations; it is not automatic differentiation through Three.js.

The image bridge uses private backend access,
`renderer.backend.get(renderTarget.texture).texture`, to obtain the native texture.
It is pinned to Three `0.185.1` and is not a stable public Three API. Existing
geometry and materials need no automatic-differentiation rewrite, but this does
not make every integration call public. A backend change can invalidate the
bridge; source identity and native-texture checks must fail rather than silently
claim compatibility with another Three version.

## Run locally

Use Node 22.12+ and installed Chrome with WebGPU. From the repository root:

```sh
npm ci
node scripts/run-scene-fit.mjs --serve
```

Open `http://127.0.0.1:5188/`, select a scene and method, then run the fit. The page
starts idle, runs one bounded fit and leaves a static comparison. Reload for
another fit. Close the tab and stop the localhost server with Ctrl+C when done.

The automated single-cell runner is:

```sh
node scripts/run-scene-fit.mjs --scene=robot --seed=11
```

Declared scenes are `rover`, `robot`, `pavilion`; seeds are `11`, `29`, `47`.
`CHROME_PATH` may point to a custom Chrome executable. The current runner is V2:
four methods, 96 training renders per method, serial GPU work and a 45-second
browser watchdog, with a pause between methods. There are at most 19 meshes per
scene, including the floor. No model downloads, custom device limits, driver
changes or enabling browser flags are required. Stop on failed parity, GPU errors
or device loss. Do not loop the matrix in the background.

CPU-only numerical self-checks and report analysis are separate:

```sh
node experiments/scene-fit/optimizers.mjs --self-test
node experiments/scene-fit/least-squares.mjs --self-test
node scripts/analyze-scene-fit.mjs --source-ref=1d42d9a --source-ref=494f052
```

The optional source refs identify the local V1 and V2 checkpoints for matching
historical source hashes after files change. Without them, the analyzer checks
current files and labels unmatched sources explicitly. A fresh clone may lack
local ignored reports or those checkpoints. Neither code self-tests nor report
analysis launches a GPU workload.

## Equal-render budget and validation

Every solver gets exactly 96 rendered training images, including its initial
point. For scalar methods this means 96 scalar queries. For V2 least squares:

```text
training renders = scalarQueries + 16 * equationBatches = 96
fitting readback bytes = 16 * scalarQueries + 176 * equationBatches
```

One equation batch renders minus/plus probes for all eight parameters, reusing
the already-rendered incumbent image. The GPU reduces the 36 independent entries
of J^T J and eight entries of J^T r; the CPU receives 44 floats and solves the
small damped system. Scalar losses use a padded 16-byte readback. There is no
whole-image readback during fitting. This accounting establishes an implementation
property; no CPU-equations timing comparison was performed.

Reference captures, initial/held-out checks, correctness readbacks and display
renders are outside the fitting budget. Their existence must not be confused with
96 total scene renders across the entire application. V2's initialization compares
all 64 expanded Hessian and eight gradient entries with CPU calculations using
`1e-6 + 1e-4 * abs(reference)` tolerance. The recorded browser checks passed.
Reports retain CPU vectors, the maximum difference and the pass assertion; they
do not retain the GPU vectors, so the analyzer cannot independently reconstruct
every component-wise comparison from these reports.

## V2 results: correct execution, no solver advantage

All nine V2 cells passed their recorded technical gates at one source identity.
GPU least squares won **0/9 training comparisons and 0/9 held-out comparisons**
against the lowest-error alternative among random, coordinate and Powell search.
The per-cell winners were:

- Rover: seed 11, Powell training / random held-out; seed 29, coordinate training /
  random held-out; seed 47, Powell on both views.
- Robot: Powell on both views for seeds 11, 29 and 47.
- Pavilion: coordinate on both views for seeds 11 and 47; seed 29, Powell training /
  coordinate held-out.

The stronger held-out baseline is chosen retrospectively per cell for reporting;
the optimizer never selects a fit using held-out loss. The analyzer retains every
method's actual errors, parameter vectors and candidate-versus-each-baseline
contrast, rather than relying on a pooled average. The result does not support
further claims that this least-squares configuration is the useful improvement.
It also does not establish that no other objective or method could work.

The verified local analysis is
`results/development/scene-fit-analysis/2026-09-09T17-57-35.608Z/summary.json`.
V2 reports run from `2026-09-09T17-49-15.542Z-robot-11` through
`2026-09-09T17-50-40.432Z-pavilion-47` under `results/development/scene-fit/`.
The summary identifies each input and checksum. Local result directories are
ignored artifacts and may be absent in a fresh clone.

## Preserved V1 result and failure

The original V1 cohort has eight passed cells and one failed cell. Rover seed 47
failed target-restoration self-loss validation before fitting; its retained report
is `2026-09-09T17-43-09.451Z-rover-47`. Among the eight passed cells, Powell beat
both scalar baselines on both views for the three robot seeds, with counterexamples
on rover and pavilion. There is no completed nine-cell V1 success claim.

A later diagnostic reproduced 42 changed byte channels after restoring the target;
fixed draw order (`renderer.sortObjects=false`) restored exact agreement. The
passing diagnostic at `2026-09-09T17-44-51.328Z-rover-47` has changed source and
does not replace the original failure. V2 reruns every solver with that fixed
order. V1's failed restoration also limits interpretation of its earlier objective
comparisons; they are retained as development evidence, not promoted after repair.

## Analyzer and evidence limits

[The analyzer](../../scripts/analyze-scene-fit.mjs) verifies report SHA256 sidecars,
source-manifest shapes, seeded initial vectors, bounds, method order, actual render
budgets, best-so-far trajectories, loss parity assertions and cleanup. V2 checks
each finite-difference probe and the exact readback equation. Sources are matched
against current files or optional historical Git blobs. Versions and source
identities form separate cohorts; failed and missing cells remain explicit.
Repeated same-source scene/seed attempts are rejected rather than silently choosing
one. Explicit report paths may select a documented cohort; incomplete coverage is
still reported. Each analysis writes a new timestamped directory without replacing
earlier summaries.

Do not compare the original V2 `serializedEvaluationMs` values: finite-difference
batches include deliberate 4 ms pacing pauses, while scalar queries exclude them.
The analyzer labels this field `legacySerializedEvaluationMsDoNotCompare`. Later
callback/timing-label repairs do not change or replace the completed matrix.

Hashes establish internal consistency, not independent authenticity. The original
runners hash sources before execution, not a complete before/after served-source
trace. Screenshots have no separate recorded checksum. Wall time includes browser,
JavaScript and pacing overhead and is descriptive, not GPU timing or throughput.
Nine small known-family fixtures do not establish general inverse rendering,
autonomous AI scene repair or cross-device performance.

The intended longer-term capability is to preserve editable scene structure while
fitting useful visual constraints. This prototype exposes the interface and its
limits; the negative comparison must remain visible. Relevant existing work is
linked in the protocols, including img2threejs screenshot refinement and g9jax
parameter fitting. No priority or new-algorithm claim is made.
