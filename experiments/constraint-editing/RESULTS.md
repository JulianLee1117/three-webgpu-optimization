# Completed results — September 9, 2026

The native TSL editing path passed the fixed four-case GPU capability gate with
generated AD and same-GPU finite differences: **eight of eight fitted lanes**.
The original penalty-solver failure is included in the portable evidence.

## GPU cohort

The wave ribbon and constant-curvature tentacle each have eight parameters and
two target phases. One authored graph per cell was rebound for evaluation,
generated coordinate derivatives and actual material deformation. All lanes used
the identical constrained solver, without per-case settings or witness
initialization. [GPU-PROTOCOL.md](GPU-PROTOCOL.md) specifies the predeclared checks.

- Maximum target error: **7.20e-6** world units on nominal length-three objects.
- Minimum withheld-time obstacle clearance: **0.01964** world units, including
  the cross-section radius; the required minimum was 0.005.
- Withheld normalized velocity change: **1.02–6.40%**, below the 20% limit.
- Position, length, root, loop and Float32-export quality gates all passed.
- All **36 rendered silhouette pairs matched exactly**. The gate allowed small
  boundary-only differences, but none occurred. Foreground coverage was 217–326
  pixels at 256×192; lighting and hidden vertices are outside this check.
- Across 804 probe points, **38,592 AD/FD Jacobian component comparisons** and
  4,824 primal comparisons passed. Maximum AD error was 1.20e-6, same-GPU FD error
  4.85e-4, and primal error 1.07e-6 against independent numeric formulas, using
  the documented absolute-plus-relative tolerances.
- Initial GPU uploads were exact. JSON reload reproduced actual GPU positions
  and Jacobians exactly in all eight fitted lanes.

Per-case velocity NRMSE, with AD then FD:

- Ribbon, early target: **0.01180 / 0.01023**.
- Tentacle, early target: **0.06235 / 0.06171**.
- Ribbon, late target: **0.01086 / 0.01086**.
- Tentacle, late target: **0.04878 / 0.06404**.

All final normalized training violations were below 0.001. Three lanes met the
solver's convergence criterion; five reached its iteration limit while meeting
the declared feasibility and quality gates. Do not describe all eight as
optimizer-converged or globally optimal.

Completed fitting times ranged from 0.214 to 5.112 seconds. AD was faster in some
cases and slower in another; methods also took different numbers of evaluations.
These are single-run workflow times including CPU work and readbacks, not GPU
timestamps or evidence of a speedup.

Tested setup: Three 0.185.1, Chrome 152.0.7977.83, Windows, RTX 5070 Ti. Before/after
temperature samples for the four cells were 45–47°C; these are not continuous peak
measurements. No browser/GPU errors were recorded and each owned device was
disposed. No model download, driver change or special browser flags were used.

## Preserved failure and CPU follow-up

The first fixed penalty solver passed the tentacle but failed the ribbon:
velocity NRMSE was **0.6339**, above 0.20, despite reaching the target and clearing
the sphere. Its feasible witness passed. Independent scalar-cost derivative
checks near both final states agreed around 1e-9, so the result was not explained
by a derivative implementation error. The failure was retained without tuning.

A separately declared follow-up used a standard augmented Lagrangian with
explicit motion/shape inequalities and added two target phases before its first
run. All four CPU cases passed, including Float32 validation. Matched handle-only
controls penetrated the sphere by 0.0344–0.0460 units, demonstrating that the
obstacle term mattered. Method and budget both changed from the first stage;
this does not isolate the effect of one inequality.

This establishes a bounded implementation capability on two authored programs.
It does not establish new optimization mathematics, general editing of generated
code, continuous-time collision safety or a performance advantage.

## Evidence and verification

[The evidence directory](evidence/README.md) contains eight full compressed reports:
the failed CPU first stage, successful CPU follow-up, four final GPU cells and
two earlier GPU parity diagnostics. Reports preserve raw arrays, traces, shaders,
masks, source hashes and reloads. Final GPU reports also embed local runtime source
snapshots. Historical source revisions are distinguished from matching live code.

```sh
npm ci
node scripts/analyze-constraint-editing.mjs --self-test
node scripts/analyze-constrained-motion.mjs --self-test --require-capability
node scripts/analyze-constraint-editing.mjs --kind=constraint-editing-gpu-v2 --self-test --require-capability
```

These commands perform no GPU work. They recompute geometry, derivatives and
constraints, inspect masks, verify hashes and reject deliberately corrupted
evidence. The first command includes the failed first stage: a successful audit
is distinct from a successful research gate.

## Interactive checks

The demo opens idle, accepts pointer target edits, exports Float32 parameters,
resets and stops an active fit. The visibility handler is tested with an explicitly
simulated hidden property. Responsive UI checks are separate from the numerical
cohort.

An initial interaction check found a zero-width texture allocation during mobile
resize. The error handler released the GPU. The display now waits for a drawable
container before resizing. The complete interaction check then passed, including
a newly dragged ribbon target with 1.31% velocity change. This GUI-only correction
does not alter the frozen solver or numeric cohort. Both local UI reports remain
under `results/development/constraint-editing-ui/`.

The reusable core is graph rebinding and batched differentiation. The example's
constraints, reference samples and independent numeric oracles are fixture-specific.
Another program still needs suitable exposed parameters and a declared editing
problem. Authors do not need to write its backward shader.
