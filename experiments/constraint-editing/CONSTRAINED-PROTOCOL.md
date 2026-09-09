# Explicitly constrained motion editing: prospective CPU stage two

Declared on 2026-09-09 before the first stage-two optimization run. Preserve every
stage-one source and report. Stage one failed its two-fixture gate because its
ribbon solution lost motion fidelity. An independent residual-gradient audit
found agreement with scalar-cost finite differences near 1e-9 at the retained
solutions. This follow-up addresses constrained optimization, not an AD bug.

## Fixed inputs and four cases

Use unchanged `programs.js`, program parameter bounds, obstacles, original states,
train/held-out samples and validation gates from `PROTOCOL.md`. The four cases,
in order, are original ribbon, original tentacle, late-target ribbon, late-target
tentacle. The new variants change only the target and its disclosed feasibility
witness. Both targets occur at t=.625, s=1. Their witnesses are:

- Ribbon: `[.30,.22,.08,.95,.14,0,.09,0]`.
- Tentacle: `[1,.13,.15,.03,.28,.10,.8,.06]`.

Targets are the independent numeric forward values at those witnesses. A CPU
preflight before optimization confirmed both witnesses satisfy the original
held-out gates: ribbon clearance .03056, velocity NRMSE .06862, position RMS .27635;
tentacle clearance .10746, velocity NRMSE .15154, position RMS .25238. These are
synthetic representability checks. The solver starts from the original parameters,
never the witnesses. Original witnesses are also revalidated. Do not change any
case after observing solver results.

## Standard constrained method

Use a Powell-Hestenes-Rockafellar augmented Lagrangian for equality constraints
c=0 and inequalities g<=0. Equality contribution is rho/2*(c+lambda/rho)^2;
inequality contribution is rho/2*max(0,g+lambda/rho)^2. Constants independent of
parameters may be omitted in inner minimization. After each outer round update
equality multipliers by lambda+=rho*c and inequality multipliers by
lambda=max(0,lambda+rho*g). This is an established constrained-optimization method,
not a new algorithm. The implementation is a small bounded prototype, not a
convergence guarantee or a globally optimal solution.

Explicit constraints are the three target coordinates (scaled by 1/.01), every
training segment's sphere-clearance deficit with .02 margin (scaled by 1/.02),
training velocity NRMSE<=.18, position RMS<=.40, and mean centerline length ratio
in [.8,1.2]. Velocity and position use squared normalized inequalities; length
bounds are scaled by .2. The .18/.40 training thresholds were specified before
the run as sampling margins for the unchanged held-out .20/.42 gates.

The base objective retains the previous position/velocity residuals and .01
parameter-change residuals. It therefore prefers motion preservation within the
feasible region, in addition to explicit limits. No semantic parameters are
locked. Parameter bounds project every trial, and both motion programs use the
same algorithm and settings.

Fixed budget: eight outer rounds, twelve inner damped-Gauss-Newton steps per
round, eight Armijo backtracking trials per inner step, initial damping .001,
step norm<=.25. Armijo coefficient is 1e-4; trial lengths are 1,1/2,...,1/128.
Accepted steps halve damping (minimum 1e-9); rejected steps multiply it by ten
(maximum 1e9). Initial rho is one. After updating multipliers, multiply rho by
five if maximum constraint violation has not halved relative to the preceding
outer round, capped at 1e6. No-descent at one inner step does not abort the outer
method. Early completion requires maximum normalized constraint violation<=.001
and projected Lagrangian gradient infinity norm<=.001. Keep the final iterate,
without selecting against held-out metrics.

The common matrix deadline is eighteen seconds for all optimizers combined;
validation and report writing have a two-second allowance. No GPU run, automatic
restart, warm start from a failed/future solution, per-case setting, or budget
escalation is allowed. Bounded numerical failure is retained as failure.

## Controls, reports and decision

Run the unchanged stage-one handle-only solver on each case, initialized exactly
as the constrained solver. Retain rigid-translation and original-state checks.
The two new targets prevent assessing the new method solely on known failed
cases; they are still within the same small authored family, not a generalization
claim over arbitrary procedural code. The method and iteration budget both differ
from stage one, so a pass cannot isolate the effect of one inequality alone.

All four constrained solutions must achieve maximum normalized training-constraint
violation<=.001 and pass every unchanged independent held-out gate; all four
handle-only controls must penetrate by >=.005. An iteration-limited result can pass
these feasibility gates without establishing stationarity or optimality. Preserve per-case
failures, raw sample arrays, final parameters/positions, all outer/inner trial
records, multipliers, source hashes and timing. Include Float32-rounded-parameter
validation as a separate prospective export check. No averaging of a failed
case into an overall pass is allowed. A positive CPU result permits a bounded
native-GPU derivative/render/export verification. It establishes neither GPU
performance nor novel constraint editing.

`solveConstrained(fixture,{evaluate,onProgress,shouldStop})` uses the same evaluator
contract as `solve` in the first stage. `fixtureVariant(kind,variant)` accepts
`original` or `late-target`. A GPU evaluator may replace only positions/Jacobians;
the residual construction and constrained solver remain shared.
