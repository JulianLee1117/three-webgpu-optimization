# Local elasticity: evidence and open questions

Research checkpoint: September 9, 2026 (PDT); records after midnight use September
10 UTC. This is an integration feasibility result, not a claim of a new physical
method, calibrated material recovery, or established industry impact.

## Independent CPU correctness

The final short numerical record is
`results/development/local-elasticity-cpu/2026-09-10T00-22-30.451Z/report.json`.
SHA256: `a7df607f4df79d6b03d81fcf91a8804be3f4c796bf1c63b1da1f00a64ef2b558`.
The independent audit SHA256 is
`c1ae3a66d7f1b13513d7388e0bc03d3173b1c5ed8141deae6bc67caf56113783`.

Both external assets pass independent NumPy/LAPACK checks of the RKPM basis,
eigenspace, energy, forces, deformation gradients, and eight short implicit
steps. Maximum force discrepancy is 9.66e-15 and Hessian discrepancy against
independent gradient differences is 2.61e-9. Every step converges and decreases
the independently recomputed incremental objective. Independent projected
gradient checks use the declared 1e-6 convergence threshold. This is numerical
agreement on the same assumed point measure; it is not physical ground truth.
These short trajectories do not reach the floor. Separate unit tests exercise
floor contact. The nineteen CPU unit tests also cover singular/inverted material
gradients, rigid motion, mass rank, hard constraints, and invalid input.

## Longer interaction check: initial failure and corrected result

The interaction record is
`results/development/local-elasticity-interaction/2026-09-10T00-13-41.554Z/report.json`.
SHA256: `07a4a7c7589059f440bb7d6ed480732d14f7199efa91d578de288809f0581e99`.

Each asset is pulled through a spring for thirty 1/60-second steps, then released
for thirty steps. Both use the UI's 256 integration points, 48 centers, six fields,
assumed material, and exact three-point base constraint. An independent control
omits the first drag; another keeps dragging at the first release step. This
checks that input changes subsequent motion instead of playing a fixed sequence.

All 120 trajectory steps converge. Both input counterfactuals change the next
state on both assets. Quadrature states remain finite and feasible, and accepted
updates reduce the objective. At pulled and released endpoints, 5,000 held-out
source points per asset have positive deformation determinants; they have zero
overlap with the quadrature sample. Endpoint minima are 0.795 for the plant and
0.858 for Spot. This detects no local folding at those samples, not everywhere
in space or at every time.

One additional plant counterfactual reaches the cooperative 100 ms solver budget
after three accepted updates. Consequently **123 of 124 total solver calls
converge and the overall record remains `failed-gates`**. Its returned state is
feasible and has lower objective, but it is not certified converged. The full
failure is retained; there was no rerun to obtain a better timing outcome.
The complete CPU run took 3.31 seconds on this setup, not a portability benchmark.

An actual pointer-driven UI test subsequently exposed near-stationary line-search
failures: energy differences were below floating-point resolution while the
projected gradient remained just above 1e-6. A regression reproduces failure at
1.51e-6. The corrected line search allows a step only when its energy change and
predicted decrease fit a declared 64-epsilon component bound and its projected
gradient in the same constraint nullspace at least halves. The original
convergence threshold is unchanged. The regression takes a real Newton update
to 5.9e-13 instead of relabeling the old state as converged.

After that code change, the same interaction protocol was rerun without changing
inputs, sample points, forces, iteration budgets, or thresholds. All **124/124
calls converge** and all interaction gates pass. The corrected record is
`results/development/local-elasticity-interaction/2026-09-10T00-23-26.774Z/report.json`,
SHA256 `40160b3408120dde0a6b5723eab6d5c35426785993a9e53f27fe44ebcf5609b6`.
It took 3.26 seconds of CPU wall time on this setup. Both the failed and corrected
records are retained. No claim of universal solver convergence follows.

## Browser and GPU checks

The first source UI check of Spot prepared all 10,000 visual splats in 0.113 s
(0.023 s for the basis), then completed four converged physics steps and reset.
It used actual WebGPU rendering and produced no browser or worker errors. This
is one cold browser observation on Windows/Chrome 152/RTX 5070 Ti; the mechanical
preparation and solve run on the CPU. It is not comparable to FreeForm's much
denser preparation benchmarks, and is not a general speedup result.

The eight-case GPU deformation check passes with synthetic fields, actual Spot
fields, reset, and a partial-workgroup tail. Maximum center error is 3.15e-8 and
covariance error 1.02e-10 against independently computed values. Negative controls
detect omitted weight derivatives and stale covariance. Sorting, output padding,
and input guards are checked. The renderer's device was explicitly destroyed.
Report: `results/development/local-elasticity-gpu/2026-09-10T00-15-03.624Z-covariance/report.json`,
SHA256 `9b570476cd69ca41bbfbb5b8f25f97df7bb129294338d567e9b063439ac05564`.
A separate NumPy audit of the retained outputs also passes. An earlier harness
404 failed before renderer creation and remains recorded.

The actual phone capture has 113,648 splats and prepared locally in 0.868 s in
the source UI interaction run. A real mouse drag, release, and reset all worked.
This older run also exposed the three near-stationary solver failures described
above; its functional UI pass is not an all-steps-converged claim. Its record is
`results/development/local-elasticity/2026-09-10T00-17-34.677Z-plant-ui/report.json`.
Standalone-package and lifecycle validation have separate records so browser
function, numerical convergence, and disposal are not conflated.

## Final standalone package and lifecycle

The tested portable archive is `local-elasticity-2026-09-10T00-32-07.938Z.zip`
(12,289,980 bytes), SHA256
`f33f7164b78b5fc770fd607e71b81e61bcbc94a2b3c747f3912cbee3bab20336`.
Its manifest SHA256 is
`d3b6c3a9c01f4d363c9e9d46e9ede94b0f4358b1053a8aa1c068e970d4d2c2da`.
It includes a static local server, one isolated Three 0.186.0 runtime, a bundled
worker, the two original assets, source snapshots and numerical evidence.
No npm installation, Python, model download or simulation backend is needed to
run this archive. Node runs the static server; a WebGPU browser runs the app.

The exact package passed actual mouse grab/release, Spot file-input loading and
reset checks. All 42 plant and four Spot solver calls converged. Reset was also
clicked while a solve was pending; the queued reset stopped scheduling and
returned all 72 coordinates exactly to zero. Synthetic 255- and 150,001-splat
files were rejected before device creation. These boundary fixtures are not
additional real-asset evaluations. There were 60 render frames, no unexpected
browser errors and no external runtime requests. The report is
`results/development/local-elasticity-bundle/2026-09-10T00-32-17.950Z/report.json`,
SHA256 `cc584f06e90adedaed60255ed03cb1baebcd76a5310b4e7a917224f3d692859a`.
The earlier passing package record is retained separately; it did not exercise
Reset during an active solve and therefore did not detect that UI bug.

A separate four-case lifecycle check uses declared synthetic visibility events
and delayed real callbacks to exercise cancellation during fetch, renderer
initialization, and a pending solve. A fourth case injects an invalid gradient
after real renderer initialization and checks the resulting setup-error cleanup.
All cases pass: three sequential devices are destroyed, workers and timers are
released, and delayed replies do not resurrect the app. This is a targeted
instrumented check, not coverage of every browser shutdown behavior. The report
is `results/development/local-elasticity-lifecycle/2026-09-10T00-33-49.850Z/report.json`,
SHA256 `4865c3a02a4a6c217de39e9d97ce3811f4ca2447356d58760bf1f30eb65e6aae`.
The first lifecycle harness selected the wrong asset and remained inconclusive;
that failed record is retained alongside the corrected checks.

The tested archive is sealed. This repository subsequently adds the package's
own QA records, lifecycle evidence and preview image; those additions do not
change the tested runtime bytes.

## What would justify a larger contribution?

The local workflow removes a real deployment step relative to Kaolin's
server-driven tools and accepts raw splats where the inspected SoftGLB pipeline
uses triangle geometry. It has not yet established an advantage in physical
fidelity, robustness, or authoring time over the strongest available workflows.

The next meaningful tests are independently supplied captures; larger bases at
fixed quadrature, loading, and material; then higher quadrature resolution;
and comparison of preparation effort and interaction quality against an
explicit mesh-conversion baseline. Different devices and browsers also need
testing. Nonuniform visual point density, thin disconnected parts, self-contact,
and full-surface collision remain substantial limits. Calling two locally
interactive objects a general solution would exceed this evidence.

## Reproduce and inspect

The source scripts retain inputs, output states, diagnostics, and source hashes.
Selected exact gzip records and their manifest are under `evidence/`; no GPU is
required to inspect them or run the independent numerical verifier. See
`scripts/check-local-elasticity-cpu.mjs`, `reference.py`, and
`scripts/check-local-elasticity-interaction.mjs`. Python is used for the audit
only; the interactive app needs a browser and a static local server.
