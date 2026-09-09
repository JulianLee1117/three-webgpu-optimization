# Native TSL training: bounded cohort results

Four field/seed cells passed, with both generated AD and GPU finite differences
in each cell. Every fit completed 160 Adam updates on the same 128 observations.
Held-out RMSE fell by **92.15–96.62%** across the eight fits. This establishes
trainability of the two selected native forward TSL graphs without a separately
authored derivative program. **No AD speed or quality advantage is established.**

The runtime used Three `0.185.1`, Chrome `152.0.7977.83`, and one NVIDIA Blackwell
adapter. This is one machine/session, two synthetic target families and two
initialization/data seeds. See [scope and prior art](PRIOR_ART.md) and the
[protocol](PROTOCOL.md) for the restrictions on the native graph interface.

## Reproduce the retained-data audit

From the repository root, using Node.js:

```sh
node scripts/analyze-trainable-tsl.mjs
```

This uses Node builtins only: no GPU, browser, model download or npm installation
is needed to audit the evidence. It reads the five complete compressed reports
in [evidence/](evidence/README.md), verifies gzip and decompressed JSON hashes
against [manifest.json](evidence/manifest.json), and writes a new timestamped
summary and SHA-256 sidecar under `results/development/trainable-tsl-analysis/`.
It never changes the reports. Local original reports and their `report.sha256`
sidecars are additionally checked when present; a fresh clone does not need them.

The analyzer verifies identical source and served-code hashes across the four
accepted cells, served/source agreement, and the current frozen `engine.js`,
`fields.js` and `autograd.js` hashes. Editing these runtime files invalidates this
cohort's source check and requires a new cohort or an explicit historical source
snapshot. The complete report includes generated shader text and source hashes.

## Learned field quality

The independent audit reconstructs the starting weights and all observations
from the declared seeds. It recomputes training and held-out metrics from the
retained final GPU weight vectors using a separate double-precision numeric
forward function. The held-out set is the fixed 961-point offset grid; training
never reads it.

- **Neural, seed 11:** held-out RMSE `0.2390429594` initially,
  `0.0154395570` after AD and `0.0154406473` after finite differences.
- **Neural, seed 29:** `0.2942043864` initially,
  `0.0099537136` after AD and `0.0099493488` after finite differences.
- **Islands, seed 11:** `0.2849986518` initially,
  `0.0223806961` after AD and `0.0223810517` after finite differences.
- **Islands, seed 29:** `0.2830571927` initially,
  `0.0211942059` after AD and `0.0211949345` after finite differences.

Every fit passes the declared `final held-out RMSE <= 0.2 * initial RMSE` gate.
The largest AD/FD final held-out RMSE difference is `4.365e-6`; its sign varies
across cells. Final maximum absolute height error is also retained and audited.
The targets are noiseless analytic functions closely related to the fitted
families. In particular, a small number of sinusoidal units can represent the
neural target. These results do not establish general reconstruction, robustness
to noisy/sparse observations, physical validity, or arbitrary game utility.

## Gradient and integration checks

Each cell retains the initial snapshot and both learned-weight snapshots. Across
these **12 snapshots**, the analyzer independently reconstructs the squared-loss
central-difference oracle with step `1e-5`, then checks all **59,136 AD** and
**59,136 GPU finite-difference** components against it. No component fails the
protocol's method-specific tolerance.

- Maximum AD absolute error: `4.9687333e-7`.
- Maximum GPU finite-difference absolute error: `5.6253649e-5`, using the fixed
  GPU perturbation `0.001`.
- Maximum difference between independently recomputed and retained CPU oracle
  entries: `8.3266727e-12`.
- All eight retained GPU starting-weight arrays exactly equal their declared
  Float32 initial weights, including the second lane after export/reload.
- All **616** retained first-step Adam state components pass comparison with
  independently recomputed mean reduction and Adam expectations. Expectations
  are checked from both retained GPU gradients and the independent CPU oracle.
- All eight original export/reload equality gates are reported as passing.
  Idle-on-load, empty GPU/browser error lists and disposed final states are also
  checked from every accepted report.

The neural graph records 49 scalar parameters, 142 native nodes and 139 scalar
tape operations; islands records 28 parameters, 127 nodes and 124 scalar tape
operations. Each cell retains 17 shader modules, including three compute shaders:
generated AD, finite differences of the same native forward graph, and the shared
reduction/Adam update. These graph statistics describe this implementation and
are not a computational speedup measurement.

There are two explicit retention limits. The original validation recorded a
maximum direct GPU prediction error, but omitted the prediction-buffer arrays;
its largest reported error is `2.4843667e-7`, below `2e-5`. Both fields have a
final additive bias, so the audit also reconstructs every prediction from the
retained bias gradient using `d(loss)/d(bias) = 2 * (prediction - target)`. This
independently checked reconstruction has the same maximum error, but is labeled
as a gradient-derived reconstruction rather than a direct readback. Likewise,
`reloadExact` records the original comparison of reloaded GPU weights and
gradients, while the second readback arrays themselves were not retained. The
analyzer verifies that recorded gate and does not claim to rerun that comparison.

## GPU control and timing limits

Both methods use the same native field, observations, Float32 initial weights,
full-batch mean reduction, Adam kernel, learning rate and 160-update budget.
Seed 11 runs AD then finite differences; seed 29 reverses the order. AD uses 128
gradient invocations per update; finite differences use `128 * parameterCount`,
with two perturbed forward evaluations per invocation. Invocation counts alone
are not evidence of a speedup.

Recorded completed-training wall times span `484.6–518.2 ms` for AD and
`485.2–510.4 ms` for finite differences. Descriptive medians are `504.7 ms` and
`502.1 ms`, respectively. These include queue synchronization and CPU scheduling;
they are not active GPU time, and this tiny matrix does not establish performance
equivalence or an AD efficiency advantage. It also does not compare optimized
implementations in other differentiation systems.

## Accepted cohort and excluded canary

The accepted complete reports are:

- `2026-09-09T20-10-03.419Z-neural-11`
- `2026-09-09T20-10-11.217Z-islands-11`
- `2026-09-09T20-10-18.800Z-neural-29`
- `2026-09-09T20-10-27.198Z-islands-29`

The earlier `2026-09-09T20-08-22.166Z-neural-11` canary remains included in the
evidence package with `included: false`. Its strict runner failed on a favicon
404. Later review also found that the reset image aliased a mutable upload array:
export/reload could therefore change the second lane's starting weights. The
final cohort uses a separate immutable reset image and retains an exact starting
weight check for each lane. The canary is not used for any quality or timing
comparison, even though it contains completed optimizer output.

The four successful reports are byte-for-byte copies of the original JSON after
decompression. The canary changes only the absolute workspace prefix in its
failure stack to `<workspace>`. Its original raw hash, public sanitized hash,
gzip hash and exact redaction field are recorded separately in the manifest.
No failed report or numerical array was replaced or omitted.
