# Native GPU integration gate, stage two

Declared before the first GPU stage-two fit cohort. The separate four-case CPU
constrained-solver matrix passed on 2026-09-09. Its original failed penalty-solver
screen remains preserved. This gate changes only the evaluator to native WebGPU
expressions and checks rendering/export; it is not an optimizer comparison.

Run the original ribbon, original tentacle, late-target ribbon and late-target
tentacle in that order. Each cell has native AD and same-GPU central parameter
finite differences, h=.001. Method order is AD/FD, FD/AD, FD/AD, AD/FD respectively.
Both use the unchanged constrained solver, samples, initial parameters and bounds.
The first actual GPU parameter upload is retained and must equal the Float32
initial values. Each fit is capped at twelve seconds and the external browser
watchdog is forty-five seconds per cell. Only one GPU cell runs at a time.

Generate the authored forward graph exactly once per cell. Rendering, AD and
parameter finite differences rebind that graph. Independently validate 67 fixed
off-centerline points at the start and at both final states. Native positions
must agree with the numeric forward oracle to 2e-5+5e-5|reference|. Generated AD
Jacobian components must agree with independent central differences h=1e-5 to
1e-4+.002|reference|; the same-GPU FD control tolerance is .001+.005|reference|.
Record all points, exact Float32 inputs, predictions and full Jacobians. CPU
derivatives are recomputed from the independent forward formula, not the AD tape.

At the initial and both final parameter states, render the actual rebound
`material.positionNode` with live parameter storage and compare it to the
independent CPU formula baked into the identical Float32 rest geometry. Use
256×192 RGBA8, no MSAA, 192 sections, ten cross-section sides, a fixed oblique
orthographic camera and times 0,.25,.59375. White silhouettes on opaque black
must be nonempty and binary. Allow at most max(2,ceil(.01×foreground pixels))
changed pixels, all within one pixel of both silhouettes' boundaries. Retain
packed masks and exact output-byte classification. This tests visible geometry,
not shading, occluded vertices or continuous-time motion.

All eight fitted lanes must satisfy the same independent held-out quality gates
and normalized training constraint violation <=.001. Also validate Float32-rounded
parameters. JSON export/reload must reproduce actual GPU positions and Jacobians
exactly. Retain both arrays, rather than only a boolean result.

The reports preserve compiled shaders, served-source hashes, snapshots of local
runtime source files, solver traces, final outputs, adapter/browser identity,
resource disposal state and before/after GPU telemetry. Any browser/GPU error,
nonfinite output, incorrect start, failed parity or failed utility gate rejects
the cell. No cell may be averaged away; changing settings requires a new stage.

Completed wall times include JavaScript constraint assembly, queue waits and
readback. No GPU-time or AD speed advantage is claimed. These four disclosed
synthetic cases do not establish behavior on arbitrary generated programs.
