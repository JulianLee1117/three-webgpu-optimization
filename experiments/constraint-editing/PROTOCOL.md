# Constraint editing of two native TSL animations

Prospective CPU screen, fixed before its first optimization run on 2026-09-09.
The question is whether a small adapter can update a running Three deformation's
uniforms to satisfy a handle and avoid an obstacle while preserving its motion.
It is not a new optimizer, physics method, or motion-editing algorithm.

## Frozen programs and fixtures

`programs.js` contains two independently written eight-parameter native TSL
forward programs and separate numeric oracles. The ribbon combines spatially
weighted traveling waves; the tentacle uses nonlinear constant-curvature bending,
including `sin(s*theta)/theta`, plus an out-of-plane wave. Parameter bounds ensure
theta >= .28, so no near-zero division is permitted. Parameter counts, coefficients,
initial values, witnesses, bounds and sphere coordinates are fixed in `fixtures.mjs`.

Material coordinate s spans [0,1], nominal length is 3, and cross-section offsets
must fit radius .025. No topology change or arbitrary existing shader ingestion
is tested. Both time functions have period one. A target at the tip at t=.25 is
generated from a disclosed feasible-witness parameter vector. A witness establishes
representability, not a recovered ground truth or natural user dataset. Validate
it independently before optimizing; if invalid, retain that failed screen and stop.
The solver receives no witness parameters.

The initial ribbon collides with a sphere near its middle; a tip-only edit should
still cross that sphere. The tentacle also requires a nonlinear bend adjustment
to hit its target. Both must maintain their root and complete looping motion.
No fixture, target, solver settings or thresholds may be tuned after the first
run to force a pass. Any follow-up requires a new prospective protocol/report.

## Samples, controls and solver

Fit all 32 centerline segments at 16 times t=i/16, with 33 equally spaced spatial
sections. Fit position retention at those same 528 points. Velocity retention uses
nine sections and eight times (i+.37)/8, with symmetric temporal difference h=1/1024.
The handle is evaluated exactly at s=1,t=.25. Collisions use exact sphere-to-segment
distance with the .025 cross-section radius and an additional .02 training margin.
Segment derivatives use the closest point's barycentric weights; the derivative
of those weights vanishes from the distance derivative away from singularities.

The common bounded damped Gauss-Newton solver uses forward position Jacobians,
central parameter differences h=1e-4 for the CPU evaluator, initial damping .001,
48 iterations, at most six trial steps per iteration, and step norm <= .25.
Damping is multiplied by ten on rejection and divided by three on acceptance.
Parameter limits project each candidate. The initial parameters are identical.
The wall limit is ten seconds per solve, with no automatic budget escalation.
Residuals are: 100 times handle displacement, 100 times positive clearance deficit,
position displacement weighted .5/sqrt(point count), normalized velocity difference
weighted .5/sqrt(velocity count), and parameter changes weighted .01. The last
accepted iterate is used; no held-out checkpoint selection occurs.

Controls are the unedited animation, a rigid translation that exactly hits the
handle (including its root-motion failure), and the identical solver with obstacle
residuals disabled. The matched handle-only solver retains the same motion/shape
and parameter terms. It is the meaningful small baseline; rigid translation alone
would be weak. A later GPU AD lane must use this same solver and residual code,
replacing only the position/Jacobian evaluator. CPU FD versus GPU AD is a workflow
comparison, not an isolated AD speed comparison. Any AD speed claim requires a
same-GPU FD evaluator and completed end-to-end timing.

## Independent gates and kill criteria

Validate 128 unseen times (i+.5)/128 with 129 spatial sections. Each sphere-to-segment
distance is exact for the dense centerline polyline. This enclosing capsule covers
a mesh formed by interpolating the corresponding ring vertices, provided all
cross sections obey radius .025. This is sampled-time evidence, not continuous
collision detection. Velocity checks use 33 sections, 64 different times
(i+.31)/64 and h=1/4096. Check loop endpoints separately.

Both fixtures individually must have finite outputs, target error <= .01, minimum
held-out clearance >= .005, position RMS displacement <= .42, velocity error
normalized by original RMS velocity <= .20, mean length ratio in [.8,1.2], and
root/loop error below 1e-8 in the independent double-precision CPU oracle. To show
the obstacle constraint is necessary, the matched handle-only control must
penetrate by at least .005. Keep every per-fixture metric: averaging away one
failure is forbidden. Failure of either witness, final quality gates, or baseline
necessity kills this first capability gate. Passing justifies a bounded GPU
integration check, not generalized editability, artistic-quality or physics claims.

## Prospective native GPU integration

Reconstruct or explicitly bind the supported native graph; do not maintain a
hand-written backward shader. Render the very compute-written positions used by
constraints, and independently compare actual GPU positions/Jacobians against
the numeric oracle. Preserve sampled arrays, initial/final uniforms, actual
export/reload outputs and source hashes. A rendered triangle handle must interpolate
the three deformed vertices; evaluating deformation at an interpolated rest point
is generally different. Only final uniforms are needed for ordinary subsequent
animation: fitting must not become a per-frame dependency. No GPU work is part
of this CPU screen; use one device and a <=45-second bounded cell if approved.

## Prior art and practical boundary

[Gleicher's 1997 spacetime motion editor](https://graphics.cs.wisc.edu/Papers/1997/Gle97a/SpacetimeEditing.pdf)
already edits existing animation with constraints while preserving original motion.
[Riso et al. 2024](https://eliemichel.github.io/SdfManipulation) already use automatic
differentiation to infer procedural parameter edits from viewport manipulation.
[Differentiable CAD, 2021](https://arxiv.org/abs/2110.01182) also couples direct edits,
constraints and procedural program parameters. [GLSL Autodiff, 2021](https://www.davepagurek.com/programming/glsl-autodiff/)
already differentiates shader deformations from a JavaScript graph. The practical
opportunity is integration with supported native Three TSL programs and reusable
uniform patches. No new mathematical capability or first browser implementation
is claimed. This test does not evaluate AI-generated programs or arbitrary rigs.
