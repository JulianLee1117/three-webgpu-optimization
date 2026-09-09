# Direct the motion: constrained edits of native TSL programs

This experiment adds a reusable parameter-editing path to two procedural Three.js
animations. Author the forward node graph once, expose point/time/parameter inputs,
and reuse that graph for rendering, batched evaluation and generated parameter
derivatives. A small shared CPU solver changes eight parameters to reach a target,
clear a sphere at sampled times and retain the original motion.

The input is a restricted native Three.js **0.185.1** expression graph. There is no
shader translation, handwritten backward shader, rig, downloaded model or inference
service. The returned parameters can drive the original animation after fitting
ends. This is an integration experiment, not a new motion-editing or AD algorithm.
It does not evaluate AI-generated code or infer useful controls automatically.

The final four-case GPU gate passed with AD and same-GPU finite differences:
eight fitted lanes, exact agreement in 36 rendered comparisons, and 1.02–6.40%
motion change. The failed first solver and successful follow-up are both retained.
[Completed results](RESULTS.md).

## Run

From the repository root, with the pinned dependencies installed:

```sh
npm run demo:constraint-editing
```

Open the printed local URL in a WebGPU-capable browser. The page starts idle.
Open an experiment, move the mint target in the right-hand view, and apply the
edit. Some target positions are infeasible within the original parameterization;
inspect the resulting metrics. Export saves the parameter patch and constraints.
Playback is capped at 30 frames per second and stops after twelve seconds. Stop,
closing the page or hiding its tab releases the owned GPU resources.

The ribbon's longitudinal tip coordinate is fixed by its formula; dragging keeps
that coordinate fixed. The tentacle's reachable targets are also limited by its
parameter family. These are explicit example controls, not inferred semantics.

The demonstration uses an RTX 5070 Ti with ordinary Chrome during development.
Other devices are not benchmarked. The batch limit is 4,096 points and 16
parameters; the fixtures each use eight. Fitting includes CPU constraint assembly,
a CPU optimizer, GPU evaluations and GPU-to-CPU readbacks. It is not entirely a
GPU solver, and no performance advantage is claimed.

## One authored graph

`model.js` invokes a forward factory exactly once. `rebind.js` clones that returned
graph onto different exact input nodes while preserving shared subexpressions.
The original graph is left untouched. `gpu.js` binds it to sample/storage inputs
and generates its three coordinate Jacobians using the restricted reverse-mode
compiler in `../material-mips/autograd.js`. The finite-difference control rebinds
that same graph onto perturbed parameters. `display.js` binds it to actual mesh
positions and animation time for rendering.

```js
const graph = makeYourForwardGraph(point, time, parameters);
const evaluator = await createProgramEvaluator({
  root: graph, position: point, time, parameters
});
const result = await evaluator.evaluate(values, samples);
// samples: flat [x,y,z,time]; positions: [sample][xyz]
// Jacobians: [sample][xyz][parameter]
evaluator.dispose();
```

Exact designated inputs are opaque boundaries. Rebinding accepts supported pure
arithmetic/math/vector nodes and constants. Unknown/unbound inputs, explicit
mutable variables, side effects, arbitrary functions, textures, matrices and
unsupported operations reject. This does not ingest arbitrary `positionNode`
programs, entire materials or JavaScript files. Source-node representation is
private Three implementation detail and remains version-specific.

## What is checked

The two programs are a spatially weighted wave ribbon and a nonlinear
constant-curvature tentacle. They are distinct, authored synthetic fixtures,
with disclosed feasible-witness targets; they are not a user study or an
independent corpus of generated programs. Only a tip handle is tested.

Quality checks use independent numeric formulas at withheld motion times and
denser spatial sections. Collision checks measure sphere-to-segment distance
including a cross-section radius. They are sampled-time checks, **not continuous
collision guarantees**. Position retention, normalized velocity change, length,
anchors, loop closure and target error are reported separately.

The first penalty solver failed its two-program gate: the ribbon cleared the
obstacle and reached the target but lost too much motion. The tentacle passed.
The initial result is preserved. The separate constrained-solver follow-up and
its additional targets are specified in [CONSTRAINED-PROTOCOL.md](CONSTRAINED-PROTOCOL.md).
See [RESULTS.md](RESULTS.md) for completed outcomes and retained evidence.

Native GPU positions and AD Jacobians are compared against independent numeric
finite differences. The same-GPU finite-difference lane is a derivative control;
CPU versus GPU runtime is not an isolated AD performance comparison.

## Prior art and scope

Spacetime motion editing, procedural direct manipulation and shader AD all have
substantial prior art. [PRIOR_ART.md](PRIOR_ART.md) cites the closest work. The
narrow contribution being evaluated is a reusable bridge between supported
native Three node graphs, ordinary shader animation and constrained parameter
editing. A matching library was not identified in the scoped search; that is not
evidence of being first.
