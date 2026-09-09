# Editable scene fitting: an integration hypothesis

Question: can a bounded fitter improve the structural parameters of ordinary
Three.js scenes from rendered reference images, while leaving the scene factory
and numeric parameters editable? This is a proposed workflow addition, not a
new optimizer, differentiable renderer or screenshot-refinement concept.

Three fixed procedural scene factories each expose eight bounded structural
parameters. Colors, lights and cameras stay fixed. The target is generated once
from declared fixture parameters; the optimizer receives only normalized bounds
and a rendering-loss callback, never target parameters or scene motion formulas.
Every candidate uses an ordinary Three scene. No custom differentiable renderer,
model download, external inference service or scene-specific optimizer is used.

Compare equal-budget random search, coordinate search and a standard direction-set
candidate. Each gets the same initial vector, 96 total image evaluations and fixed
random seeds 11, 29 and 47. Training uses a 128x128 fixed view. Evaluate the initial
and final states from a second, held-out camera that never supplies training loss.
Preserve per-trial trajectories, parameter vectors and both view errors. Validate
GPU RGB mean-square loss against a CPU pixel calculation outside fitting.

The practical gate is improvement over both simple baselines across scenes and
seeds, with held-out-view improvement as well as training-image improvement. A
single attractive fit is insufficient. Reject claims based only on color/exposure
changes, bespoke parameter knowledge, extra objective evaluations, or one-view
overfitting. There is no preregistered statistical superiority claim from nine
small synthetic cases. Wall time is descriptive; this is not a GPU timing study.

The GPU reduction transfers one padded 16-byte scalar result per evaluation.
This avoids full image readback during fitting, but is established GPU reduction,
not a novel algorithm. A correctness-only CPU image readback checks the loss.
Each browser cell is bounded to three 96-evaluation fits, low polygon counts,
128x128 offscreen images and a 45-second watchdog, with pauses between methods.
No unrestricted animation, custom GPU limits or enabling browser flags.

Prior art deliberately retained: [img2threejs](https://github.com/img2threejs/img2threejs)
already includes screenshot-guided refinement; [g9jax](https://srush.github.io/g9jax/)
solves editable parameters from output constraints; [jax-js](https://github.com/ekzhang/jax-js)
and [Slang](https://shader-slang.org/blog/2024/11/20/theres-a-lot-going-on-with-slang/)
already provide browser differentiation capabilities. The intended contribution
is a small adapter for existing Three scene factories with verified limits.
