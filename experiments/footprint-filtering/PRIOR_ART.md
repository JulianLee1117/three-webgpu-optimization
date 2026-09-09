# Prior art and claim boundary

Source review date: 9 September 2026. This is a scoped integration experiment,
not a claim to invent procedural filtering, Fourier integration, or automatic
shader transformation.

- **Dorn, Barnes, Lawrence and Weimer, 2015 — Towards Automatic Band-Limited
  Procedural Shaders.** This directly investigates automatic analytic filtering,
  exact cases, and approximations for procedural shader programs. It already
  establishes the broader goal of transforming shaders to reduce aliasing.
  [Author-hosted paper](https://dornja.github.io/publications/pg2015.pdf).

- **Yang and Barnes, 2017 — Approximate Program Smoothing Using Mean-Variance
  Statistics, with Application to Procedural Shader Bandlimiting.** Their compiler
  models intermediate program values statistically and supports multiple smoothing
  strategies, including animated shaders. It is direct prior art for propagating
  filtering information through a program. A small exact Fourier subset should
  not be represented as a replacement for this broader system.
  [Paper and abstract](https://arxiv.org/abs/1706.01208).

- **Three.js — native derivative operations and authored filtering.** TSL already
  exposes `dFdx`, `dFdy` and `fwidth`. The current procedural wood example's grid
  helper explicitly computes `fwidth(coord)` and inserts `smoothstep` transitions.
  This is a concrete baseline for manually authored derivative-based filtering;
  automatic placement or graph integration is a different implementation task.
  [TSL reference](https://threejs.org/docs/pages/TSL.html),
  [exact example source](https://github.com/mrdoob/three.js/blob/dev/examples/webgpu_tsl_wood.html).
  The `dev` link is moving upstream source, not a version pin for experiment results.

The proposed distinction is consuming an existing native TSL trigonometric color
graph, retaining correlations when expanding its terms, and producing a joint
pixel-and-shutter box average within Three's node system. Exactness applies to
constant amplitudes and affine phases under the declared footprint model. GPU
derivative estimates, nonlinear mappings and finite-precision execution do not
inherit an unconditional exactness claim.

The source review does not prove that no equivalent native implementation exists.
Publication claims should describe the implemented API and measured behavior,
credit the established methods, and remain narrower than general procedural
antialiasing or geometry motion blur. No performance or quality result is claimed
before the experiment's retained measurements pass their declared checks.
