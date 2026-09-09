# Native Three material differentiation and mip feasibility

The native shading bridge matched stock Three rendered pixels, but its generated
ideal-bilinear derivatives failed five fixed checks against finite differences of
the actual hardware-sampled forward graph. The material mip-fitting application
also failed its separate CPU usefulness screen. This direction is closed with
those limitations preserved; it does not establish exact hardware gradients or
a useful filtering improvement.

`shading.js` expands the actual installed Three `0.185.1` isotropic metallic
direct-light functions, including GGX multiscatter and the unchanged 16-by-16
half-float DFG lookup. It binds normal, view and light inputs without rewriting
the upstream BRDF arithmetic. The hardware texture samples remain in the forward
expression. Generated derivatives use an explicit bilinear replacement with
locally constant selected texels; cell boundaries and clamp kinks are treated as
piecewise boundaries. This depends on private native node and `Fn` representation
details and is pinned to this version of Three.

The validated rendering scope is opaque metalness-one shading with linear F0
`(0.65,0.3,0.12)`, unit directional lighting, flat geometry with zero geometric
roughness, 128 normal-slope/roughness cases and four fixed lights. The camera is
orthographic, with view direction `(0,0,1)`. There are no shadows, IBL, visibility
changes, tone mapping, transmission or full-scene derivatives in this test.

## Preserved outcomes

- Initial module-load failure:
  `results/development/material-mips/2026-09-09T20-39-57.618Z-parity/report.json`.
  An inline relative module URL resolved against the server root and returned
  404. The harness timed out before the experiment became available; this was
  not a shader/GPU correctness result. The failed report remains preserved.
- First GPU parity pass:
  `results/development/material-mips/2026-09-09T20-44-24.973Z-parity/report.json`.
  Across 128 cases and four lights, the expanded compute forward matched actual
  stock `MeshStandardNodeMaterial` pixels with maximum absolute difference
  approximately `7.45e-8`. All independent CPU forward and gradient checks passed
  their fixed absolute-plus-relative tolerances. The largest gradient absolute
  discrepancy was approximately `0.1241` on a gradient near `-504`, about 0.025%
  relative error. Raw outputs, gradients, compiled shaders and source hashes are
  retained. This first report did not yet contain hardware-forward GPU finite
  differences.
- CPU material-mip screen:
  `results/development/material-mips-cpu/2026-09-09T20-38-10.401Z/report.json`.
  Against the strongest fixed filtering control, fitted held-out MSE was 31.1%
  worse for `hammered` and 97.9% worse for `scored`. Both failed the predeclared
  30% improvement gate despite lower training loss. See [PROTOCOL.md](PROTOCOL.md)
  for fixture definitions, baseline details and preserved metrics. A full GPU
  utility matrix is not justified by that negative result.
- Actual hardware-forward finite-difference gate: **failed**,
  `results/development/material-mips/2026-09-09T20-52-39.642Z-parity/report.json`.
  Stock-render and independent CPU checks still passed. Of 1,536 finest-step
  parameter comparisons, 1,521 were smooth-cell checks and 15 were classified as
  crossing or near a declared boundary. Five smooth checks failed, all roughness
  derivatives at stored roughness `0.7200000286102295`. There were no WebGPU
  errors and the context was disposed. All 4,608 comparisons across the three
  steps, their perturbed Float32 parameters, and both forward RGBA outputs remain
  available. The failed gate has not been rerun or relaxed.

The last GPU harness added a separate check against central finite differences
of the actual hardware-sampled forward graph, with fixed steps `0.002`, `0.001`
and `0.0005`. It retains every perturbed parameter vector and RGBA output.
The smallest-step smooth-cell comparisons use absolute tolerance `0.002` plus
`0.025 * abs(AD derivative)`. Crossed or near-boundary cells are reported
separately and are not silently counted as passes; larger steps are diagnostics.
The first parity report has not been altered or retroactively credited with
this check. Full portable reports and a CPU verifier are in [evidence](evidence/README.md).

## What the failed derivative gate tells us

For the five failed cases, the same-step CPU bilinear finite differences agree
with AD within `1.57e-7`; actual-forward finite differences differ by
`0.00229` to `0.00399`. These are genuine interior checks: all have minimum LUT
boundary distance `0.0007500052`, 150 times the declared guard. An independent
classifier audit that renormalizes the stored light and includes upper dot-product
clamp distances reaches the same result. The frozen general classifier omits
those upper-clamp margins and uses the stored light without renormalizing it;
that limits future near-aligned probes but does not explain these five failures.

All five have smaller diagnostic error at step `0.001`. Step `0.002` crosses the
roughness cell boundary at `0.71875`, so it is not an interior derivative test.
At step `0.0005`, forward residuals of a few millionths against the ideal CPU
expression are amplified by a denominator near `0.001`. The retained outputs
verify that arithmetic relationship; they do not isolate its hardware cause.

Finite texture-filter precision is a plausible contributor, alongside shader
arithmetic. [Vulkan's sampler limits](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html)
explicitly describe sub-texel snapping, and [Direct3D's filtering specification](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm)
specifies finite coordinate precision. These are platform context, not evidence
that this browser used a particular backend or precision. The current reports
do not retain isolated hardware LUT outputs or identify a measured filter-bit
count. No sampler-quantization diagnosis, GPU defect, or mathematical AD defect
has been established. A derivative of ideal bilinear interpolation must not be
presented as an exact derivative of this observed hardware program.

## Run locally

The CPU tests require the repository's installed dependencies and perform no GPU
work:

```powershell
node --test experiments/material-mips/shading.test.mjs experiments/material-mips/autograd.test.mjs
```

One GPU correctness probe, with an external 45-second watchdog:

```powershell
node scripts/run-material-mips.mjs
```

Run GPU probes serially. `node scripts/run-material-mips.mjs --serve` exposes the
idle harness on loopback port 5194. Source imports use one Three source-module
instance; mixing separately bundled and source TSL modules is outside this
validated setup. Reports are immutable timestamped JSON with checksum sidecars;
the ignored local `results/` directories are not included merely by committing
these source files. No pretrained weights, external service or large training
dependency is involved.

This work is a native integration and correctness experiment. Slang already
demonstrates [autodiff-fitted material mipmaps](https://developer.nvidia.com/blog/how-to-get-started-with-neural-shading-for-your-game-or-application/),
and [Toksvig filtering](https://developer.download.nvidia.com/assets/gamedev/docs/Mipmapping_Normal_Maps.pdf)
and [LEAN Mapping](https://userpages.cs.umbc.edu/olano/papers/lean/) establish the
normal-filtering problem and earlier solutions. No new filtering algorithm,
performance superiority or universal material differentiation is claimed.
