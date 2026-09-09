# Material mip response fitting: bounded protocol and CPU screen

The question is whether fitting three ordinary coarse material parameters per
texel can preserve a fine normal/roughness map's direct lighting response better
than practical filters, while using derivatives of the actual stock Three TSL
shading expression. Normal-map-aware roughness filtering is established work;
neither that concept nor material fitting is claimed as new.

The fixture, optimizer, subset and quality gate below were fixed in
`fixtures.mjs` and `check-material-mips.mjs` before the first CPU run. This prose
was completed after that run, whose negative outcome is recorded below. The GPU
section remains prospective. Freeze runtime, source hashes and any amendments
before a GPU cohort; do not replace failed reports or retune these CPU results.

## Fixed maps and representation

- Two deterministic procedural materials: `hammered` (crossed waves and diagonal
  detail) and `scored` (modulated directional machining ridges and cross ridges).
  Fine normals are analytic derivatives of the height formulas in `fixtures.mjs`;
  roughness varies spatially and is correlated with the microstructure. These are
  synthetic fixtures, not scanned materials or independent physical measurements.
- Fine map: `64 x 64`, with Float32 texels `(normal.x, normal.y, normal.z, roughness)`.
  Coarse map: `16 x 16`, one `4 x 4` fine-texel footprint per coarse texel.
- Each fitted coarse texel has `(slopeX, slopeY, roughness)`, decoded with
  `normalize(slopeX, slopeY, 1)`. Slopes are constrained to `[-1.5,1.5]`, perceptual
  roughness to `[0.08,1]`; storage uses a fourth padding component of zero.
- Metalness is one, constant linear F0 is `(0.65,0.3,0.12)`, light radiance is one,
  and geometric roughness is zero on a flat surface. Evaluate the stock isotropic
  direct metallic response including `NdotL` and the stock multiscatter/DFG path.
  Exclude IBL, shadows, visibility changes, tone mapping and color conversion.
  `shading.js` supplies the independent numeric transcription and actual upstream
  source bridge. Preserve the unchanged DFG table and identify its source/hash.

One isotropic lobe cannot generally reproduce an anisotropic or multimodal mixture
of fine normals and roughness values. The representational limit is part of the
experiment, especially for directional machining. No claim about complete PBR
materials, arbitrary lights or physical reconstruction follows from this test.

## Observations and held-out directions

Every training target is the arithmetic mean of **all 16** fine-texel linear RGB
responses under the same light/view pair. It is rounded once to Float32 for the
shared target buffer. Target order is texel-major then direction-minor, with
`vec4(R,G,B,1)` per observation. Fine texels receive equal area weights.

Use 32 training pairs and 64 held-out pairs from the fixed Halton construction in
`makeDirections`. Training indices are `1..32`; held-out indices are `65..128`.
Radical-inverse bases are `(2,3)` for light elevation/azimuth and `(5,7)` for view.
Both vectors have positive Z in `[0.45,1]` before Float32 rounding. The code checks
there is no identical pair in both sets. Held-out responses never select weights,
initialization, a grid candidate, an iteration count or a hyperparameter.

This target represents a single fixed box footprint. It does not yet validate
hardware bilinear/trilinear sampling, anisotropic footprints, complete mip chains,
transitions between mip levels, moving-camera shimmer or surface visibility.
Those require additional experiments rather than inference from this matrix.

## Filtering controls

All controls use the same fine bytes and coarse representation:

1. **Average:** average the 16 decoded unit normals and normalize the result;
   average the authored perceptual roughness values. Store the result as Float32.
2. **Variance filter:** keep the average-normal direction and apply the practical
   vMF roughness-channel formula used by Filament's `roughness-prefilter`. For
   mean-normal length `r`, `kappa = (3*r-r^3)/(1-r^2)` and variance is
   `0.25/kappa`; filtered roughness is
   `sqrt(meanRoughness^2 + min(2*variance,0.2^2))`. Use zero added variance at `r=1`.
   The fixed threshold is `0.2`, without calibration to these fixtures. This is
   the source's channel-filter convention, not an exact GGX convolution theorem.
   [Filament primary implementation, lines 184–217](https://github.com/google/filament/blob/main/tools/roughness-prefilter/src/main.cpp#L184-L217),
   checked 2026-09-09. That source explicitly corrects a factor in the earlier
   vMF formulation; the experiment uses its corrected `0.25/kappa` variance.
3. **Roughness-only search:** hold each average-normal direction fixed and search
   48 uniformly spaced, Float32 roughness values including `0.08` and `1`. Select
   the minimum **training** linear RGB MSE independently per coarse texel; the
   first candidate wins exact ties. This is a stronger, objective-matched control
   than a fixed variance formula. Retain all chosen indices and output bytes.

The optimizer starts from this roughness-only search result, so any additional
benefit must come from optimizing the normal slopes and refining roughness.
Both GPU optimizer lanes must receive exactly the same initialization and zero
Adam moments.

Relevant prior art includes [Toksvig's normal-variation filtering](https://developer.download.nvidia.com/whitepapers/2006/Mipmapping_Normal_Maps.pdf),
[LEAN mapping](https://userpages.cs.umbc.edu/olano/papers/lean/), and Three's own
historical [r121 RoughnessMipmapper](https://github.com/mrdoob/three.js/blob/r121/examples/jsm/utils/RoughnessMipmapper.js).
This experiment must not compare only with average roughness and characterize
normal-aware mip filtering as an absent capability.

## Independent CPU feasibility check

Run `node scripts/check-material-mips.mjs`. It performs no GPU execution and uses
no AD traversal. The eight predetermined coarse indices are generated by the
seeded shuffle `0x6d697073`: `56,67,89,96,99,192,195,223`. These locations are fixed
before inspecting target responses or results and are identical for both maps.

Each texel receives 128 full-batch Adam updates using double-precision central
finite differences of the independent numeric shader, step `1e-4`. The objective
is mean squared error over all 32 training directions and three RGB channels.
Adam uses learning rate `0.02`, betas `(0.9,0.999)` and epsilon `1e-8`, followed by
the parameter clamps above. Evaluate the final iteration; no checkpoint is chosen
using held-out data. A ten-second wall cap applies to each fixture. Preserve all
parameters, targets, direction pairs, baselines, traces, failures and source hashes
in a timestamped report and SHA-256 sidecar; do not retry with adjusted settings.

For each material, the predeclared capability gate is **at least 30% lower
held-out linear RGB MSE than the strongest of the three controls**. Here strongest
means the lowest aggregate held-out MSE among those fixed controls; this is a
conservative evaluation comparison, not a choice fed to the optimizer. Both
materials must pass separately. Record per-texel MSE and peak absolute RGB error;
peak error has no predeclared acceptance threshold, so any regression must remain
visible. A lower MSE alone does not establish improvement at every direction.

### First CPU outcome: failed

The unmodified first run is
`results/development/material-mips-cpu/2026-09-09T20-38-10.401Z/report.json`, SHA-256
`048d849e153d036c0f1dff59974d13415bf242a47e2689a322d9c9dc7b2fa72b`.
Source hashes remained unchanged during execution. Each fixture finished in about
half a second; neither hit its wall cap.

- `hammered`: strongest control was roughness-only search, held-out MSE
  `0.0164968`; fitted parameters reached `0.0216281`, **31.1% worse**. Training MSE
  fell from `0.00888291` to `0.00551707`. Peak held-out absolute error decreased
  from `1.90663` to `1.71675`.
- `scored`: roughness-only search had held-out MSE `0.0193985`; fitting reached
  `0.0383955`, **97.9% worse**. Training MSE fell from `0.151474` to `0.0163086`.
  Peak held-out absolute error increased from `2.98037` to `4.88021`.

The optimizer descended on its training objective, but additional parameter
freedom did not generalize across the held-out lighting set. This is a negative
feasibility result; the small subset does not prove that every material or fitting
strategy must fail. A denser direction set, normal regularization or a richer
material representation would be a new prospectively defined study, not a fix
that may silently replace this run. A full GPU utility matrix is not justified by
this screen alone.

## Subsequent GPU correctness outcome and unrun control gates

The isolated stock-forward parity probe subsequently passed, but the later
hardware-forward FD probe failed five of 1,521 smooth finest-step checks; another
15 comparisons were excluded by the predeclared boundary classification. See
the [README](README.md) and [portable evidence](evidence/README.md) for the complete
reports, fixed steps/tolerances and CPU reanalysis. The first pass did not test
hardware finite differences. The generated LUT derivative is an ideal-bilinear
surrogate and has not passed the exact hardware-forward claim. No tolerances or
workload were changed to replace the failure, and no GPU utility training followed.

The remaining optimization/control requirements below describe what a separately
justified study would have needed; they are not completed checks or an active run
plan. This experiment is closed at the failed hardware-FD and CPU utility gates.

If a separately justified follow-up proceeds, first freeze its declared workload
and changes. Run one GPU task at a time, at most 256 coarse texels by 32 direction
pairs, with 128 updates initially and an external 45-second watchdog. Do not
automatically escalate after a failure. Release all owned buffers, textures,
renderer and device after completion, cancellation or page hiding.

1. Verify the adapted native forward graph against the independent numeric shader
   and an actual stock Three material render under matched direct lighting,
   color/roughness/normal conventions and linear output. Keep raw predictions,
   input normals, directions and roughness, not only pass booleans. Record source
   and shader hashes. A proxy BRDF with different stock arithmetic cannot validate
   this premise.
2. Check generated AD gradients against independent central differences away
   from nonsmooth clamp points and DFG cell boundaries. Declare tolerances and
   boundary handling before execution. Preserve boundary cases separately; do
   not treat a zero gradient through a texture lookup as an exact derivative.
   The DFG rule is the local derivative of the unchanged bilinear table within
   each cell, with explicit boundary limitations.
3. Check the actual reduced gradient and the complete first Adam state against
   an independent CPU mean and update. Retain each lane's exact starting weights
   and moment state; a reset must not alias mutable uploaded or exported data.
4. Compare AD with central finite differences of the same GPU forward shading
   program using the same observations, starting bytes, reduction, Adam updates,
   parameter bounds and completed iteration count. This isolates differentiation
   method; a CPU implementation is not the primary performance control.
5. Evaluate complete maps under all 64 held-out pairs and apply the two-material
   gate above. Retain each control's bytes and every failure. Export ordinary
   normal/roughness data and verify a stock-material reload from that data.

Record complete pipeline wall time and optional properly scoped GPU timings.
Shader invocation counts, compilation time and CPU queue synchronization must
not be conflated with active kernel time. The scope is quality at a fixed coarse
representation; no speed advantage over other filters or differentiation systems
is established by the current CPU screen.
