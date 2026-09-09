# Native TSL footprint filtering: fixed first GPU screen

Frozen before the first browser execution. This is a capability/correctness and
image-error screen, not a performance campaign. Three is pinned to 0.185.1.

## Question

Can an automatic transform of the original native color graph match its affine
pixel-and-shutter integral, including products with correlated phases? Does that
reduce error against the same integral compared with sampling at the pixel center
and with a finite supersampling budget?

## Fixed inputs and controls

`fixtures.mjs` contains three independently authored RGB programs: correlated
weave, squared-wave energy and three-wave interference. It separately implements
the original numeric products, hand-applied product-to-sum identities, a manual
filtered shader and direct-product Gauss-Legendre quadrature. The CPU screen has
24 boxes; 32×32×16 and 48×48×24 quadrature orders must agree with the independent
analytic integral within 2e-8. Inactive axes collapse to one sample.

`gpu-probe.js` freezes four camera-footprint/exposure cells for each program:
magnified, minified, anisotropic and combined shutter. The geometry is a full
96×64 quad with affine UVs. All twelve cells retain the same forward expressions.
The four lanes are:

- Point sampling of the original native color graph.
- The automatically expanded and integrated native graph.
- A separately authored exact product-to-sum filtered shader.
- 64 midpoint evaluations of the original graph, rebound to sample coordinates:
  8×8 spatial samples for zero shutter; 4×4×4 samples for positive shutter.

Supersampling is an approximation to the same box integral. It is a strong
finite-budget control, not a ground truth when the pattern exceeds its sample
rate. The manual exact filter is the analytical baseline; the compiler is not
expected to improve upon its mathematics. No temporal accumulation/TAA comparison
is made because its history/visibility target differs from this material integral.

## Measurements and gates

Retain all Float32 RGBA pixels, all independent reference pixels, actual WGSL,
source hashes/snapshots, settings, compiler term counts, error examples and device
cleanup. The first render calibrates UV orientation and linear output to 2e-6.
Every point-render RGB channel must match the original numeric function to 5e-4;
every compiled/manual RGB channel must match the box reference to 5e-4. Alpha must
match one to 2e-6. Any nonfinite output, device error or failed cleanup rejects.

Report RMSE and maximum absolute error for every cell and lane. A useful scoped
result additionally needs at least 2× lower RMSE than the point lane in every
minified, anisotropic and combined-shutter cell. Magnification must stay within
the absolute correctness tolerance. Report actual finite-supersampling results
without making a universal ordering or speed claim. No measured timing threshold
or benchmark is inferred from expression counts or total harness wall time.

## Execution bounds and interpretation

One ordinary Chrome device, no feature flags, at most 64 small renders, a
40-second device watchdog and a 45-second browser watchdog. Run sequentially,
preserve failed attempts, and stop when the complete prescribed matrix is done.
Pre/post `nvidia-smi` samples are context, not peak-temperature measurements.

The real-arithmetic identity is exact for constant amplitudes and affine phases.
Finite-precision shader results are measured. Nonlinear/perspective mappings in
the interactive folded-surface illustration are only local approximations and
are not covered by this affine accuracy result. Lighting, visibility, camera
motion during exposure, arbitrary materials and geometry antialiasing are outside
scope. All filtering mathematics has documented prior art.
