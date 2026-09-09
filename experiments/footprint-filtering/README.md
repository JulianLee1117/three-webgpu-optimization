# Filtering a procedural TSL graph over a pixel and shutter

An experiment in transforming an existing native Three.js TSL color expression
into its average over a spatial pixel footprint and a shutter interval. The
intended benefit is reducing aliasing in procedural patterns without rewriting
each pattern's filtering formula.

**Working demo and verified native integration.** All twelve fixed GPU cases
pass. Across the nine minification/exposure cases, RGB error falls by over 99.99%
versus point sampling and is lower than the matched 64-sample control. The manual
analytical baseline also passes; no speedup or new filtering theory is claimed.
See the [complete comparison and limits](RESULTS.md).

![Signal Loom](demo.png)

From the repository root:

```sh
npm run demo:footprint-filtering
```

Open the printed localhost URL in a WebGPU-capable Chrome or Edge. The demo
starts idle; previews stop after twelve seconds at a maximum of 30 frames per
second. Hiding or closing the page releases the device. No model download,
external service or training is involved.

```sh
npm run test:footprint-filtering
node scripts/analyze-footprint-filtering.mjs --self-test
```

Those checks are CPU-only and use the included [portable evidence](evidence/README.md).
`npm run check:footprint-filtering` starts a separate bounded GPU reproduction;
run it only once at a time. `node scripts/package-footprint-filtering.mjs --zip`
creates a standalone local bundle and integrity verifier under `deliverables/`.

## API

`compileFootprint(root, { inputs, time, shutter })` returns `{ node, stats }`.
`root` is an existing native scalar or `vec2`/`vec3`/`vec4` color expression.
`inputs` names exact native spatial/parameter boundaries; `time` is a separate
optional scalar input. Uniform phase parameters are held fixed over the shutter.
`shutter` defaults to zero and is a duration in the same units as `time`.
It can be a finite nonnegative number, scalar float constant, or scalar float
uniform. Other shader expressions and vector shutters reject. The uniform's
initial value is checked; keeping subsequent updates finite and nonnegative is
the caller's responsibility. `node` can be assigned to a node material's color input.
Typed uniforms/attributes and supported exact swizzles make suitable boundaries;
convenience varying accessors such as `positionLocal` are not automatically
unwrapped into supported inputs. The example uses an explicitly typed attribute.

```js
import { attribute, uniform, sin, vec3 } from 'three/tsl';
import { compileFootprint, evaluateFootprint } from './compiler.js';

const coordinates = attribute('position', 'vec3');
const clock = uniform(0); // advance this value in seconds
const shutter = uniform(1 / 60);
const phase = coordinates.x.mul(20).add(clock.mul(6));
const wave = sin(phase);
const original = vec3(wave.mul(wave));
const filtered = compileFootprint(original, {
  inputs: [coordinates], time: clock, shutter
});
material.colorNode = filtered.node;
```

`evaluateFootprint(root, { values, inputs, time, shutter, dx, dy })` is a CPU
helper returning `{ value, stats }`. `values` is a `Map` from those exact native
inputs to numeric scalar/vector values. `dx` and `dy` describe input changes
across one full pixel, in `inputs` order; vector entries must match their input
width. Empty arrays mean zero spatial change. For the example above:

```js
const numeric = evaluateFootprint(original, {
  inputs: [coordinates], time: clock, shutter: 1 / 60,
  values: new Map([[coordinates, [0.2, 0, 0]], [clock, 0.5]]),
  dx: [[0.04, 0, 0]], dy: [[0, 0.04, 0]]
});
```

The CPU helper uses native AD for local phase derivatives. It checks finite
values, scalar time, and exact footprint widths, including unused declared
inputs. It does not emulate GPU derivative quads or GPU floating-point error.

Color operations are numeric constants, `+`, `-`, `*`, division by a nonzero
numeric constant, `mix`, scalar/vector construction and conversion, swizzles,
integer powers from zero to four, and `sin`/`cos` of supported pure scalar phases.
Variables outside phases are not supported amplitudes. General functions,
texture reads and mutable nodes reject. Nonlinear phases supported by the shared
AD compiler are accepted as local approximations; the caller must distinguish
them from the exact affine class. The compiler does not certify phase affinity.

Expansion is bounded to 2,048 color nodes, depth 128, 128 complex terms per
channel and harmonic degree 16; the phase AD compiler has its own bounds.
Expressions can exceed these limits even when algebraic simplification by a
more powerful compiler could reduce them. The source graph is not mutated.

## Mathematical scope

The target class is a finite trigonometric polynomial: sums and products of
sines/cosines with constant amplitudes and phases affine in designated spatial
coordinates and time. Expanding products into a shared set of Fourier terms
preserves relationships between repeated inputs. For example, averaging
`sin(x) * sin(x)` over many periods gives **0.5**; multiplying the two separately
averaged sine values would incorrectly give zero.

For one complex term, averaging an affine phase over independent, uniform pixel
and shutter coordinates multiplies its center value by
`sinc(dx / 2) * sinc(dy / 2) * sinc(dt / 2)`. Here `sinc(z) = sin(z) / z`, with
`sinc(0) = 1`, and `dx`, `dy`, `dt` are the phase changes across the full pixel
width, pixel height, and shutter duration. Summing the integrated terms gives the
joint box integral. This is an exact mathematical identity within that model.

That identity is distinct from a renderer accuracy guarantee. Shader arithmetic
has finite precision. A GPU footprint derived from `dFdx`/`dFdy` describes a local
screen-space approximation; perspective or nonlinear coordinate mappings need
not be affine over a pixel. A local temporal derivative likewise does not make
a nonlinear animation affine over the shutter. Those cases require separate
error measurements and must be labeled approximations.

## Limits and validation

This is not a general antialiasing system. It does not integrate visibility,
moving geometry, silhouettes, occlusion, lighting, or arbitrary material graphs.
Noise, discontinuities, nonlinear phase warps, and unsupported functions are
outside the exact contract. Unsupported graph operations reject; accepted
nonlinear phase expressions retain the approximation caveat above.

Validation should compare the transformed graph with independent numerical
integration, including correlated products, near-zero frequencies, cancellation,
and combined spatial/temporal footprints. Rendered quality and cost need their
own fixed test cases and strong filtering/supersampling baselines. Any final
results should distinguish exact-class fixtures from local-footprint examples.

Run the CPU tests without allocating a GPU:

```sh
node --test experiments/footprint-filtering/compiler.test.mjs
```

They independently integrate plain JavaScript formulas with numerical quadrature
and interpret the actual generated native graph. They cover correlated products,
beats, joint space/time integration, vector conversions, zero-width footprints,
immutability, invalid inputs and resource bounds. These tests do not establish
GPU rendering quality or performance.

The contribution under investigation is a reusable native TSL transformation.
The filtering mathematics has substantial [prior art](./PRIOR_ART.md).
