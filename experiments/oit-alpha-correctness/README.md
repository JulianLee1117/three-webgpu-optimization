# OIT foreground alpha in Three.js r186

**A reproducible alpha-output defect in the stock `OITPassNode`.** A half-transparent red plane over a transparent clear produces RGBA `[127, 0, 0, 0]`: its foreground coverage is missing from alpha. Ordinary blending produces `[127, 0, 0, 127]`.

The affected return expression mixes foreground and background RGB but forwards only the beauty pass's alpha. The proposed one-expression correction includes foreground coverage:

```js
// Stock alpha:
beautyNode.a
// Proposed alpha:
float(1).sub(revealageNode).add(beautyNode.a.mul(revealageNode))
```

This is ordinary Porter–Duff coverage composition, not a new transparency algorithm. The isolated HTTP server serves the stock module unchanged at its normal addon URL and a corrected copy at `/overlay/OITPassNode.js`. Its sole textual replacement is the final alpha expression. It never edits `node_modules`.

The pinned source is `three-r186@0.186.0`, [OITPassNode.js at r186](https://github.com/mrdoob/three.js/blob/r186/examples/jsm/tsl/display/OITPassNode.js), SHA-256 `06671bdb6572a0a900cc8f4d0809dc07f63b2ab2c9da645417aaab28fa9f0769`. The served correction hashes to `014dc925678cff395b96d7ac8b3e160b26d977926fd955f2796e36613031d0d3`.

## Retained result

Five fixtures compare ordinary blending, stock OIT, and corrected OIT:

- One 50% layer over transparent black: stock alpha **0**, expected/output correction **127**.
- Empty transparent scene: all lanes **0**.
- One 50% layer over opaque black: all lanes **255**.
- Two 50% layers over transparent black: stock **0**, correction **191**.
- One 50% layer over a 25% opaque black clear: stock **64**, correction **159**.

Numbers above are actual 8-bit alpha, including rounding. The independent coverage oracle is `1 - (1 - backgroundAlpha) * product(1 - layerOpacity)`, with a prospectively declared tolerance of two output bytes. All 15,360 output pixels were subsequently checked on CPU. Corrected RGBA is byte-for-byte identical to ordinary blending in these fixtures; RGB is byte-for-byte identical between stock and corrected OIT.

The GPU run used one disposable WebGPU renderer, 32×32 RGBA8 targets without MSAA, linear color, no tone mapping, and `RenderPipeline.outputColorTransform = false`. It made 15 fixture/lane renders, or 35 underlying renderer calls including OIT subpasses. No browser/GPU errors occurred; the device was explicitly destroyed. Recorded GPU temperature was 39°C before and after. These observations make no performance claim.

## Verify without a GPU

```sh
node scripts/analyze-oit-alpha.mjs --self-test
```

The verifier uses only Node built-ins. It verifies the manifest and gzip/report checksums, embedded source snapshots, exact correction, raw padded readbacks, packed pixels, controls, coverage math, and four deliberate corruptions. It does not pretend to rerun the GPU or independently recover every module from the recorded served-source hashes.

The compressed evidence contains the exact, unredacted final report: seven source snapshots, hashes for all 589 served modules, all raw padded readbacks and packed RGBA pixels, fixtures, errors, and cleanup records. Report SHA-256: `9e76d7618746129b2023e80a15bc57d874d9673f90649a31182a809c9194d7b7`. See [the manifest](evidence/manifest.json).

To reproduce the GPU experiment after installing repository dependencies and Chrome:

```sh
node scripts/check-oit-alpha.mjs
```

It has a 45-second watchdog and starts no persistent UI. Two earlier harness failures are recorded in the manifest and retained locally under `results/development/oit-alpha-correctness/`: an incorrect device-property check before any render, then an incorrect assumption about unpadded readback after one direct render. Neither produced an OIT result.

## Scope

This establishes an alpha defect and a correction for five simple, same-color plane/clear fixtures on one browser/device. It does not validate all OIT RGB compositing, mixed-color layer ordering, transmission, MSAA, HDR, depth intersections, arbitrary pass chains, export pipelines, or other backends/devices. The opaque-background control explains why the defect can remain invisible in an ordinary opaque canvas. No upstream submission is included.

The probe and verifier are MIT licensed. Retained Three.js source remains under its original MIT license; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
