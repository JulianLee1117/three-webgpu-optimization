# Closest-point triangle query selects the wrong edge

Confirmed correctness finding, September 9, 2026. This file records the geometric error, local correction and supporting evidence.

The current `three-mesh-bvh` TSL/WebGPU triangle helper can return a valid point on an obtuse triangle that is not its closest point. The answer can change when the same triangle's vertices are reordered. An isolated GPU dispatch reproduces the error without a BVH, deformation, refitting, or rendering. A thin, nondegenerate triangle makes the reported distance approximately **500 times too large**; the final local correction passes all 200 isolated regression cases. This ratio describes a distance error, not a speedup.

## Geometric counterexample

**Suggested title:** `WebGPU closestPointToTriangle can select the wrong edge at an obtuse corner`

Affected source: `three-mesh-bvh` 0.9.15, commit `8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab`, specifically [the edge-selection branches in src/webgpu/tsl/fns.js](https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/src/webgpu/tsl/fns.js#L33). `BVHComputeData` uses this helper in [getClosestPointToPointFn.js](https://github.com/gkjohnson/three-mesh-bvh/blob/8747a3c418f1dafa7c3ab1b3c1ffdecc531a2eab/src/webgpu/shapecastFns/getClosestPointToPointFn.js#L94). The three relevant upstream files were unchanged on `master` when checked on September 9.

Minimal example, with all coordinates in the XY plane:

```text
A = ( 0, 0, 0)
B = ( 2, 0, 0)
C = (-1, 1, 0)
P = ( 1,-2, 0)

Expected closest point: (1,0,0), barycentrics (0.5,0.5,0), distance squared 4
Actual closest point:   (0,0,0), barycentrics (1,0,0),     distance squared 5
```

The projected barycentric values in the helper's local variables are `u = -0.5`, `v = -2`, `w = 3.5`. Both `u` and `v` are negative, but the first `if` selects segment CA and prevents the following branch from considering AB. Projection onto CA clamps to A; the true minimum is the midpoint of AB. The inputs and intermediate values in this example are exactly representable in binary, so the discrepancy is not a small floating-point tolerance effect.

The same geometric triangle fails in 3 of its 6 vertex orders: `ABC`, `CAB`, and `CBA`. The other three orders return the expected point. A valid barycentric sum or an on-triangle test alone therefore cannot catch the error.

The error can be much larger relative to the true distance on thin triangles. Keep A and B unchanged, set `C = (-1, 0.001, 0)` and `P = (1, -0.002, 0)`, then upload the inputs as f32. The actual GPU result is again A, with squared distance `1.0000040531158447`; both CPU oracles find `(1,0,0)` with squared distance `0.0000040000003799796195`. The reported distance is about 1.000002 instead of 0.002, approximately a 500-fold overestimate. The corrected GPU result is `(1,0,0)` with squared distance `0.000004000000444648322`. This thin fixture also fails in the same 3 of 6 vertex orders in the original helper.

This is separate from [issue #914](https://github.com/gkjohnson/three-mesh-bvh/issues/914), resolved by [PR #915](https://github.com/gkjohnson/three-mesh-bvh/pull/915) on August 1, 2026. That change corrected the order of exported barycentric components in the WebGL and deprecated WGSL helpers. The current TSL helper already returns ABC weights and its caller reconstructs them correctly. The failure here concerns which point those weights select.

Run the CPU reproduction from this repository's root:

```sh
node experiments/gpu-deform-collider/triangle-query-counterexample.mjs --self-test
```

The [reproducer](triangle-query-counterexample.mjs) contains a literal JavaScript transcription of the pinned primitive, a separate plane-interior/three-segment reference, and a `Three.Triangle.closestPointToPoint` oracle. Its 24 named cases include all vertex orders and off-plane controls. The original fails 6 cases; the reference agrees with the expected points and weights. A deterministic 10,000-case CPU sweep finds 249 original mismatches and no reference mismatches, with maximum reference squared-distance error `3.552713678800501e-15`. These counts describe that fixture distribution, not the prevalence of failures in arbitrary scenes.

The isolated GPU reproduction uses the actual imported WGSL helper:

```sh
node scripts/run-gpu-deform-collider.mjs --triangle-probe
node scripts/run-gpu-deform-collider.mjs --triangle-probe --corrected
```

Each command now runs one bounded dispatch with 200 nondegenerate triangle/query cases: 72 analytic cases and 128 seeded random cases. The v2 probe adds thin triangles with heights 0.01 and 0.001 in every vertex order to the original 188-case set. It records actual input/output arrays, two separately implemented CPU answers, tolerances, the imported helper URL, and served module hashes. The harness closes the browser under a 45-second watchdog. `--corrected` uses an experiment-local HTTP source overlay; it does not modify the installed dependency. See [triangle-probe.js](triangle-probe.js) and the [runner](../../scripts/run-gpu-deform-collider.mjs).

The original GPU helper fails 10 of 200 cases: the three failing orders of the integer example, three orders of each thin example, and one random case. Both CPU oracles agree on all 200 cases. The final local correction uses the standard seven Voronoi regions for nondegenerate triangles, with a boundary-segment fallback for zero area. It passes 200/200 cases, including every order of all three counterexamples. Passing this finite probe does not establish correctness for arbitrary geometry or floating-point scales.

Suggested regression coverage includes obtuse triangles with two negative projected barycentric coordinates, all six vertex orders, face/edge/vertex regions, and points above and below the triangle plane. This is a standard closest-point geometry correction, not a new algorithm or a performance claim.

## Immutable evidence

The isolated reports use Chrome `152.0.7977.83`, Three.js `0.185.1`, and an NVIDIA Blackwell adapter. Within each original/corrected pair, fixture inputs, expected answers, tolerances, and probe source are identical; the only differing served module is `src/webgpu/tsl/fns.js`. Both v2 reports have no GPU errors or changed-source warnings. Report checksum sidecars were independently verified.

- [Original v2 GPU report](../../results/development/gpu-deform-collider/2026-09-09T19-18-49.653Z-wave-16-128/report.json): status `reproduced-upstream-failure`, 10/200 mismatches. SHA-256: `3121e489aea3c620ed5f49d3ce73b76f577642499e39c3cc413eddd37c05a4f7`.
- [Final correction v2 GPU report](../../results/development/gpu-deform-collider/2026-09-09T19-18-51.659Z-wave-16-128/report.json): 200/200 passed, maximum point error `3.6945515974648303e-7`, maximum squared-distance error `1.5002911233352734e-6`. SHA-256: `75c1b1c87a57dad99a2ff59d8137e56b64f403bd528d2754747925d15a265ce0`.
- Final correction served `fns.js` SHA-256: `4b81a1348d16857dc29639ea823c21c7ffa89e80eed03ea19ce28609d2cbe462`.
- Shared v2 `triangle-probe.js` SHA-256: `0060e749b0456d6161ad1e25c0083cd96d890eb42568ac6ad69a2251f02e854e`.

Earlier 188-case evidence is retained separately:

- [Original isolated GPU report](../../results/development/gpu-deform-collider/2026-09-09T19-15-33.654Z-wave-16-128/report.json): status `reproduced-upstream-failure`, 4 mismatches, maximum point error 1 and squared-distance error 1. SHA-256: `d33d8a1c368b295e17790642b6dff02f632e7197517182cb9463a597f6c488c2`.
- [Initial correction isolated GPU report](../../results/development/gpu-deform-collider/2026-09-09T19-15-35.848Z-wave-16-128/report.json): 188/188 passed, maximum point error `3.6945515974648303e-7`, maximum squared-distance error `1.5002911233352734e-6`. SHA-256: `b678c40855a1329f2bcaedddc4bfc2ae008112d7cc593800129cf8dee0c90539`.
- Original served `fns.js` SHA-256: `ddd53d39cca6b095df625448c17e43826087614fab2a56f308cf4c21cb2b7b66`.
- Initial correction served `fns.js` SHA-256: `145d2be9dd64ea2295281a89ac73ff3604c41925a0818aa622701cdcd5184dd7`.
- Shared `triangle-probe.js` SHA-256: `b02a9c3dfa51a53d8df57e7a0c229b369ebcfcd518b33b85c218897281a088ee`.
- [Final Voronoi correction on the original 188-case probe](../../results/development/gpu-deform-collider/2026-09-09T19-17-42.470Z-wave-16-128/report.json): 188/188 passed. SHA-256: `d2280460fd8e4133752545ad01b9b81bdc5b8d0f00ff35fefaf3b767fbc593db`.

The isolated GPU gates require point error at most `2e-4` and squared-distance error at most `2e-5 + 2e-5 * expectedDistanceSq`, finite output, and valid barycentric weights. Tiny and near-collinear random triangles are excluded; the v2 analytic cases explicitly include the two thin triangles described above. The minimal failure exceeds these tolerances by orders of magnitude.

The issue was discovered while testing GPU-deformed collider geometry. The [original twist report](../../results/development/gpu-deform-collider/2026-09-09T19-08-42.797Z-twist-32-64/report.json) has identical query failures under exact refitting and interval bounds, despite passing descendant-geometry bounds and metadata checks. At time 2.4, each method has two wrong distances and two returned points that are not closest on their own reported triangles. The isolated probe above supplies the simpler evidence that the primitive itself is incorrect.

The [first local correction attempt](../../results/development/gpu-deform-collider/2026-09-09T19-12-03.379Z-twist-32-64/report.json) had a barycentric sign error and failed; its report and proof remain untouched. The [subsequent corrected twist run](../../results/development/gpu-deform-collider/2026-09-09T19-12-20.360Z-twist-32-64/report.json) passed its distance/on-returned-triangle gates at times 0, 0.7, and 2.4 for both bounds methods. Its report SHA-256 is `4ca4190b2cfed06362e8ac7d9b27fbd32800584a400bef853ca5d2b842c8f6e4`; its proof SHA-256 is `06bc0b3a76b18c6c33281b28fb26fb9feb01da69a3d84e5e4bff0bfdacee4e15`. Both were verified. This run is supplementary integration evidence, not the final strict point-regression assessment: symmetric torus queries may have multiple equally close points, and the report's direct point-to-oracle difference is not its acceptance gate.

## Correction status and boundaries

The [review patch](../../patches/three-mesh-bvh-closest-point.patch) contains the final seven-region Voronoi correction and changes only `src/webgpu/tsl/fns.js`. It was generated from the frozen [local correction module](closest-point-correction.mjs), SHA-256 `bd635cfc5a5902fce26ed6f655619a2bae03a708653cf1d6183f9e10916159d4`. Patch SHA-256: `5b4ed0117362211565e3cb49567c454d0bece8ce34da30b0f418cd48f7fabc42`.

Patch application was checked in a temporary source copy with `git -c core.autocrlf=false apply --check`, then applied there. The resulting bytes exactly matched the final GPU-tested module hash above. The newline option prevents Windows Git settings from rewriting the verification copy's line endings. A maintainer can run the same `git apply --check` from the pinned upstream repository with the patch's absolute path.

The initial plane-interior/three-segment alternative passed the isolated v1 set but failed a stricter point-on-returned-triangle gate in subsequent [wave](../../results/development/gpu-deform-collider/2026-09-09T19-16-02.816Z-wave-32-64/report.json) and [ripple](../../results/development/gpu-deform-collider/2026-09-09T19-16-11.041Z-ripple-32-64/report.json) integration runs. Its squared-distance gates passed, while one query in each affected sample failed the point gate. Those failed reports are retained; their SHA-256 values are `ddd48613665d22d832a927ea50401eceeb116a9f0c9f0980c1b6b4db1f79f0e8` and `7e69dba509fa72b140e571c575886aaba23178a0bbd557c566e3c5f121cc4da5`. Selecting between nearly equal f32 squared segment distances can choose measurably different boundary points, so this version is not the final patch.

The final Voronoi version passed the same strict gates in the [wave follow-up](../../results/development/gpu-deform-collider/2026-09-09T19-17-44.481Z-wave-32-64/report.json), without widening tolerances. Report SHA-256: `2d75eb0bc764eb54a232f6ad97684b3808860aad2a7e6c741152409339348133`. Broader collider-matrix results are separate from this primitive finding. The CPU three-segment reference remains a separate reference implementation, not the final WGSL patch.

The installed library remains unchanged. The overlay changes only the primitive function in `src/webgpu/tsl/fns.js`; the upstream packed traversal, query caller, buffers, and fixtures remain in use.

The WebGL and deprecated WGSL helpers contain related source logic and merit separate review. No WebGL shader run or GLSL correction is claimed here. There is also no general proof for degenerate/near-degenerate triangles, arbitrary coordinate magnitudes, every GPU driver, continuous collision detection, or performance improvement. The confirmed contribution is a reproducible geometric correctness bug in the current TSL/WebGPU closest-point primitive.
