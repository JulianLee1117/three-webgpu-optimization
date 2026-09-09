# Retained footprint-filtering evidence

These gzip files contain the complete JSON records, including every Float32
RGBA pixel, CPU reference array, actual WGSL, source hashes and snapshots,
browser/GPU errors, watchdog state, disposal records and pre/post telemetry.
They are not reduced metric summaries.

From a fresh checkout with the pinned dependencies installed:

```sh
npm ci
node scripts/analyze-footprint-filtering.mjs --self-test
```

This command uses only the CPU. It verifies the primary report checksums and
captured sources, reconstructs every reference from the independent fixture,
checks all twelve numerical cells and nine separate image-error improvement
gates, and rejects three deliberately corrupted copies. It writes a new analysis
under `results/development/footprint-filtering-analysis/`; raw evidence is unchanged.

- `report.json.gz`: the complete final-source run at
  `2026-09-09T22-36-41.222Z`, with 12 passing cells and 49 renders. Both compiler
  and manual lanes satisfy the prospective 5e-4 absolute RGB tolerance. Every
  minified, anisotropic and combined-shutter cell passes the separate 2× RMSE
  improvement criterion against point sampling.
- `first-pass.json.gz`: the first complete successful matrix at
  `2026-09-09T22-32-26.078Z`. The later run follows an API shutter-input guard and
  an interactive UI stale-open fix; the supported filtering mathematics did
  not change. All 1,204,224 retained RGBA values, including calibration, and all
  442,368 reference RGB values are exactly identical between the two successful
  runs. Each retained per-cell quality record is also identical.
- `load-failure.json.gz`: the initial attempt at
  `2026-09-09T22-31-26.064Z`, retained as an excluded setup failure. A relative
  module URL requested `/gpu-probe.js` and returned 404 before the probe could
  be invoked. It contains no experiment GPU result. The only public-byte change
  replaces the local workspace prefix in `/failure` with `<workspace>`.

To audit the earlier successful run separately:

```sh
node scripts/analyze-footprint-filtering.mjs --report=experiments/footprint-filtering/evidence/first-pass.json.gz --self-test
```

`manifest.json` records original JSON, public JSON and gzip SHA256 hashes, source
changes between runs, and the exact failure-stack redaction. Each report also
has a `.sha256` sidecar for its decompressed public JSON. The manifest has its
own SHA256 sidecar. The successful reports preserve every original byte.

The live numeric fixture must match its captured hash; other snapshots may
describe historical source. Hashes establish byte consistency, not independent
proof of GPU execution. This is one browser/device cohort of affine RGB
trigonometric fixtures. It establishes neither general shader antialiasing nor
a speed advantage. Point sampling slightly outperforms the compiler in the
magnified interference cell; this result is retained rather than omitted.

`ui-report.json.gz` and `portable-ui-report.json.gz` retain the successful source
and packaged UI checks. They test controls, the real preview timeout, mobile
width, device release/reopen, a simulated visibility handler, and isolation of
a delayed failure from an obsolete initialization. They are separate UI records,
not additional affine numerical cohorts; the affine analyzer is not their runner.
Their complete-byte hashes are included in the same manifest.
