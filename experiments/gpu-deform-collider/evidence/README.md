# Portable triangle-query evidence

These two gzip archives preserve the complete original JSON report bytes from
the isolated 200-case WebGPU triangle-query experiment on 2026-09-09. No fields
were removed or sanitized. Decompression reproduces the independently recorded
raw SHA-256 hashes in `manifest.json` exactly.

- `original-200-report.json.gz`: 10 mismatches among 200 cases: three ordinary
  obtuse-triangle vertex orders, six thin-triangle vertex orders and one random
  case. Original report status: `reproduced-upstream-failure`.
- `corrected-200-report.json.gz`: zero mismatches among the same 200 cases.
  Original report status: `passed`.

Each report contains 72 analytic cases and 128 seeded random cases, all inputs,
two CPU reference results per case, actual GPU outputs (1,600 f32 values), exact
tolerances, individual error classifications, adapter/browser information,
telemetry and source SHA-256 maps. Both CPU references agreed on every case.
All 18 recorded local source hashes agree between reports. Of the 67 served
source hashes, only `src/webgpu/tsl/fns.js` changes. The public source may have
evolved since these captures; historical hashes are retained, not asserted to
match the current checkout.

For `A=(0,0,0)`, `B=(2,0,0)`, `C=(-1,0.001,0)` and `P=(1,-0.002,0)`, the
original GPU result reports squared distance `1.0000040531158447`; the CPU
reference is approximately `0.000004`. That is approximately a 500-fold
**distance overestimate**, not a speedup. Three of six vertex orders fail for
that analytic triangle. The corrected GPU output is approximately `0.000004`.
The [finding and source correction](../TRIANGLE-QUERY-FINDING.md) explain the
wrong-edge selection and scope.

Run the verifier from the repository root with Node.js 22 or later:

```powershell
node experiments/gpu-deform-collider/evidence/verify.mjs
```

It uses only Node built-ins: no installation, browser, network, model download
or GPU. It checks the manifest and archive hashes, lossless raw hashes, all
packed GPU values, case counts, numerical mismatch classifications, identical
case inputs/tolerances, and the single served-source difference. A separate
double-precision closest-point calculation also checks each stored reference.
This verifies the archived evidence; it does not rerun the GPU experiment.

To inspect a decompressed report without changing any archive:

```powershell
node --input-type=module -e "import{readFileSync}from'node:fs';import{gunzipSync}from'node:zlib';process.stdout.write(gunzipSync(readFileSync('experiments/gpu-deform-collider/evidence/original-200-report.json.gz')))"
```

This is a single-dispatch, nondegenerate-triangle correctness probe on one
NVIDIA Blackwell adapter and browser build. It does not establish universal
correctness, behavior on every GPU, degenerate-triangle coverage, a GLSL fix,
collision-system correctness or performance. The inherited outer report fields
`family=wave`, `segments=16` and `queryCount=128` are runner settings; the actual
isolated case count is `triangleProbe.caseCount=200`. The preserved
`protocolComplete=false` indicates this was not the larger deformation benchmark.

The privacy review found no user home paths or credentials. Historical loopback
helper URLs, browser version and coarse GPU telemetry are deliberately retained
as provenance; those URLs are not live links or external endpoints. Gzip stores
no source filename or user directory. `manifest.sha256` covers the manifest;
the manifest covers the archives and companion files, excluding itself and its
checksum sidecar to avoid a circular hash. Hashes verify consistency with these
captured bytes, not external attestation of the GPU run.
