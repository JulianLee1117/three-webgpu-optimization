# Retained constraint-editing evidence

These complete CPU and GPU reports preserve both stages of the experiment. They contain
the original JSON bytes compressed with gzip; no report data or paths were redacted.
`manifest.json` records compressed and decompressed SHA256 checksums, byte counts,
report kinds, outcomes and exact source snapshots. `manifest.sha256` checks the
manifest itself. A negative research result remains included evidence.

- `cpu-v1.json.gz`: first fixed soft-penalty screen, **failed**. The ribbon cleared
  its obstacle but exceeded the motion-retention limit. Original JSON SHA256:
  `f57da1bf064596d130c872554b6f8ec0875cedf51861a4b38fa7076253400369`.
- `cpu-v2.json.gz`: separately declared constrained follow-up, **passed all four
  cases**, including two new targets. Original JSON SHA256:
  `69fe78ff3fa33b5af96a993812752884704fd2071b0606881830adc830f1cd21`.
- Four `2026-09-09T21-*.json.gz` reports: final GPU constrained cohort, two authored
  programs by two declared targets. Both AD and finite-difference optimizer lanes
  passed in each case. Each complete report includes initial/final numeric parity
  arrays, solver records, export/reload arrays, binary rendered masks, exact runtime
  source snapshots, served source hashes, error observations and sampled telemetry.
- Two `2026-09-09T20-59-*.json.gz` reports: initial GPU compute parity diagnostics.
  They predate renderer-mask checks and contain hashes but no historical UI source
  bytes. Their core compute/compiler/oracle files still match; the auditor names
  historical files that are hash-recorded only.

From a fresh repository clone with Node >=22.12:

```sh
npm ci
node scripts/analyze-constraint-editing.mjs --self-test
node scripts/analyze-constrained-motion.mjs --self-test --require-capability
node scripts/analyze-constraint-editing.mjs --kind=constraint-editing-gpu-v2 --self-test --require-capability
```

These commands verify retained data without running an optimizer, opening a
browser or requesting a GPU. The general auditor includes the failed v1 result;
its overall research gate remains false when v1 is included. The explicit GPU-v2
filter verifies the final cohort separately. The v1 auditor succeeds when it
correctly verifies the failed experiment; `--require-capability` rejects v1.
Each audit writes a new timestamped analysis under `results/development/` and
leaves the compressed evidence untouched.

The v2 audit rechecks fixtures, known feasible witnesses, original states,
rigid-translation controls, matched handle-only controls, final uniforms, all
held-out metrics, Float32 parameter export, constraints, multiplier updates and
iteration limits. It reconstructs every constrained trial proposal from retained
states and reevaluates its cost and acceptance decision. It does not select new
parameters or perform a fresh optimization. The shared analytic constraint chain
also underwent separate finite-difference checks before this run.

`source/cpu-v1/` and `source/cpu-v2/` contain the report-pinned source bytes, retaining
their original relative module paths. A snapshot is historical evidence, not an
assertion that later UI files were present during the CPU run. The v2 auditor
verifies these bytes and explicitly reports whether it used matching live sources
or a pinned snapshot.

GPU verification rebuilds every 67-point numeric reference and parameter Jacobian
from the pinned independent program, checks GPU AD and GPU finite differences
against those references, and checks exact Float32 uploads. It recomputes final
training diagnostics from the retained GPU positions and independently checks
held-out quality at both exported and Float32 parameter values. It verifies
iteration budgets, recorded Armijo decisions, lane resets and reload equality.
It does not reconstruct the full GPU optimizer trajectory: intermediate GPU
Jacobians and rejected candidate vectors were not retained.

The final rendering gate compares the live storage-backed native `positionNode`
with the independent CPU formula baked into the same rest tube. At each of three
fixed times, a 256 by 192 non-MSAA render retains two packed binary masks. The
prospectively declared allowance is at most max(2, ceil(1% of the larger foreground))
different pixels, each within one pixel of both silhouette boundaries; interior
mismatches fail. Empty renders fail. The auditor independently decodes these masks
and recomputes their comparison. One off-axis silhouette does not validate hidden
vertices, lighting or normals. Byte-validity histograms are retained observations;
the full RGBA buffers are not included.

GPU source snapshots are embedded verbatim in each final JSON. Their hashes are
verified separately from later live UI files. Numeric oracle and core compiler
files must still match the sources used by this auditor. No report bytes were
edited, and all original report sidecar checksums and gzip round trips were checked.

The successful CPU screen establishes feasibility for two authored programs and
four disclosed targets. It does not establish GPU performance, arbitrary shader
support, artistic quality, or continuous-time collision safety. The constrained
method and iteration budget both changed from v1; this comparison does not isolate
one change as the sole cause of the improved outcome.
