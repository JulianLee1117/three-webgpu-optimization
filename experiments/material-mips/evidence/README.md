# Material-mips retained evidence

Four complete JSON reports are retained as gzip archives. The manifest identifies
each original path, raw SHA-256, portable SHA-256, archive SHA-256 and byte size.
The original local reports were not edited. Compression is lossless relative to
the portable JSON bytes.

- `load-failure`: module URL/404 timeout before a GPU probe was available.
- `forward-parity`: 128 cases by four lights; all 1,536 stock pixel-component
  comparisons and the independent CPU checks passed. Maximum stock pixel
  difference was approximately `7.45e-8`. No hardware FD check existed yet.
- `hardware-fd`: the same 128 inputs and four lights, with three fixed FD steps.
  Stock forward parity still passed, but five of 1,521 finest-step smooth
  derivative checks failed; 15 boundary comparisons were separately excluded.
  All 4,608 h-series comparisons, perturbed parameters, raw RGBA, shaders, source
  hashes and failure details are retained. No WebGPU errors; context disposed.
- `cpu-utility`: eight predetermined texels per material, with all inputs,
  optimization traces and controls. Held-out MSE was 31.1% worse for `hammered`
  and 97.9% worse for `scored` than the strongest fixed control. Both failed.

The only sanitization replaces the absolute local workspace prefix in `/failure`
of `load-failure` and `hardware-fd` with `<workspace>/`. The failure message,
script name and line number remain. The manifest records the two original hashes,
two different portable hashes, replacement count and removed-prefix hashes.
Successful forward-parity and CPU-utility archives are byte-identical to their
original reports. A credential/private-path scan found no other removal needed;
loopback URLs and public source links are retained.

From the repository root, this command uses only Node built-ins and no GPU,
browser, network, installed Three package or model:

```powershell
node scripts/analyze-material-mips.mjs
```

It verifies hashes and archive sizes, compares the shared input bytes, recomputes
stock pixel and all hardware FD checks from raw outputs, checks recorded branch
classification and independently audits the five failures, and recomputes CPU
utility aggregates from per-texel metrics. Successful verification means the
retained **failures** and pass counts are internally reproducible; it does not
turn either failed gate into a pass.

With repository dependencies installed, add `--numeric` to recompute every CPU
control/learned train and held-out MSE, per-texel MSE and peak error from the
retained parameters and targets. It also evaluates ideal bilinear CPU differences
at the exact retained plus/minus Float32 parameters for each of the five failed
hardware cases. It does not retrain or run GPU code.

```powershell
node scripts/analyze-material-mips.mjs --numeric
```

[analysis.json](analysis.json) retains that numerical analysis. The numeric oracle
and upstream dependencies are checked against retained hashes before import. The
complete current `shading.js` hash matches the GPU report but differs from the
earlier CPU report's complete file hash; the analysis records both explicitly.
All retained CPU metrics still recompute within `1e-11 * (1 + abs(reference))`.
This is numeric consistency, not a claim that the historical complete source
files were identical. Source tables are preserved even where later documentation
or harness files changed. The verifier does not reconstruct every historical
source module, rebuild the targets from fine maps, or replay the optimizer.

The h-series supports a discrepancy between ideal-bilinear AD and observed
hardware-forward differences at the fixed tolerance. Small forward residuals
are amplified by the step denominator. Isolated hardware LUT samples were not
retained, so sampler quantization is a plausible explanation rather than an
established cause. See the [experiment README](../README.md) for scope and primary
precision references. No universal material gradient or useful mip-filtering
improvement is established.
