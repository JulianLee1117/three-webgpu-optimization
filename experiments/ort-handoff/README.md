# Strongest-baseline result: no new handoff performance advantage

The September 9 screen does **not** support a new performance claim for the
external-buffer patch. Stock three.js 0.185.1 can expose its own allocated storage
buffer through `renderer.backend.get(attribute).buffer`, allowing ORT's prebound
output to use the same buffer that Three renders. This is private backend access;
the proposed public API still has an ownership and integration rationale.

The four-lane screen used the same patched Three build for every lane. At 16,384
vertices, conventional median serialized update latencies were 2.970 ms borrowed,
2.980 ms internal, 2.990 ms GPU-copy and 6.500 ms CPU-copy. At 65,536 vertices they
were 2.9875, 2.9875, 3.0125 and 5.940 ms respectively.

The paired internal-minus-borrowed median was -0.0175 ms at 16,384 vertices, with
a 95% block-bootstrap interval of [-0.060, +0.040] ms. At 65,536 it was +0.0125 ms,
with [-0.0325, +0.115] ms. GPU-copy intervals also crossed zero. This is no
demonstrated advantage, not proof of equivalence. These single-session CPU-observed
completion latencies include synchronization and scheduling; they are not GPU
execution times, throughput or FPS. The stronger CPU-copy penalty does not make
the borrowed-buffer API a novel performance capability.

Every scalar matched the independent model and all lane images matched exactly at
two time points. The stock-only probe passed buffer identity, scalar correctness
and teardown without patched imports. Unlike the earlier borrowed-buffer canary,
this screen did not record a native operation trace; its claim is shared output
buffer access and absence of an application download/upload in the direct lanes.

Raw evidence is preserved under `results/development/ort-handoff/`:

- `2026-09-09T16-40-17.581Z-stock-128`
- `2026-09-09T16-42-53.403Z-four-lane-128`
- `2026-09-09T16-43-55.534Z-four-lane-256`

The verified analysis is
`results/development/ort-handoff-analysis/2026-09-09T16-47-39.613Z/summary.json`.
It checks report hashes, correctness/cleanup, all 24 permutations, and eight
warmups per lane. It resamples complete blocks 10,000 times with a fixed seed.
Use its conventional medians; the original runner summary chose the upper middle
observation for an even sample count.

```sh
node scripts/run-ort-handoff-screen.mjs --stock --size=128
node scripts/run-ort-handoff-screen.mjs --size=128
# Allow a cooldown before the larger independent browser session.
node scripts/run-ort-handoff-screen.mjs --size=256
node scripts/analyze-ort-handoff-screen.mjs
```

See [the protocol](PROTOCOL.md) for limits and the declared decision rule. This
development screen did not rerun any sealed historical experiment.

The next concrete contribution is the [ZipDepth export fidelity fix](../zipdepth-export/README.md).
GPU-only ONNX integration itself is prior art: [ORT GPU IO binding](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
and [pmndrs denoiser's Three integration](https://github.com/pmndrs/denoiser/blob/main/docs/guides/three-js-render-targets.md).
