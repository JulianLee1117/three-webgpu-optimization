# Benchmark protocol

## Research question

Under which combinations of visibility, instance count, geometry cost, material cost, geometry diversity, camera motion, and device class does GPU visibility compaction improve a Three.js WebGPU workload after all compute and submission overhead is included?

## Initial comparison

The first benchmark milestone will compare equivalent representations of the same scene:

1. bundled per-geometry instancing with all instances submitted;
2. Three.js `BatchedMesh` with per-object culling disabled;
3. Three.js `BatchedMesh` with per-object culling enabled;
4. current Three Blocks GPU instance culling;
5. the independent atlas implementation;
6. the independent separate-geometry implementation.

Historical implementations may be included when their exact version, source, license, and compatibility constraints can be established.

The comparison is split into two questions so that submission overhead is not mistaken for a better culling kernel:

1. **Single-geometry parity:** compare draw-all, current Three Blocks instance culling, and fixed-slice compaction with one indexed geometry and identical instance data.
2. **Heterogeneous scaling:** compare stock per-geometry culling submissions, coalesced culling work in one explicit compute submission, historical heterogeneous indirect batching, and one shared fixed-slice culler across 32, 128, and 512 geometry buckets.

The primary implementation hypothesis is that one explicitly scheduled compute submission can update many heterogeneous indirect commands before a cached render bundle is replayed. The benchmark must separately identify savings from submission coalescing and savings from the compaction layout.

## Controlled variables

The headline subset will hold the renderer, dependency versions, shaders, transforms, geometry, viewport, device, browser, and camera constant while sweeping:

- visible fraction: approximately 20%, 80%, and 99%;
- geometry buckets: 32, 128, and 512;
- camera behavior: frozen and deterministic motion;
- geometry workload: synthetic control and indexed production assets;
- material workload: minimal and representative PBR.

Visibility experiments must avoid changing field of view when that would confound visibility with projected coverage. Camera position or controlled scene layouts are preferred.

## Measurements

Each accepted run records:

- GPU compute-pass, render-pass, and combined-pass distributions;
- CPU culling, renderer preparation/submission, and frame-body distributions;
- visible fraction and submitted survivor count;
- direct and indirect command counts;
- render-bundle rebuild count;
- initialization cost and memory use where measurable;
- timestamp resolution and backend information;
- correctness results and deterministic configuration identifiers.

Pass-duration sums are not described as presentation latency or end-to-end frame latency. Queue gaps and compositor work require separate measurement.

## Correctness gate

Timing data is accepted only after validation reports:

- no duplicate, missing, invisible, wrong-bucket, or out-of-range survivor IDs;
- no in-capacity overflow;
- valid indirect command fields and strides;
- exact object-ID target agreement;
- color and linear-depth agreement within documented tolerances;
- no unexpected render-bundle rebuild for buffer-only updates;
- no WebGPU validation error during deterministic camera and mutation tests.

## Statistical protocol

- Warm up pipelines and the benchmark workload before capture.
- Rotate or randomize mode order.
- Use at least five repetitions for published headline cells.
- Retain frame-level samples, not only per-run summaries.
- Report medians, tail distributions, absolute milliseconds, and frame-budget context.
- Preserve raw results as immutable inputs to versioned analysis scripts.

## Initial decision gates

On the existing discrete-GPU test system, a production-style path remains interesting if it achieves all of the following:

- at approximately 20% visibility, at least a 40% GPU-pass reduction versus equivalent bundled draw-all;
- at approximately 80% visibility, at least a 10% reduction in a meaningful geometry workload;
- near full visibility, no more than a 5% regression;
- zero correctness failures;
- acceptable CPU submission scaling through 128 geometry buckets, followed by explicit evaluation at 512.

Broader guidance requires a second materially different GPU family. These thresholds are falsification gates for the research direction, not universal performance claims.

For the heterogeneous comparison, the independent path must also beat the best compatible existing implementation by at least 10% in GPU pass time or 0.10 ms in combined CPU and GPU cost at low visibility. Otherwise the useful output is the benchmark or an upstream scheduling improvement, not another culling implementation.
