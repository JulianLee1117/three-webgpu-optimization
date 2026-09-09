# ONNX export replaces learned global-context attention with average pooling

The reproduction comprises `prepare.py`, `probe.py`, the recorded reports, and `preserve-global-context.patch`. Current square-input evidence: `results/development/zipdepth-export/2026-09-09T16-56-54.752+00-00/report.json`. Rectangular-input confirmation: `results/development/zipdepth-export/2026-09-09T16-56-57.499+00-00/report.json`. Browser evidence: `results/development/zipdepth-browser/2026-09-09T16-57-30.246Z/report.json`.

At revision `91f3fd21e131641f51e8d35736d1958350180e3a`, exporting the released Base NPU checkpoint changes its outputs because the exporter replaces learned spatial attention with unweighted average pooling. Removing that substitution restores numerical agreement with the preserved PyTorch reference in three procedural-image tests.

The original [GlobalContextBlock forward](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/zipdepth/model/architecture.py#L269-L279) computes a learned scoring convolution, spatial softmax, and weighted sum. However, [export.py lines 69–73](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/scripts/export.py#L69-L73) monkey-patch that forward method to use `avg_pool2d`, bypassing the scoring weights. These operations are not generally equivalent.

The [pre-export sanity check](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/scripts/export.py#L92-L104) runs after this mutation and prints only the output shape. It compares no output values against the original model. A runtime comparison against the mutated model also passes, so that comparison alone misses the discrepancy.

Reproduction uses Python 3.13, PyTorch `2.8.0+cpu`, ONNX `1.19.0`, ONNX Runtime `1.22.1`, and NumPy `2.3.3`. From the reproduction bundle's root, in an environment containing those dependencies:

```sh
python experiments/zipdepth-export/prepare.py
python experiments/zipdepth-export/probe.py
python experiments/zipdepth-export/probe.py --height 384 --width 512
```

The preparation script fetches the pinned source and `zipdepth_base_npu.pth`; the checkpoint SHA-256 is `627c04fda584133ead4310074884a4a037061b4c01ba86e73e492ea30fab570d`. The probe loads all checkpoint keys strictly, uses Base/balanced/NPU configuration, preserves a reference before exporter mutation, and exports independent copies through the original and minimally corrected code. Both exports retain the same other export transformations. Inference uses CPU only, with two compute threads, and ONNX opset 17.

Results for deterministic 384×384 inputs, compared with the preserved fused PyTorch reference:

- RGB gradient: original export maximum absolute difference `0.02282126`; relative L1 difference `0.309756`. Corrected maximum absolute difference `3.76e-7`.
- Checkerboard and constant RGB: original relative L1 differences `0.040755` and `0.086062`; corrected maximum absolute differences `1.79e-7` and `2.61e-7`.
- All corrected outputs pass `allclose(rtol=1e-4, atol=1e-5)`; all original outputs fail. Original ONNX agrees with the already-mutated PyTorch model within `1.68e-7` maximum absolute difference across these cases.
- The separate fused-versus-unfused checkpoint gate passes on all three inputs, with maximum absolute difference below `1.98e-7`. Repeating the three cases at 384×512 also passes the fusion and corrected-export gates; corrected maximum absolute difference is below `3.13e-7`.

Here relative L1 means `sum(abs(export-reference)) / sum(abs(reference))`. These are raw output-fidelity measurements on procedural inputs, not depth-accuracy scores, visual-quality claims, performance measurements, or a dataset evaluation. The separate GPU checkpoint and mobile runtimes have not been tested.

The exported NPU checkpoint was also checked through ONNX Runtime WebGPU in Chrome 152 on an RTX 5070 Ti, with preallocated stock three.js output buffers. On the 384×384 gradient, corrected output versus the preserved CPU reference had maximum absolute difference `3.67e-7` and every output passed the same tolerance. The original export differed by up to `0.02282134`. Both WebGPU outputs matched their respective CPU ONNX outputs within `2.11e-7`. The browser reproduction and screenshot are included separately; this does not turn the fidelity test into a depth-accuracy evaluation.

The minimal correction deletes the GlobalContextBlock override and changes the following `elif` to `if`, retaining the original learned pooling. The corrected export uses standard `Softmax` and `MatMul` nodes and passes ONNX checker and CPU inference. Regression coverage compares against an unmodified reference captured before export and compares the unfused checkpoint with the fused reference separately.

The checked README and main paper do not document an average-pooling approximation; the [paper's Equation 4](https://arxiv.org/html/2607.08771v1#S3.SS1) specifies softmax-weighted pooling. The supplementary material was not checked. Whether this replacement is an intentional deployment approximation remains unresolved; its fidelity tradeoff is distinct from export equivalence.
