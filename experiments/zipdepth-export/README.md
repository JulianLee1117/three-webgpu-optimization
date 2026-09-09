# ZipDepth export fidelity: a reproduced bug and a working browser fix

The released ZipDepth Base NPU model uses learned spatial attention. Its ONNX
exporter replaces that computation with unweighted averaging. We reproduced the
resulting output changes, removed the substitution, and verified that the fixed
model matches a preserved PyTorch reference in CPU and browser execution.

This is an export-fidelity finding with a visible three.js reproduction. The tests
measure agreement with the checkpoint; they do not measure depth accuracy or speed.

## Reproduce from a fresh clone

The tested setup is Windows, Node 22.12+, Python 3.13 managed by `uv`, and installed
Google Chrome with WebGPU support. Install Git, Node and uv first. These PowerShell
commands install the pinned dependencies and generate the model fixtures locally:

```powershell
git clone https://github.com/JulianLee1117/three-webgpu-optimization.git
cd three-webgpu-optimization
npm ci
uv venv .local-research/zipdepth-export-env --python 3.13
uv pip install --python .local-research/zipdepth-export-env/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
uv pip install --python .local-research/zipdepth-export-env/Scripts/python.exe onnx==1.19.0 onnxruntime==1.22.1 numpy==2.3.3
.local-research/zipdepth-export-env/Scripts/python.exe experiments/zipdepth-export/prepare.py
.local-research/zipdepth-export-env/Scripts/python.exe experiments/zipdepth-export/probe.py
.local-research/zipdepth-export-env/Scripts/python.exe experiments/zipdepth-export/probe.py --width 512
npm run check:zipdepth
```

If you already cloned the repository, start with `npm ci` from its root. Python
dependencies stay in `.local-research/zipdepth-export-env`; npm dependencies stay
in `node_modules`. The CPU-only Torch Windows wheel is about 591 MiB, and the
official checkpoint is 27,295,474 bytes. The preparation script pins the upstream
revision and all six downloaded file hashes. Generated ONNX models, checkpoints
and environment directories are not included in the public source package.

No training or CUDA package is involved. Export and inference use two CPU compute
threads. The browser runner launches an independent Chrome session, runs two
inferences, captures the result, disposes resources, and closes the session with a
45-second watchdog. It uses Playwright's installed-Chrome discovery. For a custom
installation, set `CHROME_PATH` to your Chrome executable before running it:

```powershell
$env:CHROME_PATH = 'D:\Apps\Chrome\chrome.exe'
npm run check:zipdepth
```

No custom browser flags, driver changes or elevated device limits are used. The
CPU commands above are the tested Windows setup; Linux environments use the
virtual environment's `bin/python` in place of `Scripts/python.exe`. Other platforms
and package builds have not been validated by this reproduction.

## Local quickstart after preparation

Once the 384×384 CPU probe has passed, run this from the repository root:

```sh
npm run demo:zipdepth
```

Open [the local demo](http://127.0.0.1:5187/) in Chrome and click **Run two bounded
inferences**. It starts idle, runs each model once, and leaves a static three-panel
comparison. Reload to repeat. There is no animation or background inference loop.
The exact same height/color scale is used for every panel. Closing the tab releases
its GPU resources; Ctrl+C stops the localhost-only server. Interactive serve mode
limits the inference count; the 45-second watchdog applies to the automated check.

The runner selects the newest completed 384×384 CPU fixture and verifies its
report and artifact hashes. A missing fixture directory or unfinished report gives
the preparation commands. A checksum mismatch stops the run.

## Evidence and interpretation

Pinned upstream: [`91f3fd21e131641f51e8d35736d1958350180e3a`](https://github.com/fabiotosi92/ZipDepth/tree/91f3fd21e131641f51e8d35736d1958350180e3a).
The [learned block](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/zipdepth/model/architecture.py#L269-L279)
uses a scoring convolution, spatial softmax and weighted sum. The
[export override](https://github.com/fabiotosi92/ZipDepth/blob/91f3fd21e131641f51e8d35736d1958350180e3a/scripts/export.py#L69-L73)
discards those scoring weights. The original export's shape-only sanity check runs
after mutation, so it does not catch the change.

The reproduction loads every checkpoint key strictly, keeps independent copies,
checks fusion separately, and compares against the model before export mutation.
Three procedural RGB inputs (gradient, checkerboard and constant) were tested at
384×384 and 384×512. All six corrected CPU outputs pass `rtol=1e-4, atol=1e-5`;
all original outputs fail. Both original and corrected ONNX agree with their
respective post-export PyTorch models, isolating the exporter substitution.

The browser test uses Chrome 152.0.7977.83, stock Three r185, ORT Web
`1.30.0-dev.20260904-d47fd8824`, and the local RTX 5070 Ti. On the 384×384 gradient:

- Original WebGPU export versus preserved reference: max absolute difference
  0.02282134; 147,339 of 147,456 elements fail the tolerance.
- Corrected WebGPU export versus reference: max difference 3.67e-7; every element
  passes the same tolerance.
- Both WebGPU outputs match their CPU ONNX outputs within 2.11e-7 maximum difference.
- Both ONNX outputs retain the preallocated Three-owned GPUBuffer identity. The
  browser renders those buffers, then reads them back only for numerical validation.
- GPU error lists were empty and cleanup passed. Before/after GPU spot checks were
  44°C and 46°C; these are not continuous peak-temperature measurements.

These differences measure fidelity to the original checkpoint, not correctness
against real-world depth. The inputs are procedural, not a representative image
dataset. The separate GPU checkpoint, mobile hardware and quantized exports remain
untested. An intentional deployment approximation is possible, but we found no
description in the README or main paper; supplementary material was not checked.

The minimal [source patch](preserve-global-context.patch) removes the learned-to-mean
override and changes the following `elif` to `if`. All other export changes remain.
The corrected graph uses ordinary ONNX Softmax and MatMul, and executes in ORT WebGPU.
The [technical export finding](EXPORT_FINDING.md) records the computation change, reproduction and limits.

## Preserved files

- CPU square: `results/development/zipdepth-export/2026-09-09T16-56-54.752+00-00/`
- CPU rectangle: `results/development/zipdepth-export/2026-09-09T16-56-57.499+00-00/`
- Browser: `results/development/zipdepth-browser/2026-09-09T16-57-30.246Z/`

Each CPU report records source hashes, checkpoint identity, package versions,
per-case differences and every generated artifact hash. Binary inputs, reference
outputs and both ONNX models are retained. The browser runner verifies the CPU
report and artifact hashes before using them, then records its own source hashes,
metrics, cleanup and screenshot. The first exploratory CPU report at 16:48 is
retained but its report sidecar has a Windows newline mismatch; it is superseded
by the verified square and rectangle reports, not used by the browser runner.

The final reports above include pinned cached-download verification and optional
onnxsim metadata. onnxsim was absent, so no simplification was performed. Earlier
16:49 reports have valid numeric evidence but their patch artifacts use Windows
line endings; the final patch is LF-normalized and `git apply --check` passes on
the exact upstream source. Those earlier artifacts remain unchanged.

## Attribution

ZipDepth and its exporter are by Fabio Tosi and collaborators. The upstream
[MIT license](LICENSE-ZipDepth.txt) is retained here. The model checkpoint is
downloaded from the pinned official repository into the local research cache;
generated models remain local research artifacts. Three.js and ONNX Runtime keep
their package license notices. This reproduction does not redistribute Three Blocks.
