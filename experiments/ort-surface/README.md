# ONNX → three.js without a CPU round trip

A working research demo of GPU-generated inputs, ONNX WebGPU inference, and
three.js vertex displacement sharing one GPU device and external buffers.

This is a useful integration contribution, not a new machine-learning technique
or a demonstrated universal speedup. It implements the missing Three-side buffer
adoption and ownership semantics and demonstrates an actual tensor-to-render path.

**Later September 9 qualification:** the [strongest-baseline comparison](../ort-handoff/README.md)
found no demonstrated timing advantage over an existing direct path using
Three-owned buffers obtained through private backend access. The patch's remaining
value is a public API for externally owned buffers. It is not evidence that
GPU-only ONNX/Three integration was previously impossible. The latest concrete
finding is the [ZipDepth export-fidelity fix](../zipdepth-export/README.md).

## Run the portable demo

Install Node.js 22.12 or later if needed, extract the complete demo folder, and run:

```sh
node serve.mjs
```

Open `http://127.0.0.1:5186` in Chrome with WebGPU support. Click **Load demo**, then
**Animate 8s**. **Check parity** compares every model output with an independent
CPU calculation and requires byte-identical images from the shared and copying
paths. The server binds only to localhost; stop it with Ctrl+C. No network service,
account, model download, enabling browser flags, or GPU configuration change is needed.

The page starts idle. Animation stops after eight seconds, at 120 updates, or when
the page is hidden. It is capped at 15 updates per second. Resolution is bounded
at 65,536 vertices; the default is 16,384. Reload to apply a changed resolution.

In the full research repository, equivalent commands are:

```sh
node scripts/prepare-ort-surface.mjs
node scripts/run-ort-surface.mjs --serve
node scripts/run-ort-surface.mjs --size=128 --animate
node --test scripts/test-external-storage.mjs
```

## What works

1. A TSL compute shader fills an external input tensor with grid coordinates and time.
2. ONNX Runtime runs a five-operator sine network into a preallocated output tensor.
3. Three.js reads that same output GPUBuffer in its vertex and color nodes.
4. The output can also be explicitly downloaded and reuploaded for a comparison.

At the default 16,384 scalar outputs, the comparison transfers 64 KiB down and
64 KiB up per update. The shared handoff removes those 128 KiB of CPU transfers.
The UI's “0 bytes” refers to the tensor handoff, not all application traffic:
small uniforms and weights still originate on the CPU. GPU intermediates inside
ORT are outside the handoff claim. Correctness checks deliberately read data back.

The model has 161 float parameters and is 891 bytes. It fits random sine features
to a synthetic traveling wave using a CPU ridge fit. It is **not a learned physics
solver**. `train.py` regenerates its weights with NumPy on one CPU thread; the
ONNX binary and JSON reference weights are already included. No training is needed
to run the demo. The model's simplicity makes the buffer path easy to inspect.

## Observed evidence, September 9, 2026

Chrome 152.0.7977.83 / NVIDIA RTX 5070 Ti / Windows, patched Three `186dev`,
official `onnxruntime-web@1.30.0-dev.20260904-d47fd8824`:

- 4,096 vertices: both tested time points passed; largest CPU-reference absolute
  error was 2.21e-7; both image comparisons had zero unequal bytes.
- 16,384 vertices: both tested time points passed; largest CPU-reference absolute
  error was 3.39e-7; both image comparisons had zero unequal bytes.
- The 16,384-vertex animation completed 118 updates within its eight-second limit.
- Native traces show input-generation and inference dispatches plus indexed draws
  using the same device and the declared buffers. No mapping, CPU writes, clears,
  or GPU copies involving the input/output buffers occurred during shared proof updates.
- Material, compute-node, attribute, renderer and ORT teardown passed; neither
  borrower destroyed the application's buffers. GPU error lists were empty.
- All 14 external-attribute unit cases passed. The source patch applies to its
  pinned upstream base `c4ffe022f2a4f982b42b7da5af79a87066a138ae`.
- The repository suite passed 962 tests with five expected Windows skips; the
  production build and focused candidate lint passed.

These are bounded development correctness runs on one implementation. They are
not statistically controlled timing results. The recorded update rate is capped
intentionally and must not be advertised as a throughput or FPS measurement.

## The Three.js API change

```js
const outputAttribute = new THREE.ExternalStorageBufferAttribute(outputGpuBuffer, {
  device, type: 'float', count: vertexCount
});
const heights = storage(outputAttribute).toReadOnly();
material.positionNode = vec3(x, heights.element(vertexIndex), z);
```

The experimental attribute declares an immutable element type and byte range.
It has no CPU array and does not own its GPUBuffer. It validates device identity,
alignment, buffer usage, mapping state, and bounds. Scalar, vec2 and vec4 types
are supported. Packed vec3 is rejected because WGSL storage arrays need a 16-byte
stride; packed triples can be exposed as scalars.

Dispose consumers before disposing the attribute, wait for submitted work before
recycling the buffer, and let its owner destroy it. The patch adds lifetime tracking
for native bind groups and does not put borrowed buffers in the backend binding
cache. It is a candidate patch, not an upstream release or an accepted PR.

## Prior art and limits

ONNX already provides GPU tensor IO binding; that part is prior art:
https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

The current Three StorageBufferAttribute constructor takes a count or typed CPU
array; this patch supplies a native external storage buffer route:
https://github.com/mrdoob/three.js/blob/aaf21735d5eef2321d84ec6623c500ac935cbab7/src/renderers/common/StorageBufferAttribute.js

Existing Three work on attribute disposal and cached bind-group lifetime affects
the prototype's compatibility:
https://github.com/mrdoob/three.js/pull/33468

The official nightly is pinned because this work exercises the newer custom-device
WebGPU path. In this nightly, `ort.env.webgpu.device` did not reflect the explicitly
supplied per-session device; the demo validates actual native command-encoder
ownership instead. Do not replace that proof with an assignment to the global flag.

Earlier attempts are retained in the research repository, including page-loading
failures and the rejected global-device check. They were development failures;
their results are not silently promoted to successful evidence.

Remaining work before a broad performance claim: representative trained models,
matched latency measurements, other GPUs/browsers, and current API compatibility. The
current contribution is a functioning, reusable zero-copy **handoff** with a
small visual reproduction. [API and ownership details](EXTERNAL_BUFFER_API.md) document the prototype.

A separate `three-external-storage-buffer-current-dev.patch` mechanically ports
the change to inspected upstream dev `aaf21735d5eef2321d84ec6623c500ac935cbab7`.
Its application was checked against those exact upstream files. That newer source
has not been browser-tested here; the demo deliberately retains the tested base.

## Licensing

Three.js and ONNX Runtime are MIT-licensed; their notices are in `licenses/`.
The demo, model and training script are offered under the included demo MIT license.
The ONNX runtime is bundled for offline reproduction. No Three Blocks code is
included in the portable demo.
