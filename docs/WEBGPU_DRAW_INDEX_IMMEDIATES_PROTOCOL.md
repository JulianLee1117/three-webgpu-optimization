# WebGPU `drawIndex` via immediate data

Status: offline implementation and evidence gates complete. No browser or GPU execution is authorized by this document.

## Result sought

Add native TSL `drawIndex` support to Three.js's WebGPU renderer by using WebGPU immediate data. The intended upstream result is a small engine capability with explicit semantics and a reviewable source patch, not a machine-specific speed claim.

The public claim is limited to correctness:

- an explicit TSL `drawIndex` use can select the zero-based physical draw-command ordinal emitted for one render object;
- direct draws receive `0`, while BatchedMesh and array-indirect loops receive `0..N-1` in command order;
- the value resets for each new logical draw list, including each ArrayCamera layer;
- pipelines which do not explicitly use `drawIndex` remain immediate-free;
- default BatchedMesh rendering retains its existing backend-specific ID transport;
- unsupported or compute-stage use is rejected deterministically during WGSL construction; Three.js's established render-material recovery reports that TSL error and builds its ordinary immediate-free fallback material, while compute build errors propagate.

This work does **not** claim fewer draw calls, a performance improvement, universal browser support, or correctness beyond the tested implementations.

## Frozen source identity

The patch is authored against the clean Three.js source below and must fail closed against any other source:

- commit: `994260a7a59abe466de8e59db1ecc9e30352751d`
- tree: `7f3d4a048ea19bef1a488b0e2e243e47f6b30002`
- package: `three@0.185.0`
- runtime revision: `186dev`

The sibling checkout `../upstream-three` is an immutable input. A repository-owned unified diff is applied only to a generated staging copy.

## Capability contract

Support is true only when both facts are observed after the actual `GPUDevice` is acquired:

1. `navigator.gpu.wgslLanguageFeatures.has( 'immediate_address_space' ) === true`;
2. `device.limits.maxImmediateSize >= 4`.

The renderer must not add `maxImmediateSize` to `requiredLimits`, invent a `GPUFeatureName`, or infer support from the presence of a JavaScript method alone. A requiring draw also checks that its actual render-pass or render-bundle encoder exposes callable `setImmediates()` and fails before emitting a draw if the API surface is inconsistent.

## Shader and pipeline contract

An explicit supported `drawIndex` use causes the vertex WGSL module to contain exactly one of each declaration:

```wgsl
requires immediate_address_space;
var<immediate> nodeDrawIndex : u32;
```

Fragment use is carried through the existing flat varying path, so only the vertex entry point accesses the immediate variable. Compute use is rejected. The requirement is emitted from structured builder metadata, never by searching generated shader text, and is not routed through Three.js's `enable`-directive serializer.

The builder records `immediateSize = 4`; an unused pipeline records `0`. In this first integration, byte range `[0, 4)` is reserved exclusively for `drawIndex`; any future immediate input must introduce an explicit encoding plan rather than sharing that range implicitly. The size is transported through `NodeBuilderState`, copied into `GPUPipelineLayoutDescriptor.immediateSize`, and retained beside the native render pipeline. Missing, null, or non-`0`/`4` pipeline metadata is an error. The reusable descriptor always resets to zero in a `finally` block. Compute layouts are always zero.

## Command contract

For a pipeline whose immediate size is four, the backend records the following adjacent command pair for every physical draw:

```text
setImmediates(0, Uint32Array.of(ordinal), 0, 1)
draw-or-drawIndirect(...)
```

The production implementation reuses one `Uint32Array(1)` but never caches encoder immediate state. The value is set again after pipeline switches, before zero-count draws that are still physically recorded, and while recording render bundles. Ordinary indexed, non-indexed, and instanced draws use ordinal zero. Batched and array-indirect loops use the loop index, independent of indirect byte offsets.

Default WebGPU BatchedMesh lookup continues to use `instanceIndex` with the existing `firstInstance = i`; its `batchIndirectIndex` remains the original instance ID fetched from the indirection texture. Enabling explicit `drawIndex` must not silently move this internal lookup onto immediate data.

## Evidence gates

Before any browser/GPU run, all of the following must pass offline:

- the tracked patch round-trips byte-for-byte through Git and changes only its allowlisted source files;
- source commit, tree, cleanliness, Git blobs, canonical preimage hashes, patch hash, and postimage hashes are pinned;
- generation is deterministic and cannot replace a prior output after a failed check;
- source/code-generation tests cover unused, vertex, fragment, repeated, unsupported, compute, and established render-material recovery cases;
- descriptor tests cover `4 -> 0 -> 0`, synchronous failure cleanup, and render/compute isolation;
- fake render-pass and render-bundle traces cover direct, indexed, instanced, BatchedMesh, scalar-indirect, array-indirect, zero-count, empty-list, pipeline-switch, and missing-setter paths;
- default BatchedMesh remains immediate-free;
- the repository test suite, production build, and `git diff --check` pass.

The timing-free, one-shot browser correctness protocol is now frozen in
`protocols/three-draw-index-canary-v1.json`. Its exact command is:

```powershell
npm.cmd run probe:three-draw-index -- --browser "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

The command is not authorized merely by this document. It may run only after
the browser page, runner, independent validator, source hashes, output path,
and artifact policy pass offline tests and the user explicitly approves this
exact browser/GPU execution.

The frozen run uses Chrome's default WebGPU exposure with no custom browser
arguments. It launches once, requests one device with no descriptor (and thus
no promoted limit), and never retries or falls back to developer flags. It
must capability-exit before constructing a renderer or generating a shader if
the WGSL language feature, the actual four-byte device limit, or either render
encoder setter is unavailable.

The run validates exact pixel values and command traces for a direct fragment
use, nonmonotonic indexed-indirect order, cached and explicitly re-recorded
render bundles, a nonidentity BatchedMesh order proving compact `drawIndex`
versus original `batchIndirectIndex`, a default immediate-free BatchedMesh,
and independent ArrayCamera layer resets. It also validates shader/pipeline
identity, compilation messages, Three.js error interception, nested and outer
error-scope balance through resource disposal, uncaptured errors, and final
intentional device loss. It collects no benchmark timings.

The exact report path is
`results/development/three-draw-index-default-exposure-canary/canary-draw-index-default-exposure-994260a-v1/report.json`.
The runner creates that run directory exclusively, refuses replacement, writes
canonical UTF-8/LF JSON with `wx`, rereads it, marks it read-only, and writes a
separately verified SHA-256 sidecar. A clean capability exit is `unsupported`,
not a correctness pass.

## Primary references

- [GPUWeb specification](https://gpuweb.github.io/gpuweb/)
- [WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/)
- [Chrome WebGPU immediate-data announcement](https://developer.chrome.com/blog/new-in-webgpu-149-150)
- [Three.js WebGPU `drawIndex` feature request](https://github.com/mrdoob/three.js/issues/33576)
