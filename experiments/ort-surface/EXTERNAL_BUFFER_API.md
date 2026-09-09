# WebGPU: Borrow external storage buffers in TSL

Applications that already hold tensor or simulation output in a native GPUBuffer
currently need a copy or private backend access to consume that data in a Three
storage node. The prototype `ExternalStorageBufferAttribute(buffer, { device, type, count,
byteOffset })` lets a caller-owned buffer be bound directly on its originating
device. The renderer allocates no replacement data and never destroys the buffer.

The attribute exposes an immutable typed byte range without a CPU array. It
validates usage, mapping state, device identity, alignment and bounds. Scalar,
vec2 and vec4 float/int/uint elements are supported; vertex/index/indirect,
packed vec3, atomic and structured-buffer interpretations are rejected.

Native bind-group references retain lifetime tokens. Attribute disposal rejects
live consumers, releases Three bookkeeping after consumers are disposed, and
preserves the caller's GPUBuffer. Borrowed buffers bypass native bind-group
caching to avoid references outliving an old binding. Renderer shutdown releases
remaining external-attribute tokens and listeners.

Validation: 14 unit cases cover ownership, exact ranges, rejection paths and
disposal. A local Chrome 152 / RTX 5070 Ti demonstration generates a GPU input
tensor in TSL, runs official ONNX Runtime WebGPU into preallocated output, then
uses the identical buffer for a Three surface. At 16,384 vertices, two distinct
outputs agree with an independent CPU reference within 3.4e-7 and render identical
images to an explicit download/upload comparison. Traces show no input/output
buffer copy or mapping in the shared updates; cleanup reports no GPU errors or
borrower destruction. No performance improvement is claimed from these tests.

Lifetime behavior overlaps the attribute-disposal work in three.js PR #33468.
The new class/type restrictions are experimental, and the rebased current-dev
patch has not passed the full upstream test suite. The browser evidence uses the
pinned September 4 source base.

A subsequent four-lane screen found no demonstrated latency advantage over
prebinding ORT output to an existing Three-owned buffer through private backend
access. The ordinary r185 path also passed correctness without patched imports.
The prototype therefore addresses external ownership and integration,
not introducing GPU tensor IO binding or a previously unavailable speedup.
