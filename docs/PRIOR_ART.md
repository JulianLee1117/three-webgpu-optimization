# Prior art and positioning

GPU frustum culling, survivor compaction, indirect drawing, and retained command submission are established rendering techniques. This project does not claim to have invented them.

The relevant question is narrower: how these mechanisms behave through Three.js WebGPU APIs, how existing ecosystem implementations compare under equivalent workloads, and whether a measurable implementation or integration improvement remains.

## Direct Three.js ecosystem work

- [Three Blocks instance culling](https://threejs-blocks.com/docs/blocks/instance-culling) provides GPU instance culling and compacted survivors for WebGPU.
- [Three Blocks `ComputeInstanceCulling`](https://threejs-blocks.com/docs/api/ComputeInstanceCulling) exposes indirect arguments, survivor data, optional sorting, and diagnostic access.
- The [Three Blocks changelog](https://threejs-blocks.com/changelog) records the removal of an earlier indirect-batching implementation in version 0.11.0.
- The published `three-blocks@0.10.0` package exposes `IndirectBatchedMesh` and `ComputeBatchCulling`, making it the direct historical heterogeneous-geometry comparison.
- [InstancedMesh2](https://agargaro.github.io/instanced-mesh/) provides per-instance culling and sorting in a WebGL-focused implementation and belongs in a separate renderer lane.
- [Three.ez BatchedMesh extensions](https://github.com/agargaro/batched-mesh-extensions) provide CPU/BVH culling options, including a WebGPU build.

The published Three Blocks package uses the PolyForm Noncommercial license. Its source may be studied and the package may be exercised within its license terms, but it will not be copied into this repository. Benchmark integrations will remain version-pinned and clearly separated from this project's own source.

## Three.js and WebGPU constraints

- [`BufferGeometry.setIndirect`](https://threejs.org/docs/pages/BufferGeometry.html) allows application-provided indirect buffers and offsets.
- [`BundleGroup`](https://threejs.org/docs/pages/BundleGroup.html) exposes WebGPU render bundles through Three.js.
- Portable WebGPU indirect commands require `firstInstance` to remain zero unless the optional `indirect-first-instance` feature is available; implementations need a portable indexing strategy.
- Portable WebGPU does not currently provide a counted multi-draw operation equivalent to native `drawIndirectCount` workflows.

## Claim policy

Potential contributions are classified as one of:

- **established** — already demonstrated in prior systems or libraries;
- **replicated** — independently reproduced under a documented configuration;
- **differentiated** — directly compared and shown to have a material implementation distinction;
- **unverified** — plausible but not yet tested against the relevant baseline;
- **falsified** — contradicted by controlled evidence.

Any public performance result must name the exact comparison, tested configuration, absolute timing, validation status, and known limits. A result on one device or scene is not generalized to WebGPU as a whole.
