# Three.js WebGPU Optimization

Evidence-driven research into GPU visibility, indirect rendering, and retained command submission for Three.js WebGPU.

## Objective

This project investigates where GPU-driven rendering can deliver meaningful, reproducible improvements in Three.js applications. The current work centers on GPU frustum culling, visible-instance compaction, compute-written indirect commands, and render-bundle reuse across repeated geometry.

The goal is not a synthetic headline. It is a technically defensible result that identifies:

- the workloads where an optimization wins;
- the crossover where its overhead exceeds its benefit;
- the implementation constraints that matter in Three.js and WebGPU;
- whether the result supports an upstream contribution, a reusable implementation, or a documented negative result.

## Current status

A controlled prototype has produced a substantial low-visibility reduction in timestamped GPU work on one NVIDIA/D3D12 configuration. It also passes exact visible-instance membership checks in the tested scenes. Those results remain preliminary until the implementation is compared with existing Three.js ecosystem options, exercised with indexed PBR assets and dynamic scenes, and reproduced on another GPU family.

The core GPU-culling pattern is established prior art. This repository focuses on rigorous comparative measurement and on finding narrower Three.js-specific improvements that survive production-style validation.

## Primary hypothesis

Scenes with immutable geometry ownership may not need a general prefix-allocation-and-redistribution pipeline. A fixed-slice specialization can preallocate one survivor range per geometry, reset its counters, append visible IDs directly into those ranges, keep indirect `firstInstance` values at zero, and replay separate geometry commands from a cached render bundle.

That design trades mutation flexibility for fewer compute stages and a portable command layout. The key experiment is whether one shared compute submission and direct per-geometry append remain materially faster than current and historical ecosystem implementations once indexed geometry, full transforms, PBR materials, camera motion, and output equivalence are included.

## Evidence standard

Results published here will include:

- pinned Three.js, browser, operating-system, and device details;
- equivalent workloads and shaders across compared modes;
- GPU timestamps separated from CPU preparation and submission time;
- exact survivor validation plus render-target color, depth, and object-ID checks;
- repeated runs, distributions, raw machine-readable data, and analysis code;
- negative and near-parity results alongside wins;
- cross-device evidence before any general performance claim.

The initial benchmark protocol is documented in [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md). Direct ecosystem prior art is recorded in [docs/PRIOR_ART.md](docs/PRIOR_ART.md).

## Scope

The first controlled comparison targets plain Three.js draw-all baselines, `BatchedMesh`, current Three Blocks instance culling, and the independent prototype under the same WebGPU harness. WebGL-only approaches will be measured separately and will not be presented as same-backend comparisons.

This repository is under active research and does not yet expose a supported library API.
