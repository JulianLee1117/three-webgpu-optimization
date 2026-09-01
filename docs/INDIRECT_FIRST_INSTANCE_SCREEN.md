# Indirect `firstInstance` addressing screen

## Research question

For fixed-slice heterogeneous rendering in Three.js WebGPU, does moving each
bucket's survivor-list base from a per-vertex `uint` attribute into the fifth
word of its indexed indirect command materially reduce render-pass time?

This is a development screening experiment, not candidate evidence and not a
claim that `firstInstance` is a new technique. WGSL defines
`@builtin(instance_index)` to begin at the draw's `firstInstance`, and WebGPU
permits a nonzero value in an indirect command when the optional
`indirect-first-instance` device feature is enabled. Three.js r185 exposes the
feature and forwards public indirect geometry commands to WebGPU. Three Blocks
0.10 also uses the underlying addressing technique. The open question is the
marginal value of this specialization inside this repository's fixed-ownership
representation.

Primary references:

- [WGSL `instance_index` semantics](https://www.w3.org/TR/WGSL/#built-in-values-instance-index)
- [WebGPU `indirect-first-instance` feature](https://www.w3.org/TR/webgpu/#indirect-first-instance)
- [Three.js `BufferGeometry.setIndirect`](https://threejs.org/docs/pages/BufferGeometry.html#setIndirect)

## Causal contrast

The portable control retains the current address expression:

```text
visibleIds[bucketBase + instanceIndex]
command.firstInstance = 0
```

The feature lane uses:

```text
visibleIds[instanceIndex]
command.firstInstance = bucketBase
```

The feature lane omits `bucketBase` from the merged vertex layout. This removes
one four-byte vertex input and one integer addition/dependency. It does not
change the survivor-list partition, matrices, bounds, cull order, geometry
payload, index ranges, visible counts, command count, render-object count,
lighting, camera, target, fragment material, or draw order. Both lanes retain
32 indexed indirect draws, one static `BundleGroup`, and the same two-dispatch
fixed-slice culling algorithm.

The feature lane is refused unless
`renderer.hasFeature('indirect-first-instance')` is true. The zero-
`firstInstance` implementation remains the portable path. Introducing
instance-stepped vertex attributes would require a fresh correctness audit,
because `firstInstance` offsets those attributes as well as the WGSL builtin.

## Development screen

The screen fixes Three.js 0.185.1, 65,536 objects, 32 medium indexed geometry
buckets, the baseline layout, a 1280 by 720 target, device pixel ratio 1,
reversed depth, and the repository's fixed camera and seed. It evaluates two
visibility levels:

- 99%, the high-dose cell for vertex invocations;
- 20%, a dose-response control using the same representation.

Six repetitions alternate portable/feature and feature/portable order. The
visibility order is balanced across repetitions. Each lane receives 300 warmup
frames and 240 retained frames. GPU render timestamps are the primary metric;
total GPU-pass time is secondary because culling is intentionally unchanged.
Each repetition-level estimate is the feature-lane median minus the adjacent
portable-lane median. For an even sample count, the median is the arithmetic
mean of the two central values. Frames are repeated measurements, not
independent replicates.

Before timing each lane must pass full survivor membership and indirect-command
readback. All five command words are checked for every bucket, including the
lane-specific `firstInstance`. Stable full-frame RGBA8, depth32float, and RGB24
object-ID captures must match exactly across the two lanes within each workload
cell. Page, console, or WebGPU validation errors reject the screen.

## Predeclared decision

Negative deltas favor indirect `firstInstance` addressing.

Promote the mechanism to a tightly interleaved, render-only candidate protocol
only if the 99% cell meets every condition:

- median render improvement is at least 0.05 ms and at least 10%;
- at least five of six repetition estimates are negative;
- neither order position reverses the direction;
- the 20% cell has no regression greater than 0.02 ms or 5%;
- every exact-output, command, timestamp, and environment check passes.

Stop treating the mechanism as a standalone research direction if the 99%
cell improves by less than 0.02 ms or less than 5%, fewer than four of six
estimates are negative, or the dose-response control exceeds its regression
bound. An intermediate result may receive one exact repeat of this unchanged
screen; if it remains intermediate, it is stopped rather than threshold-tuned.

Regardless of outcome, the result is scoped as a feature-gated Three.js
integration ablation. It cannot attribute the broader difference from other
libraries or versions, which also vary geometry topology, storage topology,
materials, bindings, and render-object count.

## Follow-on direction

If this narrow screen stops, the next experiment is an untimed exact
visible-contributor oracle for the high-overlap scene. That experiment asks
whether removing fully occluded frustum survivors before vertex shading has
enough render-only headroom to justify a conservative previous-depth hierarchy.
It is a separate protocol and does not reinterpret this screen.
