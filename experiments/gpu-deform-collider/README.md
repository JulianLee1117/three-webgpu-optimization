# Native shader geometry research

This direction produced a concrete correctness finding and two restricted TSL
compiler prototypes. They have different evidence and should be presented separately.

## Verified finding: a triangle-distance error in three-mesh-bvh WebGPU

The current triangle helper can choose the wrong edge of an obtuse triangle.
For A=(0,0,0), B=(2,0,0), C=(-1,.001,0) and P=(1,-.002,0), the GPU reports
distance about 1.000002 instead of .002: roughly 500 times too large. Reordering
the same three vertices can change the result. This is a **distance error**,
not a performance speedup, and belongs to the three-mesh-bvh WebGPU helper,
not Three.js core.

The isolated GPU reproduction uses 200 cases: the stock helper fails 10; the
current correction passes 200. The simpler integer-coordinate example yields
squared distances 5 versus 4. Two CPU references agree. All numbers come from
the pinned Three 0.185.1 / three-mesh-bvh 0.9.15 implementation; this is not a
claim about every version or every GPU.

- [Finding, evidence and limitations](TRIANGLE-QUERY-FINDING.md)
- [Upstream source patch](../../patches/three-mesh-bvh-closest-point.patch)
- [Isolated GPU probe](triangle-probe.js)
- [CPU counterexample and vertex-order tests](triangle-query-counterexample.mjs)

```sh
npm ci
npm run demo:triangle-query
npm run check:triangle-query
npm run check:triangle-query:fixed
```

The interactive page starts idle. Its comparison executes one small GPU dispatch
and releases the device; controls change shape thickness and vertex order.
The original library stays installed unchanged. The fixed CLI uses an explicit
experiment-local source overlay that changes only the current triangle helper.
The standalone interactive page runs both helpers side by side without an overlay.

## Experimental compiler: native TSL graphs to bounds and derivatives

`tsl-intervals.js` interprets an existing native scalar/vec3 TSL expression graph
and emits a second TSL graph for its interval bounds. `tsl-derivatives.js` emits
the analytic time derivative. They support pure arithmetic, sin/cos and vector
construction/swizzling, with explicit input boundaries. Arbitrary functions,
mutable variables, texture reads, loops and general neural models are unsupported.

The bounds prototype maps every immutable rest BVH box independently through the
same GPU deformation graph used for rendering. It retains the upstream packed
hierarchy and query API. One bound dispatch replaces 8–10 standard refit passes in
these fixtures, or covers multiple time samples. Steady construction/query work
does not read vertices back to the CPU. Validation deliberately does.

```sh
npm run demo:shader-bounds
node scripts/run-gpu-deform-collider.mjs --intervals --throughput --corrected --family=wave --segments=32 --queries=64
node scripts/run-gpu-deform-collider.mjs --derivatives --corrected --family=twist
node scripts/analyze-tsl-bounds.mjs
```

The six corrected screen cells pass the tested enclosure/query checks, including
temporal samples and failing stale/endpoint-only controls. Performance is mixed:
looser boxes can increase traversal cost. [The results](RESULTS.md) do not support
a general speedup or algorithmic novelty. Swept bounds are a broad-phase
building block, not continuous collision detection. Trig and arithmetic padding
are empirical, not a formal floating-point guarantee.

The derivative compiler passes 1500 independent analytic CPU cases. Generated
GPU derivatives also passed checks against finite differences of actual GPU
deformation for wave, ripple and twist at three times. No contact-dynamics demo
or differentiable rasterizer is implied by those derivative checks.

Prior art includes [automatic vertex-shader bounding (2009)](https://fileadmin.cs.lth.se/graphics/research/papers/2009/tcu/a19-hasselgren.pdf),
[BD-Tree (2004)](https://graphics.cs.cmu.edu/projects/bdtree/),
[shader autodiff for deformation (2021)](https://www.davepagurek.com/blog/realtime-deformation/),
and [an existing GPU BVH draft implementation](https://github.com/gkjohnson/three-mesh-bvh/pull/853).
The possible contribution is the native TSL integration. These are established
mathematical techniques, and the standard refitter is a comparison implementation.

Read [the protocol](PROTOCOL.md), [the narrower implemented screen](THROUGHPUT-SCREEN.md)
and [the prior-art audit](PRIOR_ART.md). Raw reports and generated proof artifacts
remain local under `results/development/gpu-deform-collider`; hashes identify them.
Failed prototypes and their reports are retained.
