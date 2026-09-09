# TSL deformation intervals over the existing packed BVH

Status: predeclared bounded screen; no passing result is implied. Pin this file,
fixture source, generated shaders, Three version and three-mesh-bvh revision in
the first report. Changes after a run create a new protocol/source version.

## Question and scope

Can a compiler of an existing, pure native TSL expression DAG generate useful
deformation bounds directly over immutable rest BVH boxes, while preserving the
upstream packed layout and unchanged closest-point traversal? The candidate
evaluates each node independently in one compute dispatch. It neither reads
deformed vertices to construct bounds nor reduces child bounds bottom-up.

The mechanism is established interval bounding applied to an existing Three
graph/interface. It is not invented interval arithmetic, a new BVH algorithm,
automatic differentiation, first GPU refit, full physics, or continuous collision
detection. [PRIOR_ART.md](./PRIOR_ART.md) records overlapping implementations.
Its earlier refit proposal is historical; this protocol specifies the stronger
GPU-to-GPU baseline and the interval-specific enclosure requirement below.

Use fixed topology, one visible identity-transformed mesh, one vec4f position
attribute and at most 4,096 triangles. Construct the pinned upstream hierarchy
once, including its TLAS/cluster packing. All lanes retain identical indices,
topology, queries, transforms and GPU-authored vertex positions. Rendering, when
shown, consumes those same positions. CPU initialization and validation may read
data; steady-state bounds construction and queries must not download vertices.

## Lanes and workloads

1. **Exact refit:** deform actual vertices on the GPU, bound triangle leaves,
   then union parents bottom-up through the TLAS. Use the existing refitter and
   allocations; do not insert CPU downloads, rebuilding, or extra synchronization.
2. **Instant interval:** evaluate the compiler's bounds on every immutable rest
   node box with the time interval `[t,t]`, then query the same current vertices.
3. **Swept interval:** bound each rest box over `[t0,t1]` once, then reuse those
   boxes while current vertices and queries advance through that window.

Swept boxes are broad-phase enclosures, not time-of-impact answers. Traversal
still intersects the current triangles. Exact refit remains the correctness and
timing baseline at each subframe, including when one swept update is amortized.
A stale-rest-bounds lane is a deliberately failing validation control, never
the performance comparator.

Use the three native graph families in `engine.js`, with rest coordinates p:

- `wave`: `(p.x, p.y + .65*sin(2.1*p.x+t), p.z)` on a 4-by-4 plane. Coarse
  rest boxes span more than one phase period; narrow-phase examples alone would
  miss this range-handling requirement.
- `ripple`: `(p.x, p.y + .45*sin(5*p.x+t)*cos(4*p.z-t), p.z)` on that plane.
  Wide spatial phases and products make this an intentional loose-bounds case.
- `twist`: rotate `(p.x,p.z)` by `1.2*p.y+.6*t`, retaining p.y, on a torus of
  major radius 1 and tube radius .35. Correlated products and spatial rotation
  test interval dependency inflation on a closed, nonconvex surface.

The first canary is 16 segments and 128 queries for one family. It checks
execution/correctness plumbing only and is not the matrix, timing result or
practical-superiority gate. Then use 32 segments (torus radial segments 16) and
64/1,024 queries, still below 4,096 triangles. Seed is 4242. Retain the exact
query-generation manifest; augment random probes with explicit geometric cases
before the full correctness matrix. An affine translation/shear may be a cheap
CPU compiler test, but do not replace a difficult family after seeing results.

Instant validation times are `0`, `.7`, `2.4`; corresponding swept windows are
`[0,.75]`, `[.75,1.5]`, `[2.4,3.2]`. These times do not themselves prove a
reversal or maximum-deformation case: identify actual extrema in the source and
retain the designed interior-extremum negative control. At least one interval
must contain a temporal extremum missed by the union of its two endpoint boxes.
Keep every family, including losses. One family/time-index is one cell; nine
cells are the full screen. Any extra diagnostic stays separately labeled.

## Correctness before timing

Run the independent CPU self-tests first. For each cell:

- Preserve exact node metadata words 6/7, indices and transforms. Verify actual
  shared-buffer identities and zero steady-state vertex-readback bytes.
- Read actual GPU vertices and bounds at both endpoints and seven fixed interior
  fractions `1/8` through `7/8`. These validation times are not fitting samples
  or inputs used to construct the interval. Retain the arrays or hashes plus a
  reproducible raw-data artifact. No CPU deformation formula supplies the oracle.
- `validateBounds` must independently enclose every referenced triangle vertex
  at every ancestor, including TLAS and cluster nodes. Require all packed
  triangles to be covered. Exact refit additionally requires child-box nesting.
  Interval lanes use `requireChildContainment:false`: padded child-box escapes
  are reported diagnostically, but any descendant vertex escape still fails.
  A deliberately too-tight interior ancestor must fail this relaxed-mode check.
- At each validation time, run 64 deterministic closest-point probes spanning
  near-surface, far, edge/corner, inside-closed-mesh and deformation-extremum
  positions. Compare unchanged upstream traversal with `bruteForceQueries`
  over actual GPU-readback vertices. Distances are unsigned. Verify the returned
  triangle really produces the reported closest point; different tied IDs are
  allowed. At multiple equally close points, validate objective value and surface
  membership rather than require one arbitrary point from the CPU tie-breaker.
- Fix tolerances from rest-scene diagonal `L`: squared-distance error at most
  `1e-5*L*L + 2e-4*oracleDistanceSq`; point-on-reported-triangle residual at most
  `1e-4*L`. Report maximum and p95 errors, not just pass counts. Bounds must
  enclose actual f32 coordinates with zero tolerance; roundoff accommodation
  belongs in the generated bounds, not a permissive validator.
- Stale bounds must fail a designed displaced-geometry query. Endpoint-only
  bounds must fail the designed interior-time enclosure control. Otherwise the
  fixture has not demonstrated the respective need.

Enclosing vertices encloses the rasterized/current collision triangles: those
triangles interpolate their deformed vertices. Do not confuse them with the
nonlinear image of every point on an undeformed triangle.

Finite sampling alone does not prove enclosure for all times. Review interval
rules and supported domains, including transcendental extrema and f32 rounding.
Reject unknown/opaque nodes, unsafe division and nonfinite intervals explicitly.
Without a justified outward error bound for GPU operations, describe the result
as **sample-validated enclosure**, not mathematically certified conservative
bounds or guaranteed continuous-time coverage.

## Cost and completion boundaries

Measure 64 and 1,024 queries per subframe. A batch advances eight evenly spaced
subframes including both window endpoints. Exact/instant lanes rebuild bounds
each subframe; swept builds once per batch. Also report non-amortized update cost
and dispatch count, so reuse does not conceal slower individual queries.

Allocate/compile first, perform one fixed untimed warmup batch per lane, then six
paired blocks using each of the six lane-order permutations exactly once, with
a fixed seeded order. All lanes receive identical deformation, queries, queue
completion boundaries and output consumption. If the watchdog leaves a block
incomplete, retain it as incomplete; do not summarize it as the full design.

Primary timing is CPU-observed completed deformation -> bounds -> query batch
latency, ending at GPU completion. Label this wall time, not GPU time. Record
encoding/submission time separately. If timestamp-query is available, additionally
measure GPU deformation, bounds and traversal with equivalent boundaries; do
not require browser flags or substitute submission time when it is unavailable.
Validation readbacks are outside timing. Rendering is excluded from the numerical
screen and reported separately for any later demo.

Report per cell/query scale: conventional median, nearest-rank p95, paired
candidate-minus-exact differences, update/query/full-batch costs, bound-volume
inflation at leaves and internal nodes (handle zero-volume boxes explicitly),
and visited-node/triangle-test counts only if separately instrumented traversal
is available. Do not change the production query kernel just for timing.

One GPU task at a time; a hard 45-second cap per cell, no automatic retries or
load escalation. Stop/dispose on validation error, nonfinite output, device loss
or timeout. Retain failed artifacts. This is a single-session diagnostic.

## Decision

Any enclosure, metadata or closest-query failure kills the claimed supported
domain, regardless of speed. Compilation success alone is insufficient. If a
node update is faster but traversal inflation loses the full-batch comparison,
record a negative result for that workload.

A useful initial signal requires correctness in all nine cells and at least
20% lower median completed batch latency versus exact refit in two families at
1,024 queries, with no more than 10% regression in the third. Instant and swept
reuse results stay separate; a swept win does not imply a per-frame interval
win. This gate prioritizes a meaningful demo, not statistical confirmation.
Failure means no performance finding from this screen; retain any narrower
compatibility result honestly. Broader device/browser or performance claims need
separately authorized, versioned replication after this bounded decision.
