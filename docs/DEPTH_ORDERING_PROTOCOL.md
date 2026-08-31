# Coarse depth-ordering protocol

## Research question

Can a per-bucket, coarse GPU-generated front-to-back bin-block traversal materially reduce Three.js WebGPU render-pass time when native draw count, retained render-object topology, geometry, material, visible membership, and compute submission count are held fixed?

The primary causal comparison is front-to-back versus reverse depth-bin traversal within each of the 32 bucket draws. Those lanes execute the same four compute kernels and shader operations; one boolean uniform changes the traversal of eight physical depth bins. Bucket draw order remains fixed, so this is not a global instance sort. Atomic fixed-slice compaction is a simpler contextual baseline, not the sole basis for attributing a render-pass difference to order.

## Fixed candidate workload

- Three.js 0.185.1 and the repository's pinned dependency lock;
- 65,536 objects, 32 indexed geometry buckets, and 99% target visibility;
- one merged indexed geometry, one material, one mesh in one static `BundleGroup`, and 32 native indexed indirect draws;
- a 1280 by 720 viewport with device pixel ratio 1, fixed camera, fixed lighting, and antialiasing disabled;
- Three.js `reversedDepthBuffer` enabled for every compared lane;
- eight physical depth bins using sphere-nearest view depth, `-viewZ - radius`;
- one high-overlap layout and one low-overlap negative control.

The two layouts preserve object IDs, bucket membership, visibility, Z positions, rotations, scales, bounds radii, geometry, material, and random-number consumption. Only the visible X/Y placement changes: the high-overlap layout uses extents 1.5 by 0.9, while the low-overlap control uses 75 by 40. That intervention also changes screen-tile and coverage distribution, so absolute cross-layout differences are not attributed to overlap alone. The depth-bin range is derived once from the expected visible spheres and recorded in the scenario manifest.

Reversed depth is a correctness requirement for this matrix, not a strategy-specific intervention. With the original 0.1-to-1000 conventional depth mapping, the deliberately dense overlap produced order-dependent ties after depth quantization. Reversed depth removes that ambiguity on the tested backend and allows exact color, depth, and object-ID parity to remain a fail-closed prerequisite. A run is rejected unless the page reports the renderer option as active.

## Compared lanes

| Lane | Compute dispatches | Compute submissions | Render objects | Native draws | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| Atomic fixed-slice | 2 | 1 | 1 | 32 | Simpler unsorted compaction context |
| Eight-bin front-to-back | 4 | 1 | 1 | 32 | Candidate ordering |
| Eight-bin reverse | 4 | 1 | 1 | 32 | Same-work adverse-order control |

The depth-binned lanes execute:

1. reset 256 bucket/bin records, 32 indirect counts, and overflow state;
2. classify and count 65,536 objects;
3. compute one ordered eight-bin prefix per bucket;
4. scatter deterministically into the fixed bucket slices, using one invocation per bucket to scan that bucket's contiguous object IDs in ascending order.

The fourth kernel still performs one classifier lookup and at most one offset increment and survivor write per object, but its ownership is bucket-local rather than object-parallel. This makes within-bin order stable without adding a pass or changing the front/reverse work. Its dispatch shape is 32 work items; the other three dispatch shapes are 256, 65,536, and 32 work items.

Counts, write offsets, starts, and explicit padding share one 16-byte bin record. This keeps the largest compute stage at eight storage-buffer bindings, the WebGPU baseline per-stage limit. The runner records the device limit and rejects values below eight. Both ordered lanes retain zero `firstInstance` in every native indirect command.

## Correctness and representation gates

Timing is refused unless all applicable checks pass:

- independent CPU frustum membership equals the predetermined visible subset;
- GPU survivor membership and per-bucket canonical SHA-256 commitments are exact;
- every five-word indexed indirect command has the expected index count, instance count, cumulative first index, zero base vertex, and zero `firstInstance`;
- all eight per-bucket bin counts, ordered starts, totals, and survivor blocks match an independent CPU reconstruction;
- every ordered snapshot records a traversal-normalized SHA-256 commitment over the unsorted object-ID sequence inside each physical bucket/bin. The parity and preflight snapshots must match, as must the front/reverse pair within every repetition and layout;
- overflow is zero;
- the ordered lanes expose exactly one mesh, geometry identity, material identity, and bundle-record callback;
- the static bundle is recorded once before timing and is not rebuilt during warmup or measurement;
- all four dispatches execute in one Three.js compute submission;
- the browser reports no page, console, or WebGPU validation error.

Every candidate preflight computes one snapshot, validates that exact snapshot, and then renders it twice into offscreen targets. The two captures must have identical full-frame SHA-256 commitments for RGBA8 color, depth32float, and RGB24 object ID. Readbacks must each contain exactly 3,686,400 bytes. The object-ID pass uses zero as background and object ID plus one as the encoded value; it must contain nonzero coverage and no out-of-range or non-visible ID. The same exact render identity must recur across all modes and repetitions within each layout cell.

Each bucket has one scatter owner, so its atomic write offsets advance in ascending object-ID order and physical-bin sequences are algorithmically stable. Traversal-normalized sequence commitments make any observed front/reverse difference fail closed, and exact validation payloads must recur at preflight, timing start, and post-trial. Timed frames are not read back because that would perturb measurement; the estimand remains a repeated coarse bin-block traversal, not a global sort.

## Trial order and measurements

The matrix contains all six permutations of the three modes. Each repetition evaluates both layouts, alternating which layout appears first. This produces exactly 36 trials and balances every mode across each within-layout order position.

Each trial uses 300 warmup frames followed by 240 measured frames, for 8,640 retained frame rows. GPU timestamp evidence separates compute-pass, render-pass, and summed-pass duration. CPU evidence separates common update, compute submission, render submission, total submission, and frame-body duration. Summed GPU passes or CPU-plus-GPU accounting are not described as presentation latency.

Scenario, geometry, source, lockfile, environment, parity, validation, trial-order, lifecycle, and telemetry evidence are hash-bound into the run artifacts. The runner persists each timing-end completion invariant, and directory analysis independently revalidates all three exact-validation captures, their semantic commitments, and the lifecycle invariant. Candidate status additionally requires a clean tracked source tree that remains unchanged through teardown.

## Preregistered interpretation

All timing thresholds operate on the median of the six repetition-level paired deltas, using front-to-back minus the comparator. Negative values favor front-to-back. The depth-ordering direction remains supported on the first device only if all of the following hold:

- in the high-overlap layout, the median front-to-back versus reverse GPU-render delta is at most -10% or at most -0.10 ms;
- exactly six high-overlap pairs are present and at least five have a strictly negative GPU-render delta. Each absolute front-to-back order position and each absolute reverse order position must contain exactly two pairs with a negative median delta. The front-first and reverse-first relative-order strata must each contain exactly three pairs with a negative median delta;
- in the low-overlap control, the absolute median front-to-back versus reverse GPU-render delta is strictly less than both 10% and 0.10 ms. Crossing either bound is a material control difference;
- as a contextual whole-mechanism comparison, front-to-back improves total timestamped GPU pass time versus atomic fixed-slice by at least 10% or 0.10 ms;
- in the low-overlap contextual whole-mechanism comparison, the front-to-back lane's median total-GPU-pass regression versus atomic fixed-slice is no greater than 5%;
- every correctness, representation, lifecycle, provenance, and exact render-parity gate passes.

The analyzer reports front-to-back versus reverse under `causalContrasts`. It reports both ordered lanes versus atomic fixed-slice separately under `contextualWholeMechanismComparisons`, because those comparisons change the compaction algorithm and dispatch count in addition to ordering. Failure of the reverse-order contrast rejects an ordering mechanism claim even if front-to-back happens to beat atomic compaction. Failure of the low-overlap control rejects a clean overlap-dependent interpretation. Results from one GPU/backend remain single-device evidence until reproduced on a materially different GPU family.
