# Delayed-depth contact: CPU feasibility screen

**Result: no-go for the tested general adapter.** One global constant-velocity
estimate helps simple translation but fails the required reversal, independent
motion and deformation controls. This is a retained negative result, not a GPU
collision feature or a neural-depth improvement.

[reference.mjs](reference.mjs) contains the complete protocol, analytic scenes,
adapters, metrics, gates and CPU self-tests. It uses no packages, network, GPU,
workers or model downloads. From the repository root:

```sh
node experiments/depth-contact/reference.mjs --self-test
node experiments/depth-contact/reference.mjs
```

The default full screen prints JSON to stdout. `--quick` runs a smaller diagnostic
and is explicitly not the complete gate. Optional single-case flags are documented
in the source. The commands do not write files themselves.

## Test boundary

Eight deterministic analytic heightfield scenes each run at depth rates of 5 and
10 Hz. Depth is 64x64 with 80 ms delivery delay; probes run on a 60 Hz clock with
one simulated second of warmup and four measured seconds. There are 288 identical
vertical point probes per frame, or 69,120 measured queries per method per cell.
Simulation time advances without wall-clock sleeps.

All methods use captured camera coordinates and the same conservative edge-aware
sampling. The candidate estimates one global translation from two depth centroids
and mean heights; it receives no object IDs, true velocities or future samples.
Four prespecified hold/expiry/smoothing baselines compete. For the kill/go decision,
the baseline with the fewest false contacts among those within five percentage
points of hold's miss rate is chosen using oracle labels. This intentionally strong
retrospective comparator is not a deployable selection algorithm.

The gate requires at least 50% fewer false contacts where sufficient baseline
errors exist, at most five percentage points of extra misses and no more than
0.005 world units of additional mean residual penetration. Edge and post-reversal
strata must pass separately; static/camera-only controls and bounded CPU adapter
cost also apply. Pooled improvement cannot hide a failed scenario.

## Recorded outcome, September 9

The full 16-cell screen and all ten self-tests completed. The decision is
`no-go-for-general-adapter`, with six failed scene/rate cells:

- Motion reversal fails at both rates. Across the full case, false contacts fall
  by about 66% and 63%, but immediately after turns they increase **394 to 628** at
  5 Hz and **204 to 418** at 10 Hz. At 5 Hz, post-turn misses also increase by
  9.85 percentage points and mean residual penetration by 0.0112 world units.
- Opposing surfaces show no false-contact reduction: **255 to 255** at 5 Hz and
  **74 to 74** at 10 Hz. A single centroid velocity cannot represent both motions.
- Deformation reduces false contacts **598 to 395** and **424 to 228**, but the
  33.9% and 46.2% reductions fail the 50% requirement at both rates.

Linear translation, vertical translation and edge sweep pass the motion gates at
both rates, and the static/camera-only controls pass. These narrow successes do
not overturn the general-adapter rejection. CPU cost gates passed in this local
run; those timings exclude depth generation, GPU execution, transfers, neural
inference and browser integration.

The immutable local report is
`results/development/depth-contact/2026-09-09T17-44-08.425Z/report.json`, with an
adjacent SHA256 sidecar. It retains all method counts, denominators, edge/reversal
metrics, approximate CPU costs, self-tests and runtime metadata. Report SHA256:
`fb97cc2648b1b4b62a80a33d8fe12ecf4ca54b2ba98bf4bf63a92acd7d338e51`.
The source was hashed before and after execution and remained unchanged:
`1f9f59f4c3fe72534974e84ddaba13008e5ab57c8b523c4e9788aae725b30d1b`.
Ignored local reports may be absent in a fresh clone; rerunning the CPU command
reproduces deterministic contact counts, while local timings will vary.

This screen models top-surface membership and one upward correction, not full
rigid-body dynamics, vertical walls, occlusion tracking or realistic neural-depth
noise. The many deterministic queries are not independent statistical samples.
No WebGPU/frame-pacing measurement or real-camera demo was run for this direction.
A new method would need its own mechanism and controls; the current result does
not justify porting this global predictor into a general Three.js component.
