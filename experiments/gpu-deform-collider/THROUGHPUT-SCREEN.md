# Prospective throughput screen, 9 September 2026

This narrower implementation screen follows the passing small canaries. It does
not claim completion of every proposed gate in PROTOCOL.md. Source hashes and
all executed module hashes are retained per run; failed runs stay in place.

Run wave, ripple and twist at 32 segments and 64 / 1024 queries. The two planes
have 2048 triangles; the torus has 1024. One browser/device at a time; 45-second
watchdog per run. Device defaults plus the standard optional timestamp-query
feature; no experimental flags. No render work enters timing.

Before timing, validate exact and instantaneous compiled bounds at 0, .7 and
2.4. Validate compiled swept boxes and unchanged upstream queries at nine times
per window, for [0,.75], [.75,1.5], [2.4,3.2]. Test every triangle vertex against
every ancestor with ZERO containment tolerance. Query oracle uses the first 64
probes: 16 declared world-space points and 48 seeded random probes. These are
not asserted to be exact on-surface features at every animation time. Compare
exhaustive CPU closest points over actual GPU vertex readbacks, with returned
triangle/point/distance consistency. Allow 2e-4*(1+distanceSq) squared-distance
error and 2e-4 componentwise point error; preserve raw values for stricter audits.
Do not reject valid equidistant triangle ties. Save raw readbacks, original
geometry/packing, probes and generated shader source to a hashed proof artifact.

Stale rest bounds at .7, and the union of exact endpoint bounds at 0 and 3.2
used at 1.6, must be rejected by the geometry oracle. This checks enclosure
failures; it does not require every random query to miss. A missing failure
halts timing instead of silently removing the control.

Timing uses three lanes: standard bottom-up GPU refit each step, compiled bounds
each step, and one compiled swept update reused across 16 steps over[.7,1.5].
Swept preparation includes its extra initial deformation, so the candidate pays
for that redundant setup. Each lane updates the same current vertex positions
and runs the same upstream queries at all 16 times. Six permutations rotate lane
order. One batch per lane warms pipelines before 18 measured batches. Report
raw data, CPU submission time, queue-completed wall time and optional GPU
timeline duration. The latter INCLUDES submission gaps, not just active kernel
time. Do not replace completed-work comparisons with bounds-only timings.

This single-session screen establishes feasibility and identifies regressions.
Any apparent benefit needs a frozen repeat, adversarial workloads and a stronger
alternative refitter comparison before general performance claims. Padding is
empirical, and temporal samples are not a floating-point containment proof.
The implementation is a restricted TSL integration of established interval
bounding; mathematical novelty is not claimed.
