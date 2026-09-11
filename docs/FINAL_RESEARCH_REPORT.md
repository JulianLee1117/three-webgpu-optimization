# Final Three.js / WebGPU research report

Research closed September 11, 2026. This report summarizes the retained findings;
it does not authorize new benchmark attempts or supersede original protocols.

## Overall conclusion

The research produced **four concrete correctness findings**, a useful automatic
procedural-material filtering integration, several working authoring demos, and
scoped performance results. It did **not** establish a new graphics or physics
algorithm, a general Three.js speedup, or an industry-level breakthrough.

The strongest technical result is the independently reproduced wrong-edge
selection in the WebGPU triangle-distance helper. The strongest visible reusable
capability is automatic pixel/exposure filtering of a restricted native TSL graph
class. Light Vault is the most complete optical creation-and-play workflow.
These are different kinds of contributions and should remain distinct.

The tested versions are pinned in each experiment. “Confirmed” below means
reproduced in those versions and fixtures. It does not mean accepted upstream,
first discovered worldwide, fixed in a released library, or validated on every
device. No upstream acceptance is claimed by this closing report.

## 1. Triangle-distance queries can choose the wrong edge

**Classification: confirmed correctness finding and local correction.**

In `three-mesh-bvh` 0.9.15, the TSL/WebGPU closest-point helper can select the
wrong edge of an obtuse triangle. Reordering the same three vertices can change
the answer even though the geometry is unchanged. For a thin, nondegenerate
triangle, the measured distance is about **1 instead of 0.002: a 500-fold
overestimate**. This is an error ratio, never a speedup.

The isolated original GPU helper fails 10 of 200 fixed cases; the corrected
helper passes 200/200. Two independent CPU references agree on every case.
The original and corrected probes differ in one served helper module. A separate
10,000-case CPU sweep found 249 original mismatches in that chosen distribution.

**Practical significance:** closest-point queries support contact, picking,
distance fields and geometry tools. The counterexample exposes a primitive-level
failure; effects in a complete production collision system were not measured.
The correction uses established Voronoi-region geometry. Arbitrary scales and
degenerate inputs are not universally validated by this finite test suite.

- [Geometric counterexample, reproduction and limits](../experiments/gpu-deform-collider/TRIANGLE-QUERY-FINDING.md)
- [Minimal review patch](../patches/three-mesh-bvh-closest-point.patch)
- [Portable original/corrected GPU evidence](../experiments/gpu-deform-collider/evidence/README.md)

## 2. Three r186 OIT loses transparent foreground alpha

**Classification: confirmed correctness finding and one-expression correction.**

A 50%-transparent red plane over a transparent clear produces RGBA
`[127, 0, 0, 0]` with the pinned stock OIT pass, instead of ordinary blending's
`[127, 0, 0, 127]`. The RGB is present while the foreground coverage is absent.
The returned alpha uses only the background beauty alpha.

Including foreground coverage with
`1 - revealage + backgroundAlpha * revealage` fixes all five controlled fixtures.
Across 15,360 retained output pixels, corrected RGBA is byte-identical to ordinary
blending; stock and corrected OIT RGB are also identical.

**Practical significance:** coverage matters when placing a transparent result
over another image or page. The opaque-background control explains why an
ordinary opaque canvas can hide the problem. This is standard coverage algebra,
not a new transparency method. Mixed-color ordering, transmission, MSAA, HDR and
arbitrary pass chains were not validated.

- [Finding, pinned source, correction and five fixtures](../experiments/oit-alpha-correctness/README.md)
- [Immutable evidence manifest](../experiments/oit-alpha-correctness/evidence/manifest.json)

## 3. Rotated Gaussian splats can be missed by CPU picking

**Classification: confirmed correctness finding and conservative diagnostic fix.**

Three r186's Gaussian-splat raycast uses the maximum covariance diagonal to
bound an ellipsoid. After rotation, that value can underestimate its longest
axis. Both the outer object sphere and the per-splat sphere can reject a ray
that actually intersects the ellipsoid.

The unmodified raycast misses two of six fixed cases. Enlarging only one of the
two bounds is insufficient. A combined conservative trace-bound correction and
a control with both sphere rejections disabled agree with an independent
ellipsoid-intersection oracle in all six cases.

**Practical significance:** a user can click through part of an elongated,
rotated splat because the broad-phase bound is too small. The exact fixture
establishes query false negatives, not their frequency in real captures or a
rendering defect. Identity object transforms were tested. The trace bound is a
simple conservative diagnostic, not an optimized general-purpose final patch.

- [Reproduction, covariance example and pinned source](../experiments/splat-query-correctness/README.md)
- [Closing CPU reproduction](../experiments/splat-query-correctness/evidence/2026-09-11T07-07-30.378Z/report.json)

## 4. ZipDepth export changes the checkpoint's computation

**Classification: reproduced model-export fidelity finding.**

The pinned ZipDepth Base NPU exporter replaces learned spatial attention pooling
with an unweighted average. Its post-mutation shape check does not detect that
the exported model now computes something different from the original checkpoint.
Removing that substitution restores agreement with an independently preserved
PyTorch reference.

All six corrected procedural CPU fixtures pass `rtol=1e-4, atol=1e-5`; all six
original exports fail. In the retained 384² browser gradient case, 147,339 of
147,456 original output values fail the tolerance. Every corrected value passes,
with maximum absolute reference difference about `3.67e-7`.

**Practical significance:** exporting a model can silently alter its semantics
while retaining valid shapes and successful execution. The corrected graph runs
in ORT WebGPU and renders through Three-owned buffers. This establishes fidelity
to the checkpoint, not improved real-world depth accuracy, faster inference or
better generalization. No representative image dataset was evaluated.

- [Reproduction and exact limits](../experiments/zipdepth-export/README.md)
- [Minimal exporter patch](../experiments/zipdepth-export/preserve-global-context.patch)
- [Closing retained reports](final-evidence/README.md)

## 5. Signal Loom: automatic native TSL pixel/exposure filtering

**Classification: verified reusable integration with restricted scope.**

The compiler transforms finite trigonometric native TSL color expressions into
their pixel/exposure integral. It expands products jointly, preserving genuine
low-frequency beats between high-frequency patterns instead of filtering each
factor independently. The original graph remains unchanged.

The fixed screen passes 12/12 correctness cases and 9/9 nonmagnified utility
cases across three programs. In those nine cases, RGB RMSE falls from
`0.061812–0.256219` under point sampling to `5.33e-7–2.97e-6`: a
**99.9976%–99.9997% error reduction**. It also beats the matched finite 64-sample
control in those cases. The separately authored analytical shader has comparable
accuracy. At magnification, the numerical supersampling control is more accurate;
even point sampling beats the compiler in one already-low-error cell.

**Practical significance:** procedural patterns can remain visually stable when
small or exposed over time without hand-authoring an integral for each material.
The exact class is constant-amplitude trigonometric polynomials with affine
phases. The lit perspective demo uses local footprint approximations. Arbitrary
noise, textures, nonlinear phases, geometry motion blur and general shader
support are not established. **No speed advantage was measured.** Filtering
theory and compiler precedents are established prior art.

- [Implementation and runnable demo](../experiments/footprint-filtering/README.md)
- [Full comparisons and retained pixels](../experiments/footprint-filtering/RESULTS.md)
- [Prior art](../experiments/footprint-filtering/PRIOR_ART.md)

## 6. Working creation tools and demos

These implementations demonstrate capabilities; they do not establish new
underlying mathematics.

- **Light Vault:** one fixed phase plate forms different images at different
  distances. Actual computed GPU intensity determines captures. Users can draw a
  pair, fit a new plate and exchange a checked seal file. The retained run passes
  61 browser checks and 23 CPU tests, including export/reload/play. Its HI/BY fit
  took about four seconds. Established multi-plane holography and browser CGH
  are direct precedents. [Demo, reusable GPU module and evidence](../experiments/light-vault/README.md).
- **Sunprint:** draw a mark, fit a refracting surface and independently propagate
  photons through that geometry in native TSL. The target image does not enter
  the forward renderer. This is interactive inverse-optics integration, with
  substantial freeform-optics prior art. [Experiment](../experiments/caustic-sketch/README.md).
- **Constrained motion editing:** edit targets of two native animated TSL
  programs while retaining motion and satisfying sampled obstacle constraints.
  Eight final GPU lanes pass and 36 silhouette comparisons match the numeric
  geometry. General continuous collision avoidance and new optimization
  mathematics are not established. [Results](../experiments/constraint-editing/RESULTS.md).
- **Trainable TSL:** generate gradients from a forward native graph and fit
  procedural/neural fields. Four field/seed cells pass the declared gate; a
  finite-difference baseline performs similarly. [Results](../experiments/trainable-tsl/RESULTS.md).
- **MatterForge, Carry the City and local elasticity:** material casting,
  motion transfer and captured-object deformation work within authored fixtures
  and documented contact/mechanical approximations. They are retained tools,
  without a demonstrated general simulation or graphics breakthrough.
  [Experiment index](../experiments/README.md).

## Performance results and rejected directions

Fixed-ownership visibility compaction produced scoped single-device results:
against the historical Three Blocks 0.10 path, timestamped GPU-pass reductions
ranged from 12.1% to 33.1% across the declared visibility cells and replications.
That historical path was removed in 0.11. Against the 0.11 coalesced diagnostic,
compute is cheaper but rendering is slower; near-full visibility approaches a
crossover. This is not a general comparison with the supported stock schedule.
The retained one-versus-many representation study separately demonstrates a
CPU submission cost at 32 and 128 buckets. [Exact results and conditions](CANDIDATE_RESULTS.md).

The indirect `firstInstance` studies show material aggregate same-device timing
signals, but **do not satisfy their required two-matrix confirmation rules**.
Failed nuisance-interaction gates are not erased by favorable medians. Frozen
front-to-back ordering did not confirm the initial apparent gain. Immediate-data
capability canaries do not supply a successful full r185 timing result.

Material mip fitting regressed against the strongest baseline. Scene fitting
worked as a tool but showed no solver advantage. External ORT-buffer adoption
worked as an API prototype but did not outperform the strongest existing direct
route. The delayed-depth adapter failed reversals and independently moving
surfaces. The tested small vision model was too inconsistent for fair gameplay.
These are retained negative or limited findings, not unfinished successes.

## Closing verification and preservation

During closeout, CPU-only checks reverified the portable triangle and OIT GPU
evidence, reanalyzed the full filtering screen, and reran the pinned Gaussian
CPU reproduction. They passed with the outcomes above. **No new GPU benchmark,
model training or inference campaign was run to close this research.** Recorded
GPU results retain their original dates, source hashes and acceptance rules.

The public repository retains source, patches, selected immutable evidence,
screenshots and reproduction instructions. Local raw results, including failed
attempts, are preserved in verified compressed archives during storage cleanup.
Disposable dependency environments, duplicate upstream checkouts and generated
build trees may be recreated from documented versions. Historical paths under
ignored `results/` require restoring the local archive; they are not promised
to exist in a fresh clone. Unfinished exploratory source is archived locally.

The justified conclusion is useful correctness work and several inspectable
browser tools. Priority, broad hardware performance, production readiness and
scientific novelty remain outside the demonstrated evidence.
