# Research directions

**Closed September 11, 2026.** See the [final report](FINAL_RESEARCH_REPORT.md).
The priorities and continuation criteria below are historical records, not
ongoing research commitments.

Decision checkpoint: September 9, 2026. The target is a substantial, reusable
Three.js/WebGPU improvement that can be explained and demonstrated simply.
This document records research priorities and the evidence behind each decision.
See the [experiment index](../experiments/README.md)
for existing commands and the documents that govern each result.

## Current experiment: authorable optical puzzles

[The Light Vault](../experiments/light-vault/README.md) extends drawing with light
to a fixed phase aperture that holds different images at different depths. The
user moves a screen, captures the actual computed images to open a vault, then
encodes and exchanges their own pair. The forward GPU component receives a source
spectrum and distance; target images never enter its wave calculation.

The supporting result is a working combination: native Three.js TSL wave
propagation, independent CPU numerical checks, image-based capture validation,
and portable phase files with physically checked reference metadata. Browser
CGH and multi-plane fitting have direct prior art, including WebGPU work at
Digital Holography 2026. This is an integration experiment, with no demonstrated
new optical algorithm or industry-level advantage. See the
[source audit and model limits](../experiments/light-vault/PHYSICS.md).

The reusable component can support optical game props and interactive design
tools. More image planes, arbitrary complex scenes, manufacturing and broad
device performance remain unvalidated. The current prototype deliberately
rejects custom marks it cannot make distinct enough to play.

## Retained experiment: drawing with refractive glass

[Sunprint](../experiments/caustic-sketch/README.md) puts inverse caustic design
behind a drawing canvas: target mark → optimized surface → independent forward
light simulation. Changing receiver distance and flattening the glass make the
geometry's effect directly testable. The browser uses a bounded CPU worker for
fitting and Three r186 TSL for photon propagation and accumulation.

The mathematical family has substantial prior art, including Mitsuba's existing
heightmap-caustic tutorial and fabricated freeform optical elements. The
[source audit](../experiments/caustic-sketch/PRIOR_ART.md) includes recent 2026
double-freeform work, which controls more than this single-surface experiment.
The justified result here is an interactive implementation with independent
verification, not a claim of a new optical principle or best solver.

A parallel feasibility screen tested a vision-only browser observer with
SmolVLM-256M, a fixed question, frozen rendered camera pixels, and a higher
precision vision-encoder control. Both configurations produced inconsistent
descriptions of clear objects. That particular model/export/runtime combination
was rejected as the foundation for fair gameplay; this does not establish that
vision-driven gameplay or the model family is generally infeasible. No model
download or inference dependency is needed by Sunprint.

## Retained direction: physical material editing and construction

[MatterForge](../experiments/matter-forge/README.md) is the main new prototype:
closed mesh → persistent particles → liquid through a grille → cast in a mold →
solidified structure → real simulated cargo. The mold is removed before the load
test. A control releases the identical cast as liquid. This targets destructible
props, material reuse and construction puzzles in agent-authored browser worlds.

The revised casting fixture passes independent CPU and WebGPU checks. Both
supports receive material; the solid cast holds cargo while the liquid control
lets it fall. Earlier settling, local-cut and mold-design failures remain
documented. A local GLB adapter supplies geometry and material colors without a
service or inference request. The verified construction geometry is still a
deliberately small authored fixture.

APIC/MLS-MPM and browser model-to-particle conversion have direct prior art.
The proposed contribution is a reusable material-editing workflow with observable
gameplay consequences, not a new solver or an established industry breakthrough.
The next substantive evidence must come from independent assets and construction
tasks, better contact fidelity, and measured interactive rendering. More visual
detail alone does not establish novelty.

## Source audit: extensions to AI scene creation

The captured-object tool is retained, but its bend-and-release interaction does
not establish the broader scene-creation improvement sought here. The next
direction must add a useful capability to an existing creation workflow and
demonstrate it on substantive content. A more elaborate scene alone would not
establish that contribution.

The September 9 source audit found close precedents for several candidate
directions. [Dream Loop](https://github.com/achimala/dream-loop) already combines
concept images, Blender modeling and visual criticism. The author of a
[Blender-to-TSL demonstration](https://www.youtube.com/watch?v=vA4O6n9UUl8)
already demonstrates shader and geometry-node conversion through Blender MCP.
[Disney's Neural Render Proxies](https://studios.disneyresearch.com/2026/07/01/neural-render-proxies-for-interactive-and-differentiable-lighting/)
fit lighting to image-space edits and generative targets, with a fixed scene
and camera. [MUSE](https://arxiv.org/abs/2606.14168) preserves requirements during
scene edits, while [Fly, Fail, Fix](https://research.nvidia.com/publication/2025-08_fly-fail-fix-iterative-game-repair-reinforcement-learning-and-large-multimodal)
uses play traces to guide game repairs. None of those broad promises should be
presented as this repository's discovery.

Two narrower component questions remain open. First, can actual Three motion
produce continuous geometric constraints useful to a scene-generating agent,
beyond ordinary functional masks and replay checks? Second, can the documented
cross-runtime impulse-response limitation in
[three-steam-audio](https://github.com/kwaa/three-steam-audio#reflections-and-reverb)
be removed so browser scenes can retain discrete simulated echoes instead of
only parametric reverberation? These are investigation targets, not established
capability or novelty claims. They remain separate component investigations;
MatterForge is the current demonstrable construction direction.

The separate [r186 Gaussian picking finding](../experiments/splat-query-correctness/README.md)
is concrete: the actual upstream raycast misses two of six fixed cases, while
an independent intersection oracle and a conservative-bound variant agree.
It is a CPU query defect, not a new rendering technique.

## Retained tool: local mechanics for captured point-based objects

The [local-elasticity experiment](../experiments/local-elasticity/README.md)
investigates browser preparation and simulation of an imported Gaussian-splat
object, with the original appearance following the mechanical field in Three
r186. Its initial data comes from an independent phone capture and a separate
mesh-derived splat asset, not bespoke deformation fixtures.

The useful question is whether captured/AI-produced point representations can
become interactive browser assets without mesh conversion or server preparation.
This is a creative integration opportunity, not a new elasticity algorithm.
FreeForm already establishes the underlying method; Kaolin already offers
server-driven web interaction, and SoftGLB already provides local mesh upload
and soft-body simulation. Those systems are the relevant prior art.

This prototype uses a much smaller mechanical basis than FreeForm's main
examples. Numerical parity is necessary but does not establish physical fidelity,
high-quality contact, arbitrary capture support, or a performance advantage.
The decisive continuation criteria are useful behavior on independently supplied
assets, controlled refinement comparisons, and demonstrable improvements over
existing authoring workflows. The complete methods and current limits are in the
experiment documentation. There is no demonstrated industry breakthrough yet.

## Current material direction: automatic pixel and exposure integration

[Signal Loom](../experiments/footprint-filtering/README.md) is a reusable transform
of finite native TSL trigonometric color expressions. Expanding products before
filtering preserves the interference between phases; individual-factor filtering
does not. Native screen derivatives and generated time derivatives supply the
pixel/exposure factors. An independent exact filter and numerical quadrature
provide the reference, with point sampling and a matched 64-sample control.

The final twelve-case GPU screen passes correctness and all nine nonmagnified
utility gates. Error reductions exceed 99.99% relative to point sampling in those
nine cases. The manual analytical shader has comparable accuracy, and supersampling
is more accurate at magnification where errors are already tiny. There is no
timing result or universal improvement claim. Full reports and CPU verifiers are
portable; a close/reopen lifecycle race found in review was fixed separately.

This is an actionable native material-tool improvement. Automatic procedural
filtering and its Fourier mathematics have direct prior art; neither is claimed
new. The next substantive extension would need independently sourced practical
materials or a separately bounded nonlinear-phase/error study, not more repetitions
of these affine fixtures. Geometry motion blur and general AI-shader support are
not established. The older experiments below retain their own completed decisions.

## Current capability: constrained edits of native shader motion

[The constrained editor](../experiments/constraint-editing/README.md) extends native
graph differentiation with exact input rebinding, a shared constrained solver,
pointer target edits and ordinary parameter export. The four fixed cases cover
a wave ribbon and a nonlinear constant-curvature tentacle at two target phases.
All eight AD/FD GPU lanes pass withheld-time quality and export checks; 36 actual
rendered silhouettes match independent geometry exactly. This is a useful bounded
integration, not a new motion-editing algorithm. The fixture reference oracles
and constraints remain explicit; arbitrary generated programs are not evaluated.

The first penalty solver failed motion retention on one program. Its report is
preserved. A separately specified augmented-Lagrangian follow-up added explicit
motion/shape constraints and two new targets before testing. It passed all four
CPU cases and the GPU gate. The next meaningful generality test needs externally
authored supported graphs and editing tasks, with the same adapter and declared
constraints. More authored examples alone would not establish that generality.

[Material mip fitting](../experiments/material-mips/README.md) was also investigated.
Stock Three shading can be expanded without copying its forward BRDF equations,
and rendered parity passes. However, ideal bilinear gradients do not satisfy all
strict hardware-forward derivative checks, and both fitted mip fixtures regress
on unseen lighting. Close that filtering claim at its negative gates. Retain the
extended pure-graph math and evidence without promoting the failed application.

## Foundation: trainable native shader graphs

The [trainable TSL experiment](../experiments/trainable-tsl/README.md) generates
reverse parameter gradients from a restricted native Three shader graph.
A procedural Gaussian field and a sine neural network each learn from 128
observations, use the learned parameters directly in a material, and export
editable weights. The demo also accepts painted height observations.

All four fixed field/seed cases pass the numerical checks and reduce held-out
RMSE by more than 80%. The same-GPU finite-difference control achieves similar
quality and completed times, so no speed or optimizer superiority is established.
The useful integration is automatic backward generation for an existing native
graph. Slang, jax-js, glsl-autodiff, A-delta and Three neural training already
cover substantial adjacent capabilities; the experiment's prior-art document
defines the scope. These are two hand-authored synthetic targets, not a test of
autonomous AI scene creation or arbitrary material reconstruction.

The [portable evidence](../experiments/trainable-tsl/RESULTS.md) includes source
identities, all gradient snapshots, same-start checks and the excluded canary.
The next practical extension would need a real author-supplied material or
deformation graph and an explicit application-level benefit.

## Correctness contribution: triangle distance

The native shader/bounds investigation exposed a reproducible edge-selection
bug in the current three-mesh-bvh WebGPU triangle helper. A nonzero-area thin
triangle can return distance about 500 times too large, and its answer changes
with vertex ordering. The [finding](../experiments/gpu-deform-collider/TRIANGLE-QUERY-FINDING.md)
includes an isolated GPU comparison and a local correction; 200 fixed cases pass.
The correction uses standard closest-point region tests. The interactive
demonstration gives a direct explanation of the failure.

The broader native TSL compiler now emits interval bounds and time derivatives
from a restricted pure graph. Keep that reusable code as an experimental
direction. The six [bounds screen cells](../experiments/gpu-deform-collider/RESULTS.md)
do not establish general completed-work improvement: saved refit passes can be
outweighed by looser traversal bounds. Automatic shader bounding and shader AD
have substantial prior art. Continue this direction only around a demonstrated
application gap, not a first-ever algorithm or pass-count-only speedup claim.

## Completed contribution: ZipDepth export fidelity

Keep the [ZipDepth reproduction](../experiments/zipdepth-export/README.md) as a
concrete correctness contribution. At the pinned upstream revision, export
replaces learned spatial attention with averaging. The minimal patch restores
agreement with the preserved checkpoint computation on three procedural inputs
at two shapes, plus a WebGPU browser check of the square gradient fixture.

The contribution is the identified computation change, strict reproduction,
minimal fix and visible comparison on identical scales. It does not establish
better real-world depth accuracy, improved model speed, a new depth model or a
new GPU-only inference capability. The separate GPU checkpoint, mobile execution,
quantization and representative image quality remain outside the tested scope.
The upstream approximation may have been intentional; no explanation was found
in the checked README/main paper, and supplementary material was not checked.

The experiment README identifies the final CPU/browser reports and retained
exploratory failures. The [export finding](../experiments/zipdepth-export/EXPORT_FINDING.md)
records the computation change and reproduction. That evidence
can be reviewed independently of whichever creative direction is selected next.

## API value retained: external GPU buffers

The [borrowed-buffer prototype](../experiments/ort-surface/README.md) demonstrates
explicit external ownership, validation and cleanup around a Three storage
attribute. It has API value, but its original performance premise did not survive
the [stronger baseline](../experiments/ort-handoff/README.md).

Stock Three 0.185.1 can expose a Three-owned GPUBuffer through private backend
access and let ORT write directly into it. The four-lane screen found no
demonstrated advantage for borrowing over this existing direct path or the
GPU-copy control. Its intervals crossing zero are not proof of equivalence.
The measurements are single-session, CPU-observed serialized latencies, not GPU
timestamps or FPS. Comparing only against CPU download/upload would leave out
the relevant existing alternatives.

Retain the prototype as a possible public API/ownership proposal. Another
performance campaign needs a concrete workload limitation or measured advantage
against those stronger controls. The tiny wave remains a buffer-path fixture.

## Historical studies: inspect, do not restart from old recommendations

The visibility, depth-ordering, indirect-addressing and immediate-data studies
are preserved with their original protocols and outcomes. The
[candidate results](CANDIDATE_RESULTS.md) distinguish scoped fixed-slice wins,
negative ordering results, unconfirmed addressing signals and the completed
standalone reversal. The [immediate-data result](WEBGPU_IMMEDIATE_DATA_RESULTS.md)
retains a failed full Phase 0 attempt despite useful earlier capability canaries.

These are evidence about specific workloads and environments. None supports a
universal culling, sorting or addressing speedup. A repaired harness does not
retroactively pass an old result. Consumed attempts, ledgers and failure artifacts
remain unchanged; a navigation update does not reopen their execution budgets.

## Delayed-depth contact: completed screen, no-go for the general adapter

The [CPU screen](../experiments/depth-contact/README.md) now tests captured-camera
hold, expiry and smoothing against one global centroid/height velocity predictor.
It uses analytic surfaces, 64x64 depth delivered at 5/10 Hz with 80 ms delay, and
identical point probes at 60 Hz. All methods preserve capture coordinates; stale
input is not passed into a current-frame-only collider as a supposed library bug.

All 16 scene/rate cells completed and the self-tests passed, but the global
predictor fails the reversal, opposing-motion and deformation gates at both rates.
The full decision is `no-go-for-general-adapter`. In the 5 Hz post-reversal window,
false contacts rise from 394 to 628 and misses increase by 9.85 percentage points,
despite a favorable whole-scenario average. That is why separate controls matter.

The source and immutable report retain narrow translation successes, all method
counts and CPU-only cost measurements. There is no GPU implementation, real
neural-depth validation or full collision dynamics result. Do not pursue a general
port of this predictor or hide its failures behind the simple-motion examples.

## Editable scene fitting: working prototype, negative comparison

The selected creative prototype is an [editable scene fitter](../experiments/scene-fit/README.md):
ordinary Three scene factories expose bounded geometry parameters, rendered images
supply the objective, and the result stays editable as parameter JSON. The intended
application is useful correction of editable, including AI-generated, scenes.
The actual evaluation uses three hand-authored known-family fixtures, eight
dimensions each, fixed colors/lights/cameras and a second held-out view.

V1 compares standard random, coordinate and Powell-style searches. Eight of nine
original cells passed; rover/47 failed exact target restoration before fitting.
The failure and later changed-source diagnostics remain separate. V2 fixes draw
order and adds GPU central finite differences, GPU normal equations and a bounded
Levenberg-Marquardt solve. Every method receives 96 training renders, including
the 16 perturbation renders per equation batch. These are established algorithms
and reductions, not a new optimizer or automatic differentiation through Three.

All nine V2 cells passed their recorded technical checks. The least-squares
candidate won no training or held-out comparisons against the lowest-error
alternative. The practical superiority gate is therefore not met. Existing scalar
solvers still produce editable fits; that working integration is the present
deliverable, with its negative comparison visible. No measured performance or
general autonomous scene-repair claim follows.

Do not continue tuning merely to erase this result. A further research stage needs
a specific practical capability gap and a strong baseline, with a new bounded
protocol. An attractive combination of tools is not by itself a substantial
finding. Per-cell metrics, source identities, limits and entry commands are in the
experiment README and analyzer; the older visibility experiments remain sealed.

## Discovery constraints and existing work

Do not restart generic Gaussian-splat integration as an unexplored direction.
[Three r186](https://github.com/mrdoob/three.js/releases/tag/r186), released September
8, already includes native Gaussian splats; multiple-object sorting and sorting
acceleration have active [PR #34290](https://github.com/mrdoob/three.js/pull/34290)
and [PR #34235](https://github.com/mrdoob/three.js/pull/34235).
[Spark](https://github.com/sparkjsdev/spark/blob/728fc4053904e03c6f63f524a2ea4e4793935cbe/README.md)
already provides programmable GPU edits, displacement and skeletal animation.
[Gaussian Splat Lite](https://github.com/WilliamLiu-1997/Gaussian-Splat-Lite/blob/aa7964913c1ee7b8d071498b871365dfd9ef7e36/README.md)
already provides WebGPU sorting, multiple objects, depth occlusion, stochastic
rendering and streamed LOD. These September 9 checks identify overlap, not a claim
that every possible splat improvement is exhausted.

GPU-resident neural inference/rendering and generic depth interactions also have
prior art. The external-buffer and ZipDepth READMEs retain relevant attribution;
new proposals need their own current source audit rather than inheriting a
priority claim from this repository.

## Source of truth and continuation rules

- Use experiment-local protocols, source/checkpoint identities and immutable raw
  artifacts to establish the execution. Use the associated verified analyzer and
  decision rules to establish the outcome.
- Follow the final evidence identifiers in the experiment README or result doc.
  Local ignored artifacts may be unavailable in a fresh clone; do not treat their
  absence as permission to manufacture a replacement historical run.
- Keep new feasibility work in a separate experiment with explicit status.
  Implementation, correctness, measured utility and broader novelty are distinct
  milestones; record which have actually been reached.
- Honor the bounded execution limits in the [experiment index](../experiments/README.md)
  and each runner. No new background GPU loops, driver changes or broad model
  downloads follow from this roadmap.

This document is navigation and research intent. It does not promote development
evidence to candidate evidence or change a sealed decision.
