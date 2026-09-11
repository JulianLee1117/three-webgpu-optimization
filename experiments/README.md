# Research experiment index

**Research is concluded.** The [final report](../docs/FINAL_RESEARCH_REPORT.md)
is the closing assessment. The entries below preserve the individual studies;
their historical commands are reproduction instructions, not active work.

Closing presentation reviewed September 11, 2026. Confirmed findings lead this
index. Later optical and material demos remain unpolished feasibility experiments;
their technical checks do not establish an impressive or intuitive experience
for a general audience. The [research directions](../docs/RESEARCH_DIRECTIONS.md)
preserve historical intentions and continuation criteria.

## OIT alpha — confirmed Three r186 coverage defect

[The isolated WebGPU reproduction](oit-alpha-correctness/README.md) compares
ordinary blending, stock OIT, and an alpha-only correction in five controlled
fixtures. Stock omits transparent foreground coverage; the correction matches
all 15,360 retained RGBA pixels. A portable CPU verifier checks pixels, source
identity, readback layout, controls, and four deliberate corruptions.

```sh
node scripts/analyze-oit-alpha.mjs --self-test
```

## Gaussian picking — confirmed Three r186 CPU query defect

[The isolated reproduction](splat-query-correctness/README.md) invokes the
unmodified Gaussian-splat raycast. Its sphere rejection underbounds rotated
elongated splats and misses two of six fixed cases. A conservative-bound
correction agrees with an independent ellipsoid oracle in all six. Portable
source snapshots, input arrays and evidence are included; no GPU is involved.

```sh
node scripts/check-splat-query-correctness.mjs
```

## Triangle queries and native shader graphs — new finding and exploratory tools

[This experiment](gpu-deform-collider/README.md) found an obtuse-triangle edge
selection error in the current three-mesh-bvh WebGPU helper. A tested thin
triangle reports distance about 1 instead of .002; the corrected helper passes
the 200-case isolated GPU suite. Its interactive demo changes vertex order while
preserving geometry. `npm run demo:triangle-query` starts it idle.

The same work implements native TSL interval and derivative graph transforms.
Six geometry/query screen cells pass tested enclosures and queries after the
triangle correction, but completed-work performance is mixed. This remains an
integration prototype, not a new bounding algorithm or a general speedup.

- [Correctness finding and upstream patch](gpu-deform-collider/TRIANGLE-QUERY-FINDING.md)
- [Portable original/corrected GPU evidence and CPU verifier](gpu-deform-collider/evidence/README.md)
- [Bounds results](gpu-deform-collider/RESULTS.md)
- [Scope and prior art](gpu-deform-collider/PRIOR_ART.md)

## ZipDepth export fidelity — completed correctness finding

The released Base NPU checkpoint's learned spatial pooling is replaced by an
unweighted average during the official export. Removing that substitution restores
reference agreement on six procedural CPU fixtures and one bounded WebGPU browser
reproduction. This measures export fidelity, not real-image depth accuracy or speed.

- [Finding, tested setup, evidence identifiers and limitations](zipdepth-export/README.md)
- [Minimal upstream source patch](zipdepth-export/preserve-global-context.patch)
- [Technical export finding](zipdepth-export/EXPORT_FINDING.md)

From the repository root, after following the experiment's Python/environment
preparation instructions:

```powershell
.local-research/zipdepth-export-env/Scripts/python.exe experiments/zipdepth-export/prepare.py
.local-research/zipdepth-export-env/Scripts/python.exe experiments/zipdepth-export/probe.py
.local-research/zipdepth-export-env/Scripts/python.exe experiments/zipdepth-export/probe.py --width 512
npm run check:zipdepth
npm run demo:zipdepth
```

Preparation verifies six pinned upstream hashes and downloads a 27,295,474-byte
checkpoint. CPU export/inference uses two compute threads; the isolated Windows
CPU Torch wheel is about 591 MiB. The browser check runs two inferences, validates
the result and disposes resources, with a 45-second watchdog. Serve mode starts
idle and runs only the two inferences requested by its button; it has no animation
loop. The experiment README is the source of truth for dependencies, fixture
selection, report hashes and platform limits.

## Procedural footprint filtering — verified native material integration

[Signal Loom](footprint-filtering/README.md) expands an existing native TSL color
graph into joint Fourier terms and integrates each over a pixel and a material
exposure interval. This retains correlations and low-frequency beats between
fine patterns. Three programs and four fixed footprint/exposure conditions pass
all twelve GPU correctness cases. All nine nonmagnified cases reduce RGB RMSE
by over 99.99% versus point sampling and beat the matched 64-sample control.
The manually filtered baseline also passes; no speed advantage is measured.

The exact mathematical class is constant-amplitude trigonometric polynomials
with affine phases. The folded, lit interactive illustration uses local
footprint approximations. Filtering theory is established; this result concerns
the automatic native TSL integration. Read its [scope and results](footprint-filtering/RESULTS.md).

```sh
npm run demo:footprint-filtering
npm run test:footprint-filtering
node scripts/analyze-footprint-filtering.mjs --self-test
```

## The Light Vault — exploratory wave-optics interaction

[The Light Vault](light-vault/README.md) encodes two images into one phase pattern.
Move a detector through its computed wave field, capture both clear images to
open the vault, then draw and save a new seal. Three r186 TSL calculates an inverse
FFT at every requested depth. Capture checks read actual GPU intensity; a saved
seal imports only after its phase reproduces both reference images.

The intent was an editable optical-puzzle mechanism. The retained value is a
wave propagator, local fitting and phase-file exchange. Graphics and interaction
remain unpolished; broader usefulness in a game or educational tool is unproven.
The [physical contract and prior art](light-vault/PHYSICS.md) distinguish the
verified implementation from claims that have not been established.

```sh
npm run demo:light-vault
```

## Sunprint — interactive inverse optics

[Sunprint](caustic-sketch/README.md) turns a drawing into a shaped glass surface.
An independent Three.js TSL pass refracts light through that surface to form the
screen image. Move the screen, flatten the glass, draw a different mark, or save
the approximate mesh. The CPU optimizer and GPU light renderer use established
methods; the experiment demonstrates an accessible, reusable browser workflow.
Its [prior-art notes](caustic-sketch/PRIOR_ART.md) identify close precedents.

```sh
npm run demo:sunprint
```

## MatterForge — experimental material-state editing

The intent is persistent material editing for construction games, tested through
a controlled solid-versus-liquid comparison. Fluid appearance and interaction
remain unpolished, with coarse contacts and explicit supports.
[MatterForge](matter-forge/README.md) casts liquid through a grille, solidifies
the resulting particles, removes the mold, and drops real simulated cargo onto
the cast. Its liquid control starts from the identical state. The revised CPU
and WebGPU fixture passes; the earlier failures and coarse contact limits remain
explicit. The live Three r186 scene accepts closed local GLB geometry.

```sh
node scripts/run-matter-forge.mjs --serve
```

## Carry the City — playable motion-transfer prototype

[Carry the City](carry-the-city/README.md) applies authored swimming and flying
motion to an inhabited town. The shared field drives street deformation, rigid
attachments, a docking endpoint and released-prop motion. The rescue route,
restart, collision release and cleanup pass a bounded browser check; 17 CPU tests
pass. This is a creative combination of established techniques with approximate
gameplay collision. General navigation, physics and a novel algorithm are not
established.

```sh
npm run demo:carry-the-city
```

## Local elasticity — retained captured-object tool

[Local elasticity](local-elasticity/README.md) combines browser RKPM preparation,
an implicit elastic solver, and the native Three r186 Gaussian splat renderer.
The initial assets are an external phone capture and mesh-derived splats. There
is no supplied rig, remote simulation, trained model, or motion recording.
Material and integration volume are assumptions; the sparse model does not
establish physical accuracy or general support. FreeForm owns the underlying
method, and existing browser GLB soft-body tools are explicit prior art.

```sh
npm run demo:local-elasticity
npm run test:local-elasticity
npm run check:local-elasticity
```

Read [the bounded evidence](local-elasticity/RESULTS.md), including quality and
out-of-sample limitations, before making claims about this capability.

## Constrained shader editing — verified bounded capability experiment

[Direct the motion](constraint-editing/README.md) rebinds one authored native TSL
graph for rendering, batched evaluation and generated parameter derivatives.
The demo edits tip targets on two animated programs, avoids a sphere at sampled
times, retains motion and exports eight parameters. All eight final GPU lanes
pass and all 36 rendered silhouette pairs match the independent numeric geometry
exactly. This is a native integration, not new optimization mathematics or a
general speedup. The failed first penalty solver is preserved alongside the
successful separately declared constrained follow-up.

Run `npm run demo:constraint-editing`. [Results and CPU verifiers](constraint-editing/RESULTS.md).

## Material mip fitting — candidate rejected; shading parity retained

[The stock-shading bridge](material-mips/README.md) expands actual Three 0.185.1
metallic direct-light functions. Rendered forward parity passes, while five of
1,521 smooth hardware-forward finite-difference checks fail fixed tolerances.
Both mip-fitting fixtures also regress against the strongest roughness-only
baseline on withheld lighting. Preserve these limits; no filtering improvement
or exact hardware-gradient claim is justified. Compressed evidence and a CPU
verifier are included.

## Trainable native TSL — verified bounded capability experiment

[Train a forward shader graph](trainable-tsl/README.md) using automatically
generated reverse gradients. The lab includes a 28-parameter procedural field,
a 49-parameter sine neural field, a height-example editor and weight export.
Four field/seed cells passed initial/final GPU gradient checks and the held-out
capability gate. A same-GPU finite-difference control achieves similar accuracy
and completed times. This is a native graph integration, not a new AD algorithm
or demonstrated speedup.

Run `npm run demo:trainable-tsl`. The [results](trainable-tsl/RESULTS.md) include
portable evidence verifiable with `node scripts/analyze-trainable-tsl.mjs`.

## External-buffer handoff — working API prototype, no demonstrated performance advantage

The [synthetic wave demo](ort-surface/README.md) validates a borrowed GPUBuffer
attribute and ownership behavior. It is a small integration fixture, not a learned
physics solver. The later [strongest-baseline screen](ort-handoff/README.md) found
that stock Three 0.185.1 already exposes a direct ORT output route through private
backend access. The proposed public API still has a usability and lifetime
management rationale; the screen did not establish a performance advantage over
that existing direct route or the GPU-copy control.

```sh
npm run prepare:ort
npm run demo:ort
npm run test:external-storage
npm run analyze:ort-handoff
```

The demo starts idle and caps animation at eight seconds, 120 updates, 15 updates
per second and 65,536 vertices. Hiding the page stops animation. Its recorded
update rate is not a throughput benchmark.

Read the [handoff protocol](ort-handoff/PROTOCOL.md) before any new screen. Its
runner entry is `node scripts/run-ort-handoff-screen.mjs`; the README specifies
the stock-only probe and independent 128/256 size sessions. Each timed page is
limited to 128 total warmup/measured updates, at most 15 updates per second and
a 30-second runner deadline, with a cooldown between sessions. The completed
null screen does not need to be rerun to continue another direction.

The recorded endpoint is CPU-observed serialized completion latency, including
synchronization. It is not GPU execution time, frame latency or FPS. Use the
verified analyzer's conventional medians, not the original summary's upper-middle
median. A win over download/upload alone does not establish new performance.

## Historical visibility and indirect-rendering studies — preserved evidence

The older harness lives in the repository's main source, scripts and analysis
directories. These studies are retained for inspection; they are not the active
research task.

- [Candidate results and exact evidence identifiers](../docs/CANDIDATE_RESULTS.md):
  fixed-slice specialization has scoped single-device positive results; coarse
  depth ordering and frozen render ordering did not establish the proposed win;
  the indirect `firstInstance` studies did not meet their confirmation rules.
- [Benchmark protocol](../docs/BENCHMARK_PROTOCOL.md): workload, timestamp boundary,
  provenance and environment requirements.
- [Live candidate ledger](../docs/FIRST_INSTANCE_LIVE_CANDIDATE_LEDGER.md) and
  [standalone deployment protocol](../docs/INDIRECT_FIRST_INSTANCE_STANDALONE_DEPLOYMENT_PROTOCOL.md):
  fixed attempt/replication accounting and decision rules.
- [Immediate-data results](../docs/WEBGPU_IMMEDIATE_DATA_RESULTS.md): narrow
  capability canaries passed, but the sole full r185 Phase 0 attempt failed and
  captured no timing result. Subsequent source repairs do not change that outcome.

For an existing complete local result, the generic analysis entry is:

```sh
npm run analyze -- results/runs/<existing-run-id>
```

Other studies have dedicated verifiers linked from their result/protocol docs.
Raw run directories are generally ignored local artifacts and may be absent in a
fresh clone. Follow the actual retained identifier; do not invent a new candidate
series to replace an unfavorable result. Historical runner commands in the root
README and package scripts are not invitations to reopen consumed one-shot gates.

## Delayed-depth contact — completed CPU screen, general adapter rejected

The [CPU reference and result](depth-contact/README.md) test slowly delivered depth
against captured-camera hold, expiry and smoothing baselines. A global velocity
predictor passes simple translation controls but fails after reversals and with
opposing or deforming surfaces at both 5 and 10 Hz. The full 16-cell screen returns
`no-go-for-general-adapter`; its narrow successes do not justify a general GPU
port. No neural model or GPU contact simulation was run.

```sh
node experiments/depth-contact/reference.mjs --self-test
node experiments/depth-contact/reference.mjs
```

These commands are CPU-only and print results without file writes. The default
screen uses 64x64 depth, eight scenes, two rates, one simulated second of warmup
and four measured seconds. Its README identifies the saved report/source hashes.

## Editable scene fitting — working bounded prototype, negative solver comparison

The [scene-fit experiment](scene-fit/README.md) fits eight structural dimensions
in each of three ordinary Three scenes and exports editable parameter values.
The aim is a useful interface for editable, including AI-generated, scene
factories; the current cases are hand-authored known families.

V1 remains incomplete with eight passed cells and a retained rover/47
target-restoration failure. V2 fixes draw order and compares four standard solvers
with exactly 96 training renders each. All nine V2 cells pass recorded technical
gates, but the GPU finite-difference/least-squares candidate has no lowest-error
wins on either training or held-out views. This is a usable prototype with a
negative comparison, not a demonstrated solver advance or speedup.

```sh
node scripts/run-scene-fit.mjs --serve
node scripts/analyze-scene-fit.mjs --source-ref=1d42d9a --source-ref=494f052
```

Serve mode starts idle at `http://127.0.0.1:5188/`; the selected fit stops after its
96-render budget. The automated runner is limited to four fits per cell, 128x128
training images and a 45-second watchdog. Analysis is CPU-only, keeps versions and
source identities separate and does not fill failed cells with later diagnostics.
The optional source refs are the local historical V1/V2 checkpoints. See the experiment
README for setup, commands, per-scene outcomes and evidence limitations, and the
[directions document](../docs/RESEARCH_DIRECTIONS.md#editable-scene-fitting-working-prototype-negative-comparison)
for the continuation decision.

## Execution and evidence boundaries

All commands above run from the repository root. Use `npm ci` with the checked-in
lockfile for the documented Node 22.12+ setup. Local browser demos require a
compatible WebGPU browser; compatibility beyond recorded environments is untested.
Run one bounded GPU task at a time, honor each runner's stop conditions, and close
the demo tab and localhost server when finished. Stop on GPU validation errors,
device loss or failed parity. These entries do not request browser feature flags,
driver changes, elevated device limits or background benchmark loops.

The experiment's protocol and immutable artifacts determine what was tested; its
verified analysis determines the recorded decision. Navigation documents summarize
that evidence and do not supersede it. Preserve failures, old hashes and partial
reports. A correctness pass, a visually attractive demo and a performance result
are separate claims. Existing GPU-resident inference/rendering and splat/depth
interaction techniques must receive attribution; no priority claim follows from
combining them here.
