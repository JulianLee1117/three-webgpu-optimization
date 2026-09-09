# External buffer handoff: strongest-baseline screen

This new development screen does not alter or reopen any historical one-shot.
Its purpose is to decide whether the borrowed-buffer patch deserves a performance
research campaign. The model is deliberately the existing tiny synthetic wave:
this cheaply exposes handoff overhead before downloading a larger real model.

Compare four lanes on the same patched Three source, one ORT session, same
GPU-generated input, output shape, camera, target, topology and shader arithmetic:

- borrowed: ORT writes an application-owned buffer exposed by the new attribute.
- internal: ORT writes directly into an ordinary Three-owned storage buffer,
  obtained through `renderer.backend.get(attribute).buffer` after initialization.
  This uses an existing internal API; it is the critical zero-copy control.
- gpu-copy: ORT writes the shared application buffer, then one native GPU copy
  transfers it into an ordinary Three-owned storage buffer before rendering.
- cpu-copy: same ORT output, then download through a reused staging buffer and
  upload into an ordinary Three-owned storage buffer before rendering.

Before timing, compare all output scalars with the independent CPU model and
require all four 384x256 RGBA8 images to match exactly at two time points.
GPU validation scopes surround the correctness phase and the timing phase, not
each timed update. No instrumentation, screenshots, console/DOM updates, or
per-sample error-scope changes occur inside the measured interval. ORT validation
uses normal `wgpuOnly`; native WebGPU validation remains enabled.

The endpoint is CPU-observed serialized completion latency from immediately
before input generation/inference to `queue.onSubmittedWorkDone()` after drawing.
It includes JavaScript, submission, copying, scheduling and synchronization.
It is not a GPU timestamp, display latency, or a frame-rate measurement.

Each size uses eight warmups per lane followed by 24 four-lane blocks. All 24
permutations appear once, in a deterministic shuffled block order. Each block
uses the same time input for every lane. Individual samples are paced at at most
15 updates/second outside timing. Run 128x128, then 256x256 vertices in separate
browser sessions with a cooldown. Each page allows no more than 128 total timed
and warmup updates; the runner has a 30-second deadline. Stop on GPU errors,
device loss, or failed parity. No custom browser flags or elevated device limits.

Also run a timing-free ordinary-three@0.185.1 internal-lane probe, with no patched
Three imports, to check that zero-copy is already possible in the installed release.

Report raw samples, per-lane median/p95, block-paired median differences and a
fixed-seed bootstrap interval over complete blocks. Use the 24 blocks as the
sampling units, not pixels/vertices. Results are a single-session diagnostic;
the interval does not measure cross-session or cross-device uncertainty.

Decision: only pursue a new performance claim if borrowed meaningfully beats
the existing internal zero-copy path. A CPU-copy-only win does not establish
novel performance. If internal matches borrowed, retain the patch as an API and
ownership contribution without claiming a performance improvement.
