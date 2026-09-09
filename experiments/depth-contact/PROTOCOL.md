# Delayed depth contact: exploratory feasibility gate

The question is whether a reusable adapter can improve contact decisions when
depth arrives more slowly than a simulation updates. This is not a claim that
depth collisions are new, or that a fresh-depth API is faulty when misused with
stale data. Plume already implements GPU/TSL depth collisions for current-frame
rendered depth; a fair delayed-input baseline must retain capture-camera matrices.

First use deterministic, finite analytic height fields. Capture sampled depth at
5/10 Hz, deliver it after a fixed delay, and evaluate scripted probes at 60 Hz.
Compare captured-camera hold, expiry and smoothing with a two-frame motion
predictor. Only depth samples, masks, timestamps and capture-camera metadata may
reach the predictor. Exact scene motion is reserved for the scoring oracle.

Include stationary geometry, moving camera, translation, reversal, nonrigid
motion, silhouette edges and independently moving surfaces. Report false and
missed contacts, penetration and execution cost per scene and sampling rate.
Do not hide reversals or multi-object failures in an aggregate average. These
are local top-surface queries, not a complete rigid-body simulation or a neural
depth accuracy evaluation.

The proposed continuation gate is at least 50% fewer false contacts than the
strongest simple baseline, with no more than five percentage points additional
missed contacts and bounded additional cost. Reject an apparent improvement
obtained just by disabling collisions. If frozen camera matrices or a simple
expiry rule explain the improvement, retain it as ordinary integration work.

This initial numerical screen is CPU-only. Only a convincing result warrants
porting the candidate to GPU and connecting neural depth. The screen is
exploratory; it is not a preregistered benchmark or an external replication.

Prior art: [Plume depth collision](https://github.com/travisdmathis/plume/blob/main/packages/plume/src/modules/update/depth-collision.ts),
[Interverse height-field collision](https://github.com/aiira-co/three-particles/blob/main/src/providers/DepthCollisionProvider.ts).
