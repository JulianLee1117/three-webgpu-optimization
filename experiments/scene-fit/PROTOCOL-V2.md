# GPU finite-difference scene fitting: exploratory follow-up

V1 is preserved at local source commit `1d42d9a`. Eight original cells passed;
rover/47 failed the exact target-restoration check before fitting. A diagnostic
rerun reproduced 42 changed byte channels. Fixing draw order (`sortObjects=false`)
restored exact agreement. The original failed reports stay in the results tree.
This follow-up reruns all methods with fixed draw order; it does not pool versions.

Keep the three factories, eight bounded dimensions, seeds 11/29/47, raw RGB MSE,
128-square training images, and final held-out camera from V1. Compare random,
coordinate, Powell-style direction search, and bounded Levenberg–Marquardt using
central finite differences. These are established algorithms, not new methods.

Every method receives 96 candidate **renders**. For least squares, each normal
equation batch costs 16 renders (two per parameter); each trial costs one. The
initial candidate costs one. Validation, fixed reference captures and final
held-out evaluation are outside this fitting budget and explicitly recorded.
The same target parameters remain inaccessible to every optimizer. The fit may
only alter the declared dimensions; colors, lights and cameras remain fixed.

The candidate stores perturbed images in a GPU texture array, reduces JᵀJ and
Jᵀr on the GPU, transfers 44 floats (176 bytes), and solves the small damped
system on the CPU. Scalar trial losses transfer 16 bytes. There is no whole-image
readback during fitting. This is an implementation property, not a measured
speedup: no CPU-equations timing baseline is claimed. A separate pixel readback
must validate all 64 Hessian and eight gradient entries before accepting results.

Report each scene/seed independently, fitting and held-out errors, actual render
counts, scalar queries, normal equation batches, readback bytes and failures.
The practical gate remains improvement over simple baselines across scenes and
seeds, including the held-out view. Report comparison with Powell as well. Do
not hide failure behind a pooled average or treat these nine synthetic fixtures
as evidence of general inverse rendering or autonomous AI scene repair.

Limit each cell to four fits, a 45-second watchdog and serial GPU submission.
No new dependencies, model weights, large allocations, custom device limits or
experimental browser flags. Stop on parity errors or device loss. A method that
fails this screen remains a prototype, regardless of visual appeal.
