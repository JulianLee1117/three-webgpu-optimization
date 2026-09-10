# Three r186 Gaussian raycast: nonconservative sphere rejection

**Confirmed CPU correctness defect:** the actual `GaussianSplat.raycast()` misses
two intersecting rotated anisotropic ellipsoids in six fixed cases. An isolated
conservative-bound variant and a narrow-phase control both match an independent
scalar intersection oracle. No renderer, browser, GPU, or external asset is used.

Run from the repository root with the installed dependencies and Node 24:

```sh
node scripts/check-splat-query-correctness.mjs
```

The checker creates a fresh evidence directory on each run. It never edits
`node_modules`, contacts upstream, or changes an application renderer. Source
hash verification rejects an unexpected Three revision.

## Defect

For a positive-semidefinite covariance `C`, a cutoff-`k` ellipsoid has largest
radius `k * sqrt(lambdaMax(C))`. The maximum diagonal entry is a coordinate-axis
variance; it can be smaller than the largest eigenvalue after rotation.

The fixture has center `(0,0,0)`, covariance

```text
C = [ .505  .495  0   ]
    [ .495  .505  0   ]
    [ 0     0     .01 ]
```

Its eigenvalues before Float32 storage rounding are `1, .01, .01`, and the long
axis is `(1,1,0)/sqrt(2)`. A +Z ray starts at
`(1.8/sqrt(2), 1.8/sqrt(2), -1)`. Its closest distance to the center is `1.8`.
With Three's cutoff `k=2`, the true long radius is `2`, but the rejection radius
is only `2*sqrt(.505) = 1.42127`. The sphere test rejects the ray even though the
ellipsoid quadratic intersects it.

Three adds `max(diag(C))*1e-4` to all diagonal entries before its narrow-phase
inverse. The independent oracle includes that exact regularization and the
actual Float32 covariance values. It predicts the near hit at distance
`0.9125927679795267`; the original raycast returns no hits.

Two locations must be distinguished:

- The object bounding sphere uses the maximum diagonal entry.
- The per-splat sphere rejection also uses the maximum diagonal entry.

Overriding only the outer sphere with a large sphere still misses. Changing only
the per-splat sphere also misses. The checker separately exercises both partial
corrections, the combined correction, and unchanged upstream narrow phase with
the two sphere rejections disabled. It also retains aligned, center-ray,
outside-ray, faint-opacity, and three-axis-rotation controls.

## Source and correction scope

The installed Three `0.186.0` addon matches the source at
[commit 148ef33ecb6d2502ff796d4554abd1549c95d519](https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/objects/GaussianSplat.js).
Tag `r186` was created September 8, 2026, at `19:12:48 UTC`; its annotated tag
object is `819fadd6b663b74d828c6af72a543024f74d3877`.

Relevant pinned lines:
[object sphere, line 339](https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/objects/GaussianSplat.js#L339),
[outer rejection, line 366](https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/objects/GaussianSplat.js#L366),
[per-splat bound, line 515](https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/objects/GaussianSplat.js#L515),
and [covariance regularization, line 530](https://github.com/mrdoob/three.js/blob/148ef33ecb6d2502ff796d4554abd1549c95d519/examples/jsm/objects/GaussianSplat.js#L530).

Source SHA256:
`ea3f148d39413ef31781cdbb1c97a8972b2dce01ef5eebe6ef800c2b1ba7ab42`.

The diagnostic correction uses `k*sqrt(trace(C + delta*I))`, which bounds the
regularized ellipsoid because its largest eigenvalue cannot exceed its trace.
It also enlarges the common radius used for the bounding box, keeping that
diagnostic conservative around the regularization boundary. This is intentionally
simple; it is not claimed to be the tightest or fastest upstream fix. The
narrow-phase quadratic, opacity rule, and all other behavior remain upstream code.

This evidence establishes CPU query false negatives on the declared identity-
transform cases. It does not establish a visual-rendering defect, an occurrence
rate on real scans, a GPU defect, or universally validated collision behavior.
The existing captured-object experiment uses its own bounds and center-based
picking, so this finding is kept separate from its evidence.

## Retained evidence

[The first report](evidence/2026-09-10T00-50-00.094Z/report.json) records all inputs,
oracle quadratics, every variant's hits and bounds, exact source snapshots,
variant hashes, and resolved r186 module hashes. Its result is
`defect-confirmed`: original false negatives `2/6`; corrected and narrow-phase
control mismatches `0/6`. That status means the bug was reproduced, not that the
original library passed.

Report SHA256:
`b04348341c77c04a258832bae06b174e5b49cfecec35fb402209bda8015320c0`.
The adjacent `report.sha256` contains the checksum. All Three imports in the
checker resolve to the same r186 source runtime. The in-memory overlays and
their exact substitutions are readable in the checker; the source file remains
unchanged. Upstream source included for provenance remains under Three's MIT
license, reproduced in [THREE-LICENSE.txt](THREE-LICENSE.txt).
