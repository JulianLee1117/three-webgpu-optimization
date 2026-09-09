# Local standalone demo

The verified bundle was generated with `node scripts/package-constraint-editing.mjs --zip`.
It contains one source-consistent Three 0.185.1 runtime, the app, research notes,
Three's MIT notice, an integrity verifier and a loopback server. No npm install
or model download is required to run the extracted bundle.

```sh
node verify.mjs
node serve.mjs
```

Open the printed URL in a WebGPU-capable browser. The page starts idle.

Build identifier: `constraint-editing-demo-2026-09-09T21-24-48.930Z`.
ZIP size: **344,965 bytes**.
ZIP SHA-256: `633e6a598ce6a68cb21952772f452d8e2a07be5d396022bbd6265ef9a9615f60`.
Manifest SHA-256: `ca2d39acf02c6c38e7c046ca91eee98e2c8f3435174f9fbb6fdb8d42802f4aa2`.
Bundles are local generated artifacts under ignored `deliverables/`; the script
reproduces their structure from the repository source.

## Completed bundle checks

The packaged app passed idle-start, pointer-target editing, parameter download,
reset, active-fit stop, reopening and mobile layout checks. A separately simulated
hidden-property event verified the visibility cleanup handler. The deliberately
dragged tentacle target was outside its reachable curve; the UI correctly retained
and reported its unsatisfied target constraint rather than claiming success.

The preset late-target tentacle was then checked inside the actual bundle with
both AD and same-GPU finite differences. Both lanes passed utility, all three
initial/final derivative snapshots passed, JSON reloads were exact, and every
rendered comparison passed. No browser/GPU errors were recorded.

Local reports:

- `results/development/constraint-editing-ui/2026-09-09T21-19-05.457Z-source-ribbon-resize-fix/`
- `results/development/constraint-editing-ui/2026-09-09T21-25-51.676Z-portable-tentacle/`
- `results/development/constraint-editing-ui/2026-09-09T21-26-44.740Z-portable-preset-probe/`

These are packaging/interaction checks separate from the portable fixed numerical
cohort. The latter's complete reports and CPU verifiers are in `evidence/`.
