# Standalone Signal Loom delivery

The source demo is `npm run demo:footprint-filtering` at loopback port 5197.
The portable folder requires only Node to serve the included browser bundle:

```sh
node verify.mjs
node serve.mjs
```

Open the printed loopback URL on port 5198. The UI starts idle. A user-requested
preview stops after twelve seconds; Release GPU or page hiding disposes it.
No dependency install, model download, external assets or service is needed
inside the portable folder. A WebGPU-capable browser is still required.

## Verified bundle

Local artifact: `deliverables/footprint-filtering-demo-2026-09-09T22-42-15.667Z.zip`.
Size: **327,950 bytes**.
ZIP SHA-256: `ee06c3974f87991f9400c0f1ec97e1b9c399c30555b1f18b4bd755692807ef3f`.
Manifest SHA-256: `1027f8d8d3330097ac29a279bb7dd73c95f5364c9d950f6e53e28935dac729d5`.

All fourteen file hashes and ZIP entry hashes were verified. The bundle has
586 captured input modules and exactly one Three source TSLCore runtime, with
no mixed build/source runtime. Three's MIT notice is included; this does not
assign that license to all research code.

The packaged browser UI passed `2026-09-09T22-43-35.082Z-ui`, including all three
patterns, control changes, actual preview timeout, mobile width, release/reopen,
simulated visibility handling and delayed obsolete-initialization rejection.
There were zero recorded browser errors. The same test of the source UI passed
`2026-09-09T22-37-55.334Z-ui`.

The numerical affine GPU cohort is separate and described in [RESULTS.md](RESULTS.md).
The bundle UI is a curved, lit illustration, not a numerical proof of perspective
or lighting integration. No new timing/FPS result follows from packaging.

Generate another immutable local bundle with:

```sh
node scripts/package-footprint-filtering.mjs --zip
```

Generated bundles remain outside Git. Public source, reproducible commands and
full compressed numerical evidence are in the experiment directory.
