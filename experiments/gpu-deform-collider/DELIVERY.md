# Verified local delivery, 9 September 2026

Portable package:
`deliverables/triangle-query-demo-2026-09-09T19-36-46.437Z.zip`

- Size: 1,050,604 bytes.
- SHA-256: `88242441646747a550d56e42ca767173c214de693d4102b3c4c21fa3b85aef85`.
- All 29 packaged files verified individually, including inside the ZIP.
- Static server: 14 checks passed. Extract and run `node serve.mjs`; no npm
  installation is needed for this package. Server binds loopback port 5191.
- Final bundled demo: all 18 height/order comparisons passed, corrected
  distances matched analytic answers, and every comparison released its GPU
  device. Mobile layout had no horizontal overflow. Early Stop cancellation
  also released resources. [Package UI report](../../deliverables/triangle-query/2026-09-09T19-37-35.042Z/report.json).

The package includes the original/corrected 200-case GPU reports, patch, MIT
licenses, screenshots, and a short slide clip assembled from three actual GPU
comparisons. Each screenshot is held for two seconds; it is not an FPS recording.
The demo defaults to original and corrected functions in the same dispatch.

Other completed checks:

- CPU reference 83 known-answer checks; refit schedule validation passed.
- Native TSL interval compiler: 2000 independently evaluated samples passed.
- Native TSL derivative compiler: 1500 analytic samples passed.
- Generated GPU derivatives passed numerical checks at 0,.7,2.4 on wave, twist
  and ripple. Reports: `2026-09-09T19-21-57.149Z-wave-16-128`,
  `2026-09-09T19-21-59.718Z-twist-16-128`,
  `2026-09-09T19-22-02.197Z-ripple-16-128` under the experiment result directory.
- Triangle CPU comparison: 249 original disagreements in 10000 deterministic
  cases; independent corrected edge reference had zero disagreements.
- Final triangle GPU comparison: original 10/200 mismatches; corrected 0/200.
- Bounds audit: 198 positive snapshots passed; all 12 invalid-bound controls
  detected. Performance remains mixed; see [RESULTS.md](RESULTS.md).
- Production build passed. Existing repository tests: 962 passed, 5 skipped,
  zero failures (967 total). No historical GPU study was rerun.

The package builder is `scripts/package-triangle-demo.mjs`. It intentionally
requires the locally retained, pinned evidence files and verifies their hashes.
Earlier partial packages and failed reports remain preserved; only the package
identified above has the final bundled-browser verification recorded here.

The [technical finding](TRIANGLE-QUERY-FINDING.md) records the cause, correction,
reproduction and remaining limits.
