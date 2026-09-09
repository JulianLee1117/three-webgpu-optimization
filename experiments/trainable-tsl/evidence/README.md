# Complete recorded evidence

This directory contains the complete JSON reports for the four final runs and
one excluded canary, compressed with gzip. These are not summaries: the reports
retain recorded source hashes, served-module hashes, generated shader code,
gradient checks and references, sample and parameter arrays, training and
validation results, browser diagnostics and resource-cleanup state.

From a fresh clone, run the CPU-only evidence audit from the repository root:

```sh
node scripts/analyze-trainable-tsl.mjs
```

The audit uses these committed reports. It does not launch a browser or repeat
GPU training. See [the experiment protocol](../PROTOCOL.md) and
[results](../RESULTS.md) for what was measured and the limits of the findings.

## Included and excluded runs

The final cohort contains both `neural` and `islands`, each at seeds 11 and 29:

- `2026-09-09T20-10-03.419Z-neural-11`
- `2026-09-09T20-10-11.217Z-islands-11`
- `2026-09-09T20-10-18.800Z-neural-29`
- `2026-09-09T20-10-27.198Z-islands-29`

The earlier `2026-09-09T20-08-22.166Z-neural-11` canary is retained but excluded
from the final numerical cohort. Its strict runner failed on a favicon 404.
Later review also found a second-lane parameter-reset alias in that version, so
its numerical comparisons are not treated as final evidence. The original
canary report was left unchanged in the local research archive.

## Integrity and the one redaction

[manifest.json](manifest.json) uses the schema
`trainable-tsl-public-evidence-v1`. Each entry identifies its cohort membership,
compressed filename, byte lengths, redactions and exclusion reason. Its hashes
have distinct meanings:

- `originalRawSHA256` hashes the original report bytes, verified against the
  original `report.sha256` sidecar before packaging.
- `publicJSONSHA256` hashes the JSON bytes after decompression.
- `gzipSHA256` hashes the committed compressed file.

For all four included reports, decompression recovers the exact original bytes;
their original and public JSON hashes are identical. No report fields were
omitted or rewritten, and the included reports contained no absolute user or
workspace paths.

For the excluded canary only, the absolute workspace prefix in the `/failure`
stack-trace string was replaced with `<workspace>`. Every other JSON byte was
preserved. The manifest explicitly records that redaction and both hashes;
the original raw hash identifies the unmodified local report, which cannot be
reconstructed from the redacted public file alone. All five compressed files
were checked by decompressing and hashing their complete contents.
