# Public evidence derivation for the live `firstInstance` study

Candidate runs are retained as private originals. The committed sanitizer
leaves them unchanged, derives a separate public bundle, preserves
every manifest-bound artifact, and recomputes the byte length and SHA-256 of
every derived artifact. It refuses a
source artifact that does not match the private manifest, a non-candidate or
incomplete live run, unavailable or malformed telemetry, an existing output
directory, or an output located inside the private run. An observed pre/post
process-set mismatch is preserved: it remains a completed candidate with a
failed, non-replaceable environment gate and overall evidence decision.

Run it only after the private candidate has finalized:

```sh
npm run sanitize:first-instance-live -- results/candidate-series/<derived-series>/<attempt>/runs/<private-run-id> results/public/<public-run-id>
```

Successful command output contains portable bundle labels, the run ID, source
and implementation commitments, manifest digests, and the artifact count. It
does not print the resolved private or public filesystem path.

The source directory is read twice and is never written. The private bundle is
independently accepted by the strict directory verifier before derivation.
Output is assembled in a new staging directory, independently accepted by the
same verifier, and renamed into the requested new directory only after the
private artifacts and clean candidate source are confirmed unchanged.

## Frozen redaction allowlist

No JSON value outside the following normalized JSON paths may change. `[*]`
means each element at that one declared array location; it is not a recursive
wildcard.

| Artifact | JSON path | Fixed public replacement |
| --- | --- | --- |
| `metadata.json` | `$.environment.note` | `[redacted: unrelated private environment note]` |
| `metadata.json` | `$.environment.gpuTelemetry.summary.gpus[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `metadata.json` | `$.environment.gpuTelemetry.coverageAudit.gpuIdentities[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `metadata.json` | `$.environment.gpuTelemetry.computeProcesses.pre.processes[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `metadata.json` | `$.environment.gpuTelemetry.computeProcesses.pre.processes[*].pid` | `100000 + one-based-process-ordinal` |
| `metadata.json` | `$.environment.gpuTelemetry.computeProcesses.pre.processes[*].processName` | `resident-gpu-process-{one-based-process-ordinal}.redacted` |
| `metadata.json` | `$.environment.gpuTelemetry.computeProcesses.post.processes[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `metadata.json` | `$.environment.gpuTelemetry.computeProcesses.post.processes[*].pid` | `100000 + one-based-process-ordinal` |
| `metadata.json` | `$.environment.gpuTelemetry.computeProcesses.post.processes[*].processName` | `resident-gpu-process-{one-based-process-ordinal}.redacted` |
| `metadata.json` | `$.liveFirstInstanceEnvironmentAudit.computeProcessIdentityComparison.pre[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `metadata.json` | `$.liveFirstInstanceEnvironmentAudit.computeProcessIdentityComparison.pre[*].pid` | `100000 + one-based-process-ordinal` |
| `metadata.json` | `$.liveFirstInstanceEnvironmentAudit.computeProcessIdentityComparison.pre[*].processName` | `resident-gpu-process-{one-based-process-ordinal}.redacted` |
| `metadata.json` | `$.liveFirstInstanceEnvironmentAudit.computeProcessIdentityComparison.post[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `metadata.json` | `$.liveFirstInstanceEnvironmentAudit.computeProcessIdentityComparison.post[*].pid` | `100000 + one-based-process-ordinal` |
| `metadata.json` | `$.liveFirstInstanceEnvironmentAudit.computeProcessIdentityComparison.post[*].processName` | `resident-gpu-process-{one-based-process-ordinal}.redacted` |
| `gpu-telemetry-summary.json` | `$.summary.gpus[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `gpu-telemetry-summary.json` | `$.coverageAudit.gpuIdentities[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `gpu-telemetry-summary.json` | `$.computeProcesses.pre.processes[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `gpu-telemetry-summary.json` | `$.computeProcesses.pre.processes[*].pid` | `100000 + one-based-process-ordinal` |
| `gpu-telemetry-summary.json` | `$.computeProcesses.pre.processes[*].processName` | `resident-gpu-process-{one-based-process-ordinal}.redacted` |
| `gpu-telemetry-summary.json` | `$.computeProcesses.post.processes[*].gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |
| `gpu-telemetry-summary.json` | `$.computeProcesses.post.processes[*].pid` | `100000 + one-based-process-ordinal` |
| `gpu-telemetry-summary.json` | `$.computeProcesses.post.processes[*].processName` | `resident-gpu-process-{one-based-process-ordinal}.redacted` |

The only CSV redaction is:

| Artifact | Column | Fixed public replacement |
| --- | --- | --- |
| `gpu-telemetry.csv` | `gpuUuid` | `GPU-PUBLIC-DEVICE-{zero-based-device-ordinal}` |

Device ordinals are assigned by code-point sorting the private UUID set.
Process ordinals are assigned by code-point GPU UUID, numeric PID, then
code-point process-name order over the union of the private pre-run and post-run
`(gpuUuid, pid, processName)` tuples. The templates are independent of timing
and result values. They preserve device grouping, each phase's process-set
cardinality, distinct process identities, and whether the sets are equal
without publishing the original identifiers. Memory values and all
research-relevant telemetry remain unchanged.

The sanitizer parses every JSON artifact and rejects a sensitive key at any
other shape. It also rejects Windows home/drive paths, Unix home paths, file
URLs, and UNC paths outside the one environment-note path, rather than trying
to rewrite them recursively. Every JSON transformation is compared to its
private value tree, and a changed value outside the table is fatal. CSV schemas
and row widths are checked before the one named column is replaced.

## Manifest and provenance boundary

The public `artifact-manifest.json` retains the run ID, file roles, and required
and optional declarations from the private manifest, then recomputes each
present file's byte count and SHA-256 from the derived bytes. Its
`bundleProvenance` record labels the output `public-derived`, labels its source
`private-original`, records the SHA-256 of the unchanged private
`artifact-manifest.json`, records the clean candidate commit and tree, and
embeds the exact allowlists above. It also records the SHA-256 of every source
file in the four-file derived-byte/provenance closure:
`scripts/live-evidence-sanitizer-policy.mjs`,
`scripts/nvidia-telemetry.mjs`, `scripts/sanitize-live-evidence.mjs`, and
`scripts/source-provenance.mjs`. Those hashes are computed from the recorded
commit's Git blobs. The recorded candidate tree binds the rest of the tracked
repository. The sanitizer checks that its executing closure and start/end clean
source provenance match that commit; the analyzer resolves the same Git objects
independently, so a later checkout does not change the meaning of an existing
bundle.

The public manifest provides an integrity check only for the derived artifacts
named in its `files` array. It is not a signature and does not establish author
authenticity. The recorded private-manifest digest is a commitment to the
private source manifest; without that private manifest, the public bundle
cannot verify its relationship to the omitted private artifact bytes. The
strict verifier also rejects symlinks and any undeclared directory entry.
