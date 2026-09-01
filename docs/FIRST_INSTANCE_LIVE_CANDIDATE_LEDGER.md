# Live `firstInstance` candidate-series ledger

The live compute-plus-render experiment uses a committed candidate orchestrator,
a canonical root registry, and an independent pair verifier. This closes the
selection boundary around the protocol's exactly-two-matrix rule: one tracked
study identity has one derived series directory, every browser/device attempt
is reserved before the runner starts, and every preserved attempt is classified
without consulting its performance result.

## Canonical root claim and external anchor

Initialization is a separate, non-timing phase and requires the clean frozen
candidate commit:

```sh
npm run candidate:init:first-instance-live
```

The study key is exactly the SHA-256 of canonical JSON containing
`experimentId`, Git tree, tracked-file digest, and package-lock digest. The Git
commit is deliberately omitted from key uniqueness, so an empty commit over the
same tree cannot open another series. The root `candidate-series-registry.jsonl`
claim nevertheless binds the exact commit and the full source identity,
including the installed execution-dependency closure. Changing `node_modules`
under the same tracked study therefore rejects instead of deriving another
series. A changed tracked tree derives a visibly different study key.

The series basename is
`first-device-live-<first-16-study-key-hex>`. Initialization appends and syncs a
`source-claimed` event, opens that one series ledger, then appends and syncs a
`series-materialized` event that binds the series ID and opening-event digest.
A root lock serializes this sequence. Verification requires claim time ≤ series
opening time ≤ materialization time across the two hash chains. A crash after
the claim can resume only that pending claim; a deleted or mismatched
materialized series is terminal.

Initialization prints one deterministic annotated-tag name, target commit, and
canonical message. Create that exact annotated tag and publish it before any
candidate attempt. The ordinary run command refuses to start unless the local
tag is annotated, targets the frozen commit, and contains the exact claim,
series-opening, and materialization digests. Remote publication is checked as
an operational pre-run step; local code verifies the current annotated tag's
target commit and canonical message.
Using the three exact reported values, the corresponding Git operations are:

```sh
git tag -a <reported-tag-name> <reported-target-commit> -m '<reported-canonical-message>'
git push origin refs/tags/<reported-tag-name>
```

The root registry and exact inventory prevent an alternate folder, copied
ledger, sibling series, empty commit, or changed installed closure from
resetting the two-matrix count. The pair verifier rechecks every registered
sibling ledger, rejects either root or series locks, and reports the study key,
registry digests, and anchor-tag digest. Deleting both the ignored registry root
and its series remains locally indistinguishable from first initialization;
the published tag is an external checkpoint only for an observer that fetched
or otherwise retained its object ID or ref before timing. Neither hash chain nor
tag establishes authorship.

## Running the frozen candidate series

After the reported annotated tag has been created and published, start or
resume the one derived candidate series from the same clean frozen commit:

```sh
npm run candidate:first-instance-live
```

The command continues through the frozen series. A preregistered numerical miss
or a stable compute-process-set mismatch still consumes its matrix slot, so
matrix 2 runs unchanged. A permitted infrastructure failure keeps the same
matrix ordinal and stops for infrastructure remediation; rerunning the same
command then uses a fresh browser/device session for that ordinal. Once matrix
1 is valid, the command proceeds directly to matrix 2 and invokes the pair
verifier after both complete. A nonretryable implementation/evidence failure
ends the source series immediately.

The pair can also be verified independently:

```sh
npm run verify:candidate:first-instance-live -- results/candidate-series/first-device-live-<16-hex-study-key-prefix>
```

Use the exact derived directory reported by initialization in place of the
illustrative basename above. Verification also requires the annotated tag to be
available locally.

The result is a structured decision. `decision.pass` is true only when both
matrices independently pass the preregistered numerical decision and the full
environment/evidence decision.

## Append-only event model

The root registry and each `candidate-attempts.jsonl` are append-only event
streams with internally
reproducible chronology and content commitments. The opening event binds the
series to the clean Git commit, tree, tracked-file
digest, dependency-lock digest, and installed execution-dependency closure.
That closure hashes every regular file under `node_modules` in sorted relative
path order, excluding only `.bin/**` tool shims and generated `.vite*/**`
caches; symlinks and other entry types outside those exclusions are rejected.
Candidate serving uses Vite in custom-app mode, disables dependency discovery
and optimization, serves the exact retained tracked `index.html`, and uses a
fresh external cache that must remain empty. The guard rejects decoded
`/@vite/*`, `/@id/*`, and excluded `.vite*` requests, module IDs, and
references. Its runtime audit commits the canonical source, exact 2xx browser
response variants, response counts, and security/cache headers for every
requested project or `node_modules` JavaScript/CSS resource. The verifier
rehashes every retained source, reconstructs the audit aggregate and counts,
and requires the captured transformed-response commitments to be byte-identical
across the two matrices.
Each attempt then has exactly two events:

1. `attempt-reserved`, appended and synced before the benchmark child process
   starts;
2. `attempt-finalized`, binding the exit disposition, independently derived
   classification, run directory, Git and installed-dependency state after the
   child exits, and a recursive content digest of all preserved attempt files.

Attempt and event ordinals must be contiguous. A retry retains its candidate
matrix ordinal. The series-directory inventory must exactly equal the attempt
directories named in the ledger, so an extra or omitted attempt fails
verification. Interrupted reservations are finalized from their preserved
files only after an exact closed-child lifecycle marker is present; a live
child or ambiguous missing completion marker fails closed and cannot overlap a
new attempt. Every candidate `metadata.json` also contains a structured
reservation record with the series ID, reservation-event digest,
matrix/attempt ordinals, source commit/tree, and dependency-closure digest. The
pair verifier matches it back to the pre-timing event exactly.

Every event commits to the preceding event, and the verifier recomputes the
root chain, series chain, exact namespace, and attempt-content commitments. The
artifact manifests, registry, ledger, and current local tag establish
reproducible byte consistency and an internally consistent reported chronology.
External pre-timing chronology additionally depends on a remote observer having
fetched or retained the tag-object ID or ref before timing. None of these
records is a signature or establishes authorship or authenticity.

## Closed classifications

The final classification is one of exactly:

| Classification | Effect |
| --- | --- |
| `valid-candidate` | Consumes matrix 1 or 2, independent of its numerical or stable-process-set result. |
| `infrastructure-invalid-retryable` | Preserves the attempt and permits a fresh-session retry of the same matrix ordinal. |
| `implementation/evidence-failure-nonretryable` | Terminates the frozen-source series. A correction requires a new clean commit and a new two-matrix series. |

Retryable reasons are closed to:

- browser or device loss;
- telemetry-collector failure, including invalid or missing process snapshots;
- pre-metadata run-artifact incompleteness backed by an exact spawn failure,
  operating-system termination, Windows abnormal status, or the frozen
  runner's allowlisted artifact-persistence I/O marker.

Artifact loss is not itself retry provenance. The independently reconstructed
`classificationEvidence.artifactInfrastructureFailureObserved` field is true
only for one of those closed infrastructure records. Spawn/termination records
require that no readable benchmark metadata exists. A filesystem-I/O marker may
also bind readable failed metadata, but only when that metadata's error record
contains the same allowlisted I/O code. Complete metadata cannot be discarded
by deleting or corrupting its manifest, and failed shader, validation, binding,
command, or other implementation metadata cannot borrow a later I/O marker.
A run accepted by the strict verifier also requires a zero child exit with no
signal; byte-valid artifacts do not override a contradictory child disposition.

The verifier derives those reasons from the preserved metadata, child
lifecycle, telemetry disposition, artifact manifest, and byte inventory. An
ordinary WebGPU validation, shader, binding, command, membership, address,
output, lifecycle, schedule, provenance, or post-hoc verification failure is
nonretryable. Performance, drift, nuisance interactions, stable
compute-process-set mismatch, directional results, and near-threshold results
never enter attempt classification.

## Pair identity and decision

Each valid run is first passed through the repository's strict run-directory
verifier and analyzer. Its run provenance must exactly match the source
identity reserved before timing. The two matrices must then match in:

- clean commit, tree, tracked-source digest, and dependency-lock digest;
- Node platform and architecture, browser executable/build/launch identity,
  and user agent;
- WebGPU backend, physical adapter fields and driver, and telemetry GPU UUID
  set;
- the frozen single-NVIDIA-device adapter-to-telemetry name association and its
  independently reconstructed commitments;
- feature availability, timestamp support, cross-origin isolation, viewport,
  reversed-depth state, and storage-buffer limit;
- exact protocol and workload commitments.

The matrices must have distinct run IDs and chronological, non-overlapping
sessions. Timer quantum is checked independently against the frozen upper bound
inside each run; measured quantum is not required to be numerically identical
between sessions.
