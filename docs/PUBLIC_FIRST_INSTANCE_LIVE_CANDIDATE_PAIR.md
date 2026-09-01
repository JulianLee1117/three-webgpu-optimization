# Public candidate-pair receipt for the live `firstInstance` study

The public pair receipt connects two sanitized candidate-run bundles to the
disclosed selection records whose timestamps precede timing. It is derived only
after the private series is closed and the existing private pair verifier
accepts exactly two valid candidate matrices. Either matrix may still miss a
numerical or environment decision gate; a non-pass is preserved unchanged.

The local hash chains and tag checks do not independently establish external
preexistence. That stronger claim requires a remote observer to retain or fetch
the tag-object ID and ref before timing.

The builder takes one finalized private series, the two already-sanitized public
run directories, and a new output directory:

```sh
npm run build:public-pair:first-instance-live -- results/candidate-series/<derived-series> results/public/<matrix-1-run-id> results/public/<matrix-2-run-id> results/public-pairs/<pair-id>
```

It refuses an existing output or an output inside an input. The private pair
verifier and strict public run verifier both run before any output is committed.
The builder re-runs the committed sanitizer on each exact ledger-selected
private run and requires the complete derived directory, including its
manifest, to be byte-identical to the supplied public bundle. It also verifies
that each public bundle's private-manifest commitment equals the SHA-256 of the
corresponding finalized private `artifact-manifest.json`. Start/end snapshots
must show that both selected private runs and both public inputs remained
unchanged throughout derivation.

## Closed receipt bundle

The output directory contains exactly:

- `candidate-series-registry.jsonl`: the byte-exact, hash-chained root registry;
- `candidate-attempts.jsonl`: the byte-exact, hash-chained series ledger;
- `public-candidate-pair-receipt.json`: the safe private-verifier projection,
  disclosed selection bindings, decisions, and public-run bindings;
- `public-candidate-pair-manifest.json`: byte counts and SHA-256 values for the
  other three files.

The receipt binds each matrix ordinal, attempt ordinal, run ID, reservation-event
digest, private-manifest commitment, and public `artifact-manifest.json` byte
count and SHA-256. It also binds the candidate commit and tree, sanitizer policy,
root-registry claim and materialization, ledger final event, and canonical
annotated-tag message digest. It contains no absolute input or output path.

The disclosed registry and ledger have closed schemas. Their directory names
are portable relative names derived by the candidate orchestrator. Neither file
contains process IDs, process names, GPU UUIDs, browser profile paths, or machine
home paths. The receipt projects only the fields used by the pair decision and
rejects private-identifier keys or machine-local paths.

## Independent verification

Given the receipt bundle and the same two public-derived run directories, run:

```sh
npm run verify:public-pair:first-instance-live -- results/public-pairs/<pair-id> results/public/<matrix-1-run-id> results/public/<matrix-2-run-id>
```

The public verifier:

1. rejects a symlink, missing entry, extra entry, byte-count mismatch, or digest
   mismatch in the receipt bundle;
2. replays the root-registry and series-ledger hash chains, verifies the selected
   source claim, cross-file claim/opening/materialization/first-reservation
   chronology, and materialization, then checks the matching current local
   annotated-tag target and canonical message;
3. independently accepts both supplied run directories with the strict
   public-derived run verifier and rejects either directory if it changes during
   individual or pair-level verification;
4. matches each public reservation and run ID to the selected valid ledger
   attempt, requiring reservation time <= run start <= run completion <= ledger
   finalization time;
5. checks same-source, dependency, browser, backend, adapter, pseudonymized GPU,
   workload, and non-overlapping-session identity across matrices;
6. recomputes both numerical, environment, and overall run decisions from the
   public artifacts, then recomputes pair eligibility and the conjunctive pair
   decision;
7. matches the supplied public manifest bytes to the SHA-256 and byte counts in
   the receipt.

The public run-directory order is irrelevant; matrix ordinals determine the
canonical order. Verification reports portable run IDs and hashes, not local
filesystem locations.

## Evidence boundary

The public receipt reproduces the disclosed selection chronology, selected-run
bindings, both run decisions, and the final pair decision. It does not include
the private attempt directories. Consequently, content digests for private
runner logs, lifecycle records, source-after captures, invalid attempts, and
other private attempt bytes remain commitments. The verifier can replay the
ledger's recorded classifications but cannot independently reconstruct a
private retry or failure classification from bytes that were not disclosed.

The private pair verifier checks the exact on-disk root inventory, including
every series declared by the registry. The public bundle discloses the root
registry and the selected series ledger, not every other private series
directory. A public-only verifier therefore cannot reconstruct that private
root inventory and reports it as unverified.

The deterministic sanitizer re-derivation is a private builder gate. A
public-only verifier cannot repeat that derivation without the selected private
run bytes. Likewise, each public run records and the builder checks the
corresponding private-manifest digest, but a public-only verifier does not
possess those private manifest bytes. It therefore reports that
private-original manifest bytes and private attempt-content commitments are not
verified.

The registry tag, hash chains, receipt manifest, and run manifests provide
integrity and consistency checks. The verifier checks the current local tag
target and canonical message, not an immutable or externally timestamped tag
object. None of these records establishes author identity, external
preexistence, or external authenticity.
