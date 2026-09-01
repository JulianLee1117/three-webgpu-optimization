# Public candidate-pair package for the live `firstInstance` study

The publication package is a recorded-encoder-deterministic, lossless `.tar.br`
containing one verified public candidate-pair receipt bundle and the two
sanitized public run directories it binds. It contains no private run
directory, private artifact, credential, or machine-local output path.

## Build

Create a new archive only after the public pair receipt and both public runs are
final:

```sh
npm run build:public-package:first-instance-live -- results/public-pairs/<pair-id> results/public/<matrix-1-run-id> results/public/<matrix-2-run-id> results/public-packages/<pair-id>.tar.br
```

Before constructing archive bytes, the builder invokes the existing strict
public-pair verifier on those exact inputs. That gate replays the disclosed
selection records, independently verifies both public run directories, and
recomputes their run and pair decisions. Start/end byte snapshots reject an
input change during either verification or package construction. The output
must be new, must not be inside an input directory, and must be smaller than
100,000,000 bytes.

Packaging must run under the exact Node version recorded by both public runs.
The internal manifest records that Node version together with the build
runtime's Brotli and zlib versions. A mismatch between either run and the build
runtime fails before encoding.

The archive contains these namespaces:

- `pair/`: exactly the four files in the verified public receipt bundle;
- `runs/matrix-1/` and `runs/matrix-2/`: the complete sanitized run artifacts;
- `public-candidate-pair-package-manifest.json`: the package policy, verified
  pair decision, pair-manifest commitment, both public artifact-manifest
  commitments, and sorted SHA-256/byte commitments for every source file.

Only non-symlink regular files are accepted. The package manifest and CLI output
use portable labels and run IDs, not absolute paths.

## Canonical encoding

Policy `first-instance-live-public-candidate-pair-tar-br-v1` fixes:

- Unicode-code-point path order and normalized portable relative paths;
- one regular ustar entry per file, with mode `0644`, UID/GID/mtime zero, no
  owner names, no device numbers, zero padding, and exactly two terminal zero
  blocks;
- Brotli generic mode, quality 9, window 22, and the exact uncompressed size as
  the encoder size hint;
- no more than 64 entries, 256 MiB for any one file, or 512 MiB for the complete
  canonical tar stream, including headers, padding, and terminator.

The same accepted source bytes produce the same archive bytes when the recorded
Node, Brotli, and zlib encoder identity is the same. The builder checks this
same-runtime determinism. Compressed-byte determinism is not claimed across a
different encoder identity.

## Verify without extraction

```sh
npm run verify:public-package:first-instance-live -- results/public-packages/<pair-id>.tar.br
```

The verifier decompresses and parses the archive in memory. It does not create
files or follow archive paths. It rejects an absolute, parent-relative,
backslash, duplicate, overlong, out-of-order, extra, symlink, or non-regular
entry; noncanonical tar metadata or padding; an invalid Brotli stream; an
archive at or above the compressed size limit; more than 64 entries; a file
above 256 MiB; and a canonical tar stream above 512 MiB.

It then validates the exact package schema and namespaces, every source-file
byte count and SHA-256, the embedded receipt-bundle manifest, receipt schema,
registry and ledger hash chains, final chain bindings, both public artifact
manifests, each run's complete declared-file inventory, public sanitizer
provenance shape, run IDs, matrix/attempt reservations, candidate commit/tree,
and receipt commitments. A final byte read rejects an archive that changes
during verification.

The verifier uses the local Brotli decoder but does not recompress the stream.
It validates the decompressed canonical tar and all embedded commitments, binds
the recorded encoder's Node version to both packaged runs, and reports that
same-encoder compressed-byte reproduction was not checked. This permits
verification on a compatible runtime whose encoder emits a different valid
Brotli representation.

## Evidence boundary

The strict public-pair verifier is a mandatory builder gate. The archive-only
verifier confirms the resulting byte-level package, disclosed chain, receipt,
and public-run bindings without extraction; it does not replay the complete
scientific analyzer from the embedded artifacts. Recomputing the full run and
pair decisions independently requires presenting the losslessly recovered
receipt bundle and run directories to
`verify:public-pair:first-instance-live`.

The package discloses no private attempt bytes. Private attempt-content and
private artifact-manifest digests therefore remain commitments unless their
original bytes are separately disclosed. SHA-256 manifests and hash chains
provide integrity and consistency, not an author signature, external timestamp,
or authenticity proof.
