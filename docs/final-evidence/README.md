# Additional evidence retained at closeout

The three gzip files preserve the **unchanged original ZipDepth report bytes**
from the two CPU fixtures and the browser reproduction described in the
[experiment documentation](../../experiments/zipdepth-export/README.md).
Their original SHA-256 sidecars were checked before packaging. The
[manifest](manifest.json) records compressed and uncompressed hashes.

`zipdepth-proof.png` is the original browser screenshot. Reports retain their
original source identities and dates. This is preservation of existing evidence,
not a fresh model inference or independent hardware attestation. The public
reports contain metrics and artifact hashes; complete numeric arrays and ONNX
models remain in the verified local research archive. Rebuilding the model is
documented in the experiment README.

Verify the archived reports without installing dependencies:

```sh
node docs/final-evidence/verify.mjs
```
