# Complete retained records

The initial numerical canary passed. The initial interaction report retains its failed all-solves-converged gate (one plant solve reached its time budget), and the source UI record retains three near-converged line-search failures. These records were not replaced.

A documented energy-roundoff fix preserves the convergence threshold and requires real projected-gradient reduction. The separate cpu-fixed-report and its independent audit passed; interaction-fixed-report passed124/124 solves under the same inputs and settings. Eight GPU covariance cases and a separate NumPy audit are included. These are bounded numerical checks, not proof of calibrated physical fidelity.

Each .json.gz expands to the exact original JSON bytes; manifest.json records raw and compressed SHA256 hashes. No report fields were removed or redacted. Embedded source snapshots preserve the implementation evaluated at each stage.

From the repository root, with Python and NumPy available:

```sh
python experiments/local-elasticity/reference.py --report experiments/local-elasticity/evidence/cpu-fixed-report.json.gz
python experiments/local-elasticity/reference.py --report experiments/local-elasticity/evidence/gpu-covariance-report.json.gz
```

The first command independently rebuilds RKPM matrices/eigenspace, energy/forces, Hessians and four steps per asset. It verifies the raw checksum and embedded source hashes. The second recomputes the full deformation Jacobian and covariance transformation from retained Float32 GPU inputs, without using a GPU. CPU PyTorch is only needed for the optional reference.py --self-test. The interaction report retains its own complete trajectory and held-out data; the four-step verifier does not certify that separate trajectory gate.

## Packaged UI and lifecycle records

- bundle-ui-before-reset-fix.json.gz: Passed earlier packaged grab/release/reset-after-idle cases; superseded by the later reset-in-flight and count-boundary checks.
- bundle-ui-final.json.gz: Passed final packaged plant grab/release/reset-while-pending, Spot file upload, early count rejection;46/46 solves converged,60frames,no external requests.
- lifecycle-final.json.gz: Passed4/4 revised-source cancellation/error-cleanup cases;3 sequential devices destroyed,1render,no unexpected errors.

Packaged UI reports name the exact package manifest they tested. The sealed ZIP is not modified when later QA records are added here. Earlier passes cover their recorded cases and source version; they do not retroactively cover later boundary tests.
