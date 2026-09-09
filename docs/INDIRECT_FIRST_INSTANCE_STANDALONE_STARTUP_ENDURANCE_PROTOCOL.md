# Standalone startup-endurance qualification protocol

## Purpose and decision boundary

This protocol is a fail-closed infrastructure qualification for the corrected
v2 standalone indirect-`firstInstance` candidate. It was added after the v1
capture stopped on a page-readiness timeout aligned with Windows Tcpip Event ID
4231. Its only question is whether the host can repeatedly start, configure,
and cleanly disconnect the candidate's fresh browser/device topology while the
Windows dynamic TCP port state remains within fixed bounds.

The probe is permanently `analysisEligible: false`. It does not measure either
lane, estimate a portable-versus-feature effect, amend the candidate's sample
size, or provide evidence for any efficacy claim. A pass permits the v2
candidate to start; it does not contribute to or predict the candidate
decision.

## Frozen execution

The probe first runs one disposable strict-portable forced-feature-off gate,
then runs the first eight complete quartets from standalone Matrix 1 in this
exact order:

```text
AX AY BX BY  AX AY BX BY
```

That is 33 sequential fresh browser/device lifecycles: the gate followed by
exactly 32 scheduled sessions, comprising 16 portable and 16 feature with both
`H` and `L` initial visibility orders balanced. Each session uses the same
strict boot query, Chromium arguments, WebGPU backend, viewport, workload, and
selected-lane factory as the candidate. It creates a fresh Chromium process,
context, page, renderer, adapter/device, and Playwright temporary profile. The
preceding browser must be observed disconnected, followed by the same fixed
two-second post-disconnect interval, before the next launch. One Vite server
serves the complete probe and may not restart.

Each page must reach readiness and capture the selected configuration. It must
show the expected Three.js revision and WebGPU backend, the real
`indirect-first-instance` capability, cross-origin isolation, reversed depth,
timestamp availability, clean WebGPU error/device-loss records, and the fixed
viewport and workload. Exactly one assigned lane may be constructed and
primed; the absent lane must remain unconstructed. The resource ledger must
retain one indirect command buffer, two compute nodes, one material, one mesh,
and one bundle with the lane-appropriate address mode. The exact untimed
timestamp-pool pre-prime is allowed.

The probe may not invoke `startTrial`, enter warmup or measurement, retain a
timing row, compute an estimator, or issue a numerical decision. The disposable
forced-feature-off gate performs only its existing correctness/shader-evidence
challenge; none of the 32 scheduled sessions performs a shader challenge. The
assigned initial visibility exists only to exercise the candidate's strict
startup configuration.

## Fixed Windows TCP readiness gates

A bounded aggregate TCP snapshot is captured immediately before every browser
launch and after every observed disconnect plus its two-second delay. A
complete probe therefore retains 66 scheduled snapshots around 33 browsers.
Every snapshot must
pass all of the following conditions:

- the configured Windows TCP settings and dynamic port ranges are available
  and internally consistent;
- the bounded Windows System-log query for Tcpip Event ID 4231 is available
  and parseable;
- the latest Event 4231, if present, is more than 10 minutes old;
- every reported dynamic range has at least
  `max(4096, ceil(0.25 * numberOfPorts))` free unique local ports.

For each range, the free margin is the configured range size minus the number
of unique local ports currently used by TCP connections in that range. Total
connections, unique local ports, `TIME_WAIT` connections, and unique
`TIME_WAIT` local ports are retained as aggregate context. No per-connection
row is retained.

These gates are deliberately conservative, not a model of the Windows TCP
allocator. Allocation can also depend on tuple, address family, and network
compartment, and Event 4231 does not identify the exhausting process. Any
query or parse failure is therefore a failure, not missing-at-random evidence.

## Pass, failure, and stopping rules

The qualification passes only if the forced-feature-off gate and all 32
sessions complete in order, all 66 scheduled TCP snapshots pass, every
page/configuration and cross-session environment check passes, all 33 browser
lifecycles are sequential and clean, the Vite audit records exactly 33
successful entry documents and the expected standalone module, and source plus
installed dependency bytes match from start to finish.

There is one predeclared probe attempt, zero retries, and no session
replacement. The first failure stops the probe. A failed or interrupted
artifact is retained unchanged and blocks the corresponding v2 candidate
series; another probe may not be launched merely to obtain a passing interval.
Any correction requires another explicitly documented clean source identity
and series.

The fixed result root below must be absent before launch. The probe claims it
with a non-recursive exclusive directory creation; once that root exists, a
second invocation fails before it can create another random run directory.
Thus a failed, interrupted, or completed invocation consumes the sole attempt.

The successful probe qualifies only the exact clean commit and installed
dependency closure it records. Editing tracked source, changing dependencies,
or changing the candidate execution path after the probe invalidates the
qualification. Ordinary quiet-GPU and process-set readiness checks still apply
before the candidate; the startup probe does not replace them.

## Evidence and execution

Run from the repository root on Windows:

```powershell
npm.cmd run probe:first-instance-standalone-startup-endurance
```

Artifacts are written incrementally under:

```text
results/development/first-instance-standalone-startup-endurance/
  first-instance-standalone-startup-endurance-<timestamp>-<nonce>/
```

For this immediate v2 workflow, qualification-to-candidate binding is enforced
operationally by keeping the tree and installed dependency bytes unchanged and
comparing their recorded identities. There is not yet a separate standalone
qualification artifact verifier; add one before generalizing the
result beyond this controlled launch.

The completion manifest must state `analysisEligible: false`, contain no
numerical decision or efficacy conclusion, and bind the frozen plan, source and
dependency identities, the forced-feature-off gate, 32 session records, 33
browser lifecycles, 66 TCP snapshots, the Vite runtime audit, and incremental
journal. A failed directory is evidence of
the qualification failure and must not be deleted or treated as a warmup.

Only after a complete pass, and without source or dependency changes, may the
single corrected full capture write to:

```text
results/candidate-standalone-deployment-v2/
  first-instance-standalone-deployment-v2-<timestamp>-<nonce>/
```

The full candidate remains governed by the
[standalone deployment protocol](INDIRECT_FIRST_INSTANCE_STANDALONE_DEPLOYMENT_PROTOCOL.md),
including its original workload, two-matrix schedule, correctness gates,
estimators, interaction bounds, numerical thresholds, and no-replacement rule.
