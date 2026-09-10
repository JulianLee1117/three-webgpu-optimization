# Retained Light Vault verification

[report.json](report.json) is the unchanged output of the bounded browser run
started at 2026-09-10 05:12:25 UTC. All 61 checks passed in 18.2 seconds, using
one isolated headless Chrome 152.0.7977.83 session on the local Windows/RTX 5070 Ti
setup. The browser closed at the end. The only warnings concern Chrome ignoring
`powerPreference` on Windows.

The report contains before/after source hashes, hashes of served files and
screenshots, numerical comparisons against the float64 CPU reference, captures,
and a newly fitted pair of images. Moving the card never changes the source
phase. Capture reads actual GPU intensity, and an ambiguous field is rejected.
Reset during capture cannot restore a stale memory; disposal waits for readback.

The [saved HI/BY seal](light-vault-seal.json) was downloaded through the UI,
reopened through the file input after reloading the page, and used to unlock the
vault with two actual optical captures. Its references are unencrypted metadata
for checking images. They are never inputs to GPU propagation. Open this file
with **Make your own seal → Open a saved seal**.

The custom fit took 4006.4 ms. A warmed, two-second pointer sweep delivered 40
inputs and 40 recorded updates, with median 3.6 ms and p95 5.1 ms from the input
event to completion of that render's GPU submission. This includes the wave
calculation and scene rendering. It excludes presentation to a physical display
and is not a frame-rate benchmark or general hardware guarantee.

Maximum relative L2 errors over the checked browser states were less than
6.46 × 10⁻⁶ for the complex field and 2.17 × 10⁻⁶ for the actual intensity texture.
These test agreement between numerical implementations, not accuracy against
a physical optical experiment. [PHYSICS.md](../PHYSICS.md) documents sampling,
finite-aperture checks, cross-talk and other limitations.

[cpu-tests.txt](cpu-tests.txt) records 23 passing CPU checks. The volume test
independently reconstructs all 820 points in the first and last preview slices.
The fixed-pitch padding check uses the committed phase binary without refitting.

The PNGs are unedited screenshots from the browser run. `mobile.png` exercises
a 390 × 844 desktop-browser viewport; it is not evidence from a physical phone.
