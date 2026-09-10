# Light Vault: physical and numerical contract

This experiment fits **one unchanged phase-only aperture** whose propagated intensity resembles different drawings at different distances. The mechanism is established multi-plane computer-generated holography. The useful implementation goal is a small, inspectable browser component: editable targets, a fixed phase plate, and a movable observation plane calculated from the same wave field. It is not a new law of optics or the first browser holography implementation.

## Implemented model

`wave.mjs` is an original float64 JavaScript implementation of a separable complex FFT, scalar angular-spectrum propagation, and equal-weight multi-plane Gerchberg–Saxton-style alternating projection. It supplies the CPU reference and the 80-iteration fit in `worker.mjs`. `gpu.mjs` implements the forward propagator as native Three.js r186 TSL compute: float32 complex buffers, one transfer-function initialization, and 16 radix-2 inverse-FFT passes for the 256² grid. The final pass writes actual intensity to an `rgba32float` storage texture. No target image enters the GPU propagation module.

The committed introductory plate uses:

- A 256 × 256 complex grid with 10 µm sample pitch: a 2.56 mm square computational window.
- A circular, uniformly illuminated aperture with radius 54 samples, or 0.54 mm. Its amplitude is zero outside the aperture on every source projection.
- Monochromatic light with wavelength 532 nm, incident normally with uniform phase before the plate.
- A heart target at 12 mm and an A target at 30 mm; both are constrained by one phase array.
- Eighty fitting iterations and random seed 731. Phase samples are committed in `assets/initial-phase.bin` as little-endian float32 radians; `assets/initial-phase.json` records the optics and fit history. New drawings are fitted in a bounded worker and accepted only if both reconstructed images pass the capture criterion.

The original CPU fit produced coarse intended-target cosine similarities of approximately 0.914 and 0.956, versus 0.533 and 0.560 for the other target. These values describe intensity averaged into 4 × 4 pixel bins, not full-resolution image quality. The intermediate 20 mm plane is ambiguous, as expected for two independently specified images rather than a consistent volumetric object. The committed phase is quantized to float32; current checks load that asset directly rather than an ignored development solution.

## Propagation convention

All distances and wavelengths are in metres. Arrays use interleaved real and imaginary components. The forward FFT uses the negative exponential; the inverse uses the positive exponential and divides by N². Unshifted FFT frequency indices are

```text
signed(k) = k, for k < N/2; otherwise k - N
f(k) = signed(k) / (N × pitch)
```

For time dependence exp(−iωt), propagation toward positive z is

```text
u₀(x,y) = A(x,y) exp(iφ(x,y))
u_z = IFFT[ FFT(u₀) H_z ]
I_z = |u_z|²
H_z = exp[i (2π/λ) z (sqrt(1 − s) − 1)]
s = λ²(f_x² + f_y²)
```

The source removes the spatially constant carrier phase 2πz/λ, which does not change intensity. It evaluates the remaining phase in the numerically stable, equivalent form

```text
phase(H_z) = −(2π/λ) z s / (1 + sqrt(1 − s))
```

This is the scalar angular-spectrum transfer function, not the quadratic Fresnel approximation. In the paraxial limit it becomes −πλz(f_x² + f_y²). No additional inverse-distance amplitude factor belongs in this FFT transfer-function formulation.

Samples with s ≥ 1 are discarded, rather than modeling evanescent near fields. With the current 10 µm pitch and 532 nm wavelength, every sampled frequency is propagating. Therefore conjugating H gives an inverse and preserves the discrete complex-field norm. If frequencies are discarded, conjugation is only the adjoint: it cannot restore the removed information.

The physical power is proportional to Σ|u|² × pitch². Checks report the unscaled sample sum; the introductory aperture's input value is 9176. Each observation plane receives the same propagated input power. Power is not divided by the number of target planes. The GPU intensity texture preserves this unscaled convention; exposure and colour are applied only when displaying it.

## Fitting objective and its limitations

Each target is an **amplitude** array. Desired intensity must be square-rooted before fitting. Each plane's target amplitude is normalized to the input energy. At each iteration the solver:

1. Propagates the current aperture field to each target depth.
2. Replaces its amplitude with the desired amplitude while retaining its phase.
3. Back-propagates these constrained complex fields and adds them with equal weights.
4. Retains the argument of that sum, then restores the fixed source aperture amplitude.

The logged loss is mean relative squared amplitude error across planes. It is sampled before that iteration's phase update, so the last logged value is not a separate measurement of the returned final field. Equal weights make this a consensus GS-style method; the current implementation does **not** implement adaptive weighted GS or mixed-region amplitude freedom.

Arbitrary image pairs need not be mutually satisfiable by one phase-only plate. Noise, speckle, imperfect dark regions, and inter-plane cross-talk are real outcomes. A later weighted variant could reweight planes or bright target pixels and reserve an unconstrained noise region, but would remain in the established weighted-GS/MRAF family and would require its own numerical evidence. A third plane should be added only after retaining readable targets and measuring the resulting loss of contrast or efficiency.

## Independent checks and finite-aperture interpretation

Run the current 23 CPU tests, including 11 wave/solver tests, 7 saved-seal tests, and 5 volume-preview checks:

```sh
npm run test:light-vault
```

Run the bounded browser numerical and interaction harness separately:

```sh
npm run check:light-vault
```

The CPU checks cover an independent direct 8 × 8 DFT oracle, FFT round trip and Parseval normalization, an analytic tilted plane wave, propagation followed by its inverse, and the distinct noninvertible case where evanescent frequencies are removed. They also check bounded inputs and fitting, deterministic phase/aperture behavior, committed-asset padding, seal round trips, invalid files, and physically inconsistent phase/reference pairs. No command-line solution path is needed.

The browser harness compares the actual GPU complex field and storage texture with the independent CPU path, and checks capture behavior, user controls, saved seals, and cleanup. The retained [integrated report](evidence/report.json) passes 61 checks; [CPU output](evidence/cpu-tests.txt) records all 23 CPU checks. Maximum relative L2 error across the checked browser states is 6.46 × 10⁻⁶ for the complex field and 2.17 × 10⁻⁶ for the actual GPU intensity texture. This does not establish performance or browser correctness on other devices. Fresh runs write screenshots and source hashes under `results/development/light-vault-browser/`.

The introductory solution was also copied into the centre of a zero-filled 512 × 512 grid at the **same physical pitch, aperture size, and unchanged phase**. No refitting or image normalization was performed. The original CPU comparison of the central 256 × 256 region gave:

- At 12 mm: complex relative L2 error 0.588%; intensity relative L2 error 0.226%. The expanded simulation places 0.00345% of input energy outside the original window.
- At 30 mm: complex relative L2 error 3.167%; intensity relative L2 error 0.379%. Energy outside the original window is 0.1001%.
- Energy in the expanded window's outer 32-pixel border is 0.000403% and 0.003007% at those depths. Expanded total-energy relative error is below 2 × 10⁻¹³.

The current padding test repeats this comparison against the committed float32 phase asset and emits a `PADDING_REPORT` with measured errors. The two designed images are stable against this boundary-size change. That supports interpreting them as diffraction from the specified isolated aperture within the sampled scalar model. It does not establish convergence for a continuous manufactured surface, arbitrary farther depths, newly fitted phase plates, or other wavelengths.

An FFT is periodic, and norm conservation alone cannot expose wraparound. Every extension of the depth range needs padding checks. Angular-spectrum transfer functions can themselves become undersampled at large propagation distances; the present implementation has no additional band-limited-AS filter. Fixed-pitch padding and a separate fixed-geometry sample-refinement check are different tests, and only the former has been completed here.

## What the display may legitimately show

Moving the observation plane recalculates GPU propagation at the requested depth using the same source spectrum. The application does not interpolate target images, blend target textures, or refit the plate during a depth scan. The phase mesh also remains stationary when the decorative vault door opens. The receiver and magnified card sample the same storage texture, using a fixed central crop, 3 × 3 display filter, and common exposure. The plate's colours visualize phase, not its physical appearance.

Capturing reads the actual unfiltered GPU intensity texture. `matchImage()` averages it into 4 × 4 bins and compares it with squared reference amplitudes. A capture requires cosine similarity at least 0.85 and a margin at least 0.15 over the other reference; both distinct matches open the vault. Position alone is not an unlock condition. Reference images affect fitting and verification, but do not supply forward-rendered pixels.

`volume-worker.mjs` independently propagates the same phase to 32 depths from 4 to 38 mm on the CPU. It averages the central region into 32 × 32 bins and displays selected weak points as an intensity visualization. The worker has a four-second computation budget; its optional failure does not affect the GPU receiver. These points suggest scattering for orientation and are not a participating-medium transport solution. The surrounding generated atlas is decorative base-colour artwork, documented in `ASSETS.md`.

The visible source, receiver, and depth spacing are schematically magnified. Physical calculations use SI units; the scene's dimensions must not be read as a fabrication scale.

The setup assumes coherent, monochromatic, collimated illumination. Ordinary sunlight does not satisfy that model. Observation planes are alternative detector positions; multiple opaque screens cannot simultaneously occupy the beam and leave downstream propagation unchanged.

A transmissive relief approximation would use h = λφ/[2π(n−1)] modulo a 2π phase wrap, subject to a material and fabrication model. At refractive index 1.49, a complete phase wrap is about 1.09 µm high. This experiment does not validate manufacturing, a printable STL, phase quantization, material dispersion, vector polarization, reflection losses, or physical calibration.

## Portable seals

`seal.mjs` exports one phase array plus fixed optics and two normalized byte-intensity references in `light-vault-seal-v1` JSON. Parsing enforces the fixed grid, wavelength, pitch, aperture, distances, finite phase in [−π, π], reference dimensions and ink, and a 3.5 MB size limit. It then propagates the imported phase on the CPU and verifies both distinct references before scene state changes. The references are verification metadata; the phase alone drives the forward field. This format provides reproducible puzzles, not encryption, authenticity, or tamper-proof storage.

## Prior art and contribution boundary

- [Nishitsuji et al., *Towards browser-based real-time CGH computation using WebGPU*, Digital Holography 2026](https://opg.optica.org/abstract.cfm?uri=DH-2026-Tu4A.15) already reports browser CGH across browsers and operating systems, including comparison with CUDA. Its accessible abstract does not establish its precise algorithm or whether it includes this particular editing interaction.
- [slmsuite's official multi-plane holography tutorial](https://slmsuite.readthedocs.io/en/latest/_examples/multiplane_holography.html) explicitly combines weighted target planes into one SLM phase pattern using weighted-GS methods. This is direct algorithmic precedent.
- [Kavaklı et al., *Realistic Defocus Blur for Multiplane Computer-Generated Holography*](https://arxiv.org/abs/2205.07030), with [author code](https://github.com/complight/realistic_defocus), addresses focused and defocused content in multi-plane holography. Recognizable slices alone do not establish physically consistent 3D imagery or natural defocus.
- [Matsushima and Shimobaba, *Band-Limited Angular Spectrum Method*](https://kansai-u.repo.nii.ac.jp/record/11127/files/KU-1100-20091015-50.pdf) documents transfer-function sampling errors and remedies. [waveprop](https://github.com/ebezzam/waveprop) is an additional author-provided propagation reference.
- [OpticsSandbox](https://optics-sandbox-webapp.vercel.app/) is a browser optics tool. Its [creator's June 2026 description](https://www.reddit.com/r/Optics/comments/1u2zf59/im_an_optics_phd_student_and_i_built_a_free/) describes angular-spectrum simulation, imported phase masks, and a GS hologram calculator. These features were researched, not independently exercised in this review.

The present contribution is a compact Three.js TSL propagator connected to editable two-image seals and an actual-intensity capture game, with independent CPU checks. Browser execution, multiple focal images, FFT propagation, and GS fitting are all established capabilities. This demonstrates a useful creative integration; it does not establish an algorithmic or industry-level breakthrough.
