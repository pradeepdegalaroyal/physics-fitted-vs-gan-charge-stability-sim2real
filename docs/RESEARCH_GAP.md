# Research gap: closing the simulation-to-measurement gap for charge stability diagrams

## Setting

Silicon spin-qubit devices are tuned with charge stability diagrams (CSDs): a charge sensor is read out while two gate voltages are swept, and the honeycomb of charge-transition lines tells the experimenter which electron numbers sit in which dot. Physics-based simulators such as the constant-interaction model (CIM) give unlimited CSDs with exact transition labels, but their images are clean. Measured images show thermal broadening, a non-linear sensor response, 1/f drift, telegraph charge jumps and pick-up. A network trained on the clean simulation therefore degrades on measured data. Generative models (for example the generative adversarial network (GAN)), synthetic data and domain adaptation are the obvious remedies, and the open question is *which kind of generative model is the right one when only a handful of unlabeled measurements exist and labels must stay physically exact*.

The brief supplied for this study contains no paper on spin-qubit data, so the related work below is the nearest relevant material (hardware noise, noise characterisation and mitigation, pretrained and transfer-learned scientific models, hardware-aware compilation). Every claim about the literature is limited to what those abstracts say. The brief tags all papers with the single role "other"; the *role* column below is my own grouping (landmark: at least 400 citations; recent: published 2027 per the metadata; review: survey papers; adjacent: other fields).

## Related work

| # | Paper (first author, year, digital object identifier (DOI)) | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| 1 | Cai et al., 2023, 10.1103/RevModPhys.95.045005 | review | Surveys quantum error mitigation and states that the method should be chosen by the dominant noise type | No generative data model; no image data |
| 2 | Kim et al., 2023, 10.1038/s41586-023-06096-3 | landmark | Characterises and controllably manipulates noise on a 127-qubit device so that expectation values can be trusted | Noise is learned for circuits, not for sensor images |
| 3 | Acharya et al., 2023, 10.1038/s41586-022-05434-1 | landmark | Builds an error budget from an accurate model of the experiment | Model is hand-built per device; no sim-to-real data translation |
| 4 | Zobrist et al., 2025, 10.1038/s41586-024-08449-y | landmark | Below-threshold surface-code memories; finds rare correlated error events | Correlated events are observed, not synthesised |
| 5 | Sivak et al., 2023, 10.1038/s41586-023-05782-6 | landmark | Real-time error correction tuned with model-free reinforcement learning | Learning acts on control, not on synthetic training data |
| 6 | Bluvstein et al., 2024, 10.1038/s41586-023-06927-3 | landmark | Logical processor on reconfigurable atom arrays | Platform differs; no data generation |
| 7 | Moses et al., 2023, 10.1103/PhysRevX.13.041052 | landmark | Benchmarks a trapped-ion processor with several application benchmarks | No simulation-to-measurement comparison |
| 8 | Deng et al., 2023, 10.1038/s42256-023-00716-3 | landmark | Crystal Hamiltonian Graph Neural Network (CHGNet): a neural network pretrained on large simulation datasets for atomistic modelling | Source and target are both simulations; no measured-noise gap |
| 9 | Musaelian et al., 2023, 10.1038/s41467-023-36329-y | adjacent | Allegro: local equivariant network that generalises out of distribution | Physical symmetry is built into the model, but no domain shift to measured data is studied |
| 10 | Lin et al., 2027, 10.1016/j.ces.2026.124987 | recent | Transfer learning from sparse data with explicit physicochemical constraints; ablations | Tabular molecular data; not images |
| 11 | Xu et al., 2027, 10.1016/j.saa.2026.128945 | recent | Deep transfer learning for spectra converted to pseudo-images, fine-tuned on few target samples | Needs labelled target samples; no generative model |
| 12 | Du et al., 2027, 10.1016/j.future.2026.108830 | recent | Compiler that weights hardware edges by calibration metrics and evaluates on noisy emulators calibrated from hardware | Circuit mapping only |
| 13 | Zhang et al., 2027, 10.1016/j.future.2026.108829 | recent | Links operator-level guarantees to finite-budget cost with a simulated depolarising-noise proxy | Shows that a *proxy* can track a budgeted cost; no image data |
| 14 | Fan et al., 2027, 10.1016/j.ins.2026.124264 | recent | Quantum implicit neural network for image representation, state-vector simulation | Image compression, not a domain gap |

## Where the papers meet

- The experimental papers (rows 2 to 7) show that progress depends on an *accurate, device-specific noise model* (rows 2 and 3) and on rare correlated events that are only seen in real data (row 4). The review (row 1) states that the right mitigation depends on the dominant noise type.
- The transfer-learning papers (rows 8, 10, 11) show that pretraining on simulation plus a small amount of target data works, and that explicit physical constraints help (row 10), but they use tabular or spectral data and need target *labels* or target *simulations*.
- Row 12 shows that parameters of a physical model can be tied directly to measured calibration data, and row 13 shows that a cheap proxy can stand in for an expensive budgeted measurement.

None of these studies compares, on image data, a **physics-structured generator whose few parameters are fitted to unlabeled measurements** with a **free-form adversarial image translator (GAN-based)**, nor asks what happens to label fidelity and downstream accuracy when the measurements contain effects that the physics model does not contain.

## The gap

When only a few unlabeled target images exist, does fitting the 11 parameters of a physics-structured measurement model beat an unpaired adversarial image translator for the downstream task, and at what strength of *unmodelled* artefacts does the free-form translator overtake it?

**Novelty (one sentence).** This is a controlled sweep, on simulated charge stability diagrams, of the number of unlabeled target images against the strength of unmodelled artefacts, comparing a parameter-fitted physics simulator (labels exact by construction) with a CycleGAN-style translator (labels inherited and possibly drifting) and with a no-target-data augmentation baseline.

## Falsifiable hypotheses and metrics

- **H1.** With 32 unlabeled target images and no unmodelled artefacts, the physics-fitted simulator closes at least 0.8 of the Dice gap between a source-only model and an oracle trained on labelled target-like data, and exceeds the adversarial translator by at least 0.1 in gap closed.
- **H2.** The advantage of the fitted simulator shrinks as the unmodelled artefact strength grows (fitted-minus-GAN gap closed is lower at artefact strength 1.0 than at 0.0).
- **Primary metric:** Dice coefficient of transition-line segmentation on held-out target-proxy images; derived metric: *gap closed* = (Dice of method minus Dice source-only) / (Dice oracle minus Dice source-only), paired by seed.
- **Secondary metric:** random-feature Fréchet distance (RF-FD) between generated and target images (independent of the quantity the fit minimises).

## Scale and honesty

Everything is simulation at 64 x 64 pixels (px). The "target" is a *proxy* generated by a richer simulator, not a measurement; the proxy shares the model family of the fitted simulator, which favours the fit, and the artefact-strength sweep is the knob that removes that favour. The result is a map of when each approach works under stated assumptions, not evidence about any particular device.
