# COST2100 results

NMSE versus SNR in the Semi-Urban (SU) and High-Speed Rail (HSR) scenarios of the COST2100 channel model.

| Semi-Urban (SU) | High-Speed Rail (HSR) |
|:---:|:---:|
| ![NMSE vs. SNR, SU](NMSE_SU_sens.png) | ![NMSE vs. SNR, HSR](NMSE_HSR_sens.png) |
| [NMSE_SU_sens.pdf](NMSE_SU_sens.pdf) | [NMSE_HSR_sens.pdf](NMSE_HSR_sens.pdf) |

## Setup

| Scenario | Carrier | Delay spread | SCS | Velocity | Max Doppler | Propagation |
|---|---|---|---|---|---|---|
| HSR | 5 GHz | 100 ns | 60 kHz | 350 km/h | ~1620.4 Hz | Strong LoS (K = 13 dB) |
| SU | 3.5 GHz | 1000 ns | 30 kHz | 40 km/h | ~129.6 Hz | Rich scattering (K = 3 dB) |

- 44,000 consecutive OFDM frames per scenario on a `72 x 14` grid with `L = 72` pilots. The first 40,000 frames are used for training (36,000) and validation (4,000); the last 4,000 frames form the test set.
- A-MMSE and the mismatched LMMSE are single fixed filters obtained from the same 36,000 training frames. The mismatched LMMSE uses the sample second moments of the true training channels.
- RA-A-MMSE (50%, 25%, 10%) uses rank `r = 36, 18, 7`.
- The Oracle LMMSE is a reference, not a baseline: it is recomputed for every test frame from the true channel of that frame, so it is a lower bound that a receiver cannot realize.
- NMSE is the ratio of the summed squared error to the summed channel energy over the 4,000 test frames, with one noise realization per frame.

## NMSE at 35 dB

| Scenario | Oracle LMMSE (reference) | A-MMSE | RA-A-MMSE (50%) | Mismatched LMMSE |
|---|---|---|---|---|
| SU | 1.75e-5 | 2.07e-5 | 4.84e-5 | 7.91e-5 |
| HSR | 1.74e-5 | 2.48e-4 | 3.01e-4 | 4.09e-4 |

In SU, the A-MMSE nearly coincides with the Oracle LMMSE. In HSR the channel changes rapidly from frame to frame, so every fixed linear filter keeps a gap to the per-frame oracle at high SNR.

Results under SNR mismatch are in [`SNR_mismatch/`](SNR_mismatch/).
