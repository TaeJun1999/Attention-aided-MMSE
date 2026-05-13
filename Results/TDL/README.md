## TDL Channel Estimation Results

This folder contains the NMSE vs. SNR simulation results of **A-MMSE** evaluated on 3GPP TDL (Tapped Delay Line) channel models (TDL-A through TDL-E).

---

### About TDL Channel Models

The 3GPP TDL channel models represent standardized multipath fading environments used in 5G NR link-level simulations:

| Model | Delay Spread | Characteristics |
|-------|-------------|-----------------|
| TDL-A | High (~316 ns) | Non-line-of-sight (NLOS), rich scattering |
| TDL-B | High (~316 ns) | NLOS, different power-delay profile |
| TDL-C | Very high (~1000 ns) | NLOS, dense multipath |
| TDL-D | Low (~30 ns) | Line-of-sight (LOS), strong direct component |
| TDL-E | Very low (~5 ns) | Strong LOS, near-static channel |

---

### NMSE vs. SNR Results

A-MMSE is compared against baseline channel estimators including LS, ideal MMSE, and DNN-based methods. Results demonstrate that A-MMSE consistently achieves near-optimal NMSE performance across all TDL models while maintaining low inference complexity through its single linear estimation step.

| Channel | Figure |
|---------|--------|
| TDL-A | [TDLA_SNR.pdf](TDLA_SNR.pdf) |
| TDL-B | [TDLB_SNR.pdf](TDLB_SNR.pdf) |
| TDL-C | [TDLC_SNR.pdf](TDLC_SNR.pdf) |
| TDL-D | [TDLD_SNR.pdf](TDLD_SNR.pdf) |
| TDL-E | [TDLE_SNR.pdf](TDLE_SNR.pdf) |
