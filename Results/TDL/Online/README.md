## Online A-MMSE: Channel Adaptation Results

This folder contains the online adaptation result of **A-MMSE** under a non-stationary channel environment, where the channel model switches mid-stream from TDL-D to TDL-E.

---

### Experiment Setup

The experiment evaluates the ability of Online A-MMSE to adapt its filter in real time as channel statistics change, without any prior knowledge of the new channel environment.

- **Total channel realizations:** 20,000
- **First 10,000:** TDL-D (LOS, low delay spread, ~30 ns)
- **Next 10,000:** TDL-E (strong LOS, very low delay spread, ~5 ns)
- **SNR:** Fixed during the experiment

---

### Key Observations

- **Convergence in TDL-D:** NMSE steadily decreases over the first 10,000 realizations as the online filter adapts to TDL-D statistics, eventually reaching ~10⁻³.
- **Channel switch spike:** At realization #10,000, the channel abruptly switches to TDL-E, causing an immediate NMSE spike back toward ~10⁻².
- **Re-adaptation in TDL-E:** The Online A-MMSE rapidly re-converges under TDL-E, demonstrating strong online tracking capability even after a sudden channel distribution shift.

---

### Figure

| Description | File |
|-------------|------|
| NMSE vs. Channel Realization (TDL-D → TDL-E) | [Online_result.pdf](Online_result.pdf) |
