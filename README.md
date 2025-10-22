## Attention-Aided MMSE for OFDM Channel Estimation: Learning Linear Filters with Attention

This repository implements a Transformer-based filter generator that is combined with an MMSE-style post-processing step for OFDM channel estimation. The model learns to produce a 2D complex filter over the OFDM grid given pilot observations and then applies an MMSE-like operation to estimate the full channel.

### Paper URL
https://arxiv.org/abs/2506.00452

### Abstract
In orthogonal frequency division multiplexing (OFDM), accurate channel estimation is crucial. Classical signal processing based approaches, such as minimum mean-squared error (MMSE) estimation, often require second-order statistics that are difficult to obtain in practice. Recent deep neural networks based methods have been introduced to address this; yet they often suffer from high inference complexity. This paper proposes an Attention-aided MMSE (A-MMSE), a novel model-based DNN framework that learns the optimal MMSE filter via the Attention Transformer. Once trained, the A-MMSE estimates the channel through a single linear operation for channel estimation, eliminating nonlinear activations during inference and thus reducing computational complexity. To enhance the learning efficiency of the A-MMSE, we develop a two-stage Attention encoder, designed to effectively capture the channel correlation structure. Additionally, a rank-adaptive extension of the proposed A-MMSE allows flexible trade-offs between complexity and channel estimation accuracy. Extensive simulations with 3GPP TDL channel models demonstrate that the proposed A-MMSE consistently outperforms other baseline methods in terms of normalized MSE across a wide range of signal-to-noise ratio (SNR) conditions. In particular, the A-MMSE and its rank-adaptive extension establish a new frontier in the performance-complexity trade-off, providing a powerful yet highly efficient solution for practical channel estimation.

### Key Features
- Transformer encoder stack to produce filters from pilot observations
- Multiple sharing strategies for the produced filter:
  - No sharing (per-sample filter)
  - Per-batch sharing (use the batch-average filter)
  - Global sharing (use a single global filter for all samples)
- Trainable shared filter and Transformer-Global-Filter via a learnable token
- Optional hybrid blending between shared and per-sample filters
- MATLAB-friendly outputs (.mat) for downstream evaluation

### Repository Structure
- `main.py`: Training, validation, simple in-script test (sanity check), and saving outputs (global filter and predictions)
- `AMMSE.py`: Model definition
  - Input preprocessing `(B, 2, Np, 1) → (B, 2*Np, 1)`
  - Two-stage Transformer encoders
  - Residual Fully Connected network (RA module); RA modules can be stacked. When using RA modules, consider adding an activation to the final `fc4` as well
  - Sharing/Global/Hybrid modes utilities + global output computation and persistence
- `data.py`: Data loading from `.mat` files, tensor shaping, and pilot index generation
- `AMMSE_filter/`: Placeholder for custom MMSE implementations (currently empty)

### Data
`data.py` expects two `.mat` files (edit paths as needed inside `data.py`):
- Perfect channel grid: `ofdmGrid` from `5G_NR_TDL_E.mat`
- Noisy data: `noisy_data_30dB` from `TDLE_30dB.mat`

Shapes used in code:
- OFDM grid: `(samples, 72, 14)` → reshaped (column-major) to `(samples, 2, 72*14, 1)` for inputs
- Targets are complex arrays matching the full grid

### Environment
- Python 3.x
- TensorFlow 2.x
- NumPy, SciPy, scikit-learn

Example installation:
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow numpy scipy scikit-learn
```

### Quick Start
1) Ensure dataset paths in `data.py` are correct.
2) Run training:
```bash
python main.py
```

During training, the script will:
- Train the Transformer + MMSE pipeline
- Save a global filter in MATLAB format
- Optionally compute in-script metrics (sanity check) and save all test predictions (.mat)

### Model Options (in `main.py`)
- `USE_TRANSFORMER_GLOBAL_FILTER` (default True):
  - Use a learnable global token to produce a single filter for all samples
- `USE_TRAINABLE_SHARED_FILTER`:
  - Enable a trainable shared filter parameter (optionally initialized from a precomputed global output)
- `USE_HYBRID_BLEND`:
  - Blend a shared filter with the per-sample transformer output via a trainable `alpha`

Additional sharing control resides in `AMMSE.py`:
- `set_sharing_mode(mode)` where `mode ∈ {0, 1, 2}`:
  - 0: no sharing (per-sample)
  - 1: per-batch sharing (use batch mean)
  - 2: global sharing (use `global_output` or `shared_filter`)

Priority inside the model call path:
1) Transformer-Global-Filter (learnable token)
2) Hybrid blending (shared + per-sample)
3) `sharing_mode` (0/1/2)

### Outputs
The script saves:
- Global filter (.mat): see `global_filter_path` in `main.py`
- All test predictions (.mat): see `mat_path` in `main.py`

### Important: Evaluation Note
The “Evaluate the model” section within `main.py` (printing test loss/NMSE) is a sanity check to confirm training behavior. It is NOT the final evaluation for reporting.

For actual testing/evaluation:
1) Train to produce the global filter `.mat`
2) In MATLAB (or your evaluation environment), load the saved filter and apply it to pilot observations to estimate the full channel
3) Compute metrics (MSE/NMSE/BER, etc.) on the reconstructed channels

### Citation (BibTeX)
```bibtex
@misc{ha2025attentionaidedmmseofdmchannel,
      title={Attention-Aided MMSE for OFDM Channel Estimation: Learning Linear Filters with Attention}, 
      author={TaeJun Ha and Chaehyun Jung and Hyeonuk Kim and Jeongwoo Park and Jeonghun Park},
      year={2025},
      eprint={2506.00452},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2506.00452}, 
}
```


