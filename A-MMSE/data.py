import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf

###############################################################################
# 1. Data loading from .mat file
###############################################################################
def load_channel(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    # Load .mat files
    perfect = loadmat("/home/HTJ/HTJ/Pilot_desing NN/AMMSE_DATA/TDL-E/5G_NR_TDL_E.mat")["ofdmGrid"]
    perfect = np.transpose(perfect, [2, 0, 1])  # (samples, 72, 14)
    print("Perfect shape:", perfect.shape)
    samples = perfect.shape[0]

    noisy = loadmat("/home/HTJ/HTJ/Pilot_desing NN/AMMSE_DATA/TDL-E/TDLE_30dB.mat")["noisy_data_30dB"]
    noisy = np.transpose(noisy, [2, 0, 1])  # (samples, 72, 14)
    print("Noisy shape:", noisy.shape)

    # Separate real and imaginary parts, then reshape (column-wise with order='F')
    perfect_real = np.real(perfect).reshape(perfect.shape[0], -1, 1, order='F')  # (samples, 72*14, 1)
    perfect_imag = np.imag(perfect).reshape(perfect.shape[0], -1, 1, order='F')  # (samples, 72*14, 1)

    noisy_real = np.real(noisy).reshape(noisy.shape[0], -1, 1, order='F')  # (samples, 72*14, 1)
    noisy_imag = np.imag(noisy).reshape(noisy.shape[0], -1, 1, order='F')  # (samples, 72*14, 1)

    # Stack real and imaginary parts along a new axis
    perfect_combined = np.stack((perfect_real, perfect_imag), axis=1)  # (samples, 2, 72*14, 1)
    noisy_combined = np.stack((noisy_real, noisy_imag), axis=1)  # (samples, 2, 72*14, 1)

    perfect_complex = perfect_real + 1j * perfect_imag  # create NumPy complex array
    print("Perfect complex shape:", perfect_complex.shape)
    
    # Optionally skip the first N samples
    skip_samples = 0
    if samples > skip_samples:
        perfect_combined = perfect_combined[skip_samples:]
        noisy_combined = noisy_combined[skip_samples:]
        perfect_complex = perfect_complex[skip_samples:]
        print(f"Skipped first {skip_samples} samples")
        print(f"Remaining samples: {perfect_combined.shape[0]}")
    else:
        print(f"Warning: Total samples ({samples}) <= skip_samples ({skip_samples})")

    print("Perfect combined shape:", perfect_combined.shape)  # (samples, 2, 72*14, 1)
    print("Noisy combined shape:", noisy_combined.shape)  # (samples, 2, 72*14, 1)

    perfect_1 = perfect_combined[0:1,:,:,:]
    print("Batch 1 :", perfect_1)

    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    assert np.isclose(total_ratio, 1.0), f"train_ratio + val_ratio + test_ratio should be 1, but got {total_ratio}"

    # Compute indices - last 10% for test set
    total_samples = len(noisy_combined)
    test_start_idx = int(total_samples * (1 - test_ratio))
    
    # Take the last 10% as test data
    x_test = noisy_combined[test_start_idx:]
    y_test = perfect_complex[test_start_idx:]
    
    # Split the first 90% into train/val randomly
    x_front = noisy_combined[:test_start_idx]
    y_front = perfect_complex[:test_start_idx]
    
    # Random split for train/val (val is 1/9 of the first 90%, equals 10% overall)
    val_ratio_adjusted = val_ratio / (1 - test_ratio)  # relative ratio within the first 90%
    x_train, x_val, y_train, y_val = train_test_split(x_front, y_front, 
                                                      test_size=val_ratio_adjusted, 
                                                      random_state=random_state)
    
    # Log dataset sizes
    print(f"Training set: {len(x_train)} samples ({len(x_train)/len(noisy_combined):.1%})")
    print(f"Validation set: {len(x_val)} samples ({len(x_val)/len(noisy_combined):.1%})")
    print(f"Test set: {len(x_test)} samples ({len(x_test)/len(noisy_combined):.1%})")
  
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

###############################################################################
# 2. Uniform pilot index
###############################################################################
def DMRs_MapA_config1_comb2(num_pilots):
    """
    Uniform pilot indices for order='F' reshape
    Original matrix: (72, 14) -> reshape to (1008,) with order='F'
    Column-wise: [row0_col0, row1_col0, ..., row71_col0, row0_col1, row1_col1, ...]
    Pick pilots from columns 2, 5, 8, and 11.
    After reshape, (row j, col i) maps to index (j + 72 * i)
    """
    if (num_pilots == 72):
        # choose columns 2 and 11 (36 each)
        # column 2: [2, 74, 146, 218, 290, 362, 434, 506, 578, 650, 722, 794, 866, 938]
        # column 11: [11, 83, 155, 227, 299, 371, 443, 515, 587, 659, 731, 803, 875, 947]
        idx_unif = [145 + (i) for i in range(0, 72, 2)] + [793 + (i) for i in range(0, 72, 2)]

    elif (num_pilots == 36):
        # choose only column 2 (even rows)
        idx_unif = [145 + (i) for i in range(0, 72, 2)]

    elif (num_pilots == 108):
        # choose columns 2, 5, and 11 (36 each)
        idx_unif = [145 + (i) for i in range(0, 72, 2)] + [361 + (i) for i in range(0, 72, 2)] + [793 + (i) for i in range(0, 72, 2)]

    elif (num_pilots == 144):
        # choose columns 2, 5, 8, and 11 (36 each)
        idx_unif = [145 + (i) for i in range(0, 72, 2)] + [361 + (i) for i in range(0, 72, 2)] + [577 + (i) for i in range(0, 72, 2)] + [793 + (i) for i in range(0, 72, 2)]
    
    return idx_unif

###############################################################################
# 3. Quick sanity check
###############################################################################
def check_data():
    """Print dataset shapes and pilot indices for a quick sanity check."""
    try:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_channel()
        
        print("\n=== DATA SHAPES ===")
        print(f"train: {x_train.shape} -> {y_train.shape}")
        print(f"val:   {x_val.shape} -> {y_train.shape}")
        print(f"test:  {x_test.shape} -> {y_test.shape}")
        
        print("\n=== PILOT INDICES ===")
        for num_pilots in [36, 72, 108, 144]:
            indices = DMRs_MapA_config1_comb2(num_pilots)
            print(f"{num_pilots} pilots: {len(indices)}")
            print(f"  indices: {indices}")
            print(f"  index range: {min(indices)} ~ {max(indices)}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data()