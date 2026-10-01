import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# Set these before running. Each array is (72, 14, frames): subcarriers x OFDM symbols x frames.
CLEAN_PATH = ""   # .mat file with the true channels
CLEAN_VAR = ""    # variable name in CLEAN_PATH
NOISY_PATH = ""   # .mat file with the noisy observations at the training SNR
NOISY_VAR = ""    # variable name in NOISY_PATH


def load_channel(num_val=4000, num_test=4000, random_state=42):
    """Noisy grids x (samples, 2, NM, 1) and true channels y (samples, NM, 1) for train/val/test."""
    if not (CLEAN_PATH and CLEAN_VAR and NOISY_PATH and NOISY_VAR):
        raise ValueError("Set CLEAN_PATH, CLEAN_VAR, NOISY_PATH and NOISY_VAR at the top of data.py")

    perfect = np.transpose(loadmat(CLEAN_PATH)[CLEAN_VAR], [2, 0, 1])  # (samples, 72, 14)
    noisy = np.transpose(loadmat(NOISY_PATH)[NOISY_VAR], [2, 0, 1])

    # Column-major vec as in MATLAB: (subcarrier j, symbol i) -> j + 72 * i
    perfect_real = np.real(perfect).reshape(perfect.shape[0], -1, 1, order='F')  # (samples, 72*14, 1)
    perfect_imag = np.imag(perfect).reshape(perfect.shape[0], -1, 1, order='F')
    noisy_real = np.real(noisy).reshape(noisy.shape[0], -1, 1, order='F')
    noisy_imag = np.imag(noisy).reshape(noisy.shape[0], -1, 1, order='F')

    noisy_combined = np.stack((noisy_real, noisy_imag), axis=1)  # (samples, 2, 72*14, 1)
    perfect_complex = perfect_real + 1j * perfect_imag

    # Section VI-A: the last num_test frames in time are the test set, and num_val frames drawn
    # at random from the earlier frames are the validation set.
    # With 44,000 frames: train 36,000 / val 4,000 / test = frames 40,000-44,000.
    x_test = noisy_combined[-num_test:]
    y_test = perfect_complex[-num_test:]
    x_train, x_val, y_train, y_val = train_test_split(noisy_combined[:-num_test], perfect_complex[:-num_test],
                                                      test_size=num_val,
                                                      random_state=random_state)
    print(f"{len(noisy_combined)} frames: train {len(x_train)}, val {len(x_val)}, test {len(x_test)}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def DMRs_MapA_config1_comb2(num_pilots):
    """
    Pilot indices into the column-major vec of the (72, 14) grid, (row j, column i) -> j + 72 * i.
    Comb-2 DM-RS on rows 1, 3, ..., 71 (0-based) of columns 2 and 11 for 72 pilots;
    36, 108 and 144 pilots use columns 2 / 2, 5, 11 / 2, 5, 8, 11.
    """
    if num_pilots == 72:
        # column 2: [145, 147, ..., 215], column 11: [793, 795, ..., 863]
        idx_unif = [145 + i for i in range(0, 72, 2)] + [793 + i for i in range(0, 72, 2)]

    elif num_pilots == 36:
        idx_unif = [145 + i for i in range(0, 72, 2)]

    elif num_pilots == 108:
        idx_unif = [145 + i for i in range(0, 72, 2)] + [361 + i for i in range(0, 72, 2)] + [793 + i for i in range(0, 72, 2)]

    elif num_pilots == 144:
        idx_unif = [145 + i for i in range(0, 72, 2)] + [361 + i for i in range(0, 72, 2)] + [577 + i for i in range(0, 72, 2)] + [793 + i for i in range(0, 72, 2)]

    return idx_unif


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_channel()
    print("x:", x_train.shape, x_val.shape, x_test.shape)
    print("y:", y_train.shape, y_val.shape, y_test.shape)
    for n in (36, 72, 108, 144):
        print(f"{n} pilots:", DMRs_MapA_config1_comb2(n))
