"""
Train the rank-adaptive (RA) module of RA-A-MMSE (Section V) with the A-MMSE filter held fixed.

W_RA = W_AMMSE U_1 V_1^T ... U_S V_S^T, with real U_s, V_s (L x r_s) and L >= r_1 >= ... >= r_S = r.
Loss and train/val split are the same as in main.py. W_RA is then factorized as A B^T and (A, B) is saved;
the UE estimates the channel as A (B^T y_p).

    CUDA_VISIBLE_DEVICES=0 python train_ra.py --w_ammse <global_filter.mat> --ranks <r_1 ... r_S> \
        --lr <lr> --epochs <epochs> --batch_size <batch> --patience <patience> --out <ra.mat>
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras 2 (TF >= 2.16 needs `pip install tf_keras`)
import argparse
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from data import load_channel, DMRs_MapA_config1_comb2
from AMMSE import complex_mse

parser = argparse.ArgumentParser(description="Train the RA module of RA-A-MMSE with the A-MMSE held fixed.")
parser.add_argument("--w_ammse", required=True, help="global filter .mat saved by main.py")
parser.add_argument("--ranks", type=int, nargs="+", required=True, help="r_1 ... r_S, non-increasing")
parser.add_argument("--lr", type=float, required=True, help="Adam learning rate")
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--patience", type=int, required=True, help="early stopping patience on val_loss")
parser.add_argument("--out", required=True, help="output .mat with A, B and ranks")
args = parser.parse_args()

tf.keras.utils.set_random_seed(42)

# W_AMMSE (NM x L) from the saved (1, 2, L, NM) array, same conversion as TransformerModel.call
g = sio.loadmat(args.w_ammse)["global_filter"]
W = (g[0, 0] + 1j * g[0, 1]).T.astype(np.complex64)
NM, L = W.shape
ranks = args.ranks
if not all(1 <= r <= L for r in ranks) or any(b > a for a, b in zip(ranks, ranks[1:])):
    parser.error(f"ranks must satisfy {L} >= r_1 >= ... >= r_S >= 1, got {ranks}")


def to_complex(x, h):
    """(B, 2, NM, 1) noisy grid, (B, NM, 1) channel -> pilots y_p (B, L), channel h (B, NM)."""
    x = x[:, :, DMRs_MapA_config1_comb2(L), 0]
    return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64), h[..., 0].astype(np.complex64)


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_channel()
yp_train, h_train = to_complex(x_train, y_train)
yp_val, h_val = to_complex(x_val, y_val)
yp_test, h_test = to_complex(x_test, y_test)


class RAModule(tf.keras.Model):
    """h_hat = W_AMMSE U_1 V_1^T ... U_S V_S^T y_p; W_AMMSE is a constant, only U_s, V_s are trained."""

    def __init__(self, W, ranks):
        super().__init__()
        self.W = tf.constant(W)
        self.pairs = [(self.add_weight(shape=(W.shape[1], r), initializer="random_normal", name=f"U_{s}"),
                       self.add_weight(shape=(W.shape[1], r), initializer="random_normal", name=f"V_{s}"))
                      for s, r in enumerate(ranks, 1)]

    def call(self, y):                          # y: (B, L) complex
        t = tf.transpose(y)                     # (L, B)
        for U, V in reversed(self.pairs):       # U_S V_S^T acts first
            U, V = tf.cast(U, t.dtype), tf.cast(V, t.dtype)
            t = U @ tf.matmul(V, t, transpose_a=True)
        return tf.transpose(self.W @ t)         # (B, NM)


model = RAModule(W, ranks)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss=complex_mse)
model.fit(yp_train, h_train, validation_data=(yp_val, h_val),
          epochs=args.epochs, batch_size=args.batch_size,
          callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)])

# Form W_RA once and factorize it: W_RA = P S Q^H (rank <= r)  ->  A = P_r S_r, B = (Q^H)_r^T
W_ra = W.astype(np.complex128)
for U, V in model.pairs:
    W_ra = W_ra @ (U.numpy() @ V.numpy().T)
P, s, Qh = np.linalg.svd(W_ra, full_matrices=False)
r = ranks[-1]
A = (P[:, :r] * s[:r]).astype(np.complex64)   # (NM, r)
B = Qh[:r].T.astype(np.complex64)             # (L, r)
err = np.linalg.norm(A @ B.T - W_ra) / np.linalg.norm(W_ra)
assert err < 1e-4, f"rank-{r} factorization error {err:.2e}"

sio.savemat(args.out, {"A": A, "B": B, "ranks": np.array(ranks)})
print(f"Saved A {A.shape}, B {B.shape} to {args.out} (factorization error {err:.2e})")

# Sanity check on the test frames with the deployed two-step product, NMSE as in Eq. (3)
nmse = lambda h_hat: np.sum(np.abs(h_hat - h_test) ** 2) / np.sum(np.abs(h_test) ** 2)
print(f"Test NMSE  A-MMSE (full rank): {nmse(yp_test @ W.T):.4e}   "
      f"RA-A-MMSE (r = {r}): {nmse((yp_test @ B) @ A.T):.4e}")
