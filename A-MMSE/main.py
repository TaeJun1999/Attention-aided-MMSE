import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras 2 (TF >= 2.16 needs `pip install tf_keras`)
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data import load_channel, DMRs_MapA_config1_comb2
from AMMSE import TransformerModel, complex_mse

# Pick GPUs with CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=0 python main.py
all_gpus = tf.config.list_physical_devices('GPU')
for gpu in all_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"Visible GPUs: {all_gpus or 'none, running on CPU'}")
strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE = 50
EPOCHS = 150

NUM_PILOT = 72
D_MODEL1 = 72      # frequency encoder embedding = N subcarriers
D_MODEL2 = 1008    # temporal encoder projection = NM resource elements
D_MODEL_OUT = 1008
NUM_HEADS_1 = 36   # pilots per DM-RS symbol
NUM_HEADS_2 = 14   # OFDM symbols
D_FFN1 = NUM_PILOT * 4
D_FFN2 = NUM_PILOT * 4
DROPOUT_RATE = 0.1

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_channel()

# Keep only the pilot positions of the noisy grid: (samples, 2, L, 1)
pilot_indices = DMRs_MapA_config1_comb2(NUM_PILOT)
x_train = x_train[:, :, pilot_indices, :]
x_val = x_val[:, :, pilot_indices, :]
x_test = x_test[:, :, pilot_indices, :]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .shuffle(buffer_size=len(x_train))\
    .batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


def MMSE_filter_2D_tensorflow(inputs):
    """Channel estimate W y_p: filters (B, NM, L) and noisy pilots (B, 2, L, 1) -> (B, NM, 1)."""
    W_ammse, pilot_observations = inputs
    pilot_obs_cmplx = tf.complex(pilot_observations[:, 0, :, :],
                                 pilot_observations[:, 1, :, :])
    return W_ammse @ pilot_obs_cmplx


with strategy.scope():
    transformer_model = TransformerModel(
        num_pilot=NUM_PILOT,
        d_model1=D_MODEL1,
        d_model2=D_MODEL2,
        num_heads_1=NUM_HEADS_1,
        num_heads_2=NUM_HEADS_2,
        d_ffn1=D_FFN1,
        d_ffn2=D_FFN2,
        d_model_out=D_MODEL_OUT,
        num_layers_1=1,
        num_layers_2=1,
        dropout_rate=DROPOUT_RATE,
    )

    input_shape = (None, x_train.shape[1], NUM_PILOT, x_train.shape[-1])
    transformer_model.build(input_shape)
    transformer_model.summary()

    x_input = layers.Input(shape=(x_train.shape[1], NUM_PILOT, x_train.shape[-1]))  # (B, 2, num_pilot, 1)

    # Filter mode (see AMMSE.py). Default: one filter generated from the global token
    USE_TRAINABLE_SHARED_FILTER = False
    USE_TRANSFORMER_GLOBAL_FILTER = True

    if USE_TRANSFORMER_GLOBAL_FILTER:
        transformer_model.enable_transformer_global_filter(True, init_from='global_output')
        W_ammse = transformer_model(x_input, training=True)
    elif USE_TRAINABLE_SHARED_FILTER:
        # Start from a saved global output if there is one
        try:
            transformer_model.load_global_output("./AMMSE_revision_mse")
        except Exception:
            pass
        transformer_model.enable_trainable_shared_filter(enabled=True, init_from_global=True)
        transformer_model.set_sharing_mode(2)
        W_ammse = transformer_model(x_input, training=True)
    else:
        W_ammse = transformer_model(x_input)

    MMSE_Estimation_2D = layers.Lambda(
        MMSE_filter_2D_tensorflow, name="2D_MMSE_Estimation"
    )([W_ammse, x_input])

    final_transformer_model = Model(inputs=x_input, outputs=MMSE_Estimation_2D)
    final_transformer_model.summary()

    # MSE loss (28) with Adam (Section IV-B3)
    final_transformer_model.compile(
        optimizer=Adam(learning_rate=0.00015),
        loss=complex_mse
    )

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, min_lr=1e-14)
]

transformer_history = final_transformer_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# One filter for all test frames, (1, 2, L, NM)
if USE_TRANSFORMER_GLOBAL_FILTER:
    global_filter = transformer_model.compute_transformer_global_output()
elif USE_TRAINABLE_SHARED_FILTER:
    try:
        transformer_model.save_shared_filter("./AMMSE_revision_mse_shared_filter")
    except Exception as e:
        print(f"Failed to save shared filter: {e}")
    global_filter = transformer_model.shared_filter
else:
    transformer_model.calculate_global_output(val_dataset)
    transformer_model.save_global_output("./AMMSE_revision_mse")
    global_filter = transformer_model.global_output

# W_AMMSE = (g[0,0] + 1j*g[0,1]).T has shape (NM, L); train_ra.py reads this file
global_filter_path = "./AMMSE_revision_mse/global_filter_E_72_15dB_mse.mat"
os.makedirs(os.path.dirname(global_filter_path), exist_ok=True)
global_filter = np.asarray(global_filter)
sio.savemat(global_filter_path, {'global_filter': global_filter})
print(f"Saved global filter {global_filter.shape} to {global_filter_path}")

if USE_TRANSFORMER_GLOBAL_FILTER:
    transformer_model.enable_transformer_global_filter(True)
elif USE_TRAINABLE_SHARED_FILTER:
    try:
        transformer_model.load_shared_filter("./AMMSE_revision_mse_shared_filter")
    except Exception as e:
        print(f"Failed to load shared filter: {e}")
    transformer_model.set_sharing_mode(2)
    transformer_model.enable_trainable_shared_filter(True, init_from_global=False)
else:
    transformer_model.load_global_output("./AMMSE_revision_mse")
    transformer_model.set_sharing_mode(2)

# Sanity check only. For reported numbers, apply the saved filter to the test pilots
# (e.g. in MATLAB) and compute the NMSE there.
transformer_mse = final_transformer_model.evaluate(test_dataset)
y_power = np.mean(np.abs(y_test) ** 2)
transformer_nmse = transformer_mse / y_power
print(f"Test MSE: {transformer_mse:.8f}, NMSE: {transformer_nmse:.8f} ({10 * np.log10(transformer_nmse):.2f} dB)")

# All test predictions for MATLAB, (NM, 1, N_test)
all_pred = np.concatenate([final_transformer_model(x_batch).numpy() for x_batch, _ in test_dataset])
all_pred_t = np.transpose(all_pred, (1, 2, 0))
mat_path = "./AMMSE_revision_mse/AMMSE_pred_72_E_15dB_mse.mat"
sio.savemat(mat_path, {"transformer_outputs": all_pred_t})
print(f"Saved test predictions {all_pred_t.shape} to {mat_path}")
