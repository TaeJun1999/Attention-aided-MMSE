import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import Huber, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data import load_channel, DMRs_MapA_config1_comb2
import numpy as np
import scipy.io as sio

# ===========================
# 0. GPU selection (using tf.config)
# ===========================
gpu_ids = [2]  # e.g., use only GPUs 0, 2, and 4
all_gpus = tf.config.list_physical_devices('GPU')
if not all_gpus:
    print("[INFO] No available GPU. Running on CPU.")
else:
    selected_gpus = [all_gpus[i] for i in gpu_ids]
    tf.config.set_visible_devices(selected_gpus, 'GPU')
    # Enable memory growth
    for gpu in selected_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] Selected GPUs: {selected_gpus}")

# ===========================
# 1. Multi-GPU strategy
# ===========================
# After hiding GPUs via set_visible_devices, use MirroredStrategy
strategy = tf.distribute.MirroredStrategy()  # if devices are not specified, only visible devices are used

# ===========================
# 2. Hyperparameters
# ===========================
BATCH_SIZE = 50
EPOCHS = 150
SNR_DB = 15 # SNR in dB

NUM_PILOT = 72
D_MODEL1 = 72
D_MODEL2 = 1008
D_MODEL_OUT = 1008
NUM_HEADS_1 = 36
NUM_HEADS_2 = 14
D_FFN1 = NUM_PILOT * 4
D_FFN2 = NUM_PILOT * 4
DROPOUT_RATE = 0.1

# ===========================
# 3. Load data
# ===========================
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_channel()

# ===========================
# 4. Select pilot indices
# ===========================
pilot_indices = DMRs_MapA_config1_comb2(NUM_PILOT)
x_train = x_train[:, :, pilot_indices, :]
x_val = x_val[:, :, pilot_indices, :]
x_test = x_test[:, :, pilot_indices, :]

# ===========================
# 5. Convert to tf.data.Dataset
# ===========================
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .batch(BATCH_SIZE)\
    .shuffle(buffer_size=len(x_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
print("[INFO] y_train.shape:", y_train.shape)

# ===========================
# 6. Transformer model definition
#    -> created inside strategy.scope() below
# ===========================
from AMMSE import TransformerModel  # custom module

# ===========================
# 7. Define MMSE post-processing function
# ===========================
def MMSE_filter_2D_tensorflow(inputs):
    """
    TensorFlow-based MMSE 2D filter function (example)
    """
    full_cov_2D, pilot_observations, snr_dB, pilot_indices = inputs

    sigma2 = tf.pow(10.0, -snr_dB / 10.0)
    pilot_obs_cmplx = tf.complex(pilot_observations[:, 0, :, :],
                                 pilot_observations[:, 1, :, :])

  
    h_est = full_cov_2D @ pilot_obs_cmplx

    return h_est

# ===========================
# 8. Build & compile the model with Multi-GPU
# ===========================
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

    # Prepare input shape and build the Transformer model
    input_shape = (None, x_train.shape[1], NUM_PILOT, x_train.shape[-1])
    transformer_model.build(input_shape)
    transformer_model.summary()

    # 8.2 Final model: apply MMSE filter to Transformer output
    x_input = layers.Input(shape=(x_train.shape[1], NUM_PILOT, x_train.shape[-1]))  # (B, 2, num_pilot, 1)

    # ----- Trainable Shared Filter mode switches -----
    USE_TRAINABLE_SHARED_FILTER = False
    USE_HYBRID_BLEND = False  # hybrid disabled (use transformer-global-token mode)
    USE_TRANSFORMER_GLOBAL_FILTER = True  # Transformer generates a single shared filter via global token
    SHARED_FILTER_PATH = "./AMMSE_revision_mse_shared_filter"

    if USE_TRANSFORMER_GLOBAL_FILTER:
        # Single shared filter via Transformer global token
        transformer_model.enable_transformer_global_filter(True, init_from='global_output')
        full_cov_2D = transformer_model(x_input, training=True)  # learn via Transformer path
    elif USE_TRAINABLE_SHARED_FILTER:
        # Initialize shared filter from global output if available
        try:
            transformer_model.load_global_output("./AMMSE_revision_mse")
        except Exception:
            pass
        transformer_model.enable_trainable_shared_filter(enabled=True, init_from_global=True)
        transformer_model.set_sharing_mode(2)
        full_cov_2D = transformer_model(x_input, training=True)
    else:
        full_cov_2D = transformer_model(x_input)

    # SNR tensor
    SNR_TENSOR = tf.constant(SNR_DB, dtype=tf.float32)

    MMSE_Estimation_2D = layers.Lambda(
        MMSE_filter_2D_tensorflow, name="2D_MMSE_Estimation"
    )([full_cov_2D, x_input, SNR_TENSOR, pilot_indices])

    final_transformer_model = Model(inputs=x_input, outputs=MMSE_Estimation_2D)
    final_transformer_model.summary()

    # 8.3 Compile model
    huber_loss = Huber(delta=0.1)
    final_transformer_model.compile(
        optimizer=Nadam(learning_rate=0.00015),
        loss=huber_loss,
        metrics=[MeanSquaredError()]
    )

# ===========================
# 9. Train the model
# ===========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=4, min_lr=1e-14)
]

print("\n[INFO] Starting Multi-GPU Transformer training...")
transformer_history = final_transformer_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

if 'USE_TRANSFORMER_GLOBAL_FILTER' in globals() and USE_TRANSFORMER_GLOBAL_FILTER:
    # Transformer global-token single filter: nothing extra to save (θ and token constitute the filter)
    pass
elif 'USE_TRAINABLE_SHARED_FILTER' in globals() and USE_TRAINABLE_SHARED_FILTER:
    # Save the trained shared filter
    try:
        transformer_model.save_shared_filter("./AMMSE_revision_mse_shared_filter")
    except Exception as e:
        print(f"[WARN] Failed to save shared filter: {e}")
else:
    transformer_model.calculate_global_output(val_dataset)
    transformer_model.save_global_output("./AMMSE_revision_mse")

# Save global filter
print("\n[INFO] Saving global filter...")
global_filter_path = "./AMMSE_revision_mse/global_filter_E_72_15dB_mse.mat"

try:
    # Extract global filter (use transformer's global output)
    global_filter = transformer_model.get_global_output()  # get global output
    
    # Save in MATLAB-friendly format
    if global_filter is not None:
        # Adjust dims if needed
        if len(global_filter.shape) == 3:  # if shape is (1008, 72, 1)
            global_filter_save = global_filter
        else:
            global_filter_save = global_filter
            
        sio.savemat(global_filter_path, {'global_filter': global_filter_save})
        print(f"[INFO] Global filter saved to: {global_filter_path}")
        print(f"[INFO] Global filter shape: {global_filter_save.shape}")
    else:
        print("[WARNING] Global filter is None, skipping save")
        
except Exception as e:
    print(f"[ERROR] Failed to save global filter: {e}")
    print("[INFO] Trying alternative method...")
    
    # Alternative: compute directly from validation data (memory-efficient)
    try:
        print("[INFO] Computing global filter in a memory-efficient way...")
        
        val_predictions = []
        batch_count = 0
        
        for x_batch, _ in val_dataset:
            try:
                batch_pred = transformer_model(x_batch)
                batch_pred_np = batch_pred.numpy()
                val_predictions.append(batch_pred_np)
                
                # Periodically free memory
                if batch_count % 5 == 0:
                    tf.keras.backend.clear_session()
                    import gc
                    gc.collect()
                    print(f"[INFO] Processed {batch_count} batches, GC done")
                
                batch_count += 1
                
            except tf.errors.ResourceExhaustedError:
                print(f"[WARNING] OOM at batch {batch_count}, skipping")
                tf.keras.backend.clear_session()
                gc.collect()
                continue
        
        if val_predictions:
            # Concatenate all predictions
            all_predictions = np.concatenate(val_predictions, axis=0)
            global_filter_alt = np.mean(all_predictions, axis=0)
            
            # Adjust dims
            if len(global_filter_alt.shape) == 2:
                global_filter_alt = global_filter_alt[:, :, np.newaxis]
            
            # Save
            sio.savemat(global_filter_path, {'global_filter': global_filter_alt})
            print(f"[INFO] Alternative global filter saved to: {global_filter_path}")
            print(f"[INFO] Alternative global filter shape: {global_filter_alt.shape}")
            
            # Cleanup
            del val_predictions, all_predictions
            gc.collect()
            
        else:
            print("[ERROR] OOM on all batches")
            
    except Exception as e2:
        print(f"[ERROR] Alternative method also failed: {e2}")

if 'USE_TRANSFORMER_GLOBAL_FILTER' in globals() and USE_TRANSFORMER_GLOBAL_FILTER:
    transformer_model.enable_transformer_global_filter(True)
elif 'USE_TRAINABLE_SHARED_FILTER' in globals() and USE_TRAINABLE_SHARED_FILTER:
    try:
        transformer_model.load_shared_filter("./AMMSE_revision_mse_shared_filter")
    except Exception as e:
        print(f"[WARN] 공유 필터 로드 실패: {e}")
    transformer_model.set_sharing_mode(2)
    transformer_model.enable_trainable_shared_filter(True, init_from_global=False)
else:
    transformer_model.load_global_output("./AMMSE_revision_mse")
    transformer_model.set_sharing_mode(2)

# ===========================
# 10. Evaluate the model
# ===========================
print("\n[INFO] Testing AMMSE with Huber Loss...")
transformer_loss, transformer_mse = final_transformer_model.evaluate(test_dataset)
print(f"Transformer - Test Loss (MSE): {transformer_loss:.8f}, Test MAE: {transformer_mse:.8f}")

# Compute signal power (mean squared of y_test)
y_power = tf.reduce_mean(tf.square(tf.abs(y_test)))
print("y_test.shape:", y_test.shape)
print("y_power.numpy():", y_power.numpy())
# Compute NMSE (MSE / signal power)
transformer_nmse = transformer_mse / y_power.numpy() 
transformer_nmse_dB = 10 * np.log10(transformer_nmse)
print(f"Transformer - Test NMSE: {transformer_nmse:.8f} ({transformer_nmse_dB:.8f} dB)")


# ===========================
# 11. Save all test predictions (.mat)
# ===========================
import os
import numpy as np
import scipy.io as sio
import tensorflow as tf

save_dir = "./AMMSE_revision_mse"          # use AMMSE_revision folder
os.makedirs(save_dir, exist_ok=True)

print("\n[INFO] Collecting ALL predictions from final_transformer_model...")

pred_list = []                                   # accumulate per-batch predictions
for x_batch, _ in test_dataset:
    batch_pred = final_transformer_model(x_batch)    # (B, 1008, 72) complex
    pred_list.append(batch_pred.numpy())             # move to CPU memory

# Concatenate to (N_test, 1008, 72)
all_pred = np.concatenate(pred_list, axis=0)
print("[INFO] Collected prediction shape :", all_pred.shape)   # e.g. (15120, 1008, 72)

# MATLAB-friendly order: (1008, 72, N_test)
all_pred_t = np.transpose(all_pred, (1, 2, 0))
print(all_pred_t.shape)

mat_path = os.path.join(save_dir, "AMMSE_pred_72_E_15dB_mse.mat")
sio.savemat(mat_path, {"transformer_outputs": all_pred_t})

print(f"[INFO] Saved ALL final outputs  →  {mat_path}")
print(f"[INFO] Saved shape (transposed): {all_pred_t.shape}")  # (1008, 72, N_test)

# ===========================
# NOTE ON TESTING / EVALUATION
# ===========================
# The "Evaluate the model" section above (printing test loss/NMSE) is a sanity
# check to verify that training progressed reasonably. It is NOT the final
# evaluation used for reporting or deployment.
#
# For the actual test/evaluation used in practice, load the saved global filter
# file (MATLAB .mat) and perform channel estimation externally (e.g., in MATLAB)
# using that filter with the true pilot observations. Then compute the desired
# metrics (e.g., MSE/NMSE/BER) on the reconstructed channels.
#
# Saved resources in this script:
# - Global filter (MAT-file): global_filter_path
# - All test predictions (MAT-file): mat_path
#
# Recommended workflow for final evaluation:
# 1) Train this script to produce the global filter file (.mat).
# 2) In MATLAB or your evaluation environment, load the saved filter and apply
#    it to the test dataset's pilots to estimate the full channel.
# 3) Compute and report your official metrics based on those external estimates.