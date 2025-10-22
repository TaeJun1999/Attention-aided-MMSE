import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import gc

###############################################################################
# 1) Input Preprocessing: Convert (B, 2, num_pilot, 1) → (B, 2*num_pilot, 1)
###############################################################################
class InputPreprocessing(layers.Layer):
    def call(self, x):
        x = tf.squeeze(x, axis=-1)  # (B, 2, num_pilot)
        x_real = x[:, 0, :]
        x_imag = x[:, 1, :]
        x_real = tf.expand_dims(x_real, axis=-1)
        x_imag = tf.expand_dims(x_imag, axis=-1)
        x = tf.concat([x_real, x_imag], axis=1)
        return x

###############################################################################
# 2) Transformer Encoder Block
###############################################################################
class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, d_ffn, dropout_rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ffn, activation='relu'),
            layers.Dense(d_model)
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.norm2(x + ffn_output)
        return x

###############################################################################
# 3) Residual Fully Connected Network
#    RA module(s): can be stacked; multiple RA modules may be used.
###############################################################################
class ResidualFC(layers.Layer):
    def __init__(self, num_pilot, d_model, d_model_out):
        super().__init__()
        self.num_pilot = num_pilot
        self.seq_length = 2 * num_pilot
        self.d_model = d_model
        self.d_model_out = d_model_out
        self.fc1 = layers.Dense(2 * num_pilot * d_model_out, activation='gelu')
        self.norm_fc1 = layers.LayerNormalization(epsilon=1e-6)
        self.shortcut_proj1 = layers.Dense(2 * num_pilot * d_model_out)
        self.fc2 = layers.Dense(2 * num_pilot * d_model_out, activation='gelu')
        self.norm_fc2 = layers.LayerNormalization(epsilon=1e-6)
        self.shortcut_proj2 = layers.Dense(2 * num_pilot * d_model_out)
        self.fc3 = layers.Dense(2 * num_pilot * d_model_out, activation='gelu')
        self.norm_fc3 = layers.LayerNormalization(epsilon=1e-6)
        self.shortcut_proj3 = layers.Dense(2 * num_pilot * d_model_out)
        # Note: when using RA module(s), consider adding an activation to fc4 as well
        self.fc4 = layers.Dense(2 * num_pilot * d_model_out)

        ################ For RA-A-MMSE ################
        #self.out_fc_real = layers.Dense(d_model_out, activation='gelu')
        #self.out_fc_imag = layers.Dense(d_model_out, activation='gelu')
        
        #self.u_real1 = self.add_weight(shape = (d_model_out, round(num_pilot * 0.9)), initializer = "random_normal", trainable = True)
        #self.u_imag1 = self.add_weight(shape = (d_model_out, round(num_pilot * 0.9)), initializer = "random_normal", trainable = True)

        #self.v_real1 = self.add_weight(shape = (round(num_pilot * 0.9), d_model_out), initializer = "random_normal", trainable = True)
        #self.v_imag1 = self.add_weight(shape = (round(num_pilot * 0.9), d_model_out), initializer = "random_normal", trainable = True)
        
        
    
    def call(self, x):
        x_flat = tf.reshape(x, (-1, self.seq_length * self.d_model))
        x_fc1 = self.fc1(x_flat)
        x_fc1 = self.norm_fc1(x_fc1 + self.shortcut_proj1(x_flat))
        x_fc2 = self.fc2(x_fc1)
        x_fc2 = self.norm_fc2(x_fc2 + self.shortcut_proj2(x_fc1))
        x_fc3 = self.fc3(x_fc2)
        x_fc3 = self.norm_fc3(x_fc3 + self.shortcut_proj3(x_fc2))
        x_out = self.fc4(x_fc3)
        x_out = tf.reshape(x_out, (-1, 2 * self.num_pilot, self.d_model_out))
        real_part, imag_part = tf.split(x_out, num_or_size_splits=2, axis=1)
        
        ################ For RA-A-MMSE ################
        #real_part = self.out_fc_real(real_part)
        #imag_part = self.out_fc_imag(imag_part)

        #real_part = tf.matmul(real_part, self.u_real1)
        #imag_part = tf.matmul(imag_part, self.u_imag1)

        #real_part = tf.matmul(real_part, self.v_real1)
        #imag_part = tf.matmul(imag_part, self.v_imag1)
      
        x_out = tf.stack([real_part, imag_part], axis=1)
        return x_out

###############################################################################
# 4) Transformer Model (memory-optimized)
###############################################################################
class TransformerModel(tf.keras.Model):
    def __init__(self, num_pilot, d_model1, d_model2, num_heads_1, num_heads_2, 
                 d_ffn1, d_ffn2, d_model_out, num_layers_1=1, num_layers_2=1, dropout_rate=0.1):
        super().__init__()
        self.input_preprocess = InputPreprocessing()
        self.embedding = layers.Dense(d_model1)

        # 첫 번째 Transformer Encoder Block 반복 가능
        self.encoders1 = [TransformerEncoder(d_model1, num_heads_1, d_ffn1, dropout_rate) for _ in range(num_layers_1)]
        
        self.proj = layers.Dense(d_model2)

        # 두 번째 Transformer Encoder Block 반복 가능
        self.encoders2 = [TransformerEncoder(d_model2, num_heads_2, d_ffn2, dropout_rate) for _ in range(num_layers_2)]
        
        self.fc_network = ResidualFC(num_pilot, d_model2, d_model_out)
        
        # Output sharing mode (0: no sharing, 1: per-batch sharing, 2: global sharing)
        self.sharing_mode = 0
        
        # Container for a precomputed/stored global output
        self.global_output = None
        
        # Trainable shared filter toggle and variable
        self.use_trainable_shared_filter = False
        self.shared_filter = None  # expected shape: (1, 2, num_pilot, d_model_out)
        
        # Hybrid blending (shared + transformer) configuration
        self.use_hybrid_blend = False
        self.blend_alpha = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.5),
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0),
            trainable=True,
            name="blend_alpha"
        )

        # Transformer-Global Filter mode: generate a single filter F_θ from a trainable global token
        self.use_transformer_global_filter = False
        self.global_token = self.add_weight(
            shape=(1, 2, num_pilot, 1),
            initializer=tf.zeros_initializer(),
            trainable=True,
            name="transformer_global_token"
        )
        
        # Trainable weight for batch contribution (used by some averaging strategies)
        self.batch_contribution = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.5),  # 초기값은 0.5 (균형)
            constraint=lambda x: tf.clip_by_value(x, 0, 1),  # 0과 1 사이로 제한
            trainable=True,
            name='batch_contribution'
        )
    
    def process_batch(self, x, training=False):
        """Generic batch processing routine (internal method)."""
        x = self.input_preprocess(x)
        x = self.embedding(x)

        # 첫 번째 Encoder 반복
        for encoder in self.encoders1:
            x = encoder(x, training=training)

        x = self.proj(x)

        # 두 번째 Encoder 반복
        for encoder in self.encoders2:
            x = encoder(x, training=training)

        out = self.fc_network(x)

        print("out.shape:", out.shape)
        return out
    
    def call(self, inputs, training=False):
        # Transformer-Global Filter: always produce a single shared filter from the transformer
        if self.use_transformer_global_filter:
            # Feed only the global token through the transformer path to generate a global filter
            out_global = self.process_batch(self.global_token, training=False)  # (1, 2, num_pilot, d_model_out)
            batch_size = tf.shape(inputs)[0]
            out = tf.tile(out_global, [batch_size, 1, 1, 1])
        # Hybrid blending: combine shared and transformer-generated outputs
        elif self.use_hybrid_blend:
            # 1) Transformer path
            out_trans = self.process_batch(inputs, training=training)  # (B, 2, num_pilot, d_model_out)
            # 2) Shared path
            batch_size = tf.shape(inputs)[0]
            if self.use_trainable_shared_filter and (self.shared_filter is not None):
                out_shared = tf.tile(self.shared_filter, [batch_size, 1, 1, 1])
            elif self.global_output is not None:
                out_shared = tf.tile(self.global_output, [batch_size, 1, 1, 1])
            else:
                out_shared = out_trans  # fallback
            # 3) Weighted sum
            out = self.blend_alpha * out_shared + (1.0 - self.blend_alpha) * out_trans
        else:
            # Sharing mode 2: always use one shared filter for both training and inference
            if self.sharing_mode == 2:
                batch_size = tf.shape(inputs)[0]
                if self.use_trainable_shared_filter and (self.shared_filter is not None):
                    out = tf.tile(self.shared_filter, [batch_size, 1, 1, 1])
                elif self.global_output is not None:
                    out = tf.tile(self.global_output, [batch_size, 1, 1, 1])
                else:
                    # Fallback to per-sample processing if not initialized
                    out = self.process_batch(inputs, training=training)
            elif self.sharing_mode == 1:
                out = self.process_batch(inputs, training=training)
                batch_mean = tf.reduce_mean(out, axis=0, keepdims=True)
                batch_size = tf.shape(inputs)[0]
                out = tf.tile(batch_mean, [batch_size, 1, 1, 1])
            else:
                # No sharing: per-sample processing
                out = self.process_batch(inputs, training=training)
        
        # Common output conversion
        out = tf.transpose(out, perm=[0, 1, 3, 2])
        out = tf.complex(out[:, 0, :, :], out[:, 1, :, :])
        return out
    
    # 메모리 효율적인 글로벌 출력 계산 메서드
    def calculate_global_output(self, dataset, gc_interval=10):
        """
        Memory-efficient computation of a global output (incremental mean with periodic GC).
        
        Args:
            dataset: TensorFlow dataset of (x, y).
            gc_interval: how often to trigger garbage collection (in number of batches).
        
        Returns:
            global_output: the computed global output tensor.
        """
        print("[INFO] Computing global output (memory-optimized)...")
        
        # Backup original sharing_mode
        original_mode = self.sharing_mode
        self.sharing_mode = 0  # disable sharing during computation
        
        # Variables for incremental mean computation
        running_mean = None
        total_samples = 0
        batch_count = 0
        
        # Warm-up with the first batch to determine shapes
        for x_batch, _ in dataset.take(1):
            # Initial computation to confirm output shape
            sample_output = self.process_batch(x_batch, training=False)
            # Initialize the running mean to zeros
            running_mean = tf.zeros_like(tf.reduce_mean(sample_output, axis=0, keepdims=True))
            break
        
        # Iterate over the dataset and accumulate batch means
        for x_batch, _ in dataset:
            batch_size = tf.shape(x_batch)[0]
            
            # Compute outputs with the standard processing path
            batch_output = self.process_batch(x_batch, training=False)
            
            # Average within the batch
            batch_mean = tf.reduce_mean(batch_output, axis=0, keepdims=True)
            
            # Incrementally update the running mean (Welford-like update)
            # new_mean = old_mean + (batch_mean - old_mean) * (batch_size / new_total)
            old_total = total_samples
            total_samples += batch_size
            weight = tf.cast(batch_size, tf.float32) / tf.cast(total_samples, tf.float32)
            running_mean += (batch_mean - running_mean) * weight
            
            # Explicitly release batch tensors
            del batch_output
            del batch_mean
            
            batch_count += 1
            if batch_count % gc_interval == 0:
                # Periodically run garbage collection
                gc.collect()
                print(f"[INFO] Processed {batch_count} batches... (total {total_samples} samples, GC done)")
        
        # Set the final mean as the global output
        self.global_output = running_mean
        
        # Final memory cleanup
        gc.collect()
        
        # Restore original sharing mode
        self.sharing_mode = original_mode
        
        print(f"[INFO] Global output computed! (processed {total_samples} samples)")
        return self.global_output
    
    # 배치 크기 조절을 통한 효율적 글로벌 출력 계산
    def calculate_global_output_with_small_batches(self, dataset, max_samples_per_batch=10):
        """
        Compute a global output using smaller sub-batches for memory efficiency.
        
        Args:
            dataset: TensorFlow dataset of (x, y).
            max_samples_per_batch: max number of samples per sub-batch.
        
        Returns:
            global_output: the computed global output tensor.
        """
        print("[INFO] Computing global output with small batches...")
        
        # Backup original sharing_mode
        original_mode = self.sharing_mode
        self.sharing_mode = 0  # disable sharing during computation
        
        # Variables for incremental mean computation
        running_mean = None
        total_samples = 0
        batch_count = 0
        
        # Iterate over the dataset and split large batches into smaller ones
        for x_batch, _ in dataset:
            original_batch_size = tf.shape(x_batch)[0]
            
            # Split the batch into smaller sub-batches if necessary
            for start_idx in range(0, original_batch_size, max_samples_per_batch):
                end_idx = min(start_idx + max_samples_per_batch, original_batch_size)
                sub_batch = x_batch[start_idx:end_idx]
                sub_batch_size = end_idx - start_idx
                
                # Process the current mini sub-batch
                mini_output = self.process_batch(sub_batch, training=False)
                mini_mean = tf.reduce_mean(mini_output, axis=0, keepdims=True)
                
                # Initialize the running mean or update it incrementally
                if running_mean is None:
                    running_mean = mini_mean
                    total_samples = sub_batch_size
                else:
                    # Incremental mean update
                    old_total = total_samples
                    total_samples += sub_batch_size
                    weight = tf.cast(sub_batch_size, tf.float32) / tf.cast(total_samples, tf.float32)
                    running_mean += (mini_mean - running_mean) * weight
                
                # Release tensors from the mini sub-batch
                del mini_output
                del mini_mean
                
                gc.collect()
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"[INFO] Processed {batch_count} batches... (total {total_samples} samples)")
        
        # Set the final mean as the global output
        self.global_output = running_mean
        
        # Final memory cleanup
        gc.collect()
        
        # Restore original sharing mode
        self.sharing_mode = original_mode
        
        print(f"[INFO] Global output computed! (processed {total_samples} samples)")
        return self.global_output
    
    def set_sharing_mode(self, mode):
        """
        Configure output sharing mode.
        
        Args:
            mode: 0 (no sharing), 1 (per-batch sharing), or 2 (global sharing).
        """
        if mode not in [0, 1, 2]:
            raise ValueError("Sharing mode must be one of 0, 1, 2.")
        
        self.sharing_mode = mode
        
        if mode == 2 and self.global_output is None:
            print("[WARN] Global output not computed yet. Call calculate_global_output() first.")
        
        mode_names = {0: "disabled", 1: "per-batch", 2: "global"}
        print(f"[INFO] Sharing mode set to '{mode_names[mode]}'")
        
        return self
    
    def save_global_output(self, filepath):
        """Save the global output tensor to a file."""
        if self.global_output is None:
            raise ValueError("Global output has not been computed.")
        
        # Ensure the output directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[INFO] Created directory {directory}")
        
        # Auto-append .npy extension if missing
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'
            
        np_array = self.global_output.numpy()
        np.save(filepath, np_array)
        print(f"[INFO] Saved global output to {filepath}")
    
    def load_global_output(self, filepath):
        """Load the global output tensor from a file."""
        # 확장자가 없는 경우 .npy 추가
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'
            
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
            
        try:
            np_array = np.load(filepath)
            self.global_output = tf.convert_to_tensor(np_array)
            print(f"[INFO] Loaded global output from {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to load global output: {e}")
            raise
    
    def get_global_output(self):
        """Return the current global output tensor (or None)."""
        if self.global_output is None:
            print("[WARN] Global output has not been computed.")
            return None
        return self.global_output

    # ===== 학습 가능한 공유 필터 관련 유틸 =====
    def enable_trainable_shared_filter(self, enabled=True, init_from_global=True):
        """
        Enable a trainable shared filter and optionally initialize it.
        If init_from_global=True and a global_output exists, it is used as the initializer.
        """
        self.use_trainable_shared_filter = enabled
        if enabled and self.shared_filter is None:
            num_pilot = self.fc_network.num_pilot
            d_model_out = self.fc_network.d_model_out
            init = tf.zeros_initializer()
            if init_from_global and (self.global_output is not None):
                init = tf.constant_initializer(self.global_output.numpy())
            self.shared_filter = self.add_weight(
                shape=(1, 2, num_pilot, d_model_out),
                initializer=init,
                trainable=True,
                name='trainable_shared_filter'
            )
            print("[INFO] Trainable shared filter initialized.")
        return self

    def set_shared_filter_from_numpy(self, np_array):
        """Set the shared filter from an external numpy array.
        Expected shapes: (1, 2, Np, Ns) or (2, Np, Ns).
        """
        if np_array is None:
            raise ValueError("np_array is None")
        if np_array.ndim == 3 and np_array.shape[0] == 2:
            np_array = np_array[np.newaxis, ...]
        if np_array.ndim != 4:
            raise ValueError(f"Unexpected shape for shared filter: {np_array.shape}")
        if self.shared_filter is None:
            self.shared_filter = self.add_weight(
                shape=np_array.shape,
                initializer=tf.constant_initializer(np_array),
                trainable=True,
                name='trainable_shared_filter'
            )
        else:
            self.shared_filter.assign(np_array)
        self.use_trainable_shared_filter = True
        print("[INFO] Shared filter set from external array.")
        return self

    def save_shared_filter(self, filepath):
        """Save the trained shared filter to a .npy file."""
        if self.shared_filter is None:
            raise ValueError("Shared filter is not set.")
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[INFO] Created directory {directory}")
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'
        np.save(filepath, self.shared_filter.numpy())
        print(f"[INFO] Saved shared filter to {filepath}")

    def load_shared_filter(self, filepath):
        """Load a shared filter from a .npy file and set it as trainable."""
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        arr = np.load(filepath)
        self.set_shared_filter_from_numpy(arr)
        print(f"[INFO] Loaded shared filter from {filepath}")

    def enable_hybrid_blend(self, enabled=True, init_alpha=None):
        """Enable/disable hybrid blending and optionally set the initial alpha."""
        self.use_hybrid_blend = enabled
        if (init_alpha is not None) and (0.0 <= init_alpha <= 1.0):
            self.blend_alpha.assign([init_alpha])
        mode = "ON" if enabled else "OFF"
        print(f"[INFO] Hybrid blend mode: {mode}, alpha={float(self.blend_alpha.numpy()[0]):.3f}")
        return self

    def enable_transformer_global_filter(self, enabled=True, init_from=None):
        """
        Enable/disable Transformer-Global Filter mode.
        init_from: None | 'global_output' | 'shared_filter'
        """
        self.use_transformer_global_filter = enabled
        if enabled and init_from is not None:
            if init_from == 'global_output' and (self.global_output is not None):
                # Initialize from the average across the last axis to match token shape
                # Global token has shape (1, 2, Np, 1)
                token_init = tf.reduce_mean(self.global_output, axis=-1, keepdims=True)  # (1,2,Np,1)
                self.global_token.assign(token_init)
            elif init_from == 'shared_filter' and (self.shared_filter is not None):
                token_init = tf.reduce_mean(self.shared_filter, axis=-1, keepdims=True)  # (1,2,Np,1)
                self.global_token.assign(token_init)
        mode = "ON" if enabled else "OFF"
        print(f"[INFO] Transformer-Global Filter mode: {mode}")
        return self

    def compute_transformer_global_output(self, training=False):
        """Compute and store the Transformer global output generated from the global token."""
        out_global = self.process_batch(self.global_token, training=training)  # (1,2,Np,Ns)
        self.global_output = out_global
        return self.global_output


###############################################################################
# 5) Test
###############################################################################
if __name__ == "__main__":
    B = 1
    num_pilot = 72
    d_model1 = 72
    d_model2 = 1008
    num_heads_1 = 36
    num_heads_2 = 14
    d_ffn1 = num_pilot * 4
    d_ffn2 = num_pilot * 4
    d_model_out = 1008
    num_layers_1 = 1  # 첫 번째 Encoder 반복 횟수
    num_layers_2 = 1  # 두 번째 Encoder 반복 횟수
    dropout_rate = 0

    x_input = tf.random.normal((B, 2, num_pilot, 1))
    
    model = TransformerModel(num_pilot, d_model1, d_model2, num_heads_1, num_heads_2, 
                             d_ffn1, d_ffn2, d_model_out, num_layers_1, num_layers_2, dropout_rate)
    
    model.build(input_shape=(B, 2, num_pilot, 1))
    model.summary()
    
    # Basic output mode test
    model.set_sharing_mode(0)
    out1 = model(x_input)
    print("\n===== Output (no sharing) =====")
    print("shape:", out1.shape)
    
    # Per-batch sharing mode test
    model.set_sharing_mode(1)
    out2 = model(x_input)
    print("\n===== Output (per-batch sharing) =====")
    print("shape:", out2.shape)
    
    # Build a small mock dataset for testing
    mock_dataset = tf.data.Dataset.from_tensor_slices((x_input, x_input)).batch(8)
    
    # Compute global output in a memory-efficient way
    model.calculate_global_output(mock_dataset)
    
    # Global sharing mode test
    model.set_sharing_mode(2)
    out3 = model(x_input)
    print("\n===== Output (global sharing) =====")
    print("shape:", out3.shape)
    
    # Verify that all batch samples are identical
    are_all_same = True
    first_sample = out3[0]
    for i in range(1, B):
        if not tf.reduce_all(tf.equal(first_sample, out3[i])):
            are_all_same = False
            break
    
    print("Are all batch outputs identical?", are_all_same)