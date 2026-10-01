import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def complex_mse(y_true, y_pred):
    """Loss (28): mean |e|^2 of the complex channel estimate."""
    e = tf.cast(y_true, y_pred.dtype) - y_pred
    return tf.reduce_mean(tf.math.real(e) ** 2 + tf.math.imag(e) ** 2, axis=-1)


class InputPreprocessing(layers.Layer):
    """(B, 2, L, 1) -> (B, 2L, 1): real parts followed by imaginary parts."""
    def call(self, x):
        x = tf.squeeze(x, axis=-1)  # (B, 2, num_pilot)
        x_real = x[:, 0, :]
        x_imag = x[:, 1, :]
        x_real = tf.expand_dims(x_real, axis=-1)
        x_imag = tf.expand_dims(x_imag, axis=-1)
        x = tf.concat([x_real, x_imag], axis=1)
        return x


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


class ResidualFC(layers.Layer):
    """Residual fully connected decoder: (B, 2L, NM) -> filter coefficients (B, 2, L, NM)."""
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
        self.fc4 = layers.Dense(2 * num_pilot * d_model_out)

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
        x_out = tf.stack([real_part, imag_part], axis=1)
        return x_out


class TransformerModel(tf.keras.Model):
    """Two-stage attention encoder and ResidualFC decoder; call() returns complex filters (B, NM, L)."""
    def __init__(self, num_pilot, d_model1, d_model2, num_heads_1, num_heads_2,
                 d_ffn1, d_ffn2, d_model_out, num_layers_1=1, num_layers_2=1, dropout_rate=0.1):
        super().__init__()
        self.input_preprocess = InputPreprocessing()
        self.embedding = layers.Dense(d_model1)

        # Frequency encoder
        self.encoders1 = [TransformerEncoder(d_model1, num_heads_1, d_ffn1, dropout_rate) for _ in range(num_layers_1)]

        self.proj = layers.Dense(d_model2)

        # Temporal encoder
        self.encoders2 = [TransformerEncoder(d_model2, num_heads_2, d_ffn2, dropout_rate) for _ in range(num_layers_2)]

        self.fc_network = ResidualFC(num_pilot, d_model2, d_model_out)

        # Output sharing mode (0: no sharing, 1: per-batch sharing, 2: global sharing)
        self.sharing_mode = 0

        # Precomputed filter used by sharing mode 2, (1, 2, num_pilot, d_model_out)
        self.global_output = None

        # Trainable shared filter, (1, 2, num_pilot, d_model_out)
        self.use_trainable_shared_filter = False
        self.shared_filter = None

        # Hybrid mode: alpha * shared filter + (1 - alpha) * per-sample filter
        self.use_hybrid_blend = False
        self.blend_alpha = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.5),
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0),
            trainable=True,
            name="blend_alpha"
        )

        # Global-token mode: a single filter generated from a trainable token instead of the input
        self.use_transformer_global_filter = False
        self.global_token = self.add_weight(
            shape=(1, 2, num_pilot, 1),
            initializer=tf.zeros_initializer(),
            trainable=True,
            name="transformer_global_token"
        )

    def process_batch(self, x, training=False):
        """Pilots (B, 2, L, 1) -> filter coefficients (B, 2, L, NM)."""
        x = self.input_preprocess(x)
        x = self.embedding(x)

        for encoder in self.encoders1:
            x = encoder(x, training=training)

        x = self.proj(x)

        for encoder in self.encoders2:
            x = encoder(x, training=training)

        return self.fc_network(x)

    def call(self, inputs, training=False):
        if self.use_transformer_global_filter:
            # Only the token goes through the network; its filter is used for the whole batch
            out_global = self.process_batch(self.global_token, training=False)  # (1, 2, num_pilot, d_model_out)
            batch_size = tf.shape(inputs)[0]
            out = tf.tile(out_global, [batch_size, 1, 1, 1])
        elif self.use_hybrid_blend:
            out_trans = self.process_batch(inputs, training=training)  # (B, 2, num_pilot, d_model_out)
            batch_size = tf.shape(inputs)[0]
            if self.use_trainable_shared_filter and (self.shared_filter is not None):
                out_shared = tf.tile(self.shared_filter, [batch_size, 1, 1, 1])
            elif self.global_output is not None:
                out_shared = tf.tile(self.global_output, [batch_size, 1, 1, 1])
            else:
                out_shared = out_trans
            out = self.blend_alpha * out_shared + (1.0 - self.blend_alpha) * out_trans
        else:
            if self.sharing_mode == 2:
                batch_size = tf.shape(inputs)[0]
                if self.use_trainable_shared_filter and (self.shared_filter is not None):
                    out = tf.tile(self.shared_filter, [batch_size, 1, 1, 1])
                elif self.global_output is not None:
                    out = tf.tile(self.global_output, [batch_size, 1, 1, 1])
                else:
                    # No shared filter yet: fall back to per-sample filters
                    out = self.process_batch(inputs, training=training)
            elif self.sharing_mode == 1:
                out = self.process_batch(inputs, training=training)
                batch_mean = tf.reduce_mean(out, axis=0, keepdims=True)
                batch_size = tf.shape(inputs)[0]
                out = tf.tile(batch_mean, [batch_size, 1, 1, 1])
            else:
                out = self.process_batch(inputs, training=training)

        # (B, 2, L, NM) -> complex (B, NM, L)
        out = tf.transpose(out, perm=[0, 1, 3, 2])
        out = tf.complex(out[:, 0, :, :], out[:, 1, :, :])
        return out

    def calculate_global_output(self, dataset):
        """Average the per-sample filters over a dataset and store the result as global_output."""
        running_mean, total_samples = 0.0, 0
        for x_batch, _ in dataset:
            batch_output = self.process_batch(x_batch, training=False)
            batch_size = x_batch.shape[0]
            total_samples += batch_size
            running_mean += (tf.reduce_mean(batch_output, axis=0, keepdims=True) - running_mean) * (batch_size / total_samples)
        self.global_output = running_mean
        print(f"Global output computed from {total_samples} samples")
        return self.global_output

    def set_sharing_mode(self, mode):
        """0: per-sample filters, 1: batch-mean filter, 2: one shared filter for all samples."""
        if mode not in [0, 1, 2]:
            raise ValueError("Sharing mode must be one of 0, 1, 2.")

        self.sharing_mode = mode

        if mode == 2 and self.global_output is None:
            print("Warning: global output is not set yet, call calculate_global_output() first")

        mode_names = {0: "disabled", 1: "per-batch", 2: "global"}
        print(f"Sharing mode: {mode_names[mode]}")

        return self

    def save_global_output(self, filepath):
        """Save global_output as .npy (the extension is added if missing)."""
        if self.global_output is None:
            raise ValueError("Global output has not been computed.")

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'

        np.save(filepath, self.global_output.numpy())
        print(f"Saved global output to {filepath}")

    def load_global_output(self, filepath):
        """Load global_output from .npy (the extension is added if missing)."""
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'
        self.global_output = tf.convert_to_tensor(np.load(filepath))
        print(f"Loaded global output from {filepath}")

    # Trainable shared filter
    def enable_trainable_shared_filter(self, enabled=True, init_from_global=True):
        """Turn on the trainable shared filter, initialized from global_output if available."""
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
            print("Trainable shared filter initialized")
        return self

    def set_shared_filter_from_numpy(self, np_array):
        """Set the shared filter from an array of shape (1, 2, Np, Ns) or (2, Np, Ns)."""
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
        return self

    def save_shared_filter(self, filepath):
        """Save the shared filter as .npy (the extension is added if missing)."""
        if self.shared_filter is None:
            raise ValueError("Shared filter is not set.")
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'
        np.save(filepath, self.shared_filter.numpy())
        print(f"Saved shared filter to {filepath}")

    def load_shared_filter(self, filepath):
        """Load the shared filter from .npy and make it trainable."""
        if not filepath.endswith('.npy'):
            filepath = filepath + '.npy'
        self.set_shared_filter_from_numpy(np.load(filepath))
        print(f"Loaded shared filter from {filepath}")

    def enable_hybrid_blend(self, enabled=True, init_alpha=None):
        """Turn hybrid blending on or off, optionally setting the initial alpha."""
        self.use_hybrid_blend = enabled
        if (init_alpha is not None) and (0.0 <= init_alpha <= 1.0):
            self.blend_alpha.assign([init_alpha])
        print(f"Hybrid blend: {'on' if enabled else 'off'}, alpha = {float(self.blend_alpha.numpy()[0]):.3f}")
        return self

    def enable_transformer_global_filter(self, enabled=True, init_from=None):
        """Turn global-token mode on or off. init_from: None, 'global_output' or 'shared_filter'."""
        self.use_transformer_global_filter = enabled
        if enabled and init_from is not None:
            # The token is (1, 2, Np, 1): initialize it with the filter averaged over the last axis
            if init_from == 'global_output' and (self.global_output is not None):
                self.global_token.assign(tf.reduce_mean(self.global_output, axis=-1, keepdims=True))
            elif init_from == 'shared_filter' and (self.shared_filter is not None):
                self.global_token.assign(tf.reduce_mean(self.shared_filter, axis=-1, keepdims=True))
        print(f"Global-token mode: {'on' if enabled else 'off'}")
        return self

    def compute_transformer_global_output(self, training=False):
        """Generate the filter from the global token and store it as global_output."""
        out_global = self.process_batch(self.global_token, training=training)  # (1, 2, Np, Ns)
        self.global_output = out_global
        return self.global_output
