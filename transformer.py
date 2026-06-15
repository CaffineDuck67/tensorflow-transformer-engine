

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout, LayerNormalization
import numpy as np

from masks import create_padding_mask, create_look_ahead_mask, create_combined_mask


# ===========================================================================
# Positional Encoding 
# ===========================================================================

def positional_encoding(position, d_model):
   
    positions = np.arange(position)[:, np.newaxis]          # (pos, 1)
    dims      = np.arange(d_model)[np.newaxis, :]           # (1, d_model)

    angle_rads = positions / np.power(
        10000,
        (2 * (dims // 2)) / np.float32(d_model)
    )  # shape: (position, d_model)

    # Apply sin to even indices (0, 2, 4, …) and cos to odd indices (1, 3, 5, …)
    # This correctly interleaves sin and cos along the last dimension.
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # even dims → sin
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # odd  dims → cos

    # Add batch dimension: (1, position, d_model)
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# ===========================================================================
# Multi-Head Attention 
# ===========================================================================

class MultiHeadAttention(tf.keras.layers.Layer):
  
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model   = d_model
        self.depth     = d_model // num_heads  # dimension per head

        # Learned linear projections for Q, K, V and the output
        self.wq    = Dense(d_model, name="wq")
        self.wk    = Dense(d_model, name="wk")
        self.wv    = Dense(d_model, name="wv")
        self.dense = Dense(d_model, name="out_proj")

    # -----------------------------------------------------------------------
    def split_heads(self, x, batch_size):
        """
        Reshape (batch, seq_len, d_model) → (batch, num_heads, seq_len, depth).
        Each head sees its own slice of the embedding dimension.
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # -----------------------------------------------------------------------
    def scaled_dot_product_attention(self, q, k, v, mask):
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale by √d_k to keep gradients stable
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask: add a large negative number so softmax → ~0 there
        if mask is not None:
            scaled_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    # -----------------------------------------------------------------------
    def call(self, q, k, v, mask=None):
        """
        Args:
            q, k, v : token representations — shape (batch, seq_len, d_model)
            mask    : optional mask (see masks.py)

        Returns:
            output          : (batch, seq_q, d_model)
            attention_weights: (batch, num_heads, seq_q, seq_k)
        """
        batch_size = tf.shape(q)[0]

        # Project inputs into Q, K, V spaces
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Attend
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # Merge heads back: (batch, seq_q, num_heads, depth) → (batch, seq_q, d_model)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.dense(concat_attention)
        return output, attention_weights

    # -----------------------------------------------------------------------
    def get_config(self):
        """Enable model saving/loading via model.save() and tf.keras.models.load_model()."""
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config


# ===========================================================================
# Position-wise Feed-Forward Network
# ===========================================================================

class PositionwiseFeedforward(tf.keras.layers.Layer):
    """
    Two-layer fully-connected network applied independently to each position.

        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    The inner layer expands the dimension to `dff`, then the outer layer
    projects back to `d_model`.

    Args:
        d_model : output (and input) dimension
        dff     : inner/hidden dimension (typically 4 × d_model)
    """

    def __init__(self, d_model, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff     = dff
        self.dense1  = Dense(dff,     activation='relu', name="ffn_inner")
        self.dense2  = Dense(d_model,                    name="ffn_outer")

    def call(self, x):
        return self.dense2(self.dense1(x))

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "dff": self.dff})
        return config


# ===========================================================================
# Encoder Block  (self-attention only)
# ===========================================================================

class TransformerBlock(tf.keras.layers.Layer):
    """
    Single Encoder layer:
        x → MultiHeadSelfAttention → Add & Norm → FFN → Add & Norm

    Args:
        d_model      : model dimension
        num_heads    : number of attention heads
        dff          : feed-forward inner dimension
        dropout_rate : dropout probability (applied after each sub-layer)
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.dff          = dff
        self.dropout_rate = dropout_rate

        self.att        = MultiHeadAttention(d_model, num_heads)
        self.ffn        = PositionwiseFeedforward(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1   = Dropout(dropout_rate)
        self.dropout2   = Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        # --- Sub-layer 1: self-attention ---
        attn_output, _ = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)        # residual + norm

        # --- Sub-layer 2: feed-forward ---
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)       # residual + norm

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ===========================================================================
# Decoder Block  (masked self-attention + cross-attention + FFN)
# ===========================================================================

class DecoderBlock(tf.keras.layers.Layer):
    """
    Single Decoder layer — three sub-layers:
      1. Masked multi-head self-attention  (uses look-ahead mask)
      2. Cross-attention over encoder output  (uses padding mask)
      3. Feed-forward network

    Args:
        d_model        : model dimension
        num_heads      : attention heads
        dff            : feed-forward inner dimension
        dropout_rate   : dropout probability
    """

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.dff          = dff
        self.dropout_rate = dropout_rate

        self.self_att  = MultiHeadAttention(d_model, num_heads)   # masked self-attn
        self.cross_att = MultiHeadAttention(d_model, num_heads)   # cross-attn

        self.ffn = PositionwiseFeedforward(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x               : target embeddings (batch, tgt_len, d_model)
            enc_output      : encoder output    (batch, src_len, d_model)
            training        : bool — enables dropout during training
            look_ahead_mask : causal mask for self-attention
            padding_mask    : padding mask for cross-attention (hides src PADs)
        """
        # --- Sub-layer 1: masked self-attention (Q=K=V = decoder input) ---
        self_attn_output, _ = self.self_att(x, x, x, look_ahead_mask)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(x + self_attn_output)

        # --- Sub-layer 2: cross-attention (Q = decoder, K/V = encoder) ---
        cross_attn_output, _ = self.cross_att(out1, enc_output, enc_output, padding_mask)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output)

        # --- Sub-layer 3: feed-forward ---
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ===========================================================================
# Encoder  (stack of TransformerBlock layers)
# ===========================================================================

class Encoder(tf.keras.layers.Layer):
    """
    Full encoder: embedding + positional encoding + N × TransformerBlock.

    Args:
        num_layers               : number of stacked encoder blocks
        d_model                  : model dimension
        num_heads                : attention heads per block
        dff                      : feed-forward inner dimension
        input_vocab_size         : source vocabulary size
        maximum_position_encoding: maximum expected sequence length
        dropout_rate             : dropout probability
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers               = num_layers
        self.d_model                  = d_model
        self.num_heads                = num_heads
        self.dff                      = dff
        self.input_vocab_size         = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate             = dropout_rate

        self.embedding    = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout      = Dropout(dropout_rate)
        self.enc_layers   = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        # Token embedding + positional encoding
        x = self.embedding(x)                          # (batch, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # scale (paper §3.4)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "maximum_position_encoding": self.maximum_position_encoding,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ===========================================================================
# Decoder  (stack of DecoderBlock layers)
# ===========================================================================

class Decoder(tf.keras.layers.Layer):
    """
    Full decoder: embedding + positional encoding + N × DecoderBlock.

    Args: (mirrors Encoder, but uses target_vocab_size)
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers               = num_layers
        self.d_model                  = d_model
        self.num_heads                = num_heads
        self.dff                      = dff
        self.target_vocab_size        = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate             = dropout_rate

        self.embedding    = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout      = Dropout(dropout_rate)
        self.dec_layers   = [
            DecoderBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # scale (paper §3.4)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, training=training,
                          look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "target_vocab_size": self.target_vocab_size,
            "maximum_position_encoding": self.maximum_position_encoding,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ===========================================================================
# Transformer  (full model)
# ===========================================================================

class Transformer(tf.keras.Model):
    """
    Complete Encoder–Decoder Transformer.

    Inputs:
        (inp, tar) — source and (shifted-right) target token ID tensors.

    Output:
        Logits of shape (batch, tgt_seq_len, target_vocab_size).
        Apply softmax externally or use SparseCategoricalCrossentropy
        with from_logits=True during training.
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers               = num_layers
        self.d_model                  = d_model
        self.num_heads                = num_heads
        self.dff                      = dff
        self.input_vocab_size         = input_vocab_size
        self.target_vocab_size        = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate             = dropout_rate

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff,
            input_vocab_size, maximum_position_encoding, dropout_rate
        )
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff,
            target_vocab_size, maximum_position_encoding, dropout_rate
        )
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, training=False, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            inputs          : tuple (inp, tar) — source and target token IDs
            training        : bool
            look_ahead_mask : combined mask for decoder self-attention
            padding_mask    : padding mask for encoder output (cross-attention)
        """
        inp, tar = inputs
        enc_output = self.encoder(inp, training=training, mask=padding_mask)
        dec_output = self.decoder(
            tar, enc_output, training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=padding_mask
        )
        final_output = self.final_layer(dec_output)  # (batch, tgt_len, vocab_size)
        return final_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "maximum_position_encoding": self.maximum_position_encoding,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ===========================================================================
# Quick smoke test — uses real masks from masks.py
# ===========================================================================

if __name__ == "__main__":
    # --- Hyper-parameters ---
    num_layers    = 4
    d_model       = 128
    num_heads     = 8
    dff           = 512
    input_vocab_size  = 8500
    target_vocab_size = 8000
    maximum_position_encoding = 10000
    dropout_rate  = 0.1

    # --- Build model ---
    transformer = Transformer(
        num_layers, d_model, num_heads, dff,
        input_vocab_size, target_vocab_size,
        maximum_position_encoding, dropout_rate
    )

    # --- Synthetic batch ---
    batch_size = 64
    src_len    = 50
    tgt_len    = 50

    inputs  = tf.random.uniform((batch_size, src_len), dtype=tf.int64,
                                minval=0, maxval=input_vocab_size)
    targets = tf.random.uniform((batch_size, tgt_len), dtype=tf.int64,
                                minval=0, maxval=target_vocab_size)

    # --- Generate real masks (replaces the old mask=None) ---
    enc_padding_mask  = create_padding_mask(inputs)      # (batch, 1, 1, src_len)
    combined_mask     = create_combined_mask(targets)    # (batch, 1, tgt_len, tgt_len)
    dec_padding_mask  = create_padding_mask(inputs)      # (batch, 1, 1, src_len)

    # --- Forward pass ---
    output = transformer(
        (inputs, targets),
        training=True,
        look_ahead_mask=combined_mask,
        padding_mask=enc_padding_mask,
    )

    print("Output shape:", output.shape)
    