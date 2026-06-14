import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Dropout, LayerNormalization
import numpy as np


def positional_encoding(position, d_model):
    """Fixed and cleaner positional encoding"""
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    
    angle_rads = angle_rads[np.newaxis, ...]
    angle_rads = tf.cast(angle_rads, dtype=tf.float32)
    
    sines = tf.math.sin(angle_rads[:, :, 0::2])
    cosines = tf.math.cos(angle_rads[:, :, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


class PositionwiseFeedforward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super().__init__()
        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))


class TransformerBlock(tf.keras.layers.Layer):
    """Used for Encoder (self-attention only)"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedforward(d_model, dff)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        attn_output, _ = self.att(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder block with masked self-attention + cross-attention"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_att = MultiHeadAttention(d_model, num_heads)
        self.cross_att = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedforward(d_model, dff)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # Masked Self Attention
        self_attn_output, _ = self.self_att(x, x, x, look_ahead_mask)
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(x + self_attn_output)

        # Cross Attention
        cross_attn_output, _ = self.cross_att(out1, enc_output, enc_output, padding_mask)
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output)

        # Feed Forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout = Dropout(dropout_rate)
        self.enc_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) 
                          for _ in range(num_layers)]

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dropout = Dropout(dropout_rate)
        self.dec_layers = [DecoderBlock(d_model, num_heads, dff, dropout_rate) 
                          for _ in range(num_layers)]

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, training, look_ahead_mask, padding_mask)
        
        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, maximum_position_encoding, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, maximum_position_encoding, dropout_rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inputs, training=False, look_ahead_mask=None, padding_mask=None):
        inp, tar = inputs
        enc_output = self.encoder(inp, training=training, mask=padding_mask)
        dec_output = self.decoder(tar, enc_output, training=training,
                                  look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output


# ========================== TEST ==========================
if __name__ == "__main__":
    num_layers = 4
    d_model = 128
    num_heads = 8
    dff = 512
    input_vocab_size = 8500
    target_vocab_size = 8000
    maximum_position_encoding = 10000
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers, d_model, num_heads, dff,
        input_vocab_size, target_vocab_size, maximum_position_encoding, dropout_rate
    )

    inputs = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=input_vocab_size)
    targets = tf.random.uniform((64, 50), dtype=tf.int64, minval=0, maxval=target_vocab_size)

    output = transformer((inputs, targets), training=True)
    print("Output shape:", output.shape)   