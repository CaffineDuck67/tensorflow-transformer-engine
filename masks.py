

import tensorflow as tf


# ---------------------------------------------------------------------------
# 1. Padding mask
# ---------------------------------------------------------------------------

def create_padding_mask(seq):
    """
    Create a mask that hides PAD tokens (integer value 0) in a sequence.

    Args:
        seq: int tensor of shape (batch_size, seq_len)

    Returns:
        mask: float tensor of shape (batch_size, 1, 1, seq_len)
              1.0 where seq == 0 (pad), 0.0 elsewhere.

    The extra dimensions (1, 1) let TensorFlow broadcast the mask against
    the attention logits of shape (batch, num_heads, query_len, key_len).
    """
    # Cast to float: 1.0 at pad positions, 0.0 elsewhere
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Expand dims so shape becomes (batch, 1, 1, seq_len)
    return mask[:, tf.newaxis, tf.newaxis, :]


# ---------------------------------------------------------------------------
# 2. Look-ahead mask
# ---------------------------------------------------------------------------

def create_look_ahead_mask(size):
    """
    Create a causal (look-ahead) mask of shape (size, size).

    Position [i, j] is 1 (blocked) when j > i, meaning token i cannot
    attend to token j because j comes AFTER i in the sequence.

    Example for size=4:
        [[0, 1, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]]

    Args:
        size: int — sequence length (both query and key share the same
              length in self-attention).

    Returns:
        mask: float tensor of shape (size, size)
    """
    # 1 - lower-triangular matrix gives upper-triangle of 1s (future positions)
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # shape: (size, size)


# ---------------------------------------------------------------------------
# 3. Combined mask for the decoder's masked self-attention
# ---------------------------------------------------------------------------

def create_combined_mask(target):
    """
    Combine padding mask and look-ahead mask for the decoder self-attention.

    The decoder's first sub-layer needs to:
      (a) ignore PAD tokens in the target sequence, AND
      (b) not look at future target tokens.

    We OR both masks together (take the maximum element-wise).

    Args:
        target: int tensor of shape (batch_size, seq_len) — target token IDs

    Returns:
        combined_mask: float tensor of shape (batch_size, 1, seq_len, seq_len)
    """
    # Padding mask for the target: (batch, 1, 1, seq_len)
    padding_mask = create_padding_mask(target)

    # Look-ahead mask: (seq_len, seq_len)
    seq_len = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)

    # Broadcast-combine: take element-wise maximum so that a position is
    # blocked if EITHER mask says to block it.
    # After tf.maximum: shape (batch, 1, seq_len, seq_len)
    combined_mask = tf.maximum(padding_mask, look_ahead_mask)
    return combined_mask


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    # --- padding mask ---
    sample_seq = tf.constant([[7, 3, 0, 0], [1, 2, 3, 0]])  # 0 = PAD
    pad_mask = create_padding_mask(sample_seq)
    print("Padding mask shape:", pad_mask.shape)   # (2, 1, 1, 4)
    print(pad_mask[0])                              # [[0, 0, 1, 1]]

    # --- look-ahead mask ---
    lah_mask = create_look_ahead_mask(4)
    print("\nLook-ahead mask (4x4):")
    print(lah_mask.numpy())

    # --- combined mask ---
    sample_target = tf.constant([[5, 1, 0, 0], [3, 2, 1, 0]])
    comb_mask = create_combined_mask(sample_target)
    print("\nCombined mask shape:", comb_mask.shape)  # (2, 1, 4, 4)
    print("Batch 0:\n", comb_mask[0, 0].numpy())