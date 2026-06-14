# Nano Transformer (TensorFlow)

A from-scratch implementation of the Transformer architecture using TensorFlow and Keras.

This project focuses on building core deep learning concepts (attention mechanism, encoder-decoder structure) without relying on prebuilt Transformer APIs.

---

## Features

- Multi-Head Self Attention
- Scaled Dot-Product Attention
- Encoder-Decoder Transformer Architecture
- Positional Encoding (sinusoidal)
- Feedforward Neural Network blocks
- Layer Normalization + Dropout
- Fully modular TensorFlow/Keras implementation

---

## Architecture Overview

The model follows the original Transformer design:

- Encoder stack (self-attention + feedforward)
- Decoder stack (masked self-attention + cross-attention)
- Final linear projection to vocabulary space

---

## How It Works

1. Input tokens are converted into embeddings
2. Positional encoding is added
3. Encoder processes input sequence
4. Decoder generates output using:
   - Masked self-attention
   - Cross-attention with encoder output
5. Final dense layer produces token probabilities

---

## Example Output

```bash
Output shape: (64, 50, 8000)