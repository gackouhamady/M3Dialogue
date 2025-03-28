# MatchingAttention: Mathematical Setup

This module implements a matching attention mechanism, which computes attention weights between a candidate vector and a memory matrix, then returns an attention-pooled representation.

## Notation

Let ![Equation1](https://link-to-your-image.png) be the memory matrix with sequence length \( L \) and memory dimension \( d_m \).

Let ![Equation2](https://link-to-your-image.png) be the candidate vector with dimension \( d_c \).

Let ![Equation3](https://link-to-your-image.png) be an optional binary mask for the memory positions.
