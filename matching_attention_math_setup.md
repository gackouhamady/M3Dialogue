
# MatchingAttention: Mathematical Setup

This module implements a matching attention mechanism, which computes attention weights between a candidate vector and a memory matrix, then returns an attention-pooled representation.

## Notation

Let  $$ \( M \in \mathbb{R}^{L 	imes d_m} \)$$ be the memory matrix with sequence length \( L \) and memory dimension \( d_m \).

Let \( x \in \mathbb{R}^{d_c} \) be the candidate vector with dimension \( d_c \).

Let \( 	ext{mask} \in \{0,1\}^L \) be an optional binary mask for the memory positions.

## Architecture

### Linear Transformation:
First, the candidate vector \( x \) is transformed to match the memory dimension:

\[
x' = W x + b
\]

where \( W \in \mathbb{R}^{d_m 	imes d_c} \), \( b \in \mathbb{R}^{d_m} \).

This is implemented by `self.transform = nn.Linear(cand_dim, mem_dim, bias=True)`.

### Attention Computation:
The attention scores \( lpha \) are computed as:

\[
lpha' = 	anh(x'^	op M) \odot 	ext{mask}
\]

\[
lpha = 	ext{softmax}(lpha', 	ext{dim}=1)
\]

where \( \odot \) denotes element-wise multiplication and the mask is broadcast appropriately.

### Normalization:
The attention weights are normalized to sum to 1 over the valid (unmasked) positions:

\[
lpha_{	ext{norm}} = rac{lpha \odot 	ext{mask}}{\sum (lpha \odot 	ext{mask})}
\]

### Attention Pooling:
The final attended representation is computed as:

\[
h = \sum_{i=1}^{L} lpha_i M_i
\]

This is implemented efficiently as a batched matrix multiplication.

## Forward Pass

The forward pass implements the following computation:

```python
# Input: M (memory), x (candidate), mask (optional)

1. M_ = M.permute(1, 2, 0)  # Rearrange memory for matmul
2. x_ = transform(x).unsqueeze(1)  # Project candidate
3. Compute attention scores: alpha_ = tanh(bmm(x_, M_)) * mask
4. Compute softmax: alpha_ = softmax(alpha_, dim=2)
5. Normalize over valid positions: alpha = alpha_masked / alpha_sum
6. Compute attention pool: attn_pool = bmm(alpha, M.transpose(0, 1))

# Output: attn_pool (attended vector), alpha (attention weights)
```

## Key Properties

- The attention mechanism is "matching" style - it compares the candidate directly against memory entries.
- Mask support allows for variable-length sequences.
- The tanh activation provides non-linearity in attention computation.
- Normalization ensures attention weights sum to 1 over valid positions.

This implementation is commonly used in dialogue systems and other tasks where you need to match a current state against a memory of previous states.

This explanation provides both the mathematical formulation and the connection to the implementation details in the code. You can adjust the level of detail or notation style to match your project's conventions.
