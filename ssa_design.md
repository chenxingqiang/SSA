# SSA — Subquadratic Selective Attention: Architecture Specification

## Overview

SSA replaces the O(n^2) dense Q·K^T computation in standard attention with a three-stage
codebook-routed sparse pipeline. The total cost per head scales linearly with sequence length.

### Notation

| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| n | Sequence length | 1K–1M |
| h_q | Number of query heads | 40 |
| h_kv | Number of KV heads (GQA) | 8 |
| g | GQA group size (h_q / h_kv) | 5 |
| d | Full head dimension | 128 |
| d_s | Small (routing) dimension | 32 |
| N_c | Codebook size | 2048 |
| b | Keys assigned per code | 4 |
| a | Codes selected per query | 16 |
| k | Final candidates per query | 256 (prototype), 16K (production) |

---

## Stage 1: Codebook Routing

### Objective

For each query at position i, produce a candidate set S_i of approximately k key positions
that are likely relevant, without computing the full n×n attention matrix.

### Tensor Flow

```
Input:  Q_full ∈ R^{n × h_q × d}       # full query heads
        K_full ∈ R^{n × h_kv × d}      # full key heads
        C ∈ R^{N_c × d_s}              # learnable codebook
        W_qr ∈ R^{d × d_s}             # query routing projection
        W_kr ∈ R^{d × d_s}             # key routing projection

Step 1a: Project to routing space
  Q_route = Q_full · W_qr   → R^{n × h_q × d_s}
  K_route = K_full · W_kr   → R^{n × h_kv × d_s}

Step 1b: Apply RoPE (position encoding) to Q_route and K_route
  (Consistent with main attention to preserve distance relationships)

Step 1c: Query-code scores
  Q_codes = Q_route · C^T  → R^{n × h_q × N_c}

Step 1d: Key-code scores + assignment
  K_codes = K_route · C^T  → R^{n × h_kv × N_c}
  K_assign = top_b_indices(K_codes, dim=-1)  → R^{n × h_kv × b}  (b code indices per key)

### GQA routing optimization:
  Since multiple Q heads share each KV head, routing is done per KV head:
    Q_codes = mean(Q_codes_grouped, dim=head)  → R^{n × h_kv × N_c}
  or equivalently, route each Q head independently but share the same K_assign.
```

### Key Design Decisions

1. **Separate routing projections**: W_qr and W_kr are distinct from the main attention
   projections W_q and W_k. This allows the router to specialize on discriminative features
   while the main attention head focuses on representation.

2. **Shared codebook across heads**: A single codebook C is shared across all KV heads.
   Each head uses the same routing geometry but makes independent code selections.

3. **Multi-hot key assignment (b > 1)**: Each key is assigned to b codes, not just 1.
   This improves recall — a key that sits at the boundary between two codebook centroids
   can be reached through either. b=4 is a good default.

---

## Stage 2: Candidate Selection

### Objective

Convert per-query code preferences (Q_codes) and per-key code assignments (K_assign)
into a fixed-size candidate index set S_i for each query.

### Algorithm

```
For each query position i and KV head h:

1. Select top-a codes:
     top_codes = argsort(Q_codes[i, h], descending=True)[:a]   → [a]

2. Collect all keys assigned to any of those codes:
     candidates = {j | K_assign[j, h] ∩ top_codes ≠ ∅}

3. Re-score candidates using full-dimensional Q and K:
     scores = Q_full[i] · K_full[candidates]^T / sqrt(d)      → [|candidates|]

4. Select top-k:
     S_{i,h} = argsort(scores, descending=True)[:k]            → [k]
```

### Complexity

- Step 1-2: O(n · N_c · d_s) for the scoring, O(n · b) for the index lookup.
  Note: Step 2 is implemented via scatter/gather on the code-index inverted list,
  not iteration.

- Step 3: O(n · avg_candidates · d) where avg_candidates ≈ n · (a · b / N_c).
  With a=16, b=4, N_c=2048: avg_candidates ≈ n · 64/2048 ≈ n/32.
  This is the dominant term and grows with n.

- To maintain fixed k, we need to cap |candidates|. If too many keys map to the
  selected codes, use the code scores (Q_codes) to pre-filter before step 3.

### Causal Masking

For autoregressive models, only keys at position j ≤ i are eligible:
```
candidates = {j | j ≤ i AND K_assign[j, h] ∩ top_codes ≠ ∅}
```
This is applied as a mask on the inverted index lookup.

---

## Stage 3: Sparse Attention

### Objective

Compute exact scaled dot-product attention over the selected keys only.

### Computation

```
For each query i, KV head h:
  candidates = S_{i,h}                                     # pre-selected indices [k]
  attn_logits = Q_full[i, group_heads] · K_full[candidates]^T / sqrt(d)   # [g, k]
  attn_weights = softmax(attn_logits, dim=-1)              # [g, k]
  output[i, group_heads] = attn_weights · V_full[candidates]               # [g, d]
```

### Implementation Strategy (Prototype)

In the pure PyTorch prototype, we implement this via:

```
For each query i:
  gather K[candidates[i]] → [k, d]
  gather V[candidates[i]] → [k, d]
  compute Q[i] · K_gathered^T → softmax → · V_gathered
```

This is a `for` loop over queries — acceptable for correctness testing at small n.
For production, a block-sparse CUDA kernel (e.g., Triton `tl.dot` with block masks)
replaces the gather + dense matmul with true sparsity.

---

## Training Considerations

### Gumbel-Softmax for Differentiable Code Assignment

During training, code assignments are made differentiable:

```
K_codes = K_route · C^T
K_assign_soft = gumbel_softmax(K_codes, temperature=τ, hard=True) → nearly one-hot
# When τ → 0, approaches hard assignment
# When τ → 1, softens for gradient flow
```

Annealing schedule: τ starts at 1.0, decays to 0.1 over 10K steps.

### Auxiliary Routing Loss

To prevent the router from collapsing to a lazy policy (always routing to nearby keys),
an auxiliary loss encourages the routing scores to match the true attention distribution:

```
L_aux = KL(routing_scores || true_attention)
```

This is computed periodically (not every step) using full dense attention as the teacher.
The routing scores are the Q_codes similarity, mapped onto keys through code assignments.

### RL for Long-Context Retrieval

Following the SSA paper's three-stage training (pre-train → SFT → RL), the RL stage
targets retrieval failures:

- Reward: +1 for correct retrieval from distant evidence, +0 otherwise
- Penalty: -0.1 for answering from local context when distant evidence would yield a
  different (better) answer

This requires a dataset of long-context prompts where the correct answer depends on
evidence distributed across the context, not concentrated near the query.

---

## Qwen 3.6 Integration

### Attention Block Replacement

Qwen's transformer block uses grouped query attention. The replacement point is the
attention computation within each block:

```python
# Original (simplified):
attn_output = scaled_dot_product_attention(Q, K, V, is_causal=True)

# SSA replacement:
router = CodebookRouter(d, d_s, N_c, b, a, k)
candidates = router(Q, K, causal_mask=True)     # per-KV-head indices
attn_output = sparse_exact_attention(Q, K, V, candidates, is_causal=True)
```

### Weight Compatibility

The SSA module adds:
- W_qr, W_kr: small routing projections (d × d_s each, per head)
- Codebook C: (N_c × d_s)

These are new parameters not present in the original Qwen checkpoint.
Strategy: initialize these with small random weights, then fine-tune the full model
(or freeze original weights and train only the routing components).

### KV Cache Compatibility

SSA's sparse attention affects KV cache behavior:
- The KV cache still stores all past keys and values
- During prefill, routing is computed over the full sequence
- During decode, routing is incremental: new token's K is assigned to codes, and
  its query selects codes → candidates from the growing cache

For efficient decode, the inverted index (code → list of key positions) is maintained
incrementally, avoiding re-scanning the full KV cache on each step.

---

## Verification Strategy

### Correctness

```
Test: SSA with k=n vs dense attention
Input: random Q, K, V at n=128
Assert: max(|SSA_output - dense_output|) < 1e-5
Reason: When k=n, all keys are selected. Output must match dense exactly.
```

### Routing Quality

```
Test: Oracle comparison
Compute: true_top_k = argsort(full_attention_weights)[:k]
         ssa_candidates = router(Q, K)
Metric: recall = |ssa_candidates ∩ true_top_k| / k
Assert: recall > 0.9  (90% of true top-k keys captured)
```

### Position Independence

```
Test: Needle retrieval at variable positions
Setup: Place a unique key vector at positions 10% and 90% of a random sequence
Metric: routing_recall at each position
Assert: |recall(10%) - recall(90%)| < 0.05
```

### Linear Scaling

```
Test: FLOP count at n = [1024, 2048, 4096, 8192, 16384, 32768]
Fit: FLOPs = a * n + b
Assert: R^2 > 0.99 for linear fit
        FLOPs(2n) / FLOPs(n) ≈ 2.0 ± 0.2
```
