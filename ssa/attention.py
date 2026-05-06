"""Sparse exact attention computation (Stage 3 of SSA).

Computes softmax attention over a pre-selected set of key positions.
Prototype uses gather-based approach; production would use block-sparse CUDA kernels.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def sparse_exact_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute exact scaled dot-product attention over pre-selected key positions.

    For each query position i and KV head h, computes:
        attn_i = softmax(Q_i · K_{indices[i,h]}^T / sqrt(d)) · V_{indices[i,h]}

    Args:
        q: [seq_len, num_q_heads, head_dim] — with RoPE applied
        k: [seq_len, num_kv_heads, head_dim] — with RoPE applied
        v: [seq_len, num_kv_heads, head_dim]
        indices: [seq_len, num_kv_heads, top_k] — pre-selected key indices

    Returns:
        output: [seq_len, num_q_heads, head_dim]
    """
    seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    top_k = indices.shape[-1]

    if scale is None:
        scale = head_dim**0.5

    group_size = num_q_heads // num_kv_heads
    assert num_q_heads % num_kv_heads == 0

    # Gather K and V at selected indices for each KV head
    # indices: [seq_len, num_kv_heads, top_k] — per-query-position indices
    #
    # For each query position i and KV head h, we need:
    #   K at positions indices[i, h, :] → [head_dim, top_k]
    # Use index_select: for each head, gather all unique positions
    # Then reshape to per-query form.
    #
    # Strategy: flatten queries, do batched gather, then reshape back.

    # Step 1: For each KV head, gather K and V at the indices for each query position
    # We use torch.gather on the seq_len dimension of K and V

    # K: [seq_len, num_kv_heads, head_dim]
    # indices: [seq_len, num_kv_heads, top_k]
    # For each (i, h), we want K[indices[i,h,:], h, :]
    # Approach: gather along dim=0 (seq_len) of K:
    #   Expand K to [seq_len, num_kv_heads, 1, head_dim]
    #   Indices: [seq_len, num_kv_heads, top_k, 1] (expanded)
    #   Gather along dim=0 → [seq_len, num_kv_heads, top_k, head_dim]

    K_exp = k.unsqueeze(2)  # [seq_len, num_kv_heads, 1, head_dim]
    # Expand K to be indexable by the indices for each query position
    # Use index_select + reshape instead of gather for clarity

    # index_select on dim=0: for each head, select keys at all referenced positions
    # indices: [seq_len, num_kv_heads, top_k] — convert to flat index
    flat_indices = indices.reshape(-1)  # [seq_len * num_kv_heads * top_k]
    # Repeat for each head_dim entry
    flat_indices_k = flat_indices.unsqueeze(-1).expand(-1, head_dim)  # [N, head_dim]

    # For head-aware gathering, we need to handle the head dimension
    # Reshape K to [seq_len * num_kv_heads, head_dim] and add head offsets
    K_flat = k.reshape(seq_len * num_kv_heads, head_dim)
    V_flat = v.reshape(seq_len * num_kv_heads, head_dim)

    # For each (query_i, head_h), the indices need to be offset by head_h * seq_len
    head_offsets = (
        torch.arange(num_kv_heads, device=k.device) * seq_len
    )  # [num_kv_heads]
    # Expand to match flat_indices: [seq_len * num_kv_heads * top_k]
    head_offsets_expanded = (
        head_offsets.unsqueeze(0).unsqueeze(-1)
        .expand(seq_len, -1, top_k)
        .reshape(-1)
    )

    # Adjusted flat indices into the [seq_len * num_kv_heads, head_dim] space
    gather_indices = flat_indices + head_offsets_expanded  # [seq_len * num_kv_heads * top_k]

    # Gather: expand gather_indices for head_dim
    gather_indices_k = gather_indices.unsqueeze(-1).expand(-1, head_dim)
    K_gathered = K_flat.gather(dim=0, index=gather_indices_k)  # [N, head_dim]
    V_gathered = V_flat.gather(dim=0, index=gather_indices_k)  # [N, head_dim]

    # Reshape back: [seq_len, num_kv_heads, top_k, head_dim]
    K_gathered = K_gathered.reshape(seq_len, num_kv_heads, top_k, head_dim)
    V_gathered = V_gathered.reshape(seq_len, num_kv_heads, top_k, head_dim)

    # Step 2: Compute Q·K^T for selected keys
    # Q grouped by GQA: [seq_len, num_kv_heads, group_size, head_dim]
    q_grouped = q.reshape(seq_len, num_kv_heads, group_size, head_dim)

    # Scaled dot product: [seq_len, num_kv_heads, group_size, top_k]
    attn_logits = torch.einsum("nhgd,nhkd->nhgk", q_grouped, K_gathered) / scale

    attn_weights = F.softmax(attn_logits, dim=-1)  # [seq_len, num_kv_heads, g, top_k]

    # Step 3: Weighted sum of values
    output = torch.einsum("nhgk,nhkd->nhgd", attn_weights, V_gathered)
    output = output.reshape(seq_len, num_q_heads, head_dim)

    return output


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Standard dense scaled dot-product attention (reference implementation).

    Args:
        q: [seq_len, num_q_heads, head_dim] — with RoPE applied
        k: [seq_len, num_kv_heads, head_dim] — with RoPE applied
        v: [seq_len, num_kv_heads, head_dim]
        causal_mask: Optional [seq_len, seq_len] bool mask
        scale: Optional scale factor

    Returns:
        output: [seq_len, num_q_heads, head_dim]
    """
    seq_len, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape

    if scale is None:
        scale = head_dim**0.5

    group_size = num_q_heads // num_kv_heads

    # Expand KV for GQA
    k_expanded = k.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(seq_len, num_q_heads, head_dim)
    v_expanded = v.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(seq_len, num_q_heads, head_dim)

    # Standard attention
    attn_logits = torch.einsum("shd,thd->sht", q, k_expanded) / scale

    if causal_mask is not None:
        attn_logits = attn_logits.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))

    attn_weights = F.softmax(attn_logits, dim=-1)
    output = torch.einsum("sht,thd->shd", attn_weights, v_expanded)

    return output
