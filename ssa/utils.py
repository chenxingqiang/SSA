"""Utility functions for SSA: masking, GQA helpers, validation, RoPE."""

import torch
import torch.nn.functional as F


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor.

    Args:
        x: [..., seq_len, head_dim] — last dim is head_dim, second-last is seq_len.
           Leading dims (batch, heads) are broadcast.
        freqs: [seq_len, rot_dim] — rotation frequencies. rot_dim should be
               >= x.shape[-1]. The first head_dim//2 pairs are used.

    Returns:
        Tensor of same shape as x with RoPE applied.
    """
    seq_len_x = x.shape[-2]
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    # Slice freqs to match: [seq_len_x, half_dim]
    freqs = freqs[:seq_len_x, :half_dim]

    x_rot = x[..., :half_dim]
    x_pass = x[..., half_dim:]

    cos = freqs.cos()
    sin = freqs.sin()

    # Broadcast leading dims: cos/sin are [seq_len, half_dim]
    # x_rot is [..., seq_len, half_dim]
    # Need to insert dims between leading dims and seq_len
    n_leading = x.dim() - 2
    for _ in range(n_leading):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    x_rot_out = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rot_out, x_pass], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (used in RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def precompute_rope_freqs(
    head_dim: int, seq_len: int, theta: float = 10000.0, device=None
) -> torch.Tensor:
    """Precompute RoPE frequency bases.

    Args:
        head_dim: Full head dimension (must be even)
        seq_len: Maximum sequence length
        theta: RoPE base frequency
        device: Target device

    Returns:
        [seq_len, head_dim] frequency tensor (cos/sin pairs interleaved)
    """
    dim = head_dim // 2
    # Standard RoPE: freqs for each pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [seq_len, dim // 2]
    # Duplicate for complex representation (cos/sin pairs)
    freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
    return freqs


def build_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Build a causal attention mask.

    Args:
        seq_len: Sequence length
        device: Target device

    Returns:
        [seq_len, seq_len] boolean mask, True where attention is allowed
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def expand_kv_for_gqa(
    kv: torch.Tensor, num_q_heads: int, num_kv_heads: int
) -> torch.Tensor:
    """Expand KV tensors to match query heads for grouped query attention.

    Args:
        kv: [batch, seq_len, num_kv_heads, head_dim] or [seq_len, num_kv_heads, head_dim]
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads

    Returns:
        [batch, seq_len, num_q_heads, head_dim] with KV repeated per group
    """
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
    group_size = num_q_heads // num_kv_heads
    return kv.unsqueeze(-3).expand(
        *kv.shape[:-2], group_size, kv.shape[-2], kv.shape[-1]
    ).reshape(*kv.shape[:-2], num_q_heads, kv.shape[-1])


def validate_indices(
    indices: torch.Tensor, seq_len: int, max_k: int, name: str = "indices"
) -> None:
    """Validate that candidate indices are within bounds and properly shaped."""
    assert indices.dim() >= 2, f"{name} must have at least 2 dims, got shape {indices.shape}"
    assert indices.shape[-1] <= max_k, (
        f"{name} last dim {indices.shape[-1]} exceeds max_k {max_k}"
    )
    assert indices.max() < seq_len, (
        f"{name} max index {indices.max()} >= seq_len {seq_len}"
    )
    assert indices.min() >= 0, f"{name} min index {indices.min()} < 0"
