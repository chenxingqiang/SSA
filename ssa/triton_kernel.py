"""Triton block-sparse attention kernel for SSA.

Replaces the gather-based sparse_exact_attention with a true block-sparse
CUDA kernel. The prototype attention.py gather approach is O(n*k*d) but
suffers from gather overhead and memory indirection on GPU. This Triton
kernel eliminates gather, computes attention directly on selected blocks.

Design:
  - Keys are logically [seq_len, num_kv_heads, head_dim]
  - For each query block (B_q queries) and KV head, the router produces
    a sparse block mask of shape [B_q, num_kv_heads, B_k] where B_k entries
    are selected block indices.
  - The kernel loads Q blocks, loads selected K/V blocks via the mask,
    computes attention, and writes output.

Blocking:
  - Block size: 64 tokens (tunable)
  - head_dim: 64-128 (power of 2 for aligned loads)
  - k blocks per query: top_k / 64 (e.g., 256/64 = 4 blocks)

This kernel requires: triton>=2.0, CUDA GPU with compute capability >= 7.0.
WILL NOT RUN on Mac — this is the production implementation target.
"""

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def ssa_block_sparse_attention_kernel(
        # Inputs (all 1D flattened)
        Q_ptr,
        K_ptr,
        V_ptr,
        block_indices_ptr,
        # Output
        O_ptr,
        # Strides (scalar ints)
        Q_stride_s,
        Q_stride_h,
        K_stride_s,
        K_stride_h,
        # Dimensions (scalar ints)
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        # block_indices stride
        block_stride_h,
        block_stride_nk,
        # Constants
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        N_K_BLOCKS: tl.constexpr,
        SCALE: tl.constexpr,
        GQA_GROUP: tl.constexpr,
    ):
        """Triton kernel for SSA block-sparse attention.

        Grid: (n_blocks_q, num_q_heads)
        One program per (query_block, q_head).
        """
        pid_block = tl.program_id(0)
        pid_qhead = tl.program_id(1)
        pid_h = pid_qhead // GQA_GROUP

        q_start = pid_block * BLOCK_SIZE
        q_offs = q_start + tl.arange(0, BLOCK_SIZE)
        q_mask = q_offs < seq_len

        # Load Q for this head: [BLOCK_SIZE, HEAD_DIM]
        q_base = (q_offs * Q_stride_s) + (pid_qhead * Q_stride_h)
        q_tile = tl.load(
            Q_ptr + q_base[:, None] + tl.arange(0, HEAD_DIM)[None, :],
            mask=q_mask[:, None], other=0.0
        )

        # Load K/V block IDs for this (block, KV_head)
        block_idx_base = (pid_block * num_kv_heads + pid_h) * block_stride_h

        # Online softmax accumulators
        o_tile = tl.zeros([BLOCK_SIZE, HEAD_DIM], dtype=tl.float32)
        m_prev = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
        l_prev = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for bk in range(N_K_BLOCKS):
            # Load one block index at offset
            bk_offset = block_idx_base + bk * block_stride_nk
            k_block = tl.load(block_indices_ptr + bk_offset)

            k_start = k_block * BLOCK_SIZE
            k_offs = k_start + tl.arange(0, BLOCK_SIZE)
            k_mask = k_offs < seq_len

            # Load K: [BLOCK_SIZE, HEAD_DIM]
            k_base = (k_offs * K_stride_s) + (pid_h * K_stride_h)
            k_tile = tl.load(
                K_ptr + k_base[:, None] + tl.arange(0, HEAD_DIM)[None, :],
                mask=k_mask[:, None], other=0.0
            )

            # Load V: [BLOCK_SIZE, HEAD_DIM]
            v_tile = tl.load(
                V_ptr + k_base[:, None] + tl.arange(0, HEAD_DIM)[None, :],
                mask=k_mask[:, None], other=0.0
            )

            # Compute attention
            scores = tl.dot(q_tile, tl.trans(k_tile)) * SCALE
            causal = q_offs[:, None] >= (k_offs[None, :])
            scores = tl.where(causal & q_mask[:, None] & k_mask[None, :],
                            scores, float("-inf"))

            # Online softmax
            m_curr = tl.max(scores, axis=1)
            m_new = tl.maximum(m_prev, m_curr)
            alpha = tl.exp(m_prev - m_new)
            beta = tl.exp(m_curr - m_new)
            o_tile = o_tile * alpha[:, None]
            p = tl.exp(scores - m_new[:, None])
            pv = tl.dot(p.to(tl.float16), v_tile)
            o_tile = o_tile + pv * beta[:, None]
            l_prev = l_prev * alpha + tl.sum(p, axis=1)
            m_prev = m_new

        # Write output
        o_base = (q_offs * Q_stride_s) + (pid_qhead * Q_stride_h)
        tl.store(
            O_ptr + o_base[:, None] + tl.arange(0, HEAD_DIM)[None, :],
            o_tile.to(Q_ptr.dtype.element_ty),
            mask=q_mask[:, None]
        )


    def ssa_triton_attention(
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        block_indices: "torch.Tensor",
        block_size: int = 64,
        scale: float = None,
    ) -> "torch.Tensor":
        """Triton-based block-sparse attention for SSA.

        Args:
            q: [seq_len, num_q_heads, head_dim] (with RoPE)
            k: [seq_len, num_kv_heads, head_dim] (with RoPE)
            v: [seq_len, num_kv_heads, head_dim]
            block_indices: [ceil(seq_len/block_size), num_kv_heads, n_k_blocks]
                          Each entry is a block index (0-based) into K/V.
            block_size: Tokens per block (64 or 128, power of 2)
            scale: Attention scale. Default: 1/sqrt(head_dim).

        Returns:
            output: [seq_len, num_q_heads, head_dim]

        Note: This function requires CUDA and will not run on Mac/MPS.
        """
        import torch

        seq_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]
        n_k_blocks = block_indices.shape[-1]
        group_size = num_q_heads // num_kv_heads

        if scale is None:
            scale = 1.0 / (head_dim**0.5)

        # Pad sequence to multiple of block_size
        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
            k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))
            seq_len_padded = seq_len + pad_len
        else:
            seq_len_padded = seq_len

        n_blocks_q = seq_len_padded // block_size
        Q_stride_s = num_q_heads * head_dim
        Q_stride_h = head_dim
        K_stride_s = num_kv_heads * head_dim
        K_stride_h = head_dim
        # block_indices strides: layout [n_blocks_q, num_kv_heads, n_k_blocks]
        block_stride_h = n_k_blocks       # stride between KV heads
        block_stride_nk = 1               # stride within n_k_blocks dimension

        # Flatten tensors to 1D
        q_flat = q.reshape(-1).contiguous()
        k_flat = k.reshape(-1).contiguous()
        v_flat = v.reshape(-1).contiguous()
        bi_flat = block_indices.reshape(-1).contiguous()

        # Allocate output
        O = torch.empty(seq_len_padded, num_q_heads, head_dim,
                       device=q.device, dtype=q.dtype)
        O_flat = O.reshape(-1)

        # Grid: one program per (query_block, q_head)
        grid = (n_blocks_q, num_q_heads)
        ssa_block_sparse_attention_kernel[grid](
            q_flat, k_flat, v_flat, bi_flat, O_flat,
            Q_stride_s, Q_stride_h, K_stride_s, K_stride_h,
            seq_len, num_q_heads, num_kv_heads, head_dim,
            block_stride_h, block_stride_nk,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            N_K_BLOCKS=n_k_blocks,
            SCALE=scale,
            GQA_GROUP=group_size,
        )

        # Trim padding
        if pad_len > 0:
            O = O[:seq_len]

        return O


def convert_indices_to_blocks(
    indices: "torch.Tensor",
    block_size: int,
) -> "torch.Tensor":
    """Convert per-token candidate indices to per-block candidate block indices.

    The router produces per-query candidates: indices [seq_len, num_kv_heads, top_k].
    The Triton kernel operates on blocks of BLOCK_SIZE tokens.

    This function aggregates per-token indices into per-block block indices:
      1. Group queries into blocks of size BLOCK_SIZE
      2. For each query block, collect the union of key blocks referenced by
         any query in that block
      3. Select the top n_k_blocks (most frequently referenced)

    Args:
        indices: [seq_len, num_kv_heads, top_k] token-level candidate indices
        block_size: Tokens per block

    Returns:
        block_indices: [ceil(seq_len/block_size), num_kv_heads, n_k_blocks]
                       Block-level indices into K/V blocks
    """
    import torch

    seq_len, num_kv_heads, top_k = indices.shape
    n_blocks_q = (seq_len + block_size - 1) // block_size
    n_blocks_k = (seq_len + block_size - 1) // block_size

    # Convert token indices to block indices
    token_block_ids = indices // block_size  # [seq_len, num_kv_heads, top_k]

    # Pad sequence to block boundary
    pad_len = n_blocks_q * block_size - seq_len
    if pad_len > 0:
        token_block_ids = torch.nn.functional.pad(
            token_block_ids, (0, 0, 0, 0, 0, pad_len), value=0
        )
        seq_len_padded = seq_len + pad_len
    else:
        seq_len_padded = seq_len

    # Reshape: group queries into blocks
    # token_block_ids: [seq_len_padded, num_kv_heads, top_k]
    # → [n_blocks_q, block_size, num_kv_heads, top_k]
    block_queries = token_block_ids.reshape(
        n_blocks_q, block_size, num_kv_heads, top_k
    )

    # For each query block, count references to each key block
    # block_count: [n_blocks_q, num_kv_heads, n_blocks_k]
    block_count = torch.zeros(
        n_blocks_q, num_kv_heads, n_blocks_k,
        dtype=torch.long, device=indices.device
    )

    # Scatter counts
    for bq in range(n_blocks_q):
        for h in range(num_kv_heads):
            blocks_for_queries = block_queries[bq, :, h, :]  # [block_size, top_k]
            unique_blocks, counts = blocks_for_queries.unique(return_counts=True)
            block_count[bq, h, unique_blocks.long()] = counts.long()

    # Select top n_k_blocks per query block
    n_k_blocks = min(top_k, n_blocks_k)
    _, top_blocks = block_count.topk(n_k_blocks, dim=-1)
    # top_blocks: [n_blocks_q, num_kv_heads, n_k_blocks]

    return top_blocks


# Fallback when Triton is not available (e.g., on Mac)
if not HAS_TRITON:

    def ssa_triton_attention(*args, **kwargs):
        raise RuntimeError(
            "Triton kernel requires CUDA GPU and triton>=2.0. "
            "Use ssa.attention.sparse_exact_attention for CPU/MPS fallback."
        )

    def convert_indices_to_blocks(*args, **kwargs):
        raise RuntimeError(
            "Triton utilities require CUDA GPU and triton>=2.0."
        )
