"""SSA — Sub-quadratic Selective Attention.

A linearly-scaling attention mechanism using codebook-based content-dependent
routing to replace the O(n^2) dense Q·K^T computation in standard transformers.

Modules:
    CodebookRouter: Content-dependent key selection (Stages 1-2)
    sparse_exact_attention: Exact attention over pre-selected keys (Stage 3)
    SSAAttention: Full drop-in attention replacement
    dense_attention: Standard attention for comparison/testing
"""

from .router import CodebookRouter
from .attention import sparse_exact_attention, dense_attention
from .ssa_layer import SSAAttention, ToyTransformerBlock
from .utils import apply_rope, precompute_rope_freqs, build_causal_mask, expand_kv_for_gqa

__all__ = [
    "CodebookRouter",
    "sparse_exact_attention",
    "dense_attention",
    "SSAAttention",
    "ToyTransformerBlock",
    "apply_rope",
    "precompute_rope_freqs",
    "build_causal_mask",
    "expand_kv_for_gqa",
]
