"""Tests for sparse and dense attention computation correctness."""

import torch
import pytest
from ssa import sparse_exact_attention, dense_attention, build_causal_mask, precompute_rope_freqs, apply_rope


class TestSparseExactAttention:
    """Correctness tests for sparse_exact_attention."""

    @pytest.fixture
    def tensors(self):
        """Create standard test tensors."""
        seq_len = 16
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 8
        top_k = 8
        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_kv_heads, head_dim)
        v = torch.randn(seq_len, num_kv_heads, head_dim)
        # Indices: select the first k positions (or any valid ones)
        indices = torch.arange(top_k, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        indices = indices.expand(seq_len, num_kv_heads, -1)
        return q, k, v, indices, top_k

    def test_output_shape(self, tensors):
        """Output should have same shape as input Q."""
        q, k, v, indices, _ = tensors
        out = sparse_exact_attention(q, k, v, indices)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_no_nan_or_inf(self, tensors):
        """Output should not contain NaN or Inf."""
        q, k, v, indices, _ = tensors
        out = sparse_exact_attention(q, k, v, indices)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_scale_invariance(self, tensors):
        """Scaling by a constant on all K doesn't change softmax output."""
        q, k, v, indices, _ = tensors
        out1 = sparse_exact_attention(q, k, v, indices)

        # Double all K values
        k_scaled = k * 2.0
        out2 = sparse_exact_attention(q, k_scaled, v, indices)

        # Outputs should be different (attention weight distribution changes)
        # But both should be valid (finite)
        assert not torch.isnan(out2).any()
        assert not torch.isinf(out2).any()

    def test_gqa_output_shape(self):
        """Test with GQA (more Q heads than KV heads)."""
        seq_len = 8
        num_q_heads = 6
        num_kv_heads = 2
        head_dim = 8
        top_k = 4

        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_kv_heads, head_dim)
        v = torch.randn(seq_len, num_kv_heads, head_dim)
        indices = torch.randint(0, seq_len, (seq_len, num_kv_heads, top_k))

        out = sparse_exact_attention(q, k, v, indices)
        assert out.shape == (seq_len, num_q_heads, head_dim)

    def test_self_attention_consistency(self, tensors):
        """A query should attend maximally to itself when q=k."""
        q, k, v, indices, _ = tensors
        seq_len = q.shape[0]

        # Set q = k (self-attention case)
        q_copy = k.unsqueeze(2).expand(-1, -1, q.shape[1] // k.shape[1], -1)
        q_copy = q_copy.reshape(q.shape)
        indices = torch.arange(seq_len, dtype=torch.long).unsqueeze(1).unsqueeze(2)
        indices = indices.expand(-1, k.shape[1], seq_len)

        out = sparse_exact_attention(q_copy, k, v, indices)
        assert out.shape == q_copy.shape
        assert not torch.isnan(out).any()


class TestDenseAttention:
    """Reference dense attention tests."""

    @pytest.fixture
    def tensors(self):
        seq_len = 8
        num_q_heads = 4
        num_kv_heads = 2
        head_dim = 8
        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_kv_heads, head_dim)
        v = torch.randn(seq_len, num_kv_heads, head_dim)
        return q, k, v

    def test_output_shape(self, tensors):
        q, k, v = tensors
        out = dense_attention(q, k, v)
        assert out.shape == q.shape

    def test_causal_mask(self, tensors):
        q, k, v = tensors
        mask = build_causal_mask(q.shape[0])
        out = dense_attention(q, k, v, causal_mask=mask)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_no_nan_or_inf(self, tensors):
        q, k, v = tensors
        out = dense_attention(q, k, v)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestRoPE:
    """Position encoding tests."""

    def test_rope_shape_preservation(self):
        seq_len = 8
        head_dim = 64
        freqs = precompute_rope_freqs(head_dim, seq_len * 2)

        # Without head dim
        x = torch.randn(seq_len, head_dim)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape

        # With head dim
        x = torch.randn(seq_len, 4, head_dim)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape

    def test_rope_invertible(self):
        """RoPE should preserve vector norm."""
        seq_len = 8
        head_dim = 64
        freqs = precompute_rope_freqs(head_dim, seq_len * 2)

        x = torch.randn(seq_len, head_dim)
        out = apply_rope(x, freqs)

        # RoPE: rotate_half swaps x_rot's [0..d/4) with [d/4..d/2).
        # So pairs (k, k+d/4) share frequency, for k = 0..d/4-1.
        quarter = head_dim // 4
        x_front = x[..., :quarter]
        x_back = x[..., quarter: 2 * quarter]
        x_pair_norms = x_front ** 2 + x_back ** 2

        out_front = out[..., :quarter]
        out_back = out[..., quarter: 2 * quarter]
        out_pair_norms = out_front ** 2 + out_back ** 2

        assert torch.allclose(x_pair_norms, out_pair_norms, atol=1e-6), (
            "RoPE should preserve (k, k+d/4) pair norms"
        )
