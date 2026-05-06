"""Codebook-based content-dependent routing for SSA.

Stage 1-2 of the SSA pipeline:
  1. Project Q and K to routing space, score against learnable codebook
  2. Select top-k candidate keys per query via code-level inverted index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CodebookRouter(nn.Module):
    """Content-dependent router that selects k candidate keys per query using
    a learned codebook for sub-quadratic candidate generation.

    Architecture:
        Q_full → W_qr → Q_route ─┐
                                   ├→ · C^T → code scores → top-a codes → inverted index
        K_full → W_kr → K_route ─┘  · C^T → code scores → top-b assignment ─┘
                                                              ↓
                              candidates (keys sharing any selected code)
                                                              ↓
                              re-score with full Q,K → top-k per query
    """

    def __init__(
        self,
        head_dim: int = 128,
        route_dim: int = 32,
        num_kv_heads: int = 8,
        num_codebook: int = 2048,
        codes_per_key: int = 4,
        codes_per_query: int = 16,
        top_k: int = 256,
        temperature: float = 1.0,
    ):
        """
        Args:
            head_dim: Full head dimension (d)
            route_dim: Routing projection dimension (d_s)
            num_kv_heads: Number of KV heads (routing is done per KV head)
            num_codebook: Number of codebook entries (N_c)
            codes_per_key: Number of codes each key is assigned to (b)
            codes_per_query: Number of top codes selected per query (a)
            top_k: Number of final candidates per query (k)
            temperature: Temperature for Gumbel-Softmax (lower = harder)
        """
        super().__init__()
        self.head_dim = head_dim
        self.route_dim = route_dim
        self.num_kv_heads = num_kv_heads
        self.num_codebook = num_codebook
        self.codes_per_key = codes_per_key
        self.codes_per_query = codes_per_query
        self.top_k = top_k
        self.temperature = temperature

        # Routing projections
        self.W_qr = nn.Linear(head_dim, route_dim, bias=False)
        self.W_kr = nn.Linear(head_dim, route_dim, bias=False)

        # Learnable codebook: [num_kv_heads, num_codebook, route_dim]
        # Per-head codebooks allow different routing geometries per head
        self.codebook = nn.Parameter(
            torch.randn(num_kv_heads, num_codebook, route_dim) / route_dim**0.5
        )

        # Optional: full re-scoring projection (reuse W_qr/W_kr or use main Q/K)
        # For prototype: use input Q_full and K_full directly for re-scoring

    def forward(
        self,
        q_full: torch.Tensor,
        k_full: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route queries to candidate key positions.

        Note: RoPE should be applied to q_full and k_full externally before
        calling this method, so that re-scoring scores incorporate position.

        Args:
            q_full: [seq_len, num_q_heads, head_dim] — with RoPE applied
            k_full: [seq_len, num_kv_heads, head_dim] — with RoPE applied
            causal_mask: Optional [seq_len, seq_len] bool mask. True where attention
                        is allowed. If None, no masking.
            hard: If True, use hard code assignment (argmax). If False, use Gumbel-Softmax.

        Returns:
            indices: [seq_len, num_kv_heads, top_k] candidate key indices
            scores: [seq_len, num_kv_heads, top_k] attention scores for candidates
        """
        seq_len = q_full.shape[0]
        device = q_full.device

        # --- Stage 1: Project to routing space ---
        # Routing projections use raw Q/K (before RoPE) for content-based routing.
        # The re-scoring step (Stage 2) uses full Q/K with RoPE.
        q_route = self.W_qr(q_full)  # [seq_len, num_q_heads, route_dim]
        k_route = self.W_kr(k_full)  # [seq_len, num_kv_heads, route_dim]

        # Group query heads to KV head count (GQA)
        num_q_heads = q_route.shape[1]
        group_size = num_q_heads // self.num_kv_heads
        assert num_q_heads % self.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

        # Mean-pool query routes per KV group: [seq_len, num_kv_heads, route_dim]
        q_route_grouped = q_route.view(seq_len, self.num_kv_heads, group_size, self.route_dim)
        q_route_grouped = q_route_grouped.mean(dim=2)

        # --- Stage 1b: Score against codebook ---
        # Normalize for cosine-like scoring
        q_route_norm = F.normalize(q_route_grouped, dim=-1)
        k_route_norm = F.normalize(k_route, dim=-1)
        codebook_norm = F.normalize(self.codebook, dim=-1)  # [h_kv, N_c, d_s]

        # Q-code scores: [seq_len, num_kv_heads, N_c]
        q_code_scores = torch.einsum("shd,hcd->shc", q_route_norm, codebook_norm)
        q_code_scores = q_code_scores / self.temperature

        # K-code scores: [seq_len, num_kv_heads, N_c]
        k_code_scores = torch.einsum("shd,hcd->shc", k_route_norm, codebook_norm)
        k_code_scores = k_code_scores / self.temperature

        # --- Stage 1c: Key-to-code assignment ---
        if hard:
            _, k_code_assign = k_code_scores.topk(self.codes_per_key, dim=-1)
            # [seq_len, num_kv_heads, codes_per_key]
        else:
            k_code_assign = F.gumbel_softmax(
                k_code_scores.view(-1, self.num_codebook),
                tau=self.temperature,
                hard=True,
            ).view(seq_len, self.num_kv_heads, self.num_codebook)

        # --- Stage 2: Candidate selection ---
        # For each query, select top-a codes
        _, top_q_codes = q_code_scores.topk(self.codes_per_query, dim=-1)
        # [seq_len, num_kv_heads, a]

        # Build inverted index: for each (head, code), collect key positions
        indices = self._select_candidates(
            q_full, k_full, q_code_scores, k_code_assign, top_q_codes, causal_mask
        )

        return indices

    def _select_candidates(
        self,
        q_full: torch.Tensor,
        k_full: torch.Tensor,
        q_code_scores: torch.Tensor,
        k_code_assign: torch.Tensor,
        top_q_codes: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Select top-k candidates using codebook index matching + full-dimensional re-scoring.

        For each query position i and head h:
            1. Query selected codes → find keys assigned to those codes via inverted index
            2. Re-score candidates with full Q/K dot products
            3. Take top-k

        Returns:
            indices: [seq_len, num_kv_heads, top_k] candidate key indices
        """
        seq_len = q_full.shape[0]
        num_kv_heads = self.num_kv_heads
        device = q_full.device

        # Group Q heads to KV head groups for re-scoring
        num_q_heads = q_full.shape[1]
        group_size = num_q_heads // num_kv_heads
        q_full_grouped = q_full.view(seq_len, num_kv_heads, group_size, self.head_dim)

        # Compute full attention scores for candidate re-scoring
        full_scores = torch.einsum(
            "shgd,thd->shgt", q_full_grouped, k_full
        ) / (self.head_dim**0.5)
        # full_scores: [seq_len, num_kv_heads, group_size, seq_len]

        if causal_mask is not None:
            full_scores = full_scores.masked_fill(
                ~causal_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Aggregate across group heads for ranking
        full_scores_routing = full_scores.mean(dim=2)  # [seq_len, num_kv_heads, seq_len]

        # Build candidate mask using code matching
        candidate_mask = self._build_candidate_mask(
            top_q_codes, k_code_assign, seq_len, num_kv_heads, device
        )
        # candidate_mask: [seq_len, num_kv_heads, seq_len]

        # Apply candidate mask to scores
        full_scores_routing = full_scores_routing.masked_fill(~candidate_mask, float("-inf"))

        # Select top-k per query (clamp to seq_len for short sequences)
        effective_k = min(self.top_k, seq_len)
        _, indices = full_scores_routing.topk(effective_k, dim=-1)
        # Pad to top_k size if needed (for shape consistency)
        if effective_k < self.top_k:
            pad = torch.zeros(
                seq_len, num_kv_heads, self.top_k - effective_k,
                dtype=indices.dtype, device=device
            )
            indices = torch.cat([indices, pad], dim=-1)
        # indices: [seq_len, num_kv_heads, top_k]

        return indices

    def _build_candidate_mask(
        self,
        top_q_codes: torch.Tensor,
        k_code_assign: torch.Tensor,
        seq_len: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build boolean mask indicating which keys are candidates for each query.

        For each query at position i and head h, a key at position j is a
        candidate if they share at least one code assignment.

        Args:
            top_q_codes: [seq_len, num_kv_heads, a] — codes selected by each query
            k_code_assign: [seq_len, num_kv_heads, b] — codes assigned to each key

        Returns:
            mask: [seq_len, num_kv_heads, seq_len] True where key is candidate
        """
        n = seq_len
        h = num_kv_heads
        a = self.codes_per_query
        b = self.codes_per_key

        # Query codes: [n_q, 1, h, a, 1] — expand for cross-product across positions
        q_codes_exp = top_q_codes.view(n, 1, h, a, 1)  # [n_q, 1, h, a, 1]

        # Key codes: [1, n_k, h, 1, b] — expand for cross-product across positions
        k_codes_exp = k_code_assign.view(1, n, h, 1, b)  # [1, n_k, h, 1, b]

        # Compare: [n_q, n_k, h, a, b] — element-wise code equality
        code_match = (q_codes_exp == k_codes_exp)  # [n_q, n_k, h, a, b]

        # Any code overlap between query i and key j? → [n_q, n_k, h]
        any_match = code_match.any(dim=-1).any(dim=-1)  # [n_q, n_k, h]

        # Transpose to canonical [n, h, n] format
        mask = any_match.permute(0, 2, 1)  # [n_q, h, n_k]

        return mask

    def get_code_assignment(self, k_full: torch.Tensor) -> torch.Tensor:
        """Get code assignments for keys (useful for KV cache management).

        Args:
            k_full: [seq_len, num_kv_heads, head_dim]

        Returns:
            assignments: [seq_len, num_kv_heads, codes_per_key] code indices
        """
        k_route = self.W_kr(k_full)
        k_route_norm = F.normalize(k_route, dim=-1)
        codebook_norm = F.normalize(self.codebook, dim=-1)
        k_code_scores = torch.einsum("shd,hcd->shc", k_route_norm, codebook_norm)
        _, assignments = k_code_scores.topk(self.codes_per_key, dim=-1)
        return assignments
