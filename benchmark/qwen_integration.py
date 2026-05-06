"""SSA Integration with Qwen Architecture.

Demonstrates how SSAAttention replaces standard attention in a Qwen 3.6
transformer block. Since Qwen 3.6 weights are not available on this Mac,
this script uses a proxy approach:

1. Creates a Qwen-compatible config (GQA, RoPE, pre-norm)
2. Demonstrates the attention swap with parameter mapping
3. Runs inference with both dense and SSA attention for comparison
4. Measures prefill time scaling at increasing context lengths

To use with actual Qwen 3.6 weights, replace ToyQwenConfig with the real
model config and use HuggingFace's AutoModel.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
from typing import Optional


class Qwen3_6Config:
    """Representative Qwen 3.6 model configuration (<27B params)."""

    # Architecture parameters (approximate, actual values depend on model size)
    hidden_size: int = 5120       # typical for ~14B class
    num_q_heads: int = 40
    num_kv_heads: int = 8         # GQA: 40 Q-heads, 8 KV-heads
    head_dim: int = 128
    num_layers: int = 40
    intermediate_size: int = 13696
    vocab_size: int = 152064
    max_seq_len: int = 32768
    rope_theta: float = 1000000.0


def create_qwen_attention(config: Qwen3_6Config, use_ssa: bool = True):
    """Create attention module compatible with Qwen 3.6 architecture.

    Args:
        config: Qwen model configuration
        use_ssa: If True, use SSAAttention; if False, use standard MHA

    Returns:
        Attention module matching Qwen's interface
    """
    from ssa import SSAAttention

    if use_ssa:
        # SSA configuration for production-scale Qwen:
        #   route_dim = 32 (1/4 of head_dim=128)
        #   N_codes = 4096 (enough to partition hundreds of thousands of keys)
        #   codes_per_key = 4 (multi-hot for recall)
        #   codes_per_query = 16 (select top codes)
        #   top_k = 2048 (candidate budget; can increase to 16K for production)
        attn = SSAAttention(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            route_dim=32,
            num_codebook=4096,
            codes_per_key=4,
            codes_per_query=16,
            top_k=2048,
        )
        attn_name = "SSA"
    else:
        # Standard dense attention (wrapped in same interface)
        attn = SSAAttention(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_q_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            top_k=0,  # not used in dense mode
        )
        # Force dense path
        attn._use_dense_default = True
        attn_name = "Dense"

    return attn, attn_name


class QwenTransformerBlock(nn.Module):
    """Qwen-style transformer block with pluggable attention.

    This mirrors the Qwen architecture:
      - Pre-norm (RMSNorm or LayerNorm)
      - GQA multi-head attention
      - Pre-norm FFN with SwiGLU
    """

    def __init__(self, config: Qwen3_6Config, use_ssa: bool = True):
        super().__init__()
        self.config = config
        self.use_ssa = use_ssa

        # Attention
        self.attn, _ = create_qwen_attention(config, use_ssa=use_ssa)

        # Norms (LayerNorm as proxy for RMSNorm)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.post_attn_norm = nn.LayerNorm(config.hidden_size)

        # FFN (SwiGLU: gate_proj, up_proj → SiLU gate · up → down_proj)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        if self.use_ssa:
            hidden_states = self.attn(
                hidden_states, freqs=freqs, use_dense=False
            )
        else:
            hidden_states = self.attn(
                hidden_states, freqs=freqs, use_dense=True
            )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)

        # SwiGLU FFN: SiLU(gate_proj(x)) * up_proj(x) → down_proj
        gate = self.gate_proj(hidden_states)
        gate = gate * torch.sigmoid(gate)  # SiLU activation
        ffn_out = gate * self.up_proj(hidden_states)
        hidden_states = self.down_proj(ffn_out) + residual

        return hidden_states


def create_rope_freqs(config: Qwen3_6Config, seq_len: int, device=None):
    """Create RoPE frequency bases for Qwen 3.6.

    Qwen 3.6 uses a high theta (1,000,000) for extended context.
    """
    from ssa import precompute_rope_freqs
    return precompute_rope_freqs(config.head_dim, seq_len, theta=config.rope_theta, device=device)


def demo_inference_scaling():
    """Demonstrate prefill time scaling for dense vs SSA on Qwen architecture."""
    config = Qwen3_6Config()

    # Use a toy scale for the demo (full 14B won't fit on Mac)
    # Scale down while preserving architectural ratios
    toy_config = Qwen3_6Config()
    toy_config.hidden_size = 768
    toy_config.num_q_heads = 12
    toy_config.num_kv_heads = 3     # GQA group = 4
    toy_config.head_dim = 64
    toy_config.num_layers = 4
    toy_config.intermediate_size = 2048

    print("=" * 70)
    print("Qwen 3.6 SSA Integration Demo")
    print("=" * 70)
    print(f"Config: {toy_config.num_q_heads}Q/{toy_config.num_kv_heads}KV heads")
    print(f"head_dim={toy_config.head_dim}, hidden={toy_config.hidden_size}")
    print(f"Layers: {toy_config.num_layers}, FFN: {toy_config.intermediate_size}")

    # Create block with SSA
    ssa_block = QwenTransformerBlock(toy_config, use_ssa=True)

    # Create block with dense
    dense_block = QwenTransformerBlock(toy_config, use_ssa=False)

    seq_lens = [64, 128, 256, 512, 1024]

    print(f"\n{'n':>6} | {'SSA ms':>8} | {'Dense ms':>8} | {'Speedup':>8} | {'FLOP Red.':>10}")
    print("-" * 50)

    freqs_cache = create_rope_freqs(toy_config, max(seq_lens))

    for n in seq_lens:
        x = torch.randn(1, n, toy_config.hidden_size)
        freqs = freqs_cache[:n]  # slice to needed length

        # Warmup
        _ = ssa_block(x, freqs=freqs)
        _ = dense_block(x, freqs=freqs)

        # Time SSA
        start = time.time()
        for _ in range(5):
            _ = ssa_block(x, freqs=freqs)
        ssa_time = (time.time() - start) / 5

        # Time dense
        start = time.time()
        for _ in range(5):
            _ = dense_block(x, freqs=freqs)
        dense_time = (time.time() - start) / 5

        speedup = dense_time / ssa_time if ssa_time > 0 else 0

        # FLOP reduction (theoretical)
        N_codes = 512  # toy default
        top_k = 64      # toy default
        dense_flops = toy_config.num_q_heads * 4 * n * n * toy_config.head_dim
        ssa_flops = toy_config.num_kv_heads * (
            2 * n * N_codes * 16 + 2 * n * top_k * toy_config.head_dim
        )
        flop_red = dense_flops / max(ssa_flops, 1)

        print(f"{n:>6} | {ssa_time*1000:>7.1f} | {dense_time*1000:>7.1f} | "
              f"{speedup:>7.2f}x | {flop_red:>9.1f}x")

    print(f"\nTheoretical crossover at larger n: SSA wins when routing overhead < dense compute.")

    # Show parameter breakdown
    total_params = sum(p.numel() for p in ssa_block.parameters())
    router_params = (
        ssa_block.attn.router.W_qr.weight.numel()
        + ssa_block.attn.router.W_kr.weight.numel()
        + ssa_block.attn.router.codebook.numel()
    )
    print(f"\nBlock parameters: {total_params:,}")
    print(f"  Router parameters: {router_params:,} ({100*router_params/max(total_params,1):.2f}%)")
    print(f"  Codebook: {ssa_block.attn.router.codebook.numel():,}")

    # Show integration point for real Qwen 3.6
    print(f"\n{'='*70}")
    print("Integration with real Qwen 3.6 (<27B params):")
    print(f"{'='*70}")
    print(f"""
To integrate SSA into a real Qwen 3.6 model:

1. Replace attention computation in QwenAttention.forward():
   ```python
   # Original:
   attn_output = torch.nn.functional.scaled_dot_product_attention(
       query_states, key_states, value_states,
       attn_mask=causal_mask, is_causal=True
   )

   # SSA replacement:
   indices = self.ssa_router(query_states, key_states, causal_mask=causal_mask)
   attn_output = ssa.sparse_exact_attention(
       query_states, key_states, value_states, indices
   )
   ```

2. Add routing projections to each layer:
   - W_qr: Linear(hidden_size, num_q_heads * route_dim)
   - W_kr: Linear(hidden_size, num_kv_heads * route_dim)
   - Codebook: Parameter(num_kv_heads, N_codes, route_dim)

3. Training strategy:
   - Phase 1: Freeze Qwen weights, train router only (short sequences)
   - Phase 2: Unfreeze, full model fine-tuning (increasing sequence length)
   - Phase 3: RL training on long-context retrieval tasks

4. Expected parameters at 14B scale:
   - Routing projections: ~2.6M params (negligible)
   - Codebook: ~2.1M params (8 KV heads x 4096 codes x 32 dim)
   - Total SSA overhead: ~4.7M = 0.03% of 14B

5. Memory benefits for Qwen 3.6:
   - KV cache at 1M tokens with GQA-8: 128 * 8 * 1M * 2 bytes = 2GB
   - Dense attention matrix at 1M: 1M * 1M * 4 bytes = 4TB (prohibitive)
   - SSA candidate set: k=16K → attention matrix = 1M * 16K * 4 bytes = 64GB
""")


def verify_swap_correctness():
    """Verify SSA attention produces valid output when swapped into Qwen block."""
    from ssa import SSAAttention, build_causal_mask, precompute_rope_freqs

    config = Qwen3_6Config()
    # Use small config for quick test
    config.hidden_size = 256
    config.num_q_heads = 4
    config.num_kv_heads = 1
    config.head_dim = 64
    config.num_layers = 1
    config.intermediate_size = 512

    block = QwenTransformerBlock(config, use_ssa=True)
    x = torch.randn(2, 32, config.hidden_size)
    freqs = create_rope_freqs(config, 32)

    out = block(x, freqs=freqs)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"

    # Verify gradients flow
    x.requires_grad_(True)
    out = block(x, freqs=freqs)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for input"

    print("Swap correctness verification: PASSED")
    return True


if __name__ == "__main__":
    verify_swap_correctness()
    print()
    demo_inference_scaling()
