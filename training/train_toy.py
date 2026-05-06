"""End-to-end SSA router training on toy model.

Demonstrates Stage 1 (router warmup) with:
  - Curriculum scheduling: 64 -> 128 -> 256 -> 512 tokens
  - Auxiliary KL loss: align SSA routing with dense attention teacher
  - Synthetic data: multi-hop retrieval, needle-in-haystack
  - Routing quality tracking: recall@k, precision@k

Usage:
    python training/train_toy.py --steps 200 --device cuda
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import time
from typing import Tuple, List
import random

from ssa import SSAAttention, ToyTransformerBlock, CodebookRouter
from ssa.utils import build_causal_mask, precompute_rope_freqs
from ssa.attention import dense_attention
from training.data import (
    generate_multi_hop_retrieval,
    generate_needle_in_haystack,
    generate_misleading_context,
    CurriculumScheduler,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SSA router warmup training (toy)")
    parser.add_argument("--steps", type=int, default=200, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint dir")
    return parser.parse_args()


def build_toy_model(device="cuda"):
    """Build a small multi-layer SSA transformer for training.

    Config:
      - 2 layers, 256 hidden, 4 Q-heads, 2 KV-heads, head_dim=64
      - top_k=64, N_codes=256, route_dim=16
    """
    config = type("Config", (), {})()
    config.hidden_size = 256
    config.num_q_heads = 4
    config.num_kv_heads = 2
    config.head_dim = 64
    config.intermediate_size = 512
    config.num_layers = 2
    config.top_k = 64
    config.num_codebook = 1024
    config.route_dim = 32
    config.codes_per_key = 8
    config.codes_per_query = 16
    config.max_seq_len = 1024

    ssa_kwargs = dict(
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        route_dim=config.route_dim,
        num_codebook=config.num_codebook,
        codes_per_key=config.codes_per_key,
        codes_per_query=config.codes_per_query,
        top_k=config.top_k,
    )

    layers = nn.ModuleList()
    for _ in range(config.num_layers):
        layers.append(ToyTransformerBlock(hidden_size=config.hidden_size, **ssa_kwargs))

    model = nn.Module()
    model.layers = layers
    model.config = config
    model = model.to(device)

    return model


def freeze_except_router(model):
    """Freeze all weights except router components."""
    for name, param in model.named_parameters():
        is_router = ('router' in name or 'codebook' in name or
                     'W_qr' in name or 'W_kr' in name)
        param.requires_grad = is_router

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def compute_auxiliary_loss(model, hidden_states, freqs):
    """Compute KL divergence between SSA routing and dense attention.

    Uses differentiable codebook scores from the router, comparing
    the code-level routing distribution with dense attention aggregated
    to code-level targets. This allows gradients to flow through the
    codebook and routing projections.
    """
    total_loss = torch.tensor(0.0, device=hidden_states.device)
    n_layers = 0

    seq_len = hidden_states.shape[0]
    causal_mask = build_causal_mask(seq_len, device=hidden_states.device)

    for layer in model.layers:
        attn = layer.attn
        if not hasattr(attn, 'router'):
            continue

        router = attn.router

        # Project Q/K
        q = attn.q_proj(hidden_states).view(seq_len, attn.num_q_heads, attn.head_dim)
        k = attn.k_proj(hidden_states).view(seq_len, attn.num_kv_heads, attn.head_dim)

        # Apply RoPE for dense teacher
        q_rope = torch.zeros_like(q)
        k_rope = torch.zeros_like(k)
        for h in range(attn.num_q_heads):
            q_rope[:, h, :] = apply_rope_simple(q[:, h, :], freqs, h % attn.num_kv_heads)
        for h in range(attn.num_kv_heads):
            k_rope[:, h, :] = apply_rope_simple(k[:, h, :], freqs, h)

        # Dense attention distribution (teacher, per KV head)
        group_size = attn.num_q_heads // attn.num_kv_heads
        k_expanded = k_rope.unsqueeze(2).expand(-1, -1, group_size, -1)
        k_expanded = k_expanded.reshape(seq_len, attn.num_q_heads, attn.head_dim)

        scores_dense = torch.einsum("shd,thd->sht", q_rope, k_expanded)
        scores_dense = scores_dense / (attn.head_dim ** 0.5)
        scores_dense = scores_dense.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
        attn_dense = F.softmax(scores_dense, dim=-1)  # [seq_len, num_q_heads, seq_len]

        # Aggregate to per-KV-head dense attention
        dense_per_kv = attn_dense.view(seq_len, attn.num_kv_heads, group_size, seq_len)
        dense_per_kv = dense_per_kv.max(dim=2).values  # [seq_len, num_kv_heads, seq_len]

        # Differentiable routing: get code-level scores
        q_route = router.W_qr(q)  # [seq_len, num_q_heads, route_dim]
        k_route = router.W_kr(k)  # [seq_len, num_kv_heads, route_dim]

        # Group Q routing
        q_route_g = q_route.view(seq_len, attn.num_kv_heads, group_size, router.route_dim)
        q_route_g = q_route_g.mean(dim=2)  # [seq_len, num_kv_heads, route_dim]

        q_route_norm = F.normalize(q_route_g, dim=-1)
        k_route_norm = F.normalize(k_route, dim=-1)
        codebook_norm = F.normalize(router.codebook, dim=-1)

        # Q-code scores (differentiable)
        q_code_scores = torch.einsum("shd,hcd->shc", q_route_norm, codebook_norm)

        # K-code assignment probabilities (differentiable)
        k_code_logits = torch.einsum("shd,hcd->shc", k_route_norm, codebook_norm)
        k_code_probs = F.softmax(k_code_logits, dim=-1)  # [seq_len, num_kv_heads, N_c]

        # Routing distribution: q_code_scores ⋅ k_code_probs^T
        # For each query (s,h), the routing weight for key position t is:
        #   sum_c q_code_scores[s,h,c] * k_code_probs[t,h,c]
        routing_logits = torch.einsum("shc,thc->sht", q_code_scores, k_code_probs)

        # Apply causal mask with safe softmax
        routing_logits = routing_logits.masked_fill(
            ~causal_mask.unsqueeze(1), float("-inf")
        )
        # Safe softmax: subtract max first to avoid exp overflow, then mask
        routing_max = routing_logits.max(dim=-1, keepdim=True).values
        routing_exp = torch.exp(routing_logits - routing_max)
        routing_exp = routing_exp * causal_mask.unsqueeze(1).float()
        routing_probs = routing_exp / (routing_exp.sum(dim=-1, keepdim=True) + 1e-10)

        # KL(Dense || Routing): sum Dense * log(Dense / Routing)
        # Clamp to prevent log(0) = -inf
        eps = 1e-10
        dense_clamped = dense_per_kv.clamp(min=eps, max=1.0)
        routing_clamped = routing_probs.clamp(min=eps, max=1.0)
        kl = (dense_clamped * (dense_clamped.log() - routing_clamped.log())).sum(dim=-1).mean()
        total_loss = total_loss + kl
        n_layers += 1

    if n_layers > 0:
        total_loss = total_loss / n_layers

    return total_loss


def apply_rope_simple(x, freqs, head_idx):
    """Apply RoPE to a single head's tensor. Uses the real apply_rope."""
    from ssa.utils import apply_rope as _apply_rope
    return _apply_rope(x, freqs)


def compute_routing_quality(attn, q, k, causal_mask):
    """Measure routing recall@k and precision@k.

    Compares SSA candidate set with top-k dense attention keys.
    """
    with torch.no_grad():
        seq_len = q.shape[0]
        group_size = attn.num_q_heads // attn.num_kv_heads
        top_k = attn.router.top_k

        # Dense top-k per KV head
        k_exp = k.unsqueeze(2).expand(-1, -1, group_size, -1)
        k_exp = k_exp.reshape(seq_len, attn.num_q_heads, attn.head_dim)
        scores_dense = torch.einsum("shd,thd->sht", q, k_exp)
        scores_dense = scores_dense / (attn.head_dim ** 0.5)
        scores_dense = scores_dense.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))

        # SSA indices
        indices = attn.router(q, k, causal_mask=causal_mask, hard=True)

        total_recall = 0.0
        total_precision = 0.0
        count = 0

        for kv_h in range(attn.num_kv_heads):
            q_start = kv_h * group_size
            scores_kv = scores_dense[:, q_start:q_start+group_size, :]
            scores_kv = scores_kv.max(dim=1).values  # [seq_len, seq_len]
            _, dense_topk = scores_kv.topk(min(top_k, seq_len), dim=-1)

            for i in range(seq_len):
                ssa_set = set(indices[i, kv_h, :].tolist())
                dense_set = set(dense_topk[i, :top_k].tolist())

                if len(dense_set) > 0 and len(ssa_set) > 0:
                    overlap = ssa_set & dense_set
                    recall = len(overlap) / len(dense_set)
                    precision = len(overlap) / len(ssa_set)
                    total_recall += recall
                    total_precision += precision
                    count += 1

        if count == 0:
            return 0.0, 0.0
        return total_recall / count, total_precision / count


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    model = build_toy_model(device)
    trainable, total = freeze_except_router(model)
    print(f"Model: {total:,} params, {trainable:,} trainable ({100*trainable/total:.2f}%)")

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps)

    curriculum = CurriculumScheduler(stage=1)
    # Override for toy scale: 64 -> 128 -> 256 -> 512
    curriculum.levels = [64, 128, 256, 512]
    curriculum.steps_per_level = args.steps // len(curriculum.levels)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nCurriculum: {curriculum.levels}")
    print(f"Steps per level: {curriculum.steps_per_level}")
    print(f"Total steps: {args.steps}")
    print(f"\n{'='*60}")

    metrics = {"kl_loss": [], "recall": [], "precision": [], "seq_len": []}

    for step in range(args.steps):
        seq_len = curriculum.get_seq_len(step)
        model.train()

        # Generate synthetic training data
        task_type = step % 3
        if task_type == 0:
            text, question, answer = generate_multi_hop_retrieval(
                seq_len, num_hops=min(3, seq_len // 32)
            )
        elif task_type == 1:
            text, question, answer = generate_misleading_context(seq_len)
        else:
            needle_text, ratio = generate_needle_in_haystack(seq_len)

        # Encode text as random embeddings (toy: no real tokenizer)
        num_tokens = min(seq_len, len(text.split()))
        hidden = torch.randn(1, num_tokens, model.config.hidden_size, device=device)

        # Precompute RoPE freqs
        freqs = precompute_rope_freqs(model.config.head_dim, num_tokens,
                                       theta=10000.0, device=device)

        # Forward pass through layers
        x = hidden
        for layer in model.layers:
            x = layer(x, freqs=freqs, use_dense=False)

        # Auxiliary routing loss (use first batch element)
        aux_loss = compute_auxiliary_loss(model, hidden[0], freqs)

        # Backward
        optimizer.zero_grad()
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        # Compute routing quality metrics
        if step % args.log_interval == 0 or step == args.steps - 1:
            with torch.no_grad():
                h = hidden[0]
                recall, precision = compute_routing_quality(
                    model.layers[0].attn,
                    model.layers[0].attn.q_proj(h).view(num_tokens, model.config.num_q_heads, model.config.head_dim),
                    model.layers[0].attn.k_proj(h).view(num_tokens, model.config.num_kv_heads, model.config.head_dim),
                    build_causal_mask(num_tokens, device=device)
                )

            metrics["kl_loss"].append(aux_loss.item())
            metrics["recall"].append(recall)
            metrics["precision"].append(precision)
            metrics["seq_len"].append(seq_len)

            level = curriculum.get_level(step)
            print(
                f"step {step:>5d}/{args.steps} | seq={seq_len:>4d} | level={level} | "
                f"kl={aux_loss.item():.4f} | recall={recall:.3f} | prec={precision:.3f}"
            )

        # Save checkpoint
        if (step + 1) % (args.steps // 4) == 0:
            path = os.path.join(args.save_dir, f"router_step{step+1}.pt")
            router_state = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    router_state[name] = param.data.clone()
            torch.save(router_state, path)
            print(f"  -> saved {path}")

    # Final save
    path = os.path.join(args.save_dir, "router_final.pt")
    router_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            router_state[name] = param.data.clone()
    torch.save(router_state, path)

    # Print summary
    print(f"\n{'='*60}")
    print("Training complete.")
    print(f"  Initial KL loss: {metrics['kl_loss'][0]:.4f}")
    print(f"  Final KL loss:   {metrics['kl_loss'][-1]:.4f}")
    if len(metrics["kl_loss"]) > 1:
        delta_kl = metrics["kl_loss"][0] - metrics["kl_loss"][-1]
        print(f"  KL reduction:    {delta_kl:.4f}")
    print(f"  Initial recall:  {metrics['recall'][0]:.4f}")
    print(f"  Final recall:    {metrics['recall'][-1]:.4f}")
    print(f"  Initial prec:    {metrics['precision'][0]:.4f}")
    print(f"  Final prec:      {metrics['precision'][-1]:.4f}")
    print(f"\nSaved: {path}")

    return metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
