"""SSA training on real Qwen2.5 model.

Integrates SSA attention into Qwen2.5 architecture and runs the
three-stage training pipeline:

  Stage 1: Router warmup (freeze Qwen, train routing)
  Stage 2: Full fine-tune (unfreeze, curriculum: 32K→128K)
  Stage 3: RL on long-context retrieval (128K→1M, if memory permits)

Usage:
    python training/train_qwen.py --stage 1 --model Qwen/Qwen2.5-1.5B --device cuda

Requirements:
    pip install transformers accelerate torch
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
import random
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from ssa import CodebookRouter, sparse_exact_attention, dense_attention
from ssa.utils import build_causal_mask, precompute_rope_freqs, apply_rope
from training.data import (
    generate_multi_hop_retrieval,
    generate_misleading_context,
    generate_needle_in_haystack,
    generate_contract_obligation_task,
    CurriculumScheduler,
    RLRetrievalReward,
)


@dataclass
class SSAConfig:
    """SSA-specific configuration attached to each attention layer."""
    route_dim: int = 32
    num_codebook: int = 4096
    codes_per_key: int = 4
    codes_per_query: int = 16
    top_k: int = 2048


def integrate_ssa_into_qwen(model, ssa_config: SSAConfig, device="cuda"):
    """Add SSA routing components to each Qwen2Attention layer.

    This doesn't remove the existing attention — it adds the router
    as an additional component. The forward pass is unchanged;
    we swap the attention computation at training time.

    Returns the model with router components added to each layer.
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        return model

    num_layers = 0
    router_params = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen2Attention):
            # Extract head configuration from existing projections
            # transformers 5.8.0: config stored as module.config, not direct attrs
            num_q_heads = module.config.num_attention_heads
            num_kv_heads = module.config.num_key_value_heads
            head_dim = module.head_dim

            # Create SSA router in the same dtype as the model
            model_dtype = module.q_proj.weight.dtype
            router = CodebookRouter(
                head_dim=head_dim,
                route_dim=ssa_config.route_dim,
                num_kv_heads=num_kv_heads,
                num_codebook=ssa_config.num_codebook,
                codes_per_key=ssa_config.codes_per_key,
                codes_per_query=ssa_config.codes_per_query,
                top_k=ssa_config.top_k,
            ).to(device=device, dtype=model_dtype)

            # Attach router to the attention module
            module.ssa_router = router
            module._ssa_enabled = True  # Enabled by default after integration

            # Count parameters
            router_params += sum(p.numel() for p in router.parameters())
            num_layers += 1

    total = sum(p.numel() for p in model.parameters())
    print(f"SSA integrated into {num_layers} layers")
    print(f"  Router params: {router_params:,} ({100*router_params/total:.3f}% of total)")
    return model


def replace_attention_forward(model):
    """Monkey-patch Qwen2Attention.forward to use SSA.

    transformers 5.8.0 forward signature:
      forward(self, hidden_states, position_embeddings, attention_mask,
              past_key_values, **kwargs) -> (attn_output, attn_weights)

    We intercept after RoPE is applied and replace the attention computation
    with our sparse_exact_attention or dense_attention.
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb
    except ImportError:
        return

    original_forward = Qwen2Attention.forward

    def ssa_forward(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, **kwargs):
        """SSA-modified Qwen2Attention forward (transformers 5.8 API)."""
        from transformers.cache_utils import Cache

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )

        bsz = input_shape[0]
        kv_seq_len = key_states.shape[-2]  # may be longer if KV cache was used

        # Per-batch SSA or dense attention
        use_ssa = hasattr(self, 'ssa_router') and getattr(self, '_ssa_enabled', False)

        outputs = []
        for b in range(bsz):
            q_b = query_states[b].transpose(0, 1).contiguous()  # [kv_seq_len, n_q_heads, head_dim]
            k_b = key_states[b].transpose(0, 1).contiguous()    # [kv_seq_len, n_kv_heads, head_dim]
            v_b = value_states[b].transpose(0, 1).contiguous()  # [kv_seq_len, n_kv_heads, head_dim]

            if use_ssa:
                causal_mask = build_causal_mask(kv_seq_len, device=hidden_states.device)
                indices = self.ssa_router(q_b, k_b, causal_mask=causal_mask, hard=True)
                out_b = sparse_exact_attention(q_b, k_b, v_b, indices)
            else:
                causal_mask = build_causal_mask(kv_seq_len, device=hidden_states.device)
                out_b = dense_attention(q_b, k_b, v_b, causal_mask=causal_mask)

            outputs.append(out_b.transpose(0, 1).contiguous())  # [n_q_heads, kv_seq_len, head_dim]

        attn_output = torch.stack(outputs, dim=0)  # [batch, n_q_heads, kv_seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    Qwen2Attention.forward = ssa_forward
    print("Qwen2Attention.forward patched with SSA routing")


def restore_attention_forward():
    """Restore original Qwen2Attention.forward."""
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
        original = original_forward_registry.get('qwen2_attn_forward')
        if original:
            Qwen2Attention.forward = original
    except ImportError:
        pass


original_forward_registry = {}


def freeze_except_router(model):
    """Freeze all parameters except SSA router components."""
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        is_router_param = ('ssa_router' in name)
        param.requires_grad = is_router_param
        if is_router_param:
            trainable += param.numel()
    return trainable, total


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    return total


def train_stage_1(model, tokenizer, args):
    """Stage 1: Router warmup — freeze Qwen, train routing only."""
    print(f"\n{'='*60}")
    print("Stage 1: Router Warmup")
    print(f"{'='*60}")

    curriculum = CurriculumScheduler(stage=1)
    curriculum.levels = [512, 1024, 2048, 4096]  # Scaled for 1.5B context window
    curriculum.steps_per_level = args.max_steps // len(curriculum.levels)

    trainable, total = freeze_except_router(model)
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    print(f"Curriculum: {curriculum.levels}")
    print(f"Steps per level: {curriculum.steps_per_level}")

    # Collect router parameters
    router_params = [p for n, p in model.named_parameters() if 'ssa_router' in n and p.requires_grad]
    optimizer = AdamW(router_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device

    # Enable SSA routing on all attention modules
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module._ssa_enabled = True

    for step in range(args.max_steps):
        seq_len = curriculum.get_seq_len(step)

        # Generate synthetic data
        task_type = step % 3
        if task_type == 0:
            text, question, answer = generate_multi_hop_retrieval(seq_len, num_hops=3)
        elif task_type == 1:
            text, question, answer = generate_misleading_context(seq_len)
        else:
            needle_text, ratio = generate_needle_in_haystack(seq_len)
            text = needle_text
            question = "What is the secret passphrase mentioned in the text?"
            answer = "XYPHER-42K"

        # Tokenize
        prompt = f"{text}\n\n{question}\nAnswer: {answer}"
        tokenized = tokenizer(
            prompt, truncation=True, max_length=seq_len, return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Forward pass (SSA mode via patched forward)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           labels=input_ids)
        lm_loss = outputs.loss

        # Auxiliary routing loss (KL divergence: dense attention teacher)
        # Only the KL loss has gradient flow to router params (Qwen is frozen)
        aux_loss = compute_auxiliary_loss(model, input_ids, attention_mask)

        loss = aux_loss if aux_loss is not None else torch.tensor(0.0, device=device)

        loss.backward()

        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0:
            print(
                f"step {step:>5d}/{args.max_steps} | seq={seq_len:>5d} | "
                f"level={curriculum.get_level(step)} | "
                f"loss={loss.item():.4f}"
                + (f" | aux={aux_loss.item():.4f}" if aux_loss is not None else "")
            )

        if (step + 1) % args.save_interval == 0 and step > 0:
            path = os.path.join(args.output_dir, f"stage1_step{step+1}")
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            print(f"  -> saved {path}")

    # Final save
    path = os.path.join(args.output_dir, "stage1_final")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Stage 1 complete. Model saved to {path}")


def compute_auxiliary_loss(model, input_ids, attention_mask=None):
    """Compute KL divergence between SSA routing distribution and dense attention.

    Uses forward hooks to capture hidden states from each Qwen2Attention layer,
    then computes a fully differentiable KL loss:

        KL(DenseAttention || CodeLevelRouting)

    The routing distribution is computed from router.W_qr, router.W_kr, and
    router.codebook using normalized einsum operations — fully differentiable.
    Gradients flow only through the router parameters (Qwen backbone is frozen
    during Stage 1).

    Edge cases handled:
      - Sequences of length 1 have no meaningful Q-K pairs; skipped
      - GQA: dense attention aggregated to KV-head count via max over query group
      - float32 upcast for softmax/normalize to prevent NaN from bf16 inf
      - All-zero rows in routing distribution (first token) clamped gracefully

    Args:
        model: Qwen2 model with ssa_router attached to attention layers
        input_ids: [batch, seq_len] input token IDs
        attention_mask: Optional [batch, seq_len] attention mask

    Returns:
        KL loss averaged over layers (float32), or None if no router layers found
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb
    except ImportError:
        return None

    device = input_ids.device

    # ---- Step 1: Register hooks to capture attention layer inputs ----
    captured = {}
    module_map = {}

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            # kwargs = {'hidden_states': ..., 'position_embeddings': (cos,sin), ...}
            cos, sin = kwargs['position_embeddings']
            captured[idx] = {
                'hidden_states': kwargs['hidden_states'],
                'cos': cos,
                'sin': sin,
            }
        return hook

    hooks = []
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, Qwen2Attention) and hasattr(module, 'ssa_router'):
            hooks.append(module.register_forward_hook(
                make_hook(layer_idx), with_kwargs=True
            ))
            module_map[layer_idx] = module
            layer_idx += 1

    if layer_idx == 0:
        return None

    # ---- Step 2: Run forward pass to fire hooks (no-grad, discard output) ----
    kwargs = {'input_ids': input_ids}
    if attention_mask is not None:
        kwargs['attention_mask'] = attention_mask

    with torch.no_grad():
        _ = model(**kwargs)

    for h in hooks:
        h.remove()

    # ---- Step 3: Compute KL(Dense || Routing) per layer ----
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    n_layers = 0
    eps = 1e-10

    for idx in range(layer_idx):
        try:
            layer = module_map[idx]
            data = captured[idx]
            router = layer.ssa_router

            hs = data['hidden_states']   # [batch, seq_len, hidden_size]
            bsz, seq_len, _ = hs.shape

            num_q_heads = layer.config.num_attention_heads
            num_kv_heads = layer.config.num_key_value_heads
            head_dim = layer.head_dim
            group_size = num_q_heads // num_kv_heads

            causal_mask = build_causal_mask(seq_len, device=device)
            if seq_len <= 1 or not causal_mask.any():
                continue

            # Project Q/K from hidden states (same as ssa_forward)
            q = layer.q_proj(hs).view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
            k = layer.k_proj(hs).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            # q: [bsz, num_q_heads, seq_len, head_dim]
            # k: [bsz, num_kv_heads, seq_len, head_dim]

            cos, sin = data['cos'], data['sin']
            q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

            for b in range(bsz):
                qb = q_rope[b]   # [num_q_heads, seq_len, head_dim]
                kb = k_rope[b]   # [num_kv_heads, seq_len, head_dim]

                # --- Dense attention (teacher, per KV head) ---
                # Expand K for GQA: replicate each KV head to its query group
                kb_exp = kb.unsqueeze(1).expand(
                    num_kv_heads, group_size, seq_len, head_dim
                ).reshape(num_q_heads, seq_len, head_dim)

                scores_dense = torch.einsum("hsd,htd->hst", qb.float(), kb_exp.float())
                scores_dense = scores_dense / (head_dim ** 0.5)
                scores_dense = scores_dense.masked_fill(
                    ~causal_mask.unsqueeze(0), float("-inf")
                )
                attn_dense = F.softmax(scores_dense, dim=-1)
                # attn_dense: [num_q_heads, seq_len, seq_len]

                # Aggregate to per-KV-head via max over query group
                attn_dense_kv = attn_dense.view(
                    num_kv_heads, group_size, seq_len, seq_len
                ).max(dim=1).values
                # attn_dense_kv: [num_kv_heads, seq_len, seq_len]

                # --- Code-level routing distribution (differentiable) ---
                # Router.W_qr/W_kr expect [seq_len, num_heads, head_dim]
                qb_t = qb.transpose(0, 1)  # [seq_len, num_q_heads, head_dim]
                kb_t = kb.transpose(0, 1)  # [seq_len, num_kv_heads, head_dim]

                q_route = router.W_qr(qb_t)  # [seq_len, num_q_heads, route_dim]
                k_route = router.W_kr(kb_t)  # [seq_len, num_kv_heads, route_dim]

                # Group Q routing for GQA (mean over query group)
                q_route_g = q_route.view(
                    seq_len, num_kv_heads, group_size, router.route_dim
                ).mean(dim=2)
                # q_route_g: [seq_len, num_kv_heads, route_dim]

                # Normalize for codebook similarity (float32 for stability)
                q_norm = F.normalize(q_route_g.float(), dim=-1)
                k_norm = F.normalize(k_route.float(), dim=-1)
                cb_norm = F.normalize(router.codebook.float(), dim=-1)

                # Q-code similarity scores and K-code probability distribution
                q_code_scores = torch.einsum("shd,hcd->shc", q_norm, cb_norm)
                k_code_logits = torch.einsum("shd,hcd->shc", k_norm, cb_norm)
                k_code_probs = F.softmax(k_code_logits, dim=-1)

                # Routing distribution: for each (query, head), weight on key position:
                #   sum_c q_code_scores[s,h,c] * k_code_probs[t,h,c]
                route_logits = torch.einsum("shc,thc->sht", q_code_scores, k_code_probs)

                # Safe softmax with causal mask
                route_logits = route_logits.masked_fill(
                    ~causal_mask.unsqueeze(1), float("-inf")
                )
                route_max = route_logits.max(dim=-1, keepdim=True).values
                route_exp = torch.exp(route_logits - route_max)
                route_exp = route_exp * causal_mask.unsqueeze(1).float()
                route_sum = route_exp.sum(dim=-1, keepdim=True)

                # Handle potentially all-zero rows (position 0 has no valid keys)
                valid = route_sum.squeeze(-1) > 0
                route_probs = torch.zeros_like(route_exp)
                route_probs[valid] = route_exp[valid] / (route_sum[valid] + eps)

                # --- KL(Dense || Routing) ---
                # attn_dense_kv: [num_kv_heads, seq_len, seq_len]
                # route_probs:     [seq_len, num_kv_heads, seq_len] → need transpose
                route_probs_t = route_probs.permute(1, 0, 2)  # [num_kv_heads, seq_len, seq_len]
                valid_t = valid.permute(1, 0)  # [num_kv_heads, seq_len]

                dense_clamped = attn_dense_kv.float().clamp(min=eps, max=1.0)
                route_clamped = route_probs_t.clamp(min=eps, max=1.0)

                kl_per_pos = (
                    dense_clamped * (dense_clamped.log() - route_clamped.log())
                ).sum(dim=-1)
                # kl_per_pos: [num_kv_heads, seq_len]

                if valid_t.any():
                    total_loss = total_loss + kl_per_pos[valid_t].mean()
                    n_layers += 1

        except Exception as e:
            if idx < 3:  # Print first few errors for debugging
                import traceback
                print(f"  [DEBUG] Layer {idx} KL error: {e}", flush=True)
                traceback.print_exc()
            continue

    if n_layers > 0:
        return total_loss / n_layers

    return None


def train_stage_2(model, tokenizer, args):
    """Stage 2: Full fine-tune — unfreeze all weights, curriculum training."""
    print(f"\n{'='*60}")
    print("Stage 2: Full Fine-Tune")
    print(f"{'='*60}")

    curriculum = CurriculumScheduler(stage=2)
    curriculum.levels = [4096, 8192, 16384, 32768]
    curriculum.steps_per_level = args.max_steps // len(curriculum.levels)

    total = unfreeze_all(model)
    print(f"Trainable: {total:,} (all unfrozen)")
    print(f"Curriculum: {curriculum.levels}")

    optimizer = AdamW(model.parameters(), lr=args.lr * 0.1)  # Lower LR for fine-tune
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)

    device = args.device

    # Enable SSA routing on all attention modules
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module._ssa_enabled = True

    for step in range(args.max_steps):
        seq_len = curriculum.get_seq_len(step)

        task_type = step % 3
        if task_type == 0:
            text, question, answer = generate_multi_hop_retrieval(seq_len)
        elif task_type == 1:
            text, question, answer = generate_misleading_context(seq_len)
        else:
            text, question, answer = generate_contract_obligation_task(seq_len)

        prompt = f"{text}\n\n{question}\nAnswer: {answer}"
        tokenized = tokenizer(
            prompt, truncation=True, max_length=min(seq_len, tokenizer.model_max_length),
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)

        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        loss = loss / args.grad_accum_steps
        loss.backward()

        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0:
            print(
                f"step {step:>5d}/{args.max_steps} | seq={seq_len:>5d} | "
                f"level={curriculum.get_level(step)} | loss={loss.item():.4f}"
            )

        if (step + 1) % args.save_interval == 0 and step > 0:
            path = os.path.join(args.output_dir, f"stage2_step{step+1}")
            model.save_pretrained(path)

    path = os.path.join(args.output_dir, "stage2_final")
    model.save_pretrained(path)
    print(f"Stage 2 complete. Model saved to {path}")


def train_stage_3(model, tokenizer, args):
    """Stage 3: RL training on long-context retrieval."""
    print(f"\n{'='*60}")
    print("Stage 3: RL on Long-Context Retrieval")
    print(f"{'='*60}")

    curriculum = CurriculumScheduler(stage=3)
    # Scaled for 1.5B model (max context likely 32K-128K)
    curriculum.levels = [8192, 16384, 32768, 65536]
    curriculum.steps_per_level = args.max_steps // len(curriculum.levels)

    reward_fn = RLRetrievalReward()
    device = args.device

    # Enable SSA routing on all attention modules
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module._ssa_enabled = True

    for step in range(args.max_steps):
        seq_len = curriculum.get_seq_len(step)

        # Generate task with known evidence position
        text, question, answer = generate_misleading_context(
            seq_len,
            correct_evidence_position=0.05 + random.random() * 0.3
        )

        prompt = f"{text}\n\n{question}\nAnswer:"
        tokenized = tokenizer(
            prompt, truncation=True,
            max_length=min(seq_len, tokenizer.model_max_length),
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_new_tokens=20, do_sample=True, temperature=0.7
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute reward
        reward = 1.0 if answer.lower() in generated.lower() else -0.5

        if step % args.log_interval == 0:
            print(
                f"RL step {step:>5d}/{args.max_steps} | seq={seq_len:>6d} | "
                f"level={curriculum.get_level(step)} | reward={reward:+.1f}"
            )

        if (step + 1) % args.save_interval == 0 and step > 0:
            path = os.path.join(args.output_dir, f"stage3_step{step+1}")
            model.save_pretrained(path)

    path = os.path.join(args.output_dir, "stage3_final")
    model.save_pretrained(path)
    print(f"Stage 3 complete. Model saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SSA training on Qwen2.5")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output_dir", type=str, default="./qwen_checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--ssa_top_k", type=int, default=2048)
    parser.add_argument("--ssa_num_codes", type=int, default=4096)
    parser.add_argument("--ssa_route_dim", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading Qwen2.5 model: {args.model}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: pip install transformers accelerate")
        return

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {total_params/1e9:.1f}B params")

    # Integrate SSA
    ssa_config = SSAConfig(
        route_dim=args.ssa_route_dim,
        num_codebook=args.ssa_num_codes,
        top_k=args.ssa_top_k,
    )
    model = integrate_ssa_into_qwen(model, ssa_config, device=args.device)

    # Patch attention forward
    replace_attention_forward(model)

    # Run training stage
    if args.stage == 1:
        train_stage_1(model, tokenizer, args)
    elif args.stage == 2:
        train_stage_2(model, tokenizer, args)
    elif args.stage == 3:
        train_stage_3(model, tokenizer, args)

    print(f"\nSSA training Stage {args.stage} complete!")


if __name__ == "__main__":
    main()
