"""Three-stage training loop for SSA on Qwen 3.6.

Usage (when Qwen weights and CUDA GPU are available):
    python training/train.py --stage 1 --model_path qwen-3.6-14b --output_dir ./checkpoints

The script supports three stages as described in the SSA paper:
  1. Router warmup (freeze model, train routing)
  2. Full fine-tune (unfreeze, curriculum)
  3. RL on long-context retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
from typing import Optional, Dict, Any

from .data import (
    generate_multi_hop_retrieval,
    generate_misleading_context,
    generate_contract_obligation_task,
    CurriculumScheduler,
    RLRetrievalReward,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SSA training pipeline")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                       help="Training stage (1=router, 2=full, 3=RL)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Qwen 3.6 model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (keep small for long context)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=5000,
                       help="Maximum training steps")
    parser.add_argument("--grad_accum_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=500,
                       help="Checkpoint save interval")
    return parser.parse_args()


def replace_attention_with_ssa(model: nn.Module) -> nn.Module:
    """Replace standard attention layers with SSA attention.

    Walks the model, finds attention modules, and replaces them with
    SSAAttention initialized with compatible parameters.

    Args:
        model: Loaded Qwen 3.6 model

    Returns:
        Model with SSA attention modules
    """
    from ssa import SSAAttention

    replaced = 0
    for name, module in model.named_modules():
        # Qwen uses QwenAttention (or similar) class
        # Check for standard attention patterns
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') \
                and hasattr(module, 'v_proj') and hasattr(module, 'o_proj'):
            # Extract dimensions from existing projections
            hidden_size = module.hidden_size
            num_q_heads = module.num_heads
            num_kv_heads = module.num_key_value_heads
            head_dim = module.head_dim

            # Create SSA replacement
            ssa_attn = SSAAttention(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                route_dim=head_dim // 4,  # 32 for head_dim=128
                num_codebook=4096,
                codes_per_key=4,
                codes_per_query=16,
                top_k=2048,
            )

            # Copy pretrained QKV weights
            ssa_attn.q_proj.load_state_dict(module.q_proj.state_dict())
            ssa_attn.k_proj.load_state_dict(module.k_proj.state_dict())
            ssa_attn.v_proj.load_state_dict(module.v_proj.state_dict())
            ssa_attn.o_proj.load_state_dict(module.o_proj.state_dict())

            # Replace in parent module
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr_name, ssa_attn)

            replaced += 1
            print(f"  Replaced attention in {name}")

    print(f"Replaced {replaced} attention layers with SSA")
    return model


def freeze_model_weights(model: nn.Module, train_only_router: bool = True):
    """Freeze or unfreeze model weights.

    Args:
        model: The model
        train_only_router: If True, freeze everything except routing components
    """
    for name, param in model.named_parameters():
        if train_only_router:
            # Only train router components
            is_router = 'router' in name or 'W_qr' in name or \
                       'W_kr' in name or 'codebook' in name
            param.requires_grad = is_router
        else:
            param.requires_grad = True


def train_stage_1_router_warmup(model, optimizer, scheduler, args):
    """Stage 1: Freeze model weights, train routing network only.

    - Short context (4K → 32K curriculum)
    - Only router parameters are updated
    - Auxiliary routing loss + LM loss
    """
    curriculum = CurriculumScheduler(stage=1)
    freeze_model_weights(model, train_only_router=True)

    # Verify only router params are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    for step in range(args.max_steps):
        seq_len = curriculum.get_seq_len(step)

        # Generate training data
        text, question, answer = generate_multi_hop_retrieval(seq_len, num_hops=3)

        # Tokenize (placeholder — replace with real tokenizer)
        # input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)

        # # Forward pass
        # outputs = model(**input_ids, output_attentions=False)
        # lm_loss = outputs.loss

        # # Compute auxiliary routing loss (KL with dense teacher)
        # aux_loss = compute_auxiliary_routing_loss(model, input_ids)
        # loss = lm_loss + 0.1 * aux_loss

        # loss.backward()

        if (step + 1) % args.grad_accum_steps == 0:
            # optimizer.step()
            # scheduler.step()
            # optimizer.zero_grad()
            pass

        if step % args.log_interval == 0:
            print(f"Step {step:>5d} | seq_len={seq_len:>6d} |"
                  f" level={curriculum.get_level(step)}")

        if step % args.save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(args.output_dir, f"stage1_step{step}.pt")
            # torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    print("Stage 1 complete.")


def train_stage_2_full_finetune(model, optimizer, scheduler, args):
    """Stage 2: Full model fine-tune with curriculum.

    - Longer context (32K → 128K curriculum)
    - All parameters trainable
    - Reduced auxiliary loss weight
    """
    curriculum = CurriculumScheduler(stage=2)
    freeze_model_weights(model, train_only_router=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} (all weights unfrozen)")

    for step in range(args.max_steps):
        seq_len = curriculum.get_seq_len(step)

        # Mix of data types for Stage 2
        task_type = step % 3
        if task_type == 0:
            text, question, answer = generate_multi_hop_retrieval(seq_len)
        elif task_type == 1:
            text, question, answer = generate_misleading_context(seq_len)
        else:
            text, question, answer = generate_contract_obligation_task(seq_len)

        # Forward + backward (placeholder)
        if (step + 1) % args.grad_accum_steps == 0:
            pass

        if step % args.log_interval == 0:
            print(f"Step {step:>5d} | seq_len={seq_len:>6d} |"
                  f" level={curriculum.get_level(step)} | task={task_type}")

        if step % args.save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(args.output_dir, f"stage2_step{step}.pt")
            print(f"  Saved checkpoint: {checkpoint_path}")

    print("Stage 2 complete.")


def train_stage_3_rl(model, args):
    """Stage 3: Reinforcement learning on long-context retrieval.

    - Very long context (128K → 1M curriculum)
    - Policy gradient / PPO on retrieval correctness
    - Reward: +1 for correct with right evidence, -0.5 for incorrect
    - Targets the "lazy local reasoning" failure mode
    """
    curriculum = CurriculumScheduler(stage=3)
    reward_fn = RLRetrievalReward()

    print("Stage 3: RL training on long-context retrieval")
    print(f"  Curriculum: {curriculum.levels} tokens")
    print(f"  Reward structure: +1.0 correct+evidence, -0.5 incorrect")

    for step in range(args.max_steps):
        seq_len = curriculum.get_seq_len(step)

        # Generate retrieval task with known evidence position
        text, question, answer = generate_misleading_context(
            seq_len, correct_evidence_position=0.05 + random.random() * 0.3
        )

        # # Forward pass to get model response + attention distribution
        # with torch.no_grad():
        #     outputs = model(**input_ids, output_attentions=True)
        #     response = tokenizer.decode(outputs.logits.argmax(-1)[0])
        #     # Get top attended positions from the SSA router
        #     attended = get_top_attended_positions(model, input_ids)

        # # Compute reward
        # reward = reward_fn(response, answer, attended, evidence_position)
        # rewards.append(reward)

        # # Policy gradient step (simplified — PPO in practice)
        # policy_loss = -reward * outputs.log_prob
        # policy_loss.backward()

        if step % args.log_interval == 0:
            print(f"RL Step {step:>5d} | seq_len={seq_len:>7d} |"
                  f" level={curriculum.get_level(step)}")

    print("Stage 3 complete.")


def main():
    """Entry point. Placeholder — requires Qwen 3.6 weights and CUDA GPU.

    To run when hardware is available:
        python training/train.py --stage 1 --model_path /path/to/qwen-3.6-14b
    """
    args = parse_args()

    print("=" * 60)
    print(f"SSA Training Pipeline — Stage {args.stage}")
    print("=" * 60)

    # Load model (placeholder)
    print(f"\nLoading Qwen 3.6 from: {args.model_path}")
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model = None  # Placeholder
    print("  [Placeholder: set model = AutoModel.from_pretrained(args.model_path)]")

    # Replace attention with SSA
    if args.stage == 1:
        print("\nReplacing attention layers with SSA...")
        # model = replace_attention_with_ssa(model)
        print("  [Placeholder: attention replacement]")

    # Optimizer
    optimizer = AdamW(
        [p for p in (model.parameters() if model else []) if p.requires_grad],
        lr=args.lr,
    ) if model else None
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps) if optimizer else None

    # Run stage
    if args.stage == 1:
        train_stage_1_router_warmup(model, optimizer, scheduler, args)
    elif args.stage == 2:
        train_stage_2_full_finetune(model, optimizer, scheduler, args)
    elif args.stage == 3:
        train_stage_3_rl(model, args)

    print(f"\nTraining pipeline (Stage {args.stage}) ready.")
    print("Connect Qwen 3.6 weights and CUDA GPU to execute.")


if __name__ == "__main__":
    import random
    main()
