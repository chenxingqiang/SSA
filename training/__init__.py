"""SSA training pipeline.

Three-stage training process:
  1. Router warmup: Freeze model weights, train routing components
  2. Full fine-tune: Unfreeze, curriculum training with increasing context
  3. RL: Reinforcement learning on long-context retrieval tasks
"""

from .data import (
    generate_needle_in_haystack,
    generate_multi_hop_retrieval,
    generate_misleading_context,
    generate_contract_obligation_task,
    CurriculumScheduler,
    RLRetrievalReward,
)

__all__ = [
    "generate_needle_in_haystack",
    "generate_multi_hop_retrieval",
    "generate_misleading_context",
    "generate_contract_obligation_task",
    "CurriculumScheduler",
    "RLRetrievalReward",
]
