"""Synthetic data generators for SSA training.

Three data types needed for the three-stage training process:

1. Pre-training: Long-form text with cross-reference structure
2. SFT: Instruction-following with evidence distributed across long context
3. RL: Retrieval tasks where local evidence is misleading
"""

import torch
import random
from typing import List, Dict, Tuple, Optional


def generate_needle_in_haystack(
    seq_len: int,
    needle: str = "The secret passphrase is: XYPHER-42K",
    distractor_tokens: int = 500,
) -> Tuple[str, float]:
    """Generate a needle-in-haystack test case.

    Places a needle at a random position in a sea of distractor tokens.
    Returns (text, needle_position_ratio) where ratio ∈ [0, 1].
    """
    # Generate distractor tokens (random "words")
    distractors = _generate_distractors(seq_len - len(needle.split()))

    position = random.randint(0, len(distractors))
    ratio = position / max(len(distractors), 1)

    distractors.insert(position, needle)
    text = " ".join(distractors)

    return text, ratio


def generate_multi_hop_retrieval(
    seq_len: int,
    num_hops: int = 3,
) -> Tuple[str, str, str]:
    """Generate a multi-hop retrieval task.

    Distributes evidence across the sequence. The model must connect
    multiple non-adjacent pieces to answer.

    Example:
        [fact_a: "John works at Acme Corp"]
        ... (thousands of tokens) ...
        [fact_b: "Acme Corp's CEO is Sarah"]
        ... (thousands of tokens) ...
        Question: "Who is John's CEO?"
        Answer: "Sarah"

    Returns (text, question, answer).
    """
    entities = _generate_entity_chain(num_hops)
    distractors = _generate_distractors(seq_len)

    # Place evidence at spaced intervals
    segment_len = seq_len // (num_hops + 1)
    for i, (source, relation, target) in enumerate(entities):
        pos = segment_len * (i + 1) + random.randint(-segment_len // 4, segment_len // 4)
        pos = max(0, min(pos, len(distractors) - 1))
        fact = f"[FACT] {source} {relation} {target}."
        distractors[pos] = fact

    # Place question near the end
    question = f"Question: {entities[0][0]} -> ... -> {entities[-1][2]}. What is the final entity?"
    # The answer is the last target
    answer = entities[-1][2]

    # Place question
    q_pos = int(0.9 * len(distractors))
    distractors[q_pos] = question

    text = " ".join(distractors)
    return text, question, answer


def generate_misleading_context(
    seq_len: int,
    correct_evidence_position: float = 0.1,
) -> Tuple[str, str, str]:
    """Generate a case where local context is misleading.

    Places the correct answer FAR from the question, and semantically
    similar but wrong answers NEAR the question. Tests whether the model
    retrieves from the correct position or defaults to proximity.

    Example:
        [correct: "The Python version used is 3.11"]
        ... (thousands of tokens about unrelated topics) ...
        [misleading: "Python 3.12 was recently released with new features"]
        [misleading: "The project was initially built with Python 3.9"]
        Question: "What Python version does this project use?"
        Correct: "3.11"

    Returns (text, question, answer).
    """
    question = "What Python version does this project use?"
    answer = "3.11"

    distractors = _generate_distractors(seq_len)

    # Place correct evidence early
    correct_pos = int(correct_evidence_position * seq_len)
    distractors[correct_pos] = "[EVIDENCE] The Python version used in production is 3.11. [END EVIDENCE]"

    # Place misleading evidence near question (late in sequence)
    mislead_pos = int(0.85 * seq_len)
    distractors[mislead_pos] = "Python 3.12 was recently released with significant performance improvements."

    mislead_pos2 = int(0.88 * seq_len)
    distractors[mislead_pos2] = "The legacy codebase was originally Python 3.9."

    # Question at the end
    q_pos = int(0.95 * seq_len)
    distractors[q_pos] = question

    text = " ".join(distractors)
    return text, question, answer


def generate_contract_obligation_task(
    seq_len: int,
) -> Tuple[str, str, str]:
    """Generate a legal-contract-style multi-reference task.

    The answer depends on reconciling a definition, a clause, and an
    exception clause, each in different parts of the document.

    Returns (text, question, answer).
    """
    definition = (
        "[DEFINITION] 'Qualified Expenses' means any expense incurred after "
        "January 1, 2025, that is directly related to Project Alpha and "
        "approved in writing by the Steering Committee. [END DEFINITION]"
    )

    clause = (
        "[CLAUSE 7.3] The Client shall reimburse the Contractor for all "
        "Qualified Expenses within 30 days of receiving an itemized invoice. [END CLAUSE 7.3]"
    )

    exception = (
        "[CLAUSE 7.4] Notwithstanding Clause 7.3, expenses exceeding $50,000 "
        "in any single month require pre-approval by the Client's CFO. [END CLAUSE 7.4]"
    )

    question = (
        "Question: The Contractor submitted an invoice with $42,000 in Qualified "
        "Expenses related to Project Alpha. Does this require CFO pre-approval? "
        "Answer yes or no."
    )
    answer = "No"

    distractors = _generate_distractors(seq_len)

    # Place at different positions
    distractors[int(0.05 * seq_len)] = definition
    distractors[int(0.4 * seq_len)] = clause
    distractors[int(0.7 * seq_len)] = exception
    distractors[int(0.95 * seq_len)] = question

    text = " ".join(distractors)
    return text, question, answer


def _generate_distractors(n: int) -> List[str]:
    """Generate n distractor sentences that look like natural text."""
    topics = [
        "The quarterly financial review indicated stable revenue growth.",
        "Employee satisfaction surveys show improved work-life balance metrics.",
        "The new infrastructure deployment reduced latency by 23ms on average.",
        "Customer support tickets have decreased 15% since the last update.",
        "Regulatory compliance audits are scheduled for the upcoming quarter.",
        "The engineering team completed the migration to the new data platform.",
        "Market analysis suggests strong demand in the Asia-Pacific region.",
        "Security patches were applied to all production systems last night.",
        "The product roadmap has been updated to reflect customer feedback.",
        "Training sessions on the new tooling will begin next Monday.",
        "Database performance optimization reduced query times by 40%.",
        "The design team finalized the new UI component library.",
        "Partner integrations are on track for the Q3 release window.",
        "User research findings will be presented at the all-hands meeting.",
        "The CI/CD pipeline now includes automated security scanning.",
        "Documentation updates have been published for the latest API version.",
        "The cloud migration project has entered its final validation phase.",
        "Accessibility improvements were deployed across all web properties.",
        "The incident response playbook was updated based on recent learnings.",
        "A new monitoring dashboard provides real-time system health metrics.",
    ]
    return [random.choice(topics) for _ in range(n)]


def _generate_entity_chain(num_hops: int) -> List[Tuple[str, str, str]]:
    """Generate a chain of (subject, relation, object) triples.

    Each hop connects: subject --relation--> object.
    The chain forms: A --r1--> B --r2--> C --r3--> D ...
    """
    names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
        "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
        "Maria", "Nathan", "Olivia", "Paul",
    ]
    relations = [
        "reports to", "works for", "manages", "collaborates with",
        "is the manager of", "is employed by", "supervises",
        "is partnered with", "is a subsidiary of", "owns shares in",
    ]
    chain = []
    for i in range(num_hops):
        source = names[random.randint(0, len(names) - 1)]
        target = names[random.randint(0, len(names) - 1)]
        while target == source:
            target = names[random.randint(0, len(names) - 1)]
        relation = relations[random.randint(0, len(relations) - 1)]
        chain.append((source, relation, target))
    return chain


class CurriculumScheduler:
    """Progressive sequence length scheduling for SSA training.

    Stage 1 (Router warmup): 4K → 8K → 16K → 32K
    Stage 2 (Full fine-tune): 32K → 64K → 128K
    Stage 3 (RL): 128K → 256K → 512K → 1M
    """

    def __init__(self, stage: int = 1):
        self.stage = stage
        if stage == 1:
            self.levels = [4096, 8192, 16384, 32768]
            self.steps_per_level = 500
        elif stage == 2:
            self.levels = [32768, 65536, 131072]
            self.steps_per_level = 1000
        else:
            self.levels = [131072, 262144, 524288, 1048576]
            self.steps_per_level = 500

    def get_seq_len(self, step: int) -> int:
        """Get target sequence length for a given training step."""
        level_idx = min(step // self.steps_per_level, len(self.levels) - 1)
        return self.levels[level_idx]

    def get_level(self, step: int) -> int:
        """Get current curriculum level index."""
        return min(step // self.steps_per_level, len(self.levels) - 1)


class RLRetrievalReward:
    """Reward function for RL training on long-context retrieval.

    Rewards:
        +1.0: Correct answer using evidence from the right position
        +0.0: Correct answer but used local/distractor evidence
        -0.1: Used local evidence when distant evidence would give different answer
        -0.5: Incorrect answer
    """

    def __call__(
        self,
        predicted: str,
        ground_truth: str,
        attended_positions: torch.Tensor,
        evidence_position: int,
    ) -> float:
        correct = predicted.strip().lower() == ground_truth.strip().lower()

        # Check if model attended to the correct evidence
        attended_evidence = (attended_positions - evidence_position).abs().min() < 100

        if correct and attended_evidence:
            return 1.0  # Correct + right evidence
        elif correct and not attended_evidence:
            return 0.1  # Correct but used wrong evidence (lucky)
        else:
            return -0.5  # Incorrect
