How SSA Makes Long Context Practical
Date

Note: In this paper we share third-party verified benchmarks. A comprehensive model card is coming soon!

SubQ is built around SSA, Subquadratic Sparse Attention, a linearly scaling attention mechanism designed for long-context retrieval, reasoning, and software engineering workloads.

The core claim is simple: the hard problems enterprise AI needs to solve are long-context problems. Codebase, contracts, enterprise corpora, databases, spreadsheets, research corpora, and long-running agent sessions rarely fail because the answer is absent. They fail because the relevant evidence is distributed across a large body of context, referenced indirectly, and meaningful only when multiple pieces are held in view at once.

Dense attention made modern language models possible, but it also made long context expensive. Every token compares against every other token, so attention grows quadratically with sequence length. SSA changes that scaling behavior. Instead of computing every pairwise interaction, SSA uses content-dependent selection to route attention toward the positions that matter, regardless of where those positions appear in the sequence.

This matters because long-context capability is not just a larger prompt window. A nominal context window tells you how many tokens a model can process. A functional context window tells you how many tokens a model can reliably reason over. SSA is designed for the second problem.

SubQ keeps up with frontier dense-attention models on MRCR v2, achieves parity across core long-context retrieval tasks, and reaches a 52.2× prefill speedup over dense attention at 1M tokens. The result is a model architecture that makes million-token contexts cheaper to serve, faster to iterate on, and more useful for production workflows where retrieval failure is not acceptable.

Below, we explain what breaks in current long-context systems, how SSA works, how it was trained, and what the results imply for real software engineering and enterprise AI deployments.

Why long context is still unsolved
Most enterprise AI work does not look like a clean Q&A task over a short passage. It looks like:

a codebase where a function is defined in one module, called in dozens of others, and constrained by tests elsewhere
a contract where an obligation depends on a definition, an exception, and a referenced clause several pages apart
a research workflow where a conclusion depends on reconciling evidence across many papers
a long-running coding task where prior planning decisions, intermediate edits, review notes, and regressions all matter
These are not lookup problems. They are multi-hop reasoning problems over fragmented corpora.

The failure mode of short-context systems is not merely that they are missing some context. It is that they are forced to reason about fragments. When the whole artifact does not fit in context, systems compensate by chunking, retrieving, summarizing, and orchestrating. Those techniques are useful, but they introduce their own failure modes.

A RAG system preserves semantic similarity, but loses position, hierarchy, neighboring context, and reference structure. A chunk may contain the right text while losing why that text matters. Agentic workflows decompose large tasks into smaller model calls, but errors compound across steps, orchestration logic becomes hand-authored policy, and context is repeatedly compressed between calls. Ultimately, the human curation of these systems makes them subject to the bitter lesson, reducing their ability to generalize.

The industry response has been to build scaffolding around the model. SSA is an attempt to remove more of the reason that scaffolding is necessary.

The cost of dense attention
Attention is a retrieval operation built into the model. Each token acts as a query, comparing itself against every other token, scoring their relevance, and aggregating their information into its next representation.

This mechanism is powerful because it gives every token access to the full context. It is expensive for the same reason: every query compares against every key. The result is an all-pairs computation whose cost grows quadratically with sequence length.

At small context sizes, this is tolerable. At the scales real-world problems require—hundreds of thousands to millions of tokens—it becomes the dominant constraint. Doubling the context does not double the cost; it quadruples it. What was manageable quickly becomes prohibitive for training, serving, and iteration.

Worse, most of this computation does not matter. In trained models, the vast majority of attention weights are near zero. The model still performs the full comparison, but only a small fraction of those interactions meaningfully influence the output. Dense attention is not just quadratic, it is wastefully quadratic.

FlashAttention improved how this computation is executed. By avoiding materialization of the full attention matrix and optimizing memory movement, it made dense attention far more practical at today's context lengths. But it does not change the underlying scaling. The number of comparisons remains the same. The model still performs quadratic work; it simply performs that work more efficiently.

The same pattern holds for system-level workarounds. Retrieval pipelines, context compaction, recursive decomposition, and agentic orchestration all make dense-attention systems more usable. None of them change the scaling law. They route around the limitation, but the quadratic cost remains the boundary they are routing around.

What prior efficient architectures traded away
The field has spent years trying to make attention cheaper. The difficulty is not reducing cost. It is reducing cost without breaking retrieval.

Every prior approach has made that tradeoff somewhere.

Fixed-pattern sparse attention reduces compute by limiting which positions a token can attend to. Sliding windows, strided patterns, and dilated masks shrink the search space enough to achieve subquadratic scaling. But the routing decision is made in advance, based on position rather than content. The model decides where to look before it knows what it is looking for. When the relevant information falls outside the pattern, it is simply not seen.

State space models and recurrent alternatives take a different approach. They remove the all-pairs comparison entirely, replacing it with a compressed state that evolves across the sequence. This yields linear scaling by construction. It also introduces a constraint: the state has fixed capacity. As the sequence grows, information must be summarized, blurred, or discarded. These models preserve gist and structure. They are weaker at retrieving a specific fact introduced arbitrarily far back in the context, because that fact may no longer exist in recoverable form.

Hybrid architectures combine both ideas. Efficient layers carry most of the compute, while dense attention layers are retained to preserve retrieval. This works in practice, but it does not change the underlying scaling behavior. The dense layers remain load-bearing. As context grows, their quadratic cost dominates, and the model stays in the regime it was meant to escape. The benefit is scalar.

DeepSeek Sparse Attention is a newer sparse approach. It offsets attention's quadratic cost to a lightning indexer that selects, for each query, which keys to attend to. The indexer is itself quadratic. It scores every query against every key, with small constants but the same O(n²) scaling the architecture was meant to escape. The complexity has been moved, not removed.

The pattern is consistent. Fixed sparsity achieves efficiency by giving up content-dependent routing. Recurrent models achieve efficiency by giving up exact retrieval. Hybrids recover capability by reintroducing dense attention, and with it, the original cost. DeepSeek Sparse Attention scales quadratically and becomes cost-prohibitive at very large scale.

The open problem is not "make attention faster." It is more precise: build a mechanism that is efficient, content-dependent, and capable of retrieving from arbitrary positions across long context.

That is the role SSA is designed to play.

How SSA works
SSA—Subquadratic Selective Attention—changes how attention work is allocated.

The core idea is content-dependent selection. For each query, the model selects which parts of the sequence are worth attending to, and computes attention exactly over those positions.

Dense attention assumes every pair might matter, so it evaluates all of them. In practice, almost none do. Most pairwise interactions carry negligible signal, but the model still pays the full quadratic cost to compute them. SSA removes that assumption. It does not approximate attention. It restricts attention to the positions that actually carry signal, and skips the rest.

This gives SSA three properties that matter together:

Linear scaling in compute and memory. Attention cost grows with the number of selected positions rather than the full sequence, making long context economically usable.
Content-dependent routing. The model decides where to look based on meaning, not position. Relevant information can be retrieved regardless of where it appears.
Sparse retrieval from arbitrary positions. Unlike recurrent or compressed approaches, SSA preserves the ability to recover specific information introduced far earlier in the sequence.
The practical distinction is important. SSA is not just a faster implementation of dense attention. It reduces the amount of attention work the model performs. That reduction is what shows up as speed.

Measured in wall-clock input processing time, SSA achieves a 7.2× input processing speedup over standard attention with FlashAttention-2 on B200s at 128K tokens. FlashAttention-3 did not produce a speedup on B200s over FlashAttention-2. At 256K, that rises to 13.2×. At 512K, 23.0×. At 1M tokens, 52.2×.

Context length	SSA speed increase vs. Flash Attention on B200s
128K	7.2×
256K	13.2×
512K	23.0×
1M	52.2×
This is the throughput inversion that matters for production. Dense attention becomes slower relative to SSA as context grows. SSA becomes more advantageous exactly where long-context workloads become most valuable.

Training SSA for long-context behavior
Architecture is necessary, but not sufficient. A model can have a long context window and still fail to use it well. SSA is trained to make long-context use reliable, not just possible.

We used a three-stage training process:

Pre-training establishes base language modeling capability and the long-context representations the selection mechanism uses.
Supervised fine-tuning shapes behavior toward instruction following, structured reasoning, and code generation patterns required by enterprise workloads.
Reinforcement learning targets the behaviors that are hardest to induce through supervised examples alone: reliable long-context retrieval and coding behavior that uses available context aggressively rather than defaulting to local reasoning.
That final stage matters. Long-context failures often look plausible. A model may answer from nearby context because the nearby evidence is easier to use, even when the decisive evidence appears much earlier in the sequence. It may produce a locally correct code patch that violates an interface defined elsewhere. It may summarize a prior decision rather than preserve the exact constraint that should govern a later step. SSA's RL stage is designed around those failure modes.

The training data emphasizes long-form sources with high information density and cross-reference structure. This is the kind of data that forces the selection mechanism to learn routing over large positional distances. The goal is not benchmark memorization. The goal is to teach the model to attend to what matters regardless of where it sits.

Training infrastructure: making million-token experiments practical
Long-context training is not just a modeling problem. It is a systems problem that only appears at scale.

At million-token sequence lengths, failure modes that are invisible at shorter contexts become binding: memory pressure, sequence partitioning across devices, gradient instability, numerical precision, and kernel efficiency. These are not edge cases. They are the constraints that determine whether training runs at all.

The system trains stably at 1M tokens and beyond, maintains linear memory scaling across the training pipeline, and uses distributed sequence parallelism to shard sequences across devices when they exceed single-device limits.

The consequence is not just that long-context training becomes possible. It becomes iterable.

Under dense attention, long-context experiments are expensive enough that they are treated as reserved runs. With SSA's linear scaling, they become routine. The development loop changes: more ablations, more evaluations, faster feedback, and targeted fixes on the behaviors that actually matter at long context.

This is the deeper implication. SSA does not only reduce the cost of inference. It reduces the cost of learning long-context behavior in the first place.

Evaluating functional context, not nominal context
An advertised context window does not tell you how much context a model can use. The question is whether the model can retrieve, connect, and reason over evidence distributed across that window.

We evaluate SubQ across two axes:

Deployment viability: compute reduction and wall-clock speed
Retrieval capability: RULER, and MRCR v2
More general benchmarks will be published in the model card (coming soon).

Needle-in-a-Haystack tests exact retrieval of a single target.

RULER extends this to multi-hop retrieval, aggregation, variable tracking, and selective filtering.

MRCR v2 goes further: the model must locate and integrate multiple pieces of evidence distributed across the context, where the relevant set is not given in advance.

This is closer to the shape of real work. Finding one fact is not enough. The model has to determine which pieces of evidence matter, and combine them into a coherent answer.

Results
Compute and speed
SSA's linear scaling means doubling context length doubles the computational cost of attention, rather than quadrupling it. At 1M tokens, we see a 62.5× attention FLOP reduction relative to standard quadratic attention.

Context length	Attention FLOP reduction vs. standard attention
128K	8×
1M	62.5×
Wall-clock speed is the more product-relevant result. SSA achieves a 52.2× prefill speedup over dense attention at 1M tokens. That is the difference between a long-context system that behaves like an interactive tool and one that feels like an offline batch job.

Context length	Input processing speed increase
128K	7.2×
256K	13.2×
512K	23.0×
1M	52.2×
RULER
RULER tests retrieval and reasoning behaviors beyond simple needle lookup, including multi-hop retrieval, aggregation, variable tracking, and selective filtering.

Model	RULER @ 128K
SSA / SubQ	95.0%
Opus 4.6	94.8%
For enterprise workflows, this matters because multi-hop tasks compound. A missed reference early in the chain can corrupt every conclusion downstream.

MRCR v2
MRCR v2 is the most demanding retrieval benchmark. It evaluates the ability to locate and integrate multiple non-adjacent pieces of evidence across long context.

Model	MRCR v2 score
SSA / SubQ	65.9%
Gemini 3.1 Pro	26.3%
Opus 4.6	78.3%
Opus 4.7	32.2%
GPT 5.4	36.6%
GPT 5.5	74.0%
SubQ scores 65.9%, well in the range of Opus 4.6 at 78, and ahead of GPT 5.4 at 39, and Gemini 3.1 Pro at 23.

This result is the clearest evidence for the difference between nominal and functional context. A model may accept a long input while still failing to reason reliably over that input. MRCR v2 surfaces that gap because it requires the model to retrieve and combine evidence, not merely process tokens.

SWE-Bench Verified
SWE-Bench Verified evaluates end-to-end software engineering capability on real GitHub issues. It is not a pure retrieval benchmark. It tests whether the model can use codebase understanding to localize bugs, reason about implementation constraints, and produce patches.

Model	SWE-Bench Verified
SSA / SubQ	81.8%
Gemini 3.1 Pro	80.6%
Opus 4.6	80.8%
Opus 4.7	87.6%
GPT 5.4	Not reported
GPT 5.5	Not reported