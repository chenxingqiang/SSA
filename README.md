# SSA — Subquadratic Selective Attention

SSA is a research prototype of a linearly-scaling attention mechanism. It
replaces the O(n²) dense `Q·Kᵀ` computation in standard transformer attention
with a three-stage **codebook-routed sparse pipeline**, so that the per-head
cost scales linearly with sequence length `n`.

The repository contains a pure-PyTorch reference implementation, a (work in
progress) Triton kernel, training scripts, integration glue for the
[Qwen 3.6](https://huggingface.co/Qwen) family of models, and tests covering
correctness, routing quality, and scaling behaviour.

For the full architectural specification, see [`ssa_design.md`](ssa_design.md).
The validation target and motivation are described in [`target.md`](target.md).

---

## How it works

Standard attention computes `softmax(Q·Kᵀ / √d) · V`, which costs `O(n²·d)`.
SSA approximates this in three stages:

1. **Codebook routing.** A small learnable codebook `C ∈ ℝ^{N_c × d_s}` is
   shared across heads. Queries and keys are projected into a low-dimensional
   routing space (`d_s ≪ d`) and scored against the codebook. Each key is
   assigned to its top-`b` codes; each query selects its top-`a` codes.
2. **Candidate selection.** The selected codes induce, via an inverted index,
   a candidate set of approximately `k` keys per query. Candidates are
   re-scored with the full-dimensional `Q` and `K` and pruned to exactly `k`.
3. **Sparse exact attention.** Standard scaled dot-product attention is
   computed over the `k` selected keys per query, yielding the final output.

The result is exact softmax attention over a learned, content-dependent
sparse subset of the sequence. When `k = n`, SSA reduces to dense attention
(this is checked in the tests). For `k ≪ n`, total work is roughly
`O(n · N_c · d_s + n · k · d)`, which is linear in `n`.

Grouped-query attention (GQA), RoPE, and causal masking are all supported.

---

## Repository layout

```
ssa/                  Core library
├── __init__.py         Public API
├── router.py           CodebookRouter: stages 1–2 (routing + candidate selection)
├── attention.py        sparse_exact_attention (stage 3) + dense reference
├── ssa_layer.py        SSAAttention drop-in module + ToyTransformerBlock
├── utils.py            RoPE, causal masks, GQA expansion helpers
└── triton_kernel.py    Block-sparse CUDA kernel (production path)

tests/                PyTest suite
├── test_router.py        Routing recall, top-k coverage
├── test_attention.py     SSA(k=n) ≡ dense, sparse correctness
├── test_integration.py   End-to-end SSAAttention / ToyTransformerBlock
└── test_scaling.py       FLOP / time scaling vs. n

training/             Training entry points
├── train.py            Generic training loop
├── train_toy.py        Small synthetic-task training (sanity check)
├── train_qwen.py       Fine-tuning Qwen 3.6 with SSA attention swapped in
└── data.py             Dataset utilities

benchmark/            Profiling and integration scripts
├── profile_scaling.py  Measure latency / FLOPs vs. sequence length
└── qwen_integration.py Swap SSA into a Qwen model and run inference

ssa_design.md         Full architecture specification
target.md             Project goals and validation target
AGENTS.md             Notes for AI coding agents working in this repo
```

---

## Installation

The prototype targets Python ≥ 3.10 and PyTorch ≥ 2.1.

```bash
git clone https://github.com/chenxingqiang/SSA.git
cd SSA

# Recommended: a virtual environment
python -m venv .venv
source .venv/bin/activate

pip install torch pytest
# Optional, for the production sparse path and Qwen integration:
pip install triton transformers
```

There is no `setup.py` yet — the `ssa` package is imported directly from the
repository root.

---

## Quick start

Use `SSAAttention` as a drop-in replacement for scaled dot-product attention:

```python
import torch
from ssa import SSAAttention

n, d_model = 1024, 512
h_q, h_kv = 8, 2          # GQA: 8 query heads, 2 KV heads
d_head = d_model // h_q

attn = SSAAttention(
    d_model=d_model,
    h_q=h_q,
    h_kv=h_kv,
    d_head=d_head,
    d_s=32,               # routing dimension
    n_c=2048,             # codebook size
    b=4,                  # codes per key
    a=16,                 # codes per query
    k=256,                # candidates per query
    causal=True,
)

x = torch.randn(2, n, d_model)
y = attn(x)               # (2, n, d_model)
```

Or use the lower-level pieces directly:

```python
from ssa import CodebookRouter, sparse_exact_attention, dense_attention

router = CodebookRouter(d=d_head, d_s=32, n_c=2048, b=4, a=16, k=256)
candidates = router(Q, K, causal=True)            # [n, h_kv, k] indices
out = sparse_exact_attention(Q, K, V, candidates) # exact attention on those k keys
```

---

## Testing

Run the full suite:

```bash
pytest -q
```

Notable invariants checked:

- **Correctness floor:** with `k = n`, SSA output equals dense attention
  within numerical tolerance.
- **Routing recall:** the candidate set captures ≥ 90% of the true top-`k`
  keys from full dense attention on randomized inputs.
- **Linear scaling:** FLOPs as a function of `n` fit a linear model with
  `R² > 0.99`.

---

## Training and Qwen integration

- `training/train_toy.py` runs the minimal `ToyTransformerBlock` on a
  synthetic task — useful for verifying that the routing components learn
  before scaling up.
- `training/train_qwen.py` and `benchmark/qwen_integration.py` swap SSA into
  Qwen 3.6 attention blocks. The new parameters (routing projections
  `W_qr`, `W_kr` and the codebook `C`) are initialized randomly; the rest of
  the checkpoint is loaded as-is and either fine-tuned or frozen. See
  [`ssa_design.md`](ssa_design.md) for details on weight compatibility, the
  Gumbel-softmax annealing schedule, and the auxiliary routing loss.

---

## Status

This is an **early research prototype**. The pure-PyTorch path is the
reference implementation and is what the tests exercise. The Triton sparse
kernel and the Qwen 3.6 integration are still being validated and may
change. Contributions and bug reports are welcome.

## License

No license file is present yet; treat the code as "all rights reserved" by
the repository owner until one is added.
