#!/usr/bin/env python3
"""SSA scaling benchmark: verify linear FLOP growth and measure speedup.

Usage:
    python benchmark/profile_scaling.py
"""

import torch
import time
import math
import sys
sys.path.insert(0, ".")


def count_dense_flops(seq_len: int, num_q_heads: int, num_kv_heads: int,
                      head_dim: int) -> float:
    """Estimate FLOPs for standard dense attention per forward pass."""
    # Q·K^T: n * n * 2 * head_dim (multiply + add)
    # softmax: n * n * 5 (approx exp + sum + normalize)
    # V matmul: n * n * 2 * head_dim
    return num_q_heads * (4 * seq_len * seq_len * head_dim + 5 * seq_len * seq_len)


def count_ssa_flops(seq_len: int, num_kv_heads: int, head_dim: int,
                    N_codes: int, route_dim: int, top_k: int) -> float:
    """Estimate FLOPs for SSA per forward pass."""
    # Routing (Q+C + K+C): 2 * n * N_codes * (2 * route_dim)
    # Re-scoring: n * top_k * (2 * head_dim)
    # Sparse softmax: n * top_k * 5
    # Sparse V matmul: n * top_k * (2 * head_dim)
    pre_scoring = 2 * seq_len * N_codes * 2 * route_dim
    re_scoring = seq_len * top_k * 2 * head_dim
    softmax = seq_len * top_k * 5
    v_matmul = seq_len * top_k * 2 * head_dim
    # Per KV head, then multiplied by num_kv_heads
    return num_kv_heads * (pre_scoring + re_scoring + softmax + v_matmul)


def fit_linear_model(x: list, y: list) -> tuple:
    """Fit y = a*x + b, return (a, b, r_squared)."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    a = num / den
    b = mean_y - a * mean_x

    y_pred = [a * xi + b for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot

    return a, b, r2


def main():
    print("=" * 70)
    print("SSA Scaling Benchmark")
    print("=" * 70)

    # Configuration
    hidden_size = 512
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 64
    route_dim = 16
    N_codes = 512
    top_k = 64
    batch_size = 1

    # Sequence lengths to test (limited on Mac)
    seq_lens = [128, 256, 512, 1024, 2048, 4096]

    print(f"\nConfig: h={num_q_heads}q/{num_kv_heads}kv, head_dim={head_dim}, "
          f"top_k={top_k}, N_codes={N_codes}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")

    print(f"\n{'n':>6} | {'Dense GFLOPS':>12} | {'SSA GFLOPS':>12} | "
          f"{'Reduction':>10} | {'SSA ms':>8} | {'Dense ms':>8} | {'Speedup':>8}")
    print("-" * 70)

    ssa_times = []
    dense_times = []
    dense_gflops_list = []
    ssa_gflops_list = []

    from ssa import SSAAttention, precompute_rope_freqs

    # Pre-create the module (weights are shared across sequence lengths)
    ssa_module = SSAAttention(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        route_dim=route_dim,
        num_codebook=N_codes,
        codes_per_key=4,
        codes_per_query=8,
        top_k=top_k,
    )

    for n in seq_lens:
        x = torch.randn(batch_size, n, hidden_size)

        # Warmup
        for _ in range(2):
            _ = ssa_module(x, use_dense=True)
            _ = ssa_module(x, use_dense=False)

        # Time dense
        start = time.time()
        for _ in range(3):
            _ = ssa_module(x, use_dense=True)
        dense_time = (time.time() - start) / 3

        # Time SSA
        start = time.time()
        for _ in range(3):
            _ = ssa_module(x, use_dense=False)
        ssa_time = (time.time() - start) / 3

        # FLOP calculations
        dense_gf = count_dense_flops(n, num_q_heads, num_kv_heads, head_dim) / 1e9
        ssa_gf = count_ssa_flops(n, num_kv_heads, head_dim, N_codes, route_dim, top_k) / 1e9
        reduction = dense_gf / ssa_gf if ssa_gf > 0 else 0
        speedup = dense_time / ssa_time if ssa_time > 0 else 0

        ssa_times.append(ssa_time)
        dense_times.append(dense_time)
        dense_gflops_list.append(dense_gf)
        ssa_gflops_list.append(ssa_gf)

        print(f"{n:>6} | {dense_gf:>12.2f} | {ssa_gf:>12.2f} | "
              f"{reduction:>9.1f}x | {ssa_time*1000:>7.1f} | {dense_time*1000:>7.1f} | {speedup:>7.2f}x")

    # Fit models
    _, _, r2_ssa = fit_linear_model(seq_lens, ssa_gflops_list)

    # Quadratic fit for dense: y = a * n^2
    x_sq = [n ** 2 for n in seq_lens]
    _, _, r2_dense = fit_linear_model(x_sq, dense_gflops_list)

    print(f"\n{'='*70}")
    print(f"Scaling Analysis:")
    print(f"  SSA R^2 (linear fit):     {r2_ssa:.6f}  (1.0 = perfect linear)")
    print(f"  Dense R^2 (quadratic fit): {r2_dense:.6f}  (1.0 = perfect quadratic)")

    # Verify claims
    print(f"\nVerification:")
    print(f"  Linear scaling:   {'PASS' if r2_ssa > 0.99 else 'FAIL'} (R^2={r2_ssa:.4f})")
    print(f"  Quadratic dense:  {'PASS' if r2_dense > 0.99 else 'FAIL'} (R^2={r2_dense:.4f})")

    # FLOP ratios (doubling check)
    print(f"\nFLOP Ratios on Doubling:")
    for i in range(1, len(seq_lens)):
        if seq_lens[i] / seq_lens[i-1] == 2:
            ssa_ratio = ssa_gflops_list[i] / ssa_gflops_list[i-1]
            dense_ratio = dense_gflops_list[i] / dense_gflops_list[i-1]
            print(f"  n: {seq_lens[i-1]} -> {seq_lens[i]}: "
                  f"SSA {ssa_ratio:.2f}x, Dense {dense_ratio:.2f}x "
                  f"(target: SSA~2x, Dense~4x)")

    print()


if __name__ == "__main__":
    main()
