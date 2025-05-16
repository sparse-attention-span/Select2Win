import triton
import triton.language as tl
import torch

# Triton kernel to perform block-sparse attention given precomputed selection indices.
# Q: [B, H, M, D] queries
# K: [B, H, N_total, D] keys
# V: [B, H, N_total, D] values
# sel_idx: [B, H, M, N_sel] selection indices for each query
# O: [B, H, M, D] output

def _ensure_divisible(x, block):
    return (x + block - 1) // block * block

@triton.jit
def sparse_attention_kernel(
    Q_ptr, K_ptr, V_ptr, sel_idx_ptr, O_ptr,
    B, H, M, N_total, D, N_sel,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    # Program identifiers
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    # Range of queries this program instance will handle
    m_start = tl.program_id(2) * BLOCK_M
    offsets_m = m_start + tl.arange(0, BLOCK_M)

    # Load queries: [BLOCK_M, D]
    q = tl.load(
        Q_ptr + batch_id * stride_qb + head_id * stride_qh + offsets_m[:, None] * stride_qm
                + tl.arange(0, BLOCK_D)[None, :] * stride_qd,
        mask=offsets_m[:, None] < M, other=0.0)

    # Load selection indices: [BLOCK_M, N_sel]
    sel_idx = tl.load(
        sel_idx_ptr + batch_id * stride_qb + head_id * stride_qh
                + offsets_m[:, None] * N_sel + tl.arange(0, N_sel)[None, :],
        mask=offsets_m[:, None] < M, other=0)

    # Initialize accumulator for output and max-k for stability
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    max_score = tl.full((BLOCK_M,), -1e9, dtype=tl.float32)
    lse = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over selected key blocks
    for i in range(N_sel):
        # Gather key vectors: [BLOCK_M, D]
        idx = sel_idx[:, i]
        k = tl.load(
            K_ptr + batch_id * stride_kb + head_id * stride_kh
                    + idx[:, None] * stride_kn + tl.arange(0, BLOCK_D)[None, :] * stride_kd,
            mask=offsets_m[:, None] < M, other=0.0)
        # Compute dot product Q·K^T: [BLOCK_M]
        score = tl.sum(q * k, axis=1)
        # Update max for softmax stability
        max_score = tl.maximum(max_score, score)
        lse = tl.where(i == 0,
                       0.0,
                       tl.exp(max_score - max_score) * lse)
        # Accumulate log-sum-exp
        lse = lse + tl.exp(score - max_score)
        # Load values: [BLOCK_M, D]
        v = tl.load(
            V_ptr + batch_id * stride_vb + head_id * stride_vh
                    + idx[:, None] * stride_vn + tl.arange(0, BLOCK_D)[None, :] * stride_vd,
            mask=offsets_m[:, None] < M, other=0.0)
        # Accumulate output o_acc += exp(score)[:, None] * v
        w = tl.exp(score - max_score)
        o_acc += w[:, None] * v

    # Final normalization: o = o_acc / lse[:, None]
    o = o_acc / lse[:, None]

    # Write back output
    tl.store(
        O_ptr + batch_id * stride_ob + head_id * stride_oh + offsets_m[:, None] * stride_om
                + tl.arange(0, BLOCK_D)[None, :] * stride_od,
        o,
        mask=offsets_m[:, None] < M)

# Launcher function

def sparse_attention(
    Q, K, V, sel_idx,
):
    # Q: [B, H, M, D], K/V: [B, H, N_total, D], sel_idx: [B, H, M, N_sel]
    B, H, M, D = Q.shape
    _, _, N_total, _ = K.shape
    N_sel = sel_idx.shape[-1]

    # Allocate output
    O = torch.empty_like(Q)

    # Define block sizes (tunable)
    BLOCK_M = 64  # number of queries per block
    BLOCK_D = D  # full feature dims per block

    grid = (B, H, (M + BLOCK_M - 1) // BLOCK_M)
    sparse_attention_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), sel_idx.data_ptr(), O.data_ptr(),
        B, H, M, N_total, D, N_sel,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BLOCK_M, BLOCK_D
    )
    return O

if __name__ == "__main__":

    # Define dimensions
    B, H, M, D = 2, 3, 4, 5        # batch, heads, queries, feature dim
    N_total = 7                   # total keys/values
    N_sel = 3                     # number of selected keys per query

    # Create toy Q, K, V
    Q = torch.arange(B * H * M * D, dtype=torch.float32).reshape(B, H, M, D)
    K = torch.arange(B * H * N_total * D, dtype=torch.float32).reshape(B, H, N_total, D)
    V = torch.arange(B * H * N_total * D, dtype=torch.float32).reshape(B, H, N_total, D)

    # Create toy selection indices in [0, N_total)
    sel_idx = torch.randint(0, N_total, (B, H, M, N_sel), dtype=torch.int32)

    # Print for inspection
    print("Q:", Q)
    print("K:", K)
    print("V:", V)
    print("sel_idx:", sel_idx)

    # Run sparse attention and print output
    O = sparse_attention(Q, K, V, sel_idx)
    print("O:", O)
