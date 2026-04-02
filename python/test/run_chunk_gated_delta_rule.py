#!/usr/bin/env python3
"""
Run torch_chunk_gated_delta_rule on npz inputs and save the output for comparison.

Usage:
    python run_chunk_gated_delta_rule.py --input input.npz --output output.npz \
        [--chunk_size 64] [--use_qk_l2norm] [--dtype float32|float16|bfloat16]

Input npz should contain:
    in0: query   [B, S, num_heads, D]
    in1: key     [B, S, num_heads, D]
    in2: value   [B, S, num_heads, D]
    in3: g       [B, S, num_heads]
    in4: beta    [B, S, num_heads]
    in5: recurrent_state [B, num_heads, D, D]
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]

    batch_size, num_qk_heads, sequence_length, k_head_dim = key.shape
    num_heads = value.shape[1]
    v_head_dim = value.shape[-1]
    if num_heads // num_qk_heads > 1:
        # If num_heads is a multiple of num_qk_heads, we can repeat key/value to match num_heads
        query = query.repeat_interleave(num_heads // num_qk_heads, dim=1)
        key = key.repeat_interleave(num_heads // num_qk_heads, dim=1)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1]**0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
                      diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
                            if initial_state is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
                      diagonal=1)

    # Precompute all chunk-independent operations before the loop
    intra_chunk_attn = (query @ key.transpose(-1, -2) * decay_mask).masked_fill_(mask, 0)
    q_g = query * g[..., None].exp()
    g_last_exp = g[:, :, :, -1, None, None].exp()
    k_g_diff_t = (key * (g[:, :, :, -1:] - g)[..., None].exp()).transpose(-1, -2)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = value[:, :, i] - v_prime
        attn_inter = q_g[:, :, i] @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = (last_recurrent_state * g_last_exp[:, :, i] +
                                k_g_diff_t[:, :, i] @ v_new)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1,
                                          core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def main():
    parser = argparse.ArgumentParser(description="Run torch_chunk_gated_delta_rule on npz inputs")
    parser.add_argument("--input", required=True, help="Path to input npz file")
    parser.add_argument("--output",
                        default="chunk_gated_delta_rule_ref.npz",
                        help="Path to output npz file")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size")
    parser.add_argument("--disable_qk_l2norm",
                        action="store_true",
                        help="Disable L2 norm for Q and K")
    parser.add_argument("--dtype",
                        type=str,
                        default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for inputs and output (default: float32)")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    data = np.load(args.input)
    query = torch.from_numpy(data["in0"]).to(torch_dtype)  # [B, S, num_qk_heads, D]
    key = torch.from_numpy(data["in1"]).to(torch_dtype)  # [B, S, num_qk_heads, D]
    value = torch.from_numpy(data["in2"]).to(torch_dtype)  # [B, S, num_heads, D]
    g = torch.from_numpy(data["in3"]).to(torch_dtype)  # [B, S, num_heads]
    beta = torch.from_numpy(data["in4"]).to(torch_dtype)  # [B, S, num_heads]
    recurrent_state = torch.from_numpy(data["in5"]).to(torch_dtype)  # [B, num_heads, D, D]

    print(f"query:  {query.shape}, dtype={query.dtype}")
    print(f"key:    {key.shape}, dtype={key.dtype}")
    print(f"value:  {value.shape}, dtype={value.dtype}")
    print(f"g:      {g.shape}, dtype={g.dtype}")
    print(f"beta:   {beta.shape}, dtype={beta.dtype}")
    print(f"state:  {recurrent_state.shape}, dtype={recurrent_state.dtype}")
    print(f"chunk_size:     {args.chunk_size}")
    print(f"disable_qk_l2norm:  {args.disable_qk_l2norm}")

    with torch.no_grad():
        attn_out, recurrent_state = torch_chunk_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            chunk_size=args.chunk_size,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=not args.disable_qk_l2norm,
        )

    print(f"chunk_gated_delta_rule: {attn_out.shape}, dtype={attn_out.dtype}")

    # bfloat16 is not supported by numpy, convert to float32 for saving
    out = {
        "chunk_gated_delta_rule": attn_out.float().numpy(),
        "recurrent_state": recurrent_state.float().numpy()
    }

    np.savez(args.output, **out)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
