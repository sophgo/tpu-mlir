import argparse
import numpy as np
import torch
import torch.nn.functional as F


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(query,
                                     key,
                                     value,
                                     g,
                                     beta,
                                     last_recurrent_state,
                                     use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).squeeze(2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch_size, num_qk_heads, sequence_length, k_head_dim = key.shape
    num_heads = value.shape[1]
    v_head_dim = value.shape[-1]
    if num_heads // num_qk_heads > 1:
        # If num_heads is a multiple of num_qk_heads, we can repeat key/value to match num_heads
        query = query.repeat_interleave(num_heads // num_qk_heads, dim=1)
        key = key.repeat_interleave(num_heads // num_qk_heads, dim=1)

    scale = 1 / (query.shape[-1]**0.5)
    query = query * scale
    g = g.exp().unsqueeze(-1).unsqueeze(-1)
    beta = beta.unsqueeze(-1)

    last_recurrent_state = last_recurrent_state * g
    kv_mem = (last_recurrent_state * key.unsqueeze(-1)).sum(dim=-2)
    delta = (value - kv_mem) * beta
    last_recurrent_state = last_recurrent_state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    core_attn_out = (last_recurrent_state * query.unsqueeze(-1)).sum(dim=-2)

    # Restore shape: [batch, heads, v_head_dim] -> [batch, 1, heads, v_head_dim] -> [batch, 1, heads, v_head_dim]
    core_attn_out = core_attn_out.unsqueeze(2).transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def main():
    parser = argparse.ArgumentParser(
        description="Run torch_recurrent_gated_delta_rule on npz inputs")
    parser.add_argument("--input", required=True, help="Path to input npz file")
    parser.add_argument("--output",
                        default="recurrent_gated_delta_rule_ref.npz",
                        help="Path to output npz file")
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
    query = torch.from_numpy(data["in0"]).to(torch_dtype)  # [B, 1, num_heads, D]
    key = torch.from_numpy(data["in1"]).to(torch_dtype)  # [B, 1, num_heads, D]
    value = torch.from_numpy(data["in2"]).to(torch_dtype)  # [B, 1, num_heads, D]
    g = torch.from_numpy(data["in3"]).to(torch_dtype)  # [B, 1, num_heads]
    beta = torch.from_numpy(data["in4"]).to(torch_dtype)  # [B, 1, num_heads]
    recurrent_state = torch.from_numpy(data["in5"]).to(torch_dtype)  # [B, num_heads, D, D]

    print(f"query:  {query.shape}, dtype={query.dtype}")
    print(f"key:    {key.shape}, dtype={key.dtype}")
    print(f"value:  {value.shape}, dtype={value.dtype}")
    print(f"g:      {g.shape}, dtype={g.dtype}")
    print(f"beta:   {beta.shape}, dtype={beta.dtype}")
    print(f"state:  {recurrent_state.shape}, dtype={recurrent_state.dtype}")
    print(f"disable_qk_l2norm:  {args.disable_qk_l2norm}")

    with torch.no_grad():
        attn_out, recurrent_state = torch_recurrent_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            last_recurrent_state=recurrent_state,
            use_qk_l2norm_in_kernel=not args.disable_qk_l2norm,
        )

    print(f"recurrent_gated_delta_rule: {attn_out.shape}, dtype={attn_out.dtype}")

    # bfloat16 is not supported by numpy, convert to float32 for saving
    out = {
        "recurrent_gated_delta_rule": attn_out.float().numpy(),
        "recurrent_state": recurrent_state.float().numpy()
    }

    np.savez(args.output, **out)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
