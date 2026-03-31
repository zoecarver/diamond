"""Drop-in conv2d replacement: im2col on host + ttnn.matmul + bias.

Tests the full API that diamond_play.py would use, matching the interface
of the existing tt_conv2d wrapper.
"""
import torch
import torch.nn.functional as F
import ttnn
import time

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def pad_to_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


_conv_cache = {}


def tt_conv2d_matmul(x_tt, device, in_ch, out_ch, batch, h, w,
                     kernel_size=(3, 3), padding=(1, 1),
                     cache_key=None, sd=None):
    """Conv2d via im2col + matmul. Drop-in replacement for ttnn.conv2d wrapper.

    x_tt: device tensor [batch, 1, h*w, in_ch] (from ttnn.conv2d output format)
    Returns: (output_tt, out_h, out_w) matching ttnn.conv2d interface
    """
    K = kernel_size[0]
    P = padding[0]
    out_h = h  # same padding
    out_w = w
    HW = out_h * out_w

    # Cache weight matrix and bias (only computed once)
    if cache_key and cache_key not in _conv_cache:
        # Load weights from state dict
        w_key = f"inner_model.{cache_key}.weight" if "inner_model" not in cache_key else f"{cache_key}.weight"
        b_key = w_key.replace(".weight", ".bias")

        # Try both key formats
        if w_key not in sd:
            w_key = f"{cache_key}.weight"
            b_key = f"{cache_key}.bias"

        weight = sd[w_key]  # [C_out, C_in, K, K]
        bias_1d = sd[b_key]  # [C_out]

        # Reshape weight for im2col matmul: [K*K*C_in, C_out]
        K2C = K * K * in_ch
        w_mat = weight.permute(0, 2, 3, 1).reshape(out_ch, -1).t()  # [K2C, C_out]

        k2c_pad = pad_to_tile(K2C)
        cout_pad = pad_to_tile(out_ch)
        hw_pad = pad_to_tile(HW)

        w_padded = torch.zeros(k2c_pad, cout_pad, dtype=torch.bfloat16)
        w_padded[:K2C, :out_ch] = w_mat.to(torch.bfloat16)

        bias_padded = torch.zeros(hw_pad, cout_pad, dtype=torch.bfloat16)
        bias_row = torch.zeros(cout_pad, dtype=torch.bfloat16)
        bias_row[:out_ch] = bias_1d.to(torch.bfloat16)
        bias_padded[:] = bias_row

        w_tt = ttnn.from_torch(w_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)
        b_tt = ttnn.from_torch(bias_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)
        _conv_cache[cache_key] = (w_tt, b_tt, k2c_pad, cout_pad, hw_pad)

    w_tt, b_tt, k2c_pad, cout_pad, hw_pad = _conv_cache[cache_key]

    # Get input back to host for im2col
    x_raw = ttnn.to_torch(x_tt)
    # x_raw is [batch, 1, h*w, padded_ch] from ttnn conv output format
    x_nhwc = x_raw.reshape(batch, h, w, -1)[:, :, :, :in_ch].float()

    # im2col on host
    x_padded = F.pad(x_nhwc, (0, 0, P, P, P, P, 0, 0))
    patches = []
    for di in range(K):
        for dj in range(K):
            patches.append(x_padded[:, di:di+out_h, dj:dj+out_w, :])
    col = torch.cat(patches, dim=-1).reshape(batch * HW, K * K * in_ch)

    col_padded = torch.zeros(hw_pad, k2c_pad, dtype=torch.bfloat16)
    col_padded[:HW, :K * K * in_ch] = col.to(torch.bfloat16)

    col_tt = ttnn.from_torch(col_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=DRAM)

    # Matmul + bias
    out_tt = ttnn.matmul(col_tt, w_tt)
    out_tt = ttnn.add(out_tt, b_tt)
    ttnn.deallocate(col_tt)

    # Reshape to match ttnn.conv2d output: [batch, 1, h*w, out_ch_padded]
    out_tt = ttnn.reshape(out_tt, [batch, 1, HW, cout_pad])

    return out_tt, out_h, out_w


# --- Test against PyTorch ---
device = ttnn.open_device(device_id=0, l1_small_size=32768)

torch.manual_seed(42)

for H, C_in, C_out, name in [
    (64, 15, 64, "conv_in"),
    (64, 64, 64, "resblock_64"),
    (32, 64, 64, "resblock_32"),
    (16, 64, 64, "resblock_16"),
    (8, 64, 64, "resblock_8"),
    (64, 64, 3, "conv_out"),
]:
    print(f"\n{name}: {H}x{H}, {C_in}->{C_out}")

    weight = torch.randn(C_out, C_in, 3, 3)
    bias = torch.randn(C_out)
    x_nchw = torch.randn(1, C_in, H, H)

    # PyTorch reference
    ref = F.conv2d(x_nchw, weight, bias, padding=1)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * H, C_out)

    # Simulate ttnn.conv2d output format: [1, 1, H*W, C_padded]
    x_nhwc = x_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)
    cin_pad = pad_to_tile(C_in)
    x_padded_ch = torch.zeros(1, H, H, cin_pad, dtype=torch.bfloat16)
    x_padded_ch[:, :, :, :C_in] = x_nhwc
    x_flat = x_padded_ch.reshape(1, 1, H * H, cin_pad)
    x_tt = ttnn.from_torch(x_flat, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)

    # Build fake state dict
    sd = {f"{name}.weight": weight, f"{name}.bias": bias}

    # Clear cache for fresh test
    _conv_cache.clear()

    t0 = time.time()
    out_tt, oh, ow = tt_conv2d_matmul(x_tt, device, C_in, C_out, 1, H, H,
                                       cache_key=name, sd=sd)
    t1 = time.time()

    cout_pad = pad_to_tile(C_out)
    out_torch = ttnn.to_torch(out_tt).reshape(1, 1, H * H, cout_pad)
    out_torch = out_torch[0, 0, :H * H, :C_out].float()
    p = pcc(ref_2d, out_torch)
    print(f"  PCC={p:.6f}  time={1000*(t1-t0):.1f}ms  {'PASS' if p > 0.99 else 'FAIL'}")

ttnn.close_device(device)
