"""TT-Lang conv2d via im2col + matmul. Step 1: im2col on host, matmul on device."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def im2col_host(x_nhwc, kernel_size=3, padding=1):
    """Gather 3x3 patches into [H*W, K*K*C] matrix.

    x_nhwc: [B, H, W, C] float tensor
    Returns: [H*W, K*K*C] with (di, dj, c) ordering to match weight layout
    """
    B, H, W, C = x_nhwc.shape
    K = kernel_size
    # Pad spatial dims
    x_pad = F.pad(x_nhwc, (0, 0, padding, padding, padding, padding, 0, 0))
    # x_pad: [B, H+2, W+2, C]

    patches = []
    for di in range(K):
        for dj in range(K):
            patches.append(x_pad[:, di:di+H, dj:dj+W, :])  # [B, H, W, C]

    # Stack: [B, H, W, K*K, C] -> reshape to [B*H*W, K*K*C]
    stacked = torch.stack(patches, dim=3)  # [B, H, W, K*K, C]
    return stacked.reshape(B * H * W, K * K * C)


def reshape_weight(weight_oihw):
    """Reshape conv weight from [C_out, C_in, K, K] to [K*K*C_in, C_out].

    Weight ordering must match im2col: (di, dj, c_in) along columns.
    """
    C_out, C_in, K, _ = weight_oihw.shape
    # [C_out, C_in, K, K] -> [C_out, K, K, C_in] -> [C_out, K*K*C_in] -> transpose
    return weight_oihw.permute(0, 2, 3, 1).reshape(C_out, -1).t()


def test_conv2d(H, W, C_in, C_out, device):
    print(f"\nTest: {H}x{W}, {C_in}->{C_out}")

    torch.manual_seed(42)
    x_nchw = torch.randn(1, C_in, H, W)
    weight = torch.randn(C_out, C_in, 3, 3)
    bias = torch.randn(C_out)

    # PyTorch reference
    ref = F.conv2d(x_nchw, weight, bias, padding=1)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * W, C_out)

    # im2col on host
    x_nhwc = x_nchw.permute(0, 2, 3, 1)  # [1, H, W, C_in]
    col = im2col_host(x_nhwc, kernel_size=3, padding=1)  # [H*W, 9*C_in]
    w_mat = reshape_weight(weight)  # [9*C_in, C_out]

    # Pad to tile boundaries
    HW = H * W
    K2C = 9 * C_in
    hw_pad = ((HW + TILE - 1) // TILE) * TILE
    k2c_pad = ((K2C + TILE - 1) // TILE) * TILE
    cout_pad = ((C_out + TILE - 1) // TILE) * TILE

    col_padded = torch.zeros(hw_pad, k2c_pad, dtype=torch.bfloat16)
    col_padded[:HW, :K2C] = col.to(torch.bfloat16)

    w_padded = torch.zeros(k2c_pad, cout_pad, dtype=torch.bfloat16)
    w_padded[:K2C, :C_out] = w_mat.to(torch.bfloat16)

    bias_padded = torch.zeros(1, cout_pad, dtype=torch.bfloat16)
    bias_padded[0, :C_out] = bias.to(torch.bfloat16)
    # Expand bias to full output size for elementwise add
    bias_expanded = bias_padded.expand(hw_pad, cout_pad).contiguous()

    # To device
    col_tt = ttnn.from_torch(col_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=device, memory_config=DRAM)
    w_tt = ttnn.from_torch(w_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)
    bias_tt = ttnn.from_torch(bias_expanded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=DRAM)

    # Matmul + bias
    out_tt = ttnn.matmul(col_tt, w_tt)
    out_tt = ttnn.add(out_tt, bias_tt)

    # Compare
    out_torch = ttnn.to_torch(out_tt).reshape(hw_pad, cout_pad)[:HW, :C_out].float()
    p = pcc(ref_2d, out_torch)
    maxdiff = (ref_2d.float() - out_torch).abs().max().item()
    print(f"  PCC={p:.6f}  maxdiff={maxdiff:.4f}  {'PASS' if p > 0.99 else 'FAIL'}")
    return p


device = ttnn.open_device(device_id=0, l1_small_size=32768)

# Test all model conv sizes
results = []
# conv_in: 15->64
results.append(test_conv2d(64, 64, 15, 64, device))
# Main resblocks: 64->64 at all spatial sizes
for hw in [64, 32, 16, 8]:
    results.append(test_conv2d(hw, hw, 64, 64, device))
# conv_out: 64->3
results.append(test_conv2d(64, 64, 64, 3, device))

print(f"\nAll results: {['PASS' if p > 0.99 else 'FAIL' for p in results]}")
print(f"Min PCC: {min(results):.6f}")

ttnn.close_device(device)
