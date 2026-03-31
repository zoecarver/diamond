"""Conv2d via ttnn.slice for spatial shifts + ttnn.matmul accumulation.

If ttnn.slice works on device tensors, this gives us a fully traceable conv2d
with no host work and no TT-Lang needed.

Layout: padded input as [H_pad, W_pad * C] in tile layout.
Shifted slice: input[di:di+H, dj*C:(dj+W)*C] -> [H, W*C] -> reshape [H*W, C]
"""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def pad_to_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


def test_conv2d_slice(H, W, C_in, C_out, device):
    print(f"\nTest: {H}x{W}, {C_in}->{C_out}")

    torch.manual_seed(42)
    x_nchw = torch.randn(1, C_in, H, W)
    weight = torch.randn(C_out, C_in, 3, 3)
    bias = torch.randn(C_out)

    # PyTorch reference
    ref = F.conv2d(x_nchw, weight, bias, padding=1)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * W, C_out)

    # Padded input as [H_pad, W_pad * C_in]
    x_nhwc = x_nchw.permute(0, 2, 3, 1)  # [1, H, W, C_in]
    x_padded = F.pad(x_nhwc, (0, 0, 1, 1, 1, 1, 0, 0))  # [1, H+2, W+2, C_in]
    H_p, W_p = H + 2, W + 2
    x_2d = x_padded.reshape(H_p, W_p * C_in).to(torch.bfloat16)

    # Tile-pad the 2D padded input
    hp_tile = pad_to_tile(H_p)
    wc_tile = pad_to_tile(W_p * C_in)
    x_2d_padded = torch.zeros(hp_tile, wc_tile, dtype=torch.bfloat16)
    x_2d_padded[:H_p, :W_p * C_in] = x_2d

    x_tt = ttnn.from_torch(x_2d_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)

    # Weight slices [C_in, C_out] for each (di, dj), tile-padded
    cin_pad = pad_to_tile(C_in)
    cout_pad = pad_to_tile(C_out)
    hw_pad = pad_to_tile(H * W)

    weight_tts = []
    for di in range(3):
        for dj in range(3):
            w_slice = weight[:, :, di, dj].t()  # [C_in, C_out]
            w_padded = torch.zeros(cin_pad, cout_pad, dtype=torch.bfloat16)
            w_padded[:C_in, :C_out] = w_slice.to(torch.bfloat16)
            w_tt = ttnn.from_torch(w_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=DRAM)
            weight_tts.append(w_tt)

    # Bias expanded
    bias_padded = torch.zeros(hw_pad, cout_pad, dtype=torch.bfloat16)
    bias_row = torch.zeros(cout_pad, dtype=torch.bfloat16)
    bias_row[:C_out] = bias.to(torch.bfloat16)
    bias_padded[:] = bias_row
    bias_tt = ttnn.from_torch(bias_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=DRAM)

    # Accumulate 9 shifted matmuls
    acc = None
    WC = W * C_in
    wc_pad = pad_to_tile(WC)

    for idx, (di, dj) in enumerate([(di, dj) for di in range(3) for dj in range(3)]):
        # Slice: x_tt[di:di+H, dj*C_in:(dj+W)*C_in]
        row_start = di
        row_end = di + H
        col_start = dj * C_in
        col_end = col_start + WC

        # ttnn.slice wants padded end coordinates on tile boundary
        row_end_pad = pad_to_tile(row_end)
        col_end_pad = pad_to_tile(col_end)
        col_start_pad = (col_start // TILE) * TILE  # align to tile

        # Try slicing
        try:
            sliced = ttnn.slice(x_tt, [row_start, col_start], [row_end_pad, col_end_pad])
            # Reshape to [H*W, C_in] (with tile padding)
            sliced_reshaped = ttnn.reshape(sliced, [hw_pad, cin_pad])
            term = ttnn.matmul(sliced_reshaped, weight_tts[idx])
            if acc is None:
                acc = term
            else:
                acc = ttnn.add(acc, term)
                ttnn.deallocate(term)
            ttnn.deallocate(sliced)
        except Exception as e:
            print(f"  ttnn.slice failed for (di={di}, dj={dj}): {e}")
            ttnn.close_device(device)
            return 0.0

    acc = ttnn.add(acc, bias_tt)

    # Compare
    out_torch = ttnn.to_torch(acc).reshape(hw_pad, cout_pad)[:H * W, :C_out].float()
    p = pcc(ref_2d, out_torch)
    maxdiff = (ref_2d.float() - out_torch).abs().max().item()
    print(f"  PCC={p:.6f}  maxdiff={maxdiff:.4f}  {'PASS' if p > 0.99 else 'FAIL'}")

    # Cleanup
    for t in weight_tts:
        ttnn.deallocate(t)
    ttnn.deallocate(bias_tt)
    ttnn.deallocate(acc)

    return p


device = ttnn.open_device(device_id=0, l1_small_size=32768)

results = []
results.append(test_conv2d_slice(64, 64, 15, 64, device))
if results[-1] > 0:
    for hw in [64, 32, 16, 8]:
        results.append(test_conv2d_slice(hw, hw, 64, 64, device))
    results.append(test_conv2d_slice(64, 64, 64, 3, device))

    print(f"\nAll results: {['PASS' if p > 0.99 else 'FAIL' for p in results]}")
    print(f"Min PCC: {min(results):.6f}")

ttnn.close_device(device)
