"""TT-Lang conv2d: im2col in datamovement + matmul on device.

Strategy: build im2col matrix on device via TT-Lang dm_read gathering
3x3 patches, then matmul with ttnn.matmul.

Two kernels:
1. im2col_kernel: TT-Lang kernel that gathers patches from input -> col matrix
2. ttnn.matmul: col @ weight + bias
"""
import torch
import torch.nn.functional as F
import ttnn
import ttl

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def make_im2col_kernel(H, W, C_in, kernel_size=3, padding=1):
    """Create im2col kernel for specific spatial size.

    Input:  x_pad [H_pad * W_pad, C_in] - zero-padded input in tile layout
    Output: col [(H*W)_pad, (K*K*C_in)_pad] - im2col matrix in tile layout

    The dm_read gathers 3x3 patches from x_pad into col format.
    The compute thread is a passthrough (just copies tiles).
    """
    K = kernel_size
    P = padding
    H_pad = H + 2 * P
    W_pad = W + 2 * P
    HW = H * W
    K2C = K * K * C_in

    # Padded dimensions for tile alignment
    hw_tiles = (HW + TILE - 1) // TILE
    k2c_tiles = (K2C + TILE - 1) // TILE
    hw_pad_tiles = (H_pad * W_pad + TILE - 1) // TILE
    c_in_tiles = (C_in + TILE - 1) // TILE

    @ttl.kernel(grid=(1, 1))
    def im2col_kernel(x_pad, col):
        """Gather 3x3 patches from padded input into im2col matrix."""
        # col is [hw_tiles * TILE, k2c_tiles * TILE]
        # Each row of col = flattened 3x3 patch (K*K*C_in values)
        # We iterate output tiles in row-major order

        x_dfb = ttl.make_dataflow_buffer_like(x_pad, shape=(1, 1), buffer_factor=2)
        col_dfb = ttl.make_dataflow_buffer_like(col, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            for r in range(hw_tiles):
                for c in range(k2c_tiles):
                    with x_dfb.wait() as xv, col_dfb.reserve() as o:
                        o.store(xv)

        @ttl.datamovement()
        def dm_read():
            for r in range(hw_tiles):
                for c in range(k2c_tiles):
                    # Each col tile [TILE, TILE] contains elements from the
                    # im2col matrix. We need to figure out which input tiles
                    # to read. For now, read a zero tile and fill correctly.
                    # This is the hard part - tile-level gather.
                    #
                    # For each output row i (in range [r*TILE, (r+1)*TILE)):
                    #   spatial pos: y = i // W, x = i % W
                    #   For each col j (in range [c*TILE, (c+1)*TILE)):
                    #     patch index: di = (j // C_in) // K
                    #     dj = (j // C_in) % K
                    #     ch = j % C_in
                    #     input row = (y + di) * W_pad + (x + dj)
                    #     input col = ch
                    #
                    # This scatter pattern doesn't map to tile reads cleanly.
                    # We'd need per-element indexing which TT-Lang doesn't support.
                    #
                    # WORKAROUND: Read the needed input tile and let compute
                    # handle the shuffle. But compute can't do arbitrary indexing
                    # within tiles either.
                    #
                    # This approach won't work at tile granularity.
                    # We need a different strategy.
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x_pad[0, 0], blk)
                        tx.wait()

        @ttl.datamovement()
        def dm_write():
            for r in range(hw_tiles):
                for c in range(k2c_tiles):
                    with col_dfb.wait() as blk:
                        tx = ttl.copy(blk, col[r, c])
                        tx.wait()

    return im2col_kernel


# The tile-level im2col gather is fundamentally hard because the 3x3 spatial
# offsets create sub-tile scatter patterns. Each output tile needs elements
# from many different input tiles at non-aligned offsets.
#
# Better approach: use the "shifted matmul" decomposition.
# Conv2d = sum over (di, dj) of: shifted_input[:, :, di, dj] @ weight[di, dj]
# Each shift is a spatial translation that CAN be done at tile granularity
# if we pre-pad and slice correctly.
#
# For 3x3 conv with padding=1:
#   output = sum_{di=0}^{2} sum_{dj=0}^{2} input_shifted[di,dj] @ W[di,dj]
#
# where input_shifted[di,dj] = x_padded[di:di+H, dj:dj+W, :]
# reshaped to [H*W, C_in]
#
# and W[di,dj] = weight[:, :, di, dj] reshaped to [C_in, C_out]
#
# Each of the 9 terms is a standard matmul on [H*W, C_in] @ [C_in, C_out].
# The spatial shift is just an offset in the dm_read!

print("=" * 60)
print("Approach: shifted matmul decomposition")
print("conv2d = sum of 9 matmuls with spatially shifted inputs")
print("=" * 60)


def test_shifted_matmul_conv2d(H, W, C_in, C_out, device):
    """Conv2d via 9 shifted matmuls. All on device via ttnn.matmul."""
    print(f"\nTest: {H}x{W}, {C_in}->{C_out}")

    torch.manual_seed(42)
    x_nchw = torch.randn(1, C_in, H, W)
    weight = torch.randn(C_out, C_in, 3, 3)
    bias = torch.randn(C_out)

    # PyTorch reference
    ref = F.conv2d(x_nchw, weight, bias, padding=1)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * W, C_out)

    # Pad input: [1, H, W, C_in] -> [1, H+2, W+2, C_in]
    x_nhwc = x_nchw.permute(0, 2, 3, 1)
    x_padded = F.pad(x_nhwc, (0, 0, 1, 1, 1, 1, 0, 0))  # pad H and W by 1
    H_p, W_p = H + 2, W + 2

    # Tile-align dimensions
    HW = H * W
    hw_pad = ((HW + TILE - 1) // TILE) * TILE
    cin_pad = ((C_in + TILE - 1) // TILE) * TILE
    cout_pad = ((C_out + TILE - 1) // TILE) * TILE

    # Prepare 9 shifted input slices and weight slices
    # For each (di, dj), input_shifted = x_padded[:, di:di+H, dj:dj+W, :]
    # reshaped to [H*W, C_in], then padded to tile alignment
    shifted_inputs = []
    weight_slices = []
    for di in range(3):
        for dj in range(3):
            # Shifted input slice
            sliced = x_padded[:, di:di+H, dj:dj+W, :].reshape(HW, C_in)
            padded = torch.zeros(hw_pad, cin_pad, dtype=torch.bfloat16)
            padded[:HW, :C_in] = sliced.to(torch.bfloat16)
            shifted_inputs.append(padded)

            # Weight slice: weight[:, :, di, dj] -> [C_in, C_out]
            w_slice = weight[:, :, di, dj].t()  # [C_in, C_out]
            w_padded = torch.zeros(cin_pad, cout_pad, dtype=torch.bfloat16)
            w_padded[:C_in, :C_out] = w_slice.to(torch.bfloat16)
            weight_slices.append(w_padded)

    # Send all to device
    input_tts = [ttnn.from_torch(s, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=device, memory_config=DRAM) for s in shifted_inputs]
    weight_tts = [ttnn.from_torch(w, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=DRAM) for w in weight_slices]

    # Bias
    bias_padded = torch.zeros(hw_pad, cout_pad, dtype=torch.bfloat16)
    bias_row = torch.zeros(cout_pad, dtype=torch.bfloat16)
    bias_row[:C_out] = bias.to(torch.bfloat16)
    bias_padded[:] = bias_row
    bias_tt = ttnn.from_torch(bias_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=DRAM)

    # Accumulate 9 matmuls
    acc = ttnn.matmul(input_tts[0], weight_tts[0])
    for i in range(1, 9):
        term = ttnn.matmul(input_tts[i], weight_tts[i])
        acc = ttnn.add(acc, term)
        ttnn.deallocate(term)
    acc = ttnn.add(acc, bias_tt)

    # Compare
    out_torch = ttnn.to_torch(acc).reshape(hw_pad, cout_pad)[:HW, :C_out].float()
    p = pcc(ref_2d, out_torch)
    maxdiff = (ref_2d.float() - out_torch).abs().max().item()
    print(f"  PCC={p:.6f}  maxdiff={maxdiff:.4f}  {'PASS' if p > 0.99 else 'FAIL'}")

    # Cleanup
    for t in input_tts:
        ttnn.deallocate(t)
    for t in weight_tts:
        ttnn.deallocate(t)
    ttnn.deallocate(bias_tt)
    ttnn.deallocate(acc)

    return p


device = ttnn.open_device(device_id=0, l1_small_size=32768)

results = []
results.append(test_shifted_matmul_conv2d(64, 64, 15, 64, device))
for hw in [64, 32, 16, 8]:
    results.append(test_shifted_matmul_conv2d(hw, hw, 64, 64, device))
results.append(test_shifted_matmul_conv2d(64, 64, 64, 3, device))

print(f"\nAll results: {['PASS' if p > 0.99 else 'FAIL' for p in results]}")
print(f"Min PCC: {min(results):.6f}")

ttnn.close_device(device)
