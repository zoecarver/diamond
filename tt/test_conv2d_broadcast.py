"""Conv2d via scalar-tile broadcast layout + TT-Lang.

Each element is broadcast to a full 32x32 tile. Spatial shifts become
tile-aligned shifts. The conv is done as tile-level scalar multiply-accumulate.

Start with 8x8 spatial, 64 channels (small enough to fit in DRAM).
"""
import torch
import torch.nn.functional as F
import ttnn
import ttl
import time

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_scalar_tiles(tensor_2d, device):
    """Convert [N, C] tensor to scalar tile layout: each element -> 32x32 tile.

    Input: [N, C] float tensor
    Output: device tensor [N*TILE, C*TILE] where tile[i, j] = tensor_2d[i, j] * ones(32, 32)
    """
    N, C = tensor_2d.shape
    # Expand each element to 32x32
    expanded = tensor_2d.unsqueeze(2).unsqueeze(3).expand(N, C, TILE, TILE)
    # Reshape to [N*TILE, C*TILE]
    result = expanded.permute(0, 2, 1, 3).reshape(N * TILE, C * TILE).to(torch.bfloat16)
    return ttnn.from_torch(result, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)


def from_scalar_tiles(tt_tensor, N, C):
    """Convert scalar tile tensor back to [N, C].

    Read one element per tile (top-left corner).
    """
    raw = ttnn.to_torch(tt_tensor)
    # Shape is [N*TILE, C*TILE], take every TILE-th row and column
    return raw.reshape(N, TILE, C, TILE)[:, 0, :, 0].float()


# --- Direct conv kernel in TT-Lang ---

def make_broadcast_conv2d(H, W, C_in, C_out, K=3, P=1):
    """Conv2d kernel operating on scalar-tile broadcast layout.

    input_bc:  [(H+2P)*(W+2P), C_in] tiles (scalar broadcast)
    weight_bc: [C_out, 9*C_in] tiles (scalar broadcast, flattened di/dj/ci)
    bias_bc:   [1, C_out] tiles (scalar broadcast)
    output_bc: [H*W, C_out] tiles (scalar broadcast)
    """
    H_p = H + 2 * P
    W_p = W + 2 * P
    HW = H * W

    @ttl.kernel(grid=(1, 1))
    def broadcast_conv2d(input_bc, weight_bc, bias_bc, output_bc):
        inp_dfb = ttl.make_dataflow_buffer_like(input_bc, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(weight_bc, shape=(1, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(bias_bc, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)
        tmp_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            for pos in range(HW):
                for co in range(C_out):
                    # Start with bias
                    with b_dfb.wait() as bv, acc_dfb.reserve() as acc:
                        acc.store(bv)

                    # Accumulate 9 * C_in terms
                    for ki in range(K * K * C_in):
                        with inp_dfb.wait() as iv, w_dfb.wait() as wv:
                            with acc_dfb.wait() as prev, tmp_dfb.reserve() as tmp:
                                tmp.store(prev + iv * wv)
                            with tmp_dfb.wait() as tv, acc_dfb.reserve() as acc:
                                acc.store(tv)

                    # Write final accumulated value
                    with acc_dfb.wait() as final_val, out_dfb.reserve() as o:
                        o.store(final_val)

        @ttl.datamovement()
        def dm_read():
            for pos in range(HW):
                y = pos // W
                x = pos % W
                for co in range(C_out):
                    # Read bias for this output channel
                    with b_dfb.reserve() as blk:
                        tx = ttl.copy(bias_bc[0, co], blk); tx.wait()

                    # Read 9 * C_in input/weight pairs
                    for di in range(K):
                        for dj in range(K):
                            for ci in range(C_in):
                                # Input at shifted position
                                inp_row = (y + di) * W_p + (x + dj)
                                with inp_dfb.reserve() as blk:
                                    tx = ttl.copy(input_bc[inp_row, ci], blk); tx.wait()
                                # Weight for (co, ci, di, dj)
                                w_col = (di * K + dj) * C_in + ci
                                with w_dfb.reserve() as blk:
                                    tx = ttl.copy(weight_bc[co, w_col], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            for pos in range(HW):
                for co in range(C_out):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, output_bc[pos, co]); tx.wait()

    return broadcast_conv2d


# --- Test ---

def test_broadcast_conv2d(H, W, C_in, C_out, device):
    print(f"\nTest: {H}x{W}, {C_in}->{C_out}")

    torch.manual_seed(42)
    x_nchw = torch.randn(1, C_in, H, W)
    weight = torch.randn(C_out, C_in, 3, 3)
    bias = torch.randn(C_out)

    # PyTorch reference
    ref = F.conv2d(x_nchw, weight, bias, padding=1)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * W, C_out)

    # Create scalar tile tensors
    H_p, W_p = H + 2, W + 2
    x_nhwc = x_nchw.permute(0, 2, 3, 1)  # [1, H, W, C_in]
    x_padded = F.pad(x_nhwc, (0, 0, 1, 1, 1, 1, 0, 0))[0]  # [H_p, W_p, C_in]
    x_flat = x_padded.reshape(H_p * W_p, C_in)  # [H_p*W_p, C_in]

    # Weight: [C_out, 9*C_in] - flattened (di, dj, ci)
    w_flat = weight.permute(0, 2, 3, 1).reshape(C_out, -1)  # [C_out, 9*C_in]

    # Bias: [1, C_out]
    b_flat = bias.unsqueeze(0)  # [1, C_out]

    print(f"  Creating scalar tile tensors...", flush=True)
    t0 = time.time()
    input_bc = to_scalar_tiles(x_flat, device)
    weight_bc = to_scalar_tiles(w_flat, device)
    bias_bc = to_scalar_tiles(b_flat, device)

    # Output: [H*W, C_out] scalar tiles
    out_flat = torch.zeros(H * W, C_out)
    output_bc = to_scalar_tiles(out_flat, device)
    print(f"  Tensor creation: {1000*(time.time()-t0):.0f}ms")

    print(f"  Running conv kernel...", flush=True)
    t0 = time.time()
    conv_kernel = make_broadcast_conv2d(H, W, C_in, C_out)
    conv_kernel(input_bc, weight_bc, bias_bc, output_bc)
    elapsed = time.time() - t0
    print(f"  Kernel: {1000*elapsed:.0f}ms")

    # Read back and compare
    out_2d = from_scalar_tiles(output_bc, H * W, C_out)
    p = pcc(ref_2d, out_2d)
    maxdiff = (ref_2d.float() - out_2d).abs().max().item()
    print(f"  PCC={p:.6f}  maxdiff={maxdiff:.4f}  {'PASS' if p > 0.99 else 'FAIL'}")

    return p


device = ttnn.open_device(device_id=0, l1_small_size=32768)

# Start small: 4x4 with 4 channels to validate concept quickly
test_broadcast_conv2d(4, 4, 4, 4, device)

ttnn.close_device(device)
