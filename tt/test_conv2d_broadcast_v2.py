"""Optimized broadcast conv2d: multi-core, efficient accumulation.

Scalar tile layout: each element -> 32x32 tile. Spatial shifts are tile-aligned.
grid="auto" distributes output spatial positions across cores.
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
    """[N, C] -> device tensor [N*TILE, C*TILE] with each element broadcast to 32x32."""
    N, C = tensor_2d.shape
    expanded = tensor_2d.unsqueeze(2).unsqueeze(3).expand(N, C, TILE, TILE)
    result = expanded.permute(0, 2, 1, 3).reshape(N * TILE, C * TILE).to(torch.bfloat16)
    return ttnn.from_torch(result, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)


def from_scalar_tiles(tt_tensor, N, C):
    """Device tensor [N*TILE, C*TILE] -> [N, C] by reading top-left of each tile."""
    raw = ttnn.to_torch(tt_tensor)
    return raw.reshape(N, TILE, C, TILE)[:, 0, :, 0].float()


def make_broadcast_conv2d(H, W, C_in, C_out, K=3, P=1):
    H_p = H + 2 * P
    W_p = W + 2 * P
    HW = H * W
    K2C = K * K * C_in

    @ttl.kernel(grid="auto")
    def broadcast_conv2d(input_bc, weight_bc, bias_bc, output_bc):
        grid_cols, _ = ttl.grid_size(dims=2)
        positions_per_core = -(-HW // grid_cols)

        inp_dfb = ttl.make_dataflow_buffer_like(input_bc, shape=(1, 1), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(weight_bc, shape=(1, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(bias_bc, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_pos in range(positions_per_core):
                pos = core_x * positions_per_core + local_pos
                if pos < HW:
                    for co in range(C_out):
                        # Initialize accumulator with bias
                        with b_dfb.wait() as bv, acc_dfb.reserve() as acc:
                            acc.store(bv)
                        # Multiply-accumulate 9*C_in terms
                        for ki in range(K2C):
                            with inp_dfb.wait() as iv, w_dfb.wait() as wv, acc_dfb.wait() as prev, acc_dfb.reserve() as new_acc:
                                new_acc.store(prev + iv * wv)
                        # Write output
                        with acc_dfb.wait() as final_val, out_dfb.reserve() as o:
                            o.store(final_val)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_pos in range(positions_per_core):
                pos = core_x * positions_per_core + local_pos
                if pos < HW:
                    y = pos // W
                    x = pos % W
                    for co in range(C_out):
                        with b_dfb.reserve() as blk:
                            tx = ttl.copy(bias_bc[0, co], blk); tx.wait()
                        for di in range(K):
                            for dj in range(K):
                                for ci in range(C_in):
                                    inp_row = (y + di) * W_p + (x + dj)
                                    with inp_dfb.reserve() as blk:
                                        tx = ttl.copy(input_bc[inp_row, ci], blk); tx.wait()
                                    w_col = (di * K + dj) * C_in + ci
                                    with w_dfb.reserve() as blk:
                                        tx = ttl.copy(weight_bc[co, w_col], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.core(dims=2)
            for local_pos in range(positions_per_core):
                pos = core_x * positions_per_core + local_pos
                if pos < HW:
                    for co in range(C_out):
                        with out_dfb.wait() as blk:
                            tx = ttl.copy(blk, output_bc[pos, co]); tx.wait()

    return broadcast_conv2d


def test_broadcast_conv2d(H, W, C_in, C_out, device):
    print(f"\nTest: {H}x{H}, {C_in}->{C_out}", flush=True)

    torch.manual_seed(42)
    x_nchw = torch.randn(1, C_in, H, W)
    weight = torch.randn(C_out, C_in, 3, 3)
    bias = torch.randn(C_out)

    ref = F.conv2d(x_nchw, weight, bias, padding=1)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * W, C_out)

    H_p, W_p = H + 2, W + 2
    x_nhwc = x_nchw.permute(0, 2, 3, 1)
    x_padded = F.pad(x_nhwc, (0, 0, 1, 1, 1, 1, 0, 0))[0]
    x_flat = x_padded.reshape(H_p * W_p, C_in)
    w_flat = weight.permute(0, 2, 3, 1).reshape(C_out, -1)
    b_flat = bias.unsqueeze(0)

    t0 = time.time()
    input_bc = to_scalar_tiles(x_flat, device)
    weight_bc = to_scalar_tiles(w_flat, device)
    bias_bc = to_scalar_tiles(b_flat, device)
    output_bc = to_scalar_tiles(torch.zeros(H * W, C_out), device)
    print(f"  Tensors: {1000*(time.time()-t0):.0f}ms", flush=True)

    # Input size in tiles
    inp_tiles = (H_p * W_p) * C_in
    w_tiles = C_out * 9 * C_in
    print(f"  Input: {inp_tiles} tiles ({inp_tiles*2/1024:.0f}KB), "
          f"Weight: {w_tiles} tiles ({w_tiles*2/1024:.0f}KB)", flush=True)

    t0 = time.time()
    conv_kernel = make_broadcast_conv2d(H, W, C_in, C_out)
    conv_kernel(input_bc, weight_bc, bias_bc, output_bc)
    elapsed = time.time() - t0
    print(f"  Kernel: {1000*elapsed:.0f}ms", flush=True)

    out_2d = from_scalar_tiles(output_bc, H * W, C_out)
    p = pcc(ref_2d, out_2d)
    maxdiff = (ref_2d.float() - out_2d).abs().max().item()
    print(f"  PCC={p:.6f}  maxdiff={maxdiff:.4f}  {'PASS' if p > 0.99 else 'FAIL'}")
    return p


device = ttnn.open_device(device_id=0, l1_small_size=32768)

# Quick validation at small size
test_broadcast_conv2d(4, 4, 4, 4, device)

# Model size: 8x8, 64->64
test_broadcast_conv2d(8, 8, 64, 64, device)

ttnn.close_device(device)
