"""Broadcast conv2d with matmul accumulation for bfloat16 precision.

Rearrange weight/input on host so that batches of 8 kernel elements
are contiguous in memory, enabling (1,8) @ (8,1) matmul accumulation.
"""
import torch
import torch.nn.functional as F
import ttnn
import ttl
import time

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
BATCH = 8


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_scalar_tiles(tensor_2d, device):
    N, C = tensor_2d.shape
    expanded = tensor_2d.unsqueeze(2).unsqueeze(3).expand(N, C, TILE, TILE)
    result = expanded.permute(0, 2, 1, 3).reshape(N * TILE, C * TILE).to(torch.bfloat16)
    return ttnn.from_torch(result, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)


def from_scalar_tiles(tt_tensor, N, C):
    raw = ttnn.to_torch(tt_tensor)
    return raw.reshape(N, TILE, C, TILE)[:, 0, :, 0].float()


def make_broadcast_conv2d(H, W, C_in, C_out, K=3, P=1):
    """Conv2d on scalar tile layout.

    For each output position and channel: accumulate K*K*C_in terms.
    Uses matmul (1, BATCH) @ (BATCH, 1) -> (1, 1) to accumulate in float32.

    input_pre:  [HW, NUM_BATCHES, 1, BATCH] tiles - pre-gathered input patches
    weight_pre: [C_out, NUM_BATCHES, BATCH, 1] tiles - pre-arranged weights
    bias_bc:    [1, C_out] tiles - scalar broadcast
    output_bc:  [HW, C_out] tiles - scalar broadcast
    """
    HW = H * W
    K2C = K * K * C_in
    NUM_BATCHES = K2C // BATCH

    @ttl.kernel(grid="auto")
    def broadcast_conv2d(input_pre, weight_pre, bias_bc, output_bc):
        grid_cols, _ = ttl.grid_size(dims=2)
        positions_per_core = -(-HW // grid_cols)

        inp_dfb = ttl.make_dataflow_buffer_like(input_pre, shape=(1, BATCH), buffer_factor=2)
        w_dfb = ttl.make_dataflow_buffer_like(weight_pre, shape=(BATCH, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(bias_bc, shape=(1, 1), buffer_factor=2)
        partial_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(output_bc, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.core(dims=2)
            for local_pos in range(positions_per_core):
                pos = core_x * positions_per_core + local_pos
                if pos < HW:
                    for co in range(C_out):
                        with b_dfb.wait() as bv, acc_dfb.reserve() as acc:
                            acc.store(bv)
                        for batch_idx in range(NUM_BATCHES):
                            with inp_dfb.wait() as iv, w_dfb.wait() as wv:
                                with partial_dfb.reserve() as p:
                                    p.store(iv @ wv)
                            with partial_dfb.wait() as pv, acc_dfb.wait() as prev, acc_dfb.reserve() as new_acc:
                                new_acc.store(prev + pv)
                        with acc_dfb.wait() as final_val, out_dfb.reserve() as o:
                            o.store(final_val)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.core(dims=2)
            for local_pos in range(positions_per_core):
                pos = core_x * positions_per_core + local_pos
                if pos < HW:
                    for co in range(C_out):
                        with b_dfb.reserve() as blk:
                            tx = ttl.copy(bias_bc[0, co], blk); tx.wait()
                        for batch_idx in range(NUM_BATCHES):
                            # Input: row = pos, block cols [batch_idx*BATCH, (batch_idx+1)*BATCH)
                            with inp_dfb.reserve() as blk:
                                tx = ttl.copy(input_pre[pos, batch_idx * BATCH:(batch_idx + 1) * BATCH], blk)
                                tx.wait()
                            # Weight: (BATCH, 1) block
                            w_start = co * K2C + batch_idx * BATCH
                            with w_dfb.reserve() as blk:
                                tx = ttl.copy(weight_pre[w_start:w_start + BATCH, 0:1], blk)
                                tx.wait()

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


def test(H, W, C_in, C_out, device):
    print(f"\nTest: {H}x{H}, {C_in}->{C_out}", flush=True)

    torch.manual_seed(42)
    x_nchw = torch.randn(1, C_in, H, W)
    weight = torch.randn(C_out, C_in, 3, 3)
    bias = torch.randn(C_out)

    ref = F.conv2d(x_nchw, weight, bias, padding=1)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * W, C_out)

    K, P = 3, 1
    H_p, W_p = H + 2, W + 2
    HW = H * W
    K2C = K * K * C_in
    NUM_BATCHES = K2C // BATCH

    # Pre-gather input patches: for each output position, flatten the 3x3*C_in patch
    x_padded = F.pad(x_nchw.permute(0, 2, 3, 1), (0, 0, 1, 1, 1, 1, 0, 0))[0]
    # input_patches[pos, ki] = input[y+di, x+dj, ci]
    input_patches = torch.zeros(HW, K2C)
    for pos in range(HW):
        y, x = pos // W, pos % W
        ki = 0
        for di in range(K):
            for dj in range(K):
                for ci in range(C_in):
                    input_patches[pos, ki] = x_padded[y + di, x + dj, ci]
                    ki += 1

    # Pre-arrange weights: weight_flat[co, ki] = weight[co, ci, di, dj]
    weight_flat = weight.permute(0, 2, 3, 1).reshape(C_out, -1)

    # Convert to scalar tiles
    # Input: [HW, K2C] -> scalar tiles [HW, K2C] tiles
    # Each row is one spatial position, columns are the 576 kernel elements
    input_pre = to_scalar_tiles(input_patches, device)

    # Weight: [C_out * NUM_BATCHES * BATCH, 1] in tile space
    # Read as (BATCH, 1) blocks: weight_pre[co*NUM_BATCHES*BATCH + batch_idx*BATCH : +BATCH, 0:1]
    weight_col = weight_flat.reshape(C_out * K2C, 1)
    weight_pre = to_scalar_tiles(weight_col, device)

    bias_bc = to_scalar_tiles(bias.unsqueeze(0), device)
    output_bc = to_scalar_tiles(torch.zeros(HW, C_out), device)

    print(f"  K2C={K2C}, NUM_BATCHES={NUM_BATCHES}, BATCH={BATCH}", flush=True)
    print(f"  input_pre tiles: [{HW}, {K2C}]", flush=True)
    print(f"  weight_pre tiles: [{C_out * K2C}, 1]", flush=True)

    print(f"  Running (includes compile)...", flush=True)
    conv_kernel = make_broadcast_conv2d(H, W, C_in, C_out)
    t0 = time.time()
    conv_kernel(input_pre, weight_pre, bias_bc, output_bc)
    elapsed = time.time() - t0
    print(f"  Kernel: {1000*elapsed:.1f}ms", flush=True)

    out_2d = from_scalar_tiles(output_bc, HW, C_out)
    p = pcc(ref_2d, out_2d)
    maxdiff = (ref_2d.float() - out_2d).abs().max().item()
    print(f"  PCC={p:.6f}  maxdiff={maxdiff:.4f}  {'PASS' if p > 0.99 else 'FAIL'}")
    return p


device = ttnn.open_device(device_id=0, l1_small_size=32768)
# Small test first: 4x4 with 8 channels (K2C=72, 9 batches)
test(4, 4, 8, 8, device)
# Model size
test(8, 8, 64, 64, device)
ttnn.close_device(device)
