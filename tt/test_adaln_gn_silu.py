"""Test fused AdaGroupNorm + SiLU kernel against PyTorch reference."""
import torch
import torch.nn.functional as F
import ttnn
import sys
sys.path.insert(0, "/tmp")

TILE = 32
GN_EPS = 1e-5


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


import ttl


def make_adaln_gn_silu_kernel(num_groups):
    """GroupNorm + AdaLN modulation + SiLU in a single kernel.

    Inputs:
        x:          [seq_tiles*TILE, num_groups*TILE] - spatial data
        scale:      [TILE, num_groups*TILE] - per-channel scale (broadcast across seq)
        shift:      [TILE, num_groups*TILE] - per-channel shift
        scaler:     [TILE, TILE] - ones tile for reductions
        mean_scale: [TILE, TILE] - 1/N tile for mean computation
        out:        [seq_tiles*TILE, num_groups*TILE] - output buffer
    """

    @ttl.kernel(grid=(1, 1))
    def adaln_gn_silu_kernel(x, scale, shift, scaler, mean_scale, out):
        seq_tiles = x.shape[0] // TILE

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=1)
        shift_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, 1), buffer_factor=1)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        mean_tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        mean_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        # Intermediate DFB for SiLU workaround (computed intermediate reuse bug)
        mod_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for g in range(num_groups):
                    with scale_dfb.wait() as scale_tile, shift_dfb.wait() as shift_tile:
                        # Pass 1: sum(x) -> mean
                        with x_dfb.wait() as x0:
                            with red_dfb.reserve() as r:
                                r.store(ttl.math.reduce_sum(x0, sc, dims=[0, 1]))
                        with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                            acc.store(rv)
                        for t in range(seq_tiles - 1):
                            with x_dfb.wait() as xi:
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(xi, sc, dims=[0, 1]))
                            with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as new_a:
                                new_a.store(prev + rv)

                        with acc_dfb.wait() as total_sum, mean_tmp_dfb.reserve() as mean_s:
                            mean_s.store(total_sum * ms)
                        with mean_tmp_dfb.wait() as ms_val, mean_bc_dfb.reserve() as mean_bc:
                            mean_bc.store(ttl.math.broadcast(ms_val, dims=[0, 1]))

                        # Pass 2+3
                        with mean_bc_dfb.wait() as mean_bc:
                            with x_dfb.wait() as x0:
                                with sq_dfb.reserve() as sq:
                                    sq.store((x0 - mean_bc) * (x0 - mean_bc))
                            with sq_dfb.wait() as sqv:
                                with red_dfb.reserve() as r:
                                    r.store(ttl.math.reduce_sum(sqv, sc, dims=[0, 1]))
                            with red_dfb.wait() as rv, acc_dfb.reserve() as acc:
                                acc.store(rv)
                            for t in range(seq_tiles - 1):
                                with x_dfb.wait() as xi:
                                    with sq_dfb.reserve() as sq:
                                        sq.store((xi - mean_bc) * (xi - mean_bc))
                                with sq_dfb.wait() as sqv:
                                    with red_dfb.reserve() as r:
                                        r.store(ttl.math.reduce_sum(sqv, sc, dims=[0, 1]))
                                with red_dfb.wait() as rv, acc_dfb.wait() as prev, acc_dfb.reserve() as new_a:
                                    new_a.store(prev + rv)

                            with acc_dfb.wait() as total_var:
                                with mean_tmp_dfb.reserve() as var_s:
                                    var_s.store(total_var * ms + ttl.math.fill(ms, 1e-5))
                            with mean_tmp_dfb.wait() as var_eps:
                                with red_dfb.reserve() as rsq:
                                    rsq.store(ttl.math.rsqrt(var_eps))
                            with red_dfb.wait() as rsq_val, inv_bc_dfb.reserve() as inv_bc:
                                inv_bc.store(ttl.math.broadcast(rsq_val, dims=[0, 1]))

                            # Pass 3: normalize + modulate, SiLU via intermediate DFB
                            with inv_bc_dfb.wait() as inv_std_bc:
                                for t in range(seq_tiles):
                                    with x_dfb.wait() as xi, mod_dfb.reserve() as mod:
                                        normed = (xi - mean_bc) * inv_std_bc
                                        mod.store(normed * (scale_tile + ttl.math.fill(scale_tile, 1.0)) + shift_tile)
                                    with mod_dfb.wait() as mv, out_dfb.reserve() as o:
                                        o.store(mv * ttl.math.sigmoid(mv))

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            for g in range(num_groups):
                with scale_dfb.reserve() as blk:
                    tx = ttl.copy(scale[0, g], blk); tx.wait()
                with shift_dfb.reserve() as blk:
                    tx = ttl.copy(shift[0, g], blk); tx.wait()
                for t in range(seq_tiles):
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[t, g], blk); tx.wait()
                for t in range(seq_tiles):
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[t, g], blk); tx.wait()
                for t in range(seq_tiles):
                    with x_dfb.reserve() as blk:
                        tx = ttl.copy(x[t, g], blk); tx.wait()

        @ttl.datamovement()
        def dm_write():
            for g in range(num_groups):
                for t in range(seq_tiles):
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, out[t, g]); tx.wait()

    return adaln_gn_silu_kernel


adaln_gn_silu_2g = make_adaln_gn_silu_kernel(2)


# --- Test ---

def test_adaln_gn_silu(H, W, C, num_groups):
    """Compare fused kernel vs PyTorch: group_norm -> adaln modulate -> silu."""
    print(f"\nTest: H={H}, W={W}, C={C}, groups={num_groups}")

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    torch.manual_seed(42)
    x_nchw = torch.randn(1, C, H, W)
    scale_1d = torch.randn(1, C) * 0.5
    shift_1d = torch.randn(1, C) * 0.5

    # PyTorch reference
    normed = F.group_norm(x_nchw, num_groups, eps=GN_EPS)
    modulated = normed * (1 + scale_1d[:, :, None, None]) + shift_1d[:, :, None, None]
    ref = F.silu(modulated)
    ref_2d = ref.permute(0, 2, 3, 1).reshape(H * W, C)

    # TT tensors
    x_2d = x_nchw.permute(0, 2, 3, 1).reshape(H * W, C).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x_2d, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)

    scale_padded = scale_1d[0].to(torch.bfloat16).unsqueeze(0).expand(TILE, C).contiguous()
    shift_padded = shift_1d[0].to(torch.bfloat16).unsqueeze(0).expand(TILE, C).contiguous()
    scale_tt = ttnn.from_torch(scale_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)
    shift_tt = ttnn.from_torch(shift_padded, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)

    seq_tiles = (H * W) // TILE
    N = seq_tiles * TILE * TILE
    scaler = ttnn.from_torch(
        torch.ones(TILE, TILE, dtype=torch.bfloat16),
        ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    mean_scale = ttnn.from_torch(
        torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16),
        ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    out_tt = ttnn.from_torch(
        torch.zeros(H * W, C, dtype=torch.bfloat16),
        ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    adaln_gn_silu_2g(x_tt, scale_tt, shift_tt, scaler, mean_scale, out_tt)

    out_torch = ttnn.to_torch(out_tt).reshape(H * W, C)[:, :C].float()
    p = pcc(ref_2d, out_torch)
    maxdiff = (ref_2d.float() - out_torch).abs().max().item()
    print(f"  PCC={p:.6f}  maxdiff={maxdiff:.4f}")

    ttnn.close_device(device)
    return p


if __name__ == "__main__":
    p = test_adaln_gn_silu(64, 64, 64, 2)
    print(f"\nResult: PCC={p:.6f} {'PASS' if p > 0.995 else 'FAIL'}")
