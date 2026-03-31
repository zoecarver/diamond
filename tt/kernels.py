"""
TT-Lang fused kernels for Diamond world model.

Patterns adapted from toy-wm. All kernels use grid="auto" with
streaming tile distribution across all available cores.
"""

import ttl

TILE = 32


# ---------------------------------------------------------------------------
# GroupNorm: 3-pass algorithm (mean, variance, normalize)
# Replaces ttnn.group_norm which crashes at small spatial sizes.
# ---------------------------------------------------------------------------

def make_groupnorm_kernel(num_groups):

    @ttl.kernel(grid=(1, 1))
    def groupnorm_kernel(x, scaler, mean_scale, out):
        seq_tiles = x.shape[0] // TILE

        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
        ms_dfb = ttl.make_dataflow_buffer_like(mean_scale, shape=(1, 1), buffer_factor=1)
        sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        acc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        mean_tmp_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
        mean_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        inv_bc_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with sc_dfb.wait() as sc, ms_dfb.wait() as ms:
                for g in range(num_groups):
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

                    # Pass 2+3 with mean_bc alive
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

                        with inv_bc_dfb.wait() as inv_std_bc:
                            for t in range(seq_tiles):
                                with x_dfb.wait() as xi, out_dfb.reserve() as o:
                                    o.store((xi - mean_bc) * inv_std_bc)

        @ttl.datamovement()
        def dm_read():
            with sc_dfb.reserve() as blk:
                tx = ttl.copy(scaler[0, 0], blk); tx.wait()
            with ms_dfb.reserve() as blk:
                tx = ttl.copy(mean_scale[0, 0], blk); tx.wait()
            for g in range(num_groups):
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

    return groupnorm_kernel


groupnorm_2g = make_groupnorm_kernel(2)


# ---------------------------------------------------------------------------
# Fused AdaGroupNorm + SiLU: GroupNorm + scale/shift modulation + SiLU
# Replaces: groupnorm -> add(scale,1) -> multiply -> add(shift) -> silu
# Uses intermediate DFB for SiLU to work around computed-intermediate bug.
# ---------------------------------------------------------------------------

def make_adaln_gn_silu_kernel(num_groups):

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


# ---------------------------------------------------------------------------
# Elementwise streaming helpers
# ---------------------------------------------------------------------------

@ttl.kernel(grid="auto")
def silu_kernel(x, out):
    """out = x * sigmoid(x)"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, out_dfb.reserve() as o:
                    o.store(xv * ttl.math.sigmoid(xv))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


@ttl.kernel(grid="auto")
def add_kernel(a, b, out):
    """out = a + b"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = a.shape[0] // TILE
    col_tiles = a.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                    o.store(av + bv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(a[row, col], blk); tx.wait()
                with b_dfb.reserve() as blk:
                    tx = ttl.copy(b[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


@ttl.kernel(grid="auto")
def mul_kernel(a, b, out):
    """out = a * b"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = a.shape[0] // TILE
    col_tiles = a.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                    o.store(av * bv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with a_dfb.reserve() as blk:
                    tx = ttl.copy(a[row, col], blk); tx.wait()
                with b_dfb.reserve() as blk:
                    tx = ttl.copy(b[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# ---------------------------------------------------------------------------
# Fused AdaLN modulation + SiLU: out = silu(x * (1 + scale) + shift)
# Replaces: add(scale,1) -> multiply(x,_) -> add(_,shift) -> silu
# shift and scale are pre-expanded to match x shape on host side.
# Uses intermediate DFB for SiLU (workaround for computed-intermediate bug).
# ---------------------------------------------------------------------------

@ttl.kernel(grid="auto")
def adaln_silu_kernel(x, shift, scale, out):
    """out = silu(x * (1 + scale) + shift)

    x, out: [seq_tiles*TILE, C] - full spatial data
    shift, scale: [TILE, C] - per-channel, broadcast across spatial dim
    """
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    sh_dfb = ttl.make_dataflow_buffer_like(shift, shape=(1, 1), buffer_factor=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, sh_dfb.wait() as shv, sc_dfb.wait() as scv, tmp_dfb.reserve() as tmp:
                    tmp.store(xv * (scv + ttl.math.fill(scv, 1.0)) + shv)
                with tmp_dfb.wait() as tv, out_dfb.reserve() as o:
                    o.store(tv * ttl.math.sigmoid(tv))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with sh_dfb.reserve() as blk:
                    tx = ttl.copy(shift[0, col], blk); tx.wait()
                with sc_dfb.reserve() as blk:
                    tx = ttl.copy(scale[0, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# ---------------------------------------------------------------------------
# Fused diffusion precondition: out = c_skip * noisy + c_out * model_output
# c_skip and c_out are scalar tensors pre-expanded to match spatial dims.
# ---------------------------------------------------------------------------

@ttl.kernel(grid="auto")
def precondition_kernel(noisy, model_out, c_skip, c_out, out):
    """out = c_skip * noisy + c_out * model_out"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = noisy.shape[0] // TILE
    col_tiles = noisy.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    n_dfb = ttl.make_dataflow_buffer_like(noisy, shape=(1, 1), buffer_factor=2)
    m_dfb = ttl.make_dataflow_buffer_like(model_out, shape=(1, 1), buffer_factor=2)
    cs_dfb = ttl.make_dataflow_buffer_like(c_skip, shape=(1, 1), buffer_factor=2)
    co_dfb = ttl.make_dataflow_buffer_like(c_out, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with n_dfb.wait() as nv, m_dfb.wait() as mv, cs_dfb.wait() as csv, co_dfb.wait() as cov, out_dfb.reserve() as o:
                    o.store(csv * nv + cov * mv)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with n_dfb.reserve() as blk:
                    tx = ttl.copy(noisy[row, col], blk); tx.wait()
                with m_dfb.reserve() as blk:
                    tx = ttl.copy(model_out[row, col], blk); tx.wait()
                with cs_dfb.reserve() as blk:
                    tx = ttl.copy(c_skip[row, col], blk); tx.wait()
                with co_dfb.reserve() as blk:
                    tx = ttl.copy(c_out[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()


# ---------------------------------------------------------------------------
# Fused Euler step: out = x + d * dt
# d = (x - denoised) / sigma_hat, so: out = x + ((x - denoised) / sigma) * dt
# For efficiency we fuse: out = x + dt_over_sigma * (x - denoised)
# dt_over_sigma is a scalar tensor pre-expanded.
# ---------------------------------------------------------------------------

@ttl.kernel(grid="auto")
def euler_step_kernel(x, denoised, dt_over_sigma, out):
    """out = x + dt_over_sigma * (x - denoised)"""
    grid_cols, _ = ttl.grid_size(dims=2)
    row_tiles = x.shape[0] // TILE
    col_tiles = x.shape[1] // TILE
    total_tiles = row_tiles * col_tiles
    tiles_per_core = -(-total_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    d_dfb = ttl.make_dataflow_buffer_like(denoised, shape=(1, 1), buffer_factor=2)
    s_dfb = ttl.make_dataflow_buffer_like(dt_over_sigma, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                with x_dfb.wait() as xv, d_dfb.wait() as dv, s_dfb.wait() as sv, out_dfb.reserve() as o:
                    o.store(xv + sv * (xv - dv))

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with x_dfb.reserve() as blk:
                    tx = ttl.copy(x[row, col], blk); tx.wait()
                with d_dfb.reserve() as blk:
                    tx = ttl.copy(denoised[row, col], blk); tx.wait()
                with s_dfb.reserve() as blk:
                    tx = ttl.copy(dt_over_sigma[row, col], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.core(dims=2)
        for local_t in range(tiles_per_core):
            t = core_x * tiles_per_core + local_t
            if t < total_tiles:
                row = t // col_tiles
                col = t % col_tiles
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[row, col]); tx.wait()
