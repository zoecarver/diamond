"""
TT-Lang GroupNorm kernel for Diamond world model.

GroupNorm with num_groups groups, 1 tile per group (32 channels/group).
Three-pass algorithm:
  Pass 1: accumulate sum(x) -> mean
  Pass 2: accumulate sum((x - mean)^2) -> variance, then inv_std
  Pass 3: normalize (x - mean) * inv_std

Follows the RMSNorm pattern from toy-wm for DFB management.
"""

import ttl

TILE = 32


def make_groupnorm_kernel(num_groups):

    @ttl.kernel(grid=(1, 1))
    def groupnorm_kernel(x, scaler, mean_scale, out):
        """
        x:          [seq_tiles*32, num_groups*32] tensor
        scaler:     [32, 32] tile of ones
        mean_scale: [32, 32] tile filled with 1/(seq_tiles * 32 * 32)
        out:        same shape as x
        """
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
                    # === Pass 1: sum(x) -> mean ===
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

                    # === Pass 2+3 with mean_bc alive ===
                    with mean_bc_dfb.wait() as mean_bc:
                        # Pass 2: sum((x - mean)^2) -> variance -> inv_std
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

                        # Pass 3: normalize
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
