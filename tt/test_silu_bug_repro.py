"""Minimal reproducer: x * sigmoid(x) gives wrong results when x is a computed intermediate.

Works:    DFB.wait() as x -> x * sigmoid(x)           # x from DFB directly
Broken:   DFB.wait() as a, b -> y = a + b -> y * sigmoid(y)  # y is computed
Workaround: store y to intermediate DFB, read back, then apply SiLU
"""
import torch
import ttnn
import ttl

TILE = 32


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# --- Bug case: SiLU on computed intermediate ---

@ttl.kernel(grid=(1, 1))
def silu_on_computed_BROKEN(a, b, out):
    """out = silu(a + b) -- BROKEN: computed intermediate reuse with sigmoid"""
    tiles = a.shape[0] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        for t in range(tiles):
            with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
                y = av + bv
                o.store(y * ttl.math.sigmoid(y))

    @ttl.datamovement()
    def dm_read():
        for t in range(tiles):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[t, 0], blk); tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[t, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        for t in range(tiles):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[t, 0]); tx.wait()


# --- Workaround: intermediate DFB ---

@ttl.kernel(grid=(1, 1))
def silu_on_computed_FIXED(a, b, out):
    """out = silu(a + b) -- FIXED: intermediate DFB materializes computed value"""
    tiles = a.shape[0] // TILE

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    tmp_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        for t in range(tiles):
            with a_dfb.wait() as av, b_dfb.wait() as bv, tmp_dfb.reserve() as tmp:
                tmp.store(av + bv)
            with tmp_dfb.wait() as tv, out_dfb.reserve() as o:
                o.store(tv * ttl.math.sigmoid(tv))

    @ttl.datamovement()
    def dm_read():
        for t in range(tiles):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[t, 0], blk); tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[t, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        for t in range(tiles):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[t, 0]); tx.wait()


# --- Reference: SiLU on direct DFB value (works) ---

@ttl.kernel(grid=(1, 1))
def silu_on_dfb_WORKS(x, out):
    """out = silu(x) -- WORKS: x directly from DFB"""
    tiles = x.shape[0] // TILE

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        for t in range(tiles):
            with x_dfb.wait() as xv, out_dfb.reserve() as o:
                o.store(xv * ttl.math.sigmoid(xv))

    @ttl.datamovement()
    def dm_read():
        for t in range(tiles):
            with x_dfb.reserve() as blk:
                tx = ttl.copy(x[t, 0], blk); tx.wait()

    @ttl.datamovement()
    def dm_write():
        for t in range(tiles):
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[t, 0]); tx.wait()


# --- Test ---

device = ttnn.open_device(device_id=0, l1_small_size=32768)
DRAM = ttnn.DRAM_MEMORY_CONFIG

torch.manual_seed(42)
N = 128  # 4 tiles
a_t = torch.randn(N, TILE, dtype=torch.bfloat16)
b_t = torch.randn(N, TILE, dtype=torch.bfloat16)

# Reference
sum_ab = (a_t.float() + b_t.float())
ref_silu = sum_ab * torch.sigmoid(sum_ab)

def to_dev(t):
    return ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)

a_tt = to_dev(a_t)
b_tt = to_dev(b_t)

# Test 1: BROKEN - SiLU on computed intermediate
out1 = to_dev(torch.zeros_like(a_t))
silu_on_computed_BROKEN(a_tt, b_tt, out1)
r1 = ttnn.to_torch(out1).reshape(N, TILE)[:, :TILE].float()
p1 = pcc(ref_silu, r1)
print(f"BROKEN (computed intermediate): PCC={p1:.6f}")

# Test 2: FIXED - intermediate DFB workaround
out2 = to_dev(torch.zeros_like(a_t))
silu_on_computed_FIXED(a_tt, b_tt, out2)
r2 = ttnn.to_torch(out2).reshape(N, TILE)[:, :TILE].float()
p2 = pcc(ref_silu, r2)
print(f"FIXED  (intermediate DFB):      PCC={p2:.6f}")

# Test 3: WORKS - SiLU on direct DFB value
sum_tt = to_dev((a_t.float() + b_t.float()).to(torch.bfloat16))
out3 = to_dev(torch.zeros_like(a_t))
silu_on_dfb_WORKS(sum_tt, out3)
r3 = ttnn.to_torch(out3).reshape(N, TILE)[:, :TILE].float()
p3 = pcc(ref_silu, r3)
print(f"WORKS  (direct DFB value):      PCC={p3:.6f}")

ttnn.close_device(device)
