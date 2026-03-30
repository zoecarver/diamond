"""Test TT-Lang GroupNorm kernel against PyTorch reference."""
import torch
import torch.nn.functional as F
import ttnn

TILE = 32
EPS = 1e-5
NUM_GROUPS = 2
CHANNELS = 64

device = ttnn.open_device(device_id=0)

from groupnorm_kernel import groupnorm_2g


def to_tt(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_groupnorm(spatial_h, spatial_w):
    hw = spatial_h * spatial_w
    seq_tiles = hw // TILE
    N = seq_tiles * TILE * TILE  # elements per group tile column

    torch.manual_seed(42)
    x_2d = torch.randn(hw, CHANNELS, dtype=torch.bfloat16)

    # PyTorch reference: reshape to [1, C, H*W] for group_norm
    x_nchw = x_2d.float().T.unsqueeze(0).unsqueeze(-1)  # [1, C, H*W, 1]
    x_nchw = x_nchw.reshape(1, CHANNELS, hw)
    ref_nchw = F.group_norm(x_nchw, NUM_GROUPS, eps=EPS)
    ref = ref_nchw.reshape(CHANNELS, hw).T.to(torch.bfloat16)  # [H*W, C]

    # TT-Lang kernel
    x_tt = to_tt(x_2d)
    out_tt = to_tt(torch.zeros(hw, CHANNELS, dtype=torch.bfloat16))
    scaler_tt = to_tt(torch.ones(TILE, TILE, dtype=torch.bfloat16))
    mean_scale_tt = to_tt(torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16))

    groupnorm_2g(x_tt, scaler_tt, mean_scale_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    max_diff = (result.float() - ref.float()).abs().max().item()
    mean_diff = (result.float() - ref.float()).abs().mean().item()

    status = "PASS" if max_diff < 1.0 else "FAIL"
    print(f"GroupNorm {spatial_h}x{spatial_w} (seq_tiles={seq_tiles}): "
          f"max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} [{status}]")
    if max_diff >= 1.0:
        print(f"  ref[0,:5]:    {ref[0,:5].tolist()}")
        print(f"  result[0,:5]: {result[0,:5].tolist()}")
    return status == "PASS"


print("Testing TT-Lang GroupNorm kernel (2 groups, 64 channels)")
print("=" * 60)

results = []
for h, w in [(8, 8), (16, 16), (32, 32), (64, 64)]:
    try:
        ok = test_groupnorm(h, w)
        results.append((f"{h}x{w}", ok))
    except Exception as e:
        print(f"GroupNorm {h}x{w}: ERROR - {e}")
        results.append((f"{h}x{w}", False))

print("\n" + "=" * 60)
print("Summary:")
for name, ok in results:
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")

all_pass = all(ok for _, ok in results)
print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")

ttnn.close_device(device)
