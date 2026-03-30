"""Isolate AdaGroupNorm modulation PCC issue."""
import math
import torch
import torch.nn.functional as F
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG

device = ttnn.open_device(device_id=0, l1_small_size=32768)


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def tt_dev(t):
    return ttnn.from_torch(t.to(torch.bfloat16), ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


torch.manual_seed(42)

# Simulate what adaln_modulate does
# x: [1, 1, 4096, 64] on device (TILE_LAYOUT)
# scale, shift: computed from cond, shape [1, 64] -> [1, 1, 1, 64]

x_nchw = torch.randn(1, 64, 64, 64)  # [N, C, H, W]
scale_1d = torch.randn(1, 64) * 0.5
shift_1d = torch.randn(1, 64) * 0.5

# PyTorch reference (NCHW)
ref = x_nchw * (1 + scale_1d[:, :, None, None]) + shift_1d[:, :, None, None]

# TT path: x is [1, 1, H*W, C]
x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, 4096, 64)
x_tt = tt_dev(x_nhwc)

# scale/shift as [1, 1, 1, 64]
scale_4d = scale_1d.unsqueeze(1).unsqueeze(1).to(torch.bfloat16)  # [1, 1, 1, 64]
shift_4d = shift_1d.unsqueeze(1).unsqueeze(1).to(torch.bfloat16)  # [1, 1, 1, 64]

print(f"x_tt shape: {x_tt.shape}")
print(f"scale_4d shape: {scale_4d.shape}")
print(f"shift_4d shape: {shift_4d.shape}")

scale_tt = tt_dev(scale_4d)
shift_tt = tt_dev(shift_4d)

print(f"scale_tt shape: {scale_tt.shape}")
print(f"shift_tt shape: {shift_tt.shape}")

# Method 1: current approach
one_plus_scale = ttnn.add(scale_tt, 1.0)
mod = ttnn.multiply(x_tt, one_plus_scale)
out1_tt = ttnn.add(mod, shift_tt)
out1 = ttnn.to_torch(out1_tt).reshape(1, 64, 64, 64).permute(0, 3, 1, 2)[:, :64, :, :]
# Wait - out1 is [1, 1, 4096, 64] -> reshape to [1, 64, 64, 64] but that's NHWC
# Need [1, 64, 64, -1] then permute
out1_raw = ttnn.to_torch(out1_tt)
print(f"out1_raw shape: {out1_raw.shape}")
out1_nhwc = out1_raw.reshape(1, 64, 64, 64)  # [N, H, W, C]
out1_nchw = out1_nhwc.permute(0, 3, 1, 2)
p1 = pcc(ref, out1_nchw)
print(f"\nMethod 1 (current): PCC={p1:.6f}  maxdiff={(ref.float()-out1_nchw.float()).abs().max():.4f}")

# Let's check what scale/shift look like after round-trip
scale_back = ttnn.to_torch(scale_tt)
shift_back = ttnn.to_torch(shift_tt)
print(f"\nscale_tt readback shape: {scale_back.shape}")
print(f"scale ref[:5]:  {scale_1d[0, :5].tolist()}")
print(f"scale tt[:5]:   {scale_back.flatten()[:5].tolist()}")

# Check: does broadcasting work correctly?
# scale_tt is [1, 1, 1, 64]. x_tt is [1, 1, 4096, 64].
# ttnn.multiply should broadcast dim 2: 1 -> 4096
# Let's verify with a simpler test
print("\n--- Simple broadcast test ---")
a = tt_dev(torch.ones(1, 1, 4096, 64, dtype=torch.bfloat16) * 2.0)
b = tt_dev(torch.ones(1, 1, 1, 64, dtype=torch.bfloat16) * 3.0)
c = ttnn.multiply(a, b)
c_torch = ttnn.to_torch(c)
print(f"2.0 * 3.0 broadcast: expected 6.0, got {c_torch[0, 0, 0, 0]:.1f} and {c_torch[0, 0, 2000, 32]:.1f}")

# Check: what if scale_tt shape doesn't broadcast as expected?
# Maybe the issue is that [1,1,1,64] doesn't broadcast to [1,1,4096,64] in TILE_LAYOUT
# because tile layout requires dim sizes to be multiples of 32?
# [1,1,1,64] in TILE_LAYOUT might actually be [1,1,32,64] (padded)
print(f"\nscale_tt TILE shape: {scale_tt.shape}")

# Method 2: expand scale/shift on host to match x shape before sending to device
scale_exp = scale_1d.unsqueeze(2).unsqueeze(3).expand_as(x_nchw)  # [1, 64, 64, 64]
shift_exp = shift_1d.unsqueeze(2).unsqueeze(3).expand_as(x_nchw)
# Convert to NHWC [1, 1, 4096, 64]
scale_nhwc = scale_exp.permute(0, 2, 3, 1).reshape(1, 1, 4096, 64)
shift_nhwc = shift_exp.permute(0, 2, 3, 1).reshape(1, 1, 4096, 64)
scale2_tt = tt_dev(scale_nhwc)
shift2_tt = tt_dev(shift_nhwc)

one_plus_scale2 = ttnn.add(scale2_tt, 1.0)
mod2 = ttnn.multiply(x_tt, one_plus_scale2)
out2_tt = ttnn.add(mod2, shift2_tt)
out2_raw = ttnn.to_torch(out2_tt)
out2_nhwc = out2_raw.reshape(1, 64, 64, 64)
out2_nchw = out2_nhwc.permute(0, 3, 1, 2)
p2 = pcc(ref, out2_nchw)
print(f"\nMethod 2 (pre-expanded): PCC={p2:.6f}  maxdiff={(ref.float()-out2_nchw.float()).abs().max():.4f}")

print("\nDone.")
ttnn.close_device(device)
