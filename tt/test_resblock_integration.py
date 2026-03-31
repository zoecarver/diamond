"""Mini integration test: one resblock using adaln_silu_kernel."""
import torch
import torch.nn.functional as F
import ttnn
import sys
sys.path.insert(0, "/tmp")
from kernels import groupnorm_2g, adaln_silu_kernel

TILE = 32
GN_EPS = 1e-5
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0, l1_small_size=32768)

torch.manual_seed(42)
B, C, H, W = 1, 64, 64, 64
HW = H * W
num_groups = 2

# Input
x_nchw = torch.randn(B, C, H, W)

# AdaLN params
scale_1d = torch.randn(1, C) * 0.5
shift_1d = torch.randn(1, C) * 0.5

# PyTorch reference: group_norm -> modulate -> silu
normed = F.group_norm(x_nchw, num_groups, eps=GN_EPS)
modulated = normed * (1 + scale_1d[:, :, None, None]) + shift_1d[:, :, None, None]
ref = F.silu(modulated)
ref_2d = ref.permute(0, 2, 3, 1).reshape(HW, C)

# --- TT path: groupnorm_2g then adaln_silu_kernel ---
print("Step 1: groupnorm_2g...", flush=True)
x_2d = x_nchw.permute(0, 2, 3, 1).reshape(HW, C).to(torch.bfloat16)
x_tt = ttnn.from_torch(x_2d, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=device, memory_config=DRAM)

seq_tiles = HW // TILE
N = seq_tiles * TILE * TILE
scaler = ttnn.from_torch(
    torch.ones(TILE, TILE, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
mean_scale = ttnn.from_torch(
    torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
gn_out = ttnn.from_torch(
    torch.zeros(HW, C, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

groupnorm_2g(x_tt, scaler, mean_scale, gn_out)

# Check groupnorm
gn_torch = ttnn.to_torch(gn_out).reshape(HW, C)[:, :C].float()
normed_ref = normed.permute(0, 2, 3, 1).reshape(HW, C)
p_gn = pcc(normed_ref, gn_torch)
print(f"  GroupNorm PCC={p_gn:.6f}")

print("Step 2: adaln_silu_kernel...", flush=True)
scale_tt = ttnn.from_torch(
    scale_1d[0].to(torch.bfloat16).unsqueeze(0).expand(TILE, C).contiguous(),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
shift_tt = ttnn.from_torch(
    shift_1d[0].to(torch.bfloat16).unsqueeze(0).expand(TILE, C).contiguous(),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
adaln_out = ttnn.from_torch(
    torch.zeros(HW, C, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

adaln_silu_kernel(gn_out, shift_tt, scale_tt, adaln_out)

out_torch = ttnn.to_torch(adaln_out).reshape(HW, C)[:, :C].float()
p = pcc(ref_2d, out_torch)
maxdiff = (ref_2d.float() - out_torch).abs().max().item()
print(f"  Full (GN + AdaLN + SiLU) PCC={p:.6f}  maxdiff={maxdiff:.4f}")
print(f"  {'PASS' if p > 0.995 else 'FAIL'}")

ttnn.close_device(device)
