"""Test adaln_silu_kernel: out = silu(x * (1 + scale) + shift), grid="auto"."""
import torch
import torch.nn.functional as F
import ttnn
import sys
sys.path.insert(0, "/tmp")
from kernels import adaln_silu_kernel

TILE = 32


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0, l1_small_size=32768)
DRAM = ttnn.DRAM_MEMORY_CONFIG

torch.manual_seed(42)
H, W, C = 64, 64, 64
HW = H * W

x_2d = torch.randn(HW, C, dtype=torch.bfloat16)
scale_1d = torch.randn(1, C) * 0.5
shift_1d = torch.randn(1, C) * 0.5

# PyTorch reference
x_f = x_2d.float()
s_f = scale_1d.float()
h_f = shift_1d.float()
modulated = x_f * (1 + s_f) + h_f
ref = modulated * torch.sigmoid(modulated)  # SiLU

# TT tensors
x_tt = ttnn.from_torch(x_2d, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=device, memory_config=DRAM)
# scale/shift as [TILE, C] broadcast across rows
scale_tt = ttnn.from_torch(
    scale_1d[0].to(torch.bfloat16).unsqueeze(0).expand(TILE, C).contiguous(),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
shift_tt = ttnn.from_torch(
    shift_1d[0].to(torch.bfloat16).unsqueeze(0).expand(TILE, C).contiguous(),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
out_tt = ttnn.from_torch(
    torch.zeros(HW, C, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

print("Running adaln_silu_kernel...", flush=True)
adaln_silu_kernel(x_tt, shift_tt, scale_tt, out_tt)

out_torch = ttnn.to_torch(out_tt).reshape(HW, C)[:, :C].float()
p = pcc(ref, out_torch)
maxdiff = (ref - out_torch).abs().max().item()
print(f"PCC={p:.6f}  maxdiff={maxdiff:.4f}")
print(f"{'PASS' if p > 0.995 else 'FAIL'}")

ttnn.close_device(device)
