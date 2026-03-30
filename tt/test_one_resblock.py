"""Test a single ResBlock with real weights to isolate the crash."""
import sys
import math
import torch
import torch.nn.functional as F
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
GN_EPS = 1e-5

print("Downloading weights...", flush=True)
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="eloialonso/diamond",
                       filename="atari_100k/models/Breakout.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)
sd = {k[len("denoiser."):]: v.float() for k, v in ckpt.items() if k.startswith("denoiser.")}
print(f"Loaded {len(sd)} params", flush=True)

print("Opening device...", flush=True)
device = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = device.compute_with_storage_grid_size()
core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
print(f"Device open, grid={grid}", flush=True)

conv_config = ttnn.Conv2dConfig(
    weights_dtype=ttnn.bfloat16,
    shard_layout=None,
    deallocate_activation=False,
    output_layout=ttnn.TILE_LAYOUT,
)
compute_config = ttnn.init_device_compute_kernel_config(
    device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
)


def tt_host(t):
    return ttnn.from_torch(t, ttnn.bfloat16)

def tt_dev(t):
    return ttnn.from_torch(t.to(torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)


# Test conv_in first
print("\n--- conv_in ---", flush=True)
B, H, W = 1, 64, 64
x_nchw = torch.randn(B, 15, H, W, dtype=torch.bfloat16).float()
x_nhwc = x_nchw.permute(0, 2, 3, 1)
x_host = tt_host(x_nhwc)

conv_in_w = tt_host(sd["inner_model.conv_in.weight"])
conv_in_b = tt_host(sd["inner_model.conv_in.bias"].reshape(1, 1, 1, -1))

[x_tt, [oh, ow], _] = ttnn.conv2d(
    input_tensor=x_host, weight_tensor=conv_in_w,
    in_channels=15, out_channels=64, device=device,
    bias_tensor=conv_in_b, kernel_size=(3, 3), stride=(1, 1),
    padding=(1, 1), dilation=(1, 1), batch_size=B,
    input_height=H, input_width=W,
    conv_config=conv_config, compute_config=compute_config,
    groups=1, return_output_dim=True, return_weights_and_bias=True,
)
print(f"  conv_in output: {oh}x{ow}, shape={x_tt.shape}", flush=True)

# Test group_norm
print("\n--- group_norm (no weight/bias, like AdaGroupNorm) ---", flush=True)
try:
    normed = ttnn.group_norm(x_tt, num_groups=2, epsilon=GN_EPS,
                              inplace=False, core_grid=core_grid)
    print(f"  group_norm output: {normed.shape}", flush=True)
except Exception as e:
    print(f"  group_norm FAILED: {e}", flush=True)
    normed = None

# Test group_norm with weight/bias
print("\n--- group_norm (with weight/bias, like norm_out) ---", flush=True)
try:
    gn_w = tt_dev(sd["inner_model.norm_out.norm.weight"].reshape(1, 1, 1, -1))
    gn_b = tt_dev(sd["inner_model.norm_out.norm.bias"].reshape(1, 1, 1, -1))
    normed2 = ttnn.group_norm(x_tt, num_groups=2, epsilon=GN_EPS,
                               weight=gn_w, bias=gn_b,
                               inplace=False, core_grid=core_grid)
    print(f"  group_norm+wb output: {normed2.shape}", flush=True)
except Exception as e:
    print(f"  group_norm+wb FAILED: {e}", flush=True)

# Test silu
print("\n--- silu ---", flush=True)
if normed is not None:
    act = ttnn.silu(normed)
    print(f"  silu output: {act.shape}", flush=True)

# Test second conv (resblock conv1)
print("\n--- conv1 (64->64, 64x64) ---", flush=True)
if normed is not None:
    act = ttnn.silu(normed)
    c1_w = tt_host(sd["inner_model.unet.d_blocks.0.resblocks.0.conv1.weight"])
    c1_b = tt_host(sd["inner_model.unet.d_blocks.0.resblocks.0.conv1.bias"].reshape(1, 1, 1, -1))
    [c1_out, [ch, cw], _] = ttnn.conv2d(
        input_tensor=act, weight_tensor=c1_w,
        in_channels=64, out_channels=64, device=device,
        bias_tensor=c1_b, kernel_size=(3, 3), stride=(1, 1),
        padding=(1, 1), dilation=(1, 1), batch_size=B,
        input_height=oh, input_width=ow,
        conv_config=conv_config, compute_config=compute_config,
        groups=1, return_output_dim=True, return_weights_and_bias=True,
    )
    print(f"  conv1 output: {ch}x{cw}, shape={c1_out.shape}", flush=True)

# Test residual add
print("\n--- residual add ---", flush=True)
if normed is not None:
    result = ttnn.add(c1_out, x_tt)
    print(f"  add output: {result.shape}", flush=True)
    out = ttnn.to_torch(result)
    print(f"  values: [{out.min():.3f}, {out.max():.3f}]", flush=True)

print("\nALL DONE", flush=True)
ttnn.close_device(device)
