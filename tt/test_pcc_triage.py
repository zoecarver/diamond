"""Layer-by-layer PCC comparison: TT implementation vs PyTorch reference."""
import math
import torch
import torch.nn.functional as F
import ttnn
import sys
sys.path.insert(0, "/tmp")
from kernels import groupnorm_2g

SIGMA_DATA = 0.5
SIGMA_OFFSET_NOISE = 0.3
COND_CH = 256
GN_EPS = 1e-5
GN_GROUP_SIZE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def compare(name, ref_nchw, tt_tensor, batch=1, h=64, w=64):
    """Compare PyTorch NCHW reference to TT [N, 1, H*W, C] tensor."""
    tt_raw = ttnn.to_torch(tt_tensor)
    c = ref_nchw.shape[1]
    # TT output is [N, 1, H*W, C] -> reshape to [N, C, H, W]
    tt_nhwc = tt_raw.reshape(batch, h, w, -1)[:, :, :, :c]
    tt_nchw = tt_nhwc.permute(0, 3, 1, 2)
    p = pcc(ref_nchw, tt_nchw)
    maxdiff = (ref_nchw.float() - tt_nchw.float()).abs().max().item()
    print(f"  {name}: PCC={p:.6f}  maxdiff={maxdiff:.4f}  shape={list(ref_nchw.shape)}")
    return p, tt_nchw


# ---- Setup ----
print("Loading weights...", flush=True)
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/models/Breakout.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)
sd = {k[len("denoiser."):]: v.float() for k, v in ckpt.items() if k.startswith("denoiser.")}
num_actions = sd["inner_model.act_emb.0.weight"].shape[0]

device = ttnn.open_device(device_id=0, l1_small_size=32768)
conv_config = ttnn.Conv2dConfig(
    weights_dtype=ttnn.bfloat16, shard_layout=None,
    deallocate_activation=False, output_layout=ttnn.TILE_LAYOUT)
compute_config = ttnn.init_device_compute_kernel_config(
    device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4)

def tt_host(t):
    return ttnn.from_torch(t, ttnn.bfloat16)

def tt_dev(t):
    return ttnn.from_torch(t.to(torch.bfloat16), ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

def tt_conv2d(x, w, b, in_ch, out_ch, batch, h, w_):
    [out, [oh, ow], _] = ttnn.conv2d(
        input_tensor=x, weight_tensor=w, in_channels=in_ch, out_channels=out_ch,
        device=device, bias_tensor=b, kernel_size=(3, 3), stride=(1, 1),
        padding=(1, 1), dilation=(1, 1), batch_size=batch, input_height=h, input_width=w_,
        conv_config=conv_config, compute_config=compute_config,
        groups=1, return_output_dim=True, return_weights_and_bias=True)
    return out, oh, ow

def tt_conv2d_1x1(x, w, b, in_ch, out_ch, batch, h, w_):
    [out, [oh, ow], _] = ttnn.conv2d(
        input_tensor=x, weight_tensor=w, in_channels=in_ch, out_channels=out_ch,
        device=device, bias_tensor=b, kernel_size=(1, 1), stride=(1, 1),
        padding=(0, 0), dilation=(1, 1), batch_size=batch, input_height=h, input_width=w_,
        conv_config=conv_config, compute_config=compute_config,
        groups=1, return_output_dim=True, return_weights_and_bias=True)
    return out, oh, ow

def tt_group_norm(x_tt, num_groups):
    """TT-Lang GroupNorm on [N, 1, H*W, C] tensor."""
    shape = x_tt.shape
    hw = shape[2] if len(shape) == 4 else shape[0]
    channels = shape[3] if len(shape) == 4 else shape[1]
    x_2d = ttnn.reshape(x_tt, [hw, channels])
    seq_tiles = hw // TILE
    N = seq_tiles * TILE * TILE
    scaler = ttnn.from_torch(torch.ones(TILE, TILE, dtype=torch.bfloat16),
                             ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    mean_scale = ttnn.from_torch(torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16),
                                 ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    out_2d = ttnn.from_torch(torch.zeros(hw, channels, dtype=torch.bfloat16),
                             ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    groupnorm_2g(x_2d, scaler, mean_scale, out_2d)
    ttnn.deallocate(scaler)
    ttnn.deallocate(mean_scale)
    return ttnn.reshape(out_2d, shape)


# ---- Create identical inputs ----
torch.manual_seed(42)
B = 1
H, W = 64, 64
prev_obs = torch.randn(B, 4, 3, H, W)
prev_act = torch.randint(0, num_actions, (B, 4))
obs_flat = prev_obs.reshape(B, -1, H, W)  # [1, 12, 64, 64]

sigma = torch.tensor([5.0])  # first sigma
sigma_adj = (sigma**2 + SIGMA_OFFSET_NOISE**2).sqrt()
c_in = 1 / (sigma_adj**2 + SIGMA_DATA**2).sqrt()
c_skip = SIGMA_DATA**2 / (sigma_adj**2 + SIGMA_DATA**2)
c_out = sigma_adj * c_skip.sqrt()
c_noise = sigma_adj.log() / 4

torch.manual_seed(123)
noisy = torch.randn(B, 3, H, W)
rescaled_obs = obs_flat / SIGMA_DATA
rescaled_noise = noisy * c_in[:, None, None, None]

# Conditioning (host-side, identical for both)
def fourier_features(sigma, weight):
    f = 2 * math.pi * sigma.unsqueeze(1) @ weight
    return torch.cat([f.cos(), f.sin()], dim=-1)

noise_emb = fourier_features(c_noise, sd["inner_model.noise_emb.weight"])
act_emb = sd["inner_model.act_emb.0.weight"][prev_act.long()].reshape(B, -1)
combined = noise_emb + act_emb
cond = F.linear(combined, sd["inner_model.cond_proj.0.weight"],
                sd["inner_model.cond_proj.0.bias"])
cond = F.silu(cond)
cond = F.linear(cond, sd["inner_model.cond_proj.2.weight"],
                sd["inner_model.cond_proj.2.bias"])

print(f"\nConditioning: shape={cond.shape}, range=[{cond.min():.3f}, {cond.max():.3f}]")

# ---- conv_in ----
print("\n=== conv_in ===", flush=True)
cat_input = torch.cat((rescaled_obs, rescaled_noise), dim=1)  # [1, 15, 64, 64]

# PyTorch reference
ref_conv_in = F.conv2d(cat_input, sd["inner_model.conv_in.weight"],
                       sd["inner_model.conv_in.bias"], padding=1)

# TT
cat_nhwc = cat_input.permute(0, 2, 3, 1).to(torch.bfloat16)
x_host = tt_host(cat_nhwc)
conv_in_w = tt_host(sd["inner_model.conv_in.weight"])
conv_in_b = tt_host(sd["inner_model.conv_in.bias"].reshape(1, 1, 1, -1))
x_tt, oh, ow = tt_conv2d(x_host, conv_in_w, conv_in_b, 15, 64, B, H, W)

p_conv_in, tt_conv_in_nchw = compare("conv_in", ref_conv_in, x_tt)

# ---- First ResBlock: d_blocks.0.resblocks.0 ----
prefix = "inner_model.unet.d_blocks.0.resblocks.0"
in_ch, out_ch = 64, 64
ng = max(1, in_ch // GN_GROUP_SIZE)

print(f"\n=== ResBlock {prefix} (64->64, 64x64) ===", flush=True)

# --- norm1 (AdaGroupNorm) ---
# PyTorch: group_norm then modulate
ref_gn1 = F.group_norm(ref_conv_in, ng, eps=GN_EPS)

# TT GroupNorm
gn1_tt = tt_group_norm(x_tt, ng)
p_gn1, tt_gn1_nchw = compare("norm1 (group_norm only)", ref_gn1, gn1_tt)

# Modulation: scale, shift from linear
scale_shift = F.linear(cond, sd[f"{prefix}.norm1.linear.weight"],
                       sd[f"{prefix}.norm1.linear.bias"])
scale, shift = scale_shift.chunk(2, dim=-1)
ref_adaln1 = ref_gn1 * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

# TT modulation (NHWC: scale is [B, C] -> [B, 1, 1, C])
scale_4d = scale.unsqueeze(1).unsqueeze(1).to(torch.bfloat16)
shift_4d = shift.unsqueeze(1).unsqueeze(1).to(torch.bfloat16)
scale_tt = tt_dev(scale_4d)
shift_tt = tt_dev(shift_4d)
one_plus_scale = ttnn.add(scale_tt, 1.0)
mod_tt = ttnn.multiply(gn1_tt, one_plus_scale)
adaln1_tt = ttnn.add(mod_tt, shift_tt)
p_adaln1, tt_adaln1_nchw = compare("norm1 (adaln full)", ref_adaln1, adaln1_tt)

# --- silu ---
ref_silu1 = F.silu(ref_adaln1)
silu1_tt = ttnn.silu(adaln1_tt)
p_silu1, tt_silu1_nchw = compare("silu after norm1", ref_silu1, silu1_tt)

# --- conv1 ---
ref_conv1 = F.conv2d(ref_silu1, sd[f"{prefix}.conv1.weight"],
                     sd[f"{prefix}.conv1.bias"], padding=1)
c1_w = tt_host(sd[f"{prefix}.conv1.weight"])
c1_b = tt_host(sd[f"{prefix}.conv1.bias"].reshape(1, 1, 1, -1))
conv1_tt, c1h, c1w = tt_conv2d(silu1_tt, c1_w, c1_b, in_ch, out_ch, B, oh, ow)
p_conv1, tt_conv1_nchw = compare("conv1", ref_conv1, conv1_tt, h=c1h, w=c1w)

# --- norm2 + modulate + silu ---
ref_gn2 = F.group_norm(ref_conv1, ng, eps=GN_EPS)
gn2_tt = tt_group_norm(conv1_tt, ng)
p_gn2, _ = compare("norm2 (group_norm only)", ref_gn2, gn2_tt, h=c1h, w=c1w)

scale_shift2 = F.linear(cond, sd[f"{prefix}.norm2.linear.weight"],
                        sd[f"{prefix}.norm2.linear.bias"])
scale2, shift2 = scale_shift2.chunk(2, dim=-1)
ref_adaln2 = ref_gn2 * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
ref_silu2 = F.silu(ref_adaln2)

# TT
scale2_tt = tt_dev(scale2.unsqueeze(1).unsqueeze(1).to(torch.bfloat16))
shift2_tt = tt_dev(shift2.unsqueeze(1).unsqueeze(1).to(torch.bfloat16))
one_plus_s2 = ttnn.add(scale2_tt, 1.0)
mod2_tt = ttnn.multiply(gn2_tt, one_plus_s2)
adaln2_tt = ttnn.add(mod2_tt, shift2_tt)
silu2_tt = ttnn.silu(adaln2_tt)
p_silu2, _ = compare("silu after norm2", ref_silu2, silu2_tt, h=c1h, w=c1w)

# --- conv2 ---
ref_conv2 = F.conv2d(ref_silu2, sd[f"{prefix}.conv2.weight"],
                     sd[f"{prefix}.conv2.bias"], padding=1)
c2_w = tt_host(sd[f"{prefix}.conv2.weight"])
c2_b = tt_host(sd[f"{prefix}.conv2.bias"].reshape(1, 1, 1, -1))
conv2_tt, c2h, c2w = tt_conv2d(silu2_tt, c2_w, c2_b, out_ch, out_ch, B, c1h, c1w)
p_conv2, _ = compare("conv2", ref_conv2, conv2_tt, h=c2h, w=c2w)

# --- residual ---
ref_resblock = ref_conv2 + ref_conv_in  # skip = identity (64->64)
res_tt = ttnn.add(conv2_tt, x_tt)
p_res, _ = compare("resblock output", ref_resblock, res_tt, h=c2h, w=c2w)

# ---- Also test norm_out at the end ----
print(f"\n=== norm_out (GroupNorm + weight/bias) ===", flush=True)
# Use conv_in output as a proxy (it's 64ch at 64x64)
ref_norm_out = F.group_norm(ref_conv_in, ng,
                            sd["inner_model.norm_out.norm.weight"],
                            sd["inner_model.norm_out.norm.bias"], eps=GN_EPS)

gn_out_tt = tt_group_norm(x_tt, ng)
# Apply weight and bias
gn_w = ttnn.from_torch(
    sd["inner_model.norm_out.norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, -1),
    ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
gn_b = ttnn.from_torch(
    sd["inner_model.norm_out.norm.bias"].to(torch.bfloat16).reshape(1, 1, 1, -1),
    ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
# Need TILE layout for multiply/add
gn_w_tile = ttnn.to_layout(gn_w, ttnn.TILE_LAYOUT)
gn_b_tile = ttnn.to_layout(gn_b, ttnn.TILE_LAYOUT)
norm_out_tt = ttnn.multiply(gn_out_tt, gn_w_tile)
norm_out_tt = ttnn.add(norm_out_tt, gn_b_tile)
p_norm_out, _ = compare("norm_out (gn + w + b)", ref_norm_out, norm_out_tt)

# ---- Summary ----
print("\n" + "=" * 60)
print("PCC Summary:")
print(f"  conv_in:          {p_conv_in:.6f}")
print(f"  group_norm:       {p_gn1:.6f}")
print(f"  adaln_modulate:   {p_adaln1:.6f}")
print(f"  silu:             {p_silu1:.6f}")
print(f"  conv1:            {p_conv1:.6f}")
print(f"  group_norm2:      {p_gn2:.6f}")
print(f"  silu2:            {p_silu2:.6f}")
print(f"  conv2:            {p_conv2:.6f}")
print(f"  resblock:         {p_res:.6f}")
print(f"  norm_out:         {p_norm_out:.6f}")
print("=" * 60)

ttnn.close_device(device)
