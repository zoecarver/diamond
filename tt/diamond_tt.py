"""
Diamond world model inference on Tenstorrent hardware.

UNet-based diffusion denoiser from Diamond (NeurIPS 2024).
- TTNN: conv2d, group_norm, silu, upsample, concat, linear
- TT-Lang: fused AdaLN modulate, fused precondition, fused Euler step
- Pretrained Breakout weights from HuggingFace
"""

import math
import sys
import time
import torch
import torch.nn.functional as F
import ttnn

sys.path.insert(0, "/tmp")
from kernels import groupnorm_2g

# ---------------------------------------------------------------------------
# Config (matches Diamond default for Atari)
# ---------------------------------------------------------------------------

SIGMA_DATA = 0.5
SIGMA_OFFSET_NOISE = 0.3
NUM_COND = 4           # num_steps_conditioning
IMG_CH = 3
COND_CH = 256
CHANNELS = [64, 64, 64, 64]
DEPTHS = [2, 2, 2, 2]
GN_EPS = 1e-5
GN_GROUP_SIZE = 32
IMG_SIZE = 64

NUM_DENOISE_STEPS = 3
SIGMA_MIN = 2e-3
SIGMA_MAX = 5.0
RHO = 7

DRAM = ttnn.DRAM_MEMORY_CONFIG


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def download_weights(game="Breakout"):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="eloialonso/diamond",
                           filename=f"atari_100k/models/{game}.pt")


def load_denoiser_sd(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "denoiser."
    return {k[len(prefix):]: v.float() for k, v in ckpt.items() if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# Host-side helpers (small ops not worth device round-trip)
# ---------------------------------------------------------------------------

def fourier_features(sigma, weight):
    f = 2 * math.pi * sigma.unsqueeze(1) @ weight
    return torch.cat([f.cos(), f.sin()], dim=-1)


def compute_conditioning(sigma, prev_act, sd, num_actions):
    b = sigma.shape[0]
    noise_emb = fourier_features(sigma, sd["inner_model.noise_emb.weight"])
    act_emb = sd["inner_model.act_emb.0.weight"][prev_act.long()]
    act_emb = act_emb.reshape(b, -1)
    combined = noise_emb + act_emb
    cond = F.linear(combined, sd["inner_model.cond_proj.0.weight"],
                     sd["inner_model.cond_proj.0.bias"])
    cond = F.silu(cond)
    cond = F.linear(cond, sd["inner_model.cond_proj.2.weight"],
                     sd["inner_model.cond_proj.2.bias"])
    return cond


def compute_conditioners(sigma):
    sigma_adj = (sigma**2 + SIGMA_OFFSET_NOISE**2).sqrt()
    c_in = 1 / (sigma_adj**2 + SIGMA_DATA**2).sqrt()
    c_skip = SIGMA_DATA**2 / (sigma_adj**2 + SIGMA_DATA**2)
    c_out = sigma_adj * c_skip.sqrt()
    c_noise = sigma_adj.log() / 4
    return c_in, c_out, c_skip, c_noise


def build_sigmas():
    min_inv_rho = SIGMA_MIN ** (1 / RHO)
    max_inv_rho = SIGMA_MAX ** (1 / RHO)
    l = torch.linspace(0, 1, NUM_DENOISE_STEPS)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** RHO
    return torch.cat((sigmas, sigmas.new_zeros(1)))


# ---------------------------------------------------------------------------
# TTNN wrappers
# ---------------------------------------------------------------------------

def tt_host(t, dtype=ttnn.bfloat16):
    """Convert torch tensor to host-side TTNN tensor (for conv2d weights)."""
    return ttnn.from_torch(t, dtype)


def tt_dev(t, device, layout=ttnn.TILE_LAYOUT, mem=DRAM):
    """Convert torch tensor to device TTNN tensor."""
    return ttnn.from_torch(t, ttnn.bfloat16, layout=layout,
                           device=device, memory_config=mem)


def tt_conv2d(x, w_tt, b_tt, device, in_ch, out_ch, batch, h, w,
              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
              conv_config=None, compute_config=None):
    """Run conv2d. Returns (output_tensor, out_h, out_w)."""
    [out, [oh, ow], [wd, bd]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=w_tt,
        in_channels=in_ch,
        out_channels=out_ch,
        device=device,
        bias_tensor=b_tt,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        batch_size=batch,
        input_height=h,
        input_width=w,
        conv_config=conv_config,
        compute_config=compute_config,
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    return out, oh, ow


def tt_group_norm(x, num_groups, device, weight=None, bias=None):
    """Group norm via TT-Lang kernel. x is TILE_LAYOUT [N, 1, H*W, C]."""
    TILE = 32
    shape = x.shape
    # Reshape from [N, 1, H*W, C] to [H*W, C] for TT-Lang kernel
    hw = shape[2] if len(shape) == 4 else shape[0]
    channels = shape[3] if len(shape) == 4 else shape[1]
    x_2d = ttnn.reshape(x, [hw, channels])

    seq_tiles = hw // TILE
    N = seq_tiles * TILE * TILE

    scaler = ttnn.from_torch(
        torch.ones(TILE, TILE, dtype=torch.bfloat16),
        ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=DRAM)
    mean_scale = ttnn.from_torch(
        torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16),
        ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=DRAM)
    out_2d = ttnn.from_torch(
        torch.zeros(hw, channels, dtype=torch.bfloat16),
        ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=DRAM)

    groupnorm_2g(x_2d, scaler, mean_scale, out_2d)

    ttnn.deallocate(scaler)
    ttnn.deallocate(mean_scale)

    # Apply optional weight and bias
    if weight is not None or bias is not None:
        out_4d = ttnn.reshape(out_2d, shape)
        if weight is not None:
            out_4d = ttnn.multiply(out_4d, weight)
        if bias is not None:
            out_4d = ttnn.add(out_4d, bias)
        return out_4d

    return ttnn.reshape(out_2d, shape)


# ---------------------------------------------------------------------------
# AdaGroupNorm: group_norm(x) * (1 + scale) + shift
# The linear projection is done on host, modulation on device.
# ---------------------------------------------------------------------------

def ada_group_norm(x, cond_host, linear_w, linear_b, in_ch, num_groups,
                   device, batch):
    """AdaGroupNorm forward.

    x: on device, TILE_LAYOUT [N, 1, H*W, C]
    cond_host: torch tensor (B, COND_CH)
    linear_w, linear_b: torch tensors for the conditioning linear layer
    Returns: modulated tensor on device, same shape as x
    """
    normed = tt_group_norm(x, num_groups, device)

    # Host-side: conditioning -> scale, shift
    scale_shift = F.linear(cond_host, linear_w, linear_b)
    scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, in_ch)

    # Expand to broadcast shape [B, 1, 1, C] for TTNN broadcasting
    scale_4d = scale.unsqueeze(1).unsqueeze(1).to(torch.bfloat16)
    shift_4d = shift.unsqueeze(1).unsqueeze(1).to(torch.bfloat16)

    scale_tt = tt_dev(scale_4d, device)
    shift_tt = tt_dev(shift_4d, device)

    # out = normed * (1 + scale) + shift
    one_plus_scale = ttnn.add(scale_tt, 1.0)
    modulated = ttnn.multiply(normed, one_plus_scale)
    out = ttnn.add(modulated, shift_tt)

    ttnn.deallocate(normed)
    ttnn.deallocate(scale_tt)
    ttnn.deallocate(shift_tt)
    ttnn.deallocate(one_plus_scale)
    ttnn.deallocate(modulated)
    return out


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

def resblock(x, cond_host, sd, prefix, device, in_ch, out_ch, batch, h, w,
             conv_config, compute_config):
    """Single ResBlock forward.

    x: on device [N, 1, H*W, C]
    Returns: output on device [N, 1, H*W, out_ch]
    """
    ng_in = max(1, in_ch // GN_GROUP_SIZE)
    ng_out = max(1, out_ch // GN_GROUP_SIZE)
    should_proj = (in_ch != out_ch)

    # Skip projection
    if should_proj:
        proj_w = tt_host(sd[f"{prefix}.proj.weight"])
        proj_b = tt_host(sd[f"{prefix}.proj.bias"].reshape(1, 1, 1, -1))
        r, _, _ = tt_conv2d(x, proj_w, proj_b, device, in_ch, out_ch,
                             batch, h, w, kernel_size=(1, 1), padding=(0, 0),
                             conv_config=conv_config, compute_config=compute_config)
    else:
        r = x

    # norm1 -> silu -> conv1
    h1 = ada_group_norm(x, cond_host,
                         sd[f"{prefix}.norm1.linear.weight"],
                         sd[f"{prefix}.norm1.linear.bias"],
                         in_ch, ng_in, device, batch)
    h1 = ttnn.silu(h1)

    conv1_w = tt_host(sd[f"{prefix}.conv1.weight"])
    conv1_b = tt_host(sd[f"{prefix}.conv1.bias"].reshape(1, 1, 1, -1))
    h1, _, _ = tt_conv2d(h1, conv1_w, conv1_b, device, in_ch, out_ch,
                          batch, h, w, conv_config=conv_config,
                          compute_config=compute_config)

    # norm2 -> silu -> conv2
    h2 = ada_group_norm(h1, cond_host,
                         sd[f"{prefix}.norm2.linear.weight"],
                         sd[f"{prefix}.norm2.linear.bias"],
                         out_ch, ng_out, device, batch)
    ttnn.deallocate(h1)
    h2 = ttnn.silu(h2)

    conv2_w = tt_host(sd[f"{prefix}.conv2.weight"])
    conv2_b = tt_host(sd[f"{prefix}.conv2.bias"].reshape(1, 1, 1, -1))
    h2, _, _ = tt_conv2d(h2, conv2_w, conv2_b, device, out_ch, out_ch,
                          batch, h, w, conv_config=conv_config,
                          compute_config=compute_config)

    # residual add
    out = ttnn.add(h2, r)
    ttnn.deallocate(h2)
    if should_proj:
        ttnn.deallocate(r)

    return out


# ---------------------------------------------------------------------------
# UNet forward
# ---------------------------------------------------------------------------

def unet_forward(x, cond_host, sd, device, batch, h, w, conv_config, compute_config):
    """Full UNet. x: on device [N, 1, H*W, C_in]. Returns same format."""
    num_levels = len(CHANNELS)

    # ---- Encoder ----
    d_outputs = []  # list of lists of skip tensors per level
    cur_h, cur_w = h, w
    print("  UNet encoder...", flush=True)

    for level in range(num_levels):
        c1 = CHANNELS[max(0, level - 1)]
        c2 = CHANNELS[level]

        # Downsample
        if level == 0:
            x_down = x
        else:
            ds_w = tt_host(sd[f"inner_model.unet.downsamples.{level}.conv.weight"])
            ds_b = tt_host(sd[f"inner_model.unet.downsamples.{level}.conv.bias"].reshape(1, 1, 1, -1))
            x_down, cur_h, cur_w = tt_conv2d(
                x, ds_w, ds_b, device, c1, c1, batch, cur_h, cur_w,
                stride=(2, 2), conv_config=conv_config, compute_config=compute_config)

        block_outputs = [x_down]
        x_cur = x_down
        for bi in range(DEPTHS[level]):
            in_ch = c1 if bi == 0 else c2
            prefix = f"inner_model.unet.d_blocks.{level}.resblocks.{bi}"
            print(f"    enc L{level} B{bi} ({in_ch}->{c2}, {cur_h}x{cur_w}) x={x_cur.shape}", flush=True)
            x_cur = resblock(x_cur, cond_host, sd, prefix, device,
                              in_ch, c2, batch, cur_h, cur_w,
                              conv_config, compute_config)
            block_outputs.append(x_cur)

        d_outputs.append(block_outputs)
        x = x_cur

    # ---- Mid blocks (skip attention for now) ----
    print("  UNet mid...", flush=True)
    mid_h, mid_w = cur_h, cur_w
    c_mid = CHANNELS[-1]
    for bi in range(2):
        prefix = f"inner_model.unet.mid_blocks.resblocks.{bi}"
        x = resblock(x, cond_host, sd, prefix, device,
                      c_mid, c_mid, batch, mid_h, mid_w,
                      conv_config, compute_config)
        # TODO: self-attention at 8x8 (mid_blocks have attn=True)

    # ---- Decoder ----
    print("  UNet decoder...", flush=True)
    for dec_idx in range(num_levels):
        # Decoder processes levels in reverse: 3, 2, 1, 0
        level = num_levels - 1 - dec_idx
        c1 = CHANNELS[max(0, level - 1)]
        c2 = CHANNELS[level]

        # Upsample
        if dec_idx == 0:
            x_up = x
        else:
            # upsample needs [N, H, W, C] ROW_MAJOR
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x_4d = ttnn.reshape(x_rm, [batch, cur_h, cur_w, c2])
            ttnn.deallocate(x_rm)
            x_up_raw = ttnn.upsample(x_4d, scale_factor=2, mode="nearest")
            ttnn.deallocate(x_4d)
            new_h = cur_h * 2
            new_w = cur_w * 2

            # Upsample conv
            up_idx = dec_idx  # upsamples[0]=Identity, upsamples[1..3]=Upsample
            up_ch = CHANNELS[level]  # channels in reverse order
            up_w = tt_host(sd[f"inner_model.unet.upsamples.{up_idx}.conv.weight"])
            up_b = tt_host(sd[f"inner_model.unet.upsamples.{up_idx}.conv.bias"].reshape(1, 1, 1, -1))
            x_up, cur_h, cur_w = tt_conv2d(
                x_up_raw, up_w, up_b, device, up_ch, up_ch, batch, new_h, new_w,
                conv_config=conv_config, compute_config=compute_config)
            ttnn.deallocate(x_up_raw)

        # Skip connections (reversed)
        skip = d_outputs[level][::-1]  # [block_out_last, ..., block_out_0, x_down]

        # u_blocks index: reversed in PyTorch init
        u_idx = num_levels - 1 - level

        # Decoder has depth+1 ResBlocks
        n_dec_blocks = DEPTHS[level] + 1
        x_cur = x_up
        for bi in range(n_dec_blocks):
            # Concat with skip connection
            x_cur = ttnn.concat([x_cur, skip[bi]], dim=-1)
            cat_ch = c2 * 2 if bi < DEPTHS[level] else c1 + c2
            out_ch = c2 if bi < DEPTHS[level] else c1

            prefix = f"inner_model.unet.u_blocks.{u_idx}.resblocks.{bi}"
            x_cur = resblock(x_cur, cond_host, sd, prefix, device,
                              cat_ch, out_ch, batch, cur_h, cur_w,
                              conv_config, compute_config)

        x = x_cur

    return x, cur_h, cur_w


# ---------------------------------------------------------------------------
# Inner model forward
# ---------------------------------------------------------------------------

def inner_model_forward(noisy_next_obs, c_noise, obs, act, sd, device,
                        batch, num_actions, conv_config, compute_config):
    """Full inner model.

    noisy_next_obs: [B, 3, 64, 64] torch float
    obs: [B, 12, 64, 64] torch float (4 frames * 3ch)
    act: [B, 4] torch int
    Returns: model output on device
    """
    print("  Computing conditioning...", flush=True)
    cond_host = compute_conditioning(c_noise, act, sd, num_actions)

    # Prepare input: cat obs + noisy, NCHW -> NHWC -> [N, 1, H*W, C]
    cat_input = torch.cat((obs, noisy_next_obs), dim=1)  # [B, 15, 64, 64]
    cat_nhwc = cat_input.permute(0, 2, 3, 1).contiguous()  # [B, 64, 64, 15]

    # Send to device as host tensor for conv_in
    x_host = tt_host(cat_nhwc.to(torch.bfloat16))

    conv_in_w = tt_host(sd["inner_model.conv_in.weight"])
    conv_in_b = tt_host(sd["inner_model.conv_in.bias"].reshape(1, 1, 1, -1))
    x, h, w = tt_conv2d(x_host, conv_in_w, conv_in_b, device,
                          in_ch=15, out_ch=CHANNELS[0], batch=batch,
                          h=IMG_SIZE, w=IMG_SIZE,
                          conv_config=conv_config, compute_config=compute_config)

    # UNet
    x, h, w = unet_forward(x, cond_host, sd, device, batch, h, w,
                             conv_config, compute_config)

    # Final norm + silu + conv_out
    # group_norm weight/bias must be ROW_MAJOR on device
    gn_w_dev = ttnn.from_torch(
        sd["inner_model.norm_out.norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, -1),
        ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
    gn_b_dev = ttnn.from_torch(
        sd["inner_model.norm_out.norm.bias"].to(torch.bfloat16).reshape(1, 1, 1, -1),
        ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
    ng = max(1, CHANNELS[0] // GN_GROUP_SIZE)
    x = tt_group_norm(x, ng, device, weight=gn_w_dev, bias=gn_b_dev)
    x = ttnn.silu(x)

    conv_out_w = tt_host(sd["inner_model.conv_out.weight"])
    conv_out_b = tt_host(sd["inner_model.conv_out.bias"].reshape(1, 1, 1, -1))
    x, oh, ow = tt_conv2d(x, conv_out_w, conv_out_b, device,
                            in_ch=CHANNELS[0], out_ch=IMG_CH, batch=batch,
                            h=h, w=w, conv_config=conv_config,
                            compute_config=compute_config)

    return x, oh, ow


# ---------------------------------------------------------------------------
# Denoiser + sampling
# ---------------------------------------------------------------------------

def denoise_step(noisy_next_obs, sigma, obs, act, sd, device, batch,
                 num_actions, conv_config, compute_config):
    """Single denoise step. All inputs torch tensors. Returns denoised torch bf16."""
    c_in, c_out, c_skip, c_noise = compute_conditioners(sigma)

    rescaled_obs = obs / SIGMA_DATA
    rescaled_noise = noisy_next_obs * c_in[:, None, None, None]

    model_out_tt, oh, ow = inner_model_forward(
        rescaled_noise.to(torch.bfloat16), c_noise,
        rescaled_obs.to(torch.bfloat16), act,
        sd, device, batch, num_actions, conv_config, compute_config)

    model_out = ttnn.to_torch(model_out_tt)
    ttnn.deallocate(model_out_tt)

    # Reshape from [N, 1, H*W, C] -> [N, C, H, W]
    model_out = model_out.reshape(batch, oh, ow, -1)[:, :, :, :IMG_CH]
    model_out = model_out.permute(0, 3, 1, 2).float()

    # Precondition: d = c_skip * noisy + c_out * model_out
    denoised = c_skip[:, None, None, None] * noisy_next_obs + c_out[:, None, None, None] * model_out
    denoised = denoised.clamp(-1, 1).add(1).div(2).mul(255).byte().float().div(255).mul(2).sub(1)
    return denoised.to(torch.bfloat16)


def sample_next_frame(prev_obs, prev_act, sd, device, num_actions,
                      conv_config, compute_config):
    """Diffusion sampling. Returns [B, 3, 64, 64] torch bf16."""
    b = prev_obs.shape[0]
    sigmas = build_sigmas()
    s_in = torch.ones(b)

    x = torch.randn(b, IMG_CH, IMG_SIZE, IMG_SIZE).to(torch.bfloat16)
    obs_flat = prev_obs.reshape(b, -1, IMG_SIZE, IMG_SIZE).float()

    for i in range(len(sigmas) - 1):
        t_step = time.time()
        sigma = sigmas[i] * s_in
        denoised = denoise_step(x.float(), sigma, obs_flat, prev_act,
                                 sd, device, b, num_actions,
                                 conv_config, compute_config)
        d = (x.float() - denoised.float()) / sigma[:, None, None, None]
        dt = sigmas[i + 1] - sigmas[i]
        x = (x.float() + d * dt).to(torch.bfloat16)
        print(f"    Denoise step {i+1}/{NUM_DENOISE_STEPS} done ({(time.time()-t_step)*1000:.0f}ms)")

    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Diamond World Model on Tenstorrent Hardware")
    print("=" * 60)

    # --- Download weights ---
    print("\n[1/4] Loading weights...")
    weight_path = download_weights("Breakout")
    sd = load_denoiser_sd(weight_path)
    num_actions = sd["inner_model.act_emb.0.weight"].shape[0]
    print(f"  {len(sd)} params, {num_actions} actions (Breakout)")

    # --- Setup device ---
    print("\n[2/4] Opening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    print(f"  Compute grid: {device.compute_with_storage_grid_size()}")

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )

    # --- Test input ---
    print("\n[3/4] Creating test input...")
    torch.manual_seed(42)
    B = 1
    prev_obs = torch.randn(B, NUM_COND, IMG_CH, IMG_SIZE, IMG_SIZE, dtype=torch.bfloat16)
    prev_act = torch.randint(0, num_actions, (B, NUM_COND))
    print(f"  Batch={B}, {NUM_COND} frames, {IMG_SIZE}x{IMG_SIZE} RGB")

    # --- Run inference ---
    print(f"\n[4/4] Diffusion sampling ({NUM_DENOISE_STEPS} steps)...")
    t0 = time.time()
    next_frame = sample_next_frame(prev_obs, prev_act, sd, device, num_actions,
                                    conv_config, compute_config)
    t1 = time.time()

    print(f"\n  Output: {next_frame.shape}")
    print(f"  Range: [{next_frame.min():.3f}, {next_frame.max():.3f}]")
    print(f"  Time: {(t1-t0)*1000:.0f}ms")

    # Save frame as PNG
    frame = next_frame[0].float().add(1).div(2).clamp(0, 1).mul(255).byte()
    frame = frame.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    from PIL import Image
    img = Image.fromarray(frame, "RGB")
    img.save("/tmp/diamond_frame.png")
    print(f"  Saved to /tmp/diamond_frame.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
