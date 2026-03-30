"""
Diamond world model inference with synthetic Breakout-like initial frames.

Generates 4 initial conditioning frames resembling Breakout (black bg, colored
bricks, paddle, ball), then runs the diffusion model autoregressively to
predict future frames.
"""
import math
import time
import sys
import torch
import torch.nn.functional as F
import ttnn
import numpy as np

sys.path.insert(0, "/tmp")
from kernels import groupnorm_2g

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGMA_DATA = 0.5
SIGMA_OFFSET_NOISE = 0.3
NUM_COND = 4
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
L1 = ttnn.L1_MEMORY_CONFIG
TILE = 32

# Weight and tensor caches -- populated on first forward pass, reused thereafter
_conv_weight_cache = {}   # cache_key -> (device_w, device_b)
_gn_cache = {}            # (hw, channels) -> (scaler, mean_scale)


# ---------------------------------------------------------------------------
# Create synthetic Breakout frames
# ---------------------------------------------------------------------------

def make_breakout_frame(ball_y=50, ball_x=32, paddle_x=28):
    """Create a simple Breakout-like 64x64 RGB frame in [-1, 1]."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)

    # Bricks (rows 8-20, colored)
    colors = [
        (200, 50, 50),   # red
        (200, 150, 50),  # orange
        (50, 200, 50),   # green
        (50, 50, 200),   # blue
    ]
    for i, color in enumerate(colors):
        y = 8 + i * 3
        for bx in range(0, 64, 10):
            img[y:y+2, bx:bx+8] = color

    # Ball (small white square)
    by, bx = int(ball_y), int(ball_x)
    img[max(0, by-1):by+2, max(0, bx-1):bx+2] = (255, 255, 255)

    # Paddle (white bar at bottom)
    px = int(paddle_x)
    img[60:62, max(0, px):min(64, px+8)] = (200, 200, 200)

    # Score area (dark gray top)
    img[0:6, :] = (40, 40, 40)

    # Convert to [-1, 1] float tensor [C, H, W]
    t = torch.from_numpy(img).float().div(255).mul(2).sub(1)
    return t.permute(2, 0, 1)  # [3, 64, 64]


def make_initial_frames():
    """Create 4 conditioning frames with slight ball movement."""
    frames = []
    for i in range(NUM_COND):
        ball_y = 45 - i * 3
        ball_x = 30 + i * 2
        frames.append(make_breakout_frame(ball_y, ball_x, paddle_x=28))
    return torch.stack(frames).unsqueeze(0)  # [1, 4, 3, 64, 64]


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
# Host-side helpers
# ---------------------------------------------------------------------------

def fourier_features(sigma, weight):
    f = 2 * math.pi * sigma.unsqueeze(1) @ weight
    return torch.cat([f.cos(), f.sin()], dim=-1)


def compute_conditioning(sigma, prev_act, sd, num_actions):
    b = sigma.shape[0]
    noise_emb = fourier_features(sigma, sd["inner_model.noise_emb.weight"])
    act_emb = sd["inner_model.act_emb.0.weight"][prev_act.long()].reshape(b, -1)
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
# TTNN helpers
# ---------------------------------------------------------------------------

def tt_host(t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype)


def tt_dev(t, device, layout=ttnn.TILE_LAYOUT, mem=DRAM):
    return ttnn.from_torch(t.to(torch.bfloat16), ttnn.bfloat16, layout=layout,
                           device=device, memory_config=mem)


def tt_conv2d(x, device, in_ch, out_ch, batch, h, w,
              kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
              conv_config=None, compute_config=None,
              cache_key=None, sd=None):
    """Conv2d with L1 weight caching. First call caches device weights."""
    if cache_key and cache_key in _conv_weight_cache:
        w_tt, b_tt = _conv_weight_cache[cache_key]
    else:
        w_tt = tt_host(sd[f"{cache_key}.weight"])
        b_tt = tt_host(sd[f"{cache_key}.bias"].reshape(1, 1, 1, -1))

    [out, [oh, ow], [wd, bd]] = ttnn.conv2d(
        input_tensor=x, weight_tensor=w_tt, in_channels=in_ch,
        out_channels=out_ch, device=device, bias_tensor=b_tt,
        kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=(1, 1), batch_size=batch, input_height=h, input_width=w,
        conv_config=conv_config, compute_config=compute_config,
        groups=1, return_output_dim=True, return_weights_and_bias=True)

    if cache_key and cache_key not in _conv_weight_cache:
        _conv_weight_cache[cache_key] = (wd, bd)

    return out, oh, ow


def tt_group_norm(x, num_groups, device):
    shape = x.shape
    hw = shape[2] if len(shape) == 4 else shape[0]
    channels = shape[3] if len(shape) == 4 else shape[1]
    x_2d = ttnn.reshape(x, [hw, channels])
    seq_tiles = hw // TILE
    N = seq_tiles * TILE * TILE

    cache_key = (hw, channels)
    if cache_key not in _gn_cache:
        scaler = ttnn.from_torch(
            torch.ones(TILE, TILE, dtype=torch.bfloat16),
            ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        mean_scale = ttnn.from_torch(
            torch.full((TILE, TILE), 1.0 / N, dtype=torch.bfloat16),
            ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        out_2d = ttnn.from_torch(
            torch.zeros(hw, channels, dtype=torch.bfloat16),
            ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        _gn_cache[cache_key] = (scaler, mean_scale, out_2d)

    scaler, mean_scale, out_2d = _gn_cache[cache_key]
    groupnorm_2g(x_2d, scaler, mean_scale, out_2d)
    return ttnn.reshape(out_2d, shape)


# ---------------------------------------------------------------------------
# Batch precompute AdaGroupNorm scale/shift for all resblocks in one step.
# Eliminates per-resblock F.linear + host->device transfers during forward.
# ---------------------------------------------------------------------------

_adaln_params = {}  # prefix -> (scale_buf, shift_buf), allocated once on device

def precompute_adaln_params(cond_host, sd, device):
    """Compute all AdaGroupNorm scale/shift on host, write to fixed device buffers."""
    for key in sd:
        if not key.endswith(".norm1.linear.weight") and not key.endswith(".norm2.linear.weight"):
            continue
        prefix = key.rsplit(".weight", 1)[0]
        scale_shift = F.linear(cond_host, sd[f"{prefix}.weight"], sd[f"{prefix}.bias"])
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale_host = ttnn.from_torch(
            scale.unsqueeze(1).unsqueeze(1).to(torch.bfloat16), ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT)
        shift_host = ttnn.from_torch(
            shift.unsqueeze(1).unsqueeze(1).to(torch.bfloat16), ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT)

        if prefix not in _adaln_params:
            # First call: allocate device buffers
            scale_buf = tt_dev(scale.unsqueeze(1).unsqueeze(1).to(torch.bfloat16), device)
            shift_buf = tt_dev(shift.unsqueeze(1).unsqueeze(1).to(torch.bfloat16), device)
            _adaln_params[prefix] = (scale_buf, shift_buf)
        else:
            # Subsequent calls: update in place (no allocation)
            scale_buf, shift_buf = _adaln_params[prefix]
            ttnn.copy_host_to_device_tensor(scale_host, scale_buf)
            ttnn.copy_host_to_device_tensor(shift_host, shift_buf)


def ada_group_norm(x, adaln_prefix, num_groups, device):
    """AdaGroupNorm using pre-computed scale/shift from _adaln_params."""
    normed = tt_group_norm(x, num_groups, device)
    scale_tt, shift_tt = _adaln_params[adaln_prefix]
    one_plus_scale = ttnn.add(scale_tt, 1.0)
    modulated = ttnn.multiply(normed, one_plus_scale)
    out = ttnn.add(modulated, shift_tt)
    ttnn.deallocate(normed)
    ttnn.deallocate(one_plus_scale)
    ttnn.deallocate(modulated)
    return out


def resblock(x, sd, prefix, device, in_ch, out_ch, batch, h, w,
             conv_config, compute_config):
    ng_in = max(1, in_ch // GN_GROUP_SIZE)
    ng_out = max(1, out_ch // GN_GROUP_SIZE)
    should_proj = (in_ch != out_ch)

    if should_proj:
        r, _, _ = tt_conv2d(x, device, in_ch, out_ch,
                             batch, h, w, kernel_size=(1, 1), padding=(0, 0),
                             conv_config=conv_config, compute_config=compute_config,
                             cache_key=f"{prefix}.proj", sd=sd)
    else:
        r = x

    h1 = ada_group_norm(x, f"{prefix}.norm1.linear", ng_in, device)
    h1 = ttnn.silu(h1)
    h1, _, _ = tt_conv2d(h1, device, in_ch, out_ch,
                          batch, h, w, conv_config=conv_config,
                          compute_config=compute_config,
                          cache_key=f"{prefix}.conv1", sd=sd)

    h2 = ada_group_norm(h1, f"{prefix}.norm2.linear", ng_out, device)
    ttnn.deallocate(h1)
    h2 = ttnn.silu(h2)
    h2, _, _ = tt_conv2d(h2, device, out_ch, out_ch,
                          batch, h, w, conv_config=conv_config,
                          compute_config=compute_config,
                          cache_key=f"{prefix}.conv2", sd=sd)

    out = ttnn.add(h2, r)
    ttnn.deallocate(h2)
    if should_proj:
        ttnn.deallocate(r)
    return out


def unet_forward(x, sd, device, batch, h, w, conv_config, compute_config):
    num_levels = len(CHANNELS)
    d_outputs = []
    cur_h, cur_w = h, w

    for level in range(num_levels):
        c1 = CHANNELS[max(0, level - 1)]
        c2 = CHANNELS[level]
        if level == 0:
            x_down = x
        else:
            x_down, cur_h, cur_w = tt_conv2d(
                x, device, c1, c1, batch, cur_h, cur_w,
                stride=(2, 2), conv_config=conv_config, compute_config=compute_config,
                cache_key=f"inner_model.unet.downsamples.{level}.conv", sd=sd)

        block_outputs = [x_down]
        x_cur = x_down
        for bi in range(DEPTHS[level]):
            in_ch = c1 if bi == 0 else c2
            prefix = f"inner_model.unet.d_blocks.{level}.resblocks.{bi}"
            x_cur = resblock(x_cur, sd, prefix, device,
                              in_ch, c2, batch, cur_h, cur_w, conv_config, compute_config)
            block_outputs.append(x_cur)
        d_outputs.append(block_outputs)
        x = x_cur

    mid_h, mid_w = cur_h, cur_w
    c_mid = CHANNELS[-1]
    for bi in range(2):
        prefix = f"inner_model.unet.mid_blocks.resblocks.{bi}"
        x = resblock(x, sd, prefix, device,
                      c_mid, c_mid, batch, mid_h, mid_w, conv_config, compute_config)

    for dec_idx in range(num_levels):
        level = num_levels - 1 - dec_idx
        c1 = CHANNELS[max(0, level - 1)]
        c2 = CHANNELS[level]
        if dec_idx == 0:
            x_up = x
        else:
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x_4d = ttnn.reshape(x_rm, [batch, cur_h, cur_w, c2])
            ttnn.deallocate(x_rm)
            x_up_raw = ttnn.upsample(x_4d, scale_factor=2, mode="nearest")
            ttnn.deallocate(x_4d)
            new_h = cur_h * 2
            new_w = cur_w * 2
            up_idx = dec_idx
            up_ch = CHANNELS[level]
            x_up, cur_h, cur_w = tt_conv2d(
                x_up_raw, device, up_ch, up_ch, batch, new_h, new_w,
                conv_config=conv_config, compute_config=compute_config,
                cache_key=f"inner_model.unet.upsamples.{up_idx}.conv", sd=sd)
            ttnn.deallocate(x_up_raw)

        skip = d_outputs[level][::-1]
        u_idx = num_levels - 1 - level
        n_dec_blocks = DEPTHS[level] + 1
        x_cur = x_up
        for bi in range(n_dec_blocks):
            x_cur = ttnn.concat([x_cur, skip[bi]], dim=-1)
            cat_ch = c2 * 2 if bi < DEPTHS[level] else c1 + c2
            out_ch = c2 if bi < DEPTHS[level] else c1
            prefix = f"inner_model.unet.u_blocks.{u_idx}.resblocks.{bi}"
            x_cur = resblock(x_cur, sd, prefix, device,
                              cat_ch, out_ch, batch, cur_h, cur_w, conv_config, compute_config)
        x = x_cur

    return x, cur_h, cur_w


_norm_out_cache = {}  # "w"/"b" -> device tensor


def model_device_forward(input_tt, sd, device, batch, conv_config, compute_config):
    """Pure device forward: conv_in -> UNet -> norm_out -> conv_out.
    All inputs must be on device. No host I/O. Traceable."""
    x, h, w = tt_conv2d(input_tt, device,
                          in_ch=15, out_ch=CHANNELS[0], batch=batch,
                          h=IMG_SIZE, w=IMG_SIZE,
                          conv_config=conv_config, compute_config=compute_config,
                          cache_key="inner_model.conv_in", sd=sd)

    x, h, w = unet_forward(x, sd, device, batch, h, w, conv_config, compute_config)

    ng = max(1, CHANNELS[0] // GN_GROUP_SIZE)
    x = tt_group_norm(x, ng, device)
    if "w" not in _norm_out_cache:
        gn_w = ttnn.from_torch(
            sd["inner_model.norm_out.norm.weight"].to(torch.bfloat16).reshape(1, 1, 1, -1),
            ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        gn_b = ttnn.from_torch(
            sd["inner_model.norm_out.norm.bias"].to(torch.bfloat16).reshape(1, 1, 1, -1),
            ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=L1)
        _norm_out_cache["w"] = gn_w
        _norm_out_cache["b"] = gn_b
    x = ttnn.multiply(x, _norm_out_cache["w"])
    x = ttnn.add(x, _norm_out_cache["b"])
    x = ttnn.silu(x)

    x, oh, ow = tt_conv2d(x, device,
                            in_ch=CHANNELS[0], out_ch=IMG_CH, batch=batch,
                            h=h, w=w, conv_config=conv_config,
                            compute_config=compute_config,
                            cache_key="inner_model.conv_out", sd=sd)
    return x, oh, ow


def prepare_input_and_run(noisy_next_obs, c_noise, obs, act, sd, device,
                          batch, num_actions, conv_config, compute_config,
                          input_buf=None, trace_id=None):
    """Host-side prep (conditioning, adaln, input copy) + device forward.
    If trace_id is set, replays trace instead of running ops."""
    cond_host = compute_conditioning(c_noise, act, sd, num_actions)
    precompute_adaln_params(cond_host, sd, device)

    cat_input = torch.cat((obs, noisy_next_obs), dim=1)
    cat_nhwc = cat_input.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)
    cat_host = ttnn.from_torch(cat_nhwc, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    if trace_id is not None:
        ttnn.copy_host_to_device_tensor(cat_host, input_buf)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    else:
        x_host = tt_host(cat_nhwc)
        return model_device_forward(x_host, sd, device, batch, conv_config, compute_config)


def sample_next_frame(prev_obs, prev_act, sd, device, num_actions,
                      conv_config, compute_config, trace_id=None,
                      input_buf=None, output_buf=None):
    """Diffusion sampling loop. If trace_id is provided, uses traced execution."""
    b = prev_obs.shape[0]
    sigmas = build_sigmas()
    obs_flat = prev_obs.reshape(b, -1, IMG_SIZE, IMG_SIZE).float()
    rescaled_obs = (obs_flat / SIGMA_DATA).to(torch.bfloat16)

    x_host = torch.randn(b, IMG_CH, IMG_SIZE, IMG_SIZE).to(torch.bfloat16)

    for i in range(len(sigmas) - 1):
        ttnn.synchronize_device(device)
        t0 = time.time()

        sigma = sigmas[i]
        c_in, c_out, c_skip, c_noise = compute_conditioners(sigma.unsqueeze(0))
        rescaled_noise = (x_host.float() * c_in[:, None, None, None]).to(torch.bfloat16)

        if trace_id is not None:
            prepare_input_and_run(
                rescaled_noise, c_noise, rescaled_obs, prev_act,
                sd, device, b, num_actions, conv_config, compute_config,
                input_buf=input_buf, trace_id=trace_id)
            model_out_tt = output_buf
            oh, ow = IMG_SIZE, IMG_SIZE
        else:
            model_out_tt, oh, ow = prepare_input_and_run(
                rescaled_noise, c_noise, rescaled_obs, prev_act,
                sd, device, b, num_actions, conv_config, compute_config)

        # Precondition on device
        x_nhwc = x_host.permute(0, 2, 3, 1).contiguous().reshape(1, 1, oh * ow, IMG_CH)
        x_tt = tt_dev(x_nhwc, device)

        denoised_tt = ttnn.add(
            ttnn.multiply(x_tt, c_skip.item()),
            ttnn.multiply(model_out_tt, c_out.item()))
        if trace_id is None:
            ttnn.deallocate(model_out_tt)

        denoised_tt = ttnn.clip(denoised_tt, -1.0, 1.0)

        # Euler step on device
        if i < len(sigmas) - 2:
            dt = sigmas[i + 1] - sigmas[i]
            dt_over_sigma = (dt / sigma).item()
            diff_tt = ttnn.subtract(x_tt, denoised_tt)
            x_new_tt = ttnn.add(x_tt, ttnn.multiply(diff_tt, dt_over_sigma))
            ttnn.deallocate(diff_tt)
            ttnn.deallocate(denoised_tt)
            ttnn.deallocate(x_tt)
            x_raw = ttnn.to_torch(x_new_tt)
            ttnn.deallocate(x_new_tt)
            x_host = x_raw.reshape(b, oh, ow, -1)[:, :, :, :IMG_CH].permute(0, 3, 1, 2).to(torch.bfloat16)
        else:
            ttnn.deallocate(x_tt)
            x_raw = ttnn.to_torch(denoised_tt)
            ttnn.deallocate(denoised_tt)
            x_host = x_raw.reshape(b, oh, ow, -1)[:, :, :, :IMG_CH].permute(0, 3, 1, 2).to(torch.bfloat16)

        ttnn.synchronize_device(device)
        print(f"    Step {i+1}/{NUM_DENOISE_STEPS} ({(time.time()-t0)*1000:.0f}ms)", flush=True)

    return x_host


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Diamond World Model - Breakout Inference")
    print("=" * 60)

    print("\n[1/5] Loading weights...")
    weight_path = download_weights("Breakout")
    sd = load_denoiser_sd(weight_path)
    num_actions = sd["inner_model.act_emb.0.weight"].shape[0]
    print(f"  {len(sd)} params, {num_actions} actions")

    print("\n[2/5] Opening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=32768,
                               trace_region_size=200_000_000)
    print(f"  Grid: {device.compute_with_storage_grid_size()}")

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16, shard_layout=None,
        deallocate_activation=False, output_layout=ttnn.TILE_LAYOUT)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4)

    print("\n[3/5] Creating Breakout frames...")
    prev_obs = make_initial_frames()  # [1, 4, 3, 64, 64]
    # Action 1 = "fire" in most Atari configs, 3 = "right"
    prev_act = torch.tensor([[1, 3, 3, 1]])  # fire, right, right, fire
    print(f"  {NUM_COND} frames, obs range: [{prev_obs.min():.2f}, {prev_obs.max():.2f}]")

    # Save initial frames
    from PIL import Image
    for i in range(NUM_COND):
        frame = prev_obs[0, i].add(1).div(2).clamp(0, 1).mul(255).byte()
        frame = frame.permute(1, 2, 0).numpy()
        Image.fromarray(frame, "RGB").save(f"/tmp/diamond_init_{i}.png")
    print(f"  Saved initial frames to /tmp/diamond_init_*.png")

    print(f"\n[4/7] Warmup pass (populates weight caches, compiles ops)...")
    B = 1
    t_warmup = time.time()
    _ = sample_next_frame(prev_obs, prev_act, sd, device,
                           num_actions, conv_config, compute_config)
    print(f"  Warmup: {(time.time()-t_warmup)*1000:.0f}ms")
    print(f"  Cached {len(_conv_weight_cache)} conv weights, {len(_gn_cache)} GN helpers, "
          f"{len(_norm_out_cache)} norm_out params")

    # --- Capture trace of model_device_forward ---
    print(f"\n[5/7] Capturing trace...")
    # Pre-allocate persistent input buffer on DRAM
    dummy_input = torch.randn(B, IMG_SIZE, IMG_SIZE, 15, dtype=torch.bfloat16)
    input_buf = ttnn.from_torch(dummy_input, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=device, memory_config=DRAM)
    # Pre-populate adaln params for trace capture
    sigmas = build_sigmas()
    c_in, c_out, c_skip, c_noise = compute_conditioners(sigmas[0].unsqueeze(0))
    cond_host = compute_conditioning(c_noise, prev_act, sd, num_actions)
    precompute_adaln_params(cond_host, sd, device)

    # Capture: pure device ops from conv_in through conv_out
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output_buf, _, _ = model_device_forward(
        input_buf, sd, device, B, conv_config, compute_config)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    print(f"  Trace captured (id={trace_id})")

    print(f"\n[6/7] Generating frames (traced, autoregressive, {NUM_DENOISE_STEPS} denoise steps each)...")
    NUM_GEN_FRAMES = 8
    all_frames = [prev_obs[0, i] for i in range(NUM_COND)]
    obs_buffer = prev_obs.clone()
    act_buffer = prev_act.clone()

    for frame_idx in range(NUM_GEN_FRAMES):
        t0 = time.time()
        print(f"\n  Frame {frame_idx + 1}/{NUM_GEN_FRAMES}:", flush=True)

        next_frame = sample_next_frame(obs_buffer, act_buffer, sd, device,
                                        num_actions, conv_config, compute_config,
                                        trace_id=trace_id, input_buf=input_buf,
                                        output_buf=output_buf)

        # Shift obs buffer: drop oldest, append new
        obs_buffer = torch.cat([obs_buffer[:, 1:], next_frame.float().unsqueeze(1)], dim=1)
        # Shift act buffer with a new action (alternate fire/right)
        new_act = torch.tensor([[3 if frame_idx % 2 == 0 else 1]])
        act_buffer = torch.cat([act_buffer[:, 1:], new_act], dim=1)

        all_frames.append(next_frame[0].float())
        elapsed = time.time() - t0
        print(f"    Total: {elapsed*1000:.0f}ms, range: [{next_frame.min():.2f}, {next_frame.max():.2f}]")

    print(f"\n[7/7] Saving frames...")
    for i, frame in enumerate(all_frames):
        img = frame.add(1).div(2).clamp(0, 1).mul(255).byte()
        img = img.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img, "RGB").save(f"/tmp/diamond_frame_{i:02d}.png")
    print(f"  Saved {len(all_frames)} frames to /tmp/diamond_frame_*.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
