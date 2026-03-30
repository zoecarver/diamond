# Diamond on Tenstorrent

Diffusion world model ([DIAMOND](https://diamond-wm.github.io), NeurIPS 2024) running on Tenstorrent Blackhole hardware. Generates Atari Breakout frames autoregressively using pretrained weights from HuggingFace.

## Architecture

UNet-based diffusion denoiser. 4-level encoder/decoder, channels [64,64,64,64], depth 2 per level, no attention except mid blocks. 3 denoising steps per frame using Euler sampling. Conditioning: 4 prior RGB frames (64x64) + 4 actions, projected via Fourier noise embedding + action embedding + MLP.

## Performance

~120ms per frame (3x 40ms denoise steps) on a single Blackhole chip after warmup, without tracing.

## Kernels

| Operation | Backend | Notes |
|-----------|---------|-------|
| GroupNorm | TT-Lang | 3-pass kernel (mean, variance, normalize). Replaces ttnn.group_norm which crashes at small spatial sizes. |
| SiLU, Add, Mul, AdaLN modulate, Precondition, Euler step | TT-Lang | Fused elementwise kernels with grid="auto" streaming. |
| Conv2d, Upsample, Concat | TTNN | Standard ops. |
| Conditioning (Fourier features, embeddings, MLP) | PyTorch CPU | |

## How to run

Requires a Tenstorrent device accessible via the [tt-connect-remote-device](https://docs.tenstorrent.com) scripts.

```bash
# Copy TT-Lang kernels to remote (only needed once or after edits)
scripts/copy-file.sh tt/kernels.py

# Generate 8 autoregressive Breakout frames
scripts/run-test.sh --hw tt/diamond_play.py
```

Weights are downloaded automatically from `eloialonso/diamond` on HuggingFace.

## Files

| File | Description |
|------|-------------|
| `tt/diamond_play.py` | End-to-end inference: creates Breakout frames, runs diffusion loop, saves PNGs |
| `tt/diamond_tt.py` | Core model: UNet forward pass, denoiser, sampling |
| `tt/kernels.py` | TT-Lang fused kernels (GroupNorm, SiLU, AdaLN, etc.) |
| `tt/groupnorm_kernel.py` | Standalone GroupNorm kernel + test |
| `tt/test_pcc_triage.py` | Layer-by-layer PCC comparison vs PyTorch reference |

## Output

4 synthetic conditioning frames followed by 8 model-generated frames:

| Input frames (conditioning) | | | |
|---|---|---|---|
| ![](tt/doc/frame_00.png) | ![](tt/doc/frame_01.png) | ![](tt/doc/frame_02.png) | ![](tt/doc/frame_03.png) |

| Generated frames | | | |
|---|---|---|---|
| ![](tt/doc/frame_04.png) | ![](tt/doc/frame_05.png) | ![](tt/doc/frame_06.png) | ![](tt/doc/frame_07.png) |
| ![](tt/doc/frame_08.png) | ![](tt/doc/frame_09.png) | ![](tt/doc/frame_10.png) | ![](tt/doc/frame_11.png) |
