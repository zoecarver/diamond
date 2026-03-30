"""Minimal conv2d test following tt-metal's sweep test pattern."""

import torch
import ttnn

# Key: l1_small_size must be set for conv2d to work
device = ttnn.open_device(device_id=0, l1_small_size=32768)

print("Device compute grid:", device.compute_with_storage_grid_size())

# Following the pattern from tt-metal/tests/sweep_framework/sweep_utils/conv2d_common.py:
# - Input: NHWC torch tensor -> ttnn.from_torch WITHOUT device
# - Weight: OIHW torch tensor -> ttnn.from_torch WITHOUT device
# - Bias: [1,1,1,Cout] torch tensor -> ttnn.from_torch WITHOUT device
# - conv2d handles device placement internally

def test_conv(name, B, H, W, Cin, Cout, stride=1, pad=1):
    print(f"\n{name}: [{B},{H},{W},{Cin}] -> stride={stride}")

    x_nchw = torch.randn(B, Cin, H, W, dtype=torch.bfloat16).float()
    x_nhwc = x_nchw.permute(0, 2, 3, 1)
    tt_input = ttnn.from_torch(x_nhwc, ttnn.bfloat16)

    w = torch.randn(Cout, Cin, 3, 3, dtype=torch.bfloat16).float()
    tt_weight = ttnn.from_torch(w, ttnn.bfloat16)

    b = torch.randn(1, 1, 1, Cout, dtype=torch.bfloat16).float()
    tt_bias = ttnn.from_torch(b, ttnn.bfloat16)

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

    try:
        [out, [out_h, out_w], [w_dev, b_dev]] = ttnn.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            in_channels=Cin,
            out_channels=Cout,
            device=device,
            bias_tensor=tt_bias,
            kernel_size=(3, 3),
            stride=(stride, stride),
            padding=(pad, pad),
            dilation=(1, 1),
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=conv_config,
            compute_config=compute_config,
            groups=1,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        print(f"  Output: {out_h}x{out_w}, shape={out.shape}")

        # Verify vs PyTorch
        ref = torch.nn.functional.conv2d(x_nchw, w, b.reshape(-1), stride=stride, padding=pad)
        result = ttnn.to_torch(out)
        result = result.reshape(B, out_h, out_w, -1)[:, :, :, :Cout]
        result_nchw = result.permute(0, 3, 1, 2)
        diff = (result_nchw.float() - ref.float()).abs().max().item()
        print(f"  Max diff vs PyTorch: {diff:.4f}  PASS={diff < 2.0}")
    except Exception as e:
        err = str(e).split('\n')[0][:200]
        print(f"  FAILED: {err}")


# Test cases matching Diamond UNet shapes
test_conv("conv_in (15->64, 64x64)", B=1, H=64, W=64, Cin=15, Cout=64)
test_conv("resblock (64->64, 64x64)", B=1, H=64, W=64, Cin=64, Cout=64)
test_conv("downsample (64->64, 64x64, s2)", B=1, H=64, W=64, Cin=64, Cout=64, stride=2)
test_conv("resblock (64->64, 32x32)", B=1, H=32, W=32, Cin=64, Cout=64)
test_conv("resblock (64->64, 16x16)", B=1, H=16, W=16, Cin=64, Cout=64)
test_conv("resblock (64->64, 8x8)", B=1, H=8, W=8, Cin=64, Cout=64)
test_conv("conv_out (64->3, 64x64)", B=1, H=64, W=64, Cin=64, Cout=3)

ttnn.close_device(device)
print("\nDone.")
