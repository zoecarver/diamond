"""Test conv2d with on-device tensors (needed for tracing)."""

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG

device = ttnn.open_device(device_id=0, l1_small_size=32768)

B, H, W, Cin, Cout = 1, 64, 64, 64, 64

# On-device input (NHWC format, flattened to [N, 1, H*W, C])
x_nchw = torch.randn(B, Cin, H, W, dtype=torch.bfloat16).float()
x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(B, 1, H * W, Cin)
x_tt = ttnn.from_torch(x_nhwc, ttnn.bfloat16, device=device,
                         layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)

# Weight/bias on host (conv2d handles placement)
w = torch.randn(Cout, Cin, 3, 3, dtype=torch.bfloat16).float()
b = torch.randn(1, 1, 1, Cout, dtype=torch.bfloat16).float()
tt_w = ttnn.from_torch(w, ttnn.bfloat16)
tt_b = ttnn.from_torch(b, ttnn.bfloat16)

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

print("Test: on-device input conv2d [1,64,64,64] -> [1,64,64,64]")
[out, [out_h, out_w], [w_dev, b_dev]] = ttnn.conv2d(
    input_tensor=x_tt,
    weight_tensor=tt_w,
    in_channels=Cin,
    out_channels=Cout,
    device=device,
    bias_tensor=tt_b,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
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

# Second conv using on-device output from first (chained)
print("\nTest: chained conv2d (output of first -> input of second)")
[out2, [out_h2, out_w2], _] = ttnn.conv2d(
    input_tensor=out,
    weight_tensor=tt_w,
    in_channels=Cout,
    out_channels=Cout,
    device=device,
    bias_tensor=tt_b,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    batch_size=B,
    input_height=out_h,
    input_width=out_w,
    conv_config=conv_config,
    compute_config=compute_config,
    groups=1,
    return_output_dim=True,
    return_weights_and_bias=True,
)
print(f"  Output: {out_h2}x{out_w2}, shape={out2.shape}")

# Also test with return_weights_and_bias to reuse weights
print("\nTest: reuse device weights for third conv")
[out3, [out_h3, out_w3], _] = ttnn.conv2d(
    input_tensor=out2,
    weight_tensor=w_dev,
    in_channels=Cout,
    out_channels=Cout,
    device=device,
    bias_tensor=b_dev,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    batch_size=B,
    input_height=out_h2,
    input_width=out_w2,
    conv_config=conv_config,
    compute_config=compute_config,
    groups=1,
    return_output_dim=True,
    return_weights_and_bias=True,
)
print(f"  Output: {out_h3}x{out_w3}, shape={out3.shape}")

result = ttnn.to_torch(out3)
print(f"\n  Final result shape: {result.shape}")
print(f"  Value range: [{result.min():.3f}, {result.max():.3f}]")
print("\nALL PASS")

ttnn.close_device(device)
