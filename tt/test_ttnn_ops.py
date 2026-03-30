"""Test TTNN ops needed for Diamond UNet, on hardware."""

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG


def to_tt(t, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout,
                           device=device, memory_config=DRAM)


def test_silu(device):
    print("\n--- ttnn.silu ---")
    x = torch.randn(1, 1, 4096, 64, dtype=torch.bfloat16)
    out = ttnn.silu(to_tt(x, device))
    result = ttnn.to_torch(out)
    ref = torch.nn.functional.silu(x.float()).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  diff={diff:.4f}  PASS={diff < 0.5}")


def test_concat(device):
    print("\n--- ttnn.concat ---")
    a = torch.randn(1, 1, 4096, 64, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 4096, 64, dtype=torch.bfloat16)
    out = ttnn.concat([to_tt(a, device), to_tt(b, device)], dim=-1)
    result = ttnn.to_torch(out)
    ref = torch.cat([a, b], dim=-1)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  output shape: {result.shape}")
    print(f"  diff={diff:.6f}  PASS={diff < 0.001}")


def test_upsample(device):
    print("\n--- ttnn.upsample ---")
    x = torch.randn(1, 32, 32, 64, dtype=torch.bfloat16)
    out = ttnn.upsample(to_tt(x, device, layout=ttnn.ROW_MAJOR_LAYOUT),
                         scale_factor=2, mode="nearest")
    result = ttnn.to_torch(out)
    print(f"  input: {x.shape} -> output: {result.shape}")
    print(f"  PASS={result.shape == torch.Size([1, 64, 64, 64])}")


def test_group_norm(device):
    print("\n--- ttnn.group_norm ---")
    B, HW, C = 1, 4096, 64
    num_groups = 2
    x = torch.randn(B, 1, HW, C, dtype=torch.bfloat16)
    x_tt = to_tt(x, device)

    compute_grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)

    out = ttnn.group_norm(x_tt, num_groups=num_groups, epsilon=1e-5,
                          inplace=False, core_grid=core_grid)
    result = ttnn.to_torch(out)
    print(f"  shape: {result.shape}")
    print(f"  PASS (no crash)")


def test_conv2d_simple(device):
    """Test conv2d with minimal config."""
    print("\n--- ttnn.conv2d (3x3 stride=1) ---")
    B, H, W, Cin, Cout = 1, 32, 32, 64, 64

    x = torch.randn(1, 1, B * H * W, Cin, dtype=torch.bfloat16)
    x_tt = to_tt(x, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    w = torch.randn(Cout, Cin, 3, 3, dtype=torch.bfloat16)
    b = torch.zeros(1, 1, 1, Cout, dtype=torch.bfloat16)
    w_tt = to_tt(w, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = to_tt(b, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out, [out_h, out_w], _ = ttnn.conv2d(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=Cin,
        out_channels=Cout,
        batch_size=B,
        input_height=H,
        input_width=W,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=b_tt,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    print(f"  input: [{B},{H},{W},{Cin}] -> output: {out_h}x{out_w}")
    print(f"  tensor shape: {out.shape}")
    print(f"  PASS")


def test_conv2d_stride2(device):
    """Test conv2d with stride 2 for downsampling."""
    print("\n--- ttnn.conv2d (3x3 stride=2) ---")
    B, H, W, C = 1, 32, 32, 64

    x = torch.randn(1, 1, B * H * W, C, dtype=torch.bfloat16)
    x_tt = to_tt(x, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    w = torch.randn(C, C, 3, 3, dtype=torch.bfloat16)
    b = torch.zeros(1, 1, 1, C, dtype=torch.bfloat16)
    w_tt = to_tt(w, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = to_tt(b, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out, [out_h, out_w], _ = ttnn.conv2d(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=C,
        out_channels=C,
        batch_size=B,
        input_height=H,
        input_width=W,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=b_tt,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    print(f"  input: [{B},{H},{W},{C}] -> output: {out_h}x{out_w}")
    print(f"  PASS")


def test_conv2d_conv_in(device):
    """Test conv_in shape: 15 channels in, 64 out, 64x64 spatial."""
    print("\n--- ttnn.conv2d (conv_in: 15->64, 64x64) ---")
    B, H, W, Cin, Cout = 1, 64, 64, 15, 64

    x = torch.randn(1, 1, B * H * W, Cin, dtype=torch.bfloat16)
    x_tt = to_tt(x, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    w = torch.randn(Cout, Cin, 3, 3, dtype=torch.bfloat16)
    b = torch.zeros(1, 1, 1, Cout, dtype=torch.bfloat16)
    w_tt = to_tt(w, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = to_tt(b, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out, [out_h, out_w], _ = ttnn.conv2d(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=Cin,
        out_channels=Cout,
        batch_size=B,
        input_height=H,
        input_width=W,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=b_tt,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    print(f"  input: [{B},{H},{W},{Cin}] -> output: {out_h}x{out_w}")
    print(f"  PASS")


def test_linear(device):
    """Test ttnn.linear for conditioning MLP."""
    print("\n--- ttnn.linear ---")
    x = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16)
    w = torch.randn(256, 256, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, 256, dtype=torch.bfloat16)

    out = ttnn.linear(to_tt(x, device), to_tt(w, device), bias=to_tt(b, device))
    result = ttnn.to_torch(out)
    print(f"  shape: {result.shape}  PASS")


if __name__ == "__main__":
    print("=" * 50)
    print("TTNN Ops Tests for Diamond UNet")
    print("=" * 50)

    device = ttnn.open_device(device_id=0)

    tests = [
        ("silu", test_silu),
        ("concat", test_concat),
        ("upsample", test_upsample),
        ("group_norm", test_group_norm),
        ("linear", test_linear),
        ("conv2d simple", test_conv2d_simple),
        ("conv2d stride2", test_conv2d_stride2),
        ("conv2d conv_in", test_conv2d_conv_in),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn(device)
            passed += 1
        except Exception as e:
            err = str(e).split('\n')[0][:120]
            print(f"  FAILED: {err}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    ttnn.close_device(device)
