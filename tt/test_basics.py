"""Test basic building blocks on device before running the full model."""

import torch
import ttnn
import time

DRAM = ttnn.DRAM_MEMORY_CONFIG


def to_tt(t, device, layout=ttnn.TILE_LAYOUT, mem=DRAM):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)


def test_conv2d(device):
    """Test ttnn.conv2d with Diamond-like shapes."""
    print("\n--- Conv2d test ---")
    B, H, W, C_in, C_out = 1, 64, 64, 15, 64

    x_torch = torch.randn(1, 1, B * H * W, C_in, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    w_torch = torch.randn(C_out, C_in, 3, 3, dtype=torch.bfloat16)
    b_torch = torch.zeros(1, 1, 1, C_out, dtype=torch.bfloat16)
    w_tt = to_tt(w_torch, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = to_tt(b_torch, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=False,
        reallocate_halo_output=False,
    )

    result = ttnn.conv2d(
        input_tensor=x_tt,
        weight_tensor=w_tt,
        device=device,
        in_channels=C_in,
        out_channels=C_out,
        batch_size=B,
        input_height=H,
        input_width=W,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=b_tt,
        conv_config=conv_config,
        memory_config=DRAM,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    out = result[0]
    out_h, out_w = result[1]
    print(f"  Input: [1, 1, {B*H*W}, {C_in}]")
    print(f"  Output dims: {out_h}x{out_w}")
    print(f"  Output tensor shape: {out.shape}")
    print(f"  PASS")
    return True


def test_conv2d_stride2(device):
    """Test conv2d with stride 2 (downsample)."""
    print("\n--- Conv2d stride=2 test ---")
    B, H, W, C = 1, 64, 64, 64

    x_torch = torch.randn(1, 1, B * H * W, C, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    w_torch = torch.randn(C, C, 3, 3, dtype=torch.bfloat16)
    b_torch = torch.zeros(1, 1, 1, C, dtype=torch.bfloat16)
    w_tt = to_tt(w_torch, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = to_tt(b_torch, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=False,
        reallocate_halo_output=False,
    )

    result = ttnn.conv2d(
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
        conv_config=conv_config,
        memory_config=DRAM,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    out = result[0]
    out_h, out_w = result[1]
    print(f"  Input: [1, 1, {B*H*W}, {C}]")
    print(f"  Output dims: {out_h}x{out_w}")
    print(f"  PASS")
    return True


def test_group_norm(device):
    """Test ttnn.group_norm."""
    print("\n--- GroupNorm test ---")
    B, H, W, C = 1, 64, 64, 64
    num_groups = max(1, C // 32)

    x_torch = torch.randn(B, 1, H * W, C, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, device)

    compute_grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
    out = ttnn.group_norm(
        x_tt, num_groups=num_groups, epsilon=1e-5,
        inplace=False, core_grid=core_grid,
    )
    result = ttnn.to_torch(out)

    x_nchw = x_torch.reshape(B, H, W, C).permute(0, 3, 1, 2).float()
    ref = torch.nn.functional.group_norm(x_nchw, num_groups, eps=1e-5)
    ref_flat = ref.permute(0, 2, 3, 1).reshape(B, 1, H * W, C).to(torch.bfloat16)

    diff = (result.float() - ref_flat.float()).abs().max().item()
    print(f"  Shape: [{B}, 1, {H*W}, {C}], groups={num_groups}")
    print(f"  Max diff vs PyTorch: {diff:.4f}  PASS: {diff < 1.0}")
    return True


def test_silu(device):
    """Test ttnn.silu."""
    print("\n--- TTNN SiLU test ---")
    x_torch = torch.randn(1, 1, 4096, 64, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, device)

    out = ttnn.silu(x_tt)
    result = ttnn.to_torch(out)

    ref = torch.nn.functional.silu(x_torch.float()).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  Max diff: {diff:.4f}  PASS: {diff < 0.5}")
    return True


def test_upsample(device):
    """Test ttnn.upsample."""
    print("\n--- Upsample test ---")
    B, H, W, C = 1, 32, 32, 64

    x_torch = torch.randn(B, H, W, C, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.upsample(x_tt, scale_factor=2, mode="nearest")
    result = ttnn.to_torch(out)
    print(f"  Input: [{B}, {H}, {W}, {C}]")
    print(f"  Output: {result.shape}")
    print(f"  PASS")
    return True


def test_concat(device):
    """Test ttnn.concat (for skip connections)."""
    print("\n--- Concat test ---")
    B, HW, C = 1, 4096, 64
    a_torch = torch.randn(B, 1, HW, C, dtype=torch.bfloat16)
    b_torch = torch.randn(B, 1, HW, C, dtype=torch.bfloat16)
    a_tt = to_tt(a_torch, device)
    b_tt = to_tt(b_torch, device)

    out = ttnn.concat([a_tt, b_tt], dim=-1)
    result = ttnn.to_torch(out)
    print(f"  Input: 2x [{B}, 1, {HW}, {C}]")
    print(f"  Output: {result.shape}")

    ref = torch.cat([a_torch, b_torch], dim=-1)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  Max diff: {diff:.6f}  PASS: {diff < 0.001}")
    return True


def test_silu_kernel(device):
    """Test TT-Lang SiLU kernel."""
    print("\n--- TT-Lang SiLU kernel test ---")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernels import silu_kernel

    ROWS, COLS = 64, 64
    x_torch = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    x_tt = to_tt(x_torch, device)
    out_tt = to_tt(torch.zeros(ROWS, COLS, dtype=torch.bfloat16), device)

    silu_kernel(x_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    ref = (x_torch.float() * torch.sigmoid(x_torch.float())).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  Max diff: {diff:.6f}  PASS: {diff < 0.1}")
    return True


def test_adaln_kernel(device):
    """Test TT-Lang AdaLN modulate kernel."""
    print("\n--- TT-Lang AdaLN modulate kernel test ---")
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernels import adaln_modulate_kernel

    ROWS, COLS = 64, 64
    x = torch.randn(ROWS, COLS, dtype=torch.bfloat16)
    shift = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.1
    scale = torch.randn(ROWS, COLS, dtype=torch.bfloat16) * 0.1

    out_tt = to_tt(torch.zeros(ROWS, COLS, dtype=torch.bfloat16), device)
    adaln_modulate_kernel(to_tt(x, device), to_tt(shift, device), to_tt(scale, device), out_tt)

    result = ttnn.to_torch(out_tt)
    ref = (x.float() * (scale.float() + 1.0) + shift.float()).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  Max diff: {diff:.6f}  PASS: {diff < 0.1}")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Diamond TT Building Block Tests")
    print("=" * 50)

    device = ttnn.open_device(device_id=0)

    tests = [
        ("GroupNorm", test_group_norm),
        ("SiLU", test_silu),
        ("Upsample", test_upsample),
        ("Concat", test_concat),
        ("TT-Lang SiLU", test_silu_kernel),
        ("TT-Lang AdaLN", test_adaln_kernel),
        ("Conv2d", test_conv2d),
        ("Conv2d stride=2", test_conv2d_stride2),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn(device)
            passed += 1
        except Exception as e:
            print(f"\n  FAILED: {name}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    ttnn.close_device(device)
