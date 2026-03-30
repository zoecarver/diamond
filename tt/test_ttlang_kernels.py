"""Test TT-Lang kernels in the simulator."""

import torch
import ttnn
from kernels import silu_kernel, add_kernel, mul_kernel, adaln_modulate_kernel, precondition_kernel, euler_step_kernel

DRAM = ttnn.DRAM_MEMORY_CONFIG


def to_tt(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=DRAM)


def zeros_tt(shape, device):
    return to_tt(torch.zeros(shape, dtype=torch.bfloat16), device)


def test_silu(device):
    print("--- SiLU kernel ---")
    x = torch.randn(64, 64, dtype=torch.bfloat16)
    out = zeros_tt((64, 64), device)
    silu_kernel(to_tt(x, device), out)
    result = ttnn.to_torch(out)
    ref = (x.float() * torch.sigmoid(x.float())).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  max_diff={diff:.6f}  PASS={diff < 0.1}")
    assert diff < 0.1, f"SiLU failed: {diff}"


def test_add(device):
    print("--- Add kernel ---")
    a = torch.randn(64, 128, dtype=torch.bfloat16)
    b = torch.randn(64, 128, dtype=torch.bfloat16)
    out = zeros_tt((64, 128), device)
    add_kernel(to_tt(a, device), to_tt(b, device), out)
    result = ttnn.to_torch(out)
    ref = (a.float() + b.float()).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  max_diff={diff:.6f}  PASS={diff < 0.01}")
    assert diff < 0.01, f"Add failed: {diff}"


def test_mul(device):
    print("--- Mul kernel ---")
    a = torch.randn(64, 128, dtype=torch.bfloat16)
    b = torch.randn(64, 128, dtype=torch.bfloat16)
    out = zeros_tt((64, 128), device)
    mul_kernel(to_tt(a, device), to_tt(b, device), out)
    result = ttnn.to_torch(out)
    ref = (a.float() * b.float()).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  max_diff={diff:.6f}  PASS={diff < 0.01}")
    assert diff < 0.01, f"Mul failed: {diff}"


def test_adaln(device):
    print("--- AdaLN modulate kernel ---")
    x = torch.randn(64, 64, dtype=torch.bfloat16)
    shift = torch.randn(64, 64, dtype=torch.bfloat16) * 0.1
    scale = torch.randn(64, 64, dtype=torch.bfloat16) * 0.1
    out = zeros_tt((64, 64), device)
    adaln_modulate_kernel(to_tt(x, device), to_tt(shift, device), to_tt(scale, device), out)
    result = ttnn.to_torch(out)
    ref = (x.float() * (scale.float() + 1.0) + shift.float()).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  max_diff={diff:.6f}  PASS={diff < 0.1}")
    assert diff < 0.1, f"AdaLN failed: {diff}"


def test_precondition(device):
    print("--- Precondition kernel ---")
    noisy = torch.randn(64, 64, dtype=torch.bfloat16)
    model_out = torch.randn(64, 64, dtype=torch.bfloat16)
    c_skip = torch.full((64, 64), 0.8, dtype=torch.bfloat16)
    c_out = torch.full((64, 64), 0.3, dtype=torch.bfloat16)
    out = zeros_tt((64, 64), device)
    precondition_kernel(
        to_tt(noisy, device), to_tt(model_out, device),
        to_tt(c_skip, device), to_tt(c_out, device), out,
    )
    result = ttnn.to_torch(out)
    ref = (c_skip.float() * noisy.float() + c_out.float() * model_out.float()).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  max_diff={diff:.6f}  PASS={diff < 0.1}")
    assert diff < 0.1, f"Precondition failed: {diff}"


def test_euler_step(device):
    print("--- Euler step kernel ---")
    x = torch.randn(64, 64, dtype=torch.bfloat16)
    denoised = torch.randn(64, 64, dtype=torch.bfloat16)
    dt_over_sigma = torch.full((64, 64), -0.5, dtype=torch.bfloat16)
    out = zeros_tt((64, 64), device)
    euler_step_kernel(
        to_tt(x, device), to_tt(denoised, device),
        to_tt(dt_over_sigma, device), out,
    )
    result = ttnn.to_torch(out)
    ref = (x.float() + dt_over_sigma.float() * (x.float() - denoised.float())).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  max_diff={diff:.6f}  PASS={diff < 0.1}")
    assert diff < 0.1, f"Euler step failed: {diff}"


def test_larger_shapes(device):
    print("--- SiLU on larger tensor (128x256) ---")
    x = torch.randn(128, 256, dtype=torch.bfloat16)
    out = zeros_tt((128, 256), device)
    silu_kernel(to_tt(x, device), out)
    result = ttnn.to_torch(out)
    ref = (x.float() * torch.sigmoid(x.float())).to(torch.bfloat16)
    diff = (result.float() - ref.float()).abs().max().item()
    print(f"  max_diff={diff:.6f}  PASS={diff < 0.1}")
    assert diff < 0.1, f"Large SiLU failed: {diff}"


if __name__ == "__main__":
    print("=" * 50)
    print("TT-Lang Kernel Tests")
    print("=" * 50)

    device = ttnn.open_device(device_id=0)

    try:
        test_silu(device)
        test_add(device)
        test_mul(device)
        test_adaln(device)
        test_precondition(device)
        test_euler_step(device)
        test_larger_shapes(device)
        print("\n" + "=" * 50)
        print("ALL TT-LANG KERNEL TESTS PASSED")
        print("=" * 50)
    except AssertionError as e:
        print(f"\nFAILED: {e}")
    finally:
        ttnn.close_device(device)
