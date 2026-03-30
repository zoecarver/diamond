"""Test group_norm at 32x32 spatial (the failing size)."""
import torch
import ttnn

device = ttnn.open_device(device_id=0, l1_small_size=32768)
grid = device.compute_with_storage_grid_size()
print(f"Grid: {grid}", flush=True)

# 64x64 spatial - this worked
print("\ngroup_norm at 64x64...", flush=True)
x64 = ttnn.from_torch(
    torch.randn(1, 1, 4096, 64, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
out64 = ttnn.group_norm(x64, num_groups=2, epsilon=1e-5,
                         inplace=False, core_grid=core_grid)
print(f"  OK: {out64.shape}", flush=True)

# 32x32 spatial - this might fail
print("\ngroup_norm at 32x32...", flush=True)
x32 = ttnn.from_torch(
    torch.randn(1, 1, 1024, 64, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
try:
    out32 = ttnn.group_norm(x32, num_groups=2, epsilon=1e-5,
                             inplace=False, core_grid=core_grid)
    print(f"  OK: {out32.shape}", flush=True)
except Exception as e:
    print(f"  FAILED with full grid. Trying smaller grid...", flush=True)
    # Try smaller core grid
    for y in range(grid.y, 0, -1):
        for x in range(grid.x, 0, -1):
            try:
                cg = ttnn.CoreGrid(y=y, x=x)
                out32 = ttnn.group_norm(x32, num_groups=2, epsilon=1e-5,
                                         inplace=False, core_grid=cg)
                print(f"  OK with grid ({x},{y}): {out32.shape}", flush=True)
                break
            except:
                continue
        else:
            continue
        break

# 16x16
print("\ngroup_norm at 16x16...", flush=True)
x16 = ttnn.from_torch(
    torch.randn(1, 1, 256, 64, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
try:
    out16 = ttnn.group_norm(x16, num_groups=2, epsilon=1e-5,
                             inplace=False, core_grid=core_grid)
    print(f"  OK: {out16.shape}", flush=True)
except Exception as e:
    print(f"  FAILED: {str(e)[:100]}", flush=True)

# 8x8
print("\ngroup_norm at 8x8...", flush=True)
x8 = ttnn.from_torch(
    torch.randn(1, 1, 64, 64, dtype=torch.bfloat16),
    ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
try:
    out8 = ttnn.group_norm(x8, num_groups=2, epsilon=1e-5,
                            inplace=False, core_grid=core_grid)
    print(f"  OK: {out8.shape}", flush=True)
except Exception as e:
    print(f"  FAILED: {str(e)[:100]}", flush=True)

print("\nDone.", flush=True)
ttnn.close_device(device)
