"""Check weight keys match our expected structure."""
import torch

# Download
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="eloialonso/diamond",
                       filename="atari_100k/models/Breakout.pt")

ckpt = torch.load(path, map_location="cpu", weights_only=False)
prefix = "denoiser."
sd = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

print(f"Total denoiser params: {len(sd)}")
print(f"Num actions: {sd['inner_model.act_emb.0.weight'].shape[0]}")

# Check key prefixes
expected = [
    "inner_model.noise_emb.weight",
    "inner_model.act_emb.0.weight",
    "inner_model.cond_proj.0.weight",
    "inner_model.conv_in.weight",
    "inner_model.conv_in.bias",
    "inner_model.unet.d_blocks.0.resblocks.0.norm1.linear.weight",
    "inner_model.unet.d_blocks.0.resblocks.0.conv1.weight",
    "inner_model.unet.d_blocks.0.resblocks.0.conv2.weight",
    "inner_model.unet.downsamples.1.conv.weight",
    "inner_model.unet.mid_blocks.resblocks.0.norm1.linear.weight",
    "inner_model.unet.mid_blocks.resblocks.0.attn.norm.norm.weight",
    "inner_model.unet.mid_blocks.resblocks.0.attn.qkv_proj.weight",
    "inner_model.unet.mid_blocks.resblocks.0.attn.out_proj.weight",
    "inner_model.unet.u_blocks.0.resblocks.0.norm1.linear.weight",
    "inner_model.unet.u_blocks.0.resblocks.0.proj.weight",
    "inner_model.unet.upsamples.1.conv.weight",
    "inner_model.norm_out.norm.weight",
    "inner_model.conv_out.weight",
]

print("\nKey check:")
for k in expected:
    found = k in sd
    shape = sd[k].shape if found else "MISSING"
    print(f"  {'OK' if found else 'XX'} {k}: {shape}")

# List all u_blocks indices
u_keys = sorted(set(k.split(".resblocks")[0] for k in sd if "u_blocks" in k))
print(f"\nu_blocks found: {u_keys}")

up_keys = sorted(k for k in sd if "upsamples" in k)
print(f"upsample keys: {up_keys}")

ds_keys = sorted(k for k in sd if "downsamples" in k)
print(f"downsample keys: {ds_keys}")
