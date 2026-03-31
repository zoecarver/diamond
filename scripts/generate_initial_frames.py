"""
Generate 4 initial conditioning frames for each Atari 100k game.

For each game, resets the environment with fixed seed, takes a few FIRE actions
to start gameplay, then captures 4 frames using the same preprocessing as
DIAMOND (64x64, frame_skip=4, max-pooled, [-1,1] float range).

Outputs per game:
  results/data/{game}/initial_frames.pt  -- tensor [1, 4, 3, 64, 64] in [-1,1]
  results/data/{game}/initial_actions.pt -- tensor [1, 4] of action indices
  results/data/{game}/frame_{i}.png      -- PNG preview of each frame
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Import AtariPreprocessing directly to avoid pulling in the full src dependency chain
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "atari_preprocessing",
    str(Path(__file__).resolve().parent.parent / "src" / "envs" / "atari_preprocessing.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AtariPreprocessing = _mod.AtariPreprocessing

import ale_py
import gymnasium


ATARI_100K_GAMES = [
    "Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone",
    "Boxing", "Breakout", "ChopperCommand", "CrazyClimber", "DemonAttack",
    "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo",
    "Krull", "KungFuMaster", "MsPacman", "Pong", "PrivateEye", "Qbert",
    "RoadRunner", "Seaquest", "UpNDown",
]

IMG_SIZE = 64
SEED = 42
NUM_FRAMES = 4


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC observation to float CHW tensor in [-1, 1]."""
    return torch.from_numpy(obs).float().div(255).mul(2).sub(1).permute(2, 0, 1)


def generate_frames_for_game(game: str, out_dir: Path):
    env_id = f"{game}NoFrameskip-v4"
    env = gymnasium.make(env_id, full_action_space=False, frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env, noop_max=0, frame_skip=4, screen_size=IMG_SIZE)

    obs, info = env.reset(seed=SEED)
    action_meanings = env.unwrapped.get_action_meanings()
    fire_idx = action_meanings.index("FIRE") if "FIRE" in action_meanings else 0

    # Many games require FIRE to start. Press it a couple times.
    for _ in range(2):
        obs, _, terminated, truncated, _ = env.step(fire_idx)
        if terminated or truncated:
            obs, _ = env.reset(seed=SEED)

    frames = []
    actions = []
    frames.append(obs_to_tensor(obs))
    actions.append(fire_idx)

    # Collect remaining frames with NOOP (action 0) to get natural start state
    for _ in range(NUM_FRAMES - 1):
        obs, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            obs, _ = env.reset(seed=SEED)
        frames.append(obs_to_tensor(obs))
        actions.append(0)

    env.close()

    obs_tensor = torch.stack(frames).unsqueeze(0)   # [1, 4, 3, 64, 64]
    act_tensor = torch.tensor(actions).unsqueeze(0)  # [1, 4]

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(obs_tensor, out_dir / "initial_frames.pt")
    torch.save(act_tensor, out_dir / "initial_actions.pt")

    for i, frame in enumerate(frames):
        img = frame.add(1).div(2).clamp(0, 1).mul(255).byte()
        img = img.permute(1, 2, 0).numpy()
        Image.fromarray(img, "RGB").save(out_dir / f"frame_{i}.png")

    print(f"  {game}: obs {list(obs_tensor.shape)}, "
          f"range [{obs_tensor.min():.2f}, {obs_tensor.max():.2f}], "
          f"actions {actions}")


def main():
    base_dir = Path(__file__).resolve().parent.parent / "results" / "data"
    print(f"Generating {NUM_FRAMES} initial frames for {len(ATARI_100K_GAMES)} games")
    print(f"Output: {base_dir}/{{game}}/\n")

    for game in ATARI_100K_GAMES:
        generate_frames_for_game(game, base_dir / game)

    print(f"\nDone. Generated frames for {len(ATARI_100K_GAMES)} games.")


if __name__ == "__main__":
    main()
