import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
from go2_env_terrain import Go2EnvTerrain


def load_cfgs(log_dir: str, stage: int = None):
    """
    Load configs from a log directory.
    Handles both flat runs (5-item pickle) and terrain/curriculum runs (6-item pickle).
    If stage is given, loads from the matching stage subdirectory.
    """
    if stage is not None:
        # Find the stage subdirectory
        candidates = [
            d for d in os.listdir(log_dir)
            if d.startswith(f"stage_{stage:02d}_")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No stage directory matching 'stage_{stage:02d}_*' found in {log_dir}"
            )
        cfg_path = os.path.join(log_dir, candidates[0], "cfgs.pkl")
    else:
        cfg_path = os.path.join(log_dir, "cfgs.pkl")

    data = pickle.load(open(cfg_path, "rb"))

    if len(data) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg = data
    elif len(data) == 5:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = data
        terrain_cfg = None
    else:
        raise ValueError(f"Unexpected pickle length {len(data)} in {cfg_path}")

    return env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100,
                        help="Checkpoint iteration to load (used for flat runs)")
    parser.add_argument("--stage", type=int, default=None,
                        help="Curriculum stage to evaluate (for go2-hills runs). "
                             "Use --weights to point at a specific .pt file instead.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Direct path to a .pt weights file (overrides --ckpt/--stage lookup)")
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu)

    log_dir = f"logs/{args.exp_name}"

    env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg = load_cfgs(
        log_dir, stage=args.stage
    )
    reward_cfg["reward_scales"] = {}

    # Build the right environment type
    if terrain_cfg is not None and terrain_cfg.get("enabled", False):
        print(f"[Eval] Loading terrain env — difficulty {terrain_cfg.get('difficulty', 0)}")
        env = Go2EnvTerrain(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            terrain_cfg=terrain_cfg,
            show_viewer=True,
        )
    else:
        print("[Eval] Loading flat env")
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Resolve which weights to load
    if args.weights:
        weights_path = args.weights
    elif args.stage is not None:
        weights_path = os.path.join(log_dir, f"weights_stage_{args.stage:02d}.pt")
        if not os.path.exists(weights_path):
            # Fall back to final weights
            weights_path = os.path.join(log_dir, "weights_final.pt")
    else:
        # Classic flat-run checkpoint
        weights_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"[Eval] Loading weights from: {weights_path}")

    # Curriculum runs save raw state_dicts; flat runs use runner.load()
    if weights_path.endswith("weights_stage") or "weights_" in os.path.basename(weights_path):
        state_dict = torch.load(weights_path, map_location=gs.device)
        runner.alg.actor_critic.load_state_dict(state_dict)
    else:
        runner.load(weights_path)

    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()

"""
# Evaluate a flat walking run:
python go2_eval.py -e go2-walking --ckpt 100

# Evaluate the final stage of a curriculum hills run:
python go2_eval.py -e go2-hills --stage 3

# Evaluate a specific stage:
python go2_eval.py -e go2-hills --stage 1

# Evaluate with explicit weights file:
python go2_eval.py -e go2-hills --weights logs/go2-hills/weights_final.pt
"""
