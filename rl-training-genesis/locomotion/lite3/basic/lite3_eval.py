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

from lite3_env import Lite3Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="lite3-walking")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Lite3Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.ckpt == -1:
        # find the highest numbered checkpoint
        pts = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        ckpt = max(int(f[6:-3]) for f in pts)
        print(f"Loading latest checkpoint: model_{ckpt}.pt")
    else:
        ckpt = args.ckpt
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    step_count = 0
    with torch.no_grad():
        try:
            while True:
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
                
                step_count += 1
                # Print telemetry every 10 simulation steps
                if step_count % 10 == 0:
                    cmd_x = env.commands[0, 0].item()
                    vel_x = env.base_lin_vel[0, 0].item()
                    pitch = env.base_euler[0, 1].item()
                    rew_val = rews[0].item()
                    print(f"[LOG] Step {step_count:04d} | Cmd Vx: {cmd_x:.2f} | Real Vx: {vel_x:.2f} | Pitch: {pitch:.1f}° | Rew: {rew_val:.3f}")
        except gs.GenesisException:
            pass  # viewer was closed


if __name__ == "__main__":
    main()

"""
# evaluation (loads latest checkpoint automatically)
python rl-training-genesis/locomotion/lite3/basic/lite3_eval.py -e lite3-walking -v
# or a specific checkpoint
python rl-training-genesis/locomotion/lite3/basic/lite3_eval.py -e lite3-walking -v --ckpt 1000
"""
