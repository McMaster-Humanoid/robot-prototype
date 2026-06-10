import argparse
import os
import time
import torch
import numpy as np

import genesis as gs
from go2_balancingBall_kf_env import Go2BalancingBallEnv


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", type=str, required=True,
                   help="Path to a trained model_*.pt checkpoint")
    p.add_argument("--num_envs",    type=int,   default=1)
    p.add_argument("--headless",    action="store_true",
                   help="Run without the Genesis viewer")
    p.add_argument("--device",      type=str,   default=None,
                   help="Device (default: cuda if available else cpu)")
    p.add_argument("--ball_urdf",   type=str,   default="models/ball.urdf")
    p.add_argument("--episodes",    type=int,   default=5,
                   help="Number of episodes to evaluate")
    p.add_argument("--max_steps",   type=int,   default=1000)
    return p.parse_args()


class SimplePolicy(torch.nn.Module):

    def __init__(self, num_obs: int, num_act: int, hidden=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = num_obs
        for h in hidden:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, num_act))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

def resolve_checkpoint(path: str) -> str:
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    if os.path.isfile(path):
        return os.path.abspath(path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt = os.path.join(script_dir, path)
    if os.path.isfile(alt):
        return alt
    return path

def load_policy(checkpoint_path: str, device: str) -> SimplePolicy:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "actor_critic" in ckpt:
        state = ckpt["actor_critic"]
    else:
        state = ckpt

    actor_state = {
        k.replace("actor.", ""): v
        for k, v in state.items()
        if k.startswith("actor.")
    }

    policy = SimplePolicy(
        num_obs=Go2BalancingBallEnv.NUM_OBS,
        num_act=Go2BalancingBallEnv.NUM_ACT,
    ).to(device)
    policy.load_state_dict(actor_state, strict=False)
    policy.eval()
    return policy


def evaluate(args):
    print(f"[test] Loading checkpoint: {args.checkpoint}")
    policy = load_policy(args.checkpoint, args.device)

    print(f"[test] Creating environment (num_envs={args.num_envs}, viewer={not args.headless})")
    env = Go2BalancingBallEnv(
        num_envs=args.num_envs,
        env_spacing=3.0,
        show_viewer=not args.headless,
        device=args.device,
        ball_urdf=args.ball_urdf,
    )
    env.reset()

    all_ep_returns   = []
    all_ep_lengths   = []
    all_ball_dists   = []

    for ep in range(args.episodes):
        env.reset()
        ep_return = torch.zeros(args.num_envs, device=args.device)
        done_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        step = 0
        ball_dist_log = []

        while step < args.max_steps and not done_mask.all():
            with torch.no_grad():
                obs = env.obs_buf.clone()
                actions = policy(obs).clamp(-1.0, 1.0)

            obs, rew, done, info = env.step(actions)

            ep_return += rew * (~done_mask).float()
            done_mask |= done

            kf_pos = env.kf.position                         # (N, 2)
            ball_dist_log.append(kf_pos.norm(dim=1).mean().item())

            step += 1
            if not args.headless:
                time.sleep(env.DT * 0.5)

        ep_len   = info["episode_len"].float().mean().item()
        ep_ret   = ep_return.mean().item()
        mean_bd  = np.mean(ball_dist_log)

        all_ep_returns.append(ep_ret)
        all_ep_lengths.append(ep_len)
        all_ball_dists.append(mean_bd)

        print(f"  Episode {ep+1:3d}: return={ep_ret:8.2f}  "
              f"length={ep_len:6.1f}  "
              f"mean_ball_dist={mean_bd:.4f} m")

    print("\n[test] Summary over {} episodes:".format(args.episodes))
    print(f"  Mean return    : {np.mean(all_ep_returns):.2f} ± {np.std(all_ep_returns):.2f}")
    print(f"  Mean ep length : {np.mean(all_ep_lengths):.1f}")
    print(f"  Mean ball dist : {np.mean(all_ball_dists):.4f} m  "
          f"(target < {Go2BalancingBallEnv.TERMINATION_BALL_DIST:.2f} m)")

    if not args.headless:
        print("\n[test] Viewer open. Press Ctrl+C to exit.")
        try:
            with torch.no_grad():
                obs = env.obs_buf.clone()
                while True:
                    actions = policy(obs).clamp(-1.0, 1.0)
                    obs, rew, done, info = env.step(actions)
                    if done.any():
                        env.reset()
                        obs = env.obs_buf.clone()
                    time.sleep(env.DT * 0.5)
        except KeyboardInterrupt:
            print("Exiting...")


if __name__ == "__main__":
    args = get_args()
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    if not os.path.isfile(args.checkpoint):
        print(f"[eval] Error: checkpoint not found: {args.checkpoint}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "logs", "go2_ball_kf")
        pts = []
        if os.path.isdir(log_dir):
            pts = sorted(f for f in os.listdir(log_dir) if f.endswith(".pt"))
            if pts:
                print(f"  Available in logs/go2_ball_kf/:")
                for p in pts:
                    print(f"    -c logs/go2_ball_kf/{p}")
            else:
                print("  No .pt checkpoints in logs/go2_ball_kf yet.")
        if not pts:
            print("  Run training first:  python go2_balancingBall_kf_train.py")
            print("  (Checkpoints save every 100 iters; model_latest.pt at the end.)")
        raise SystemExit(1)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cpu":
        print("[eval] Using CPU (CUDA not available or --device cpu)")
    backend = gs.gpu if args.device == "cuda" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning", performance_mode=True)
    evaluate(args)