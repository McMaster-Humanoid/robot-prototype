

import argparse
import os
import torch

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from go2_balancingBall_kf_env import Go2BalancingBallEnv


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-N", "--num_envs",        type=int,   default=64)
    p.add_argument("-M", "--max_iterations",  type=int,   default=100)
    p.add_argument("-s", "--env_spacing", type=float, default=3.0)
    p.add_argument("-B", "--backend",         type=str,   default="gpu")
    p.add_argument("-e", "--exp_name", type=str,   default="go2_ball_kf")
    p.add_argument("-b", "--ball_urdf", type=str,   default="models/ball.urdf")
    p.add_argument("-r", "--resume",    type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("-l", "--log_dir",   type=str,   default=None,
                   help="Log directory (default: logs/<exp_name>)")
    p.add_argument("-v", "--vis",       action="store_true", help="Visualize", default=False)
    return p.parse_args()


def make_train_cfg(args):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
    }


class EnvWrapper:

    def __init__(self, env: Go2BalancingBallEnv):
        self.env = env
        self.num_envs         = env.num_envs
        self.num_obs          = Go2BalancingBallEnv.NUM_OBS
        self.num_actions      = Go2BalancingBallEnv.NUM_ACT
        self.num_privileged_obs = None
        self.device           = env.device
        self.max_episode_length = Go2BalancingBallEnv.MAX_EPISODE_LEN

    @property
    def episode_length_buf(self):
        return self.env.episode_len_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_len_buf.copy_(value)

    def step(self, actions):
        obs, rew, done, info = self.env.step(actions)
        infos = {**info, "observations": {}}
        return obs, rew, done, infos

    def reset(self):
        self.env.reset()
        return self.env.obs_buf.clone(), None

    def get_observations(self):
        obs = self.env.obs_buf.clone()
        extras = {"observations": {}}
        return obs, extras


def main():
    args = get_args()
    args.log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    use_gpu = args.backend == "gpu" and torch.cuda.is_available()
    if args.backend == "gpu" and not torch.cuda.is_available():
        print("[train] CUDA not available, using CPU.")
    args.device = "cuda" if use_gpu else "cpu"

    backend = gs.gpu if use_gpu else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning", performance_mode=True)

    os.makedirs(args.log_dir, exist_ok=True)

    print("[train] Creating environment...")
    env = Go2BalancingBallEnv(
        num_envs=args.num_envs,
        env_spacing=args.env_spacing,
        show_viewer=args.vis,
        device=args.device,
        ball_urdf=args.ball_urdf,
    )
    env.reset()

    wrapped = EnvWrapper(env)

    train_cfg = make_train_cfg(args)

    runner = OnPolicyRunner(
        env=wrapped,
        train_cfg=train_cfg,
        log_dir=args.log_dir,
        device=args.device,
    )

    if args.resume:
        runner.load(args.resume)

    print(f"[train] Starting PPO training for {args.max_iterations} iterations …")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("[train] Done. Model saved to:", args.log_dir)

    if args.vis:
        print("[train] Viewer open. Press Ctrl+C to exit.")
        policy = runner.get_inference_policy(device=args.device)
        try:
            with torch.no_grad():
                obs, _ = wrapped.reset()
                while True:
                    actions = policy(obs)
                    obs, _, done, _ = wrapped.step(actions)
                    if done.any():
                        obs, _ = wrapped.reset()
        except KeyboardInterrupt:
            print("Exiting...")


if __name__ == "__main__":
    main()