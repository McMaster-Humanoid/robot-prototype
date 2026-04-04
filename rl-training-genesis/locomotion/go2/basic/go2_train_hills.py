import argparse
import gc
import os
import pickle
import shutil
from importlib import metadata

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

import torch
import genesis as gs

from go2_env_terrain import Go2EnvTerrain


def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "noise_std_type": "log",
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 0.5],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }
    terrain_cfg = {
        "enabled": True,
        "terrain_type": "hills",
        "size": 8.0,
        "resolution": 0.1,
        "change_interval": 500,
        "difficulty": 0,  # overridden per stage
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg


def difficulty_for_stage(stage: int, num_stages: int) -> int:
    """Spread difficulties 0–9 evenly across stages."""
    if num_stages <= 1:
        return 0
    return min(round(stage * 9 / (num_stages - 1)), 9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-hills")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=2000,
                        help="Total training iterations across all curriculum stages")
    parser.add_argument("--change_interval", type=int, default=500,
                        help="Iterations per curriculum stage before terrain gets harder")
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    if args.backend == "gpu":
        gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)
    else:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    num_stages = max(1, args.max_iterations // args.change_interval)
    weights_path = None  # updated after each stage

    print(f"\n{'='*70}")
    print(f"CURRICULUM TRAINING: {num_stages} stages × {args.change_interval} iterations")
    print(f"Terrain difficulty progression: ", end="")
    print(" → ".join(str(difficulty_for_stage(s, num_stages)) for s in range(num_stages)))
    print(f"{'='*70}\n")

    for stage in range(num_stages):
        difficulty = difficulty_for_stage(stage, num_stages)

        print(f"\n{'='*70}")
        print(f"STAGE {stage + 1}/{num_stages}  |  Difficulty {difficulty}/9")
        print(f"{'='*70}")

        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
        terrain_cfg["difficulty"] = difficulty
        terrain_cfg["change_interval"] = args.change_interval
        train_cfg = get_train_cfg(args.exp_name, args.change_interval)

        # Each stage logs to its own subdirectory so checkpoints don't collide
        stage_log_dir = os.path.join(log_dir, f"stage_{stage:02d}_diff_{difficulty}")
        os.makedirs(stage_log_dir, exist_ok=True)

        # Save full config for eval (final stage config is what eval will use)
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg],
            open(os.path.join(stage_log_dir, "cfgs.pkl"), "wb"),
        )

        env = Go2EnvTerrain(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            terrain_cfg=terrain_cfg,
            show_viewer=args.vis,
        )

        runner = OnPolicyRunner(env, train_cfg, stage_log_dir, device=gs.device)

        # Transfer weights from the previous stage
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=gs.device)
            runner.alg.actor_critic.load_state_dict(state_dict)
            print(f"[Curriculum] Loaded weights from stage {stage} → continuing on harder terrain")

        env.reset()
        runner.learn(num_learning_iterations=args.change_interval, init_at_random_ep_len=True)

        # Save weights so the next stage can load them
        weights_path = os.path.join(log_dir, f"weights_stage_{stage:02d}.pt")
        torch.save(runner.alg.actor_critic.state_dict(), weights_path)
        print(f"[Curriculum] Stage {stage + 1} complete. Weights saved → {weights_path}")

        # Free the Genesis scene before building the next one
        del runner
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write a symlink/copy of final weights to a predictable path for eval
    final_weights = os.path.join(log_dir, "weights_final.pt")
    import shutil as _shutil
    _shutil.copy2(weights_path, final_weights)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"Final weights: {final_weights}")
    print(f"To evaluate:   python go2_eval.py -e {args.exp_name} --stage {num_stages - 1}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
