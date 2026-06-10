import argparse
import os
import pickle
import shutil
import torch
from importlib import metadata

try:
    metadata.version("rsl-rl-lib")
except metadata.PackageNotFoundError:
    print("Warning: rsl-rl-lib not found, ensure it is installed.")

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from g1_locomotion_rg_env import Go2Env


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,           
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0005,        
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
            "init_noise_std": 0.5,          
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
        "save_interval": 50,                
        "empirical_normalization": None,
        "seed": 1,
    }
    
    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12, 
        "default_joint_angles": { 
            # Slight bend in knees and hips helps stability
            "left_hip_pitch_joint": -0.2,       
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.45,             
            "left_ankle_pitch_joint": -0.25,     
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.2,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.45,            
            "right_ankle_pitch_joint": -0.25,
            "right_ankle_roll_joint": 0.0,
        },
        "joint_names": [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ],
        "kp": 150.0,      # Increased for better weight support
        "kd": 4.0,        
        "termination_if_roll_greater_than": 30,
        "termination_if_pitch_greater_than": 30,
        "base_init_pos": [0.0, 0.0, 0.74],    
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.45,                  
        "simulate_action_latency": False,     
        "clip_actions": 1.0,                  
    }
    
    obs_cfg = {
        "num_obs": 48,
        "obs_scales": {
            "lin_vel": 2.0, 
            "ang_vel": 0.25, 
            "dof_pos": 1.0, 
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.72,          
        "reward_scales": {
            # Positive rewards (encourage good behavior)
            "alive": 2.0,              # Survival bonus - most important early on
            "tracking_lin_vel": 1.0,   # Reduced - less important until standing

            "forward": 2.0,           
            "no_motion": -0.5,
            "feet_air_time": 1.5,
            "foot_clearance": 0.3,
            "lin_vel_z": -2.0,
            "narrow_stance": -1.5,
            "no_sideways": -2.0,
            
            # Penalties (now properly scaled for radians)
            "base_height": -10.0,      # Reduced from -50
            "orientation": -2.0,       # Reduced - now uses radians internally
            "action_rate": -0.01,      # Smoothing
            "similar_to_default": -0.2, # Reduced - allow some deviation
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        # Start with standing (zero velocity) - learn to balance first
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1_12dof")
    parser.add_argument("-B", "--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=501)
    parser.add_argument("--backend", type=str, choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("-v", "--vis", action="store_true", help="Visualize", default=True)
    args = parser.parse_args()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend)

    log_dir = f"rl-training-genesis/logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, 
        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=args.vis
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    if args.vis:
        print("Training complete. Viewer open. Press Ctrl+C to exit.")
        policy = runner.get_inference_policy(device=gs.device)
        try:
            with torch.inference_mode():
                obs, _ = env.reset()
                while True:
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if dones.any(): obs, _ = env.reset()
        except KeyboardInterrupt:
            print("Exiting...")

if __name__ == "__main__":
    main()

"""
# training
python rl-training-genesis/g1_locomotion_rg_train.py --backend gpu -v
python rl-training-genesis/g1_locomotion_rg_train.py --backend gpu
python rl-training-genesis/g1_locomotion_rg_train.py
"""