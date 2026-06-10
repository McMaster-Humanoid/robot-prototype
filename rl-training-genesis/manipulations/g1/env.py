"""
Genesis-based G1 Manipulation Environment for LeRobot/VLA
Adapted from rl-training-genesis/go2_env.py for manipulation tasks with vision
"""

import math
import os
import numpy as np
import torch
import gymnasium as gym
from typing import Optional, Dict, Any

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat

class G1ManipulationEnv(gym.Env):
    """
    Genesis-based G1 humanoid manipulation environment with vision for VLA/LeRobot.
    
    Features:
    - Unitree G1 humanoid robot with 14 DOF arms
    - Camera observations for VLA training
    - Manipulation tasks (reach, pick, place)
    - Compatible with LeRobot data collection and ACT training
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        num_envs: int = 1,
        task: str = "reach",
        render_mode: Optional[str] = None,
        show_viewer: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        
        # Initialize Genesis (if not already initialized)
        if not hasattr(gs, 'device') or gs.device is None:
            backend = gs.gpu if device == "cuda" else gs.cpu
            gs.init(backend=backend)
        
        self.num_envs = num_envs
        self.task = task
        self.render_mode = render_mode
        self.device = device
        
        # Simulation parameters
        self.dt = 0.02  # 50 Hz
        self.max_episode_length = 250  # 5 seconds at 50Hz
        
        # G1 configuration
        self.num_arm_dof = 14  # 7 per arm
        self.num_waist_dof = 3
        self.num_actions = self.num_arm_dof  # Control arms only
        
        # Create Genesis scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
                gravity=(0, 0, -9.81)
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=30,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.5, -1.5, 1.5),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=50,
                max_FPS=int(1.0 / self.dt),
            ),
            show_viewer=show_viewer,
        )
        
        # Add ground plane
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )
        
        # Add G1 robot (23DOF with arms!)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        urdf_path = os.path.join(project_root, "rl-training-genesis", "models", "g1_23dof.urdf")
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"G1 URDF not found at {urdf_path}")
        
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=(0.0, 0.0, 1.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        
        # Add camera for VLA vision input
        self.camera = self.scene.add_camera(
            res=(480, 640),
            pos=(1.2, 0.0, 1.5),
            lookat=(0.0, 0.0, 1.0),
            fov=60,
            GUI=False,
        )
        
        # Add manipulation target (cube) - positioned for G1 reach
        self.target_cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.05, 0.05, 0.05),
                pos=(0.4, 0.0, 1.2),  # In front of G1 at chest height
                fixed=False,
            )
        )
        
        # Define arm joint names for G1_12DOF BEFORE building scene
        # G1_12DOF: legs(12) + waist(1) + arms(10 = 5 per arm) = 23 total DOF
        # For manipulation, control arms (10 DOF total, 5 per arm)
        self.arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint",
        ]
        
        # Build scene
        self.scene.build(n_envs=num_envs)
        
        # NOW get DOF indices (after scene is built)
        arm_dof_list = []
        for name in self.arm_joint_names:
            joint = self.robot.get_joint(name)
            if joint is not None and joint.dof_start >= 0:
                arm_dof_list.append(joint.dof_start)
        
        self.arm_dof_idx = torch.tensor(
            arm_dof_list,
            dtype=gs.tc_int,
            device=gs.device,
        )
        
        # Set PD control parameters for arms
        self.kp = 100.0
        self.kd = 10.0
        if len(self.arm_dof_idx) > 0:
            self.robot.set_dofs_kp([self.kp] * len(self.arm_dof_idx), self.arm_dof_idx)
            self.robot.set_dofs_kv([self.kd] * len(self.arm_dof_idx), self.arm_dof_idx)
        
        # Initial robot pose
        self.init_base_pos = torch.tensor([0.0, 0.0, 1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
        
        # Define Gymnasium spaces based on actual DOFs
        actual_arm_dofs = len(self.arm_dof_idx)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, 
                shape=(480, 640, 3), 
                dtype=np.uint8
            ),
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(actual_arm_dofs * 2,),  # positions + velocities
                dtype=np.float32
            ),
        })
        
        # Action: arm joint positions
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(actual_arm_dofs,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.episode_step = 0
        
        print(f"G1ManipulationEnv initialized:")
        print(f"  - Task: {self.task}")
        print(f"  - Num envs: {self.num_envs}")
        print(f"  - Arm DOF: {len(self.arm_dof_idx)}")
        print(f"  - Observation space: {self.observation_space}")
        print(f"  - Action space: {self.action_space}")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset robot to initial pose
        init_arm_pos = torch.zeros(len(self.arm_dof_idx), dtype=gs.tc_float, device=gs.device)
        if len(self.arm_dof_idx) > 0:
            self.robot.set_dofs_position(init_arm_pos, self.arm_dof_idx, zero_velocity=True)
        
        # Reset cube position (randomize slightly for G1 workspace)
        if seed is not None:
            torch.manual_seed(seed)
        cube_x = 0.4 + torch.rand(1).item() * 0.1 - 0.05
        cube_y = torch.rand(1).item() * 0.2 - 0.1
        self.target_cube.set_pos((cube_x, cube_y, 1.2))
        
        # Step simulation to settle
        for _ in range(10):
            self.scene.step()
        
        # Reset episode tracking
        self.episode_step = 0
        
        # Get initial observation
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """Execute action and return next observation."""
        # Convert action to torch tensor
        action_tensor = torch.from_numpy(action).to(dtype=gs.tc_float, device=gs.device)
        
        # Scale action from [-1, 1] to joint limits
        # For now, use small range around current position
        if len(self.arm_dof_idx) > 0:
            current_pos = self.robot.get_dofs_position(self.arm_dof_idx)[0]  # First env
            target_pos = current_pos + action_tensor * 0.1  # Small increments
            target_pos = torch.clamp(target_pos, -np.pi, np.pi)
            
            # Send position command
            self.robot.control_dofs_position(
                target_pos.unsqueeze(0),  # Add batch dimension
                self.arm_dof_idx
            )
        
        # Step simulation
        self.scene.step()
        self.episode_step += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        terminated = self.episode_step >= self.max_episode_length
        truncated = False
        
        info = {
            "episode_step": self.episode_step,
            "task": self.task,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Render camera image
        rgb = self.camera.render(rgb=True, depth=False)[0]  # First env, RGB only
        
        # Get robot state (joint positions and velocities)
        if len(self.arm_dof_idx) > 0:
            joint_pos = self.robot.get_dofs_position(self.arm_dof_idx)[0]  # First env
            joint_vel = self.robot.get_dofs_velocity(self.arm_dof_idx)[0]  # First env
            state = torch.cat([joint_pos, joint_vel])
        else:
            state = torch.zeros(0, device=gs.device)
        
        # Convert to numpy
        # Camera render returns numpy array, robot state is torch tensor
        if isinstance(rgb, np.ndarray):
            image = (rgb * 255).astype(np.uint8)
        else:
            image = (rgb.cpu().numpy() * 255).astype(np.uint8)
            
        obs = {
            "image": image,
            "state": state.cpu().numpy().astype(np.float32),
        }
        
        return obs
    
    def _compute_reward(self) -> float:
        """Compute task reward."""
        # Simple reward: negative distance from end-effector to cube
        # TODO: Implement proper task-specific rewards
        reward = 0.0
        return reward
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._get_obs()["image"]
        return None
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'scene'):
            self.scene.viewer.stop() if self.scene.viewer else None


# Convenience function for LeRobot
def make_env(task="reach", render_mode=None, **kwargs):
    """Factory function to create G1 manipulation environment."""
    return G1ManipulationEnv(
        task=task,
        render_mode=render_mode,
        **kwargs
    )


if __name__ == "__main__":
    # Test the environment
    print("Testing G1ManipulationEnv...")
    
    env = G1ManipulationEnv(
        num_envs=1,
        task="reach",
        show_viewer=True,
    )
    
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"State shape: {obs['state'].shape}")
    
    print("\nRunning 100 random steps...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.3f}, terminated={terminated}")
        
        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
    
    print("\nTest complete! Environment working.")
    env.close()
