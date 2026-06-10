
import math
import os
import numpy as np
import torch
import genesis as gs
from genesis.utils.misc import tensor_to_array

class BallKalmanFilter:


    def __init__(self, dt: float = 0.02, num_envs: int = 1, device: str = "cuda"):
        self.dt = dt
        self.num_envs = num_envs
        self.device = device

        self.F = torch.eye(4, device=device).unsqueeze(0).repeat(num_envs, 1, 1)
        self.F[:, 0, 2] = dt
        self.F[:, 1, 3] = dt

        self.H = torch.zeros(num_envs, 2, 4, device=device)
        self.H[:, 0, 0] = 1.0
        self.H[:, 1, 1] = 1.0

        q = (1.0 * dt) ** 2
        self.Q = torch.zeros(num_envs, 4, 4, device=device)
        self.Q[:, 0, 0] = q * 0.25 * dt**2
        self.Q[:, 1, 1] = q * 0.25 * dt**2
        self.Q[:, 2, 2] = q
        self.Q[:, 3, 3] = q

        r = 0.01**2
        self.R = torch.eye(2, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * r

        self.x = torch.zeros(num_envs, 4, 1, device=device)
        self.P = torch.eye(4, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * 0.1

    def reset(self, env_ids: torch.Tensor):
        self.x[env_ids] = 0.0
        self.P[env_ids] = torch.eye(4, device=self.device) * 0.1

    def predict(self):
        # x = F x
        self.x = torch.bmm(self.F, self.x)
        # P = F P Fᵀ + Q
        FP = torch.bmm(self.F, self.P)
        self.P = torch.bmm(FP, self.F.transpose(1, 2)) + self.Q

    def update(self, z: torch.Tensor):
        z = z.unsqueeze(2)                              # (N, 2, 1)
        y = z - torch.bmm(self.H, self.x)
        HP = torch.bmm(self.H, self.P)
        S = torch.bmm(HP, self.H.transpose(1, 2)) + self.R
        PHt = torch.bmm(self.P, self.H.transpose(1, 2))
        K = torch.bmm(PHt, torch.linalg.inv(S))
        self.x = self.x + torch.bmm(K, y)
        I_KH = torch.eye(4, device=self.device).unsqueeze(0) - torch.bmm(K, self.H)
        self.P = torch.bmm(I_KH, self.P)

    @property
    def position(self) -> torch.Tensor:
        return self.x[:, :2, 0]

    @property
    def velocity(self) -> torch.Tensor:
        return self.x[:, 2:, 0]

    @property
    def uncertainty(self) -> torch.Tensor:
        return torch.stack([self.P[:, 0, 0], self.P[:, 1, 1]], dim=1)


class Go2BalancingBallEnv:


    NUM_OBS        = 56
    NUM_ACT        = 12
    DT             = 0.02
    SIM_DT         = 0.005
    DECIMATION     = 4

    BACK_CENTRE_LOCAL = torch.tensor([0.0, 0.0, 0.16])
    TARGET_HEIGHT     = 0.35

    KP = 20.0
    KD = 0.5

    DEFAULT_DOFS_POS = torch.tensor([
        0.0,  0.8, -1.5,   
        0.0,  0.8, -1.5,   
        0.0,  1.0, -1.5,   
        0.0,  1.0, -1.5,   
    ])

    ACTION_SCALE = 0.25

    RWD_BALL_CENTRE  = 5.0
    RWD_BALL_VEL     = 2.0
    RWD_BASE_HEIGHT  = 1.0
    RWD_BASE_ORI     = 1.0
    RWD_ACTION_RATE  = 0.05
    RWD_JOINT_LIMITS = 0.5

    MAX_EPISODE_LEN  = 1000 
    TERMINATION_BALL_DIST = 0.20

    def __init__(
        self,
        num_envs:    int  = 64,
        env_spacing: float = 3.0,
        show_viewer: bool  = False,
        device:      str   = "cuda",
        ball_urdf:   str   = "ball.urdf",
    ):
        self.num_envs    = num_envs
        self.device      = device
        self.show_viewer = show_viewer
        self.ball_urdf   = ball_urdf

        # Scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.SIM_DT, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(2.0, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
            rigid_options=gs.options.RigidOptions(
                dt=self.SIM_DT,
                gravity=(0.0, 0.0, -9.81),
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )

        # Ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",  # Genesis built-in
                pos=(0.0, 0.0, 0.40),
            ),
        )

        #Ball
        ball_start_pos = (0.0, 0.0, 0.40 + 0.16 + 0.075)   
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(
            radius=0.075,
            pos=ball_start_pos,
            fixed=False,
            ),
        )

        self.scene.build(n_envs=num_envs, env_spacing=(env_spacing, env_spacing))

        self._init_joint_indices()

        # PD gains
        self.robot.set_dofs_kp([self.KP] * self.NUM_ACT, self.dof_idx)
        self.robot.set_dofs_kv([self.KD] * self.NUM_ACT, self.dof_idx)

        # Kalman Filter
        self.kf = BallKalmanFilter(
            dt=self.DT,
            num_envs=num_envs,
            device=device,
        )

        self.obs_buf          = torch.zeros(num_envs, self.NUM_OBS,  device=device)
        self.rew_buf          = torch.zeros(num_envs,                device=device)
        self.reset_buf        = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episode_len_buf  = torch.zeros(num_envs, dtype=torch.int,  device=device)
        self.last_action      = torch.zeros(num_envs, self.NUM_ACT,  device=device)
        self.last_last_action = torch.zeros(num_envs, self.NUM_ACT,  device=device)

        # Gravity vector
        self._gravity_world = torch.tensor([0.0, 0.0, -1.0], device=device)

        # Velocity commands
        self.commands = torch.zeros(num_envs, 3, device=device)

    def _init_joint_indices(self):
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.dof_idx = [
            self.robot.get_joint(name).dof_idx_local for name in joint_names
        ]
        self.default_dofs = self.DEFAULT_DOFS_POS.to(self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        # Reset robot
        self.robot.set_dofs_position(
            position=self.default_dofs.unsqueeze(0).repeat(len(env_ids), 1),
            dofs_idx_local=self.dof_idx,
            zero_velocity=True,
            envs_idx=env_ids,
        )

        # Reset ball
        ball_pos = torch.zeros(len(env_ids), 3, device=self.device)
        ball_pos[:, 2] = 0.40 + 0.16 + 0.075   # world z
        self.ball.set_pos(ball_pos, zero_velocity=True, envs_idx=env_ids)

        # Reset Kalman Filter
        self.kf.reset(env_ids)

        # Reset buffers 
        self.episode_len_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.last_last_action[env_ids] = 0.0

        self.scene.step()
        self._compute_obs(env_ids)

        # Set initial camera when viewer is shown (env 0 centered)
        if self.show_viewer and hasattr(self.scene, "viewer") and self.scene.viewer is not None and self.num_envs > 0:
            self._update_viewer_camera()

    def _update_viewer_camera(self):
        try:
            robot_pos = tensor_to_array(self.robot.get_pos()[0])
            lookat_z = 0.5  # between robot base (~0.4) and ball (~0.64)
            self.scene.viewer.set_camera(
                pos=(robot_pos[0] + 2.0, robot_pos[1], lookat_z + 1.2),
                lookat=(robot_pos[0], robot_pos[1], lookat_z),
            )
        except Exception:
            pass

    def step(self, actions: torch.Tensor):

        self.last_last_action[:] = self.last_action
        self.last_action[:] = actions

        targets = self.default_dofs + actions * self.ACTION_SCALE

        # Physics steps
        for _ in range(self.DECIMATION):
            self.robot.control_dofs_position(
                position=targets,
                dofs_idx_local=self.dof_idx,
            )
            self.scene.step()

        self.kf.predict()
        ball_pos_body = self._ball_pos_body_frame()
        self.kf.update(ball_pos_body[:, :2])

        # Camera tracking: keep robot and ball centered when viewer is shown
        if self.show_viewer and hasattr(self.scene, "viewer") and self.scene.viewer is not None and self.num_envs > 0:
            self._update_viewer_camera()

        # Observations & rewards
        self._compute_obs()
        self._compute_reward()
        self._check_termination()

        self.episode_len_buf += 1

        done_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.reset(done_ids)

        info = {"episode_len": self.episode_len_buf.clone()}
        return self.obs_buf.clone(), self.rew_buf.clone(), self.reset_buf.clone(), info

    def _compute_obs(self, env_ids: torch.Tensor | None = None):
        ids = env_ids if env_ids is not None else slice(None)

        base_lin_vel  = self.robot.get_vel()[ids]
        base_ang_vel  = self.robot.get_ang()[ids]
        base_quat     = self.robot.get_quat()[ids]
        joint_pos     = self.robot.get_dofs_position(self.dof_idx)[ids]
        joint_vel     = self.robot.get_dofs_velocity(self.dof_idx)[ids]
        proj_grav     = self._project_gravity(base_quat)
        kf_pos = self.kf.position
        kf_vel = self.kf.velocity
        kf_unc = self.kf.uncertainty.clamp(0, 1)
        kf_pos3 = torch.cat([kf_pos, torch.zeros(len(kf_pos), 1, device=self.device)], 1)
        kf_vel3 = torch.cat([kf_vel, torch.zeros(len(kf_vel), 1, device=self.device)], 1)

        obs = torch.cat([
            base_lin_vel,
            base_ang_vel,
            proj_grav,
            self.commands[ids],
            joint_pos - self.default_dofs,
            joint_vel,
            self.last_action[ids],
            kf_pos3[:, :3],
            kf_vel3[:, :3],
            kf_unc,
        ], dim=1)

        self.obs_buf[ids] = obs

    def _project_gravity(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        gx = 2*(x*z - w*y)
        gy = 2*(y*z + w*x)
        gz = w*w - x*x - y*y + z*z
        return torch.stack([gx, gy, gz], dim=1) * (-1.0)

    def _ball_pos_body_frame(self) -> torch.Tensor:
        ball_world  = self.ball.get_pos()
        robot_world = self.robot.get_pos()
        robot_quat  = self.robot.get_quat()
        rel_world = ball_world - robot_world
        rel_body  = self._rotate_vec_inv_quat(rel_world, robot_quat)
        return rel_body - self.BACK_CENTRE_LOCAL.to(self.device)

    def _rotate_vec_inv_quat(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        qc = torch.stack([w, -x, -y, -z], dim=1)
        return self._rotate_vec_quat(v, qc)

    def _rotate_vec_quat(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        rx = (w*w + x*x - y*y - z*z)*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz
        ry = 2*(x*y + w*z)*vx + (w*w - x*x + y*y - z*z)*vy + 2*(y*z - w*x)*vz
        rz = 2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (w*w - x*x - y*y + z*z)*vz
        return torch.stack([rx, ry, rz], dim=1)

    def _compute_reward(self):
        # Ball centering
        ball_body = self._ball_pos_body_frame()[:, :2]
        ball_dist = ball_body.norm(dim=1)
        r_centre = torch.exp(-5.0 * ball_dist)

        # Ball velocity
        kf_vel_mag = self.kf.velocity.norm(dim=1)
        r_ball_vel = torch.exp(-2.0 * kf_vel_mag)

        # Base height
        base_h = self.robot.get_pos()[:, 2]
        r_height = torch.exp(-20.0 * (base_h - self.TARGET_HEIGHT).pow(2))

        # Base orientation
        proj_grav = self._project_gravity(self.robot.get_quat())
        r_ori = torch.exp(-5.0 * (proj_grav[:, :2].norm(dim=1)))

        # Action rate
        r_action = -((self.last_action - self.last_last_action).pow(2).sum(dim=1))

        # Weighted sum
        self.rew_buf = (
              self.RWD_BALL_CENTRE * r_centre
            + self.RWD_BALL_VEL   * r_ball_vel
            + self.RWD_BASE_HEIGHT * r_height
            + self.RWD_BASE_ORI   * r_ori
            + self.RWD_ACTION_RATE * r_action
        )

    def _check_termination(self):
        # Ball too far from back
        ball_body_planar = self._ball_pos_body_frame()[:, :2]
        ball_fell = ball_body_planar.norm(dim=1) > self.TERMINATION_BALL_DIST
        proj_grav = self._project_gravity(self.robot.get_quat())
        robot_fell = proj_grav[:, 2] < 0.5
        timed_out = self.episode_len_buf >= self.MAX_EPISODE_LEN
        self.reset_buf[:] = ball_fell | robot_fell | timed_out