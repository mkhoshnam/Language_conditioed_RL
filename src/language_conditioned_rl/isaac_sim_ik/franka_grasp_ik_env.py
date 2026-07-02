# Franka grasp/place CURRICULUM task -- TASK-SPACE (differential IK) variant.
#
# WHY THIS VARIANT EXISTS: three joint-space runs failed the same way -- PPO
# cannot discover the top-down wrist orientation through 7 raw joint deltas
# (metrics: xy solved to 2cm, downness stuck at 0.1-0.2). Here the action space
# is [dx, dy, dz, dyaw, grip] in the robot base frame and the IK controller
# HOLDS the gripper pointing straight down at all times. Orientation is no
# longer a learning problem; "move down 2cm" is one action instead of a
# 7-joint coordination puzzle. This mirrors the stock Isaac-Lift-Cube-Franka
# recipe, which trains reliably for exactly this reason.
#
# (original joint-space header follows)
# Franka grasp/place CURRICULUM task (standalone, trained stage-by-stage).
#
# Drawer is always reset already open. This is policy #2 only:
#   policy 1: open drawer
#   policy 2: approach -> grasp -> lift -> place cube into open drawer
#
# Curriculum stages supported by cfg.curriculum_stage:
#   approach : learn to move above/near the cube, gripper open, top-down
#   grasp    : learn align -> descend -> close at the right time
#   lift     : learn real cube lift after grasp
#   place    : learn lift + move cube toward drawer target
#   full     : full pick/place reward
#
# Important: no handle reward and no drawer-opening reward exist here.

from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform, subtract_frame_transforms


@configclass
class FrankaGraspCurriculumEnvCfg(DirectRLEnvCfg):
    # --- env ---
    episode_length_s = 10.0
    decimation = 2
    # task-space actions: [dx, dy, dz, dyaw, grip]
    action_space = 5
    observation_space = 37
    state_space = 0

    # --- task-space control ---
    ee_pos_step = 0.015      # meters per control step per unit action
    ee_yaw_step = 0.08       # radians per control step per unit action
    # workspace box in ROBOT BASE frame (base sits at world (1,0,0)):
    # cube is at base-frame (-0.45, 0.48, 0.33), drawer target around (-0.55, 0, 0.7)
    # BUGFIX: the robot base spawns with rot=(0,0,0,1) = 180deg yaw, so the old
    # base-frame box was on the WRONG SIDE -- the target could never reach the
    # cube. The box is now in the env-local WORLD frame (relative to env origin):
    # cube at (0.55, 0.48, 0.33), drawer place target around (0.3-0.5, 0, 0.6-0.75).
    # z floor: fingertips are ~10cm below the panda_hand frame when pointing
    # down. hand z >= 0.38 keeps fingertips >= ~0.28 -- just below cube-grasp
    # height (cube center 0.33) and NEVER into the floor/pedestal body. This
    # deletes the whole "dive into the table" region from exploration.
    ws_low = (0.15, -0.40, 0.38)
    ws_high = (0.90, 0.85, 0.90)

    # curriculum stage: approach | grasp | lift | place | full
    curriculum_stage = "full"

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True, clone_in_fabric=True
    )

    # --- robot ---
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0, stiffness=80.0, damping=4.0
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0, stiffness=80.0, damping=4.0
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2
            ),
        },
    )

    # --- cabinet ---
    cabinet = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            rot=(0.1, 0.0, 0.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0, stiffness=0.0, damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0, stiffness=10.0, damping=2.5,
            ),
        },
    )

    # --- table/pedestal ---
    table_height = 0.30
    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.20, table_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.22)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.48, table_height / 2)),
    )

    # --- cube: stock Isaac Lab "Dex Cube" USD (the one that grasps in teleop).
    # It carries its own physics material / friction, so no override is needed. ---
    cube_rest_z = 0.33
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),  # ~0.064 m, same size the stock lift task grasps
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        # spawn a touch above the pedestal top (0.30) so it drops and settles onto it
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.48, 0.35)),
    )

    # --- ground ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # --- control ---
    action_scale = 5.0
    dof_velocity_scale = 0.1

    # --- drawer context ---
    drawer_open_amount = 0.35

    # --- reward shaping ---
    action_penalty_scale = 0.02
    vel_penalty_scale = 0.003

    # smooth approach/grasp/lift/place terms
    reach_scale = 1.0
    align_scale = 2.5
    descend_scale = 2.0
    orient_scale = 1.0
    keep_open_scale = 0.3
    grasp_scale = 4.0
    grasp_hold_scale = 5.0
    premature_close_scale = 2.5
    lift_scale = 18.0
    lifted_height = 0.05
    place_scale = 4.0
    place_bonus = 8.0
    place_success_dist = 0.06

    # --- grip geometry (cube = DexCube 0.08m * 0.8 scale = 0.064m wide) ---
    # A REAL grasp blocks each finger at ~cube_half_width = 0.032, so the old
    # finger_closing=(0.04-q)/0.04 saturates at 0.20 while holding the cube.
    # These bounds define "fingers are squeezing something cube-sized".
    cube_width = 0.064
    grip_width_min = 0.045   # 2*finger_pos below this = closed on air / crushing
    grip_width_max = 0.075   # above this = fingers not really closed yet
    grip_bonus_scale = 3.0   # per-step bonus for a verified grip on the cube
    drop_penalty = 5.0       # one-time penalty when cube leaves the pedestal

    # --- reverse-curriculum / pre-grasp initialization ---
    # Fraction of resets that start with the cube already inside the closed
    # gripper (teleported to the grasp frame, fingers closing). The trainer
    # sets this per stage: 0 for approach/grasp, >0 for lift/place/full.
    pregrasp_ratio = 0.0


class FrankaGraspCurriculumEnv(DirectRLEnv):
    cfg: FrankaGraspCurriculumEnvCfg

    def __init__(self, cfg: FrankaGraspCurriculumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos, xformable, device):
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            px, py, pz = world_pos[0] - env_pos[0], world_pos[1] - env_pos[1], world_pos[2] - env_pos[2]
            qx, qy, qz = world_quat.imaginary[0], world_quat.imaginary[1], world_quat.imaginary[2]
            qw = world_quat.real
            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self._finger_dof_idx, _ = self._robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        grasp_rot, grasp_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        grasp_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = grasp_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = grasp_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))

        # ---- differential IK: absolute pose commands, orientation locked down ----
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)
        self._arm_joint_ids = list(range(7))
        self._ee_body_idx = self._robot.find_bodies("panda_hand")[0][0]
        # for fixed-base articulations the jacobian body index is body_index - 1
        self._ee_jacobi_idx = self._ee_body_idx - 1
        # hand z-axis points out of the palm; 180deg about WORLD x maps it to
        # world -z (down). All IK quantities below are in the WORLD frame, which
        # matches the frame PhysX expresses jacobians in -- do NOT convert to the
        # base frame (base has a 180deg yaw; mixing frames flips x/y).
        self._down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.ee_target_pos_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_yaw = torch.zeros(self.num_envs, device=self.device)
        self._ee_needs_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._ws_low = torch.tensor(self.cfg.ws_low, device=self.device)
        self._ws_high = torch.tensor(self.cfg.ws_high, device=self.device)

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]
        self.drawer_joint_idx = self._cabinet.find_joints("drawer_top_joint")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.place_target_pos = torch.zeros((self.num_envs, 3), device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cabinet = Articulation(self.cfg.cabinet)
        self._cube = RigidObject(self.cfg.cube)
        self._table = RigidObject(self.cfg.table)
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["cabinet"] = self._cabinet
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.rigid_objects["table"] = self._table

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------- actions
    def _sanitize_buffers(self):
        """De-poison persistent buffers that were reassigned inside rsl_rl's
        torch.inference_mode() rollout collection. In-place writes on inference
        tensors crash outside inference mode (e.g. resets during evaluation).
        clone() outside inference mode converts them back to normal tensors."""
        for name in ("ee_target_pos_b", "ee_yaw", "_ee_needs_init", "gripper_closed_latch"):
            t = getattr(self, name, None)
            if t is not None and t.is_inference():
                setattr(self, name, t.clone())

    def _pre_physics_step(self, actions: torch.Tensor):
        self._sanitize_buffers()
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # ---- current end-effector pose in WORLD frame (matches jacobian frame) ----
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_idx]
        # env-local coordinates for target bookkeeping / workspace clamping
        ee_pos_local = ee_pos_w - self.scene.env_origins

        # freshly reset envs: start the target at the measured ee position
        if self._ee_needs_init.any():
            # During eval, tensors may become inference tensors. Clone the
            # destination buffers before indexed/in-place updates.
            self.ee_target_pos_b = self.ee_target_pos_b.clone()
            self.ee_yaw = self.ee_yaw.clone()
            self._ee_needs_init = self._ee_needs_init.clone()

            ids = self._ee_needs_init.clone()
            self.ee_target_pos_b[ids] = ee_pos_local[ids].clone()
            self.ee_yaw[ids] = 0.0
            self._ee_needs_init = torch.zeros_like(self._ee_needs_init)

        # ---- integrate position/yaw deltas, clamp to env-local workspace ----
        self.ee_target_pos_b = self.ee_target_pos_b + self.actions[:, 0:3] * self.cfg.ee_pos_step
        self.ee_target_pos_b = torch.clamp(self.ee_target_pos_b, self._ws_low, self._ws_high)
        self.ee_yaw = torch.clamp(
            self.ee_yaw + self.actions[:, 3] * self.cfg.ee_yaw_step, -1.57, 1.57
        )

        # orientation = fixed top-down in WORLD frame, composed with commanded yaw
        yaw_quat = quat_from_angle_axis(self.ee_yaw, self._z_axis)
        target_quat_w = quat_mul(yaw_quat, self._down_quat)
        target_pos_w = self.ee_target_pos_b + self.scene.env_origins

        # ---- differential IK -> arm joint targets (all WORLD frame) ----
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._arm_joint_ids
        ]
        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        ik_cmd = torch.cat([target_pos_w, target_quat_w], dim=-1)
        self._ik.set_command(ik_cmd)
        arm_targets = self._ik.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)
        self.robot_dof_targets[:, self._arm_joint_ids] = torch.clamp(
            arm_targets,
            self.robot_dof_lower_limits[self._arm_joint_ids],
            self.robot_dof_upper_limits[self._arm_joint_ids],
        )

        # ---- gripper: binary with hysteresis (same as joint-space variant) ----
        if not hasattr(self, "gripper_closed_latch"):
            self.gripper_closed_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        grip_cmd = self.actions[:, 4]
        self.gripper_closed_latch = torch.where(
            grip_cmd < -0.3, torch.ones_like(self.gripper_closed_latch),
            torch.where(grip_cmd > 0.3, torch.zeros_like(self.gripper_closed_latch), self.gripper_closed_latch),
        )
        finger_target = torch.where(
            self.gripper_closed_latch,
            torch.zeros(self.num_envs, device=self.device),
            torch.full((self.num_envs,), 0.04, device=self.device),
        )
        self.robot_dof_targets[:, self._finger_dof_idx] = finger_target.unsqueeze(-1)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ---------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # Cube knocked off the pedestal -> terminate (and penalize in reward).
        # Otherwise a knocked-off cube leaves the env collecting garbage reward
        # on the floor for the rest of a 600-step episode.
        self.cube_dropped = self.cube_pos_w[:, 2] < (self.cfg.cube_rest_z - 0.10)
        return self.cube_dropped, truncated

    # -------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        # IK-variant reward: MINIMAL and waypoint-based.
        # Under task-space IK the orientation is guaranteed by the controller,
        # so all align/orient/descend/down terms are free reward -- the policy
        # farmed them while standing still (mean reward ~= sum of free terms).
        # Each stage now has ONE dominant signal:
        #   approach: distance to pre-grasp waypoint (cube + 8cm)
        #   grasp:    distance to cube + close-when-straddling + verified grip
        #   lift:     grip + cube height
        #   place:    grip + height + cube-to-target distance
        stage = str(self.cfg.curriculum_stage)

        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions ** 2, dim=-1)
        vel_penalty = self.cfg.vel_penalty_scale * torch.sum(
            self._robot.data.joint_vel[:, :7] ** 2, dim=-1
        )
        penalties = action_penalty + vel_penalty

        grasp_pos = self.robot_grasp_pos
        cube = self.cube_pos_w

        # pre-grasp waypoint: 8cm above the cube center
        waypoint = cube.clone()
        waypoint[:, 2] += 0.08
        d_wp = torch.norm(grasp_pos - waypoint, dim=-1)
        d_grasp = torch.norm(grasp_pos - cube, dim=-1)
        xy_dist = torch.norm(grasp_pos[:, :2] - cube[:, :2], dim=-1)
        z_above = grasp_pos[:, 2] - cube[:, 2]

        # two-scale tracking: coarse term has gradient even at 80cm, sharp term
        # dominates the final 10cm. A single sharp tanh is numerically dead far
        # away -- that is what stalled the previous run.
        def track(d):
            return 3.0 * (1.0 - torch.tanh(d / 0.12)) + 1.0 * (1.0 - torch.tanh(d / 0.50))

        # fingers / verified grip (same detector as before)
        finger_pos = self._robot.data.joint_pos[:, self._finger_dof_idx].mean(dim=-1)
        finger_open = torch.clamp(finger_pos / 0.04, 0.0, 1.0)
        grip_width = 2.0 * finger_pos
        cube_between = (xy_dist < 0.045) & (z_above > -0.045) & (z_above < 0.045)
        squeezing = (grip_width > self.cfg.grip_width_min) & (grip_width < self.cfg.grip_width_max)
        closed_latch = getattr(
            self, "gripper_closed_latch",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        closed_f = closed_latch.float()
        grip_on_cube = (cube_between & squeezing & closed_latch).float()

        grip_cmd = self.actions[:, 4]
        close_cmd = torch.clamp(-grip_cmd, 0.0, 1.0)

        # "near" = grasp frame straddling the cube; the only place closing pays
        near = 1.0 - torch.tanh(d_grasp / 0.05)

        cube_lift = torch.clamp(cube[:, 2] - self.cfg.cube_rest_z, min=0.0)
        lift = self.cfg.lift_scale * torch.clamp(cube_lift, max=0.20)
        lifted = (cube_lift > self.cfg.lifted_height).float()

        d_place = torch.norm(cube - self.place_target_pos, dim=-1)
        place_reward = (1.0 - torch.tanh(d_place / 0.50)) * lifted
        placed_bonus = (d_place < self.cfg.place_success_dist).float() * lifted

        dropped = getattr(
            self, "cube_dropped",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        ).float()
        drop_pen = self.cfg.drop_penalty * dropped

        if stage == "approach":
            reward = (
                track(d_wp)                 # THE signal: go to cube + 8cm
                + 0.3 * finger_open         # small: stay open
                - 0.5 * close_cmd           # small: don't close during approach
                - drop_pen - penalties
            )
        elif stage == "grasp":
            reward = (
                track(d_grasp)              # descend onto the cube
                + 4.0 * near * closed_f     # close exactly when straddling it
                - 1.0 * (1.0 - near) * closed_f   # not before
                + self.cfg.grip_bonus_scale * grip_on_cube
                + 20.0 * torch.clamp(cube_lift, max=0.10) * grip_on_cube
                - drop_pen - penalties
            )
        elif stage == "lift":
            reward = (
                track(d_grasp) * (1.0 - grip_on_cube)   # tracking done once gripped
                + 4.0 * near * closed_f
                - 1.0 * (1.0 - near) * closed_f
                + self.cfg.grip_bonus_scale * grip_on_cube
                + lift
                - drop_pen - penalties
            )
        elif stage in ("place", "full"):
            reward = (
                track(d_grasp) * (1.0 - grip_on_cube)
                + 2.0 * near * closed_f
                + self.cfg.grip_bonus_scale * grip_on_cube
                + lift
                + self.cfg.place_scale * place_reward
                + self.cfg.place_bonus * placed_bonus
                - drop_pen - penalties
            )
        else:
            raise ValueError(f"Unknown curriculum_stage={stage!r}. Use approach/grasp/lift/place/full.")

        return reward

    # --------------------------------------------------------- validation helpers
    def get_stage_success(self, stage: str | None = None) -> torch.Tensor:
        """Return per-env boolean success for curriculum validation.

        This is intentionally separate from the reward. The trainer uses this
        during validation to decide whether to advance to the next stage.
        """
        self._compute_intermediate_values()
        stage = str(stage or self.cfg.curriculum_stage)

        grasp_pos = self.robot_grasp_pos
        cube = self.cube_pos_w
        xy_dist = torch.norm(grasp_pos[:, :2] - cube[:, :2], dim=-1)
        z_above = grasp_pos[:, 2] - cube[:, 2]

        approach = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)
        downness = torch.clamp(-approach[:, 2], 0.0, 1.0)

        finger_pos = self._robot.data.joint_pos[:, self._finger_dof_idx].mean(dim=-1)
        finger_open = torch.clamp(finger_pos / 0.04, 0.0, 1.0)
        grip_width = 2.0 * finger_pos
        squeezing = (grip_width > self.cfg.grip_width_min) & (grip_width < self.cfg.grip_width_max)
        cube_between = (xy_dist < 0.045) & (z_above > -0.045) & (z_above < 0.045)

        cube_lift = torch.clamp(cube[:, 2] - self.cfg.cube_rest_z, min=0.0)
        lifted = cube_lift > self.cfg.lifted_height
        d_place = torch.norm(cube - self.place_target_pos, dim=-1)

        # Stage-specific success definitions. Keep them a little looser than the
        # final reward optimum so the curriculum can advance instead of getting
        # stuck on a tiny threshold.
        # Approach success = arm reaches a reasonable pre-grasp pose.
        # Do NOT require gripper-open here; gripper timing belongs to grasp stage.
        approach_ok = (
            (xy_dist < 0.08)
            & (z_above > -0.04)
            & (z_above < 0.14)
            & (downness > 0.50)
        )
        # OLD BUG: required finger_closing > 0.55 == grip width < 3.6cm, which is
        # physically impossible while holding the 6.4cm cube. A real grasp gives
        # finger_closing = 0.20. Success was unsatisfiable by construction.
        # NEW: grasp = fingers squeezing something cube-sized, cube between them,
        # and the cube has actually risen >= 1cm off its rest height (ground truth
        # that the grip is real -- an air-close cannot lift the cube).
        grasp_ok = cube_between & squeezing & (cube_lift > 0.01)
        lift_ok = lifted
        place_ok = lifted & (d_place < self.cfg.place_success_dist)

        if stage == "approach":
            return approach_ok
        if stage == "grasp":
            return grasp_ok
        if stage == "lift":
            return lift_ok
        if stage in ("place", "full"):
            return place_ok
        raise ValueError(f"Unknown stage={stage!r}")

    def get_stage_metrics(self) -> dict[str, float]:
        """Return simple scalar diagnostics for printing during validation."""
        self._compute_intermediate_values()
        grasp_pos = self.robot_grasp_pos
        cube = self.cube_pos_w
        xy_dist = torch.norm(grasp_pos[:, :2] - cube[:, :2], dim=-1)
        z_above = grasp_pos[:, 2] - cube[:, 2]
        approach = tf_vector(self.robot_grasp_rot, self.gripper_forward_axis)
        downness = torch.clamp(-approach[:, 2], 0.0, 1.0)
        finger_pos = self._robot.data.joint_pos[:, self._finger_dof_idx].mean(dim=-1)
        grip_width = 2.0 * finger_pos
        squeezing = (grip_width > self.cfg.grip_width_min) & (grip_width < self.cfg.grip_width_max)
        cube_between = (xy_dist < 0.045) & (z_above > -0.045) & (z_above < 0.045)
        cube_lift = torch.clamp(cube[:, 2] - self.cfg.cube_rest_z, min=0.0)
        d_place = torch.norm(cube - self.place_target_pos, dim=-1)
        dropped = self.cube_pos_w[:, 2] < (self.cfg.cube_rest_z - 0.10)
        return {
            "xy_dist_cm": float((xy_dist.mean() * 100.0).item()),
            "z_above_cm": float((z_above.mean() * 100.0).item()),
            "downness": float(downness.mean().item()),
            "grip_width_cm": float((grip_width.mean() * 100.0).item()),
            "grip_on_cube": float((cube_between & squeezing).float().mean().item()),
            "cube_lift_cm": float((cube_lift.mean() * 100.0).item()),
            "d_place_cm": float((d_place.mean() * 100.0).item()),
            "dropped": float(dropped.float().mean().item()),
        }

    # ---------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self._sanitize_buffers()

        if not hasattr(self, "gripper_closed_latch"):
            self.gripper_closed_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gripper_closed_latch[env_ids] = False
        if hasattr(self, "_ee_needs_init"):
            self._ee_needs_init[env_ids] = True

        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_dof_targets[env_ids] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        cab_pos = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        cab_vel = torch.zeros_like(cab_pos)
        cab_pos[:, self.drawer_joint_idx] = self.cfg.drawer_open_amount
        self._cabinet.write_joint_state_to_sim(cab_pos, cab_vel, env_ids=env_ids)

        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, 0:3] += self.scene.env_origins[env_ids]
        cube_state[:, 0:2] += sample_uniform(-0.04, 0.04, (len(env_ids), 2), self.device)
        self._cube.write_root_pose_to_sim(cube_state[:, 0:7], env_ids=env_ids)
        self._cube.write_root_velocity_to_sim(torch.zeros_like(cube_state[:, 7:]), env_ids=env_ids)

        self._compute_intermediate_values(env_ids)

        # ---- reverse curriculum: start some episodes ALREADY GRASPING ----
        # Teleport the cube into the grasp frame with fingers closing on it.
        # This lets lift/place learn from a solved-grasp state, decoupling the
        # hard exploration problem (grasp) from the downstream skills.
        ratio = float(getattr(self.cfg, "pregrasp_ratio", 0.0))
        if ratio > 0.0 and len(env_ids) > 0:
            mask = torch.rand(len(env_ids), device=self.device) < ratio
            if mask.any():
                pg_ids = env_ids[mask]
                # fingers slightly wider than cube half-width, target = closed
                half = self.cfg.cube_width / 2.0 + 0.002
                jp = self._robot.data.joint_pos[pg_ids].clone()
                jp[:, self._finger_dof_idx] = half
                jv = torch.zeros_like(jp)
                self._robot.write_joint_state_to_sim(jp, jv, env_ids=pg_ids)
                tgt = self.robot_dof_targets[pg_ids].clone()
                tgt[:, self._finger_dof_idx] = 0.0  # squeeze
                self.robot_dof_targets[pg_ids] = tgt
                self._robot.set_joint_position_target(tgt, env_ids=pg_ids)
                if hasattr(self, "gripper_closed_latch"):
                    self.gripper_closed_latch[pg_ids] = True
                # place cube at the grasp frame (recompute after finger write)
                self._compute_intermediate_values(pg_ids)
                pg_state = self._cube.data.default_root_state[pg_ids].clone()
                pg_state[:, 0:3] = self.robot_grasp_pos[pg_ids]
                pg_state[:, 7:] = 0.0
                self._cube.write_root_pose_to_sim(pg_state[:, 0:7], env_ids=pg_ids)
                self._cube.write_root_velocity_to_sim(pg_state[:, 7:], env_ids=pg_ids)

    # ----------------------------------------------------------- observations
    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        )
        to_handle = self.drawer_grasp_pos - self.robot_grasp_pos
        cube_to_grasp = self.cube_pos_w - self.robot_grasp_pos
        cube_to_place = self.place_target_pos - self.cube_pos_w
        cube_pos_b = self.cube_pos_w - self.scene.env_origins

        stage_to_idx = {"approach": 0, "grasp": 1, "lift": 2, "place": 3, "full": 4}
        stage_onehot = torch.zeros((self.num_envs, 5), device=self.device)
        stage_onehot[:, stage_to_idx.get(str(self.cfg.curriculum_stage), 4)] = 1.0

        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                to_handle,
                self._cabinet.data.joint_pos[:, self.drawer_joint_idx].unsqueeze(-1),
                self._cabinet.data.joint_vel[:, self.drawer_joint_idx].unsqueeze(-1),
                cube_pos_b,
                cube_to_grasp,
                cube_to_place,
                stage_onehot,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------- auxiliary values
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]

        robot_grasp_rot, robot_grasp_pos = tf_combine(
            hand_rot, hand_pos, self.robot_local_grasp_rot[env_ids], self.robot_local_grasp_pos[env_ids]
        )
        drawer_grasp_rot, drawer_grasp_pos = tf_combine(
            drawer_rot, drawer_pos, self.drawer_local_grasp_rot[env_ids], self.drawer_local_grasp_pos[env_ids]
        )
        self.robot_grasp_rot[env_ids] = robot_grasp_rot
        self.robot_grasp_pos[env_ids] = robot_grasp_pos
        self.drawer_grasp_rot[env_ids] = drawer_grasp_rot
        self.drawer_grasp_pos[env_ids] = drawer_grasp_pos

        inward_w = tf_vector(drawer_grasp_rot, self.drawer_inward_axis[env_ids])
        up_w = tf_vector(drawer_grasp_rot, self.drawer_up_axis[env_ids])
        self.place_target_pos[env_ids] = drawer_grasp_pos + 0.15 * inward_w + 0.04 * up_w

        if not hasattr(self, "cube_pos_w"):
            self.cube_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_pos_w[env_ids] = self._cube.data.root_pos_w[env_ids]
