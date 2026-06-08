import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PROJECT_ROOT / "third_party" / "calvin_franka_scene" / "calvin_scene.xml"

TABLE_TOP_Z = 0.230
BLOCK_HALF = 0.025
BLOCK_Z = TABLE_TOP_Z + BLOCK_HALF
WORKSPACE_X = (0.33, 0.67)
WORKSPACE_Y = (-0.18, 0.18)
BLOCK_X_RANGE = (0.36, 0.55)
BLOCK_Y_RANGE = (-0.13, 0.13)
TARGET_X_RANGE = (0.34, 0.67)
TARGET_Y_RANGE = (-0.18, 0.18)

BLOCK_NAMES = ("red_block", "blue_block", "green_block")
TARGET_NAMES = ("yellow_plate", "purple_plate", "cyan_bowl", "orange_plate")
TASKS = tuple(
    (
        block,
        target,
        f"put the {block.replace('_', ' ')} in the {target.replace('_', ' ')}",
    )
    for block in BLOCK_NAMES
    for target in TARGET_NAMES
)
CAMERAS = ("fixed_scene", "wrist_camera")

SUCCESS_RADIUS = 0.055
SUCCESS_RADIUS_START = 0.075
SUCCESS_RADIUS_MIN = 0.060
SUCCESS_HOLD_STEPS = 2
TRANSPORT_HOLD_STEPS = 5
APPROACH_OFFSET = np.array([-0.055, 0.0, 0.090], dtype=np.float64)
APPROACH_RADIUS = 0.060
REACH_RADIUS = 0.110
REACH_HOLD_STEPS = 3
GRASP_HOLD_STEPS = 4
LIFT_HOLD_STEPS = 5
LIFT_HEIGHT = 0.055
SECURE_GRASP_HEIGHT = 0.012
LIFT_CURRICULUM_START = 0.020
TRANSPORT_LIFT_HEIGHT = 0.065
EXCESS_LIFT_HEIGHT = 0.105
CARRY_HIGH_HEIGHT = 0.120
CARRY_CRITICAL_HEIGHT = 0.150
MAX_TRANSPORT_SUCCESS_HEIGHT = 0.120
PLACE_LOWER_RADIUS = 0.090
RELEASE_RADIUS = 0.075
GATE_RELEASE_RADIUS = 0.110
RELEASE_UNLOCK_HEIGHT = 0.105
TRANSPORT_RADIUS = 0.065
PLACE_MIN_CARRY_HEIGHT = 0.045
PLACE_CARRY_HEIGHT_BAND = 0.035
PLACE_HOLD_RADIUS = 0.135
PLACE_LOST_RADIUS = 0.170
GRIPPER_HOLD_CLOSED = 0.35
GRIPPER_RELEASE_OPEN = 0.75
SETTLE_HEIGHT_RADIUS = 0.035
SETTLE_SPEED = 0.090
MAX_STEPS = 280

EE_POS_DELTA = 0.032
EE_ROT_DELTA = 0.120
CARRY_POS_SCALE = 0.85
CARRY_ROT_SCALE = 0.30
JOINT_DELTA_LIMIT = 0.080
IK_DAMPING = 0.080
ROTATION_IK_WEIGHT = 0.30
CTRL_LAG_LIMIT = 0.20
CONTROL_SUBSTEPS = 15
GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSED_CTRL = 0.0
GRIPPER_CTRL_DELTA = 34.0
FINGER_OPEN_Q = 0.040

STAGE_REACH = 0
STAGE_GRASP = 1
STAGE_LIFT = 2
STAGE_TRANSPORT = 3
STAGE_PLACE = 4
STAGE_NAMES = ("reach", "grasp", "lift", "transport", "place")


class RealFrankaPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, render_mode=None, fixed_task_index=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self._enable_robot_gravity_compensation()
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.fixed_task_index = fixed_task_index

        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        self._arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self._finger_joint_names = ["finger_joint1", "finger_joint2"]
        self._act_joint_names = self._arm_joint_names + self._finger_joint_names
        self._act_qpos_addr = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in self._act_joint_names
            ],
            dtype=np.int32,
        )
        self._act_qvel_addr = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in self._act_joint_names
            ],
            dtype=np.int32,
        )
        self._jnt_range = self.model.jnt_range[:9].copy()
        self._arm_act_ctrlrange = self.model.actuator_ctrlrange[:7].copy()

        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self._block_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in BLOCK_NAMES
        }
        self._block_joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            for name in BLOCK_NAMES
        }
        self._block_qpos_addr = {
            name: self.model.jnt_qposadr[jid]
            for name, jid in self._block_joint_ids.items()
        }
        self._block_qvel_addr = {
            name: self.model.jnt_dofadr[jid]
            for name, jid in self._block_joint_ids.items()
        }
        self._target_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in TARGET_NAMES
        }
        self._finger_body_ids = {
            "left_finger": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
            ),
            "right_finger": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
            ),
        }

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(101,), dtype=np.float32
        )

        self._arm_ctrl_target = np.zeros(7, dtype=np.float64)
        self._gripper_ctrl_target = GRIPPER_OPEN_CTRL
        self.selected_block = BLOCK_NAMES[0]
        self.selected_target = TARGET_NAMES[0]
        self.language_goal = TASKS[0][2]
        self._prev_reach_dist = 0.0
        self._prev_approach_dist = 0.0
        self._best_approach_dist = np.inf
        self._prev_place_dist = 0.0
        self._prev_lift_height = 0.0
        self._prev_ee_z = 0.0
        self._prev_settle_score = 0.0
        self._step_count = 0
        self._success_hold = 0
        self._ever_grasped = False
        self._ever_lifted = False
        self._max_lift_height = 0.0
        self.curriculum_dist = 0.10
        self.curriculum_lift_height = LIFT_CURRICULUM_START
        self.success_radius = SUCCESS_RADIUS_START
        self.task_stage = STAGE_REACH

        if render_mode in ("human", "rgb_array"):
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        else:
            self.renderer = None

    def _enable_robot_gravity_compensation(self):
        for name in (
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
            "left_finger",
            "right_finger",
        ):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self.model.body_gravcomp[body_id] = 1.0

    def _task_index(self):
        if self.fixed_task_index is not None:
            return int(self.fixed_task_index) % len(TASKS)
        return int(self.np_random.integers(0, len(TASKS)))

    def _set_task(self):
        block, target, goal = TASKS[self._task_index()]
        self.selected_block = block
        self.selected_target = target
        self.language_goal = goal

    def _set_block_pose(self, name, xy):
        qpos_addr = self._block_qpos_addr[name]
        qvel_addr = self._block_qvel_addr[name]
        self.data.qpos[qpos_addr : qpos_addr + 7] = np.array(
            [xy[0], xy[1], BLOCK_Z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def _set_target_pose(self, name, xy):
        body_id = self._target_body_ids[name]
        z = TABLE_TOP_Z + 0.004
        if "bowl" in name:
            z = TABLE_TOP_Z + 0.008
        self.model.body_pos[body_id] = np.array([xy[0], xy[1], z], dtype=np.float64)

    def _sample_xy(self, used, x_range, y_range, min_sep):
        for _ in range(500):
            xy = np.array(
                [
                    self.np_random.uniform(*x_range),
                    self.np_random.uniform(*y_range),
                ],
                dtype=np.float64,
            )
            if all(np.linalg.norm(xy - other) >= min_sep for other in used):
                return xy
        return np.array([0.50, 0.0], dtype=np.float64)

    def _sample_target_xy(self, block_xy, used):
        min_radius = min(0.085, self.curriculum_dist * 0.70)
        max_radius = max(min_radius + 0.010, self.curriculum_dist)
        for _ in range(500):
            direction = self.np_random.normal(size=2)
            direction /= np.linalg.norm(direction) + 1e-9
            radius = self.np_random.uniform(min_radius, max_radius)
            xy = block_xy + direction * radius
            if (
                TARGET_X_RANGE[0] <= xy[0] <= TARGET_X_RANGE[1]
                and TARGET_Y_RANGE[0] <= xy[1] <= TARGET_Y_RANGE[1]
                and all(np.linalg.norm(xy - other) >= 0.070 for other in used)
            ):
                return xy
        return np.clip(
            block_xy + np.array([min(self.curriculum_dist, 0.16), 0.0]),
            [TARGET_X_RANGE[0], TARGET_Y_RANGE[0]],
            [TARGET_X_RANGE[1], TARGET_Y_RANGE[1]],
        )

    def _randomize_scene(self):
        used = []
        block_xy = {}
        for name in BLOCK_NAMES:
            xy = self._sample_xy(used, BLOCK_X_RANGE, BLOCK_Y_RANGE, min_sep=0.080)
            used.append(xy)
            block_xy[name] = xy
            self._set_block_pose(name, xy)

        selected_xy = block_xy[self.selected_block]
        target_xy = self._sample_target_xy(selected_xy, used)
        used.append(target_xy)
        self._set_target_pose(self.selected_target, target_xy)

        for name in TARGET_NAMES:
            if name == self.selected_target:
                continue
            xy = self._sample_xy(used, TARGET_X_RANGE, TARGET_Y_RANGE, min_sep=0.095)
            used.append(xy)
            self._set_target_pose(name, xy)

    def _block_pos(self, name=None):
        name = name or self.selected_block
        return self.data.xpos[self._block_body_ids[name]].copy()

    def _block_vel(self, name=None):
        name = name or self.selected_block
        return self.data.cvel[self._block_body_ids[name], 3:6].copy()

    def _target_pos(self, name=None):
        name = name or self.selected_target
        return self.data.xpos[self._target_body_ids[name]].copy()

    def _approach_pos(self, block=None):
        if block is None:
            block = self._block_pos()
        return block + APPROACH_OFFSET

    def _finger_q(self):
        return self.data.qpos[self._act_qpos_addr[7:9]].copy()

    def _close_fraction(self):
        open_frac = float(np.clip(np.mean(self._finger_q()) / FINGER_OPEN_Q, 0.0, 1.0))
        return 1.0 - open_frac

    def _gripper_q_target(self):
        return (self._gripper_ctrl_target / GRIPPER_OPEN_CTRL) * FINGER_OPEN_Q

    def _active_lift_goal(self):
        if self.task_stage == STAGE_LIFT:
            return float(
                np.clip(self.curriculum_lift_height, LIFT_CURRICULUM_START, LIFT_HEIGHT)
            )
        return LIFT_HEIGHT

    def _ee_down_metrics(self):
        xmat = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        local_z = xmat[:, 2]
        z_down = float(np.clip(np.dot(local_z, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(np.clip(z_down, -1.0, 1.0))))
        return z_down, tilt_deg

    def _ee_jacobian(self):
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        arm_dofs = self._act_qvel_addr[:7]
        return np.vstack((jacp[:, arm_dofs], jacr[:, arm_dofs]))

    def _ik_delta(self, pos_delta, rot_delta):
        twist = np.concatenate([pos_delta, ROTATION_IK_WEIGHT * rot_delta]).astype(
            np.float64
        )
        if not np.all(np.isfinite(twist)):
            return np.zeros(7, dtype=np.float64)

        jac = self._ee_jacobian()
        lhs = jac @ jac.T + (IK_DAMPING * IK_DAMPING) * np.eye(6)
        try:
            dq = jac.T @ np.linalg.solve(lhs, twist)
        except np.linalg.LinAlgError:
            dq = np.zeros(7, dtype=np.float64)
        return np.clip(dq, -JOINT_DELTA_LIMIT, JOINT_DELTA_LIMIT)

    def _finger_block_contacts(self):
        block_body = self._block_body_ids[self.selected_block]
        left_body = self._finger_body_ids["left_finger"]
        right_body = self._finger_body_ids["right_finger"]
        left = False
        right = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            bodies = {body1, body2}
            if block_body not in bodies:
                continue
            if left_body in bodies:
                left = True
            if right_body in bodies:
                right = True
        return left, right

    def _get_obs(self):
        q = self.data.qpos[self._act_qpos_addr].copy()
        dq = self.data.qvel[self._act_qvel_addr].copy()
        ctrl_q = np.concatenate(
            [self._arm_ctrl_target, np.full(2, self._gripper_q_target())]
        )
        ctrl_err = ctrl_q - q
        ee = self.data.site_xpos[self._ee_site_id].copy()
        ee_xmat = self.data.site_xmat[self._ee_site_id].copy()
        block = self._block_pos()
        block_vel = self._block_vel()
        target = self._target_pos()
        approach = self._approach_pos(block)
        left_contact, right_contact = self._finger_block_contacts()
        lift_height = max(0.0, float(block[2] - BLOCK_Z))
        cube_speed = float(np.linalg.norm(block_vel))
        cube_on_table = abs(float(block[2]) - BLOCK_Z) < 0.035
        released = self._close_fraction() < 0.25 or not (left_contact and right_contact)
        block_onehot = np.zeros(len(BLOCK_NAMES), dtype=np.float32)
        block_onehot[BLOCK_NAMES.index(self.selected_block)] = 1.0
        target_onehot = np.zeros(len(TARGET_NAMES), dtype=np.float32)
        target_onehot[TARGET_NAMES.index(self.selected_target)] = 1.0
        stage_onehot = np.zeros(len(STAGE_NAMES), dtype=np.float32)
        stage_onehot[int(self.task_stage)] = 1.0
        all_blocks = np.concatenate([self._block_pos(name) for name in BLOCK_NAMES])
        all_targets = np.concatenate([self._target_pos(name) for name in TARGET_NAMES])
        status = np.array(
            [
                self._close_fraction(),
                lift_height / LIFT_HEIGHT,
                float(self._ever_grasped),
                float(self._ever_lifted),
                float(left_contact),
                float(right_contact),
                float(released),
                float(cube_on_table and cube_speed < SETTLE_SPEED),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                q,
                dq,
                ctrl_err,
                ee,
                ee_xmat,
                block,
                block_vel,
                target,
                block - ee,
                target - block,
                approach,
                approach - ee,
                all_blocks,
                all_targets,
                block_onehot,
                target_onehot,
                status,
                stage_onehot,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "task_index" in options:
            self.fixed_task_index = int(options["task_index"])

        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        self._set_task()
        self._randomize_scene()
        self.data.qvel[:] = 0.0
        self._arm_ctrl_target = self.data.qpos[self._act_qpos_addr[:7]].copy()
        self._gripper_ctrl_target = GRIPPER_OPEN_CTRL
        self.data.ctrl[:7] = self._arm_ctrl_target
        self.data.ctrl[7] = self._gripper_ctrl_target

        mujoco.mj_forward(self.model, self.data)

        ee = self.data.site_xpos[self._ee_site_id].copy()
        block = self._block_pos()
        target = self._target_pos()
        self._prev_reach_dist = float(np.linalg.norm(block - ee))
        self._prev_approach_dist = float(np.linalg.norm(self._approach_pos(block) - ee))
        self._best_approach_dist = self._prev_approach_dist
        self._prev_place_dist = float(np.linalg.norm((target - block)[:2]))
        self._prev_lift_height = 0.0
        self._prev_ee_z = float(ee[2])
        self._prev_settle_score = 0.0
        self._step_count = 0
        self._success_hold = 0
        self._ever_grasped = False
        self._ever_lifted = False
        self._max_lift_height = 0.0
        return self._get_obs(), {"language_goal": self.language_goal}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        ee_pos_action = np.clip(action[:3], -1.0, 1.0)
        ee_rot_action = np.clip(action[3:6], -1.0, 1.0)
        gripper_action = float(np.clip(action[6], -1.0, 1.0))

        pre_block = self._block_pos()
        pre_target = self._target_pos()
        pre_place_dist = float(np.linalg.norm((pre_target - pre_block)[:2]))
        pre_lift_height = max(0.0, float(pre_block[2] - BLOCK_Z))
        pre_ee = self.data.site_xpos[self._ee_site_id].copy()
        pre_reach_dist = float(np.linalg.norm(pre_block - pre_ee))
        pre_left_contact, pre_right_contact = self._finger_block_contacts()
        pre_two_finger_contact = pre_left_contact and pre_right_contact
        pre_close_frac = self._close_fraction()
        pre_target_close_frac = 1.0 - float(
            np.clip(self._gripper_ctrl_target / GRIPPER_OPEN_CTRL, 0.0, 1.0)
        )
        pre_holding_or_lifting = (
            self._ever_grasped
            or self._ever_lifted
            or pre_lift_height > SECURE_GRASP_HEIGHT
            or (pre_two_finger_contact and pre_close_frac > 0.20)
            or (pre_reach_dist < 0.075 and pre_close_frac > 0.62)
            or (pre_reach_dist < 0.090 and pre_target_close_frac > 0.35)
        )
        carry_guard_active = self.task_stage >= STAGE_TRANSPORT and pre_holding_or_lifting
        if carry_guard_active:
            ee_pos_action = ee_pos_action * CARRY_POS_SCALE
            ee_rot_action = ee_rot_action * CARRY_ROT_SCALE
            if pre_lift_height > TRANSPORT_LIFT_HEIGHT:
                ee_pos_action[2] = min(ee_pos_action[2], 0.0)
            if pre_lift_height > CARRY_HIGH_HEIGHT:
                ee_pos_action[2] = min(ee_pos_action[2], -0.15)
            if pre_lift_height > CARRY_CRITICAL_HEIGHT:
                ee_pos_action[2] = min(ee_pos_action[2], -0.45)

        pos_delta = ee_pos_action.astype(np.float64) * EE_POS_DELTA
        rot_delta = ee_rot_action.astype(np.float64) * EE_ROT_DELTA
        arm_delta = self._ik_delta(pos_delta, rot_delta)

        q_arm = self.data.qpos[self._act_qpos_addr[:7]].copy()
        next_arm = self._arm_ctrl_target + arm_delta
        next_arm = np.clip(next_arm, q_arm - CTRL_LAG_LIMIT, q_arm + CTRL_LAG_LIMIT)
        next_arm = np.clip(
            next_arm, self._arm_act_ctrlrange[:, 0], self._arm_act_ctrlrange[:, 1]
        )

        next_gripper = self._gripper_ctrl_target - gripper_action * GRIPPER_CTRL_DELTA
        next_gripper = float(
            np.clip(next_gripper, GRIPPER_CLOSED_CTRL, GRIPPER_OPEN_CTRL)
        )
        gripper_open_allowed = True

        if self.task_stage == STAGE_TRANSPORT:
            if pre_holding_or_lifting:
                gripper_open_allowed = False


        elif self.task_stage >= STAGE_PLACE:

            if pre_holding_or_lifting:
                gripper_open_allowed = (

                        pre_place_dist < GATE_RELEASE_RADIUS

                        and pre_lift_height < 0.090

                )

        if not gripper_open_allowed:
            next_gripper = GRIPPER_CLOSED_CTRL

        self._arm_ctrl_target = next_arm
        self._gripper_ctrl_target = next_gripper
        self.data.ctrl[:7] = self._arm_ctrl_target
        self.data.ctrl[7] = self._gripper_ctrl_target

        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        ee = obs[27:30]
        block = obs[39:42]
        block_vel = obs[42:45]
        target = obs[45:48]
        reach_dist = float(np.linalg.norm(block - ee))
        approach_pos = self._approach_pos(block)
        approach_dist = float(np.linalg.norm(approach_pos - ee))
        place_dist = float(np.linalg.norm((target - block)[:2]))
        ee_target_dist = float(np.linalg.norm((target - ee)[:2]))
        close_frac = self._close_fraction()
        lift_height = max(0.0, float(block[2] - BLOCK_Z))
        lift_goal = self._active_lift_goal()
        cube_speed = float(np.linalg.norm(block_vel))
        cube_on_table = abs(float(block[2]) - BLOCK_Z) < 0.035
        left_contact, right_contact = self._finger_block_contacts()
        two_finger_contact = left_contact and right_contact
        vertically_aligned = abs(float(ee[2] - block[2])) < 0.075
        grip_ready = (
            (two_finger_contact and close_frac > 0.25)
            or (reach_dist < 0.085 and close_frac > 0.55 and vertically_aligned)
        )
        grasped = (
            two_finger_contact
            and reach_dist < 0.090
            and close_frac > 0.25
            and vertically_aligned
        )
        first_grasp = grasped and not self._ever_grasped
        if grasped:
            self._ever_grasped = True

        lifted = (
            lift_height > lift_goal
            and self._ever_grasped
            and reach_dist < 0.120
            and close_frac > 0.25
            and cube_speed < 0.45
        )
        full_lifted = (
            lift_height > LIFT_HEIGHT
            and self._ever_grasped
            and reach_dist < 0.130
            and close_frac > 0.25
            and cube_speed < 0.50
        )
        if full_lifted:
            self._ever_lifted = True
        secure_grasped = grasped and lift_height > SECURE_GRASP_HEIGHT
        self._max_lift_height = max(self._max_lift_height, lift_height)

        reach_progress = self._prev_reach_dist - reach_dist
        approach_progress = self._prev_approach_dist - approach_dist
        best_approach_progress = self._best_approach_dist - approach_dist
        place_progress = self._prev_place_dist - place_dist
        lift_progress = lift_height - self._prev_lift_height
        ee_z_progress = float(ee[2] - self._prev_ee_z)
        gripper_open = 1.0 - close_frac
        opened_gripper = gripper_open > GRIPPER_RELEASE_OPEN
        held_like = (
            two_finger_contact
            or (close_frac > GRIPPER_HOLD_CLOSED and reach_dist < PLACE_HOLD_RADIUS)
        )
        lost_object = self._ever_lifted and not held_like and reach_dist > PLACE_HOLD_RADIUS
        released = opened_gripper or not held_like
        ee_z_down, ee_tilt_deg = self._ee_down_metrics()

        reward = 9.0 * np.clip(reach_progress, -0.02, 0.02) - 0.36 * reach_dist - 0.04
        if reach_dist < 0.10:
            reward += 0.05 * close_frac
        elif close_frac > 0.45:
            reward -= 0.07 * close_frac
        if first_grasp:
            reward += 2.0

        if self.task_stage == STAGE_REACH:
            reward += 36.0 * np.clip(approach_progress, -0.02, 0.02)
            if best_approach_progress > 0.0:
                reward += 20.0 * min(best_approach_progress, 0.035)
            elif approach_progress < -0.002:
                reward -= 0.03
            reward -= 0.78 * approach_dist
            if approach_dist < 0.12:
                reward += 0.10
            if approach_dist < 0.09:
                reward += 0.18
            if approach_dist < APPROACH_RADIUS:
                reward += 0.70
            if close_frac > 0.25:
                reward -= 0.09 * close_frac

        if self.task_stage == STAGE_GRASP:
            reward += 0.06 if grip_ready else 0.0
            if reach_dist < 0.09:
                reward += 0.10 * close_frac
            if two_finger_contact:
                reward += 0.20
            if grasped:
                reward += 0.45
            if secure_grasped:
                reward += 0.85

        if self.task_stage == STAGE_LIFT and (grip_ready or lift_height > 0.004):
            lift_fraction = np.clip(lift_height / lift_goal, 0.0, 1.2)
            reward += 120.0 * np.clip(lift_progress, -0.008, 0.008)
            reward += 0.14 * min(lift_fraction, 1.0)
            if grip_ready:
                reward += 8.0 * np.clip(ee_z_progress, -0.008, 0.008)
            if self._ever_grasped and lift_height < SECURE_GRASP_HEIGHT:
                reward -= 0.03

        elif self.task_stage >= STAGE_TRANSPORT and (grip_ready or lift_height > 0.004):
            excess_lift = max(0.0, lift_height - EXCESS_LIFT_HEIGHT)
            high_lift = max(0.0, lift_height - CARRY_HIGH_HEIGHT)
            critical_lift = max(0.0, lift_height - CARRY_CRITICAL_HEIGHT)
            carry_lift_error = abs(lift_height - TRANSPORT_LIFT_HEIGHT)
            down_score = max(0.0, ee_z_down)
            if self._ever_lifted or lift_height > PLACE_MIN_CARRY_HEIGHT:
                reward += 0.35 * down_score
                if down_score < 0.75:
                    reward -= 0.65 * (0.75 - down_score)
            reward -= (
                6.0 * excess_lift
                + 55.0 * excess_lift * excess_lift
                + 10.0 * high_lift
                + 70.0 * high_lift * high_lift
                + 2.0 * critical_lift
            )
            lift_fraction = np.clip(lift_height / LIFT_HEIGHT, 0.0, 1.2)
            if not self._ever_lifted:
                reward += 135.0 * np.clip(lift_progress, -0.008, 0.008)
                reward += 0.24 * min(lift_fraction, 1.0)
                if grip_ready:
                    reward += 9.0 * np.clip(ee_z_progress, -0.008, 0.008)
                if close_frac < GRIPPER_HOLD_CLOSED or not held_like:
                    reward -= 0.18
            else:
                carry_height_score = float(
                    np.clip(1.0 - carry_lift_error / PLACE_CARRY_HEIGHT_BAND, 0.0, 1.0)
                )
                if place_dist > PLACE_LOWER_RADIUS:
                    reward += 0.16 + 0.34 * carry_height_score
                    reward -= 10.0 * max(0.0, PLACE_MIN_CARRY_HEIGHT - lift_height)
                    if close_frac < GRIPPER_HOLD_CLOSED or not held_like:
                        reward -= 0.70
                    elif close_frac > GRIPPER_HOLD_CLOSED and held_like:
                        reward += 0.22
                else:
                    reward += 0.08 + 0.14 * carry_height_score

        if self.task_stage >= STAGE_TRANSPORT and (full_lifted or self._ever_lifted):
            transport_ready = (
                lift_height >= PLACE_MIN_CARRY_HEIGHT
                and close_frac > GRIPPER_HOLD_CLOSED
                and held_like
            )
            if place_dist > PLACE_LOWER_RADIUS:
                if transport_ready:
                    reward += 36.0 * np.clip(place_progress, -0.02, 0.02)
                    reward += 0.06 - 1.10 * place_dist

                    # Force the gripper/EE to move above the target plate/bowl during transport.
                    reward += 0.10 - 0.90 * ee_target_dist
                    if ee_target_dist < 0.080:
                        reward += 0.25
                    if ee_target_dist < 0.060:
                        reward += 0.35
                else:
                    reward -= 0.25 + 8.0 * max(0.0, PLACE_MIN_CARRY_HEIGHT - lift_height)
                if opened_gripper or lost_object:
                    reward -= 1.80 + 1.70 * min(place_dist, 0.20)
                else:
                    reward += 0.16 * min(close_frac, 1.0)
            else:
                reward += 20.0 * np.clip(place_progress, -0.02, 0.02)
                reward += 0.06 - 0.80 * place_dist
                down_score = max(0.0, ee_z_down)
                if self.task_stage == STAGE_TRANSPORT:
                    reward += 0.90 * down_score
                    if down_score < 0.75:
                        reward -= 1.20 * (0.75 - down_score)
                else:
                    reward += 0.22 * down_score
                    if down_score < 0.45:
                        reward -= 0.22 * (0.45 - down_score)
                if self.task_stage == STAGE_TRANSPORT:
                    if opened_gripper or lost_object:
                        reward -= 1.20
                    else:
                        reward += 0.25 * min(close_frac, 1.0)

        if self.task_stage >= STAGE_PLACE and self._ever_lifted and place_dist < PLACE_LOWER_RADIUS:
            lowering_progress = self._prev_lift_height - lift_height
            reward += 92.0 * np.clip(lowering_progress, -0.006, 0.008)
            if lift_height < TRANSPORT_LIFT_HEIGHT:
                reward += 0.22 * (
                    1.0 - np.clip(lift_height / TRANSPORT_LIFT_HEIGHT, 0.0, 1.0)
                )
            if lift_height < 0.050:
                reward += 0.50 * (1.0 - close_frac)
            elif opened_gripper or not held_like:
                reward -= 0.25

        low_cube_speed = cube_speed < SETTLE_SPEED
        valid_release = self._ever_lifted and place_dist < RELEASE_RADIUS and cube_on_table and released
        open_near_target = (
            self._ever_lifted
            and place_dist < RELEASE_RADIUS
            and opened_gripper
        )
        dropped_far = (
            self._ever_lifted
            and (opened_gripper or lost_object)
            and place_dist > PLACE_LOWER_RADIUS
            and (lift_height < TRANSPORT_LIFT_HEIGHT or cube_on_table)
        )
        place_ee_down_score = (
            max(0.0, ee_z_down)
            if self._ever_lifted and place_dist < PLACE_LOWER_RADIUS
            else 0.0
        )
        place_precision = float(np.clip(1.0 - place_dist / SUCCESS_RADIUS, 0.0, 1.0))
        release_zone = float(np.clip(1.0 - place_dist / RELEASE_RADIUS, 0.0, 1.0))
        table_score = float(np.clip(1.0 - lift_height / SETTLE_HEIGHT_RADIUS, 0.0, 1.0))
        speed_score = float(np.clip(1.0 - cube_speed / SETTLE_SPEED, 0.0, 1.0))
        release_score = float(np.clip(gripper_open, 0.0, 1.0))
        if not held_like and place_dist < RELEASE_RADIUS:
            release_score = max(release_score, 0.75)
        settle_score = place_precision * table_score * speed_score * release_score
        settle_progress = settle_score - self._prev_settle_score

        if self.task_stage >= STAGE_PLACE and self._ever_lifted:
            if (opened_gripper or lost_object) and place_dist > RELEASE_RADIUS:
                reward -= 0.55 + 0.60 * min(place_dist, 0.20)
            if opened_gripper and lift_height > TRANSPORT_LIFT_HEIGHT:
                reward -= 0.25
            if place_dist < RELEASE_RADIUS:
                reward += 2.50 * release_zone
                reward += 3.2 * np.clip(settle_progress, -0.20, 0.20)
                reward += 2.50 * settle_score
                reward += 4.00 * release_score * release_zone

                # Penalize holding near target, even if the cube is still slightly high.
                reward -= 0.60 * close_frac * release_zone

                if cube_speed > SETTLE_SPEED and lift_height < 0.055:
                    reward -= 0.85 * min(cube_speed, 0.50) * release_zone
            elif (opened_gripper or lost_object) and lift_height < 0.050:
                reward -= 0.20

            if place_dist < self.success_radius and not released:
                reward -= 0.80 * close_frac
            if (opened_gripper or lost_object) and place_dist > PLACE_LOWER_RADIUS:
                reward -= 0.32 * min(place_dist, 0.20)
        elif self.task_stage >= STAGE_PLACE and self._ever_grasped and released:
            reward -= 0.30

        placed = (
            self._ever_lifted
            and place_dist < self.success_radius
            and cube_on_table
            and low_cube_speed
            and released
        )

        if self.task_stage == STAGE_REACH:
            stage_done = approach_dist < APPROACH_RADIUS and reach_dist < REACH_RADIUS
            hold_target = REACH_HOLD_STEPS
            success_bonus = 5.0
            if stage_done:
                reward += 0.9
        elif self.task_stage == STAGE_GRASP:
            stage_done = grasped
            hold_target = GRASP_HOLD_STEPS
            success_bonus = 6.5
        elif self.task_stage == STAGE_LIFT:
            stage_done = lifted
            hold_target = LIFT_HOLD_STEPS
            success_bonus = 14.5 + 20.0 * lift_goal
            if lifted:
                reward += 1.3 + 8.0 * lift_height
        elif self.task_stage == STAGE_TRANSPORT:
            transport_done = (
                self._ever_lifted
                and place_dist < TRANSPORT_RADIUS
                and lift_height >= PLACE_MIN_CARRY_HEIGHT
                and lift_height <= MAX_TRANSPORT_SUCCESS_HEIGHT
                and held_like
                and not opened_gripper
                and cube_speed < 0.55
                and ee_z_down > 0.55
                and ee_target_dist < 0.085
            )
            stage_done = transport_done
            hold_target = TRANSPORT_HOLD_STEPS
            success_bonus = 18.0
            if transport_done:
                reward += 2.2 + 0.45 * self._success_hold
        else:
            stage_done = placed
            hold_target = SUCCESS_HOLD_STEPS
            success_bonus = 38.0
            if placed:
                reward += 4.5 + 0.55 * self._success_hold

        if stage_done:
            self._success_hold += 1
        else:
            self._success_hold = 0

        action_cost = (
            0.0045 * float(np.square(ee_pos_action).sum())
            + 0.0018 * float(np.square(ee_rot_action).sum())
            + 0.0022 * abs(gripper_action)
        )
        if self.task_stage == STAGE_TRANSPORT and not gripper_open_allowed and gripper_action < 0.0:
            reward -= 0.04 * abs(gripper_action)
        ik_cost = 0.012 * float(np.square(arm_delta).sum())
        speed_cost = 0.0008 * float(np.linalg.norm(obs[9:18]))
        reward -= action_cost + ik_cost + speed_cost

        terminated = self._success_hold >= hold_target
        if terminated:
            reward += success_bonus

        unstable = (
            not np.all(np.isfinite(obs))
            or block[2] > TABLE_TOP_Z + 0.60
            or block[2] < TABLE_TOP_Z - 0.08
            or np.linalg.norm(self.data.qvel[:9]) > 45.0
        )
        if unstable:
            reward -= 12.0
        reward = float(np.clip(reward, -15.0, 25.0))

        self._prev_reach_dist = reach_dist
        self._prev_approach_dist = approach_dist
        self._best_approach_dist = min(self._best_approach_dist, approach_dist)
        self._prev_place_dist = place_dist
        self._prev_lift_height = lift_height
        self._prev_ee_z = float(ee[2])
        self._prev_settle_score = settle_score
        self._step_count += 1
        truncated = self._step_count >= MAX_STEPS or unstable

        info = {
            "success": bool(terminated),
            "task_success": bool(terminated if self.task_stage == STAGE_PLACE else False),
            "stage": int(self.task_stage),
            "stage_name": STAGE_NAMES[int(self.task_stage)],
            "language_goal": self.language_goal,
            "selected_block": self.selected_block,
            "selected_target": self.selected_target,
            "reach_dist": reach_dist,
            "approach_dist": approach_dist,
            "best_approach_dist": self._best_approach_dist,
            "place_dist": place_dist,
            "ee_target_dist": ee_target_dist,
            "block_z": float(block[2]),
            "grasped": float(grasped or self._ever_grasped),
            "secure_grasped": float(secure_grasped),
            "lifted": float(lifted or self._ever_lifted),
            "full_lifted": float(full_lifted or self._ever_lifted),
            "placed": float(placed),
            "released": float(valid_release),
            "raw_released": float(released),
            "held_like": float(held_like),
            "opened_gripper": float(opened_gripper),
            "open_near_target": float(open_near_target),
            "dropped_far": float(dropped_far),
            "cube_on_table": float(cube_on_table),
            "cube_speed": cube_speed,
            "low_cube_speed": float(low_cube_speed),
            "success_radius": float(self.success_radius),
            "settle_score": settle_score,
            "max_lift_height": self._max_lift_height,
            "lift_goal_height": lift_goal,
            "gripper_closed": close_frac,
            "gripper_open": gripper_open,
            "post_lift_gripper_open": gripper_open if self._ever_lifted else 0.0,
            "gripper_open_allowed": float(gripper_open_allowed),
            "gripper_gate_closed": float(not gripper_open_allowed),
            "carry_guard": float(carry_guard_active),
            "over_lift": max(0.0, lift_height - TRANSPORT_LIFT_HEIGHT),
            "ee_z_down": ee_z_down,
            "ee_tilt_deg": ee_tilt_deg,
            "place_ee_down_score": place_ee_down_score,
            "hold": self._success_hold,
            "unstable": float(unstable),
            "ee_delta_norm": float(np.linalg.norm(pos_delta)),
            "ik_delta_norm": float(np.linalg.norm(arm_delta)),
        }
        return obs, reward, terminated, truncated, info

    def render(self, camera="fixed_scene"):
        if self.renderer is None:
            return None
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {CAMERAS}, got {camera!r}")
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.renderer:
            self.renderer.close()
