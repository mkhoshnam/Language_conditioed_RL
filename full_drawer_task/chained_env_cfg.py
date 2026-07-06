import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    ObjectTableSceneCfg,
    RewardsCfg as LiftRewardsCfg,
    TerminationsCfg as LiftTerminationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

from full_drawer_task import chained_mdp


@configclass
class SharedDrawerCubeSceneCfg(ObjectTableSceneCfg):
    # Same scene for both policies:
    # cube in front, drawer/cabinet on the side.
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.45, 0.55, 0.40),
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
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
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )


@configclass
class OpenDrawerStageRewardsCfg(LiftRewardsCfg):
    ee_to_drawer_handle = RewTerm(
        func=chained_mdp.ee_to_drawer_handle_reward,
        weight=4.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "handle_pos": (0.45, 0.34, 0.58),
        },
    )

    gripper_close_near_handle = RewTerm(
        func=chained_mdp.gripper_close_near_handle_dense_reward,
        weight=15.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_finger_.*"]),
            "handle_pos": (0.45, 0.34, 0.58),
        },
    )

    pull_after_handle = RewTerm(
        func=chained_mdp.pull_after_real_handle_grip_reward,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_finger_.*"]),
            "handle_pos": (0.45, 0.34, 0.58),
            "pull_target": (0.45, 0.14, 0.58),
        },
    )

    drawer_open = RewTerm(
        func=chained_mdp.drawer_open_reward,
        weight=120.0,
        params={
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "open_threshold": 0.32,
        },
    )


@configclass
class OpenDrawerStageTerminationsCfg(LiftTerminationsCfg):
    success = DoneTerm(
        func=chained_mdp.drawer_open_success,
        params={
            "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            "open_threshold": 0.32,
        },
    )


@configclass
class PickPlaceAfterDrawerRewardsCfg(LiftRewardsCfg):
    cube_inside_drawer = RewTerm(
        func=chained_mdp.cube_inside_drawer_reward,
        weight=35.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target": (0.42, 0.38, 0.44),
            "half_extents": (0.17, 0.18, 0.10),
        },
    )


@configclass
class PickPlaceAfterDrawerTerminationsCfg(LiftTerminationsCfg):
    success = DoneTerm(
        func=chained_mdp.cube_inside_drawer_success,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target": (0.42, 0.38, 0.44),
            "half_extents": (0.14, 0.15, 0.08),
        },
    )


@configclass
class BaseSharedDrawerCubeEnvCfg(FrankaCubeLiftEnvCfg):
    scene: SharedDrawerCubeSceneCfg = SharedDrawerCubeSceneCfg(num_envs=4096, env_spacing=3.0)

    def __post_init__(self):
        super().__post_init__()

        # same geometry for both stages
        self.scene.object.init_state.pos = [0.55, -0.15, 0.055]

        # target is inside the side drawer
        self.commands.object_pose.resampling_time_range = (100000.0, 100000.0)
        self.commands.object_pose.ranges.pos_x = (0.42, 0.42)
        self.commands.object_pose.ranges.pos_y = (0.38, 0.38)
        self.commands.object_pose.ranges.pos_z = (0.44, 0.44)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (0.0, 0.0)

        self.episode_length_s = 10.0
        self.observations.policy.enable_corruption = False

        # extra observations shared by both policies:
        # policy 1 sees drawer/handle;
        # policy 2 also gets the same obs shape, so final policy switching is possible.
        self.observations.policy.drawer_joint_pos = ObsTerm(
            func=chained_mdp.drawer_joint_pos_obs,
            params={
                "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            },
        )
        self.observations.policy.ee_to_drawer_handle = ObsTerm(
            func=chained_mdp.ee_to_drawer_handle_vec_obs,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
                "handle_pos": (0.45, 0.34, 0.58),
            },
        )
        self.observations.policy.drawer_place_target = ObsTerm(
            func=chained_mdp.drawer_target_pos_obs,
            params={
                "target": (0.42, 0.38, 0.44),
            },
        )

        self.observations.policy.ee_to_pull_target = ObsTerm(
            func=chained_mdp.ee_to_pull_target_vec_obs,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
                "pull_target": (0.45, 0.14, 0.58),
            },
        )

        # camera for later video: sees cube in front + drawer on side
        self.viewer.eye = (-1.7, 2.2, 1.5)
        self.viewer.lookat = (0.45, 0.25, 0.45)

        # safer contact capacity
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024


@configclass
class OpenDrawerStageEnvCfg(BaseSharedDrawerCubeEnvCfg):
    rewards: OpenDrawerStageRewardsCfg = OpenDrawerStageRewardsCfg()
    terminations: OpenDrawerStageTerminationsCfg = OpenDrawerStageTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # drawer closed at start
        self.scene.cabinet.init_state.joint_pos["drawer_top_joint"] = 0.0

        # disable cube/lift/place reward for drawer-opening stage
        for name in [
            "reaching_object",
            "lifting_object",
            "object_goal_tracking",
            "object_goal_tracking_fine_grained",
        ]:
            if hasattr(self.rewards, name):
                getattr(self.rewards, name).weight = 0.0

        if hasattr(self.rewards, "action_rate"):
            self.rewards.action_rate.weight = -1e-4
        if hasattr(self.rewards, "joint_vel"):
            self.rewards.joint_vel.weight = -1e-4


@configclass
class PickPlaceAfterDrawerStageEnvCfg(BaseSharedDrawerCubeEnvCfg):
    rewards: PickPlaceAfterDrawerRewardsCfg = PickPlaceAfterDrawerRewardsCfg()
    terminations: PickPlaceAfterDrawerTerminationsCfg = PickPlaceAfterDrawerTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # this stage starts from the END CONDITION of policy 1:
        # drawer already open. Later we will replace this with collected
        # final states from the trained drawer-opening policy.
        self.scene.cabinet.init_state.joint_pos["drawer_top_joint"] = 0.35

        self.rewards.reaching_object.weight = 2.0
        self.rewards.lifting_object.weight = 18.0
        self.rewards.object_goal_tracking.weight = 25.0
        self.rewards.object_goal_tracking_fine_grained.weight = 12.0

        if hasattr(self.rewards, "action_rate"):
            self.rewards.action_rate.weight = -1e-4
        if hasattr(self.rewards, "joint_vel"):
            self.rewards.joint_vel.weight = -1e-4
