import torch
from isaaclab.managers import SceneEntityCfg


def _asset_joint_pos(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def _robot_body_pos_env(env, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    return pos_w - env.scene.env_origins


def _object_pos_env(env, object_cfg: SceneEntityCfg):
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w - env.scene.env_origins


def drawer_open_reward(
    env,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
    open_threshold: float = 0.32,
):
    q = _asset_joint_pos(env, cabinet_cfg).squeeze(-1)
    return torch.clamp(q / open_threshold, 0.0, 1.5)


def drawer_open_success(
    env,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
    open_threshold: float = 0.32,
):
    q = _asset_joint_pos(env, cabinet_cfg).squeeze(-1)
    return q > open_threshold


def ee_to_drawer_handle_reward(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
    handle_pos=(0.45, 0.34, 0.58),
):
    ee_pos = _robot_body_pos_env(env, robot_cfg)
    target = torch.tensor(handle_pos, device=ee_pos.device).unsqueeze(0)
    dist = torch.norm(ee_pos - target, dim=-1)
    return torch.exp(-dist / 0.12)


def cube_inside_drawer_reward(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target=(0.42, 0.38, 0.44),
    half_extents=(0.17, 0.18, 0.10),
):
    pos = _object_pos_env(env, object_cfg)
    target_t = torch.tensor(target, device=pos.device).unsqueeze(0)
    half_t = torch.tensor(half_extents, device=pos.device).unsqueeze(0)

    dist = torch.norm(pos - target_t, dim=-1)
    inside = (torch.abs(pos - target_t) < half_t).all(dim=-1)

    return torch.exp(-dist / 0.06) + 5.0 * inside.float()


def cube_inside_drawer_success(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target=(0.42, 0.38, 0.44),
    half_extents=(0.14, 0.15, 0.08),
):
    pos = _object_pos_env(env, object_cfg)
    target_t = torch.tensor(target, device=pos.device).unsqueeze(0)
    half_t = torch.tensor(half_extents, device=pos.device).unsqueeze(0)
    return (torch.abs(pos - target_t) < half_t).all(dim=-1)


def drawer_joint_pos_obs(
    env,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
):
    return _asset_joint_pos(env, cabinet_cfg)


def ee_to_drawer_handle_vec_obs(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
    handle_pos=(0.45, 0.34, 0.58),
):
    ee_pos = _robot_body_pos_env(env, robot_cfg)
    target = torch.tensor(handle_pos, device=ee_pos.device).unsqueeze(0)
    return target - ee_pos


def drawer_target_pos_obs(
    env,
    target=(0.42, 0.38, 0.44),
):
    return torch.tensor(target, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)


def gripper_close_near_handle_reward(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_finger_.*"]),
    handle_pos=(0.45, 0.34, 0.58),
):
    robot = env.scene[robot_cfg.name]

    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    target = torch.tensor(handle_pos, device=ee_pos.device).unsqueeze(0)
    dist = torch.norm(ee_pos - target, dim=-1)

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    gripper_closed = torch.mean(finger_pos, dim=-1) < 0.018

    near = dist < 0.10
    return near.float() * gripper_closed.float()


def pull_after_handle_reward(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_finger_.*"]),
    pull_target=(0.45, 0.14, 0.58),
):
    robot = env.scene[robot_cfg.name]

    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    target = torch.tensor(pull_target, device=ee_pos.device).unsqueeze(0)
    dist = torch.norm(ee_pos - target, dim=-1)

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    gripper_closed = torch.mean(finger_pos, dim=-1) < 0.018

    return gripper_closed.float() * torch.exp(-dist / 0.15)


def ee_to_pull_target_vec_obs(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
    pull_target=(0.45, 0.14, 0.58),
):
    ee_pos = _robot_body_pos_env(env, robot_cfg)
    target = torch.tensor(pull_target, device=ee_pos.device).unsqueeze(0)
    return target - ee_pos


def gripper_close_near_handle_dense_reward(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_finger_.*"]),
    handle_pos=(0.45, 0.34, 0.58),
):
    robot = env.scene[robot_cfg.name]

    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins
    target = torch.tensor(handle_pos, device=ee_pos.device).unsqueeze(0)
    dist = torch.norm(ee_pos - target, dim=-1)

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    mean_finger = torch.mean(finger_pos, dim=-1)

    # Franka fingers open around 0.04, closed near 0.0
    closedness = 1.0 - torch.clamp(mean_finger / 0.04, 0.0, 1.0)
    near_handle = torch.exp(-dist / 0.08)

    return near_handle * closedness


def pull_after_real_handle_grip_reward(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"], joint_names=["panda_finger_.*"]),
    handle_pos=(0.45, 0.34, 0.58),
    pull_target=(0.45, 0.14, 0.58),
):
    robot = env.scene[robot_cfg.name]

    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :] - env.scene.env_origins

    handle = torch.tensor(handle_pos, device=ee_pos.device).unsqueeze(0)
    pull = torch.tensor(pull_target, device=ee_pos.device).unsqueeze(0)

    dist_handle = torch.norm(ee_pos - handle, dim=-1)
    dist_pull = torch.norm(ee_pos - pull, dim=-1)

    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    mean_finger = torch.mean(finger_pos, dim=-1)
    closedness = 1.0 - torch.clamp(mean_finger / 0.04, 0.0, 1.0)

    # Crucial: only reward pulling if it first learned to close near the handle.
    handle_gate = torch.exp(-dist_handle / 0.08)
    pull_progress = torch.exp(-dist_pull / 0.12)

    return handle_gate * closedness * pull_progress
