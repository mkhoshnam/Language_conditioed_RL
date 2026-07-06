import gymnasium as gym

gym.register(
    id="Mohammad-OpenDrawerStage-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "full_drawer_task.chained_env_cfg:OpenDrawerStageEnvCfg",
        "rsl_rl_cfg_entry_point": "full_drawer_task.chained_rsl_rl_ppo_cfg:OpenDrawerStagePPORunnerCfg",
    },
)

gym.register(
    id="Mohammad-PickPlaceAfterDrawerStage-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "full_drawer_task.chained_env_cfg:PickPlaceAfterDrawerStageEnvCfg",
        "rsl_rl_cfg_entry_point": "full_drawer_task.chained_rsl_rl_ppo_cfg:PickPlaceAfterDrawerPPORunnerCfg",
    },
)
