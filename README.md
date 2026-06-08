# Multi task Language conditioned RL training (This is an ongoing project, update monthly- stay tuned)

It uses the MuJoCo Menagerie Franka Panda scene in `calvin_franka_scene/` with:

- real Panda joints, meshes, gripper tendon, contacts, and gravity compensation
- fixed scene camera and wrist camera
- randomized colored block and target placements
- goal-conditioned tasks such as `put the red block in the yellow plate`
- PPO with staged curriculum: reach -> grasp -> lift -> transport -> place/release
- carry/release gate: after grasp/lift, transport keeps the gripper closed; final place unlocks release only near the target after lowering
- carry-height guard: while carrying, motion is slowed and upward motion is blocked once the cube is already above the desired transport height
- best checkpoint saving so later PPO drift does not overwrite the best policy

The RL action is still continuous and real: end-effector translation, wrist rotation, and gripper command. The policy must learn when to approach, close, lift, transport, lower, release, and settle the block.

## Start Training

From the repo root:

```bash
cd your path
source franka_rl_env/bin/activate
rm -rf real_franka_pick_place/checkpoints
python3 -u real_franka_pick_place/train.py
```

Or use the launcher:

```bash
./real_franka_pick_place/start_training.sh
```

Useful long-run command:

```bash
./real_franka_pick_place/start_training.sh > real_franka_pick_place/train_latest.log 2>&1
tail -f real_franka_pick_place/train_latest.log
```

## Resume From Good Lift Checkpoint

If reach/grasp/lift are good and place drifted, resume from the protected pre-place checkpoint. This restarts at the transport stage before learning the final release:

```bash
cd your path
source franka_rl_env/bin/activate
RESUME_CKPT=real_franka_pick_place/checkpoints/ppo_real_franka_pre_place.pt \
./real_franka_pick_place/start_training.sh
```

The current reward is phase-gated: lift first, transport the held block above the target, then lower, release, and settle. The trainer waits for reliable transport before entering release/place. In the logs, `Gate` should be low before grasp, then active after the object is held/lifted. `Guard` should be active while carrying, and `OverLift` should fall; in `Stage place`, release unlocks only when the block is near the target and low enough.

## Best Checkpoints

Training saves several protected best models:

- `checkpoints/ppo_real_franka_best_stage.pt`
- `checkpoints/ppo_real_franka_best_place_success.pt`
- `checkpoints/ppo_real_franka_best_place_hard.pt`
- `checkpoints/ppo_real_franka_best_settle.pt`

Use the best-place-success checkpoint first for videos.

## Evaluate And Record

```bash
N_EPISODES=20 VIDEO_EPISODES=5 CAMERA=both \
VIDEO_PATH=real_franka_pick_place/eval_best.mp4 \
python3 real_franka_pick_place/evaluate.py \
real_franka_pick_place/checkpoints/ppo_real_franka_best_place_success.pt
```

Camera options are `fixed_scene`, `wrist_camera`, or `both`.

## Smoke Test

```bash
python3 real_franka_pick_place/smoke_test.py
```

This saves render checks in `real_franka_pick_place/renders/`.
