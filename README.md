# Language_conditioed_RL

> Ongoing project: language-conditioned reinforcement learning for robotic pick-and-place.

This project trains a policy that receives a user language prompt and conditions the robot behavior on that task. Example goals include instructions such as placing a colored block into a specified plate or bowl. The RL policy is trained with staged PPO so the robot learns to reach, grasp, lift, transport, place, release, and settle the object.

## Environment

![Language-conditioned RL environment](Env.png)

## Project Idea

The goal is to connect natural-language task descriptions with robotic control. A prompt defines the selected object and target, and the policy uses that conditioning signal to complete the manipulation task in simulation.

## What Is Included

- `env.py`: MuJoCo/Gymnasium environment for the language-conditioned task.
- `train.py`: PPO training with staged curriculum.
- `evaluate.py`: Policy evaluation and video recording.
- `ppo.py`: PPO implementation.
- `smoke_test.py`: Quick environment sanity check.
- `Env.png`: environment preview.

## Run

```bash
python train.py
```

Evaluate a checkpoint:

```bash
python evaluate.py path/to/checkpoint.pt
```

Run a quick smoke test:

```bash
python smoke_test.py
```
