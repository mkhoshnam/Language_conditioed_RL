# True Language-Conditioned Transformer PPO

This is the runnable implementation of the proposed token-level language policy.
It is deliberately separate from the repository's original structured-task PPO,
so both systems remain reproducible.

## What makes this implementation truly language conditioned

The actor receives two streams:

1. raw command word-piece embeddings from a frozen pretrained MiniLM encoder;
2. task-agnostic semantic tokens for the robot, end effector, gripper, three
   blocks, and four destinations.

The actor never receives the selected block, selected destination, task index,
place/stack bit, reward stage, or a parser-produced goal vector. Those privileged
labels remain inside the environment for reward calculation and curriculum
metrics only. Language and state tokens meet in one shared transformer, with
separate learned `[ACT]` and `[VALUE]` readouts.

## Architecture

| Component | Default |
| --- | --- |
| Language encoder | frozen `sentence-transformers/all-MiniLM-L6-v2` |
| Hidden size | 256 |
| Transformer | 4 layers, 8 heads, FFN 1024, dropout 0.1 |
| State tokens | robot, end effector, gripper, 3 blocks, 4 targets |
| Policy | tanh-squashed diagonal Gaussian over 7 continuous actions |
| Critic | scalar value from the `[VALUE]` token |
| Optimizer | AdamW, lower learning rate for language projection |
| RL update | clipped PPO, GAE, value loss, entropy bonus, gradient clipping |

## Install

Use the same Python environment as the base project, then update the editable
installation so the pretrained text dependency is available:

```bash
pip install -e .
```

The first real training run downloads the configured MiniLM checkpoint from
Hugging Face and then uses the local cache. The deterministic `hash` text backend
exists only so unit and integration smoke tests can run offline; it is not the
research configuration.

## Train placing first

```bash
./scripts/transformer/train_place.sh
```

The place run progresses through reach, grasp, lift, transport, release, and
settle. Twenty percent of episodes continue to sample earlier objectives after
each promotion, and every later stage still begins from the full episode reset.
Canonical commands are mixed with paraphrases.

For a short launch check:

```bash
python scripts/transformer/train.py \
  --config configs/transformer/place.json \
  --steps 4096
```

## Transfer placing to stacking

The transfer config loads the best place checkpoint and reduces the learning
rate. It starts with 10% stacking episodes and increases the stack fraction only
when grasp/lift remain strong, stacking improves, and placing is retained.

```bash
./scripts/transformer/train_transfer.sh
```

If the place checkpoint has another name:

```bash
./scripts/transformer/train_transfer.sh \
  --resume checkpoints/transformer/my_place_checkpoint.pt
```

The complete parameter set transfers: language projection, state tokenizers,
transformer, actor, critic, and exploration scale. MiniLM remains frozen.

## Evaluate

```bash
python scripts/transformer/evaluate.py \
  checkpoints/transformer/transformer_place_to_stack_best.pt \
  --episodes 50 --paraphrases
```

Evaluate an exact user instruction and run a same-scene command intervention:

```bash
python scripts/transformer/evaluate.py CHECKPOINT.pt \
  --command "put the red block on top of the blue block" \
  --episodes 20 --intervention
```

## Validation

```bash
python scripts/transformer/smoke_test.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

For scientific transfer evidence, compare at least three equal-budget runs:
stacking from scratch, joint place+stack from scratch, and place-initialized
stacking. Report steps to reach/grasp/lift/stack thresholds, final success,
place retention, held-out source-destination combinations, paraphrases, and
variance across seeds.
