# Language-Conditioned Multi-Task RL

> Multi-task reinforcement learning for language-directed placing and stacking with a simulated Franka Panda.

This project trains one PPO agent to understand a natural-language goal, pick the
requested block, and either place it in a target or stack it on another block. The
policy learns the full continuous-control sequence: reach, grasp, lift, transport,
lower, release, and settle.

![Language-conditioned RL environment](assets/Env.png)

## Highlights

- 18 language-conditioned goals: 12 place tasks and 6 ordered stack tasks
- shared PPO representation with separate actor, critic, and exploration heads for each skill
- skill-aware advantage normalization to keep place and stack learning balanced
- staged curriculum from reaching through stable release
- progressive stack-task mixing with independent success reporting
- release gates, carry-height guards, and contact-aware stack success checks
- deterministic local command parser with optional OpenAI-backed parsing
- backward-compatible loading for older single-head, 101-feature checkpoints
- fixed-scene and wrist-camera rendering, evaluation, and sequence recording
- separate true-language transformer PPO with raw token conditioning and place-to-stack transfer

## True-Language Transformer Extension

The original structured-task PPO remains available and unchanged. A separate
implementation now matches the proposed true-language architecture: frozen
pretrained word-piece embeddings, typed robot/entity tokens, a shared
multimodal transformer, and independent action/value readouts. The actor does
not receive a task ID, parser goal vector, selected-object one-hot, skill bit, or
curriculum-stage input.

Train the placing curriculum, then transfer the complete checkpoint to stacking:

```bash
./scripts/transformer/train_place.sh
./scripts/transformer/train_transfer.sh
```

See [the transformer PPO guide](docs/transformer_ppo.md) for architecture,
configuration, evaluation, smoke tests, and the transfer protocol.

## Environment

The CALVIN-style MuJoCo scene includes:

- a Franka Panda model with meshes, contacts, gripper tendon, and gravity compensation
- red, blue, and green movable blocks
- yellow, purple, orange, and cyan destinations
- randomized block and destination layouts
- end-effector translation, wrist rotation, and gripper actions

The 104-value policy observation contains robot state, all block and target
positions, task identity, curriculum stage, a place/stack one-hot selector, and
the destination height. Stack success additionally requires block-to-block
contact at the correct height after release.

## Repository Layout

```text
assets/                         project images
scripts/                        training, evaluation, and smoke-test entry points
src/language_conditioned_rl/    environment, PPO, parser, and task catalog
tests/                          multi-task catalog and policy-routing tests
third_party/calvin_franka_scene MuJoCo Franka scene, meshes, and upstream license
```

Checkpoints, videos, logs, caches, and generated renders are ignored so training
artifacts do not clutter the repository.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/mkhoshnam/Language_conditioed_RL.git
cd Language_conditioed_RL
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For a non-editable setup, `pip install -r requirements.txt` also works.

## Quick Check

Run the environment smoke test:

```bash
python scripts/smoke_test.py
```

It steps a randomized task and saves fixed, wrist, and combined camera images to
`renders/`.

Run the lightweight multi-task tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Task Catalog

Every source block can be placed in every fixed destination:

```text
put the {red|blue|green} block in the {yellow|purple|orange} plate
put the {red|blue|green} block in the cyan bowl
```

Every block can also be stacked on either of the other two blocks:

```text
stack the red block on the blue block
stack the blue block on the green block
stack the green block on the red block
...and the other three ordered combinations
```

The canonical task definitions and stable task indices live in
`src/language_conditioned_rl/task_config.py`.

## Training

Start a fresh curriculum run:

```bash
./scripts/start_training.sh
```

or:

```bash
python scripts/train.py
```

Useful controls are environment variables, so experiments do not require source
edits:

```bash
TOTAL_STEPS=12000000 \
STACK_TASK_FRACTION_START=0.20 \
STACK_TASK_FRACTION_MAX=0.50 \
./scripts/start_training.sh
```

The stack fraction only increases after enough stack episodes meet the configured
success threshold. Important settings include:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `TOTAL_STEPS` | `12000000` | total environment steps |
| `N_STEPS` | `4096` | rollout length per PPO update |
| `STACK_TASK_FRACTION_START` | `0.20` | initial probability of a stack episode |
| `STACK_TASK_FRACTION_MAX` | `0.50` | maximum stack probability |
| `STACK_TASK_FRACTION_SUCCESS` | `0.35` | stack success needed before increasing the mix |
| `RESUME_CKPT` | unset | checkpoint used to resume training |
| `CKPT_DIR` | `checkpoints/` | output directory |

Resume an existing model:

```bash
RESUME_CKPT=checkpoints/ppo_real_franka_best_place_success.pt \
./scripts/start_training.sh
```

Older single-head checkpoints are expanded into the dual-head network when they
are loaded. Their learned place behavior initializes both skill heads, allowing
stack training to begin from a useful manipulation policy.

For stack-only specialization from a strong place checkpoint:

```bash
RESUME_CKPT=checkpoints/ppo_real_franka_best_place_success.pt \
STACK_SPECIALIZE=1 \
STACK_TASK_FRACTION_START=1.0 \
./scripts/start_training.sh
```

This freezes the shared trunk, place actor, place critic, and observation
statistics while training the stack-specific heads.

## Checkpoints

Training keeps periodic snapshots and protected best models under `checkpoints/`:

```text
ppo_real_franka_best_stage.pt
ppo_real_franka_best_place_success.pt
ppo_real_franka_best_place_hard.pt
ppo_real_franka_best_settle.pt
ppo_real_franka_final.pt
```

Model files are intentionally not committed because they are generated artifacts
and can become large.

## Evaluation

Evaluate a checkpoint across randomized tasks:

```bash
python scripts/evaluate.py checkpoints/ppo_real_franka_best_place_success.pt
```

Evaluate one language command:

```bash
COMMAND="stack the red block on the blue block" \
N_EPISODES=20 VIDEO_EPISODES=3 CAMERA=both \
python scripts/evaluate.py checkpoints/ppo_real_franka_best_place_success.pt
```

`CAMERA` accepts `fixed_scene`, `wrist_camera`, or `both`.

## Language Commands and Sequences

Parse a command without starting MuJoCo:

```bash
python -m language_conditioned_rl.llm_parser \
  "put the green block on top of the red block"
```

If `OPENAI_API_KEY` is available, the parser can use the OpenAI API. If the API
is unavailable, deterministic color and object matching is used automatically.

Run place and stack commands in the same scene:

```bash
SEQUENCE_COMMANDS="put the red block in the yellow plate ;; stack the green block on the blue block" \
N_SEQUENCES=10 VIDEO_SEQUENCES=1 CAMERA=fixed_scene \
python scripts/evaluate_sequence.py checkpoints/ppo_real_franka_best_place_success.pt
```

Separate skill checkpoints can be selected when needed:

```bash
PLACE_CKPT=checkpoints/place.pt \
STACK_CKPT=checkpoints/stack.pt \
python scripts/evaluate_sequence.py
```

## Third-Party Assets

The Franka Panda assets originate from MuJoCo Menagerie. Their Apache-2.0
license is preserved in
`third_party/calvin_franka_scene/MENAGERIE_PANDA_LICENSE`.

## License

Project code is released under the MIT License. Third-party assets retain their
original license terms.
