# CALVIN-Style Franka Scene

This folder contains the MuJoCo XML and mesh assets used by the
language-conditioned pick-and-place environment.

The scene includes:

- Franka Panda robot
- tabletop workspace
- red, blue, and green blocks
- yellow, purple, and orange plates
- cyan bowl target
- fixed scene camera
- wrist camera

Run the project-level smoke test from the repository root:

```bash
python scripts/smoke_test.py
```

The Franka Panda mesh/XML assets are copied from MuJoCo Menagerie and keep
their original Apache-2.0 license in `MENAGERIE_PANDA_LICENSE`.
