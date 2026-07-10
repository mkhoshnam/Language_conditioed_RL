BLOCK_NAMES = ("red_block", "blue_block", "green_block")
TARGET_NAMES = ("yellow_plate", "purple_plate", "cyan_bowl", "orange_plate")
SKILLS = ("place", "stack")

_PLACE_TASKS = tuple(
    (
        block,
        target,
        "place",
        f"put the {block.replace('_', ' ')} in the {target.replace('_', ' ')}",
    )
    for block in BLOCK_NAMES
    for target in TARGET_NAMES
)
_STACK_TASKS = tuple(
    (
        block,
        destination,
        "stack",
        f"stack the {block.replace('_', ' ')} on the {destination.replace('_', ' ')}",
    )
    for block in BLOCK_NAMES
    for destination in BLOCK_NAMES
    if destination != block
)

TASKS = _PLACE_TASKS + _STACK_TASKS
PLACE_TASK_INDICES = tuple(i for i, task in enumerate(TASKS) if task[2] == "place")
STACK_TASK_INDICES = tuple(i for i, task in enumerate(TASKS) if task[2] == "stack")
