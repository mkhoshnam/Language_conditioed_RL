import json
import os

from language_conditioned_rl.env import BLOCK_NAMES, TARGET_NAMES, TASKS


def task_to_index(block: str, target: str) -> int:
    for i, (b, t, _) in enumerate(TASKS):
        if b == block and t == target:
            return i
    raise ValueError(f"Invalid task: block={block}, target={target}")


def rule_fallback(command: str):
    text = command.lower().replace("-", " ").replace("_", " ")

    block_aliases = {
        "red_block": ["red block", "red cube", "red object", "red"],
        "blue_block": ["blue block", "blue cube", "blue object", "blue"],
        "green_block": ["green block", "green cube", "green object", "green"],
    }

    target_aliases = {
        "yellow_plate": ["yellow plate", "yellow dish", "yellow"],
        "purple_plate": ["purple plate", "purple dish", "purple"],
        "cyan_bowl": ["cyan bowl", "blue bowl", "bowl", "cyan"],
        "orange_plate": ["orange plate", "orange dish", "orange"],
    }

    block = None
    target = None

    for name, aliases in block_aliases.items():
        if any(alias in text for alias in aliases):
            block = name
            break

    for name, aliases in target_aliases.items():
        if any(alias in text for alias in aliases):
            target = name
            break

    if block is None or target is None:
        raise ValueError(
            f"Could not parse command: {command!r}. "
            f"Valid blocks: {BLOCK_NAMES}, valid targets: {TARGET_NAMES}"
        )

    idx = task_to_index(block, target)
    return {
        "block": block,
        "target": target,
        "task_index": idx,
        "canonical_goal": TASKS[idx][2],
        "source": "rule_fallback",
    }


def parse_command(command: str):
    schema = {
        "type": "object",
        "properties": {
            "block": {
                "type": "string",
                "enum": list(BLOCK_NAMES),
            },
            "target": {
                "type": "string",
                "enum": list(TARGET_NAMES),
            },
        },
        "required": ["block", "target"],
        "additionalProperties": False,
    }

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "Parse the robot command into exactly one block and one target. "
                        f"Valid blocks: {list(BLOCK_NAMES)}. "
                        f"Valid targets: {list(TARGET_NAMES)}. "
                        "Return only the closest valid pair."
                    ),
                },
                {"role": "user", "content": command},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pick_place_task",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        data = json.loads(response.output_text)
        block = data["block"]
        target = data["target"]
        idx = task_to_index(block, target)

        return {
            "block": block,
            "target": target,
            "task_index": idx,
            "canonical_goal": TASKS[idx][2],
            "source": "llm",
        }

    except Exception:
        return rule_fallback(command)


if __name__ == "__main__":
    import sys

    cmd = " ".join(sys.argv[1:])
    result = parse_command(cmd)
    print(json.dumps(result, indent=2))
