import json
import os

from language_conditioned_rl.task_config import BLOCK_NAMES, SKILLS, TARGET_NAMES, TASKS


def task_to_index(block: str, destination: str, skill: str) -> int:
    for i, (b, d, s, _) in enumerate(TASKS):
        if b == block and d == destination and s == skill:
            return i
    raise ValueError(
        f"Invalid task: block={block}, destination={destination}, skill={skill}"
    )


def _matches(text, aliases_by_name):
    matches = []
    for name, aliases in aliases_by_name.items():
        for alias in aliases:
            start = text.find(alias)
            while start >= 0:
                matches.append((start, -len(alias), name))
                start = text.find(alias, start + 1)
    matches.sort()
    return matches


def _result(block, destination, skill, source):
    idx = task_to_index(block, destination, skill)
    return {
        "block": block,
        "destination": destination,
        "target": destination,
        "skill": skill,
        "destination_type": "block" if skill == "stack" else "target",
        "task_index": idx,
        "canonical_goal": TASKS[idx][3],
        "source": source,
    }


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

    block_matches = _matches(text, block_aliases)
    target_matches = _matches(text, target_aliases)
    mentioned_blocks = {name for _, _, name in block_matches}
    stack_requested = (
        "stack" in text
        or "on top" in text
        or (not target_matches and len(mentioned_blocks) >= 2)
    )
    skill = "stack" if stack_requested else "place"

    if not block_matches:
        raise ValueError(
            f"Could not parse command: {command!r}. "
            f"Valid blocks: {BLOCK_NAMES}, valid targets: {TARGET_NAMES}"
        )

    source_pos, _, block = block_matches[0]
    if skill == "stack":
        destination_matches = [
            (pos, length, name)
            for pos, length, name in block_matches
            if name != block and pos >= source_pos
        ]
        if not destination_matches:
            destination_matches = [
                (pos, length, name)
                for pos, length, name in block_matches
                if name != block
            ]
        if not destination_matches:
            raise ValueError(
                f"Could not parse stack destination from command: {command!r}. "
                f"Valid destination blocks: {BLOCK_NAMES}"
            )
        destination = destination_matches[0][2]
    else:
        if not target_matches:
            raise ValueError(
                f"Could not parse place target from command: {command!r}. "
                f"Valid targets: {TARGET_NAMES}"
            )
        destination = target_matches[0][2]

    return _result(block, destination, skill, "rule_fallback")


def parse_command(command: str):
    destinations = list(TARGET_NAMES) + list(BLOCK_NAMES)
    schema = {
        "type": "object",
        "properties": {
            "block": {
                "type": "string",
                "enum": list(BLOCK_NAMES),
            },
            "destination": {
                "type": "string",
                "enum": destinations,
            },
            "skill": {
                "type": "string",
                "enum": list(SKILLS),
            },
        },
        "required": ["block", "destination", "skill"],
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
                        "Parse the robot command into one source block, one destination, "
                        "and one skill. "
                        f"Valid blocks: {list(BLOCK_NAMES)}. "
                        f"Valid targets for place: {list(TARGET_NAMES)}. "
                        f"Valid destination blocks for stack: {list(BLOCK_NAMES)}. "
                        "Use skill='place' for putting a block in/on a plate or bowl. "
                        "Use skill='stack' for stacking one block on another block. "
                        "For stack, destination must be a different block."
                    ),
                },
                {"role": "user", "content": command},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "multi_skill_task",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        data = json.loads(response.output_text)
        block = data["block"]
        destination = data["destination"]
        skill = data["skill"]
        task_to_index(block, destination, skill)

        return _result(block, destination, skill, "llm")

    except Exception:
        return rule_fallback(command)


if __name__ == "__main__":
    import sys

    cmd = " ".join(sys.argv[1:])
    result = parse_command(cmd)
    print(json.dumps(result, indent=2))
