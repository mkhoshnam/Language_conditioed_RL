from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import torch

from language_conditioned_rl.task_config import TASKS


PLACE_TEMPLATES = (
    "put the {source} in the {destination}",
    "place the {source} inside the {destination}",
    "move the {source} into the {destination}",
    "please put the {source} in the {destination}",
    "can you place the {source} inside the {destination}",
)
STACK_TEMPLATES = (
    "stack the {source} on the {destination}",
    "put the {source} on top of the {destination}",
    "place the {source} above the {destination}",
    "please stack the {source} onto the {destination}",
    "can you put the {source} on the {destination}",
)


class CommandSampler:
    """Produces language variation while preserving the privileged reward task."""

    def __init__(self, paraphrase_probability: float = 0.35, seed: int = 7):
        self.paraphrase_probability = float(paraphrase_probability)
        self.rng = np.random.default_rng(seed)

    def sample(self, task_index: int) -> str:
        source, destination, skill, canonical = TASKS[int(task_index)]
        if self.rng.random() >= self.paraphrase_probability:
            return canonical
        templates = STACK_TEMPLATES if skill == "stack" else PLACE_TEMPLATES
        template = templates[int(self.rng.integers(0, len(templates)))]
        return template.format(
            source=source.replace("_", " "),
            destination=destination.replace("_", " "),
        )


@dataclass
class LanguageBatch:
    embeddings: torch.Tensor
    padding_mask: torch.Tensor


class FrozenTextEncoder:
    """Frozen pretrained token encoder with an explicit offline test backend.

    The default backend is a Hugging Face pretrained encoder. ``hash`` is a
    deterministic, frozen word-token embedding used only for CI and smoke tests;
    it keeps those checks independent of network/model-cache availability.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device | str,
        max_tokens: int = 32,
        backend: str = "pretrained",
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_tokens = int(max_tokens)
        self.backend = backend
        self._cache: dict[str, torch.Tensor] = {}

        if backend == "pretrained":
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "The pretrained language backend requires `transformers`. "
                    "Install the project with `pip install -e .`."
                ) from exc
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
            self.hidden_size = int(self.model.config.hidden_size)
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        elif backend == "hash":
            self.tokenizer = None
            self.model = None
            self.hidden_size = 128
            generator = torch.Generator(device="cpu").manual_seed(1729)
            self._hash_table = torch.randn(4096, self.hidden_size, generator=generator) * 0.08
        else:
            raise ValueError(f"unknown text backend: {backend}")

    @staticmethod
    def _wordpieces(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())

    def _encode_hash(self, text: str) -> torch.Tensor:
        pieces = ["[CLS]", *self._wordpieces(text), "[SEP]"][: self.max_tokens]
        indices = []
        for piece in pieces:
            digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
            indices.append(int.from_bytes(digest, "little") % self._hash_table.shape[0])
        return self._hash_table[torch.tensor(indices, dtype=torch.long)].clone()

    def _encode_pretrained(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded).last_hidden_state[0]
        length = int(encoded["attention_mask"][0].sum().item())
        return output[:length].detach().cpu()

    def encode(self, commands: list[str] | tuple[str, ...]) -> LanguageBatch:
        encoded = []
        for command in commands:
            if command not in self._cache:
                value = (
                    self._encode_pretrained(command)
                    if self.backend == "pretrained"
                    else self._encode_hash(command)
                )
                self._cache[command] = value
            encoded.append(self._cache[command])

        max_length = max(item.shape[0] for item in encoded)
        batch = torch.zeros(
            len(encoded), max_length, self.hidden_size, dtype=torch.float32, device=self.device
        )
        padding_mask = torch.ones(
            len(encoded), max_length, dtype=torch.bool, device=self.device
        )
        for index, item in enumerate(encoded):
            length = item.shape[0]
            batch[index, :length] = item.to(self.device)
            padding_mask[index, :length] = False
        return LanguageBatch(batch, padding_mask)
