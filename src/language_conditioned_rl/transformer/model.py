from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from language_conditioned_rl.transformer.config import ModelConfig
from language_conditioned_rl.transformer.language import LanguageBatch
from language_conditioned_rl.transformer.observation import STATE_DIMS


N_STATE_TOKENS = 10  # robot + end effector + gripper + 3 blocks + 4 targets


def _token_mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.GELU(),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, output_dim),
    )


class StateTokenizer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        hidden = config.state_mlp_hidden
        self.robot = _token_mlp(STATE_DIMS["robot"], hidden, d_model)
        self.end_effector = _token_mlp(STATE_DIMS["end_effector"], hidden, d_model)
        self.gripper = _token_mlp(STATE_DIMS["gripper"], hidden, d_model)
        self.block = _token_mlp(STATE_DIMS["blocks"], hidden, d_model)
        self.target = _token_mlp(STATE_DIMS["targets"], hidden, d_model)
        self.type_embedding = nn.Embedding(5, d_model)
        self.slot_embedding = nn.Embedding(N_STATE_TOKENS, d_model)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.cat(
            [
                self.robot(state["robot"]).unsqueeze(1),
                self.end_effector(state["end_effector"]).unsqueeze(1),
                self.gripper(state["gripper"]).unsqueeze(1),
                self.block(state["blocks"]),
                self.target(state["targets"]),
            ],
            dim=1,
        )
        type_ids = torch.tensor(
            [0, 1, 2, 3, 3, 3, 4, 4, 4, 4], dtype=torch.long, device=tokens.device
        )
        slot_ids = torch.arange(N_STATE_TOKENS, device=tokens.device)
        return tokens + self.type_embedding(type_ids) + self.slot_embedding(slot_ids)


class TransformerActorCritic(nn.Module):
    """Shared multimodal transformer with independent ACT and VALUE readouts."""

    def __init__(self, config: ModelConfig, text_hidden_size: int, action_dim: int = 7):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        d_model = config.d_model

        self.language_projection = nn.Sequential(
            nn.Linear(text_hidden_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.state_tokenizer = StateTokenizer(config)
        self.language_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.readout_tokens = nn.Parameter(torch.empty(1, 2, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 2 + config.max_language_tokens + N_STATE_TOKENS, d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.actor = nn.Sequential(
            nn.Linear(d_model, config.actor_hidden),
            nn.GELU(),
            nn.Linear(config.actor_hidden, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(d_model, config.actor_hidden),
            nn.GELU(),
            nn.Linear(config.actor_hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.readout_tokens, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)
        nn.init.normal_(self.language_type, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(
        self,
        state: dict[str, torch.Tensor],
        language: LanguageBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, language_length = language.embeddings.shape[:2]
        if language_length > self.config.max_language_tokens:
            raise ValueError("language batch exceeds configured maximum length")
        language_tokens = self.language_projection(language.embeddings) + self.language_type
        state_tokens = self.state_tokenizer(state)
        readout = self.readout_tokens.expand(batch_size, -1, -1)
        sequence = torch.cat([readout, language_tokens, state_tokens], dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.shape[1]]

        prefix_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=sequence.device)
        state_mask = torch.zeros(
            batch_size, N_STATE_TOKENS, dtype=torch.bool, device=sequence.device
        )
        padding_mask = torch.cat([prefix_mask, language.padding_mask, state_mask], dim=1)
        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)

        mean = self.actor(encoded[:, 0])
        value = self.value(encoded[:, 1]).squeeze(-1)
        log_std = self.log_std.clamp(
            self.config.min_log_std, self.config.max_log_std
        ).expand_as(mean)
        return mean, log_std.exp(), value

    @staticmethod
    def _log_prob(distribution: Normal, raw_action: torch.Tensor, action: torch.Tensor):
        correction = torch.log(1.0 - action.square() + 1.0e-6)
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    def act(self, state: dict[str, torch.Tensor], language: LanguageBatch):
        mean, std, value = self(state, language)
        distribution = Normal(mean, std)
        raw_action = distribution.sample()
        action = torch.tanh(raw_action)
        return action, self._log_prob(distribution, raw_action, action), value

    def deterministic_act(self, state: dict[str, torch.Tensor], language: LanguageBatch):
        mean, _, value = self(state, language)
        return torch.tanh(mean), value

    def evaluate_actions(
        self,
        state: dict[str, torch.Tensor],
        language: LanguageBatch,
        action: torch.Tensor,
    ):
        mean, std, value = self(state, language)
        distribution = Normal(mean, std)
        safe_action = action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        raw_action = torch.atanh(safe_action)
        log_probability = self._log_prob(distribution, raw_action, safe_action)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probability, value, entropy

    def parameter_count(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return trainable, total
