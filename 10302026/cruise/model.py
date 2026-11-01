"""Paper ART: modality encoders, causal attention, action and next-state heads."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 6
    heads: int = 6
    embedding: int = 384
    context_length: int = 100
    max_steps: int = 100
    dropout: float = .1
    task_conditioning: bool = True
    max_zones: int = 3


class ART(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        d = config.embedding
        self.encoders = nn.ModuleList([nn.Linear(1, d), nn.Linear(1, d), nn.Linear(7, d), nn.Linear(3, d)])
        self.timestep = nn.Embedding(config.max_steps, d)
        self.task = nn.Linear(6+4*config.max_zones, d) if config.task_conditioning else None
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, config.heads, 4*d, config.dropout,
                                           activation="relu", batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.action_head, self.state_head = nn.Linear(d, 3), nn.Linear(d, 7)
        # Independent initialization for cloned TransformerEncoder layers.
        self.apply(self._initialize)
        for block in self.backbone.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)

    def forward(self, states, actions, rtgs, ctgs, timesteps, context=None):
        b, k, _ = states.shape
        if k > self.config.context_length: raise ValueError("Sequence exceeds the model context length")
        time = self.timestep(timesteps)
        tokens = torch.stack([encoder(value)+time for encoder, value in
                              zip(self.encoders, (rtgs, ctgs, states, actions))], dim=2).reshape(b, 4*k, -1)
        prefix = 0
        if self.task is not None:
            tokens = torch.cat([self.task(context).unsqueeze(1), tokens], dim=1)
            prefix = 1
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool), diagonal=1)
        hidden = self.backbone(self.norm(tokens), mask=mask)[:, prefix:]
        return self.action_head(hidden[:, 2::4]), self.state_head(hidden[:, 3::4])
