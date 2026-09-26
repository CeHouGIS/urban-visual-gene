"""Minimal DINOv3 ViT inference backport for the project's Python 3.8 runtime.

The local checkpoint declares the Hugging Face ``dinov3_vit`` architecture,
which was added after the newest Transformers release that still supports
Python 3.8.  This module implements only the frozen inference path used by the
existing patch-token extractor, with parameter names matching the safetensors
checkpoint exactly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from torch import nn


class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(
            torch.empty(1, config.num_register_tokens, config.hidden_size)
        )
        self.patch_embeddings = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embeddings(pixel_values)
        patches = patches.flatten(2).transpose(1, 2)
        batch = len(pixel_values)
        return torch.cat(
            (
                self.cls_token.expand(batch, -1, -1),
                self.register_tokens.expand(batch, -1, -1),
                patches,
            ),
            dim=1,
        )


def patch_rope(
    pixel_values: torch.Tensor, config: SimpleNamespace
) -> tuple[torch.Tensor, torch.Tensor]:
    height = pixel_values.shape[-2] // config.patch_size
    width = pixel_values.shape[-1] // config.patch_size
    with torch.autocast(device_type="cuda", enabled=False):
        coords_h = torch.arange(0.5, height, dtype=torch.float32, device=pixel_values.device)
        coords_w = torch.arange(0.5, width, dtype=torch.float32, device=pixel_values.device)
        coords_h = 2.0 * coords_h / height - 1.0
        coords_w = 2.0 * coords_w / width - 1.0
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / config.rope_theta ** torch.arange(
            0, 1, 4 / head_dim, dtype=torch.float32, device=pixel_values.device
        )
        angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.cos(angles), torch.sin(angles)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_count = sin.shape[-2]
    prefix_count = query.shape[-2] - patch_count
    q_prefix, q_patches = query.split((prefix_count, patch_count), dim=-2)
    k_prefix, k_patches = key.split((prefix_count, patch_count), dim=-2)
    q_patches = q_patches * cos + rotate_half(q_patches) * sin
    k_patches = k_patches * cos + rotate_half(k_patches) * sin
    return (
        torch.cat((q_prefix, q_patches), dim=-2),
        torch.cat((k_prefix, k_patches), dim=-2),
    )


class DINOv3ViTAttention(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        dimension = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dimension // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(dimension, dimension, bias=config.query_bias)
        self.k_proj = nn.Linear(dimension, dimension, bias=config.key_bias)
        self.v_proj = nn.Linear(dimension, dimension, bias=config.value_bias)
        self.o_proj = nn.Linear(dimension, dimension, bias=config.proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = apply_rope(query, key, *position_embeddings)
        # This is the default backend selected by modern Transformers for the
        # checkpoint and matches the patch-token cache extraction path.
        output = nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, scale=self.scaling
        )
        output = output.transpose(1, 2).contiguous()
        return self.o_proj(output.reshape(batch, tokens, -1))


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.gelu(self.up_proj(hidden_states)))


class DINOv3ViTLayer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv3ViTMLP(config)
        self.layer_scale2 = DINOv3ViTLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.layer_scale1(
            self.attention(self.norm1(hidden_states), position_embeddings)
        )
        hidden_states = hidden_states + self.layer_scale2(
            self.mlp(self.norm2(hidden_states))
        )
        return hidden_states


class DINOv3ViTModel(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        if config.use_gated_mlp:
            raise NotImplementedError("the local ViT-B checkpoint does not use a gated MLP")
        if config.drop_path_rate != 0:
            raise NotImplementedError("drop path is not needed for the local inference checkpoint")
        self.config = config
        self.embeddings = DINOv3ViTEmbeddings(config)
        self.layer = nn.ModuleList(
            [DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = patch_rope(pixel_values, self.config)
        for layer in self.layer:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states[:, 0],
        )


def load_dinov3_vit(model_dir: Path) -> DINOv3ViTModel:
    config_dict = json.loads((model_dir / "config.json").read_text())
    config = SimpleNamespace(**config_dict)
    model = DINOv3ViTModel(config)
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"DINOv3 checkpoint contract mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model
