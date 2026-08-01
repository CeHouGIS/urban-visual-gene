from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseVisualBackbone(nn.Module, ABC):
    """Common interface for frozen visual feature backbones."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def extract_features(self, images, layer_index: int, token_type: str):
        raise NotImplementedError

