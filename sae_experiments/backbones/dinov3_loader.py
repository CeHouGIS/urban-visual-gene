from __future__ import annotations

from pathlib import Path

import torch


def load_dinov3_model(model_name: str | None = None, checkpoint_path: str | None = None):
    """Load a DINOv3 model without baking model-specific constants into SAE code."""
    if checkpoint_path and not model_name:
        path = Path(checkpoint_path)
        if path.is_file():
            obj = torch.load(path, map_location="cpu")
            if hasattr(obj, "forward"):
                return obj
            raise TypeError(
                "checkpoint_path points to a state_dict-like object; pass a model_name so it can be loaded first"
            )
    if not model_name:
        raise ValueError("model_name or checkpoint_path is required")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)
    return model


def infer_dinov3_embed_dim(model) -> int:
    for obj in (model, getattr(model, "config", None)):
        if obj is None:
            continue
        for name in ("embed_dim", "hidden_size", "hidden_dim", "d_model"):
            val = getattr(obj, name, None)
            if val is not None:
                return int(val)
    raise AttributeError("could not infer DINOv3 embedding dimension from model/config")


def infer_dinov3_num_blocks(model) -> int:
    for obj in (model, getattr(model, "config", None)):
        if obj is None:
            continue
        for name in ("num_hidden_layers", "num_layers", "depth", "n_layers"):
            val = getattr(obj, name, None)
            if val is not None:
                return int(val)
    for name in ("blocks", "layers", "encoder.layer", "encoder.layers"):
        cur = model
        ok = True
        for part in name.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            try:
                return len(cur)
            except TypeError:
                pass
    raise AttributeError("could not infer DINOv3 transformer block count")
