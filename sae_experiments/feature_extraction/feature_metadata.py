from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class FeatureCacheMetadata:
    model_name: str
    checkpoint_identifier: str
    feature_layer: int
    token_type: str
    embed_dim: int
    image_input_size: int
    preprocessing: dict
    prefix_tokens_removed: int
    patch_grid_size: tuple[int, int] | None
    source_image_count: int
    extraction_date: str

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        checkpoint_identifier: str,
        feature_layer: int,
        token_type: str,
        embed_dim: int,
        image_input_size: int,
        preprocessing: dict,
        prefix_tokens_removed: int,
        patch_grid_size: tuple[int, int] | None,
        source_image_count: int,
    ):
        return cls(
            model_name=model_name,
            checkpoint_identifier=checkpoint_identifier,
            feature_layer=int(feature_layer),
            token_type=token_type,
            embed_dim=int(embed_dim),
            image_input_size=int(image_input_size),
            preprocessing=preprocessing,
            prefix_tokens_removed=int(prefix_tokens_removed),
            patch_grid_size=patch_grid_size,
            source_image_count=int(source_image_count),
            extraction_date=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

