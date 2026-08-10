"""Visual backbone adapters."""

from .base_backbone import BaseVisualBackbone
from .dinov3_feature_extractor import DINOv3Backbone, DINOv3FeatureExtractor

__all__ = ["BaseVisualBackbone", "DINOv3Backbone", "DINOv3FeatureExtractor"]

