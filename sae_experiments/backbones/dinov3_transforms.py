from __future__ import annotations


def build_dinov3_transform(config):
    """Build preprocessing from the selected DINOv3 checkpoint/processor.

    The returned callable delegates resize/crop/tensor conversion/normalization
    to the checkpoint's image processor instead of copying old DINOv2 constants.
    """
    model_name = getattr(config, "model_name", None)
    checkpoint_path = getattr(config, "checkpoint_path", None)
    if not model_name and not checkpoint_path:
        raise ValueError("DINOv3 transform requires config.model_name or config.checkpoint_path")
    from transformers import AutoImageProcessor

    source = model_name or checkpoint_path
    processor = AutoImageProcessor.from_pretrained(source)
    input_size = getattr(config, "input_size", None)

    def transform(images):
        kwargs = {"return_tensors": "pt"}
        if input_size:
            kwargs["size"] = {"height": int(input_size), "width": int(input_size)}
        return processor(images=images, **kwargs)["pixel_values"]

    transform.processor = processor
    return transform

