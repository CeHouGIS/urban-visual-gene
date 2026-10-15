"""DINOv3 + SAE experiment components."""

from sae_experiments.paths import REPO_ROOT

# Keep existing CLI commands and in-flight workers valid after the reorganization.
__path__.append(str(REPO_ROOT / "experiments/vitl30_panorama_w4096_k32/code"))
