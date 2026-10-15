"""Compatibility entry points for the earlier global BatchTopK experiment."""
from sae_experiments.paths import REPO_ROOT

__path__.append(str(REPO_ROOT / "experiments/global307_baseline_w1024_k8/code"))
