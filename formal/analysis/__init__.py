"""Compatibility imports for analysis code now grouped by experiment."""
from sae_experiments.paths import REPO_ROOT

__path__.append(str(REPO_ROOT / "sae_experiments/semantic_tools"))
for _version in (
    "global307_dictionary_scaling_k8", "global307_w4096_k_sweep",
    "global307_w2048_k64", "global307_w4096_k64",
):
    __path__.append(str(REPO_ROOT / "experiments" / _version / "code"))
