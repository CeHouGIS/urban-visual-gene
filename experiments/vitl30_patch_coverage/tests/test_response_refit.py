import torch

from experiments.vitl30_multiscale_prototypes.code.model import PrototypeAutoencoder
from experiments.vitl30_patch_coverage.code.response_refit import measure, prefer, refit


def test_refit_keeps_dictionary_frozen_and_best_hard_coverage():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = PrototypeAutoencoder(dimensions=4, scales=(3,), per_scale=2, rank=1, hidden=4)
    before = {name: parameter.clone() for name, parameter in model.state_dict().items()}
    target = torch.rand(1, 4, 5, 5)
    with torch.no_grad():
        prediction, maps = model(target)
    baseline = measure(prediction, target)
    result = refit(model, target, maps, arm='coverage', steps=10, lr=.1)
    assert result['best']['covered_patch_count'] >= baseline['covered_patch_count']
    assert result['best']['valid_patch_count'] == baseline['valid_patch_count']
    assert result['best_step'] in (0, 10)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, before[name])


def test_prefer_uses_coverage_before_reconstruction_error():
    old = {'covered_patch_count': 5, 'image_nmse': .1}
    assert prefer({'covered_patch_count': 6, 'image_nmse': .9}, old)
    assert not prefer({'covered_patch_count': 4, 'image_nmse': .01}, old)
