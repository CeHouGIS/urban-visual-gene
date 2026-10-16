"""Discrete prototype inference must use an actual slot and remain trainable."""
import torch

from experiments.vitl30_relational_prototypes.code.model import LandscapeDictionary, loss_terms


def test_all_encoder_arms_have_discrete_usable_slots():
    torch.manual_seed(42)
    x = torch.rand(3, 4, 16)
    context = torch.rand(3, 4, 4)
    for mode in ('composition', 'context', 'relation'):
        model = LandscapeDictionary(families=4, slots=5, hidden=8, mode=mode)
        prediction, probabilities, winner = model(x, context)
        assert prediction.shape == x.shape
        assert winner.shape == (3, 5)
        assert torch.all(winner.sum(-1) == 1)
        assert torch.allclose(probabilities.sum(-1), torch.ones(3))
        objective, _ = loss_terms(prediction, x, probabilities)
        objective.backward()
        assert model.dictionary.grad is not None and torch.isfinite(model.dictionary.grad).all()
        assert model.readout[-1].weight.grad is not None
