import numpy as np
import pytest
import torch
from experiments.vitl30_open_panorama_prototypes.code.data import aggregate_pano, join_headings
from experiments.vitl30_open_panorama_prototypes.code.model import RingDictionary, rotated_bank, sparse_code

def test_heading_rows_join_horizontally():
    x = np.arange(4 * 3 * 3).reshape(1, 4, 3, 3)
    y = join_headings(x)
    for h in range(4):
        np.testing.assert_array_equal(y[..., h*3:(h+1)*3], x[..., h, :, :])

def test_csr_heading_and_patch_coordinates():
    # A marker at the lower-right patch of each view must stay inside that view.
    rows = np.array([h * 784 + 783 for h in range(4)])
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=3136))]
    groups = np.arange(4096) % 64
    output = aggregate_pano(np.array([1,2,3,4]), np.array([0,0,0,0]), ptr, np.ones(4096), groups)
    np.testing.assert_allclose(output[0, 13, [13,27,41,55]], np.arange(1,5) / 32)
    assert np.count_nonzero(output) == 4

def test_ring_wrap_and_nonnegative_reconstruction():
    templates = torch.zeros(2, 2, 2, 8)
    templates[0, 0, 0, 7] = 1
    templates[1, 1, 1, 2] = 1
    bank = rotated_bank(templates)
    assert bank[1].reshape(2,2,8)[0,0,1] == 1
    x = (bank[1] + bank[4]).reshape(1,2,2,8)
    model = RingDictionary(templates)
    reconstruction, code = model(x)
    assert (model.templates >= 0).all() and (code >= 0).all()
    assert (x - reconstruction).square().sum() < .001
    assert code.reshape(1,2,4).sum(2).gt(0).sum() == 2
