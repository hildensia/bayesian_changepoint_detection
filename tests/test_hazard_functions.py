"""
Tests for the constant hazard function.

These pin the contract of ``constant_hazard``: the value ``1 / lam``, the
output shape, dtype and device for each accepted form of ``r``, and the
rejection of ``lam`` values for which ``1 / lam`` is not a probability.
"""

import math

import pytest
import torch

from bayesian_changepoint_detection.hazard_functions import constant_hazard

pytestmark = pytest.mark.behaviour


def test_int_r_gives_a_vector_of_that_length():
    hazard = constant_hazard(10.0, 7, device="cpu")
    assert hazard.shape == (7,)
    assert hazard.dtype == torch.float32
    assert hazard.device.type == "cpu"
    assert torch.equal(hazard, torch.full((7,), 0.1))


def test_tensor_r_keeps_its_shape_and_device():
    r = torch.arange(12).reshape(3, 4)
    hazard = constant_hazard(4.0, r)
    assert hazard.shape == (3, 4)
    assert hazard.dtype == torch.float32
    assert hazard.device == r.device
    assert torch.all(hazard == 0.25)


def test_sequence_r_is_converted_to_a_tensor():
    hazard = constant_hazard(5.0, [0, 1, 2], device="cpu")
    assert isinstance(hazard, torch.Tensor)
    assert hazard.shape == (3,)
    assert torch.allclose(hazard, torch.full((3,), 0.2))


def test_lam_one_is_a_change_at_every_step():
    assert torch.all(constant_hazard(1.0, 3, device="cpu") == 1.0)


def test_infinite_lam_is_hazard_zero():
    assert torch.all(constant_hazard(math.inf, 3, device="cpu") == 0.0)


@pytest.mark.parametrize("lam", [0.5, 0.999, 0.0, -1.0, -math.inf, math.nan])
def test_lam_below_one_or_nan_is_rejected(lam):
    # 0 < lam < 1 used to be accepted and gave a "probability" 1/lam > 1.
    with pytest.raises(ValueError, match="at least 1"):
        constant_hazard(lam, 3, device="cpu")
