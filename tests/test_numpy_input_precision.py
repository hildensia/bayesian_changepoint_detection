"""
NumPy float64 input far from zero reaches the detectors at full precision.

Up to 1.1.0 ``ensure_tensor`` turned every non-tensor into float32, so a
float64 NumPy series at 1e8 was quantized to steps of 8 before the
(float64, centered) offline statistics or the (centered) online state ever
saw it. The detectors must now give the same result for a float64 NumPy
array as for the equivalent float64 tensor.
"""

from functools import partial

import numpy as np
import pytest
import torch

from bayesian_changepoint_detection import (
    StudentT,
    const_prior,
    constant_hazard,
    offline_changepoint_detection,
    offline_likelihoods,
    online_changepoint_detection,
)

pytestmark = pytest.mark.behavior


def series(offset=1e8, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(offset, 1, 40), rng.normal(offset + 3, 1, 40)])


def test_offline_numpy_equals_float64_tensor():
    x = series()
    prior = partial(const_prior, p=1 / 81)
    run = partial(
        offline_changepoint_detection,
        prior_function=prior,
        likelihood_model=offline_likelihoods.StudentT(device="cpu", mu0=1e8),
        device="cpu",
    )
    Q_np, _, P_np = run(x)
    Q_t, _, P_t = run(torch.tensor(x, dtype=torch.float64))
    assert torch.equal(Q_np, Q_t)
    assert torch.equal(P_np, P_t)
    found = torch.where(torch.exp(P_np).sum(0) > 0.5)[0].tolist()
    assert len(found) == 1 and abs(found[0] - 39) <= 3


def test_online_numpy_equals_float64_tensor():
    x = series()
    hazard = partial(constant_hazard, 100, device="cpu")
    R_np, _ = online_changepoint_detection(
        x, hazard, StudentT(alpha=0.1, beta=0.01, mu=1e8, device="cpu"), device="cpu"
    )
    R_t, _ = online_changepoint_detection(
        torch.tensor(x, dtype=torch.float64),
        hazard,
        StudentT(alpha=0.1, beta=0.01, mu=1e8, device="cpu"),
        device="cpu",
    )
    assert torch.equal(R_np, R_t)
