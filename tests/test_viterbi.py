"""
viterbi_changepoints: the most probable run-length path under the BOCPD
model, checked against an exhaustive search over segmentations
(``reference_map_segmentation``), against the forward pass on a clear
change, and for its run-length bookkeeping.
"""

from functools import partial

import pytest
import torch

from bayesian_changepoint_detection import (
    compute_run_length_posterior,
    constant_hazard,
    get_map_changepoints,
    offline_likelihoods,
    online_changepoint_detection,
    online_likelihoods,
    viterbi_changepoints,
)
from tests._reference_bocpd import reference_map_segmentation

HYPER = dict(alpha=0.1, beta=0.01, kappa=1.0, mu=0.0)


def _online():
    return online_likelihoods.StudentT(**HYPER, device="cpu")


def _segment_log_likelihoods(data):
    """Offline closed form of the same Normal-Gamma model (pinned elsewhere
    against scipy), so only the path search is under test here."""
    off = offline_likelihoods.StudentT(
        alpha0=HYPER["alpha"],
        beta0=HYPER["beta"],
        kappa0=HYPER["kappa"],
        mu0=HYPER["mu"],
        device="cpu",
    )
    x = data.double()
    off.setup(x)
    T = len(x)
    P = torch.full((T, T), float("-inf"), dtype=torch.float64)
    for t in range(T):
        P[t, t:] = off.pdf_rows(x, t)
    return P.numpy()


@pytest.mark.math
@pytest.mark.parametrize("seed", range(6))
def test_matches_exhaustive_map_segmentation(seed):
    torch.manual_seed(seed)
    if seed % 2 == 0:
        data = torch.cat([torch.randn(4), torch.randn(5) + 4])
    else:
        data = torch.cat([torch.randn(3), torch.randn(3) + 5, torch.randn(3) - 5])
    lam = 5.0
    _, expected_starts = reference_map_segmentation(
        _segment_log_likelihoods(data), len(data), 1 / lam
    )

    path, changepoints = viterbi_changepoints(
        data, partial(constant_hazard, lam, device="cpu"), _online(), device="cpu"
    )
    assert changepoints.tolist() == expected_starts


@pytest.mark.behaviour
def test_path_bookkeeping():
    torch.manual_seed(1)
    data = torch.cat([torch.randn(30), torch.randn(30) + 4, torch.randn(30)])
    path, changepoints = viterbi_changepoints(
        data, partial(constant_hazard, 50, device="cpu"), _online(), device="cpu"
    )
    assert path.shape == (len(data) + 1,) and path.dtype == torch.long
    assert path[0] == 0
    steps = path[1:] - path[:-1]
    # every step grows the run by one or resets it to zero
    assert torch.all((steps == 1) | (path[1:] == 0))
    assert changepoints.dtype == torch.long
    assert changepoints.tolist() == (torch.where(path[1:] == 0)[0] + 1).tolist()
    assert len(changepoints) == 2
    assert abs(int(changepoints[0]) - 30) <= 2 and abs(int(changepoints[1]) - 60) <= 2


@pytest.mark.behaviour
def test_agrees_with_forward_pass_on_a_clear_change():
    torch.manual_seed(0)
    data = torch.cat([torch.randn(80), torch.randn(80) + 5])
    hazard = partial(constant_hazard, 100, device="cpu")
    _, changepoints = viterbi_changepoints(data, hazard, _online(), device="cpu")
    R, _ = online_changepoint_detection(data, hazard, _online(), device="cpu")
    assert changepoints.tolist() == [80]
    assert changepoints.tolist() == get_map_changepoints(R).tolist()


@pytest.mark.behaviour
def test_no_change_series_gives_no_changepoints():
    torch.manual_seed(3)
    data = torch.randn(120)
    path, changepoints = viterbi_changepoints(
        data, partial(constant_hazard, 100, device="cpu"), _online(), device="cpu"
    )
    assert changepoints.numel() == 0
    assert path[-1] == len(data)


@pytest.mark.behaviour
def test_multivariate():
    torch.manual_seed(1)
    data = torch.cat([torch.randn(40, 3), torch.randn(40, 3) + 3])
    _, changepoints = viterbi_changepoints(
        data,
        partial(constant_hazard, 50, device="cpu"),
        online_likelihoods.MultivariateT(dims=3, device="cpu"),
        device="cpu",
    )
    assert len(changepoints) == 1 and abs(int(changepoints[0]) - 40) <= 2


@pytest.mark.behaviour
def test_input_validation():
    hazard = partial(constant_hazard, 50, device="cpu")
    with pytest.raises(ValueError, match="at least one observation"):
        viterbi_changepoints(torch.zeros(0), hazard, _online(), device="cpu")
    with pytest.raises(ValueError, match="NaN or Inf"):
        viterbi_changepoints(
            torch.tensor([1.0, float("nan")]), hazard, _online(), device="cpu"
        )


@pytest.mark.behaviour
def test_compute_run_length_posterior_is_the_forward_pass():
    torch.manual_seed(2)
    data = torch.cat([torch.randn(20), torch.randn(20) + 3])
    hazard = partial(constant_hazard, 30, device="cpu")
    R_direct, _ = online_changepoint_detection(data, hazard, _online(), device="cpu")
    R = compute_run_length_posterior(data, hazard, _online(), device="cpu")
    assert torch.equal(R, R_direct)


@pytest.mark.behaviour
def test_model_built_on_another_device_is_moved(monkeypatch):
    """viterbi_changepoints must move the model's prior tensors like the
    forward pass does, not only relabel its device attribute."""
    moved = {}
    model = _online()
    original_to = model.to

    def spy(device):
        moved["device"] = torch.device(device)
        return original_to(device)

    monkeypatch.setattr(model, "to", spy)
    torch.manual_seed(0)
    viterbi_changepoints(
        torch.randn(10), partial(constant_hazard, 20, device="cpu"), model, device="cpu"
    )
    assert moved["device"].type == "cpu"
