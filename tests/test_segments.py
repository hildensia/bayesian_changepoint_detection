"""
Tests for ``segment_statistics`` (direction and size of each change, #42).

Means, standard deviations and Welch standard errors are compared with
NumPy computed directly on the slices; the rest pins the index conventions,
input checks and the detector-to-summary pipeline.
"""

from functools import partial

import numpy as np
import pytest
import torch

from bayesian_changepoint_detection import (
    SegmentStatistics,
    StudentT,
    constant_hazard,
    get_map_changepoints,
    online_changepoint_detection,
    segment_statistics,
)

pytestmark = pytest.mark.behavior


def steps(means, length=50, seed=0, dims=None):
    gen = torch.Generator().manual_seed(seed)
    shape = (length,) if dims is None else (length, dims)
    return torch.cat([torch.randn(shape, generator=gen) + m for m in means])


def test_matches_numpy_on_each_slice():
    data = steps([0.0, 3.0, -1.0])
    stats = segment_statistics(data, [50, 100])
    assert isinstance(stats, SegmentStatistics)
    assert stats.starts.tolist() == [0, 50, 100]
    assert stats.ends.tolist() == [50, 100, 150]
    assert stats.lengths.tolist() == [50, 50, 50]
    array = data.double().numpy()
    slices = [array[0:50], array[50:100], array[100:150]]
    np.testing.assert_allclose(stats.means.numpy(), [s.mean() for s in slices])
    np.testing.assert_allclose(stats.stds.numpy(), [s.std(ddof=1) for s in slices])
    diffs = np.diff([s.mean() for s in slices])
    se = [
        np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        for a, b in zip(slices, slices[1:])
    ]
    np.testing.assert_allclose(stats.mean_changes.numpy(), diffs)
    np.testing.assert_allclose(stats.z_scores.numpy(), diffs / np.array(se))


def test_direction_after_online_detection():
    data = steps([0.0, 3.0, 0.0], seed=1)
    R, _ = online_changepoint_detection(
        data,
        partial(constant_hazard, 250, device="cpu"),
        StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cpu"),
        device="cpu",
    )
    stats = segment_statistics(data, get_map_changepoints(R, min_separation=10))
    assert torch.sign(stats.mean_changes).tolist() == [1.0, -1.0]
    assert torch.all(stats.z_scores.abs() > 10)


def test_multivariate_data_gives_per_dimension_rows():
    data = steps([0.0, 2.0], dims=3, seed=2)
    stats = segment_statistics(data, torch.tensor([50]))
    assert stats.means.shape == (2, 3)
    assert stats.mean_changes.shape == (1, 3)
    assert torch.all(stats.mean_changes > 1.0)


def test_starts_are_sorted_and_zero_is_optional():
    data = steps([0.0, 1.0, 2.0])
    a = segment_statistics(data, [100, 50])
    b = segment_statistics(data, [0, 50, 100])
    assert a.starts.tolist() == b.starts.tolist() == [0, 50, 100]
    assert torch.equal(a.means, b.means)


def test_no_changepoints_is_one_segment():
    data = steps([1.0])
    stats = segment_statistics(data, [])
    assert stats.lengths.tolist() == [50]
    assert stats.mean_changes.numel() == 0


def test_single_point_segments():
    data = torch.tensor([1.0, 5.0, 5.0, 2.0])
    stats = segment_statistics(data, [1, 3])
    assert stats.stds.tolist() == [0.0, 0.0, 0.0]
    # zero spread on both sides: the standard error is 0, so no z-score
    assert torch.isnan(stats.z_scores).all()
    assert stats.mean_changes.tolist() == [4.0, -3.0]


@pytest.mark.parametrize("starts", [[-1], [10], [3, 3]])
def test_invalid_starts(starts):
    with pytest.raises(ValueError, match="starts"):
        segment_statistics(torch.zeros(10), starts)


def test_invalid_data():
    with pytest.raises(ValueError, match="shape"):
        segment_statistics(torch.zeros(2, 2, 2), [1])
