"""
Tests for the synthetic data generators in ``generate_data``.

The tests and examples lean on these series, so their contracts are pinned
here: shapes, dtypes, segment lengths, reproducibility under a seed, and the
per-segment statistics each docstring promises (checked on long segments,
with tolerances several standard errors wide).
"""

import pytest
import torch

from bayesian_changepoint_detection import generate_data as gd

pytestmark = pytest.mark.behavior


def segments(partition, data):
    """Split ``data`` into the segments described by ``partition``."""
    return torch.split(data, partition.tolist())


class TestNormalTimeSeries:
    def test_shapes_and_lengths(self):
        partition, data = gd.generate_normal_time_series(
            5, min_length=20, max_length=40, seed=0, device="cpu"
        )
        assert partition.shape == (5,)
        assert partition.dtype == torch.long
        assert torch.all((partition >= 20) & (partition <= 40))
        assert data.shape == (int(partition.sum()), 1)
        assert data.dtype == torch.float32
        assert data.device.type == "cpu"

    def test_seed_reproduces_and_distinguishes(self):
        first = gd.generate_normal_time_series(3, seed=1, device="cpu")
        again = gd.generate_normal_time_series(3, seed=1, device="cpu")
        other = gd.generate_normal_time_series(3, seed=2, device="cpu")
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
        assert not torch.equal(first[1][:50], other[1][:50])

    def test_seed_none_does_not_reseed(self):
        # With seed=None the draws come from the global generator's current
        # state, so two consecutive calls differ; a hidden reseed would make
        # them equal.
        first = gd.generate_normal_time_series(2, 5, 5, seed=None, device="cpu")
        second = gd.generate_normal_time_series(2, 5, 5, seed=None, device="cpu")
        assert not torch.equal(first[1], second[1])


class TestMultivariateNormalTimeSeries:
    def test_shapes_and_reproducibility(self):
        partition, data = gd.generate_multivariate_normal_time_series(
            4, dims=3, min_length=10, max_length=30, seed=3, device="cpu"
        )
        assert partition.shape == (4,)
        assert torch.all((partition >= 10) & (partition <= 30))
        assert data.shape == (int(partition.sum()), 3)
        _, again = gd.generate_multivariate_normal_time_series(
            4, dims=3, min_length=10, max_length=30, seed=3, device="cpu"
        )
        assert torch.equal(data, again)

    def test_old_name_is_the_same_function(self):
        new = gd.generate_multivariate_normal_time_series(
            2, 2, 5, 9, seed=4, device="cpu"
        )
        old = gd.generate_multinormal_time_series(2, 2, 5, 9, seed=4, device="cpu")
        assert torch.equal(new[0], old[0]) and torch.equal(new[1], old[1])


class TestCorrelationChangeExample:
    def test_segments_have_the_documented_correlations(self):
        # Sample correlation of n = 5000 draws has standard error about
        # (1 - rho^2) / sqrt(n) <= 0.015; the tolerance is 0.05.
        partition, data = gd.generate_correlation_change_example(
            min_length=5000, max_length=5000, seed=5, device="cpu"
        )
        assert partition.tolist() == [5000, 5000, 5000]
        assert data.shape == (15000, 2)
        for segment, rho in zip(segments(partition, data), [0.75, 0.0, -0.75]):
            assert torch.corrcoef(segment.T)[0, 1].item() == pytest.approx(
                rho, abs=0.05
            )
            assert torch.allclose(segment.mean(0), torch.zeros(2), atol=0.1)

    def test_old_name_is_the_same_function(self):
        new = gd.generate_correlation_change_example(10, 20, seed=6, device="cpu")
        old = gd.generate_xuan_motivating_example(10, 20, seed=6, device="cpu")
        assert torch.equal(new[0], old[0]) and torch.equal(new[1], old[1])


class TestMeanShiftExample:
    def test_means_alternate_between_zero_and_the_shift(self):
        # Standard error of a segment mean: 1 / sqrt(2000) ~= 0.022.
        partition, data = gd.generate_mean_shift_example(
            num_segments=4,
            segment_length=2000,
            shift_magnitude=3.0,
            seed=7,
            device="cpu",
        )
        assert partition.tolist() == [2000] * 4
        assert data.shape == (8000, 1)
        means = [segment.mean().item() for segment in segments(partition, data)]
        assert means == pytest.approx([0.0, 3.0, 0.0, 3.0], abs=0.1)


class TestVarianceChangeExample:
    def test_default_levels(self):
        # Relative standard error of a sample variance: sqrt(2 / n) = 0.02.
        partition, data = gd.generate_variance_change_example(
            segment_length=5000, seed=8, device="cpu"
        )
        assert partition.tolist() == [5000] * 3
        assert data.shape == (15000, 1)
        variances = [s.var().item() for s in segments(partition, data)]
        assert variances == pytest.approx([0.5, 2.0, 0.8], rel=0.1)

    def test_custom_levels_accept_a_list(self):
        partition, data = gd.generate_variance_change_example(
            num_segments=2,
            segment_length=5000,
            variance_levels=[4.0, 0.25],
            seed=9,
            device="cpu",
        )
        variances = [s.var().item() for s in segments(partition, data)]
        assert variances == pytest.approx([4.0, 0.25], rel=0.1)

    def test_levels_must_match_the_number_of_segments(self):
        with pytest.raises(ValueError, match="must match num_segments"):
            gd.generate_variance_change_example(
                num_segments=4, variance_levels=[1.0, 2.0], device="cpu"
            )
