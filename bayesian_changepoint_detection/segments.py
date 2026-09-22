"""
Describe the segments between detected changepoints.

The detectors say *where* a series changes; ``segment_statistics`` says
*how*: the mean and spread of every segment, and for each changepoint the
direction and size of the change in mean (issue #42).
"""

from collections.abc import Sequence
from typing import NamedTuple, Union

import torch

from .device import ensure_tensor


class SegmentStatistics(NamedTuple):
    """
    Per-segment summaries returned by ``segment_statistics``.

    All fields are CPU tensors; the statistics are float64. For ``K``
    changepoints there are ``K + 1`` segments. ``means`` and
    ``stds`` have shape ``[K + 1]`` for univariate data and ``[K + 1, D]``
    for ``[T, D]`` data; the change fields have ``K`` rows.

    Attributes
    ----------
    starts, ends : torch.Tensor
        First index and one-past-last index of each segment (``long``).
    lengths : torch.Tensor
        Number of observations in each segment (``long``).
    means : torch.Tensor
        Sample mean of each segment.
    stds : torch.Tensor
        Sample standard deviation of each segment (``n - 1`` in the
        denominator; 0 for a segment of one observation).
    mean_changes : torch.Tensor
        ``means[k + 1] - means[k]``: positive when the series moves up at
        changepoint ``k``.
    z_scores : torch.Tensor
        ``mean_changes`` divided by its standard error
        ``sqrt(std_k^2 / n_k + std_{k+1}^2 / n_{k+1})`` (Welch). A rough
        guide to how clearly the mean moved; it ignores that the
        changepoint was chosen from the same data, so it overstates
        significance. ``nan`` when both segments have zero spread.
    """

    starts: torch.Tensor
    ends: torch.Tensor
    lengths: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor
    mean_changes: torch.Tensor
    z_scores: torch.Tensor


def segment_statistics(
    data: Union[torch.Tensor, Sequence[float]],
    starts: Union[torch.Tensor, Sequence[int]],
) -> SegmentStatistics:
    """
    Mean, spread and direction of change for the segments between changepoints.

    Parameters
    ----------
    data : torch.Tensor or array-like
        The series, ``[T]`` or ``[T, D]``.
    starts : torch.Tensor or sequence of int
        Index of the first observation of each new segment, as returned by
        ``get_map_changepoints``. Index 0 may be included or not. The
        offline detector reports the *last* index of the old segment
        instead, so add 1 to its positions:
        ``segment_statistics(data, torch.where(probs > 0.5)[0] + 1)``.

    Returns
    -------
    SegmentStatistics
        See the class for the fields.

    Raises
    ------
    ValueError
        If ``data`` is not ``[T]`` or ``[T, D]`` or has NaN or Inf, or a
        start is outside ``[0, T)`` or repeated.

    Examples
    --------
    >>> stats = segment_statistics(data, get_map_changepoints(R))
    >>> stats.mean_changes  # tensor([ 3.1, -2.9]): up at the first change, down at the second
    """
    # A handful of means: computed on the CPU in float64 (MPS has no float64).
    data = ensure_tensor(data, device="cpu")
    if data.dim() not in (1, 2) or data.shape[0] == 0:
        raise ValueError(
            f"data must have shape [T] or [T, D] with T >= 1, got {list(data.shape)}"
        )
    if not bool(torch.isfinite(data).all()):
        raise ValueError("data contains NaN or Inf; remove or impute them first")
    T = data.shape[0]
    starts = ensure_tensor(starts, device="cpu").to(torch.long).reshape(-1)
    if starts.numel() and (bool((starts < 0).any()) or bool((starts >= T).any())):
        raise ValueError(f"segment starts must be in [0, {T}), got {starts.tolist()}")
    if torch.unique(starts).numel() != starts.numel():
        raise ValueError(f"segment starts must be distinct, got {starts.tolist()}")
    starts = torch.sort(starts).values
    if starts.numel() == 0 or int(starts[0]) != 0:
        starts = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=data.device), starts]
        )
    ends = torch.cat([starts[1:], torch.tensor([T], device=data.device)])
    lengths = ends - starts

    values = data.to(torch.float64)
    means, stds = [], []
    for start, end in zip(starts.tolist(), ends.tolist()):
        segment = values[start:end]
        means.append(segment.mean(dim=0))
        if end - start > 1:
            stds.append(segment.std(dim=0, unbiased=True))
        else:
            stds.append(torch.zeros_like(segment[0]))
    means = torch.stack(means)
    stds = torch.stack(stds)

    mean_changes = means[1:] - means[:-1]
    n = lengths.to(torch.float64)
    if data.dim() == 2:
        n = n.unsqueeze(-1)
    standard_error = torch.sqrt(stds[:-1] ** 2 / n[:-1] + stds[1:] ** 2 / n[1:])
    z_scores = torch.where(
        standard_error > 0,
        mean_changes / standard_error,
        torch.full_like(mean_changes, float("nan")),
    )
    return SegmentStatistics(starts, ends, lengths, means, stds, mean_changes, z_scores)
