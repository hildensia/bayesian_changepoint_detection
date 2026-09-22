"""
Benchmark data and scoring, in NumPy only, so every version is fed and judged
by the same code.
"""

import numpy as np

# Four equal segments; the shifts are about one to three noise standard
# deviations, so detection is not trivial but every version should manage it.
SEGMENT_MEANS = (0.0, 2.0, -1.0, 1.5)
MARGIN = 5  # a detection within this many observations of a true change counts


def make_data(n_obs, dims=1, seed=0, segment_length=None):
    """Piecewise-constant mean, unit Gaussian noise, float64.

    Returns ``(data, truth)``: ``data`` is ``[n_obs]`` (``dims == 1``) or
    ``[n_obs, dims]``, and ``truth`` the data indices where segments start
    (after the first). By default there are four equal segments; with
    ``segment_length`` there are ``n_obs // segment_length`` segments whose
    means cycle through ``SEGMENT_MEANS``.
    In more than one dimension each segment's mean vector is its scalar mean
    times a fixed random unit direction, scaled so that the shift has the same
    Mahalanobis size as in one dimension.
    """
    rng = np.random.default_rng(seed)
    count = n_obs // segment_length if segment_length else len(SEGMENT_MEANS)
    levels = [SEGMENT_MEANS[i % len(SEGMENT_MEANS)] for i in range(count)]
    sizes = [n_obs // count] * (count - 1)
    sizes.append(n_obs - sum(sizes))
    if dims == 1:
        means = [np.full(size, mean) for size, mean in zip(sizes, levels)]
        data = np.concatenate(means) + rng.standard_normal(n_obs)
    else:
        direction = rng.standard_normal(dims)
        direction /= np.linalg.norm(direction)
        means = [
            np.tile(mean * direction, (size, 1)) for size, mean in zip(sizes, levels)
        ]
        data = np.concatenate(means) + rng.standard_normal((n_obs, dims))
    truth = np.cumsum(sizes)[:-1].tolist()
    return data, [int(t) for t in truth]


def offline_starts(changepoint_probs, threshold=0.5):
    """Segment starts from the offline marginal changepoint probabilities.

    ``changepoint_probs[i]`` (``exp(Pcp).sum(0)``) is the probability that a
    segment ends at data index ``i``, so the next one starts at ``i + 1``.
    Each run of consecutive indices above ``threshold`` is one detection, at
    its most probable index.
    """
    probs = np.asarray(changepoint_probs, dtype=float)
    above = np.flatnonzero(probs > threshold)
    starts = []
    for run in np.split(above, np.flatnonzero(np.diff(above) > 1) + 1):
        if len(run):
            starts.append(int(run[np.argmax(probs[run])]) + 1)
    return starts


def online_starts(R, min_separation=MARGIN):
    """Segment starts from an online run-length posterior ``R``.

    Column ``t`` of ``R`` is the posterior after ``t`` observations; its MAP
    run length ``r`` implies a segment that started at data index ``t - r``.
    A start is reported the first time the implied start moves forward to a
    new position at least ``min_separation`` past the last one reported (in
    the spirit of ``get_map_changepoints``, re-implemented here because 0.4
    and 1.0.0 lack it, so that every version is scored identically).
    """
    R = np.asarray(R)
    map_run_lengths = R[:, 1:].argmax(axis=0)
    return starts_from_map_run_lengths(map_run_lengths, min_separation)


def starts_from_map_run_lengths(map_run_lengths, min_separation=MARGIN):
    """``map_run_lengths[k]`` is the MAP run length after ``k + 1`` observations."""
    starts, last = [], 0
    for k, run_length in enumerate(map_run_lengths):
        start = k + 1 - int(run_length)
        if start > last and start - last >= max(min_separation, 1):
            starts.append(start)
            last = start
    return starts


def score(detected, truth, margin=MARGIN):
    """Precision, recall and F1 with a tolerance margin.

    Each true change can be matched by at most one detection within
    ``margin`` observations (greedy, closest first), as in the F1 measure of
    van den Burg and Williams (2020). ``correct`` means every true change was
    found and nothing else was reported.
    """
    unmatched = list(truth)
    true_positives = 0
    for d in sorted(detected):
        if not unmatched:
            break
        distances = [abs(d - t) for t in unmatched]
        best = int(np.argmin(distances))
        if distances[best] <= margin:
            true_positives += 1
            unmatched.pop(best)
    precision = true_positives / len(detected) if detected else 1.0
    recall = true_positives / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": true_positives == len(truth) == len(detected),
    }
