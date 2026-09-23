"""
Changepoint evaluation metrics of TCPDBench, against several annotators.

Ported from ``analysis/scripts/metrics.py`` of TCPDBench
(https://github.com/alan-turing-institute/TCPDBench), copyright (c) 2020 The
Alan Turing Institute, MIT license; see van den Burg and Williams (2020), An
evaluation of change point detection algorithms, arXiv:2003.06222, section 4.
The logic is unchanged; the doctests are theirs and run in
``tests/test_benchmark_metrics.py``.

Locations are 0-based indices of the first observation of a new segment.
Both metrics treat index 0 as a changepoint of every segmentation.
"""


def true_positives(T, X, margin=5):
    """True positives without double counting: each element of ``X`` can
    match at most one element of ``T``, the closest within ``margin``.

    >>> sorted(true_positives({1, 10, 20, 23}, {3, 8, 20}))
    [1, 10, 20]
    >>> sorted(true_positives({1, 10, 20, 23}, {1, 3, 8, 20}))
    [1, 10, 20]
    >>> sorted(true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20}))
    [1, 10, 20]
    >>> true_positives(set(), {1, 2, 3})
    set()
    >>> true_positives({1, 2, 3}, set())
    set()
    """
    X = set(X)
    TP = set()
    for tau in T:
        close = sorted((abs(tau - x), x) for x in X if abs(tau - x) <= margin)
        if not close:
            continue
        TP.add(tau)
        X.remove(close[0][1])
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """F-measure against several annotators.

    Precision is taken against the union of all annotators' changepoints;
    recall is the mean over annotators, so a method is rewarded for finding
    what every annotator marked. ``annotations`` maps annotator id to a list
    of locations.

    >>> f_measure({1: [10, 20], 2: [11, 20], 3: [10], 4: [0, 5]}, [10, 20])
    1.0
    >>> f_measure({1: [], 2: [10], 3: [50]}, [10])
    0.9090909090909091
    >>> f_measure({1: [], 2: [10], 3: [50]}, [])
    0.8
    """
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)
    X = set(predictions)
    X.add(0)
    Tstar = set().union(*Tks.values())
    K = len(Tks)
    P = len(true_positives(Tstar, X, margin=margin)) / len(X)
    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)
    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def overlap(A, B):
    """Jaccard index of two sets.

    >>> overlap({1, 2, 3}, set())
    0.0
    >>> overlap({1, 2, 3}, {2, 5})
    0.25
    >>> overlap({1, 2, 3}, {1, 2, 3})
    1.0
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """The partition of ``range(n_obs)`` given by the changepoint locations.

    >>> partition_from_cps([], 5)
    [{0, 1, 2, 3, 4}]
    >>> partition_from_cps([3, 5], 8)
    [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    >>> partition_from_cps([1, 2, 7], 8)
    [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    >>> partition_from_cps([0, 4], 6)
    [{0, 1, 2, 3}, {4, 5}]
    """
    partition = []
    current = set()
    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(n_obs):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(S, Sprime):
    """Covering of segmentation ``S`` by ``Sprime`` (Arbelaez et al. 2010, eq. 8).

    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2, 3}, {4, 5}, {6}])
    0.8333333333333334
    >>> cover_single([{1, 2, 3, 4, 5, 6}], [{1, 2, 3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1, 2, 3}, {4, 5, 6}], [{1, 2}, {3, 4}, {5, 6}])
    0.6666666666666666
    >>> cover_single([{1}, {2}, {3}, {4, 5, 6}], [{1, 2, 3, 4, 5, 6}])
    0.3333333333333333
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    return sum(len(R) * max(overlap(R, Rp) for Rp in Sprime) for R in S) / T


def covering(annotations, predictions, n_obs):
    """Mean covering of each annotator's segmentation by the prediction.

    >>> covering({1: [10, 20], 2: [10], 3: [0, 5]}, [10, 20], 45)
    0.7962962962962963
    >>> covering({1: [], 2: [10], 3: [40]}, [10], 45)
    0.7954144620811286
    >>> covering({1: [], 2: [10], 3: [40]}, [], 45)
    0.8189300411522634
    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)
    Cs = [cover_single(Ak[k], pX) for k in Ak]
    return sum(Cs) / len(Cs)
