"""
Core Bayesian changepoint detection algorithms.

This module implements both online and offline Bayesian changepoint detection
algorithms using PyTorch for efficient computation and GPU acceleration.
"""

import warnings

import torch
from typing import Union, Callable, Tuple, Optional
from .device import ensure_tensor, get_device
from .online_likelihoods import BaseLikelihood as OnlineLikelihood
from .offline_likelihoods import BaseLikelihood as OfflineLikelihood


def _nan_to_neg_inf(x: torch.Tensor) -> torch.Tensor:
    """Map NaN to -inf while leaving +/-inf untouched.

    ``torch.nan_to_num`` with default ``posinf``/``neginf`` would also clamp
    infinities to finite extrema, turning impossible (log-probability -inf)
    entries into finite values that then take part in later recursions.
    """
    return torch.where(torch.isnan(x), torch.full_like(x, float('-inf')), x)


def offline_changepoint_detection(
    data: torch.Tensor,
    prior_function: Callable[[int], float],
    likelihood_model: OfflineLikelihood,
    truncate: float = -40.0,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Offline Bayesian changepoint detection using dynamic programming.
    
    Computes the exact posterior distribution over changepoint locations
    using the algorithm described in Fearnhead (2006).
    
    Parameters
    ----------
    data : torch.Tensor
        Time series data of shape [T] or [T, D] where T is time and D is dimensions.
    prior_function : callable
        Log prior probability mass of a segment length: ``prior_function(l)``
        returns ``log P(length = l)`` for ``l = 1 .. T`` (Fearnhead's ``g``).
        Use ``const_prior``, ``geometric_prior`` or ``negative_binomial_prior``
        with ``functools.partial``. The mass on lengths ``1 .. T - 1`` must be
        below 1 (always true for a proper distribution; for ``const_prior``
        this means ``p * (T - 1) < 1``).
    likelihood_model : OfflineLikelihood
        Likelihood model for computing segment probabilities.
    truncate : float, optional
        Log probability threshold for truncating computation (default: -40.0).
        More negative values = more accurate but slower computation.
    device : str, torch.device, or None, optional
        Device to place tensors on.
        
    Returns
    -------
    Q : torch.Tensor
        Log evidence for data[t:] for each time t. Shape: [T].
    P : torch.Tensor  
        Log likelihood of segment [t, s] with no changepoints. Shape: [T, T].
    Pcp : torch.Tensor
        Log probability of j-th changepoint at time t. Shape: [T-1, T-1].
        
    Examples
    --------
    >>> import torch
    >>> from functools import partial
    >>> from bayesian_changepoint_detection import (
    ...     offline_changepoint_detection, const_prior, StudentT
    ... )
    >>> 
    >>> data = torch.randn(100)
    >>> prior_func = partial(const_prior, p=0.01)
    >>> likelihood = StudentT()
    >>> Q, P, Pcp = offline_changepoint_detection(data, prior_func, likelihood)
    >>> 
    >>> # Get changepoint probabilities
    >>> changepoint_probs = torch.exp(Pcp).sum(0)
    >>> detected_changepoints = torch.where(changepoint_probs > 0.5)[0]
    
    Notes
    -----
    This algorithm has O(T^2) time complexity in the worst case, but the truncation
    parameter can make it approximately O(T) for most practical cases.

    Model (Fearnhead 2006, section 2): segment lengths are i.i.d. with mass
    function ``g``, except the last segment, whose length is only known to be
    at least what is observed, ``P(length >= l) = 1 - G(l - 1)`` with
    ``G(l) = sum_{i <= l} g(i)``. Segments are independent given the
    changepoints. With ``P[t, s]`` the log marginal likelihood of
    ``data[t:s+1]``, the backward recursion (eq. 2 of the paper, 0-indexed) is

        Q[t] = sum_{s=t}^{T-2} P[t, s] Q[s+1] g(s+1-t)  +  P[t, T-1] (1 - G(T-1-t))

    and the changepoint posteriors are ``Pcp[0, t] = P[0, t] Q[t+1] g(t+1) / Q[0]``
    for the first changepoint (first segment ``data[0:t+1]`` has length
    ``t + 1``) and, for the ``j``-th, a sum over the previous changepoint ``s``
    of ``Pcp[j-1, s] P[s+1, t] Q[t+1] g(t-s) / Q[s+1]``. Versions up to 1.0.x
    evaluated ``g`` at length minus one in the first row, paired ``g`` with
    the wrong segment length in the later rows, and included a "length 0"
    term in ``G``; none of that is visible with ``const_prior``, all of it is
    with the geometric and negative binomial priors.

    References
    ----------
    Fearnhead, P. (2006). Exact and efficient Bayesian inference for multiple
    changepoint problems. Statistics and Computing, 16(2), 203-213.
    """
    device = get_device(device)
    if device.type == "mps":
        # The offline recursion needs float64, which MPS does not support.
        warnings.warn(
            "MPS does not support float64; running offline changepoint "
            "detection on CPU instead."
        )
        device = torch.device("cpu")
    data = ensure_tensor(data, device=device)
    dtype = torch.float64

    n = data.shape[0]  # First dimension is time
    if n == 0:
        raise ValueError("data must contain at least one observation")
    if not bool(torch.isfinite(data).all()):
        raise ValueError("data contains NaN or Inf; remove or impute them first")

    # Precompute per-dataset sufficient statistics (cumulative sums) so that
    # every pdf_rows call below is a single vectorized pass. The caller's
    # tensor is passed on unchanged (shape and dtype included): built-in
    # likelihoods re-enter setup() and hit the cache, while third-party
    # likelihoods that only implement pdf(data, t, s) see the same data they
    # would have seen before setup() existed.
    if hasattr(likelihood_model, "device"):
        likelihood_model.device = device
    setup = getattr(likelihood_model, "setup", None)
    if setup is not None:
        setup(data)

    # Initialize arrays
    Q = torch.zeros(n, device=device, dtype=dtype)
    P = torch.full((n, n), float('-inf'), device=device, dtype=dtype)

    # Segment-length prior in log space, indexed by length: g[l] = log g(l)
    # for l = 1 .. n; a segment of length 0 is impossible, so g[0] = -inf and
    # G[l] = log sum_{i=1}^{l} g(i) comes straight out of the cumulative sum.
    g = torch.full((n + 1,), float('-inf'), device=device, dtype=dtype)
    for length in range(1, n + 1):
        g[length] = float(prior_function(length))
    G = torch.logcumsumexp(g, dim=0)
    if n > 1 and bool(G[n - 1] > 1e-12):
        raise ValueError(
            "prior_function puts total mass "
            f"{torch.exp(G[n - 1]).item():.4g} > 1 on segment lengths "
            f"1..{n - 1}; it must be a (sub-)probability mass function on "
            "lengths. For const_prior use p < 1 / (T - 1), e.g. p = 1 / (T + 1)."
        )
    # log(1 - G(l)) for every l, stable all the way up to G = 1 (-> -inf).
    log_one_minus_G = torch.log(-torch.expm1(torch.clamp(G, max=0.0)))

    # Initialize the last time point
    P[n - 1, n - 1] = likelihood_model.pdf(data, n - 1, n)
    Q[n - 1] = P[n - 1, n - 1]

    # Dynamic programming: work backwards through time. For each start point
    # t, likelihoods of all segments [t, s] are computed in one vectorized
    # call; the truncated logaddexp recursion (Fearnhead 2006, eq. 3) is
    # evaluated with a running logcumsumexp instead of a Python loop.
    for t in reversed(range(n - 1)):
        # row[j] = log p(data[t:t+1+j]) for j = 0 .. n-1-t
        row = likelihood_model.pdf_rows(data, t).to(device=device, dtype=dtype)
        P[t, t:] = row

        # summand[j] = P[t, t+j] + Q[t+j+1] + g[j+1] for j = 0 .. n-2-t
        summand = row[:n - 1 - t] + Q[t + 1:] + g[1:n - t]
        running = torch.logcumsumexp(summand, dim=0)

        # Truncate the sum where later terms cannot contribute anymore
        # (identical to breaking out of the sequential loop).
        truncated = (summand - running) < truncate
        if bool(truncated.any()):
            cutoff = int(torch.nonzero(truncated)[0])
        else:
            cutoff = summand.shape[0] - 1
        P_next_cp = running[cutoff]

        # Last segment data[t:] has length n - t; its prior probability is
        # P(length >= n - t) = 1 - G(n - 1 - t).
        Q[t] = torch.logaddexp(P_next_cp, P[t, n - 1] + log_one_minus_G[n - 1 - t])

    # Compute changepoint probability matrix
    Pcp = torch.full((n - 1, n - 1), float('-inf'), device=device, dtype=dtype)

    # First changepoint at t: the first segment is data[0:t+1], length t + 1.
    if n > 1:
        Pcp[0, :] = _nan_to_neg_inf(P[0, :n - 1] + Q[1:] + g[1:n] - Q[0])

    # Subsequent changepoints. For each j the sum over the previous
    # changepoint s = j-1+i (rows) for every t = j+c (columns) is one masked
    # logsumexp over an [m, m] matrix
    #   M[i, c] = Pcp[j-1, s] - Q[s+1] + P[s+1, t] + Q[t+1] + g(t - s),
    # where the segment data[s+1:t+1] has length t - s = c - i + 1 >= 1,
    # i.e. only i <= c contributes.
    for j in range(1, n - 1):
        m = n - 1 - j
        head = Pcp[j - 1, j - 1:n - 2] - Q[j:n - 1]  # [m], indexed by i
        rows = torch.arange(m, device=device).unsqueeze(1)
        cols = torch.arange(m, device=device).unsqueeze(0)
        length = (cols - rows + 1).clamp(min=0)     # 0 where i > c -> g[0] = -inf
        M = head.unsqueeze(1) + P[j:n - 1, j:n - 1] + Q[j + 1:].unsqueeze(0) + g[length]
        M = M.masked_fill(rows > cols, float('-inf'))
        Pcp[j, j:] = _nan_to_neg_inf(torch.logsumexp(M, dim=0))

    return Q, P, Pcp


def online_changepoint_detection(
    data: torch.Tensor,
    hazard_function: Callable[[torch.Tensor], torch.Tensor],
    likelihood_model: OnlineLikelihood,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Online Bayesian changepoint detection with run length filtering.
    
    Processes data sequentially, maintaining a posterior distribution over
    run lengths (time since last changepoint) as described in Adams & MacKay (2007).
    
    Parameters
    ----------
    data : torch.Tensor
        Time series data of shape [T] or [T, D] where T is time and D is dimensions.
    hazard_function : callable
        Function that takes run length tensor and returns hazard probabilities.
        Should accept torch.Tensor of run lengths and return torch.Tensor of same shape.
    likelihood_model : OnlineLikelihood
        Online likelihood model that maintains sufficient statistics.
    device : str, torch.device, or None, optional
        Device to place tensors on.
        
    Returns
    -------
    R : torch.Tensor
        Run length posterior. ``R[r, t]`` is ``P(run length = r | x_0..x_{t-1})``,
        i.e. column ``t`` is the posterior after ``t`` observations; column 0
        is the prior (all mass at run length 0). Shape: ``[T+1, T+1]``.
    map_run_lengths : torch.Tensor
        ``argmax`` of each column of ``R``: the most likely run length after
        ``t`` observations. Shape: ``[T+1]``, dtype ``long``. A changepoint
        shows up as a drop in this sequence; ``get_map_changepoints`` turns
        the drops into segment start indices, and
        ``changepoint_probabilities`` gives a calibrated probability per
        position at a chosen detection lag.
        
    Examples
    --------
    >>> import torch
    >>> from functools import partial
    >>> from bayesian_changepoint_detection import (
    ...     online_changepoint_detection, constant_hazard, StudentT,
    ...     get_map_changepoints, changepoint_probabilities,
    ... )
    >>> 
    >>> _ = torch.manual_seed(0)
    >>> data = torch.cat([torch.randn(80), torch.randn(80) + 5])
    >>> hazard_func = partial(constant_hazard, 100)  # Expected run length = 100
    >>> likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0, device="cpu")
    >>> R, map_run_lengths = online_changepoint_detection(
    ...     data, hazard_func, likelihood, device="cpu"
    ... )
    >>> get_map_changepoints(R)
    tensor([80])
    >>> changepoint_probabilities(R, lag=10)[80] > 0.85
    tensor(True)
    
    Notes
    -----
    This algorithm has O(T^2) time complexity but is naturally online and can
    process streaming data. The run length distribution is normalized at each
    step for numerical stability.

    **Why the second return value is a run length and not a probability.**
    Under the Adams & MacKay recursion the posterior probability of run
    length 0 after each observation, ``R[0, t]``, is the hazard evaluated
    under the *previous* run-length distribution; with a constant hazard it
    is identically ``1/lam`` and carries no information about the data. The
    evidence for a changepoint at position ``tau`` accumulates in the
    following columns as mass at run length ``k`` in column ``tau + k``.
    Versions 1.0.x returned the un-normalized ``R[0, t]`` under the name
    ``changepoint_probs``; that quantity could not detect changepoints. This
    version restores the pre-1.0 return value (the MAP run length) and adds
    ``changepoint_probabilities`` for the lagged probability.
    
    References
    ----------
    Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
    arXiv preprint arXiv:0710.3742.
    """
    # Keep the likelihood model's device authoritative when none is given, and
    # move everything (model state, data, run-length matrix) to one device so
    # mixed CPU/GPU inputs cannot collide mid-recursion.
    if device is None and hasattr(likelihood_model, "device"):
        device = likelihood_model.device
    device = get_device(device)
    if hasattr(likelihood_model, "to"):
        likelihood_model.to(device)
    data = ensure_tensor(data, device=device)
    
    if data.dim() == 1:
        T = data.shape[0]
    else:
        T = data.shape[0]  # First dimension is time
    
    # Initialize run length probability matrix
    R = torch.zeros(T + 1, T + 1, device=device, dtype=torch.float32)
    R[0, 0] = 1.0  # Initially, run length is 0 with probability 1
    
    # Process each data point sequentially
    for t in range(T):
        # Get current data point
        if data.dim() == 1:
            x = data[t]
        else:
            x = data[t]
        
        # Evaluate predictive probabilities under current parameters
        # This gives us p(x_t | x_{1:t-1}, r_{t-1}) for all possible run lengths
        pred_log_probs = likelihood_model.pdf(x)
        
        # Convert to probabilities (but keep in log space for stability)
        pred_probs = torch.exp(pred_log_probs)
        
        # Evaluate hazard function for current run lengths
        run_lengths = torch.arange(t + 1, device=device, dtype=torch.float32)
        H = hazard_function(run_lengths)
        
        # Growth probabilities: shift probabilities down and right,
        # scaled by hazard function and predictive probabilities
        # R[r+1, t+1] = R[r, t] * p(x_t | r) * (1 - H(r))
        R[1:t + 2, t + 1] = R[0:t + 1, t] * pred_probs * (1 - H)
        
        # Changepoint probability: mass accumulates at r = 0
        # R[0, t+1] = sum_r R[r, t] * p(x_t | r) * H(r)
        R[0, t + 1] = torch.sum(R[0:t + 1, t] * pred_probs * H)
        
        # Normalize run length probabilities for numerical stability
        total_prob = torch.sum(R[:, t + 1])
        if total_prob > 0:
            R[:, t + 1] = R[:, t + 1] / total_prob
        
        # Update likelihood model parameters with new observation
        likelihood_model.update_theta(x, t=t)
    
    map_run_lengths = torch.argmax(R, dim=0)
    return R, map_run_lengths


def changepoint_probabilities(R: torch.Tensor, lag: int = 10) -> torch.Tensor:
    """
    Probability that a new segment started at each position, judged ``lag``
    observations later.

    ``result[tau] = R[lag, tau + lag]``: the posterior probability, after
    observing ``x_0 .. x_{tau+lag-1}``, that the current run length is exactly
    ``lag``, which is the event "the segment containing the latest point began
    at ``tau``". This is the quantity the original notebook plotted as
    ``R[Nw, Nw:]`` and the natural online detector with a fixed decision delay.

    Parameters
    ----------
    R : torch.Tensor
        Run length posterior from ``online_changepoint_detection``.
    lag : int, optional
        Detection delay in observations (default 10). ``lag=0`` gives the
        run-length-0 posterior, which under a constant hazard equals the hazard
        rate for every ``tau >= 1`` (and 1 at ``tau = 0``, the prior) and is
        therefore uninformative; use ``lag >= 1``.

    Returns
    -------
    torch.Tensor
        Shape ``[T + 1 - lag]``; entry ``tau`` refers to data index ``tau``.
        The last ``lag`` positions cannot be judged yet and are not returned.

    Examples
    --------
    >>> probs = changepoint_probabilities(R, lag=10)
    >>> detected = torch.where(probs > 0.5)[0]
    """
    if lag < 0 or lag >= R.shape[0]:
        raise ValueError(f"lag must be in [0, {R.shape[0] - 1}], got {lag}")
    n_cols = R.shape[1]
    return R[lag, lag:n_cols]


def get_map_changepoints(
    R: torch.Tensor,
    threshold: Optional[float] = None,
    min_separation: int = 0,
) -> torch.Tensor:
    """
    Segment start indices implied by the MAP run-length path.

    After ``t`` observations the MAP run length ``r_t = argmax R[:, t]`` says
    the current segment began at data index ``t - r_t``. Whenever that implied
    start moves forward (the MAP run length drops), a changepoint is
    reported at the new start. This is the classic BOCPD decision rule and is
    what the pre-1.0 versions of this library exposed as ``maxes``.

    Parameters
    ----------
    R : torch.Tensor
        Run length posterior from ``online_changepoint_detection``.
    threshold : float, optional
        Deprecated and ignored. Earlier versions thresholded ``R[0, :]``, which
        is not a changepoint signal (see ``online_changepoint_detection``).
        For a thresholded probability use ``changepoint_probabilities``.
    min_separation : int, optional
        When the posterior is split between two nearby starts the MAP path can
        flip between them and both get reported. Starts closer than this many
        observations to an earlier reported start are dropped (default 0:
        report every distinct start).

    Returns
    -------
    torch.Tensor
        Sorted data indices at which a new segment starts (``long``). Index 0
        is never reported.

    Examples
    --------
    >>> R, map_run_lengths = online_changepoint_detection(data, hazard_func, likelihood)
    >>> get_map_changepoints(R)
    """
    if threshold is not None:
        warnings.warn(
            "get_map_changepoints(threshold=...) is ignored: R[0, :] is not a "
            "changepoint probability. Use changepoint_probabilities(R, lag) "
            "for a thresholded detector.",
            DeprecationWarning,
            stacklevel=2,
        )
    n_cols = R.shape[1]
    map_run_lengths = torch.argmax(R, dim=0)
    columns = torch.arange(n_cols, device=R.device)
    segment_start = columns - map_run_lengths  # implied start after t obs
    # A changepoint is a forward move of the implied start. Report each
    # distinct start once, at the first column that implies it.
    moved = torch.zeros(n_cols, dtype=torch.bool, device=R.device)
    moved[1:] = segment_start[1:] > segment_start[:-1]
    starts = segment_start[moved]
    starts = torch.unique(starts[starts > 0])
    if min_separation > 0 and starts.numel() > 1:
        kept = [starts[0]]
        for candidate in starts[1:]:
            if candidate - kept[-1] >= min_separation:
                kept.append(candidate)
        starts = torch.stack(kept)
    return starts


def compute_run_length_posterior(
    data: torch.Tensor,
    hazard_function: Callable[[torch.Tensor], torch.Tensor],
    likelihood_model: OnlineLikelihood,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Compute the full run length posterior distribution.
    
    This is a convenience function that returns just the run length
    posterior from online changepoint detection.
    
    Parameters
    ----------
    data : torch.Tensor
        Time series data.
    hazard_function : callable
        Hazard function for changepoint prior.
    likelihood_model : OnlineLikelihood
        Online likelihood model.
    device : str, torch.device, or None, optional
        Device to place tensors on.
        
    Returns
    -------
    torch.Tensor
        Run length posterior distribution R[r, t].
        
    Examples
    --------
    >>> posterior = compute_run_length_posterior(data, hazard_func, likelihood)
    >>> # Most likely run length at each time
    >>> map_run_lengths = torch.argmax(posterior, dim=0)
    """
    R, _ = online_changepoint_detection(data, hazard_function, likelihood_model, device)
    return R


def viterbi_changepoints(
    data: torch.Tensor,
    hazard_function: Callable[[torch.Tensor], torch.Tensor],
    likelihood_model: OnlineLikelihood,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the most likely sequence of changepoints using Viterbi algorithm.
    
    This finds the single most likely sequence of run lengths, rather than
    maintaining the full posterior distribution.
    
    Parameters
    ----------
    data : torch.Tensor
        Time series data.
    hazard_function : callable
        Hazard function for changepoint prior.
    likelihood_model : OnlineLikelihood
        Online likelihood model.
    device : str, torch.device, or None, optional
        Device to place tensors on.
        
    Returns
    -------
    run_lengths : torch.Tensor
        Most likely run length sequence.
    changepoints : torch.Tensor
        Indices of detected changepoints.
        
    Examples
    --------
    >>> run_lengths, changepoints = viterbi_changepoints(data, hazard_func, likelihood)
    >>> print(f"Changepoints at: {changepoints}")
    """
    device = get_device(device)
    data = ensure_tensor(data, device=device)
    
    if data.dim() == 1:
        T = data.shape[0]
    else:
        T = data.shape[0]
    
    # Viterbi tables
    log_probs = torch.full((T + 1, T + 1), float('-inf'), device=device)
    backpointers = torch.zeros((T + 1, T + 1), device=device, dtype=torch.long)
    
    # Initialize
    log_probs[0, 0] = 0.0
    
    # Forward pass
    for t in range(T):
        if data.dim() == 1:
            x = data[t]
        else:
            x = data[t]
        
        pred_log_probs = likelihood_model.pdf(x)
        
        run_lengths = torch.arange(t + 1, device=device, dtype=torch.float32)
        H = hazard_function(run_lengths)
        
        # Growth transitions (no changepoint)
        for r in range(t + 1):
            if log_probs[r, t] > float('-inf'):
                new_prob = (
                    log_probs[r, t] + 
                    pred_log_probs[r] + 
                    torch.log(1 - H[r])
                )
                if new_prob > log_probs[r + 1, t + 1]:
                    log_probs[r + 1, t + 1] = new_prob
                    backpointers[r + 1, t + 1] = r
        
        # Changepoint transitions
        total_changepoint_prob = torch.tensor(float('-inf'), device=device)
        for r in range(t + 1):
            if log_probs[r, t] > float('-inf'):
                cp_prob = (
                    log_probs[r, t] + 
                    pred_log_probs[r] + 
                    torch.log(H[r])
                )
                total_changepoint_prob = torch.logaddexp(total_changepoint_prob, cp_prob)
        
        if total_changepoint_prob > log_probs[0, t + 1]:
            log_probs[0, t + 1] = total_changepoint_prob
            # Find best predecessor for changepoint
            best_r = -1
            best_prob = float('-inf')
            for r in range(t + 1):
                if log_probs[r, t] > float('-inf'):
                    cp_prob = (
                        log_probs[r, t] + 
                        pred_log_probs[r] + 
                        torch.log(H[r])
                    )
                    if cp_prob > best_prob:
                        best_prob = cp_prob
                        best_r = r
            backpointers[0, t + 1] = best_r
        
        likelihood_model.update_theta(x, t=t)
    
    # Backward pass to find best path
    run_lengths = torch.zeros(T + 1, device=device, dtype=torch.long)
    
    # Find best final run length
    best_final_r = torch.argmax(log_probs[:, T])
    run_lengths[T] = best_final_r
    
    # Trace back
    for t in reversed(range(T)):
        run_lengths[t] = backpointers[run_lengths[t + 1], t + 1]
    
    # Extract changepoints (where run length resets to 0)
    changepoints = torch.where(run_lengths[1:] == 0)[0] + 1
    
    return run_lengths, changepoints