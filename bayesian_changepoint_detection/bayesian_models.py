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
        Function that returns log prior probability for a segment of given length.
        Should take an integer (segment length) and return a float (log probability).
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
    g = torch.zeros(n, device=device, dtype=dtype)
    P = torch.full((n, n), float('-inf'), device=device, dtype=dtype)

    # Compute prior probabilities in log space
    for t in range(n):
        g[t] = prior_function(t)
    G = torch.logcumsumexp(g, dim=0)

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

        # Compute (1 - G) in numerically stable way
        if G[n - 1 - t] < -1e-15:  # exp(-1e-15) ≈ 0.99999...
            antiG = torch.log(1 - torch.exp(G[n - 1 - t]))
        else:
            # For G close to 1, use approximation (1 - G) ≈ -log(G)
            antiG = torch.log(-G[n - 1 - t])

        # Combine changepoint and no-changepoint probabilities
        Q[t] = torch.logaddexp(P_next_cp, P[t, n - 1] + antiG)

    # Compute changepoint probability matrix
    Pcp = torch.full((n - 1, n - 1), float('-inf'), device=device, dtype=dtype)

    # First changepoint probabilities
    if n > 1:
        Pcp[0, :] = torch.nan_to_num(
            P[0, :n - 1] + Q[1:] + g[:n - 1] - Q[0], nan=float('-inf')
        )

    # Subsequent changepoint probabilities. For each j the inner loop over t
    # is one masked logsumexp over a [m, m] matrix M with
    # M[i, t-j] = Pcp[j-1, j-1+i] + g[i] - Q[j+i] + P[j+i, t] + Q[t+1],
    # restricted to i <= t - j.
    for j in range(1, n - 1):
        m = n - 1 - j
        head = Pcp[j - 1, j - 1:n - 2] + g[:m] - Q[j:n - 1]  # [m]
        M = head.unsqueeze(1) + P[j:n - 1, j:n - 1] + Q[j + 1:].unsqueeze(0)
        mask = torch.ones(m, m, dtype=torch.bool, device=device).triu()
        M = M.masked_fill(~mask, float('-inf'))
        Pcp[j, j:] = torch.nan_to_num(
            torch.logsumexp(M, dim=0), nan=float('-inf')
        )

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
        Run length probability matrix. R[r, t] is the probability at time t
        that the current run length is r. Shape: [T+1, T+1].
    changepoint_probs : torch.Tensor
        Probability of changepoint at each time step. Shape: [T+1].
        
    Examples
    --------
    >>> import torch
    >>> from functools import partial
    >>> from bayesian_changepoint_detection import (
    ...     online_changepoint_detection, constant_hazard, StudentT
    ... )
    >>> 
    >>> data = torch.randn(100)
    >>> hazard_func = partial(constant_hazard, 250)  # Expected run length = 250
    >>> likelihood = StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0)
    >>> R, changepoint_probs = online_changepoint_detection(data, hazard_func, likelihood)
    >>> 
    >>> # Detect changepoints with probability > 0.5
    >>> detected = torch.where(changepoint_probs > 0.5)[0]
    >>> print(f"Changepoints detected at: {detected}")
    
    Notes
    -----
    This algorithm has O(T^2) time complexity but is naturally online and can
    process streaming data. The run length distribution is normalized at each
    step for numerical stability.
    
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
    
    # Track changepoint probabilities (probability of changepoint at each time)
    changepoint_probs = torch.zeros(T + 1, device=device, dtype=torch.float32)
    changepoint_probs[0] = 1.0  # Changepoint at time 0 by definition
    
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
        
        # Store changepoint probability for this time step
        changepoint_probs[t + 1] = R[0, t + 1].clone()
        
        # Normalize run length probabilities for numerical stability
        total_prob = torch.sum(R[:, t + 1])
        if total_prob > 0:
            R[:, t + 1] = R[:, t + 1] / total_prob
        
        # Update likelihood model parameters with new observation
        likelihood_model.update_theta(x, t=t)
    
    return R, changepoint_probs


def get_map_changepoints(
    R: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Extract Maximum A Posteriori (MAP) changepoint estimates.
    
    Parameters
    ----------
    R : torch.Tensor
        Run length probability matrix from online_changepoint_detection.
    threshold : float, optional
        Probability threshold for declaring a changepoint (default: 0.5).
        
    Returns
    -------
    torch.Tensor
        Indices of detected changepoints.
        
    Examples
    --------
    >>> R, changepoint_probs = online_changepoint_detection(data, hazard_func, likelihood)
    >>> changepoints = get_map_changepoints(R, threshold=0.3)
    """
    # Get the most likely run length at each time step
    map_run_lengths = torch.argmax(R, dim=0)
    
    # Changepoints occur when run length drops to 0
    changepoint_mask = (map_run_lengths == 0)
    
    # Also check direct changepoint probabilities if available
    if R.shape[0] > 1:
        changepoint_probs = R[0, :]
        high_prob_mask = (changepoint_probs > threshold)
        changepoint_mask = changepoint_mask | high_prob_mask
    
    # Return indices of changepoints (excluding the first time point)
    changepoints = torch.where(changepoint_mask[1:])[0] + 1
    
    return changepoints


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