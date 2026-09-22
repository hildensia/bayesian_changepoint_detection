"""
Streaming (incremental) online changepoint detection.

``online_changepoint_detection`` needs the whole series up front and returns
the ``[T + 1, T + 1]`` run-length matrix ``R``. ``OnlineChangepointDetector``
runs the same Adams & MacKay (2007) recursion one observation at a time and
keeps only the current column of ``R``, so it works on a stream of unknown
length (issue #13). With ``max_run_length`` set, memory and time per
observation are bounded as well.
"""

from typing import Callable, Optional, Union

import torch

from .device import ensure_tensor, get_device
from .online_likelihoods import BaseLikelihood as OnlineLikelihood


class OnlineChangepointDetector:
    """
    Incremental Bayesian online changepoint detection (Adams & MacKay 2007).

    Feed observations one at a time with ``update``. After each one the
    detector holds the posterior over the current run length (the number of
    observations since the last changepoint), which is the column of ``R``
    that ``online_changepoint_detection`` would have produced at that step.

    Parameters
    ----------
    hazard_function : callable
        Maps a tensor of run lengths to hazard probabilities, e.g.
        ``partial(constant_hazard, 250)``.
    likelihood_model : online_likelihoods.BaseLikelihood
        A fresh online likelihood (``StudentT``, ``MultivariateT``). It is
        updated in place and owned by the detector from now on; do not share
        it with another detector or with ``online_changepoint_detection``.
    max_run_length : int or None, optional
        If given, keep run lengths ``0 .. max_run_length`` only: after each
        update, longer run lengths are dropped and the posterior is
        renormalized, and the likelihood's parameters for them are pruned.
        Memory and time per observation are then O(``max_run_length``)
        instead of growing with the stream. The result is exact while fewer
        than ``max_run_length`` observations have been seen; after that it is
        the posterior conditioned on the current segment being at most
        ``max_run_length`` long, which is a good approximation when segments
        are shorter than that. Default None: exact, and memory grows by one
        entry per observation.
    device : str, torch.device, or None, optional
        Device for the computation; defaults to the likelihood's device.

    Examples
    --------
    >>> from functools import partial
    >>> detector = OnlineChangepointDetector(
    ...     partial(constant_hazard, 250), StudentT(), max_run_length=500
    ... )
    >>> for x in stream:
    ...     detector.update(x)
    ...     if detector.changepoint_probability(lag=10) > 0.5:
    ...         print("segment started at", detector.t - 10)

    Notes
    -----
    The recursion, per observation ``x_t`` with run-length posterior ``r``
    and predictive densities ``p(x_t | r)``:
    ``r'[k + 1] = r[k] p(x_t | k) (1 - H(k))`` and
    ``r'[0] = sum_k r[k] p(x_t | k) H(k)``, then ``r'`` is normalized.
    Adams & MacKay (2007), algorithm 1; the same arithmetic as
    ``online_changepoint_detection``.
    """

    def __init__(
        self,
        hazard_function: Callable[[torch.Tensor], torch.Tensor],
        likelihood_model: OnlineLikelihood,
        max_run_length: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if max_run_length is not None:
            if isinstance(max_run_length, bool) or not isinstance(max_run_length, int):
                raise TypeError(
                    f"max_run_length must be an int or None, got {max_run_length!r}"
                )
            if max_run_length < 1:
                raise ValueError(
                    f"max_run_length must be at least 1, got {max_run_length}"
                )
            if not getattr(likelihood_model, "_run_length_state", ()):
                raise ValueError(
                    f"{type(likelihood_model).__name__} does not support "
                    "max_run_length: it declares no _run_length_state to prune"
                )
        if device is None and hasattr(likelihood_model, "device"):
            device = likelihood_model.device
        self.device = get_device(device)
        if hasattr(likelihood_model, "to"):
            likelihood_model.to(self.device)
        self.hazard_function = hazard_function
        self.likelihood_model = likelihood_model
        self.max_run_length = max_run_length
        self._t = 0
        # Before any data the run length is 0 with probability 1 (R[:, 0]).
        self._posterior = torch.ones(1, device=self.device, dtype=torch.float32)

    @property
    def t(self) -> int:
        """Number of observations processed so far."""
        return self._t

    @property
    def run_length_posterior(self) -> torch.Tensor:
        """
        ``P(run length = r | observations so far)`` for ``r = 0, 1, ...``.

        A copy, of length ``t + 1`` (or ``max_run_length + 1`` once the bound
        is reached). Equal to ``R[:len, t]`` of ``online_changepoint_detection``
        on the same data.
        """
        return self._posterior.clone()

    @property
    def map_run_length(self) -> int:
        """The most probable current run length (``argmax`` of the posterior)."""
        return int(torch.argmax(self._posterior))

    def changepoint_probability(self, lag: int) -> float:
        """
        Probability that a new segment started ``lag`` observations ago.

        That is ``P(run length = lag)``: the segment containing the latest
        observation began at data index ``t - lag``. This is entry
        ``t - lag`` of ``changepoint_probabilities(R, lag)``, available as soon
        as observation ``t - 1`` has been processed. As with that function,
        ``lag = 0`` is uninformative under a constant hazard; use ``lag >= 1``.

        Raises
        ------
        ValueError
            If ``lag`` is negative, larger than ``t``, or beyond
            ``max_run_length``.
        """
        if lag < 0 or lag >= self._posterior.numel():
            raise ValueError(
                f"lag must be in [0, {self._posterior.numel() - 1}] after "
                f"{self._t} observations, got {lag}"
            )
        return float(self._posterior[lag])

    def _check_observation(self, x) -> torch.Tensor:
        x = ensure_tensor(x, device=self.device)
        if x.is_complex():
            raise ValueError(f"observation must be real, got dtype {x.dtype}")
        dims = getattr(self.likelihood_model, "dims", None)
        if isinstance(dims, int) and not isinstance(dims, bool):
            if x.dim() == 0 and dims == 1:
                x = x.reshape(1)
            if tuple(x.shape) != (dims,):
                raise ValueError(
                    f"the likelihood expects observations of shape [{dims}], "
                    f"got {list(x.shape)}"
                )
        elif x.numel() != 1:
            raise ValueError(
                f"the likelihood expects scalar observations, got shape {list(x.shape)}"
            )
        if not bool(torch.isfinite(x).all()):
            raise ValueError("observation is NaN or Inf; remove or impute it first")
        return x

    def update(self, x) -> torch.Tensor:
        """
        Process one observation and return the new run-length posterior.

        Parameters
        ----------
        x : float, torch.Tensor or array-like
            A scalar for a univariate likelihood, a ``[dims]`` vector for a
            multivariate one. It is checked before the state changes: a
            rejected observation leaves the detector as it was.

        Returns
        -------
        torch.Tensor
            ``run_length_posterior`` after ``x`` (a copy).
        """
        x = self._check_observation(x)
        r = self._posterior
        n = r.numel()

        pred_probs = torch.exp(self.likelihood_model.pdf(x))
        run_lengths = torch.arange(n, device=self.device, dtype=torch.float32)
        H = self.hazard_function(run_lengths)

        posterior = torch.zeros(n + 1, device=self.device, dtype=torch.float32)
        posterior[1:] = r * pred_probs * (1 - H)  # growth
        posterior[0] = torch.sum(r * pred_probs * H)  # changepoint
        total = torch.sum(posterior)
        if total > 0:
            posterior = posterior / total

        self.likelihood_model.update_theta(x, t=self._t)
        self._t += 1

        if self.max_run_length is not None and n + 1 > self.max_run_length + 1:
            keep = self.max_run_length + 1
            posterior = posterior[:keep]
            total = torch.sum(posterior)
            if total > 0:
                posterior = posterior / total
            self.likelihood_model.prune(keep)

        self._posterior = posterior
        return posterior.clone()
