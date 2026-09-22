# Benchmarks

Reproducible timings of the detectors, across released versions, on the same
data and parameters. Only measured numbers are published; each result file
records the hardware, software versions and commits it came from.

## Running

From the repository root, in an environment with the package's dev extra:

```bash
python benchmarks/performance.py                     # full suite: current checkout, v1.1.0, v1.0.0, v0.4
python benchmarks/performance.py --versions current  # this checkout only
python benchmarks/performance.py --versions current,origin/master  # a branch against master (any git ref)
python benchmarks/performance.py --suite quick       # a minute; what CI runs (current only)
python benchmarks/performance.py --output benchmarks/results/<date>-<machine>-<device>.json
python benchmarks/performance.py --report benchmarks/results/<file>.json  # tables of a saved run
```

Options: `--device cpu|mps|cuda` (the PyTorch versions), `--repeats N`
(default 5), `--budget SECONDS` (default 300 per case), `--threads N` (torch
intra-op threads). The `v0.4` baseline is the NumPy implementation and also
needs `scipy` and `decorator` (`pip install decorator`). Older versions are
exported with `git archive <tag>`, so the tags must be present (a shallow
clone has none; `git fetch --tags`).

## Protocol

- **Isolation.** Every (version, workload, size) case runs in a fresh Python
  process whose `PYTHONPATH` points at that version; the worker records the
  module file it imported and the driver rejects the case if it is not the
  intended copy.
- **Data.** `workloads.make_data`: four equal segments with means 0, 2, -1,
  1.5 and unit Gaussian noise, float64, fixed seed. In 5-D each mean is that
  scalar times a fixed unit direction. The true changepoints are the three
  segment starts.
- **Parameters, the same for every version.** Offline: `StudentT` (or
  `MultivariateT(dims=5)`) with its defaults, `const_prior(p = 1 / (n + 1))`,
  each version's default `truncate` (-40 up to 1.1.0, exact since). Online:
  `StudentT(alpha=0.1, beta=0.1, kappa=1, mu=0)` (or `MultivariateT(dims=5)`),
  `constant_hazard(n / 4)`. Streaming: a longer series of 250-point
  segments (means cycling through the four above), the same `StudentT`,
  `constant_hazard(250)` and `OnlineChangepointDetector` with
  `max_run_length=1000`. The bound conditions the posterior on segments no
  longer than it, so it must exceed the segment length: a first run with
  2 500-point segments and a bound of 500 forced a changepoint every 500
  observations (F1 near 0).
- **Input dtype.** float64, except float32 for 1.0.0, which fails on float64
  input to its multivariate online likelihood.
- **Timing.** One warm-up call on a 40-point series (imports, allocator,
  kernel compilation), then up to `--repeats` timed calls of the detector on
  the benchmark series, each preceded by another untimed warm-up call (which
  also resets 0.4's per-series likelihood cache). Stops early once the timed
  calls exceed the budget, and skips a size whose predicted single-call time
  (quadratic extrapolation from the previous size) exceeds it. Reported: the
  median, with min, max and the number of calls in the JSON. Accelerators are
  synchronized before the clock is read. Data preparation is outside the
  timed region; moving the data to the device is inside it.
- **Memory.** Increase of the process's peak resident set size over its value
  after the warm-up (`ru_maxrss`), in the JSON. It includes allocator slack
  and is only a coarse guide.
- **Detection quality.** Every timed call is scored against the true
  changepoints: F1 with a margin of 5 observations (each true change matched
  at most once), the measure of van den Burg and Williams (2020). Offline
  detections are the peaks of runs where the marginal changepoint probability
  exceeds 0.5; online detections are forward moves of the start implied by
  the MAP run length, at least 5 observations apart. The same code scores
  every version. This is a sanity check on a synthetic series, not a quality
  benchmark; see "Reference datasets" below.

## Caveats

- The offline `StudentT` likelihood of 0.4 and 1.0.0 scored each point under
  the posterior of the whole segment (an approximation); from 1.1.0 it is
  the exact marginal likelihood. Timings compare what each version computes
  by default, not identical arithmetic.
- 1.0.0's online detector crashes on a machine with MPS even with
  `device="cpu"`; the worker hides MPS for 1.0.0 so its CPU path can be timed,
  and the report says so. 0.4's multivariate online likelihood fails with a
  `NameError` (it uses `islice` without importing it) and is reported as an
  error.
- Timings depend on the machine, its load, the torch build and thread count.
  Compare numbers only within one result file.

## Results

Result files live in [`results/`](results/). The tables in the project
README come from the file named there.

## Detection quality on real data (TCPD)

`benchmarks/tcpd.py` scores the detectors on the Turing Change Point Dataset
(van den Burg and Williams, 2020): real series whose changepoints five
annotators marked independently, with TCPDBench's F1 (margin 5) and
covering metrics against all annotators (`metrics.py`, ported from
TCPDBench with its own examples as tests). It follows TCPDBench's protocol
(standardized series, BOCPD defaults `lambda = 100`, `a = b = k = 1`, and the
same 500-setting oracle grid) and puts the result next to TCPDBench's
published BOCPD scores on the same series.

```bash
python benchmarks/tcpd.py                 # default settings, about 15 s
python benchmarks/tcpd.py --oracle        # plus the grid, about an hour
python benchmarks/tcpd.py --report benchmarks/results/2026-09-23-tcpd.json
```

The 32 series TCPD redistributes are downloaded from a pinned TCPD commit,
checked against TCPD's checksums and cached in `benchmarks/.cache/`; the
series themselves are not copied into this repository (several carry their
own licenses), and the ten that TCPD can only rebuild from third-party
sources are not used. One series with missing values (`uk_coal_employ`) is
skipped, as TCPDBench's BOCPD also has no result for it.

Mean over the 30 univariate series with a TCPDBench BOCPD score
([`results/2026-09-23-tcpd.json`](results/2026-09-23-tcpd.json)):

| method | F1 | cover |
|---|---|---|
| no changepoints (TCPDBench's `zero` baseline) | 0.668 | 0.575 |
| TCPDBench BOCPD (R package `ocp`), default | 0.696 | 0.636 |
| this library, online, MAP segmentation (`viterbi_changepoints`), default | 0.694 | 0.637 |
| this library, online, filtered MAP run length (`get_map_changepoints`), default | 0.571 | 0.566 |
| this library, offline (`StudentT`, `const_prior(1/(n+1))`), default | **0.739** | **0.664** |
| TCPDBench BOCPD, oracle (best of 500 settings per series) | 0.890 | 0.789 |
| this library, online, MAP segmentation, oracle | 0.887 | 0.791 |

What this shows:

- The online model agrees with an independent implementation on real data:
  with TCPDBench's settings its MAP segmentation gives the same F1 as
  TCPDBench's BOCPD on 27 of the 31 series it scores, and the same averages
  to within 0.003, with default and with tuned settings.
- The offline detector, with no tuning, beats default BOCPD on both
  metrics.
- The filtered readout, which reports changes as the data arrive, scores
  below the do-nothing baseline on whole-series segmentation: without
  hindsight, trends and slow drifts restart the run length repeatedly. For
  a finished series, use `viterbi_changepoints` or the offline detector.

The oracle rows use the best setting per series and metric, chosen on the
evaluation data itself, so they are an upper bound, not an achievable score.

## Reference datasets

TCPD is the reference quality benchmark (above). The synthetic series in
`performance.py` remain the timing workload and a sanity check.

van den Burg, G. J. J., and Williams, C. K. I. (2020). An evaluation of
change point detection algorithms. arXiv:2003.06222.
