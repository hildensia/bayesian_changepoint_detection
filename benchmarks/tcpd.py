"""
Detection quality on the Turing Change Point Dataset (TCPD).

TCPD (van den Burg and Williams, 2020) is a set of real-world series whose
changepoints were marked by five annotators each; TCPDBench scores methods
on it with an F1 measure (margin 5) and a segmentation covering measure,
both against all annotators (``metrics.py``). This script runs this
library's detectors on the series with TCPDBench's protocol and puts the
result next to TCPDBench's published BOCPD scores for the same series.

    python benchmarks/tcpd.py                 # default settings
    python benchmarks/tcpd.py --oracle        # plus the TCPDBench grid (slow)
    python benchmarks/tcpd.py --output benchmarks/results/<file>.json
    python benchmarks/tcpd.py --report benchmarks/results/<file>.json

Data: the 32 series that TCPD redistributes in its repository, fetched from
a pinned commit, checked against TCPD's MD5 checksums, and cached under
``benchmarks/.cache/tcpd`` (gitignored). The other ten series must be built
from third-party sources with TCPD's own script and are not used; the
series are not copied into this repository because several carry their own
licenses. Protocol, following TCPDBench:

- every series is standardized (each dimension to zero mean, unit
  variance), as TCPDBench's loader does;
- ``online``: ``online_likelihoods.StudentT(alpha=1, beta=1, kappa=1,
  mu=0)`` and ``constant_hazard(100)``, TCPDBench's defaults for BOCPD, read
  out two ways: ``viterbi_changepoints`` (the MAP segmentation, the
  counterpart of the ``maxCPs`` output of the R package ``ocp`` that
  TCPDBench scores) and ``get_map_changepoints(R)`` (where the filtered MAP
  run length moves the segment start forward, available as the stream
  arrives);
- ``online, oracle`` (``--oracle``): the Viterbi readout over TCPDBench's grid
  (hazard 10, 50, 100, 200; alpha, beta, kappa each 0.01, 0.1, 1, 10, 100:
  500 settings), keeping the best F1 and the best cover per series
  separately, as TCPDBench does;
- ``offline, default``: ``offline_likelihoods.StudentT()`` and
  ``const_prior(1 / (n + 1))``; changepoints are the peaks of the runs where
  the marginal changepoint probability exceeds 0.5 (not part of TCPDBench,
  which has no Fearnhead-style offline method);
- ``zero``: no changepoints, TCPDBench's baseline;
- multivariate series use ``MultivariateT(dims=d)`` with its default prior
  (the oracle varies only the hazard for them: alpha, beta and kappa are
  univariate parameters);
- series with missing values are skipped (TCPDBench's BOCPD also has no
  result for them).
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import urllib.request
from functools import partial
from itertools import product

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import covering, f_measure  # noqa: E402
from workloads import offline_starts  # noqa: E402

import bayesian_changepoint_detection as bcd  # noqa: E402
from bayesian_changepoint_detection import (  # noqa: E402
    offline_likelihoods,
    online_likelihoods,
)

TCPD_COMMIT = "e8f19a3635e3b7f1a8aff59ce7f4d9bea17525c0"
TCPDBENCH_COMMIT = (
    "167210005c09d2c44f9b2083b95f989a64be4b6b"  # published scores, for comparison only
)
TCPD_RAW = f"https://raw.githubusercontent.com/alan-turing-institute/TCPD/{TCPD_COMMIT}"
TCPDBENCH_SCORES = (
    "https://raw.githubusercontent.com/alan-turing-institute/TCPDBench/"
    f"{TCPDBENCH_COMMIT}/analysis/output/scores"
)
CACHE = os.path.join(HERE, ".cache", "tcpd")

# The series redistributed in the TCPD repository at TCPD_COMMIT.
SERIES = (
    "bank brent_spot businv centralia children_per_woman co2_canada "
    "construction debt_ireland gdp_argentina gdp_croatia gdp_iran gdp_japan "
    "global_co2 homeruns jfk_passengers lga_passengers nile ozone "
    "quality_control_1 quality_control_2 quality_control_3 quality_control_4 "
    "quality_control_5 rail_lines run_log seatbelts shanghai_license "
    "uk_coal_employ unemployment_nl us_population usd_isk well_log"
).split()

ORACLE_GRID = {
    "lam": (10, 50, 100, 200),
    "alpha": (0.01, 0.1, 1.0, 10.0, 100.0),
    "beta": (0.01, 0.1, 1.0, 10.0, 100.0),
    "kappa": (0.01, 0.1, 1.0, 10.0, 100.0),
}


def _fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        with open(path, "wb") as f:
            f.write(payload)
    with open(path, "rb") as f:
        return f.read()


def load_tcpd():
    """``{name: (data [T, d] standardized or None, annotations, n_dim)}``."""
    checksums = json.loads(
        _fetch(f"{TCPD_RAW}/checksums.json", os.path.join(CACHE, "checksums.json"))
    )["checksums"]
    annotations = json.loads(
        _fetch(f"{TCPD_RAW}/annotations.json", os.path.join(CACHE, "annotations.json"))
    )
    series = {}
    for name in SERIES:
        payload = _fetch(
            f"{TCPD_RAW}/datasets/{name}/{name}.json",
            os.path.join(CACHE, f"{name}.json"),
        )
        expected = checksums[f"{name}.json"]
        expected = expected if isinstance(expected, list) else [expected]
        if hashlib.md5(payload).hexdigest() not in expected:
            raise RuntimeError(f"{name}.json does not match TCPD's checksum")
        record = json.loads(payload)
        raw = [s["raw"] for s in record["series"]]
        if any(v is None for column in raw for v in column):
            data = None  # missing values
        else:
            data = np.array(raw, dtype=np.float64).T
            std = data.std(axis=0, ddof=1)
            data = (data - data.mean(axis=0)) / np.where(std > 0, std, 1.0)
        series[name] = (data, annotations[name], record["n_dim"])
    return series


def _online_likelihood(dims, alpha, beta, kappa):
    if dims == 1:
        return online_likelihoods.StudentT(
            alpha=alpha, beta=beta, kappa=kappa, mu=0.0, device="cpu"
        )
    return online_likelihoods.MultivariateT(dims=dims, device="cpu")


def _series_tensor(data):
    return torch.as_tensor(data if data.shape[1] > 1 else data[:, 0])


def online_cps(data, lam=100, alpha=1.0, beta=1.0, kappa=1.0):
    """Filtering readout: forward moves of the MAP-implied segment start."""
    R, _ = bcd.online_changepoint_detection(
        _series_tensor(data),
        partial(bcd.constant_hazard, lam),
        _online_likelihood(data.shape[1], alpha, beta, kappa),
        device="cpu",
    )
    return [int(c) for c in bcd.get_map_changepoints(R)]


def viterbi_cps(data, lam=100, alpha=1.0, beta=1.0, kappa=1.0):
    """MAP segmentation under the same model."""
    _, changepoints = bcd.viterbi_changepoints(
        _series_tensor(data),
        partial(bcd.constant_hazard, lam),
        _online_likelihood(data.shape[1], alpha, beta, kappa),
        device="cpu",
    )
    return [int(c) for c in changepoints]


def offline_cps(data):
    n_obs, dims = data.shape
    x = torch.as_tensor(data if dims > 1 else data[:, 0])
    if dims == 1:
        likelihood = offline_likelihoods.StudentT(device="cpu")
    else:
        likelihood = offline_likelihoods.MultivariateT(dims=dims, device="cpu")
    prior = partial(bcd.const_prior, p=1.0 / (n_obs + 1))
    _, _, Pcp = bcd.offline_changepoint_detection(x, prior, likelihood, device="cpu")
    return offline_starts(torch.exp(Pcp).sum(0).numpy())


def _scores(annotations, cps, n_obs):
    return {
        "f1": f_measure(annotations, cps),
        "cover": covering(annotations, cps, n_obs),
        "cps": cps,
    }


def run(oracle):
    series = load_tcpd()
    published = {}
    for key in ("default_f1", "default_cover", "oracle_f1", "oracle_cover"):
        url = f"{TCPDBENCH_SCORES}/{key}_scores.json"
        scores = json.loads(_fetch(url, os.path.join(CACHE, f"{key}_scores.json")))
        published[key] = {name: scores.get(name, {}) for name in SERIES}

    results = {}
    for name, (data, annotations, n_dim) in series.items():
        entry = {
            "n_obs": None if data is None else int(data.shape[0]),
            "n_dim": n_dim,
            "tcpdbench": {
                key: {m: published[key][name].get(m) for m in ("bocpd", "zero")}
                for key in published
            },
        }
        if data is None:
            entry["skipped"] = "missing values"
            results[name] = entry
            print(f"{name:20} skipped (missing values)", flush=True)
            continue
        n_obs = data.shape[0]
        entry["zero"] = _scores(annotations, [], n_obs)
        entry["online_default"] = _scores(annotations, online_cps(data), n_obs)
        entry["viterbi_default"] = _scores(annotations, viterbi_cps(data), n_obs)
        entry["offline_default"] = _scores(annotations, offline_cps(data), n_obs)
        if oracle:
            best_f1 = best_cover = None
            settings = (
                product(*ORACLE_GRID.values())
                if n_dim == 1
                else [(lam, 1, 1, 1) for lam in ORACLE_GRID["lam"]]
            )
            for lam, alpha, beta, kappa in settings:
                s = _scores(
                    annotations, viterbi_cps(data, lam, alpha, beta, kappa), n_obs
                )
                s["setting"] = {
                    "lam": lam,
                    "alpha": alpha,
                    "beta": beta,
                    "kappa": kappa,
                }
                if best_f1 is None or s["f1"] > best_f1["f1"]:
                    best_f1 = s
                if best_cover is None or s["cover"] > best_cover["cover"]:
                    best_cover = s
            entry["online_oracle_f1"] = best_f1
            entry["online_oracle_cover"] = best_cover
        results[name] = entry
        summary = "  ".join(
            f"{key.split('_')[0]} F1 {entry[key]['f1']:.3f} cover {entry[key]['cover']:.3f}"
            for key in ("viterbi_default", "online_default", "offline_default")
        )
        print(f"{name:20} n={n_obs:4} d={n_dim}  {summary}", flush=True)
    return {
        "environment": {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "library": bcd.__version__,
            "library_commit": _library_commit(),
            "tcpd_commit": TCPD_COMMIT,
            "oracle": oracle,
        },
        "series": results,
    }


def _library_commit():
    import subprocess

    try:
        return subprocess.run(
            [
                "git",
                "-C",
                os.path.dirname(bcd.__file__),
                "describe",
                "--always",
                "--dirty",
                "--tags",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values), len(values)) if values else (float("nan"), 0)


def markdown(report):
    """Averages over the univariate series on which TCPDBench's BOCPD has a
    score, so every column averages the same series."""
    rows = report["series"]
    common = [
        name
        for name, e in rows.items()
        if e["n_dim"] == 1
        and "skipped" not in e
        and e["tcpdbench"]["default_f1"]["bocpd"] is not None
    ]
    columns = [
        ("zero (no changepoints)", lambda e, m: e["zero"][m]),
        (
            "TCPDBench BOCPD, default",
            lambda e, m: e["tcpdbench"][f"default_{m}"]["bocpd"],
        ),
        (
            "this library, online, MAP segmentation (Viterbi), default",
            lambda e, m: e["viterbi_default"][m],
        ),
        (
            "this library, online, filtered MAP run length, default",
            lambda e, m: e["online_default"][m],
        ),
        ("this library, offline, default", lambda e, m: e["offline_default"][m]),
    ]
    if report["environment"]["oracle"]:
        columns += [
            (
                "TCPDBench BOCPD, oracle",
                lambda e, m: e["tcpdbench"][f"oracle_{m}"]["bocpd"],
            ),
            (
                "this library, online, MAP segmentation (Viterbi), oracle",
                lambda e, m: e[f"online_oracle_{m}"][m],
            ),
        ]
    lines = [
        f"Univariate TCPD series redistributed by TCPD, with a TCPDBench BOCPD "
        f"score: {len(common)}. Mean over those series (higher is better).",
        "",
        "| method | F1 | cover |",
        "|---|---|---|",
    ]
    for label, get in columns:
        f1, _ = _mean([get(rows[n], "f1") for n in common])
        cover, _ = _mean([get(rows[n], "cover") for n in common])
        lines.append(f"| {label} | {f1:.3f} | {cover:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.report:
        with open(args.report) as f:
            print(markdown(json.load(f)))
        return
    report = run(args.oracle)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    print()
    print(markdown(report))


if __name__ == "__main__":
    main()
