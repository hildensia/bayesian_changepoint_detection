"""
Time the detectors, and compare released versions on the same workloads.

Every (version, workload, size) case runs in a fresh process (``_worker.py``)
against an exported copy of that version (``git archive <tag>``), so versions
cannot share imports, caches or thread pools. Each case runs a warm-up on a
small series, then ``--repeats`` timed runs on the benchmark series (fewer
when they exceed ``--budget`` seconds); the report gives the median and the
range. Each run is also scored against the true changepoints (F1 with a
margin of 5 observations), so a fast version that misses changes shows up.

Run from the repository root::

    python benchmarks/performance.py                       # full suite, all versions
    python benchmarks/performance.py --versions current    # this checkout only
    python benchmarks/performance.py --suite quick         # seconds; CI smoke run
    python benchmarks/performance.py --report benchmarks/results/<file>.json

Versions: ``current`` (this checkout, uncommitted changes included) and any
git tag, by default ``v1.1.0``, ``v1.0.0`` and ``v0.4`` (the NumPy
implementation, which additionally needs ``scipy`` and ``decorator``).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ALL_VERSIONS = ("current", "v1.1.0", "v1.0.0", "v0.4")

# (workload, dims, sizes, versions that support it)
SUITES = {
    "full": [
        ("offline", 1, (250, 500, 1000, 2000), ALL_VERSIONS),
        ("offline", 5, (250, 500, 1000), ("current", "v1.1.0", "v1.0.0")),
        ("online", 1, (250, 500, 1000, 2000, 5000), ALL_VERSIONS),
        ("online", 5, (250, 500, 1000), ALL_VERSIONS),
        ("streaming", 1, (10_000, 50_000), ("current",)),
    ],
    "quick": [
        ("offline", 1, (100, 200), ALL_VERSIONS),
        ("offline", 5, (100,), ("current", "v1.1.0", "v1.0.0")),
        ("online", 1, (100, 200), ALL_VERSIONS),
        ("online", 5, (100,), ALL_VERSIONS),
        ("streaming", 1, (1000,), ("current",)),
    ],
}


def _git(*args):
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _export(version, workdir):
    """Directory to put on PYTHONPATH for ``version``, and its commit."""
    if version == "current":
        try:
            commit = _git("describe", "--always", "--dirty", "--tags")
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "unknown"
        return REPO, commit
    target = os.path.join(workdir, version)
    os.makedirs(target)
    archive = subprocess.run(
        ["git", "-C", REPO, "archive", version, "bayesian_changepoint_detection"],
        capture_output=True,
        check=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", target], input=archive, check=True)
    return target, _git("rev-parse", "--short", f"{version}^{{commit}}")


def _cpu_name():
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _environment(args):
    import numpy
    import torch

    return {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torch_threads": args.threads or torch.get_num_threads(),
        "device": args.device,
        "repeats": args.repeats,
        "budget_seconds": args.budget,
    }


def _run_case(case, pythonpath):
    env = dict(os.environ, PYTHONPATH=pythonpath)
    completed = subprocess.run(
        [sys.executable, os.path.join(HERE, "_worker.py"), json.dumps(case)],
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1:] or ["no output"]
        return {**case, "error": tail[0]}
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    expected = os.path.realpath(pythonpath)
    if not os.path.realpath(result["module_file"] or "").startswith(expected):
        return {**case, "error": f"imported {result['module_file']}, not {expected}"}
    return result


def run(args):
    versions = args.versions.split(",")
    unknown = set(versions) - set(ALL_VERSIONS) - set(_git("tag").split())
    if unknown:
        sys.exit(f"unknown versions: {sorted(unknown)}")
    report = {"environment": _environment(args), "versions": {}, "results": []}
    with tempfile.TemporaryDirectory() as workdir:
        paths = {}
        for version in versions:
            paths[version], commit = _export(version, workdir)
            report["versions"][version] = commit
        selected = args.workloads.split(",") if args.workloads else None
        for workload, dims, sizes, supported in SUITES[args.suite]:
            if selected and workload not in selected:
                continue
            for version in versions:
                if version not in supported and version in ALL_VERSIONS:
                    continue
                previous = None
                for n_obs in sizes:
                    case = {
                        "version": version,
                        "workload": workload,
                        "dims": dims,
                        "n": n_obs,
                        "device": "cpu" if version == "v0.4" else args.device,
                        "repeats": args.repeats,
                        "budget": args.budget,
                        "threads": args.threads,
                    }
                    if previous is not None and "times" in previous:
                        # Both detectors are O(n^2); skip sizes predicted to
                        # take longer than the budget for a single run.
                        predicted = (
                            min(previous["times"]) * (n_obs / previous["n"]) ** 2
                        )
                        if predicted > args.budget:
                            result = {**case, "skipped": f"predicted {predicted:.0f} s"}
                            report["results"].append(result)
                            print(_line(result), flush=True)
                            continue
                    result = _run_case(case, paths[version])
                    report["results"].append(result)
                    print(_line(result), flush=True)
                    previous = result
    return report


def _line(result):
    head = f"{result['workload']:>9} d={result['dims']} n={result['n']:>6} {result['version']:>8}"
    if "skipped" in result:
        return f"{head}  skipped ({result['skipped']})"
    if "error" in result:
        return f"{head}  error: {result['error']}"
    times = result["times"]
    return (
        f"{head}  median {median(times):8.3f} s  (min {min(times):.3f}, "
        f"max {max(times):.3f}, runs {len(times)})  F1 {result['f1']:.2f}"
    )


def _cell(result):
    if result is None:
        return ""
    if "skipped" in result:
        return "skipped"
    if "error" in result:
        return "error"
    seconds = median(result["times"])
    text = f"{seconds:.3g} s" if seconds >= 0.01 else f"{seconds * 1e3:.2g} ms"
    return f"{text} (F1 {result['f1']:.2f})"


def markdown(report):
    """One table per workload: sizes down, versions across."""
    env = report["environment"]
    lines = [
        f"Measured {env['date']} on {env['cpu']} ({env['platform']}), "
        f"Python {env['python']}, torch {env['torch']}, "
        f"{env['torch_threads']} threads, device `{env['device']}`. "
        "Median wall-clock time of the detector call; F1 against the true "
        "changepoints with a margin of 5 observations.",
        "",
    ]
    versions = list(report["versions"])
    keys = []
    for r in report["results"]:
        key = (r["workload"], r["dims"])
        if key not in keys:
            keys.append(key)
    for workload, dims in keys:
        rows = [
            r
            for r in report["results"]
            if (r["workload"], r["dims"]) == (workload, dims)
        ]
        present = [v for v in versions if any(r["version"] == v for r in rows)]
        lines.append(f"**{workload}, {dims}-D**")
        lines.append("")
        header = ["n"] + [f"{v} (`{report['versions'][v]}`)" for v in present]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for n_obs in sorted({r["n"] for r in rows}):
            cells = [f"{n_obs:,}"]
            for v in present:
                match = [r for r in rows if r["n"] == n_obs and r["version"] == v]
                cells.append(_cell(match[0] if match else None))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if workload != "offline" and any(r.get("mps_hidden") for r in rows):
            lines.append(
                "- v1.0.0 was timed with MPS hidden: on a machine with MPS its "
                'online detector crashes even with `device="cpu"` (fixed in 1.1.0).'
            )
        errors = sorted({(r["version"], r["error"]) for r in rows if "error" in r})
        for version, error in errors:
            lines.append(f"- {version} error: `{error}`")
        if errors:
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=sorted(SUITES), default="full")
    parser.add_argument("--versions", default=",".join(ALL_VERSIONS))
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda (not v0.4)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--budget", type=float, default=300.0, help="seconds per case (soft)"
    )
    parser.add_argument("--threads", type=int, default=None, help="torch threads")
    parser.add_argument(
        "--workloads", help="comma-separated subset: offline, online, streaming"
    )
    parser.add_argument("--output", help="write the JSON report here")
    parser.add_argument("--report", help="print the Markdown tables of a saved report")
    args = parser.parse_args()

    if args.report:
        with open(args.report) as f:
            print(markdown(json.load(f)))
        return
    report = run(args)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    print()
    print(markdown(report))


if __name__ == "__main__":
    main()
