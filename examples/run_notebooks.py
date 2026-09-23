"""
Run the code cells of the example notebooks, headless, in order.

CI uses this instead of Jupyter: it needs no kernel, and a notebook that
drifts from the API fails the build like the example scripts do. IPython
magics (``%timeit``, ``%matplotlib``) and shell escapes are skipped; plots go
to the non-interactive Agg backend.

    python examples/run_notebooks.py [notebook.ipynb ...]
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
NOTEBOOKS = [HERE / "Example_Code.ipynb", HERE / "Multivariate_Example.ipynb"]


def run(path):
    notebook = json.loads(Path(path).read_text())
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        exec(
            compile("\n".join(lines), f"{Path(path).name}, cell {index}", "exec"),
            namespace,
        )
    print(f"ran every code cell of {Path(path).name}")


if __name__ == "__main__":
    for path in sys.argv[1:] or NOTEBOOKS:
        run(path)
