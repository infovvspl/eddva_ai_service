"""
Isolated runner for LLM-generated PLOTTING code (see ai_services.core.diagram).

Executed as a SEPARATE process by diagram._execute_plot_code:

    python -m ai_services.core._plot_runner   < {"code": "..."}

WHY A SEPARATE RUNNER FROM solver._sandbox_runner
That one pre-populates the full scientific namespace — sympy, scipy, rdkit,
pint, pubchempy — because a solver may need any of them. Importing rdkit alone
costs about 11 seconds, which consumed the entire execution timeout before a
single line of plotting code ran: every diagram timed out.

A figure needs numpy and matplotlib, and those import in well under half a
second. Keeping this runner separate also keeps the solver's behaviour
untouched, and gives plotting a much smaller surface than a namespace holding
network- and filesystem-capable libraries it has no use for.

The process-level hardening is the caller's (hard wall-clock timeout,
secret-stripped environment, POSIX rlimits) and is identical to the solver's.
This module reads a JSON payload {"code": <str>} on stdin and writes one JSON
result line to stdout prefixed with RESULT_SENTINEL. It has no side effects on
import.
"""

import base64
import io
import json
import sys
import traceback

RESULT_SENTINEL = "__PLOT_RESULT__"

#: Refuse to return an unreasonably large PNG. A textbook-style line drawing is
#: tens of kilobytes; megabytes means the code drew something pathological.
_MAX_PNG_BYTES = 6 * 1024 * 1024

#: One question gets one figure. More means the code did something other than
#: what was asked, and the extras are dropped rather than guessed between.
_MAX_FIGURES = 1


def _build_namespace(stdout_capture):
    """numpy and matplotlib only — everything a figure needs, nothing else.

    matplotlib is REQUIRED here and its import error is reported rather than
    swallowed. Treating it as optional (as the solver's runner does, where it
    genuinely is) turned a plain environment problem into the opaque message
    "matplotlib is unavailable", with the real cause — matplotlib failing to
    resolve a home directory for its config — nowhere in the output.
    """
    ns = {
        "__builtins__": __builtins__,
        "print": lambda *args: stdout_capture.write(" ".join(map(str, args)) + "\n"),
    }
    try:
        exec(
            "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
            ns,
        )
    except Exception as exc:
        return ns, f"matplotlib could not be loaded: {exc}"

    # Optional: a figure can be drawn without either, so neither is fatal.
    # sympy is cheap (~0.25s) and occasionally useful for plotting an
    # expression a question states symbolically.
    for stmt in ("import numpy as np", "import sympy as sp"):
        try:
            exec(stmt, ns)
        except Exception:
            pass
    return ns, None


def _run(code: str) -> dict:
    stdout_capture = io.StringIO()
    ns, import_error = _build_namespace(stdout_capture)
    plt = ns.get("plt")
    if plt is None:
        return {"success": False, "stdout": "", "graphs": [],
                "error": import_error or "matplotlib is unavailable in the plot sandbox"}
    try:
        # Any figure left open by a previous run in this process would be
        # collected as if this run had drawn it. The process is fresh, but the
        # close is cheap insurance against that becoming untrue.
        plt.close("all")
        exec(code, ns)

        graphs = []
        for number in plt.get_fignums()[:_MAX_FIGURES]:
            figure = plt.figure(number)
            buf = io.BytesIO()
            figure.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            raw = buf.getvalue()
            buf.close()
            if not raw or len(raw) > _MAX_PNG_BYTES:
                continue
            graphs.append("data:image/png;base64," + base64.b64encode(raw).decode("utf-8"))

        return {"success": True, "stdout": stdout_capture.getvalue(),
                "graphs": graphs, "error": None}
    except Exception as exc:
        return {"success": False, "stdout": stdout_capture.getvalue(), "graphs": [],
                "error": f"{exc}\n{traceback.format_exc()}"}
    finally:
        try:
            plt.close("all")
        except Exception:
            pass


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = _run(payload.get("code", ""))
    except Exception as exc:
        result = {"success": False, "stdout": "", "graphs": [],
                  "error": f"plot runner error: {exc}"}
    sys.stdout.write(RESULT_SENTINEL + json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
