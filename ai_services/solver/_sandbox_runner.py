"""
Isolated runner for LLM-generated solver code (security boundary for the
scientific solver — see docs/PRODUCTION_AUDIT.md, finding C1).

This module is executed as a SEPARATE process by scientific_solver._execute_code:

    python -m ai_services.solver._sandbox_runner   < {"code": "..."}

Why a separate process (not in-process exec):
  - A hard wall-clock timeout can actually kill runaway/infinite-loop code
    (an in-process thread cannot be force-killed).
  - The parent launches us with a secret-stripped environment, so generated
    code cannot read GROQ/DB/REDIS credentials from os.environ.
  - On POSIX the parent additionally applies CPU/memory/file-size rlimits.

It reads a JSON payload {"code": <str>} on stdin, executes it with the standard
scientific namespace pre-populated, and writes a single JSON result line to
stdout, prefixed with RESULT_SENTINEL so the parent can find it amid any library
chatter. It must have no side effects on import.
"""

import sys
import io
import json
import base64
import traceback

RESULT_SENTINEL = "__SANDBOX_RESULT__"


def _build_namespace(stdout_capture):
    """Pre-populate the scientific namespace, importing each lib defensively.

    A missing optional library (e.g. rdkit not installed) must not abort the
    whole run — code that doesn't use it should still execute.
    """
    ns = {
        "__builtins__": __builtins__,
        "print": lambda *args: stdout_capture.write(" ".join(map(str, args)) + "\n"),
    }
    imports = [
        "import numpy as np",
        "import sympy as sp",
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
        "from scipy import integrate, optimize, stats",
        "from rdkit import Chem",
        "from rdkit.Chem import Descriptors, AllChem",
        "import pint",
        "import pubchempy as pcp",
        "ureg = pint.UnitRegistry()",
        ("from ai_services.solver.scientific_solver import "
         "RuleBasedChiralDetector as ChiralDetector, StepValidator, "
         "OpenBabelFallback as OpenBabel, MaximaFallback as Maxima"),
    ]
    for stmt in imports:
        try:
            exec(stmt, ns)
        except Exception:
            # Optional dependency unavailable — leave it out of the namespace.
            pass
    return ns


def _run(code: str) -> dict:
    stdout_capture = io.StringIO()
    ns = _build_namespace(stdout_capture)
    results = {}
    plt = ns.get("plt")
    try:
        if plt is not None:
            plt.clf()
        exec(code, ns)

        excluded = {
            "__builtins__", "print", "np", "sp", "plt", "matplotlib",
            "integrate", "optimize", "stats", "Chem", "Descriptors",
            "AllChem", "pint", "pcp", "ureg", "ChiralDetector",
            "StepValidator", "OpenBabel", "Maxima",
        }
        for k, v in ns.items():
            if k in excluded or k.startswith("_"):
                continue
            try:
                if isinstance(v, (int, float, str, list, dict, bool, type(None))):
                    json.dumps(v)
                    results[k] = v
                else:
                    results[k] = str(v)
            except Exception:
                results[k] = str(v)

        graphs = []
        if plt is not None:
            for i in plt.get_fignums():
                fig = plt.figure(i)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                graphs.append("data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8"))
                buf.close()

        return {"success": True, "stdout": stdout_capture.getvalue(),
                "results": results, "graphs": graphs, "error": None}
    except Exception as e:
        return {"success": False, "stdout": stdout_capture.getvalue(),
                "results": results, "graphs": [],
                "error": f"{e}\n{traceback.format_exc()}"}


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        code = payload.get("code", "")
        result = _run(code)
    except Exception as e:
        result = {"success": False, "stdout": "", "results": {},
                  "graphs": [], "error": f"sandbox runner error: {e}"}
    sys.stdout.write(RESULT_SENTINEL + json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
