"""
Draw the diagram a question needs, when the textbook does not contain it.

WHY THIS EXISTS
textbook_figures covers diagrams the school's own book actually prints. It
cannot cover the far more common exam case: a question whose figure is invented
along with the question. "Show that A(1,1), B(4,1), C(4,5) are the vertices of a
right-angled triangle" needs a specific triangle on a specific grid, and no
textbook contains that exact picture — but the question's own data determines it
completely, so it can be DRAWN rather than found.

That covers a large share of school papers: coordinate geometry, graphs of
functions, distance-time and velocity-time plots, number lines, bar charts,
histograms, pie charts, angle and triangle constructions.

Drawing beats searching here on every axis that matters. The figure is correct by
construction rather than approximately relevant, it matches the question's own
numbers exactly, and it raises no copyright question at all — which a web image
on a printed exam paper does.

HOW
The paper generator writes a short spec — [PLOT: ...] — and this turns it into
matplotlib code, runs that code in the EXISTING hardened sandbox
(ai_services.solver._sandbox_runner, reached through
ScientificSolver._execute_code), and returns the rendered PNG.

The sandbox is a separate process with a hard wall-clock timeout, a
secret-stripped environment, and POSIX rlimits on CPU, memory and file size. It
is the security boundary and it is not reimplemented here.

DEFENCE IN DEPTH
The generated code is also screened by AST before it is ever launched: no
imports, no attribute escapes, no builtins that touch the filesystem, the
network or the interpreter. The sandbox would contain a hostile script anyway;
this makes sure an obviously wrong one never runs, and turns a class of silent
weirdness into a clear rejection.
"""
import ast
import base64
import io
import logging
import os
import re

logger = logging.getLogger("ai_services.diagram")

#: Model that writes the plotting code. The same one the scientific solver uses
#: to write solver code — this is the same task, with a narrower output.
_DIAGRAM_MODEL = os.getenv("DIAGRAM_CODE_MODEL", "openai/gpt-oss-120b")

#: A spec longer than this is not a diagram description, it is a question that
#: leaked into the marker.
_MAX_SPEC_CHARS = 400

#: Generated plotting code is short by nature. A long script is a sign the model
#: is doing something other than drawing.
_MAX_CODE_CHARS = 4000

#: One retry, with the failure fed back. Beyond that the paper is better off
#: without the figure than waiting on a model that cannot draw it.
_MAX_ATTEMPTS = 2

#: A rendered figure that is almost entirely blank means the code ran but drew
#: nothing — a silent failure that would otherwise reach a student as an empty
#: white box on their exam paper.
_MIN_INK_FRACTION = 0.004

# --- classroom legibility -------------------------------------------------
# The figure used to be drawn at matplotlib's default 6.4x4.8in and then
# fitted into its slide region. For a process band of 8.9x1.69in that is a
# 0.35x reduction, so the prompt's "fontsize 10 or more" floor arrived on the
# slide at 3.5pt. Nothing about the code or the prompt was wrong; the scale
# was. The fix is to draw the figure at exactly the size it will occupy, so
# the point size the code sets is the point size the class sees.
#
# 13pt is the floor for a projected slide. Below roughly that, a figure label
# is unreadable from the back of a room even though it looks fine in a
# screenshot viewed at arm's length.
_MIN_SLIDE_PT = 13.0

# A drawing crammed with labels is unreadable however large the type, and the
# band is short. Derived from the area actually available rather than fixed.
_MIN_LABEL_BUDGET = 4
_MAX_LABEL_BUDGET = 10

# Ink above this fraction means the canvas is a solid mass rather than a
# diagram - dense hatching, an oversized fill, or a wall of text.
_MAX_INK_FRACTION = 0.55

#: Everything the sandbox namespace already provides. Generated code needs no
#: imports at all, so any import is a red flag rather than a convenience.
_ALLOWED_NAMES = {
    "np", "sp", "plt", "matplotlib", "integrate", "optimize", "stats",
    "print", "range", "len", "abs", "min", "max", "sum", "round", "sorted",
    "enumerate", "zip", "list", "tuple", "dict", "set", "str", "int", "float",
    "bool", "True", "False", "None", "map", "filter", "reversed", "any", "all",
    "divmod", "pow",
}

#: Attribute names that are the classic sandbox-escape path.
_FORBIDDEN_ATTRS = {
    "__globals__", "__builtins__", "__subclasses__", "__class__", "__bases__",
    "__mro__", "__code__", "__closure__", "__dict__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__import__", "__loader__", "__spec__",
}

#: Builtins that reach the filesystem, the network or the interpreter.
_FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "input", "__import__", "exit", "quit",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
    "breakpoint", "help",
}

_SYSTEM_PROMPT = (
    "You write matplotlib code that draws a single figure for a school exam "
    "question. You output CODE ONLY — no prose, no markdown fences, no "
    "explanation.\n\n"
    "RULES\n"
    "1. NO import statements. numpy is already available as np, sympy as sp, "
    "matplotlib.pyplot as plt.\n"
    "1b. Output raw Python statements only. Do NOT wrap them in braces, "
    "brackets, quotes, JSON or a function.\n"
    "2. Create exactly ONE figure. Never call plt.show() and never save to a "
    "file.\n"
    "3. Draw ONLY what the specification describes. Never annotate the answer "
    "to the question — no computed lengths, no areas, no solution values. The "
    "figure is the question's given information, not its solution.\n"
    "4. Label what the specification names (points, axes, units). Keep labels "
    "large enough to read in print: fontsize 10 or more.\n"
    "5. Use plain black-on-white line art in the style of a printed textbook "
    "figure. No seaborn styles, no colour gradients, no decorative titles.\n"
    "6. Set axis limits explicitly so nothing is clipped, and call "
    "plt.tight_layout() at the end.\n"
    "7. Use only plain arithmetic and the libraries above. No file access, no "
    "network, no os or sys."
)


class DiagramError(RuntimeError):
    """The figure could not be drawn. The caller should omit it, not retry."""


def _unwrap_braces(text: str) -> str:
    """Drop a `{ ... }` wrapper the model put around otherwise-valid code.

    gpt-oss-120b returns the plotting code wrapped in braces even with
    json_mode off — good code, made unparseable by two characters, and every
    diagram failed with "invalid syntax (line 2)".

    The braces are removed ONLY when what is inside them parses as Python, so
    this can never turn a genuine dict literal into something else.
    """
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text
    inner = stripped[1:-1].strip()
    if not inner:
        return text
    try:
        ast.parse(inner)
    except SyntaxError:
        return text
    return inner


def _clean_code(content: str) -> str:
    """Strip markdown fences, brace wrappers and any prose around the code."""
    text = str(content or "").strip()
    fence = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    # plt.show() blocks on a non-interactive backend in some configurations and
    # is never wanted here; the sandbox collects figures itself.
    lines = [ln for ln in lines if "plt.show()" not in ln]
    return _unwrap_braces("\n".join(lines).strip()).strip()


def screen_code(code: str) -> "str | None":
    """Return a reason to refuse this code, or None if it may run.

    An allowlist over the AST, not a blocklist over the text: string matching
    for "import os" is trivially defeated, and this has to be the kind of check
    that cannot be written around.
    """
    if not code.strip():
        return "empty code"
    if len(code) > _MAX_CODE_CHARS:
        return f"code too long ({len(code)} chars)"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed"
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            return f"forbidden attribute: {node.attr}"
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Only screen calls and known-dangerous builtins; a local variable
            # named `data` is fine and must not be rejected.
            if node.id in _FORBIDDEN_CALLS:
                return f"forbidden builtin: {node.id}"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                return f"forbidden call: {func.id}"
            if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_CALLS:
                return f"forbidden call: {func.attr}"
    return None


def _execute_plot_code(code: str) -> dict:
    """Run plotting code in an isolated process and collect the figure.

    The hardening mirrors ScientificSolver._execute_code exactly — hard
    wall-clock timeout, secret-stripped environment, POSIX rlimits, own process
    group so the timeout kill takes children too. What differs is the child:
    core._plot_runner loads numpy and matplotlib rather than the full scientific
    namespace, because importing rdkit for a plot costs ~11 seconds and timed
    out every diagram before its first line ran.

    Disable entirely with DIAGRAM_EXEC_ENABLED=false.
    """
    import json
    import subprocess
    import sys

    if os.getenv("DIAGRAM_EXEC_ENABLED", "true").lower() not in ("true", "1", "yes"):
        return {"success": False, "graphs": [],
                "error": "diagram rendering is disabled (DIAGRAM_EXEC_ENABLED=false)"}

    timeout_s = int(os.getenv("DIAGRAM_EXEC_TIMEOUT", "20"))
    # project root = .../ai_services/core/diagram.py -> up 3
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # matplotlib resolves its config directory through Path.home() unless told
    # otherwise, and a secret-stripped environment does not carry whatever the
    # platform needs for that (HOME on POSIX, USERPROFILE on Windows) — it
    # raised "Could not determine home directory" and the import failed before
    # any code ran. MPLCONFIGDIR settles it on every platform, and pointing it
    # at a STABLE directory also keeps the font cache between runs; a fresh
    # directory each time makes matplotlib rebuild that cache per diagram.
    import tempfile
    mpl_config = os.path.join(tempfile.gettempdir(), "eddva-mplconfig")
    try:
        os.makedirs(mpl_config, exist_ok=True)
    except Exception:
        mpl_config = tempfile.gettempdir()

    # Deliberately excludes GROQ_*, GEMINI_*, DB_*, REDIS_URL and everything
    # else, so generated code cannot read a credential even if it tries.
    clean_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": base_dir,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": mpl_config,
        "HOME": os.environ.get("HOME") or tempfile.gettempdir(),
    }
    for key in ("SYSTEMROOT", "TEMP", "TMP", "LD_LIBRARY_PATH", "USERPROFILE"):
        if os.environ.get(key):
            clean_env[key] = os.environ[key]

    preexec = None
    if os.name == "posix":
        def _apply_limits():
            import resource
            cpu = timeout_s + 2
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            memory = 2 * 1024 * 1024 * 1024        # 2 GB address space
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            file_size = 25 * 1024 * 1024           # 25 MB max file write
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
            os.setsid()
        preexec = _apply_limits

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ai_services.core._plot_runner"],
            input=json.dumps({"code": code}),
            capture_output=True, text=True,
            timeout=timeout_s, cwd=base_dir, env=clean_env,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "graphs": [], "error": f"drawing timed out after {timeout_s}s"}
    except Exception as exc:
        return {"success": False, "graphs": [], "error": f"plot sandbox launch failed: {exc}"}

    from ai_services.core._plot_runner import RESULT_SENTINEL
    out = proc.stdout or ""
    index = out.rfind(RESULT_SENTINEL)
    if index == -1:
        return {"success": False, "graphs": [],
                "error": f"plot sandbox produced no result (rc={proc.returncode}): "
                         f"{(proc.stderr or '')[:400]}"}
    try:
        return json.loads(out[index + len(RESULT_SENTINEL):])
    except Exception as exc:
        return {"success": False, "graphs": [], "error": f"could not parse plot result: {exc}"}


def _ink_fraction(png_bytes: bytes) -> float:
    """Fraction of non-paper pixels — the blank-figure guard."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:                                  # pragma: no cover
        return 1.0
    try:
        with Image.open(io.BytesIO(png_bytes)) as image:
            arr = np.array(image.convert("L"))
    except Exception:
        return 0.0
    if arr.size == 0:
        return 0.0
    return float((arr < 245).sum()) / float(arr.size)


def label_budget(width_in, height_in):
    """How many text labels this much canvas can carry legibly."""
    area = max(float(width_in or 0) * float(height_in or 0), 0.0)
    return max(_MIN_LABEL_BUDGET, min(_MAX_LABEL_BUDGET, int(round(area / 2.0))))


def _preamble(width_in, height_in, base_pt, strict=False, axes=True):
    """rcParams that make the figure land on the slide at a readable size.

    Set as defaults rather than by rewriting the model's code: whatever
    figure the code creates inherits them, so the constraint holds without
    fighting the generated code for control of the figure.
    """
    lw = 2.6 if strict else 2.0
    rc = {
        'figure.figsize': (round(float(width_in), 3), round(float(height_in), 3)),
        'figure.dpi': 150,
        'font.size': base_pt,
        'axes.titlesize': base_pt + 1,
        'axes.labelsize': base_pt,
        'xtick.labelsize': base_pt - 1,
        'ytick.labelsize': base_pt - 1,
        'legend.fontsize': base_pt - 1,
        'lines.linewidth': lw,
        'lines.markersize': lw * 3.0,
        'patch.linewidth': lw,
        'axes.linewidth': lw,
        'xtick.major.width': lw,
        'ytick.major.width': lw,
        'savefig.pad_inches': 0.02,
    }
    if not axes:
        # A labelled-structure figure is not a graph. Left to the defaults,
        # matplotlib draws spines and 0-2-4-6-8-10 tick labels around a plant
        # cell, which a rasterised benchmark slide showed doing exactly that.
        # Switched off here rather than asked for in the prompt, because the
        # model forgets and the renderer should not have to care.
        rc.update({
            'axes.spines.left': False, 'axes.spines.right': False,
            'axes.spines.top': False, 'axes.spines.bottom': False,
            'xtick.bottom': False, 'xtick.labelbottom': False,
            'ytick.left': False, 'ytick.labelleft': False,
            'xtick.top': False, 'ytick.right': False,
        })
    # _SAVE_BBOX: the caller has sized the figure to its slide region, so
    # the saved geometry must be kept exactly. A tight crop would change
    # the aspect according to how sparse the drawing happened to be.
    return "plt.rcParams.update(%r)\n_SAVE_BBOX = None\n" % (rc,)

def raise_small_fonts(code, floor=_MIN_SLIDE_PT):
    """Lift any undersized fontsize in the code up to the floor.

    The model reliably writes `fontsize=12` out of print-figure habit.
    Rejecting the figure for it threw away an otherwise good drawing and
    usually failed the retry the same way, so the slide lost its diagram over
    one number. The preamble's defaults are already right for the region, so
    this is a mechanical repair, not a judgement: only things that need the
    figure redrawn - too many labels, a legend - are left as rejections.
    """
    def lift(match):
        value = float(match.group(2))
        if value >= floor:
            return match.group(0)
        return "%s=%g" % (match.group(1), floor)

    return re.sub(r"(fontsize|labelsize)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                  lift, code)


def _legibility_problem(code, width_in, height_in, allow_text=True):
    """Why this code would draw an unreadable figure, or None.

    Checked on the code rather than the pixels because text size and label
    count are stated there explicitly, and measuring glyph heights out of a
    PNG is far more work for a worse answer.
    """
    budget = label_budget(width_in, height_in) if allow_text else 0
    labels = len(re.findall(r"\.(?:text|annotate)\s*\(", code))
    labels += len(re.findall(r"plt\.(?:text|annotate)\s*\(", code))
    if not allow_text and labels:
        return ("the figure draws %d text label(s); this figure must contain "
                "no text at all because the labels are placed outside it"
                % labels)
    if labels > budget:
        return ("the figure has %d labels but only about %d fit legibly in a "
                "%.1f by %.1f inch space - draw fewer, larger elements"
                % (labels, budget, width_in, height_in))

    if re.search(r"\.legend\s*\(", code) and (width_in * height_in) < 12:
        return ("a legend does not fit in this space - label the parts "
                "directly on the drawing instead")
    return None


def _build_user_prompt(spec, subject, class_name, board, previous_error="",
                       width_in=None, height_in=None, label_text=True):
    context = ", ".join(
        part for part in (
            (board or "").strip().upper(), (class_name or "").strip(), (subject or "").strip(),
        ) if part
    )
    lines = []
    if context:
        lines.append(f"This figure is for a {context} examination paper.")
    lines.append(f"Draw exactly this: {spec.strip()}")
    if width_in and height_in:
        budget = label_budget(width_in, height_in)
        # The size is stated because the model has no other way to know it,
        # and because the figure is now placed at the size it is drawn: the
        # old pipeline shrank it ~0.35x, turning a 10pt label into 3.5pt.
        guidance = [
            ('This figure will be shown at %.1f by %.1f inches on a projected '
             'slide, at the size you draw it - it is NOT scaled down '
             'afterwards. So:') % (width_in, height_in),
            '- do NOT set figure.figsize, fontsize or linewidth; the defaults '
            'are already set for this size and are the ones that stay readable.',
            ('- draw at most %d text labels. Fewer, larger elements beat a '
             'complete but unreadable drawing.') % budget,
            '- no legend; label the parts directly on the drawing.',
            '- fill the width you have been given rather than drawing a small '
            'figure in the middle of it.',
        ]
        if not label_text:
            guidance = [g for g in guidance if 'text labels' not in g]
            guidance.append(
                '- draw NO text at all: no labels, no titles, no annotations. '
                'The labels are added outside the figure afterwards, where '
                'they can be placed without colliding.')
        lines.append(chr(10).join([''] + guidance))
    if previous_error:
        # The previous attempt is fed back rather than simply retried: an
        # unguided retry of a deterministic model reproduces the same failure.
        lines.append(
            "\nYour previous attempt was rejected for this reason, so fix it:\n"
            f"{previous_error}"
        )
    lines.append("\nOutput the matplotlib code only.")
    return "\n".join(lines)


def render_diagram(spec, subject="", class_name="", board="",
                   width_in=None, height_in=None, axes=True, labels=True):
    """Draw `spec` and return it as a PNG data URI.

    `axes=False` suppresses spines and tick labels, for a figure that is a
    drawing rather than a plot.

    `labels=False` asks for an unlabelled figure, for the case where the
    caller places the labels itself. That is the better arrangement whenever
    it applies: matplotlib has no idea where it has already put text, and a
    rasterised slide showed two of its labels printed over each other.

    `width_in`/`height_in` are the size the figure will actually occupy where
    it is placed. Given them, the figure is drawn at exactly that size, so
    the point sizes in the code are the point sizes the class sees, and the
    result is checked for classroom legibility before it is accepted.
    Omitting them keeps the original behaviour for existing callers.

    Returns {"success", "image_base64", "code", "error", "attempts"}. Never
    raises for an ordinary failure — a paper is better off missing one figure
    than failing to generate.
    """
    spec = str(spec or "").strip()
    if not spec:
        return {"success": False, "image_base64": None, "code": "",
                "error": "empty specification", "attempts": 0}
    if len(spec) > _MAX_SPEC_CHARS:
        return {"success": False, "image_base64": None, "code": "",
                "error": f"specification too long ({len(spec)} chars)", "attempts": 0}

    try:
        from ai_services.core.llm_client import get_llm
    except Exception as exc:
        logger.warning("Diagram rendering unavailable: %s", exc)
        return {"success": False, "image_base64": None, "code": "",
                "error": f"unavailable: {exc}", "attempts": 0}

    llm = get_llm()
    previous_error = ""
    code = ""
    sized = bool(width_in and height_in)

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(spec, subject, class_name, board,
                                               previous_error, width_in,
                                               height_in, labels),
                model=_DIAGRAM_MODEL,
                json_mode=False,
            )
            code = _clean_code((response or {}).get("content", ""))
        except Exception as exc:
            logger.warning("Diagram code generation failed (attempt %d): %s", attempt, exc)
            previous_error = str(exc)[:300]
            continue

        refusal = screen_code(code)
        if refusal:
            logger.warning("Diagram code rejected (attempt %d): %s", attempt, refusal)
            previous_error = refusal
            continue

        if sized:
            code = raise_small_fonts(code)
            problem = _legibility_problem(code, width_in, height_in,
                                          allow_text=labels)
            if problem:
                # Retried once with the reason fed back, then the caller falls
                # through to the next visual strategy. A technically valid but
                # unreadable figure is worse than no figure.
                logger.info("Diagram rejected as unreadable (attempt %d): %s",
                            attempt, problem)
                previous_error = problem
                continue
            # Defaults, not a rewrite: whatever figure the code creates
            # inherits the size and type scale this slide region needs.
            runnable = _preamble(width_in, height_in, _MIN_SLIDE_PT,
                                 strict=attempt > 1, axes=axes) + code
        else:
            runnable = code

        result = _execute_plot_code(runnable)
        if not result.get("success"):
            previous_error = str(result.get("error") or "execution failed")[:300]
            logger.warning("Diagram execution failed (attempt %d): %s", attempt, previous_error[:160])
            continue

        graphs = result.get("graphs") or []
        if not graphs:
            previous_error = "the code produced no figure"
            logger.warning("Diagram produced no figure (attempt %d)", attempt)
            continue

        data_uri = graphs[0]
        comma = data_uri.find(",")
        try:
            png = base64.b64decode(data_uri[comma + 1:]) if comma >= 0 else b""
        except Exception:
            png = b""
        if not png:
            previous_error = "the figure could not be decoded"
            continue
        ink = _ink_fraction(png)
        if sized and ink > _MAX_INK_FRACTION:
            previous_error = ("the figure is a solid mass of ink rather than a "
                              "diagram - draw fewer, simpler shapes")
            logger.info("Diagram too dense (attempt %d, ink=%.2f)", attempt, ink)
            continue
        if ink < _MIN_INK_FRACTION:
            # Ran, drew nothing. Without this the student gets an empty white
            # box on their paper where the diagram should be.
            previous_error = "the figure came out blank"
            logger.warning("Diagram came out blank (attempt %d)", attempt)
            continue

        logger.info("Diagram rendered in %d attempt(s): %r", attempt, spec[:80])
        return {"success": True, "image_base64": data_uri, "code": code,
                "error": None, "attempts": attempt}

    return {"success": False, "image_base64": None, "code": code,
            "error": previous_error or "could not draw this figure",
            "attempts": _MAX_ATTEMPTS}
