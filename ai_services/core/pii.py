"""Keep directly-identifying fields out of third-party model prompts.

A student's name adds nothing to the quality of an analysis — the model reasons
about marks, attendance and interest scores, and uses the name only to address
the reader. So the name is replaced with a sentinel on the way out and restored
on the way back: the provider sees an anonymous profile, and the student still
reads their own name.

The sentinel is deliberately ugly rather than name-like. If restoration ever
fails, the reader sees an obviously broken placeholder instead of a stranger's
name, which would look like one student's report leaking into another's.
"""
import re

# Chosen to survive JSON encoding and tokenisation intact, and to be something
# no model would produce spontaneously.
STUDENT_TOKEN = "__STUDENT__"

# Values that carry no identity, so there is nothing to protect and nothing to
# restore. Compared case-insensitively after stripping.
_PLACEHOLDER_NAMES = {"", "student", "the student", "unknown", "n/a", "na", "none"}

# Models occasionally reformat a sentinel — dropping an underscore, adding
# spaces, changing case. Matching those variants keeps restoration working
# instead of leaving debris in a student-facing report.
_TOKEN_VARIANTS = re.compile(
    r"_{1,3}\s*STUDENT\s*_{1,3}",
    re.IGNORECASE,
)


def needs_pseudonym(name) -> bool:
    """Whether this value is a real name worth protecting."""
    return str(name or "").strip().lower() not in _PLACEHOLDER_NAMES


def to_prompt_name(name) -> str:
    """The value to put in a prompt in place of the real name.

    A generic profile keeps its generic label; a real name becomes the sentinel.
    """
    return STUDENT_TOKEN if needs_pseudonym(name) else "Student"


def restore_name(value, name):
    """Put the real name back everywhere the sentinel survived.

    Walks nested dicts and lists because the career report is a structured
    object, not a flat string — the name can appear in any section the model
    chose to personalise. Returns `value` untouched when there is no real name
    to restore, so callers can apply this unconditionally.
    """
    if not needs_pseudonym(name):
        return value

    real = str(name).strip()

    def walk(node):
        if isinstance(node, str):
            return _TOKEN_VARIANTS.sub(real, node)
        if isinstance(node, dict):
            # Keys can carry the sentinel too if the model built a keyed
            # section per person.
            return {walk(k) if isinstance(k, str) else k: walk(v)
                    for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(walk(v) for v in node)
        return node

    return walk(value)


def contains_token(value) -> bool:
    """Whether any sentinel is still present — used by tests and diagnostics."""
    if isinstance(value, str):
        return bool(_TOKEN_VARIANTS.search(value))
    if isinstance(value, dict):
        return any(contains_token(k) or contains_token(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_token(v) for v in value)
    return False
