"""What a picture may contain for a given class, board, subject and chapter.

Phase 7 produced a Friction slide whose figure was genuinely about friction
and completely wrong for Class 8: it carried mu_s*N and mu_k*N, vector
notation, and a force-versus-friction graph with static and kinetic regions.
Every automated check passed, because every automated check read *text* and
the problem was in the *pixels*.

Two things follow.

First, deciding this needs someone to look at the image — but the looking and
the deciding are different jobs. A vision model is good at reporting what is
visible and bad at judging whether a syllabus permits it, so it reports
observations and this module makes the decision. That also means the rule can
be corrected without touching a prompt.

Second, the rule has to be a table rather than a Friction special case. The
same failure is waiting in every chapter with a senior-school treatment:
Light has lens-maker formulae, Sound has wave equations, Metals has
electrochemical series. So rules are keyed class -> subject -> chapter with
inheritance, and a chapter inherits its subject's rules and the subject
inherits the class's.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

# Notation that marks a figure as pitched above junior secondary wherever it
# appears. Kept at class level because it is about mathematical maturity
# rather than any one topic.
_CLASS_8_NOTATION: Tuple[str, ...] = (
    "μ", "mu_s", "mu_k", "μs", "μk",
    "∫", "∑", "∂", "√", "Δ", "θ", "λ", "ω", "φ",
    "vector notation", "unit vector", "cross product", "dot product",
    "free body diagram", "free-body diagram",
    "differential", "derivative", "integral", "logarith",
    "sin θ", "cos θ", "tan θ", "sinθ", "cosθ", "tanθ",
)

_CLASS_8_CONCEPTS: Tuple[str, ...] = (
    "coefficient of friction", "newton's second law equation",
    "vector resolution", "force decomposition", "resultant vector",
    "equilibrium equations", "moment of inertia", "angular",
    "logarithmic scale", "exponential decay",
)

# class -> subject -> chapter. A lookup merges the three levels, so a chapter
# entry only has to name what is special about that chapter.
RULES: Dict[str, Dict[str, Dict[str, Dict[str, Sequence[str]]]]] = {
    "8": {
        "_subject_default": {
            "_chapter_default": {
                "forbidden_notation": _CLASS_8_NOTATION,
                "forbidden_concepts": _CLASS_8_CONCEPTS,
                "forbidden_visual_terms": (),
            },
        },
        "science": {
            "_chapter_default": {
                "forbidden_visual_terms": (
                    "calvin cycle", "thylakoid", "krebs", "glycolysis",
                    "electron transport", "atp synthase", "nadph",
                ),
            },
            "friction": {
                # The chapter that produced the defect.
                "forbidden_visual_terms": (
                    "static region", "kinetic region", "limiting friction graph",
                    "friction graph", "f-n graph", "normal reaction equation",
                ),
                "forbidden_notation": ("μsn", "μkn", "fs = μ", "fk = μ",
                                       "f_s,max", "fs,max"),
                "forbidden_concepts": ("coefficient of friction",
                                       "static and kinetic friction graph",
                                       "friction as a function of applied force"),
            },
            "light": {
                "forbidden_visual_terms": ("lens maker", "lens formula",
                                           "1/f = 1/v", "snell", "refractive index",
                                           "critical angle", "dispersion prism graph"),
                "forbidden_concepts": ("total internal reflection",
                                       "lens magnification formula"),
            },
            "sound": {
                "forbidden_visual_terms": ("wave equation", "v = fλ", "v = f lambda",
                                           "doppler", "harmonic series",
                                           "resonance tube formula"),
                "forbidden_concepts": ("frequency wavelength equation",
                                       "standing wave harmonics"),
            },
            "force and pressure": {
                "forbidden_visual_terms": ("free body diagram", "pascal's law derivation",
                                           "hydraulic press equation"),
                "forbidden_concepts": ("vector resolution of forces",)
            },
            "metals and non-metals": {
                "forbidden_visual_terms": ("electrochemical series",
                                           "standard electrode potential",
                                           "half cell", "galvanic cell"),
            },
            "cell structure and functions": {
                "forbidden_visual_terms": ("endoplasmic reticulum lumen",
                                           "golgi cisternae", "ribosome subunit",
                                           "mitochondrial cristae"),
            },
        },
    },
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _class_key(class_name: str) -> str:
    m = re.search(r"(\d{1,2})", str(class_name or ""))
    return m.group(1) if m else ""


def lookup(class_name: str = "", subject: str = "", chapter: str = "") -> Dict[str, List[str]]:
    """Merged rules for one slide's audience.

    Levels are merged rather than overridden: a chapter adds to what its
    subject forbids, and a subject adds to what the class forbids. A chapter
    entry therefore only names what is special about that chapter.
    """
    merged: Dict[str, List[str]] = {"forbidden_notation": [],
                                    "forbidden_concepts": [],
                                    "forbidden_visual_terms": []}
    cls = RULES.get(_class_key(class_name))
    if not cls:
        return merged

    subject_key = _norm(subject)
    chapter_key = _norm(chapter)

    def absorb(block):
        for field in merged:
            for value in (block or {}).get(field, ()) or ():
                v = _norm(value)
                if v and v not in merged[field]:
                    merged[field].append(v)

    for subject_name in ("_subject_default", subject_key):
        subject_block = cls.get(subject_name) or {}
        for chapter_name in ("_chapter_default", chapter_key):
            absorb(subject_block.get(chapter_name))
    return merged


def all_forbidden(class_name: str = "", subject: str = "",
                  chapter: str = "") -> List[str]:
    """Everything this audience must not see, as one flat list."""
    rules = lookup(class_name, subject, chapter)
    out: List[str] = []
    for field in ("forbidden_notation", "forbidden_concepts",
                  "forbidden_visual_terms"):
        for value in rules[field]:
            if value not in out:
                out.append(value)
    return out


def violations(observations: Iterable[str], class_name: str = "",
               subject: str = "", chapter: str = "") -> List[str]:
    """Which forbidden things the vision model reported seeing.

    Matching is forward only - the rule must appear inside what was
    observed - because the model paraphrases outward: asked about
    "coefficient of friction" it reports "coefficients of friction shown as
    mu_s and mu_k", which contains the rule.

    The reverse direction was tried and is wrong: it let the single word
    "friction" match the rule "coefficient of friction", so a perfectly
    ordinary Class 8 friction figure was rejected for being advanced.
    """
    forbidden = all_forbidden(class_name, subject, chapter)
    if not forbidden:
        return []
    seen = [_norm(o) for o in observations if _norm(o)]
    if not seen:
        return []

    found: List[str] = []
    for rule in forbidden:
        for observation in seen:
            if _rule_matches(rule, observation):
                if rule not in found:
                    found.append(rule)
                break
    return found


_STOP = frozenset(("of", "and", "the", "a", "an", "as", "in", "to", "with"))


def _rule_matches(rule: str, observation: str) -> bool:
    """Does `observation` report the thing `rule` forbids?

    Symbols and short tokens ("mu_s", "μ") are matched as substrings. A
    multi-word rule is matched on its significant WORDS, each as a prefix,
    because the model paraphrases and inflects: "coefficient of friction" has
    to match "coefficients of friction shown as mu_s" (plural), while the
    bare word "friction" must still NOT match it - which a plain substring
    test got wrong in both directions.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", rule) if w and w not in _STOP]
    if len(words) < 2:
        return rule in observation
    obs_words = [w for w in re.split(r"[^a-z0-9]+", observation) if w]
    return all(any(o.startswith(w) or w.startswith(o) for o in obs_words)
               for w in words)


def describe(class_name: str = "", subject: str = "", chapter: str = "") -> str:
    """A short line for the vision prompt, naming the audience."""
    parts = [p for p in (class_name, subject, chapter) if p]
    return " / ".join(parts) if parts else "(not stated)"
