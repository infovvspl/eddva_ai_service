"""One optional vision check on the single image a slide actually chose.

Deterministic relevance (``core.ppt_v2.relevance``) reads captions and hosts.
That catches the picture that is about the wrong thing, and it is free. It
cannot catch the picture that is about the *right* thing and still teaches
nothing — the case the four-topic run produced: a "What is Light?" slide given
a fluorescent lamp starter schematic. Caption on topic, host reputable,
figure useless for the objective.

Telling those apart needs someone to look at the image, so this module asks a
vision model exactly one narrow question:

    does this picture contain the objects and relationships the slide needs?

It is not asked to judge aesthetics, composition, or the slide as a whole. A
small structured answer keeps the call cheap and the failure modes few.

**Cost.** Off unless ``PPT_V2_VISION_VALIDATION`` is on. Then at most
``MAX_VALIDATIONS_PER_SLIDE`` calls for a whole slide, on the chosen candidate
only — never per search result, never on a diagram or an icon composition
(those are drawn to our own spec, so there is nothing to discover), and never
when deterministic scoring already answered. Results are cached under the
tenant, so a repeated deck costs nothing.

**Safety.** Every failure path — disabled, no provider, timeout, junk JSON,
exception — returns "unknown" and the caller carries on. A validator must
never be the reason a teacher loses a deck.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Two, not one: a rejected first candidate deserves a second look, and a
# third would cost more than the picture is worth.
MAX_VALIDATIONS_PER_SLIDE = 2

# Wall-clock budget for one look, in seconds.
#
# A live run measured a single Gemini vision call at 156s, which on its own
# was more than twice the whole deck's normal generation time. The provider's
# own timeout did not bound it, and a teacher is waiting, so the budget is
# enforced here: past it the answer is abandoned and the deterministic
# decision stands. Env-tunable because a slower model may be worth waiting
# for in a batch job.
TIMEOUT_S = 25.0
# A look shorter than this cannot finish; skip it rather than start it.
MIN_USEFUL_TIMEOUT_S = 5.0


def timeout_s() -> float:
    try:
        return max(5.0, float((os.getenv("PPT_V2_VISION_TIMEOUT", "") or "").strip()
                              or TIMEOUT_S))
    except (TypeError, ValueError):
        return TIMEOUT_S


def _ask_with_timeout(ask, image, prompt, budget):
    """Run `ask` on a worker thread and give up after `budget` seconds.

    The thread is left to finish on its own - there is no way to cancel a
    blocking HTTP call - but nothing waits for it, so a slow provider costs
    the picture rather than the request.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(ask, image, prompt)
        try:
            return future.result(timeout=budget), None
        except _Timeout:
            return None, "timed out after %.0fs" % budget
    finally:
        pool.shutdown(wait=False)

# Only these strategies are worth looking at. A programmatic diagram and an
# icon composition are drawn from our own specification, so a vision model
# can only confirm what we already know.
VALIDATED_STRATEGIES = frozenset({"searched_image", "cached_asset"})


def enabled() -> bool:
    return (os.getenv("PPT_V2_VISION_VALIDATION", "false") or "").strip().lower() \
        in ("1", "true", "yes", "on")


# Decision codes. The application returns one of these; the model never does.
PASSED = "PASSED"
CLASS_LEVEL_VIOLATION = "CLASS_LEVEL_VIOLATION"
REQUIRED_CONCEPT_MISSING = "REQUIRED_CONCEPT_MISSING"
RELATIONSHIP_MISSING = "RELATIONSHIP_MISSING"
COMPETING_CONCEPT = "COMPETING_CONCEPT"


@dataclass
class VisualContract:
    """What the picture has to show for this slide to work.

    Assembled from fields the planner already fills; `relationships` is the
    one genuinely new piece, and it is the one that separates topic relevance
    from teaching relevance. "A mirror" is on topic. "An incident ray
    striking a mirror and a reflected ray leaving it" is the lesson.
    """

    purpose: str = ""                                   # the learning objective
    required_objects: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    visual_kind: str = ""
    # The audience. Without it a figure can be exactly on topic and still
    # wrong: a Friction figure carrying mu_s*N and a static/kinetic graph is
    # about friction and useless to Class 8.
    class_name: str = ""
    board: str = ""
    subject: str = ""
    chapter: str = ""

    @property
    def is_meaningful(self) -> bool:
        """Is there enough here to ask a useful question?

        Without objects or relationships the model has nothing concrete to
        check, and a vague question returns a vague yes. Better not to spend
        the call.
        """
        return bool(self.required_objects or self.relationships)

    def to_dict(self) -> Dict[str, Any]:
        return {"purpose": self.purpose,
                "requiredObjects": list(self.required_objects),
                "relationships": list(self.relationships),
                "avoid": list(self.avoid),
                "visualKind": self.visual_kind,
                "className": self.class_name, "board": self.board,
                "subject": self.subject, "chapter": self.chapter}


@dataclass
class VisionVerdict:
    passed: bool
    missing: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    reason: str = ""
    available: bool = True          # False => never looked; do not treat as a fail
    cached: bool = False
    latency_ms: int = 0
    model: str = ""
    decision: str = PASSED
    observations: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.available:
            return "vision: not available (%s)" % (self.reason or "disabled")
        return "vision: %s%s%s" % (
            "pass" if self.passed else "fail",
            (" missing=" + ",".join(self.missing)) if self.missing else "",
            (" conflicts=" + ",".join(self.conflicts)) if self.conflicts else "")


def unavailable(reason: str) -> VisionVerdict:
    return VisionVerdict(passed=True, available=False, reason=reason)


_PROMPT = """Report what is VISIBLE in this picture. You are not deciding whether it can
be used - the application decides that from your report.

AUDIENCE: %(audience)s
OBJECTIVE: %(purpose)s
LOOK FOR: %(objects)s
SHOWING: %(relationships)s
WOULD CONFLICT: %(avoid)s

Rules: report only what you can see; do not infer the topic; do not judge how
attractive it is.

Put in "adv" anything visible that belongs to a HIGHER class than the
audience: Greek letters, subscripted force symbols, arrows over letters, any
algebraic formula, a graph with named axes or regions.

Reply with ONLY this JSON, at most 4 SHORT items per list, 3 words each:
{"adv":[],"seen":[],"rel":[],"conf":[]}

adv  = above-the-class things visible
seen = which of LOOK FOR are visible
rel  = which of SHOWING are actually shown
conf = which of WOULD CONFLICT are visible

Keep it short. No verdict, no score, no extra keys."""


def build_prompt(contract: VisualContract) -> str:
    """Ask for observations, naming the audience the figure has to suit."""
    from ai_services.core.ppt_v2 import class_rules

    none = "(none)"
    return _PROMPT % {
        "audience": class_rules.describe(contract.class_name, contract.subject,
                                         contract.chapter) or none,
        "purpose": contract.purpose or "(not stated)",
        "objects": ", ".join(contract.required_objects) or none,
        "relationships": ", ".join(contract.relationships) or none,
        "avoid": ", ".join(contract.avoid) or none,
    }


def _parse(text: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of a vision reply, tolerantly."""
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Models like to wrap the object in a sentence.
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            return None
    return None


def _as_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v or "").strip()]
    if value:
        return [str(value).strip()]
    return []


_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_FILLER = re.compile(r"\b(symbols?|molecules?|formulae?|formulas?|labels?|icons?|"
                     r"notation|diagram|the|a|an)\b", re.I)


def _norm_term(text: str) -> str:
    text = str(text or "").translate(_SUBSCRIPTS).lower()
    text = _FILLER.sub(" ", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _taught_by(item: str, contract: "VisualContract") -> bool:
    """Is this "advanced" item something the slide itself sets out to show?"""
    core = _norm_term(item)
    if not core:
        return False
    taught = _norm_term(" ".join([contract.purpose or ""]
                                 + list(contract.required_objects or [])
                                 + list(contract.relationships or [])))
    return bool(re.search(r"\b" + re.escape(core) + r"\b", taught))


def interpret(payload: Dict[str, Any], contract: VisualContract) -> VisionVerdict:
    """Decide, from the model's observations, whether the picture is usable.

    The decision is made here and only here. A model asked for a verdict
    supplies one cheerfully and is not accountable for a syllabus; what it is
    good at is listing what it can see. So any "usable" field it volunteers
    is ignored, and the rules live in class_rules where they can be corrected
    without touching a prompt.

    Checked in order of how badly each one fails the slide:
      CLASS_LEVEL_VIOLATION     above the class - wrong for this audience
      COMPETING_CONCEPT         a different topic dominates the picture
      REQUIRED_CONCEPT_MISSING  most of what the slide names is absent
      RELATIONSHIP_MISSING      the objects are there, the physics is not
    """
    from ai_services.core.ppt_v2 import class_rules

    # Compact keys are what the prompt asks for, because the shared vision
    # helper caps its output and an eight-field schema came back truncated
    # mid-string. The long spellings are still accepted: a model that
    # volunteers them, and the test fixtures, both keep working.
    visible = _as_list(payload.get("visible_concepts"))
    advanced = (_as_list(payload.get("adv"))
                + _as_list(payload.get("advanced_terms"))
                + _as_list(payload.get("equations"))
                + _as_list(payload.get("advanced_notation"))
                + _as_list(payload.get("class_level_concerns")))
    objects_seen = (_as_list(payload.get("seen"))
                    or _as_list(payload.get("required_objects_seen"))
                    or _as_list(payload.get("visible_objects")))
    relations_seen = (_as_list(payload.get("rel"))
                      or _as_list(payload.get("required_relationships_seen"))
                      or _as_list(payload.get("shown_relationships")))
    conflicts = (_as_list(payload.get("conf"))
                 or _as_list(payload.get("conflicts")))

    observations = {"visible_concepts": visible, "advanced": advanced,
                    "objects_seen": objects_seen,
                    "relationships_seen": relations_seen,
                    "conflicts": conflicts}

    def seen(needle: str, haystack) -> bool:
        n = needle.lower()
        return any(n in h.lower() or h.lower() in n for h in haystack)

    missing = [o for o in contract.required_objects if not seen(o, objects_seen)]

    # 1. Above the class. The chapter rules get the model's own words plus
    #    anything it listed as a concern, so a paraphrase still matches.
    # Only the advanced lists are checked against the rules. Feeding
    # visible_concepts in as well meant the topic word itself ("friction")
    # was tested against "coefficient of friction".
    class_hits = class_rules.violations(
        advanced, contract.class_name, contract.subject, contract.chapter)
    # The model's own "advanced" list, with no class rule behind it, is an
    # opinion - and on a live Class 10 Respiration slide it called CO2, O2
    # and H2O "above the class" on a slide that teaches exactly them. What
    # the slide itself teaches cannot be above the slide.
    if not class_hits:
        advanced = [a for a in advanced if not _taught_by(a, contract)]
    if class_hits or advanced:
        detail = class_hits or advanced[:3]
        return VisionVerdict(
            passed=False, missing=missing, conflicts=conflicts,
            decision=CLASS_LEVEL_VIOLATION, observations=observations,
            reason="above the class: %s" % ", ".join(detail[:3]))

    # 2. A competing topic dominates it.
    if conflicts:
        return VisionVerdict(
            passed=False, missing=missing, conflicts=conflicts,
            decision=COMPETING_CONCEPT, observations=observations,
            reason="shows %s" % ", ".join(conflicts[:2]))

    # 3. Most of what the slide names is simply not in the picture.
    if contract.required_objects and len(missing) > len(contract.required_objects) // 2:
        return VisionVerdict(
            passed=False, missing=missing, conflicts=conflicts,
            decision=REQUIRED_CONCEPT_MISSING, observations=observations,
            reason="missing %s" % ", ".join(missing[:3]))

    # 4. The objects are there but the picture does not show them doing
    #    anything - a mirror and a torch on a table, not a reflection.
    #
    #    Only decisive when the objects are ALSO thin. A live run rejected
    #    two perfectly good plant-cell diagrams because the planner had asked
    #    for the relationship "each organelle labeled with its function",
    #    which no picture can show: the requirement was pedagogical, not
    #    visual. When everything the slide names is visible, that is enough,
    #    and an unsatisfiable relationship should not throw the figure away.
    if contract.relationships and not relations_seen:
        if missing:
            return VisionVerdict(
                passed=False, missing=missing, conflicts=conflicts,
                decision=RELATIONSHIP_MISSING, observations=observations,
                reason="shows none of the required relationships, and is "
                       "missing %s" % ", ".join(missing[:2]))
        return VisionVerdict(
            passed=True, missing=missing, conflicts=[],
            decision=PASSED, observations=observations,
            reason="shows every required object, though no relationship was "
                   "reported")

    return VisionVerdict(passed=True, missing=missing, conflicts=[],
                         decision=PASSED, observations=observations,
                         reason="shows what the slide needs")


def validate(image: str, contract: VisualContract,
             ask: Optional[Callable[[str, str], str]] = None,
             cache_get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
             cache_put: Optional[Callable[[str, Dict[str, Any]], None]] = None,
             on_usage: Optional[Callable[..., None]] = None,
             model: str = "", timeout: Optional[float] = None) -> VisionVerdict:
    """Check one image against one contract. Never raises.

    ``timeout`` overrides the configured wait for this one call - the deck's
    deadline may leave less than that. Under a few seconds nobody looks, and
    the caller's deterministic decision stands.

    `ask(image, prompt) -> text` is injected: the existing vision helper
    lives in the view layer, and a core module importing a view is a cycle
    waiting to happen.
    """
    if not enabled():
        return unavailable("disabled")
    if not ask:
        return unavailable("no vision provider supplied")
    if not (image or "").strip():
        return unavailable("no image")
    if not contract.is_meaningful:
        # Nothing concrete to check; a vague question gets a vague yes.
        return unavailable("contract has nothing checkable")
    # Only the deck's deadline can make a look pointless; a short configured
    # timeout is an operator's choice and is honoured as it stands.
    if timeout is not None and timeout < MIN_USEFUL_TIMEOUT_S:
        return unavailable("no time left in the deck for a vision check")
    wait = timeout_s() if timeout is None else min(timeout, timeout_s())

    key = cache_key(image, contract)
    if cache_get:
        try:
            hit = cache_get(key)
        except Exception:
            hit = None
        if isinstance(hit, dict) and "passed" in hit:
            verdict = VisionVerdict(
                passed=bool(hit.get("passed")),
                missing=_as_list(hit.get("missing")),
                conflicts=_as_list(hit.get("conflicts")),
                reason=str(hit.get("reason") or ""), cached=True,
                model=str(hit.get("model") or ""))
            logger.info("ppt_v2 vision: cache hit | %s", verdict)
            return verdict

    started = time.perf_counter()
    try:
        answer, timed_out = _ask_with_timeout(ask, image, build_prompt(contract), wait)
    except Exception as exc:
        logger.warning("ppt_v2 vision: provider failed: %s", exc)
        if on_usage:
            on_usage(ok=False, latency_ms=int((time.perf_counter() - started) * 1000),
                     model=model, error=str(exc)[:200])
        return unavailable("provider error")
    if timed_out:
        logger.warning("ppt_v2 vision: %s; keeping the deterministic decision",
                       timed_out)
        if on_usage:
            on_usage(ok=False, latency_ms=int((time.perf_counter() - started) * 1000),
                     model=model, error=timed_out)
        return unavailable(timed_out)

    latency_ms = int((time.perf_counter() - started) * 1000)
    payload = _parse(answer)
    if payload is None:
        # Log the head of the reply, not just its length. A bare length tells
        # you nothing about whether the model refused, answered in prose, or
        # returned a different shape - and this path is otherwise silent
        # because it degrades safely. Truncated, and the reply is a
        # description of a public image, so there is nothing sensitive in it.
        logger.warning("ppt_v2 vision: unparseable reply (%d chars): %r",
                       len(str(answer or "")), str(answer or "")[:160])
        if on_usage:
            on_usage(ok=False, latency_ms=latency_ms, model=model,
                     error="unparseable reply")
        return unavailable("unparseable reply")

    verdict = interpret(payload, contract)
    verdict.latency_ms = latency_ms
    verdict.model = model
    logger.info("ppt_v2 vision: %s | %dms", verdict, latency_ms)

    if cache_put:
        try:
            cache_put(key, {"passed": verdict.passed, "missing": verdict.missing,
                            "conflicts": verdict.conflicts,
                            "reason": verdict.reason, "model": model})
        except Exception:
            pass
    if on_usage:
        on_usage(ok=True, latency_ms=latency_ms, model=model,
                 passed=verdict.passed)
    return verdict


def cache_key(image: str, contract: VisualContract) -> str:
    """Identity of a check: this picture against this contract."""
    import hashlib
    basis = "|".join([
        (image or "")[:512],
        contract.purpose,
        ",".join(contract.required_objects),
        ",".join(contract.relationships),
        ",".join(contract.avoid),
        # The audience is part of the question: the same picture is fine for
        # one class and wrong for another, so the verdicts must not collide.
        contract.class_name, contract.subject, contract.chapter,
    ])
    return "ppt_v2:vision:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def build_contract(*, learning_objective: str = "", objects: Sequence[str] = (),
                   must_show: Sequence[str] = (), relationships: Sequence[str] = (),
                   avoid: Sequence[str] = (), visual_kind: str = "",
                   class_name: str = "", board: str = "", subject: str = "",
                   chapter: str = "") -> VisualContract:
    """Assemble the contract from the fields the planner already fills."""
    def clean(values):
        out = []
        for v in values or ():
            v = re.sub(r"\s+", " ", str(v or "")).strip()
            if v and v.lower() not in {o.lower() for o in out}:
                out.append(v)
        return out

    return VisualContract(
        purpose=re.sub(r"\s+", " ", str(learning_objective or "")).strip(),
        # Four, not six. A live Friction slide listed five objects spanning
        # four different examples - "block on rough floor", "block on smooth
        # floor", "wheel on road", "boat in water" - which no single picture
        # contains. Three were reported missing, the half-threshold tripped,
        # and a perfectly good textbook figure was discarded. The planner
        # lists in priority order, so the first few are the subject and the
        # tail is elaboration; checking the subject is what the slide needs.
        required_objects=clean(list(must_show) + list(objects))[:4],
        relationships=clean(relationships)[:4],
        avoid=clean(avoid)[:6],
        visual_kind=(visual_kind or "").strip().lower(),
        class_name=(class_name or "").strip(), board=(board or "").strip(),
        subject=(subject or "").strip(), chapter=(chapter or "").strip(),
    )
