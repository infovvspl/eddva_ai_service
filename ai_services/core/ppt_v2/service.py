"""V2 orchestration: one function the view can call.

The view stays thin on purpose. ``ppt.py`` is already 66KB and holds the V1
pipeline; duplicating any of it here would mean two places to fix every bug.
What the view does is decide the version and call :func:`generate_v2`. What
this module does is run the five stages and hand back the same
``{"title", "slides"}`` payload shape V1 returns, so the response contract
does not move.

    PLANNER    what each slide teaches and what to show   planner.py
    COMPOSER   layout, theme, prompts, geometry           compose.py
    PRODUCERS  turn a visual spec into an actual asset    producers.py
    QA         is this structurally renderable            qa.py
    RESPONSE   V1-compatible keys + a nested v2 block     schema.to_v1_dict

Asset resolution is concurrent, with the same worker ceiling V1 uses for its
image downloads. Doing it slide-by-slide would be the single easiest way to
turn a 40-second deck into a timeout: generated images are the slowest step
in the pipeline by an order of magnitude.

Image search is injected rather than imported, because it lives in the view
layer and ``core`` importing a view is a cycle waiting to happen.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ai_services.core.ppt_v2 import gate as gate_module
from ai_services.core.ppt_v2 import qa
from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
from ai_services.core.ppt_v2.design import studio_theme_key
from ai_services.core.ppt_v2.flags import ai_image_budget
from ai_services.core.ppt_v2 import labels as label_policy
from ai_services.core.ppt_v2.planner import PlannerError, focus_topic_for, plan_deck
from ai_services.core.ppt_v2.producers import ProducerSet
from ai_services.core.ppt_v2.schema import SlideSpec, VisualStrategy
from ai_services.core.ppt_v2.visuals import detect_capabilities, resolve_visual

logger = logging.getLogger(__name__)

# Matches V1's _IMAGE_WORKERS. Asset work is network-bound and the providers
# are shared, so raising this would mostly buy more rate-limit errors.
ASSET_WORKERS = 5


class V2Error(Exception):
    """V2 could not produce a deck. ``reason`` is a short stable code."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason


@dataclass
class V2Result:
    title: str
    slides: List[SlideSpec]
    issues: List[Any] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    planner_model: str = ""
    planner_attempts: int = 1
    grounded: bool = False
    asset_attempts: List[Dict[str, Any]] = field(default_factory=list)
    ai_images_made: int = 0
    # slide number -> the strategy the content called for, before availability
    # and budget were applied.
    wanted: Dict[int, VisualStrategy] = field(default_factory=dict)
    vision_verdicts: List[Any] = field(default_factory=list)
    # Per generated image: text the model drew, text we erased, labels the
    # renderer will overlay. Diagnostic only - nothing reads it to decide
    # anything - but the three numbers are what tells a prompt problem apart
    # from an image-damage problem when a deck looks wrong.
    image_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    gate: Optional[Any] = None

    def to_payload(self) -> Dict[str, Any]:
        """The ``data`` block of the API response.

        V1-compatible keys sit at the top of each slide; the full V2
        specification rides underneath in ``v2``. A renderer that has not
        learned V2 reads the former and ignores the latter.
        """
        return {
            "title": self.title,
            "slides": [s.to_v1_dict() for s in self.slides],
            "generation": {
                "version": "v2",
                "plannerModel": self.planner_model,
                "plannerAttempts": self.planner_attempts,
                "grounded": self.grounded,
                "aiImagesGenerated": self.ai_images_made,
                # Three values, not two, because they answer different
                # questions. "wanted" is what the content called for before
                # availability was considered; "planned" is what the budget
                # and the deployment allowed; "resolved" is what was actually
                # produced. Collapsing them hides the most useful fact of
                # all - that a slide wanted a generated figure and the
                # deployment had no token for one.
                "visualStrategies": [
                    {"slide": s.slide_number,
                     "wanted": (self.wanted.get(s.slide_number) or s.visual.strategy).value,
                     "planned": s.visual.strategy.value,
                     "resolved": (s.visual.resolved_strategy.value
                                  if s.visual.resolved_strategy else "")}
                    for s in self.slides
                ],
                "structuralIssues": [str(i) for i in self.issues],
                # The three QA layers, reported separately. Added alongside
                # structuralIssues rather than replacing it: "QA failed" hides
                # which of three different questions was answered no.
                "qa": self.gate.to_dict() if self.gate else None,
                "timings": {k: round(v, 2) for k, v in self.timings.items()},
            },
        }


def generate_v2(*, topic: str, slide_count: int, language: str = "English",
                ctx: Optional[Dict[str, str]] = None, board: str = "",
                board_rules: str = "", passages: Optional[Sequence[Any]] = None,
                institute_id: Any = None, vertical: str = "school",
                search_fn: Optional[Callable[..., Dict[str, Any]]] = None,
                cache: Any = None, llm: Any = None,
                on_planner_usage: Optional[Callable[..., None]] = None,
                on_asset: Optional[Callable[..., None]] = None,
                vision_ask=None, vision_model: str = "",
                on_vision_usage=None,
                on_progress: Optional[Callable[..., None]] = None,
                on_event: Optional[Callable[..., None]] = None,
                auto_range: Optional[Tuple[int, int]] = None,
                theme_key: str = "") -> V2Result:
    """Run the V2 pipeline. Raises :class:`V2Error` on an unrecoverable stage.

    ``theme_key`` is the teacher's studio theme (design.STUDIO_PALETTES); an
    empty or unknown one keeps the subject's colours.

    ``on_progress(stage, done=, total=, partial=)`` hears each stage; from
    composition on, ``partial`` carries the deck as far as it exists, so a
    background job can show the slides before their pictures are in.
    """
    ctx = ctx or {}

    def emit(stage, **kw):
        if on_progress:
            try:
                on_progress(stage, **kw)
            except Exception:  # never fail a deck over a progress report
                logger.exception("ppt_v2: progress callback failed")

    emit("planning")
    timings: Dict[str, float] = {}
    started = time.perf_counter()

    theme_key = studio_theme_key(theme_key) or ""
    logger.info("ppt_v2: start | topic=%r slides=%d grounded=%s scope=%r theme=%s",
                topic[:80], slide_count, bool(passages),
                "/".join(x for x in (ctx.get("className"), ctx.get("subjectName"),
                                     ctx.get("chapterName")) if x),
                theme_key or "subject")

    # ---- 1. plan ----------------------------------------------------------
    t0 = time.perf_counter()
    plan_args = dict(
        topic=topic, slide_count=slide_count, language=language,
        class_name=ctx.get("className", ""), subject=ctx.get("subjectName", ""),
        chapter=ctx.get("chapterName", "") or ctx.get("topicName", ""),
        focus_topic=focus_topic_for(ctx),
        board=board, board_rules=board_rules, passages=passages,
        institute_id=institute_id, llm=llm, on_usage=on_planner_usage,
        on_event=on_event, auto_range=auto_range,
    )
    try:
        plan = plan_deck(**plan_args)
        # Off the chapter? Re-plan once now - before composition and before a
        # single picture is paid for - rather than ship it and have the
        # teacher generate again.
        from ai_services.core.ppt_v2 import scope
        from ai_services.core.ppt_v2.planner import revise_args as _revise_args, revise_slides
        plan = scope.replan_if_off_scope(
            plan, revise=lambda note, bad: revise_slides(
                slides=bad, note=note, **_revise_args(plan_args)),
            passages=passages, class_name=ctx.get("className", ""),
            started=started, label="ppt_v2", on_event=on_event)
        _fact_check(plan, passages=passages, topic=topic,
                    chapter=ctx.get("chapterName", "") or ctx.get("topicName", ""),
                    llm=llm, institute_id=institute_id, started=started,
                    on_event=on_event, on_usage=on_planner_usage)
    except PlannerError as exc:
        timings["planner"] = time.perf_counter() - t0
        logger.warning("ppt_v2: planner failed | reason=%s", exc.reason)
        raise V2Error(exc.reason, str(exc))
    timings["planner"] = time.perf_counter() - t0
    logger.info("ppt_v2: planner done | slides=%d attempts=%d model=%s %.1fs",
                len(plan.slides), plan.attempts, plan.model, timings["planner"])
    emit("composing")
    if on_event:
        on_event("layout", "Designing the layout of each slide", "active")

    # ---- 2. compose -------------------------------------------------------
    t0 = time.perf_counter()
    caps = detect_capabilities(ai_image_budget=ai_image_budget())
    deck = compose_deck(
        plan.slides,
        DeckContext(topic=topic, subject=ctx.get("subjectName", ""),
                    chapter=ctx.get("chapterName", ""),
                    class_name=ctx.get("className", ""), board=board,
                    theme_key=theme_key),
        caps=caps,
    )
    timings["compose"] = time.perf_counter() - t0

    if not deck.slides:
        raise V2Error("empty_deck", "composition produced no slides")
    # With an automatic count the plan decided the number; hold it to that.
    expected = len(plan.slides) if auto_range else slide_count
    if len(deck.slides) != expected:
        # The planner enforced this already; a mismatch here means composition
        # dropped something, which is a bug rather than a model failure.
        raise V2Error("slide_count",
                      "composed %d slides for a %d-slide request"
                      % (len(deck.slides), expected))

    for slide in deck.slides:
        logger.info("ppt_v2: slide %d | type=%s layout=%s visual=%s",
                    slide.slide_number, slide.slide_type.value,
                    slide.layout.name if slide.layout else "-",
                    slide.visual.strategy.value)
    for note in deck.notes:
        logger.info("ppt_v2: compose note | %s", note)

    # The deck as it stands: text and layout now, pictures as they land.
    snapshot_lock = __import__("threading").Lock()

    def snapshot() -> Dict[str, Any]:
        with snapshot_lock:
            slides = []
            for s in deck.slides:
                d = s.to_v1_dict()
                if (s.visual.strategy is not VisualStrategy.NONE
                        and s.visual.resolved_strategy is None):
                    d["v2"]["visual"]["pending"] = True
                slides.append(d)
            return {"title": plan.title or topic, "slides": slides,
                    "generation": {"version": "v2", "partial": True}}

    wanting = sum(1 for s in deck.slides if s.visual.strategy is not VisualStrategy.NONE)
    if on_event:
        on_event("layout", "Designed %d slides; %d need a picture" % (len(deck.slides), wanting),
                 "done")
    emit("pictures", done=0, total=wanting, partial=snapshot())
    pictures_done = [0]

    def slide_done(_slide):
        with snapshot_lock:
            pictures_done[0] += 1
            done = pictures_done[0]
        emit("pictures", done=done, total=wanting, partial=snapshot())

    # ---- 3. assets --------------------------------------------------------
    t0 = time.perf_counter()
    producers = _build_producers(ctx=ctx, board=board, search_fn=search_fn,
                                 cache=cache, institute_id=institute_id,
                                 vertical=vertical, topic=topic,
                                 vision_ask=vision_ask,
                                 vision_model=vision_model,
                                 on_vision_usage=on_vision_usage)
    # The deck's deadline. The backend gives up on the whole request, so a
    # deck that is late is a deck the teacher never sees. Optional steps (a
    # generated picture, a vision check) give way as it approaches.
    producers.deadline = started + deadline_s()
    producers.on_event = on_event
    _resolve_assets(deck.slides, producers, on_asset=on_asset, on_slide_done=slide_done)
    timings["assets"] = time.perf_counter() - t0

    # The label policy depends on which producer won, so it can only be
    # applied now. A LEGEND visual's anchored labels become a leader-less key.
    for slide in deck.slides:
        label_policy.apply_policy(slide.visual)

    ai_made = sum(1 for s in deck.slides
                  if s.visual.resolved_strategy is VisualStrategy.AI_GENERATED_IMAGE)

    # ---- 4. structural QA -------------------------------------------------
    issues = qa.check_deck(deck.slides)
    gate = gate_module.evaluate(deck.slides, issues,
                                vision_verdicts=producers.vision_verdicts,
                                advanced_terms=_advanced_terms(deck.slides))
    logger.info("ppt_v2: qa | structural=%s semantic=%s visual=%s%s",
                gate.structural.status, gate.semantic.status,
                gate.visual.status,
                (" codes=" + ",".join(gate.codes)) if gate.codes else "")
    blocking = qa.errors(issues)
    if blocking:
        # Structural errors mean the renderer would draw something broken.
        # Better to fall back to V1 than to ship a deck with text off the slide.
        logger.warning("ppt_v2: structural QA failed | %s",
                       "; ".join(str(i) for i in blocking[:3]))
        raise V2Error("qa_failed", "; ".join(str(i) for i in blocking[:3]))
    if issues:
        logger.info("ppt_v2: structural QA warnings | %d", len(issues))

    timings["total"] = time.perf_counter() - started
    logger.info("ppt_v2: complete | slides=%d ai_images=%d %.1fs",
                len(deck.slides), ai_made, timings["total"])

    return V2Result(
        title=plan.title or topic, slides=deck.slides, issues=issues,
        timings=timings, planner_model=plan.model,
        planner_attempts=plan.attempts, grounded=plan.grounded,
        asset_attempts=producers.attempts, ai_images_made=ai_made,
        vision_verdicts=list(producers.vision_verdicts), gate=gate,
        image_diagnostics=list(producers.image_diagnostics),
        wanted={s.slide_number: deck.plan.wanted_for(s.slide_number)
                for s in deck.slides if deck.plan
                and deck.plan.wanted_for(s.slide_number)},
    )


def _fact_check(plan, **kwargs) -> None:
    """Correct statements that contradict the book, before anything is built.

    Never raises: a check that fails leaves the plan as it was written.
    """
    from ai_services.core.ppt_v2 import factcheck
    try:
        fixes = factcheck.check_facts(plan.slides, **kwargs)
        if fixes:
            logger.info("ppt_v2: fact check corrected %d statement(s)", len(fixes))
    except Exception:
        logger.exception("ppt_v2: fact check failed; slides left as planned")


# How each way of filling a slide's picture reads in the activity feed.
_PICTURE_OUTCOME = {
    VisualStrategy.AI_GENERATED_IMAGE: "illustration ready",
    VisualStrategy.SEARCHED_IMAGE: "textbook figure added",
    VisualStrategy.CACHED_ASSET: "picture added",
    VisualStrategy.PROGRAMMATIC_DIAGRAM: "diagram drawn",
    VisualStrategy.ICON_COMPOSITION: "icon layout added",
}


def deadline_s() -> float:
    """Seconds a whole deck may take. Must sit under the backend's wait (240s)
    and under the web server's worker timeout wherever it is deployed."""
    try:
        return max(60.0, float(os.getenv("PPT_DEADLINE_S", "") or DEFAULT_DEADLINE_S))
    except ValueError:
        return DEFAULT_DEADLINE_S


DEFAULT_DEADLINE_S = 200.0


def _advanced_terms(slides):
    """Terminology in the deck that belongs to a higher class.

    Reuses the list relevance.py already maintains rather than keeping a
    second dictionary; the planner prompt is the primary guard and this is
    the check that it held.
    """
    import re

    from ai_services.core.ppt_v2.relevance import ADVANCED_TERMS

    text = " ".join(
        " ".join([s.title.text or "", s.subtitle or "", s.learning_objective or ""]
                 + [(b.label or "") + " " + (b.text or "") for b in s.content_blocks])
        for s in slides)
    return [t for t in ADVANCED_TERMS if re.search(r"\b" + t, text, re.I)]


def _build_producers(*, ctx: Dict[str, str], board: str, search_fn, cache,
                     institute_id, vertical: str, topic: str = "",
                     vision_ask=None, vision_model: str = "",
                     on_vision_usage=None) -> ProducerSet:
    """Wire the producers to the real services, with a tenant-scoped cache.

    The cache is the project's existing ``ResponseCache``, which already keys
    on ``(institute_id, vertical, feature, prompt_hash)``. That scoping is the
    whole reason to reuse it rather than keying on the raw prompt: two schools
    generating a deck from their own chapter text must not be able to read
    each other's assets out of a shared key.
    """
    get_fn = put_fn = None
    if cache is not None and institute_id is not None:
        feature = "ppt_v2_asset"

        def get_fn(key: str):  # noqa: F811
            return cache.get(str(institute_id), feature, key, vertical)

        def put_fn(key: str, value: Dict[str, Any]):  # noqa: F811
            cache.set(str(institute_id), feature, key, value, vertical)

    return ProducerSet(
        subject=ctx.get("subjectName", ""), class_name=ctx.get("className", ""),
        board=board, search_fn=search_fn, cache_get=get_fn, cache_put=put_fn,
        search_ctx=dict(ctx), topic=topic,
        chapter=ctx.get("chapterName", "") or topic, vision_ask=vision_ask,
        vision_model=vision_model, on_vision_usage=on_vision_usage,
    )


def _resolve_assets(slides: Sequence[SlideSpec], producers: ProducerSet,
                    on_asset: Optional[Callable[..., None]] = None,
                    on_slide_done: Optional[Callable[[SlideSpec], None]] = None) -> None:
    """Fetch every slide's visual, concurrently.

    One slide's asset failing must not affect another's, so each resolution is
    independent and the fallback chain inside ``resolve_visual`` absorbs its
    own errors. A slide that wants nothing is skipped rather than queued.
    """
    wanted = [s for s in slides if s.visual.strategy is not VisualStrategy.NONE]
    if not wanted:
        return

    producer_map = producers.as_map()

    def run(slide: SlideSpec) -> None:
        def record(strategy, ok, detail):
            producers.record(slide.slide_number, strategy, ok, detail)
            level = logger.info if ok else logger.warning
            level("ppt_v2: slide %d asset | strategy=%s ok=%s%s",
                  slide.slide_number, strategy.value, ok,
                  " detail=%s" % detail[:120] if detail else "")
            if on_asset:
                on_asset(slide.slide_number, strategy, ok, detail)

        producers.begin_slide(slide.slide_number)
        try:
            resolve_visual(slide.visual, producer_map, on_attempt=record)
            producers.note(_PICTURE_OUTCOME.get(slide.visual.resolved_strategy,
                                                "laid out without a picture"),
                           "done" if slide.visual.resolved_strategy in _PICTURE_OUTCOME
                           else "warn")
        finally:
            producers.end_slide()
        if on_slide_done:
            on_slide_done(slide)

    with ThreadPoolExecutor(max_workers=min(ASSET_WORKERS, len(wanted))) as pool:
        list(pool.map(run, wanted))
