"""Three kinds of QA, kept apart, and a vocabulary for failures.

"QA failed" is not a useful thing to be told. Three genuinely different
questions get conflated by that phrase, they are answered by different
machinery at different cost, and they fail for different reasons:

  STRUCTURAL  Can this be drawn at all? Geometry on the slide, text that
              fits, contrast that reads, a layout the renderer implements.
              Decided from the specification alone, for free. (qa.py)

  SEMANTIC    Is the content right for this slide and this class? Topic
              relevance, competing concepts, terminology inside the
              syllabus, a visual contract worth checking against.
              Decided from text, for free. (relevance.py, planner rules)

  VISUAL      Does the chosen picture actually teach the objective? Only
              answerable by looking at the image, so it costs a model call
              and is therefore optional and rationed. (vision.py)

A deck can pass structurally and fail semantically — that is exactly what
the photosynthesis-and-respiration diagram did. It can pass both and fail
visually: the fluorescent lamp starter schematic on a "What is Light?"
slide. Reporting one verdict hides which of those happened.

Each layer reports pass / warn / fail / skipped independently, and
`skipped` is distinct from `pass`: a visual layer that never ran has not
approved anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIPPED = "skipped"

# ── Failure taxonomy ────────────────────────────────────────────────────────
# A code per failure mode, so a benchmark run says which thing broke rather
# than how many things broke.
CONTENT_ERROR = "CONTENT_ERROR"                    # planner produced nothing usable
CONTENT_SCOPE_ERROR = "CONTENT_SCOPE_ERROR"        # terminology above the class
VISUAL_CONTRACT_ERROR = "VISUAL_CONTRACT_ERROR"    # nothing checkable to ask for
ASSET_RELEVANCE_ERROR = "ASSET_RELEVANCE_ERROR"    # no candidate cleared the bar
VISION_VALIDATION_ERROR = "VISION_VALIDATION_ERROR"  # the picture failed a look
LAYOUT_ERROR = "LAYOUT_ERROR"                      # geometry off-slide or colliding
TEXT_OVERFLOW = "TEXT_OVERFLOW"                    # more text than the region holds
DIAGRAM_LEGIBILITY = "DIAGRAM_LEGIBILITY"          # drawn but unreadable
RENDER_ERROR = "RENDER_ERROR"                      # pptx/preview could not be built
PROVIDER_ERROR = "PROVIDER_ERROR"                  # upstream model or search was down
MISSING_VISUAL = "MISSING_VISUAL"                  # slide wanted a picture, got none

# Structural QA issue codes map onto the taxonomy.
_STRUCTURAL_CODES = {
    "text_overflow": TEXT_OVERFLOW,
    "overlap": LAYOUT_ERROR,
    "out_of_bounds": LAYOUT_ERROR,
    "no_layout": LAYOUT_ERROR,
    "no_visual_region": LAYOUT_ERROR,
    "unrenderable_layout": LAYOUT_ERROR,
    "missing_token": LAYOUT_ERROR,
    "low_contrast": LAYOUT_ERROR,
    "bad_colour": LAYOUT_ERROR,
    "schema": CONTENT_ERROR,
    "missing_asset": MISSING_VISUAL,
    "no_fallback": ASSET_RELEVANCE_ERROR,
    "no_prompt": VISUAL_CONTRACT_ERROR,
    "no_search_terms": VISUAL_CONTRACT_ERROR,
    "no_alt_text": WARN,
    # Label placement. Reusing LAYOUT_ERROR rather than inventing a category:
    # an overlapping or out-of-bounds label is a geometry failure.
    "label_overlap": LAYOUT_ERROR,
    "duplicate_label": LAYOUT_ERROR,
    "label_out_of_bounds": LAYOUT_ERROR,
    "zero_size_label": LAYOUT_ERROR,
    "label_unplaced": LAYOUT_ERROR,
    "no_visuals": MISSING_VISUAL,
    # The semantic label contract. A lost required label is a broken visual
    # contract, not a layout problem.
    "required_labels_lost": VISUAL_CONTRACT_ERROR,
    "labels_unspecified": WARN,
}

# Planner / service failure reasons map onto it too.
_REASON_CODES = {
    "llm_error": PROVIDER_ERROR,
    "malformed_json": CONTENT_ERROR,
    "invalid_schema": CONTENT_ERROR,
    "slide_count": CONTENT_ERROR,
    "empty_deck": CONTENT_ERROR,
    "qa_failed": LAYOUT_ERROR,
    "unexpected_error": RENDER_ERROR,
}


def classify_issue(code: str) -> str:
    return _STRUCTURAL_CODES.get(code, LAYOUT_ERROR)


def classify_reason(reason: str) -> str:
    return _REASON_CODES.get(reason, PROVIDER_ERROR)


@dataclass
class LayerResult:
    status: str = SKIPPED
    codes: List[str] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (PASS, WARN, SKIPPED)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "codes": list(self.codes),
                "detail": self.detail}


@dataclass
class GateResult:
    structural: LayerResult = field(default_factory=LayerResult)
    semantic: LayerResult = field(default_factory=LayerResult)
    visual: LayerResult = field(default_factory=LayerResult)

    @property
    def passed(self) -> bool:
        """Only structural failure blocks a deck.

        Semantic and visual failures mean a picture was rejected and the
        fallback chain moved on — the deck is still usable, just plainer. A
        structural failure means the renderer would draw something broken.
        """
        return self.structural.ok

    @property
    def codes(self) -> List[str]:
        out: List[str] = []
        for layer in (self.structural, self.semantic, self.visual):
            for code in layer.codes:
                if code not in out:
                    out.append(code)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """The shape that goes into the API response.

        Added alongside the existing `structuralIssues` key rather than
        replacing it, so nothing reading the current contract breaks.
        """
        return {"structural": self.structural.status,
                "semantic": self.semantic.status,
                "visual": self.visual.status,
                "codes": self.codes,
                "detail": {"structural": self.structural.detail,
                           "semantic": self.semantic.detail,
                           "visual": self.visual.detail}}


def evaluate(slides: Sequence[Any], issues: Sequence[Any],
             vision_verdicts: Optional[Sequence[Any]] = None,
             advanced_terms: Sequence[str] = ()) -> GateResult:
    """Report the three layers separately for one deck."""
    from ai_services.core.ppt_v2 import qa
    from ai_services.core.ppt_v2.schema import VisualStrategy

    result = GateResult()

    # ---- structural -------------------------------------------------------
    errors = qa.errors(issues)
    warnings = [i for i in issues if i not in errors]
    if errors:
        result.structural = LayerResult(
            FAIL, sorted({classify_issue(i.code) for i in errors}),
            "; ".join(str(i) for i in errors[:3]))
    elif warnings:
        result.structural = LayerResult(
            WARN, sorted({classify_issue(i.code) for i in warnings}),
            "%d warning(s)" % len(warnings))
    else:
        result.structural = LayerResult(PASS, [], "no structural issues")

    # ---- semantic ---------------------------------------------------------
    semantic_codes: List[str] = []
    notes: List[str] = []
    if advanced_terms:
        semantic_codes.append(CONTENT_SCOPE_ERROR)
        notes.append("terminology above the class: %s"
                     % ", ".join(sorted(set(advanced_terms))[:4]))

    wanted_a_picture = [s for s in slides
                        if s.visual.strategy is not VisualStrategy.NONE]
    got_nothing = [s for s in wanted_a_picture
                   if s.visual.resolved_strategy in (None, VisualStrategy.NONE)]
    if got_nothing:
        semantic_codes.append(ASSET_RELEVANCE_ERROR)
        notes.append("%d slide(s) wanted a picture and no candidate was "
                     "relevant enough" % len(got_nothing))

    uncheckable = [s for s in wanted_a_picture
                   if not (s.visual.must_show or s.visual.icons
                           or s.visual.relationships)]
    if uncheckable:
        semantic_codes.append(VISUAL_CONTRACT_ERROR)
        notes.append("%d slide(s) have no checkable visual contract"
                     % len(uncheckable))

    if CONTENT_SCOPE_ERROR in semantic_codes:
        result.semantic = LayerResult(FAIL, semantic_codes, "; ".join(notes))
    elif semantic_codes:
        result.semantic = LayerResult(WARN, semantic_codes, "; ".join(notes))
    else:
        result.semantic = LayerResult(PASS, [], "on topic and in scope")

    # ---- visual -----------------------------------------------------------
    verdicts = [v for v in (vision_verdicts or []) if getattr(v, "available", False)]
    if not verdicts:
        # Nothing looked at a picture, so nothing has been approved. Not a
        # pass; a pass would claim a check that never happened.
        result.visual = LayerResult(
            SKIPPED, [], "no image was inspected (validation disabled, "
                         "unavailable, or every candidate was high confidence)")
    else:
        failed = [v for v in verdicts if not v.passed]
        if failed:
            result.visual = LayerResult(
                WARN, [VISION_VALIDATION_ERROR],
                "%d of %d inspected image(s) rejected: %s"
                % (len(failed), len(verdicts),
                   "; ".join(v.reason for v in failed[:2])))
        else:
            result.visual = LayerResult(
                PASS, [], "%d image(s) inspected and accepted" % len(verdicts))

    return result
