"""PEDAGOGICAL STRATEGY - how to teach a given shape of content.

    content type  ->  pedagogical strategy  ->  visual strategy  ->  layout

Deterministic and table-driven. The table is the whole policy, so it can be
read, argued with and tested in one place.

Three things a row decides:

  bridge      Which existing SlideType carries the slide into the layout
              system. No new layouts: a flowchart and a cell are both "a
              central figure with named parts", and the layout registry
              already has that shape. The layout is an implementation detail
              of the representation, not the representation.

  visual      The primary visual strategy, or None to DELEGATE to the
              existing content-sensitive engine (visuals.preferred_strategy).
              Every content type that existed before this module delegates,
              so decks that were right stay exactly as they were. Only the
              new types, whose structure the old engine could not see, name
              a visual of their own.

  excluded    Visual strategies that would make this representation WRONG,
              removed from the fallback chain without reordering what is
              left. A generated image of a map invents geography; a generated
              image of a geometric figure invents measurements; a generated
              flowchart invents arrows. For those, a drawn figure or a real
              source is the only honest picture.

The fallback ORDER is never decided here - visuals.fallback_chain owns it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple

from ai_services.core.ppt_v2.content import ContentAnalysis, ContentType
from ai_services.core.ppt_v2.schema import SlideType, VisualStrategy


class PedagogicalStrategy(str, Enum):
    INTRODUCTION = "introduction"
    SEQUENTIAL_PROCESS = "sequential_process"
    LABELLED_STRUCTURE = "labelled_structure"
    EXPLAIN_WITH_SUPPORT = "explain_with_support"
    CAUSAL_CHAIN = "causal_chain"
    SIDE_BY_SIDE = "side_by_side"
    HIERARCHICAL_GROUPING = "hierarchical_grouping"
    CIRCULAR_SEQUENCE = "circular_sequence"
    CHRONOLOGICAL_SEQUENCE = "chronological_sequence"
    SPATIAL_MAP = "spatial_map"
    QUANTITATIVE_GRAPH = "quantitative_graph"
    FORMULA_BREAKDOWN = "formula_breakdown"
    STEPWISE_SOLUTION = "stepwise_solution"
    MEASURED_FIGURE = "measured_figure"
    DECISION_FLOW = "decision_flow"
    COMPONENT_RELATIONSHIP = "component_relationship"
    APPARATUS_TO_CONCLUSION = "apparatus_to_conclusion"
    CONTEXTUAL_CASE = "contextual_case"
    TERM_AND_EXAMPLE = "term_and_example"
    RETRIEVAL_PRACTICE = "retrieval_practice"
    CONSOLIDATION = "consolidation"


_AI = VisualStrategy.AI_GENERATED_IMAGE
_DIAGRAM = VisualStrategy.PROGRAMMATIC_DIAGRAM
_SEARCH = VisualStrategy.SEARCHED_IMAGE
_ICONS = VisualStrategy.ICON_COMPOSITION
_NONE = VisualStrategy.NONE


@dataclass(frozen=True)
class Rule:
    strategy: PedagogicalStrategy
    bridge: SlideType
    visual: Optional[VisualStrategy]          # None = delegate to the existing engine
    excluded: FrozenSet[VisualStrategy] = frozenset()
    labels_required: bool = False
    why: str = ""


RULES: Dict[ContentType, Rule] = {
    # --- existing types: delegate, so prior decks are unchanged -------------
    ContentType.TITLE: Rule(
        PedagogicalStrategy.INTRODUCTION, SlideType.TITLE, None,
        why="opens the deck and frames the topic"),
    ContentType.PROCESS: Rule(
        PedagogicalStrategy.SEQUENTIAL_PROCESS, SlideType.PROCESS, None,
        why="an ordered sequence of inputs, transformation and outputs is carried "
            "by numbered steps; the visual shows the things involved"),
    ContentType.ANATOMY_STRUCTURE: Rule(
        PedagogicalStrategy.LABELLED_STRUCTURE, SlideType.ANATOMY, None,
        labels_required=True,
        why="a structure whose named parts matter needs one central figure with "
            "the parts labelled"),
    ContentType.CONCEPT_EXPLANATION: Rule(
        PedagogicalStrategy.EXPLAIN_WITH_SUPPORT, SlideType.CONCEPT, None,
        why="an idea and its supporting facts, with one picture that grounds it"),
    ContentType.CAUSE_EFFECT: Rule(
        PedagogicalStrategy.CAUSAL_CHAIN, SlideType.CAUSE_EFFECT, None,
        why="a directional relationship from cause to effect"),
    ContentType.COMPARISON: Rule(
        PedagogicalStrategy.SIDE_BY_SIDE, SlideType.COMPARISON, None,
        why="two things set against each other along shared dimensions"),
    ContentType.TIMELINE: Rule(
        PedagogicalStrategy.CHRONOLOGICAL_SEQUENCE, SlideType.TIMELINE, None,
        why="events in time order; the timeline itself is the visual"),
    ContentType.FORMULA_EQUATION: Rule(
        PedagogicalStrategy.FORMULA_BREAKDOWN, SlideType.FORMULA, None,
        why="an equation drawn natively, with each variable explained"),
    ContentType.CASE_STUDY: Rule(
        PedagogicalStrategy.CONTEXTUAL_CASE, SlideType.EXAMPLE, None,
        why="a real situation that grounds the idea, best shown as it really looks"),
    ContentType.GRAPH_DATA: Rule(
        PedagogicalStrategy.QUANTITATIVE_GRAPH, SlideType.DATA, None,
        excluded=frozenset({_AI}),
        why="quantities on axes; plotted from the data, never photographed or "
            "imagined"),
    ContentType.SUMMARY: Rule(
        PedagogicalStrategy.CONSOLIDATION, SlideType.SUMMARY, None,
        why="consolidates what was taught; the grid of points is the visual"),
    ContentType.REVISION_QUIZ: Rule(
        PedagogicalStrategy.RETRIEVAL_PRACTICE, SlideType.QUIZ, None,
        why="retrieval practice; a picture would give the answer away"),

    # --- new types: their structure needs a visual of its own ---------------
    ContentType.CLASSIFICATION: Rule(
        PedagogicalStrategy.HIERARCHICAL_GROUPING, SlideType.SUMMARY, _ICONS,
        why="groups and their members read as a grid of labelled cards"),
    ContentType.CYCLE: Rule(
        PedagogicalStrategy.CIRCULAR_SEQUENCE, SlideType.ANATOMY, _DIAGRAM,
        excluded=frozenset({_AI}), labels_required=True,
        why="the stages return to the start; a linear row of steps hides the "
            "one thing a cycle teaches, so it is drawn as a loop"),
    ContentType.GEOGRAPHIC_MAP: Rule(
        PedagogicalStrategy.SPATIAL_MAP, SlideType.ANATOMY, _SEARCH,
        excluded=frozenset({_AI}), labels_required=True,
        why="where things are is the lesson; a real map is the only honest "
            "source, a generated one invents geography"),
    ContentType.WORKED_EXAMPLE: Rule(
        PedagogicalStrategy.STEPWISE_SOLUTION, SlideType.FORMULA, _NONE,
        why="problem, worked steps, result; the working IS the visual and a "
            "decorative picture costs attention"),
    ContentType.GEOMETRIC_DIAGRAM: Rule(
        PedagogicalStrategy.MEASURED_FIGURE, SlideType.ANATOMY, _DIAGRAM,
        excluded=frozenset({_AI}), labels_required=True,
        why="shapes, angles and measurements must be exact, so the figure is "
            "drawn from its specification"),
    ContentType.FLOWCHART_ALGORITHM: Rule(
        PedagogicalStrategy.DECISION_FLOW, SlideType.CONCEPT, _DIAGRAM,
        excluded=frozenset({_AI}), labels_required=True,
        why="nodes and directed edges; a generated image would invent arrows"),
    ContentType.SYSTEM_ARCHITECTURE: Rule(
        PedagogicalStrategy.COMPONENT_RELATIONSHIP, SlideType.ANATOMY, _DIAGRAM,
        excluded=frozenset({_AI}), labels_required=True,
        why="named components and how they connect, drawn as a labelled system"),
    ContentType.EXPERIMENT_PROCEDURE: Rule(
        PedagogicalStrategy.APPARATUS_TO_CONCLUSION, SlideType.PROCESS, _SEARCH,
        excluded=frozenset({_AI}), labels_required=True,
        why="apparatus, procedure, observation, conclusion as steps, beside a "
            "real photograph of the set-up; generated apparatus is unreliable"),
    ContentType.DEFINITION_TERMINOLOGY: Rule(
        PedagogicalStrategy.TERM_AND_EXAMPLE, SlideType.SUMMARY, _NONE,
        why="one card per term - the term as its heading, the meaning and an "
            "example beneath - across the full width with no picture "
            "competing; the words ARE the content. (Rasterising settled it: "
            "an icon spent half the slide on one glyph for the example's noun, "
            "and the hero layout without a picture ran every definition into "
            "one paragraph and lost the terms)"),
}


@dataclass
class Decision:
    """The strategy layer's answer for one slide, with its reasoning."""

    content_type: ContentType
    strategy: PedagogicalStrategy
    slide_type: SlideType
    visual: Optional[VisualStrategy]
    excluded: Tuple[VisualStrategy, ...] = ()
    labels_required: bool = False
    reason: str = ""
    analysis: Optional[ContentAnalysis] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contentType": self.content_type.value,
            "pedagogicalStrategy": self.strategy.value,
            "slideType": self.slide_type.value,
            "visual": self.visual.value if self.visual else None,
            "excluded": [v.value for v in self.excluded],
            "labelsRequired": self.labels_required,
            "reason": self.reason,
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }


def decide(analysis: ContentAnalysis) -> Decision:
    rule = RULES.get(analysis.content_type) or RULES[ContentType.CONCEPT_EXPLANATION]
    # Labels the plan names are required whatever the type: the planner put
    # them there because the picture needs them read.
    labels = rule.labels_required or bool(analysis.required_labels)
    reason = rule.why
    if analysis.reason:
        reason = "%s. Classified: %s" % (rule.why, analysis.reason)
    return Decision(content_type=analysis.content_type, strategy=rule.strategy,
                    slide_type=rule.bridge, visual=rule.visual,
                    excluded=tuple(sorted(rule.excluded, key=lambda v: v.value)),
                    labels_required=labels, reason=reason, analysis=analysis)
