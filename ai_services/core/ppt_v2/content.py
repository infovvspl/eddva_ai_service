"""CONTENT ANALYSIS - what kind of thing is this slide trying to teach?

The first question the engine asks of a slide is not "which layout looks
closest" but "what does the student need to understand, and what SHAPE does
that understanding have". A sequence, a structure with named parts, two
things set side by side, a quantity changing along an axis, a decision with
branches. That shape is the content type. Everything after it - the
pedagogical strategy, the visual, the layout - is chosen to fit the shape.

Content types are deliberately subject-agnostic. "A labelled structure" is
the same teaching problem in a plant cell, a motherboard and a parliament;
"a quantity on an axis" is the same in a demand curve and a population
graph. Subject changes the vocabulary, the class-level rules and the theme,
never the type system.

Two sources, one decision:

  1. The planner may DECLARE a type. It knows the pedagogy, so a declared
     type is the starting point - but a model asked for a label will supply
     one cheerfully, so the declaration is VALIDATED against the slide's own
     content. A "timeline" with no dates, a "worked example" with no worked
     numbers, a "formula" with no formula: each is rejected and the type is
     inferred instead.
  2. Inference reads shape cues from the slide's text - generic vocabulary
     of shape ("compare", "types of", "algorithm", "angle", years in two
     blocks) - never topic names. "concept" is the planner's own "unsure"
     answer, so it is treated as an invitation to infer, not a decision.

An unknown or unsupported type falls back to CONCEPT_EXPLANATION. Nothing in
here raises on bad input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple



# Whole-word matching, with a plain plural allowed. Deliberately NOT the
# package's textmatch.matches_any, which matches word PREFIXES by design (so
# a subject keyword "molecul" catches "molecules"). For shape cues a prefix
# match is wrong: "data" caught "database" and turned a DBMS architecture
# slide into a graph; "era" would catch "erase", "term" "terminate".
_WORD_CACHE: Dict[Tuple[str, ...], "re.Pattern"] = {}


def matches_any(text: str, cues: Sequence[str]) -> bool:
    if not text or not cues:
        return False
    key = tuple(cues)
    pattern = _WORD_CACHE.get(key)
    if pattern is None:
        parts = [r"\s+".join(re.escape(w) for w in c.split()) for c in cues if c]
        pattern = re.compile(r"\b(?:%s)(?:s|es)?\b" % "|".join(parts))
        _WORD_CACHE[key] = pattern
    return bool(pattern.search(re.sub(r"\s+", " ", str(text).lower())))


class ContentType(str, Enum):
    """The shape of what a slide teaches. Renderer-agnostic."""

    TITLE = "title"                                # structural: the cover
    PROCESS = "process"
    ANATOMY_STRUCTURE = "anatomy_structure"
    CONCEPT_EXPLANATION = "concept_explanation"
    CAUSE_EFFECT = "cause_effect"
    COMPARISON = "comparison"
    CLASSIFICATION = "classification"
    CYCLE = "cycle"
    TIMELINE = "timeline"
    GEOGRAPHIC_MAP = "geographic_map"
    GRAPH_DATA = "graph_data"
    FORMULA_EQUATION = "formula_equation"
    WORKED_EXAMPLE = "worked_example"
    GEOMETRIC_DIAGRAM = "geometric_diagram"
    FLOWCHART_ALGORITHM = "flowchart_algorithm"
    SYSTEM_ARCHITECTURE = "system_architecture"
    EXPERIMENT_PROCEDURE = "experiment_procedure"
    CASE_STUDY = "case_study"
    DEFINITION_TERMINOLOGY = "definition_terminology"
    REVISION_QUIZ = "revision_quiz"
    SUMMARY = "summary"


DEFAULT_TYPE = ContentType.CONCEPT_EXPLANATION

# Types whose position in the deck decides them, not their wording. Never
# re-inferred: a summary that happens to mention "steps" is still a summary.
_STRUCTURAL = frozenset({ContentType.TITLE, ContentType.SUMMARY,
                         ContentType.REVISION_QUIZ})

# Every spelling we accept for a type. The legacy slideType vocabulary is
# here too, so a planner payload written before content types existed maps
# onto the same decisions it always produced.
_ALIASES: Dict[str, ContentType] = {
    # legacy slideType values
    "title": ContentType.TITLE, "cover": ContentType.TITLE, "intro": ContentType.TITLE,
    "concept": ContentType.CONCEPT_EXPLANATION,
    "process": ContentType.PROCESS,
    "comparison": ContentType.COMPARISON,
    "cause_effect": ContentType.CAUSE_EFFECT,
    "anatomy": ContentType.ANATOMY_STRUCTURE,
    "timeline": ContentType.TIMELINE,
    "formula": ContentType.FORMULA_EQUATION,
    # Legacy "example" meant a real-world example - a searched photograph
    # beside the facts - not a worked numerical solution.
    "example": ContentType.CASE_STUDY,
    "data": ContentType.GRAPH_DATA,
    "summary": ContentType.SUMMARY,
    "quiz": ContentType.REVISION_QUIZ,
    # near-synonyms models actually reach for
    "introduction": ContentType.CONCEPT_EXPLANATION, "content": ContentType.CONCEPT_EXPLANATION,
    "bullets": ContentType.CONCEPT_EXPLANATION, "explanation": ContentType.CONCEPT_EXPLANATION,
    "steps": ContentType.PROCESS, "sequence": ContentType.PROCESS,
    "procedure": ContentType.EXPERIMENT_PROCEDURE, "experiment": ContentType.EXPERIMENT_PROCEDURE,
    "activity": ContentType.EXPERIMENT_PROCEDURE,
    "compare": ContentType.COMPARISON, "vs": ContentType.COMPARISON,
    "contrast": ContentType.COMPARISON, "difference": ContentType.COMPARISON,
    "causes": ContentType.CAUSE_EFFECT, "effects": ContentType.CAUSE_EFFECT,
    "structure": ContentType.ANATOMY_STRUCTURE, "parts": ContentType.ANATOMY_STRUCTURE,
    "labelled": ContentType.ANATOMY_STRUCTURE, "labeled": ContentType.ANATOMY_STRUCTURE,
    "diagram": ContentType.ANATOMY_STRUCTURE, "anatomy_structure": ContentType.ANATOMY_STRUCTURE,
    "classification": ContentType.CLASSIFICATION, "types": ContentType.CLASSIFICATION,
    "categories": ContentType.CLASSIFICATION, "hierarchy_groups": ContentType.CLASSIFICATION,
    "cycle": ContentType.CYCLE,
    "chart": ContentType.GRAPH_DATA, "graph": ContentType.GRAPH_DATA, "table": ContentType.GRAPH_DATA,
    "history": ContentType.TIMELINE, "chronology": ContentType.TIMELINE,
    "map": ContentType.GEOGRAPHIC_MAP,
    "equation": ContentType.FORMULA_EQUATION, "formulae": ContentType.FORMULA_EQUATION,
    "worked_example": ContentType.WORKED_EXAMPLE, "solution": ContentType.WORKED_EXAMPLE,
    "problem": ContentType.WORKED_EXAMPLE,
    "geometry": ContentType.GEOMETRIC_DIAGRAM, "geometric": ContentType.GEOMETRIC_DIAGRAM,
    "flowchart": ContentType.FLOWCHART_ALGORITHM, "algorithm": ContentType.FLOWCHART_ALGORITHM,
    "architecture": ContentType.SYSTEM_ARCHITECTURE, "system": ContentType.SYSTEM_ARCHITECTURE,
    "hierarchy": ContentType.SYSTEM_ARCHITECTURE,
    "case": ContentType.CASE_STUDY,
    "definition": ContentType.DEFINITION_TERMINOLOGY, "terminology": ContentType.DEFINITION_TERMINOLOGY,
    "vocabulary": ContentType.DEFINITION_TERMINOLOGY, "terms": ContentType.DEFINITION_TERMINOLOGY,
    "recap": ContentType.SUMMARY, "conclusion": ContentType.SUMMARY,
    "takeaways": ContentType.SUMMARY, "key_points": ContentType.SUMMARY,
    "question": ContentType.REVISION_QUIZ, "mcq": ContentType.REVISION_QUIZ,
    "check": ContentType.REVISION_QUIZ, "practice": ContentType.REVISION_QUIZ,
    "revision": ContentType.REVISION_QUIZ,
}


def normalise_type(value: Any) -> Optional[ContentType]:
    """A declared type, or None when it names nothing we know."""
    raw = re.sub(r"[\s\-/]+", "_", str(value or "").strip().lower())
    if not raw:
        return None
    try:
        return ContentType(raw)
    except ValueError:
        return _ALIASES.get(raw)


# --- shape cues ---------------------------------------------------------------
#
# Generic vocabulary of SHAPE, matched on word boundaries. No topic names: a
# cue has to mean the same thing in any subject. Where a word is ambiguous on
# its own ("side", "region"), the scoring threshold - two hits, or one in the
# title or objective - is what keeps it from deciding alone.

_CUES: Dict[ContentType, Tuple[str, ...]] = {
    ContentType.CYCLE: (
        "cycle", "cyclic", "cyclical", "circulates", "circulation", "returns to",
        "round and round", "repeats itself"),
    ContentType.TIMELINE: (
        "timeline", "chronology", "chronological", "century", "centuries", "decade",
        "era", "dynasty", "reign", "in the year", "bce"),
    ContentType.COMPARISON: (
        "compare", "comparison", "comparing", "versus", "vs", "difference between",
        "differences between", "differ", "differs", "similarities", "contrast", "unlike",
        "whereas", "on the other hand"),
    ContentType.CLASSIFICATION: (
        "types of", "kinds of", "classification", "classified", "classify",
        "categories", "category", "grouped into", "divided into", "classes of"),
    ContentType.FLOWCHART_ALGORITHM: (
        "algorithm", "flowchart", "flow chart", "pseudocode", "pseudo code",
        "repeat until", "if else", "loop", "iterate", "iteration", "terminate",
        "decision box"),
    ContentType.SYSTEM_ARCHITECTURE: (
        "architecture", "components", "hierarchy", "hierarchical", "branches of",
        "tiers", "tier", "client", "server", "database", "network",
        "interacts with", "interact with"),
    ContentType.GEOGRAPHIC_MAP: (
        "map", "located", "location", "regions", "distribution", "latitude",
        "longitude", "continent", "river", "rivers", "basin", "coast", "coastal",
        "plateau", "plains", "spread across", "north", "south", "east", "west"),
    ContentType.GRAPH_DATA: (
        "graph", "bar graph", "pie chart", "histogram", "chart", "plot", "axis",
        "x axis", "y axis", "curve", "frequency", "tally", "trend", "slopes",
        "table of values"),
    ContentType.FORMULA_EQUATION: (
        "formula", "equation", "is given by", "is equal to", "proportional",
        "expression for", "law states"),
    ContentType.WORKED_EXAMPLE: (
        "solve", "solution", "worked example", "find the value", "step 1",
        "substitute", "substituting", "simplify", "calculate", "therefore",
        "transpose", "transposing", "check the answer"),
    ContentType.GEOMETRIC_DIAGRAM: (
        "angle", "angles", "triangle", "quadrilateral", "quadrilaterals", "polygon",
        "parallel", "perpendicular", "diagonal", "diagonals", "vertex", "vertices",
        "parallelogram", "rhombus", "trapezium", "rectangle", "radius", "diameter",
        "incident ray", "reflected ray", "angle of incidence", "angle of reflection",
        "ray diagram", "normal"),
    ContentType.EXPERIMENT_PROCEDURE: (
        "experiment", "activity", "apparatus", "materials required", "observation",
        "observe", "we observe", "conclusion", "test tube", "beaker", "set up",
        "setup", "measure the"),
    ContentType.CASE_STUDY: (
        "case study", "real life", "real world", "in the case of", "for instance",
        "example of"),
    ContentType.DEFINITION_TERMINOLOGY: (
        "definition", "define", "defined as", "meaning of", "is called", "are called",
        "refers to", "terminology", "term", "terms", "vocabulary", "is known as"),
    ContentType.CAUSE_EFFECT: (
        "causes", "cause of", "effects", "effect of", "leads to", "results in",
        "because of", "due to", "consequence", "consequences", "impact of"),
    ContentType.ANATOMY_STRUCTURE: (
        "structure of", "parts of", "labelled diagram", "labeled diagram",
        "cross section", "anatomy"),
    ContentType.PROCESS: (
        "process", "steps", "stages", "stage", "sequence"),
}

# When two types score equally, the more specific wins. CONCEPT is never in a
# tie - it is what is left when nothing scores.
_PRIORITY: Tuple[ContentType, ...] = (
    ContentType.WORKED_EXAMPLE, ContentType.FORMULA_EQUATION,
    ContentType.FLOWCHART_ALGORITHM, ContentType.GEOMETRIC_DIAGRAM,
    ContentType.GRAPH_DATA, ContentType.TIMELINE, ContentType.CYCLE,
    ContentType.GEOGRAPHIC_MAP, ContentType.EXPERIMENT_PROCEDURE,
    ContentType.SYSTEM_ARCHITECTURE, ContentType.COMPARISON,
    ContentType.CLASSIFICATION, ContentType.CAUSE_EFFECT,
    ContentType.ANATOMY_STRUCTURE, ContentType.DEFINITION_TERMINOLOGY,
    ContentType.CASE_STUDY, ContentType.PROCESS,
)

# Evidence needed to move a slide off the CONCEPT default. One hit in body
# text alone is not enough; one in the title or objective (weighted 2) is.
INFERENCE_THRESHOLD = 2

_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
# A relation between named quantities: v = u + at, F = ma, A = l x b,
# 2x + 3 = 7. A bare "=" in prose does not count.
_FORMULA = re.compile(
    r"(?:\b[A-Za-z]{1,3}\s*(?:=|∝)\s*[A-Za-z0-9(√π][^.;]{0,40})"
    r"|(?:\b\d*[a-z]\s*[+\-×x*/]\s*\d+\s*=\s*-?\d+)")
_NUMERIC_STEP = re.compile(r"\d")


@dataclass
class ContentAnalysis:
    """Everything the strategy layer needs to know about one slide's content."""

    subject: str = ""
    class_level: str = ""
    chapter: str = ""
    topic: str = ""
    content_type: ContentType = DEFAULT_TYPE
    learning_objective: str = ""
    key_concepts: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    required_visual_elements: List[str] = field(default_factory=list)
    required_labels: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    numerical_steps: List[str] = field(default_factory=list)
    class_level_constraints: List[str] = field(default_factory=list)
    visual_complexity: str = "low"
    # Provenance: how the type was reached, and the evidence. Recorded so a
    # deck can say WHY a slide came out the shape it did.
    declared_type: Optional[ContentType] = None
    source: str = "default"           # declared | inferred | corrected | default
    reason: str = ""
    scores: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject, "classLevel": self.class_level,
            "chapter": self.chapter, "topic": self.topic,
            "contentType": self.content_type.value,
            "learningObjective": self.learning_objective,
            "keyConcepts": list(self.key_concepts),
            "relationships": list(self.relationships),
            "requiredVisualElements": list(self.required_visual_elements),
            "requiredLabels": list(self.required_labels),
            "formulas": list(self.formulas),
            "numericalSteps": list(self.numerical_steps),
            "classLevelConstraints": list(self.class_level_constraints),
            "visualComplexity": self.visual_complexity,
            "declaredType": self.declared_type.value if self.declared_type else None,
            "source": self.source, "reason": self.reason,
        }


# --- reading a planner slide -------------------------------------------------

def _texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            out.extend(_texts(v))
        return out
    if isinstance(value, dict):
        return _texts(value.get("text") or value.get("label") or value.get("name"))
    return [str(value)]


def _pick(data: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in data and data[k] not in (None, "", []):
            return data[k]
    return None


def _blocks(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = _pick(slide, "blocks", "contentBlocks", "content_blocks", "bullets", "points") or []
    out = []
    for b in raw if isinstance(raw, list) else []:
        if isinstance(b, str):
            out.append({"text": b})
        elif isinstance(b, dict):
            out.append(b)
    return out


def _visual(slide: Dict[str, Any]) -> Dict[str, Any]:
    v = _pick(slide, "visual", "image") or {}
    if isinstance(v, str):
        return {"description": v}
    return v if isinstance(v, dict) else {}


def _dedupe(items: Iterable[str], limit: int = 12) -> List[str]:
    seen, out = set(), []
    for item in items:
        t = re.sub(r"\s+", " ", str(item or "")).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:limit]


def extract_formulas(texts: Sequence[str]) -> List[str]:
    found = []
    for t in texts:
        for m in _FORMULA.finditer(t or ""):
            found.append(m.group(0).strip())
    return _dedupe(found, limit=6)


def heading_hits(heading: str) -> Dict[ContentType, int]:
    """Which types the title/subtitle/objective itself names."""
    return {t: 1 for t, cues in _CUES.items() if matches_any(heading, cues)}


def score_types(heading: str, body: Sequence[str], visual: str = "",
                block_texts: Sequence[str] = ()) -> Dict[ContentType, int]:
    """Shape evidence per type. Heading hits count double."""
    scores: Dict[ContentType, int] = {}
    for ctype, cues in _CUES.items():
        s = 2 if matches_any(heading, cues) else 0
        s += sum(1 for t in body if matches_any(t, cues))
        s += 1 if visual and matches_any(visual, cues) else 0
        if s:
            scores[ctype] = s

    # Structural evidence that vocabulary alone misses.
    dated = sum(1 for t in block_texts if _YEAR.search(t or ""))
    if dated >= 2:
        scores[ContentType.TIMELINE] = scores.get(ContentType.TIMELINE, 0) + 3
    if extract_formulas(list(block_texts) + [heading]):
        scores[ContentType.FORMULA_EQUATION] = scores.get(ContentType.FORMULA_EQUATION, 0) + 2
    return scores


# --- structural signals (V2: safe) -------------------------------------------------
#
# What the planner's payload SAYS about shape, independently of its wording.
#
# V1 read structure but trusted it too readily. A held-out probe caught four
# false positives: a numbered list became a process, concept bullets that
# gave examples became definitions, the parts of one eye and the forces on
# one book became "system architecture". A wrong specific type picks a wrong
# representation, while CONCEPT_EXPLANATION always renders safely - so V2 asks
# each signal for evidence DISTINCTIVE of its shape, not merely compatible:
#
#   sequence    marking (a step kind or an ordinal label) is not enough. The
#               later stages must BUILD ON earlier ones - carry a word forward
#               or refer back to it - and framing repeated in every stage is
#               parallel structure, not progression.
#   worked      an answer carried forward: a result computed in one line is
#               used in a later one. Numbers in a numbered list are not that.
#   definition  explicit definition blocks, or terms whose text is written as
#               a definition. Giving an example is what concept bullets do.
#   system      relationships whose subject is a component and whose object is
#               another component, over three or more components - not under a
#               part-whole framing, and not when everything acts on one
#               passive object.
#
# Vocabulary used: grammatical only (pronouns, "if"/"otherwise", definitional
# copulas such as "is called") plus the existing part-whole cue list. No
# subject or topic words.

_ORDINAL_LABEL = re.compile(r"^\s*(?:step\s*)?(?:0?[1-9]|1[0-9]|20)\s*[.):-]?\s*$", re.I)
_BRANCH = re.compile(r"^\s*if\b|\b(?:otherwise|else)\b", re.I)
_ARITHMETIC = re.compile(r"\d\s*[=+\-×*/÷]\s*\(?\d")
_RESULT = re.compile(r"=\s*(\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A pronoun within the first three words: the stage is about what came before.
_BACK_REFERENCE = re.compile(r"^\W*(?:\w+\W+){0,2}(?:it|its|this|these|they|them|those)\b", re.I)
# Text written as a definition of its term.
_DEFINES = re.compile(
    r"\b(?:is called|are called|is known as|are known as|refers to|refer to|"
    r"stands for|means|is defined as|are defined as)\b"
    r"|^(?:a|an|the)\s+(?:word|term|name)\s+(?:for|that|which|used|given)\b", re.I)
_COPULA_AFTER_TERM = re.compile(r"^(?:is|are|means|refers to|stands for)\b", re.I)

# Short function words that carry no content; a shared one is not a link.
_STOP = frozenset("""
about above after again against also among another around because been before
being below between both could does doing down during each either every from
further have having here into itself just more most much must near never only
other over same should some such than that their theirs them then there these
they this those through under until upon very were what when where which while
whom whose will with within without would your yours make makes made take takes
took uses used using become becomes became keep keeps kept give gives given
""".split())

# Evidence a specific type needs over its nearest rival before it is trusted.
DISTINCT_MARGIN = 2


def _kind(block: Dict[str, Any]) -> str:
    return str(block.get("kind") or "").strip().lower()


def _is_marked(block: Dict[str, Any]) -> bool:
    """Positionally marked by the planner: a step block or an ordinal label.

    The ONE definition of "marked" used everywhere - structural scoring and
    the worked-example fillability check alike. Marking is a claim of order;
    whether the content honours it is decided by sequence_evidence.
    """
    return _kind(block) == "step" or bool(_ORDINAL_LABEL.match(str(block.get("label") or "")))


def _stem(word: str) -> str:
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and re.search(r"(?:s|x|z|ch|sh)es$", w):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _content_words(text: str, exclude: FrozenSet[str] = frozenset()) -> set:
    out = set()
    for word in re.findall(r"[A-Za-z]{4,}", text or ""):
        stem = _stem(word)
        if word.lower() not in _STOP and stem not in exclude:
            out.add(stem)
    return out


@dataclass
class SequenceEvidence:
    marked: List[int] = field(default_factory=list)
    builds: int = 0          # later marked stages that build on an earlier one
    needed: int = 0

    @property
    def is_sequence(self) -> bool:
        return len(self.marked) >= 3 and self.builds >= self.needed


def sequence_evidence(blocks: Sequence[Dict[str, Any]], heading: str = "") -> SequenceEvidence:
    """Is this an ordered sequence in which later stages depend on earlier ones?

    1 -> 2 -> 3 where each stage works on what the last produced is a
    sequence. "1. item  2. item  3. item" is a list, however it is numbered.
    """
    marked = [i for i, b in enumerate(blocks) if _is_marked(b)]
    ev = SequenceEvidence(marked=marked)
    if len(marked) < 3:
        return ev
    texts = [" ".join(_texts(blocks[i].get("text"))) for i in marked]
    # The topic is named everywhere; repeating it is not progression.
    exclude = frozenset(_stem(w) for w in re.findall(r"[A-Za-z]{4,}", heading or ""))
    words = [_content_words(t, exclude) for t in texts]
    # Framing repeated in EVERY stage is parallel structure.
    common = set.intersection(*words) if words else set()
    for k in range(1, len(texts)):
        earlier = set().union(*words[:k]) - common
        if _BACK_REFERENCE.search(texts[k]) or ((words[k] - common) & earlier):
            ev.builds += 1
    later = len(marked) - 1
    ev.needed = -(-2 * later // 3)            # ceil(2/3 of the later stages)
    return ev


def worked_evidence(blocks: Sequence[Dict[str, Any]]) -> int:
    """Lines that reuse a result computed earlier: an answer progression."""
    marked = [i for i, b in enumerate(blocks) if _is_marked(b)]
    if len(marked) < 2:
        return 0
    texts = [" ".join(_texts(blocks[i].get("text"))) for i in marked]
    if sum(1 for t in texts if _ARITHMETIC.search(t)) < 2:
        return 0
    carried, results = 0, set()
    for t in texts:
        if results & set(_NUMBER.findall(t)):
            carried += 1
        results |= set(_RESULT.findall(t))
    return carried


def _entity_names(visual: Dict[str, Any]) -> List[str]:
    """Names the picture is about: its objects, required items and labels.

    Kept to short names (four words or fewer) - an annotation that is a whole
    sentence is a caption, not an entity.
    """
    raw = (_texts(visual.get("objects"))
           + _texts(visual.get("mustShow") or visual.get("must_show"))
           + _texts(visual.get("annotations") or visual.get("labels")))
    out: List[str] = []
    for name in raw:
        name = re.sub(r"\([^)]*\)", " ", str(name).lower())
        name = re.sub(r"\s+", " ", name).strip()
        if len(name) >= 3 and len(name.split()) <= 4 and name not in out:
            out.append(name)
    return out


def _first_entity(text: str, entities: Sequence[str]) -> Optional[str]:
    """The entity mentioned earliest in `text`, whole-word and plural-tolerant."""
    best, best_pos = None, None
    low = re.sub(r"\s+", " ", (text or "").lower())
    for e in entities:
        m = re.search(r"\b%s(?:s|es)?\b" % r"\s+".join(re.escape(w) for w in e.split()), low)
        if m and (best_pos is None or m.start() < best_pos):
            best, best_pos = e, m.start()
    return best


def component_interactions(blocks: Sequence[Dict[str, Any]], visual: Dict[str, Any],
                           heading: str, seq: SequenceEvidence) -> Tuple[int, str]:
    """(nodes, note): components that act on OTHER components.

    Zero unless all of these hold:
      - no part-whole framing (heading or picture say these belong to one
        object - its parts are not independent components);
      - not an ordered sequence (stages through components are a process);
      - three or more blocks each about a distinct named component;
      - relationships whose SUBJECT is one of those components and which name
        another, together covering three or more components;
      - no passive hub: nothing that every relationship acts on while never
        acting itself (several forces on one object are not a system).
    """
    entities = _entity_names(visual)
    if not entities:
        return 0, ""
    framing = "%s %s" % (heading, " ".join(_texts(visual.get("description"))))
    if matches_any(framing, _CUES[ContentType.ANATOMY_STRUCTURE]):
        return 0, "part-whole framing"
    if seq.is_sequence:
        return 0, "ordered sequence"

    subjects: List[str] = []
    for blk, label in zip(blocks, normalise_labels(blocks)):
        text = " ".join(_texts(blk.get("text")))
        subject = (_first_entity(label, entities)
                   or _first_entity(" ".join(text.split()[:6]), entities))
        if subject and subject not in subjects:
            subjects.append(subject)
    if len(subjects) < 3:
        return 0, ""

    edges = []
    for rel in _texts(visual.get("relationships")):
        lead = " ".join(rel.split()[:4])          # article + the subject
        subject = _first_entity(lead, subjects)
        if not subject:
            continue
        others = [e for e in subjects if e != subject and _first_entity(rel, [e])]
        if others:
            edges.append((subject, others))
    nodes = {s for s, _ in edges} | {o for _, objs in edges for o in objs}
    if len(nodes) < 3:
        return 0, ""
    if len(edges) >= 2:
        actors = {s for s, _ in edges}
        for node in nodes:
            if node not in actors and all(node in objs for _, objs in edges):
                return 0, "everything acts on one passive object"
    return len(nodes), ""


# Leading numbering on a label: "1. Egg", "4) Pupa", "Step 2: Mix". One or two
# digits only, so a year ("1857 Revolt") is never taken for numbering.
_LEADING_ORDINAL = re.compile(r"^\s*(?:step\s*)?(\d{1,2})\s*[.):\-]\s*(?=\S)", re.I)
# A bare leading number ("03 Adult") is numbering only when the labels count
# up together from 0 or 1; on its own it may be content ("12 Months").
_BARE_ORDINAL = re.compile(r"^\s*(\d{1,2})\s+(?=[A-Za-z])")


def normalise_labels(blocks: Sequence[Dict[str, Any]]) -> List[str]:
    """Block labels with structural numbering removed, for comparison only.

    The blocks themselves are not changed - the slide still renders "1. Egg"
    - but asking whether a later stage refers back to an earlier one needs
    "Egg". A label that is nothing but an ordinal ("1", "Step 2") names
    nothing and normalises to "".
    """
    raw = [" ".join(_texts(b.get("label"))) for b in blocks]
    bare = [_BARE_ORDINAL.match(lab) for lab in raw]
    numbers = [int(m.group(1)) for m in bare if m]
    counting = (len(numbers) >= 2 and numbers[0] in (0, 1)
                and numbers == list(range(numbers[0], numbers[0] + len(numbers))))
    out: List[str] = []
    for lab, m in zip(raw, bare):
        if _ORDINAL_LABEL.match(lab):
            out.append("")
            continue
        stripped = _LEADING_ORDINAL.sub("", lab, count=1)
        if stripped == lab and m and counting:
            stripped = lab[m.end():]
        out.append(stripped.strip())
    return out


def _refers_back(label: str, text: str) -> bool:
    """Does `text` name what `label` names?

    Whole word and plural-tolerant ("Egg" in "lays eggs"). A multi-word label
    may be referred to by its head - its last significant word.
    """
    label = (label or "").strip().lower()
    if not label or len(label.split()) > 4:
        return False
    if matches_any(text, (label,)):
        return True
    words = [w for w in re.findall(r"[a-z]{3,}", label) if w not in _STOP]
    return len(label.split()) > 1 and bool(words) and matches_any(text, (words[-1],))


def structural_scores(blocks: Sequence[Dict[str, Any]], visual: Dict[str, Any],
                      heading: str = "") -> Tuple[Dict[ContentType, int], List[str]]:
    """(scores, signals): distinctive shape evidence from structure alone.

    Added to the lexical scores. The chooser then demands a clear margin
    before trusting any specific type; see _choose.
    """
    scores: Dict[ContentType, int] = {}
    signals: List[str] = []

    def add(ctype: ContentType, points: int, signal: str) -> None:
        scores[ctype] = scores.get(ctype, 0) + points
        signals.append(signal)

    texts = [" ".join(_texts(b.get("text"))) for b in blocks]
    # Numbering stripped ("1. Egg" -> "Egg"): every comparison below is about
    # what a label NAMES, not where it sits.
    labels = normalise_labels(blocks)
    seq = sequence_evidence(blocks, heading)

    # A sequence is read at its most specific: an answer progression, a loop
    # back to the start, or a branch. The specific reading TAKES the sequence
    # evidence rather than competing with a generic process for it - scored
    # side by side, a cycle's own sequence became its rival and the margin
    # rule threw both away.
    carried = worked_evidence(blocks)
    if carried >= 1:
        add(ContentType.WORKED_EXAMPLE, 3, "answer carried forward x%d" % carried)
    elif seq.is_sequence:
        seq_note = "dependent sequence %d/%d" % (seq.builds, len(seq.marked) - 1)
        first = labels[seq.marked[0]]
        if first and _refers_back(first, texts[seq.marked[-1]]):
            add(ContentType.CYCLE, 2 + 3, seq_note + " that closes on its first stage")
        elif any(_BRANCH.search(texts[i]) for i in seq.marked):
            add(ContentType.FLOWCHART_ALGORITHM, 2 + 2, seq_note + " with a conditional branch")
        else:
            add(ContentType.PROCESS, 2, seq_note)

    # Terms, each with a meaning written as a definition.
    defined = sum(1 for b in blocks if _kind(b) == "definition")
    if defined >= 2:
        add(ContentType.DEFINITION_TERMINOLOGY, 3, "definition blocks x%d" % defined)
    else:
        glossed = 0
        for lab, txt in zip(labels, texts):
            term = lab.strip().lower()
            if not term or _ORDINAL_LABEL.match(term) or len(term.split()) > 4:
                continue
            body = txt.strip().lower()
            if (_DEFINES.search(body) or _COPULA_AFTER_TERM.match(body)
                    or (body.startswith(term + " ")
                        and _COPULA_AFTER_TERM.match(body[len(term) + 1:]))):
                glossed += 1
        if glossed >= 2 and glossed * 2 >= len(blocks):
            add(ContentType.DEFINITION_TERMINOLOGY, 2, "term->meaning pairs x%d" % glossed)

    # Two sides set against each other.
    kinds = {_kind(b) for b in blocks}
    if {"compare_left", "compare_right"} <= kinds:
        add(ContentType.COMPARISON, 3, "paired left/right blocks")

    # An equation stated as one; the letter-formula regex in score_types
    # already counts the written-out kind.
    if (kinds & {"equation", "formula"}
            and not extract_formulas(texts + labels + [heading])):
        add(ContentType.FORMULA_EQUATION, 2, "equation block")

    # The planner says the picture is a chart.
    if str(visual.get("kind") or "").strip().lower() == "chart":
        add(ContentType.GRAPH_DATA, 2, "chart visual")

    nodes, note = component_interactions(blocks, visual, heading, seq)
    if nodes:
        add(ContentType.SYSTEM_ARCHITECTURE, 3, "%d interacting components" % nodes)

    return scores, signals


# --- specificity -----------------------------------------------------------------
#
# A dependent sequence is evidence that SOMETHING is ordered. Process is the
# generic reading of that; a cycle, an algorithm, a worked solution and an
# experiment are specific readings of the same structure. The real-planner
# benchmark showed the generic reading winning on points: two life cycles,
# each named a cycle in its own heading, were read as processes because the
# sequence's points went to process. When a specific sequence type has its
# OWN evidence - enough to qualify, and either named by the heading or found
# in the structure - the sequence points go to it instead. A passing mention
# in the body is not enough, and two specific readings tied move nothing.

SEQUENCE_SPECIFIC: Tuple[ContentType, ...] = (
    ContentType.WORKED_EXAMPLE, ContentType.CYCLE,
    ContentType.FLOWCHART_ALGORITHM, ContentType.EXPERIMENT_PROCEDURE,
)

# Planner declarations that commit to no particular shape. CONCEPT is the
# planner's own "unsure" and is already inferred; PROCESS is the declaration
# the real planner fell back to for cycles, algorithms, worked solutions and
# experiments. Every other declaration is specific and is never overridden.
GENERIC_DECLARATIONS = frozenset({ContentType.PROCESS})
# Evidence an inferred type needs before it may replace a generic declaration.
STRONG_EVIDENCE = 4


def credit_specific_sequence(scores: Dict[ContentType, int],
                             lexical: Dict[ContentType, int],
                             structure: Dict[ContentType, int],
                             named: Dict[ContentType, int],
                             signals: List[str]) -> Optional[ContentType]:
    """Move generic sequence points to a specific sequence type. In place.

    Structural PROCESS points only ever come from a dependent sequence that no
    specific structural reading claimed, so they are exactly the points that
    may move. Returns the type credited, or None.
    """
    generic = structure.get(ContentType.PROCESS, 0)
    if not generic:
        return None
    qualified = []
    for ctype in SEQUENCE_SPECIFIC:
        own = lexical.get(ctype, 0) + structure.get(ctype, 0)
        if own >= INFERENCE_THRESHOLD and (named.get(ctype) or structure.get(ctype)):
            qualified.append((own, ctype))
    if not qualified:
        return None
    qualified.sort(key=lambda q: q[0], reverse=True)
    if len(qualified) > 1 and qualified[0][0] == qualified[1][0]:
        return None
    target = qualified[0][1]
    scores[ContentType.PROCESS] = scores.get(ContentType.PROCESS, 0) - generic
    if scores[ContentType.PROCESS] <= 0:
        scores.pop(ContentType.PROCESS)
    scores[target] = scores.get(target, 0) + generic
    signals.append("sequence credited to %s, the more specific reading" % target.value)
    return target


def _choose(scores: Dict[ContentType, int], exclude: Sequence[ContentType] = (),
            heading: Optional[Dict[ContentType, int]] = None
            ) -> Tuple[Optional[ContentType], int, str]:
    """(type or None, score, note). Prefer a specific type only when distinctive.

    1. Only types at or above the threshold are eligible.
    2. Highest score wins; ties go to the heading-named type, then priority.
    3. A type the heading names is not overridden unless the rival leads it
       by DISTINCT_MARGIN.
    4. A winner the heading does not name must lead its nearest eligible
       rival by DISTINCT_MARGIN; otherwise the evidence is ambiguous and the
       answer is the safe default.
    """
    heading = heading or {}
    ranked = [(scores.get(t, 0), heading.get(t, 0), -i, t)
              for i, t in enumerate(_PRIORITY) if t not in exclude]
    ranked.sort(key=lambda r: r[:3], reverse=True)
    eligible = [r for r in ranked if r[0] >= INFERENCE_THRESHOLD]
    if not eligible:
        return None, ranked[0][0] if ranked else 0, ""
    top = eligible[0]
    named = [r for r in eligible if r[1]]
    if named and not top[1] and top[0] < named[0][0] + DISTINCT_MARGIN:
        top = named[0]
    rivals = [r for r in eligible if r[3] is not top[3]]
    if rivals and not top[1] and top[0] - rivals[0][0] < DISTINCT_MARGIN:
        return None, top[0], "ambiguous between %s and %s" % (top[3].value, rivals[0][3].value)
    return top[3], top[0], ""


def _best(scores: Dict[ContentType, int], exclude: Sequence[ContentType] = (),
          heading: Optional[Dict[ContentType, int]] = None
          ) -> Tuple[Optional[ContentType], int]:
    chosen, score, _note = _choose(scores, exclude, heading)
    return chosen, score


def fillable(ctype: ContentType, blocks: Sequence[Dict[str, Any]],
             block_texts: Sequence[str], formulas: Sequence[str],
             numeric_steps: Sequence[str]) -> Tuple[bool, str]:
    """Can this slide's content actually carry the declared type?

    The checks are the minimum the shape needs to exist at all - not a
    quality bar. A comparison needs two sides; a timeline needs dates; a
    worked example needs worked numbers.
    """
    n = len(blocks)
    if ctype is ContentType.COMPARISON and n < 2:
        return False, "a comparison needs two sides; the slide has %d block(s)" % n
    if ctype is ContentType.TIMELINE:
        dated = sum(1 for t in block_texts if _YEAR.search(t or "")
                    or matches_any(t, _CUES[ContentType.TIMELINE]))
        if dated < 2:
            return False, "a timeline needs at least two dated points; found %d" % dated
    if ctype is ContentType.FORMULA_EQUATION and not formulas:
        return False, "declared a formula, but no formula appears on the slide"
    if ctype is ContentType.WORKED_EXAMPLE and len(numeric_steps) < 2:
        return False, ("a worked example needs at least two worked steps; found %d"
                       % len(numeric_steps))
    if ctype in (ContentType.PROCESS, ContentType.CYCLE,
                 ContentType.FLOWCHART_ALGORITHM, ContentType.EXPERIMENT_PROCEDURE) and n < 2:
        return False, "a sequence needs at least two stages; the slide has %d" % n
    return True, ""


def _complexity(n: int) -> str:
    return "low" if n <= 3 else ("medium" if n <= 7 else "high")


def analyse(slide: Dict[str, Any], *, subject: str = "", class_level: str = "",
            chapter: str = "", topic: str = "") -> ContentAnalysis:
    """Classify one planner slide. Never raises; unknown -> CONCEPT_EXPLANATION."""
    slide = slide if isinstance(slide, dict) else {}
    blocks = _blocks(slide)
    visual = _visual(slide)

    title = " ".join(_texts(_pick(slide, "title", "heading")))
    subtitle = " ".join(_texts(_pick(slide, "subtitle")))
    objective = " ".join(_texts(_pick(slide, "learningObjective", "learning_objective",
                                      "objective")))
    block_texts = [" ".join(_texts(b.get("label")) + _texts(b.get("text"))) for b in blocks]
    vdesc = " ".join(_texts(visual.get("description")))

    formulas = extract_formulas(block_texts + [title, subtitle])
    formulas += [t for b in blocks if str(b.get("kind", "")).lower() in ("formula", "equation")
                 for t in _texts(b.get("text"))]
    formulas = _dedupe(formulas, limit=6)
    # The same notion of a marked stage that structural scoring uses: a step
    # block or an ordinal label. A worked example numbered "1, 2, 3" was
    # rejected here while the scorer counted it, and fell through to process.
    numeric_steps = [t for b, t in zip(blocks, block_texts)
                     if _is_marked(b) and _NUMERIC_STEP.search(t)]

    declared_raw = _pick(slide, "contentType", "content_type")
    declared = normalise_type(declared_raw)
    legacy = normalise_type(_pick(slide, "slideType", "slide_type", "type"))
    declared = declared or legacy

    heading = " ".join(x for x in (title, subtitle, objective) if x)
    scores = score_types(heading, block_texts, vdesc, block_texts)
    lexical = dict(scores)
    structure, signals = structural_scores(blocks, visual, heading)
    for ctype, points in structure.items():
        scores[ctype] = scores.get(ctype, 0) + points
    named = heading_hits(heading)
    credit_specific_sequence(scores, lexical, structure, named, signals)

    analysis = ContentAnalysis(subject=subject, class_level=class_level,
                               chapter=chapter, topic=topic,
                               learning_objective=objective, declared_type=declared,
                               formulas=formulas, numerical_steps=numeric_steps,
                               scores={t.value: s for t, s in scores.items()})

    if declared in _STRUCTURAL:
        analysis.content_type, analysis.source = declared, "declared"
        analysis.reason = "%s is decided by its place in the deck" % declared.value
    elif declared and declared is not DEFAULT_TYPE:
        ok, why = fillable(declared, blocks, block_texts, formulas, numeric_steps)
        upgrade = None
        if ok and declared in GENERIC_DECLARATIONS:
            # A generic declaration is kept unless the content says, clearly,
            # something more specific. Specific declarations are never touched.
            candidate, cscore, _note = _choose(scores, heading=named)
            if (candidate is not None and candidate not in (declared, DEFAULT_TYPE)
                    and cscore >= STRONG_EVIDENCE
                    and cscore - scores.get(declared, 0) >= DISTINCT_MARGIN
                    and fillable(candidate, blocks, block_texts, formulas, numeric_steps)[0]):
                upgrade = (candidate, cscore)
        if upgrade:
            analysis.content_type, analysis.source = upgrade[0], "upgraded"
            analysis.reason = ("planner declared %s, a generic shape; the content shows %s "
                               "clearly (score %d vs %d)%s" % (
                                   declared.value, upgrade[0].value, upgrade[1],
                                   scores.get(declared, 0),
                                   "; structure: %s" % ", ".join(signals) if signals else ""))
        elif ok:
            analysis.content_type, analysis.source = declared, "declared"
            analysis.reason = "planner declared %s and the content can carry it" % declared.value
        else:
            inferred, score = _best(scores, exclude=(declared,), heading=named)
            analysis.content_type = inferred or DEFAULT_TYPE
            analysis.source = "corrected"
            analysis.reason = "declared %s rejected: %s; %s" % (
                declared.value, why,
                "inferred %s (score %d)" % (inferred.value, score) if inferred
                else "no stronger shape found, so concept explanation")
    else:
        inferred, score, note = _choose(scores, heading=named)
        if inferred:
            ok, why = fillable(inferred, blocks, block_texts, formulas, numeric_steps)
            if not ok:
                inferred2, score2 = _best(scores, exclude=(inferred,), heading=named)
                if inferred2 and fillable(inferred2, blocks, block_texts,
                                          formulas, numeric_steps)[0]:
                    inferred, score = inferred2, score2
                else:
                    inferred = None
        if inferred:
            analysis.content_type, analysis.source = inferred, "inferred"
            analysis.reason = "%s from the content's shape cues (score %d)%s" % (
                inferred.value, score,
                "; structure: %s" % ", ".join(signals) if signals else "")
        else:
            analysis.content_type, analysis.source = DEFAULT_TYPE, (
                "declared" if declared is DEFAULT_TYPE else "default")
            analysis.reason = ("%s; explain the concept" % note if note
                               else "no shape cue strong enough; explain the concept")

    analysis.key_concepts = _dedupe(
        _texts(_pick(slide, "highlightedTerms", "keyTerms", "key_terms"))
        + [str(b.get("label") or "") for b in blocks], limit=8)
    analysis.relationships = _dedupe(_texts(visual.get("relationships")), limit=6)
    analysis.required_visual_elements = _dedupe(
        _texts(visual.get("mustShow") or visual.get("must_show"))
        + _texts(visual.get("objects")), limit=8)
    analysis.required_labels = _dedupe(
        _texts(visual.get("annotations") or visual.get("labels")), limit=8)
    analysis.visual_complexity = _complexity(
        len(analysis.required_visual_elements) + len(analysis.relationships)
        + len(analysis.required_labels))

    try:
        from ai_services.core.ppt_v2 import class_rules
        analysis.class_level_constraints = list(
            class_rules.all_forbidden(class_level, subject, chapter or topic))[:12]
    except Exception:
        analysis.class_level_constraints = []

    return analysis
