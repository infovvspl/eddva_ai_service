"""Scoring a candidate image against what the slide actually needs.

The defect this exists to fix: an introductory "What is Photosynthesis?"
slide was given a diagram of *photosynthesis and respiration*. Tracing it,
the cause was not that diagram or that word - it was that relevance was
one-sided and positional:

  * ``_caption_is_relevant`` answers one question, "does the caption share a
    word with my query?" The caption "Photosynthesis and Respiration" shares
    "photosynthesis", so it passes;
  * the search loop returns the **first** candidate that passes, so a
    passable result that happens to rank highly beats a better one below it.

So a candidate carrying a second, competing process scored the same as one
carrying only the subject, and won on position. Any topic with a well-known
counterpart has the same problem: mitosis pulls meiosis, arteries pull veins,
series circuits pull parallel ones.

The fix is to score both directions - reward what the slide requires, punish
what it should not contain - and then pick the best candidate rather than the
first acceptable one.

Deliberately deterministic and free: it reads the title and snippet the
search provider already returns. No multimodal call, no second LLM pass, and
therefore no per-image cost.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ai_services.core.ppt_v2.textmatch import matches_any

logger = logging.getLogger(__name__)

# Confidence bands, and what each one costs.
#
# The deterministic score already separates "clearly right" from "clearly
# wrong" for free. What it cannot separate is the middle: a caption that is
# on topic and a host that is reputable, where the picture may still teach
# nothing. That middle band is the only place a vision call buys anything,
# so it is the only place one is spent.
#
#   score >= CONFIDENT   HIGH    accept as is, no vision call
#   ACCEPT..CONFIDENT    MEDIUM  ask a vision model, if one is enabled
#   score <  ACCEPT      LOW     reject; fall through to another strategy
#
# Both are env-overridable so the band can be widened or closed entirely
# without a release.
ACCEPT_THRESHOLD = 1.0
CONFIDENT_THRESHOLD = 3.0

HIGH = "high"
MEDIUM = "medium"
LOW = "low"


def _env_float(name, default):
    try:
        return float((os.getenv(name, "") or "").strip() or default)
    except (TypeError, ValueError):
        return default


def thresholds():
    """(accept, confident), from configuration."""
    accept = _env_float("PPT_V2_RELEVANCE_ACCEPT", ACCEPT_THRESHOLD)
    confident = _env_float("PPT_V2_RELEVANCE_CONFIDENT", CONFIDENT_THRESHOLD)
    # A confident bar below the accept bar would mean "accept everything
    # without looking", which is the opposite of what it is for.
    return accept, max(confident, accept)


def confidence(score):
    """Which band a score falls in.

    An absent score lands in MEDIUM, not HIGH: a caller that could not tell
    us how confident it was has not earned the benefit of the doubt, and
    MEDIUM is the band that gets looked at. (A None also used to raise here,
    which would have taken the whole asset path down.)
    """
    if score is None:
        return MEDIUM
    try:
        score = float(score)
    except (TypeError, ValueError):
        return MEDIUM
    accept, confident = thresholds()
    if score >= confident:
        return HIGH
    if score >= accept:
        return MEDIUM
    return LOW


def needs_vision(score):
    """Is this the ambiguous middle, where a vision call is worth its cost?"""
    return confidence(score) == MEDIUM

# Weights. Required concepts matter more than the bare topic word, because
# "photosynthesis" appears in everything the query returns and tells us
# almost nothing; "stomata" or "chloroplast" tells us the figure is the right
# *kind* of figure.
W_TOPIC = 1.0
W_REQUIRED = 1.2
W_VISUAL_KIND = 0.4
W_CONFLICT = -2.5          # outweighs a topic match on its own, by design
W_ADVANCED = -1.2
W_JUNK = -1.5
W_UNSUITABLE = -3.0        # a product listing or a video thumbnail
W_TRUSTED = 0.8            # textbooks, universities, reference works

# Concepts that habitually travel together in search results but teach
# different things. Listing a pair here means "if the slide is about one, a
# figure that also covers the other is worse, not richer".
#
# Not a blocklist: nothing is forbidden outright, and a slide that genuinely
# wants the comparison declares it in `required`, which cancels the penalty.
CONCEPT_CONFLICTS: Dict[str, Tuple[str, ...]] = {
    "photosynthesis": ("respiration", "cellular respiration", "krebs",
                       "glycolysis", "mitochondri"),
    "respiration": ("photosynthesis", "chloroplast"),
    "mitosis": ("meiosis",),
    "meiosis": ("mitosis",),
    # Stems, not dictionary forms: matching is prefix-at-a-word-boundary, so
    # the key "artery" never matched "arteries". "arter" covers artery,
    # arteries and arterial.
    "arter": ("vein", "venous"),
    "vein": ("arter",),
    "series circuit": ("parallel circuit",),
    "parallel circuit": ("series circuit",),
    "reflection": ("refraction",),
    "refraction": ("reflection",),
    "conduction": ("convection", "radiation"),
    "convection": ("conduction", "radiation"),
    "evaporation": ("condensation",),
    "condensation": ("evaporation",),
    "solid": ("liquid", "gas"),
    "mass": ("weight",),
    "weight": ("mass",),
    "speed": ("acceleration",),
    "asexual reproduction": ("sexual reproduction",),
    "sexual reproduction": ("asexual reproduction",),
}

# Terminology that marks a figure as pitched above the junior-secondary
# classes. Deliberately short: this is a signal, not a curriculum.
ADVANCED_TERMS: Tuple[str, ...] = (
    "krebs", "calvin cycle", "thylakoid", "nadph", "atp synthase",
    "electron transport", "glycolysis", "oxidative phosphorylation",
    "quantum", "eigen", "differential equation", "integral", "logarithm",
    "hybridi[sz]ation", "stoichiometr", "enthalp", "entrop", "molarity",
    "gibbs", "lagrang", "hamiltonian",
)

# Results whose captions mark them as the wrong *object* entirely: a
# worksheet, a book cover, a quiz, a slide of somebody else's deck.
JUNK_TERMS: Tuple[str, ...] = (
    "worksheet", "answer key", "quiz", "question paper", "mcq", "test paper",
    "book cover", "buy now", "amazon", "for sale", "price", "coupon",
    "powerpoint template", "ppt template", "free download", "clipart set",
    "stock photo", "shutterstock", "watermark", "sample paper",
)

# Hosts whose images are not teaching figures, whatever the caption says.
# Three live examples, all on topic and all useless: an Etsy product photo on
# a Force and Pressure slide, a watermarked sciencephoto.com preview on a
# Sound slide, and a YouTube "maxresdefault" thumbnail on a Light slide.
# Caption scoring could not see any of them. The host can.
UNSUITABLE_HOSTS: Tuple[str, ...] = (
    "etsy", "etsystatic", "amazon", "ebay", "alibaba", "aliexpress",
    "flipkart", "indiamart", "redbubble", "zazzle", "teacherspayteachers",
    # Teaching-resource marketplaces. Educational in subject but commercial
    # in form: their images are product previews. A live run put a Twinkl
    # listing on a Photosynthesis slide - two poster thumbnails side by side
    # with an "ink saving" badge across one of them.
    "twinkl", "tes.com", "teachstarter", "edrawmax", "numerade",
    "coursehero", "studocu", "chegg", "quizlet", "brainly",
    "shutterstock", "istockphoto", "gettyimages", "dreamstime", "alamy",
    "123rf", "depositphotos", "sciencephoto", "adobestock", "vecteezy",
    "freepik", "canva", "slideshare", "slidesharecdn", "pinimg", "pinterest",
    # Video thumbnails. On topic by caption, but built to be clicked: overlaid
    # title text, an arrow, often a presenter's face. A Light slide was given
    # a ytimg "maxresdefault" frame, which is not a figure of anything.
    "ytimg", "youtube", "vimeocdn", "dailymotion", "tiktokcdn",
)

# Sources whose figures are usually drawn for teaching.
TRUSTED_HOSTS: Tuple[str, ...] = (
    ".edu", ".ac.", ".gov", "ncert", "nasa", "britannica", "khanacademy",
    "hyperphysics", "byjus", "toppr", "vedantu", "openstax", "libretexts",
    "wikimedia", "wikipedia", "nih.gov", "noaa", "bbc.co.uk/bitesize",
)

# Words that carry no subject meaning inside a chapter title.
_TOPIC_STOPWORDS = frozenset((
    "and", "the", "of", "in", "its", "for", "with", "a", "an", "to", "on",
    "introduction", "basics", "chapter", "class", "part", "unit",
))

_CLASS_RE = re.compile(r"(\d{1,2})")


def host_adjustment(url: str) -> Tuple[float, str]:
    """Score contribution from where the image lives, not what it is called."""
    u = str(url or "").lower()
    if not u:
        return 0.0, ""
    for host in UNSUITABLE_HOSTS:
        if host in u:
            return W_UNSUITABLE, "unsuitable:" + host
    for host in TRUSTED_HOSTS:
        if host in u:
            return W_TRUSTED, "trusted:" + host.strip(".")
    return 0.0, ""


def _class_number(class_name: str) -> Optional[int]:
    m = _CLASS_RE.search(str(class_name or ""))
    if not m:
        return None
    value = int(m.group(1))
    return value if 1 <= value <= 12 else None


@dataclass
class RelevanceSpec:
    """What the slide needs its picture to be, and not to be."""

    topic: str = ""
    required: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    visual_kind: str = ""
    class_name: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"topic": self.topic, "required": list(self.required),
                "avoid": list(self.avoid), "visualKind": self.visual_kind,
                "className": self.class_name}


def _norm(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for v in values or ():
        v = re.sub(r"\s+", " ", str(v or "")).strip().lower()
        if v and v not in out:
            out.append(v)
    return out


def conflicts_for(topic: str, required: Sequence[str] = ()) -> List[str]:
    """Competing concepts for a topic, minus anything the slide asked for.

    The subtraction is what keeps this a preference rather than a blocklist:
    a "Photosynthesis vs Respiration" comparison slide lists respiration in
    `required`, so it stops counting against the candidate.
    """
    haystack = " ".join(_norm([topic]) + _norm(required))
    wanted = set(_norm(required))
    out: List[str] = []
    for concept, competing in CONCEPT_CONFLICTS.items():
        if not matches_any(haystack, [concept]):
            continue
        for other in competing:
            if other in out:
                continue
            # Asked for explicitly? Then it belongs on the slide.
            if any(other in w or w in other for w in wanted):
                continue
            out.append(other)
    return out


def build_spec(topic: str = "", required: Sequence[str] = (),
               avoid: Sequence[str] = (), visual_kind: str = "",
               class_name: str = "") -> RelevanceSpec:
    """Assemble the scoring spec, combining declared and derived avoidance.

    `avoid` is whatever the planner said should not appear; the conflict
    table fills in the counterpart concepts the planner did not think to
    mention.
    """
    required_n = _norm(required)
    derived = conflicts_for(topic, required_n)
    return RelevanceSpec(
        topic=(topic or "").strip(),
        required=required_n,
        avoid=_norm(list(avoid) + derived),
        visual_kind=(visual_kind or "").strip().lower(),
        class_name=(class_name or "").strip(),
    )


@dataclass
class Verdict:
    score: float
    accepted: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        return confidence(self.score)

    @property
    def needs_vision(self) -> bool:
        return needs_vision(self.score)

    def __str__(self) -> str:
        return "%.2f %s%s" % (self.score, "accept" if self.accepted else "reject",
                              (" (" + "; ".join(self.reasons) + ")") if self.reasons else "")


_KIND_WORDS = {
    "diagram": ("diagram", "labelled", "labeled", "cross section", "schematic"),
    "illustration": ("illustration", "drawing", "infographic", "cartoon"),
    "chart": ("chart", "graph", "plot"),
    "photo": ("photo", "photograph", "image of"),
}


def score_candidate(text: str, spec: RelevanceSpec, url: str = "") -> Verdict:
    """Score one search result from its caption / snippet and its host.

    Positive for the topic, the required concepts, the right kind of figure
    and a teaching source; negative for competing concepts, out-of-band
    terminology, captions that mark the result as the wrong sort of object,
    and hosts that sell or licence pictures rather than publish them.
    """
    caption = re.sub(r"\s+", " ", str(text or "")).strip()
    if not caption:
        # No caption to judge. Neutral, not rejected: the existing source
        # ranking still has a view, and some good figures carry no title.
        return Verdict(0.0, False, ["no caption"])


    score = 0.0
    reasons: List[str] = []

    # Topic credit is proportional to how much of the topic the caption
    # actually names, for two reasons found the hard way:
    #   * lowercased, because matches_any lowercases the haystack but not the
    #     needle, so a capitalised "Photosynthesis" silently never matched;
    #   * by word rather than as a phrase, because a chapter called "Force
    #     and Pressure" never appears verbatim in a caption, so a perfectly
    #     good "Pressure diagram" scored nothing for its topic.
    topic_words = [w for w in re.split(r"\W+", (spec.topic or "").lower())
                   if len(w) > 2 and w not in _TOPIC_STOPWORDS]
    if topic_words:
        hit_topic = [w for w in topic_words if matches_any(caption, [w])]
        if hit_topic:
            score += W_TOPIC * (len(hit_topic) / len(topic_words))
            reasons.append("topic:%d/%d" % (len(hit_topic), len(topic_words)))

    hit_required = [r for r in spec.required if matches_any(caption, [r])]
    if hit_required:
        score += W_REQUIRED * len(hit_required)
        reasons.append("required:" + ",".join(hit_required[:3]))

    kind_words = _KIND_WORDS.get(spec.visual_kind)
    if kind_words and matches_any(caption, kind_words):
        score += W_VISUAL_KIND
        reasons.append("kind")

    hit_conflict = [a for a in spec.avoid if matches_any(caption, [a])]
    if hit_conflict:
        score += W_CONFLICT * len(hit_conflict)
        reasons.append("conflict:" + ",".join(hit_conflict[:3]))

    n = _class_number(spec.class_name)
    if n is not None and n <= 10:
        hit_advanced = [t for t in ADVANCED_TERMS
                        if re.search(r"\b" + t, caption, re.I)]
        if hit_advanced:
            score += W_ADVANCED * len(hit_advanced)
            reasons.append("advanced:" + ",".join(hit_advanced[:2]))

    hit_junk = [j for j in JUNK_TERMS if matches_any(caption, [j])]
    if hit_junk:
        score += W_JUNK * len(hit_junk)
        reasons.append("junk:" + ",".join(hit_junk[:2]))

    host_delta, host_reason = host_adjustment(url)
    if host_delta:
        score += host_delta
        reasons.append(host_reason)

    score = round(score, 2)
    accept, _ = thresholds()
    return Verdict(score, score >= accept, reasons)


def best_candidate(candidates: Sequence[Tuple[str, object]], spec: RelevanceSpec
                   ) -> Tuple[Optional[object], Optional[Verdict]]:
    """Pick the highest-scoring acceptable candidate, or nothing.

    "Or nothing" is the important half. Returning the least-bad option is how
    an unrelated picture ends up on a slide; the caller should fall through to
    the next visual strategy instead, because a clean diagram teaches more
    than a wrong photograph.
    """
    best = None
    best_verdict: Optional[Verdict] = None
    for text, payload in candidates:
        verdict = score_candidate(text, spec, str(payload or ""))
        if not verdict.accepted:
            continue
        if best_verdict is None or verdict.score > best_verdict.score:
            best, best_verdict = payload, verdict
    return best, best_verdict
