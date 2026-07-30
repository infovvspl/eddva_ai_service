"""
Education-board registry — the second axis of personalisation, after `vertical`.

`vertical` says WHO the learner is (school Classes 1-12 vs coaching JEE/NEET).
`board`    says WHICH SYLLABUS they follow (CBSE, ICSE/CISCE, State).

Why this exists: the school prompts used to hardcode "NCERT" and
"Class 10 Board Exam (CBSE)". NCERT is CBSE's textbook body — an ICSE school
follows CISCE and different prescribed books, so CBSE-flavoured framing produced
wrong chapter references and the wrong paper pattern for ICSE clients.

Resolution precedence (mirrors verticals, handled in TenantAuthMiddleware):
    X-Board header  >  Institute.board  >  DEFAULT_BOARD env  >  "cbse"

Adding a board later = one BoardProfile entry here. No engine changes.

NOTE ON ACCURACY: the CBSE profile is well established. The ICSE/State wording is
a reasonable FIRST PASS written from the public exam pattern — it is deliberately
about *framing* (which syllabus body, which textbooks, which paper style) rather
than specific chapter claims, so it cannot invent syllabus detail. Have a subject
expert refine `guidance` before leaning on it heavily.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_BOARD = "cbse"


@dataclass(frozen=True)
class BoardProfile:
    """Immutable description of one education board."""

    key: str
    display_name: str
    authority: str      # the examining/─governing body
    textbooks: str      # what "the textbook" means for this board
    guidance: str       # prompt block injected into school-vertical prompts

    def instruction(self) -> str:
        """The block prepended to a school system prompt for this board."""
        return (
            f"BOARD CONTEXT — {self.display_name} ({self.authority}):\n"
            f"{self.guidance}\n"
            "Follow this board's syllabus, terminology and paper pattern. Do NOT "
            "reference textbooks, chapter names or exam patterns from another board.\n"
            "----------------------------------------------------------------------\n"
        )


BOARDS: Dict[str, BoardProfile] = {
    "cbse": BoardProfile(
        key="cbse",
        display_name="CBSE",
        authority="Central Board of Secondary Education",
        textbooks="NCERT textbooks",
        guidance=(
            "- Align strictly to the NCERT textbook for the class: use NCERT definitions, "
            "examples and chapter terminology.\n"
            "- Use the CBSE paper pattern: objective/MCQ, assertion-reason, very short (1 mark), "
            "short (2-3 marks) and long (5 marks) answers, plus competency-based questions.\n"
            "- Keep to CBSE subject naming (e.g. 'Science' as one subject up to Class 10)."
        ),
    ),
    "icse": BoardProfile(
        key="icse",
        display_name="ICSE",
        authority="CISCE — Council for the Indian School Certificate Examinations",
        textbooks="CISCE-prescribed textbooks (e.g. Selina, Frank, Concise)",
        guidance=(
            # Deliberately does NOT name the other board's textbook body: naming it
            # still plants the term in the prompt (the test suite asserts an ICSE
            # prompt is free of it). Describe what to follow, not what to avoid.
            "- Follow the CISCE/ICSE syllabus and its prescribed textbooks only. Do not cite "
            "another board's textbooks or their chapter numbering.\n"
            "- Use the ICSE paper pattern: Section A (compulsory short-answer questions) and "
            "Section B (questions with internal choice), with answers that are more detailed "
            "and descriptive than CBSE equivalents.\n"
            "- ICSE treats Physics, Chemistry and Biology as separate subjects earlier than "
            "CBSE, and places heavier weight on English language and literature. Reflect that "
            "in subject naming, depth and answer style."
        ),
    ),
    "ib": BoardProfile(
        key="ib",
        display_name="IB",
        authority="the International Baccalaureate Organization",
        textbooks="IB-approved resources for the programme",
        guidance=(
            "- Follow the IB programme framework (PYP / MYP / DP as appropriate for the "
            "year group), not an Indian national board syllabus.\n"
            "- Use IB conventions: inquiry-led explanations, conceptual understanding and "
            "real-world application over rote recall; criterion-referenced assessment "
            "language rather than Indian board mark schemes.\n"
            "- Use international spelling and units, and avoid India-specific exam framing."
        ),
    ),
    "state": BoardProfile(
        key="state",
        display_name="State Board",
        authority="the relevant State Board of Secondary Education",
        textbooks="the State Board prescribed textbooks",
        guidance=(
            "- Follow the State Board syllabus and its prescribed textbooks for the class.\n"
            "- Use the State Board paper pattern and keep the depth at state-syllabus level.\n"
            "- Do not assume another board's textbooks; when the exact state is unknown, stay "
            "with a general, syllabus-neutral treatment rather than citing a specific book."
        ),
    ),
}

# Aliases seen in real tenant data (the backend stores whatever the school typed at
# registration), normalised onto the canonical keys above.
_ALIASES = {
    "cbse": "cbse", "central board of secondary education": "cbse", "ncert": "cbse",
    "icse": "icse", "cisce": "icse", "isc": "icse",
    "state": "state", "state board": "state", "stateboard": "state",
    "ib": "ib", "international baccalaureate": "ib", "ib board": "ib",
    "pyp": "ib", "myp": "ib", "ib dp": "ib",
}


def env_default_board() -> str:
    """Deployment-level default board (DEFAULT_BOARD env), validated."""
    return normalize_board(os.getenv("DEFAULT_BOARD"))


def normalize_board(value: Optional[str]) -> str:
    """
    Coerce any incoming value into a known board key.
    Falls back to DEFAULT_BOARD for empty/unknown values.
    """
    if not value:
        return DEFAULT_BOARD
    key = str(value).strip().lower()
    if key in BOARDS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]

    # Look for matching keywords of known central boards
    if any(k in key for k in ["cbse", "central board", "ncert"]):
        return "cbse"
    if any(k in key for k in ["icse", "cisce", "isc"]):
        return "icse"
    if any(k in key for k in ["ib", "baccalaureate", "pyp", "myp"]):
        return "ib"

    # Any other custom board is treated as a State Board
    return "state"


def get_board(board: Optional[str]) -> BoardProfile:
    """Return the BoardProfile, normalized and guaranteed valid."""
    return BOARDS[normalize_board(board)]


def board_instruction(board: Optional[str]) -> str:
    """Prompt block for a board. Safe for any input."""
    profile = get_board(board)
    raw_board = (board or "").strip()

    # If the resolved board profile is "state" and we have the raw board name,
    # build a highly specific board context using the state name!
    if profile.key == "state" and raw_board:
        raw_lower = raw_board.lower()
        if "board" in raw_lower or "state" in raw_lower:
            display_name = raw_board
        else:
            display_name = f"{raw_board} State Board"

        guidance = (
            f"- Follow the {display_name} syllabus and its prescribed textbooks for the class.\n"
            f"- Use the {display_name} paper pattern and keep the depth at that syllabus level.\n"
            f"- Do not assume another board's textbooks; stay with this specific {display_name} curriculum treatment."
        )

        return (
            f"BOARD CONTEXT — {display_name}:\n"
            f"{guidance}\n"
            f"Follow this board's syllabus, terminology and paper pattern. Do NOT "
            "reference textbooks, chapter names or exam patterns from another board.\n"
            "----------------------------------------------------------------------\n"
        )

    return profile.instruction()
