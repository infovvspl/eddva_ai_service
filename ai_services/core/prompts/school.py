"""
School vertical (Classes 1-10) prompt layer.

DESIGN — derive, don't duplicate.
Each school system prompt is generated FROM its coaching/base counterpart at
import time by `schoolify()`:

  1. neutralise competitive-exam framing (JEE / NEET / IIT / AIIMS / UPSC ...)
  2. prepend a school-audience block (grade level, board, language, difficulty)

Why not hand-write a second copy of every prompt?
  * The base prompts embed the exact JSON output schema the views parse. A
    hand-written variant that drifts from that schema silently breaks response
    parsing. Deriving keeps the schema byte-identical.
  * A future edit to a base prompt is inherited by the school variant
    automatically — there is no second copy to forget to update.

Only prompts that genuinely differ are overridden (SCHOOL_OVERRIDE_FEATURES).
Everything else is shared with the base, per the "share by default, override by
exception" rule of the vertical layer.
"""

import re

# ── The school audience contract, prepended to every school system prompt ─────
SCHOOL_AUDIENCE = (
    "AUDIENCE — READ FIRST (overrides any conflicting instruction below):\n"
    "You are teaching SCHOOL students in Classes 1-10 following the CBSE / ICSE / "
    "State-board syllabus. They are NOT competitive-exam aspirants.\n"
    "- Pitch every explanation at the student's class level. Use simple, everyday "
    "language and short sentences; explain any term you introduce.\n"
    # Board-neutral on purpose: the specific board (CBSE/ICSE/State) and its
    # textbooks are supplied per-request by the BOARD CONTEXT block that
    # core/boards.py prepends. Naming NCERT here would mis-frame ICSE schools.
    "- Stay inside the prescribed textbook scope for the class and board. Do NOT "
    "pull in topics from higher classes.\n"
    # Deliberately does NOT name JEE/NEET here: naming them still plants the
    # token in a school prompt (and the test suite asserts school prompts are
    # free of competitive framing). Describe the pattern to avoid instead.
    "- Use board-exam style (1/2/3/5-mark answers, definitions, reasoning). NEVER "
    "use entrance/competitive-exam patterns: no trick questions, no multi-step "
    "derivations, no Olympiad-level difficulty, no advanced shortcuts.\n"
    "- Be warm and encouraging. These are young learners; build confidence, never "
    "intimidate.\n"
    "- Keep the required output format/JSON schema below EXACTLY as specified.\n"
    "\n"
    "----------------------------------------------------------------------\n"
)

# ── Phrase-level neutralisation (longest/most specific first) ──────────────────
_REPLACEMENTS = [
    ("expert academic evaluator for Indian competitive exam preparation (JEE, NEET, UPSC, etc.)",
     "expert academic evaluator for CBSE / ICSE / State-board school exams (Classes 1-10)"),
    ("students preparing for competitive exams", "school students in Classes 1-10"),
    ("Indian competitive exams", "the CBSE / ICSE / State-board school curriculum"),
    ("competitive exam preparation", "school board exam preparation"),
    ("competitive exams", "school board exams"),
    ("competitive exam", "school board exam"),
    ("JEE/NEET students", "school students in Classes 1-10"),
    ("JEE and NEET students", "school students in Classes 1-10"),
    ("JEE/NEET", "school board exam"),
    ("JEE and NEET", "the school board curriculum"),
    ("expert JEE and NEET teacher", "expert school teacher for Classes 1-10"),
    ("top colleges (IIT, AIIMS, NIT, etc.)", "reputed schools and Class 11 streams"),
    # Board-neutral: the board's own textbooks are named by the BOARD CONTEXT block.
    ("NCERT/JEE/NEET syllabus", "the prescribed board syllabus"),
    ("NCERT keywords", "prescribed-textbook keywords"),
    ("NCERT", "the prescribed textbook"),
    ("CBSE, JEE, and NEET curriculum", "CBSE / ICSE / State-board curriculum (Classes 1-10)"),
    ("CBSE/NEET", "CBSE / ICSE / State board"),
]

# Catch anything the phrase list missed (e.g. a lone "JEE" in a bullet).
_TOKEN_SWEEP = [
    (re.compile(r"\bJEE\s*(?:/|&|and|,)\s*NEET\b", re.I), "school board exam"),
    (re.compile(r"\bJEE\s+Advanced\b", re.I), "school board exam"),
    (re.compile(r"\bJEE\b", re.I), "school board exam"),
    (re.compile(r"\bNEET\b", re.I), "school board exam"),
    (re.compile(r"\bUPSC\b", re.I), "school board exam"),
    (re.compile(r"\bIIT\b", re.I), "a good Class 11 stream"),
    (re.compile(r"\bAIIMS\b", re.I), "a good Class 11 stream"),
    (re.compile(r"\bNIT\b", re.I), "a good Class 11 stream"),
]


def schoolify(system_prompt: str) -> str:
    """Turn a coaching/base system prompt into its school (Classes 1-10) variant.

    Only the *framing* is rewritten — the output format / JSON schema in the base
    prompt is left untouched, because the views parse against it.
    """
    out = system_prompt or ""
    for old, new in _REPLACEMENTS:
        out = out.replace(old, new)
    for pattern, new in _TOKEN_SWEEP:
        out = pattern.sub(new, out)
    return SCHOOL_AUDIENCE + out


# ── Which features get a school variant ───────────────────────────────────────
# Student-facing generation, where grade level and board framing change the output.
SCHOOL_OVERRIDE_FEATURES = (
    "tutor_session",
    "tutor_continue",
    "content_recommend",
    "stt_notes",
    "feedback_generate",
    "notes_analyze",
    "plan_generate",
    "syllabus_generate",
    "quiz_generate",
    "feedback_analyze",
    "content_suggest",
    "test_generate",
    "career_roadmap",
    "study_plan",
    "notes_generate",
    "evaluate_batch",
)

# Not meaningful for Classes 1-10 — gated off for the school vertical instead of
# being given a nonsensical school variant (see verticals.PROFILES).
SCHOOL_DISABLED_FEATURES = frozenset({
    "resume_analyze",   # 10-year-olds do not have résumés
    "interview_prep",   # college/job interview coaching
})
