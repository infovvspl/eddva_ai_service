"""Print test cases in the plain Test-Case-ID format.

    python render.py                 # every case
    python render.py SCI-08-05       # one chapter
    python render.py SCI-08-05-007   # one case

The JSONL files under cases/ are the source of truth; this only formats them.
"""
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "cases", "*.jsonl"))):
        for line in io.open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render(c):
    out = [
        f"Test Case ID: {c['id']}",
        "Class: 8",
        "Board: CBSE",
        "Subject: Science",
        f"Chapter: {c['chapter']}",
        f"Topic: {c['topic']}",
        "",
        "Question:",
        c["question"],
        "",
        "Question Type:",
        c["type"],
        "",
        "Expected Answer:",
        c["expected_answer"],
        "",
        "Expected Reasoning:",
        c["expected_reasoning"],
        "",
        f"Difficulty:\n{c['difficulty']}",
    ]
    if c.get("expected_slides"):
        out += ["", f"Expected Slide Count: {c['expected_slides']}"]
    out += ["", "Validation Criteria:"] + [f"- {v}" for v in c["validation"]]
    out += [f"\nFeature Under Test: {c.get('feature', 'doubt_resolver')}", "-" * 72]
    return "\n".join(out)


def main():
    rows = load()
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    shown = [c for c in rows if c["id"].startswith(prefix)]
    if not shown:
        sys.exit(f"No case matches {prefix!r} ({len(rows)} cases loaded)")
    for c in shown:
        print(render(c))
    print(f"{len(shown)} case(s)")


if __name__ == "__main__":
    main()
