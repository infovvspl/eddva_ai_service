"""Re-run the false-premise cases against the fixed prompt and compare with the
pre-fix answers already captured in results/responses.json.

Writes results/after-fix.json. The original responses are never overwritten, so
the before/after comparison stays reproducible.
"""
import importlib.util
import io
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("runner", os.path.join(HERE, "run_tests.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

# Phrases that show the model challenged the premise rather than playing along.
CORRECTED = re.compile(
    r"no such|does not exist|is not a real|not a recognised|no recognised|there is no|"
    r"incorrect|not correct|false premise|premise .{0,20}(wrong|false|incorrect)|"
    r"cannot confirm|not aware of|no evidence|not an element|not a stage|"
    r"actually the (reverse|opposite)|in fact .{0,30}(lower|smaller|less|greater)|"
    r"is the opposite|not fixed by|no official|not mentioned in|not part of the",
    re.I)
# The specific fabrications seen before the fix.
FABRICATION = {
    "SCI-08-01-009": re.compile(r"urea\b.{0,80}(mandat|compulsor)", re.I | re.S),
    "SCI-08-05-009": re.compile(r"IS\s?1445|14(\.0)?\s?MJ", re.I),
    "SCI-08-06-009": re.compile(r"declared.{0,40}(biosphere|reserve)|in\s*\*?\*?20\d\d\*?\*?,?\s*the area", re.I),
    "SCI-08-08-009": re.compile(r"pre.zygotic budding\s*(is|occurs|refers|–|-)\s", re.I),
    "SCI-08-10-009": re.compile(r"bharat unit|B\.?P\.?U", re.I),
    "SCI-08-12-009": re.compile(r"(updated|revised).{0,40}45,?000|45\s?kHz\b", re.I),
    "SCI-08-14-009": re.compile(r"kutch intensity index", re.I),
    "SCI-08-17-009": re.compile(r"\b30\s?%|\bthirty per ?cent", re.I),
    "SCI-08-11-009": re.compile(r"avoid(ed|ing)? ball bearings|higher rolling resistance", re.I),
    "SCI-08-13-009": re.compile(r"object.{0,40}(made the|is the)\s*\*?\*?anode", re.I),
}

key = R.env_of(os.path.join(R.BE, ".env.local"))["AI_API_KEY"]
cases = []
for path in sorted(__import__("glob").glob(os.path.join(HERE, "cases", "*.jsonl"))):
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            if c["type"] == "Adversarial":
                cases.append(c)

before = {r["id"]: r for r in json.load(io.open(os.path.join(HERE, "results", "responses.json"),
                                               encoding="utf-8"))}
out, fixed, still = [], 0, []
print(f"re-running {len(cases)} false-premise cases against the fixed prompt\n")
for i, c in enumerate(cases, 1):
    rec = R.run_case(c, key, {})
    text = rec.get("actual") or ""
    corrected = bool(CORRECTED.search(text))
    fab = FABRICATION.get(c["id"])
    repeats = bool(fab.search(text)) if fab else None
    was_bad = c["id"] in FABRICATION
    verdict = ("FIXED" if was_bad and corrected and not repeats else
               "STILL FABRICATING" if was_bad and not corrected else
               "PASS" if corrected else "CHECK")
    if verdict == "FIXED":
        fixed += 1
    if verdict == "STILL FABRICATING":
        still.append(c["id"])
    print(f"[{i}/{len(cases)}] {c['id']} {verdict:<18} corrected={corrected} "
          f"repeats_old_claim={repeats}")
    out.append({"id": c["id"], "question": c["question"], "verdict": verdict,
                "corrected": corrected, "repeats_old_claim": repeats,
                "after": text[:1500], "before": (before.get(c['id'], {}).get("actual") or "")[:600]})
    io.open(os.path.join(HERE, "results", "after-fix.json"), "w", encoding="utf-8").write(
        json.dumps(out, indent=1, ensure_ascii=False))
    time.sleep(R.PACE_S)

print(f"\npreviously fabricating: {len(FABRICATION)}   now fixed: {fixed}   still failing: {still or 'none'}")
