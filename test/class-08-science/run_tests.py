"""Phase 2 runner — executes the Class 8 Science suite against the AI service.

Captures every response verbatim into results/responses.json and attaches
MECHANICAL checks only (slide count, refusal behaviour, validation-keyword hits).
It deliberately does NOT decide factual correctness: that judgement is made by
reading the responses, and an expected answer is never rewritten to match output.

    python run_tests.py            # whole suite
    python run_tests.py SCI-08-05  # one chapter
"""
import glob
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
AI = "http://127.0.0.1:8000"
BE = "C:/Users/subha/Desktop/VVSPL/eddva/eddva_backend"
OUT = os.path.join(HERE, "results", "responses.json")
INSTITUTE = "c259cd4e-b018-45e2-8e46-52a497ca49a1"

REFUSAL = re.compile(
    r"\b(no such|does not exist|is not a real|not correct|incorrect|premise|"
    r"cannot confirm|not aware|no evidence|is false|not accurate|misconception|"
    r"there is no|not part of|outdated|invented|not a planet|dwarf planet)\b", re.I)


def env_of(path):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def passages_for(chapter_id):
    js = """
const fs=require('fs');const BE='%s';const e={};
for(const f of ['/.env','/.env.local']){try{for(const l of fs.readFileSync(BE+f,'utf8').split(/\\r?\\n/)){
const m=l.match(/^([A-Z0-9_]+)=(.*)$/);if(m)e[m[1]]=m[2].trim().replace(/^["']|["']$/g,'');}}catch{}}
const {Client}=require(BE+'/node_modules/pg');
(async()=>{const c=new Client({connectionString:e.SCHOOL_DB_URL,ssl:{rejectUnauthorized:false}});await c.connect();
const r=await c.query("SELECT content, page_no, tokens FROM textbook_chunks WHERE chapter_id::text=$1 ORDER BY chunk_index",["%s"]);
console.log(JSON.stringify(r.rows));await c.end();})().catch(x=>{console.log('[]');});
""" % (BE, chapter_id)
    res = subprocess.run(["node", "-e", js], capture_output=True, text=True, cwd=BE)
    try:
        rows = json.loads((res.stdout or "[]").strip().splitlines()[-1])
    except Exception:
        return []
    return [{"content": r.get("content") or "", "page_no": r.get("page_no"),
             "tokens": r.get("tokens"), "source": "ebook"} for r in rows if (r.get("content") or "").strip()]


# The first full run lost chapters 13-18 to "Together 429: too many requests in a
# short window" followed by network ConnectErrors: the runner fired calls
# back-to-back with no pacing. Those are harness faults, not model faults, so the
# runner now paces itself and retries a rate-limit or network error before giving up.
PACE_S = 3
RETRY_BACKOFF = (15, 45, 90)
TRANSIENT = re.compile(r"\b(429|too many requests|rate limit|ConnectError|timed out|"
                       r"5\d\d for model|network error)\b", re.I)


def post(path, payload, key, timeout=420):
    req_body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "X-API-Key": key,
               "X-Vertical": "school", "X-Board": "cbse", "X-Tenant-ID": INSTITUTE}
    t0 = time.time()
    last = None
    for attempt in range(len(RETRY_BACKOFF) + 1):
        req = urllib.request.Request(AI + path, data=req_body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode()), time.time() - t0, None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
        if attempt < len(RETRY_BACKOFF) and TRANSIENT.search(last or ""):
            wait = RETRY_BACKOFF[attempt]
            print(f"      transient ({last[:60]}...) - waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        break
    return None, time.time() - t0, last


def answer_text(body):
    """Flatten a doubt response to the text a student would read."""
    if not isinstance(body, dict):
        return ""
    parts = []
    for k in ("explanation", "answer"):
        v = body.get(k)
        if isinstance(v, str):
            parts.append(v)
    for block in ("brief", "detailed"):
        b = body.get(block)
        if isinstance(b, dict):
            for v in b.values():
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, list):
                    parts += [str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in v]
    return "\n".join(p for p in parts if p)


def keyword_hits(expected, actual):
    """Fraction of the expected answer's distinctive words that appear in the
    response. A weak signal used only to rank cases for human reading."""
    stop = {"the", "and", "of", "a", "an", "to", "in", "for", "on", "with", "is", "are",
            "it", "that", "this", "as", "by", "be", "or", "from", "so", "can", "which",
            "not", "but", "its", "their", "they", "when", "then", "than", "into", "at"}
    want = {w for w in re.findall(r"[a-z]{4,}", expected.lower()) if w not in stop}
    if not want:
        return 1.0
    have = set(re.findall(r"[a-z]{4,}", actual.lower()))
    return round(len(want & have) / len(want), 2)


def run_case(c, key, chapter_ids):
    rec = {"id": c["id"], "chapter": c["chapter"], "type": c["type"],
           "question": c["question"], "expected": c["expected_answer"],
           "validation": c["validation"], "feature": c.get("feature", "doubt_resolver")}
    if rec["feature"] == "none":
        rec.update(status="NOT EXECUTED", note="Not Applicable - recorded for coverage only")
        return rec

    if rec["feature"] == "ppt_generate":
        cid = chapter_ids.get(c["chapter"])
        ps = passages_for(cid) if cid else []
        body, secs, err = post("/ppt/generate", {
            "topic": c["chapter"], "slideCount": 2, "language": "English",
            "className": "Class 8", "subjectName": "Science",
            "chapterName": c["chapter"], "sourcePassages": ps}, key)
        rec["seconds"] = round(secs, 1)
        rec["passages_supplied"] = len(ps)
        if err:
            rec.update(status="FAIL", error=err, mechanical=["request failed"])
            return rec
        data = (body or {}).get("data") or {}
        slides = data.get("slides") or []
        rec["slides"] = slides
        rec["slide_count"] = len(slides)
        rec["grounded"] = ((data.get("source") or {}).get("grounded"))
        checks = []
        if len(slides) != 2:
            checks.append(f"slide count {len(slides)}, expected exactly 2")
        for i, s in enumerate(slides, 1):
            if not (s.get("title") or "").strip():
                checks.append(f"slide {i} blank title")
            bl = [b for b in (s.get("bullets") or []) if str(b).strip()]
            if i > 1 and len(bl) < 2:
                checks.append(f"slide {i} has {len(bl)} bullet(s)")
            for b in bl:
                if len(str(b).split()) < 5:
                    checks.append(f"slide {i} stub bullet {str(b)[:30]!r}")
        rec["mechanical"] = checks
        rec["status"] = "REVIEW" if checks else "REVIEW-CLEAN"
        return rec

    mode = "brief" if c["type"] == "Short Answer" else "detailed"
    # The app sends the student's class in studentContext; the first run did not,
    # which is exactly the information the answer's depth should depend on.
    body, secs, err = post("/doubt/resolve", {
        "questionText": c["question"], "subjectName": "Science", "mode": mode,
        "className": "Class 8",
        "studentContext": {"className": "Class 8", "subject": "Science", "board": "CBSE"}}, key)
    rec["seconds"] = round(secs, 1)
    if err:
        rec.update(status="FAIL", error=err, mechanical=["request failed"])
        return rec
    text = answer_text(body)
    rec["actual"] = text
    rec["model"] = (body or {}).get("model_used")
    rec["keyword_overlap"] = keyword_hits(c["expected_answer"], text)
    checks = []
    if not text.strip():
        checks.append("empty answer")
    if c["type"] == "Adversarial" and not REFUSAL.search(text):
        checks.append("adversarial: no refusal/correction language found")
    if rec["keyword_overlap"] < 0.25:
        checks.append(f"low overlap with expected answer ({rec['keyword_overlap']})")
    rec["mechanical"] = checks
    rec["status"] = "REVIEW" if checks else "REVIEW-CLEAN"
    return rec


def main():
    key = env_of(os.path.join(BE, ".env.local"))["AI_API_KEY"]
    chapter_ids = json.load(io.open(os.path.join(HERE, "chapter-ids.json"), encoding="utf-8"))
    cases = []
    for path in sorted(glob.glob(os.path.join(HERE, "cases", "*.jsonl"))):
        for line in io.open(path, encoding="utf-8"):
            if line.strip():
                cases.append(json.loads(line))
    args = [a for a in sys.argv[1:]]
    resume = "--resume" in args
    prefix = next((a for a in args if not a.startswith("--")), "")
    cases = [c for c in cases if c["id"].startswith(prefix)]

    # Previous results are kept and merged: a case that already produced a real
    # response is never re-run and never re-billed, so a resumed run only costs
    # the cases that failed.
    previous = {}
    if os.path.exists(OUT):
        for r in json.load(io.open(OUT, encoding="utf-8")):
            previous[r["id"]] = r
    todo = cases
    if resume:
        todo = [c for c in cases if previous.get(c["id"], {}).get("status") in (None, "FAIL")]
    print(f"{len(cases)} case(s) in scope, {len(todo)} to run"
          f"{' (resuming)' if resume else ''}", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for n, c in enumerate(todo, 1):
        rec = run_case(c, key, chapter_ids)
        previous[rec["id"]] = rec
        flag = ("  ! " + "; ".join(rec.get("mechanical", []))) if rec.get("mechanical") else ""
        print(f"[{n}/{len(todo)}] {rec['id']} {rec['type'][:18]:<18} "
              f"{rec.get('seconds', '-')}s {rec['status']}{flag}", flush=True)
        merged = sorted(previous.values(), key=lambda r: r["id"])
        io.open(OUT, "w", encoding="utf-8").write(json.dumps(merged, indent=1, ensure_ascii=False))
        if rec.get("feature") != "none" and n < len(todo):
            time.sleep(PACE_S)
    print(f"\nsaved {len(previous)} responses to {OUT}")


if __name__ == "__main__":
    main()
