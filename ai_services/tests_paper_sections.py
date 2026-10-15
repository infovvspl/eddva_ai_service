"""Assessment papers: objective and written sections written in parallel, then joined.

No network: the model is a fake. The plan strings are built the way the
backend builds them (school-assessment.service.ts, aiGenerateDraft).
"""
import threading
import time
from unittest import mock

from django.test import SimpleTestCase, TestCase

from ai_services.core import paper_sections as ps

A = "- Section A — Multiple Choice Questions: exactly 5 questions, each with four options labelled (a), (b), (c), (d) and exactly one correct option. 1 mark each."
B = "- Section B — True or False: exactly 5 statements. 1 mark each."
D = "- Section D — Short Answer: exactly 3 questions. 3 marks each."
E = "- Section E — Long Answer: exactly 2 questions. 5 marks each."
HEADER = 'Begin with a paper header (Subject, Class, Maximum Marks, Time Allowed) and a brief "General Instructions" list.'
DIFF = "Difficulty mix: the paper must contain exactly 5 easy, 5 intermediate and 5 hard questions in total (not per section)."
FIG = "\nFIGURES AVAILABLE — these diagrams come from the students' own textbook chapter:\n  F1 — Fig 5.9 Human respiratory system"
PLOT = "\nDRAWN DIAGRAMS: For a question whose figure is fully determined by the question's own data ... Use at most 4 drawn diagrams."


def extra(with_diff=True):
    return "\n".join(["Produce a COMPLETE school examination QUESTION PAPER.", DIFF if with_diff else "Difficulty: easy.",
                      HEADER, "Include ONLY these sections:", A, B, D, E, "Output ONLY the Markdown question paper."]) + FIG + PLOT


def plan(with_diff=True):
    return {
        "sections": [{"letter": "A", "kind": "objective", "count": 5, "text": A},
                     {"letter": "B", "kind": "objective", "count": 5, "text": B},
                     {"letter": "D", "kind": "written", "count": 3, "text": D},
                     {"letter": "E", "kind": "written", "count": 2, "text": E}],
        "header": HEADER,
        "difficulty": {"line": DIFF, "easy": 5, "medium": 5, "hard": 5} if with_diff else None,
        "figures": FIG, "plots": PLOT,
    }


class InstructionTests(SimpleTestCase):
    def setUp(self):
        self.gs = ps.groups(plan())

    def test_two_groups_objective_then_written(self):
        self.assertEqual([[s["letter"] for s in g] for g in self.gs], [["A", "B"], ["D", "E"]])

    def test_the_objective_call_writes_the_header_and_only_its_sections(self):
        text = ps.group_instructions(extra(), plan(), self.gs, 0)
        self.assertIn(A, text); self.assertIn(B, text)
        self.assertNotIn(D, text); self.assertNotIn(E, text)
        self.assertIn(HEADER, text)
        self.assertNotIn("FIGURES AVAILABLE", text)        # figures go to the written call
        self.assertNotIn("DRAWN DIAGRAMS", text)
        self.assertIn(ps.ONLY_THESE, text)

    def test_the_written_call_has_no_header_and_gets_the_figures(self):
        text = ps.group_instructions(extra(), plan(), self.gs, 1)
        self.assertIn(D, text); self.assertNotIn(A, text)
        self.assertNotIn(HEADER, text); self.assertIn(ps.NO_HEADER, text)
        self.assertIn("FIGURES AVAILABLE", text); self.assertIn("DRAWN DIAGRAMS", text)

    def test_the_difficulty_mix_is_split_and_adds_up(self):
        t0 = ps.group_instructions(extra(), plan(), self.gs, 0)
        t1 = ps.group_instructions(extra(), plan(), self.gs, 1)
        self.assertIn("exactly 4 easy, 3 intermediate and 3 hard", t0)   # 10 objective questions
        self.assertIn("exactly 1 easy, 2 intermediate and 2 hard", t1)   # 5 written questions
        self.assertNotIn(DIFF, t0 + t1)

    def test_split_counts_is_exact_for_every_group(self):
        for total, shares in (({"easy": 7, "medium": 7, "hard": 6}, [15, 5]),
                              ({"easy": 0, "medium": 3, "hard": 2}, [2, 3]),
                              ({"easy": 1, "medium": 1, "hard": 1}, [1, 2])):
            out = ps.split_counts(total, shares)
            self.assertEqual([sum(g.values()) for g in out], shares)
            for lv in total:
                self.assertEqual(sum(g[lv] for g in out), total[lv])

    def test_a_plan_that_does_not_match_gives_none(self):
        bad = plan(); bad["sections"][2]["text"] = "- Section D — something else"
        self.assertIsNone(ps.group_instructions(extra(), bad, ps.groups(bad), 0))


class MergeTests(SimpleTestCase):
    def test_papers_then_one_answer_key_in_order(self):
        out = ps.merge([
            "# Science\n## General Instructions\n1. All\n## Section A\nQ1. x\n## Answer Key\n### Section A\nQ1. Answer: a",
            "## Section D\nQ1. y\n## **Answer Key**\n### Section D\nQ1. Answer: because",
        ])
        self.assertEqual(out.count("Answer Key"), 1)
        self.assertLess(out.index("## Section A"), out.index("## Section D"))
        self.assertLess(out.index("## Section D"), out.index("## Answer Key"))
        self.assertLess(out.index("### Section A"), out.index("### Section D"))

    def test_a_part_without_a_key_still_joins(self):
        out = ps.merge(["## Section A\nQ1. x", "## Section D\nQ1. y\n## Answer Key\n### Section D\nQ1. Answer: z"])
        self.assertIn("## Section A", out)
        self.assertIn("### Section D", out)


class RunTests(SimpleTestCase):
    def _prompt(self):
        return "Topic: Respiration\nAdditional instructions: " + extra() + "\nWrite it.\n"

    def test_both_calls_run_at_the_same_time(self):
        active, peak, lock = [0], [0], threading.Lock()

        def complete(prompt):
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with lock:
                active[0] -= 1
            part = "A" if A in prompt else "D"
            return {"content": "## Section %s\nQ1. q\n## Answer Key\n### Section %s\nQ1. Answer: a" % (part, part),
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50}, "model": "m", "latency_ms": 200}

        out = ps.run(complete, user_prompt=self._prompt(), extra=extra(), plan=plan())
        self.assertEqual(peak[0], 2)
        self.assertEqual(out["parts"], 2)
        self.assertEqual(out["usage"], {"prompt_tokens": 200, "completion_tokens": 100})
        self.assertLess(out["content"].index("## Section A"), out["content"].index("## Section D"))

    def test_single_call_when_there_is_nothing_to_split_or_it_is_off(self):
        only_objective = plan(); only_objective["sections"] = only_objective["sections"][:2]
        self.assertIsNone(ps.run(lambda p: {}, user_prompt=self._prompt(), extra=extra(), plan=only_objective))
        self.assertIsNone(ps.run(lambda p: {}, user_prompt=self._prompt(), extra=extra(), plan=None))
        with mock.patch.dict("os.environ", {"ASSESSMENT_PARALLEL_SECTIONS": "false"}):
            self.assertIsNone(ps.run(lambda p: {}, user_prompt=self._prompt(), extra=extra(), plan=plan()))

    def test_single_call_when_a_part_fails_or_is_empty(self):
        def failing(prompt):
            if D in prompt:
                raise RuntimeError("provider down")
            return {"content": "x"}
        self.assertIsNone(ps.run(failing, user_prompt=self._prompt(), extra=extra(), plan=plan()))
        self.assertIsNone(ps.run(lambda p: {"content": "" if D in p else "x"},
                                 user_prompt=self._prompt(), extra=extra(), plan=plan()))


class _FakeLLM:
    """Grounded route; records every prompt; answers per group."""

    def __init__(self):
        self.prompts = []
        self.lock = threading.Lock()

    def can_route(self, **kw):
        return True

    def complete(self, system_prompt, user_prompt, **kw):
        with self.lock:
            self.prompts.append(user_prompt)
        part = "A" if A in user_prompt and D not in user_prompt else ("D" if D in user_prompt and A not in user_prompt else "ALL")
        return {"content": "## Section %s\nQ1. q\n## Answer Key\n### Section %s\nQ1. Answer: a" % (part, part),
                "model": "mock", "latency_ms": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class GenerateEndpointTests(TestCase):
    API_KEY = "apexiq-dev-secret-key-2026"

    def setUp(self):
        from ai_services.middleware import invalidate_institute_cache
        from ai_services.models import Institute
        invalidate_institute_cache()
        Institute.objects.create(name="T", slug="t", api_key=self.API_KEY, vertical="school", is_active=True)

    def _post(self, body, coverage=("Glycolysis in cytoplasm", "Fermentation", "Aerobic respiration")):
        llm = _FakeLLM()
        src = lambda passages, **kw: {"passages": passages, "digest_text": "", "mode": "topic_section",
                                      "reason": "test", "tokens_before": 1, "tokens_after": 1,
                                      "coverage": list(coverage) if coverage else None}
        with mock.patch("ai_services.views.bridge.get_llm", return_value=llm), \
                mock.patch("ai_services.core.assessment_source.select", side_effect=src):
            resp = self.client.post("/content/generate", data=body, content_type="application/json",
                                    HTTP_X_API_KEY=self.API_KEY, HTTP_X_VERTICAL="school")
        return resp, llm

    def _body(self, **kw):
        body = {"topicName": "Respiration", "subjectName": "Science", "chapterName": "Life Processes",
                "contentType": "assessment_paper", "extraContext": extra(), "paperPlan": plan(),
                "sourcePassages": [{"page_no": 9, "content": "Glucose is broken into pyruvate.", "tokens": 8}]}
        body.update(kw)
        return body

    def test_a_paper_is_written_by_two_calls_and_joined(self):
        resp, llm = self._post(self._body())
        self.assertEqual(resp.status_code, 200, resp.content[:300])
        self.assertEqual(len(llm.prompts), 2)
        content = resp.json()["content"]
        self.assertLess(content.index("## Section A"), content.index("## Section D"))
        self.assertEqual(content.count("Answer Key"), 1)

    def test_without_a_plan_it_is_one_call_as_before(self):
        body = self._body(); body.pop("paperPlan")
        resp, llm = self._post(body)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("Section ALL", resp.json()["content"])

    def test_coverage_comes_from_the_book_with_no_extra_call(self):
        resp, llm = self._post(self._body())
        self.assertFalse(any("List the distinct sub-areas" in p for p in llm.prompts))
        self.assertTrue(all("MANDATORY COVERAGE" in p and "Fermentation" in p for p in llm.prompts))

    def test_without_book_coverage_the_sub_area_call_still_runs(self):
        resp, llm = self._post(self._body(), coverage=None)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(any("List the distinct sub-areas" in p for p in llm.prompts))


class DivisionTests(SimpleTestCase):
    def test_coverage_items_each_go_to_one_group_by_share(self):
        out = ps.divide(["a", "b", "c", "d", "e", "f"], [10, 5])
        self.assertEqual(sorted(out[0] + out[1]), ["a", "b", "c", "d", "e", "f"])
        self.assertEqual((len(out[0]), len(out[1])), (4, 2))

    def test_pages_are_split_into_contiguous_runs_by_share(self):
        self.assertEqual(ps.divide_pages([9, 10, 11, 12], [15, 5]), [[9, 10, 11], [12]])
        self.assertEqual(ps.divide_pages([9, 12], [15, 5]), [[9], [12]])
        self.assertEqual(ps.divide_pages([9], [15, 5]), [])        # too few pages to divide

    def test_each_call_is_told_its_pages_and_its_share_of_coverage(self):
        items = ["glycolysis", "fermentation", "lactic acid", "breathing"]
        prompt = "Topic\n" + ps.coverage_block(items) + "Additional instructions: " + extra() + "\n"
        seen = []
        ps.run(lambda p: seen.append(p) or {"content": "## Section A\nQ1\n## Answer Key\nQ1. Answer: a"},
               user_prompt=prompt, extra=extra(), plan=plan(), coverage=items, pages=[9, 10, 11, 12])
        self.assertEqual(len(seen), 2)
        obj, written = (seen[0], seen[1]) if A in seen[0] else (seen[1], seen[0])
        self.assertIn("pp.9-11", obj); self.assertIn("p.12", written)
        self.assertNotIn(ps.coverage_block(items), obj + written)   # each got its own share
        self.assertIn("MANDATORY COVERAGE FOR YOUR SECTIONS", obj)


class BigPaperTests(SimpleTestCase):
    def _plan(self, counts):
        return {"sections": [{"letter": l, "kind": "objective" if l in "ABC" else "written",
                              "count": n, "text": "- Section %s" % l} for l, n in zip("ABCDE", counts)]}

    def test_a_normal_paper_stays_objective_and_written(self):
        self.assertEqual([[s["letter"] for s in g] for g in ps.groups(self._plan((5, 5, 5, 3, 2)))],
                         [["A", "B", "C"], ["D", "E"]])

    def test_a_100_mark_paper_stays_two_calls_by_default(self):
        self.assertEqual(len(ps.groups(self._plan((20, 10, 10, 10, 6)))), 2)

    def test_more_calls_when_configured_are_balanced_and_in_order(self):
        with mock.patch.dict("os.environ", {"ASSESSMENT_PARALLEL_MAX_CALLS": "3"}):
            gs = ps.groups(self._plan((20, 10, 10, 10, 6)))
        self.assertEqual([[s["letter"] for s in g] for g in gs], [["A", "B"], ["C", "D"], ["E"]])

    def test_the_number_of_calls_can_be_capped(self):
        with mock.patch.dict("os.environ", {"ASSESSMENT_PARALLEL_MAX_CALLS": "2"}):
            self.assertEqual(len(ps.groups(self._plan((20, 10, 10, 10, 6)))), 2)
