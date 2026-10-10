"""Each planner and fact-check call carries only the textbook pages it needs.

A live 7-slide image deck sent about 48,000 input tokens in a minute: the same
~10,000-token chapter text went with the outline and again with every batch
of slides. Together rate-limited it, the planner timed out, and the deck fell
back to the old generator. These tests pin the fix:

  * a whole-chapter deck's outline still reads the chapter (unchanged);
  * a topic deck's outline reads that topic's pages, within the topic budget;
  * each batch, slide rewrite and fact-check batch reads the pages matching
    its own slides, within the slice budget;
  * a whole deck now sends well under half what it did;
  * at most PPT_PLANNER_MAX_PARALLEL calls are in flight at once, and a
    failed batch pauses before its one retry.

No network: the LLM is a stub that records every prompt.
"""
import re
import threading
import time
from unittest import mock

from django.test import SimpleTestCase

from ai_services.core.ppt_v2 import factcheck, planner
from ai_services.core.ppt_v2.planner import PlannerError, build_planner_prompts, plan_deck, source_block

# ~400-token passages across a long chapter: reflection first, then refraction
# and lenses, as in the CBSE "Light" chapter.
# "light" is on nearly every page, as it is in the real chapter: matching it
# must not pull refraction pages into a reflection deck.
SECTIONS = [
    ("reflection of light mirror incident ray reflected ray angle of incidence normal plane mirror image", 8),
    ("concave mirror convex mirror light pole focus centre of curvature principal axis", 8),
    ("refraction of light glass slab refractive index bending speed of light medium", 10),
    ("lens convex lens concave lens light optical centre power dioptre magnification", 10),
]


def chapter():
    out, page = [], 160
    for words, n in SECTIONS:
        for i in range(n):
            text = ("%s. " % words) * 20                  # ~1,600 chars, ~400 tokens
            out.append({"page_no": page, "chunk_index": i, "content": text, "source": "ebook"})
            page += 1
    return out


BOOK = chapter()                                          # 36 passages, ~14,500 tokens
tokens = lambda text: len(text) // 4                      # the grounding module's estimate  # noqa: E731


def book_part(user_prompt: str) -> str:
    return user_prompt.split("SOURCE MATERIAL", 1)[1] if "SOURCE MATERIAL" in user_prompt else ""


class Recorder:
    """Outlines `n` slides about mirrors; fills each batch as asked; records prompts."""

    def __init__(self, n=7):
        self.n, self.calls, self.lock = n, [], threading.Lock()
        self.in_flight = self.peak = 0

    def complete(self, **kw):
        with self.lock:
            self.calls.append(kw)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(0.02)                              # let parallel calls overlap
            if "You outline educational slide decks" in kw["system_prompt"]:
                titles = ["Plane mirror reflection", "Angle of incidence and reflection",
                          "Concave mirror focus", "Convex mirror uses", "Centre of curvature",
                          "Image in a plane mirror", "Summary of reflection"]
                return {"content": {"title": "Reflection", "slides": [
                    {"slideType": "concept", "title": titles[i % len(titles)], "focus": "mirror"}
                    for i in range(self.n)]}}
            n = int(re.search(r"Return exactly (\d+) slides", kw["user_prompt"]).group(1))
            return {"content": {"title": "", "slides": [
                {"slideType": "concept", "title": "Slide %d" % (i + 1),
                 "blocks": [{"kind": "bullet", "label": "x", "text": "A mirror reflects light."}],
                 "visual": {"kind": "none"}} for i in range(n)]}}
        finally:
            with self.lock:
                self.in_flight -= 1


class SourceBlockTests(SimpleTestCase):
    def test_a_chapter_deck_still_reads_the_chapter(self):
        body = source_block(BOOK, "Light", "Light")
        self.assertGreater(tokens(body), 9000)            # the full budget, as before
        self.assertLessEqual(tokens(body), planner._SOURCE_TOKEN_BUDGET + 200)

    def test_a_query_picks_the_matching_pages_within_its_budget(self):
        body = source_block(BOOK, "Light", "Light", query="concave mirror focus pole",
                            budget=planner.slice_budget())
        self.assertLessEqual(tokens(body), planner.slice_budget() + 200)
        self.assertIn("concave mirror", body)
        self.assertNotIn("refractive index", body)

    def test_budgets_can_be_tuned(self):
        with mock.patch.dict("os.environ", {"PPT_TOPIC_TOKEN_BUDGET": "2000",
                                            "PPT_SLICE_TOKEN_BUDGET": "1200",
                                            "PPT_PLANNER_MAX_PARALLEL": "2"}):
            self.assertEqual((planner.topic_budget(), planner.slice_budget(), planner.max_parallel()),
                             (2000, 1200, 2))
        with mock.patch.dict("os.environ", {"PPT_SLICE_TOKEN_BUDGET": "lots",
                                            "PPT_PLANNER_MAX_PARALLEL": "-"}):
            self.assertEqual(planner.slice_budget(), planner._SLICE_TOKEN_BUDGET)
            self.assertEqual(planner.max_parallel(), 3)


class PromptTests(SimpleTestCase):
    def test_a_topic_outline_reads_the_topics_pages_only(self):
        _s, user = build_planner_prompts(topic="Reflection of Light", slide_count=6,
                                         chapter="Light", passages=BOOK,
                                         focus_topic="Reflection of Light")
        part = book_part(user)
        self.assertLessEqual(tokens(part), planner.topic_budget() + 300)
        self.assertIn("plane mirror", part)
        self.assertNotIn("refractive index", part)
        self.assertNotIn("dioptre", part)

    def test_a_batch_reads_the_pages_its_slides_teach(self):
        outline = [{"title": "Concave mirror focus", "focus": "pole and focus of a concave mirror"},
                   {"title": "Convex mirror", "focus": "centre of curvature"},
                   {"title": "Refraction in a glass slab", "focus": "refractive index"}]
        _s, user = build_planner_prompts(topic="Light", slide_count=2, chapter="Light",
                                         passages=BOOK, part={"outline": outline, "start": 0,
                                                              "deck_size": 3})
        part = book_part(user)
        self.assertLessEqual(tokens(part), planner.slice_budget() + 300)
        self.assertIn("concave mirror", part)
        self.assertNotIn("refractive index", part)        # slide 3 is not this batch's

    def test_a_rewrite_reads_the_pages_of_the_slides_it_rewrites(self):
        llm = Recorder()
        llm.complete = mock.Mock(side_effect=RuntimeError("401 stop"))
        with self.assertRaises(PlannerError):
            planner.revise_slides(slides=[{"title": "Power of a lens", "blocks": [
                {"text": "Power is measured in dioptre."}]}], note="fix", topic="Light",
                chapter="Light", passages=BOOK, llm=llm)
        part = book_part(llm.complete.call_args.kwargs["user_prompt"])
        self.assertLessEqual(tokens(part), planner.slice_budget() + 300)
        self.assertIn("dioptre", part)


class WholeDeckTests(SimpleTestCase):
    def _plan(self, **kw):
        llm = Recorder(n=7)
        with mock.patch.object(planner, "parallel_enabled", return_value=True):
            result = plan_deck(topic="Reflection of Light", slide_count=7, chapter="Light",
                               passages=BOOK, llm=llm, **kw)
        return llm, result

    def test_a_topic_deck_sends_well_under_half_the_old_input(self):
        llm, result = self._plan(focus_topic="Reflection of Light", auto_range=(5, 10))
        self.assertEqual(len(result.slides), 7)
        sent = sum(tokens(c["system_prompt"]) + tokens(c["user_prompt"]) for c in llm.calls)
        # Before: ~10k of book in each of the four calls (outline + three batches).
        before = 4 * (planner._SOURCE_TOKEN_BUDGET + 2100)
        self.assertLess(sent, before * 0.5, "sent %d tokens" % sent)
        for call in llm.calls:
            self.assertLessEqual(tokens(book_part(call["user_prompt"])), planner.topic_budget() + 300)

    def test_no_more_than_the_cap_runs_at_once(self):
        llm = Recorder(n=16)
        with mock.patch.object(planner, "parallel_enabled", return_value=True), \
             mock.patch.dict("os.environ", {"PPT_PLANNER_MAX_PARALLEL": "2"}):
            plan_deck(topic="Light", slide_count=16, chapter="Light", passages=BOOK, llm=llm)
        self.assertLessEqual(llm.peak, 2)
        self.assertEqual(len(llm.calls), 1 + 6)            # the outline, then six batches

    def test_a_failed_batch_pauses_before_its_retry(self):
        llm = Recorder(n=6)
        real = llm.complete
        failed = {"n": 0}

        def flaky(**kw):
            if "writing slides" in kw["user_prompt"] and not failed["n"]:
                failed["n"] += 1
                raise RuntimeError("Together rate_limited (429)")
            return real(**kw)
        llm.complete = flaky
        with mock.patch.object(planner, "parallel_enabled", return_value=True), \
             mock.patch.object(planner.time, "sleep") as pause:
            result = plan_deck(topic="Light", slide_count=6, chapter="Light", passages=BOOK, llm=llm)
        self.assertEqual(len(result.slides), 6)
        pause.assert_any_call(planner.RETRY_PAUSE_S)


class FactCheckTests(SimpleTestCase):
    def test_each_batch_is_checked_against_its_own_pages(self):
        prompts = []

        class LLM:
            def complete(self, **kw):
                prompts.append(kw["user_prompt"])
                return {"content": {"issues": []}}
        slides = [{"title": "Concave mirror", "blocks": [{"text": "A concave mirror has a focus."}]},
                  {"title": "Lens power", "blocks": [{"text": "Power of a lens is in dioptre."}]}]
        factcheck.check_facts(slides, passages=BOOK, topic="Light", chapter="Light", llm=LLM())
        self.assertEqual(len(prompts), 1)                  # two slides, one batch of five
        part = prompts[0].split("STATEMENTS:")[0]
        self.assertLessEqual(tokens(part), factcheck.check_budget() + 300)
        self.assertIn("concave mirror", part)
        self.assertIn("dioptre", part)
        self.assertNotIn("glass slab", part)               # nothing it checks is about that


class CallTimeoutTests(SimpleTestCase):
    """A stalled call is cut at the planner's cap and retried while the deck has time.

    A live deck lost its V2 build to one Together request that hung until the
    120s capability timeout, 137s into the deck - past the old 90s retry window.
    """

    def test_the_router_takes_the_shorter_timeout_and_never_a_longer_one(self):
        from ai_services.tests_model_router import TOGETHER_ID, make_router, req
        env = {"TOGETHER_MODEL_GPT_OSS_120B": TOGETHER_ID,
               "TOGETHER_MODEL_GPT_OSS_120B_STRUCTURED_OUTPUT": "prompted"}
        with mock.patch("ai_services.core.provider_events.emit"):
            r, adapters = make_router(env)
            policy = r.plan(req(capability="content")).policy.timeout_s
            for asked, expected in ((60, 60), (None, policy), (10_000, policy)):
                r.execute(req(capability="content", timeout_s=asked))
                served = [c for a in adapters.values() for c in a.calls]
                self.assertEqual(served[-1][1].timeout_s, expected, asked)

    def test_planner_and_fact_check_calls_carry_the_cap(self):
        llm = Recorder(n=6)
        with mock.patch.object(planner, "parallel_enabled", return_value=True):
            plan_deck(topic="Light", slide_count=6, chapter="Light", passages=BOOK, llm=llm)
        self.assertTrue(llm.calls)
        self.assertEqual({c.get("timeout_s") for c in llm.calls}, {planner.call_timeout()})
        seen = []

        class LLM:
            def complete(self, **kw):
                seen.append(kw.get("timeout_s"))
                return {"content": {"issues": []}}
        factcheck.check_facts([{"title": "Mirror", "blocks": [{"text": "A mirror reflects."}]}],
                              passages=BOOK, topic="Light", chapter="Light", llm=LLM())
        self.assertEqual(seen, [planner.call_timeout()])

    def test_a_failed_batch_may_retry_later_in_the_deck_than_before(self):
        now = time.perf_counter()
        self.assertTrue(planner._batch_retry_ok("timed out", now - 110))     # past the old 90s
        self.assertFalse(planner._batch_retry_ok("timed out", now - 200))
        self.assertFalse(planner._batch_retry_ok("402 credit limit", now))   # never retried
        self.assertEqual(planner.call_timeout(), 60.0)
        with mock.patch.dict("os.environ", {"PPT_PLANNER_CALL_TIMEOUT_S": "45",
                                            "PPT_PLANNER_RETRY_UNTIL_S": "120"}):
            self.assertEqual((planner.call_timeout(), planner.batch_retry_until()), (45.0, 120.0))


class BackupModelTests(SimpleTestCase):
    """A call that stalls on the primary planner model is retried on the backup.

    Live, gpt-oss-120b stalled three batches at once for 60s while GLM-5.3-Flash
    (the "grounded" capability) answered in ~9s and wrote a valid V2 plan in 17s.
    """

    def _flaky(self, llm, fail_when):
        real = llm.complete
        failed = {"n": 0}

        def complete(**kw):
            if fail_when(kw) and not failed["n"]:
                failed["n"] += 1
                raise RuntimeError("Together request timed out after 60s")
            return real(**kw)
        llm.complete = complete

    def _plan(self, llm, **env):
        with mock.patch.object(planner, "parallel_enabled", return_value=True), \
             mock.patch.object(planner.time, "sleep"), mock.patch.dict("os.environ", env):
            return plan_deck(topic="Light", slide_count=6, chapter="Light", passages=BOOK, llm=llm)

    def test_a_stalled_batch_is_retried_on_the_backup_model(self):
        llm = Recorder(n=6)
        self._flaky(llm, lambda kw: "writing slides 1-" in kw["user_prompt"])
        result = self._plan(llm)
        self.assertEqual(len(result.slides), 6)
        retried = [c for c in llm.calls if "writing slides 1-" in c["user_prompt"]]
        self.assertEqual([c["capability"] for c in retried], ["grounded"])   # the 2nd attempt
        first_tries = [c for c in llm.calls if "writing slides 4-" in c["user_prompt"]]
        self.assertEqual({c["capability"] for c in first_tries}, {"content"})

    def test_a_stalled_outline_is_retried_on_the_backup_model(self):
        llm = Recorder(n=6)
        self._flaky(llm, lambda kw: "You outline educational slide decks" in kw["system_prompt"])
        result = self._plan(llm)
        self.assertEqual(len(result.slides), 6)
        outlines = [c for c in llm.calls if "You outline" in c["system_prompt"]]
        self.assertEqual([c["capability"] for c in outlines], ["grounded"])

    def test_the_backup_can_be_switched_off(self):
        llm = Recorder(n=6)
        self._flaky(llm, lambda kw: "writing slides 1-" in kw["user_prompt"])
        self._plan(llm, PPT_PLANNER_BACKUP_CAPABILITY="")
        retried = [c for c in llm.calls if "writing slides 1-" in c["user_prompt"]]
        self.assertEqual([c["capability"] for c in retried], ["content"])

    def test_the_fact_check_retry_uses_the_backup_too(self):
        seen = []

        class LLM:
            def complete(self, **kw):
                seen.append(kw["capability"])
                if len(seen) == 1:
                    raise RuntimeError("timed out")
                return {"content": {"issues": []}}
        with mock.patch.object(factcheck.time, "sleep"):
            factcheck.check_facts([{"title": "Mirror", "blocks": [{"text": "A mirror reflects."}]}],
                                  passages=BOOK, topic="Light", chapter="Light", llm=LLM())
        self.assertEqual(seen, ["content", "grounded"])
