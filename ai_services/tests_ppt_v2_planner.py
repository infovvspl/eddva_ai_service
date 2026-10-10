"""The planner must never hand downstream a deck of the wrong shape.

Two failure modes matter here and they are not the same.

A planner that *crashes* costs the teacher their deck, so every malformed
thing a model can emit - a string where an object belongs, a missing array, a
fenced code block, nothing at all - has to come back as a controlled
PlannerError with a reason code, not a traceback.

A planner that *silently complies* is worse. Asked for 2 slides and returning
3 is the exact bug that made "exactly 2 slides" untestable on the V1 path, so
the count is enforced rather than hoped for: trim when that is safe, retry
once when it is not, and raise rather than invent a slide nobody wrote.

No network: the LLM is a stub in every test.
"""
import json

from django.test import SimpleTestCase

from ai_services.core.ppt_v2.planner import (
    PLANNER_MAX_TOKENS,
    PLANNER_MIN_TOKENS,
    PLANNER_SLIDE_TYPES,
    PlannerError,
    build_planner_prompts,
    enforce_slide_count,
    plan_deck,
    planner_output_budget,
    validate_payload,
)


def slide(title="A slide", **kw):
    base = {"slideType": "concept", "title": title,
            "blocks": [{"kind": "bullet", "text": "A fact about the topic."}],
            "visual": {"kind": "illustration", "description": "a leaf"}}
    base.update(kw)
    return base


def payload(n=2, title="Deck"):
    return {"title": title, "slides": [slide("Slide %d" % (i + 1)) for i in range(n)]}


class _StubLLM:
    """Returns each scripted response in turn and records what it was asked."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0) if self.responses else {}
        if isinstance(item, BaseException):
            raise item
        content = item if not isinstance(item, dict) or "content" not in item else item["content"]
        return {"content": content, "model": "stub-model",
                "tokens_input": 10, "tokens_output": 20, "latency_ms": 5}


class PromptTests(SimpleTestCase):
    def test_the_requested_count_is_stated_in_both_prompts(self):
        system, user = build_planner_prompts("Photosynthesis", 2)
        self.assertIn("EXACTLY 2 slide objects", system)
        self.assertIn("exactly 2 slides", user)

    def test_the_schema_lists_the_allowed_slide_types(self):
        system, _ = build_planner_prompts("T", 3)
        for slide_type in PLANNER_SLIDE_TYPES:
            self.assertIn(slide_type, system)

    def test_the_planner_is_told_not_to_choose_layout_or_colour(self):
        # Keeping these out of the prompt is what keeps the composer testable.
        system, _ = build_planner_prompts("T", 3)
        self.assertIn("You do not choose layouts, colours", system)

    def test_curriculum_scope_reaches_the_prompt(self):
        _, user = build_planner_prompts("Photosynthesis", 2, class_name="Class 8",
                                        subject="Science", chapter="Crop Production",
                                        board="CBSE")
        for value in ("Class 8", "Science", "Crop Production", "CBSE"):
            self.assertIn(value, user)

    def test_the_class_level_is_stated_prominently_not_just_in_the_scope_line(self):
        # The first live run produced Calvin-cycle and thylakoid content for a
        # Class 8 request: the class was present, but buried in a scope line
        # among four other fields, and the model wrote past the syllabus.
        _, user = build_planner_prompts("Photosynthesis", 2, class_name="Class 8",
                                        subject="Science", board="CBSE")
        self.assertIn("AUDIENCE: Class 8 students", user)
        self.assertIn("Do not use vocabulary from a higher class", user)
        self.assertLess(user.index("AUDIENCE:"), user.index("Curriculum scope:"))

    def test_the_syllabus_rule_names_the_terms_that_went_wrong(self):
        system, _ = build_planner_prompts("Photosynthesis", 2, class_name="Class 8")
        self.assertIn("Calvin cycle", system)
        self.assertIn("STAY INSIDE THE SYLLABUS", system)

    def test_no_audience_line_when_no_class_was_given(self):
        _, user = build_planner_prompts("Photosynthesis", 2)
        self.assertNotIn("AUDIENCE:", user)

    def test_board_rules_lead_the_system_prompt(self):
        system, _ = build_planner_prompts("T", 2, board_rules="FOLLOW NCERT.")
        self.assertTrue(system.startswith("FOLLOW NCERT."))

    def test_grounded_passages_are_included_and_fenced_off(self):
        _, user = build_planner_prompts("T", 2, passages=["Leaves contain chlorophyll."])
        self.assertIn("SOURCE MATERIAL", user)
        self.assertIn("Leaves contain chlorophyll.", user)
        self.assertIn("do not supplement them from general knowledge", user)

    def test_passages_are_truncated_rather_than_sent_whole(self):
        # A whole chapter costs latency and money for no accuracy gain.
        from ai_services.core.ppt_v2.planner import _MAX_SOURCE_CHARS
        _, user = build_planner_prompts("T", 2, passages=["x" * 100000])
        self.assertLess(len(user), _MAX_SOURCE_CHARS + 10000)
        self.assertIn("x" * 1000, user)

    def test_passages_given_as_objects_are_accepted(self):
        _, user = build_planner_prompts("T", 2, passages=[{"text": "From the book."}])
        self.assertIn("From the book.", user)

    def test_no_passages_means_no_source_section(self):
        _, user = build_planner_prompts("T", 2)
        self.assertNotIn("SOURCE MATERIAL", user)


class ValidationTests(SimpleTestCase):
    def test_a_good_payload_passes(self):
        title, slides, problems = validate_payload(payload(2), 2)
        self.assertEqual((title, len(slides), problems), ("Deck", 2, []))

    def test_a_non_object_is_rejected_with_a_reason(self):
        _, slides, problems = validate_payload(["not", "a", "deck"], 2)
        self.assertEqual(slides, [])
        self.assertIn("not an object", problems[0])

    def test_a_missing_slides_array_is_rejected(self):
        _, slides, problems = validate_payload({"title": "x"}, 2)
        self.assertEqual(slides, [])
        self.assertIn("no 'slides' array", problems[0])

    def test_a_titleless_slide_is_dropped_but_the_deck_survives(self):
        data = payload(2)
        data["slides"][0]["title"] = "   "
        _, slides, problems = validate_payload(data, 2)
        self.assertEqual(len(slides), 1)
        self.assertIn("slide 1 has no title", problems)

    def test_a_deck_of_only_bad_slides_is_rejected(self):
        _, slides, problems = validate_payload({"slides": [1, 2, 3]}, 2)
        self.assertEqual(slides, [])
        self.assertIn("no usable slides", problems[-1])

    def test_a_missing_deck_title_falls_back_to_none(self):
        data = payload(2)
        data.pop("title")
        title, _, _ = validate_payload(data, 2)
        self.assertIsNone(title)


class SlideCountTests(SimpleTestCase):
    def test_the_exact_count_passes_through(self):
        slides, action = enforce_slide_count([slide(), slide()], 2)
        self.assertEqual((len(slides), action), (2, "exact"))

    def test_too_many_slides_are_trimmed(self):
        slides, action = enforce_slide_count([slide("a"), slide("b"), slide("c")], 2)
        self.assertEqual(action, "trimmed")
        self.assertEqual([s["title"] for s in slides], ["a", "b"])

    def test_too_few_slides_are_reported_not_padded(self):
        # Padding would mean inventing content nobody wrote.
        slides, action = enforce_slide_count([slide()], 2)
        self.assertEqual((len(slides), action), (1, "short"))


class PlanDeckTests(SimpleTestCase):
    def test_a_clean_plan_comes_back_validated(self):
        llm = _StubLLM(payload(2))
        result = plan_deck(topic="Photosynthesis", slide_count=2, llm=llm)
        self.assertEqual(len(result.slides), 2)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.model, "stub-model")
        self.assertEqual(result.title, "Deck")

    def test_json_arriving_as_a_string_is_parsed(self):
        llm = _StubLLM(json.dumps(payload(2)))
        self.assertEqual(len(plan_deck(topic="T", slide_count=2, llm=llm).slides), 2)

    def test_malformed_json_is_a_controlled_error(self):
        # Broken twice: the one retry is spent, then a controlled error.
        llm = _StubLLM("{not json at all", "{still not json")
        with self.assertRaises(PlannerError) as ctx:
            plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(ctx.exception.reason, "malformed_json")

    def test_an_llm_exception_is_a_controlled_error(self):
        llm = _StubLLM(RuntimeError("provider is down"), RuntimeError("still down"))
        with self.assertRaises(PlannerError) as ctx:
            plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(ctx.exception.reason, "llm_error")

    def test_an_unusable_shape_is_a_controlled_error(self):
        llm = _StubLLM({"slides": "two of them"})
        with self.assertRaises(PlannerError) as ctx:
            plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(ctx.exception.reason, "invalid_schema")

    def test_too_many_slides_are_trimmed_without_a_second_call(self):
        llm = _StubLLM(payload(5))
        result = plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(len(result.slides), 2)
        self.assertEqual(len(llm.calls), 1)

    def test_too_few_slides_triggers_exactly_one_retry(self):
        llm = _StubLLM(payload(1), payload(2))
        result = plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(len(result.slides), 2)
        self.assertEqual(result.attempts, 2)
        self.assertIn("EXACTLY 2 slide objects", llm.calls[1]["user_prompt"])

    def test_a_persistently_short_deck_is_an_error_not_a_short_deck(self):
        llm = _StubLLM(payload(1), payload(1))
        with self.assertRaises(PlannerError) as ctx:
            plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(ctx.exception.reason, "slide_count")
        self.assertEqual(len(llm.calls), 2)

    def test_retries_are_bounded_at_two_calls(self):
        llm = _StubLLM(*[payload(1)] * 6)
        with self.assertRaises(PlannerError):
            plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(len(llm.calls), 2)

    def test_a_broken_reply_is_retried_once_instead_of_losing_the_deck(self):
        # A 10-slide reply cut off mid-JSON used to end the V2 deck at once,
        # and the teacher generated again. One retry is cheaper than that.
        llm = _StubLLM("{bad", payload(2))
        result = plan_deck(topic="T", slide_count=2, llm=llm)
        self.assertEqual(len(result.slides), 2)
        self.assertEqual(result.attempts, 2)
        self.assertIn("not valid JSON", llm.calls[1]["user_prompt"])

    def test_a_provider_hiccup_is_retried_once(self):
        llm = _StubLLM(RuntimeError("Together server_error"), payload(2))
        self.assertEqual(len(plan_deck(topic="T", slide_count=2, llm=llm).slides), 2)

    def test_an_error_that_will_repeat_is_not_retried(self):
        for message in ("HTTP 402 credit limit", "401 invalid api key"):
            llm = _StubLLM(RuntimeError(message), payload(2))
            with self.assertRaises(PlannerError):
                plan_deck(topic="T", slide_count=2, llm=llm)
            self.assertEqual(len(llm.calls), 1, message)

    def test_a_revision_note_reaches_the_model(self):
        llm = _StubLLM(payload(2))
        plan_deck(topic="T", slide_count=2, llm=llm, revision_note="Remove: krebs.")
        self.assertIn("Remove: krebs.", llm.calls[0]["user_prompt"])

    def test_it_asks_the_router_for_the_content_capability(self):
        llm = _StubLLM(payload(2))
        plan_deck(topic="T", slide_count=2, llm=llm, institute_id="inst-1")
        call = llm.calls[0]
        self.assertEqual(call["capability"], "content")
        self.assertEqual(call["feature"], "ppt_v2_plan")
        self.assertTrue(call["json_mode"])
        self.assertEqual(call["institute_id"], "inst-1")

    def test_usage_is_reported_for_every_attempt(self):
        seen = []
        llm = _StubLLM(payload(1), payload(2))
        plan_deck(topic="T", slide_count=2, llm=llm,
                  on_usage=lambda result, attempt: seen.append(attempt))
        self.assertEqual(seen, [1, 2])

    def test_a_grounded_plan_is_marked_grounded(self):
        llm = _StubLLM(payload(2))
        result = plan_deck(topic="T", slide_count=2, llm=llm,
                           passages=["From the chapter."])
        self.assertTrue(result.grounded)


class OutputBudgetTests(SimpleTestCase):
    """A fixed 6000-token budget cut a 10-slide plan off mid-JSON (the planner
    model reasons first, and reasoning is billed against the same budget), so
    the deck fell back to V1. The budget now grows with the deck."""

    def test_a_small_deck_keeps_the_old_floor(self):
        self.assertEqual(planner_output_budget(2), PLANNER_MIN_TOKENS)

    def test_a_ten_slide_deck_gets_well_over_the_budget_that_truncated_it(self):
        self.assertGreaterEqual(planner_output_budget(10), 10000)

    def test_the_budget_grows_with_the_deck_and_is_capped(self):
        budgets = [planner_output_budget(n) for n in range(1, 26)]
        self.assertEqual(budgets, sorted(budgets))
        self.assertEqual(max(budgets), PLANNER_MAX_TOKENS)

    def test_nonsense_counts_do_not_break_the_budget(self):
        for n in (0, -3, None):
            self.assertEqual(planner_output_budget(n), PLANNER_MIN_TOKENS)

    def test_plan_deck_sends_the_scaled_budget(self):
        import os
        from unittest import mock
        llm = _StubLLM(payload(10))
        with mock.patch.dict(os.environ, {"PPT_PLANNER_PARALLEL": "false"}):
            plan_deck(topic="T", slide_count=10, llm=llm)
        self.assertEqual(llm.calls[0]["max_tokens"], planner_output_budget(10))

    def test_an_explicit_budget_is_still_honoured(self):
        llm = _StubLLM(payload(2))
        plan_deck(topic="T", slide_count=2, llm=llm, max_tokens=1234)
        self.assertEqual(llm.calls[0]["max_tokens"], 1234)


class TextbookSourceTests(SimpleTestCase):
    """The backend sends passages as {"page_no", "content", ...}. The planner
    read only "text", so every real chapter was dropped and decks logged as
    grounded were written from general knowledge."""

    BOOK = [
        {"page_no": 102, "chunk_index": 0, "content": "Breathing brings air into the lungs.", "tokens": 8},
        {"page_no": 103, "chunk_index": 0, "content": "Pyruvate is broken down in mitochondria.", "tokens": 8},
    ]

    def test_the_books_own_content_field_reaches_the_planner(self):
        _, user = build_planner_prompts("Respiration", 2, passages=self.BOOK)
        self.assertIn("Breathing brings air into the lungs.", user)
        self.assertIn("Pyruvate is broken down in mitochondria.", user)

    def test_each_passage_carries_its_page(self):
        _, user = build_planner_prompts("Respiration", 2, passages=self.BOOK)
        self.assertIn("[p.102]", user)
        self.assertIn("[p.103]", user)

    def test_the_planner_is_told_to_stop_where_the_book_stops(self):
        _, user = build_planner_prompts("Respiration", 2, passages=self.BOOK)
        self.assertIn("if the book stops at a level of detail, so does the deck", user)

    def test_a_deck_is_only_called_grounded_when_book_text_was_sent(self):
        grounded = plan_deck(topic="T", slide_count=2, llm=_StubLLM(payload(2)),
                             passages=self.BOOK)
        empty = plan_deck(topic="T", slide_count=2, llm=_StubLLM(payload(2)),
                          passages=[{"page_no": 1, "content": "   "}])
        self.assertTrue(grounded.grounded)
        self.assertFalse(empty.grounded)

    def test_no_usable_text_means_no_empty_source_heading(self):
        # It used to say "use ONLY these facts" and then list nothing.
        _, user = build_planner_prompts("T", 2, passages=[{"page_no": 4, "content": ""}])
        self.assertNotIn("SOURCE MATERIAL", user)


class ReviseSlidesTests(SimpleTestCase):
    def test_it_sends_the_slides_and_the_note_and_returns_the_same_count(self):
        from ai_services.core.ppt_v2.planner import revise_slides
        llm = _StubLLM(payload(1))
        out = revise_slides(slides=[slide("Krebs")], note="REVISION: drop Krebs.",
                            topic="Respiration", llm=llm)
        self.assertEqual(len(out), 1)
        prompt = llm.calls[0]["user_prompt"]
        self.assertIn("REVISION: drop Krebs.", prompt)
        self.assertIn('"Krebs"', prompt)

    def test_a_rewrite_of_the_wrong_length_is_an_error(self):
        from ai_services.core.ppt_v2.planner import revise_slides
        with self.assertRaises(PlannerError):
            revise_slides(slides=[slide("a")], note="n", topic="T", llm=_StubLLM(payload(3)))


class ParallelPlanningTests(SimpleTestCase):
    """Outline first, then the slides in parallel batches. One call writing a
    whole 10-slide deck took 60-140s live; the batches run side by side."""

    BOOK = [{"page_no": 7, "content": "Leaves make food using sunlight."}]

    class Smart:
        """Answers the outline call with an outline and each batch with as many
        slides as its prompt asks for, recording every prompt."""

        def __init__(self, fail_batches=0, fail_outline=False):
            import threading
            self.calls, self.lock = [], threading.Lock()
            self.fail_batches, self.fail_outline = fail_batches, fail_outline

        def complete(self, **kw):
            import re
            with self.lock:
                self.calls.append(kw)
                fail = None
                if "You outline educational slide decks" in kw["system_prompt"]:
                    if self.fail_outline:
                        raise RuntimeError("outline server_error")
                    n = int(re.search(r"Return exactly (\d+) outline", kw["user_prompt"]).group(1))
                    return {"content": {"title": "Deck", "slides": [
                        {"slideType": "concept", "title": "Outlined %d" % (i + 1),
                         "focus": "f"} for i in range(n)]}, "model": "stub"}
                if "YOUR SLIDES" in kw["user_prompt"] and self.fail_batches:
                    self.fail_batches -= 1
                    fail = "{not json"
            if fail:
                return {"content": fail, "model": "stub"}
            n = int(re.search(r"Return exactly (\d+) slides", kw["user_prompt"]).group(1))
            first = re.search(r"writing slides (\d+)-", kw["user_prompt"])
            base = int(first.group(1)) if first else 1
            return {"content": {"title": "", "slides": [
                slide("Slide %d" % (base + i)) for i in range(n)]}, "model": "stub"}

    def test_a_ten_slide_deck_is_an_outline_and_four_parallel_batches(self):
        llm = self.Smart()
        result = plan_deck(topic="Photosynthesis", slide_count=10, llm=llm, passages=self.BOOK)
        self.assertEqual([s["title"] for s in result.slides],
                         ["Slide %d" % i for i in range(1, 11)])
        self.assertEqual(len(llm.calls), 5)              # 1 outline + 4 batches of <=3
        self.assertTrue(result.grounded)

    def test_every_batch_sees_the_whole_outline_and_the_book(self):
        llm = self.Smart()
        plan_deck(topic="Photosynthesis", slide_count=6, llm=llm, passages=self.BOOK)
        batches = [c["user_prompt"] for c in llm.calls if "YOUR SLIDES" in c["user_prompt"]]
        self.assertEqual(len(batches), 2)
        for prompt in batches:
            self.assertIn("Outlined 1", prompt)
            self.assertIn("Outlined 6", prompt)
            self.assertIn("Leaves make food using sunlight.", prompt)

    def test_each_batch_is_handed_over_as_soon_as_it_is_written(self):
        # Image decks paint a batch while the others are still being written.
        got = []
        result = plan_deck(topic="Photosynthesis", slide_count=7, llm=self.Smart(),
                           on_batch=lambda start, slides, size: got.append((start, len(slides), size)))
        self.assertEqual(sorted(got), [(0, 3, 7), (3, 3, 7), (6, 1, 7)])
        self.assertEqual(len(result.slides), 7)

    def test_a_failing_batch_callback_never_costs_the_plan(self):
        def boom(*a):
            raise RuntimeError("painting queue down")
        result = plan_deck(topic="T", slide_count=6, llm=self.Smart(), on_batch=boom)
        self.assertEqual(len(result.slides), 6)

    def test_a_broken_batch_is_retried_once(self):
        llm = self.Smart(fail_batches=1)
        result = plan_deck(topic="T", slide_count=6, llm=llm)
        self.assertEqual(len(result.slides), 6)

    def test_a_failed_outline_falls_back_to_one_call(self):
        llm = self.Smart(fail_outline=True)
        result = plan_deck(topic="T", slide_count=6, llm=llm)
        self.assertEqual(len(result.slides), 6)
        self.assertNotIn("YOUR SLIDES", llm.calls[-1]["user_prompt"])

    def test_small_decks_and_corrections_stay_one_call(self):
        llm = self.Smart()
        plan_deck(topic="T", slide_count=4, llm=llm)
        self.assertEqual(len(llm.calls), 1)
        llm = self.Smart()
        plan_deck(topic="T", slide_count=6, llm=llm, revision_note="REVISION: x")
        self.assertEqual(len(llm.calls), 1)

    def test_it_can_be_switched_off(self):
        import os
        from unittest import mock
        llm = self.Smart()
        with mock.patch.dict(os.environ, {"PPT_PLANNER_PARALLEL": "false"}):
            plan_deck(topic="T", slide_count=6, llm=llm)
        self.assertEqual(len(llm.calls), 1)

    def test_a_repeatable_error_does_not_fall_back_into_a_second_failure(self):
        class NoCredit:
            def __init__(self):
                self.calls = 0

            def complete(self, **kw):
                self.calls += 1
                raise RuntimeError("HTTP 402 credit limit")
        llm = NoCredit()
        with self.assertRaises(PlannerError):
            plan_deck(topic="T", slide_count=6, llm=llm)
        self.assertEqual(llm.calls, 1)

