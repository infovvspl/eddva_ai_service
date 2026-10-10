"""Generalisation benchmark: sixteen chapters, ten subjects, one engine.

What this measures is STRATEGY SELECTION - that the engine decides how to
teach from the shape of the content - not that a file comes out. Every case
is a realistic planner slide for the stated class, and three things are
checked:

  declared    the planner names a content type; the engine validates it and
              picks the pedagogical strategy, visual and layout.
  inferred    the same slide with the planner saying only "concept" (its own
              "unsure"); what the engine reaches from the content alone is
              RECORDED for every case and ASSERTED only where the slide's text
              carries unambiguous shape vocabulary. Cases where inference
              honestly falls back to concept explanation are listed, not hidden.
  composed    the slide through compose_deck: layout, fallback chain, the
              exclusions (no generated maps, flowcharts or measured figures),
              structural QA, and - with every producer stubbed - the label
              contract after asset resolution.

No LLM, no image provider, no network.
"""
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from unittest import mock

from ai_services.core.ppt_v2 import content, labels, pedagogy, qa
from ai_services.core.ppt_v2.content import ContentType as CT
from ai_services.core.ppt_v2.pedagogy import PedagogicalStrategy as PS
from ai_services.core.ppt_v2.schema import VisualStrategy as VS

AI, DIAGRAM, SEARCH = VS.AI_GENERATED_IMAGE, VS.PROGRAMMATIC_DIAGRAM, VS.SEARCHED_IMAGE
ICONS, NONE = VS.ICON_COMPOSITION, VS.NONE


def bullets(*texts):
    return [{"kind": "bullet", "label": t.split(":")[0], "text": t.split(":", 1)[-1].strip()}
            for t in texts]


def steps(*texts):
    return [{"kind": "step", "label": "Step %d" % (i + 1), "text": t}
            for i, t in enumerate(texts)]


@dataclass
class Case:
    name: str
    subject: str
    chapter: str
    slide: Dict[str, Any]
    content_type: CT
    strategy: PS
    visual: VS                       # primary, with every capability available
    layout: str
    no_ai: bool = False              # AI must not appear anywhere in the chain
    infers: bool = False             # inference alone must reach content_type
    class_name: str = "Class 8"

    def payload(self, declared=True):
        data = dict(self.slide)
        if declared:
            data["contentType"] = self.content_type.value
        else:
            data.pop("contentType", None)
            data["slideType"] = "concept"
        return data


BENCHMARK: List[Case] = [
    # ---------------- SCIENCE ----------------
    Case("Photosynthesis", "Science", "Photosynthesis", {
        "slideType": "process", "title": "How a Leaf Makes Its Food",
        "learningObjective": "Put the stages of photosynthesis in order",
        "blocks": steps("Chlorophyll in the leaf absorbs energy from sunlight.",
                        "Roots take in water and stomata take in carbon dioxide.",
                        "The leaf uses this energy to make glucose.",
                        "Oxygen is released into the air through the stomata."),
        "visual": {"kind": "illustration",
                   "description": "a green leaf in sunlight with water and carbon dioxide going in and oxygen coming out",
                   "objects": ["green leaf", "sunlight", "water", "carbon dioxide"],
                   "relationships": ["sunlight falls on the leaf", "oxygen leaves the leaf"],
                   "annotations": [{"text": "Sunlight", "anchor": [0.8, 0.1]},
                                   {"text": "Carbon dioxide in", "anchor": [0.2, 0.4]},
                                   {"text": "Water in", "anchor": [0.4, 0.9]},
                                   {"text": "Glucose made", "anchor": [0.5, 0.5]},
                                   {"text": "Oxygen out", "anchor": [0.8, 0.5]}]}},
        CT.PROCESS, PS.SEQUENTIAL_PROCESS, ICONS, "process_steps", infers=True),

    Case("Cell Structure", "Science", "Cell Structure and Functions", {
        "slideType": "anatomy", "title": "Parts of a Plant Cell",
        "learningObjective": "Name the main parts of a plant cell and say what each does",
        "blocks": bullets("Cell wall: A stiff outer layer that gives the cell its shape.",
                          "Nucleus: Controls the activities of the cell.",
                          "Chloroplast: Contains chlorophyll and makes food.",
                          "Vacuole: A large space that stores water."),
        "visual": {"kind": "diagram", "description": "a plant cell cut open to show its parts",
                   "objects": ["cell wall", "nucleus", "chloroplast", "vacuole"],
                   "annotations": [{"text": "Cell wall", "anchor": [0.05, 0.5]},
                                   {"text": "Nucleus", "anchor": [0.5, 0.4]},
                                   {"text": "Chloroplast", "anchor": [0.3, 0.7]},
                                   {"text": "Vacuole", "anchor": [0.6, 0.6]}]}},
        CT.ANATOMY_STRUCTURE, PS.LABELLED_STRUCTURE, AI, "anatomy_labeled", infers=True),

    Case("Light / Reflection", "Science", "Light", {
        "slideType": "concept", "title": "Reflection from a Plane Mirror",
        "learningObjective": "State the laws of reflection using a ray diagram",
        "blocks": bullets("Incident ray: The ray of light that strikes the mirror.",
                          "Reflected ray: The ray that bounces back from the mirror.",
                          "Normal: A line at 90 degrees to the mirror where the ray strikes.",
                          "Law: The angle of incidence equals the angle of reflection."),
        "visual": {"kind": "diagram", "description": "a ray diagram at a plane mirror",
                   "objects": ["plane mirror", "incident ray", "reflected ray", "normal"],
                   "annotations": [{"text": "Incident ray", "anchor": [0.2, 0.3]},
                                   {"text": "Reflected ray", "anchor": [0.8, 0.3]},
                                   {"text": "Normal", "anchor": [0.5, 0.2]}]}},
        CT.GEOMETRIC_DIAGRAM, PS.MEASURED_FIGURE, DIAGRAM, "anatomy_labeled",
        no_ai=True, infers=True),

    Case("Force and Pressure", "Science", "Force and Pressure", {
        "slideType": "concept", "title": "What Is Pressure?",
        "learningObjective": "Explain why the same force can press harder on a small area",
        "blocks": bullets("Pressure: The force acting on a unit area of a surface.",
                          "Small area: A sharp knife cuts easily because its edge is thin.",
                          "Large area: Wide straps on a school bag feel lighter on the shoulder."),
        "visual": {"kind": "photo", "description": "a sharp knife cutting an apple",
                   "objects": ["knife", "apple"]}},
        CT.CONCEPT_EXPLANATION, PS.EXPLAIN_WITH_SUPPORT, SEARCH, "concept_hero_split",
        infers=True),

    Case("Metals and Non-metals", "Science", "Materials: Metals and Non-Metals", {
        "slideType": "comparison", "title": "Metals Compared with Non-metals",
        "learningObjective": "Compare the physical properties of metals and non-metals",
        "blocks": [{"kind": "compare_left", "label": "Metals",
                    "text": "Metals are shiny, hard and good conductors of heat."},
                   {"kind": "compare_right", "label": "Non-metals",
                    "text": "Non-metals are dull, brittle and poor conductors of heat."}],
        "visual": {"kind": "photo", "description": "a copper wire beside a lump of coal",
                   "objects": ["copper wire", "coal"]}},
        # Icons, not a photograph: the comparison layout carries both sides
        # natively, and the existing engine deliberately prefers an icon pair
        # unless the objects are on its photographable list.
        CT.COMPARISON, PS.SIDE_BY_SIDE, ICONS, "comparison", infers=True),

    # ---------------- MATHEMATICS ----------------
    Case("Linear Equations", "Mathematics", "Linear Equations in One Variable", {
        "slideType": "formula", "title": "Solve 3x + 5 = 20",
        "learningObjective": "Solve a linear equation by transposing terms",
        "blocks": steps("Transpose 5 to the right side: 3x = 20 - 5 = 15.",
                        "Divide both sides by 3: x = 15 / 3 = 5.",
                        "Check the answer: 3(5) + 5 = 20, so x = 5 is correct."),
        "visual": {"kind": "none"}},
        CT.WORKED_EXAMPLE, PS.STEPWISE_SOLUTION, NONE, "formula_equation", infers=True),

    Case("Quadrilaterals", "Mathematics", "Understanding Quadrilaterals", {
        "slideType": "concept", "title": "Properties of a Parallelogram",
        "learningObjective": "List the properties of the sides, angles and diagonals of a parallelogram",
        "blocks": bullets("Sides: Opposite sides are parallel and equal in length.",
                          "Angles: Opposite angles are equal.",
                          "Diagonals: The diagonals bisect each other."),
        "visual": {"kind": "diagram", "description": "a parallelogram ABCD with both diagonals drawn",
                   "objects": ["parallelogram", "diagonal"],
                   "annotations": [{"text": "AB parallel to DC", "anchor": [0.5, 0.1]},
                                   {"text": "Diagonals bisect", "anchor": [0.5, 0.5]}]}},
        CT.GEOMETRIC_DIAGRAM, PS.MEASURED_FIGURE, DIAGRAM, "anatomy_labeled",
        no_ai=True, infers=True),

    Case("Data Handling", "Mathematics", "Data Handling", {
        "slideType": "concept", "title": "Reading a Bar Graph",
        "learningObjective": "Read values and compare categories from a bar graph",
        "blocks": bullets("Survey: 40 students named their favourite sport.",
                          "Axis: The height of each bar shows the number of students.",
                          "Compare: Cricket has the tallest bar, so it is the most popular."),
        "visual": {"kind": "chart", "description": "a bar graph of favourite sports",
                   "objects": ["bar graph", "axis"]}},
        CT.GRAPH_DATA, PS.QUANTITATIVE_GRAPH, DIAGRAM, "concept_hero_split",
        no_ai=True, infers=True),

    # ---------------- SOCIAL SCIENCE ----------------
    Case("Indian National Movement", "Social Science", "The Making of the National Movement", {
        "slideType": "timeline", "title": "Milestones of the National Movement",
        "learningObjective": "Place the key events of the national movement in order",
        "blocks": bullets("1885: The Indian National Congress was formed.",
                          "1905: The partition of Bengal led to the Swadeshi movement.",
                          "1920: Gandhiji launched the Non-Cooperation Movement.",
                          "1942: The Quit India Movement began.",
                          "1947: India became independent."),
        "visual": {"kind": "none"}},
        CT.TIMELINE, PS.CHRONOLOGICAL_SEQUENCE, NONE, "timeline", infers=True),

    Case("Parliament", "Social Science", "Why Do We Need a Parliament?", {
        "slideType": "anatomy", "title": "How Parliament Is Organised",
        "learningObjective": "Name the parts of Parliament and how a bill passes between them",
        "blocks": bullets("President: The head of the country, who signs a bill into law.",
                          "Lok Sabha: Members are elected directly by the people.",
                          "Rajya Sabha: Members are elected by the state legislatures.",
                          "Bill: A bill must pass both Houses and be signed by the President."),
        "visual": {"kind": "diagram",
                   "description": "the President, Lok Sabha and Rajya Sabha connected by arrows showing how a bill moves",
                   "objects": ["President", "Lok Sabha", "Rajya Sabha"],
                   "relationships": ["a bill moves from one House to the other"],
                   "annotations": [{"text": "Lok Sabha", "anchor": [0.25, 0.6]},
                                   {"text": "Rajya Sabha", "anchor": [0.75, 0.6]},
                                   {"text": "President", "anchor": [0.5, 0.15]}]}},
        CT.SYSTEM_ARCHITECTURE, PS.COMPONENT_RELATIONSHIP, DIAGRAM, "anatomy_labeled",
        no_ai=True),

    Case("Water Resources", "Social Science", "Water Resources", {
        "slideType": "concept", "title": "Major Rivers and Dams of India",
        "learningObjective": "Locate the major rivers and dams of India on a map",
        "blocks": bullets("Ganga: Flows from the Himalayas east across the northern plains.",
                          "Narmada: Flows west through central India to the Arabian Sea.",
                          "Bhakra Nangal: A large dam located on the Sutlej river in the north."),
        "visual": {"kind": "photo", "description": "a map of India showing major rivers and dams",
                   "objects": ["map of India", "rivers", "dams"],
                   "annotations": [{"text": "Ganga", "anchor": [0.6, 0.35]},
                                   {"text": "Narmada", "anchor": [0.35, 0.55]}]}},
        CT.GEOGRAPHIC_MAP, PS.SPATIAL_MAP, SEARCH, "anatomy_labeled",
        no_ai=True, infers=True),

    # ---------------- ECONOMICS ----------------
    Case("Demand and Supply", "Economics", "Demand and Supply", {
        "slideType": "concept", "title": "The Demand Curve",
        "learningObjective": "Read a demand curve and explain why it slopes downward",
        "blocks": bullets("Axes: Price is on the y axis and quantity demanded on the x axis.",
                          "Slope: The curve slopes downward from left to right.",
                          "Reason: When price falls, people buy more of the good."),
        "visual": {"kind": "chart", "description": "a downward-sloping demand curve",
                   "objects": ["demand curve", "price axis", "quantity axis"]}},
        CT.GRAPH_DATA, PS.QUANTITATIVE_GRAPH, DIAGRAM, "concept_hero_split",
        no_ai=True, infers=True, class_name="Class 11"),

    # ---------------- ENGLISH ----------------
    Case("Grammar", "English", "Parts of Speech", {
        "slideType": "concept", "title": "Nouns, Verbs and Adjectives",
        "learningObjective": "Identify nouns, verbs and adjectives in a sentence",
        "blocks": [{"kind": "definition", "label": "Noun",
                    "text": "A noun is called a naming word, as in 'The dog barked'."},
                   {"kind": "definition", "label": "Verb",
                    "text": "A verb is called an action word, as in 'The dog barked'."},
                   {"kind": "definition", "label": "Adjective",
                    "text": "An adjective is a describing word, as in 'the brown dog'."}],
        "visual": {"kind": "illustration", "description": "a dog barking",
                   "objects": ["dog"]}},
        CT.DEFINITION_TERMINOLOGY, PS.TERM_AND_EXAMPLE, NONE, "summary_grid",
        infers=True),

    Case("Poetry", "English", "Poetry: The Ant and the Cricket", {
        "slideType": "concept", "title": "What the Poem Teaches Us",
        "learningObjective": "Explain the lesson of the poem in your own words",
        "blocks": bullets("Cricket: The cricket sang all summer and saved nothing.",
                          "Ant: The ant worked hard and stored food for winter.",
                          "Lesson: We should plan ahead for difficult times."),
        "visual": {"kind": "illustration", "description": "an ant carrying food past a singing cricket",
                   "objects": ["ant", "cricket"]}},
        CT.CONCEPT_EXPLANATION, PS.EXPLAIN_WITH_SUPPORT, AI, "concept_hero_split",
        infers=True),

    # ---------------- COMPUTER SCIENCE ----------------
    Case("Algorithm", "Computer Science", "Algorithms and Flowcharts", {
        "slideType": "process", "title": "Algorithm to Find the Largest of Three Numbers",
        "learningObjective": "Trace an algorithm that compares three numbers",
        "blocks": steps("Read three numbers A, B and C.",
                        "If A is greater than both B and C, A is the largest.",
                        "Otherwise compare B and C and print the larger one."),
        "visual": {"kind": "diagram", "description": "a flowchart with decision boxes",
                   "objects": ["start", "decision box", "output"],
                   "annotations": [{"text": "Is A > B?", "anchor": [0.5, 0.3]}]}},
        CT.FLOWCHART_ALGORITHM, PS.DECISION_FLOW, DIAGRAM, "concept_hero_split",
        no_ai=True, infers=True, class_name="Class 11"),

    Case("Database / DBMS", "Computer Science", "Database Concepts", {
        "slideType": "anatomy", "title": "Parts of a Database System",
        "learningObjective": "Name the components of a database system and how they interact",
        "blocks": bullets("User: Asks questions of the data through an application.",
                          "Application: Sends the user's request to the DBMS.",
                          "DBMS: The software that stores, finds and protects the data.",
                          "Database: The organised collection of data itself."),
        "visual": {"kind": "diagram",
                   "description": "a user, an application, a DBMS and a database connected by arrows",
                   "objects": ["user", "application", "DBMS", "database"],
                   "relationships": ["requests flow from the user through the DBMS to the database"],
                   "annotations": [{"text": "DBMS", "anchor": [0.6, 0.5]}]}},
        CT.SYSTEM_ARCHITECTURE, PS.COMPONENT_RELATIONSHIP, DIAGRAM, "anatomy_labeled",
        no_ai=True, infers=True, class_name="Class 11"),
]


# Inference misses we know about and do not paper over. Without a declared
# type, "Name the parts of Parliament" reads as a labelled structure: the cue
# is real ("parts of"), the reading is just the wrong one. Adding civics words
# to the cue lists to force it would be fitting the benchmark - the thing the
# engine must not do. The consequence is pinned in a test below: the miss
# routes to a generated image, which must then pass fail-closed vision
# against a contract that names the relationship. The planner is asked to
# declare the type, and the declared path is what the first test class asserts.
#
# Parliament was the first entry. Structural inference (separate named
# components with relationships drawn between them) now reaches system
# architecture without any civics vocabulary, so the entry is gone - which is
# what the pinning test below exists to force.
KNOWN_INFERENCE_MISSES: dict = {"Parliament": CT.ANATOMY_STRUCTURE}
# Back again under Safe Structural V2, by design. V1 reached system
# architecture here, but through a signal that also called the parts of one
# eye and the forces on one book a "system". V2 requires a relationship whose
# subject is a component; this slide's only relationship ("a bill moves from
# one House to the other") has none, so the heading's "parts of" decides. The
# declared path is unaffected, and the miss routes to a generated image that
# must pass fail-closed vision - pinned below.


def caps(**kw):
    from ai_services.core.ppt_v2.visuals import VisualCapabilities
    base = dict(ai_images=True, image_search=True, diagrams=True, cache=True,
                ai_image_budget=20)
    base.update(kw)
    return VisualCapabilities(**base)


def compose(case: Case, declared=True):
    from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
    return compose_deck(
        [case.payload(declared)],
        DeckContext(topic=case.chapter, subject=case.subject, chapter=case.chapter,
                    class_name=case.class_name, board="CBSE"),
        caps=caps()).slides[0]


def decide(case: Case, declared=True):
    a = content.analyse(case.payload(declared), subject=case.subject,
                        class_level=case.class_name, chapter=case.chapter,
                        topic=case.chapter)
    return pedagogy.decide(a)


def resolve_with_stubs(slide):
    """Run the real producers with every external service stubbed to succeed."""
    from ai_services.core.ppt_v2.producers import ProducerSet
    from ai_services.core.ppt_v2.visuals import resolve_visual

    ps = ProducerSet(search_fn=lambda t, ti, c=None, a=None: {
        "imageUrl": "https://s.test/%s.jpg" % abs(hash(t))})
    with mock.patch("ai_services.core.image_generation.generate_note_image",
                    return_value={"ok": True, "url": "https://ai.test/g.png",
                                  "provider": "together", "model": "m"}), \
         mock.patch("ai_services.core.diagram.render_diagram",
                    return_value={"success": True,
                                  "image_base64": "data:image/png;base64,AAA"}):
        resolve_visual(slide.visual, ps.as_map())
    labels.apply_policy(slide.visual)
    return slide


class DeclaredPathTests(unittest.TestCase):
    def test_every_case_is_decided_as_designed(self):
        for case in BENCHMARK:
            with self.subTest(case.name):
                d = decide(case)
                self.assertIs(d.content_type, case.content_type)
                self.assertIs(d.strategy, case.strategy)
                self.assertEqual(d.analysis.source, "declared", d.analysis.reason)

    def test_every_case_composes_to_the_expected_visual_and_layout(self):
        for case in BENCHMARK:
            with self.subTest(case.name):
                s = compose(case)
                self.assertEqual(s.content_type, case.content_type.value)
                self.assertIs(s.visual.strategy, case.visual)
                self.assertEqual(s.layout.name, case.layout)

    def test_falsifying_generators_never_appear_in_the_chain(self):
        for case in BENCHMARK:
            if case.no_ai:
                with self.subTest(case.name):
                    s = compose(case)
                    self.assertNotIn(AI, [s.visual.strategy] + s.visual.fallbacks)

    def test_every_chain_still_ends_somewhere(self):
        for case in BENCHMARK:
            with self.subTest(case.name):
                s = compose(case)
                chain = [s.visual.strategy] + s.visual.fallbacks
                self.assertIs(chain[-1], NONE)


class InferredPathTests(unittest.TestCase):
    def test_shape_vocabulary_alone_reaches_the_right_type(self):
        for case in BENCHMARK:
            if case.infers:
                with self.subTest(case.name):
                    d = decide(case, declared=False)
                    self.assertIs(d.content_type, case.content_type, d.reason)

    def test_cases_without_decisive_vocabulary_fall_back_gracefully(self):
        # Recorded honestly: without a declaration these reach concept
        # explanation - the graceful default - rather than a wrong guess.
        for case in BENCHMARK:
            if not case.infers and case.name not in KNOWN_INFERENCE_MISSES:
                with self.subTest(case.name):
                    d = decide(case, declared=False)
                    self.assertIn(d.content_type, (case.content_type,
                                                   CT.CONCEPT_EXPLANATION), d.reason)

    def test_known_misses_are_pinned_not_hidden(self):
        # If inference improves, this fails and the entry should be removed.
        for name, reached in KNOWN_INFERENCE_MISSES.items():
            case = next(c for c in BENCHMARK if c.name == name)
            with self.subTest(name):
                self.assertIs(decide(case, declared=False).content_type, reached)

    def test_a_known_miss_still_cannot_ship_an_unvalidated_picture(self):
        # The miss routes to a generated image. That image must be inspected
        # against a contract carrying the slide's relationship, fail-closed.
        from ai_services.core.ppt_v2.producers import ProducerSet

        case = next(c for c in BENCHMARK if c.name == "Parliament")
        s = compose(case, declared=False)
        self.assertIs(s.visual.strategy, AI)
        ps = ProducerSet(subject=case.subject, class_name=case.class_name,
                         chapter=case.chapter)
        contract = ps.visual_contract(s.visual)
        self.assertTrue(contract.relationships)
        self.assertTrue(ps._validation_required(contract))


class AdversarialDeclarationTests(unittest.TestCase):
    """A planner can name a type the content cannot carry. It gets corrected."""

    def _case(self, name):
        return next(c for c in BENCHMARK if c.name == name)

    def test_a_timeline_claimed_for_a_comparison_is_corrected(self):
        data = self._case("Metals and Non-metals").payload()
        data["contentType"] = "timeline"
        a = content.analyse(data, subject="Science", class_level="Class 8")
        self.assertEqual(a.source, "corrected")
        self.assertIs(a.content_type, CT.COMPARISON)

    def test_a_worked_example_with_no_working_is_corrected(self):
        data = self._case("Poetry").payload()
        data["contentType"] = "worked_example"
        a = content.analyse(data, subject="English", class_level="Class 8")
        self.assertEqual(a.source, "corrected")
        self.assertIsNot(a.content_type, CT.WORKED_EXAMPLE)

    def test_a_topic_named_layout_is_not_a_type(self):
        data = self._case("Force and Pressure").payload()
        data["contentType"] = "photosynthesis_layout"
        a = content.analyse(data, subject="Science", class_level="Class 8")
        self.assertIs(a.content_type, CT.CONCEPT_EXPLANATION)


class CompositionQualityTests(unittest.TestCase):
    def test_no_structural_errors_before_assets(self):
        for case in BENCHMARK:
            with self.subTest(case.name):
                s = compose(case)
                self.assertEqual([str(i) for i in qa.errors(qa.check_deck([s]))], [])

    def test_the_label_contract_holds_after_resolution(self):
        for case in BENCHMARK:
            with self.subTest(case.name):
                s = resolve_with_stubs(compose(case))
                self.assertEqual([str(i) for i in qa.check_label_contract(s)
                                  if i.severity == qa.ERROR], [])
                if s.visual.labels_required and s.visual.annotations:
                    self.assertIn(s.visual.label_policy, ("anchored", "legend", "native"))

    def test_the_label_contract_holds_when_every_preferred_producer_fails(self):
        # Force each case down its chain to the last raster producer.
        from ai_services.core.ppt_v2.producers import ProducerSet
        from ai_services.core.ppt_v2.visuals import resolve_visual

        for case in BENCHMARK:
            with self.subTest(case.name):
                s = compose(case)
                ps = ProducerSet(search_fn=lambda t, ti, c=None, a=None: {
                    "imageUrl": "https://s.test/last.jpg"})
                producers = ps.as_map()
                producers[AI] = lambda _s: {"ok": False, "error": "vision rejected: x"}
                producers[DIAGRAM] = lambda _s: {"ok": False, "error": "illegible"}
                resolve_visual(s.visual, producers)
                labels.apply_policy(s.visual)
                self.assertEqual([str(i) for i in qa.check_label_contract(s)
                                  if i.severity == qa.ERROR], [])
                if s.visual.resolved_strategy is SEARCH and s.visual.annotations:
                    self.assertTrue(all(a.leader is False for a in s.visual.annotations))

    def test_a_worked_example_states_its_problem_once(self):
        # Rasterising the benchmark showed Step 1 in the equation band AND
        # again as the first card. The band states the problem; nothing in it
        # repeats below.
        from ai_services.core.ppt_v2.schema import BlockKind

        s = compose(next(c for c in BENCHMARK if c.name == "Linear Equations"))
        first = s.content_blocks[0]
        self.assertIs(first.kind, BlockKind.EQUATION)
        self.assertEqual(first.text, "3x + 5 = 20")
        cards = [b.text for b in s.content_blocks if b.kind is not BlockKind.EQUATION]
        self.assertEqual(len(cards), 3)
        self.assertNotIn(first.text, cards)

    def test_the_formula_column_holds_exactly_the_cards_that_render(self):
        # One slot per card: the equation lives in its own band. A slot for
        # it too left a dead box and squeezed the cards until text sat on
        # the border - visible in the raster, invisible to QA, because QA
        # also indexed slots by raw block position and so measured each slot
        # against the block before the one actually drawn there.
        s = compose(next(c for c in BENCHMARK if c.name == "Linear Equations"))
        slots = sorted(k for k in s.layout.regions if k.startswith("block_"))
        self.assertEqual(slots, ["block_0", "block_1", "block_2"])
        self.assertIn("Transpose 5", qa._slot_text(s, "block_0"))
        self.assertIn("Check the answer", qa._slot_text(s, "block_2"))
        self.assertEqual(qa._slot_text(s, "equation"), "3x + 5 = 20")

    def test_a_formula_slide_without_an_equation_block_gets_one(self):
        from ai_services.core.ppt_v2.compose import DeckContext, compose_deck
        from ai_services.core.ppt_v2.schema import BlockKind

        s = compose_deck([{"slideType": "formula", "title": "Speed",
                           "blocks": [{"kind": "bullet", "label": "Formula",
                                       "text": "We write v = d / t for speed."},
                                      {"kind": "bullet", "label": "Units",
                                       "text": "Speed is measured in metres per second."}]}],
                         DeckContext(subject="Science", class_name="Class 8"),
                         caps=caps()).slides[0]
        self.assertIs(s.content_blocks[0].kind, BlockKind.EQUATION)
        self.assertIn("v = d / t", s.content_blocks[0].text)

    def test_every_payload_round_trips_to_the_wire_format(self):
        import json
        for case in BENCHMARK:
            with self.subTest(case.name):
                wire = compose(case).to_v1_dict()
                json.dumps(wire)
                self.assertEqual(wire["v2"]["contentType"], case.content_type.value)


if __name__ == "__main__":
    unittest.main()
