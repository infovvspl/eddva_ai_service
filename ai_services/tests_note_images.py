import os
import unittest
from unittest.mock import patch

from ai_services.core.note_images import enrich_notes_with_images


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": self.content}


class FakeSequenceLLM:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": self.contents.pop(0)}


class NoteImageEnrichmentTests(unittest.TestCase):
    def test_skips_candidate_when_evidence_is_not_in_notes(self):
        notes = "# Motion\n\nVelocity is displacement divided by time."
        llm = FakeLLM(
            {
                "images": [
                    {
                        "section_heading": "Motion",
                        "caption": "Projectile motion path",
                        "visual_description": "A projectile moving in a parabolic path",
                        "evidence_quote": "A projectile moves in a parabolic path.",
                    }
                ]
            }
        )

        with patch.dict(
            os.environ,
            {
                "NOTES_IMAGE_GENERATION_ENABLED": "true",
                "NOTES_IMAGE_OVERLAY_LABELS_ENABLED": "true",
                "NOTES_IMAGE_MAX_PER_LECTURE": "3",
                "HF_IMAGE_TOKEN": "test-hf-image-token",
                "HF_IMAGE_MODEL": "black-forest-labs/FLUX.1-schnell",
            },
            clear=False,
        ):
            enriched, images, meta = enrich_notes_with_images(notes, "Motion", "en", "test", llm)

        self.assertEqual(enriched, notes)
        self.assertEqual(images, [])
        self.assertEqual(meta["count"], 0)

    def test_inserts_image_after_matching_section_for_grounded_candidate(self):
        evidence = "A convex lens forms a real inverted image when the object is beyond 2F."
        notes = (
            f"# Ray Optics\n\n## Convex Lens\n\n{evidence}\n\n"
            "Students should trace the two principal rays carefully: one ray parallel to the principal axis "
            "passes through the focus after refraction, and another ray through the optical center continues "
            "almost undeviated. This construction helps compare object position, image position, size, and "
            "orientation for different cases. The diagram is useful because the words real, inverted, enlarged, "
            "diminished, and same size become easier to connect with ray intersections.\n\n"
            "## Summary\n\nLens rules matter for locating images accurately in numerical and conceptual questions."
        )
        llm = FakeLLM(
            {
                "images": [
                    {
                        "section_heading": "Convex Lens",
                        "caption": "Convex lens image formation beyond 2F",
                        "visual_description": "Ray diagram for a convex lens with object beyond 2F",
                        "evidence_quote": evidence,
                    }
                ]
            }
        )

        with patch.dict(
            os.environ,
            {
                "NOTES_IMAGE_GENERATION_ENABLED": "true",
                "NOTES_IMAGE_OVERLAY_LABELS_ENABLED": "true",
                "NOTES_IMAGE_MAX_PER_LECTURE": "3",
                "HF_IMAGE_TOKEN": "test-hf-image-token",
                "HF_IMAGE_MODEL": "black-forest-labs/FLUX.1-schnell",
            },
            clear=False,
        ), patch(
            "ai_services.core.note_images.generate_note_image",
            return_value={
                "ok": True,
                "url": "https://example.com/lens.png",
                "provider": "huggingface",
                "model": "black-forest-labs/FLUX.1-schnell",
                "image_size": "768x512",
                "aspect_ratio": "768:512",
            },
        ):
            enriched, images, meta = enrich_notes_with_images(notes, "Ray Optics", "en", "test", llm)

        self.assertEqual(len(images), 1)
        self.assertEqual(meta["count"], 1)
        self.assertIn("![Convex lens image formation beyond 2F](https://example.com/lens.png)", enriched)
        self.assertLess(enriched.index("![Convex lens"), enriched.index(evidence))

    def test_dynamic_planning_allows_more_than_three_grounded_images(self):
        evidences = [
            "The cerebrum has frontal lobe functions for voluntary movement and planning.",
            "The parietal lobe processes touch, pressure, pain, and temperature sensations.",
            "The temporal lobe is associated with hearing and memory related processing.",
            "The occipital lobe receives and interprets visual information from the eyes.",
        ]
        notes = "# Cerebrum\n\n" + "\n\n".join(
            (
                f"## Section {i + 1}\n\n{e} "
                "This section includes enough explanatory classroom detail for students to connect the concept "
                "with a visual diagram in their lecture notes. The explanation is intentionally detailed enough "
                "to resemble real generated notes rather than a tiny unit-test stub."
            )
            for i, e in enumerate(evidences)
        )
        llm = FakeLLM(
            {
                "images": [
                    {
                        "section_heading": f"Section {i + 1}",
                        "caption": [
                            "Frontal lobe planning diagram",
                            "Parietal lobe sensation diagram",
                            "Temporal lobe hearing diagram",
                            "Occipital lobe vision diagram",
                        ][i],
                        "visual_description": [
                            "Educational visual for frontal lobe planning and voluntary movement",
                            "Educational visual for parietal lobe touch and temperature sensation",
                            "Educational visual for temporal lobe hearing and memory processing",
                            "Educational visual for occipital lobe visual interpretation",
                        ][i],
                        "evidence_quote": evidence,
                    }
                    for i, evidence in enumerate(evidences)
                ]
            }
        )

        with patch.dict(
            os.environ,
            {
                "NOTES_IMAGE_GENERATION_ENABLED": "true",
                "NOTES_IMAGE_MAX_PER_LECTURE": "0",
                "NOTES_IMAGE_HARD_MAX_PER_LECTURE": "8",
                "HF_IMAGE_TOKEN": "test-hf-image-token",
                "HF_IMAGE_MODEL": "black-forest-labs/FLUX.1-schnell",
            },
            clear=False,
        ), patch(
            "ai_services.core.note_images.generate_note_image",
            side_effect=[
                {
                    "ok": True,
                    "url": f"https://example.com/diagram-{i + 1}.png",
                    "provider": "huggingface",
                    "model": "black-forest-labs/FLUX.1-schnell",
                    "image_size": "768x512",
                    "aspect_ratio": "768:512",
                }
                for i in range(4)
            ],
        ):
            enriched, images, meta = enrich_notes_with_images(notes, "Cerebrum", "en", "test", llm)

        self.assertEqual(len(images), 4)
        self.assertEqual(meta["count"], 4)
        self.assertIn("![Occipital lobe vision diagram](https://example.com/diagram-4.png)", enriched)

    def test_duplicate_same_diagram_candidates_are_combined(self):
        evidence_1 = "The frontal lobe controls voluntary movement and planning in the cerebrum."
        evidence_2 = "The parietal lobe processes touch, pressure, pain, and temperature in the cerebrum."
        notes = (
            "# Cerebrum\n\n"
            f"## Lobes\n\n{evidence_1}\n\n{evidence_2}\n\n"
            "A single labeled cerebrum diagram helps students compare the lobes together instead of studying "
            "separate repeated diagrams for each lobe. This classroom note includes enough surrounding explanation "
            "to activate the same image enrichment path used for full lecture notes in production. Additional details "
            "explain that both labels belong on the same cerebrum outline, so the generated visual should combine them."
        )
        llm = FakeLLM(
            {
                "images": [
                    {
                        "section_heading": "Lobes",
                        "diagram_group": "cerebrum lobe diagram",
                        "caption": "Cerebrum lobe diagram",
                        "visual_description": "Cerebrum diagram labeling the frontal lobe",
                        "evidence_quote": evidence_1,
                    },
                    {
                        "section_heading": "Lobes",
                        "diagram_group": "cerebrum lobe diagram",
                        "caption": "Cerebrum lobe diagram",
                        "visual_description": "Cerebrum diagram labeling the parietal lobe",
                        "evidence_quote": evidence_2,
                    },
                ]
            }
        )
        captured_prompts = []

        with patch.dict(
            os.environ,
            {
                "NOTES_IMAGE_GENERATION_ENABLED": "true",
                "NOTES_IMAGE_MAX_PER_LECTURE": "0",
                "HF_IMAGE_TOKEN": "test-hf-image-token",
                "HF_IMAGE_MODEL": "black-forest-labs/FLUX.1-schnell",
            },
            clear=False,
        ), patch(
            "ai_services.core.note_images.generate_note_image",
            side_effect=lambda prompt: captured_prompts.append(prompt)
            or {
                "ok": True,
                "url": "https://example.com/cerebrum-lobes.png",
                "provider": "huggingface",
                "model": "black-forest-labs/FLUX.1-schnell",
                "image_size": "768x512",
                "aspect_ratio": "768:512",
            },
        ):
            enriched, images, meta = enrich_notes_with_images(notes, "Cerebrum", "en", "test", llm)

        self.assertEqual(len(images), 1)
        self.assertEqual(meta["count"], 1)
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("frontal lobe", captured_prompts[0])
        self.assertIn("parietal lobe", captured_prompts[0])
        self.assertIn("No labels", captured_prompts[0])
        self.assertIn("No arrows", captured_prompts[0])
        self.assertIn("clean visual only", captured_prompts[0])
        self.assertIn("![Cerebrum lobe diagram](https://example.com/cerebrum-lobes.png)", enriched)
        self.assertNotIn("Legend:", enriched)

    def test_shared_diagram_group_candidates_are_combined_without_topic_hardcoding(self):
        evidence_1 = "The brain and spinal cord together form the central nervous system."
        evidence_2 = "The brain contains the cerebrum, cerebellum, pons, and medulla oblongata."
        notes = (
            "# Central Nervous System\n\n"
            f"## Brain and Spinal Cord\n\n{evidence_1}\n\n"
            f"## Brain Structure\n\n{evidence_2}\n\n"
            "A single CNS diagram can show the brain and spinal cord relationship along with the major brain parts. "
            "This classroom note includes enough surrounding explanation to match the full lecture notes path and "
            "to avoid creating repeated brain outline diagrams for closely related CNS concepts."
        )
        llm = FakeLLM(
            {
                "images": [
                    {
                        "section_heading": "Brain and Spinal Cord",
                        "diagram_group": "central nervous system overview",
                        "caption": "Brain and Spinal Cord Relationship",
                        "visual_description": "Diagram showing the brain connected to the spinal cord",
                        "evidence_quote": evidence_1,
                    },
                    {
                        "section_heading": "Brain Structure",
                        "diagram_group": "central nervous system overview",
                        "caption": "Brain Structure",
                        "visual_description": "Diagram showing cerebrum, cerebellum, pons, and medulla",
                        "evidence_quote": evidence_2,
                    },
                ]
            }
        )
        captured_prompts = []

        with patch.dict(
            os.environ,
            {
                "NOTES_IMAGE_GENERATION_ENABLED": "true",
                "NOTES_IMAGE_MAX_PER_LECTURE": "0",
                "HF_IMAGE_TOKEN": "test-hf-image-token",
                "HF_IMAGE_MODEL": "black-forest-labs/FLUX.1-schnell",
            },
            clear=False,
        ), patch(
            "ai_services.core.note_images.generate_note_image",
            side_effect=lambda prompt: captured_prompts.append(prompt)
            or {
                "ok": True,
                "url": "https://example.com/cns.png",
                "provider": "huggingface",
                "model": "black-forest-labs/FLUX.1-schnell",
                "image_size": "768x512",
                "aspect_ratio": "768:512",
            },
        ):
            enriched, images, meta = enrich_notes_with_images(notes, "CNS", "en", "test", llm)

        self.assertEqual(len(images), 1)
        self.assertEqual(meta["count"], 1)
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("brain connected to the spinal cord", captured_prompts[0])
        self.assertIn("cerebrum, cerebellum, pons, and medulla", captured_prompts[0])
        self.assertIn("No labels", captured_prompts[0])
        self.assertIn("No arrows", captured_prompts[0])
        self.assertIn("clean visual only", captured_prompts[0])
        self.assertNotIn("Legend:", enriched)

    def test_encodes_post_generation_overlay_labels_in_image_alt(self):
        evidence = "The digestive system includes the mouth, esophagus, stomach, small intestine, and large intestine."
        notes = (
            "# Digestion\n\n"
            f"## Digestive System\n\n{evidence}\n\n"
            "A digestive system pathway diagram helps students follow food movement through the body. "
            "The same diagram should remain grounded to the named organs in the classroom notes and avoid "
            "extra unsupported anatomy. The teacher explains that a single pathway visual is useful because "
            "students can see the order of the digestive tract instead of memorizing disconnected names. "
            "The notes repeatedly focus on the mouth, esophagus, stomach, small intestine, and large intestine "
            "as the relevant visible organs for this lecture section, so the image should not introduce unrelated "
            "organs or decorative details."
        )
        llm = FakeLLM(
            {
                "images": [
                    {
                        "section_heading": "Digestive System",
                        "diagram_group": "digestive system pathway",
                        "caption": "Digestive system pathway",
                        "visual_description": "Diagram showing the digestive system pathway",
                        "label_terms": ["mouth", "esophagus", "stomach", "small intestine", "large intestine"],
                        "evidence_quote": evidence,
                    }
                ]
            }
        )

        with patch.dict(
            os.environ,
            {
                "NOTES_IMAGE_GENERATION_ENABLED": "true",
                "NOTES_IMAGE_MAX_PER_LECTURE": "3",
                "HF_IMAGE_TOKEN": "test-hf-image-token",
                "HF_IMAGE_MODEL": "black-forest-labs/FLUX.1-schnell",
            },
            clear=False,
        ), patch(
            "ai_services.core.note_images.generate_note_image",
            return_value={
                "ok": True,
                "url": "https://example.com/digestion.png",
                "path": "D:\\fake\\digestion.png",
                "mime_type": "image/png",
                "provider": "huggingface",
                "model": "black-forest-labs/FLUX.1-schnell",
                "image_size": "768x512",
                "aspect_ratio": "768:512",
            },
        ), patch(
            "ai_services.core.note_images.label_generated_note_image",
            return_value=([{"text": "stomach", "x": 0.52, "y": 0.44}], None),
        ) as label_mock:
            enriched, images, meta = enrich_notes_with_images(notes, "Digestion", "en", "test", llm)

        self.assertEqual(len(images), 1)
        self.assertEqual(meta["count"], 1)
        self.assertEqual(images[0]["overlay_labels"][0]["text"], "stomach")
        self.assertIn("stomach", images[0]["prompt"])
        self.assertIn("small intestine", images[0]["prompt"])
        self.assertEqual(label_mock.call_args.args[3], "en")
        self.assertIn("![Digestive system pathway <<NOTE_IMAGE_OVERLAY:", enriched)
        self.assertIn("](https://example.com/digestion.png)", enriched)
        self.assertNotIn("Legend:", enriched)

    def test_odia_overlay_labeling_receives_odia_language(self):
        evidence = "ପାଚନତନ୍ତ୍ରରେ ମୁଖ, ଖାଦ୍ୟନଳୀ, ପାକସ୍ଥଳୀ, କ୍ଷୁଦ୍ରାନ୍ତ୍ର ଏବଂ ବୃହଦାନ୍ତ୍ର ଥାଏ।"
        notes = (
            "# ପାଚନତନ୍ତ୍ର\n\n"
            f"## ପାଚନ ଅଙ୍ଗ\n\n{evidence}\n\n"
            "ଏହି ପାଠ୍ୟାଂଶରେ ଖାଦ୍ୟ ଗତି କରୁଥିବା ପଥକୁ ଚିତ୍ର ଦ୍ୱାରା ବୁଝାଇବା ଉପଯୁକ୍ତ। "
            "ଶିକ୍ଷାର୍ଥୀମାନେ ମୁଖରୁ ଆରମ୍ଭ କରି ଖାଦ୍ୟନଳୀ, ପାକସ୍ଥଳୀ, କ୍ଷୁଦ୍ରାନ୍ତ୍ର ଏବଂ "
            "ବୃହଦାନ୍ତ୍ର ପର୍ଯ୍ୟନ୍ତ ଅଙ୍ଗଗୁଡ଼ିକର କ୍ରମ ବୁଝିପାରିବେ। ଏହି ଚିତ୍ର କେବଳ ନୋଟ୍ସରେ "
            "ଥିବା ଅଙ୍ଗଗୁଡ଼ିକୁ ଦେଖାଇବା ଉଚିତ। ଚିତ୍ରଟି ଶ୍ରେଣୀ ନୋଟ୍ସ ସହିତ ସମ୍ପର୍କିତ ରହିବା "
            "ଦରକାର ଏବଂ ନୋଟ୍ସରେ ନଥିବା ଅନ୍ୟ କୌଣସି ଅଙ୍ଗ କିମ୍ବା ଅତିରିକ୍ତ ବିବରଣୀ ଯୋଡ଼ିବା "
            "ଉଚିତ ନୁହେଁ। ପାଚନତନ୍ତ୍ରର ଏହି ଅଂଶରେ ଅଙ୍ଗଗୁଡ଼ିକର ସ୍ଥାନ ଓ କ୍ରମ ବୁଝିବା "
            "ମୁଖ୍ୟ ଉଦ୍ଦେଶ୍ୟ, ତେଣୁ ଚିତ୍ର ଶିକ୍ଷାମୂଳକ ଏବଂ ସରଳ ହେବା ଦରକାର।"
        )
        llm = FakeLLM(
            {
                "images": [
                    {
                        "section_heading": "ପାଚନ ଅଙ୍ଗ",
                        "diagram_group": "odia digestive pathway",
                        "caption": "ପାଚନତନ୍ତ୍ର",
                        "visual_description": "ପାଚନ ଅଙ୍ଗଗୁଡ଼ିକର ଚିତ୍ର",
                        "evidence_quote": evidence,
                    }
                ]
            }
        )

        with patch.dict(
            os.environ,
            {
                "NOTES_IMAGE_GENERATION_ENABLED": "true",
                "NOTES_IMAGE_MAX_PER_LECTURE": "3",
                "HF_IMAGE_TOKEN": "test-hf-image-token",
                "HF_IMAGE_MODEL": "black-forest-labs/FLUX.1-schnell",
            },
            clear=False,
        ), patch(
            "ai_services.core.note_images._gemini_image_plan",
            return_value={
                "images": [
                    {
                        "section_heading": "ପାଚନ ଅଙ୍ଗ",
                        "diagram_group": "odia digestive pathway",
                        "caption": "ପାଚନତନ୍ତ୍ର",
                        "visual_description": "ପାଚନ ଅଙ୍ଗଗୁଡ଼ିକର ଚିତ୍ର",
                        "evidence_quote": evidence,
                    }
                ]
            },
        ), patch(
            "ai_services.core.note_images.generate_note_image",
            return_value={
                "ok": True,
                "url": "https://example.com/odia-digestion.png",
                "path": "D:\\fake\\odia-digestion.png",
                "mime_type": "image/png",
                "provider": "huggingface",
                "model": "black-forest-labs/FLUX.1-schnell",
                "image_size": "768x512",
                "aspect_ratio": "768:512",
            },
        ), patch(
            "ai_services.core.note_images.label_generated_note_image",
            return_value=([{"text": "ପାକସ୍ଥଳୀ", "x": 0.5, "y": 0.45}], None),
        ) as label_mock:
            enriched, images, meta = enrich_notes_with_images(notes, "ପାଚନତନ୍ତ୍ର", "od", "test", llm)

        self.assertEqual(len(images), 1)
        self.assertEqual(meta["count"], 1)
        self.assertEqual(label_mock.call_args.args[3], "od")
        self.assertIn("<<NOTE_IMAGE_OVERLAY:", enriched)

if __name__ == "__main__":
    unittest.main()
