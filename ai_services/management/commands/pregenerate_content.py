"""
Pre-generate static content during off-peak hours using Batch API.

Usage:
    python manage.py pregenerate_content --type tests --topics "Physics Ch1,Chemistry Ch2,Math Ch3"
    python manage.py pregenerate_content --type resources --topics "Thermodynamics,Organic Chemistry"

Run via cron at 2 AM IST:
    0 2 * * * cd /app && python manage.py pregenerate_content --type tests --topics-file topics.txt

Saves ~30% latency cost by serving pre-generated content from cache.
"""

import json
import logging
from django.core.management.base import BaseCommand

from ai_services.core.batch_processor import BatchProcessor
from ai_services.core.cache import ResponseCache
from ai_services.core.prompt_templates import get_template

logger = logging.getLogger("ai_services.pregenerate")


class Command(BaseCommand):
    help = "Pre-generate MCQs and content resources offline for caching"

    def add_arguments(self, parser):
        parser.add_argument(
            "--type",
            choices=["tests", "resources"],
            required=True,
            help="Type of content to pre-generate",
        )
        parser.add_argument(
            "--topics",
            type=str,
            help="Comma-separated list of topics",
        )
        parser.add_argument(
            "--topics-file",
            type=str,
            help="File with one topic per line",
        )
        parser.add_argument(
            "--num-questions",
            type=int,
            default=10,
            help="Number of questions per topic (for tests)",
        )
        parser.add_argument(
            "--difficulty",
            type=str,
            default="medium",
            choices=["easy", "medium", "hard"],
        )
        parser.add_argument(
            "--institute-id",
            type=str,
            default="pregenerate",
        )

    def handle(self, *args, **options):
        topics = self._get_topics(options)
        if not topics:
            self.stderr.write("No topics provided. Use --topics or --topics-file.")
            return

        content_type = options["type"]
        institute_id = options["institute_id"]

        self.stdout.write(f"Pre-generating {content_type} for {len(topics)} topics...")

        if content_type == "tests":
            self._pregenerate_tests(topics, options, institute_id)
        else:
            self._pregenerate_resources(topics, institute_id)

        self.stdout.write(self.style.SUCCESS("Pre-generation complete. Results cached."))

    def _get_topics(self, options) -> list:
        if options.get("topics"):
            return [t.strip() for t in options["topics"].split(",") if t.strip()]
        if options.get("topics_file"):
            with open(options["topics_file"]) as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def _pregenerate_tests(self, topics, options, institute_id):
        template = get_template("test_generate")
        num_q = options["num_questions"]
        difficulty = options["difficulty"]

        prompts = [
            template.user_template.format(
                topic=t,
                num_questions=num_q,
                difficulty=difficulty,
                question_types="mcq, true_false, short_answer",
            )
            for t in topics
        ]

        processor = BatchProcessor()
        job = processor.create_job("test_generate", institute_id, prompts)
        processor.run(job)

        # Cache all successful results
        cache = ResponseCache()
        for item, prompt in zip(job.items, prompts):
            if item.result:
                cache.set(institute_id, "test_generate", prompt, item.result)
                self.stdout.write(f"  Cached: {topics[prompts.index(prompt)]}")

        progress = job.progress
        self.stdout.write(
            f"  Tests: {progress['completed']}/{progress['total']} succeeded, "
            f"{progress['failed']} failed"
        )

    def _pregenerate_resources(self, topics, institute_id):
        template = get_template("content_suggest")
        cache = ResponseCache()
        processor = BatchProcessor()

        prompts = [template.user_template.format(topic=t) for t in topics]
        job = processor.create_job("content_suggest", institute_id, prompts)
        processor.run(job)

        for item, prompt in zip(job.items, prompts):
            if item.result:
                cache.set(institute_id, "content_suggest", prompt, item.result)
                self.stdout.write(f"  Cached: {topics[prompts.index(prompt)]}")

        progress = job.progress
        self.stdout.write(
            f"  Resources: {progress['completed']}/{progress['total']} succeeded, "
            f"{progress['failed']} failed"
        )
