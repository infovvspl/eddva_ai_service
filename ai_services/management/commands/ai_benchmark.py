"""
Inspect the model router and benchmark candidate models locally.

    python manage.py ai_benchmark --list
    python manage.py ai_benchmark --plan --feature doubt_resolve --model openai/gpt-oss-120b
    python manage.py ai_benchmark --discover together
    python manage.py ai_benchmark --candidates capability:content \\
        --system-file prompts/notes_system.txt --prompt-file prompts/chapter.txt \\
        --repeat 3 --out bench.jsonl

Benchmarks call candidate adapters directly: production routing, policies and
tenant usage are untouched. Credentials are only ever reported as configured
yes/no.
"""
import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError


def _read(text, path):
    if path:
        return Path(path).read_text(encoding="utf-8")
    return text or ""


class Command(BaseCommand):
    help = "Inspect AI routing and benchmark candidate models without changing production routing."
    requires_migrations_checks = False

    def add_arguments(self, parser):
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--list", action="store_true", help="Show the model registry and configured state.")
        mode.add_argument("--plan", action="store_true", help="Show the route a request would take.")
        mode.add_argument("--discover", metavar="PROVIDER", help="List model ids the provider account exposes.")
        mode.add_argument("--candidates", metavar="SELECTOR",
                          help="'all', 'capability:<cap>', or comma-separated registry ids.")
        parser.add_argument("--feature")
        parser.add_argument("--capability")
        parser.add_argument("--model", help="Pinned provider model id (as a legacy call site would pass).")
        parser.add_argument("--provider", default="groq")
        parser.add_argument("--vision", action="store_true")
        parser.add_argument("--long-context", action="store_true")
        parser.add_argument("--json-mode", action="store_true")
        parser.add_argument("--prompt")
        parser.add_argument("--prompt-file")
        parser.add_argument("--system", default="You are an expert Indian school teacher.")
        parser.add_argument("--system-file")
        parser.add_argument("--max-tokens", type=int, default=1024)
        parser.add_argument("--temperature", type=float, default=0.3)
        parser.add_argument("--repeat", type=int, default=1)
        parser.add_argument("--timeout", type=float, default=120.0)
        parser.add_argument("--raw-system-prompt", action="store_true",
                            help="Skip LLMClient's prompt shaping (Groq always applies it).")
        parser.add_argument("--show-output", action="store_true")
        parser.add_argument("--out", help="Append JSONL records to this file.")

    def handle(self, *args, **opts):
        from ai_services.core import routing
        from ai_services.core.routing import benchmark

        router = routing.get_router()

        if opts["list"]:
            return self._list(router)
        if opts["plan"]:
            return self._plan(router, opts)
        if opts["discover"]:
            return self._discover(router, opts["discover"])

        user_prompt = _read(opts["prompt"], opts["prompt_file"])
        if not user_prompt.strip():
            raise CommandError("--prompt or --prompt-file is required for a benchmark run")
        system_prompt = _read(opts["system"], opts["system_file"])
        ids = benchmark.candidate_ids_for(router, opts["candidates"])
        if not ids:
            raise CommandError(f"selector {opts['candidates']!r} matched no models")

        records = []
        for rid in ids:
            recs = benchmark.benchmark_candidate(
                router, rid, system_prompt=system_prompt, user_prompt=user_prompt,
                json_mode=opts["json_mode"], max_tokens=opts["max_tokens"],
                temperature=opts["temperature"], repeat=opts["repeat"], timeout_s=opts["timeout"],
                legacy_prompt_shaping=not opts["raw_system_prompt"], include_output=opts["show_output"],
            )
            for r in recs:
                self.stdout.write(json.dumps(r, ensure_ascii=False, default=str))
            records.extend(recs)

        if opts["out"]:
            with open(opts["out"], "a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        self.stdout.write("\nSUMMARY (measurements only — not a quality ranking)")
        for row in benchmark.summarize(records):
            self.stdout.write(json.dumps(row, default=str))

    def _list(self, router):
        cfg = router.config
        self.stdout.write(f"router_enabled={cfg.enabled} fallback_enabled={cfg.fallback_enabled}")
        for w in cfg.warnings:
            self.stdout.write(self.style.WARNING(f"config warning: {w}"))
        header = f"{'registry id':28} {'provider':9} {'model id':12} {'creds':6} {'quality':8} {'cost':6} {'source':11} capabilities"
        self.stdout.write(header)
        for rid in sorted(cfg.models):
            s = cfg.models[rid]
            adapter = router.adapters.get(s.provider)
            creds = "yes" if (adapter and adapter.is_configured()) else "no"
            mid = "configured" if s.is_configured else "MISSING"
            self.stdout.write(
                f"{rid:28} {s.provider:9} {mid:12} {creds:6} {s.quality_tier:8} {s.cost_tier:6} "
                f"{s.metadata_source:11} {','.join(sorted(s.capabilities))}"
            )
        self.stdout.write("\npolicies:")
        for cap, p in sorted(cfg.policies.items()):
            self.stdout.write(
                f"  {cap:12} primary={p.primary} fallbacks={list(p.fallbacks)} candidates={list(p.candidates)}"
            )

    def _plan(self, router, opts):
        from ai_services.core.routing import AIRequest, RoutingError

        req = AIRequest(
            system_prompt="", user_prompt="", feature=opts["feature"], capability=opts["capability"],
            model=opts["model"], provider=opts["provider"], requires_vision=opts["vision"],
            requires_long_context=opts["long_context"], json_mode=opts["json_mode"],
        )
        try:
            plan = router.plan(req)
        except RoutingError as exc:
            self.stdout.write(self.style.ERROR(f"NO ROUTE: {exc}"))
            return
        self.stdout.write(f"capability={plan.capability}")
        for c in plan.candidates:
            self.stdout.write(f"  {c.role:8} {c.spec.id:28} provider={c.spec.provider} model={c.model_id} source={c.source}")
        for rid, reason in plan.skipped:
            self.stdout.write(f"  skipped  {rid:28} {reason}")

    def _discover(self, router, provider):
        from ai_services.core.routing import ProviderError

        adapter = router.adapters.get(provider)
        if adapter is None or not hasattr(adapter, "list_models"):
            self.stdout.write(f"NOT SUPPORTED: model discovery for provider {provider!r}")
            return
        if not adapter.is_configured():
            self.stdout.write(f"BLOCKED: {provider} credentials not configured")
            return
        try:
            ids = adapter.list_models()
        except ProviderError as exc:
            self.stdout.write(self.style.ERROR(f"FAIL ({exc.kind}): {exc}"))
            return
        for mid in ids:
            self.stdout.write(mid)
        self.stdout.write(f"\n{len(ids)} model ids")
