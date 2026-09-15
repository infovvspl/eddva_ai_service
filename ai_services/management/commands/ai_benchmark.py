"""
Inspect the model router and benchmark candidate models locally.

    python manage.py ai_benchmark --list
    python manage.py ai_benchmark --plan --feature doubt_resolve --model openai/gpt-oss-120b
    python manage.py ai_benchmark --discover together
    python manage.py ai_benchmark --provider together --model qwen3.8-flash --prompt-file chapter.txt
    python manage.py ai_benchmark --candidates capability:content \\
        --system-file prompts/notes_system.txt --prompt-file prompts/chapter.txt \\
        --repeat 3 --out bench.jsonl

Benchmarks call candidate adapters directly: production routing, policies,
local overrides and tenant usage are untouched. Credentials are only ever
reported as configured yes/no.
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
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--list", action="store_true", help="Show the model registry and configured state.")
        mode.add_argument("--plan", action="store_true", help="Show the route a request would take.")
        mode.add_argument("--discover", metavar="PROVIDER",
                          help="List the provider account's model ids and validate configured ids against them.")
        mode.add_argument("--candidates", metavar="SELECTOR",
                          help="'all', 'capability:<cap>', or comma-separated registry ids / provider:alias.")
        parser.add_argument("--feature")
        parser.add_argument("--capability")
        parser.add_argument("--model",
                            help="With --plan: pinned provider model id. Otherwise: benchmark this model "
                                 "(alias such as qwen3.8-flash, registry id, or configured provider model id).")
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

        if opts["candidates"]:
            ids = benchmark.candidate_ids_for(router, opts["candidates"])
            if not ids:
                raise CommandError(f"selector {opts['candidates']!r} matched no models")
        elif opts["model"]:
            rid = benchmark.resolve_selector(router, opts["provider"], opts["model"])
            if rid is None:
                known = sorted(s.alias for s in router.config.models.values() if s.provider == opts["provider"])
                raise CommandError(
                    f"--provider {opts['provider']} --model {opts['model']!r} is not a registered model. "
                    f"Registered {opts['provider']} aliases: {known}"
                )
            ids = [rid]
        else:
            raise CommandError("choose one of --list, --plan, --discover, --candidates, or --provider/--model")

        user_prompt = _read(opts["prompt"], opts["prompt_file"])
        if not user_prompt.strip():
            raise CommandError("--prompt or --prompt-file is required for a benchmark run")
        system_prompt = _read(opts["system"], opts["system_file"])

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
        from ai_services.core.routing.registry import support_label

        cfg = router.config
        self.stdout.write(
            f"router_enabled={cfg.enabled} fallback_enabled={cfg.fallback_enabled} debug={cfg.debug} "
            f"local_overrides={dict(cfg.overrides) or 'none'}"
        )
        for w in cfg.warnings:
            self.stdout.write(self.style.WARNING(f"config warning: {w}"))
        self.stdout.write(
            f"{'registry id':27} {'provider':8} {'model id':10} {'creds':5} {'json':8} {'long_ctx':8} "
            f"{'vision':7} {'ground':7} {'stream':7} {'quality':8} {'cost':6} {'source':10} {'env var':34} capabilities"
        )
        for rid in sorted(cfg.models):
            s = cfg.models[rid]
            adapter = router.adapters.get(s.provider)
            creds = "yes" if (adapter and adapter.is_configured()) else "no"
            mid = "configured" if s.is_configured else "MISSING"
            self.stdout.write(
                f"{rid:27} {s.provider:8} {mid:10} {creds:5} {s.structured_output:8} "
                f"{support_label(s.long_context):8} {support_label(s.multimodal):7} "
                f"{support_label(s.supports_grounding):7} {support_label(s.streaming):7} {s.quality_tier:8} "
                f"{s.cost_tier:6} {s.metadata_source:10} {(s.model_env or '-'):34} {','.join(sorted(s.capabilities))}"
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
        from ai_services.core.routing import benchmark

        result = benchmark.discover(router, provider)
        status = result["status"]
        if status != benchmark.PASS:
            detail = result.get("reason") or result.get("error") or ""
            kind = f" ({result['kind']})" if result.get("kind") else ""
            self.stdout.write(f"{status}{kind}: model discovery for {provider!r} {detail}".rstrip())
            return
        for rec in result["models"]:
            self.stdout.write(json.dumps(rec, ensure_ascii=False, default=str))
        self.stdout.write(f"\n{result['count']} models reported by {provider}")
        self.stdout.write("\nconfigured registry ids validated against the account:")
        for v in result["validation"]:
            present = {True: "present", False: "NOT FOUND in account", None: "not configured"}[v["present_in_account"]]
            self.stdout.write(f"  {v['registry_id']:28} {v['env'] or '-':34} {present}")
