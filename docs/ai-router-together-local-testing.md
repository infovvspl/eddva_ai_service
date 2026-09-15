# Testing Together models locally through the EDVA UI

This is a **local-development** procedure. It changes nothing in DEV or production:
every setting below lives in your local `eddva_ai_service/.env`, and the model
override is refused unless Django `DEBUG` is true.

Production defaults are unchanged. With none of these variables set, every
feature keeps its current model (Groq GPT-OSS for text, Gemini for grounded /
vision / Odia work), and cross-provider fallback stays off.

The request path is unchanged too:

```
EDVA UI -> NestJS -> AiBridgeService -> Django view -> LLMClient -> EDVA router -> Together adapter -> Together API
```

Authentication, tenant and user attribution, AI admission and usage telemetry all
run exactly as they do for Groq. The override only changes which model the router
sends an already-authorised request to.

---

## 1. Add your Together key

In `eddva_ai_service/.env` (never commit this file):

```
TOGETHER_API_KEY=<your key>
```

The key is only read from this variable. It is never logged, returned in errors,
written to telemetry (only a short sha256 fingerprint is), or printed by the
benchmark command.

## 2. Discover the real model ids

Model ids are **never built in** — they must come from your account:

```
python manage.py ai_benchmark --discover together
```

Copy the exact ids you want into:

```
TOGETHER_MODEL_GPT_OSS_120B=<id from discovery>
TOGETHER_MODEL_QWEN38_FLASH=<id from discovery>
TOGETHER_MODEL_GLM53_FLASH=<id from discovery>
TOGETHER_MODEL_DEEPSEEK_V4_FLASH=<id from discovery>
TOGETHER_MODEL_QWEN37_MAX=<id from discovery>
```

Run `--discover together` again: the validation table at the end must say
`present` for every id you configured.

## 3. Record what you have verified about each model

Every Together capability starts as **UNKNOWN**, and UNKNOWN never satisfies a
requirement. Set a flag only after you have verified it for that model:

```
<MODEL ENV>_STRUCTURED_OUTPUT=native|prompted|none|unknown
<MODEL ENV>_LONG_CONTEXT=true|false|unknown
<MODEL ENV>_MULTIMODAL=true|false|unknown
<MODEL ENV>_GROUNDING=true|false|unknown
<MODEL ENV>_CONTEXT_TOKENS=<integer>|unknown
```

For example: `TOGETHER_MODEL_QWEN38_FLASH_STRUCTURED_OUTPUT=prompted`.

Structured output matters most. Several EDVA features ask for a JSON answer:

| Value | What happens to a JSON request |
|---|---|
| `native` | sent with `response_format: json_object` |
| `prompted` | JSON requested by instruction and parsed from the text (explicit compatibility path) |
| `unknown` / `none` | **refused with a clear error** — never silently downgraded to text |

Check the current state at any time:

```
python manage.py ai_benchmark --list
```

## 4. Benchmark from the command line (optional)

Runs a model directly, bypassing policy, fallback, override and tenant usage:

```
python manage.py ai_benchmark --provider together --model qwen3.8-flash --prompt-file chapter.txt
python manage.py ai_benchmark --candidates capability:content --prompt-file chapter.txt --repeat 3 --out bench.jsonl
```

The output is measurements only — latency, tokens, pass/fail. It is not a quality
ranking. Judge quality in the UI.

## 5. Route the real UI to a Together model

In `eddva_ai_service/.env`:

```
DJANGO_DEBUG=true
AI_MODEL_OVERRIDE_ENABLED=true
AI_MODEL_OVERRIDE_CONTENT=together:qwen3.8-flash
```

Use one variable per capability, or `AI_MODEL_OVERRIDE_ALL` for everything
without its own override:

```
AI_MODEL_OVERRIDE_REASONING=together:gpt-oss-120b
AI_MODEL_OVERRIDE_LIGHTWEIGHT=together:deepseek-v4-flash
AI_MODEL_OVERRIDE_ALL=together:glm-5.3-flash
```

The value is `provider:alias`, where the alias is taken from the registry id.
The registered provider model id also works, but an unregistered model is rejected.

**Restart the Django AI service** after changing `.env`. Then confirm:

- The startup log shows `LOCAL MODEL OVERRIDE ACTIVE (development only, DEBUG=true): content -> together/qwen3.8-flash`.
- `ai_benchmark --list` shows `local_overrides={'content': 'together/qwen3.8-flash'}`.
- Every request the override serves logs
  `AI route attempt | ... source=local_override provider=together model=<id> outcome=success`.

If `DJANGO_DEBUG` is not true, the override is **refused** and logged as such,
and routing stays on its defaults.

### Which UI features each capability affects

| UI feature | Django endpoint | Capability | Asks for JSON |
|---|---|---|---|
| Student doubt | `/doubt/resolve` | reasoning (subject detection: lightweight) | partly |
| Tutor | `/tutor/session`, `/tutor/continue` | reasoning | no |
| Quiz generation | `/quiz/generate` | reasoning | no |
| Practice test / assessment | `/test/generate/` | reasoning | yes |
| Teacher recording analysis | `/teacher/analyze-recording` | reasoning | yes |
| Subjective rubrics / grading | `/grading/*` | reasoning | yes |
| Textbook / chapter / assessment-paper content (ungrounded) | `/content/generate` | content | no |
| Lecture notes (writing, merging, polishing) | `/stt/notes*` | content (transcript cleanup: lightweight) | no |
| PPT (ungrounded deck, slide regeneration) | `/ppt/*` | content | yes |

A feature marked **yes** needs `<MODEL ENV>_STRUCTURED_OUTPUT` set to `native` or
`prompted`. Otherwise it fails with an explicit `structured_output` error. That is
intended: an unverified model is not handed JSON work.

### Not affected by any override (still Gemini / specialist providers)

- Grounded generation from an indexed textbook: grounded PPT, grounded content and assessment papers (Gemini).
- Image understanding, OCR and vision doubts (Gemini, plus the existing Groq vision client).
- Odia-language notes and quizzes (Gemini).
- Transcription and translation (Whisper, Sarvam).
- Slide / note image generation (FLUX).

Those call paths do not go through `LLMClient`, so they are untouched. They will be
considered only after UI benchmarking.

## 6. Timeouts

- Together per-attempt timeout comes from the capability policy: reasoning 60 s, content 120 s.
  Override it with `AI_ROUTE_<CAPABILITY>_TIMEOUT_S`.
- NestJS applies its own timeout on top. Locally, `AI_TIMEOUT_MS` in `eddva_backend/.env.local`
  governs doubts, tutor and quiz. A slower model can be cut off there even though Django finishes.

## 7. Telemetry during a test

- Usage rows record the model as `together:<model id>`, so a Together call is distinguishable
  from Groq even when the model id is the same.
- `estimatedCost` is **null** for Together models: there is no configured price, and no price is invented.
- If Together does not report token usage, tokens are recorded as 0 and the log says
  `tokens=not-reported`.
- Provider failures and fallbacks appear as provider events with `provider=together`.

## 8. Turning it off

Remove the `AI_MODEL_OVERRIDE_*` lines, or set `AI_MODEL_OVERRIDE_ENABLED=false`, then restart.
`AI_ROUTER_ENABLED=false` bypasses the router entirely.

Cross-provider fallback (`AI_ROUTER_FALLBACK_ENABLED`) stays **off**. It will be enabled and
tested separately.
