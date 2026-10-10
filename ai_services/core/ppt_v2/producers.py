"""Adapters from a visual strategy to the service that actually makes the asset.

Each service already exists and already works; none of them is rewritten here.
The job of this module is only to give them one shape - the ``Producer``
contract in :mod:`ai_services.core.ppt_v2.visuals` - so the fallback chain can
walk them without knowing what any of them is:

    AI_GENERATED_IMAGE     core.image_generation.generate_note_image
    PROGRAMMATIC_DIAGRAM   core.diagram.render_diagram
    SEARCHED_IMAGE         views.ppt._fetch_image_for_slide, injected
    CACHED_ASSET           a lookup the caller supplies
    ICON_COMPOSITION       nothing - the renderer draws shapes

The image search is **injected rather than imported**: it lives in
``views/ppt.py``, and a ``core`` module importing a view is the kind of cycle
that works until the day someone imports this from a management command. The
caller passes it in; this module stays importable on its own.

Retries: none are added here. ``generate_note_image`` already retries a cold
model once and reports ``fatal`` when retrying is pointless, and the fallback
chain tries each strategy exactly once. That is the whole retry budget, on
purpose - an image pipeline that retries at three levels is how a 40-second
deck becomes a timeout.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from ai_services.core.ppt_v2 import labels
from ai_services.core.ppt_v2.schema import VisualSpec, VisualStrategy

logger = logging.getLogger(__name__)

# Errors that mean "this will fail the same way next time". Recorded on the
# spec so a deck does not burn its remaining slides on a provider that is
# simply not configured.
_FATAL = "fatal"
# Generating a picture and then checking it takes 15-40s.
AI_IMAGE_MIN_REMAINING_S = 45.0
# One slide's picture may take this long, whatever the deck has left. A live
# deck's last picture took ~80s (generate, check, redraw, check, search, check)
# and held the whole deck's Download button for it.
SLIDE_PICTURE_BUDGET_S = 90.0
# Time a vision check must leave for the rest of the deck to finish.
VISION_MARGIN_S = 10.0


def _guided_prompt(prompt: str, verdict) -> str:
    """The prompt again, with what the vision check found missing or wrong.

    Only when somebody actually looked and named something; a check that did
    not happen gives nothing to steer by, and a redraw would be a coin toss.
    """
    if verdict is None or not getattr(verdict, "available", False):
        return ""
    missing = [m for m in (getattr(verdict, "missing", None) or []) if m][:4]
    wrong = [c for c in (getattr(verdict, "conflicts", None) or []) if c][:4]
    if not (missing or wrong):
        return ""
    out = prompt.rstrip(". ")
    if missing:
        out += ". The picture MUST clearly show: " + "; ".join(missing)
    if wrong:
        out += ". It must NOT show: " + "; ".join(wrong)
    return out + "."


class ProducerSet:
    """The producers for one deck, sharing per-deck state.

    Shared state matters: once the HuggingFace token turns out to be missing,
    every later slide should skip straight to the next strategy instead of
    making the same doomed call ten more times.
    """

    def __init__(self, subject: str = "", class_name: str = "", board: str = "",
                 search_fn: Optional[Callable[..., Dict[str, Any]]] = None,
                 cache_get: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
                 cache_put: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 search_ctx: Optional[Dict[str, Any]] = None,
                 topic: str = "", chapter: str = "",
                 vision_ask=None, vision_model: str = "",
                 on_vision_usage=None):
        self.subject = subject
        self.class_name = class_name
        self.board = board
        self.topic = topic
        self.chapter = chapter
        self._vision_ask = vision_ask
        self._vision_model = vision_model
        self._on_vision_usage = on_vision_usage
        self.vision_verdicts: list = []
        # Per generated image: what the model drew, what we erased, what the
        # renderer will put back. Diagnostic only; see _record_image_diagnostics.
        self.image_diagnostics: list = []
        self._search_fn = search_fn
        self._cache_get = cache_get
        self._cache_put = cache_put
        self._search_ctx = search_ctx or {}
        self._disabled: set = set()
        # perf_counter() time the whole deck must be done by; None = no limit.
        # Set by the service. Optional slow steps check it and give way.
        self.deadline: Optional[float] = None
        # Per-slide deadline, per thread: each slide resolves on its own.
        self._slide = __import__("threading").local()
        # on_event(key, text, state): the activity feed. Set by the service.
        self.on_event: Optional[Callable[..., None]] = None
        self.attempts: list = []
        # Assets already placed in this deck. Two slides on one topic produce
        # near-identical search queries, and the first live run put the exact
        # same photograph on both slides of a two-slide deck - which reads as
        # a bug to anyone looking at it. Shared across threads because asset
        # resolution is concurrent.
        self._used: set = set()
        self._lock = __import__("threading").Lock()

    def _claim(self, identity: str) -> bool:
        """Reserve an asset for one slide. False if another slide has it."""
        if not identity:
            return True
        with self._lock:
            if identity in self._used:
                return False
            self._used.add(identity)
            return True

    # -- individual producers ------------------------------------------------

    def begin_slide(self, number: Optional[int] = None) -> None:
        """Start this thread's slide clock: SLIDE_PICTURE_BUDGET_S, within the deck's."""
        limit = time.perf_counter() + SLIDE_PICTURE_BUDGET_S
        self._slide.deadline = limit if self.deadline is None else min(limit, self.deadline)
        self._slide.number = number

    def note(self, text: str, state: str = "active") -> None:
        """A line in the activity feed for the slide this thread is working on."""
        number = getattr(self._slide, "number", None)
        if not self.on_event or number is None:
            return
        try:
            self.on_event("picture-%d" % number, "Slide %d: %s" % (number, text), state)
        except Exception:
            logger.exception("ppt_v2: activity event failed")

    def end_slide(self) -> None:
        self._slide.deadline = None

    def _remaining(self) -> Optional[float]:
        deadline = getattr(self._slide, "deadline", None) or self.deadline
        return None if deadline is None else deadline - time.perf_counter()

    def _vision_timeout(self) -> Optional[float]:
        """How long one vision check may take now, leaving a margin to finish."""
        left = self._remaining()
        return None if left is None else max(0.0, left - VISION_MARGIN_S)

    def ai_image(self, spec: VisualSpec) -> Dict[str, Any]:
        if VisualStrategy.AI_GENERATED_IMAGE in self._disabled:
            return {"ok": False, "error": "disabled earlier in this deck"}
        left = self._remaining()
        if left is not None and left < AI_IMAGE_MIN_REMAINING_S:
            # Generating, then checking, a picture takes most of a minute. With
            # less than that left the slide takes the faster route (a searched
            # figure) instead of making the whole deck late.
            return {"ok": False, "error": "no time left for a generated image (%.0fs)" % left}
        prompt = (spec.image_prompt or spec.description or "").strip()
        if not prompt:
            return {"ok": False, "error": "no prompt"}

        from ai_services.core.image_generation import generate_note_image
        attempt_prompt = prompt
        for attempt in (1, 2):
            self.note("generating an illustration" if attempt == 1
                      else "redrawing the illustration with the missing details")
            result = generate_note_image(attempt_prompt) or {}
            if not result.get("ok"):
                break
            asset = {"ok": True, "url": result.get("url") or "",
                     "source": result.get("provider") or "huggingface",
                     "model": result.get("model") or "",
                     # Prompted with the scene and OCR-stripped: the renderer
                     # owns the labels and may point at the planned anchors.
                     "labelPolicy": (labels.ANCHORED if spec.annotations
                                     else labels.NO_LABELS)}
            self._record_image_diagnostics(spec, result)
            rejection = self._reject_generated(spec, result)
            if rejection is None:
                self._remember(spec, asset)
                return asset
            # Not remembered and not returned. Caching it would hand the same
            # rejected picture back through CACHED_ASSET later, and returning
            # it would put a picture that teaches the wrong thing on the slide.
            verdict = rejection.pop("_verdict", None)
            guided = _guided_prompt(prompt, verdict)
            left = self._remaining()
            if (attempt == 1 and guided
                    and (left is None or left >= AI_IMAGE_MIN_REMAINING_S)):
                # One redraw, told exactly what the look found missing. A live
                # 10-slide deck lost two of its three generated pictures to
                # specific, fixable misses ("arrow from pyruvate to lactic
                # acid") and the slides fell back to icon rows.
                logger.info("ppt_v2: redrawing once against the vision findings")
                attempt_prompt = guided
                continue
            return rejection

        if result.get(_FATAL):
            # Not configured, or the model does not exist. Every later slide
            # would fail identically; stop asking.
            logger.info("ppt_v2: AI images disabled for this deck: %s",
                        result.get("error"))
            self._disabled.add(VisualStrategy.AI_GENERATED_IMAGE)
        return {"ok": False, "error": str(result.get("error") or "generation failed")}

    def _reject_generated(self, spec: VisualSpec,
                          result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check a generated image against the slide's visual contract.

        Returns None to accept, or the failure dict the fallback chain needs.

        This is the same machinery the search path uses - the same contract,
        the same vision call, the same class rules - and not a second
        validation framework. What differs is only the trigger.

        A searched image arrives with a caption, so the deterministic scorer
        can place it in a confidence band and a HIGH band buys its way out of
        the call. A generated image has no caption: the only text describing
        it is the prompt we wrote ourselves, and scoring our own prompt would
        return HIGH every time. So the band cannot be the gate here, and the
        contract is: a look is worth paying for when there is something
        concrete to check AND a real chance of getting it wrong - a chapter
        with known class-level risks, a relationship the picture has to show,
        or concepts it has to stay away from. A plain "draw a leaf" contract
        gets no call.

        One look, never a regeneration. A second image costs provider money
        and a second vision call to judge it; the fallback chain already has
        four cheaper ways to fill the region.

        **Fail-closed.** Once validation is required, "nobody successfully
        looked" is a rejection, not a pass. This is the one place the AI path
        deliberately diverges from the search path, and the first live run is
        why: FLUX returned an image, the vision call ran for nine seconds,
        the reply did not parse, the verdict came back `available=False` and
        the picture went onto a Class 8 slide unexamined. A searched image
        has provenance to fall back on - a caption, a host, a deterministic
        score - so treating "nobody looked" as neutral there is defensible.
        Freshly generated pixels have none of that. If the check that was
        required could not be carried out, the asset does not ship.
        """
        from ai_services.core.ppt_v2 import vision

        contract = self.visual_contract(spec)
        if not self._validation_required(contract):
            logger.info("ppt_v2: generated image accepted without a vision call "
                        "(no class-level risk, nothing to verify beyond objects)")
            return None

        if not vision.enabled():
            return self._uninspected(spec, "vision validation is disabled")
        if not self._vision_ask:
            return self._uninspected(spec, "no vision provider supplied")

        self.note("checking the illustration is accurate for the lesson")
        verdict = vision.validate(
            self._vision_reference(result), contract,
            ask=self._vision_ask, cache_get=self._cache_get,
            cache_put=self._cache_put, model=self._vision_model,
            on_usage=self._on_vision_usage, timeout=self._vision_timeout())
        self.vision_verdicts.append(verdict)

        if not verdict.available:
            # Unparseable, timed out, no provider, no image: all the same
            # thing from here - the required inspection did not happen.
            return self._uninspected(spec, verdict.reason or "no verdict")
        if not verdict.passed:
            logger.info("ppt_v2: vision rejected the generated image (%s)",
                        verdict.reason)
            # The verdict rides along so the caller can redraw against exactly
            # what was missing; slides are checked concurrently, so a shared
            # "last verdict" could belong to another slide.
            return {"ok": False, "error": "vision rejected: %s" % verdict.reason,
                    "_verdict": verdict}
        return None

    def _record_image_diagnostics(self, spec: VisualSpec,
                                  result: Dict[str, Any]) -> None:
        """Three separate numbers about one generated picture.

        The first live run could not tell them apart, and they answer
        different questions:

          generated  - did the model draw text into the image at all?
                       A prompt problem, fixed by prompting.
          removed    - how many regions did we paint out?
                       An image-damage problem: every one leaves a white box.
          overlaid   - how many real text labels will the renderer place
                       on top? This is the repair, and it is what makes
                       erasing the generated text survivable.

        Recorded, never acted on: nothing here changes a decision.
        """
        generated = result.get("embedded_text_detected")
        entry = {
            "slide_prompt": (spec.image_prompt or "")[:80],
            "embedded_text_generated": generated,
            "embedded_text_removed": int(result.get("embedded_text_removed") or 0),
            "embedded_text_samples": list(result.get("embedded_text_samples") or []),
            "strip_error": result.get("embedded_text_strip_error"),
            "annotations_overlaid": len(spec.annotations or []),
        }
        with self._lock:
            self.image_diagnostics.append(entry)
        logger.info("ppt_v2: generated image text | generated=%s removed=%d "
                    "annotations_overlaid=%d%s",
                    "unmeasured" if generated is None else generated,
                    entry["embedded_text_removed"], entry["annotations_overlaid"],
                    " | NOTHING WILL REPLACE THE ERASED TEXT"
                    if entry["embedded_text_removed"] and not entry["annotations_overlaid"]
                    else "")

    def _validation_required(self, contract) -> bool:
        """Does this slide's picture have to be inspected before it is used?

        Required when there is something concrete to check AND a real chance
        of getting it wrong: a chapter carrying known class-level risks, a
        relationship the picture has to show, or concepts it has to stay away
        from. A plain "draw a leaf" contract is not worth a call, and a
        generated image that satisfies one is not worth rejecting either.
        """
        from ai_services.core.ppt_v2 import class_rules

        if not contract.is_meaningful:
            return False
        at_risk = bool(class_rules.all_forbidden(
            self.class_name, self.subject, self.chapter))
        return bool(at_risk or contract.relationships or contract.avoid)

    @staticmethod
    def _uninspected(spec: VisualSpec, reason: str) -> Dict[str, Any]:
        logger.info("ppt_v2: generated image not used - the required check "
                    "could not be made (%s)", reason)
        return {"ok": False,
                "error": "vision could not inspect the generated image: %s" % reason}

    @staticmethod
    def _vision_reference(result: Dict[str, Any]) -> str:
        """What to hand the vision helper for a generated image.

        The saved file, inlined, rather than the public URL. That URL points
        at this service's own static route, which is reachable from a
        developer's laptop and not necessarily from wherever the vision
        helper runs - and the bytes are already on disk, so there is no
        reason to make a round trip to ourselves to get them back.

        Encoded exactly as the URL path encodes a searched image - RGB, at
        most 1024px, JPEG quality 80 - so the two routes to the same vision
        model differ in nothing but where the bytes came from. The raw PNG
        was 3-5x larger for the same pixels. (That was not why the first two
        live runs came back unparseable - the answer was being truncated by
        the output budget, fixed in views.bridge - but a difference between
        the path that works and the path that did not is a variable worth
        removing.)
        """
        path = str(result.get("path") or "")
        if path:
            try:
                return _jpeg_data_uri(path)
            except Exception as exc:
                logger.debug("ppt_v2: could not inline the generated image: %s", exc)
        return str(result.get("url") or "")

    def diagram(self, spec: VisualSpec) -> Dict[str, Any]:
        source = (spec.diagram_spec or spec.description or "").strip()
        if not source:
            return {"ok": False, "error": "no diagram specification"}
        # A figure we draw labels its own parts, because only the drawing
        # code knows where they are. It used to be told NOT to label itself
        # whenever the slide had annotations, on the promise the renderer
        # would - and the renderer only ever labelled generated images, so a
        # rejected AI image fell back to a diagram nobody labelled. Now the
        # required terms travel into the specification, verbatim.
        required = [a.text.strip() for a in spec.annotations if (a.text or "").strip()]
        if required:
            source = "%s\nLabel these parts on the figure, " "using exactly these words: %s" % (
                source, "; ".join(required))

        from ai_services.core.diagram import render_diagram
        # The region size is what makes the figure legible: drawn at exactly
        # the size it will occupy, the point sizes in the generated code are
        # the point sizes the class sees.
        w, h = spec.region_in or (0.0, 0.0)
        # Axes only for a chart. A labelled-structure figure is a drawing,
        # and a benchmark slide showed matplotlib putting 0-2-4-6-8-10 tick
        # labels around a plant cell.
        wants_axes = bool(spec.search_terms and any(
            k in " ".join(spec.search_terms).lower()
            for k in ("graph", "chart", "plot", "axis", "versus")))
        result = render_diagram(source, subject=self.subject,
                                class_name=self.class_name, board=self.board,
                                width_in=w or None, height_in=h or None,
                                axes=wants_axes,
                                # Always self-labelled. Double printing is
                                # prevented by the label policy instead:
                                # NATIVE tells the renderer to draw nothing.
                                labels=True) or {}
        if result.get("success") and result.get("image_base64"):
            # render_diagram returns a data URI, which the renderer places the
            # same way it places a downloaded image.
            asset = {"ok": True, "base64": result["image_base64"],
                     "source": "matplotlib", "labelPolicy": labels.NATIVE}
            self._remember(spec, asset)
            return asset
        return {"ok": False, "error": str(result.get("error") or "diagram failed")}

    def _search_once(self, spec: VisualSpec) -> Dict[str, Any]:
        if self._search_fn is None:
            return {"ok": False, "error": "no search function supplied"}
        duplicate: Optional[Dict[str, Any]] = None

        for term in (spec.search_terms or [spec.description]):
            term = (term or "").strip()
            if not term:
                continue
            try:
                found = self._search_fn(term, term, self._search_ctx,
                                        self._scorer(spec)) or {}
            except TypeError:
                # A search function without the scorer parameter (V1's own
                # signature) still works; it just filters one-sidedly.
                try:
                    found = self._search_fn(term, term, self._search_ctx) or {}
                except TypeError:
                    found = self._search_fn(term, term) or {}
            url = found.get("imageUrl") or ""
            if not (url or found.get("imageBase64")):
                continue

            asset = {"ok": True, "url": url,
                     "base64": found.get("imageBase64") or "",
                     "source": "image_search",
                     "relevanceScore": found.get("relevanceScore"),
                     "relevanceConfidence": found.get("relevanceConfidence") or "",
                     # Nobody composed this picture to our anchors: the
                     # required terms go in as a key, never as arrows.
                     "labelPolicy": (labels.LEGEND if spec.annotations
                                     else labels.NO_LABELS)}
            if self._claim(url):
                self._remember(spec, asset)
                return asset
            # Another slide already has this exact picture. Keep looking at
            # the broader queries, but hold on to it: if nothing unique turns
            # up, an on-topic image shown twice still teaches more than a
            # generic shape composition would.
            logger.info("ppt_v2: %r duplicates an image already in this deck", term)
            duplicate = duplicate or asset

        if duplicate is not None:
            logger.info("ppt_v2: no unique image found; reusing one already in the deck")
            return duplicate
        return {"ok": False, "error": "no relevant image found"}

    def search(self, spec: VisualSpec) -> Dict[str, Any]:
        """A searched image, checked once by a vision model where that helps.

        Deterministic scoring reads captions and hosts, which catches the
        picture about the wrong thing. It cannot catch the picture about the
        right thing that still teaches nothing - a "What is Light?" slide was
        given a fluorescent lamp starter schematic: on topic, reputable host,
        useless for the objective.

        So the chosen candidate gets one look, at most twice per slide. A
        rejected candidate is marked used so the next search skips it. If
        nothing passes, this reports failure and the fallback chain tries a
        different kind of visual - which is the point: an unrelated picture
        is worse than a drawn figure.
        """
        from ai_services.core.ppt_v2 import class_rules, relevance, vision

        contract = self.visual_contract(spec)
        budget = (vision.MAX_VALIDATIONS_PER_SLIDE
                  if (vision.enabled() and self._vision_ask
                      and contract.is_meaningful) else 0)

        last = {"ok": False, "error": "no relevant image found"}
        for attempt in range(max(budget, 1)):
            self.note("searching for a matching textbook figure" if attempt == 0
                      else "searching for a better figure")
            found = self._search_once(spec)
            if not found.get("ok"):
                return found
            if not budget:
                return found

            # The policy: a vision call buys nothing on a candidate the
            # deterministic score already placed in the HIGH band, and a LOW
            # one was rejected before it got here. Only the middle is worth
            # paying for.
            score = found.get("relevanceScore")
            # HIGH confidence means "on topic", not "at the right level".
            # The deterministic layer reads captions and hosts and cannot see
            # a mu_s*N on the figure, which is how a Class 11 friction graph
            # reached a Class 8 slide with a good caption and a reputable
            # host. So when the chapter has known class-level risks, the
            # chosen image is looked at whatever its confidence.
            at_risk = bool(class_rules.all_forbidden(
                self.class_name, self.subject, self.chapter))
            if (score is not None and not relevance.needs_vision(score)
                    and not at_risk):
                logger.info("ppt_v2: %s confidence, no vision call needed",
                            found.get("relevanceConfidence") or "high")
                return found
            if at_risk and score is not None and not relevance.needs_vision(score):
                logger.info("ppt_v2: high confidence but %s has class-level "
                            "risks; checking the image", self.chapter or "chapter")

            self.note("checking the figure fits the lesson and the class")
            verdict = vision.validate(
                found.get("url") or found.get("base64") or "", contract,
                ask=self._vision_ask, cache_get=self._cache_get,
                cache_put=self._cache_put, model=self._vision_model,
                on_usage=self._on_vision_usage, timeout=self._vision_timeout())
            self.vision_verdicts.append(verdict)

            # "Not available" is not a failure: it means nobody looked, so
            # the deterministic decision stands.
            if verdict.passed or not verdict.available:
                return found

            logger.info("ppt_v2: vision rejected %r for slide (%s)",
                        (found.get("url") or "")[:80], verdict.reason)
            self._claim(found.get("url") or "")   # skip it on the next pass
            last = {"ok": False,
                    "error": "vision rejected: %s" % verdict.reason}
        return last

    def visual_contract(self, spec: VisualSpec):
        """The teaching contract for this slide's picture."""
        from ai_services.core.ppt_v2 import vision
        return vision.build_contract(
            learning_objective=spec.objective or spec.description,
            objects=list(spec.icons), must_show=list(spec.must_show),
            relationships=list(spec.relationships),
            avoid=list(spec.avoid_concepts),
            visual_kind=spec.strategy.value if spec.strategy else "",
            # The audience. A figure can be exactly on topic and still wrong
            # for the class, which is how a Class 11 friction graph reached a
            # Class 8 slide past every text-based check.
            class_name=self.class_name, board=self.board,
            subject=self.subject, chapter=self.chapter)

    def _scorer(self, spec: VisualSpec):
        """A caption scorer for this slide's picture.

        Built per slide because `required` and `avoid` are per slide. The
        search loop calls it for every candidate caption, which costs nothing
        beyond string matching - no model, no extra request.
        """
        from ai_services.core.ppt_v2 import relevance

        rel = relevance.build_spec(
            topic=self.topic or spec.description,
            required=list(spec.must_show) + list(spec.icons),
            avoid=list(spec.avoid_concepts),
            visual_kind=spec.strategy.value if spec.strategy else "",
            class_name=self.class_name,
        )
        self.relevance_spec = rel
        return lambda caption, url="": relevance.score_candidate(caption, rel, url)

    def cached(self, spec: VisualSpec) -> Dict[str, Any]:
        if self._cache_get is None:
            return {"ok": False, "error": "no cache configured"}
        key = cache_key(spec)
        try:
            hit = self._cache_get(key)
        except Exception as exc:
            logger.debug("ppt_v2: asset cache read failed: %s", exc)
            return {"ok": False, "error": "cache unavailable"}
        if hit and (hit.get("url") or hit.get("base64")):
            return {"ok": True, "url": hit.get("url") or "",
                    "base64": hit.get("base64") or "",
                    "source": hit.get("source") or "cache",
                    # Entries cached before policies existed have none; we do
                    # not know their composition, so never point into them.
                    "labelPolicy": hit.get("labelPolicy") or (
                        labels.LEGEND if spec.annotations else labels.NO_LABELS)}
        return {"ok": False, "error": "cache miss"}

    def icons(self, spec: VisualSpec) -> Dict[str, Any]:
        # Nothing to fetch: the renderer composes shapes and glyphs from the
        # slide's own icon names. This is the strategy that cannot fail, and
        # it only succeeds if there is something to draw.
        if spec.icons or spec.must_show or spec.description:
            return {"ok": True, "source": "shapes", "labelPolicy": labels.NATIVE}
        return {"ok": False, "error": "nothing to compose"}

    # -- plumbing ------------------------------------------------------------

    def _remember(self, spec: VisualSpec, asset: Dict[str, Any]) -> None:
        if self._cache_put is None:
            return
        try:
            self._cache_put(cache_key(spec), asset)
        except Exception as exc:
            logger.debug("ppt_v2: asset cache write failed: %s", exc)

    def as_map(self) -> Dict[VisualStrategy, Callable[[VisualSpec], Dict[str, Any]]]:
        return {
            VisualStrategy.AI_GENERATED_IMAGE: self.ai_image,
            VisualStrategy.PROGRAMMATIC_DIAGRAM: self.diagram,
            VisualStrategy.SEARCHED_IMAGE: self.search,
            VisualStrategy.CACHED_ASSET: self.cached,
            VisualStrategy.ICON_COMPOSITION: self.icons,
        }

    def record(self, slide_number: int, strategy: VisualStrategy,
               ok: bool, detail: str = "") -> None:
        """Callback for ``resolve_visual``, so a deck can explain itself."""
        self.attempts.append({"slide": slide_number, "strategy": strategy.value,
                              "ok": ok, "detail": detail})


def _jpeg_data_uri(path: str, max_side: int = 1024, quality: int = 80) -> str:
    """A local image as the data URI the vision helper's URL path produces."""
    import base64
    from io import BytesIO

    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_side:
            ratio = max_side / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)),
                             Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64,%s" % base64.b64encode(buffer.getvalue()).decode("ascii")


def cache_key(spec: VisualSpec) -> str:
    """Stable key for an asset. The prompt is the identity of the image."""
    import hashlib
    basis = "|".join([
        spec.image_prompt or "",
        spec.diagram_spec or "",
        ",".join(spec.search_terms or []),
        spec.description or "",
    ])
    return "ppt_v2:asset:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]
