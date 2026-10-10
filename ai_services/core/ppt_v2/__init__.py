"""V2 PPT generation: content -> visual story -> composition -> renderer.

V1 goes topic -> content -> one search term -> first usable image -> deck. The
images are chosen reasonably well (ranked, filtered, relevance-scored), but every
slide lands in the same layout, so decks read as a wall of bullets beside a photo.

V2 separates three things V1 conflates:

    CONTENT   what the slide teaches      (schema.SlideSpec.content_blocks)
    VISUAL    what the learner should see (schema.VisualSpec + visuals.py)
    LAYOUT    where everything sits       (layouts.py + design.py)

Nothing in this package knows about PowerPoint, pptxgenjs or the browser.
Regions are normalised 0..1 rectangles and colours are tokens, so the same
specification can be rendered by the existing pptExport.js today and by a
server-side renderer later without changing the planner.

V1 remains the default; V2 is selected per request (see views/ppt.py).
"""
from ai_services.core.ppt_v2.compose import DeckContext, ComposedDeck, compose_deck
from ai_services.core.ppt_v2.design import build_theme
from ai_services.core.ppt_v2.flags import V1, V2, ai_image_budget, ppt_version
from ai_services.core.ppt_v2.layouts import LAYOUTS, select_layout
from ai_services.core.ppt_v2.producers import ProducerSet
from ai_services.core.ppt_v2.qa import check_deck, report
from ai_services.core.ppt_v2.schema import (
    SPEC_VERSION,
    Annotation,
    ContentBlock,
    LayoutSpec,
    SlideSpec,
    ThemeSpec,
    TitleSpec,
    VisualSpec,
    VisualStrategy,
    validate_deck,
)
from ai_services.core.ppt_v2.visuals import (
    DeckVisualPlan,
    VisualCapabilities,
    detect_capabilities,
    resolve_visual,
)

__all__ = [
    "LAYOUTS",
    "SPEC_VERSION",
    "V1",
    "V2",
    "ComposedDeck",
    "DeckContext",
    "DeckVisualPlan",
    "ProducerSet",
    "VisualCapabilities",
    "ai_image_budget",
    "build_theme",
    "check_deck",
    "compose_deck",
    "detect_capabilities",
    "ppt_version",
    "report",
    "resolve_visual",
    "select_layout",
    "Annotation",
    "ContentBlock",
    "LayoutSpec",
    "SlideSpec",
    "ThemeSpec",
    "TitleSpec",
    "VisualSpec",
    "VisualStrategy",
    "validate_deck",
]
