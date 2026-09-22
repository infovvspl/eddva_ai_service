"""
Diagram rendering endpoint — NestJS ai-bridge.

POST /diagram/render  → draw one figure described in words, return it as a PNG

For the case textbook_figures cannot serve: a question whose diagram is invented
along with the question. "Show that A(1,1), B(4,1), C(4,5) are the vertices of a
right-angled triangle" needs a specific picture that no textbook prints, but the
question's own data determines it completely — so it is drawn rather than found.

Stateless, like the rest of this service: the image comes back in the response
and the NestJS backend owns storing it.

Metered under `content_generate`, the same feature as the paper it belongs to.
A paper and its figures billing to one feature keeps quota accounting coherent,
and reuses a feature key the tenant registry already knows — a made-up one would
be denied by tenant_gate.
"""
import logging

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from ai_services.core.diagram import render_diagram
from ai_services.core.usage_logger import log_usage

from .base import metered

logger = logging.getLogger("ai_services.diagram")


@api_view(["POST"])
@metered("content_generate")
def render(request):
    """
    POST /diagram/render

    Body: { spec, subjectName?, className?, board? }
    Returns: { success, data: { imageBase64, attempts } }
             or 422 with a reason when the figure could not be drawn.

    A failure here is ordinary, not exceptional: the caller drops the figure and
    keeps the paper. It is reported as 422 rather than 5xx so the backend can
    tell "this could not be drawn" from "the service is broken" — one means omit
    a figure, the other means stop.
    """
    spec = str(request.data.get("spec") or "").strip()
    if not spec:
        return Response({"error": "spec is required"}, status=status.HTTP_400_BAD_REQUEST)

    institute_id = getattr(request, "institute_id", None)
    vertical = getattr(request, "vertical", None) or "school"
    board = str(request.data.get("board") or getattr(request, "board", "") or "")

    result = render_diagram(
        spec,
        subject=str(request.data.get("subjectName") or ""),
        class_name=str(request.data.get("className") or ""),
        board=board,
    )

    log_usage(
        institute_id=institute_id, institute_type=vertical,
        feature_id="diagram_render", feature_category="content_generation",
        model_used="matplotlib_sandbox",
        success=bool(result.get("success")),
        error_message=None if result.get("success") else str(result.get("error") or "")[:300],
    )

    if not result.get("success"):
        logger.info("Diagram not drawn (%s): %s", spec[:70], str(result.get("error"))[:160])
        return Response(
            {"success": False, "error": result.get("error") or "could not draw this figure"},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    return Response({
        "success": True,
        "data": {
            "imageBase64": result["image_base64"],
            "attempts": result.get("attempts", 1),
        },
    })
