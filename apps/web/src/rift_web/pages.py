"""Template rendering helper: every page gets the current user for the top bar."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from rift_web.auth import current_user


def page(
    request: Request, name: str, context: dict[str, Any] | None = None, *, status: int = 200
) -> Response:
    templates: Jinja2Templates = request.app.state.templates
    ctx = {"user": current_user(request), **(context or {})}
    return templates.TemplateResponse(request, name, ctx, status_code=status)
