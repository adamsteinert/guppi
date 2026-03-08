"""Optional API key authentication middleware."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class OptionalAPIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that optionally enforces xi-api-key header authentication."""

    async def dispatch(self, request: Request, call_next):
        config = getattr(request.app.state, "config", None)

        # Skip auth if not configured
        if not config or not config.api_key_required or not config.api_key:
            return await call_next(request)

        # Skip for docs/health endpoints
        if request.url.path in ("/docs", "/openapi.json", "/redoc", "/health"):
            return await call_next(request)

        # Check xi-api-key header
        api_key = request.headers.get("xi-api-key")
        if api_key != config.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}},
            )

        return await call_next(request)
