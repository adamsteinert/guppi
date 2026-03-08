"""FastAPI application factory for the ElevenLabs-compatible TTS server."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .auth import OptionalAPIKeyMiddleware
from .config import ServerConfig, load_config
from .engine.tts_engine import TTSEngine
from .routes import models_route, tts, user, voices, websocket


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage TTS engine lifecycle."""
    config: ServerConfig = app.state.config
    logger.info(f"Loading TTS models from {config.model_dir}...")

    engine = TTSEngine(
        model_dir=config.model_dir,
        default_voice=config.default_voice,
    )
    app.state.engine = engine
    logger.info("TTS server ready")

    yield

    logger.info("Shutting down TTS server")


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="GLaDOS TTS API",
        description="ElevenLabs-compatible TTS API powered by GLaDOS and Kokoro voices",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.config = config

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API key auth
    app.add_middleware(OptionalAPIKeyMiddleware)

    # Routes
    app.include_router(tts.router, prefix="/v1")
    app.include_router(voices.router, prefix="/v1")
    app.include_router(models_route.router, prefix="/v1")
    app.include_router(websocket.router, prefix="/v1")
    app.include_router(user.router, prefix="/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
