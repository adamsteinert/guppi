"""Model listing endpoint matching ElevenLabs API."""

from fastapi import APIRouter, Request

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(request: Request):
    """List all available TTS models."""
    engine = request.app.state.engine
    return engine.get_model_list()
