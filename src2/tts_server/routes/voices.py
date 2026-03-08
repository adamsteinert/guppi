"""Voice listing and detail endpoints matching ElevenLabs API."""

from fastapi import APIRouter, HTTPException, Request

from ..models.responses import VoiceSettingsResponse

router = APIRouter(tags=["voices"])


@router.get("/voices")
async def list_voices(request: Request):
    """List all available voices."""
    engine = request.app.state.engine
    voices = engine.get_voice_list()
    return {"voices": [v.model_dump() for v in voices]}


@router.get("/voices/settings/default")
async def default_voice_settings():
    """Get the default voice settings."""
    return VoiceSettingsResponse().model_dump()


@router.get("/voices/{voice_id}")
async def get_voice(voice_id: str, request: Request):
    """Get details for a specific voice."""
    engine = request.app.state.engine
    voice = engine.get_voice_detail(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    return voice.model_dump()


@router.get("/voices/{voice_id}/settings")
async def get_voice_settings(voice_id: str, request: Request):
    """Get settings for a specific voice."""
    engine = request.app.state.engine
    voice = engine.get_voice_detail(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    return VoiceSettingsResponse().model_dump()
