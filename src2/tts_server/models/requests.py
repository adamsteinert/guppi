"""Pydantic request models matching the ElevenLabs API."""

from typing import Optional

from pydantic import BaseModel, Field


class VoiceSettings(BaseModel):
    """Voice settings for TTS synthesis."""

    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)
    use_speaker_boost: bool = True
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class TTSRequest(BaseModel):
    """Request body for /v1/text-to-speech/{voice_id}."""

    text: str = Field(..., min_length=1, max_length=10000)
    model_id: Optional[str] = "eleven_multilingual_v2"
    voice_settings: Optional[VoiceSettings] = None
    language_code: Optional[str] = None
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295)
    previous_text: Optional[str] = None
    next_text: Optional[str] = None
    apply_text_normalization: Optional[str] = "auto"


class WSInitMessage(BaseModel):
    """WebSocket initialization message."""

    text: str = " "
    voice_settings: Optional[VoiceSettings] = None
    xi_api_key: Optional[str] = None


class WSTextChunk(BaseModel):
    """WebSocket text chunk message."""

    text: str
    try_trigger_generation: bool = True
