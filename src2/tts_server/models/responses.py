"""Pydantic response models matching the ElevenLabs API."""

from typing import Optional

from pydantic import BaseModel


class VoiceSettingsResponse(BaseModel):
    """Default voice settings response."""

    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    speed: float = 1.0


class VoiceResponse(BaseModel):
    """Voice object matching ElevenLabs voice schema."""

    voice_id: str
    name: str
    category: str = "premade"
    labels: dict = {}
    description: Optional[str] = None
    preview_url: Optional[str] = None
    settings: Optional[VoiceSettingsResponse] = None
    high_quality_base_model_ids: list[str] = []
    # Extra metadata (not in ElevenLabs but useful)
    sample_rate: int = 24000
    engine: str = "kokoro"


class ModelLanguage(BaseModel):
    """Language supported by a model."""

    language_id: str
    name: str


class ModelResponse(BaseModel):
    """Model object matching ElevenLabs model schema."""

    model_id: str
    name: str
    description: str = ""
    can_be_finetuned: bool = False
    can_do_text_to_speech: bool = True
    can_do_voice_conversion: bool = False
    can_use_style: bool = False
    can_use_speaker_boost: bool = False
    serves_pro_voices: bool = False
    token_cost_factor: float = 1.0
    max_characters_request_free_user: int = 5000
    max_characters_request_subscribed_user: int = 10000
    languages: list[ModelLanguage] = []
