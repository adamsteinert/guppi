"""Voice ID resolution, aliases, and metadata for ElevenLabs compatibility."""

from typing import Optional

from loguru import logger

from ..models.responses import VoiceResponse, VoiceSettingsResponse

# Metadata for known voices
VOICE_METADATA: dict[str, dict] = {
    "glados": {
        "name": "GLaDOS",
        "engine": "glados",
        "category": "premade",
        "description": "The iconic GLaDOS voice from Portal - robotic and sardonic",
        "labels": {"accent": "robotic", "gender": "female", "age": "synthetic"},
        "sample_rate": 22050,
    },
    "af_alloy": {
        "name": "Alloy",
        "engine": "kokoro",
        "category": "premade",
        "description": "Clear, versatile female voice",
        "labels": {"accent": "american", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "af_aoede": {
        "name": "Aoede",
        "engine": "kokoro",
        "category": "premade",
        "description": "Warm female voice",
        "labels": {"accent": "american", "gender": "female", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "af_bella": {
        "name": "Bella",
        "engine": "kokoro",
        "category": "premade",
        "description": "Soft, friendly female voice",
        "labels": {"accent": "american", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "af_jessica": {
        "name": "Jessica",
        "engine": "kokoro",
        "category": "premade",
        "description": "Professional female voice",
        "labels": {"accent": "american", "gender": "female", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "af_kore": {
        "name": "Kore",
        "engine": "kokoro",
        "category": "premade",
        "description": "Clear female voice",
        "labels": {"accent": "american", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "af_nicole": {
        "name": "Nicole",
        "engine": "kokoro",
        "category": "premade",
        "description": "Warm female voice",
        "labels": {"accent": "american", "gender": "female", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "af_nova": {
        "name": "Nova",
        "engine": "kokoro",
        "category": "premade",
        "description": "Bright, energetic female voice",
        "labels": {"accent": "american", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "af_river": {
        "name": "River",
        "engine": "kokoro",
        "category": "premade",
        "description": "Smooth, flowing female voice",
        "labels": {"accent": "american", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "af_sarah": {
        "name": "Sarah",
        "engine": "kokoro",
        "category": "premade",
        "description": "Friendly female voice",
        "labels": {"accent": "american", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "af_sky": {
        "name": "Sky",
        "engine": "kokoro",
        "category": "premade",
        "description": "Light, airy female voice",
        "labels": {"accent": "american", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "am_adam": {
        "name": "Adam",
        "engine": "kokoro",
        "category": "premade",
        "description": "Clear male voice",
        "labels": {"accent": "american", "gender": "male", "age": "young"},
        "sample_rate": 24000,
    },
    "am_echo": {
        "name": "Echo",
        "engine": "kokoro",
        "category": "premade",
        "description": "Resonant male voice",
        "labels": {"accent": "american", "gender": "male", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "am_eric": {
        "name": "Eric",
        "engine": "kokoro",
        "category": "premade",
        "description": "Professional male voice",
        "labels": {"accent": "american", "gender": "male", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "am_fenrir": {
        "name": "Fenrir",
        "engine": "kokoro",
        "category": "premade",
        "description": "Bold male voice",
        "labels": {"accent": "american", "gender": "male", "age": "young"},
        "sample_rate": 24000,
    },
    "am_liam": {
        "name": "Liam",
        "engine": "kokoro",
        "category": "premade",
        "description": "Warm male voice",
        "labels": {"accent": "american", "gender": "male", "age": "young"},
        "sample_rate": 24000,
    },
    "am_michael": {
        "name": "Michael",
        "engine": "kokoro",
        "category": "premade",
        "description": "Deep, authoritative male voice",
        "labels": {"accent": "american", "gender": "male", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "am_onyx": {
        "name": "Onyx",
        "engine": "kokoro",
        "category": "premade",
        "description": "Deep, rich male voice",
        "labels": {"accent": "american", "gender": "male", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "am_puck": {
        "name": "Puck",
        "engine": "kokoro",
        "category": "premade",
        "description": "Playful male voice",
        "labels": {"accent": "american", "gender": "male", "age": "young"},
        "sample_rate": 24000,
    },
    "bf_alice": {
        "name": "Alice",
        "engine": "kokoro",
        "category": "premade",
        "description": "British female voice",
        "labels": {"accent": "british", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "bf_emma": {
        "name": "Emma",
        "engine": "kokoro",
        "category": "premade",
        "description": "British female voice",
        "labels": {"accent": "british", "gender": "female", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "bf_isabella": {
        "name": "Isabella",
        "engine": "kokoro",
        "category": "premade",
        "description": "Elegant British female voice",
        "labels": {"accent": "british", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "bf_lily": {
        "name": "Lily",
        "engine": "kokoro",
        "category": "premade",
        "description": "Gentle British female voice",
        "labels": {"accent": "british", "gender": "female", "age": "young"},
        "sample_rate": 24000,
    },
    "bm_daniel": {
        "name": "Daniel",
        "engine": "kokoro",
        "category": "premade",
        "description": "British male voice",
        "labels": {"accent": "british", "gender": "male", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "bm_fable": {
        "name": "Fable",
        "engine": "kokoro",
        "category": "premade",
        "description": "Storytelling British male voice",
        "labels": {"accent": "british", "gender": "male", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "bm_george": {
        "name": "George",
        "engine": "kokoro",
        "category": "premade",
        "description": "Deep British male voice",
        "labels": {"accent": "british", "gender": "male", "age": "middle-aged"},
        "sample_rate": 24000,
    },
    "bm_lewis": {
        "name": "Lewis",
        "engine": "kokoro",
        "category": "premade",
        "description": "British male voice",
        "labels": {"accent": "british", "gender": "male", "age": "young"},
        "sample_rate": 24000,
    },
}


_GLD_PREFIX = "gldv1id"


class VoiceMapper:
    """Resolves voice IDs and provides voice metadata."""

    def __init__(self, available_voices: list[str]) -> None:
        self._available = set(available_voices)

    def update_available(self, voices: list[str]) -> None:
        """Update the set of available voices."""
        self._available = set(voices)

    def resolve(self, voice_id: str) -> Optional[str]:
        """Resolve a voice_id to an internal voice name. Returns None if not found."""
        # Resolve gldv1id alias: strip prefix and match with underscores removed
        if voice_id.lower().startswith(_GLD_PREFIX):
            suffix = voice_id[len(_GLD_PREFIX):].lower()
            for v in self._available:
                if v.replace("_", "").lower() == suffix:
                    return v
            logger.warning(f"Voice alias '{voice_id}' not found (suffix '{suffix}')")
            return None
        # Direct match
        if voice_id in self._available:
            return voice_id
        # Case-insensitive match
        for v in self._available:
            if v.lower() == voice_id.lower():
                return v
        logger.warning(f"Voice '{voice_id}' not found")
        return None

    def to_voice_response(self, internal_name: str) -> VoiceResponse:
        """Create an ElevenLabs-compatible VoiceResponse for a voice."""
        meta = VOICE_METADATA.get(internal_name, {})
        return VoiceResponse(
            voice_id=internal_name,
            name=meta.get("name", internal_name),
            category=meta.get("category", "premade"),
            labels=meta.get("labels", {}),
            description=meta.get("description"),
            sample_rate=meta.get("sample_rate", 24000),
            engine=meta.get("engine", "kokoro"),
            settings=VoiceSettingsResponse(),
            high_quality_base_model_ids=["kokoro-v1"] if meta.get("engine") != "glados" else ["glados-v1"],
        )

    def get_all_voices(self) -> list[VoiceResponse]:
        """Get VoiceResponse objects for all available voices."""
        return [self.to_voice_response(v) for v in sorted(self._available)]
