"""TTS engine wrapper: adapts TTSProcessor for the ElevenLabs-compatible API."""

from typing import Optional

import numpy as np
from loguru import logger

from glados2.audio.tts_processor import TTSProcessor

from ..models.responses import ModelLanguage, ModelResponse, VoiceResponse
from .voice_mapping import VoiceMapper


class TTSEngine:
    """Wraps TTSProcessor with ElevenLabs-compatible interface."""

    def __init__(self, model_dir: str, default_voice: str = "af_alloy") -> None:
        self._model_dir = model_dir
        self._default_voice = default_voice

        logger.info(f"Initializing TTS engine (model_dir={model_dir}, voice={default_voice})")
        self._tts = TTSProcessor(
            voice=default_voice,
            model_dir=model_dir,
            sample_rate=22050,
        )
        self._voice_mapper = VoiceMapper(self._tts.get_available_voices())
        logger.info(f"TTS engine ready with {len(self._tts.get_available_voices())} voices")

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        model_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech. Returns (float32_audio, native_sample_rate)."""
        # Resolve voice
        internal_voice = self._voice_mapper.resolve(voice_id)
        if internal_voice is None:
            internal_voice = self._default_voice
            logger.warning(f"Voice '{voice_id}' not found, using default '{internal_voice}'")

        # Synthesize
        audio = await self._tts.synthesize_speech(text, voice=internal_voice, speed=speed)

        if audio is None or len(audio) < 100:
            raise ValueError(f"Synthesis failed for voice '{voice_id}' with text: {text[:50]}...")

        # Determine native sample rate
        native_sr = 22050 if internal_voice == "glados" else 24000

        return audio, native_sr

    def get_voice_list(self) -> list[VoiceResponse]:
        """Get all available voices as ElevenLabs-compatible objects."""
        return self._voice_mapper.get_all_voices()

    def get_voice_detail(self, voice_id: str) -> Optional[VoiceResponse]:
        """Get details for a specific voice."""
        internal = self._voice_mapper.resolve(voice_id)
        if internal is not None:
            return self._voice_mapper.to_voice_response(internal)
        return None

    def get_model_list(self) -> list[dict]:
        """Get available models as ElevenLabs-compatible objects."""
        models = []
        info = self._tts.get_model_info()

        if info.get("kokoro_loaded"):
            # Primary Kokoro model
            models.append(
                ModelResponse(
                    model_id="kokoro-v1",
                    name="Kokoro v1.0",
                    description="Multi-voice TTS with 26 voices at 24kHz. High quality, natural sounding.",
                    can_use_style=False,
                    can_use_speaker_boost=False,
                    languages=[ModelLanguage(language_id="en", name="English")],
                ).model_dump()
            )
            # ElevenLabs compatibility aliases
            models.append(
                ModelResponse(
                    model_id="eleven_multilingual_v2",
                    name="Kokoro v1.0 (ElevenLabs compat)",
                    description="Maps to Kokoro v1.0. Use for ElevenLabs SDK compatibility.",
                    languages=[ModelLanguage(language_id="en", name="English")],
                ).model_dump()
            )
            models.append(
                ModelResponse(
                    model_id="eleven_flash_v2_5",
                    name="Kokoro v1.0 (Flash compat)",
                    description="Maps to Kokoro v1.0. Use for ElevenLabs SDK compatibility.",
                    languages=[ModelLanguage(language_id="en", name="English")],
                ).model_dump()
            )

        if info.get("glados_loaded"):
            models.append(
                ModelResponse(
                    model_id="glados-v1",
                    name="GLaDOS v1",
                    description="The iconic GLaDOS VITS/Piper voice at 22050Hz.",
                    languages=[ModelLanguage(language_id="en", name="English")],
                ).model_dump()
            )

        return models
