"""Text-to-Speech processor for GLaDOS 2.0."""

import asyncio
import numpy as np
from typing import Optional, Union, AsyncGenerator
import threading
from pathlib import Path

from loguru import logger

try:
    import onnxruntime as ort
except ImportError:
    logger.warning("ONNX Runtime not available - TTS will use fallback")
    ort = None

try:
    import soundfile as sf
except ImportError:
    logger.warning("soundfile not available")
    sf = None

from .tts_glados import GladosSynthesizer
from .tts_kokoro import KokoroSynthesizer
from .spoken_text_converter import SpokenTextConverter


class TTSProcessor:
    """
    Text-to-Speech processor supporting GLaDOS and Kokoro voices.

    This class acts as a factory/manager that delegates to specialized
    synthesizers for GLaDOS and Kokoro voices. It provides a unified
    async interface with proper cancellation support and streaming capabilities.
    """

    def __init__(
        self,
        voice: str = "glados",
        model_dir: str = "models/TTS",
        sample_rate: int = 22050
    ):
        """
        Initialize the TTS processor.

        Args:
            voice: Default voice to use
            model_dir: Directory containing TTS model files
            sample_rate: Default sample rate (used for fallback synthesis)
        """
        self.voice = voice
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

        # Text normalizer (converts numbers, currency, etc. to words for TTS)
        self._text_converter = SpokenTextConverter()

        # Synthesizer instances
        self.glados_synthesizer: Optional[GladosSynthesizer] = None
        self.kokoro_synthesizer: Optional[KokoroSynthesizer] = None

        # Load synthesizers
        self._load_synthesizers()

    def _load_synthesizers(self) -> None:
        """Load GLaDOS and Kokoro synthesizers."""
        if not ort:
            logger.warning("ONNX Runtime not available, using fallback TTS")
            return

        # Load GLaDOS synthesizer
        glados_path = self.model_dir / "glados.onnx"
        phoneme_path = self.model_dir / "phoneme_to_id.pkl"

        if glados_path.exists():
            try:
                self.glados_synthesizer = GladosSynthesizer(
                    model_path=glados_path,
                    phoneme_to_id_path=phoneme_path if phoneme_path.exists() else None
                )
                logger.info("Initialized GLaDOS synthesizer")
            except Exception as e:
                logger.error(f"Failed to initialize GLaDOS synthesizer: {e}")

        # Load Kokoro synthesizer
        kokoro_path = self.model_dir / "kokoro-v1.0.fp16.onnx"
        voices_path = self.model_dir / "kokoro-voices-v1.0.bin"

        if kokoro_path.exists() and voices_path.exists():
            try:
                self.kokoro_synthesizer = KokoroSynthesizer(
                    model_path=kokoro_path,
                    voices_path=voices_path
                )
                logger.info("Initialized Kokoro synthesizer")
            except Exception as e:
                logger.error(f"Failed to initialize Kokoro synthesizer: {e}")

    async def synthesize_speech(self, text: str, voice: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice to use (overrides default)

        Returns:
            numpy array of audio samples, or None if synthesis failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        voice = voice or self.voice
        self._cancelled.clear()

        logger.info(f"Synthesizing speech with {voice} voice: {text[:50]}...")

        # Run synthesis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._synthesize_sync, text, voice)
        except Exception as e:
            logger.error(f"Error in async synthesis: {e}")
            return None

    def _synthesize_sync(self, text: str, voice: str) -> Optional[np.ndarray]:
        """Synchronous synthesis method."""
        with self._lock:
            if self._cancelled.is_set():
                logger.debug("Synthesis cancelled before starting")
                return None

            # Normalize text: convert numbers, currency, etc. to spoken words
            # so the phonemizer can handle them (it can't pronounce raw digits)
            text = self._text_converter.text_to_spoken(text)
            logger.debug(f"Normalized text for TTS: {text[:80]}...")

            # Route to appropriate synthesizer based on voice
            if voice == "glados":
                result = self._synthesize_glados(text)
            elif voice.startswith(("af_", "am_", "bf_", "bm_")):
                result = self._synthesize_kokoro(text, voice)
            else:
                logger.warning(f"Unknown voice '{voice}', using fallback")
                result = None

            # Fall back if synthesis failed or produced too little audio
            if result is None or len(result) < 100:
                if result is not None:
                    logger.warning(f"Synthesis produced only {len(result)} samples, using fallback")
                return self._fallback_synthesize(text, voice)

            # Post-process audio to reduce popping and improve quality
            result = self._postprocess_audio(result)

            return result

    def _synthesize_glados(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using GLaDOS synthesizer."""
        if not self.glados_synthesizer:
            logger.warning("GLaDOS synthesizer not available")
            return None

        if self._cancelled.is_set():
            return None

        try:
            audio = self.glados_synthesizer.generate_speech_audio(text)

            if audio is not None:
                logger.debug(f"GLaDOS synthesis completed: {len(audio)} samples")

            return audio
        except Exception as e:
            logger.error(f"GLaDOS synthesis error: {e}")
            return None

    def _synthesize_kokoro(self, text: str, voice: str) -> Optional[np.ndarray]:
        """Synthesize using Kokoro synthesizer."""
        if not self.kokoro_synthesizer:
            logger.warning("Kokoro synthesizer not available")
            return None

        if self._cancelled.is_set():
            return None

        try:
            audio = self.kokoro_synthesizer.generate_speech_audio(text, voice=voice)

            if audio is not None:
                logger.debug(f"Kokoro synthesis completed: {len(audio)} samples @ 24kHz")

            return audio
        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}")
            return None

    def _fallback_synthesize(self, text: str, voice: str) -> np.ndarray:
        """
        Fallback synthesis using simple tone generation.

        For testing when real TTS models aren't available.
        """
        logger.info(f"Using fallback synthesis for voice '{voice}': {text}")

        # Generate simple tones representing speech
        duration = max(1.0, len(text) * 0.08)  # ~80ms per character
        samples = int(self.sample_rate * duration)

        # Create a simple melody based on text characteristics
        audio_data = np.zeros(samples, dtype=np.float32)

        # Add some tonal variation based on text
        for i, char in enumerate(text.lower()):
            if self._cancelled.is_set():
                break

            if char.isalpha():
                # Map characters to frequencies
                freq = 200 + (ord(char) - ord('a')) * 20  # 200-720 Hz range
                start_sample = int((i / len(text)) * samples)
                end_sample = min(start_sample + int(self.sample_rate * 0.1), samples)

                # Generate tone
                t = np.linspace(0, 0.1, end_sample - start_sample)
                tone = 0.1 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 5)  # Decay

                audio_data[start_sample:end_sample] += tone

        # Add some speech-like modulation
        if not self._cancelled.is_set():
            # Apply simple formant-like filtering
            modulation = 0.05 * np.sin(2 * np.pi * 5 * np.linspace(0, duration, samples))
            audio_data = audio_data * (1 + modulation)

        # Apply post-processing to reduce pops
        audio_data = self._postprocess_audio(audio_data)

        return audio_data

    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Post-process audio to reduce popping, clicks, and improve quality.

        Steps:
        1. Remove DC offset (prevents pops)
        2. Normalize to prevent clipping
        3. Apply fade in/fade out (prevents clicks at edges)
        4. Soft-clip to handle any remaining peaks
        """
        if len(audio) == 0:
            return audio

        # 1. Remove DC offset (mean centering)
        audio = audio - np.mean(audio)

        # 2. Normalize to -0.95 to 0.95 range (leave headroom)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        # 3. Apply fade in/fade out to prevent clicks
        fade_samples = min(int(self.sample_rate * 0.01), len(audio) // 10)  # 10ms or 10% of audio

        if fade_samples > 0:
            # Fade in (first few samples)
            fade_in = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in

            # Fade out (last few samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out

        # 4. Soft clipping for any remaining peaks (prevents harsh clipping)
        audio = np.tanh(audio)

        # Ensure float32 output (numpy operations can promote to float64)
        return audio.astype(np.float32)

    def cancel_synthesis(self) -> None:
        """Cancel ongoing synthesis."""
        logger.debug("TTS synthesis cancellation requested")
        self._cancelled.set()

    def is_synthesis_cancelled(self) -> bool:
        """Check if synthesis was cancelled."""
        return self._cancelled.is_set()

    def set_voice(self, voice: str) -> None:
        """Set the default voice."""
        self.voice = voice
        logger.info(f"TTS voice set to: {voice}")

    def get_available_voices(self) -> list[str]:
        """Get list of available voices."""
        voices = []

        # Add GLaDOS voice if available
        if self.glados_synthesizer:
            voices.append("glados")

        # Add Kokoro voices if available
        if self.kokoro_synthesizer:
            kokoro_voices = self.kokoro_synthesizer.get_available_voices()
            voices.extend(kokoro_voices)

        # Always have fallback available
        if not voices:
            voices = ["glados"]

        return voices

    def get_model_info(self) -> dict:
        """Get information about loaded TTS models."""
        # Check if phonemizer is available in either synthesizer
        phonemizer_loaded = False
        if self.glados_synthesizer and self.glados_synthesizer.phonemizer:
            phonemizer_loaded = True
        elif self.kokoro_synthesizer and self.kokoro_synthesizer.phonemizer:
            phonemizer_loaded = True

        info = {
            "glados_loaded": self.glados_synthesizer is not None,
            "kokoro_loaded": self.kokoro_synthesizer is not None,
            "phonemizer_loaded": phonemizer_loaded,
            "current_voice": self.voice,
            "available_voices": self.get_available_voices(),
            "default_sample_rate": self.sample_rate,
        }

        # Add synthesizer-specific info
        if self.glados_synthesizer:
            info["glados_sample_rate"] = self.glados_synthesizer.sample_rate

        if self.kokoro_synthesizer:
            info["kokoro_sample_rate"] = self.kokoro_synthesizer.sample_rate
            info["kokoro_voices_count"] = len(self.kokoro_synthesizer.get_available_voices())

        return info
