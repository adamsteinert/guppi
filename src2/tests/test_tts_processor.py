"""Unit tests for TTS Processor."""

import asyncio
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from glados2.audio.tts_processor import TTSProcessor


class TestTTSProcessor:
    """Test suite for TTS Processor."""

    @pytest.fixture
    def tts_processor(self):
        """Create TTS processor instance."""
        return TTSProcessor(
            voice="glados",
            model_dir="models/TTS",
            sample_rate=22050
        )

    def test_initialization(self, tts_processor):
        """Test TTS processor initialization."""
        assert tts_processor is not None
        assert tts_processor.voice == "glados"
        assert tts_processor.sample_rate == 22050
        assert tts_processor.model_dir == Path("models/TTS")

    def test_available_voices(self, tts_processor):
        """Test that available voices are returned."""
        voices = tts_processor.get_available_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "glados" in voices

    def test_set_voice(self, tts_processor):
        """Test setting voice."""
        tts_processor.set_voice("bm_george")
        assert tts_processor.voice == "bm_george"

    @pytest.mark.asyncio
    async def test_fallback_synthesis(self, tts_processor):
        """Test fallback synthesis (when models not loaded)."""
        # Force fallback by using non-existent model
        tts = TTSProcessor(
            voice="glados",
            model_dir="nonexistent",
            sample_rate=22050
        )

        audio = await tts.synthesize_speech("Hello world")

        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32

    @pytest.mark.asyncio
    async def test_synthesis_empty_text(self, tts_processor):
        """Test synthesis with empty text."""
        audio = await tts_processor.synthesize_speech("")
        assert audio is None

        audio = await tts_processor.synthesize_speech("   ")
        assert audio is None

    @pytest.mark.asyncio
    async def test_synthesis_cancellation(self, tts_processor):
        """Test synthesis cancellation."""
        # Start synthesis
        task = asyncio.create_task(
            tts_processor.synthesize_speech("This is a long text that should take time to synthesize")
        )

        # Cancel immediately
        tts_processor.cancel_synthesis()

        # Wait for task
        audio = await task

        # Should return None or partial audio
        assert audio is None or isinstance(audio, np.ndarray)

    def test_model_info(self, tts_processor):
        """Test getting model information."""
        info = tts_processor.get_model_info()

        assert isinstance(info, dict)
        assert "glados_loaded" in info
        assert "kokoro_loaded" in info
        assert "current_voice" in info
        assert "available_voices" in info
        assert "sample_rate" in info
        assert info["sample_rate"] == 22050

    def test_postprocess_audio_dc_removal(self, tts_processor):
        """Test audio post-processing removes DC offset."""
        # Create audio with DC offset
        audio = np.array([0.5, 0.6, 0.7, 0.6, 0.5], dtype=np.float32)

        processed = tts_processor._postprocess_audio(audio)

        # Mean should be close to zero
        assert abs(np.mean(processed)) < 0.1

    def test_postprocess_audio_normalization(self, tts_processor):
        """Test audio post-processing normalizes amplitude."""
        # Create audio with large amplitude
        audio = np.array([5.0, -5.0, 3.0, -3.0], dtype=np.float32)

        processed = tts_processor._postprocess_audio(audio)

        # Should be normalized to [-1, 1] range (actually [-0.95, 0.95])
        assert np.max(np.abs(processed)) <= 1.0

    def test_postprocess_audio_fade(self, tts_processor):
        """Test audio post-processing applies fade in/out."""
        # Create audio with variation (constant audio becomes zero after DC removal)
        audio = np.concatenate([
            np.linspace(0, 0.5, 250, dtype=np.float32),
            np.full(500, 0.5, dtype=np.float32),
            np.linspace(0.5, 0, 250, dtype=np.float32)
        ])

        processed = tts_processor._postprocess_audio(audio)

        # First and last samples should be faded (smaller than middle)
        assert abs(processed[0]) < abs(processed[len(processed) // 2])
        assert abs(processed[-1]) < abs(processed[len(processed) // 2])

    def test_postprocess_empty_audio(self, tts_processor):
        """Test post-processing handles empty audio."""
        audio = np.array([], dtype=np.float32)
        processed = tts_processor._postprocess_audio(audio)
        assert len(processed) == 0


class TestTTSGLaDOSSpecific:
    """Tests specific to GLaDOS voice synthesis."""

    @pytest.fixture
    def glados_tts(self):
        """Create GLaDOS TTS processor."""
        return TTSProcessor(
            voice="glados",
            model_dir="models/TTS",
            sample_rate=22050
        )

    def test_glados_voice_selection(self, glados_tts):
        """Test GLaDOS voice is properly selected."""
        assert glados_tts.voice == "glados"

    @pytest.mark.asyncio
    async def test_glados_synthesis_fallback(self, glados_tts):
        """Test GLaDOS synthesis with fallback (no model)."""
        # This will use fallback if model not loaded
        audio = await glados_tts.synthesize_speech("The cake is a lie")

        assert audio is not None
        assert len(audio) > 0

    def test_phonemes_to_ids(self, glados_tts):
        """Test phoneme to ID conversion."""
        # If phoneme_to_id is loaded, test it
        if glados_tts.phoneme_to_id:
            phonemes = "həˈloʊ"  # Example phonemes
            ids = glados_tts._phonemes_to_ids(phonemes)

            assert isinstance(ids, list)
            # Should have BOS, phoneme IDs, and EOS
            assert len(ids) > 0


class TestTTSKokoroSpecific:
    """Tests specific to Kokoro voice synthesis."""

    @pytest.fixture
    def kokoro_tts(self):
        """Create Kokoro TTS processor."""
        return TTSProcessor(
            voice="bm_george",
            model_dir="models/TTS",
            sample_rate=22050
        )

    def test_kokoro_voice_selection(self, kokoro_tts):
        """Test Kokoro voice is properly selected."""
        assert kokoro_tts.voice == "bm_george"

    def test_kokoro_voices_available(self, kokoro_tts):
        """Test Kokoro voices are in available list."""
        voices = kokoro_tts.get_available_voices()

        # Check for some Kokoro voices
        kokoro_voices = ["bm_george", "bf_emma", "am_adam", "af_sky"]
        for voice in kokoro_voices:
            if kokoro_tts.kokoro_session:
                assert voice in voices

    @pytest.mark.asyncio
    async def test_kokoro_synthesis_fallback(self, kokoro_tts):
        """Test Kokoro synthesis with fallback (no model)."""
        audio = await kokoro_tts.synthesize_speech("Hello from George")

        assert audio is not None
        assert len(audio) > 0

    def test_voice_embedding_generation(self, kokoro_tts):
        """Test voice embedding generation."""
        embedding = kokoro_tts._get_voice_embedding("bm_george")

        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 256)  # Expected shape


class TestTTSIntegration:
    """Integration tests for TTS with real models (if available)."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_glados_full_pipeline(self):
        """Test full GLaDOS synthesis pipeline with real model."""
        tts = TTSProcessor(
            voice="glados",
            model_dir="models/TTS",
            sample_rate=22050
        )

        # Skip if model not loaded
        if not tts.glados_session:
            pytest.skip("GLaDOS model not loaded")

        audio = await tts.synthesize_speech("Testing GLaDOS voice synthesis")

        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32

        # Check audio characteristics
        assert np.max(np.abs(audio)) <= 1.0  # Normalized
        assert len(audio) > tts.sample_rate * 0.5  # At least 0.5 seconds

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_kokoro_full_pipeline(self):
        """Test full Kokoro synthesis pipeline with real model."""
        tts = TTSProcessor(
            voice="bm_george",
            model_dir="models/TTS",
            sample_rate=22050
        )

        # Skip if model not loaded
        if not tts.kokoro_session:
            pytest.skip("Kokoro model not loaded")

        audio = await tts.synthesize_speech("Testing Kokoro voice synthesis with George")

        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32

        # Check audio characteristics
        assert np.max(np.abs(audio)) <= 1.0  # Normalized
        assert len(audio) > tts.sample_rate * 0.5  # At least 0.5 seconds

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_voices(self):
        """Test switching between multiple voices."""
        tts = TTSProcessor(model_dir="models/TTS", sample_rate=22050)

        test_voices = ["glados", "bm_george", "af_sky", "am_adam"]

        for voice in test_voices:
            tts.set_voice(voice)
            audio = await tts.synthesize_speech(f"Testing {voice} voice")

            assert audio is not None
            assert len(audio) > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_synthesis_quality(self):
        """Test that synthesized audio has good quality characteristics."""
        tts = TTSProcessor(
            voice="bm_george",
            model_dir="models/TTS",
            sample_rate=22050
        )

        audio = await tts.synthesize_speech(
            "The quick brown fox jumps over the lazy dog"
        )

        assert audio is not None

        # Check for DC offset (should be minimal after post-processing)
        dc_offset = abs(np.mean(audio))
        assert dc_offset < 0.01, f"DC offset too high: {dc_offset}"

        # Check for clipping (no samples at exactly +/-1.0)
        clipped_samples = np.sum(np.abs(audio) >= 1.0)
        assert clipped_samples == 0, f"Found {clipped_samples} clipped samples"

        # Check for silence (RMS energy should be reasonable)
        rms_energy = np.sqrt(np.mean(audio ** 2))
        assert rms_energy > 0.001, f"Audio too quiet: RMS={rms_energy}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
