"""Kokoro Text-to-Speech synthesizer for GLaDOS 2.0."""

from pathlib import Path
from typing import Optional
import zipfile
import io
import numpy as np
from numpy.typing import NDArray

from loguru import logger

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
except ImportError:
    logger.warning("ONNX Runtime not available")
    ort = None

# Import the phonemizer from the original GLaDOS implementation
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    from glados.TTS.phonemizer import Phonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    logger.warning("Phonemizer not available - Kokoro requires phonemizer")
    PHONEMIZER_AVAILABLE = False
    Phonemizer = None


class KokoroSynthesizer:
    """
    Kokoro multi-voice TTS synthesizer.

    Supports 26 voices with proper IPA phoneme processing.

    Attributes:
        session: ONNX inference session for the Kokoro model
        phonemizer: Phonemizer instance for text-to-IPA conversion
        voices: Dictionary mapping voice names to embeddings
        vocab: IPA phoneme-to-ID mapping
        sample_rate: Audio sample rate (24000 Hz for Kokoro)
    """

    # Constants
    MAX_CHUNK_PHONEMES = 509  # Max phonemes per inference chunk (voice embedding has 510 slots, 0-indexed)
    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_path: Path,
        voices_path: Path
    ):
        """
        Initialize the Kokoro synthesizer.

        Args:
            model_path: Path to the kokoro-v1.0.fp16.onnx model file
            voices_path: Path to the kokoro-voices-v1.0.bin file
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.sample_rate = self.SAMPLE_RATE
        self.session: Optional[ort.InferenceSession] = None
        self.phonemizer = None
        self.voices: dict[str, NDArray] = {}
        self.vocab = self._build_vocab()

        # Load the model
        self._load_model()

        # Load voice embeddings
        if voices_path.exists():
            self._load_voices(voices_path)
        else:
            logger.error(f"Kokoro voices file not found: {voices_path}")

        # Load phonemizer (REQUIRED for Kokoro)
        if PHONEMIZER_AVAILABLE and Phonemizer is not None:
            try:
                self.phonemizer = Phonemizer()
                logger.info("Loaded Kokoro phonemizer")
            except Exception as e:
                logger.error(f"Failed to load phonemizer: {e}")
        else:
            logger.error("Phonemizer not available - Kokoro will not work properly!")

    def _load_model(self) -> None:
        """Load the Kokoro ONNX model."""
        if not ort:
            logger.warning("ONNX Runtime not available")
            return

        if not self.model_path.exists():
            logger.error(f"Kokoro model not found at {self.model_path}")
            return

        try:
            providers = ort.get_available_providers()
            # Remove problematic providers
            if "TensorrtExecutionProvider" in providers:
                providers.remove("TensorrtExecutionProvider")
            if "CoreMLExecutionProvider" in providers:
                providers.remove("CoreMLExecutionProvider")

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=ort.SessionOptions(),
                providers=providers
            )
            logger.info(f"Loaded Kokoro TTS model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Kokoro model: {e}")

    @staticmethod
    def _build_vocab() -> dict[str, int]:
        """
        Build phoneme-to-ID mapping for Kokoro TTS.

        Uses IPA (International Phonetic Alphabet) symbols.
        This matches the original Kokoro implementation.

        Returns:
            Dictionary mapping IPA symbols to IDs
        """
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad, *_punctuation, *_letters, *_letters_ipa]
        vocab = {symbols[i]: i for i in range(len(symbols))}
        return vocab

    def _load_voices(self, voices_path: Path) -> None:
        """
        Load Kokoro voice embeddings from ZIP file.

        Each voice has 510 style vectors (one for each possible phoneme length).

        Args:
            voices_path: Path to kokoro-voices-v1.0.bin ZIP file
        """
        try:
            with zipfile.ZipFile(voices_path, 'r') as zf:
                for filename in zf.namelist():
                    if filename.endswith('.npy'):
                        voice_name = filename.replace('.npy', '')

                        with zf.open(filename) as f:
                            embedding = np.load(io.BytesIO(f.read()))
                            self.voices[voice_name] = embedding
                            logger.debug(f"Loaded voice '{voice_name}': shape {embedding.shape}")

            logger.info(f"Loaded {len(self.voices)} Kokoro voice embeddings")
        except Exception as e:
            logger.error(f"Failed to load voice embeddings from {voices_path}: {e}")

    def get_available_voices(self) -> list[str]:
        """Get list of available voice names."""
        return list(self.voices.keys())

    def generate_speech_audio(
        self,
        text: str,
        voice: str = "af_alloy",
        speed: float = 1.0
    ) -> Optional[NDArray[np.float32]]:
        """
        Convert input text to synthesized speech audio.

        Args:
            text: The text to be converted to speech
            voice: Voice name (e.g., 'bm_george', 'af_alloy')
            speed: Speech speed multiplier (default: 1.0)

        Returns:
            Array of audio samples (float32) at 24kHz, or None if synthesis failed
        """
        if not self.session:
            logger.error("Kokoro model not loaded")
            return None

        if not self.phonemizer:
            logger.error("Phonemizer not available - cannot synthesize")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        if voice not in self.voices:
            logger.error(f"Voice '{voice}' not found. Available: {list(self.voices.keys())[:5]}...")
            return None

        try:
            # Step 1: Convert text to IPA phonemes using real phonemizer
            phonemes_list = self.phonemizer.convert_to_phonemes([text], "en_us")
            phonemes = phonemes_list[0] if phonemes_list else ""
            logger.debug(f"Phonemes ({len(phonemes)}): {phonemes[:100]}...")

            if not phonemes:
                logger.error("Phonemizer produced no output")
                return None

            # Step 2: Convert phonemes to IDs using IPA vocabulary
            all_ids = self._phonemes_to_ids(phonemes)
            logger.debug(f"Phoneme IDs ({len(all_ids)}): {all_ids[:20]}...")

            if len(all_ids) == 0:
                logger.error("No phoneme IDs generated")
                return None

            # Step 3: Split into chunks if needed (voice embeddings support up to MAX_CHUNK_PHONEMES)
            chunks = [
                all_ids[i : i + self.MAX_CHUNK_PHONEMES]
                for i in range(0, len(all_ids), self.MAX_CHUNK_PHONEMES)
            ]
            if len(chunks) > 1:
                logger.debug(f"Splitting {len(all_ids)} phoneme IDs into {len(chunks)} chunks")

            audio_parts: list[np.ndarray] = []
            for chunk_ids in chunks:
                # Wrap with BOS/EOS markers
                tokens = [[0, *chunk_ids, 0]]

                # Get voice embedding based on chunk phoneme length
                voice_array = self._get_voice_embedding(voice, len(chunk_ids))

                # Run Kokoro TTS inference
                outputs = self.session.run(
                    None,
                    {
                        "tokens": tokens,
                        "style": voice_array,
                        "speed": np.ones(1, dtype=np.float32) * speed,
                    },
                )

                # Extract and trim trailing silence Kokoro adds
                chunk_audio = outputs[0]
                if len(chunk_audio) > 8000:
                    chunk_audio = chunk_audio[:-8000]
                audio_parts.append(np.array(chunk_audio, dtype=np.float32))

            audio_data = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
            logger.debug(f"Kokoro synthesis completed: {len(audio_data)} samples @ {self.sample_rate}Hz")

            return audio_data

        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        """
        Convert phoneme string to IDs using Kokoro IPA vocabulary.

        Args:
            phonemes: IPA phoneme string

        Returns:
            List of phoneme IDs
        """
        # Map each IPA phoneme character to its ID
        ids = [self.vocab.get(p) for p in phonemes]
        ids = [i for i in ids if i is not None]

        return ids

    def _get_voice_embedding(self, voice: str, phoneme_length: int) -> NDArray[np.float32]:
        """
        Get voice embedding for the specified voice and phoneme length.

        CRITICAL: Voice embeddings are indexed by phoneme sequence LENGTH!
        The voice file contains 510 style vectors, one for each possible length (0-509).
        Chunks must be kept within MAX_CHUNK_PHONEMES to stay in valid range.

        Args:
            voice: Voice name
            phoneme_length: Number of phonemes in the sequence

        Returns:
            Voice embedding array (shape: 1, 256)
        """
        if voice not in self.voices:
            logger.warning(f"Voice '{voice}' not found, using zero embedding")
            return np.zeros((1, 256), dtype=np.float32)

        voice_vector = self.voices[voice]  # Shape: (510, 1, 256)
        logger.debug(f"Voice vector shape: {voice_vector.shape}, phoneme length: {phoneme_length}")

        # Clamp phoneme length to valid range
        if phoneme_length >= voice_vector.shape[0]:
            logger.warning(f"Phoneme length {phoneme_length} exceeds max {voice_vector.shape[0]-1}, clamping")
            phoneme_length = voice_vector.shape[0] - 1

        # Select the specific style vector for this phoneme length (CRITICAL!)
        voice_array = voice_vector[phoneme_length]  # Shape: (1, 256)
        logger.debug(f"Selected voice array for length {phoneme_length}: {voice_array.shape}")

        return voice_array.astype(np.float32)

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "session") and self.session is not None:
            del self.session
