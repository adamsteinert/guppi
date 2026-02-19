"""GLaDOS Text-to-Speech synthesizer for GLaDOS 2.0."""

from pathlib import Path
from pickle import load
from typing import Optional, Any
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
    logger.warning("GLaDOS phonemizer not available - will use fallback")
    PHONEMIZER_AVAILABLE = False
    Phonemizer = None


class GladosSynthesizer:
    """
    GLaDOS voice synthesizer based on the Piper/VITS model.

    Attributes:
        session: ONNX inference session for the GLaDOS model
        phonemizer: Phonemizer instance for text-to-phoneme conversion
        phoneme_to_id: Dictionary mapping phonemes to IDs
        sample_rate: Audio sample rate (22050 Hz for GLaDOS)
    """

    # Constants
    MAX_WAV_VALUE = 32767.0
    SAMPLE_RATE = 22050

    # Phoneme markers
    PAD = "_"  # padding (0)
    BOS = "^"  # beginning of sentence
    EOS = "$"  # end of sentence

    def __init__(
        self,
        model_path: Path,
        phoneme_to_id_path: Optional[Path] = None
    ):
        """
        Initialize the GLaDOS synthesizer.

        Args:
            model_path: Path to the glados.onnx model file
            phoneme_to_id_path: Optional path to phoneme_to_id.pkl file
        """
        self.model_path = model_path
        self.sample_rate = self.SAMPLE_RATE
        self.session: Optional[ort.InferenceSession] = None
        self.phonemizer = None
        self.phoneme_to_id: Optional[dict] = None

        # Load the model
        self._load_model()

        # Load phonemizer
        if PHONEMIZER_AVAILABLE and Phonemizer is not None:
            try:
                self.phonemizer = Phonemizer()
                logger.info("Loaded GLaDOS phonemizer")
            except Exception as e:
                logger.warning(f"Failed to load phonemizer: {e}")

        # Load phoneme-to-ID mapping
        if phoneme_to_id_path and phoneme_to_id_path.exists():
            self._load_phoneme_mapping(phoneme_to_id_path)

    def _load_model(self) -> None:
        """Load the GLaDOS ONNX model."""
        if not ort:
            logger.warning("ONNX Runtime not available")
            return

        if not self.model_path.exists():
            logger.error(f"GLaDOS model not found at {self.model_path}")
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
            logger.info(f"Loaded GLaDOS TTS model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load GLaDOS model: {e}")

    def _load_phoneme_mapping(self, path: Path) -> None:
        """Load phoneme-to-ID mapping from pickle file."""
        try:
            with open(path, "rb") as f:
                self.phoneme_to_id = dict(load(f))
            logger.info(f"Loaded phoneme-to-ID mapping from {path}")
            logger.debug(f"Phoneme mapping has {len(self.phoneme_to_id)} entries")

            # Verify required markers
            for marker in [self.BOS, self.EOS, self.PAD]:
                if marker in self.phoneme_to_id:
                    logger.debug(f"Marker '{marker}' -> {self.phoneme_to_id[marker]}")
                else:
                    logger.warning(f"Missing marker '{marker}' in phoneme_to_id")
        except Exception as e:
            logger.warning(f"Failed to load phoneme_to_id.pkl: {e}")

    def generate_speech_audio(self, text: str) -> Optional[NDArray[np.float32]]:
        """
        Convert input text to synthesized speech audio.

        Args:
            text: The text to be converted to speech

        Returns:
            Array of audio samples (float32), or None if synthesis failed
        """
        if not self.session:
            logger.error("GLaDOS model not loaded")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        try:
            # Convert text to phoneme IDs
            phoneme_ids = self._text_to_phoneme_ids(text)

            if len(phoneme_ids) == 0:
                logger.warning("No phoneme IDs generated from text")
                return None

            # Synthesize audio from phoneme IDs
            audio = self._synthesize_ids_to_audio(phoneme_ids)

            return np.array(audio, dtype=np.float32)

        except Exception as e:
            logger.error(f"GLaDOS synthesis error: {e}")
            return None

    def _text_to_phoneme_ids(self, text: str) -> list[int]:
        """
        Convert text to phoneme IDs.

        Args:
            text: Input text

        Returns:
            List of phoneme IDs
        """
        # Try using proper phonemizer if available
        if self.phonemizer and self.phoneme_to_id:
            try:
                # Convert text to phonemes
                phoneme_list = self.phonemizer.convert_to_phonemes([text], "en_us")
                if phoneme_list:
                    phonemes = phoneme_list[0]
                    logger.debug(f"Phonemes: {phonemes[:100]}...")

                    # Convert phonemes to IDs
                    ids = self._phonemes_to_ids(phonemes)
                    logger.debug(f"Phoneme IDs count: {len(ids)}")

                    if len(ids) > 0:
                        return ids
                    else:
                        logger.warning("Phoneme-to-ID conversion produced empty result")
            except Exception as e:
                logger.warning(f"Phonemizer failed: {e}, using fallback")

        # Fallback: Simple character encoding
        logger.debug("Using character encoding fallback")
        return [ord(c) % 256 for c in text[:200]]

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        """
        Convert phonemes to phoneme IDs.

        Format: BOS + phonemes (with padding) + EOS

        Args:
            phonemes: Phoneme string

        Returns:
            List of phoneme IDs
        """
        if not self.phoneme_to_id:
            return []

        ids: list[int] = list(self.phoneme_to_id.get(self.BOS, [0]))

        for phoneme in phonemes:
            if phoneme not in self.phoneme_to_id:
                continue

            ids.extend(self.phoneme_to_id[phoneme])
            ids.extend(self.phoneme_to_id.get(self.PAD, [0]))

        ids.extend(self.phoneme_to_id.get(self.EOS, [0]))

        return ids

    def _synthesize_ids_to_audio(
        self,
        phoneme_ids: list[int],
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_w: float = 0.8
    ) -> NDArray[np.float32]:
        """
        Synthesize audio from phoneme IDs using the GLaDOS model.

        Args:
            phoneme_ids: List of phoneme IDs
            noise_scale: Controls randomness of generation (default: 0.667)
            length_scale: Controls speech duration (default: 1.0)
            noise_w: Additional noise parameter (default: 0.8)

        Returns:
            Audio waveform as numpy array
        """
        # Prepare inputs
        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

        # Prepare input dictionary
        inputs = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }

        # Add speaker ID if model expects it
        if len(self.session.get_inputs()) > 3:
            inputs["sid"] = np.array([0], dtype=np.int64)

        # Run inference
        outputs = self.session.run(None, inputs)

        # Extract audio and squeeze dimensions
        audio = outputs[0].squeeze()

        return audio.astype(np.float32)

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "session") and self.session is not None:
            del self.session
