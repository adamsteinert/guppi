"""Text-to-Speech processor for GLaDOS 2.0."""

import asyncio
import numpy as np
from typing import Optional, Union, AsyncGenerator
import threading
from pathlib import Path
import tempfile
import os
from pickle import load

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

# Import the phonemizer from the original GLaDOS implementation
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    from glados.TTS.phonemizer import Phonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    logger.warning("GLaDOS phonemizer not available - will use character encoding fallback")
    PHONEMIZER_AVAILABLE = False
    Phonemizer = None


class TTSProcessor:
    """
    Text-to-Speech processor supporting GLaDOS and Kokoro voices.
    
    Converts text to audio with proper cancellation support and
    streaming capabilities.
    """
    
    def __init__(
        self,
        voice: str = "glados",
        model_dir: str = "models/TTS",
        sample_rate: int = 22050
    ):
        self.voice = voice
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._cancelled = threading.Event()

        # Model sessions
        self.glados_session: Optional[ort.InferenceSession] = None
        self.kokoro_session: Optional[ort.InferenceSession] = None
        self.phonemizer_session: Optional[ort.InferenceSession] = None

        # GLaDOS specific components
        self.phonemizer = None  # Phonemizer instance if available
        self.phoneme_to_id: Optional[dict] = None

        # Kokoro specific components
        self.voice_embeddings: dict = {}  # Voice name -> embedding mapping

        # GLaDOS constants
        self.PAD = "_"
        self.BOS = "^"
        self.EOS = "$"

        # Load models
        self._load_models()
        
    def _load_models(self) -> None:
        """Load TTS models."""
        if not ort:
            logger.warning("ONNX Runtime not available, using fallback TTS")
            return

        # Load GLaDOS model
        glados_path = self.model_dir / "glados.onnx"
        if glados_path.exists():
            try:
                self.glados_session = ort.InferenceSession(
                    str(glados_path),
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"Loaded GLaDOS TTS model from {glados_path}")

                # Load GLaDOS phonemizer and phoneme mapping
                if PHONEMIZER_AVAILABLE and Phonemizer is not None:
                    try:
                        self.phonemizer = Phonemizer()
                        logger.info("Loaded GLaDOS phonemizer")
                    except Exception as e:
                        logger.warning(f"Failed to load phonemizer: {e}")

                # Load phoneme-to-ID mapping
                phoneme_to_id_path = self.model_dir / "phoneme_to_id.pkl"
                if phoneme_to_id_path.exists():
                    try:
                        with open(phoneme_to_id_path, "rb") as f:
                            self.phoneme_to_id = dict(load(f))
                        logger.info(f"Loaded phoneme-to-ID mapping from {phoneme_to_id_path}")
                        logger.debug(f"Phoneme mapping has {len(self.phoneme_to_id)} entries")
                        # Check for required markers
                        for marker in [self.BOS, self.EOS, self.PAD]:
                            if marker in self.phoneme_to_id:
                                logger.debug(f"Marker '{marker}' -> {self.phoneme_to_id[marker]}")
                            else:
                                logger.warning(f"Missing marker '{marker}' in phoneme_to_id")
                    except Exception as e:
                        logger.warning(f"Failed to load phoneme_to_id.pkl: {e}")
                else:
                    logger.warning(f"phoneme_to_id.pkl not found at {phoneme_to_id_path}")

            except Exception as e:
                logger.error(f"Failed to load GLaDOS model: {e}")

        # Load Kokoro model
        kokoro_path = self.model_dir / "kokoro-v1.0.fp16.onnx"
        if kokoro_path.exists():
            try:
                self.kokoro_session = ort.InferenceSession(
                    str(kokoro_path),
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"Loaded Kokoro TTS model from {kokoro_path}")

                # Load Kokoro voice embeddings
                voices_path = self.model_dir / "kokoro-voices-v1.0.bin"
                if voices_path.exists():
                    self._load_voice_embeddings(voices_path)
                else:
                    logger.warning(f"Kokoro voices file not found: {voices_path}")

            except Exception as e:
                logger.error(f"Failed to load Kokoro model: {e}")

        # Load Phonemizer model (for Kokoro)
        phonemizer_path = self.model_dir / "phomenizer_en.onnx"
        if phonemizer_path.exists():
            try:
                self.phonemizer_session = ort.InferenceSession(
                    str(phonemizer_path),
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"Loaded Phonemizer model from {phonemizer_path}")
            except Exception as e:
                logger.error(f"Failed to load Phonemizer model: {e}")

    def _load_voice_embeddings(self, voices_path: Path) -> None:
        """Load Kokoro voice embeddings from ZIP file."""
        try:
            import zipfile
            import io

            with zipfile.ZipFile(voices_path, 'r') as zf:
                # Load all voice embeddings
                for filename in zf.namelist():
                    if filename.endswith('.npy'):
                        voice_name = filename.replace('.npy', '')

                        # Read the numpy array from the ZIP
                        with zf.open(filename) as f:
                            embedding = np.load(io.BytesIO(f.read()))
                            self.voice_embeddings[voice_name] = embedding
                            logger.debug(f"Loaded voice embedding for {voice_name}: shape {embedding.shape}")

            logger.info(f"Loaded {len(self.voice_embeddings)} Kokoro voice embeddings")

        except Exception as e:
            logger.error(f"Failed to load voice embeddings from {voices_path}: {e}")
                
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

            # Choose synthesis method based on voice and available models
            if voice == "glados" and self.glados_session:
                result = self._glados_synthesize(text)
                # Fall back if GLaDOS synthesis failed or produced too little audio
                if result is None or len(result) < 100:
                    if result is not None:
                        logger.warning(f"GLaDOS synthesis produced only {len(result)} samples, using fallback")
                    else:
                        logger.warning("GLaDOS synthesis failed, using fallback")
                    return self._fallback_synthesize(text, voice)
                return result
            elif voice.startswith(("af_", "am_", "bf_", "bm_")) and self.kokoro_session:
                result = self._kokoro_synthesize(text, voice)
                # Fall back if Kokoro synthesis failed
                if result is None:
                    logger.warning("Kokoro synthesis failed, using fallback")
                    return self._fallback_synthesize(text, voice)
                return result
            else:
                return self._fallback_synthesize(text, voice)
                
    def _glados_synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using GLaDOS model."""
        try:
            # Preprocess text for GLaDOS model (convert to phoneme IDs)
            phoneme_ids = self._preprocess_text_glados(text)

            if self._cancelled.is_set():
                return None

            # Prepare required inputs for GLaDOS model
            phoneme_ids_array = np.expand_dims(phoneme_ids, 0)  # Add batch dimension
            phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)

            # Scales: [noise_scale, length_scale, noise_w]
            scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

            # Speaker ID (None for single-speaker models)
            sid = None

            # Run GLaDOS TTS inference with all required inputs
            inputs = {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
            }

            # Only add sid if the model expects it
            if len(self.glados_session.get_inputs()) > 3:
                inputs["sid"] = np.array([0], dtype=np.int64) if sid is None else sid

            outputs = self.glados_session.run(None, inputs)

            if self._cancelled.is_set():
                return None

            # Extract audio from outputs
            logger.debug(f"GLaDOS output shape: {outputs[0].shape}")
            audio_data = self._extract_audio_glados(outputs)
            logger.debug(f"GLaDOS synthesis completed: {len(audio_data)} samples")
            return audio_data

        except Exception as e:
            logger.error(f"GLaDOS synthesis error: {e}")
            logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            return None
            
    def _kokoro_synthesize(self, text: str, voice: str) -> Optional[np.ndarray]:
        """Synthesize using Kokoro model."""
        try:
            # First, convert text to phonemes if phonemizer is available
            phonemes = text  # Default fallback
            if self.phonemizer_session:
                phonemes = self._text_to_phonemes(text)

            if self._cancelled.is_set():
                return None

            # Preprocess for Kokoro model
            processed_input = self._preprocess_text_kokoro(phonemes, voice)

            if self._cancelled.is_set():
                return None

            # Run Kokoro TTS inference
            input_names = [inp.name for inp in self.kokoro_session.get_inputs()]
            logger.debug(f"Kokoro model inputs: {input_names}")

            # Log expected shapes for debugging
            for inp in self.kokoro_session.get_inputs():
                logger.debug(f"Input '{inp.name}' expects shape: {inp.shape}, type: {inp.type}")

            inputs = {}

            # Map inputs by name (Kokoro expects: tokens, style, speed)
            for input_name in input_names:
                if input_name == 'tokens':
                    inputs[input_name] = processed_input
                    logger.debug(f"Tokens shape: {processed_input.shape}")
                elif input_name == 'style':
                    # Get voice embedding (already shaped as (1, 256))
                    style_embedding = self._get_voice_embedding(voice)
                    inputs[input_name] = style_embedding
                    logger.debug(f"Style shape: {style_embedding.shape}")
                elif input_name == 'speed':
                    # Speed factor (1.0 = normal speed)
                    inputs[input_name] = np.array([1.0], dtype=np.float32)
                else:
                    logger.warning(f"Unknown Kokoro input: {input_name}")

            outputs = self.kokoro_session.run(None, inputs)
            
            if self._cancelled.is_set():
                return None
                
            # Extract audio from outputs
            audio_data = self._extract_audio_kokoro(outputs)
            logger.debug(f"Kokoro synthesis completed: {len(audio_data)} samples")
            return audio_data
            
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
        
    def _preprocess_text_glados(self, text: str) -> np.ndarray:
        """Preprocess text for GLaDOS model.

        Converts text to phoneme IDs using the GLaDOS phonemizer.

        Steps:
        1. Convert text to phonemes using phonemizer
        2. Map phonemes to IDs using phoneme_to_id mapping
        3. Add BOS/EOS markers and padding
        """
        # Try to use proper phonemizer if available
        if self.phonemizer and self.phoneme_to_id:
            try:
                # Convert text to phonemes
                phoneme_list = self.phonemizer.convert_to_phonemes([text], "en_us")
                if phoneme_list:
                    phonemes = phoneme_list[0]
                    logger.debug(f"Phonemes: {phonemes[:100]}...")  # Debug
                    # Convert phonemes to IDs
                    ids = self._phonemes_to_ids(phonemes)
                    logger.debug(f"Phoneme IDs count: {len(ids)}, first 10: {ids[:10]}")  # Debug
                    if len(ids) > 0:
                        return np.array(ids, dtype=np.int64)
                    else:
                        logger.warning("Phoneme-to-ID conversion produced empty result")
            except Exception as e:
                logger.warning(f"Phonemizer failed: {e}, falling back to character encoding")
                import traceback
                logger.debug(traceback.format_exc())

        # Fallback: Simple character-based encoding
        logger.debug("Using character encoding fallback for GLaDOS")
        encoded = np.array([ord(c) % 256 for c in text[:200]], dtype=np.int64)
        return encoded  # Return 1D array, batch dimension added in _glados_synthesize

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        """
        Convert phonemes to phoneme IDs.

        This follows the GLaDOS model's expected format:
        - Start with BOS (beginning of sentence) marker
        - Add each phoneme's ID followed by padding
        - End with EOS (end of sentence) marker
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
        
    def _preprocess_text_kokoro(self, text: str, voice: str) -> np.ndarray:
        """Preprocess text for Kokoro model."""
        # This is model-specific - adjust based on Kokoro model requirements
        # Kokoro expects int64 tokens
        encoded = np.array([ord(c) for c in text[:200]], dtype=np.int64)
        return encoded.reshape(1, -1)
        
    def _text_to_phonemes(self, text: str) -> str:
        """Convert text to phonemes using phonemizer model."""
        try:
            # Simplified phonemizer - real implementation would use the ONNX model
            # For now, return the original text as fallback
            return text
        except Exception as e:
            logger.error(f"Phonemizer error: {e}")
            return text
            
    def _get_voice_embedding(self, voice: str) -> np.ndarray:
        """Get voice embedding for Kokoro model."""
        if voice in self.voice_embeddings:
            embedding = self.voice_embeddings[voice]
            logger.debug(f"Loaded voice embedding for {voice}: shape {embedding.shape}")

            # Kokoro embeddings are (510, 1, 256), squeeze to (510, 256)
            if len(embedding.shape) == 3 and embedding.shape[1] == 1:
                embedding = embedding.squeeze(1)

            # Model expects (1, 256) - take mean of all 510 vectors
            # This gives a single representative vector for the voice
            embedding = np.mean(embedding, axis=0, keepdims=True)
            logger.debug(f"Averaged voice embedding to: {embedding.shape}")

            return embedding.astype(np.float32)
        else:
            logger.warning(f"Voice embedding not found for '{voice}', using fallback")
            # Return a zero embedding as fallback
            return np.zeros((1, 256), dtype=np.float32)
        
    def _extract_audio_glados(self, outputs) -> np.ndarray:
        """Extract audio data from GLaDOS model outputs."""
        # GLaDOS model outputs audio with shape [batch, 1, 1, samples]
        # Squeeze to remove all dimensions except the last one (samples)
        audio_tensor = outputs[0].squeeze()
        audio_data = audio_tensor.astype(np.float32)

        # Post-process to reduce popping and improve quality
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
        
    def _extract_audio_kokoro(self, outputs) -> np.ndarray:
        """Extract audio data from Kokoro model outputs."""
        # This is model-specific - adjust based on your model's output format
        audio_tensor = outputs[0]  # Assume first output is audio
        audio_data = audio_tensor.flatten().astype(np.float32)

        # Post-process to reduce popping
        audio_data = self._postprocess_audio(audio_data)

        return audio_data
        
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
        voices = ["glados"]  # Always available (fallback)
        
        if self.kokoro_session:
            # Add Kokoro voices
            voices.extend([
                "af_alloy", "af_aoede", "af_jessica", "af_kore", "af_nicole", 
                "af_nova", "af_river", "af_saraha", "af_sky",
                "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
                "am_michael", "am_onyx", "am_puck",
                "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
            ])
            
        return voices
        
    def get_model_info(self) -> dict:
        """Get information about loaded TTS models."""
        return {
            "glados_loaded": self.glados_session is not None,
            "kokoro_loaded": self.kokoro_session is not None,
            "phonemizer_loaded": self.phonemizer_session is not None,
            "current_voice": self.voice,
            "available_voices": self.get_available_voices(),
            "sample_rate": self.sample_rate,
        }