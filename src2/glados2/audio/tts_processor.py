"""Text-to-Speech processor for GLaDOS 2.0."""

import asyncio
import numpy as np
from typing import Optional, Union, AsyncGenerator
import threading
from pathlib import Path
import tempfile
import os

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
            except Exception as e:
                logger.error(f"Failed to load Kokoro model: {e}")
                
        # Load Phonemizer model
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
                return self._glados_synthesize(text)
            elif voice.startswith(("af_", "am_", "bf_", "bm_")) and self.kokoro_session:
                return self._kokoro_synthesize(text, voice)
            else:
                return self._fallback_synthesize(text, voice)
                
    def _glados_synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using GLaDOS model."""
        try:
            # Preprocess text for GLaDOS model
            processed_text = self._preprocess_text_glados(text)
            
            if self._cancelled.is_set():
                return None
                
            # Run GLaDOS TTS inference
            input_name = self.glados_session.get_inputs()[0].name
            inputs = {input_name: processed_text}
            outputs = self.glados_session.run(None, inputs)
            
            if self._cancelled.is_set():
                return None
                
            # Extract audio from outputs
            audio_data = self._extract_audio_glados(outputs)
            logger.debug(f"GLaDOS synthesis completed: {len(audio_data)} samples")
            return audio_data
            
        except Exception as e:
            logger.error(f"GLaDOS synthesis error: {e}")
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
            inputs = {}
            
            if len(input_names) >= 2:
                inputs[input_names[0]] = processed_input
                inputs[input_names[1]] = self._get_voice_embedding(voice)
            else:
                inputs[input_names[0]] = processed_input
                
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
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.7
                
        return audio_data
        
    def _preprocess_text_glados(self, text: str) -> np.ndarray:
        """Preprocess text for GLaDOS model."""
        # This is model-specific - adjust based on your GLaDOS model requirements
        # For now, return a simple character-based encoding
        encoded = np.array([ord(c) for c in text[:200]], dtype=np.int32)  # Limit length
        return encoded.reshape(1, -1)  # Add batch dimension
        
    def _preprocess_text_kokoro(self, text: str, voice: str) -> np.ndarray:
        """Preprocess text for Kokoro model."""
        # This is model-specific - adjust based on Kokoro model requirements
        encoded = np.array([ord(c) for c in text[:200]], dtype=np.int32)
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
        # This would typically load from kokoro-voices-v1.0.bin
        # For now, return a dummy embedding
        return np.random.randn(1, 256).astype(np.float32)
        
    def _extract_audio_glados(self, outputs) -> np.ndarray:
        """Extract audio data from GLaDOS model outputs."""
        # This is model-specific - adjust based on your model's output format
        audio_tensor = outputs[0]  # Assume first output is audio
        return audio_tensor.flatten().astype(np.float32)
        
    def _extract_audio_kokoro(self, outputs) -> np.ndarray:
        """Extract audio data from Kokoro model outputs."""
        # This is model-specific - adjust based on your model's output format
        audio_tensor = outputs[0]  # Assume first output is audio
        return audio_tensor.flatten().astype(np.float32)
        
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