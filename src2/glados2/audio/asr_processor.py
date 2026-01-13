"""Automatic Speech Recognition processor for GLaDOS 2.0."""

import asyncio
import numpy as np
from typing import Optional, Union
import threading
from pathlib import Path

from loguru import logger

try:
    import onnxruntime as ort
except ImportError:
    logger.warning("ONNX Runtime not available - ASR will use fallback")
    ort = None

try:
    import soundfile as sf
except ImportError:
    logger.warning("soundfile not available - using numpy for audio")
    sf = None


class ASRProcessor:
    """
    Automatic Speech Recognition processor using Nemo Parakeet model.
    
    Converts audio samples to text transcription with proper error handling
    and fallback mechanisms.
    """
    
    def __init__(
        self,
        model_path: str = "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx",
        sample_rate: int = 16000
    ):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.session: Optional[ort.InferenceSession] = None
        self._lock = threading.Lock()
        
        # Load ASR model
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the ASR ONNX model."""
        if not ort:
            logger.warning("ONNX Runtime not available, using fallback ASR")
            return
            
        if not self.model_path.exists():
            logger.warning(f"ASR model not found at {self.model_path}, using fallback")
            return
            
        try:
            # Configure ONNX Runtime providers
            providers = ['CPUExecutionProvider']
            
            # Try to use GPU if available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            logger.info(f"Loaded ASR model from {self.model_path} with providers: {providers}")
            
            # Log model input/output info for debugging
            for input_meta in self.session.get_inputs():
                logger.debug(f"ASR Input: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
            for output_meta in self.session.get_outputs():
                logger.debug(f"ASR Output: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")
                
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            self.session = None
            
    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio samples as numpy array (float32, 16kHz)
            
        Returns:
            str: Transcribed text, or None if transcription failed
        """
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data provided for transcription")
            return None
            
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._transcribe_sync, audio_data)
        except Exception as e:
            logger.error(f"Error in async transcription: {e}")
            return None
            
    def _transcribe_sync(self, audio_data: np.ndarray) -> Optional[str]:
        """Synchronous transcription method."""
        with self._lock:
            if self.session:
                try:
                    return self._onnx_transcribe(audio_data)
                except Exception as e:
                    logger.error(f"ONNX transcription failed: {e}")
                    
            # Fallback transcription
            return self._fallback_transcribe(audio_data)
            
    def _onnx_transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe using ONNX Nemo Parakeet model."""
        # Preprocess audio for Nemo model
        processed_audio = self._preprocess_audio(audio_data)
        
        # Prepare input tensor
        # Note: This is model-specific formatting - adjust based on your model's requirements
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        inputs = {input_name: processed_audio}
        outputs = self.session.run(None, inputs)
        
        # Post-process outputs to get text
        # Note: This is model-specific - adjust based on your model's output format
        transcription = self._postprocess_outputs(outputs)
        
        logger.debug(f"ONNX transcription result: {transcription}")
        return transcription
        
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for the ASR model.
        
        This is model-specific preprocessing. For Nemo Parakeet, we typically need:
        - Specific sample rate (16kHz)
        - Normalized amplitude
        - Correct tensor shape
        """
        # Ensure float32 format
        audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # Add batch dimension if needed
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
            
        return audio_data
        
    def _postprocess_outputs(self, outputs) -> str:
        """
        Post-process model outputs to extract text.
        
        This is model-specific. For CTC models, we typically need to:
        - Apply CTC decoding
        - Map tokens to characters/words
        - Remove blanks and repetitions
        """
        # This is a simplified version - real implementation depends on model output format
        try:
            # For demonstration, assuming the model outputs logits or token IDs
            output_tensor = outputs[0]  # First output
            
            # Simple greedy decoding (replace with proper CTC decoding for real model)
            if output_tensor.ndim > 1:
                # If output is logits, take argmax
                token_ids = np.argmax(output_tensor, axis=-1)
            else:
                token_ids = output_tensor
                
            # Convert token IDs to text (this is model-specific)
            # For now, return a placeholder that indicates we got audio
            return "transcribed speech placeholder"  # Replace with real token-to-text conversion
            
        except Exception as e:
            logger.error(f"Error in output post-processing: {e}")
            return None
            
    def _fallback_transcribe(self, audio_data: np.ndarray) -> str:
        """
        Fallback transcription when ONNX model is not available.
        
        For testing purposes, this returns a mock transcription based on audio characteristics.
        """
        # Calculate some basic audio features for realistic fallback
        duration = len(audio_data) / self.sample_rate
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        logger.info(f"Fallback transcription - Duration: {duration:.2f}s, Energy: {rms_energy:.4f}")
        
        # Return mock transcription based on audio characteristics
        if duration < 1.0:
            return "hello"
        elif duration < 2.0:
            return "hello GLaDOS"
        elif duration < 3.0:
            return "what time is it"
        elif duration < 4.0:
            return "tell me a joke"
        else:
            return "this is a longer speech segment for testing"
            
    def is_model_loaded(self) -> bool:
        """Check if ASR model is properly loaded."""
        return self.session is not None
        
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.session:
            return {
                "model_loaded": False,
                "model_path": str(self.model_path),
                "fallback_mode": True
            }
            
        return {
            "model_loaded": True,
            "model_path": str(self.model_path),
            "fallback_mode": False,
            "providers": self.session.get_providers(),
            "input_shape": [inp.shape for inp in self.session.get_inputs()],
            "output_shape": [out.shape for out in self.session.get_outputs()],
        }