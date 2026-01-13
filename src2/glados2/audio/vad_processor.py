"""Voice Activity Detection processor for GLaDOS 2.0."""

import asyncio
import numpy as np
from typing import Callable, Optional
import threading
import time
from collections import deque

from loguru import logger

try:
    import onnxruntime as ort
except ImportError:
    logger.warning("ONNX Runtime not available - VAD will use fallback detection")
    ort = None


class VADProcessor:
    """
    Voice Activity Detection processor using Silero VAD model.
    
    Processes audio chunks to detect speech activity with configurable
    thresholds and buffering.
    """
    
    def __init__(
        self,
        model_path: str = "models/ASR/silero_vad_v5.onnx",
        sample_rate: int = 16000,
        threshold: float = 0.8,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 1000,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        
        # Audio processing
        self.window_size_samples = int(sample_rate * 0.032)  # 32ms windows
        self.audio_buffer = deque(maxlen=int(sample_rate * 0.8))  # 800ms buffer
        
        # State tracking
        self.is_speech_active = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.collected_audio = []
        
        # Load VAD model
        self.session: Optional[ort.InferenceSession] = None
        self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> None:
        """Load the Silero VAD ONNX model."""
        if not ort:
            logger.warning("ONNX Runtime not available, using fallback VAD")
            return
            
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']  # Start with CPU, can upgrade to GPU
            )
            logger.info(f"Loaded VAD model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            self.session = None
            
    def process_audio_chunk(self, audio_data: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        Process an audio chunk and return speech detection result.
        
        Args:
            audio_data: Audio samples as numpy array (float32, 16kHz)
            
        Returns:
            tuple: (is_voice_detected, completed_speech_audio)
                - is_voice_detected: Whether voice is currently detected
                - completed_speech_audio: Complete speech segment if speech just ended, None otherwise
        """
        current_time = time.time()
        
        # Add to circular buffer
        self.audio_buffer.extend(audio_data)
        
        # Detect voice activity in this chunk
        voice_probability = self._detect_voice_activity(audio_data)
        is_voice_detected = voice_probability > self.threshold
        
        # State machine for speech detection
        completed_audio = None
        
        if is_voice_detected:
            if not self.is_speech_active:
                # Speech started
                logger.debug("Speech activity started")
                self.is_speech_active = True
                self.speech_start_time = current_time
                # Include buffered audio before speech start
                self.collected_audio = [np.array(self.audio_buffer, dtype=np.float32)]
            else:
                # Continue collecting speech
                self.collected_audio.append(audio_data.copy())
                
            self.last_speech_time = current_time
            
        else:
            if self.is_speech_active:
                # Check if silence duration is long enough to end speech
                silence_duration = current_time - (self.last_speech_time or current_time)
                
                if silence_duration >= (self.min_silence_duration_ms / 1000.0):
                    # Speech ended
                    speech_duration = current_time - (self.speech_start_time or current_time)
                    
                    if speech_duration >= (self.min_speech_duration_ms / 1000.0):
                        # Valid speech segment
                        logger.debug(f"Speech segment completed: {speech_duration:.2f}s")
                        completed_audio = np.concatenate(self.collected_audio)
                    else:
                        logger.debug("Speech segment too short, discarded")
                        
                    # Reset state
                    self.is_speech_active = False
                    self.speech_start_time = None
                    self.last_speech_time = None
                    self.collected_audio.clear()
                    
        return is_voice_detected, completed_audio
        
    def _detect_voice_activity(self, audio_data: np.ndarray) -> float:
        """
        Detect voice activity in audio chunk.
        
        Returns:
            float: Voice probability (0.0 to 1.0)
        """
        if self.session and len(audio_data) >= self.window_size_samples:
            try:
                # Use Silero VAD model
                return self._silero_vad_inference(audio_data)
            except Exception as e:
                logger.warning(f"VAD model inference failed: {e}")
                
        # Fallback: simple energy-based detection
        return self._energy_based_vad(audio_data)
        
    def _silero_vad_inference(self, audio_data: np.ndarray) -> float:
        """Run Silero VAD model inference."""
        # Prepare input for Silero VAD (expects specific input format)
        if len(audio_data) < self.window_size_samples:
            # Pad if necessary
            padded = np.zeros(self.window_size_samples, dtype=np.float32)
            padded[:len(audio_data)] = audio_data
            audio_data = padded
        else:
            # Take first window_size_samples
            audio_data = audio_data[:self.window_size_samples]
            
        # Normalize audio to [-1, 1] range
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # Reshape for ONNX model input
        input_tensor = audio_data.reshape(1, -1)
        
        # Run inference
        inputs = {self.session.get_inputs()[0].name: input_tensor}
        output = self.session.run(None, inputs)
        
        # Extract speech probability
        speech_prob = float(output[0][0][0])  # Model-specific output format
        return speech_prob
        
    def _energy_based_vad(self, audio_data: np.ndarray) -> float:
        """Simple energy-based voice activity detection fallback."""
        if len(audio_data) == 0:
            return 0.0
            
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Simple threshold-based detection
        # This is a very basic fallback - adjust threshold as needed
        energy_threshold = 0.01  # Adjust based on your microphone sensitivity
        
        if rms_energy > energy_threshold:
            return min(rms_energy * 10, 1.0)  # Scale to 0-1 range
        else:
            return 0.0
            
    def reset(self) -> None:
        """Reset VAD state."""
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.collected_audio.clear()
        self.audio_buffer.clear()
        logger.debug("VAD processor reset")
        
    def get_status(self) -> dict:
        """Get current VAD status."""
        return {
            "is_speech_active": self.is_speech_active,
            "speech_duration": (
                time.time() - self.speech_start_time 
                if self.speech_start_time else 0.0
            ),
            "buffer_size": len(self.audio_buffer),
            "collected_samples": sum(len(chunk) for chunk in self.collected_audio)
        }