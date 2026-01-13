"""Audio management system for GLaDOS 2.0."""

import asyncio
from typing import Optional, Callable
from enum import Enum
import threading
import numpy as np

from loguru import logger
import sounddevice as sd

from ..core.event_bus import EventBus, EventType
from ..core.state_manager import StateManager, AppState
from .vad_processor import VADProcessor
from .asr_processor import ASRProcessor
from .tts_processor import TTSProcessor


class AudioState(Enum):
    """Audio system states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    PLAYING = "playing"
    MUTED = "muted"
    ERROR = "error"


class AudioManager:
    """
    Manages all audio operations for GLaDOS 2.0.
    
    This replaces the complex threading and queue management in the original
    with a cleaner, more stable architecture.
    """
    
    def __init__(self, event_bus: EventBus, state_manager: StateManager, config: dict = None):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._audio_state = AudioState.IDLE
        
        # Audio settings from config
        config = config or {}
        self._sample_rate = config.get('sample_rate', 16000)
        self._chunk_size = config.get('chunk_size', 1024)
        self._microphone_muted = config.get('microphone_muted', False)
        self._speaker_muted = config.get('speaker_muted', False)
        self._volume = config.get('volume', 1.0)
        
        # Audio processors
        self._vad = VADProcessor(
            sample_rate=self._sample_rate,
            threshold=config.get('vad_threshold', 0.8),
            min_speech_duration_ms=config.get('min_speech_duration_ms', 250),
            min_silence_duration_ms=config.get('min_silence_duration_ms', 1000)
        )
        self._asr = ASRProcessor(sample_rate=self._sample_rate)
        self._tts = TTSProcessor(
            voice=config.get('voice', 'glados'),
            sample_rate=config.get('tts_sample_rate', 22050)
        )
        
        # Audio streaming
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._listening_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        
        # Control flags
        self._stop_listening = threading.Event()
        self._playback_cancelled = threading.Event()
        
        # Subscribe to relevant events
        self._event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        self._event_bus.subscribe(EventType.MESSAGE_RECEIVED, self._on_message_received)
        
    def _on_state_changed(self, event_data: dict) -> None:
        """Handle application state changes."""
        new_state = event_data.get("new_state")
        
        if new_state == AppState.LISTENING:
            asyncio.create_task(self.start_listening())
        elif new_state == AppState.PLAYING_AUDIO:
            # Audio playback will be started via play_audio method
            pass
        elif new_state == AppState.IDLE:
            asyncio.create_task(self.stop_all_audio())
            
    def _on_message_received(self, event_data: dict) -> None:
        """Handle incoming messages and trigger TTS for assistant responses."""
        role = event_data.get("role")
        content = event_data.get("content")
        
        if role == "assistant" and content:
            # Trigger TTS for assistant responses
            logger.info(f"Processing assistant response for TTS: {content[:50]}...")
            asyncio.create_task(self.synthesize_and_play(content))
            
    async def start_listening(self) -> None:
        """Start listening for voice input."""
        if self._microphone_muted:
            logger.warning("Cannot start listening: microphone is muted")
            return
            
        if self._listening_task and not self._listening_task.done():
            logger.warning("Already listening")
            return
            
        logger.info("Starting audio listening...")
        self._audio_state = AudioState.LISTENING
        self._stop_listening.clear()
        self._vad.reset()
        
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "listening"})
        
        # Start listening task
        self._listening_task = asyncio.create_task(self._listen_loop())
        
    async def stop_all_audio(self) -> None:
        """Stop all audio operations."""
        logger.info("Stopping all audio operations...")
        
        # Stop listening
        self._stop_listening.set()
        if self._listening_task and not self._listening_task.done():
            self._listening_task.cancel()
            
        # Stop playback
        self._playback_cancelled.set()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            
        # Close audio streams
        if self._input_stream:
            self._input_stream.close()
            self._input_stream = None
            
        if self._output_stream:
            self._output_stream.close()
            self._output_stream = None
            
        self._audio_state = AudioState.IDLE
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "idle"})
        
    async def _listen_loop(self) -> None:
        """Main audio listening loop with VAD and ASR processing."""
        try:
            logger.info("Starting audio capture...")
            
            # Audio callback for input stream
            audio_queue = asyncio.Queue()
            
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio input status: {status}")
                # Convert to numpy array and put in queue
                audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                try:
                    audio_queue.put_nowait(audio_data.astype(np.float32))
                except asyncio.QueueFull:
                    pass  # Drop frame if queue is full
                    
            # Start input stream
            self._input_stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self._chunk_size,
                callback=audio_callback
            )
            self._input_stream.start()
            
            logger.info("Audio capture started, waiting for speech...")
            
            # Process audio chunks
            while not self._stop_listening.is_set():
                try:
                    # Get audio chunk with timeout
                    audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    
                    # Process with VAD
                    is_voice, completed_audio = self._vad.process_audio_chunk(audio_chunk)
                    
                    if completed_audio is not None:
                        logger.info("Speech segment detected, transcribing...")
                        await self._process_speech_segment(completed_audio)
                        
                except asyncio.TimeoutError:
                    continue  # Check stop flag
                except Exception as e:
                    logger.error(f"Error in listen loop: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Audio listening error: {e}")
        finally:
            if self._input_stream:
                self._input_stream.close()
                self._input_stream = None
            logger.info("Audio listening stopped")
            
    async def _process_speech_segment(self, audio_data: np.ndarray) -> None:
        """Process a detected speech segment through ASR and send to LLM."""
        try:
            # Update state to processing
            self._state_manager.set_state(AppState.PROCESSING_AUDIO)
            
            # Transcribe audio
            transcription = await self._asr.transcribe_audio(audio_data)
            
            if transcription:
                logger.info(f"Transcription: {transcription}")
                
                # Publish transcription event
                self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                    "role": "user",
                    "content": transcription,
                    "audio_duration": len(audio_data) / self._sample_rate
                })
                
                # Transition to LLM processing
                self._state_manager.set_state(AppState.CALLING_LLM)
                
            else:
                logger.warning("Transcription failed")
                self._state_manager.set_state(AppState.IDLE)
                
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            self._state_manager.set_state(AppState.ERROR)
        
    async def play_audio(self, audio_data: np.ndarray, interruptible: bool = True) -> bool:
        """
        Play audio data with proper cancellation support.
        
        Args:
            audio_data: Audio samples as numpy array
            interruptible: Whether playback can be interrupted
            
        Returns:
            bool: True if playback completed successfully, False if interrupted
        """
        if self._speaker_muted:
            logger.info("Audio playback skipped: speaker is muted")
            return False
            
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data for playback")
            return False
            
        logger.info(f"Starting audio playback: {len(audio_data)} samples")
        self._audio_state = AudioState.PLAYING
        self._playback_cancelled.clear()
        
        # Calculate duration
        duration = len(audio_data) / self._tts.sample_rate
        
        self._event_bus.publish(EventType.AUDIO_PLAYBACK_STARTED, {
            "interruptible": interruptible,
            "duration": duration
        })
        
        try:
            # Apply volume
            audio_data = audio_data * self._volume
            
            # Create playback task
            self._playback_task = asyncio.create_task(
                self._play_audio_async(audio_data, interruptible)
            )
            
            result = await self._playback_task
            
            if result:
                logger.info("Audio playback completed successfully")
                self._event_bus.publish(EventType.AUDIO_PLAYBACK_COMPLETED, {"completed": True})
            else:
                logger.info("Audio playback was interrupted")
                
            return result
            
        except asyncio.CancelledError:
            logger.info("Audio playback cancelled")
            return False
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            self._audio_state = AudioState.ERROR
            return False
        finally:
            if self._audio_state == AudioState.PLAYING:
                self._audio_state = AudioState.IDLE
                
    async def _play_audio_async(self, audio_data: np.ndarray, interruptible: bool) -> bool:
        """Asynchronous audio playback with sounddevice."""
        try:
            # Use sounddevice for playback
            playback_finished = asyncio.Event()
            playback_success = [True]  # Use list to modify from callback
            
            def playback_finished_callback():
                playback_finished.set()
                
            def playback_error_callback(error):
                logger.error(f"Audio playback error: {error}")
                playback_success[0] = False
                playback_finished.set()
                
            # Start playback
            sd.play(
                audio_data,
                samplerate=self._tts.sample_rate,
                blocking=False
            )
            
            # Simulate finished callback since sd.play doesn't have one
            def check_playback():
                import time
                time.sleep(len(audio_data) / self._tts.sample_rate)
                playback_finished_callback()
                
            import threading
            threading.Thread(target=check_playback, daemon=True).start()
            
            # Wait for playback to finish or cancellation
            while not playback_finished.is_set():
                if self._playback_cancelled.is_set():
                    sd.stop()  # Stop playback
                    return False
                    
                await asyncio.sleep(0.01)  # Check every 10ms
                
            return playback_success[0]
            
        except Exception as e:
            logger.error(f"Error in async audio playback: {e}")
            return False
                
    async def synthesize_and_play(self, text: str, voice: str = None) -> bool:
        """
        Synthesize text to speech and play it.
        
        Args:
            text: Text to synthesize
            voice: Voice to use (optional)
            
        Returns:
            bool: True if synthesis and playback succeeded
        """
        try:
            # Set state to TTS generation
            self._state_manager.set_state(AppState.GENERATING_TTS)
            
            self._event_bus.publish(EventType.TTS_STARTED, {
                "text": text,
                "voice": voice or self._tts.voice
            })
            
            # Synthesize speech
            audio_data = await self._tts.synthesize_speech(text, voice)
            
            if audio_data is None:
                logger.error("TTS synthesis failed")
                self._state_manager.set_state(AppState.ERROR)
                return False
                
            self._event_bus.publish(EventType.TTS_COMPLETED, {
                "success": True,
                "duration": len(audio_data) / self._tts.sample_rate
            })
            
            # Set state to playing audio
            self._state_manager.set_state(AppState.PLAYING_AUDIO)
            
            # Play the synthesized audio
            success = await self.play_audio(audio_data, interruptible=True)
            
            # Return to idle state
            if success:
                self._state_manager.set_state(AppState.IDLE)
            else:
                self._state_manager.set_state(AppState.ERROR)
                
            return success
            
        except Exception as e:
            logger.error(f"Error in synthesize and play: {e}")
            self._state_manager.set_state(AppState.ERROR)
            return False
    
    def interrupt_playback(self) -> None:
        """Interrupt current audio playback."""
        if self._playback_task and not self._playback_task.done():
            logger.info("Interrupting audio playback")
            self._playback_cancelled.set()
            
    def set_microphone_muted(self, muted: bool) -> None:
        """Set microphone mute state."""
        self._microphone_muted = muted
        logger.info(f"Microphone {'muted' if muted else 'unmuted'}")
        
    def set_speaker_muted(self, muted: bool) -> None:
        """Set speaker mute state."""
        self._speaker_muted = muted
        logger.info(f"Speaker {'muted' if muted else 'unmuted'}")
        
    def set_volume(self, volume: float) -> None:
        """Set audio volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to {self._volume:.1%}")
        
    def get_audio_state(self) -> AudioState:
        """Get current audio state."""
        return self._audio_state
        
    def is_microphone_muted(self) -> bool:
        """Check if microphone is muted."""
        return self._microphone_muted
        
    def is_speaker_muted(self) -> bool:
        """Check if speaker is muted."""
        return self._speaker_muted
        
    def get_volume(self) -> float:
        """Get current volume."""
        return self._volume