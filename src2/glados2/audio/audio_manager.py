"""Audio management system for GLaDOS 2.0."""

import asyncio
from typing import Optional, Callable
from enum import Enum
import threading

from loguru import logger

from ..core.event_bus import EventBus, EventType
from ..core.state_manager import StateManager, AppState


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
    
    def __init__(self, event_bus: EventBus, state_manager: StateManager):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._audio_state = AudioState.IDLE
        
        # Audio settings
        self._sample_rate = 16000
        self._microphone_muted = False
        self._speaker_muted = False
        self._volume = 1.0
        
        # Playback control
        self._current_playback: Optional[asyncio.Task] = None
        self._playback_cancelled = threading.Event()
        
        # Subscribe to relevant events
        self._event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        
    def _on_state_changed(self, event_data: dict) -> None:
        """Handle application state changes."""
        new_state = event_data.get("new_state")
        
        if new_state == AppState.LISTENING:
            self._start_listening()
        elif new_state == AppState.PLAYING_AUDIO:
            # Audio playback will be started via play_audio method
            pass
        elif new_state == AppState.IDLE:
            self._stop_all_audio()
            
    def _start_listening(self) -> None:
        """Start listening for voice input."""
        if self._microphone_muted:
            logger.warning("Cannot start listening: microphone is muted")
            return
            
        logger.info("Starting audio listening...")
        self._audio_state = AudioState.LISTENING
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "listening"})
        
        # TODO: Start actual audio capture
        # This is a stub - implement actual microphone capture
        
    def _stop_all_audio(self) -> None:
        """Stop all audio operations."""
        logger.info("Stopping all audio operations...")
        
        # Cancel any ongoing playback
        if self._current_playback and not self._current_playback.done():
            self._playback_cancelled.set()
            self._current_playback.cancel()
            
        self._audio_state = AudioState.IDLE
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "idle"})
        
    async def play_audio(self, audio_data: bytes, interruptible: bool = True) -> bool:
        """
        Play audio data with proper cancellation support.
        
        Args:
            audio_data: Raw audio data to play
            interruptible: Whether playback can be interrupted
            
        Returns:
            bool: True if playback completed successfully, False if interrupted
        """
        if self._speaker_muted:
            logger.info("Audio playback skipped: speaker is muted")
            return False
            
        logger.info("Starting audio playback...")
        self._audio_state = AudioState.PLAYING
        self._playback_cancelled.clear()
        
        self._event_bus.publish(EventType.AUDIO_PLAYBACK_STARTED, {
            "interruptible": interruptible,
            "duration": len(audio_data) / (self._sample_rate * 2)  # Rough estimate
        })
        
        try:
            # TODO: Implement actual audio playback
            # This is a stub - replace with real audio playback
            playback_duration = len(audio_data) / (self._sample_rate * 2)
            
            # Simulate playback with cancellation support
            for i in range(int(playback_duration * 10)):  # Check every 100ms
                if self._playback_cancelled.is_set():
                    logger.info("Audio playback cancelled")
                    return False
                    
                await asyncio.sleep(0.1)
                
            logger.info("Audio playback completed")
            self._event_bus.publish(EventType.AUDIO_PLAYBACK_COMPLETED, {"completed": True})
            return True
            
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
                
    def interrupt_playback(self) -> None:
        """Interrupt current audio playback."""
        if self._current_playback and not self._current_playback.done():
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