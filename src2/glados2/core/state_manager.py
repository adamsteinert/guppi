"""State management for GLaDOS 2.0."""

from enum import Enum
from typing import Optional
from loguru import logger
import threading

from .event_bus import EventBus, EventType


class AppState(Enum):
    """Application states for GLaDOS 2.0."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_AUDIO = "processing_audio"
    CALLING_LLM = "calling_llm"
    GENERATING_TTS = "generating_tts"
    PLAYING_AUDIO = "playing_audio"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class StateManager:
    """
    Thread-safe state manager for GLaDOS 2.0.
    
    Manages application state transitions and ensures consistency
    across the entire application.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self._current_state = AppState.INITIALIZING
        self._previous_state: Optional[AppState] = None
        self._lock = threading.RLock()
        self._event_bus = event_bus or EventBus()
        
    def get_state(self) -> AppState:
        """Get the current application state."""
        with self._lock:
            return self._current_state
            
    def get_previous_state(self) -> Optional[AppState]:
        """Get the previous application state."""
        with self._lock:
            return self._previous_state
            
    def set_state(self, new_state: AppState) -> bool:
        """
        Set a new application state.
        
        Returns:
            bool: True if state change was successful, False if transition is invalid
        """
        with self._lock:
            if not self._is_valid_transition(self._current_state, new_state):
                logger.warning(f"Invalid state transition from {self._current_state.value} to {new_state.value}")
                return False
                
            old_state = self._current_state
            self._previous_state = self._current_state
            self._current_state = new_state
            
            logger.info(f"State changed: {old_state.value} -> {new_state.value}")
            
            # Publish state change event
            self._event_bus.publish(EventType.STATE_CHANGED, {
                "previous_state": old_state,
                "new_state": new_state
            })
            
            return True
            
    def can_transition_to(self, target_state: AppState) -> bool:
        """Check if transition to target state is valid."""
        with self._lock:
            return self._is_valid_transition(self._current_state, target_state)
            
    def _is_valid_transition(self, from_state: AppState, to_state: AppState) -> bool:
        """
        Check if a state transition is valid.
        
        This prevents invalid state transitions that could cause instability.
        """
        # Allow all transitions from INITIALIZING and to SHUTTING_DOWN
        if from_state == AppState.INITIALIZING or to_state == AppState.SHUTTING_DOWN:
            return True
            
        # Allow transitions to ERROR from any state
        if to_state == AppState.ERROR:
            return True
            
        # Allow transitions back to IDLE from most states
        if to_state == AppState.IDLE and from_state in [
            AppState.LISTENING,
            AppState.PROCESSING_AUDIO,
            AppState.CALLING_LLM,
            AppState.GENERATING_TTS,
            AppState.PLAYING_AUDIO,
            AppState.ERROR
        ]:
            return True
            
        # Define valid state transitions
        valid_transitions = {
            AppState.IDLE: [
                AppState.LISTENING,
                AppState.CALLING_LLM,  # For text input
                AppState.GENERATING_TTS,  # For direct TTS
            ],
            AppState.LISTENING: [
                AppState.IDLE,  # Cancel listening
                AppState.PROCESSING_AUDIO,  # Voice detected
            ],
            AppState.PROCESSING_AUDIO: [
                AppState.IDLE,  # Processing failed/cancelled
                AppState.CALLING_LLM,  # ASR completed
            ],
            AppState.CALLING_LLM: [
                AppState.IDLE,  # LLM failed/cancelled
                AppState.GENERATING_TTS,  # LLM response received
            ],
            AppState.GENERATING_TTS: [
                AppState.IDLE,  # TTS failed/cancelled
                AppState.PLAYING_AUDIO,  # TTS completed
            ],
            AppState.PLAYING_AUDIO: [
                AppState.IDLE,  # Playback completed/interrupted
                AppState.LISTENING,  # Interruptible mode
            ],
            AppState.ERROR: [
                AppState.IDLE,  # Recovery
                AppState.INITIALIZING,  # Restart
            ]
        }
        
        return to_state in valid_transitions.get(from_state, [])
        
    def is_busy(self) -> bool:
        """Check if the application is currently processing something."""
        with self._lock:
            busy_states = [
                AppState.PROCESSING_AUDIO,
                AppState.CALLING_LLM,
                AppState.GENERATING_TTS,
                AppState.PLAYING_AUDIO
            ]
            return self._current_state in busy_states
            
    def is_ready_for_input(self) -> bool:
        """Check if the application can accept new input."""
        with self._lock:
            ready_states = [AppState.IDLE, AppState.LISTENING]
            return self._current_state in ready_states