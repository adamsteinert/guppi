"""Event bus for decoupled communication between components."""

from enum import Enum
from typing import Any, Callable, Dict, List
from loguru import logger
import asyncio


class EventType(Enum):
    """Types of events that can be published on the event bus."""
    STATE_CHANGED = "state_changed"
    MESSAGE_RECEIVED = "message_received"
    AUDIO_STATUS_CHANGED = "audio_status_changed"
    AUDIO_DEVICE_CHANGED = "audio_device_changed"  # System audio device changed
    LLM_RESPONSE_STARTED = "llm_response_started"
    LLM_RESPONSE_CHUNK = "llm_response_chunk"
    LLM_RESPONSE_COMPLETED = "llm_response_completed"
    TTS_STARTED = "tts_started"
    TTS_COMPLETED = "tts_completed"
    AUDIO_PLAYBACK_STARTED = "audio_playback_started"
    AUDIO_PLAYBACK_COMPLETED = "audio_playback_completed"
    INTERRUPT_REQUESTED = "interrupt_requested"
    LISTENING_STOPPED = "listening_stopped"  # User stopped listening, process collected audio
    ERROR_OCCURRED = "error_occurred"
    SHUTDOWN_REQUESTED = "shutdown_requested"


class EventBus:
    """
    Central event bus for decoupled communication between components.
    
    This replaces the tightly-coupled callbacks and direct method calls
    in the original architecture with a clean pub-sub pattern.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._async_subscribers: Dict[EventType, List[Callable]] = {}
        
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to an event type with a synchronous callback."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type.value}")
        
    def subscribe_async(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to an event type with an asynchronous callback."""
        if event_type not in self._async_subscribers:
            self._async_subscribers[event_type] = []
        self._async_subscribers[event_type].append(callback)
        logger.debug(f"Async subscribed to {event_type.value}")
        
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass
                
        if event_type in self._async_subscribers:
            try:
                self._async_subscribers[event_type].remove(callback)
            except ValueError:
                pass
                
    def publish(self, event_type: EventType, data: Dict[str, Any] = None) -> None:
        """Publish an event synchronously."""
        if data is None:
            data = {}
            
        logger.debug(f"Publishing {event_type.value} with data: {data}")
        
        # Call synchronous subscribers
        for callback in self._subscribers.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in event subscriber for {event_type.value}: {e}")
                
    async def publish_async(self, event_type: EventType, data: Dict[str, Any] = None) -> None:
        """Publish an event asynchronously."""
        if data is None:
            data = {}
            
        logger.debug(f"Async publishing {event_type.value} with data: {data}")
        
        # Call synchronous subscribers first
        self.publish(event_type, data)
        
        # Call asynchronous subscribers
        tasks = []
        for callback in self._async_subscribers.get(event_type, []):
            try:
                tasks.append(asyncio.create_task(callback(data)))
            except Exception as e:
                logger.error(f"Error creating async task for {event_type.value}: {e}")
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)