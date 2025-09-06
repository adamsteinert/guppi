"""LLM management system for GLaDOS 2.0."""

import asyncio
from typing import Optional, AsyncGenerator, Dict, Any, List
from enum import Enum
import json

from loguru import logger

from ..core.event_bus import EventBus, EventType
from ..core.state_manager import StateManager, AppState


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMManager:
    """
    Manages LLM interactions for GLaDOS 2.0.
    
    Provides a stable, cancellable interface for LLM communications
    with proper error handling and streaming support.
    """
    
    def __init__(self, event_bus: EventBus, state_manager: StateManager):
        self._event_bus = event_bus
        self._state_manager = state_manager
        
        # Configuration
        self._provider = LLMProvider.OLLAMA
        self._model = "llama3.2"
        self._completion_url = "http://localhost:11434/api/generate"
        self._api_key: Optional[str] = None
        
        # Conversation history
        self._conversation_history: List[Dict[str, str]] = []
        self._system_prompt = "You are GLaDOS, the sarcastic AI from Portal."
        
        # Current request tracking
        self._current_request: Optional[asyncio.Task] = None
        self._request_cancelled = asyncio.Event()
        
    async def send_message(self, message: str, streaming: bool = True) -> Optional[str]:
        """
        Send a message to the LLM and get a response.
        
        Args:
            message: User message to send
            streaming: Whether to use streaming response
            
        Returns:
            Full response text, or None if cancelled/error
        """
        if not self._state_manager.can_transition_to(AppState.CALLING_LLM):
            logger.warning("Cannot send LLM message: invalid state")
            return None
            
        self._state_manager.set_state(AppState.CALLING_LLM)
        self._request_cancelled.clear()
        
        # Add message to conversation history
        self._conversation_history.append({"role": "user", "content": message})
        
        self._event_bus.publish(EventType.LLM_RESPONSE_STARTED, {
            "message": message,
            "streaming": streaming
        })
        
        try:
            if streaming:
                return await self._send_streaming_request(message)
            else:
                return await self._send_single_request(message)
                
        except asyncio.CancelledError:
            logger.info("LLM request cancelled")
            return None
        except Exception as e:
            logger.error(f"Error in LLM request: {e}")
            self._state_manager.set_state(AppState.ERROR)
            return None
        finally:
            if self._state_manager.get_state() == AppState.CALLING_LLM:
                self._state_manager.set_state(AppState.IDLE)
                
    async def _send_streaming_request(self, message: str) -> Optional[str]:
        """Send a streaming LLM request."""
        logger.info(f"Sending streaming LLM request: {message[:50]}...")
        
        full_response = ""
        
        try:
            # TODO: Implement actual LLM streaming request
            # This is a stub - replace with real LLM API calls
            
            # Simulate streaming response
            simulated_response = f"This is GLaDOS responding to: {message}. How delightfully mundane."
            words = simulated_response.split()
            
            for i, word in enumerate(words):
                if self._request_cancelled.is_set():
                    logger.info("Streaming request cancelled")
                    return None
                    
                chunk = word + (" " if i < len(words) - 1 else "")
                full_response += chunk
                
                self._event_bus.publish(EventType.LLM_RESPONSE_CHUNK, {
                    "chunk": chunk,
                    "full_response_so_far": full_response
                })
                
                await asyncio.sleep(0.1)  # Simulate streaming delay
                
            # Add response to conversation history
            self._conversation_history.append({"role": "assistant", "content": full_response})
            
            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": full_response,
                "success": True
            })
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in streaming LLM request: {e}")
            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": full_response,
                "success": False,
                "error": str(e)
            })
            raise
            
    async def _send_single_request(self, message: str) -> Optional[str]:
        """Send a non-streaming LLM request."""
        logger.info(f"Sending single LLM request: {message[:50]}...")
        
        try:
            # TODO: Implement actual LLM single request
            # This is a stub - replace with real LLM API calls
            
            # Simulate processing time
            for i in range(10):
                if self._request_cancelled.is_set():
                    return None
                await asyncio.sleep(0.1)
                
            response = f"GLaDOS says: You asked '{message}' and I'm pretending to process it."
            
            # Add to conversation history
            self._conversation_history.append({"role": "assistant", "content": response})
            
            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": response,
                "success": True
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in single LLM request: {e}")
            raise
            
    def cancel_current_request(self) -> None:
        """Cancel the current LLM request."""
        if self._current_request and not self._current_request.done():
            logger.info("Cancelling LLM request")
            self._request_cancelled.set()
            self._current_request.cancel()
            
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
        logger.info("Conversation history cleared")
        
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self._conversation_history.copy()
        
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self._system_prompt = prompt
        logger.info("System prompt updated")
        
    def configure_provider(self, provider: LLMProvider, **kwargs) -> None:
        """Configure the LLM provider."""
        self._provider = provider
        
        if "model" in kwargs:
            self._model = kwargs["model"]
        if "completion_url" in kwargs:
            self._completion_url = kwargs["completion_url"]
        if "api_key" in kwargs:
            self._api_key = kwargs["api_key"]
            
        logger.info(f"LLM provider configured: {provider.value} with model {self._model}")
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Get current provider configuration."""
        return {
            "provider": self._provider.value,
            "model": self._model,
            "completion_url": self._completion_url,
            "has_api_key": self._api_key is not None
        }