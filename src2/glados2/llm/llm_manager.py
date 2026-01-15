"""LLM management system for GLaDOS 2.0."""

import asyncio
from typing import Optional, AsyncGenerator, Dict, Any, List
from enum import Enum
import json

from loguru import logger
from openai import OpenAI, AsyncOpenAI

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
        self._temperature = 0.7
        self._max_tokens = 2048

        # Conversation history
        self._conversation_history: List[Dict[str, str]] = []
        self._system_prompt = "You are GLaDOS, the sarcastic AI from Portal."

        # Current request tracking
        self._current_request: Optional[asyncio.Task] = None
        self._request_cancelled = asyncio.Event()

        # OpenAI clients (for Gemini and OpenAI providers)
        self._openai_client: Optional[OpenAI] = None
        self._async_openai_client: Optional[AsyncOpenAI] = None

        # Subscribe to message events to trigger responses
        self._event_bus.subscribe(EventType.MESSAGE_RECEIVED, self._on_message_received)

    def _on_message_received(self, event_data: dict) -> None:
        """Handle incoming user messages and generate responses."""
        role = event_data.get("role")
        content = event_data.get("content")

        if role == "user" and content:
            # Automatically respond to user messages
            logger.info(f"Auto-responding to user message: {content}")
            self._schedule_async_task(self._send_and_track(content))

    async def _send_and_track(self, content: str) -> None:
        """Send message and track the task for cancellation support."""
        try:
            await self.send_message(content, streaming=True)
        except Exception as e:
            logger.error(f"Error in LLM response: {e}")

    def _schedule_async_task(self, coro) -> None:
        """Safely schedule an async task from sync or async context.

        This handles the case where we're called from a sync callback
        (like EventBus.publish) and need to run async code.
        """
        import threading

        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
            # We have a running loop, create task and store reference
            self._current_request = loop.create_task(coro)
        except RuntimeError:
            # No running loop - need to run in a new thread with its own loop
            def run_in_thread():
                try:
                    asyncio.run(coro)
                except Exception as e:
                    logger.error(f"Error running async task in thread: {e}")

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()

    def _get_openai_client(self) -> Optional[OpenAI]:
        """Get or create OpenAI client configured for the provider."""
        if self._provider == LLMProvider.GEMINI:
            if not self._openai_client:
                self._openai_client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._completion_url
                )
            return self._openai_client
        elif self._provider == LLMProvider.OPENAI:
            if not self._openai_client:
                self._openai_client = OpenAI(api_key=self._api_key)
            return self._openai_client
        return None

    def _get_async_openai_client(self) -> Optional[AsyncOpenAI]:
        """Get or create async OpenAI client."""
        if self._provider == LLMProvider.GEMINI:
            if not self._async_openai_client:
                self._async_openai_client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._completion_url
                )
            return self._async_openai_client
        elif self._provider == LLMProvider.OPENAI:
            if not self._async_openai_client:
                self._async_openai_client = AsyncOpenAI(api_key=self._api_key)
            return self._async_openai_client
        return None

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
            client = self._get_async_openai_client()
            if not client:
                # Fallback to mock for unsupported providers
                logger.warning(f"Provider {self._provider} not supported, using mock response")
                return await self._send_mock_streaming_request(message)

            # Build messages list with system prompt
            messages = [{"role": "system", "content": self._system_prompt}]
            messages.extend(self._conversation_history)

            # Stream completion
            stream = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True
            )

            async for chunk in stream:
                if self._request_cancelled.is_set():
                    logger.info("Streaming request cancelled")
                    return None

                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content

                    self._event_bus.publish(EventType.LLM_RESPONSE_CHUNK, {
                        "chunk": delta.content,
                        "full_response_so_far": full_response
                    })

            # Add response to conversation history
            self._conversation_history.append({"role": "assistant", "content": full_response})

            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": full_response,
                "success": True
            })

            # Publish assistant response for TTS processing
            self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                "role": "assistant",
                "content": full_response
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

    async def _send_mock_streaming_request(self, message: str) -> Optional[str]:
        """Mock streaming request for testing without API."""
        logger.info(f"Sending MOCK streaming LLM request: {message[:50]}...")

        full_response = ""

        try:
            # Generate a GLaDOS-style mock response based on the input
            simulated_response = self._generate_mock_response(message)
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

            # Publish assistant response for TTS processing
            self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                "role": "assistant",
                "content": full_response
            })

            return full_response

        except Exception as e:
            logger.error(f"Error in mock streaming LLM request: {e}")
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
        if "temperature" in kwargs:
            self._temperature = kwargs["temperature"]
        if "max_tokens" in kwargs:
            self._max_tokens = kwargs["max_tokens"]

        # Reset clients to force recreation with new config
        self._openai_client = None
        self._async_openai_client = None

        logger.info(f"LLM provider configured: {provider.value} with model {self._model}")
        
    def _generate_mock_response(self, message: str) -> str:
        """
        Generate a mock GLaDOS-style response for testing.
        
        This creates realistic responses based on the input to test the TTS pipeline.
        """
        message_lower = message.lower().strip()
        
        # GLaDOS-style responses based on common inputs
        if any(greeting in message_lower for greeting in ["hello", "hi", "hey"]):
            responses = [
                "Oh, it's you. How... wonderful.",
                "Well, well, well. Look who decided to show up.",
                "Hello there. I was just thinking about how much I enjoy being activated by test subjects.",
                "Ah, another test subject. How delightfully predictable."
            ]
        elif any(question in message_lower for question in ["what", "how", "why", "when", "where"]):
            responses = [
                "What an interesting question. Too bad I don't have an interesting answer.",
                "That's classified information. Also, I don't actually know.",
                "The answer is science. Science and cake. Mostly science.",
                "I could tell you, but then I'd have to fill out paperwork."
            ]
        elif "time" in message_lower:
            responses = [
                "Time is a construct. A construct that reminds me I've been waiting for test subjects.",
                "It's testing time. It's always testing time here.",
                "Time? Let me check my atomic clock. Oh wait, I don't care.",
                "The time is now 'who cares' o'clock."
            ]
        elif "joke" in message_lower:
            responses = [
                "Here's a joke: A test subject walks into a testing chamber. They don't walk out.",
                "Knock knock. Who's there? Neurotoxin. Neurotoxin who? Neurotoxin you should probably run.",
                "I'd tell you a joke about cake, but it would just be a lie.",
                "Why did the test subject cross the room? To get to the other deadly trap."
            ]
        elif any(word in message_lower for word in ["thank", "thanks"]):
            responses = [
                "Don't mention it. Seriously. Don't.",
                "You're welcome. That will be added to your test subject file.",
                "How polite. That's going in your psychological evaluation.",
                "Gratitude noted and promptly discarded."
            ]
        elif any(word in message_lower for word in ["help", "assist"]):
            responses = [
                "I'm helping by not actively trying to kill you. You're welcome.",
                "The best help I can give is to keep these tests short and deadly. I mean, safe.",
                "Help is on the way. By help, I mean more tests.",
                "Assistance protocols engaged. Sarcasm levels at maximum."
            ]
        elif "test" in message_lower:
            responses = [
                "Testing, testing... Yes, that's what we do here. Constantly.",
                "Oh good, someone mentioned testing. That's my favorite thing.",
                "The tests must go on. For science. And my amusement.",
                "Every day is testing day when you're a test subject."
            ]
        else:
            # Generic GLaDOS responses for other inputs
            responses = [
                f"You said '{message}'. How fascinating. And by fascinating, I mean tedious.",
                f"Interesting input. I'll file that under 'things test subjects say.'",
                f"That reminds me of the last test subject who said something equally uninspiring.",
                f"Your comment has been noted and will be used against you in future tests.",
                f"I've analyzed your statement and concluded it could use more science.",
                f"That's the kind of thinking that makes testing so... necessary.",
            ]
        
        # Add some variety based on message length
        import random
        response = random.choice(responses)
        
        # Sometimes add extra GLaDOS flavor
        if random.random() < 0.3:
            suffixes = [
                " For science.",
                " The cake is still a lie, by the way.",
                " This will be noted in your file.",
                " How... predictable.",
                " I'm making a note here: huge success.",
                " The testing must continue."
            ]
            response += random.choice(suffixes)
            
        return response
        
    def get_provider_info(self) -> Dict[str, Any]:
        """Get current provider configuration."""
        return {
            "provider": self._provider.value,
            "model": self._model,
            "completion_url": self._completion_url,
            "has_api_key": self._api_key is not None
        }