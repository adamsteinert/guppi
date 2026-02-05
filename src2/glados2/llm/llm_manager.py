"""LLM management system for GLaDOS 2.0."""

import asyncio
from typing import Optional, AsyncGenerator, Dict, Any, List, TYPE_CHECKING
from enum import Enum
import json

from loguru import logger
from openai import OpenAI, AsyncOpenAI

from ..core.event_bus import EventBus, EventType
from ..core.state_manager import StateManager, AppState

if TYPE_CHECKING:
    from ..tools.tool_manager import ToolManager


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
    
    def __init__(self, event_bus: EventBus, state_manager: StateManager,
                 tool_manager: Optional["ToolManager"] = None):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._tool_manager = tool_manager

        # Configuration
        self._provider = LLMProvider.OLLAMA
        self._model = "llama3.2"
        self._completion_url = "http://localhost:11434/api/generate"
        self._api_key: Optional[str] = None
        self._temperature = 0.7
        self._max_tokens = 2048

        # Conversation history
        self._conversation_history: List[Dict[str, Any]] = []
        self._system_prompt = "You are GLaDOS, the sarcastic AI from Portal."

        # Current request tracking
        self._current_request: Optional[asyncio.Task] = None
        self._request_cancelled = asyncio.Event()

        # Tool iteration tracking (reset per request)
        self._tool_iterations = 0
        self._max_tool_iterations = 5

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
        """Send a streaming LLM request with tool support."""
        logger.info(f"Sending streaming LLM request: {message[:50]}...")

        # Reset tool iteration counter for new request
        self._tool_iterations = 0

        full_response = ""
        accumulated_tool_calls: List[Dict[str, Any]] = []

        try:
            client = self._get_async_openai_client()
            if not client:
                # Fallback to mock for unsupported providers
                logger.warning(f"Provider {self._provider} not supported, using mock response")
                return await self._send_mock_streaming_request(message)

            # Build messages list with system prompt
            messages: List[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
            messages.extend(self._conversation_history)

            # Build request parameters
            request_params: Dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "stream": True
            }

            # Add tools if available
            if self._tool_manager and self._tool_manager.has_tools():
                request_params["tools"] = self._tool_manager.get_tools_openai_format()
                request_params["tool_choice"] = "auto"

            # Stream completion
            stream = await client.chat.completions.create(**request_params)

            async for chunk in stream:
                if self._request_cancelled.is_set():
                    logger.info("Streaming request cancelled")
                    return None

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                delta = choice.delta

                # Handle content
                if delta.content:
                    full_response += delta.content
                    self._event_bus.publish(EventType.LLM_RESPONSE_CHUNK, {
                        "chunk": delta.content,
                        "full_response_so_far": full_response
                    })

                # Handle tool calls in streaming (accumulate fragments)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index is not None:
                            # Extend list if needed
                            while len(accumulated_tool_calls) <= tc.index:
                                accumulated_tool_calls.append({
                                    "id": "",
                                    "function": {"name": "", "arguments": ""}
                                })
                            # Accumulate fragments
                            if tc.id:
                                accumulated_tool_calls[tc.index]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    accumulated_tool_calls[tc.index]["function"]["name"] = tc.function.name
                                if tc.function.arguments:
                                    accumulated_tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

            # Check for tool calls to execute
            if accumulated_tool_calls and self._tool_manager:
                return await self._handle_tool_calls(messages, accumulated_tool_calls)

            # No tool calls - normal response flow
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

    async def _handle_tool_calls(self,
                                 messages: List[Dict[str, Any]],
                                 tool_calls: List[Dict[str, Any]]) -> Optional[str]:
        """
        Execute tool calls and continue the conversation.

        Supports multi-turn tool usage where the LLM may call
        multiple tools or chain tool calls.
        """
        from ..tools.tool_types import ToolCall

        self._tool_iterations += 1

        if self._tool_iterations > self._max_tool_iterations:
            logger.warning(f"Max tool iterations ({self._max_tool_iterations}) reached")
            error_msg = "I've reached the maximum number of tool calls. Please try rephrasing your request."
            self._conversation_history.append({"role": "assistant", "content": error_msg})
            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": error_msg,
                "success": True
            })
            self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                "role": "assistant",
                "content": error_msg
            })
            return error_msg

        # Build assistant message with tool calls
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                }
                for tc in tool_calls
            ]
        }
        messages.append(assistant_message)

        # Also add to conversation history
        self._conversation_history.append(assistant_message)

        # Execute each tool call
        tool_results: List[Dict[str, Any]] = []
        for tc in tool_calls:
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            tool_call = ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=arguments
            )

            logger.info(f"Executing tool: {tool_call.name}")
            result = await self._tool_manager.execute_tool(tool_call)

            # Format result for API
            result_content = "\n".join(
                item.get("text", str(item))
                for item in result.content
            )

            tool_result_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_content
            }
            tool_results.append(tool_result_msg)

        # Add tool results to messages
        messages.extend(tool_results)

        # Also add to conversation history
        self._conversation_history.extend(tool_results)

        # Continue the conversation with tool results
        return await self._continue_after_tools(messages)

    async def _continue_after_tools(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Continue the conversation after tool execution."""
        client = self._get_async_openai_client()
        if not client:
            return None

        full_response = ""
        accumulated_tool_calls: List[Dict[str, Any]] = []

        request_params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True
        }

        # Include tools for potential chained calls
        if self._tool_manager and self._tool_manager.has_tools():
            request_params["tools"] = self._tool_manager.get_tools_openai_format()
            request_params["tool_choice"] = "auto"

        stream = await client.chat.completions.create(**request_params)

        async for chunk in stream:
            if self._request_cancelled.is_set():
                return None

            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta

            if delta.content:
                full_response += delta.content
                self._event_bus.publish(EventType.LLM_RESPONSE_CHUNK, {
                    "chunk": delta.content,
                    "full_response_so_far": full_response
                })

            # Handle additional tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index is not None:
                        while len(accumulated_tool_calls) <= tc.index:
                            accumulated_tool_calls.append({
                                "id": "",
                                "function": {"name": "", "arguments": ""}
                            })
                        if tc.id:
                            accumulated_tool_calls[tc.index]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                accumulated_tool_calls[tc.index]["function"]["name"] = tc.function.name
                            if tc.function.arguments:
                                accumulated_tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

        # Check for chained tool calls
        if accumulated_tool_calls:
            return await self._handle_tool_calls(messages, accumulated_tool_calls)

        # Final response - reset tool iteration counter
        self._tool_iterations = 0

        self._conversation_history.append({"role": "assistant", "content": full_response})

        self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
            "full_response": full_response,
            "success": True
        })

        self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
            "role": "assistant",
            "content": full_response
        })

        return full_response

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
        if "max_tool_iterations" in kwargs:
            self._max_tool_iterations = kwargs["max_tool_iterations"]

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