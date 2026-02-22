"""LLM management system for GLaDOS 2.0."""

from __future__ import annotations

import asyncio
import threading
from typing import Optional, AsyncGenerator, Dict, Any, List, TYPE_CHECKING
from enum import Enum
import json

from loguru import logger

from ..core.event_bus import EventBus, EventType
from ..core.state_manager import StateManager, AppState

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from ..core.worker_loop import WorkerLoop
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
                 tool_manager: Optional["ToolManager"] = None,
                 worker_loop: Optional["WorkerLoop"] = None):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._tool_manager = tool_manager
        self._worker_loop = worker_loop

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
        # threading.Event (not asyncio.Event) because cancellation is
        # signalled across threads — asyncio.Event is not thread-safe.
        self._request_cancelled = threading.Event()

        # Tool iteration tracking (reset per request)
        self._tool_iterations = 0
        self._max_tool_iterations = 5

        # TTS summarization settings
        self._summarize_long_responses = True
        self._max_speech_words = 150  # ~30 seconds at typical speech rate

        # OpenAI clients (for OpenAI and Ollama providers)
        self._openai_client: Optional[OpenAI] = None
        self._async_openai_client: Optional[AsyncOpenAI] = None

        # Native Google GenAI client (for Gemini provider)
        self._genai_client: Optional[Any] = None

        # Gemini thinking config
        self._thinking_enabled = False
        self._thinking_level = "MEDIUM"

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
        """Schedule an async coroutine on the persistent worker loop.

        When a worker loop is available, dispatches to it so that MCP
        tool connections (which are bound to the loop where they were
        created) remain usable during tool execution.

        Falls back to spawning a new thread when no worker loop is
        configured (e.g. in unit tests).
        """
        if self._worker_loop and self._worker_loop.is_running:
            self._worker_loop.run(coro)
        else:
            def run_in_thread():
                try:
                    asyncio.run(coro)
                except Exception as e:
                    logger.error(f"Error running async task in thread: {e}")

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()

    def _get_openai_client(self) -> Optional[OpenAI]:
        """Get or create OpenAI client configured for the provider."""
        from openai import OpenAI

        if self._provider == LLMProvider.OPENAI:
            if not self._openai_client:
                self._openai_client = OpenAI(api_key=self._api_key)
            return self._openai_client
        elif self._provider == LLMProvider.OLLAMA:
            if not self._openai_client:
                self._openai_client = OpenAI(
                    api_key="ollama",
                    base_url=self._completion_url,
                )
            return self._openai_client
        return None

    def _get_async_openai_client(self) -> Optional[AsyncOpenAI]:
        """Get or create async OpenAI client."""
        from openai import AsyncOpenAI

        if self._provider == LLMProvider.OPENAI:
            if not self._async_openai_client:
                self._async_openai_client = AsyncOpenAI(api_key=self._api_key)
            return self._async_openai_client
        elif self._provider == LLMProvider.OLLAMA:
            if not self._async_openai_client:
                self._async_openai_client = AsyncOpenAI(
                    api_key="ollama",
                    base_url=self._completion_url,
                )
            return self._async_openai_client
        return None

    def _get_genai_client(self) -> Any:
        """Get or create native Google GenAI client for Gemini provider."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package is required for native Gemini provider. "
                "Install with: uv add google-genai"
            )
        if self._genai_client is None:
            self._genai_client = genai.Client(api_key=self._api_key)
        return self._genai_client

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
        """Send a streaming LLM request, dispatching to the appropriate provider."""
        logger.info(f"Sending streaming LLM request: {message[:50]}...")

        # Reset tool iteration counter for new request
        self._tool_iterations = 0

        if self._provider == LLMProvider.GEMINI:
            return await self._send_gemini_streaming_request(message)
        else:
            return await self._send_openai_streaming_request(message)

    async def _send_openai_streaming_request(self, message: str) -> Optional[str]:
        """Send a streaming request via OpenAI-compatible API (OpenAI/Ollama)."""
        full_response = ""
        accumulated_tool_calls: List[Dict[str, Any]] = []

        try:
            client = self._get_async_openai_client()
            if not client:
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

            # Publish assistant response for TTS processing (with summarization if needed)
            await self._publish_assistant_response_for_tts(full_response)

            return full_response

        except Exception as e:
            logger.error(f"Error in streaming LLM request: {e}")
            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": full_response,
                "success": False,
                "error": str(e)
            })
            raise

    # ── Gemini-native methods ──────────────────────────────────────────

    async def _send_gemini_streaming_request(self, message: str) -> Optional[str]:
        """Send a streaming request using the native Google GenAI SDK."""
        full_response = ""
        accumulated_function_calls: list = []
        # Preserve all raw model parts (including thoughts with signatures)
        # so that thought_signature metadata is not lost when sending
        # function call results back to the API.
        raw_model_parts: list = []

        try:
            client = self._get_genai_client()

            # Build contents from conversation history
            contents = self._build_gemini_contents(message)
            config = self._build_gemini_config()

            # Stream response using async client
            stream = await client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )

            async for chunk in stream:
                if self._request_cancelled.is_set():
                    logger.info("Gemini streaming request cancelled")
                    return None

                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    raw_model_parts.append(part)

                    # Skip thinking/thought parts - don't send to TTS or UI
                    if getattr(part, "thought", False):
                        logger.debug(f"Thinking: {getattr(part, 'text', '')[:80]}...")
                        continue

                    # Handle text content
                    if part.text:
                        full_response += part.text
                        self._event_bus.publish(EventType.LLM_RESPONSE_CHUNK, {
                            "chunk": part.text,
                            "full_response_so_far": full_response,
                        })

                    # Handle function calls
                    if hasattr(part, "function_call") and part.function_call:
                        accumulated_function_calls.append(part.function_call)

            # Process any function calls
            if accumulated_function_calls and self._tool_manager:
                return await self._handle_gemini_tool_calls(
                    contents, accumulated_function_calls, config,
                    raw_model_parts=raw_model_parts,
                )

            # Normal text response
            self._conversation_history.append({"role": "assistant", "content": full_response})

            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": full_response,
                "success": True,
            })

            await self._publish_assistant_response_for_tts(full_response)
            return full_response

        except Exception as e:
            logger.error(f"Error in Gemini streaming request: {e}")
            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": full_response,
                "success": False,
                "error": str(e),
            })
            raise

    def _build_gemini_contents(self, new_message: str) -> list:
        """Convert conversation history to Gemini Content objects.

        Maps OpenAI-format history dicts to ``genai_types.Content`` objects.
        Skips tool call/result messages (these are handled inline during
        tool calling exchanges).
        """
        from google.genai import types as genai_types

        contents = []

        for msg in self._conversation_history:
            role = msg["role"]
            content_text = msg.get("content")

            if role == "user":
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=content_text)],
                ))
            elif role == "assistant" and content_text:
                # "assistant" → "model" in Gemini
                contents.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_text(text=content_text)],
                ))
            # role == "tool" or assistant with tool_calls: skip (ephemeral)

        # Add the new user message
        contents.append(genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=new_message)],
        ))

        return contents

    def _build_gemini_config(self) -> Any:
        """Build GenerateContentConfig for Gemini requests."""
        from google.genai import types as genai_types

        config_kwargs: Dict[str, Any] = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_tokens,
            "system_instruction": self._system_prompt,
        }

        # Add thinking config if enabled
        if self._thinking_enabled:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=self._thinking_level.upper(),
            )

        # Add tools if available (we handle function calls manually via ToolManager)
        if self._tool_manager and self._tool_manager.has_tools():
            config_kwargs["tools"] = self._tool_manager.get_tools_gemini_format()
            # Disable automatic function calling - we route through ToolManager/MCP
            config_kwargs["automatic_function_calling"] = (
                genai_types.AutomaticFunctionCallingConfig(disable=True)
            )

        return genai_types.GenerateContentConfig(**config_kwargs)

    async def _handle_gemini_tool_calls(
        self,
        contents: list,
        function_calls: list,
        config: Any,
        *,
        raw_model_parts: Optional[list] = None,
    ) -> Optional[str]:
        """Execute Gemini function calls and continue the conversation."""
        from google.genai import types as genai_types
        from ..tools.tool_types import ToolCall as ToolCallType

        self._tool_iterations += 1

        if self._tool_iterations > self._max_tool_iterations:
            logger.warning(f"Max tool iterations ({self._max_tool_iterations}) reached")
            error_msg = "I've reached the maximum number of tool calls. Please try rephrasing your request."
            self._conversation_history.append({"role": "assistant", "content": error_msg})
            self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
                "full_response": error_msg,
                "success": True,
            })
            self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                "role": "assistant",
                "content": error_msg,
            })
            return error_msg

        # Append model response to contents, preserving original parts
        # (including thought signatures required by thinking mode).
        if raw_model_parts:
            contents.append(genai_types.Content(role="model", parts=raw_model_parts))
        else:
            # Fallback: reconstruct from function_calls (no thinking mode)
            model_parts = [
                genai_types.Part.from_function_call(
                    name=fc.name,
                    args=dict(fc.args) if fc.args else {},
                )
                for fc in function_calls
            ]
            contents.append(genai_types.Content(role="model", parts=model_parts))

        # Record in OpenAI-format conversation history for persistence
        tool_calls_for_history = []
        for i, fc in enumerate(function_calls):
            tool_calls_for_history.append({
                "id": f"gemini_call_{i}",
                "function": {
                    "name": fc.name,
                    "arguments": json.dumps(dict(fc.args) if fc.args else {}),
                },
            })
        self._conversation_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls_for_history,
        })

        # Execute each function call through ToolManager
        response_parts = []
        for i, fc in enumerate(function_calls):
            arguments = dict(fc.args) if fc.args else {}
            call_id = f"gemini_call_{i}"

            tool_call = ToolCallType(id=call_id, name=fc.name, arguments=arguments)
            logger.info(f"Executing tool: {tool_call.name}")
            result = await self._tool_manager.execute_tool(tool_call)

            result_content = "\n".join(
                item.get("text", str(item)) for item in result.content
            )

            response_parts.append(
                genai_types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_content},
                )
            )

            # Record in OpenAI-format history
            self._conversation_history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_content,
            })

        # Add function responses as a user message (Google's convention)
        contents.append(genai_types.Content(role="user", parts=response_parts))

        # Continue conversation with tool results
        return await self._continue_gemini_after_tools(contents, config)

    async def _continue_gemini_after_tools(self, contents: list, config: Any) -> Optional[str]:
        """Continue Gemini conversation after tool execution."""
        client = self._get_genai_client()

        full_response = ""
        accumulated_function_calls: list = []
        raw_model_parts: list = []

        stream = await client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        )

        async for chunk in stream:
            if self._request_cancelled.is_set():
                return None

            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                raw_model_parts.append(part)

                if getattr(part, "thought", False):
                    logger.debug(f"Thinking: {getattr(part, 'text', '')[:80]}...")
                    continue

                if part.text:
                    full_response += part.text
                    self._event_bus.publish(EventType.LLM_RESPONSE_CHUNK, {
                        "chunk": part.text,
                        "full_response_so_far": full_response,
                    })

                if hasattr(part, "function_call") and part.function_call:
                    accumulated_function_calls.append(part.function_call)

        # Chained tool calls
        if accumulated_function_calls:
            return await self._handle_gemini_tool_calls(
                contents, accumulated_function_calls, config,
                raw_model_parts=raw_model_parts,
            )

        # Final response
        self._tool_iterations = 0
        self._conversation_history.append({"role": "assistant", "content": full_response})

        self._event_bus.publish(EventType.LLM_RESPONSE_COMPLETED, {
            "full_response": full_response,
            "success": True,
        })

        await self._publish_assistant_response_for_tts(full_response)
        return full_response

    # ── End Gemini-native methods ────────────────────────────────────

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

            # Publish assistant response for TTS processing (with summarization if needed)
            await self._publish_assistant_response_for_tts(full_response)

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
            # Error messages are short, no need to summarize
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

        # Publish assistant response for TTS processing (with summarization if needed)
        await self._publish_assistant_response_for_tts(full_response)

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
        if "summarize_long_responses" in kwargs:
            self._summarize_long_responses = kwargs["summarize_long_responses"]
        if "max_speech_words" in kwargs:
            self._max_speech_words = kwargs["max_speech_words"]

        if "thinking_enabled" in kwargs:
            self._thinking_enabled = kwargs["thinking_enabled"]
        if "thinking_level" in kwargs:
            self._thinking_level = kwargs["thinking_level"]

        # Reset clients to force recreation with new config
        self._openai_client = None
        self._async_openai_client = None
        self._genai_client = None

        logger.info(f"LLM provider configured: {provider.value} with model {self._model}")

    async def _publish_assistant_response_for_tts(self, full_response: str) -> None:
        """
        Publish assistant response for TTS, summarizing if too long.

        If the response exceeds max_speech_words and summarization is enabled,
        requests a summary from the LLM and publishes that for TTS instead.
        The full response is still shown in the UI/console.
        """
        word_count = len(full_response.split())

        if self._summarize_long_responses and word_count > self._max_speech_words:
            logger.info(f"Response too long for TTS ({word_count} words > {self._max_speech_words}), summarizing...")

            # Get a summary for TTS
            summary = await self._get_tts_summary(full_response)

            if summary:
                logger.info(f"Using summary for TTS ({len(summary.split())} words)")
                self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                    "role": "assistant",
                    "content": summary,
                    "is_summary": True,
                    "original_word_count": word_count
                })
            else:
                # Summarization failed, use original
                logger.warning("Summarization failed, using original response for TTS")
                self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                    "role": "assistant",
                    "content": full_response
                })
        else:
            # Response is short enough, use as-is
            self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                "role": "assistant",
                "content": full_response
            })

    async def _get_tts_summary(self, text: str) -> Optional[str]:
        """
        Get a brief summary of the text suitable for TTS.

        Returns a condensed version that captures the key points
        in under max_speech_words.
        """
        summary_prompt = (
            f"Summarize the following response in 2-3 sentences "
            f"(under {self._max_speech_words} words) for text-to-speech.\n"
            f"Keep the same tone and personality. Focus on the key points.\n\n"
            f"Response to summarize:\n{text}\n\nBrief summary:"
        )

        try:
            if self._provider == LLMProvider.GEMINI:
                from google.genai import types as genai_types

                client = self._get_genai_client()
                response = await client.aio.models.generate_content(
                    model=self._model,
                    contents=summary_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=200,
                    ),
                )
                return response.text.strip() if response.text else None
            else:
                client = self._get_async_openai_client()
                if not client:
                    return None
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=200,
                    stream=False,
                )
                summary = response.choices[0].message.content.strip()
                return summary if summary else None

        except Exception as e:
            logger.error(f"Error getting TTS summary: {e}")
            return None

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