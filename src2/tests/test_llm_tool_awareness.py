"""Tests for LLM agent awareness of MCP tools.

Verifies that:
1. Tools are included in LLM API request params when available
2. Tools are NOT included when no ToolManager is provided
3. Tool call results are fed back to the LLM for follow-up reasoning
4. Tool chaining respects iteration limits
5. Tool manager initialization runs on the correct event loop
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from glados2.core.event_bus import EventBus, EventType
from glados2.core.state_manager import StateManager, AppState
from glados2.config.config_manager import ToolsConfig
from glados2.llm.llm_manager import LLMManager, LLMProvider
from glados2.tools.tool_manager import ToolManager
from glados2.tools.tool_types import ToolCall, ToolResult, ToolSource, ToolSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_manager(event_bus, state_manager, tool_manager=None):
    """Create an LLMManager configured for OpenAI provider (uses AsyncOpenAI)."""
    mgr = LLMManager(event_bus, state_manager, tool_manager=tool_manager)
    mgr.configure_provider(
        LLMProvider.OPENAI,
        model="gpt-4",
        api_key="test-key",
    )
    mgr.set_system_prompt("You are a test assistant.")
    return mgr


def _register_echo_tool(tool_manager: ToolManager):
    """Register a simple echo tool on the manager."""
    async def echo_fn(name: str, args: dict):
        class Result:
            content = [MagicMock(text=f"echo: {args.get('message', '')}")]
        return Result()

    tool_manager.register_builtin_tool(
        name="echo",
        function=echo_fn,
        description="Echoes the input message",
        parameters={"message": {"type": "string", "description": "Message to echo"}},
        required=["message"],
    )


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
    """Build a mock streaming chunk matching the OpenAI delta format."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def _make_tool_call_delta(index, tc_id=None, name=None, arguments=None):
    """Build a mock tool_call delta fragment."""
    tc = MagicMock()
    tc.index = index
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def state_manager(event_bus):
    return StateManager(event_bus)


@pytest.fixture
def tools_config():
    return ToolsConfig(enabled=True, mcp_servers=[], tool_timeout_seconds=5.0, max_tool_iterations=3)


@pytest.fixture
def tool_manager(event_bus, tools_config):
    tm = ToolManager(event_bus, tools_config)
    _register_echo_tool(tm)
    return tm


# ---------------------------------------------------------------------------
# Tests: Tool inclusion in request params
# ---------------------------------------------------------------------------

class TestToolsInRequestParams:
    """Verify tools are correctly included/excluded from LLM API calls."""

    @pytest.mark.asyncio
    async def test_tools_included_when_tool_manager_has_tools(
        self, event_bus, state_manager, tool_manager
    ):
        """When a ToolManager with registered tools is provided, the request
        params sent to the OpenAI client MUST include a 'tools' key."""
        mgr = _make_llm_manager(event_bus, state_manager, tool_manager)

        captured_params = {}

        async def fake_create(**kwargs):
            captured_params.update(kwargs)
            # Return an async iterator that yields a single content chunk
            async def stream():
                yield _make_stream_chunk(content="Hello!", finish_reason="stop")
            return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            await mgr.send_message("hi", streaming=True)

        assert "tools" in captured_params, "tools key missing from request params"
        assert captured_params["tool_choice"] == "auto"

        # Verify the tool spec structure
        tools = captured_params["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "echo"
        assert "parameters" in tools[0]["function"]

    @pytest.mark.asyncio
    async def test_tools_excluded_when_no_tool_manager(
        self, event_bus, state_manager
    ):
        """When no ToolManager is provided, request params must NOT have tools."""
        mgr = _make_llm_manager(event_bus, state_manager, tool_manager=None)

        captured_params = {}

        async def fake_create(**kwargs):
            captured_params.update(kwargs)
            async def stream():
                yield _make_stream_chunk(content="Hello!", finish_reason="stop")
            return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            await mgr.send_message("hi", streaming=True)

        assert "tools" not in captured_params, "tools key should not be present"
        assert "tool_choice" not in captured_params

    @pytest.mark.asyncio
    async def test_tools_excluded_when_tool_manager_empty(
        self, event_bus, state_manager
    ):
        """When ToolManager exists but has no tools, request params must NOT have tools."""
        empty_tm = ToolManager(event_bus, ToolsConfig(enabled=True))
        mgr = _make_llm_manager(event_bus, state_manager, tool_manager=empty_tm)

        captured_params = {}

        async def fake_create(**kwargs):
            captured_params.update(kwargs)
            async def stream():
                yield _make_stream_chunk(content="Hello!", finish_reason="stop")
            return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            await mgr.send_message("hi", streaming=True)

        assert "tools" not in captured_params


# ---------------------------------------------------------------------------
# Tests: Tool execution and result feedback
# ---------------------------------------------------------------------------

class TestToolExecutionFeedback:
    """Verify tool calls are executed and results fed back to LLM."""

    @pytest.mark.asyncio
    async def test_tool_call_triggers_execution_and_followup(
        self, event_bus, state_manager, tool_manager
    ):
        """When the LLM responds with a tool call, the tool must be executed
        and a follow-up request made with the tool result in messages."""
        mgr = _make_llm_manager(event_bus, state_manager, tool_manager)

        call_count = 0
        all_captured_params = []

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            all_captured_params.append(dict(kwargs))

            if call_count == 1:
                # First call: LLM returns a tool call
                async def stream():
                    yield _make_stream_chunk(
                        tool_calls=[
                            _make_tool_call_delta(
                                index=0,
                                tc_id="call_1",
                                name="echo",
                                arguments='{"message": "test"}'
                            )
                        ]
                    )
                return stream()
            else:
                # Second call (after tool result): LLM returns text
                async def stream():
                    yield _make_stream_chunk(content="The echo said: test", finish_reason="stop")
                return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            result = await mgr.send_message("echo test", streaming=True)

        # LLM was called twice: once for initial request, once after tool result
        assert call_count == 2, f"Expected 2 LLM calls, got {call_count}"

        # The follow-up request must contain the tool result message
        followup_messages = all_captured_params[1]["messages"]
        tool_role_msgs = [m for m in followup_messages if m.get("role") == "tool"]
        assert len(tool_role_msgs) == 1, "Expected one tool result message in follow-up"
        assert "echo: test" in tool_role_msgs[0]["content"]

        # Final response from LLM
        assert result == "The echo said: test"

    @pytest.mark.asyncio
    async def test_tool_result_in_conversation_history(
        self, event_bus, state_manager, tool_manager
    ):
        """Tool call assistant messages and tool results must be persisted
        in conversation history for future context."""
        mgr = _make_llm_manager(event_bus, state_manager, tool_manager)

        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                async def stream():
                    yield _make_stream_chunk(
                        tool_calls=[
                            _make_tool_call_delta(0, "call_1", "echo", '{"message": "hi"}')
                        ]
                    )
                return stream()
            else:
                async def stream():
                    yield _make_stream_chunk(content="Done.", finish_reason="stop")
                return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            await mgr.send_message("say hi", streaming=True)

        history = mgr.get_conversation_history()

        # Should contain: user msg, assistant tool_calls msg, tool result, assistant final
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

        # The tool result should contain the echo output
        tool_msgs = [m for m in history if m["role"] == "tool"]
        assert any("echo: hi" in m["content"] for m in tool_msgs)


# ---------------------------------------------------------------------------
# Tests: Tool chaining / iteration limits
# ---------------------------------------------------------------------------

class TestToolChaining:
    """Verify tool chaining and max iteration enforcement."""

    @pytest.mark.asyncio
    async def test_max_tool_iterations_enforced(
        self, event_bus, state_manager, tool_manager
    ):
        """When the LLM keeps calling tools beyond the max iteration limit,
        it should stop and return an error message."""
        mgr = _make_llm_manager(event_bus, state_manager, tool_manager)
        mgr._max_tool_iterations = 2

        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            # Always return a tool call (simulating infinite loop)
            async def stream():
                yield _make_stream_chunk(
                    tool_calls=[
                        _make_tool_call_delta(0, f"call_{call_count}", "echo", '{"message": "loop"}')
                    ]
                )
            return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            result = await mgr.send_message("loop forever", streaming=True)

        # Should have stopped after max_tool_iterations + 1 calls
        # (initial call + max_tool_iterations continuation calls)
        assert call_count <= 3, f"Expected at most 3 LLM calls, got {call_count}"
        assert result is not None
        assert "maximum" in result.lower() or "max" in result.lower()


# ---------------------------------------------------------------------------
# Tests: Event loop lifecycle (the bug we fixed)
# ---------------------------------------------------------------------------

class TestEventLoopLifecycle:
    """Verify MCP connections work when initialized on the correct loop."""

    @pytest.mark.asyncio
    async def test_tool_manager_works_on_same_loop(self, event_bus):
        """Tool manager initialized and used on the same event loop should work."""
        config = ToolsConfig(enabled=True, mcp_servers=[], tool_timeout_seconds=5.0)
        tm = ToolManager(event_bus, config)
        _register_echo_tool(tm)

        # Initialize and execute on the same loop (the happy path)
        await tm.initialize()

        call = ToolCall(id="call_1", name="echo", arguments={"message": "hello"})
        result = await tm.execute_tool(call)

        assert not result.is_error
        assert "echo: hello" in result.content[0]["text"]

    def test_tool_manager_fails_across_loops(self, event_bus):
        """Demonstrate that async objects created on one loop cannot be used
        on another — this is the bug that the event loop fix addresses.

        We simulate the old pattern: initialize in asyncio.run() (loop A),
        then try to use in a different asyncio.run() (loop B).
        For builtin tools this still works (no I/O bound to a loop), but
        the pattern itself is the problem for MCP subprocess connections.
        """
        config = ToolsConfig(enabled=True, mcp_servers=[], tool_timeout_seconds=5.0)
        tm = ToolManager(event_bus, config)
        _register_echo_tool(tm)

        # Loop A: initialize
        asyncio.run(tm.initialize())

        # Loop B: execute — builtin tools still work since they don't
        # hold loop-bound resources, but this pattern is what broke
        # MCP connections (subprocess stdio streams are loop-bound).
        call = ToolCall(id="call_1", name="echo", arguments={"message": "hello"})
        result = asyncio.run(tm.execute_tool(call))

        # This succeeds for builtins, but would fail for real MCP tools
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_openai_format_available_after_init(self, event_bus):
        """After initialization, tools should be available in OpenAI format."""
        config = ToolsConfig(enabled=True, mcp_servers=[])
        tm = ToolManager(event_bus, config)
        _register_echo_tool(tm)

        await tm.initialize()

        tools = tm.get_tools_openai_format()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "echo"
        assert tm.has_tools() is True


# ---------------------------------------------------------------------------
# Tests: Worker loop integration
# ---------------------------------------------------------------------------

class TestWorkerLoopIntegration:
    """Verify LLMManager dispatches to the worker loop when available."""

    @pytest.mark.asyncio
    async def test_schedule_async_task_uses_worker_loop(self, event_bus, state_manager):
        """_schedule_async_task should run on the worker loop, not a random thread."""
        from glados2.core.worker_loop import WorkerLoop

        worker = WorkerLoop(name="test")
        worker.start()
        try:
            mgr = _make_llm_manager(event_bus, state_manager)
            mgr._worker_loop = worker

            execution_loop = None

            async def probe():
                nonlocal execution_loop
                execution_loop = asyncio.get_running_loop()

            mgr._schedule_async_task(probe())

            # Give the worker loop time to execute
            await asyncio.sleep(0.5)

            assert execution_loop is worker.loop
        finally:
            worker.stop()

    @pytest.mark.asyncio
    async def test_schedule_async_task_fallback_without_worker(self, event_bus, state_manager):
        """Without a worker loop, _schedule_async_task should spawn a thread."""
        mgr = _make_llm_manager(event_bus, state_manager)
        # No worker loop set (the default)

        execution_loop = None
        done = asyncio.Event()

        async def probe():
            nonlocal execution_loop
            execution_loop = asyncio.get_running_loop()
            # Signal from the worker thread won't set our event directly,
            # so we just store the loop reference.

        mgr._schedule_async_task(probe())

        # Give the background thread time
        await asyncio.sleep(0.5)

        # It ran on SOME loop, but not the test's loop
        assert execution_loop is not None
        assert execution_loop is not asyncio.get_running_loop()
