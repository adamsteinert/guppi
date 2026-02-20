"""Tests for MCP tool discovery and execution via the worker loop.

Validates that:
1. Tools discovered from MCP servers are properly registered
2. Tool execution works when init and execution share the same event loop
3. The full chat-loop flow (LLM → tool call → execution → follow-up) works
   with tools initialized on the worker loop
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from glados2.core.event_bus import EventBus, EventType
from glados2.core.state_manager import StateManager, AppState
from glados2.core.worker_loop import WorkerLoop
from glados2.config.config_manager import ToolsConfig, MCPServerConfig
from glados2.llm.llm_manager import LLMManager, LLMProvider
from glados2.tools.tool_manager import ToolManager
from glados2.tools.tool_types import ToolCall, ToolResult, ToolSource, ToolSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_mcp_tool(name: str, description: str, params: dict, required: list | None = None):
    """Create a mock MCP tool object matching what MCPClient._discover_tools returns."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = {
        "type": "object",
        "properties": params,
        "required": required or [],
    }
    return tool


def _register_echo_tool(tm: ToolManager):
    """Register a simple echo tool for testing."""
    async def echo_fn(name: str, args: dict):
        class Result:
            content = [MagicMock(text=f"echo: {args.get('message', '')}")]
        return Result()

    tm.register_builtin_tool(
        name="echo",
        function=echo_fn,
        description="Echoes the input message",
        parameters={"message": {"type": "string", "description": "Message to echo"}},
        required=["message"],
    )


def _make_llm_manager(event_bus, state_manager, tool_manager=None, worker_loop=None):
    mgr = LLMManager(event_bus, state_manager, tool_manager=tool_manager, worker_loop=worker_loop)
    mgr.configure_provider(LLMProvider.OPENAI, model="gpt-4", api_key="test-key")
    mgr.set_system_prompt("You are a test assistant.")
    return mgr


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None):
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
def worker():
    w = WorkerLoop(name="test")
    w.start()
    yield w
    w.stop()


# ---------------------------------------------------------------------------
# Tests: MCP Tool Discovery
# ---------------------------------------------------------------------------

class TestMCPToolDiscovery:
    """Verify tools from MCP servers are discovered and registered."""

    @pytest.mark.asyncio
    async def test_mcp_tools_registered_after_connect(self, event_bus, worker):
        """When an MCP server connects and returns tools, they should appear
        in the ToolManager registry."""
        mock_tools = [
            _make_mock_mcp_tool(
                "list_projects",
                "List all projects",
                {"filter": {"type": "string", "description": "Optional filter"}},
            ),
            _make_mock_mcp_tool(
                "get_project",
                "Get project details",
                {"name": {"type": "string", "description": "Project name"}},
                required=["name"],
            ),
        ]

        config = ToolsConfig(
            enabled=True,
            mcp_servers=[
                MCPServerConfig(name="test-server", command="echo", args=["test"], enabled=True)
            ],
        )
        tm = ToolManager(event_bus, config)

        # Mock MCPClient so no real subprocess is spawned.
        # get_tools() is a regular (sync) method, so use MagicMock.
        with patch("glados2.tools.tool_manager.MCPClient") as MockMCPClient:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.get_tools = MagicMock(return_value=mock_tools)
            MockMCPClient.return_value = mock_client

            await worker.run_await(tm.initialize())

        assert tm.has_tools()
        names = tm.get_tool_names()
        assert "list_projects" in names
        assert "get_project" in names

    @pytest.mark.asyncio
    async def test_mcp_tools_available_in_openai_format(self, event_bus, worker):
        """Discovered MCP tools should be convertible to OpenAI format."""
        mock_tools = [
            _make_mock_mcp_tool(
                "search",
                "Search for items",
                {"query": {"type": "string", "description": "Search query"}},
                required=["query"],
            ),
        ]

        config = ToolsConfig(
            enabled=True,
            mcp_servers=[
                MCPServerConfig(name="test-server", command="echo", args=[], enabled=True)
            ],
        )
        tm = ToolManager(event_bus, config)

        with patch("glados2.tools.tool_manager.MCPClient") as MockMCPClient:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.get_tools = MagicMock(return_value=mock_tools)
            MockMCPClient.return_value = mock_client

            await worker.run_await(tm.initialize())

        tools = tm.get_tools_openai_format()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"
        assert tools[0]["function"]["parameters"]["required"] == ["query"]

    @pytest.mark.asyncio
    async def test_failed_mcp_connection_does_not_crash(self, event_bus, worker):
        """If an MCP server fails to connect, initialization should continue."""
        config = ToolsConfig(
            enabled=True,
            mcp_servers=[
                MCPServerConfig(name="broken-server", command="nonexistent", args=[], enabled=True)
            ],
        )
        tm = ToolManager(event_bus, config)

        with patch("glados2.tools.tool_manager.MCPClient") as MockMCPClient:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=False)
            MockMCPClient.return_value = mock_client

            await worker.run_await(tm.initialize())

        assert not tm.has_tools()

    @pytest.mark.asyncio
    async def test_disabled_mcp_server_skipped(self, event_bus, worker):
        """Disabled MCP servers should not be connected to."""
        config = ToolsConfig(
            enabled=True,
            mcp_servers=[
                MCPServerConfig(name="disabled-server", command="echo", args=[], enabled=False)
            ],
        )
        tm = ToolManager(event_bus, config)

        with patch("glados2.tools.tool_manager.MCPClient") as MockMCPClient:
            await worker.run_await(tm.initialize())
            MockMCPClient.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Tool Execution on Worker Loop
# ---------------------------------------------------------------------------

class TestToolExecutionOnWorkerLoop:
    """Verify tool execution works when init and execution share the same loop."""

    @pytest.mark.asyncio
    async def test_builtin_tool_on_worker_loop(self, event_bus, worker):
        """Built-in tools should work when initialized and executed on the worker loop."""
        config = ToolsConfig(enabled=True, mcp_servers=[])
        tm = ToolManager(event_bus, config)
        _register_echo_tool(tm)

        await worker.run_await(tm.initialize())

        call = ToolCall(id="call_1", name="echo", arguments={"message": "hello"})
        result = await worker.run_await(tm.execute_tool(call))

        assert not result.is_error
        assert "echo: hello" in result.content[0]["text"]

    @pytest.mark.asyncio
    async def test_mcp_tool_execution_on_worker_loop(self, event_bus, worker):
        """MCP tools should be callable when init and execution are on the same loop."""
        mock_tools = [
            _make_mock_mcp_tool(
                "greet",
                "Greet someone",
                {"name": {"type": "string"}},
                required=["name"],
            ),
        ]

        config = ToolsConfig(
            enabled=True,
            mcp_servers=[
                MCPServerConfig(name="test-server", command="echo", args=[], enabled=True)
            ],
        )
        tm = ToolManager(event_bus, config)

        # Mock the MCP client and its call_tool method
        mock_call_result = MagicMock()
        mock_call_result.content = [MagicMock(text="Hello, Ada!")]

        with patch("glados2.tools.tool_manager.MCPClient") as MockMCPClient:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.get_tools = MagicMock(return_value=mock_tools)
            mock_client.call_tool = AsyncMock(return_value=mock_call_result)
            MockMCPClient.return_value = mock_client

            await worker.run_await(tm.initialize())

        call = ToolCall(id="call_1", name="greet", arguments={"name": "Ada"})
        result = await worker.run_await(tm.execute_tool(call))

        assert not result.is_error
        assert "Hello, Ada!" in result.content[0]["text"]

    @pytest.mark.asyncio
    async def test_tool_events_published_during_execution(self, event_bus, worker):
        """Tool execution should publish started/completed events."""
        config = ToolsConfig(enabled=True, mcp_servers=[])
        tm = ToolManager(event_bus, config)
        _register_echo_tool(tm)

        await worker.run_await(tm.initialize())

        events = []
        event_bus.subscribe(EventType.TOOL_EXECUTION_STARTED, lambda d: events.append(("started", d)))
        event_bus.subscribe(EventType.TOOL_EXECUTION_COMPLETED, lambda d: events.append(("completed", d)))

        call = ToolCall(id="call_1", name="echo", arguments={"message": "test"})
        await worker.run_await(tm.execute_tool(call))

        assert len(events) == 2
        assert events[0][0] == "started"
        assert events[1][0] == "completed"
        assert events[1][1]["tool_name"] == "echo"


# ---------------------------------------------------------------------------
# Tests: Full Chat Loop with Tools via Worker Loop
# ---------------------------------------------------------------------------

class TestChatLoopWithWorkerLoop:
    """End-to-end: LLM call → tool call → execution on worker loop → follow-up."""

    @pytest.mark.asyncio
    async def test_full_tool_call_flow_on_worker_loop(self, event_bus, state_manager, worker):
        """Complete flow: LLM returns tool call, tool executes on worker loop,
        result fed back to LLM, final response returned."""
        config = ToolsConfig(enabled=True, mcp_servers=[])
        tm = ToolManager(event_bus, config)
        _register_echo_tool(tm)
        await worker.run_await(tm.initialize())

        mgr = _make_llm_manager(event_bus, state_manager, tool_manager=tm, worker_loop=worker)

        call_count = 0
        all_params = []

        async def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            all_params.append(dict(kwargs))

            if call_count == 1:
                # First call: LLM returns a tool call
                async def stream():
                    yield _make_stream_chunk(
                        tool_calls=[
                            _make_tool_call_delta(0, "call_1", "echo", '{"message": "world"}')
                        ]
                    )
                return stream()
            else:
                # Second call: LLM returns final text
                async def stream():
                    yield _make_stream_chunk(content="The echo said: world", finish_reason="stop")
                return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            result = await mgr.send_message("echo world", streaming=True)

        assert call_count == 2
        assert result == "The echo said: world"

        # Verify tools were in first request
        assert "tools" in all_params[0]
        assert all_params[0]["tools"][0]["function"]["name"] == "echo"

        # Verify tool result was in follow-up request
        followup_msgs = all_params[1]["messages"]
        tool_msgs = [m for m in followup_msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "echo: world" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_worker_loop_dispatch_from_event_bus(self, event_bus, state_manager, worker):
        """When LLMManager receives a MESSAGE_RECEIVED event, it should
        dispatch the LLM call to the worker loop (not spawn a new thread)."""
        config = ToolsConfig(enabled=True, mcp_servers=[])
        tm = ToolManager(event_bus, config)
        _register_echo_tool(tm)
        await worker.run_await(tm.initialize())

        mgr = _make_llm_manager(event_bus, state_manager, tool_manager=tm, worker_loop=worker)

        execution_loop = None

        async def fake_create(**kwargs):
            nonlocal execution_loop
            execution_loop = asyncio.get_running_loop()
            async def stream():
                yield _make_stream_chunk(content="OK", finish_reason="stop")
            return stream()

        state_manager.set_state(AppState.IDLE)

        with patch.object(mgr, "_get_async_openai_client") as mock_client_fn:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = fake_create
            mock_client_fn.return_value = mock_client

            # Trigger via event bus (like the real app does)
            event_bus.publish(EventType.MESSAGE_RECEIVED, {
                "role": "user",
                "content": "hello",
            })

            # Wait for the worker loop to process
            await asyncio.sleep(1.0)

        # Verify the LLM call ran on the worker loop, not a random new loop
        assert execution_loop is worker.loop
