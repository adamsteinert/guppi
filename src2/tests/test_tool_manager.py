"""Tests for the tool manager system."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from glados2.core.event_bus import EventBus, EventType
from glados2.config.config_manager import ToolsConfig, MCPServerConfig
from glados2.tools.tool_manager import ToolManager
from glados2.tools.tool_types import ToolSpec, ToolCall, ToolResult, ToolSource


class TestToolManager:
    """Tests for ToolManager class."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def tools_config(self):
        return ToolsConfig(
            enabled=True,
            mcp_servers=[],
            tool_timeout_seconds=5.0,
            max_tool_iterations=3
        )

    @pytest.fixture
    def tool_manager(self, event_bus, tools_config):
        return ToolManager(event_bus, tools_config)

    def test_initialization(self, tool_manager):
        """Test tool manager initializes correctly."""
        assert tool_manager.has_tools() is False
        assert tool_manager.get_tool_names() == []
        assert tool_manager.tool_timeout == 5.0
        assert tool_manager.max_iterations == 3

    def test_register_builtin_tool(self, tool_manager):
        """Test registering a built-in tool."""
        async def my_tool(name: str, args: dict):
            return f"Result: {args.get('input', '')}"

        tool_manager.register_builtin_tool(
            name="test_tool",
            function=my_tool,
            description="A test tool",
            parameters={"input": {"type": "string", "description": "Input value"}},
            required=["input"]
        )

        assert tool_manager.has_tools()
        assert "test_tool" in tool_manager.get_tool_names()

    def test_get_tools_openai_format(self, tool_manager):
        """Test generating OpenAI-compatible tool specs."""
        async def my_tool(name: str, args: dict):
            return "result"

        tool_manager.register_builtin_tool(
            name="test_tool",
            function=my_tool,
            description="A test tool",
            parameters={"input": {"type": "string"}},
            required=["input"]
        )

        tools = tool_manager.get_tools_openai_format()

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test_tool"
        assert tools[0]["function"]["description"] == "A test tool"
        assert tools[0]["function"]["parameters"]["type"] == "object"
        assert "input" in tools[0]["function"]["parameters"]["properties"]
        assert tools[0]["function"]["parameters"]["required"] == ["input"]

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, tool_manager, event_bus):
        """Test successful tool execution."""
        async def echo_tool(name: str, args: dict):
            class MockResult:
                content = [MagicMock(text=args.get("message", ""))]
            return MockResult()

        tool_manager.register_builtin_tool(
            name="echo",
            function=echo_tool,
            description="Echo the input",
            parameters={"message": {"type": "string"}}
        )

        # Track events
        events_received = []
        event_bus.subscribe(EventType.TOOL_EXECUTION_STARTED, lambda d: events_received.append(("started", d)))
        event_bus.subscribe(EventType.TOOL_EXECUTION_COMPLETED, lambda d: events_received.append(("completed", d)))

        call = ToolCall(id="call_123", name="echo", arguments={"message": "hello"})
        result = await tool_manager.execute_tool(call)

        assert not result.is_error
        assert result.tool_name == "echo"
        assert result.call_id == "call_123"
        assert len(result.content) > 0
        assert "hello" in result.content[0]["text"]

        # Check events were published
        assert len(events_received) == 2
        assert events_received[0][0] == "started"
        assert events_received[1][0] == "completed"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tool_manager, event_bus):
        """Test executing an unknown tool returns error."""
        events_received = []
        event_bus.subscribe(EventType.TOOL_EXECUTION_ERROR, lambda d: events_received.append(d))

        call = ToolCall(id="call_123", name="nonexistent_tool", arguments={})
        result = await tool_manager.execute_tool(call)

        assert result.is_error
        assert "Unknown tool" in result.content[0]["text"]
        assert len(events_received) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, tool_manager, event_bus):
        """Test tool execution timeout."""
        async def slow_tool(name: str, args: dict):
            await asyncio.sleep(10)  # Longer than timeout
            return "result"

        # Use a very short timeout for testing
        tool_manager._config.tool_timeout_seconds = 0.1

        tool_manager.register_builtin_tool(
            name="slow_tool",
            function=slow_tool,
            description="A slow tool",
            parameters={}
        )

        events_received = []
        event_bus.subscribe(EventType.TOOL_EXECUTION_ERROR, lambda d: events_received.append(d))

        call = ToolCall(id="call_123", name="slow_tool", arguments={})
        result = await tool_manager.execute_tool(call)

        assert result.is_error
        assert "timed out" in result.content[0]["text"]
        assert len(events_received) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, tool_manager, event_bus):
        """Test tool execution error handling."""
        async def failing_tool(name: str, args: dict):
            raise ValueError("Tool failed!")

        tool_manager.register_builtin_tool(
            name="failing_tool",
            function=failing_tool,
            description="A failing tool",
            parameters={}
        )

        events_received = []
        event_bus.subscribe(EventType.TOOL_EXECUTION_ERROR, lambda d: events_received.append(d))

        call = ToolCall(id="call_123", name="failing_tool", arguments={})
        result = await tool_manager.execute_tool(call)

        assert result.is_error
        assert "Tool failed!" in result.content[0]["text"]
        assert len(events_received) == 1

    @pytest.mark.asyncio
    async def test_cleanup(self, tool_manager):
        """Test cleanup clears all tools."""
        async def my_tool(name: str, args: dict):
            return "result"

        tool_manager.register_builtin_tool(
            name="test_tool",
            function=my_tool,
            description="A test tool",
            parameters={}
        )

        assert tool_manager.has_tools()

        await tool_manager.cleanup()

        assert not tool_manager.has_tools()


class TestToolTypes:
    """Tests for tool type dataclasses."""

    def test_tool_spec_creation(self):
        """Test ToolSpec dataclass."""
        spec = ToolSpec(
            name="test",
            description="A test tool",
            parameters={"input": {"type": "string"}},
            required=["input"],
            source=ToolSource.BUILTIN
        )

        assert spec.name == "test"
        assert spec.description == "A test tool"
        assert spec.source == ToolSource.BUILTIN
        assert spec.server_name is None

    def test_tool_call_creation(self):
        """Test ToolCall dataclass."""
        call = ToolCall(
            id="call_123",
            name="my_tool",
            arguments={"key": "value"}
        )

        assert call.id == "call_123"
        assert call.name == "my_tool"
        assert call.arguments == {"key": "value"}

    def test_tool_result_creation(self):
        """Test ToolResult dataclass."""
        result = ToolResult(
            call_id="call_123",
            tool_name="my_tool",
            content=[{"type": "text", "text": "Result"}],
            is_error=False
        )

        assert result.call_id == "call_123"
        assert result.tool_name == "my_tool"
        assert not result.is_error
        assert result.content[0]["text"] == "Result"
