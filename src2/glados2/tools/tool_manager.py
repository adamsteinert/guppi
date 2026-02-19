"""Tool management system for GLaDOS 2.0."""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from mcp import StdioServerParameters

from ..core.event_bus import EventBus, EventType
from ..config.config_manager import ToolsConfig, MCPServerConfig
from .mcp_client import MCPClient
from .tool_types import ToolSpec, ToolCall, ToolResult, ToolSource


class ToolManager:
    """
    Manages tool registration and execution for GLaDOS 2.0.

    Handles both MCP server tools and built-in tools,
    providing a unified interface for tool discovery and execution.
    """

    def __init__(self, event_bus: EventBus, config: Optional[ToolsConfig] = None):
        self._event_bus = event_bus
        self._config = config or ToolsConfig()

        # Tool registry
        self._tools: Dict[str, ToolSpec] = {}
        self._tool_functions: Dict[str, Callable] = {}

        # MCP clients
        self._mcp_clients: Dict[str, MCPClient] = {}

    @property
    def tool_timeout(self) -> float:
        return self._config.tool_timeout_seconds

    @property
    def max_iterations(self) -> int:
        return self._config.max_tool_iterations

    async def initialize(self) -> None:
        """Initialize the tool manager and connect to MCP servers."""
        if not self._config.enabled:
            logger.info("Tools disabled in configuration")
            return

        for server_config in self._config.mcp_servers:
            if not server_config.enabled:
                logger.debug(f"MCP server {server_config.name} is disabled")
                continue

            await self._connect_mcp_server(server_config)

        logger.info(f"Tool manager initialized with {len(self._tools)} tools")

    async def _connect_mcp_server(self, config: MCPServerConfig) -> None:
        """Connect to a single MCP server and register its tools."""
        server_name = config.name

        if not config.command:
            logger.warning(f"MCP server {server_name} has no command configured")
            return

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None
        )

        client = MCPClient(server_name, server_params, self._event_bus)

        if await client.connect():
            self._mcp_clients[server_name] = client

            # Register tools from this server
            for tool in client.get_tools():
                self._register_mcp_tool(server_name, tool, client)
        else:
            logger.warning(f"Failed to connect to MCP server: {server_name}")

    def _register_mcp_tool(self, server_name: str, tool: Any, client: MCPClient) -> None:
        """Register a tool from an MCP server."""
        tool_name = getattr(tool, 'name', str(tool))
        description = getattr(tool, 'description', '')
        input_schema = getattr(tool, 'inputSchema', {})

        spec = ToolSpec(
            name=tool_name,
            description=description,
            parameters=input_schema.get('properties', {}),
            required=input_schema.get('required', []),
            source=ToolSource.MCP,
            server_name=server_name
        )

        self._tools[tool_name] = spec

        # Create a closure that captures the client reference
        async def tool_executor(name: str, arguments: dict, _client=client) -> Any:
            return await _client.call_tool(name, arguments)

        self._tool_functions[tool_name] = tool_executor

        logger.debug(f"Registered MCP tool: {tool_name} from {server_name}")

    def register_builtin_tool(self,
                              name: str,
                              function: Callable,
                              description: str,
                              parameters: Dict[str, Any],
                              required: Optional[List[str]] = None) -> None:
        """Register a built-in tool."""
        spec = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            required=required or [],
            source=ToolSource.BUILTIN
        )

        self._tools[name] = spec
        self._tool_functions[name] = function

        logger.debug(f"Registered built-in tool: {name}")

    def get_tools_openai_format(self) -> List[Dict]:
        """
        Get all tools in OpenAI API format.

        This format is used by the chat completions API
        for tool calling.
        """
        tool_specs = []

        for name, spec in self._tools.items():
            tool_specs.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': spec.description,
                    'parameters': {
                        'type': 'object',
                        'properties': spec.parameters,
                        'required': spec.required
                    }
                }
            })

        return tool_specs

    def get_tools_gemini_format(self) -> List[Any]:
        """
        Get all tools as google-genai FunctionDeclaration objects.

        Uses OpenAPI schema format as recommended by Google for Gemini 3.
        """
        from google.genai import types

        declarations = []
        for name, spec in self._tools.items():
            # Convert JSON Schema properties to genai Schema objects
            schema_properties = {}
            for param_name, param_schema in spec.parameters.items():
                param_type = param_schema.get("type", "string").upper()
                schema_properties[param_name] = types.Schema(
                    type=types.Type(param_type),
                    description=param_schema.get("description", ""),
                )

            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=spec.description,
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties=schema_properties,
                        required=spec.required,
                    ),
                )
            )

        return declarations

    def has_tools(self) -> bool:
        """Check if any tools are registered."""
        return len(self._tools) > 0

    def get_tool_names(self) -> List[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.

        Publishes events for tool execution visibility.
        """
        tool_name = tool_call.name
        call_id = tool_call.id

        if tool_name not in self._tools:
            error_msg = f"Unknown tool: {tool_name}"
            self._event_bus.publish(EventType.TOOL_EXECUTION_ERROR, {
                "tool_name": tool_name,
                "call_id": call_id,
                "error": error_msg
            })
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                content=[{"type": "text", "text": error_msg}],
                is_error=True
            )

        # Publish start event
        self._event_bus.publish(EventType.TOOL_EXECUTION_STARTED, {
            "tool_name": tool_name,
            "arguments": tool_call.arguments,
            "call_id": call_id
        })

        start_time = time.time()

        try:
            # Execute with timeout
            tool_func = self._tool_functions[tool_name]

            result = await asyncio.wait_for(
                tool_func(tool_name, tool_call.arguments),
                timeout=self._config.tool_timeout_seconds
            )

            duration_ms = (time.time() - start_time) * 1000

            # Convert MCP result to our format
            content = self._extract_content(result)

            tool_result = ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                content=content,
                is_error=False
            )

            self._event_bus.publish(EventType.TOOL_EXECUTION_COMPLETED, {
                "tool_name": tool_name,
                "call_id": call_id,
                "result": content,
                "duration_ms": duration_ms
            })

            return tool_result

        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {self._config.tool_timeout_seconds}s"
            logger.warning(f"Timeout executing tool {tool_name}")
            self._event_bus.publish(EventType.TOOL_EXECUTION_ERROR, {
                "tool_name": tool_name,
                "call_id": call_id,
                "error": error_msg
            })
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                content=[{"type": "text", "text": error_msg}],
                is_error=True
            )

        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error(f"Error executing tool {tool_name}: {e}")
            self._event_bus.publish(EventType.TOOL_EXECUTION_ERROR, {
                "tool_name": tool_name,
                "call_id": call_id,
                "error": error_msg
            })
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                content=[{"type": "text", "text": error_msg}],
                is_error=True
            )

    def _extract_content(self, result: Any) -> List[Dict[str, Any]]:
        """Extract content from MCP result into our format."""
        content = []

        if hasattr(result, 'content'):
            for item in result.content:
                if hasattr(item, 'text'):
                    content.append({"type": "text", "text": item.text})
                elif hasattr(item, 'data'):
                    content.append({"type": "data", "data": item.data})
                else:
                    content.append({"type": "text", "text": str(item)})
        elif isinstance(result, str):
            content.append({"type": "text", "text": result})
        elif isinstance(result, dict):
            content.append({"type": "text", "text": str(result)})
        else:
            content.append({"type": "text", "text": str(result)})

        return content if content else [{"type": "text", "text": ""}]

    async def cleanup(self) -> None:
        """Disconnect all MCP servers and clean up."""
        for client in self._mcp_clients.values():
            await client.disconnect()
        self._mcp_clients.clear()
        self._tools.clear()
        self._tool_functions.clear()
        logger.info("Tool manager cleaned up")
