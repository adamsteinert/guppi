"""MCP server client for GLaDOS 2.0."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any, List, Optional

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.event_bus import EventBus, EventType


class MCPClient:
    """
    MCP server client for GLaDOS 2.0.

    Manages connection to a single MCP server and provides
    tool discovery and execution capabilities.
    """

    def __init__(self,
                 server_name: str,
                 server_params: StdioServerParameters,
                 event_bus: EventBus):
        self._server_name = server_name
        self._server_params = server_params
        self._event_bus = event_bus
        self._session: Optional[ClientSession] = None
        self._client = None
        self._exit_stack = AsyncExitStack()
        self._initialized = False
        self._tools: List[Any] = []

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def is_connected(self) -> bool:
        return self._initialized and self._session is not None

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            await self._exit_stack.__aenter__()

            # Create stdio client
            self._client = stdio_client(self._server_params)
            read, write = await self._client.__aenter__()

            # Create session
            session = ClientSession(read, write)
            self._session = await session.__aenter__()
            await self._session.initialize()

            self._initialized = True

            # Discover tools
            await self._discover_tools()

            self._event_bus.publish(EventType.MCP_SERVER_CONNECTED, {
                "server_name": self._server_name
            })

            logger.info(f"Connected to MCP server: {self._server_name}")
            return True

        except FileNotFoundError as e:
            logger.error(f"MCP server command not found for {self._server_name}: {e}")
            return False
        except PermissionError as e:
            logger.error(f"Permission denied running MCP server {self._server_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self._server_name}: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._initialized:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(f"Error during MCP client cleanup for {self._server_name}: {e}")

            self._initialized = False
            self._session = None
            self._client = None

            self._event_bus.publish(EventType.MCP_SERVER_DISCONNECTED, {
                "server_name": self._server_name
            })

            logger.info(f"Disconnected from MCP server: {self._server_name}")

    async def _discover_tools(self) -> None:
        """Discover available tools from the server."""
        if not self._session:
            return

        try:
            result = await self._session.list_tools()

            # Handle different response formats
            # The MCP library may return tools in various formats
            if hasattr(result, 'tools'):
                self._tools = result.tools
            else:
                # Fallback parsing (tuple format from some servers)
                try:
                    _, _, tools_list = result
                    self._tools = tools_list
                except (TypeError, ValueError):
                    # Try treating result as iterable of tools
                    self._tools = list(result) if result else []

            self._event_bus.publish(EventType.MCP_TOOLS_DISCOVERED, {
                "server_name": self._server_name,
                "tools": [self._tool_to_dict(t) for t in self._tools]
            })

            logger.info(f"Discovered {len(self._tools)} tools from {self._server_name}")

        except Exception as e:
            logger.error(f"Error discovering tools from {self._server_name}: {e}")

    def _tool_to_dict(self, tool: Any) -> dict:
        """Convert MCP tool to dictionary."""
        return {
            "name": getattr(tool, 'name', str(tool)),
            "description": getattr(tool, 'description', ''),
            "inputSchema": getattr(tool, 'inputSchema', {})
        }

    def get_tools(self) -> List[Any]:
        """Get list of available tools."""
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Execute a tool with given arguments."""
        if not self._session:
            raise RuntimeError(f"Not connected to MCP server: {self._server_name}")

        return await self._session.call_tool(name, arguments=arguments)
