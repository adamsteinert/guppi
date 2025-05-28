import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import Any, List


class MCPClient:
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session = None
        self._client = None
        self.exit_stack = AsyncExitStack()
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
        await self.exit_stack.aclose()
        self._initialized = False

    def is_ready(self) -> bool:
        """Check if the client is ready to process requests."""
        return self._initialized and self.session is not None

    async def initialize(self):
        """Manually initialize the client when not using 'async with'."""
        if not self._initialized:
            await self.exit_stack.__aenter__()
            self._initialized = True

    async def connect(self):
        """Establishes connection to MCP server"""
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write, logging_callback=self.handle_log_message)
        #await session.set_logging_level(types.LoggingLevel.DEBUG)
        self.session = await session.__aenter__()
        await self.session.initialize()

    # Define a callback to handle log messages from the server
    async def handle_log_message(self, notification: types.LoggingMessageNotification):
        """Handle log messages from the server"""
        print(f"[{notification.level.upper()}] {notification.data}")
        if notification.logger:
            print(f"Logger: {notification.logger}")

    async def cleanup(self):
        """Clean up resources."""
        if self._initialized:
            await self.__aexit__(None, None, None)

    async def handle_progress(notification: types.ProgressNotification):
        """Handle progress notifications from the server"""
        token = notification.progressToken
        progress = notification.progress
        total = notification.total
        message = getattr(notification, 'message', '')

        if total:
            percentage = (progress / total) * 100
            print(f"Progress [{token}]: {progress}/{total} ({percentage:.1f}%) - {message}")
        else:
            print(f"Progress [{token}]: {progress} - {message}")

    async def get_available_tools(self) -> List[Any]:
        """List available tools"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        tools = await self.session.list_tools()
        # The response format may vary depending on the server implementation
        # We need to handle it carefully. For now it's handled for mcp git server
        # TODO: Handle other servers
        try:
            _, _, tools_list = tools
            return tools_list
        except Exception as e:
            print(f"Error parsing tools response: {e}")
            print(f"Raw tools response: {tools}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool with given arguments"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return result