"""Type definitions for tool system in GLaDOS 2.0."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class ToolSource(Enum):
    """Source of a tool registration."""
    MCP = "mcp"
    BUILTIN = "builtin"


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters
    required: List[str]
    source: ToolSource
    server_name: Optional[str] = None  # For MCP tools


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution."""
    call_id: str
    tool_name: str
    content: List[Dict[str, Any]]  # Content items with type and text/data
    is_error: bool = False
