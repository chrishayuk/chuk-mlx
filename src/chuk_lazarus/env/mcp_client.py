"""
MCP Tool Client - Interface for calling MCP (Model Context Protocol) tools.

This module provides the MCPToolClient class for executing tool calls,
with support for caching and both sync/async execution.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class MCPToolClient:
    """
    Interface for calling MCP tools.

    In production, this would connect to actual MCP servers.
    For training, can be mocked or use cached responses.

    Usage:
        client = MCPToolClient()
        client.register_tool("calculator", calc_handler)
        result = client.call_sync("calculator", {"expression": "2+2"})
    """

    def __init__(self, tools: Dict[str, Callable] = None):
        """
        Initialize the MCP client.

        Args:
            tools: Dictionary mapping tool names to handler functions
        """
        self.tools = tools or {}
        self.cache: Dict[str, Any] = {}
        self.call_count: int = 0

    def register_tool(self, name: str, handler: Callable):
        """
        Register a tool handler.

        Args:
            name: Name of the tool
            handler: Callable that implements the tool logic
        """
        self.tools[name] = handler

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Name of the tool to remove

        Returns:
            True if tool was removed, False if it didn't exist
        """
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def list_tools(self) -> list:
        """Return list of registered tool names."""
        return list(self.tools.keys())

    def clear_cache(self):
        """Clear the result cache."""
        self.cache.clear()

    async def call(
        self,
        tool_name: str,
        args: Dict,
        use_cache: bool = True
    ) -> Any:
        """
        Call an MCP tool asynchronously.

        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool
            use_cache: Whether to use cached results

        Returns:
            Result from the tool

        Raises:
            ValueError: If tool is not registered
        """
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for {tool_name}")
            return self.cache[cache_key]

        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        self.call_count += 1
        handler = self.tools[tool_name]

        # Handle async or sync handlers
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**args)
        else:
            result = handler(**args)

        if use_cache:
            self.cache[cache_key] = result

        return result

    def call_sync(self, tool_name: str, args: Dict, use_cache: bool = True) -> Any:
        """
        Call an MCP tool synchronously.

        This is a convenience wrapper for non-async contexts.

        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool
            use_cache: Whether to use cached results

        Returns:
            Result from the tool
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.call(tool_name, args, use_cache))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dictionary with call count, cache size, registered tools
        """
        return {
            "call_count": self.call_count,
            "cache_size": len(self.cache),
            "registered_tools": len(self.tools),
            "tool_names": list(self.tools.keys()),
        }
