# pylint: disable=line-too-long
"""Supervisor Node.

This node is the first entrypoint of the agent. It:
1. Uses the pre-determined plan to fetch all calls and emails
2. Calls the MCP tools directly to retrieve the context
3. Passes control to the final_answer node
"""

import asyncio
import json
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent.config import MCP_SERVER_URL, AccountInfo, AgentState, DataSource

# Thread pool for running async code from sync context
_executor = ThreadPoolExecutor(max_workers=4)


def _run_async(coro: Coroutine[Any, Any, str]) -> str:
    """Run async coroutine from sync context, handling existing event loops."""
    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)

    # There's a running loop (e.g., FastAPI), run in a new thread

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro))
        return future.result()


class MCPClient:
    """Simple MCP client wrapper for tool calls using streamable_http."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Async method to call an MCP tool."""
        async with streamable_http_client(self.server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

                # Extract text from result
                if result.content:
                    return "\n".join(
                        block.text for block in result.content if hasattr(block, "text")
                    )
                return ""

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Sync wrapper to call an MCP tool."""
        try:
            return _run_async(self._call_tool(tool_name, arguments))
        except Exception as e:
            return f"Error calling {tool_name}: {str(e)}"


# Initialize MCP client
mcp_client = MCPClient(MCP_SERVER_URL)


def create_mcp_node() -> Callable[[AgentState], dict[str, Any]]:
    """Create the MCP node function."""

    def mcp_node(state: AgentState) -> dict[str, Any]:
        """Supervisor node - orchestrates the agent execution.

        This node:
        1. Uses the fixed plan (fetch all calls and emails)
        2. Calls the MCP tools to retrieve data
        3. Stores the retrieved context in state
        """
        account_id = state["account_id"]
        # Execute the plan: call MCP tools
        if state["data_source"] in [DataSource.calls]:
            mcp_data = json.loads(
                mcp_client.call_tool("calls", {"account_id": account_id})
            )
        elif state["data_source"] in [DataSource.emails]:
            mcp_data = json.loads(
                mcp_client.call_tool("emails", {"account_id": account_id})
            )
        else:  # both
            mcp_data = json.loads(
                mcp_client.call_tool("calls_emails", {"account_id": account_id})
            )

        account_info = AccountInfo.model_validate(mcp_data)
        found = mcp_data.get("found", False)
        if not found:
            return {"final_response": mcp_data["error"], "end": True}
        return {"account_info": account_info, "end": False}

    return mcp_node
