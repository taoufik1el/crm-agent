# pylint: disable=line-too-long
"""Agent Main Entry Point.

This module provides the main interface to run the agent.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from langgraph.graph import StateGraph
from pydantic import BaseModel

from agent.config import AgentState

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class AgentInput(BaseModel):
    """Input schema for the agent."""

    user_query: str
    account_id: int


class AgentOutput(BaseModel):
    """Output schema for the agent."""

    response: str


def run_agent(
    agent: StateGraph, user_query: str, account_id: int, baseline: bool = False
) -> str:
    """Run the agent with the given user query and account ID.

    Args:
        agent: The agent graph to run
        user_query: The user's question
        account_id: The account ID to query data for
        baseline: Whether to run in baseline mode

    Returns:
        The agent's response as a string
    """
    initial_state = AgentState(
        user_query=user_query,
        account_id=account_id,
        baseline=baseline,
        calls=[],
        emails=[],
        plans=[],
        context="",
        end=False,
        final_response="",
        messages=[],
    )

    # Run the agent
    result = agent.invoke(initial_state)
    return result["final_response"]  # type: ignore[no-any-return]


async def stream_agent(
    agent: StateGraph, user_query: str, account_id: int, baseline: bool = False
) -> AsyncGenerator[str]:  # type: ignore[type-arg]
    """Run the agent in streaming mode, yielding tokens as they are generated.

    Args:
        agent: The agent graph to run
        user_query: The user's question
        account_id: The account ID to query data for
        baseline: Whether to run in baseline mode
    Yields:
        Tokens from the agent's response as they are generated
    """
    initial_state = AgentState(
        user_query=user_query,
        account_id=account_id,
        baseline=baseline,
        calls=[],
        emails=[],
        plans=[],
        context="",
        end=False,
        final_response="",
        messages=[],
    )

    # Stream tokens from the agent
    for msg_chunk, metadata in agent.stream(initial_state, stream_mode="messages"):
        for token in msg_chunk.content:
            if metadata.get("langgraph_node") != "final_answer":
                continue
            yield token.get("text", "")
            await asyncio.sleep(0)
