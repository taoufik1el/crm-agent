# pylint: disable=line-too-long
"""Agent Main Entry Point.

This module provides the main interface to run the agent.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from agent.config import AgentState
from agent.graph import create_agent_graph

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


def run_agent(user_query: str, account_id: int, baseline: bool = False) -> str:
    """Run the agent with the given user query and account ID.

    Args:
        user_query: The user's question
        account_id: The account ID to query data for
        baseline: Whether to run in baseline mode

    Returns:
        The agent's response as a string
    """
    # Initialize the state
    agent = create_agent_graph(streaming=False)

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
    user_query: str, account_id: int, baseline: bool = False
) -> AsyncGenerator[str]:  # type: ignore[type-arg]
    """Run the agent in streaming mode, yielding tokens as they are generated.

    Args:
        user_query: The user's question
        account_id: The account ID to query data for
        baseline: Whether to run in baseline mode
    Yields:
        Tokens from the agent's response as they are generated
    """
    # Initialize the state
    agent = create_agent_graph(streaming=True)

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


def main() -> None:
    """Main function for testing the agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the ML Engineer Agent")
    parser.add_argument("--query", "-q", type=str, required=True, help="User query")
    parser.add_argument(
        "--account-id", "-a", type=int, required=True, help="Account ID"
    )

    args = parser.parse_args()

    logging.info(f"Running agent with query: {args.query}")
    logging.info(f"Account ID: {args.account_id}")
    logging.info("-" * 50)

    response = run_agent(args.query, args.account_id)

    logging.info("Response:")
    logging.info(response)


if __name__ == "__main__":
    main()
