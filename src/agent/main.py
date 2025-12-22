# pylint: disable=line-too-long
"""Agent Main Entry Point.

This module provides the main interface to run the agent.
"""

import logging
from typing import Any

from pydantic import BaseModel

from agent.graph import agent

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
    user_query: str, account_id: int, baseline: bool = False
) -> tuple[str, dict[str, Any]]:
    """Run the agent with the given user query and account ID.

    Args:
        user_query: The user's question
        account_id: The account ID to query data for
        baseline: Whether to run in baseline mode

    Returns:
        The agent's response as a string
    """
    # Initialize the state
    initial_state = {
        "user_query": user_query,
        "account_id": account_id,
        "baseline": baseline,
        "calls": [],
        "emails": [],
        "plans": [],
        "context": "",
        "end": False,
        "final_response": "",
        "messages": [],
        "llm_usage": {},
    }

    # Run the agent
    result = agent.invoke(initial_state)
    return result["final_response"], result["llm_usage"]


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

    response, _ = run_agent(args.query, args.account_id)

    logging.info("Response:")
    logging.info(response)


if __name__ == "__main__":
    main()
