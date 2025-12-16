# pylint: disable=line-too-long
"""
Agent Main Entry Point

This module provides the main interface to run the agent.
"""

from typing import Optional
from pydantic import BaseModel

from graph import agent
from config import AgentState, RetrievedContext


class AgentInput(BaseModel):
    """Input schema for the agent."""

    user_query: str
    account_id: int


class AgentOutput(BaseModel):
    """Output schema for the agent."""

    response: str


def run_agent(user_query: str, account_id: int) -> str:
    """
    Run the agent with the given user query and account ID.

    Args:
        user_query: The user's question
        account_id: The account ID to query data for

    Returns:
        The agent's response as a string
    """
    # Initialize the state
    initial_state = {
        "user_query": user_query,
        "account_id": account_id,
        "messages": [],
        "plan": None,
        "context": RetrievedContext(),
        "final_response": "",
    }

    # Run the agent
    result = agent.invoke(initial_state)

    return result["final_response"]


def main():
    """Main function for testing the agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the ML Engineer Agent")
    parser.add_argument("--query", "-q", type=str, required=True, help="User query")
    parser.add_argument(
        "--account-id", "-a", type=int, required=True, help="Account ID"
    )

    args = parser.parse_args()

    print(f"Running agent with query: {args.query}")
    print(f"Account ID: {args.account_id}")
    print("-" * 50)

    response = run_agent(args.query, args.account_id)

    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
