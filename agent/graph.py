# pylint: disable=line-too-long
"""
Agent Graph Definition

This module defines the LangGraph workflow for the agent.
The graph follows the structure:
    __start__ → supervisor → final_answer → __end__

The supervisor node:
- Uses a pre-determined plan (fetch all transcripts and emails)
- Calls MCP tools directly to retrieve context

The final_answer node:
- Uses GPT-4o-mini to generate the response based on the context
"""

from langgraph.graph import StateGraph, END, START

from config import AgentState
from nodes import supervisor_node, final_answer_node


def create_agent_graph() -> StateGraph:
    """
    Create and compile the agent graph.

    The graph structure is:
        __start__ → supervisor → final_answer → __end__

    Returns:
        The compiled agent graph
    """
    # Initialize the graph with our state
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("final_answer", final_answer_node)

    # Define the edges
    # START → supervisor
    workflow.add_edge(START, "supervisor")

    # supervisor → final_answer
    workflow.add_edge("supervisor", "final_answer")

    # final_answer → END
    workflow.add_edge("final_answer", END)

    # Compile the graph
    graph = workflow.compile()

    return graph


# Create the agent instance
agent = create_agent_graph()
