# pylint: disable=line-too-long
"""Agent Graph Definition.

This module defines the LangGraph workflow for the agent.
The graph follows the structure:
    __start__ → supervisor → final_answer → __end__

The supervisor node:
- Uses a pre-determined plan (fetch all transcripts and emails)
- Calls MCP tools directly to retrieve context

The final_answer node:
- Uses GPT-4o-mini to generate the response based on the context
"""

from langgraph.graph import END, START, StateGraph

from agent.config import AgentState
from agent.llm_utils import get_llm
from agent.nodes import (
    create_final_answer_node,
    create_mcp_node,
    create_plan_executer_node,
    create_planner_node,
)

# Initialize LLM
openai_llm = get_llm(
    llm_provider="openai", model_name="gpt-4o-mini", reasoning_effort="none"
)
openai_reasoning_llm = get_llm(
    llm_provider="openai", model_name="gpt-5-mini", reasoning_effort="low"
)


def create_agent_graph() -> StateGraph:
    """Create and compile the agent graph.

    The graph structure is:
        __start__ → supervisor → final_answer → __end__

    Returns:
        The compiled agent graph
    """
    # Initialize the graph with our state
    workflow = StateGraph(AgentState)

    # # Add nodes
    # workflow.add_node(node="question_router", action=create_question_router_node(openai_llm))
    workflow.add_node(node="mcp", action=create_mcp_node())
    workflow.add_node(node="planner", action=create_planner_node(openai_reasoning_llm))
    workflow.add_node(node="plan_executer", action=create_plan_executer_node())
    workflow.add_node(node="final_answer", action=create_final_answer_node(openai_llm))

    # # Define the edges
    # workflow.add_edge(start_key=START, end_key="question_router")
    workflow.add_edge(start_key=START, end_key="mcp")
    workflow.add_conditional_edges(
        "mcp", lambda state: state.get("end"), path_map={True: END, False: "planner"}
    )
    workflow.add_conditional_edges(
        "planner",
        lambda state: state.get("end"),
        path_map={True: END, False: "plan_executer"},
    )
    workflow.add_edge(start_key="plan_executer", end_key="final_answer")
    workflow.add_edge(start_key="final_answer", end_key=END)
    # Compile the graph
    graph = workflow.compile()

    return graph


# Create the agent instance
agent = create_agent_graph()
