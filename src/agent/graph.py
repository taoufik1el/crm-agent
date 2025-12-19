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

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent.config import FINAL_ANSWER_MODEL, LLM_PROVIDER, AgentState
from agent.nodes import (
    create_final_answer_node,
    create_item_selector_node,
    create_mcp_node,
    create_question_router_node,
)


def get_llm() -> BaseChatModel:
    """Get the LLM based on the configured provider."""
    if LLM_PROVIDER == "google":
        return ChatGoogleGenerativeAI(model=FINAL_ANSWER_MODEL, temperature=0)
    else:
        return ChatOpenAI(model=FINAL_ANSWER_MODEL, temperature=0)


def create_agent_graph(llm: BaseChatModel) -> StateGraph:
    """Create and compile the agent graph.

    The graph structure is:
        __start__ → supervisor → final_answer → __end__

    Returns:
        The compiled agent graph
    """
    # Initialize the graph with our state
    workflow = StateGraph(AgentState)

    # # Add nodes
    workflow.add_node(node="question_router", action=create_question_router_node(llm))
    workflow.add_node(node="mcp", action=create_mcp_node())
    workflow.add_node(node="item_selector", action=create_item_selector_node(llm))
    workflow.add_node(node="final_answer", action=create_final_answer_node(llm))

    # # Define the edges
    workflow.add_edge(start_key=START, end_key="question_router")
    workflow.add_edge(start_key="question_router", end_key="mcp")
    # workflow.add_edge(start_key="mcp", end_key="item_selector")
    workflow.add_conditional_edges(
        "mcp",
        lambda state: state.get("end"),
        path_map={True: END, False: "item_selector"},
    )
    workflow.add_edge(start_key="item_selector", end_key="final_answer")
    workflow.add_edge(start_key="final_answer", end_key=END)
    # Compile the graph
    graph = workflow.compile()

    return graph


# Initialize LLM
llm = get_llm()
# Create the agent instance
agent = create_agent_graph(llm)
