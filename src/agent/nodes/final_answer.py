# pylint: disable=line-too-long
"""Final Answer Node.

This node receives the user query and retrieved context,
then uses GPT-4o-mini to generate the final response.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.config import AgentState
from agent.llm_utils import safe_run_llm, update_llm_usage

# System prompt for the final answer LLM
FINAL_ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about accounts based on their interaction history.

You have access to the following context about the account:

{context}

Based on this context, answer the user's question accurately and concisely.
If the information needed to answer the question is not available in the context, say so clearly.
"""


def create_final_answer_node(
    llm: BaseChatModel,
) -> Callable[[AgentState], dict[str, Any]]:
    """Create the final answer node function."""

    def final_answer_node(state: AgentState) -> dict[str, Any]:
        """Final answer node - generates the response using GPT-4o-mini.

        This node:
        1. Receives the user query and retrieved context from state
        2. Constructs a prompt with the context
        3. Calls GPT-4o-mini to generate the final answer

        Args:
            state: The current agent state with context

        Returns:
            Updated state with the final response
        """
        user_query = state["user_query"]
        context = state["context"]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FINAL_ANSWER_SYSTEM_PROMPT),
                ("human", "User's question: {user_query}"),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        response, llm_success, usage = safe_run_llm(
            chain, llm_inputs={"context": context, "user_query": user_query}
        )
        global_llm_usage = update_llm_usage(state, usage)
        if not llm_success:
            # If LLM failed, return an error response and end the agent workflow
            return {
                "final_response": "Error: Unable to generate plan at this time.",
                "end": True,
                "llm_usage": global_llm_usage,
            }
        return {"final_response": response, "llm_usage": global_llm_usage}

    return final_answer_node
