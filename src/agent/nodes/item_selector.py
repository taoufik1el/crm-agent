from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import AgentState, ItemIndices

ITEM_SELECTOR_SYSTEM_PROMPT = """You are an intelligent agent tasked with selecting the most relevant calls and emails to answer a user question.

Inputs:
1. Query: A specific question about an account or prospect.
2. Call summaries: A list of call summaries containing information about the phone call like Current ERP, Pain Points, Decision Makers, Budget, Next Steps, Metrics, etc.
3. Email subjects: A list of email subjects records.

Your task:
- Examine each summary and Subject.
- Determine which calls and emails are **likely to contain the information needed to answer the question** even if that information is implicit, indirect, or expressed as weak signals.
- Ignore items that are irrelevant to the question.
- Only return the indices of relevant calls and emails (0-based).

Output format (strictly follow this JSON structure): {"calls": [list of relevant call indices], "emails": [list of relevant email indices]}

Rules:
1. Only include items that could actually contain information to answer the question.
2. Do not include irrelevant items.
3. Always output valid JSON. No explanations or extra text.
4. If no calls or no emails are present, return an empty list for that field."""


def create_item_selector_node(
    llm: BaseChatModel,
) -> Callable[[AgentState], dict[str, Any]]:
    """Create the item selector node function."""

    def item_selector_node(state: AgentState) -> dict[str, Any]:
        """Item selector node - selects relevant calls and emails using GPT-4o-mini.

        This node:
        1. Receives the user query and account info from state
        2. Constructs a prompt with the context
        3. Calls GPT-4o-mini to select relevant items

        Args:
            state: The current agent state with context

        Returns:
            Updated state with selected item indices
        """
        user_query = state["user_query"]
        selector_llm = llm.with_structured_output(ItemIndices)

        messages = [
            SystemMessage(content=ITEM_SELECTOR_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Query: {user_query}\n{state['account_info'].summaries_as_str()}"
                )
            ),
        ]
        # Generate the response
        response = selector_llm.invoke(messages)
        return {"selected_item_indices": response}

    return item_selector_node
