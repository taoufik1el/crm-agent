from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent.config import AgentState, DataSource


class RouteDecision(BaseModel):
    """Model for the routing decision output."""

    sources: DataSource = Field(
        ..., description="Data sources to use for answering the question"
    )


QUESTION_ROUTER_SYSTEM_PROMPT = """You are a routing agent in a sales question-answering system.

Your task is to decide which data sources must be queried to answer the user's question.

Data sources:

1. "calls"
   - Phone call records between sales staff and contacts
   - Includes call names, transcripts, summaries, dates, and CRM fields
   - Used for spoken discussions, call outcomes, gatekeepers, decision-makers, objections, or verbal agreements

2. "emails"
   - Emails sent by sales staff to accounts or prospects
   - Includes subject lines and email content
   - Used for written communication, follow-ups, confirmations, urgent requests, blockers, or formal information

3. "calls_emails"
    - Both phone call records and email records

Rules:
1. The "sources" field must be one of "calls", "emails", or "calls_emails".
2. Use "calls" if the question is best answered from call transcripts or CRM fields.
3. Use "emails" if the question is best answered from email that the sender is a sales man and receiver is the prospect.
4. Use "calls_emails" only if both sources are likely needed.
5. Never invent new source names.
6. Always output **valid JSON/Pydantic-compatible syntax**, no explanations."""


def create_question_router_node(
    llm: BaseChatModel,
) -> Callable[[AgentState], dict[str, Any]]:
    """Create the question router node function."""

    def question_router_node(state: AgentState) -> dict[str, Any]:
        user_query = state["user_query"]

        # Create the prompt
        prompt = ChatPromptTemplate.from_messages(
            [("system", QUESTION_ROUTER_SYSTEM_PROMPT), ("human", "{user_query}")]
        )

        # TODO: handle exceptions and fallback if LLM fails
        classifier_llm = llm.with_structured_output(RouteDecision)
        # Create the chain
        chain = prompt | classifier_llm
        # Generate the response
        response = chain.invoke({"user_query": user_query})

        return {"data_source": response.sources}

    return question_router_node
