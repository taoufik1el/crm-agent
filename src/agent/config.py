# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long

import os
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel

# LLM settings
# TODO: Build LLM settings config in another file

# MCP Server settings
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8002/mcp")


class Message(BaseModel):
    """Message structure for chat."""

    role: str
    content: str


class Response(BaseModel):
    """Response to user."""

    response: str


ToolName = Literal[
    "filter_by_topics",
    "filter_by_keywords",
    "filter_by_date",
    "take_last_element",
    "compute_len",
]


class Call(BaseModel):
    """Call data model."""

    date: str
    content: str
    topics: list[str]
    interaction_type: str = "call"


class Email(BaseModel):
    """Email data model."""

    date: str
    content: str
    topics: list[str]
    interaction_type: str = "email"


class ToolCall(BaseModel):
    """A single tool call in a plan."""

    tool: ToolName
    params: dict[str, str | list[str]] | None = None


class PlanSeries(BaseModel):
    """A series of tool calls forming a plan."""

    steps: list[ToolCall]
    title: str


class PlannerOutput(BaseModel):
    """Output structure for the planner."""

    plans: list[PlanSeries]


class AgentState(MessagesState):
    """State class to track the agent execution."""

    # Input
    user_query: str
    account_id: int

    # True if baseline
    baseline: bool

    # The full calls and emails
    calls: list[Call]
    emails: list[Email]

    # Planning Agent's results for filtering calls and emails to get relevant context
    plans: list[PlanSeries]

    # Retrieved context
    context: str

    # Variable to indicate end of execution when no data found in MCP
    end: bool

    # Output
    final_response: str

    # Tracking
    messages: Annotated[list[BaseMessage], add_messages]
