# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long

import os
from enum import StrEnum
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# LLM settings
# Set LLM_PROVIDER to "google" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# Google Gemini settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = "gemini-2.5-flash"

# Active model based on provider
FINAL_ANSWER_MODEL = GOOGLE_MODEL if LLM_PROVIDER == "google" else OPENAI_MODEL

# MCP Server settings
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8002/mcp")


class Message(BaseModel):
    """Message structure for chat."""

    role: str
    content: str


class Response(BaseModel):
    """Response to user."""

    response: str


class Step(TypedDict):
    """A single step in the plan."""

    tool: str  # Name of the tool to use: either transcripts or emails
    description: str  # Description of what this step does


class Plan(BaseModel):
    """Plan to follow - pre-determined for this agent."""

    steps: list[Step] = Field(description="Steps to execute in order")


class RetrievedContext(BaseModel):
    """Context retrieved from MCP tools."""

    transcripts: str | None = None
    emails: str | None = None


class DataSource(StrEnum):
    """Data source enumeration."""

    calls = "calls"
    emails = "emails"
    both = "calls_emails"


class ItemIndices(BaseModel):
    """Indices of selected items."""

    calls: list[int]
    emails: list[int]


class Call(BaseModel):
    """Call transcript data model."""

    date: str
    call_name: str
    transcript: str
    summary: str
    crm_fields: list[dict[str, str]]

    def crm_as_str(self) -> str:
        """Convert CRM fields to a string representation."""
        return "\n".join(
            f"  - {field['key']}: {field['value']}" for field in self.crm_fields
        )

    def summary_as_str(self) -> str:
        """Return summary as a string."""
        return f"   date: {self.date}\n   summary: {self.summary}"

    def transcript_as_str(self) -> str:
        """Return transcript as a string."""
        return f"   date: {self.date}\n   transcript: {self.transcript}"


class Email(BaseModel):
    """Email data model."""

    date: str
    subject: str
    content: str

    def subject_as_str(self) -> str:
        """Return subject as a string."""
        return f"   date: {self.date}\n   subject: {self.subject}"

    def content_as_str(self) -> str:
        """Return content as a string."""
        return f"   date: {self.date}\n   content: {self.content}"


class AccountInfo(BaseModel):
    """Account information data model."""

    account_id: int
    tenant_name: str
    account_name: str
    calls: list[Call]
    emails: list[Email]

    def crms_as_str(self) -> str:
        """Convert all CRM fields from calls to a string representation."""
        return "\n".join(
            f"Call CRMs {e}:\n{call.crm_as_str()}" for e, call in enumerate(self.calls)
        )

    def summaries_as_str(self, selected_ids: list[int] | None = None) -> str:
        """Convert all summaries from calls to a string representation."""
        selected_ids = selected_ids or list(range(len(self.calls)))
        return "\n".join(
            f"Call Summary {i}, Date {self.calls[i].date}:\n{self.calls[i].summary_as_str()}"
            for i in selected_ids
        )

    def transcripts_as_str(self, selected_ids: list[int] | None = None) -> str:
        """Convert all transcripts from calls to a string representation."""
        selected_ids = selected_ids or list(range(len(self.calls)))
        return "\n".join(
            f"Call Transcript {i}. Date {self.calls[i].date}:\n{self.calls[i].transcript_as_str()}"
            for i in selected_ids
        )

    def subjects_as_str(self) -> str:
        """Convert all subjects from emails to a string representation."""
        return "\n".join(
            f"Email Subject {e}:\n{email.subject_as_str()}"
            for e, email in enumerate(self.emails)
        )

    def contents_as_str(self, selected_ids: list[int] | None = None) -> str:
        """Convert all contents from emails to a string representation."""
        selected_ids = selected_ids or list(range(len(self.emails)))
        return "\n".join(
            f"Email Content {i}. Date {self.emails[i].date}:\n{self.emails[i].content_as_str()}"
            for i in selected_ids
        )

    def context(self, selected_ids: ItemIndices | None = None) -> str:
        """Get context strings for selected calls and emails."""
        if selected_ids is None:
            return "\n".join([self.transcripts_as_str(), self.contents_as_str()])
        return "\n".join(
            [
                self.transcripts_as_str(selected_ids.calls),
                self.contents_as_str(selected_ids.emails),
            ]
        )

    # get crms + subjects as str
    def crms_and_subjects_as_str(self) -> str:
        """Convert all CRM fields from calls and subjects from emails to a string representation."""
        crms_str = self.crms_as_str()
        subjects_str = self.subjects_as_str()
        return "\n".join([crms_str, subjects_str])


class AgentState(MessagesState):
    """State class to track the agent execution."""

    # Input
    user_query: str
    account_id: int

    # calls or emails or both
    data_source: DataSource | None

    # Retrieved data from MCP server
    account_info: AccountInfo | None

    # Context as ids of selected calls and emails
    selected_item_indices: ItemIndices | None

    # Process control
    end: bool

    # Output
    final_response: str

    # Tracking
    messages: Annotated[list[BaseMessage], add_messages]
