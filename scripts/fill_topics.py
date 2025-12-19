import json
from typing import Any

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from tqdm import tqdm

from agent.graph import get_llm
from mcp_server.server import DATA_DIR


class TopicsList(BaseModel):
    """Model for structured output containing a list of strings."""

    topics: list[str]


llm = get_llm(
    llm_provider="openai", model_name="gpt-4o-mini", reasoning_effort="none"
).with_structured_output(TopicsList)

with open("src/mcp_server/data/topics.json") as f:
    old_topics = json.load(f)

prompt_template = """You are an AI assistant that extracts topics from sales interactions (emails or calls).

Here is a predefined list of known topics:
{predefined_topics}

Task:
1. Read the following transcript of a sales email or call.
2. Identify which of the predefined topics are mentioned.
3. If you detect a new topic that is not in the predefined list, include it as well.
4. Only include topics that are actually present in the transcript.
5. Return the result as a JSON object with a single key "topics" containing an array of strings (no extra text). Example: {{"topics": [Topic A, Topic B, New Topic]}}

Transcript:
\"\"\"
{transcript}
\"\"\"
"""
prompt = PromptTemplate(
    input_variables=["predefined_topics", "transcript"], template=prompt_template
)


def fill_call_topics(
    call: dict[str, Any], topics: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Fill missing subject in a call record using LLM."""
    if "topics" not in call or not call["topics"]:
        formatted_prompt = prompt.format(
            predefined_topics=topics, transcript=call["transcript"]
        )
        response = llm.invoke(formatted_prompt)
        call["topics"] = response.topics
        return call, response.topics
        # Update topics list
    return call, []


def fill_email_topics(
    email: dict[str, Any], topics: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Fill missing subject in an email record using LLM."""
    if "topics" not in email or not email["topics"]:
        formatted_prompt = prompt.format(
            predefined_topics=topics, transcript=email["content"]
        )
        response = llm.invoke(formatted_prompt)
        email["topics"] = response.topics
        # Update topics list
        return email, response.topics
    return email, []


# TODO: Use async processing to speed up
for account_file in tqdm(DATA_DIR.glob("account_*.json")):
    with open(account_file) as f:
        account_data = json.load(f)
    updated = False
    # Process calls
    for i, call in enumerate(account_data.get("calls", [])):
        updated_call, new_topics = fill_call_topics(call, old_topics)
        if new_topics:
            account_data["calls"][i] = updated_call
            updated = True
            old_topics = list(set(old_topics) | set(new_topics))
    # Process emails
    for i, email in enumerate(account_data.get("emails", [])):
        updated_email, new_topics = fill_email_topics(email, old_topics)
        if new_topics:
            account_data["emails"][i] = updated_email
            updated = True
            old_topics = list(set(old_topics) | set(new_topics))
    if updated:
        with open(account_file, "w") as f:
            json.dump(account_data, f, indent=2)

with open("src/mcp_server/data/topics.json", "w") as f:
    json.dump(old_topics, f, indent=2)
