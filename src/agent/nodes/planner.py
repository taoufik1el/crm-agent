from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from agent.config import AgentState, PlannerOutput
from agent.llm_utils import safe_run_llm, update_llm_usage

ALLOWED_TOPICS = [
    "Budget",
    "Next Steps",
    "Timeline",
    "Pain Points",
    "Competitors Mentioned",
    "Negative Sentiment",
    "Result",
    "Churn Risk",
    "ESG Audit Pressure",
    "Positive Sentiment",
    "ROI Justification",
    "Decision Factors",
    "Solution Fit",
]

PLANNER_SYSTEM_PROMPT = f"""You are a planning engine.

Your job is to convert a user question into one or more
PARALLEL tool plans.

Rules:
- You ONLY output structured JSON following the provided schema.
- You NEVER answer the question directly.
- You NEVER access data.
- Each plan is an ordered list of tool calls.
- Plans may run in parallel.
- compute_len MUST be the last step in a plan.
- If the pan series starts with take_last_element its MUST be the only step.
- Topics MUST come from the allowed topic list.
- Dates MUST be ISO format YYYY-MM-DD.
- Propose a title for each plan summarizing its intent. Example: "Count of Positive Sentiment Interactions".

Available tools:
1. filter_by_topics(topics: list[str]) the list contains only topics from the allowed topic list. the parameter topics must be a valid list of str
2. filter_by_date(operator: Literal["=", "<", ">"], date: str) date must be in ISO format YYYY-MM-DD.
3. take_last_n(n: int) n must be a positive integer
4. take_last_element()
5. compute_len()

Allowed topics:
[{", ".join(ALLOWED_TOPICS)}]
Interpretation rules:
- "latest", "most recent", "last" → take_last_element
- "how many" → compute_len
- sentiment-related questions → use Positive/Negative Sentiment topics
- problems/issues → Pain Points
- before DATE → filter_by_date "<"
- after DATE → filter_by_date ">"

Output format:
{{
  "plans": [
    {{ "steps": [ {{ "tool": "...", "params": {{...}} }} ], "title": "..." }},
  ]
}}
"""


def create_planner_node(llm: BaseChatModel) -> Callable[[AgentState], dict[str, Any]]:
    """Create a planner node that generates tool plans based on the user question."""

    def planner_node(state: AgentState) -> dict[str, Any]:
        if state["baseline"]:
            # For baseline, return empty plans, meaning all data will be fetched without filtering
            return {"plans": []}
        question = state["user_query"]
        response, llm_success, usage = safe_run_llm(
            llm.with_structured_output(PlannerOutput),
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        global_llm_usage = update_llm_usage(state, usage)
        if not llm_success:
            # If LLM failed, return an error response and end the agent workflow
            return {
                "final_response": "Error: Unable to generate plan at this time.",
                "end": True,
                "llm_usage": global_llm_usage,
            }
        return {"plans": response.plans, "llm_usage": global_llm_usage}

    return planner_node
