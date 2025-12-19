from collections.abc import Callable
from typing import Any

from agent.config import AgentState, Call, Email, PlanSeries, ToolCall

# Tool implementations


def filter_by_topics(
    calls: list[Call], emails: list[Email], topics: list[str]
) -> list[Call | Email]:
    """Filter interactions by topics."""
    topics_set = set(topics)
    return [item for item in (calls + emails) if topics_set.intersection(item.topics)]


def filter_by_date(
    calls: list[Call], emails: list[Email], operator: str, date: str
) -> list[Call | Email]:
    """Filter interactions by date with given operator."""
    if operator not in {"=", "<", ">"}:
        raise ValueError(f"Invalid operator: {operator}")

    if operator == "=":
        return [i for i in (calls + emails) if i["date"] == date]

    if operator == "<":
        return [i for i in (calls + emails) if i["date"] < date]

    if operator == ">":
        return [i for i in (calls + emails) if i["date"] > date]
    return calls + emails


def take_last_n(calls: list[Call], emails: list[Email], n: int) -> list[Call | Email]:
    """Take the last n interactions."""
    combined = calls[-n:] + emails[-n:]
    return combined


def take_last_element(calls: list[Call], emails: list[Email]) -> list[Call | Email]:
    """Take the last interaction."""
    result = []
    if calls:
        result.append(calls[-1])
    if emails:
        result.append(emails[-1])
    return result


def compute_len(calls: list[Call], emails: list[Email]) -> int:
    """Compute the length of interactions."""
    return len(calls) + len(emails)


TOOL_REGISTRY: dict[str, Callable[..., list[Call | Email] | int]] = {
    "filter_by_topics": filter_by_topics,
    "filter_by_date": filter_by_date,
    "take_last_n": take_last_n,
    "take_last_element": take_last_element,
    "compute_len": compute_len,
}


def execute_plan_series(
    calls: list[Call], emails: list[Email], plan_series: list[ToolCall]
) -> list[Call | Email] | int:
    """Execute a series of tool calls forming a plan.

    Args:
        calls: List of Call objects
        emails: List of Email objects
        plan_series: List of tool call dictionaries

    Returns:
        The result of executing the plan series, either a list of interactions or an integer.
    """
    current_calls = calls
    current_emails = emails

    for tool_call in plan_series:
        tool_name = tool_call.tool
        params = tool_call.params

        tool_func = TOOL_REGISTRY.get(tool_name)
        if not tool_func:
            raise ValueError(f"Tool {tool_name} not found in registry.")

        if tool_name == "compute_len":
            return tool_func(current_calls, current_emails)
        else:
            result = tool_func(current_calls, current_emails, **params or {})
            # Update current calls and emails based on the result
            current_calls = [item for item in result if item.interaction_type == "call"]  # type: ignore[union-attr]
            current_emails = [
                item
                for item in result  # type: ignore[union-attr]
                if item.interaction_type == "email"
            ]

    return current_calls + current_emails


def build_context(plan: PlanSeries, plan_result: list[Call | Email] | int) -> str:
    """Build context string from interactions."""
    if not isinstance(plan_result, list):
        return f"{plan.title}: \n   {plan_result}"
    context_parts = []
    for item in plan_result:
        context_parts.append(
            f"   {item.interaction_type} Date: {item.date}\nContent: {item.content}\n"
        )
    return f"{plan.title}\n{'\n'.join(context_parts)}"


def create_plan_executer_node() -> Callable[[AgentState], dict[str, Any]]:
    """Execute the plans to construct the final context using multiprocessing."""

    def plan_executer_node(state: AgentState) -> dict[str, Any]:
        calls = state.get("calls", [])
        emails = state.get("emails", [])
        plans = state.get("plans", [])
        all_results = []

        def run_plan(plan: PlanSeries) -> str:
            plan_series = plan.steps
            plan_result = execute_plan_series(calls, emails, plan_series)
            return build_context(plan, plan_result)

        # dont use multiprocessing
        for plan in plans:
            result = run_plan(plan)
            all_results.append(result)

        return {"context": "\n".join(all_results)}

    return plan_executer_node
