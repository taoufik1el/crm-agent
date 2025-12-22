from typing import Any, Literal

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agent.config import AgentState


def get_llm(
    llm_provider: Literal["google", "openai"],
    model_name: str,
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"],
) -> BaseChatModel:
    """Get the LLM based on the configured provider."""
    if llm_provider == "google":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    else:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            reasoning={"effort": reasoning_effort}
            if reasoning_effort != "none"
            else {},
        )


def safe_run_llm(
    llm: BaseChatModel, llm_inputs: Any
) -> tuple[Any, bool, dict[str, Any]]:
    """Invoke LLM with usage tracking and error handling."""
    llm_success = False
    with get_usage_metadata_callback() as usage_cb:
        try:
            response = llm.invoke(llm_inputs)
            # access token usage
            llm_success = True
        except Exception:
            response = "Error during LLM invocation"
        finally:
            usage = usage_cb.usage_metadata
    return response, llm_success, usage


def update_llm_usage(state: AgentState, usage: dict[str, Any]) -> dict[str, Any]:
    """Update the state with LLM usage information."""
    for model_name, model_usage in usage.items():
        if model_name in state.get("llm_usage", {}):
            for key, value in model_usage.items():
                state["llm_usage"][model_name][key] += value
        else:
            state["llm_usage"][model_name] = model_usage
    return state["llm_usage"]  # type: ignore[no-any-return]
