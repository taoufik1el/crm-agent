import logging
from collections.abc import Generator
from typing import Any, Literal

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_llm(
    llm_provider: Literal["google", "openai"],
    model_name: str,
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"],
    streaming: bool,
) -> BaseChatModel:
    """Get the LLM based on the configured provider."""
    if llm_provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=0, streaming=streaming
        )
    else:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            streaming=streaming,
            reasoning={"effort": reasoning_effort}
            if reasoning_effort != "none"
            else {},
        )


def safe_run_llm(
    llm: BaseChatModel | RunnableSequence, llm_inputs: Any
) -> tuple[Any, bool]:
    """Invoke LLM with usage tracking and error handling."""
    llm_success = False
    try:
        with get_usage_metadata_callback() as usage_cb:
            response = llm.invoke(llm_inputs)
            logging.info(f"LLM usage: {usage_cb.usage_metadata}")
        llm_success = True
    except Exception:
        response = "Error during LLM invocation"
    return response, llm_success


def safe_stream_llm(  # type: ignore[return]
    llm: BaseChatModel | RunnableSequence, llm_inputs: Any
) -> Generator[Any | None, None, dict[str, Any]]:
    """Stream LLM output token by token with usage tracking.

    Yields:
        Tuple[token, llm_success, usage]
        - token: each chunk of text from the LLM
        - llm_success: True/False (only True after first token if no error yet)
        - usage: dict of LLM usage metadata (final after streaming ends)
    """
    try:
        with get_usage_metadata_callback() as usage_cb:
            for chunk in llm.stream(llm_inputs):
                yield from chunk
            logging.info(f"LLM usage: {usage_cb.usage_metadata}")
    except (RuntimeError, ValueError):
        # Yield error message once
        yield None
