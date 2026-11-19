"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand", "overloaded")
_MAX_ATTEMPTS = 6


def _is_transient(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(m in str(exc) for m in _TRANSIENT_MARKERS):
            return True
        exc = getattr(exc, "error", None) or exc.__cause__ or exc.__context__
    return False


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    Retries with exponential backoff on transient provider errors (503 / 429).
    """
    import asyncio

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await _chat_once(agent, runner, user_message, session_id)
        except Exception as exc:
            if attempt == _MAX_ATTEMPTS or not _is_transient(exc):
                raise
            wait = min(10 * 2 ** (attempt - 1), 90)
            print(f"[retry] transient provider error (attempt {attempt}/{_MAX_ATTEMPTS}); waiting {wait}s")
            await asyncio.sleep(wait)


async def _chat_once(agent, runner, user_message: str, session_id=None):
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    return final_response, session
