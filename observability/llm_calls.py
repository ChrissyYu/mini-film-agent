from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterator

from llm_profiles.bindings import get_prompt_profile_name
from llm_profiles.registry import get_llm_profile
from prompts.models import RenderedPrompt
from state import LLMCallTraceEvent


_ACTIVE_LLM_CALL_EVENTS: ContextVar[
    list[LLMCallTraceEvent] | None
] = ContextVar(
    "active_llm_call_events",
    default=None,
)
_ERROR_TRACE_ATTRIBUTE = "_llm_call_trace_event"


@contextmanager
def collect_llm_call_trace() -> Iterator[
    list[LLMCallTraceEvent]
]:
    """
    收集当前节点内实际发生的LLM调用事件。

    collector只在一次节点调用期间生效；节点完成后把列表作为局部State更新返回，
    再由FilmState的operator.add reducer累计，避免使用跨请求的全局可变列表。
    """
    events: list[LLMCallTraceEvent] = []
    token = _ACTIVE_LLM_CALL_EVENTS.set(
        events
    )

    try:
        yield events
    finally:
        _ACTIVE_LLM_CALL_EVENTS.reset(
            token
        )


def _build_llm_call_event(
    node: str,
    rendered_prompt: RenderedPrompt,
    status: str,
    duration_ms: float,
    error_type: str | None,
) -> LLMCallTraceEvent:
    """
    从Prompt元数据、Binding和Profile构造JSON-safe调用事件。
    """
    profile_name = get_prompt_profile_name(
        rendered_prompt.name
    )
    profile = get_llm_profile(
        profile_name
    )

    return {
        "node": node,
        "prompt_name": rendered_prompt.name,
        "prompt_version": rendered_prompt.version,
        "prompt_chars": rendered_prompt.chars,
        "llm_profile": profile.name,
        "model_name": profile.model,
        "temperature": profile.temperature,
        "status": status,
        "duration_ms": duration_ms,
        "error_type": error_type,
    }


def _record_llm_call_event(
    event: LLMCallTraceEvent,
) -> None:
    """
    将事件追加到当前节点collector；没有collector时保持调用函数可独立使用。
    """
    active_events = (
        _ACTIVE_LLM_CALL_EVENTS.get()
    )

    if active_events is not None:
        active_events.append(
            event
        )


def invoke_structured_llm(
    structured_llm: Any,
    rendered_prompt: RenderedPrompt,
    *,
    node: str,
) -> Any:
    """
    调用现有structured LLM，并记录最小可观测元数据。

    事件不包含Prompt正文、模型响应、用户输入、Memory、API Key或traceback。
    失败时保留原异常类型和抛出行为，只在异常对象上附加安全事件，
    供执行失败后的Summary尽可能统计这次真实调用。
    """
    start_time = perf_counter()

    try:
        result = structured_llm.invoke(
            rendered_prompt.text
        )
    except Exception as exc:
        duration_ms = round(
            (perf_counter() - start_time) * 1000,
            2,
        )
        event = _build_llm_call_event(
            node=node,
            rendered_prompt=rendered_prompt,
            status="failed",
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
        )
        _record_llm_call_event(
            event
        )

        # 部分第三方异常可能禁止动态属性；附加失败事件不能覆盖原始异常。
        try:
            setattr(
                exc,
                _ERROR_TRACE_ATTRIBUTE,
                event,
            )
        except Exception:
            pass

        raise

    duration_ms = round(
        (perf_counter() - start_time) * 1000,
        2,
    )
    event = _build_llm_call_event(
        node=node,
        rendered_prompt=rendered_prompt,
        status="success",
        duration_ms=duration_ms,
        error_type=None,
    )
    _record_llm_call_event(
        event
    )

    return result


def get_failed_llm_call_event(
    error: BaseException | None,
) -> LLMCallTraceEvent | None:
    """
    从失败异常读取调用事件；旧异常或非LLM异常返回None。
    """
    if error is None:
        return None

    event = getattr(
        error,
        _ERROR_TRACE_ATTRIBUTE,
        None,
    )

    return event if isinstance(event, dict) else None
