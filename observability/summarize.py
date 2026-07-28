from collections.abc import Mapping
from math import isfinite
from typing import Any, Literal

from observability.llm_calls import (
    get_failed_llm_call_event,
)
from observability.models import ExecutionSummary


ISSUE_STATUSES = (
    "resolved",
    "unresolved",
    "regressed",
)    # M7当前只统计这三种历史问题状态


ExecutionStatus = Literal[
    "completed",
    "waiting_for_human",
    "failed",
]    # 执行级Summary允许的状态，由API或调用方根据Graph结果传入


def _as_list(
    value: Any,
) -> list:
    """
    将可能缺失或类型异常的State字段安全转换为list。

    旧State或异常State里字段可能缺失、为None或类型不符合预期。
    汇总逻辑只读取可安全遍历的list，避免观测能力反向影响业务执行。
    """
    return value if isinstance(value, list) else []


def _safe_non_negative_int(
    value: Any,
) -> int:
    """
    安全读取非负整数计数。

    Revision计数来自现有State；缺失或非法时按0处理，兼容旧执行记录。
    """
    if isinstance(value, bool):
        return 0

    if isinstance(value, int) and value >= 0:
        return value

    return 0


def _safe_duration_ms(
    value: Any,
) -> float | None:
    """
    校验并返回可参与汇总的节点耗时。

    只累计真实有效且非负的节点耗时。

    字符串、None、负数、NaN等都视为无效，避免污染执行级active_duration_ms。
    """
    if isinstance(value, bool):
        return None

    if not isinstance(value, int | float):
        return None

    duration = float(value)

    if not isfinite(duration) or duration < 0:
        return None

    return duration


def _count_review_rounds(
    state: Mapping[str, Any],
    history_key: str,
    trace_node_name: str,
    node_execution_counts: dict[str, int],
) -> int:
    """
    计算Story或Scene Review轮次。

    Review轮次优先使用review_history。

    旧State可能还没有history字段，此时退回到Trace中对应review节点的执行次数。
    """
    if history_key in state:
        return len(
            _as_list(
                state.get(history_key)
            )
        )

    return node_execution_counts.get(
        trace_node_name,
        0,
    )


def _count_latest_issue_statuses(
    history: Any,
) -> dict[str, int]:
    """
    聚合Review History中每个issue的最新状态。

    同一issue按strip后的文本识别；如果同一文本出现多次，
    以后出现的historical_issue_checks状态为准。
    """
    latest_status_by_issue: dict[str, str] = {}

    for history_event in _as_list(history):
        if not isinstance(history_event, Mapping):
            continue

        # 旧格式history可能没有historical_issue_checks，直接忽略即可。
        for issue_check in _as_list(
            history_event.get("historical_issue_checks")
        ):
            if not isinstance(issue_check, Mapping):
                continue

            issue = str(
                issue_check.get("issue") or ""
            ).strip()
            status = str(
                issue_check.get("status") or ""
            ).strip()

            if not issue or status not in ISSUE_STATUSES:
                continue

            latest_status_by_issue[issue] = status

    status_counts = {
        "resolved": 0,
        "unresolved": 0,
        "regressed": 0,
    }

    for status in latest_status_by_issue.values():
        status_counts[status] += 1

    return status_counts


def _summarize_error(
    error: BaseException | None,
) -> tuple[str | None, str | None]:
    """
    提取异常类型和简短消息。

    Summary只保存可读摘要，不保存traceback、Prompt、API Key或其他长上下文。
    """
    if error is None:
        return None, None

    error_message = str(error).strip()

    return (
        type(error).__name__,
        error_message[:200] if error_message else None,
    )


def _collect_llm_call_events(
    state: Mapping[str, Any],
    error: BaseException | None,
) -> list[Mapping[str, Any]]:
    """
    读取State中的LLM调用事件，并补入尚未提交State的失败调用。

    正常节点返回后，事件由FilmState reducer写入checkpoint；
    如果LLM调用直接抛错，节点没有机会提交局部State，此时从原异常附带的
    安全事件中补入Summary。若State已经包含同一事件则不重复统计。
    """
    events = [
        event
        for event in _as_list(
            state.get("llm_call_trace")
        )
        if isinstance(event, Mapping)
    ]
    failed_event = get_failed_llm_call_event(
        error
    )

    if (
        failed_event is not None
        and failed_event not in events
    ):
        events.append(
            failed_event
        )

    return events


def _summarize_llm_calls(
    llm_call_events: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    聚合LLM调用次数、耗时以及Prompt/Profile/Model使用情况。

    这里只读取最小元数据，不把完整llm_call_trace复制进ExecutionSummary。
    """
    successful_count = 0
    failed_count = 0
    active_duration_ms = 0.0
    prompt_versions: dict[str, list[str]] = {}
    profile_usage: dict[str, int] = {}
    model_usage: dict[str, int] = {}

    for event in llm_call_events:
        status = event.get("status")
        if status == "success":
            successful_count += 1
        elif status == "failed":
            failed_count += 1

        duration_ms = _safe_duration_ms(
            event.get("duration_ms")
        )
        if duration_ms is not None:
            active_duration_ms += duration_ms

        prompt_name = event.get(
            "prompt_name"
        )
        prompt_version = event.get(
            "prompt_version"
        )
        if (
            isinstance(prompt_name, str)
            and prompt_name
            and isinstance(prompt_version, str)
            and prompt_version
        ):
            versions = prompt_versions.setdefault(
                prompt_name,
                [],
            )
            if prompt_version not in versions:
                versions.append(
                    prompt_version
                )

        profile_name = event.get(
            "llm_profile"
        )
        if isinstance(profile_name, str) and profile_name:
            profile_usage[profile_name] = (
                profile_usage.get(
                    profile_name,
                    0,
                )
                + 1
            )

        model_name = event.get(
            "model_name"
        )
        if isinstance(model_name, str) and model_name:
            model_usage[model_name] = (
                model_usage.get(
                    model_name,
                    0,
                )
                + 1
            )

    return {
        "llm_call_count": len(
            llm_call_events
        ),
        "successful_llm_call_count": successful_count,
        "failed_llm_call_count": failed_count,
        "llm_active_duration_ms": round(
            active_duration_ms,
            2,
        ),
        "prompt_versions": prompt_versions,
        "profile_usage": profile_usage,
        "model_usage": model_usage,
    }


def build_execution_summary(
    state: Mapping[str, Any],
    status: ExecutionStatus,
    error: BaseException | None = None,
) -> ExecutionSummary:
    """
    从FilmState构建执行级Summary。

    这是纯汇总函数：只读取传入state，不修改State、不触发Graph、不访问外部存储。

    返回值表示一次Graph执行的轻量观测快照，可用于API返回、日志记录或后续持久化。
    """
    # 读取Trace列表：Trace是节点级观测数据，所有节点耗时和执行次数都从这里计算。
    execution_trace = _as_list(
        state.get("execution_trace")
    )
    active_duration_ms = 0.0    # 累计有效duration_ms，忽略缺失、非法和负数
    successful_node_count = 0    # 统计节点Trace中status为success的事件
    node_execution_counts: dict[str, int] = {}    # 按node名称累计执行次数
    failed_node = None    # 如果Trace里出现failed状态，记录第一个明确失败节点

    # 遍历TraceEvent，计算节点耗时、成功节点数和节点执行频次。
    for trace_event in execution_trace:
        if not isinstance(trace_event, Mapping):
            continue

        duration_ms = _safe_duration_ms(
            trace_event.get("duration_ms")
        )
        if duration_ms is not None:
            active_duration_ms += duration_ms

        # 当前graph.trace_node真实写入的节点状态是success。
        if trace_event.get("status") == "success":
            successful_node_count += 1

        if trace_event.get("status") == "failed" and failed_node is None:
            trace_failed_node = trace_event.get("node")
            if isinstance(trace_failed_node, str) and trace_failed_node:
                failed_node = trace_failed_node

        node_name = trace_event.get("node")
        if isinstance(node_name, str) and node_name:
            node_execution_counts[node_name] = (
                node_execution_counts.get(node_name, 0) + 1
            )

    # 人工反馈历史来自HITL节点；approve和revise都算反馈，只有revise算人工修订。
    human_feedback_history = _as_list(
        state.get("human_feedback_history")
    )
    human_revision_count = sum(
        1
        for feedback_event in human_feedback_history
        if (
            isinstance(feedback_event, Mapping)
            and feedback_event.get("decision") == "revise"
        )
    )

    error_type, error_message = _summarize_error(
        error
    )
    llm_call_events = _collect_llm_call_events(
        state,
        error,
    )
    llm_summary = _summarize_llm_calls(
        llm_call_events
    )

    # 节点Trace没有失败事件时，LLM失败元数据仍可确定实际失败节点。
    if failed_node is None:
        for llm_call_event in llm_call_events:
            if llm_call_event.get("status") != "failed":
                continue

            llm_failed_node = llm_call_event.get(
                "node"
            )
            if isinstance(llm_failed_node, str) and llm_failed_node:
                failed_node = llm_failed_node
                break

    # 组装最终ExecutionSummary：
    # - Review轮次优先读取history，兼容旧State时回退到Trace节点次数；
    # - Revision次数直接读取State中已有计数；
    # - memory_update_status保留原始状态，便于区分saved/skipped/failed。
    # - issue状态只聚合Review History中结构化的historical_issue_checks。
    return ExecutionSummary(
        execution_id=str(
            state.get("execution_id") or ""
        ),    # 缺失时返回空字符串，避免旧State报错
        status=status,
        current_stage=str(
            state.get("current_stage") or "unknown"
        ),    # 缺失时标记unknown
        active_duration_ms=round(
            active_duration_ms,
            2,
        ),    # 保留两位小数，和trace_node中的毫秒粒度保持接近
        trace_event_count=len(execution_trace),    # Trace列表长度，包含非success事件或旧格式事件
        successful_node_count=successful_node_count,    # 仅统计明确success的节点事件
        node_execution_counts=node_execution_counts,    # 每个节点执行次数，可反映Review-Revision循环
        story_review_rounds=_count_review_rounds(
            state,
            "story_review_history",
            "review_story",
            node_execution_counts,
        ),    # Story Review轮次
        story_revision_count=_safe_non_negative_int(
            state.get("story_revision_count")
        ),    # Story Revision次数
        scene_review_rounds=_count_review_rounds(
            state,
            "scene_review_history",
            "review_scene",
            node_execution_counts,
        ),    # Scene Review轮次
        scene_revision_count=_safe_non_negative_int(
            state.get("scene_revision_count")
        ),    # Scene Revision次数
        human_feedback_count=len(
            human_feedback_history
        ),    # 人工审核反馈事件总数
        human_revision_count=human_revision_count,    # 人工选择revise的次数
        memory_update_status=state.get(
            "memory_update_status"
        ),    # Memory更新状态原样透出
        failed_node=failed_node,    # 失败节点优先从Trace中读取，无法确定时为None
        error_type=error_type,    # 异常类型摘要
        error_message=error_message,    # 简短异常消息，不包含traceback
        story_issue_status_counts=_count_latest_issue_statuses(
            state.get("story_review_history")
        ),    # Story历史issue最新状态计数
        scene_issue_status_counts=_count_latest_issue_statuses(
            state.get("scene_review_history")
        ),    # Scene历史issue最新状态计数
        **llm_summary,    # LLM调用轻量聚合，不重复返回详细llm_call_trace
    )
