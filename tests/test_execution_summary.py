import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observability.models import ExecutionSummary
from observability.summarize import build_execution_summary


def _trace_event(
    node: str,
    duration_ms,
    status: str = "success",
) -> dict:
    """
    构造测试用TraceEvent字典，保持与graph.trace_node真实字段一致。
    """
    return {
        "execution_id": "exec_demo",
        "node": node,
        "status": status,
        "stage": f"{node}_completed",
        "duration_ms": duration_ms,
    }


def test_build_summary_for_completed_execution():
    """
    正常完成时，Summary应聚合Trace、Review、Revision、人工反馈和Memory状态。
    """
    state = {
        "execution_id": "exec_demo",
        "current_stage": "memory_updated",
        "execution_trace": [
            _trace_event("retrieve_memory", 1.2),
            _trace_event("review_story", 2.3),
            _trace_event("review_scene", 3.4),
        ],
        "story_review_history": [
            {"passed": False},
            {"passed": True},
        ],
        "scene_review_history": [
            {"passed": True},
        ],
        "story_revision_count": 1,
        "scene_revision_count": 0,
        "human_feedback_history": [
            {
                "scope": "story",
                "decision": "approve",
                "feedback": None,
            },
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "结尾更克制",
            },
        ],
        "memory_update_status": "saved",
    }

    summary = build_execution_summary(
        state,
        "completed",
    )

    assert isinstance(summary, ExecutionSummary)
    assert summary.execution_id == "exec_demo"
    assert summary.status == "completed"
    assert summary.current_stage == "memory_updated"
    assert summary.active_duration_ms == 6.9
    assert summary.trace_event_count == 3
    assert summary.successful_node_count == 3
    assert summary.node_execution_counts == {
        "retrieve_memory": 1,
        "review_story": 1,
        "review_scene": 1,
    }
    assert summary.story_review_rounds == 2
    assert summary.story_revision_count == 1
    assert summary.scene_review_rounds == 1
    assert summary.scene_revision_count == 0
    assert summary.human_feedback_count == 2
    assert summary.human_revision_count == 1
    assert summary.memory_update_status == "saved"
    assert summary.failed_node is None
    assert summary.error_type is None
    assert summary.error_message is None
    assert summary.story_issue_status_counts == {
        "resolved": 0,
        "unresolved": 0,
        "regressed": 0,
    }
    assert summary.scene_issue_status_counts == {
        "resolved": 0,
        "unresolved": 0,
        "regressed": 0,
    }


def test_empty_trace_is_supported():
    """
    空Trace或缺少节点执行记录时，Summary仍应返回0值指标。
    """
    summary = build_execution_summary(
        {
            "execution_id": "exec_empty",
            "current_stage": "initialized",
            "execution_trace": [],
        },
        "waiting_for_human",
    )

    assert summary.execution_id == "exec_empty"
    assert summary.status == "waiting_for_human"
    assert summary.active_duration_ms == 0
    assert summary.trace_event_count == 0
    assert summary.successful_node_count == 0
    assert summary.node_execution_counts == {}


def test_same_node_execution_is_counted_multiple_times():
    """
    同一节点可能因Review-Revision循环执行多次，需要按节点名累计次数。
    """
    summary = build_execution_summary(
        {
            "execution_trace": [
                _trace_event("review_story", 1.0),
                _trace_event("revise_story", 2.0),
                _trace_event("review_story", 3.0),
            ],
        },
        "completed",
    )

    assert summary.node_execution_counts["review_story"] == 2
    assert summary.node_execution_counts["revise_story"] == 1
    assert summary.successful_node_count == 3


def test_invalid_missing_or_negative_duration_is_ignored():
    """
    duration_ms可能来自旧Trace或异常数据；无效值不参与active_duration_ms。
    """
    summary = build_execution_summary(
        {
            "execution_trace": [
                _trace_event("a", 10.0),
                _trace_event("b", -2.0),
                _trace_event("c", "3.0"),
                {"node": "d", "status": "success"},
                _trace_event("e", float("nan")),
            ],
        },
        "failed",
    )

    assert summary.status == "failed"
    assert summary.active_duration_ms == 10.0
    assert summary.trace_event_count == 5
    assert summary.successful_node_count == 5


def test_review_rounds_fallback_to_trace_when_history_missing():
    """
    新State优先用review_history；旧State缺少history时退回Trace节点次数。
    """
    summary = build_execution_summary(
        {
            "execution_trace": [
                _trace_event("review_story", 1.0),
                _trace_event("review_story", 1.0),
                _trace_event("review_scene", 1.0),
            ],
            "story_revision_count": 2,
            "scene_revision_count": 1,
        },
        "completed",
    )

    assert summary.story_review_rounds == 2
    assert summary.story_revision_count == 2
    assert summary.scene_review_rounds == 1
    assert summary.scene_revision_count == 1


def test_review_history_takes_precedence_over_trace_counts():
    """
    如果State已有Review History，应以历史记录为准，而不是Trace里的节点次数。
    """
    summary = build_execution_summary(
        {
            "execution_trace": [
                _trace_event("review_story", 1.0),
                _trace_event("review_story", 1.0),
            ],
            "story_review_history": [
                {"passed": True},
            ],
            "scene_review_history": [],
        },
        "completed",
    )

    assert summary.story_review_rounds == 1
    assert summary.scene_review_rounds == 0


def test_human_approve_and_revise_counts():
    """
    human_feedback_count统计全部人工事件，human_revision_count只统计revise。
    """
    summary = build_execution_summary(
        {
            "human_feedback_history": [
                {"decision": "approve", "feedback": None},
                {"decision": "revise", "feedback": "改结尾"},
                {"decision": "revise", "feedback": "再弱化旁白"},
            ],
        },
        "waiting_for_human",
    )

    assert summary.human_feedback_count == 3
    assert summary.human_revision_count == 2


def test_old_state_missing_optional_fields_does_not_fail():
    """
    缺少可选字段的旧State仍可汇总，缺省值保持保守。
    """
    summary = build_execution_summary(
        {},
        "failed",
    )

    assert summary.execution_id == ""
    assert summary.current_stage == "unknown"
    assert summary.active_duration_ms == 0
    assert summary.trace_event_count == 0
    assert summary.story_review_rounds == 0
    assert summary.scene_review_rounds == 0
    assert summary.memory_update_status is None
    assert summary.story_issue_status_counts == {
        "resolved": 0,
        "unresolved": 0,
        "regressed": 0,
    }


def test_summary_does_not_modify_input_state():
    """
    汇总函数是纯读取函数，不应改动传入State。
    """
    state = {
        "execution_id": "exec_immutable",
        "execution_trace": [
            _trace_event("review_story", 1.0),
        ],
        "human_feedback_history": [
            {"decision": "revise", "feedback": "改结尾"},
        ],
    }
    original_state = copy.deepcopy(state)

    build_execution_summary(
        state,
        "completed",
    )

    assert state == original_state


def test_issue_status_counts_use_latest_status_by_issue_text():
    """
    同一issue多次出现时，以最后一次historical_issue_checks状态计数。
    """
    state = {
        "story_review_history": [
            {
                "historical_issue_checks": [
                    {
                        "issue": " 结尾没有回应冲突 ",
                        "status": "unresolved",
                        "evidence": "第一轮仍存在",
                    },
                    {
                        "issue": "角色动机不清",
                        "status": "resolved",
                        "evidence": "已补齐",
                    },
                ],
            },
            {
                "historical_issue_checks": [
                    {
                        "issue": "结尾没有回应冲突",
                        "status": "resolved",
                        "evidence": "第二轮已解决",
                    },
                    {
                        "issue": "角色动机不清",
                        "status": "regressed",
                        "evidence": "又变弱",
                    },
                ],
            },
        ],
        "scene_review_history": [
            {
                "historical_issue_checks": [
                    {
                        "issue": "总时长不一致",
                        "status": "unresolved",
                        "evidence": "仍不一致",
                    },
                ],
            },
        ],
    }

    summary = build_execution_summary(
        state,
        "completed",
    )

    assert summary.story_issue_status_counts == {
        "resolved": 1,
        "unresolved": 0,
        "regressed": 1,
    }
    assert summary.scene_issue_status_counts == {
        "resolved": 0,
        "unresolved": 1,
        "regressed": 0,
    }


def test_old_review_history_without_issue_checks_is_ignored():
    """
    旧Review History只有issues/suggestions时，不参与issue状态计数。
    """
    summary = build_execution_summary(
        {
            "story_review_history": [
                {
                    "issues": [
                        "旧问题",
                    ],
                    "suggestions": [],
                }
            ],
            "scene_review_history": [
                {
                    "historical_issue_checks": [
                        {
                            "issue": " ",
                            "status": "resolved",
                            "evidence": "空问题应忽略",
                        },
                        {
                            "issue": "状态异常",
                            "status": "unknown",
                            "evidence": "非M7状态应忽略",
                        },
                    ],
                }
            ],
        },
        "completed",
    )

    assert summary.story_issue_status_counts == {
        "resolved": 0,
        "unresolved": 0,
        "regressed": 0,
    }
    assert summary.scene_issue_status_counts == {
        "resolved": 0,
        "unresolved": 0,
        "regressed": 0,
    }


def test_failed_node_and_error_summary():
    """
    failed_node来自失败Trace，error摘要只保留类型和短消息。
    """
    summary = build_execution_summary(
        {
            "execution_trace": [
                _trace_event("retrieve_memory", 1.0),
                _trace_event("review_story", 2.0, status="failed"),
            ],
            "current_stage": "story_review_failed",
        },
        "failed",
        error=RuntimeError("review failed because model timeout"),
    )

    assert summary.status == "failed"
    assert summary.failed_node == "review_story"
    assert summary.error_type == "RuntimeError"
    assert summary.error_message == "review failed because model timeout"
    assert "Traceback" not in summary.error_message
