from typing import Any

from langgraph.types import interrupt

from state import FilmState


def _dump_model(
    value: Any,
) -> Any:
    """
    将Pydantic对象转换为普通JSON数据；普通值原样返回。
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json"
        )

    return value


def human_review_story(
    state: FilmState,
) -> dict[str, Any]:
    """
    在故事大纲完成后暂停Graph，等待人工确认是否继续生成分场。
    """
    payload = {
        "type": "story_review_required",
        "execution_id": state["execution_id"],
        "message": "故事大纲已完成，请确认是否继续生成分场。",
        "film_brief": _dump_model(
            state["film_brief"]
        ),
        "characters": [
            _dump_model(character)
            for character in state["characters"]
        ],
        "story_outline": _dump_model(
            state["story_outline"]
        ),
        "story_review": _dump_model(
            state["story_review_result"]
        ),
    }

    # interrupt会把JSON可序列化payload交给外部调用方；
    # 恢复前不要执行写文件、更新Memory等副作用。
    resume_value = interrupt(payload)

    if not isinstance(resume_value, dict):
        raise ValueError(
            "human_review_story恢复输入必须是dict。"
        )

    decision = resume_value.get("decision")
    feedback = resume_value.get("feedback")

    if decision not in {"approve", "revise"}:
        raise ValueError(
            "human_review_story decision只能是approve或revise。"
        )

    if feedback is not None:
        feedback = str(feedback).strip()

    if decision == "revise" and not feedback:
        raise ValueError(
            "human_review_story选择revise时feedback不能为空。"
        )

    if decision == "approve":
        feedback = feedback or None

    return {
        "human_review_decision": decision,
        "human_feedback": feedback,
        "human_feedback_history": [
            {
                "scope": "story",
                "decision": decision,
                "feedback": feedback,
            }
        ],
        "current_stage": "human_story_review_completed",
    }
