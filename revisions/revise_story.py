import json

from llm_profiles.factory import create_structured_llm
from memory.context import format_story_memory_context
from observability.llm_calls import (
    collect_llm_call_trace,
    invoke_structured_llm,
)
from prompts.renderer import render_prompt
from state import FilmState
from schemas import StoryOutline


# ================= LLM配置 =================

story_revise_llm = create_structured_llm(
    "revision.story",
    StoryOutline,
)


def _collect_history_issue_constraints(
    history: list[dict],
    active_limit: int = 8,
    resolved_limit: int = 2,
) -> tuple[list[str], list[str]]:
    """
    按历史问题最新状态拆分回归约束，旧格式问题按unknown处理。
    """
    issue_status_by_text = {}
    ordered_issues = []

    def remember_issue(
        issue: str,
        status: str,
    ) -> None:
        cleaned_issue = str(issue).strip()

        if not cleaned_issue:
            return

        normalized_status = (
            status
            if status in {
                "resolved",
                "unresolved",
                "regressed",
                "unknown",
            }
            else "unknown"
        )

        if cleaned_issue in ordered_issues:
            ordered_issues.remove(cleaned_issue)

        ordered_issues.append(cleaned_issue)
        issue_status_by_text[cleaned_issue] = normalized_status

    for history_event in history:
        # 旧格式history只有issues，没有状态；默认按unknown继续防回归。
        for issue in history_event.get("issues", []):
            remember_issue(
                issue,
                "unknown",
            )

        # 新格式以historical_issue_checks中的最近状态为准。
        for check in history_event.get("historical_issue_checks", []):
            remember_issue(
                check.get("issue", ""),
                check.get("status", "unknown"),
            )

    active_issues = [
        issue
        for issue in ordered_issues
        if issue_status_by_text.get(issue) in {
            "unresolved",
            "regressed",
            "unknown",
        }
    ][-active_limit:]

    resolved_reminders = [
        issue
        for issue in ordered_issues
        if issue_status_by_text.get(issue) == "resolved"
    ][-resolved_limit:]

    return active_issues, resolved_reminders


# ================= revise node =================

def revise_story(state:FilmState) -> dict:
    """
    根据审核结果修改故事大纲
    """

    required_keys=[
        "film_brief",
        "characters",
        "story_outline",
        "story_review_result"
    ]


    for key in required_keys:
        if key not in state:
            raise ValueError(
                f"revise_story缺少字段:{key}"
            )

    film_brief = (
        state["film_brief"]
        .model_dump()
    )

    characters = [
        c.model_dump()
        for c in state["characters"]
    ]

    current_story_outline = (
        state["story_outline"]
        .model_dump()
    )
    story_memory_text = format_story_memory_context(
        state.get("user_memory")
    )    # Story修订只参考故事相关长期Memory，不读取scene_preferences

    story_review_result = (
        state["story_review_result"]
        .model_dump()
    )
    story_review_issues = story_review_result.get(
        "issues",
        [],
    )
    story_review_suggestions = story_review_result.get(
        "suggestions",
        [],
    )

    human_feedback = state.get(
        "human_feedback"
    )
    if human_feedback:
        human_feedback = human_feedback.strip()

    human_feedback_text = (
        human_feedback
        if human_feedback
        else "无"
    )

    active_history_issues, resolved_history_reminders = (
        _collect_history_issue_constraints(
            state.get("story_review_history", [])
        )
    )
    active_history_issues_text = (
        json.dumps(
            active_history_issues,
            ensure_ascii=False,
            indent=2,
        )
        if active_history_issues
        else "无历史问题"
    )
    resolved_history_reminders_text = (
        json.dumps(
            resolved_history_reminders,
            ensure_ascii=False,
            indent=2,
        )
        if resolved_history_reminders
        else "无已解决问题提醒"
    )

    rendered = render_prompt(
        "revision.story",
        version="v1",
        film_brief=film_brief,
        characters=characters,
        current_story_outline=current_story_outline,
        story_memory_text=story_memory_text,
        human_feedback_text=human_feedback_text,
        story_review_issues=story_review_issues,
        story_review_suggestions=story_review_suggestions,
        active_history_issues_text=(
            active_history_issues_text
        ),
        resolved_history_reminders_text=(
            resolved_history_reminders_text
        ),
    )

    with collect_llm_call_trace() as llm_call_events:
        new_story_outline: StoryOutline = (
            invoke_structured_llm(
                story_revise_llm,
                rendered,
                node="revise_story",
            )
        )

    return {
        "story_outline": new_story_outline,     # 修改后的故事大纲
        "story_revision_count":
            state.get(
                "story_revision_count",
                0
            ) + 1,      # 故事大纲修改次数 +1
        "current_stage":"story_revised_completed",  # 故事大纲修改完成
        "llm_call_trace": llm_call_events,

    }


# ================= Test =================

if __name__=="__main__":

    print(
        "请使用真实pipeline测试"
    )
