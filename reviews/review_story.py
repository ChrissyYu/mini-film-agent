import json

from llm_profiles.factory import create_structured_llm
from memory.context import format_story_memory_context
from observability.llm_calls import (
    collect_llm_call_trace,
    invoke_structured_llm,
)
from prompts.renderer import render_prompt
from state import FilmState
from schemas import StoryReviewResult, StoryCriticResult


# ================= LLM配置 =================

story_critic_llm = create_structured_llm(
    "review.story",
    StoryCriticResult,
)


# ================= 规则审核 =================


def check_story_fields(state:FilmState):
    """
    检查故事大纲字段是否为空
    """
    issues = []
    
    if "story_outline" not in state:
        issues.append(
            "缺少story_outline，无法审核"
        )
        return issues
    
    story_outline = state["story_outline"]

    required_fields=[
        "setup",
        "conflict",
        "turning_point",
        "ending",
        "theme"
    ]

    for field in required_fields:
        value = getattr(
            story_outline,
            field
        )
        if not value:
            issues.append(
                f"故事大纲字段{field}为空"
            )

    return issues


# ================= LLM审核 =================

def _collect_history_issue_context(
    history: list[dict],
) -> list[dict]:
    """
    从审核历史中提取去空、去重后的旧问题，并附带最近一次状态。
    """
    issues = []
    seen = set()
    latest_status_by_issue = {}

    for history_event in history:
        for issue in history_event.get("issues", []):
            cleaned_issue = str(issue).strip()

            if not cleaned_issue:
                continue

            if cleaned_issue in seen:
                continue

            issues.append(cleaned_issue)
            seen.add(cleaned_issue)

        # 旧格式history没有historical_issue_checks，使用get保证兼容。
        for check in history_event.get("historical_issue_checks", []):
            issue = str(check.get("issue", "")).strip()
            status = str(check.get("status", "")).strip()

            if not issue:
                continue

            latest_status_by_issue[issue] = (
                status
                if status
                else "unknown"
            )

            if issue in seen:
                continue

            issues.append(issue)
            seen.add(issue)

    return [
        {
            "issue": issue,
            "latest_status": latest_status_by_issue.get(
                issue,
                "unknown",
            ),
        }
        for issue in issues
    ]


def llm_review_story(state: FilmState) -> StoryCriticResult:
    """
    使用LLM进行故事大纲语义一致性审核
    """

    required_keys=[
        "film_brief",
        "characters",
        "story_outline"
    ]

    for key in required_keys:
        if key not in state:
            raise ValueError(
                f"review_story缺少字段:{key}"
            )

    film_brief = (
        state["film_brief"]
        .model_dump()
    )

    characters=[
        c.model_dump()
        for c in state["characters"]
    ]

    story_outline=(
        state["story_outline"]
        .model_dump()
    )
    user_idea = state.get(
        "user_idea",
        "未提供",
    )
    story_memory_text = format_story_memory_context(
        state.get("user_memory")
    )    # Story审核只参考Story相关Memory，不读取scene_preferences

    history_issue_context = _collect_history_issue_context(
        state.get("story_review_history", [])
    )
    history_issues_text = (
        json.dumps(
            history_issue_context,
            ensure_ascii=False,
            indent=2,
        )
        if history_issue_context
        else "无历史问题"
    )

    rendered = render_prompt(
        "review.story",
        version="v1",
        film_brief=film_brief,
        user_idea=user_idea,
        characters=characters,
        story_outline=story_outline,
        story_memory_text=story_memory_text,
        history_issues_text=history_issues_text,
    )

    result = invoke_structured_llm(
        story_critic_llm,
        rendered,
        node="review_story",
    )

    # 兼容部分模型返回dict
    if isinstance(result, dict):
        result = StoryCriticResult(**result)

    return result


# ================= 主审核节点 =================

def review_story(state:FilmState):

    issues=[]
    suggestions=[]

    # 规则检查
    issues.extend(
        check_story_fields(state)
    )

    # LLM检查
    with collect_llm_call_trace() as llm_call_events:
        story_critic_result = llm_review_story(state)
    
    issues.extend(
        story_critic_result.issues
    )
    suggestions.extend(
        story_critic_result.suggestions
    )

    historical_issue_checks = (
        story_critic_result.historical_issue_checks
    )

    resolved_history_issues = {
        check.issue.strip()
        for check in historical_issue_checks
        if check.status == "resolved"
        and check.issue.strip()
    }
    issues = [
        issue
        for issue in issues
        if str(issue).strip() not in resolved_history_issues
    ]

    # 历史问题若仍未解决或发生回归，本轮必须按阻断问题处理。
    for check in historical_issue_checks:
        if check.status not in {
            "unresolved",
            "regressed",
        }:
            continue

        issue = check.issue.strip()

        if not issue:
            continue

        if issue in issues:
            continue

        issues.append(issue)

    passed = (
        len(issues)==0
        and story_critic_result.passed
    )

    story_review_result = StoryReviewResult(
        passed = passed,
        issues = issues,
        suggestions = suggestions,
        historical_issue_checks=historical_issue_checks,
    )

    return {
        "story_review_result": story_review_result,
        "story_review_history": [
            {
                "revision_round": state.get(
                    "story_revision_count",
                    0,
                ),
                "passed": story_review_result.passed,
                "issues": story_review_result.issues,
                "suggestions": story_review_result.suggestions,
                "historical_issue_checks": [
                    check.model_dump()
                    for check
                    in story_review_result.historical_issue_checks
                ],
            }
        ],
        "current_stage": "story_review_completed",
        "llm_call_trace": llm_call_events,
    }
