import json

from memory.context import format_story_memory_context
from nodes import llm
from state import FilmState
from schemas import StoryOutline


# ================= LLM配置 =================

story_revise_llm = llm.with_structured_output(StoryOutline)


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

    prompt = f"""
你是一名专业影视编剧。

你的任务是根据审核反馈，对已有故事大纲进行最小必要修改，
使其达到可以进入分场规划阶段的标准。

不要重新创作一个完全不同的故事。
应保留原始创意、主题、主要角色和已有的合理内容。

【影片需求】
{film_brief}

【角色设定】
{characters}

【原故事大纲】
{current_story_outline}

【Story阶段可参考的长期Memory】
{story_memory_text}

Memory使用原则：
- 当前人工反馈和本轮Review优先于长期Memory。
- 长期Memory仅作为可参考的稳定偏好，不是硬性修订要求。
- 如果长期Memory与当前人工反馈、本轮Review或当前任务冲突，必须忽略Memory。
- Story修订不参考scene_preferences。

【当前人工意见——优先级最高】
{human_feedback_text}

【本轮机器审核问题——必须处理】
issues:
{story_review_issues}
suggestions:
{story_review_suggestions}

【仍需避免的历史问题】
{active_history_issues_text}

【已解决问题的防回归提醒】
{resolved_history_reminders_text}

这些问题当前已解决，不需要为了它们重新修改，只需避免本次修改让它们回归。

优先级规则：
最新人工意见
> 本轮机器审核问题
> 仍未解决或回归的历史问题
> 已解决问题的防回归提醒
> 长期 Memory

请输出完整的 StoryOutline，包含：

setup:
故事开端与初始状态

conflict:
核心冲突

turning_point:
改变冲突方向的关键转折

ending:
对核心冲突的回应与收束

theme:
核心主题

修改要求：

1. 如果存在当前人工意见，必须优先遵守当前人工意见。
2. 在满足人工意见的前提下，如果机器审核 issues 存在，必须处理 issues 指出的阻断性问题。
3. 机器审核 suggestions 仅作为优化参考，不需要全部采纳。
4. 修改时不得重新引入“仍需避免的历史问题”；历史为空时按“无历史问题”处理。
5. “已解决问题的防回归提醒”当前不需要重新修改，只需避免回归。
6. 只修改与问题相关的字段；没有问题的内容尽量保持原样。
7. 保留原故事的核心方向、主题和主要人物关系。
8. 保证 setup → conflict → turning_point → ending
   具有清晰、可理解的因果关系。
9. 不新增主要角色，不修改角色身份、性格和核心动机。
10. 不得让角色行为明确违反已有角色设定。
11. 修复问题时，不要引入新的逻辑断裂、角色冲突或无关剧情。
12. 故事复杂度应适配目标影片时长。
13. 每个字段保持简洁，通常使用1至2句话。
14. 只写故事骨架，不写分场数量、镜头、机位、摄影参数或详细拍摄动作。
15. continuity_constraints 无需逐条写进故事大纲，
    但不得明确描述与其冲突的状态变化。

请输出完整修改后的 StoryOutline。
不要解释修改过程。
"""

    new_story_outline:StoryOutline = (
        story_revise_llm.invoke(prompt)
    )

    return {
        "story_outline": new_story_outline,     # 修改后的故事大纲
        "story_revision_count":
            state.get(
                "story_revision_count",
                0
            ) + 1,      # 故事大纲修改次数 +1
        "current_stage":"story_revised_completed"  # 故事大纲修改完成

    }


# ================= Test =================

if __name__=="__main__":

    print(
        "请使用真实pipeline测试"
    )
