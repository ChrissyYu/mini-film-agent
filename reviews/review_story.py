import json

from memory.context import format_story_memory_context
from nodes import llm
from state import FilmState
from schemas import StoryReviewResult, StoryCriticResult


# ================= LLM配置 =================

story_critic_llm = llm.with_structured_output(StoryCriticResult)


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

    prompt = f"""
你是一名专业影视编剧审核专家。

你的任务是判断这个短片故事大纲是否已经足以进入分场规划阶段，
不是重新创作故事，也不是寻找所有可以优化的细节。

【影片需求】
{film_brief}

【用户本次要求】
{user_idea}

【角色设定】
{characters}

【故事大纲】
{story_outline}

【Story阶段可参考的长期Memory】
{story_memory_text}

Memory使用原则：
- 当前用户要求优先于长期Memory。
- 长期Memory仅作为可参考的稳定偏好。
- 如果长期Memory与当前任务冲突，审核时忽略Memory，以当前任务为准。
- Story审核不参考scene_preferences。

【此前曾发现的问题及最近状态】
{history_issues_text}

回归检查要求：
- 请检查此前曾发现的问题是否已经解决；
- 请检查这些旧问题是否在本轮故事大纲中重新出现；
- 同时检查本轮是否产生新的阻断性问题；
- 历史为空时按“无历史问题”处理，不要臆造旧问题。
- 如果存在历史问题，请在 historical_issue_checks 中逐条输出状态判断；
- resolved 表示历史问题在当前版本中已不存在；
- unresolved 表示历史问题一直没有解决；
- regressed 表示该问题最近一次已是 resolved，但当前版本再次出现；
- 若某个历史问题是 unresolved 或 regressed，issues 中必须包含该问题或等价描述，且 passed 不能为 true；
- resolved 的历史问题不要继续加入当前 issues。

请审核以下内容：

1. 故事结构
- setup 是否建立了基本情境
- conflict 是否存在清晰矛盾
- turning_point 是否改变了冲突方向
- ending 是否回应了核心冲突

2. 因果逻辑
- 关键事件之间是否基本连贯
- 是否存在严重到无法理解的逻辑跳跃
- 轻微铺垫不足如果可在分场阶段补足，不算严重问题

3. 角色一致性
- 主要角色的关键选择是否明显违背核心动机
- 是否出现承担关键剧情功能但未定义的主要角色
- 不要因为细小动作未体现角色习惯而判定失败

4. 主题与时长
- 是否符合影片主题和类型
- 故事是否复杂到无法在目标时长内表达

审核边界：
只审核故事大纲层面的核心问题。
不要审核具体镜头、场景数量、构图、道具摆放、摄影流程或细节动作，
这些属于后续 Scene Planning 和 Scene Review。

passed 判定：
- 只有存在阻断后续分场生成的严重问题时，passed=false
- 故事核心成立、主要因果可理解时，passed=true
- passed=true 不代表故事完美

输出要求：

passed:
故事是否足以进入分场规划阶段

issues:
只列出会导致 passed=false 的严重问题

suggestions:
列出非阻断性的优化建议，可以为空

不要重新生成故事。
"""

    result = story_critic_llm.invoke(prompt)

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
        "current_stage": "story_review_completed"
    }
