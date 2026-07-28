import json

from llm_profiles.factory import create_structured_llm
from memory.context import format_scene_memory_context
from observability.llm_calls import (
    collect_llm_call_trace,
    invoke_structured_llm,
)
from prompts.renderer import render_prompt
from state import FilmState
from schemas import SceneReviewResult, SceneCriticResult


#=============== 配置 LLM ===============
scene_critic_llm = create_structured_llm(
    "review.scene",
    SceneCriticResult,
)


# ======================== 规则审核 =============================


# 审核scene中必要字段是否完整
def check_scene_fields(state: FilmState):

    issues = []
    for scene in state["scenes"]:
        if scene.scene_id <= 0:
            issues.append(
                f"场景{scene.scene_id}中scene_id非法"
            )
        if scene.duration_sec <= 0:
            issues.append(
                f"场景{scene.scene_id}中duration_sec非法"
            )
        if not scene.location:
            issues.append(
                f"场景{scene.scene_id}缺少location"
            )
        if not scene.characters:
            issues.append(
                f"场景{scene.scene_id}缺少characters"
            )
        if not scene.action:
            issues.append(
                f"场景{scene.scene_id}缺少action"
            )
        if not scene.visual_goal:
            issues.append(
                f"场景{scene.scene_id}缺少visual_goal"
            )

    return issues



# 审核总时长
def check_duration(state: FilmState):

    issues = []
    target_duration = state["film_brief"].target_duration_sec
    actual_duration = sum(
        scene.duration_sec
        for scene in state["scenes"]
    )
    if actual_duration != target_duration:
        issues.append(
            f"总时长不一致：目标{target_duration}秒，实际{actual_duration}秒"
        )

    return issues


# 审核scene数量
def check_scene_count(state):

    issues=[]
    expected = (
        state["film_brief"]
        .recommended_scene_count
    )
    actual = len(state["scenes"])

    if abs(actual - expected) > 1:
        issues.append(
            f"场景数量异常："
            f"推荐{expected}个，"
            f"实际{actual}个"
        )
    return issues


# 审核角色一致性
def check_characters_consistency(state: FilmState):

    issues = []
    defined_characters = {
        character.name
        for character in state["characters"]
    }

    for scene in state["scenes"]:
        for character_name in scene.characters:
            if character_name not in defined_characters:
                issues.append(
                    f"场景{scene.scene_id}出现未定义角色：{character_name}"
                )
    return issues


# ======================= LLM Critic =============================


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


def llm_review(state: FilmState) -> SceneCriticResult:
    """
    使用LLM进行语义一致性审核
    """

    required_keys = [
        "film_brief",
        "characters",
        "story_outline",
        "scenes"
    ]

    for key in required_keys:
        if key not in state:
            raise ValueError(
                f"review缺少必要字段:{key}"
            )

    film_brief = (
        state["film_brief"]
        .model_dump()
    )

    characters = [
        c.model_dump()
        for c in state["characters"]
    ]

    story_outline = (
        state["story_outline"]
        .model_dump()
    )

    scenes = [
        s.model_dump()
        for s in state["scenes"]
    ]
    user_idea = state.get(
        "user_idea",
        "未提供",
    )
    scene_memory_text = format_scene_memory_context(
        state.get("user_memory")
    )    # Scene审核同时参考故事偏好和分场偏好，当前任务仍优先

    history_issue_context = _collect_history_issue_context(
        state.get("scene_review_history", [])
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
        "review.scene",
        version="v1",
        film_brief=film_brief,
        user_idea=user_idea,
        characters=characters,
        story_outline=story_outline,
        scenes=scenes,
        scene_memory_text=scene_memory_text,
        history_issues_text=history_issues_text,
    )
    result = invoke_structured_llm(
        scene_critic_llm,
        rendered,
        node="review_scene",
    )

    # 兼容部分模型返回dict
    if isinstance(result, dict):
        result = SceneCriticResult(**result)

    return result


# ======================= 主审核节点 =============================


def review_scene(state: FilmState):

    issues = []
    suggestions = []

    # 规则检查
    issues.extend(
        check_scene_fields(state)
    )
    issues.extend(
        check_duration(state)
    )
    issues.extend(
        check_scene_count(state)
    )
    issues.extend(
        check_characters_consistency(state)
    )

    # LLM检查
    with collect_llm_call_trace() as llm_call_events:
        scene_critic_result = llm_review(state)
    issues.extend(
        scene_critic_result.issues
    )
    suggestions.extend(
        scene_critic_result.suggestions
    )

    historical_issue_checks = (
        scene_critic_result.historical_issue_checks
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
        and scene_critic_result.passed
    )

    scene_review_result = SceneReviewResult(
        passed=passed,
        issues=issues,
        suggestions=suggestions,
        historical_issue_checks=historical_issue_checks,
    )

    return {
        "scene_review_result": scene_review_result,
        "scene_review_history": [
            {
                "revision_round": state.get(
                    "scene_revision_count",
                    0,
                ),
                "passed": scene_review_result.passed,
                "issues": scene_review_result.issues,
                "suggestions": scene_review_result.suggestions,
                "historical_issue_checks": [
                    check.model_dump()
                    for check
                    in scene_review_result.historical_issue_checks
                ],
            }
        ],
        "current_stage": "scene_review_completed",
        "llm_call_trace": llm_call_events,

    }


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    from schemas import FilmBrief, StoryOutline, Scene, Character


    characters = [
        Character(
            name="林砚",
            role="摄影成员",
            appearance="黑框眼镜",
            personality=["沉静"],
            motivation="记录真实瞬间",
            continuity_constraints=[
                "始终佩戴眼镜"
            ]
        ),
        Character(
            name="陈屿",
            role="纪念册主编",
            appearance="白衬衫",
            personality=["认真"],
            motivation="保存毕业记忆",
            continuity_constraints=[
                "始终携带纪念册"
            ]
        ),
        Character(
            name="许昭",
            role="协调者",
            appearance="学士服",
            personality=["热情"],
            motivation="帮助大家完成毕业照",
            continuity_constraints=[
                "始终佩戴校牌"
            ]
        )
    ]

    state: FilmState = {
        "film_brief": FilmBrief(
            target_duration_sec=60,
            genre="青春校园",
            core_theme="毕业告别",
            visual_style="温暖青春",
            recommended_scene_count=6
        ),
        "characters": characters,
        "story_outline": StoryOutline(
            setup="毕业照拍摄前，三人共同准备毕业仪式。",
            conflict="三人面对离别时采用不同方式，希望延长共同时间。",
            turning_point="拍摄最后一组照片时，三人意识到毕业即将结束。",
            ending="三人接受分别，并保存共同回忆。",
            theme="告别不是失去，而是记住共同经历。"
        ),
        "scenes": [
            Scene(
                scene_id=1,
                duration_sec=20,
                location="操场",
                characters=[
                    "林砚",
                    "陈屿",
                    "许昭"
                ],
                action="准备毕业照",
                dialogue="",
                visual_goal="建立人物关系"
            ),
            Scene(
                scene_id=2,
                duration_sec=40,
                location="教学楼",
                characters=[
                    "林砚",
                    "陈屿",
                    "未知角色"
                ],
                action="讨论毕业",
                dialogue="",
                visual_goal="表现离别情绪"
            )
        ]
    }

    result = review_scene(state)

    print(
        json.dumps(
            {
                "scene_review_result":
                result["scene_review_result"]
                .model_dump()
            },
            ensure_ascii=False,
            indent=2
        )
    )
