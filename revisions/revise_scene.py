import json

from llm_profiles.factory import create_structured_llm
from memory.context import format_scene_memory_context
from observability.llm_calls import (
    collect_llm_call_trace,
    invoke_structured_llm,
)
from prompts.renderer import render_prompt
from state import FilmState
from schemas import SceneList


# ================= LLM配置 =================

scene_revise_llm = create_structured_llm(
    "revision.scene",
    SceneList,
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

# 根据review反馈，对已有scene进行修复
def revise_scene(state: FilmState) -> dict:

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

    current_scenes = [
        s.model_dump()
        for s in state["scenes"]
    ]
    scene_memory_text = format_scene_memory_context(
        state.get("user_memory")
    )    # Scene修订同时参考故事偏好和分场偏好，但人工反馈与本轮Review优先

    scene_review_result = (
        state["scene_review_result"]
        .model_dump()
    )
    scene_review_issues = scene_review_result.get(
        "issues",
        [],
    )
    scene_review_suggestions = scene_review_result.get(
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
            state.get("scene_review_history", [])
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

    current_scene_json = json.dumps(
        current_scenes,
        ensure_ascii=False,
        indent=2
    )

    rendered = render_prompt(
        "revision.scene",
        version="v1",
        film_brief=film_brief,
        characters=characters,
        story_outline=story_outline,
        current_scene_json=current_scene_json,
        scene_memory_text=scene_memory_text,
        human_feedback_text=human_feedback_text,
        scene_review_issues=scene_review_issues,
        scene_review_suggestions=scene_review_suggestions,
        active_history_issues_text=(
            active_history_issues_text
        ),
        resolved_history_reminders_text=(
            resolved_history_reminders_text
        ),
    )

    with collect_llm_call_trace() as llm_call_events:
        new_scene_list = (
            invoke_structured_llm(
                scene_revise_llm,
                rendered,
                node="revise_scene",
            )
        )

    return {
        "scenes": new_scene_list.scenes, # 修改后的scene列表
        "scene_revision_count":
            state.get(
                "scene_revision_count",
                0
            ) + 1, # 分场方案修改次数+1
        "current_stage": "scene_revised_completed", # 分场方案修改完成
        "llm_call_trace": llm_call_events,
    }

# ================= 测试 =================


if __name__ == "__main__":

    from schemas import (
        FilmBrief,
        StoryOutline,
        Scene,
        Character,
        ReviewResult
    )

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
            motivation="帮助完成毕业照",
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
        ), # 影片需求

        "characters": characters, # 角色设定

        "story_outline": StoryOutline(
            setup="三人准备毕业照",
            conflict="面对离别产生不同情绪",
            turning_point="最后一组照片意识到告别",
            ending="接受分别并保存回忆",
            theme="告别不是失去"
        ), # 故事大纲

        "scenes":[
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
                visual_goal="建立关系"
            ),
            Scene(
                scene_id=2,
                duration_sec=40,
                location="教学楼",
                characters=[
                    "林砚",
                    "未知角色"
                ],
                action="讨论毕业",
                dialogue="",
                visual_goal="表现离别"
            )
        ],

        "review_result": ReviewResult(
            passed=False,
            issues=[
                "scene数量不足",
                "scene2存在未知角色"
            ],
            suggestions=[
                "增加场景",
                "删除未知角色"
            ]
        ),
        "scene_revision_count":0,
        "story_revision_count": 0
    }

    result = revise_scene(state)

    print(
        json.dumps(
            {
                "scenes":[
                    s.model_dump()
                    for s in result["scenes"]
                ]
            },
            ensure_ascii=False,
            indent=2
        )
    )
