import json

from memory.context import format_scene_memory_context
from nodes import llm
from state import FilmState
from schemas import SceneList


# ================= LLM配置 =================

scene_revise_llm = llm.with_structured_output(SceneList) # 根据review反馈，对已有scene进行修复


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

    prompt = f"""
你是一名影视分场方案修改专家。

现在已有一份短片分场方案，审核阶段发现了一些问题。
你的任务是在保留原故事、角色和分场结构的基础上，
进行最小必要修改，而不是重新创作整套方案。

【影片需求】
{film_brief}

【角色设定】
{characters}

【故事大纲】
{story_outline}

【当前分场方案】
{current_scene_json}

【Scene阶段可参考的长期Memory】
{scene_memory_text}

Memory使用原则：
- 当前人工反馈和本轮Review优先于长期Memory。
- 长期Memory只作稳定偏好参考，不是硬性修订要求。
- 如果长期Memory与当前人工反馈、本轮Review或当前任务冲突，必须忽略Memory。
- story_preferences用于继承故事方向；scene_preferences用于约束分场执行。

【当前人工意见——优先级最高】
{human_feedback_text}

【本轮机器审核问题——必须处理】
issues:
{scene_review_issues}
suggestions:
{scene_review_suggestions}

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

修改要求：

1. 如果存在当前人工意见，必须优先遵守当前人工意见。
在满足人工意见的前提下，如果机器审核 issues 存在，必须处理其中明确指出的阻断性问题；suggestions 仅作为优化参考。
只修改存在问题的scene；没有问题的scene尽量保持原样。

2. 修改时不得重新引入“仍需避免的历史问题”；历史为空时按“无历史问题”处理。
“已解决问题的防回归提醒”当前不需要重新修改，只需避免回归。

3. 保持以下内容不变：
- 故事主题和核心冲突
- 角色身份、性格和核心动机
- setup → conflict → turning point → ending 的叙事顺序

4. 严格遵守角色continuity_constraints：
- 不得摘下、解开、替换、转移或改变被要求固定的服装、道具、外观和状态
- 如果原action明确违反约束，替换为不违反约束但叙事作用相近的动作
- 不需要在每个scene中重复说明所有约束仍然成立
- 未被明确改变的状态默认保持不变

5. 禁止：
- 新增角色
- 修改角色设定
- 为修复一个问题引入新的连续性冲突
- 添加与故事大纲无关的新事件

6. 分场结构要求：
- scene_id连续且不重复
- 场景数量符合影片需求
- 所有duration_sec之和等于目标时长
- characters只能使用已定义角色
- action描述可理解的剧情行为
- visual_goal描述该场景的叙事目的

7. 如果审核反馈包含推测性问题，
例如“某动作可能导致道具变化”或“未明确重复说明约束状态”，
但原action没有明确改变continuity_constraint，
则无需为此添加冗余说明。

8. 不要添加：
- 摄影机参数
- 机位、景别、运镜等详细镜头语言
- 视频生成prompt
- 与问题无关的额外细节

请输出完整修改后的SceneList。
"""

    new_scene_list = (
        scene_revise_llm.invoke(prompt)
    )

    return {
        "scenes": new_scene_list.scenes, # 修改后的scene列表
        "scene_revision_count":
            state.get(
                "scene_revision_count",
                0
            ) + 1, # 分场方案修改次数+1
        "current_stage": "scene_revised_completed" # 分场方案修改完成
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
