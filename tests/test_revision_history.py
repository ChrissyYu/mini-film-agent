import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import (
    Character,
    FilmBrief,
    Scene,
    SceneList,
    SceneReviewResult,
    StoryOutline,
    StoryReviewResult,
)

revise_story_module = importlib.import_module("revisions.revise_story")
revise_scene_module = importlib.import_module("revisions.revise_scene")


class FakeStructuredLLM:
    """
    捕获Prompt并返回预设修订结果，避免调用真实LLM。
    """

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.result


def _prompt_section(
    prompt: str,
    heading: str,
    next_heading: str,
) -> str:
    """
    截取Prompt中的指定段落，避免只靠全文包含误判。
    """
    start = prompt.index(heading) + len(heading)
    end = prompt.index(next_heading, start)
    return prompt[start:end]


def _character() -> Character:
    """
    构造测试角色。
    """
    return Character(
        name="林夏",
        role="主角",
        appearance="白衬衫",
        personality=["克制"],
        motivation="面对告别",
        continuity_constraints=[],
    )


def _film_brief() -> FilmBrief:
    """
    构造测试影片需求。
    """
    return FilmBrief(
        target_duration_sec=10,
        genre="校园",
        core_theme="成长",
        visual_style="自然克制",
        recommended_scene_count=1,
    )


def _story_outline() -> StoryOutline:
    """
    构造测试故事大纲。
    """
    return StoryOutline(
        setup="林夏毕业前夕准备离校。",
        conflict="她想留下却必须离开。",
        turning_point="她意识到离开也是成长。",
        ending="她平静告别校园。",
        theme="克制的成长。",
    )


def _scene() -> Scene:
    """
    构造测试分场。
    """
    return Scene(
        scene_id=1,
        duration_sec=10,
        location="校园",
        characters=["林夏"],
        action="林夏平静走出校门。",
        dialogue="",
        visual_goal="表现克制告别。",
    )


def _story_state(
    story_review_history=None,
    human_feedback="结尾改成开放式。",
) -> dict:
    """
    构造Story Revision所需State。
    """
    return {
        "film_brief": _film_brief(),
        "characters": [_character()],
        "story_outline": _story_outline(),
        "story_review_result": StoryReviewResult(
            passed=False,
            issues=["本轮机器问题"],
            suggestions=["本轮机器建议"],
        ),
        "story_review_history": story_review_history or [],
        "human_feedback": human_feedback,
        "story_revision_count": 0,
    }


def _scene_state(
    scene_review_history=None,
    human_feedback="减少对白。",
) -> dict:
    """
    构造Scene Revision所需State。
    """
    return {
        "film_brief": _film_brief(),
        "characters": [_character()],
        "story_outline": _story_outline(),
        "scenes": [_scene()],
        "scene_review_result": SceneReviewResult(
            passed=False,
            issues=["本轮分场问题"],
            suggestions=["本轮分场建议"],
        ),
        "scene_review_history": scene_review_history or [],
        "human_feedback": human_feedback,
        "scene_revision_count": 0,
    }


def test_story_history_issues_are_cleaned_deduped_and_added_to_prompt(monkeypatch):
    """
    Story旧格式历史issues应按unknown进入仍需避免清单。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": [" 旧问题A ", "", "旧问题B"],
                    "suggestions": ["历史建议不要进入"],
                },
                {
                    "revision_round": 1,
                    "passed": False,
                    "issues": ["旧问题A", "旧问题C"],
                    "suggestions": [],
                },
            ]
        )
    )

    prompt = fake_llm.prompts[-1]

    assert "旧问题A" in prompt
    assert "旧问题B" in prompt
    assert "旧问题C" in prompt
    assert prompt.count("旧问题A") == 1
    assert "历史建议不要进入" not in prompt
    assert "【仍需避免的历史问题】" in prompt


def test_scene_history_issues_are_added_to_prompt(monkeypatch):
    """
    Scene历史issues应进入仍需避免清单。
    """
    fake_llm = FakeStructuredLLM(
        SceneList(scenes=[_scene()])
    )
    monkeypatch.setattr(
        revise_scene_module,
        "scene_revise_llm",
        fake_llm,
    )

    revise_scene_module.revise_scene(
        _scene_state(
            scene_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": ["角色连续性冲突"],
                    "suggestions": [],
                }
            ]
        )
    )

    prompt = fake_llm.prompts[-1]

    assert "角色连续性冲突" in prompt
    assert "【仍需避免的历史问题】" in prompt


def test_history_suggestions_are_not_used_as_regression_constraints(monkeypatch):
    """
    历史suggestions不应进入历史问题回归约束。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": ["旧问题A"],
                    "suggestions": ["历史建议不要进入"],
                }
            ]
        )
    )

    assert "历史建议不要进入" not in fake_llm.prompts[-1]


def test_history_issues_are_limited_to_recent_eight(monkeypatch):
    """
    仍需避免的历史问题超过8条时，只传入最近8条。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": index,
                    "passed": False,
                    "issues": [f"旧问题{index}"],
                    "suggestions": [],
                }
                for index in range(10)
            ]
        )
    )

    prompt = fake_llm.prompts[-1]

    assert "旧问题0" not in prompt
    assert "旧问题1" not in prompt
    for index in range(2, 10):
        assert f"旧问题{index}" in prompt


def test_empty_history_works(monkeypatch):
    """
    历史为空时应写入无历史问题并正常修订。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    result = revise_story_module.revise_story(
        _story_state(story_review_history=[])
    )

    assert "无历史问题" in fake_llm.prompts[-1]
    assert result["story_revision_count"] == 1
    assert result["current_stage"] == "story_revised_completed"


def test_current_human_feedback_and_machine_review_still_in_prompt(monkeypatch):
    """
    当前人工反馈和本轮机器审核问题仍应存在于Prompt中。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _story_state(
            story_review_history=[],
            human_feedback="最新人工意见",
        )
    )

    prompt = fake_llm.prompts[-1]

    assert "【当前人工意见——优先级最高】" in prompt
    assert "最新人工意见" in prompt
    assert "【本轮机器审核问题——必须处理】" in prompt
    assert "本轮机器问题" in prompt
    assert "本轮机器建议" in prompt
    assert "最新人工意见\n> 本轮机器审核问题\n> 仍未解决或回归的历史问题\n> 已解决问题的防回归提醒\n> 长期 Memory" in prompt


def test_status_filters_active_and_resolved_history_sections(monkeypatch):
    """
    unresolved/regressed进入仍需避免，resolved只进入防回归提醒。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": [
                        "未解决问题",
                        "回归问题",
                        "已解决问题",
                    ],
                    "suggestions": ["历史建议不要进入"],
                    "historical_issue_checks": [
                        {
                            "issue": "未解决问题",
                            "status": "unresolved",
                            "evidence": "仍存在",
                        },
                        {
                            "issue": "回归问题",
                            "status": "regressed",
                            "evidence": "再次出现",
                        },
                        {
                            "issue": "已解决问题",
                            "status": "resolved",
                            "evidence": "已不存在",
                        },
                    ],
                }
            ]
        )
    )

    prompt = fake_llm.prompts[-1]
    active_section = _prompt_section(
        prompt,
        "【仍需避免的历史问题】",
        "【已解决问题的防回归提醒】",
    )
    resolved_section = _prompt_section(
        prompt,
        "【已解决问题的防回归提醒】",
        "优先级规则：",
    )

    assert "未解决问题" in active_section
    assert "回归问题" in active_section
    assert "已解决问题" not in active_section
    assert "已解决问题" in resolved_section
    assert "历史建议不要进入" not in prompt


def test_resolved_regression_reminders_are_limited_to_two(monkeypatch):
    """
    resolved防回归提醒最多保留最近2条。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": index,
                    "passed": True,
                    "issues": [f"已解决问题{index}"],
                    "suggestions": [],
                    "historical_issue_checks": [
                        {
                            "issue": f"已解决问题{index}",
                            "status": "resolved",
                            "evidence": "已解决",
                        }
                    ],
                }
                for index in range(4)
            ]
        )
    )

    prompt = fake_llm.prompts[-1]
    resolved_section = _prompt_section(
        prompt,
        "【已解决问题的防回归提醒】",
        "优先级规则：",
    )

    assert "已解决问题0" not in resolved_section
    assert "已解决问题1" not in resolved_section
    assert "已解决问题2" in resolved_section
    assert "已解决问题3" in resolved_section


def test_same_issue_uses_latest_status(monkeypatch):
    """
    同一issue应以最近一次状态为准。
    """
    fake_llm = FakeStructuredLLM(_story_outline())
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": ["反复问题"],
                    "suggestions": [],
                    "historical_issue_checks": [
                        {
                            "issue": "反复问题",
                            "status": "unresolved",
                            "evidence": "仍存在",
                        }
                    ],
                },
                {
                    "revision_round": 1,
                    "passed": True,
                    "issues": [],
                    "suggestions": [],
                    "historical_issue_checks": [
                        {
                            "issue": "反复问题",
                            "status": "resolved",
                            "evidence": "已修复",
                        }
                    ],
                },
            ]
        )
    )

    prompt = fake_llm.prompts[-1]
    active_section = _prompt_section(
        prompt,
        "【仍需避免的历史问题】",
        "【已解决问题的防回归提醒】",
    )
    resolved_section = _prompt_section(
        prompt,
        "【已解决问题的防回归提醒】",
        "优先级规则：",
    )

    assert "反复问题" not in active_section
    assert "反复问题" in resolved_section


def test_scene_status_filtered_history_sections(monkeypatch):
    """
    Scene链路也应根据最新状态拆分历史约束。
    """
    fake_llm = FakeStructuredLLM(
        SceneList(scenes=[_scene()])
    )
    monkeypatch.setattr(
        revise_scene_module,
        "scene_revise_llm",
        fake_llm,
    )

    revise_scene_module.revise_scene(
        _scene_state(
            scene_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": [
                        "分场仍需避免",
                        "分场已解决",
                    ],
                    "suggestions": ["Scene历史建议不要进入"],
                    "historical_issue_checks": [
                        {
                            "issue": "分场仍需避免",
                            "status": "regressed",
                            "evidence": "再次出现",
                        },
                        {
                            "issue": "分场已解决",
                            "status": "resolved",
                            "evidence": "已解决",
                        },
                    ],
                }
            ]
        )
    )

    prompt = fake_llm.prompts[-1]
    active_section = _prompt_section(
        prompt,
        "【仍需避免的历史问题】",
        "【已解决问题的防回归提醒】",
    )
    resolved_section = _prompt_section(
        prompt,
        "【已解决问题的防回归提醒】",
        "优先级规则：",
    )

    assert "分场仍需避免" in active_section
    assert "分场已解决" not in active_section
    assert "分场已解决" in resolved_section
    assert "Scene历史建议不要进入" not in prompt
    assert "本轮分场问题" in prompt
    assert "本轮分场建议" in prompt
