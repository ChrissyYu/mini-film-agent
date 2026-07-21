import sys
from pathlib import Path
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import (
    Character,
    FilmBrief,
    Scene,
    SceneCriticResult,
    StoryCriticResult,
    StoryOutline,
)

review_story_module = importlib.import_module("reviews.review_story")
review_scene_module = importlib.import_module("reviews.review_scene")


class FakeStructuredLLM:
    """
    捕获Prompt并返回预设审核结果，避免调用真实LLM。
    """

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.result


def _story_state(
    story_review_history=None,
    story_revision_count=0,
):
    """
    构造Story Review所需最小State。
    """
    return {
        "film_brief": FilmBrief(
            target_duration_sec=60,
            genre="校园",
            core_theme="成长",
            visual_style="自然克制",
            recommended_scene_count=5,
        ),
        "characters": [
            Character(
                name="林夏",
                role="主角",
                appearance="白衬衫",
                personality=["克制"],
                motivation="面对告别",
                continuity_constraints=[],
            )
        ],
        "story_outline": StoryOutline(
            setup="林夏毕业前夕准备离校。",
            conflict="她想留下却必须离开。",
            turning_point="她意识到离开也是成长。",
            ending="她平静告别校园。",
            theme="克制的成长。",
        ),
        "story_revision_count": story_revision_count,
        "story_review_history": story_review_history or [],
    }


def _scene_state(
    scene_review_history=None,
    scene_revision_count=0,
):
    """
    构造Scene Review所需最小State。
    """
    character = Character(
        name="林夏",
        role="主角",
        appearance="白衬衫",
        personality=["克制"],
        motivation="面对告别",
        continuity_constraints=[],
    )

    return {
        "film_brief": FilmBrief(
            target_duration_sec=10,
            genre="校园",
            core_theme="成长",
            visual_style="自然克制",
            recommended_scene_count=1,
        ),
        "characters": [
            character,
        ],
        "story_outline": StoryOutline(
            setup="林夏毕业前夕准备离校。",
            conflict="她想留下却必须离开。",
            turning_point="她意识到离开也是成长。",
            ending="她平静告别校园。",
            theme="克制的成长。",
        ),
        "scenes": [
            Scene(
                scene_id=1,
                duration_sec=10,
                location="校园",
                characters=[
                    "林夏",
                ],
                action="林夏平静走出校门。",
                dialogue="",
                visual_goal="表现克制告别。",
            )
        ],
        "scene_revision_count": scene_revision_count,
        "scene_review_history": scene_review_history or [],
    }


def test_story_review_history_accumulates_without_overwrite(monkeypatch):
    """
    多轮Story Review应返回可累积的单条历史记录。
    """
    first_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=False,
            issues=[
                "核心冲突不明确",
            ],
            suggestions=[
                "强化人物目标",
            ],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        first_llm,
    )

    first_result = review_story_module.review_story(
        _story_state(story_revision_count=0)
    )
    history = first_result["story_review_history"]

    second_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[
                "结尾可以更克制",
            ],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        second_llm,
    )

    second_result = review_story_module.review_story(
        _story_state(
            story_review_history=history,
            story_revision_count=1,
        )
    )
    history = history + second_result["story_review_history"]

    assert len(history) == 2
    assert history[0]["revision_round"] == 0
    assert history[0]["issues"] == [
        "核心冲突不明确",
    ]
    assert history[1]["revision_round"] == 1
    assert history[1]["issues"] == []


def test_scene_review_history_accumulates(monkeypatch):
    """
    多轮Scene Review应返回可累积的单条历史记录。
    """
    first_llm = FakeStructuredLLM(
        SceneCriticResult(
            passed=False,
            issues=[
                "场景动作不清晰",
            ],
            suggestions=[],
        )
    )
    monkeypatch.setattr(
        review_scene_module,
        "scene_critic_llm",
        first_llm,
    )

    first_result = review_scene_module.review_scene(
        _scene_state(scene_revision_count=0)
    )
    history = first_result["scene_review_history"]

    second_llm = FakeStructuredLLM(
        SceneCriticResult(
            passed=True,
            issues=[],
            suggestions=[
                "视觉目标可以更具体",
            ],
        )
    )
    monkeypatch.setattr(
        review_scene_module,
        "scene_critic_llm",
        second_llm,
    )

    second_result = review_scene_module.review_scene(
        _scene_state(
            scene_review_history=history,
            scene_revision_count=1,
        )
    )
    history = history + second_result["scene_review_history"]

    assert len(history) == 2
    assert history[0]["revision_round"] == 0
    assert history[0]["issues"] == [
        "场景动作不清晰",
    ]
    assert history[1]["revision_round"] == 1


def test_story_history_issues_are_deduplicated_in_prompt(monkeypatch):
    """
    历史issues应去空、去重，并进入下一轮Story Review Prompt。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    review_story_module.review_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": [
                        "核心冲突不明确",
                        "",
                    ],
                    "suggestions": [
                        "不应进入Prompt",
                    ],
                },
                {
                    "revision_round": 1,
                    "passed": False,
                    "issues": [
                        "核心冲突不明确",
                        "结尾没有回应冲突",
                    ],
                    "suggestions": [
                        "也不应进入Prompt",
                    ],
                },
            ],
            story_revision_count=2,
        )
    )

    prompt = fake_llm.prompts[-1]

    assert "核心冲突不明确" in prompt
    assert "结尾没有回应冲突" in prompt
    assert prompt.count("核心冲突不明确") == 1
    assert "不应进入Prompt" not in prompt
    assert "也不应进入Prompt" not in prompt


def test_scene_history_issues_are_used_for_regression_check(monkeypatch):
    """
    已解决的旧Scene问题仍应作为回归检查项进入Prompt。
    """
    fake_llm = FakeStructuredLLM(
        SceneCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
        )
    )
    monkeypatch.setattr(
        review_scene_module,
        "scene_critic_llm",
        fake_llm,
    )

    review_scene_module.review_scene(
        _scene_state(
            scene_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": [
                        "场景总时长不一致",
                    ],
                    "suggestions": [],
                }
            ],
            scene_revision_count=1,
        )
    )

    prompt = fake_llm.prompts[-1]

    assert "场景总时长不一致" in prompt
    assert "请检查此前曾发现的问题是否已经解决" in prompt
    assert "重新出现" in prompt


def test_empty_history_works_and_mentions_no_history(monkeypatch):
    """
    历史为空时应正常工作，并明确写入无历史问题。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    result = review_story_module.review_story(
        _story_state()
    )

    assert "无历史问题" in fake_llm.prompts[-1]
    assert result["story_review_result"].passed is True


def test_existing_review_return_fields_are_preserved(monkeypatch):
    """
    新增history后，现有审核返回字段仍应保留。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[
                "可强化主题表达",
            ],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    result = review_story_module.review_story(
        _story_state()
    )

    assert "story_review_result" in result
    assert "current_stage" in result
    assert "story_review_history" in result
    assert result["current_stage"] == "story_review_completed"
