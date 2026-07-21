import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.context import format_story_memory_context
from memory.models import UserMemory
from schemas import (
    Character,
    FilmBrief,
    StoryCriticResult,
    StoryOutline,
    StoryReviewResult,
)

nodes_module = importlib.import_module("nodes")
review_story_module = importlib.import_module("reviews.review_story")
revise_story_module = importlib.import_module("revisions.revise_story")


class FakeStructuredLLM:
    """
    捕获Prompt并返回预设结构化结果，避免调用真实LLM。
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
    截取Prompt中的指定段落，避免规则说明里的字段名影响判断。
    """
    start = prompt.index(heading) + len(heading)
    end = prompt.index(next_heading, start)
    return prompt[start:end]


def _film_brief() -> FilmBrief:
    """
    构造测试影片需求。
    """
    return FilmBrief(
        target_duration_sec=60,
        genre="校园",
        core_theme="成长",
        visual_style="现实主义",
        recommended_scene_count=5,
    )


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


def _story_memory() -> UserMemory:
    """
    构造包含Story和Scene偏好的长期Memory。
    """
    return UserMemory(
        user_id="demo_user",
        preferred_genres=["校园"],
        style_preferences=["现实主义"],
        disliked_elements=["大量旁白"],
        preferred_duration_sec=60,
        additional_preferences=["少角色"],
        story_preferences=["故事结尾保持克制开放"],
        scene_preferences=["分场动作保持可拍摄"],
    )


def _base_state(user_memory=None) -> dict:
    """
    构造Story节点所需最小State。
    """
    return {
        "user_idea": "这次生成一个夸张卡通化的校园故事。",
        "film_brief": _film_brief(),
        "characters": [_character()],
        "story_outline": _story_outline(),
        "story_review_result": StoryReviewResult(
            passed=False,
            issues=["本轮机器问题"],
            suggestions=["本轮机器建议"],
        ),
        "story_review_history": [],
        "human_feedback": "最新人工意见",
        "story_revision_count": 0,
        "user_memory": user_memory,
    }


def test_format_story_memory_context_excludes_scene_preferences():
    """
    Story Memory Context只包含Story相关字段，不包含scene_preferences。
    """
    context = format_story_memory_context(
        _story_memory()
    )

    assert "preferred_genres" in context
    assert "style_preferences" in context
    assert "disliked_elements" in context
    assert "preferred_duration_sec" in context
    assert "additional_preferences" in context
    assert "story_preferences" in context
    assert "故事结尾保持克制开放" in context
    assert "scene_preferences" not in context
    assert "分场动作保持可拍摄" not in context


def test_empty_story_memory_context_is_safe():
    """
    空Memory应输出无长期偏好，不能报错。
    """
    assert format_story_memory_context(None) == "无长期偏好"
    assert (
        format_story_memory_context(
            UserMemory(user_id="demo_user")
        )
        == "无长期偏好"
    )


def test_plan_story_prompt_uses_story_memory(monkeypatch):
    """
    plan_story Prompt应包含Story Memory，并保留当前用户要求。
    """
    fake_llm = FakeStructuredLLM(
        _story_outline()
    )
    monkeypatch.setattr(
        nodes_module,
        "story_outline_llm",
        fake_llm,
    )

    nodes_module.plan_story(
        _base_state(user_memory=_story_memory())
    )

    prompt = fake_llm.prompts[-1]
    memory_section = _prompt_section(
        prompt,
        "【Story阶段可参考的长期Memory】",
        "Memory使用原则：",
    )

    assert "【Story阶段可参考的长期Memory】" in prompt
    assert "故事结尾保持克制开放" in memory_section
    assert "scene_preferences" not in memory_section
    assert "分场动作保持可拍摄" not in memory_section
    assert "这次生成一个夸张卡通化的校园故事" in prompt
    assert "当前用户要求优先于长期Memory" in prompt
    assert "与当前任务冲突" in prompt


def test_review_story_prompt_uses_story_memory(monkeypatch):
    """
    review_story Prompt应包含Story Memory，并保留审核输入。
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
        _base_state(user_memory=_story_memory())
    )

    prompt = fake_llm.prompts[-1]
    memory_section = _prompt_section(
        prompt,
        "【Story阶段可参考的长期Memory】",
        "Memory使用原则：",
    )

    assert "【Story阶段可参考的长期Memory】" in prompt
    assert "故事结尾保持克制开放" in memory_section
    assert "scene_preferences" not in memory_section
    assert "分场动作保持可拍摄" not in memory_section
    assert "这次生成一个夸张卡通化的校园故事" in prompt
    assert "当前用户要求优先于长期Memory" in prompt
    assert "如果长期Memory与当前任务冲突" in prompt


def test_revise_story_prompt_uses_story_memory_with_lower_priority(monkeypatch):
    """
    revise_story Prompt应保留人工反馈和Review内容，并声明它们优先于Memory。
    """
    fake_llm = FakeStructuredLLM(
        _story_outline()
    )
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    revise_story_module.revise_story(
        _base_state(user_memory=_story_memory())
    )

    prompt = fake_llm.prompts[-1]
    memory_section = _prompt_section(
        prompt,
        "【Story阶段可参考的长期Memory】",
        "Memory使用原则：",
    )

    assert "【Story阶段可参考的长期Memory】" in prompt
    assert "故事结尾保持克制开放" in memory_section
    assert "scene_preferences" not in memory_section
    assert "分场动作保持可拍摄" not in memory_section
    assert "最新人工意见" in prompt
    assert "本轮机器问题" in prompt
    assert "本轮机器建议" in prompt
    assert "当前人工反馈和本轮Review优先于长期Memory" in prompt
    assert "如果长期Memory与当前人工反馈、本轮Review或当前任务冲突" in prompt


def test_plan_story_prompt_handles_empty_memory(monkeypatch):
    """
    plan_story遇到空Memory时应写入无长期偏好。
    """
    fake_llm = FakeStructuredLLM(
        _story_outline()
    )
    monkeypatch.setattr(
        nodes_module,
        "story_outline_llm",
        fake_llm,
    )

    nodes_module.plan_story(
        _base_state(user_memory=None)
    )

    assert "无长期偏好" in fake_llm.prompts[-1]
