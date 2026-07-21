import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.context import format_scene_memory_context, format_story_memory_context
from memory.models import UserMemory
from schemas import (
    Character,
    FilmBrief,
    Scene,
    SceneCriticResult,
    SceneList,
    SceneReviewResult,
    StoryOutline,
)

nodes_module = importlib.import_module("nodes")
review_scene_module = importlib.import_module("reviews.review_scene")
revise_scene_module = importlib.import_module("revisions.revise_scene")


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
    截取Prompt中的指定段落，避免规则说明影响字段判断。
    """
    start = prompt.index(heading) + len(heading)
    end = prompt.index(next_heading, start)
    return prompt[start:end]


def _film_brief() -> FilmBrief:
    """
    构造测试影片需求。
    """
    return FilmBrief(
        target_duration_sec=10,
        genre="校园",
        core_theme="成长",
        visual_style="现实主义",
        recommended_scene_count=1,
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


def _scene_memory() -> UserMemory:
    """
    构造包含七类Scene可读字段的长期Memory。
    """
    return UserMemory(
        user_id="demo_user",
        preferred_genres=["校园"],
        style_preferences=["现实主义"],
        disliked_elements=["大量旁白"],
        preferred_duration_sec=10,
        additional_preferences=["少角色"],
        story_preferences=["故事结尾保持克制开放"],
        scene_preferences=["分场动作保持可拍摄"],
    )


def _base_state(user_memory=None) -> dict:
    """
    构造Scene节点所需最小State。
    """
    return {
        "user_idea": "这次生成一个夸张卡通化的校园故事。",
        "film_brief": _film_brief(),
        "characters": [_character()],
        "story_outline": _story_outline(),
        "scenes": [_scene()],
        "scene_review_result": SceneReviewResult(
            passed=False,
            issues=["本轮分场问题"],
            suggestions=["本轮分场建议"],
        ),
        "scene_review_history": [
            {
                "revision_round": 0,
                "passed": False,
                "issues": ["历史分场问题"],
                "suggestions": [],
            }
        ],
        "human_feedback": "最新人工反馈",
        "scene_revision_count": 0,
        "user_memory": user_memory,
    }


def test_format_scene_memory_context_includes_all_scene_fields():
    """
    Scene Memory Context应包含七类字段，并同时保留story/scene偏好。
    """
    context = format_scene_memory_context(
        _scene_memory()
    )

    assert "preferred_genres" in context
    assert "style_preferences" in context
    assert "disliked_elements" in context
    assert "preferred_duration_sec" in context
    assert "additional_preferences" in context
    assert "story_preferences" in context
    assert "scene_preferences" in context
    assert "故事结尾保持克制开放" in context
    assert "分场动作保持可拍摄" in context


def test_empty_scene_memory_context_is_safe():
    """
    空Memory应输出无长期偏好，不能报错。
    """
    assert format_scene_memory_context(None) == "无长期偏好"
    assert (
        format_scene_memory_context(
            UserMemory(user_id="demo_user")
        )
        == "无长期偏好"
    )


def test_write_scenes_prompt_uses_scene_memory(monkeypatch):
    """
    write_scenes Prompt应包含七类Scene Memory字段和当前用户要求。
    """
    fake_llm = FakeStructuredLLM(
        SceneList(scenes=[_scene()])
    )
    monkeypatch.setattr(
        nodes_module,
        "write_scenes_llm",
        fake_llm,
    )

    nodes_module.write_scenes(
        _base_state(user_memory=_scene_memory())
    )

    prompt = fake_llm.prompts[-1]
    memory_section = _prompt_section(
        prompt,
        "【Scene阶段可参考的长期Memory】",
        "Memory使用原则：",
    )

    assert "preferred_genres" in memory_section
    assert "style_preferences" in memory_section
    assert "disliked_elements" in memory_section
    assert "preferred_duration_sec" in memory_section
    assert "additional_preferences" in memory_section
    assert "story_preferences" in memory_section
    assert "scene_preferences" in memory_section
    assert "故事结尾保持克制开放" in memory_section
    assert "分场动作保持可拍摄" in memory_section
    assert "这次生成一个夸张卡通化的校园故事" in prompt
    assert "当前用户要求优先于长期Memory" in prompt
    assert "story_preferences用于继承故事方向" in prompt
    assert "scene_preferences用于约束分场执行" in prompt


def test_review_scene_prompt_uses_scene_memory(monkeypatch):
    """
    review_scene Prompt应包含Scene Memory，并保留当前要求和历史约束。
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
        _base_state(user_memory=_scene_memory())
    )

    prompt = fake_llm.prompts[-1]
    memory_section = _prompt_section(
        prompt,
        "【Scene阶段可参考的长期Memory】",
        "Memory使用原则：",
    )

    assert "story_preferences" in memory_section
    assert "scene_preferences" in memory_section
    assert "故事结尾保持克制开放" in memory_section
    assert "分场动作保持可拍摄" in memory_section
    assert "这次生成一个夸张卡通化的校园故事" in prompt
    assert "历史分场问题" in prompt
    assert "当前用户要求优先于长期Memory" in prompt


def test_revise_scene_prompt_uses_scene_memory_with_lower_priority(monkeypatch):
    """
    revise_scene Prompt应保留反馈、Review和历史约束，并声明它们优先于Memory。
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
        _base_state(user_memory=_scene_memory())
    )

    prompt = fake_llm.prompts[-1]
    memory_section = _prompt_section(
        prompt,
        "【Scene阶段可参考的长期Memory】",
        "Memory使用原则：",
    )

    assert "story_preferences" in memory_section
    assert "scene_preferences" in memory_section
    assert "故事结尾保持克制开放" in memory_section
    assert "分场动作保持可拍摄" in memory_section
    assert "最新人工反馈" in prompt
    assert "本轮分场问题" in prompt
    assert "本轮分场建议" in prompt
    assert "历史分场问题" in prompt
    assert "当前人工反馈和本轮Review优先于长期Memory" in prompt
    assert "如果长期Memory与当前人工反馈、本轮Review或当前任务冲突" in prompt


def test_write_scenes_prompt_handles_empty_memory(monkeypatch):
    """
    write_scenes遇到空Memory时应写入无长期偏好。
    """
    fake_llm = FakeStructuredLLM(
        SceneList(scenes=[_scene()])
    )
    monkeypatch.setattr(
        nodes_module,
        "write_scenes_llm",
        fake_llm,
    )

    nodes_module.write_scenes(
        _base_state(user_memory=None)
    )

    assert "无长期偏好" in fake_llm.prompts[-1]


def test_story_memory_context_is_unchanged():
    """
    M5不应改变Story Memory Context，Story仍不读取scene_preferences。
    """
    context = format_story_memory_context(
        _scene_memory()
    )

    assert "story_preferences" in context
    assert "故事结尾保持克制开放" in context
    assert "scene_preferences" not in context
    assert "分场动作保持可拍摄" not in context
