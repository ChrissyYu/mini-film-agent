import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nodes
from prompts.models import PromptSpec
from prompts.registry import (
    DuplicatePromptError,
    PromptRegistry,
    UnknownPromptNameError,
    UnknownPromptVersionError,
    get_prompt,
)
from prompts.renderer import PromptRenderError, PromptRenderer, render_prompt
from schemas import Character, CharacterList, FilmBrief


PRODUCTION_PROMPT_VARIABLES = {
    "generation.analyze_brief": (
        "user_idea",
        "memory_text",
    ),
    "generation.design_characters": (
        "user_idea",
        "genre",
        "core_theme",
        "visual_style",
    ),
    "generation.plan_story": (
        "user_idea",
        "genre",
        "core_theme",
        "visual_style",
        "target_duration_sec",
        "characters_json",
        "story_memory_text",
    ),
    "generation.write_scenes": (
        "target_duration_sec",
        "user_idea",
        "story_outline_json",
        "characters_json",
        "scene_memory_text",
        "recommended_scene_count",
    ),
    "review.story": (
        "film_brief",
        "user_idea",
        "characters",
        "story_outline",
        "story_memory_text",
        "history_issues_text",
    ),
    "review.scene": (
        "film_brief",
        "user_idea",
        "characters",
        "story_outline",
        "scenes",
        "scene_memory_text",
        "history_issues_text",
    ),
    "revision.story": (
        "film_brief",
        "characters",
        "current_story_outline",
        "story_memory_text",
        "human_feedback_text",
        "story_review_issues",
        "story_review_suggestions",
        "active_history_issues_text",
        "resolved_history_reminders_text",
    ),
    "revision.scene": (
        "film_brief",
        "characters",
        "story_outline",
        "current_scene_json",
        "scene_memory_text",
        "human_feedback_text",
        "scene_review_issues",
        "scene_review_suggestions",
        "active_history_issues_text",
        "resolved_history_reminders_text",
    ),
    "memory.candidate_extraction": (
        "user_idea",
        "current_memory_text",
        "recent_human_feedback_text",
    ),
    "memory.conservative_verifier": (
        "candidate_text",
    ),
}


def _prompt_spec(
    *,
    name: str = "test.prompt",
    version: str = "v1",
    template: str = "内容：{{value}}",
    required_variables: tuple[str, ...] = ("value",),
) -> PromptSpec:
    """
    构造独立PromptSpec，避免测试污染项目默认Registry。
    """
    return PromptSpec(
        name=name,
        version=version,
        template=template,
        required_variables=required_variables,
    )


def test_prompt_spec_is_immutable():
    """
    PromptSpec注册后不可被原地修改，避免版本内容悄然漂移。
    """
    spec = _prompt_spec()

    with pytest.raises(ValidationError):
        spec.version = "v2"


def test_registry_gets_explicit_and_default_v1():
    registry = PromptRegistry()
    spec = _prompt_spec()
    registry.register(spec)

    assert registry.get("test.prompt") is spec
    assert registry.get("test.prompt", "v1") is spec


def test_registry_rejects_duplicate_and_unknown_prompt():
    registry = PromptRegistry()
    registry.register(
        _prompt_spec()
    )

    with pytest.raises(
        DuplicatePromptError,
        match="test.prompt:v1",
    ):
        registry.register(
            _prompt_spec()
        )

    with pytest.raises(
        UnknownPromptNameError,
        match="unknown.prompt",
    ):
        registry.get(
            "unknown.prompt"
        )

    with pytest.raises(
        UnknownPromptVersionError,
        match="v2",
    ):
        registry.get(
            "test.prompt",
            "v2",
        )


def test_renderer_validates_missing_and_extra_variables():
    registry = PromptRegistry()
    registry.register(
        _prompt_spec()
    )
    renderer = PromptRenderer(
        registry
    )

    with pytest.raises(
        PromptRenderError,
        match="缺少变量.*value",
    ):
        renderer.render(
            "test.prompt"
        )

    with pytest.raises(
        PromptRenderError,
        match="多余变量.*extra",
    ):
        renderer.render(
            "test.prompt",
            value="合法值",
            extra="多余值",
        )


def test_renderer_interpolates_only_declared_placeholders_and_keeps_json():
    registry = PromptRegistry()
    registry.register(
        _prompt_spec(
            template=(
                'JSON示例：{"kind": "character"}\n'
                "用户值：{{value}}"
            ),
        )
    )
    renderer = PromptRenderer(
        registry
    )

    rendered = renderer.render(
        "test.prompt",
        value="SENTINEL_VALUE",
    )

    assert rendered.text == (
        'JSON示例：{"kind": "character"}\n'
        "用户值：SENTINEL_VALUE"
    )
    assert "{{value}}" not in rendered.text
    assert rendered.name == "test.prompt"
    assert rendered.version == "v1"
    assert rendered.chars == len(
        rendered.text
    )


def test_renderer_preserves_double_braces_from_dynamic_user_text():
    """
    动态用户原文中的双大括号不是模板变量，必须保持原样。
    """
    registry = PromptRegistry()
    registry.register(
        _prompt_spec()
    )
    renderer = PromptRenderer(
        registry
    )

    rendered = renderer.render(
        "test.prompt",
        value="用户原文 {{literal_text}}",
    )

    assert (
        "用户原文 {{literal_text}}"
        in rendered.text
    )


def test_renderer_rejects_required_variable_missing_from_template():
    registry = PromptRegistry()
    registry.register(
        _prompt_spec(
            template="内容：{{value}}",
            required_variables=(
                "value",
                "missing_from_template",
            ),
        )
    )
    renderer = PromptRenderer(
        registry
    )

    with pytest.raises(
        PromptRenderError,
        match="required_variables未出现在模板中",
    ):
        renderer.render(
            "test.prompt",
            value="合法值",
            missing_from_template="不会被静默忽略",
        )


def test_design_characters_prompt_is_registered_and_rendered():
    spec = get_prompt(
        "generation.design_characters"
    )
    rendered = render_prompt(
        "generation.design_characters",
        user_idea="SENTINEL_IDEA",
        genre="SENTINEL_GENRE",
        core_theme="SENTINEL_THEME",
        visual_style="SENTINEL_STYLE",
    )

    assert spec.version == "v1"
    assert spec.required_variables == (
        "user_idea",
        "genre",
        "core_theme",
        "visual_style",
    )
    assert "SENTINEL_IDEA" in rendered.text
    assert "SENTINEL_GENRE" in rendered.text
    assert "SENTINEL_THEME" in rendered.text
    assert "SENTINEL_STYLE" in rendered.text
    assert "{{" not in rendered.text
    assert rendered.chars == len(
        rendered.text
    )


@pytest.mark.parametrize(
    (
        "prompt_name",
        "required_variables",
    ),
    PRODUCTION_PROMPT_VARIABLES.items(),
)
def test_all_production_prompts_are_registered_and_render_sentinels(
    prompt_name,
    required_variables,
):
    """
    10个生产Prompt都必须具有固定v1契约，且所有动态变量能完成替换。
    """
    spec = get_prompt(
        prompt_name
    )
    variables = {
        variable_name: (
            f"SENTINEL_{variable_name.upper()}"
        )
        for variable_name in required_variables
    }

    rendered = render_prompt(
        prompt_name,
        **variables,
    )

    assert spec.version == "v1"
    assert (
        spec.required_variables
        == required_variables
    )
    assert rendered.name == prompt_name
    assert rendered.version == "v1"
    assert rendered.chars == len(
        rendered.text
    )
    assert "{{" not in rendered.text

    for sentinel in variables.values():
        assert sentinel in rendered.text


def test_design_characters_uses_rendered_prompt_with_structured_llm(
    monkeypatch,
):
    """
    节点仍调用原有structured LLM，只把Prompt来源切换到Renderer。
    """

    class FakeCharacterLLM:
        def __init__(self):
            self.prompts = []

        def invoke(
            self,
            prompt: str,
        ) -> CharacterList:
            self.prompts.append(prompt)
            return CharacterList(
                characters=[
                    Character(
                        name="测试角色",
                        role="主角",
                        appearance="简洁衣着",
                        personality=["克制"],
                        motivation="完成目标",
                        continuity_constraints=[
                            "服装保持一致",
                            "发型保持一致",
                        ],
                    )
                ]
            )

    fake_llm = FakeCharacterLLM()
    monkeypatch.setattr(
        nodes,
        "character_llm",
        fake_llm,
    )

    result = nodes.design_characters(
        {
            "user_idea": "SENTINEL_USER_IDEA",
            "film_brief": FilmBrief(
                target_duration_sec=30,
                genre="SENTINEL_GENRE",
                core_theme="SENTINEL_CORE_THEME",
                visual_style="SENTINEL_VISUAL_STYLE",
                recommended_scene_count=4,
            ),
        }
    )

    prompt = fake_llm.prompts[-1]

    assert "SENTINEL_USER_IDEA" in prompt
    assert "SENTINEL_GENRE" in prompt
    assert "SENTINEL_CORE_THEME" in prompt
    assert "SENTINEL_VISUAL_STYLE" in prompt
    assert "设计必要的角色列表" in prompt
    assert "{{" not in prompt
    assert result["characters"][0].name == "测试角色"
    assert result["current_stage"] == "characters_completed"
