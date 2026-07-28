import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_profiles.factory as factory_module
from llm_profiles.bindings import (
    UnknownPromptProfileBindingError,
    get_prompt_profile_name,
    prompt_profile_bindings,
)
from llm_profiles.registry import (
    UnknownLLMProfileError,
    get_llm_profile,
    llm_profile_registry,
)
from prompts.models import PromptSpec
from prompts.registry import get_prompt
from schemas import FilmBrief


EXPECTED_BINDINGS = {
    "generation.analyze_brief": "fast",
    "generation.design_characters": "balanced",
    "generation.plan_story": "strong",
    "generation.write_scenes": "strong",
    "review.story": "strong",
    "review.scene": "strong",
    "revision.story": "strong",
    "revision.scene": "strong",
    "memory.candidate_extraction": "balanced",
    "memory.conservative_verifier": "critical",
}


def test_prompt_spec_does_not_contain_model_configuration():
    """
    Prompt版本与模型绑定保持独立，PromptSpec不保存模型字段。
    """
    assert set(
        PromptSpec.model_fields
    ) == {
        "name",
        "version",
        "template",
        "required_variables",
    }


def test_four_llm_profiles_keep_current_model_configuration():
    profiles = (
        llm_profile_registry.all()
    )

    assert set(profiles) == {
        "fast",
        "balanced",
        "strong",
        "critical",
    }

    for profile_name, profile in profiles.items():
        assert profile.name == profile_name
        assert (
            profile.provider
            == "dashscope_openai_compatible"
        )
        assert profile.model == "qwen-plus"
        assert profile.temperature == 0
        assert profile.base_url == (
            "https://dashscope.aliyuncs.com/"
            "compatible-mode/v1"
        )
        assert (
            profile.api_key_env
            == "DASHSCOPE_API_KEY"
        )


def test_ten_prompts_are_bound_once_to_valid_profiles():
    bindings = (
        prompt_profile_bindings.all()
    )
    profile_names = set(
        llm_profile_registry.all()
    )

    assert dict(bindings) == EXPECTED_BINDINGS
    assert len(bindings) == 10

    for prompt_name, profile_name in bindings.items():
        assert (
            get_prompt_profile_name(
                prompt_name
            )
            == profile_name
        )
        assert profile_name in profile_names
        assert (
            get_prompt(prompt_name).name
            == prompt_name
        )


def test_unknown_profile_and_unbound_prompt_raise_clear_errors():
    with pytest.raises(
        UnknownLLMProfileError,
        match="unknown_profile",
    ):
        get_llm_profile(
            "unknown_profile"
        )

    with pytest.raises(
        UnknownPromptProfileBindingError,
        match="unknown.prompt",
    ):
        get_prompt_profile_name(
            "unknown.prompt"
        )


def test_factory_preserves_current_parameters_and_structured_output(
    monkeypatch,
):
    """
    使用Fake ChatOpenAI验证Factory，不创建真实客户端或访问网络。
    """

    class FakeChatOpenAI:
        instances = []

        def __init__(
            self,
            **kwargs,
        ):
            self.kwargs = kwargs
            self.structured_schemas = []
            self.__class__.instances.append(
                self
            )

        def with_structured_output(
            self,
            output_schema,
        ):
            self.structured_schemas.append(
                output_schema
            )
            return (
                "structured",
                output_schema,
            )

    factory_module.clear_llm_factory_cache()
    monkeypatch.setattr(
        factory_module,
        "ChatOpenAI",
        FakeChatOpenAI,
    )
    monkeypatch.setenv(
        "DASHSCOPE_API_KEY",
        "safe-test-placeholder",
    )

    try:
        fast_llm = (
            factory_module
            .get_llm_for_prompt(
                "generation.analyze_brief"
            )
        )
        strong_llm = (
            factory_module
            .get_llm_for_prompt(
                "generation.plan_story"
            )
        )
        structured_llm = (
            factory_module
            .create_structured_llm(
                "generation.analyze_brief",
                FilmBrief,
            )
        )

        # 四个Profile当前参数相同，因此继续复用同一个底层模型客户端。
        assert fast_llm is strong_llm
        assert len(
            FakeChatOpenAI.instances
        ) == 1
        assert fast_llm.kwargs == {
            "api_key": "safe-test-placeholder",
            "base_url": (
                "https://dashscope.aliyuncs.com/"
                "compatible-mode/v1"
            ),
            "model": "qwen-plus",
            "temperature": 0,
        }
        assert structured_llm == (
            "structured",
            FilmBrief,
        )
        assert (
            fast_llm.structured_schemas
            == [
                FilmBrief,
            ]
        )
    finally:
        factory_module.clear_llm_factory_cache()
