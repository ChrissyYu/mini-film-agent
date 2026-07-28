from types import MappingProxyType
from typing import Mapping

from llm_profiles.registry import (
    LLMProfileRegistry,
    llm_profile_registry,
)


class PromptProfileBindingError(Exception):
    """
    Prompt与Profile绑定错误的基类。
    """


class DuplicatePromptProfileBindingError(
    PromptProfileBindingError
):
    """
    同一Prompt被重复绑定。
    """


class UnknownPromptProfileBindingError(
    PromptProfileBindingError
):
    """
    Prompt尚未绑定任何Profile。
    """


class PromptProfileBindingRegistry:
    """
    保存prompt_name到profile_name的一对一绑定。
    """

    def __init__(
        self,
        profile_registry: LLMProfileRegistry,
    ) -> None:
        self._profile_registry = (
            profile_registry
        )
        self._bindings: dict[str, str] = {}

    def bind(
        self,
        prompt_name: str,
        profile_name: str,
    ) -> None:
        """
        建立绑定；Profile必须存在，Prompt不能重复绑定。
        """
        self._profile_registry.get(
            profile_name
        )

        if prompt_name in self._bindings:
            raise DuplicatePromptProfileBindingError(
                f"Prompt已绑定LLM Profile：{prompt_name}"
            )

        self._bindings[prompt_name] = (
            profile_name
        )

    def get_profile_name(
        self,
        prompt_name: str,
    ) -> str:
        """
        获取Prompt绑定的Profile名称。
        """
        if prompt_name not in self._bindings:
            raise UnknownPromptProfileBindingError(
                f"Prompt未绑定LLM Profile：{prompt_name}"
            )

        return self._bindings[prompt_name]

    def all(
        self,
    ) -> Mapping[str, str]:
        """
        返回只读绑定映射。
        """
        return MappingProxyType(
            self._bindings.copy()
        )


prompt_profile_bindings = (
    PromptProfileBindingRegistry(
        llm_profile_registry
    )
)


for prompt_name, profile_name in {
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
}.items():
    prompt_profile_bindings.bind(
        prompt_name,
        profile_name,
    )


def get_prompt_profile_name(
    prompt_name: str,
) -> str:
    """
    获取项目默认Prompt绑定的Profile名称。
    """
    return (
        prompt_profile_bindings
        .get_profile_name(
            prompt_name
        )
    )
