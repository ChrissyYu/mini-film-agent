from types import MappingProxyType
from typing import Mapping

from llm_profiles.models import LLMProfile


class LLMProfileRegistryError(Exception):
    """
    LLM Profile注册和查找错误的基类。
    """


class DuplicateLLMProfileError(
    LLMProfileRegistryError
):
    """
    同名Profile被重复注册。
    """


class UnknownLLMProfileError(
    LLMProfileRegistryError
):
    """
    Registry中不存在指定Profile。
    """


class LLMProfileRegistry:
    """
    保存不可变LLMProfile，并提供显式查找错误。
    """

    def __init__(self) -> None:
        self._profiles: dict[
            str,
            LLMProfile,
        ] = {}

    def register(
        self,
        profile: LLMProfile,
    ) -> None:
        """
        注册Profile；已存在的名称不允许覆盖。
        """
        if profile.name in self._profiles:
            raise DuplicateLLMProfileError(
                f"LLM Profile已重复注册：{profile.name}"
            )

        self._profiles[profile.name] = profile

    def get(
        self,
        name: str,
    ) -> LLMProfile:
        """
        按名称获取Profile。
        """
        if name not in self._profiles:
            raise UnknownLLMProfileError(
                f"未知LLM Profile：{name}"
            )

        return self._profiles[name]

    def all(
        self,
    ) -> Mapping[str, LLMProfile]:
        """
        返回只读Profile映射，供校验和测试使用。
        """
        return MappingProxyType(
            self._profiles.copy()
        )


llm_profile_registry = LLMProfileRegistry()

_CURRENT_PROVIDER = (
    "dashscope_openai_compatible"
)
_CURRENT_MODEL = "qwen-plus"
_CURRENT_TEMPERATURE = 0
_CURRENT_BASE_URL = (
    "https://dashscope.aliyuncs.com/"
    "compatible-mode/v1"
)
_CURRENT_API_KEY_ENV = "DASHSCOPE_API_KEY"


for profile_name in (
    "fast",
    "balanced",
    "strong",
    "critical",
):
    # 第一版四个Profile保持完全相同的现有模型参数，只建立治理边界。
    llm_profile_registry.register(
        LLMProfile(
            name=profile_name,
            provider=_CURRENT_PROVIDER,
            model=_CURRENT_MODEL,
            temperature=_CURRENT_TEMPERATURE,
            base_url=_CURRENT_BASE_URL,
            api_key_env=_CURRENT_API_KEY_ENV,
        )
    )


def get_llm_profile(
    name: str,
) -> LLMProfile:
    """
    从项目默认Registry获取LLMProfile。
    """
    return llm_profile_registry.get(
        name
    )
