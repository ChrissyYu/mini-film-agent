import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from llm_profiles.bindings import (
    get_prompt_profile_name,
)
from llm_profiles.models import LLMProfile
from llm_profiles.registry import (
    get_llm_profile,
)


def _profile_cache_key(
    profile: LLMProfile,
) -> tuple[str, str, float, str, str]:
    """
    生成实际连接参数键；配置相同的Profile复用同一个底层客户端。
    """
    return (
        profile.provider,
        profile.model,
        profile.temperature,
        profile.base_url,
        profile.api_key_env,
    )


@lru_cache(maxsize=None)
def _create_cached_llm(
    provider: str,
    model: str,
    temperature: float,
    base_url: str,
    api_key_env: str,
) -> ChatOpenAI:
    """
    根据Profile创建并缓存底层ChatOpenAI客户端。
    """
    if provider != (
        "dashscope_openai_compatible"
    ):
        raise ValueError(
            f"不支持的LLM provider：{provider}"
        )

    load_dotenv()
    api_key = os.getenv(
        api_key_env
    )

    if not api_key:
        raise ValueError(
            f"未找到{api_key_env}, 请在.env文件中配置"
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def get_llm_for_prompt(
    prompt_name: str,
) -> ChatOpenAI:
    """
    根据Prompt Binding取得对应Profile并创建底层模型。
    """
    profile_name = (
        get_prompt_profile_name(
            prompt_name
        )
    )
    profile = get_llm_profile(
        profile_name
    )
    return _create_cached_llm(
        *_profile_cache_key(
            profile
        )
    )


def create_structured_llm(
    prompt_name: str,
    output_schema: Any,
) -> Any:
    """
    为指定Prompt绑定原有structured output schema。
    """
    return get_llm_for_prompt(
        prompt_name
    ).with_structured_output(
        output_schema
    )


def clear_llm_factory_cache() -> None:
    """
    清理Factory缓存，供无网络单元测试隔离模型构造使用。
    """
    _create_cached_llm.cache_clear()
