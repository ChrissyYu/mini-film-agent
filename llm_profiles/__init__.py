"""
统一LLM Profile、Prompt Binding与模型创建入口。
"""

from llm_profiles.bindings import (
    get_prompt_profile_name,
    prompt_profile_bindings,
)
from llm_profiles.factory import (
    create_structured_llm,
    get_llm_for_prompt,
)
from llm_profiles.models import LLMProfile
from llm_profiles.registry import (
    get_llm_profile,
    llm_profile_registry,
)

__all__ = [
    "LLMProfile",
    "create_structured_llm",
    "get_llm_for_prompt",
    "get_llm_profile",
    "get_prompt_profile_name",
    "llm_profile_registry",
    "prompt_profile_bindings",
]
