"""
项目统一Prompt Registry与Renderer入口。
"""

from prompts.models import PromptSpec, RenderedPrompt
from prompts.registry import get_prompt, prompt_registry
from prompts.renderer import render_prompt

__all__ = [
    "PromptSpec",
    "RenderedPrompt",
    "get_prompt",
    "prompt_registry",
    "render_prompt",
]
