import json
from typing import Any

from memory.models import UserMemory


def format_story_memory_context(
    user_memory: UserMemory | dict[str, Any] | None,
) -> str:
    """
    统一格式化Story阶段可参考的长期Memory。

    Story节点只读取全局偏好和story_preferences，
    不读取scene_preferences，避免把分场动作层面的偏好提前混入故事大纲。
    """
    if user_memory is None:
        return "无长期偏好"

    if isinstance(user_memory, UserMemory):
        memory_data = user_memory.model_dump(
            mode="json",
        )
    else:
        memory_data = user_memory

    story_memory = {
        "preferred_genres": memory_data.get("preferred_genres", []),
        "style_preferences": memory_data.get("style_preferences", []),
        "disliked_elements": memory_data.get("disliked_elements", []),
        "preferred_duration_sec": memory_data.get("preferred_duration_sec"),
        "additional_preferences": memory_data.get("additional_preferences", []),
        "story_preferences": memory_data.get("story_preferences", []),
    }

    has_memory = any(
        value
        for value in story_memory.values()
    )

    if not has_memory:
        return "无长期偏好"

    return json.dumps(
        story_memory,
        ensure_ascii=False,
        indent=2,
    )


def format_scene_memory_context(
    user_memory: UserMemory | dict[str, Any] | None,
) -> str:
    """
    统一格式化Scene阶段可参考的长期Memory。

    Scene阶段同时读取story_preferences和scene_preferences：
    前者用于继承稳定的故事方向，后者用于约束分场、动作和执行表达。
    """
    if user_memory is None:
        return "无长期偏好"

    if isinstance(user_memory, UserMemory):
        memory_data = user_memory.model_dump(
            mode="json",
        )
    else:
        memory_data = user_memory

    scene_memory = {
        "preferred_genres": memory_data.get("preferred_genres", []),
        "style_preferences": memory_data.get("style_preferences", []),
        "disliked_elements": memory_data.get("disliked_elements", []),
        "preferred_duration_sec": memory_data.get("preferred_duration_sec"),
        "additional_preferences": memory_data.get("additional_preferences", []),
        "story_preferences": memory_data.get("story_preferences", []),
        "scene_preferences": memory_data.get("scene_preferences", []),
    }

    has_memory = any(
        value
        for value in scene_memory.values()
    )

    if not has_memory:
        return "无长期偏好"

    return json.dumps(
        scene_memory,
        ensure_ascii=False,
        indent=2,
    )
