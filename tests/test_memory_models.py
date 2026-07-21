import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.models import MemoryUpdate, UserMemory


def test_user_memory_contains_stage_preference_fields():
    """
    UserMemory应包含故事和分场阶段的偏好字段。
    """

    assert "story_preferences" in UserMemory.model_fields
    assert "scene_preferences" in UserMemory.model_fields


def test_memory_update_contains_stage_preference_fields():
    """
    MemoryUpdate应包含故事和分场阶段的偏好增量字段。
    """

    assert "story_preferences_to_add" in MemoryUpdate.model_fields
    assert "scene_preferences_to_add" in MemoryUpdate.model_fields


def test_old_memory_json_without_new_fields_can_parse():
    """
    旧格式Memory JSON没有新字段时，应使用空列表默认值保持兼容。
    """

    user_memory = UserMemory.model_validate(
        {
            "user_id": "demo_user",
            "style_preferences": [
                "现实主义",
            ],
        }
    )

    assert user_memory.user_id == "demo_user"
    assert user_memory.style_preferences == [
        "现实主义",
    ]
    assert user_memory.story_preferences == []
    assert user_memory.scene_preferences == []


def test_new_stage_preference_fields_dump_and_parse():
    """
    新增字段应能正常传入、导出，并再次解析。
    """

    user_memory = UserMemory(
        user_id="demo_user",
        story_preferences=[
            "开放式结尾",
        ],
        scene_preferences=[
            "少量对白",
        ],
    )

    dumped_memory = user_memory.model_dump(
        mode="json",
    )
    reparsed_memory = UserMemory.model_validate(
        dumped_memory
    )

    assert dumped_memory["story_preferences"] == [
        "开放式结尾",
    ]
    assert dumped_memory["scene_preferences"] == [
        "少量对白",
    ]
    assert reparsed_memory == user_memory


def test_unknown_fields_are_forbidden():
    """
    extra='forbid'应继续生效，未知字段必须触发校验错误。
    """

    with pytest.raises(ValidationError):
        UserMemory.model_validate(
            {
                "user_id": "demo_user",
                "unknown_field": "不允许的字段",
            }
        )

    with pytest.raises(ValidationError):
        MemoryUpdate.model_validate(
            {
                "should_update": True,
                "unknown_field": "不允许的字段",
            }
        )
