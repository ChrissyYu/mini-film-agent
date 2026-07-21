import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import store as memory_store
from memory.merge import merge_user_memory
from memory.models import MemoryUpdate, UserMemory
from memory import update_memory as update_memory_module


def test_merge_when_should_update_false_keeps_content():
    """
    should_update为False时，不应改变已有Memory内容。
    """

    current_memory = UserMemory(
        user_id="demo_user_001",
        preferred_genres=["青春片"],
        style_preferences=["自然克制"],
        disliked_elements=["大量旁白"],
        preferred_duration_sec=60,
        additional_preferences=["校园场景"],
    )

    update = MemoryUpdate(
        should_update=False,
        preferred_genres_to_add=["科幻片"],
        preferred_duration_sec=90,
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory == current_memory
    assert merged_memory is not current_memory


def test_merge_lists_appends_new_items_and_deduplicates():
    """
    列表字段应保留旧值顺序，并把未出现过的新值追加到后面。
    """

    current_memory = UserMemory(
        user_id="demo_user_001",
        preferred_genres=["青春片", "现实主义剧情片"],
        style_preferences=["自然克制"],
        disliked_elements=["大量旁白"],
        additional_preferences=["校园场景"],
    )

    update = MemoryUpdate(
        should_update=True,
        preferred_genres_to_add=["青春片", "悬疑片"],
        style_preferences_to_add=["自然克制", "生活化"],
        disliked_elements_to_add=["大量旁白", "强行反转"],
        additional_preferences_to_add=["校园场景", "群像关系"],
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory.preferred_genres == [
        "青春片",
        "现实主义剧情片",
        "悬疑片",
    ]
    assert merged_memory.style_preferences == [
        "自然克制",
        "生活化",
    ]
    assert merged_memory.disliked_elements == [
        "大量旁白",
        "强行反转",
    ]
    assert merged_memory.additional_preferences == [
        "校园场景",
        "群像关系",
    ]


def test_merge_story_preferences():
    """
    story_preferences_to_add应合并进UserMemory.story_preferences。
    """
    current_memory = UserMemory(
        user_id="demo_user_001",
        story_preferences=["克制开放式结尾"],
    )

    update = MemoryUpdate(
        should_update=True,
        story_preferences_to_add=[
            "克制开放式结尾",
            "冲突保持生活化",
        ],
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory.story_preferences == [
        "克制开放式结尾",
        "冲突保持生活化",
    ]


def test_merge_scene_preferences():
    """
    scene_preferences_to_add应合并进UserMemory.scene_preferences。
    """
    current_memory = UserMemory(
        user_id="demo_user_001",
        scene_preferences=["动作描述可拍摄"],
    )

    update = MemoryUpdate(
        should_update=True,
        scene_preferences_to_add=[
            "动作描述可拍摄",
            "减少解释性对白",
        ],
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory.scene_preferences == [
        "动作描述可拍摄",
        "减少解释性对白",
    ]


def test_merge_ignores_empty_strings():
    """
    空字符串和只有空格的字符串不应写入Memory。
    """

    current_memory = UserMemory(
        user_id="demo_user_001",
        preferred_genres=["青春片"],
    )

    update = MemoryUpdate(
        should_update=True,
        preferred_genres_to_add=["", "   ", "剧情片"],
        style_preferences_to_add=[" ", "生活化"],
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory.preferred_genres == [
        "青春片",
        "剧情片",
    ]
    assert merged_memory.style_preferences == [
        "生活化",
    ]


def test_merge_stage_preferences_ignore_empty_and_duplicate_items():
    """
    阶段偏好也应去除首尾空格、过滤空字符串并保持首次出现顺序。
    """
    current_memory = UserMemory(
        user_id="demo_user_001",
        story_preferences=["克制结尾"],
        scene_preferences=["动作具体"],
    )

    update = MemoryUpdate(
        should_update=True,
        story_preferences_to_add=[
            " ",
            "克制结尾",
            " 生活化冲突 ",
            "生活化冲突",
        ],
        scene_preferences_to_add=[
            "",
            "动作具体",
            " 可拍摄 ",
        ],
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory.story_preferences == [
        "克制结尾",
        "生活化冲突",
    ]
    assert merged_memory.scene_preferences == [
        "动作具体",
        "可拍摄",
    ]


def test_merge_duration_overwrites_old_value():
    """
    update中有新的影片时长时，应覆盖旧的影片时长。
    """

    current_memory = UserMemory(
        user_id="demo_user_001",
        preferred_duration_sec=60,
    )

    update = MemoryUpdate(
        should_update=True,
        preferred_duration_sec=90,
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory.preferred_duration_sec == 90


def test_merge_duration_none_keeps_old_value():
    """
    update中没有新的影片时长时，应保留旧的影片时长。
    """

    current_memory = UserMemory(
        user_id="demo_user_001",
        preferred_duration_sec=60,
    )

    update = MemoryUpdate(
        should_update=True,
        preferred_duration_sec=None,
    )

    merged_memory = merge_user_memory(
        current_memory=current_memory,
        update=update,
    )

    assert merged_memory.preferred_duration_sec == 60


def test_update_memory_skips_when_no_real_increment(monkeypatch):
    """
    提取器误报should_update但合并后无真实变化时，不应重复写文件。
    """
    current_memory = UserMemory(
        user_id="demo_user_001",
        story_preferences=["克制结尾"],
    )

    def fake_extract_memory_update(
        user_idea,
        current_memory_arg,
        human_feedback_history,
    ):
        return MemoryUpdate(
            should_update=True,
            story_preferences_to_add=[
                "克制结尾",
                " ",
            ],
        )

    def fake_save_user_memory(user_memory):
        raise AssertionError("无真实增量时不应保存Memory")

    monkeypatch.setattr(
        update_memory_module,
        "extract_memory_update",
        fake_extract_memory_update,
    )
    monkeypatch.setattr(
        update_memory_module,
        "save_user_memory",
        fake_save_user_memory,
    )

    result = update_memory_module.update_memory(
        {
            "user_id": "demo_user_001",
            "user_idea": "这次生成一个校园故事。",
            "user_memory": current_memory,
        }
    )

    assert result["memory_update_status"] == "skipped"
    assert result["user_memory"] == current_memory


def test_update_memory_saves_when_stage_preference_has_real_increment(monkeypatch):
    """
    有真实阶段偏好增量时，应保存更新后的UserMemory。
    """
    current_memory = UserMemory(
        user_id="demo_user_001",
    )
    saved_memories = []

    def fake_extract_memory_update(
        user_idea,
        current_memory_arg,
        human_feedback_history,
    ):
        return MemoryUpdate(
            should_update=True,
            scene_preferences_to_add=[
                "分场动作更具体",
            ],
        )

    monkeypatch.setattr(
        update_memory_module,
        "extract_memory_update",
        fake_extract_memory_update,
    )
    monkeypatch.setattr(
        update_memory_module,
        "save_user_memory",
        lambda user_memory: saved_memories.append(user_memory),
    )

    result = update_memory_module.update_memory(
        {
            "user_id": "demo_user_001",
            "user_idea": "这次生成一个校园故事。",
            "user_memory": current_memory,
        }
    )

    assert result["memory_update_status"] == "saved"
    assert saved_memories[0].scene_preferences == [
        "分场动作更具体",
    ]


def test_old_memory_json_loads_and_save_backfills_stage_fields(
    tmp_path,
    monkeypatch,
):
    """
    旧格式JSON缺少新字段时仍可读取，保存后会补全story/scene字段。
    """
    monkeypatch.setattr(
        memory_store,
        "MEMORY_DIR",
        tmp_path,
    )
    memory_path = tmp_path / "legacy_user.json"
    memory_path.write_text(
        json.dumps(
            {
                "user_id": "legacy_user",
                "style_preferences": [
                    "现实主义",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    user_memory = memory_store.load_user_memory(
        "legacy_user",
    )

    assert user_memory.story_preferences == []
    assert user_memory.scene_preferences == []

    memory_store.save_user_memory(
        user_memory,
    )

    saved_data = json.loads(
        memory_path.read_text(
            encoding="utf-8",
        )
    )

    assert saved_data["story_preferences"] == []
    assert saved_data["scene_preferences"] == []
