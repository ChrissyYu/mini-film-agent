import os

import pytest

from memory.models import UserMemory
from memory.retrieve_memory import retrieve_memory
from memory.store import get_memory_path, load_user_memory, save_user_memory
from memory.update_memory import update_memory
from nodes import analyze_brief


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MEMORY_V1_INTEGRATION") != "1",
    reason="需要真实LLM调用；设置RUN_MEMORY_V1_INTEGRATION=1后运行。",
)


def _reset_memory_file(
    user_id: str,
) -> None:
    """
    删除本测试用户的Memory文件，保证每个用例从干净状态开始。
    """
    memory_path = get_memory_path(user_id)

    if memory_path.exists():
        memory_path.unlink()


def _text_contains_any(
    text: str,
    keywords: list[str],
) -> bool:
    """
    判断文本中是否包含任一关键词，减少LLM措辞差异带来的脆弱性。
    """
    return any(
        keyword in text
        for keyword in keywords
    )


def _brief_text(
    brief,
) -> str:
    """
    将FilmBrief转成可检索文本，方便检查Memory是否影响需求分析。
    """
    return " ".join(
        [
            brief.genre,
            brief.core_theme,
            brief.visual_style,
        ]
    )


def test_memory_isolated_between_different_users():
    """
    不同用户的Memory应分别保存、分别读取，并影响各自的需求分析。
    """
    user_a = "memory_v1_user_A"
    user_b = "memory_v1_user_B"

    _reset_memory_file(user_a)
    _reset_memory_file(user_b)

    first_update = update_memory(
        {
            "user_id": user_a,
            "user_idea": "以后帮我生成短片时，我喜欢现实主义风格。",
            "user_memory": UserMemory(user_id=user_a),
        }
    )
    second_update = update_memory(
        {
            "user_id": user_b,
            "user_idea": "以后帮我生成短片时，我喜欢荒诞喜剧风格。",
            "user_memory": UserMemory(user_id=user_b),
        }
    )

    assert first_update["memory_update_status"] == "saved"
    assert second_update["memory_update_status"] == "saved"

    memory_a = load_user_memory(user_a)
    memory_b = load_user_memory(user_b)

    assert _text_contains_any(
        " ".join(memory_a.style_preferences + memory_a.preferred_genres),
        ["现实主义"],
    )
    assert _text_contains_any(
        " ".join(memory_b.style_preferences + memory_b.preferred_genres),
        ["荒诞", "喜剧"],
    )

    brief_a = analyze_brief(
        {
            "user_id": user_a,
            "user_idea": "再生成一个短片。",
            "user_memory": retrieve_memory({"user_id": user_a})["user_memory"],
        }
    )["film_brief"]
    brief_b = analyze_brief(
        {
            "user_id": user_b,
            "user_idea": "再生成一个短片。",
            "user_memory": retrieve_memory({"user_id": user_b})["user_memory"],
        }
    )["film_brief"]

    brief_a_text = _brief_text(brief_a)
    brief_b_text = _brief_text(brief_b)

    assert _text_contains_any(brief_a_text, ["现实", "纪实", "真实", "生活化"])
    assert not _text_contains_any(brief_a_text, ["荒诞喜剧"])
    assert _text_contains_any(brief_b_text, ["荒诞", "喜剧"])


def test_current_request_overrides_memory():
    """
    当前请求与长期Memory冲突时，需求分析应优先服从当前请求。
    """
    user_id = "memory_v1_override_user"

    _reset_memory_file(user_id)

    save_user_memory(
        UserMemory(
            user_id=user_id,
            style_preferences=[
                "现实主义",
                "克制表达",
            ],
        )
    )

    brief = analyze_brief(
        {
            "user_id": user_id,
            "user_idea": "这次生成一个夸张、卡通化的荒诞喜剧。",
            "user_memory": retrieve_memory({"user_id": user_id})["user_memory"],
        }
    )["film_brief"]

    brief_text = _brief_text(brief)

    assert _text_contains_any(brief_text, ["荒诞", "喜剧", "卡通", "夸张"])
    assert not (
        "现实主义" in brief.genre
        and "克制" in brief.visual_style
    )


def test_one_time_request_does_not_write_memory():
    """
    一次性任务要求不应写入长期Memory。
    """
    user_id = "memory_v1_one_time_user"

    _reset_memory_file(user_id)

    result = update_memory(
        {
            "user_id": user_id,
            "user_idea": "这次生成一个60秒校园悬疑故事。",
            "user_memory": UserMemory(user_id=user_id),
        }
    )

    assert result["memory_update_status"] == "skipped"
    assert result["memory_update"].should_update is False
    assert result["memory_update"].preferred_genres_to_add == []
    assert result["memory_update"].style_preferences_to_add == []
    assert result["memory_update"].preferred_duration_sec is None
    assert result["user_memory"].preferred_duration_sec is None
    assert not _text_contains_any(
        " ".join(
            result["user_memory"].preferred_genres
            + result["user_memory"].style_preferences
            + result["user_memory"].additional_preferences
        ),
        ["悬疑", "校园", "60秒", "60"],
    )
    assert not get_memory_path(user_id).exists()
