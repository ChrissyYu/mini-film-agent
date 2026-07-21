import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.models import UserMemory
from memory.retrieve_memory import retrieve_memory
from memory import store as memory_store
from memory.store import load_user_memory, validate_user_id


@pytest.fixture(autouse=True)
def isolated_memory_dir(
    tmp_path,
    monkeypatch,
):
    """
    每个测试都使用独立临时Memory目录。

    测试不能依赖本地运行时生成的data/user_memory文件；
    否则换一台机器或清理本地数据后，测试结果就会不稳定。
    """
    monkeypatch.setattr(
        memory_store,
        "MEMORY_DIR",
        tmp_path / "user_memory",
    )
    memory_store.save_user_memory(
        UserMemory(
            user_id="demo_user_001",
            preferred_genres=["青春片"],
        )
    )


def test_load_existing_demo_memory():
    """
    已有Memory文件时，应读取到用户长期偏好。
    """

    user_memory = load_user_memory("demo_user_001")

    assert user_memory.user_id == "demo_user_001"
    assert "青春片" in user_memory.preferred_genres


def test_load_missing_memory_returns_empty_memory():
    """
    用户尚无Memory文件时，应返回一份空的UserMemory。
    """

    user_memory = load_user_memory("missing_demo_user")

    assert user_memory == UserMemory(user_id="missing_demo_user")


def test_invalid_user_id_is_rejected():
    """
    非法user_id不能被用于构造Memory路径。
    """

    with pytest.raises(ValueError):
        validate_user_id("../bad-user")


def test_retrieve_memory_node_returns_memory_state():
    """
    retrieve_memory节点应把读取到的Memory写回Graph State。
    """

    result = retrieve_memory(
        {
            "user_id": "demo_user_001",
        }
    )

    assert result["current_stage"] == "memory_retrieved"
    assert result["user_memory"].user_id == "demo_user_001"
    assert "青春片" in result["user_memory"].preferred_genres
