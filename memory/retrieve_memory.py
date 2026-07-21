from typing import Any

from memory.store import load_user_memory
from state import FilmState


def retrieve_memory(
    state: FilmState,
) -> dict[str, Any]:
    """
    根据user_id读取用户长期记忆，
    并写入当前Graph State。
    """
    user_id = state.get("user_id")

    if not user_id:
        raise ValueError(
            "retrieve_memory需要有效的user_id。"
        )

    user_memory = load_user_memory(
        user_id=user_id,
    )

    return {
        "user_memory": user_memory,
        "current_stage": "memory_retrieved",
    }