from typing import Any

from memory.extract_memory import extract_memory_update    # 从用户输入中提取长期偏好增量
from memory.merge import merge_user_memory    # 将增量合并到当前完整Memory
from memory.store import save_user_memory    # 将更新后的完整Memory保存到本地JSON
from state import FilmState    # Graph中流转的状态结构


def update_memory(
    state: FilmState,    # 当前Graph State，需包含user_id、user_idea和user_memory
) -> dict[str, Any]:
    """
    根据用户本次输入更新长期Memory。

    流程严格保持为：
    1. 从本次输入中提取长期偏好增量；
    2. 将增量合并到当前Memory；
    3. 只有确实需要更新时，才保存新的Memory文件。
    """
    try:
        user_id = state["user_id"]    # 当前用户身份，用于保持更新逻辑显式依赖用户
        user_idea = state["user_idea"]    # 用户本次输入，用于提取长期偏好
        current_memory = state["user_memory"]    # 当前已读取到的完整长期记忆
        human_feedback_history = state.get(
            "human_feedback_history",
            [],
        )    # 只把用户人工反馈传给提取器，不混入机器审核或最终生成结果

        update = extract_memory_update(
            user_idea,
            current_memory,
            human_feedback_history,
        )    # 只提取增量，不保存、不覆盖

        merged_memory = merge_user_memory(
            current_memory,
            update,
        )    # 得到合并后的新Memory对象

        has_real_update = (
            update.should_update
            and merged_memory != current_memory
        )    # 只有合并后Memory真的变化，才需要持久化，避免重复写文件

        memory_update_status = "skipped"   # 表示没有真实长期偏好增量，未保存

        if has_real_update:
            save_user_memory(
                merged_memory
            )    # 只有确实提取到长期偏好时才写入本地JSON
            memory_update_status = "saved"   # 表示提取到长期偏好并已保存

        return {
            "user_memory": merged_memory,    # 更新后供后续节点使用的完整Memory
            "memory_update": update,    # 本次提取出的Memory增量
            "memory_update_status": memory_update_status,    # 本次Memory是否保存
            "memory_update_error": None,    # 成功或跳过时没有错误信息
            "current_stage": "memory_updated",    # 标记Memory更新步骤完成
        }

    except Exception as exc:
        return {
            "memory_update_status": "failed",    # Memory更新失败，但不阻断最终影片结果
            "memory_update_error": str(exc),    # 记录失败原因，便于后续排查
            "current_stage": "memory_update_failed",    # 标记Memory更新失败
        }
