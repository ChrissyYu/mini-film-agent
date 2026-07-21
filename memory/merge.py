from memory.models import MemoryUpdate, UserMemory    # MemoryUpdate是增量，UserMemory是合并后的完整记忆


def _append_unique_items(
    current_items: list[str],    # 已有列表内容
    new_items: list[str],    # 本次需要追加的新内容
) -> list[str]:
    """
    将新的字符串列表追加到旧列表后面。

    规则：
    - 保留旧列表的原有顺序；
    - 新值只在旧列表中不存在时追加；
    - 忽略空字符串和只有空格的字符串。
    """
    merged_items: list[str] = []    # 保存去重后的合并结果
    seen_items: set[str] = set()    # 记录已出现内容，用于保持顺序去重

    for item in current_items + new_items:
        normalized_item = item.strip()    # 去掉首尾空格，避免空白内容进入Memory

        if not normalized_item:
            continue

        if normalized_item in seen_items:
            continue

        merged_items.append(normalized_item)
        seen_items.add(normalized_item)

    return merged_items


def merge_user_memory(
    current_memory: UserMemory,    # 当前完整长期记忆
    update: MemoryUpdate,    # 本次提取出的长期偏好增量
) -> UserMemory:
    """
    将一次MemoryUpdate合并进已有UserMemory。

    参数：
    - current_memory：当前已经保存的用户长期记忆；
    - update：从本次用户请求中提取出的长期偏好增量。

    返回：
    - 一份新的UserMemory，不修改传入的current_memory。
    """
    if not update.should_update:
        return current_memory.model_copy(deep=True)    # 无更新时返回等价的新对象，不改原对象

    preferred_duration_sec = current_memory.preferred_duration_sec    # 默认保留旧的长期时长偏好

    if update.preferred_duration_sec is not None:
        preferred_duration_sec = update.preferred_duration_sec    # 只有增量里有长期时长偏好时才覆盖

    return UserMemory(
        user_id=current_memory.user_id,    # 用户身份保持不变
        preferred_genres=_append_unique_items(
            current_memory.preferred_genres,
            update.preferred_genres_to_add,
        ),    # 合并喜欢的影片类型
        style_preferences=_append_unique_items(
            current_memory.style_preferences,
            update.style_preferences_to_add,
        ),    # 合并风格偏好
        disliked_elements=_append_unique_items(
            current_memory.disliked_elements,
            update.disliked_elements_to_add,
        ),    # 合并不喜欢的元素
        preferred_duration_sec=preferred_duration_sec,    # 合并后的长期时长偏好
        additional_preferences=_append_unique_items(
            current_memory.additional_preferences,
            update.additional_preferences_to_add,
        ),    # 合并其他长期偏好
        story_preferences=_append_unique_items(
            current_memory.story_preferences,
            update.story_preferences_to_add,
        ),    # 合并故事大纲阶段的长期偏好
        scene_preferences=_append_unique_items(
            current_memory.scene_preferences,
            update.scene_preferences_to_add,
        ),    # 合并分场设计阶段的长期偏好
    )
