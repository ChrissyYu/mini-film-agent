import json

from memory.models import MemoryUpdate, UserMemory    # MemoryUpdate是提取结果，UserMemory是当前已有记忆
from nodes import llm    # 复用项目统一配置好的LLM客户端


# 使用项目现有LLM客户端，并通过structured output约束输出结构。
memory_update_llm = llm.with_structured_output(
    MemoryUpdate    # 约束LLM必须返回长期偏好增量结构
)


def _format_current_memory(
    current_memory: UserMemory,
) -> str:
    """
    按字段展示当前Memory，便于LLM按对应字段做语义比较。
    """
    memory_data = {
        "preferred_genres": current_memory.preferred_genres,
        "style_preferences": current_memory.style_preferences,
        "disliked_elements": current_memory.disliked_elements,
        "preferred_duration_sec": current_memory.preferred_duration_sec,
        "additional_preferences": current_memory.additional_preferences,
        "story_preferences": current_memory.story_preferences,
        "scene_preferences": current_memory.scene_preferences,
    }

    return json.dumps(
        memory_data,
        ensure_ascii=False,
        indent=2,
    )


def _format_recent_human_feedback(
    human_feedback_history: list[dict] | None,
) -> str:
    """
    在代码中先过滤和截断人工反馈，避免空反馈或过长历史干扰Memory提取。
    """
    human_feedback_items = []

    for feedback_event in human_feedback_history or []:
        raw_feedback = feedback_event.get(
            "feedback",
        )

        if raw_feedback is None:
            continue

        feedback = str(raw_feedback).strip()

        if not feedback:
            continue

        human_feedback_items.append(
            {
                "scope": feedback_event.get("scope"),    # story或scene，用于约束写入对应阶段偏好
                "decision": feedback_event.get("decision"),    # approve或revise，保留人工选择语境
                "feedback": feedback,    # 只保留用户主动写下的反馈，不带机器审核或生成结果
            }
        )

    # 只取最近8条用户反馈，避免长对话把Prompt撑大，也降低一次性细节被反复放大的风险。
    recent_human_feedback_items = human_feedback_items[-8:]

    return (
        json.dumps(
            recent_human_feedback_items,
            ensure_ascii=False,
            indent=2,
        )
        if recent_human_feedback_items
        else "无人工反馈"
    )


def _normalize_memory_update(
    update: MemoryUpdate,
    current_memory: UserMemory,
) -> MemoryUpdate:
    """
    对LLM结构化输出做一致性校正，防止should_update与增量字段互相矛盾。
    """
    if not update.should_update:
        return MemoryUpdate(
            should_update=False,
        )

    def clean_items(
        items: list[str],
    ) -> list[str]:
        cleaned_items = []
        seen_items = set()

        for item in items:
            cleaned_item = str(item).strip()

            if not cleaned_item:
                continue

            if cleaned_item in seen_items:
                continue

            cleaned_items.append(cleaned_item)
            seen_items.add(cleaned_item)

        return cleaned_items

    story_preferences_to_add = clean_items(
        update.story_preferences_to_add
    )
    scene_preferences_to_add = clean_items(
        update.scene_preferences_to_add
    )

    # 作用域优先：明确属于Story/Scene的偏好不应同时写入全局字段。
    # 这里处理确定性的同文本重复；近义重复由提取器根据Prompt在输出前判断。
    scoped_preference_texts = set(
        story_preferences_to_add
        + scene_preferences_to_add
        + current_memory.story_preferences
        + current_memory.scene_preferences
    )

    def clean_global_items(
        items: list[str],
    ) -> list[str]:
        return [
            item
            for item in clean_items(items)
            if item not in scoped_preference_texts
        ]

    normalized_update = MemoryUpdate(
        should_update=True,
        preferred_genres_to_add=clean_items(
            update.preferred_genres_to_add
        ),
        style_preferences_to_add=clean_global_items(
            update.style_preferences_to_add
        ),
        disliked_elements_to_add=clean_global_items(
            update.disliked_elements_to_add
        ),
        preferred_duration_sec=update.preferred_duration_sec,
        additional_preferences_to_add=clean_global_items(
            update.additional_preferences_to_add
        ),
        story_preferences_to_add=story_preferences_to_add,
        scene_preferences_to_add=scene_preferences_to_add,
    )

    has_increment = any(
        [
            normalized_update.preferred_genres_to_add,
            normalized_update.style_preferences_to_add,
            normalized_update.disliked_elements_to_add,
            normalized_update.additional_preferences_to_add,
            normalized_update.story_preferences_to_add,
            normalized_update.scene_preferences_to_add,
            normalized_update.preferred_duration_sec is not None,
        ]
    )

    if not has_increment:
        return MemoryUpdate(
            should_update=False,
        )

    return normalized_update


def extract_memory_update(
    user_idea: str,    # 用户本次输入的原始需求
    current_memory: UserMemory,    # 当前已经读取到的完整长期记忆
    human_feedback_history: list[dict] | None = None,    # 本次执行中累计的人工反馈历史
) -> MemoryUpdate:
    """
    从用户本次输入中提取长期偏好增量。

    参数：
    - user_idea：用户本次输入的原始需求；
    - current_memory：当前已经保存的用户长期记忆，用于判断哪些偏好已存在；
    - human_feedback_history：本次执行中的人工审核反馈，只读取用户写下的反馈文本。

    返回：
    - MemoryUpdate，只包含用户明确表达的长期偏好增量。
    """
    current_memory_text = _format_current_memory(
        current_memory
    )    # 结构化展示各字段，便于提取器按字段语义去重
    recent_human_feedback_text = _format_recent_human_feedback(
        human_feedback_history
    )    # 人工反馈先由代码过滤、截断，再交给LLM判断长期价值

    # 语义去重由提取器在输出增量前完成；merge只负责后续确定性的字符串去重和空值过滤。
    # 这样可以避免“克制开放式结尾”和“含蓄开放结局”这类近义偏好反复写入Memory。
    # 提取长期偏好增量的边界提示词
    prompt = f"""
你负责从用户文字中提取可跨不同作品复用的长期影视创作偏好增量。

【用户本次输入】
{user_idea}

【当前长期 Memory】
{current_memory_text}

【本次人工审核反馈】
{recent_human_feedback_text}

仅输出符合 MemoryUpdate 结构的数据。

提取边界：
- 只依据用户输入和人工反馈，不依据机器 Review、模型生成内容或 final_output。
- 只保存用户明确表达或明显可长期复用的稳定偏好。
- 当前人物、地点、场次、具体剧情和单次时长等任务专属要求不保存。
- 无法确定是否具有长期价值时，不更新。
- 只返回新增内容，不覆盖已有 Memory，不复制整段用户原话。

作用域与字段：
- 故事、大纲、叙事和剧情结构偏好，只进入 story_preferences_to_add。
- 分场、场景、镜头和场景动作偏好，只进入 scene_preferences_to_add。
- preferred_genres_to_add：长期喜欢的影片类型。
- style_preferences_to_add：适用于整体创作的长期风格偏好。
- disliked_elements_to_add：适用于整体创作的长期排斥元素。
- preferred_duration_sec：仅提取长期时长偏好。
- additional_preferences_to_add：其他长期全局偏好。
- 只有用户明确表示适用于整体创作时，Scoped 偏好才可提升为全局偏好。

语义去重：
- 新候选必须与对应字段的已有 Memory 比较。
- 含义相同或高度重合时，即使措辞不同，也不要重复输出。
- Scoped 候选与全局候选语义重合时，只保留 Scoped 候选。
- 已有 Scoped 近义偏好时，不得改写到全局字段绕过去重。
- Story 与 Scene 分别管理，不跨作用域互相删除。
- 新偏好应规范化为简短、稳定、可复用的表达。

没有真实新增内容时，返回 should_update=false。
"""

    raw_update = memory_update_llm.invoke(prompt)    # 返回结构化MemoryUpdate，不保存也不合并

    return _normalize_memory_update(
        raw_update,
        current_memory,
    )    # 校正LLM输出，避免无效增量触发后续保存
