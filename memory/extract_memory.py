import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from llm_profiles.factory import create_structured_llm
from memory.models import MemoryUpdate, UserMemory    # MemoryUpdate是最终增量，UserMemory是当前已有记忆
from observability.llm_calls import invoke_structured_llm
from prompts.models import RenderedPrompt
from prompts.renderer import render_prompt


MemoryField = Literal[
    "preferred_genres_to_add",
    "style_preferences_to_add",
    "disliked_elements_to_add",
    "preferred_duration_sec",
    "additional_preferences_to_add",
    "story_preferences_to_add",
    "scene_preferences_to_add",
]
MemorySource = Literal[
    "user_idea",
    "human_feedback",
]
MemoryClaimType = Literal[
    "explicit_preference",
    "task_constraint",
    "inferred_preference",
]
MemoryDecision = Literal[
    "ACCEPT",
    "REJECT",
]


class MemoryCandidate(BaseModel):
    """
    Memory提取内部候选。

    候选只代表“可能写入”的主张，不拥有写入UserMemory的权限；
    后续必须通过原文证据校验和保守Verifier。
    """

    field: MemoryField = Field(description="候选目标字段")
    value: str = Field(description="规范化后的候选偏好值")
    source: MemorySource = Field(description="候选来自用户输入还是人工反馈")
    evidence: str = Field(description="用户原文中的直接证据")
    claim_type: MemoryClaimType = Field(description="候选主张类型")


class MemoryCandidateBatch(BaseModel):
    """
    候选提取阶段的结构化输出。
    """

    candidates: list[MemoryCandidate] = Field(default_factory=list)


class MemoryCandidateDecisionItem(MemoryCandidate):
    """
    Verifier对单个候选的保守裁决。
    """

    decision: MemoryDecision = Field(description="ACCEPT或REJECT")


class MemoryCandidateVerification(BaseModel):
    """
    批量Verifier结构化输出。
    """

    decisions: list[MemoryCandidateDecisionItem] = Field(default_factory=list)


# 生产路径使用两段式structured output：候选提取 -> 保守验证。
memory_candidate_llm = create_structured_llm(
    "memory.candidate_extraction",
    MemoryCandidateBatch
)
memory_verifier_llm = create_structured_llm(
    "memory.conservative_verifier",
    MemoryCandidateVerification
)

# 仅保留给旧测试monkeypatch；正常运行时不参与Memory提取。
memory_update_llm = None


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


def _collect_recent_human_feedback_items(
    human_feedback_history: list[dict] | None,
) -> list[dict]:
    """
    提取有效人工反馈事件。

    approve但没有文字反馈、空字符串反馈都不作为Memory来源；
    有文字的人工反馈保留给LLM判断是否具有跨任务复用价值。
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
    return human_feedback_items[-8:]


def _format_recent_human_feedback_items(
    recent_human_feedback_items: list[dict],
) -> str:
    """
    将已过滤的人工反馈格式化给LLM。
    """
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


def _source_texts_for_candidate(
    candidate: MemoryCandidate,
    user_idea: str,
    recent_human_feedback_items: list[dict],
) -> list[str]:
    """
    根据候选source找到可验证的原始文本集合。
    """
    if candidate.source == "user_idea":
        return [
            user_idea or "",
        ]

    return [
        feedback_item["feedback"]
        for feedback_item in recent_human_feedback_items
    ]


def _candidate_has_valid_evidence(
    candidate: MemoryCandidate,
    user_idea: str,
    recent_human_feedback_items: list[dict],
) -> bool:
    """
    确定性证据校验：evidence必须非空，并且必须逐字出现在对应source原文中。

    这里不判断“校园/地点/人物”等实体类型，只验证候选是否真的有用户原文证据；
    内容是否足以成为长期偏好交给后续保守Verifier。
    """
    evidence = candidate.evidence.strip()

    if not evidence:
        return False

    return any(
        evidence in source_text
        for source_text in _source_texts_for_candidate(
            candidate,
            user_idea,
            recent_human_feedback_items,
        )
    )


def _candidate_key(
    candidate: MemoryCandidate,
) -> tuple[str, str, str, str, str]:
    """
    生成候选身份键，避免Verifier凭空新增未验证候选。
    """
    return (
        candidate.field,
        candidate.value.strip(),
        candidate.source,
        candidate.evidence.strip(),
        candidate.claim_type,
    )


def _filter_candidates_with_evidence(
    candidates: list[MemoryCandidate],
    user_idea: str,
    recent_human_feedback_items: list[dict],
) -> list[MemoryCandidate]:
    """
    过滤掉缺少原文证据或证据无法在source中定位的候选。
    """
    return [
        candidate
        for candidate in candidates
        if candidate.value.strip()
        and _candidate_has_valid_evidence(
            candidate,
            user_idea,
            recent_human_feedback_items,
        )
    ]


def _candidate_batch_to_json(
    candidates: list[MemoryCandidate],
) -> str:
    """
    将候选转换成JSON文本，供Verifier批量判断。
    """
    return json.dumps(
        [
            candidate.model_dump(
                mode="json",
            )
            for candidate in candidates
        ],
        ensure_ascii=False,
        indent=2,
    )


def _accepted_candidates_from_verification(
    verification: MemoryCandidateVerification,
    validated_candidates: list[MemoryCandidate],
    user_idea: str,
    recent_human_feedback_items: list[dict],
) -> list[MemoryCandidate]:
    """
    只接收Verifier明确ACCEPT、且仍能匹配已验证候选身份的结果。
    """
    validated_candidate_by_key = {
        _candidate_key(candidate): candidate
        for candidate in validated_candidates
    }
    accepted_candidates = []

    for decision_item in verification.decisions:
        if decision_item.decision != "ACCEPT":
            continue

        if not _candidate_has_valid_evidence(
            decision_item,
            user_idea,
            recent_human_feedback_items,
        ):
            continue

        candidate = validated_candidate_by_key.get(
            _candidate_key(decision_item)
        )

        if candidate is not None:
            accepted_candidates.append(candidate)

    return accepted_candidates


def _duration_from_candidate_value(
    value: str,
) -> int | None:
    """
    从候选值中提取秒数；解析失败则拒绝写入duration。
    """
    duration_match = re.search(
        r"\d+",
        value,
    )

    if duration_match is None:
        return None

    duration_sec = int(
        duration_match.group()
    )

    if duration_sec <= 0:
        return None

    return duration_sec


def _memory_update_from_candidates(
    candidates: list[MemoryCandidate],
    current_memory: UserMemory,
) -> MemoryUpdate:
    """
    将Verifier接受的候选转换回现有MemoryUpdate结构，外部Merge/Store职责保持不变。
    """
    update_data = {
        "should_update": True,
        "preferred_genres_to_add": [],
        "style_preferences_to_add": [],
        "disliked_elements_to_add": [],
        "preferred_duration_sec": None,
        "additional_preferences_to_add": [],
        "story_preferences_to_add": [],
        "scene_preferences_to_add": [],
    }

    for candidate in candidates:
        value = candidate.value.strip()

        if not value:
            continue

        if candidate.field == "preferred_duration_sec":
            duration_sec = _duration_from_candidate_value(
                value
            )

            if duration_sec is not None:
                update_data["preferred_duration_sec"] = duration_sec

            continue

        update_data[candidate.field].append(
            value
        )

    return _normalize_memory_update(
        MemoryUpdate(**update_data),
        current_memory,
    )


def _render_candidate_prompt(
    user_idea: str,
    current_memory: UserMemory,
    recent_human_feedback_items: list[dict],
) -> RenderedPrompt:
    """
    渲染第一段候选提取Prompt，并保留调用追踪所需元数据。
    """
    current_memory_text = _format_current_memory(
        current_memory
    )
    recent_human_feedback_text = _format_recent_human_feedback_items(
        recent_human_feedback_items
    )

    return render_prompt(
        "memory.candidate_extraction",
        version="v1",
        user_idea=user_idea,
        current_memory_text=current_memory_text,
        recent_human_feedback_text=(
            recent_human_feedback_text
        ),
    )


def _build_candidate_prompt(
    user_idea: str,
    current_memory: UserMemory,
    recent_human_feedback_items: list[dict],
) -> str:
    """
    返回候选提取Prompt正文，保留既有测试和内部调用契约。
    """
    return _render_candidate_prompt(
        user_idea,
        current_memory,
        recent_human_feedback_items,
    ).text


def _render_verifier_prompt(
    candidates: list[MemoryCandidate],
) -> RenderedPrompt:
    """
    渲染保守Verifier Prompt，并保留调用追踪所需元数据。
    """
    candidate_text = _candidate_batch_to_json(
        candidates
    )

    return render_prompt(
        "memory.conservative_verifier",
        version="v1",
        candidate_text=candidate_text,
    )


def _build_verifier_prompt(
    candidates: list[MemoryCandidate],
) -> str:
    """
    返回Verifier Prompt正文，保留既有测试和内部调用契约。
    """
    return _render_verifier_prompt(
        candidates
    ).text


def _legacy_memory_update_for_existing_tests(
    prompt: str,
    current_memory: UserMemory,
) -> MemoryUpdate | None:
    """
    兼容旧测试中直接monkeypatch memory_update_llm的方式。

    正常生产路径下memory_update_llm为None，不会走这里。
    """
    if memory_update_llm is None:
        return None

    raw_update = memory_update_llm.invoke(
        prompt
    )

    if isinstance(
        raw_update,
        MemoryUpdate,
    ):
        return _normalize_memory_update(
            raw_update,
            current_memory,
        )

    return None


def extract_memory_update(
    user_idea: str,    # 用户本次输入的原始需求
    current_memory: UserMemory,    # 当前已经读取到的完整长期记忆
    human_feedback_history: list[dict] | None = None,    # 本次执行中累计的人工反馈历史
) -> MemoryUpdate:
    """
    从用户本次输入和人工反馈中提取长期偏好增量。

    Pipeline：
    1. 候选提取LLM只提出带source/evidence的候选；
    2. 代码层验证evidence必须存在于用户原文；
    3. 保守Verifier批量决定ACCEPT/REJECT；
    4. 只有ACCEPT候选才转换为MemoryUpdate。
    """
    recent_human_feedback_items = _collect_recent_human_feedback_items(
        human_feedback_history
    )

    # 没有任何用户文本来源时直接跳过；这不是长期关键词Gate，只是基础输入保护。
    if not (user_idea or "").strip() and not recent_human_feedback_items:
        return MemoryUpdate(
            should_update=False,
        )

    candidate_rendered = _render_candidate_prompt(
        user_idea,
        current_memory,
        recent_human_feedback_items,
    )

    legacy_update = _legacy_memory_update_for_existing_tests(
        candidate_rendered.text,
        current_memory,
    )

    if legacy_update is not None:
        return legacy_update

    candidate_batch = invoke_structured_llm(
        memory_candidate_llm,
        candidate_rendered,
        node="update_memory",
    )
    validated_candidates = _filter_candidates_with_evidence(
        candidate_batch.candidates,
        user_idea,
        recent_human_feedback_items,
    )

    if not validated_candidates:
        return MemoryUpdate(
            should_update=False,
        )

    verifier_rendered = _render_verifier_prompt(
        validated_candidates
    )
    verification = invoke_structured_llm(
        memory_verifier_llm,
        verifier_rendered,
        node="update_memory",
    )
    accepted_candidates = _accepted_candidates_from_verification(
        verification,
        validated_candidates,
        user_idea,
        recent_human_feedback_items,
    )

    if not accepted_candidates:
        return MemoryUpdate(
            should_update=False,
        )

    return _memory_update_from_candidates(
        accepted_candidates,
        current_memory,
    )
