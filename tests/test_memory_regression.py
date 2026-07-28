import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.models import UserMemory

extract_memory_module = importlib.import_module("memory.extract_memory")


class FakeCandidateLLM:
    """
    回归测试用候选提取LLM替身。
    """

    def __init__(self, candidates):
        self.result = extract_memory_module.MemoryCandidateBatch(
            candidates=candidates,
        )
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.result


class FakeVerifierLLM:
    """
    回归测试用批量Verifier替身。
    """

    def __init__(self, decisions):
        self.result = extract_memory_module.MemoryCandidateVerification(
            decisions=decisions,
        )
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.result


def _candidate(
    field: str,
    value: str,
    source: str,
    evidence: str,
    claim_type: str = "explicit_preference",
):
    return extract_memory_module.MemoryCandidate(
        field=field,
        value=value,
        source=source,
        evidence=evidence,
        claim_type=claim_type,
    )


def _decision(
    candidate,
    decision: str,
):
    return extract_memory_module.MemoryCandidateDecisionItem(
        **candidate.model_dump(
            mode="json",
        ),
        decision=decision,
    )


def _patch_pipeline(
    monkeypatch,
    candidates,
    decisions,
):
    fake_candidate_llm = FakeCandidateLLM(
        candidates,
    )
    fake_verifier_llm = FakeVerifierLLM(
        decisions,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_candidate_llm",
        fake_candidate_llm,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_verifier_llm",
        fake_verifier_llm,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        None,
    )
    return fake_candidate_llm, fake_verifier_llm


def _empty_memory() -> UserMemory:
    return UserMemory(
        user_id="memory_guard_user",
    )


def _assert_no_increment(update) -> None:
    assert update.should_update is False
    assert update.preferred_genres_to_add == []
    assert update.style_preferences_to_add == []
    assert update.disliked_elements_to_add == []
    assert update.preferred_duration_sec is None
    assert update.additional_preferences_to_add == []
    assert update.story_preferences_to_add == []
    assert update.scene_preferences_to_add == []


def test_one_time_creative_request_does_not_create_memory(monkeypatch):
    """
    真实失败样本：候选可被提出，但Verifier必须拒绝一次性创作内容。
    """
    duration = _candidate(
        "preferred_duration_sec",
        "20秒",
        "user_idea",
        "20秒",
        "task_constraint",
    )
    campus = _candidate(
        "preferred_genres_to_add",
        "校园片",
        "user_idea",
        "校园晨跑短片",
        "inferred_preference",
    )
    self_encouragement = _candidate(
        "story_preferences_to_add",
        "自我鼓励",
        "user_idea",
        "自我鼓励",
        "inferred_preference",
    )
    fake_candidate_llm, fake_verifier_llm = _patch_pipeline(
        monkeypatch,
        [
            duration,
            campus,
            self_encouragement,
        ],
        [
            _decision(duration, "REJECT"),
            _decision(campus, "REJECT"),
            _decision(self_encouragement, "REJECT"),
        ],
    )

    update = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个20秒的校园晨跑短片，主角在操场完成一次小小的自我鼓励。",
        current_memory=_empty_memory(),
    )

    _assert_no_increment(update)
    assert len(fake_candidate_llm.prompts) == 1
    assert len(fake_verifier_llm.prompts) == 1


def test_dislike_without_old_hard_gate_keyword_is_saved(monkeypatch):
    """
    “我讨厌...”不在旧Hard Gate白名单中，仍能凭证据和Verifier保存。
    """
    candidate = _candidate(
        "disliked_elements_to_add",
        "大团圆结局",
        "user_idea",
        "我讨厌大团圆结局",
    )
    _patch_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    update = extract_memory_module.extract_memory_update(
        user_idea="我讨厌大团圆结局。",
        current_memory=_empty_memory(),
    )

    assert update.disliked_elements_to_add == [
        "大团圆结局",
    ]


def test_mixed_task_genre_and_preference_only_saves_preference(monkeypatch):
    """
    同一句中，当前任务题材被拒绝，明确偏好被接受。
    """
    campus = _candidate(
        "preferred_genres_to_add",
        "校园片",
        "user_idea",
        "这次写一个校园片",
        "task_constraint",
    )
    dislike = _candidate(
        "disliked_elements_to_add",
        "大团圆结局",
        "user_idea",
        "我讨厌大团圆结局",
    )
    _patch_pipeline(
        monkeypatch,
        [
            campus,
            dislike,
        ],
        [
            _decision(campus, "REJECT"),
            _decision(dislike, "ACCEPT"),
        ],
    )

    update = extract_memory_module.extract_memory_update(
        user_idea="这次写一个校园片，我讨厌大团圆结局。",
        current_memory=_empty_memory(),
    )

    assert update.preferred_genres_to_add == []
    assert update.disliked_elements_to_add == [
        "大团圆结局",
    ]


def test_current_hitl_feedback_is_rejected_as_long_term_memory(monkeypatch):
    """
    当前场次修改可以供Revision使用，但Verifier应拒绝写入长期Memory。
    """
    candidate = _candidate(
        "scene_preferences_to_add",
        "减少场景动作",
        "human_feedback",
        "第三场动作太多了，把争吵删掉一些",
        "task_constraint",
    )
    _patch_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "REJECT",
            )
        ],
    )

    update = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_empty_memory(),
        human_feedback_history=[
            {
                "scope": "scene",
                "decision": "revise",
                "feedback": "第三场动作太多了，把争吵删掉一些。",
            }
        ],
    )

    _assert_no_increment(update)


def test_long_term_hitl_story_feedback_is_saved(monkeypatch):
    """
    明确长期的人工反馈可以保存到合适的story scope。
    """
    candidate = _candidate(
        "story_preferences_to_add",
        "故事大纲少写具体动作，关注人物关系",
        "human_feedback",
        "以后故事大纲少写具体动作，我更关注人物关系",
    )
    _patch_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    update = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_empty_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "以后故事大纲少写具体动作，我更关注人物关系。",
            }
        ],
    )

    assert update.story_preferences_to_add == [
        "故事大纲少写具体动作，关注人物关系",
    ]


def test_missing_evidence_is_rejected_deterministically(monkeypatch):
    """
    evidence不在原文中时，代码层拒绝，Verifier没有机会放行。
    """
    candidate = _candidate(
        "style_preferences_to_add",
        "现实主义",
        "user_idea",
        "用户没有说过的证据",
    )
    _, fake_verifier_llm = _patch_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    update = extract_memory_module.extract_memory_update(
        user_idea="我讨厌大团圆结局。",
        current_memory=_empty_memory(),
    )

    _assert_no_increment(update)
    assert fake_verifier_llm.prompts == []
