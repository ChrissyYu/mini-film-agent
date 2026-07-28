import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.models import MemoryUpdate, UserMemory

extract_memory_module = importlib.import_module("memory.extract_memory")
update_memory_module = importlib.import_module("memory.update_memory")


class FakeCandidateLLM:
    """
    捕获候选提取Prompt并返回预设候选，避免调用真实LLM。
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
    捕获Verifier Prompt并返回预设裁决，避免调用真实LLM。
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
    """
    构造内部Memory候选，测试重点放在证据和Verifier流程上。
    """
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
    """
    Verifier必须回传同一个候选身份，代码才会接受ACCEPT。
    """
    return extract_memory_module.MemoryCandidateDecisionItem(
        **candidate.model_dump(
            mode="json",
        ),
        decision=decision,
    )


def _patch_memory_pipeline(
    monkeypatch,
    candidates,
    decisions,
):
    """
    同时替换候选提取LLM和批量Verifier。
    """
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


def _assert_prompt_has_keywords(
    prompt: str,
    keywords: list[str],
) -> None:
    """
    Prompt会继续迭代；测试绑定稳定规则和字段，不绑定整段自然语言。
    """
    for keyword in keywords:
        assert keyword in prompt


def _current_memory() -> UserMemory:
    """
    构造测试用当前长期Memory。
    """
    return UserMemory(
        user_id="demo_user",
        style_preferences=["现实主义"],
    )


def _assert_no_increment(update) -> None:
    """
    统一断言没有任何长期偏好增量。
    """
    assert update.should_update is False
    assert update.preferred_genres_to_add == []
    assert update.style_preferences_to_add == []
    assert update.disliked_elements_to_add == []
    assert update.preferred_duration_sec is None
    assert update.additional_preferences_to_add == []
    assert update.story_preferences_to_add == []
    assert update.scene_preferences_to_add == []


def test_candidate_and_verifier_prompts_describe_new_pipeline(monkeypatch):
    """
    Prompt应明确候选无写入权、Evidence Validation前置、Verifier保守裁决。
    """
    candidate = _candidate(
        "story_preferences_to_add",
        "故事结尾保持克制开放",
        "user_idea",
        "以后故事结尾保持克制开放",
    )
    fake_candidate_llm, fake_verifier_llm = _patch_memory_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "REJECT",
            )
        ],
    )

    extract_memory_module.extract_memory_update(
        user_idea="以后故事结尾保持克制开放。",
        current_memory=_current_memory(),
    )

    candidate_prompt = fake_candidate_llm.prompts[-1]
    verifier_prompt = fake_verifier_llm.prompts[-1]

    _assert_prompt_has_keywords(
        candidate_prompt,
        [
            "没有写入Memory的权限",
            "evidence必须逐字来自对应source原文",
            "explicit_preference",
            "task_constraint",
            "inferred_preference",
            "story_preferences_to_add",
            "scene_preferences_to_add",
            "语义比较",
            "简短",
            "稳定",
        ],
    )
    _assert_prompt_has_keywords(
        verifier_prompt,
        [
            "ACCEPT",
            "REJECT",
            "高置信",
            "必须REJECT单次duration",
            "不确定时一律REJECT",
            "不使用浮点confidence",
        ],
    )


def test_current_memory_is_displayed_by_fields(monkeypatch):
    """
    当前Memory应按字段结构化展示，便于候选提取按字段比较。
    """
    fake_candidate_llm, _ = _patch_memory_pipeline(
        monkeypatch,
        [],
        [],
    )

    extract_memory_module.extract_memory_update(
        user_idea="我讨厌大团圆结局。",
        current_memory=UserMemory(
            user_id="demo_user",
            preferred_genres=["校园"],
            style_preferences=["现实主义"],
            disliked_elements=["大量旁白"],
            preferred_duration_sec=60,
            additional_preferences=["少角色"],
            story_preferences=["克制开放式结尾"],
            scene_preferences=["动作可拍摄"],
        ),
    )

    prompt = fake_candidate_llm.prompts[-1]

    assert '"preferred_genres"' in prompt
    assert '"style_preferences"' in prompt
    assert '"disliked_elements"' in prompt
    assert '"preferred_duration_sec"' in prompt
    assert '"additional_preferences"' in prompt
    assert '"story_preferences"' in prompt
    assert '"scene_preferences"' in prompt


def test_explicit_dislike_without_old_regex_keyword_can_be_saved(monkeypatch):
    """
    “我讨厌...”不依赖旧regex关键词，也能通过证据和Verifier写入。
    """
    candidate = _candidate(
        "disliked_elements_to_add",
        "大团圆结局",
        "user_idea",
        "我讨厌大团圆结局",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="我讨厌大团圆结局。",
        current_memory=_current_memory(),
    )

    assert result.should_update is True
    assert result.disliked_elements_to_add == [
        "大团圆结局",
    ]


def test_one_time_request_rejected_by_conservative_verifier(monkeypatch):
    """
    一次性创作需求即使形成候选，也会被Verifier拒绝，避免写入长期Memory。
    """
    duration = _candidate(
        "preferred_duration_sec",
        "20",
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
    _patch_memory_pipeline(
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

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个20秒的校园晨跑短片，主角在操场完成一次小小的自我鼓励。",
        current_memory=_current_memory(),
    )

    _assert_no_increment(result)


def test_mixed_task_and_preference_only_saves_explicit_preference(monkeypatch):
    """
    本次题材选择不应变成偏好；同一句里的明确厌恶可以保存。
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
    _patch_memory_pipeline(
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

    result = extract_memory_module.extract_memory_update(
        user_idea="这次写一个校园片，我讨厌大团圆结局。",
        current_memory=_current_memory(),
    )

    assert result.preferred_genres_to_add == []
    assert result.disliked_elements_to_add == [
        "大团圆结局",
    ]


def test_current_story_scene_feedback_is_rejected(monkeypatch):
    """
    只能解释为当前分场修改的HITL反馈，不应提升为长期Memory。
    """
    candidate = _candidate(
        "scene_preferences_to_add",
        "减少场景动作",
        "human_feedback",
        "第三场动作太多了，把争吵删掉一些",
        "task_constraint",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "REJECT",
            )
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "scene",
                "decision": "revise",
                "feedback": "第三场动作太多了，把争吵删掉一些。",
            }
        ],
    )

    _assert_no_increment(result)


def test_long_term_story_feedback_can_be_saved(monkeypatch):
    """
    明确长期的story反馈可以写入story_preferences。
    """
    candidate = _candidate(
        "story_preferences_to_add",
        "故事大纲少写具体动作，关注人物关系",
        "human_feedback",
        "以后故事大纲少写具体动作，我更关注人物关系",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "以后故事大纲少写具体动作，我更关注人物关系。",
            }
        ],
    )

    assert result.story_preferences_to_add == [
        "故事大纲少写具体动作，关注人物关系",
    ]
    assert result.scene_preferences_to_add == []


def test_missing_evidence_is_rejected_before_verifier(monkeypatch):
    """
    evidence不存在于对应原文时，代码层直接拒绝，Verifier不会被调用。
    """
    candidate = _candidate(
        "style_preferences_to_add",
        "克制表达",
        "user_idea",
        "不存在的证据",
    )
    fake_candidate_llm, fake_verifier_llm = _patch_memory_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="我讨厌大团圆结局。",
        current_memory=_current_memory(),
    )

    _assert_no_increment(result)
    assert len(fake_candidate_llm.prompts) == 1
    assert fake_verifier_llm.prompts == []


def test_blank_input_and_no_feedback_skips_all_llm(monkeypatch):
    """
    没有用户输入且没有有效人工反馈时，基础输入保护直接跳过。
    """
    fake_candidate_llm, fake_verifier_llm = _patch_memory_pipeline(
        monkeypatch,
        [],
        [],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="   ",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "approve",
                "feedback": None,
            }
        ],
    )

    _assert_no_increment(result)
    assert fake_candidate_llm.prompts == []
    assert fake_verifier_llm.prompts == []


def test_only_recent_eight_non_empty_feedback_items_are_used(monkeypatch):
    """
    人工反馈超过8条时，只把最近8条非空用户反馈放入候选Prompt。
    """
    fake_candidate_llm, _ = _patch_memory_pipeline(
        monkeypatch,
        [],
        [],
    )
    feedback_history = [
        {
            "scope": "story",
            "decision": "revise",
            "feedback": f"长期反馈{index}",
        }
        for index in range(10)
    ]
    feedback_history.insert(
        3,
        {
            "scope": "scene",
            "decision": "approve",
            "feedback": "   ",
        },
    )

    extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=feedback_history,
    )

    prompt = fake_candidate_llm.prompts[-1]

    assert "长期反馈0" not in prompt
    assert "长期反馈1" not in prompt
    for index in range(2, 10):
        assert f"长期反馈{index}" in prompt
    assert '"feedback": "   "' not in prompt


def test_approve_with_text_feedback_is_kept(monkeypatch):
    """
    approve如果带有文字反馈，仍应保留给候选提取器判断。
    """
    fake_candidate_llm, _ = _patch_memory_pipeline(
        monkeypatch,
        [],
        [],
    )

    extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "approve",
                "feedback": "以后也保持这种克制结尾。",
            }
        ],
    )

    prompt = fake_candidate_llm.prompts[-1]

    assert '"decision": "approve"' in prompt
    assert "以后也保持这种克制结尾" in prompt


def test_machine_review_and_final_output_do_not_enter_prompt(monkeypatch):
    """
    Prompt只应包含用户输入和用户反馈字段，不应混入机器Review或最终输出。
    """
    fake_candidate_llm, _ = _patch_memory_pipeline(
        monkeypatch,
        [],
        [],
    )
    machine_review_sentinel = "SENTINEL_MACHINE_REVIEW_CONTENT"
    final_output_sentinel = "SENTINEL_FINAL_OUTPUT_CONTENT"

    extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "以后故事结尾更克制。",
                "story_review_result": machine_review_sentinel,
                "final_output": final_output_sentinel,
            }
        ],
    )

    prompt = fake_candidate_llm.prompts[-1]

    assert "以后故事结尾更克制" in prompt
    assert machine_review_sentinel not in prompt
    assert final_output_sentinel not in prompt
    _assert_prompt_has_keywords(
        prompt,
        [
            "机器 Review",
            "final_output",
        ],
    )


def test_story_and_scene_scopes_are_not_mixed(monkeypatch):
    """
    story和scene反馈应分别进入各自字段。
    """
    story_candidate = _candidate(
        "story_preferences_to_add",
        "故事冲突保持内敛",
        "human_feedback",
        "以后故事冲突保持内敛",
    )
    scene_candidate = _candidate(
        "scene_preferences_to_add",
        "分场减少解释性对白",
        "human_feedback",
        "以后分场减少解释性对白",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [
            story_candidate,
            scene_candidate,
        ],
        [
            _decision(story_candidate, "ACCEPT"),
            _decision(scene_candidate, "ACCEPT"),
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "以后故事冲突保持内敛。",
            },
            {
                "scope": "scene",
                "decision": "revise",
                "feedback": "以后分场减少解释性对白。",
            },
        ],
    )

    assert result.story_preferences_to_add == [
        "故事冲突保持内敛",
    ]
    assert result.scene_preferences_to_add == [
        "分场减少解释性对白",
    ]


def test_scoped_preference_is_not_also_global_after_conversion(monkeypatch):
    """
    Scoped候选和全局候选同文本时，最终只保留Scoped字段。
    """
    story_candidate = _candidate(
        "story_preferences_to_add",
        "故事结尾保持克制开放",
        "user_idea",
        "以后故事结尾保持克制开放",
    )
    global_candidate = _candidate(
        "additional_preferences_to_add",
        "故事结尾保持克制开放",
        "user_idea",
        "以后故事结尾保持克制开放",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [
            story_candidate,
            global_candidate,
        ],
        [
            _decision(story_candidate, "ACCEPT"),
            _decision(global_candidate, "ACCEPT"),
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后故事结尾保持克制开放。",
        current_memory=_current_memory(),
    )

    assert result.story_preferences_to_add == [
        "故事结尾保持克制开放",
    ]
    assert result.additional_preferences_to_add == []


def test_scene_scoped_preference_is_not_also_global_after_conversion(
    monkeypatch,
):
    """
    Scene候选与全局候选同文本时，最终只保留Scene作用域字段。
    """
    scene_candidate = _candidate(
        "scene_preferences_to_add",
        "分场动作保持具体可拍摄",
        "user_idea",
        "以后分场动作保持具体可拍摄",
    )
    global_candidate = _candidate(
        "additional_preferences_to_add",
        "分场动作保持具体可拍摄",
        "user_idea",
        "以后分场动作保持具体可拍摄",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [
            scene_candidate,
            global_candidate,
        ],
        [
            _decision(scene_candidate, "ACCEPT"),
            _decision(global_candidate, "ACCEPT"),
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后分场动作保持具体可拍摄。",
        current_memory=_current_memory(),
    )

    assert result.scene_preferences_to_add == [
        "分场动作保持具体可拍摄",
    ]
    assert result.additional_preferences_to_add == []


def test_existing_scoped_preference_duplicate_becomes_no_update(monkeypatch):
    """
    候选与已有Scoped Memory同文本时，normalize后没有真实增量。
    """
    candidate = _candidate(
        "additional_preferences_to_add",
        "故事大纲聚焦人物关系与故事线推进，弱化具体动作描写",
        "user_idea",
        "以后大纲别堆太多动作",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后大纲别堆太多动作，主要写人物关系和情节发展。",
        current_memory=UserMemory(
            user_id="demo_user",
            story_preferences=[
                "故事大纲聚焦人物关系与故事线推进，弱化具体动作描写",
            ],
        ),
    )

    _assert_no_increment(result)


def test_should_update_false_clears_leftover_increments():
    """
    即使上游返回残留字段，should_update=False也必须归一化为空增量。
    """
    result = extract_memory_module._normalize_memory_update(
        MemoryUpdate(
            should_update=False,
            style_preferences_to_add=[
                "残留风格",
            ],
            story_preferences_to_add=[
                "残留故事偏好",
            ],
            preferred_duration_sec=60,
        ),
        _current_memory(),
    )

    _assert_no_increment(result)


def test_should_update_true_without_increments_becomes_false():
    """
    should_update=True但没有真实增量时，应校正为无需更新。
    """
    result = extract_memory_module._normalize_memory_update(
        MemoryUpdate(
            should_update=True,
        ),
        _current_memory(),
    )

    _assert_no_increment(result)


def test_story_and_scene_same_preference_remain_independent():
    """
    Story与Scene作用域独立，相同文本不能跨作用域互相删除。
    """
    result = extract_memory_module._normalize_memory_update(
        MemoryUpdate(
            should_update=True,
            story_preferences_to_add=[
                "表达保持克制",
            ],
            scene_preferences_to_add=[
                "表达保持克制",
            ],
        ),
        _current_memory(),
    )

    assert result.should_update is True
    assert result.story_preferences_to_add == [
        "表达保持克制",
    ]
    assert result.scene_preferences_to_add == [
        "表达保持克制",
    ]


def test_approve_without_feedback_is_filtered_from_candidate_prompt(monkeypatch):
    """
    即使user_idea非空，无文字approve也不应成为候选提取上下文。
    """
    fake_candidate_llm, _ = _patch_memory_pipeline(
        monkeypatch,
        [],
        [],
    )

    extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "approve",
                "feedback": None,
            }
        ],
    )

    prompt = fake_candidate_llm.prompts[-1]

    assert "无人工反馈" in prompt
    assert '"decision": "approve"' not in prompt


def test_duration_candidate_can_be_converted(monkeypatch):
    """
    Verifier接受长期时长候选后，会转换为preferred_duration_sec。
    """
    candidate = _candidate(
        "preferred_duration_sec",
        "60秒",
        "user_idea",
        "以后生成的短片默认控制在60秒",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [candidate],
        [
            _decision(
                candidate,
                "ACCEPT",
            )
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后生成的短片默认控制在60秒。",
        current_memory=_current_memory(),
    )

    assert result.should_update is True
    assert result.preferred_duration_sec == 60


def test_verifier_cannot_accept_invented_candidate(monkeypatch):
    """
    Verifier如果返回未经过证据校验的候选，代码层不会接受。
    """
    original_candidate = _candidate(
        "disliked_elements_to_add",
        "大团圆结局",
        "user_idea",
        "我讨厌大团圆结局",
    )
    invented_candidate = _candidate(
        "style_preferences_to_add",
        "现实主义",
        "user_idea",
        "我讨厌大团圆结局",
    )
    _patch_memory_pipeline(
        monkeypatch,
        [original_candidate],
        [
            _decision(
                invented_candidate,
                "ACCEPT",
            )
        ],
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="我讨厌大团圆结局。",
        current_memory=_current_memory(),
    )

    _assert_no_increment(result)


def test_update_memory_passes_human_feedback_history_to_extractor(monkeypatch):
    """
    update_memory应把State中的人工反馈历史传给提取器。
    """
    captured_args = {}
    current_memory = _current_memory()
    memory_update = extract_memory_module.MemoryUpdate(
        should_update=False,
    )

    def fake_extract_memory_update(
        user_idea,
        current_memory_arg,
        human_feedback_history,
    ):
        captured_args["user_idea"] = user_idea
        captured_args["current_memory"] = current_memory_arg
        captured_args["human_feedback_history"] = human_feedback_history
        return memory_update

    monkeypatch.setattr(
        update_memory_module,
        "extract_memory_update",
        fake_extract_memory_update,
    )
    monkeypatch.setattr(
        update_memory_module,
        "merge_user_memory",
        lambda current_memory, update: current_memory,
    )

    result = update_memory_module.update_memory(
        {
            "user_id": "demo_user",
            "user_idea": "这次生成一个校园故事。",
            "user_memory": current_memory,
            "human_feedback_history": [
                {
                    "scope": "story",
                    "decision": "revise",
                    "feedback": "以后结尾更克制。",
                }
            ],
        }
    )

    assert result["memory_update_status"] == "skipped"
    assert captured_args["human_feedback_history"] == [
        {
            "scope": "story",
            "decision": "revise",
            "feedback": "以后结尾更克制。",
        }
    ]
