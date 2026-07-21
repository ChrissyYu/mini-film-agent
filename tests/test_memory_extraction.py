import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.models import MemoryUpdate, UserMemory

extract_memory_module = importlib.import_module("memory.extract_memory")
update_memory_module = importlib.import_module("memory.update_memory")


class FakeMemoryUpdateLLM:
    """
    捕获Prompt并返回预设MemoryUpdate，避免调用真实LLM。
    """

    def __init__(self, result: MemoryUpdate):
        self.result = result
        self.prompts = []

    def invoke(self, prompt: str) -> MemoryUpdate:
        self.prompts.append(prompt)
        return self.result


def _assert_prompt_has_keywords(
    prompt: str,
    keywords: list[str],
) -> None:
    """
    Prompt会按阶段持续精简；测试只绑定关键语义词，避免依赖整句文案。
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


def test_prompt_contains_semantic_dedup_and_normalization_rules(monkeypatch):
    """
    Prompt应明确语义比较、作用域比较和规范化规则。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    extract_memory_module.extract_memory_update(
        user_idea="以后故事结尾更含蓄。",
        current_memory=_current_memory(),
    )

    prompt = fake_llm.prompts[-1]

    assert "提取边界：" in prompt
    assert "作用域与字段：" in prompt
    assert "语义去重：" in prompt
    _assert_prompt_has_keywords(
        prompt,
        [
            "长期",
            "复用",
            "任务专属要求不保存",
            "story_preferences_to_add",
            "scene_preferences_to_add",
            "对应字段",
            "已有 Memory",
            "Scoped",
            "全局",
            "简短",
            "稳定",
        ],
    )


def test_current_memory_is_displayed_by_fields(monkeypatch):
    """
    当前Memory应按字段结构化展示，便于按字段比较。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    extract_memory_module.extract_memory_update(
        user_idea="以后故事更克制。",
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

    prompt = fake_llm.prompts[-1]

    assert '"preferred_genres"' in prompt
    assert '"style_preferences"' in prompt
    assert '"disliked_elements"' in prompt
    assert '"preferred_duration_sec"' in prompt
    assert '"additional_preferences"' in prompt
    assert '"story_preferences"' in prompt
    assert '"scene_preferences"' in prompt


def test_story_feedback_can_enter_story_preferences(monkeypatch):
    """
    通用story人工反馈可以提取为故事偏好增量。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            story_preferences_to_add=[
                "故事结尾保持克制开放",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "以后故事结尾尽量保持克制开放。",
            }
        ],
    )

    prompt = fake_llm.prompts[-1]

    assert result.should_update is True
    assert result.story_preferences_to_add == [
        "故事结尾保持克制开放",
    ]
    assert result.scene_preferences_to_add == []
    assert '"scope": "story"' in prompt
    assert "以后故事结尾尽量保持克制开放" in prompt


def test_story_scoped_preference_is_not_also_global(monkeypatch):
    """
    Story限定偏好不应同时写入全局字段。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            story_preferences_to_add=[
                "故事结尾保持克制开放",
            ],
            disliked_elements_to_add=[
                "故事结尾保持克制开放",
            ],
            additional_preferences_to_add=[
                "故事结尾保持克制开放",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后故事结尾保持克制开放。",
        current_memory=_current_memory(),
    )

    assert result.should_update is True
    assert result.story_preferences_to_add == [
        "故事结尾保持克制开放",
    ]
    assert result.disliked_elements_to_add == []
    assert result.additional_preferences_to_add == []


def test_story_semantic_duplicate_is_not_output(monkeypatch):
    """
    story近义偏好已存在时，提取结果应为空增量。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后故事结尾尽量含蓄一点。",
        current_memory=UserMemory(
            user_id="demo_user",
            story_preferences=[
                "故事结尾保持克制开放",
            ],
        ),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "今后结局别说太满，保持开放。",
            }
        ],
    )

    assert result.should_update is False
    assert result.story_preferences_to_add == []
    assert result.scene_preferences_to_add == []
    assert "故事结尾保持克制开放" in fake_llm.prompts[-1]


def test_scene_feedback_can_enter_scene_preferences(monkeypatch):
    """
    通用scene人工反馈可以提取为分场偏好增量。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            scene_preferences_to_add=[
                "分场动作保持可拍摄",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "scene",
                "decision": "revise",
                "feedback": "今后分场动作都写得更具体、可拍摄。",
            }
        ],
    )

    prompt = fake_llm.prompts[-1]

    assert result.should_update is True
    assert result.scene_preferences_to_add == [
        "分场动作保持可拍摄",
    ]
    assert result.story_preferences_to_add == []
    assert '"scope": "scene"' in prompt
    assert "今后分场动作都写得更具体、可拍摄" in prompt


def test_scene_scoped_preference_is_not_also_global(monkeypatch):
    """
    Scene限定偏好不应同时写入全局字段。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            scene_preferences_to_add=[
                "场景动作减少解释性对白",
            ],
            disliked_elements_to_add=[
                "场景动作减少解释性对白",
            ],
            style_preferences_to_add=[
                "场景动作减少解释性对白",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后场景动作减少解释性对白。",
        current_memory=_current_memory(),
    )

    assert result.should_update is True
    assert result.scene_preferences_to_add == [
        "场景动作减少解释性对白",
    ]
    assert result.disliked_elements_to_add == []
    assert result.style_preferences_to_add == []


def test_scene_semantic_duplicate_is_not_output(monkeypatch):
    """
    scene近义偏好已存在时，提取结果应为空增量。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次继续生成校园故事。",
        current_memory=UserMemory(
            user_id="demo_user",
            scene_preferences=[
                "分场动作保持可拍摄",
            ],
        ),
        human_feedback_history=[
            {
                "scope": "scene",
                "decision": "revise",
                "feedback": "以后每场动作都要能实际拍出来。",
            }
        ],
    )

    assert result.should_update is False
    assert result.scene_preferences_to_add == []
    assert result.story_preferences_to_add == []
    assert "分场动作保持可拍摄" in fake_llm.prompts[-1]


def test_story_and_scene_similar_preferences_do_not_dedup_across_scope(monkeypatch):
    """
    story与scene内容相似时，不应跨作用域去重。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            scene_preferences_to_add=[
                "分场表达保持克制",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=UserMemory(
            user_id="demo_user",
            story_preferences=[
                "故事表达保持克制",
            ],
        ),
        human_feedback_history=[
            {
                "scope": "scene",
                "decision": "revise",
                "feedback": "以后分场表达也保持克制。",
            }
        ],
    )

    assert result.should_update is True
    assert result.scene_preferences_to_add == [
        "分场表达保持克制",
    ]
    assert result.story_preferences_to_add == []
    _assert_prompt_has_keywords(
        fake_llm.prompts[-1],
        [
            "Story",
            "Scene",
            "分别管理",
            "不跨作用域互相删除",
        ],
    )


def test_story_and_scene_scoped_preferences_remain_independent(monkeypatch):
    """
    Story与Scene scoped偏好即使文本相同，也不互相删除。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            story_preferences_to_add=[
                "表达保持克制",
            ],
            scene_preferences_to_add=[
                "表达保持克制",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后故事和分场表达都保持克制。",
        current_memory=_current_memory(),
    )

    assert result.should_update is True
    assert result.story_preferences_to_add == [
        "表达保持克制",
    ]
    assert result.scene_preferences_to_add == [
        "表达保持克制",
    ]


def test_different_new_preference_can_still_be_output(monkeypatch):
    """
    确实不同的新偏好仍应允许输出。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            story_preferences_to_add=[
                "故事冲突保持生活化",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后故事冲突尽量从日常生活里生长出来。",
        current_memory=UserMemory(
            user_id="demo_user",
            story_preferences=[
                "故事结尾保持克制开放",
            ],
        ),
    )

    assert result.should_update is True
    assert result.story_preferences_to_add == [
        "故事冲突保持生活化",
    ]


def test_existing_scoped_preference_cannot_be_rewritten_as_global(monkeypatch):
    """
    已有scoped近义偏好不能通过全局字段重复写入。
    """
    existing_preference = "故事大纲聚焦人物关系与故事线推进，弱化具体动作描写"
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            disliked_elements_to_add=[
                existing_preference,
            ],
            additional_preferences_to_add=[
                existing_preference,
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后大纲别堆太多动作，主要写人物关系和情节发展。",
        current_memory=UserMemory(
            user_id="demo_user",
            story_preferences=[
                existing_preference,
            ],
        ),
    )

    assert result.should_update is False
    assert result.disliked_elements_to_add == []
    assert result.additional_preferences_to_add == []


def test_explicit_global_preference_can_enter_global_field(monkeypatch):
    """
    用户明确扩大到所有创作的偏好，仍可进入全局字段。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            disliked_elements_to_add=[
                "所有创作减少大量旁白",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后所有创作都不要大量旁白。",
        current_memory=UserMemory(
            user_id="demo_user",
            story_preferences=[
                "故事大纲减少解释性文字",
            ],
        ),
    )

    assert result.should_update is True
    assert result.disliked_elements_to_add == [
        "所有创作减少大量旁白",
    ]


def test_scope_example_semantic_duplicate_becomes_no_update(monkeypatch):
    """
    M6.1示例属于story scoped语义重复，没有其他增量时不更新。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
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

    prompt = fake_llm.prompts[-1]

    assert result.should_update is False
    assert result.story_preferences_to_add == []
    assert result.disliked_elements_to_add == []
    assert "以后大纲别堆太多动作，主要写人物关系和情节发展" in prompt
    _assert_prompt_has_keywords(
        prompt,
        [
            "故事",
            "大纲",
            "story_preferences_to_add",
            "disliked_elements_to_add",
            "整体创作",
            "对应字段",
            "含义相同",
        ],
    )


def test_current_task_specific_feedback_is_not_saved(monkeypatch):
    """
    绑定当前人物、地点或剧情的修改不应写入长期Memory。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "把林夏改成在图书馆遇到陈屿。",
            }
        ],
    )

    assert result.should_update is False
    assert result.story_preferences_to_add == []
    _assert_prompt_has_keywords(
        fake_llm.prompts[-1],
        [
            "当前人物",
            "地点",
            "场次",
            "具体剧情",
            "任务专属要求不保存",
        ],
    )


def test_approve_without_feedback_does_not_add_feedback_to_prompt(monkeypatch):
    """
    approve且无文字反馈时，不应产生可提取的人工反馈。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="再生成一个校园故事。",
        current_memory=_current_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "approve",
                "feedback": None,
            }
        ],
    )

    assert result.should_update is False
    assert "无人工反馈" in fake_llm.prompts[-1]


def test_story_and_scene_scopes_are_not_mixed(monkeypatch):
    """
    story和scene反馈应分别受各自字段约束。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            story_preferences_to_add=[
                "故事冲突保持内敛",
            ],
            scene_preferences_to_add=[
                "分场减少解释性对白",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
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

    prompt = fake_llm.prompts[-1]

    assert result.story_preferences_to_add == [
        "故事冲突保持内敛",
    ]
    assert result.scene_preferences_to_add == [
        "分场减少解释性对白",
    ]
    _assert_prompt_has_keywords(
        prompt,
        [
            "故事",
            "大纲",
            "剧情结构",
            "story_preferences_to_add",
            "分场",
            "场景",
            "场景动作",
            "scene_preferences_to_add",
        ],
    )


def test_user_idea_extraction_still_works_without_human_feedback(monkeypatch):
    """
    没有人工反馈时，仍保留从用户输入提取长期偏好的能力。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            style_preferences_to_add=[
                "现实主义",
                "克制表达",
            ],
            disliked_elements_to_add=[
                "大量旁白",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后帮我生成短片时，请保持现实主义和克制表达，不要使用大量旁白。",
        current_memory=_current_memory(),
    )

    assert result.should_update is True
    assert result.style_preferences_to_add == [
        "现实主义",
        "克制表达",
    ]
    assert result.disliked_elements_to_add == [
        "大量旁白",
    ]
    assert "无人工反馈" in fake_llm.prompts[-1]


def test_only_recent_eight_non_empty_feedback_items_are_used(monkeypatch):
    """
    人工反馈超过8条时，只把最近8条非空用户反馈放入Prompt。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
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

    prompt = fake_llm.prompts[-1]

    assert "长期反馈0" not in prompt
    assert "长期反馈1" not in prompt
    for index in range(2, 10):
        assert f"长期反馈{index}" in prompt
    assert '"feedback": "   "' not in prompt


def test_approve_with_text_feedback_is_kept(monkeypatch):
    """
    approve如果带有文字反馈，仍应保留给提取器判断。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
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

    prompt = fake_llm.prompts[-1]

    assert '"decision": "approve"' in prompt
    assert "以后也保持这种克制结尾" in prompt


def test_prompt_does_not_include_machine_review_or_final_output(monkeypatch):
    """
    Prompt只应包含用户反馈字段，不应混入机器Review或最终输出。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
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

    prompt = fake_llm.prompts[-1]

    assert "以后故事结尾更克制" in prompt
    assert machine_review_sentinel not in prompt
    assert final_output_sentinel not in prompt
    _assert_prompt_has_keywords(
        prompt,
        [
            "不依据",
            "机器 Review",
            "final_output",
        ],
    )


def test_update_memory_passes_human_feedback_history_to_extractor(monkeypatch):
    """
    update_memory应把State中的人工反馈历史传给提取器。
    """
    captured_args = {}
    current_memory = _current_memory()
    memory_update = MemoryUpdate(
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


def test_should_update_false_clears_leftover_increments(monkeypatch):
    """
    LLM返回should_update=False时，残留增量字段应被代码清空。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
            style_preferences_to_add=[
                "残留风格",
            ],
            story_preferences_to_add=[
                "残留故事偏好",
            ],
            preferred_duration_sec=60,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
    )

    assert result.should_update is False
    assert result.style_preferences_to_add == []
    assert result.story_preferences_to_add == []
    assert result.preferred_duration_sec is None


def test_should_update_true_without_increments_becomes_false(monkeypatch):
    """
    LLM返回should_update=True但没有任何增量时，应自动改为False。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="这次生成一个校园故事。",
        current_memory=_current_memory(),
    )

    assert result.should_update is False
    assert result.preferred_genres_to_add == []
    assert result.preferred_duration_sec is None


def test_valid_increment_is_not_removed(monkeypatch):
    """
    合法非空增量不应被输出校正误删。
    """
    fake_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            scene_preferences_to_add=[
                "分场动作更具体",
            ],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_llm,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea="以后分场动作更具体。",
        current_memory=_current_memory(),
    )

    assert result.should_update is True
    assert result.scene_preferences_to_add == [
        "分场动作更具体",
    ]
