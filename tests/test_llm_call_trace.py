import importlib
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

import nodes
from memory.models import UserMemory
from observability.llm_calls import (
    collect_llm_call_trace,
    get_failed_llm_call_event,
    invoke_structured_llm,
)
from observability.summarize import build_execution_summary
from prompts.renderer import render_prompt
from schemas import (
    Character,
    CharacterList,
    FilmBrief,
    StoryCriticResult,
    StoryOutline,
)
from state import FilmState


extract_memory_module = importlib.import_module(
    "memory.extract_memory"
)
update_memory_module = importlib.import_module(
    "memory.update_memory"
)
review_story_module = importlib.import_module(
    "reviews.review_story"
)


class FakeStructuredLLM:
    """
    返回预设结果并记录调用次数，不访问真实模型或网络。
    """

    def __init__(
        self,
        results=None,
        error: Exception | None = None,
    ):
        self.results = list(
            results or []
        )
        self.error = error
        self.prompts = []

    def invoke(
        self,
        prompt: str,
    ):
        self.prompts.append(
            prompt
        )

        if self.error is not None:
            raise self.error

        if not self.results:
            return None

        return self.results.pop(0)


def _film_brief() -> FilmBrief:
    return FilmBrief(
        target_duration_sec=30,
        genre="校园",
        core_theme="成长",
        visual_style="自然克制",
        recommended_scene_count=3,
    )


def _character_result() -> CharacterList:
    return CharacterList(
        characters=[
            Character(
                name="林夏",
                role="主角",
                appearance="白衬衫",
                personality=["克制"],
                motivation="完成告别",
                continuity_constraints=[],
            )
        ]
    )


def _story_review_state(
    history=None,
) -> FilmState:
    return {
        "user_idea": "生成一个校园故事",
        "film_brief": _film_brief(),
        "characters": _character_result().characters,
        "story_outline": StoryOutline(
            setup="林夏准备离校。",
            conflict="她舍不得告别。",
            turning_point="她理解离开也是成长。",
            ending="她平静离开。",
            theme="克制的成长。",
        ),
        "story_review_history": history or [],
        "story_revision_count": len(
            history or []
        ),
    }


def test_single_structured_llm_call_records_registry_metadata(
    monkeypatch,
):
    """
    单次生成调用应只记录一条来自RenderedPrompt、Binding和Profile的元数据。
    """
    fake_llm = FakeStructuredLLM(
        [
            _character_result(),
        ]
    )
    monkeypatch.setattr(
        nodes,
        "character_llm",
        fake_llm,
    )

    result = nodes.design_characters(
        {
            "user_idea": "SENSITIVE_USER_IDEA_SENTINEL",
            "film_brief": _film_brief(),
        }
    )
    event = result[
        "llm_call_trace"
    ][0]

    assert event == {
        "node": "design_characters",
        "prompt_name": "generation.design_characters",
        "prompt_version": "v1",
        "prompt_chars": len(
            fake_llm.prompts[0]
        ),
        "llm_profile": "balanced",
        "model_name": "qwen-plus",
        "temperature": 0,
        "status": "success",
        "duration_ms": event["duration_ms"],
        "error_type": None,
    }
    assert event["duration_ms"] >= 0
    assert set(event) == {
        "node",
        "prompt_name",
        "prompt_version",
        "prompt_chars",
        "llm_profile",
        "model_name",
        "temperature",
        "status",
        "duration_ms",
        "error_type",
    }
    assert (
        "SENSITIVE_USER_IDEA_SENTINEL"
        not in str(event)
    )


def test_review_loop_returns_one_llm_event_per_round(
    monkeypatch,
):
    """
    Review每执行一轮都应追加新事件，后一次不能覆盖前一次。
    """
    fake_llm = FakeStructuredLLM(
        [
            StoryCriticResult(
                passed=False,
                issues=["冲突不清"],
                suggestions=[],
            ),
            StoryCriticResult(
                passed=True,
                issues=[],
                suggestions=[],
            ),
        ]
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    first = review_story_module.review_story(
        _story_review_state()
    )
    second = review_story_module.review_story(
        _story_review_state(
            first[
                "story_review_history"
            ]
        )
    )
    llm_call_trace = (
        first["llm_call_trace"]
        + second["llm_call_trace"]
    )

    assert len(llm_call_trace) == 2
    assert [
        event["prompt_name"]
        for event in llm_call_trace
    ] == [
        "review.story",
        "review.story",
    ]
    assert all(
        event["node"] == "review_story"
        for event in llm_call_trace
    )


def _memory_candidate(
):
    return extract_memory_module.MemoryCandidate(
        field="disliked_elements_to_add",
        value="大团圆结局",
        source="user_idea",
        evidence="我讨厌大团圆结局",
        claim_type="explicit_preference",
    )


def test_memory_verifier_is_traced_only_when_invoked(
    monkeypatch,
):
    """
    无有效候选时只调用candidate；有有效候选时再记录verifier。
    """
    monkeypatch.setattr(
        update_memory_module,
        "save_user_memory",
        lambda memory: None,
    )

    empty_candidate_llm = FakeStructuredLLM(
        [
            extract_memory_module.MemoryCandidateBatch(
                candidates=[],
            ),
        ]
    )
    verifier_not_called = FakeStructuredLLM(
        error=AssertionError(
            "没有有效候选时不应调用Verifier"
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_candidate_llm",
        empty_candidate_llm,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_verifier_llm",
        verifier_not_called,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        None,
    )

    skipped_result = update_memory_module.update_memory(
        {
            "user_id": "memory_trace_user",
            "user_idea": "这次生成一个校园故事",
            "user_memory": UserMemory(
                user_id="memory_trace_user"
            ),
        }
    )

    assert [
        event["prompt_name"]
        for event in skipped_result["llm_call_trace"]
    ] == [
        "memory.candidate_extraction",
    ]

    candidate = _memory_candidate()
    candidate_llm = FakeStructuredLLM(
        [
            extract_memory_module.MemoryCandidateBatch(
                candidates=[candidate],
            ),
        ]
    )
    verifier_llm = FakeStructuredLLM(
        [
            extract_memory_module.MemoryCandidateVerification(
                decisions=[
                    extract_memory_module.MemoryCandidateDecisionItem(
                        **candidate.model_dump(
                            mode="json",
                        ),
                        decision="ACCEPT",
                    )
                ]
            ),
        ]
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_candidate_llm",
        candidate_llm,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_verifier_llm",
        verifier_llm,
    )

    saved_result = update_memory_module.update_memory(
        {
            "user_id": "memory_trace_user",
            "user_idea": "我讨厌大团圆结局。",
            "user_memory": UserMemory(
                user_id="memory_trace_user"
            ),
        }
    )

    assert [
        event["prompt_name"]
        for event in saved_result["llm_call_trace"]
    ] == [
        "memory.candidate_extraction",
        "memory.conservative_verifier",
    ]


def test_failed_call_keeps_safe_event_and_enters_summary():
    """
    失败调用保持原异常类型；事件和Summary只记录error_type等安全元数据。
    """
    rendered = render_prompt(
        "generation.design_characters",
        version="v1",
        user_idea="SENSITIVE_USER_IDEA",
        genre="SENSITIVE_GENRE",
        core_theme="SENSITIVE_THEME",
        visual_style="SENSITIVE_STYLE",
    )
    original_error = RuntimeError(
        "provider request failed"
    )
    fake_llm = FakeStructuredLLM(
        error=original_error
    )

    with pytest.raises(
        RuntimeError,
    ) as error_info:
        with collect_llm_call_trace():
            invoke_structured_llm(
                fake_llm,
                rendered,
                node="design_characters",
            )

    assert error_info.value is original_error
    event = get_failed_llm_call_event(
        error_info.value
    )
    assert event is not None
    assert event["status"] == "failed"
    assert event["error_type"] == "RuntimeError"
    assert "SENSITIVE_" not in str(event)
    assert "provider request failed" not in str(event)

    summary = build_execution_summary(
        {
            "execution_id": "exec_failed_llm",
            "execution_trace": [],
        },
        "failed",
        error=error_info.value,
    )

    assert summary.llm_call_count == 1
    assert summary.successful_llm_call_count == 0
    assert summary.failed_llm_call_count == 1
    assert summary.failed_node == "design_characters"


def test_execution_summary_aggregates_llm_trace_without_copying_details():
    """
    Summary聚合调用次数和版本使用情况，但模型中不包含详细trace字段。
    """
    state = {
        "execution_id": "exec_llm_summary",
        "llm_call_trace": [
            {
                "node": "review_story",
                "prompt_name": "review.story",
                "prompt_version": "v1",
                "prompt_chars": 100,
                "llm_profile": "strong",
                "model_name": "qwen-plus",
                "temperature": 0,
                "status": "success",
                "duration_ms": 12.5,
                "error_type": None,
            },
            {
                "node": "review_story",
                "prompt_name": "review.story",
                "prompt_version": "v1",
                "prompt_chars": 110,
                "llm_profile": "strong",
                "model_name": "qwen-plus",
                "temperature": 0,
                "status": "success",
                "duration_ms": 7.5,
                "error_type": None,
            },
            {
                "node": "update_memory",
                "prompt_name": "memory.conservative_verifier",
                "prompt_version": "v2",
                "prompt_chars": 80,
                "llm_profile": "critical",
                "model_name": "qwen-plus",
                "temperature": 0,
                "status": "failed",
                "duration_ms": -1,
                "error_type": "TimeoutError",
            },
        ],
    }

    summary = build_execution_summary(
        state,
        "failed",
    )
    payload = summary.model_dump()

    assert summary.llm_call_count == 3
    assert summary.successful_llm_call_count == 2
    assert summary.failed_llm_call_count == 1
    assert summary.llm_active_duration_ms == 20.0
    assert summary.prompt_versions == {
        "review.story": ["v1"],
        "memory.conservative_verifier": ["v2"],
    }
    assert summary.profile_usage == {
        "strong": 2,
        "critical": 1,
    }
    assert summary.model_usage == {
        "qwen-plus": 3,
    }
    assert "llm_call_trace" not in payload


def test_hitl_resume_keeps_llm_trace_from_before_and_after_pause():
    """
    同一thread_id恢复后，Checkpointer中的append reducer应保留暂停前后两次调用。
    """
    fake_llm = FakeStructuredLLM(
        results=[
            object(),
            object(),
        ]
    )

    def call_llm(
        state: FilmState,
    ) -> dict:
        rendered = render_prompt(
            "generation.design_characters",
            version="v1",
            user_idea="测试输入",
            genre="校园",
            core_theme="成长",
            visual_style="克制",
        )

        with collect_llm_call_trace() as events:
            invoke_structured_llm(
                fake_llm,
                rendered,
                node=state["current_stage"],
            )

        return {
            "llm_call_trace": events,
        }

    def before_pause(
        state: FilmState,
    ) -> dict:
        state_with_node = {
            **state,
            "current_stage": "before_pause",
        }
        return {
            **call_llm(state_with_node),
            "current_stage": "before_pause_completed",
        }

    def pause_for_human(
        state: FilmState,
    ) -> dict:
        interrupt(
            {
                "type": "test_review_required",
            }
        )
        return {
            "current_stage": "human_review_completed",
        }

    def after_resume(
        state: FilmState,
    ) -> dict:
        state_with_node = {
            **state,
            "current_stage": "after_resume",
        }
        return {
            **call_llm(state_with_node),
            "current_stage": "completed",
        }

    builder = StateGraph(
        FilmState
    )
    builder.add_node(
        "before_pause",
        before_pause,
    )
    builder.add_node(
        "pause_for_human",
        pause_for_human,
    )
    builder.add_node(
        "after_resume",
        after_resume,
    )
    builder.add_edge(
        START,
        "before_pause",
    )
    builder.add_edge(
        "before_pause",
        "pause_for_human",
    )
    builder.add_edge(
        "pause_for_human",
        "after_resume",
    )
    builder.add_edge(
        "after_resume",
        END,
    )
    graph = builder.compile(
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {
            "thread_id": "exec_llm_hitl",
        }
    }

    list(
        graph.stream(
            {
                "execution_id": "exec_llm_hitl",
                "current_stage": "initialized",
                "llm_call_trace": [],
            },
            config=config,
            stream_mode="updates",
        )
    )
    paused_state = graph.get_state(
        config
    )

    assert paused_state.next == (
        "pause_for_human",
    )
    assert len(
        paused_state.values[
            "llm_call_trace"
        ]
    ) == 1

    list(
        graph.stream(
            Command(
                resume={
                    "decision": "approve",
                }
            ),
            config=config,
            stream_mode="updates",
        )
    )
    completed_state = graph.get_state(
        config
    )
    trace = completed_state.values[
        "llm_call_trace"
    ]

    assert len(trace) == 2
    assert [
        event["node"]
        for event in trace
    ] == [
        "before_pause",
        "after_resume",
    ]
