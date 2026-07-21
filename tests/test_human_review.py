import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import film_graph, film_hitl_graph, route_after_human_review, trace_node
from reviews.human_review_story import human_review_story
from schemas import Character, FilmBrief, StoryOutline, StoryReviewResult
from state import FilmState


def _base_state(
    execution_id: str,
) -> FilmState:
    """
    构造人工故事审核节点所需的最小State。
    """
    return {
        "user_id": "hitl_test_user",
        "execution_id": execution_id,
        "user_idea": "生成一个校园故事",
        "film_brief": FilmBrief(
            target_duration_sec=60,
            genre="校园",
            core_theme="成长",
            visual_style="自然克制",
            recommended_scene_count=5,
        ),
        "characters": [
            Character(
                name="林夏",
                role="主角",
                appearance="白衬衫",
                personality=["克制"],
                motivation="面对告别",
                continuity_constraints=[],
            )
        ],
        "story_outline": StoryOutline(
            setup="毕业前夕，林夏准备告别校园。",
            conflict="她想留下却必须离开。",
            turning_point="她意识到告别也是成长。",
            ending="她平静离开校园。",
            theme="克制的成长。",
        ),
        "story_review_result": StoryReviewResult(
            passed=True,
            issues=[],
            suggestions=[],
        ),
        "story_revision_count": 0,
        "scene_revision_count": 0,
        "execution_trace": [],
        "current_stage": "story_review_completed",
    }


def _build_test_human_graph():
    """
    构造只用于测试HITL暂停/恢复的小图，避免调用真实LLM节点。
    """

    def fake_write_scenes(
        state: FilmState,
    ) -> dict:
        return {
            "current_stage": "fake_write_scenes_completed",
        }

    def fake_revise_story(
        state: FilmState,
    ) -> dict:
        assert state["human_feedback"]
        return {
            "story_revision_count": state.get(
                "story_revision_count",
                0,
            ) + 1,
            "current_stage": "fake_story_revised",
        }

    builder = StateGraph(FilmState)

    builder.add_node(
        "human_review_story",
        trace_node("human_review_story", human_review_story),
    )
    builder.add_node(
        "write_scenes",
        trace_node("write_scenes", fake_write_scenes),
    )
    builder.add_node(
        "revise_story",
        trace_node("revise_story", fake_revise_story),
    )

    builder.add_edge(
        START,
        "human_review_story",
    )
    builder.add_conditional_edges(
        "human_review_story",
        route_after_human_review,
        {
            "write_scenes": "write_scenes",
            "revise_story": "revise_story",
        },
    )
    builder.add_edge(
        "write_scenes",
        END,
    )
    builder.add_edge(
        "revise_story",
        END,
    )

    return builder.compile(
        checkpointer=InMemorySaver(),
    )


def _config(
    execution_id: str,
) -> dict:
    """
    HITL恢复必须复用同一个thread_id。
    """
    return {
        "configurable": {
            "thread_id": execution_id,
        },
    }


def test_film_graph_keeps_auto_flow_without_human_review_node():
    """
    原自动Graph不应被插入human_review_story节点。
    """
    node_names = set(
        film_graph.get_graph().nodes.keys()
    )

    assert "human_review_story" not in node_names
    assert "write_scenes" in node_names


def test_film_hitl_graph_contains_human_review_node():
    """
    HITL独立Graph应包含人工故事审核节点。
    """
    node_names = set(
        film_hitl_graph.get_graph().nodes.keys()
    )

    assert "human_review_story" in node_names


def test_human_review_interrupts_before_write_scenes():
    """
    首次执行应暂停在人工审核点，且尚未进入write_scenes。
    """
    graph = _build_test_human_graph()
    execution_id = "exec_hitl_interrupt"

    events = list(
        graph.stream(
            _base_state(execution_id),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )
    node_names = {
        node_name
        for event in events
        for node_name in event.keys()
    }
    graph_state = graph.get_state(
        _config(execution_id)
    )

    assert "__interrupt__" in node_names
    assert "write_scenes" not in node_names
    assert graph_state.next == (
        "human_review_story",
    )


def test_human_review_approve_resumes_to_write_scenes():
    """
    approve恢复后，应继续进入测试用write_scenes节点。
    """
    graph = _build_test_human_graph()
    execution_id = "exec_hitl_approve"

    list(
        graph.stream(
            _base_state(execution_id),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )

    events = list(
        graph.stream(
            Command(
                resume={
                    "decision": "approve",
                    "feedback": None,
                }
            ),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )
    node_names = [
        node_name
        for event in events
        for node_name in event.keys()
    ]

    assert "human_review_story" in node_names
    assert "write_scenes" in node_names

    graph_state = graph.get_state(
        _config(execution_id)
    )
    feedback_history = graph_state.values[
        "human_feedback_history"
    ]

    assert feedback_history == [
        {
            "scope": "story",
            "decision": "approve",
            "feedback": None,
        }
    ]


def test_human_review_revise_resumes_to_revise_story():
    """
    revise且feedback非空时，应进入测试用故事修订节点。
    """
    graph = _build_test_human_graph()
    execution_id = "exec_hitl_revise"

    list(
        graph.stream(
            _base_state(execution_id),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )

    events = list(
        graph.stream(
            Command(
                resume={
                    "decision": "revise",
                    "feedback": "结尾改成更加克制的开放式结局。",
                }
            ),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )
    node_names = [
        node_name
        for event in events
        for node_name in event.keys()
    ]

    assert "human_review_story" in node_names
    assert "revise_story" in node_names

    graph_state = graph.get_state(
        _config(execution_id)
    )
    feedback_history = graph_state.values[
        "human_feedback_history"
    ]

    assert feedback_history == [
        {
            "scope": "story",
            "decision": "revise",
            "feedback": "结尾改成更加克制的开放式结局。",
        }
    ]


def test_human_review_history_accumulates_multiple_revisions():
    """
    连续两轮revise后，应保留两次人工反馈历史。
    """
    graph = _build_test_human_graph()
    execution_id = "exec_hitl_two_revisions"

    list(
        graph.stream(
            _base_state(execution_id),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )

    list(
        graph.stream(
            Command(
                resume={
                    "decision": "revise",
                    "feedback": "结尾改成更加克制的开放式结局。",
                }
            ),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )

    # 测试小图的revise_story直接结束；再次从人工节点启动，用同一thread_id模拟第二轮人工反馈。
    list(
        graph.stream(
            _base_state(execution_id),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )

    list(
        graph.stream(
            Command(
                resume={
                    "decision": "revise",
                    "feedback": "把开端改得更安静。",
                }
            ),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )

    graph_state = graph.get_state(
        _config(execution_id)
    )
    feedback_history = graph_state.values[
        "human_feedback_history"
    ]

    assert len(feedback_history) == 2
    assert feedback_history[0] == {
        "scope": "story",
        "decision": "revise",
        "feedback": "结尾改成更加克制的开放式结局。",
    }
    assert feedback_history[1] == {
        "scope": "story",
        "decision": "revise",
        "feedback": "把开端改得更安静。",
    }


def test_human_review_revise_requires_feedback():
    """
    revise但feedback为空时，应校验失败。
    """
    graph = _build_test_human_graph()
    execution_id = "exec_hitl_empty_feedback"

    list(
        graph.stream(
            _base_state(execution_id),
            config=_config(execution_id),
            stream_mode="updates",
        )
    )

    with pytest.raises(ValueError):
        list(
            graph.stream(
                Command(
                    resume={
                        "decision": "revise",
                        "feedback": "   ",
                    }
                ),
                config=_config(execution_id),
                stream_mode="updates",
            )
        )
