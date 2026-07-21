import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as api_main


client = TestClient(api_main.app)


class FakeInterrupt:
    """
    模拟LangGraph Interrupt对象，只暴露value给API提取payload。
    """

    def __init__(
        self,
        value: dict,
    ) -> None:
        self.value = value


class FakeGraphState:
    """
    模拟LangGraph checkpoint状态。
    """

    def __init__(
        self,
        values: dict | None = None,
        next_nodes: tuple[str, ...] = (),
    ) -> None:
        self.values = values or {}
        self.next = next_nodes


def _parse_sse_events(
    response,
) -> list[dict]:
    """
    解析SSE响应，返回event名称和JSON data。
    """
    raw_text = response.text
    chunks = [
        chunk
        for chunk in raw_text.split("\n\n")
        if chunk.strip()
    ]
    events = []

    for chunk in chunks:
        lines = chunk.splitlines()
        event_line = next(
            line
            for line in lines
            if line.startswith("event: ")
        )
        data_line = next(
            line
            for line in lines
            if line.startswith("data: ")
        )
        events.append(
            {
                "event": event_line.removeprefix("event: "),
                "data": json.loads(
                    data_line.removeprefix("data: ")
                ),
            }
        )

    return events


def _event_names(
    events: list[dict],
) -> list[str]:
    """
    提取SSE事件名称列表，便于断言事件顺序。
    """
    return [
        event["event"]
        for event in events
    ]


def _trace(
    execution_id: str,
    node: str,
) -> dict:
    """
    构造测试用TraceEvent。
    """
    return {
        "execution_id": execution_id,
        "node": node,
        "status": "success",
        "stage": f"{node}_completed",
        "duration_ms": 1.0,
    }


def _review_payload(
    execution_id: str,
) -> dict:
    """
    构造测试用人工审核payload。
    """
    return {
        "type": "story_review_required",
        "execution_id": execution_id,
        "message": "故事大纲已完成，请确认是否继续生成分场。",
    }


def _assert_all_execution_ids(
    events: list[dict],
    execution_id: str,
) -> None:
    """
    校验所有事件里的execution_id保持一致。
    """
    for event in events:
        assert event["data"]["execution_id"] == execution_id


def test_hitl_stream_start_waiting_for_human(monkeypatch):
    """
    流式Start遇到interrupt时，应发送human_review_required后结束。
    """

    class FakeHitlGraph:
        def stream(self, graph_input, config=None, stream_mode=None):
            execution_id = graph_input["execution_id"]
            assert config["configurable"]["thread_id"] == execution_id
            assert stream_mode == "updates"

            yield {
                "review_story": {
                    "current_stage": "story_review_completed",
                    "execution_trace": [
                        _trace(execution_id, "review_story")
                    ],
                }
            }
            yield {
                "__interrupt__": [
                    FakeInterrupt(
                        _review_payload(execution_id)
                    )
                ]
            }

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        "/api/v1/films/hitl/stream/start",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )
    events = _parse_sse_events(response)
    execution_id = events[0]["data"]["execution_id"]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _event_names(events) == [
        "started",
        "node_completed",
        "human_review_required",
    ]
    assert "completed" not in _event_names(events)
    _assert_all_execution_ids(
        events,
        execution_id,
    )
    assert events[-1]["data"]["status"] == "waiting_for_human"
    assert events[-1]["data"]["review_payload"]["type"] == "story_review_required"


def test_hitl_stream_start_completed(monkeypatch):
    """
    流式Start如果未暂停并完成，应发送completed和final_output。
    """

    class FakeHitlGraph:
        def stream(self, graph_input, config=None, stream_mode=None):
            execution_id = graph_input["execution_id"]
            assert config["configurable"]["thread_id"] == execution_id
            assert stream_mode == "updates"

            yield {
                "review_story": {
                    "current_stage": "story_review_completed",
                    "execution_trace": [
                        _trace(execution_id, "review_story")
                    ],
                }
            }
            yield {
                "finalize": {
                    "current_stage": "finalized",
                    "final_output": {
                        "user_idea": graph_input["user_idea"],
                    },
                    "execution_trace": [
                        _trace(execution_id, "finalize")
                    ],
                }
            }

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        "/api/v1/films/hitl/stream/start",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )
    events = _parse_sse_events(response)

    assert _event_names(events) == [
        "started",
        "node_completed",
        "node_completed",
        "completed",
    ]
    assert events[-1]["data"]["final_output"] == {
        "user_idea": "生成一个校园故事",
    }


def test_hitl_stream_start_error(monkeypatch):
    """
    流式Start异常时，应发送error且不暴露内部异常。
    """

    class FakeHitlGraph:
        def stream(self, graph_input, config=None, stream_mode=None):
            raise RuntimeError(
                "internal fake graph error"
            )

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        "/api/v1/films/hitl/stream/start",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )
    events = _parse_sse_events(response)
    raw_text = response.text

    assert _event_names(events) == [
        "started",
        "error",
    ]
    assert "completed" not in _event_names(events)
    assert "internal fake graph error" not in raw_text
    assert "Traceback" not in raw_text


def test_hitl_stream_resume_approve_completed(monkeypatch):
    """
    流式Resume approve后，应继续执行到completed。
    """
    execution_id = "exec_stream_approve"

    class FakeHitlGraph:
        def get_state(self, config):
            return FakeGraphState(
                values={
                    "current_stage": "story_review_completed",
                },
                next_nodes=("human_review_story",),
            )

        def stream(self, graph_input, config=None, stream_mode=None):
            assert config["configurable"]["thread_id"] == execution_id
            assert graph_input.resume["decision"] == "approve"
            assert stream_mode == "updates"

            yield {
                "human_review_story": {
                    "current_stage": "human_story_review_completed",
                    "execution_trace": [
                        _trace(execution_id, "human_review_story")
                    ],
                }
            }

        def get_state(self, config):  # noqa: F811
            return FakeGraphState(
                values={
                    "current_stage": "memory_updated",
                    "final_output": {
                        "user_idea": "生成一个校园故事",
                    },
                    "execution_trace": [
                        _trace(execution_id, "human_review_story")
                    ],
                    "memory_update_status": "skipped",
                },
                next_nodes=("human_review_story",),
            )

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        f"/api/v1/films/hitl/stream/{execution_id}/resume",
        json={
            "decision": "approve",
            "feedback": None,
        },
    )
    events = _parse_sse_events(response)

    assert _event_names(events) == [
        "resumed",
        "node_completed",
        "completed",
    ]
    assert events[0]["data"]["execution_id"] == execution_id
    assert events[0]["data"]["decision"] == "approve"
    assert events[-1]["data"]["final_output"] == {
        "user_idea": "生成一个校园故事",
    }


def test_hitl_stream_resume_revise_waiting_again(monkeypatch):
    """
    流式Resume revise后可能再次暂停并返回新的审核payload。
    """
    execution_id = "exec_stream_revise"

    class FakeHitlGraph:
        def get_state(self, config):
            return FakeGraphState(
                values={
                    "current_stage": "story_review_completed",
                },
                next_nodes=("human_review_story",),
            )

        def stream(self, graph_input, config=None, stream_mode=None):
            assert config["configurable"]["thread_id"] == execution_id
            assert graph_input.resume["decision"] == "revise"
            assert graph_input.resume["feedback"] == "结尾更克制。"
            assert stream_mode == "updates"

            yield {
                "revise_story": {
                    "current_stage": "story_revised_completed",
                    "execution_trace": [
                        _trace(execution_id, "revise_story")
                    ],
                }
            }
            yield {
                "__interrupt__": [
                    FakeInterrupt(
                        _review_payload(execution_id)
                    )
                ]
            }

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        f"/api/v1/films/hitl/stream/{execution_id}/resume",
        json={
            "decision": "revise",
            "feedback": "结尾更克制。",
        },
    )
    events = _parse_sse_events(response)

    assert _event_names(events) == [
        "resumed",
        "node_completed",
        "human_review_required",
    ]
    assert "completed" not in _event_names(events)
    assert events[-1]["data"]["review_payload"]["execution_id"] == execution_id


def test_hitl_stream_resume_missing_checkpoint_returns_404(monkeypatch):
    """
    checkpoint不存在时，应在SSE开始前返回404。
    """

    class FakeHitlGraph:
        def get_state(self, config):
            return FakeGraphState()

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        "/api/v1/films/hitl/stream/exec_missing/resume",
        json={
            "decision": "approve",
            "feedback": None,
        },
    )

    assert response.status_code == 404


def test_hitl_stream_resume_without_pending_interrupt_returns_409(monkeypatch):
    """
    checkpoint存在但没有待处理human_review_story时，应返回409。
    """

    class FakeHitlGraph:
        def get_state(self, config):
            return FakeGraphState(
                values={
                    "current_stage": "memory_updated",
                },
                next_nodes=(),
            )

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        "/api/v1/films/hitl/stream/exec_done/resume",
        json={
            "decision": "approve",
            "feedback": None,
        },
    )

    assert response.status_code == 409


def test_hitl_stream_resume_revise_empty_feedback_returns_422():
    """
    revise但feedback为空时，请求校验应返回422。
    """

    response = client.post(
        "/api/v1/films/hitl/stream/exec_empty_feedback/resume",
        json={
            "decision": "revise",
            "feedback": "   ",
        },
    )

    assert response.status_code == 422


def test_hitl_stream_content_type(monkeypatch):
    """
    HITL SSE响应Content-Type应为text/event-stream。
    """

    class FakeHitlGraph:
        def stream(self, graph_input, config=None, stream_mode=None):
            execution_id = graph_input["execution_id"]
            yield {
                "__interrupt__": [
                    FakeInterrupt(
                        _review_payload(execution_id)
                    )
                ]
            }

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        "/api/v1/films/hitl/stream/start",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )

    assert response.headers["content-type"].startswith("text/event-stream")
