import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as api_main


client = TestClient(api_main.app)


class FakeInterrupt:
    """
    模拟LangGraph Interrupt对象。
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


def _waiting_payload(
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


def test_hitl_start_waiting_for_human(monkeypatch):
    """
    start接口遇到interrupt时，应返回waiting_for_human。
    """

    class FakeHitlGraph:
        def stream(self, graph_input, config=None, stream_mode=None):
            execution_id = graph_input["execution_id"]
            assert config == {
                "configurable": {
                    "thread_id": execution_id,
                },
            }
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
                        _waiting_payload(execution_id)
                    )
                ]
            }

        def get_state(self, config):
            execution_id = config["configurable"]["thread_id"]
            return FakeGraphState(
                values={
                    "current_stage": "story_review_completed",
                    "execution_trace": [
                        _trace(execution_id, "review_story")
                    ],
                },
                next_nodes=("human_review_story",),
            )

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        "/api/v1/films/hitl/start",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["execution_id"].startswith("exec_")
    assert data["status"] == "waiting_for_human"
    assert data["current_stage"] == "story_review_completed"
    assert data["review_payload"]["type"] == "story_review_required"
    assert data["execution_trace"][0]["node"] == "review_story"
    assert data["execution_summary"]["status"] == "waiting_for_human"
    assert data["execution_summary"]["execution_id"] == data["execution_id"]
    assert data["execution_summary"]["trace_event_count"] == 1
    assert data["execution_summary"]["node_execution_counts"] == {
        "review_story": 1,
    }


def test_hitl_resume_approve_completed(monkeypatch):
    """
    approve恢复后，应返回completed和final_output。
    """

    execution_id = "exec_hitl_approve"

    class FakeHitlGraph:
        def get_state(self, config):
            return FakeGraphState(
                values={
                    "current_stage": "story_review_completed",
                    "execution_trace": [
                        _trace(execution_id, "review_story")
                    ],
                },
                next_nodes=("human_review_story",),
            )

        def stream(self, graph_input, config=None, stream_mode=None):
            assert config["configurable"]["thread_id"] == execution_id
            assert stream_mode == "updates"

            yield {
                "human_review_story": {
                    "current_stage": "human_story_review_completed",
                    "execution_trace": [
                        _trace(execution_id, "human_review_story")
                    ],
                }
            }
            yield {
                "finalize": {
                    "current_stage": "finalized",
                    "final_output": {
                        "user_idea": "生成一个校园故事",
                    },
                    "execution_trace": [
                        _trace(execution_id, "finalize")
                    ],
                }
            }
            yield {
                "update_memory": {
                    "current_stage": "memory_updated",
                    "memory_update_status": "skipped",
                    "execution_trace": [
                        _trace(execution_id, "update_memory")
                    ],
                }
            }

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        f"/api/v1/films/hitl/{execution_id}/resume",
        json={
            "decision": "approve",
            "feedback": None,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["execution_id"] == execution_id
    assert data["status"] == "completed"
    assert data["current_stage"] == "memory_updated"
    assert data["final_output"] == {
        "user_idea": "生成一个校园故事",
    }
    assert data["memory_update_status"] == "skipped"
    assert data["execution_summary"]["status"] == "completed"
    assert data["execution_summary"]["execution_id"] == execution_id
    assert data["execution_summary"]["memory_update_status"] == "skipped"
    assert [
        event["node"]
        for event in data["execution_trace"]
    ] == [
        "review_story",
        "human_review_story",
        "finalize",
        "update_memory",
    ]
    assert (
        len(data["execution_trace"])
        == data["execution_summary"]["trace_event_count"]
    )


def test_hitl_resume_revise_waiting_again(monkeypatch):
    """
    revise恢复后可能再次进入人工审核等待。
    """

    execution_id = "exec_hitl_revise"

    class FakeHitlGraph:
        def __init__(self):
            self.get_state_count = 0

        def get_state(self, config):
            self.get_state_count += 1

            if self.get_state_count == 1:
                return FakeGraphState(
                    values={
                        "current_stage": "story_review_completed",
                        "execution_trace": [
                            _trace(execution_id, "review_story")
                        ],
                    },
                    next_nodes=("human_review_story",),
                )

            return FakeGraphState(
                values={
                    "current_stage": "story_review_completed",
                    "execution_trace": [
                        _trace(execution_id, "review_story"),
                        _trace(execution_id, "revise_story")
                    ],
                },
                next_nodes=("human_review_story",),
            )

        def stream(self, graph_input, config=None, stream_mode=None):
            assert config["configurable"]["thread_id"] == execution_id
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
                        _waiting_payload(execution_id)
                    )
                ]
            }

    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        FakeHitlGraph(),
    )

    response = client.post(
        f"/api/v1/films/hitl/{execution_id}/resume",
        json={
            "decision": "revise",
            "feedback": "结尾更克制。",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "waiting_for_human"
    assert data["review_payload"]["execution_id"] == execution_id
    assert [
        event["node"]
        for event in data["execution_trace"]
    ] == [
        "review_story",
        "revise_story",
    ]
    assert data["execution_summary"]["status"] == "waiting_for_human"
    assert data["execution_summary"]["execution_id"] == execution_id
    assert data["execution_summary"]["node_execution_counts"] == {
        "review_story": 1,
        "revise_story": 1,
    }
    assert (
        len(data["execution_trace"])
        == data["execution_summary"]["trace_event_count"]
    )


def test_hitl_start_then_resume_returns_complete_trace(
    monkeypatch,
):
    """
    同一execution_id从Start暂停再Resume后，响应应包含暂停前后的完整有序Trace。
    """

    class StatefulFakeHitlGraph:
        def __init__(self):
            self.execution_id = None
            self.values = {}
            self.next = ()

        def stream(
            self,
            graph_input,
            config=None,
            stream_mode=None,
        ):
            assert stream_mode == "updates"

            if isinstance(graph_input, dict):
                self.execution_id = graph_input[
                    "execution_id"
                ]
                assert config["configurable"][
                    "thread_id"
                ] == self.execution_id

                for node_name in (
                    "retrieve_memory",
                    "review_story",
                ):
                    trace_event = _trace(
                        self.execution_id,
                        node_name,
                    )
                    self.values.setdefault(
                        "execution_trace",
                        [],
                    ).append(
                        trace_event
                    )
                    self.values[
                        "current_stage"
                    ] = trace_event["stage"]
                    yield {
                        node_name: {
                            "current_stage": trace_event[
                                "stage"
                            ],
                            "execution_trace": [
                                trace_event
                            ],
                        }
                    }

                self.next = (
                    "human_review_story",
                )
                yield {
                    "__interrupt__": [
                        FakeInterrupt(
                            _waiting_payload(
                                self.execution_id
                            )
                        )
                    ]
                }
                return

            assert config["configurable"][
                "thread_id"
            ] == self.execution_id
            self.next = ()

            for node_name in (
                "human_review_story",
                "write_scenes",
                "finalize",
                "update_memory",
            ):
                trace_event = _trace(
                    self.execution_id,
                    node_name,
                )
                self.values[
                    "execution_trace"
                ].append(
                    trace_event
                )
                self.values[
                    "current_stage"
                ] = trace_event["stage"]
                node_update = {
                    "current_stage": trace_event[
                        "stage"
                    ],
                    "execution_trace": [
                        trace_event
                    ],
                }

                if node_name == "finalize":
                    self.values["final_output"] = {
                        "user_idea": "测试故事",
                    }
                    node_update["final_output"] = (
                        self.values[
                            "final_output"
                        ]
                    )

                if node_name == "update_memory":
                    self.values[
                        "memory_update_status"
                    ] = "skipped"
                    node_update[
                        "memory_update_status"
                    ] = "skipped"

                yield {
                    node_name: node_update
                }

        def get_state(
            self,
            config,
        ):
            return FakeGraphState(
                values={
                    **self.values,
                    "execution_trace": list(
                        self.values.get(
                            "execution_trace",
                            [],
                        )
                    ),
                },
                next_nodes=self.next,
            )

    fake_graph = StatefulFakeHitlGraph()
    monkeypatch.setattr(
        api_main,
        "film_hitl_graph",
        fake_graph,
    )

    start_response = client.post(
        "/api/v1/films/hitl/start",
        json={
            "user_id": "trace_test_user",
            "user_idea": "生成一个测试故事",
        },
    )
    start_data = start_response.json()
    execution_id = start_data[
        "execution_id"
    ]

    assert start_response.status_code == 200
    assert start_data["status"] == "waiting_for_human"
    assert [
        event["node"]
        for event in start_data["execution_trace"]
    ] == [
        "retrieve_memory",
        "review_story",
    ]

    resume_response = client.post(
        f"/api/v1/films/hitl/{execution_id}/resume",
        json={
            "decision": "approve",
            "feedback": None,
        },
    )
    resume_data = resume_response.json()
    trace_nodes = [
        event["node"]
        for event in resume_data[
            "execution_trace"
        ]
    ]

    assert resume_response.status_code == 200
    assert resume_data["status"] == "completed"
    assert resume_data["execution_id"] == execution_id
    assert trace_nodes == [
        "retrieve_memory",
        "review_story",
        "human_review_story",
        "write_scenes",
        "finalize",
        "update_memory",
    ]
    assert len(trace_nodes) == len(
        set(trace_nodes)
    )
    assert (
        len(resume_data["execution_trace"])
        == resume_data[
            "execution_summary"
        ][
            "trace_event_count"
        ]
    )


def test_hitl_resume_revise_empty_feedback_returns_422():
    """
    revise时feedback为空应由请求模型返回422。
    """

    response = client.post(
        "/api/v1/films/hitl/exec_empty_feedback/resume",
        json={
            "decision": "revise",
            "feedback": "   ",
        },
    )

    assert response.status_code == 422


def test_hitl_resume_missing_checkpoint_returns_404(monkeypatch):
    """
    execution_id没有checkpoint时，应返回404。
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
        "/api/v1/films/hitl/exec_missing/resume",
        json={
            "decision": "approve",
            "feedback": None,
        },
    )

    assert response.status_code == 404


def test_hitl_resume_without_pending_interrupt_returns_409(monkeypatch):
    """
    checkpoint存在但没有待处理interrupt时，应返回409。
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
        "/api/v1/films/hitl/exec_done/resume",
        json={
            "decision": "approve",
            "feedback": None,
        },
    )

    assert response.status_code == 409


def test_hitl_graph_exception_returns_500(monkeypatch):
    """
    Graph内部异常应返回500，且不暴露原始异常。
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
        "/api/v1/films/hitl/start",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )
    data = response.json()

    assert response.status_code == 500
    assert data["detail"]["execution_id"].startswith("exec_")
    assert data["detail"]["message"] == "HITL Graph执行失败。"
    assert data["detail"]["execution_summary"]["status"] == "failed"
    assert data["detail"]["execution_summary"]["error_type"] == "RuntimeError"
    assert data["detail"]["execution_summary"]["error_message"] == "HITL Graph执行失败。"
    assert "internal fake graph error" not in str(data)
    assert "Traceback" not in str(data)
