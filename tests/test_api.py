import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as api_main


app = api_main.app

client = TestClient(app)


def _llm_call_event(
    node: str = "analyze_brief",
) -> dict:
    """
    构造API Summary测试用的最小LLM调用事件。
    """
    return {
        "node": node,
        "prompt_name": "generation.analyze_brief",
        "prompt_version": "v1",
        "prompt_chars": 128,
        "llm_profile": "fast",
        "model_name": "qwen-plus",
        "temperature": 0,
        "status": "success",
        "duration_ms": 4.5,
        "error_type": None,
    }


def _parse_sse_events(
    response,
) -> list[dict]:
    """
    解析SSE事件，避免只靠字符串包含关系判断结构。
    """
    chunks = [
        chunk
        for chunk in response.text.split("\n\n")
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


def test_health_check():
    """
    健康检查接口应返回服务可用状态。
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_generate_film_success(monkeypatch):
    """
    非流式生成接口应把请求转为FilmState，并返回Graph结果。
    """

    def fake_invoke(initial_state, config=None):
        """
        模拟Graph执行，避免测试调用真实LLM。
        """
        assert initial_state["user_id"] == "test_user_001"
        assert initial_state["user_idea"] == "生成一个校园故事"
        assert initial_state["execution_id"].startswith("exec_")
        assert config == {
            "configurable": {
                "thread_id": initial_state["execution_id"],
            },
        }

        return {
            "execution_id": initial_state["execution_id"],
            "current_stage": "memory_updated",
            "final_output": {
                "user_idea": initial_state["user_idea"],
            },
            "execution_trace": [
                {
                    "execution_id": initial_state["execution_id"],
                    "node": "retrieve_memory",
                    "status": "success",
                    "stage": "memory_retrieved",
                    "duration_ms": 1.23,
                }
            ],
            "llm_call_trace": [
                _llm_call_event(),
            ],
            "memory_update_status": "skipped",
        }

    monkeypatch.setattr(
        api_main.film_graph,
        "invoke",
        fake_invoke,
    )

    response = client.post(
        "/api/v1/films/generate",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )

    response_data = response.json()

    assert response.status_code == 200
    assert response_data["execution_id"].startswith("exec_")
    assert response_data["status"] == "completed"
    assert response_data["current_stage"] == "memory_updated"
    assert response_data["final_output"] == {
        "user_idea": "生成一个校园故事",
    }
    assert (
        response_data["execution_trace"][0]["execution_id"]
        == response_data["execution_id"]
    )
    assert response_data["memory_update_status"] == "skipped"
    assert response_data["execution_summary"]["status"] == "completed"
    assert response_data["execution_summary"]["execution_id"] == response_data["execution_id"]
    assert response_data["execution_summary"]["trace_event_count"] == 1
    assert response_data["execution_summary"]["successful_node_count"] == 1
    assert response_data["execution_summary"]["node_execution_counts"] == {
        "retrieve_memory": 1,
    }
    assert response_data["execution_summary"]["memory_update_status"] == "skipped"
    assert response_data["execution_summary"]["llm_call_count"] == 1
    assert response_data["execution_summary"]["successful_llm_call_count"] == 1
    assert response_data["execution_summary"]["llm_active_duration_ms"] == 4.5
    assert response_data["execution_summary"]["prompt_versions"] == {
        "generation.analyze_brief": ["v1"],
    }
    assert response_data["execution_summary"]["profile_usage"] == {
        "fast": 1,
    }
    assert "llm_call_trace" not in response_data


def test_generate_film_rejects_empty_user_idea():
    """
    user_idea去除首尾空格后为空时，应返回422。
    """

    response = client.post(
        "/api/v1/films/generate",
        json={
            "user_id": "test_user_001",
            "user_idea": "   ",
        },
    )

    assert response.status_code == 422


def test_generate_film_rejects_invalid_user_id():
    """
    user_id包含非法字符时，应返回422。
    """

    response = client.post(
        "/api/v1/films/generate",
        json={
            "user_id": "bad/user",
            "user_idea": "生成一个校园故事",
        },
    )

    assert response.status_code == 422


def test_generate_film_graph_exception(monkeypatch):
    """
    Graph执行异常时，接口应返回500和本次execution_id。
    """

    def fake_invoke(initial_state, config=None):
        """
        模拟Graph失败，避免测试调用真实LLM。
        """
        assert config == {
            "configurable": {
                "thread_id": initial_state["execution_id"],
            },
        }
        raise RuntimeError(
            "fake graph error with internal stack"
        )

    monkeypatch.setattr(
        api_main.film_graph,
        "invoke",
        fake_invoke,
    )

    response = client.post(
        "/api/v1/films/generate",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )

    response_data = response.json()

    assert response.status_code == 500
    assert response_data["detail"]["execution_id"].startswith("exec_")
    assert response_data["detail"]["message"] == "Film Graph执行失败。"
    assert response_data["detail"]["execution_summary"]["status"] == "failed"
    assert response_data["detail"]["execution_summary"]["error_type"] == "RuntimeError"
    assert response_data["detail"]["execution_summary"]["error_message"] == "Film Graph执行失败。"
    assert "Traceback" not in str(response_data)
    assert "fake graph error" not in str(response_data)


def test_stream_film_completed_includes_summary_only_in_terminal_event(monkeypatch):
    """
    普通SSE完成时，只在completed终态事件中携带execution_summary。
    """

    def fake_stream(initial_state, config=None, stream_mode=None):
        execution_id = initial_state["execution_id"]
        assert config == {
            "configurable": {
                "thread_id": execution_id,
            },
        }
        assert stream_mode == "updates"

        yield {
            "analyze_brief": {
                "current_stage": "brief_completed",
                "execution_trace": [
                    {
                        "execution_id": execution_id,
                        "node": "analyze_brief",
                        "status": "success",
                        "stage": "brief_completed",
                        "duration_ms": 1.0,
                    }
                ],
                "llm_call_trace": [
                    _llm_call_event(),
                ],
            }
        }
        yield {
            "finalize": {
                "current_stage": "finalized",
                "final_output": {
                    "user_idea": initial_state["user_idea"],
                },
                "execution_trace": [
                    {
                        "execution_id": execution_id,
                        "node": "finalize",
                        "status": "success",
                        "stage": "finalized",
                        "duration_ms": 2.0,
                    }
                ],
            }
        }

    monkeypatch.setattr(
        api_main.film_graph,
        "stream",
        fake_stream,
    )

    response = client.post(
        "/api/v1/films/stream",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )
    events = _parse_sse_events(response)

    assert [
        event["event"]
        for event in events
    ] == [
        "started",
        "node_completed",
        "node_completed",
        "completed",
    ]
    assert "execution_summary" not in events[0]["data"]
    assert "execution_summary" not in events[1]["data"]
    assert "execution_summary" not in events[2]["data"]
    assert events[-1]["data"]["execution_summary"]["status"] == "completed"
    assert events[-1]["data"]["execution_summary"]["trace_event_count"] == 2
    assert events[-1]["data"]["execution_summary"]["active_duration_ms"] == 3.0
    assert events[-1]["data"]["execution_summary"]["llm_call_count"] == 1
    assert events[-1]["data"]["execution_summary"]["model_usage"] == {
        "qwen-plus": 1,
    }
    assert events[-1]["data"]["final_output"] == {
        "user_idea": "生成一个校园故事",
    }


def test_stream_film_error_includes_failed_summary(monkeypatch):
    """
    普通SSE异常时，error终态事件应包含failed Summary且不暴露原始异常。
    """

    def fake_stream(initial_state, config=None, stream_mode=None):
        execution_id = initial_state["execution_id"]
        yield {
            "retrieve_memory": {
                "current_stage": "memory_retrieved",
                "execution_trace": [
                    {
                        "execution_id": execution_id,
                        "node": "retrieve_memory",
                        "status": "success",
                        "stage": "memory_retrieved",
                        "duration_ms": 1.0,
                    }
                ],
            }
        }
        raise RuntimeError(
            "internal stream failure"
        )

    monkeypatch.setattr(
        api_main.film_graph,
        "stream",
        fake_stream,
    )

    response = client.post(
        "/api/v1/films/stream",
        json={
            "user_id": "test_user_001",
            "user_idea": "生成一个校园故事",
        },
    )
    events = _parse_sse_events(response)

    assert [
        event["event"]
        for event in events
    ] == [
        "started",
        "node_completed",
        "error",
    ]
    assert events[-1]["data"]["execution_summary"]["status"] == "failed"
    assert events[-1]["data"]["execution_summary"]["trace_event_count"] == 1
    assert events[-1]["data"]["execution_summary"]["error_type"] == "RuntimeError"
    assert events[-1]["data"]["execution_summary"]["error_message"] == "Film Graph执行失败。"
    assert "internal stream failure" not in response.text
