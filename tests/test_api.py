import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as api_main


app = api_main.app

client = TestClient(app)


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
    assert "Traceback" not in str(response_data)
    assert "fake graph error" not in str(response_data)
