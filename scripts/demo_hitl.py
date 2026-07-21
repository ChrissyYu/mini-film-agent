import json
import sys
from pathlib import Path
from uuid import uuid4

from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import film_hitl_graph
from state import FilmState


def _build_config(
    execution_id: str,
) -> dict:
    """
    使用execution_id作为thread_id，确保暂停和恢复落在同一条Graph线程上。
    """
    return {
        "configurable": {
            "thread_id": execution_id,
        },
    }


def _print_json(
    title: str,
    value,
) -> None:
    """
    以JSON格式打印调试信息，便于人工检查Graph状态。
    """
    print(f"\n{title}")
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def _get_interrupt_payload(
    event: dict,
):
    """
    从LangGraph stream事件中提取interrupt payload。
    """
    interrupt_events = event.get("__interrupt__")

    if not interrupt_events:
        return None

    interrupt_event = interrupt_events[0]

    if hasattr(interrupt_event, "value"):
        return interrupt_event.value

    return interrupt_event


def _read_human_decision() -> dict:
    """
    从终端读取人工审核决定。
    """
    while True:
        decision = input(
            "\n请输入人工审核决定（approve/revise）："
        ).strip()

        if decision in {"approve", "revise"}:
            break

        print("只能输入 approve 或 revise。")

    feedback = None

    if decision == "revise":
        while True:
            feedback = input(
                "请输入非空修改意见："
            ).strip()

            if feedback:
                break

            print("选择 revise 时 feedback 不能为空。")

    return {
        "decision": decision,
        "feedback": feedback,
    }


def _run_until_pause_or_end(
    graph_input,
    config: dict,
) -> str:
    """
    运行Graph，直到遇到人工暂停、正常完成或失败。
    """
    try:
        for event in film_hitl_graph.stream(
            graph_input,
            config=config,
            stream_mode="updates",
        ):
            interrupt_payload = _get_interrupt_payload(
                event
            )

            if interrupt_payload is not None:
                _print_json(
                    "Interrupt Payload:",
                    interrupt_payload,
                )
                graph_state = film_hitl_graph.get_state(
                    config
                )
                print(
                    "\ngraph.get_state(config).next:"
                )
                print(graph_state.next)
                return "waiting_for_human"

            for node_name, node_update in event.items():
                current_stage = node_update.get(
                    "current_stage"
                )
                if current_stage:
                    print(
                        f"节点完成：{node_name} | {current_stage}"
                    )

        return "completed"

    except Exception as exc:
        print(
            f"\nGraph执行失败：{exc}"
        )
        return "failed"


def main() -> None:
    """
    本地真实验证 film_hitl_graph 的暂停与恢复。
    """
    execution_id = f"exec_{uuid4().hex}"
    config = _build_config(
        execution_id
    )

    initial_state: FilmState = {
        "user_id": "demo_user_001",
        "execution_id": execution_id,
        "user_idea": (
            "毕业季同学们一起拍摄毕业照，面临分别，依依不舍，"
            "生成一段60秒的青春校园影片"
        ),
        "story_revision_count": 0,
        "scene_revision_count": 0,
        "execution_trace": [],
        "current_stage": "initialized",
    }

    print(
        f"Execution ID：{execution_id}"
    )

    status = _run_until_pause_or_end(
        initial_state,
        config,
    )

    while status == "waiting_for_human":
        resume_value = _read_human_decision()
        status = _run_until_pause_or_end(
            Command(
                resume=resume_value
            ),
            config,
        )

    if status == "completed":
        graph_state = film_hitl_graph.get_state(
            config
        )
        final_state = graph_state.values

        _print_json(
            "Final Output:",
            final_state.get("final_output"),
        )
        _print_json(
            "Execution Trace:",
            final_state.get("execution_trace", []),
        )
        print(
            "\n最终 current_stage:"
        )
        print(
            final_state.get("current_stage")
        )
        return

    print(
        "\n状态：failed"
    )


if __name__ == "__main__":
    main()
