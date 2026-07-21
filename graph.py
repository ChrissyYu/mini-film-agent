import json
from functools import wraps
from time import perf_counter
from typing import Callable, Literal
from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from state import FilmState
from nodes import (
    analyze_brief,
    design_characters,
    plan_story,
    write_scenes,
)

from reviews import review_scene, review_story
from reviews.human_review_story import human_review_story
from revisions import revise_scene, revise_story
from memory.retrieve_memory import retrieve_memory
from memory.update_memory import update_memory


# ============================================================
# 节点函数类型
# ============================================================
# 接收一个FilmState，返回一个局部State更新字典的函数。
NodeFunction = Callable[[FilmState], dict[str, any]]


# ============================================================
# 循环控制配置
# ============================================================

MAX_STORY_REVISIONS = 3
MAX_SCENE_REVISIONS = 2


# Checkpointer保存的是Graph线程状态，用于后续HITL恢复同一轮执行。
# InMemorySaver只适合本地开发和HITL功能验证；进程重启后checkpoint会丢失。
# 用户长期偏好的JSON Memory与Checkpointer职责不同：前者保存用户偏好，后者保存Graph执行线程状态。
checkpointer = InMemorySaver()


# ============================================================
# 条件路由函数
# ============================================================

def route_after_story_review(
    state: FilmState,
) -> Literal["write_scenes", "revise_story"]:
    """
    根据故事审核结果决定下一步。

    1. 审核通过：进入分场生成。
    2. 审核失败但未达到修订上限：修改故事。
    3. 达到修订上限：停止继续修改故事，进入分场生成。
    """
    story_review_result = state["story_review_result"]
    story_revision_count = state.get("story_revision_count", 0)

    if story_review_result.passed:
        return "write_scenes"

    if story_revision_count >= MAX_STORY_REVISIONS:
        print(
            f"[Story Router] 故事审核仍未通过，"
            f"但已达到最大修订次数 {MAX_STORY_REVISIONS}，"
            "继续进入分场规划。"
        )
        return "write_scenes"

    return "revise_story"


def route_after_scene_review(
    state: FilmState,
) -> Literal["finalize", "revise_scene"]:
    """
    根据分场审核结果决定下一步。

    1. 审核通过：整理最终结果。
    2. 审核失败但未达到修订上限：修改分场。
    3. 达到修订上限：停止继续修改，输出当前最好结果。
    """
    scene_review_result = state["scene_review_result"]
    scene_revision_count = state.get("scene_revision_count", 0)

    if scene_review_result.passed:
        return "finalize"

    if scene_revision_count >= MAX_SCENE_REVISIONS:
        print(
            f"[Scene Router] 分场审核仍未通过，"
            f"但已达到最大修订次数 {MAX_SCENE_REVISIONS}，"
            "输出当前版本。"
        )
        return "finalize"

    return "revise_scene"


def route_after_human_review(
    state: FilmState,
) -> Literal["write_scenes", "revise_story"]:
    """
    根据人工故事审核结果决定是否继续分场生成。
    """
    decision = state.get("human_review_decision")

    if decision == "approve":
        return "write_scenes"

    if decision == "revise":
        return "revise_story"

    raise ValueError(
        "human_review_decision只能是approve或revise。"
    )


# ============================================================
# 最终输出节点
# ============================================================

def finalize(state: FilmState) -> dict:
    """
    将分散在 FilmState 中的 Pydantic 对象整理为普通字典，
    形成可以打印、保存或返回 API 的最终策划案。
    """

    story_review = state.get("story_review_result")
    scene_review = state.get("scene_review_result")

    final_output = {
        "user_idea": state["user_idea"],
        "film_brief": state["film_brief"].model_dump(),
        "characters": [
            character.model_dump()
            for character in state["characters"]
        ],
        "story_outline": state["story_outline"].model_dump(),
        "scenes": [
            scene.model_dump()
            for scene in state["scenes"]
        ],
        "review_summary": {
            "story": (
                story_review.model_dump()
                if story_review is not None
                else None
            ),
            "scene": (
                scene_review.model_dump()
                if scene_review is not None
                else None
            ),
            "story_revision_count": state.get(
                "story_revision_count",
                0,
            ),
            "scene_revision_count": state.get(
                "scene_revision_count",
                0,
            ),
        },
    }

    return {
        "final_output": final_output,
        "current_stage": "finalized",
    }


#================= 执行记录 节点包装器 ==============

def trace_node(
    node_name: str,
    node_function: NodeFunction,
) -> NodeFunction:
    """
    为节点增加统一的执行轨迹记录。
    原节点负责业务逻辑；
    包装器负责记录节点名称、执行状态、阶段和耗时。
    """

    @wraps(node_function)     # 保留原函数的元信息
    def wrapped(state: FilmState) -> dict:
        # execution_id必须由Graph外部入口生成，并在整次执行中保持不变。
        execution_id = state.get("execution_id")
        if not execution_id:
            raise ValueError(
                f"节点 {node_name} 执行时缺少execution_id。"
            )

        # 记录节点开始时间
        start_time = perf_counter()

        # 执行原始节点函数
        result = node_function(state)

        # 对节点返回值进行基础检查，避免后续在result.get处
        # 出现难以理解的报错。
        if not isinstance(result, dict):
            raise TypeError(
                f"节点 {node_name} 必须返回dict，"
                f"实际返回类型为："
                f"{type(result).__name__}"
            )

        # 计算原始节点执行耗时
        duration_ms = round(
            (perf_counter() - start_time) * 1000,
            2,
        )

        # 优先读取节点执行后返回的新stage
        current_stage = result.get(
            "current_stage",
            state.get("current_stage", "unknown"),
        )

        # 构造本次节点执行记录
        trace_event = {
            "execution_id": execution_id,
            "node": node_name,
            "status": "success",
            "stage": current_stage,
            "duration_ms": duration_ms,
        }

        # 保留原节点返回结果，同时返回一条新增trace
        # FilmState中的execution_trace使用operator.add reducer，
        # LangGraph会把这一条记录追加到历史trace后面。
        return {
            **result,
            "execution_trace": [trace_event],
        }
    # 返回包装后的新节点函数，而不是立即执行它（这样在graph build时可以构造得到包装后的节点，而不是立即执行）
    return wrapped


# ============================================================
# 构建 Graph
# ============================================================

def build_graph():
    """
    创建并编译 Film Agent Graph。
    """

    builder = StateGraph(FilmState)

    # ---------------- 添加节点 ----------------

    builder.add_node(
        "retrieve_memory",
        trace_node("retrieve_memory", retrieve_memory),
    )

    builder.add_node(
        "analyze_brief",
        trace_node("analyze_brief", analyze_brief),
    )

    builder.add_node(
        "design_characters",
        trace_node("design_characters", design_characters),
    )

    builder.add_node(
        "plan_story",
        trace_node("plan_story", plan_story),
    )

    builder.add_node(
        "review_story",
        trace_node("review_story", review_story),
    )

    builder.add_node(
        "revise_story",
        trace_node("revise_story", revise_story),
    )

    builder.add_node(
        "write_scenes",
        trace_node("write_scenes", write_scenes),
    )

    builder.add_node(
        "review_scene",
        trace_node("review_scene", review_scene),
    )

    builder.add_node(
        "revise_scene",
        trace_node("revise_scene", revise_scene),
    )

    builder.add_node(
        "finalize",
        trace_node("finalize", finalize),
    )

    builder.add_node(
        "update_memory",
        trace_node("update_memory", update_memory),
    )

    # ---------------- 主流程边 ----------------

    builder.add_edge(
        START,
        "retrieve_memory",
    )

    builder.add_edge(
        "retrieve_memory",
        "analyze_brief",
    )

    builder.add_edge(
        "analyze_brief",
        "design_characters",
    )

    builder.add_edge(
        "design_characters",
        "plan_story",
    )

    builder.add_edge(
        "plan_story",
        "review_story",
    )

    # ---------------- 故事审核条件边 ----------------

    builder.add_conditional_edges(
        "review_story",
        route_after_story_review,
        {
            "write_scenes": "write_scenes",    # 格式意义："路由函数返回的标签": "Graph中的节点名称"，表示如果路由函数返回"write_scenes"，则跳转到"write_scenes"节点
            "revise_story": "revise_story",
        },
    )

    # 故事修订后再次审核，而不是重新 plan_story
    builder.add_edge(
        "revise_story",
        "review_story",
    )

    # ---------------- 分场审核 ----------------

    builder.add_edge(
        "write_scenes",
        "review_scene",
    )

    builder.add_conditional_edges(
        "review_scene",
        route_after_scene_review,
        {
            "finalize": "finalize",
            "revise_scene": "revise_scene",
        },
    )

    # 分场修订后再次审核，而不是重新 write_scenes
    builder.add_edge(
        "revise_scene",
        "review_scene",
    )

    # ---------------- 结束 ----------------

    builder.add_edge(
        "finalize",
        "update_memory",
    )

    builder.add_edge(
        "update_memory",
        END,
    )

    # 编译Graph，返回可invoke/stream的可执行Graph对象。
    return builder.compile(
        checkpointer=checkpointer,
    )


# 编译后的 Graph，可以被其他文件直接导入
film_graph = build_graph()


def build_hitl_graph():
    """
    创建带人工故事审核暂停点的 Film Agent Graph。
    """

    builder = StateGraph(FilmState)

    # ---------------- 添加节点 ----------------

    builder.add_node(
        "retrieve_memory",
        trace_node("retrieve_memory", retrieve_memory),
    )

    builder.add_node(
        "analyze_brief",
        trace_node("analyze_brief", analyze_brief),
    )

    builder.add_node(
        "design_characters",
        trace_node("design_characters", design_characters),
    )

    builder.add_node(
        "plan_story",
        trace_node("plan_story", plan_story),
    )

    builder.add_node(
        "review_story",
        trace_node("review_story", review_story),
    )

    builder.add_node(
        "human_review_story",
        trace_node("human_review_story", human_review_story),
    )

    builder.add_node(
        "revise_story",
        trace_node("revise_story", revise_story),
    )

    builder.add_node(
        "write_scenes",
        trace_node("write_scenes", write_scenes),
    )

    builder.add_node(
        "review_scene",
        trace_node("review_scene", review_scene),
    )

    builder.add_node(
        "revise_scene",
        trace_node("revise_scene", revise_scene),
    )

    builder.add_node(
        "finalize",
        trace_node("finalize", finalize),
    )

    builder.add_node(
        "update_memory",
        trace_node("update_memory", update_memory),
    )

    # ---------------- 主流程边 ----------------

    builder.add_edge(
        START,
        "retrieve_memory",
    )

    builder.add_edge(
        "retrieve_memory",
        "analyze_brief",
    )

    builder.add_edge(
        "analyze_brief",
        "design_characters",
    )

    builder.add_edge(
        "design_characters",
        "plan_story",
    )

    builder.add_edge(
        "plan_story",
        "review_story",
    )

    # 原自动流程准备进入write_scenes时，HITL图先暂停给人工审核故事大纲。
    builder.add_conditional_edges(
        "review_story",
        route_after_story_review,
        {
            "write_scenes": "human_review_story",
            "revise_story": "revise_story",
        },
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
        "revise_story",
        "review_story",
    )

    # ---------------- 分场审核 ----------------

    builder.add_edge(
        "write_scenes",
        "review_scene",
    )

    builder.add_conditional_edges(
        "review_scene",
        route_after_scene_review,
        {
            "finalize": "finalize",
            "revise_scene": "revise_scene",
        },
    )

    builder.add_edge(
        "revise_scene",
        "review_scene",
    )

    builder.add_edge(
        "finalize",
        "update_memory",
    )

    builder.add_edge(
        "update_memory",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
    )


# 带HITL暂停点的独立Graph；保留film_graph不变，避免影响现有API。
film_hitl_graph = build_hitl_graph()


# ============================================================
# 本地端到端测试
# ============================================================

if __name__ == "__main__":
    execution_id = f"exec_{uuid4().hex}"
    config = {
        "configurable": {
            "thread_id": execution_id,
        },
    }    # thread_id是Checkpointer查找和恢复Graph状态的键；当前复用execution_id，避免维护两套执行标识。

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

    # 根据stream事件记录实际经过的节点
    execution_path: list[str] = []

    # 收集每个节点返回的execution_trace事件
    execution_trace: list[dict[str, any]] = []

    # 保存finalize节点产生的最终输出
    final_output: dict[str, any] | None = None

    print("\n" + "=" * 80)
    print("FILM GRAPH EXECUTION STARTED")
    print(f"Execution ID：{initial_state['execution_id']}")
    print("=" * 80)

    for event in film_graph.stream(
        initial_state,
        config=config,    # 后续恢复HITL时必须继续使用相同thread_id。
        stream_mode="updates",
    ):
        # event通常是：
        # {
        #     "节点名称": {
        #         "该节点返回的局部state更新": ...
        #     }
        # }

        for node_name, node_update in event.items():
            execution_path.append(node_name)

            print(f"\n>>> 正在执行节点：{node_name}")

            current_stage = node_update.get("current_stage")
            if current_stage:
                print(f"    当前阶段：{current_stage}")
            
            # stream_mode="updates"下，node_update中的execution_trace通常只包含当前节点新增的一条记录。
            if "execution_trace" in node_update:
                new_trace_events = node_update[
                    "execution_trace"
                ]
                execution_trace.extend(
                    new_trace_events
                )
                # 展示执行记录
                for trace_event in new_trace_events:
                    print(
                        "    执行记录："
                        f"{trace_event['execution_id']} | "
                        f"{trace_event['status']} | "
                        f"{trace_event['duration_ms']} ms"
                    )

            # 展示故事审核结果
            if "story_review_result" in node_update:
                review_result = node_update["story_review_result"]

                print(
                    f"    故事审核："
                    f"{'通过' if review_result.passed else '未通过'}"
                )

                if review_result.issues:
                    print("    故事问题：")
                    for issue in review_result.issues:
                        print(f"      - {issue}")

            # 展示故事修订次数
            if "story_revision_count" in node_update:
                print(
                    "    故事修订次数："
                    f"{node_update['story_revision_count']}"
                )

            # 展示分场审核结果
            if "scene_review_result" in node_update:
                review_result = node_update["scene_review_result"]

                print(
                    f"    分场审核："
                    f"{'通过' if review_result.passed else '未通过'}"
                )

                if review_result.issues:
                    print("    分场问题：")
                    for issue in review_result.issues:
                        print(f"      - {issue}")

            # 展示分场修订次数
            if "scene_revision_count" in node_update:
                print(
                    "    分场修订次数："
                    f"{node_update['scene_revision_count']}"
                )

            # 捕获最终输出
            if "final_output" in node_update:
                final_output = node_update["final_output"]


    print("\n" + "=" * 80)
    print("GRAPH EXECUTION FINISHED")
    print("=" * 80)

    print("\n实际执行路径：")
    print(" → ".join(execution_path))

    # 输出完整节点执行轨迹
    print("\n执行轨迹详情：")
    if execution_trace:
        trace_execution_ids = {
            trace_event["execution_id"]
            for trace_event in execution_trace
        }
        assert trace_execution_ids == {
            initial_state["execution_id"]
        }
        print(
            "\n所有执行轨迹均关联到同一个Execution ID。"
        )

        for index, trace_event in enumerate(
            execution_trace,
            start=1,
        ):
            print(
                f"{index}. "
                f"{trace_event['execution_id']} | "
                f"{trace_event['node']} | "
                f"{trace_event['status']} | "
                f"{trace_event['stage']} | "
                f"{trace_event['duration_ms']} ms"
            )
    else:
        print("未获取到execution_trace。")

    # 对比stream路径与Graph trace路径
    trace_path = [
        trace_event["node"]
        for trace_event in execution_trace
    ]
    if trace_path == execution_path:
        print(
            "\n执行路径与execution_trace一致。"
        )
    else:
        print(
            "\n警告：stream路径与"
            "execution_trace不一致。"
        )
        print(
            f"stream路径：{execution_path}"
        )
        print(
            f"trace路径：{trace_path}"
        )

    # 统计节点本身的累计执行时间
    total_node_duration_ms = sum(
        trace_event["duration_ms"]
        for trace_event in execution_trace
    )
    print(
        "\n节点累计耗时："
        f"{total_node_duration_ms:.2f} ms"
    )

    # 最终输出
    if final_output is not None:
        print("\n最终输出：")

        print(
            json.dumps(
                final_output,
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("\n警告：Graph执行结束，但没有获得final_output。")
