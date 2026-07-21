import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from app.schemas import (
    FilmGenerateRequest,
    FilmGenerateResponse,
    HitlFilmResponse,
    HitlResumeRequest,
)
from app.sse import format_sse_event
from graph import film_graph, film_hitl_graph
from state import FilmState


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Mini Film Agent",
    version="0.1.0",
)


def _build_graph_config(
    execution_id: str,
) -> dict:
    """
    构造Checkpointer配置。

    Start和Resume必须使用同一个thread_id；当前直接复用execution_id，
    避免API层维护两套执行标识。
    """
    return {
        "configurable": {
            "thread_id": execution_id,
        },
    }


def _extract_interrupt_payload(
    event: dict,
) -> dict | None:
    """
    从LangGraph stream事件中提取可返回给客户端的interrupt payload。
    """
    interrupt_events = event.get("__interrupt__")

    if not interrupt_events:
        return None

    interrupt_event = interrupt_events[0]

    if hasattr(interrupt_event, "value"):
        return interrupt_event.value

    return interrupt_event


def _run_hitl_graph_until_pause_or_end(
    graph_input,
    config: dict,
    execution_id: str,
) -> HitlFilmResponse:
    """
    运行HITL Graph，直到暂停等待人工或完整结束。
    """
    final_output = None
    current_stage = "initialized"
    memory_update_status = None
    execution_trace = []

    for event in film_hitl_graph.stream(
        graph_input,
        config=config,
        stream_mode="updates",
    ):
        interrupt_payload = _extract_interrupt_payload(
            event
        )

        if interrupt_payload is not None:
            graph_state = film_hitl_graph.get_state(
                config
            )
            state_values = graph_state.values or {}

            return HitlFilmResponse(
                execution_id=execution_id,
                status="waiting_for_human",
                current_stage=state_values.get(
                    "current_stage",
                    current_stage,
                ),
                review_payload=interrupt_payload,
                execution_trace=state_values.get(
                    "execution_trace",
                    execution_trace,
                ),
                memory_update_status=state_values.get(
                    "memory_update_status",
                    memory_update_status,
                ),
            )

        for node_name, node_update in event.items():
            if node_name == "__interrupt__" or not isinstance(node_update, dict):
                continue

            current_stage = node_update.get(
                "current_stage",
                current_stage,
            )

            if "final_output" in node_update:
                final_output = node_update["final_output"]

            if "memory_update_status" in node_update:
                memory_update_status = node_update[
                    "memory_update_status"
                ]

            execution_trace.extend(
                node_update.get(
                    "execution_trace",
                    [],
                )
            )

    if final_output is None:
        raise RuntimeError(
            "HITL Graph执行结束后缺少final_output。"
        )

    return HitlFilmResponse(
        execution_id=execution_id,
        status="completed",
        current_stage=current_stage,
        final_output=final_output,
        execution_trace=execution_trace,
        memory_update_status=memory_update_status,
    )


def _ensure_hitl_checkpoint_waiting(
    execution_id: str,
    config: dict,
) -> None:
    """
    检查指定execution_id是否存在可恢复的人工审核checkpoint。

    当前InMemorySaver要求Start和Resume命中同一Python进程；
    服务重启后checkpoint会丢失，也不适合多worker部署。
    """
    graph_state = film_hitl_graph.get_state(
        config
    )

    if not graph_state.values and not graph_state.next:
        raise HTTPException(
            status_code=404,
            detail={
                "execution_id": execution_id,
                "message": "未找到可恢复的HITL执行。",
            },
        )

    if not graph_state.next or "human_review_story" not in graph_state.next:
        raise HTTPException(
            status_code=409,
            detail={
                "execution_id": execution_id,
                "message": "当前执行没有待处理的人工审核。",
            },
        )


@app.get("/health")
def health_check() -> dict[str, str]:
    """
    服务健康检查接口。
    """
    return {
        "status": "ok",
    }


@app.post(
    "/api/v1/films/generate",
    response_model=FilmGenerateResponse,
)
def generate_film(
    request: FilmGenerateRequest,
) -> FilmGenerateResponse:
    """
    非流式影片生成接口。
    """
    # API入口生成一次execution_id，用于标识本次Graph执行。
    execution_id = f"exec_{uuid4().hex}"

    try:
        # 构造Graph初始State，后续所有节点共享同一个execution_id。
        initial_state: FilmState = {
            "user_id": request.user_id,
            "execution_id": execution_id,
            "user_idea": request.user_idea,
            "story_revision_count": 0,
            "scene_revision_count": 0,
            "execution_trace": [],
            "current_stage": "initialized",
        }
        config = {
            "configurable": {
                "thread_id": execution_id,
            },
        }    # thread_id是Checkpointer查找和恢复Graph状态的键；当前复用execution_id避免维护两套标识。

        # 当前Graph和LLM调用均为同步逻辑，因此这里使用同步invoke。
        final_state = film_graph.invoke(
            initial_state,
            config=config,    # 后续恢复HITL时必须继续使用相同thread_id。
        )

        if "final_output" not in final_state:
            raise RuntimeError(
                "Film Graph执行完成后缺少final_output。"
            )

        # 整理API响应，只返回调用方需要的结果和执行轨迹。
        return FilmGenerateResponse(
            execution_id=execution_id,
            status="completed",
            current_stage=final_state.get(
                "current_stage",
                "unknown",
            ),
            final_output=final_state["final_output"],
            execution_trace=final_state.get(
                "execution_trace",
                [],
            ),
            memory_update_status=final_state.get(
                "memory_update_status"
            ),
        )

    except Exception:
        logger.exception(
            "Film Graph执行失败，execution_id=%s",
            execution_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "execution_id": execution_id,
                "message": "Film Graph执行失败。",
            },
        )


@app.post(
    "/api/v1/films/hitl/start",
    response_model=HitlFilmResponse,
)
def start_hitl_film(
    request: FilmGenerateRequest,
) -> HitlFilmResponse:
    """
    启动带人工故事审核暂停点的非流式HITL生成。
    """
    execution_id = f"exec_{uuid4().hex}"
    config = _build_graph_config(
        execution_id
    )

    initial_state: FilmState = {
        "user_id": request.user_id,
        "execution_id": execution_id,
        "user_idea": request.user_idea,
        "story_revision_count": 0,
        "scene_revision_count": 0,
        "execution_trace": [],
        "current_stage": "initialized",
    }

    try:
        return _run_hitl_graph_until_pause_or_end(
            initial_state,
            config,
            execution_id,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "HITL Graph启动失败，execution_id=%s",
            execution_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "execution_id": execution_id,
                "message": "HITL Graph执行失败。",
            },
        )


@app.post(
    "/api/v1/films/hitl/{execution_id}/resume",
    response_model=HitlFilmResponse,
)
def resume_hitl_film(
    execution_id: str,
    request: HitlResumeRequest,
) -> HitlFilmResponse:
    """
    恢复已经暂停在人工故事审核点的HITL生成。
    """
    config = _build_graph_config(
        execution_id
    )

    try:
        _ensure_hitl_checkpoint_waiting(
            execution_id,
            config,
        )

        return _run_hitl_graph_until_pause_or_end(
            Command(
                resume={
                    "decision": request.decision,
                    "feedback": request.feedback,
                }
            ),
            config,
            execution_id,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "HITL Graph恢复失败，execution_id=%s",
            execution_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "execution_id": execution_id,
                "message": "HITL Graph执行失败。",
            },
        )


@app.post("/api/v1/films/hitl/stream/start")
def stream_start_hitl_film(
    request: FilmGenerateRequest,
) -> StreamingResponse:
    """
    节点级SSE流式启动HITL影片生成。
    """
    # API入口生成一次execution_id，并同时作为Checkpointer的thread_id。
    execution_id = f"exec_{uuid4().hex}"
    config = _build_graph_config(
        execution_id
    )

    initial_state: FilmState = {
        "user_id": request.user_id,
        "execution_id": execution_id,
        "user_idea": request.user_idea,
        "story_revision_count": 0,
        "scene_revision_count": 0,
        "execution_trace": [],
        "current_stage": "initialized",
    }

    def event_generator():
        """
        启动HITL Graph，并将节点进度转成SSE事件。
        """
        final_output = None
        current_stage = "initialized"
        memory_update_status = None
        execution_trace = []

        try:
            yield format_sse_event(
                "started",
                {
                    "execution_id": execution_id,
                    "status": "running",
                },
            )

            for event in film_hitl_graph.stream(
                initial_state,
                config=config,    # 当前InMemorySaver要求resume命中同一Python进程和同一thread_id
                stream_mode="updates",    # 只接收节点局部更新，避免再次运行Graph
            ):
                interrupt_payload = _extract_interrupt_payload(
                    event
                )

                if interrupt_payload is not None:
                    # interrupt后checkpoint已经保存，SSE流可以结束；
                    # 后续人工决定由独立resume请求提交。
                    yield format_sse_event(
                        "human_review_required",
                        {
                            "execution_id": execution_id,
                            "status": "waiting_for_human",
                            "review_payload": interrupt_payload,
                        },
                    )
                    return

                for node_name, node_update in event.items():
                    if node_name == "__interrupt__" or not isinstance(node_update, dict):
                        continue

                    current_stage = node_update.get(
                        "current_stage",
                        current_stage,
                    )

                    if "final_output" in node_update:
                        final_output = node_update["final_output"]

                    if "memory_update_status" in node_update:
                        memory_update_status = node_update[
                            "memory_update_status"
                        ]

                    new_trace_events = node_update.get(
                        "execution_trace",
                        [],
                    )

                    for trace_event in new_trace_events:
                        trace_execution_id = trace_event.get(
                            "execution_id"
                        )
                        if trace_execution_id != execution_id:
                            raise RuntimeError(
                                "TraceEvent中的execution_id与接口execution_id不一致。"
                            )

                        execution_trace.append(
                            trace_event
                        )

                        yield format_sse_event(
                            "node_completed",
                            {
                                "execution_id": execution_id,
                                "node": node_name,
                                "status": trace_event["status"],
                                "stage": trace_event["stage"],
                                "duration_ms": trace_event["duration_ms"],
                            },
                        )

            if final_output is None:
                raise RuntimeError(
                    "HITL Graph执行完成后缺少final_output。"
                )

            yield format_sse_event(
                "completed",
                {
                    "execution_id": execution_id,
                    "status": "completed",
                    "current_stage": current_stage,
                    "final_output": final_output,
                    "execution_trace": execution_trace,
                    "memory_update_status": memory_update_status,
                },
            )

        except Exception:
            logger.exception(
                "HITL Graph流式启动失败，execution_id=%s",
                execution_id,
            )
            yield format_sse_event(
                "error",
                {
                    "execution_id": execution_id,
                    "status": "failed",
                    "message": "Film Graph执行失败。",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",    # 复用标准SSE响应类型
        headers={
            "Cache-Control": "no-cache",    # 避免缓存阻塞流式事件
            "X-Accel-Buffering": "no",    # 避免代理缓冲导致人工审核事件延迟
        },
    )


@app.post("/api/v1/films/hitl/stream/{execution_id}/resume")
def stream_resume_hitl_film(
    execution_id: str,
    request: HitlResumeRequest,
) -> StreamingResponse:
    """
    节点级SSE流式恢复HITL影片生成。
    """
    config = _build_graph_config(
        execution_id
    )

    # 404/409必须在StreamingResponse开始前判断；
    # 一旦SSE响应头发出，就不能再可靠改成HTTP错误状态。
    _ensure_hitl_checkpoint_waiting(
        execution_id,
        config,
    )

    def event_generator():
        """
        使用同一thread_id恢复HITL Graph，并持续输出节点级SSE事件。
        """
        try:
            yield format_sse_event(
                "resumed",
                {
                    "execution_id": execution_id,
                    "status": "running",
                    "decision": request.decision,
                },
            )

            for event in film_hitl_graph.stream(
                Command(
                    resume={
                        "decision": request.decision,
                        "feedback": request.feedback,
                    }
                ),
                config=config,    # resume必须复用原thread_id；当前InMemorySaver要求请求命中同一Python进程。
                stream_mode="updates",
            ):
                interrupt_payload = _extract_interrupt_payload(
                    event
                )

                if interrupt_payload is not None:
                    # revise后可能再次暂停；checkpoint已保存，后续继续由独立resume请求提交。
                    yield format_sse_event(
                        "human_review_required",
                        {
                            "execution_id": execution_id,
                            "status": "waiting_for_human",
                            "review_payload": interrupt_payload,
                        },
                    )
                    return

                for node_name, node_update in event.items():
                    if node_name == "__interrupt__" or not isinstance(node_update, dict):
                        continue

                    new_trace_events = node_update.get(
                        "execution_trace",
                        [],
                    )

                    for trace_event in new_trace_events:
                        trace_execution_id = trace_event.get(
                            "execution_id"
                        )
                        if trace_execution_id != execution_id:
                            raise RuntimeError(
                                "TraceEvent中的execution_id与接口execution_id不一致。"
                            )

                        yield format_sse_event(
                            "node_completed",
                            {
                                "execution_id": execution_id,
                                "node": node_name,
                                "status": trace_event["status"],
                                "stage": trace_event["stage"],
                                "duration_ms": trace_event["duration_ms"],
                            },
                        )

            # stream结束后从最新checkpoint读取完整最终状态，
            # 避免依赖某个局部node_update里恰好包含所有字段。
            graph_state = film_hitl_graph.get_state(
                config
            )
            final_state = graph_state.values or {}
            final_output = final_state.get(
                "final_output"
            )

            if final_output is None:
                raise RuntimeError(
                    "HITL Graph执行完成后缺少final_output。"
                )

            yield format_sse_event(
                "completed",
                {
                    "execution_id": execution_id,
                    "status": "completed",
                    "current_stage": final_state.get(
                        "current_stage",
                        "unknown",
                    ),
                    "final_output": final_output,
                    "execution_trace": final_state.get(
                        "execution_trace",
                        [],
                    ),
                    "memory_update_status": final_state.get(
                        "memory_update_status"
                    ),
                },
            )

        except Exception:
            logger.exception(
                "HITL Graph流式恢复失败，execution_id=%s",
                execution_id,
            )
            yield format_sse_event(
                "error",
                {
                    "execution_id": execution_id,
                    "status": "failed",
                    "message": "Film Graph执行失败。",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",    # 复用标准SSE响应类型
        headers={
            "Cache-Control": "no-cache",    # 避免缓存阻塞流式事件
            "X-Accel-Buffering": "no",    # 避免代理缓冲导致恢复事件延迟
        },
    )


@app.post("/api/v1/films/stream")
def stream_film(
    request: FilmGenerateRequest,
) -> StreamingResponse:
    """
    节点级SSE流式影片生成接口。
    """
    # API入口生成一次execution_id，用于标识本次Graph执行。
    execution_id = f"exec_{uuid4().hex}"

    # 构造Graph初始State，后续所有节点共享同一个execution_id。
    initial_state: FilmState = {
        "user_id": request.user_id,
        "execution_id": execution_id,
        "user_idea": request.user_idea,
        "story_revision_count": 0,
        "scene_revision_count": 0,
        "execution_trace": [],
        "current_stage": "initialized",
    }
    config = {
        "configurable": {
            "thread_id": execution_id,
        },
    }    # thread_id用于Checkpointer定位本次Graph线程；当前与execution_id保持一致。

    def event_generator():
        """
        使用普通同步生成器逐步输出Graph节点级SSE事件。
        """
        final_output = None
        current_stage = "initialized"
        memory_update_status = None
        execution_trace = []

        try:
            yield format_sse_event(
                "started",
                {
                    "execution_id": execution_id,
                    "status": "running",
                },
            )

            # 当前Graph和LLM调用均为同步逻辑，因此这里使用同步stream；
            # updates模式只返回每个节点的局部更新，便于转成节点级进度事件。
            for event in film_graph.stream(
                initial_state,
                config=config,    # HITL恢复同一线程时需要复用这个thread_id。
                stream_mode="updates",    # 只接收节点局部更新，不等待完整最终State
            ):
                for node_name, node_update in event.items():
                    # 局部更新来自不同节点，因此需要分别捕获最终输出、阶段和Memory状态。
                    current_stage = node_update.get(
                        "current_stage",
                        current_stage,
                    )

                    if "final_output" in node_update:
                        final_output = node_update["final_output"]

                    if "memory_update_status" in node_update:
                        memory_update_status = node_update[
                            "memory_update_status"
                        ]

                    new_trace_events = node_update.get(
                        "execution_trace",
                        [],
                    )

                    for trace_event in new_trace_events:
                        trace_execution_id = trace_event.get(
                            "execution_id"
                        )
                        # Trace必须继承本次请求的execution_id，避免不同执行的事件串在一起。
                        if trace_execution_id != execution_id:
                            raise RuntimeError(
                                "TraceEvent中的execution_id与接口execution_id不一致。"
                            )

                        execution_trace.append(
                            trace_event
                        )

                        yield format_sse_event(
                            "node_completed",
                            {
                                "execution_id": execution_id,
                                "node": node_name,
                                "status": trace_event["status"],
                                "stage": trace_event["stage"],
                                "duration_ms": trace_event["duration_ms"],
                            },
                        )

            # finalize节点应产生final_output；如果缺失，说明Graph未完整产出影片结果。
            if final_output is None:
                raise RuntimeError(
                    "Film Graph执行完成后缺少final_output。"
                )

            yield format_sse_event(
                "completed",
                {
                    "execution_id": execution_id,
                    "status": "completed",
                    "current_stage": current_stage,
                    "final_output": final_output,
                    "execution_trace": execution_trace,
                    "memory_update_status": memory_update_status,
                },
            )

        except Exception:
            logger.exception(
                "Film Graph流式执行失败，execution_id=%s",
                execution_id,
            )
            # 流式响应一旦开始，HTTP 200响应头通常已经发出；
            # 后续Graph错误不能再改成HTTP 500，只能通过SSE error事件告知客户端。
            yield format_sse_event(
                "error",
                {
                    "execution_id": execution_id,
                    "status": "failed",
                    "message": "Film Graph执行失败。",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",    # 告诉客户端按SSE事件流解析响应
        headers={
            "Cache-Control": "no-cache",    # 避免客户端或代理缓存流式事件
            "X-Accel-Buffering": "no",    # 关闭Nginx等代理缓冲，尽量及时推送小事件
        },
    )
