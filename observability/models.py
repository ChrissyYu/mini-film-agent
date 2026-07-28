from typing import Literal

from pydantic import BaseModel, Field


class ExecutionSummary(BaseModel):
    """
    单次Graph执行的轻量汇总。

    这里只聚合已经存在于State和Trace中的稳定字段；
    issue状态、失败节点和错误信息会在后续M7.2继续扩展。
    """

    execution_id: str    # 标识某一次Graph执行
    status: Literal["completed", "waiting_for_human", "failed"]    # 执行级状态，不等同于单个TraceEvent的success
    current_stage: str    # 汇总时State所处的最新阶段
    active_duration_ms: float = Field(ge=0)    # 有效节点耗时之和，单位毫秒
    trace_event_count: int = Field(ge=0)    # execution_trace中的事件数量
    successful_node_count: int = Field(ge=0)    # status为success的节点事件数量
    node_execution_counts: dict[str, int] = Field(default_factory=dict)    # 每个节点实际执行次数
    story_review_rounds: int = Field(ge=0)    # Story Review轮次，优先来自story_review_history
    story_revision_count: int = Field(ge=0)    # Story修订次数，直接读取State计数
    scene_review_rounds: int = Field(ge=0)    # Scene Review轮次，优先来自scene_review_history
    scene_revision_count: int = Field(ge=0)    # Scene修订次数，直接读取State计数
    human_feedback_count: int = Field(ge=0)    # 人工反馈事件总数
    human_revision_count: int = Field(ge=0)    # decision为revise的人工反馈次数
    memory_update_status: str | None = None    # 本次长期Memory更新状态，例如saved、skipped或failed
    failed_node: str | None = None    # 失败节点名称；优先来自status为failed的TraceEvent
    error_type: str | None = None    # 异常类型名称，不包含traceback
    error_message: str | None = None    # 简短异常消息，不保存堆栈或敏感上下文
    story_issue_status_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "resolved": 0,
            "unresolved": 0,
            "regressed": 0,
        },
    )    # Story历史issue最新状态计数
    scene_issue_status_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "resolved": 0,
            "unresolved": 0,
            "regressed": 0,
        },
    )    # Scene历史issue最新状态计数
    llm_call_count: int = Field(ge=0)    # 本次执行真实发生的LLM调用总数
    successful_llm_call_count: int = Field(ge=0)    # status为success的LLM调用数量
    failed_llm_call_count: int = Field(ge=0)    # status为failed的LLM调用数量
    llm_active_duration_ms: float = Field(ge=0)    # 有效LLM调用耗时之和，不包含人工等待时间
    prompt_versions: dict[str, list[str]] = Field(default_factory=dict)    # 每个Prompt实际使用过的版本，按首次出现顺序去重
    profile_usage: dict[str, int] = Field(default_factory=dict)    # 每个LLM Profile的调用次数
    model_usage: dict[str, int] = Field(default_factory=dict)    # 每个模型名称的调用次数
