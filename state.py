from typing import Literal, NotRequired, TypedDict
import json

from memory.models import MemoryUpdate, UserMemory
from schemas import Character, StoryOutline, Scene, FilmBrief, SceneReviewResult, StoryReviewResult

import operator
from typing import Annotated, TypedDict

# 单个节点的执行记录
class TraceEvent(TypedDict):
    execution_id: str    # 标识某一次Graph执行
    node: str
    status: str
    stage: str
    duration_ms: float


class LLMCallTraceEvent(TypedDict):
    node: str    # 发起本次LLM调用的Graph节点
    prompt_name: str    # Prompt Registry中的稳定名称
    prompt_version: str    # 本次实际使用的Prompt版本
    prompt_chars: int    # 最终Prompt字符数，不保存Prompt正文
    llm_profile: str    # Prompt Binding解析出的LLM Profile
    model_name: str    # LLM Profile中的模型名称
    temperature: float    # LLM Profile中的temperature
    status: Literal["success", "failed"]    # 本次真实LLM调用结果
    duration_ms: float    # 本次LLM调用耗时，单位毫秒
    error_type: str | None    # 失败时仅记录异常类型，不保存异常正文或堆栈


class HumanFeedbackEvent(TypedDict):
    scope: Literal["story", "scene"]    # 人工反馈作用范围
    decision: Literal["approve", "revise"]    # 人工选择：通过或要求修改
    feedback: str | None    # 人工反馈内容；approve时可以为空


class ReviewHistoryEvent(TypedDict):
    revision_round: int    # 本次审核对应的修订轮次
    passed: bool    # 本轮审核是否通过
    issues: list[str]    # 本轮发现的阻断性问题
    suggestions: list[str]    # 本轮给出的非阻断建议
    historical_issue_checks: NotRequired[list[dict]]    # 本轮对历史问题的状态判断；旧记录可以没有该字段


# 电影创作状态
class FilmState(TypedDict, total = False):
    # 用户身份
    user_id: str    # 标识用户，不等同于某一次Graph执行

    # 执行身份
    execution_id: str    # 标识某一次Graph执行，同一轮执行中的TraceEvent共用该ID

    # 用户本次输入
    user_idea: str

    # 本次读取到的长期记忆
    user_memory: UserMemory    # 当前用户已读取到的完整长期记忆

    # 本次提取到的长期记忆增量
    memory_update: MemoryUpdate    # 本次输入中提取出的长期偏好增量

    # 本次长期记忆更新状态
    memory_update_status: str    # 本次Memory更新结果，例如saved、skipped或failed

    # 本次长期记忆更新错误
    memory_update_error: str | None    # Memory更新失败时记录错误信息，成功时为空

    # 需求分析结果
    film_brief: FilmBrief

    # 创作中间结果
    characters: list[Character]
    story_outline: StoryOutline
    scenes: list[Scene]

    # 审核结果
    scene_review_result: SceneReviewResult
    story_review_result: StoryReviewResult

    # 模型审核历史
    story_review_history: Annotated[
        list[ReviewHistoryEvent],
        operator.add,
    ]    # 每次Story Review追加一条记录，用于本次执行内的回归检查
    scene_review_history: Annotated[
        list[ReviewHistoryEvent],
        operator.add,
    ]    # 每次Scene Review追加一条记录，用于本次执行内的回归检查

    # 人工故事审核结果
    human_review_decision: Literal["approve", "revise"]    # 人工选择：通过或要求修改
    human_feedback: str | None    # 人工修改意见；选择revise时必须非空

    # 人工审核反馈历史
    human_feedback_history: Annotated[
        list[HumanFeedbackEvent],
        operator.add,
    ]    # 每次人工审核追加一条事件，保留多轮反馈历史

    # 修订次数
    story_revision_count: int     # Story层修订次数
    scene_revision_count: int    # Scene层修订次数

    # 执行跟踪记录
    # Annotated表示给类型附加额外信息，operator.add为额外信息，告诉 LangGraph，当这个字段收到多个更新时，使用“加法”把旧值和新值合并。
    execution_trace: Annotated[
        list[TraceEvent],
        operator.add,
    ]     

    # LLM调用跟踪记录与节点Trace分开保存；每次真实调用只追加一条元数据事件。
    llm_call_trace: Annotated[
        list[LLMCallTraceEvent],
        operator.add,
    ]

    # 最终输出
    final_output: dict

    # 当前状态
    current_stage: str


if __name__ == "__main__":
    state: FilmState = {
        "user_id": "demo_user_001",
        "execution_id": "exec_demo",
        "user_idea": "一个关于爱情和冒险的故事",
        "story_revision_count": 0,
        "scene_revision_count": 0,
    }
    print(state)
    print(state["user_idea"])
    print(json.dumps(state, ensure_ascii=False, indent=2))
