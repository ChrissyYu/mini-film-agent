import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


USER_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{1,64}$"
)    # API层user_id规则：字母、数字、下划线、连字符，长度1至64


class FilmGenerateRequest(BaseModel):
    """
    影片生成接口的请求体。
    """

    user_id: str = Field(
        min_length=1,
        max_length=64,
        description="用户ID，只允许字母、数字、下划线和连字符。",
    )    # 标识用户，用于读取和更新该用户的长期Memory
    user_idea: str = Field(
        max_length=4000,
        description="用户本次影片创意或生成需求。",
    )    # 用户本次输入，去除首尾空格后不能为空

    @field_validator("user_id")
    @classmethod
    def validate_user_id(
        cls,
        value: str,
    ) -> str:
        """
        校验user_id只能使用安全字符，避免被用于构造异常路径。
        """
        if not USER_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "user_id只能包含字母、数字、下划线和连字符，长度为1至64。"
            )

        return value

    @field_validator("user_idea")
    @classmethod
    def validate_user_idea(
        cls,
        value: str,
    ) -> str:
        """
        清理用户输入首尾空格，并禁止空需求。
        """
        stripped_value = value.strip()

        if not stripped_value:
            raise ValueError(
                "user_idea去除首尾空格后不能为空。"
            )

        return stripped_value


class TraceEventResponse(BaseModel):
    """
    API返回给调用方的单条执行轨迹。
    """

    execution_id: str    # 标识某一次Graph执行
    node: str    # 节点名称
    status: str    # 节点执行状态
    stage: str    # 节点执行后所处阶段
    duration_ms: float    # 节点执行耗时，单位毫秒


class FilmGenerateResponse(BaseModel):
    """
    影片生成接口的响应体。
    """

    execution_id: str    # 标识某一次Graph执行
    status: str    # Graph是否完整执行完成，不代表审核一定通过
    current_stage: str    # Graph结束时的当前阶段
    final_output: dict[str, Any]    # 最终影片策划结果
    execution_trace: list[TraceEventResponse]    # 本次执行产生的节点轨迹
    memory_update_status: str | None    # Memory更新状态，可能为空


class HitlResumeRequest(BaseModel):
    """
    HITL人工审核恢复请求体。
    """

    decision: Literal["approve", "revise"]    # 人工决定：通过或要求修改
    feedback: str | None = None    # 人工修改意见；revise时必须非空

    @model_validator(mode="after")
    def validate_feedback(self):
        """
        revise时必须提供非空feedback；approve时允许为空。
        """
        if self.feedback is not None:
            self.feedback = self.feedback.strip()

        if self.decision == "revise" and not self.feedback:
            raise ValueError(
                "decision为revise时feedback不能为空。"
            )

        if self.decision == "approve" and not self.feedback:
            self.feedback = None

        return self


class HitlFilmResponse(BaseModel):
    """
    HITL启动或恢复接口的响应体。
    """

    execution_id: str    # 标识某一次Graph执行，同时作为thread_id
    status: Literal["waiting_for_human", "completed"]    # 当前HITL执行状态
    current_stage: str    # 当前Graph阶段
    review_payload: dict[str, Any] | None = None    # 等待人工审核时返回的审核材料
    final_output: dict[str, Any] | None = None    # 完成时返回的最终影片策划结果
    execution_trace: list[TraceEventResponse] = Field(default_factory=list)    # 已完成节点轨迹
    memory_update_status: str | None = None    # Memory更新状态，可能为空
