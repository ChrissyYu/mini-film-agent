from typing import Literal

from pydantic import BaseModel, ConfigDict


class LLMProfile(BaseModel):
    """
    描述一组可复用的模型配置，不包含真实API Key。

    Prompt版本与模型配置彼此独立；Profile只保存模型连接参数，
    API Key在Factory创建客户端时按环境变量名称读取。
    """

    model_config = ConfigDict(frozen=True)

    name: str
    provider: Literal[
        "dashscope_openai_compatible"
    ]
    model: str
    temperature: float
    base_url: str
    api_key_env: str
