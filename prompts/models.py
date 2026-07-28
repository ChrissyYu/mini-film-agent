from pydantic import BaseModel, ConfigDict


class PromptSpec(BaseModel):
    """
    描述一个具有稳定名称、版本和变量契约的Prompt模板。
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    template: str
    required_variables: tuple[str, ...]


class RenderedPrompt(BaseModel):
    """
    保存最终发送文本及其来源元数据，便于后续追踪Prompt版本。
    """

    model_config = ConfigDict(frozen=True)

    text: str
    name: str
    version: str
    chars: int
