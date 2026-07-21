from pydantic import BaseModel, ConfigDict, Field


class UserMemory(BaseModel):
    """
    单个用户的长期偏好。

    这里只保存跨多次影片生成仍然有价值的稳定偏好，
    不保存某一次请求的具体故事内容。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    user_id: str
    preferred_genres: list[str] = Field(default_factory=list)      # 喜欢的影片类型
    style_preferences: list[str] = Field(default_factory=list)      # 风格偏好
    disliked_elements: list[str] = Field(default_factory=list)        # 不喜欢的元素
    preferred_duration_sec: int | None = Field(default=None, ge=1, le=3600)    # 偏好的影片时长（秒）
    additional_preferences: list[str] = Field(default_factory=list)           # 其他偏好

    # 分阶段偏好
    story_preferences: list[str] = Field(default_factory=list)           # 故事大纲设计偏好
    scene_preferences: list[str] = Field(default_factory=list)           # 分场设计偏好

class MemoryUpdate(BaseModel):
    """
    从一次用户请求中提取出的长期偏好增量。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    should_update: bool                                                    # 是否需要更新长期偏好
    preferred_genres_to_add: list[str] = Field(default_factory=list)    # 需要添加的喜欢的影片类型
    style_preferences_to_add: list[str] = Field(default_factory=list)    # 需要添加的风格偏好
    disliked_elements_to_add: list[str] = Field(default_factory=list)    # 需要添加的不喜欢的元素
    preferred_duration_sec: int | None = Field(default=None, ge=1, le=3600)    # 需要添加的偏好的影片时长（秒）
    additional_preferences_to_add: list[str] = Field(default_factory=list)    # 需要添加的其他偏好
    
    # 分阶段偏好
    story_preferences_to_add: list[str] = Field(default_factory=list)      # 需要添加的故事大纲设计偏好
    scene_preferences_to_add: list[str] = Field(default_factory=list)       # 需要添加的分场设计偏好