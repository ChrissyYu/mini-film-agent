#=============== 定义角色、故事梗概、场景、审核结果的结构化数据格式 ================
from typing import Literal

from pydantic import BaseModel, Field

# 需求解析结果
class FilmBrief(BaseModel):
    target_duration_sec: int = Field(gt=0, description="影片目标总时长，单位为秒",)                # 目标时长（秒）
    genre: str = Field(description="叙事类型，只填写如科幻、校园、爱情、悬疑，不填写摄影或视觉风格")      # 电影类型
    core_theme: str = Field(description="影片希望表达的核心情感或思想")                             # 核心主题
    visual_style: str = Field(description="简短描述整体视觉风格，包括色调、氛围或艺术方向，不超过30字")         # 视觉风格
    recommended_scene_count: int = Field(gt=0, description="根据影片时长和故事复杂度推荐的分场数量")     # 推荐的分镜数量

# 角色列表
class Character(BaseModel):
    name: str                           # 角色名称
    role: str                           # 身份
    appearance: str                     # 外貌和穿着特征
    personality: list[str] = Field(default_factory=list)    # 性格特征
    motivation: str                     # 行为动机
    continuity_constraints: list[str] = Field(default_factory=list)    # 连续性约束

class CharacterList(BaseModel):
    characters: list[Character] = Field(
        min_length = 1,
        max_length = 3,
        description="短片中的主要角色列表"
        )

# 故事大纲
class StoryOutline(BaseModel):
    setup: str = Field(description="故事开始阶段的人物状态、环境背景和初始关系，不包含具体镜头描述")
    conflict: str = Field(description="推动故事发展的核心矛盾或人物目标冲突")
    turning_point: str = Field(description="改变故事方向的重要事件或人物认知变化")
    ending: str = Field(description="故事最终结果以及人物情感变化")
    theme: str = Field(description="故事最终表达的核心情感或思想，应与创作主题一致")

# 分场镜头设计
class Scene(BaseModel):
    scene_id: int = Field(gt=0)         # 场景编号
    duration_sec: int = Field(gt=0)     # 持续时间（秒）
    location: str                       # 场景地点
    characters: list[str] = Field(default_factory=list)    # 角色名称列表
    action: str                         # 动作描述
    dialogue: str = ""                  # 对话内容
    visual_goal: str                   # 视觉目标，该场景希望实现的画面或叙事效果

class SceneList(BaseModel):
    scenes: list[Scene] = Field(
        min_length=1,
        max_length=10,
        description="短片分场列表"
    )

# 历史问题状态判断
class ReviewIssueCheck(BaseModel):
    issue: str                           # 被追踪的历史问题文本
    status: Literal["resolved", "unresolved", "regressed"]    # 当前版本中的问题状态
    evidence: str                        # 基于当前版本内容给出的简洁判断依据


# scene审核结果
class SceneReviewResult(BaseModel):     # rule + llm scene最终审核结果
    passed: bool                        # 是否通过审核
    issues: list[str] = Field(default_factory=list)    # 审核结果问题列表
    suggestions: list[str] = Field(default_factory=list)    # 审核结果建议列表
    historical_issue_checks: list[ReviewIssueCheck] = Field(default_factory=list)    # 历史问题本轮状态判断

class SceneCriticResult(BaseModel):       # llm scene critic审核结果
    passed: bool                        # 是否通过审核
    issues: list[str] = Field(default_factory=list)    # llm审核结果问题列表
    suggestions: list[str] = Field(default_factory=list)    # llm审核结果建议列表
    historical_issue_checks: list[ReviewIssueCheck] = Field(default_factory=list)    # llm对历史问题的状态判断

# story审核结果
class StoryReviewResult(BaseModel):     # rule + llm story最终审核结果
    passed: bool                        # 是否通过审核
    issues: list[str] = Field(default_factory=list)    # 审核结果问题列表
    suggestions: list[str] = Field(default_factory=list)    # 审核结果建议列表
    historical_issue_checks: list[ReviewIssueCheck] = Field(default_factory=list)    # 历史问题本轮状态判断

class StoryCriticResult(BaseModel):       # llm story critic审核结果
    passed: bool                        # 是否通过审核
    issues: list[str] = Field(default_factory=list)    # llm审核结果问题列表
    suggestions: list[str] = Field(default_factory=list)    # llm审核结果建议列表
    historical_issue_checks: list[ReviewIssueCheck] = Field(default_factory=list)    # llm对历史问题的状态判断



if __name__ == "__main__":
    character = Character(name="John Doe", role="主角", appearance="穿着西装", personality=["勇敢", "聪明"], motivation="保护家人")
    print(character.model_dump_json(indent=2))
    print(character)
    print(character.name)
