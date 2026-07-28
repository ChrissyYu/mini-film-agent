from pathlib import Path

from prompts.models import PromptSpec


class PromptRegistryError(Exception):
    """
    Prompt注册和查找错误的基类。
    """


class DuplicatePromptError(PromptRegistryError):
    """
    同一name和version被重复注册。
    """


class UnknownPromptNameError(PromptRegistryError):
    """
    Registry中不存在指定Prompt名称。
    """


class UnknownPromptVersionError(PromptRegistryError):
    """
    Prompt名称存在，但指定版本不存在。
    """


class PromptRegistry:
    """
    按name和version保存不可变PromptSpec。
    """

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], PromptSpec] = {}

    def register(
        self,
        spec: PromptSpec,
    ) -> None:
        """
        注册Prompt；相同name和version不允许覆盖。
        """
        key = (
            spec.name,
            spec.version,
        )

        if key in self._specs:
            raise DuplicatePromptError(
                f"Prompt已重复注册：{spec.name}:{spec.version}"
            )

        self._specs[key] = spec

    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> PromptSpec:
        """
        获取Prompt；未传version时固定使用v1。
        """
        resolved_version = version or "v1"
        known_versions = {
            registered_version
            for registered_name, registered_version in self._specs
            if registered_name == name
        }

        if not known_versions:
            raise UnknownPromptNameError(
                f"未知Prompt名称：{name}"
            )

        key = (
            name,
            resolved_version,
        )

        if key not in self._specs:
            raise UnknownPromptVersionError(
                f"Prompt {name} 不存在版本：{resolved_version}"
            )

        return self._specs[key]


prompt_registry = PromptRegistry()


def _load_template(
    relative_path: str,
) -> str:
    """
    从prompts目录读取UTF-8模板文本。
    """
    template_path = (
        Path(__file__).resolve().parent
        / relative_path
    )
    return template_path.read_text(
        encoding="utf-8",
    )


prompt_registry.register(
    PromptSpec(
        name="generation.design_characters",
        version="v1",
        template=_load_template(
            "generation/design_characters.v1.txt"
        ),
        required_variables=(
            "user_idea",
            "genre",
            "core_theme",
            "visual_style",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="generation.analyze_brief",
        version="v1",
        template=_load_template(
            "generation/analyze_brief.v1.txt"
        ),
        required_variables=(
            "user_idea",
            "memory_text",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="generation.plan_story",
        version="v1",
        template=_load_template(
            "generation/plan_story.v1.txt"
        ),
        required_variables=(
            "user_idea",
            "genre",
            "core_theme",
            "visual_style",
            "target_duration_sec",
            "characters_json",
            "story_memory_text",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="generation.write_scenes",
        version="v1",
        template=_load_template(
            "generation/write_scenes.v1.txt"
        ),
        required_variables=(
            "target_duration_sec",
            "user_idea",
            "story_outline_json",
            "characters_json",
            "scene_memory_text",
            "recommended_scene_count",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="review.story",
        version="v1",
        template=_load_template(
            "review/story.v1.txt"
        ),
        required_variables=(
            "film_brief",
            "user_idea",
            "characters",
            "story_outline",
            "story_memory_text",
            "history_issues_text",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="review.scene",
        version="v1",
        template=_load_template(
            "review/scene.v1.txt"
        ),
        required_variables=(
            "film_brief",
            "user_idea",
            "characters",
            "story_outline",
            "scenes",
            "scene_memory_text",
            "history_issues_text",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="revision.story",
        version="v1",
        template=_load_template(
            "revision/story.v1.txt"
        ),
        required_variables=(
            "film_brief",
            "characters",
            "current_story_outline",
            "story_memory_text",
            "human_feedback_text",
            "story_review_issues",
            "story_review_suggestions",
            "active_history_issues_text",
            "resolved_history_reminders_text",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="revision.scene",
        version="v1",
        template=_load_template(
            "revision/scene.v1.txt"
        ),
        required_variables=(
            "film_brief",
            "characters",
            "story_outline",
            "current_scene_json",
            "scene_memory_text",
            "human_feedback_text",
            "scene_review_issues",
            "scene_review_suggestions",
            "active_history_issues_text",
            "resolved_history_reminders_text",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="memory.candidate_extraction",
        version="v1",
        template=_load_template(
            "memory/candidate_extraction.v1.txt"
        ),
        required_variables=(
            "user_idea",
            "current_memory_text",
            "recent_human_feedback_text",
        ),
    )
)

prompt_registry.register(
    PromptSpec(
        name="memory.conservative_verifier",
        version="v1",
        template=_load_template(
            "memory/conservative_verifier.v1.txt"
        ),
        required_variables=(
            "candidate_text",
        ),
    )
)


def get_prompt(
    name: str,
    version: str | None = None,
) -> PromptSpec:
    """
    从项目默认Registry获取PromptSpec。
    """
    return prompt_registry.get(
        name,
        version,
    )
