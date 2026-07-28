import re
from typing import Any

from prompts.models import PromptSpec, RenderedPrompt
from prompts.registry import PromptRegistry, prompt_registry


class PromptRenderError(ValueError):
    """
    Prompt变量契约或插值结果不合法。
    """


_VARIABLE_PATTERN = re.compile(
    r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}"
)


class PromptRenderer:
    """
    只替换{{variable}}占位符，普通JSON大括号保持原样。
    """

    def __init__(
        self,
        registry: PromptRegistry,
    ) -> None:
        self._registry = registry

    def render(
        self,
        name: str,
        version: str | None = None,
        **variables: Any,
    ) -> RenderedPrompt:
        """
        校验变量集合并渲染指定版本的Prompt。
        """
        spec = self._registry.get(
            name,
            version,
        )
        required_variables = set(
            spec.required_variables
        )
        provided_variables = set(
            variables
        )
        missing_variables = (
            required_variables
            - provided_variables
        )
        extra_variables = (
            provided_variables
            - required_variables
        )

        if missing_variables:
            raise PromptRenderError(
                "Prompt缺少变量："
                + ", ".join(
                    sorted(missing_variables)
                )
            )

        if extra_variables:
            raise PromptRenderError(
                "Prompt存在多余变量："
                + ", ".join(
                    sorted(extra_variables)
                )
            )

        template_variables = set(
            _VARIABLE_PATTERN.findall(
                spec.template
            )
        )
        undeclared_variables = (
            template_variables
            - required_variables
        )
        unused_required_variables = (
            required_variables
            - template_variables
        )

        if undeclared_variables:
            raise PromptRenderError(
                "模板包含未声明变量："
                + ", ".join(
                    sorted(undeclared_variables)
                )
            )

        if unused_required_variables:
            raise PromptRenderError(
                "required_variables未出现在模板中："
                + ", ".join(
                    sorted(unused_required_variables)
                )
            )

        replaced_variables = set()

        def replace_variable(
            match: re.Match,
        ) -> str:
            variable_name = match.group(1)
            replaced_variables.add(
                variable_name
            )
            return str(
                variables[variable_name]
            )

        text = _VARIABLE_PATTERN.sub(
            replace_variable,
            spec.template,
        )
        unreplaced_variables = (
            required_variables
            - replaced_variables
        )

        if unreplaced_variables:
            raise PromptRenderError(
                "Prompt渲染后仍有占位符："
                + ", ".join(
                    sorted(
                        unreplaced_variables
                    )
                )
            )

        return RenderedPrompt(
            text=text,
            name=spec.name,
            version=spec.version,
            chars=len(text),
        )


prompt_renderer = PromptRenderer(
    prompt_registry
)


def render_prompt(
    name: str,
    version: str | None = None,
    **variables: Any,
) -> RenderedPrompt:
    """
    使用项目默认Registry渲染Prompt。
    """
    return prompt_renderer.render(
        name,
        version,
        **variables,
    )
