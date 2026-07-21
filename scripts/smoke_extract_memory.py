import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.extract_memory import extract_memory_update
from memory.models import UserMemory


def print_memory_update(
    label: str,
    user_idea: str,
) -> None:
    """
    调用长期偏好提取函数，并打印结构化结果。
    """
    current_memory = UserMemory(
        user_id="demo_user_001",
    )

    update = extract_memory_update(
        user_idea=user_idea,
        current_memory=current_memory,
    )

    print(f"\n{label}")
    print(
        json.dumps(
            update.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    print_memory_update(
        label="输入A",
        user_idea="以后生成的短片都尽量采用现实主义风格，不要大量旁白。",
    )

    print_memory_update(
        label="输入B",
        user_idea="这次生成一个60秒的校园悬疑故事。",
    )
