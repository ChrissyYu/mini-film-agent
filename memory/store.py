import json
import re
from pathlib import Path

from pydantic import ValidationError

from memory.models import UserMemory

# 负责 Memory 的持久化读写
# 项目根目录：
# mini_film_agent/memory/store.py
# parent = memory
# parent.parent = mini_film_agent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEMORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "user_memory"
)

USER_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{1,64}$"
)


class MemoryStoreError(RuntimeError):
    """
    Memory文件读取或写入失败。
    """


def validate_user_id(
    user_id: str,
) -> None:
    """
    限制user_id只能包含安全字符，
    避免user_id被用来构造任意文件路径。
    """
    if not USER_ID_PATTERN.fullmatch(user_id):
        raise ValueError(
            "user_id只能包含字母、数字、"
            "下划线和连字符，长度为1至64。"
        )


def get_memory_path(
    user_id: str,
) -> Path:
    """
    根据user_id得到对应的Memory文件路径。
    """
    validate_user_id(user_id)

    return MEMORY_DIR / f"{user_id}.json"


def load_user_memory(
    user_id: str,
) -> UserMemory:
    """
    加载指定用户的长期记忆。

    如果该用户尚无Memory文件，
    返回一份空的UserMemory。
    """
    memory_path = get_memory_path(user_id)

    if not memory_path.exists():
        return UserMemory(
            user_id=user_id,
        )

    try:
        raw_text = memory_path.read_text(
            encoding="utf-8",
        )

        raw_data = json.loads(raw_text)

        user_memory = UserMemory.model_validate(
            raw_data
        )

    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
    ) as exc:
        raise MemoryStoreError(
            f"读取用户Memory失败：{memory_path}"
        ) from exc

    # 防止文件名和文件内容中的user_id不一致
    if user_memory.user_id != user_id:
        raise MemoryStoreError(
            "Memory文件中的user_id与"
            "请求的user_id不一致。"
        )

    return user_memory


def save_user_memory(
    user_memory: UserMemory,
) -> None:
    """
    将用户长期记忆保存为JSON文件。

    当前采用临时文件替换方式，
    避免写入中途失败导致原文件损坏。
    """
    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_path = get_memory_path(
        user_memory.user_id
    )

    temporary_path = memory_path.with_suffix(
        ".tmp"
    )

    memory_json = json.dumps(
        user_memory.model_dump(
            mode="json",
        ),
        ensure_ascii=False,
        indent=2,
    )

    try:
        temporary_path.write_text(
            memory_json,
            encoding="utf-8",
        )

        temporary_path.replace(
            memory_path
        )

    except OSError as exc:
        raise MemoryStoreError(
            f"保存用户Memory失败：{memory_path}"
        ) from exc