import json
from typing import Any


def format_sse_event(
    event_name: str,
    data: dict[str, Any],
) -> str:
    """
    将事件名称和数据字典格式化为标准SSE文本。
    """
    json_data = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )    # 使用紧凑JSON，避免SSE data中出现多余空白

    return (
        f"event: {event_name}\n"
        f"data: {json_data}\n"
        "\n"    # SSE事件必须以空行结束，也就是整体以\n\n收尾
    )
