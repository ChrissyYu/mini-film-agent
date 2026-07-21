import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.models import UserMemory
from memory.store import save_user_memory


demo_memory = UserMemory(
    user_id="demo_user_001",
    preferred_genres=[
        "青春片",
        "现实主义剧情片",
    ],
    style_preferences=[
        "自然克制的情感表达",
        "避免过度煽情",
        "生活化人物关系",
    ],
    disliked_elements=[
        "大量旁白",
        "悬浮的青春疼痛文学",
    ],
    preferred_duration_sec=60,
)

save_user_memory(
    demo_memory
)

print("Demo Memory保存成功。")