import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import extract_memory as extract_memory_module
from memory import store as memory_store
from memory.models import MemoryUpdate, UserMemory
from memory.retrieve_memory import retrieve_memory
from memory.update_memory import update_memory
from schemas import (
    Character,
    FilmBrief,
    Scene,
    SceneList,
    StoryOutline,
)

nodes_module = importlib.import_module("nodes")


class FakeMemoryUpdateLLM:
    """
    用预设MemoryUpdate替代真实LLM，测试仍走真实提取函数和Prompt组装。
    """

    def __init__(self, result: MemoryUpdate):
        self.result = result
        self.prompts = []

    def invoke(self, prompt: str) -> MemoryUpdate:
        self.prompts.append(prompt)
        return self.result


class FakeStructuredLLM:
    """
    捕获节点Prompt并返回结构化结果，避免调用真实网络。
    """

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.result


def _use_tmp_memory_dir(
    monkeypatch,
    tmp_path,
) -> None:
    """
    将Memory读写隔离到tmp_path，避免读写正式data/user_memory目录。
    """
    monkeypatch.setattr(
        memory_store,
        "MEMORY_DIR",
        tmp_path,
    )


def _film_brief() -> FilmBrief:
    """
    构造Story/Scene节点需要的影片需求。
    """
    return FilmBrief(
        target_duration_sec=10,
        genre="校园",
        core_theme="成长",
        visual_style="现实主义",
        recommended_scene_count=1,
    )


def _character() -> Character:
    """
    构造Story/Scene节点需要的角色。
    """
    return Character(
        name="林夏",
        role="主角",
        appearance="白衬衫",
        personality=["克制"],
        motivation="面对告别",
        continuity_constraints=[],
    )


def _story_outline() -> StoryOutline:
    """
    构造Scene节点需要的故事大纲。
    """
    return StoryOutline(
        setup="林夏毕业前夕准备离校。",
        conflict="她想留下却必须离开。",
        turning_point="她意识到离开也是成长。",
        ending="她平静告别校园。",
        theme="克制的成长。",
    )


def _scene() -> Scene:
    """
    构造Scene节点fake返回值。
    """
    return Scene(
        scene_id=1,
        duration_sec=10,
        location="校园",
        characters=["林夏"],
        action="林夏平静走出校门。",
        dialogue="",
        visual_goal="表现克制告别。",
    )


def _story_state(
    user_memory: UserMemory,
) -> dict:
    """
    构造下一次请求的Story节点State。
    """
    return {
        "user_id": user_memory.user_id,
        "user_idea": "再生成一个校园故事。",
        "user_memory": user_memory,
        "film_brief": _film_brief(),
        "characters": [_character()],
    }


def _scene_state(
    user_memory: UserMemory,
) -> dict:
    """
    构造下一次请求的Scene节点State。
    """
    return {
        "user_id": user_memory.user_id,
        "user_idea": "再生成一个校园故事。",
        "user_memory": user_memory,
        "film_brief": _film_brief(),
        "characters": [_character()],
        "story_outline": _story_outline(),
    }


def test_memory_saved_loaded_and_used_by_story_and_scene_prompts(
    monkeypatch,
    tmp_path,
):
    """
    验证长期Memory从提取、合并、保存、重新读取，到进入Story/Scene Prompt的完整链路。
    """
    _use_tmp_memory_dir(
        monkeypatch,
        tmp_path,
    )
    fake_memory_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=True,
            style_preferences_to_add=["现实主义"],
            disliked_elements_to_add=["大量旁白"],
            story_preferences_to_add=["故事结尾保持克制开放"],
            scene_preferences_to_add=["分场动作保持可拍摄"],
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_memory_llm,
    )

    first_memory = retrieve_memory(
        {
            "user_id": "memory_e2e_user",
        }
    )["user_memory"]

    result = update_memory(
        {
            "user_id": "memory_e2e_user",
            "user_idea": "以后保持现实主义，不要大量旁白。",
            "user_memory": first_memory,
            "human_feedback_history": [
                {
                    "scope": "story",
                    "decision": "revise",
                    "feedback": "以后故事结尾保持克制开放。",
                },
                {
                    "scope": "scene",
                    "decision": "revise",
                    "feedback": "以后分场动作保持可拍摄。",
                },
            ],
        }
    )

    assert result["memory_update_status"] == "saved"

    # 重新从JSON读取，模拟下一次独立请求使用已持久化Memory。
    loaded_memory = retrieve_memory(
        {
            "user_id": "memory_e2e_user",
        }
    )["user_memory"]

    assert loaded_memory.style_preferences == ["现实主义"]
    assert loaded_memory.disliked_elements == ["大量旁白"]
    assert loaded_memory.story_preferences == ["故事结尾保持克制开放"]
    assert loaded_memory.scene_preferences == ["分场动作保持可拍摄"]

    fake_story_llm = FakeStructuredLLM(
        _story_outline()
    )
    fake_scene_llm = FakeStructuredLLM(
        SceneList(scenes=[_scene()])
    )
    monkeypatch.setattr(
        nodes_module,
        "story_outline_llm",
        fake_story_llm,
    )
    monkeypatch.setattr(
        nodes_module,
        "write_scenes_llm",
        fake_scene_llm,
    )

    nodes_module.plan_story(
        _story_state(loaded_memory)
    )
    nodes_module.write_scenes(
        _scene_state(loaded_memory)
    )

    story_prompt = fake_story_llm.prompts[-1]
    scene_prompt = fake_scene_llm.prompts[-1]

    assert "现实主义" in story_prompt
    assert "大量旁白" in story_prompt
    assert "故事结尾保持克制开放" in story_prompt
    assert "分场动作保持可拍摄" not in story_prompt

    assert "现实主义" in scene_prompt
    assert "大量旁白" in scene_prompt
    assert "故事结尾保持克制开放" in scene_prompt
    assert "分场动作保持可拍摄" in scene_prompt


def test_task_specific_feedback_is_not_saved(
    monkeypatch,
    tmp_path,
):
    """
    当前人物、地点、场次等单次修改不应写入长期Memory。
    """
    _use_tmp_memory_dir(
        monkeypatch,
        tmp_path,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        FakeMemoryUpdateLLM(
            MemoryUpdate(
                should_update=False,
            )
        ),
    )

    result = update_memory(
        {
            "user_id": "task_specific_user",
            "user_idea": "这次生成一个校园故事。",
            "user_memory": UserMemory(user_id="task_specific_user"),
            "human_feedback_history": [
                {
                    "scope": "story",
                    "decision": "revise",
                    "feedback": "把林夏改成在图书馆遇到陈屿。",
                }
            ],
        }
    )

    assert result["memory_update_status"] == "skipped"
    assert not memory_store.get_memory_path(
        "task_specific_user"
    ).exists()


def test_duplicate_preference_is_not_rewritten(
    monkeypatch,
    tmp_path,
):
    """
    重复偏好经提取校正和真实合并后，不应重复写入，也不应重复保存。
    """
    _use_tmp_memory_dir(
        monkeypatch,
        tmp_path,
    )
    memory_store.save_user_memory(
        UserMemory(
            user_id="duplicate_user",
            story_preferences=["故事结尾保持克制开放"],
        )
    )
    existing_memory = memory_store.load_user_memory(
        "duplicate_user"
    )
    before_text = memory_store.get_memory_path(
        "duplicate_user"
    ).read_text(
        encoding="utf-8",
    )

    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        FakeMemoryUpdateLLM(
            MemoryUpdate(
                should_update=True,
                story_preferences_to_add=["故事结尾保持克制开放"],
            )
        ),
    )

    result = update_memory(
        {
            "user_id": "duplicate_user",
            "user_idea": "以后结尾继续克制开放。",
            "user_memory": existing_memory,
        }
    )

    after_text = memory_store.get_memory_path(
        "duplicate_user"
    ).read_text(
        encoding="utf-8",
    )

    assert result["memory_update_status"] == "skipped"
    assert result["user_memory"].story_preferences == [
        "故事结尾保持克制开放",
    ]
    assert after_text == before_text


def test_different_user_memory_is_isolated(
    monkeypatch,
    tmp_path,
):
    """
    不同user_id应读写不同JSON文件，互不影响。
    """
    _use_tmp_memory_dir(
        monkeypatch,
        tmp_path,
    )
    memory_store.save_user_memory(
        UserMemory(
            user_id="user_a",
            story_preferences=["现实主义故事"],
        )
    )
    memory_store.save_user_memory(
        UserMemory(
            user_id="user_b",
            scene_preferences=["荒诞喜剧分场"],
        )
    )

    user_a_memory = retrieve_memory(
        {
            "user_id": "user_a",
        }
    )["user_memory"]
    user_b_memory = retrieve_memory(
        {
            "user_id": "user_b",
        }
    )["user_memory"]

    assert user_a_memory.story_preferences == ["现实主义故事"]
    assert user_a_memory.scene_preferences == []
    assert user_b_memory.story_preferences == []
    assert user_b_memory.scene_preferences == ["荒诞喜剧分场"]


def test_machine_review_content_does_not_enter_memory_prompt(
    monkeypatch,
    tmp_path,
):
    """
    Memory提取Prompt只应包含用户输入和人工反馈，不应混入机器Review内容。
    """
    _use_tmp_memory_dir(
        monkeypatch,
        tmp_path,
    )
    fake_memory_llm = FakeMemoryUpdateLLM(
        MemoryUpdate(
            should_update=False,
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        fake_memory_llm,
    )

    result = update_memory(
        {
            "user_id": "review_noise_user",
            "user_idea": "这次生成一个校园故事。",
            "user_memory": UserMemory(user_id="review_noise_user"),
            "story_review_result": "机器Review不应进入Memory",
            "final_output": {
                "text": "final_output不应进入Memory",
            },
            "human_feedback_history": [
                {
                    "scope": "story",
                    "decision": "revise",
                    "feedback": "以后故事结尾更克制。",
                    "story_review_result": "嵌套机器Review也不应进入Memory",
                    "final_output": "嵌套final_output也不应进入Memory",
                }
            ],
        }
    )

    prompt = fake_memory_llm.prompts[-1]

    assert result["memory_update_status"] == "skipped"
    assert "以后故事结尾更克制" in prompt
    assert "机器Review不应进入Memory" not in prompt
    assert "final_output不应进入Memory" not in prompt
    assert "嵌套机器Review也不应进入Memory" not in prompt
    assert "嵌套final_output也不应进入Memory" not in prompt


def test_legacy_memory_json_is_compatible_and_backfilled(
    monkeypatch,
    tmp_path,
):
    """
    旧格式JSON缺少story/scene字段时仍可读取，保存后会补全新字段。
    """
    _use_tmp_memory_dir(
        monkeypatch,
        tmp_path,
    )
    memory_path = tmp_path / "legacy_user.json"
    memory_path.write_text(
        json.dumps(
            {
                "user_id": "legacy_user",
                "style_preferences": ["现实主义"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    user_memory = retrieve_memory(
        {
            "user_id": "legacy_user",
        }
    )["user_memory"]

    assert user_memory.story_preferences == []
    assert user_memory.scene_preferences == []

    memory_store.save_user_memory(
        user_memory,
    )
    saved_data = json.loads(
        memory_path.read_text(
            encoding="utf-8",
        )
    )

    assert saved_data["story_preferences"] == []
    assert saved_data["scene_preferences"] == []
