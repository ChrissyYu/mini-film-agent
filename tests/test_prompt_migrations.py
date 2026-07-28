import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nodes
from memory.models import UserMemory
from schemas import (
    Character,
    FilmBrief,
    Scene,
    SceneCriticResult,
    SceneList,
    SceneReviewResult,
    StoryCriticResult,
    StoryOutline,
    StoryReviewResult,
)


review_story_module = importlib.import_module(
    "reviews.review_story"
)
review_scene_module = importlib.import_module(
    "reviews.review_scene"
)
revise_story_module = importlib.import_module(
    "revisions.revise_story"
)
revise_scene_module = importlib.import_module(
    "revisions.revise_scene"
)
extract_memory_module = importlib.import_module(
    "memory.extract_memory"
)


class CaptureLLM:
    """
    捕获最终发送文本，并返回预设structured output。
    """

    def __init__(
        self,
        result,
    ):
        self.result = result
        self.prompts = []

    def invoke(
        self,
        prompt: str,
    ):
        self.prompts.append(prompt)
        return self.result


def _film_brief() -> FilmBrief:
    return FilmBrief(
        target_duration_sec=30,
        genre="SENTINEL_GENRE",
        core_theme="SENTINEL_CORE_THEME",
        visual_style="SENTINEL_VISUAL_STYLE",
        recommended_scene_count=3,
    )


def _characters() -> list[Character]:
    return [
        Character(
            name="SENTINEL_CHARACTER",
            role="主角",
            appearance="SENTINEL_APPEARANCE",
            personality=["克制"],
            motivation="SENTINEL_MOTIVATION",
            continuity_constraints=[
                "SENTINEL_CONTINUITY",
            ],
        )
    ]


def _story_outline() -> StoryOutline:
    return StoryOutline(
        setup="SENTINEL_SETUP",
        conflict="SENTINEL_CONFLICT",
        turning_point="SENTINEL_TURNING_POINT",
        ending="SENTINEL_ENDING",
        theme="SENTINEL_STORY_THEME",
    )


def _scenes() -> list[Scene]:
    return [
        Scene(
            scene_id=1,
            duration_sec=30,
            location="SENTINEL_LOCATION",
            characters=[
                "SENTINEL_CHARACTER",
            ],
            action="SENTINEL_ACTION",
            dialogue="SENTINEL_DIALOGUE",
            visual_goal="SENTINEL_VISUAL_GOAL",
        )
    ]


def _memory() -> UserMemory:
    return UserMemory(
        user_id="prompt_test_user",
        style_preferences=[
            "SENTINEL_GLOBAL_MEMORY",
        ],
        story_preferences=[
            "SENTINEL_STORY_MEMORY",
        ],
        scene_preferences=[
            "SENTINEL_SCENE_MEMORY",
        ],
    )


def _assert_sentinels(
    prompt: str,
    *sentinels: str,
) -> None:
    assert "{{" not in prompt

    for sentinel in sentinels:
        assert sentinel in prompt


def test_analyze_brief_uses_registry_prompt(
    monkeypatch,
):
    fake_llm = CaptureLLM(
        _film_brief()
    )
    monkeypatch.setattr(
        nodes,
        "brief_llm",
        fake_llm,
    )

    result = nodes.analyze_brief(
        {
            "user_idea": "SENTINEL_USER_IDEA",
            "user_memory": _memory(),
        }
    )

    _assert_sentinels(
        fake_llm.prompts[-1],
        "SENTINEL_USER_IDEA",
        "SENTINEL_GLOBAL_MEMORY",
        "SENTINEL_STORY_MEMORY",
        "SENTINEL_SCENE_MEMORY",
    )
    assert (
        result["current_stage"]
        == "brief_completed"
    )


def test_plan_story_uses_registry_prompt(
    monkeypatch,
):
    fake_llm = CaptureLLM(
        _story_outline()
    )
    monkeypatch.setattr(
        nodes,
        "story_outline_llm",
        fake_llm,
    )

    result = nodes.plan_story(
        {
            "user_idea": "SENTINEL_USER_IDEA",
            "film_brief": _film_brief(),
            "characters": _characters(),
            "user_memory": _memory(),
        }
    )

    _assert_sentinels(
        fake_llm.prompts[-1],
        "SENTINEL_USER_IDEA",
        "SENTINEL_GENRE",
        "SENTINEL_CORE_THEME",
        "SENTINEL_VISUAL_STYLE",
        "SENTINEL_CHARACTER",
        "SENTINEL_STORY_MEMORY",
    )
    assert (
        result["current_stage"]
        == "story_outline_completed"
    )


def test_write_scenes_uses_registry_prompt(
    monkeypatch,
):
    fake_llm = CaptureLLM(
        SceneList(
            scenes=_scenes()
        )
    )
    monkeypatch.setattr(
        nodes,
        "write_scenes_llm",
        fake_llm,
    )

    result = nodes.write_scenes(
        {
            "user_idea": "SENTINEL_USER_IDEA",
            "film_brief": _film_brief(),
            "characters": _characters(),
            "story_outline": _story_outline(),
            "user_memory": _memory(),
        }
    )

    _assert_sentinels(
        fake_llm.prompts[-1],
        "SENTINEL_USER_IDEA",
        "SENTINEL_SETUP",
        "SENTINEL_CHARACTER",
        "SENTINEL_STORY_MEMORY",
        "SENTINEL_SCENE_MEMORY",
    )
    assert (
        result["current_stage"]
        == "scenes_completed"
    )


def test_story_review_uses_registry_prompt(
    monkeypatch,
):
    fake_llm = CaptureLLM(
        StoryCriticResult(
            passed=True,
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    review_story_module.llm_review_story(
        {
            "user_idea": "SENTINEL_USER_IDEA",
            "film_brief": _film_brief(),
            "characters": _characters(),
            "story_outline": _story_outline(),
            "user_memory": _memory(),
            "story_review_history": [
                {
                    "issues": [
                        "SENTINEL_STORY_HISTORY",
                    ],
                }
            ],
        }
    )

    _assert_sentinels(
        fake_llm.prompts[-1],
        "SENTINEL_USER_IDEA",
        "SENTINEL_GENRE",
        "SENTINEL_CHARACTER",
        "SENTINEL_SETUP",
        "SENTINEL_STORY_MEMORY",
        "SENTINEL_STORY_HISTORY",
    )


def test_scene_review_uses_registry_prompt(
    monkeypatch,
):
    fake_llm = CaptureLLM(
        SceneCriticResult(
            passed=True,
        )
    )
    monkeypatch.setattr(
        review_scene_module,
        "scene_critic_llm",
        fake_llm,
    )

    review_scene_module.llm_review(
        {
            "user_idea": "SENTINEL_USER_IDEA",
            "film_brief": _film_brief(),
            "characters": _characters(),
            "story_outline": _story_outline(),
            "scenes": _scenes(),
            "user_memory": _memory(),
            "scene_review_history": [
                {
                    "issues": [
                        "SENTINEL_SCENE_HISTORY",
                    ],
                }
            ],
        }
    )

    _assert_sentinels(
        fake_llm.prompts[-1],
        "SENTINEL_USER_IDEA",
        "SENTINEL_CHARACTER",
        "SENTINEL_SETUP",
        "SENTINEL_ACTION",
        "SENTINEL_STORY_MEMORY",
        "SENTINEL_SCENE_MEMORY",
        "SENTINEL_SCENE_HISTORY",
    )


def test_story_revision_uses_registry_prompt(
    monkeypatch,
):
    fake_llm = CaptureLLM(
        _story_outline()
    )
    monkeypatch.setattr(
        revise_story_module,
        "story_revise_llm",
        fake_llm,
    )

    result = revise_story_module.revise_story(
        {
            "film_brief": _film_brief(),
            "characters": _characters(),
            "story_outline": _story_outline(),
            "story_review_result": StoryReviewResult(
                passed=False,
                issues=[
                    "SENTINEL_STORY_ISSUE",
                ],
                suggestions=[
                    "SENTINEL_STORY_SUGGESTION",
                ],
            ),
            "story_review_history": [
                {
                    "issues": [
                        "SENTINEL_ACTIVE_STORY_HISTORY",
                    ],
                    "historical_issue_checks": [
                        {
                            "issue": "SENTINEL_RESOLVED_STORY_HISTORY",
                            "status": "resolved",
                            "evidence": "已解决",
                        }
                    ],
                }
            ],
            "human_feedback": "SENTINEL_HUMAN_FEEDBACK",
            "user_memory": _memory(),
            "story_revision_count": 0,
        }
    )

    _assert_sentinels(
        fake_llm.prompts[-1],
        "SENTINEL_GENRE",
        "SENTINEL_CHARACTER",
        "SENTINEL_SETUP",
        "SENTINEL_STORY_MEMORY",
        "SENTINEL_HUMAN_FEEDBACK",
        "SENTINEL_STORY_ISSUE",
        "SENTINEL_STORY_SUGGESTION",
        "SENTINEL_ACTIVE_STORY_HISTORY",
        "SENTINEL_RESOLVED_STORY_HISTORY",
    )
    assert result["story_revision_count"] == 1


def test_scene_revision_uses_registry_prompt(
    monkeypatch,
):
    fake_llm = CaptureLLM(
        SceneList(
            scenes=_scenes()
        )
    )
    monkeypatch.setattr(
        revise_scene_module,
        "scene_revise_llm",
        fake_llm,
    )

    result = revise_scene_module.revise_scene(
        {
            "film_brief": _film_brief(),
            "characters": _characters(),
            "story_outline": _story_outline(),
            "scenes": _scenes(),
            "scene_review_result": SceneReviewResult(
                passed=False,
                issues=[
                    "SENTINEL_SCENE_ISSUE",
                ],
                suggestions=[
                    "SENTINEL_SCENE_SUGGESTION",
                ],
            ),
            "scene_review_history": [
                {
                    "issues": [
                        "SENTINEL_ACTIVE_SCENE_HISTORY",
                    ],
                    "historical_issue_checks": [
                        {
                            "issue": "SENTINEL_RESOLVED_SCENE_HISTORY",
                            "status": "resolved",
                            "evidence": "已解决",
                        }
                    ],
                }
            ],
            "human_feedback": "SENTINEL_HUMAN_FEEDBACK",
            "user_memory": _memory(),
            "scene_revision_count": 0,
        }
    )

    _assert_sentinels(
        fake_llm.prompts[-1],
        "SENTINEL_GENRE",
        "SENTINEL_CHARACTER",
        "SENTINEL_SETUP",
        "SENTINEL_ACTION",
        "SENTINEL_STORY_MEMORY",
        "SENTINEL_SCENE_MEMORY",
        "SENTINEL_HUMAN_FEEDBACK",
        "SENTINEL_SCENE_ISSUE",
        "SENTINEL_SCENE_SUGGESTION",
        "SENTINEL_ACTIVE_SCENE_HISTORY",
        "SENTINEL_RESOLVED_SCENE_HISTORY",
    )
    assert result["scene_revision_count"] == 1


def test_memory_candidate_and_verifier_use_registry_prompts(
    monkeypatch,
):
    candidate = extract_memory_module.MemoryCandidate(
        field="story_preferences_to_add",
        value="SENTINEL_MEMORY_VALUE",
        source="user_idea",
        evidence="SENTINEL_EVIDENCE",
        claim_type="explicit_preference",
    )
    candidate_llm = CaptureLLM(
        extract_memory_module.MemoryCandidateBatch(
            candidates=[
                candidate,
            ]
        )
    )
    verifier_llm = CaptureLLM(
        extract_memory_module.MemoryCandidateVerification(
            decisions=[
                extract_memory_module.MemoryCandidateDecisionItem(
                    **candidate.model_dump(
                        mode="json",
                    ),
                    decision="REJECT",
                )
            ]
        )
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_candidate_llm",
        candidate_llm,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_verifier_llm",
        verifier_llm,
    )
    monkeypatch.setattr(
        extract_memory_module,
        "memory_update_llm",
        None,
    )

    result = extract_memory_module.extract_memory_update(
        user_idea=(
            "SENTINEL_USER_IDEA SENTINEL_EVIDENCE"
        ),
        current_memory=_memory(),
        human_feedback_history=[
            {
                "scope": "story",
                "decision": "revise",
                "feedback": "SENTINEL_HUMAN_FEEDBACK",
            }
        ],
    )

    _assert_sentinels(
        candidate_llm.prompts[-1],
        "SENTINEL_USER_IDEA",
        "SENTINEL_EVIDENCE",
        "SENTINEL_GLOBAL_MEMORY",
        "SENTINEL_STORY_MEMORY",
        "SENTINEL_SCENE_MEMORY",
        "SENTINEL_HUMAN_FEEDBACK",
    )
    candidate_json = (
        extract_memory_module
        ._candidate_batch_to_json(
            [
                candidate,
            ]
        )
    )
    verifier_prompt = (
        verifier_llm.prompts[-1]
    )

    assert candidate_json in verifier_prompt
    assert "SENTINEL_MEMORY_VALUE" in verifier_prompt
    assert "SENTINEL_EVIDENCE" in verifier_prompt
    assert "{{" not in verifier_prompt
    assert result.should_update is False
