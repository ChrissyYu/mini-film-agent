import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import (
    Character,
    FilmBrief,
    ReviewIssueCheck,
    Scene,
    SceneCriticResult,
    StoryCriticResult,
    StoryOutline,
)

review_story_module = importlib.import_module("reviews.review_story")
review_scene_module = importlib.import_module("reviews.review_scene")


class FakeStructuredLLM:
    """
    捕获Prompt并返回预设审核结果，避免调用真实LLM。
    """

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.result


def _character() -> Character:
    """
    构造测试角色。
    """
    return Character(
        name="林夏",
        role="主角",
        appearance="白衬衫",
        personality=["克制"],
        motivation="面对告别",
        continuity_constraints=[],
    )


def _film_brief() -> FilmBrief:
    """
    构造测试影片需求。
    """
    return FilmBrief(
        target_duration_sec=10,
        genre="校园",
        core_theme="成长",
        visual_style="自然克制",
        recommended_scene_count=1,
    )


def _story_outline() -> StoryOutline:
    """
    构造测试故事大纲。
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
    构造测试分场。
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


def _story_state(story_review_history=None) -> dict:
    """
    构造Story Review所需最小State。
    """
    return {
        "film_brief": _film_brief(),
        "characters": [_character()],
        "story_outline": _story_outline(),
        "story_review_history": story_review_history or [],
        "story_revision_count": 1,
    }


def _scene_state(scene_review_history=None) -> dict:
    """
    构造Scene Review所需最小State。
    """
    return {
        "film_brief": _film_brief(),
        "characters": [_character()],
        "story_outline": _story_outline(),
        "scenes": [_scene()],
        "scene_review_history": scene_review_history or [],
        "scene_revision_count": 1,
    }


def test_story_historical_issue_can_be_marked_resolved(monkeypatch):
    """
    历史问题resolved时不应继续进入当前issues。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=["核心冲突不明确"],
            suggestions=[],
            historical_issue_checks=[
                ReviewIssueCheck(
                    issue="核心冲突不明确",
                    status="resolved",
                    evidence="当前故事已经写明离校与告别的冲突。",
                )
            ],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    result = review_story_module.review_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": ["核心冲突不明确"],
                    "suggestions": [],
                }
            ]
        )
    )

    review_result = result["story_review_result"]
    history_event = result["story_review_history"][0]

    assert review_result.passed is True
    assert review_result.issues == []
    assert review_result.historical_issue_checks[0].status == "resolved"
    assert history_event["historical_issue_checks"][0]["status"] == "resolved"


def test_story_unresolved_issue_is_forced_into_current_issues(monkeypatch):
    """
    unresolved历史问题必须进入当前issues，并使审核不通过。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
            historical_issue_checks=[
                ReviewIssueCheck(
                    issue="结尾没有回应冲突",
                    status="unresolved",
                    evidence="当前结尾仍未回应离校冲突。",
                )
            ],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    result = review_story_module.review_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": ["结尾没有回应冲突"],
                    "suggestions": [],
                }
            ]
        )
    )

    review_result = result["story_review_result"]

    assert review_result.passed is False
    assert "结尾没有回应冲突" in review_result.issues


def test_story_resolved_issue_can_regress(monkeypatch):
    """
    曾经resolved的问题再次出现时，应标记regressed并进入当前issues。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
            historical_issue_checks=[
                ReviewIssueCheck(
                    issue="人物动机前后矛盾",
                    status="regressed",
                    evidence="当前版本再次让角色做出违背动机的选择。",
                )
            ],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    result = review_story_module.review_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": ["人物动机前后矛盾"],
                    "suggestions": [],
                    "historical_issue_checks": [
                        {
                            "issue": "人物动机前后矛盾",
                            "status": "resolved",
                            "evidence": "上一轮已经修复。",
                        }
                    ],
                }
            ]
        )
    )

    review_result = result["story_review_result"]
    prompt = fake_llm.prompts[-1]

    assert review_result.passed is False
    assert "人物动机前后矛盾" in review_result.issues
    assert review_result.historical_issue_checks[0].status == "regressed"
    assert '"latest_status": "resolved"' in prompt


def test_no_history_returns_empty_historical_issue_checks(monkeypatch):
    """
    无历史问题时应正常返回空historical_issue_checks。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
            historical_issue_checks=[],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    result = review_story_module.review_story(
        _story_state()
    )

    assert "无历史问题" in fake_llm.prompts[-1]
    assert result["story_review_result"].historical_issue_checks == []
    assert result["story_review_history"][0]["historical_issue_checks"] == []


def test_old_review_history_without_checks_is_compatible(monkeypatch):
    """
    旧格式Review History缺少historical_issue_checks时仍可审核。
    """
    fake_llm = FakeStructuredLLM(
        StoryCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
            historical_issue_checks=[
                ReviewIssueCheck(
                    issue="旧格式问题",
                    status="resolved",
                    evidence="当前版本已处理。",
                )
            ],
        )
    )
    monkeypatch.setattr(
        review_story_module,
        "story_critic_llm",
        fake_llm,
    )

    result = review_story_module.review_story(
        _story_state(
            story_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": ["旧格式问题"],
                    "suggestions": [],
                }
            ]
        )
    )

    assert "旧格式问题" in fake_llm.prompts[-1]
    assert result["story_review_result"].passed is True


def test_scene_historical_issue_status_flow(monkeypatch):
    """
    Scene Review也应支持历史问题状态判断与当前issues联动。
    """
    fake_llm = FakeStructuredLLM(
        SceneCriticResult(
            passed=True,
            issues=[],
            suggestions=[],
            historical_issue_checks=[
                ReviewIssueCheck(
                    issue="场景动作不清晰",
                    status="unresolved",
                    evidence="当前action仍然缺少明确动作推进。",
                ),
                ReviewIssueCheck(
                    issue="视觉目标过于笼统",
                    status="resolved",
                    evidence="当前visual_goal已有明确叙事作用。",
                ),
            ],
        )
    )
    monkeypatch.setattr(
        review_scene_module,
        "scene_critic_llm",
        fake_llm,
    )

    result = review_scene_module.review_scene(
        _scene_state(
            scene_review_history=[
                {
                    "revision_round": 0,
                    "passed": False,
                    "issues": [
                        "场景动作不清晰",
                        "视觉目标过于笼统",
                    ],
                    "suggestions": [],
                }
            ]
        )
    )

    review_result = result["scene_review_result"]
    history_event = result["scene_review_history"][0]

    assert review_result.passed is False
    assert "场景动作不清晰" in review_result.issues
    assert "视觉目标过于笼统" not in review_result.issues
    assert history_event["historical_issue_checks"][0]["status"] == "unresolved"
    assert "historical_issue_checks" in fake_llm.prompts[-1]
