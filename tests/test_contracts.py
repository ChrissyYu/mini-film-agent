from graph import (
    MAX_SCENE_REVISIONS,
    MAX_STORY_REVISIONS,
    build_graph,
    route_after_scene_review,
    route_after_story_review,
)
from schemas import (
    SceneCriticResult,
    SceneReviewResult,
    StoryCriticResult,
    StoryReviewResult,
)


# ============================================================
# Review / Critic Schema 契约测试
# ============================================================

def test_story_critic_result_fields():
    """
    检查 StoryCriticResult 是否包含审核所需字段。
    """

    result = StoryCriticResult(
        passed=True,
        issues=[],
        suggestions=[],
    )

    assert result.passed is True
    assert result.issues == []
    assert result.suggestions == []


def test_story_review_result_fields():
    """
    检查 StoryReviewResult 是否包含审核所需字段。
    """

    result = StoryReviewResult(
        passed=False,
        issues=["故事转折缺少因果关系"],
        suggestions=["增加明确的转折触发事件"],
    )

    assert result.passed is False
    assert result.issues == ["故事转折缺少因果关系"]
    assert result.suggestions == ["增加明确的转折触发事件"]


def test_scene_critic_result_fields():
    """
    检查 SceneCriticResult 是否包含审核所需字段。

    该测试可以提前发现：
    SceneCriticResult 缺少 passed 等字段的问题。
    """

    result = SceneCriticResult(
        passed=True,
        issues=[],
        suggestions=[],
    )

    assert result.passed is True
    assert result.issues == []
    assert result.suggestions == []


def test_scene_review_result_fields():
    """
    检查 SceneReviewResult 是否包含审核所需字段。
    """

    result = SceneReviewResult(
        passed=False,
        issues=["场景总时长不等于目标时长"],
        suggestions=["重新调整各场景时长"],
    )

    assert result.passed is False
    assert result.issues == ["场景总时长不等于目标时长"]
    assert result.suggestions == ["重新调整各场景时长"]


# ============================================================
# Story 路由测试
# ============================================================

def test_story_route_when_review_passed():
    """
    故事审核通过时，应进入 write_scenes。
    """

    state = {
        "story_review_result": StoryReviewResult(
            passed=True,
            issues=[],
            suggestions=[],
        ),
        "story_revision_count": 0,
    }

    result = route_after_story_review(state)

    assert result == "write_scenes"


def test_story_route_when_review_failed():
    """
    故事审核未通过且未达到修订上限时，
    应进入 revise_story。
    """

    state = {
        "story_review_result": StoryReviewResult(
            passed=False,
            issues=["核心冲突不明确"],
            suggestions=["强化角色之间的核心分歧"],
        ),
        "story_revision_count": 0,
    }

    result = route_after_story_review(state)

    assert result == "revise_story"


def test_story_route_when_review_failed_after_one_revision():
    """
    故事已经修改过一次，但还没有达到修订上限时，
    仍应进入 revise_story。
    """

    state = {
        "story_review_result": StoryReviewResult(
            passed=False,
            issues=["转折仍然不够清晰"],
            suggestions=["补充具体触发事件"],
        ),
        "story_revision_count": 1,
    }

    result = route_after_story_review(state)

    assert result == "revise_story"


def test_story_route_when_revision_limit_reached():
    """
    故事审核未通过，但已经达到最大修订次数时，
    应停止继续修改，进入 write_scenes。
    """

    state = {
        "story_review_result": StoryReviewResult(
            passed=False,
            issues=["故事仍存在部分问题"],
            suggestions=[],
        ),
        "story_revision_count": MAX_STORY_REVISIONS,
    }

    result = route_after_story_review(state)

    assert result == "write_scenes"


# ============================================================
# Scene 路由测试
# ============================================================

def test_scene_route_when_review_passed():
    """
    分场审核通过时，应进入 finalize。
    """

    state = {
        "scene_review_result": SceneReviewResult(
            passed=True,
            issues=[],
            suggestions=[],
        ),
        "scene_revision_count": 0,
    }

    result = route_after_scene_review(state)

    assert result == "finalize"


def test_scene_route_when_review_failed():
    """
    分场审核未通过且未达到修订上限时，
    应进入 revise_scene。
    """

    state = {
        "scene_review_result": SceneReviewResult(
            passed=False,
            issues=["场景时长分配不合理"],
            suggestions=["重新分配各场景时长"],
        ),
        "scene_revision_count": 0,
    }

    result = route_after_scene_review(state)

    assert result == "revise_scene"


def test_scene_route_when_review_failed_after_one_revision():
    """
    分场已经修改过一次，但还没有达到修订上限时，
    仍应进入 revise_scene。
    """

    state = {
        "scene_review_result": SceneReviewResult(
            passed=False,
            issues=["仍有场景出现未定义角色"],
            suggestions=["删除未定义角色"],
        ),
        "scene_revision_count": 1,
    }

    result = route_after_scene_review(state)

    assert result == "revise_scene"


def test_scene_route_when_revision_limit_reached():
    """
    分场审核未通过，但已经达到最大修订次数时，
    应停止继续修改，进入 finalize。
    """

    state = {
        "scene_review_result": SceneReviewResult(
            passed=False,
            issues=["分场仍然存在部分问题"],
            suggestions=[],
        ),
        "scene_revision_count": MAX_SCENE_REVISIONS,
    }

    result = route_after_scene_review(state)

    assert result == "finalize"


# ============================================================
# Graph 编译测试
# ============================================================

def test_graph_can_compile():
    """
    检查 Graph 是否可以正常构建和编译。
    """

    graph = build_graph()

    assert graph is not None
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")


def test_compiled_graph_exposes_expected_nodes():
    """
    检查编译后的 Graph 是否包含预期节点。

    注意：
    get_graph() 返回的是 Graph 的结构描述，
    不会执行任何节点，也不会调用 LLM。
    """

    graph = build_graph()
    graph_structure = graph.get_graph()

    node_names = set(graph_structure.nodes.keys())

    expected_nodes = {
        "__start__",
        "retrieve_memory",
        "analyze_brief",
        "design_characters",
        "plan_story",
        "review_story",
        "revise_story",
        "write_scenes",
        "review_scene",
        "revise_scene",
        "finalize",
        "update_memory",
        "__end__",
    }

    assert expected_nodes.issubset(node_names)
