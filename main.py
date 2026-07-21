import json

from state import FilmState
from nodes import analyze_brief, design_characters, plan_story, write_scenes
from memory.retrieve_memory import retrieve_memory


def print_state(state: FilmState, stage: str):
    """
    打印当前state中的内容
    """

    print("\n" + "=" * 80)
    print(f"CURRENT STAGE: {stage}")
    print("=" * 80)

    for key, value in state.items():

        print(f"\n[{key}]")

        # Pydantic BaseModel
        if hasattr(value, "model_dump"):
            print(
                json.dumps(
                    value.model_dump(),
                    ensure_ascii=False,
                    indent=2
                )
            )

        # list[BaseModel]
        elif isinstance(value, list) and value:
            if hasattr(value[0], "model_dump"):
                print(
                    json.dumps(
                        [
                            item.model_dump()
                            for item in value
                        ],
                        ensure_ascii=False,
                        indent=2
                    )
                )
            else:
                print(value)

        else:
            print(value)



def run_pipeline(user_idea, user_id="demo_user_001"):

    # 初始state
    state: FilmState = {
        "user_id": user_id,
        "user_idea": user_idea,
        "story_revision_count": 0,
        "scene_revision_count": 0,
    }

    print_state(
        state,
        "INITIAL STATE"
    )


    # ==========================
    # Node 0: retrieve_memory
    # ==========================
    update = retrieve_memory(state)

    print("\n\nNode Output:")
    print(update)

    state.update(update)

    print_state(
        state,
        "AFTER retrieve_memory"
    )


    # ==========================
    # Node 1: analyze_brief
    # ==========================
    update = analyze_brief(state)

    print("\n\nNode Output:")
    print(update)

    state.update(update)

    print_state(
        state,
        "AFTER analyze_brief"
    )


    # ==========================
    # Node 2: design_characters
    # ==========================
    update = design_characters(state)

    print("\n\nNode Output:")
    print(update)

    state.update(update)

    print_state(
        state,
        "AFTER design_characters"
    )


    # ==========================
    # Node 3: plan_story
    # ==========================
    update = plan_story(state)

    print("\n\nNode Output:")
    print(update)

    state.update(update)

    print_state(
        state,
        "AFTER plan_story"
    )


    # ==========================
    # Node 4: write_scenes
    # ==========================
    update = write_scenes(state)

    print("\n\nNode Output:")
    print(update)

    state.update(update)

    print_state(
        state,
        "AFTER write_scenes"
    )


    return state



if __name__ == "__main__":

    final_state = run_pipeline(
        user_idea="毕业季同学们一起拍摄毕业照，面临分别，依依不舍，生成60秒青春校园影片",
        user_id="demo_user_001",
    )

    print("\n\nPipeline Finished!")
