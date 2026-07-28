import json

from llm_profiles.factory import create_structured_llm
from memory.context import format_scene_memory_context, format_story_memory_context
from observability.llm_calls import (
    collect_llm_call_trace,
    invoke_structured_llm,
)
from prompts.renderer import render_prompt
from state import FilmState
from schemas import FilmBrief, CharacterList, StoryOutline, Character, SceneList, Scene

#=============== 配置 structured LLM ===============
brief_llm = create_structured_llm(
    "generation.analyze_brief",
    FilmBrief,
)
character_llm = create_structured_llm(
    "generation.design_characters",
    CharacterList,
)
story_outline_llm = create_structured_llm(
    "generation.plan_story",
    StoryOutline,
)
write_scenes_llm = create_structured_llm(
    "generation.write_scenes",
    SceneList,
)

#=============== nodes函数定义 ===============
# 分析创意
def analyze_brief(state: FilmState) -> dict:
    user_idea = state["user_idea"]
    user_memory = state["user_memory"]

    if user_memory is None:
        memory_text = "{}" # 如果用户记忆为空，则返回空字符串
    else:
        memory_text = user_memory.model_dump_json(
            ensure_ascii=False,
            indent=2,
        )

    rendered = render_prompt(
        "generation.analyze_brief",
        version="v1",
        user_idea=user_idea,
        memory_text=memory_text,
    )
    with collect_llm_call_trace() as llm_call_events:
        film_brief: FilmBrief = invoke_structured_llm(
            brief_llm,
            rendered,
            node="analyze_brief",
        )

    return {
        "film_brief": film_brief,
        "current_stage": "brief_completed",
        "llm_call_trace": llm_call_events,
    }


# 设计角色
def design_characters(state: FilmState) -> dict:
    user_idea = state["user_idea"]
    film_brief = state["film_brief"]
    genre = film_brief.genre
    core_theme = film_brief.core_theme
    visual_style = film_brief.visual_style

    rendered = render_prompt(
        "generation.design_characters",
        version="v1",
        user_idea=user_idea,
        genre=genre,
        core_theme=core_theme,
        visual_style=visual_style,
    )
    with collect_llm_call_trace() as llm_call_events:
        characters_result: CharacterList = invoke_structured_llm(
            character_llm,
            rendered,
            node="design_characters",
        )

    return {
        "characters": characters_result.characters,
        "current_stage": "characters_completed",
        "llm_call_trace": llm_call_events,
    }


# 设计故事大纲
def plan_story(state: FilmState) -> dict:
    user_idea = state["user_idea"]
    film_brief = state["film_brief"]
    story_memory_text = format_story_memory_context(
        state.get("user_memory")
    )    # Story阶段统一使用过滤后的长期Memory，不读取scene_preferences

    target_duration_sec = film_brief.target_duration_sec
    genre = film_brief.genre
    core_theme = film_brief.core_theme
    visual_style = film_brief.visual_style
    
    characters_data = [character.model_dump() for character in state["characters"]] # 将list格式的characters，每个单独取出，转换为json格式
    characters_json = json.dumps(characters_data, ensure_ascii=False, indent=2)
    
    rendered = render_prompt(
        "generation.plan_story",
        version="v1",
        user_idea=user_idea,
        genre=genre,
        core_theme=core_theme,
        visual_style=visual_style,
        target_duration_sec=target_duration_sec,
        characters_json=characters_json,
        story_memory_text=story_memory_text,
    )
    with collect_llm_call_trace() as llm_call_events:
        story_outline: StoryOutline = invoke_structured_llm(
            story_outline_llm,
            rendered,
            node="plan_story",
        )

    return {
        "story_outline": story_outline,
        "current_stage": "story_outline_completed",
        "llm_call_trace": llm_call_events,
    }


# 设计分镜场景规划
def write_scenes(state: FilmState) -> dict:
    user_idea = state["user_idea"]
    film_brief = state["film_brief"]
    story_outline = state["story_outline"]
    scene_memory_text = format_scene_memory_context(
        state.get("user_memory")
    )    # Scene阶段同时参考故事偏好和分场偏好，但当前任务始终优先

    target_duration_sec = film_brief.target_duration_sec

    characters_data_simple = [{"name": character.name,"appearance": character.appearance} for character in state["characters"]]
    characters_json = json.dumps(characters_data_simple, ensure_ascii=False, indent=2) # 将characters转换为json格式

    story_outline_json = json.dumps(story_outline.model_dump(), ensure_ascii=False, indent=2) # 将story_outline转换为json格式

    rendered = render_prompt(
        "generation.write_scenes",
        version="v1",
        target_duration_sec=target_duration_sec,
        user_idea=user_idea,
        story_outline_json=story_outline_json,
        characters_json=characters_json,
        scene_memory_text=scene_memory_text,
        recommended_scene_count=(
            film_brief.recommended_scene_count
        ),
    )
    with collect_llm_call_trace() as llm_call_events:
        scenes_result: SceneList = invoke_structured_llm(
            write_scenes_llm,
            rendered,
            node="write_scenes",
        )

    return {
        "scenes": scenes_result.scenes,
        "current_stage": "scenes_completed",
        "llm_call_trace": llm_call_events,
    }



#=============== 测试代码 ===============
# 测试分析创意功能
"""
if __name__ == "__main__":
    state: FilmState = {
        "user_idea": "毕业季同学们一起拍摄毕业照，面临分别，依依不舍，生成一段60秒的青春校园影片",
        "scene_revision_count": 0,
        "story_revision_count": 0,
    }
    result = analyze_brief(state)
    print(
        json.dumps(
            {
                "film_brief": result["film_brief"].model_dump()
            },
            ensure_ascii=False,
            indent=2
        )
    )
"""

# 测试设计角色功能
"""
if __name__ == "__main__":
    state: FilmState = {
        "user_idea": "毕业季同学们一起拍摄毕业照，面临分别，依依不舍，生成一段60秒的青春校园影片",
        "scene_revision_count": 0,
        "story_revision_count": 0,
        genre: "青春校园",
        core_theme: "青春告别与分别不舍",
        visual_style: "自然光主导，柔焦+轻微胶片颗粒感；色调以清透青橙色系为主（晨光/午后暖调）；构图强调群像互动与留白，多用中景跟拍与低角度仰拍增强青春张力；穿插0.5秒内快切特写（系松动的领带、攥紧的毕业册边角、飘起的学士帽穗）"
    }
    result = design_characters(state)
    printable_result = {
        "characters": [
            character.model_dump() for character in result["characters"]
        ]
    }
    print(json.dumps(printable_result, ensure_ascii=False, indent=2))
"""

# 测试设计故事大纲功能
"""
if __name__ == "__main__":

    characters = [
        Character(
            name = "林砚",
            role = "摄影组成员／暗中掌镜者",
            appearance = "浅灰蓝衬衫+深藏青西装马甲，领带微松、一角翘起；黑框细边眼镜，额前碎发略乱；手腕戴一只旧款银色机械表",
            personality = [
                "沉静",
                "观察型",
                "手巧"
            ],
            motivation = "想用镜头悄悄记住每个人最自然的样子，而非摆拍的完美瞬间",
            continuity_constraints = [
                "领带始终微松且右下角翘起，不被整理",
                "银色机械表始终戴在左手腕，表带扣为哑光银色小圆扣",
                "黑框眼镜鼻托处有细微划痕，始终可见"
            ]
        ),
        Character(
            name = "陈屿",
            role = "班级纪念册主编",
            appearance = "白衬衫袖口卷至小臂，露出淡褐色胎记；米白阔腿裤配帆布鞋；左耳单戴一枚哑光铜色小圆耳钉；手持一本边缘磨损的硬壳毕业纪念册",
            personality = [
                "温柔",
                "内敛",
                "执念型"
            ],
            motivation = "想把此刻所有人的温度和呼吸都存进这本册子里，对抗即将来临的空白",
            continuity_constraints = [
                "毕业纪念册始终拿在左手，封面烫金校徽有局部褪色痕迹",
                "左耳铜色圆耳钉始终佩戴，无其他耳饰",
                "白衬衫右袖口卷至小臂中段，折痕清晰，不放下也不再上卷"
            ]
        ),            
        Character(
            name = "吴昭",
            role = "现场协调人／气氛锚点",
            appearance = "深红学士袍敞开穿，内搭亮黄色T恤；头发扎高马尾，一根蓝色橡皮筋松垮缠绕；右手腕系一条褪色蓝白校运会手环；正踮脚帮别人扶歪斜的学士帽",
            personality = [
                "鲜活",
                "停不下来",
                "共情力强"
            ],
            motivation = "用不停动作来延缓分别的实感——只要还在帮忙，就还没到说再见的时候",
            continuity_constraints = [
                "蓝色橡皮筋始终缠绕在右手腕，松垮但未脱落",
                "校运会手环始终系于右手腕，蓝白条纹有明显日晒褪色区",
                "亮黄色T恤领口处有一处米粒大小的白色颜料渍（左肩缝线旁）"
            ]
        )
    ]
    state: FilmState = {
        "user_idea": "毕业季同学们一起拍摄毕业照，面临分别，依依不舍，生成一段60秒的青春校园影片",
        "scene_revision_count": 0,
        "story_revision_count": 0,
        "film_brief": FilmBrief(
            target_duration_sec = 60,
            genre = "青春校园",
            core_theme = "青春告别与分别不舍",
            visual_style = "自然光主导，柔焦+轻微胶片颗粒感；色调以清透青橙色系为主（晨光/午后暖调）；构图强调群像互动与留白，多用中景跟拍与低角度仰拍增强青春张力；穿插0.5秒内快切特写（系松动的领带、攥紧的毕业册边角、飘起的学士帽穗）"
        ),
        "characters": characters,
    }
    result = plan_story(state)
    print(
        json.dumps(
            {
                "story_outline": result["story_outline"].model_dump()
            },
            ensure_ascii=False,
            indent=2
        )
    )
"""

# 测试设计分场镜头功能
"""
if __name__ == "__main__":
   characters = [
        Character(
            name = "林砚",
            role = "摄影组成员／暗中掌镜者",
            appearance = "浅灰蓝衬衫+深藏青西装马甲，领带微松、一角翘起；黑框细边眼镜，额前碎发略乱；手腕戴一只旧款银色机械表",
            personality = [
                "沉静",
                "观察型",
                "手巧"
            ],
            motivation = "想用镜头悄悄记住每个人最自然的样子，而非摆拍的完美瞬间",
            continuity_constraints = [
                "领带始终微松且右下角翘起，不被整理",
                "银色机械表始终戴在左手腕，表带扣为哑光银色小圆扣",
                "黑框眼镜鼻托处有细微划痕，始终可见"
            ]
        ),
        Character(
            name = "陈屿",
            role = "班级纪念册主编",
            appearance = "白衬衫袖口卷至小臂，露出淡褐色胎记；米白阔腿裤配帆布鞋；左耳单戴一枚哑光铜色小圆耳钉；手持一本边缘磨损的硬壳毕业纪念册",
            personality = [
                "温柔",
                "内敛",
                "执念型"
            ],
            motivation = "想把此刻所有人的温度和呼吸都存进这本册子里，对抗即将来临的空白",
            continuity_constraints = [
                "毕业纪念册始终拿在左手，封面烫金校徽有局部褪色痕迹",
                "左耳铜色圆耳钉始终佩戴，无其他耳饰",
                "白衬衫右袖口卷至小臂中段，折痕清晰，不放下也不再上卷"
            ]
        ),            
        Character(
            name = "吴昭",
            role = "现场协调人／气氛锚点",
            appearance = "深红学士袍敞开穿，内搭亮黄色T恤；头发扎高马尾，一根蓝色橡皮筋松垮缠绕；右手腕系一条褪色蓝白校运会手环；正踮脚帮别人扶歪斜的学士帽",
            personality = [
                "鲜活",
                "停不下来",
                "共情力强"
            ],
            motivation = "用不停动作来延缓分别的实感——只要还在帮忙，就还没到说再见的时候",
            continuity_constraints = [
                "蓝色橡皮筋始终缠绕在右手腕，松垮但未脱落",
                "校运会手环始终系于右手腕，蓝白条纹有明显日晒褪色区",
                "亮黄色T恤领口处有一处米粒大小的白色颜料渍（左肩缝线旁）"
            ]
        )
    ]
   
   state: FilmState = {
        "user_idea": "毕业季同学们一起拍摄毕业照，面临分别，依依不舍，生成一段60秒的青春校园影片",
        "scene_revision_count": 0,
        "story_revision_count": 0,
        "characters": characters,
        "story_outline": StoryOutline(
            setup = "毕业照拍摄前半小时，三人因分工自然聚拢：林砚调试设备，陈屿核对名单，吴昭协调站位，默契尚在。",
            conflict = "三人对‘告别’的应对方式产生隐性张力：林砚回避集体仪式，陈屿固守纪念册实体，吴昭以行动消解离别实感。",
            turning_point = "摄影师喊‘最后一组’时，吴昭突然停顿，林砚按下快门，陈屿合上纪念册——三人同步意识到‘此刻正在结束’。",
            ending = "林砚将偷拍胶片悄悄塞进陈屿的纪念册夹层；吴昭松开扶帽的手，三人静立，分别成为可确认的共同事实。",
            theme = "真正的告别不是瞬间断裂，而是当各自守护的方式终于彼此托住时，放手才获得重量。"
        ),
        film_brief: FilmBrief(
            target_duration_sec = 60,
            genre = "青春校园",
            core_theme = "青春告别与分别不舍",
            visual_style = "自然光主导，柔焦+轻微胶片颗粒感；色调以清透青橙色系为主（晨光/午后暖调）；构图强调群像互动与留白，多用中景跟拍与低角度仰拍增强青春张力；穿插0.5秒内快切特写（系松动的领带、攥紧的毕业册边角、飘起的学士帽穗）"
        ),
    }
    result = write_scenes(state)
    printable_result = {
        "scenes": [
            scene.model_dump() for scene in result["scenes"]
        ]
    }
    print(json.dumps(printable_result, ensure_ascii=False, indent=2))
"""
