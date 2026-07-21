import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

from memory.context import format_scene_memory_context, format_story_memory_context
from state import FilmState
from schemas import FilmBrief, CharacterList, StoryOutline, Character, SceneList, Scene

#=============== 配置 LLM ===============
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise ValueError("未找到DASHSCOPE_API_KEY, 请在.env文件中配置")

llm = ChatOpenAI(
    api_key=api_key, # 阿里云dashscope控制台获取sk开头密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # 千问兼容接口固定地址
    model="qwen-plus", # 必须选支持function call的模型：qwen-plus/qwen-turbo/qwen2.5系列
    temperature=0 # 降低需求提取的随机性，使结构化输出更稳定
)

#=============== 配置 LLM with structured output 函数 ===============
brief_llm = llm.with_structured_output(FilmBrief) # 分析创意
character_llm = llm.with_structured_output(CharacterList) # 设计角色
story_outline_llm = llm.with_structured_output(StoryOutline) # 设计故事大纲
write_scenes_llm = llm.with_structured_output(SceneList) # 设计分场镜头

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

    prompt = f"""

    你是一名影视前期策划专家。请分析下面的短片创意，并提取结构化的创作需求。
    【用户本次需求】
    {user_idea}
    【用户长期偏好】
    {memory_text}
    
    使用长期偏好时必须遵守：
    - 用户本次需求的优先级高于长期偏好
    - 只使用与当前任务相关的偏好
    - 不要强行加入无关的历史偏好
    - 如果当前需求与长期偏好冲突，以当前需求为准
    - 如果长期偏好为空，则只根据本次需求进行策划

    要求：
    1. 提取目标时长，单位为秒；
    2. 判断主要影片类型；
    3. 总结作品的核心主题；
    4. 给出一句简短视觉风格描述，只包含整体色调、氛围和艺术方向，不描述镜头、构图、剪辑、摄影技巧，视觉风格控制在30字以内；
    5. 根据目标时长和故事复杂度，推荐合理的分场数量。
    参考：
    - 30秒以内：3-5个场景
    - 30-90秒：4-8个场景
    - 90秒以上：8个以上场景
    不要机械按照规则，根据剧情复杂度调整。
    6. 不要继续创作角色和剧情。
    """
    film_brief: FilmBrief = brief_llm.invoke(prompt)
    return {
        "film_brief": film_brief,
        "current_stage": "brief_completed"
    }


# 设计角色
def design_characters(state: FilmState) -> dict:
    user_idea = state["user_idea"]
    film_brief = state["film_brief"]
    genre = film_brief.genre
    core_theme = film_brief.core_theme
    visual_style = film_brief.visual_style

    prompt = f"""
    你是一名影视前期策划师。请根据下面的短片创意，设计出结构化的角色列表。
    用户输入的创意：{user_idea}
    影片类型：{genre}
    核心主题：{core_theme}
    视觉风格：{visual_style}
    要求：
    1. 设计必要的角色列表；
    2. 每个角色包含：角色名称、身份、外貌与穿着、性格特征、行为动机、连续性约束；
    3. 每个角色的性格特征不超过3项，性格特征使用简短形容词或短语；
    4. 行为动机用一句简洁的话描述；
    5. 每个角色给出2至3条连续性约束。连续性约束应描述角色在不同场景中保持不变的外貌、服装、配饰或关键行为特征。不要要求某个细节在每个镜头中都必须可见，不要规定物体必须朝向镜头，不要约束由机位、光照或构图导致的视觉效果；
    6. 控制角色数量，通常为1至3个主要角色；
    7. 不同角色的名字应明显不同，避免同字、近音或容易混淆；
    8. 不要继续创作剧情或场景。
    """
    characters_result: CharacterList = character_llm.invoke(prompt)
    return {
        "characters": characters_result.characters,
        "current_stage": "characters_completed"
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
    
    prompt = f"""
    你是一名专业影视编剧和前期策划师。
    请根据用户创意、影片需求和角色设定，设计一个适合短片制作的故事结构大纲。

    用户输入的创意：{user_idea}
    影片类型：{genre}
    核心主题：{core_theme}
    视觉风格：{visual_style}
    目标时长：{target_duration_sec}秒
    角色列表：{characters_json}

    【Story阶段可参考的长期Memory】
    {story_memory_text}

    Memory使用原则：
    - 当前用户要求优先于长期Memory。
    - 长期Memory仅作为可参考的稳定偏好，不是硬性任务。
    - 如果长期Memory与当前任务冲突，必须忽略Memory，以当前任务为准。
    - Story阶段不参考scene_preferences，避免把分场设计偏好混入故事大纲。

    请输出结构化故事大纲：
    - setup：故事开端
    - conflict：核心冲突
    - turning_point：关键转折
    - ending：故事结局
    - theme：核心主题

    严格要求：
    1. 输出的是故事策划阶段的大纲，不是文学故事，也不是拍摄脚本；
    2. 只描述：
    - 人物关系变化；
    - 核心事件；
    - 情绪变化；
    - 故事因果关系；
    3. 不描述：
    - 具体环境描写；
    - 人物外貌细节；
    - 人物动作细节；
    - 道具细节；
    - 镜头语言；
    - 画面语言；
    4. 不使用：
    - “镜头”
    - “画面”
    - “特写”
    等视觉文学化表达；
    5. 每个字段50字以内；
    6. 故事只包含一个核心冲突，一个关键转折；
    7. 不新增角色，不展开对白；
    8. 后续节点会根据该大纲生成具体分场，因此当前只提供故事骨架。
    9. 不要为故事结尾设计具体事件、道具或动作，保留后续分场创作空间。
    """
    story_outline: StoryOutline = story_outline_llm.invoke(prompt)
    return {
        "story_outline": story_outline,
        "current_stage": "story_outline_completed"
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

    prompt = f"""
    你是一名影视制作规划师。
    请根据故事大纲和角色设定，
    生成一个适合{target_duration_sec}秒短片的分场规划。

    用户创意: {user_idea}
    故事大纲: {story_outline_json}
    角色列表: {characters_json}

    【Scene阶段可参考的长期Memory】
    {scene_memory_text}

    Memory使用原则：
    - 当前用户要求优先于长期Memory。
    - 长期Memory只作稳定偏好参考，不是硬性任务。
    - 如果长期Memory与当前任务冲突，必须忽略Memory，以当前任务为准。
    - story_preferences用于继承故事方向；scene_preferences用于约束分场执行。

    要求：
    1. 根据推荐分场数量生成场景。
    推荐场景数量：{film_brief.recommended_scene_count}
    允许根据故事完整性上下浮动1个场景。
    不要为了满足数量增加无关事件。
    2. 每个场景包含：
    - 场景编号
    - 时长
    - 地点
    - 参与角色
    - 场景动作
    - 必要对白
    - 视觉目标
    3. 每个场景 duration_sec 必须为正整数，且不超过{target_duration_sec}秒，所有 scene 的 duration_sec 之和必须等于目标时长{target_duration_sec}秒；
    4. 每个场景必须推动故事发展；
    5. 只能使用已有角色，不新增角色；
    6. action只描述人物行为和剧情推进，不描述摄影画面、光影、气氛。
    7. visual_goal描述该场景希望传达的情绪或叙事目的；
    8. 不生成：
    - 摄影机参数
    - 镜头运动
    - 分镜编号
    - 视频生成prompt；
    9. 输出的是前期策划分场，不是完整剧本。
    10. 每个场景只描述一个主要事件；
    11. 场景之间应该形成连续叙事关系，而不是独立片段；
    12. 不要为了填充时长增加无关事件；
    """
    scenes_result: SceneList = write_scenes_llm.invoke(prompt)
    return {
        "scenes": scenes_result.scenes,
        "current_stage": "scenes_completed"
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
