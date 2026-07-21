# Mini Film Agent

一个基于 LangGraph 的有状态短片策划 Agent Workflow：系统从一句用户创意出发，依次完成需求解析、角色设定、故事大纲和分场规划，并通过 Story / Scene 双层 Review–Revision、Human-in-the-loop、历史问题追踪和长期 Memory，构建可审核、可修订、可持续适配用户偏好的创作闭环。

## 目录

- [核心能力与亮点](#核心能力与亮点)
- [系统架构](#系统架构)
- [Runtime Flow](#runtime-flow)
- [Review–Revision 与 Issue 状态追踪](#reviewrevision-与-issue-状态追踪)
- [HITL 暂停与恢复](#hitl-暂停与恢复)
- [长期 Memory](#长期-memory)
- [一次 Query 的完整流动](#一次-query-的完整流动)
- [API 与最小使用示例](#api-与最小使用示例)
  - [普通生成](#普通生成)
  - [HITL Start](#hitl-start)
  - [HITL Resume](#hitl-resume)
- [技术栈与项目结构](#技术栈与项目结构)
- [快速启动与测试](#快速启动与测试)
- [Roadmap](#roadmap)
  - [当前版本](#当前版本)
  - [下一阶段演进](#下一阶段演进)
- [License](#license)

## 核心能力与亮点

- **LangGraph 状态工作流**：`FilmState` 串联创意、角色、故事、分场、审核、Memory、HITL 和 Trace。
- **Story / Scene 双层闭环**：故事大纲和分场规划分别审核、分别修订，避免一次性生成后直接结束。
- **结构化输出**：核心产物使用 Pydantic schema 约束，包括 `FilmBrief`、`StoryOutline`、`SceneList` 和 Review 结果。
- **Issue 状态追踪**：历史问题会被判断为 `resolved`、`unresolved`、`regressed`，并影响后续审核和修订。
- **Human-in-the-loop**：`film_hitl_graph` 在故事大纲完成后通过 LangGraph `interrupt()` 暂停，支持 `resume` 后继续执行。
- **长期 Memory**：按全局、Story、Scene 三类作用域保存偏好，并在后续生成、审核和修订中作为软参考。
- **节点级 Trace**：每个节点记录同一个 `execution_id` 下的执行状态、阶段和耗时。
- **FastAPI + SSE**：提供非流式生成、节点级流式输出，以及 HITL 启动 / 恢复接口。

## 系统架构

```mermaid
flowchart LR
    C["Client"] --> A["FastAPI / SSE"]
    A --> G["LangGraph Workflow Runtime"]

    G --> W["Film Workflow Nodes<br/>Brief / Characters / Story / Scenes"]
    W --> Q["Review / Revision Loops"]
    Q --> H["Human-in-the-loop"]
    W --> L["LLM<br/>Qwen via DashScope"]
    Q --> L

    G <--> K["Checkpointer<br/>InMemorySaver"]
    G <--> M["Long-term Memory<br/>JSON Store"]
    G --> T["Execution Trace"]
```

FastAPI 负责入口、响应模型和 SSE 事件；LangGraph 负责节点路由、循环、checkpoint 与恢复；Memory 和 Trace 分别承担长期偏好与单次执行审计。

## Runtime Flow

普通自动流程使用 `film_graph`：

```mermaid
flowchart TD
    A["Start: user_id + user_idea + execution_id"] --> B["retrieve_memory"]
    B --> C["analyze_brief"]
    C --> D["design_characters"]
    D --> E["plan_story"]
    E --> F["review_story"]
    F -->|needs revision| G["revise_story"]
    G --> F
    F -->|ready| H["write_scenes"]
    H --> I["review_scene"]
    I -->|needs revision| J["revise_scene"]
    J --> I
    I -->|ready| K["finalize"]
    K --> L["update_memory"]
    L --> M["END"]
```

带人工审核的流程使用独立 `film_hitl_graph`，只在 Story 大纲准备进入分场前暂停：

```mermaid
flowchart TD
    A["review_story"] -->|machine revise| B["revise_story"]
    B --> A
    A -->|ready for scenes| C["human_review_story interrupt"]
    C -->|approve| D["write_scenes"]
    C -->|revise + feedback| B
```

## Review–Revision 与 Issue 状态追踪

项目把审核和修订拆开处理：Review 只判断是否足以进入下一阶段，Revision 只做最小必要修改。

- Story 层：`review_story` 审核故事结构、因果逻辑、角色一致性和主题时长；失败时进入 `revise_story`，最多修订 3 轮。
- Scene 层：`review_scene` 审核字段、总时长、场景数量、角色一致性和分场语义；失败时进入 `revise_scene`，最多修订 2 轮。
- `issues` 表示阻断性问题，修订时必须处理。
- `suggestions` 表示非阻断建议，只作为参考。

审核历史会记录在 `story_review_history` 和 `scene_review_history` 中。历史 issue 的状态包括：

- `resolved`：当前版本中已不存在。
- `unresolved`：一直没有解决。
- `regressed`：之前已解决，但当前版本再次出现。

Revision 会把 `unresolved` / `regressed` / 旧格式 `unknown` 放入“仍需避免的历史问题”，把少量 `resolved` 放入“防回归提醒”，避免旧问题反复回潮。

## HITL 暂停与恢复

HITL 当前只接在 Story 大纲机器审核通过之后、`write_scenes` 之前。

暂停时，`human_review_story` 通过 `interrupt()` 返回可 JSON 序列化的审核材料，包括 `film_brief`、`characters`、`story_outline` 和 `story_review`。恢复时客户端提交：

```json
{
  "decision": "approve",
  "feedback": null
}
```

或：

```json
{
  "decision": "revise",
  "feedback": "结尾改成更加克制的开放式结局。"
}
```

`approve` 会继续进入分场生成；`revise` 会把人工反馈写入 State，回到 `revise_story`，之后再次审核并可能再次暂停。

当前 Checkpointer 使用 `InMemorySaver`，`thread_id` 复用 `execution_id`。这适合本地 demo 和 HITL 验证，但服务重启后 checkpoint 会丢失，也不适合多 worker 部署。

## 长期 Memory

Memory 保存的是跨多次生成仍然稳定的用户偏好，不保存某一次请求里的具体人物、地点、场次或剧情细节。

`UserMemory` 分为三类作用域：

- 全局偏好：`preferred_genres`、`style_preferences`、`disliked_elements`、`preferred_duration_sec`、`additional_preferences`
- Story 偏好：`story_preferences`
- Scene 偏好：`scene_preferences`

写入链路在 `finalize` 后执行：

```text
extract_memory_update → merge_user_memory → save_user_memory
```

提取器只参考用户本次输入和人工反馈，不根据机器 Review、生成结果或 `final_output` 推测偏好。合并阶段会过滤空值、去重、保留原顺序；有真实增量时保存 JSON，状态为 `saved`，否则为 `skipped`。

消费侧按节点分层：

- `analyze_brief` 读取完整长期偏好。
- Story 节点使用 `format_story_memory_context()`，不读取 `scene_preferences`。
- Scene 节点使用 `format_scene_memory_context()`，同时读取 `story_preferences` 和 `scene_preferences`。

所有使用 Memory 的 Prompt 都明确：当前请求优先于长期 Memory，冲突时忽略 Memory。

## 一次 Query 的完整流动

1. API 入口生成 `execution_id`，并作为 LangGraph `thread_id`。
2. 构造初始 `FilmState`。
3. `retrieve_memory` 读取该用户长期 Memory。
4. `analyze_brief` 解析类型、主题、视觉风格、时长和推荐分场数量。
5. `design_characters` 生成主要角色与连续性约束。
6. `plan_story` 生成结构化故事大纲。
7. `review_story` / `revise_story` 完成 Story 层闭环。
8. `write_scenes` 生成分场规划。
9. `review_scene` / `revise_scene` 完成 Scene 层闭环。
10. `finalize` 汇总最终策划案。
11. `update_memory` best-effort 更新长期偏好。
12. API 返回 `final_output`、`execution_trace` 和 `memory_update_status`。

每条 TraceEvent 至少包含：

```json
{
  "execution_id": "exec_xxx",
  "node": "review_story",
  "status": "success",
  "stage": "story_review_completed",
  "duration_ms": 123.45
}
```

## API 与最小使用示例

接口列表：

- `GET /health`
- `POST /api/v1/films/generate`
- `POST /api/v1/films/stream`
- `POST /api/v1/films/hitl/start`
- `POST /api/v1/films/hitl/{execution_id}/resume`
- `POST /api/v1/films/hitl/stream/start`
- `POST /api/v1/films/hitl/stream/{execution_id}/resume`

### 普通生成

```bash
curl -X POST http://127.0.0.1:8000/api/v1/films/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user_001",
    "user_idea": "生成一个60秒的校园毕业短片"
  }'
```

### HITL Start

```bash
curl -X POST http://127.0.0.1:8000/api/v1/films/hitl/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user_001",
    "user_idea": "生成一个克制表达的校园毕业故事"
  }'
```

如果暂停，响应中的 `status` 为 `waiting_for_human`，并返回 `review_payload`。

### HITL Resume

```bash
curl -X POST http://127.0.0.1:8000/api/v1/films/hitl/exec_xxx/resume \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "revise",
    "feedback": "结尾改成更加克制的开放式结局。"
  }'
```

`decision` 只能是 `approve` 或 `revise`；`revise` 时 `feedback` 去除首尾空格后必须非空。

## 技术栈与项目结构

主要技术栈：

- Python
- LangGraph
- LangChain OpenAI-compatible client
- Pydantic
- FastAPI
- SSE

当前 LLM 配置在 `nodes.py` 中，使用 DashScope OpenAI-compatible endpoint：

```text
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
model = qwen-plus
env = DASHSCOPE_API_KEY
```

项目结构：

```text
mini_film_agent/
├── app/
│   ├── main.py          # FastAPI 非流式、SSE、HITL 接口
│   ├── schemas.py       # API 请求与响应模型
│   └── sse.py           # SSE 事件格式化
├── memory/
│   ├── models.py        # UserMemory / MemoryUpdate
│   ├── retrieve_memory.py
│   ├── extract_memory.py
│   ├── merge.py
│   ├── update_memory.py
│   ├── store.py
│   └── context.py       # Story / Scene Memory 上下文格式化
├── reviews/
│   ├── review_story.py
│   ├── review_scene.py
│   └── human_review_story.py
├── revisions/
│   ├── revise_story.py
│   └── revise_scene.py
├── scripts/
│   └── demo_hitl.py     # 本地 HITL 暂停与恢复验证脚本
├── tests/
├── graph.py             # film_graph / film_hitl_graph
├── nodes.py             # 需求、角色、故事、分场节点
├── schemas.py           # 影片策划结构化模型
└── state.py             # FilmState 与 Trace / History TypedDict
```

## 快速启动与测试

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

配置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

启动 API：

```bash
uvicorn app.main:app --reload
```

本地运行自动 Graph：

```bash
python graph.py
```

本地验证 HITL：

```bash
python scripts/demo_hitl.py
```

运行测试：

```bash
pytest -q
```

## Roadmap

### 当前版本

- [x] 基于 LangGraph 的自动生成流程与独立 HITL 流程。
- [x] Story / Scene 双层 Review–Revision 闭环。
- [x] `issues` / `suggestions` 分层处理。
- [x] `resolved` / `unresolved` / `regressed` 历史问题状态追踪。
- [x] 全局、Story、Scene 分作用域长期 Memory。
- [x] Memory 读取、提取、合并、JSON 持久化与节点消费。
- [x] FastAPI 非流式、SSE 流式及 HITL Start / Resume 接口。
- [x] 节点级 `execution_trace` 与统一 `execution_id`。
- [x] 覆盖 Memory、Review History、Issue Status、HITL、API、SSE 和端到端链路的自动化测试。

### 下一阶段演进

当前版本已完成核心 Agent Workflow 闭环；后续将主要聚焦运行时可靠性、Prompt 可治理性、执行可观测性和多模态创作能力。

- [ ] **持久化 Agent Runtime**：将 Checkpointer 升级为持久化存储，支持服务重启后的 HITL 恢复，并为多实例运行预留一致性设计。
- [ ] **统一 Prompt 管理**：集中管理 Prompt 模板、变量校验和版本信息，为 Prompt 回溯、A/B 测试及评测复现提供基础。
- [ ] **执行级 Observability 与 Evaluation**：在节点 Trace 之上增加 Execution Summary、Token / 成本统计、历史执行查询和基于轨迹的 Agent 评测。
- [ ] **多模态视觉预演**：接入图片与视频生成工具，根据角色设定和分场规划生成角色参考图、分镜图及短视频预演，并逐步补充异步任务轮询、失败重试、生成参数管理和资产结果回写。
- [ ] **存储与部署扩展**：将本地 JSON Memory 抽象为可替换 Store，并逐步补充鉴权、限流和多 worker 部署能力。

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
