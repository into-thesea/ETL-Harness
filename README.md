# Governed

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-6b72db.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个用于构建、运行和管控 LLM Agent 的框架。用 LangGraph 做编排骨架，围绕它手写了一层管控运行时（工具调度、权限、记忆、审批、沙箱、审计与追踪），并内置一条端到端的数据分析工作流作为首个领域实例。

## 这是什么

直接用 LangGraph 搭 Agent，规划、记忆、工具调用、权限这些都要自己拼装。Governed 把这些围绕模型的"管控工程"收敛成一个可复用的运行时：你注册业务工具、组装图，就能得到一个带权限、审批、审计和断点恢复的 Agent。

框架本身不绑定业务——换一套工具就是另一个应用。仓库里自带的数据分析工具集（体检、清洗、EDA、SQL、出图、代码执行）是第一个完整示例，跑通"数据获取 → 清洗 → 探查 → 建模 → 可视化 → 报告"这条链路。

## 特性

**规划与编排**
- 顶层 Plan-and-Execute：模型先把目标拆成子任务，再逐个执行；每个子任务内部走 ReAct 循环
- 子 Agent 委派：专业子任务交给独立上下文、独立工具集的子 Agent，只回结构化结论，不污染主上下文
- 8 个内置角色（数据体检 / 清洗 / 分析 / 出图 / 代码 / 报告 / 质检 / 执行）

**上下文与记忆**
- 虚拟文件系统（VFS）：大结果落盘，prompt 里只留摘要和文件引用，需要时再读全文
- 上下文管理：超长观察值自动沉淀、历史消息自动压缩，带计数统计
- 分层记忆：Redis 短期 + Milvus 长期向量 + 进程内工作记忆

**工具与技能**
- Tool Broker 统一入口：注册、参数校验、限流、异常兜底，结果统一为 `(ok, text, artifacts)`
- 可插拔中间件：在 before/after LLM、工具、文件等节点挂 hook，横切逻辑不侵入业务
- Skills 系统：把 SOP、模板、SQL、脚本打包成 `SKILL.md`，按相关度渐进式注入

**安全与管控**
- 两层数据权限：PDP 控制"谁能调哪个工具"，行列级权限改写 SQL 强制注入行过滤与列白名单
- 服务端 Bearer 令牌鉴权，角色由令牌决定；审批按身份判定，不能自批
- 高风险操作走 LangGraph interrupt，暂停等人审批后再继续
- 中文 PII 识别（身份证 / 手机号 / 银行卡，校验位防误报）与数据质量红线
- 代码执行进隔离沙箱；沙箱不可用或凭据缺失时 fail closed，绝不在本机直接跑

**服务化与可观测**
- FastAPI 提供同步与 SSE 流式接口，审批中断状态落盘、重启可续
- 全链路审计与 trace 埋点，经 Kafka 上送（带本地 spool，断网不丢）
- 每步状态 checkpoint，崩溃后从断点恢复

## 架构

从上到下分四层，横切的管控逻辑以中间件形式注入：

```
接入层        FastAPI：任务提交 / SSE 流式 / 状态查询 / 审批（Bearer 鉴权）
编排层        LangGraph：顶层 Plan-and-Execute，子任务内 ReAct（think → action → final）
管控运行时    Tool Broker · 中间件 · PDP / 行列权限 · 审批 · 上下文管理 · Skills · 沙箱
状态与记忆    VFS · Redis 短期 · Milvus 长期 · 工作记忆 · Checkpoint
基础设施      Docker Compose：Redis / Milvus / Kafka / MinIO（+ 可选 MySQL/PG）
```

一次请求的完整调用链、各模块的详细设计见 [`docs/项目计划.md`](docs/项目计划.md)；更完整的架构设计文档仍在整理中。

## 安装

```bash
# 克隆后进入目录
git clone https://github.com/into-thesea/governed.git
cd governed

# 建议用独立虚拟环境
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

# 配置环境变量
copy .env.example .env            # 编辑 .env 填入 API Key 等
```

## 快速开始

### 1. 起基础设施

确保 Docker Desktop 已运行：

```bash
cd infra
docker-compose up -d
```

默认端口：Redis 6379、Milvus 19530、Kafka 9092、MinIO 9000（控制台 9001）。

> 端口被占用（本机可能跑着其他项目的容器）时，不要停别人的容器，改 `infra/docker-compose.yml` 里的端口映射，或只起当前需要的服务。

**沙箱服务端**（仅 `code_executor` 依赖）是独立进程，不在 compose 内：

```powershell
cd infra\opensandbox-server
python -m venv .venv
.venv\Scripts\python.exe -m pip install opensandbox-server==0.2.3
Copy-Item sandbox.toml.example sandbox.toml

$env:OPENSANDBOX_SERVER_API_KEY = "your-long-random-key"   # 与客户端 .env 的 SANDBOX_API_KEY 一致
.\start.ps1
```

沙箱没起或没配 key 时，`code_executor` 会明确报错并 fail closed；其余工具不依赖沙箱，照常可用。

### 2. 跑一个示例

```bash
# 不需要 API Key，用 mock LLM 跑通整条链路
python examples\mock_llm_demo.py

# 配好真实 LLM 后跑端到端分析
python examples\data_analysis_demo.py
```

### 3. 起 API 服务

服务端默认开启鉴权，未配置令牌会直接启动失败。先在 `.env` 里配：

```bash
AUTH_ENABLED=true
AUTH_TOKENS={"<令牌1>":{"role":"analyst","name":"张三"},"<令牌2>":{"role":"admin","name":"李四"}}
AUTH_APPROVER_ROLES=admin
```

启动（注意 `--factory`，应用由工厂构造，模块里没有 `app` 变量）：

```bash
python -m uvicorn --factory harness.server.app:create_app --host 0.0.0.0 --port 8000
```

调用接口（角色由令牌决定，请求体里没有 role 字段）：

```bash
TOKEN=<令牌>

# 创建任务
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"goal": "分析这份销售数据的趋势"}'

# 查询状态 / 待审批项（THREAD 为返回的 thread_id）
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks/$THREAD
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks/$THREAD/approvals

# 提交审批：需审批角色，且不能批准自己发起的任务（同角色的其他人可以审批）
curl -X POST http://localhost:8000/api/v1/tasks/$THREAD/approval \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"approved": false, "comment": "来源不明，不允许"}'

# SSE 流式订阅（EventSource 不能设请求头，这条路由额外接受 ?token=）
curl -N "http://localhost:8000/api/v1/tasks/$THREAD/stream?token=$TOKEN"
```

`/docs`、`/openapi.json` 也需要令牌，只有 `/health` 匿名。本机开发可设 `AUTH_ENABLED=false`（会打 WARNING，且只有一个 admin 身份）。

## 配置

除鉴权外，下列能力都通过 `.env` 开关，默认不改变最小部署行为：

| 能力 | 关键配置 | 默认 | 开启后 |
|---|---|---|---|
| 工具级权限 | `PERMISSION_PDP_RULES` | 空（放行） | 按角色限定可调用的工具 |
| 行列级权限 | `PERMISSION_ENABLED=true` | 关闭 | 给 SQL 注入行过滤与列白名单，无法安全改写的语句直接拒绝 |
| PII 脱敏 | `PII_ENABLED` | 开 | 识别身份证/手机号/银行卡并替换为占位符 |
| 数据质量门 | `QUALITY_DATA_CHECK_ENABLED` | 开 | 缺失率/重复率超红线即暂停待确认，子任务由 Critic 质检 |
| 审批持久化 | `CHECKPOINT_BACKEND` | sqlite | 中断状态落盘，重启可续；`memory` 则重启即丢 |
| 多数据源 | `DATASOURCE_SOURCES` | 空 | 配置命名 MySQL/PG 源，`sql_query` 按需切换 |
| 对象存储 | `MINIO_ENABLED` | 关 | VFS 大文件落 MinIO；默认落本地磁盘（`VFS_LOCAL_ROOT`） |

完整取值见 [`.env.example`](.env.example)，规则 JSON 写法见 `harness/config.py` 各 Settings 类的 docstring。安全相关能力均为 fail closed。

## 项目结构

```
governed/
├── harness/                # 框架库
│   ├── config.py           # 配置（pydantic-settings，按域分组）
│   ├── models.py  state.py # 数据模型 / LangGraph 状态
│   ├── tool_broker.py      # 工具统一调度
│   ├── middleware.py  pdp.py  audit.py
│   ├── checkpoint.py  mcp_adapter.py  orchestrator.py
│   ├── nodes.py  graph.py  # 图节点与构建
│   ├── agents/             # 子 Agent 注册
│   ├── context/            # 上下文管理（沉淀 + 压缩）
│   ├── datasources/        # SQLAlchemy 多数据源
│   ├── permissions/        # 行列级数据权限
│   ├── memory/             # working / short_term / long_term
│   ├── planning/           # 规划、任务存储、质量门
│   ├── vfs/                # 虚拟文件系统
│   ├── skills/             # loader + 各 SKILL.md
│   ├── trace/              # tracer + Kafka 生产
│   ├── sandbox/            # 沙箱客户端与执行器
│   └── server/             # FastAPI（app/auth/service/schemas/run）
├── tools/                  # 内置数据分析工具集
├── infra/                  # docker-compose、沙箱镜像、opensandbox-server
├── examples/               # mock 与真实 LLM 示例
├── tests/                  # pytest 测试
└── docs/                   # 设计文档、计划、问题与阶段存档
```

## 扩展

- **加工具**：定义 `ToolDef`，写 `handler(args, context) -> (ok, text, artifacts)`，`broker.register(...)`
- **加中间件**：继承 `Middleware`，实现需要的 hook，注册到 manager
- **加 Skill**：按约定写 `SKILL.md` 放到技能目录，loader 自动发现
- **加子 Agent**：定义其工具集与系统提示，在 Orchestrator 注册后用 `delegate(...)` 委派

## 文档

- [`docs/项目计划.md`](docs/项目计划.md)：总体计划、架构决策与最新进展
- [`docs/遇到的问题.md`](docs/遇到的问题.md)：踩坑记录与收口情况
- [`docs/阶段存档_20260926.md`](docs/阶段存档_20260926.md)、[`阶段存档_20260927.md`](docs/阶段存档_20260927.md)：阶段交付存档
- [`docs/学习笔记.md`](docs/学习笔记.md)：模块学习地图与面试复习（个人学习用）

## Roadmap

- [ ] 真实 LLM 稳定性收敛：减少幻觉文件名与多余的工具调用
- [ ] 沙箱控制面纳入 docker-compose（当前用独立脚本过渡）
- [ ] 真实 MySQL/PostgreSQL 端到端联调
- [ ] 补充架构设计文档、提升测试覆盖率

## License

基于 [MIT License](LICENSE) 开源。
