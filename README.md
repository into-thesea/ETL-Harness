# Governed

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-6b72db.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个用于构建、运行和管控 LLM Agent 的框架。用 LangGraph 做编排骨架，围绕它手写了一层管控运行时（工具调度、权限、记忆、审批、沙箱、审计与追踪），并内置一条端到端的数据分析工作流作为首个领域实例。

## 这是什么

直接用 LangGraph 搭 Agent，规划、记忆、工具调用、权限这些都要自己拼装。Governed 把这些围绕模型的"管控工程"收敛成一个可复用的Harness：你注册业务工具、组装图，就能得到一个带权限、审批、审计和断点恢复的 Agent。

框架本身不绑定业务——换一套工具就是另一个应用。仓库里自带的数据分析工具集（体检、清洗、EDA、SQL、出图、代码执行）是第一个完整示例，跑通"数据获取 → 清洗 → 探查 → 建模 → 可视化 → 报告"这条链路。

## 特性

**规划与编排**
- 顶层 Plan-and-Execute：模型先把目标拆成子任务，再逐个执行；每个子任务内部走 ReAct 循环
- 子 Agent 委派：专业子任务交给独立上下文、独立工具集的子 Agent，只回结构化结论，不污染主上下文
- 3 个核心子 Agent（数据探查与清洗 / 分析建模与可视化 / 报告生成与质检）—— 角色数量由**上下文隔离需求**决定而非业务步骤，每个都有独立上下文、专属工具白名单与迭代预算

**上下文与记忆**
- 虚拟文件系统（VFS）：大结果落盘，prompt 里只留摘要和文件引用，需要时再读全文
- 上下文管理：超长观察值自动沉淀、历史消息自动压缩，带计数统计
- 记忆分三层，按**时间尺度**切：任务内上下文（图状态 + 自动压缩沉淀）、任务断点（检查点落盘，重启可续）、跨会话经验（向量检索，默认 pgvector）

**工具与技能**
- Tool Broker 统一入口：注册、参数校验、限流、异常兜底，结果统一为 `(ok, text, artifacts)`
- 工具结果缓存：只读工具按"参数 + 输入文件身份"命中，文件变了立即失效，没有 TTL 可猜；命中仍走权限、校验、熔断、限流与审计
- 可插拔中间件：在 before/after LLM、工具、文件等节点挂 hook，横切逻辑不侵入业务
- Skills 系统：把 SOP、模板、SQL、脚本打包成 `SKILL.md`，按相关度渐进式注入

**安全与管控**
- 两层数据权限：PDP 控制"谁能调哪个工具"，行列级权限改写 SQL 强制注入行过滤与列白名单
- 服务端 Bearer 令牌鉴权，角色由令牌决定；审批按身份判定，不能自批
- 高风险操作走 LangGraph interrupt，暂停等人审批后再继续
- 审批支持「本任务内不再询问」：对某个高危工具一次性给出常驻的允许或拒绝，免去同一任务里反复弹同一张卡；豁免**只跳过审批这一步**，权限判定与沙箱照旧，授予与每次生效都落审计并推到控制台
- 审批按**调用**分级：领域包声明工具的风险策略（只回答"多危险"），部署方配置阈值（回答"问还是放"）。低风险**且**有沙箱等确定性机制兜底时自动放行，高风险才停下来问人，触及红线的直接拒。自动放行的前提是"有机制兜底"而非"判断它安全"，每次放行都带风险等级与兜底机制名留痕
- 执行前计划审批（可选）：计划生成后**停下来**摆给审批人，批准了才开始动手；驳回则带着意见退回重规划，改出来的计划同样要过审 —— 驳回不会把任务判死
- 无人值守是**说出口**的选择：挂有需审批工具的部署必须声明自己有没有审批通道（`SERVER_APPROVAL_CHANNEL`，不配置则启动失败），超时无人处理的审批由后台自动驳回并推进任务 —— 不会静默卡在等待里
- 中文 PII 识别（身份证 / 手机号 / 银行卡，校验位防误报）与数据质量红线
- 代码执行进隔离沙箱：非 root 运行 + Capability 剥离（9 项）+ `no_new_privileges` + seccomp（Docker 官方默认 profile）+ 进程数上限，文件只经 Filesystem API 与宿主显式交换（禁止宿主路径 bind mount）；沙箱不可用或凭据缺失时 fail closed，绝不在本机直接跑

**服务化与可观测**
- FastAPI 提供同步与 SSE 流式接口，审批中断状态落盘、重启可续
- 内置 Web 控制台（`/`，零构建）：任务列表、运行详情（子任务状态机 + 工具 / 子 Agent 实时事件时间线）、审批台、管控面、运行时指标、工作区与产物、插件页；原生 HTML/CSS/JS，无 npm、无构建产物，数据一律走受鉴权保护的接口
- 全链路审计与 trace 埋点，经 Kafka 上送（带本地 spool，断网不丢）
- 每步状态 checkpoint，崩溃后从断点恢复

## 架构

从上到下分四层，横切的管控逻辑以中间件形式注入：

```
接入层        FastAPI：任务提交 / SSE 流式 / 状态查询 / 审批（Bearer 鉴权）
编排层        LangGraph：顶层 Plan-and-Execute，子任务内 ReAct（think → action → final）
管控运行时    Tool Broker · 中间件 · PDP / 行列权限 · 审批 · 上下文管理 · Skills · 沙箱
状态与记忆    VFS · 图状态与上下文压缩 · 检查点 · 长期向量记忆（pgvector / milvus / local）
基础设施      Docker Compose：Milvus / Kafka（+ 可选 MinIO / MySQL / PG）
```

一次请求的完整调用链与各模块详细设计见架构设计文档（整理中）。

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

# 以可编辑方式安装本包（**必须**：领域包通过 entry points 发现，
# 而 entry points 只认已安装的发行版）
pip install -e .

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

默认端口：Milvus 19530、Kafka 9092。MinIO（9000 / 控制台 9001）**默认不启用**，VFS 落本地磁盘；需要对象存储时再开，见「配置」。

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
python -m examples.mock_llm_demo

# 配好真实 LLM 后跑端到端分析（未配 Key 时自动回落到脚本化决策）
python -m examples.data_analysis_demo
```

> 用 `-m` 模块方式运行：这些示例要 `import harness`，直接当脚本跑（`python examples\x.py`）不会把项目根加进 `sys.path`。

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

浏览器打开 `http://localhost:8000/` 就是控制台。它只匿名取到**外壳**（HTML/CSS/JS，
不含数据）；首次用 `http://localhost:8000/?token=<令牌>` 打开，令牌会存进本机浏览器，
之后的接口调用都带 `Authorization` 头。数据一律走受鉴权保护的 `/api/v1`。

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

# 已挂载的领域包与装载状态（能挂上哪些领域、各自贡献了什么）
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/packages

# 控制台用的只读观测接口：会话枚举 / 管控面 / 单会话指标 / 单会话产物
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/control-plane
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks/$THREAD/metrics
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks/$THREAD/artifacts
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
| 审批通道 | `SERVER_APPROVAL_CHANNEL` | **无默认** | `http`（有人审）/ `none`（无人值守）。挂有需审批工具时**必填**，否则启动失败 —— 强迫把"这个部署有没有人审"说出口 |
| 无人应审批 | `SERVER_APPROVAL_UNATTENDED` | `auto_reject` | 超时无人处理时自动驳回并推进；`block` 则一直等（会打 WARNING） |
| 风险阈值 | `SERVER_APPROVAL_THRESHOLD` / `SERVER_APPROVAL_DENY_THRESHOLD` | `medium` / `critical` | 高过前者问人，达到后者直接拒；两条线都由部署方配 |
| 执行前计划审批 | `SERVER_PLAN_APPROVAL` | 关 | 计划生成后停下来等人批准再执行；驳回则带着意见退回重规划，改出来的计划同样过审。**需要有人审**：通道不是 `http` 时开了它启动即失败 |
| 多数据源 | `DATASOURCE_SOURCES` | 空 | 配置命名 MySQL/PG 源，`sql_query` 按需切换 |
| 对象存储 | `MINIO_ENABLED` | 关 | VFS 大文件落 MinIO；默认落本地磁盘（`VFS_LOCAL_ROOT`） |
| 事件落盘 | `EVENT_ENABLED` | 开 | 一个任务一个 append-only JSONL（`EVENT_DIR`）。控制台据此**回放已结束的任务**；不落盘则跑完就只剩状态快照。保留 `EVENT_RETENTION_DAYS` 天，启动时清理 |

完整取值见 [`.env.example`](.env.example)，规则 JSON 写法见 `harness/config.py` 各 Settings 类的 docstring。安全相关能力均为 fail closed。

部署形态分三档，切换只靠配置、不改代码：**Dev**（零外部容器，开箱即用）/ **Standard**（单机 + PostgreSQL/pgvector）/ **Scale**（完整中间件栈）。每档的关键配置与已实测结论见 [`deploy/README.md`](deploy/README.md)。

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
│   ├── memory/             # 长期向量记忆（后端可切换）
│   ├── planning/           # 规划、任务存储、质量门
│   ├── vfs/                # 虚拟文件系统
│   ├── skills/             # 通用技能加载器（SKILL.md → 渐进式披露）
│   ├── domain/             # 领域包机制：发现 / 挂载 / 卸载 / 回收
│   ├── trace/              # tracer + Kafka 生产
│   ├── sandbox/            # 沙箱客户端与执行器
│   └── server/             # FastAPI（app/auth/service/schemas/run）
│       └── static/         #   零构建控制台（index.html + assets/，原生 HTML/CSS/JS）
├── packages/               # 领域包（以 entry points 挂载到框架）
│   └── data_analysis/      #   数据分析：tools / agents / skills / 离线能力
├── infra/                  # docker-compose、沙箱镜像、opensandbox-server
├── examples/               # 端到端示例
└── tests/                  # pytest 测试
```

## 扩展

- **加一个领域（工具 + 角色 + 技能一起）**：实现 `DomainPackage`（`name` / `version` / `requires` / `contributes` / `apply`），在 `apply` 里用 `PackageContext` 注册三样东西，再到 `pyproject.toml` 的 `governed.domain_packages` 分组声明 entry point。框架启动时自动发现并挂载，**卸载时逐项回收**（工具注册、角色注册、技能条目、两处按名归属的缓存）。框架侧不出现任何领域名词。
- **加工具**：在包的 `apply` 里 `ctx.register_tools([(ToolDef, handler), ...])` —— `handler(args, context) -> (ok, text, artifacts)`，与调用协议（ReAct / Function Calling / MCP）解耦
- **加子 Agent**：`ctx.register_agents([...])`，定义其工具白名单、系统提示与步数/超时预算
- **加 Skill**：按约定写 `SKILL.md` 放进包的技能目录，由包声明该目录；技能目录下的 `references/` 是**按需附件**，正文指到时才经 `skill_reference` 读取，不占默认上下文
- **加中间件**：继承 `Middleware`，实现需要的 hook，注册到 manager（工具结果缓存在 Broker 链路里而非中间件里 —— 因为它的命中必须落在权限/熔断/限流**之后**）

## Roadmap

- [ ] 真实 LLM 稳定性收敛：减少幻觉文件名与多余的工具调用
- [ ] 控制台：执行前的计划审批（当前为只读计划视图）、事件落盘与历史回放
- [ ] 沙箱控制面纳入 docker-compose（当前用独立脚本过渡）
- [ ] 真实 MySQL/PostgreSQL 端到端联调
- [ ] 补充架构设计文档、提升测试覆盖率

## License

基于 [MIT License](LICENSE) 开源。
