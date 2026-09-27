# ETL-Harness · 工业级 Agent Harness Engineering 框架

> 基于 LangGraph 编排引擎 + 手写 Harness Engineering 管控层的企业级 Agent 框架。
> 完整实现：任务规划、虚拟文件系统、子 Agent 委派、上下文管理、可插拔中间件、Skills 技能系统、全生命周期记忆（Redis+Milvus）、安全沙箱（Docker）、链路追踪（Kafka）、FastAPI 服务化。

---

## 目录

- [一、项目定位](#一项目定位)
- [二、设计哲学](#二设计哲学)
- [三、架构总览](#三架构总览)
- [四、核心模块详解](#四核心模块详解)
- [五、技术栈与选型](#五技术栈与选型)
- [六、目录结构](#六目录结构)
- [七、快速开始](#七快速开始)
- [八、核心概念](#八核心概念)
- [九、与 LangChain / LangGraph 的关系](#九与-langchain--langgraph-的关系)
- [十、扩展指南](#十扩展指南)
- [十一、学习路径](#十一学习路径)
- [十二、面试要点](#十二面试要点)

---

## 一、项目定位

### 1.1 是什么

ETL-Harness 是一个**工业级 Agent Harness Engineering 框架**。它提供了企业级 Agent 开发所需的完整运行时能力层：任务规划、虚拟文件系统、子 Agent 委派、上下文管理、可插拔中间件、Skills 技能系统、全生命周期记忆、安全沙箱、链路追踪。开发者注册业务工具并组装，即可得到一个带完整工业级管控的 Agent。

### 1.2 解决什么问题

| 问题 | 本框架的解法 |
|------|-------------|
| 复杂任务无法拆解，单 Agent 力不从心 | 任务规划层：LLM 拆解为子任务，状态追踪，子 Agent 委派 |
| 大结果塞满上下文，Token 爆炸 | 虚拟文件系统 + 上下文管理：大结果沉淀文件，prompt 只保留摘要 |
| 工具调用无法统一管控 | Tool Broker + 可插拔中间件：统一入口，Hook 机制支持缓存/重试/PII/权限 |
| 权限控制靠 Prompt 约束 LLM 自觉 | PDP 策略决策点：确定性权限判断，不赌 LLM |
| 记忆就是对话列表，没有层次 | 全生命周期记忆：Redis 短期 + Milvus 长期向量 + 工作记忆 |
| 领域知识无法复用 | Skills 技能系统：模板/SOP/Prompt/SQL/脚本打包为可复用 Skill |
| 执行不可信代码不安全 | 安全沙箱：Docker 容器双层隔离，网络/进程/文件系统隔离 |
| 出了问题不知道 Agent 干了什么 | 链路追踪：全链路埋点，Kafka 上送，执行链路可视化 |
| 长任务崩溃后从头开始 | Checkpoint 断点恢复 + Redis 任务持久化：崩溃后从断点继续 |
| 框架无法服务化 | FastAPI+ASGI：高性能 HTTP 服务，同步/流式 API |

### 1.3 不是什么

- ❌ 不是某个具体业务应用（框架核心通用，换工具=换应用）
- ❌ 不是 LangChain 的替代品（LangGraph 做编排引擎，本框架做 Harness 管控层）
- ❌ 不是面向终端用户的产品（是框架库 + API 服务，需要前端或客户端接入）

---

## 二、设计哲学

### 2.1 框架只做管控，业务由你注入

框架不绑定任何业务。工具决定业务，框架负责管控。

```
换不同的工具 = 不同的应用
├── 注册数据清洗+EDA+SQL+统计 → 数据分析 Agent
├── 注册特征分析+规则配置+测试验证 → 风控策略运维 Agent
├── 注册订单查询+退款+物流 → 电商客服 Agent
└── 注册代码执行+文件读写 → 代码 Agent
```

### 2.2 确定性优先，LLM 只做推理

- 任务状态管理、工具执行、权限判断、审计记录、文件操作 → 确定性代码
- 思考、决策、自然语言理解、任务拆解 → LLM
- 关键路径上有 LLM 的地方，都有确定性兜底

### 2.3 管控点横切，不侵入业务

PDP 权限检查、审计记录、限流、PII 检测、缓存、重试——这些横切关注点通过中间件 Hook 机制统一注入，业务工具的实现函数不需要知道这些管控逻辑的存在。

### 2.4 大结果沉淀，小上下文推理

工具执行的大结果（长日志、测试报告、规则 diff、数据集）沉淀到虚拟文件系统，模型上下文中只保留阶段性摘要、文件链接和下一步计划。减少 Token 消耗，提升长任务稳定性。

### 2.5 可观测性内建

Agent 的每一步思考、每次工具调用、每个状态变更、每次文件操作、每次子 Agent 委派都可追溯。不是出了问题再加日志，而是设计时就把链路追踪做进去。

---

## 三、架构总览

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        接入层 (FastAPI + ASGI)                            │
│   POST /agent/run  │  POST /agent/stream (SSE)  │  查询/恢复/审批 API    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│                      编排层 (LangGraph StateGraph)                        │
│                                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐    │
│  │  planner  │───▶│  think   │───▶│  action  │───▶│   final      │    │
│  │ (任务规划)│    │ (LLM思考)│    │(工具执行) │    │  (答案整理)  │    │
│  └──────────┘    └────┬─────┘    └────┬─────┘    └──────────────┘    │
│                        │                 │                                │
│                        └──── 循环 ◄──────┘                                │
│                              (条件边)                                      │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼─────────────────────────────────────┐
│                    Harness 管控层 (手写核心)                               │
│                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Tool Broker │  │  Middleware │  │     PDP     │  │  Audit Logger│ │
│  │ 工具注册调度 │  │ 可插拔Hook  │  │ 策略决策点  │  │  审计链路    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘ │
│         │                │                 │                 │          │
│  ┌──────▼────────────────▼─────────────────▼─────────────────▼───────┐ │
│  │                    可插拔中间件 Hook 链                                │ │
│  │  before_llm → before_tool → [PDP→限流→PII→缓存] → 执行 → after_tool │ │
│  └───────────────────────────────────┬─────────────────────────────────┘ │
│                                      │                                     │
│  ┌───────────────────────────────────▼─────────────────────────────────┐ │
│  │                        运行时能力层                                    │ │
│  │                                                                       │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │ │
│  │  │  任务规划     │  │  子Agent委派  │  │     Skills 技能系统       │  │ │
│  │  │ TaskPlanner  │  │ Orchestrator │  │  模板/SOP/Prompt/SQL/脚本 │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │ │
│  │                                                                       │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │ │
│  │  │ 虚拟文件系统  │  │  上下文管理   │  │     安全沙箱             │  │ │
│  │  │ /workspace   │  │ 大结果沉淀VFS │  │  Docker容器双层隔离      │  │ │
│  │  │ /reports/logs│  │ prompt只留摘要│  │  网络/进程/文件系统隔离  │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │ │
│  └───────────────────────────────────┬─────────────────────────────────┘ │
│                                      │                                     │
│  ┌───────────────────────────────────▼─────────────────────────────────┐ │
│  │                        状态与记忆层                                    │ │
│  │                                                                       │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │ │
│  │  │ Redis短期  │  │ Milvus长期 │  │  工作记忆   │  │  Checkpoint  │ │ │
│  │  │ 对话/任务  │  │ 向量检索   │  │  key-value  │  │  断点恢复    │ │ │
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────┬───────────────────────────────────┘
                                        │
┌───────────────────────────────────────▼───────────────────────────────────┐
│                     基础设施层 (Docker Compose)                              │
│   Redis 7.x  │  Milvus 2.x  │  Kafka 3.x  │  MinIO  │  Zookeeper       │
└───────────────────────────────────────────────────────────────────────────┘
```

### 3.2 调用链路

一次完整的 Agent 运行：

```
用户请求 (HTTP API)
    │
    ▼
┌─────────────┐
│ 任务规划     │  LLM 拆解目标为子任务清单，存入 Redis
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ think 节点   │  1. 从 State 构建 prompt（goal+记忆+工具+Skills+任务进度）
│ (LLM 思考)  │  2. 中间件 before_llm Hook（缓存/日志）
│             │  3. 调 LLM
│             │  4. 中间件 after_llm Hook
│             │  5. 解析输出：thought/action/action_input/final
└──────┬──────┘
       │
       ▼ 条件边判断
       │
   ┌───┴───┐
   │有action?│
   └───┬───┘
       │
   ┌───┴──────────────────────────────────┐
   │ 是                                     │ 否
   ▼                                        ▼
┌─────────────┐                     ┌─────────────┐
│ action 节点  │                     │ final 节点  │
│ (工具执行)   │                     │ (答案整理)   │
│             │                     └──────┬──────┘
│ 1.中间件    │                            │
│   before_tool│                            ▼
│ 2.PDP权限检查│                     ┌─────────────┐
│ 3.参数校验   │                     │  链路追踪    │
│ 4.限流检查   │                     │  Kafka 上送  │
│ 5.执行工具   │                     └─────────────┘
│   (沙箱隔离) │                            │
│ 6.大结果沉淀 │                            ▼
│   到VFS      │                     ┌─────────────┐
│ 7.中间件     │                     │    END      │
│   after_tool │                     │  返回结果    │
│ 8.审计+追踪  │                     └─────────────┘
│ 9.写记忆     │
└──────┬──────┘
       │
       ▼
  普通边回到 think（循环）
```

---

## 四、核心模块详解

### 4.1 Tool Broker（工具代理）

所有工具调用过统一入口。五层防护 + 中间件 Hook：

```
Agent 请求调用工具
    │
    ▼
1. 中间件 before_tool Hook（缓存/日志/PII检测）
    │
    ▼
2. 工具存在性检查 → 不存在返回失败
    │
    ▼
3. PDP 权限检查 → 拒绝返回"权限不足"observation
    │
    ▼
4. 参数校验（JSON Schema）→ 必填缺失返回失败
    │
    ▼
5. 限流检查（滑动时间窗口，Redis 分布式计数）→ 超限返回失败
    │
    ▼
6. 调用实现函数（高风险工具走安全沙箱）→ try/except 捕获异常
    │
    ▼
7. 大结果沉淀到 VFS（上下文管理）
    │
    ▼
8. 中间件 after_tool Hook
    │
    ▼
9. 审计记录 + 链路追踪（Kafka 上送）
    │
    ▼
10. 结果包装：统一返回 (ok, text, artifacts)
```

### 4.2 可插拔中间件（Middleware）

在关键节点注入 Hook，支持横切关注点的可插拔扩展。

**Hook 点：**

| Hook 点 | 触发时机 | 典型用途 |
|---------|---------|---------|
| `before_llm` | LLM 调用前 | 缓存命中检查、Prompt 日志、成本预估 |
| `after_llm` | LLM 调用后 | 响应缓存、Token 统计、内容审核 |
| `before_tool` | 工具调用前 | PII 检测、参数脱敏、权限校验 |
| `after_tool` | 工具调用后 | 结果缓存、异常告警、性能统计 |
| `before_file` | 文件操作前 | 病毒扫描、敏感文件拦截 |
| `after_file` | 文件操作后 | 文件索引更新、版本留痕 |
| `task_state_change` | 任务状态变更 | 通知推送、进度追踪、SLA 监控 |

**内置中间件：**
- 缓存中间件（LLM 响应/工具结果缓存）
- 重试中间件（失败自动重试，指数退避）
- PII 检测中间件（敏感信息识别和脱敏）
- 权限校验中间件（PDP 集成）
- 日志采集中间件（全链路日志）
- 敏感操作拦截中间件（高风险操作人工审批）

### 4.3 任务规划（TaskPlanner）

将复杂目标拆解为可执行子任务，维护完整生命周期。

**任务状态机：**

```
pending ──开始──▶ in_progress ──成功──▶ completed
    │                  │
    │                  └──失败──▶ failed
    │
    └──取消──▶ cancelled
```

**核心能力：**
- LLM 拆解：根据用户目标生成子任务清单，包含依赖关系
- 状态追踪：每个子任务的 pending/in_progress/completed/failed 状态
- Redis 持久化：任务状态存入 Redis，崩溃后可恢复
- 进度计算：自动计算整体完成度
- 子 Agent 委派：复杂子任务委派给专业子 Agent 执行

### 4.4 虚拟文件系统（VFS）

抽象统一的文件系统接口，Agent 可持续操作的工作资产。

**目录结构：**

| 目录 | 用途 | 典型内容 |
|------|------|---------|
| `/workspace` | 工作目录 | 上传的数据文件、中间结果 |
| `/reports` | 报告目录 | 分析报告、测试报告、策略文档 |
| `/logs` | 日志目录 | 执行日志、调试信息、错误日志 |
| `/policies` | 策略目录 | 策略 DSL、规则配置、权限策略 |
| `/memories` | 记忆目录 | 导出的记忆快照、经验文档 |

**核心能力：**
- 文件读写/编辑/搜索/删除
- 版本留痕：每次修改保存版本，支持 diff 和回滚
- MinIO 后端：生产级对象存储
- 本地回退：MinIO 不可用时用本地文件系统
- 大结果沉淀：工具执行的大结果自动写入 VFS

### 4.5 上下文管理（ContextManager）

解决大结果塞满上下文的问题。

**核心策略：**
1. 工具执行结果超过阈值（如 500 字）→ 自动沉淀到 VFS
2. 上下文中只保留：阶段性摘要 + 文件链接 + 下一步计划
3. LLM 需要详细内容时 → 通过文件链接从 VFS 读取
4. 上下文窗口自动管理：短期记忆 FIFO，工作记忆按需注入

**效果：** Token 消耗降低 60-80%，长任务稳定性显著提升。

### 4.6 Skills 技能系统

将领域知识打包为可复用 Skill，渐进式披露。

**Skill 内容类型：**
- 规则模板（风控规则、数据清洗规则）
- 排查 SOP（标准操作流程）
- Prompt 模板（特定任务的优化 Prompt）
- 领域文档（业务知识、术语表）
- SQL 模板（常用查询模板）
- 脚本（可执行的分析脚本）

**渐进式披露：**
- 不是所有 Skill 都注入 prompt
- 根据当前任务上下文、用户目标、历史行为，检索最相关的 Skill
- 只注入 top-k 相关 Skill 的摘要，需要时再加载完整内容

### 4.7 全生命周期记忆

| 层级 | 存储 | 存什么 | 生命周期 | 检索方式 |
|------|------|--------|----------|---------|
| 短期记忆 | Redis | 最近N轮对话、任务计划、审批断点、临时数据 | 当前会话 | FIFO 按时间 |
| 工作记忆 | 内存 | 任务执行中的关键事实/中间结果 | 当前任务 | key-value |
| 长期记忆 | Milvus | 跨会话的规则经验、策略模板、用户偏好、历史结论 | 持久化 | 向量相似度检索 |

**长期记忆向量检索：**
1. 添加记忆时计算 Embedding，存入 Milvus
2. 检索时用 query Embedding 做余弦相似度计算
3. 返回 top-k 最相关的记忆
4. 支持按用户/会话/类型过滤

### 4.8 安全沙箱

基于 Docker 容器的双层隔离执行环境。

**隔离维度：**
- 网络隔离：默认禁止网络访问，可按需开放白名单
- 进程隔离：独立容器，不影响主机
- 文件系统隔离：独立工作目录，只读挂载必要文件
- 资源限制：内存上限、CPU 上限、执行超时

**使用场景：**
- 执行 LLM 生成的 Python 分析代码
- 执行 Skill 中的脚本
- 运行不可信的第三方工具
- 高风险操作的隔离执行

### 4.9 链路追踪

全链路埋点，Kafka 上送，执行链路可视化。

**埋点覆盖：**
- 用户请求（入口）
- Agent 规划（任务拆解）
- 子 Agent 委派（委派/执行/返回）
- 记忆访问（读/写/检索）
- 工具调用（调用前/调用后/结果）
- 文件操作（读/写/编辑/删除）
- 人工审批（申请/通过/拒绝）
- 中间评估（每步的中间状态）

**Trace 模型：**
- `trace_id`：整条链路唯一标识
- `span_id`：单个操作标识
- `parent_span_id`：父操作标识（构建调用树）
- `service_name`：服务名
- `operation`：操作名
- `start_time` / `duration_ms`：耗时
- `tags`：自定义标签（工具名/参数hash/结果等）
- `status`：成功/失败

### 4.10 子 Agent 委派（Orchestrator）

主 Agent 负责任务分发和结果汇总，专业子 Agent 独立执行。

**委派流程：**
1. 主 Agent 识别需要专业能力的子任务
2. 创建子 Agent（独立上下文、独立工具集、独立记忆）
3. 子 Agent 执行任务，仅返回结构化结论
4. 主 Agent 汇总所有子 Agent 的结论
5. 降低主上下文污染（子 Agent 的详细执行过程不进入主上下文）

**典型子 Agent：**
- 特征分析 Agent（数据分析领域）
- 规则配置 Agent（风控领域）
- 测试验证 Agent
- 异常排查 Agent
- 知识检索 Agent

---

## 五、技术栈与选型

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 语言 | Python | 3.10+ | 主语言 |
| 编排引擎 | LangGraph | 1.x | 状态图编排 + Checkpoint |
| 消息格式 | langchain-core | 0.3.x | 标准消息格式 |
| LLM 调用 | OpenAI SDK + httpx | 1.x | OpenAI 兼容接口 |
| 数据校验 | Pydantic | 2.x | 数据结构定义 |
| 配置管理 | pydantic-settings | 2.x | 环境变量配置 |
| 短期记忆 | Redis | 7.x | 对话/任务/缓存/限流 |
| 长期记忆 | Milvus | 2.x | 向量相似度检索 |
| 事件总线 | Kafka | 3.x | 审计/链路追踪/事件 |
| 文件存储 | MinIO | latest | VFS 对象存储后端 |
| 服务层 | FastAPI + uvicorn | 0.104+ | HTTP API + SSE 流式 |
| 安全沙箱 | Docker SDK | 6.x | 容器隔离执行 |
| 韧性 | tenacity | 8.x | 重试机制 |
| 异步文件 | aiofiles | 23.x | 异步文件 IO |

---

## 六、目录结构

```
ETL-Harness/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── infra/                             # 基础设施编排
│   └── docker-compose.yml             # Redis+Milvus+Kafka+MinIO+Zookeeper
│
├── harness/                           # 核心框架库
│   ├── __init__.py
│   ├── config.py                      # 全局配置
│   ├── models.py                      # 数据模型
│   ├── state.py                       # LangGraph State
│   ├── llm_client.py                  # LLM 客户端
│   ├── tool_broker.py                 # 工具调度
│   ├── middleware.py                  # 可插拔中间件
│   ├── pdp.py                         # 权限决策
│   ├── audit.py                       # 审计日志
│   ├── orchestrator.py                # 子 Agent 委派
│   ├── nodes.py                       # ReAct 节点
│   ├── graph.py                       # ReAct 图构建
│   ├── memory/                        # 记忆层
│   │   ├── short_term.py              # Redis 短期记忆
│   │   ├── long_term.py               # Milvus 长期向量
│   │   └── working.py                 # 工作记忆
│   ├── planning/                      # 任务规划
│   │   ├── planner.py
│   │   └── task_store.py
│   ├── vfs/                           # 虚拟文件系统
│   │   ├── vfs.py
│   │   ├── storage.py
│   │   └── versioning.py
│   ├── context/                       # 上下文管理
│   │   └── manager.py
│   ├── skills/                        # Skills 系统
│   │   ├── skill.py
│   │   ├── registry.py
│   │   └── loader.py
│   ├── trace/                         # 链路追踪
│   │   ├── tracer.py
│   │   └── kafka_producer.py
│   ├── sandbox/                       # 安全沙箱
│   │   ├── executor.py
│   │   └── isolator.py
│   └── server/                        # FastAPI 服务
│       ├── app.py
│       ├── routes.py
│       ├── schemas.py
│       └── deps.py
│
├── tools/                             # 数据分析工具集
│   ├── data_inspector.py
│   ├── data_cleaner.py
│   ├── eda.py
│   ├── sql_query.py
│   ├── chart_generator.py
│   └── code_executor.py
│
├── examples/                          # 可运行示例
├── tests/                             # 单元测试
├── docs/                              # 文档
└── data/                              # 运行时数据
```

---

## 七、快速开始

### 7.1 环境准备

```bash
# 1. 克隆项目
cd D:\ETL-Harness

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
copy .env.example .env
# 编辑 .env，填入 API Key 和基础设施配置
```

### 7.2 启动基础设施

```bash
# 确保 Docker Desktop 已启动
cd infra
docker-compose up -d

# 验证服务状态
docker-compose ps
# redis: 6379
# milvus: 19530
# kafka: 9092
# minio: 9000 (控制台: 9001)
```

### 7.3 用 Mock LLM 跑通（不需要 API Key）

```bash
python examples\mock_llm_demo.py
```

### 7.4 启动 FastAPI 服务

服务端**默认开启鉴权**：未配置令牌会**直接启动失败**（避免"以为开了、实际没开"）。先配令牌：

```bash
# .env
AUTH_ENABLED=true
AUTH_TOKENS={"<长随机令牌>":{"role":"analyst","name":"张三"},"<另一个令牌>":{"role":"admin","name":"李四"}}
AUTH_APPROVER_ROLES=admin
```

启动：

```bash
python -m uvicorn --factory harness.server.app:create_app --host 0.0.0.0 --port 8000
```

> 注意 `--factory`：应用由 `create_app()` 工厂构造，模块里**没有** `app` 变量。
> `/docs`、`/openapi.json` 同样需要令牌；只有 `/health` 匿名（给探活用）。
> 本机开发可设 `AUTH_ENABLED=false`：服务端会打 WARNING，且只有一个身份（admin）。

### 7.5 调用 Agent API

所有业务路由都需要 `Authorization: Bearer <令牌>`；**角色由令牌决定**，请求体里没有 `role` 字段。

```bash
TOKEN=<令牌>

# 创建任务
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"goal": "分析这份销售数据的趋势"}'

# 查询状态 / 待审批项（THREAD 为上一步返回的 thread_id）
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks/$THREAD
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/tasks/$THREAD/approvals

# 提交审批：需审批角色，且**不能批准自己发起的任务**
curl -X POST http://localhost:8000/api/v1/tasks/$THREAD/approval \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"approved": false, "comment": "来源不明，不允许"}'

# 流式订阅：浏览器原生 EventSource 不能设请求头，故**这条路由**额外接受 ?token=
curl -N "http://localhost:8000/api/v1/tasks/$THREAD/stream?token=$TOKEN"
```

**迁移提示（2026-09-27 起）**：旧客户端若在请求体里传 `role`，会被忽略（不报错，但也不再生效）；
请改为在 `AUTH_TOKENS` 里给对应令牌配角色。

---

## 八、核心概念

### 8.1 ReAct 循环

Thought → Action → Observation 循环，LLM 思考决策，确定性代码执行。

### 8.2 Harness Engineering

Agent 开发中的"管控工程"——不关注 LLM 本身，而关注围绕 LLM 的工程管控：工具调度、权限、记忆、审计、安全、可观测性。

### 8.3 大结果沉淀

工具执行的大结果不直接塞进 prompt，而是写入虚拟文件系统，上下文中只保留摘要和文件链接。LLM 需要时通过链接读取。

### 8.4 渐进式披露

Skills 不全部注入 prompt，而是根据当前任务上下文检索最相关的 top-k，只注入摘要，需要时再加载完整内容。

### 8.5 子 Agent 委派

主 Agent 不亲自执行所有任务，而是将专业子任务委派给独立的子 Agent，子 Agent 有独立的上下文和工具集，仅返回结构化结论，降低主上下文污染。

---

## 九、与 LangChain / LangGraph 的关系

| 技术 | 定位 | 本框架中的角色 |
|------|------|----------------|
| LangChain | LLM 应用开发框架 | 只用 langchain-core 消息格式 |
| LangGraph | 状态图编排库 | 编排引擎（骨架） |
| ETL-Harness | Agent Harness Engineering 框架 | 手写的管控层（血肉） |

**用 LangGraph 做骨架（状态图+Checkpoint+流式），Harness 管控层的血肉全部自己写。**

---

## 十、扩展指南

### 10.1 加一个新工具

1. 定义 ToolDef（name/description/parameters/required_role/rate_limit）
2. 写实现函数 `def handler(args, context) -> tuple[bool, str, dict]`
3. 注册到 Broker：`broker.register(tool_def, handler)`

### 10.2 加一个中间件

1. 继承 Middleware 基类
2. 实现需要的 Hook 方法（before_llm/after_llm/before_tool/after_tool 等）
3. 注册到 MiddlewareManager：`manager.register(my_middleware)`

### 10.3 加一个 Skill

1. 继承 Skill 基类
2. 定义名称/描述/触发条件/内容
3. 注册到 SkillRegistry：`registry.register(my_skill)`

### 10.4 加一个子 Agent

1. 定义子 Agent 的工具集和系统提示
2. 在 Orchestrator 中注册：`orchestrator.register_sub_agent("feature_analysis", sub_agent_config)`
3. 主 Agent 通过 `orchestrator.delegate("feature_analysis", task)` 委派

---

## 十一、学习路径

> 本章标记项目中每个模块对应的学习目标和掌握程度。
> **⭐ 核心学习模块**：需要逐行理解语法和设计，我给代码+逐步教。
> **📖 了解模块**：知道干什么的、怎么用就行，不需要逐行学语法。

### 11.1 你的学习目标清单

| # | 学习目标 | 项目内覆盖情况 | 对应模块 |
|---|---------|--------------|---------|
| 1 | **Python 基础** | ✅ 全程覆盖 | 所有模块（类型标注/字典/类/异常/文件IO/装饰器） |
| 2 | **类与多态** | ✅ 核心覆盖 | tool_broker / memory / middleware（基类继承、魔术方法、ABC抽象基类） |
| 3 | **LangGraph** | ✅ 核心覆盖 | state.py / nodes.py / graph.py（StateGraph/节点/条件边/Checkpointer） |
| 4 | **Agent 框架** | ✅ 全程覆盖 | 整个项目就是 Agent Harness 框架（ReAct/Tool Broker/PDP/记忆/审计） |
| 5 | **FastAPI** | ✅ 项目覆盖 | server/（路由/依赖注入/SSE流式/Pydantic Schema） |
| 6 | **Skill** | ✅ 项目覆盖 | skills/（Skill基类/注册中心/渐进式披露加载） |
| 7 | **写 Prompt 的格式** | ✅ 项目覆盖 | nodes.py（System Prompt / ReAct Prompt / 工具描述注入 /  Few-shot） |
| 8 | **MCP** | ⚠️ 项目未内置，扩展方向 | 可用 FastMCP 将框架工具暴露为 MCP 服务（见 10.5 扩展） |
| 9 | **异步编程** | ⚠️ 可选，项目以同步为主 | server/（ASGI异步）、llm_client（可扩展async）、aiofiles（异步文件IO） |

### 11.2 模块学习地图

#### ⭐ 核心学习模块（必须逐行理解）

| 顺序 | 模块 | 学习目标 | 关键语法点 | 状态 |
|------|------|---------|-----------|------|
| 1 | `tool_broker.py` | Python基础 + 类设计 + 统一入口模式 | 类型标注、字典操作、self、异常处理、限流算法、Callable类型别名 | ✅ 已学完 |
| 2 | `memory/working.py` | 类封装 + 魔术方法 | `__init__`、`self._data`、三元表达式、字典推导式、`__len__`/`__contains__`/`__getitem__`/`__setitem__` | ⏳ 正在学 |
| 3 | `memory/short_term.py` | Redis操作 + try/except降级 + FIFO | Redis客户端、`try/except`、连接降级模式、List的`rpush/ltrim`、JSON序列化 | 待学 |
| 4 | `memory/long_term.py` | 向量检索 + Embedding + 外部服务集成 | OpenAI Embedding API、pymilvus、余弦相似度、本地回退模式 | 待学 |
| 5 | `middleware.py` | 类与多态 + 继承 + 上下文管理器 | ABC抽象基类、继承、方法重写、`@contextmanager`、Hook模式、责任链 | 待学 |
| 6 | `nodes.py` | LangGraph + Prompt格式 + LLM输出解析 | LangGraph节点函数约定、Prompt构建（System/Few-shot/工具描述）、JSON解析、条件边函数 | 待写待学 |
| 7 | `graph.py` | LangGraph图构建 + Checkpoint | StateGraph、`add_node`/`add_edge`/`add_conditional_edges`、`set_entry_point`、编译、SqliteSaver | 待写待学 |

#### 📖 了解模块（知道干什么、怎么用即可）

| 模块 | 干什么的 | 怎么用 |
|------|---------|--------|
| `config.py` | 全局配置管理，从.env加载所有配置 | `from harness.config import settings; settings.redis.host` |
| `models.py` | 25个Pydantic数据模型，所有跨模块数据结构 | `from harness.models import ToolDef, TaskPlan` |
| `llm_client.py` | OpenAI兼容LLM客户端，JSON解析自修 | `client.chat_json(messages)` |
| `state.py` | LangGraph State定义（18+字段） | `AgentState` TypedDict |
| `pdp.py` | 权限决策点，细粒度权限判断 | `pdp.check(role, tool_name, context)` |
| `audit.py` | 审计日志，Kafka上报+本地回退 | `audit.log(record)` |
| `trace/` | 链路追踪，全链路埋点Kafka上送 | `with tracer.span("operation"):` |
| `vfs/` | 虚拟文件系统，MinIO+本地回退，版本留痕 | `vfs.write(path, content)` / `vfs.read(path)` |
| `planning/` | 任务规划，LLM拆解子任务，Redis持久化 | `planner.plan(goal)` / `planner.update_status(task_id, status)` |
| `skills/` | Skills技能系统，领域知识打包，渐进式披露 | `registry.register(skill)` / `loader.load_relevant(context)` |
| `context/` | 上下文管理，大结果沉淀VFS，prompt只留摘要 | `ctx_manager.sink_large_result(tool_name, result)` |
| `sandbox/` | 安全沙箱，Docker容器双层隔离 | `sandbox.execute(code)` |
| `orchestrator.py` | 子Agent委派，主Agent分发专业子Agent | `orchestrator.delegate(sub_agent_name, task)` |
| `server/` | FastAPI服务，HTTP API + SSE流式 | `uvicorn harness.server.app:app` |
| `tools/` | 数据分析工具集（6个工具） | 注册到Broker即可用 |

### 11.3 学习顺序建议

```
阶段1：Python基础 + 类（已完成 tool_broker）
    │
    ▼
阶段2：记忆层（working → short_term → long_term）
    │  学：类封装、魔术方法、Redis、向量检索、降级模式
    ▼
阶段3：中间件（middleware.py）
    │  学：继承、多态、ABC抽象基类、上下文管理器、Hook模式
    ▼
阶段4：LangGraph核心（state.py → nodes.py → graph.py）
    │  学：TypedDict、节点函数、条件边、StateGraph、Checkpointer
    │  学：Prompt格式（System/Few-shot/工具描述/ReAct）
    ▼
阶段5：Agent框架整合（跑通完整ReAct循环）
    │  理解：LLM→Broker→工具→观察→LLM 的完整闭环
    ▼
阶段6：FastAPI服务（server/）
    │  学：路由、依赖注入、Pydantic Schema、SSE流式
    ▼
阶段7：扩展能力（Skills / MCP / 异步编程）
       Skills：项目内 skills/ 模块
       MCP：用 FastMCP 暴露工具（额外学习）
       异步：server/ 的 ASGI 异步模式（可选深入）
```

### 11.4 各学习目标在项目中的具体落点

#### Python 基础
- **类型标注**：所有模块的函数参数/返回值/变量标注
- **字典操作**：tool_broker 的 `_tools` 注册表、memory 的 `_data`
- **异常处理**：tool_broker 的 `invoke` 方法 try/except、所有外部服务连接的降级
- **文件IO**：vfs 的本地存储后端、audit 的 JSON Lines 写入
- **装饰器**：`@contextmanager`（middleware/tracer）、`@property`
- **推导式**：memory 的字典推导式、列表推导式

#### 类与多态
- **类的基本结构**：所有模块都是类（`__init__`、实例方法、self）
- **继承**：middleware 的 `Middleware` 基类 → `LoggingMiddleware`/`RetryMiddleware` 等子类
- **多态**：不同中间件实现同一个 `before_tool` 方法，行为不同
- **ABC抽象基类**：`Middleware(ABC)` 定义接口规范
- **魔术方法**：`WorkingMemory` 的 `__len__`/`__contains__`/`__getitem__`/`__setitem__`
- **类方法 vs 静态方法 vs 实例方法**：各模块中的方法设计

#### LangGraph
- **StateGraph**：graph.py 中的图构建
- **节点函数**：nodes.py 中的 think_node/action_node/final_node
- **条件边**：`should_continue` 函数 + `add_conditional_edges`
- **普通边**：action → think 的循环边
- **Checkpointer**：SqliteSaver 断点恢复
- **add_messages**：state.py 中的消息合并语义
- **TypedDict**：AgentState 状态定义

#### Agent 框架
- **ReAct循环**：think → action → observation → think
- **Tool Broker**：统一工具调度入口
- **PDP**：权限决策点
- **分层记忆**：短期/工作/长期
- **审计链路**：全链路可追溯
- **任务规划**：复杂目标拆解
- **子Agent委派**：多Agent编排
- **中间件**：横切关注点注入
- **上下文管理**：大结果沉淀

#### FastAPI
- **路由定义**：`@app.post("/agent/run")`
- **依赖注入**：`deps.py` 中的 Broker/PDP/Memory 实例注入
- **Pydantic Schema**：`schemas.py` 的请求/响应模型
- **SSE流式**：`/agent/stream` 的 Server-Sent Events
- **ASGI**：uvicorn 异步服务器
- **CORS/异常处理/生命周期**：app.py 中的应用配置

#### Skill
- **Skill基类**：`skills/skill.py` 的 Skill 定义
- **注册中心**：`skills/registry.py` 的注册/检索/分类
- **渐进式披露**：`skills/loader.py` 的相关度检索 + top-k 加载
- **Skill内容类型**：模板/SOP/Prompt/文档/SQL/脚本

#### 写 Prompt 的格式
- **System Prompt**：nodes.py 中的系统提示（角色定义/能力边界/输出格式约束）
- **ReAct Prompt**：Thought/Action/Observation 格式约定
- **工具描述注入**：`broker.list_tool_descriptions()` 生成的工具列表
- **Few-shot示例**：nodes.py 中的示例注入
- **记忆注入**：短期/工作/长期记忆的格式化注入
- **输出格式约束**：要求 LLM 输出 JSON，包含 thought/action/action_input/final_answer

#### MCP（项目未内置，扩展学习）
- 项目当前没有 MCP 模块
- 扩展方向：用 `FastMCP` 将框架的 Tool Broker 工具暴露为 MCP 服务
- 学习路径：安装 FastMCP → 定义 MCP 工具 → 启动 MCP 服务 → 客户端连接
- 这是独立于本项目的额外学习目标

#### 异步编程（可选深入）
- 项目以同步为主，但以下位置涉及异步：
  - **server/**：FastAPI 本身是 ASGI 异步框架
  - **llm_client**：可扩展为 `async def chat()`
  - **aiofiles**：requirements 中的异步文件IO库
  - **pytest-asyncio**：异步测试
- 如果时间有限，先掌握同步版本，异步作为进阶学习

### 11.5 已完成 vs 待学习进度

| 模块 | 代码状态 | 学习状态 |
|------|---------|---------|
| tool_broker.py | ✅ 已完成 | ✅ 已学完（大量语法问答） |
| memory/working.py | ✅ 已完成 | ⏳ 正在学（代码已贴，语法点已讲） |
| memory/short_term.py | ✅ 已完成 | ⏳ 待学 |
| memory/long_term.py | ✅ 已完成 | ⏳ 待学 |
| middleware.py | ✅ 已完成 | ⏳ 待学 |
| config.py | ✅ 已完成 | 📖 了解 |
| models.py | ✅ 已完成 | 📖 了解 |
| trace/ | ✅ 已完成 | 📖 了解 |
| vfs/ | ✅ 已完成 | 📖 了解 |
| state.py | ✅ 已完成 | ⏳ 待学（LangGraph部分） |
| nodes.py | ⏳ 待写 | ⏳ 待学 |
| graph.py | ⏳ 待写 | ⏳ 待学 |
| planning/ | ⏳ 待写 | 📖 了解 |
| skills/ | ⏳ 待写 | 📖 了解 |
| context/ | ⏳ 待写 | 📖 了解 |
| audit.py | ⏳ 待升级 | 📖 了解 |
| pdp.py | ✅ 已有 | 📖 了解 |
| sandbox/ | ⏳ 待写 | 📖 了解 |
| orchestrator.py | ⏳ 待写 | 📖 了解 |
| server/ | ⏳ 待写 | ⏳ 待学（FastAPI部分） |
| tools/ | ⏳ 待写 | 📖 了解 |

---

## 十二、面试要点

### 必背核心概念

1. **Harness Engineering**：Agent 开发中的管控工程，围绕 LLM 的工程层
2. **任务规划**：复杂目标拆解为子任务，状态机管理，Redis 持久化
3. **虚拟文件系统**：大结果沉淀，目录抽象，版本留痕，MinIO 后端
4. **上下文管理**：大结果不塞 prompt，只留摘要+文件链接，省 Token
5. **可插拔中间件**：Hook 机制，横切关注点（缓存/重试/PII/权限）可插拔
6. **Skills 系统**：领域知识打包为可复用 Skill，渐进式披露
7. **全生命周期记忆**：Redis 短期 + Milvus 长期向量 + 工作记忆
8. **安全沙箱**：Docker 容器双层隔离，网络/进程/文件系统隔离
9. **链路追踪**：全链路埋点，Kafka 上送，trace_id/span_id 调用树
10. **子 Agent 委派**：主 Agent 分发，子 Agent 独立上下文执行，仅返回结构化结论
11. **Tool Broker**：统一入口，五层防护 + 中间件 Hook
12. **PDP**：细粒度权限决策，确定性判断不赌 LLM
13. **ReAct 循环**：LangGraph 状态图实现，think/action/final
14. **Checkpoint**：LangGraph 断点恢复，每步状态持久化

### 项目描述模板

> "我实现了一个工业级 Agent Harness Engineering 框架，基于 LangGraph 编排引擎和手写管控层。框架包含完整的运行时能力层：任务规划（复杂目标拆解为子任务，Redis 持久化状态）、虚拟文件系统（大结果沉淀，MinIO 后端，版本留痕）、子 Agent 委派（主 Agent 分发，专业子 Agent 独立上下文执行）、上下文管理（大结果不塞 prompt，只留摘要+文件链接，省 60-80% Token）、可插拔中间件（Hook 机制支持缓存/重试/PII检测/权限校验）、Skills 技能系统（领域知识打包为可复用 Skill，渐进式披露）、全生命周期记忆（Redis 短期 + Milvus 长期向量检索）、安全沙箱（Docker 容器双层隔离）、链路追踪（全链路埋点，Kafka 上送）、FastAPI 服务化。框架不绑定业务，注册数据分析工具就是数据分析 Agent，注册风控工具就是风控 Agent。"

---

*文档版本：v3.0（工业级 Harness Engineering）*
*最后更新：2026-09-23*
