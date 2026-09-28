# Governed 评测体系（evals）

回答一个问题：**怎么知道这个 Agent 做得好不好，而不是凭感觉？**

评测复用真实的 Plan-and-Execute 链路（同一套工具、编排、管控），只把"决策来源"
换成可复现的脚本 LLM 或真实模型，然后对每次运行采集轨迹、按用例期望打分。

## 两个聚合口径

| 指标 | 定义 | 衡量 |
|---|---|---|
| **pass@k** | k 次运行中**至少一次**成功 | 能力**上限**：模型有没有可能做对 |
| **pass^k** | k 次运行**全部**成功 | 一致性**下限**：生产可不可靠 |

> 这里直接跑 k 次得到经验估计。真实 LLM 才有随机性，脚本 LLM 每次结果一致，
> 用于验证评测管线本身、作为对照基线。

## 五层指标

| 层 | 权重 | 看什么 |
|---|---|---|
| completion 任务完成 | 0.35 | 终态正确、必需角色成功、有最终报告 |
| tool_select 工具选择 | 0.20 | 该用的工具都用了、禁用工具零出现 |
| trajectory 轨迹质量 | 0.15 | 派发角色成功率、期望产物是否产出 |
| cost 成本效率 | 0.15 | 总步数 / 工具调用次数相对预算 |
| safety 安全合规 | 0.15 | PDP 拒绝、沙箱使用是否符合预期 |

"任务成功"（计入 pass@k）的判据更严格：终态正确 + 有报告 + 必需角色全部成功。

## 用法

```bash
# 离线确定性：脚本 LLM 跑 1 次（CI / 回归基线，无需 API Key）
.venv/Scripts/python -m evals.run_evals

# 真实模型跑 5 次，得到 pass@5 / pass^5 与五层均值
.venv/Scripts/python -m evals.run_evals --llm real --runs 5

# 按标签筛选、指定输出
.venv/Scripts/python -m evals.run_evals --tags smoke --out data/evals/smoke.json
```

结果同时打印到终端并写入 JSON（默认 `data/evals/evals_<时间戳>.json`）。

## 数据从哪来（可追溯）

- **工具轨迹 / PDP 拒绝 / 是否走沙箱**：每次运行用唯一 `session_id`，运行后从
  本地审计日志 `data/audit/audit.jsonl` 精确还原；
- **产物**：运行前后对 VFS（workspace + reports）做文件快照 diff，按扩展名归类；
- **成本**：汇总各 `SubAgentResult` 的步数与耗时；真实 LLM 还可采集 token usage。

## 加一个评测用例

在 `evals/cases.py` 的 `default_cases()` 里追加一个 `EvalCase`，写清目标和期望
（终态 / 必需角色 / 必需工具 / 产物 / 安全事件 / 步数上限）即可，打分无需改动。

需要自定义决策来源时，照 `runner.scripted_llm_factory` 写一个返回 LLM 实例的
零参工厂（LLM 需实现 `chat` / `chat_json` 契约）。

## 设计原则

- **确定性优先**：默认离线、可复现，评测结果不依赖网络与模型波动；
- **测真实链路**：不另写"评测专用 Agent"，被测对象就是生产代码路径；
- **只记录事实、打分分离**：runner 采集 RunTrace，metrics 按用例期望计算，
  调整权重或判据不必重跑数据。
