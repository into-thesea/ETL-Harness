"""evals.runner —— 复用真实 Plan-and-Execute 链路，跑 k 次并采集轨迹。

设计要点：
- 每次运行给一个唯一 session_id，图内全部审计记录都挂在它上面，
  运行后从 audit.jsonl 精确还原"工具调用序列 / PDP 拒绝 / 是否走沙箱"；
- 运行前后对 VFS 做文件快照 diff，得到本次真实新增产物（按扩展名归类）；
- LLM 由工厂产生：scripted（确定性回放，离线、可复现）或 real（真实模型，
  多次运行才有分布，pass@k/pass^k 才有意义）。
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from harness.agents.registry import AgentRegistry
from harness.audit import get_audit_logger
from harness.context import ContextManager
from harness.orchestrator import build_plan_execute_graph, make_plan_execute_state
from harness.planning import QualityGate, TaskPlanner, TaskStore
from harness.tool_broker import ToolBroker
from harness.vfs import VirtualFileSystem
from tools import register_builtin_tools
from tools.common import reports_dir, workspace_dir

LLMFactory = Callable[[], Any]


# ----------------------------------------------------------------------
# 轨迹数据结构
# ----------------------------------------------------------------------
@dataclass
class RunTrace:
    """一次运行采集到的全部事实（只记录，不含打分）。"""

    case_id: str
    run_index: int
    session_id: str
    llm_mode: str

    # 终态
    status: str
    final_answer: str
    error: Optional[str]

    # 子 Agent 级
    agents: List[Dict[str, Any]] = field(default_factory=list)

    # 工具轨迹（来自审计）
    tools_used: List[str] = field(default_factory=list)
    tool_calls: int = 0
    denials: int = 0
    sandbox_calls: int = 0

    # 聚合成本
    total_steps: int = 0
    total_duration_ms: int = 0

    # 产物
    artifact_kinds: List[str] = field(default_factory=list)
    new_files: List[str] = field(default_factory=list)

    token_usage: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------
# LLM 工厂
# ----------------------------------------------------------------------
def scripted_llm_factory() -> LLMFactory:
    """确定性脚本 LLM 工厂：每次返回一个新的回放器，并确保脏数据存在。"""

    def _make() -> Any:
        from examples.data_analysis_demo import (
            CLEAN_STEM,
            RAW_FILE,
            ScriptedAnalysisLLM,
            make_dirty_data,
        )

        make_dirty_data()
        return ScriptedAnalysisLLM(RAW_FILE, CLEAN_STEM)

    return _make


def real_llm_factory() -> LLMFactory:
    """真实 LLM 工厂：按当前配置（.env）构造 LLMClient。"""

    def _make() -> Any:
        from harness.config import settings

        key = (settings.llm.api_key or "").strip()
        if not key:
            raise RuntimeError(
                "真实 LLM 评测需要配置 API Key（.env 的 LLM_API_KEY / DEEPSEEK_API_KEY）"
            )
        from harness.llm_client import LLMClient

        return LLMClient(
            api_key=key,
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            timeout=float(settings.llm.timeout_seconds),
        )

    return _make


# ----------------------------------------------------------------------
# 图构建（同步，与 examples.data_analysis_demo 同一装配方式）
# ----------------------------------------------------------------------
def build_graph(llm: Any) -> Any:
    """构造一次完整的顶层 Plan-and-Execute 编译图（同步内核）。"""
    broker = ToolBroker(audit_logger=get_audit_logger())
    register_builtin_tools(broker)
    registry = AgentRegistry()
    store = TaskStore(backend="memory")
    # 评测以硬校验质量门为准，关闭语义 Critic，避免引入额外 LLM 噪声
    gate = QualityGate(llm=llm, use_critic=False)
    planner = TaskPlanner(llm, broker=broker, available_agents=registry.names())
    context_manager = ContextManager(vfs=VirtualFileSystem())

    from harness.config import settings

    return build_plan_execute_graph(
        llm,
        broker,
        planner=planner,
        store=store,
        registry=registry,
        gate=gate,
        max_replans=2,
        tool_mode=settings.runtime.agent_tool_mode,
        context_manager=context_manager,
    )


# ----------------------------------------------------------------------
# VFS 快照 / 审计读取
# ----------------------------------------------------------------------
def _snapshot_files() -> Dict[str, float]:
    """记录 workspace + reports 下文件及其 mtime。

    用 mtime 而非仅文件名集合：固定文件名的产物（cleaned 数据集、图表）在
    重复运行时会被覆写，文件名不变、mtime 变化，需据此识别为本次产物。
    """
    snap: Dict[str, float] = {}
    for base in (workspace_dir({}), reports_dir({})):
        if not os.path.isdir(base):
            continue
        for root, _, names in os.walk(base):
            for n in names:
                p = os.path.join(root, n)
                try:
                    snap[p] = os.path.getmtime(p)
                except OSError:
                    continue
    return snap


_DATASET_EXT = {".csv", ".parquet", ".xlsx", ".xls", ".json"}
_CHART_EXT = {".png", ".jpg", ".jpeg", ".svg"}
_REPORT_EXT = {".md", ".html", ".pdf"}


def _classify_artifact(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in _DATASET_EXT:
        return "dataset"
    if ext in _CHART_EXT:
        return "chart"
    if ext in _REPORT_EXT:
        return "report"
    return None


def _read_audit_for_session(session_id: str) -> List[Dict[str, Any]]:
    """从本地审计日志按 session_id 还原本次工具调用记录（按写入顺序）。"""
    local_file = get_audit_logger().local_file
    records: List[Dict[str, Any]] = []
    if not os.path.exists(local_file):
        return records
    with open(local_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("session_id") == session_id:
                records.append(rec)
    return records


# ----------------------------------------------------------------------
# 运行
# ----------------------------------------------------------------------
def run_once(case: Any, run_index: int, llm_factory: LLMFactory, llm_mode: str) -> RunTrace:
    """对单个用例完整运行一次，返回 RunTrace。"""
    llm = llm_factory()
    graph = build_graph(llm)

    session_id = f"{case.case_id}_r{run_index}_{uuid.uuid4().hex[:8]}"
    before = _snapshot_files()

    state = graph.invoke(
        make_plan_execute_state(case.goal, session_id=session_id),
        config={"recursion_limit": 120},
    )

    after = _snapshot_files()
    # 本次产物：新增文件，或 mtime 变化（被本次运行覆写）的文件
    new_files = sorted(
        p for p, mt in after.items()
        if p not in before or before[p] != mt
    )

    audit_records = _read_audit_for_session(session_id)

    # 工具轨迹
    tools_used: List[str] = []
    for rec in audit_records:
        name = rec.get("tool_name")
        if name and name not in tools_used:
            tools_used.append(name)
    denials = sum(1 for rec in audit_records if rec.get("pdp_decision") == "deny")
    sandbox_calls = sum(1 for rec in audit_records if rec.get("sandbox_used"))

    # 产物归类
    kinds: List[str] = []
    for path in new_files:
        kind = _classify_artifact(path)
        if kind and kind not in kinds:
            kinds.append(kind)
    if state.get("final_answer") and "report" not in kinds:
        kinds.append("report")

    # 子 Agent 结果
    agents: List[Dict[str, Any]] = []
    for r in state.get("sub_results", []) or []:
        agents.append({
            "name": r.sub_agent_name,
            "success": bool(r.success),
            "steps": int(r.steps_taken),
            "duration_ms": int(r.duration_ms),
            "error": r.error,
        })

    total_steps = sum(a["steps"] for a in agents)
    total_duration = sum(a["duration_ms"] for a in agents)

    usage_total = getattr(llm, "usage_total", None)

    return RunTrace(
        case_id=case.case_id,
        run_index=run_index,
        session_id=session_id,
        llm_mode=llm_mode,
        status=str(state.get("status", "unknown")),
        final_answer=str(state.get("final_answer") or ""),
        error=state.get("error"),
        agents=agents,
        tools_used=tools_used,
        tool_calls=len(audit_records),
        denials=denials,
        sandbox_calls=sandbox_calls,
        total_steps=total_steps,
        total_duration_ms=total_duration,
        artifact_kinds=kinds,
        new_files=[os.path.basename(p) for p in new_files],
        token_usage=dict(usage_total) if usage_total else None,
    )


def run_case(case: Any, runs: int, llm_factory: LLMFactory, llm_mode: str) -> List[RunTrace]:
    """对单个用例独立运行 runs 次（每次新建图与 LLM，互不共享状态）。"""
    return [run_once(case, i, llm_factory, llm_mode) for i in range(runs)]
