"""tests.test_long_term_memory —— 长期记忆：向量后端协议 + 记忆策略。

分三层测，各自的关注点不同：

- **VectorStore 协议**：纯存取语义（相似度排序、主体隔离、阈值过滤、同 id 覆盖、
  容量淘汰、跨实例持久化），用 :class:`LocalVectorStore` 离线验证。
- **记忆策略**：写什么、写多少、留多久，用假的 Embedding 客户端注入确定性向量，
  离线验证，不打真实 API。
- **PgVectorStore**：对真实 PostgreSQL + pgvector 的验证标 ``needs_db``，
  只有库起着时才跑（``--require-db`` 可要求严格）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from harness.config import settings
from harness.memory.long_term import EXPERIENCE, LongTermMemory
from harness.memory.vector_store import LocalVectorStore, PgVectorStore, VectorRecord

DIM = settings.embedding.dim
PG_DSN = "postgresql://harness_admin:harness_admin_pw@127.0.0.1:55432/harness"


def _vec(*components: float) -> list[float]:
    """补齐到配置维度。余弦只看方向，前几维足够表达测试意图。"""
    return list(components) + [0.0] * max(DIM - len(components), 0)


class _FakeEmbeddingClient:
    """按文本内容解析向量的假 Embedding。

    真实 Embedding 的向量取决于远端模型，无法精确断言排序与阈值；这里让测试用
    一个**解析函数**决定每段文本对应哪个方向，从而能对"相似度排序""阈值过滤"
    下确定断言。注意解析函数的入参是**实际被嵌入的文本**（写入时是拼接好的
    content，检索时是查询串），两者要按同一规则映射才有一致的相似度。
    """

    def __init__(self, resolve=None, fail: bool = False):
        self.resolve = resolve or (lambda text: (0.0, 0.0, 1.0))
        self.fail = fail
        self.calls: list[str] = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model: str, input: str):  # noqa: A002 - 对齐 SDK 形参名
        self.calls.append(input)
        if self.fail:
            raise RuntimeError("embedding 服务不可用")
        return SimpleNamespace(data=[SimpleNamespace(embedding=_vec(*self.resolve(input)))])


def _record(rid: str, vector: tuple[float, ...], *, agent: str = "t1", ts: float = 0.0) -> VectorRecord:
    """构造一条 1536 维（配置维度）的记录，供本地后端用。"""
    return VectorRecord(
        id=rid, content=f"记忆 {rid}", vector=_vec(*vector),
        metadata={"agent_id": agent, "type": EXPERIENCE}, timestamp=ts,
    )


def _small_record(rid: str, vector: tuple[float, ...], *, agent: str = "t1", ts: float = 0.0) -> VectorRecord:
    """三维记录，供 pgvector 集成用例（小维度建表更快）。"""
    return VectorRecord(
        id=rid, content=f"记忆 {rid}", vector=list(vector),
        metadata={"agent_id": agent, "type": EXPERIENCE}, timestamp=ts,
    )


def _by_keyword(needle: str) -> object:
    """含关键词的文本映射到 (1,0,0)，其余映射到 (0,1,0)。"""
    return lambda text: (1.0, 0.0, 0.0) if needle in text else (0.0, 1.0, 0.0)


@pytest.fixture
def store(tmp_path) -> LocalVectorStore:
    return LocalVectorStore(str(tmp_path / "vectors.json"))


@pytest.fixture
def ltm(tmp_path) -> LongTermMemory:
    return LongTermMemory(
        agent_id="t1",
        store=LocalVectorStore(str(tmp_path / "ltm.json")),
        embedding_client=_FakeEmbeddingClient(),
    )


# ======================================================================
# VectorStore 协议
# ======================================================================
class TestVectorStoreProtocol:
    def test_search_orders_by_similarity(self, store: LocalVectorStore) -> None:
        store.add([_record("a", (1.0, 0.0, 0.0)), _record("b", (0.0, 1.0, 0.0))])
        hits = store.search(_vec(1.0, 0.1, 0.0), 2, agent_id="t1")
        assert [h.id for h in hits] == ["a", "b"], [(h.id, h.score) for h in hits]
        assert hits[0].score > hits[1].score

    def test_search_is_isolated_by_agent(self, store: LocalVectorStore) -> None:
        """多主体共存时检索不串味 —— 这是多租户部署的底线。"""
        store.add([_record("mine", (1.0, 0.0, 0.0), agent="t1"),
                   _record("theirs", (1.0, 0.0, 0.0), agent="t2")])
        hits = store.search(_vec(1.0, 0.0, 0.0), 10, agent_id="t1")
        assert [h.id for h in hits] == ["mine"]
        assert store.count(agent_id="t1") == 1
        assert store.count(agent_id="t2") == 1
        assert store.count() == 2

    def test_min_score_filters(self, store: LocalVectorStore) -> None:
        store.add([_record("a", (1.0, 0.0, 0.0)), _record("b", (0.0, 1.0, 0.0))])
        hits = store.search(_vec(1.0, 0.0, 0.0), 10, agent_id="t1", min_score=0.9)
        assert [h.id for h in hits] == ["a"]

    def test_same_id_overwrites(self, store: LocalVectorStore) -> None:
        """内容指纹做 id：同一段内容重复写入只覆盖，不堆积重复项。"""
        store.add([_record("a", (1.0, 0.0, 0.0))])
        store.add([_record("a", (1.0, 0.0, 0.0))])
        assert store.count() == 1

    def test_delete_clear_count(self, store: LocalVectorStore) -> None:
        store.add([_record("a", (1.0, 0.0, 0.0)), _record("b", (0.0, 1.0, 0.0))])
        assert store.delete(["a", "missing"]) == 1
        assert store.count() == 1
        store.clear(agent_id="t1")
        assert store.count() == 0

    def test_prune_keeps_newest(self, store: LocalVectorStore) -> None:
        store.add([
            _record("old", (1.0, 0.0, 0.0), ts=1.0),
            _record("mid", (1.0, 0.0, 0.0), ts=2.0),
            _record("new", (1.0, 0.0, 0.0), ts=3.0),
        ])
        assert store.prune(agent_id="t1", keep=2) == 1
        remaining = {h.id for h in store.search(_vec(1.0, 0.0, 0.0), 10, agent_id="t1")}
        assert remaining == {"mid", "new"}

    def test_persists_across_instances(self, tmp_path) -> None:
        path = str(tmp_path / "v.json")
        first = LocalVectorStore(path)
        first.add([_record("a", (1.0, 0.0, 0.0))])
        assert LocalVectorStore(path).count() == 1


# ======================================================================
# 记忆策略
# ======================================================================
class TestMemoryPolicy:
    def test_remember_experience_writes_content_and_metadata(self, ltm: LongTermMemory) -> None:
        mid = ltm.remember_experience(
            goal="分析销售趋势", final_answer="Q3 环比上升 12%", session_id="s1",
        )
        assert mid
        assert ltm.count() == 1
        hits = ltm.search("分析销售趋势")
        assert hits and "Q3 环比上升 12%" in hits[0].content
        assert hits[0].metadata["type"] == EXPERIENCE
        assert hits[0].metadata["session_id"] == "s1"

    def test_empty_input_not_written(self, ltm: LongTermMemory) -> None:
        assert ltm.remember_experience(goal="", final_answer="有结论") is None
        assert ltm.remember_experience(goal="有目标", final_answer="") is None
        assert ltm.count() == 0

    def test_same_experience_not_duplicated(self, ltm: LongTermMemory) -> None:
        for _ in range(3):
            ltm.remember_experience(goal="分析销售趋势", final_answer="Q3 上升")
        assert ltm.count() == 1

    def test_overlong_content_truncated(self, ltm: LongTermMemory) -> None:
        """超长结论按上限截断成摘要，且拼出来的整条不超过上限。"""
        ltm.max_content_chars = 300
        ltm.remember_experience(goal="目标", final_answer="长" * 5000)
        content = ltm.get_all()[0].content
        assert len(content) <= 300
        assert content.endswith("…(摘要截断)")
        assert content.startswith("目标：目标")

    def test_capacity_evicts_oldest(self, ltm: LongTermMemory) -> None:
        """容量上界必须生效：没有淘汰策略的向量库会无限增长。"""
        ltm.max_items = 2
        for i in range(5):
            ltm.remember_experience(goal=f"第 {i} 次分析任务", final_answer=f"结论 {i}")
        assert ltm.count() == 2

    def test_short_query_is_not_searched(self, ltm: LongTermMemory) -> None:
        """过短的查询直接跳过，避免"分析"这类泛词命中一堆无关记忆。"""
        ltm.remember_experience(goal="分析销售数据的季度趋势", final_answer="上升")
        assert ltm.recall_for_goal("分析") == []
        assert ltm.recall_for_goal("分析销售数据的季度趋势")

    def test_recall_filters_by_similarity_threshold(self, tmp_path) -> None:
        ltm = LongTermMemory(
            agent_id="t1",
            store=LocalVectorStore(str(tmp_path / "v.json")),
            embedding_client=_FakeEmbeddingClient(_by_keyword("销售数据")),
        )
        ltm.remember_experience(goal="第一次分析销售数据", final_answer="结论")
        assert ltm.recall_for_goal("再分析一次销售数据")           # 同方向 → 相似度 1.0
        assert ltm.recall_for_goal("完全无关的查询内容在这里") == []  # 正交 → 0.0 < 阈值

    def test_embedding_failure_degrades_without_raising(self, tmp_path) -> None:
        """Embedding 挂掉时读写都退化为空操作，绝不把异常抛给任务。"""
        ltm = LongTermMemory(
            agent_id="t1",
            store=LocalVectorStore(str(tmp_path / "v.json")),
            embedding_client=_FakeEmbeddingClient(fail=True),
        )
        assert ltm.remember_experience(goal="分析数据", final_answer="结论") is None
        assert ltm.search("分析数据") == []
        assert ltm.build_context_text("分析数据的趋势") == ""
        assert ltm.is_ready() is False
        assert ltm.degraded_reason and "RuntimeError" in ltm.degraded_reason

    def test_build_context_text_renders_hits(self, tmp_path) -> None:
        ltm = LongTermMemory(
            agent_id="t1",
            store=LocalVectorStore(str(tmp_path / "v.json")),
            embedding_client=_FakeEmbeddingClient(_by_keyword("销售数据")),
        )
        ltm.remember_experience(goal="分析销售数据的趋势", final_answer="环比上升 12%")
        text = ltm.build_context_text("再分析一次销售数据")
        assert "长期记忆" in text and "环比上升 12%" in text

    def test_stats_reports_backend(self, ltm: LongTermMemory) -> None:
        ltm.remember_experience(goal="分析销售数据", final_answer="结论")
        stats = ltm.stats()
        assert stats["backend"] == "local"
        assert stats["items"] == 1
        assert stats["max_items"] == settings.memory.long_term_max_items
        assert stats["ready"] is True


# ======================================================================
# 接入运行链路（端到端）
# ======================================================================
class _WireLLM:
    """最小脚本 LLM：单步计划 → 执行体直接给结论 → 汇总给最终报告。"""

    PLAN = {
        "tasks": [{
            "title": "分析任务", "description": "对数据做分析",
            "assigned_to": "reporter", "depends_on": [],
            "acceptance_criteria": ["给出结论"], "expected_artifacts": [],
        }]
    }

    def chat_json(self, messages) -> dict:
        if "质量门裁判" in messages[0]["content"]:
            return {"passed": True, "reason": "ok", "needs_human": False}
        return self.PLAN

    def chat(self, messages, temperature=None) -> str:
        system = messages[0]["content"] if messages else ""
        if "报告汇总者" in system:
            return "最终报告：销售数据环比上升 12%"
        return json.dumps({"final_answer": "子任务结论：分析完成"}, ensure_ascii=False)


class TestWiringIntoOrchestration:
    """长期记忆接在规划的检索与收尾的写入上 —— 这两处断了，记忆就是死代码。"""

    def _graph(self, ltm: LongTermMemory):
        from harness.agents.registry import AgentRegistry
        from harness.orchestrator import build_plan_execute_graph
        from harness.planning import QualityGate, TaskPlanner
        from harness.tool_broker import ToolBroker

        llm = _WireLLM()
        return build_plan_execute_graph(
            llm, ToolBroker(),
            registry=AgentRegistry(),
            planner=TaskPlanner(llm, available_agents=["reporter"]),
            gate=QualityGate(llm=llm),
            long_term_memory=ltm,
        )

    def test_experience_written_then_recalled(self, tmp_path) -> None:
        from harness.orchestrator import make_plan_execute_state

        ltm = LongTermMemory(
            agent_id="default",
            store=LocalVectorStore(str(tmp_path / "v.json")),
            embedding_client=_FakeEmbeddingClient(_by_keyword("销售数据")),
        )
        graph = self._graph(ltm)

        # 第一次运行：库里还没有经验，检索为空；收尾后应沉淀 1 条
        first = graph.invoke(
            make_plan_execute_state("分析销售数据的季度趋势"),
            {"configurable": {"thread_id": "run-1"}},
        )
        assert first["status"] == "finished", first.get("error")
        assert ltm.count() == 1, "任务成功收尾后必须沉淀一条经验"

        # 第二次运行：规划前应检索到上次的经验，并带进 state 供子任务使用
        second = graph.invoke(
            make_plan_execute_state("再分析一次销售数据"),
            {"configurable": {"thread_id": "run-2"}},
        )
        context = second.get("long_term_context") or ""
        assert "长期记忆" in context, "规划前未检索长期记忆"
        assert "环比上升 12%" in context, f"检索到的经验没进上下文：{context!r}"

    def test_unavailable_memory_does_not_break_task(self, tmp_path) -> None:
        """记忆整个不可用时任务照常跑完 —— 它是增强项，不是必需项。"""
        from harness.orchestrator import make_plan_execute_state

        ltm = LongTermMemory(
            agent_id="default",
            store=LocalVectorStore(str(tmp_path / "v.json")),
            embedding_client=_FakeEmbeddingClient(fail=True),
        )
        graph = self._graph(ltm)
        out = graph.invoke(
            make_plan_execute_state("分析销售数据的季度趋势"),
            {"configurable": {"thread_id": "run-x"}},
        )
        assert out["status"] == "finished"
        assert out["final_answer"]
        assert ltm.count() == 0


# ======================================================================
# 真实 PostgreSQL + pgvector
# ======================================================================
@pytest.mark.needs_db
class TestPgVectorStoreIntegration:
    def test_crud_isolation_and_prune(self) -> None:
        store = PgVectorStore(PG_DSN, "governed_test_vectors", 3)
        assert store.is_available(), store.last_error

        store.clear()
        store.add([
            _small_record("a", (1.0, 0.0, 0.0), agent="t1", ts=1.0),
            _small_record("b", (0.0, 1.0, 0.0), agent="t1", ts=2.0),
            _small_record("c", (1.0, 0.0, 0.0), agent="t2", ts=3.0),
        ])
        assert store.count() == 3
        assert store.count(agent_id="t1") == 2

        hits = store.search([1.0, 0.0, 0.0], 3, agent_id="t1", min_score=0.0)
        assert [h.id for h in hits] == ["a", "b"]
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)
        assert all(h.id != "c" for h in hits), "检索必须按主体隔离"

        assert store.prune(agent_id="t1", keep=1) == 1
        assert store.count(agent_id="t1") == 1
        assert store.delete(["c"]) == 1
        assert store.count() == 1

        store.clear()
        assert store.count() == 0
        store.close()
