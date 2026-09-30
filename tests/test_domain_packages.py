"""tests.test_domain_packages —— 领域包机制（挂载 / 卸载 / 回收 / 事务化）。

用**测试内定义的假包**，不依赖真领域包（真包在切片 4 才搬进来）。这里钉死四条规矩：

1. 注册即撤销 —— 卸载逆序、且必须**收干净**（五处副作用逐处断言）；
2. 激活事务化 —— ``apply`` 中途抛错要回滚已完成的注册，不留半个活包；
3. 已 dispose 的上下文上再注册 → **报错**，不是静默成功；
4. 依赖声明错误 vs 依赖未就绪，是两种结果（FAILED / PENDING），不能混为一谈。

五处副作用：工具注册 / 角色注册 / 技能条目 / 工具结果缓存 / 子图编译缓存。
漏任何一处就是"假卸载"（禁用了还能被调用、重挂载后命中旧结果）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.agents.registry import AgentRegistry
from harness.cache import ToolResultCache
from harness.domain import (
    FrameworkHandles,
    PackageContext,
    PackageDisposedError,
    PackageError,
    PackageManager,
    PackageState,
)
from harness.models import SubAgentDef, ToolDef
from harness.orchestrator import SubgraphCache
from harness.skills.loader import SkillRegistry
from harness.tool_broker import ToolBroker


# ======================================================================
# 脚手架
# ======================================================================
def _tool(name: str) -> tuple[ToolDef, object]:
    def handler(args: dict, context: dict) -> tuple[bool, str, dict]:
        return True, f"{name} 的结果", {}

    return ToolDef(name=name, description=f"{name} 说明", parameters={}), handler


def _agent(name: str) -> SubAgentDef:
    return SubAgentDef(
        name=name,
        description=f"{name} 说明",
        system_prompt=f"你是 {name}",
        tools=["t1"],
        required_role="analyst",
        max_steps=7,
        timeout_seconds=99,
    )


def _skill_dir(tmp_path: Path, skill_name: str) -> str:
    directory = tmp_path / skill_name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: 用例技能\n---\n\n正文\n", encoding="utf-8"
    )
    return str(directory)


class FakePackage:
    """最小可用领域包：注册两个工具、一个角色、一个技能目录，并登记一个自定义副作用。"""

    name = "fake"
    version = "1.0.0"
    description = "用例假包"
    provider = "tests"
    requires: list[str] = []
    contributes = {"tools": ["t1", "t2"], "agents": ["fake-agent"]}

    def __init__(self, tmp_path: Path, *, fail_after_tools: bool = False) -> None:
        self.tmp_path = tmp_path
        self.fail_after_tools = fail_after_tools
        self.teardowns: list[str] = []

    def apply(self, ctx: PackageContext) -> None:
        ctx.register_tools([_tool("t1"), _tool("t2")])
        if self.fail_after_tools:
            raise RuntimeError("故意在注册一半时炸掉")
        ctx.register_agents([_agent("fake-agent")])
        ctx.add_skill_directory(_skill_dir(self.tmp_path, "fake-skill"))
        ctx.effect(lambda: (self.teardowns.append("effect-1"), None)[1])


@pytest.fixture
def framework(tmp_path):
    """一套真的框架侧对象（工具 / 角色 / 技能 / 两处缓存）。"""
    broker = ToolBroker(sandbox_executor=False, circuit_breaker=False, cache=False)
    handles = FrameworkHandles(
        tools=broker,
        agents=AgentRegistry(defs={}),
        skills=SkillRegistry(),
        result_cache=ToolResultCache(max_entries=8),
        subgraph_cache=SubgraphCache(),
    )
    return handles


@pytest.fixture
def manager(framework):
    return PackageManager(framework)


def _seed_caches(framework) -> None:
    """让两处缓存里各有与假包工具/角色相关的条目（模拟"挂载期间被用过"）。"""
    framework.result_cache.put("k1", "t1", (True, "旧结果", {}))
    framework.result_cache.put("k2", "other", (True, "别人的", {}))
    framework.subgraph_cache.set("fake-agent", object())
    framework.subgraph_cache.set("other-agent", object())


# ======================================================================
# 1. 挂载与卸载：五处副作用都要收干净
# ======================================================================
class TestMountUnmount:
    def test_mount_registers_everything(self, manager, framework, tmp_path) -> None:
        manager.register_package(FakePackage(tmp_path))
        info = manager.mount("fake")

        assert info.state is PackageState.ACTIVE
        assert {t.name for t in framework.tools.list_tools()} == {"t1", "t2"}
        assert framework.agents.names() == ["fake-agent"]
        assert framework.skills.names() == ["fake-skill"]

    def test_unmount_clears_all_five_side_effects(self, manager, framework, tmp_path) -> None:
        package = FakePackage(tmp_path)
        manager.register_package(package)
        manager.mount("fake")
        _seed_caches(framework)
        assert len(framework.result_cache) == 2

        manager.unmount("fake")

        # ① 工具 ② 角色 ③ 技能
        assert framework.tools.list_tools() == []
        assert framework.agents.names() == []
        assert framework.skills.names() == []
        # ④ 工具结果缓存：只清本包工具的条目，别人的留着
        assert len(framework.result_cache) == 1
        assert framework.result_cache.get("k2") is not None
        assert framework.result_cache.get("k1") is None
        # ⑤ 子图编译缓存：同理
        assert framework.subgraph_cache.names() == ["other-agent"]
        # 自定义副作用也被撤销
        assert package.teardowns == ["effect-1"]
        # 状态收敛
        assert manager.get_info("fake").state is PackageState.DISPOSED

    def test_unmount_is_idempotent(self, manager, tmp_path) -> None:
        manager.register_package(FakePackage(tmp_path))
        manager.mount("fake")

        manager.unmount("fake")
        manager.unmount("fake")

        assert manager.get_info("fake").state is PackageState.DISPOSED

    def test_mount_is_idempotent(self, manager, framework, tmp_path) -> None:
        manager.register_package(FakePackage(tmp_path))
        manager.mount("fake")
        manager.mount("fake")

        assert [t.name for t in framework.tools.list_tools()] == ["t1", "t2"]

    def test_remount_after_unmount_works(self, manager, framework, tmp_path) -> None:
        manager.register_package(FakePackage(tmp_path))
        manager.mount("fake")
        manager.unmount("fake")
        info = manager.mount("fake")

        assert info.state is PackageState.ACTIVE
        assert {t.name for t in framework.tools.list_tools()} == {"t1", "t2"}

    def test_unknown_package_raises(self, manager) -> None:
        with pytest.raises(KeyError):
            manager.mount("nope")
        with pytest.raises(KeyError):
            manager.unmount("nope")


# ======================================================================
# 2. 激活事务化
# ======================================================================
class TestTransactionalActivation:
    def test_failed_apply_rolls_back_registrations(self, manager, framework, tmp_path) -> None:
        manager.register_package(FakePackage(tmp_path, fail_after_tools=True))

        info = manager.mount("fake")

        assert info.state is PackageState.FAILED
        assert "已回滚" in info.error
        # 已注册的两个工具必须被撤销 —— 否则就是"半个活着的包"
        assert framework.tools.list_tools() == []
        assert framework.agents.names() == []

    def test_failed_apply_does_not_touch_other_packages(self, manager, framework, tmp_path) -> None:
        """回滚只能撤自己：另一包已挂载的工具不得受影响。"""
        other = FrameworkHandles(
            tools=framework.tools, agents=framework.agents, skills=framework.skills
        )
        other_manager = PackageManager(other)
        other_manager.register_package(FakePackage(tmp_path))
        other_manager.mount("fake")

        manager.register_package(FakePackage(tmp_path, fail_after_tools=True))
        manager.mount("fake")

        assert {t.name for t in framework.tools.list_tools()} == {"t1", "t2"}


# ======================================================================
# 3. 已卸载的上下文上注册 → 报错
# ======================================================================
class TestDisposedContext:
    def test_register_after_dispose_raises(self, framework) -> None:
        ctx = PackageContext("fake", framework)
        ctx.register_tools([_tool("t1")])
        ctx.dispose()

        assert ctx.disposed is True
        with pytest.raises(PackageDisposedError):
            ctx.register_tools([_tool("t3")])

    def test_every_registration_kind_is_guarded(self, framework, tmp_path) -> None:
        ctx = PackageContext("fake", framework)
        ctx.dispose()

        with pytest.raises(PackageDisposedError):
            ctx.register_agents([_agent("a")])
        with pytest.raises(PackageDisposedError):
            ctx.add_skill_directory(_skill_dir(tmp_path, "s"))
        with pytest.raises(PackageDisposedError):
            ctx.effect(lambda: None)

    def test_dispose_is_idempotent(self, framework) -> None:
        ctx = PackageContext("fake", framework)
        ctx.register_tools([_tool("t1")])
        ctx.dispose()
        ctx.dispose()  # 不抛、不重复撤销

        assert framework.tools.list_tools() == []


# ======================================================================
# 4. 逆序撤销
# ======================================================================
class TestReverseOrder:
    def test_disposers_run_in_reverse_registration_order(self, framework) -> None:
        order: list[str] = []
        ctx = PackageContext("fake", framework)
        ctx.effect(lambda: (order.append("A"), lambda: order.append("A⁻¹"))[1])
        ctx.effect(lambda: (order.append("B"), lambda: order.append("B⁻¹"))[1])

        ctx.dispose()

        assert order == ["A", "B", "B⁻¹", "A⁻¹"]

    def test_one_failing_disposer_does_not_stop_the_rest(self, framework) -> None:
        """撤销动作失败必须响亮但不能中断其余回收 —— 否则残留无人知晓。"""
        reached: list[str] = []
        ctx = PackageContext("fake", framework)

        def _boom() -> None:
            raise RuntimeError("撤销时炸了")

        ctx.effect(lambda: _boom)
        ctx.effect(lambda: (None, lambda: reached.append("ok"))[1])

        ctx.dispose()  # 不抛

        assert reached == ["ok"]


# ======================================================================
# 5. 依赖：声明错误 vs 未就绪
# ======================================================================
class TestRequires:
    def test_missing_service_keeps_package_pending(self, framework, tmp_path) -> None:
        """框架没提供该服务（如未配技能系统）→ PENDING，不 apply。"""
        framework.skills = None
        package = FakePackage(tmp_path)
        package.requires = ["skills"]
        manager = PackageManager(framework)
        manager.register_package(package)

        info = manager.mount("fake")

        assert info.state is PackageState.PENDING
        assert "未就绪" in info.error
        assert framework.tools.list_tools() == []  # 没执行 apply

    def test_unknown_service_name_fails_loudly(self, framework, tmp_path) -> None:
        """拼错服务名是**声明错误**，必须当场失败 —— 否则一个永远起不来的包
        看起来只是"暂时没就绪"。"""
        package = FakePackage(tmp_path)
        package.requires = ["toolz"]  # 拼错
        manager = PackageManager(framework)
        manager.register_package(package)

        info = manager.mount("fake")

        assert info.state is PackageState.FAILED
        assert "未知服务" in info.error

    def test_resolve_requires_directly(self, framework) -> None:
        ctx = PackageContext("fake", framework)

        assert ctx.resolve_requires(["tools", "agents"]) == []
        framework.skills = None
        assert ctx.resolve_requires(["skills"]) == ["skills"]
        with pytest.raises(PackageError):
            ctx.resolve_requires(["nope"])


# ======================================================================
# 6. 发现：坏包不能拖垮其余
# ======================================================================
class TestDiscovery:
    def test_broken_entry_point_is_skipped(self, manager, tmp_path) -> None:
        class _BadEntry:
            name = "broken"
            value = "no.such.module:Package"

            def load(self):
                raise ImportError("模块不存在")

        class _GoodEntry:
            name = "fake"
            value = "tests:fake"

            def __init__(self, package):
                self._package = package

            def load(self):
                return self._package

        infos = manager.discover([_BadEntry(), _GoodEntry(FakePackage(tmp_path))])

        by_name = {i.name: i for i in infos}
        assert by_name["broken"].state is PackageState.FAILED
        assert "导入失败" in by_name["broken"].error
        assert by_name["fake"].state is PackageState.PENDING  # 已登记，尚未挂载

    def test_discover_does_not_apply(self, manager, framework, tmp_path) -> None:
        """发现阶段不得执行副作用 —— 注册只发生在 mount。"""

        class _Entry:
            name = "fake"

            def load(self):
                return FakePackage(tmp_path)

        manager.discover([_Entry()])

        assert framework.tools.list_tools() == []


# ======================================================================
# 6b. 真领域包走一遍：挂载 → 卸载
# ======================================================================
class TestRealDomainPackage:
    """用**真的**数据分析领域包（不是假包）验一遍挂载与卸载。

    这一条曾抓到真问题：entry point 指到的是**类**，而 mount 直接调 `apply(ctx)`
    —— 等价于未绑定调用，`ctx` 被当成 `self`，包挂在 FAILED 上，于是框架"装配成功"
    却一个工具、一个角色都没有（服务照常起来，只是什么也做不了）。
    """

    def _manager(self, framework):
        from packages.data_analysis import DataAnalysisPackage

        manager = PackageManager(framework)
        manager.register_package(DataAnalysisPackage)
        return manager

    def test_mount_then_unmount_leaves_nothing_behind(self, framework) -> None:
        manager = self._manager(framework)

        info = manager.mount("data_analysis")
        assert info.state is PackageState.ACTIVE, info.error
        assert len(framework.tools.list_tools()) == 7
        assert sorted(framework.agents.names()) == ["analyst", "data-explorer", "reporter"]
        assert len(framework.skills.names()) >= 6          # 领域技能库
        # 角色预算字段没在搬迁中丢（这是搬迁最容易退化的地方）
        analyst = framework.agents.get("analyst")
        assert (analyst.max_steps, analyst.timeout_seconds) == (14, 300)
        assert "code_executor" in analyst.tools

        manager.unmount("data_analysis")
        assert framework.tools.list_tools() == []
        assert framework.agents.names() == []
        assert framework.skills.names() == []

    def test_class_entry_point_is_instantiated(self, framework) -> None:
        """entry point 指到类时要自动实例化（作者最自然的写法）。"""
        from packages.data_analysis import DataAnalysisPackage

        manager = PackageManager(framework)
        manager.register_package(DataAnalysisPackage)

        assert manager.get_info("data_analysis").name == "data_analysis"
        assert manager.mount("data_analysis").state is PackageState.ACTIVE


# ======================================================================
# 7. 前置：空注册表必须真的是空的
# ======================================================================
class TestEmptyRegistry:
    def test_framework_ships_no_roles_at_all(self) -> None:
        """框架**不内置任何角色**：不传是空，传空也是空。

        这一条是领域包机制的**前置**，也曾经是个真缺陷：原实现是
        `defs or build_default_agents()` —— 把显式的空字典当成"没传"，于是"清空注册表"
        反而长出三个领域角色。角色现在由领域包挂载进来（见 TestRealDomainPackage）。
        """
        assert AgentRegistry().names() == []
        assert AgentRegistry(defs={}).names() == []
        assert AgentRegistry(defs=None).names() == []

    def test_registry_does_not_alias_caller_dict(self) -> None:
        given: dict = {}
        registry = AgentRegistry(defs=given)
        registry.register(_agent("a"))

        assert given == {}, "注册表不该改调用方传进来的字典"


# ======================================================================
# 8. 清单
# ======================================================================
class TestListing:
    def test_info_carries_contributions_and_state(self, manager, tmp_path) -> None:
        manager.register_package(FakePackage(tmp_path))
        info = manager.mount("fake")

        assert info.name == "fake"
        assert info.version == "1.0.0"
        assert info.provider == "tests"
        assert info.contributes["agents"] == ["fake-agent"]
        assert "工具 2" in info.contributes_summary

    def test_summary_counts_by_state(self, manager, tmp_path) -> None:
        manager.register_package(FakePackage(tmp_path))
        manager.mount("fake")

        summary = manager.summary()

        assert summary["total"] == 1
        assert summary["by_state"] == {"active": 1}
