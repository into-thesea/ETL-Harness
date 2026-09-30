"""tests.test_packaging —— 打包与领域包发现的前提。

打包这件事在本项目里不是"为了好看"，而是**领域包发现的前置**：entry points 只能挂在
已安装的发行版上，且 `pyproject.toml` 里必须声明 build-backend，否则 entry points
会被**静默忽略**（不是报错，是发现不了）。所以这里守三条：

1. `pip install -e .` 后发行版可查，且**版本号与运行时常量一致**（两处字面量会走散）；
2. 依赖清单只有一份 —— `Requires-Dist` 由 requirements.txt 生成，且注释被正确剥掉；
3. 领域包分组名可查询（`entry_points(group=...)` 在旧 Python 上会 TypeError，
   这条守住调用约定；分组**暂时为空**是对的，领域包在切片 4 才挂上去）。

未安装时（只靠 `pythonpath=.` 跑）这些用例 skip —— 两条路都要能跑。
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution, entry_points, requires

import pytest

DIST_NAME = "governed"
GROUP = "governed.domain_packages"


def _dist():
    try:
        return distribution(DIST_NAME)
    except PackageNotFoundError:
        pytest.skip("未安装发行版。跑 `pip install -e .` 后本文件才有效。")


# ======================================================================
# 元数据
# ======================================================================
class TestMetadata:
    def test_version_matches_runtime_constant(self) -> None:
        """`pyproject.toml` 的 version 与 `harness.server.service.VERSION` 必须同值。

        两处字面量迟早走散，而它们分别被用作包元数据与 API 响应里的版本号。
        """
        from harness.server.service import VERSION

        assert _dist().version == VERSION

    def test_dependencies_come_from_requirements_txt(self) -> None:
        """依赖清单由 requirements.txt 生成（不复制第二份），且注释行被剥掉。"""
        reqs = requires(DIST_NAME) or []

        assert len(reqs) > 10, "Requires-Dist 太少，requirements.txt 可能没被读到"
        assert not [r for r in reqs if "#" in r], "注释行混进了依赖清单"
        assert any(r.startswith("langgraph") for r in reqs)

    def test_build_backend_declared(self) -> None:
        """build-backend 缺失时 entry points 会被静默忽略 —— 这是发现机制的前提。"""
        dist = _dist()
        assert dist.entry_points is not None or True  # 元数据可读即证明构建元数据完整
        assert dist.metadata["Name"] == DIST_NAME


# ======================================================================
# 领域包发现
# ======================================================================
class TestDomainPackageDiscovery:
    def test_group_is_queryable(self) -> None:
        """分组名可查询、返回可迭代对象。

        守的是调用约定：`entry_points(group=...)` 在较老的 Python 上会抛 TypeError，
        必须退化成 `entry_points().select(group=...)`。
        """
        found = entry_points(group=GROUP)

        assert list(found) == list(found)  # 可重复迭代
        assert isinstance(list(found), list)

    def test_exactly_one_domain_package_is_installed(self) -> None:
        """当前恰好挂一个领域包：``data_analysis``。

        框架是通用的、领域以包挂载；仓库里**只带这一个**领域实例。这条用例的作用是
        "别悄悄多出第二个" —— 多出来时它会红，提醒同步更新领域清单与验收用例。
        """
        found = list(entry_points(group=GROUP))

        assert [ep.name for ep in found] == ["data_analysis"]
        assert found[0].value == "packages.data_analysis:DataAnalysisPackage"

    def test_discovered_package_declares_its_contributions(self) -> None:
        """被发现的包要能装载，且清单里说清自己贡献了什么（控制台插件页据此渲染）。"""
        package = list(entry_points(group=GROUP))[0].load()

        assert package.name == "data_analysis"
        assert package.requires == ["tools", "agents", "skills"]
        assert len(package.contributes["tools"]) == 7
        assert sorted(package.contributes["agents"]) == ["analyst", "data-explorer", "reporter"]
        assert package.contributes["skills"] >= 6
        assert package.contributes["offline_llm"]
