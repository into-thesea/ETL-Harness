"""harness.domain.manager —— 领域包的发现、挂载、卸载与清单。

发现走 **entry points**（分组名 ``governed.domain_packages``）。它只能看到**已安装的
发行版**（``pip install -e .``），这是硬约束，不是选择 —— 见 `docs/技术选型决策.md` D-005。

三条从调研直接拿来的做法：

- 逐个 ``ep.load()`` **各自 try/except**，坏的跳过并告警 —— 一个坏插件不得让整个
  应用起不来；
- **加载期不执行副作用**：``ep.load()`` 只导入模块，真正注册发生在 ``mount`` 里的
  ``apply(ctx)``；
- **激活事务化**：``apply`` 抛错 → 回滚已完成的注册 → ``FAILED``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Optional

from .protocol import (
    DomainPackage,
    FrameworkHandles,
    PackageContext,
    PackageState,
)

logger = logging.getLogger(__name__)

# 领域包的分组名。名字须符合 entry points 规范的 `^\\w+(\\.\\w+)*$`。
ENTRY_POINT_GROUP = "governed.domain_packages"


@dataclass
class PackageInfo:
    """一个领域包对外的清单（控制台插件页直接渲染它）。"""

    name: str
    version: str = ""
    description: str = ""
    provider: str = ""
    requires: list[str] = field(default_factory=list)
    contributes: dict[str, Any] = field(default_factory=dict)
    state: PackageState = PackageState.PENDING
    error: str = ""
    status_note: str = ""
    """包自己如实说明哪部分还没接线（三档判据：已接入运行链路 / 代码存在但未接入 /
    只有描述）。装配状态由 :attr:`state` 表达，这一项说的是**包内**的诚实度。"""

    @property
    def contributes_summary(self) -> str:
        """一行摘要，供界面显示（如"工具 7 · 子 Agent 3 · 技能 12"）。"""
        parts = []
        for key, label in (("tools", "工具"), ("agents", "子 Agent"), ("skills", "技能")):
            value = self.contributes.get(key)
            if isinstance(value, int):
                parts.append(f"{label} {value}")
            elif isinstance(value, (list, tuple)) and value:
                parts.append(f"{label} {len(value)}")
        return " · ".join(parts) if parts else "（未声明贡献）"


class PackageManager:
    """领域包的注册中心。

    Args:
        handles: 框架侧句柄（工具 / 角色 / 技能 / 两处缓存）。
    """

    def __init__(self, handles: FrameworkHandles) -> None:
        self.handles = handles
        self._packages: dict[str, Any] = {}          # name -> 包对象
        self._infos: dict[str, PackageInfo] = {}     # name -> 清单
        self._contexts: dict[str, PackageContext] = {}

    # ------------------------------------------------------------------
    # 发现
    # ------------------------------------------------------------------
    def discover(self, entries: Optional[list[EntryPoint]] = None) -> list[PackageInfo]:
        """从 entry points 载入领域包清单（**不执行 apply**）。

        Args:
            entries: 显式给定的 entry points（测试用）；None 时从 ``ENTRY_POINT_GROUP``
                发现。
        """
        found = list(entries) if entries is not None else list(entry_points(group=ENTRY_POINT_GROUP))
        for ep in found:
            try:
                package = ep.load()
            except Exception as e:  # noqa: BLE001 - 一个坏包不得影响其余
                logger.exception("领域包 %s 导入失败，跳过", ep.name)
                self._infos[ep.name] = PackageInfo(
                    name=ep.name,
                    state=PackageState.FAILED,
                    error=f"导入失败：{type(e).__name__}: {e}",
                )
                continue
            self.register_package(package)
        return self.list_packages()

    def register_package(self, package: Any) -> PackageInfo:
        """登记一个领域包对象（不走 entry points 的路径，如内置包或测试）。

        entry point 指到**类**是常见写法（`pkg.module:DataAnalysisPackage` 读起来就是
        "这个包"），而 `mount` 要的是实例 —— 这里统一实例化，免得作者踩到
        `apply() missing 1 required positional argument`（`ctx` 被当成了 `self`）。
        """
        if isinstance(package, type):
            package = package()

        name = str(getattr(package, "name", "") or "")
        if not name:
            raise ValueError("领域包必须有 name")
        self._packages[name] = package
        info = PackageInfo(
            name=name,
            version=str(getattr(package, "version", "") or ""),
            description=str(getattr(package, "description", "") or ""),
            provider=str(getattr(package, "provider", "") or ""),
            requires=list(getattr(package, "requires", []) or []),
            contributes=dict(getattr(package, "contributes", {}) or {}),
            status_note=str(getattr(package, "status_note", "") or ""),
            state=PackageState.PENDING,
        )
        self._infos[name] = info
        return info

    # ------------------------------------------------------------------
    # 挂载 / 卸载
    # ------------------------------------------------------------------
    def mount(self, name: str) -> PackageInfo:
        """挂载一个领域包；失败时**回滚**已完成的注册。

        Raises:
            KeyError: 包未登记。
        """
        if name not in self._packages:
            raise KeyError(f"领域包 {name!r} 未登记")

        info = self._infos[name]
        if info.state is PackageState.ACTIVE:
            return info

        ctx = PackageContext(name, self.handles)
        try:
            missing = ctx.resolve_requires(info.requires)
        except Exception as e:  # 声明错误：当场失败，不装成"没就绪"
            info.state = PackageState.FAILED
            info.error = f"{type(e).__name__}: {e}"
            logger.error("领域包 %s 声明错误：%s", name, e)
            return info

        if missing:
            info.state = PackageState.PENDING
            info.error = f"依赖未就绪：{missing}"
            logger.info("领域包 %s 停在 PENDING，依赖未就绪：%s", name, missing)
            return info

        info.state = PackageState.LOADING
        info.error = ""
        try:
            self._packages[name].apply(ctx)
        except Exception as e:  # noqa: BLE001 - 激活必须事务化
            ctx.dispose()  # 回滚：已注册的工具/角色/技能全部撤销
            info.state = PackageState.FAILED
            info.error = f"apply 失败（已回滚）：{type(e).__name__}: {e}"
            logger.exception("领域包 %s 挂载失败，已回滚其注册", name)
            return info

        self._contexts[name] = ctx
        info.state = PackageState.ACTIVE
        logger.info(
            "领域包 %s 已挂载：%s", name, info.contributes_summary
        )
        return info

    def unmount(self, name: str) -> PackageInfo:
        """卸载：逆序撤销注册 + 清两处按名字归属的缓存。

        Raises:
            KeyError: 包未登记。
        """
        if name not in self._infos:
            raise KeyError(f"领域包 {name!r} 未登记")

        info = self._infos[name]
        ctx = self._contexts.pop(name, None)
        if ctx is None:
            # 从未挂载（或已卸载）：把状态收敛到 DISPOSED，不做无谓的撤销
            if info.state in (PackageState.PENDING, PackageState.FAILED):
                info.state = PackageState.DISPOSED
            return info

        info.state = PackageState.UNLOADING
        ctx.dispose()
        info.state = PackageState.DISPOSED
        logger.info("领域包 %s 已卸载", name)
        return info

    def mount_all(self) -> list[PackageInfo]:
        """按登记顺序挂载全部包（依赖未就绪的停在 PENDING，不算失败）。"""
        for name in list(self._packages):
            self.mount(name)
        return self.list_packages()

    # ------------------------------------------------------------------
    # 清单
    # ------------------------------------------------------------------
    def list_packages(self) -> list[PackageInfo]:
        """全部领域包清单（含未挂载与导入失败的），按名字排序。"""
        return sorted(self._infos.values(), key=lambda i: i.name)

    def get_info(self, name: str) -> Optional[PackageInfo]:
        return self._infos.get(name)

    def context(self, name: str) -> Optional[PackageContext]:
        """取挂载中的上下文（测试与卸载断言用）。"""
        return self._contexts.get(name)

    def summary(self) -> dict[str, Any]:
        """给界面用的一行汇总。"""
        states: dict[str, int] = {}
        for info in self._infos.values():
            states[info.state.value] = states.get(info.state.value, 0) + 1
        return {"total": len(self._infos), "by_state": states, "group": ENTRY_POINT_GROUP}


__all__ = ["ENTRY_POINT_GROUP", "PackageInfo", "PackageManager"]
