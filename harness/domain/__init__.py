"""harness.domain —— 领域包机制（框架不认识任何领域名词）。

领域能力（工具 / 子 Agent / 技能）以**包**的形式挂载：包经 entry points 被发现，
经 :class:`PackageContext` 注册，卸载时由框架回收它造成的一切副作用。

完整设计与验收标准见 `docs/领域包与插件接口设计.md`；决策与「何时该回头」见
`docs/技术选型决策.md` D-005。
"""

from .manager import ENTRY_POINT_GROUP, PackageInfo, PackageManager
from .protocol import (
    DomainPackage,
    FrameworkHandles,
    PackageContext,
    PackageDisposedError,
    PackageError,
    PackageState,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "DomainPackage",
    "FrameworkHandles",
    "PackageContext",
    "PackageDisposedError",
    "PackageError",
    "PackageInfo",
    "PackageManager",
    "PackageState",
]
