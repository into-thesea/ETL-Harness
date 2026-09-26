"""harness.planning.data_quality —— 数据质量校验节点（确定性红线）。

计划 §0.2 新增能力：**缺失率异常或数据质量越线则暂停分析**，而不是让脏数据一路
流到结论。被质量门调用后返回"暂停原因"，由门转成 ``GateDecision.HUMAN`` ——
即复用既有的 interrupt → 人工审批通道（该中断已可跨进程重启恢复）。

红线必须是**确定性**的：只读体检工具产出的结构化画像
（``tools/data_inspector.py`` 的 ``inspection``），不做 LLM 判断、不猜。
阈值一律来自配置（D17），见 ``QualitySettings``。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from harness.models import SubAgentResult, TaskStep

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """按阈值检查体检画像，越线即返回暂停原因（None 表示通过）。"""

    def __init__(self, config: Any = None) -> None:
        if config is None:
            from harness.config import settings

            config = settings.quality
        self.enabled = bool(getattr(config, "data_check_enabled", True))
        self.missing_rate_max = float(getattr(config, "missing_rate_max", 0.3))
        self.duplicate_rate_max = float(getattr(config, "duplicate_rate_max", 0.3))

    def check(self, task: TaskStep, result: SubAgentResult) -> Optional[str]:
        """返回暂停原因；``None`` 表示无需人工介入。

        阈值语义：**缺失率落在 (阈值, 1.0) 之间**才暂停 —— 全空列见
        :meth:`_is_cleaning_target`。

        ``task`` 参与签名以便将来按任务类型分档阈值（现未使用，保持与
        ``ArtifactChecker`` 一致的调用形状）。
        """
        if not self.enabled:
            return None

        inspection = (result.artifacts or {}).get("inspection")
        if not isinstance(inspection, dict):
            # 本子任务没产出体检画像 —— 与"数据质量"无关，不拦。
            # 不能因为某个工具没跑就把整条链卡住。
            return None

        problems = self._column_problems(inspection)
        duplicate = (inspection.get("quality") or {}).get("duplicate_rate")
        if isinstance(duplicate, (int, float)) and duplicate > self.duplicate_rate_max:
            problems.append(
                f"重复率 {duplicate} 超过阈值 {self.duplicate_rate_max}"
            )

        if not problems:
            return None
        reason = "数据质量红线触发，暂停分析待人工确认：" + "；".join(problems)
        logger.warning("Data quality pause: %s", reason)
        return reason

    def _column_problems(self, inspection: dict) -> list[str]:
        problems: list[str] = []
        for column in inspection.get("schema") or []:
            if not isinstance(column, dict):
                continue
            rate = column.get("missing_rate")
            if not isinstance(rate, (int, float)) or rate <= self.missing_rate_max:
                continue
            if self._is_cleaning_target(column, rate):
                continue
            problems.append(
                f"列 {column.get('name')} 缺失率 {rate} 超过阈值 {self.missing_rate_max}"
            )
        return problems

    @staticmethod
    def _is_cleaning_target(column: dict, rate: float) -> bool:
        """全空列（零非空值）是**清洗目标**，不是分析红线。

        现实里的表格常有整列留空，交给 cleaner 删除即可；体检器也已把它单独标成
        ``all_null`` 高风险。若在这里拦，真实数据几乎永远进不了分析 ——
        这条边界是实测撞出来的（示例脏数据里的 ``blank_note`` 整列为空，
        一旦按"缺失率 100% > 30%"拦截，整条链路第一步就停）。
        """
        non_null = column.get("non_null")
        if isinstance(non_null, int) and non_null == 0:
            return True
        return rate >= 1.0


__all__ = ["DataQualityChecker"]
