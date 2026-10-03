"""harness.approval_policy —— 审批策略：一次工具调用该自动放行、问人、还是直接拒。

分工：**领域包提供风险策略**（只回答"多危险"），**框架提供机制**（阈值判定 + 三档分流 +
兜底断言 + 留痕）。框架侧不出现任何领域名词。

决策见 `docs/技术选型决策.md` D-006；设计见 `docs/审批策略设计.md`。
"""

from __future__ import annotations

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
#: 判定不出来的一律按需审批（未知不放过）
RISK_UNKNOWN = "unknown"

__all__ = ["RISK_LOW", "RISK_MEDIUM", "RISK_HIGH", "RISK_UNKNOWN"]
