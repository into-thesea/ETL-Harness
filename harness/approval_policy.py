"""harness.approval_policy —— 审批策略：一次工具调用该自动放行、问人、还是直接拒。

分工：**领域包提供风险策略**（只回答"多危险"），**框架提供机制**（阈值判定 + 三档分流 +
兜底断言 + 留痕）。框架侧不出现任何领域名词。

决策见 `docs/技术选型决策.md` D-006；设计见 `docs/审批策略设计.md`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
#: 判定不出来的一律按需审批（未知不放过）
RISK_UNKNOWN = "unknown"

DECISION_AUTO = "auto"
DECISION_ASK = "ask"
DECISION_DENY = "deny"

#: 可比较的风险等级；``unknown`` 不在此表内 —— 它不可比较，一律按需审批
_RISK_ORDER = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2}


@dataclass(frozen=True)
class ApprovalDecision:
    """一次调用的审批判定结果（含可留痕的理由与依据）。"""

    decision: str
    reason: str
    risk: str
    policy: str                      # "default"（未声明策略）| "tool"（工具自己的策略）
    fallback: Optional[str] = None   # 命中的兜底机制名；None 表示没有


def decide_approval(
    tool_name: str,
    args: Optional[dict] = None,
    *,
    risk_policy: Optional[Callable[[dict], str]] = None,
    threshold: str = RISK_MEDIUM,
    fallback: Optional[str] = None,
) -> ApprovalDecision:
    """决定这一次调用：自动放行 / 问人 / 直接拒。

    * 未声明策略 → ``ask``（与"没有本机制"时的行为一致，这是零回归的根）；
    * 策略抛错或返回非法值 → ``ask``（**fail closed**：策略是别人写的代码，它会错）；
    * ``unknown`` 不可比较，一律 ``ask``；
    * 风险高于阈值 → ``ask``；
    * 风险不高于阈值但**没有兜底机制** → ``ask``（自动放行的唯一理由是机制兜底）；
    * 阈值配错 → 按最保守处理（全都问）。

    风险值**精确匹配**词表，不做 strip/lower 归一：归一化会把拼错的 ``"LOW "`` 悄悄
    变成自动放行（见 ``docs/技术选型决策.md`` D-006 引的 fail-open 事故）。
    """
    if risk_policy is None:
        return ApprovalDecision(DECISION_ASK, "未声明风险策略，按需审批", RISK_UNKNOWN, "default")

    if threshold not in _RISK_ORDER:
        logger.warning("阈值取值非法（%r），按最保守处理：全部按需审批", threshold)
        return ApprovalDecision(
            DECISION_ASK, f"阈值非法（{threshold!r}），按需审批", RISK_UNKNOWN, "tool")

    try:
        raw = risk_policy(dict(args or {}))
    except Exception:  # noqa: BLE001 - 策略是领域包写的代码，出错不能拖垮主流程
        logger.exception("风险策略执行失败，降级为按需审批：%s", tool_name)
        return ApprovalDecision(DECISION_ASK, "风险策略执行失败，按需审批", RISK_UNKNOWN, "tool")

    risk = raw if isinstance(raw, str) else ""
    if risk not in _RISK_ORDER:
        return ApprovalDecision(
            DECISION_ASK, f"风险等级无法判定（{raw!r}），按需审批", RISK_UNKNOWN, "tool")

    if _RISK_ORDER[risk] > _RISK_ORDER[threshold]:
        return ApprovalDecision(DECISION_ASK, f"风险 {risk} 高于阈值 {threshold}", risk, "tool")

    if not fallback:
        return ApprovalDecision(
            DECISION_ASK, f"风险 {risk} 可自动放行，但当前没有可用的兜底机制", risk, "tool")

    return ApprovalDecision(
        DECISION_AUTO, f"风险 {risk} 且兜底机制可用（{fallback}）", risk, "tool", fallback)


__all__ = [
    "RISK_LOW", "RISK_MEDIUM", "RISK_HIGH", "RISK_UNKNOWN",
    "DECISION_AUTO", "DECISION_ASK", "DECISION_DENY",
    "ApprovalDecision", "decide_approval",
]
