"""harness.pdp —— 策略决策点（Policy Decision Point，PDP）。

负责回答一个问题："某个角色，能不能调用某个工具？"

设计原则：
- 权限判断是【确定性代码】，不靠 prompt 约束 LLM 自觉。
- check 只做判断并返回 (是否允许, 理由)，绝不抛异常——
  权限不足是预期内的业务结果，要让上层把理由作为 observation 喂回 LLM。

规则匹配优先级（从高到低）：
    1. 精确匹配：(角色, 具体工具名)
    2. 通配匹配：(角色, "*")，表示该角色对所有工具的默认态度
    3. 默认策略：default_policy（默认 deny，最小权限原则）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PDP:
    """细粒度权限决策点。

    使用方式：
        pdp = PDP(default_policy="deny")
        pdp.add_rule("admin", "*", "allow")              # admin 放行所有工具
        pdp.add_rule("analyst", "calculator", "allow")   # analyst 只能用计算器
        allowed, reason = pdp.check("analyst", "sql_query", context)
    """

    ALLOW = "allow"
    DENY = "deny"

    def __init__(self, default_policy: str = DENY):
        """初始化 PDP。

        Args:
            default_policy: 没有任何规则匹配时的默认策略，默认 deny（最小权限）。
        """
        # 权限表：键是 (角色, 工具名) 元组，值是 "allow" / "deny"
        self._rules: dict[tuple[str, str], str] = {}
        self._default_policy = default_policy

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, config: Any) -> "PDP":
        """从 ``PermissionSettings`` 构造。

        默认 ``default_policy="allow"``：本方法用于把此前**根本没接进装配**的
        PDP 接上 —— 默认必须与"没接"等价（放行），否则一上线就把所有工具调用拦死。
        要最小权限就配 ``PERMISSION_PDP_DEFAULT_POLICY=deny`` 再显式列白名单。

        Raises:
            ValueError: ``PERMISSION_PDP_RULES`` 不是合法 JSON 数组。
        """
        import json

        default = str(
            getattr(config, "pdp_default_policy", cls.DENY) or cls.DENY
        ).strip().lower()
        if default not in (cls.ALLOW, cls.DENY):
            # 绝不默认成 allow：写错一个字母就变成"全放行"，而运维以为自己配了最小权限
            raise ValueError(
                f"PERMISSION_PDP_DEFAULT_POLICY 取值非法：{default!r}（只支持 allow | deny）"
            )
        pdp = cls(default_policy=default)

        raw = str(getattr(config, "pdp_rules", "") or "").strip()
        if not raw:
            return pdp
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"PERMISSION_PDP_RULES 不是合法 JSON：{e}") from e
        if not isinstance(parsed, list):
            raise ValueError("PERMISSION_PDP_RULES 必须是规则数组（JSON list）")
        for rule in parsed:
            if not isinstance(rule, dict) or "role" not in rule or "tool" not in rule:
                raise ValueError(f"PDP 规则缺少 role/tool 字段：{rule!r}")
            pdp.add_rule(str(rule["role"]), str(rule["tool"]), str(rule.get("effect", cls.ALLOW)))
        return pdp

    # ------------------------------------------------------------------
    # 规则管理
    # ------------------------------------------------------------------
    def add_rule(self, role: str, tool_name: str, effect: str = ALLOW) -> None:
        """添加（或覆盖）一条权限规则。

        Args:
            role: 角色名，如 "admin" / "analyst" / "viewer"。
            tool_name: 工具名；用 "*" 表示该角色对所有工具的通配规则。
            effect: "allow" 放行或 "deny" 拒绝。
        """
        if effect not in (self.ALLOW, self.DENY):
            raise ValueError(f"effect 必须是 allow 或 deny，得到: {effect}")
        self._rules[(role, tool_name)] = effect
        logger.info("PDP rule added: role=%s tool=%s effect=%s", role, tool_name, effect)

    def remove_rule(self, role: str, tool_name: str) -> bool:
        """删除一条规则，返回是否删到了。"""
        key = (role, tool_name)
        if key in self._rules:
            del self._rules[key]
            logger.info("PDP rule removed: role=%s tool=%s", role, tool_name)
            return True
        return False

    def list_rules(self) -> dict[str, str]:
        """列出所有规则（便于调试/展示）。"""
        return {f"{role}:{tool}": effect for (role, tool), effect in self._rules.items()}

    # ------------------------------------------------------------------
    # 核心：权限判断
    # ------------------------------------------------------------------
    def check(self, role: str, tool_name: str, context: Optional[dict] = None) -> tuple[bool, str]:
        """判断某角色能否调用某工具。

        Args:
            role: 调用者角色。
            tool_name: 想调用的工具名。
            context: 调用上下文（预留，可用于按时间段/数据级别等做更细判断）。

        Returns:
            (是否允许, 理由)。允许时理由为空字符串；拒绝时理由说明原因。
            无论允许还是拒绝都正常返回，不抛异常。
        """
        # 1. 精确匹配：(角色, 具体工具名)
        exact = self._rules.get((role, tool_name))
        if exact is not None:
            if exact == self.ALLOW:
                return True, ""
            return False, f"显式拒绝：角色 '{role}' 无权调用工具 '{tool_name}'"

        # 2. 通配匹配：(角色, "*")
        wildcard = self._rules.get((role, "*"))
        if wildcard is not None:
            if wildcard == self.ALLOW:
                return True, ""
            return False, f"通配规则拒绝：角色 '{role}' 无权调用工具 '{tool_name}'"

        # 3. 兜底：默认策略（默认 deny）
        if self._default_policy == self.ALLOW:
            return True, ""
        return False, f"默认拒绝：未给角色 '{role}' 配置工具 '{tool_name}' 的访问权限"

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------
    def allow(self, role: str, tool_name: str) -> None:
        """显式放行。"""
        self.add_rule(role, tool_name, self.ALLOW)

    def deny(self, role: str, tool_name: str) -> None:
        """显式拒绝。"""
        self.add_rule(role, tool_name, self.DENY)

    @property
    def default_policy(self) -> str:
        return self._default_policy


__all__ = ["PDP"]
