"""harness.server.auth —— 服务端鉴权（Bearer 令牌 → 角色）。

**为什么必须有**：在此之前服务端没有任何鉴权，而且请求体里的 ``role`` 是客户端自己
填的（默认 ``admin``）—— 能连上端口就等于以管理员身份跑任务，还能批准自己触发的高危
操作。现在：

- 角色**只能来自令牌**（``AUTH_TOKENS`` 里配的那份映射），请求体的 ``role`` 一律无效；
- 业务路由与 ``/docs`` / ``/openapi.json`` 都要令牌，``/health`` 除外（探活）；
- ``?token=`` **只对 SSE 路由**生效 —— 浏览器原生 ``EventSource`` 不能设请求头，
  但查询参数一旦被普遍接受就成了绕过鉴权的后门；
- 令牌比对用 :func:`hmac.compare_digest`，且**日志里永不记录令牌原文**。

未配置令牌却启用鉴权时，服务端**启动即失败**（见 :func:`build_authenticator`）：
静默放行正是加鉴权时最危险的失败方向。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# 不需要鉴权的路径（探活）
ANONYMOUS_PATHS = frozenset({"/health"})

# 控制台**外壳**（HTML / CSS / JS）可匿名取：它是纯静态资源，本身不含任何数据 ——
# 任务、指标、审批一律走受保护的 /api/v1（前端把令牌放在 Authorization 头里）。
# 若连外壳都要令牌，浏览器就永远打不开填令牌的页面，令牌无处可填、控制台形同不存在。
_SHELL_PATHS = frozenset({"/", "/index.html", "/favicon.ico"})
_SHELL_PREFIXES = ("/assets/",)


def _is_console_shell(path: str) -> bool:
    """是否控制台外壳资源（静态、无数据）。"""
    return path in _SHELL_PATHS or path.startswith(_SHELL_PREFIXES)

# 401 响应头：告诉调用方用哪种认证方式
_WWW_AUTHENTICATE = 'Bearer realm="governed"'


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Principal:
    """一次请求背后的身份（由令牌推导，不接受调用方自称）。"""

    role: str
    name: str
    token_hash: str = ""
    """令牌指纹（SHA-256 前 16 位），用于身份判定，不存令牌原文。"""
    authenticated: bool = True
    """是否来自真实令牌。``False`` 只出现在"鉴权关闭"的开发模式下 ——
    那种模式只有一个身份，"谁批准谁"的职责分离规则无从成立，故不适用。"""


class Authenticator:
    """令牌表与解析规则。"""

    def __init__(self, config: Any = None) -> None:
        if config is None:
            from harness.config import settings

            config = settings.auth
        self.enabled = bool(getattr(config, "enabled", True))
        self.approver_roles = {
            r.strip() for r in str(getattr(config, "approver_roles", "") or "").split(",") if r.strip()
        }
        self._table: dict[str, Principal] = (
            _load_tokens(str(getattr(config, "tokens", "") or "")) if self.enabled else {}
        )

    # ------------------------------------------------------------------
    def resolve(self, token: Optional[str]) -> Optional[Principal]:
        """令牌 → 身份；无效返回 None。逐个 :func:`hmac.compare_digest`（避免提前返回）。"""
        if not token:
            return None
        for candidate, principal in self._table.items():
            if hmac.compare_digest(candidate, token):
                return principal
        return None

    def is_approver(self, principal: Principal) -> bool:
        return principal.role in self.approver_roles

    # ------------------------------------------------------------------
    @staticmethod
    def extract_token(request: Request) -> Optional[str]:
        """从 ``Authorization: Bearer`` 取令牌；SSE 路由额外接受 ``?token=``。"""
        header = request.headers.get("authorization") or ""
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
        if _allows_query_token(request.url.path):
            return request.query_params.get("token") or None
        return None


def _allows_query_token(path: str) -> bool:
    """只有流式订阅允许把令牌放查询参数（EventSource 不能设请求头）。"""
    return path.rstrip("/").endswith("/stream")


def _load_tokens(raw: str) -> dict[str, Principal]:
    """解析 ``AUTH_TOKENS``；缺失 / 非法一律 RuntimeError（启动即失败）。

    Raises:
        RuntimeError: 未配置令牌、非 JSON 对象、或某个令牌缺少 ``role``。
    """
    if not raw.strip():
        raise RuntimeError(
            "AUTH_ENABLED=true 但未配置 AUTH_TOKENS —— 没有凭据就不该对外提供服务。"
            "请配置令牌表，或显式设 AUTH_ENABLED=false（仅限本机开发）"
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"AUTH_TOKENS 不是合法 JSON：{e}") from e
    if not isinstance(parsed, dict) or not parsed:
        raise RuntimeError('AUTH_TOKENS 必须是非空对象：{"<令牌>": {"role": "...", "name": "..."}}')

    table: dict[str, Principal] = {}
    for token, meta in parsed.items():
        if not isinstance(meta, dict) or not meta.get("role"):
            raise RuntimeError(
                f"AUTH_TOKENS 中令牌 {token[:4]}… 缺少 role —— 没有角色的令牌无法做权限判定"
            )
        table[str(token)] = Principal(
            role=str(meta["role"]), name=str(meta.get("name") or meta["role"]),
            token_hash=_hash_token(str(token)),
        )
    return table


def build_authenticator(config: Any = None) -> Authenticator:
    """构造鉴权器；未启用时打 WARNING（别让人以为自己在跑一个受保护的服务）。"""
    auth = Authenticator(config)
    if not auth.enabled:
        logger.warning(
            "服务端鉴权已关闭（AUTH_ENABLED=false）：任何能连上端口的人都能创建任务、"
            "查看状态并批准高危操作。仅限本机开发使用。"
        )
    else:
        logger.info("服务端鉴权已启用：%d 个令牌，审批角色 %s", len(auth._table), sorted(auth.approver_roles))
    return auth


def install_auth(app: FastAPI, auth: Authenticator) -> None:
    """把鉴权装成 HTTP 中间件（统一覆盖业务路由与 /docs、/openapi.json）。

    例外只有两处：探活 ``/health``，以及控制台**外壳**（见 :func:`_is_console_shell`）。
    两者都不含数据，页面上的数据仍要带令牌去取。
    """

    @app.middleware("http")
    async def _auth_gate(request: Request, call_next):  # type: ignore[no-untyped-def]
        if (
            not auth.enabled
            or request.url.path in ANONYMOUS_PATHS
            or _is_console_shell(request.url.path)
        ):
            return await call_next(request)

        principal = auth.resolve(auth.extract_token(request))
        if principal is None:
            # 只记路径与来源，**不记令牌原文**
            client = request.client.host if request.client else "?"
            logger.warning("鉴权失败：%s %s from %s", request.method, request.url.path, client)
            return JSONResponse(
                {"detail": "缺少或无效的凭据"},
                status_code=401,
                headers={"WWW-Authenticate": _WWW_AUTHENTICATE},
            )

        request.state.principal = principal
        return await call_next(request)


def principal_of(request: Request) -> Principal:
    """取当前请求的身份（中间件已放行；鉴权关闭时给一个显式的开发身份）。"""
    principal = getattr(request.state, "principal", None)
    if principal is None:
        return Principal(role="admin", name="auth-disabled", authenticated=False)
    return principal


__all__ = [
    "ANONYMOUS_PATHS",
    "Authenticator",
    "Principal",
    "build_authenticator",
    "install_auth",
    "principal_of",
]
