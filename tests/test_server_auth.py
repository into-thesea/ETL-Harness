"""tests.test_server_auth —— 服务端鉴权与角色来源（对内网/对外提供服务的前提）。

为何存在：这个服务此前**完全没有鉴权**，而且请求体里的 ``role`` 是客户端自己填的
（``schemas.py`` 默认 ``"admin"``，``service.create_task`` 的默认也是 ``admin``）——
也就是说任何能连上端口的人**默认以 admin 身份**跑任务，还能直接批准自己触发的高危操作。
本文件把四件事钉死：

1. 业务路由必须带令牌（``/health`` 除外）；``/docs``、``/openapi.json`` 一并纳入；
2. **角色只能来自令牌**，请求体里再传 ``role`` 一律无效（否则鉴权只是装饰）；
3. ``?token=`` **只有 SSE 路由认**（浏览器 EventSource 不能设请求头），其它路由带了
   也当没带 —— 否则查询参数会变成绕过鉴权的通用后门；
4. 审批需要审批人角色，且**不能批准自己发起的任务**（职责分离）。

其余用例通过 ``tests/conftest.py`` 把 ``AUTH_ENABLED`` 设为 ``false`` 后再跑。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_server_auth.py -q
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from harness.server.app import create_app
from tests._smoke_server import ApprovalLLM, _offline_service

ANALYST_TOKEN = "tok-analyst-0123456789"
ADMIN_TOKEN = "tok-admin-0123456789"
TOKENS = {
    ANALYST_TOKEN: {"role": "analyst", "name": "分析员"},
    ADMIN_TOKEN: {"role": "admin", "name": "管理员"},
}

ANALYST = {"Authorization": f"Bearer {ANALYST_TOKEN}"}
ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

# 需要鉴权的业务路由（method, path, 是否有请求体）
PROTECTED = [
    ("post", "/api/v1/tasks", {"goal": "随便看看"}),
    ("get", "/api/v1/tasks/nonexistent", None),
    ("get", "/api/v1/tasks/nonexistent/approvals", None),
    ("post", "/api/v1/tasks/nonexistent/approval", {"approved": True}),
    ("get", "/api/v1/tasks/nonexistent/stream", None),
]


@pytest.fixture
def auth_env(monkeypatch):
    """打开鉴权并注入令牌表（在 create_app 之前生效）。"""
    from harness.config import settings

    monkeypatch.setattr(settings.auth, "enabled", True)
    monkeypatch.setattr(settings.auth, "tokens", json.dumps(TOKENS, ensure_ascii=False))
    monkeypatch.setattr(settings.auth, "approver_roles", "admin")
    return settings.auth


def _client(service=None) -> TestClient:
    return TestClient(create_app(service or _offline_service()))


def _send(client: TestClient, method: str, path: str, body, headers):
    kwargs = {"headers": headers} if headers else {}
    if body is not None:
        kwargs["json"] = body
    return client.request(method.upper(), path, **kwargs)


# ----------------------------------------------------------------------
# 1：没有令牌 / 令牌错误 → 401
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_protected_routes_reject_missing_token(auth_env, method, path, body) -> None:
    with _client() as client:
        resp = _send(client, method, path, body, headers=None)
    assert resp.status_code == 401, f"{method} {path} 未鉴权却放行：{resp.status_code}"


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_protected_routes_reject_wrong_token(auth_env, method, path, body) -> None:
    with _client() as client:
        resp = _send(client, method, path, body, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401, f"{method} {path} 错令牌却放行"


def test_401_says_how_to_authenticate(auth_env) -> None:
    with _client() as client:
        resp = client.get("/api/v1/tasks/nonexistent")
    assert resp.headers.get("WWW-Authenticate", "").lower().startswith("bearer"), dict(resp.headers)


def test_health_is_anonymous(auth_env) -> None:
    """探活不该需要令牌（否则编排层的健康检查全废）。"""
    with _client() as client:
        assert client.get("/health").status_code == 200


def test_openapi_and_docs_require_token(auth_env) -> None:
    """对外服务不白送接口清单。"""
    with _client() as client:
        assert client.get("/openapi.json").status_code == 401
        assert client.get("/docs").status_code == 401
        assert client.get("/openapi.json", headers=ANALYST).status_code == 200


def test_valid_token_is_accepted(auth_env) -> None:
    with _client() as client:
        resp = client.post("/api/v1/tasks", json={"goal": "随便看看"}, headers=ANALYST)
    assert resp.status_code == 200, resp.text
    assert resp.json()["thread_id"]


# ----------------------------------------------------------------------
# 2：角色只能来自令牌
# ----------------------------------------------------------------------
def test_role_comes_from_token_not_request_body(auth_env, monkeypatch) -> None:
    """请求体里塞 role=admin 必须被忽略 —— 否则鉴权只是装饰。"""
    seen: list[str] = []
    service = _offline_service()

    async def _capture(goal: str, context: str = "", role: str = "admin") -> str:
        seen.append(role)
        return "thread-1"

    monkeypatch.setattr(service, "create_task", _capture)
    with _client(service) as client:
        resp = client.post(
            "/api/v1/tasks",
            json={"goal": "随便看看", "role": "admin"},   # 客户端自称 admin
            headers=ANALYST,                              # 令牌是 analyst
        )
    assert resp.status_code == 200, resp.text
    assert seen == ["analyst"], f"角色应来自令牌，实际：{seen}"


def test_request_body_role_field_is_gone(auth_env) -> None:
    """schema 里不该再留 role 字段（留着就会被当成契约）。"""
    from harness.server import schemas

    assert "role" not in schemas.CreateTaskRequest.model_fields


# ----------------------------------------------------------------------
# 3：?token= 只有 SSE 认
# ----------------------------------------------------------------------
def test_query_token_rejected_on_normal_routes(auth_env) -> None:
    with _client() as client:
        resp = client.get(f"/api/v1/tasks/nonexistent?token={ANALYST_TOKEN}")
    assert resp.status_code == 401, "查询参数令牌不得成为通用后门"


def test_query_token_accepted_on_stream(auth_env) -> None:
    """浏览器 EventSource 不能设请求头 —— 这条路由必须支持 ?token=。"""
    service = _offline_service()
    with TestClient(create_app(service)) as client:
        thread_id = client.post(
            "/api/v1/tasks", json={"goal": "对销售数据做端到端分析"}, headers=ANALYST
        ).json()["thread_id"]

        with client.stream(
            "GET", f"/api/v1/tasks/{thread_id}/stream?token={ANALYST_TOKEN}"
        ) as resp:
            assert resp.status_code == 200, resp.status_code
            first = next(resp.iter_lines(), "")
            assert "event:" in first, first


# ----------------------------------------------------------------------
# 4：审批需要审批人角色，且不能自批
# ----------------------------------------------------------------------
def _paused_thread(client: TestClient, headers, service) -> str:
    """造一个进入待审批的任务（coder → code_executor 触发 interrupt）。"""
    thread_id = client.post(
        "/api/v1/tasks", json={"goal": "运行一段代码"}, headers=headers
    ).json()["thread_id"]
    import time

    deadline = time.time() + 60
    while time.time() < deadline:
        state = client.get(f"/api/v1/tasks/{thread_id}", headers=headers).json()
        if state["status"] in ("awaiting_approval", "finished", "failed"):
            assert state["status"] == "awaiting_approval", state
            return thread_id
        time.sleep(0.2)
    raise AssertionError("超时未进入待审批")


def test_non_approver_cannot_approve(auth_env) -> None:
    service = _service_with_approval_llm()
    with _client(service) as client:
        thread_id = _paused_thread(client, ANALYST, service)
        resp = client.post(
            f"/api/v1/tasks/{thread_id}/approval",
            json={"approved": True, "comment": "自己放行"},
            headers=ANALYST,
        )
    assert resp.status_code == 403, f"非审批角色不该能审批：{resp.status_code} {resp.text}"


def test_approver_cannot_approve_own_task(auth_env) -> None:
    """职责分离：发起人不能自己批自己（否则审批门形同虚设）。"""
    service = _service_with_approval_llm()
    with _client(service) as client:
        thread_id = _paused_thread(client, ADMIN, service)   # 管理员发起
        resp = client.post(
            f"/api/v1/tasks/{thread_id}/approval",
            json={"approved": True, "comment": "自批"},
            headers=ADMIN,                                   # 管理员审批
        )
    assert resp.status_code == 403, f"不应允许自批：{resp.status_code} {resp.text}"


def test_other_approver_can_approve(auth_env) -> None:
    """正路：analyst 发起 → admin 审批，放行。"""
    service = _service_with_approval_llm()
    with _client(service) as client:
        thread_id = _paused_thread(client, ANALYST, service)
        resp = client.post(
            f"/api/v1/tasks/{thread_id}/approval",
            json={"approved": False, "comment": "不允许该操作"},
            headers=ADMIN,
        )
    assert resp.status_code == 200, resp.text


def _service_with_approval_llm():
    from harness.server.service import HarnessService

    return HarnessService(llm=ApprovalLLM())


# ----------------------------------------------------------------------
# 5：没配令牌就别启动
# ----------------------------------------------------------------------
def test_enabled_without_tokens_fails_fast(monkeypatch) -> None:
    from harness.config import settings

    monkeypatch.setattr(settings.auth, "enabled", True)
    monkeypatch.setattr(settings.auth, "tokens", "")
    with pytest.raises(RuntimeError, match="AUTH_TOKENS"):
        create_app(_offline_service())


def test_bad_tokens_json_fails_fast(monkeypatch) -> None:
    from harness.config import settings

    monkeypatch.setattr(settings.auth, "enabled", True)
    monkeypatch.setattr(settings.auth, "tokens", "{not json")
    with pytest.raises(RuntimeError, match="AUTH_TOKENS"):
        create_app(_offline_service())
