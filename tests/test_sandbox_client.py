"""tests/test_sandbox_client.py —— SandboxClient 的 fail-closed 判定（不依赖真实沙箱）。

覆盖 available() 的三条拒绝路径中可离线验证的两条：
  1. 未配置 SANDBOX_API_KEY → 不可用（本轮新增，错误要早于网络探测暴露）
  2. 服务端不可达 → 不可用，并指向 start.ps1
"""

from __future__ import annotations

from types import SimpleNamespace

from harness.sandbox.client import SandboxClient


def _client_with(**overrides) -> SandboxClient:
    """构造使用内存配置的 SandboxClient，不触碰真实 settings。"""
    base = {
        "server_url": "http://127.0.0.1:8080",
        "api_key": "",
        "image": "etl-harness-sandbox:latest",
    }
    base.update(overrides)
    return SandboxClient(SimpleNamespace(**base))


def test_missing_api_key_is_unavailable() -> None:
    client = _client_with(api_key="")
    ok, reason = client.available()
    assert ok is False
    assert "SANDBOX_API_KEY" in reason


def test_server_unreachable_is_unavailable(monkeypatch) -> None:
    import httpx

    def _raise(*args, **kwargs):
        raise httpx.ConnectError("[stub] connection refused")

    monkeypatch.setattr(httpx, "get", _raise)

    client = _client_with(api_key="some-configured-key")
    ok, reason = client.available()
    assert ok is False
    # 应指向恢复入口，而不是模糊的网络错误
    assert "start.ps1" in reason
