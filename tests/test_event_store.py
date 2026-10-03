"""tests.test_event_store —— 事件落盘与历史回放。

设计见 `docs/技术选型决策.md` D-008，协议见 `docs/控制台与事件协议设计.md`。

三条不变量：

1. **一个任务一个 append-only JSONL** —— 回放 = 读一个文件，不需要索引；
2. **读取必须容忍截断** —— 进程被杀会留下半行，跳过坏行，不抛；
3. **`thread_id` 会被当文件名** —— 而它来自 URL 路径参数，所以必须先消毒，
   否则 `/tasks/..%2F..%2Fetc%2Fpasswd/stream` 就是一次任意文件读取。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from harness.event_store import EventStore
from harness.events import EventBus


def _ev(thread: str, seq: int, etype: str = "TOOL_CALL_START") -> dict:
    return {"type": etype, "thread_id": thread, "seq": seq, "ts": 1.0 * seq,
            "data": {"n": seq}}


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    store = EventStore(str(tmp_path))
    for i in range(1, 4):
        store.append(_ev("t1", i))

    events = store.read("t1")
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["type"] == "TOOL_CALL_START"
    assert events[0]["data"] == {"n": 1}


def test_threads_are_isolated(tmp_path: Path) -> None:
    store = EventStore(str(tmp_path))
    store.append(_ev("t1", 1))
    store.append(_ev("t2", 2))

    assert [e["seq"] for e in store.read("t1")] == [1]
    assert [e["seq"] for e in store.read("t2")] == [2]
    assert store.read("never-seen") == []


def test_read_tolerates_truncated_last_line(tmp_path: Path) -> None:
    """进程被杀会留下半行 —— 跳过它，别让整个任务的历史都读不出来。"""
    store = EventStore(str(tmp_path))
    store.append(_ev("t1", 1))
    with open(tmp_path / "t1.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"type": "TOOL_CALL_END", "seq": 2, "data": {"un": ')  # 半行

    events = store.read("t1")
    assert [e["seq"] for e in events] == [1], "坏行之后的都放弃，但前面的必须留下"


def test_path_traversal_is_refused(tmp_path: Path) -> None:
    """thread_id 来自 URL 路径参数 —— 不能拿它直接当文件名。"""
    store = EventStore(str(tmp_path))
    secret = tmp_path / ".." / "secret.jsonl"
    secret.write_text(json.dumps(_ev("x", 1)), encoding="utf-8")

    assert store.read("../../secret") == []
    assert store.read("..\\..\\secret") == []

    store.append(_ev("../../evil", 1))
    assert list(tmp_path.glob("**/*.jsonl")) == [], "越界的 id 连写都不该发生"


def test_disabled_store_is_inert(tmp_path: Path) -> None:
    store = EventStore(str(tmp_path), enabled=False)
    store.append(_ev("t1", 1))
    assert store.read("t1") == []
    assert list(tmp_path.glob("*.jsonl")) == []


def test_append_never_raises_on_io_failure(tmp_path: Path) -> None:
    """落盘是观测旁路 —— 它坏了不能拖垮主流程（与审计、trace 出口同一条规矩）。"""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")     # 目录位置是个文件
    store = EventStore(str(blocker / "events"))

    store.append(_ev("t1", 1))                    # 不抛
    assert store.read("t1") == []


def test_max_file_bytes_caps_appends(tmp_path: Path) -> None:
    store = EventStore(str(tmp_path), max_file_bytes=400)
    for i in range(1, 40):
        store.append(_ev("t1", i))

    events = store.read("t1")
    assert 0 < len(events) < 39, "超过封顶后不再写，但已写的要留住"
    assert list(tmp_path.glob("t1.jsonl.*")) == [], "封顶不是轮转：不产生分片文件"


def test_cleanup_drops_old_files_only(tmp_path: Path) -> None:
    store = EventStore(str(tmp_path), retention_days=1)
    store.append(_ev("old", 1))
    store.append(_ev("fresh", 1))

    old = tmp_path / "old.jsonl"
    stale = time.time() - 3 * 86400
    os.utime(old, (stale, stale))

    assert store.cleanup() == 1
    assert not old.exists()
    assert (tmp_path / "fresh.jsonl").exists()


def test_cleanup_disabled_when_retention_is_zero(tmp_path: Path) -> None:
    """``retention_days=0`` 表示不清理 —— 与审批不过期用 ``<=0`` 同一套写法。"""
    store = EventStore(str(tmp_path), retention_days=0)
    store.append(_ev("old", 1))
    old = tmp_path / "old.jsonl"
    stale = time.time() - 999 * 86400
    os.utime(old, (stale, stale))

    assert store.cleanup() == 0
    assert old.exists()


# ======================================================================
# 总线接线：事件真的会被写下来，且在 unbind 之后仍然读得到
# ======================================================================
def test_bus_persists_events_for_bound_threads(tmp_path: Path) -> None:
    bus = EventBus(store=EventStore(str(tmp_path)))
    bus.bind("t1", "tr1")
    bus.publish("tr1", "TOOL_CALL_START", {"tool": "x"})

    events = EventStore(str(tmp_path)).read("t1")
    assert [e["type"] for e in events] == ["TOOL_CALL_START"]
    assert events[0]["thread_id"] == "t1" and events[0]["seq"] == 1


def test_bus_does_not_persist_unbound_traces(tmp_path: Path) -> None:
    """没有任务在跑是**最常见**的路径 —— 它不该在磁盘上留下任何东西。"""
    bus = EventBus(store=EventStore(str(tmp_path)))
    assert bus.publish("nobody", "TOOL_CALL_START", {}) == 0
    assert list(tmp_path.glob("*.jsonl")) == []


def test_events_outlive_unbind_on_disk(tmp_path: Path) -> None:
    """这就是"落盘"的全部意义：内存历史随 unbind 清掉，盘上的还在。"""
    store = EventStore(str(tmp_path))
    bus = EventBus(store=store)
    bus.bind("t1", "tr1")
    bus.publish("tr1", "RUN_STARTED", {"goal": "g"})
    bus.unbind("t1")

    assert bus.is_bound("t1") is False
    assert store.read("t1") and len(store.read("t1")) == 1, "内存清了，盘上必须还在"


def test_sse_replays_a_finished_task_from_disk(tmp_path: Path, monkeypatch) -> None:
    """端到端：任务跑完（总线已解绑）后重连 SSE，事件从**磁盘**回放。

    这是"跑完还能看见"的证明 —— 也是方向三最后一块的验收。走真实服务、真实图、
    真实 HTTP；只有落盘开关与目录被指到 tmp。
    """
    import time as _time

    from fastapi.testclient import TestClient

    from harness.config import settings
    from harness.events import build_event_bus, reset_event_bus
    from harness.server.app import create_app
    from tests.test_console_api import TERMINAL, _finished_service

    monkeypatch.setattr(settings.event, "enabled", True)
    monkeypatch.setattr(settings.event, "dir", str(tmp_path / "events"))
    reset_event_bus(None)                      # 重建总线，让它按新配置造落盘器

    try:
        svc = _finished_service()
        with TestClient(create_app(svc)) as client:
            tid = client.post("/api/v1/tasks", json={"goal": "对销售数据做端到端分析"}).json()["thread_id"]

            deadline = _time.time() + 120
            status = {}
            while _time.time() < deadline:
                status = client.get(f"/api/v1/tasks/{tid}").json()
                if status["status"] in TERMINAL and not build_event_bus().is_bound(tid):
                    break
                _time.sleep(0.2)
            assert status.get("status") == "finished", status

            # 此刻内存历史已经没了 —— 下面这些事件只可能来自磁盘
            assert build_event_bus().is_bound(tid) is False
            assert (tmp_path / "events" / f"{tid}.jsonl").exists(), "任务事件必须落了盘"

            lines: list[str] = []
            with client.stream("GET", f"/api/v1/tasks/{tid}/stream") as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    lines.append(line)
                    if line.strip() == "event: done" or len(lines) > 2000:
                        break
            blob = "\n".join(lines)

            assert "event: RUN_STARTED" in blob, "回放里应有根 Span 事件"
            assert "event: TOOL_CALL_START" in blob, "回放里应有工具调用事件"
            assert "event: SUBAGENT_STARTED" in blob, "回放里应有子 Agent 事件"
    finally:
        reset_event_bus(None)
