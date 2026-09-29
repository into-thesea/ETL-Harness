"""tests._smoke_c6 —— C6 可靠投递冒烟：spool 缓冲 → 恢复补发 → 清理（离线）。

不连真实 Kafka：用 __new__ 跳过 __init__ 的后台连接线程，注入 fake broker。

运行（项目根）：
    .venv\\Scripts\\python.exe -m tests._smoke_c6
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from types import SimpleNamespace

from harness.trace.kafka_producer import KafkaProducerWrapper


def _make_wrapper(tmpdir: str) -> KafkaProducerWrapper:
    """构造一个不启动连接线程、spool 指向 tmpdir 的 wrapper（离线隔离）。"""
    w = KafkaProducerWrapper.__new__(KafkaProducerWrapper)
    w._producer = None
    w._connected = False
    w._spool_dir = tmpdir
    w._dead_letter_dir = os.path.join(tmpdir, "_dead_letter")
    os.makedirs(w._dead_letter_dir, exist_ok=True)
    w._spool_lock = threading.RLock()
    w._spool_counter = 0
    w._spool_size = 0
    w._spool_dropped = 0
    w._drain_lock = threading.Lock()
    return w


class _FakeFuture:
    def get(self, timeout=None):
        return SimpleNamespace()

    def add_callback(self, *a, **k):
        return self

    def add_errback(self, *a, **k):
        return self


class _FakeBroker:
    """模拟已连接的 KafkaProducer：记录收到的消息。"""

    def __init__(self) -> None:
        self.received: list[tuple] = []

    def send(self, topic, value, key):
        self.received.append((topic, value, key))
        return _FakeFuture()

    def flush(self, timeout=None):
        pass

    def close(self):
        pass


class _BrokenFuture:
    def get(self, timeout=None):
        raise RuntimeError("broker not writable")

    def add_callback(self, *a, **k):
        return self

    def add_errback(self, *a, **k):
        return self


class _BrokenBroker:
    def send(self, topic, value, key):
        return _BrokenFuture()


def test_spool_then_replay() -> None:
    w = _make_wrapper(tempfile.mkdtemp())

    # 未连接：两条消息落 spool
    w.send("governed_audit", {"trace_id": "t1", "n": 1}, key="t1")
    w.send("governed_audit", {"trace_id": "t2", "n": 2}, key="t2")
    assert w.spool_status()["pending"] == 2

    # 连接恢复：补发
    broker = _FakeBroker()
    w._producer = broker
    w._connected = True
    w._drain_spool()

    assert len(broker.received) == 2, broker.received
    # 保序：t1 先于 t2
    assert [v["trace_id"] for _, v, _ in broker.received] == ["t1", "t2"]
    assert w.spool_status()["pending"] == 0
    print("[1] 未连接缓冲 → 恢复后按序补发 → spool 清空 ok")


def test_replay_failure_keeps_spool() -> None:
    w = _make_wrapper(tempfile.mkdtemp())
    w.send("governed_audit", {"trace_id": "t9"}, key="t9")
    assert w.spool_status()["pending"] == 1

    # broker 写入确认失败：spool 文件必须保留，不能丢
    w._producer = _BrokenBroker()
    w._connected = True
    w._drain_spool()
    assert w.spool_status()["pending"] == 1

    # 恢复后仍可正常补发
    broker = _FakeBroker()
    w._producer = broker
    w._drain_spool()
    assert w.spool_status()["pending"] == 0
    assert len(broker.received) == 1
    print("[2] 补发失败保留 spool、再次恢复补发 ok")


def test_corrupt_to_dead_letter() -> None:
    w = _make_wrapper(tempfile.mkdtemp())
    # 一个合法、一个损坏
    w.send("governed_audit", {"trace_id": "ok"}, key="ok")
    with open(os.path.join(w._spool_dir, "000_corrupt.json"), "w", encoding="utf-8") as fh:
        fh.write("{not-json")

    w._producer = _FakeBroker()
    w._connected = True
    w._drain_spool()

    assert w.spool_status()["pending"] == 0  # 合法的已补发
    assert os.path.exists(os.path.join(w._dead_letter_dir, "000_corrupt.json"))
    print("[3] 损坏 spool 文件进 dead-letter、不阻断补发 ok")


def test_spool_cap_drops_oldest() -> None:
    """spool 必须有上限。

    Kafka 长期不可达时 spool 只进不出 —— 没有上限就是一条写满磁盘的慢路径。
    上限只丢**未上送**的消息，审计的本地留档由 audit.jsonl 独立保证。
    """
    from harness.config import settings

    original = settings.kafka.spool_max_files
    settings.kafka.spool_max_files = 5
    try:
        with tempfile.TemporaryDirectory() as tmp:
            w = _make_wrapper(tmp)
            for i in range(20):
                w._spool_message("governed_audit", {"i": i}, key=str(i))

            names = sorted(n for n in os.listdir(tmp) if n.endswith(".json"))
            assert len(names) == 5, f"应保留上限条数，实际 {len(names)}"

            status = w.spool_status()
            assert status["pending"] == 5
            assert status["dropped"] == 15, status

            newest = json.loads(open(os.path.join(tmp, names[-1]), encoding="utf-8").read())
            assert newest["message"]["i"] == 19, "丢的必须是最旧的，最新的要留着"
    finally:
        settings.kafka.spool_max_files = original
    print("[4] spool 达上限丢最旧的、保留最新 ok")


def _main() -> None:
    test_spool_then_replay()
    test_replay_failure_keeps_spool()
    test_corrupt_to_dead_letter()
    test_spool_cap_drops_oldest()
    print("\n=== C6（Kafka spool 缓冲 + 恢复后补发）冒烟全部通过 ===")


if __name__ == "__main__":
    _main()
