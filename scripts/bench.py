"""scripts/bench.py —— Tool Broker 的最小基准（评价体系 D6 的取证工具）。

量三件事，都**不碰网络、不碰 LLM、不碰沙箱**（沙箱要起容器，那是压测另一件事）：

1. **单次工具延迟**（D6.1）：顺序调用 N 次，给 p50 / p95；
2. **并发吞吐**（D6.2）：线程池打满，给 QPS；
3. **缓存对照**（D6.3 的机制侧证据）：同一调用重复打，缓存开 / 关各一次。

**这不是压测**：单进程、单机、本地内存工具 —— 它量的是**管控链路自身的开销**
（PDP / 校验 / 熔断 / 限流 / 审计 / 事件），不是系统容量。容量要看部署形态。

用法::

    .venv/Scripts/python.exe scripts/bench.py [--n 2000] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

from harness.models import ToolDef
from harness.tool_broker import ToolBroker

_STATE = {"role": "admin", "session_id": "bench", "trace_id": ""}


def _build(*, cache: bool) -> ToolBroker:
    """一个只有"回声"工具的 Broker。关掉沙箱/熔断，隔离出管控链路本身的成本。

    缓存放**显式实例**而不是 ``cache=True`` —— 构造参数只认 ``False``（关闭）、
    ``None``（按配置自动）和实例；传 ``True`` 会被原样存下去，直到 ``invoke`` 深处
    才炸成 ``AttributeError: 'bool' object has no attribute 'fingerprint'``。
    （已记进评价报告 D7.2，本轮不改。）
    """
    from harness.cache import ToolResultCache

    broker = ToolBroker(
        sandbox_executor=False,
        circuit_breaker=False,
        cache=ToolResultCache(max_entries=1024) if cache else False,
    )
    broker.register(
        ToolDef(
            name="echo",
            description="原样回显",
            parameters={"type": "object", "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
            cacheable=cache,
            rate_limit_per_min=10_000_000,     # 基准不该被限流挡住
        ),
        lambda args, ctx: (True, args.get("text", ""), {}),
    )
    return broker


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    idx = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return p50, ordered[idx], statistics.fmean(ordered)


def bench_latency(n: int) -> tuple[float, float, float]:
    broker = _build(cache=False)
    samples: list[float] = []
    for i in range(n):
        t0 = time.perf_counter()
        broker.invoke("echo", {"text": f"hi-{i}"}, _STATE)
        samples.append((time.perf_counter() - t0) * 1000)   # ms
    return _percentiles(samples)


def bench_throughput(n: int, workers: int) -> float:
    """QPS：把 n 次调用摊到 workers 个线程上 —— 单线程量的是延迟，多线程才量得到吞吐。"""
    broker = _build(cache=False)
    barrier = threading.Barrier(workers)
    per_worker = max(1, n // workers)

    def run(worker: int) -> None:
        barrier.wait()                                      # 同时起跑，别让先跑的占便宜
        for i in range(per_worker):
            broker.invoke("echo", {"text": f"w{worker}-{i}"}, _STATE)

    threads = [threading.Thread(target=run, args=(w,)) for w in range(workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    return (per_worker * workers) / elapsed


def bench_cache(n: int) -> tuple[float, float, int]:
    """同一调用重复打：缓存关 vs 开。返回 (关掉的中位延迟, 打开的中位延迟, 命中次数)。"""
    payload = {"text": "same"}

    off = _build(cache=False)
    off_samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        off.invoke("echo", dict(payload), _STATE)
        off_samples.append((time.perf_counter() - t0) * 1000)

    on = _build(cache=True)
    on_samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        on.invoke("echo", dict(payload), _STATE)
        on_samples.append((time.perf_counter() - t0) * 1000)

    # 命中数从缓存对象上读：``get_stats()`` 里**没有** cache 这一项，
    # 第一版读错了地方，于是 A/B 对照的"命中 0 次"看着像缓存没生效。
    hits = int(getattr(on.cache, "hits", 0))
    return statistics.median(off_samples), statistics.median(on_samples), hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool Broker 最小基准")
    parser.add_argument("--n", type=int, default=2000, help="顺序调用次数")
    parser.add_argument("--concurrency", type=int, default=8, help="并发线程数")
    args = parser.parse_args()

    p50, p95, mean = bench_latency(args.n)
    print(f"单次工具延迟（n={args.n}）  p50={p50:.3f} ms  p95={p95:.3f} ms  均值={mean:.3f} ms")

    qps = bench_throughput(args.n, args.concurrency)
    print(f"并发吞吐（{args.concurrency} 线程）  {qps:.0f} QPS")

    off_ms, on_ms, hits = bench_cache(max(200, args.n // 4))
    print(f"缓存对照  关闭 {off_ms:.3f} ms → 打开 {on_ms:.3f} ms（命中 {hits} 次）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
