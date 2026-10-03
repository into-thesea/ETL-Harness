"""scripts/depgraph.py —— 内部依赖图与循环依赖检查（评价体系 D10 的取证工具）。

**为什么不引 pydeps / import-linter**：我们只要两个结论 —— "有没有环"和"谁耦合谁"。
标准库的 ``ast`` 就够，而少一个开发依赖就是少一处要维护、少一处会腐坏的东西
（评价体系 §7 只要求"生成依赖图并查环"，没要求用哪个工具）。

用法::

    .venv/Scripts/python.exe scripts/depgraph.py            # 报告
    .venv/Scripts/python.exe scripts/depgraph.py --check    # 有环则退出码 1（可进 CI）

只分析**包内**依赖（``harness.*`` 与 ``packages.*`` 之间的互相引用），外部库不进来 ——
它们不进环的判断，进来只会把图淹掉。
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("harness", "packages")


def module_name(path: Path, root: Path) -> tuple[str, bool]:
    """``(模块名, 是不是包)``。

    第二个返回值不能省：``__init__.py`` 代表的**就是那个包本身**，而普通模块的所在包
    要少一层。第一版把两者当同一种东西处理，于是 ``packages/data_analysis/__init__.py``
    里的 ``from .package import …`` 被解成了 ``packages.package``（凭空多出一个节点）。
    """
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _resolve_relative(current: str, is_package: bool, level: int, module: str | None) -> str:
    """把 ``from ..x import y`` 还原成绝对模块名（``level`` 是点号个数）。"""
    parts = current.split(".")
    base = parts if is_package else parts[:-1]     # 当前模块**所在的包**
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, *(module.split(".") if module else [])])


def _imports(tree: ast.AST, current: str, is_package: bool) -> set[str]:
    """本模块引用的**包内**模块名（绝对导入 + 相对导入都归一成绝对名）。"""
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PACKAGES:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            name = (
                _resolve_relative(current, is_package, node.level, node.module)
                if node.level else (node.module or "")
            )
            if name and name.split(".")[0] in PACKAGES:
                found.add(name)
    return found


def build_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for package in PACKAGES:
        base = ROOT / package
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            name, is_package = module_name(path, ROOT)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:            # 解析不了要响亮，别静默少一条边
                raise SystemExit(f"无法解析 {path}：{exc}") from exc
            graph[name] |= _imports(tree, name, is_package)
    return {k: v for k, v in graph.items()}


def _top(name: str) -> str:
    """到"二级包"为止的归属：``harness.server.app`` -> ``harness.server``。"""
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else parts[0]


def collapse(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    """把模块图塌成二级包图 —— 查环与看耦合在这一层才读得动。"""
    out: dict[str, set[str]] = defaultdict(set)
    for src, targets in graph.items():
        for dst in targets:
            a, b = _top(src), _top(dst)
            if a != b:
                out[a].add(b)
    return out


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan 强连通分量；大小 >1 的分量就是环（自指也算）。"""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    counter = 0
    cycles: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal counter
        index[node] = low[node] = counter
        counter += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in sorted(graph.get(node, ())):
            if nxt not in index:
                strongconnect(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == node:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))
    for node in sorted(graph):
        if node not in index:
            strongconnect(node)
    return cycles


def report(graph: dict[str, set[str]], top_n: int) -> int:
    collapsed = collapse(graph)
    cycles = find_cycles(collapsed)

    print(f"模块数：{len(graph)}    包内依赖边：{sum(len(v) for v in graph.values())}")
    print(f"二级包数：{len(collapsed)}")

    fan_out = sorted(collapsed.items(), key=lambda kv: len(kv[1]), reverse=True)
    fan_in: dict[str, int] = defaultdict(int)
    for targets in collapsed.values():
        for t in targets:
            fan_in[t] += 1

    print(f"\n扇出最大（改了它要连带测的地方最多）  top {top_n}")
    for name, targets in fan_out[:top_n]:
        print(f"  {name:<28} -> {len(targets):>2} 个包：{', '.join(sorted(targets))}")

    print(f"\n被依赖最多（动它风险最大）  top {top_n}")
    for name, count in sorted(fan_in.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
        print(f"  {name:<28} <- {count} 个包")

    if cycles:
        print(f"\n循环依赖（二级包）：{len(cycles)} 处 —— 必须为 0")
        for component in cycles:
            print(f"  [X] {' <-> '.join(component)}")
        return 1
    print("\n循环依赖（二级包）：0 处 [OK]")

    # 塌成包会把**小环藏起来**（A.x 引 B.y、B.y 又引 A.z）。再查一遍模块级，
    # 但只报不拦 —— 模块级小环在 Python 里常见且未必有害，值不值得拆要看人。
    module_cycles = find_cycles(graph)
    if module_cycles:
        print(f"循环依赖（模块级）：{len(module_cycles)} 处（只报不拦，供人判断）")
        for component in module_cycles[:5]:
            print(f"  [!] {' <-> '.join(component)}")
        if len(module_cycles) > 5:
            print(f"  ...还有 {len(module_cycles) - 5} 处")
    else:
        print("循环依赖（模块级）：0 处 [OK]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="内部依赖图与循环依赖检查")
    parser.add_argument("--check", action="store_true", help="有环则退出码 1（可进 CI）")
    parser.add_argument("--top", type=int, default=8, help="榜单长度")
    args = parser.parse_args()

    code = report(build_graph(), args.top)
    return code if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
