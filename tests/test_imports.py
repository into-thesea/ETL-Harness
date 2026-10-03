"""tests.test_imports —— 模块导入冒烟：遍历 harness/ 与 packages/ 全部模块。

对应阶段7验收口径「模块导入 100%」：任何模块存在导入期错误（语法、循环导入、
缺失依赖）都会在这里失败，而不是等到运行期。

**遍历的包名要跟着代码走**：原先是 ``("harness", "tools")``，而领域能力早已迁到
``packages/`` —— ``tools/`` 不存在，于是这个"100% 导入"的检查**静默漏掉了整个领域
半边**，报告上却显示通过。与 ``.coveragerc`` 的 ``source`` 是同一类腐坏。
"""

from __future__ import annotations

import importlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: 需要逐个 import 的包。新增顶层包时要一起加进来 —— 漏了不会报错，只会少查。
PACKAGES = ("harness", "packages")


def _iter_modules() -> list[str]:
    mods: list[str] = []
    for pkg in PACKAGES:
        for path in (ROOT / pkg).rglob("*.py"):
            rel = path.relative_to(ROOT).with_suffix("")
            mods.append(".".join(rel.parts))
    return sorted(mods)


def test_all_modules_importable() -> None:
    failures: dict[str, str] = {}
    mods = _iter_modules()
    for name in mods:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"
    assert not failures, (
        "以下模块导入失败：\n" + "\n".join(f"  {k}: {v}" for k, v in failures.items())
    )
    print(f"[imports] {len(mods)} modules importable ok")
