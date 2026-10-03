"""tests.test_console_frontend —— 控制台前端的一致性检查（挂账 #14）。

**为什么只有这几条**：前端是零构建的原生 JS（`harness/server/static/`，无 npm、无打包），
上框架的成本远大于收益。但"零构建"的代价是**前端没有任何用例照着**，切片 3 的两个真缺陷
（跳转路由名与路由表不一致、重复 class 属性让 `hidden` 失效）都是靠人眼读出来的。

挂账 #14 当初写明的触发条件是"**前端再加一个页面或路由时**，加一个最小的『路由表 ↔ 跳转
目标』一致性检查即可，不必上框架"。本文件就是那一条，触发条件已满足（计划审批页这两轮改过）。

检查的是**三处必须彼此认识**的东西：

- `index.html` 侧栏的 `data-route`（人点得到什么）；
- `app.js` 的 `ROUTES` 表（点了之后渲染什么）；
- `app.js` 里 `render` 指向的函数（渲染函数是不是真的存在）。

不做的事：不跑 JS、不查 DOM、不测交互。**能抓到"名字对不上"，抓不到"逻辑写错了"** —— 这是
有意划的界，别把它当成前端测试的替代品。
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "harness" / "server" / "static"
APP_JS = STATIC / "assets" / "app.js"
INDEX_HTML = STATIC / "index.html"

#: 路由表条目：``name: { title: "…", render: renderX … }``
_ENTRY = re.compile(r'(\w+):\s*\{\s*title:\s*"([^"]*)",\s*render:\s*(\w+)')
_ROUTES_BLOCK = re.compile(r"var ROUTES = \{(.*?)\n  \};", re.S)
_FUNCTION = re.compile(r"function\s+(\w+)\s*\(")
#: 代码里写死的跳转目标：``go("#/detail")`` / ``location.hash = "#/tasks"``。
#: 字符集必须含 `-` `_` `数字` —— 只用 ``[A-Za-z]+`` 会在 ``#/approvals-old`` 的连字符处
#: **截断**，把 ``approvals`` 当成合法路由，于是这条检查对写错的路由名视而不见
#: （这不是推演：第一版就是这么写的，靠变异检查才发现）。
_HASH_TARGET = re.compile(r"""["']#/([A-Za-z0-9_-]+)""")
_NAV_ROUTE = re.compile(r'data-route="([^"]+)"')


def _parse_routes() -> dict[str, str]:
    """``{路由名: 渲染函数名}``。"""
    block = _ROUTES_BLOCK.search(APP_JS.read_text(encoding="utf-8"))
    assert block, "app.js 里找不到 ROUTES 路由表（改过形状就要同步改本检查）"
    return {name: fn for name, _title, fn in _ENTRY.findall(block.group(1))}


def test_route_table_is_parseable() -> None:
    """先证明这个检查真的读到了东西 —— 正则匹配不到时不该静默全过。"""
    routes = _parse_routes()
    assert len(routes) >= 8, f"路由表只解析出 {routes}，多半是形状变了"


def test_every_nav_item_points_at_a_real_route() -> None:
    """侧栏点得到的每一项，路由表里都得有。"""
    routes = _parse_routes()
    nav = _NAV_ROUTE.findall(INDEX_HTML.read_text(encoding="utf-8"))
    assert nav, "index.html 里找不到侧栏 nav 项"

    unknown = [r for r in nav if r not in routes]
    assert not unknown, f"侧栏指向了路由表里不存在的路由：{unknown}（路由表有 {sorted(routes)}）"


def test_every_hardcoded_jump_target_exists() -> None:
    """代码里写死的 ``#/xxx`` 必须都在路由表里 —— 切片 3 那个 bug 就是这条。"""
    routes = _parse_routes()
    targets = set(_HASH_TARGET.findall(APP_JS.read_text(encoding="utf-8")))

    unknown = sorted(t for t in targets if t not in routes)
    assert not unknown, f"跳转到了不存在的路由：{unknown}"


def test_every_route_renderer_is_defined() -> None:
    """路由表指着的渲染函数得真的存在（拼错函数名时页面会白屏且不报错）。"""
    routes = _parse_routes()
    defined = set(_FUNCTION.findall(APP_JS.read_text(encoding="utf-8")))

    missing = sorted({fn for fn in routes.values() if fn not in defined})
    assert not missing, f"路由表引用了未定义的渲染函数：{missing}"


def test_contextual_routes_are_reachable_from_the_nav() -> None:
    """带 id 的页（运行详情/计划/指标/产物）也要能从侧栏进去 —— 否则只有 URL 直达，
    而侧栏点进去时会因为拿不到 id 而显示"请先选一个任务"，等于没入口。"""
    routes = _parse_routes()
    block = _ROUTES_BLOCK.search(APP_JS.read_text(encoding="utf-8")).group(1)
    contextual = set(re.findall(r"(\w+):\s*\{[^}]*contextual:\s*true", block))
    nav = set(_NAV_ROUTE.findall(INDEX_HTML.read_text(encoding="utf-8")))

    assert contextual, "没有解析到任何 contextual 路由 —— 本检查已失效，请同步改"
    missing = sorted(contextual - nav)
    assert not missing, f"这些页要求带 id，却没有侧栏入口：{missing}"
