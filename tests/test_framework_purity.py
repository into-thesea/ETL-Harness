"""tests.test_framework_purity —— 框架侧不得认识领域名词。

验收标准（`docs/领域包与插件接口设计.md` §1）：**框架是通用的，数据分析只是挂在它上面的
一个领域包**。所以框架侧不该出现任何领域工具名 / 领域角色名 —— 出现了，就说明领域知识
漏进了框架。

两件事分开验：

1. **字符串字面量扫描（AST，跳过 docstring）** —— 注释与文档里的领域举例是允许的
   （那是说明材料），所以不能靠 grep 文本：grep 分不出注释与代码。用 AST 取所有
   非 docstring 的字符串常量。
2. **行为断言** —— 默认值本身：默认角色名、规划器有没有内置名录、注册表未知名回退到哪。
   这些比字面量扫描更准，也是"默认值跑偏"这类问题（如 Critic 那次的默认值不一致）
   唯一靠得住的防线。

当前只扫**领域工具名**；领域角色名随角色定义一起搬走（切片 4），届时在这里加一条。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent / "harness"

# 数据分析领域的工具名（首个挂载的领域包所提供）。框架侧一个都不该出现。
DOMAIN_TOOL_NAMES = {
    "data_inspector",
    "data_cleaner",
    "eda",
    "sql_query",
    "chart_generator",
    "code_executor",
    "skill_reference",
}


def _docstring_constants(tree: ast.AST) -> set[int]:
    """收集所有 docstring 常量的 id（模块 / 类 / 函数体的第一条字符串语句）。"""
    ids: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ids.add(id(body[0].value))
    return ids


def _code_string_literals(path: Path) -> list[tuple[int, str]]:
    """返回该文件里**非 docstring** 的字符串字面量 [(行号, 值)]。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _docstring_constants(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
        ):
            found.append((node.lineno, node.value))
    return found


# ======================================================================
# 1. 字面量扫描
# ======================================================================
class TestNoDomainNamesInFramework:
    def _violations(self, path: Path) -> list[str]:
        return [
            f"{path.relative_to(FRAMEWORK_DIR.parent)}:{lineno} -> {value!r}"
            for lineno, value in _code_string_literals(path)
            # 精确匹配：`sql_query` 命中，`sql_query_guard`（技能名）不算
            if value in DOMAIN_TOOL_NAMES
        ]

    def test_no_domain_tool_name_appears_in_code(self) -> None:
        """框架侧一律不得出现领域工具名（含角色定义 —— 它们已搬进领域包）。

        领域工具名该由**工具自己**声明（`sandbox_task` / `pii_skip` / `cacheable`），
        而不是框架按名点名 —— 框架按名点名就意味着它认识那个领域。
        """
        violations: list[str] = []
        for path in sorted(FRAMEWORK_DIR.rglob("*.py")):
            violations += self._violations(path)
        assert not violations, (
            "框架侧出现了领域工具名（该由工具自己声明，而不是框架按名点名）：\n  "
            + "\n  ".join(violations)
        )


# ======================================================================
# 1b. 依赖方向：框架不得 import 领域
# ======================================================================
DOMAIN_TOP_LEVEL = ("tools", "packages", "examples")

# 允许的例外：框架的**注释**里提到领域（说明材料），以及下面这条用例自身。
# 判据是 AST 里的 import 语句，所以注释与字符串不会误报。


def _framework_imports(path: Path) -> list[tuple[int, str]]:
    """返回该文件 import 的顶层模块名 [(行号, 模块名)]。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # 相对 import（level>0）不可能指到领域，跳过
            if node.level == 0 and node.module:
                found.append((node.lineno, node.module))
    return found


class TestImportDirection:
    def test_framework_never_imports_domain(self) -> None:
        """框架**绝不** import 领域模块。

        这条比"字符串里没有领域名"更本质：方向一旦反过来（框架 import 领域），
        "换一套工具就是另一个应用"就不成立了 —— 换掉领域就等于改框架。

        实测抓到过三处：`sandbox/executor.py` 取项目根、`server/service.py` 注册内置
        工具与取 workspace 路径，以及**服务层的离线 LLM 直接 import 领域示例模块**。
        """
        violations: list[str] = []
        for path in sorted(FRAMEWORK_DIR.rglob("*.py")):
            for lineno, module in _framework_imports(path):
                top = module.split(".")[0]
                if top in DOMAIN_TOP_LEVEL:
                    violations.append(
                        f"{path.relative_to(FRAMEWORK_DIR.parent)}:{lineno} -> import {module}"
                    )
        assert not violations, (
            "框架 import 了领域模块（方向反了：领域应依赖框架）：\n  " + "\n  ".join(violations)
        )

    def test_scanner_actually_sees_strings(self) -> None:
        """守扫描器本身：它得真的扫到东西，否则上面那条会永远"通过"。"""
        literals = _code_string_literals(FRAMEWORK_DIR / "models.py")
        assert any(value == "default" for _, value in literals)
        # docstring 必须被跳过
        assert not any("数据分析" in value for _, value in literals)


# ======================================================================
# 2. 默认值（行为断言）
# ======================================================================
class TestNeutralDefaults:
    def test_default_role_is_neutral(self) -> None:
        """框架的默认角色名是中性值，不是某个领域角色。"""
        from harness.models import AgentRun, SubAgentDef, ToolDef

        assert ToolDef(name="t", description="d", parameters={}).required_role == "default"
        assert (
            SubAgentDef(name="a", description="d", system_prompt="p").required_role == "default"
        )
        assert AgentRun(session_id="s", agent_id="a", goal="g").role == "default"

    def test_registry_does_not_fall_back_to_a_domain_role(self) -> None:
        """未知子 Agent 名不得回退到某个领域角色；空注册表要直接报错并说明原因。"""
        from harness.agents.registry import AgentRegistry

        empty = AgentRegistry(defs={})
        with pytest.raises(KeyError) as exc:
            empty.get("whoever")
        assert "领域包" in str(exc.value), "报错要指向真正的原因（没挂领域包）"

    def test_registry_falls_back_within_what_is_registered(self) -> None:
        """有注册表时按"声明的默认 → 注册序第一个"回退，且**回退要响亮**。"""
        from harness.agents.registry import AgentRegistry
        from harness.models import SubAgentDef

        def _def(name: str) -> SubAgentDef:
            return SubAgentDef(name=name, description=f"{name} 说明", system_prompt="p")

        registry = AgentRegistry(defs={"one": _def("one"), "two": _def("two")})
        assert registry.get("nope").name == "one"          # 注册序第一个
        registry.default_name = "two"
        assert registry.get("nope").name == "two"          # 声明优先

    def test_planner_has_no_builtin_role_catalog(self) -> None:
        """规划器不内置名录：没传角色就是没有（并响亮告警），不是悄悄用领域默认。"""
        from harness.planning.planner import TaskPlanner

        planner = TaskPlanner(llm=None)

        assert planner.roles == []
        assert planner.default_agent == ""
        assert "没有任何可派发的子 Agent" in planner._agent_hints()

    def test_planner_derives_default_agent_from_roles(self) -> None:
        from harness.planning.planner import TaskPlanner

        planner = TaskPlanner(llm=None, available_agents=["a", "b"])

        assert planner.default_agent == "a"
        assert TaskPlanner(llm=None, available_agents=["a"], default_agent="z").default_agent == "z"

    def test_pii_skip_is_declared_by_tools_not_by_config(self) -> None:
        """框架配置不再按名点名领域工具（默认空）；跳过与否由工具自己声明。"""
        from harness.config import PIISettings

        assert PIISettings().skip_tools == ""

    def test_sandbox_task_is_declared_by_tools(self) -> None:
        from harness.models import ToolDef

        assert ToolDef(name="t", description="d", parameters={}).sandbox_task == ""
