"""tests._smoke_broker —— ToolBroker 核心能力冒烟（注册/调用/校验/异常/限流/描述）。

运行（项目根）：
    .venv\\Scripts\\python.exe -m tests._smoke_broker
"""

from __future__ import annotations

from harness.tool_broker import ToolBroker
from harness.models import ToolDef


def test_broker() -> None:
    b = ToolBroker()

    def handler(args, ctx):
        return True, f"结果: {args['x'] * 2}", {"doubled": args["x"] * 2}

    td = ToolDef(
        name="dbl",
        description="翻倍",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer", "description": "要翻倍的数"}},
            "required": ["x"],
        },
        rate_limit_per_min=100,
    )
    b.register(td, handler)
    ok, text, arts = b.invoke("dbl", {"x": 21}, {})
    assert ok and text == "结果: 42" and arts["doubled"] == 42
    print("1. 注册+调用 ok")

    ok, text, _ = b.invoke("nope", {}, {})
    assert not ok and "不存在" in text
    print("2. 工具不存在 ok")

    ok, text, _ = b.invoke("dbl", {}, {})
    assert not ok and "x" in text
    print("3. 缺必填参数 ok")

    def bad(args, ctx):
        raise RuntimeError("boom")

    b.register(
        ToolDef(name="bad", description="d",
                parameters={"type": "object", "properties": {}, "required": []}),
        bad,
    )
    ok, text, _ = b.invoke("bad", {}, {})
    assert not ok and "boom" in text
    print("4. 异常捕获 ok")

    b2 = ToolBroker()
    b2.register(
        ToolDef(name="fast", description="d",
                parameters={"type": "object", "properties": {}, "required": []},
                rate_limit_per_min=3),
        lambda a, c: (True, "ok", {}),
    )
    for i in range(3):
        ok, _, _ = b2.invoke("fast", {}, {})
        assert ok, f"第{i+1}次应该通过"
    ok, text, _ = b2.invoke("fast", {}, {})
    assert not ok and "限流" in text
    print("5. 限流 ok")

    desc = b.list_tool_descriptions()
    assert "dbl" in desc and "翻倍" in desc and "x" in desc
    print("6. 描述文本 ok")
    print("=== tool_broker 全部通过 ===")


def _main() -> None:
    test_broker()


if __name__ == "__main__":
    _main()
