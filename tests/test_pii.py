"""tests.test_pii —— 中文 PII 规则层与"真正接上"。

为何存在（计划 §0.2 新增能力；C4 已拍板 PII **自研中文规则层**，不用 Presidio）：

1. `PIIDetectionMiddleware` 在仓库里早已存在，但**从未被任何地方注册** ——
   装配点不传 middleware，等于 PII 脱敏在生产空转。本文件把"接上"钉死。
2. 现有规则只有 4 条裸正则，身份证**没有校验位验证**（普通 18 位数字会被误脱敏），
   `after_tool` 只脱敏文本、**不脱敏 artifacts**（而真实行数据在 artifacts 里），
   且不实现 `before_llm` —— 发给模型的提示词根本没脱敏。
3. 覆盖边界要明确：本框架控制的**每一处 LLM 出站**都要过规则层，包括
   规划器、质量门裁判与顶层汇总（不只有子 Agent 的推理节点）。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_pii.py -q
"""

from __future__ import annotations

import json

import pytest

from harness.middleware import PIIDetectionMiddleware, mask_text
from harness.models import MiddlewareConfig

# ---- 测试夹具（校验位已算好，改动前先验算）--------------------------------
VALID_ID = "110101199003074557"      # 18 位，GB 11643 校验位正确
PLAIN_18 = "110101199003074550"      # 18 位，校验位错误 → 不是身份证
VALID_CARD = "4111111111111111"      # 16 位，Luhn 正确
PLAIN_16 = "4111111111111112"        # 16 位，Luhn 错误
PHONE = "13800138000"
EMAIL = "zhang.san@example.com"


# ----------------------------------------------------------------------
# 规则层：单条规则
# ----------------------------------------------------------------------
def test_masks_valid_id_card() -> None:
    masked, counts = mask_text(f"身份证 {VALID_ID} 已登记")
    assert VALID_ID not in masked
    assert "[MASKED_ID_CARD]" in masked
    assert counts["id_card"] == 1


def test_keeps_eighteen_digits_with_bad_checksum() -> None:
    """纯 18 位数字不是身份证 —— 不能因为"像"就脱敏。"""
    masked, counts = mask_text(f"编号 {PLAIN_18}")
    assert PLAIN_18 in masked, f"校验位错误不应被当作身份证：{masked}"
    assert counts.get("id_card", 0) == 0


def test_checksum_validation_can_be_disabled() -> None:
    masked, counts = mask_text(f"编号 {PLAIN_18}", validate_checksum=False)
    assert PLAIN_18 not in masked
    assert counts["id_card"] == 1


def test_masks_phone() -> None:
    masked, counts = mask_text(f"联系电话 {PHONE}")
    assert PHONE not in masked
    assert "[MASKED_PHONE]" in masked
    assert counts["phone"] == 1


def test_phone_like_digits_inside_longer_number_not_masked() -> None:
    """20 位数字里的 11 位子串不是手机号。"""
    raw = "91" + PHONE + "0000" * 0 + "123"
    masked, counts = mask_text(raw)
    assert raw in masked, f"长数字串不应被切出手机号：{masked}"
    assert counts.get("phone", 0) == 0


def test_masks_luhn_valid_bank_card() -> None:
    masked, counts = mask_text(f"卡号 {VALID_CARD}")
    assert VALID_CARD not in masked
    assert "[MASKED_BANK_CARD]" in masked
    assert counts["bank_card"] == 1


def test_keeps_sixteen_digits_failing_luhn() -> None:
    masked, counts = mask_text(f"订单号 {PLAIN_16}")
    assert PLAIN_16 in masked, f"Luhn 不通过不应被当作卡号：{masked}"


def test_masks_email() -> None:
    masked, counts = mask_text(f"邮箱 {EMAIL}")
    assert EMAIL not in masked
    assert counts["email"] == 1


def test_masks_multiple_kinds_in_one_text() -> None:
    masked, counts = mask_text(f"{PHONE} / {VALID_ID} / {EMAIL}")
    for raw in (PHONE, VALID_ID, EMAIL):
        assert raw not in masked
    assert counts["phone"] == 1 and counts["id_card"] == 1 and counts["email"] == 1


def test_text_without_pii_is_untouched() -> None:
    text = "销售额环比增长 12.5%，共 1000 行数据。"
    masked, counts = mask_text(text)
    assert masked == text
    assert not counts


# ----------------------------------------------------------------------
# 中间件：artifacts、工具白名单、LLM 出站
# ----------------------------------------------------------------------
def _middleware(**pure_params) -> PIIDetectionMiddleware:
    from harness.config import PIISettings

    return PIIDetectionMiddleware(PIISettings(**pure_params) if pure_params else None)


def test_after_tool_masks_artifacts_recursively() -> None:
    mw = _middleware()
    result = (True, "查询完成", {"rows": [{"phone": PHONE, "name": "张三"}],
                                 "meta": {"contact": {"email": EMAIL}}})
    ok, text, artifacts = mw.after_tool(_ctx(), "sql_query", result)
    assert ok is True
    blob = json.dumps(artifacts, ensure_ascii=False)
    assert PHONE not in blob and EMAIL not in blob, "artifacts 里的真实数据必须脱敏"


def test_after_tool_keeps_text_masked() -> None:
    mw = _middleware()
    _, text, _ = mw.after_tool(_ctx(), "data_inspector", (True, f"发现 {PHONE}", {}))
    assert PHONE not in text


def test_sql_and_code_args_are_not_masked() -> None:
    """SQL/代码里的号码是**查询条件**，脱敏会把查询改坏、结果变样。"""
    mw = _middleware()
    _, args = mw.before_tool(_ctx(), "sql_query", {"sql": f"select * from t where phone='{PHONE}'"})
    assert PHONE in args["sql"], f"SQL 入参不得被脱敏：{args}"

    _, args = mw.before_tool(_ctx(), "code_executor", {"code": f"print('{PHONE}')"})
    assert PHONE in args["code"]


def test_other_tool_args_are_masked() -> None:
    mw = _middleware()
    _, args = mw.before_tool(_ctx(), "data_inspector", {"file_path": f"{PHONE}.csv"})
    assert PHONE not in args["file_path"]


def test_before_llm_masks_message_content() -> None:
    mw = _middleware()
    messages = [
        {"role": "system", "content": "你是数据分析师"},
        {"role": "user", "content": f"客户 {PHONE} 的身份证是 {VALID_ID}"},
    ]
    out = mw.before_llm(_ctx(), messages)
    blob = json.dumps(out, ensure_ascii=False)
    assert PHONE not in blob and VALID_ID not in blob, "发给模型的提示词必须脱敏"
    assert out[0]["content"] == "你是数据分析师", "system 不应被改动"


def test_disabled_middleware_passes_through() -> None:
    mw = _middleware(enabled=False)
    messages = [{"role": "user", "content": PHONE}]
    assert mw.before_llm(_ctx(), messages)[0]["content"] == PHONE


def _ctx():
    from harness.middleware import MiddlewareContext

    return MiddlewareContext(operation="test")


# ----------------------------------------------------------------------
# 端到端：接上装配后，任何一次 LLM 调用都看不到原文
# ----------------------------------------------------------------------
class _PIIProbeLLM:
    """记录收到的每一条提示词；规划/裁判返回合法结构以驱动完整链路。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def _record(self, messages) -> str:
        blob = "\n".join(str(m.get("content", "")) for m in messages)
        self.seen.append(blob)
        return messages[0]["content"] if messages else ""

    def chat_json(self, messages) -> dict:
        system = self._record(messages)
        if "质量门裁判" in system:
            return {"passed": True, "reason": "ok", "needs_human": False}
        return {
            "tasks": [{
                "title": "汇总", "description": "汇总体检结论",
                "assigned_to": "reporter", "depends_on": [],
                "acceptance_criteria": ["给出结论"], "expected_artifacts": [],
            }]
        }

    def chat(self, messages, temperature=None) -> str:
        system = self._record(messages)
        if "报告汇总者" in system:
            return "最终报告：已完成。"
        return json.dumps({"final_answer": "已完成"}, ensure_ascii=False)


def test_no_llm_call_ever_sees_raw_pii() -> None:
    """覆盖规划器 / 子 Agent 推理 / 质量门裁判 / 顶层汇总四处出站。"""
    import asyncio
    import time

    from harness.server.service import HarnessService

    llm = _PIIProbeLLM()
    service = HarnessService(llm=llm)

    async def scenario() -> None:
        thread_id = await service.create_task(f"联系人 {PHONE}，身份证 {VALID_ID}，请汇总分析")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = await service.get_status(thread_id)
            if state and state["status"] in ("finished", "failed"):
                break
            await asyncio.sleep(0.1)

    asyncio.run(scenario())

    assert llm.seen, "未捕获到任何 LLM 调用，测试无效"
    leaked = [blob for blob in llm.seen if PHONE in blob or VALID_ID in blob]
    assert not leaked, (
        f"有 {len(leaked)}/{len(llm.seen)} 次 LLM 调用收到了未脱敏的 PII；"
        f"首个片段：{leaked[0][:200] if leaked else ''}"
    )
