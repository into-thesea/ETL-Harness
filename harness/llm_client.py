"""harness.llm_client —— LLM 客户端（OpenAI 兼容接口）。

只依赖 ``openai>=1.0``，因此可以对接任何 OpenAI 兼容服务：
DeepSeek / 通义千问 / Kimi / Ollama / OpenAI 官方……

核心能力：
- ``chat``：普通对话，返回纯文本；
- ``chat_json``：要求模型输出 JSON，自动解析成 dict；
  JSON 解析失败时做一次"自修重试"——把模型上次的非法输出回灌给它，
  让它看到自己的错误并重新输出合法 JSON（Reflexion 思路的最小实现）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI


@dataclass
class ToolCall:
    """模型发起的一次函数调用（原生 Function Calling）。

    ``id`` 是服务端给的调用标识，回传结果时必须用 ``role="tool"`` +
    ``tool_call_id`` 与之配对，否则多轮调用会被服务端拒绝。
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolChatResult:
    """带工具的原生对话结果。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] = field(default_factory=dict)
    """可直接追加回历史的 assistant 消息（含 tool_calls），保持与 OpenAI 契约一致。"""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class JSONParseError(RuntimeError):
    """两次尝试都无法把模型输出解析成 JSON。"""


class LLMClient:
    """OpenAI 兼容 LLM 客户端。

    Args:
        api_key: API 密钥。
        base_url: 兼容端点，例如 ``https://api.deepseek.com``。
        model: 模型名，例如 ``deepseek-chat``。
        temperature: 默认采样温度。
        timeout: 单次请求超时（秒）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # 运行时配置
    # ------------------------------------------------------------------
    def set_model(self, model: str) -> None:
        """运行时切换模型，无需重建客户端。"""
        self.model = model

    # ------------------------------------------------------------------
    # 底层调用
    # ------------------------------------------------------------------
    def _raw_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        """底层调用。

        json_mode=True 时请求服务端的原生 JSON 输出（response_format），
        由服务端保证输出是合法 JSON，从根上减少解析失败。部分 OpenAI 兼容
        端点不支持该参数，因此失败时自动退回普通调用（由 chat_json 的
        容错解析与自修重试兜底）。
        """
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                **kwargs,
            )
        except Exception:
            if not json_mode:
                raise
            # 端点不支持 response_format：退回普通调用
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
            )
        content = resp.choices[0].message.content or ""
        return content.strip()

    # ------------------------------------------------------------------
    # 对外方法
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        """普通对话，返回模型输出的纯文本。"""
        return self._raw_chat(messages, self.temperature if temperature is None else temperature)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """要求模型输出 JSON 并解析为 dict。

        优先使用服务端原生 JSON 模式；解析失败时自动重试一次：把模型上次的
        非法回复作为 assistant 消息回灌，再追加一条 user 消息要求"重新输出
        合法 JSON"。第二次仍失败则抛出 :class:`JSONParseError`。
        """
        temp = self.temperature if temperature is None else temperature
        raw = self._raw_chat(messages, temp, json_mode=True)

        parsed = _try_extract_json(raw)
        if parsed is not None:
            return parsed

        # —— 第一次失败，自修重试 ——
        retry_messages = list(messages)
        retry_messages.append({"role": "assistant", "content": raw})
        retry_messages.append(
            {
                "role": "user",
                "content": "你上次的输出不是合法 JSON。请只输出一个完整且合法的 JSON 对象，"
                "不要使用 Markdown 代码块，不要输出任何解释文字。",
            }
        )
        raw2 = self._raw_chat(retry_messages, temp)
        parsed2 = _try_extract_json(raw2)
        if parsed2 is not None:
            return parsed2

        raise JSONParseError(
            f"模型输出两次都无法解析为 JSON。\n第一次输出: {raw[:300]}\n第二次输出: {raw2[:300]}"
        )

    # ------------------------------------------------------------------
    # 原生 Function Calling
    # ------------------------------------------------------------------
    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        tool_choice: str | dict = "auto",
        temperature: Optional[float] = None,
    ) -> ToolChatResult:
        """带工具清单的原生对话（OpenAI Function Calling）。

        工具清单经 API 的 ``tools`` 参数下发，``tool_calls`` 的结构由服务端
        保证 —— 这是相对 ReAct「拼文本 + 解析 JSON」的核心收益。

        Args:
            messages: 对话历史，可含 ``role="tool"`` + ``tool_call_id`` 的消息。
            tools: OpenAI tools 结构，由 ``broker.list_tools_openai_format()`` 生成。
            tool_choice: ``"auto"`` / ``"none"`` / ``"required"``，或指定某个函数。
            temperature: 覆盖默认温度。

        Returns:
            ToolChatResult。``tool_calls`` 为空表示模型直接给出了回答。
        """
        temp = self.temperature if temperature is None else temperature
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temp,
            tools=tools or None,
            tool_choice=tool_choice if tools else None,
        )
        message = resp.choices[0].message
        content = (message.content or "").strip()

        calls: list[ToolCall] = []
        raw_calls: list[dict[str, Any]] = []
        for tc in (getattr(message, "tool_calls", None) or []):
            raw_args = getattr(tc.function, "arguments", None) or "{}"
            try:
                parsed = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {"_value": parsed}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=parsed))
            # 保留原始 arguments 字符串，使 assistant 消息按原样回灌历史
            raw_calls.append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": raw_args},
            })

        raw_message: dict[str, Any] = {"role": "assistant", "content": content}
        if raw_calls:
            raw_message["tool_calls"] = raw_calls

        return ToolChatResult(content=content, tool_calls=calls, raw_message=raw_message)

    @staticmethod
    def tool_result_message(tool_call_id: str, content: str) -> dict[str, Any]:
        """构造工具结果消息（``role="tool"`` + 与调用配对的 ``tool_call_id``）。"""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ----------------------------------------------------------------------
# JSON 提取工具
# ----------------------------------------------------------------------
def _try_extract_json(text: str) -> Optional[dict[str, Any]]:
    """从模型输出中尽力提取一个 JSON 对象。

    容忍模型用 ```json ... ``` 包裹、或在 JSON 前后夹杂少量说明文字。
    提取不到时返回 None。
    """
    if not text:
        return None

    # 1) 直接尝试整段解析
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) 抠 ```json ... ``` 代码块
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass

    # 3) 抓第一个 { 到最后一个 } 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    return None


__all__ = ["LLMClient", "JSONParseError"]
