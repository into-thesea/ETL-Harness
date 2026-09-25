"""harness.context.manager —— 上下文管理器（Context Manager）。

长程任务里，单条提示不能随执行步数无限增长，否则会撑爆模型上下文窗口、推高成本、
稀释关键信息。本模块在 ReAct 内核的两个确定时机介入：

1. action 之后（settle_observation）：工具返回的大段结果不直接进对话历史，超过
   阈值就"沉淀"到 VFS（虚拟文件系统），对话里只保留一段摘要 + 一个文件引用
   （文件卡片）；模型需要原始细节时，可再用文件工具按路径读取（渐进式披露）。
2. think 之前（compact_history）：按上下文预算组装历史——始终保留任务描述（锚点）
   与最近若干轮原文，更早的中间过程折叠成一条"前期操作回顾"，把提示长度控制在
   预算内，同时不丢失关键脉络。

设计原则：
- 不依赖外部服务即可工作：VFS 缺省时退化为"硬截断"，LLM 摘要器缺省时退化为
  确定性规则摘要；Redis/Milvus/MinIO 只改变持久化/检索后端，不影响本模块运行。
- 与编排解耦：本模块不 import LangGraph，只处理 OpenAI 风格的消息 dict，
  由 nodes.py 在 think/action 节点调用，是否启用经依赖注入决定。
- 可单测、可快照：沉淀引用与预算可导出/恢复，供黑板与 Checkpointer 使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from harness.config import settings

logger = logging.getLogger(__name__)

# 一条 OpenAI 风格消息：{"role": "system|user|assistant|tool", "content": "..."}
Message = dict

# 消息类型约束：单个角色名，或角色名元组。None 表示不约束。
# 对应 langchain_core.trim_messages 的 start_on / end_on 参数。
MessageTypes = str | tuple[str, ...] | None

# 摘要器：把若干条旧消息压缩成一段"前期操作回顾"（生产中通常接 LLM）
Summarizer = Callable[[list[Message]], str]


@dataclass
class ContextBudget:
    """上下文预算与沉淀策略。

    所有默认值来自配置段 ``ContextSettings``（环境变量前缀 ``CONTEXT_``），
    构造时可逐项覆盖 —— 便于按不同模型的上下文窗口调整。
    """

    sink_threshold_chars: int = settings.context.sink_threshold_chars
    """工具结果超过该字符数即沉淀到 VFS（全文落盘，提示里只留摘要 + 文件卡片）。"""

    observation_head_chars: int = settings.context.observation_head_chars
    """沉淀后，在提示中保留的结果头部摘要字符数。"""

    observation_char_limit: int = settings.context.observation_char_limit
    """无 VFS（或沉淀失败）时，单条 observation 允许进入提示的最大字符数，超出硬截断。"""

    keep_recent_messages: int = settings.context.keep_recent_messages
    """压缩历史时，始终保留最近多少条消息的原文。"""

    max_history_chars: int = settings.context.max_history_chars
    """历史（不含 system 与工具说明）的字符软上限，超出则折叠更早的消息。"""

    summary_head_chars: int = settings.context.summary_head_chars
    """确定性降级摘要中，每条旧消息最多保留多少字符。"""


class ContextManager:
    """长程任务上下文管理器。

    使用方式：
        vfs = VirtualFileSystem()
        cm = ContextManager(vfs=vfs)                      # summarizer 可选（接 LLM）
        # action 后：
        observation = cm.settle_observation(tool, raw_text, session_id)
        # think 前：
        messages = [system] + cm.compact_history(history, long_term_context=ltm)
    """

    def __init__(
        self,
        vfs: Optional[object] = None,
        summarizer: Optional[Summarizer] = None,
        budget: Optional[ContextBudget] = None,
    ) -> None:
        self.vfs = vfs                       # 可选 VirtualFileSystem；None 时只截断不沉淀
        self.summarizer = summarizer        # 可选 LLM 摘要函数；None 时用确定性摘要
        self.budget = budget or ContextBudget()
        # session_id -> 已沉淀文件的引用（文件卡片），可进任务黑板/快照
        self._refs: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # 长度估算
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """粗略估算 token 数。

        优先用 tiktoken（安装了即较准）；否则启发式：东亚字符约 1 字 1 token，
        其余字符约 4 个 1 token。仅用于预算判断，不要求精确。
        """
        if not text:
            return 0
        try:
            import tiktoken  # 可选依赖，未安装则走启发式

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
            other = len(text) - cjk
            return cjk + (max(1, round(other / 4)) if other else 0)

    # ------------------------------------------------------------------
    # 时机一：工具大结果沉淀（action 节点之后调用）
    # ------------------------------------------------------------------
    def settle_observation(
        self,
        tool_name: str,
        observation: str,
        session_id: str = "default",
    ) -> str:
        """返回"真正应该进入对话历史"的 observation 文本。

        - 有 VFS 且结果超过沉淀阈值：全文写入 VFS，返回摘要 + 文件卡片；
        - 否则若超过单条上限：硬截断兜底；
        - 否则原样返回。
        """
        b = self.budget
        text = observation or ""

        # 1) 大结果沉淀到 VFS，提示里只留摘要 + 路径
        if self.vfs is not None and len(text) > b.sink_threshold_chars:
            try:
                summary = text[: b.observation_head_chars]
                if len(text) > b.observation_head_chars:
                    summary += "..."
                vfs_path, _ = self.vfs.sink_large_result(
                    tool_name, text, session_id=session_id, summary=summary
                )
                ref = {
                    "tool": tool_name,
                    "path": vfs_path,
                    "chars": len(text),
                    "session_id": session_id,
                }
                self._refs.setdefault(session_id, []).append(ref)
                logger.info(
                    "Settled large %s result (%d chars) -> %s",
                    tool_name, len(text), vfs_path,
                )
                return (
                    f"【工具 {tool_name} 返回结果较长（{len(text)} 字符），完整内容已沉淀到 "
                    f"{vfs_path}；需要原始细节时可用文件工具按该路径读取。】\n"
                    f"结果摘要：\n{summary}"
                )
            except Exception as e:  # noqa: BLE001 - 沉淀失败不得中断执行，降级为截断
                logger.warning("VFS sink failed, fallback to truncation: %s", e)

        # 2) 无 VFS 或沉淀失败：硬截断兜底
        if len(text) > b.observation_char_limit:
            return text[: b.observation_char_limit] + "\n…（结果过长，已截断）"
        return text

    # ------------------------------------------------------------------
    # 时机二：历史压缩（think 节点之前调用）
    # ------------------------------------------------------------------
    def compact_history(
        self,
        messages: list[Message],
        *,
        long_term_context: str = "",
        start_on: MessageTypes = ("user", "assistant"),
        end_on: MessageTypes = None,
    ) -> list[Message]:
        """按预算压缩对话历史，返回可直接拼到 system 之后的消息列表。

        策略：
        - 第一条 user（任务描述）作为锚点始终保留；
        - 最近 keep_recent_messages 条保留原文；
        - 更早的消息折叠为一条"前期操作回顾"（有 LLM 摘要器用 LLM，否则确定性摘要）；
        - 折叠后仍超预算，从最旧的非锚点消息开始丢弃（最后防线）；
        - long_term_context 作为一条 system 记忆插在最前（对应 state.long_term_context）。
        注意：system 消息不由本方法管理（由 nodes 单独前插），这里会忽略传入的 system。

        Args:
            start_on: 裁剪后**保留窗口允许以何角色开头**。默认 ``("user", "assistant")``
                —— 即**不允许以 ``role="tool"`` 开头**。
            end_on: 保留窗口允许以何角色结尾；默认 None（不约束）。

        关于 start_on 的默认值（**与 langchain_core.trim_messages 的默认 None 不同**）：

        本方法服务的是本项目的 ReAct 消息协议，而该协议有一个硬约束 ——
        **每条 ``role="tool"`` 必须紧跟其发起它的 ``assistant(tool_calls=…)``**，
        否则服务端会以 400 拒绝（OpenAI Function Calling 契约）。裁剪若把窗口起点
        落在 tool 消息上，就切断了这层配对，且**只在历史较长时发作**（短测试测不出）。

        langchain_core 同样承认这个约束（见其 ``trim_messages`` 文档：
        *"generally a ToolMessage can only appear after an AIMessage that involved
        a tool call"*），但它把责任交给调用方（自行传 ``start_on``）。
        这里采用同一机制，只是默认值取本协议的**安全值** —— 避免调用方忘记设置
        而引入只在长任务中才暴露的间歇性 400。需要时可显式传 ``None`` 关闭约束。
        """
        b = self.budget
        history = [m for m in (messages or []) if m.get("role") != "system"]
        if not history:
            return self._with_long_term([], long_term_context)

        # 锚点：第一条 user 始终保留
        anchor: list[Message] = []
        body = history
        if history[0].get("role") == "user":
            anchor = [history[0]]
            body = history[1:]

        # 预算内且消息不多：无需压缩（仍须保证不违反 start_on/end_on 约束）
        if (
            self._total_chars(anchor) + self._total_chars(body) <= b.max_history_chars
            and len(body) <= b.keep_recent_messages
        ):
            body = self._align_window(body, start_on, end_on)
            return self._with_long_term(anchor + body, long_term_context)

        recent = body[-b.keep_recent_messages:] if b.keep_recent_messages > 0 else []
        older = body[: -b.keep_recent_messages] if b.keep_recent_messages > 0 else list(body)

        # 窗口起点对齐：被挤出去的靠前消息并入 older 一起摘要（而非直接丢弃，避免丢信息）
        aligned = self._align_window(recent, start_on, end_on)
        dropped = len(recent) - len(aligned)
        if dropped > 0:
            older = older + recent[:dropped]
        recent = aligned

        packed: list[Message] = list(anchor)
        if older:
            review = self._summarize(older)
            packed.append({
                "role": "user",
                "content": f"【前期操作回顾（更早的中间过程已压缩，需要细节可调文件）】\n{review}",
            })
        packed.extend(recent)

        # 最后防线：仍超预算则丢弃最旧的非锚点消息
        if self._total_chars(packed) > b.max_history_chars:
            packed = self._enforce_budget(packed, start_on=start_on, end_on=end_on)
        return self._with_long_term(packed, long_term_context)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _summarize(self, older: list[Message]) -> str:
        """把旧消息压缩成回顾文本：优先 LLM 摘要器，失败/缺省走确定性规则。"""
        if self.summarizer is not None:
            try:
                return self.summarizer(older)
            except Exception as e:  # noqa: BLE001 - 摘要器故障必须可降级
                logger.warning("Summarizer failed, use deterministic summary: %s", e)

        role_tag = {"assistant": "决策", "user": "观察", "tool": "工具"}
        lines: list[str] = []
        for m in older:
            role = m.get("role", "user")
            content = (m.get("content") or "").replace("\n", " ").strip()
            if not content:
                continue
            clip = content[: self.budget.summary_head_chars]
            if len(content) > self.budget.summary_head_chars:
                clip += "…"
            lines.append(f"- {role_tag.get(role, role)}：{clip}")
        return "\n".join(lines) if lines else "（无）"

    def _enforce_budget(
        self,
        messages: list[Message],
        *,
        start_on: MessageTypes = None,
        end_on: MessageTypes = None,
    ) -> list[Message]:
        """最后防线：从最旧的非锚点消息开始丢弃，直到总字符不超过预算。

        丢弃后仍须做窗口边界对齐（``start_on``/``end_on``）—— 逐条丢弃会切断
        ``assistant(tool_calls)`` 与 ``role="tool"`` 的配对，必须事后修正。
        """
        anchor: list[Message] = []
        rest = messages
        if messages and messages[0].get("role") == "user":
            anchor = [messages[0]]
            rest = messages[1:]

        total = self._total_chars(anchor)
        kept: list[Message] = []
        for m in reversed(rest):  # 倒序优先保留较新的消息
            chars = len(m.get("content", ""))
            if total + chars > self.budget.max_history_chars:
                continue
            kept.append(m)
            total += chars
        window = self._align_window(list(reversed(kept)), start_on, end_on)
        return anchor + window

    # ------------------------------------------------------------------
    # 窗口边界对齐（对应 langchain_core.trim_messages 的 start_on / end_on）
    # ------------------------------------------------------------------
    @staticmethod
    def _matches(role: str, types: MessageTypes) -> bool:
        """角色是否落在允许的类型集合内；types 为 None 表示不约束。"""
        if types is None:
            return True
        if isinstance(types, str):
            return role == types
        return role in types

    @classmethod
    def _align_window(
        cls,
        window: list[Message],
        start_on: MessageTypes,
        end_on: MessageTypes,
    ) -> list[Message]:
        """把保留窗口的**起点/终点**对齐到允许的角色。

        起点不合法 → 从最旧方向剔除到第一个合法角色；终点不合法 → 从最新方向
        剔除到最后那个合法角色；全不匹配则返回空窗口。None 表示该项不约束。

        与 langchain_core.trim_messages 同一思路（它把序列反转后用 ``end_on``
        表达起点约束），用于防止裁剪把 Function Calling 的调用/结果配对切断。
        """
        if start_on is not None:
            for i, m in enumerate(window):
                if cls._matches(m.get("role", ""), start_on):
                    window = window[i:]
                    break
            else:
                return []

        if end_on is not None:
            for i in range(len(window) - 1, -1, -1):
                if cls._matches(window[i].get("role", ""), end_on):
                    window = window[: i + 1]
                    break
            else:
                return []

        return window

    @staticmethod
    def _total_chars(messages: list[Message]) -> int:
        return sum(len(m.get("content", "")) for m in messages)

    @staticmethod
    def _with_long_term(messages: list[Message], long_term_context: str) -> list[Message]:
        """把检索到的长期记忆作为一条 system 消息前插（无则原样返回）。"""
        if not (long_term_context or "").strip():
            return messages
        return [
            {"role": "system", "content": f"【相关长期记忆】\n{long_term_context.strip()}"}
        ] + messages

    # ------------------------------------------------------------------
    # 沉淀引用与快照（供任务黑板 / Checkpointer）
    # ------------------------------------------------------------------
    def settled_refs(self, session_id: str = "default") -> list[dict]:
        """返回某会话已沉淀到 VFS 的文件引用列表（文件卡片）。"""
        return list(self._refs.get(session_id, []))

    def snapshot(self) -> dict:
        """导出上下文状态快照（沉淀引用 + 预算），供断点恢复。"""
        return {
            "refs": {k: list(v) for k, v in self._refs.items()},
            "budget": dict(self.budget.__dict__),
        }

    def restore(self, snapshot: dict) -> None:
        """从快照恢复。"""
        self._refs = {k: list(v) for k, v in (snapshot.get("refs") or {}).items()}


__all__ = ["ContextManager", "ContextBudget", "Message", "MessageTypes", "Summarizer"]
