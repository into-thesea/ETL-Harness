"""harness.memory.working —— 工作记忆。

存储任务执行中的关键事实/中间结果，key-value 结构。
工作记忆只在当前任务生命周期内有效，任务结束后清除。

设计约定：
- 纯内存实现（不需要 Redis/Milvus），因为工作记忆是临时的。
- 支持 key-value 读写、批量操作、序列化/反序列化。
- 支持元数据（每个值可以附带类型/来源/时间戳）。
"""

from __future__ import annotations

import time
from typing import Any, Optional


class WorkingMemory:
    """工作记忆管理器。

    存储任务执行中的关键变量和中间结果。

    使用方式：
        wm = WorkingMemory()
        wm.set("order_id", "ORD-001", source="user_input")
        wm.set("total_amount", 1500.0, source="calculator")
        order_id = wm.get("order_id")
        all_data = wm.all()
    """

    def __init__(self):
        self._data: dict[str, dict] = {}  # key -> {value, metadata, timestamp}

    def set(self, key: str, value: Any, source: str = "", metadata: Optional[dict] = None) -> None:
        """设置一个工作记忆项。"""
        self._data[key] = {
            "value": value,
            "source": source,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取工作记忆项的值。"""
        item = self._data.get(key)
        return item["value"] if item else default

    def get_item(self, key: str) -> Optional[dict]:
        """获取完整的工作记忆项（含元数据）。"""
        return self._data.get(key)

    def has(self, key: str) -> bool:
        """检查 key 是否存在。"""
        return key in self._data

    def delete(self, key: str) -> bool:
        """删除工作记忆项。"""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def all(self) -> dict[str, Any]:
        """获取所有工作记忆的 key-value。"""
        return {k: v["value"] for k, v in self._data.items()}

    def all_with_metadata(self) -> dict[str, dict]:
        """获取所有工作记忆（含元数据）。"""
        return dict(self._data)

    def keys(self) -> list[str]:
        """获取所有 key。"""
        return list(self._data.keys())

    def values(self) -> list[Any]:
        """获取所有值。"""
        return [v["value"] for v in self._data.values()]

    def clear(self) -> None:
        """清空所有工作记忆。"""
        self._data.clear()

    def update(self, data: dict[str, Any], source: str = "batch") -> None:
        """批量更新工作记忆。"""
        for k, v in data.items():
            self.set(k, v, source=source)

    def build_context_text(self) -> str:
        """格式化为 LLM 上下文文本。"""
        if not self._data:
            return ""
        lines = ["【工作记忆】"]
        for key, item in self._data.items():
            value = item["value"]
            source = item.get("source", "")
            source_str = f"（来源：{source}）" if source else ""
            lines.append(f"- {key}: {value}{source_str}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """导出快照（用于 Checkpoint）。"""
        return {"data": dict(self._data)}

    def restore(self, snapshot: dict) -> None:
        """从快照恢复。"""
        self._data = snapshot.get("data", {})

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)


__all__ = ["WorkingMemory"]
