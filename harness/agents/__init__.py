"""harness.agents —— 子 Agent 注册表与委派支持。

**注册表本身是框架能力，角色定义属于领域**：框架不内置任何角色，角色由领域包经
``PackageContext.register_agents`` 挂载进来（见 :mod:`harness.domain`）。
"""

from harness.agents.registry import AgentRegistry

__all__ = ["AgentRegistry"]
