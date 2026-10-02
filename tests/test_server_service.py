"""tests.test_server_service —— 服务装配的离线 LLM 回退路径。

覆盖 ``HarnessService._select_llm`` 的**无 API Key 分支**：真实 LLM 优先，未配置 Key 时
退回已挂载领域包贡献的离线工厂（``contributes["offline_llm"]``）。

这条分支此前没有覆盖 —— 所有服务层用例都显式注入 ``llm=``，直接从 ``_llm_override`` 取，
不会走到 ``_select_llm``。于是"起服务但不配 Key"（控制台的无 Key 演示路径）会直接
装不起来。这个测试把该路径固定住。

运行（项目根）：
    .venv\\Scripts\\python.exe -m pytest tests/test_server_service.py -q
"""

from __future__ import annotations

from harness.config import settings
from harness.server.service import HarnessService


def test_assemble_without_api_key_uses_package_contributed_llm(monkeypatch) -> None:
    """无 API Key：装配照常完成，LLM 取领域包贡献的离线工厂。

    必须走完整的 ``assemble()``：``_select_llm`` 依赖装配过程中建好的领域包管理器，
    单独调它到不了这条分支 —— 而"起服务但没配 Key"走的正是完整装配。
    """
    monkeypatch.setattr(settings.llm, "api_key", "")

    svc = HarnessService(auto_assemble=False)  # 不注入 llm=，让装配真正去选
    assert svc.assemble() is not None

    assert svc.llm is not None, "无 Key 时应回退到领域包贡献的离线 LLM"
    # 框架侧不认识任何具体领域实现：拿到的一定是"被贡献出来"的工厂产物。
    assert type(svc.llm).__module__.split(".")[0] == "packages", (
        f"离线 LLM 应来自领域包，实际 {type(svc.llm).__module__}"
    )
